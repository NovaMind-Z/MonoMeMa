from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.config.config import ms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from lib.demo import show
import scipy.io as sio
import scipy.sparse
import xml.etree.ElementTree as ET

from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.utils.py_cpu_nms import py_cpu_nms

from slimdemo.adaptive import Adaptive_Network


classes = ('__background__',        # 这里开头字母全部小写
           'car', 'van', 'truck', 'boat',
           'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc',
           )

_data_path = 'E:\\pyproject\\datasets\\KITTI2012'

def draw_bounding_boxes(_im, bbox, color_box=(0, 0, 0), thick_bbox=3):
    '''
    cv2 draw boundding box
    '''
    # int_fun = lambda x_list: [int(x) for x in x_list]
    # bbox = int_fun(bbox)
    im_box = cv2.rectangle(_im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=color_box, thickness=thick_bbox)
    # cv2.imshow('gkd', _im)
    # cv2.imshow('img', _im)
    # im_box = cv2.rectangle(_im, (329, 182), (356, 217), color= color_box, thickness= thick_bbox)

    return im_box



def draw_bbox_with_depth(im, bbox, pd, gd):
    '''
    cv2 color:(B, G, R)
    '''
    # print(kiname, bbox, pd, gd)
    # bbox = [int(b) for b in bbox]
    pd = 10 * pd[:, 1]
    bbox = bbox.astype(int)
    for i in range(len(bbox)):
        bbox_i = bbox[i, :]
        im = draw_bounding_boxes(im, bbox_i, color_box=(230, 182, 65), thick_bbox=3)
        im = cv2.putText(im, '{:.1f}m'.format(pd[i]),
                     (bbox_i[0], bbox_i[1] - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (45, 179, 247), 2, 2)
    # im = cv2.putText(im, '{:.1f}m'.format(gd),
    #                  (bbox[0], bbox[3] + 24),
    #                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 222), 2, 2)
    return im

def _load_pascal_annotation(index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(_data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    if not False:
        # Exclude the samples labeled as difficult
        non_diff_objs = [
            obj for obj in objs if int(obj.find('difficult').text) == 0]
        # if len(non_diff_objs) != len(objs):
        #     print 'Removed {} difficult objects'.format(
        #         len(objs) - len(non_diff_objs))
        objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    dis = np.zeros((num_objs, 3), dtype=np.float32)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 10), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    name = []
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        name.append(obj.find('filename'))
        bbox = obj.find('bndbox')
        distance = obj.find('distance')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) -1   # 更改的地方  这里不能-1 因为kitti里最小就是0了 -1就不对了
        y1 = float(bbox.find('ymin').text) -1
        x2 = float(bbox.find('xmax').text) -1
        y2 = float(bbox.find('ymax').text) -1

        dis_x = float(distance.find('disx').text)
        dis_y = float(distance.find('disy').text)
        dis_z = float(distance.find('disz').text) / 10

        _class_to_ind = dict(list(zip(classes, list(range(10)))))
        cls = _class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        dis[ix, :] = [dis_x, dis_y, dis_z]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'img_name': name,
            'dis': dis,       # 加上距离
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    # if cfg.USE_GPU_NMS and not force_cpu:
    #  return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    # else:
    # return cpu_nms(dets, thresh)
    return py_cpu_nms(dets=dets, thresh=thresh)




def final_nms(scores, boxes, dis_pre):

    cls_index = np.argmax(scores, axis=1)
    front_index = np.where(cls_index > 0)[0]
    cls_index = cls_index[front_index]
    boxes = boxes[front_index]         #(none, 40)
    dis_pre = dis_pre[front_index]     #(none, 20)
    # dis_truth = dis_truth[front_index] #(none, 20)
    scores = scores[front_index]

    bbox = np.zeros((boxes.shape[0], 4))
    pd = np.zeros((boxes.shape[0], 2))
    for i in range(boxes.shape[0]):
        bbox[i, :] = boxes[i, cls_index[i] * 4:cls_index[i] * 4 + 4]
        pd[i, :] = dis_pre[i, cls_index[i] * 2:cls_index[i] * 2 + 2]
    scores = np.max(scores, axis=1)
    dets = np.hstack((bbox, scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, 0.7)
    bbox = bbox[keep]
    pd = pd[keep]
    return bbox, pd


def demo(sess, net, image_name, memory_storex, memory_storey, kitti_memory_0323, AN, sess2):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)
    im = cv2.resize(im, (1242, 375))
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, bbox_pred, _, rois, fc = im_detect(sess, net, im, memory_storex, memory_storey)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, bbox_pred.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    im_shape = im.shape[: 2]
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(rois, box_deltas)
    boxes = clip_boxes(pred_boxes, im_shape)

    # show.vis_detections(image_name, scores, boxes, dis_pre, fc, NMS_THRESH, CONF_THRESH)
    show.vis_detections(image_name, scores, boxes, fc, kitti_memory_0323, AN, sess2, NMS_THRESH, CONF_THRESH)


def count1():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)



if __name__ == '__main__':
    g1 = tf.Graph()  # 加载到Session 1的graph
    g2 = tf.Graph()  # 加载到Session 2的graph

    sess1 = tf.Session(graph=g1)  # Session1
    sess2 = tf.Session(graph=g2)  # Session2

    tfmodel = 'E:\\pyproject\\amden_new\\default\\kitti_2012_train\\default\\vgg16_faster_rcnn_iter_256_430000.ckpt'
    # tfmodel = 'E:\\pyproject\\amden_new\\default\\kitti_2012_train\\default\\vgg16_faster_rcnn_iter_140000_avden_60000.ckpt'

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True


    with sess1.as_default():
        with g1.as_default():
            # load network
            net = vgg16(batch_size=1)
            net.create_architecture(sess1, "TEST", 10,
                                    tag='default', anchor_scales=[8, 16, 32])
            saver = tf.train.Saver()
            saver.restore(sess1, tfmodel)
            count1()

    with sess2.as_default():  # 1
        with g2.as_default():
            AN = Adaptive_Network()
            AN.build_network()
            AN.loss()
            saver = tf.train.Saver()
            saver.restore(sess2,
                          "E:\\pyproject\\AMDEN\\data\\amden_ckpt\\lm100_z_new\\avden_z_264000.ckpt")  # kitti
            count1()

    print('Loaded network {:s}'.format(tfmodel))

    # ---------------------------------------------------------------------
    # kitti_memory_0323 = np.load("E:\\pyproject\\AMDEN\\data\\mat\\kitti_memory100_z.npy", allow_pickle=True)
    # kitti_memory_0323 = np.load("E:\\pyproject\\AMDEN\\data\\mat\\kitti_memory100_z_random_same.npy", allow_pickle=True)
    kitti_memory_0323 = np.load("E:\\pyproject\\AMDEN\\data\\mat\\memories\\kitti_memory_high.npy", allow_pickle=True)
    memory_storex = np.load('E:\\pyproject\\amden_new\\data\\mat\\memory_storex.npy', allow_pickle=True)
    memory_storey = np.load('E:\\pyproject\\amden_new\\data\\mat\\memory_storey.npy', allow_pickle=True)

    # cityscapes
    # citylist = []
    # for file in os.listdir(cfg.FLAGS.city_path):
    #     for name in os.listdir(os.path.join(cfg.FLAGS.city_path, file)):
    #         file_name = os.path.join(cfg.FLAGS.city_path, file, name)
    #         print("city name", file_name)
    #         citylist.append(file_name)
    #
    # for im_name in citylist:
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #     print('Demo for /{}'.format(im_name))
    #     demo(sess1, net, im_name, memory_storex, memory_storey, kitti_memory_0323, AN, sess2)
    # KITTI
    test_list = open('E:\\pyproject\\datasets\\KITTI2012\\ImageSets\\Main\\val.txt').readlines()
    kitti_list = [os.path.join(cfg.FLAGS.kitti_png, test_name[:-1] + ".png") for test_name in test_list]
    for i in range(len(kitti_list)):
        im_name = os.path.join(cfg.FLAGS.kitti_png, kitti_list[i])
        # if '000266' in im_name:
        index = test_list[i]
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for /{}'.format(im_name))
        # demo(sess, net, im_name, index, memory_storex, memory_storey)
        demo(sess1, net, im_name, memory_storex, memory_storey, kitti_memory_0323, AN, sess2)

    # # synthia
    # for name in os.listdir(cfg.FLAGS.synthia_path):
    #     im_name = os.path.join(cfg.FLAGS.synthia_path, name)
    #     demo(sess1, net, im_name, memory_storex, memory_storey, kitti_memory_0323, AN, sess2)

    # KITTI2015
    # for file in os.listdir(cfg.FLAGS.kitti2015_path):
    #     if (file == 'image_2') or (file == 'image_3'):
    #         for name in os.listdir(os.path.join(cfg.FLAGS.kitti2015_path, file)):
    #             if name.split('_')[-1] == '10.png':
    #                 im_name = os.path.join(cfg.FLAGS.kitti2015_path, file, name)
    #                 print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #                 print('Demo for /{}'.format(im_name))
    #                 demo(sess1, net, im_name, memory_storex, memory_storey, kitti_memory_0323, AN, sess2)

    # apollo
    for file in os.listdir(cfg.FLAGS.apollo_path):
        if (file == 'camera_5'):
            number = 0
            for name in os.listdir(os.path.join(cfg.FLAGS.apollo_path, file)):
                # if name.split('_')[-1] == '10.png':
                im_name = os.path.join(cfg.FLAGS.apollo_path, file, name)
                number += 1
                if number == 1000:
                    break
                if number % 2 == 1:
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print('Demo for /{}'.format(im_name))
                    demo(sess1, net, im_name, memory_storex, memory_storey, kitti_memory_0323, AN, sess2)
    sio.savemat("E:\\pyproject\\AMDEN\\data\\mat\\apollo_train01_new.mat", ms)



