import numpy as np
import cv2
import matplotlib.pyplot as plt
from lib.config.config import FLAGS, RGB_list, CLASSES, ms
# from lib.demo.dis_model import reg
import lib.config.config as cfg
from novamind.ops.im_ops import draw_box
import os
from lib.utils.nms_wrapper import nms
import math

import xml.dom.minidom
from adaptive import Adaptive_Network

# read bounding boxes of Imagenet
# 打开xml文档
def get_single_kitti_xml(xml_file):
    dom = xml.dom.minidom.parse(xml_file)  # 'E:\\KITTI2012\\Annotations\\000007.xml'
    # 得到文档元素对象
    root = dom.documentElement
    name = ["xmin", "ymin", "xmax", "ymax", "disx", "disy", "disz"]

    objs = root.getElementsByTagName("object")
    ann = []
    for obj in objs:
        s_ann = []
        box = obj.getElementsByTagName("bndbox")
        dis = obj.getElementsByTagName("distance")
        cls = obj.getElementsByTagName("name")
        # print(box, dis, cls[0].childNodes[0].nodeValue)
        s_ann.append(cls[0].childNodes[0].nodeValue)
        for bb in name[:4]:
            # print(box[0].getElementsByTagName(bb)[0].firstChild.data)
            s_ann.append(box[0].getElementsByTagName(bb)[0].firstChild.data)
        for dd in name[4:]:
            # print(dis[0].getElementsByTagName(dd)[0].firstChild.data)
            s_ann.append(dis[0].getElementsByTagName(dd)[0].firstChild.data)
        # print(s_ann)
        ann.append(s_ann)
    return ann


def find_box_index(pre_box, image_name):  # pre_box: xmin, ymin, xmax, ymax
    # ann = get_single_kitti_xml(os.path.join(FLAGS.Ann, str(image_name.split('.')[0]) + ".xml"))
    # print(os.path.join(FLAGS.Ann, str(image_name.split('.')[0].split('\\')[5]) + ".xml"))
    ann = get_single_kitti_xml(os.path.join(FLAGS.Ann, str(image_name.split('.')[0].split('\\')[5]) + ".xml"))
    diff_list = []
    for signal_obj in ann: # signal_obj: [cls_name, xmin, ymin, xmax, ymax, disy disx disz]
        det_bbox = [float(b) for b in signal_obj[1:5]]
        get_bbox_diff = sum([abs(det_bbox[i] - pre_box[i]) for i in range(4)])
        diff_list.append(get_bbox_diff)
    index = diff_list.index(min(diff_list))
    f_out = ann[index]
    obj_type = f_out[0]  # 类名
    out_box = [int(float(b)) for b in f_out[1:5]]
    dis_x, dis_y, dis_z = float(f_out[5]), float(f_out[6]), float(f_out[7])
    return obj_type, out_box, [dis_x, dis_y, dis_z]


def get_depth(img, bbox):
    # disp_path = 'F:\\KITTI2015\\data_scene_flow\\training\\disp_noc_1'
    # disp_path = 'E:\\pyproject\\datasets\\Apollo\\stereo_train_1\\stereo_train_001\\disparity'
    # disp_name = os.path.join(disp_path, img.split('\\')[-1])
    # disp_name = disp_name.replace('jpg', 'png')

    disp_path = 'E:\\pyproject\\datasets\\CItySpaces\\disparity_trainvaltest\\disparity'
    img = img.split('\\')[-3:]
    disp_name = os.path.join(disp_path, img[0], img[1], img[2].replace('leftImg8bit', 'disparity'))

    disp = cv2.imread(disp_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
    disp[disp > 0] = (disp[disp > 0]) / 256
    disp[disp <= 0] = -1
    depth = (0.209313 * 2262.52) / disp
    depth = cv2.resize(depth, (1242, 375))
    depth = depth.T
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    center_x = (bbox[0] + bbox[2])/2
    center_y = (bbox[1] + bbox[3])/2
    bbox[0] = center_x - width/4
    bbox[1] = center_y - height/4
    bbox[2] = center_x + width/4
    bbox[3] = center_y + height/4
    depth_sum = 0
    count = 0
    depth_list = []
    for x in range(int(bbox[0]), int(bbox[2])):
        for y in range(int(bbox[1]), int(bbox[3])):
            depth_list.append(depth[x, y])
    for i in range(len(depth_list)):
        if depth_list[i] <= 80 and depth_list[i] >= 1:
            depth_sum += depth_list[i]
            count += 1
    if count > 0:
        return depth_sum / count
    else:
        return 80


def vis_detections(image_name, scores, boxes, fc,  kitti_memory_0323, AN, sess2, NMS_THRESH=0.1, thresh=0.1):
    im_real = cv2.imread(image_name)
    im_real = cv2.resize(im_real, (1242, 375))
    im_raw = cv2.imread(image_name)
    im_raw = cv2.resize(im_real, (1242, 375))
    height, weight = im_real.shape[0], im_real.shape[1]
    # print(weight, height)
    ann_ind = 0
    store_city = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        # if cls_ind in [4, 5, 6]:
        #     break
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        # cls_dis = dis[:, 3 * cls_ind:3 * (cls_ind + 1)]
        # cls_dis = dis[:, 2 * cls_ind:2 * (cls_ind + 1)]

        cls_scores = scores[:, cls_ind]
        # print(cls_scores)
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # print("dets", dets.shape)
        keep = nms(dets, NMS_THRESH)
        # print("keep", dets.shape)
        # print(keep)
        det = dets[keep, :]
        keep_fc = fc[keep, :]
        # print("keep_fc", keep_fc.shape)
        # print("det", det.shape, det[:, 4])
        """Draw detected bounding boxes."""
        inds = np.where(det[:, 4] >= thresh)[0]
        # print("inds", inds)
        if len(inds) == 0:
            # print("No Boundding Box")
            continue

        for i in inds:
            color_ind = (cls_ind + i) % 8
            # print("color_ind", color_ind, cls_ind, i)
            bbox = det[i, :4]
            score = det[i, 4]
            pre_box = (bbox[0], bbox[1], bbox[2], bbox[3]) #kitti
            # pre_box = (bbox[0] * 1.65, bbox[1] * 2.73, bbox[2] * 1.65, bbox[3] * 2.73) #city
            # pre_box = (bbox[0]*1.57, bbox[1]*2.88, bbox[2]*1.57, bbox[3]*2.88)# synthia
            # pre_box = (bbox[0]*2.52, bbox[1]*2.56, bbox[2]*2.52, bbox[3]*2.56)# apollo


            # -------------------------------------------------
            # CITY
            # if score > 0.2:
            #     store_city.append([keep_fc[i, :], pre_box, ddis])

            # KITTI
            # -------------------------------------------------
            if score > 0.1:
                # label
                obj_type, out_box, real_dis = find_box_index(list(pre_box), image_name)
            #     # store_city.append([keep_fc[i, :], pre_box, ddis, real_dis])
            #     store_city.append([keep_fc[i, :], pre_box, real_dis])

            # # synthia
            # # -------------------------------------------------
            # if score > 0.1:
            #     # label
            #     store_city.append([keep_fc[i, :], pre_box])



            # # # pred box and dis
            if score > 0.1:
                ann_ind += 1
                ann_ind %= 8
                AN.data_process(np.reshape(keep_fc[i, :], (1, -1)), kitti_memory_0323)
                ddis, neighbor_key, neighbor_box, neighbor_label = AN.run_test(sess2)

                # real_dis = get_depth(image_name, list(pre_box))
                # store_city.append([keep_fc[i, :], pre_box, ddis, real_dis])
                # print(ddis, real_dis)
                real_dis = real_dis[2]

                # im_real = cv2.imread(image_name)

                im_real = draw_box(im_real, pre_box[:4], color_box=RGB_list[int(color_ind % 8)], thick_bbox=2)
                # im_raw = draw_box(im_raw, pre_box[:4], color_box=RGB_list[int(color_ind % 8)], thick_bbox=2)
                # cv2.imwrite('F:/CVPR_code_slim/data/out/' + 'queary.jpg', im_real)
                im_real = cv2.putText(im_real, '{:.1f}m'.format(float(ddis)),
                                      (int(pre_box[0]), int(pre_box[1] - 8)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (45, 179, 247), 2, 2)

                im_real = cv2.putText(im_real, '{:.1f}m'.format(real_dis),
                                      (int(bbox[0]), int(bbox[3] + 24)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 222), 2, 2)

                # cv2.imwrite('F:/CVPR_code_slim/data/out/' + 'answear.jpg', im_real)

                # for i in range(AN.k):
                #     im_memory = cv2.imread(neighbor_key[i])
                #     im_memory = draw_box(im_memory, neighbor_box[i], color_box=RGB_list[int(color_ind % 8)], thick_bbox=2)
                #     im_memory = cv2.putText(im_memory, '{:.1f}m'.format(float(neighbor_label[i])),
                #                           (int(neighbor_box[i][0]), int(neighbor_box[i][1] - 8)),
                #                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (45, 179, 247), 2, 2)
                #     cv2.imwrite('F:/CVPR_code_slim/data/out/memory_{}.jpg'.format(i), im_memory)


                # im_real = draw_box(im_real, [weight - 31 * margin, 10 * margin * ann_ind, weight, 10 * margin * (ann_ind + 1)]
                #                    , color_box=RGB_list[ann_ind], thick_bbox=-1)
                # im_real = cv2.putText(im_real, '{:s} {:.1f} {:.1f} {:.1f}'.format('p_d', ddis[2], ddis[0], ddis[1]),
                #                       (int(weight - 31 * margin), int(10 * margin * ann_ind + 3 * margin)),
                #                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, 2)
                # im_real = cv2.putText(im_real, '{:s} {:.1f} {:.1f} {:.1f}'.format('r_d', real_dis[2], real_dis[0], real_dis[1]),
                #                       (int(weight - 31 * margin), int(10 * margin * ann_ind + 8 * margin)),
                #                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, 2)
                # im_real = cv2.putText(im_real, '{:s} {:.3f}'.format(CLASSES[cls_ind], score),
                #                       (int(pre_box[0]), int(pre_box[1] - margin)),
                #                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, 2)
    # im_real = cv2.resize(im_real, (2048, 1024))
    # cv2.imshow('KITTI', im_real)
    png_name = image_name.split("\\")[-1]
    # print("png_name", png_name)
    # ms[png_name]  = store_city
    cv2.imwrite(os.path.join('F:\\CVPR_code_slim\\data\\kitti_out\\pd_gd', png_name), im_real)
    # cv2.imwrite(os.path.join('F:\\CVPR_code_slim\\data\\kitti_out\\raw', png_name), im_raw)
    # cv2.imshow('KITTI', im_real)
    # cv2.waitKey()
