from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

from lib.config import config as cfg
from lib.layer_utils.anchor_target_layer import anchor_target_layer
from lib.layer_utils.proposal_layer import proposal_layer
from lib.layer_utils.proposal_target_layer import proposal_target_layer
from lib.layer_utils.proposal_top_layer import proposal_top_layer
from lib.layer_utils.snippets import generate_anchors_pre


class Network(object):
    def __init__(self, batch_size=1):
        self._feat_stride = [16, ]
        self._feat_compress = [1. / 16., ]
        self._batch_size = batch_size
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}


        self.ada={}
        self.depth = 2
        self.k = 10
        # self.inner_epoch=tf.placeholder(tf.int32, shape=[1,])
        self.memory_storex = tf.placeholder(tf.float32, shape=[None, 4096])
        self.memory_storey = tf.placeholder(tf.float32, shape=[None, self.depth])
        # self.memory_storey = tf.placeholder(tf.float32, shape=[None, 1])
        self.is_training = True
        self.lstm_out = 512
        self.batch_size = cfg.FLAGS.batch_size
        self.zeta_d = 0.1



    # Summaries #
    def _add_image_summary(self, image, boxes):
        # add back mean
        image += cfg.FLAGS2["pixel_means"]
        # bgr to rgb (opencv uses bgr)
        channels = tf.unstack(image, axis=-1)
        image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
        # dims for normalization
        width = tf.to_float(tf.shape(image)[2])
        height = tf.to_float(tf.shape(image)[1])
        # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[1] / height,
                          cols[0] / width,
                          cols[3] / height,
                          cols[2] / width], axis=1)
        # add batch dimension (assume batch_size==1)
        # assert image.get_shape()[0] == 1
        boxes = tf.expand_dims(boxes, dim=0)
        image = tf.image.draw_bounding_boxes(image, boxes)

        return tf.summary.image('ground_truth', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    # Custom Layers #
    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name):
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe, tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name == 'rpn_cls_prob_reshape':
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._feat_stride, self._anchors, self._anchors_dis, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([cfg.FLAGS.rpn_top_n, 5])
            rpn_scores.set_shape([cfg.FLAGS.rpn_top_n, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            # 通过tf.py_func(func, inp, Tout, stateful=True, name=None)可以将任意的python函数func转变为TensorFlow op。
            # func接收的输入必须是numpy array，可以接受多个输入参数；输出也是numpy array，也可以有多个输出。inp传入输入值，Tout指定输出的基本数据类型。
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                           self._feat_stride, self._anchors, self._anchors_dis, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 8])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be backpropagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name):
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name):
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, dis_inside_weights, dis_outside_weights, dis_target, gt_dis_label = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

            rois.set_shape([cfg.FLAGS.batch_size, 5])
            roi_scores.set_shape([cfg.FLAGS.batch_size])
            labels.set_shape([cfg.FLAGS.batch_size, 1])
            bbox_targets.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            dis_target.set_shape([cfg.FLAGS.batch_size, self._num_classes * 3])
            bbox_inside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 4])
            dis_inside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 3])
            dis_outside_weights.set_shape([cfg.FLAGS.batch_size, self._num_classes * 3])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['dis_targets'] = dis_target
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
            self._proposal_targets['dis_inside_weights'] = dis_inside_weights
            self._proposal_targets['dis_outside_weights'] = dis_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + 'default'):
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0, 0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[0, 1] / np.float32(self._feat_stride[0])))
            anchors, anchors_dis, anchor_length = tf.py_func(generate_anchors_pre,
                                                [height, width,
                                                 self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                [tf.float32, tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchors_dis.set_shape([None, 3])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchors_dis = anchors_dis
            self._anchor_length = anchor_length

    def build_network(self, sess, is_training=True):
        raise NotImplementedError

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box


    def out_put(self, pre_dis, cls_prob, dis_targets, labels, bbox_pred):
        index = tf.where(tf.arg_max(cls_prob, 1) > 0)
        pre_dis = tf.gather(pre_dis, index)
        pre_dis = tf.squeeze(pre_dis, axis=1)
        cls_prob = tf.gather(cls_prob, index)
        cls_prob = tf.squeeze(cls_prob, axis=1)
        dis_targets = tf.gather(dis_targets, index)
        dis_targets = tf.squeeze(dis_targets, axis=1)
        labels = tf.gather(labels, index)
        labels = tf.squeeze(labels, axis=1)
        bbox_pred = tf.gather(bbox_pred, index)
        bbox_pred = tf.squeeze(bbox_pred, axis=1)

        cls_onehot = tf.one_hot(indices=tf.arg_max(cls_prob, 1), depth=self._num_classes, axis=1)
        cls_onehot = tf.reshape(cls_onehot,[-1, self._num_classes, 1])
        pre_dis = tf.reshape(pre_dis, [-1, self._num_classes, self.depth])
        pre_dis = tf.reduce_sum(pre_dis*cls_onehot, axis=1)
        bbox_pred = tf.reshape(bbox_pred, [-1, self._num_classes, 4])
        bbox_pred = tf.reduce_sum(bbox_pred*cls_onehot, axis=1)


        lab_onehot = tf.one_hot(indices=labels, depth=self._num_classes, axis=1)
        lab_onehot = tf.reshape(lab_onehot, [-1, self._num_classes, 1])
        dis_targets = tf.reshape(dis_targets, [-1, self._num_classes, self.depth])
        dis_targets = tf.reduce_sum(dis_targets*lab_onehot, axis=1)


        self._predictions['dis_targets_output'] = dis_targets
        self._predictions['dis_output'] = pre_dis
        self._predictions['bbox_output'] = bbox_pred

        # norm_loss = tf.norm(tf.abs(pre_dis - dis_targets), axis=1)
        # loss_dis = tf.reduce_mean(norm_loss)

    def distance_loss(self, pre_dis, dis_targets, dis_inside_weights):
        diff_loss = tf.abs(pre_dis-dis_targets)
        diff_all = tf.reduce_sum(diff_loss*dis_inside_weights, axis=1)
        diff_foreward = tf.gather(diff_all, tf.where(diff_all > 0))
        dis_loss = tf.reduce_mean(diff_foreward)

        return 0.1 * dis_loss

    def norm(self, x, y):
        return np.sqrt(x**2 + y**2)


    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('loss_' + self._tag):
            # RPN, class loss rpn_cls_prob
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            # rpn_cls_score = tf.reshape(self._predictions['rpn_cls_prob'], [-1, 2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            # cls_score = self._predictions["cls_prob"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])

            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(cls_score, [-1, self._num_classes]), labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)



            # RCNN, dis loss
            pre_dis = self.ada["dis"]
            # 将loss改为每个特征向量的（D-D*）的平方和开根
            dis_targets = self._proposal_targets['dis_targets']
            dis_targets = tf.reshape(dis_targets, (-1, self._num_classes, 3))
            dis_targets_x = tf.reshape(dis_targets[:, :, 0], (-1, self._num_classes, 1))
            dis_targets_z = tf.reshape(dis_targets[:, :, 2], (-1, self._num_classes, 1))
            # dis_targets_t = tf.py_func(self.norm, [dis_targets_x, dis_targets_z], [dis_targets_z.dtype])
            dis_targets = tf.concat([dis_targets_x, dis_targets_z], axis=-1)
            dis_targets = tf.reshape(dis_targets, (-1, self._num_classes * self.depth))
            dis_inside_weights = self._proposal_targets['dis_inside_weights']
            dis_inside_weights = tf.reshape(dis_inside_weights, (-1, self._num_classes, 3))
            dis_inside_weights_x = tf.reshape(dis_inside_weights[:, :, 0],(-1, self._num_classes, 1))
            dis_inside_weights_x = tf.zeros_like(dis_inside_weights_x)
            dis_inside_weights_z = tf.reshape(dis_inside_weights[:, :, 2],(-1, self._num_classes, 1))
            dis_inside_weights = tf.concat([dis_inside_weights_x, dis_inside_weights_z], axis=-1)
            dis_inside_weights = tf.reshape(dis_inside_weights, (-1, self._num_classes * self.depth))

            # dis_inside_weight_t = tf.reshape(dis_inside_weights_z, (-1, self._num_classes))

            #zjk 优化代码中。。。
            self.out_put(pre_dis, self._predictions['cls_prob'], dis_targets, self._proposal_targets['labels'], self._predictions['bbox_pred'])

            loss_dis = self.distance_loss(pre_dis, dis_targets, dis_inside_weights)

            # norm_loss=tf.norm(tf.abs(pre_dis - dis_targets), axis=1)
            # loss_dis=tf.reduce_mean(norm_loss)
            self.ada["loss_dis"] = loss_dis

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            # loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + loss_dis
            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            # loss = cross_entropy + rpn_cross_entropy
            # loss = loss_dis
            self._losses['total_loss'] = loss
            self._event_summaries.update(self._losses)

        return loss

    def create_architecture(self, sess, mode, num_classes, tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 8])
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizer here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.FLAGS.weight_decay)
        if cfg.FLAGS.bias_decay:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # batch_norm_params = {
        #     'decay': 0.997,
        #     'epsilon': 0.001,
        #     'scale': None,
        #     'updates_collections': tf.GraphKeys.UPDATE_OPS,
        #     'is_training': True if mode == 'TRAIN' else False
        # }

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0),
                       normalizer_fn=slim.batch_norm
                       ):
            rois, cls_prob, bbox_pred, fc = self.build_network(sess, training)

        # 记录第一网络的输出
        self._predictions['fc'] = fc

        layers_to_output = {'rois': rois}
        layers_to_output.update(self._predictions)

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if mode == 'TEST':
            stds = np.tile(np.array(cfg.FLAGS2["bbox_normalize_stds"]), (self._num_classes))
            means = np.tile(np.array(cfg.FLAGS2["bbox_normalize_means"]), (self._num_classes))
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self._add_image_summary(self._image, self._gt_boxes))
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            for var in self._act_summaries:
                self._add_act_summary(var)
            for var in self._train_summaries:
                self._add_train_summary(var)

        self._summary_op = tf.summary.merge_all()
        if not testing:
            self._summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    # def test_image(self, sess, image, im_info):
    #     feed_dict = {self._image: image,
    #                  self._im_info: im_info}
    #     cls_score, cls_prob, bbox_pred, dis_pre, rois, fc = sess.run([self._predictions["cls_score"],
    #                                                      self._predictions['cls_prob'],
    #                                                      self._predictions['bbox_pred'],
    #                                                      self._predictions['dis_pred'],
    #                                                      self._predictions['rois'],
    #                                                      self._predictions['fc']],
    #                                                     feed_dict=feed_dict)
    #     return cls_score, cls_prob, bbox_pred, dis_pre, rois, fc

    def test_image(self, sess, image, im_info, memory_storex, memory_storey):
        feed_dict = {self._image: image,
                     self._im_info: im_info,
                     self.memory_storex: memory_storex,
                     self.memory_storey: memory_storey}
        cls_score, cls_prob, bbox_pred, dis_pre, rois, fc = sess.run([self._predictions["cls_score"],
                                                         self._predictions['cls_prob'],
                                                         self._predictions['bbox_pred'],
                                                         self.ada['dis'],
                                                         self._predictions['rois'],
                                                         self._predictions['fc']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, dis_pre, rois, fc
    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs, train_op, memory_storex, memory_storey):
        feed_dict = {self._image: blobs['data'],
                     self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self.memory_storex: memory_storex,
                     self.memory_storey: memory_storey}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_dis, loss, dis_output, labels, dis_targets, dis_targets_output, cls_prob, fc, dis_pre, dis_inside_weights, bbox_output, rois, bbox_pred, bbox_inside_weights, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self.ada['loss_dis'],
                                                                            self._losses['total_loss'],
                                                                            self._predictions['dis_output'],
                                                                            self._proposal_targets['labels'],
                                                                            self._proposal_targets['dis_targets'],
                                                                            self._predictions['dis_targets_output'],
                                                                            self._predictions['cls_prob'],
                                                                            self._predictions['fc'],
                                                                            self.ada['dis'],
                                                                            self._proposal_targets['dis_inside_weights'],
                                                                            self._predictions['bbox_output'],
                                                                            self._predictions['rois'],
                                                                            self._predictions['bbox_pred'],
                                                                            self._proposal_targets['bbox_inside_weights'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_dis, loss, dis_output, labels, dis_targets, dis_targets_output, cls_prob, fc, dis_pre, dis_inside_weights, bbox_output, rois, bbox_pred, bbox_inside_weights


    def val_step(self, sess, blobs, memory_storex, memory_storey):
        feed_dict = {self._image: blobs['data'],
                     self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self.memory_storex: memory_storex,
                     self.memory_storey: memory_storey}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_dis, loss, dis_output, labels, dis_targets, dis_targets_output, cls_prob, fc, dis_pre, dis_inside_weights, bbox_output, rois = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self.ada['loss_dis'],
                                                                            self._losses['total_loss'],
                                                                            self._predictions['dis_output'],
                                                                            self._proposal_targets['labels'],
                                                                            self._proposal_targets['dis_targets'],
                                                                            self._predictions['dis_targets_output'],
                                                                            self._predictions['cls_prob'],
                                                                            self._predictions['fc'],
                                                                            self.ada['dis'],
                                                                            self._proposal_targets['dis_inside_weights'],
                                                                            self._predictions['bbox_output'],
                                                                            self._predictions['rois'],
                                                                            ],
                                                                           feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_dis, loss, dis_output, labels, dis_targets, dis_targets_output, cls_prob, fc, dis_pre, dis_inside_weights, bbox_output, rois




    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss_dis, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                     self._losses['rpn_loss_box'],
                                                                                     self._losses['cross_entropy'],
                                                                                     self._losses['loss_box'],
                                                                                     self._losses['loss_dis'],
                                                                                     self._losses['total_loss'],
                                                                                     self._summary_op,
                                                                                     train_op],
                                                                                    feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        sess.run([train_op], feed_dict=feed_dict)
