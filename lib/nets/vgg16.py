
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import copy
# slim库是tensorflow中的一个高层封装，它将原来很多tf中复杂的函数进一步封装，省去了很多重复的参数，以及平时不会考虑到的参数。

import lib.config.config as cfg
from lib.nets.network import Network

import tensorflow.contrib.rnn as rnn
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes


class LSTM():
    def __init__(self, Seg, hidden_size, batch_size):
        self.Seg = Seg
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.reuse = False

    def call_lstm(self, x, init_state):  # x: [batch_size, time_step, input_size]

        init_state = slim.fully_connected(init_state, self.lstm_out,
                                   weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                   trainable=True, activation_fn=None, scope='F')

        # data = tf.split(x, self.Seg, 1)
        data = x  # 这里需要副本吗
        # data=copy.deepcopy(x)
        lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size, reuse=self.reuse)
        outputs = list()
        (c_pre, h_pre) = lstm_cell.zero_state(tf.shape(init_state)[0], dtype=tf.float32)
        state = (rnn.LSTMStateTuple(c_pre, init_state))
        # state = init_state
        with tf.variable_scope('Attention_Model'):
            for timestep in range(self.Seg):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                inda = data[:,timestep,:]
                # inda = tf.expand_dims(inda, 0)
                # print("inda", inda)
                # help(lstm_cell)
                (cell_output, state) = lstm_cell(inda, state)   #zjk疑问： state没有得到传递啊
                # print("cell_output", cell_output)
                outputs.append(cell_output)
        self.reuse = True
        h_state = outputs[-1]
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm")
        return h_state



class vgg16(Network,LSTM):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)
        LSTM.__init__(self, self.k, self.lstm_out, self.batch_size)


    def arg_sort(self, raw, n, flags=False):
        '''
        @raw 一维列表
        @n 要返回n个最大值索引
        @flags 默认False求最小值 True返回索引最大值
        根据列表返回列表的前n个最大值的索引位置
        '''
        copy_raw = raw[::]
        copy_raw = [[index, node]for index, node in enumerate(copy_raw)]
        copy_raw.sort(key=lambda f:f[1], reverse=flags)
        return [num for num in copy_raw[:n]]


    def memory_search(self, queary, memory_storex, memory_cls, memory_box, cls, area):
        memory_area = (memory_box[:, 3] - memory_box[:, 1])*(memory_box[:, 2] - memory_box[:, 0])
        index_cls = tf.where(tf.equal(tf.squeeze(memory_cls, axis=1), tf.cast(cls, memory_cls.dtype)))
        memory_area_cls = tf.gather(memory_area, index_cls)
        if memory_area_cls.get_shape()[0] >= self.k:
            distance_list = tf.abs(area - memory_area_cls)
            sort_dis = tf.nn.top_k(-1 * tf.reshape(distance_list, (1, -1)), self.k)[1]
        else:
            distance_list = tf.norm((memory_storex - queary), axis=1)
            sort_dis = tf.nn.top_k(-1 * tf.reshape(distance_list, (1, -1)), self.k)[1]
        return sort_dis


    def KNN(self, queays):
        '''
        memory_store:  memory_size*[[4096 dim vector], [1 dim dis]]
        queay: queay list [None, 4096]
        TopK: top-k
        '''

        for ii in range(self.batch_size):
            queary = tf.gather(queays, ii)

            distance_list = tf.norm((self.memory_storex - queary), axis=1)

            # sort_dis = self.arg_sort(distance_list, self.k)  # 欧式距离最小的
            sort_dis = tf.nn.top_k(-1 * tf.reshape(distance_list, (1, -1)), self.k)[1]
            sort_dis = tf.squeeze(sort_dis, axis=0)
            # print("sort_dis", sort_dis)
            if ii == 0:
                memory_x = tf.expand_dims(tf.gather(self.memory_storex, sort_dis), 0)
                memory_y = tf.expand_dims(tf.gather(self.memory_storey, sort_dis), 0)

            else:
                memory_x = tf.concat([memory_x, tf.expand_dims(tf.gather(self.memory_storex, sort_dis), 0)], axis=0)
                memory_y = tf.concat([memory_y, tf.expand_dims(tf.gather(self.memory_storey, sort_dis), 0)], axis=0)

        return memory_x, memory_y

    def build_ada(self, fc):

        # find neighbor
        neighbor_vector, neighbor_label = self.KNN(fc)

        emb_label = slim.fully_connected(neighbor_label, self.lstm_out,
                                         weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                         trainable=self.is_training, activation_fn=None, scope='enbedded_label')

        # neighbor embedding concat embedded label
        concat_vector = tf.concat([neighbor_vector, emb_label], 2)

        memory_state = slim.fully_connected(concat_vector, self.lstm_out,
                                            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                            trainable=self.is_training, activation_fn=None, scope='concat_vector')

        out_state = self.call_lstm(memory_state, fc)

        # logits linear layers
        dis = slim.fully_connected(out_state, self.depth*self._num_classes,
                                   weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                   trainable=self.is_training, activation_fn=None, scope='out')

        self.ada["dis"] = dis



    def build_network(self, sess, is_training=True):
        with tf.variable_scope('vgg_16', 'vgg_16'):

            # select initializer
            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # Build head
            net = self.build_head(is_training)

            # Build rpn
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training, initializer)

            # Build proposals
            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

            # Build predictions
            cls_score, cls_prob, bbox_pred, fc = self.build_predictions(net, rois, is_training, initializer, initializer_bbox)

            self.build_ada(fc)

            self._predictions["rpn_cls_score"] = rpn_cls_score
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
            self._predictions["f"] = rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
            self._predictions["cls_score"] = cls_score
            self._predictions["cls_prob"] = cls_prob
            self._predictions["bbox_pred"] = bbox_pred  #偏差值， 表示ROI的位置，需要反映射回原图
            self._predictions["rois"] = rois[:, 1:5]
            self._predictions["fc"] = fc

            self._score_summaries.update(self._predictions)

            return rois, cls_prob, bbox_pred, fc

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            # if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
            #     self._variables_to_fix[v.name] = v
            #     continue
            # # exclude the first conv layer to swap RGB to BGR
            # if v.name == 'vgg_16/conv1/conv1_1/weights:0':
            #     self._variables_to_fix[v.name] = v
            #     continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        # print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16'):
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv,
                                              "vgg_16/fc7/weights": fc7_conv,
                                              "vgg_16/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                              self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                              self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))

    def build_head(self, is_training):

        # Main network
        # Layer  1
        net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # Layer 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # Layer 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # Layer 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')

        # Append network to summaries
        self._act_summaries.append(net)

        # Append network as head layer
        self._layers['head'] = net

        return net

    def build_rpn(self, net, is_training, initializer):

        # Build anchor component
        self._anchor_component()

        # Create RPN Layer
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
        # mean, variance = tf.nn.moments(rpn, [1, 2, 3])
        # rpn = tf.nn.batch_normalization(rpn, mean, variance, 0, 1, 0.0001)
        # print("rpn", rpn)
        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')

        # Change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):

        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")

            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            if cfg.FLAGS.test_mode == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.FLAGS.test_mode == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
        return rois

    def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):

        # Crop image ROIs
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        pool5_flat = slim.flatten(pool5, scope='flatten')

        # Fully connected layers
        # print("pool5_flat shape", pool5_flat)
        mean, variance = tf.nn.moments(pool5_flat, [0, 1])
        pool5_flat = tf.nn.batch_normalization(pool5_flat, mean, variance, 0, 1, 0.0001)
        fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
        if is_training:
            fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')

        fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
        # mean, variance = tf.nn.moments(fc7, [0, 1])
        # fc7 = tf.nn.batch_normalization(fc7, mean, variance, 0, 1, 0.0001)
        if is_training:
            fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')
            # mean, variance = tf.nn.moments(fc7, [0, 1])
            # fc7 = tf.nn.batch_normalization(fc7, mean, variance, 0, 1, 0.0001)
        # Scores and predictions
        cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training, activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope='bbox_pred')

        return cls_score, cls_prob, bbox_prediction, fc7
