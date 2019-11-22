from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.layer_utils.generate_anchors import generate_anchors


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'
    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]  # A = 9
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]  # 1872
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])
    # print("anchors", anchors.shape,  height*feat_stride)
    anchors_dis = np.zeros((K * A, 3))
    anchors_dis[:, 0] = ((anchors[:, 2] + anchors[:, 0])/2 - width * feat_stride/2)*0.01234   # x对应距离
    anchors_dis[:, 1] = (anchors[:, 3] + anchors[:, 1]) * 0.0008963  # y对应距离 0.13349
    anchors_dis[:, 2] = (375 - anchors[:, 3]) * 0.13349  # z  以米为单位
    anchors_dis = anchors_dis.astype(np.float32, copy=False)
    # print(anchors_dis)
    return anchors, anchors_dis, length   # anchors (16848, 4)  anchors_dis  (16848, 3)
