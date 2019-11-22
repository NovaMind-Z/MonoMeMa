from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from lib.utils.cython_bbox import bbox_overlaps

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    gt_boxes.shape:[:, 8]  标准的图片的label  比如大小是[4, 8]说明只有4个box
    rpn_rois (2000, 8)   是所有假设的box区域和dis 后面还要做调整
    """
    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    all_scores = rpn_scores

    # Include ground-truth boxes in the set of candidate rois
    if cfg.FLAGS.proposal_use_gt:
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        # not sure if it a wise appending, but anyway i am not using it
        all_scores = np.vstack((all_scores, zeros))

    num_images = 1
    rois_per_image = cfg.FLAGS.batch_size / num_images
    fg_rois_per_image = np.round(cfg.FLAGS.proposal_fg_fraction * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights, dis_inside_weights, dis_target = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)
    # print("out", dis_target.shape, bbox_targets.shape)
    rois = rois.reshape(-1, 8)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    dis_inside_weights = dis_inside_weights.reshape(-1, _num_classes * 3)
    dis_outside_weights = np.array(dis_inside_weights > 0).astype(np.float32)
    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, dis_inside_weights, dis_outside_weights, dis_target


def _get_bbox_regression_labels(bbox_target_data, gt_dis_label, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    dis_targets = np.zeros((clss.size, 3 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    dis_inside_weights = np.zeros(dis_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        start_d = int(3 * cls)
        end_d = start_d + 3
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        dis_targets[ind, start_d:end_d] = gt_dis_label[ind, :]
        bbox_inside_weights[ind, start:end] = cfg.FLAGS2["bbox_inside_weights"]
        dis_inside_weights[ind, start_d:end_d] = cfg.FLAGS2["dis_inside_weights"]
    return bbox_targets, dis_targets, bbox_inside_weights, dis_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    # print(ex_rois, gt_rois)
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.FLAGS.bbox_normalize_targets_precomputed:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.FLAGS2["bbox_normalize_means"]))
                   / np.array(cfg.FLAGS2["bbox_normalize_stds"]))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.  gt_boxes:[6, 8],     all_rois[2000, 8]
    """
    print("------------------------------------------------------------------------------------------------------------\r\n")
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),   # all_rois[1:5]是box  [5:] 是dis
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))    # gt_boxes[:4]是box   [5:] 是dis
    # print(overlaps.shape) (2000, 3)
    gt_assignment = overlaps.argmax(axis=1)   # 1表示行最大值地方  gt_assignment.shape = 2000
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    print(max_overlaps, "\r\n", gt_boxes[gt_assignment, 5:])
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.FLAGS.roi_fg_threshold)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.FLAGS.roi_bg_threshold_high) &
                       (max_overlaps >= cfg.FLAGS.roi_bg_threshold_low))[0]

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    if fg_inds.size > 0 and bg_inds.size > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    else:
        # fg_inds = []
        # bg_inds = cfg.bg_inds
        # # print("fg_rois_per_image", type(fg_rois_per_image), fg_rois_per_image)
        # fg_rois_per_image = 0
        # pass
        # # print(fg_inds.size, bg_inds.size)
        import pdb
        pdb.set_trace()  # pdb调试程序,程序停止。

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    # print("check", type(labels), labels, "\r\n python", keep_inds)
    labels = labels[np.array(keep_inds, dtype=np.int)]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds]   # (256, 8)
    roi_scores = all_scores[keep_inds]

    # bbox_target_data.shape (256, 5)
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    gt_dis_label = gt_boxes[gt_assignment[keep_inds], 5:]   # (256, 3)这个是经过处理后的所有dis 标准label

    # print(gt_dis_label)
    # bbox_targets.shape (256, 40)  dis_targets (256, 30)
    bbox_targets, dis_targets, bbox_inside_weights, dis_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, gt_dis_label, num_classes)
    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, dis_inside_weights, dis_targets

