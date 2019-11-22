import tensorflow as tf
import numpy as np


def area(boxes):
  """Computes area of boxes.

  Args:
    boxes: Numpy array with shape [N, 4] holding N boxes

  Returns:
    a numpy array with shape [N*1] representing box areas
  """
  return (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])


def intersection(boxes1, boxes2):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes
    boxes2: a numpy array with shape [M, 4] holding M boxes

  Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
  """
  (y_min1, x_min1, y_max1, x_max1) = boxes1
  (y_min2, x_min2, y_max2, x_max2) = boxes2

  all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
  all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
  intersect_heights = np.maximum(
      np.zeros(all_pairs_max_ymin.shape),
      all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
  all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
  intersect_widths = np.maximum(
      np.zeros(all_pairs_max_xmin.shape),
      all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def IOU(boxes1, boxes2):
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    # print(area1, area2, intersect, "--------------------------------->>>>>>>>>>>>>>>>")
    union = np.expand_dims(area1, axis=1) + np.expand_dims(
        area2, axis=0) - intersect
    return intersect / union


def get_label(box, index,  x_left, y_top, x_right, y_bottom):
    """
    box  (1, 37, 62, 4, 9)   x, y, w, h
    return: cls_label : [1, 37, 62, 18]
    """
    cls_label = np.zeros((1, 37, 62, 2, 9))
    for ii in range(37):
        for jj in range(62):
            for kk in range(9):
                box1_x_left = box[0, ii, jj, 0, kk] - box[0, ii, jj, 2, kk] / 2
                box1_x_right = box[0, ii, jj, 0, kk] + box[0, ii, jj, 2, kk] / 2
                box1_y_top = box[0, ii, jj, 1, kk] - box[0, ii, jj, 3, kk] / 2
                box1_y_bottom = box[0, ii, jj, 1, kk] + box[0, ii, jj, 3, kk] / 2
                if box1_x_left > 0 and box1_x_right < 1000 and box1_y_top > 0 and box1_y_bottom < 600:
                    box1 = (box1_y_top, box1_x_left, box1_y_bottom, box1_x_right)
                    iou_list = []
                    for pp in range(len(index)):
                        box2 = (y_top[pp], x_left[pp], y_bottom[pp], x_right[pp])
                        iou_list.append(IOU(box1, box2))
                        print(iou_list)
                    if np.max(iou_list) > 0.5:
                        #print("---------------------------------------", np.max(iou_list))
                        cls_label[0, ii, jj, :, kk] = (1, 0)
                        #print(cls_label[0, ii, jj, :, kk])
                    else:
                        cls_label[0, ii, jj, :, kk] = (0, 1)
                else:
                    cls_label[0, ii, jj, :, kk] = (0, 1)  # fg:[1, 0]  bg:[0, 1]
    #print(cls_label)
    return cls_label  # np.reshape(cls_label, [1, 37, 62, 18])
