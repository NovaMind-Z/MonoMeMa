import numpy as np


def bbox_overlaps(boxes, query_boxes):
    # 输入：
    # boxes: [len_inds_inside, 4]  表示所有的anchor len_inds_inside是anchor的个数 4表示bbox的四个值
    # query_boxes:shape: [K, 5]  K表示一张图上有几个物体 5：bbox +类别
    # 输出：overlaps shape: [len_inds_inside, K] ,每一列表示len_inds_inside个anchor和目标Box的重叠度，K列为每一个目标都算一次
    # 这个函数为计算所有的anchor和标准的box的重叠度IOU
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K))

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps
