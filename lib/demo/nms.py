import numpy as np
import tensorflow as tf

def nms(cls_np, box, cls_label, cls, box_all_anchors):   # cls [1, 37, 62, 18]  box (1, 37, 62, 4, 9)  cls_label [1, 37, 62, 2, 9]
    cls_reshape = np.reshape(np.reshape(np.squeeze(cls_np), (37, 62, 9, 2)), (37*62*9, 2))
    box_reshape = np.reshape((np.transpose(np.reshape(np.squeeze(box), (37*62, 4, 9)), (0, 2, 1))), (37 * 62 * 9, 4))
    cls_label_reshape = np.reshape(np.reshape(np.transpose(np.squeeze(cls_label), (0, 1, 3, 2)), (37 * 62, 9, 2)), (37*62*9, 2))
    #print(cls_reshape.shape)  # [20646, 2]
    cls_tf = tf.reshape(tf.reshape(tf.squeeze(cls), (37, 62, 9, 2)), (37*62*9, 2))
    box_tf = tf.reshape(tf.transpose(tf.squeeze(box_all_anchors), (0, 1, 3, 2)), (37 * 62 * 9, 4))
    # for ii in range(37*62*9):
    #     if cls_label_reshape[ii, 0] == 1:
    #         print(cls_label_reshape[ii, 0])
    #         print("+++++++++++", cls_reshape[ii, :])
    #         fg_cls.append(cls_reshape[ii, :])
    #         fg_label.append(cls_label_reshape[ii, :])
    #     else:
    #         continue

    cls_array = cls_reshape[:, 0]

    cls_sort = cls_array[::-1][:512]   # 降序之后前512个最大的值
    cls_sort_index = np.argsort(-cls_array)

    cls_sorted = np.zeros((512, 2))
    cls_tf_sorted = []
    for ii in range(512):
        cls_sorted[ii, :] = cls_reshape[cls_sort_index[ii], :]
        cls_tf_sorted.append(cls_tf[cls_sort_index[ii], :])
    cls_tf_sorted = tf.reshape(cls_tf_sorted, (512, 2))
    print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP", cls_tf_sorted)
    fg_label = np.zeros((512, 2))
    for ii in range(512):
        fg_label[ii, :] = cls_label_reshape[cls_sort_index[ii], :]
        print(cls_label_reshape[cls_sort_index[ii], :])

    box_sort = np.zeros((2000, 4))
    box_tf_sorted = []
    for ii in range(2000):
        box_sort[ii, :] = box_reshape[cls_sort_index[ii], :]
        box_tf_sorted.append(box_tf[cls_sort_index[ii], :])
    box_tf_sorted = tf.reshape(box_tf_sorted, (2000, 4))
    print(box_tf_sorted)
    return cls_sorted, box_sort, fg_label, cls_tf_sorted, box_tf_sorted
