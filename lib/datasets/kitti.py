from lib.config.config import FLAGS, KITTI_classes, RGB_list

import numpy as np

import os
import cv2


class kitti:
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.img_name_list = []
        for img_name in os.listdir(FLAGS.KITTI_Image_Training_PATH):
            self.img_name_list.append(img_name)
            # print(img_name, img_name.split('.'))
        self.ratio_x = FLAGS.input_image[0] / FLAGS.input_image_orl_size[0]
        self.ratio_y = FLAGS.input_image[1] / FLAGS.input_image_orl_size[1]
        # print(self.ratio_x, self.ratio_y)

    def __call__(self, data_name="train"):
        if (self.indexes[data_name] + 1) * self.batch_size > len(self.img_name_list):
            self.indexes[data_name] = 0

        im = cv2.resize(cv2.imread(os.path.join(FLAGS.KITTI_Image_Training_PATH, self.img_name_list[self.indexes[data_name]])), (500, 375))  # # im : (600, 1000, 3)
        f_line = open(
            os.path.join(FLAGS.KITTI_Image_Label_PATH, str(self.img_name_list[self.indexes[data_name]].split('.')[0]) + ".txt")).readlines()
        self.indexes[data_name] += 1
        (index, x_left, y_top, x_right, y_bottom, dis_x, dis_y, dis_z) = [], [], [], [], [], [], [], []
        for index_, f in enumerate(f_line):
            signal_obj = f.split(" ")
            index.append(KITTI_classes.index(signal_obj[0]))   # index type
            dis_x.append(float(signal_obj[11]))
            dis_y.append(float(signal_obj[12]))
            dis_z.append(float(signal_obj[13]))
            x_left.append(self.ratio_x*float(signal_obj[4]))
            y_top.append(self.ratio_y*float(signal_obj[5]))
            x_right.append(self.ratio_x*float(signal_obj[6]))
            y_bottom.append(self.ratio_y*float(signal_obj[7]))
        #     im = cv2.rectangle(im, (int(x_left[index_]), int(y_top[index_])),
        #                        (int(x_right[index_]), int(y_bottom[index_])), color=RGB_list[0],
        #                        thickness=5)
        # cv2.imshow("test", im)
        # cv2.waitKey()
        # im = np.reshape(im, (FLAGS.input_image[0], FLAGS.input_image[1], FLAGS.input_image[2]))
        im = np.reshape(im, (1, FLAGS.input_image[1], FLAGS.input_image[0], FLAGS.input_image[2]))
        return im, index,  x_left, y_top, x_right, y_bottom, dis_x, dis_y, dis_z

# k = read_kitti()
# while True:
#     im, index,  x_left, y_top, x_right, y_bottom, dis_x, dis_y, dis_z = k()
#     print(index, x_left, y_top, x_right, y_bottom, dis_x, dis_y, dis_z)
#     cv2.imshow('t', im)
#     cv2.waitKey()
