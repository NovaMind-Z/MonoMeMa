# coding=utf-8
import cv2
import os
from DeepLearning.python import int_fun
for f in os.listdir('E:\\KITTI2012\\OneShot'):
    im = cv2.imread('E:\\KITTI2012\\OneShot\\' + f)
    im = cv2.resize(im, (900, 800))
    cv2.imwrite('E:\\KITTI2012\\OneShot\\' + f, im)