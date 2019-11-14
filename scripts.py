import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import math
im_name = 'E:\\pyproject\\datasets\\Apollo\\stereo_train_1\\stereo_train_001\\disparity\\171206_034630104_Camera_5.png'
# im_name = 'E:\\pyproject\\datasets\\CItySpaces\\disparity_trainvaltest\\disparity\\test\\berlin\\berlin_000000_000019_disparity.png'
disp = cv2.imread(im_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
# disp = cv2.resize(disp, (1242, 375))
disp[disp > 0] = (disp[disp > 0]) / 256
disp[disp == 0] = -1
depth = (0.36 * 2301.31) / disp
depth = cv2.resize(depth, (1242, 375))

# depth = 0.54 * 721/disp
depth = np.resize(depth, (375, 1242))
cv2.imshow('a', depth)
print('a')
