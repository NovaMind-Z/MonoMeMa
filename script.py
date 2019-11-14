import numpy as np
#
# kitti_train_npy = np.load("E:/pyproject/AMDEN/data/mat/kitti_train.npy",allow_pickle=True)
# _kitti_train = kitti_train_npy.tolist()

import cv2
import scipy.io as sio

# a = cv2.imread('E:\\pyproject\\datasets\\KITTI2012\\PNGImages\\000105.png')
# b = cv2.rectangle(a, (974, 97), (1226, 373), (230, 182, 65), 3)
# b = cv2.rectangle(b, (530, 176), (553, 194), (230, 182, 65), 3)
# cv2.imshow('bb', b)
# cv2.waitKey(0)
# print('a')
# city_val = sio.loadmat("E:/pyproject/AMDEN/data/mat/city_val.mat") # all val
#
# kitti_val = sio.loadmat("E:/pyproject/AMDEN/data/mat/kitti_val.mat")  # kitti all train data
# img = kitti_val['000112.png']
# print('a')


city_train_npy = np.load("E:/pyproject/AMDEN/data/mat/city_train.npy", allow_pickle=True)
city_train_npy_new = np.load("E:/pyproject/amden_new/data/mat/city_train_new.npy", allow_pickle=True)
city_val_npy = np.load("E:/pyproject/AMDEN/data/mat/city_val.npy", allow_pickle=True)
city_val_npy_new = np.load("E:/pyproject/amden_new/data/mat/city_val_new.npy", allow_pickle=True)
print('a')




def MAEs(pds, gds):
    mae = 0  # abs rel
    aqrel = 0
    rmse = 0
    logrmse = 0
    ard = 0
    count = 0
    di_count = 0
    log_count = 0
    the1, the2, the3 = 0, 0, 0
    for i, pd in enumerate(pds):
        if gds[i] < 1000:
            if gds[i] == 0:
                gds[i] += 10e-8
            mae += abs(pd - gds[i])
            rmse += (pd - gds[i]) ** 2

            if gds[i] <= 0.1:
                di_count += 1
            else:
                aqrel += abs(pd - gds[i]) ** 2 / gds[i]
                ard += abs(pd - gds[i]) / gds[i]
            if gds[i] < 1:
                log_count += 1
            else:
                logrmse += (np.log(pd) - np.log(gds[i])) ** 2
            # threshold = 1.25 1.25**2 1.25**3
            if pd / gds[i] < 1.25:
                the1 += 1
            if pd / gds[i] < 1.25 ** 2:
                the2 += 1
            if pd / gds[i] < 1.25 ** 3:
                the3 += 1
            count += 1
    print("count", ard, count)
    print("MAE:", mae / count)
    print("RMSE:", np.sqrt(rmse / count))
    print("Log RMSE:", np.sqrt(logrmse / (count - log_count)))
    print("ARD/AbsRel:", ard / (count - di_count))
    print("aqrel", aqrel / (count - di_count))
    print("Threshold", the1 / count, the2 / count, the3 / count)
    com_acc(pds, gds)


def com_acc(pre, rea):
    '''
    pre [[x, y, z], [x, y, z], ...]
    '''
    count = 0
    acc_list = [0 for _ in range(6)]
    for i, data in enumerate(pre):
        pd = data
        gd = rea[i]

        count += 1
        for i, zeta in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3]):
            if abs(gd - pd)/gd < zeta:
                acc_list[i] += 1
    acc = [a/count for a in acc_list]
    print("Acc:", acc)


pd = np.load('E:\\pyproject\\AMDEN\\data\\result_pre.npy')
gd = np.load('E:\\pyproject\\AMDEN\\data\\result_rea.npy')



pos_25 = np.where(gd <= 25)
MAEs(list(pd[pos_25]), list(gd[pos_25]))

pos_50 = np.where((gd <= 50) & (gd > 25))
MAEs(list(pd[pos_50]), list(gd[pos_50]))

pos_100 = np.where(gd <= 100 & (gd > 50))
MAEs(list(pd[pos_100]), list(gd[pos_100]))

print('a')