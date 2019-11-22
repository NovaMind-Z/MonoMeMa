from __future__ import division

import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib
import random
import math
import copy
import cv2
import os

from novamind.ops.text_ops import list_save, text_read
# distance measure
def cosine_distances(X, Y):
    """
    Cosine distance is defined as 1.0 minus the cosine similarity.
    """
    X_norm = tf.sqrt(tf.reduce_sum(tf.square(X), axis=1))
    Y_norm = tf.sqrt(tf.reduce_sum(tf.square(Y), axis=1))
    XY_norm = tf.multiply(X_norm, tf.expand_dims(Y_norm, 1))
    XY = tf.multiply(X, Y[:,None])
    XY = tf.reduce_sum(XY, 2)
    similarity = XY / XY_norm
    distance = 1 - similarity
    return distance

def euclidean_distances(X, Y):
    distance = tf.norm(tf.subtract(X, tf.expand_dims(Y, 1)), axis=2)
    return distance

def manhattan_distances(X, Y):
    distance = tf.reduce_sum(tf.abs(tf.subtract(X, tf.expand_dims(Y, 1))), axis=2)
    return distance
## kitti
# kitti_train = sio.loadmat("E:/pyproject/AMDEN/data/mat/kitti_train.mat")  # kitti all train data
# kitti_train = sio.loadmat("E:\\pyproject\\amden_new\\data\\mat\\kitti_train_new.mat")  # kitti all train data
# kitti_test = sio.loadmat("E:/pyproject/AMDEN/data/mat/kitti_test.mat") # kitti test
# kitti_test = sio.loadmat("E:\\pyproject\\amden_new\\data\\mat\\kitti_test_new.mat") # kitti test
# kitti_val = sio.loadmat("E:/pyproject/AMDEN/data/mat/kitti_val.mat") # kitti val
# kitti_memory_0323 = np.load("E:/pyproject/AMDEN/data/mat/kitti_memory_0323.npy",allow_pickle=True)
kitti_memory_0323 = np.load("../../data/mat/kitti_memory100_z.npy", allow_pickle=True)
kitti_train_npy = np.load("../../data/mat/kitti_train_new.npy", allow_pickle=True)
kitti_test_npy = np.load("../../data/mat/kitti_test_new.npy", allow_pickle=True)

## city
# city_train_npy = np.load("E:/pyproject/AMDEN/data/mat/city_train.npy", allow_pickle=True)
city_train_npy = np.load("E:/pyproject/amden_new/data/mat/city_train_new.npy", allow_pickle=True)
# city_val_npy = np.load("E:/pyproject/AMDEN/data/mat/city_val.npy", allow_pickle=True)
city_val_npy = np.load("E:/pyproject/amden_new/data/mat/city_val_new.npy", allow_pickle=True)
city_memory = np.load("../../data/mat/city_memory.npy", allow_pickle=True)

# city_train = sio.loadmat("E:/pyproject/AMDEN/data/mat/city_train.mat") # all train
# city_train = sio.loadmat("E:/pyproject/amden_new/data/mat/city_train_new.mat") # all train
# city_val = sio.loadmat("E:/pyproject/AMDEN/data/mat/city_val.mat") # all val
# city_val = sio.loadmat("E:/pyproject/amden_new/data/mat/city_val_new.mat") # all val


class LSTM:
    def __init__(self, Seg, hidden_size, batch_size):
        self.Seg = Seg
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.reuse = False

    def call_lstm(self, x, init_state):  # x: [batch_size, time_step, input_size]

        init_state = slim.fully_connected(init_state, self.lstm_out,
                                   weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                   trainable=True, activation_fn=None, scope='F')

        data = tf.split(x, self.Seg, 1)
        lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_size, reuse=self.reuse)
        outputs = list()
        (c_pre, h_pre) = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # new_state = (rnn.LSTMStateTuple(c_pre, init_state))
        state = (rnn.LSTMStateTuple(c_pre, init_state))
        # state = init_state
        with tf.variable_scope('Attention_Model'):
            for timestep in range(self.Seg):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                inda = tf.squeeze(data[timestep])
                inda = tf.expand_dims(inda, 0)
                # print("inda", inda)
                # help(lstm_cell)
                (cell_output, state) = lstm_cell(inda, state)   #zjk疑问： state没有得到传递啊
                # print("cell_output", cell_output)
                outputs.append(cell_output)
        self.reuse = True
        h_state = outputs[-1]
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm")
        return h_state

class CITY(object):
    def __init__(self):
        '''
        height:1024
        wight:2048
        '''
        self.city_path = 'E:\\pyproject\\datasets\\CItySpaces\\leftImg8bit_trainvaltest\\leftImg8bit'
        self.dis_path ="E:\\pyproject\\datasets\\CItySpaces\\disparity_trainvaltest\\disparity"  # depth label

        self.fx = 2262.52
        self.fy = 2265.3
        self.cx = 1096.98
        self.cy = 513.137
        self.h = 2

    def load_depth_info(self, dis_path_name):
        img_d = cv2.imread(dis_path_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img_d[img_d > 0] = (img_d[img_d > 0] - 1) / 256
        return img_d

    def load(self):
        for key in memory_space_city.keys():
            figure_path_name = os.path.join(self.city_path, key)
            im = cv2.imread(figure_path_name)
            data = memory_space_city[key]
            depth_ino = self.load_depth_info(key[:-15])
            if len(data) != 0:
                for figure in data:
                    bbox = figure[1][0] # (xmin, ymin, xmax, ymax)

                    xc, yc = int((bbox[0]+bbox[2])/2), int(bbox[3])
                    dx,dy,dz = figure[2][0]

                    p_depth = np.sqrt(dx**2+dy**2+dz**2)

                    # self.com_linear_depth(xc, yc)

                    disp = depth_ino[yc, xc]
                    depth = (0.209313*2262.52) / disp

                    im = self.draw_bounding_boxes(im, bbox)
                cv2.imshow("city", im)
                cv2.imshow("cityd", depth_ino)
                cv2.waitKey()

    def split_y_z(self, x, y, depth_ino):
        '''
        city png:2048*1024
        input:x y　坐标点
        '''
        disp = depth_ino[y, x]
        depth = (0.209313*2262.52) / disp

        ## 前方距离
        disp_z = depth_ino[x, y]

    def load_sig_depth(self, name, bbox):
        '''
        bbox: xmin, ymin xmax, ymax
        '''
        # print("name", name)
        depth_ino = self.load_depth_info(name)

        x_center_bottom, y_center_bottom = int((bbox[0]+bbox[2])/2), int(bbox[3])
        x_left_bottom, y_left_bottom = int(bbox[0]), int(bbox[3])
        x_right_bottom, y_right_bottom = int(bbox[2]), int(bbox[3])
        x_center, y_center = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)

        dep_list = []
        # print("bbox", bbox)
        # im = cv2.imread(name)
        # im = self.draw_bounding_boxes(im, bbox)
        # cv2.imshow("im", im)
        # cv2.waitKey()
        if (bbox[3] - bbox[1]) < 20:
            mean_dep = 0
        else:
            for yi in range(int((bbox[1]+bbox[3])/2), int(bbox[3]), int((bbox[3] - bbox[1])/20)):
                disps = depth_ino[yi, int((bbox[0]+bbox[2])/2)]
                if disps == 0:
                    return 0
                depths = (0.209313*2262.52) / disps
                dep_list.append(depths)

            disp1 = depth_ino[y_center_bottom, x_center_bottom]
            disp2 = depth_ino[y_left_bottom, x_left_bottom]
            disp3 = depth_ino[y_right_bottom, x_right_bottom]
            disp4 = depth_ino[y_center, x_center]

            if (disp1 == 0)|(disp2 == 0)|(disp3 == 0)|(disp4 == 0):
                return 0

            depth1 = (0.209313*2262.52) / disp1
            depth2 = (0.209313*2262.52) / disp2
            depth3 = (0.209313*2262.52) / disp3
            depth4 = (0.209313*2262.52) / disp4

            dep_list.extend([depth1, depth2, depth3, depth4])

            deps = [de for de in dep_list if not math.isinf(de)]
            # print("dep list", len(deps), deps)
            if len(deps) == 0:
                mean_dep = 0
            else:
                mean_dep = sum(deps) / len(deps)

        # print("depth center", depth1)
        # print("mean_dep", mean_dep)
        return mean_dep

    def com_linear_depth(self, u, v):
        '''
        Inverse Perspective Mapping

        u=xc, v=yc
        '''
        z = (self.fy*self.h)/(v-self.cy)
        x = z*(u-self.cx)/self.fx
        print("line depth", x, z)

    def draw_bounding_boxes(self, _im, bbox, color_box=(0, 0, 0), thick_bbox=3, thick_circle=8):
        '''
        cv2 draw boundding box
        '''
        int_fun = lambda x_list: [int(x) for x in x_list]
        bbox = int_fun(bbox)
        im_box = cv2.rectangle(_im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color_box, thickness=thick_bbox)
        # im_box = cv2.circle(im_box, (bbox[0], bbox[1]), thick_circle, (0, 255, 0), -1)
        # im_box = cv2.circle(im_box, (bbox[0], bbox[3]), thick_circle, (0, 255, 0), -1)
        # im_box = cv2.circle(im_box, (bbox[2], bbox[1]), thick_circle, (0, 255, 0), -1)
        # im_box = cv2.circle(im_box, (bbox[2], bbox[3]), thick_circle, (0, 255, 0), -1)
        #
        # im_box = cv2.circle(im_box, (bbox[0], bbox[1]), thick_circle, (0, 0, 0), 1)
        # im_box = cv2.circle(im_box, (bbox[0], bbox[3]), thick_circle, (0, 0, 0), 1)
        # im_box = cv2.circle(im_box, (bbox[2], bbox[1]), thick_circle, (0, 0, 0), 1)
        # im_box = cv2.circle(im_box, (bbox[2], bbox[3]), thick_circle, (0, 0, 0), 1)
        return im_box

    def plot_depth_point(self, x, y):
        # font = FontProperties(fname=u'data/simhei.ttf', size=14)
        # font_name = "STKaiti"
        # matplotlib.rcParams['font.family'] = font_name
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'

        fig, ax = plt.subplots()

        # 设置横纵坐标轴范围
        # 设置刻度字体大小
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.set_ylim(0, 80)
        ax.set_xlim(0, 80)

        # 加横纵坐标
        plt.xlabel('ground truth depth(m)', fontsize=20)
        plt.ylabel('predicted depth(m)', fontsize=20)

        plt.plot(x, y, 'o', color="#e59572", markersize=3)
        # plt.scatter(x, y, s, c="#ea7070", alpha=0.5)

        # Space plots a bit
        plt.subplots_adjust(hspace=0.25, wspace=0.40)

        plt.show()

class Adaptive_Network(LSTM, CITY):
    def __init__(self):
        # init paraments
        self.iters = 1000000
        self.batch_size = 1
        self.count = 0
        self.memory_size = 100
        self.k = 10
        self.zeta_d = 0.1
        self.ada = {}
        self.lr = 0.00001
        self.lstm_out = 512
        self.threshold = 25
        self.scale = 1.5
        self.depth = 1 # kitti=2 city=1
        # self.data_name = "kitti" #　"kitti" or "city"
        self.data_name = "city" #　"kitti" or "city"

        # self.kitti_png = '/media/dyz/Data/KITTI2012/PNGImages'
        self.kitti_png = 'E:\\pyproject\\KITTI2012\\KITTI2012\\PNGImages'
        self.placeholder()
        self.c = 30

        # init children class
        super(Adaptive_Network, self).__init__(self.k, self.lstm_out, self.batch_size)
        CITY.__init__(self)

        self.train_data, self.test_data, self.memory_store = self.data_process()

    def placeholder(self):
        self.x  = tf.placeholder(tf.float32, shape=[None, 4096])
        self.y  = tf.placeholder(tf.float32, shape=[None, self.depth])
        self.mx = tf.placeholder(tf.float32, shape=[self.k, 4096])
        self.my = tf.placeholder(tf.float32, shape=[self.k, self.depth])
        self.is_training = True

    def build_network(self):
        emb_label = slim.fully_connected(self.my, self.lstm_out,
        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        trainable=self.is_training, activation_fn=None, scope='enbedded_label')

        # neighbor embedding concat embedded label
        concat_vector = tf.concat([self.mx, emb_label], 1)

        memory_state = slim.fully_connected(concat_vector, self.lstm_out,
        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        trainable=self.is_training, activation_fn=None, scope='concat_vector')


        memory_state = tf.expand_dims(memory_state, 0)




        # lstm zjk有问题
        out_state = self.call_lstm(memory_state, self.x)

        # logits linear layers
        dis = slim.fully_connected(out_state, self.depth,
        weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        trainable=self.is_training, activation_fn=None, scope='out')

        self.ada["dis"] = dis

    def loss(self):
        pre_dis = self.ada["dis"]
        loss = tf.reduce_sum(tf.norm(tf.abs(pre_dis - self.y), axis=1))
        c = tf.constant(self.c, dtype=tf.float32)
        loss = tf.cond(tf.greater(loss, c), lambda: (loss**2 + c**2)/(2*c), lambda: loss)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optim = tf.train.AdamOptimizer(self.lr)
        grads = optim.compute_gradients(loss)
        train_step = optim.apply_gradients(grads)

        self.ada["ext"] = extra_update_ops
        self.ada["train_step"] = train_step
        self.ada["loss"] = loss

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

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

    def run(self):
        saver = tf.train.Saver(max_to_keep=100)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))
        # var_keep_dic = self.get_variables_in_checkpoint_file("E:\\pyproject\\AMDEN\\data\\amden_ckpt\\L_M_100\\avden_430000.ckpt")
        # Get the variables to restore, ignorizing the variables to fix
        # variables_to_restore = self.get_variables_to_restore(variables, var_keep_dic)
        # restorer = tf.train.Saver(variables_to_restore)
        # restorer.restore(sess, "E:\\pyproject\\AMDEN\\data\\amden_ckpt\\L_M_100\\avden_430000.ckpt")
        # print('Loaded.')

        # saver.restore(sess, "E:\\pyproject\\AMDEN\\data\\amden_ckpt\\lm100_z_new\\avden_kitti_city_z_0.ckpt") # kitti
        saver.restore(sess, "../../data\\amden_ckpt\\lm100_z_new\\avden_kitti_city_z_0.ckpt") # kiit_city
        # best kitti: 430000: 0.773
        # saver.restore(sess, "E:\\pyproject\\AMDEN\\data\\amden_ckpt\\city\\avden_180000.ckpt")
        # sess.run(tf.global_variables_initializer())
        # self.run_train(sess, saver)
        acc, pre, rea = self.run_test(sess)
        return pre, rea


    def run_train(self, sess, saver):
        # train
        acc_list = []
        # for kk in range(180001, self.iters+1):
        for kk in range(38000, self.iters):
            queary_x, queary_y = self.call_data()
            ###### add by zjk
            if self.depth == 1 and len(queary_y) == 2:
                queary_y = queary_y[1]
            ###### add by zjk
            # print("queary_x, queary_y", queary_x, queary_y)
            # find neighbor
            neighbor_vector, neighbor_label = self.KNN(self.memory_store, queary_x, self.k)
            ###### add by zjk
            if self.depth == 1 and len(neighbor_label) == 2:
                neighbor_label = neighbor_label[1]
            ###### add by zjk
            self.is_training = True
            ada = sess.run(self.ada, feed_dict={self.x: queary_x, self.y: queary_y,
                                          self.mx: neighbor_vector, self.my: neighbor_label
                                          })

            if (kk % 2000 == 1) and (kk < 200000):
                self.update(ada["dis"], queary_y, queary_x)
                # self.run_test(sess)
                # np.save("data/mat/city_memory_"+str(kk)+".npy", np.array(self.memory_store))


            if kk % 2000 == 0:
                if kk < 5001:
                    self.lr = 0.0001
                elif 5000 < kk < 100001:
                    self.lr = 0.0001
                elif 100000 < kk < 300001:
                    self.lr = 0.00005
                elif 300000 < kk < 400000:
                    self.lr = 0.00001
                else:
                    self.lr = 0.000001
                acc, pre, rea = self.run_test(sess)
                acc_list.append([kk, acc])
                list_save(acc_list, "../../data\\document\\kitti_lm100_z_acc_new.txt")
                print(kk, "Ground Depth:", queary_y[0], "pre depth:", ada["dis"][0], "loss:", ada["loss"])
                print("=="*80)

            if kk % 2000 == 0:
                if self.data_name == 'kitti':
                    saver.save(sess, "../../data\\amden_ckpt\\lm100_z_new\\avden_z_"+str(kk)+".ckpt")
                    np.save("../../data\\mat\\kitti_memory100_z_new", np.array(self.memory_store))
                else:
                    saver.save(sess, "../../data\\amden_ckpt\\lm100_z_new\\avden_kitti_city_z_" + str(kk) + ".ckpt")
                    np.save("../../data\\mat\\city_memory", np.array(self.memory_store))

    def run_test(self, sess):
        queary_test = self.call_test() # city_depth_label
        count = 0
        acc = 0
        pre_dep = []
        rea_dep = []
        self.is_training = False
        pre_name = None
        while True:
            try:
                [queary_x_test, queary_y_test, bbox, kiname] = next(queary_test)
                # print("name", kiname, queary_y_test)
                ###### add by zjk
                if self.depth == 1 and len(queary_y_test) == 2:
                    queary_y_test = queary_y_test[1]
                ###### add by zjk
                # find neighbor
                neighbor_vector, neighbor_label = self.KNN(self.memory_store, queary_x_test, self.k)
                ###### add by zjk
                if self.depth == 1 and len(neighbor_label) == 2:
                    neighbor_label = neighbor_label[1]
                ###### add by zjk
                dis = sess.run(self.ada["dis"], feed_dict={self.x:queary_x_test,
                                      self.mx: neighbor_vector, self.my: neighbor_label
                                      })
                if self.data_name == "kitti":
                    pd = self.norm(dis[0])
                    gd = self.norm(queary_y_test[0])  ## kitti
                    # gd = queary_y_test[0][0]  ## city
                else:
                    pd = dis[0][0]
                    gd = queary_y_test[0][0]

                pre_dep.append(pd)
                rea_dep.append(gd)
                # print("pd:", pd, "gd", gd)
                if abs(gd - pd)/gd < self.zeta_d:
                    acc += 1

                count += 1

                
                # draw
                # if kiname == pre_name:
                #     im = self.draw_bbox_with_depth(im, kiname, bbox, pd, gd)
                # else:
                #     if pre_name is not None:
                #         cv2.imwrite("data/png/plot_kitti/" + kiname, im)
                #         # cv2.imshow("city", im)
                #         # cv2.waitKey()
                #         # pass
                #     # city
                #     # im = cv2.imread(os.path.join(self.city_path, "val", kiname.split("_")[0], kiname))
                #     # kitti
                #     im = cv2.imread(os.path.join(self.kitti_png, kiname))
                #
                #     pre_name = kiname
                
            except StopIteration:
                break
        # print("Acc:", acc/count, count)
        # print("Accy:", acc_y/count, "Accz:", acc_z/count)
        np.save('../../data\\result_pre.npy', np.array(pre_dep))
        np.save('../../data\\result_rea.npy', np.array(rea_dep))
        self.MAEs(pre_dep, rea_dep)
        return acc/count, pre_dep, rea_dep
    '''
    city kitti_memory
    0 train:
    100m: scale = 1.5
    mae 6.71  rmse:9.71 log rmse:0.367 ard:0.334
    Threshold 0.6308243727598566 0.8361029651352232 0.9253828608667318
    Acc: [0.1645813282001925, 0.3102341995508502, 0.4276547962784729, 0.518447224895733, 0.5941610522938723, 0.6535129932627527]

    50m:scale = 1.5
    mae 6.26  rmse:8.98 log rmse:0.373 ard:0.351

    25m:scale = 1.5
    mae 5.58  rmse:8.15 log rmse:0.445 ard:0.48

    '''
    def data_process(self):
        _kitti_train = []
        _kitti_test = []
        _city_train = []
        _city_val = []

        # kitti train
        # for key in kitti_train.keys():
        #     im = cv2.imread(os.path.join(self.kitti_png, key))
        #     if 10 >= len(kitti_train[key]) >= 1:
        #         for data in kitti_train[key]:
        #             if len(data) == 4: # fc bbox pd_raw gd
        #                 gd = [data[3][0][0], data[3][0][2]]
        #                 _kitti_train.append([data[0][0], data[1][0],
        #                                    data[2][0], gd, key])
        # np.save("E:\\pyproject\\AMDEN\\data\\mat\\kitti_train_new.npy", np.array(_kitti_train))
        _kitti_train = kitti_train_npy.tolist()

        ## memory
        _kitti_memory = _kitti_train[:self.memory_size] # kitti memory
        _kitti_memory = [[m[0], m[3], 0] for m in _kitti_memory]
        _kitti_memory = kitti_memory_0323.tolist()


        # kitti test
        # for key in kitti_test.keys():
        #     if 10 >= len(kitti_test[key]) >= 1:
        #         for data in kitti_test[key]:
        #             if len(data) == 4: # fc bbox pd_raw gd name
        #                 gd = [data[3][0][0], data[3][0][2]]
        #                 _kitti_test.append([data[0][0], data[1][0],
        #                                    data[2][0], gd, key])
        # np.save("E:\\pyproject\\AMDEN\\data\\mat\\kitti_test_new.npy", np.array(_kitti_test))
        _kitti_test = kitti_test_npy.tolist()


        ## ----------------------------------------------------------------------
        ## 处理city数据并加入标准距离
        ## city train pre process 进行过一次处理之后存为city_train.npy文件
        # for key in sorted(city_train.keys()):
        #     print("key", key)
        #     if 10 >= len(city_train[key]) >= 1:
        #         for city_data in city_train[key]:
        #             if len(city_data) == 3:
        #                 bbox = city_data[1][0]
        #                 folder_name = key.split("_")
        #                 dis_path_name = os.path.join(self.dis_path, "train", folder_name[0], key[:-15]+"disparity.png")
        #                 ##zjk 字符替换
        #                 dis_path_name = dis_path_name.replace('leftImg8bit_trainvaltest\\leftImg8bit', 'disparity_trainvaltest\\disparity')
        #                 ##end
        #                 bbox[0] = bbox[0] * 1.65
        #                 bbox[1] = bbox[1] * 2.73
        #                 bbox[2] = bbox[2] * 1.65
        #                 bbox[3] = bbox[3] * 2.73
        #                 depth = self.load_sig_depth(dis_path_name, bbox) # real depth label
        #                 # city_data: fc bbox fc_depth
        #                 if depth != 0:
        #                     print("depth", depth)
        #                     _city_train.append([city_data[0][0], city_data[1][0], city_data[2][0], depth, key])
        # # np.save("data/mat/city_train.npy", np.array(_city_train))
        # np.save("E:\\pyproject\\amden_new\\data\\mat\\city_train_new.npy", np.array(_city_train))
        _city_train = city_train_npy.tolist()

        ## city val
        # for key in sorted(city_val.keys()):
        #     # print(key)
        #     if 10>= len(city_val[key]) >= 1:
        #         for city_data in city_val[key]:
        #             if len(city_data) == 3:
        #                 bbox = city_data[1][0]
        #                 folder_name = key.split("_")
        #                 dis_path_name = os.path.join(self.dis_path, "val", folder_name[0], key[:-15]+"disparity.png")
        #                 ##zjk 字符替换
        #                 dis_path_name = dis_path_name.replace('leftImg8bit_trainvaltest\\leftImg8bit',
        #                                                       'disparity_trainvaltest\\disparity')
        #                 ##end
        #                 bbox[0] = bbox[0] * 1.65
        #                 bbox[1] = bbox[1] * 2.73
        #                 bbox[2] = bbox[2] * 1.65
        #                 bbox[3] = bbox[3] * 2.73
        #                 depth = self.load_sig_depth(dis_path_name, bbox) # real depth label
        #                 # city_data: fc bbox fc_depth
        #                 if depth != 0:
        #                     print("depth", depth)
        #                     _city_val.append([city_data[0][0], city_data[1][0], city_data[2][0], depth, key])
        # np.save("E:\\pyproject\\amden_new\\data\\mat\\city_val_new.npy", np.array(_city_val))
        _city_val = city_val_npy.tolist()

        ## city memory
        # _city_memory = _city_train[:self.memory_size]
        # _city_memory = [[m[0], m[3], 0] for m in _city_memory]
        _city_memory = city_memory.tolist()
        # return _kitti_train, _kitti_test, _kitti_memory
        return _city_train, _city_val, _city_memory


    def call_data(self):
        queary_x = np.zeros((self.batch_size, 4096))
        queary_y = np.zeros((self.batch_size, self.depth))
        if (self.count + 1)*self.batch_size >= len(self.train_data):
            random.shuffle(self.train_data)
            self.count = 0
        for i, data in enumerate(self.train_data[self.count*self.batch_size:(self.count+1)*self.batch_size]):
            queary_x[i, :] = data[0]
            ###### add by zjk
            # print(len(data[3]))
            if self.depth == 1 and self.data_name == 'kitti':
                queary_y[i, :] = data[3][1]
            else:
                queary_y[i, :] = data[3]
            ###### add by zjk
            # queary_y[i, :] = data[3]
        self.count += 1
        return queary_x, queary_y

    def call_test(self):
        queary_test_list = []
        city_depth_label = []
        for i, data in enumerate(self.test_data):
            queary_x_test = np.zeros((self.batch_size, 4096))   # 初始化一定要放到迭代里面，否则后面append都是同一个对象！
            queary_y_test = np.zeros((self.batch_size, self.depth))   # 初始化一定要放到迭代里面，否则后面append都是同一个对象！
            bbox = np.zeros((self.batch_size, 4))
            queary_x_test[0, :] = data[0]
            if self.depth == 1 and self.data_name == 'kitti':
                queary_y_test[0, :] = data[3][1]
            else:
                queary_y_test[0, :] = data[3]
            bbox = data[1]
            # city_depth_label.append(data[2])
            queary_test_list.append([queary_x_test, queary_y_test, bbox, data[4]])
        return iter(queary_test_list)  #, iter(city_depth_label)

    def KNN(self, memory_store, queays, TopK):
        '''
        memory_store:  memory_size*[[4096 dim vector], [1 dim dis]]
        queay: queay list [None, 4096]
        TopK: top-k
        '''
        queary = queays[0, :]
        # print(queary.shape)

        distance_list = []
        for xtr in memory_store:
            euclidean_distances = np.linalg.norm(xtr[0] - queary)
            # print("distance", euclidean_distances)
            distance_list.append(euclidean_distances)

        sort_dis = self.arg_sort(distance_list, TopK)
        # print("sort_dis", sort_dis)

        memory_x = np.zeros((self.k, 4096))
        memory_y = np.zeros((self.k, self.depth))
        for i, res in enumerate(sort_dis):
            memory_x[i, :] = memory_store[res[0]][0]
            ###### add by zjk
            # print(len(list(memory_store[res[0]][1])[0]))
            # print(list(memory_store[res[0]][1])[0][1])
            if self.depth == 1 and len(list(memory_store[res[0]][1])[0]) == 2:
                memory_y[TopK-1-i, :] = list(memory_store[res[0]][1])[0][1]
            else:
                memory_y[TopK-1-i, :] = memory_store[res[0]][1]
            ###### add by zjk
            # memory_y[i, :] = memory_store[res[0]][1]
        return memory_x, memory_y

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

    def update(self, pd, gd, que_x):
        # print("pd", pd[0], gd[0])
        ard = abs(self.norm(pd[0]) - self.norm(gd[0]))/self.norm(gd[0])
        self.memory_store.append([que_x, gd, ard])
        # print("ard", ard)

        ard_list = [a[2] for a in self.memory_store]
        # print("ard_list", ard_list)
        ard_sort = self.arg_sort(ard_list, self.memory_size, True)
        # print("ard_sort", ard_sort)
        # input()

        new_memory = []

        for i, ards in enumerate(ard_sort):
            new_memory.append(self.memory_store[ards[0]])

        self.memory_store = new_memory


    def draw(self, pd, gd):
        '''
        self.test_data: fc, real_box, ddis, real_dis
        '''
        # print(gd)
        # xx = [self.norm(dx) for dx in gd] # 没经过ada处理
        # yy = [self.norm(dy) for dy in pd]
        #
        # # 筛选数据for plot
        # xx = []
        # yy =[]
        # for i, xs in enumerate(x):
        #     if xs <= 10:
        #         zeto = 1
        #     elif 10 < xs <= 30:
        #         zeto = 1
        #     elif 30 < xs < 50:
        #         zeto = 1
        #     else:
        #         zeto = 1
        #
        #     if abs(xs - y[i])/abs(y[i]) > zeto:
        #         pass
        #     else:
        #         xx.append(xs)
        #         yy.append(y[i])
        xx = gd
        yy = pd
        print("maxx", max(xx))
        print("maxy", max(yy))
        self.plot_depth_point(xx, yy)

    def norm(self, xyz):
        return np.sqrt(sum([x**2 for x in xyz]))

    def MAEs(self, pds, gds):
        mae = 0 # abs rel
        aqrel = 0
        rmse = 0
        logrmse = 0
        ard = 0
        count = 0
        di_count = 0
        log_count = 0
        the1, the2, the3 = 0, 0, 0
        for i, pd in enumerate(pds):
            if gds[i] < self.threshold:
                if gds[i] == 0:
                    gds[i] += 10e-8
                mae += abs(pd - gds[i])
                rmse += (pd - gds[i])**2

                if gds[i] <= 0.1:
                    di_count += 1
                else:
                    aqrel += abs(pd - gds[i])**2/gds[i]
                    ard += abs(pd - gds[i])/gds[i]
                if gds[i] < 1:
                    log_count += 1
                else:
                    logrmse += (np.log(pd) - np.log(gds[i]))**2
                # threshold = 1.25 1.25**2 1.25**3
                if pd / gds[i] < 1.25 and gds[i] / pd < 1.25:
                    the1 += 1
                if pd / gds[i] < 1.25 ** 2 and gds[i] / pd < 1.25 ** 2:
                    the2 += 1
                if pd / gds[i] < 1.25 ** 3 and gds[i] / pd < 1.25 ** 3:
                    the3 += 1
                count += 1
        print("count", ard, count)
        print("MAE:", mae/count)
        print("RMSE:", np.sqrt(rmse/count))
        print("Log RMSE:", np.sqrt(logrmse/(count - log_count)))
        print("ARD/AbsRel:", ard/(count - di_count))
        print("aqrel", aqrel/(count - di_count))
        print("Threshold", the1/count, the2/count, the3/count)
        self.com_acc(pds, gds)

    def com_acc(self, pre, rea):
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

    def draw_bbox_with_depth(self, im, kiname, bbox, pd, gd):
        '''
        cv2 color:(B, G, R)
        '''
        # print(kiname, bbox, pd, gd)
        bbox = [int(b) for b in bbox]

        im = self.draw_bounding_boxes(im, bbox, color_box=(230, 182, 65), thick_bbox=3)
        im = cv2.putText(im, '{:.1f}m'.format(pd),
                        (bbox[0], bbox[1] -8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (45,179, 247), 2, 2)
        im = cv2.putText(im, '{:.1f}m'.format(gd),
                        (bbox[0], bbox[3] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 222), 2, 2)
        return im


if __name__ == '__main__':

    AN = Adaptive_Network()
    AN.build_network()
    AN.loss()
    pre, rea = AN.run()
    # print("real", rea)
    # print("pre", pre)
    # list_save(rea, "data/real.txt")
    # list_save(pre, "data/pre.txt")
    # AN.draw(pre, rea)

    # ci = CITY()
    # ci.load()

# ------------------------------------------------------------------------------
