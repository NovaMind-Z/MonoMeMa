import os
import os.path as osp

import numpy as np
import tensorflow as tf 
import lib.config.flags_tf as flags_tf

FLAGS = flags_tf.FLAGS


# FLAGS = flags_tf.FLAGS
FLAGS2 = {}

######################
# General Parameters #
######################
# FLAGS2["pixel_means"] = np.array([[[102.9801, 115.9465, 122.7717]]])  # voc
FLAGS2["pixel_means"] = np.array([[[95.8774, 98.7605, 93.8331]]])  # kitti

flags_tf.DEFINE_integer('rng_seed', 3, "Tensorflow seed for reproducibility")

######################
# Network Parameters #
######################
flags_tf.DEFINE_string('network', "vgg16", "The network to be used as backbone")

#######################
# Training Parameters #
#######################
flags_tf.DEFINE_float('weight_decay', 0.0005, "Weight decay, for regularization")
flags_tf.DEFINE_float('learning_rate', 0.001, "Learning rate")
flags_tf.DEFINE_float('momentum', 0.9, "Momentum")
# flags_tf.DEFINE_float('gamma', 0.1, "Factor for reducing the learning rate")
flags_tf.DEFINE_float('gamma', 0.99, "Factor for reducing the learning rate")


flags_tf.DEFINE_integer('batch_size', 300, "Network batch size during training")
#zjk修改了训练轮数
# flags_tf.DEFINE_integer('max_iters', 2000000, "Max iteration")
flags_tf.DEFINE_integer('max_iters', 1000000, "Max iteration")

flags_tf.DEFINE_integer('step_size', 10000,
                            "Step size for reducing the learning rate, currently only support one step")
flags_tf.DEFINE_integer('display', 40,
                            "Iteration intervals for showing the loss during training, on command line interface")

flags_tf.DEFINE_string('initializer', "truncated", "Network initialization parameters")
# flags_tf.DEFINE_string('pretrained_model', "./data/imagenet_weights/vgg_16.ckpt", "Pretrained network weights") #zjk将vgg16改成了vgg_16

# flags_tf.DEFINE_string('pretrained_model', "E:\\pyproject\\amden_new\\default\\kitti_2012_train\\default\\vgg16_faster_rcnn_iter_256_410000.ckpt", "Pretrained network weights") #zjk将vgg16改成了vgg_16

flags_tf.DEFINE_string('pretrained_model', 'E:\\pyproject\\amden_new\\default\\kitti_2012_train\\default\\vgg16_faster_rcnn_iter_256_430000.ckpt', "Pretrained network weights") #zjk将vgg16改成了vgg_16

flags_tf.DEFINE_string('dis_weight', './data/imagenet_weights/dis_weights.txt', 'image distance network weights')
# zjk修改
# flags_tf.DEFINE_string('kitti_png', '/media/dyz/Data/KITTI2012/PNGImages',
#                            'image distance network weights')
flags_tf.DEFINE_string('kitti_png', 'E:\\pyproject\\datasets\\KITTI2012\\PNGImages',
                           'image distance network weights')

flags_tf.DEFINE_string('city_path', 'E:\\pyproject\\datasets\\CItySpaces\\leftImg8bit_trainvaltest\\leftImg8bit\\val',
                           'city scapes')
flags_tf.DEFINE_string('synthia_path', 'F:\\SYNTHIA\\SYNTHIA-SF\\SEQ1\\RGBLeft',
                           'synthia')
flags_tf.DEFINE_string('kitti2015_path', 'F:\\KITTI2015\\data_scene_flow\\training',
                           'kitti2015')
flags_tf.DEFINE_string('apollo_path', 'E:\\pyproject\\datasets\\Apollo\\stereo_train_1\\stereo_train_001',
                           'apollo_scapes')
# zjk修改
# flags_tf.DEFINE_string('Ann', '/media/dyz/Data/KITTI2012/Annotations',
#                            'image distance network weights')
flags_tf.DEFINE_string('Ann', 'E:\\pyproject\\datasets\\KITTI2012\\Annotations',
                           'image distance network weights')

flags_tf.DEFINE_string('train_path', "E:", "train path")


flags_tf.DEFINE_boolean('bias_decay', False, "Whether to have weight decay on bias as well")
flags_tf.DEFINE_boolean('double_bias', True, "Whether to double the learning rate for bias")
flags_tf.DEFINE_boolean('use_all_gt', True, "Whether to use all ground truth bounding boxes for training, "
                                                "For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''")
flags_tf.DEFINE_integer('max_size', 1242, "Max pixel size of the longest side of a scaled input image")
flags_tf.DEFINE_integer('test_max_size', 1242, "Max pixel size of the longest side of a scaled input image")
flags_tf.DEFINE_integer('ims_per_batch', 1, "Images to use per minibatch")
flags_tf.DEFINE_integer('snapshot_iterations', 5000, "Iteration to take snapshot")

FLAGS2["scales"] = (600,)
FLAGS2["test_scales"] = (600,)

######################
# Testing Parameters #
######################
flags_tf.DEFINE_string('test_mode', "top", "Test mode for bbox proposal")  # nms, top

##################
# RPN Parameters #
##################
flags_tf.DEFINE_float('rpn_negative_overlap', 0.3, "IOU < thresh: negative example")
flags_tf.DEFINE_float('rpn_positive_overlap', 0.7, "IOU >= thresh: positive example")
flags_tf.DEFINE_float('rpn_fg_fraction', 0.5, "Max number of foreground examples")
flags_tf.DEFINE_float('rpn_train_nms_thresh', 0.7, "NMS threshold used on RPN proposals")
flags_tf.DEFINE_float('rpn_test_nms_thresh', 0.7, "NMS threshold used on RPN proposals")

flags_tf.DEFINE_integer('rpn_train_pre_nms_top_n', 12000,
                            "Number of top scoring boxes to keep before apply NMS to RPN proposals")
flags_tf.DEFINE_integer('rpn_train_post_nms_top_n', 2000,
                            "Number of top scoring boxes to keep before apply NMS to RPN proposals")
flags_tf.DEFINE_integer('rpn_test_pre_nms_top_n', 6000,
                            "Number of top scoring boxes to keep before apply NMS to RPN proposals")
# flags_tf.DEFINE_integer('rpn_test_post_nms_top_n', 300,
#                             "Number of top scoring boxes to keep before apply NMS to RPN proposals")

# flags_tf.DEFINE_integer('rpn_test_post_nms_top_n', 128,
#                             "Number of top scoring boxes to keep before apply NMS to RPN proposals")

flags_tf.DEFINE_integer('rpn_test_post_nms_top_n', 300,
                            "Number of top scoring boxes to keep before apply NMS to RPN proposals")

flags_tf.DEFINE_integer('rpn_batchsize', 256, "Total number of examples")  #zjk注释
flags_tf.DEFINE_integer('rpn_positive_weight', -1,
                            'Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p).'
                            'Set to -1.0 to use uniform example weighting')
# flags_tf.DEFINE_integer('rpn_top_n', 300,
#                             "Only useful when TEST.MODE is 'top', specifies the number of top proposals to select")
# flags_tf.DEFINE_integer('rpn_top_n', 128,
#                             "Only useful when TEST.MODE is 'top', specifies the number of top proposals to select")
flags_tf.DEFINE_integer('rpn_top_n', 300,
                            "Only useful when TEST.MODE is 'top', specifies the number of top proposals to select")

flags_tf.DEFINE_boolean('rpn_clobber_positives', False,
                            "If an anchor satisfied by positive and negative conditions set to negative")

#######################
# Proposal Parameters #
#######################
flags_tf.DEFINE_float('proposal_fg_fraction', 0.25,
                          "Fraction of minibatch that is labeled foreground (i.e. class > 0)")
flags_tf.DEFINE_boolean('proposal_use_gt', False,
                            "Whether to add ground truth boxes to the pool when sampling regions")

###########################
# Bounding Box Parameters #
###########################
flags_tf.DEFINE_float('roi_fg_threshold', 0.5,
                          "Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)")
flags_tf.DEFINE_float('roi_bg_threshold_high', 0.5,
                          "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")
flags_tf.DEFINE_float('roi_bg_threshold_low', 0.0,
                          "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")

flags_tf.DEFINE_boolean('bbox_normalize_targets_precomputed', True,
                            "# Normalize the targets using 'precomputed' (or made up) means and stdevs (BBOX_NORMALIZE_TARGETS must also be True)")
flags_tf.DEFINE_boolean('test_bbox_reg', True, "Test using bounding-box regressors")

FLAGS2["bbox_inside_weights"] = (1.0, 1.0, 1.0, 1.0)
FLAGS2["bbox_normalize_means"] = (0.0, 0.0, 0.0, 0.0)
FLAGS2["bbox_normalize_stds"] = (0.1, 0.1, 0.1, 0.1)

FLAGS2["dis_inside_weights"] = (1.0, 1.0, 1.0)
##################
# ROI Parameters #
##################
flags_tf.DEFINE_integer('roi_pooling_size', 7, "Size of the pooled region after RoI pooling")

######################
# Dataset Parameters #
######################
FLAGS2["root_dir"] = osp.abspath(
    osp.join(osp.dirname(__file__), '..', '..'))  # 输出该脚本所在的完整路径 F:\ipython\code\Mask_RCNN\Mask_R-CNN
FLAGS2["data_dir"] = osp.abspath(osp.join(FLAGS2["root_dir"], 'data'))

######################
# other parameters
######################
#zjk修改： 调整了训练轮数，但是这个可能是第二个网络的参数喽
flags_tf.DEFINE_integer('iteration_numbers', 1000000, '''Iteration Number of training''')
# flags_tf.DEFINE_integer('iteration_numbers', 1000, '''Iteration Number of training''')

flags_tf.DEFINE_integer('display_step', 100, '''Display number to show the acc''')
# flags_tf.DEFINE_float('input_image', [500, 375, 3], '''Input image size [weight, height, depth]''')
flags_tf.DEFINE_multi_integer('input_image_orl_size', [1242, 375, 3], '''Input image size [weight, height, depth]''')
# flags_tf.DEFINE_multi_integer('input_image_orl_size', [2048, 1024, 3], '''Input image size [weight, height, depth]''')
flags_tf.DEFINE_integer('classes_numbers', 10, '''classes number of model''')
flags_tf.DEFINE_integer('cls_depth', 18, '''2*9 cls for object cls 1: foreground  0:background''')
flags_tf.DEFINE_integer('bbox_depth', 36, '''4*9 change coordinates for fix anchors''')

flags_tf.DEFINE_string(
    'dataset_name', 'coco',
    'The name of the dataset to convert, one of "coco", "cifar10", "flowers", "mnist".')

flags_tf.DEFINE_string(
    'dataset_dir', 'data',
    'The directory where the output TFRecords and temporary files are saved.')

flags_tf.DEFINE_string(
    'dataset_split_name', 'train2014',
    'data name for train or test')

flags_tf.DEFINE_boolean('vis', False, 'Show some visual masks')

KITTI_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
# color_list = ['red', 'blue', 'pink', 'yellow', 'green', 'gold', 'pink', 'gray', 'purple', 'yellow]
RGB_list = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0), (0, 255, 0), (244, 164, 96), (139, 101, 118),
            (207, 207, 207), (0, 255, 255)]
# The URL where the coco data can be downloaded.

_TRAIN_DATA_URL = "https://msvocds.blob.core.windows.net/coco2014/train2014.zip"
_VAL_DATA_URL = "https://msvocds.blob.core.windows.net/coco2014/val2014.zip"
_INS_LABEL_URL = "https://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip"
_KPT_LABEL_URL = "https://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip"
_CPT_LABEL_URL = "https://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip"
DATA_URLS = [
    _TRAIN_DATA_URL, _VAL_DATA_URL,
    _INS_LABEL_URL, _KPT_LABEL_URL, _CPT_LABEL_URL,
]

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__',
           'car', 'van', 'truck', 'boat',
           'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_550000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'kitti_voc': ('kitti_2012_trainval',), 'pascal_voc': ('voc_2007_trainval',),
            'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

bg_inds = np.array([443, 443, 1128, 1128, 443, 1128, 443, 443, 443, 443, 1128, 443, 443, 1128, 443,
           443, 443, 1128, 443, 1128, 443, 443, 1128, 443, 1128, 1128, 1128, 443, 1128, 443,
           443, 443, 443, 1128, 1128, 1128, 1128, 1128, 443, 443, 443, 1128, 443, 1128, 443,
           443, 443, 443, 443, 443, 443, 443, 1128, 443, 1128, 1128, 443, 1128, 443, 1128,
           1128, 1128, 1128, 1128, 443, 1128, 1128, 443, 443, 1128, 1128, 443, 1128, 1128, 1128,
           443, 1128, 1128, 443, 1128, 1128, 1128, 1128, 443, 1128, 443, 443, 1128, 443, 443,
           443, 443, 443, 1128, 443, 1128, 443, 443, 1128, 443, 443, 443, 443, 1128, 443,
           443, 1128, 1128, 1128, 1128, 443, 443, 443, 1128, 443, 443, 1128, 1128, 443, 443,
           1128, 443, 443, 443, 443, 1128, 443, 1128, 1128, 443, 1128, 1128, 443, 443, 443,
           443, 1128, 1128, 443, 1128, 443, 1128, 1128, 1128, 443, 443, 443, 1128, 1128, 1128,
           1128, 1128, 1128, 1128, 1128, 443, 1128, 443, 1128, 443, 1128, 443, 443, 443, 443,
           1128, 1128, 1128, 1128, 443, 1128, 1128, 1128, 1128, 1128, 1128, 443, 1128, 1128, 443,
           1128, 443, 443, 443, 1128, 443, 443, 1128, 1128, 1128, 443, 443, 443, 1128, 443,
           1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 443, 443, 1128, 443, 1128, 443,
           443, 443, 443, 1128, 1128, 443, 1128, 443, 1128, 443, 1128, 443, 443, 443, 443,
           443, 443, 443, 1128, 443, 443, 1128, 443, 443, 443, 1128, 443, 443, 1128, 443,
           443, 1128, 443, 1128, 1128, 443, 443, 1128, 443, 1128, 443, 443, 1128, 443, 443,
           443], dtype=np.int32)


def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(FLAGS2["root_dir"], FLAGS2["root_dir"], 'default', imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
ms = {}

x = []
y = []
z = []

px = []
py = []
pz = []
