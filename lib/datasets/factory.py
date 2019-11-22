from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from lib.datasets.pascal_voc import pascal_voc
from lib.datasets.coco import coco
from lib.datasets.kitti_voc import kitti_voc
import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
# lambda作为一个表达式，定义了一个匿名函数, :前是函数输入 之后是输出 函数名师 = 前面的_sets
# 很明显，pascal_voc是一个类，这是调用了该类的构造函数，返回的也是该类的一个实例，所以这下我们清楚了imdb实际上就是pascal_voc的一个实例；

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up kitti_2012_<split>
for year in ['2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'kitti_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: kitti_voc(split, year))


def get_imdb(name):
  """Get an imdb (image database) by name.
  get_imdb这个函数的实现原理：_sets是一个字典，字典的key是数据集的名称，字典的value是一个lambda表达式（即一个函数指针），
  __sets内容如下：
  {'voc_2012_trainval': <function <lambda> at 0x000001D384B28A60>, 'kitti_2012_test': <function <lambda> at 0x000001D384FAF048>, 'voc_2007_trainval': <function <lambda> at 0x000001D384B28840>, 'coco_2014_minival': <function <lambda> at 0x000001D384B28C80>, 'voc_2007_val': <function <lambda> at 0x000001D384B286A8>, 'voc_2012_val': <function <lambda> at 0x000001D384B289D8>, 'coco_2014_trainval': <function <lambda> at 0x000001D384B28D90>, 'coco_2015_test-dev': <function <lambda> at 0x000001D384B28EA0>, 'coco_2014_val': <function <lambda> at 0x000001D384B28BF8>, 'coco_2014_train': <function <lambda> at 0x000001D384B28B70>, 'voc_2007_train': <function <lambda> at 0x000001D382802730>, 'coco_2014_valminusminival': <function <lambda> at 0x000001D384B28D08>, 'voc_2012_test': <function <lambda> at 0x000001D384B28AE8>, 'voc_2007_test': <function <lambda> at 0x000001D384B288C8>, 'kitti_2012_train': <function <lambda> at 0x000001D384B28F28>, 'coco_2015_test': <function <lambda> at 0x000001D384B28E18>, 'voc_2012_train': <function <lambda> at 0x000001D384B28950>}
    前面是名称 0x000001D384B28A60表示内存地址 每个object地址不一样 还都是lambda属性
    __sets[name]()如下：
    <lib.datasets.pascal_voc.pascal_voc object at 0x00000178713107F0>
    可以看到返回的是指向pascal_voc.py函数的object  0x00000178713107F0是这个object的内存地址
  """
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()    # 这句话实际上是调用函数，返回数据集imdb


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
