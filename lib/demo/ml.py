import tensorflow as tf
from lib.config.config import FLAGS


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    # TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables


def smooth(x):
    if tf.abs(x) < 1:
        return 0.5*(x**2)
    else:
        return tf.abs(x) - 0.5


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights=1, bbox_outside_weights=1, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
        out_loss_box,
        axis=dim
    ))
    return loss_box

def loss_function(pi, pt, box_reg, box_gro, optimizer="SGD"):  # [512, 2]  box_reg(2000, 4):A   box_gro((?, 37, 62, 4, 9)):G
    rpn_cross_entropy = tf.reduce_mean(tf.square(pi - pt))
    # box_loss = _smooth_l1_loss(box_reg, box_gro)
    loss = rpn_cross_entropy
    if optimizer == "Mom":  # Momentum
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9).minimize(loss)
    elif optimizer == 'RMSProp':
        train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
    else:        # SGD
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    return loss