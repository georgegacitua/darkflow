import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from ..yolo.misc import show
import numpy as np
import os
import math
import random as r

def expit_tensor(x):
	return 1. / (1. + tf.exp(-x))

def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])
    sconf = float(m['object_scale'])
    snoob = float(m['noobject_scale'])
    scoor = float(m['coord_scale'])
    H, W, _ = m['out_size']
    B, C = m['num'], m['classes']
    HW = H * W # number of grid cells
    anchors = m['anchors']

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tH       = {}'.format(H))
    print('\tW       = {}'.format(W))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, HW, B, C]
    size2 = [None, HW, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)
    _confs = tf.placeholder(tf.float32, size2)
    _coord = tf.placeholder(tf.float32, size2 + [5])
    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2)
    _upleft = tf.placeholder(tf.float32, size2 + [2])
    _botright = tf.placeholder(tf.float32, size2 + [2])

    self.placeholders = {
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright
    }

    # Extract the coordinate prediction from net.out
    net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (5 + 1 + C)])
    coords = net_out_reshape[:, :, :, :, :5]
    coords = tf.reshape(coords, [-1, H*W, B, 5])
    adjusted_coords_xy = expit_tensor(coords[:,:,:,0:2])
    adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    adjusted_coords_angle = tf.reshape(tf.math.tanh(coords[:,:,:,4]), [-1,H*W,B,1])
    coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_coords_angle], 3)

    adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 5])
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B, 1])

    adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 6:])
    adjusted_prob = tf.reshape(adjusted_prob, [-1, H*W, B, C])

    adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_coords_angle, adjusted_c, adjusted_prob], 3)

    #Area
    pred_w = tf.pow(coords[:,:,:,2], 2) * np.reshape([W, H], [1, 1, 1, 1])
    pred_h = tf.pow(coords[:, :, :, 3], 2) * np.reshape([W, H], [1, 1, 1, 1])
    #area_pred = wh[:,:,:,0] * wh[:,:,:,1]
    pred_x = coords[:,:,:,0]
    pred_y = coords[:,:,:,1]
    pred_angles = tf.math.acos(coords[:,:,:, 4])
    pred_left = pred_x - tf.multiply(pred_w, tf.math.abs(tf.math.cos(pred_angles))) - tf.multiply(pred_h, tf.math.sin(pred_angles))
    pred_right = pred_x + tf.multiply(pred_w, tf.math.abs(tf.math.cos(pred_angles))) + tf.multiply(pred_h, tf.math.sin(pred_angles))
    pred_up = pred_y - tf.multiply(pred_h, tf.math.sin(pred_angles)) - tf.multiply(pred_w, tf.math.abs(tf.math.cos(pred_angles)))
    pred_down = pred_y + tf.multiply(pred_h, tf.math.sin(pred_angles)) + tf.multiply(pred_w, tf.math.abs(tf.math.cos(pred_angles)))
    pred_width = tf.maximum(0.0, pred_right - pred_left)
    pred_height = tf.maximum(0.0, pred_down - pred_up)
    pred_areas = tf.multiply(pred_width, pred_height)
    #floor = centers - (wh * .5)
    #ceil  = centers + (wh * .5)
    #True Area
    true_w = tf.pow(_coord[:, :, :, 2], 2) * np.reshape([W, H], [1, 1, 1, 1])
    true_h = tf.pow(_coord[:, :, :, 3], 2) * np.reshape([W, H], [1, 1, 1, 1])
    true_x = _coord[:, :, :, 0]
    true_y = _coord[:, :, :, 1]
    true_angles = tf.math.acos(_coord[:, :, :, 4])
    true_left = true_x - tf.multiply(true_w, tf.math.abs(tf.math.cos(true_angles))) - tf.multiply(true_h, tf.math.sin(true_angles))
    true_right = true_x + tf.multiply(true_w, tf.math.abs(tf.math.cos(true_angles))) + tf.multiply(true_h, tf.math.sin(true_angles))
    true_up = true_y - tf.multiply(true_h, tf.math.sin(true_angles)) - tf.multiply(true_w, tf.math.abs(tf.math.cos(true_angles)))
    true_down = true_y + tf.multiply(true_h, tf.math.sin(true_angles)) + tf.multiply(true_w, tf.math.abs(tf.math.cos(true_angles)))
    true_width = tf.maximum(0, true_right - true_left)
    true_height = tf.maximum(0, true_down - true_up)
    true_areas = tf.multiply(true_width, true_height)

    # calculate the intersection areas
    #intersect_upleft   = tf.maximum(floor, _upleft)
    #intersect_botright = tf.minimum(ceil , _botright)
    intersect_up = tf.maximum(pred_up, true_up)
    intersect_left = tf.maximum(pred_left, true_left)
    intersect_right = tf.maximum(pred_right, true_right)
    intersect_down = tf.maximum(pred_down, true_down)

    intersect_width = tf.maximum(intersect_right - intersect_left, 0.0)
    intersect_heigtht = tf.maximum(intersect_down- intersect_up, 0.0)
    intersect = tf.multiply(intersect_width, intersect_heigtht)

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, true_areas + pred_areas - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs
    weight_coo = tf.concat(5 * [tf.expand_dims(confs, -1)], 3)
    cooid = scoor * weight_coo
    weight_pro = tf.concat(C * [tf.expand_dims(confs, -1)], 3)
    proid = sprob * weight_pro

    self.fetch += [_probs, confs, conid, cooid, proid]
    true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
    wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)

    print('Building {} loss'.format(m['model']))
    loss = tf.pow(adjusted_net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reshape(loss, [-1, H*W*B*(5 + 1 + C)])
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)