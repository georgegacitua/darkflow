from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from ..yolo.predict import preprocess
from ..yolo.data import shuffle
from copy import deepcopy
import pickle
import numpy as np
import os

def _batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    meta = self.meta
    labels = meta['labels']
    
    H, W, _ = meta['out_size']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']

    # preprocess
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    path = os.path.join(self.FLAGS.dataset, jpg)
    print('jpg:', jpg)
    img = self.preprocess(path, allobj)

    # Calculate regression target
    cellx = 1. * w / W
    celly = 1. * h / H
    for obj in allobj:
        #centerx = .5*(obj[1]+obj[3]) #xmin, xmax
        #centery = .5*(obj[2]+obj[4]) #ymin, ymax
        centerx = obj[1] #Center x ellipse
        centery = obj[2] #Center y ellipse
        a = obj[3] #Major axis
        b = obj[4] #Minor axis
        angle = obj[5] #Angle
        #Normalize angle
        a_cos = a * np.cos(angle)/w
        b_sin = b * np.sin(angle)/h
        angle = np.arctan2(b_sin, a_cos)
        cx = centerx / cellx
        cy = centery / celly
        #New Bounding box limits
        lim_x = np.sqrt(np.power(a * np.cos(angle),2) + np.power(b * np.sin(angle),2))
        lim_y = np.sqrt(np.power(a * np.sin(angle), 2) + np.power(b * np.cos(angle), 2))

        if cx >= W or cy >= H: return None, None
        #Modified objects
        obj[3] = lim_x / w
        obj[4] = lim_y / h
        obj[5] = np.cos(angle)
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery
        obj += [int(np.floor(cy) * W + np.floor(cx))]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([H*W,B,C])
    confs = np.zeros([H*W,B])
    coord = np.zeros([H*W,B,5])
    proid = np.zeros([H*W,B,C])
    prear = np.zeros([H*W,5])
    for obj in allobj:
        print('object:', obj)
        probs[obj[6], :, :] = [[0.]*C] * B
        probs[obj[6], :, labels.index(obj[0])] = 1.
        proid[obj[6], :, :] = [[1.]*C] * B
        coord[obj[6], :, :] = [obj[1:6]] * B
        prear[obj[6],0] = obj[1] - obj[3]**2 * .5 * W # xleft
        prear[obj[6],1] = obj[2] - obj[4]**2 * .5 * H # yup
        prear[obj[6],2] = obj[1] + obj[3]**2 * .5 * W # xright
        prear[obj[6],3] = obj[2] + obj[4]**2 * .5 * H # ybot
        prear[obj[6],4]  = obj[5] #cos angle
        confs[obj[6], :] = [1.] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft; 
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer 
    loss_feed_val = {
        'probs': probs, 'confs': confs, 
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft, 
        'botright': botright
    }

    return inp_feed_val, loss_feed_val

