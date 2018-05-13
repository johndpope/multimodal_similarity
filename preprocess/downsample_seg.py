import glob
import os
import pdb
import numpy as np
import copy
from skimage import measure


def softmax(x):    # important to do in-place for large array
    #return np.exp(x-np.max(x,axis=-1,keepdims=True)) / \
    #        np.sum(np.exp(x-np.max(x,axis=-1,keepdims=True)),axis=-1,keepdims=True)
    temp = np.max(x,axis=-1,keepdims=True)
    res = np.subtract(x, temp, out=x)    # do subtraction in-place
    del temp
    res = np.exp(x, x)    # do exp in-place
    res = np.divide(x, np.sum(x, axis=-1, keepdims=True), x)    # do divide in-place

    return x

seg_root = '/mnt/data/honda_100h_archive/semantic_segmentation/'
feat_root = '/mnt/work/honda_100h/features/'

L = 3    # number of pyramid levels

files = glob.glob(seg_root+'*.npy')
for i, f in enumerate(files):
    session_id = os.path.basename(f).replace('_seg.npy',"")
    
    output_name = os.path.basename(f).replace('.npy', '_down.npy')
    print (i, '/', len(files)-1, ":", output_name)
    if os.path.isfile(feat_root+output_name):
        continue

    seg = np.load(f)
    N, H, W, D = seg.shape
    print ("Original: ", N, H, W, D)

    # for memory concern
    feat1 = measure.block_reduce(seg[:N//2], (1,5,5,1), np.max)
    seg = seg[N//2:]
    feat2 = measure.block_reduce(seg, (1,5,5,1), np.max)

    del seg
    feat = np.concatenate((feat1,feat2), axis=0)
    del feat1
    del feat2
    N, H, W, D = feat.shape
    print ("Downsampled: ", N, H, W, D)

    feat = softmax(feat)

    np.save(feat_root+output_name, feat)

