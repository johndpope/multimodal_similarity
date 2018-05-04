# Extract spatial pyramid features of semantic segmentation
# Reference: Semantic segmentation as image representation for scene recognition

import glob
import os
import pdb
import numpy as np

def softmax(x):
    return np.exp(x-np.max(x,axis=-1,keepdims=True)) / \
            np.sum(np.exp(x-np.max(x,axis=-1,keepdims=True)),axis=-1,keepdims=True)

seg_root = '/mnt/data/honda_100h_archive/semantic_segmentation/'
feat_root = '/mnt/work/honda_100h/features/'

L = 3    # number of pyramid levels

files = glob.glob(seg_root+'*.npy')
for f in files:
    output_name = os.path.basename(f).replace('.npy', '_sp.npy')
    print (output_name)

    seg = np.load(f)
    seg = softmax(seg)
    N, H, W, D = seg.shape

    feat = []
    # for each level
    for l in range(L):
        h_size = H // (2**l)
        w_size = W // (2**l)
        # for each bin
        for i in range(2**l):
            for j in range(2**l):
                region = seg[:, i*h_size:(i+1)*h_size, j*w_size:(j+1)*w_size, :]
                # get histogram (soft) within a bin
                feat.append(np.mean(region, axis=(1,2)))

    # concatenate features of different levels and bins
    feat = np.concatenate(feat, axis=1)
    np.save(feat_root+output_name, feat)

