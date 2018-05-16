import glob
import pickle as pkl
import numpy as np
import sys
import os
import pdb

data_root = '/mnt/work/CUB_200_2011/'
output_root = data_root+'data/'

with open(data_root+'attributes/image_attribute_labels.txt', 'r') as fin:
    lines = fin.read().strip().split('\n')


train_att = np.zeros((5864, 312), dtype='float32')
test_att = np.zeros((5924, 312), dtype='float32')

for i in range(len(lines)):
    line = lines[i].split(' ')
    img_id = int(line[0])
    att_id = int(line[1])
    att_flag = int(line[2])
    att_conf = int(line[3])

    if img_id <= 5864:
        if att_flag == 1:
            if att_conf == 3:
                att_flag *= 0.75
            elif att_conf == 2:
                att_flag *= 0.5
            train_att[img_id-1, att_id-1] = att_flag
    else:
        if att_flag == 1:
            if att_conf == 3:
                att_flag *= 0.75
            elif att_conf == 2:
                att_flag *= 0.5
            test_att[img_id-5895, att_id-1] = att_flag
pdb.set_trace()

np.save(output_root+'att_train.npy', train_att)
np.save(output_root+'att_test.npy', test_att)
