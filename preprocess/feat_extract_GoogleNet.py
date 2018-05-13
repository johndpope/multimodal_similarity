import glob
import pickle as pkl
import numpy as np
import sys
import os
import pdb
from PIL import Image
#import gensim

def extract_feat(frames_name):
    """
    Reference:
        1. Vasili's codes
        2. https://github.com/tensorflow/models/issues/429#issuecomment-277885861
    """

    slim_dir = "/home/xyang/workspace/models/research/slim"
    checkpoints_dir = slim_dir + "/pretrain"
    checkpoints_file = checkpoints_dir + '/inception_v1.ckpt'
    batch_size = 256

    sys.path.append(slim_dir)
    from nets import inception
    import tensorflow as tf
    slim = tf.contrib.slim
    image_size = inception.inception_v1.default_image_size

    feat = []
    with tf.Graph().as_default():
        input_batch = tf.placeholder(dtype=tf.uint8,
                                     shape=(batch_size, 300, 300, 3))
        resized_images = tf.image.resize_images(
            tf.image.convert_image_dtype(input_batch, dtype=tf.float32),
            [image_size, image_size]
            )
        preprocessed_images = tf.multiply(tf.subtract(resized_images, 0.5), 2.0)
        
        # Create the model, use the default arg scope to configure
        # the batch norm parameters.
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, endpoints = inception.inception_v1(preprocessed_images,
                                                        num_classes=1001,
                                                        is_training=False)
        pool5 = endpoints['AvgPool_0a_7x7']

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoints_file)

            for i in range(0, len(frames_name), batch_size):
                print (i, '/', len(frames_name))
                current_batch = np.zeros((batch_size, 300, 300, 3), dtype=np.uint8)
                for j in range(batch_size):
                    if i+j == len(frames_name):
                        j -= 1
                        break
                    img = Image.open(frames_name[i+j]).convert('RGB').resize((300,300))
                    current_batch[j] = np.array(img)

                feat_batch = sess.run(pool5, feed_dict={input_batch: current_batch})
                feat_batch = np.squeeze(feat_batch)
                feat.append(feat_batch[:j+1].astype('float32'))

    return np.concatenate(feat, axis=0)

data_root = '/mnt/work/CUB_200_2011/'
images_root = data_root+'images/'
output_root = data_root+'data/'

with open(data_root+'images.txt', 'r') as fin:
    image_files = fin.read().strip().split('\n')

with open(data_root+'image_class_labels.txt', 'r') as fin:
    labels = fin.read().strip().split('\n')

train_files = []
train_label = []
test_files = []
test_label = []
for i in range(len(image_files)):
    label = int(labels[i].split(' ')[1])
    if label <= 100:
        train_files.append(images_root+image_files[i].split(' ')[1])
        train_label.append(label)
    else:
        test_files.append(images_root+image_files[i].split(' ')[1])
        test_label.append(label)


train_feat = extract_feat(train_files)
test_feat = extract_feat(test_files)

np.save(output_root+'feat_train.npy', train_feat)
np.save(output_root+'feat_test.npy', test_feat)
np.save(output_root+'label_train.npy', np.asarray(train_label, dtype='int32'))
np.save(output_root+'label_test.npy', np.asarray(test_label, dtype='int32'))
