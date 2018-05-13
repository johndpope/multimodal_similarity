import glob
import pickle as pkl
import numpy as np
import sys
import os
import pdb
from PIL import Image
import gensim

def extract_feat(frames_name):
    """
    Reference:
        1. Vasili's codes
        2. https://github.com/tensorflow/models/issues/429#issuecomment-277885861
    """

    slim_dir = "/home/xyang/workspace/models/research/slim"
    checkpoints_dir = slim_dir + "/pretrain"
    checkpoints_file = checkpoints_dir + '/inception_resnet_v2_2016_08_30.ckpt'
    batch_size = 256

    sys.path.append(slim_dir)
    from nets import inception
    import tensorflow as tf
    slim = tf.contrib.slim
    image_size = inception.inception_resnet_v2.default_image_size

    feat_conv = []
    feat_fc = []
    probs = []
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
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            logits, endpoints = inception.inception_resnet_v2(preprocessed_images,
                                                              is_training=False)
        pre_pool = endpoints['Conv2d_7b_1x1']
        pre_logits_flatten = endpoints['PreLogitsFlatten']
        probabilities = endpoints['Predictions']

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, checkpoints_file)

            for i in range(0, len(frames_name), batch_size):
                print i, '/', len(frames_name)
                current_batch = np.zeros((batch_size, 300, 300, 3), dtype=np.uint8)
                for j in range(batch_size):
                    if i+j == len(frames_name):
                        j -= 1
                        break
                    img = Image.open(frames_name[i+j]).convert('RGB').resize((300,300))
                    current_batch[j] = np.array(img)

                temp_conv, temp_fc, prob = sess.run([pre_pool,
                                                     pre_logits_flatten,
                                                     probabilities],
                                                     feed_dict={input_batch:
                                                         current_batch})
#                feat_conv.append(temp_conv.astype('float32'))
                feat_fc.append(temp_fc[:j+1].astype('float32'))
                probs.append(prob[:j+1].astype('float32'))

    return np.concatenate(feat_fc, axis=0), np.concatenate(probs, axis=0)
#    return np.concatenate(feat_conv, axis=0), np.concatenate(feat_fc, axis=0), np.concatenate(probs, axis=0)

result_dir = '/mnt/work/Stanford40/results/'
label_dir = '/mnt/work/Stanford40/ImageSplits/'
image_dir = '/mnt/work/Stanford40/JPEGImages/'

label_files = glob.glob(label_dir+'*')

word2vec = gensim.models.KeyedVectors.load_word2vec_format('/home/xyang/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
text_dict = pkl.load(open(result_dir+'label.pkl','r'))['text_dict']    # for one-hot representation

# load training data
train_names = []
train_labels = []
train_text = []

for f in label_files:
    if '_train' in f:
        fin = open(f, 'r')
        count = 0
        for line in fin:
            l = line.strip()
            train_names.append(image_dir+l)
            train_labels.append(l[:-8])
            count += 1

        # extract text feature
        vectors = [word2vec[w].reshape(1,-1) for w in l[:-8].split('_') if w in word2vec]
        vectors = np.concatenate(vectors, axis=0)
        vectors = np.mean(vectors, axis=0)
        train_text.append(np.tile(vectors.reshape(1,-1), (count,1)) + (np.random.rand(count,300)-0.5)*0.01)    # add tiny permutation

#        # one-hot representation
#        v = np.zeros((count, len(text_dict.keys())))
#        for w in l[:-8].split('_'):
#            v[:, text_dict[w]] = 1
#        train_text.append(v)

train_text = np.concatenate(train_text, axis=0)

train_feats, train_probs = extract_feat(train_names)
train_data = {'names': train_names, 'feats': train_feats, 'label':train_labels, 'probs':train_probs, 'text':train_text}

pkl.dump(train_data, open(result_dir+'train_data.pkl', 'w'))


# load testing data
test_names = []
test_labels = []
test_text = []

for f in label_files:
    if '_test' in f:
        fin = open(f, 'r')
        for l in fin:
            l = line.strip()
            test_names.append(image_dir+l)
            test_labels.append(l[:-8])

        # extract text feature
        vectors = [word2vec[w].reshape(1,-1) for w in l[:-8].split('_') if w in word2vec]
        vectors = np.concatenate(vectors, axis=0)
        vectors = np.mean(vectors, axis=0)
        test_text.append(np.tile(vectors.reshape(1,-1), (count,1)) + (np.random.rand(count,300)-0.5)*0.01)    # add tiny permutation

test_text = np.concatenate(test_text, axis=0)

test_feats, test_probs = extract_feat(test_names)
test_data = {'names': test_names, 'feats': test_feats, 'label':test_labels, 'probs':test_probs, 'text': test_text}

pkl.dump(test_data, open(result_dir+'test_data.pkl', 'w'))

