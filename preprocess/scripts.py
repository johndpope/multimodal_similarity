import numpy as np
import glob
import os
import pdb
import sys

"""
check consitency and completeness of extracted frames & features
"""
#frame_dir='/mnt/work/honda_100h/frames/'
#feature_dir='/mnt/work/honda_100h/features/'
#
#feature_files = glob.glob(feature_dir+'*sensors.npy')
#n = len(feature_files)
#correct = 0
#wrong_list = []
#empty_list = []
#for i in range(n):
#    session_id = os.path.basename(feature_files[i]).split('_')[0]
#    N = np.load(feature_files[i].strip('_sensors')).shape[0]
#
#    if os.path.isdir(frame_dir+session_id):
#        l = len(glob.glob(frame_dir+session_id+'/*jpg'))
#        if l == N:
#            correct += 1
#        else:
#            wrong_list.append((session_id,l-N))
#    else:
#        empty_list.append(session_id)
#
#print correct, "out of ", n, "is correct"
#print "wrong_list: ", wrong_list
#print "empty_list: ", empty_list


"""
remove frames of sessions without label (137 out of 203 has labels)
"""
#import shutil
#
#frame_dir='/mnt/work/honda_100h/frames/'
#feature_dir='/mnt/work/honda_100h/features/'
#
#feature_files = glob.glob(feature_dir+'*sensors.npy')
#frame_files = glob.glob(frame_dir+'*')
#
#labeled_sessions = []
#for i in range(len(feature_files)):
#    session_id = os.path.basename(feature_files[i]).split('_')[0]
#    labeled_sessions.append(session_id)
#
#for i in range(len(frame_files)):
#    session_id = os.path.basename(frame_files[i])
#    if session_id not in labeled_sessions:
#        shutil.rmtree(frame_dir+session_id)
        
"""
Test code for itertools.combinations
"""
#import itertools
#import random
#
#idx = {0: [1,2,3], 1: [4,5], 2:[6,7,8,9]}
#for key in idx:
#    random.shuffle(idx[key])
#    idx[key] = itertools.combinations(idx[key], 2)
#
#l = []
#while len(l) < 8:
#    keys = idx.keys()
#    for key in keys:
#        try:
#            l.append(idx[key].next())
#        except:
#            del idx[key]
#            pass
#
#print l
#print idx


"""
Get train/val/test splits
"""
#sys.path.append('../')
#from configs.train_config import TrainConfig
#
#cfg = TrainConfig().parse()
#feature_files = glob.glob(cfg.feature_root+'*sensors.npy')
#all_session = [os.path.basename(f).split('_')[0] for f in feature_files]
#assert len(all_session) == 137
#train_session = [session for session in all_session if session not in cfg.test_session]
#val_session = train_session[-8:]
#train_session = train_session[:-8]
#
#invalid_set = ['201704131012', '201710031436', '201703080946']    # only contain 1 event
#
#with open(cfg.DATA_ROOT+'train_session.txt', 'w') as fout:
#    for s in train_session:
#        if s not in invalid_set:
#            fout.write(s+'\n')
#
#with open(cfg.DATA_ROOT+'val_session.txt', 'w') as fout:
#    for s in val_session:
#        if s not in invalid_set:
#            fout.write(s+'\n')
#
#with open(cfg.DATA_ROOT+'test_session.txt', 'w') as fout:
#    for s in cfg.test_session:
#        if s not in invalid_set:
#            fout.write(s+'\n')


"""
Normalize canbus sensor data
        ○ [accel, steer angle, steer speed, vel, brake, left, right, yaw]
        ○ zero mean, unit std for accel, vel, brake
        ○ unit std for steer angle, steer speed, yaw (keep the sign meaningful)
        ○ -1, 1 for left and right
"""

#all_feats = []
#with open('/mnt/work/honda_100h/all_session.txt') as fin:
#    for line in fin:
#        session_id = line.strip()
#
#        feats = np.load('/mnt/work/honda_100h/features/'+session_id+'_sensors.npy')
#        all_feats.append(feats)
#
#all_feats = np.concatenate(all_feats, axis=0)
#mu = np.mean(all_feats, axis=0)
#std = np.std(all_feats, axis=0) + np.finfo(float).tiny
#
#with open('/mnt/work/honda_100h/all_session.txt') as fin:
#    for line in fin:
#        session_id = line.strip()
#
#        feats = np.load('/mnt/work/honda_100h/features/'+session_id+'_sensors.npy')
##        new_feats = (feats-mu) / std
#        new_feats = feats
#        new_feats[:,0] = (feats[:,0]-mu[0]) / std[0]
#        new_feats[:,3] = (feats[:,3]-mu[3]) / std[3]
#        new_feats[:,4] = (feats[:,4]-mu[4]) / std[4]
#
#        new_feats[:,1] = feats[:,1] / std[1]
#        new_feats[:,2] = feats[:,2] / std[2]
#        new_feats[:,7] = feats[:,7] / std[7]
#
#        new_feats[np.where(feats[:,5]==0)[0],5] = -1
#        new_feats[np.where(feats[:,6]==0)[0],6] = -1
#
#        np.save('/mnt/work/honda_100h/features/'+session_id+'_sensors_normalized.npy', new_feats)


"""
Read generated tfrecords and test whether they are correct
"""

#import tensorflow as tf
#import time
#tf_root ='/mnt/work/honda_100h/tfrecords/'
#
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#
#filename_queue = tf.train.string_input_producer([tf_root+"201706071021.tfrecords"])
#reader = tf.TFRecordReader()
#_, serialized_example = reader.read(filename_queue)
#
#context_features={'label': tf.FixedLenFeature([], tf.int64),
#                  'length': tf.FixedLenFeature([], tf.int64)}
#sequence_features={
#                   'sensors': tf.FixedLenSequenceFeature([8], tf.float32)}
##sequence_features={'resnet': tf.FixedLenSequenceFeature([98304], tf.float32),
##                   'sensors': tf.FixedLenSequenceFeature([8], tf.float32)}
#
#context, feature_lists = tf.parse_single_sequence_example(serialized_example,
#        context_features=context_features, sequence_features=sequence_features)
#
##c = tf.contrib.learn.run_n(context, n=3, feed_dict=None)
#start_time = time.time()
#s = tf.contrib.learn.run_n(feature_lists, n=10, feed_dict=None)
#duration = time.time() - start_time
#
#print ("Duration %.3f" % duration)


"""
Test new data generator pipeline with tfrecord
"""

#import tensorflow as tf
#import sys
#sys.path.append('../')
#from src.data_io import event_generator, session_generator
#from configs.base_config import BaseConfig
#import time
#
## --------------------------------------------------
#n_seg = 3
#def prepare_input(feat):
#    average_duration = feat.shape[0] // n_seg
#    if average_duration > 0:
#        offsets = np.multiply(range(n_seg), average_duration) + np.random.randint(average_duration, size=n_seg)
#    else:
#        raise NotImplementedError
#
#    return feat[offsets].astype('float32')
#
#feat_paths_ph = tf.placeholder(tf.string, shape=[None, 1])
#label_paths_ph = tf.placeholder(tf.string, shape=[None, 1])
#train_data = session_generator(feat_paths_ph, label_paths_ph, sess_per_batch=1, num_threads=1, shuffled=False, preprocess_func=prepare_input)
#train_sess_iterator = train_data.make_initializable_iterator()
#next_train = train_sess_iterator.get_next()
#
#feat_names = [('/mnt/work/honda_100h/features/201704141420.npy',)]
#label_names = [('/mnt/work/honda_100h/labels/201704141420_goal.pkl',)]
#
## --------------------------------------------------
#def prepare_input_tf(feat):
#    average_duration = tf.floordiv(tf.shape(feat)[0], n_seg)
#    offsets = tf.add(tf.multiply(tf.range(n_seg,dtype=tf.int32), average_duration),
#            tf.random_uniform(shape=(1,n_seg),maxval=average_duration,dtype=tf.int32))
#    return tf.gather_nd(feat, tf.reshape(offsets, [-1,1]))
#
#feat_dict = {'resnet': 98304}
#context_dict = {'label': 'int', 'length': 'int'}
#tf_paths_ph = tf.placeholder(tf.string, shape=[None])
#dataset = event_generator(tf_paths_ph, feat_dict, context_dict,
#        event_per_batch=151, num_threads=1, shuffled=False, preprocess_func=prepare_input_tf)
#sess_iterator = dataset.make_initializable_iterator()
#next_train2 = sess_iterator.get_next()
#
#filenames = ['/mnt/work/honda_100h/tfrecords/201704141420.tfrecords']
##filenames = ['/mnt/work/honda_100h/tfrecords/201704141420.tfrecords',
##             '/mnt/work/honda_100h/tfrecords/201703081617.tfrecords',
##             '/mnt/work/honda_100h/tfrecords/201703081653.tfrecords',
##             '/mnt/work/honda_100h/tfrecords/201703081723.tfrecords',
##             '/mnt/work/honda_100h/tfrecords/201703081749.tfrecords']
##cfg = BaseConfig().parse()
##filenames = [cfg.tfrecords_root+session_id+'.tfrecords' for session_id in cfg.train_session]
#
##np.random.shuffle(filenames)
#
## --------------------------------------------------
#def _get_context_feature(ctype):
#    if ctype == 'int':
#        return tf.FixedLenFeature([], tf.int64)
#    elif ctype == 'float':
#        return tf.FixedLenFeature([], tf.float32)
#    elif ctype == 'str':
#        return tf.FixedLenFeature([], tf.string)
#    else:
#        raise NotImplementedError
#
#reader = tf.TFRecordReader()
#filename_queue = tf.train.string_input_producer(filenames,num_epochs=1)
#_,serialized_example =reader.read(filename_queue)
#context_features={}
#for key, value in context_dict.items():
#    context_features[key] = _get_context_feature(value)
#
#sequence_features={}
#for key, value in feat_dict.items():
#    sequence_features[key] = tf.FixedLenSequenceFeature([value], tf.float32)
#
#cont, feat_lists = tf.parse_single_sequence_example(serialized_example,
#        context_features=context_features, sequence_features=sequence_features)
#
## --------------------------------------------------
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#with tf.Session() as sess:
#    for epoch in range(3):
#        sess.run(sess_iterator.initializer, feed_dict={tf_paths_ph: filenames})
#        sess.run(train_sess_iterator.initializer, feed_dict={feat_paths_ph: feat_names, label_paths_ph: label_names})
#
#        while True:
#            try:
#                start_time = time.time()
#                feat = np.load(feat_names[0][0])
#                duration_np = time.time() - start_time
##                print (feat.shape)
#
#                start_time = time.time()
#                s = tf.contrib.learn.run_n(feat_lists, n=151, feed_dict=None)
#                duration_tf = time.time() - start_time
##                print (len(s))
#
#                start_time = time.time()
#                eve, se, lab = sess.run(next_train)
#                duration_sess = time.time() - start_time
#
##                print (eve.shape)
#
#                start_time = time.time()
#                context, feature_lists = sess.run(next_train2)
#                duration_eve = time.time() - start_time
#                
##                print (feature_lists['resnet'].shape)
#
#                print ("Time_np: %.3f, Time_tf: %.3f, Time_sess: %.3f, Time_eve: %.3f" % (duration_np, duration_tf, duration_sess, duration_eve))
#
#            except tf.errors.OutOfRangeError:
#                print ("Epoch %d done" % (epoch+1))
#                break



"""
Test tensorflow version of prepare_input
"""

#import tensorflow as tf
#import numpy as np
#
#max_time = 15
#def rnn_prepare_input(feat):
#    """
#    feat -- feature sequence, [time_steps, n_h, n_w, n_input]
#    """
#
#    new_feat = np.zeros((max_time,)+feat.shape[1:], dtype='float32')
#    if feat.shape[0] > max_time:
#        new_feat = feat[:max_time]
#    else:
#        new_feat[:feat.shape[0]] = feat
#
#    return np.expand_dims(new_feat, 0)
#
#def rnn_prepare_input_tf(feat):
#
#    new_feat = tf.cond(tf.shape(feat)[0]>max_time, feat[:max_time], 
#            tf.pad(feat, tf.constant([[0, max_time-tf.shape(feat)[0]],[0,0]]), "CONSTANT"))
#    return tf.expand_dims(new_feat, 0)
#
#feat = np.asarray([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]])
#
#print ("Numpy results")
#print (rnn_prepare_input(feat))
#
#feat_ph = tf.placeholder(tf.int32, shape=[None,2])
#output = rnn_prepare_input_tf(feat_ph)
#
#print ("Tensorflow results")
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#with tf.Session() as sess:
#    new_feat = sess.run(output, feed_dict={feat_ph:feat})
#    print (new_feat)

#########################################

#
#n_seg = 3
#
#def prepare_input(feat):
#    average_duration = feat.shape[0] // n_seg
#    offsets = np.multiply(range(n_seg), average_duration) + np.random.randint(average_duration, size=n_seg)
#    return feat[offsets]
#
#def prepare_input_tf(feat):
#    average_duration = tf.floordiv(tf.shape(feat)[0], n_seg)
#    offsets = tf.add(tf.multiply(tf.range(n_seg,dtype=tf.int32), average_duration),
#            tf.random_uniform(shape=(1,n_seg),maxval=average_duration,dtype=tf.int32))
#
#    return tf.gather_nd(feat, tf.reshape(offsets, [-1,1]))
#
#feat = np.asarray([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]])
#
#print ("Numpy results")
#print (prepare_input(feat))
#
#feat_ph = tf.placeholder(tf.int32, shape=[None,2])
#output = prepare_input_tf(feat_ph)
#
#print ("Tensorflow results")
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#with tf.Session() as sess:
#    new_feat = sess.run(output, feed_dict={feat_ph:feat})
#    print (new_feat)




"""
Comparing Reading speed of Example and SequenceExample
"""

#import tensorflow as tf
#import time
#import pickle
#
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#
##session_id = '201703081617'
#session_id = '201704141420'
#
#start_time = time.time()
#feats = np.load('/mnt/work/honda_100h/features/'+session_id+'.npy')
#duration = time.time()- start_time
#print ("Duration: %.3f" % duration)
#label = pickle.load(open('/mnt/work/honda_100h/labels/'+session_id+'_goal.pkl', 'rb'))
#s = label['s']
#G = label['G']
#label = label['label']

#np.save('201704141420_float32.npy', feats.astype('float32'))

#writer = tf.python_io.TFRecordWriter(session_id+'_example.tfrecords')
#for i in range(feats.shape[0]):
#    d_feature = {}
#    d_feature['resnet'] = tf.train.Feature(float_list=tf.train.FloatList(value=feats[i].flatten()))
#    d_feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]]))
#    features = tf.train.Features(feature=d_feature)
#    example = tf.train.Example(features=features)
#    writer.write(example.SerializeToString())
#writer.close()

#writer = tf.python_io.TFRecordWriter('201704141420_seqexample.tfrecords')
#for i in range(len(label['G']))
#for i in range(feats.shape[0]):
#    d_feature = {}
#    d_feature['resnet'] = tf.train.Feature(float_list=tf.train.FloatList(value=feats[i].flatten()))
#    d_feature['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]]))
#    features = tf.train.Features(feature=d_feature)
#    example = tf.train.Example(features=features)
#    writer.write(example.SerializeToString())
#writer.close()

#reader = tf.TFRecordReader()
#filename_queue = tf.train.string_input_producer(["201704141420.tfrecords"],num_epochs=1)
#_,serialized_example =reader.read(filename_queue)
#batch = tf.train.batch(tensors=[serialized_example],batch_size=151)
#parsed_example = tf.parse_example(batch,features={'resnet':tf.FixedLenFeature([98304], tf.float32)})

#n_seg = 3
#def prepare_input(feat):
#    average_duration = feat.shape[0] // n_seg
#    if average_duration > 0:
#        offsets = np.multiply(range(n_seg), average_duration) + np.random.randint(average_duration, size=n_seg)
#    else:
#        raise NotImplementedError
#    return feat[offsets].astype('float32')
#
#def _input_parser(serialized_example):
#    parsed_example = tf.parse_single_example(serialized_example, features={'resnet':tf.FixedLenFeature([98304], tf.float32)})
#    return parsed_example
#
#tf_paths_ph = tf.placeholder(tf.string, shape=[None])
#dataset = tf.data.TFRecordDataset(tf_paths_ph)
#dataset = dataset.map(_input_parser, num_parallel_calls=1)
#dataset = dataset.batch(9000)
#iterator = dataset.make_initializable_iterator()
#features = iterator.get_next()
#
#sess = tf.Session()
#for i in range(5):
#    sess.run(iterator.initializer, feed_dict={tf_paths_ph: [session_id+'_example.tfrecords']})
#    start_time = time.time()
#    while True:
#        try:
#            output = sess.run(features, feed_dict=None)
#        except tf.errors.OutOfRangeError:
#            break
#    duration = time.time() - start_time
#    print (output['resnet'].dtype)
#
#    start_time = time.time()
#    #feats = np.load('/mnt/work/honda_100h/features/'+session_id+'.npy')
#    feats = np.load(session_id+'_float32.npy')
#    label = pickle.load(open('/mnt/work/honda_100h/labels/'+session_id+'_goal.pkl', 'rb'))
#    s = label['s']
#    G = label['G']
#    for i in range(len(G)):
#        if s[i+1]-s[i] > 5:
#            temp = feats[s[i]:s[i+1]]
#    print (feats.dtype)
#    duration2 = time.time() - start_time
#
#    print ("Duration: %.3f, Duration_np: %.3f" % (duration, duration2))



"""
Test tf.gather for slicing tensor
"""

#import tensorflow as tf
#
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#
#ph = tf.placeholder(tf.float32, shape=[None, 3])
#AB = tf.gather(ph, [0,1], axis=1)
#AC = tf.gather(ph, [0,2], axis=1)
#output = tf.concat((AB,AC), axis=0)
#
#with tf.Session() as sess:
#    ph_val, output_val = sess.run([ph, output], feed_dict={ph: np.random.rand(5,3)})
#
#print (ph_val)
#print (output_val)


"""
Test overlapping of two clusters by measuring intra_class similarity (distance)
"""

#import pickle
#
#def intra_sim(feat):
#    mu = np.mean(feat, axis=0)
#    dist = np.linalg.norm(feat-mu, axis=1)
#    return np.mean(dist)
#
#
#data = pickle.load(open('/mnt/work/honda_100h/results/kmeans_20180403-004256/train_data.pkl','rb'))
#feats = data['feats']
#labels = data['labels']
#
#NUM_CLUSTER=20
#
#origin = np.zeros((NUM_CLUSTER,NUM_CLUSTER), dtype='float32')    # half of sum of two original intra-class similarity
#joint = np.zeros((NUM_CLUSTER,NUM_CLUSTER), dtype='float32')    # intra-class similarity of merged cluster
#
#for i in range(NUM_CLUSTER):
#    for j in range(i+1, NUM_CLUSTER):
#        idx_i = np.where(labels==i)[0]
#        idx_j = np.where(labels==j)[0]
#
#        origin[i,j] = 0.5 * (intra_sim(feats[idx_i]) + intra_sim(feats[idx_j]))
#        joint[i,j] = intra_sim(np.concatenate((feats[idx_i], feats[idx_j]), axis=0))
#
#indicator = joint < origin
#
#print ("Origin:", origin)
#print ("Joint:", joint)
#print ("Indicator:", indicator)


"""
Test dcca_loss
"""

import tensorflow as tf
from sklearn.cross_decomposition import CCA
from scipy.stats.stats import pearsonr
sys.path.append('../src/')
from networks import dcca_loss

U = np.random.random_sample(1800).reshape(600,3)
V = np.random.random_sample(1800).reshape(600,3)
result = 0.0
for i in range(3):
    result += pearsonr(U[:,i], V[:,i])[0]
print ("Raw data results: ", result)

cca = CCA(n_components=3)
U_c, V_c = cca.fit_transform(U, V)
result = 0.0
for i in range(3):
    result += pearsonr(U_c[:,i], V_c[:,i])[0]
print ("Sklearn results: ", result)

X1 = tf.placeholder(tf.float32, shape=[None,3])
X2 = tf.placeholder(tf.float32, shape=[None,3])
corr = dcca_loss(X1, X2, K=3, rcov1=1e-4, rcov2=1e-4)
with tf.Session() as sess:
    correlation = sess.run(corr, feed_dict={X1: U, X2: V})
print ("dcca results:", -correlation)
