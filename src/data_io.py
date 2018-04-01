import os
import numpy as np
import pickle as pkl
import tensorflow as tf
import random
import pdb

def load_data_and_label(feat_path, label_path, preprocess_func=None):
    """
    Load one session (data + label)
    """

    MIN_LENGTH = 5
    MAX_LENGTH = 90

    if preprocess_func is None:
        # identity function
        preprocess_func = lambda x: x

    feats = np.load(feat_path, 'r')
    label = pkl.load(open(label_path, 'rb'))

    events = []
    labels = []
    for i in range(len(label['G'])):
        length = label['s'][i+1] - label['s'][i]
        if length > MIN_LENGTH:    # ignore short (background) clips
            length = min(length, MAX_LENGTH)
            events.append(np.expand_dims(preprocess_func(feats[label['s'][i] : label['s'][i]+length]), 0))
            labels.append(label['G'][i])

    events = np.concatenate(events, axis=0).astype('float32')
    labels = np.asarray(labels).reshape(-1,1)

    return events, labels

def event_generator(tf_paths, feat_dict, context_dict, event_per_batch, num_threads=2, shuffled=True, preprocess_func=None):
    """
    Generator iterator of sesssions

    feat_paths -- placeholder for tfrecord paths
    feat_dict -- feature dictionary for parsing feature_lists, e.g. {'resnet': 98304, 'sensors':8}
    context_dict -- dictionary for parsing context, e.g. {'label': 'int', 'length': 'int'}
    preprocess_func -- preprocessing function, if needed
    """

    dataset = tf.data.TFRecordDataset(tf_paths)
    
    def _get_context_feature(ctype):
        if ctype == 'int':
            return tf.FixedLenFeature([], tf.int64)
        elif ctype == 'float':
            return tf.FixedLenFeature([], tf.float32)
        elif ctype == 'str':
            return tf.FixedLenFeature([], tf.string)
        else:
            raise NotImplementedError
    
    def _input_parser(serialized_example):
        context_features={}
        for key, value in context_dict.items():
            context_features[key] = _get_context_feature(value)

        sequence_features={}
        for key, value in feat_dict.items():
            sequence_features[key] = tf.FixedLenSequenceFeature([value], tf.float32)

        context, feature_lists = tf.parse_single_sequence_example(serialized_example,
                context_features=context_features, sequence_features=sequence_features)

        for key, value in feature_lists.items():
            feature_lists[key] = preprocess_func(value)

        return context, feature_lists

    dataset = dataset.map(_input_parser,
                        num_parallel_calls = num_threads)
#    dataset = dataset.apply(tf.contrib.data.map_and_batch(_input_parser,
#                        batch_size=event_per_batch,
#                        num_parallel_batches=2))

    if shuffled:
        dataset = dataset.shuffle(buffer_size=600)

#    padded_shapes = ({key:[] for key in context_dict},
#                    {key:[None, value] for key,value in feat_dict.items()})
#    dataset = dataset.padded_batch(event_per_batch, padded_shapes=padded_shapes)
    dataset = dataset.batch(event_per_batch)
    dataset = dataset.prefetch(1)    # test different values
    
    return dataset


def session_generator(feat_paths, label_paths, sess_per_batch, num_threads=2, shuffled=True, preprocess_func=None):
    """
    Generator iterator of sesssions (Old version without using tfrecords)

    feat_paths -- placeholder for feature paths
    label_paths -- placeholder for label_paths
    preprocess_func -- preprocessing function, if needed
    """

    dataset = tf.data.Dataset.from_tensor_slices((feat_paths, label_paths))
    
    def _input_parser(feat_path, label_path):
        events = []
        sess = []
        labels = []
        for s in range(sess_per_batch):
            #### very important to have decode() for tf r1.6 ####
            feats = np.load(feat_path[s].decode(), 'r')
            label = pkl.load(open(label_path[s].decode(), 'rb'))

            for i in range(len(label['G'])):
                # ignore short (background) clips
                if label['s'][i+1] - label['s'][i] > 5:
                    events.append(np.expand_dims(preprocess_func(feats[label['s'][i] : label['s'][i+1]]), 0))
                    sess.append(os.path.basename(feat_path[s].decode()).split('.')[0])
                    labels.append(label['G'][i])

        events = np.concatenate(events, axis=0)
        sess = np.asarray(sess).reshape(-1,1)
        labels = np.asarray(labels).reshape(-1,1)

        return events, sess, labels

    # fix doc issue according to https://github.com/tensorflow/tensorflow/issues/11786
    dataset = dataset.map(lambda feat_path, label_path:
                        tuple(tf.py_func(_input_parser, [feat_path, label_path],
                            [tf.float32, tf.string, tf.int32])),
                        num_parallel_calls = num_threads)
    dataset = dataset.prefetch(1)
    
    return dataset

