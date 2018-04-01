import numpy as np
import pickle
import tensorflow as tf
import pdb

session_file = '/mnt/work/honda_100h/all_session.txt'
label_root = '/mnt/work/honda_100h/labels/'
feature_root = '/mnt/work/honda_100h/features/'
tf_root = '/mnt/work/honda_100h/tfrecords2/'

MIN_LENGTH = 5
MAX_LENGTH = 90

with open(session_file, 'r') as fin:
    session_count = 0
    for line in fin:
        session_id = line.strip()

        # session label
        label = pickle.load(open(label_root+session_id+'_goal.pkl','rb'))

        ########## collect features here ##############
        all_feats = []
        names = []

        # resnet feature
        feats = np.load(feature_root+session_id+'.npy').astype('float32')
        all_feats.append(feats)
        names.append('resnet')

        # sensor feature (normalized)
        feats = np.load(feature_root+session_id+'_sensors_normalized.npy').astype('float32')
        all_feats.append(feats)
        names.append('sensors')


        ##############################################

        # Loop through labels and build SequenceExample protobuff
        count = 0
        for i in range(len(label['G'])):
            print ("{}: {}, {} / {}".format(session_count, session_id, i+1, len(label['G'])))

            length = label['s'][i+1] - label['s'][i]
            if length > MIN_LENGTH:    # ignore short (background) clips
                writer = tf.python_io.TFRecordWriter(tf_root+session_id+'_'+str(count)+'.tfrecords')

                length = min(length, MAX_LENGTH)
                lab = label['G'][i]

                # define feature_lists
                feature_list = {}
                for j in range(len(all_feats)):
                    feats = all_feats[j]
                    event = feats[label['s'][i] : label['s'][i]+length]

                    feature = []    # tf.train.Feature
                    for k in range(event.shape[0]):
                        feature.append(tf.train.Feature(float_list=tf.train.FloatList(value=event[k].flatten())))

                    feature_list[names[j]] = tf.train.FeatureList(feature=feature)
                feature_lists = tf.train.FeatureLists(feature_list=feature_list)

                # define context
                context = tf.train.Features(feature={'label':
                                        tf.train.Feature(int64_list=tf.train.Int64List(value=[lab])),
                                        'length':
                                        tf.train.Feature(int64_list=tf.train.Int64List(value=[length])),
                                        'session_id':
                                        tf.train.Feature(bytes_list=tf.train.BytesList(value=[session_id.encode('utf-8')])),
                                        'event_id':
                                        tf.train.Feature(int64_list=tf.train.Int64List(value=[count]))})

                # generate one Sequence Example
                example = tf.train.SequenceExample(
                        context = context,
                        feature_lists = feature_lists)

                # write to file
                writer.write(example.SerializeToString())
                writer.close()

                count += 1

        session_count += 1
