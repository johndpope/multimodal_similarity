
from datetime import datetime
import os
import time
import sys
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import itertools
import random
import pdb
from six import iteritems
import glob
from sklearn.metrics import accuracy_score

sys.path.append('../')
from configs.train_config import TrainConfig
from data_io import session_generator, load_data_and_label, prepare_dataset
import networks
import utils

def main():

    cfg = TrainConfig().parse()
    print (cfg.name)
    np.random.seed(seed=cfg.seed)

    # prepare dataset
    val_session = cfg.val_session
    val_set = prepare_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)

        # load backbone model
        if cfg.network == "tsn":
            model_emb = networks.TSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "rtsn":
            model_emb = networks.RTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim, n_input=cfg.n_input)
        elif cfg.network == "convtsn":
            model_emb = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "convrtsn":
            model_emb = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim, n_h=cfg.n_h, n_w=cfg.n_w, n_C=cfg.n_C, n_input=cfg.n_input)
        elif cfg.network == "convbirtsn":
            model_emb = networks.ConvBiRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        else:
            raise NotImplementedError
        model_ver = networks.PDDM(n_input=cfg.emb_dim)

        # get the embedding
        if cfg.feat == "sensors" or cfg.feat == "segment":
            input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None])
        elif cfg.feat == "resnet" or cfg.feat == "segment_down":
            input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
        dropout_ph = tf.placeholder(tf.float32, shape=[])
        model_emb.forward(input_ph, dropout_ph)
        if cfg.normalized:
            embedding = tf.nn.l2_normalize(model_emb.hidden, axis=-1, epsilon=1e-10)
        else:
            embedding = model_emb.hidden

        # split the embedding
        emb_A = embedding[:(tf.shape(embedding)[0]//2)]
        emb_B = embedding[(tf.shape(embedding)[0]//2):]
        model_ver.forward(tf.stack((emb_A, emb_B), axis=1))
        pddm = model_ver.prob

        restore_saver = tf.train.Saver()

        # prepare validation data
        val_sess = []
        val_feats = []
        val_labels = []
        val_boundaries = []
        for session in val_set:
            session_id = os.path.basename(session[1]).split('_')[0]
            eve_batch, lab_batch, boundary = load_data_and_label(session[0], session[-1], model_emb.prepare_input_test)    # use prepare_input_test for testing time
            val_feats.append(eve_batch)
            val_labels.append(lab_batch)
            val_sess.extend([session_id]*eve_batch.shape[0])
            val_boundaries.extend(boundary)

        val_feats = np.concatenate(val_feats, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        print ("Shape of val_feats: ", val_feats.shape)

        # Start running the graph
        if cfg.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            print ("Restoring pretrained model: %s" % cfg.model_path)
            restore_saver.restore(sess, cfg.model_path)


            fout_fp = open(os.path.join(os.path.dirname(cfg.model_path), 'val_fp.txt'), 'w')
            fout_fn = open(os.path.join(os.path.dirname(cfg.model_path), 'val_fn.txt'), 'w')
            fout_fp.write('id_A\tid_B\tlabel_A\tlabel_B\tprob_0\tprob_1\n')
            fout_fn.write('id_A\tid_B\tlabel_A\tlabel_B\tprob_0\tprob_1\n')
            count = 0
            count_high = 0    # high confidence (0.9)
            count_fp = 0
            count_fn = 0

            for i in range(val_feats.shape[0]):
                print ("%d/%d" % (i,val_feats.shape[0]))
                if val_labels[i] == 0:
                    continue
                A_input = np.tile(val_feats[i], (val_feats.shape[0]-i,1,1))
                AB_input = np.vstack((A_input, val_feats[i:]))    # concatenate along axis 0
                temp_prob = sess.run(pddm, feed_dict={input_ph: AB_input, dropout_ph:1.0})
                count += temp_prob.shape[0]

                threshold = 0.8
                for j in range(temp_prob.shape[0]):
                    if temp_prob[j, 0] > threshold or temp_prob[j, 1] > threshold:
                        count_high += 1
                        if val_labels[i] == val_labels[i+j] and temp_prob[j, 0]>threshold:
                            count_fn += 1
                            fout_fn.write("{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\n".format(i,i+j,val_labels[i,0],val_labels[i+j,0],temp_prob[j,0],temp_prob[j,1]))
                        elif val_labels[i] != val_labels[i+j] and temp_prob[j,1] > threshold:
                            count_fp += 1
                            fout_fp.write("{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\n".format(i,i+j,val_labels[i,0],val_labels[i+j,0],temp_prob[j,0],temp_prob[j,1]))
            fout_fp.close()
            fout_fn.close()

            print ("High confidence (%f) pairs ratio: %.4f" % (threshold, float(count_high)/count))
            print ("Consistent pairs ratio: %.4f" % (float(count_high-count_fp-count_fn)/count_high))
            print ("False positive pairs ratio: %.4f" % (float(count_fp)/count_high))
            print ("False negative pairs ratio: %.4f" % (float(count_fn)/count_high))


if __name__ == "__main__":
    main()
