
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
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        with tf.variable_scope("modality_sensors"):
            sensors_emb_dim = 32
            model_emb_sensors = networks.RTSN(n_seg=cfg.num_seg, emb_dim=sensors_emb_dim)
            model_pairsim_sensors = networks.PairSim(n_input=sensors_emb_dim)

            input_sensors_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, 8])
            dropout_ph = tf.placeholder(tf.float32, shape=[])
            model_emb_sensors.forward(input_sensors_ph, dropout_ph)

            var_list = {}
            for v in tf.global_variables():
                if v.op.name.startswith("modality_sensors"):
                    var_list[v.op.name.replace("modality_sensors/","")] = v
            restore_saver_sensors = tf.train.Saver(var_list)

        # Sensors branch
        emb_sensors = model_emb_sensors.hidden
        A_sensors = emb_sensors[:(tf.shape(emb_sensors)[0]//2)]
        B_sensors = emb_sensors[(tf.shape(emb_sensors)[0]//2):]
        AB_pairs_sensors = tf.stack([A_sensors, B_sensors], axis=1)
        model_pairsim_sensors.forward(AB_pairs_sensors, dropout_ph)
        prob_sensors = model_pairsim_sensors.prob

        # prepare validation data
        val_sess = []
        val_feats = []
        val_labels = []
        val_boundaries = []
        for session in val_set:
            session_id = os.path.basename(session[1]).split('_')[0]
            eve_batch, lab_batch, boundary = load_data_and_label(session[0], session[-1], model_emb_sensors.prepare_input_test)    # use prepare_input_test for testing time
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
            restore_saver_sensors.restore(sess, cfg.model_path)

            fout = open(os.path.join(os.path.dirname(cfg.model_path), 'val_inconsistent.txt'), 'w')
            fout.write('id_A\tid_B\tlabel_A\tlabel_B\tprob_0\tprob_1\n')
            for i in range(val_feats.shape[0]):
                print ("%d/%d" % (i,val_feats.shape[0]))
                if val_labels[i] == 0:
                    continue
                A_input = np.tile(val_feats[i], (val_feats.shape[0],1,1))
                AB_input = np.vstack((A_input, val_feats))

                temp_prob = sess.run(prob_sensors, feed_dict={input_sensors_ph: AB_input, dropout_ph:1.0})

                for j in range(temp_prob.shape[0]):
                    if val_labels[i] == val_labels[j] and temp_prob[j, 0]>0.95:
                        fout.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(i,j,val_labels[i,0],val_labels[j,0],temp_prob[j,0],temp_prob[j,1]))
                    elif val_labels[i] != val_labels[j] and temp_prob[j,1] > 0.95:
                        fout.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(i,j,val_labels[i,0],val_labels[j,0],temp_prob[j,0],temp_prob[j,1]))
            fout.close()


if __name__ == "__main__":
    main()
