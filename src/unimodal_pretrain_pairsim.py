"""
Pretrain unimodal similarity prediction on high-confidence points obtained from kmeans
"""

from datetime import datetime
import os
import time
import sys
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import random
import pdb
import glob
import functools
import pickle
import itertools
from sklearn.metrics import accuracy_score

sys.path.append('../')
from configs.train_config import TrainConfig
from data_io import session_generator, load_data_and_label, prepare_dataset
import networks
import utils

def enumerate_batch(labels, num_pos, phase):
    """
    generate indices for a batch
    the number of negative samples are controlled accroding to phase

    labels are orgainized as: [0,..0, 1..1, 2...2, ...] with same number of data for each classes

    reference: DeepReID: Deep Filter Pairing Neural Network for Person Re-Identification 
    """

    label_num = np.max(labels) + 1
    all_idx = np.transpose(np.arange(len(labels)).reshape(-1, len(labels)//label_num))
    labels = labels[np.random.permutation(labels.shape[0])]

    for start, end in zip(range(0, all_idx.shape[0], num_pos),
                        range(num_pos, all_idx.shape[0]+num_pos, num_pos)):
        end = min(end, all_idx.shape[0])

        idx = range(start, end)
        perm = list(itertools.permutations(idx, 2))

        A_idx = []
        B_idx = []
        for i in range(label_num):
            for p in perm:
                A_idx.append(all_idx[p[0], i])
                B_idx.append(all_idx[p[1], i])

            neg_num = int(phase * len(perm))
            neg_count = 0
            neg_label = list(range(label_num))
            neg_label.remove(i)
            while neg_count < neg_num:
                temp = np.random.randint(start,end)
                A_idx.append(all_idx[temp, i])
                B_idx.append(all_idx[temp, neg_label[np.random.randint(len(neg_label))]])
                neg_count += 1

        yield A_idx, B_idx

def prepare_val(labels):

    label_list = np.squeeze(labels).tolist()
    unique_label = set(label_list)

    A_idx = []
    B_idx = []
    for l in unique_label:
        idx = np.where(labels == l)[0]
        perm = list(itertools.permutations(idx, 2))

        # pair the first (highest confidence) data to all other positive data
        count = 0
        for p in perm:
            if not p[0] == idx[0]:
                break
            A_idx.append(p[0])
            B_idx.append(p[1])
            count += 1

        neg_idx = np.where(labels != l)[0]
        for i in range(count):
            A_idx.append(idx[0])
            B_idx.append(neg_idx[np.random.randint(len(neg_idx))])

    return A_idx, B_idx

    

def main():

    cfg = TrainConfig().parse()
    print (cfg.name)

    # use model_path to indicate directory of clustering results
    result_dir = os.path.join(cfg.model_path,
            cfg.name+'_'+datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    utils.write_configure_to_file(cfg, result_dir)
    np.random.seed(seed=cfg.seed)

     
    # load data
    train_data = pickle.load(open(os.path.join(cfg.model_path, 'train_data.pkl'), 'rb'))
    val_data = pickle.load(open(os.path.join(cfg.model_path, 'val_data.pkl'), 'rb'))

    val_A_idx, val_B_idx = prepare_val(val_data['labels'])
    val_input = np.concatenate([np.expand_dims(val_data['feats'][val_A_idx], axis=1),
                                np.expand_dims(val_data['feats'][val_B_idx], axis=1)], axis=1)
    val_label = (val_data['labels'][val_A_idx] == val_data['labels'][val_B_idx]).astype('int32')
    print ("Shape of validation data: ".format(val_input.shape))


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        # load backbone model
        input_ph = tf.placeholder(tf.float32, shape=[None, 2, cfg.emb_dim])
        dropout_ph = tf.placeholder(tf.float32, shape=[])
        model = networks.PairSim(n_input=cfg.emb_dim)
        model.forward(input_ph, dropout_ph)
        logits = model.logits
        prob = model.prob
        pred = tf.argmax(logits, -1)

        label_ph = tf.placeholder(tf.int32, shape=[None])
        CE_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=logits))
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = CE_loss + regularization_loss * cfg.lambda_l2

        tf.summary.scalar('learning_rate', lr_ph)
        train_op = utils.optimize(total_loss, global_step, cfg.optimizer,
                lr_ph, tf.global_variables())

        saver = tf.train.Saver(max_to_keep=10)

        summary_op = tf.summary.merge_all()

        # Start running the graph
        if cfg.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            ################## Training loop ##################
            epoch = -1
            while epoch < cfg.max_epochs-1:
                step = sess.run(global_step, feed_dict=None)

                # learning rate schedule, reference: "In defense of Triplet Loss"
                if epoch < cfg.static_epochs:
                    learning_rate = cfg.learning_rate
                else:
                    learning_rate = cfg.learning_rate * \
                            0.001**((epoch-cfg.static_epochs)/(cfg.max_epochs-cfg.static_epochs))

                # defin phase to control the number of negative sample
                if epoch < cfg.static_epochs:
                    phase = 1
                else:
                    phase = 1 + (epoch-cfg.static_epochs) / float((cfg.max_epochs-cfg.static_epochs)/5)

                # loop for batch
                # use cfg.batch_size to indicate num_pos chosen for a batch
                batch_count = 0
                for A_idx, B_idx in enumerate_batch(train_data['labels'], cfg.batch_size, phase):
                    batch_input = np.concatenate([np.expand_dims(train_data['feats'][A_idx], axis=1),
                                        np.expand_dims(train_data['feats'][B_idx], axis=1)], axis=1)
                    batch_label = (train_data['labels'][A_idx] == train_data['labels'][B_idx]).astype('int32')

                    start_time_train = time.time()
                    err, y_pred, _, step, summ = sess.run([total_loss, pred, train_op, global_step, summary_op],
                                    feed_dict = {input_ph: batch_input,
                                                dropout_ph: cfg.keep_prob,
                                                label_ph: np.squeeze(batch_label),
                                                lr_ph: learning_rate})

                    # calculate accuracy
                    acc = accuracy_score(batch_label, y_pred)

                    train_time = time.time() - start_time_train
                    print ("Epoch: [%d][%d/%d]\tTrain_time: %.3f\tLoss %.4f\tAcc: %.4f" % \
                                    (epoch+1, batch_label.sum(), batch_label.shape[0], train_time, err, acc))
                    batch_count += 1

                    summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err),
                                            tf.Summary.Value(tag="train_acc", simple_value=acc),
                                            tf.Summary.Value(tag="pos_ratio", simple_value=float(batch_label.sum())/batch_label.shape[0])])
                    summary_writer.add_summary(summary, step)
                    summary_writer.add_summary(summ, step)
                    
                print ("Epoch %d done!" % (epoch+1))
                epoch += 1

                # validation on val_set
                print ("Evaluating on validation set...")
                val_err, val_pred, val_prob = sess.run([total_loss, pred, prob], feed_dict={
                                            input_ph: val_input, dropout_ph: 1.0, label_ph: np.squeeze(val_label)})
        
                val_acc = accuracy_score(val_label, val_pred)
                summary = tf.Summary(value=[tf.Summary.Value(tag="Validation acc", simple_value=val_acc),
                                            tf.Summary.Value(tag="Validation loss", simple_value=val_err)])
                summary_writer.add_summary(summary, step)
        
                # save model
                saver.save(sess, os.path.join(result_dir, cfg.name+'.ckpt'), global_step=step)
        
        # print log for analysis
        with open(os.path.join(result_dir, 'val_results.txt'), 'w') as fout:
            fout.write("A_idx\tB_idx\tlabel\tprob_0\tprob_1\n")

            for i in range(val_prob.shape[0]):
                fout.write("%d\t%d\t%d\t%.4f\t%.4f\n" % 
                        (val_A_idx[i], val_B_idx[i], val_label[i], val_prob[i,0], val_prob[i,1]))

if __name__ == "__main__":
    main()
