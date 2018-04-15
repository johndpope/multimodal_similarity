"""
Train PairSim models for similarity prediction
"""

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


def random_pairs(lab, batch_size, num_negative=1, test=False):
    """
    num_negative -- control the ratio of postive : negative
    """

    idx_dict = {}
    for i, l in enumerate(lab):
        l = int(l)
        if l not in idx_dict:
            idx_dict[l] = [i]
        else:
            idx_dict[l].append(i)
    if test:
        random.seed(1)
    for key in idx_dict:
        random.shuffle(idx_dict[key])

    foreground_keys = list(idx_dict.keys())
    foreground_keys.remove(0)
    foreground_dict = {}
    for key in foreground_keys:
        foreground_dict[key] = itertools.permutations(idx_dict[key], 2)

    pair_idx = []
    label = []
    while len(pair_idx) < batch_size * 2:
        keys = list(foreground_dict.keys())
        if len(keys) == 0:
            break
        
        for key in keys:
            try:
                an_idx, pos_idx = foreground_dict[key].__next__()
            except:
                del foreground_dict[key]
                continue

            # apend pairs and their mirrors
            pair_idx.extend([an_idx, pos_idx, pos_idx, an_idx])
            label.extend([1,1])

            # randomly select negative pairs
            all_neg = np.where(lab!=key)[0]
            for i in range(num_negative):
                neg_idx = all_neg[np.random.randint(len(all_neg))]

                pair_idx.extend([an_idx, neg_idx, neg_idx, an_idx])
                label.extend([0,0])
    return pair_idx, label

def hard_pairs(lab, prob, threshold=0.9):
    """
    get the hard samples and re-train those samples
    reference: DeepReID: Deep Filter Pairing Neural Network for Person Re-Identification
    """

    pair_idx = []
    label = []

    # hard positives
    hard_pos = np.where(np.logical_and(lab, prob[:,0]>threshold))[0]
    for idx in hard_pos:
        pair_idx.extend([2*idx, 2*idx+1, 2*idx+1, 2*idx])
        label.extend([1, 1])
    
    # hard negatives
    hard_neg = np.where(np.logical_and(lab==0, prob[:,1]>threshold))[0]
    for idx in hard_neg:
        pair_idx.extend([2*idx, 2*idx+1, 2*idx+1, 2*idx])
        label.extend([0, 0])

    return pair_idx, label, len(hard_neg)+len(hard_pos)


def main():

    cfg = TrainConfig().parse()
    print (cfg.name)
    result_dir = os.path.join(cfg.result_root, 
            cfg.name+'_'+datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    utils.write_configure_to_file(cfg, result_dir)
    np.random.seed(seed=cfg.seed)

    # prepare dataset
    train_session = cfg.train_session
    train_set = prepare_dataset(cfg.feature_root, train_session, cfg.feat, cfg.label_root)
    batch_per_epoch = len(train_set)//cfg.sess_per_batch

    val_session = cfg.val_session[:3]
    val_set = prepare_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)

        # subtract global_step by 1 if needed (for hard negative mining, keep global_step unchanged)
        subtract_global_step_op = tf.assign(global_step, global_step-1)

        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        # load backbone model
        if cfg.network == "tsn":
            model_emb = networks.TSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "rtsn":
            model_emb = networks.RTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "convtsn":
            model_emb = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "convrtsn":
            model_emb = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        else:
            raise NotImplementedError

        model_ver = networks.PairSim(n_input=cfg.emb_dim)

        # get the embedding
        if cfg.feat == "sensors":
            input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None])
        elif cfg.feat == "resnet":
            input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
        dropout_ph = tf.placeholder(tf.float32, shape=[])
        label_ph = tf.placeholder(tf.int32, shape=[None])
        model_emb.forward(input_ph, dropout_ph)
        embedding = model_emb.hidden

        # split embedding into A and B
        emb_A, emb_B = tf.unstack(tf.reshape(embedding, [-1,2,cfg.emb_dim]), 2, 1)
        pairs = tf.stack([emb_A, emb_B], axis=1)

        model_ver.forward(pairs, dropout_ph)
        logits = model_ver.logits
        prob = model_ver.prob
        pred = tf.argmax(logits, -1)

        ver_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=logits))

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = ver_loss +  regularization_loss * cfg.lambda_l2

        tf.summary.scalar('learning_rate', lr_ph)
        train_op = utils.optimize(total_loss, global_step, cfg.optimizer,
                lr_ph, tf.global_variables())

        saver = tf.train.Saver(max_to_keep=10)

        summary_op = tf.summary.merge_all()

        # session iterator for session sampling
        feat_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        label_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        train_data = session_generator(feat_paths_ph, label_paths_ph, sess_per_batch=cfg.sess_per_batch, num_threads=2, shuffled=False, preprocess_func=model_emb.prepare_input)
        train_sess_iterator = train_data.make_initializable_iterator()
        next_train = train_sess_iterator.get_next()

        # prepare validation data
        val_sess = []
        val_feats = []
        val_labels = []
        val_boundaries = []
        for session in val_set:
            session_id = os.path.basename(session[1]).split('_')[0]
            eve_batch, lab_batch, boundary = load_data_and_label(session[0], session[1], model_emb.prepare_input_test)    # use prepare_input_test for testing time
            val_feats.append(eve_batch)
            val_labels.append(lab_batch)
            val_sess.extend([session_id]*eve_batch.shape[0])
            val_boundaries.extend(boundary)
        val_feats = np.concatenate(val_feats, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        # generate metadata.tsv for visualize embedding
        with open(os.path.join(result_dir, 'metadata_val.tsv'), 'w') as fout:
            fout.write('id\tlabel\tsession_id\tstart\tend\n')
            for i in range(len(val_sess)):
                fout.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(i, val_labels[i,0], val_sess[i],
                                            val_boundaries[i][0], val_boundaries[i][1]))

        val_idx, val_labels = random_pairs(val_labels, 1000000, test=True)
        val_feats = val_feats[val_idx]
        val_labels = np.asarray(val_labels, dtype='int32')
        print ("Shape of val_feats: ", val_feats.shape)


        # Start running the graph
        if cfg.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            # load pretrain model, if needed
            if cfg.model_path:
                print ("Restoring pretrained model: %s" % cfg.model_path)
                saver.restore(sess, cfg.model_path)

            ################## Training loop ##################
            epoch = -1
            while epoch < cfg.max_epochs-1:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // batch_per_epoch

                # learning rate schedule, reference: "In defense of Triplet Loss"
                if epoch < cfg.static_epochs:
                    learning_rate = cfg.learning_rate
                else:
                    learning_rate = cfg.learning_rate * \
                            0.001**((epoch-cfg.static_epochs)/(cfg.max_epochs-cfg.static_epochs))

                # prepare data for this epoch
                random.shuffle(train_set)

                feat_paths = [path[0] for path in train_set]
                label_paths = [path[1] for path in train_set]
                # reshape a list to list of list
                # interesting hacky code from: https://stackoverflow.com/questions/10124751/convert-a-flat-list-to-list-of-list-in-python
                feat_paths = list(zip(*[iter(feat_paths)]*cfg.sess_per_batch))
                label_paths = list(zip(*[iter(label_paths)]*cfg.sess_per_batch))

                sess.run(train_sess_iterator.initializer, feed_dict={feat_paths_ph: feat_paths,
                  label_paths_ph: label_paths})

                # for each epoch
                batch_count = 1
                while True:
                    try:
                        # Hierarchical sampling (same as fast rcnn)
                        start_time_select = time.time()

                        # First, sample sessions for a batch
                        eve, se, lab = sess.run(next_train)

                        select_time1 = time.time() - start_time_select

                        # select pairs for training
                        pair_idx, train_labels = random_pairs(lab, cfg.batch_size, cfg.num_negative)

                        train_input = eve[pair_idx]
                        train_labels = np.asarray(train_labels, dtype='int32')
                        select_time2 = time.time()-start_time_select-select_time1

                        start_time_train = time.time()
                        # perform training on the selected pairs
                        err, y_pred, y_prob, _, step, summ = sess.run(
                                [total_loss, pred, prob, train_op, global_step, summary_op],
                                feed_dict = {input_ph: train_input,
                                             label_ph: train_labels,
                                             dropout_ph: cfg.keep_prob,
                                             lr_ph: learning_rate})
                        acc = accuracy_score(train_labels, y_pred)

                        negative_count = 0
                        if epoch >= cfg.negative_epochs:
                            hard_idx, hard_labels, negative_count = hard_pairs(train_labels, y_prob, 0.5)
                            if negative_count > 0:
                                hard_input = train_input[hard_idx]
                                hard_labels = np.asarray(hard_labels, dtype='int32')

                                step = sess.run(subtract_global_step_op)
                                hard_err, y_pred, _, step = sess.run(
                                        [total_loss, pred, train_op, global_step],
                                        feed_dict = {input_ph: hard_input,
                                                    label_ph: hard_labels,
                                                    dropout_ph: cfg.keep_prob,
                                                    lr_ph: learning_rate})

                        train_time = time.time() - start_time_train

                        print ("%s\tEpoch: [%d][%d/%d]\tEvent num: %d\tSelect_time1: %.3f\tSelect_time2: %.3f\tTrain_time: %.3f\tLoss: %.4f" % \
                                (cfg.name, epoch+1, batch_count, batch_per_epoch, eve.shape[0], select_time1, select_time2, train_time, err))

                        summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err),
                            tf.Summary.Value(tag="acc", simple_value=acc),
                            tf.Summary.Value(tag="negative_count", simple_value=negative_count)])
                        summary_writer.add_summary(summary, step)
                        summary_writer.add_summary(summ, step)

                        batch_count += 1
                    
                    except tf.errors.OutOfRangeError:
                        print ("Epoch %d done!" % (epoch+1))
                        break

                # validation on val_set
                print ("Evaluating on validation set...")
                val_err, val_pred, val_prob = sess.run([total_loss, pred, prob], feed_dict={input_ph: val_feats, label_ph: val_labels, dropout_ph: 1.0})
                val_acc = accuracy_score(val_labels, val_pred)

                summary = tf.Summary(value=[tf.Summary.Value(tag="Valiation acc", simple_value=val_acc),
                                            tf.Summary.Value(tag="Validation loss", simple_value=val_err)])
                summary_writer.add_summary(summary, step)

                # save model
                saver.save(sess, os.path.join(result_dir, cfg.name+'.ckpt'), global_step=step)

        # print log for analysis
        with open(os.path.join(result_dir, 'val_results.txt'), 'w') as fout:
            fout.write("acc = %.4f\n" % val_acc)
            fout.write("label\tprob_0\tprob_1\tA_idx\tB_idx\n")
            for i in range(val_prob.shape[0]):
                fout.write("%d\t%.4f\t%.4f\t%d\t%d\n" % 
                        (val_labels[i], val_prob[i,0],val_prob[i,1], val_idx[2*i], val_idx[2*i+1]))

if __name__ == "__main__":
    main()
