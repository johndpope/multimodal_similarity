"""
Old version without using tfrecords pipeline
03/25/2018
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
from sklearn.metrics import average_precision_score
import glob

sys.path.append('../')
from configs.train_config import TrainConfig
from data_io import session_generator, load_data_and_label, prepare_dataset
import networks
import utils

def select_triplets_facenet(lab, all_dist, triplet_per_batch, alpha=0.2, num_negative=3):
    """
    Select the triplets for training
    1. Sample anchor-positive pair (try to balance imbalanced classes)
    2. Semi-hard negative mining used in facenet

    Arguments:
    lab -- array of labels, [N,]
    all_dist -- distance matrix
    triplet_per_batch -- int
    alpha -- float, margin
    num_negative -- number of negative samples per anchor-positive pairs
    """

    idx_dict = {}
    for i, l in enumerate(lab):
        l = int(l)
        if l not in idx_dict:
            idx_dict[l] = [i]
        else:
            idx_dict[l].append(i)
    for key in idx_dict:
        random.shuffle(idx_dict[key])

    # create iterators for each anchor-positive pair
    foreground_keys = [key for key in idx_dict.keys()]
    foreground_dict = {}
    for key in foreground_keys:
        foreground_dict[key] = itertools.permutations(idx_dict[key], 2)

    triplet_input_idx = []
    all_neg_count = []    # for monitoring active count
    while (len(triplet_input_idx)) < triplet_per_batch * 3:
        keys = list(foreground_dict.keys())
        if len(keys) == 0:
            break

        for key in keys:
            try:
                an_idx, pos_idx = foreground_dict[key].__next__()
            except:
                # remove the key to prevent infinite loop
                del foreground_dict[key]
                continue
            
            pos_dist = all_dist[an_idx, pos_idx]
            neg_dist = np.copy(all_dist[an_idx])    # important to make a copy, otherwise is reference
            neg_dist[idx_dict[key]] = np.NaN

            all_neg = np.where(np.logical_and(neg_dist-pos_dist < alpha,
                                            pos_dist < neg_dist))[0]
            all_neg_count.append(len(all_neg))

            # continue if no proper negtive sample 
            if len(all_neg) > 0:
                for i in range(min(len(all_neg), num_negative)):
                    neg_idx = all_neg[np.random.randint(len(all_neg))]

                    triplet_input_idx.extend([an_idx, pos_idx, neg_idx])

                    if len(triplet_input_idx) >= triplet_per_batch*3:
                        return triplet_input_idx, np.mean(all_neg_count) 

    if len(triplet_input_idx) > 0:
        return triplet_input_idx, np.mean(all_neg_count)
    else:
        return None, None

def select_triplets_random(lab, triplet_per_batch, num_negative=3):
    """
    Select the triplets for training
    1. Sample anchor-positive pair (try to balance imbalanced classes)
    2. Randomly selecting negative sample for each anchor-positive pair

    Arguments:
    lab -- array of labels, [N,]
    triplet_per_batch -- int
    num_negative -- number of negative samples per anchor-positive pairs
    """

    idx_dict = {}
    for i, l in enumerate(lab):
        l = int(l)
        if l not in idx_dict:
            idx_dict[l] = [i]
        else:
            idx_dict[l].append(i)
    for key in idx_dict:
        random.shuffle(idx_dict[key])

    # create iterators for each anchor-positive pair
    foreground_keys = [key for key in idx_dict.keys()]
    foreground_dict = {}
    for key in foreground_keys:
        foreground_dict[key] = itertools.permutations(idx_dict[key], 2)

    triplet_input_idx = []
    while (len(triplet_input_idx)) < triplet_per_batch * 3:
        keys = list(foreground_dict.keys())
        if len(keys) == 0:
            break

        for key in keys:
            all_neg = np.where(lab!=key)[0]
            try:
                an_idx, pos_idx = foreground_dict[key].__next__()
            except:
                # remove the key to prevent infinite loop
                del foreground_dict[key]
                continue
            
            # randomly sample negative for the anchor-positive pair
            for i in range(num_negative):
                neg_idx = all_neg[np.random.randint(len(all_neg))]

                triplet_input_idx.extend([an_idx, pos_idx, neg_idx])

    return triplet_input_idx

"""
Reference:
    FaceNet implementation:
    https://github.com/davidsandberg/facenet
"""
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
    att_train = np.load('/mnt/work/CUB_200_2011/data/att_train.npy')
    val_att = np.load('/mnt/work/CUB_200_2011/data/att_test.npy')
    label_train = np.load('/mnt/work/CUB_200_2011/data/label_train.npy')
    label_train -= 1    # make labels start from 0
    val_labels = np.load('/mnt/work/CUB_200_2011/data/label_test.npy')
    pdb.set_trace()

    class_idx_dict = {}
    for i, l in enumerate(label_train):
        l = int(l)
        if l not in class_idx_dict:
            class_idx_dict[l] = [i]
        else:
            class_idx_dict[l].append(i)
    C = len(list(class_idx_dict.keys()))

    val_triplet_idx = select_triplets_random(val_labels, 1000)

    # generate metadata.tsv for visualize embedding
    with open(os.path.join(result_dir, 'metadata_val.tsv'), 'w') as fout:
        for l in val_labels:
            fout.write('{}\n'.format(int(l)))


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        # load backbone model
        #model_emb = networks.CUBLayer(n_input=312, n_output=cfg.emb_dim)
        model_emb = networks.OutputLayer(n_input=312, n_output=cfg.emb_dim)
        model_ver = networks.PDDM(n_input=cfg.emb_dim)

        # get the embedding
        input_ph = tf.placeholder(tf.float32, shape=[None, 312])
        dropout_ph = tf.placeholder(tf.float32, shape=[])
        model_emb.forward(input_ph, dropout_ph)
        if cfg.normalized:
            embedding = tf.nn.l2_normalize(model_emb.logits, axis=-1, epsilon=1e-10)
        else:
            embedding = model_emb.logits

        # variable for visualizing the embeddings
        emb_var = tf.Variable([0.0], name='embeddings')
        set_emb = tf.assign(emb_var, embedding, validate_shape=False)

        # calculated for monitoring all-pair embedding distance
#        diffs = utils.all_diffs_tf(embedding, embedding)
#        all_dist = utils.cdist_tf(diffs)
#        tf.summary.histogram('embedding_dists', all_dist)

        # split embedding into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embedding, [-1,3,cfg.emb_dim]), 3, 1)
        metric_loss = networks.triplet_loss(anchor, positive, negative, cfg.alpha)

        model_ver.forward(tf.stack((anchor, positive), axis=1))
        pddm_ap = model_ver.prob[:, 0]
        model_ver.forward(tf.stack((anchor, negative), axis=1))
        pddm_an = model_ver.prob[:, 0]
        pddm_loss = tf.reduce_mean(tf.maximum(tf.add(tf.subtract(pddm_ap, pddm_an), 0.6), 0.0), 0)

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = pddm_loss + 0.5 * metric_loss + regularization_loss * cfg.lambda_l2

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
            for epoch in range(cfg.max_epochs):

                # learning rate schedule, reference: "In defense of Triplet Loss"
                if epoch < cfg.static_epochs:
                    learning_rate = cfg.learning_rate
                else:
                    learning_rate = cfg.learning_rate * \
                            0.001**((epoch-cfg.static_epochs)/(cfg.max_epochs-cfg.static_epochs))


                # sample images
                class_in_batch = set()
                idx_batch = np.array([], dtype=np.int32)
                while len(idx_batch) < cfg.batch_size:
                    sampled_class = np.random.choice(list(class_idx_dict.keys()))
                    if not sampled_class in class_in_batch:
                        class_in_batch.add(sampled_class)
                        subsample_size = np.random.choice(range(5, 11))
                        subsample = np.random.permutation(class_idx_dict[sampled_class])[:subsample_size]
                        idx_batch = np.append(idx_batch, subsample)
                idx_batch = idx_batch[:cfg.batch_size]

                feat_batch = att_train[idx_batch]
                lab_batch = label_train[idx_batch]

                # Get the similarity of all events
                sim_prob = np.zeros((feat_batch.shape[0], feat_batch.shape[0]), dtype='float32')*np.nan
                comb = list(itertools.combinations(range(feat_batch.shape[0]), 2))
                for start, end in zip(range(0, len(comb), cfg.batch_size),
                                    range(cfg.batch_size, len(comb)+cfg.batch_size, cfg.batch_size)):
                    end = min(end, len(comb))
                    comb_idx = []
                    for c in comb[start:end]:
                        comb_idx.extend([c[0], c[1], c[1]])
                    emb = sess.run(pddm_ap, feed_dict={input_ph: feat_batch[comb_idx], dropout_ph: 1.0})
                    for i in range(emb.shape[0]):
                        sim_prob[comb[start+i][0], comb[start+i][1]] = emb[i]
                        sim_prob[comb[start+i][1], comb[start+i][0]] = emb[i]

                triplet_input_idx, active_count = select_triplets_facenet(lab_batch,sim_prob,cfg.triplet_per_batch,cfg.alpha,num_negative=cfg.num_negative)

                if triplet_input_idx is not None:
                    triplet_input = feat_batch[triplet_input_idx]

                    # perform training on the selected triplets
                    err, _, step, summ = sess.run([total_loss, train_op, global_step, summary_op],
                                feed_dict = {input_ph: triplet_input,
                                            dropout_ph: cfg.keep_prob,
                                            lr_ph: learning_rate})

                    print ("%s\tEpoch: %d\tImages num: %d\tTriplet num: %d\tLoss %.4f" % \
                            (cfg.name, epoch+1, feat_batch.shape[0], triplet_input.shape[0]//3, err))

                    summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err),
                            tf.Summary.Value(tag="active_count", simple_value=active_count),
                            tf.Summary.Value(tag="images_num", simple_value=feat_batch.shape[0]),
                            tf.Summary.Value(tag="triplet_num", simple_value=triplet_input.shape[0]//3)])
                    summary_writer.add_summary(summary, step)
                    summary_writer.add_summary(summ, step)

                # validation on val_set
                if (epoch+1) % 100 == 0:
                    print ("Evaluating on validation set...")
                    val_err = sess.run(total_loss, feed_dict={input_ph: val_att[val_triplet_idx], dropout_ph: 1.0})

                    summary = tf.Summary(value=[tf.Summary.Value(tag="Valiation loss", simple_value=val_err),])
                    print ("Epoch: [%d]\tloss: %.4f" % (epoch+1,val_err))

                    if (epoch+1) % 1000 == 0:
                        val_embeddings, _ = sess.run([embedding,set_emb], feed_dict={input_ph: val_att, dropout_ph: 1.0})
                        mAP, mPrec, recall = utils.evaluate_simple(val_embeddings, val_labels)

                        val_sim_prob = np.zeros((val_att.shape[0], val_att.shape[0]), dtype='float32')*np.nan
                        val_comb = list(itertools.combinations(range(val_att.shape[0]), 2))
                        for start, end in zip(range(0, len(val_comb), cfg.batch_size),
                                            range(cfg.batch_size, len(val_comb)+cfg.batch_size, cfg.batch_size)):
                            end = min(end, len(val_comb))
                            comb_idx = []
                            for c in val_comb[start:end]:
                                comb_idx.extend([c[0], c[1], c[1]])
                            emb = sess.run(pddm_ap, feed_dict={input_ph: val_att[comb_idx], dropout_ph: 1.0})
                            for i in range(emb.shape[0]):
                                val_sim_prob[val_comb[start+i][0], val_comb[start+i][1]] = emb[i]
                                val_sim_prob[val_comb[start+i][1], val_comb[start+i][0]] = emb[i]
        
                        mAP_PDDM = 0.0
                        count = 0
                        for i in range(val_labels.shape[0]):
                            if val_labels[i] > 0:
                                temp_labels = np.delete(val_labels, i, 0)
                                temp = np.delete(val_sim_prob, i, 1)
                                mAP_PDDM += average_precision_score(np.squeeze(temp_labels==val_labels[i]),
                                                np.squeeze(1 - temp[i]))
                                count += 1
                        mAP_PDDM /= count

                        print ("Epoch: [%d]\tmAP: %.4f\trecall: %.4f\tmAP_PDDM: %.4f" % (epoch+1,mAP,recall,mAP_PDDM))
                        summary = tf.Summary(value=[tf.Summary.Value(tag="Valiation mAP", simple_value=mAP),
                                            tf.Summary.Value(tag="Validation Recall@1", simple_value=recall),
                                            tf.Summary.Value(tag="Validation PDDM_mAP", simple_value=mAP_PDDM),
                                            tf.Summary.Value(tag="Validation mPrec@0.5", simple_value=mPrec)])


                        # config for embedding visualization
                        config = projector.ProjectorConfig()
                        visual_embedding = config.embeddings.add()
                        visual_embedding.tensor_name = emb_var.name
                        visual_embedding.metadata_path = os.path.join(result_dir, 'metadata_val.tsv')
                        projector.visualize_embeddings(summary_writer, config)

                    summary_writer.add_summary(summary, step)


                    # save model
                    saver.save(sess, os.path.join(result_dir, cfg.name+'.ckpt'), global_step=step)

if __name__ == "__main__":
    main()
