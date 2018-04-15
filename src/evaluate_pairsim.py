
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


def select_triplets(lab, eve_embedding, triplet_per_batch, alpha=0.2, num_negative=3, metric="squaredeuclidean"):
    """
    Select the triplets for evaluation, some of them are simple, some of them are hard negative

    Arguments:
    eve -- array of event features, [N, n_seg, (dims)]
    lab -- array of labels, [N,]
    eve_embedding -- array of event embeddings, [N, emb_dim]
    triplet_per_batch -- int
    alpha -- float, margin
    num_negative -- number of negative samples per anchor-positive pairs
    metric -- metric to calculate distance
    """

    # get distance for all pairs
    all_diff = utils.all_diffs(eve_embedding, eve_embedding)
    all_dist = utils.cdist(all_diff, metric=metric)

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
    foreground_keys = [key for key in idx_dict.keys() if not key == 0]
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

            # hard ones
            all_neg = np.where(np.logical_and(neg_dist-pos_dist < alpha,
                                            pos_dist < neg_dist))[0]
            if len(all_neg) > 0:
                neg_idx = all_neg[np.random.randint(len(all_neg))]
                triplet_input_idx.extend([an_idx, pos_idx, neg_idx])

                # simple ones
                all_neg = np.where(neg_dist-pos_dist > alpha)[0]
                neg_idx = all_neg[np.random.randint(len(all_neg))]
                triplet_input_idx.extend([an_idx, pos_idx, neg_idx])


    if len(triplet_input_idx) > 0:
        return triplet_input_idx, np.mean(all_neg_count)
    else:
        return None, None


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
    val_session = cfg.val_session
    val_set = prepare_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        with tf.variable_scope("test"):
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
    
    
            # multitask loss (verification)
            #model_ver = networks.PairSim2(n_input=cfg.emb_dim)
            model_ver = networks.PairSim(n_input=cfg.emb_dim)

        var_list = {}
        for v in tf.global_variables():
            if v.op.name.startswith("test"):
                var_list[v.op.name.replace("test/","")] = v
        restore_saver = tf.train.Saver(var_list)

        # get the embedding
        if cfg.feat == "sensors":
            input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None])
        elif cfg.feat == "resnet":
            input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
        dropout_ph = tf.placeholder(tf.float32, shape=[])
        model_emb.forward(input_ph, dropout_ph)
        if cfg.normalized:
            embedding = tf.nn.l2_normalize(model_emb.hidden, axis=-1, epsilon=1e-10)
        else:
            embedding = model_emb.hidden

        # split embedding into anchor, positive and negative and calculate distance
        anchor, positive, negative = tf.unstack(tf.reshape(embedding, [-1,3,cfg.emb_dim]), 3, 1)
        dist = tf.concat([tf.reshape(utils.cdist_tf(anchor-positive), [-1,1]), 
                          tf.reshape(utils.cdist_tf(anchor-negative), [-1,1])], axis=1)

        # verification
        pos_pairs = tf.concat([tf.expand_dims(anchor,axis=1), tf.expand_dims(positive,axis=1)], axis=1)
        pos_label = tf.ones((tf.shape(pos_pairs)[0],), tf.int32)
        neg_pairs = tf.concat([tf.expand_dims(anchor,axis=1), tf.expand_dims(negative,axis=1)], axis=1)
        neg_label = tf.zeros((tf.shape(neg_pairs)[0],), tf.int32)

        ver_pairs = tf.concat([pos_pairs, neg_pairs], axis=0)
        ver_label = tf.concat([pos_label, neg_label], axis=0)

        model_ver.forward(ver_pairs, dropout_ph)
        prob = tf.reshape(model_ver.prob[:, 1], (-1,1))
        sim = tf.concat([prob[:tf.shape(pos_pairs)[0]], prob[tf.shape(pos_pairs)[0]:]], axis=1)
        pred = tf.argmax(model_ver.logits, -1)


        # Start running the graph
        if cfg.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            print ("Restoring pretrained model: %s" % cfg.model_path)
            restore_saver.restore(sess, cfg.model_path)

            fout = open(os.path.join(os.path.dirname(cfg.model_path), 'val_pairsim.txt'), 'w')
            for i, session in enumerate(val_set):
                session_id = os.path.basename(session[1]).split('_')[0]
                print ("{0} / {1}: {2}".format(i, len(val_set), session_id))

                eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], model_emb.prepare_input_test)    # use prepare_input_test for testing time

                emb = sess.run(embedding, feed_dict={input_ph: eve_batch, dropout_ph: 1.0})

                triplet_per_batch = 10
                triplet_input_idx, negative_count = select_triplets(lab_batch,emb,triplet_per_batch,0.2)

                triplet_input = eve_batch[triplet_input_idx]
                dist_batch, sim_batch, pred_batch = sess.run([dist, sim, pred],
                                                        feed_dict={input_ph: triplet_input, dropout_ph: 1.0})

                pdb.set_trace()
                batch_label = np.hstack((np.ones((triplet_input.shape[0]//3,),dtype='int32'),
                                        np.zeros((triplet_input.shape[0]//3,),dtype='int32')))
                acc = accuracy_score(batch_label, pred_batch)
                fout.write("{}: acc = {}\n".format(session_id, acc))
                for i in range(dist_batch.shape[0]):
                    fout.write("{}\t{}\t{}\t{}\n".format(dist_batch[i,0], dist_batch[i,1], sim_batch[i,0], sim_batch[i,1]))
            fout.close()


if __name__ == "__main__":
    main()
