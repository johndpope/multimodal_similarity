"""
Extract features using pretrained model
Then perform clustering and obtain high-confidence points
"""

from datetime import datetime
import sys
import os
import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
import time

sys.path.append('../')
from configs.eval_config import EvalConfig
import networks
from utils import evaluate
from data_io import load_data_and_label, prepare_dataset
from preprocess.honda_labels import honda_labels2num, honda_num2labels


def main():

    cfg = EvalConfig().parse()
    print ("Evaluate the model: {}".format(os.path.basename(cfg.model_path)))
    np.random.seed(seed=cfg.seed)

    all_session = cfg.train_session
    all_set = prepare_dataset(cfg.feature_root, all_session, cfg.feat, cfg.label_root)

    n_input = cfg.feat_dim[cfg.feat]

    ########################### Extract features ###########################

    # load backbone model
    model = networks.Seq2seqTSN(n_seg=cfg.num_seg, n_input=n_input, emb_dim=cfg.emb_dim, reverse=cfg.reverse)

    # get the embedding
    input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, n_input])
    dropout_ph = tf.placeholder(tf.float32, shape=[])
    model.forward(input_ph, dropout_ph)
    if cfg.normalized:
        embedding = tf.nn.l2_normalize(model.hidden, axis=-1, epsilon=1e-10)
    else:
        embedding = model.hidden

    # Testing
    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        # load the model (note that model_path already contains snapshot number
        saver.restore(sess, cfg.model_path)

        eve_embeddings = []
        sessions = []
        eids_all = []
        for i, session in enumerate(all_set):
            session_id = os.path.basename(session[1]).split('_')[0]
            print ("{0} / {1}: {2}".format(i, len(all_set), session_id))

            eve_batch, _, boundary = load_data_and_label(session[0], session[1], model.prepare_input_test)
            for start, end in zip(range(0, eve_batch.shape[0], cfg.batch_size),
                                range(cfg.batch_size, eve_batch.shape[0]+cfg.batch_size, cfg.batch_size)):
                end = min(end, eve_batch.shape[0])
                emb = sess.run(embedding, feed_dict={input_ph: eve_batch[start:end],
                                                     dropout_ph: 1.0})
                eve_embeddings.append(emb)

            # for tracking data sources
            sessions.extend([session_id]*eve_batch.shape[0])
            eids_all.extend(boundary)

        eve_embeddings = np.concatenate(eve_embeddings, axis=0)

    print ("Feature extraction done!")


    ########################### Clustering ###########################

    NUM_CLUSTER = 20    # k for k-means
    NUM_HIGH = 100    # number of high-confidence points used

    kmeans = KMeans(n_clusters=NUM_CLUSTER, n_init=20)
    print ("Fitting clustering... {} points with dim {}".format(eve_embeddings.shape[0], eve_embeddings.shape[1]))
    start_time = time.time()
    kmeans.fit(eve_embeddings)
    duration = time.time() - start_time
    print ("Done. %.3f seconds used" % (duration))

    ################### Get high-confidence points ##########################

    cluster_idx = kmeans.predict(eve_embeddings)
    cluster_dist = kmeans.transform(eve_embeddings)

    feat = []
    label = []
    ses = []
    eids = []
    for i in range(NUM_CLUSTER):
        idx = np.where(cluster_idx==i)[0]
        dist = cluster_dist[idx, i]
        sorted_idx = np.argsort(dist)

        idx = idx[sorted_idx[:NUM_HIGH]]
        temp = eve_embeddings[idx]
        feat.append(temp)
        label.append(i * np.ones((temp.shape[0],1),dtype='int32'))
        for j in idx:
            ses.append(sessions[j])
            eids.append(eids_all[j])
        print ("Label {} with {} points".format(i, temp.shape[0]))

    feat = np.concatenate(feat, axis=0)
    label = np.concatenate(label, axis=0)

    #########################################################################

    # save results
    result_dir = os.path.join(os.path.dirname(cfg.model_path), 'kmeans_'+datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    pkl.dump(kmeans, open(os.path.join(result_dir, 'kmeans_model.pkl'), 'wb'))
    pkl.dump({'feats':feat, 'labels':label, 'sessions':ses, 'boundaries':eids}, 
            open(os.path.join(result_dir, 'train_data.pkl'),'wb'))

    ############################ Feature for validation #################################

    val_session = cfg.val_session
    val_set = prepare_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)

    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        # load the model (note that model_path already contains snapshot number
        saver.restore(sess, cfg.model_path)

        eve_embeddings = []
        sessions = []
        eids_all = []
        for i, session in enumerate(val_set):
            session_id = os.path.basename(session[1]).split('_')[0]
            print ("{0} / {1}: {2}".format(i, len(all_set), session_id))

            eve_batch, _, boundary = load_data_and_label(session[0], session[1], model.prepare_input_test)
            for start, end in zip(range(0, eve_batch.shape[0], cfg.batch_size),
                                range(cfg.batch_size, eve_batch.shape[0]+cfg.batch_size, cfg.batch_size)):
                end = min(end, eve_batch.shape[0])
                emb = sess.run(embedding, feed_dict={input_ph: eve_batch[start:end],
                                                     dropout_ph: 1.0})
                eve_embeddings.append(emb)

            # for tracking data sources
            sessions.extend([session_id]*eve_batch.shape[0])
            eids_all.extend(boundary)

        eve_embeddings = np.concatenate(eve_embeddings, axis=0)

    label = kmeans.predict(eve_embeddings)

    pkl.dump({'feats':eve_embeddings, 'labels':label, 'sessions':sessions, 'boundaries':eids_all}, 
            open(os.path.join(result_dir, 'val_data.pkl'),'wb'))

if __name__ == '__main__':
    main()

