"""
Clustering for unimodal features
"""

from datetime import datetime
import sys
import os
import numpy as np
import pickle as pkl
import pdb
from sklearn.cluster import KMeans
import time

sys.path.append('../')
from configs.eval_config import EvalConfig
import networks
from utils import evaluate, mean_pool_input, max_pool_input
from data_io import load_data_and_label, prepare_dataset
from preprocess.honda_labels import honda_labels2num, honda_num2labels

NUM_CLUSTER = 20    # k for k-means
NUM_HIGH = 100    # number of high-confidence points used

def main():

    cfg = EvalConfig().parse()
    np.random.seed(seed=cfg.seed)

    train_session = cfg.train_session
    train_set = prepare_dataset(cfg.feature_root, train_session, cfg.feat, cfg.label_root)


    print ("Loading data...")
    eve_embeddings = []
    sessions = []
    eids_all = []
    for i, session in enumerate(train_set):
        session_id = os.path.basename(session[1]).split('_')[0]

        eve_batch, _, boundary = load_data_and_label(session[0], session[1], mean_pool_input)     # FIXME: temporally use mean 

        eve_embeddings.append(eve_batch)
        # for tracking data sources
        sessions.extend([session_id]*eve_batch.shape[0])
        eids_all.extend(boundary)

    eve_embeddings = np.concatenate(eve_embeddings, axis=0)

    ################### Fit k-means ##########################

    model = KMeans(n_clusters=NUM_CLUSTER, n_init=20)
    print ("Fitting clustering... {} points with dim {}".format(eve_embeddings.shape[0], eve_embeddings.shape[1]))
    start_time = time.time()
    model.fit(eve_embeddings)
    duration = time.time() - start_time
    print ("Done. %.3f seconds used" % (duration))

    ################### Get high-confidence points ##########################

    cluster_idx = model.predict(eve_embeddings)
    cluster_dist = model.transform(eve_embeddings)

    feat = []
    label = []
    sess = []
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
            sess.append(sessions[j])
            eids.append(eids_all[j])

        print ("Label {} with {} points".format(i, temp.shape[0]))

    feat = np.concatenate(feat, axis=0)
    label = np.concatenate(label, axis=0)

    #########################################################################

    # save results
    result_dir = os.path.join(cfg.result_root, 'kmeans_'+datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    pkl.dump(model, open(os.path.join(result_dir, 'kmeans_model.pkl'), 'wb'))
    pkl.dump({'feats':feat, 'labels':label, 'sessions':sess, 'boundaries':eids}, 
            open(os.path.join(result_dir, 'train_data.pkl'),'wb'))


    ###################### Same things for validation data ##################

    VAL_NUM_HIGH = 10

    print ("Working on validation data...")
    val_session = cfg.val_session
    val_set = prepare_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)
    eve_embeddings = []
    sessions = []
    eids_all = []
    for i, session in enumerate(val_set):
        session_id = os.path.basename(session[1]).split('_')[0]
        eve_batch, _, boundary = load_data_and_label(session[0], session[1], mean_pool_input)     # FIXME: temporally use mean 

        eve_embeddings.append(eve_batch)
        # for tracking data sources
        sessions.extend([session_id]*eve_batch.shape[0])
        eids_all.extend(boundary)

    eve_embeddings = np.concatenate(eve_embeddings, axis=0)

    cluster_idx = model.predict(eve_embeddings)
    cluster_dist = model.transform(eve_embeddings)

    feat = []
    label = []
    sess = []
    eids = []
    for i in range(NUM_CLUSTER):
        idx = np.where(cluster_idx==i)[0]
        dist = cluster_dist[idx, i]
        sorted_idx = np.argsort(dist)

        idx = idx[sorted_idx[:VAL_NUM_HIGH]]
        temp = eve_embeddings[idx]
        feat.append(temp)
        label.append(i * np.ones((temp.shape[0],1),dtype='int32'))
        for j in idx:
            sess.append(sessions[j])
            eids.append(eids_all[j])

    feat = np.concatenate(feat, axis=0)
    label = np.concatenate(label, axis=0)

    pkl.dump({'feats':feat, 'labels':label, 'sessions':sess, 'boundaries':eids}, 
            open(os.path.join(result_dir, 'val_data.pkl'),'wb'))

if __name__ == '__main__':
    main()

