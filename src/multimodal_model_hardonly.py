"""
Multimodal similarity learning
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
from data_io import multimodal_session_generator, load_data_and_label, prepare_multimodal_dataset
import networks
import utils

def select_triplets_mul_hard(triplet_input_idx, lab, sim_prob, triplet_per_batch, triplet_per_event=2, threshold_up=0.65, threshold_down=0.35):
    triplet_selected = []
    for i in range(0, len(triplet_input_idx), 3):
        triplet = (triplet_input_idx[i], triplet_input_idx[i+1], triplet_input_idx[i+2])
        if not triplet in triplet_selected:
            triplet_selected.append(triplet)

    triplet_count = len(triplet_selected)
    adjacency = np.equal(lab, lab.T)

    for i in np.random.permutation(lab.shape[0]):
        if lab[i] > 0:    # for foreground event
            hard_pos = np.where(np.logical_and(adjacency[i], sim_prob[i]<threshold_down))[0]
            hard_neg = np.where(np.logical_and(adjacency[i]==False, sim_prob[i]>threshold_up))[0]

            if len(hard_pos) == 0:
                all_pos = np.where(adjacency[i])[0]
                if len(all_pos) == 1:
                    continue
                sim = sim_prob[i, all_pos]
                hard_pos = np.array([all_pos[np.nanargmin(sim)]], dtype='int32')
            if len(hard_neg) == 0:
                all_neg = np.where(adjacency[i]==False)[0]
                if len(all_neg) == 1:
                    continue
                sim = sim_prob[i, all_neg]
                hard_neg = np.array([all_neg[np.nanargmax(sim)]], dtype='int32')

            hard_comb = [(hp, hn) for hn in hard_neg for hp in hard_pos]
            random.shuffle(hard_comb)
            for count in range(min(triplet_per_event, len(hard_comb))):
                hp, hn = hard_comb[count]
                triplet = (i, hp, hn)
                if not triplet in triplet_selected:
                    triplet_selected.append(triplet)
                        
        if len(triplet_selected)-triplet_count >= triplet_per_batch:
            break

    triplet_selected = triplet_selected[:(triplet_count + triplet_per_batch)]
    mul_count = len(triplet_selected) - triplet_count

    triplet_input_idx = [idx for triplet in triplet_selected for idx in triplet]

    return triplet_input_idx, triplet_count, mul_count

def select_triplets_mul(triplet_input_idx, lab, sim_prob, triplet_per_batch, triplet_per_event=2, threshold_up=0.65, threshold_down=0.35):

    idx_dict = {}
    for i, l in enumerate(lab):
        l = int(l)
        if l not in idx_dict:
            idx_dict[l] = [i]
        else:
            idx_dict[l].append(i)

    triplet_selected = []
    for i in range(0, len(triplet_input_idx), 3):
        triplet = (triplet_input_idx[i], triplet_input_idx[i+1], triplet_input_idx[i+2])
        if not triplet in triplet_selected:
            triplet_selected.append(triplet)

    triplet_count = len(triplet_selected)
    adjacency = np.equal(lab, lab.T)

    struct_selected = []
    for i in np.random.permutation(lab.shape[0]):
        if lab[i] > 0:    # for foreground event
            
            ################## hard sample mining ####################
            hard_pos = np.where(np.logical_and(adjacency[i], sim_prob[i]<threshold_down))[0]
            hard_neg = np.where(np.logical_and(adjacency[i]==False, sim_prob[i]>threshold_up))[0]

            if len(hard_pos) == 0:
                all_pos = np.where(adjacency[i])[0]
                sim = sim_prob[i, all_pos]
                hard_pos = np.array([all_pos[np.argmax(sim)]], dtype='int32')
            if len(hard_neg) == 0:
                all_neg = np.where(adjacency[i]==False)[0]
                sim = sim_prob[i, all_neg]
                hard_neg = np.array([all_neg[np.argmin(sim)]], dtype='int32')

            hard_comb = [(hp, hn) for hn in hard_neg for hp in hard_pos]
            random.shuffle(hard_comb)
            for count in range(min(triplet_per_event, len(hard_comb))):
                hp, hn = hard_comb[count]
                triplet = (i, hp, hn)
                if not triplet in triplet_selected:
                    triplet_selected.append(triplet)

                    ################## structure mining ####################

                        
        if len(triplet_selected)-triplet_count >= triplet_per_batch:
            break

    triplet_selected = triplet_selected[:(triplet_count + triplet_per_batch)]
    mul_count = len(triplet_selected) - triplet_count

    triplet_input_idx = [idx for triplet in triplet_selected for idx in triplet]

    return triplet_input_idx, triplet_count, mul_count



def nopos_triplets_multimodal(sim_prob, max_num=1000):
    """
    randomly select triplets, not only use high-confidence pairs
    still enforce at least one positive and one negative in a row
    no constraints on positive rows
    """
    perm = np.random.permutation(sim_prob.shape[0])
    sim_prob = sim_prob[perm]

    mul_idx = []
    count = 0
    for i in range(sim_prob.shape[0]):
        pos_idx = np.where(sim_prob[i]>0.5)[0]
        neg_idx = np.where(sim_prob[i]<0.5)[0]
        
        np.random.shuffle(neg_idx)
        if len(pos_idx):
            neg_idx = neg_idx[:len(pos_idx)]
            idx = np.hstack((pos_idx, neg_idx))
        else:
            idx = neg_idx[:8]
        np.random.shuffle(idx)

        perm = itertools.permutations(idx, 2)
        for j in range(int(np.ceil(max_num/sim_prob.shape[0]))):
            try:
                pair = perm.__next__()
                mul_idx.extend([i, pair[0], pair[1]])
                count += 1

                if count == max_num:
                    return mul_idx, count
            except:
                break

    return mul_idx, count

def random_triplets_multimodal(sim_prob, max_num=1000):
    """
    randomly select triplets, not only use high-confidence pairs
    still enforce at least one positive and one negative in a row
    """
    perm = np.random.permutation(sim_prob.shape[0])
    sim_prob = sim_prob[perm]

    pos_rows = np.where(np.sum((sim_prob>0.5), axis=1) > 1)[0]

    mul_idx = []
    count = 0
    for i in pos_rows:
        pos_idx = np.where(sim_prob[i]>0.5)[0]
        neg_idx = np.where(sim_prob[i]<0.5)[0]
        
        np.random.shuffle(neg_idx)
        neg_idx = neg_idx[:len(pos_idx)]
        idx = np.hstack((pos_idx, neg_idx))
        np.random.shuffle(idx)

        perm = itertools.permutations(idx, 2)
        for j in range(int(np.ceil(max_num/len(pos_rows)))):
            try:
                pair = perm.__next__()
                mul_idx.extend([i, pair[0], pair[1]])
                count += 1

                if count == max_num:
                    return mul_idx, count
            except:
                break

    return mul_idx, count


def select_triplets_multimodal(sim_prob, threshold=0.8, max_num=1000):
    """
    sim_prob -- similarity probabilities from multimodal data, [N, N], 1 indicates similar and 0 indicates dissimilar
    max_num -- maximum number of triplets
    """
    perm = np.random.permutation(sim_prob.shape[0])
    sim_prob = sim_prob[perm]

    mul_idx = []
    count = 0
    for i in range(sim_prob.shape[0]):
        # get high-confidence pairs
        pos_idx = np.where(sim_prob[i]>threshold)[0]
        neg_idx = np.where(sim_prob[i]<(1-threshold))[0]
        if len(pos_idx) and len(neg_idx):
            neg_idx = np.argsort(sim_prob[i])[:len(pos_idx)]    # get same number of negatives as positives
            high_confidence = np.hstack((pos_idx, neg_idx))
            np.random.shuffle(high_confidence)    # important to get (A,-,+) triplet

            # get all combinations
            for pair in itertools.combinations(high_confidence, 2):
                mul_idx.extend([i, pair[0], pair[1]])
                count += 1

                if count == max_num:
                    return mul_idx, count

    return mul_idx, count
    

def pos_neg_pairs(lab):
    """
    return all pos-neg pairs
    """
    pos_neg_idx = []
    for i, l in enumerate(lab):
        if l > 0:    # only foreground events as anchor
            all_neg_idx = np.where(lab != l)[0]
            for neg_idx in all_neg_idx:
                pos_neg_idx.extend([i,neg_idx,neg_idx])    # add one more void element for consistent with triple loss
    return pos_neg_idx



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
    train_set = prepare_multimodal_dataset(cfg.feature_root, train_session, cfg.feat, cfg.label_root)
    if cfg.task == "supervised":    # fully supervised task
        train_set = train_set[:cfg.label_num]
    batch_per_epoch = len(train_set)//cfg.sess_per_batch
    labeled_session = train_session[:cfg.label_num]

    val_session = cfg.val_session
    val_set = prepare_multimodal_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        
        ####################### Load models here ########################
        sensors_emb_dim = 32
        segment_emb_dim = 32

        with tf.variable_scope("modality_core"):
            # load backbone model
            if cfg.network == "convtsn":
                model_emb = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
            elif cfg.network == "convrtsn":
                model_emb = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
            elif cfg.network == "convbirtsn":
                model_emb = networks.ConvBiRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
            else:
                raise NotImplementedError

            input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
            dropout_ph = tf.placeholder(tf.float32, shape=[])
            model_emb.forward(input_ph, dropout_ph)    # for lstm has variable scope

        with tf.variable_scope("modality_sensors"):
            model_emb_sensors = networks.RTSN(n_seg=cfg.num_seg, emb_dim=sensors_emb_dim)
            model_pairsim_sensors = networks.PDDM(n_input=sensors_emb_dim)

            input_sensors_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, 8])
            model_emb_sensors.forward(input_sensors_ph, dropout_ph)

            var_list = {}
            for v in tf.global_variables():
                if v.op.name.startswith("modality_sensors"):
                    var_list[v.op.name.replace("modality_sensors/","")] = v
            restore_saver_sensors = tf.train.Saver(var_list)

        with tf.variable_scope("modality_segment"):
            model_emb_segment = networks.RTSN(n_seg=cfg.num_seg, emb_dim=segment_emb_dim, n_input=357)
            model_pairsim_segment = networks.PDDM(n_input=segment_emb_dim)

            input_segment_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, 357])
            model_emb_segment.forward(input_segment_ph, dropout_ph)

            var_list = {}
            for v in tf.global_variables():
                if v.op.name.startswith("modality_segment"):
                    var_list[v.op.name.replace("modality_segment/","")] = v
            restore_saver_segment = tf.train.Saver(var_list)


        ############################# Forward Pass #############################


        # Core branch
        if cfg.normalized:
            embedding = tf.nn.l2_normalize(model_emb.hidden, axis=-1, epsilon=1e-10)
        else:
            embedding = model_emb.hidden

        # get the number of multimodal triplets (x3)
        mul_num_ph = tf.placeholder(tf.int32, shape=[])

        # variable for visualizing the embeddings
        emb_var = tf.Variable([0.0], name='embeddings')
        set_emb = tf.assign(emb_var, embedding, validate_shape=False)

        # calculated for monitoring all-pair embedding distance
        diffs = utils.all_diffs_tf(embedding, embedding)
        all_dist = utils.cdist_tf(diffs)
        tf.summary.histogram('embedding_dists', all_dist)

        # split embedding into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embedding[:(tf.shape(embedding)[0]-mul_num_ph)], [-1,3,cfg.emb_dim]), 3, 1)
        anchor_mul, positive_mul, negative_mul = tf.unstack(tf.reshape(embedding[-mul_num_ph:], [-1,3,cfg.emb_dim]), 3, 1)


        # Sensors branch
        emb_sensors = model_emb_sensors.hidden
        A_sensors, B_sensors, C_sensors = tf.unstack(tf.reshape(emb_sensors, [-1,3,sensors_emb_dim]), 3, 1)
        model_pairsim_sensors.forward(tf.stack([A_sensors, B_sensors], axis=1))
        pddm_AB_sensors = model_pairsim_sensors.prob[:, 1]
        model_pairsim_sensors.forward(tf.stack([A_sensors, C_sensors], axis=1))
        pddm_AC_sensors = model_pairsim_sensors.prob[:, 1]

        # Segment branch
        emb_segment = model_emb_segment.hidden
        A_segment, B_segment, C_segment = tf.unstack(tf.reshape(emb_segment, [-1,3,segment_emb_dim]), 3, 1)
        model_pairsim_segment.forward(tf.stack([A_segment, B_segment], axis=1))
        pddm_AB_segment = model_pairsim_segment.prob[:, 1]
        model_pairsim_segment.forward(tf.stack([A_segment, C_segment], axis=1))
        pddm_AC_segment = model_pairsim_segment.prob[:, 1]

        # fuse prob from all modalities
        prob_AB = 0.5 * (pddm_AB_sensors + pddm_AB_segment)
        prob_AC = 0.5 * (pddm_AC_sensors + pddm_AC_segment)

        ############################# Calculate loss #############################

        # triplet loss for labeled inputs
        metric_loss1 = networks.triplet_loss(anchor, positive, negative, cfg.alpha)

        # weighted triplet loss for multimodal inputs
#        if cfg.weighted:
#            metric_loss2, _ = networks.weighted_triplet_loss(anchor_mul, positive_mul, negative_mul, prob_AB, prob_AC, cfg.alpha)
#        else:
        metric_loss2 = networks.triplet_loss(anchor_mul, positive_mul, negative_mul, cfg.alpha)

        # whether to apply joint optimization
        if cfg.no_joint:
            unimodal_var_list = [v for v in tf.global_variables() if v.op.name.startswith("modality_core")]
            train_var_list = unimodal_var_list
        else:
            multimodal_var_list = [v for v in tf.global_variables() if not (v.op.name.startswith("modality_sensors/RTSN") or v.op.name.startswith("modality_segment/RTSN"))]
            train_var_list = multimodal_var_list

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.cond(tf.greater(mul_num_ph, 0),
                lambda: tf.cond(tf.equal(mul_num_ph, tf.shape(embedding)[0]), 
                    lambda: metric_loss2 * cfg.lambda_multimodal + regularization_loss * cfg.lambda_l2,
                    lambda: metric_loss1 + metric_loss2 * cfg.lambda_multimodal + regularization_loss * cfg.lambda_l2),
                lambda: metric_loss1 + regularization_loss * cfg.lambda_l2)

        tf.summary.scalar('learning_rate', lr_ph)
        train_op = utils.optimize(total_loss, global_step, cfg.optimizer,
                lr_ph, train_var_list)

        saver = tf.train.Saver(max_to_keep=10)
        summary_op = tf.summary.merge_all()    # not logging histogram of variables because it will cause problem when only unimodal_train_op is called

        summ_prob_AB = tf.summary.histogram('Prob_AB_histogram', prob_AB)
        summ_prob_AC = tf.summary.histogram('Prob_AC_histogram', prob_AC)
#        summ_weights = tf.summary.histogram('Weights_histogram', weights)

        #########################################################################

        # session iterator for session sampling
        feat_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        feat2_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        feat3_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        label_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        train_data = multimodal_session_generator(feat_paths_ph, feat2_paths_ph, feat3_paths_ph, label_paths_ph, sess_per_batch=cfg.sess_per_batch, num_threads=2, shuffled=False, preprocess_func=[model_emb.prepare_input, model_emb_sensors.prepare_input, model_emb_segment.prepare_input])
        train_sess_iterator = train_data.make_initializable_iterator()
        next_train = train_sess_iterator.get_next()

        # prepare validation data
        val_sess = []
        val_feats = []
        val_feats2 = []
        val_feats3 = []
        val_labels = []
        val_boundaries = []
        for session in val_set:
            session_id = os.path.basename(session[1]).split('_')[0]
            eve_batch, lab_batch, boundary = load_data_and_label(session[0], session[-1], model_emb.prepare_input_test)    # use prepare_input_test for testing time
            val_feats.append(eve_batch)
            val_labels.append(lab_batch)
            val_sess.extend([session_id]*eve_batch.shape[0])
            val_boundaries.extend(boundary)

            eve2_batch, _,_ = load_data_and_label(session[1], session[-1], model_emb_sensors.prepare_input_test)
            val_feats2.append(eve2_batch)

            eve3_batch, _,_ = load_data_and_label(session[2], session[-1], model_emb_segment.prepare_input_test)
            val_feats3.append(eve3_batch)
        val_feats = np.concatenate(val_feats, axis=0)
        val_feats2 = np.concatenate(val_feats2, axis=0)
        val_feats3 = np.concatenate(val_feats3, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        print ("Shape of val_feats: ", val_feats.shape)

        # generate metadata.tsv for visualize embedding
        with open(os.path.join(result_dir, 'metadata_val.tsv'), 'w') as fout:
            fout.write('id\tlabel\tsession_id\tstart\tend\n')
            for i in range(len(val_sess)):
                fout.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(i, val_labels[i,0], val_sess[i],
                                    val_boundaries[i][0], val_boundaries[i][1]))

        # compute pairwise embedding distance for each class on validation set
        # FIXME: don't use np.max
#        dist_dict = {}
#        for i in range(1, np.max(val_labels)+1):
#            temp_feat = val_feats[np.where(val_labels==i)[0]]
#            dist_dict[i] = [np.mean(utils.cdist(utils.all_diffs(temp_feat, temp_feat),
#                                metric=cfg.metric))]
#        pdb.set_trace()

        #########################################################################


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

            print ("Restoring sensors model: %s" % cfg.sensors_path)
            restore_saver_sensors.restore(sess, cfg.sensors_path)
            print ("Restoring segment model: %s" % cfg.segment_path)
            restore_saver_segment.restore(sess, cfg.segment_path)

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
                            0.01**((epoch-cfg.static_epochs)/(cfg.max_epochs-cfg.static_epochs))



                # prepare data for this epoch
                random.shuffle(train_set)

                paths = list(zip(*[iter(train_set)]*cfg.sess_per_batch))

                feat_paths = [[p[0] for p in path] for path in paths]
                feat2_paths = [[p[1] for p in path] for path in paths]
                feat3_paths = [[p[2] for p in path] for path in paths]
                label_paths = [[p[-1] for p in path] for path in paths]

                sess.run(train_sess_iterator.initializer, feed_dict={feat_paths_ph: feat_paths,
                  feat2_paths_ph: feat2_paths,
                  feat3_paths_ph: feat3_paths,
                  label_paths_ph: label_paths})

                # for each epoch
                batch_count = 1
                while True:
                    try:
                        ##################### Data loading ########################
                        start_time = time.time()
                        eve, eve_sensors, eve_segment, lab, batch_sess = sess.run(next_train)

                        # for memory concern, 1000 events are used in maximum
                        if eve.shape[0] > cfg.event_per_batch:
                            idx = np.random.permutation(eve.shape[0])[:cfg.event_per_batch]
                            eve = eve[idx]
                            eve_sensors = eve_sensors[idx]
                            eve_segment = eve_segment[idx]
                            lab = lab[idx]
                            batch_sess = batch_sess[idx]
                        load_time = time.time() - start_time
    
                        ##################### Triplet selection #####################
                        start_time = time.time()
                        # Get the embeddings of all events
                        eve_embedding = np.zeros((eve.shape[0], cfg.emb_dim), dtype='float32')
                        for start, end in zip(range(0, eve.shape[0], cfg.batch_size),
                                            range(cfg.batch_size, eve.shape[0]+cfg.batch_size, cfg.batch_size)):
                            end = min(end, eve.shape[0])
                            emb = sess.run(embedding, feed_dict={input_ph: eve[start:end], dropout_ph: 1.0})
                            eve_embedding[start:end] = np.copy(emb)
    
                        # sample triplets within sampled sessions
                        all_diff = utils.all_diffs(eve_embedding, eve_embedding)
                        triplet_input_idx, active_count = utils.select_triplets_facenet(lab,utils.cdist(all_diff,metric=cfg.metric),cfg.triplet_per_batch,cfg.alpha,num_negative=cfg.num_negative)
                        if len(triplet_input_idx) == 0:
                            continue

                        triplet_count = len(triplet_input_idx) // 3
                        multimodal_count = 0
                        if epoch >= cfg.multimodal_epochs:
                            # Get the similarity of all events
                            sim_prob = np.zeros((eve.shape[0], eve.shape[0]), dtype='float32')*np.nan
                            comb = list(itertools.combinations(range(eve.shape[0]), 2))
                            for start, end in zip(range(0, len(comb), cfg.batch_size),
                                                range(cfg.batch_size, len(comb)+cfg.batch_size, cfg.batch_size)):
                                end = min(end, len(comb))
                                comb_idx = []
                                for c in comb[start:end]:
                                    comb_idx.extend([c[0], c[1], c[1]])
                                sim = sess.run(prob_AB, feed_dict={
                                                input_sensors_ph: eve_sensors[comb_idx],
                                                input_segment_ph: eve_segment[comb_idx],
                                                dropout_ph: 1.0})
                                for i in range(sim.shape[0]):
                                    sim_prob[comb[start+i][0], comb[start+i][1]] = sim[i]
                                    sim_prob[comb[start+i][1], comb[start+i][0]] = sim[i]

                            # sample triplets from similarity prediction
                            # maximum number not exceed the number of triplet_input from facenet selection

                            # hard samples mining
                            triplet_input_idx, triplet_count, multimodal_count1 = select_triplets_mul_hard(triplet_input_idx, lab, sim_prob, cfg.triplet_per_batch, 3, 0.8, 0.2)

                            sensors_input = eve_sensors[triplet_input_idx[-(3*multimodal_count1):]]
                            segment_input = eve_segment[triplet_input_idx[-(3*multimodal_count1):]]

                            # add up all multimodal triplets
                            multimodal_count = multimodal_count1
                        
                        print (triplet_count, multimodal_count)
                        triplet_input = eve[triplet_input_idx]

                        select_time = time.time() - start_time

                        if len(triplet_input.shape) > 5:    # debugging
                            pdb.set_trace()
    
                        ##################### Start training  ########################

                        # supervised initialization
                        if multimodal_count == 0:
                            if triplet_count == 0:
                                continue
                            err, metric_err1,  _, step, summ = sess.run(
                                    [total_loss, metric_loss1, train_op, global_step, summary_op],
                                    feed_dict = {input_ph: triplet_input,
                                                 dropout_ph: cfg.keep_prob,
                                                 mul_num_ph: 0,
                                                 lr_ph: learning_rate})
                            metric_err2 = 0
                        else:
                            err, metric_err1, metric_err2, _, step, summ, s_AB, s_AC = sess.run(
                                    [total_loss, metric_loss1, metric_loss2, train_op, global_step, summary_op, summ_prob_AB, summ_prob_AC],
                                    feed_dict = {input_ph: triplet_input,
                                                 input_sensors_ph: sensors_input,
                                                 input_segment_ph: segment_input,
                                                 mul_num_ph: multimodal_count*3,
                                                 dropout_ph: cfg.keep_prob,
                                                 lr_ph: learning_rate})
                            summary_writer.add_summary(s_AB, step)
                            summary_writer.add_summary(s_AC, step)
    
    
                        print ("%s\tEpoch: [%d][%d/%d]\tEvent num: %d\tTriplet num: %d\tLoad time: %.3f\tSelect time: %.3f\tLoss %.4f" % \
                                (cfg.name, epoch+1, batch_count, batch_per_epoch, eve.shape[0], triplet_count+multimodal_count, load_time, select_time, err))
    
                        summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err),
                                    tf.Summary.Value(tag="active_count", simple_value=active_count),
                                    tf.Summary.Value(tag="triplet_count", simple_value=triplet_count),
                                    tf.Summary.Value(tag="multimodal_count", simple_value=multimodal_count),
                                    tf.Summary.Value(tag="metric_loss1", simple_value=metric_err1),
                                    tf.Summary.Value(tag="metric_loss2", simple_value=metric_err2)])
    
                        summary_writer.add_summary(summary, step)
                        summary_writer.add_summary(summ, step)

                        batch_count += 1
                    
                    except tf.errors.OutOfRangeError:
                        print ("Epoch %d done!" % (epoch+1))
                        break

                # validation on val_set
                print ("Evaluating on validation set...")
                val_embeddings, _ = sess.run([embedding, set_emb],
                                                feed_dict = {input_ph: val_feats,
                                                             dropout_ph: 1.0})
                mAP, mPrec, recall = utils.evaluate_simple(val_embeddings, val_labels)
                summary = tf.Summary(value=[tf.Summary.Value(tag="Valiation mAP", simple_value=mAP),
                                            tf.Summary.Value(tag="Validation Recall@1", simple_value=recall),
                                            tf.Summary.Value(tag="Validation mPrec@0.5", simple_value=mPrec)])
                summary_writer.add_summary(summary, step)
                print ("Epoch: [%d]\tmAP: %.4f\tmPrec: %.4f" % (epoch+1,mAP,mPrec))

                # config for embedding visualization
                config = projector.ProjectorConfig()
                visual_embedding = config.embeddings.add()
                visual_embedding.tensor_name = emb_var.name
                visual_embedding.metadata_path = os.path.join(result_dir, 'metadata_val.tsv')
                projector.visualize_embeddings(summary_writer, config)

                # save model
                saver.save(sess, os.path.join(result_dir, cfg.name+'.ckpt'), global_step=step)

if __name__ == "__main__":
    main()
