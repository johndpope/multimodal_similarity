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


def select_triplets_multimodal(sim_prob, threshold=0.7):
    """
    sim_prob -- similarity probabilities from multimodal data, [N, N], 1 indicates similar and 0 indicates dissimilar
    """
    mul_idx = []

    count = 0
    for i in range(sim_prob.shape[0]):
        # important to ignore nan
        if np.nanmax(sim_prob[i]) - np.nanmin(sim_prob[i]) > threshold:
            mul_idx.extend([i, np.nanargmax(sim_prob[i]), np.nanargmin(sim_prob[i])])
            count += 1

    return mul_idx, count
    

def pos_neg_pairs(lab):
    """
    return all pos-neg pairs
    """
    pos_neg_idx = []
    for i, l in enumerate(lab):
        if l > 0:
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
    batch_per_epoch = len(train_set)//cfg.sess_per_batch

    val_session = cfg.val_session
    val_set = prepare_multimodal_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        
        ####################### Load models here ########################

        with tf.variable_scope("modality_core"):
            # load backbone model
            if cfg.network == "convtsn":
                model_emb = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
            elif cfg.network == "convrtsn":
                model_emb = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
            else:
                raise NotImplementedError

        with tf.variable_scope("modality_sensors"):
            sensors_emb_dim = 32
            model_emb_sensors = networks.RTSN(n_seg=cfg.num_seg, emb_dim=sensors_emb_dim)
            model_pairsim_sensors = networks.PairSim(n_input=sensors_emb_dim)

            var_list = {}
            for v in tf.global_variables():
                if v.op.name.startswith("modality_sensors"):
                    var_list[v.op.name.replace("modality_sensors/","")] = v
            restore_saver_sensors = tf.train.Saver(var_list)

        ############################# Forward Pass #############################

        # Core branch
        input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
        dropout_ph = tf.placeholder(tf.float32, shape=[])

        model_emb.forward(input_ph, dropout_ph)
        if cfg.normalized:
            embedding = tf.nn.l2_normalize(model_emb.hidden, axis=-1, epsilon=1e-10)
        else:
            embedding = model_emb.hidden

        # variable for visualizing the embeddings
        emb_var = tf.Variable([0.0], name='embeddings')
        set_emb = tf.assign(emb_var, embedding, validate_shape=False)

        # calculated for monitoring all-pair embedding distance
        diffs = utils.all_diffs_tf(embedding, embedding)
        all_dist = utils.cdist_tf(diffs)
        tf.summary.histogram('embedding_dists', all_dist)

        # split embedding into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embedding, [-1,3,cfg.emb_dim]), 3, 1)


        # Sensors branch
        input_sensors_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, 8])

        model_emb_sensors.forward(input_sensors_ph, dropout_ph)
        emb_sensors = model_emb_sensors.hidden
        A_sensors, B_sensors, C_sensors = tf.unstack(tf.reshape(emb_sensors, [-1,3,sensors_emb_dim]), 3, 1)
        AB_pairs_sensors = tf.stack([A_sensors, B_sensors], axis=1)
        AC_pairs_sensors = tf.stack([A_sensors, C_sensors], axis=1)
        pairs_sensors = tf.concat([AB_pairs_sensors, AC_pairs_sensors], axis=0)
        model_pairsim_sensors.forward(pairs_sensors, dropout_ph)
        prob_sensors = model_pairsim_sensors.prob
        prob_sensors = tf.concat([prob_sensors[:tf.shape(A_sensors)[0]], prob_sensors[tf.shape(A_sensors)[0]:]], axis=1)    # shape: [N, 4]


        # fuse prob from all modalities
        prob = prob_sensors

        ############################# Calculate loss #############################

        # whether to apply joint optimization
        if cfg.no_joint:
            metric_loss = networks.triplet_loss(anchor, positive, negative, cfg.alpha)
            train_var_list = [v for v in tf.global_variables() if v.op.name.startswith("modality_core")]

        else:
            # weighted triplet loss
            metric_loss = networks.weighted_triplet_loss(anchor, positive, negative, prob, cfg.alpha)
            train_var_list = tf.global_variables()

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = metric_loss + regularization_loss * cfg.lambda_l2

        tf.summary.scalar('learning_rate', lr_ph)
        train_op = utils.optimize(total_loss, global_step, cfg.optimizer,
                lr_ph, train_var_list)

        saver = tf.train.Saver(max_to_keep=10)

        summary_op = tf.summary.merge_all()

        #########################################################################

        # session iterator for session sampling
        feat_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        feat2_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        label_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        train_data = multimodal_session_generator(feat_paths_ph, feat2_paths_ph, label_paths_ph, sess_per_batch=cfg.sess_per_batch, num_threads=2, shuffled=False, preprocess_func=[model_emb.prepare_input, model_emb_sensors.prepare_input])
        train_sess_iterator = train_data.make_initializable_iterator()
        next_train = train_sess_iterator.get_next()

        # prepare validation data
        val_sess = []
        val_feats = []
        val_feats2 = []
        val_labels = []
        val_boundaries = []
        for session in val_set:
            session_id = os.path.basename(session[1]).split('_')[0]
            eve_batch, lab_batch, boundary = load_data_and_label(session[0], session[-1], model_emb.prepare_input_test)    # use prepare_input_test for testing time
            val_feats.append(eve_batch)
            val_labels.append(lab_batch)
            val_sess.extend([session_id]*eve_batch.shape[0])
            val_boundaries.extend(boundary)

            eve2_batch, _,_ = load_data_and_label(session[1], session[-1], utils.mean_pool_input)
            val_feats2.append(eve2_batch)
        val_feats = np.concatenate(val_feats, axis=0)
        val_feats2 = np.concatenate(val_feats2, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        print ("Shape of val_feats: ", val_feats.shape)

        # generate metadata.tsv for visualize embedding
        with open(os.path.join(result_dir, 'metadata_val.tsv'), 'w') as fout:
            fout.write('id\tlabel\tsession_id\tstart\tend\n')
            for i in range(len(val_sess)):
                fout.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(i, val_labels[i,0], val_sess[i],
                                            val_boundaries[i][0], val_boundaries[i][1]))

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

                paths = list(zip(*[iter(train_set)]*cfg.sess_per_batch))

                feat_paths = [[p[0] for p in path] for path in paths]
                feat2_paths = [[p[1] for p in path] for path in paths]
                label_paths = [[p[-1] for p in path] for path in paths]

                sess.run(train_sess_iterator.initializer, feed_dict={feat_paths_ph: feat_paths,
                  feat2_paths_ph: feat2_paths,
                  label_paths_ph: label_paths})

                # for each epoch
                batch_count = 1
                while True:
                    try:
                        ##################### Data loading ########################
                        start_time = time.time()
                        eve, eve2, lab = sess.run(next_train)
                        load_time = time.time() - start_time
    
                        ##################### Triplet selection #####################
                        start_time = time.time()
                        # Get the embeddings of all events
                        eve_embedding = np.zeros((eve.shape[0], cfg.emb_dim), dtype='float32')
                        for start, end in zip(range(0, eve.shape[0], cfg.batch_size),
                                            range(cfg.batch_size, eve.shape[0]+cfg.batch_size, cfg.batch_size)):
                            end = min(end, eve.shape[0])
                            emb = sess.run(embedding, feed_dict={input_ph: eve[start:end], dropout_ph: 1.0})
                            eve_embedding[start:end] = emb
    
                        # sample triplets within sampled sessions
                        triplet_input_idx, negative_count = utils.select_triplets_facenet(lab,eve_embedding,cfg.triplet_per_batch,cfg.alpha,metric=cfg.metric)

                        multimodal_count = 0
                        if epoch >= cfg.multimodal_epochs:
                            # Get the similairty prediction of all pos-neg pairs
                            pos_neg_idx = pos_neg_pairs(lab)
                            sim_prob = np.zeros((eve.shape[0], eve.shape[0]), dtype='float32')*np.nan
                            for start, end in zip(range(0, len(pos_neg_idx), 3*cfg.batch_size),
                                            range(3*cfg.batch_size, len(pos_neg_idx)+3*cfg.batch_size, 3*cfg.batch_size)):
                                end = min(end, len(pos_neg_idx))
                                batch_idx = pos_neg_idx[start:end]
                                batch_prob = sess.run(prob, feed_dict={
                                                        input_sensors_ph: eve2[batch_idx],
                                                        dropout_ph: 1.0})
                                
                                for i in range(batch_prob.shape[0]):
                                    sim_prob[batch_idx[i*3], batch_idx[i*3+1]] = batch_prob[i,1]

                            # sample triplets from similarity prediction
                            multimodal_input_idx, multimodal_count = select_triplets_multimodal(sim_prob, threshold=0.75)
                            triplet_input_idx.extend(multimodal_input_idx)
    
                        
                        triplet_input = eve[triplet_input_idx]
                        triplet_input2 = eve2[triplet_input_idx]

                        select_time = time.time() - start_time
    
                        ##################### Start training  ########################
    
                        err,  _, step, summ = sess.run(
                                [total_loss, train_op, global_step, summary_op],
                                feed_dict = {input_ph: triplet_input,
                                             input_sensors_ph: triplet_input2,
                                             dropout_ph: cfg.keep_prob,
                                             lr_ph: learning_rate})
    
                        print ("%s\tEpoch: [%d][%d/%d]\tEvent num: %d\tLoad time: %.3f\tSelect time: %.3f\tLoss %.4f" % \
                                (cfg.name, epoch+1, batch_count, batch_per_epoch, eve.shape[0], load_time, select_time, err))
    
                        summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err),
                                    tf.Summary.Value(tag="negative_count", simple_value=negative_count),
                                    tf.Summary.Value(tag="multimodal_count", simple_value=multimodal_count)])
    
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
                mAP, mPrec = utils.evaluate_simple(val_embeddings, val_labels)

                summary = tf.Summary(value=[tf.Summary.Value(tag="Valiation mAP", simple_value=mAP),
                                            tf.Summary.Value(tag="Validation mPrec@0.5", simple_value=mPrec)])
                summary_writer.add_summary(summary, step)

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
