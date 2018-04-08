"""
Base model with batch_hard loss
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

sys.path.append('../')
from configs.train_config import TrainConfig
from data_io import session_generator, load_data_and_label, prepare_dataset
import networks
import utils

def select_batch(lab, batch_size):
    """
    Select the samples for training
    Balancing the number of samples for each class

    Arguments:
    lab -- array of labels, [N,]
    batch_size
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
    batch_idx = []
    while len(batch_idx) < batch_size:
        keys = list(idx_dict.keys())
        if len(keys) == 0:
            break

        for key in keys:
            # pop one out
            batch_idx.append(idx_dict[key][0])
            idx_dict[key] = idx_dict[key][1:]

            if len(idx_dict[key]) == 0:
                # remove the key to prevent infinite loop
                del idx_dict[key]

    return batch_idx


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
    train_session = cfg.train_session
    train_set = prepare_dataset(cfg.feature_root, train_session, cfg.feat, cfg.label_root)
    batch_per_epoch = len(train_set)//cfg.sess_per_batch

    val_session = cfg.val_session
    val_set = prepare_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        # load backbone model
        if cfg.network == "tsn":
            model = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "rtsn":
            model = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)

        # get the embedding
        input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
        label_ph = tf.placeholder(tf.float32, shape=[None])
        dropout_ph = tf.placeholder(tf.float32, shape=[])
        model.forward(input_ph, dropout_ph)
        if cfg.normalized:
            embedding = tf.nn.l2_normalize(model.hidden, axis=-1, epsilon=1e-10)
        else:
            embedding = model.hidden

        # variable for visualizing the embeddings
        emb_var = tf.Variable([0.0], name='embeddings')
        set_emb = tf.assign(emb_var, embedding, validate_shape=False)

        # calculated for monitoring all-pair embedding distance
        diffs = utils.all_diffs_tf(embedding, embedding)
        all_dist = utils.cdist_tf(diffs)
        tf.summary.histogram('embedding_dists', all_dist)

        metric_loss, num_active, diff, weights, fp, cn = networks.lifted_loss(all_dist, label_ph, cfg.alpha)

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = metric_loss + regularization_loss * cfg.lambda_l2

        tf.summary.scalar('active_ratio', num_active)
        tf.summary.scalar('learning_rate', lr_ph)
        train_op = utils.optimize(total_loss, global_step, cfg.optimizer,
                lr_ph, tf.global_variables())

        saver = tf.train.Saver(max_to_keep=10)

        summary_op = tf.summary.merge_all()

        # session iterator for session sampling
        feat_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        label_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        train_data = session_generator(feat_paths_ph, label_paths_ph, sess_per_batch=cfg.sess_per_batch, num_threads=2, shuffled=False, preprocess_func=model.prepare_input)
        train_sess_iterator = train_data.make_initializable_iterator()
        next_train = train_sess_iterator.get_next()

        # prepare validation data
        val_feats = []
        val_labels = []
        for session in val_set:
            eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], model.prepare_input_test)    # use prepare_input_test for testing time
            val_feats.append(eve_batch)
            val_labels.append(lab_batch)
        val_feats = np.concatenate(val_feats, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        print ("Shape of val_feats: ", val_feats.shape)

        # generate metadata.tsv for visualize embedding
        with open(os.path.join(result_dir, 'metadata_val.tsv'), 'w') as fout:
            for v in val_labels:
                fout.write('%d\n' % int(v))


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
                        # First, sample sessions for a batch
                        start_time_select = time.time()
                        eve, se, lab = sess.run(next_train)
                        select_time1 = time.time() - start_time_select

                        # Second, select samples for a batch
                        batch_idx = select_batch(lab,cfg.batch_size)
                        eve = eve[batch_idx]
                        lab = lab[batch_idx]

                        # Third, perform training on a batch
                        start_time_train = time.time()
                        err, _, step, summ, diff_v, weights_v, fp_v, cn_v, dist_v= sess.run([total_loss, train_op, global_step, summary_op, diff, weights, fp, cn, all_dist],
                                feed_dict = {input_ph: eve,
                                            dropout_ph: cfg.keep_prob,
                                            label_ph: np.squeeze(lab),
                                            lr_ph: learning_rate})

                        train_time = time.time() - start_time_train
                        print ("Epoch: [%d][%d/%d]\tEvent num: %d\tSelect_time: %.3f\tTrain_time: %.3f\tLoss %.4f" % \
                                (epoch+1, batch_count, batch_per_epoch, eve.shape[0], select_time1, train_time, err))

                        summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err),
                            tf.Summary.Value(tag="select_time1", simple_value=select_time1)])
                        summary_writer.add_summary(summary, step)
                        summary_writer.add_summary(summ, step)

                        batch_count += 1
                    
                    except tf.errors.OutOfRangeError:
                        print ("Epoch %d done!" % (epoch+1))
                        break

                # validation on val_set
                print ("Evaluating on validation set...")
                val_embeddings, _ = sess.run([embedding, set_emb], feed_dict={input_ph: val_feats, dropout_ph: 1.0})
                mAP, _ = utils.evaluate(val_embeddings, val_labels)

                summary = tf.Summary(value=[tf.Summary.Value(tag="Valiation mAP", simple_value=mAP)])
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
