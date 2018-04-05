"""
Pretrain unimodal similarity prediction
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

sys.path.append('../')
from configs.train_config import TrainConfig
from data_io import session_generator, load_data_and_label, prepare_dataset
import networks
import utils

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

    val_session = cfg.val_session
    val_set = prepare_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)

    # prepare validation data
    prepare_input = functools.partial(utils.tsn_prepare_input, cfg.num_seg)
    val_feats = []
    val_labels = []
    for session in val_set:
        eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], prepare_input)
        val_feats.append(eve_batch)
        val_labels.append(lab_batch)
    val_feats = np.concatenate(val_feats, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    print ("Shape of val_feats: ", val_feats.shape)

    # FIXME
    n_input=8


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        # load backbone model
        model = networks.SAE(n_input=n_input, emb_dim=cfg.emb_dim)

        # restore specific variables
        var_to_restore = [v for v in tf.all_variables() if v.name.startswith('W') or v.name.startswith('b')]
        saver_restore = tf.train.Saver(var_to_restore)

        # get the embedding
        input_ph = tf.placeholder(tf.float32, shape=[None, n_input])
        model.forward(input_ph)
        embedding = tf.nn.l2_normalize(model.hidden, axis=1, epsilon=1e-10, name='embedding')

        emb_pairs = tf.reshape(embedding, [-1, 2, cfg.emb_dim])

        pairsim = networks.PairSim(n_input=cfg.emb_dim)
        pairsim.forward(emb_pairs)
        logits = pairsim.logits
        prob = pairsim.prob

        label_ph = tf.placeholder(tf.int32, shape=[None])
        CE_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=logits)
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = CE_loss + regularization_loss * cfg.lambda_l2

        tf.summary.scalar('learning_rate', lr_ph)
        var_to_train = [v for v in tf.global_variables() if v.name.startswith("pairsim")]
        train_op = utils.optimize(total_loss, global_step, cfg.optimizer,
                lr_ph, var_to_train)


        saver = tf.train.Saver(max_to_keep=10)

        summary_op = tf.summary.merge_all()

        # session iterator for session sampling
        feat_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        label_paths_ph = tf.placeholder(tf.string, shape=[None, cfg.sess_per_batch])
        train_data = session_generator(feat_paths_ph, label_paths_ph, sess_per_batch=cfg.sess_per_batch, num_threads=2, shuffled=False, preprocess_func=prepare_input)
        train_sess_iterator = train_data.make_initializable_iterator()
        next_train = train_sess_iterator.get_next()


        # Start running the graph
        if cfg.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

        with sess.as_default():

            sess.run(tf.global_variables_initializer())

            [v, W1] = sess.run([var_to_restore, model.W_1], feed_dict=None)
            pdb.set_trace()

            # load pretrain model to initialize SAE
            if cfg.pretrained_model:
                print ("Restoring pretrained model: %s" % cfg.pretrained_model)
                saver_restore.restore(sess, cfg.pretrained_model)

            [v, W1] = sess.run([var_to_train, model.W_1], feed_dict=None)

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
                        # Sample sessions
                        start_time_select = time.time()
                        eve, _, lab = sess.run(next_train)
                        eve = eve.reshape(-1, n_input)    # reshape because we use tsn_prepare_input for sampling
                        select_time1 = time.time() - start_time_select

                        # Train on these sessions
                        for start, end in zip(range(0, eve.shape[0], cfg.batch_size),
                                            range(cfg.batch_size, eve.shape[0]+cfg.batch_size, cfg.batch_size)):
                            end = min(end, eve.shape[0])

                            start_time_train = time.time()
                            err, _, step, summ = sess.run([total_loss, train_op, global_step, summary_op],
                                    feed_dict = {input_ph: eve,
                                                lr_ph: learning_rate})
                            train_time = time.time() - start_time_train
                            print ("Epoch: [%d][%d]\tSelect_time: %.3f\tTrain_time: %.3f\tLoss %.4f" % \
                                    (epoch+1, batch_count, select_time1, train_time, err))
                            batch_count += 1

                            summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err)])
                            summary_writer.add_summary(summary, step)
                            summary_writer.add_summary(summ, step)
                    
                    except tf.errors.OutOfRangeError:
                        print ("Epoch %d done!" % (epoch+1))
                        epoch += 1
                        break

                # validation on val_set
                print ("Evaluating on validation set...")
                val_err, val_embeddings, _ = sess.run([total_loss, embedding, set_emb], feed_dict={input_ph: val_feats})
                mAP, _ = utils.evaluate(val_embeddings, val_labels)

                summary = tf.Summary(value=[tf.Summary.Value(tag="Validation mAP", simple_value=mAP),
                                            tf.Summary.Value(tag="Validation loss", simple_value=val_err)])
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
