"""
Classification baseline
Using TFRecord pipeline
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
from six import iteritems
import glob
from sklearn.metrics import accuracy_score

sys.path.append('../')
from configs.train_config import TrainConfig
from data_io import event_generator, load_data_and_label
import networks
import utils

def prepare_dataset(data_dir, sessions, feat, label_dir=None):

    if feat == 'resnet':
        appendix = '.npy'
    else:
        raise NotImplementedError

    dataset = []
    for sess in sessions:
        feat_path = os.path.join(data_dir, sess+appendix)
        label_path = os.path.join(label_dir, sess+'_goal.pkl')

        dataset.append((feat_path, label_path))

    return dataset

def write_configure_to_file(cfg, result_dir):
    with open(os.path.join(result_dir, 'config.txt'), 'w') as fout:
        for key, value in iteritems(vars(cfg)):
            fout.write('%s: %s\n' % (key, str(value)))

def main():

    cfg = TrainConfig().parse()
    print (cfg.name)
    result_dir = os.path.join(cfg.result_root, 
            cfg.name+'_'+datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    write_configure_to_file(cfg, result_dir)
    np.random.seed(seed=cfg.seed)

    # prepare dataset
    train_session = cfg.train_session
    tfrecords_files = glob.glob(cfg.tfrecords_root+'*.tfrecords')
    tfrecords_files = sorted(tfrecords_files)
    train_set = [f for f in tfrecords_files if os.path.basename(f).split('_')[0] in train_session]
    print ("Number of training events: %d" % len(train_set))

    val_session = cfg.val_session
    val_set = prepare_dataset(cfg.feature_root, val_session, cfg.feat, cfg.label_root)


    # construct the graph
    with tf.Graph().as_default():
        tf.set_random_seed(cfg.seed)
        global_step = tf.Variable(0, trainable=False)
        lr_ph = tf.placeholder(tf.float32, name='learning_rate')

        # load backbone model
        if cfg.network == "tsn":
            model = networks.ConvTSNClassifier(n_seg=cfg.num_seg, output_keep_prob=cfg.keep_prob)

        # get prediction
        input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
        output_ph = tf.placeholder(tf.int32, shape=[None])
        model.forward(input_ph)
        embedding = tf.nn.l2_normalize(model.feat, axis=1, epsilon=1e-10, name='embedding')
        logits = model.logits
        pred = tf.argmax(logits, 1)

        # variable for visualizing the embeddings
        emb_var = tf.Variable([0.0], name='embeddings')
        set_emb = tf.assign(emb_var, embedding, validate_shape=False)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_ph, logits=logits))

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = loss + regularization_loss * cfg.lambda_l2

        tf.summary.scalar('learning_rate', lr_ph)
        train_op = utils.optimize(total_loss, global_step, cfg.optimizer,
                lr_ph, tf.global_variables())

        saver = tf.train.Saver(max_to_keep=10)

        summary_op = tf.summary.merge_all()

        # session iterator for session sampling
        tf_paths_ph = tf.placeholder(tf.string, shape=[None])
        train_data = event_generator(tf_paths_ph, cfg.feat_dict, cfg.context_dict,
                event_per_batch=cfg.event_per_batch, num_threads=1, shuffled=True,
                preprocess_func=model.prepare_input_tf)
        train_sess_iterator = train_data.make_initializable_iterator()
        next_train = train_sess_iterator.get_next()

        # prepare validation data
        val_feats = []
        val_labels = []
        for session in val_set:
            eve_batch, lab_batch = load_data_and_label(session[0], session[1], model.prepare_input)
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
            if cfg.pretrained_model:
                print ("Restoring pretrained model: %s" % cfg.pretrained_model)
                saver.restore(sess, cfg.pretrained_model)

            ################## Training loop ##################
            epoch = 0
            while epoch < cfg.max_epochs:
                step = sess.run(global_step, feed_dict=None)

                # learning rate schedule, reference: "In defense of Triplet Loss"
                if epoch < cfg.static_epochs:
                    learning_rate = cfg.learning_rate
                else:
                    learning_rate = cfg.learning_rate * \
                            0.001**((epoch-cfg.static_epochs)/(cfg.max_epochs-cfg.static_epochs))

                sess.run(train_sess_iterator.initializer, feed_dict={tf_paths_ph: train_set})

                # for each epoch
                batch_count = 1
                while True:
                    try:
                        start_time_select = time.time()

                        context, feature_lists = sess.run(next_train)
                        select_time = time.time() - start_time_select

                        eve = feature_lists[cfg.feat].reshape((-1, cfg.num_seg)+cfg.feat_dim[cfg.feat])
                        lab = context['label']

                        # perform training on the batch
                        start_time_train = time.time()
                        err, y_pred, _, step, summ = sess.run([total_loss, pred, train_op, global_step, summary_op],
                                    feed_dict = {input_ph: eve,
                                                 output_ph: lab,
                                                 lr_ph: learning_rate})

                        # classification accuracy on batch
                        acc = accuracy_score(lab, y_pred)

                        train_time = time.time() - start_time_train
                        print ("Epoch: [%d: %d]\tSelect_time: %.3f\tTrain_time: %.3f\tLoss: %.4f\tAcc: %.4f" % \
                                    (epoch+1, batch_count, select_time, train_time, err, acc))

                        summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err),
                                                    tf.Summary.Value(tag="train_acc", simple_value=acc)])
                        summary_writer.add_summary(summary, step)
                        summary_writer.add_summary(summ, step)

                        batch_count += 1
                    
                    except tf.errors.OutOfRangeError:
                        print ("Epoch %d done!" % (epoch+1))
                        epoch += 1

                        break

                # validation on val_set
                print ("Evaluating on validation set...")
                val_embeddings, val_pred, _ = sess.run([embedding, pred, set_emb], feed_dict={input_ph: val_feats})
                acc = accuracy_score(val_labels, val_pred)
                mAP, _ = utils.evaluate(val_embeddings, val_labels)

                summary = tf.Summary(value=[tf.Summary.Value(tag="Valiation mAP", simple_value=mAP),
                                            tf.Summary.Value(tag="Validation ACC", simple_value=acc)])
                summary_writer.add_summary(summary, step)

                # config for embedding visualization
                config = projector.ProjectorConfig()
                visual_embedding = config.embeddings.add()
                visual_embedding.tensor_name = emb_var.name
                visual_embedding.metadata_path = os.path.join(result_dir, 'metadata_val.tsv')
                projector.visualize_embeddings(summary_writer, config)

                # write summary and save model
                saver.save(sess, os.path.join(result_dir, cfg.name+'.ckpt'), global_step=step)

if __name__ == "__main__":
    main()
