
from datetime import datetime
import os
import time
import sys
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.losses.python import metric_learning as metric_loss_ops
import numpy as np
import itertools
import random
import pdb
from six import iteritems
import glob
from PIL import Image

sys.path.append('../')
from configs.train_config import TrainConfig
from data_io import session_generator, load_data_and_label, prepare_dataset
import networks
import utils


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
    images_root = '/mnt/work/CUB_200_2011/images/'
    with open('/mnt/work/CUB_200_2011/images.txt', 'r') as fin:
        image_files = fin.read().strip().split('\n')
    with open('/mnt/work/CUB_200_2011/image_class_labels.txt', 'r') as fin:
        labels = fin.read().strip().split('\n')

    train_files = []
    train_labels = []
    val_files = []
    val_labels = []
    for i in range(len(image_files)):
        label = int(labels[i].split(' ')[1])
        if label <= 100:
            train_files.append(images_root+image_files[i].split(' ')[1])
            train_labels.append(label)
        else:
            val_files.append(images_root+image_files[i].split(' ')[1])
            val_labels.append(label)

    class_idx_dict = {}
    for i, l in enumerate(train_labels):
        l = int(l)
        if l not in class_idx_dict:
            class_idx_dict[l] = [i]
        else:
            class_idx_dict[l].append(i)
    C = len(list(class_idx_dict.keys()))

    val_images = np.zeros((len(val_files), 256, 256, 3), dtype=np.uint8)
    for i in range(len(val_files)):
        img = Image.open(val_files[i]).convert('RGB').resize((256,256))
        val_images[i] = np.array(img)

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
        model_emb = networks.CUBLayer(n_input=1024, n_output=cfg.emb_dim)

        # get the embedding
        input_ph = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        label_ph = tf.placeholder(tf.int32, shape=[None])
        dropout_ph = tf.placeholder(tf.float32, shape=[])

        pool5 = networks.Inception_V2(input_ph)
        model_emb.forward(pool5, dropout_ph)
        if cfg.normalized:
            embedding = tf.nn.l2_normalize(model_emb.logits, axis=-1, epsilon=1e-10)
        else:
            embedding = model_emb.logits

        # variable for visualizing the embeddings
        emb_var = tf.Variable([0.0], name='embeddings')
        set_emb = tf.assign(emb_var, embedding, validate_shape=False)

        # calculated for monitoring all-pair embedding distance
        diffs = utils.all_diffs_tf(embedding, embedding)
        all_dist = utils.cdist_tf(diffs)
        tf.summary.histogram('embedding_dists', all_dist)

        # use tensorflow implementation...
        if cfg.loss == 'triplet':
            metric_loss = metric_loss_ops.triplet_semihard_loss(
                          labels=label_ph,
                          embeddings=embedding,
                          margin=cfg.alpha)
        elif cfg.loss == 'lifted':
            metric_loss = metric_loss_ops.lifted_struct_loss(
                          labels=label_ph,
                          embeddings=embedding,
                          margin=cfg.alpha)
        elif cfg.loss == 'mylifted':
            metric_loss, num_active, diff, weights, fp, cn = networks.lifted_loss(all_dist, label_ph, cfg.alpha, weighted=False)

        else:
            raise NotImplementedError

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = metric_loss + regularization_loss * cfg.lambda_l2

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

                image_batch = np.zeros((len(idx_batch), 256, 256, 3), dtype=np.uint8)
                lab_batch = np.zeros((len(idx_batch), ), dtype=np.int32)
                for i, idx in enumerate(idx_batch):
                    # load image with random flipping
                    if np.random.rand() < 0.5:
                        img = Image.open(train_files[idx]).convert('RGB').resize((256,256)).transpose(Image.FLIP_LEFT_RIGHT)
                    else:
                        img = Image.open(train_files[idx]).convert('RGB').resize((256,256))
                    image_batch[i] = np.array(img)
                    lab_batch[i] = train_labels[idx]

                pdb.set_trace()
                # perform training on the selected triplets
                err, _, step, summ = sess.run([total_loss, train_op, global_step, summary_op],
                                feed_dict = {input_ph: image_batch,
                                            label_ph: lab_batch,
                                            dropout_ph: cfg.keep_prob,
                                            lr_ph: learning_rate})

                print ("%s\tEpoch: %d\tImages num: %d\tLoss %.4f" % \
                        (cfg.name, epoch+1, feat_batch.shape[0], err))

                summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=err),
                        tf.Summary.Value(tag="images_num", simple_value=feat_batch.shape[0])])
                summary_writer.add_summary(summary, step)
                summary_writer.add_summary(summ, step)

                # validation on val_set
                if (epoch+1) % 1000 == 0:
                    val_embeddings, _ = sess.run([embedding,set_emb], feed_dict={input_ph: val_images, label_ph:val_labels, dropout_ph: 1.0})
                    mAP, mPrec, recall = utils.evaluate_simple(val_embeddings, val_labels)
                    summary = tf.Summary(value=[tf.Summary.Value(tag="Valiation mAP", simple_value=mAP),
                                        tf.Summary.Value(tag="Validation Recall@1", simple_value=recall),
                                        tf.Summary.Value(tag="Validation mPrec@0.5", simple_value=mPrec)])
                    print ("Epoch: [%d]\tmAP: %.4f\trecall: %.4f" % (epoch+1,mAP,recall))

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
