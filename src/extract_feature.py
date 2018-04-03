"""
Extract features using pretrained model
"""

import sys
import os
import tensorflow as tf
import numpy as np
import pickle as pkl

sys.path.append('../')
from configs.eval_config import EvalConfig
import networks
from utils import evaluate
from data_io import load_data_and_label
from preprocess.honda_labels import honda_labels2num, honda_num2labels

def prepare_dataset(data_dir, sessions, feat, label_dir=None):

    if feat == 'resnet':
        appendix = '.npy'
    elif feat == 'sensor':
        appendix = '_sensors_normalized.npy'
    elif feat == 'segment':
        appendix = '_seg_output.npy'
    else:
        raise NotImplementedError

    dataset = []
    for sess in sessions:
        feat_path = os.path.join(data_dir, sess+appendix)
        label_path = os.path.join(label_dir, sess+'_goal.pkl')

        dataset.append((feat_path, label_path))

    return dataset

def main():

    cfg = EvalConfig().parse()
    print ("Evaluate the model: {}".format(os.path.basename(cfg.model_path)))
    np.random.seed(seed=cfg.seed)

    all_session = cfg.all_session
    all_set = prepare_dataset(cfg.feature_root, all_session, cfg.feat, cfg.label_root)

    # load backbone model
    if cfg.network == "tsn":
        model = networks.ConvTSN(n_seg=cfg.num_seg)
    elif cfg.network == "sae":
        # TODO: change 8 to general cases
        n_input = 8
        model = networks.SAE(n_input=n_input, emb_dim=cfg.emb_dim)

    # get the embedding
    input_ph = tf.placeholder(tf.float32, shape=[None, n_input])
    model.forward(input_ph)
    embedding = tf.nn.l2_normalize(model.hidden, dim=1, epsilon=1e-10, name='embedding')

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

        for i, session in enumerate(all_set):
            session_id = os.path.basename(session[1]).split('_')[0]
            print ("{0} / {1}: {2}".format(i, len(all_set), session_id))

            eve = np.load(session[0], 'r')
            eve_embedding = np.zeros((eve.shape[0], cfg.emb_dim), dtype='float32')
            for start, end in zip(range(0, eve.shape[0], cfg.batch_size),
                                range(cfg.batch_size, eve.shape[0]+cfg.batch_size, cfg.batch_size)):
                end = min(end, eve.shape[0])
                emb = sess.run(embedding, feed_dict={input_ph: eve[start:end]})
                eve_embedding[start:end] = emb

            np.save(session[0].replace('.npy', '_sae.npy'), eve_embedding)

if __name__ == '__main__':
    main()

