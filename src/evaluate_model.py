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

    test_session = cfg.test_session
    test_set = prepare_dataset(cfg.feature_root, test_session, cfg.feat, cfg.label_root)

    # load backbone model
    if cfg.network == "tsn":
        model = networks.ConvTSN(n_seg=cfg.num_seg)

    # get the embedding
    input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
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

        eve_embeddings = []
        labels = []
        for i, session in enumerate(test_set):
            session_id = os.path.basename(session[1]).split('_')[0]
            print ("{0} / {1}: {2}".format(i, len(test_set), session_id))

            eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], model.prepare_input_test)    # use prepare_input_test for testing time

            emb = sess.run(embedding, feed_dict={input_ph: eve_batch})

            eve_embeddings.append(emb)
            labels.append(lab_batch)

        eve_embeddings = np.concatenate(eve_embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

    # evaluate the results
    mAP, mAP_event = evaluate(eve_embeddings, labels)


    print ("%d events for evaluation." % labels.shape[0])
    print ("mAP = {}".format(mAP))
    keys = list(mAP_event.keys())
    keys = sorted(keys)
    for key in keys:
        print ("Event {2}: {0}, mAP = {1}".format(honda_num2labels[key],
            mAP_event[key], key))

if __name__ == '__main__':
    main()

