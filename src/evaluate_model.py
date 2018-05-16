import sys
import os
import tensorflow as tf
import numpy as np
import pickle as pkl
import time

sys.path.append('../')
from configs.eval_config import EvalConfig
import networks
from utils import evaluate, mean_pool_input
from data_io import load_data_and_label, prepare_dataset
from preprocess.label_transfer import honda_num2labels, stimuli_num2labels
import pdb

def main():

    cfg = EvalConfig().parse()
    print ("Evaluate the model: {}".format(os.path.basename(cfg.model_path)))
    np.random.seed(seed=cfg.seed)

    test_session = cfg.test_session
    test_set = prepare_dataset(cfg.feature_root, test_session, cfg.feat, cfg.label_root, cfg.label_type)

    # load backbone model
    if cfg.network == "tsn":
        model = networks.TSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
    elif cfg.network == "rtsn":
        model = networks.RTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim, n_input=cfg.n_input)
    elif cfg.network == "convtsn":
        model = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
    elif cfg.network == "convrtsn":
        model = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim, n_h=cfg.n_h, n_w=cfg.n_w, n_C=cfg.n_C, n_input=cfg.n_input)
    elif cfg.network == "seq2seqtsn":
        model = networks.Seq2seqTSN(n_seg=cfg.num_seg, n_input=n_input, emb_dim=cfg.emb_dim, reverse=cfg.reverse)
    elif cfg.network == "convbirtsn":
        model = networks.ConvBiRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
    else:
        raise NotImplementedError


    # get the embedding
    if cfg.feat == "sensors" or cfg.feat == "segment":
        input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None])
    elif cfg.feat == "resnet" or cfg.feat == "segment_down":
        input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
    dropout_ph = tf.placeholder(tf.float32, shape=[])
    model.forward(input_ph, dropout_ph)
    embedding = tf.nn.l2_normalize(model.hidden, axis=1, epsilon=1e-10, name='embedding')

    # Testing
    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # restore variables
    var_list = {}
    for v in tf.global_variables():
        var_list[cfg.variable_name+v.op.name] = v

    saver = tf.train.Saver(var_list)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        # load the model (note that model_path already contains snapshot number
        saver.restore(sess, cfg.model_path)

        duration = 0.0
        eve_embeddings = []
        labels = []
        for i, session in enumerate(test_set):
            session_id = os.path.basename(session[1]).split('_')[0]
            print ("{0} / {1}: {2}".format(i, len(test_set), session_id))

#            eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], mean_pool_input, transfer=cfg.transfer)    # use prepare_input_test for testing time
            eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], model.prepare_input_test, transfer=cfg.transfer)    # use prepare_input_test for testing time

            start_time = time.time()
            emb = sess.run(embedding, feed_dict={input_ph: eve_batch, dropout_ph: 1.0})
#            emb = eve_batch
            duration += time.time() - start_time

            eve_embeddings.append(emb)
            labels.append(lab_batch)

        eve_embeddings = np.concatenate(eve_embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

    # evaluate the results
    mAP, mAP_event, mPrec, confusion, count, recall = evaluate(eve_embeddings, np.squeeze(labels))

    mAP_macro = 0.0
    for key in mAP_event:
        mAP_macro += mAP_event[key]
    mAP_macro /= len(list(mAP_event.keys()))

    print ("%d events with dim %d for evaluation, run time: %.3f." % (labels.shape[0], eve_embeddings.shape[1], duration))
    print ("mAP = {:.4f}".format(mAP))
    print ("mAP_macro = {:.4f}".format(mAP_macro))
    print ("mPrec@0.5 = {:.4f}".format(mPrec))
    print ("Recall@1 = {:.4f}".format(recall[0]))
    print ("Recall@2 = {:.4f}".format(recall[1]))
    print ("Recall@4 = {:.4f}".format(recall[2]))
    print ("Recall@8 = {:.4f}".format(recall[3]))
    print ("Recall@16 = {:.4f}".format(recall[4]))
    print ("Recall@32 = {:.4f}".format(recall[5]))

    if cfg.label_type == 'goal':
        num2labels = honda_num2labels
    elif cfg.label_type == 'stimuli':
        num2labels = stimuli_num2labels

    keys = confusion['labels']
    for i, key in enumerate(keys):
        if key not in mAP_event:
            continue
        print ("Event {0}: {1}, ratio = {2:.4f}, mAP = {3:.4f}, mPrec@0.5 = {4:.4f}".format(
            key,
            num2labels[key],
            float(count[i]) / np.sum(count),
            mAP_event[key],
            confusion['confusion_matrix'][i, i]))

    # store results
    pkl.dump({"mAP": mAP,
              "mAP_macro": mAP_macro,
              "mAP_event": mAP_event,
              "mPrec": mPrec,
              "confusion": confusion,
              "count": count,
              "recall": recall},
              open(os.path.join(os.path.dirname(cfg.model_path), "results.pkl"), 'wb'))

if __name__ == '__main__':
    main()

