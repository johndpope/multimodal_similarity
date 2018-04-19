import sys
import os
import tensorflow as tf
import numpy as np
import pickle as pkl

sys.path.append('../')
from configs.eval_config import EvalConfig
import networks
from utils import evaluate
from data_io import load_data_and_label, prepare_dataset
from preprocess.honda_labels import honda_labels2num, honda_num2labels
import pdb

def main():

    cfg = EvalConfig().parse()
    print ("Evaluate the model: {}".format(os.path.basename(cfg.model_path)))
    np.random.seed(seed=cfg.seed)

    test_session = cfg.test_session
    test_set = prepare_dataset(cfg.feature_root, test_session, cfg.feat, cfg.label_root)

    n_input = cfg.feat_dim[cfg.feat]
    # load backbone model
    if cfg.network == "tsn":
        model = networks.TSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
    elif cfg.network == "rtsn":
        model = networks.RTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
    elif cfg.network == "convtsn":
        model = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
    elif cfg.network == "convrtsn":
        model = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
    elif cfg.network == "seq2seqtsn":
        model = networks.Seq2seqTSN(n_seg=cfg.num_seg, n_input=n_input, emb_dim=cfg.emb_dim, reverse=cfg.reverse)
    else:
        raise NotImplementedError


    # get the embedding
    if cfg.feat == "sensors":
        input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None])
    elif cfg.feat == "resnet":
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
        var_list[cfg.variable_name+'/'+v.op.name] = v

    pdb.set_trace()
    saver = tf.train.Saver(var_list)
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

            emb = sess.run(embedding, feed_dict={input_ph: eve_batch, dropout_ph: 1.0})

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

    print ("%d events for evaluation." % labels.shape[0])
    print ("mAP = {}".format(mAP))
    print ("mAP_macro = {}".format(mAP_macro))
    print ("mPrec@0.5 = {}".format(mPrec))
    print ("Recall@1 = {}, Recall@10 = {}, Recall@100 = {}".format(recall[0], recall[1], recall[2]))

    keys = confusion['labels']
    for i, key in enumerate(keys):
        if key not in mAP_event:
            continue
        print ("Event {0}: {1}, ratio = {2}, mAP = {3}, mPrec@0.5 = {4}".format(
            key,
            honda_num2labels[key],
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

