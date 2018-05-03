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
from preprocess.label_transfer import honda_num2labels
import pdb

def main():

    cfg = EvalConfig().parse()
    print ("Evaluate the model: {}".format(os.path.basename(cfg.model_path)))
    np.random.seed(seed=cfg.seed)

    test_session = cfg.test_session
    test_set = prepare_dataset(cfg.feature_root, test_session, cfg.feat, cfg.label_root)


    # get the embedding
    with tf.variable_scope("modality_core"):
        # load backbone model
        if cfg.network == "convtsn":
            model_emb = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "convrtsn":
            model_emb = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        else:
            raise NotImplementedError

        input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
        dropout_ph = tf.placeholder(tf.float32, shape=[])
        model_emb.forward(input_ph, dropout_ph)    # for lstm has variable scope

    with tf.variable_scope("hallucination_sensors"):
        sensors_emb_dim = 32
        # load backbone model
        if cfg.network == "convtsn":
            hal_emb_sensors = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=sensors_emb_dim)
        elif cfg.network == "convrtsn":
            hal_emb_sensors = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=sensors_emb_dim)
        else:
            raise NotImplementedError

        hal_emb_sensors.forward(input_ph, dropout_ph)    # for lstm has variable scope

    # Core branch
    if cfg.normalized:
        embedding = tf.nn.l2_normalize(model_emb.hidden, axis=-1, epsilon=1e-10)
        embedding_hal_sensors = tf.nn.l2_normalize(hal_emb_sensors.hidden, axis=-1, epsilon=1e-10)
    else:
        embedding = model_emb.hidden
        embedding_hal_sensors = hal_emb_sensors.hidden

    embedding_fused = tf.concat((embedding, embedding_hal_sensors), axis=1)


    # Testing
    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver(tf.global_variables())
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

            eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], model_emb.prepare_input_test)    # use prepare_input_test for testing time

            start_time = time.time()
            emb = sess.run(embedding_fused, feed_dict={input_ph: eve_batch, dropout_ph: 1.0})
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
    print ("Recall@1 = {:.4f}, Recall@10 = {:.4f}, Recall@100 = {:.4f}".format(recall[0], recall[1], recall[2]))

    keys = confusion['labels']
    for i, key in enumerate(keys):
        if key not in mAP_event:
            continue
        print ("Event {0}: {1}, ratio = {2:.4f}, mAP = {3:.4f}, mPrec@0.5 = {4:.4f}".format(
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

