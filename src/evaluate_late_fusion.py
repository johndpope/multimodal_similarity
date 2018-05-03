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

def main():

    cfg = EvalConfig().parse()
    np.random.seed(seed=cfg.seed)

    test_session = cfg.test_session
    test_set = prepare_dataset(cfg.feature_root, test_session, cfg.feat, cfg.label_root)

    ####################### Load models here ########################

    input_ph = tf.placeholder(tf.float32, shape=[None, cfg.num_seg, None, None, None])
    dropout_ph = tf.placeholder(tf.float32, shape=[])

    with tf.variable_scope("modality_core"):
        # load backbone model
        if cfg.network == "convtsn":
            model_emb = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "convrtsn":
            model_emb = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        else:
            raise NotImplementedError

        model_emb.forward(input_ph, dropout_ph)    # for lstm has variable scope

        var_list = {}
        for v in tf.global_variables():
            if v.op.name.startswith("modality_core"):
                var_list[v.op.name.replace("modality_core/","")] = v
        restore_saver = tf.train.Saver(var_list)

    with tf.variable_scope("modality_sensors"):
        sensors_emb_dim = 128
        if cfg.network == "convtsn":
            model_emb_sensors = networks.ConvTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        elif cfg.network == "convrtsn":
            model_emb_sensors = networks.ConvRTSN(n_seg=cfg.num_seg, emb_dim=cfg.emb_dim)
        else:
            raise NotImplementedError
        model_output_sensors = networks.OutputLayer(n_input=sensors_emb_dim, n_output=8)

        model_emb_sensors.forward(input_ph, dropout_ph)
        model_output_sensors.forward(tf.nn.relu(model_emb_sensors.hidden), dropout_ph)

        var_list = {}
        for v in tf.global_variables():
            if v.op.name.startswith("modality_sensors"):
                var_list[v.op.name.replace("modality_sensors/","")] = v
        restore_saver_sensors = tf.train.Saver(var_list)

    ############################# Forward Pass #############################

    # get embeddings
    embedding = tf.nn.l2_normalize(model_emb.hidden, axis=-1, epsilon=1e-10)
    if cfg.use_output:
        if cfg.normalized:
            embedding_sensors = tf.nn.l2_normalize(model_output_sensors.logits)
        else:
            embedding_sensors = model_output_sensors.logits
    else:
        embedding_sensors = tf.nn.l2_normalize(model_emb_sensors.hidden, axis=-1, epsilon=1e-10)

    #########################################################################

    # Testing
    if cfg.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        # load the model (note that model_path already contains snapshot number
        restore_saver.restore(sess, cfg.model_path)
        print ("Restoring the model: {}".format(os.path.basename(cfg.model_path)))
        restore_saver_sensors.restore(sess, cfg.sensors_path)
        print ("Restoring the model: {}".format(os.path.basename(cfg.sensors_path)))

        eve_embeddings = []
        sensors_embeddings = []
        labels = []
        for i, session in enumerate(test_set):
            session_id = os.path.basename(session[1]).split('_')[0]
            print ("{0} / {1}: {2}".format(i, len(test_set), session_id))

            eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], model_emb.prepare_input_test)    # use prepare_input_test for testing time

            emb, emb_s = sess.run([embedding, embedding_sensors], 
                    feed_dict={input_ph: eve_batch, dropout_ph: 1.0})

            eve_embeddings.append(emb)
            sensors_embeddings.append(emb_s)
            labels.append(lab_batch)

        eve_embeddings = np.concatenate(eve_embeddings, axis=0)
        sensors_embeddings = np.concatenate(sensors_embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

    # evaluate the results
    fused_embeddings = np.concatenate((eve_embeddings, sensors_embeddings), axis=1)
    mAP, mAP_event, mPrec, confusion, count, recall = evaluate(fused_embeddings, np.squeeze(labels))

    mAP_macro = 0.0
    for key in mAP_event:
        mAP_macro += mAP_event[key]
    mAP_macro /= len(list(mAP_event.keys()))

    print ("%d events with dim %d for evaluation." % (labels.shape[0], fused_embeddings.shape[1]))
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

