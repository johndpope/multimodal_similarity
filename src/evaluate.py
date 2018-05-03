import sys
import os
import numpy as np
import pickle as pkl
import pdb

sys.path.append('../')
from configs.eval_config import EvalConfig
import networks
from utils import evaluate, mean_pool_input, max_pool_input
from data_io import load_data_and_label
from preprocess.label_transfer import honda_num2labels

def prepare_dataset(data_dir, sessions, feat, label_dir=None):

    if feat == 'resnet':
        appendix = '.npy'
    elif feat == 'sensor':
        appendix = '_sensors_normalized.npy'
    elif feat == 'sensor_sae':
        appendix = '_sensors_normalized_sae.npy'
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
    print ("Evaluate the feat: {}".format(cfg.feat))
    np.random.seed(seed=cfg.seed)

    test_session = cfg.test_session
    test_set = prepare_dataset(cfg.feature_root, test_session, cfg.feat, cfg.label_root)

    if cfg.preprocess_func == "mean":
        preprocess_func = mean_pool_input
    elif cfg.preprocess_func == "max":
        preprocess_func = max_pool_input


    eve_embeddings = []
    labels = []
    for i, session in enumerate(test_set):
        session_id = os.path.basename(session[1]).split('_')[0]
        print ("{0} / {1}: {2}".format(i, len(test_set), session_id))

        eve_batch, lab_batch, _ = load_data_and_label(session[0], session[1], preprocess_func)

        eve_embeddings.append(eve_batch)
        labels.append(lab_batch)

    eve_embeddings = np.concatenate(eve_embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # evaluate the results
    print ("%d events for evaluation with dimension %d." % (labels.shape[0], eve_embeddings.shape[1]))
    mAP, mAP_event = evaluate(eve_embeddings, labels, normalize=True)#, standardize=True)

    print ("mAP = {}".format(mAP))
#    label_dict = pkl.load(open(cfg.label_root+'label_goal.pkl','rb'))
    keys = list(mAP_event.keys())
    keys = sorted(keys)
    for key in keys:
        print ("Event {2}: {0}, mAP = {1}".format(honda_num2labels[key],
            mAP_event[key], key))

if __name__ == '__main__':
    main()

