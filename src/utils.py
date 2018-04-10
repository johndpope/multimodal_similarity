import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score
import pdb
import os
from six import iteritems

def optimize(loss, global_step, optimizer, learning_rate, update_gradient_vars, log_histograms=True):

    if optimizer == 'ADAGRAD':
        opt = tf.train.Adagradoptimizer(learning_rate)
    elif optimizer == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOMENTUM':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        opt = tf.train.GradientDescentOptimizer(learning_rate)

    grads = opt.compute_gradients(loss, update_gradient_vars)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # add histograms for trainable variables
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # add histograms for gradients
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/gradients', grad)

    return apply_gradient_op

def retrieve_one(query, database, query_label=None, labels=None, normalize=False):
    """
    Retrieve from the database using given query
    Return AP if label is not None

    query -- float32, [emb_dim,]
    database -- float32, [N, emb_dim]
    query_label -- int32
    label -- int32, [N,]
    normalize -- bool, if true, normalize to unit vector
    """

    N, dim = database.shape
    if normalize:
        query /= np.linalg.norm(output)
        database /= np.linalg.norm(database, axis=1).reshape(-1,1)

    # Euclidean distance
    dist = np.linalg.norm(query.reshape(1,-1) - database, axis=1)
    idx = np.argsort(dist)

    ap = None
    if labels is not None:
        ap = average_precision_score(np.squeeze(labels==query_label),
                np.squeeze(np.max(dist) - dist))    # convert distance to score

    return dist, idx, ap

def evaluate_simple(embeddings, labels, normalize=False, standardize=False, alpha=0.5):
    """
    A simple version with only mean output

    Evaluate a given dataset with embeddings and labels
    Loop for each element as query and the rest as database
    Calculate the mean AP and mean Precision@recall

    embeddings -- float32, [N, emb_dim]
    labels -- int32, [N, ]
    normalize -- bool, whether to normalize feature to unit vector
    standardize -- bool, whether to standardize each dimension to be zero mean and unit variance
    alpha -- float, used for precision @ recall alpha
    """

    N, dim = embeddings.shape
    if normalize:
        embeddings /= np.linalg.norm(embeddings, axis=1).reshape(-1,1)
    if standardize:
        mu = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0) + np.finfo(float).tiny
        embeddings = (embeddings - mu) / std

    labels = np.squeeze(labels)
    unique_labels = sorted(set(labels.tolist()))

    aps = []
    lab = []
    precs = []
    for i in range(N):
        if labels[i] > 0:    # only for foreground events
            _, sorted_idx, ap = retrieve_one(embeddings[i], np.delete(embeddings,i,0),
                                             labels[i], np.delete(labels,i))

            if np.isnan(ap):
                print ("WARNING: encountered an AP of NaN!")
                print ("This may occur when the event only appears once.")
                print ("The event label here is {}.".format(labels[i]))
                print ("Ignore this event and carry on.")
            else:
                aps.append(ap)
                lab.append(int(labels[i]))

                # compute precision @ recall alpha (precisions are for all classes)
                prec, _ = precision_at_recall(labels[sorted_idx], labels[i], alpha)
                precs.append(prec)


    mAP = np.mean(aps)
    mPrec = np.mean(precs)

    return mAP, mPrec
    
def evaluate(embeddings, labels, normalize=False, standardize=False, alpha=0.5):
    """
    Evaluate a given dataset with embeddings and labels
    Loop for each element as query and the rest as database
    Calculate the mean AP and mean Precision@recall

    embeddings -- float32, [N, emb_dim]
    labels -- int32, [N, ]
    normalize -- bool, whether to normalize feature to unit vector
    standardize -- bool, whether to standardize each dimension to be zero mean and unit variance
    alpha -- float, used for precision @ recall alpha
    """

    N, dim = embeddings.shape
    if normalize:
        embeddings /= np.linalg.norm(embeddings, axis=1).reshape(-1,1)
    if standardize:
        mu = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0) + np.finfo(float).tiny
        embeddings = (embeddings - mu) / std

    labels = np.squeeze(labels)
    unique_labels = sorted(set(labels.tolist()))

    aps = []
    lab = []
    precs = []
    confs = []
    num_correct = [0, 0, 0]    # for K = 1, 10, 100
    for i in range(N):
        if labels[i] > 0:    # only for foreground events
            _, sorted_idx, ap = retrieve_one(embeddings[i], np.delete(embeddings,i,0),
                                             labels[i], np.delete(labels,i))

            if np.isnan(ap):
                print ("WARNING: encountered an AP of NaN!")
                print ("This may occur when the event only appears once.")
                print ("The event label here is {}.".format(labels[i]))
                print ("Ignore this event and carry on.")
            else:
                aps.append(ap)
                lab.append(int(labels[i]))

                # compute precision @ recall alpha (precisions are for all classes)
                prec, conf = precision_at_recall(labels[sorted_idx], labels[i], alpha)
                precs.append(prec)
                confs.append(conf)

                # compute recall @ K
                num_correct[0] += recall_at_K(labels[sorted_idx], labels[i], 1)
                num_correct[1] += recall_at_K(labels[sorted_idx], labels[i], 10)
                num_correct[2] += recall_at_K(labels[sorted_idx], labels[i], 100)

    mAP = np.mean(aps)
    mPrec = np.mean(precs)

    # get mAP for each event
    mAP_event = {}
    for ap, l in zip(aps, lab):
        if l not in mAP_event:
            mAP_event[l] = [ap]
        else:
            mAP_event[l].append(ap)
    for key in mAP_event:
        mAP_event[key] = np.mean(mAP_event[key])

    # get confusion matrix
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype='float32')
    count = np.zeros((len(unique_labels), 1), dtype='int32')
    for conf, l in zip(confs, lab):
        row = unique_labels.index(l)
        for key in conf:
            column = unique_labels.index(key)
            confusion_matrix[row, column] += conf[key]
        count[row] += 1

    confusion_matrix[1:] /= count[1:]
    confusion = {"confusion_matrix": confusion_matrix, "labels": unique_labels}

    # get recall @ K
    recall = [float(num) / len(lab) for num in num_correct]

    return mAP, mAP_event, mPrec, confusion, count, recall

def precision_at_recall(label_list, query_label, alpha=0.5):
    """
    Computer precision for all classes at recall alpha fro query label
    
    label_list -- [N,], sorted label_list accroding to ascending distance
    query_label -- class for query
    alpha -- 0~1, recall
    """

    num_this_label = np.sum(label_list == query_label)
    num_recall_alpha = int(alpha * num_this_label)

    unique_labels = sorted(set(label_list.tolist()))
    prec_dict = dict.fromkeys(unique_labels, 0)

    for i in range(label_list.shape[0]):
        prec_dict[label_list[i]] += 1

        if prec_dict[query_label] == num_recall_alpha:
            break

    for key in prec_dict:
        prec_dict[key] /= (i+1)

    return prec_dict[query_label], prec_dict

def recall_at_K(label_list, query_label, K=10):
    """
    reference: https://github.com/rksltnl/Deep-Metric-Learning-CVPR16/blob/master/code/evaluation/evaluate_recall.m
    """

    knn_label = label_list[:K]
    if np.sum(knn_label == query_label) > 0:
        return 1
    else:
        return 0

def mean_pool_input(feat, flatten=True):
    """
    Mean pooling as preprocess_func
    feat -- feature sequence, [time_steps, (dims)]
    """

    new_feat = np.mean(feat, axis=0)
    if flatten:
        new_feat = new_feat.flatten()
    return np.expand_dims(new_feat, 0)

def max_pool_input(feat, flatten=True):
    """
    Maxpooling as preprocess_func
    feat -- feature sequence, [time_steps, (dims)]
    """

    new_feat = np.max(feat, axis=0)
    if flatten:
        new_feat = new_feat.flatten()
    return np.expand_dims(new_feat, 0)

def all_diffs_tf(a, b):
    """
    Return a tensor of all combinations of a - b

    a -- [batch_size1, dim]
    b -- [batch_size2, dim]

    reference: https://github.com/VisualComputingInstitute/triplet-reid
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)

def all_diffs(a, b):
    """
    Return a tensor of all combinations of a - b

    a -- [batch_size1, dim]
    b -- [batch_size2, dim]

    reference: https://github.com/VisualComputingInstitute/triplet-reid
    """
    return np.expand_dims(a, axis=1) - np.expand_dims(b, axis=0)

def cdist(diff, metric='squaredeuclidean'):
    """
    Return the distance according to metric

    diff -- [..., dim], difference matrix
    metric  --   "squaredeuclidean": squared euclidean
                 "euclidean": euclidean (without squared)
                 "l1": manhattan distance
    """

    if metric == "squaredeuclidean":
        return np.sum(np.square(diff), axis=-1)
    elif metric == "euclidean":
        return np.sqrt(np.sum(np.square(diff), axis=-1) + 1e-12)
    elif metric == "l1":
        return np.sum(np.abs(diff), axis=-1)
    else:
        raise NotImplementedError

def cdist_tf(diff, metric='squaredeuclidean'):
    """
    Return the distance according to metric

    diff -- [..., dim], difference matrix
    metric  --   "squaredeuclidean": squared euclidean
                 "euclidean": euclidean (without squared)
                 "l1": manhattan distance
    """

    if metric == "squaredeuclidean":
        return tf.reduce_sum(tf.square(diff), axis=-1)
    elif metric == "euclidean":
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-12)
    elif metric == "l1":
        return tf.reduce_sum(tf.abs(diff), axis=-1)
    else:
        raise NotImplementedError

def rnn_prepare_input(max_time, feat):
    """
    feat -- feature sequence, [time_steps, n_h, n_w, n_input]
    """

    new_feat = np.zeros((max_time,)+feat.shape[1:], dtype='float32')
    if feat.shape[0] > max_time:
        new_feat = feat[:max_time]
    else:
        new_feat[:feat.shape[0]] = feat

    return np.expand_dims(new_feat, 0)

def rnn_prepare_input_tf(max_time, feat):

    new_feat = tf.zeros((max_time,)+tf.shape(feat)[1:], dtype=tf.float32)
    if tf.shape(feat)[0] > max_time:
        new_feat = feat[:max_time]
    else:
        new_feat[:tf.shape(feat)[0]] = feat

    return tf.expand_dims(new_feat, 0)


def tsn_prepare_input(n_seg, feat):
    """
    feat -- feature sequence, [time_steps, n_h, n_w, n_input]
    """

    # reference: TSN pytorch codes
    average_duration = feat.shape[0] // n_seg
    if average_duration > 0:
        offsets = np.multiply(range(n_seg), average_duration) + np.random.randint(average_duration, size=n_seg)
    else:
        raise NotImplementedError
    feat = feat[offsets].astype('float32')

    return np.expand_dims(feat, 0)

def tsn_prepare_input_test(n_seg, feat):
    """
    For testing time, no sampling
    feat -- feature sequence, [time_steps, n_h, n_w, n_input]
    """

    average_duration = feat.shape[0] // n_seg
    offsets = np.array([int(average_duration / 2.0 + average_duration * x) for x in range(n_seg)])
    feat = feat[offsets].astype('float32')

    return np.expand_dims(feat, 0)

def tsn_prepare_input_tf(n_seg, feat):
    """
    tensorflow version
    """

    average_duration = tf.floordiv(tf.shape(feat)[0], n_seg)
    offsets = tf.add(tf.multiply(tf.range(n_seg,dtype=tf.int32), average_duration),
                    tf.random_uniform(shape=(1,n_seg),maxval=average_duration,dtype=tf.int32))
    # offset should be column vector, use reshape
    return tf.gather_nd(feat, tf.reshape(offsets, [-1,1]))

def write_configure_to_file(cfg, result_dir):
    with open(os.path.join(result_dir, 'config.txt'), 'w') as fout:
        for key, value in iteritems(vars(cfg)):
            fout.write('%s: %s\n' % (key, str(value)))

