import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score
import pdb

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
    
def evaluate(embeddings, labels, normalize=False, standardize=False):
    """
    Evaluate a given dataset with embeddings and labels
    Loop for each element as query and the rest as database
    Calculate the mean AP

    embeddings -- float32, [N, emb_dim]
    labels -- int32, [N, ]
    normalize -- bool, whether to normalize feature to unit vector
    standardize -- bool, whether to standardize each dimension to be zero mean and unit variance
    """

    N, dim = embeddings.shape
    if normalize:
        embeddings /= np.linalg.norm(embeddings, axis=1).reshape(-1,1)
    if standardize:
        mu = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0) + np.finfo(float).tiny
        embeddings = (embeddings - mu) / std

    aps = []
    lab = []
    for i in range(N):
        if labels[i] > 0:    # only for foreground events
            _, _, ap = retrieve_one(embeddings[i], np.delete(embeddings,i,0),
                                    labels[i], np.delete(labels,i))

            if np.isnan(ap):
                print ("WARNING: encountered an AP of NaN!")
                print ("This may occur when the event only appears once.")
                print ("The event label here is {}.".format(labels[i]))
                print ("Ignore this event and carry on.")
            else:
                aps.append(ap)
                lab.append(int(labels[i]))
    mAP = np.mean(aps)

    # get mAP for each event
    mAP_event = {}
    for ap, l in zip(aps, lab):
        if l not in mAP_event:
            mAP_event[l] = [ap]
        else:
            mAP_event[l].append(ap)
    for key in mAP_event:
        mAP_event[key] = np.mean(mAP_event[key])

    return mAP, mAP_event

def mean_pool_input(feat, flatten=True):
    """
    Mean pooling as preprocess_func
    feat -- feature sequence, [time_steps, (dims)]
    """

    new_feat = np.mean(feat, axis=0)
    if flatten:
        return new_feat.flatten()
    else:
        return new_feat

def max_pool_input(feat, flatten=True):
    """
    Maxpooling as preprocess_func
    feat -- feature sequence, [time_steps, (dims)]
    """

    new_feat = np.max(feat, axis=0)
    if flatten:
        return new_feat.flatten()
    else:
        return new_feat

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
