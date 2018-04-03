import tensorflow as tf
import numpy as np
import utils
import functools


class SAE(object):
    """
    Stack autoencder for initializing data representation

    Use tied weights
    Denoising is not used because we don't have layer-wise pretraining
    """

    def name(self):
        return "SAE"

    def __init__(self, n_input=8, emb_dim=128):
        self.n_input = n_input
        self.emb_dim = emb_dim

        self.W_1 = tf.get_variable(name="W_1", shape=[self.n_input, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_1 = tf.get_variable(name="b_1", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.W_2 = tf.get_variable(name="W_2", shape=[self.emb_dim, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_2 = tf.get_variable(name="b_2", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        # for decoding
        self.b_3 = tf.get_variable(name="b_3", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.b_4 = tf.get_variable(name="b_4", shape=[self.n_input],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def forward(self, x):

        # encode
        h = tf.nn.relu(tf.nn.xw_plus_b(x, self.W_1, self.b_1))
        self.hidden = tf.nn.xw_plus_b(h, self.W_2, self.b_2)

        # decode
        h_recon = tf.nn.relu(tf.nn.xw_plus_b(self.hidden, tf.transpose(self.W_2), self.b_3))
        self.x_recon = tf.nn.xw_plus_b(h_recon, tf.transpose(self.W_1), self.b_4)



class PairSim(object):
    """ TripletSim layer
    Input a pair of sample A, B
    Ouput a binary classification: whether A, B is similar with each other

    One choice: first calculate distance (A-B)^2 then do prediction (reference: A Discriminatively Learned CNN Embedding for Person Re-identification)
    Or: concatenate two features then do prediction (reference: A Multi-Task Deep Network for Person Re-Identification)

    We choose the later one here.
    """

    def name(self):
        return "PairSim"

    def __init__(self, n_input=128):
        self.n_input = n_input

        self.W_pairwise = tf.get_variable(name="W_pairwise", shape=[self.n_input*2, self.n_input],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_pairwise = tf.get_variable(name="b_pairwise", shape=[self.n_input],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.W_o = tf.get_variable(name="W_o", shape=[self.n_input, 1],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_o = tf.get_variable(name="b_o", shape=[1],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

        def forward(self, x):
            """
            x -- feature pair, [batch_size, 2, n_input]
            """

            x_concat = tf.reshape(x, [-1, 2*self.n_input])

            h = tf.nn.xw_plus_b(x_concat, self.W_pairwise, self.b_pairwise)

            self.logits = tf.nn.xw_plus_b(h, self.W_o, self.b_o)    # for computing loss
            self.prob = tf.sigmoid(self.logits)


# TSN for temporal aggregation
class TSN(object):
    def name(self):
        return "TSN"

    def __init__(self, n_seg=3, emb_dim=256, n_input=1536, output_keep_prob=1.0):
        
        self.n_seg = n_seg
        self.n_input = n_input
        self.emb_dim = emb_dim
        self.output_keep_prob = output_keep_prob

        self.W_1 = tf.get_variable(name="W_1", shape=[self.n_input, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_1 = tf.get_variable(name="b_1", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.W_2 = tf.get_variable(name="W_2", shape=[self.emb_dim, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_2 = tf.get_variable(name="b_2", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def forward(self, x):
        """
        x -- input features, [batch_size, n_seg, n_input]
        """

        x_flat = tf.reshape(x, [-1, self.n_input])
        h1 = tf.nn.relu(tf.nn.xw_plus_b(x_flat, self.W_1, self.b_1))
        h1_drop = tf.nn.dropout(h1, self.output_keep_prob)

        h2 = tf.nn.xw_plus_b(h1_drop, self.W_2, self.b_2)
        h2_reshape = tf.reshape(h, [-1, self.n_seg, self.emb_dim])

        self.hidden = tf.reduce_mean(h_reshape, axis=1)

    def prepare_input(self, feat):
        return functools.partial(utils.tsn_prepare_input, self.n_seg)

    def prepare_input_tf(self, feat):
        return functools.partial(utils.tsn_prepare_input_tf, self.n_seg)

# Convolutional embedding + TSN for temporal aggregation
class ConvTSN(object):
    def name(self):
        return "ConvTSN"

    def __init__(self, n_seg=3, n_C=20, emb_dim=256, input_keep_prob=1.0, output_keep_prob=1.0, n_input=1536, n_h=8, n_w=8, n_output=11):
        
        self.n_seg = n_seg
        self.n_C = n_C
        self.n_h = n_h
        self.n_w = n_w
        self.n_input = n_input
        self.n_output = n_output
        self.emb_dim = emb_dim
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob

        self.W_emb = tf.get_variable(name="W_emb", shape=[1,1,n_input,self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.W = tf.get_variable(name="W", shape=[self.n_C*n_h*n_w, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b = tf.get_variable(name="b", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def forward(self, x):
        """
        x -- input features, [batch_size, n_seg, n_h, n_w, n_input]
        """

        x_flat = tf.reshape(x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1,1,1,1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [-1, self.n_h*self.n_w*self.n_C])
        h = tf.nn.xw_plus_b(x_emb, self.W, self.b)

        h_reshape = tf.reshape(h, [-1, self.n_seg, self.emb_dim])

        self.hidden = tf.reduce_mean(h_reshape, axis=1)

    def prepare_input(self, feat):
        return functools.partial(utils.tsn_prepare_input, self.n_seg)

    def prepare_input_tf(self, feat):
        return functools.partial(utils.tsn_prepare_input_tf, self.n_seg)


# Convolutional TSN for classification
class ConvTSNClassifier(object):
    def name(self):
        return "ConvTSN"

    def __init__(self, n_seg=3, n_C=20, emb_dim=256, input_keep_prob=1.0, output_keep_prob=1.0, n_input=1536, n_h=8, n_w=8, n_output=11):
        
        self.n_seg = n_seg
        self.n_C = n_C
        self.n_h = n_h
        self.n_w = n_w
        self.n_input = n_input
        self.n_output = n_output
        self.emb_dim = emb_dim
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob

        self.W_emb = tf.get_variable(name="W_emb", shape=[1,1,n_input,self.n_C],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.W = tf.get_variable(name="W", shape=[self.n_C*n_h*n_w, self.emb_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b = tf.get_variable(name="b", shape=[self.emb_dim],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.W_o = tf.get_variable(name="W_o", shape=[self.emb_dim, self.n_output],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_o = tf.get_variable(name="b_o", shape=[self.n_output],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

    def forward(self, x):
        """
        x -- input features, [batch_size, n_seg, n_h, n_w, n_input]
        """

        x_flat = tf.reshape(x, [-1, self.n_h, self.n_w, self.n_input])
        x_emb = tf.nn.relu(tf.nn.conv2d(input=x_flat, filter=self.W_emb,
                                        strides=[1,1,1,1], padding="VALID",
                                        data_format="NHWC"))
        x_emb = tf.reshape(x_emb, [-1, self.n_h*self.n_w*self.n_C])
        h = tf.nn.xw_plus_b(x_emb, self.W, self.b)

        h_reshape = tf.reshape(h, [-1, self.n_seg, self.emb_dim])
        self.feat = tf.reduce_mean(h_reshape, axis=1)

        h_drop = tf.nn.dropout(tf.nn.relu(h), self.output_keep_prob)
        output = tf.nn.xw_plus_b(h_drop, self.W_o, self.b_o)
        output_reshape = tf.reshape(output, [-1, self.n_seg, self.n_output])

        self.logits = tf.reduce_mean(output_reshape, axis=1)


    def prepare_input(self, feat):
        return functools.partial(utils.tsn_prepare_input, self.n_seg)

    def prepare_input_tf(self, feat):
        return functools.partial(utils.tsn_prepare_input_tf, self.n_seg)

# triplet loss
def triplet_loss(anchor, positive, negative, alpha=0.2):

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)


