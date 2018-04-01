import tensorflow as tf
import numpy as np


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



class TripletSim(object):
    """ TripletSim layer
    Input a triplet of sample A, B, C
    Ouput a binary classification: whether sim(A,B) > sim(A,C)

    Reference for pairwise comparison: Unsupervised representation learning by sorting sequence
    """

    def name(self):
        return "TripletSim"

    def __init__(self, n_input=256, output_keep_prob=1.0):
        self.n_input = n_input
        self.output_keep_prob = output_keep_prob

        self.W_pairwise = tf.get_variable(name="W_pairwise", shape=[self.n_input*2, self.n_input//2],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_pairwise = tf.get_variable(name="b_pairwise", shape=[self.n_input//2],
                            initializer=tf.zeros_initializer(),
                            trainable=True)
        self.W_o = tf.get_variable(name="W_o", shape=[2*(self.n_input//2), 1],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(1.),
                            trainable=True)
        self.b_o = tf.get_variable(name="b_pairwise", shape=[1],
                            initializer=tf.zeros_initializer(),
                            trainable=True)

        def forward(self, x):
            """
            x -- feature triplet, [batch_size, 3, n_input]
            """

            x_AB = tf.gather(x, [0,1], axis=1)
            x_AC = tf.gather(x, [0,2], axis=2)

            h_AB = tf.nn.xw_plus_b(x_AB, self.W_pairwise, self.b_pairwise)
            drop_AB = tf.nn.dropout(tf.nn.relu(h_AB), self.output_keep_prob)
            h_AC = tf.nn.xw_plus_b(x_AC, self.W_pairwise, self.b_pairwise)
            drop_AC = tf.nn.dropout(tf.nn.relu(h_AC), self.output_keep_prob)

            h_concat = tf.concat([h_AB, h_AC], axis=1)
            self.logits = tf.nn.xw_plus_b(h_concat, self.W_o, self.b_o)    # for computing loss
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
        """
        feat -- feature sequence, [time_steps, n_h, n_w, n_input]
        """

        # reference: TSN pytorch codes
        average_duration = feat.shape[0] // self.n_seg
        if average_duration > 0:
            offsets = np.multiply(range(self.n_seg), average_duration) + np.random.randint(average_duration, size=self.n_seg)
        else:
            raise NotImplementedError

        return feat[offsets].astype('float32')

    def prepare_input_tf(self, feat):
        """
        tensorflow version
        """

        average_duration = tf.floordiv(tf.shape(feat)[0], self.n_seg)
        offsets = tf.add(tf.multiply(tf.range(self.n_seg,dtype=tf.int32), average_duration),
                        tf.random_uniform(shape=(1,self.n_seg),maxval=average_duration,dtype=tf.int32))
        # offset should be column vector, use reshape
        return tf.gather_nd(feat, tf.reshape(offsets, [-1,1]))

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
        """
        feat -- feature sequence, [time_steps, n_h, n_w, n_input]
        """

        # reference: TSN pytorch codes
        average_duration = feat.shape[0] // self.n_seg
        if average_duration > 0:
            offsets = np.multiply(range(self.n_seg), average_duration) + np.random.randint(average_duration, size=self.n_seg)
        else:
            raise NotImplementedError

        return feat[offsets].astype('float32')

    def prepare_input_tf(self, feat):
        """
        tensorflow version
        """

        average_duration = tf.floordiv(tf.shape(feat)[0], self.n_seg)
        offsets = tf.add(tf.multiply(tf.range(self.n_seg,dtype=tf.int32), average_duration),
                        tf.random_uniform(shape=(1,self.n_seg),maxval=average_duration,dtype=tf.int32))
        # offset should be column vector, use reshape
        return tf.gather_nd(feat, tf.reshape(offsets, [-1,1]))


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
        """
        feat -- feature sequence, [time_steps, n_h, n_w, n_input]
        """

        # reference: TSN pytorch codes
        average_duration = feat.shape[0] // self.n_seg
        if average_duration > 0:
            offsets = np.multiply(range(self.n_seg), average_duration) + np.random.randint(average_duration, size=self.n_seg)
        else:
            raise NotImplementedError

        return feat[offsets].astype('float32')

    def prepare_input_tf(self, feat):
        """
        tensorflow version
        """

        average_duration = tf.floordiv(tf.shape(feat)[0], self.n_seg)
        offsets = tf.add(tf.multiply(tf.range(self.n_seg,dtype=tf.int32), average_duration),
                        tf.random_uniform(shape=(1,self.n_seg),maxval=average_duration,dtype=tf.int32))
        # offset should be column vector, use reshape
        return tf.gather_nd(feat, tf.reshape(offsets, [-1,1]))

# triplet loss
def triplet_loss(anchor, positive, negative, alpha=0.2):

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)


