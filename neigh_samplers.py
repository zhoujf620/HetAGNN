from __future__ import division
from __future__ import print_function

# from model.layers import Layer
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

# 增加weight_adj_info， column_adj_info
# 注释transpose
# 妈蛋，没事继承什么layer啊！有毒！
class UniformNeighborSampler(object):
# class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, features_info, adj_info, weight_adj_info, column_adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.features_info = features_info

        self.adj_info = adj_info
        self.weight_adj_info = weight_adj_info
        self.column_adj_info = column_adj_info

    def sample_feats(self, ids):
        feat_list = tf.nn.embedding_lookup(self.features_info, ids)

        return feat_list

    def sample(self, ids, num_samples):
        # ids, num_samples = inputs

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        #adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        adj_lists = tf.reshape(adj_lists, [-1, ])

        weight_adj_lists = tf.nn.embedding_lookup(self.weight_adj_info, ids)
        weight_adj_lists = tf.slice(weight_adj_lists, [0,0], [-1, num_samples])
        weight_adj_lists = tf.reshape(weight_adj_lists, [-1, ])

        column_adj_lists = tf.nn.embedding_lookup(self.column_adj_info, ids)
        column_adj_lists = tf.slice(column_adj_lists, [0, 0], [-1, num_samples])
        column_adj_lists = tf.reshape(column_adj_lists, [-1, ])

        feat_list = tf.nn.embedding_lookup(self.features_info, adj_lists)

        return adj_lists, weight_adj_lists, column_adj_lists, feat_list
