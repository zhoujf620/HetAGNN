# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

np.random.seed(123)

class NodeMinibatchIterator(object):

    """
    This minibatch iterator interate over nodes for supervised learning.

    G -- igraph object
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer of list)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    # TODO feature 还是要先采样过
    def __init__(self, placeholders, label_map, supervised_info,
                 mask = None, batch_size = 100, max_degree = 30, **kwargs):
    # def __init__(self, G, placeholders, features, id_map, weight_map, label_map, weight_dict, supervised_info,
    #              mask = None, batch_size = 100, max_degree = 30, **kwargs):
        '''
        :param mask: use to stimulate real-world leasing
        :param batch_size: default set 256
        :param supervised_info: denotes that whether to train or valid/test nodes
        :param max_degree:
        :param kwargs:
        '''
        # self.G = G
        self.placeholders = placeholders
        # self.features = features
        self.label_map = label_map
        # self.nodes = range(G.vcount())
        self.batch_size = batch_size
        # self.max_degree = max_degree
        self.batch_num = 0
        # self.mask = mask
        # self.supervised_info = supervised_info

        train_nodes, val_nodes, test_nodes = supervised_info
        # train_set, valid_set, test_set, block_set = supervised_info

        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        # don't train on nodes that only have edges to test set
        self.train_nodes = train_nodes


    # ===== for batch =====
    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        # batch = batch_nodes

        labels = np.vstack([self.label_map[node] for node in batch_nodes])
        
        feed_dict = dict()
        # feed_dict.update({self.placeholders['features']: self.features}) # 先传入全量的features好了
        # placeholder 做key，很厉害哦
        feed_dict.update({self.placeholders['labels']: labels})
        feed_dict.update({self.placeholders['batch_nodes']: batch_nodes})
        feed_dict.update({self.placeholders['batch_size']: len(batch_nodes)})

        return feed_dict, labels

    # for train
    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))

        batch_nodes = self.train_nodes[start_idx: end_idx]
        fead_dict, labels = self.batch_feed_dict(batch_nodes)
        return fead_dict, labels

    # for valuation
    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        
        val_node_subset = val_nodes[iter_num * size: min((iter_num + 1) * size, len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num + 1) * size >= len(val_nodes), val_node_subset

