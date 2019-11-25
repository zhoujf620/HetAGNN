# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

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
    def __init__(self, placeholders, features, 
                 train_adj, train_weight_adj, train_column_adj,
                 test_adj, test_weight_adj, test_column_adj,
                 label_map, supervised_info, sample_size, 
                 batch_size = 100, max_degree = 30, **kwargs):
    # def __init__(self, G, placeholders, features, id_map, weight_map, label_map, weight_dict, supervised_info,
    #              mask = None, batch_size = 100, max_degree = 30, **kwargs):
        '''
        :param mask: use to stimulate real-world leasing
        :param batch_size: default set 256
        :param supervised_info: denotes that whether to train or valid/test nodes
        :param max_degree:
        :param kwargs:
        '''
        self.placeholders = placeholders

        self.features = features

        self.train_adj = train_adj
        self.train_weight_adj = train_weight_adj
        self.train_column_adj = train_column_adj

        self.test_adj = test_adj
        self.test_weight_adj = test_weight_adj
        self.test_column_adj = test_column_adj

        self.label_map = label_map
        # self.nodes = range(G.vcount())
        self.batch_size = batch_size
        # self.max_degree = max_degree
        self.batch_num = 0
        # self.mask = mask
       
        # self.supervised_info = supervised_info
        self.train_nodes, self.val_nodes, self.test_nodes = supervised_info

        # self.val_nodes = val_nodes
        # self.test_nodes = test_nodes
        # # don't train on nodes that only have edges to test set
        # self.train_nodes = train_nodes

        self.sample_size = sample_size
        # self.sample_size = [layer_infos[i].num_samples for i in range(len(layer_infos))]


    # ===== for batch =====
    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def train_batch_sample(self, 
        # adj_info, weight_adj_info, column_adj_info, features_info, 
        batch_nodes, sample_size=[16, 8]):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        
        # inputs1 = placeholders["batch"]
        # samples = [inputs] 
        sample_nodes  = [batch_nodes]
        sampled_weight = [batch_nodes]
        sampled_column = [batch_nodes]
        sampled_feats = [np.take(self.features, batch_nodes, axis=0)]
        # sampled_feats = [tf.nn.embedding_lookup(features_info, batch_nodes)]
        # sampled_feats = [sampled_feats(batch_nodes)]

        # size of convolution support at each layer per node
        # support_size = 1
        # support_sizes = [support_size] # [1, 8, 8*16]
        for k in range(len(sample_size)):
            t = len(sample_size) - k - 1 # len(layer)-1: 0
            # support_size *= num_samples[t]

            adj_lists = np.take(self.train_adj, sample_nodes[k], axis=0)
            adj_lists = np.transpose(np.random.permutation(np.transpose(adj_lists)))
            adj_lists = adj_lists[:, :sample_size[t]]
            adj_lists = np.reshape(adj_lists, [-1, ])

            weight_adj_lists = np.take(self.train_weight_adj, sample_nodes[k], axis=0)
            weight_adj_lists = np.transpose(np.random.permutation(np.transpose(weight_adj_lists)))
            weight_adj_lists = weight_adj_lists[:, :sample_size[t]]
            weight_adj_lists = np.reshape(weight_adj_lists, [-1, ])

            column_adj_lists = np.take(self.train_column_adj, sample_nodes[k], axis=0)
            column_adj_lists = np.transpose(np.random.permutation(np.transpose(column_adj_lists)))
            column_adj_lists = column_adj_lists[:, :sample_size[t]]
            column_adj_lists = np.reshape(column_adj_lists, [-1, ])

            feat_list = np.take(self.features, adj_lists, axis=0)

            # sampler = layer_infos[t].neigh_sampler
            # data_sampled = layer_sample(sample_nodes[k], num_samples[t])

            # sample_nodes 是index 其他的是weight, column 值
            # 但还只是sample num_samples 个啊
            sample_nodes.append(adj_lists)
            sampled_weight.append(weight_adj_lists)
            sampled_column.append(column_adj_lists) 
            sampled_feats.append(feat_list)
            # sample_nodes.append(tf.reshape(node[0], [batch_size*support_size, ]))
            # sampled_weight.append(tf.reshape(node[1], [batch_size*support_size, ]))
            # sampled_column.append(tf.reshape(node[2], [batch_size*support_size, ])) 
            
            # support_sizes.append(support_size)
            # sampled_feats = tf.convert_to_tensor(sampled_feats)
            # sampled_weight = tf.convert_to_tensor(sampled_weight)
            # sampled_column = tf.convert_to_tensor(sampled_column)

        return sampled_feats, sampled_weight, sampled_column#, support_sizes

    def test_batch_sample(self, 
        batch_nodes, sample_size=[16, 8]):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        
        sample_nodes  = [batch_nodes]
        sampled_weight = [batch_nodes]
        sampled_column = [batch_nodes]
        sampled_feats = [np.take(self.features, batch_nodes, axis=0)]

        for k in range(len(sample_size)):
            t = len(sample_size) - k - 1 # len(layer)-1: 0
            # support_size *= num_samples[t]

            adj_lists = np.take(self.test_adj, sample_nodes[k], axis=0)
            adj_lists = np.transpose(np.random.permutation(np.transpose(adj_lists)))
            adj_lists = adj_lists[:, :sample_size[t]]
            adj_lists = np.reshape(adj_lists, [-1, ])

            weight_adj_lists = np.take(self.test_weight_adj, sample_nodes[k], axis=0)
            weight_adj_lists = np.transpose(np.random.permutation(np.transpose(weight_adj_lists)))
            weight_adj_lists = weight_adj_lists[:, :sample_size[t]]
            weight_adj_lists = np.reshape(weight_adj_lists, [-1, ])

            column_adj_lists = np.take(self.test_column_adj, sample_nodes[k], axis=0)
            column_adj_lists = np.transpose(np.random.permutation(np.transpose(column_adj_lists)))
            column_adj_lists = column_adj_lists[:, :sample_size[t]]
            column_adj_lists = np.reshape(column_adj_lists, [-1, ])

            feat_list = np.take(self.features, adj_lists, axis=0)

            # sample_nodes 是index 其他的是weight, column 值
            # 但还只是sample num_samples 个啊
            sample_nodes.append(adj_lists)
            sampled_weight.append(weight_adj_lists)
            sampled_column.append(column_adj_lists) 
            sampled_feats.append(feat_list)

        return sampled_feats, sampled_weight, sampled_column#, support_sizes

    def batch_feed_dict(self, batch_nodes, val=False):
        # batch = batch_nodes

        labels = np.vstack([self.label_map[node] for node in batch_nodes])
        
        t_batch = time.time()
        if not val:
            sampled_feats, sampled_weight, sampled_column = self.train_batch_sample(
                # self.train_adj, self.train_weight_adj, self.train_column_adj, self.features, 
                batch_nodes, self.sample_size)
        else:
            sampled_feats, sampled_weight, sampled_column = self.test_batch_sample(
                batch_nodes, self.sample_size)

        # 都很耗时
        t_index = time.time()
        feed_dict = dict()
        # feed_dict.update({self.placeholders['features']: self.features}) # 先传入全量的features好了
        # placeholder 做key，很厉害哦
        # feed_dict.update({self.placeholders['labels']: labels})
        # feed_dict.update({self.placeholders['batch_nodes']: batch_nodes})
        feed_dict.update({self.placeholders['batch_size']: len(batch_nodes)})

        t_info = time.time()
        feed_dict.update({self.placeholders['sampled_weight_0']: sampled_weight[0]})
        feed_dict.update({self.placeholders['sampled_column_0']: sampled_column[0]})
        feed_dict.update({self.placeholders['sampled_feats_0']: sampled_feats[0]})
        feed_dict.update({self.placeholders['sampled_weight_1']: sampled_weight[1]})
        feed_dict.update({self.placeholders['sampled_column_1']: sampled_column[1]})
        feed_dict.update({self.placeholders['sampled_feats_1']: sampled_feats[1]})
        feed_dict.update({self.placeholders['sampled_weight_2']: sampled_weight[2]})
        feed_dict.update({self.placeholders['sampled_column_2']: sampled_column[2]})
        feed_dict.update({self.placeholders['sampled_feats_2']: sampled_feats[2]})
        
        # feed_dict[self.placeholders['labels']] = labels
        # feed_dict[self.placeholders['batch_size']] = len(batch_nodes)

        # t_info = time.time()
        # feed_dict[self.placeholders['sampled_weight_0']] = sampled_weight[0]
        # feed_dict[self.placeholders['sampled_column_0']] = sampled_column[0]
        # feed_dict[self.placeholders['sampled_feats_0']] = sampled_feats[0]
        # feed_dict[self.placeholders['sampled_weight_1']] = sampled_weight[1]
        # feed_dict[self.placeholders['sampled_column_1']] = sampled_column[1]
        # feed_dict[self.placeholders['sampled_feats_1']] = sampled_feats[1]
        # feed_dict[self.placeholders['sampled_weight_2']] = sampled_weight[2]
        # feed_dict[self.placeholders['sampled_column_2']] = sampled_column[2]
        # feed_dict[self.placeholders['sampled_feats_2']] = sampled_feats[2]
    
        # print('batch time %.2f, index time %.2f, info time %.2f' % (time.time()-t_info, t_info-t_index, t_index-t_batch))

        return feed_dict, labels

    # for train
    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes)) 
        # batch_size != flags.batch_size

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
        ret_val = self.batch_feed_dict(val_nodes, val=True)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        
        val_node_subset = val_nodes[
            iter_num * size: min((iter_num + 1) * size, len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset, val=True)
        return ret_val[0], ret_val[1], (iter_num + 1) * size >= len(val_nodes), val_node_subset


