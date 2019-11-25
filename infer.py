# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import config
from utils import load_data
# from metrics import precision_recall_curve, auc, calc_f1, classification_report

from minibatch_infer import NodeMinibatchIterator
# from neigh_samplers import UniformNeighborSampler

from model.model_infer import SupervisedGraphsage

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


# === 数据 ===
print("Load testing data")
features, label_map, \
    _, _, test_nodes, \
    _, _, _, \
    adj, weight_adj, column_adj = load_data(FLAGS.train_prefix)
features = np.vstack([features, np.zeros((features.shape[1], ))])



def sample_adj(batch_nodes, sample_size=[16, 8]):
    """ Sample neighbors to be the supportive fields for multi-layer convolutions.

    Args:
        inputs: batch inputs
        batch_size: the number of inputs (different for batch inputs and negative samples).
    """
    
    sampled_index  = [batch_nodes]
    sampled_weight = [batch_nodes]
    sampled_column = [batch_nodes]
    # sampled_feats = [np.take(self.features, batch_nodes, axis=0)]

    for k in range(len(sample_size)):
        t = len(sample_size) - k - 1 # len(layer)-1: 0
        # support_size *= num_samples[t]

        adj_lists = np.take(adj, sampled_index[k], axis=0)
        adj_lists = np.transpose(np.random.permutation(np.transpose(adj_lists)))
        adj_lists = adj_lists[:, :sample_size[t]]
        adj_lists = np.reshape(adj_lists, [-1, ])

        weight_adj_lists = np.take(weight_adj, sampled_index[k], axis=0)
        weight_adj_lists = np.transpose(np.random.permutation(np.transpose(weight_adj_lists)))
        weight_adj_lists = weight_adj_lists[:, :sample_size[t]]
        weight_adj_lists = np.reshape(weight_adj_lists, [-1, ])

        column_adj_lists = np.take(column_adj, sampled_index[k], axis=0)
        column_adj_lists = np.transpose(np.random.permutation(np.transpose(column_adj_lists)))
        column_adj_lists = column_adj_lists[:, :sample_size[t]]
        column_adj_lists = np.reshape(column_adj_lists, [-1, ])

        # feat_list = np.take(self.features, adj_lists, axis=0)

        # sample_nodes 是index 其他的是weight, column 值
        # 但还只是sample num_samples 个啊
        sampled_index.append(adj_lists)
        sampled_weight.append(weight_adj_lists)
        sampled_column.append(column_adj_lists) 
        # sampled_feats.append(feat_list)

    return sampled_index, sampled_weight, sampled_column#, support_sizes

def sample_feats(sampled_index):
    sampled_feats = []
    for layer in range(len(sampled_index)):
            sampled_feats.append(np.take(features, sampled_index[layer], axis=0))
    return sampled_feats

def construct_feed_dict(placeholders, batch_nodes):

    sampled_index, sampled_weight, sampled_column = sample_adj(batch_nodes)
    sampled_feats = sample_feats(sampled_index)

    feed_dict = dict()
    # feed_dict.update({self.placeholders['features']: self.features}) # 先传入全量的features好了
    # placeholder 做key，很厉害哦
    # feed_dict.update({self.placeholders['labels']: labels})
    # feed_dict.update({self.placeholders['batch_nodes']: batch_nodes})
    feed_dict.update({placeholders['batch_size']: len(batch_nodes)})

    t_info = time.time()
    feed_dict.update({placeholders['sampled_weight_0']: sampled_weight[0]})
    feed_dict.update({placeholders['sampled_column_0']: sampled_column[0]})
    feed_dict.update({placeholders['sampled_feats_0']: sampled_feats[0]})
    feed_dict.update({placeholders['sampled_weight_1']: sampled_weight[1]})
    feed_dict.update({placeholders['sampled_column_1']: sampled_column[1]})
    feed_dict.update({placeholders['sampled_feats_1']: sampled_feats[1]})
    feed_dict.update({placeholders['sampled_weight_2']: sampled_weight[2]})
    feed_dict.update({placeholders['sampled_column_2']: sampled_column[2]})
    feed_dict.update({placeholders['sampled_feats_2']: sampled_feats[2]})

    return feed_dict


def main():

    # === 模型 ===
    print("Set graph model")
    # num_classes = 2 # label_map.shape[1]
    # feats_dim = 102 #features.shape[1]
    # placeholders = construct_placeholders(num_classes, feats_dim)
    graph_model = SupervisedGraphsage()


    minibatch = NodeMinibatchIterator(graph_model.placeholders,
                                      features,
                                      _,_, _,
                                      adj, weight_adj, column_adj, 
                                      label_map,
                                      supervised_info = [_, _, test_nodes],
                                      sample_size=[FLAGS.samples_1, FLAGS.samples_2], 
                                      batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree)
    

    for node in test_nodes[:100]:
        # feed_dict, labels = minibatch.batch_feed_dict(np.array([node]))
        # sampled_index, sampled_weight, sampled_column = sample_adj([np.array([node])])
        # sampled_feats = sample_feats(sampled_index)

        feed_dict = construct_feed_dict(graph_model.placeholders, np.array([node]))
                                    # sampled_feats, sampled_weight, sampled_column)

        node_outs = graph_model.predict(feed_dict)
        print(node_outs)
    graph_model.close_sess()


if __name__ == '__main__':
    print("Model: ", FLAGS.model)
    # tf.app.run()
    main()
