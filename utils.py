#-*- coding:utf-8 -*-
from __future__ import print_function

import os
import json
import random
import pickle as pkl

import igraph

import numpy as np
import pandas as pd
import tensorflow as tf 

from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

WALK_LEN=5
N_WALKS=50

feats_transed_columns_with_label = [
    'activityDiscount', 'billingMethod', 'brand', 'cost', 'creditFreezeAmount',        
    'creditPrivilegeAmount', 'creditScore', 'creditTotalAmount', 'discount', 'duration',
    'finalCost', 'firstPay', 'foregift', 'full', 'goodsType', 'goodsname',       
    'installment', 'insuranceFullPrice', 'insuranceInstallmentPrice', 'num',        
    'orderId', 'userId', 'oriPri', 'pay', 'payNum', 'price', 'relMobile', 'relRelation', 
    'safe', 'goods_notnew', 'createTime_Dayofweek', 'createTime_Month',
    'createTime_Is_month_start', 'createTime_Is_month_end',       
    'createTime_Is_Weekend', 'hours_to_pay', 'l_is_contacts_upload',
    'l_is_used_complet', 'l_user_gender', 'l_user_monthly_income',            
    'l_user_occupation', 'l_user_zhima_score', 'workplace', 'zmFace', 'zmGrade', 
    'zmRisk', 'age', 'num_contacts', 'th_order', 'days_to_order', 'migrant', 
    'act_times', 'diff_ip', 'diff_device', 'diff_imsi', 'diff_imei',           
    'diff_wifimac', 'loc_times', 'diff_gps', 'diff_gps100', 'diff_gps1000',
    'td_final_score', 'r_sl_inbankblack', 'r_sl_incourtblack',            
    'r_sl_inp2pblack', 'r_sl_org_cnt_recent7d', 'r_sl_install_online_cnt', 
    'r_sl_idcard_inblack', 'r_sl_creditcard_pay_cnt', 'r_sl_install_offline_cnt',
    'r_sl_org_cnt_recent90d', 'r_sl_loan_pay_cnt', 'r_sl_org_cnt', 
    'r_sl_search_cnt_recent90d', 'r_sl_search_cnt_recent7d',            
    'r_sl_search_cnt_recent60d', 'r_sl_search_cnt_recent30d',
    'r_sl_search_cnt_recent180d', 'r_sl_search_cnt_recent14d',            
    'r_sl_search_cnt', 'r_sl_phone_inblack', 'r_sl_others_cnt', 
    'r_sl_cashloan_offline_cnt', 'r_sl_org_cnt_recent60d',
    'r_sl_org_cnt_recent30d', 'r_sl_org_cnt_recent180d',
    'r_sl_org_cnt_recent14d', 'r_sl_cashloan_online_cnt',
    'payType_alipay', 'payType_jdpay', 'payType_none', 'payType_wx', 
    'insuranceMethod_free', 'insuranceMethod_full', 'insuranceMethod_installment',
    'insuranceMethod_none', 'insuranceSupportMethod_all',             
    'insuranceSupportMethod_free', 'insuranceSupportMethod_full', 
    'insuranceSupportMethod_none', 'riskState_freeAuditPass', 'riskState_gradePass',
    'riskState_none', 'riskState_pass', 'td_final_decision_Accept',
    'td_final_decision_Reject', 'td_final_decision_Review',
    'td_final_decision_none', 'order_time', 'order_label', 'user_label']


def load_data(prefix, normalize=True):

    with open('./data/'+ prefix + '_multigraph', 'r') as f:
        G = igraph.Graph.Read_Lgl(f, directed=False)
    with open('./data/'+ prefix + '_column_dict', 'rb') as f:
        column_dict = pkl.load(f)
    with open('./data/'+ prefix + '_weight_dict', 'rb') as f:
        weight_dict = pkl.load(f) # np.float64

    # id_map = G.vs['name']
    # weight_map = G.es['weight']
    
    # feats = pd.read_csv('./data/'+prefix + '_new_Feats_1.csv')
    # feats.rename(columns={'user_id': 'userId', 'order_id': 'orderId'}, inplace=True)
    feats = pd.read_csv('./data/'+prefix + '_new_Feats_0925.csv', 
                    usecols=feats_transed_columns_with_label)[feats_transed_columns_with_label]

    # === set block set ===
    # delete label >2
    block_set = set(feats[feats.order_label.isin([1, 3, 99])].index)
    # block_set = block_set.union([85764, 84015, 100335, 55167, 66907, 37698, 26921, 70542,
    #                              4415, 82024, 1263, 100018, 62650, 69397, 90000, 80336,
    #                              100030, 28776, 95452, 74494, 93069, 25193, 98885, 84515])
    block_set = block_set.union([85764, 84015, 100335, 55167, 66907, 37698, 26921, 70542,
                                 4415, 82024, 1263, 100018])
    # block_set = block_set.union([95614, 32105, 79691, 20363, 60117, 29109, 48352, 48644,
    #                              61766, 40297, 76013, 21530, 89236, 101013, 34732, 42036,
    #                              75545, 93068, 85676, 95039, 44315, 94096, 4818, 40225,
    #                              68500, 30917, 86288, 100034, 96533, 60363, 32499, 20900,
    #                              80144, 37451])
    block_set = block_set.union([95614, 32105, 79691, 20363, 60117, 29109, 48352, 48644,
                                 61766, 40297, 76013, 21530, 89236, 101013, 34732, 42036,
                                 75545, 93068, 85676, 95039, 44315, 94096, 4818, 40225,
                                 68500, 30917])

    # Block掉一部分的良性用户以取得数据平衡
    # benign_index = np.array(feats[feats.order_label==0].index)
    # block_ratio = 0
    # np.random.seed(2019)
    # block_set = block_set.union(np.random.permutation(benign_index)[:int(len(benign_index)* block_ratio)])
    print('block set length:', len(block_set))
    # block_set = set()

    label = feats.order_label.copy()
    # benign_index = label[label == 0].index
    # benign_set = set(np.random.permutation(benign_index)[30000:])

    # 不需要删掉，直接置为非train
    # Stratified sampling
    # label = feats.order_label
    # sklea.StratifiedKFold(n_splits = 2, random_state = 123)
    feats.drop(['userId','orderId', 'user_label', 'order_label', 
                # 'delivery_add_code',
                'order_time', 
                # 'diff_gps', 'diff_imei', 'diff_wifimac', 'th_order'
                ], axis=1, inplace=True)
    # feats.drop(['user_id','order_id', 'user_label', 'order_label'], axis=1, inplace=True)
    # Here needs to change in server.
    print('=== feats shape ===', feats.shape)

    x_train, x_, y_train, y_ = train_test_split(
        feats, label, test_size=0.1, train_size=0.9, random_state=123, stratify=label)
    #x_train, x_, y_train, y_ = train_test_split(feats, label, test_size=0.3, random_state=123, stratify= True)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_, y_, test_size=0.99, train_size=0.01, random_state=234, stratify=y_)
    #x_valid, x_test, y_valid, y_test = train_test_split(x_, y_, test_size=0.5, random_state=234, stratify= True)

    train_set = set(x_train.index).difference(block_set)
    #train_set = train_set.difference(benign_set)
    valid_set = set(x_valid.index).difference(block_set)
    test_set = set(x_test.index).difference(block_set)

    label_map = np.zeros(shape=(len(label), 2))
    for i in label.index:
        if label[i] == 2:
            label_map[i, 1] = 1
        else:
            label_map[i, 0] = 1

    # === from minibatch.py ===
    train_nodes = list(train_set)
    val_nodes = list(valid_set)
    test_nodes = list(test_set)
    block_nodes = list(block_set)

    train_adj, train_deg, train_weight_adj, train_column_adj, \
        test_adj, test_deg, test_weight_adj, test_column_adj = \
            construct_adj(
                G, FLAGS.max_degree, weight_dict, 
                [train_nodes, val_nodes, test_nodes, block_nodes])

    # 33526, 43, 4012
    # don't train on nodes that only have edges to test set
    train_nodes = [n for n in train_nodes if train_deg[n] > 0]
    print('num of train_nodes:', len(train_nodes))
    val_nodes = [n for n in val_nodes if test_deg[n] > 0]
    test_nodes = [n for n in test_nodes if test_deg[n] > 0]

    return feats.values, label_map, \
            train_nodes, val_nodes, test_nodes,\
            train_adj, train_weight_adj, train_column_adj, \
            test_adj, test_weight_adj, test_column_adj


def construct_adj(G, max_degree, weight_dict, supervised_info):
    nodes = range(G.vcount())
    id_map = G.vs['name']
    weight_map = G.es['weight']

    i2v, v2i = {}, {}
    for i in range(len(nodes)):
        i2v[i] = int(id_map[i])
        v2i[int(id_map[i])] = i

    column_weight = {}
    for i, tu in enumerate(G.get_edgelist()):
        u, v = tu
        tu = (i2v[u], i2v[v])
        u, v = min(tu), max(tu)
        try:
            column_weight[(u, v)][int(weight_map[i])] = weight_dict[(u, v, int(weight_map[i]))]
        except:
            column_weight[(u, v)] = {}
            column_weight[(u, v)][int(weight_map[i])] = weight_dict[(u, v, int(weight_map[i]))]

    train_nodes, val_nodes, test_nodes, block_nodes = supervised_info

    # train_nodes = list(train_set)
    # val_nodes = list(valid_set)
    # test_nodes = list(test_set)
    # block_nodes = list(block_set)

    is_train = np.zeros(shape=(len(nodes)))
    is_val = np.zeros(shape=len(nodes))
    is_block = np.zeros(shape=len(nodes))

    for n in train_nodes:
        is_train[n] = 1
    for n in block_nodes:
        is_block[n] = 1

    for n in val_nodes:
        is_val[n] = 1
    for n in test_nodes:
        is_val[n] = 1

    train_adj = len(nodes) * np.ones((len(nodes) + 1, max_degree), dtype=np.int32)
    train_deg = np.zeros((len(nodes), ), dtype=np.int32)
    train_weight_adj = len(nodes) * np.zeros((len(nodes) + 1, max_degree), dtype=np.float32)
    train_column_adj = len(nodes) * np.zeros((len(nodes) + 1, max_degree), dtype=np.int32)

    for nodeid in nodes:
        if is_val[nodeid]:
            continue

        neighbors = [i2v[neighbor] for neighbor in G.neighbors(v2i[nodeid])
                        if not is_val[i2v[neighbor]] and not is_block[i2v[neighbor]]]
        p, q = [], []
        for neighbor in set(neighbors):
            tu = (nodeid, neighbor)
            u, v = min(tu), max(tu)
            for key, value in column_weight[(u, v)].items():
                q.append(key-1)
                p.append(value)

        train_deg[nodeid] = len(neighbors)
        if len(neighbors) == 0:
            continue
            # then adj[nodeid,:] = len(self.nodes)
        if len(neighbors) >= max_degree:
            index = np.random.choice(len(neighbors), max_degree, replace=False)
            #会不会导致多属性图，无法抽取多个边对应的同一个那啥的问题
            #使用同一种random.choice对对应的边的属性type进行抽取
        elif len(neighbors) < max_degree:
            index = np.random.choice(len(neighbors), max_degree, replace=True)

        new_neighbors, new_p, new_q = [], [], []
        for i in index:
            new_neighbors.append(neighbors[i])
            new_p.append(p[i])
            new_q.append(q[i])

        train_adj[nodeid, :] = new_neighbors
        train_weight_adj[nodeid, :] = new_p
        train_column_adj[nodeid, :] = new_q

    # === test adj ===
    test_adj = len(nodes) * np.ones((len(nodes) + 1, max_degree), dtype=np.int32)
    test_deg = np.zeros((len(nodes),), dtype=np.int32)
    test_weight_adj = len(nodes) * np.zeros((len(nodes) + 1, max_degree), dtype=np.float32)
    test_column_adj = len(nodes) * np.zeros((len(nodes) + 1, max_degree), dtype=np.int32)

    for nodeid in nodes:
        neighbors = [i2v[neighbor] for neighbor in G.neighbors(v2i[nodeid])
                        if not is_block[i2v[neighbor]]]

        p, q = [], []
        for neighbor in set(neighbors):
            tu = (nodeid, neighbor)
            u, v = min(tu), max(tu)
            for key, value in column_weight[(u, v)].items():
                q.append(key-1)
                p.append(value)

        test_deg[nodeid] = len(neighbors)
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= max_degree:
            index = np.random.choice(len(neighbors), max_degree, replace=False)
        elif len(neighbors) < max_degree:
            index = np.random.choice(len(neighbors), max_degree, replace=True)

        new_neighbors, new_p, new_q = [], [], []
        for i in index:
            new_neighbors.append(neighbors[i])
            new_p.append(p[i])
            new_q.append(q[i])

        test_adj[nodeid, :] = new_neighbors
        test_weight_adj[nodeid, :] = new_p
        test_column_adj[nodeid, :] = new_q

    return train_adj, train_deg, train_weight_adj, train_column_adj, \
        test_adj, test_deg, test_weight_adj, test_column_adj



def construct_placeholders(num_classes, feats_dim):
    # Define placeholders
    # 这里的key 是供 model init 用的
    # feed_dict = {placeholders: data}
    placeholders = {
        # 'features': tf.placeholder(tf.float32, shape=(None, feats_dim)), 
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch_nodes': tf.placeholder(tf.int32, shape=(None), name='batch_nodes'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        
        # 最后要给model的，list of placeholder
        # 'support_size': tf.placeholder(name='support_size'), 
        # 'sampled_weight': [tf.placeholder(tf.float32, shape=(), name='sampled_weight') 
                                # for _ in range(10)],
        'sampled_weight_0': tf.placeholder(tf.float32, name='sampled_weight_0'),
        'sampled_column_0': tf.placeholder(tf.int32, name='sampled_column_0'), 
        'sampled_feats_0': tf.placeholder(tf.float32, name='sampled_feats_0'),

        'sampled_weight_1': tf.placeholder(tf.float32, name='sampled_weight_1'),
        'sampled_column_1': tf.placeholder(tf.int32, name='sampled_column_1'), 
        'sampled_feats_1': tf.placeholder(tf.float32, name='sampled_feats_1'),
        
        'sampled_weight_2': tf.placeholder(tf.float32, name='sampled_weight_2'),
        'sampled_column_2': tf.placeholder(tf.int32, name='sampled_column_2'), 
        'sampled_feats_2': tf.placeholder(tf.float32, name='sampled_feats_2')
    }

    return placeholders


def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
        model=FLAGS.model,
        model_size=FLAGS.model_size,
        lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
