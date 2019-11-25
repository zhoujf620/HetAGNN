# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import config
from utils import load_data, log_dir, construct_placeholders
from metrics import precision_recall_curve, auc, calc_f1, classification_report

from minibatch_train import NodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler

from model.model_train import SupervisedGraphsage, SAGEInfo

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
GPU_MEM_FRACTION = 0.8


# Define model evaluation function
# 随机采样size 个node 
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)

    node_outs_val = sess.run([model.preds, model.loss],
                             feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])

    return node_outs_val[1], mic, mac, (time.time() - t_test)

# 全量，这里到area从
def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False

    while not finished:
        feed_dict_val, batch_labels, finished, _ = \
            minibatch_iter.incremental_node_val_feed_dict( 
                size, iter_num, test=test)

        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    
    # TODO 放进model
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    report = classification_report(labels, val_preds)

    # precision, recall, thresholds = precision_recall_curve(
    #     labels[:, 1], val_preds[:, 1])
    # area = auc(recall, precision)

    return np.mean(val_losses), f1_scores[0], f1_scores[1],report, (time.time() - t_test)#, area

# 放弃，incremental_evaluate 得到的area 很奇怪 从0到0.8
def my_incremental_evaluate(sess, model, minibatch_iter, size, test= False):
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False

    while not finished:
        feed_dict_val, batch_labels, finished, _ = \
            minibatch_iter.incremental_node_val_feed_dict(
            size, iter_num, test=test)

        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)

        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1

    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)

    precision, recall, thresholds = precision_recall_curve(
        labels[:, 1], val_preds[:, 1])
    area = auc(recall, precision)

    return area


def train(train_data, test_data=None):
    features, label_map, \
        train_nodes, valid_nodes, test_nodes, \
        train_adj, train_weight_adj, train_column_adj, \
        test_adj, test_weight_adj, test_column_adj = train_data
    
    # if isinstance(list(class_map.values())[0], list):
    #     num_classes = len(list(class_map.values())[0])
    # else:
    #     num_classes = len(set(class_map.values()))
    
    num_classes = label_map.shape[1]
    feats_dim = features.shape[1]

    # 插入0行好像没什么用啊？
    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((feats_dim, ))])
    # 不晓得为啥要variable(constant(), trainable=False), 很奇怪
    features_info = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
    
    #context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes, feats_dim)
    minibatch = NodeMinibatchIterator(placeholders,
                                    #   features,
                                    #   id_map,
                                    #   weight_map,
                                      label_map,
                                    #   weight_dict,
                                      supervised_info = [train_nodes, valid_nodes, test_nodes],
                                      batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree)

    # 注意！是placeholder, 且是全量的
    # TODO shape 还有数据信息， (, train_adj.shape)
    adj_info_ph = tf.placeholder(tf.int32, shape=train_adj.shape)
    weight_adj_info_ph = tf.placeholder(tf.float32, shape=train_weight_adj.shape)
    column_adj_info_ph = tf.placeholder(tf.int32, shape=train_column_adj.shape)
    
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    weight_adj_info = tf.Variable(weight_adj_info_ph, trainable=False, name='weight_adj_info')
    column_adj_info = tf.Variable(column_adj_info_ph, trainable=False, name='column_adj_info')

    # 没有被完全赋值，只是赋值操作
    train_adj_info = tf.assign(adj_info, train_adj)
    val_adj_info = tf.assign(adj_info, test_adj)

    train_weight_adj_info = tf.assign(weight_adj_info, train_weight_adj)
    val_weight_adj_info = tf.assign(weight_adj_info, test_weight_adj)

    train_column_adj_info = tf.assign(column_adj_info, train_column_adj)
    val_column_adj_info = tf.assign(column_adj_info, test_column_adj)

    # 采样
    # TODO  features 数据还是从这里进去了
    # TODO 要拿出来啊啊啊啊啊
    sampler = UniformNeighborSampler(features_info, adj_info, weight_adj_info, column_adj_info)

    # === build model ===
    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info, weight_adj_info, column_adj_info)
        # 16, 8
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                           SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos,
                                    concat=True,
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info, weight_adj_info, column_adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="gcn",
                                    model_size=FLAGS.model_size,
                                    concat=False,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'geniepath':
        sampler = UniformNeighborSampler(adj_info, weight_adj_info, column_adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="geniepath",
                                    model_size=FLAGS.model_size,
                                    concat=False,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)
    
    elif FLAGS.model == 'cross':
        # Create model        
        # if FLAGS.samples_3 != 0:
        #     layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
        #                    SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
        #                    SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        # elif FLAGS.samples_2 != 0:
        #     layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
        #                    SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        # else:
        #     layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
        layer_infos = [SAGEInfo("node", FLAGS.samples_1, FLAGS.dim_1),
                        SAGEInfo("node", FLAGS.samples_2, FLAGS.dim_2)]
        model = SupervisedGraphsage(placeholders,
                                    feats_dim, num_classes, 
                                    sampler,
                                    # features,
                                    # adj_info, # variable
                                    # minibatch.deg,
                                    layer_infos = layer_infos,
                                    aggregator_type='cross', # 多了这一句
                                    concat=True,
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    else:
        raise Exception('Error: model name unrecognized.')


    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    # merged = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={
        adj_info_ph: train_adj,
        weight_adj_info_ph: train_weight_adj,
        column_adj_info_ph: train_column_adj})

    # === Train model ===
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    for epoch in range(FLAGS.train_epochs):
        minibatch.shuffle()

        iter = 0
        print('\n### Epoch: %04d ###' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict() # 每一次都有全量的feet 进来
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # t = time.time()
            outs = sess.run([model.opt_op, model.loss, model.preds], 
                            feed_dict=feed_dict)
            # outs = sess.run([merged, model.opt_op, model.loss, model.preds], 
            #                 feed_dict=feed_dict)
            # outs = sess.run([merged, model.loss, model.preds],feed_dict=feed_dict)
            # train_cost = outs[2]

            # if iter % FLAGS.validate_iter == 0:
            #     # Validation
            #     # do the assign operation
            #     sess.run([val_adj_info.op, val_weight_adj_info.op, val_column_adj_info.op])
                
            #     # 如果有设置采样数量的话
            #     if FLAGS.validate_batch_size == -1:
            #         val_cost, val_f1_mic, val_f1_mac,report, duration, _ = incremental_evaluate(
            #             sess, model, minibatch, FLAGS.batch_size)
            #     else:
            #         val_cost, val_f1_mic, val_f1_mac, duration = evaluate(
            #             sess, model, minibatch, FLAGS.validate_batch_size)
               
            #     sess.run([train_adj_info.op, train_weight_adj_info.op, train_column_adj_info.op])
                
            #     epoch_val_costs[-1] += val_cost

            # if total_steps % FLAGS.print_every == 0:
            #     summary_writer.add_summary(outs[0], total_steps)

            # Print results
            # avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
            # print("train_time=", "{:.5f}".format(avg_time))


            # if total_steps % FLAGS.print_every == 0:
                # train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                # train_accuracy = calc_acc(labels,outs[-1])
                # report = classification_report(labels,outs[-1])
                # print("Iter:", '%04d' % iter,
                #       "train_loss=", "{:.5f}".format(train_cost),
                #       "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                #       "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                #       "val_loss=", "{:.5f}".format(val_cost),
                #       "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                #       "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                #       "time=", "{:.5f}".format(avg_time))
                #print(report)
            
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        
        # when each epoch ends
        # show the F1 report
        if epoch % 1 == 0:

            # sess.run([val_adj_info.op, val_weight_adj_info.op, val_column_adj_info.op])
            sess.run([val_adj_info, val_weight_adj_info, val_column_adj_info])

            # val_cost, val_f1_mic, val_f1_mac, report, duration = incremental_evaluate(
            #     sess, model, minibatch, FLAGS.batch_size)
            # area = my_incremental_evaluate(
            #     sess, model, minibatch, FLAGS.batch_size)
            # # precision, recall, thresholds = precision_recall_curve(
            # #     val_labels[:, 1], val_preds[:, 1])
            # # area2 = auc(recall, precision)

            # print("Full validation stats:",
            #       "loss=", "{:.5f}".format(val_cost),
            #       "f1_micro=", "{:.5f}".format(val_f1_mic),
            #       "f1_macro=", "{:.5f}".format(val_f1_mac),
            #       "time=", "{:.5f}".format(duration))

            # print(report)
            # print('AUC',area)

            test_cost, test_f1_mic, test_f1_mac, report, duration = incremental_evaluate(
                sess, model, minibatch, FLAGS.batch_size, test=True)
            area = my_incremental_evaluate(
                sess, model, minibatch, FLAGS.batch_size, test=True)
            # precision, recall, thresholds = precision_recall_curve(
            #     test_labels[:, 1], test_preds[:, 1])
            # area2 = auc(recall, precision)

            print("Full Test stats:",
                  "loss=", "{:.5f}".format(test_cost),
                  "f1_micro=", "{:.5f}".format(test_f1_mic),
                  "f1_macro=", "{:.5f}".format(test_f1_mac),
                  "time=", "{:.5f}".format(duration))
            print(report)
            print('AUC',area)

            # once acu > 0.82, save the model
            if area > 0.83:
                model.save(sess)
                print('AUC gotcha! model saved.')

                # np.save('../data/'+FLAGS.model+'aggr'+'_precision',precision)
                # np.save('../data/'+FLAGS.model+'aggr'+'_recall',recall)

        # 应该设置下early stopping
        if total_steps > FLAGS.max_total_steps:
            break

    # model.save(sess)
    print("Optimization Finished!")

    sess.run([val_adj_info.op, val_weight_adj_info.op, val_column_adj_info.op])
    # val_cost, val_f1_mic, val_f1_mac, report, duration, area = incremental_evaluate(
    #     sess, model, minibatch, FLAGS.batch_size)
    # area = my_incremental_evaluate(
    #     sess, model, minibatch, FLAGS.batch_size)
    # precision, recall, thresholds = precision_recall_curve(
    #     val_labels[:, 1], val_preds[:, 1])
    # area = auc(recall, precision)

    # np.save('../data/val_preds.npy', val_preds)
    # np.save('../data/val_labels.npy', val_labels)
    # np.save('../data/val_cost.npy', v_cost)

    # print("Full validation stats:",
    #       "loss=", "{:.5f}".format(val_cost),
    #       "f1_micro=", "{:.5f}".format(val_f1_mic),
    #       "f1_macro=", "{:.5f}".format(val_f1_mac),
    #       "time=", "{:.5f}".format(duration))
    # print(report)
    # print('AUC', area)

    # with open(log_dir() + "val_stats.txt", "w") as fp:
    #     fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
    #              format(val_cost, val_f1_mic, val_f1_mac, duration))


    test_cost, test_f1_mic, test_f1_mac, report, duration = incremental_evaluate(
        sess, model, minibatch, FLAGS.batch_size, test=True)
    area= my_incremental_evaluate(
        sess, model, minibatch, FLAGS.batch_size,test=True)
    # precision, recall, thresholds = precision_recall_curve(
    #     test_labels[:, 1], test_preds[:, 1])
    # area = auc(recall, precision)

    # np.save('../data/test_preds.npy', test_preds)
    # np.save('../data/test_labels.npy', test_labels)
    # np.save('../data/test_cost.npy', t_cost) # prevent from override

    print("Full Test stats:",
          "loss=", "{:.5f}".format(test_cost),
          "f1_micro=", "{:.5f}".format(test_f1_mic),
          "f1_macro=", "{:.5f}".format(test_f1_mac),
          "time=", "{:.5f}".format(duration))
    print(report)
    print('AUC:',area)
    
    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                 format(test_cost, test_f1_mic, test_f1_mac))


    # saver = tf.train.Saver()
    # save_path = saver.save(sess, "/tmp/model.ckpt")

def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print(train_data[1].shape)
    print("Done loading training data..")
    train(train_data)


if __name__ == '__main__':
    print("Model: ", FLAGS.model)
    tf.app.run()
