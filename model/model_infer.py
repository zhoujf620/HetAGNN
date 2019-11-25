# -*- coding:utf-8 -*-

from collections import namedtuple

import os
import tensorflow as tf

import config
from model.inits import glorot, zeros
import model.layers as layers
from model.aggregators import CrossAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
GPU_MEM_FRACTION = 0.8

sess_config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
sess_config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
sess_config.allow_soft_placement = True

SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
    #  'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])
layer_infos = [SAGEInfo("node", FLAGS.samples_1, FLAGS.dim_1),
        SAGEInfo("node", FLAGS.samples_2, FLAGS.dim_2)]

# === 预测阶段确定的参数 ===
num_classes = 2 # label_map.shape[1]
feats_dim = 106 #features.shape[1]

aggregator_type = 'cross'
concat = True 
model_size = FLAGS.model_size
sigmoid_loss = FLAGS.sigmoid
identity_dim = FLAGS.identity_dim

class SupervisedGraphsage(object):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, **kwargs):

        # === from model.py ===
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        # logging = kwargs.get('logging', False)
        # self.logging = logging

        self.vars = {}
        # self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        # === set aggregator ===
        # 增加了两个cross, cross geniepath
        if aggregator_type == 'cross':
            self.aggregator_cls = CrossAggregator
        else:
            raise Exception("Unknown aggregator: ", aggregator_type)
        
        self.input_dim = feats_dim
        self.output_dim = num_classes # 2
        # self.sampler = sampler
        # self.adj_info = adj
        self.layer_infos = layer_infos
        self.concat = concat
        self.model_size = model_size
        self.sigmoid_loss = sigmoid_loss
        
        self.dims = [(self.input_dim) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])# 102, 64, 32

        self.aggregator_type = aggregator_type

        # === get info from placeholders ===
        # get info from placeholders...
        self.placeholders = self.construct_placeholders(self.input_dim, self.output_dim)

        # self.labels = self.placeholders['labels']
        # self.batch_nodes = placeholders["batch_nodes"]
        self.batch_size = self.placeholders["batch_size"]
        # self.support_size = placeholders['support_size']
        # self.features = placeholders['features']
        sampled_weight = [self.placeholders['sampled_weight_0'], 
                          self.placeholders['sampled_weight_1'], 
                          self.placeholders['sampled_weight_2']]
        sampled_column = [self.placeholders['sampled_column_0'], 
                          self.placeholders['sampled_column_1'], 
                          self.placeholders['sampled_column_2']]
        sampled_feats = [self.placeholders['sampled_feats_0'], 
                         self.placeholders['sampled_feats_1'], 
                         self.placeholders['sampled_feats_2']]
        self.data_sampled = [sampled_feats, sampled_weight, sampled_column]

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

        self.var_list = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=self.var_list)

        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.load(self.sess)


    def construct_placeholders(self, num_classes, feats_dim):
        # Define placeholders
        # 这里的key 是供 model init 用的
        # feed_dict = {placeholders: data}
        placeholders = {
            # 'features': tf.placeholder(tf.float32, shape=(None, feats_dim)), 
            # 'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
            # 'batch_nodes': tf.placeholder(tf.int32, shape=(None), name='batch_nodes'),
            'batch_size': tf.placeholder(tf.int32, name='batch_size'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            
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


    # === build computation graph ===
    def build(self):
        # data_sampled, support_sizes = self.sample(self.batch_nodes, self.layer_infos)
        support_size = 1 # [1, 8, 8*16]
        support_sizes = [support_size]
        for k in range(len(self.layer_infos)):
            t = len(self.layer_infos) - k -1
            support_size *= self.layer_infos[t].num_samples
            support_sizes.append(support_size)

        sample_size = [layer_info.num_samples for layer_info in self.layer_infos] # 16, 8

        self.outputs, self.aggregators = self.aggregate(
            self.data_sampled, self.dims, sample_size,
            support_sizes, concat=self.concat, model_size=self.model_size)
            # data_sampled, [self.features], self.dims, num_samples,
            # support_sizes, concat=self.concat, model_size=self.model_size)
        
        self.outputs = tf.nn.l2_normalize(self.outputs, 1)
        
        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.output_dim,
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x : x) # no non-linear activation
        # TF graph management
        self.node_preds = self.node_pred(self.outputs)

        # self._loss()
        # 不进行梯度修建
        # grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
        #         for grad, var in grads_and_vars]
        # self.grad, _ = clipped_grads_and_vars[0]
        # self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        # self.opt_op = self.optimizer.minimize(self.loss)

        self._predict()


    def aggregate(self, data_sampled, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name='aggregate', concat=False, model_size="small"):
   
        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        # hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        feats_hidden = data_sampled[0] # 根据index取feats
        weight_hidden = data_sampled[1]
        column_hidden = data_sampled[2]
        # feats_hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples[0]] # 根据index取feats
        # feats_hidden = [feat_samples for feat_samples in data_sampled[0]] # 根据index取feats
        # weight_hidden = [weight_samples for weight_samples in data_sampled[1]]
        # column_hidden = [column_samples for column_samples in data_sampled[2]]

        new_agg = aggregators is None
        if new_agg:
            aggregators = []

        # c_list = [] # 增加
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1: # 2*64, 32
                    aggregator = self.aggregator_cls(
                        dim_mult*dims[layer], dims[layer+1], act=lambda x : x, # no non-linear activation
                        dropout=self.placeholders['dropout'], 
                        name=name, concat=concat, model_size=model_size)
                else: # 这里aggregator.__init__() # 106 -> 64
                    aggregator = self.aggregator_cls(
                        dim_mult*dims[layer], dims[layer+1],
                        dropout=self.placeholders['dropout'], 
                        name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop], # 1, 8; 1
                              num_samples[len(num_samples) - hop - 1], # 8, 16; 8
                              dim_mult*dims[layer]] # 106, 106; 2 * 64
                
                weight_neigh_dims = [batch_size * support_sizes[hop],
                                     num_samples[len(num_samples)- hop -1],
                                     1]

                # h = aggregator((hidden[hop],
                #                 tf.reshape(hidden[hop + 1], neigh_dims)))
                # call aggregator

                # self_vecs, neigh_vecs, neigh_weight, neigh_column
                h = aggregator((
                    feats_hidden[hop],  
                    tf.reshape(feats_hidden[hop + 1], neigh_dims), # [1,8,106], [8, 16, 106], [1, 8, 2*64]
                    tf.reshape(weight_hidden[hop + 1], weight_neigh_dims),
                    tf.reshape(column_hidden[hop + 1], weight_neigh_dims)))
            
                next_hidden.append(h)

            feats_hidden = next_hidden
            #self.hiddenOutput.append(hidden[0])
        return feats_hidden[0], aggregators


    # def _loss(self):
    #     # Weight decay loss
    #     for aggregator in self.aggregators:
    #         for var in aggregator.vars.values():
    #             self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
    #     for var in self.node_pred.vars.values():
    #         self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
    #     # classification loss
    #     if self.sigmoid_loss:
    #         self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #                 logits=self.node_preds,
    #                 labels=self.labels))
    #     else:
    #         # 变成v2
    #         self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    #                 logits=self.node_preds,
    #                 labels=self.labels))

    #     # tf.summary.scalar('loss', self.loss)
   
    def _predict(self):
        if self.sigmoid_loss:
            self.preds =  tf.nn.sigmoid(self.node_preds)
        else:
            self.preds = tf.nn.softmax(self.node_preds)

    # === 以上是计算图部分 ===

    def predict(self, feed_dict):
        preds = self.sess.run([self.preds],
                            feed_dict=feed_dict)
        return preds
        
    def close_sess(self):
        self.sess.close()

    # def save(self, sess=None):
    #     if not sess:
    #         raise AttributeError("TensorFlow session not provided.")

    #     saver = tf.train.Saver(var_list=self.var_list)

    #     save_path = "./data/model/%s.ckpt" %(self.aggregator_type)
    #     saver.restore(sess, save_path)
    #     print("Model saved in file: %s" % save_path)


    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        # saver = tf.train.Saver(reshape=True)
        
        # saver = tf.train.Saver(var_list=self.var_list)

        # saver = tf.train.Saver()/
        # 不能硬编码啊
        save_path = "./data/model/%s.ckpt" %(self.aggregator_type)
        self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

        # ckpt_path = './data/model/%s.ckpt'%(self.aggregator_type)
        # meta_path = ckpt_path + '.meta'
