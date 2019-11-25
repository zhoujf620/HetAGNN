# -*- coding:utf-8 -*-

from collections import namedtuple

import tensorflow as tf

from model.inits import glorot, zeros
import model.layers as layers
from model.aggregators import MeanAggregator,CrossAggregator, GCNAggregator, GeniePathAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS


SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
    #  'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])


# __init__() 增加了两个aggregator, self.aggregator_type
# build() 注释了一段，不做梯度修剪
# _loss() 改成softmax_cross_entropy_with_logits_v2
# 增加save(), load() 两个函数

# class SupervisedGraphsage(models.SampleAndAggregate):
class SupervisedGraphsage(object):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, placeholders, input_dim, output_dim, sampler, 
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''
        # 越级调用GeneralizedModel.__init__ 不是 super(SupervisedGraphsage, self).__init__
        # models.GeneralizedModel.__init__(self, **kwargs)
        
        # === from model.py ===
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

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
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif aggregator_type == 'cross':
            self.aggregator_cls = CrossAggregator
        elif aggregator_type == "geniepath":
            self.aggregator_cls = GeniePathAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)
        
        self.aggregator_type = aggregator_type

        # === get info from placeholders ===
        # get info from placeholders...
        self.placeholders = placeholders
        # self.features = placeholders['features']
        self.labels = placeholders['labels']
        self.batch_nodes = placeholders["batch_nodes"]
        self.batch_size = placeholders["batch_size"]

        self.input_dim = input_dim
        self.output_dim = output_dim # 2
        self.sampler = sampler
        # self.adj_info = adj
        self.layer_infos = layer_infos
        self.concat = concat
        self.model_size = model_size
        self.sigmoid_loss = sigmoid_loss
        
        # if identity_dim > 0: # onehot embedding
        #     self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        # else:
        #     self.embeds = None
        # if features is None: 
        #     if identity_dim == 0:
        #         raise Exception("Must have a positive value for identity feature dimension if no input features given.")
        #     self.features = self.embeds
        # else:
        #     self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        #     if not self.embeds is None:
        #         self.features = tf.concat([self.embeds, self.features], axis=1)

        self.dims = [(input_dim) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])# 102, 64, 32

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()
        
        # self.var_list = tf.trainable_variables()
        var_list = [
            self.aggregators[0].vars['neigh_weights'], 
            self.aggregators[0].vars['self_weights'],
            # self.aggregators[0].vars['initial_weights'], 
            self.aggregators[0].vars['alpha'],
            self.aggregators[0].vars['self_atten'], 
            self.aggregators[0].vars['neigh_atten'],
            self.aggregators[0].vars['v'],
            self.aggregators[1].vars['neigh_weights'], 
            self.aggregators[1].vars['self_weights'],
            # self.aggregators[1].vars['initial_weights'], 
            self.aggregators[1].vars['alpha'],
            self.aggregators[1].vars['self_atten'], 
            self.aggregators[1].vars['neigh_atten'],
            self.aggregators[1].vars['v'],

            self.node_pred.vars['weights'], 
            self.node_pred.vars['bias']]
        self.saver = tf.train.Saver(var_list=var_list)
        
    def build(self):
        data_sampled, support_sizes = self.sample(self.batch_nodes, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos] # 16, 8
        
        self.outputs, self.aggregators = self.aggregate(
            data_sampled, self.dims, num_samples,
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

        self._loss()
        # 不进行梯度修建
        # grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
        #         for grad, var in grads_and_vars]
        # self.grad, _ = clipped_grads_and_vars[0]
        # self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.opt_op = self.optimizer.minimize(self.loss)

        self.preds = self.predict()


    def sample(self, batch_nodes, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        
        if batch_size is None:
            batch_size = self.batch_size

        # inputs1 = placeholders["batch"]
        # samples = [inputs] 
        sample_nodes  = [batch_nodes]
        sample_weight = [batch_nodes]
        sample_column = [batch_nodes]
        sample_feats = [self.sampler.sample_feats(batch_nodes)]

        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size] # [1, 8, 8*16]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1 # len(layer)-1: 0
            support_size *= layer_infos[t].num_samples
            
            # TODO  每一个sampler 还不一样的吗, 要多难受就有多难受
            # sampler = layer_infos[t].neigh_sampler
            data_sampled = self.sampler.sample(sample_nodes[k], layer_infos[t].num_samples)

            # sample_nodes 是index 其他的是weight, column 值
            # 但还只是sample num_samples 个啊
            sample_nodes.append(data_sampled[0])
            sample_weight.append(data_sampled[1])
            sample_column.append(data_sampled[2]) 
            sample_feats.append(data_sampled[3])
            # sample_nodes.append(tf.reshape(node[0], [batch_size*support_size, ]))
            # sample_weight.append(tf.reshape(node[1], [batch_size*support_size, ]))
            # sample_column.append(tf.reshape(node[2], [batch_size*support_size, ])) 
            
            support_sizes.append(support_size)
        return [sample_nodes, sample_weight, sample_column, sample_feats], support_sizes


    # aggregate 增加init_h()，增加init_hidden, weight_hidden, column_hidden, 
    #   aggregator 初始化增加一个dims[0]维度
    #   h 初始化
    def aggregate(self, data_sampled, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name='aggregate', concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """
        if batch_size is None:
            batch_size = self.batch_size

        # 增加 init_h
        # def init_h(hidden):
        #     new_hidden = []
        #     with tf.variable_scope(name + '_vars'):
        #         self.vars['W_x'] = glorot([dims[0], dims[1]], name='W_x')
        #         self.vars['b_x'] = zeros([dims[1]], name='b_x')
        #     for i in range(len(hidden)):
        #         new_hidden.append(tf.matmul(hidden[i], self.vars['W_x']) + self.vars['b_x'])
        #     return new_hidden

        # length: number of layers + 1
        # hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        feats_hidden = [feat_samples for feat_samples in data_sampled[3]] # 根据index取feats
        # feats_hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples[0]] # 根据index取feats
        weight_hidden = [weight_samples for weight_samples in data_sampled[1]]
        column_hidden = [column_samples for column_samples in data_sampled[2]]
        # if self.aggregator_cls == GeniePathAggregator:
        #     hidden = init_h(hidden)
        #     dims[0] = dims[1]

        new_agg = aggregators is None
        if new_agg:
            aggregators = []

        # c_list = [] # 增加
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(
                        dim_mult*dims[layer], dims[layer+1], act=lambda x : x, # no non-linear activation
                        dropout=self.placeholders['dropout'], 
                        name=name, concat=concat, model_size=model_size)
                else: # 这里aggregator.__init__()
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
                neigh_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]
                
                weight_neigh_dims = [batch_size * support_sizes[hop],
                                     num_samples[len(num_samples)- hop -1],
                                     1]

                # h = aggregator((hidden[hop],
                #                 tf.reshape(hidden[hop + 1], neigh_dims)))
                # call aggregator
                h = aggregator((
                    feats_hidden[hop],  
                    tf.reshape(feats_hidden[hop + 1], neigh_dims),
                    tf.reshape(weight_hidden[hop + 1], weight_neigh_dims),
                    tf.reshape(column_hidden[hop + 1], weight_neigh_dims)))
            
            # === 以下的代码都没动 ====
                next_hidden.append(h)

            feats_hidden = next_hidden
            #self.hiddenOutput.append(hidden[0])
        return feats_hidden[0], aggregators


    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.labels))
        else:
            # 变成v2
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.node_preds,
                    labels=self.labels))

        # tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)

    # 多了save, load。
    # just for gcn alpha。
    # def save(self, sess=None):
    #     if not sess:
    #         raise AttributeError("TensorFlow session not provided.")

    #     #saver = tf.train.Saver(reshape=True)
    #     saver = tf.train.Saver(
    #         [self.aggregators[0].vars['weights'], self.aggregators[0].vars['alpha'],
    #          self.aggregators[1].vars['weights'], self.aggregators[1].vars['alpha']])
    #     #saver = tf.train.Saver()

    #     save_path = saver.save(sess, "../data/tmp/%s.ckpt" % self.aggregator_type)
    #     print("Model saved in file: %s" % save_path)

    # def load(self, sess=None):
    #     if not sess:
    #         raise AttributeError("TensorFlow session not provided.")
    #     #saver = tf.train.Saver(reshape=True)
    #     saver = tf.train.Saver(
    #         [self.aggregators[0].vars['weights'], self.aggregators[0].vars['alpha'],
    #          self.aggregators[1].vars['weights'], self.aggregators[1].vars['alpha']])

    #     #saver = tf.train.Saver()
    #     save_path = "../data/tmp/%s.ckpt" % self.aggregator_type
    #     saver.restore(sess, save_path)
    #     print("Model restored from file: %s" % save_path)


    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")

        save_path = self.saver.save(sess, "./data/model/%s.ckpt" %(self.aggregator_type))
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        # saver = tf.train.Saver(reshape=True)

        # saver = tf.train.Saver()/
        # 不能硬编码啊
        save_path = "./data/model/%s.ckpt" %(self.aggregator_type)
        self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
