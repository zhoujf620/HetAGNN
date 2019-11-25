# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from model.layers import Layer, Dense
from model.inits import glorot, zeros


class CrossAggregator(Layer):
    """
        Aggregates via cross combine.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(CrossAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        # 写死了
        atten_dim = 64
        num_attr = 9 # 嘉琪说10 column初始为零，9的话，gather_nd 会给零值

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            
            # self.vars['initial_weights'] = glorot([102, output_dim],
            #                                       name='initial_weights')
            self.vars['alpha'] = glorot([num_attr], name='alpha')
            self.vars['self_atten'] = glorot([input_dim, atten_dim], name='self_atten')
            self.vars['neigh_atten'] = glorot([neigh_input_dim, atten_dim], name='neigh_atten')
            self.vars['v'] = glorot([atten_dim,1], name='v')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        # self_vecs, neigh_vecs = inputs
        self_vecs, neigh_vecs, neigh_weight, neigh_column = inputs
        
        # weight
        neigh_vecs = neigh_vecs * neigh_weight
        
        # attention
        alpha = tf.expand_dims(tf.gather_nd(self.vars['alpha'], neigh_column), axis=2)
        alpha_exp = tf.exp(alpha)
        atten = alpha_exp / tf.expand_dims(tf.reduce_sum(alpha_exp, axis=1), axis=1)
        neigh_vecs = neigh_vecs * atten # 不同attr的权重
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        
        # initial_vecs = tf.nn.dropout(initial_vecs, rate=self.dropout)

        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        #neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        neigh_sum = tf.reduce_sum(neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        # from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_neighs = tf.matmul(neigh_sum, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        # from_initial = tf.matmul(initial_vecs, self.vars['initial_weights'])
        # from_past = tf.add_n([from_self, from_neighs])

        # self & neighborhood attention mechanism

        self_atten_input = tf.matmul(self_vecs, self.vars["self_atten"])
        neigh_atten_input = tf.matmul(neigh_sum, self.vars["neigh_atten"])

        #Importance_self = tf.concat([from_self, from_self], axis=1)
        Importance_self = tf.add_n([self_atten_input, self_atten_input])
        #Importance_neigh = tf.concat([from_self, from_neighs], axis=1)
        Importance_neigh = tf.add_n([neigh_atten_input, self_atten_input])

        alpha_self_exp = tf.exp(tf.tanh(tf.matmul(Importance_self, self.vars['v'])))
        alpha_neigh_exp = tf.exp(tf.tanh(tf.matmul(Importance_neigh, self.vars['v'])))

        alpha_self = alpha_self_exp / (alpha_self_exp + alpha_neigh_exp)
        alpha_neigh = alpha_neigh_exp / (alpha_self_exp + alpha_neigh_exp)

        from_self = alpha_self * from_self
        from_neighs = alpha_neigh * from_neighs

        if not self.concat:
            #output = tf.reduce_max([from_initial, from_past], reduction_indices=[0])
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GeniePathAggregator(Layer):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0, bias=False, act=tf.nn.tanh, name=None,
                 concat=False, **kwargs):
        super(GeniePathAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['W_s'] = glorot([input_dim, output_dim], name='W_s')
            self.vars['W_d'] = glorot([neigh_input_dim, output_dim], name='W_d')
            self.vars['v'] = glorot([output_dim, 1], name='v')

            self.vars['W_t'] = glorot([input_dim, output_dim], name='W_t')
            self.vars['b_t'] = glorot([1, output_dim], name='b_t')

            self.vars['W_i'] = glorot([input_dim, output_dim], name='W_i')
            self.vars['b_i'] = glorot([1, output_dim], name='b_i')
            self.vars['W_f'] = glorot([input_dim, output_dim], name='W_f')
            self.vars['b_f'] = glorot([1, output_dim], name='b_f')
            self.vars['W_o'] = glorot([input_dim, output_dim], name='W_o')
            self.vars['b_o'] = glorot([1, output_dim], name='b_o')
            self.vars['W_c'] = glorot([input_dim, output_dim], name='W_c')
            self.vars['b_c'] = glorot([1, output_dim], name='b_c')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, c = inputs

        # self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        # neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)

        _, y2, y3 = neigh_vecs.shape
        y = tf.reshape(tf.matmul(tf.reshape(neigh_vecs, [-1, y3]), self.vars['W_d']), [-1, y2, y3])
        if len(self_vecs.shape) > 2:
            _, x2, x3 = self_vecs.shape
            x = tf.reshape(tf.matmul(tf.reshape(self_vecs, [-1, x3]), self.vars['W_s']), [-1, x2, x3])
            act = self.act(x + y)
        else:
            x = tf.matmul(self_vecs, self.vars['W_s'])
            act = self.act(tf.expand_dims(x, axis=1) + y)
        alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(act, [-1, y3]), \
                                                   self.vars['v']), [-1, y2, 1]), axis=1)
        output = tf.matmul(tf.reduce_sum(alpha * y, axis=1), self.vars['W_t'])  # + self.vars['b_t']
        h_tmp = self.act(output)

        # lstm_cell
        h_i = tf.nn.sigmoid(tf.matmul(h_tmp, self.vars['W_i']))
        h_f = tf.nn.sigmoid(tf.matmul(h_tmp, self.vars['W_f']))
        h_o = tf.nn.sigmoid(tf.matmul(h_tmp, self.vars['W_o']))
        h_c = self.act(tf.matmul(h_tmp, self.vars['W_c']))
        # h_i = tf.nn.sigmoid(tf.matmul(h_tmp, self.vars['W_i']) + self.vars['b_i'])
        # h_f = tf.nn.sigmoid(tf.matmul(h_tmp, self.vars['W_f']) + self.vars['b_f'])
        # h_o = tf.nn.sigmoid(tf.matmul(h_tmp, self.vars['W_o']) + self.vars['b_o'])
        # h_c = self.act(tf.matmul(h_tmp, self.vars['W_c']) + self.vars['b_c'])
        c = h_f * c + h_i * h_c
        h = h_o * self.act(c)

        return h, c


# 增加alpha 变量和计算方式
class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        # print('========== init MeanAggregator =========')
        # import traceback
        # traceback.print_stack()

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
                                               
            # self.vars['alpha'] = glorot([9], name='alpha') # 增加，居然是硬编码9！
            
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        # print('========== call MeanAggregator =========')
        # import traceback
        # traceback.print_stack()
        # self_vecs, neigh_vecs = inputs
        self_vecs, neigh_vecs, neigh_weight, neigh_column = inputs
        
        # weight
        # neigh_vecs = neigh_vecs * neigh_weight
        
        # attention
        #alpha = tf.expand_dims(tf.gather_nd(self.vars['alpha'], neigh_column), axis=2)

        #alpha_exp = tf.exp(alpha)
        #atten = alpha_exp / tf.expand_dims(tf.reduce_sum(alpha_exp, axis=1), axis=1)
        #neigh_vecs = neigh_vecs * atten

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            #output = tf.reduce_max([from_self, from_neighs],reduction_indices=[0])
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


# 增加alpha 变量和计算方式
# reduce_means 改成reduce_sum
class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                          name='neigh_weights')

            # self.vars['alpha'] = glorot([9], name='alpha') # 增加

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        # self_vecs, neigh_vecs = inputs
        self_vecs, neigh_vecs, neigh_weight, neigh_column = inputs
        
        # weight
        #neigh_vecs = neigh_vecs * neigh_weight
        
        # attention
        # alpha = tf.expand_dims(tf.gather_nd(self.vars['alpha'], neigh_column), axis=2)
        # alpha_exp = tf.exp(alpha)
        # atten = alpha_exp / tf.expand_dims(tf.reduce_sum(alpha_exp, axis=1), axis=1)
        #neigh_vecs = neigh_vecs * atten
        
        # neigh_weight = tf.exp(neigh_weight)
        # atten = neigh_weight / tf.expand_dims(tf.reduce_sum(neigh_weight, axis=1), axis=1)
        # neigh_vecs = neigh_vecs * atten
        

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        means = tf.reduce_sum( # mean改成sum
            tf.concat([neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), 
            axis=1)

        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
