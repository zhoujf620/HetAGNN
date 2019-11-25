#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'cross', 'model names. See README for possible values.') # graphsage_mean
flags.DEFINE_float('learning_rate', 0.0005, 'initial learning rate.') # 0.01
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns") # small
flags.DEFINE_string('train_prefix', 'jimi', 'prefix identifying training data. must be specified.') # ''

# left to default values in main experiments
flags.DEFINE_integer('train_epochs', 100, 'number of epochs to train.') # epochs=10, 3000
flags.DEFINE_float('dropout', 0.2, 'dropout rate (1 - keep probability).') # 0.0
flags.DEFINE_float('weight_decay', 0.0005, 'weight for l2 loss on embedding matrix.')# 0.0
flags.DEFINE_integer('max_degree', 50, 'maximum node degree.') # 128
flags.DEFINE_integer('samples_1', 16, 'number of samples in layer 1') # 25
flags.DEFINE_integer('samples_2', 8, 'number of samples in layer 2') # 10
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 64, 'Size of output dim (final is 2x this, if using concat)') # 128
flags.DEFINE_integer('dim_2', 32, 'Size of output dim (final is 2x this, if using concat)') # 128
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size',512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', True, 'whether to use sigmoid loss') # False
flags.DEFINE_integer('identity_dim', 0,
                     'Set to positive value to use identity embedding features of that dimension. Default 0.')

# logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5, "how often to run a validation minibatch.") # 5000
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.") # 1
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")
