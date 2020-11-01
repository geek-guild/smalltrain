from dateutil.parser import parse as parse_datetime
from datetime import timezone
from datetime import timedelta
from datetime import datetime
import time
import pandas as pd
import numpy as np
import math
import csv
import sys
import os

# import tensorflow as tf
import tensorflow.compat.v1 as tf


from smalltrain.data_set.img_data_set import IMGDataSet
from smalltrain.model.nn_model import NNModel
from smalltrain.model.two_dim_cnn_model import TwoDimCNNModel

import ggutils.gif_util as gif_util
import ggutils.s3_access as s3_access


# MODEL_ID_4NN = '4NN_20180808' # 4 nn model 2019/09/10
# MODEL_ID_DNN = 'DNN' # 4 nn model 2019/09/10
# MODEL_ID_1D_CNN = '1D_CNN'
# MODEL_ID_CC = 'CC' # Carbon Copy
# MODEL_ID_LIST = [MODEL_ID_4NN, MODEL_ID_DNN, MODEL_ID_1D_CNN, MODEL_ID_CC]

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

import re

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', True,
                            """Train the model using fp16.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


class TensorflowSampleCNNModel(TwoDimCNNModel):

    MODEL_ID_TF_SAMPLE_CNN = 'TF_SAMPLE_CNN'
    MODEL_ID = MODEL_ID_TF_SAMPLE_CNN

    def __init__(self):
        return

    def construct_model(self, log_dir_path, model_id=None, train_data=None, debug_mode=True, prediction_mode=False, hparams=None):

        PREFIX = '[TensorflowSampleCNNModel]'
        print('{}__init__'.format(PREFIX))

        return super().construct_model(log_dir_path, model_id, train_data, debug_mode, prediction_mode, hparams)

    def define_2d_cnn_model(self, n_layer=5, has_res_net=False, has_non_cnn_net=False):

        input_width = self.input_width
        col_size = self.col_size


        with tf.name_scope('model/'):
            # cnn_input_channels = ts_col_size
            # cnn_input_channels = col_size

            self.x = tf.placeholder(tf.float32, shape=[None, input_width, input_width, col_size])
            self.y_ = tf.placeholder(tf.float32, shape=[None, self.output_classes])

            # conv1
            with tf.variable_scope('conv1') as scope:
                W_conv1 = self.weight_variable([5, 5, 3, 64], name="weights", stddev=5e-2)

                _conv1 = tf.nn.conv2d(self.x, W_conv1, [1, 1, 1, 1], padding='SAME')
                b_conv1 = self.bias_variable([64], name="biases", value=0.0)
                pre_activation = tf.nn.bias_add(_conv1, b_conv1)
                conv1 = tf.nn.relu(pre_activation, name=scope.name)
                _activation_summary(conv1)

            # pool1
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool1')
            # norm1
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm1')
            norm1 = tf.nn.dropout(norm1, self.keep_prob)

            # conv2
            with tf.variable_scope('conv2') as scope:
                W_conv2 = self.weight_variable([5, 5, 64, 64], name="weights", stddev=5e-2)
                _conv2 = tf.nn.conv2d(norm1, W_conv2, [1, 1, 1, 1], padding='SAME')
                b_conv2 = self.bias_variable([64], name="biases")
                pre_activation = tf.nn.bias_add(_conv2, b_conv2)
                conv2 = tf.nn.relu(pre_activation, name=scope.name)
                _activation_summary(conv2)

            # norm2
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
            # pool2
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            pool2 = tf.nn.dropout(pool2, self.keep_prob)

            # local3
            with tf.variable_scope('local3') as scope:
                # Move everything into depth so we can perform a single matrix multiply.
                reshape = tf.keras.layers.Flatten()(pool2)
                dim = reshape.get_shape()[1].value
                W_local3 = self.weight_variable([dim, 384], name="weights", stddev=0.04)
                b_local3 = self.bias_variable([384], name="biases")
                local3 = tf.nn.relu(tf.matmul(reshape, W_local3) + b_local3, name=scope.name)
                _activation_summary(local3)
                local3 = tf.nn.dropout(local3, self.keep_prob)

            # local4
            with tf.variable_scope('local4') as scope:
                W_local4 = self.weight_variable([384, 192], name="weights", stddev=0.04)
                b_local4 = self.bias_variable([192], name="biases")

                local4 = tf.nn.relu(tf.matmul(local3, W_local4) + b_local4, name=scope.name)
                _activation_summary(local4)
                local4 = tf.nn.dropout(local4, self.keep_prob)

            # linear layer(WX + b),
            # We don't apply softmax here because
            # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
            # and performs the softmax internally for efficiency.
            with tf.variable_scope('softmax_linear') as scope:
                W_sl = self.weight_variable([192, NUM_CLASSES], name="weights", stddev=1/192.0)
                b_sl = self.bias_variable([NUM_CLASSES], name="biases", value=0.0)
                softmax_linear = tf.add(tf.matmul(local4, W_sl), b_sl, name=scope.name)
                _activation_summary(softmax_linear)

            return softmax_linear
            # return output_middle_layer


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
