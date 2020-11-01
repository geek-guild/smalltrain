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



class TwoDimCNNModelV2(TwoDimCNNModel):

    MODEL_ID_2D_CNN = '2D_CNN_V2'
    MODEL_ID = MODEL_ID_2D_CNN

    def __init__(self):
        return


    def pool_residual(self, x, pool):
        pooled_residual = tf.nn.avg_pool(x, ksize=[1, pool, pool, 1],
                                    strides=[1, pool, pool, 1], padding='SAME')
        print('{} pooled_residual shape:{}'.format(self.MODEL_ID, pooled_residual.shape))
        return pooled_residual

    def pad_residual(self, x, pad_channel_size):

        pad_before = pad_channel_size // 2
        pad_after = pad_channel_size - pad_before
        print('pad_channel_size:{}, pad_before:{}, padd_after:{}'.format(pad_channel_size, pad_before, pad_after))
        padded_residual = tf.pad(x, [[0, 0], [0, 0], [0, 0], [pad_before, pad_after]])
        print('{} padded_residual shape:{}'.format(self.MODEL_ID, padded_residual.shape))
        return padded_residual
