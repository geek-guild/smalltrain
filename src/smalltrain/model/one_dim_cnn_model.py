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


from smalltrain.data_set.ts_data_set import TSDataSet
from smalltrain.model.nn_model import NNModel
import ggutils.gif_util as gif_util
import ggutils.s3_access as s3_access


# MODEL_ID_4NN = '4NN_20180808' # 4 nn model 2019/09/10
# MODEL_ID_DNN = 'DNN' # 4 nn model 2019/09/10
# MODEL_ID_1D_CNN = '1D_CNN'
# MODEL_ID_CC = 'CC' # Carbon Copy
# MODEL_ID_LIST = [MODEL_ID_4NN, MODEL_ID_DNN, MODEL_ID_1D_CNN, MODEL_ID_CC]

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops



class OneDimCNNModel(NNModel):

    MODEL_ID_1D_CNN = '1D_CNN'
    MODEL_ID = MODEL_ID_1D_CNN

    def __init__(self):
        return

    # def construct_model(self, log_dir_path, model_id=None, train_data=None, debug_mode=True, prediction_mode=False, hparams=None):
    def set_params(self, log_dir_path, model_id=None, train_data=None, debug_mode=True, prediction_mode=False,
                            hparams=None):

        PREFIX = '[OneDimCNNModel]'
        print('{}__init__'.format(PREFIX))

        self.debug_mode = debug_mode
        # update by hparams
        self.hparams = hparams

        self.trainable_variables = None

        self.model_type = 'CLASSIFICATION'
        if hparams and 'model_type' in hparams.keys():
            print('{}Use model_type in hparams:{}'.format(PREFIX, hparams['model_type']))
            self.model_type = hparams['model_type']
        else:
            print('{}TODO Use ts_start with default value:{}'.format(PREFIX, self.model_type))
        self.prediction_mode = prediction_mode

        # about optimizer

        self.optimizer = 'AdamOptimizer' # Default Optimizer
        if hparams and 'optimizer' in hparams.keys():
            print('{}Use optimizer in hparams:{}'.format(PREFIX, hparams['optimizer']))
            self.optimizer = hparams['optimizer']
        if self.optimizer is None or self.optimizer not in NNModel.AVAILABLE_OPTIMIZER_LIST:
            self.optimizer = NNModel.DEFAULT_OPTIMIZER
            print('{}Use optimizer with default value:{}'.format(PREFIX, self.optimizer))

        self.l1_norm = 0
        # whether add l1_norm_reg or not
        self.add_l1_norm_reg = False
        if hparams and 'add_l1_norm_reg' in hparams.keys():
            print('{}Use add_l1_norm_reg in hparams:{}'.format(PREFIX, hparams['add_l1_norm_reg']))
            self.add_l1_norm_reg = hparams['add_l1_norm_reg']
        if self.add_l1_norm_reg is None:
            self.add_l1_norm_reg = False

        # preactivation regularization
        self.preactivation_regularization_value = 0.0
        self.add_preactivation_regularization = False
        if hparams and 'add_preactivation_regularization' in hparams.keys():
            print('{}Use add_preactivation_regularization in hparams:{}'.format(PREFIX, hparams['add_preactivation_regularization']))
            self.add_preactivation_regularization = hparams['add_preactivation_regularization']
        if self.add_preactivation_regularization is None:
            self.add_preactivation_regularization = False

        self.preactivation_regularization_value_ratio = 0.0
        if hparams and 'preactivation_regularization_value_ratio' in hparams.keys():
            print('{}Use preactivation_regularization_value_ratio in hparams:{}'.format(PREFIX, hparams['preactivation_regularization_value_ratio']))
            self.preactivation_regularization_value_ratio = hparams['preactivation_regularization_value_ratio']
            try:
                self.preactivation_regularization_value_ratio = np.float32(self.preactivation_regularization_value_ratio)
            except ValueError:
                self.preactivation_regularization_value_ratio = 0.0
                print('{}Use preactivation_regularization_value_ratio with default value:{}'.format(PREFIX, self.preactivation_regularization_value_ratio))
        else:
            print('{}Use preactivation_regularization_value_ratio with default value:{}'.format(PREFIX, self.preactivation_regularization_value_ratio))


        # self.preactivation_maxout_list = [300.0, 200.0, 54.0, 18.0, 6.0, 18.0, 54.0, 200.0, 300.0, 300.0, 300.0]
        self.preactivation_maxout_list = None
        if hparams and 'preactivation_maxout_list' in hparams.keys():
            print('{}Use preactivation_maxout_list in hparams:{}'.format(PREFIX, hparams['preactivation_maxout_list']))
            self.preactivation_maxout_list = hparams['preactivation_maxout_list']
            try:
                assert len(self.preactivation_maxout_list) > 0
            except (AssertionError, TypeError):
                self.preactivation_maxout_list = None
                print('{}Use preactivation_maxout_list with default value:{}'.format(PREFIX, self.preactivation_maxout_list))
        else:
            print('{}Use preactivation_maxout_list with default value:{}'.format(PREFIX, self.preactivation_maxout_list))

        self.train_data = train_data

        # Set col_size from
        # 1. hparams.get('col_size')
        # 2. data_set.col_size
        self.col_size = hparams.get('col_size')
        if self.col_size is None:
            try:
                self.col_size = self.data_set.col_size
            except AttributeError:
                self.col_size = None

        # update by hparams

        # (For compatibility with ver0.1.1 ```input_ts_size``` and ver0.1.2 ```input_ts_width``` )
        if hparams and 'input_ts_size' in hparams.keys():
            print('{}Use input_ts_size in hparams:{}'.format(PREFIX, hparams['input_ts_size']))
            self.input_ts_size = hparams['input_ts_size']
        else:
            print('{}TODO Use input_ts_size with default value'.format(PREFIX))

        self.input_ts_width = self.input_ts_size # (For compatibility with ver0.1.1 ```input_ts_size``` and ver0.1.2 ```input_ts_width``` )
        if hparams and 'input_ts_width' in hparams.keys():
            print('{}Use input_ts_width in hparams:{}'.format(PREFIX, hparams['input_ts_width']))
            self.input_ts_width = hparams['input_ts_width']
        else:
            print('{}TODO Use input_ts_width with default value'.format(PREFIX))

        if self.input_ts_width is None:
            self.input_ts_width = self.input_ts_size  # (For compatibility with ver0.1.1 ```input_ts_size``` and ver0.1.2 ```input_ts_width``` )
            print('{}Use input_ts_width same as input_ts_size:{}'.format(PREFIX, self.input_ts_width))

        self.input_width = self.input_ts_width

        if hparams and 'n_layer' in hparams.keys():
            print('{}Use n_layer in hparams:{}'.format(PREFIX, hparams['n_layer']))
            self.n_layer = hparams['n_layer']
        else:
            print('{}TODO Use n_layer with default value'.format(PREFIX))

        self.filter_width = 5
        if hparams and 'filter_width' in hparams.keys():
            print('{}Use filter_width in hparams:{}'.format(PREFIX, hparams['filter_width']))
            self.filter_width = hparams['filter_width']
        else:
            print('{}Use filter_width with default value:{}'.format(PREFIX, self.filter_width))

        self.cnn_channel_size = 4
        if hparams and 'cnn_channel_size' in hparams.keys():
            print('{}Use cnn_channel_size in hparams:{}'.format(PREFIX, hparams['cnn_channel_size']))
            self.cnn_channel_size = hparams['cnn_channel_size']
        else:
            print('{}TODO Use cnn_channel_size with default value'.format(PREFIX))

        self.cnn_channel_size_list = None
        if hparams and 'cnn_channel_size_list' in hparams.keys():
            print('{}Use cnn_channel_size_list in hparams:{}'.format(PREFIX, hparams['cnn_channel_size_list']))
            self.cnn_channel_size_list = hparams['cnn_channel_size_list']
        else:
            print('{}Use cnn_channel_size with default value:{}'.format(PREFIX, self.cnn_channel_size_list))

        self.pool_size_list = None
        if hparams and 'pool_size_list' in hparams.keys():
            print('{}Use pool_size_list in hparams:{}'.format(PREFIX, hparams['pool_size_list']))
            self.pool_size_list = hparams['pool_size_list']
        if self.pool_size_list is None:
            self.pool_size_list = np.ones([self.n_layer], dtype="int32")
            self.pool_size_list[0:1] = 2
            print('{}Use pool_size_list with default value:{}'.format(PREFIX, self.pool_size_list))

        self.act_func_list = None

        if hparams and 'act_func_list' in hparams.keys():
            print('{}Use act_func_list in hparams:{}'.format(PREFIX, hparams['act_func_list']))
            self.act_func_list = hparams['act_func_list']
        if self.act_func_list is None:
            self.act_func_list = np.repeat(NNModel.DEFAULT_ACT_FUNC_KEY, [self.n_layer - 1])
            print('{}Use act_func_list with default value:{}'.format(PREFIX, self.act_func_list))
        self.act_func_ref_list = self.set_act_func_ref_list(self.act_func_list, self.n_layer)
        print('{}act_func_ref_list is set :{}'.format(PREFIX, self.act_func_ref_list))

        self.test_only_mode = False
        if hparams and 'test_only_mode' in hparams.keys():
            print('{}Use test_only_mode in hparams:{}'.format(PREFIX, hparams['test_only_mode']))
            self.test_only_mode = hparams['test_only_mode']
        else:
            print('{}TODO Use test_only_mode with default value:{}'.format(PREFIX, self.test_only_mode))

        # whether has ResNet or not
        self.has_res_net = False
        if hparams and 'has_res_net' in hparams.keys():
            print('{}Use has_res_net in hparams:{}'.format(PREFIX, hparams['has_res_net']))
            self.has_res_net = hparams['has_res_net']
        else:
            print('{}Use has_res_net with default value:{}'.format(PREFIX, self.has_res_net))

        # about batch normalization
        self.has_batch_norm = True
        if hparams and 'has_batch_norm' in hparams.keys():
            print('{}Use has_batch_norm in hparams:{}'.format(PREFIX, hparams['has_batch_norm']))
            self.has_batch_norm = hparams['has_batch_norm']
        else:
            print('{}TODO Use has_batch_norm with default value:{}'.format(PREFIX, self.has_batch_norm))

        if self.has_batch_norm:
            self.bn_decay = NNModel.DEFAULT_BN_DECAY

            if hparams and 'bn_decay' in hparams.keys():
                print('{}Use bn_decay in hparams:{}'.format(PREFIX, hparams['bn_decay']))
                self.bn_decay = hparams['bn_decay']
            else:
                print('{}TODO Use bn_decay with default value:{}'.format(PREFIX, self.bn_decay))

            self.bn_eps = NNModel.DEFAULT_BN_ESP
            if hparams and 'bn_eps' in hparams.keys():
                print('{}Use bn_eps in hparams:{}'.format(PREFIX, hparams['bn_eps']))
                self.bn_eps = hparams['bn_eps']
            else:
                print('{}TODO Use bn_eps with default value:{}'.format(PREFIX, self.bn_eps))

        self.annotation_col_names = None
        if hparams and 'annotation_col_names' in hparams.keys():
            print('{}Use annotation_col_names in hparams:{}'.format(PREFIX, hparams['annotation_col_names']))
            self.annotation_col_names = hparams['annotation_col_names']

        self.annotation_col_size = 0
        if self.annotation_col_names is not None:
            self.annotation_col_size = len(self.annotation_col_names)

        # about mask_rate
        self.mask_rate = None
        if hparams and 'mask_rate' in hparams.keys():
            print('{}Use mask_rate in hparams:{}'.format(PREFIX, hparams['mask_rate']))
            self.mask_rate = hparams['mask_rate']
        if self.mask_rate is not None:
            try:
                self.mask_rate = float(self.mask_rate)
            except ValueError:
                print('{}mask_rate is not float type. reset with None'.format(PREFIX))
                self.mask_rate = None

        # output_data_names
        if hparams and 'output_data_names' in hparams.keys():
            print('{}Use output_data_names in hparams:{}'.format(PREFIX, hparams['output_data_names']))
            self.output_data_names = hparams['output_data_names']
        if self.output_data_names is not None:
            try:
                if not isinstance(self.output_data_names, list):
                    raise ValueError
                print('output_data_names size:{}'.format(len(self.output_data_names)))
            except ValueError:
                print('{}output_data_names is not list type. reset with None'.format(PREFIX))
                self.output_data_names = None


        self.restore_var_name_list = None
        if hparams and 'restore_var_name_list' in hparams.keys():
            print('{}Use restore_var_name_list in hparams:{}'.format(PREFIX, hparams['restore_var_name_list']))
            self.restore_var_name_list = hparams['restore_var_name_list']

        self.untrainable_var_name_list = None
        if hparams and 'untrainable_var_name_list' in hparams.keys():
            print('{}Use untrainable_var_name_list in hparams:{}'.format(PREFIX, hparams['untrainable_var_name_list']))
            self.untrainable_var_name_list = hparams['untrainable_var_name_list']

        # plot settings
        self.plot_x_label = None
        if hparams and 'plot_x_label' in hparams.keys():
            print('{}Use plot_x_label in hparams:{}'.format(PREFIX, hparams['plot_x_label']))
            self.plot_x_label = hparams['plot_x_label']
        self.plot_y_label = None
        if hparams and 'plot_y_label' in hparams.keys():
            print('{}Use plot_y_label in hparams:{}'.format(PREFIX, hparams['plot_y_label']))
            self.plot_y_label = hparams['plot_y_label']
        self.plot_x_data_name_in_annotation = None
        if hparams and 'plot_x_data_name_in_annotation' in hparams.keys():
            print('{}Use plot_x_data_name_in_annotation in hparams:{}'.format(PREFIX, hparams['plot_x_data_name_in_annotation']))
            self.plot_x_data_name_in_annotation = hparams['plot_x_data_name_in_annotation']
        self.plot_group_data_name_in_annotation = None
        if hparams and 'plot_group_data_name_in_annotation' in hparams.keys():
            print('{}Use plot_group_data_name_in_annotation in hparams:{}'.format(PREFIX, hparams['plot_group_data_name_in_annotation']))
            self.plot_group_data_name_in_annotation = hparams['plot_group_data_name_in_annotation']
        self.plot_x_range = None
        if hparams and 'plot_x_range' in hparams.keys():
            print('{}Use plot_x_range in hparams:{}'.format(PREFIX, hparams['plot_x_range']))
            self.plot_x_range = hparams['plot_x_range']
        self.plot_y_range = None
        if hparams and 'plot_y_range' in hparams.keys():
            print('{}Use plot_y_range in hparams:{}'.format(PREFIX, hparams['plot_y_range']))
            self.plot_y_range = hparams['plot_y_range']
        self.plot_title = None
        if hparams and 'plot_title' in hparams.keys():
            print('{}Use plot_title in hparams:{}'.format(PREFIX, hparams['plot_title']))
            self.plot_title = hparams['plot_title']
        self.plot_errors = None
        if hparams and 'plot_errors' in hparams.keys():
            print('{}Use plot_errors in hparams:{}'.format(PREFIX, hparams['plot_errors']))
            self.plot_errors = hparams['plot_errors']
        self.plot_animation = False
        if hparams and 'plot_animation' in hparams.keys():
            print('{}Use plot_animation in hparams:{}'.format(PREFIX, hparams['plot_animation']))
            self.plot_animation = hparams['plot_animation']
        if self.plot_animation is None:
            self.plot_animation = False
            print('{}Use plot_animation with default value:{}'.format(PREFIX, self.plot_animation))
        self.calc_cc_errors = False
        if hparams and 'calc_cc_errors' in hparams.keys():
            print('{}Use calc_cc_errors in hparams:{}'.format(PREFIX, hparams['calc_cc_errors']))
            self.calc_cc_errors = hparams['calc_cc_errors']
        if self.calc_cc_errors is None:
            self.calc_cc_errors = False
            print('{}Use calc_cc_errors with default value:{}'.format(PREFIX, self.calc_cc_errors))

        self.op_errors = None
        if hparams and 'op_errors' in hparams.keys():
            print('{}Use op_errors in hparams:{}'.format(PREFIX, hparams['op_errors']))
            self.op_errors = hparams['op_errors']

        # rank_boundary_list
        self.rank_boundary_list = None
        if hparams and 'rank_boundary_list' in hparams.keys():
            print('{}Use rank_boundary_list in hparams:{}'.format(PREFIX, hparams['rank_boundary_list']))
            self.rank_boundary_list = hparams['rank_boundary_list']
            if self.rank_boundary_list is not None:
                # check the members of rank_boundary_list
                len_of_rank_boundary_list = len(self.rank_boundary_list)
                if len_of_rank_boundary_list < 1:
                    self.rank_boundary_list = None
                for rank_boundary in self.rank_boundary_list:
                    try:
                        assert len(rank_boundary) > 1
                        lower = rank_boundary[0]
                        upper = rank_boundary[1]
                        print('{}rank_boundary lower:{}, func:{}'.format(PREFIX, lower, upper))
                    except Exception as e:
                        print('{}No rank_boundary_list is set because of error {} on invalid parameter:{}'.format(PREFIX, e, rank_boundary))
        else:
            print('{}No rank_boundary_list is set'.format(PREFIX))

        # cloud settings
        self.cloud_root = None
        if hparams and 'cloud_root' in hparams.keys():
            print('{}Use cloud_root in hparams:{}'.format(PREFIX, hparams['cloud_root']))
            self.cloud_root = hparams['cloud_root']
        self.prioritize_cloud = False
        if hparams and 'prioritize_cloud' in hparams.keys():
            print('{}Use prioritize_cloud in hparams:{}'.format(PREFIX, hparams['prioritize_cloud']))
            self.prioritize_cloud = hparams['prioritize_cloud']
        if self.prioritize_cloud is None:
            self.prioritize_cloud = False
            print('{}Use prioritize_cloud with default value:{}'.format(PREFIX, self.prioritize_cloud))

        # local setting

        self.save_root_dir = '/var/tensorflow/tsp/'
        if hparams and 'save_root_dir' in hparams.keys():
            print('{}Use save_root_dir in hparams:{}'.format(PREFIX, hparams['save_root_dir']))
            self.save_root_dir = hparams['save_root_dir']
        else:
            print('{}TODO Use save_root_dir with default value'.format(PREFIX))

        # check init model
        self.sess = tf.InteractiveSession()
        self.init_model_path = None
        if hparams and 'init_model_path' in hparams.keys():
            print('{}Use init_model_path in hparams:{}'.format(PREFIX, hparams['init_model_path']))
            self.init_model_path = hparams['init_model_path']
        # set output_classes in CLASSIFICATION model
        self.output_classes = None
        if hparams and 'output_classes' in hparams.keys():
            print('{}Use output_classes in hparams:{}'.format(PREFIX, hparams['output_classes']))
            self.output_classes = hparams['output_classes']
        # if output_classes is not set in CLASSIFICATION model, try to read from init_model_path
        if self.init_model_path is not None and self.model_type == 'CLASSIFICATION':
            self.output_classes = self.get_output_classes_from_model(self.init_model_path)
            hparams['output_classes'] = self.output_classes

        self.log_dir_path = log_dir_path
        self.result_sum = []

        # Move to NNModel.prepare_model
        # self.sess = tf.InteractiveSession()
        # self.define_model()
        # print('---------- time:{} DONE define_model'.format(time.time() - last_time))
        # last_time = time.time()
        # self.saver = tf.train.Saver(var_list=None, max_to_keep=None)
        # self.global_iter = 0
        # self.sess.run(tf.global_variables_initializer())
        # if self.untrainable_var_name_list is not None:
        #     self.trainable_variables = self.remove_trainable(self.untrainable_var_name_list)
        #     self.set_optimizer()
        # # restore model
        # if self.init_model_path is not None:
        #     print('restore model from {}'.format(self.init_model_path))
        #     has_restored = self.restore(self.init_model_path, self.restore_var_name_list)
        #     print('has_restored:', has_restored)
        #     # if it has not been restored, then the model will be initialized with Prob dist.
        # print('---------- time:{} DONE init model'.format(time.time() - last_time))
        # last_time = time.time()

        return self

    def auto_set_model_parameter(self):

        print('TODO auto_set_model_parameter')

        self.can_not_generate_input_output_data = None

        self.generate_data_set()

        self.input_width = self.data_set.input_ts_width
        self.col_size = self.data_set.col_size
        self.output_classes = self.data_set.output_classes

        # info_dim_size_list = []

        print('DONE auto_set_model_parameter')
        return True

    def generate_data_set(self):
        self.data_set = TSDataSet(debug_mode=self.debug_mode, prediction_mode=self.prediction_mode, hparams=self.hparams)
        self.data_set.generate_input_output_data()

    def get_output_classes_from_model(self, init_model_path):
        from smalltrain.model.operation import is_s3_path, download_to_local, upload_to_cloud

        print('[get_output_classes_from_model]Restore from init_model_path:{}'.format(init_model_path))

        local_init_model_path = init_model_path
        if self.prioritize_cloud:
            # download from S3 if the "init_model_path" is S3 path
            if is_s3_path(init_model_path):
                _paths, _global_iter_got_from_path = get_tf_model_file_paths(init_model_path)
                for _path in _paths:
                    local_init_model_path = download_to_local(path=_path, work_dir_path='/var/tmp/tsp/')
                local_init_model_path = local_init_model_path.split('.ckpt')[0] + '.ckpt'
                if _global_iter_got_from_path is not None:
                    local_init_model_path = local_init_model_path + '-' + str(_global_iter_got_from_path)
            else:
                print('[get_output_classes_from_model]Check local:{}'.format(init_model_path))

        print('[get_output_classes_from_model]Check local_init_model_path:{}'.format(local_init_model_path))

        if local_init_model_path is None or len(local_init_model_path) < 1 or os.path.isfile(local_init_model_path):
            print('[get_output_classes_from_model]local_init_model_path is empty. output_classes set None')
            self.output_classes = None
            return None
        meta_file_path = '{}.meta'.format(local_init_model_path)
        _saver = tf.train.import_meta_graph(meta_file_path)

        _saver.restore(self.sess, local_init_model_path)

        # get output_classes from last layer b_fc2 shape
        _variables = tf.get_default_graph().get_collection_ref(tf.GraphKeys.VARIABLES)
        print(_variables)
        bias_before_output_layer_name = 'model/fc/b_fc2/b_fc2:0'
        b_fc2 = tf.get_default_graph().get_tensor_by_name(bias_before_output_layer_name)

        # Reset the graph to restore after model construction
        tf.reset_default_graph()

        self.output_classes = int(b_fc2.shape[0]) # have to cast from string to integer
        return self.output_classes

    def train(self, iter_to=10000, learning_rate=1e-4, batch_size=128, dropout_ratio=0.5, l1_norm_reg_ratio=0.0, save_file_path=None, report_dir_path=None):
        from smalltrain.model.operation import is_s3_path, download_to_local, upload_to_cloud

        last_time = time.time()
        print('train with iter_to:{}, batch_size:{}, dropout_ratio:{}'.format(iter_to, batch_size, dropout_ratio))

        # TODO
        train_index = 0
        # input_data = self.data_set.input_data
        # output_data = self.data_set.output_data
        # train_index_list = self.data_set.train_index_list
        # test_index_list = self.data_set.test_index_list


        # test_size = 31 + 30  # 2015/9, 10
        # test_size = int(len(output_data) * 0.1)

        # setup each test data
        # _input_data = self.data_set.input_data

        test_data = self.data_set.get_test_input_data()
        if (self.mask_rate is not None) and self.mask_rate > 0:
            # masked_test_data = self.data_set.masked_input_data[test_index_list].astype(np.float32)
            masked_test_data = self.data_set.get_masked_test_input_data()

        # test_values = np.asarray(output_data[test_index_list], dtype=np.float32)
        test_values = self.data_set.get_test_output_data()
        if self.model_type == 'CLASSIFICATION':
            test_values_laveled = np.argmax(test_values, axis=1)
        elif self.model_type == 'REGRESSION':
            test_values = test_values.reshape(-1) # TODO

        # print('test_index_list:{}'.format(test_index_list))
        print('test_data.shape:{}'.format(test_data.shape))
        print('test_values.shape:{}'.format(test_values.shape))

        print('self.prediction_mode:{}'.format(self.prediction_mode))
        print('---------- time:{}'.format(time.time() - last_time))
        last_time = time.time()
        assert (test_data.shape[0] > 0)

        test_data_id_set = None
        if self.data_set.data_id_set is not None:
            test_data_id_set = self.data_set.get_test_data_id_set()
            print('test_data_id_set.shape:{}'.format(test_data_id_set.shape))

        test_annotation_data = None
        if self.data_set.annotation_data is not None:
            # test_annotation_data = self.data_set.annotation_data[test_index_list]
            test_annotation_data = self.data_set.get_test_annotation_data()
            print('test_annotation_data.shape:{}'.format(test_annotation_data.shape))



        # setup each train data set
        train_data_set = self.data_set
        # remove test data from train data
        # train_data_set.input_data = input_data[:-test_size].astype(np.float32)
        # train_data_set.output_data = output_data[:-test_size].astype(np.float32)

        # print('train_data_set.input_data.shape:{}'.format(train_data_set.input_data.shape))
        print('train_data_set.input_data.shape:{}'.format(self.data_set.get_train_input_data_shape()))

        # plot input and output data
        if self.model_type == 'CLASSIFICATION':
            _output_data = test_values_laveled
        else:
            _output_data = test_values

        print('test_input_data:{}'.format(test_data[:15, -1, 0]))
        print('test_output_data:{}'.format(test_values[:3]))
        print('---------- time:{}'.format(time.time() - last_time))
        last_time = time.time()
        plot_data(input_data=test_data, output_data=test_values_laveled if self.model_type == 'CLASSIFICATION' else test_values,
                           y_max=None, series_range=None,
                           report_dir_path=report_dir_path)

        print('---------- time:{} DONE plot_data'.format(time.time() - last_time))
        last_time = time.time()
        if self.debug_mode:
            if (not self.prediction_mode) and (not self.test_only_mode):
                index_to_export = 0
                self.data_set.export_data(data_kind='train_data', index=index_to_export, report_dir_path=report_dir_path)
                index_to_export = -1
                self.data_set.export_data(data_kind='train_data', index=index_to_export, report_dir_path=report_dir_path)

            index_to_export = 0
            self.data_set.export_data(data_kind='test_data', index=index_to_export, report_dir_path=report_dir_path)
            index_to_export = -1
            self.data_set.export_data(data_kind='test_data', index=index_to_export, report_dir_path=report_dir_path)

        # save all_variables names
        all_variables = [var.name for var in tf.get_default_graph().get_collection_ref('variables')]
        _report_path = os.path.join(report_dir_path, 'all_variables_names.csv')
        f = open(_report_path, 'w')
        for name in all_variables: f.write('{}\n'.format(name))
        f.close()

        print('---------- time:{} DONE save all_variables names'.format(time.time() - last_time))
        last_time = time.time()

        # save trainable_variables names
        trainable_variables_names = [var.name for var in self.get_trainable_variables()]
        _report_path = os.path.join(report_dir_path, 'trainable_variables_names.csv')
        f = open(_report_path, 'w')
        for name in trainable_variables_names: f.write('{}\n'.format(name))
        f.close()
        if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)

        print('---------- time:{} DONE upload_to_cloud'.format(time.time() - last_time))
        last_time = time.time()

        # if self.prediction_mode:
        #     # TODO
        #     return

        errors_history = None

        for i in range(iter_to):
            if (not self.test_only_mode) and (not self.prediction_mode):
                input_batch, output_batch = train_data_set.next_batch(batch_size)

                #  print('i:{}'.format(i))

                if self.global_iter == 0:
                    print('====================')
                    print('step %d, start training' % (self.global_iter))

                    print('input_batch.dtype:{}'.format(input_batch.dtype))
                    print('output_batch.dtype:{}'.format(output_batch.dtype))
                    print('input_batch.shape:{}'.format(input_batch.shape))
                    print('output_batch.shape:{}'.format(output_batch.shape))

                # train
                self.train_step.run(
                    feed_dict={self.x: input_batch, self.y_: output_batch, self.keep_prob: (1 - dropout_ratio),
                               self.learning_rate: learning_rate,
                               self.l1_norm_reg_ratio: l1_norm_reg_ratio,
                               self.is_train: True})
                summary, train_total_loss = self.sess.run([self.merged, self.total_loss]
                                                          , feed_dict={self.x: input_batch, self.y_: output_batch,
                                                                       self.keep_prob: (1 - dropout_ratio),
                                                                       self.learning_rate: learning_rate,
                                                                       self.l1_norm_reg_ratio: l1_norm_reg_ratio,
                                                                       self.is_train: True
                                                                       })

                if self.global_iter % 100 == 99:
                    # train_accuracy = accuracy.test(feed_dict={
                    # train_total_loss = self.total_loss.test(feed_dict={
                    #     self.x: input_batch, self.y_: output_batch, self.keep_prob: 1.0, self.learning_rate: learning_rate})
                    print('========================================')
                    print('step %d, training loss %g' % (self.global_iter, train_total_loss))
                    print('========================================')
                    self.train_writer.add_summary(summary, self.global_iter)

                    # print('min and max of normed train date_block_num:{}, {}'.format(min(input_batch[:,0,0]), max(input_batch[:,0,0])))

            # _test_and_report = (self.test_only_mode or self.global_iter == 9 or self.global_iter % 100 == 99)
            _test_and_report = (self.test_only_mode or self.prediction_mode or self.global_iter == 9 or self.global_iter % 100 == 99)
            # _test_and_report = (self.test_only_mode or self.global_iter % 10 == 9)

            if _test_and_report:
                # calc error
                if self.model_type == 'REGRESSION':
                    y_estimated = self.y.eval(feed_dict={
                        self.x: test_data, self.y_: test_values, self.keep_prob: 1.0, self.learning_rate: learning_rate,
                        self.l1_norm_reg_ratio: l1_norm_reg_ratio,
                        self.is_train: False
                    })
                    y_label_estimated = None

                    if self.mask_rate is not None and self.mask_rate > 0:
                        y_estimated_masked = self.y.eval(feed_dict={
                            self.x: masked_test_data, self.y_: test_values, self.keep_prob: 1.0,
                            self.learning_rate: learning_rate,
                            self.l1_norm_reg_ratio: l1_norm_reg_ratio,
                            self.is_train: False
                        })
                        y_label_estimated_masked = None


                else:
                    y_label_estimated, y_estimated = self.sess.run([self.y_label, self.y_label]
                                                                   , feed_dict={self.x: test_data, self.y_: test_values,
                                                                                self.keep_prob: 1.0,
                                                                                self.learning_rate: learning_rate,
                                                                                self.l1_norm_reg_ratio: l1_norm_reg_ratio,
                                                                                self.is_train: False})

                summary, test_total_loss = self.sess.run([self.merged, self.total_loss]
                                                         , feed_dict={self.x: test_data, self.y_: test_values,
                                                                      self.keep_prob: 1.0,
                                                                      self.learning_rate: learning_rate,
                                                                      self.l1_norm_reg_ratio: l1_norm_reg_ratio,
                                                                      self.is_train: False})
                root_mean_squared_error = None
                mean_absolute_error = None
                if self.model_type == 'REGRESSION':
                    root_mean_squared_error, mean_absolute_error = self.sess.run([self.root_mean_squared_error, self.mean_absolute_error]
                                                    , feed_dict={self.x: test_data, self.y_: test_values,
                                                                 self.keep_prob: 1.0,
                                                                 self.learning_rate: learning_rate,
                                                                 self.l1_norm_reg_ratio: l1_norm_reg_ratio,
                                                                 self.is_train: False})
                    if self.mask_rate is not None and self.mask_rate > 0:
                        root_mean_squared_error_masked, mean_absolute_error_masked = self.sess.run([self.root_mean_squared_error, self.mean_absolute_error]
                                                                , feed_dict={self.x: masked_test_data, self.y_: test_values,
                                                                             self.keep_prob: 1.0,
                                                                             self.learning_rate: learning_rate,
                                                                             self.l1_norm_reg_ratio: l1_norm_reg_ratio,
                                                                             self.is_train: False})

                    print('========================================')
                    print('step:{}, testing root_mean_squared_error:{}, mean_absolute_error:{}'.format(self.global_iter, root_mean_squared_error, mean_absolute_error))
                    print('========================================')
                    assert (root_mean_squared_error is not None)

                    new_errors = pd.DataFrame([[self.global_iter, root_mean_squared_error, mean_absolute_error]], columns=(['global_iter', 'root_mean_squared_error', 'mean_absolute_error']))
                    errors_history = pd.concat([errors_history, new_errors]) if errors_history is not None else new_errors

                    min_rmse_index = errors_history['root_mean_squared_error'].idxmin()
                    min_root_mean_squared_error = errors_history.iloc[min_rmse_index]['root_mean_squared_error']
                    min_global_iter = errors_history.iloc[min_rmse_index]['global_iter']
                    at_min_mean_absolute_error = errors_history.iloc[min_rmse_index]['mean_absolute_error']

                    print('min_global_iter:{}, min of root_mean_squared_error:{}, wirh mean_absolute_error:{}'.format(min_global_iter, min_root_mean_squared_error, at_min_mean_absolute_error))
                    if report_dir_path:
                        _report_path = os.path.join(report_dir_path, 'errors_history.csv')
                        errors_history.to_csv(_report_path, index=False)
                        if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)

                # log_scalar(writer=self.test_writer, tag='rmse', value=rmse, step=self.global_iter)

                if report_dir_path:
                    error_to_plot = None
                    error_name = None
                    if self.plot_errors is not None:

                        # TODO plor more than single error
                        for plot_error in self.plot_errors:

                            calc_range = [0, 9.0] if len(plot_error.split('DROP')) > 1 else None
                            if plot_error == 'accuracy':
                                error_to_plot = calc_accuracy_with_drop(test_values, y_estimated, rank_boundary_list=self.rank_boundary_list)
                                naive_error = None
                            else:
                                error_to_plot = calc_error_with_drop(plot_error, test_values, y_estimated, calc_range=calc_range)
                                naive_error = calc_error_with_drop(plot_error, test_values[:-1], y_estimated[1:], calc_range=calc_range)
                            error_name = 'error({})'.format(plot_error)
                            # report naive error TODO standardize
                            print('{}, error:{}, naive_error:{}'.format(error_name, error_to_plot, naive_error))

                    _offset_column_index = train_data_set.offset_column_index
                    # print('_offset_column_index:{}'.format(_offset_column_index))
                    if _offset_column_index > 0:
                        offset_values = test_data[:, 0, _offset_column_index]
                        offset_values = np.reshape(offset_values, (-1))

                        offset_value_unique_list = np.unique(offset_values)
                    else:
                        # offset_values = train_data_set.input_output_ts_offset
                        offset_value_unique_list = [train_data_set.input_output_ts_offset]

                    for _offset in offset_value_unique_list:
                        # print('_offset:{}'.format(_offset))
                        # print('offset_values:{}'.format(offset_values))
                        # print('len of offset_values:{}'.format(len(offset_values)))

                        if _offset_column_index > 0:
                            all_index_to_plot = [i for i, x in enumerate(offset_values) if math.fabs(x - _offset) < 1e-3]
                        else:
                            all_index_to_plot = list(range(len(test_data)))

                        # calc cc errors
                        input_target_value_column_index = 0 # TODO Enable to set with hyper param
                        cc_error = None
                        if self.calc_cc_errors and self.op_errors is not None:
                            true_y_to_plot_cc = _output_data[all_index_to_plot]
                            estimated_y_to_plot_cc = test_data[all_index_to_plot, -1, input_target_value_column_index]
                            for op_error in self.op_errors:
                                calc_range = [0, 9.0] if len(op_error.split('DROP')) > 1 else None

                                if op_error != 'accuracy':
                                    cc_error = calc_error_with_drop(op_error, true_y_to_plot_cc, estimated_y_to_plot_cc,
                                                                    calc_range=calc_range)
                                    cc_error_name = 'cc error({})'.format(op_error)

                            print('_offset:{}, error_name:{}, error_to_plot:{}, cc_error_name:{}, cc_error:{}'.format(_offset, error_name, error_to_plot, cc_error_name, cc_error))

                            x_to_plot_cc = list(range(len(estimated_y_to_plot_cc)))
                            _group_value = None
                            _plot_iter = None
                            title = 'Plot Ground truth and CC\nwith input-output offset:{} for group:{}'.format(
                                _offset, _group_value) if self.plot_title is None else self.plot_title.format(_offset,
                                                                                                              _group_value)

                            _report_path = plot_estmated_true(x=x_to_plot_cc, estimated_y=estimated_y_to_plot_cc, estimated_label=None, model_type=self.model_type,
                                               true_y=true_y_to_plot_cc, y_max=None, series_range=None, error=cc_error, error_name=cc_error_name, report_dir_path=report_dir_path,
                                               xlabel=self.plot_x_label, ylabel=self.plot_y_label, title=title, postfix='o{}_{}_cc'.format(_offset, _group_value), iter=_plot_iter,
                                               x_range=self.plot_x_range, y_range=self.plot_y_range)
                            if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)

                            self.calc_cc_errors = False # TODO

                        # plot in group
                        index_to_plot_group_dict = {'all':all_index_to_plot}
                        if self.plot_group_data_name_in_annotation is not None:
                            index_to_plot_group_dict = {}
                            _group_values = test_annotation_data[:, 2 + self.annotation_col_names.index(self.plot_group_data_name_in_annotation)]
                            _group_unique_values = list(set(_group_values))
                            for group_value in _group_unique_values:
                                # print('group_value:{}'.format(group_value))
                                index_to_plot = [i for i, x in enumerate(_group_values) if (x == group_value and i in all_index_to_plot)]
                                # print('index_to_plot:{}'.format(index_to_plot))
                                # print('test_annotation_data:{}'.format(test_annotation_data[index_to_plot]))
                                index_to_plot_group_dict[group_value] = index_to_plot

                        report_plot_file_list = []
                        for group_value, index_to_plot in index_to_plot_group_dict.items():
                            # print('_offset:{}, index_to_plot[:5]:{}'.format(_offset, index_to_plot[:5]))
                            estimated_y_to_plot = y_estimated[index_to_plot]
                            estimated_label_to_plot = y_label_estimated[index_to_plot] if y_label_estimated is not None else None
                            if self.mask_rate is not None and self.mask_rate > 0:
                                estimated_y_to_plot_masked = y_estimated_masked[index_to_plot]
                                estimated_label_to_plot_masked = y_label_estimated_masked[index_to_plot] if y_label_estimated_masked is not None else None

                            true_y_to_plot = _output_data[index_to_plot]

                            data_id_set_to_plot = None
                            if test_data_id_set is not None:
                                data_id_set_to_plot = test_data_id_set[index_to_plot]
                            elif test_annotation_data is not None:
                                data_id_set_to_plot = test_annotation_data[index_to_plot, 0]

                            test_annotation_data_dt_to_export = None
                            if test_annotation_data is not None:
                                test_annotation_data_dt_to_export = test_annotation_data[index_to_plot]

                            # print('len(estimated_y_to_plot):{}'.format(len(estimated_y_to_plot)))
                            x_to_plot = list(range(len(estimated_y_to_plot)))
                            if test_annotation_data is not None and self.plot_x_data_name_in_annotation is not None :
                                # print('self.plot_x_data_name_in_annotation:{}'.format(self.plot_x_data_name_in_annotation))
                                # print('self.annotation_col_names.index(self.plot_x_data_name_in_annotation):{}'.format(self.annotation_col_names.index(self.plot_x_data_name_in_annotation)))
                                # TODO x軸の表示を制御
                                x_to_plot = 1 + _offset - test_annotation_data_dt_to_export[:, 2 + self.annotation_col_names.index(self.plot_x_data_name_in_annotation)]


                            # print('len(x_to_plot):{}'.format(len(x_to_plot)))
                            # print('x_to_plot:{}'.format(x_to_plot))

                            # if self.test_only_mode:
                            if False:
                                for k, v in test_index_dict.items():
                                    postfix = k[:-11]
                                    series_range = (v[0], v[1])

                                    title = 'Plot Estimated\nwith input-output offset:{}'.format(_offset) if self.plot_title is None else self.plot_title.format(_offset)
                                    _report_path = plot_estmated_true(estimated_y=estimated_y_to_plot, estimated_label=estimated_label_to_plot, model_type=self.model_type,
                                                       true_y=None, y_max=None, series_range=series_range, error=error_to_plot, error_name=error_name, report_dir_path=report_dir_path,
                                                       xlabel=self.plot_x_label, ylabel=self.plot_y_label, title=title, postfix='{}_o{}_{}'.format(postfix, _offset, group_value))
                                    if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)

                            else:
                                title = 'Plot Ground truth and Estimated\nwith input-output offset:{} for group:{}'.format(_offset, group_value) if self.plot_title is None else self.plot_title.format(_offset, group_value)
                                plot_iter = None if self.test_only_mode or self.prediction_mode else self.global_iter
                                true_y_to_plot = None if self.prediction_mode else true_y_to_plot
                                error_to_plot = None if self.prediction_mode else error_to_plot
                                error_name = None if self.prediction_mode else error_name
                                report_plot_file_path = plot_estmated_true(x=x_to_plot, estimated_y=estimated_y_to_plot, estimated_label=estimated_label_to_plot, model_type=self.model_type,
                                                   true_y=true_y_to_plot, y_max=None, series_range=None, error=error_to_plot, error_name=error_name, report_dir_path=report_dir_path,
                                                   xlabel=self.plot_x_label, ylabel=self.plot_y_label, title=title, postfix='o{}_{}'.format(_offset, group_value), iter=plot_iter,
                                                   x_range=self.plot_x_range, y_range=self.plot_y_range)
                                if report_plot_file_list:
                                    if self.cloud_root: upload_to_cloud(report_plot_file_path, self.cloud_root, self.save_root_dir)
                                    report_plot_file_list.append(report_plot_file_path)
                                if self.mask_rate is not None and self.mask_rate > 0:
                                    _report_path = plot_estmated_true(x=x_to_plot, estimated_y=estimated_y_to_plot_masked,
                                                       estimated_label=estimated_label_to_plot_masked, model_type=self.model_type,
                                                       true_y=true_y_to_plot, y_max=None, series_range=None,
                                                       error=error_to_plot, error_name=error_name, report_dir_path=report_dir_path,
                                                       xlabel=self.plot_x_label, ylabel=self.plot_y_label, title=title,
                                                       postfix='o{}_{}_masked'.format(_offset, group_value), iter=plot_iter,
                                                       x_range = self.plot_x_range, y_range=self.plot_y_range)
                                    if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)

                                # detail plot
                                self.detail_plot = False # TODO
                                self.detail_plot_size = 24 * 10
                                self.detail_plot_size = 318
                                if self.detail_plot:
                                    x_to_plot_detail = x_to_plot[:self.detail_plot_size] if x_to_plot is not None else None
                                    true_y_to_plot_detail = true_y_to_plot[:self.detail_plot_size] if true_y_to_plot is not None else None
                                    _report_path = plot_estmated_true(x=x_to_plot_detail, estimated_y=estimated_y_to_plot[:self.detail_plot_size], estimated_label=estimated_label_to_plot[:self.detail_plot_size] if estimated_label_to_plot is not None else None, model_type=self.model_type,
                                                       true_y=true_y_to_plot_detail, y_max=None, series_range=None, error=error_to_plot, error_name=error_name, report_dir_path=report_dir_path,
                                                       xlabel=self.plot_x_label, ylabel=self.plot_y_label, title=title, postfix='l{}_o{}_{}'.format(self.detail_plot_size, _offset, group_value), iter=plot_iter,
                                                       x_range=self.plot_x_range, y_range=self.plot_y_range)
                                    if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)
                                    if self.mask_rate is not None and self.mask_rate > 0:
                                        _report_path = plot_estmated_true(x=x_to_plot_detail, estimated_y=estimated_y_to_plot_masked[:self.detail_plot_size],
                                                           estimated_label=estimated_label_to_plot_masked[
                                                                           :self.detail_plot_size] if estimated_label_to_plot_masked is not None else None,
                                                           model_type=self.model_type,
                                                           true_y=true_y_to_plot_detail, y_max=None,
                                                           series_range=None, error=error_to_plot, error_name=error_name,
                                                           report_dir_path=report_dir_path,
                                                           xlabel=self.plot_x_label, ylabel=self.plot_y_label, title=title,
                                                           postfix='l{}_o{}_{}_masked'.format(self.detail_plot_size, _offset, group_value),
                                                           iter=plot_iter,
                                                           x_range=self.plot_x_range, y_range=self.plot_y_range)
                                        if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)

                                self.export_prediction = True
                                if self.export_prediction:
                                    _size = len(estimated_y_to_plot)
                                    df_prediction_cols = ['DateTime', 'Estimated', 'MaskedEstimated', 'True']
                                    if self.annotation_col_names is not None:
                                        df_prediction_cols.extend(self.annotation_col_names)
                                    df_prediction = pd.DataFrame(np.zeros([_size, 4 + self.annotation_col_size]), columns=df_prediction_cols)
                                    df_prediction['DateTime'] = data_id_set_to_plot
                                    df_prediction['Estimated'] = estimated_y_to_plot
                                    if self.mask_rate is not None and self.mask_rate > 0: df_prediction['MaskedEstimated'] = estimated_y_to_plot_masked
                                    df_prediction['True'] = true_y_to_plot
                                    if test_annotation_data is not None:
                                        for i, col in enumerate(self.annotation_col_names):
                                            df_prediction[col] = test_annotation_data_dt_to_export[:, 2 + i]

                                    if plot_iter is None:
                                        output_file_name = 'prediction_o{}_{}.csv'.format(_offset, group_value)
                                    else:
                                        output_file_name = 'prediction_e{}_o{}_{}.csv'.format(self.global_iter, _offset, group_value)
                                    output_file_path = os.path.join(report_dir_path, output_file_name)
                                    df_prediction.to_csv(output_file_path, index=False)
                                    if self.cloud_root: upload_to_cloud(output_file_path, self.cloud_root, self.save_root_dir)

                        if self.plot_animation:
                            if len(report_plot_file_list) > 0:
                                report_plot_file_list.sort()
                                gif_report_file_path = report_plot_file_list[0] + '.gif'
                                gif_util.generate_gif_animation(src_file_path_list=report_plot_file_list, dst_file_path=gif_report_file_path)
                                # TODO if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)
                            else:
                                print('No report_plot_file_list to plot_animation')

                # if self.global_iter % 1000 == 999:

                print('test cross entropy %g' % test_total_loss)
                self.test_writer.add_summary(summary, self.global_iter)

                if save_file_path and not (self.test_only_mode or self.prediction_mode):
                    print('save model to save_file_path:{}'.format(save_file_path))
                    self.saver.save(self.sess, save_file_path, global_step=self.global_iter)
                    if self.cloud_root:
                        _paths, _global_iter_got_from_path = get_tf_model_file_paths(save_file_path, self.global_iter)
                        for _path in _paths:
                            upload_to_cloud(_path, self.cloud_root, self.save_root_dir)

            if self.test_only_mode or self.prediction_mode:
                print('DONE test_only_mode or self.prediction_mode')
                return


            self.global_iter += 1


    def define_model(self):

        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
        with tf.name_scope('model/'):
            self.keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
            self.l1_norm_reg_ratio = tf.placeholder(tf.float32)
            tf.summary.scalar('l1_norm_reg_ratio', self.l1_norm_reg_ratio)

        output_middle_layer = self.define_1d_cnn_model(n_layer=self.n_layer, has_res_net=self.has_res_net)

        with tf.name_scope('model/'):
            self.y = output_middle_layer
            print('y.shape:', self.y.shape)

            print('self.model_type :', self.model_type)

            if self.model_type == 'REGRESSION':
                self.y = tf.reshape(self.y, [-1])
                print('y reshaped to :', self.y.shape)
            else:
                self.y_label = tf.cast(tf.argmax(self.y, 1), dtype=tf.int32)
                self.y_softmax = tf.nn.softmax(self.y)

            self.learning_rate = tf.placeholder(tf.float32, shape=[])

        with tf.name_scope('precisions/'):
            with tf.name_scope('l1_norm_reg_loss'):
                self.l1_norm_reg_loss = 0.0
                if self.add_l1_norm_reg:
                    self.l1_norm_reg_loss = self.l1_norm_reg_ratio * self.l1_norm
                tf.summary.scalar('l1_norm_reg_loss', self.l1_norm_reg_loss)
            with tf.name_scope('preactivation_regularization_loss'):
                self.preactivation_regularization_loss = 0.0
                if self.add_preactivation_regularization:
                    self.preactivation_regularization_loss = self.preactivation_regularization_value_ratio * self.preactivation_regularization_value
                tf.summary.scalar('preactivation_regularization_loss', self.preactivation_regularization_loss)
            if self.model_type == 'REGRESSION':
                with tf.name_scope('mean_squared_error'):
                    self.mean_squared_error = tf.reduce_mean(tf.square(self.y_ - self.y))
                    tf.summary.scalar('mean_squared_error', self.mean_squared_error)
                with tf.name_scope('mean_absolute_error'):
                    self.mean_absolute_error = tf.reduce_mean(tf.abs(self.y_ - self.y))
                    tf.summary.scalar('mean_absolute_error', self.mean_absolute_error)
                with tf.name_scope('root_mean_squared_error'):
                    self.root_mean_squared_error = tf.sqrt(self.mean_squared_error)
                    tf.summary.scalar('root_mean_squared_error', self.root_mean_squared_error)
                with tf.name_scope('total_loss'):
                    self.total_loss = self.root_mean_squared_error
                    self.total_loss = self.total_loss + self.l1_norm_reg_loss + self.preactivation_regularization_loss
                    tf.summary.scalar('total_loss', self.total_loss)
            else:

                with tf.name_scope('cross_entropy'):
                    self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

                    tf.summary.scalar('cross_entropy', self.cross_entropy)

                with tf.name_scope('total_loss'):
                    self.total_loss = self.cross_entropy
                    self.total_loss = self.total_loss + self.l1_norm_reg_loss + self.preactivation_regularization_loss
                    tf.summary.scalar('total_loss', self.total_loss)

            self.set_optimizer()

            if self.model_type == 'CLASSIFICATION':
                # print('DEBUG self.y.shape:{}, self.y_.shape:{}'.format(self.y.shape, self.y_.shape))
                self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

        # Merge all the summaries and write them out to
        # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(self.log_dir_path + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_dir_path + '/test')

    def cnn_layer(self, x, layer_num, layer_name, conv_in_channels, conv_out_channels, filter_width=3, pool=2, actf=tf.nn.relu, has_dropout=True, has_batch_norm=True, block_name=None):

        _name = 'model/'
        if block_name:
            _name += block_name + '/'
        _name += layer_name + '/'
        with tf.name_scope(_name ):
            with tf.name_scope('W_conv'):
                W_conv = self.weight_variable([1, filter_width, conv_in_channels, conv_out_channels], name="W_conv")
                self.op_add_l1_norm(W_conv)
                self.variable_summaries(W_conv)

            if not has_batch_norm:
                with tf.name_scope('b_conv'):
                    b_conv = self.bias_variable([conv_out_channels], name="b_conv")
                    self.variable_summaries(b_conv)

            print('########## {} ########## input x shape:{} ########## W_conv:{} ########## pool:{} ########## has_batch_norm:{} ########## actf:{} ##########'.format(
                _name, x.shape, W_conv.shape, pool, has_batch_norm, actf))

            h_conv = conv1d(x, W_conv)
            if has_batch_norm:
                h_conv = self.batch_norm(h_conv)
            else:
                h_conv = tf.add(h_conv, b_conv)

            h_pool = max_pool_1x2(h_conv, pool)

            if has_dropout:
                conv_out = tf.nn.dropout(h_pool, self.keep_prob)
            else:
                conv_out = h_pool

            if actf is not None:
                if self.add_preactivation_regularization:
                    self.op_add_preactivation_regularization(preactivation=conv_out, preactivation_maxout=self.preactivation_maxout_list[layer_num])
                with tf.name_scope('actf'):
                    conv_out = actf(conv_out, name='actf')

        return conv_out

    def define_1d_cnn_model(self, n_layer=5, has_res_net=False, has_non_cnn_net=False):

        input_width = self.input_width
        col_size = self.col_size

        if has_non_cnn_net:
            ts_col_index = [0, 3, 4, 6, 7]
            non_ts_col_index = [1, 2, 5]
            # date_block_num,item_id,shop_id,item_cnt_month_normed,item_price,item_category_id,year,month
            ts_col_size = len(ts_col_index)
        else:
            ts_col_index = range(col_size)
            ts_col_size = col_size


        with tf.name_scope('model/'):
            # cnn_input_channels = ts_col_size
            # cnn_input_channels = col_size

            self.x = tf.placeholder(tf.float32, shape=[None, input_width, col_size])
            if self.model_type == 'REGRESSION':
                self.y_ = tf.placeholder(tf.float32, shape=[None])
            else:
                self.y_ = tf.placeholder(tf.float32, shape=[None, self.output_classes])

            # x_3d = tf.reshape(self.x, [-1, input_width, col_size, 1])
            # print('x_3d:', x_3d.shape)  # (?, 37, 10, 10, 1)


            self.cnn_layer_names = ['cnn_layer_{}'.format(i) for i in range(n_layer)]
            first_conv_in = self.col_size
            if self.cnn_channel_size_list is None:
                later_conv_in = self.cnn_channel_size
                conv_in = np.ones([n_layer + 1], dtype="int32") * later_conv_in
                conv_in[0] = first_conv_in
                # conv_in[1] = first_conv_in # ResNet ver 1.0 (~2019/01/08 16:01)
                conv_out_size = conv_in[-1]  # ResNet ver 2.0 (2019/01/08 16:58~)
            else:
                print('n_layer:{}'.format(n_layer))
                conv_in = np.hstack((first_conv_in, np.asarray(self.cnn_channel_size_list, dtype='int32')))
                print('conv_in:{}'.format(conv_in))
                conv_out_size = conv_in[-1]

            conved_width = self.input_width
            for pool in self.pool_size_list:
                conved_width /= pool
            conved_width = int(conved_width)

            # Input Layer
            # x = tf.transpose(self.x, perm=[0, 2, 1])
            x = self.x
            # print('x shape before delete_non_ts_col_tensor:', x.shape)
            if has_non_cnn_net:
                delete_non_ts_col_tensor = tf.cast([[[1, 0, 0, 1, 1, 0, 1, 1]]], tf.float32)
                delete_ts_col_tensor = tf.cast([[[0, 1, 1, 0, 0, 1, 0, 0]]], tf.float32)
                x = tf.multiply(x, delete_non_ts_col_tensor)
                non_ts_x = tf.multiply(x, delete_ts_col_tensor)
                # non_ts_x = tf.reshape(non_ts_x, [-1, col_size])
                non_ts_x = tf.reshape(non_ts_x, [-1, col_size * input_width])

                print('x shape after delete_non_ts_col_tensor:', x.shape)
                print('non_ts_x shape after delete ts_col_tensor:', non_ts_x.shape)

            x = tf.reshape(x, [-1, 1, input_width, first_conv_in])

            if self.has_res_net:
                n_cnn_layers = len(self.cnn_layer_names)
                print('n_cnn_layers:{}'.format(n_cnn_layers))
                N_CNN_LAYERS_IN_RES_BLOCK = 2 # TODO
                n_res_block = int(n_cnn_layers / N_CNN_LAYERS_IN_RES_BLOCK)
                print('n_res_block:{}'.format(n_res_block))

                l = 0

                # add cnn_layer without residual
                n_cnn_layers_without_res_net = int(n_cnn_layers - n_res_block * N_CNN_LAYERS_IN_RES_BLOCK)
                print('n_cnn_layers_without_res_net:{}'.format(n_cnn_layers_without_res_net))
                if n_cnn_layers_without_res_net >= 1:
                    print('add cnn_layer {} without residual'.format(n_cnn_layers_without_res_net))
                    while l < n_cnn_layers_without_res_net:
                        x = self.cnn_layer(x, layer_num=l, layer_name=self.cnn_layer_names[l], conv_in_channels=conv_in[l],
                                           conv_out_channels=conv_in[l + 1], filter_width=self.filter_width,
                                           pool=self.pool_size_list[l],
                                           actf=self.act_func_ref_list[l],
                                           has_batch_norm=self.has_batch_norm)

                        l += 1

                # add res_block
                while l < n_cnn_layers:
                    res_block_index = int(l / N_CNN_LAYERS_IN_RES_BLOCK)
                    block_name = 'res_block_{}'.format(res_block_index)
                    with tf.name_scope('model/' + block_name + '/'):
                        with tf.name_scope('x_id'):
                            x_id = tf.identity(x, name='x_id')
                            self.variable_summaries(x_id)

                        # path 1. residual
                        # layer_name = 'residual'
                        # with tf.name_scope('model/' + layer_name + '/'):
                        #     res_in = tf.identity(x, name='identity_' + x.name)
                        # path 1. cnn_layers
                        for i in range(N_CNN_LAYERS_IN_RES_BLOCK):
                            is_last_layer_in_res_block = (i == N_CNN_LAYERS_IN_RES_BLOCK - 1)
                            x = self.cnn_layer(x, layer_num=l + i, layer_name=self.cnn_layer_names[l + i], block_name=block_name, conv_in_channels=conv_in[l + i],
                                               conv_out_channels=conv_in[l + i + 1], filter_width=self.filter_width,
                                               pool=self.pool_size_list[l + i],
                                               actf=(None if is_last_layer_in_res_block else self.act_func_ref_list[l]),
                                               has_batch_norm=self.has_batch_norm)

                        # path 1. residual
                        layer_name = 'residual'
                        with tf.name_scope('model/' + block_name + '/' + layer_name + '/'):
                            # res_in = tf.identity(x_id, name='identity_' + x.name)

                            # output residual net(short_cut)
                            conv_in_channels =conv_in[l]
                            conv_out_channels = conv_in[l + N_CNN_LAYERS_IN_RES_BLOCK]
                            if pow(conv_in_channels - conv_out_channels, 2) < 1e-3:
                                # res_out = tf.identity(x_id, name='res_out')
                                with tf.name_scope('res_out'):
                                    # res_out = x_id
                                    res_out = tf.identity(x_id, name='res_out')

                            else:
                                pool_res = np.asarray(self.pool_size_list[l:l + N_CNN_LAYERS_IN_RES_BLOCK]).prod()
                                pooled_input_1 = tf.nn.avg_pool(x_id, ksize=[1, 1, pool_res, 1],
                                                                strides=[1, 1, pool_res, 1], padding='VALID')
                                print('pooled_input_1 shape:{}'.format(pooled_input_1.shape))
                                # Zero-padding
                                padded_input = tf.pad(pooled_input_1,
                                                      [[0, 0], [0, 0], [0, 0], [0, (conv_out_channels - conv_in_channels)]])
                                print('padded_input shape:{}'.format(padded_input.shape))
                                with tf.name_scope('res_out'):
                                    res_out = tf.identity(padded_input, name='res_out')
                        x_add_res = tf.add(x, res_out, name='x_add_res')

                        # add_preactivation_regularization
                        if self.add_preactivation_regularization:
                            self.op_add_preactivation_regularization(preactivation=x_add_res, preactivation_maxout=self.preactivation_maxout_list[l + N_CNN_LAYERS_IN_RES_BLOCK])

                        # activation
                        with tf.name_scope('actf_after_add'):
                            actf = self.act_func_ref_list[l]
                            actf_after_add = actf(x_add_res, name='actf_after_add')
                        print('########## layer_name:{} ########## input x shape:{} ########## conv_in_channels:{} ########## conv_out_channels:{} ########## res_out:{} ########## with activation'.format(
                                layer_name, x.shape, conv_in_channels, conv_out_channels, res_out.shape))

                        l += N_CNN_LAYERS_IN_RES_BLOCK
                        x = actf_after_add

            else:
                for l, cnn_layer_name in enumerate(self.cnn_layer_names):
                    x = self.cnn_layer(x, layer_num=l, layer_name=cnn_layer_name, conv_in_channels=conv_in[l],
                                       conv_out_channels=conv_in[l + 1], filter_width=self.filter_width, pool=self.pool_size_list[l],
                                       # actf=tf.nn.relu,
                                       actf=self.act_func_ref_list[l],
                                       has_batch_norm=self.has_batch_norm)

        layer_name = 'fc'
        with tf.name_scope('model/' + layer_name + '/'):
            # fc
            self.fc_output_dim = 500
            conv_out_flat_nodes = conved_width * conv_out_size
            with tf.name_scope('conv_out_flat'):
                conv_out_flat = tf.reshape(x, [-1, conv_out_flat_nodes], name='conv_out_flat')
                self.variable_summaries(conv_out_flat)

            ## Readout Layer
            if self.model_type == 'REGRESSION':
                y_out_dim = 1
            else:
                y_out_dim = self.output_classes


            with tf.name_scope('W_fc2'):
                W_fc2 = self.weight_variable([conv_out_flat_nodes, y_out_dim], name='W_fc2')
                self.op_add_l1_norm(W_fc2)
                self.variable_summaries(W_fc2)
            with tf.name_scope('b_fc2'):
                b_fc2 = self.bias_variable([y_out_dim], name='b_fc2')
                self.variable_summaries(b_fc2)

            if has_non_cnn_net:
                non_ts_x_1_in_dim = col_size*input_width
                non_ts_x_1_out_dim = 500
                non_ts_x = self.nn_layer(non_ts_x, input_dim=non_ts_x_1_in_dim, output_dim=non_ts_x_1_out_dim,
                                           layer_name='non_ts_x_1',
                                           actf=tf.nn.relu, last_layer=False,
                                           has_batch_norm=self.has_batch_norm)
                non_ts_x = self.nn_layer(non_ts_x, input_dim=non_ts_x_1_out_dim, output_dim=conv_out_flat_nodes,
                                           layer_name='non_ts_x_2',
                                           actf=tf.nn.relu, last_layer=False,
                                           has_batch_norm=self.has_batch_norm)
                conv_out_flat = tf.add(conv_out_flat, non_ts_x)

            output_middle_layer = tf.matmul(conv_out_flat, W_fc2) + b_fc2
            print('########## {} ########## input x shape:{} ########## W_fc2:{} ########## b_fc2:{} ##########'.format(
                layer_name, x.shape, W_fc2.shape, b_fc2.shape))

            return output_middle_layer


## Convolution and Pooling
# 1d cnn
def conv1d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # or return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def max_pool_1x2(x, pool=2):
    # pool only for time-series direction
    return tf.nn.max_pool(x, ksize=[1, 1, pool, 1],
                          strides=[1, 1, pool, 1], padding='SAME')

# 3d cnn
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

# Pooling: max pooling over 2x2 blocks
def max_pool_pxpxp(x, pool=2):
  return tf.nn.max_pool3d(x, ksize=[1, pool, pool, pool, 1], strides=[1, pool, pool, pool, 1], padding='SAME')


def log_scalar(writer, tag, value, step):
    """Log a scalar variable.
    Parameter
    ----------
    tag : basestring
        Name of the scalar
    value
    step : int
        training iteration
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                 simple_value=value)])
    writer.add_summary(summary, step)

import matplotlib.pyplot as plt
def plot_estmated_true(x, estimated_y, iter=None, estimated_label=None, model_type='CLASSIFICATION', true_y=None, y_max=None, series_range=None, error=None, error_name='error', target_data_set_name=None, report_dir_path='report/', xlabel='time series', ylabel='Label', title=None, postfix='', x_range=None, y_range=None, debug_mode=False):
    # Nothong to plot
    if len(x) < 2 or len(estimated_y) < 2:
        return

    if series_range:
        if true_y is not None:
            true_y = true_y[series_range[0]:series_range[1]]
        estimated_y = estimated_y[series_range[0]:series_range[1]]
        if estimated_label is not None:
            estimated_label = estimated_label[series_range[0]:series_range[1]]


    # 正しい検出(True positive)、未検出(False negative)、誤った検出(False positive)
    plt.clf()
    # print('true_y.max():{}'.format(true_y.max()))
    # print('true_y.min():{}'.format(true_y.min()))
    # print('estimated_y.max():{}'.format(estimated_y.max()))
    # print('estimated_y.min():{}'.format(estimated_y.min()))
    if y_max is None:
        y_max = estimated_y.max()
    y_min = estimated_y.min()
    if true_y is not None:
        y_max = max(y_max, true_y.max())
        y_min = min(y_min, true_y.min())

    if estimated_label is not None:
        # print('estimated_label.max():{}'.format(estimated_label.max()))
        # print('estimated_label.min():{}'.format(estimated_label.min()))
        y_max = max(y_max, estimated_label.max())
        y_min = y_max * -0.05
    else:
        y_min = y_min - math.fabs(y_max * 0.05)

    x = list(range(len(estimated_y))) if x is None else x

    if x_range is None: x_range = [min(x), max(x)]
    plt.xlim(x_range[0], x_range[1])

    if debug_mode:
        print('y_min:{}, y_max:{}, estimated_y:{}, true_y:{}'.format(y_min, y_max, estimated_y, true_y))

    # delete point that is out of y_range
    if y_range is not None:
        if true_y is not None:
            true_y = [y if y >= y_range[0] and y <= y_range[1] else np.nan for y in true_y]

    if y_range is None: y_range = [y_min, y_max * 1.35]
    plt.ylim(y_range[0], y_range[1])
    y_max = y_range[1]

    if true_y is not None:
        if model_type in ['CLASSIFICATION_ONOFF']:
            plt.fill_between(x, true_y, 0, where=0<true_y, color='#00a000', label="True positive")
        elif model_type in ['REGRESSION', 'CLASSIFICATION']:
            plt.plot(x, true_y, color='#00a000', label="True")

    if model_type == 'CLASSIFICATION_ONOFF':
        estimated_y_mean = [estimated_y[i][1] for i in x]
        if true_y is not None:
            plt.fill_between(x, true_y, estimated_y_mean, where=true_y<estimated_y_mean, color='#e0a000', label="False positive")
            plt.fill_between(x, true_y, estimated_y_mean, where=estimated_y_mean<true_y, color='#ff5000', label="False negative")
        else:
            plt.fill_between(x, estimated_y_mean, 0, color='#e06060', label="Positive")
    elif model_type == 'CLASSIFICATION':
        plt.plot(x, estimated_label, color='#e06060', label="Estimated")

    elif model_type == 'REGRESSION':
        plt.plot(x, estimated_y, color='#e06060', label="Estimated")

    plt.legend()
    if error:
        plt.text(x_range[0]+2, y_max * 0.8, '{}         :{:.4f}'.format(error_name, error))
    # if true_y is not None:
    #     _error = np.sqrt(np.mean((true_y - estimated_y)**2))
    #     plt.text(x_range[0]+2, y_max * 0.75, '{} in this plot:{:.4f}'.format(error_name, _error))
    if target_data_set_name:
        plt.text(x_range[0]+2, y_max * 0.8, 'Data set {}'.format(target_data_set_name))

    if title is None:
        title = 'Plot Estimated' if true_y is None else 'Plot Ground truth and Estimated'

    plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)

    if iter is None:
        _report_path = report_dir_path + 'test_plot##POSTFIX##.png'
    else:
        _report_path = report_dir_path + 'test_plot_e{}##POSTFIX##.png'.format(iter)
    _report_path = _report_path.replace('##POSTFIX##', '_{}'.format(postfix))
    plt.savefig(_report_path)

    plt.close()

    return _report_path

def scatter_plot_estmated_true(i, true_y, estimated_y, rmse, report_dir_path='report/', plot_dense_area=True, rmse_latest_min=None, rmse_latest_max=None):

    plt.clf()
    xy_min = min(min(true_y), min(estimated_y))
    xy_max = max(max(true_y), max(estimated_y))
    xy_width = xy_max - xy_min
    plt.xlim(0, xy_max)
    plt.ylim(0, xy_max)
    plt.scatter(true_y, estimated_y)
    plt.plot([0, xy_max], [0, xy_max], color="red", linewidth=2, linestyle="dashed")
    # plt.legend()
    true_y_mean = true_y.mean()

    plt.text(xy_width * 0.1, xy_max - xy_width * 0.1, 'RMSE :{:.4f}, AVE(GT) :{:.4f}'.format(rmse, true_y_mean))
    if rmse_latest_min:
        plt.text(xy_width * 0.1, xy_max - xy_width * 0.15, 'Min of latest RMSE :{:.4f}'.format(rmse_latest_min))
    if rmse_latest_max:
        plt.text(xy_width * 0.1, xy_max - xy_width * 0.2, 'Max of latest RMSE :{:.4f}'.format(rmse_latest_max))
    plt.title('Scatter plot Ground truth vs Estimated')
    plt.xlabel('Ground truth')
    plt.ylabel('Estimated')

    plt.savefig(report_dir_path + 'test_scatter_plot_e{}.png'.format(i))
    plt.close()

    if plot_dense_area:
        true_y_std = true_y.std()
        true_y_5p = true_y_mean - 2 * true_y_std
        true_y_95p = true_y_mean + 2 * true_y_std

        plt.clf()
        xy_min = true_y_5p
        xy_max = true_y_95p
        xy_width = xy_max - xy_min
        plt.xlim(0, xy_max)
        plt.ylim(0, xy_max)
        plt.scatter(true_y, estimated_y)
        plt.plot([0, xy_max], [0, xy_max], color="red", linewidth=2, linestyle="dashed")
        plt.text(xy_width * 0.1, xy_max - xy_width * 0.1,
                 'RMSE :{:.4f}, AVE(GT) :{:.4f}'.format(rmse, true_y_mean))
        if rmse_latest_min:
            plt.text(xy_width * 0.1, xy_max - xy_width * 0.15, 'Min of latest RMSE :{:.4f}'.format(rmse_latest_min))
        if rmse_latest_max:
            plt.text(xy_width * 0.1, xy_max - xy_width * 0.2, 'Max of latest RMSE :{:.4f}'.format(rmse_latest_max))
        plt.title('Scatter plot Ground truth vs Estimated')
        plt.xlabel('Ground truth')
        plt.ylabel('Estimated')

        plt.savefig(report_dir_path + 'test_scatter_plot_dense_e{}.png'.format(i))
        plt.close()


# def plot_estmated_true(i, estimated_y, estimated_label, true_y=None, y_max=None, series_range=None, rmse=None, target_data_set_name=None, report_dir_path='report/', xlabel='time series', ylabel='Label', title=None, postfix=''):
def plot_data(input_data, output_data,
                           y_max=None, series_range=None,
                           report_dir_path='report/',
                           xlabel='time series', ylabel_input_data='value', ylabel_output_data='Label (0 or 1)', title=None, postfix=''):

    ts_axis = 0 # 時系列の軸
    ts_history_axis = 1 # 時系列の推移の軸
    channel_axis = 2 # チャンネルの軸


    plt.clf()
    if y_max is None:
        y_max = max(input_data.max(), output_data.max())
    x = list(range(input_data.shape[ts_axis]))

    # plt.ylim(y_max * -0.05, y_max * 1.35)

    # draw on figure 1
    plt.figure(1)

    # Divide vertically into 2 and horizontally into 1, and draw on 1st division
    plt.subplot(211)
    plt.plot(x, output_data, label="output_data")
    plt.legend()
    # if target_data_set_name:
    #     plt.text(2, y_max * 1.2, 'Data set {}'.format(target_data_set_name))

    if title is None:
        title = 'Output data(Upper part) and Input data(Lower part)'

    plt.title(title)
    plt.ylabel(ylabel_output_data)

    # draw on 2nd division
    plt.subplot(212)
    ts_history = 0
    for channel_index in range(input_data.shape[channel_axis]):
        plt.plot(x, input_data[:, ts_history, channel_index], label="channel_index:{}".format(channel_index))

    plt.legend()
    # if target_data_set_name:
    #     plt.text(2, y_max * 1.2, 'Data set {}'.format(target_data_set_name))


    plt.xlabel(xlabel)
    plt.ylabel(ylabel_input_data)

    # show figure 1
    # plt.show()

    _report_path = report_dir_path + 'test_plot_##POSTFIX##.png'
    _report_path = _report_path.replace('##POSTFIX##', '_{}'.format(postfix))
    plt.savefig(_report_path)

    plt.close()

def rmse_by_day(target_df, sum_unit='day', datetime_col_name='DateTime'):
    datetime_list = target_df[datetime_col_name]
    if sum_unit == 'day':
        dt_group_series = [datetime(dt.year, dt.month, dt.day, 0, 0, 0) for dt in datetime_list]
    work_df = target_df.copy()
    work_df['DateTimeGroup'] = dt_group_series
    work_df = work_df.groupby(by='DateTimeGroup').mean()
    work_df['DateTime'] = work_df.index  # DateTime を補完
    rmse = np.sqrt(np.mean((work_df['True'] - work_df['Estimated'])**2))

    return rmse, work_df


def calc_error_with_drop(error_str, true_list, estimated_list, calc_range=None):
    if 'MAE' in error_str.split('_'):
        ret_error = calc_mean_absolute_error_with_drop(true_list, estimated_list, calc_range)
    elif 'RMSE' in error_str.split('_'):
        ret_error = calc_rmse_with_drop(true_list, estimated_list, calc_range)
    else:
        raise ValueError('error_str:{} contains no error definition'.format(error_str))

    return ret_error

def in_the_rank(rank_boundary, v, lower_equals=True):
    if lower_equals:
        return (rank_boundary[0] <= v and rank_boundary[1] > v)
    else:
        return (rank_boundary[0] < v and rank_boundary[1] >= v)

def calc_mean_absolute_error_with_drop(t, e, calc_range=None):
    if calc_range is None:
        index_to_calc = list(range(len(t)))
    else:
        index_to_calc = [i for i, x in enumerate(t) if
                         x >= calc_range[0] and x <= calc_range[1]]
    return np.asarray([math.fabs(t[i] - e[i]) for i in index_to_calc]).mean()

def calc_rmse_with_drop(t, e, calc_range=None):
    if calc_range is None:
        index_to_calc = list(range(len(t)))
    else:
        index_to_calc = [i for i, x in enumerate(t) if
                         x >= calc_range[0] and x <= calc_range[1]]
    return math.sqrt(np.asarray([math.pow(t[i] - e[i], 2) for i in index_to_calc]).mean())


def calc_accuracy_with_drop(t, e, calc_range=None, rank_boundary_list=None):
    if rank_boundary_list is None:
        return None
    if calc_range is None:
        index_to_calc = list(range(len(t)))
    else:
        index_to_calc = [i for i, x in enumerate(t) if
                         x >= calc_range[0] and x <= calc_range[1]]

    # remove non-rank index
    remove_non_rank_index = index_to_calc
    for rank_boundary in rank_boundary_list:
        remove_non_rank_index = np.intersect1d(remove_non_rank_index, ([i for i, x in enumerate(t) if
                         x < rank_boundary[0] or x > rank_boundary[1]]))
    # print('before remove non-rank index:{}'.format(len(index_to_calc)))
    index_to_calc = [i for i in index_to_calc if i not in remove_non_rank_index]

    all = len(index_to_calc)
    # print('all:{}'.format(all))
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0

    for rank_boundary in rank_boundary_list:
        _tp = np.asarray([1.0 for i in index_to_calc if (in_the_rank(rank_boundary, t[i]) and in_the_rank(rank_boundary, e[i]))]).sum()
        _fp = np.asarray([1.0 for i in index_to_calc if (not in_the_rank(rank_boundary, t[i]) and in_the_rank(rank_boundary, e[i]))]).sum()
        _tn = np.asarray([1.0 for i in index_to_calc if (not in_the_rank(rank_boundary, t[i]) and (not in_the_rank(rank_boundary, e[i])))]).sum()
        _fn = np.asarray([1.0 for i in index_to_calc if (in_the_rank(rank_boundary, t[i]) and (not in_the_rank(rank_boundary, e[i])))]).sum()
        print('rank_boundary:{}, _tp:{}, _fp:{}, _tn:{}, _fn:{}'.format(rank_boundary, _tp, _fp, _tn, _fn))
        tp += _tp
        fp += _fp
        tn += _tn
        fn += _fn
    print('all:{}, tp:{}, fp:{}, tn:{}, fn:{}'.format(all, tp, fp, tn, fn))
    return (tp / (tp + fn))


def get_tf_model_file_paths(tf_model_path, global_iter=None):
    if global_iter is None:
        try:
            print('global_iter is not given. try to get global_iter from tf_model_path:{}'.format(tf_model_path))
            global_iter = tf_model_path.split('.ckpt-')[1]
        except Exception as e:
            print('Can not set global_iter with tf_model_path:{} with Exception:{}'.format(tf_model_path, e))
            global_iter = None
    tf_model_path_with_iter = '{}.ckpt'.format(tf_model_path.split('.ckpt')[0])
    if global_iter is not None:
        tf_model_path_with_iter = '{}-{}'.format(tf_model_path_with_iter, global_iter)

    postfix_list = ['data-00000-of-00001', 'index', 'meta']
    return ['{}.{}'.format(tf_model_path_with_iter, postfix) for postfix in postfix_list], global_iter
