import smalltrain as st
import json

def _update_hyper_param_from_json(current_hyper_param, json_param_string, debug_mode=False):
    print('json_param_string: {}'.format(json_param_string))
    # update current_hyper_param by given json_param_string
    try:
        json_obj = json.loads(json_param_string)
        updated_hyper_param = _update_hyper_param_with_json_obj(current_hyper_param, json_obj, debug_mode)
        return updated_hyper_param
    except FileNotFoundError as e:
        print(e)
        return current_hyper_param

def _update_hyper_param_with_json_obj(current_hyper_param, json_obj, debug_mode=False):
    updated_hyper_param = current_hyper_param.copy()
    for _k in json_obj.keys():
        if _k in current_hyper_param.keys():
            _update_v = json_obj[_k]
            if debug_mode:
                print('Update exec_param {} to {}'.format(_k, _update_v))
            updated_hyper_param[_k] = _update_v
        else:
            if debug_mode:
                print('No key {} in exec_param'.format(_k))
    return updated_hyper_param


from smalltrain.model.nn_model import NNModel
DEFAULT_MODEL_ID = NNModel.MODEL_ID

class Hyperparameters:
    """Operation class as hyper parameter of train or prediction operation
    Arguments:
        params: A dictionary that maps hyper parameter keys and values
        debug_mode: Boolean, if `True` then running with debug mode.
    """

    DEFAULT_DICT = {
        'model_prefix': 'nn',
        'learning_rate': 1e-4,
        'prediction_mode': None,
        'save_root_dir': '/var/tensorflow/tsp/',
        'init_model_path': None,
        'restore_var_name_list': None,
        'untrainable_var_name_list': None,

        # About batch size
        'batch_size': 128,
        # About minibatch operation
        'evaluate_in_minibatch': False,
        # Abount multiprocessing
        'multiprocessing': True,
        'max_threads': 40,

        'iter_to': 10000,
        'dropout_ratio': 0.5,
        'train_id': 'TEST_YYYYMMDD-HHmmSS',
        'model_id': DEFAULT_MODEL_ID,
        'model_type': 'REGRESSION',
        'prediction_mode': None,
        'debug_mode': None,
        'monochrome_mode': False,
        'set_model_parameter_from_dataset': True,
        'optimizer': None,
        'input_ts_size': 12,
        'input_img_width': 12,
        'input_output_ts_offset': 1,
        'input_output_ts_offset_range': None,
        'input_output_ts_offset_list': None,
        'has_to_complement_before': True,
        'n_layer': 5,
        'num_add_fc_layers': 0,
        'fc_node_size_list': None,
        'fc_weight_stddev_list': None,
        'fc_bias_value_list': None,

        # Abount quantization
        'quantize_layer_names': None,

        # about sub model
        'sub_model_url': None,
        'sub_model_allocation': 0.0,
        'sub_model_input_point': None,
        'sub_model_output_point': None,

        # about ResNet
        'has_res_net': False,
        'num_cnn_layers_in_res_block': 2,

        'ts_start': None,
        'ts_end': None,
        'test_ts_index_from': None,
        'test_ts_index_to': None,
        'max_data_per_ts': None,
        'filter_width': 5,
        'cnn_channel_size': 4,
        'cnn_channel_size_list': None,
        'pool_size_list': None,
        'act_func_list': None,
        'cnn_weight_stddev_list': None,
        'cnn_bias_value_list': None,

        # about data augmentation
        'flip_randomly_left_right': False,
        'crop_randomly': False,
        'size_random_crop_from': None,
        'angle_rotate_randomly': None,
        'rounding_angle': 90,
        'resize_to_crop_with': None,

        # about L1 term loss
        'add_l1_norm_reg': False,
        'l1_norm_reg_ratio': 0.0,
        # about preactivation regularization
        'add_preactivation_regularization': False,
        'preactivation_regularization_value_ratio': 0.0,
        'preactivation_maxout_list': None,

        # about min-max normalization
        'has_minmax_norm': True,
        'input_min': None,
        'input_max': None,
        # about batch normalization
        'has_batch_norm': True,
        'bn_decay': NNModel.DEFAULT_BN_DECAY,
        'bn_eps': NNModel.DEFAULT_BN_ESP,
        'data_dir_path': None,
        'data_set_def_path': None,

        'cache_data_set_id': None,
        'refresh_cache_data_set': True,

        'input_data_names': None,
        'input_data_names_to_be_extended': None,
        'col_size': None,

        'output_data_names': None,
        'output_classes': None,
        # col name that has time series data
        'dt_col_name': None,
        'dt_col_format': 'YYYY-mm-DD',
        'dt_unit': 'day',
        # datetime col
        'add_dt_col_name_list': None,
        'annotation_col_names': None,
        'multi_resolution_channels': 0,
        'decrease_resolution_ratio': NNModel.DEFAULT_DECREASE_RESOLUTION_RATIO,
        'decrease_resolution_ratio_list': None,
        'target_group': None,
        'test_only_mode': None,
        'mask_rate': None,
        'col_index_to_mask': None,
        'skip_invalid_data': None,
        'valid_data_range': None,
        'plot_x_label': None,
        'plot_y_label': None,
        'plot_x_data_name_in_annotation': None,
        'plot_group_data_name_in_annotation': None,
        'plot_x_range': None,
        'plot_y_range': None,
        'plot_title': None,
        'plot_errors': None,
        'plot_animation': None,
        'calc_cc_errors': None,
        'op_errors': None,
        'rank_boundary_list': None,
        'cloud_root': None,
        'prioritize_cloud': False,
        'train_report_frequency': False,
        'test_report_frequency': False,
        'save_model_frequency': False,
        'export_to_onnx': False,
        'summarize_layer_frequency': None,
        'use_cache': False,
        'cache_db_host': 'localhost',
        'json_param': None,
        'scrpit_test': False,
    }

    def __init__(self, hparams=None, setting_file_path=None):

        self.__dict__ = Hyperparameters.DEFAULT_DICT

        if hparams is not None:
            for k in hparams.keys():
                self.__dict__[k] = hparams[k]
        if setting_file_path is not None:
            self.update_hyper_param_from_file(setting_file_path)

    def set(self, param_name, value):
        self.__dict__[param_name] = value

    def get(self, param_name):
        value = self.__dict__[param_name]
        return value

    def get_as_dict(self):
        return self.__dict__

    def update_hyper_param_from_file(self, setting_file_path):
        print('update_hyper_param_from_file with setting_file_path: {}'.format(setting_file_path))
        try:
            with open(setting_file_path) as f:
                json_obj = json.load(f)
                updated_hyper_param = _update_hyper_param_with_json_obj(self.__dict__, json_obj)
                self.__dict__ = updated_hyper_param
                # Also add setting_file_path as hparam
                self.__dict__['setting_file_path'] = setting_file_path
        except FileNotFoundError as e:
            print(e)

    def update_hyper_param_from_json(self, json_obj):
        updated_hyper_param = _update_hyper_param_with_json_obj(self.__dict__, json_obj)
        self.__dict__ = updated_hyper_param


