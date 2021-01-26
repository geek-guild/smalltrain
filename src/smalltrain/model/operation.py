import argparse

import sys
import os

import ggutils.s3_access as s3_access

import smalltrain as st

try:
    # For the case smalltrain is installed as Python library
    print('try to load smalltrain modules from Python library')
    from smalltrain.model.nn_model import NNModel
    print('smalltrain modules are ready to be loaded from Python library')
except ModuleNotFoundError:
    if os.environ.get('SMALLTRAIN_HOME'):
        # For the case the environmental value SMALLTRAIN_HOME is exported
        _smalltrain_home_path = os.environ.get('SMALLTRAIN_HOME')
        _smalltrain_home_path = os.path.join(_smalltrain_home_path, 'src')
    else:
        # Try to load smalltrain modules from current directory
        _smalltrain_home_path = './'
    print('try to load smalltrain modules from the path: {}'.format(_smalltrain_home_path))
    sys.path.append(_smalltrain_home_path)
    from smalltrain.model.nn_model import NNModel
    print('smalltrain modules are ready to be loaded from the path: {}'.format(_smalltrain_home_path))

def get_model_list():
    from smalltrain.model.one_dim_cnn_model import OneDimCNNModel
    from smalltrain.model.two_dim_cnn_model import TwoDimCNNModel
    from smalltrain.model.two_dim_cnn_model_v2 import TwoDimCNNModelV2
    model_list = [OneDimCNNModel(), TwoDimCNNModel(), TwoDimCNNModelV2()]
    model_id_list = [model.MODEL_ID for model in model_list]
    return model_list, model_id_list

MODEL_LIST, MODEL_ID_LIST = get_model_list()

def construct_model(log_dir_path, model_id, hparams, train_data=None, debug_mode=True):

    if model_id in MODEL_ID_LIST:
        for _m in MODEL_LIST:
            if _m.MODEL_ID == model_id:
                model = _m.construct_and_prepare_model(log_dir_path=log_dir_path, model_id=model_id, hparams=hparams, train_data=train_data, debug_mode=debug_mode)
                return model
    raise TypeError('Invalid model_id:{}'.format(model_id))


# MODEL_ID_4NN = '4NN_20180808' # 4 nn model 2019/09/10
# MODEL_ID_DNN = 'DNN' # 4 nn model 2019/09/10
# MODEL_ID_1D_CNN = '1D_CNN'
# MODEL_ID_CC = 'CC' # Carbon Copy

# MODEL_ID = MODEL_ID_4NN

class Operation:
    """Operation class as hyper parameter of train or prediction operation
    Arguments:
        params: A dictionary that maps hyper parameter keys and values
        debug_mode: Boolean, if `True` then running with debug mode.
    """

    def __init__(self, hparams=None, setting_file_path=None):
        self._hparam_ins = st.Hyperparameters(hparams, setting_file_path)
        self.hparams_dict = self._hparam_ins.__dict__
        print('init hparams_dict: {}'.format(self.hparams_dict))

    def get_hparams_ins(self):
        return self._hparam_ins

    def update_params_from_file(self, setting_file_path):
        self._hparam_ins.update_hyper_param_from_file(setting_file_path)
        self.hparams_dict = self._hparam_ins.__dict__

    def update_hyper_param_from_json(self, json_obj):
        self._hparam_ins.update_hyper_param_from_json(json_obj)
        self.hparams_dict = self._hparam_ins.__dict__

    def read_hyper_param_from_file(self, setting_file_path):
        '''
        This method is for the compatibility for the codes:

        :param setting_file_path:
        :return:
        '''
        self.update_params_from_file(setting_file_path=setting_file_path)
        self.hparams_dict = self._hparam_ins.__dict__
        return self.hparams_dict

    def prepare_dirs(self):
        '''
        Prepare directories used in operation
        :return:
        '''

        log_dir_path = self.hparams_dict['save_root_dir'] + '/logs/' + self.hparams_dict['train_id']
        log_dir_path = log_dir_path.replace('//', '/')
        os.makedirs(log_dir_path, exist_ok=True)
        self.log_dir_path = log_dir_path
        # Set value to hyperparameter
        self._hparam_ins.set('log_dir_path', log_dir_path)


        save_dir_path = self.hparams_dict['save_root_dir'] + '/model/' + self.hparams_dict['train_id'] + '/'
        save_dir_path = save_dir_path.replace('//', '/')
        os.makedirs(save_dir_path, exist_ok=True)
        self.save_dir_path = save_dir_path
        self._hparam_ins.set('save_dir_path', save_dir_path)

        save_file_name = 'model-{}_lr-{}_bs-{}.ckpt'.format(self.hparams_dict['model_prefix'], self.hparams_dict['learning_rate'],
                                                            self.hparams_dict['batch_size'])
        save_file_path = save_dir_path + '/' + save_file_name
        save_file_path = save_file_path.replace('//', '/')
        self.save_file_path = save_file_path
        self._hparam_ins.set('save_file_path', save_file_path)

        report_dir_path = self.hparams_dict['save_root_dir'] + '/report/' + self.hparams_dict['train_id'] + '/'
        report_dir_path = report_dir_path.replace('//', '/')
        os.makedirs(report_dir_path, exist_ok=True)
        self.report_dir_path = report_dir_path
        self._hparam_ins.set('report_dir_path', report_dir_path)

        operation_dir_path = os.path.join(self.hparams_dict['save_root_dir'], 'operation')
        operation_dir_path = os.path.join(operation_dir_path, self.hparams_dict['train_id'])
        operation_file_path = os.path.join(operation_dir_path, self.hparams_dict['train_id'] + '.json')
        os.makedirs(operation_dir_path, exist_ok=True)
        # self.operation_dir_path = operation_dir_path
        # self.operation_file_path = operation_file_path

        if self.hparams_dict['cloud_root'] is not None:
            print('Upload the hparams to cloud: {}'.format(self.hparams_dict['cloud_root']))
            upload_to_cloud(operation_file_path, self.hparams_dict['cloud_root'], self.hparams_dict['save_root_dir'])

        print('[Operation]DONE prepare_dirs')

    def construct_and_prepare_model(self, hparams=None, train_data=None):

        hparams = hparams or self.hparams_dict
        model_id = hparams['model_id']
        print('construct_and_prepare_model with model_id: {}'.format(model_id))
        if model_id in MODEL_ID_LIST:
            for _m in MODEL_LIST:
                if _m.MODEL_ID == model_id:
                    model = _m.construct_and_prepare_model(log_dir_path=hparams['log_dir_path'], model_id=model_id,
                                                           hparams=hparams, train_data=train_data,
                                                           debug_mode=hparams['debug_mode'])
                    self.model = model
                    return model
        raise TypeError('Invalid model_id:{}'.format(model_id))

    def train(self, hparams=None):
        hparams = hparams or self.hparams_dict

        if self.model is None:
            self.construct_and_prepare_model(hparams=hparams)

        self.model.train(iter_to=hparams['iter_to'], learning_rate=hparams['learning_rate'],
                    batch_size=hparams['batch_size'], dropout_ratio=hparams['dropout_ratio'],
                    l1_norm_reg_ratio=hparams['l1_norm_reg_ratio'], save_file_path=hparams['save_file_path'],
                    report_dir_path=hparams['report_dir_path'])
        print('DONE train data ')
        print('====================')

    def auto(self, hparams=None, setting_file_path=None):

        print('====================')
        print('TODO auto operation with hyper parameter: ')
        print(self.hparams_dict)
        print('====================')
        self.prepare_dirs()
        print('DONE prepare_dirs')
        print('====================')
        print('TODO construct_and_prepare_model')
        self.construct_and_prepare_model()
        print('DONE construct_and_prepare_model')
        print('====================')
        if (not self.hparams_dict.get('prediction_mode')):
            print('TODO train( or test only)')
            self.train()
            print('DONE train( or test only)')
            print('====================')
        print('DONE auto operation')
        print('====================')


def main(exec_param):
    print(exec_param)
    operation = Operation(setting_file_path=exec_param['setting_file_path'])
    operation.auto()


def _main(exec_param):

    print(exec_param)

    operation = Operation()

    if 'setting_file_path' in exec_param.keys() and exec_param['setting_file_path'] is not None:
        operation.update_params_from_file(exec_param['setting_file_path'])
    elif 'json_param' in exec_param.keys() and exec_param['json_param'] is not None:
        operation.update_hyper_param_from_json(exec_param['json_param'])

    exec_param = operation.hparams_dict
    print('updated exec_param:{}'.format(exec_param))

    #  prepare directories
    operation.prepare_dirs()

    if 'scrpit_test' in exec_param.keys() and exec_param['scrpit_test'] == True:
        test_static_methods()

        model = operation.construct_and_prepare_model()
        model.train(iter_to=1000, learning_rate=exec_param['learning_rate'], batch_size=exec_param['batch_size'], dropout_ratio=exec_param['dropout_ratio'], save_file_path=exec_param['save_file_path'])
        exit()

    model = None

    print('====================')
    print('TODO train data ')

    if model is None:
        model = operation.construct_and_prepare_model()

    operation.train()
    print('DONE train data ')
    print('====================')


from pathlib import Path
def download_to_local(path, work_dir_path='/var/tmp/tsp/'):
    ret_path = None
    # check path is local
    if os.path.exists(path): return path
    os.makedirs(work_dir_path, exist_ok=True)
    # check if s3 path
    s3_bucket_name, s3_key = get_bucket_name(path)
    if s3_bucket_name is not None:
        ret_path = os.path.join(work_dir_path, s3_key)
        os.makedirs(Path(ret_path).parent, exist_ok=True)
        s3_access.download(s3_bucket_name=s3_bucket_name, s3_key=s3_key, local_dir=work_dir_path, file_path=s3_key)

    return ret_path

import multiprocessing
def upload_to_cloud(local_path, cloud_root, local_root, with_multiprocessing=True):
    if local_path is None:
        print('No file to upload_to_cloud:local_path:{}'.format(local_path))
        return

    s3_bucket_name, s3_root_key = get_bucket_name(cloud_root)
    if s3_bucket_name is None:
        raise ValueError('Invalid cloud_root:{}'.format(cloud_root))
    if len(local_path.split(local_root)[0]) > 0:
        raise ValueError('Invalid local_path:{} or local_root:{}'.format(local_path, local_root))
    local_path_from_local_root = local_path.split(local_root)[1]
    # print('local_path_from_local_root:{}'.format(local_path_from_local_root))
    s3_key = os.path.join(s3_root_key, local_path_from_local_root)

    local_dir = Path(local_path).parent
    file_path = Path(local_path).name
    if with_multiprocessing:
        # p = multiprocessing.Process(target=s3_access.upload, args=(s3_bucket_name, s3_key, local_dir, file_path,))
        # p.start()
        send_to_s3_uploader(s3_bucket_name=s3_bucket_name, s3_key=s3_key, local_dir=local_dir, file_path=file_path)
    else:
        s3_access.upload(s3_bucket_name=s3_bucket_name, s3_key=s3_key, local_dir=local_dir, file_path=file_path)

def send_to_s3_uploader(s3_bucket_name, s3_key, local_dir, file_path, queue_file_path='/var/tmp/tsp/queue.txt'):
    mode = 'a' if os.path.isfile(queue_file_path) else 'w'
    f = open(queue_file_path, mode)
    f.write('{}, {}, {}, {}\n'.format(s3_bucket_name, s3_key, local_dir, file_path))
    f.close()

def is_s3_path(s3_path):
    s3_bucket_name, s3_key = get_bucket_name(s3_path)
    return (s3_bucket_name is not None)

def get_bucket_name(s3_path):
    if s3_path is None: return None, None
    try:
        _split = s3_path.split('s3://')
        if len(_split[0]) > 0: return None, None
        s3_bucket_name = _split[1].split('/')[0]
        s3_key = _split[1][1 + len(s3_bucket_name):]
        return s3_bucket_name, s3_key
    except IndexError as e:
        print('Can not read s3_bucket_name or s3_key from s3_path:{}'.format(s3_path))
        return None, None

def test_download_to_local():
    path = 's3://your-bucket/tsp/sample/sample.json'
    download_path = download_to_local(path)
    has_downloaded = os.path.isfile(download_path)
    print('[test_download_to_local]from:{}, to:{} has_downloaded:{}'.format(path, download_path, has_downloaded))
    assert has_downloaded

def test_upload_to_cloud():

    # case 1
    local_path = '/var/tsp/sample/test/sample_upload.txt'
    cloud_root = 's3://your-bucket/tsp/sample/test/'
    local_root = '/var/tsp/sample/test/'
    upload_to_cloud(local_path, cloud_root, local_root)


def test_static_methods():
    test_upload_to_cloud()
    exit()
    test_download_to_local()
    print('Done test_static_methods')

def main_with_train_id(train_id):
    print('TODO')


import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tsp')

    parser.add_argument('--model_prefix', '-mp', type=str, default='nn',
                        help='The prefix string representing the model')
    parser.add_argument('--save_root_dir', '-rd', type=str, default='/var/tensorflow/tsp/',
                        help='Root dir for Tensorflow FileWriter')
    parser.add_argument('--init_model_path', '-imp', type=str, default=None,
                        help='Model path to restore Tensorflow session')
    parser.add_argument('--restore_var_name_list', '-rvnl', type=list, default=None,
                        help='restore_var_name_list')
    parser.add_argument('--untrainable_var_name_list', '-utvnl', type=list, default=None,
                        help='untrainable_var_name_list')
    parser.add_argument('--learning_rate', '-ll', type=float, default=1e-4,
                        help='learning_rate of optsimizer')

    # About batch size
    parser.add_argument('--batch_size', '-bs', type=int, default=128,
                        help='batch_size')
    # About minibatch operation
    parser.add_argument('--evaluate_in_minibatch', '-enmb', type=bool, default=False,
                        help = 'Bool, Whether to evaluate in minibatch or not (Default: False)')

    parser.add_argument('--iter_to', '-itr', type=int, default=10000,
                        help='iter_to')
    parser.add_argument('--dropout_ratio', '-dr', type=float, default=0.5,
                        help='Dropout ratio')
    parser.add_argument('--train_id', '-tid', type=str, default='TEST_YYYYMMDD-HHmmSS',
                        help='id attached to model and log dir to identify train operation ')
    parser.add_argument('--model_id', '-mid', type=str, default=st.Hyperparameters.DEFAULT_DICT['model_id'],
                        help='id attached to model to identify model constructure ')
    parser.add_argument('--model_type', '-mty', type=str, default='REGRESSION',
                        help='model_type ')
    parser.add_argument('--prediction_mode', '-pmd', type=bool, default=None,
                        help='Whether prediction mode or not')
    parser.add_argument('--debug_mode', '-dmd', type=bool, default=None,
                        help='Whether debug mode or not')
    parser.add_argument('--monochrome_mode', '-mmd', type=bool, default=False,
                        help='Whether monochrome mode or not')

    parser.add_argument('--optimizer', '-otm', type=str, default=None,
                        help='String, optimizer')

    parser.add_argument('--input_ts_size', '-its', type=int, default=12,
                        help='input_ts_size')
    parser.add_argument('--input_ts_width', '-itw', type=int, default=None,
                        help='input_img_width')
    parser.add_argument('--input_img_width', '-iiw', type=int, default=32,
                        help='input_img_width')
    parser.add_argument('--input_output_ts_offset', '-iotso', type=int, default=1,
                        help='input_output_ts_offset')
    parser.add_argument('--input_output_ts_offset_range', '-iotsor', type=list, default=None,
                        help='input_output_ts_offset_range')
    parser.add_argument('--input_output_ts_offset_list', '-iotsol', type=list, default=None,
                        help='input_output_ts_offset_list')
    parser.add_argument('--has_to_complement_before', '-htcb', type=bool, default=True,
                        help='Whether complement the value before ts starts or not(Default:True)')
    parser.add_argument('--complement_ts', '-cpts', type=str, default=None,
                        help='String, Values to complement the missing time series data (Default:None)')
    parser.add_argument('--n_layer', '-nly', type=int, default=5,
                        help='n_layer')
    parser.add_argument('--num_add_fc_layers', '-nafl', type=int, default=0,
                        help='num_add_fc_layers')
    parser.add_argument('--fc_node_size_list', '-fnsl', type=list, default=None,
                        help='fc_node_size_list')
    parser.add_argument('--fc_weight_stddev_list', '-fwsl', type=list, default=None,
                        help='List of integer, the list of stddevs of weight variables in each fc layers. Default: all 0.1.')
    parser.add_argument('--fc_bias_value_list', '-fbvl', type=list, default=None,
                        help='List of integer, the list of initial values of bias variables in each fc layers. Default: all 0.1')

    # about sub model
    parser.add_argument('--sub_model_url', '-smu', type=str, default=None,
                        help='String, The sub model\'s URL (Default: None, Do not use sub model)')
    parser.add_argument('--sub_model_allocation', '-sma', type=float, default=0.0,
                        help='Float, the allocation of value which flows into the sub model (Default: 0.0, no allocation into the sub model)')
    parser.add_argument('--sub_model_input_point', '-smip', type=str, default=None,
                        help='String, The sub model input point (Default: None, Do not use sub model)')
    parser.add_argument('--sub_model_output_point', '-smop', type=str, default=None,
                        help='String, The sub model output point (Default: None, Do not use sub model)')

    # about ResNet
    parser.add_argument('--has_res_net', '-hrs', type=bool, default=False,
                        help='Whether the model has ResNet (the layers in the model has short cut) or not.')
    parser.add_argument('--num_cnn_layers_in_res_block', '-nclrb', type=int, default=2,
                        help='Integer, the number of CNN layers in one Residual Block (Default: 2)')

    parser.add_argument('--ts_start', '-tss', type=int, default=None,
                        help='ts_start')
    parser.add_argument('--ts_end', '-tse', type=int, default=None,
                        help='ts_end')

    parser.add_argument('--test_ts_index_from', '-tetsif', type=int, default=None,
                        help='test_ts_index_from')
    parser.add_argument('--test_ts_index_to', '-tetsit', type=int, default=None,
                        help='test_ts_index_to')
    parser.add_argument('--max_data_per_ts', '-mdpts', type=int, default=None,
                        help='max_data_per_ts')

    parser.add_argument('--filter_width', '-flw', type=int, default=5,
                        help='filter_width')
    parser.add_argument('--cnn_channel_size', '-ccs', type=int, default=4,
                        help='cnn_channel_size')
    parser.add_argument('--cnn_channel_size_list', '-ccsl', type=list, default=None,
                        help='cnn_channel_size_list')
    parser.add_argument('--pool_size_list', '-psl', type=list, default=None,
                        help='pool_size_list')
    parser.add_argument('--act_func_list', '-actfl', type=list, default=None,
                        help='act_func_list')
    parser.add_argument('--cnn_weight_stddev_list', '-cwsl', type=list, default=None,
                        help='List of integer, the list of stddevs of weight variables in each cnn layers. Default: all 0.1.')
    parser.add_argument('--cnn_bias_value_list', '-cbvl', type=list, default=None,
                        help='List of integer, the list of initial values of bias variables in each cnn layers. Default: all 0.1.')

    # about data augmentation
    parser.add_argument('--flip_randomly_left_right', '-frlr', type=bool, default=True,
                        help='Boolean,  of integer, the list of initial values of bias variables in each cnn layers. Default: all 0.1.')
    parser.add_argument('--crop_randomly', '-crr', type=bool, default=True,
                        help='Boolean, if true, the processed images will be randomly cropped from resized images, the size to resize is set with size_random_crop_from (Default: true).')
    parser.add_argument('--size_random_crop_from', '-srcf', type=int, default=None,
                        help='Integer, the size to which the images will be resized and from whiqch the processed images will be randomly cropped (Default: None, set input_img_width * 1.25 if crop_randomly is true)')
    parser.add_argument('--angle_rotate_randomly', '-rtrnd', type=float, default=None,
                        help='Integer, The Angle by which the image be rotated, randomly choosen between -rt <= x <= +rt (Default: 0)')
    parser.add_argument('--rounding_angle', '-rndang', type=int, default=90,
                        help='Integer, The Angle should be rounded to a multiple of rounding_angle (Default: 90)')
    parser.add_argument('--resize_to_crop_with', '-retcw', type=str, default='scaling_or_padding',
                        help='String, The image needs to be scaling_or_padding or just padding')


    # about L1 term loss
    parser.add_argument('--add_l1_norm_reg', '-al1nr', type=bool, default=False,
                        help='Whether add L1 term or not.')
    parser.add_argument('--l1_norm_reg_ratio', '-l1nrr', type=float, default=0.01,
                        help='L1 term ratio (* L1 term)')
    # about preactivation regularization
    parser.add_argument('--add_preactivation_regularization', '-aprreg', type=bool, default=False,
                        help='Whether add_preactivation_regularization or not.')
    parser.add_argument('--preactivation_regularization_value_ratio', '-prrgvr', type=float, default=0.0,
                        help='preactivation_regularization_value_ratio')
    parser.add_argument('--preactivation_maxout_list', '-prmol', type=list, default=None,
                        help='preactivation_maxout_list')

    # about min-max normalization
    parser.add_argument('--has_minmax_norm', '-hmmn', type=bool, default=True,
                        help='has_minmax_norm')
    parser.add_argument('--input_min', '-imin', type=float, default=None,
                        help='Float, min value of input data. Default: None(will be selected from input test/train data)')
    parser.add_argument('--input_max', '-imax', type=float, default=None,
                        help='Float, max value of input data. Default: None(will be selected from input test/train data)')

    # about batch normalization
    parser.add_argument('--has_batch_norm', '-hbn', type=bool, default=True,
                        help='has_batch_norm')
    parser.add_argument('--bn_decay', '-bnd', type=float, default=NNModel.DEFAULT_BN_DECAY,
                        help='batch normalization param decay')
    parser.add_argument('--bn_eps', '-bne', type=float, default=NNModel.DEFAULT_BN_ESP,
                        help='batch normalization param eps')

    parser.add_argument('--data_dir_path', '-ddp', type=str, default=None,
                        help='data_dir_path')
    parser.add_argument('--data_set_def_path', '-dsdp', type=str, default=None,
                        help='data_set_def_path')

    parser.add_argument('--input_data_names', '-idn', type=str, default=None,
                        help='input_data_names')
    parser.add_argument('--input_data_names_to_be_extended', '-idnex', type=str, default=None,
                        help='input_data_names_to_be_extended')
    parser.add_argument('--output_data_names', '-odn', type=str, default=None,
                        help='output_data_names')
    parser.add_argument('--output_classes', '-ocs', type=int, default=None,
                        help='Integer, the number of output classes (output class size) used in cassification operations. Default: None(will be set from data set or initial model)')

    # col name that has time series data
    parser.add_argument('--dt_col_name', '-tcn', type=str, default=None,
                        help='ts_col_name')
    parser.add_argument('--dt_col_format', '-tcf', type=str, default='YYYY-mm-DD',
                        help='ts_col_format')
    parser.add_argument('--dt_unit', '-tsu', type=str, default='day',
                        help='ts_unit')

    # datetime col
    parser.add_argument('--add_dt_col_name_list', '-adcnl', type=list, default=None,
                        help='add_dt_col_name_list')

    parser.add_argument('--annotation_col_names', '-acn', type=list, default=None,
                        help='annotation_col_names')

    # multi resolution channels
    parser.add_argument('--multi_resolution_channels', '-mrc', type=int, default=0,
                        help='multi resolution channels(default:not add)')
    parser.add_argument('--decrease_resolution_ratio', '-rdr', type=int, default=NNModel.DEFAULT_DECREASE_RESOLUTION_RATIO,
                        help='ratio to decrease to multi resolution channels(default:decrease by {})'.format(NNModel.DEFAULT_DECREASE_RESOLUTION_RATIO))
    parser.add_argument('--decrease_resolution_ratio_list', '-rdrl', type=list, default=None,
                        help='list of ratio to decrease to multi resolution channels. If this set, decrease_resolution_ratio setting will be ignored.')

    parser.add_argument('--target_group', '-tgr', type=str, default=None,
                        help='target_group')

    parser.add_argument('--test_only_mode', '-tomd', type=bool, default=None,
                        help='Whether calc output using test data only(without train) or not')

    parser.add_argument('--mask_rate', '-mskr', type=float, default=None,
                        help='mask_rate')
    parser.add_argument('--col_index_to_mask', '-citm', type=list, default=None,
                        help='Column index to mask. If this is None also maks_rate > 0, then none of columns will be masked.')

    parser.add_argument('--skip_invalid_data', '-sivld', type=bool, default=None,
                        help='skip_invalid_data')
    parser.add_argument('--valid_data_range', '-vldr', type=list, default=None,
                        help='valid_data_range')

    parser.add_argument('--plot_x_label', '-pxl', type=str, default=None,
                        help='plot_x_label')
    parser.add_argument('--plot_y_label', '-pyl', type=str, default=None,
                        help='plot_y_label')
    parser.add_argument('--plot_x_data_name_in_annotation', '-plxdnia', type=str, default=None,
                        help='plot_x_data_name_in_annotation')
    parser.add_argument('--plot_group_data_name_in_annotation', '-plgdnia', type=str, default=None,
                        help='plot_group_data_name_in_annotation')
    parser.add_argument('--plot_x_range', '-plxr', type=list, default=None,
                        help='plot_x_range')
    parser.add_argument('--plot_y_range', '-plyr', type=list, default=None,
                        help='plot_y_range')
    parser.add_argument('--plot_title', '-pltt', type=str, default=None,
                        help='plot_title')
    parser.add_argument('--plot_errors', '-pler', type=list, default=None,
                        help='plot_errors')
    parser.add_argument('--plot_animation', '-pla', type=bool, default=None,
                        help='plot_animation')
    parser.add_argument('--calc_cc_errors', '-cce', type=bool, default=None,
                        help='calc_cc_errors')
    parser.add_argument('--op_errors', '-opers', type=list, default=None,
                        help='op_errors')
    parser.add_argument('--rank_boundary_list', '-rbl', type=list, default=None,
                        help='rank_boundary_list')
    parser.add_argument('--cloud_root', '-clr', type=str, default=None,
                        help='String, cloud_root')
    parser.add_argument('--prioritize_cloud', '-prcl', type=bool, default=False,
                        help='Boolean, prioritize_cloud')

    # frequencies for the tasks duaring iterations
    parser.add_argument('--train_report_frequency', '-trrf', type=int, default=100,
                        help='train report frequency(default:100)')
    parser.add_argument('--test_report_frequency', '-tsrf', type=int, default=100,
                        help='test report frequency(default:100)')
    parser.add_argument('--save_model_frequency', '-smf', type=int, default=100,
                        help='save model frequency(default:100)')
    parser.add_argument('--export_to_onnx', '-eto', type=bool, default=None,
                        help = 'Boolean, whether to refresh train data stored with the key name or not (Default: false).')
    parser.add_argument('--summarize_layer_frequency', '-slf', type=int, default=1000,
                        help='Integer, summarize layerl frequency(default:1000)')
    parser.add_argument('--summarize_layer_name_list', '-slnl', type=int, default=None,
                        help='List of String, summarize_layer_name_list(Default: None)')

    parser.add_argument('--use_cache', '-ucch', type=bool, default=False,
                        help='Boolean, use_cache')
    parser.add_argument('--cache_db_host', '-cchdbh', type=str, default='localhost',
                        help='String, cache_db_host')
    parser.add_argument('--cache_data_set_id', '-cdsid', type=str, default=None,
                        help='String, Data set id. If None, then set with train_id  (Default:None)')
    parser.add_argument('--refresh_cache_data_set', '-rfds', type=bool, default=False,
                        help='Boolean, default: false. Whether to refresh train data stored with the key name or not.')

    parser.add_argument('--json_param', '-jpr', type=str, default=None,
                        help='JSON String to set parameters')
    parser.add_argument('--setting_file_path', '-sfp', type=str, default=None,
                        help='String, The setting file path of JSON String to set parameters')

    parser.add_argument('--scrpit_test', '-sct', type=bool, default=False,
                        help='Boolean, scrpit_test')

    args = parser.parse_args()
    print('args:{}'.format(args))

    exec_param = vars(args)
    print('init exec_param:{}'.format(args))

    main(exec_param)
