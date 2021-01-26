from dateutil.parser import parse as parse_datetime
from datetime import timezone
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from datetime import datetime
import time
import pandas as pd
from PIL import Image
import numpy as np
import math
import csv
import os
import sys
import cv2

from multiprocessing import Process, Manager

from ggutils.gg_data_base import GGDataBase
from ggutils.gg_hash import GGHash

from smalltrain.data_set.gg_data_set import GGData
from smalltrain.data_set.gg_data_set import GGDataSet

# TRAIN_DATA_SET_FILE_PATH = 'data/train_data_set_item_cnt_normed.csv'
# TRAIN_DATA_SET_MERGE_TEST_FILE_PATH = 'data/train_data_set_item_cnt_normed_merge_test.csv'

DT_COL_NAME = None

DEFAULT_DECREASE_RESOLUTION_RATIO = 2

import json
class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if isinstance(obj, (datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.int32, np.int64)):
            return str(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def list_to_hash(arg_list):
    json_str = json.dumps(arg_list, cls=ExtendedJSONEncoder)
    # print('json_str of arg_list:{}'.format(json_str))
    import hashlib
    data_hash_value = hashlib.sha256(json_str.encode()).hexdigest()
    # print('data_hash_value:{}'.format(data_hash_value))
    return data_hash_value


# cache_db_host = 'localhost'
# use_cache = False
witout_check_cache = False

class IMGData(GGData):
    def __init__(self, name, use_db='GGDataBase', db_host='localhost', refresh=False, dtype=np.ndarray):
        super().__init__(name, use_db, db_host, refresh, dtype)


class IMGDataSet(GGDataSet):

    def __init__(self, debug_mode=False,
                 prepare_data_mode=False, prediction_mode=False, hparams=None):
        PREFIX = '[IMGDataSet]'
        self.hparams = hparams
        super().__init__(debug_mode, prepare_data_mode, prediction_mode, hparams)
        print('{}TODO init with hparams:{}'.format(PREFIX, hparams))

        self.ts_start = 0
        if hparams and 'ts_start' in hparams.keys():
            print('{}Use ts_start in hparams:{}'.format(PREFIX, hparams['ts_start']))
            self.ts_start = hparams['ts_start']
        else:
            print('{}TODO Use ts_start with default value:{}'.format(PREFIX, self.ts_start))

        # self.ts_end = 50000 # memory error
        self.ts_end = 10000
        if hparams and 'ts_end' in hparams.keys():
            print('{}Use ts_end in hparams:{}'.format(PREFIX, hparams['ts_end']))
            self.ts_end = hparams['ts_end']
        else:
            print('{}TODO Use ts_end with default value:{}'.format(PREFIX, self.ts_end))


        self.max_data_per_img = None
        if hparams and 'max_data_per_img' in hparams.keys():
            print('{}Use max_data_per_img in hparams:{}'.format(PREFIX, hparams['max_data_per_img']))
            self.max_data_per_img = hparams['max_data_per_img']
        else:
            print('{}TODO Use max_data_per_img with default value:{}'.format(PREFIX, self.max_data_per_img))

        # self.col_size = len(self.input_data_names)
        # Ensure that self.col_size = 1 if Monochrome mode

        if hparams and 'monochrome_mode' in hparams.keys():
            print('{}Use monochrome_mode in hparams:{}'.format(PREFIX, hparams['monochrome_mode']))
            self.monochrome_mode = hparams['monochrome_mode']
        else:
            print('{}TODO Use monochrome_mode with default value'.format(PREFIX))
            self.monochrome_mode = False

        # Ensure that self.col_size = 1 if Monochrome mode
        if self.monochrome_mode:
            self.input_ch_size = 1 # channel size for one color
        else:
            self.input_ch_size = 3 # channel size for 3 colors
        self.col_size = self.input_ch_size
        print('self.col_size:{}'.format(self.col_size))

        self.input_img_width = 32
        if hparams and 'input_img_width' in hparams.keys():
            print('{}Use input_img_width in hparams:{}'.format(PREFIX, hparams['input_img_width']))
            self.input_img_width = hparams['input_img_width']
        else:
            print('{}TODO Use input_img_width with default value'.format(PREFIX))


        self.target_ts_start = self.ts_start

        self.train_ts_width = self.ts_end - self.target_ts_start


    def construct_data_ins(self, name, use_db='GGDataBase', db_host='localhost', refresh=False, dtype=np.ndarray):
        '''
        A Factory method to construct an instance of this class
        :param name:
        :param use_db:
        :param db_host:
        :param refresh:
        :param dtype:
        :return:
        '''
        return IMGData(name, use_db, db_host, refresh, dtype)


    def generate_input_output_data(self):
        print('TODO generate_input_output_data')

        dir_path = self.data_dir_path
        # data_set_file_list = self.get_data_set_file_list(dir_path)
        data_set_file_dict = self.get_data_set_file_dict(dir_path)
        print('data_set_file_dict:{} got from dir_path:{}'.format(data_set_file_dict, dir_path))
        _grid = [[k, data_set_file_dict[k]] for k in data_set_file_dict.keys()]
        _df_data_set_file = pd.DataFrame(_grid, columns=(['data_set_id', 'file_path']))

        print('### _df_data_set_file ###')
        print(_df_data_set_file)

        # merge
        self.df_data_set_def = pd.merge(self.df_data_set_def, _df_data_set_file, how='left')
        print('len df_data_set_def before select target_group:{}'.format(len(self.df_data_set_def)))

        # select target_group
        if self.target_group:
            self.df_data_set_def = self.df_data_set_def[self.df_data_set_def["group"] == self.target_group]
            self.df_data_set_def = self.df_data_set_def.reset_index()
        print('len df_data_set_def after select target_group:{}'.format(len(self.df_data_set_def)))

        data_set_cnt = len(self.df_data_set_def)

        # get label size from data_set_def file

        if 'label' in self.output_data_names:
            # set output_classes if not given with hyper parameter
            if self.output_classes is None:
                if self.model_type == 'CLASSIFICATION':
                    unique_labels = self.df_data_set_def["label"].unique()
                    print('unique_labels:{}'.format(unique_labels))
                    _max_label = self.df_data_set_def["label"].max()
                    self.output_classes = int(max(_max_label + 1, len(unique_labels)))
                    assert self.output_classes > 0
                else:
                    raise Exception('Invalid model_type:{}. only classification model type is available.'.format(self.model_type))

            if self.use_sub_label:
                work = self.df_data_set_def['sub_label'].unique()
            else:
                work = self.df_data_set_def['label'].unique()
            max_label = int(max(work))
            self.target_label_in_use = list(range(0, max_label + 1))
            print('use_sub_label:{}, target_label_in_use:{}'.format(self.use_sub_label, self.target_label_in_use))
            print('max_label: {}, len of target_label_in_use: {}'.format(max_label, len(self.target_label_in_use)))
        else:
            # set output_classes if not given with hyper parameter
            if self.output_classes is None and self.model_type == 'CLASSIFICATION':
                    raise Exception('TODO define n_labels')

        # print('n_labels:{}'.format(self.n_labels))
        print('output_classes:{}'.format(self.output_classes))

        print('train_ts_width:{}'.format(self.train_ts_width))
        self.data_size_reserved = self.train_ts_width * data_set_cnt
        print('self.data_size_reserved:{}'.format(self.data_size_reserved))

        input_data_names = self.input_data_names
        print('input_data_names:{}'.format(input_data_names))


        if self.refresh_cache_data_set:
            self.generate_cache_data()
        else:
            if not self.read_cache_data():
                print('have to generate_cache_data')
                self.generate_cache_data()

        return self.input_data, self.output_data


    def get_data_ins_dict_to_check_cached(self):
        '''
        In Image recognition, annotation_data is not used.
        No need to check annotation_data is cached.
        :return: dict, data instances's name and ref.
        '''
        return {'input_data': self.input_data, 'output_data': self.output_data}


    def generate_cache_data(self):

        input_data = IMGData(name='input_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)
        output_data = IMGData(name='output_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)

        if not self.prepare_data_mode:
            # print('self.col_size to init input_data:{}'.format(self.col_size))
            input_data = IMGData(name='input_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)

            # value output
            if self.model_type == 'CLASSIFICATION':
                output_data = IMGData(name='output_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)
            else:
                raise Exception('Invalid model_type:{}'.format(self.model_type))

            # data_id_set
            data_id_set = IMGData(name='data_id_set_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)

        else:
            output_data = None

        self.test_index_list = []
        self.train_index_list = []

        data_index = 0

        select_train_data_set_with_key_time = 0
        generate_ts_time = 0

        each_data_set_proc_start_time = time.time()
        for _data_set_index, _index in enumerate(self.df_data_set_def.index):
            print('---------- _data_set_index:{} / data_set size:{} ----------'.format(_data_set_index, len(self.df_data_set_def)))
            print('each_data_set_proc_time:{}'.format(time.time() - each_data_set_proc_start_time))
            each_data_set_proc_start_time = time.time()
            reset_time = time.time()

            _series = self.df_data_set_def.iloc[_index]

            data_set_file = _series['file_path']
            data_set_id = _series['data_set_id']
            print('data_set_file:{}'.format(data_set_file))
            print('data_set_id:{}'.format(data_set_id))

            print('data_set_file is None:{}'.format(data_set_file is None))
            print('data_set_file is np.nan:{}'.format(data_set_file is np.nan))
            if data_set_file is None or data_set_file is np.nan:
                print('data_set_file set with data_set_id:{}'.format(data_set_id))
                data_set_file = data_set_id

            label_in_use = None
            label_v = None
            if 'label' in self.output_data_names:
                labels = [_series['label'], _series['sub_label']]
                print('labels:{}'.format(labels))

                label_in_use = self.get_label_in_use(labels)

                print('label_in_use:{}'.format(label_in_use))


                _label_index = np.where(self.target_label_in_use == label_in_use)

                if self.model_type == 'CLASSIFICATION':
                    # label_v = np.eye(self.output_classes)[_label_index]  # one hot
                    label_v = np.eye(self.output_classes)[_label_index].reshape(-1)  # one hot

            else:
                print('do regression or classification to output_data_names:'.format(self.output_data_names))

            test_flag = _series['test']


            def create_img_data(data_set_file, label_in_use,
                               target_ts_start, multi_resolution_channels, decrease_resolution_ratio_list):

                # read from cache if exists
                cache = None
                if self.use_cache and not witout_check_cache:
                    cache = IMGData(name='img_data', db_host=self.cache_db_host, refresh=False, dtype=pd.DataFrame)
                    key = list_to_hash(
                        arg_list=[data_set_file, label_in_use,
                                       target_ts_start, multi_resolution_channels,
                                       decrease_resolution_ratio_list])

                    img_data = cache.get(key)
                    try:
                        # check cache
                        _check = (len(img_data) > 0)
                        # print('len(img_data):{}'.format(len(img_data)))
                        # print('Checking img_data with key:{}, _check:{}'.format(key, _check))
                        assert _check
                        print('Get cache with key:{}'.format(key))
                        return img_data
                    except Exception as e:
                        # print('No cache with key:{} with Exception:{}'.format(key, e))
                        None

                # read data_set
                try:
                    img_data = np.asarray(Image.open(data_set_file))
                except Exception as e:
                    print('error: {} with read img file:{}'.format(e, data_set_file))
                    return None
                print('DONE img_data read')

                # debug
                # if self.debug_mode:
                #     print('img_data first:', img_data[0][0][0])
                #     print('img_data last:', img_data[-1][-1][-1])

                if self.prepare_data_mode: return img_data


                # set cache
                if self.use_cache:
                    cache.set(key, img_data)
                    print('Set cache with key:{}'.format(key))

                return img_data

            select_train_data_set_with_key_time += (time.time() - reset_time)
            reset_time = time.time()

            # img_data args :
            # data_set_file, label_in_use, self.target_ts_start, self.multi_resolution_channels, self.decrease_resolution_ratio_list
            img_data = create_img_data(data_set_file,
                                     label_in_use, self.target_ts_start,
                                     self.multi_resolution_channels, self.decrease_resolution_ratio_list)
            if img_data is None:
                continue

            if self.prepare_data_mode:
                if data_index % 1000 == 0:
                    print('prepare_data_mode data_index:{}/{}'.format(data_index, self.data_size_reserved))
                data_index += 1
                continue


            data_index_from = data_index

            print('========== start with data_index:{} =========='.format(data_index))

            work_img_data = img_data.copy()

            target_ts = self.target_ts_start
            target_ts_end = self.ts_end


            print('work_img_data.shape:{}'.format(work_img_data.shape))
            target_ts_end = min([target_ts_end, work_img_data.shape[0] - 1, work_img_data.shape[1] - 1])


            only_debug_at_first_in_roop = self.debug_mode

            # generate input and output data according to max_data_per_img
            target_ts_list = np.arange(target_ts, target_ts_end + 1)
            try:
                if self.max_data_per_img is not None and self.max_data_per_img > 0:
                    np.random.shuffle(target_ts_list)
                    target_ts_list = target_ts_list[:self.max_data_per_img]
            except Exception as e:
                print('warning. an Exception occured in setting target_ts_list with self.max_data_per_img:{}. target_ts_list will be set as default. Exception:{}'.format(self.max_data_per_img, e))
                target_ts_list = np.arange(target_ts, target_ts_end + 1)

            if self.debug_mode:
                print('target_ts_list:'.format(target_ts_list))

            # TODO
            target_ts_list = [0]
            for target_ts in target_ts_list:

                if data_index % 1000 == 0:
                    print('data_index:{}/{} ({:.1f}%), target_ts:{}'.format(data_index, self.data_size_reserved, 100 * data_index / self.data_size_reserved, target_ts))

                # past ts data
                def create_each_data_id(work_img_data=None, target_ts=None):
                    # data_id: data_set_file
                    return '{}_{}'.format(data_set_file, target_ts)

                if data_id_set is not None:
                    self.set_data_ins(data_id_set, data_index, create_each_data_id(work_img_data, target_ts))

                def create_each_input_data(work_img_data, target_ts, input_img_width,
                                           input_data_names,
                    multi_resolution_channels, decrease_resolution_ratio_list):

                    # read from cache if exists
                    cache = None
                    if self.use_cache and not witout_check_cache:
                        cache = IMGData(name='work_input_data', db_host=self.cache_db_host, refresh=False, dtype=pd.DataFrame)
                        key = list_to_hash(
                            arg_list=[work_img_data, target_ts, input_img_width,
                                               input_data_names,
                        multi_resolution_channels, decrease_resolution_ratio_list])

                        _input_data = cache.get(key)
                        try:
                            # check cache
                            print('len(_input_data):{}'.format(len(_input_data)))
                            _check = (len(_input_data) == input_img_width)
                            # print('Checking cache len(_input_data) with key:{}, _check:{}'.format(key, _check))
                            assert _check
                            print('Get cache with key:{}'.format(key))
                            return _input_data
                        except Exception as e:
                            # print('No cache with key:{} with Exception:{}'.format(key, e))
                            None

                    # print('target_ts:{}'.format(target_ts))
                    # print('input_img_width:{}'.format(input_img_width))
                    # input_filterd = work_img_data[(target_ts - input_img_width):target_ts, (target_ts - input_img_width):target_ts, :]
                    input_filterd = work_img_data[target_ts:target_ts+input_img_width, target_ts:target_ts+input_img_width, :]

                    try:
                        assert (len(input_filterd) == input_img_width)
                    except AssertionError:
                        print('AssertionError len(input_filterd):{}'.format(len(input_filterd)))
                        exit(1)

                    # _input_data = input_filterd[input_data_names]
                    _input_data = input_filterd

                    # set cache
                    if self.use_cache:
                        cache.set(key, _input_data)
                        print('Set cache with key:{}'.format(key))

                    return _input_data

                # set input data
                _input_data = create_each_input_data(work_img_data, target_ts, self.input_img_width,
                                       self.input_data_names,
                                       self.multi_resolution_channels, self.decrease_resolution_ratio_list)

                if only_debug_at_first_in_roop:
                    print('_input_data.shape:{}'.format(_input_data.shape))
                    # print('_input_data:{}'.format(_input_data[:5]))

                # input_data.set(data_index, _input_data)
                self.set_data_ins(input_data, data_index, _input_data)

                # set output data
                def create_each_output_data(work_img_data, target_ts, output_data_names, label_in_use, label_v, model_type, prediction_mode):
                    each_output_data = None

                    if 'label' in output_data_names:
                        if not prediction_mode:
                            if model_type == 'CLASSIFICATION':
                                each_output_data = label_v
                            else:
                                raise Exception('only classification model type is available.')
                    else:
                        if not prediction_mode:
                            if not model_type == 'CLASSIFICATION':
                                raise Exception('only classification model type is available.')

                    return each_output_data

                each_output_data = create_each_output_data(work_img_data, target_ts, self.output_data_names, label_in_use, label_v, self.model_type, self.prediction_mode)
                # output_data.set(data_index, each_output_data)
                self.set_data_ins(output_data, data_index, each_output_data)


                if label_v is not None: print('label_v shape:{}'.format(label_v.shape))
                # print('output_data shape:{}'.format(output_data.shape()))

                if self.prediction_mode:
                    self.test_index_list.append(data_index)
                else:
                    try:
                        # skip invalid data
                        if only_debug_at_first_in_roop: print('self.skip_invalid_data:{}, self.valid_data_range:{}'.format(self.skip_invalid_data, self.valid_data_range))
                        if self.skip_invalid_data is None or (self.skip_invalid_data is not None and self.is_valid_data(output_data.get(data_index))):
                            if self.is_test_data(test_flag, target_ts - 1):
                                if only_debug_at_first_in_roop: print('add to test data. target_ts:{}'.format(target_ts))
                                self.test_index_list.append(data_index)
                            else:
                                if only_debug_at_first_in_roop: print('add to train data. target_ts:{}'.format(target_ts))
                                self.train_index_list.append(data_index)
                        else:
                            if only_debug_at_first_in_roop: print('skip output_data.get(data_index):{}'.format(output_data.get(data_index)))

                    except Exception:
                        _do_nothing = 0
                        if only_debug_at_first_in_roop: print('do not add to any data. train data. target_ts:{}'.format(target_ts))

                # TODO if (target_ts >= self.prediction_ts_start) and (target_ts <= self.prediction_ts_end):
                # TODO     self.prediction_index_list.append(data_index)

                data_index += 1
                target_ts += 1

                only_debug_at_first_in_roop = False

            generate_ts_time += (time.time() - reset_time)

        # wait for thread finish
        if self.multiprocessing:
            while len(self.thread_dict) > 0:
                print('waiting for thread finish with len:{}'.format(len(self.thread_dict)))
                time.sleep(1)

        print('DONE cut data with data_index')

        # TODO print('select_train_data_set_with_key_time / generate_ts_time:{}'.format(select_train_data_set_with_key_time / generate_ts_time))

        if self.prepare_data_mode:
            print('DONE prepare_data_mode')
            exit()

        self.test_index_list = np.asarray(self.test_index_list, dtype=np.int32)
        self.train_index_list = np.asarray(self.train_index_list, dtype=np.int32)

        # cache test_index_list and train_index_list
        forecass_cache = GGHash(name='forecass_cache_{}'.format(self.cache_data_set_id))
        forecass_cache.set('test_index_list', self.test_index_list)
        forecass_cache.set('train_index_list', self.train_index_list)

        self.set_data(input_data, output_data, annotation_data=None, data_id_set=data_id_set)
        print('DONE generate_input_output_data with input_data.get_size():{}'.format(input_data.get_size()))

        return input_data, output_data


    def binarize_img(self, img_data):
        gray_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY).astype('uint8')
        # _, binarized_img = cv2.threshold(gray_img, 30, 255, cv2.THRESH_BINARY)
        binarized_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 15, 8)
        return binarized_img

