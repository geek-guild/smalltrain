from dateutil.parser import parse as parse_datetime
from datetime import timezone
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from datetime import datetime
import time
import pandas as pd
import numpy as np
import math
import csv
import os
import sys

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

class TSData(GGData):
    def __init__(self, name, use_db='GGDataBase', db_host='localhost', refresh=False, dtype=np.ndarray):
        super().__init__(name, use_db, db_host, refresh, dtype)


class TSDataSet(GGDataSet):

    def __init__(self, debug_mode=False,
                 prepare_data_mode=False, prediction_mode=False, hparams=None):
        PREFIX = '[TSDataSet]'
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

        self.test_ts_index_from = None
        if hparams and 'test_ts_index_from' in hparams.keys():
            print('{}Use test_ts_index_from in hparams:{}'.format(PREFIX, hparams['test_ts_index_from']))
            self.test_ts_index_from = hparams['test_ts_index_from']
        if self.test_ts_index_from is None:
            self.test_ts_index_from = self.ts_start
            print('{}TODO Use test_ts_index_from with default value:{}'.format(PREFIX, self.test_ts_index_from))

        self.test_ts_index_to = None
        if hparams and 'test_ts_index_to' in hparams.keys():
            print('{}Use test_ts_index_to in hparams:{}'.format(PREFIX, hparams['test_ts_index_to']))
            self.test_ts_index_to = hparams['test_ts_index_to']
        if self.test_ts_index_to is None:
            self.test_ts_index_to = self.ts_end
            print('{}Use test_ts_index_to with default value:{}'.format(PREFIX, self.test_ts_index_to))

        # self.train_days = int((self.date_end - self.date_start).days) + 1

        self.max_data_per_ts = None
        if hparams and 'max_data_per_ts' in hparams.keys():
            print('{}Use max_data_per_ts in hparams:{}'.format(PREFIX, hparams['max_data_per_ts']))
            self.max_data_per_ts = hparams['max_data_per_ts']
        else:
            print('{}TODO Use max_data_per_ts with default value:{}'.format(PREFIX, self.max_data_per_ts))

        self.dt_col_name = DT_COL_NAME
        if hparams and 'dt_col_name' in hparams.keys():
            print('{}Use dt_col_name in hparams:{}'.format(PREFIX, hparams['dt_col_name']))
            self.dt_col_name = hparams['dt_col_name']
        else:
            print('{}No dt_col_name is set'.format(PREFIX))

        self.col_size = len(self.input_data_names)

        self.dt_col_format = None
        if hparams and 'dt_col_format' in hparams.keys():
            print('{}Use dt_col_format in hparams:{}'.format(PREFIX, hparams['dt_col_format']))
            self.dt_col_format = hparams['dt_col_format']
        else:
            print('{}Error no dt_col_format'.format(PREFIX))
            exit(1)

        self.dt_unit = None
        if hparams and 'dt_unit' in hparams.keys():
            print('{}Use dt_unit in hparams:{}'.format(PREFIX, hparams['dt_unit']))
            self.dt_unit = hparams['dt_unit']
        else:
            print('{}Error no dt_unit'.format(PREFIX))
            exit(1)

        self.dt_unit_per_sec = TSDataSet.get_seconds_with_dt_unit(dt_unit=self.dt_unit, time_step=1)
        print('self.dt_unit_per_sec:{}'.format(self.dt_unit_per_sec))

        # about add_dt_col_name_list
        self.add_dt_col_name_list = None
        if hparams and 'add_dt_col_name_list' in hparams.keys():
            print('{}Use add_dt_col_name_list in hparams:{}'.format(PREFIX, hparams['add_dt_col_name_list']))
            self.add_dt_col_name_list = hparams['add_dt_col_name_list']
            if self.add_dt_col_name_list is not None:
                # check the members of add_dt_col_name_list
                len_of_add_dt_col_name_list = len(self.add_dt_col_name_list)
                # print('len_of_add_dt_col_name_list:{}'.format(len_of_add_dt_col_name_list))
                if len_of_add_dt_col_name_list < 1:
                    self.add_dt_col_name_list = None
                for add_dt_col_name in self.add_dt_col_name_list:
                    try:
                        col_name = add_dt_col_name['col_name']
                        func = add_dt_col_name['func']
                        print('{}add_dt_col_name_list col_name:{}, func:{}'.format(PREFIX, col_name, func))
                    except KeyError:
                        print('{}Error on invalid parameter:{}'.format(PREFIX, add_dt_col_name))
                        exit(1)
        else:
            print('{}No dt_col_name is set'.format(PREFIX))

        print('self.col_size:{}'.format(self.col_size))

        # (For compatibility with ver0.1.1 ```input_ts_size``` and ver0.1.2 ```input_ts_width``` )
        self.input_ts_size = 100
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


        self.input_output_ts_offset = 1 # default:predict 1 time step ahead if to predict not label but unknown value
        if hparams and 'input_output_ts_offset' in hparams.keys():
            print('{}Use input_output_ts_offset in hparams:{}'.format(PREFIX, hparams['input_output_ts_offset']))
            self.input_output_ts_offset = hparams['input_output_ts_offset']
        else:
            print('{}Use input_output_ts_offset with default value:{}'.format(PREFIX, self.input_output_ts_offset))

        # setting input_output_ts_offset_list with input_output_ts_offset_range
        # If input_output_ts_offset_list is given, input_output_ts_offset_range is not used and input_output_ts_offset_list is used as it is.
        # If input_output_ts_offset_list is not given, it is given as the list of input_output_ts_offset_range.
        # (input_output_ts_offset_list が与えられた場合、input_output_ts_offset_rangeは使用されず、input_output_ts_offset_listがそのまま利用される。)
        # (input_output_ts_offset_list が与えられていない場合、input_output_ts_offset_range を リスト化してinput_output_ts_offset_list として利用する。)
        self.input_output_ts_offset_range = None
        if hparams and 'input_output_ts_offset_range' in hparams.keys():
            print('{}Use input_output_ts_offset_range in hparams:{}'.format(PREFIX, hparams['input_output_ts_offset_range']))
            self.input_output_ts_offset_range = hparams['input_output_ts_offset_range']
        else:
            print('{}No input_output_ts_offset_range'.format(PREFIX))

        self.input_output_ts_offset_list = [self.input_output_ts_offset]
        if hparams and 'input_output_ts_offset_list' in hparams.keys():
            print('{}Use input_output_ts_offset_list in hparams:{}'.format(PREFIX, hparams['input_output_ts_offset_list']))
            self.input_output_ts_offset_list = hparams['input_output_ts_offset_list']
        if self.input_output_ts_offset_list is None:
            print('{}TODO reset input_output_ts_offset_list because input_output_ts_offset_list set None.'.format(PREFIX))
            if self.input_output_ts_offset_range is not None:
                self.input_output_ts_offset_list = list(range(self.input_output_ts_offset_range[0], 1 + self.input_output_ts_offset_range[1]))
            else:
                self.input_output_ts_offset_list = [self.input_output_ts_offset]
        print('{}input_output_ts_offset_list is set :{}'.format(PREFIX, self.input_output_ts_offset_list))

        # add dt col
        if self.dt_col_name is not None and self.add_dt_col_name_list is not None:
            self.col_size += len(self.add_dt_col_name_list)
            _add_col_name_list = [add_dt_col_name['col_name'] for add_dt_col_name in self.add_dt_col_name_list]
            self.input_data_names.extend(_add_col_name_list)

        # add offset column
        self.offset_column_index = -1
        self.add_offset_column = (len(self.input_output_ts_offset_list) > 1)
        print('self.add_offset_column is:{} because of len(self.input_output_ts_offset_list):{}'.format(self.add_offset_column, len(self.input_output_ts_offset_list)))
        if self.add_offset_column is not None:
            self.add_offset_column = True
            self.col_size += 1
            self.offset_column_index = (self.col_size - 1)
        print('{}add_offset_column:{}'.format(PREFIX, self.add_offset_column))

        # self.target_ts_start = self.ts_start + self.input_ts_width
        self.target_ts_start = self.ts_start # start without offsets of self.input_ts_width

        self.train_ts_width = self.ts_end - self.target_ts_start

        # self.data_size = self.target_item_shop_list * self.train_ts_width


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
        return TSData(name, use_db, db_host, refresh, dtype)


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
                    self.output_classes = max(_max_label + 1, len(unique_labels))
                    assert self.output_classes > 0
                elif self.model_type == 'REGRESSION':
                    # self.output_classes = 1
                    self.output_classes = None
                else:
                    raise Exception('Invalid model_type:{}'.format(self.model_type))

            if self.use_sub_label:
                self.target_label_in_use = self.df_data_set_def['sub_label'].unique()
            else:
                self.target_label_in_use = self.df_data_set_def['label'].unique()
            self.target_label_in_use.sort()
            print('use_sub_label:{}, target_label_in_use:{}'.format(self.use_sub_label, self.target_label_in_use))
        else:
            # set output_classes if not given with hyper parameter
            if self.output_classes is None and self.model_type == 'CLASSIFICATION':
                    raise Exception('TODO define n_labels')

        # print('n_labels:{}'.format(self.n_labels))
        print('output_classes:{}'.format(self.output_classes))

        # debug
        # train_data_cnt_per_labels = self.train_data_cnt_per_labels
        # test_data_cnt_per_labels = self.test_data_cnt_per_labels

        # print('train_ts_width:{}'.format(self.train_ts_width))
        self.data_size_reserved = self.train_ts_width * data_set_cnt * len(self.input_output_ts_offset_list)
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


    def is_test_data(self, test_flag, data_index):
        '''

        :param test_flag:
          0: train data
          1: test data
          2: (for time series data) test data if data_index is withinn [self.test_ts_index_from, self.test_ts_index_to]
        :param data_index:
        :return: True or False
        '''
        if test_flag == 0:
            return False
        if self.test_ts_index_from is None and self.test_ts_index_to is None:
            # non time series data
            return True
        else:
            # time series data
            return (self.test_ts_index_from <= data_index and data_index <= self.test_ts_index_to)

    def generate_cache_data(self):

        input_data = TSData(name='input_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)
        output_data = TSData(name='output_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)
        annotation_data = TSData(name='annotation_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)

        if not self.prepare_data_mode:
            print('self.col_size to init input_data:{}'.format(self.col_size))
            input_data = TSData(name='input_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)

            # value output
            if self.model_type == 'CLASSIFICATION':
                # output_data = np.zeros([data_size, self.output_classes],
                #                            dtype=np.float32)  # target_day, class_size, item_cnt_size
                output_data = TSData(name='output_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)
                if self.annotation_col_names is not None:
                    # annotation_data = np.array([data_size, self.output_classes + 1 + len(self.annotation_col_names)])
                    # annotation_data = np.empty((data_size, 1 + 1 + len(self.annotation_col_names)), dtype=object)
                    annotation_data = TSData(name='annotation_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)
            elif self.model_type == 'REGRESSION':
                # output_data = np.zeros([data_size],
                #                            dtype=np.float32)  # target_day, class_size, item_cnt_size
                output_data = TSData(name='output_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)
                if self.annotation_col_names is not None:
                    # annotation_data = np.empty((data_size, 1 + 1 + len(self.annotation_col_names)), dtype=object)
                    annotation_data = TSData(name='annotation_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)
            else:
                raise Exception('Invalid model_type:{}'.format(self.model_type))

            # data_id_set
            data_id_set = None
            if self.dt_col_name and self.add_dt_col_name_list:
                dt_now = datetime.now()
                # data_id_set = np.array([dt_now for x in list(range(data_size))])
                data_id_set = TSData(name='data_id_set_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)

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
                    print('label_v:{}'.format(label_v))

            else:
                print('do regression or classification to output_data_names:'.format(self.output_data_names))

            test_flag = _series['test']

            def calc_size_to_complement_before(input_ts_width, target_ts_start, multi_resolution_channels,
                                               decrease_resolution_ratio_list):

                size_to_complement_before = max(0, input_ts_width - target_ts_start)
                if multi_resolution_channels > 0:
                    size_to_complement_before = max(0, (input_ts_width - 1) * max(
                        decrease_resolution_ratio_list) - target_ts_start)

                print('size_to_complement_before:{}'.format(size_to_complement_before))
                return size_to_complement_before

            def create_ts_data(data_set_file, dt_col_name, dt_unit, dt_unit_per_sec, label_in_use,
                               has_to_complement_before, input_ts_width, target_ts_start, multi_resolution_channels, decrease_resolution_ratio_list):

                # read from cache if exists
                cache = None
                if self.use_cache and not witout_check_cache:
                    cache = TSData(name='ts_data', db_host=self.cache_db_host, refresh=False, dtype=pd.DataFrame)
                    key = list_to_hash(
                        arg_list=[data_set_file, dt_col_name, dt_unit, dt_unit_per_sec, label_in_use,
                                       has_to_complement_before, input_ts_width, target_ts_start, multi_resolution_channels,
                                       decrease_resolution_ratio_list])

                    ts_data = cache.get(key)
                    try:
                        # check cache
                        _check = (len(ts_data) > 0)
                        # print('len(ts_data):{}'.format(len(ts_data)))
                        # print('Checking ts_data with key:{}, _check:{}'.format(key, _check))
                        assert _check
                        print('Get cache with key:{}'.format(key))
                        return ts_data
                    except Exception as e:
                        # print('No cache with key:{} with Exception:{}'.format(key, e))
                        None

                # read data_set
                try:
                    ts_data = pd.read_csv(data_set_file)
                except Exception as e:
                    print('error: {} with read csv:{}'.format(e, data_set_file))
                    return None
                print('DONE ts_data read')
                # TODO ts_data = self.read()

                if dt_col_name:
                    if not isinstance(ts_data[dt_col_name][0], datetime):
                        ts_data[dt_col_name] = [str(dt) for dt in ts_data[dt_col_name]]

                    if isinstance(ts_data[dt_col_name][0], str):
                        print('TODO ts_data dt col parse_datetime')
                        if dt_unit.find('relative') >= 0:
                            ts_data[dt_col_name] = [datetime.fromtimestamp(int(str_dt) * dt_unit_per_sec) for
                                                         str_dt in ts_data[dt_col_name]]
                            # print(ts_data[dt_col_name])
                        else:
                            ts_data[dt_col_name] = [parse_datetime(str_dt) for str_dt in ts_data[dt_col_name]]
                        print('DONE ts_data dt col parse_datetime')
                    ts_data.sort_values(by=[dt_col_name])
                    print('DONE ts_data.sort_values')

                    first_ts_dt = ts_data[dt_col_name][0]

                # add label to ts_data for annotation if defined with data_set_def
                if 'label' in self.output_data_names:
                    ts_data['label'] = label_in_use

                ts_data = ts_data.reset_index()
                print('DONE ts_data.reset_index')

                # debug
                if self.debug_mode:
                    print('ts_data first:', ts_data.iloc[0])
                    print('ts_data last:', ts_data.iloc[-1])

                if self.prepare_data_mode: return ts_data

                # complement_input_data
                # Complement missing time series
                if self.debug_mode:
                    print('self.complement_input_data:{}'.format(self.complement_input_data))
                if self.complement_input_data is not None:
                    df_complement_row_index = max(ts_data.index) + 1
                    for i in range(1, len(ts_data)):
                        current_ts = ts_data.iloc[i - 1][dt_col_name]
                        next_ts = ts_data.iloc[i][dt_col_name]
                        _td = current_ts - next_ts
                        if _td == TSDataSet.get_timedelta_with_dt_unit(dt_unit=dt_unit, time_step=1):
                            continue
                        current_ts += TSDataSet.get_timedelta_with_dt_unit(dt_unit=dt_unit, time_step=1)
                        while current_ts < next_ts:
                            df_complement_row = pd.DataFrame([ts_data.iloc[i].values.tolist()],
                                                             index = [df_complement_row_index],
                                                             columns = ts_data.columns)
                            df_complement_row[dt_col_name] = current_ts
                            # complement ts cols
                            for comp_col, comp_val in self.complement_input_data.items():
                                df_complement_row[comp_col] = comp_val
                            ts_data = pd.concat([ts_data, df_complement_row])
                            current_ts += TSDataSet.get_timedelta_with_dt_unit(dt_unit=dt_unit, time_step=1)
                            df_complement_row_index += 1
                    ts_data = ts_data.sort_values(by=[dt_col_name])
                    ts_data = ts_data.reset_index()
                    print('----- after self.complement_input_data:{} -----'.format(self.complement_input_data))
                    print('head rows:{}'.format(ts_data.iloc[:21]))
                    print('----------')


                # Complement missing front values
                if not has_to_complement_before:
                    return ts_data
                else:
                    size_to_complement_before = calc_size_to_complement_before(input_ts_width, target_ts_start,
                                                                               multi_resolution_channels,
                                                                               decrease_resolution_ratio_list)

                    # TODO Define complement method (fill before / foreward)
                    first_row = ts_data.iloc[0]
                    print('first_row:{}'.format(first_row))

                    l = len(ts_data)
                    print('##### before complementation #####')
                    print('head rows:{}'.format(ts_data.iloc[:5]))
                    print('dt_col_name:{}'.format(dt_col_name))
                    df_complement = pd.DataFrame([first_row.values.tolist()], index=[0 - size_to_complement_before],
                                                 columns=ts_data.columns)
                    for i in range(1, size_to_complement_before):
                        df_complement_row = pd.DataFrame([first_row.values.tolist()],
                                                         index=[i - size_to_complement_before], columns=ts_data.columns)
                        if dt_col_name:
                            _td = TSDataSet.get_timedelta_with_dt_unit(dt_unit=dt_unit,
                                                                       time_step=size_to_complement_before - i)
                            df_complement_row[dt_col_name] = first_ts_dt - _td

                        # ts_data = pd.concat([df_complement_row, ts_data])
                        df_complement = pd.concat([df_complement_row, df_complement])
                        # ts_data[i - size_to_complement_before] = df_complement_row.copy()
                        if i % 1000 == 0:
                            print('DONE complementation index:{}/{} ({:.1f}%)'.format(i, size_to_complement_before,
                                                                                      100 * i / size_to_complement_before))

                    ts_data = pd.concat([df_complement, ts_data])
                    ts_data = ts_data.sort_index()
                    print('##### after complementation #####')

                print('head rows:{}'.format(ts_data.iloc[:21]))

                # set cache
                if self.use_cache:
                    cache.set(key, ts_data)
                    print('Set cache with key:{}'.format(key))

                return ts_data

            select_train_data_set_with_key_time += (time.time() - reset_time)
            reset_time = time.time()

            # ts_data args :
            # data_set_file, self.dt_col_name, self.dt_unit, self.dt_unit_per_sec, label_in_use, self.has_to_complement_before, self.input_ts_width, self.target_ts_start, self.multi_resolution_channels, self.decrease_resolution_ratio_list
            ts_data = create_ts_data(data_set_file, self.dt_col_name, self.dt_unit, self.dt_unit_per_sec,
                                     label_in_use, self.has_to_complement_before, self.input_ts_width, self.target_ts_start,
                                     self.multi_resolution_channels, self.decrease_resolution_ratio_list)
            if ts_data is None:
                continue
            size_to_complement_before = calc_size_to_complement_before(self.input_ts_width, self.target_ts_start,
                                                                       self.multi_resolution_channels,
                                                                       self.decrease_resolution_ratio_list)

            if self.prepare_data_mode:
                if data_index % 1000 == 0:
                    print('prepare_data_mode data_index:{}/{}'.format(data_index, self.data_size_reserved))
                data_index += 1
                continue


            data_index_from = data_index

            for _input_output_ts_offset in self.input_output_ts_offset_list:

                print('========== _input_output_ts_offset:{} start with data_index:{} =========='.format(_input_output_ts_offset, data_index))


                def create_work_ts_data(ts_data, dt_col_name, add_dt_col_name_list, dt_unit, each_input_output_ts_offset):
                    PREFIX = ''
                    # read from cache if exists
                    cache = None
                    if self.use_cache and not witout_check_cache:
                        cache = TSData(name='work_ts_data', db_host=self.cache_db_host, refresh=False, dtype=pd.DataFrame)
                        key = list_to_hash(arg_list=[ts_data, dt_col_name, add_dt_col_name_list, dt_unit, each_input_output_ts_offset])
                        work_ts_data = cache.get(key)
                        try:
                            # check cache
                            # print('len(work_ts_data):{}'.format(len(work_ts_data)))
                            for i in range(min(2, len(work_ts_data))):
                                _check = work_ts_data.iloc[i][dt_col_name] == ts_data.iloc[i][dt_col_name]
                                # print('Checking cache with key:{}, _check:{}'.format(key, _check))
                                assert _check

                            print('Get cache with key:{}'.format(key))
                            return work_ts_data
                        except Exception as e:
                            # print('No cache with key:{} with Exception:{}'.format(key, e))
                            None

                    work_ts_data = ts_data.copy()
                    if dt_col_name is not None:
                        _dt_list = work_ts_data[dt_col_name]

                        _td = TSDataSet.get_timedelta_with_dt_unit(dt_unit=dt_unit,
                                                                   time_step=each_input_output_ts_offset)
                        _target_dt = [d + _td for d in _dt_list]
                        work_ts_data['TargetDateTime'] = _target_dt

                    if dt_col_name is not None and add_dt_col_name_list is not None:
                        print('##### before add datetime columns #####')
                        print('head rows:{}'.format(work_ts_data.iloc[:8]))

                        for add_dt_col_name in add_dt_col_name_list:
                            col_name = add_dt_col_name['col_name']
                            func = add_dt_col_name['func']
                            offset = float(add_dt_col_name['offset']) if 'offset' in add_dt_col_name.keys() else 0.0
                            print('add_dt_col_name_list col_name:{}, func:{}'.format(col_name, func))

                            if func.lower() in ['get_year']:
                                add_dt_list = [(float(d.year) - 2000.0) for d in _dt_list]
                            elif func.lower() in ['get_month']:
                                # add_dt_list = [-0.5 + (float(d.month) / 12.0) for d in _dt_list]
                                add_dt_list = [float(d.month) - 6.0 for d in _dt_list]
                            elif func.lower() in ['get_day']:
                                # add_dt_list = [-0.5 + (float(d.day) / 30.0) for d in _dt_list]
                                add_dt_list = [float(d.day) - 15.0 for d in _dt_list]
                            elif func.lower() in ['get_hour']:
                                add_dt_list = [float(d.hour) - 12.0 for d in _dt_list]
                            elif func.lower() in ['get_minute']:
                                add_dt_list = [float(d.minute) - 30.0 for d in _dt_list]
                            elif func.lower() in ['get_target_year']:
                                # add_dt_list = (float(target_dt.year) - 2000.0) * np.ones(len(_dt_list))
                                add_dt_list = [(float(d.year) - 2000.0) for d in _target_dt]
                            elif func.lower() in ['get_target_month']:
                                # add_dt_list = float(target_dt.month) * np.ones(len(_dt_list))
                                add_dt_list = [float(d.month) - 6.0 for d in _target_dt]
                            elif func.lower() in ['get_target_day']:
                                # add_dt_list = float(target_dt.day) * np.ones(len(_dt_list))
                                add_dt_list = [float(d.day) - 15.0 for d in _target_dt]
                            elif func.lower() in ['get_target_hour']:
                                add_dt_list = [float(d.hour) - 12.0 for d in _target_dt]
                            elif func.lower() in ['get_target_minute']:
                                add_dt_list = [float(d.minute) - 30.0 for d in _target_dt]
                            elif func.lower() in ['get_target_time_stamp_in_day']:
                                add_dt_list = [float(datetime.timestamp(d) / (60 * 60 * 24)) for d in _target_dt]


                            work_ts_data[col_name] = [x + offset for x in add_dt_list]

                        print('##### after add datetime columns #####')
                        print('head rows:{}'.format(work_ts_data.iloc[:21]))

                    # set cache
                    if self.use_cache:
                        cache.set(key, work_ts_data)
                        print('Set cache with key:{}'.format(key))

                    return work_ts_data

                work_ts_data = create_work_ts_data(ts_data, self.dt_col_name, self.add_dt_col_name_list, self.dt_unit, _input_output_ts_offset)

                target_ts = self.target_ts_start
                target_ts_end = self.ts_end
                if size_to_complement_before > 0:
                    target_ts += size_to_complement_before
                if self.has_to_complement_before:
                    target_ts_end += size_to_complement_before

                print('len(work_ts_data):{}'.format(len(work_ts_data)))
                target_ts_end = min(target_ts_end, (len(work_ts_data) - 1))


                # 予測対象がlabelでなくデータ値である場合、時系列の終点は、オフセット分だけ過去方向に戻す必要がある
                if 'label' not in self.output_data_names:
                    try:
                        target_ts_end -= self.input_output_ts_offset_list[-1]
                    except:
                        target_ts_end -= self.input_output_ts_offset

                only_debug_at_first_in_roop = self.debug_mode

                # generate input and output data according to max_data_per_ts
                target_ts_list = np.arange(target_ts, target_ts_end + 1)
                try:
                    if self.max_data_per_ts is not None and self.max_data_per_ts > 0:
                        np.random.shuffle(target_ts_list)
                        target_ts_list = target_ts_list[:self.max_data_per_ts]
                except Exception as e:
                    print('warning. an Exception occured in setting target_ts_list with self.max_data_per_ts:{}. target_ts_list will be set as default. Exception:{}'.format(self.max_data_per_ts, e))
                    target_ts_list = np.arange(target_ts, target_ts_end + 1)
                for target_ts in target_ts_list:

                    if data_index % 1000 == 0:
                        print('data_index:{}/{} ({:.1f}%), target_ts:{}'.format(data_index, self.data_size_reserved, 100 * data_index / self.data_size_reserved, target_ts))

                    # past ts data
                    def create_each_data_id(work_ts_data, target_ts):
                        return  work_ts_data.iloc[target_ts]['TargetDateTime']

                    if data_id_set is not None:
                        self.set_data_ins(data_id_set, data_index, create_each_data_id(work_ts_data, target_ts))

                    def create_each_input_data(work_ts_data, target_ts, input_ts_width, each_input_output_ts_offset,
                                               input_data_names, input_data_names_to_be_extended,
                        multi_resolution_channels, decrease_resolution_ratio_list, add_offset_column):

                        # read from cache if exists
                        cache = None
                        if self.use_cache and not witout_check_cache:
                            cache = TSData(name='work_input_data', db_host=self.cache_db_host, refresh=False, dtype=pd.DataFrame)
                            key = list_to_hash(
                                arg_list=[work_ts_data, target_ts, input_ts_width, each_input_output_ts_offset,
                                                   input_data_names, input_data_names_to_be_extended,
                            multi_resolution_channels, decrease_resolution_ratio_list, add_offset_column])

                            _input_data = cache.get(key)
                            try:
                                # check cache
                                print('len(_input_data):{}'.format(len(_input_data)))
                                _check = (len(_input_data) == input_ts_width)
                                # print('Checking cache len(_input_data) with key:{}, _check:{}'.format(key, _check))
                                assert _check
                                print('Get cache with key:{}'.format(key))
                                return _input_data
                            except Exception as e:
                                # print('No cache with key:{} with Exception:{}'.format(key, e))
                                None

                        input_filterd = work_ts_data.iloc[(target_ts - input_ts_width):target_ts]

                        try:
                            assert (len(input_filterd) == input_ts_width)
                        except AssertionError:
                            print('AssertionError len(input_filterd):{}'.format(len(input_filterd)))
                            exit(1)

                        _input_data = input_filterd[input_data_names]

                        # extend data with multi_resolution_channels
                        if multi_resolution_channels > 0:
                            for extend_level in range(1, multi_resolution_channels + 1):
                                # if only_debug_at_first_in_roop: print('extend_level:{}'.format(extend_level))
                                _current_decrease_resolution_ratio = decrease_resolution_ratio_list[extend_level - 1]

                                _input_ts_width = (input_ts_width - 1) * _current_decrease_resolution_ratio
                                _input_ts_index = range(target_ts - _input_ts_width - 1, target_ts,
                                                       _current_decrease_resolution_ratio)
                                try:
                                    assert len(_input_ts_index) == input_ts_width
                                except AssertionError:
                                    print('AssertionError len(_input_ts_index):{}'.format(len(_input_ts_index)))
                                    exit(1)
                                _extend_data = work_ts_data.iloc[_input_ts_index][input_data_names_to_be_extended]
                                _extend_data.columns = [TSDataSet.generate_extendex_data_name(data_name, extend_level)
                                                        for data_name in input_data_names_to_be_extended]
                                _input_data = np.concatenate([_input_data, _extend_data.values.tolist()], axis=1)

                        if add_offset_column:
                            # print('each_input_output_ts_offset:{}'.format(each_input_output_ts_offset))
                            _input_data = np.concatenate(
                                [_input_data, float(each_input_output_ts_offset) * np.ones((input_ts_width, 1))],
                                axis=1)

                        # set cache
                        if self.use_cache:
                            cache.set(key, _input_data)
                            print('Set cache with key:{}'.format(key))

                        return _input_data

                    # set input data
                    _input_data = create_each_input_data(work_ts_data, target_ts, self.input_ts_width, _input_output_ts_offset,
                                           self.input_data_names, self.input_data_names_to_be_extended,
                                           self.multi_resolution_channels, self.decrease_resolution_ratio_list, self.add_offset_column)

                    if only_debug_at_first_in_roop:
                        print('_input_data.shape:{}'.format(_input_data.shape))
                        print('_input_data:{}'.format(_input_data[:5]))

                    # input_data.set(data_index, _input_data)
                    self.set_data_ins(input_data, data_index, _input_data)

                    # set output data
                    def create_each_output_data(work_ts_data, target_ts, each_input_output_ts_offset, output_data_names, label_in_use, label_v, model_type, prediction_mode):
                        each_output_data = None

                        if 'label' in output_data_names:
                            if not prediction_mode:
                                if model_type == 'REGRESSION':
                                    each_output_data = label_in_use
                                else:
                                    each_output_data = label_v
                        else:
                            if not prediction_mode:
                                if model_type == 'REGRESSION':
                                    each_output_data = work_ts_data.iloc[(target_ts + each_input_output_ts_offset - 1)][self.output_data_names]
                                else:
                                    each_output_data = None
                                    raise Exception('classification for predict output values is not available.')

                        return each_output_data

                    each_output_data = create_each_output_data(work_ts_data, target_ts, _input_output_ts_offset, self.output_data_names, label_in_use, label_v, self.model_type, self.prediction_mode)
                    # output_data.set(data_index, each_output_data)
                    self.set_data_ins(output_data, data_index, each_output_data)


                    if label_v is not None: print('label_v shape:{}'.format(label_v.shape))
                    # print('output_data shape:{}'.format(output_data.shape()))

                    # annotation_data
                    if self.annotation_col_names is not None:

                        def create_each_annotation_data(work_ts_data, each_output_data, target_ts, annotation_col_names, prediction_mode):
                            annotation_data_filterd_from_input = work_ts_data.iloc[target_ts - 1][
                                annotation_col_names]
                            annotation_data_filterd_from_output_dt = work_ts_data.iloc[target_ts]['TargetDateTime']

                            each_annotation_data = np.empty((1 + 1 + len(annotation_col_names)), dtype=object)
                            each_annotation_data[0] = annotation_data_filterd_from_output_dt
                            if not prediction_mode: each_annotation_data[1] = each_output_data
                            each_annotation_data[2:] = annotation_data_filterd_from_input

                            # if only_debug_at_first_in_roop:
                            #     print('annotation_data_filterd_from_input ==========')
                            #     print(annotation_data_filterd_from_input)
                            #     print('annotation_data of data_index ==========')
                            #     print(annotation_data.get(data_index))

                            return each_annotation_data

                        each_annotation_data = create_each_annotation_data(work_ts_data, output_data.get(data_index), target_ts, self.annotation_col_names, self.prediction_mode)

                        # annotation_data.set(data_index, each_annotation_data)
                        self.set_data_ins(annotation_data, data_index, each_annotation_data)

                    if self.prediction_mode:
                        self.test_index_list.append(data_index)
                    else:
                        try:
                            # skip invalid data
                            if only_debug_at_first_in_roop: print('self.skip_invalid_data:{}, self.valid_data_range:{}'.format(self.skip_invalid_data, self.valid_data_range))
                            if self.skip_invalid_data is None or (self.skip_invalid_data is not None and self.is_valid_data(output_data.get(data_index))):
                                if self.is_test_data(test_flag, target_ts - size_to_complement_before + _input_output_ts_offset - 1):
                                    if only_debug_at_first_in_roop: print('add to test data. target_ts:{}, size_to_complement_before:{}, _input_output_ts_offset:{}'.format(target_ts, size_to_complement_before, _input_output_ts_offset))
                                    self.test_index_list.append(data_index)
                                else:
                                    if only_debug_at_first_in_roop: print('add to train data. target_ts:{}, _input_output_ts_offset:{}'.format(target_ts, _input_output_ts_offset))
                                    self.train_index_list.append(data_index)
                            else:
                                if only_debug_at_first_in_roop: print('skip output_data.get(data_index):{}'.format(output_data.get(data_index)))

                        except Exception:
                            _do_nothing = 0
                            if only_debug_at_first_in_roop: print('do not add to any data. train data. target_ts:{}, _input_output_ts_offset:{}'.format(target_ts,
                                                                                                       _input_output_ts_offset))

                    # TODO if (target_ts >= self.prediction_ts_start) and (target_ts <= self.prediction_ts_end):
                    # TODO     self.prediction_index_list.append(data_index)

                    data_index += 1
                    target_ts += 1

                    only_debug_at_first_in_roop = False

            generate_ts_time += (time.time() - reset_time)

        # wait for thread finish
        while len(self.thread_dict) > 0:
            print('waiting for thread finish with len:{}'.format(len(self.thread_dict)))
            time.sleep(1)

        # if self.annotation_col_names is not None:
        #     annotation_data = annotation_data[:data_index]

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

        self.set_data(input_data, output_data, annotation_data, data_id_set)
        print('DONE generate_input_output_data with input_data.get_size():{}'.format(input_data.get_size()))

        return input_data, output_data


    @staticmethod
    def get_timedelta_with_dt_unit(dt_unit, time_step):
        if dt_unit.lower() in ['minute', 'minutes', 'm', 'relative_m'] or dt_unit.lower().find('relative_minute') >= 0:
            return timedelta(minutes=(time_step))
        elif dt_unit.lower() in ['hour', 'hours', 'h', 'relative_h'] or dt_unit.lower().find('relative_hour') >= 0:
            return timedelta(hours=(time_step))
        elif dt_unit.lower() in ['millisecond', 'milliseconds', 'mls', 'relative_mls'] or dt_unit.lower().find('relative_millisecond') >= 0:
            return timedelta(milliseconds=(time_step))
        # default dt unit
        return timedelta(days=(time_step))

    @staticmethod
    def get_seconds_with_dt_unit(dt_unit, time_step):
        if dt_unit.lower() in ['minute', 'minutes', 'm', 'relative_m'] or dt_unit.lower().find('relative_minute') >= 0:
            return time_step * 60 # sec
        elif dt_unit.lower() in ['hour', 'hours', 'h', 'relative_h'] or dt_unit.lower().find('relative_hour') >= 0:
            return time_step * 60 * 60 # min, sec
        elif dt_unit.lower() in ['millisecond', 'milliseconds', 'mls', 'relative_mls'] or dt_unit.lower().find('relative_millisecond') >= 0:
            return time_step * 1e-3
        # default dt unit
        return time_step * 24 * 60 * 60 # hour, min, sec
