from dateutil.parser import parse as parse_datetime
from datetime import timezone
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from datetime import datetime
import time
import pandas as pd
import numpy as np
import math
import random
import csv
import os
import sys

from multiprocessing import Process, Manager

from ggutils.gg_data_base import GGDataBase
from ggutils.gg_hash import GGHash

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

class GGData:

    def __init__(self, name, use_db='GGDataBase', db_host='localhost', refresh=False, dtype=np.ndarray):
        PREFIX = '[TSData]'
        self.key_delimiter = '/'
        if use_db is None: use_db = 'Default'
        self.use_db = use_db
        self.dtype = dtype
        self.name = name
        self.group_key = 'g{}{}'.format(self.key_delimiter, name)
        if use_db == 'GGDataBase':
            self._db = GGDataBase.Instance()
            self._db.set_db(setting_file_path='/usr/local/etc/vendor/gg/redis_connection_setting.json', debug_mode=False)
            if refresh:
                print('refresh with delete group_key:{}'.format(self.group_key))
                keys = self.get_keys()
                if keys is not None:
                    for key in keys:
                        print('refresh with delete key:{}'.format(key))
                        self._db.delete(key)
                self._db.delete(self.group_key)

        elif use_db == 'Default':
            # Default simple k-v dictionary
            self._db = {}
        else:
            raise Exception('Invalid self.use_db:{}'.format(self.use_db))



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
        return GGData(name, use_db, db_host, refresh, dtype)


    def get(self, key=None):
        _iterable, key_or_list = self.is_key_iterable(key)
        if key is None or (_iterable and len(key_or_list) == 0):
            raise Exception('Invalid usage of get with empty key:{}. Use get_all_values.'.format(key))
        else:
            _iterable, key_or_list = self.is_key_iterable(key)
            if self.use_db in ['GGDataBase']:
                if _iterable:
                    return np.asarray([self.cast_dtype(self._db.read(self.create_key_with_name(k))) for k in key_or_list])
                else:
                    return self.cast_dtype(self._db.read(self.create_key_with_name(key)))
            elif self.use_db == 'Default':
                if _iterable:
                    return np.asarray([self._db[self.create_key_with_name(k)] for k in key_or_list])
                else:
                    return np.asarray(self._db[self.create_key_with_name(key)])
            raise Exception('Invalid self.use_db:{}'.format(self.use_db))

    def cast_dtype(self, value):
        if isinstance(self.dtype, np.ndarray) and not isinstance(value, np.ndarray): return np.asarray(value)
        if isinstance(self.dtype, pd.DataFrame) and not isinstance(value, pd.DataFrame):
            if isinstance(value, np.ndarray): return pd.DataFrame(value)
            raise ValueError('invalid dtype:{}'.format(self.dtype))
        # do nothing
        return value


    def set(self, key, value):
        if self.use_db in ['GGDataBase']:
            key_with_name = self.create_key_with_name(key)
            self._db.update(key_with_name, value)
            #  also set key to group_key
            keys = self._db.read(self.group_key)
            if keys is None:
                keys = [key_with_name]
            else:
                keys.append(key_with_name)
            self._db.update(self.group_key, keys)
            return
        elif self.use_db == 'Default':
            self._db[self.create_key_with_name(key)] = value
            return

        raise Exception('Invalid self.use_db:{}'.format(self.use_db))

    def get_keys(self):
        if self.use_db in ['GGDataBase']:
            if self.group_key is not None:
                # keys = self._db.read_range(self.group_key)
                keys = self._db.read(self.group_key)
                return keys
            raise ValueError('self.group_key is None')
            # try to read with name
            pattern = '{}*'.format(self.name)
            print('try to read with pattern:{}'.format(pattern))
            keys = self._db.keys(pattern=pattern)
            print('read keys:{} with pattern:{}'.format(keys, pattern))
            return keys

        elif self.use_db == 'Default':
            keys = [k for k in self._db.keys() if self.name in k]
            return keys
        raise Exception('Invalid self.use_db:{}'.format(self.use_db))

    def get_all_values(self):
        all_key_with_name = self.get_keys()
        if self.use_db in ['GGDataBase']:
            return np.asarray([self.cast_dtype(self._db.read(k)) for k in all_key_with_name])
        elif self.use_db == 'Default':
            return np.asarray([self._db[k] for k in all_key_with_name])
        raise Exception('Invalid self.use_db:{}'.format(self.use_db))

    def get_size(self):
        if self.use_db in ['Default', 'GGDataBase']:
            keys = self.get_keys()
            return len(keys)
        raise Exception('Invalid self.use_db:{}'.format(self.use_db))

    def shape(self, index=None):

        if index is None:
            # get all shape
            ret_size = [self.get_size()]
            if self.use_db in ['Default', 'GGDataBase']:
                for key in self.get_keys():
                    ret_size.extend(self.get(key).shape[0:]) # TODO only get first key-value's shape
                    return tuple(ret_size)
            raise Exception('Invalid self.use_db:{}'.format(self.use_db))

        elif index == 0:
            return self.get_size()

        elif index > 1:
            if self.use_db in ['Default', 'GGDataBase']:
                for key in self._db.keys():
                    return self.get(key).shape[index-1] # TODO only get first key-value's shape
            raise Exception('Invalid self.use_db:{}'.format(self.use_db))
        else:
            raise Exception('Invalid index:{}'.format(index))

    def create_key_with_name(self, key):
        _iterable, key_or_list = self.is_key_iterable(key)
        def add_name_to_key_if_not_contain(name, key_delimiter, key):
            if key is None or len(str(key)) < len(name): return '{}{}{}'.format(name, key_delimiter, key)
            return key if str(key).find('{}{}'.format(name, key_delimiter)) == 0 else '{}{}{}'.format(name, key_delimiter, key)

        if _iterable:
            return [add_name_to_key_if_not_contain(self.name, self.key_delimiter, key) for key in key_or_list]
        else:
            return add_name_to_key_if_not_contain(self.name, self.key_delimiter, key)

    def remove_name_from_key_with_name(self, key_with_name):
        _iterable, key_with_name_or_list = self.is_key_iterable(key_with_name)
        def remove_name_from_key_if_contain(name, key_delimiter, key):
            if key is None or len(str(key)) < len(name): return key
            return key[len(name)+len(key_delimiter):] if str(key).find('{}{}'.format(name, key_delimiter)) == 0 else key

        if _iterable:
            # removed_key = key_with_name_or_list
            # for i in range(len(removed_key)):
            #     if removed_key[i].find('{}{}'.format(self.name, self.key_delimiter)) == 0: removed_key[i] = removed_key[i][len(self.name)+len(self.key_delimiter):]
            # return removed_key
            return [remove_name_from_key_if_contain(self.name, self.key_delimiter, key_with_name) for key_with_name in key_with_name_or_list]

        else:
            return remove_name_from_key_if_contain(self.name, self.key_delimiter, key_with_name)

    def is_key_iterable(self, key):
        if key is None: return False, key
        if isinstance(key, list): return True, key
        if isinstance(key, np.ndarray) and len(key) > 1: return True, list(key)
        if isinstance(key, range): return True, list(key)
        return False, key



class GGDataSet:

    # Abstract method
    def __init__(self, debug_mode=False,
                 prepare_data_mode=False, prediction_mode=False, hparams=None):

        PREFIX = '[GGDataSet]'
        self.hparams = hparams
        print('{}TODO init with hparams:{}'.format(PREFIX, hparams))

        self.debug_mode = debug_mode
        self.prepare_data_mode = prepare_data_mode
        self.model_type = 'CLASSIFICATION'
        if hparams and 'model_type' in hparams.keys():
            print('{}Use model_type in hparams:{}'.format(PREFIX, hparams['model_type']))
            self.model_type = hparams['model_type']
        else:
            print('{}TODO Use ts_start with default value:{}'.format(PREFIX, self.model_type))

        self.prediction_mode = prediction_mode
        print('{}init with prediction_mode:{}'.format(PREFIX, prediction_mode))

        self.max_threads = 100
        self.thread_dict = Manager().dict()
        self.thread_cnt_dict = Manager().dict()
        self.thread_cnt_dict[0] = 0

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

        # (For compatibility with ver0.1.1 ```col_index_to_mask``` and ver0.1.2 ```ch_index_to_mask``` )

        self.col_index_to_mask = None
        if hparams and 'col_index_to_mask' in hparams.keys():
            print('{}Use col_index_to_mask in hparams:{}'.format(PREFIX, hparams['col_index_to_mask']))
            self.col_index_to_mask = hparams['col_index_to_mask']

        # check both mask_rate and col_index_to_mask
        if self.mask_rate is None or self.col_index_to_mask is None:
            print('{}Set both mask_rate and col_index_to_mask None because one of them is None'.format(PREFIX))
            self.mask_rate = None
            self.col_index_to_mask = None

        # set ch_index_to_mask
        self.ch_index_to_mask = self.col_index_to_mask # (For compatibility with ver0.1.1 ```col_index_to_mask``` and ver0.1.2 ```ch_index_to_mask``` )
        if hparams and 'ch_index_to_mask' in hparams.keys():
            print('{}Use ch_index_to_mask in hparams:{}'.format(PREFIX, hparams['ch_index_to_mask']))
            self.ch_index_to_mask = hparams['ch_index_to_mask']

        # check both mask_rate and ch_index_to_mask
        if self.mask_rate is None or self.ch_index_to_mask is None:
            print('{}Set both mask_rate and ch_index_to_mask None because one of them is None'.format(PREFIX))
            self.mask_rate = None
            self.ch_index_to_mask = None


        # about skipping invalid data

        # about skip_invalid_data
        self.skip_invalid_data = None
        if hparams and 'skip_invalid_data' in hparams.keys():
            print('{}Use skip_invalid_data in hparams:{}'.format(PREFIX, hparams['skip_invalid_data']))
            self.skip_invalid_data = hparams['skip_invalid_data']
        self.skip_invalid_data = (self.skip_invalid_data is not None and self.skip_invalid_data)

        # about skip_invalid_data
        self.valid_data_range = None
        if hparams and 'valid_data_range' in hparams.keys():
            print('{}Use valid_data_range in hparams:{}'.format(PREFIX, hparams['valid_data_range']))
            self.valid_data_range = hparams['valid_data_range']


        # about multi_resolution_channels
        self.multi_resolution_channels = 0
        if hparams and 'multi_resolution_channels' in hparams.keys():
            print('{}Use multi_resolution_channels in hparams:{}'.format(PREFIX, hparams['multi_resolution_channels']))
            self.multi_resolution_channels = hparams['multi_resolution_channels']
        else:
            print('{}TODO Use multi_resolution_channels with default value:{}'.format(PREFIX, self.multi_resolution_channels))

        # set decrease_resolution_ratio or decrease_resolution_ratio_list
        self.decrease_resolution_ratio_list = None
        if self.multi_resolution_channels > 0:
            # 1. decrease_resolution_ratio
            self.decrease_resolution_ratio = DEFAULT_DECREASE_RESOLUTION_RATIO
            if hparams and 'decrease_resolution_ratio' in hparams.keys():
                print('{}Use decrease_resolution_ratio in hparams:{}'.format(PREFIX, hparams['decrease_resolution_ratio']))
                self.decrease_resolution_ratio = hparams['decrease_resolution_ratio']
            else:
                print('{}TODO Use decrease_resolution_ratio with default value:{}'.format(PREFIX, self.decrease_resolution_ratio))

            # 2. decrease_resolution_ratio_list
            if hparams and 'decrease_resolution_ratio_list' in hparams.keys():
                print('{}Use decrease_resolution_ratio_list in hparams:{}'.format(PREFIX, hparams['decrease_resolution_ratio_list']))
                self.decrease_resolution_ratio_list = hparams['decrease_resolution_ratio_list']
            if self.decrease_resolution_ratio_list is None:
                print('{}TODO decrease_resolution_ratio_list is set with decrease_resolution_ratio:{} and multi_resolution_channels:{}'.format(PREFIX, self.decrease_resolution_ratio, self.multi_resolution_channels))
                self.decrease_resolution_ratio_list = [int(math.pow(self.decrease_resolution_ratio, extend_level)) for extend_level in range(1, self.multi_resolution_channels + 1)]
                print('{}DONE decrease_resolution_ratio_list is set {}'.format(PREFIX, self.decrease_resolution_ratio_list))

        if hparams and 'input_data_names' in hparams.keys():
            print('{}Use input_data_names in hparams:{}'.format(PREFIX, hparams['input_data_names']))
            self.input_data_names = hparams['input_data_names']
        else:
            print('{}Error no input_data_names'.format(PREFIX))
            exit(1)

        self.col_size = len(self.input_data_names)

        # about channels to be extended with multi_resolution_channels
        self.input_data_names_to_be_extended = None
        if hparams and 'input_data_names_to_be_extended' in hparams.keys():
            print('{}Use input_data_names_to_be_extended in hparams:{}'.format(PREFIX, hparams['input_data_names_to_be_extended']))
            self.input_data_names_to_be_extended = hparams['input_data_names_to_be_extended']
            if self.input_data_names_to_be_extended and self.input_data_names_to_be_extended is not None:
                self.col_size += len(self.input_data_names_to_be_extended) * self.multi_resolution_channels
        elif self.multi_resolution_channels > 0:
            self.input_data_names_to_be_extended = self.input_data_names
            print('{}Use input_data_names_to_be_extended with all input_data_names:{}'.format(PREFIX, self.input_data_names))
            self.col_size = len(self.input_data_names) *(1 + self.multi_resolution_channels)
        else:
            print('{}No input_data_names_to_be_extended'.format(PREFIX))


        print('self.col_size:{}'.format(self.col_size))


        if hparams and 'output_data_names' in hparams.keys():
            print('{}Use output_data_names in hparams:{}'.format(PREFIX, hparams['output_data_names']))
            self.output_data_names = hparams['output_data_names']
        else:
            print('{}Error no output_data_names'.format(PREFIX))
            exit(1)


        # Whether Has to complement the value before ts starts or not(Default:True)
        self.has_to_complement_before = True
        if hparams and 'has_to_complement_before' in hparams.keys():
            print('{}Use has_to_complement_before in hparams:{}'.format(PREFIX, hparams['has_to_complement_before']))
            self.has_to_complement_before = hparams['has_to_complement_before']
        if self.has_to_complement_before is None:
            self.has_to_complement_before = True
            print('{}Use has_to_complement_before with default value:{}'.format(PREFIX, self.has_to_complement_before))

        # S (For compatibility with ver0.1.1 ```complement_ts``` and ver0.1.2 ```complement_input_data``` )
        self.complement_ts = None
        if hparams and 'complement_ts' in hparams.keys():
            print('{}Use complement_ts in hparams:{}'.format(PREFIX, hparams['complement_ts']))
            self.complement_ts = hparams['complement_ts']
        else:
            print('{}Use complement_ts with default value:{}'.format(PREFIX, self.complement_ts))
        # E (For compatibility with ver0.1.1 ```complement_ts``` and ver0.1.2 ```complement_input_data``` )

        self.complement_input_data = self.complement_ts # (For compatibility with ver0.1.1 ```complement_ts``` and ver0.1.2 ```complement_input_data``` )
        if hparams and 'complement_input_data' in hparams.keys():
            print('{}Use complement_input_data in hparams:{}'.format(PREFIX, hparams['complement_input_data']))
            self.complement_input_data = hparams['complement_input_data']
        else:
            print('{}Use complement_input_data with default value:{}'.format(PREFIX, self.complement_input_data))

        # S (For compatibility with ver0.1.1 ```complement_ts``` and ver0.1.2 ```complement_input_data``` )
        if self.complement_input_data is None:
            self.complement_input_data = self.complement_ts
        # E (For compatibility with ver0.1.1 ```complement_ts``` and ver0.1.2 ```complement_input_data``` )

        self.data_dir_path = '/var/data/'
        if hparams and 'data_dir_path' in hparams.keys():
            print('{}Use data_dir_path in hparams:{}'.format(PREFIX, hparams['data_dir_path']))
            self.data_dir_path = hparams['data_dir_path']
        else:
            print('{}Use data_dir_path with default value:{}'.format(PREFIX, self.data_dir_path))

        self.data_set_def_path = None
        if hparams and 'data_set_def_path' in hparams.keys():
            print('{}Use data_set_def_path in hparams:{}'.format(PREFIX, hparams['data_set_def_path']))
            self.data_set_def_path = hparams['data_set_def_path']
        else:
            print('{}Error no data_set_def_path'.format(PREFIX))
            exit(1)
        # read df_data_set_def and check
        self.df_data_set_def = pd.read_csv(self.data_set_def_path)
        # _debug = self.df_data_set_def['data_set_id']

        self.annotation_col_names = None
        if hparams and 'annotation_col_names' in hparams.keys():
            print('{}Use annotation_col_names in hparams:{}'.format(PREFIX, hparams['annotation_col_names']))
            self.annotation_col_names = hparams['annotation_col_names']

        self.target_group = None
        if hparams and 'target_group' in hparams.keys():
            print('{}Use target_group in hparams:{}'.format(PREFIX, hparams['target_group']))
            self.target_group = hparams['target_group']

        self.use_sub_label = False
        if hparams and 'use_sub_label' in hparams.keys():
            print('{}Use use_sub_label in hparams:{}'.format(PREFIX, hparams['use_sub_label']))
            self.target_group = hparams['use_sub_label']

        self.test_only_mode = False
        if hparams and 'test_only_mode' in hparams.keys():
            print('{}Use test_only_mode in hparams:{}'.format(PREFIX, hparams['test_only_mode']))
            self.test_only_mode = hparams['test_only_mode']
        else:
            print('{}TODO Use test_only_mode with default value:{}'.format(PREFIX, self.test_only_mode))

        # Set output_classes if given from init model. see #51
        self.output_classes = None
        if hparams and 'output_classes' in hparams.keys():
            print('{}Use output_classes in hparams:{}'.format(PREFIX, hparams['output_classes']))
            self.output_classes = hparams['output_classes']
        # if self.output_classes is still None,  then will be set from data set.

        # use_cache
        self.use_cache = False
        if hparams and 'use_cache' in hparams.keys():
            print('{}Use use_cache in hparams:{}'.format(PREFIX, hparams['use_cache']))
            self.use_cache = hparams['use_cache']
        if self.use_cache is None:
            self.use_cache = False
            print('{}Use use_cache with default value:{}'.format(PREFIX, self.use_cache))

        # cache_db_host
        self.cache_db_host = 'localhost'
        if hparams and 'cache_db_host' in hparams.keys():
            print('{}Use cache_db_host in hparams:{}'.format(PREFIX, hparams['cache_db_host']))
            self.cache_db_host = hparams['cache_db_host']
        if self.cache_db_host is None:
            self.cache_db_host = 'localhost'
            print('{}Use cache_db_host with default value:{}'.format(PREFIX, self.cache_db_host))

        # data set id for cache
        self.cache_data_set_id = None
        if hparams and 'cache_data_set_id' in hparams.keys():
            print('{}Use cache_data_set_id in hparams:{}'.format(PREFIX, hparams['cache_data_set_id']))
            self.cache_data_set_id = hparams['cache_data_set_id']
        if self.cache_data_set_id is None:
            self.cache_data_set_id = hparams['train_id']
            print('{}Use cache_data_set_id with default value:{}'.format(PREFIX, self.cache_data_set_id))
        self.refresh_cache_data_set = False
        if hparams and 'refresh_cache_data_set' in hparams.keys():
            print('{}Use refresh_cache_data_set in hparams:{}'.format(PREFIX, hparams['refresh_cache_data_set']))
            self.refresh_cache_data_set = hparams['refresh_cache_data_set']
        if self.refresh_cache_data_set is None:
            self.refresh_cache_data_set = False
            print('{}Use refresh_cache_data_set with default value:{}'.format(PREFIX, self.refresh_cache_data_set))

        print('{}DONE init process defined in GGDataSet'.format(PREFIX))

        return

        raise NotImplementedError()


    def get_label_in_use(self, labels): # labels = (label, sub_label)
        return labels[1] if self.use_sub_label else labels[0]


    def set_data(self, input_data, output_data, annotation_data=None, data_id_set=None):
        try:
            assert (input_data.get_size() > 0)
            assert (output_data.get_size() > 0)
        except AssertionError as e:
            print(e)
            # print('[set_data]input_data:{}'.format(input_data[:5]))
            # print('[set_data]output_data:{}'.format(output_data[:5]))
            raise e
        self.input_data = input_data
        self.output_data = output_data
        self.data_size = input_data.get_size()
        self.data_id_set = data_id_set
        self.annotation_data = annotation_data
        self.masked_input_data = None
        print('self.mask_rate:{}'.format(self.mask_rate))
        if self.mask_rate is not None and self.mask_rate > 0:
            print('generate masked_input_data')
            self.masked_input_data = self.mask_data(self.input_data)
            if self.debug_mode: print(self.masked_input_data)
        if self.debug_mode:
            print('[set_data]output_data:{}'.format(self.output_data.get(range(5))))
            print('[set_data]data_id_set:{}'.format(None if data_id_set is None else self.data_id_set.get(range(5))))
            print('[set_data]annotation_data:{}'.format(None if annotation_data is None else self.annotation_data.get(range(5))))

    def mask_data(self, input_data):
        ret_data = self.construct_data_ins(name='masked_input_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=True)
        keys = input_data.get_keys()
        keys = input_data.remove_name_from_key_with_name(keys)
        debug_first = True
        for key in keys:
            value = input_data.get(key)
            if debug_first: print('mask_data with key:{}, value:{}'.format(key, value))
            if self.ch_index_to_mask is not None:
                value[:, self.ch_index_to_mask] = np.ones(
                    [value.shape[0], len(self.ch_index_to_mask)],
                    dtype=np.float32) * 9.999  # TODO
            # ret_data.set(key, value)
            self.set_data_ins(ret_data, key, value)

            debug_first = False

        return ret_data

    def next_batch(self, batch_size):
        '''
        Warning: This function is deplecated. Use next_train_batch or next_test_batch explicitly.
        :param batch_size:
        :return:
        '''
        return self.next_train_batch(batch_size)


    def next_train_batch(self, batch_size):
        train_index_size = len(self.train_index_list)

        def _generate_train_batch_index_list():
            _list = list(range(len(self.train_index_list)))
            random.shuffle(_list)
            # Repeat list in order to select the sub list having batch_size elements
            self.train_batch_index_list = np.array(_list * (2 + int(batch_size / len(self.train_index_list))), dtype=int)
            # print('self.train_batch_index_list:{}'.format(list(self.train_batch_index_list)))
            # print('self.train_index_list:{}'.format(list(self.train_index_list)))

        try:
            if self.train_batch_index_list is None:
                _generate_train_batch_index_list()
        except AttributeError:
            _generate_train_batch_index_list()

        # print('len(self.train_batch_index_list):{}'.format(len(self.train_batch_index_list)))

        if self.mask_rate is not None and self.mask_rate > np.random.rand():
            if self.debug_mode: print('Use masked_input_data')
            input_data = self.masked_input_data
        else:
            input_data = self.input_data

        try:
            next_index = int(self.train_batch_index_from) + batch_size
        except AttributeError:
            self.train_batch_index_from = 0
            next_index = batch_size

        # debug
        # print('self.train_batch_index_from:{}, next_index:{}, train_index_size:{}'.format(self.train_batch_index_from, next_index, train_index_size))
        # print('len(self.train_index_list):', len(self.train_index_list))
        # print('len(self.train_batch_index_list):', len(self.train_batch_index_list))
        # print('self.train_batch_index_list[self.train_batch_index_from:next_index]:', self.train_batch_index_list[self.train_batch_index_from:next_index])
        # print('self.train_index_list[self.train_batch_index_list[self.train_batch_index_from:next_index]]:', self.train_index_list[self.train_batch_index_list[self.train_batch_index_from:next_index]])

        _train_index_list = self.train_index_list[self.train_batch_index_list[self.train_batch_index_from:next_index]]
        ret_input_data = input_data.get(_train_index_list)
        ret_output_data = self.output_data.get(_train_index_list)
        if self.model_type == 'REGRESSION': ret_output_data = ret_output_data.reshape(-1)

        # print('len of self.train_batch_index_list[self.train_batch_index_from:next_index]:{}'.format(len(self.train_batch_index_list[self.train_batch_index_from:next_index])))
        # print('self.train_batch_index_list[self.train_batch_index_from:next_index]:{}'.format(self.train_batch_index_list[self.train_batch_index_from:next_index]))

        self.train_batch_index_from = next_index
        if self.train_batch_index_from >= train_index_size:
            self.train_batch_index_from -= train_index_size

        # print('len(ret_input_data):', len(ret_input_data))
        assert len(ret_input_data) == batch_size
        assert len(ret_output_data) == batch_size

        return ret_input_data, ret_output_data

    def next_test_batch(self, batch_size):
        test_index_size = len(self.test_index_list)

        def _generate_test_batch_index_list():
            _list = list(range(len(self.test_index_list)))
            # Repeat list in order to select the sub list having batch_size elements
            self.test_batch_index_list = np.array(_list * (2 + int(batch_size / len(self.test_index_list))), dtype=int)
            # print('self.test_batch_index_list:{}'.format(list(self.test_batch_index_list)))
            # print('self.test_index_list:{}'.format(list(self.test_index_list)))

        try:
            if self.test_batch_index_list is None:
                _generate_test_batch_index_list()
        except AttributeError:
            _generate_test_batch_index_list()

        try:
            next_index = int(self.test_batch_index_from) + batch_size
        except AttributeError:
            self.test_batch_index_from = 0
            next_index = batch_size

        # debug
        print('self.test_batch_index_from:{}, next_index:{}, test_index_size:{}'.format(self.test_batch_index_from, next_index, test_index_size))
        _test_index_list = self.test_index_list[self.test_batch_index_list[self.test_batch_index_from:next_index]]
        ret_input_data = self.input_data.get(_test_index_list)
        ret_output_data = self.output_data.get(_test_index_list)
        if self.model_type == 'REGRESSION': ret_output_data = ret_output_data.reshape(-1)

        self.test_batch_index_from = next_index
        if self.test_batch_index_from >= test_index_size:
            self.test_batch_index_from -= test_index_size

        print('len(ret_input_data):', len(ret_input_data))
        assert len(ret_input_data) == batch_size
        assert len(ret_output_data) == batch_size

        return ret_input_data, ret_output_data

    # get train data set if already generated file exist or generate
    def get_data_set_file_list(self, dir_path):
        data_set_file_list = []
        _data_set_file_src = self.df_data_set_def['data_set_id'].values.tolist()
        try:
            for file in GGDataSet.find_all_files(dir_path):
                for _file_str in _data_set_file_src:
                    if _file_str in file:
                        data_set_file_list.append(file)

        except FileNotFoundError:
            return None
        return data_set_file_list

    # get train data set if already generated file exist or generate
    def get_data_set_file_dict(self, dir_path):
        data_set_file_dict = {}
        _data_set_file_src = self.df_data_set_def['data_set_id'].values.tolist()
        try:
            for file in GGDataSet.find_all_files(dir_path):
                # data_file_postfix = 'filled.csv'
                # if (file[-len(data_file_postfix):] == data_file_postfix):
                for _file_str in _data_set_file_src:
                    if _file_str in file:
                        # print('k:{}, v:{}'.format(_file_str, file))
                        data_set_file_dict[_file_str] = file

        except FileNotFoundError:
            return None
        return data_set_file_dict

    def generate_train_data_set(self):

        return None # TODO


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

        input_data_names = self.input_data_names
        print('input_data_names:{}'.format(input_data_names))


        if self.refresh_cache_data_set:
            self.generate_cache_data()
        else:
            if not self.read_cache_data():
                print('have to generate_cache_data')
                self.generate_cache_data()

        return self.input_data, self.output_data


    def read_cache_data(self):
        print('TODO read_cache_data')
        self.input_data = self.construct_data_ins(name='input_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=False)
        self.output_data = self.construct_data_ins(name='output_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=False)
        self.data_id_set = self.construct_data_ins(name='data_id_set_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=False)
        self.annotation_data = self.construct_data_ins(name='annotation_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=False)
        self.masked_input_data = self.construct_data_ins(name='masked_input_data_{}'.format(self.cache_data_set_id), db_host=self.cache_db_host, refresh=False)

        # cache test_index_list and train_index_list
        forecass_cache = GGHash(name='forecass_cache_{}'.format(self.cache_data_set_id))
        self.test_index_list = forecass_cache.get('test_index_list')
        self.train_index_list = forecass_cache.get('train_index_list')

        for name, data_ins_to_check in self.get_data_ins_dict_to_check_cached().items():
            try:
                print('data_ins_to_check:{}, checking data_ins_to_check.get_size():{} > 0'.format(name, data_ins_to_check.get_size()))
                assert data_ins_to_check.get_size() > 0
            except Exception as e:
                print('read_cache_data failed with error:{} with instance:{}'.format(e, name))
                return False

        for name, cache_ins_to_check in {'test_index_list': self.test_index_list, 'train_index_list': self.train_index_list}.items():
            try:
                print('cache_ins_to_check:{}, checking len(cache_ins_to_check):{} > 0'.format(name, len(cache_ins_to_check)))
                assert len(cache_ins_to_check) > 0
            except Exception as e:
                print('read_cache_data failed with error:{} with instance:{}'.format(e, name))
                return False

        self.data_size = self.input_data.get_size()
        print('has_read with data_size:{}'.format(self.data_size))
        return True


    def get_data_ins_dict_to_check_cached(self):
        return {'input_data': self.input_data, 'output_data': self.output_data, 'annotation_data': self.annotation_data}


    def set_data_ins_mp(self, data_ins, data_index, data_value):
        thread_id = '{}_{}'.format(data_ins.name, data_index)
        self.thread_dict[thread_id] = thread_id
        self.thread_cnt_dict[0] = self.thread_cnt_dict[0] + 1
        if self.thread_cnt_dict[0] % 1000 == 0:
            print(
                'multiprocessing finished tread with data_ins.name:{}, data_index:{} starts. thread_dict:{}, thread_cnt:{}'.format(
                    data_ins.name,
                    data_index,
                    len(
                        self.thread_dict), self.thread_cnt_dict[0]))

        data_ins.set(data_index, data_value)
        self.thread_dict.pop(thread_id, None)


    def set_data_ins(self, data_ins, data_index, data_value):
        self.max_threads = self.max_threads or 1

        # single thread processing
        if self.max_threads <= 1:
            return data_ins.set(data_index, data_value)

        # multi thread processing
        thread_wait_time = 0.01
        while len(self.thread_dict) >= self.max_threads:
            print('multiprocessing waiting for tread with data_ins.name:{}, data_index:{} starts. thread:{}'.format(data_ins.name, data_index, len(self.thread_dict)))
            time.sleep(thread_wait_time)
            thread_wait_time = min(1.0, thread_wait_time * 2.0)

        p = Process(target=self.set_data_ins_mp, args=(data_ins, data_index, data_value,), )
        p.start()

    def is_test_data(self, test_flag, data_index=None):
        '''

        :param test_flag:
          0: train data
          1: test data
        :param data_index: Not in use for GGDataSet class
        :return: True or False
        '''
        return (test_flag > 0)

    # Abstract method
    def generate_cache_data(self):
        raise NotImplementedError()

    def is_valid_data(self, data):
        try:
            return (self.valid_data_range[0] <= data) and (self.valid_data_range[1] >= data)
        except:
            # if we can not detect the validity in range, return True (pass the data)
            return True

    def get_input_data(self, index_list, dtype=np.float32):
        '''
        get input_data with index_list
        :param index_list:
        :return:
        '''
        if index_list is None: return self.input_data.astype(dtype)
        # return self.input_data[index_list].astype(dtype)
        return self.input_data.get(index_list).astype(dtype)

    def get_output_data(self, index_list, dtype=np.float32):
        '''
        get get_output_data with index_list
        :param index_list:
        :return:
        '''
        if index_list is None: return self.output_data.astype(dtype)
        return self.output_data.get(index_list).astype(dtype)

    def get_train_input_data(self, dtype=np.float32):
        return self.get_input_data(index_list=self.train_index_list, dtype=dtype)

    def get_train_output_data(self, dtype=np.float32):
        return self.get_output_data(index_list=self.train_index_list, dtype=dtype)

    def get_test_input_data(self, dtype=np.float32):
        return self.get_input_data(index_list=self.test_index_list, dtype=dtype)

    def get_test_output_data(self, dtype=np.float32):
        return self.get_output_data(index_list=self.test_index_list, dtype=dtype)

    def get_masked_test_input_data(self, dtype=np.float32):
        return self.masked_input_data.get(self.test_index_list).astype(dtype)

    def get_test_data_id_set(self, dtype=None):
        return self.data_id_set.get(self.test_index_list)

    def get_test_annotation_data(self, dtype=None):
        return self.annotation_data.get(self.test_index_list)

    def get_train_input_data_shape(self):
        return self.input_data.get(self.train_index_list).shape

    # def export_data(self, input_data, output_data, index=0, report_dir_path=None, postfix=''):
    def export_data(self, data_kind, index=0, report_dir_path=None, postfix=''):
        if data_kind == 'train_data':
            input_data = self.get_train_input_data()
            output_data = self.get_train_output_data()
        elif data_kind == 'test_data':
            input_data = self.get_test_input_data()
            output_data = self.get_test_output_data()
        else:
            KeyError('Invalid data_kind:{}'.format(data_kind))

        if index == -1:
            index = len(input_data) - 1
        range_info = 'i{}'.format(index)

        if postfix is None or postfix == '':
            postfix = data_kind

        save_file_name = 'export_input_data-{}##POSTFIX##.csv'.format(range_info)
        _report_path = report_dir_path + '/' + save_file_name
        _report_path = _report_path.replace('//', '/')
        _report_path = _report_path.replace('##POSTFIX##', '_{}'.format(postfix))

        np.savetxt(_report_path, input_data[index], delimiter=',')
        # if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)

        save_file_name = 'export_output_data##POSTFIX##.csv'.format(range_info)
        _report_path = report_dir_path + '/' + save_file_name
        _report_path = _report_path.replace('//', '/')
        _report_path = _report_path.replace('##POSTFIX##', '_{}'.format(postfix))

        # deug
        # print('##### S output_data #####')
        # print(output_data[:3])
        # print('##### E output_data #####')

        np.savetxt(_report_path, output_data, delimiter=',')
        # if self.cloud_root: upload_to_cloud(_report_path, self.cloud_root, self.save_root_dir)

    @staticmethod
    def generate_extendex_data_name(data_name, extend_level, format='{}_ex_{}'):
        return format.format(data_name, extend_level)

    @staticmethod
    def find_all_files(directory):
        for root, dirs, files in os.walk(directory):
            yield root
            for file in files:
                yield os.path.join(root, file)


