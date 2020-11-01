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
import argparse
import glob

import ggutils.s3_access as s3_access



def get_report_dir_path(save_root_dir, train_id):
    report_dir_path = os.path.join(save_root_dir, 'report/')
    report_dir_path = os.path.join(report_dir_path, train_id)
    print(save_root_dir, train_id, report_dir_path)
    return report_dir_path

def generate_prediction_data(report_dir_path, target_datetime_str, target_datetime_col_name, target_value_col_name, second_key_col_name, prediction_dt_col_name, prediction_file_date_format='%Y-%m-%d', denormalize_value_ratio=None):
    target_dt = parse_datetime(target_datetime_str)
    target_datetime_str = target_dt.strftime(prediction_file_date_format)

    MIN_OFFSET = 3
    MAX_OFFSET = 10
    offset_to_use = None
    prediction_file_path_list = None
    for _offset in list(range(MIN_OFFSET, MAX_OFFSET+1)):
        prediction_file_name = 'prediction_o{}.0_*.csv'.format(_offset)
        prediction_file_path = os.path.join(report_dir_path, prediction_file_name)
        print('check prediction file for a path:'.format(prediction_file_path))
        prediction_file_path_list = glob.glob(prediction_file_path)
        if prediction_file_path_list:
            print('check prediction file for a path:{}'.format(prediction_file_path))
            offset_to_use = _offset
            break
    if offset_to_use is None:
        raise ValueError('No prediction file in report_dir_path:{}'.format(report_dir_path))
    else:
        print('Use offset:{} to get prediction file'.format(offset_to_use))
    # print('prediction_file_path_list:{}'.format(prediction_file_path_list))

    prediction_value_col_name = 'Estimated'

    prediction_gatherd_second_key = []
    for prediction_file_path in prediction_file_path_list:
        # skip if not prediction_file_path exists
        if not os.path.isfile(prediction_file_path):
            print('Skip to read non-exist file. prediction_file_path:{}'.format(prediction_file_path))
        df_prediction = pd.read_csv(prediction_file_path)
        target_datetime_series = df_prediction[df_prediction[prediction_dt_col_name] == target_datetime_str]
        if len(target_datetime_series) == 1:
            prediction_value = target_datetime_series[prediction_value_col_name].values.tolist()[0]
            second_key_value = target_datetime_series[second_key_col_name].values.tolist()[0]
            print('target_datetime_str:{}, prediction_value:{}, second_key_value:{}'.format(target_datetime_str, prediction_value, second_key_value))
            prediction_gatherd_second_key.append([target_datetime_str, prediction_value, second_key_value])

    df_prediction_gatherd_second_key = pd.DataFrame(prediction_gatherd_second_key, columns=[target_datetime_col_name, target_value_col_name, second_key_col_name])
    df_selected_with_second_key = df_prediction_gatherd_second_key.loc[:, [second_key_col_name, target_value_col_name]]
    df_selected_with_second_key = df_selected_with_second_key.sort_values(by=[second_key_col_name], ascending=True)

    # denormalize target value
    if denormalize_value_ratio is not None:
        df_selected_with_second_key[target_value_col_name] = [int(p * denormalize_value_ratio) for p in df_selected_with_second_key[target_value_col_name]]
    return df_selected_with_second_key

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='generate_prediction_data')
    parser.add_argument('--train_id', '-tid', type=str,
                        help='Train id')
    parser.add_argument('--save_root_dir', '-rd', type=str, default='/var/tensorflow/tsp/',
                        help='Root dir for Tensorflow FileWriter')
    parser.add_argument('--target_datetime', '-tdt', type=str,
                        help='String, target date time to generate prediction data')
    parser.add_argument('--s3_bucket_name', '-s3bn', type=str, default=None,
                        help='String, s3_bucket_name')
    parser.add_argument('--data_file_pattern', '-dfp', type=str, default='prediction_data_{}.csv',
                        help='String Date to generate prediction data')
    parser.add_argument('--data_file_columns', '-dfc', type=str, default=None,
                        help='String, splitted with comma, data_file_columns')
    parser.add_argument('--target_datetime_col_name', '-tdtcn', type=str,
                        help='String, target_datetime_col_name')
    parser.add_argument('--target_value_col_name', '-tvcl', type=str,
                        help='String, target_value_col_name')
    parser.add_argument('--second_key_col_name', '-skcl', type=str,
                        help='String, second_key_col_name')
    parser.add_argument('--prediction_dt_col_name', '-pdcn', type=str, default='DateTime',
                        help='String, prediction_dt_col_name')
    parser.add_argument('--denormalize_value_ratio', '-dnvr', type=float,
                        help='Float, denormalize_value_ratio (The target values will be multiplied by this value in order to be denormalized)')


    args = parser.parse_args()
    print('args:{}'.format(args))

    report_dir_path = get_report_dir_path(args.save_root_dir, args.train_id)

    df_selected_with_second_key = generate_prediction_data(report_dir_path, args.target_datetime, args.target_datetime_col_name, args.target_value_col_name, args.second_key_col_name,
                                                           prediction_dt_col_name = args.prediction_dt_col_name,
                                                           denormalize_value_ratio=args.denormalize_value_ratio)
    print(df_selected_with_second_key)
    prediction_data_file_name = args.data_file_pattern.format(args.target_datetime)
    # TODO os.makedirs(os.path.abspath(os.path.join(prediction_data_file_name, '..')), exist_ok=True)

    # TODO
    if args.data_file_columns is not None:
        df_selected_with_second_key.columns = args.data_file_columns.split(',')
    df_selected_with_second_key.to_csv(os.path.join(report_dir_path, prediction_data_file_name), index=False)

    if (args.s3_bucket_name is not None):
        print('Upload prediction_data_file_name:{} to s3_bucket_name:{}'.format(prediction_data_file_name, args.s3_bucket_name))
        s3_access.upload(s3_bucket_name=args.s3_bucket_name, s3_key=prediction_data_file_name, local_dir=report_dir_path, file_path=prediction_data_file_name)
