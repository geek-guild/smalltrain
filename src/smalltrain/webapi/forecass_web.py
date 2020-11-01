from flask import Flask
from flask import request
from flask import render_template
from flask import send_file
from flask import make_response
from flask import send_from_directory
from flask import jsonify
from flask import json as flask_json
from flask_thumbnails import Thumbnail
from flask_cors import CORS

import os
import json
import numpy as np
import pandas as pd
import sys

# import model.nn_model as model
from smalltrain.model import nn_model
from smalltrain.utils import tf_log_to_csv

from smalltrain.model.user import UserManager, User, UserNotFoundError
import smalltrain.utils.jwt_util as jwt_util
import smalltrain.utils.proc as proc

# app = Flask(__name__)
# set the project root directory as the static folder.
app = Flask(__name__, static_url_path='')

thumb = Thumbnail(app)
CORS(app)

operation_dir_path = '/var/tensorflow/tsp/sample/operation/'
report_dir_path = '/var/tensorflow/tsp/sample/report/'
history_dir_path = '/var/tensorflow/tsp/sample/history/'
logs_dir_path = '/var/tensorflow/tsp/sample/logs/'

ROOT_ACCESS_KEY = '/FSDg7sa8'

import logging

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

from smalltrain.utils.gg_mongo_data_base import GGMongoDataBase
from smalltrain.utils.gg_mongo_data_base import TimeoutError
db = GGMongoDataBase.Instance()
db.set_db(host='localhost')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')


@app.route(ROOT_ACCESS_KEY + '/js/<path:path>')
def send_js(path):
    return send_from_directory('html/js', path)


@app.route('/locales/<path:path>')
def send_locales_json(path):
    return send_from_directory('html/locales', path, mimetype='application/json')


@app.route(ROOT_ACCESS_KEY + '/img/<path:path>')
def send_img(path):
    return send_from_directory('html/img', path)


@app.route(ROOT_ACCESS_KEY + '/css/<path:path>')
def send_css(path):
    return send_from_directory('html/css', path)


@app.route(ROOT_ACCESS_KEY + '/forecass_signin.html')
def send_forecass_signin_html():
    return send_from_directory('html', 'forecass_signin.html')


@app.route(ROOT_ACCESS_KEY + '/forecass_web.html')
def send_html():
    return send_from_directory('html', 'forecass_web.html')


@app.route(ROOT_ACCESS_KEY + '/about/')
def about_forecass():
    return 'FORECASS ver 0.1.0'


@app.route(ROOT_ACCESS_KEY + '/test/')
def test():
    args = request.args
    return 'args:{}'.format(args)


@app.route(ROOT_ACCESS_KEY + '/train/')
def train():
    args = request.args
    return 'args:{}'.format(args)


@app.route(ROOT_ACCESS_KEY + '/signin', methods=['GET', 'POST'])
def signin():
    user_id_or_email = None
    password = None

    if request.method == 'GET':
        user_id_or_email = request.args.get('user_id_or_email', default='', type=str)
        password = request.args.get('password', default='', type=str)
    elif request.method == 'POST':
        user_id_or_email = request.form['user_id_or_email']
        password = request.form['password']

    message = None
    status = None
    user = None

    manager = UserManager(db)
    try:
        user = manager.read(user_id_or_email)
        print('user read:{}'.format(user))
    except UserNotFoundError as e:
        message = str(e)
        status = 404
    except jwt_util.InvalidJWTError as e:
        message = str(e)
        status = 400
    except TimeoutError as e:
        message = str(e)
        status = 500
    print('user:{}, message:{}, status:{}'.format(user, message, status))
    if status is not None:
        return generate_error_output(message, status)

    jwt = user.signin(password)
    print('jwt:{}'.format(jwt))

    if jwt is None:
        message = 'Invalid user id or email or password'
        status = 404
        return generate_error_output(message, status)

    ret_data = {}
    ret_data['jwt'] = jwt

    message = 'sucess'
    return_data = {
        'message': message,
        'data': ret_data
    }
    status = 200
    return generate_output('result.html', return_data, status)


@app.route(ROOT_ACCESS_KEY + '/operation/read', methods=['GET', 'POST'])
def operation_read():

    user, message, status = check_user_from_jwt_header(request)

    if status is not None:
        return generate_error_output(message, status)


    if request.method == 'GET':
        train_id = request.args.get('train_id', default='', type=str)
    elif request.method == 'POST':
        train_id = request.form['train_id']

    operation_json = read_operation_file(train_id)
    if operation_json is None:
        message = 'File not found with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)

    message = 'sucess'
    return_data = {
        'message': message,
        'operation_json': operation_json
    }
    status = 200
    return generate_output('result.html', return_data, status)

def read_operation_file(train_id):
    operation_file_path = os.path.join(operation_dir_path, '{}.json'.format(train_id))

    if not os.path.isfile(operation_file_path):
        return None

    f = open(operation_file_path, 'r')
    operation_json = json.load(f)
    f.close()
    return operation_json


import sys

@app.route(ROOT_ACCESS_KEY + '/report/read', methods=['GET', 'POST'])
def report_read():
    if request.method == 'GET':
        train_id = request.args.get('train_id', default='', type=str)
        output = request.args.get('output', default='json', type=str)
    elif request.method == 'POST':
        train_id = request.form['train_id']
        try:
            output = request.form['output']
        except Exception as e:
            print('Exception:{}'.format(e), file=sys.stderr)
            output = 'json'

    report_sub_dir_path = os.path.join(report_dir_path, '{}'.format(train_id))

    if train_id == '':
        message = 'Not set train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 404
        return generate_output('error.html', return_data, status, output)

    if not os.path.isdir(report_sub_dir_path):
        message = 'Directory not found with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 404
        return generate_output('error.html', return_data, status, output)

    file_tree = generate_file_tree(report_sub_dir_path)

    return_data = {
        'file_tree': file_tree
    }
    status = 200

    return generate_output('report.html', return_data, status, output)

@app.route(ROOT_ACCESS_KEY + '/history/read', methods=['GET', 'POST'])
def history_read():
    if request.method == 'GET':
        train_id = request.args.get('train_id', default='', type=str)
        output = request.args.get('output', default='json', type=str)
    elif request.method == 'POST':
        train_id = request.form['train_id']
        try:
            output = request.form['output']
        except Exception as e:
            print('Exception:{}'.format(e), file=sys.stderr)
            output = 'json'

    history_sub_dir_path = os.path.join(history_dir_path, '{}'.format(train_id))
    logs_sub_dir_path = os.path.join(logs_dir_path, '{}'.format(train_id))

    if train_id == '':
        message = 'Not set train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 404
        return generate_output('error.html', return_data, status, output)

    # update history from log
    tf_log_to_csv.to_csv(logs_sub_dir_path, history_sub_dir_path)

    if not os.path.isdir(history_sub_dir_path):
        message = 'Directory not found with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 404
        return generate_output('error.html', return_data, status, output)

    # 1. all the history files
    history = generate_file_tree(history_sub_dir_path)
    # 2. accuracy csv file
    accuracy_csv_path = os.path.join(history_sub_dir_path, 'csv/precisions_accuracy.csv')
    # 3. accuracy_csv
    accuracy_csv = pd.read_csv(accuracy_csv_path)
    print('len of accuracy_csv:{}'.format(len(accuracy_csv)), file=sys.stderr)

    return_data = {
        'accuracy_csv': accuracy_csv.values
    }
    status = 200

    return generate_output('result.html', return_data, status, output)

@app.route(ROOT_ACCESS_KEY + '/data_set_def/create', methods=['GET', 'POST'])
def data_set_def_create():
    if request.method == 'GET':
        train_id = request.args.get('train_id', default='', type=str)
        output = request.args.get('output', default='json', type=str)
    elif request.method == 'POST':
        train_id = request.form['train_id']
        try:
            output = request.form['output']
        except Exception as e:
            print('Exception:{}'.format(e), file=sys.stderr)
            output = 'json'

    if train_id == '':
        message = 'Not set train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 404
        return generate_output('error.html', return_data, status, output)

    # 1. read operation
    operation_json = read_operation_file(train_id)
    if operation_json is None:
        message = 'File not found with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)

    # 2. check data_dir exists
    if 'data_dir_path' not in operation_json.keys():
        message = 'No data_dir_path with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)
    data_dir_path = operation_json['data_dir_path']

    if not os.path.isdir(operation_json['data_dir_path']):
        message = 'No dir exist with data_dir_path:{}'.format(operation_json['data_dir_path'])
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)

    # 3. data_dir_path_for_labels
    if 'data_dir_path_for_labels' not in operation_json.keys():
        message = 'No data_dir_path_for_labels with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)
    dir_list = operation_json['data_dir_path_for_labels']

    def return_with_invalid_data_dir_path_for_labels(data_dir_path_for_labels):
        message = 'Invalid data_dir_path_for_labels:{}'.format(data_dir_path_for_labels)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)

    if not isinstance(dir_list, list): return_with_invalid_data_dir_path_for_labels(dir_list)
    for dir_path in dir_list:
        if not os.path.isdir(dir_path): return_with_invalid_data_dir_path_for_labels(dir_list)

    # 4. check target_group
    if 'target_group' not in operation_json.keys():
        message = 'No target_group with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)
    target_group = operation_json['target_group']

    # 5. check data_set_def_path
    if 'data_set_def_path' not in operation_json.keys():
        message = 'No data_set_def_path with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)
    data_set_def_path = operation_json['data_set_def_path']

    # 6. check n_train_data_files_for_each_label
    if ('n_train_data_files_for_each_label' not in operation_json.keys()) or ('n_test_data_files_for_each_label' not in operation_json.keys()):
        message = 'No n_train_data_files_for_each_label or n_test_data_files_for_each_label with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)
    n_train_data_files_for_each_label = None
    n_test_data_files_for_each_label = None
    try:
        n_train_data_files_for_each_label = int(operation_json['n_train_data_files_for_each_label'])
        n_test_data_files_for_each_label = int(operation_json['n_test_data_files_for_each_label'])
    except ValueError:
        message = 'Invalid n_train_data_files_for_each_label or n_test_data_files_for_each_label' \
                  'with n_train_data_files_for_each_label:{} or n_test_data_files_for_each_label:{}'.format(
            n_train_data_files_for_each_label, n_test_data_files_for_each_label)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)

    need_data_files_for_each_label = n_train_data_files_for_each_label + n_test_data_files_for_each_label


    # 7. col_names_to_check
    if 'input_data_names' not in operation_json.keys() or 'output_data_names' not in operation_json.keys():
        message = 'No input_data_names or output_data_names with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)

    col_names_list_to_check = operation_json['input_data_names']
    col_names_list_to_check.extend([col_name for col_name in operation_json['output_data_names'] if col_name != 'label'])
    # print('col_names_list_to_check:{}'.format(col_names_list_to_check), file=sys.stderr)

    import linecache

    df_data_set_all = None
    for label, dir_path in enumerate(dir_list):
        # check file_or_dir_list
        cnt = 0
        print('dir_path:{}'.format(dir_path), file=sys.stderr)
        all_files = list(find_all_files(dir_path))
        n_all_files = len(all_files)
        # print('all_files:{}'.format(all_files), file=sys.stderr)
        data_set_for_label = pd.DataFrame(np.vstack(([all_files], np.asarray([[label, label, 0, 'NOT_IN_USE', 0]] * n_all_files).T)).T,
                                          columns=['data_set_id', 'label', 'sub_label', 'test', 'group', 'checked'])
        data_set_for_label = data_set_for_label.astype(dtype={'label': 'int8',
                                        'sub_label': 'int8', 'test': 'int8', 'checked': 'int8'})
        # TODO
        test_col_index = 3
        group_col_index = 4
        checked_col_index = 5

        print('data_set_for_label_head:{}'.format(data_set_for_label.iloc[0:2]), file=sys.stderr)
        index_to_check = np.arange(n_all_files)
        np.random.shuffle(index_to_check)
        df_data_set_for_label = None
        cnt_has_checked = 0
        for index in index_to_check:
            file_to_check = all_files[index]
            # 1. check csv
            try:
                assert file_to_check.split('.')[-1] == 'csv'
                first_data = linecache.getline(file_to_check, 2)
                linecache.clearcache()
                # print('file_to_check:{}, label:{}, dir_path:{} with first_data:{}'.format(file_to_check, label,
                #                                                                                  dir_path, first_data),
                #       file=sys.stderr)
                assert len(first_data) > 0
                header_line = linecache.getline(file_to_check, 1)
                header_list = [v.replace('\n', '').replace('\r', '') for v in header_line.split(',')]
                linecache.clearcache()
                # print('file_to_check:{}, label:{}, dir_path:{} with header_list:{}'.format(file_to_check, label,
                #                                                                                  dir_path, header_list),
                #       file=sys.stderr)
                for col_name_to_check in col_names_list_to_check:
                    # print('header_list:{}, col_name_to_check:{}'.format(header_list, col_name_to_check),
                    #       file=sys.stderr)
                    assert col_name_to_check in header_list
                if cnt % 1 == 0:
                    print('Add csv file_to_check:{}, label:{}, dir_path:{}'.format(file_to_check, label, dir_path), file=sys.stderr)
                # data_series = pd.DataFrame([[file_to_check, label, label, 0, target_group], 1],
                #                            columns=['data_set_id', 'label', 'sub_label', 'test', 'group', 'checked'])
                # if df_data_set_for_label is None:
                #     df_data_set_for_label = data_series
                # else:
                #     df_data_set_for_label = pd.concat([df_data_set_for_label, data_series])
                # stop if enough data has checked
                # if len(df_data_set_for_label) >= need_data_files_for_each_label:
                #     break
                data_set_for_label.iloc[index, checked_col_index] = 1
                cnt_has_checked += 1
                # stop if enough data has checked
                if cnt_has_checked >= need_data_files_for_each_label:
                    break


            except AssertionError as e:
                if cnt % 1 == 0:
                    print('Skip no-csv file_to_check:{}, label:{}, dir_path:{} with error:{}'.format(file_to_check, label, dir_path, e), file=sys.stderr)

            cnt += 1
            # if cnt > 10: exit()

        data_set_for_label.to_csv('/tmp/debug.csv', index=False)

        # select train or test data from checked data files
        # Case not enough data has checked
        if cnt_has_checked < need_data_files_for_each_label:
            # update n_test_data_files_for_each_label with keeping the ratio (n_test_data_files_for_each_label / need_data_files_for_each_label)
            n_test_data_files_for_each_label = int(cnt_has_checked * (n_test_data_files_for_each_label / need_data_files_for_each_label))
        # select test data
        checked_index = data_set_for_label[data_set_for_label['checked'] == 1].index.values
        np.random.shuffle(checked_index)
        print('checked_index:{}'.format(checked_index), file=sys.stderr)
        data_set_for_label.iloc[checked_index[:n_test_data_files_for_each_label], test_col_index] = 1
        data_set_for_label.iloc[checked_index[:n_test_data_files_for_each_label], group_col_index] = target_group

        data_set_for_label.iloc[checked_index[n_test_data_files_for_each_label:min(cnt_has_checked, need_data_files_for_each_label)], group_col_index] = target_group

        # debug
        test_data = data_set_for_label[(data_set_for_label['test'] == 1) & (data_set_for_label['group'] == target_group)].values
        print('n_test_data_files_for_each_label:{}, test_data:{}'.format(n_test_data_files_for_each_label, test_data), file=sys.stderr)
        train_data = data_set_for_label[(data_set_for_label['checked'] == 1) & (data_set_for_label['test'] == 0)
                                        & (data_set_for_label['group'] == target_group)].values
        print('cnt_has_checked:{}, need_data_files_for_each_label:{}, train_data:{}'.format(cnt_has_checked, need_data_files_for_each_label, train_data), file=sys.stderr)

        if df_data_set_all is None:
            df_data_set_all = data_set_for_label
        else:
            df_data_set_all = pd.concat([df_data_set_all, data_set_for_label])


    # remove data_dir from data_set_id
    df_data_set_all['data_set_id'] = [data_set_id.replace(data_dir_path, '') for data_set_id in df_data_set_all['data_set_id'].values]

    df_data_set_all = df_data_set_all.sort_values(['label', 'data_set_id'])
    df_data_set_all.to_csv(data_set_def_path + '_all.csv', index=False)

    df_data_set_def = df_data_set_all[df_data_set_all['checked'] == 1]

    df_data_set_def.to_csv(data_set_def_path, index=False)
    return_data = {
        'df_data_set_def': df_data_set_def.values
    }
    status = 200

    return generate_output('result.html', return_data, status, output)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def generate_output(template, return_data, status, output='json'):
    if output == 'json':
        return generate_json_response(return_data, status)
        # return json.dumps(return_data)
    elif output == 'html':
        return render_template(template, **return_data)
    else:
        return generate_json_response(return_data, status)


def generate_error_output(message, status, output='json'):
    return_data = {
        'message': message
    }
    return generate_output('error.html', return_data, status, output)


def _generate_json_response(return_data, status):
    response = app.response_class(
        response=flask_json.dumps(return_data),
        status=status,
        mimetype='application/json'
    )
    return response


def generate_json_response(return_data, status):
    response = make_response(json.dumps(return_data, cls=JsonEncoder), status)
    response.mimetype = 'application/json'
    return response


def generate_file_tree(path):
    file_tree = dict(name=os.path.basename(path), children=[])
    try:
        lst = os.listdir(path)
    except OSError:
        pass  # ignore errors
    else:
        for name in sorted(lst):
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                file_tree['children'].append(generate_file_tree(fn))
            else:
                # contents = generate_file_download_link(fn)
                # with open(fn, encoding='utf-8') as f:
                #     contents = f.read()
                # file_tree['children'].append(dict(name=name, contents=contents))
                link = generate_file_download_link(fn)
                if name.split('.')[-1] == 'png':
                    # file_tree['children'].append(dict(name=name, link=link, image_path=fn))
                    file_tree['children'].append(dict(name=name, link=link))
                else:
                    file_tree['children'].append(dict(name=name, link=link))

    return file_tree


def generate_file_download_link(path):
    link = ROOT_ACCESS_KEY + '/file/download/?path={}'.format(path)
    filename = os.path.basename(path)
    # href_link = '<a herf=\"{}\">{}</a>'.format(link, filename)
    href_link = '<a herf=\"{}\">{}</a>'.format(link, filename)
    return link


@app.route(ROOT_ACCESS_KEY + '/file/download/')
def file_read():
    file_read_path = request.args.get('path', default='*', type=str)

    if not os.path.isfile(file_read_path):
        message = 'File not found with train_id:{}'.format(file_read_path)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)

    message = 'Can not read file:{}'.format(file_read_path)
    try:
        # from urlparse import urlparse
        filename = os.path.basename(file_read_path)
        # return send_file(file_read_path, filename=filename, attachment_filename=filename)
        response = make_response(send_file(file_read_path))
        response.headers['Content-Disposition'] = \
            'attachment; ' \
            'filename={ascii_filename};'.format(
                ascii_filename=filename
            )
        return response
    except Exception as e:
        message += 'with Exception:{}'.format(e)
        return_data = {
            'message': message
        }
        status = 500
        return generate_output('error.html', return_data, status)


@app.route(ROOT_ACCESS_KEY + '/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        train_id = request.args.get('train_id', default='', type=str)
        output = request.args.get('output', default='json', type=str)
    elif request.method == 'POST':
        train_id = request.form['train_id']
        try:
            output = request.form['output']
        except Exception as e:
            print('Exception:{}'.format(e), file=sys.stderr)
            output = 'json'

    if train_id == '':
        message = 'Not set train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 404
        return generate_output('error.html', return_data, status, output)

    # check trained operation file exests
    operation_file_path = os.path.join(operation_dir_path, '{}.json'.format(train_id))
    if not os.path.isfile(operation_file_path):
        message = 'No operation file exists with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 404
        return generate_output('error.html', return_data, status, output)

    # read trained operation file
    f = open(operation_file_path, 'r')
    operation_json = json.load(f)
    f.close()

    # check and copy as new_train_id
    times = 1
    new_train_id = '{}-PREDICTION-{}'.format(train_id, times)
    prediction_operation_file_path = os.path.join(operation_dir_path, '{}.json'.format(new_train_id))
    while os.path.isfile(prediction_operation_file_path):
        times += 1
        new_train_id = '{}-PREDICTION-{}'.format(train_id, times)
        prediction_operation_file_path = os.path.join(operation_dir_path, '{}.json'.format(new_train_id))

    operation_json['train_id'] = new_train_id
    operation_json['test_only_mode'] = True
    operation_json['parent_train_id'] = train_id
    # write prediction_operation_file_path
    with open(prediction_operation_file_path, 'w') as dumpfile:
        json.dump(operation_json, dumpfile)

    # exec prediction
    try:
        # print('input_data_names:{}'.format(operation_json['input_data_names']), file=sys.stderr)
        from smalltrain.model.operation import Operation
        operation = Operation(operation_json)

        nn_model.main(operation.params)
    except Exception as e:
        message = 'Exception:{}'.format(e)
        print(message, file=sys.stderr)
        return_data = {
            'message': message
        }
        status = 501
        return generate_output('error.html', return_data, status, output)


    return_data = {
        'new_train_id': new_train_id
    }
    status = 200

    return generate_output('report.html', return_data, status, output)


    # TODO refresh report dir
    report_sub_dir_path = os.path.join(report_dir_path, '{}'.format(train_id))
    if os.path.isdir(report_sub_dir_path):
        os.rmdir(report_sub_dir_path)

    return

@app.route(ROOT_ACCESS_KEY + '/proc/get', methods=['GET', 'POST'])
def proc_get():

    user, message, status = check_user_from_jwt_header(request)

    if status is not None:
        return generate_error_output(message, status)


    if request.method == 'GET':
        train_id = request.args.get('train_id', default='', type=str)
    elif request.method == 'POST':
        train_id = request.form['train_id']

    proc_id = proc.get_proc_id_with_train_id(train_id)
    if proc_id is None:
        message = 'Proc not found with train_id:{}'.format(train_id)
        return_data = {
            'message': message
        }
        status = 404
        return generate_output('error.html', return_data, status)
    proc_info = {}
    proc_info['proc_id'] = proc_id

    message = 'sucess'
    return_data = {
        'message': message,
        'proc_info': proc_info
    }
    status = 200
    return generate_output('result.html', return_data, status)

# check jwt Authorization
def check_user_from_jwt_header(request):
    print('/operation/read with request:{}'.format(request))
    print('/operation/read with request.headers:{}'.format(request.headers))

    message = None
    status = None
    user = None

    try:
        header_jwt = request.headers['Authorization']
        print('header_jwt:{}'.format(header_jwt))
    except KeyError as e:
        print('e:{}'.format(e))
        message = 'No JWT in headers'
        status = 400
        return user, message, status

    manager = UserManager(db)
    try:
        user = manager.read_with_jwt(header_jwt)
        print('user:{}'.format(user))
    except UserNotFoundError as e:
        message = str(e)
        status = 404
    except jwt_util.InvalidJWTError as e:
        message = str(e)
        status = 400
    print('user:{}, message:{}, status:{}'.format(user, message, status))

    return user, message, status

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

if __name__ == '__main__':
    # local test
    # app.run(debug=True, host='0.0.0.0', port=5000)
    # publish
    app.run(debug=False, host='0.0.0.0', port=5000)
