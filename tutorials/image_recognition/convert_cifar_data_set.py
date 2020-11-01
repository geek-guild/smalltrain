import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from pathlib import Path, PurePath

import pandas as pd

CODE_TRAIN = 0  # data_set_def code for train data
CODE_TEST = 1  # data_set_def code for test data

CIFAR_10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

ROOT_DIR_PATH = '/var/data/'
WORK_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'work')
SRC_DATA_DIR = os.path.join(ROOT_DIR_PATH, 'cifar-10-batches-py')
DIST_DATA_DIR = os.path.join(ROOT_DIR_PATH, 'cifar-10-image')

'''
1. Downloads cifar10 data from
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
to WORK_DIR_PATH

2. Extracts the tar file results in:

SRC_DATA_DIR
├── batches.meta  
├── data_batch_1  // training data 1 binary file
├── data_batch_2  // training data 2
├── data_batch_3  // training data 3
├── data_batch_4  // training data 4
├── data_batch_5  // training data 5
├── readme.html
└── test_batch    // testing data

3. Convert to data set files as:

DIST_DATA_DIR
├── data_batch_1  // training data 1 extracted to png files
├── data_batch_2  // training data 2
├── data_batch_3  // training data 3
├── data_batch_4  // training data 4
├── data_batch_5  // training data 5
├── data_set_def  // data set definition directory
|   └── train_cifar10_classification.csv // data set definition file for training and testing
└── test_batch    // testing data
'''

def download_cifar_10():
    file_name = CIFAR_10_URL.split('/')[-1]
    file_path_to_download = os.path.join(WORK_DIR_PATH, file_name)
    if not os.path.isfile(file_path_to_download):
        import requests
        downloaded_file = requests.get(CIFAR_10_URL)
        os.makedirs(WORK_DIR_PATH, exist_ok=True)
        open(file_path_to_download, 'wb').write(downloaded_file.content)
        print('Downloaded: {}'.format(file_path_to_download))
    else:
        print('Already exists: {}'.format(file_path_to_download))

    extract_tarfile(file_path_to_download, ROOT_DIR_PATH)
    print('Extracted tarfile: {}'.format(file_path_to_download))


def extract_tarfile(file_path, path_to_extract=None):
    if path_to_extract is None:
        path_to_extract = Path(file_path).parent
    import tarfile
    tar = tarfile.open(file_path)
    tar.extractall(path_to_extract)
    tar.close()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_class_num_from_file_path(file_path):
    class_num = None
    if file_path is None: return class_num
    try:
        class_num = int(file_path.split('.')[-2].split('_c')[1])
    except Exception as e:
        print(e)
        class_num = None
    return class_num

def get_data_set_def_path():
    file_name = 'train_cifar10_classification.csv'
    data_set_def_path = os.path.join(DIST_DATA_DIR, 'data_set_def')
    data_set_def_path = os.path.join(data_set_def_path, file_name)
    return data_set_def_path

def get_small_data_set_def_path():
    file_name = 'train_cifar10_classification_small.csv'
    small_data_set_def_path = os.path.join(DIST_DATA_DIR, 'data_set_def')
    small_data_set_def_path = os.path.join(small_data_set_def_path, file_name)
    return small_data_set_def_path

def export_data_set_def(df_src_array):
    df_data_set_def = pd.DataFrame(df_src_array, columns=['data_set_id', 'label', 'sub_label', 'test', 'group'])
    df_data_set_def['test'] = [CODE_TRAIN if x.find('test_batch') < 0 else CODE_TEST for x in
                               df_data_set_def['data_set_id']]

    # export full data set
    _file_path = get_data_set_def_path()
    os.makedirs(str(Path(_file_path).parent), exist_ok=True)
    df_data_set_def.to_csv(_file_path, index=False)

    # export small data set
    _file_path = get_small_data_set_def_path()
    os.makedirs(str(Path(_file_path).parent), exist_ok=True)
    # train data for data_batch_1_i9* (1111 files) and test data for test_batch_i9* (1111 files)
    df_small_data_set_def = df_data_set_def.loc[(df_data_set_def['data_set_id'].str.contains('data_batch_1_i9'))
                                                | df_data_set_def['data_set_id'].str.contains('test_batch_i9')]
    df_small_data_set_def.to_csv(_file_path, index=False)


if __name__ == '__main__':

    download_cifar_10()

    data_batch_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

    all_file_list = []

    for data_batch in data_batch_list:

        dir_path = os.path.join(DIST_DATA_DIR, data_batch)
        os.makedirs(dir_path, exist_ok=True)

        image_data_set = unpickle(os.path.join(SRC_DATA_DIR, data_batch))

        labels = np.array(image_data_set[b'labels'])
        images = image_data_set[b'data'].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('uint8')

        for i, image_array in enumerate(images):
            image = Image.fromarray(image_array)
            label = labels[i]
            file_name = '{}_i{}_c{}.png'.format(data_batch, i, label)
            file_path = os.path.join(dir_path, file_name)
            image.save(file_path)
            all_file_list.append(file_path)

            # assertion
            if i == 0:
                read_image_array = np.asarray(Image.open(file_path))
                assert read_image_array.shape == image_array.shape
                assert math.fabs(float(read_image_array[0][0][0] - image_array[0][0][0]) < 1e-4)
                assert math.fabs(float(read_image_array[-1][-1][-1] - image_array[-1][-1][-1]) < 1e-4)
                read_class_num = get_class_num_from_file_path(file_path)
                print('read_class_num:{}, label:{}'.format(read_class_num, label))
                assert read_class_num == label


    # prepare data_set_def
    all_label_list = [get_class_num_from_file_path(file_path) for file_path in all_file_list]
    all_data_size = len(all_file_list)
    df_src_array = np.hstack([[all_file_list, all_label_list, all_label_list, [CODE_TRAIN] * all_data_size, ['TRAIN']* all_data_size]]).T


    # add to data_set_def
    export_data_set_def(df_src_array)


