import os
import numpy as np
import random
import time
from datetime import datetime as dt
import matplotlib.pyplot as plt
from PIL import Image
import math
from pathlib import Path, PurePath
from urllib import request
import urllib.error
import pandas as pd
from ggutils import get_module_logger
import http.client
from multiprocessing import Process, Manager
import argparse

logger = get_module_logger()

CODE_TRAIN = 0  # data_set_def code for train data
CODE_TEST = 1  # data_set_def code for test data

IMAGENET_WORDS_URL = 'http://image-net.org/archive/words.txt'

ROOT_DIR_PATH = '/var/data/'
WORK_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'work')
DIST_DATA_DIR = os.path.join(ROOT_DIR_PATH, 'imagenet1000-image')

DEFAULT_HTTP_REQUEST_TIMEOUT = 60 # time out after spent 1 minute

'''
1. Downloads Imagenet word from
http://image-net.org/archive/words.txt
to WORK_DIR_PATH

2. Download to data files as:

DIST_DATA_DIR
├── [ID_0]  // images that is labeled 0(name: 'tench') with ID_0: 'n01440764'
|   ├── image_file_status.csv // csv file with columns 0:url, 1:status
|   ├── imagenet_url_list.csv // csv file with columns 0:url(ImageNet url)
|   ├── [image file] // image files
...
|
...
├── [ID_999]  // images that is labeled 0(name: 'toilet tissue') with ID_0: 'n15075141'	
├── data_set_def  // data set definition directory
    └── train_imagenet1000_classification.csv // data set definition file for training and testing
'''

STATUS_INIT = 0
STATUS_DOWNLOADED = 1
STATUS_ANAVAILABLE = 2

STATUS_FILE_NAME = 'image_file_status.csv'
IMAGENET_URL_LIST_FILE_NAME = 'imagenet_url_list.csv'

CLASS_FILE_PATH = 'imagenet_1000_class.csv'


def is_url(url):
    try:
        return url.split('://')[0].find('http') >= 0
    except Exception:
        return False


class ImageNetDownloadStatus():
    """Class to manage how the ImageNet download status is 
    self.df: DataFrame that contains the ImageNet download status for an ID.
        columns=['url', 'status']
            url: URL of files to be downloaded
            status: STATUS_INIT=0, STATUS_DOWNLOADED=1, STATUS_ANAVAILABLE=2
    """

    def __init__(self, id):
        self.id = id
        self.total_download_size = 0  # int size in byte
        dir_path = os.path.join(DIST_DATA_DIR, id)
        self.status_file_path = os.path.join(dir_path, STATUS_FILE_NAME)
        os.makedirs(dir_path, exist_ok=True)
        if os.path.isfile(self.status_file_path):
            self.df = pd.read_csv(self.status_file_path)
        else:
            df_src_array = [['https://www.geek-guild.jp/hoge.jpg', STATUS_ANAVAILABLE]]
            self.df = pd.DataFrame(df_src_array, columns=['url', 'status'])

    def update_downloaded(self, all_image_url_list):

        all_image_url_list = [image_url for image_url in all_image_url_list if not image_url is None]
        all_image_url_list = [x.replace('\r', '') for x in all_image_url_list]

        if not all_image_url_list is None:
            downloaded_file_list = [x.name for x in
                                    list(os.scandir('/var/data/imagenet1000-image/{}/'.format(self.id))) if
                                    (x.name.find(STATUS_FILE_NAME) < 0) and (
                                            x.name.find(IMAGENET_URL_LIST_FILE_NAME) < 0)]
            downloaded_url_list = [ImagenetDataSetConverter.file_name_to_image_url(file_name) for file_name in
                                   downloaded_file_list]
            downloaded_url_list = [url for url in downloaded_url_list if not url is None]

            all_image_url_list.extend(downloaded_url_list)
            all_image_url_list = list(set(all_image_url_list))

            df_src_array = np.asarray([all_image_url_list, [STATUS_INIT] * len(all_image_url_list)]).T
            self.df = pd.DataFrame(df_src_array, columns=['url', 'status'])
            self.df['status'] = [STATUS_DOWNLOADED if url in downloaded_url_list else STATUS_INIT for url in
                                 self.df['url']]
            self.save()

    def save(self):
        self.df.to_csv(self.status_file_path, index=False)

    def get_downloaded_file_cnt(self):
        downloaded_file_cnt = (self.df['status'] == STATUS_DOWNLOADED).sum()
        return downloaded_file_cnt

    def summary(self):
        downloaded_file_cnt = self.get_downloaded_file_cnt()
        all_file_cnt = len(self.df)
        logger.info('summary of status with id: {}, downloaded_file_cnt: {}, all_file_cnt: {}'.format(self.id, downloaded_file_cnt, all_file_cnt))

    def download(self, file_cnt_to_download=2, retry_cnt=10, http_request_timeout=None):

        self.summary()
        downloaded_file_cnt = self.get_downloaded_file_cnt()
        logger.info('downloading from ImageNet with id: {}, downloaded_file_cnt: {}, file_cnt_to_download:{}'.format(self.id, downloaded_file_cnt, file_cnt_to_download))
        while (downloaded_file_cnt < file_cnt_to_download) and (retry_cnt > 0):
            # choose URL to download
            df_to_download = self.df[self.df['status'] == STATUS_INIT]
            if df_to_download.empty:
                logger.info('No file to download with id:{}'.format(self.id))
                return
            _index = int(random.random() * len(df_to_download))
            df_to_download = df_to_download.iloc[_index]
            image_url = df_to_download['url']
            try:
                image = ImagenetDataSetConverter.download_image(image_url, http_request_timeout=http_request_timeout)
                _file_name = ImagenetDataSetConverter.image_url_to_file_name(image_url)
                path = os.path.join(DIST_DATA_DIR, self.id)
                path = os.path.join(path, _file_name)
                logger.info('path to write_image:{}'.format(path))
                ImagenetDataSetConverter.write_image(path, image)
                # check image by getsize
                file_size = os.path.getsize(path)
                self.total_download_size += file_size

                self.df[self.df['url'] == image_url]['status'] = STATUS_DOWNLOADED
                self.save()
                downloaded_file_cnt += 1
                logger.info('Done download with id:{}, image_url:{}'.format(self.id, image_url))
            except (urllib.error.URLError, http.client.InvalidURL) as e:
                logger.info(
                    'Failed to download, and change status to STATUS_ANAVAILABLE with id:{}, url:{}, e:{}'.format(
                        self.id, image_url, e))
                df_to_download['status'] = STATUS_ANAVAILABLE
                self.df[self.df['url'] == image_url]['status'] = STATUS_ANAVAILABLE
                self.save()
                retry_cnt -= 1

            except (KeyError, ValueError, urllib.error.HTTPError,
                    urllib.error.URLError, http.client.IncompleteRead) as e:
                logger.info('Failed to download with id:{}, url:{}, e:{}'.format(self.id, image_url, e))
                retry_cnt -= 1
            except (ConnectionResetError, TimeoutError, ConnectionResetError) as e:
                logger.info('Failed to download with id:{}, url:{}, e:{}. sleep and retry another image'.format(
                    self.id, image_url, e))
                time.sleep(3)
                retry_cnt -= 1

            logger.info(
                'while downloading from ImageNet with id: {}, downloaded_file_cnt: {}, file_cnt_to_download:{}'.format(
                    self.id, downloaded_file_cnt, file_cnt_to_download))

        logger.info(
            'finished download from ImageNet with id: {} with downloaded_file_cnt:{}, retry_cnt: {}'.format(self.id,
                                                                                                            downloaded_file_cnt,
                                                                                                            retry_cnt))


class ImagenetDataSetConverter:

    def __init__(self, max_threads=1, image_size_per_class=None, http_request_timeout=None):
        self.has_to_update_imagenet_url_list = True
        self.max_threads = max_threads or 1
        self.image_size_per_class = image_size_per_class or 10
        self.http_request_timeout = http_request_timeout or DEFAULT_HTTP_REQUEST_TIMEOUT

    @staticmethod
    def image_url_to_file_name(url):
        if not is_url(url): return None
        # replace ://
        # file_name = url.split('//')[1]
        file_name = url.replace('//', '#-+SS+-#')
        # replace /
        # file_name = file_name.replace('.', '-')
        file_name = file_name.replace('/', '#-+S+-#')
        return file_name

    @staticmethod
    def file_name_to_image_url(file_name):
        # replace //
        image_url = file_name.replace('#-+SS+-#', '//')
        # replace /
        image_url = image_url.replace('#-+S+-#', '/')
        return image_url

    def check_image_file_status(self, id_to_check):

        status = ImageNetDownloadStatus(id_to_check)

        status.update_downloaded(self.get_image_url_list_by_imagenet_id(id_to_check))

        logger.info('len of status:{}'.format(len(status.df)))

        status.save()
        logger.info('id_to_check:{}, len of status:{}'.format(id_to_check, len(status.df)))

        return status

    def get_image_url_list_by_imagenet_id(self, id):

        dir_path = os.path.join(DIST_DATA_DIR, id)
        os.makedirs(dir_path, exist_ok=True)
        imagenet_url_list_file_path = os.path.join(dir_path, IMAGENET_URL_LIST_FILE_NAME)

        _has_to_update_imagenet_url_list = self.has_to_update_imagenet_url_list
        if not self.has_to_update_imagenet_url_list:
            # check url list that already exists
            # os.makedirs(dir_path, exist_ok=True)
            if not os.path.isfile(imagenet_url_list_file_path):
                # has to update because list does not exist
                _has_to_update_imagenet_url_list = True
            else:
                with open(imagenet_url_list_file_path) as f:
                    image_url_list = f.read().splitlines()

        if _has_to_update_imagenet_url_list:
            new_url_list = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}".format(id)
            response = request.urlopen(new_url_list, timeout=self.http_request_timeout)
            body = response.read().decode('utf8')
            image_url_list = body.split('\n')
            with open(imagenet_url_list_file_path, 'w') as f:
                for item in image_url_list:
                    f.write("%s\n" % item)

        # for image_url in image_url_list:
        #     logger.debug('id:{}, image_url:{}'.format(id, image_url))
        return image_url_list

    def download_imagenet_words(self, decode=True):

        # all word list
        self.all_word_to_id_dict = {}
        response = request.urlopen(IMAGENET_WORDS_URL, timeout=self.http_request_timeout)
        body = response.read()
        if decode == True:
            body = body.decode('utf8')
        # convert to dict
        logger.debug(body)
        id_word_list = body.split('\n')
        for row in id_word_list:
            logger.debug(row)
            id, words = row.split('\t')
            words = words.replace(', ', ',')
            for word in words.split(','):
                self.all_word_to_id_dict[word] = id

        self.all_word_list = self.all_word_to_id_dict.keys()

        # check the name is in Imagenet1000 class name

        self.word_to_class_num_dict = {}
        self.class_num_to_id_dict = {}
        # Imagenet1000 word dict
        df_imagenet1000_class = pd.read_csv(CLASS_FILE_PATH)
        for index, row in df_imagenet1000_class.iterrows():
            class_num = row['class_num']
            words = row['class_name']
            words = words.replace(', ', ',')
            words = words.replace('"', '')
            for word in words.split(','):
                self.word_to_class_num_dict[word] = class_num
                # check that word is linked to imagenet id
                if word in self.all_word_list:
                    self.class_num_to_id_dict[class_num] = self.all_word_to_id_dict[word]

    @staticmethod
    def download_image(url, decode=False, http_request_timeout=None):
        UNAVAILABLE_MESSAGE = 'unavailable'
        logger.info('TODO download_image with url:{}'.format(url))
        response = request.urlopen(url, timeout=http_request_timeout)
        if response.geturl().find(UNAVAILABLE_MESSAGE) >= 0:
            raise KeyError('unavailable image url:{}'.format(url))

        body = response.read()
        if decode == True:
            body = body.decode()
        return body

    @staticmethod
    def write_image(path, image):
        file = open(path, 'wb')
        file.write(image)
        file.close()

    def summary_threads(self, last_proc_name=None):
        logger.info('#+#+# summary_threads #+#+#')
        logger.info('#+#+# last_proc_name: {} #+#+#'.format(last_proc_name))
        logger.info('#+#+# thread_id list: {} #+#+#'.format(self.thread_dict.keys()))
        logger.info('#+#+#+#+#+#+#+#+#+#+#+#+#+#')

    def download_image_by_multithreads(self, i, id):

        thread_id = '{}_{}'.format(i, id)
        self.thread_dict[thread_id] = thread_id
        self.summary_threads(last_proc_name='start thread_id:{}'.format(thread_id))
        self.thread_serial_num  = self.thread_serial_num + 1
        if self.thread_serial_num % 1000 == 0:
            logger.info(
                'Summarize multiprocessing tread with i: {}, id: {}, len of thread_dict:{}, thread_serial_num:{}'.format(
                    i, id, len(self.thread_dict), self.thread_serial_num))

        self.download_image_by_singlethread(i, id)
        self.thread_dict.pop(thread_id, None)
        self.summary_threads(last_proc_name='finished thread_id:{}'.format(thread_id))

    def download_image_by_singlethread(self, i, id):
        start_for_id_time = time.time()
        finished_ratio = 100.0 * float(i) / float(self.id_size)
        df_status = self.check_image_file_status(id)
        logger.info(
            '########## i: {} / {}, dt: {}, processing with id:{}'.format(
                i, self.id_size, dt.now(), id))

        # download a file
        logger.info('----------')
        df_status.download(file_cnt_to_download=self.image_size_per_class, http_request_timeout=self.http_request_timeout)
        lap_time = time.time()
        spent_hours = (lap_time - self.start_time) / 3600.0
        download_speed_mbps = 1e-6 * float(8 * df_status.total_download_size / (lap_time - start_for_id_time))
        logger.info('image_size_per_class: {}, finished {:.1f} %, spent_hours: {}, download_speed_mbps: {:.3f}'.format(
            self.image_size_per_class, finished_ratio, spent_hours, download_speed_mbps))

    def download_imagenet1000(self, shuffle_id=True):

        self.download_imagenet_words()

        # debug
        for k, v in self.class_num_to_id_dict.items():
            logger.debug('class_num:{}, id:{}'.format(k, v))

        self.id_list = list(self.class_num_to_id_dict.values())
        if shuffle_id:
            random.shuffle(self.id_list)
        self.id_size = len(self.id_list)
        self.start_time = time.time()

        # download images by singlethread or multithread
        self.max_threads = self.max_threads or 1
        if self.max_threads > 1:
            self.thread_dict = Manager().dict()
            self.thread_serial_num = 0

        for i, id in enumerate(self.id_list):

            if self.max_threads <= 1:
                # single thread processing
                self.download_image_by_singlethread(i, id)
            else:
                # multi thread processing
                thread_wait_time = 0.01
                if len(self.thread_dict) >= self.max_threads:
                    self.summary_threads('multiprocessing has to be wait for len(self.thread_dict) < self.max_threads')

                while len(self.thread_dict) >= self.max_threads:
                    message = 'multiprocessing waiting for thread_wait_time: {}, tread with i: {}, id: {} starts. thread:{}'.format(
                        thread_wait_time, i, id, len(self.thread_dict))
                    logger.info(message)
                    time.sleep(thread_wait_time)
                    thread_wait_time = min(self.http_request_timeout, thread_wait_time * 2.0)

                p = Process(target=self.download_image_by_multithreads, args=(i, id,), )
                p.start()

        if self.max_threads > 1:
            logger.info('check for all threads finished')
            logger.info('existing threads: {}'.format(len(self.thread_dict)))
            thread_wait_time = 0.01
            while len(self.thread_dict) > 0:
                self.summary_threads()
                logger.info('waiting for all threads finished for thread_wait_time: {}'.format(thread_wait_time))
                time.sleep(thread_wait_time)
                thread_wait_time = min(self.http_request_timeout, thread_wait_time * 2.0)

    def get_imagenet_id_from_file_path(self, file_path):
        imagenet_id = None
        if file_path is None: return imagenet_id
        try:
            imagenet_id = int(file_path.split('/')[-1])
        except Exception as e:
            logger.info(e)
            imagenet_id = None
        return imagenet_id

    def get_class_num_from_imagenet_id(self, imagenet_id):
        class_num = None
        if imagenet_id is None: return class_num
        try:
            class_num = self.imagenet_id_to_class_num_dict(imagenet_id)
        except Exception as e:
            logger.info(e)
            class_num = None
        return class_num

    def get_data_set_def_path(self):
        file_name = 'train_imagenet1000_classification.csv'
        data_set_def_path = os.path.join(DIST_DATA_DIR, 'data_set_def')
        data_set_def_path = os.path.join(data_set_def_path, file_name)
        return data_set_def_path

    def get_small_data_set_def_path(self):
        file_name = 'train_imagenet1000_classification_small.csv'
        small_data_set_def_path = os.path.join(DIST_DATA_DIR, 'data_set_def')
        small_data_set_def_path = os.path.join(small_data_set_def_path, file_name)
        return small_data_set_def_path

    def export_data_set_def(self, df_src_array, test_ratio=0.1):
        df_data_set_def = pd.DataFrame(df_src_array, columns=['data_set_id', 'label', 'sub_label', 'test', 'group'])
        df_data_set_def['test'] = [CODE_TEST if random.random() < test_ratio else CODE_TRAIN for x in
                                   df_data_set_def['data_set_id']]

        # export full data set
        _file_path = self.get_data_set_def_path()
        os.makedirs(str(Path(_file_path).parent), exist_ok=True)
        df_data_set_def.to_csv(_file_path, index=False)

        # export small data set (train data 1000, test data 1000*test_ratio)
        small_data_iloc = list(range(len(df_data_set_def)))
        random.shuffle(small_data_iloc)
        small_data_iloc = small_data_iloc[:1000]
        df_small_data_set_def = df_data_set_def.iloc[small_data_iloc]

        _file_path = self.get_small_data_set_def_path()
        os.makedirs(str(Path(_file_path).parent), exist_ok=True)
        df_small_data_set_def.to_csv(_file_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tsp')

    parser.add_argument('--image_size_per_class', '-ispc', type=int, default=10,
                        help='Integer, image size per class (Default: 10)')
    parser.add_argument('--max_threads', '-mxthr', type=int, default=1,
                        help='Integer, max threads (Default: 1, singlethread)')
    parser.add_argument('--http_request_timeout', '-hrto', type=int, default=60,
                        help='Integer, http request timeout (Default: 60 sec)')

    args = parser.parse_args()
    print('args:{}'.format(args))

    converter = ImagenetDataSetConverter(image_size_per_class=args.image_size_per_class, max_threads = args.max_threads, http_request_timeout = args.http_request_timeout)
    converter.download_imagenet1000()

    logger.info('Done on :{}'.format(dt.now()))

    # TODO export_data_set_def

    # # prepare data_set_def
    # all_label_list = [converter.get_class_num_from_imagenet_id(file_path) for file_path in all_file_list]
    # all_data_size = len(all_file_list)
    # df_src_array = np.hstack(
    #     [[all_file_list, all_label_list, all_label_list, [CODE_TRAIN] * all_data_size, ['TRAIN'] * all_data_size]]).T
    #
    # # add to data_set_def
    # converter.export_data_set_def(df_src_array)
