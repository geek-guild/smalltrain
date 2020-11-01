# usage
# pytest -v test/data_set/test_ts_data.py -k "test_ts_data_read"
# pytest -v test/data_set/test_ts_data.py -k "test_ts_data_write"


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
import glob
from ggutils.gg_data_base import GGDataBase
from ggutils import data_processor as dp


from smalltrain.data_set.ts_data_set import TSData


import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('TestTSData')


class TestTSData:

    def test_ts_data_read(self):
        log.info('test_ts_data_read')
        test_data = TSData(name='test_data')
        keys = test_data.get_keys()
        log.info('keys:{}'.format(keys))

    def test_ts_data_write(self):

        test_data = TSData(name='test_data', refresh=True)
        key = 0
        value = [0.1, 0.2]
        test_data.set(key, value)

        key = 1
        value = [1.1, 1.2]
        test_data.set(key, value)

        key = 2
        value = [2.1, 2.2]
        test_data.set(key, value)

        log.info('test_data.get_size:{}'.format(test_data.get_size()))
        assert test_data.get_size() == 3
        log.info('test_data.shape(0):{}'.format(test_data.shape(0)))
        # TODO log.info(test_data.shape(1))
        log.info('test_data.get_keys:{}'.format(test_data.get_keys()))

        log.info('test_data.get(range(3)):{}'.format(test_data.get(range(3))))
        log.info('test_data.get(range(1, 3))):{}'.format(test_data.get(range(1, 3))))
        log.info('test_data.get([0, 2])):{}'.format(test_data.get([0, 2])))
        log.info('test_data.get(np.asarray([0, 2])):{}'.format(test_data.get(np.asarray([0, 2]))))
