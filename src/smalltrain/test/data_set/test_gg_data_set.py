# usage
# pytest -v test/data_set/test_gg_data_set.py -k "test_find_all_files"



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


from smalltrain.data_set.gg_data_set import GGDataSet


import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('TestGGDataSet')


class TestGGDataSet:

    def test_find_all_files(self):

        dir_path = './test/test_data/root_dir/'
        all_files = [x for x in GGDataSet.find_all_files(directory=dir_path)]
        len_all_files = len(all_files)

        log.info('dir_path:{}, len_all_files:{}, all_files:{}'.format(dir_path, len_all_files, all_files))
        assert len_all_files == 5


