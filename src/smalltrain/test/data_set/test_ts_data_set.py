# usage
# pytest -v test/data_set/test_ts_data_set.py -k "test_get_seconds_with_dt_unit"
# pytest -v test/data_set/test_ts_data_set.py -k "test_get_timedelta_with_dt_unit"


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


from smalltrain.data_set.ts_data_set import TSDataSet


import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('TestTSDataSet')


class TestTSDataSet:

    def test_get_seconds_with_dt_unit(self):

        dt_unit = 'minute'
        time_step = 1
        dt_unit_per_sec = TSDataSet.get_seconds_with_dt_unit(dt_unit=dt_unit, time_step=time_step)

        log.info('dt_unit:{}, time_step:{}, dt_unit_per_sec:{}'.format(dt_unit, time_step, dt_unit_per_sec))
        assert dt_unit_per_sec == 60

    def test_get_timedelta_with_dt_unit(self):
        from datetime import timedelta

        dt_unit = 'minute'
        time_step = 1
        td = TSDataSet.get_timedelta_with_dt_unit(dt_unit=dt_unit, time_step=time_step)
        log.info('dt_unit:{}, time_step:{}, td:{}'.format(dt_unit, time_step, td))
        assert td == timedelta(minutes=time_step)

        dt_unit = 'hour'
        time_step = 1
        td = TSDataSet.get_timedelta_with_dt_unit(dt_unit=dt_unit, time_step=time_step)
        log.info('dt_unit:{}, time_step:{}, td:{}'.format(dt_unit, time_step, td))
        assert td == timedelta(hours=time_step)

