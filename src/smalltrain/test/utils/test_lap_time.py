import logging

# pytest -v --cov=utils
# pytest -v --cov=utils --cov-report=html
# pytest -v test/utils/test_lap_time.py -k "test_all"

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('TestUser')

from smalltrain.utils import lap_time
import time
import math

class TestHashArray:

    def test_all(self):
        log.info('---------- test_all ----------')
        self.test_start()
        self.test_finish()
        self.test_lap_time()

    def test_start(self):
        log.info('---------- test_start ----------')

        proc_name = 'proc_1'
        init_time = time.time()
        ltime = lap_time.LapTime()
        ltime.start(proc_name)
        start_time_proc_1 = ltime.start_time(proc_name)
        time.sleep(1)
        proc_name = 'proc_2'
        ltime.start(proc_name)
        start_time_proc_2 = ltime.start_time(proc_name)

        log.info('init_time:{}, start_time_proc_1:{}, start_time_proc_1:{}'.format(init_time, start_time_proc_2, start_time_proc_2))
        assert (start_time_proc_1 - init_time) < 1e-3
        assert (start_time_proc_2 - start_time_proc_1) > 1 - 1e-2
        assert (start_time_proc_2 - start_time_proc_1) < 1 + 1e-2

    def test_finish(self):
        log.info('---------- test_finish ----------')

        proc_name = 'proc_1'
        init_time = time.time()
        ltime = lap_time.LapTime()
        ltime.finish(proc_name)
        finish_time_proc_1 = ltime.finish_time(proc_name)
        time.sleep(1)
        proc_name = 'proc_2'
        ltime.finish(proc_name)
        finish_time_proc_2 = ltime.finish_time(proc_name)

        log.info('init_time:{}, finish_time_proc_1:{}, finish_time_proc_1:{}'.format(init_time, finish_time_proc_2, finish_time_proc_2))
        assert (finish_time_proc_1 - init_time) < 1e-3
        assert (finish_time_proc_2 - finish_time_proc_1) > 1 - 1e-2
        assert (finish_time_proc_2 - finish_time_proc_1) < 1 + 1e-2

    def test_lap_time(self):
        log.info('---------- test_lap_time ----------')

        proc_name = 'proc_1'
        init_time = time.time()
        ltime = lap_time.LapTime()

        ltime.start('proc_1')
        time.sleep(0.1)
        ltime.start('proc_2')
        time.sleep(0.1)
        ltime.finish('proc_1')
        time.sleep(0.1)
        ltime.finish('proc_2')

        lap_time_proc_1 = ltime.lap_time('proc_1')
        lap_time_proc_2 = ltime.lap_time('proc_2')

        log.info('lap_time_proc_1:{}, lap_time_proc_2:{}'.format(lap_time_proc_1, lap_time_proc_2))
        assert math.fabs(lap_time_proc_1 - 0.2) < 1e-2
        assert math.fabs(lap_time_proc_2 - 0.2) < 1e-2
