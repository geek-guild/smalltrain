import logging

# pytest -v --cov=utils
# pytest -v --cov=utils --cov-report=html
# pytest -v test/utils/test_hash_array.py -k "test_all"

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('TestUser')

from smalltrain.utils import hash_array
import numpy as np

class TestHashArray:

    def test_all(self):
        log.info('---------- test_all ----------')
        self.test_float_v_to_bytes()
        self.test_bytes_to_vloat_v()
        self.test_float_v_to_hash()

    def test_float_v_to_bytes(self):
        log.info('---------- test_float_v_to_bytes ----------')
        float_v = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        bytes_buf = hash_array.float_v_to_bytes(float_v)
        log.info('float_v:{}, bytes_buf:{}'.format(float_v, bytes_buf))
        assert bytes_buf == b'\xcd\xcc\x8c?\xcd\xcc\x0c@33S@'

    def test_bytes_to_vloat_v(self):
        log.info('---------- test_bytes_to_vloat_v ----------')
        bytes_buf = b'\xcd\xcc\x8c?\xcd\xcc\x0c@33S@'
        expected_float_v = np.array([1.1, 2.2, 3.3])
        actual_float_v = hash_array.bytes_to_vloat_v(bytes_buf, list_size=3)
        log.info('bytes_buf:{}, expected_float_v:{}, actual_float_v:{}'.format(bytes_buf, expected_float_v, actual_float_v))
        for i, x in enumerate(actual_float_v):
            assert x == expected_float_v[i]

    def test_float_v_to_hash(self):
        log.info('---------- test_float_v_to_hash ----------')
        float_v = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        round_dec = 'Not set'
        actual_hash_value = hash_array.float_v_to_hash(float_v)
        expected_hash_value = 'a4dddee1b1b2dcadc2ef8ff241f2a69b453ae5c9ba219f1c0fb67beba77e98aa'
        log.info('case 1. float_v:{}, round_dec:{}'.format(float_v, round_dec))
        log.info('actual_hash_value:{}'.format(actual_hash_value))
        log.info('expected_hash_value:{}'.format(expected_hash_value))
        assert actual_hash_value == expected_hash_value

        float_v = np.array([1.1009, 2.199902, 3.3003], dtype=np.float32)
        round_dec = 2
        actual_hash_value = hash_array.float_v_to_hash(float_v, round_dec)
        expected_hash_value = 'a4dddee1b1b2dcadc2ef8ff241f2a69b453ae5c9ba219f1c0fb67beba77e98aa'
        log.info('case 2. float_v:{}, round_dec:{}'.format(float_v, round_dec))
        log.info('actual_hash_value:{}'.format(actual_hash_value))
        log.info('expected_hash_value:{}'.format(expected_hash_value))
        assert actual_hash_value == expected_hash_value
