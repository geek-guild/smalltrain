import numpy as np
import struct
import hashlib

class HashArray():

    def __init__(self, np_ins=None):
        self.np_ins = np_ins


def float_v_to_bytes(float_v, round_dec=7):
    # float vector to bytes
    # usage
    # float_v = np.array([1.1, 2.2, 3.3], dtype=np.float32)
    # bytes_buf = float_v_to_bytes(float_v)
    # >>> buf
    # b'\xcd\xcc\x8c?\x9a\x99\x99?ff\xa6?'

    buf = bytes()
    # slow way
    # for x in float_v:
    #     buf += struct.pack('f', np.round(x, round_dec))
    # faster way
    float_list = [np.round(x, round_dec) for x in float_v]
    buf = struct.pack('%sf' % len(float_list), *float_list)

    return buf

def bytes_to_vloat_v(bytes_buf, list_size, round_dec=7):
    # bytes to float vector
    # usage
    # rounded_reconved_float_v = bytes_to_vloat_v(bytes_buf=b'\xcd\xcc\x8c?\x9a\x99\x99?ff\xa6?', list_size=3)
    # >> > rounded_reconved_float_v
    # [1.1, 1.2, 1.3]

    # unpack bytes buffer to float vector
    reconved_float_v = struct.unpack('f' * list_size, bytes_buf)
    # rounding
    rounded_reconved_float_v = [np.round(x, round_dec) for x in reconved_float_v]

    return rounded_reconved_float_v

def float_v_to_hash(float_v, round_dec=7):
    bytes_buf = float_v_to_bytes(float_v, round_dec)
    hash_value = hashlib.sha256(bytes_buf).hexdigest()
    return hash_value




