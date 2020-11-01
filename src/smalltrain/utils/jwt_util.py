import hmac
import hashlib
import json
import base64
import time
import sys

HEADER_MODEL = {
    'alg': 'HMAC-SHA256',
    'typ': 'JWT'
}

PAYLOAD_MODEL = {
    'sub': 'User_XXX',
    'exp': '1234567890'
}

# from smalltrain.utils.gg_setting import GGSetting
from smalltrain.utils.gg_setting import GGSetting


def get_secret_from_file(file_path=None):
    if file_path is None:
        setting = GGSetting()
        file_path = setting.get_jwt_secret_key_path()
    f = open(file_path, 'r')
    secret = f.read(0)
    f.close()
    # print('get_secret_from_file:{}'.format(secret))
    return secret


DEFAULT_SECRET = get_secret_from_file()


def create_jwt(header, payload, secret=DEFAULT_SECRET):
    signature = create_signature(header, payload, secret)
    header_b64encoded = b64encode_json(header).decode('utf-8')
    payload_b64encoded = b64encode_json(payload).decode('utf-8')
    return '{}.{}.{}'.format(header_b64encoded, payload_b64encoded, signature)


def decode_jwt(jwt, secret=DEFAULT_SECRET):
    try:
        header_b64encoded, payload_b64encoded, signature_actual = jwt.split('.')
    except ValueError as e:
        raise e

    header_decoded = json.loads(base64.urlsafe_b64decode(header_b64encoded + '==='))
    payload_decoded = json.loads(base64.urlsafe_b64decode(payload_b64encoded + '==='))

    # print(header_decoded)
    # check header
    if header_decoded['alg'].upper() != 'HMAC-SHA256':
        print('Invalid algorithm')
        return None
    if header_decoded['typ'].upper() != 'JWT':
        print('Invalid type')
        return None

    # check signature
    message = header_b64encoded + '.' + payload_b64encoded
    signature_expected = hmac_sha256(message, secret)

    return header_decoded, payload_decoded, signature_expected

class InvalidJWTError(Exception):
    def __init__(self, message='InvalidJWTError'):
        super().__init__(message)

def check_jwt(jwt, secret=DEFAULT_SECRET, raise_error=False):
    try:
        header_b64encoded, payload_b64encoded, signature_actual = jwt.split('.')
    except ValueError as e:
        # print(e)
        if raise_error: raise InvalidJWTError()
        return False
    header_decoded, payload_decoded, signature_expected = decode_jwt(jwt, secret)
    is_ok_check_jwt = signature_actual == signature_expected
    if raise_error and not is_ok_check_jwt: raise InvalidJWTError()
    return is_ok_check_jwt


def check_expiration_time(jwt, secret=DEFAULT_SECRET):
    try:
        header_b64encoded, payload_b64encoded, signature_actual = jwt.split('.')
        header_decoded, payload_decoded, signature_expected = decode_jwt(jwt, secret)
        if signature_actual != signature_expected:
            print('Invalid signature')
            return False
        exp = int(payload_decoded['exp'])
        return time.time() < exp
    except ValueError as e:
        print(e)
        return False


def get_sub(jwt, secret=DEFAULT_SECRET, raise_error=False):
    try:
        header_b64encoded, payload_b64encoded, signature_actual = jwt.split('.')
        header_decoded, payload_decoded, signature_expected = decode_jwt(jwt, secret)
        sub = payload_decoded['sub']
        return sub
    except ValueError as e:
        # print(e)
        if raise_error: raise InvalidJWTError()
        return None


def b64encode_json(json_or_dump):
    if not isinstance(json_or_dump, str): json_or_dump = json.dumps(json_or_dump)
    return base64.b64encode(json_or_dump.encode())


def create_signature(header, payload, secret=DEFAULT_SECRET):
    '''
    Create signature with HMAC-SHA256
    :param message: String, message
    :param secret: String, secret key
    :return: signature
    '''
    header_b64encoded = b64encode_json(header)
    payload_b64encoded = b64encode_json(payload)
    # print('header_b64encoded:{}'.format(header_b64encoded))
    message = header_b64encoded + '.'.encode() + payload_b64encoded
    signature = hmac_sha256(message, secret)

    return signature


def hmac_sha256(message, secret=DEFAULT_SECRET):
    '''
    Create signature with HMAC-SHA256
    :param message: String, message
    :param secret: String, secret key
    :return: hmac_sha256_signature
    '''
    if not isinstance(message, bytes): message = message.encode()
    if not isinstance(secret, bytes): secret = secret.encode()
    try:
        hmac_sha256_signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
    except TypeError:
        hmac_sha256_signature = None

    return hmac_sha256_signature


def test_create_signature():
    header = HEADER_MODEL.copy()
    payload = PAYLOAD_MODEL.copy()
    secret = '12345'
    signature1 = create_signature(header, payload, secret)
    print('signature1:{}'.format(signature1))

    payload['exp'] = '0234567890',
    signature2 = create_signature(header, payload, secret)
    print('signature2:{}'.format(signature2))


def test_create_decpde_jwt():
    print('##### test_create_decpde_jwt')
    header = HEADER_MODEL.copy()
    payload = PAYLOAD_MODEL.copy()
    jwt = create_jwt(header, payload)
    print('header:{}, payload:{}, jwt:{}'.format(header, payload, jwt))

    header_decoded, payload_decoded, signature_expected = decode_jwt(jwt)
    print('header_decoded:{}, payload:{}, signature_expected:{}'.format(header_decoded, payload, signature_expected))

    assert signature_expected == jwt.split('.')[-1]

    valid_jwt = check_jwt(jwt)
    print('valid_jwt:{}'.format(valid_jwt))
    assert valid_jwt

    print('##### try to change payload')
    payload_changed = payload.copy()
    payload_changed['exp'] = '0234567890'
    secret_invalid = '00000'
    signature = create_signature(header, payload_changed, secret_invalid)
    header_b64encoded = b64encode_json(header).decode('utf-8')
    payload_b64encoded = b64encode_json(payload_changed).decode('utf-8')
    jwt_changed = '{}.{}.{}'.format(header_b64encoded, payload_b64encoded, signature)
    print('header:{}, payload_changed:{}, jwt_changed:{}'.format(header, payload_changed, jwt_changed))
    header_decoded, payload_decoded, signature_expected = decode_jwt(jwt)
    print('header_decoded:{}, payload_decoded:{}, signature_expected:{}'.format(header_decoded, payload_decoded,
                                                                                signature_expected))

    assert signature_expected != jwt_changed.split('.')[-1]

    valid_jwt = check_jwt(jwt_changed)
    print('valid_jwt:{}'.format(valid_jwt))
    assert not valid_jwt


def test_check_expiration_time():
    print('##### test_check_expiration_time')
    header = HEADER_MODEL.copy()
    payload = PAYLOAD_MODEL.copy()
    payload['exp'] = '1555388373'  # about 2019/04/16 13:19

    jwt = create_jwt(header, payload)
    print('header:{}, payload:{}, jwt:{}'.format(header, payload, jwt))

    header_decoded, payload_decoded, signature_expected = decode_jwt(jwt)
    print('header_decoded:{}, payload:{}, signature_expected:{}'.format(header_decoded, payload, signature_expected))
    is_ok_exp = check_expiration_time(jwt)
    print('is_ok_exp:{}'.format(is_ok_exp))
    assert not is_ok_exp

    header = HEADER_MODEL.copy()
    payload = PAYLOAD_MODEL.copy()
    payload['exp'] = str(int(time.time()) + 60 * 60)  # 1 hour later

    jwt = create_jwt(header, payload)
    print('header:{}, payload:{}, jwt:{}'.format(header, payload, jwt))

    header_decoded, payload_decoded, signature_expected = decode_jwt(jwt)
    print('header_decoded:{}, payload:{}, signature_expected:{}'.format(header_decoded, payload, signature_expected))
    is_ok_exp = check_expiration_time(jwt)
    print('is_ok_exp:{}'.format(is_ok_exp))
    assert is_ok_exp


def test_create_admin_jwt():
    print('##### test_create_admin_jwt')
    header = HEADER_MODEL.copy()
    payload = PAYLOAD_MODEL.copy()
    payload['exp'] = str(int(time.time()) + 60 * 60)  # 1 hour later
    payload['sub'] = GGSetting().get_admin_user_id()

    jwt = create_jwt(header, payload)
    print('header:{}, payload:{}, jwt:{}'.format(header, payload, jwt))


if __name__ == '__main__':
    test_create_admin_jwt()
