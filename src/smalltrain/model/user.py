import pandas
import hashlib
import time
import sys
import numpy as np

from datetime import datetime
from datetime import timedelta

from smalltrain.utils.gg_mongo_data_base import GGMongoDataBase
import smalltrain.utils.jwt_util as jwt_util

EXPIRATION_HOURS_TO = 1  # TODO


class UserManager:

    def __init__(self, db=GGMongoDataBase.Instance(), debug_mode=False):
        '''
        '''
        self._db = db

    def set_db(self, db=GGMongoDataBase.Instance(), debug_mode=False):
        self._db = db

    # CRUD
    def read(self, user_id):
        _user_json = self._db.read_with_group_key(User.group_key, user_id)
        # print('[read]_user_json:{}'.format(_user_json))
        if _user_json is None:
            return None
        user = User()
        user.set_with_json(_user_json)
        return user

    def read_all(self):
        keys = self._db.keys(group_key=User.group_key)
        print('keys:{}'.format(keys))
        all_users = [self.read(key) for key in keys]
        return all_users

    def update_with_json(self, user_json):
        user = User()
        user.set_with_json(user_json)
        user.update(self._db)

    def update(self, user):
        user.update(self._db)
        self._db.save()

    def delete(self, user):
        user.delete(self._db)
        self._db.save()

    def check_password(self, user_id, password_to_check):
        user = self.read(user_id=user_id)
        if user is None:
            print('User not fount with user_id:{}'.format(user_id))
            return None
        return user.check_password(password_to_check)

    def read_with_jwt(self, jwt):
        # check jwt
        jwt_util.check_jwt(jwt, raise_error=True)
        user_id = jwt_util.get_sub(jwt, raise_error=True)
        if user_id is None: raise UserNotFoundError()

        # check user_id
        user_read = self.read(user_id)
        if user_read is None: raise UserNotFoundError()
        if user_read.user_id is None: raise UserNotFoundError()

        return user_read


class User:
    group_key = 'User'

    def __init__(self, user_id=None, user_name=None, plain_password=None, last_signin_dt=None, session_id=None):
        '''
        :param user_id: String, ID
        :param user_name: String, user name
        :param plain_password: String, plain password
        :param last_signin_dt: Datetime, last signin datetimem(in UTC)
        :param session_id: String,
        '''
        self.set(user_id, user_name, plain_password, last_signin_dt, session_id)

    @property
    def password(self):
        # raise Exception('Can not get password')
        return '********'

    @password.setter
    def password(self, plain_password):
        self.__password = encrypt_password(plain_password)

    @password.deleter
    def password(self):
        del self.__password

    def set(self, user_id=None, user_name=None, plain_password=None, last_signin_dt=None, session_id=None):
        if user_id is None:
            # create user_id
            self.user_id = self.create_user_id()
        else:
            self.user_id = user_id

        # set encripted password
        self.__password = plain_password
        self.user_name = user_name
        self.last_signin_dt = last_signin_dt
        self.session_id = session_id

    def set_with_json(self, use_json):
        # user_id
        self.user_id = use_json['user_id']

        # user_name
        try:
            self.user_name = use_json['user_name']
        except KeyError:
            self.user_name = None

        # password
        try:
            self.__password = use_json['password']
        except KeyError:
            self.__password = None

        # last_signin_dt
        try:
            self.last_signin_dt = use_json['last_signin_dt']
        except KeyError:
            self.last_signin_dt = None

        # session_id
        try:
            self.session_id = use_json['session_id']
        except KeyError:
            self.session_id = None

    def __dump(self):
        _user_json = {}
        _user_json['user_id'] = self.user_id
        _user_json['user_name'] = self.user_name
        _user_json['password'] = self.__password
        _user_json['last_signin_dt'] = self.last_signin_dt
        _user_json['session_id'] = self.session_id
        # print('[dump]_user_json:{}'.format(_user_json))
        return _user_json

    def __str__(self):
        _user_json = self.__dump()
        _user_json['password'] = '********'
        return str(_user_json)

    def create_user_id(self):
        _time = '0000000000' + str(int(time.time()) - 1546300800)  # time from 2019/01/01
        # print(_time)
        hash = _generate_sha256_hash(np.random.rand())
        src_str = '{}{}'.format(str(_time)[-8:], hash[:2])
        return src_str

    def create_session_id(self):
        _time = time.time()
        hash = _generate_sha256_hash('session_id_' + str(_time + np.random.rand()))
        return hash

    def update(self, _db):
        return _db.update_with_group_key(User.group_key, self.user_id, self.__dump())

    def delete(self, _db):
        return _db.delete_with_group_key(User.group_key, self.user_id)

    def check_password(self, password_to_check):
        return self.__password == encrypt_password(password_to_check)

    def update_password(self, password_to_update):
        self.password = password_to_update

    def signin(self, password):

        # signin
        if self.check_password(password):
            signin_time = int(time.time())
            header = jwt_util.HEADER_MODEL.copy()
            payload = jwt_util.PAYLOAD_MODEL.copy()
            payload['exp'] = str(signin_time + EXPIRATION_HOURS_TO * 60 * 60)
            payload['sub'] = self.user_id

            jwt = jwt_util.create_jwt(header, payload)
            self.last_signin_dt = datetime(1970, 1, 1) + timedelta(seconds=signin_time)
            self.update(_db=GGMongoDataBase.Instance())

            return jwt
        return None

    def signout(self):
        # check signin
        return False


_salt = 't8GB(Tg&F1'
_stretching = 10


def encrypt_password(plain_password):
    if plain_password is None: return None
    encrypted = str(plain_password) + _salt
    for i in range(_stretching):
        encrypted = _generate_sha256_hash(encrypted)
    return encrypted


def _generate_sha256_hash(src_str):
    # hash = hashlib.sha256()
    # hash.update(src_str)
    # digest = hash.hexdigest()
    digest = hashlib.sha256(str(src_str).encode()).hexdigest()
    return digest


class UserNotFoundError(Exception):
    def __init__(self, message='UserNotFoundError'):
        super().__init__(message)
