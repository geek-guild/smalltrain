import pymongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import traceback
import hashlib

from smalltrain.utils.gg_setting import GGSetting


class Singleton:
    """
    Simple Singletons class
      - Non-thread-safe
      - Use with decoration class with @Singleton
      - The @Singleton class cannot be inherited from.
      - The @Singleton class provides the singleton instance by the `Instance` method.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance.
        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class GGMongoDataBase:

    def __init__(self):
        self.set_db()  # set db with default setting

    def set_db(self, host='localhost', port=27017, db_name='ggdb', key_hashing=None, use_connection_pool=True,
               debug_mode=False):
        self._host = host
        self._port = port
        self._key_hashing = key_hashing
        __u = GGSetting().get_db_user()
        __p = GGSetting().get_db_password()
        __client = MongoClient(host=host, port=port,
                      username=__u,
                      password=__p,
                      connect=False, # to avoid error: pymongo.errors.ServerSelectionTimeoutError: No servers found yet
                      authSource='admin',
                      authMechanism='SCRAM-SHA-1')

        self._db_ins = __client[db_name]

    def get_host(self):
        return self._host

    def keys(self, pattern=None, group_key=None):
        '''
        :return: all keys
        '''
        if group_key is not None:
            _collection = self._db_ins[group_key]
            _cr = _collection.find()
            # return _cr.sort().toArray()
            print('group_key:{}, keys:{}'.format(group_key, sorted(list(_cr))))
            return sorted(list(_cr))

        ret_list = []
        all_collection_list = self._db_ins.collection_names()
        if pattern is not None:
            import fnmatch
            all_collection_list = fnmatch.filter(all_collection_list, pattern)

        for _collection in all_collection_list:
            _cr = _collection.find()
            _key_list = list(_cr)
            ret_list.extend(_key_list)

        print('pattern:{}, keys:{}'.format(pattern, sorted(ret_list)))
        return sorted(ret_list)

    def read(self, key, key_hashing=None):
        raise Exception('[read] Use read_with_group_key instead')

    def update(self, key, value, key_hashing=None):
        raise Exception('[read] Use update_with_group_key instead')

    def delete(self, key, key_hashing=None):
        raise Exception('[read] Use delete_with_group_key instead')

    def read_range(self, key, range1=None, range2=None, key_hashing=None):
        raise Exception('[read_range] is not ready to use')

    def push(self, key, value, key_hashing=None):
        raise Exception('[push] is not ready to use')

    def read_with_group_key(self, group_key, key, key_hashing=None):
        PREFIX = 'read_with_group_key'
        group_key = self.convert_key_with_key_hashing(group_key, key_hashing)
        key = self.convert_key_with_key_hashing(key, key_hashing)
        _collection = self._db_ins[group_key]
        try:
            _cr = _collection.find({'_id': key})
            read_ojb_list = list(_cr)
            print('[{}]read_ojb_list:{}'.format(PREFIX, read_ojb_list))
            assert len(read_ojb_list) < 2
            if len(read_ojb_list) == 1: return read_ojb_list[0]['value']
            return None
        except ServerSelectionTimeoutError as e:
            message = '[{}]Error with e:{}'.format(PREFIX, e)
            traceback.print_exc()
            raise TimeoutError(message)


    def update_with_group_key(self, group_key, key, value, key_hashing=None):
        group_key = self.convert_key_with_key_hashing(group_key, key_hashing)
        key = self.convert_key_with_key_hashing(key, key_hashing)
        _collection = self._db_ins[group_key]
        try:
            _collection.insert({'_id': key, 'value': value})
        except pymongo.errors.DuplicateKeyError:
            _collection.update({'_id': key}, {'$set': {'value': value}})

    def delete_with_group_key(self, group_key, key, key_hashing=None):
        group_key = self.convert_key_with_key_hashing(group_key, key_hashing)
        key = self.convert_key_with_key_hashing(key, key_hashing)
        _collection = self._db_ins[group_key]
        _collection.remove({'_id': key})

    def convert_key_with_key_hashing(self, key, key_hashing):
        if key_hashing is not None:
            print('No need to set key_hashing in MongoDB')
            return key
        if key is None: return key
        if key_hashing is not None: key_hashing = self._key_hashing
        if key_hashing is not None:
            if key_hashing == 'sha256':
                return _generate_sha256_hash(key)
            else:
                raise KeyError('key_hashing is allowed only with sha256')
        else:
            return key

    def save(self):
        warnings_message = '[save] is not ready to use'
        print(warnings_message)
        return

class TimeoutError(Exception):
    def __init__(self, message='ServerSelectionTimeoutError'):
        super().__init__(message)


def _generate_sha256_hash(src_str):
    digest = hashlib.sha256(str(src_str).encode()).hexdigest()
    return digest


def test_read():
    db = GGMongoDataBase.Instance()
    db.set_db(db_name='ggdb')

    group_key = 'TestUser'
    key = '1234567890'
    value = db.read_with_group_key(group_key, key)
    print('[test_read]{}'.format(value))


def test_update():
    db = GGMongoDataBase.Instance()
    db.set_db(db_name='test')

    group_key = 'TestUser'
    key = '1234567890'
    value = {}
    value['name'] = 'TestUserName'
    value['email'] = 'test@geek-guild.jp'
    db.update_with_group_key(group_key, key, value)
    print('[test_update]{}'.format('DOne'))


def test_keys():
    import time

    # db_host = '13.230.104.85'
    db_host = 'localhost'

    db = GGMongoDataBase.Instance()
    db.set_db(db_name='ggdb')

    start = time.time()
    group_key = 'TestUser'
    keys = db.keys(group_key=group_key)
    print('========== read with group_key:{} keys:{} in {} sec'.format(group_key, keys, time.time() - start))

    for key in keys:
        value = db.read(key)
        print('DONE read key:{}, value:{}'.format(key, value))
        # db.delete_with_group_key(group_key, key)

    pattern = '*se*'
    start = time.time()
    keys = db.keys(pattern=pattern)
    print('========== read with pattern:{} keys:{} in {} sec'.format(pattern, keys, time.time() - start))

    for key in keys:
        value = db.read(key)
        print('DONE read key:{}, value:{}'.format(key, value))
        # db.delete(key)
        # db.delete_with_group_key('TestUser', key)


def _test():
    db = GGMongoDataBase.Instance()

    for collection in db._db_ins.collection_names():
        print('find collection:'.format(collection))

    # if having error below, check port setting
    # pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 61] Connection refused


if __name__ == '__main__':
    test_read()
    exit()

    test_update()
    test_read()
    exit()

    test_keys()
    exit()

    _test()
    exit()
