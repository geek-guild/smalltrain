from smalltrain.model import user
from smalltrain.model.user import User, UserManager, UserNotFoundError
import smalltrain.utils.jwt_util as jwt_util

import logging

# pytest -v --cov=model
# pytest -v --cov=model --cov-report=html
# pytest -v test/model/test_user.py -k "test_read_all"

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('TestUser')

from smalltrain.utils.gg_mongo_data_base import GGMongoDataBase
db = GGMongoDataBase.Instance()
db.set_db(host='localhost')

class TestUser:

    def test_encrypt_password(self):
        log.info('---------- test_encrypt_password ----------')
        plain_password = '12345'
        password = user.encrypt_password(plain_password)
        log.info('plain_password:{}, password:{}'.format(plain_password, password))
        assert password == '*** change for your environment ***'

    def test_create_guest_user(self):
        log.info('---------- test_create_guest_user ----------')
        manager = UserManager(db)
        user = User()
        user.user_id = '*** change for your environment ***'
        user.user_name = 'Guest'
        user.password = '*** change for your environment ***'
        manager.update(user)
        log.info('===== updated user:{} =========='.format(user))
        return user


    def test_read_all(self):
        log.info('---------- test_read_all ----------')
        manager = UserManager(db)
        user_read_list = manager.read_all()
        log.info('===== S all users with len:{} ====='.format(len(user_read_list)))
        for user_read in user_read_list:
            log.info('===== read user_id:{}, user_read:{} =========='.format(user_read.user_id, user_read))
        log.info('===== E all users =====')


    def test_create_and_delete_guest_user(self):
        log.info('---------- test_create_and_delete_guest_user ----------')
        user = self.test_create_guest_user()
        self.test_read_all()
        manager = UserManager(db)
        manager.delete(user)
        log.info('===== updated user:{} =========='.format(user))
        self.test_read_all()

    def test_read_with_jwt(self):
        log.info('---------- test_signin ----------')
        manager = UserManager(db)
        # signin with guest user
        guest_jwt = self.test_signin()
        log.info('guest_jwt:{}'.format(guest_jwt))
        user = manager.read_with_jwt(guest_jwt)
        log.info('===== read user:{} ====='.format(user))
        assert user.user_id == '08170382ac'

        # with error
        invalid_signature_jwt = '{}{}'.format(guest_jwt[:-2], 'XX')
        log.info('invalid_signature_jwt:{}'.format(invalid_signature_jwt))
        try:
            user = manager.read_with_jwt(invalid_signature_jwt)
            assert 'Not OK witount InvalidJWTError' is None
        except jwt_util.InvalidJWTError as e:
            log.info('OK with e:{}'.format(e))

        # no use
        log.info('temporary delete guest user')
        manager.delete(user)
        try:
            user = manager.read_with_jwt(guest_jwt)
            assert 'Not OK witount UserNotFoundError' is None
        except UserNotFoundError as e:
            log.info('OK with e:{}'.format(e))

        log.info('re-create guest user')
        user = self.test_create_guest_user()

    def test_signin(self):
        log.info('---------- test_signin ----------')
        manager = UserManager(db)

        user = self.test_create_guest_user()
        jwt = user.signin('hoge')
        assert jwt is None

        plane_password = '*** change for your environment ***'
        jwt = user.signin(plane_password)
        log.info('signin returns jwt:{}'.format(jwt))
        assert jwt is not None
        assert jwt_util.check_jwt(jwt)

        return jwt

# def test_read_with_jwt():
#     jwt =
#     UserManager.read_with_jwt(jwt)
