import json


# default_setting =
# {
#   "admin_user_id": "User_XXX",
#   "db_user": "db_user",
#   "db_password": "db_password",
#   "jwt_secret_key_path": "/usr/local/etc/vendor/gg/jwt_secret.key"
# }

class GGSetting:

    DEFAULT_WEB_API_PORT = 50000

    def __init__(self):
        self.read_setting_from_file()

    def read_setting_from_file(self, file_path='/usr/local/etc/vendor/gg/smalltrain/smalltrain_conf.json'):
        with open(file_path) as f:
            self.setting_json = json.load(f)

    def get(self, key):
        return self.setting_json[key]

    def get_web_api_port(self):
        web_api_port = self.setting_json.get('web_api_port')
        try:
            web_api_port = int(web_api_port)
            assert web_api_port > 0
        except (AssertionError, ValueError) as e:
            web_api_port = GGSetting.DEFAULT_WEB_API_PORT

        return web_api_port

    def get_admin_user_id(self):
        return self.setting_json['admin_user_id']

    def get_jwt_secret_key_path(self):
        return self.setting_json['jwt_secret_key_path']

    def get_db_user(self):
        return self.setting_json['db_user']

    def get_db_password(self):
        return self.setting_json['db_password']


def test_read_setting_from_file():
    print('##### test_read_setting_from_file')
    setting = GGSetting()
    admin_user_id = setting.get('admin_user_id')
    jwt_secret_key_path = setting.get('jwt_secret_key_path')
    print('admin_user_id:{}, jwt_secret_key:{}'.format(admin_user_id, jwt_secret_key_path))

    assert admin_user_id == "User_XXX"
    assert jwt_secret_key_path == "/usr/local/etc/vendor/gg/jwt_secret.key"

    assert setting.get_admin_user_id() == "User_XXX"
    assert setting.get_jwt_secret_key_path() == "/usr/local/etc/vendor/gg/jwt_secret.key"


if __name__ == '__main__':
    test_read_setting_from_file()
