import os
from cryptography.fernet import Fernet

class Coinbase_cryption(object):
    def __init__(self, path_env='COINBASE_PATH'):
        self.file_key = os.getenv('COINBASE_PATH')+'KEY.txt'
        self.file_pswd = os.getenv('COINBASE_PATH')+'PSWD.txt'
        self.file_API = os.getenv('COINBASE_PATH')+'API.txt'

    def _generate_new_key(self, pswd):
        key = Fernet.generate_key()
        # Write
        with open(self.file_key, 'wb') as f:
            f.write(key)
        with open(self.file_pswd, 'wb') as f:
            f.write(self._encrypt(pswd))

    def _get_key(self):
        with open(self.file_key, 'rb') as f:
            key = f.readline()
        return key

    def _encrypt(self, code):
        frn = Fernet(self._get_key())
        return frn.encrypt(code)

    def _decrypt(self, code):
        frn = Fernet(self._get_key())
        return frn.decrypt(code).decode('utf-8')

    def decrypt_pswd(self):
        # Read
        with open(self.file_pswd, 'rb') as f:
            pswd_encr = f.readline()
        return self._decrypt(pswd_encr)

    def _encrypt_api(self,api_key,api_secret):
        # Write
        with open(self.file_API, 'wb') as f:
            f.write(self._encrypt(api_key))
            f.write(b"\n")
            f.write(self._encrypt(api_secret))
    
    def decrypt_api(self):
        # Read
        with open(self.file_API, 'rb') as f:
            api_key=f.readline()
            api_secret=f.readline()
        return  self._decrypt(api_key), self._decrypt(api_secret)

if __name__=="__main__":
    CoinBase_pswd = Coinbase_cryption()
    #####################################################
    # TO DO only one time
    # API_key = b'123456'
    # API_secret = b'Haha again'
    # Password = b"Haha Try Again"
    # CoinBase_pswd._generate_new_key(Password)
    # CoinBase_pswd._encrypt_api(API_key, API_secret)
    #####################################################
    print(CoinBase_pswd.decrypt_pswd())
    print(CoinBase_pswd.decrypt_api())