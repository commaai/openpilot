import Crypto.Hash.SHA256
import Crypto.Hash.SHA384
import Crypto.Hash.SHA512

from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5

from jwt.algorithms import Algorithm
from jwt.compat import string_types, text_type


class RSAAlgorithm(Algorithm):
    """
    Performs signing and verification operations using
    RSASSA-PKCS-v1_5 and the specified hash function.

    This class requires PyCrypto package to be installed.

    This is based off of the implementation in PyJWT 0.3.2
    """
    SHA256 = Crypto.Hash.SHA256
    SHA384 = Crypto.Hash.SHA384
    SHA512 = Crypto.Hash.SHA512

    def __init__(self, hash_alg):
        self.hash_alg = hash_alg

    def prepare_key(self, key):

        if isinstance(key, RSA._RSAobj):
            return key

        if isinstance(key, string_types):
            if isinstance(key, text_type):
                key = key.encode('utf-8')

            key = RSA.importKey(key)
        else:
            raise TypeError('Expecting a PEM- or RSA-formatted key.')

        return key

    def sign(self, msg, key):
        return PKCS1_v1_5.new(key).sign(self.hash_alg.new(msg))

    def verify(self, msg, key, sig):
        return PKCS1_v1_5.new(key).verify(self.hash_alg.new(msg), sig)
