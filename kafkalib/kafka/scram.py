from __future__ import absolute_import

import base64
import hashlib
import hmac
import uuid

from kafka.vendor import six


if six.PY2:
    def xor_bytes(left, right):
        return bytearray(ord(lb) ^ ord(rb) for lb, rb in zip(left, right))
else:
    def xor_bytes(left, right):
        return bytes(lb ^ rb for lb, rb in zip(left, right))


class ScramClient:
    MECHANISMS = {
        'SCRAM-SHA-256': hashlib.sha256,
        'SCRAM-SHA-512': hashlib.sha512
    }

    def __init__(self, user, password, mechanism):
        self.nonce = str(uuid.uuid4()).replace('-', '')
        self.auth_message = ''
        self.salted_password = None
        self.user = user
        self.password = password.encode('utf-8')
        self.hashfunc = self.MECHANISMS[mechanism]
        self.hashname = ''.join(mechanism.lower().split('-')[1:3])
        self.stored_key = None
        self.client_key = None
        self.client_signature = None
        self.client_proof = None
        self.server_key = None
        self.server_signature = None

    def first_message(self):
        client_first_bare = 'n={},r={}'.format(self.user, self.nonce)
        self.auth_message += client_first_bare
        return 'n,,' + client_first_bare

    def process_server_first_message(self, server_first_message):
        self.auth_message += ',' + server_first_message
        params = dict(pair.split('=', 1) for pair in server_first_message.split(','))
        server_nonce = params['r']
        if not server_nonce.startswith(self.nonce):
            raise ValueError("Server nonce, did not start with client nonce!")
        self.nonce = server_nonce
        self.auth_message += ',c=biws,r=' + self.nonce

        salt = base64.b64decode(params['s'].encode('utf-8'))
        iterations = int(params['i'])
        self.create_salted_password(salt, iterations)

        self.client_key = self.hmac(self.salted_password, b'Client Key')
        self.stored_key = self.hashfunc(self.client_key).digest()
        self.client_signature = self.hmac(self.stored_key, self.auth_message.encode('utf-8'))
        self.client_proof = xor_bytes(self.client_key, self.client_signature)
        self.server_key = self.hmac(self.salted_password, b'Server Key')
        self.server_signature = self.hmac(self.server_key, self.auth_message.encode('utf-8'))

    def hmac(self, key, msg):
        return hmac.new(key, msg, digestmod=self.hashfunc).digest()

    def create_salted_password(self, salt, iterations):
        self.salted_password = hashlib.pbkdf2_hmac(
            self.hashname, self.password, salt, iterations
        )

    def final_message(self):
        return 'c=biws,r={},p={}'.format(self.nonce, base64.b64encode(self.client_proof).decode('utf-8'))

    def process_server_final_message(self, server_final_message):
        params = dict(pair.split('=', 1) for pair in server_final_message.split(','))
        if self.server_signature != base64.b64decode(params['v'].encode('utf-8')):
            raise ValueError("Server sent wrong signature!")


