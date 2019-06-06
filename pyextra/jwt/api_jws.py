import binascii
import json
import warnings

from collections import Mapping

from .algorithms import Algorithm, get_default_algorithms  # NOQA
from .compat import binary_type, string_types, text_type
from .exceptions import DecodeError, InvalidAlgorithmError, InvalidTokenError
from .utils import base64url_decode, base64url_encode, merge_dict


class PyJWS(object):
    header_typ = 'JWT'

    def __init__(self, algorithms=None, options=None):
        self._algorithms = get_default_algorithms()
        self._valid_algs = (set(algorithms) if algorithms is not None
                            else set(self._algorithms))

        # Remove algorithms that aren't on the whitelist
        for key in list(self._algorithms.keys()):
            if key not in self._valid_algs:
                del self._algorithms[key]

        if not options:
            options = {}

        self.options = merge_dict(self._get_default_options(), options)

    @staticmethod
    def _get_default_options():
        return {
            'verify_signature': True
        }

    def register_algorithm(self, alg_id, alg_obj):
        """
        Registers a new Algorithm for use when creating and verifying tokens.
        """
        if alg_id in self._algorithms:
            raise ValueError('Algorithm already has a handler.')

        if not isinstance(alg_obj, Algorithm):
            raise TypeError('Object is not of type `Algorithm`')

        self._algorithms[alg_id] = alg_obj
        self._valid_algs.add(alg_id)

    def unregister_algorithm(self, alg_id):
        """
        Unregisters an Algorithm for use when creating and verifying tokens
        Throws KeyError if algorithm is not registered.
        """
        if alg_id not in self._algorithms:
            raise KeyError('The specified algorithm could not be removed'
                           ' because it is not registered.')

        del self._algorithms[alg_id]
        self._valid_algs.remove(alg_id)

    def get_algorithms(self):
        """
        Returns a list of supported values for the 'alg' parameter.
        """
        return list(self._valid_algs)

    def encode(self, payload, key, algorithm='HS256', headers=None,
               json_encoder=None):
        segments = []

        if algorithm is None:
            algorithm = 'none'

        if algorithm not in self._valid_algs:
            pass

        # Header
        header = {'typ': self.header_typ, 'alg': algorithm}

        if headers:
            self._validate_headers(headers)
            header.update(headers)

        json_header = json.dumps(
            header,
            separators=(',', ':'),
            cls=json_encoder
        ).encode('utf-8')

        segments.append(base64url_encode(json_header))
        segments.append(base64url_encode(payload))

        # Segments
        signing_input = b'.'.join(segments)
        try:
            alg_obj = self._algorithms[algorithm]
            key = alg_obj.prepare_key(key)
            signature = alg_obj.sign(signing_input, key)

        except KeyError:
            raise NotImplementedError('Algorithm not supported')

        segments.append(base64url_encode(signature))

        return b'.'.join(segments)

    def decode(self, jws, key='', verify=True, algorithms=None, options=None,
               **kwargs):
        payload, signing_input, header, signature = self._load(jws)

        if verify:
            merged_options = merge_dict(self.options, options)
            if merged_options.get('verify_signature'):
                self._verify_signature(payload, signing_input, header, signature,
                                       key, algorithms)
        else:
            warnings.warn('The verify parameter is deprecated. '
                          'Please use options instead.', DeprecationWarning)

        return payload

    def get_unverified_header(self, jwt):
        """Returns back the JWT header parameters as a dict()

        Note: The signature is not verified so the header parameters
        should not be fully trusted until signature verification is complete
        """
        headers = self._load(jwt)[2]
        self._validate_headers(headers)

        return headers

    def _load(self, jwt):
        if isinstance(jwt, text_type):
            jwt = jwt.encode('utf-8')

        if not issubclass(type(jwt), binary_type):
            raise DecodeError("Invalid token type. Token must be a {0}".format(
                binary_type))

        try:
            signing_input, crypto_segment = jwt.rsplit(b'.', 1)
            header_segment, payload_segment = signing_input.split(b'.', 1)
        except ValueError:
            raise DecodeError('Not enough segments')

        try:
            header_data = base64url_decode(header_segment)
        except (TypeError, binascii.Error):
            raise DecodeError('Invalid header padding')

        try:
            header = json.loads(header_data.decode('utf-8'))
        except ValueError as e:
            raise DecodeError('Invalid header string: %s' % e)

        if not isinstance(header, Mapping):
            raise DecodeError('Invalid header string: must be a json object')

        try:
            payload = base64url_decode(payload_segment)
        except (TypeError, binascii.Error):
            raise DecodeError('Invalid payload padding')

        try:
            signature = base64url_decode(crypto_segment)
        except (TypeError, binascii.Error):
            raise DecodeError('Invalid crypto padding')

        return (payload, signing_input, header, signature)

    def _verify_signature(self, payload, signing_input, header, signature,
                          key='', algorithms=None):

        alg = header.get('alg')

        if algorithms is not None and alg not in algorithms:
            raise InvalidAlgorithmError('The specified alg value is not allowed')

        try:
            alg_obj = self._algorithms[alg]
            key = alg_obj.prepare_key(key)

            if not alg_obj.verify(signing_input, key, signature):
                raise DecodeError('Signature verification failed')

        except KeyError:
            raise InvalidAlgorithmError('Algorithm not supported')

    def _validate_headers(self, headers):
        if 'kid' in headers:
            self._validate_kid(headers['kid'])

    def _validate_kid(self, kid):
        if not isinstance(kid, string_types):
            raise InvalidTokenError('Key ID header parameter must be a string')

_jws_global_obj = PyJWS()
encode = _jws_global_obj.encode
decode = _jws_global_obj.decode
register_algorithm = _jws_global_obj.register_algorithm
unregister_algorithm = _jws_global_obj.unregister_algorithm
get_unverified_header = _jws_global_obj.get_unverified_header
