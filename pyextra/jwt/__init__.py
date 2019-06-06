# -*- coding: utf-8 -*-
# flake8: noqa

"""
JSON Web Token implementation

Minimum implementation based on this spec:
http://self-issued.info/docs/draft-jones-json-web-token-01.html
"""


__title__ = 'pyjwt'
__version__ = '1.4.1'
__author__ = 'José Padilla'
__license__ = 'MIT'
__copyright__ = 'Copyright 2015 José Padilla'


from .api_jwt import (
    encode, decode, register_algorithm, unregister_algorithm,
    get_unverified_header, PyJWT
)
from .api_jws import PyJWS
from .exceptions import (
    InvalidTokenError, DecodeError, InvalidAudienceError,
    ExpiredSignatureError, ImmatureSignatureError, InvalidIssuedAtError,
    InvalidIssuerError, ExpiredSignature, InvalidAudience, InvalidIssuer,
    MissingRequiredClaimError
)
