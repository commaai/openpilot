""" JSON-RPC wrappers for version 1.0 and 2.0.

Objects diring init operation try to choose JSON-RPC 2.0 and in case of error
JSON-RPC 1.0.
from_json methods could decide what format is it by presence of 'jsonrpc'
attribute.

"""
from .utils import JSONSerializable
from .jsonrpc1 import JSONRPC10Request
from .jsonrpc2 import JSONRPC20Request


class JSONRPCRequest(JSONSerializable):

    """ JSONRPC Request."""

    @classmethod
    def from_json(cls, json_str):
        data = cls.deserialize(json_str)
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data):
        if isinstance(data, dict) and "jsonrpc" not in data:
            return JSONRPC10Request.from_data(data)
        else:
            return JSONRPC20Request.from_data(data)
