""" Exmples of usage with tests.

Tests in this file represent examples taken from JSON-RPC specification.
http://www.jsonrpc.org/specification#examples

"""
import sys
import json

from ..manager import JSONRPCResponseManager

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


def isjsonequal(json1, json2):
    return json.loads(json1) == json.loads(json2)


class TestJSONRPCExamples(unittest.TestCase):
    def setUp(self):
        self.dispatcher = {
            "return_none": lambda: None,
        }

    def test_none_as_result(self):
        req = '{"jsonrpc": "2.0", "method": "return_none", "id": 0}'
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "result": null, "id": 0}'
        ))
