""" Exmples of usage with tests.

Tests in this file represent examples taken from JSON-RPC specification.
http://www.jsonrpc.org/specification#examples

"""
import sys
import json

from ..manager import JSONRPCResponseManager
from ..jsonrpc2 import JSONRPC20Request, JSONRPC20BatchRequest

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


def isjsonequal(json1, json2):
    return json.loads(json1) == json.loads(json2)


class TestJSONRPCExamples(unittest.TestCase):
    def setUp(self):
        self.dispatcher = {
            "subtract": lambda a, b: a - b,
        }

    def test_rpc_call_with_positional_parameters(self):
        req = '{"jsonrpc": "2.0", "method": "subtract", "params": [42, 23], "id": 1}'  # noqa
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "result": 19, "id": 1}'
        ))

        req = '{"jsonrpc": "2.0", "method": "subtract", "params": [23, 42], "id": 2}'  # noqa
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "result": -19, "id": 2}'
        ))

    def test_rpc_call_with_named_parameters(self):
        def subtract(minuend=None, subtrahend=None):
            return minuend - subtrahend

        dispatcher = {
            "subtract": subtract,
            "sum": lambda *args: sum(args),
            "get_data": lambda: ["hello", 5],
        }

        req = '{"jsonrpc": "2.0", "method": "subtract", "params": {"subtrahend": 23, "minuend": 42}, "id": 3}'  # noqa
        response = JSONRPCResponseManager.handle(req, dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "result": 19, "id": 3}'
        ))

        req = '{"jsonrpc": "2.0", "method": "subtract", "params": {"minuend": 42, "subtrahend": 23}, "id": 4}'  # noqa
        response = JSONRPCResponseManager.handle(req, dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "result": 19, "id": 4}',
        ))

    def test_notification(self):
        req = '{"jsonrpc": "2.0", "method": "update", "params": [1,2,3,4,5]}'
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertEqual(response, None)

        req = '{"jsonrpc": "2.0", "method": "foobar"}'
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertEqual(response, None)

    def test_rpc_call_of_non_existent_method(self):
        req = '{"jsonrpc": "2.0", "method": "foobar", "id": "1"}'
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": "1"}'  # noqa
        ))

    def test_rpc_call_with_invalid_json(self):
        req = '{"jsonrpc": "2.0", "method": "foobar, "params": "bar", "baz]'
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": null}'  # noqa
        ))

    def test_rpc_call_with_invalid_request_object(self):
        req = '{"jsonrpc": "2.0", "method": 1, "params": "bar"}'
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": null}'  # noqa
        ))

    def test_rpc_call_batch_invalid_json(self):
        req = """[
            {"jsonrpc": "2.0", "method": "sum", "params": [1,2,4], "id": "1"},
            {"jsonrpc": "2.0", "method"
        ]"""
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": null}'  # noqa
        ))

    def test_rpc_call_with_an_empty_array(self):
        req = '[]'
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": null}'  # noqa
        ))

    def test_rpc_call_with_rpc_call_with_an_invalid_batch_but_not_empty(self):
        req = '[1]'
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isjsonequal(
            response.json,
            '{"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": null}'  # noqa
        ))

    def test_rpc_call_with_invalid_batch(self):
        req = '[1,2,3]'
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(
            response,
            json.loads("""[
                {"jsonrpc": "2.0", "error": {"code": -32600,
                "message": "Invalid Request"}, "id": null},
                {"jsonrpc": "2.0", "error": {"code": -32600,
                "message": "Invalid Request"}, "id": null},
                {"jsonrpc": "2.0", "error": {"code": -32600,
                "message": "Invalid Request"}, "id": null}
            ]""")
        )

    def test_rpc_call_batch(self):
        req = """[
            {"jsonrpc": "2.0", "method": "sum", "params": [1,2,4], "id": "1"},
            {"jsonrpc": "2.0", "method": "notify_hello", "params": [7]},
            {"jsonrpc": "2.0", "method": "subtract",
            "params": [42,23], "id": "2"},
            {"foo": "boo"},
            {"jsonrpc": "2.0", "method": "foo.get",
            "params": {"name": "myself"}, "id": "5"},
            {"jsonrpc": "2.0", "method": "get_data", "id": "9"}
        ]"""
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(
            response,
            json.loads("""[
                {"jsonrpc": "2.0", "result": 7, "id": "1"},
                {"jsonrpc": "2.0", "result": 19, "id": "2"},
                {"jsonrpc": "2.0", "error": {"code": -32600,
                "message": "Invalid Request"}, "id": null},
                {"jsonrpc": "2.0", "error": {"code": -32601,
                "message": "Method not found"}, "id": "5"},
                {"jsonrpc": "2.0", "result": ["hello", 5], "id": "9"}
            ]""")
        )

    def test_rpc_call_batch_all_notifications(self):
        req = """[
            {"jsonrpc": "2.0", "method": "notify_sum", "params": [1,2,4]},
            {"jsonrpc": "2.0", "method": "notify_hello", "params": [7]}
        ]"""
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertEqual(response, None)

    def test_rpc_call_response_request(self):
        req = '{"jsonrpc": "2.0", "method": "subtract", "params": [42, 23], "id": 1}'  # noqa
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isinstance(
            response.request,
            JSONRPC20Request
        ))
        self.assertTrue(isjsonequal(
            response.request.json,
            req
        ))

    def test_rpc_call_response_request_batch(self):
        req = """[
            {"jsonrpc": "2.0", "method": "sum", "params": [1,2,4], "id": "1"},
            {"jsonrpc": "2.0", "method": "notify_hello", "params": [7]},
            {"jsonrpc": "2.0", "method": "subtract",
            "params": [42,23], "id": "2"},
            {"jsonrpc": "2.0", "method": "foo.get",
            "params": {"name": "myself"}, "id": "5"},
            {"jsonrpc": "2.0", "method": "get_data", "id": "9"}
        ]"""
        response = JSONRPCResponseManager.handle(req, self.dispatcher)
        self.assertTrue(isinstance(
            response.request,
            JSONRPC20BatchRequest
        ))
        self.assertTrue(isjsonequal(
            response.request.json,
            req
        ))
