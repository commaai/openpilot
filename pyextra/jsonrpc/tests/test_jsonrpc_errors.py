import json
import sys

from ..exceptions import (
    JSONRPCError,
    JSONRPCInternalError,
    JSONRPCInvalidParams,
    JSONRPCInvalidRequest,
    JSONRPCMethodNotFound,
    JSONRPCParseError,
    JSONRPCServerError,
    JSONRPCDispatchException,
)

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestJSONRPCError(unittest.TestCase):
    def setUp(self):
        self.error_params = {
            "code": 0,
            "message": "",
        }

    def test_correct_init(self):
        """ Test object is created."""
        JSONRPCError(**self.error_params)

    def test_validation_incorrect_no_parameters(self):
        with self.assertRaises(ValueError):
            JSONRPCError()

    def test_code_validation_int(self):
        self.error_params.update({"code": 32000})
        JSONRPCError(**self.error_params)

    def test_code_validation_no_code(self):
        del self.error_params["code"]
        with self.assertRaises(ValueError):
            JSONRPCError(**self.error_params)

    def test_code_validation_str(self):
        self.error_params.update({"code": "0"})
        with self.assertRaises(ValueError):
            JSONRPCError(**self.error_params)

    def test_message_validation_str(self):
        self.error_params.update({"message": ""})
        JSONRPCError(**self.error_params)

    def test_message_validation_none(self):
        del self.error_params["message"]
        with self.assertRaises(ValueError):
            JSONRPCError(**self.error_params)

    def test_message_validation_int(self):
        self.error_params.update({"message": 0})
        with self.assertRaises(ValueError):
            JSONRPCError(**self.error_params)

    def test_data_validation_none(self):
        self.error_params.update({"data": None})
        JSONRPCError(**self.error_params)

    def test_data_validation(self):
        self.error_params.update({"data": {}})
        JSONRPCError(**self.error_params)

        self.error_params.update({"data": ""})
        JSONRPCError(**self.error_params)

    def test_json(self):
        error = JSONRPCError(**self.error_params)
        self.assertEqual(
            json.loads(error.json),
            self.error_params,
        )

    def test_from_json(self):
        str_json = json.dumps({
            "code": 0,
            "message": "",
            "data": {},
        })

        request = JSONRPCError.from_json(str_json)
        self.assertTrue(isinstance(request, JSONRPCError))
        self.assertEqual(request.code, 0)
        self.assertEqual(request.message, "")
        self.assertEqual(request.data, {})


class TestJSONRPCParseError(unittest.TestCase):
    def test_code_message(self):
        error = JSONRPCParseError()
        self.assertEqual(error.code, -32700)
        self.assertEqual(error.message, "Parse error")
        self.assertEqual(error.data, None)


class TestJSONRPCServerError(unittest.TestCase):
    def test_code_message(self):
        error = JSONRPCServerError()
        self.assertEqual(error.code, -32000)
        self.assertEqual(error.message, "Server error")
        self.assertEqual(error.data, None)


class TestJSONRPCInternalError(unittest.TestCase):
    def test_code_message(self):
        error = JSONRPCInternalError()
        self.assertEqual(error.code, -32603)
        self.assertEqual(error.message, "Internal error")
        self.assertEqual(error.data, None)


class TestJSONRPCInvalidParams(unittest.TestCase):
    def test_code_message(self):
        error = JSONRPCInvalidParams()
        self.assertEqual(error.code, -32602)
        self.assertEqual(error.message, "Invalid params")
        self.assertEqual(error.data, None)


class TestJSONRPCInvalidRequest(unittest.TestCase):
    def test_code_message(self):
        error = JSONRPCInvalidRequest()
        self.assertEqual(error.code, -32600)
        self.assertEqual(error.message, "Invalid Request")
        self.assertEqual(error.data, None)


class TestJSONRPCMethodNotFound(unittest.TestCase):
    def test_code_message(self):
        error = JSONRPCMethodNotFound()
        self.assertEqual(error.code, -32601)
        self.assertEqual(error.message, "Method not found")
        self.assertEqual(error.data, None)


class TestJSONRPCDispatchException(unittest.TestCase):
    def test_code_message(self):
        error = JSONRPCDispatchException(message="message",
                                         code=400, data={"param": 1})
        self.assertEqual(error.error.code, 400)
        self.assertEqual(error.error.message, "message")
        self.assertEqual(error.error.data, {"param": 1})
