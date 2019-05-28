""" Test utility functionality."""
from ..utils import JSONSerializable, DatetimeDecimalEncoder, is_invalid_params

import datetime
import decimal
import json
import sys

if sys.version_info < (3, 3):
    from mock import patch
else:
    from unittest.mock import patch

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestJSONSerializable(unittest.TestCase):

    """ Test JSONSerializable functionality."""

    def setUp(self):
        class A(JSONSerializable):
            @property
            def json(self):
                pass

        self._class = A

    def test_abstract_class(self):
        with self.assertRaises(TypeError):
            JSONSerializable()

        self._class()

    def test_definse_serialize_deserialize(self):
        """ Test classmethods of inherited class."""
        self.assertEqual(self._class.serialize({}), "{}")
        self.assertEqual(self._class.deserialize("{}"), {})

    def test_from_json(self):
        self.assertTrue(isinstance(self._class.from_json('{}'), self._class))

    def test_from_json_incorrect(self):
        with self.assertRaises(ValueError):
            self._class.from_json('[]')


class TestDatetimeDecimalEncoder(unittest.TestCase):

    """ Test DatetimeDecimalEncoder functionality."""

    def test_date_encoder(self):
        obj = datetime.date.today()

        with self.assertRaises(TypeError):
            json.dumps(obj)

        self.assertEqual(
            json.dumps(obj, cls=DatetimeDecimalEncoder),
            '"{0}"'.format(obj.isoformat()),
        )

    def test_datetime_encoder(self):
        obj = datetime.datetime.now()

        with self.assertRaises(TypeError):
            json.dumps(obj)

        self.assertEqual(
            json.dumps(obj, cls=DatetimeDecimalEncoder),
            '"{0}"'.format(obj.isoformat()),
        )

    def test_decimal_encoder(self):
        obj = decimal.Decimal('0.1')

        with self.assertRaises(TypeError):
            json.dumps(obj)

        result = json.dumps(obj, cls=DatetimeDecimalEncoder)
        self.assertTrue(isinstance(result, str))
        self.assertEqual(float(result), float(0.1))

    def test_default(self):
        encoder = DatetimeDecimalEncoder()
        with patch.object(json.JSONEncoder, 'default') as json_default:
            encoder.default("")

        self.assertEqual(json_default.call_count, 1)


class TestUtils(unittest.TestCase):

    """ Test utils functions."""

    def test_is_invalid_params_builtin(self):
        self.assertTrue(is_invalid_params(sum, 0, 0))
        # NOTE: builtin functions could not be recognized by inspect.isfunction
        # It would raise TypeError if parameters are incorrect already.
        # self.assertFalse(is_invalid_params(sum, [0, 0]))  # <- fails

    def test_is_invalid_params_args(self):
        self.assertTrue(is_invalid_params(lambda a, b: None, 0))
        self.assertTrue(is_invalid_params(lambda a, b: None, 0, 1, 2))

    def test_is_invalid_params_kwargs(self):
        self.assertTrue(is_invalid_params(lambda a: None, **{}))
        self.assertTrue(is_invalid_params(lambda a: None, **{"a": 0, "b": 1}))

    def test_invalid_params_correct(self):
        self.assertFalse(is_invalid_params(lambda: None))
        self.assertFalse(is_invalid_params(lambda a: None, 0))
        self.assertFalse(is_invalid_params(lambda a, b=0: None, 0))
        self.assertFalse(is_invalid_params(lambda a, b=0: None, 0, 0))

    def test_is_invalid_params_mixed(self):
        self.assertFalse(is_invalid_params(lambda a, b: None, 0, **{"b": 1}))
        self.assertFalse(is_invalid_params(
            lambda a, b, c=0: None, 0, **{"b": 1}))

    def test_is_invalid_params_py2(self):
        with patch('jsonrpc.utils.sys') as mock_sys:
            mock_sys.version_info = (2, 7)
            with patch('jsonrpc.utils.is_invalid_params_py2') as mock_func:
                is_invalid_params(lambda a: None, 0)

        assert mock_func.call_count == 1
