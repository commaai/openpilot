""" Utility functions for package."""
from abc import ABCMeta, abstractmethod
import datetime
import decimal
import inspect
import json
import sys

from . import six


class JSONSerializable(six.with_metaclass(ABCMeta, object)):

    """ Common functionality for json serializable objects."""

    serialize = staticmethod(json.dumps)
    deserialize = staticmethod(json.loads)

    @abstractmethod
    def json(self):
        raise NotImplementedError()

    @classmethod
    def from_json(cls, json_str):
        data = cls.deserialize(json_str)

        if not isinstance(data, dict):
            raise ValueError("data should be dict")

        return cls(**data)


class DatetimeDecimalEncoder(json.JSONEncoder):

    """ Encoder for datetime and decimal serialization.

    Usage: json.dumps(object, cls=DatetimeDecimalEncoder)
    NOTE: _iterencode does not work

    """

    def default(self, o):
        """ Encode JSON.

        :return str: A JSON encoded string

        """
        if isinstance(o, decimal.Decimal):
            return float(o)

        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()

        return json.JSONEncoder.default(self, o)


def is_invalid_params_py2(func, *args, **kwargs):
    """ Check, whether function 'func' accepts parameters 'args', 'kwargs'.

    NOTE: Method is called after funct(*args, **kwargs) generated TypeError,
    it is aimed to destinguish TypeError because of invalid parameters from
    TypeError from inside the function.

    .. versionadded: 1.9.0

    """
    funcargs, varargs, varkwargs, defaults = inspect.getargspec(func)

    unexpected = set(kwargs.keys()) - set(funcargs)
    if len(unexpected) > 0:
        return True

    params = [funcarg for funcarg in funcargs if funcarg not in kwargs]
    funcargs_required = funcargs[:-len(defaults)] \
        if defaults is not None \
        else funcargs
    params_required = [
        funcarg for funcarg in funcargs_required
        if funcarg not in kwargs
    ]

    return not (len(params_required) <= len(args) <= len(params))


def is_invalid_params_py3(func, *args, **kwargs):
    """
    Use inspect.signature instead of inspect.getargspec or
    inspect.getfullargspec (based on inspect.signature itself) as it provides
    more information about function parameters.

    .. versionadded: 1.11.2

    """
    signature = inspect.signature(func)
    parameters = signature.parameters

    unexpected = set(kwargs.keys()) - set(parameters.keys())
    if len(unexpected) > 0:
        return True

    params = [
        parameter for name, parameter in parameters.items()
        if name not in kwargs
    ]
    params_required = [
        param for param in params
        if param.default is param.empty
    ]

    return not (len(params_required) <= len(args) <= len(params))


def is_invalid_params(func, *args, **kwargs):
    """
    Method:
        Validate pre-defined criteria, if any is True - function is invalid
        0. func should be callable
        1. kwargs should not have unexpected keywords
        2. remove kwargs.keys from func.parameters
        3. number of args should be <= remaining func.parameters
        4. number of args should be >= remaining func.parameters less default
    """
    # For builtin functions inspect.getargspec(funct) return error. If builtin
    # function generates TypeError, it is because of wrong parameters.
    if not inspect.isfunction(func):
        return True

    if sys.version_info >= (3, 3):
        return is_invalid_params_py3(func, *args, **kwargs)
    else:
        # NOTE: use Python2 method for Python 3.2 as well. Starting from Python
        # 3.3 it is recommended to use inspect.signature instead.
        # In Python 3.0 - 3.2 inspect.getfullargspec is preferred but these
        # versions are almost not supported. Users should consider upgrading.
        return is_invalid_params_py2(func, *args, **kwargs)
