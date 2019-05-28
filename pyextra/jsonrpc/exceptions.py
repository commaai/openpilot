""" JSON-RPC Exceptions."""
from . import six
import json


class JSONRPCError(object):

    """ Error for JSON-RPC communication.

    When a rpc call encounters an error, the Response Object MUST contain the
    error member with a value that is a Object with the following members:

    Parameters
    ----------
    code: int
        A Number that indicates the error type that occurred.
        This MUST be an integer.
        The error codes from and including -32768 to -32000 are reserved for
        pre-defined errors. Any code within this range, but not defined
        explicitly below is reserved for future use. The error codes are nearly
        the same as those suggested for XML-RPC at the following
        url: http://xmlrpc-epi.sourceforge.net/specs/rfc.fault_codes.php

    message: str
        A String providing a short description of the error.
        The message SHOULD be limited to a concise single sentence.

    data: int or str or dict or list, optional
        A Primitive or Structured value that contains additional
        information about the error.
        This may be omitted.
        The value of this member is defined by the Server (e.g. detailed error
        information, nested errors etc.).

    """

    serialize = staticmethod(json.dumps)
    deserialize = staticmethod(json.loads)

    def __init__(self, code=None, message=None, data=None):
        self._data = dict()
        self.code = getattr(self.__class__, "CODE", code)
        self.message = getattr(self.__class__, "MESSAGE", message)
        self.data = data

    def __get_code(self):
        return self._data["code"]

    def __set_code(self, value):
        if not isinstance(value, six.integer_types):
            raise ValueError("Error code should be integer")

        self._data["code"] = value

    code = property(__get_code, __set_code)

    def __get_message(self):
        return self._data["message"]

    def __set_message(self, value):
        if not isinstance(value, six.string_types):
            raise ValueError("Error message should be string")

        self._data["message"] = value

    message = property(__get_message, __set_message)

    def __get_data(self):
        return self._data.get("data")

    def __set_data(self, value):
        if value is not None:
            self._data["data"] = value

    data = property(__get_data, __set_data)

    @classmethod
    def from_json(cls, json_str):
        data = cls.deserialize(json_str)
        return cls(
            code=data["code"], message=data["message"], data=data.get("data"))

    @property
    def json(self):
        return self.serialize(self._data)


class JSONRPCParseError(JSONRPCError):

    """ Parse Error.

    Invalid JSON was received by the server.
    An error occurred on the server while parsing the JSON text.

    """

    CODE = -32700
    MESSAGE = "Parse error"


class JSONRPCInvalidRequest(JSONRPCError):

    """ Invalid Request.

    The JSON sent is not a valid Request object.

    """

    CODE = -32600
    MESSAGE = "Invalid Request"


class JSONRPCMethodNotFound(JSONRPCError):

    """ Method not found.

    The method does not exist / is not available.

    """

    CODE = -32601
    MESSAGE = "Method not found"


class JSONRPCInvalidParams(JSONRPCError):

    """ Invalid params.

    Invalid method parameter(s).

    """

    CODE = -32602
    MESSAGE = "Invalid params"


class JSONRPCInternalError(JSONRPCError):

    """ Internal error.

    Internal JSON-RPC error.

    """

    CODE = -32603
    MESSAGE = "Internal error"


class JSONRPCServerError(JSONRPCError):

    """ Server error.

    Reserved for implementation-defined server-errors.

    """

    CODE = -32000
    MESSAGE = "Server error"


class JSONRPCException(Exception):

    """ JSON-RPC Exception."""

    pass


class JSONRPCInvalidRequestException(JSONRPCException):

    """ Request is not valid."""

    pass


class JSONRPCDispatchException(JSONRPCException):

    """ JSON-RPC Dispatch Exception.

    Should be thrown in dispatch methods.

    """

    def __init__(self, code=None, message=None, data=None, *args, **kwargs):
        super(JSONRPCDispatchException, self).__init__(args, kwargs)
        self.error = JSONRPCError(code=code, data=data, message=message)
