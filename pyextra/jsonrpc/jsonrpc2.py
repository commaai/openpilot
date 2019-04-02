from . import six
import json

from .exceptions import JSONRPCError, JSONRPCInvalidRequestException
from .base import JSONRPCBaseRequest, JSONRPCBaseResponse


class JSONRPC20Request(JSONRPCBaseRequest):

    """ A rpc call is represented by sending a Request object to a Server.

    :param str method: A String containing the name of the method to be
        invoked. Method names that begin with the word rpc followed by a
        period character (U+002E or ASCII 46) are reserved for rpc-internal
        methods and extensions and MUST NOT be used for anything else.

    :param params: A Structured value that holds the parameter values to be
        used during the invocation of the method. This member MAY be omitted.
    :type params: iterable or dict

    :param _id: An identifier established by the Client that MUST contain a
        String, Number, or NULL value if included. If it is not included it is
        assumed to be a notification. The value SHOULD normally not be Null
        [1] and Numbers SHOULD NOT contain fractional parts [2].
    :type _id: str or int or None

    :param bool is_notification: Whether request is notification or not. If
        value is True, _id is not included to request. It allows to create
        requests with id = null.

    The Server MUST reply with the same value in the Response object if
    included. This member is used to correlate the context between the two
    objects.

    [1] The use of Null as a value for the id member in a Request object is
    discouraged, because this specification uses a value of Null for Responses
    with an unknown id. Also, because JSON-RPC 1.0 uses an id value of Null
    for Notifications this could cause confusion in handling.

    [2] Fractional parts may be problematic, since many decimal fractions
    cannot be represented exactly as binary fractions.

    """

    JSONRPC_VERSION = "2.0"
    REQUIRED_FIELDS = set(["jsonrpc", "method"])
    POSSIBLE_FIELDS = set(["jsonrpc", "method", "params", "id"])

    @property
    def data(self):
        data = dict(
            (k, v) for k, v in self._data.items()
            if not (k == "id" and self.is_notification)
        )
        data["jsonrpc"] = self.JSONRPC_VERSION
        return data

    @data.setter
    def data(self, value):
        if not isinstance(value, dict):
            raise ValueError("data should be dict")

        self._data = value

    @property
    def method(self):
        return self._data.get("method")

    @method.setter
    def method(self, value):
        if not isinstance(value, six.string_types):
            raise ValueError("Method should be string")

        if value.startswith("rpc."):
            raise ValueError(
                "Method names that begin with the word rpc followed by a " +
                "period character (U+002E or ASCII 46) are reserved for " +
                "rpc-internal methods and extensions and MUST NOT be used " +
                "for anything else.")

        self._data["method"] = str(value)

    @property
    def params(self):
        return self._data.get("params")

    @params.setter
    def params(self, value):
        if value is not None and not isinstance(value, (list, tuple, dict)):
            raise ValueError("Incorrect params {0}".format(value))

        value = list(value) if isinstance(value, tuple) else value

        if value is not None:
            self._data["params"] = value

    @property
    def _id(self):
        return self._data.get("id")

    @_id.setter
    def _id(self, value):
        if value is not None and \
           not isinstance(value, six.string_types + six.integer_types):
            raise ValueError("id should be string or integer")

        self._data["id"] = value

    @classmethod
    def from_json(cls, json_str):
        data = cls.deserialize(json_str)
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data):
        is_batch = isinstance(data, list)
        data = data if is_batch else [data]

        if not data:
            raise JSONRPCInvalidRequestException("[] value is not accepted")

        if not all(isinstance(d, dict) for d in data):
            raise JSONRPCInvalidRequestException(
                "Each request should be an object (dict)")

        result = []
        for d in data:
            if not cls.REQUIRED_FIELDS <= set(d.keys()) <= cls.POSSIBLE_FIELDS:
                extra = set(d.keys()) - cls.POSSIBLE_FIELDS
                missed = cls.REQUIRED_FIELDS - set(d.keys())
                msg = "Invalid request. Extra fields: {0}, Missed fields: {1}"
                raise JSONRPCInvalidRequestException(msg.format(extra, missed))

            try:
                result.append(JSONRPC20Request(
                    method=d["method"], params=d.get("params"),
                    _id=d.get("id"), is_notification="id" not in d,
                ))
            except ValueError as e:
                raise JSONRPCInvalidRequestException(str(e))

        return JSONRPC20BatchRequest(*result) if is_batch else result[0]


class JSONRPC20BatchRequest(object):

    """ Batch JSON-RPC 2.0 Request.

    :param JSONRPC20Request *requests: requests

    """

    JSONRPC_VERSION = "2.0"

    def __init__(self, *requests):
        self.requests = requests

    @classmethod
    def from_json(cls, json_str):
        return JSONRPC20Request.from_json(json_str)

    @property
    def json(self):
        return json.dumps([r.data for r in self.requests])

    def __iter__(self):
        return iter(self.requests)


class JSONRPC20Response(JSONRPCBaseResponse):

    """ JSON-RPC response object to JSONRPC20Request.

    When a rpc call is made, the Server MUST reply with a Response, except for
    in the case of Notifications. The Response is expressed as a single JSON
    Object, with the following members:

    :param str jsonrpc: A String specifying the version of the JSON-RPC
        protocol. MUST be exactly "2.0".

    :param result: This member is REQUIRED on success.
        This member MUST NOT exist if there was an error invoking the method.
        The value of this member is determined by the method invoked on the
        Server.

    :param dict error: This member is REQUIRED on error.
        This member MUST NOT exist if there was no error triggered during
        invocation. The value for this member MUST be an Object.

    :param id: This member is REQUIRED.
        It MUST be the same as the value of the id member in the Request
        Object. If there was an error in detecting the id in the Request
        object (e.g. Parse error/Invalid Request), it MUST be Null.
    :type id: str or int or None

    Either the result member or error member MUST be included, but both
    members MUST NOT be included.

    """

    JSONRPC_VERSION = "2.0"

    @property
    def data(self):
        data = dict((k, v) for k, v in self._data.items())
        data["jsonrpc"] = self.JSONRPC_VERSION
        return data

    @data.setter
    def data(self, value):
        if not isinstance(value, dict):
            raise ValueError("data should be dict")
        self._data = value

    @property
    def result(self):
        return self._data.get("result")

    @result.setter
    def result(self, value):
        if self.error:
            raise ValueError("Either result or error should be used")
        self._data["result"] = value

    @property
    def error(self):
        return self._data.get("error")

    @error.setter
    def error(self, value):
        self._data.pop('value', None)
        if value:
            self._data["error"] = value
            # Test error
            JSONRPCError(**value)

    @property
    def _id(self):
        return self._data.get("id")

    @_id.setter
    def _id(self, value):
        if value is not None and \
           not isinstance(value, six.string_types + six.integer_types):
            raise ValueError("id should be string or integer")

        self._data["id"] = value


class JSONRPC20BatchResponse(object):

    JSONRPC_VERSION = "2.0"

    def __init__(self, *responses):
        self.responses = responses
        self.request = None  # type: JSONRPC20BatchRequest

    @property
    def data(self):
        return [r.data for r in self.responses]

    @property
    def json(self):
        return json.dumps(self.data)

    def __iter__(self):
        return iter(self.responses)
