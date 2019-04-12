from . import six

from .base import JSONRPCBaseRequest, JSONRPCBaseResponse
from .exceptions import JSONRPCInvalidRequestException, JSONRPCError


class JSONRPC10Request(JSONRPCBaseRequest):

    """ JSON-RPC 1.0 Request.

    A remote method is invoked by sending a request to a remote service.
    The request is a single object serialized using json.

    :param str method: The name of the method to be invoked.
    :param list params: An Array of objects to pass as arguments to the method.
    :param _id: This can be of any type. It is used to match the response with
        the request that it is replying to.
    :param bool is_notification: whether request notification or not.

    """

    JSONRPC_VERSION = "1.0"
    REQUIRED_FIELDS = set(["method", "params", "id"])
    POSSIBLE_FIELDS = set(["method", "params", "id"])

    @property
    def data(self):
        data = dict((k, v) for k, v in self._data.items())
        data["id"] = None if self.is_notification else data["id"]
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

        self._data["method"] = str(value)

    @property
    def params(self):
        return self._data.get("params")

    @params.setter
    def params(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("Incorrect params {0}".format(value))

        self._data["params"] = list(value)

    @property
    def _id(self):
        return self._data.get("id")

    @_id.setter
    def _id(self, value):
        self._data["id"] = value

    @property
    def is_notification(self):
        return self._data["id"] is None or self._is_notification

    @is_notification.setter
    def is_notification(self, value):
        if value is None:
            value = self._id is None

        if self._id is None and not value:
            raise ValueError("Can not set attribute is_notification. " +
                             "Request id should not be None")

        self._is_notification = value

    @classmethod
    def from_json(cls, json_str):
        data = cls.deserialize(json_str)
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data):
        if not isinstance(data, dict):
            raise ValueError("data should be dict")

        if cls.REQUIRED_FIELDS <= set(data.keys()) <= cls.POSSIBLE_FIELDS:
            return cls(
                method=data["method"], params=data["params"], _id=data["id"]
            )
        else:
            extra = set(data.keys()) - cls.POSSIBLE_FIELDS
            missed = cls.REQUIRED_FIELDS - set(data.keys())
            msg = "Invalid request. Extra fields: {0}, Missed fields: {1}"
            raise JSONRPCInvalidRequestException(msg.format(extra, missed))


class JSONRPC10Response(JSONRPCBaseResponse):

    JSONRPC_VERSION = "1.0"

    @property
    def data(self):
        data = dict((k, v) for k, v in self._data.items())
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
        if value is None:
            raise ValueError("id could not be null for JSON-RPC1.0 Response")
        self._data["id"] = value
