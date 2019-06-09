from .utils import JSONSerializable


class JSONRPCBaseRequest(JSONSerializable):

    """ Base class for JSON-RPC 1.0 and JSON-RPC 2.0 requests."""

    def __init__(self, method=None, params=None, _id=None,
                 is_notification=None):
        self.data = dict()
        self.method = method
        self.params = params
        self._id = _id
        self.is_notification = is_notification

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, dict):
            raise ValueError("data should be dict")

        self._data = value

    @property
    def args(self):
        """ Method position arguments.

        :return tuple args: method position arguments.

        """
        return tuple(self.params) if isinstance(self.params, list) else ()

    @property
    def kwargs(self):
        """ Method named arguments.

        :return dict kwargs: method named arguments.

        """
        return self.params if isinstance(self.params, dict) else {}

    @property
    def json(self):
        return self.serialize(self.data)


class JSONRPCBaseResponse(JSONSerializable):

    """ Base class for JSON-RPC 1.0 and JSON-RPC 2.0 responses."""

    def __init__(self, **kwargs):
        self.data = dict()

        try:
            self.result = kwargs['result']
        except KeyError:
            pass

        try:
            self.error = kwargs['error']
        except KeyError:
            pass

        self._id = kwargs.get('_id')

        if 'result' not in kwargs and 'error' not in kwargs:
            raise ValueError("Either result or error should be used")

        self.request = None  # type: JSONRPCBaseRequest

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, dict):
            raise ValueError("data should be dict")

        self._data = value

    @property
    def json(self):
        return self.serialize(self.data)
