from __future__ import absolute_import

import copy
import json
import logging
import time
from uuid import uuid4

from flask import Blueprint, request, Response

from ..exceptions import JSONRPCInvalidRequestException
from ..jsonrpc import JSONRPCRequest
from ..manager import JSONRPCResponseManager
from ..utils import DatetimeDecimalEncoder
from ..dispatcher import Dispatcher


logger = logging.getLogger(__name__)


class JSONRPCAPI(object):
    def __init__(self, dispatcher=None, check_content_type=True):
        """

        :param dispatcher: methods dispatcher
        :param check_content_type: if True - content-type must be
            "application/json"
        :return:

        """
        self.dispatcher = dispatcher if dispatcher is not None \
            else Dispatcher()
        self.check_content_type = check_content_type

    def as_blueprint(self, name=None):
        blueprint = Blueprint(name if name else str(uuid4()), __name__)
        blueprint.add_url_rule(
            '/', view_func=self.jsonrpc, methods=['POST'])
        blueprint.add_url_rule(
            '/map', view_func=self.jsonrpc_map, methods=['GET'])
        return blueprint

    def as_view(self):
        return self.jsonrpc

    def jsonrpc(self):
        request_str = self._get_request_str()
        try:
            jsonrpc_request = JSONRPCRequest.from_json(request_str)
        except (TypeError, ValueError, JSONRPCInvalidRequestException):
            response = JSONRPCResponseManager.handle(
                request_str, self.dispatcher)
        else:
            response = JSONRPCResponseManager.handle_request(
                jsonrpc_request, self.dispatcher)

        if response:
            response.serialize = self._serialize
            response = response.json

        return Response(response, content_type="application/json")

    def jsonrpc_map(self):
        """ Map of json-rpc available calls.

        :return str:

        """
        result = "<h1>JSON-RPC map</h1><pre>{0}</pre>".format("\n\n".join([
            "{0}: {1}".format(fname, f.__doc__)
            for fname, f in self.dispatcher.items()
        ]))
        return Response(result)

    def _get_request_str(self):
        if self.check_content_type or request.data:
            return request.data
        return list(request.form.keys())[0]

    @staticmethod
    def _serialize(s):
        return json.dumps(s, cls=DatetimeDecimalEncoder)


api = JSONRPCAPI()
