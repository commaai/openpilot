from __future__ import absolute_import

from django.views.decorators.csrf import csrf_exempt
from django.conf.urls import url
from django.conf import settings
from django.http import HttpResponse, HttpResponseNotAllowed
import copy
import json
import logging
import time

from ..exceptions import JSONRPCInvalidRequestException
from ..jsonrpc import JSONRPCRequest
from ..manager import JSONRPCResponseManager
from ..utils import DatetimeDecimalEncoder
from ..dispatcher import Dispatcher


logger = logging.getLogger(__name__)


def response_serialize(obj):
    """ Serializes response's data object to JSON. """
    return json.dumps(obj, cls=DatetimeDecimalEncoder)


class JSONRPCAPI(object):
    def __init__(self, dispatcher=None):
        self.dispatcher = dispatcher if dispatcher is not None \
            else Dispatcher()

    @property
    def urls(self):
        urls = [
            url(r'^$', self.jsonrpc, name='endpoint'),
        ]

        if getattr(settings, 'JSONRPC_MAP_VIEW_ENABLED', settings.DEBUG):
            urls.append(
                url(r'^map$', self.jsonrpc_map, name='map')
            )

        return urls

    @csrf_exempt
    def jsonrpc(self, request):
        """ JSON-RPC 2.0 handler."""
        if request.method != "POST":
            return HttpResponseNotAllowed(["POST"])

        request_str = request.body.decode('utf8')
        try:
            jsonrpc_request = JSONRPCRequest.from_json(request_str)
        except (TypeError, ValueError, JSONRPCInvalidRequestException):
            response = JSONRPCResponseManager.handle(
                request_str, self.dispatcher)
        else:
            jsonrpc_request.params = jsonrpc_request.params or {}
            jsonrpc_request_params = copy.copy(jsonrpc_request.params)
            if isinstance(jsonrpc_request.params, dict):
                jsonrpc_request.params.update(request=request)

            t1 = time.time()
            response = JSONRPCResponseManager.handle_request(
                jsonrpc_request, self.dispatcher)
            t2 = time.time()
            logger.info('{0}({1}) {2:.2f} sec'.format(
                jsonrpc_request.method, jsonrpc_request_params, t2 - t1))

        if response:
            response.serialize = response_serialize
            response = response.json

        return HttpResponse(response, content_type="application/json")

    def jsonrpc_map(self, request):
        """ Map of json-rpc available calls.

        :return str:

        """
        result = "<h1>JSON-RPC map</h1><pre>{0}</pre>".format("\n\n".join([
            "{0}: {1}".format(fname, f.__doc__)
            for fname, f in self.dispatcher.items()
        ]))
        return HttpResponse(result)


api = JSONRPCAPI()
