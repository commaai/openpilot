"""
websocket - WebSocket client library for Python

Copyright (C) 2010 Hiroki Ohtani(liris)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA  02110-1335  USA

"""


"""
define websocket exceptions
"""


class WebSocketException(Exception):
    """
    websocket exception class.
    """
    pass


class WebSocketProtocolException(WebSocketException):
    """
    If the websocket protocol is invalid, this exception will be raised.
    """
    pass


class WebSocketPayloadException(WebSocketException):
    """
    If the websocket payload is invalid, this exception will be raised.
    """
    pass


class WebSocketConnectionClosedException(WebSocketException):
    """
    If remote host closed the connection or some network error happened,
    this exception will be raised.
    """
    pass


class WebSocketTimeoutException(WebSocketException):
    """
    WebSocketTimeoutException will be raised at socket timeout during read/write data.
    """
    pass


class WebSocketProxyException(WebSocketException):
    """
    WebSocketProxyException will be raised when proxy error occurred.
    """
    pass


class WebSocketBadStatusException(WebSocketException):
    """
    WebSocketBadStatusException will be raised when we get bad handshake status code.
    """

    def __init__(self, message, status_code, status_message=None, resp_headers=None):
        msg = message % (status_code, status_message)
        super(WebSocketBadStatusException, self).__init__(msg)
        self.status_code = status_code
        self.resp_headers = resp_headers

class WebSocketAddressException(WebSocketException):
    """
    If the websocket address info cannot be found, this exception will be raised.
    """
    pass
