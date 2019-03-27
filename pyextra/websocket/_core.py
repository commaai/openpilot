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
from __future__ import print_function

import socket
import struct
import threading
import time

import six

# websocket modules
from ._abnf import *
from ._exceptions import *
from ._handshake import *
from ._http import *
from ._logging import *
from ._socket import *
from ._ssl_compat import *
from ._utils import *

__all__ = ['WebSocket', 'create_connection']

"""
websocket python client.
=========================

This version support only hybi-13.
Please see http://tools.ietf.org/html/rfc6455 for protocol.
"""


class WebSocket(object):
    """
    Low level WebSocket interface.
    This class is based on
      The WebSocket protocol draft-hixie-thewebsocketprotocol-76
      http://tools.ietf.org/html/draft-hixie-thewebsocketprotocol-76

    We can connect to the websocket server and send/receive data.
    The following example is an echo client.

    >>> import websocket
    >>> ws = websocket.WebSocket()
    >>> ws.connect("ws://echo.websocket.org")
    >>> ws.send("Hello, Server")
    >>> ws.recv()
    'Hello, Server'
    >>> ws.close()

    get_mask_key: a callable to produce new mask keys, see the set_mask_key
      function's docstring for more details
    sockopt: values for socket.setsockopt.
        sockopt must be tuple and each element is argument of sock.setsockopt.
    sslopt: dict object for ssl socket option.
    fire_cont_frame: fire recv event for each cont frame. default is False
    enable_multithread: if set to True, lock send method.
    skip_utf8_validation: skip utf8 validation.
    """

    def __init__(self, get_mask_key=None, sockopt=None, sslopt=None,
                 fire_cont_frame=False, enable_multithread=False,
                 skip_utf8_validation=False, **_):
        """
        Initialize WebSocket object.
        """
        self.sock_opt = sock_opt(sockopt, sslopt)
        self.handshake_response = None
        self.sock = None

        self.connected = False
        self.get_mask_key = get_mask_key
        # These buffer over the build-up of a single frame.
        self.frame_buffer = frame_buffer(self._recv, skip_utf8_validation)
        self.cont_frame = continuous_frame(
            fire_cont_frame, skip_utf8_validation)

        if enable_multithread:
            self.lock = threading.Lock()
            self.readlock = threading.Lock()
        else:
            self.lock = NoLock()
            self.readlock = NoLock()

    def __iter__(self):
        """
        Allow iteration over websocket, implying sequential `recv` executions.
        """
        while True:
            yield self.recv()

    def __next__(self):
        return self.recv()

    def next(self):
        return self.__next__()

    def fileno(self):
        return self.sock.fileno()

    def set_mask_key(self, func):
        """
        set function to create musk key. You can customize mask key generator.
        Mainly, this is for testing purpose.

        func: callable object. the func takes 1 argument as integer.
              The argument means length of mask key.
              This func must return string(byte array),
              which length is argument specified.
        """
        self.get_mask_key = func

    def gettimeout(self):
        """
        Get the websocket timeout(second).
        """
        return self.sock_opt.timeout

    def settimeout(self, timeout):
        """
        Set the timeout to the websocket.

        timeout: timeout time(second).
        """
        self.sock_opt.timeout = timeout
        if self.sock:
            self.sock.settimeout(timeout)

    timeout = property(gettimeout, settimeout)

    def getsubprotocol(self):
        """
        get subprotocol
        """
        if self.handshake_response:
            return self.handshake_response.subprotocol
        else:
            return None

    subprotocol = property(getsubprotocol)

    def getstatus(self):
        """
        get handshake status
        """
        if self.handshake_response:
            return self.handshake_response.status
        else:
            return None

    status = property(getstatus)

    def getheaders(self):
        """
        get handshake response header
        """
        if self.handshake_response:
            return self.handshake_response.headers
        else:
            return None

    def is_ssl(self):
        return isinstance(self.sock, ssl.SSLSocket)

    headers = property(getheaders)

    def connect(self, url, **options):
        """
        Connect to url. url is websocket url scheme.
        ie. ws://host:port/resource
        You can customize using 'options'.
        If you set "header" list object, you can set your own custom header.

        >>> ws = WebSocket()
        >>> ws.connect("ws://echo.websocket.org/",
                ...     header=["User-Agent: MyProgram",
                ...             "x-custom: header"])

        timeout: socket timeout time. This value is integer.
                 if you set None for this value,
                 it means "use default_timeout value"

        options: "header" -> custom http header list or dict.
                 "cookie" -> cookie value.
                 "origin" -> custom origin url.
                 "suppress_origin" -> suppress outputting origin header.
                 "host"   -> custom host header string.
                 "http_proxy_host" - http proxy host name.
                 "http_proxy_port" - http proxy port. If not set, set to 80.
                 "http_no_proxy"   - host names, which doesn't use proxy.
                 "http_proxy_auth" - http proxy auth information.
                                     tuple of username and password.
                                     default is None
                 "redirect_limit" -> number of redirects to follow.
                 "subprotocols" - array of available sub protocols.
                                  default is None.
                 "socket" - pre-initialized stream socket.

        """
        # FIXME: "subprotocols" are getting lost, not passed down
        # FIXME: "header", "cookie", "origin" and "host" too
        self.sock_opt.timeout = options.get('timeout', self.sock_opt.timeout)
        self.sock, addrs = connect(url, self.sock_opt, proxy_info(**options),
                                   options.pop('socket', None))

        try:
            self.handshake_response = handshake(self.sock, *addrs, **options)
            for attempt in range(options.pop('redirect_limit', 3)):
                if self.handshake_response.status in SUPPORTED_REDIRECT_STATUSES:
                    url = self.handshake_response.headers['location']
                    self.sock.close()
                    self.sock, addrs =  connect(url, self.sock_opt, proxy_info(**options),
                                                options.pop('socket', None))
                    self.handshake_response = handshake(self.sock, *addrs, **options)
            self.connected = True
        except:
            if self.sock:
                self.sock.close()
                self.sock = None
            raise

    def send(self, payload, opcode=ABNF.OPCODE_TEXT):
        """
        Send the data as string.

        payload: Payload must be utf-8 string or unicode,
                  if the opcode is OPCODE_TEXT.
                  Otherwise, it must be string(byte array)

        opcode: operation code to send. Please see OPCODE_XXX.
        """

        frame = ABNF.create_frame(payload, opcode)
        return self.send_frame(frame)

    def send_frame(self, frame):
        """
        Send the data frame.

        frame: frame data created  by ABNF.create_frame

        >>> ws = create_connection("ws://echo.websocket.org/")
        >>> frame = ABNF.create_frame("Hello", ABNF.OPCODE_TEXT)
        >>> ws.send_frame(frame)
        >>> cont_frame = ABNF.create_frame("My name is ", ABNF.OPCODE_CONT, 0)
        >>> ws.send_frame(frame)
        >>> cont_frame = ABNF.create_frame("Foo Bar", ABNF.OPCODE_CONT, 1)
        >>> ws.send_frame(frame)

        """
        if self.get_mask_key:
            frame.get_mask_key = self.get_mask_key
        data = frame.format()
        length = len(data)
        trace("send: " + repr(data))

        with self.lock:
            while data:
                l = self._send(data)
                data = data[l:]

        return length

    def send_binary(self, payload):
        return self.send(payload, ABNF.OPCODE_BINARY)

    def ping(self, payload=""):
        """
        send ping data.

        payload: data payload to send server.
        """
        if isinstance(payload, six.text_type):
            payload = payload.encode("utf-8")
        self.send(payload, ABNF.OPCODE_PING)

    def pong(self, payload):
        """
        send pong data.

        payload: data payload to send server.
        """
        if isinstance(payload, six.text_type):
            payload = payload.encode("utf-8")
        self.send(payload, ABNF.OPCODE_PONG)

    def recv(self):
        """
        Receive string data(byte array) from the server.

        return value: string(byte array) value.
        """
        with self.readlock:
            opcode, data = self.recv_data()
        if six.PY3 and opcode == ABNF.OPCODE_TEXT:
            return data.decode("utf-8")
        elif opcode == ABNF.OPCODE_TEXT or opcode == ABNF.OPCODE_BINARY:
            return data
        else:
            return ''

    def recv_data(self, control_frame=False):
        """
        Receive data with operation code.

        control_frame: a boolean flag indicating whether to return control frame
        data, defaults to False

        return  value: tuple of operation code and string(byte array) value.
        """
        opcode, frame = self.recv_data_frame(control_frame)
        return opcode, frame.data

    def recv_data_frame(self, control_frame=False):
        """
        Receive data with operation code.

        control_frame: a boolean flag indicating whether to return control frame
        data, defaults to False

        return  value: tuple of operation code and string(byte array) value.
        """
        while True:
            frame = self.recv_frame()
            if not frame:
                # handle error:
                # 'NoneType' object has no attribute 'opcode'
                raise WebSocketProtocolException(
                    "Not a valid frame %s" % frame)
            elif frame.opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY, ABNF.OPCODE_CONT):
                self.cont_frame.validate(frame)
                self.cont_frame.add(frame)

                if self.cont_frame.is_fire(frame):
                    return self.cont_frame.extract(frame)

            elif frame.opcode == ABNF.OPCODE_CLOSE:
                self.send_close()
                return frame.opcode, frame
            elif frame.opcode == ABNF.OPCODE_PING:
                if len(frame.data) < 126:
                    self.pong(frame.data)
                else:
                    raise WebSocketProtocolException(
                        "Ping message is too long")
                if control_frame:
                    return frame.opcode, frame
            elif frame.opcode == ABNF.OPCODE_PONG:
                if control_frame:
                    return frame.opcode, frame

    def recv_frame(self):
        """
        receive data as frame from server.

        return value: ABNF frame object.
        """
        return self.frame_buffer.recv_frame()

    def send_close(self, status=STATUS_NORMAL, reason=six.b("")):
        """
        send close data to the server.

        status: status code to send. see STATUS_XXX.

        reason: the reason to close. This must be string or bytes.
        """
        if status < 0 or status >= ABNF.LENGTH_16:
            raise ValueError("code is invalid range")
        self.connected = False
        self.send(struct.pack('!H', status) + reason, ABNF.OPCODE_CLOSE)

    def close(self, status=STATUS_NORMAL, reason=six.b(""), timeout=3):
        """
        Close Websocket object

        status: status code to send. see STATUS_XXX.

        reason: the reason to close. This must be string.

        timeout: timeout until receive a close frame.
            If None, it will wait forever until receive a close frame.
        """
        if self.connected:
            if status < 0 or status >= ABNF.LENGTH_16:
                raise ValueError("code is invalid range")

            try:
                self.connected = False
                self.send(struct.pack('!H', status) +
                          reason, ABNF.OPCODE_CLOSE)
                sock_timeout = self.sock.gettimeout()
                self.sock.settimeout(timeout)
                start_time = time.time()
                while timeout is None or time.time() - start_time < timeout:
                    try:
                        frame = self.recv_frame()
                        if frame.opcode != ABNF.OPCODE_CLOSE:
                            continue
                        if isEnabledForError():
                            recv_status = struct.unpack("!H", frame.data[0:2])[0]
                            if recv_status != STATUS_NORMAL:
                                error("close status: " + repr(recv_status))
                        break
                    except:
                        break
                self.sock.settimeout(sock_timeout)
                self.sock.shutdown(socket.SHUT_RDWR)
            except:
                pass

            self.shutdown()

    def abort(self):
        """
        Low-level asynchronous abort, wakes up other threads that are waiting in recv_*
        """
        if self.connected:
            self.sock.shutdown(socket.SHUT_RDWR)

    def shutdown(self):
        """close socket, immediately."""
        if self.sock:
            self.sock.close()
            self.sock = None
            self.connected = False

    def _send(self, data):
        return send(self.sock, data)

    def _recv(self, bufsize):
        try:
            return recv(self.sock, bufsize)
        except WebSocketConnectionClosedException:
            if self.sock:
                self.sock.close()
            self.sock = None
            self.connected = False
            raise


def create_connection(url, timeout=None, class_=WebSocket, **options):
    """
    connect to url and return websocket object.

    Connect to url and return the WebSocket object.
    Passing optional timeout parameter will set the timeout on the socket.
    If no timeout is supplied,
    the global default timeout setting returned by getdefauttimeout() is used.
    You can customize using 'options'.
    If you set "header" list object, you can set your own custom header.

    >>> conn = create_connection("ws://echo.websocket.org/",
         ...     header=["User-Agent: MyProgram",
         ...             "x-custom: header"])


    timeout: socket timeout time. This value is integer.
             if you set None for this value,
             it means "use default_timeout value"

    class_: class to instantiate when creating the connection. It has to implement
            settimeout and connect. It's __init__ should be compatible with
            WebSocket.__init__, i.e. accept all of it's kwargs.
    options: "header" -> custom http header list or dict.
             "cookie" -> cookie value.
             "origin" -> custom origin url.
             "suppress_origin" -> suppress outputting origin header.
             "host"   -> custom host header string.
             "http_proxy_host" - http proxy host name.
             "http_proxy_port" - http proxy port. If not set, set to 80.
             "http_no_proxy"   - host names, which doesn't use proxy.
             "http_proxy_auth" - http proxy auth information.
                                    tuple of username and password.
                                    default is None
             "enable_multithread" -> enable lock for multithread.
             "redirect_limit" -> number of redirects to follow.
             "sockopt" -> socket options
             "sslopt" -> ssl option
             "subprotocols" - array of available sub protocols.
                              default is None.
             "skip_utf8_validation" - skip utf8 validation.
             "socket" - pre-initialized stream socket.
    """
    sockopt = options.pop("sockopt", [])
    sslopt = options.pop("sslopt", {})
    fire_cont_frame = options.pop("fire_cont_frame", False)
    enable_multithread = options.pop("enable_multithread", False)
    skip_utf8_validation = options.pop("skip_utf8_validation", False)
    websock = class_(sockopt=sockopt, sslopt=sslopt,
                     fire_cont_frame=fire_cont_frame,
                     enable_multithread=enable_multithread,
                     skip_utf8_validation=skip_utf8_validation, **options)
    websock.settimeout(timeout if timeout is not None else getdefaulttimeout())
    websock.connect(url, **options)
    return websock
