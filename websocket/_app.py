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
WebSocketApp provides higher level APIs.
"""
import inspect
import select
import sys
import threading
import time
import traceback

import six

from ._abnf import ABNF
from ._core import WebSocket, getdefaulttimeout
from ._exceptions import *
from . import _logging


__all__ = ["WebSocketApp"]

class Dispatcher:
    def __init__(self, app, ping_timeout):
        self.app  = app
        self.ping_timeout = ping_timeout

    def read(self, sock, read_callback, check_callback):
        while self.app.sock.connected:
            r, w, e = select.select(
                    (self.app.sock.sock, ), (), (), self.ping_timeout)
            if r:
                if not read_callback():
                    break
            check_callback()

class SSLDispacther:
    def __init__(self, app, ping_timeout):
        self.app  = app
        self.ping_timeout = ping_timeout

    def read(self, sock, read_callback, check_callback):
        while self.app.sock.connected:
            r = self.select()
            if r:
                if not read_callback():
                    break
            check_callback()

    def select(self):
        sock = self.app.sock.sock
        if sock.pending():
            return [sock,]

        r, w, e = select.select((sock, ), (), (), self.ping_timeout)
        return r

class WebSocketApp(object):
    """
    Higher level of APIs are provided.
    The interface is like JavaScript WebSocket object.
    """

    def __init__(self, url, header=None,
                 on_open=None, on_message=None, on_error=None,
                 on_close=None, on_ping=None, on_pong=None,
                 on_cont_message=None,
                 keep_running=True, get_mask_key=None, cookie=None,
                 subprotocols=None,
                 on_data=None):
        """
        url: websocket url.
        header: custom header for websocket handshake.
        on_open: callable object which is called at opening websocket.
          this function has one argument. The argument is this class object.
        on_message: callable object which is called when received data.
         on_message has 2 arguments.
         The 1st argument is this class object.
         The 2nd argument is utf-8 string which we get from the server.
        on_error: callable object which is called when we get error.
         on_error has 2 arguments.
         The 1st argument is this class object.
         The 2nd argument is exception object.
        on_close: callable object which is called when closed the connection.
         this function has one argument. The argument is this class object.
        on_cont_message: callback object which is called when receive continued
         frame data.
         on_cont_message has 3 arguments.
         The 1st argument is this class object.
         The 2nd argument is utf-8 string which we get from the server.
         The 3rd argument is continue flag. if 0, the data continue
         to next frame data
        on_data: callback object which is called when a message received.
          This is called before on_message or on_cont_message,
          and then on_message or on_cont_message is called.
          on_data has 4 argument.
          The 1st argument is this class object.
          The 2nd argument is utf-8 string which we get from the server.
          The 3rd argument is data type. ABNF.OPCODE_TEXT or ABNF.OPCODE_BINARY will be came.
          The 4th argument is continue flag. if 0, the data continue
        keep_running: this parameter is obsolete and ignored.
        get_mask_key: a callable to produce new mask keys,
          see the WebSocket.set_mask_key's docstring for more information
        subprotocols: array of available sub protocols. default is None.
        """
        self.url = url
        self.header = header if header is not None else []
        self.cookie = cookie

        self.on_open = on_open
        self.on_message = on_message
        self.on_data = on_data
        self.on_error = on_error
        self.on_close = on_close
        self.on_ping = on_ping
        self.on_pong = on_pong
        self.on_cont_message = on_cont_message
        self.keep_running = False
        self.get_mask_key = get_mask_key
        self.sock = None
        self.last_ping_tm = 0
        self.last_pong_tm = 0
        self.subprotocols = subprotocols

    def send(self, data, opcode=ABNF.OPCODE_TEXT):
        """
        send message.
        data: message to send. If you set opcode to OPCODE_TEXT,
              data must be utf-8 string or unicode.
        opcode: operation code of data. default is OPCODE_TEXT.
        """

        if not self.sock or self.sock.send(data, opcode) == 0:
            raise WebSocketConnectionClosedException(
                "Connection is already closed.")

    def close(self, **kwargs):
        """
        close websocket connection.
        """
        self.keep_running = False
        if self.sock:
            self.sock.close(**kwargs)
            self.sock = None

    def _send_ping(self, interval, event):
        while not event.wait(interval):
            self.last_ping_tm = time.time()
            if self.sock:
                try:
                    self.sock.ping()
                except Exception as ex:
                    _logging.warning("send_ping routine terminated: {}".format(ex))
                    break

    def run_forever(self, sockopt=None, sslopt=None,
                    ping_interval=0, ping_timeout=None,
                    http_proxy_host=None, http_proxy_port=None,
                    http_no_proxy=None, http_proxy_auth=None,
                    skip_utf8_validation=False,
                    host=None, origin=None, dispatcher=None,
                    suppress_origin = False, proxy_type=None):
        """
        run event loop for WebSocket framework.
        This loop is infinite loop and is alive during websocket is available.
        sockopt: values for socket.setsockopt.
            sockopt must be tuple
            and each element is argument of sock.setsockopt.
        sslopt: ssl socket optional dict.
        ping_interval: automatically send "ping" command
            every specified period(second)
            if set to 0, not send automatically.
        ping_timeout: timeout(second) if the pong message is not received.
        http_proxy_host: http proxy host name.
        http_proxy_port: http proxy port. If not set, set to 80.
        http_no_proxy: host names, which doesn't use proxy.
        skip_utf8_validation: skip utf8 validation.
        host: update host header.
        origin: update origin header.
        dispatcher: customize reading data from socket.
        suppress_origin: suppress outputting origin header.

        Returns
        -------
        False if caught KeyboardInterrupt
        True if other exception was raised during a loop
        """

        if ping_timeout is not None and ping_timeout <= 0:
            ping_timeout = None
        if ping_timeout and ping_interval and ping_interval <= ping_timeout:
            raise WebSocketException("Ensure ping_interval > ping_timeout")
        if not sockopt:
            sockopt = []
        if not sslopt:
            sslopt = {}
        if self.sock:
            raise WebSocketException("socket is already opened")
        thread = None
        self.keep_running = True
        self.last_ping_tm = 0
        self.last_pong_tm = 0

        def teardown(close_frame=None):
            """
            Tears down the connection.
            If close_frame is set, we will invoke the on_close handler with the
            statusCode and reason from there.
            """
            if thread and thread.isAlive():
                event.set()
                thread.join()
            self.keep_running = False
            if self.sock:
                self.sock.close()
            close_args = self._get_close_args(
                close_frame.data if close_frame else None)
            self._callback(self.on_close, *close_args)
            self.sock = None

        try:
            self.sock = WebSocket(
                self.get_mask_key, sockopt=sockopt, sslopt=sslopt,
                fire_cont_frame=self.on_cont_message is not None,
                skip_utf8_validation=skip_utf8_validation,
                enable_multithread=True if ping_interval else False)
            self.sock.settimeout(getdefaulttimeout())
            self.sock.connect(
                self.url, header=self.header, cookie=self.cookie,
                http_proxy_host=http_proxy_host,
                http_proxy_port=http_proxy_port, http_no_proxy=http_no_proxy,
                http_proxy_auth=http_proxy_auth, subprotocols=self.subprotocols,
                host=host, origin=origin, suppress_origin=suppress_origin,
                proxy_type=proxy_type)
            if not dispatcher:
                dispatcher = self.create_dispatcher(ping_timeout)

            self._callback(self.on_open)

            if ping_interval:
                event = threading.Event()
                thread = threading.Thread(
                    target=self._send_ping, args=(ping_interval, event))
                thread.setDaemon(True)
                thread.start()

            def read():
                if not self.keep_running:
                    return teardown()

                op_code, frame = self.sock.recv_data_frame(True)
                if op_code == ABNF.OPCODE_CLOSE:
                    return teardown(frame)
                elif op_code == ABNF.OPCODE_PING:
                    self._callback(self.on_ping, frame.data)
                elif op_code == ABNF.OPCODE_PONG:
                    self.last_pong_tm = time.time()
                    self._callback(self.on_pong, frame.data)
                elif op_code == ABNF.OPCODE_CONT and self.on_cont_message:
                    self._callback(self.on_data, frame.data,
                                   frame.opcode, frame.fin)
                    self._callback(self.on_cont_message,
                                   frame.data, frame.fin)
                else:
                    data = frame.data
                    if six.PY3 and op_code == ABNF.OPCODE_TEXT:
                        data = data.decode("utf-8")
                    self._callback(self.on_data, data, frame.opcode, True)
                    self._callback(self.on_message, data)

                return True

            def check():
                if (ping_timeout):
                    has_timeout_expired = time.time() - self.last_ping_tm > ping_timeout
                    has_pong_not_arrived_after_last_ping = self.last_pong_tm - self.last_ping_tm < 0
                    has_pong_arrived_too_late = self.last_pong_tm - self.last_ping_tm > ping_timeout

                    if (self.last_ping_tm
                            and has_timeout_expired
                            and (has_pong_not_arrived_after_last_ping or has_pong_arrived_too_late)):
                        raise WebSocketTimeoutException("ping/pong timed out")
                return True

            dispatcher.read(self.sock.sock, read, check)
        except (Exception, KeyboardInterrupt, SystemExit) as e:
            self._callback(self.on_error, e)
            if isinstance(e, SystemExit):
                # propagate SystemExit further
                raise
            teardown()
            return not isinstance(e, KeyboardInterrupt)

    def create_dispatcher(self, ping_timeout):
        timeout = ping_timeout or 10
        if self.sock.is_ssl():
            return SSLDispacther(self, timeout)

        return Dispatcher(self, timeout)

    def _get_close_args(self, data):
        """ this functions extracts the code, reason from the close body
        if they exists, and if the self.on_close except three arguments """
        # if the on_close callback is "old", just return empty list
        if sys.version_info < (3, 0):
            if not self.on_close or len(inspect.getargspec(self.on_close).args) != 3:
                return []
        else:
            if not self.on_close or len(inspect.getfullargspec(self.on_close).args) != 3:
                return []

        if data and len(data) >= 2:
            code = 256 * six.byte2int(data[0:1]) + six.byte2int(data[1:2])
            reason = data[2:].decode('utf-8')
            return [code, reason]

        return [None, None]

    def _callback(self, callback, *args):
        if callback:
            try:
                if inspect.ismethod(callback):
                    callback(*args)
                else:
                    callback(self, *args)

            except Exception as e:
                _logging.error("error from callback {}: {}".format(callback, e))
                if _logging.isEnabledForDebug():
                    _, _, tb = sys.exc_info()
                    traceback.print_tb(tb)
