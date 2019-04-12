# -*- coding: utf-8 -*-
#

import sys
sys.path[0:0] = [""]

import os
import os.path
import socket

import six

# websocket-client
import websocket as ws
from websocket._handshake import _create_sec_websocket_key, \
    _validate as _validate_header
from websocket._http import read_headers
from websocket._url import get_proxy_info, parse_url
from websocket._utils import validate_utf8

if six.PY3:
    from base64 import decodebytes as base64decode
else:
    from base64 import decodestring as base64decode

if sys.version_info[0] == 2 and sys.version_info[1] < 7:
    import unittest2 as unittest
else:
    import unittest

try:
    from ssl import SSLError
except ImportError:
    # dummy class of SSLError for ssl none-support environment.
    class SSLError(Exception):
        pass

# Skip test to access the internet.
TEST_WITH_INTERNET = os.environ.get('TEST_WITH_INTERNET', '0') == '1'

# Skip Secure WebSocket test.
TEST_SECURE_WS = True
TRACEABLE = True


def create_mask_key(_):
    return "abcd"


class SockMock(object):
    def __init__(self):
        self.data = []
        self.sent = []

    def add_packet(self, data):
        self.data.append(data)

    def recv(self, bufsize):
        if self.data:
            e = self.data.pop(0)
            if isinstance(e, Exception):
                raise e
            if len(e) > bufsize:
                self.data.insert(0, e[bufsize:])
            return e[:bufsize]

    def send(self, data):
        self.sent.append(data)
        return len(data)

    def close(self):
        pass


class HeaderSockMock(SockMock):

    def __init__(self, fname):
        SockMock.__init__(self)
        path = os.path.join(os.path.dirname(__file__), fname)
        with open(path, "rb") as f:
            self.add_packet(f.read())


class WebSocketTest(unittest.TestCase):
    def setUp(self):
        ws.enableTrace(TRACEABLE)

    def tearDown(self):
        pass

    def testDefaultTimeout(self):
        self.assertEqual(ws.getdefaulttimeout(), None)
        ws.setdefaulttimeout(10)
        self.assertEqual(ws.getdefaulttimeout(), 10)
        ws.setdefaulttimeout(None)

    def testParseUrl(self):
        p = parse_url("ws://www.example.com/r")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com/r/")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/r/")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com/")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com:8080/r")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com:8080/")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/")
        self.assertEqual(p[3], False)

        p = parse_url("ws://www.example.com:8080")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/")
        self.assertEqual(p[3], False)

        p = parse_url("wss://www.example.com:8080/r")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], True)

        p = parse_url("wss://www.example.com:8080/r?key=value")
        self.assertEqual(p[0], "www.example.com")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r?key=value")
        self.assertEqual(p[3], True)

        self.assertRaises(ValueError, parse_url, "http://www.example.com/r")

        if sys.version_info[0] == 2 and sys.version_info[1] < 7:
            return

        p = parse_url("ws://[2a03:4000:123:83::3]/r")
        self.assertEqual(p[0], "2a03:4000:123:83::3")
        self.assertEqual(p[1], 80)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], False)

        p = parse_url("ws://[2a03:4000:123:83::3]:8080/r")
        self.assertEqual(p[0], "2a03:4000:123:83::3")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], False)

        p = parse_url("wss://[2a03:4000:123:83::3]/r")
        self.assertEqual(p[0], "2a03:4000:123:83::3")
        self.assertEqual(p[1], 443)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], True)

        p = parse_url("wss://[2a03:4000:123:83::3]:8080/r")
        self.assertEqual(p[0], "2a03:4000:123:83::3")
        self.assertEqual(p[1], 8080)
        self.assertEqual(p[2], "/r")
        self.assertEqual(p[3], True)

    def testWSKey(self):
        key = _create_sec_websocket_key()
        self.assertTrue(key != 24)
        self.assertTrue(six.u("¥n") not in key)

    def testWsUtils(self):
        key = "c6b8hTg4EeGb2gQMztV1/g=="
        required_header = {
            "upgrade": "websocket",
            "connection": "upgrade",
            "sec-websocket-accept": "Kxep+hNu9n51529fGidYu7a3wO0=",
            }
        self.assertEqual(_validate_header(required_header, key, None), (True, None))

        header = required_header.copy()
        header["upgrade"] = "http"
        self.assertEqual(_validate_header(header, key, None), (False, None))
        del header["upgrade"]
        self.assertEqual(_validate_header(header, key, None), (False, None))

        header = required_header.copy()
        header["connection"] = "something"
        self.assertEqual(_validate_header(header, key, None), (False, None))
        del header["connection"]
        self.assertEqual(_validate_header(header, key, None), (False, None))

        header = required_header.copy()
        header["sec-websocket-accept"] = "something"
        self.assertEqual(_validate_header(header, key, None), (False, None))
        del header["sec-websocket-accept"]
        self.assertEqual(_validate_header(header, key, None), (False, None))

        header = required_header.copy()
        header["sec-websocket-protocol"] = "sub1"
        self.assertEqual(_validate_header(header, key, ["sub1", "sub2"]), (True, "sub1"))
        self.assertEqual(_validate_header(header, key, ["sub2", "sub3"]), (False, None))

        header = required_header.copy()
        header["sec-websocket-protocol"] = "sUb1"
        self.assertEqual(_validate_header(header, key, ["Sub1", "suB2"]), (True, "sub1"))


    def testReadHeader(self):
        status, header, status_message = read_headers(HeaderSockMock("data/header01.txt"))
        self.assertEqual(status, 101)
        self.assertEqual(header["connection"], "Upgrade")

        HeaderSockMock("data/header02.txt")
        self.assertRaises(ws.WebSocketException, read_headers, HeaderSockMock("data/header02.txt"))

    def testSend(self):
        # TODO: add longer frame data
        sock = ws.WebSocket()
        sock.set_mask_key(create_mask_key)
        s = sock.sock = HeaderSockMock("data/header01.txt")
        sock.send("Hello")
        self.assertEqual(s.sent[0], six.b("\x81\x85abcd)\x07\x0f\x08\x0e"))

        sock.send("こんにちは")
        self.assertEqual(s.sent[1], six.b("\x81\x8fabcd\x82\xe3\xf0\x87\xe3\xf1\x80\xe5\xca\x81\xe2\xc5\x82\xe3\xcc"))

        sock.send(u"こんにちは")
        self.assertEqual(s.sent[1], six.b("\x81\x8fabcd\x82\xe3\xf0\x87\xe3\xf1\x80\xe5\xca\x81\xe2\xc5\x82\xe3\xcc"))

        sock.send("x" * 127)

    def testRecv(self):
        # TODO: add longer frame data
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        something = six.b("\x81\x8fabcd\x82\xe3\xf0\x87\xe3\xf1\x80\xe5\xca\x81\xe2\xc5\x82\xe3\xcc")
        s.add_packet(something)
        data = sock.recv()
        self.assertEqual(data, "こんにちは")

        s.add_packet(six.b("\x81\x85abcd)\x07\x0f\x08\x0e"))
        data = sock.recv()
        self.assertEqual(data, "Hello")

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def testIter(self):
        count = 2
        for _ in ws.create_connection('ws://stream.meetup.com/2/rsvps'):
            count -= 1
            if count == 0:
                break

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def testNext(self):
        sock = ws.create_connection('ws://stream.meetup.com/2/rsvps')
        self.assertEqual(str, type(next(sock)))

    def testInternalRecvStrict(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        s.add_packet(six.b("foo"))
        s.add_packet(socket.timeout())
        s.add_packet(six.b("bar"))
        # s.add_packet(SSLError("The read operation timed out"))
        s.add_packet(six.b("baz"))
        with self.assertRaises(ws.WebSocketTimeoutException):
            sock.frame_buffer.recv_strict(9)
        # if six.PY2:
        #     with self.assertRaises(ws.WebSocketTimeoutException):
        #         data = sock._recv_strict(9)
        # else:
        #     with self.assertRaises(SSLError):
        #         data = sock._recv_strict(9)
        data = sock.frame_buffer.recv_strict(9)
        self.assertEqual(data, six.b("foobarbaz"))
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.frame_buffer.recv_strict(1)

    def testRecvTimeout(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        s.add_packet(six.b("\x81"))
        s.add_packet(socket.timeout())
        s.add_packet(six.b("\x8dabcd\x29\x07\x0f\x08\x0e"))
        s.add_packet(socket.timeout())
        s.add_packet(six.b("\x4e\x43\x33\x0e\x10\x0f\x00\x40"))
        with self.assertRaises(ws.WebSocketTimeoutException):
            sock.recv()
        with self.assertRaises(ws.WebSocketTimeoutException):
            sock.recv()
        data = sock.recv()
        self.assertEqual(data, "Hello, World!")
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()

    def testRecvWithSimpleFragmentation(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        # OPCODE=TEXT, FIN=0, MSG="Brevity is "
        s.add_packet(six.b("\x01\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C"))
        # OPCODE=CONT, FIN=1, MSG="the soul of wit"
        s.add_packet(six.b("\x80\x8fabcd\x15\n\x06D\x12\r\x16\x08A\r\x05D\x16\x0b\x17"))
        data = sock.recv()
        self.assertEqual(data, "Brevity is the soul of wit")
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()

    def testRecvWithFireEventOfFragmentation(self):
        sock = ws.WebSocket(fire_cont_frame=True)
        s = sock.sock = SockMock()
        # OPCODE=TEXT, FIN=0, MSG="Brevity is "
        s.add_packet(six.b("\x01\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C"))
        # OPCODE=CONT, FIN=0, MSG="Brevity is "
        s.add_packet(six.b("\x00\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C"))
        # OPCODE=CONT, FIN=1, MSG="the soul of wit"
        s.add_packet(six.b("\x80\x8fabcd\x15\n\x06D\x12\r\x16\x08A\r\x05D\x16\x0b\x17"))

        _, data = sock.recv_data()
        self.assertEqual(data, six.b("Brevity is "))
        _, data = sock.recv_data()
        self.assertEqual(data, six.b("Brevity is "))
        _, data = sock.recv_data()
        self.assertEqual(data, six.b("the soul of wit"))

        # OPCODE=CONT, FIN=0, MSG="Brevity is "
        s.add_packet(six.b("\x80\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C"))

        with self.assertRaises(ws.WebSocketException):
            sock.recv_data()

        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()

    def testClose(self):
        sock = ws.WebSocket()
        sock.sock = SockMock()
        sock.connected = True
        sock.close()
        self.assertEqual(sock.connected, False)

        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        sock.connected = True
        s.add_packet(six.b('\x88\x80\x17\x98p\x84'))
        sock.recv()
        self.assertEqual(sock.connected, False)

    def testRecvContFragmentation(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        # OPCODE=CONT, FIN=1, MSG="the soul of wit"
        s.add_packet(six.b("\x80\x8fabcd\x15\n\x06D\x12\r\x16\x08A\r\x05D\x16\x0b\x17"))
        self.assertRaises(ws.WebSocketException, sock.recv)

    def testRecvWithProlongedFragmentation(self):
        sock = ws.WebSocket()
        s = sock.sock = SockMock()
        # OPCODE=TEXT, FIN=0, MSG="Once more unto the breach, "
        s.add_packet(six.b("\x01\x9babcd.\x0c\x00\x01A\x0f\x0c\x16\x04B\x16\n\x15"
                           "\rC\x10\t\x07C\x06\x13\x07\x02\x07\tNC"))
        # OPCODE=CONT, FIN=0, MSG="dear friends, "
        s.add_packet(six.b("\x00\x8eabcd\x05\x07\x02\x16A\x04\x11\r\x04\x0c\x07"
                           "\x17MB"))
        # OPCODE=CONT, FIN=1, MSG="once more"
        s.add_packet(six.b("\x80\x89abcd\x0e\x0c\x00\x01A\x0f\x0c\x16\x04"))
        data = sock.recv()
        self.assertEqual(
            data,
            "Once more unto the breach, dear friends, once more")
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()

    def testRecvWithFragmentationAndControlFrame(self):
        sock = ws.WebSocket()
        sock.set_mask_key(create_mask_key)
        s = sock.sock = SockMock()
        # OPCODE=TEXT, FIN=0, MSG="Too much "
        s.add_packet(six.b("\x01\x89abcd5\r\x0cD\x0c\x17\x00\x0cA"))
        # OPCODE=PING, FIN=1, MSG="Please PONG this"
        s.add_packet(six.b("\x89\x90abcd1\x0e\x06\x05\x12\x07C4.,$D\x15\n\n\x17"))
        # OPCODE=CONT, FIN=1, MSG="of a good thing"
        s.add_packet(six.b("\x80\x8fabcd\x0e\x04C\x05A\x05\x0c\x0b\x05B\x17\x0c"
                           "\x08\x0c\x04"))
        data = sock.recv()
        self.assertEqual(data, "Too much of a good thing")
        with self.assertRaises(ws.WebSocketConnectionClosedException):
            sock.recv()
        self.assertEqual(
            s.sent[0],
            six.b("\x8a\x90abcd1\x0e\x06\x05\x12\x07C4.,$D\x15\n\n\x17"))

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def testWebSocket(self):
        s = ws.create_connection("ws://echo.websocket.org/")
        self.assertNotEqual(s, None)
        s.send("Hello, World")
        result = s.recv()
        self.assertEqual(result, "Hello, World")

        s.send(u"こにゃにゃちは、世界")
        result = s.recv()
        self.assertEqual(result, "こにゃにゃちは、世界")
        s.close()

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def testPingPong(self):
        s = ws.create_connection("ws://echo.websocket.org/")
        self.assertNotEqual(s, None)
        s.ping("Hello")
        s.pong("Hi")
        s.close()

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    @unittest.skipUnless(TEST_SECURE_WS, "wss://echo.websocket.org doesn't work well.")
    def testSecureWebSocket(self):
        if 1:
            import ssl
            s = ws.create_connection("wss://echo.websocket.org/")
            self.assertNotEqual(s, None)
            self.assertTrue(isinstance(s.sock, ssl.SSLSocket))
            s.send("Hello, World")
            result = s.recv()
            self.assertEqual(result, "Hello, World")
            s.send(u"こにゃにゃちは、世界")
            result = s.recv()
            self.assertEqual(result, "こにゃにゃちは、世界")
            s.close()
        #except:
        #    pass

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def testWebSocketWihtCustomHeader(self):
        s = ws.create_connection("ws://echo.websocket.org/",
                                 headers={"User-Agent": "PythonWebsocketClient"})
        self.assertNotEqual(s, None)
        s.send("Hello, World")
        result = s.recv()
        self.assertEqual(result, "Hello, World")
        s.close()

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def testAfterClose(self):
        s = ws.create_connection("ws://echo.websocket.org/")
        self.assertNotEqual(s, None)
        s.close()
        self.assertRaises(ws.WebSocketConnectionClosedException, s.send, "Hello")
        self.assertRaises(ws.WebSocketConnectionClosedException, s.recv)

    def testNonce(self):
        """ WebSocket key should be a random 16-byte nonce.
        """
        key = _create_sec_websocket_key()
        nonce = base64decode(key.encode("utf-8"))
        self.assertEqual(16, len(nonce))


class WebSocketAppTest(unittest.TestCase):

    class NotSetYet(object):
        """ A marker class for signalling that a value hasn't been set yet.
        """

    def setUp(self):
        ws.enableTrace(TRACEABLE)

        WebSocketAppTest.keep_running_open = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.keep_running_close = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.get_mask_key_id = WebSocketAppTest.NotSetYet()

    def tearDown(self):
        WebSocketAppTest.keep_running_open = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.keep_running_close = WebSocketAppTest.NotSetYet()
        WebSocketAppTest.get_mask_key_id = WebSocketAppTest.NotSetYet()

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def testKeepRunning(self):
        """ A WebSocketApp should keep running as long as its self.keep_running
        is not False (in the boolean context).
        """

        def on_open(self, *args, **kwargs):
            """ Set the keep_running flag for later inspection and immediately
            close the connection.
            """
            WebSocketAppTest.keep_running_open = self.keep_running

            self.close()

        def on_close(self, *args, **kwargs):
            """ Set the keep_running flag for the test to use.
            """
            WebSocketAppTest.keep_running_close = self.keep_running

        app = ws.WebSocketApp('ws://echo.websocket.org/', on_open=on_open, on_close=on_close)
        app.run_forever()

        # if numpy is installed, this assertion fail
        # self.assertFalse(isinstance(WebSocketAppTest.keep_running_open,
        #                             WebSocketAppTest.NotSetYet))

        # self.assertFalse(isinstance(WebSocketAppTest.keep_running_close,
        #                             WebSocketAppTest.NotSetYet))

        # self.assertEqual(True, WebSocketAppTest.keep_running_open)
        # self.assertEqual(False, WebSocketAppTest.keep_running_close)

    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def testSockMaskKey(self):
        """ A WebSocketApp should forward the received mask_key function down
        to the actual socket.
        """

        def my_mask_key_func():
            pass

        def on_open(self, *args, **kwargs):
            """ Set the value so the test can use it later on and immediately
            close the connection.
            """
            WebSocketAppTest.get_mask_key_id = id(self.get_mask_key)
            self.close()

        app = ws.WebSocketApp('ws://echo.websocket.org/', on_open=on_open, get_mask_key=my_mask_key_func)
        app.run_forever()

        # if numpu is installed, this assertion fail
        # Note: We can't use 'is' for comparing the functions directly, need to use 'id'.
        # self.assertEqual(WebSocketAppTest.get_mask_key_id, id(my_mask_key_func))


class SockOptTest(unittest.TestCase):
    @unittest.skipUnless(TEST_WITH_INTERNET, "Internet-requiring tests are disabled")
    def testSockOpt(self):
        sockopt = ((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),)
        s = ws.create_connection("ws://echo.websocket.org", sockopt=sockopt)
        self.assertNotEqual(s.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY), 0)
        s.close()


class UtilsTest(unittest.TestCase):
    def testUtf8Validator(self):
        state = validate_utf8(six.b('\xf0\x90\x80\x80'))
        self.assertEqual(state, True)
        state = validate_utf8(six.b('\xce\xba\xe1\xbd\xb9\xcf\x83\xce\xbc\xce\xb5\xed\xa0\x80edited'))
        self.assertEqual(state, False)
        state = validate_utf8(six.b(''))
        self.assertEqual(state, True)


class ProxyInfoTest(unittest.TestCase):
    def setUp(self):
        self.http_proxy = os.environ.get("http_proxy", None)
        self.https_proxy = os.environ.get("https_proxy", None)
        if "http_proxy" in os.environ:
            del os.environ["http_proxy"]
        if "https_proxy" in os.environ:
            del os.environ["https_proxy"]

    def tearDown(self):
        if self.http_proxy:
            os.environ["http_proxy"] = self.http_proxy
        elif "http_proxy" in os.environ:
            del os.environ["http_proxy"]

        if self.https_proxy:
            os.environ["https_proxy"] = self.https_proxy
        elif "https_proxy" in os.environ:
            del os.environ["https_proxy"]

    def testProxyFromArgs(self):
        self.assertEqual(get_proxy_info("echo.websocket.org", False, proxy_host="localhost"), ("localhost", 0, None))
        self.assertEqual(get_proxy_info("echo.websocket.org", False, proxy_host="localhost", proxy_port=3128), ("localhost", 3128, None))
        self.assertEqual(get_proxy_info("echo.websocket.org", True, proxy_host="localhost"), ("localhost", 0, None))
        self.assertEqual(get_proxy_info("echo.websocket.org", True, proxy_host="localhost", proxy_port=3128), ("localhost", 3128, None))

        self.assertEqual(get_proxy_info("echo.websocket.org", False, proxy_host="localhost", proxy_auth=("a", "b")),
            ("localhost", 0, ("a", "b")))
        self.assertEqual(get_proxy_info("echo.websocket.org", False, proxy_host="localhost", proxy_port=3128, proxy_auth=("a", "b")),
            ("localhost", 3128, ("a", "b")))
        self.assertEqual(get_proxy_info("echo.websocket.org", True, proxy_host="localhost", proxy_auth=("a", "b")),
            ("localhost", 0, ("a", "b")))
        self.assertEqual(get_proxy_info("echo.websocket.org", True, proxy_host="localhost", proxy_port=3128, proxy_auth=("a", "b")),
            ("localhost", 3128, ("a", "b")))

        self.assertEqual(get_proxy_info("echo.websocket.org", True, proxy_host="localhost", proxy_port=3128, no_proxy=["example.com"], proxy_auth=("a", "b")),
            ("localhost", 3128, ("a", "b")))
        self.assertEqual(get_proxy_info("echo.websocket.org", True, proxy_host="localhost", proxy_port=3128, no_proxy=["echo.websocket.org"], proxy_auth=("a", "b")),
            (None, 0, None))

    def testProxyFromEnv(self):
        os.environ["http_proxy"] = "http://localhost/"
        self.assertEqual(get_proxy_info("echo.websocket.org", False), ("localhost", None, None))
        os.environ["http_proxy"] = "http://localhost:3128/"
        self.assertEqual(get_proxy_info("echo.websocket.org", False), ("localhost", 3128, None))

        os.environ["http_proxy"] = "http://localhost/"
        os.environ["https_proxy"] = "http://localhost2/"
        self.assertEqual(get_proxy_info("echo.websocket.org", False), ("localhost", None, None))
        os.environ["http_proxy"] = "http://localhost:3128/"
        os.environ["https_proxy"] = "http://localhost2:3128/"
        self.assertEqual(get_proxy_info("echo.websocket.org", False), ("localhost", 3128, None))

        os.environ["http_proxy"] = "http://localhost/"
        os.environ["https_proxy"] = "http://localhost2/"
        self.assertEqual(get_proxy_info("echo.websocket.org", True), ("localhost2", None, None))
        os.environ["http_proxy"] = "http://localhost:3128/"
        os.environ["https_proxy"] = "http://localhost2:3128/"
        self.assertEqual(get_proxy_info("echo.websocket.org", True), ("localhost2", 3128, None))


        os.environ["http_proxy"] = "http://a:b@localhost/"
        self.assertEqual(get_proxy_info("echo.websocket.org", False), ("localhost", None, ("a", "b")))
        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        self.assertEqual(get_proxy_info("echo.websocket.org", False), ("localhost", 3128, ("a", "b")))

        os.environ["http_proxy"] = "http://a:b@localhost/"
        os.environ["https_proxy"] = "http://a:b@localhost2/"
        self.assertEqual(get_proxy_info("echo.websocket.org", False), ("localhost", None, ("a", "b")))
        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        os.environ["https_proxy"] = "http://a:b@localhost2:3128/"
        self.assertEqual(get_proxy_info("echo.websocket.org", False), ("localhost", 3128, ("a", "b")))

        os.environ["http_proxy"] = "http://a:b@localhost/"
        os.environ["https_proxy"] = "http://a:b@localhost2/"
        self.assertEqual(get_proxy_info("echo.websocket.org", True), ("localhost2", None, ("a", "b")))
        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        os.environ["https_proxy"] = "http://a:b@localhost2:3128/"
        self.assertEqual(get_proxy_info("echo.websocket.org", True), ("localhost2", 3128, ("a", "b")))

        os.environ["http_proxy"] = "http://a:b@localhost/"
        os.environ["https_proxy"] = "http://a:b@localhost2/"
        os.environ["no_proxy"] = "example1.com,example2.com"
        self.assertEqual(get_proxy_info("example.1.com", True), ("localhost2", None, ("a", "b")))
        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        os.environ["https_proxy"] = "http://a:b@localhost2:3128/"
        os.environ["no_proxy"] = "example1.com,example2.com, echo.websocket.org"
        self.assertEqual(get_proxy_info("echo.websocket.org", True), (None, 0, None))

        os.environ["http_proxy"] = "http://a:b@localhost:3128/"
        os.environ["https_proxy"] = "http://a:b@localhost2:3128/"
        os.environ["no_proxy"] = "127.0.0.0/8, 192.168.0.0/16"
        self.assertEqual(get_proxy_info("127.0.0.1", False), (None, 0, None))
        self.assertEqual(get_proxy_info("192.168.1.1", False), (None, 0, None))


if __name__ == "__main__":
    unittest.main()
