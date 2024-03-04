from __future__ import absolute_import, division

import copy
import errno
import io
import logging
from random import shuffle, uniform

# selectors in stdlib as of py3.4
try:
    import selectors  # pylint: disable=import-error
except ImportError:
    # vendored backport module
    from kafka.vendor import selectors34 as selectors

import socket
import struct
import threading
import time

from kafka.vendor import six

import kafka.errors as Errors
from kafka.future import Future
from kafka.metrics.stats import Avg, Count, Max, Rate
from kafka.oauth.abstract import AbstractTokenProvider
from kafka.protocol.admin import SaslHandShakeRequest, DescribeAclsRequest_v2
from kafka.protocol.commit import OffsetFetchRequest
from kafka.protocol.offset import OffsetRequest
from kafka.protocol.produce import ProduceRequest
from kafka.protocol.metadata import MetadataRequest
from kafka.protocol.fetch import FetchRequest
from kafka.protocol.parser import KafkaProtocol
from kafka.protocol.types import Int32, Int8
from kafka.scram import ScramClient
from kafka.version import __version__


if six.PY2:
    ConnectionError = socket.error
    TimeoutError = socket.error
    BlockingIOError = Exception

log = logging.getLogger(__name__)

DEFAULT_KAFKA_PORT = 9092

SASL_QOP_AUTH = 1
SASL_QOP_AUTH_INT = 2
SASL_QOP_AUTH_CONF = 4

try:
    import ssl
    ssl_available = True
    try:
        SSLEOFError = ssl.SSLEOFError
        SSLWantReadError = ssl.SSLWantReadError
        SSLWantWriteError = ssl.SSLWantWriteError
        SSLZeroReturnError = ssl.SSLZeroReturnError
    except AttributeError:
        # support older ssl libraries
        log.warning('Old SSL module detected.'
                    ' SSL error handling may not operate cleanly.'
                    ' Consider upgrading to Python 3.3 or 2.7.9')
        SSLEOFError = ssl.SSLError
        SSLWantReadError = ssl.SSLError
        SSLWantWriteError = ssl.SSLError
        SSLZeroReturnError = ssl.SSLError
except ImportError:
    # support Python without ssl libraries
    ssl_available = False
    class SSLWantReadError(Exception):
        pass
    class SSLWantWriteError(Exception):
        pass

# needed for SASL_GSSAPI authentication:
try:
    import gssapi
    from gssapi.raw.misc import GSSError
except ImportError:
    #no gssapi available, will disable gssapi mechanism
    gssapi = None
    GSSError = None


AFI_NAMES = {
    socket.AF_UNSPEC: "unspecified",
    socket.AF_INET: "IPv4",
    socket.AF_INET6: "IPv6",
}


class ConnectionStates(object):
    DISCONNECTING = '<disconnecting>'
    DISCONNECTED = '<disconnected>'
    CONNECTING = '<connecting>'
    HANDSHAKE = '<handshake>'
    CONNECTED = '<connected>'
    AUTHENTICATING = '<authenticating>'


class BrokerConnection(object):
    """Initialize a Kafka broker connection

    Keyword Arguments:
        client_id (str): a name for this client. This string is passed in
            each request to servers and can be used to identify specific
            server-side log entries that correspond to this client. Also
            submitted to GroupCoordinator for logging with respect to
            consumer group administration. Default: 'kafka-python-{version}'
        reconnect_backoff_ms (int): The amount of time in milliseconds to
            wait before attempting to reconnect to a given host.
            Default: 50.
        reconnect_backoff_max_ms (int): The maximum amount of time in
            milliseconds to backoff/wait when reconnecting to a broker that has
            repeatedly failed to connect. If provided, the backoff per host
            will increase exponentially for each consecutive connection
            failure, up to this maximum. Once the maximum is reached,
            reconnection attempts will continue periodically with this fixed
            rate. To avoid connection storms, a randomization factor of 0.2
            will be applied to the backoff resulting in a random range between
            20% below and 20% above the computed value. Default: 1000.
        request_timeout_ms (int): Client request timeout in milliseconds.
            Default: 30000.
        max_in_flight_requests_per_connection (int): Requests are pipelined
            to kafka brokers up to this number of maximum requests per
            broker connection. Default: 5.
        receive_buffer_bytes (int): The size of the TCP receive buffer
            (SO_RCVBUF) to use when reading data. Default: None (relies on
            system defaults). Java client defaults to 32768.
        send_buffer_bytes (int): The size of the TCP send buffer
            (SO_SNDBUF) to use when sending data. Default: None (relies on
            system defaults). Java client defaults to 131072.
        socket_options (list): List of tuple-arguments to socket.setsockopt
            to apply to broker connection sockets. Default:
            [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
        security_protocol (str): Protocol used to communicate with brokers.
            Valid values are: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL.
            Default: PLAINTEXT.
        ssl_context (ssl.SSLContext): pre-configured SSLContext for wrapping
            socket connections. If provided, all other ssl_* configurations
            will be ignored. Default: None.
        ssl_check_hostname (bool): flag to configure whether ssl handshake
            should verify that the certificate matches the brokers hostname.
            default: True.
        ssl_cafile (str): optional filename of ca file to use in certificate
            verification. default: None.
        ssl_certfile (str): optional filename of file in pem format containing
            the client certificate, as well as any ca certificates needed to
            establish the certificate's authenticity. default: None.
        ssl_keyfile (str): optional filename containing the client private key.
            default: None.
        ssl_password (callable, str, bytes, bytearray): optional password or
            callable function that returns a password, for decrypting the
            client private key. Default: None.
        ssl_crlfile (str): optional filename containing the CRL to check for
            certificate expiration. By default, no CRL check is done. When
            providing a file, only the leaf certificate will be checked against
            this CRL. The CRL can only be checked with Python 3.4+ or 2.7.9+.
            default: None.
        ssl_ciphers (str): optionally set the available ciphers for ssl
            connections. It should be a string in the OpenSSL cipher list
            format. If no cipher can be selected (because compile-time options
            or other configuration forbids use of all the specified ciphers),
            an ssl.SSLError will be raised. See ssl.SSLContext.set_ciphers
        api_version (tuple): Specify which Kafka API version to use.
            Accepted values are: (0, 8, 0), (0, 8, 1), (0, 8, 2), (0, 9),
            (0, 10). Default: (0, 8, 2)
        api_version_auto_timeout_ms (int): number of milliseconds to throw a
            timeout exception from the constructor when checking the broker
            api version. Only applies if api_version is None
        selector (selectors.BaseSelector): Provide a specific selector
            implementation to use for I/O multiplexing.
            Default: selectors.DefaultSelector
        state_change_callback (callable): function to be called when the
            connection state changes from CONNECTING to CONNECTED etc.
        metrics (kafka.metrics.Metrics): Optionally provide a metrics
            instance for capturing network IO stats. Default: None.
        metric_group_prefix (str): Prefix for metric names. Default: ''
        sasl_mechanism (str): Authentication mechanism when security_protocol
            is configured for SASL_PLAINTEXT or SASL_SSL. Valid values are:
            PLAIN, GSSAPI, OAUTHBEARER, SCRAM-SHA-256, SCRAM-SHA-512.
        sasl_plain_username (str): username for sasl PLAIN and SCRAM authentication.
            Required if sasl_mechanism is PLAIN or one of the SCRAM mechanisms.
        sasl_plain_password (str): password for sasl PLAIN and SCRAM authentication.
            Required if sasl_mechanism is PLAIN or one of the SCRAM mechanisms.
        sasl_kerberos_service_name (str): Service name to include in GSSAPI
            sasl mechanism handshake. Default: 'kafka'
        sasl_kerberos_domain_name (str): kerberos domain name to use in GSSAPI
            sasl mechanism handshake. Default: one of bootstrap servers
        sasl_oauth_token_provider (AbstractTokenProvider): OAuthBearer token provider
            instance. (See kafka.oauth.abstract). Default: None
    """

    DEFAULT_CONFIG = {
        'client_id': 'kafka-python-' + __version__,
        'node_id': 0,
        'request_timeout_ms': 30000,
        'reconnect_backoff_ms': 50,
        'reconnect_backoff_max_ms': 1000,
        'max_in_flight_requests_per_connection': 5,
        'receive_buffer_bytes': None,
        'send_buffer_bytes': None,
        'socket_options': [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
        'sock_chunk_bytes': 4096,  # undocumented experimental option
        'sock_chunk_buffer_count': 1000,  # undocumented experimental option
        'security_protocol': 'PLAINTEXT',
        'ssl_context': None,
        'ssl_check_hostname': True,
        'ssl_cafile': None,
        'ssl_certfile': None,
        'ssl_keyfile': None,
        'ssl_crlfile': None,
        'ssl_password': None,
        'ssl_ciphers': None,
        'api_version': (0, 8, 2),  # default to most restrictive
        'selector': selectors.DefaultSelector,
        'state_change_callback': lambda node_id, sock, conn: True,
        'metrics': None,
        'metric_group_prefix': '',
        'sasl_mechanism': None,
        'sasl_plain_username': None,
        'sasl_plain_password': None,
        'sasl_kerberos_service_name': 'kafka',
        'sasl_kerberos_domain_name': None,
        'sasl_oauth_token_provider': None
    }
    SECURITY_PROTOCOLS = ('PLAINTEXT', 'SSL', 'SASL_PLAINTEXT', 'SASL_SSL')
    SASL_MECHANISMS = ('PLAIN', 'GSSAPI', 'OAUTHBEARER', "SCRAM-SHA-256", "SCRAM-SHA-512")

    def __init__(self, host, port, afi, **configs):
        self.host = host
        self.port = port
        self.afi = afi
        self._sock_afi = afi
        self._sock_addr = None
        self._api_versions = None

        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs[key]

        self.node_id = self.config.pop('node_id')

        if self.config['receive_buffer_bytes'] is not None:
            self.config['socket_options'].append(
                (socket.SOL_SOCKET, socket.SO_RCVBUF,
                 self.config['receive_buffer_bytes']))
        if self.config['send_buffer_bytes'] is not None:
            self.config['socket_options'].append(
                 (socket.SOL_SOCKET, socket.SO_SNDBUF,
                 self.config['send_buffer_bytes']))

        assert self.config['security_protocol'] in self.SECURITY_PROTOCOLS, (
            'security_protocol must be in ' + ', '.join(self.SECURITY_PROTOCOLS))

        if self.config['security_protocol'] in ('SSL', 'SASL_SSL'):
            assert ssl_available, "Python wasn't built with SSL support"

        if self.config['security_protocol'] in ('SASL_PLAINTEXT', 'SASL_SSL'):
            assert self.config['sasl_mechanism'] in self.SASL_MECHANISMS, (
                'sasl_mechanism must be in ' + ', '.join(self.SASL_MECHANISMS))
            if self.config['sasl_mechanism'] in ('PLAIN', 'SCRAM-SHA-256', 'SCRAM-SHA-512'):
                assert self.config['sasl_plain_username'] is not None, (
                    'sasl_plain_username required for PLAIN or SCRAM sasl'
                )
                assert self.config['sasl_plain_password'] is not None, (
                    'sasl_plain_password required for PLAIN or SCRAM sasl'
                )
            if self.config['sasl_mechanism'] == 'GSSAPI':
                assert gssapi is not None, 'GSSAPI lib not available'
                assert self.config['sasl_kerberos_service_name'] is not None, 'sasl_kerberos_service_name required for GSSAPI sasl'
            if self.config['sasl_mechanism'] == 'OAUTHBEARER':
                token_provider = self.config['sasl_oauth_token_provider']
                assert token_provider is not None, 'sasl_oauth_token_provider required for OAUTHBEARER sasl'
                assert callable(getattr(token_provider, "token", None)), 'sasl_oauth_token_provider must implement method #token()'
        # This is not a general lock / this class is not generally thread-safe yet
        # However, to avoid pushing responsibility for maintaining
        # per-connection locks to the upstream client, we will use this lock to
        # make sure that access to the protocol buffer is synchronized
        # when sends happen on multiple threads
        self._lock = threading.Lock()

        # the protocol parser instance manages actual tracking of the
        # sequence of in-flight requests to responses, which should
        # function like a FIFO queue. For additional request data,
        # including tracking request futures and timestamps, we
        # can use a simple dictionary of correlation_id => request data
        self.in_flight_requests = dict()

        self._protocol = KafkaProtocol(
            client_id=self.config['client_id'],
            api_version=self.config['api_version'])
        self.state = ConnectionStates.DISCONNECTED
        self._reset_reconnect_backoff()
        self._sock = None
        self._send_buffer = b''
        self._ssl_context = None
        if self.config['ssl_context'] is not None:
            self._ssl_context = self.config['ssl_context']
        self._sasl_auth_future = None
        self.last_attempt = 0
        self._gai = []
        self._sensors = None
        if self.config['metrics']:
            self._sensors = BrokerConnectionMetrics(self.config['metrics'],
                                                    self.config['metric_group_prefix'],
                                                    self.node_id)

    def _dns_lookup(self):
        self._gai = dns_lookup(self.host, self.port, self.afi)
        if not self._gai:
            log.error('DNS lookup failed for %s:%i (%s)',
                      self.host, self.port, self.afi)
            return False
        return True

    def _next_afi_sockaddr(self):
        if not self._gai:
            if not self._dns_lookup():
                return
        afi, _, __, ___, sockaddr = self._gai.pop(0)
        return (afi, sockaddr)

    def connect_blocking(self, timeout=float('inf')):
        if self.connected():
            return True
        timeout += time.time()
        # First attempt to perform dns lookup
        # note that the underlying interface, socket.getaddrinfo,
        # has no explicit timeout so we may exceed the user-specified timeout
        self._dns_lookup()

        # Loop once over all returned dns entries
        selector = None
        while self._gai:
            while time.time() < timeout:
                self.connect()
                if self.connected():
                    if selector is not None:
                        selector.close()
                    return True
                elif self.connecting():
                    if selector is None:
                        selector = self.config['selector']()
                        selector.register(self._sock, selectors.EVENT_WRITE)
                    selector.select(1)
                elif self.disconnected():
                    if selector is not None:
                        selector.close()
                        selector = None
                    break
            else:
                break
        return False

    def connect(self):
        """Attempt to connect and return ConnectionState"""
        if self.state is ConnectionStates.DISCONNECTED and not self.blacked_out():
            self.last_attempt = time.time()
            next_lookup = self._next_afi_sockaddr()
            if not next_lookup:
                self.close(Errors.KafkaConnectionError('DNS failure'))
                return self.state
            else:
                log.debug('%s: creating new socket', self)
                assert self._sock is None
                self._sock_afi, self._sock_addr = next_lookup
                self._sock = socket.socket(self._sock_afi, socket.SOCK_STREAM)

            for option in self.config['socket_options']:
                log.debug('%s: setting socket option %s', self, option)
                self._sock.setsockopt(*option)

            self._sock.setblocking(False)
            self.state = ConnectionStates.CONNECTING
            self.config['state_change_callback'](self.node_id, self._sock, self)
            log.info('%s: connecting to %s:%d [%s %s]', self, self.host,
                     self.port, self._sock_addr, AFI_NAMES[self._sock_afi])

        if self.state is ConnectionStates.CONNECTING:
            # in non-blocking mode, use repeated calls to socket.connect_ex
            # to check connection status
            ret = None
            try:
                ret = self._sock.connect_ex(self._sock_addr)
            except socket.error as err:
                ret = err.errno

            # Connection succeeded
            if not ret or ret == errno.EISCONN:
                log.debug('%s: established TCP connection', self)

                if self.config['security_protocol'] in ('SSL', 'SASL_SSL'):
                    log.debug('%s: initiating SSL handshake', self)
                    self.state = ConnectionStates.HANDSHAKE
                    self.config['state_change_callback'](self.node_id, self._sock, self)
                    # _wrap_ssl can alter the connection state -- disconnects on failure
                    self._wrap_ssl()

                elif self.config['security_protocol'] == 'SASL_PLAINTEXT':
                    log.debug('%s: initiating SASL authentication', self)
                    self.state = ConnectionStates.AUTHENTICATING
                    self.config['state_change_callback'](self.node_id, self._sock, self)

                else:
                    # security_protocol PLAINTEXT
                    log.info('%s: Connection complete.', self)
                    self.state = ConnectionStates.CONNECTED
                    self._reset_reconnect_backoff()
                    self.config['state_change_callback'](self.node_id, self._sock, self)

            # Connection failed
            # WSAEINVAL == 10022, but errno.WSAEINVAL is not available on non-win systems
            elif ret not in (errno.EINPROGRESS, errno.EALREADY, errno.EWOULDBLOCK, 10022):
                log.error('Connect attempt to %s returned error %s.'
                          ' Disconnecting.', self, ret)
                errstr = errno.errorcode.get(ret, 'UNKNOWN')
                self.close(Errors.KafkaConnectionError('{} {}'.format(ret, errstr)))
                return self.state

            # Needs retry
            else:
                pass

        if self.state is ConnectionStates.HANDSHAKE:
            if self._try_handshake():
                log.debug('%s: completed SSL handshake.', self)
                if self.config['security_protocol'] == 'SASL_SSL':
                    log.debug('%s: initiating SASL authentication', self)
                    self.state = ConnectionStates.AUTHENTICATING
                else:
                    log.info('%s: Connection complete.', self)
                    self.state = ConnectionStates.CONNECTED
                    self._reset_reconnect_backoff()
                self.config['state_change_callback'](self.node_id, self._sock, self)

        if self.state is ConnectionStates.AUTHENTICATING:
            assert self.config['security_protocol'] in ('SASL_PLAINTEXT', 'SASL_SSL')
            if self._try_authenticate():
                # _try_authenticate has side-effects: possibly disconnected on socket errors
                if self.state is ConnectionStates.AUTHENTICATING:
                    log.info('%s: Connection complete.', self)
                    self.state = ConnectionStates.CONNECTED
                    self._reset_reconnect_backoff()
                    self.config['state_change_callback'](self.node_id, self._sock, self)

        if self.state not in (ConnectionStates.CONNECTED,
                              ConnectionStates.DISCONNECTED):
            # Connection timed out
            request_timeout = self.config['request_timeout_ms'] / 1000.0
            if time.time() > request_timeout + self.last_attempt:
                log.error('Connection attempt to %s timed out', self)
                self.close(Errors.KafkaConnectionError('timeout'))
                return self.state

        return self.state

    def _wrap_ssl(self):
        assert self.config['security_protocol'] in ('SSL', 'SASL_SSL')
        if self._ssl_context is None:
            log.debug('%s: configuring default SSL Context', self)
            self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)  # pylint: disable=no-member
            self._ssl_context.options |= ssl.OP_NO_SSLv2  # pylint: disable=no-member
            self._ssl_context.options |= ssl.OP_NO_SSLv3  # pylint: disable=no-member
            self._ssl_context.verify_mode = ssl.CERT_OPTIONAL
            if self.config['ssl_check_hostname']:
                self._ssl_context.check_hostname = True
            if self.config['ssl_cafile']:
                log.info('%s: Loading SSL CA from %s', self, self.config['ssl_cafile'])
                self._ssl_context.load_verify_locations(self.config['ssl_cafile'])
                self._ssl_context.verify_mode = ssl.CERT_REQUIRED
            else:
                log.info('%s: Loading system default SSL CAs from %s', self, ssl.get_default_verify_paths())
                self._ssl_context.load_default_certs()
            if self.config['ssl_certfile'] and self.config['ssl_keyfile']:
                log.info('%s: Loading SSL Cert from %s', self, self.config['ssl_certfile'])
                log.info('%s: Loading SSL Key from %s', self, self.config['ssl_keyfile'])
                self._ssl_context.load_cert_chain(
                    certfile=self.config['ssl_certfile'],
                    keyfile=self.config['ssl_keyfile'],
                    password=self.config['ssl_password'])
            if self.config['ssl_crlfile']:
                if not hasattr(ssl, 'VERIFY_CRL_CHECK_LEAF'):
                    raise RuntimeError('This version of Python does not support ssl_crlfile!')
                log.info('%s: Loading SSL CRL from %s', self, self.config['ssl_crlfile'])
                self._ssl_context.load_verify_locations(self.config['ssl_crlfile'])
                # pylint: disable=no-member
                self._ssl_context.verify_flags |= ssl.VERIFY_CRL_CHECK_LEAF
            if self.config['ssl_ciphers']:
                log.info('%s: Setting SSL Ciphers: %s', self, self.config['ssl_ciphers'])
                self._ssl_context.set_ciphers(self.config['ssl_ciphers'])
        log.debug('%s: wrapping socket in ssl context', self)
        try:
            self._sock = self._ssl_context.wrap_socket(
                self._sock,
                server_hostname=self.host,
                do_handshake_on_connect=False)
        except ssl.SSLError as e:
            log.exception('%s: Failed to wrap socket in SSLContext!', self)
            self.close(e)

    def _try_handshake(self):
        assert self.config['security_protocol'] in ('SSL', 'SASL_SSL')
        try:
            self._sock.do_handshake()
            return True
        # old ssl in python2.6 will swallow all SSLErrors here...
        except (SSLWantReadError, SSLWantWriteError):
            pass
        except (SSLZeroReturnError, ConnectionError, TimeoutError, SSLEOFError):
            log.warning('SSL connection closed by server during handshake.')
            self.close(Errors.KafkaConnectionError('SSL connection closed by server during handshake'))
        # Other SSLErrors will be raised to user

        return False

    def _try_authenticate(self):
        assert self.config['api_version'] is None or self.config['api_version'] >= (0, 10)

        if self._sasl_auth_future is None:
            # Build a SaslHandShakeRequest message
            request = SaslHandShakeRequest[0](self.config['sasl_mechanism'])
            future = Future()
            sasl_response = self._send(request)
            sasl_response.add_callback(self._handle_sasl_handshake_response, future)
            sasl_response.add_errback(lambda f, e: f.failure(e), future)
            self._sasl_auth_future = future

        for r, f in self.recv():
            f.success(r)

        # A connection error could trigger close() which will reset the future
        if self._sasl_auth_future is None:
            return False
        elif self._sasl_auth_future.failed():
            ex = self._sasl_auth_future.exception
            if not isinstance(ex, Errors.KafkaConnectionError):
                raise ex  # pylint: disable-msg=raising-bad-type
        return self._sasl_auth_future.succeeded()

    def _handle_sasl_handshake_response(self, future, response):
        error_type = Errors.for_code(response.error_code)
        if error_type is not Errors.NoError:
            error = error_type(self)
            self.close(error=error)
            return future.failure(error_type(self))

        if self.config['sasl_mechanism'] not in response.enabled_mechanisms:
            return future.failure(
                Errors.UnsupportedSaslMechanismError(
                    'Kafka broker does not support %s sasl mechanism. Enabled mechanisms are: %s'
                    % (self.config['sasl_mechanism'], response.enabled_mechanisms)))
        elif self.config['sasl_mechanism'] == 'PLAIN':
            return self._try_authenticate_plain(future)
        elif self.config['sasl_mechanism'] == 'GSSAPI':
            return self._try_authenticate_gssapi(future)
        elif self.config['sasl_mechanism'] == 'OAUTHBEARER':
            return self._try_authenticate_oauth(future)
        elif self.config['sasl_mechanism'].startswith("SCRAM-SHA-"):
            return self._try_authenticate_scram(future)
        else:
            return future.failure(
                Errors.UnsupportedSaslMechanismError(
                    'kafka-python does not support SASL mechanism %s' %
                    self.config['sasl_mechanism']))

    def _send_bytes(self, data):
        """Send some data via non-blocking IO

        Note: this method is not synchronized internally; you should
        always hold the _lock before calling

        Returns: number of bytes
        Raises: socket exception
        """
        total_sent = 0
        while total_sent < len(data):
            try:
                sent_bytes = self._sock.send(data[total_sent:])
                total_sent += sent_bytes
            except (SSLWantReadError, SSLWantWriteError):
                break
            except (ConnectionError, TimeoutError) as e:
                if six.PY2 and e.errno == errno.EWOULDBLOCK:
                    break
                raise
            except BlockingIOError:
                if six.PY3:
                    break
                raise
        return total_sent

    def _send_bytes_blocking(self, data):
        self._sock.settimeout(self.config['request_timeout_ms'] / 1000)
        total_sent = 0
        try:
            while total_sent < len(data):
                sent_bytes = self._sock.send(data[total_sent:])
                total_sent += sent_bytes
            if total_sent != len(data):
                raise ConnectionError('Buffer overrun during socket send')
            return total_sent
        finally:
            self._sock.settimeout(0.0)

    def _recv_bytes_blocking(self, n):
        self._sock.settimeout(self.config['request_timeout_ms'] / 1000)
        try:
            data = b''
            while len(data) < n:
                fragment = self._sock.recv(n - len(data))
                if not fragment:
                    raise ConnectionError('Connection reset during recv')
                data += fragment
            return data
        finally:
            self._sock.settimeout(0.0)

    def _try_authenticate_plain(self, future):
        if self.config['security_protocol'] == 'SASL_PLAINTEXT':
            log.warning('%s: Sending username and password in the clear', self)

        data = b''
        # Send PLAIN credentials per RFC-4616
        msg = bytes('\0'.join([self.config['sasl_plain_username'],
                               self.config['sasl_plain_username'],
                               self.config['sasl_plain_password']]).encode('utf-8'))
        size = Int32.encode(len(msg))

        err = None
        close = False
        with self._lock:
            if not self._can_send_recv():
                err = Errors.NodeNotReadyError(str(self))
                close = False
            else:
                try:
                    self._send_bytes_blocking(size + msg)

                    # The server will send a zero sized message (that is Int32(0)) on success.
                    # The connection is closed on failure
                    data = self._recv_bytes_blocking(4)

                except (ConnectionError, TimeoutError) as e:
                    log.exception("%s: Error receiving reply from server", self)
                    err = Errors.KafkaConnectionError("%s: %s" % (self, e))
                    close = True

        if err is not None:
            if close:
                self.close(error=err)
            return future.failure(err)

        if data != b'\x00\x00\x00\x00':
            error = Errors.AuthenticationFailedError('Unrecognized response during authentication')
            return future.failure(error)

        log.info('%s: Authenticated as %s via PLAIN', self, self.config['sasl_plain_username'])
        return future.success(True)

    def _try_authenticate_scram(self, future):
        if self.config['security_protocol'] == 'SASL_PLAINTEXT':
            log.warning('%s: Exchanging credentials in the clear', self)

        scram_client = ScramClient(
            self.config['sasl_plain_username'], self.config['sasl_plain_password'], self.config['sasl_mechanism']
        )

        err = None
        close = False
        with self._lock:
            if not self._can_send_recv():
                err = Errors.NodeNotReadyError(str(self))
                close = False
            else:
                try:
                    client_first = scram_client.first_message().encode('utf-8')
                    size = Int32.encode(len(client_first))
                    self._send_bytes_blocking(size + client_first)

                    (data_len,) = struct.unpack('>i', self._recv_bytes_blocking(4))
                    server_first = self._recv_bytes_blocking(data_len).decode('utf-8')
                    scram_client.process_server_first_message(server_first)

                    client_final = scram_client.final_message().encode('utf-8')
                    size = Int32.encode(len(client_final))
                    self._send_bytes_blocking(size + client_final)

                    (data_len,) = struct.unpack('>i', self._recv_bytes_blocking(4))
                    server_final = self._recv_bytes_blocking(data_len).decode('utf-8')
                    scram_client.process_server_final_message(server_final)

                except (ConnectionError, TimeoutError) as e:
                    log.exception("%s: Error receiving reply from server", self)
                    err = Errors.KafkaConnectionError("%s: %s" % (self, e))
                    close = True

        if err is not None:
            if close:
                self.close(error=err)
            return future.failure(err)

        log.info(
            '%s: Authenticated as %s via %s', self, self.config['sasl_plain_username'], self.config['sasl_mechanism']
        )
        return future.success(True)

    def _try_authenticate_gssapi(self, future):
        kerberos_damin_name = self.config['sasl_kerberos_domain_name'] or self.host
        auth_id = self.config['sasl_kerberos_service_name'] + '@' + kerberos_damin_name
        gssapi_name = gssapi.Name(
            auth_id,
            name_type=gssapi.NameType.hostbased_service
        ).canonicalize(gssapi.MechType.kerberos)
        log.debug('%s: GSSAPI name: %s', self, gssapi_name)

        err = None
        close = False
        with self._lock:
            if not self._can_send_recv():
                err = Errors.NodeNotReadyError(str(self))
                close = False
            else:
                # Establish security context and negotiate protection level
                # For reference RFC 2222, section 7.2.1
                try:
                    # Exchange tokens until authentication either succeeds or fails
                    client_ctx = gssapi.SecurityContext(name=gssapi_name, usage='initiate')
                    received_token = None
                    while not client_ctx.complete:
                        # calculate an output token from kafka token (or None if first iteration)
                        output_token = client_ctx.step(received_token)

                        # pass output token to kafka, or send empty response if the security
                        # context is complete (output token is None in that case)
                        if output_token is None:
                            self._send_bytes_blocking(Int32.encode(0))
                        else:
                            msg = output_token
                            size = Int32.encode(len(msg))
                            self._send_bytes_blocking(size + msg)

                        # The server will send a token back. Processing of this token either
                        # establishes a security context, or it needs further token exchange.
                        # The gssapi will be able to identify the needed next step.
                        # The connection is closed on failure.
                        header = self._recv_bytes_blocking(4)
                        (token_size,) = struct.unpack('>i', header)
                        received_token = self._recv_bytes_blocking(token_size)

                    # Process the security layer negotiation token, sent by the server
                    # once the security context is established.

                    # unwraps message containing supported protection levels and msg size
                    msg = client_ctx.unwrap(received_token).message
                    # Kafka currently doesn't support integrity or confidentiality security layers, so we
                    # simply set QoP to 'auth' only (first octet). We reuse the max message size proposed
                    # by the server
                    msg = Int8.encode(SASL_QOP_AUTH & Int8.decode(io.BytesIO(msg[0:1]))) + msg[1:]
                    # add authorization identity to the response, GSS-wrap and send it
                    msg = client_ctx.wrap(msg + auth_id.encode(), False).message
                    size = Int32.encode(len(msg))
                    self._send_bytes_blocking(size + msg)

                except (ConnectionError, TimeoutError) as e:
                    log.exception("%s: Error receiving reply from server",  self)
                    err = Errors.KafkaConnectionError("%s: %s" % (self, e))
                    close = True
                except Exception as e:
                    err = e
                    close = True

        if err is not None:
            if close:
                self.close(error=err)
            return future.failure(err)

        log.info('%s: Authenticated as %s via GSSAPI', self, gssapi_name)
        return future.success(True)

    def _try_authenticate_oauth(self, future):
        data = b''

        msg = bytes(self._build_oauth_client_request().encode("utf-8"))
        size = Int32.encode(len(msg))

        err = None
        close = False
        with self._lock:
            if not self._can_send_recv():
                err = Errors.NodeNotReadyError(str(self))
                close = False
            else:
                try:
                    # Send SASL OAuthBearer request with OAuth token
                    self._send_bytes_blocking(size + msg)

                    # The server will send a zero sized message (that is Int32(0)) on success.
                    # The connection is closed on failure
                    data = self._recv_bytes_blocking(4)

                except (ConnectionError, TimeoutError) as e:
                    log.exception("%s: Error receiving reply from server", self)
                    err = Errors.KafkaConnectionError("%s: %s" % (self, e))
                    close = True

        if err is not None:
            if close:
                self.close(error=err)
            return future.failure(err)

        if data != b'\x00\x00\x00\x00':
            error = Errors.AuthenticationFailedError('Unrecognized response during authentication')
            return future.failure(error)

        log.info('%s: Authenticated via OAuth', self)
        return future.success(True)

    def _build_oauth_client_request(self):
        token_provider = self.config['sasl_oauth_token_provider']
        return "n,,\x01auth=Bearer {}{}\x01\x01".format(token_provider.token(), self._token_extensions())

    def _token_extensions(self):
        """
        Return a string representation of the OPTIONAL key-value pairs that can be sent with an OAUTHBEARER
        initial request.
        """
        token_provider = self.config['sasl_oauth_token_provider']

        # Only run if the #extensions() method is implemented by the clients Token Provider class
        # Builds up a string separated by \x01 via a dict of key value pairs
        if callable(getattr(token_provider, "extensions", None)) and len(token_provider.extensions()) > 0:
            msg = "\x01".join(["{}={}".format(k, v) for k, v in token_provider.extensions().items()])
            return "\x01" + msg
        else:
            return ""

    def blacked_out(self):
        """
        Return true if we are disconnected from the given node and can't
        re-establish a connection yet
        """
        if self.state is ConnectionStates.DISCONNECTED:
            if time.time() < self.last_attempt + self._reconnect_backoff:
                return True
        return False

    def connection_delay(self):
        """
        Return the number of milliseconds to wait, based on the connection
        state, before attempting to send data. When disconnected, this respects
        the reconnect backoff time. When connecting or connected, returns a very
        large number to handle slow/stalled connections.
        """
        time_waited = time.time() - (self.last_attempt or 0)
        if self.state is ConnectionStates.DISCONNECTED:
            return max(self._reconnect_backoff - time_waited, 0) * 1000
        else:
            # When connecting or connected, we should be able to delay
            # indefinitely since other events (connection or data acked) will
            # cause a wakeup once data can be sent.
            return float('inf')

    def connected(self):
        """Return True iff socket is connected."""
        return self.state is ConnectionStates.CONNECTED

    def connecting(self):
        """Returns True if still connecting (this may encompass several
        different states, such as SSL handshake, authorization, etc)."""
        return self.state in (ConnectionStates.CONNECTING,
                              ConnectionStates.HANDSHAKE,
                              ConnectionStates.AUTHENTICATING)

    def disconnected(self):
        """Return True iff socket is closed"""
        return self.state is ConnectionStates.DISCONNECTED

    def _reset_reconnect_backoff(self):
        self._failures = 0
        self._reconnect_backoff = self.config['reconnect_backoff_ms'] / 1000.0

    def _update_reconnect_backoff(self):
        # Do not mark as failure if there are more dns entries available to try
        if len(self._gai) > 0:
            return
        if self.config['reconnect_backoff_max_ms'] > self.config['reconnect_backoff_ms']:
            self._failures += 1
            self._reconnect_backoff = self.config['reconnect_backoff_ms'] * 2 ** (self._failures - 1)
            self._reconnect_backoff = min(self._reconnect_backoff, self.config['reconnect_backoff_max_ms'])
            self._reconnect_backoff *= uniform(0.8, 1.2)
            self._reconnect_backoff /= 1000.0
            log.debug('%s: reconnect backoff %s after %s failures', self, self._reconnect_backoff, self._failures)

    def _close_socket(self):
        if hasattr(self, '_sock') and self._sock is not None:
            self._sock.close()
            self._sock = None

    def __del__(self):
        self._close_socket()

    def close(self, error=None):
        """Close socket and fail all in-flight-requests.

        Arguments:
            error (Exception, optional): pending in-flight-requests
                will be failed with this exception.
                Default: kafka.errors.KafkaConnectionError.
        """
        if self.state is ConnectionStates.DISCONNECTED:
            return
        with self._lock:
            if self.state is ConnectionStates.DISCONNECTED:
                return
            log.info('%s: Closing connection. %s', self, error or '')
            self._update_reconnect_backoff()
            self._sasl_auth_future = None
            self._protocol = KafkaProtocol(
                client_id=self.config['client_id'],
                api_version=self.config['api_version'])
            self._send_buffer = b''
            if error is None:
                error = Errors.Cancelled(str(self))
            ifrs = list(self.in_flight_requests.items())
            self.in_flight_requests.clear()
            self.state = ConnectionStates.DISCONNECTED
            # To avoid race conditions and/or deadlocks
            # keep a reference to the socket but leave it
            # open until after the state_change_callback
            # This should give clients a change to deregister
            # the socket fd from selectors cleanly.
            sock = self._sock
            self._sock = None

        # drop lock before state change callback and processing futures
        self.config['state_change_callback'](self.node_id, sock, self)
        sock.close()
        for (_correlation_id, (future, _timestamp)) in ifrs:
            future.failure(error)

    def _can_send_recv(self):
        """Return True iff socket is ready for requests / responses"""
        return self.state in (ConnectionStates.AUTHENTICATING,
                              ConnectionStates.CONNECTED)

    def send(self, request, blocking=True):
        """Queue request for async network send, return Future()"""
        future = Future()
        if self.connecting():
            return future.failure(Errors.NodeNotReadyError(str(self)))
        elif not self.connected():
            return future.failure(Errors.KafkaConnectionError(str(self)))
        elif not self.can_send_more():
            return future.failure(Errors.TooManyInFlightRequests(str(self)))
        return self._send(request, blocking=blocking)

    def _send(self, request, blocking=True):
        future = Future()
        with self._lock:
            if not self._can_send_recv():
                # In this case, since we created the future above,
                # we know there are no callbacks/errbacks that could fire w/
                # lock. So failing + returning inline should be safe
                return future.failure(Errors.NodeNotReadyError(str(self)))

            correlation_id = self._protocol.send_request(request)

            log.debug('%s Request %d: %s', self, correlation_id, request)
            if request.expect_response():
                sent_time = time.time()
                assert correlation_id not in self.in_flight_requests, 'Correlation ID already in-flight!'
                self.in_flight_requests[correlation_id] = (future, sent_time)
            else:
                future.success(None)

        # Attempt to replicate behavior from prior to introduction of
        # send_pending_requests() / async sends
        if blocking:
            self.send_pending_requests()

        return future

    def send_pending_requests(self):
        """Attempts to send pending requests messages via blocking IO
        If all requests have been sent, return True
        Otherwise, if the socket is blocked and there are more bytes to send,
        return False.
        """
        try:
            with self._lock:
                if not self._can_send_recv():
                    return False
                data = self._protocol.send_bytes()
                total_bytes = self._send_bytes_blocking(data)

            if self._sensors:
                self._sensors.bytes_sent.record(total_bytes)
            return True

        except (ConnectionError, TimeoutError) as e:
            log.exception("Error sending request data to %s", self)
            error = Errors.KafkaConnectionError("%s: %s" % (self, e))
            self.close(error=error)
            return False

    def send_pending_requests_v2(self):
        """Attempts to send pending requests messages via non-blocking IO
        If all requests have been sent, return True
        Otherwise, if the socket is blocked and there are more bytes to send,
        return False.
        """
        try:
            with self._lock:
                if not self._can_send_recv():
                    return False

                # _protocol.send_bytes returns encoded requests to send
                # we send them via _send_bytes()
                # and hold leftover bytes in _send_buffer
                if not self._send_buffer:
                    self._send_buffer = self._protocol.send_bytes()

                total_bytes = 0
                if self._send_buffer:
                    total_bytes = self._send_bytes(self._send_buffer)
                    self._send_buffer = self._send_buffer[total_bytes:]

            if self._sensors:
                self._sensors.bytes_sent.record(total_bytes)
            # Return True iff send buffer is empty
            return len(self._send_buffer) == 0

        except (ConnectionError, TimeoutError, Exception) as e:
            log.exception("Error sending request data to %s", self)
            error = Errors.KafkaConnectionError("%s: %s" % (self, e))
            self.close(error=error)
            return False

    def can_send_more(self):
        """Return True unless there are max_in_flight_requests_per_connection."""
        max_ifrs = self.config['max_in_flight_requests_per_connection']
        return len(self.in_flight_requests) < max_ifrs

    def recv(self):
        """Non-blocking network receive.

        Return list of (response, future) tuples
        """
        responses = self._recv()
        if not responses and self.requests_timed_out():
            log.warning('%s timed out after %s ms. Closing connection.',
                        self, self.config['request_timeout_ms'])
            self.close(error=Errors.RequestTimedOutError(
                'Request timed out after %s ms' %
                self.config['request_timeout_ms']))
            return ()

        # augment responses w/ correlation_id, future, and timestamp
        for i, (correlation_id, response) in enumerate(responses):
            try:
                with self._lock:
                    (future, timestamp) = self.in_flight_requests.pop(correlation_id)
            except KeyError:
                self.close(Errors.KafkaConnectionError('Received unrecognized correlation id'))
                return ()
            latency_ms = (time.time() - timestamp) * 1000
            if self._sensors:
                self._sensors.request_time.record(latency_ms)

            log.debug('%s Response %d (%s ms): %s', self, correlation_id, latency_ms, response)
            responses[i] = (response, future)

        return responses

    def _recv(self):
        """Take all available bytes from socket, return list of any responses from parser"""
        recvd = []
        err = None
        with self._lock:
            if not self._can_send_recv():
                log.warning('%s cannot recv: socket not connected', self)
                return ()

            while len(recvd) < self.config['sock_chunk_buffer_count']:
                try:
                    data = self._sock.recv(self.config['sock_chunk_bytes'])
                    # We expect socket.recv to raise an exception if there are no
                    # bytes available to read from the socket in non-blocking mode.
                    # but if the socket is disconnected, we will get empty data
                    # without an exception raised
                    if not data:
                        log.error('%s: socket disconnected', self)
                        err = Errors.KafkaConnectionError('socket disconnected')
                        break
                    else:
                        recvd.append(data)

                except (SSLWantReadError, SSLWantWriteError):
                    break
                except (ConnectionError, TimeoutError) as e:
                    if six.PY2 and e.errno == errno.EWOULDBLOCK:
                        break
                    log.exception('%s: Error receiving network data'
                                  ' closing socket', self)
                    err = Errors.KafkaConnectionError(e)
                    break
                except BlockingIOError:
                    if six.PY3:
                        break
                    # For PY2 this is a catchall and should be re-raised
                    raise

            # Only process bytes if there was no connection exception
            if err is None:
                recvd_data = b''.join(recvd)
                if self._sensors:
                    self._sensors.bytes_received.record(len(recvd_data))

                # We need to keep the lock through protocol receipt
                # so that we ensure that the processed byte order is the
                # same as the received byte order
                try:
                    return self._protocol.receive_bytes(recvd_data)
                except Errors.KafkaProtocolError as e:
                    err = e

        self.close(error=err)
        return ()

    def requests_timed_out(self):
        with self._lock:
            if self.in_flight_requests:
                get_timestamp = lambda v: v[1]
                oldest_at = min(map(get_timestamp,
                                    self.in_flight_requests.values()))
                timeout = self.config['request_timeout_ms'] / 1000.0
                if time.time() >= oldest_at + timeout:
                    return True
            return False

    def _handle_api_version_response(self, response):
        error_type = Errors.for_code(response.error_code)
        assert error_type is Errors.NoError, "API version check failed"
        self._api_versions = dict([
            (api_key, (min_version, max_version))
            for api_key, min_version, max_version in response.api_versions
        ])
        return self._api_versions

    def get_api_versions(self):
        if self._api_versions is not None:
            return self._api_versions

        version = self.check_version()
        if version < (0, 10, 0):
            raise Errors.UnsupportedVersionError(
                "ApiVersion not supported by cluster version {} < 0.10.0"
                .format(version))
        # _api_versions is set as a side effect of check_versions() on a cluster
        # that supports 0.10.0 or later
        return self._api_versions

    def _infer_broker_version_from_api_versions(self, api_versions):
        # The logic here is to check the list of supported request versions
        # in reverse order. As soon as we find one that works, return it
        test_cases = [
            # format (<broker version>, <needed struct>)
            ((2, 5, 0), DescribeAclsRequest_v2),
            ((2, 4, 0), ProduceRequest[8]),
            ((2, 3, 0), FetchRequest[11]),
            ((2, 2, 0), OffsetRequest[5]),
            ((2, 1, 0), FetchRequest[10]),
            ((2, 0, 0), FetchRequest[8]),
            ((1, 1, 0), FetchRequest[7]),
            ((1, 0, 0), MetadataRequest[5]),
            ((0, 11, 0), MetadataRequest[4]),
            ((0, 10, 2), OffsetFetchRequest[2]),
            ((0, 10, 1), MetadataRequest[2]),
        ]

        # Get the best match of test cases
        for broker_version, struct in sorted(test_cases, reverse=True):
            if struct.API_KEY not in api_versions:
                continue
            min_version, max_version = api_versions[struct.API_KEY]
            if min_version <= struct.API_VERSION <= max_version:
                return broker_version

        # We know that ApiVersionResponse is only supported in 0.10+
        # so if all else fails, choose that
        return (0, 10, 0)

    def check_version(self, timeout=2, strict=False, topics=[]):
        """Attempt to guess the broker version.

        Note: This is a blocking call.

        Returns: version tuple, i.e. (0, 10), (0, 9), (0, 8, 2), ...
        """
        timeout_at = time.time() + timeout
        log.info('Probing node %s broker version', self.node_id)
        # Monkeypatch some connection configurations to avoid timeouts
        override_config = {
            'request_timeout_ms': timeout * 1000,
            'max_in_flight_requests_per_connection': 5
        }
        stashed = {}
        for key in override_config:
            stashed[key] = self.config[key]
            self.config[key] = override_config[key]

        def reset_override_configs():
            for key in stashed:
                self.config[key] = stashed[key]

        # kafka kills the connection when it doesn't recognize an API request
        # so we can send a test request and then follow immediately with a
        # vanilla MetadataRequest. If the server did not recognize the first
        # request, both will be failed with a ConnectionError that wraps
        # socket.error (32, 54, or 104)
        from kafka.protocol.admin import ApiVersionRequest, ListGroupsRequest
        from kafka.protocol.commit import OffsetFetchRequest, GroupCoordinatorRequest

        test_cases = [
            # All cases starting from 0.10 will be based on ApiVersionResponse
            ((0, 10), ApiVersionRequest[0]()),
            ((0, 9), ListGroupsRequest[0]()),
            ((0, 8, 2), GroupCoordinatorRequest[0]('kafka-python-default-group')),
            ((0, 8, 1), OffsetFetchRequest[0]('kafka-python-default-group', [])),
            ((0, 8, 0), MetadataRequest[0](topics)),
        ]

        for version, request in test_cases:
            if not self.connect_blocking(timeout_at - time.time()):
                reset_override_configs()
                raise Errors.NodeNotReadyError()
            f = self.send(request)
            # HACK: sleeping to wait for socket to send bytes
            time.sleep(0.1)
            # when broker receives an unrecognized request API
            # it abruptly closes our socket.
            # so we attempt to send a second request immediately
            # that we believe it will definitely recognize (metadata)
            # the attempt to write to a disconnected socket should
            # immediately fail and allow us to infer that the prior
            # request was unrecognized
            mr = self.send(MetadataRequest[0](topics))

            selector = self.config['selector']()
            selector.register(self._sock, selectors.EVENT_READ)
            while not (f.is_done and mr.is_done):
                selector.select(1)
                for response, future in self.recv():
                    future.success(response)
            selector.close()

            if f.succeeded():
                if isinstance(request, ApiVersionRequest[0]):
                    # Starting from 0.10 kafka broker we determine version
                    # by looking at ApiVersionResponse
                    api_versions = self._handle_api_version_response(f.value)
                    version = self._infer_broker_version_from_api_versions(api_versions)
                log.info('Broker version identified as %s', '.'.join(map(str, version)))
                log.info('Set configuration api_version=%s to skip auto'
                         ' check_version requests on startup', version)
                break

            # Only enable strict checking to verify that we understand failure
            # modes. For most users, the fact that the request failed should be
            # enough to rule out a particular broker version.
            if strict:
                # If the socket flush hack did not work (which should force the
                # connection to close and fail all pending requests), then we
                # get a basic Request Timeout. This is not ideal, but we'll deal
                if isinstance(f.exception, Errors.RequestTimedOutError):
                    pass

                # 0.9 brokers do not close the socket on unrecognized api
                # requests (bug...). In this case we expect to see a correlation
                # id mismatch
                elif (isinstance(f.exception, Errors.CorrelationIdError) and
                      version == (0, 10)):
                    pass
                elif six.PY2:
                    assert isinstance(f.exception.args[0], socket.error)
                    assert f.exception.args[0].errno in (32, 54, 104)
                else:
                    assert isinstance(f.exception.args[0], ConnectionError)
            log.info("Broker is not v%s -- it did not recognize %s",
                     version, request.__class__.__name__)
        else:
            reset_override_configs()
            raise Errors.UnrecognizedBrokerVersion()

        reset_override_configs()
        return version

    def __str__(self):
        return "<BrokerConnection node_id=%s host=%s:%d %s [%s %s]>" % (
            self.node_id, self.host, self.port, self.state,
            AFI_NAMES[self._sock_afi], self._sock_addr)


class BrokerConnectionMetrics(object):
    def __init__(self, metrics, metric_group_prefix, node_id):
        self.metrics = metrics

        # Any broker may have registered summary metrics already
        # but if not, we need to create them so we can set as parents below
        all_conns_transferred = metrics.get_sensor('bytes-sent-received')
        if not all_conns_transferred:
            metric_group_name = metric_group_prefix + '-metrics'

            bytes_transferred = metrics.sensor('bytes-sent-received')
            bytes_transferred.add(metrics.metric_name(
                'network-io-rate', metric_group_name,
                'The average number of network operations (reads or writes) on all'
                ' connections per second.'), Rate(sampled_stat=Count()))

            bytes_sent = metrics.sensor('bytes-sent',
                                        parents=[bytes_transferred])
            bytes_sent.add(metrics.metric_name(
                'outgoing-byte-rate', metric_group_name,
                'The average number of outgoing bytes sent per second to all'
                ' servers.'), Rate())
            bytes_sent.add(metrics.metric_name(
                'request-rate', metric_group_name,
                'The average number of requests sent per second.'),
                Rate(sampled_stat=Count()))
            bytes_sent.add(metrics.metric_name(
                'request-size-avg', metric_group_name,
                'The average size of all requests in the window.'), Avg())
            bytes_sent.add(metrics.metric_name(
                'request-size-max', metric_group_name,
                'The maximum size of any request sent in the window.'), Max())

            bytes_received = metrics.sensor('bytes-received',
                                            parents=[bytes_transferred])
            bytes_received.add(metrics.metric_name(
                'incoming-byte-rate', metric_group_name,
                'Bytes/second read off all sockets'), Rate())
            bytes_received.add(metrics.metric_name(
                'response-rate', metric_group_name,
                'Responses received sent per second.'),
                Rate(sampled_stat=Count()))

            request_latency = metrics.sensor('request-latency')
            request_latency.add(metrics.metric_name(
                'request-latency-avg', metric_group_name,
                'The average request latency in ms.'),
                Avg())
            request_latency.add(metrics.metric_name(
                'request-latency-max', metric_group_name,
                'The maximum request latency in ms.'),
                Max())

        # if one sensor of the metrics has been registered for the connection,
        # then all other sensors should have been registered; and vice versa
        node_str = 'node-{0}'.format(node_id)
        node_sensor = metrics.get_sensor(node_str + '.bytes-sent')
        if not node_sensor:
            metric_group_name = metric_group_prefix + '-node-metrics.' + node_str

            bytes_sent = metrics.sensor(
                node_str + '.bytes-sent',
                parents=[metrics.get_sensor('bytes-sent')])
            bytes_sent.add(metrics.metric_name(
                'outgoing-byte-rate', metric_group_name,
                'The average number of outgoing bytes sent per second.'),
                Rate())
            bytes_sent.add(metrics.metric_name(
                'request-rate', metric_group_name,
                'The average number of requests sent per second.'),
                Rate(sampled_stat=Count()))
            bytes_sent.add(metrics.metric_name(
                'request-size-avg', metric_group_name,
                'The average size of all requests in the window.'),
                Avg())
            bytes_sent.add(metrics.metric_name(
                'request-size-max', metric_group_name,
                'The maximum size of any request sent in the window.'),
                Max())

            bytes_received = metrics.sensor(
                node_str + '.bytes-received',
                parents=[metrics.get_sensor('bytes-received')])
            bytes_received.add(metrics.metric_name(
                'incoming-byte-rate', metric_group_name,
                'Bytes/second read off node-connection socket'),
                Rate())
            bytes_received.add(metrics.metric_name(
                'response-rate', metric_group_name,
                'The average number of responses received per second.'),
                Rate(sampled_stat=Count()))

            request_time = metrics.sensor(
                node_str + '.latency',
                parents=[metrics.get_sensor('request-latency')])
            request_time.add(metrics.metric_name(
                'request-latency-avg', metric_group_name,
                'The average request latency in ms.'),
                Avg())
            request_time.add(metrics.metric_name(
                'request-latency-max', metric_group_name,
                'The maximum request latency in ms.'),
                Max())

        self.bytes_sent = metrics.sensor(node_str + '.bytes-sent')
        self.bytes_received = metrics.sensor(node_str + '.bytes-received')
        self.request_time = metrics.sensor(node_str + '.latency')


def _address_family(address):
    """
        Attempt to determine the family of an address (or hostname)

        :return: either socket.AF_INET or socket.AF_INET6 or socket.AF_UNSPEC if the address family
                 could not be determined
    """
    if address.startswith('[') and address.endswith(']'):
        return socket.AF_INET6
    for af in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.inet_pton(af, address)
            return af
        except (ValueError, AttributeError, socket.error):
            continue
    return socket.AF_UNSPEC


def get_ip_port_afi(host_and_port_str):
    """
        Parse the IP and port from a string in the format of:

            * host_or_ip          <- Can be either IPv4 address literal or hostname/fqdn
            * host_or_ipv4:port   <- Can be either IPv4 address literal or hostname/fqdn
            * [host_or_ip]        <- IPv6 address literal
            * [host_or_ip]:port.  <- IPv6 address literal

        .. note:: IPv6 address literals with ports *must* be enclosed in brackets

        .. note:: If the port is not specified, default will be returned.

        :return: tuple (host, port, afi), afi will be socket.AF_INET or socket.AF_INET6 or socket.AF_UNSPEC
    """
    host_and_port_str = host_and_port_str.strip()
    if host_and_port_str.startswith('['):
        af = socket.AF_INET6
        host, rest = host_and_port_str[1:].split(']')
        if rest:
            port = int(rest[1:])
        else:
            port = DEFAULT_KAFKA_PORT
        return host, port, af
    else:
        if ':' not in host_and_port_str:
            af = _address_family(host_and_port_str)
            return host_and_port_str, DEFAULT_KAFKA_PORT, af
        else:
            # now we have something with a colon in it and no square brackets. It could be
            # either an IPv6 address literal (e.g., "::1") or an IP:port pair or a host:port pair
            try:
                # if it decodes as an IPv6 address, use that
                socket.inet_pton(socket.AF_INET6, host_and_port_str)
                return host_and_port_str, DEFAULT_KAFKA_PORT, socket.AF_INET6
            except AttributeError:
                log.warning('socket.inet_pton not available on this platform.'
                            ' consider `pip install win_inet_pton`')
                pass
            except (ValueError, socket.error):
                # it's a host:port pair
                pass
            host, port = host_and_port_str.rsplit(':', 1)
            port = int(port)

            af = _address_family(host)
            return host, port, af


def collect_hosts(hosts, randomize=True):
    """
    Collects a comma-separated set of hosts (host:port) and optionally
    randomize the returned list.
    """

    if isinstance(hosts, six.string_types):
        hosts = hosts.strip().split(',')

    result = []
    afi = socket.AF_INET
    for host_port in hosts:

        host, port, afi = get_ip_port_afi(host_port)

        if port < 0:
            port = DEFAULT_KAFKA_PORT

        result.append((host, port, afi))

    if randomize:
        shuffle(result)

    return result


def is_inet_4_or_6(gai):
    """Given a getaddrinfo struct, return True iff ipv4 or ipv6"""
    return gai[0] in (socket.AF_INET, socket.AF_INET6)


def dns_lookup(host, port, afi=socket.AF_UNSPEC):
    """Returns a list of getaddrinfo structs, optionally filtered to an afi (ipv4 / ipv6)"""
    # XXX: all DNS functions in Python are blocking. If we really
    # want to be non-blocking here, we need to use a 3rd-party
    # library like python-adns, or move resolution onto its
    # own thread. This will be subject to the default libc
    # name resolution timeout (5s on most Linux boxes)
    try:
        return list(filter(is_inet_4_or_6,
                           socket.getaddrinfo(host, port, afi,
                                              socket.SOCK_STREAM)))
    except socket.gaierror as ex:
        log.warning('DNS lookup failed for %s:%d,'
                    ' exception was %s. Is your'
                    ' advertised.listeners (called'
                    ' advertised.host.name before Kafka 9)'
                    ' correct and resolvable?',
                    host, port, ex)
        return []
