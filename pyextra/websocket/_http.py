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
import errno
import os
import socket
import sys

import six

from ._exceptions import *
from ._logging import *
from ._socket import*
from ._ssl_compat import *
from ._url import *

if six.PY3:
    from base64 import encodebytes as base64encode
else:
    from base64 import encodestring as base64encode

__all__ = ["proxy_info", "connect", "read_headers"]

try:
    import socks
    ProxyConnectionError = socks.ProxyConnectionError
    HAS_PYSOCKS = True
except:
    class ProxyConnectionError(BaseException):
        pass
    HAS_PYSOCKS = False

class proxy_info(object):

    def __init__(self, **options):
        self.type = options.get("proxy_type") or "http"
        if not(self.type in ['http', 'socks4', 'socks5', 'socks5h']):
            raise ValueError("proxy_type must be 'http', 'socks4', 'socks5' or 'socks5h'")
        self.host = options.get("http_proxy_host", None)
        if self.host:
            self.port = options.get("http_proxy_port", 0)
            self.auth = options.get("http_proxy_auth", None)
            self.no_proxy = options.get("http_no_proxy", None)
        else:
            self.port = 0
            self.auth = None
            self.no_proxy = None

def _open_proxied_socket(url, options, proxy):
    hostname, port, resource, is_secure = parse_url(url)

    if not HAS_PYSOCKS:
        raise WebSocketException("PySocks module not found.")

    ptype = socks.SOCKS5
    rdns = False
    if proxy.type == "socks4":
        ptype = socks.SOCKS4
    if proxy.type == "http":
        ptype = socks.HTTP
    if proxy.type[-1] == "h":
        rdns = True

    sock = socks.create_connection(
            (hostname, port),
            proxy_type = ptype,
            proxy_addr = proxy.host,
            proxy_port = proxy.port,
            proxy_rdns = rdns,
            proxy_username = proxy.auth[0] if proxy.auth else None,
            proxy_password = proxy.auth[1] if proxy.auth else None,
            timeout = options.timeout,
            socket_options = DEFAULT_SOCKET_OPTION + options.sockopt
    )

    if is_secure:
        if HAVE_SSL:
            sock = _ssl_socket(sock, options.sslopt, hostname)
        else:
            raise WebSocketException("SSL not available.")

    return sock, (hostname, port, resource)


def connect(url, options, proxy, socket):
    if proxy.host and not socket and not (proxy.type == 'http'):
        return _open_proxied_socket(url, options, proxy)

    hostname, port, resource, is_secure = parse_url(url)

    if socket:
        return socket, (hostname, port, resource)

    addrinfo_list, need_tunnel, auth = _get_addrinfo_list(
        hostname, port, is_secure, proxy)
    if not addrinfo_list:
        raise WebSocketException(
            "Host not found.: " + hostname + ":" + str(port))

    sock = None
    try:
        sock = _open_socket(addrinfo_list, options.sockopt, options.timeout)
        if need_tunnel:
            sock = _tunnel(sock, hostname, port, auth)

        if is_secure:
            if HAVE_SSL:
                sock = _ssl_socket(sock, options.sslopt, hostname)
            else:
                raise WebSocketException("SSL not available.")

        return sock, (hostname, port, resource)
    except:
        if sock:
            sock.close()
        raise


def _get_addrinfo_list(hostname, port, is_secure, proxy):
    phost, pport, pauth = get_proxy_info(
        hostname, is_secure, proxy.host, proxy.port, proxy.auth, proxy.no_proxy)
    try:
        if not phost:
            addrinfo_list = [ai for ai in socket.getaddrinfo(
                hostname, port, 0, 0, socket.SOL_TCP)
                if (ai[0] == socket.AF_INET6 and socket.has_ipv6) or ai[0] != socket.AF_INET6
            ]
            return addrinfo_list, False, None
        else:
            pport = pport and pport or 80
            # when running on windows 10, the getaddrinfo used above
            # returns a socktype 0. This generates an error exception:
            #_on_error: exception Socket type must be stream or datagram, not 0
            # Force the socket type to SOCK_STREAM
            addrinfo_list = socket.getaddrinfo(phost, pport, 0, socket.SOCK_STREAM, socket.SOL_TCP)
            return addrinfo_list, True, pauth
    except socket.gaierror as e:
        raise WebSocketAddressException(e)


def _open_socket(addrinfo_list, sockopt, timeout):
    err = None
    for addrinfo in addrinfo_list:
        family, socktype, proto = addrinfo[:3]
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        for opts in DEFAULT_SOCKET_OPTION:
            sock.setsockopt(*opts)
        for opts in sockopt:
            sock.setsockopt(*opts)

        address = addrinfo[4]
        err = None
        while not err:
            try:
                sock.connect(address)
            except ProxyConnectionError as error:
                err = WebSocketProxyException(str(error))
                err.remote_ip = str(address[0])
                continue
            except socket.error as error:
                error.remote_ip = str(address[0])
                try:
                    eConnRefused = (errno.ECONNREFUSED, errno.WSAECONNREFUSED)
                except:
                    eConnRefused = (errno.ECONNREFUSED, )
                if error.errno == errno.EINTR:
                    continue
                elif error.errno in eConnRefused:
                    err = error
                    continue
                else:
                    raise error
            else:
                break
        else:
            continue
        break
    else:
        if err:
            raise err

    return sock


def _can_use_sni():
    return six.PY2 and sys.version_info >= (2, 7, 9) or sys.version_info >= (3, 2)


def _wrap_sni_socket(sock, sslopt, hostname, check_hostname):
    context = ssl.SSLContext(sslopt.get('ssl_version', ssl.PROTOCOL_SSLv23))

    if sslopt.get('cert_reqs', ssl.CERT_NONE) != ssl.CERT_NONE:
        cafile = sslopt.get('ca_certs', None)
        capath = sslopt.get('ca_cert_path', None)
        if cafile or capath:
            context.load_verify_locations(cafile=cafile, capath=capath)
        elif hasattr(context, 'load_default_certs'):
            context.load_default_certs(ssl.Purpose.SERVER_AUTH)
    if sslopt.get('certfile', None):
        context.load_cert_chain(
            sslopt['certfile'],
            sslopt.get('keyfile', None),
            sslopt.get('password', None),
        )
    # see
    # https://github.com/liris/websocket-client/commit/b96a2e8fa765753e82eea531adb19716b52ca3ca#commitcomment-10803153
    context.verify_mode = sslopt['cert_reqs']
    if HAVE_CONTEXT_CHECK_HOSTNAME:
        context.check_hostname = check_hostname
    if 'ciphers' in sslopt:
        context.set_ciphers(sslopt['ciphers'])
    if 'cert_chain' in sslopt:
        certfile, keyfile, password = sslopt['cert_chain']
        context.load_cert_chain(certfile, keyfile, password)
    if 'ecdh_curve' in sslopt:
        context.set_ecdh_curve(sslopt['ecdh_curve'])

    return context.wrap_socket(
        sock,
        do_handshake_on_connect=sslopt.get('do_handshake_on_connect', True),
        suppress_ragged_eofs=sslopt.get('suppress_ragged_eofs', True),
        server_hostname=hostname,
    )


def _ssl_socket(sock, user_sslopt, hostname):
    sslopt = dict(cert_reqs=ssl.CERT_REQUIRED)
    sslopt.update(user_sslopt)

    certPath = os.environ.get('WEBSOCKET_CLIENT_CA_BUNDLE')
    if certPath and os.path.isfile(certPath) \
            and user_sslopt.get('ca_certs', None) is None \
            and user_sslopt.get('ca_cert', None) is None:
        sslopt['ca_certs'] = certPath
    elif certPath and os.path.isdir(certPath) \
            and user_sslopt.get('ca_cert_path', None) is None:
        sslopt['ca_cert_path'] = certPath

    check_hostname = sslopt["cert_reqs"] != ssl.CERT_NONE and sslopt.pop(
        'check_hostname', True)

    if _can_use_sni():
        sock = _wrap_sni_socket(sock, sslopt, hostname, check_hostname)
    else:
        sslopt.pop('check_hostname', True)
        sock = ssl.wrap_socket(sock, **sslopt)

    if not HAVE_CONTEXT_CHECK_HOSTNAME and check_hostname:
        match_hostname(sock.getpeercert(), hostname)

    return sock


def _tunnel(sock, host, port, auth):
    debug("Connecting proxy...")
    connect_header = "CONNECT %s:%d HTTP/1.0\r\n" % (host, port)
    # TODO: support digest auth.
    if auth and auth[0]:
        auth_str = auth[0]
        if auth[1]:
            auth_str += ":" + auth[1]
        encoded_str = base64encode(auth_str.encode()).strip().decode()
        connect_header += "Proxy-Authorization: Basic %s\r\n" % encoded_str
    connect_header += "\r\n"
    dump("request header", connect_header)

    send(sock, connect_header)

    try:
        status, resp_headers, status_message = read_headers(sock)
    except Exception as e:
        raise WebSocketProxyException(str(e))

    if status != 200:
        raise WebSocketProxyException(
            "failed CONNECT via proxy status: %r" % status)

    return sock


def read_headers(sock):
    status = None
    status_message = None
    headers = {}
    trace("--- response header ---")

    while True:
        line = recv_line(sock)
        line = line.decode('utf-8').strip()
        if not line:
            break
        trace(line)
        if not status:

            status_info = line.split(" ", 2)
            status = int(status_info[1])
            if len(status_info) > 2:
                status_message = status_info[2]
        else:
            kv = line.split(":", 1)
            if len(kv) == 2:
                key, value = kv
                headers[key.lower()] = value.strip()
            else:
                raise WebSocketException("Invalid header")

    trace("-----------------------")

    return status, headers, status_message
