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

import os
import socket
import struct

from six.moves.urllib.parse import urlparse


__all__ = ["parse_url", "get_proxy_info"]


def parse_url(url):
    """
    parse url and the result is tuple of
    (hostname, port, resource path and the flag of secure mode)

    url: url string.
    """
    if ":" not in url:
        raise ValueError("url is invalid")

    scheme, url = url.split(":", 1)

    parsed = urlparse(url, scheme="ws")
    if parsed.hostname:
        hostname = parsed.hostname
    else:
        raise ValueError("hostname is invalid")
    port = 0
    if parsed.port:
        port = parsed.port

    is_secure = False
    if scheme == "ws":
        if not port:
            port = 80
    elif scheme == "wss":
        is_secure = True
        if not port:
            port = 443
    else:
        raise ValueError("scheme %s is invalid" % scheme)

    if parsed.path:
        resource = parsed.path
    else:
        resource = "/"

    if parsed.query:
        resource += "?" + parsed.query

    return hostname, port, resource, is_secure


DEFAULT_NO_PROXY_HOST = ["localhost", "127.0.0.1"]


def _is_ip_address(addr):
    try:
        socket.inet_aton(addr)
    except socket.error:
        return False
    else:
        return True


def _is_subnet_address(hostname):
    try:
        addr, netmask = hostname.split("/")
        return _is_ip_address(addr) and 0 <= int(netmask) < 32
    except ValueError:
        return False


def _is_address_in_network(ip, net):
    ipaddr = struct.unpack('I', socket.inet_aton(ip))[0]
    netaddr, bits = net.split('/')
    netmask = struct.unpack('I', socket.inet_aton(netaddr))[0] & ((2 << int(bits) - 1) - 1)
    return ipaddr & netmask == netmask


def _is_no_proxy_host(hostname, no_proxy):
    if not no_proxy:
        v = os.environ.get("no_proxy", "").replace(" ", "")
        no_proxy = v.split(",")
    if not no_proxy:
        no_proxy = DEFAULT_NO_PROXY_HOST

    if hostname in no_proxy:
        return True
    elif _is_ip_address(hostname):
        return any([_is_address_in_network(hostname, subnet) for subnet in no_proxy if _is_subnet_address(subnet)])

    return False


def get_proxy_info(
        hostname, is_secure, proxy_host=None, proxy_port=0, proxy_auth=None,
        no_proxy=None, proxy_type='http'):
    """
    try to retrieve proxy host and port from environment
    if not provided in options.
    result is (proxy_host, proxy_port, proxy_auth).
    proxy_auth is tuple of username and password
     of proxy authentication information.

    hostname: websocket server name.

    is_secure:  is the connection secure? (wss)
                looks for "https_proxy" in env
                before falling back to "http_proxy"

    options:    "http_proxy_host" - http proxy host name.
                "http_proxy_port" - http proxy port.
                "http_no_proxy"   - host names, which doesn't use proxy.
                "http_proxy_auth" - http proxy auth information.
                                    tuple of username and password.
                                    default is None
                "proxy_type"      - if set to "socks5" PySocks wrapper
                                    will be used in place of a http proxy.
                                    default is "http"
    """
    if _is_no_proxy_host(hostname, no_proxy):
        return None, 0, None

    if proxy_host:
        port = proxy_port
        auth = proxy_auth
        return proxy_host, port, auth

    env_keys = ["http_proxy"]
    if is_secure:
        env_keys.insert(0, "https_proxy")

    for key in env_keys:
        value = os.environ.get(key, None)
        if value:
            proxy = urlparse(value)
            auth = (proxy.username, proxy.password) if proxy.username else None
            return proxy.hostname, proxy.port, auth

    return None, 0, None
