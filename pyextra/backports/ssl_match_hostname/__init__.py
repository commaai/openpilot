"""The match_hostname() function from Python 3.7.0, essential when using SSL."""

import sys
import socket as _socket

try:
    # Divergence: Python-3.7+'s _ssl has this exception type but older Pythons do not
    from _ssl import SSLCertVerificationError
    CertificateError = SSLCertVerificationError
except:
    class CertificateError(ValueError):
        pass


__version__ = '3.7.0.1'


# Divergence: Added to deal with ipaddess as bytes on python2
def _to_text(obj):
    if isinstance(obj, str) and sys.version_info < (3,):
        obj = unicode(obj, encoding='ascii', errors='strict')
    elif sys.version_info >= (3,) and isinstance(obj, bytes):
        obj = str(obj, encoding='ascii', errors='strict')
    return obj


def _to_bytes(obj):
    if isinstance(obj, str) and sys.version_info >= (3,):
        obj = bytes(obj, encoding='ascii', errors='strict')
    elif sys.version_info < (3,) and isinstance(obj, unicode):
        obj = obj.encode('ascii', 'strict')
    return obj


def _dnsname_match(dn, hostname):
    """Matching according to RFC 6125, section 6.4.3

    - Hostnames are compared lower case.
    - For IDNA, both dn and hostname must be encoded as IDN A-label (ACE).
    - Partial wildcards like 'www*.example.org', multiple wildcards, sole
      wildcard or wildcards in labels other then the left-most label are not
      supported and a CertificateError is raised.
    - A wildcard must match at least one character.
    """
    if not dn:
        return False

    wildcards = dn.count('*')
    # speed up common case w/o wildcards
    if not wildcards:
        return dn.lower() == hostname.lower()

    if wildcards > 1:
        # Divergence .format() to percent formatting for Python < 2.6
        raise CertificateError(
            "too many wildcards in certificate DNS name: %s" % repr(dn))

    dn_leftmost, sep, dn_remainder = dn.partition('.')

    if '*' in dn_remainder:
        # Only match wildcard in leftmost segment.
        # Divergence .format() to percent formatting for Python < 2.6
        raise CertificateError(
            "wildcard can only be present in the leftmost label: "
            "%s." % repr(dn))

    if not sep:
        # no right side
        # Divergence .format() to percent formatting for Python < 2.6
        raise CertificateError(
            "sole wildcard without additional labels are not support: "
            "%s." % repr(dn))

    if dn_leftmost != '*':
        # no partial wildcard matching
        # Divergence .format() to percent formatting for Python < 2.6
        raise CertificateError(
            "partial wildcards in leftmost label are not supported: "
            "%s." % repr(dn))

    hostname_leftmost, sep, hostname_remainder = hostname.partition('.')
    if not hostname_leftmost or not sep:
        # wildcard must match at least one char
        return False
    return dn_remainder.lower() == hostname_remainder.lower()


def _inet_paton(ipname):
    """Try to convert an IP address to packed binary form

    Supports IPv4 addresses on all platforms and IPv6 on platforms with IPv6
    support.
    """
    # inet_aton() also accepts strings like '1'
    # Divergence: We make sure we have native string type for all python versions
    try:
        b_ipname = _to_bytes(ipname)
    except UnicodeError:
        raise ValueError("%s must be an all-ascii string." % repr(ipname))

    # Set ipname in native string format
    if sys.version_info < (3,):
        n_ipname = b_ipname
    else:
        n_ipname = ipname

    if n_ipname.count('.') == 3:
        try:
            return _socket.inet_aton(n_ipname)
        # Divergence: OSError on late python3.  socket.error earlier.
        # Null bytes generate ValueError on python3(we want to raise
        # ValueError anyway), TypeError # earlier
        except (OSError, _socket.error, TypeError):
            pass

    try:
        return _socket.inet_pton(_socket.AF_INET6, n_ipname)
    # Divergence: OSError on late python3.  socket.error earlier.
    # Null bytes generate ValueError on python3(we want to raise
    # ValueError anyway), TypeError # earlier
    except (OSError, _socket.error, TypeError):
        # Divergence .format() to percent formatting for Python < 2.6
        raise ValueError("%s is neither an IPv4 nor an IP6 "
                         "address." % repr(ipname))
    except AttributeError:
        # AF_INET6 not available
        pass

    # Divergence .format() to percent formatting for Python < 2.6
    raise ValueError("%s is not an IPv4 address." % repr(ipname))


def _ipaddress_match(ipname, host_ip):
    """Exact matching of IP addresses.

    RFC 6125 explicitly doesn't define an algorithm for this
    (section 1.7.2 - "Out of Scope").
    """
    # OpenSSL may add a trailing newline to a subjectAltName's IP address
    ip = _inet_paton(ipname.rstrip())
    return ip == host_ip


def match_hostname(cert, hostname):
    """Verify that *cert* (in decoded format as returned by
    SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125
    rules are followed.

    The function matches IP addresses rather than dNSNames if hostname is a
    valid ipaddress string. IPv4 addresses are supported on all platforms.
    IPv6 addresses are supported on platforms with IPv6 support (AF_INET6
    and inet_pton).

    CertificateError is raised on failure. On success, the function
    returns nothing.
    """
    if not cert:
        raise ValueError("empty or no certificate, match_hostname needs a "
                         "SSL socket or SSL context with either "
                         "CERT_OPTIONAL or CERT_REQUIRED")
    try:
        # Divergence: Deal with hostname as bytes
        host_ip = _inet_paton(_to_text(hostname))
    except ValueError:
        # Not an IP address (common case)
        host_ip = None
    except UnicodeError:
        # Divergence: Deal with hostname as byte strings.
        # IP addresses should be all ascii, so we consider it not
        # an IP address if this fails
        host_ip = None
    dnsnames = []
    san = cert.get('subjectAltName', ())
    for key, value in san:
        if key == 'DNS':
            if host_ip is None and _dnsname_match(value, hostname):
                return
            dnsnames.append(value)
        elif key == 'IP Address':
            if host_ip is not None and _ipaddress_match(value, host_ip):
                return
            dnsnames.append(value)
    if not dnsnames:
        # The subject is only checked when there is no dNSName entry
        # in subjectAltName
        for sub in cert.get('subject', ()):
            for key, value in sub:
                # XXX according to RFC 2818, the most specific Common Name
                # must be used.
                if key == 'commonName':
                    if _dnsname_match(value, hostname):
                        return
                    dnsnames.append(value)
    if len(dnsnames) > 1:
        raise CertificateError("hostname %r "
            "doesn't match either of %s"
            % (hostname, ', '.join(map(repr, dnsnames))))
    elif len(dnsnames) == 1:
        raise CertificateError("hostname %r "
            "doesn't match %r"
            % (hostname, dnsnames[0]))
    else:
        raise CertificateError("no appropriate commonName or "
            "subjectAltName fields were found")
