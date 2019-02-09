/*  =========================================================================
    zproxy - run a steerable proxy in the background

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZPROXY_H_INCLUDED__
#define __ZPROXY_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
//  Create new zproxy actor instance. The proxy switches messages between
//  a frontend socket and a backend socket; use the FRONTEND and BACKEND
//  commands to configure these:
//
//      zactor_t *proxy = zactor_new (zproxy, NULL);
//
//  Destroy zproxy instance. This destroys the two sockets and stops any
//  message flow between them:
//
//      zactor_destroy (&proxy);
//
//  Note that all zproxy commands are synchronous, so your application always
//  waits for a signal from the actor after each command.
//
//  Enable verbose logging of commands and activity:
//
//      zstr_send (proxy, "VERBOSE");
//      zsock_wait (proxy);
//
//  Specify frontend socket type -- see zsock_type_str () -- and attach to
//  endpoints, see zsock_attach (). Note that a proxy socket is always
//  serverish:
//
//      zstr_sendx (proxy, "FRONTEND", "XSUB", endpoints, NULL);
//      zsock_wait (proxy);
//
//  Specify backend socket type -- see zsock_type_str () -- and attach to
//  endpoints, see zsock_attach (). Note that a proxy socket is always
//  serverish:
//
//      zstr_sendx (proxy, "BACKEND", "XPUB", endpoints, NULL);
//      zsock_wait (proxy);
//
//  Capture all proxied messages; these are delivered to the application
//  via an inproc PULL socket that you have already bound to the specified
//  endpoint:
//
//      zstr_sendx (proxy, "CAPTURE", endpoint, NULL);
//      zsock_wait (proxy);
//
//  Pause the proxy. A paused proxy will cease processing messages, causing
//  them to be queued up and potentially hit the high-water mark on the
//  frontend or backend socket, causing messages to be dropped, or writing
//  applications to block:
//
//      zstr_sendx (proxy, "PAUSE", NULL);
//      zsock_wait (proxy);
//
//  Resume the proxy. Note that the proxy starts automatically as soon as it
//  has a properly attached frontend and backend socket:
//
//      zstr_sendx (proxy, "RESUME", NULL);
//      zsock_wait (proxy);
//
//  Configure an authentication domain for the "FRONTEND" or "BACKEND" proxy
//  socket -- see zsock_set_zap_domain (). Call before binding socket:
//
//      zstr_sendx (proxy, "DOMAIN", "FRONTEND", "global", NULL);
//      zsock_wait (proxy);
//
//  Configure PLAIN authentication for the "FRONTEND" or "BACKEND" proxy
//  socket -- see zsock_set_plain_server (). Call before binding socket:
//
//      zstr_sendx (proxy, "PLAIN", "BACKEND", NULL);
//      zsock_wait (proxy);
//
//  Configure CURVE authentication for the "FRONTEND" or "BACKEND" proxy
//  socket -- see zsock_set_curve_server () -- specifying both the public and
//  secret keys of a certificate as Z85 armored strings -- see
//  zcert_public_txt () and zcert_secret_txt (). Call before binding socket:
//
//      zstr_sendx (proxy, "CURVE", "FRONTEND", public_txt, secret_txt, NULL);
//      zsock_wait (proxy);
//
//  This is the zproxy constructor as a zactor_fn; the argument is a
//  character string specifying frontend and backend socket types as two
//  uppercase strings separated by a hyphen:
CZMQ_EXPORT void
    zproxy (zsock_t *pipe, void *unused);

//  Selftest
CZMQ_EXPORT void
    zproxy_test (bool verbose);
//  @end

#ifdef __cplusplus
}
#endif

#endif
