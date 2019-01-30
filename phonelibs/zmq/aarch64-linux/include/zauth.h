/*  =========================================================================
    zauth - authentication for ZeroMQ security mechanisms

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZAUTH_H_INCLUDED__
#define __ZAUTH_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
#define CURVE_ALLOW_ANY "*"

//  CZMQ v3 API (for use with zsock, not zsocket, which is deprecated).
//
//  Create new zauth actor instance. This installs authentication on all 
//  zsock sockets. Until you add policies, all incoming NULL connections are
//  allowed (classic ZeroMQ behaviour), and all PLAIN and CURVE connections
//  are denied:
//  
//      zactor_t *auth = zactor_new (zauth, NULL);
//
//  Destroy zauth instance. This removes authentication and allows all
//  connections to pass, without authentication:
//  
//      zactor_destroy (&auth);
//
//  Note that all zauth commands are synchronous, so your application always
//  waits for a signal from the actor after each command.
//
//  Enable verbose logging of commands and activity. Verbose logging can help
//  debug non-trivial authentication policies:
//
//      zstr_send (auth, "VERBOSE");
//      zsock_wait (auth);
//
//  Allow (whitelist) a list of IP addresses. For NULL, all clients from
//  these addresses will be accepted. For PLAIN and CURVE, they will be
//  allowed to continue with authentication. You can call this method
//  multiple times to whitelist more IP addresses. If you whitelist one
//  or more addresses, any non-whitelisted addresses are treated as
//  blacklisted:
//  
//      zstr_sendx (auth, "ALLOW", "127.0.0.1", "127.0.0.2", NULL);
//      zsock_wait (auth);
//  
//  Deny (blacklist) a list of IP addresses. For all security mechanisms,
//  this rejects the connection without any further authentication. Use
//  either a whitelist, or a blacklist, not not both. If you define both
//  a whitelist and a blacklist, only the whitelist takes effect:
//  
//      zstr_sendx (auth, "DENY", "192.168.0.1", "192.168.0.2", NULL);
//      zsock_wait (auth);
//
//  Configure PLAIN authentication using a plain-text password file. You can
//  modify the password file at any time; zauth will reload it automatically
//  if modified externally:
//  
//      zstr_sendx (auth, "PLAIN", filename, NULL);
//      zsock_wait (auth);
//
//  Configure CURVE authentication, using a directory that holds all public
//  client certificates, i.e. their public keys. The certificates must be in
//  zcert_save format. You can add and remove certificates in that directory
//  at any time. To allow all client keys without checking, specify
//  CURVE_ALLOW_ANY for the directory name:
//
//      zstr_sendx (auth, "CURVE", directory, NULL);
//      zsock_wait (auth);
//
//  Configure GSSAPI authentication, using an underlying mechanism (usually
//  Kerberos) to establish a secure context and perform mutual authentication:
//
//      zstr_sendx (auth, "GSSAPI", NULL);
//      zsock_wait (auth);
//
//  This is the zauth constructor as a zactor_fn:
CZMQ_EXPORT void
    zauth (zsock_t *pipe, void *certstore);

//  Selftest
CZMQ_EXPORT void
    zauth_test (bool verbose);
//  @end

#ifdef __cplusplus
}
#endif

#endif
