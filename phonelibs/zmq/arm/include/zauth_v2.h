/*  =========================================================================
    zauth_v2 - authentication for ZeroMQ servers (deprecated)

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZAUTH_V2_H_INCLUDED__
#define __ZAUTH_V2_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
#ifndef CURVE_ALLOW_ANY
#   define CURVE_ALLOW_ANY "*"
#endif

//  Constructor
//  Install authentication for the specified context. Returns a new zauth
//  object that you can use to configure authentication. Note that until you
//  add policies, all incoming NULL connections are allowed (classic ZeroMQ
//  behaviour), and all PLAIN and CURVE connections are denied. If there was
//  an error during initialization, returns NULL.
CZMQ_EXPORT zauth_t *
    zauth_new (zctx_t *ctx);
    
//  Destructor
CZMQ_EXPORT void
    zauth_destroy (zauth_t **self_p);

//  Allow (whitelist) a single IP address. For NULL, all clients from this
//  address will be accepted. For PLAIN and CURVE, they will be allowed to
//  continue with authentication. You can call this method multiple times 
//  to whitelist multiple IP addresses. If you whitelist a single address,
//  any non-whitelisted addresses are treated as blacklisted.
CZMQ_EXPORT void
    zauth_allow (zauth_t *self, const char *address);

//  Deny (blacklist) a single IP address. For all security mechanisms, this
//  rejects the connection without any further authentication. Use either a
//  whitelist, or a blacklist, not not both. If you define both a whitelist 
//  and a blacklist, only the whitelist takes effect.
CZMQ_EXPORT void
    zauth_deny (zauth_t *self, const char *address);

//  Configure PLAIN authentication for a given domain. PLAIN authentication
//  uses a plain-text password file. To cover all domains, use "*". You can
//  modify the password file at any time; it is reloaded automatically.
CZMQ_EXPORT void
    zauth_configure_plain (zauth_t *self, const char *domain, const char *filename);
    
//  Configure CURVE authentication for a given domain. CURVE authentication
//  uses a directory that holds all public client certificates, i.e. their
//  public keys. The certificates must be in zcert_save () format. To cover
//  all domains, use "*". You can add and remove certificates in that
//  directory at any time. To allow all client keys without checking, specify
//  CURVE_ALLOW_ANY for the location.
CZMQ_EXPORT void
    zauth_configure_curve (zauth_t *self, const char *domain, const char *location);
    
//  Configure GSSAPI authentication for a given domain. GSSAPI authentication
//  uses an underlying mechanism (usually Kerberos) to establish a secure
//  context and perform mutual authentication. To cover all domains, use "*".
CZMQ_EXPORT void
    zauth_configure_gssapi (zauth_t *self, char *domain);

//  Enable verbose tracing of commands and activity
CZMQ_EXPORT void
    zauth_set_verbose (zauth_t *self, bool verbose);
    
//  Selftest
CZMQ_EXPORT void
    zauth_v2_test (bool verbose);
//  @end

#ifdef __cplusplus
}
#endif

#endif
