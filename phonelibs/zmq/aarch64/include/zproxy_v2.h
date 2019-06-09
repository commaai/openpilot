/*  =========================================================================
    zproxy_v2 - run a steerable proxy in the background (deprecated)

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZPROXY_V2_H_INCLUDED__
#define __ZPROXY_V2_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface

//  Constructor
//  Create a new zproxy object. You must create the frontend and backend
//  sockets, configure them, and connect or bind them, before you pass them
//  to the constructor. Do NOT use the sockets again, after passing them to
//  this method.
CZMQ_EXPORT zproxy_t *
    zproxy_new (zctx_t *ctx, void *frontend, void *backend);

//  Destructor
//  Destroy a zproxy object; note this first stops the proxy.
CZMQ_EXPORT void
    zproxy_destroy (zproxy_t **self_p);

//  Copy all proxied messages to specified endpoint; if this is NULL, any
//  in-progress capturing will be stopped. You must already have bound the
//  endpoint to a PULL socket.
CZMQ_EXPORT void
    zproxy_capture (zproxy_t *self, const char *endpoint);

//  Pauses a zproxy object; a paused proxy will cease processing messages,
//  causing them to be queued up and potentially hit the high-water mark on
//  the frontend socket, causing messages to be dropped, or writing
//  applications to block.
CZMQ_EXPORT void
    zproxy_pause (zproxy_t *self);

//  Resume a zproxy object
CZMQ_EXPORT void
    zproxy_resume (zproxy_t *self);

// Self test of this class
CZMQ_EXPORT void
    zproxy_v2_test (bool verbose);
//  @end

#ifdef __cplusplus
}
#endif

#endif
