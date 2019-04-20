/*  =========================================================================
    zmonitor_v2 - socket event monitor (deprecated)

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZMONITOR_V2_H_INCLUDED__
#define __ZMONITOR_V2_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
//  This code needs backporting to work with ZMQ v3.2
#if (ZMQ_VERSION_MAJOR == 4)

//  Create a new socket monitor
CZMQ_EXPORT zmonitor_t *
    zmonitor_new (zctx_t *ctx, void *socket, int events);

//  Destroy a socket monitor
CZMQ_EXPORT void
    zmonitor_destroy (zmonitor_t **self_p);

//  Receive a status message from the monitor; if no message arrives within
//  500 msec, or the call was interrupted, returns NULL.
CZMQ_EXPORT zmsg_t *
    zmonitor_recv (zmonitor_t *self);

//  Get the ZeroMQ socket, for polling 
CZMQ_EXPORT void *
    zmonitor_socket (zmonitor_t *self);

//  Enable verbose tracing of commands and activity
CZMQ_EXPORT void
    zmonitor_set_verbose (zmonitor_t *self, bool verbose);
#endif          //  ZeroMQ 4.0 or later

// Self test of this class
CZMQ_EXPORT void
    zmonitor_v2_test (bool verbose);
// @end

#ifdef __cplusplus
}
#endif

#endif
