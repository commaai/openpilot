/*  =========================================================================
    zbeacon - LAN discovery and presence (deprecated)
    
    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZBEACON_V2_H_INCLUDED__
#define __ZBEACON_V2_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
//  Create a new beacon on a certain UDP port. If the system does not
//  support UDP broadcasts (lacking a useful interface), returns NULL.
//  To force the beacon to operate on a given port, set the environment
//  variable ZSYS_INTERFACE, or call zsys_set_interface() beforehand.
//  If you are using the new zsock API then pass NULL as the ctx here.
CZMQ_EXPORT zbeacon_t *
    zbeacon_new (zctx_t *ctx, int port_nbr);
    
//  Destroy a beacon
CZMQ_EXPORT void
    zbeacon_destroy (zbeacon_t **self_p);

//  Return our own IP address as printable string
CZMQ_EXPORT char *
    zbeacon_hostname (zbeacon_t *self);

//  Set broadcast interval in milliseconds (default is 1000 msec)
CZMQ_EXPORT void
    zbeacon_set_interval (zbeacon_t *self, int interval);

//  Filter out any beacon that looks exactly like ours
CZMQ_EXPORT void
    zbeacon_noecho (zbeacon_t *self);

//  Start broadcasting beacon to peers at the specified interval
CZMQ_EXPORT void
    zbeacon_publish (zbeacon_t *self, byte *transmit, size_t size);
    
//  Stop broadcasting beacons
CZMQ_EXPORT void
    zbeacon_silence (zbeacon_t *self);

//  Start listening to other peers; zero-sized filter means get everything
CZMQ_EXPORT void
    zbeacon_subscribe (zbeacon_t *self, byte *filter, size_t size);

//  Stop listening to other peers
CZMQ_EXPORT void
    zbeacon_unsubscribe (zbeacon_t *self);

//  Get beacon ZeroMQ socket, for polling or receiving messages
CZMQ_EXPORT void *
    zbeacon_socket (zbeacon_t *self);

//  Self test of this class
CZMQ_EXPORT void
    zbeacon_v2_test (bool verbose);
//  @end

#ifdef __cplusplus
}
#endif

#endif
