/*  =========================================================================
    zmonitor - socket event monitor

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZMONITOR_H_INCLUDED__
#define __ZMONITOR_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
//  Create new zmonitor actor instance to monitor a zsock_t socket:
//
//      zactor_t *monitor = zactor_new (zmonitor, mysocket);
//
//  Destroy zmonitor instance.
//
//      zactor_destroy (&monitor);
//
//  Enable verbose logging of commands and activity.
//
//      zstr_send (monitor, "VERBOSE");
//
//  Listen to monitor event type (zero or types, ending in NULL):
//      zstr_sendx (monitor, "LISTEN", type, ..., NULL);
//  
//      Events:
//      CONNECTED
//      CONNECT_DELAYED
//      CONNECT_RETRIED
//      LISTENING
//      BIND_FAILED
//      ACCEPTED
//      ACCEPT_FAILED
//      CLOSED
//      CLOSE_FAILED
//      DISCONNECTED
//      MONITOR_STOPPED
//      ALL
//
//  Start monitor; after this, any further LISTEN commands are ignored.
//
//      zstr_send (monitor, "START");
//      zsock_wait (monitor);
//
//  Receive next monitor event:
//
//      zmsg_t *msg = zmsg_recv (monitor);
//
//  This is the zmonitor constructor as a zactor_fn; the argument can be
//  a zactor_t, zsock_t, or libzmq void * socket:
CZMQ_EXPORT void
    zmonitor (zsock_t *pipe, void *sock);

//  Selftest
CZMQ_EXPORT void
    zmonitor_test (bool verbose);
//  @end
#ifdef __cplusplus
}
#endif

#endif
