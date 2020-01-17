/*  =========================================================================
    zsocket - working with 0MQ sockets

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZSOCKET_H_INCLUDED__
#define __ZSOCKET_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
//  This port range is defined by IANA for dynamic or private ports
//  We use this when choosing a port for dynamic binding.
#define ZSOCKET_DYNFROM     0xc000
#define ZSOCKET_DYNTO       0xffff

//  Callback function for zero-copy methods
typedef void (zsocket_free_fn) (void *data, void *arg);

//  Create a new socket within our CZMQ context, replaces zmq_socket.
//  Use this to get automatic management of the socket at shutdown.
//  Note: SUB sockets do not automatically subscribe to everything; you
//  must set filters explicitly.
CZMQ_EXPORT void *
    zsocket_new (zctx_t *self, int type);

//  Destroy a socket within our CZMQ context, replaces zmq_close.
CZMQ_EXPORT void
    zsocket_destroy (zctx_t *ctx, void *self);

//  Bind a socket to a formatted endpoint. If the port is specified as
//  '*', binds to any free port from ZSOCKET_DYNFROM to ZSOCKET_DYNTO
//  and returns the actual port number used. Otherwise asserts that the
//  bind succeeded with the specified port number. Always returns the
//  port number if successful.
CZMQ_EXPORT int
    zsocket_bind (void *self, const char *format, ...);

//  Unbind a socket from a formatted endpoint.
//  Returns 0 if OK, -1 if the endpoint was invalid or the function
//  isn't supported.
CZMQ_EXPORT int
    zsocket_unbind (void *self, const char *format, ...);

//  Connect a socket to a formatted endpoint
//  Returns 0 if OK, -1 if the endpoint was invalid.
CZMQ_EXPORT int
    zsocket_connect (void *self, const char *format, ...);

//  Disconnect a socket from a formatted endpoint
//  Returns 0 if OK, -1 if the endpoint was invalid or the function
//  isn't supported.
CZMQ_EXPORT int
    zsocket_disconnect (void *self, const char *format, ...);

//  Poll for input events on the socket. Returns TRUE if there is input
//  ready on the socket, else FALSE.
CZMQ_EXPORT bool
    zsocket_poll (void *self, int msecs);

//  Returns socket type as printable constant string
CZMQ_EXPORT const char *
    zsocket_type_str (void *self);

//  Send data over a socket as a single message frame.
//  Accepts these flags: ZFRAME_MORE and ZFRAME_DONTWAIT.
//  Returns -1 on error, 0 on success
CZMQ_EXPORT int
    zsocket_sendmem (void *self, const void *data, size_t size, int flags);

//  Send a signal over a socket. A signal is a zero-byte message.
//  Signals are used primarily between threads, over pipe sockets.
//  Returns -1 if there was an error sending the signal.
CZMQ_EXPORT int
    zsocket_signal (void *self);

//  Wait on a signal. Use this to coordinate between threads, over
//  pipe pairs. Returns -1 on error, 0 on success.
CZMQ_EXPORT int
    zsocket_wait (void *self);

//  Self test of this class
CZMQ_EXPORT void
    zsocket_test (bool verbose);
//  @end

//  Compiler hints
CZMQ_EXPORT int zsocket_bind (void *self, const char *format, ...) CHECK_PRINTF (2);
CZMQ_EXPORT int zsocket_unbind (void *self, const char *format, ...) CHECK_PRINTF (2);
CZMQ_EXPORT int zsocket_connect (void *self, const char *format, ...) CHECK_PRINTF (2);
CZMQ_EXPORT int zsocket_disconnect (void *self, const char *format, ...) CHECK_PRINTF (2);

//  Emulation of widely-used 2.x socket options
CZMQ_EXPORT void zsocket_set_hwm (void *self, int hwm);

#ifdef __cplusplus
}
#endif

#endif
