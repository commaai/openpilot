/*  =========================================================================
    zctx - working with 0MQ contexts

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZCTX_H_INCLUDED__
#define __ZCTX_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif


//  @interface
//  Create new context, returns context object, replaces zmq_init
CZMQ_EXPORT zctx_t *
    zctx_new (void);

//  Destroy context and all sockets in it, replaces zmq_term
CZMQ_EXPORT void
    zctx_destroy (zctx_t **self_p);

//  Create new shadow context, returns context object
CZMQ_EXPORT zctx_t *
    zctx_shadow (zctx_t *self);
//  @end
	
//  Create a new context by shadowing a plain zmq context
CZMQ_EXPORT zctx_t *
zctx_shadow_zmq_ctx (void *zmqctx);

//  @interface
//  Raise default I/O threads from 1, for crazy heavy applications
//  The rule of thumb is one I/O thread per gigabyte of traffic in
//  or out. Call this method before creating any sockets on the context,
//  or calling zctx_shadow, or the setting will have no effect.
CZMQ_EXPORT void
    zctx_set_iothreads (zctx_t *self, int iothreads);

//  Set msecs to flush sockets when closing them, see the ZMQ_LINGER
//  man page section for more details. By default, set to zero, so
//  any in-transit messages are discarded when you destroy a socket or
//  a context.
CZMQ_EXPORT void
    zctx_set_linger (zctx_t *self, int linger);

//  Set initial high-water mark for inter-thread pipe sockets. Note that
//  this setting is separate from the default for normal sockets. You 
//  should change the default for pipe sockets *with care*. Too low values
//  will cause blocked threads, and an infinite setting can cause memory
//  exhaustion. The default, no matter the underlying ZeroMQ version, is
//  1,000.
CZMQ_EXPORT void
    zctx_set_pipehwm (zctx_t *self, int pipehwm);
    
//  Set initial send HWM for all new normal sockets created in context.
//  You can set this per-socket after the socket is created.
//  The default, no matter the underlying ZeroMQ version, is 1,000.
CZMQ_EXPORT void
    zctx_set_sndhwm (zctx_t *self, int sndhwm);
    
//  Set initial receive HWM for all new normal sockets created in context.
//  You can set this per-socket after the socket is created.
//  The default, no matter the underlying ZeroMQ version, is 1,000.
CZMQ_EXPORT void
    zctx_set_rcvhwm (zctx_t *self, int rcvhwm);

//  Return low-level 0MQ context object, will be NULL before first socket
//  is created. Use with care.
CZMQ_EXPORT void *
    zctx_underlying (zctx_t *self);

//  Self test of this class
CZMQ_EXPORT void
    zctx_test (bool verbose);
//  @end

//  Create socket within this context, for CZMQ use only
void *
    zctx__socket_new (zctx_t *self, int type);

//  Create pipe socket within this context, for CZMQ use only
void *
    zctx__socket_pipe (zctx_t *self);

//  Destroy socket within this context, for CZMQ use only
void
    zctx__socket_destroy (zctx_t *self, void *socket);
	
//  Initialize the low-level 0MQ context object, for CZMQ use only
void
	zctx__initialize_underlying(zctx_t *self);
//  @end
    
#ifdef __cplusplus
}
#endif

#endif
