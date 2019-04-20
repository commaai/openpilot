/*  =========================================================================
    zthread - working with system threads (deprecated)

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZTHREAD_H_INCLUDED__
#define __ZTHREAD_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
//  Detached threads follow POSIX pthreads API
typedef void *(zthread_detached_fn) (void *args);

//  Attached threads get context and pipe from parent
typedef void (zthread_attached_fn) (void *args, zctx_t *ctx, void *pipe);

//  Create a detached thread. A detached thread operates autonomously
//  and is used to simulate a separate process. It gets no ctx, and no
//  pipe.
CZMQ_EXPORT int
    zthread_new (zthread_detached_fn *thread_fn, void *args);

//  Create an attached thread. An attached thread gets a ctx and a PAIR
//  pipe back to its parent. It must monitor its pipe, and exit if the
//  pipe becomes unreadable. Do not destroy the ctx, the thread does this
//  automatically when it ends.
CZMQ_EXPORT void *
    zthread_fork (zctx_t *ctx, zthread_attached_fn *thread_fn, void *args);

//  Self test of this class
CZMQ_EXPORT void
    zthread_test (bool verbose);
//  @end

#ifdef __cplusplus
}
#endif

#endif
