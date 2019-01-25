/*  =========================================================================
    zmutex - working with mutexes

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __ZMUTEX_H_INCLUDED__
#define __ZMUTEX_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif

//  @interface
//  This is a deprecated class, and will be removed over time. It is
//  provided in stable builds to support old applications. You should
//  stop using this class, and migrate any code that is still using it.

//  Create a new mutex container
CZMQ_EXPORT zmutex_t *
    zmutex_new (void);

//  Destroy a mutex container
CZMQ_EXPORT void
    zmutex_destroy (zmutex_t **self_p);

//  Lock mutex
CZMQ_EXPORT void
    zmutex_lock (zmutex_t *self);

//  Unlock mutex
CZMQ_EXPORT void
    zmutex_unlock (zmutex_t *self);

//  Try to lock mutex
CZMQ_EXPORT int
    zmutex_try_lock (zmutex_t *self);

//  Self test of this class.
CZMQ_EXPORT void
    zmutex_test (bool verbose);
//  @end

#ifdef __cplusplus
}
#endif

#endif
