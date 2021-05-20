/*
 * Copyright (C) 2007 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _LIBS_CUTILS_THREADS_H
#define _LIBS_CUTILS_THREADS_H

#include  <sys/types.h>

#if !defined(_WIN32)
#include <pthread.h>
#else
#include <windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/***********************************************************************/
/***********************************************************************/
/*****                                                             *****/
/*****         local thread storage                                *****/
/*****                                                             *****/
/***********************************************************************/
/***********************************************************************/

extern pid_t gettid();

#if !defined(_WIN32)

typedef struct {
    pthread_mutex_t   lock;
    int               has_tls;
    pthread_key_t     tls;
} thread_store_t;

#define  THREAD_STORE_INITIALIZER  { PTHREAD_MUTEX_INITIALIZER, 0, 0 }

#else // !defined(_WIN32)

typedef struct {
    int               lock_init;
    int               has_tls;
    DWORD             tls;
    CRITICAL_SECTION  lock;
} thread_store_t;

#define  THREAD_STORE_INITIALIZER  { 0, 0, 0, {0, 0, 0, 0, 0, 0} }

#endif // !defined(_WIN32)

typedef void  (*thread_store_destruct_t)(void*  value);

extern void*  thread_store_get(thread_store_t*  store);

extern void   thread_store_set(thread_store_t*          store,
                               void*                    value,
                               thread_store_destruct_t  destroy);

/***********************************************************************/
/***********************************************************************/
/*****                                                             *****/
/*****         mutexes                                             *****/
/*****                                                             *****/
/***********************************************************************/
/***********************************************************************/

#if !defined(_WIN32)

typedef pthread_mutex_t   mutex_t;

#define  MUTEX_INITIALIZER  PTHREAD_MUTEX_INITIALIZER

static __inline__ void  mutex_lock(mutex_t*  lock)
{
    pthread_mutex_lock(lock);
}
static __inline__ void  mutex_unlock(mutex_t*  lock)
{
    pthread_mutex_unlock(lock);
}
static __inline__ int  mutex_init(mutex_t*  lock)
{
    return pthread_mutex_init(lock, NULL);
}
static __inline__ void mutex_destroy(mutex_t*  lock)
{
    pthread_mutex_destroy(lock);
}

#else // !defined(_WIN32)

typedef struct {
    int                init;
    CRITICAL_SECTION   lock[1];
} mutex_t;

#define  MUTEX_INITIALIZER  { 0, {{ NULL, 0, 0, NULL, NULL, 0 }} }

static __inline__ void  mutex_lock(mutex_t*  lock)
{
    if (!lock->init) {
        lock->init = 1;
        InitializeCriticalSection( lock->lock );
        lock->init = 2;
    } else while (lock->init != 2)
        Sleep(10);

    EnterCriticalSection(lock->lock);
}

static __inline__ void  mutex_unlock(mutex_t*  lock)
{
    LeaveCriticalSection(lock->lock);
}
static __inline__ int  mutex_init(mutex_t*  lock)
{
    InitializeCriticalSection(lock->lock);
    lock->init = 2;
    return 0;
}
static __inline__ void  mutex_destroy(mutex_t*  lock)
{
    if (lock->init) {
        lock->init = 0;
        DeleteCriticalSection(lock->lock);
    }
}
#endif // !defined(_WIN32)

#ifdef __cplusplus
}
#endif

#endif /* _LIBS_CUTILS_THREADS_H */
