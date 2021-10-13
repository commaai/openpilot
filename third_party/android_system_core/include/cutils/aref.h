/*
 * Copyright (C) 2013 The Android Open Source Project
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

#ifndef _CUTILS_AREF_H_
#define _CUTILS_AREF_H_

#include <stddef.h>
#include <sys/cdefs.h>

#include <cutils/atomic.h>

__BEGIN_DECLS

#define AREF_TO_ITEM(aref, container, member) \
    (container *) (((char*) (aref)) - offsetof(container, member))

struct aref
{
    volatile int32_t count;
};

static inline void aref_init(struct aref *r)
{
    r->count = 1;
}

static inline int32_t aref_count(struct aref *r)
{
    return r->count;
}

static inline void aref_get(struct aref *r)
{
    android_atomic_inc(&r->count);
}

static inline void aref_put(struct aref *r, void (*release)(struct aref *))
{
    if (android_atomic_dec(&r->count) == 1)
        release(r);
}

__END_DECLS

#endif // _CUTILS_AREF_H_
