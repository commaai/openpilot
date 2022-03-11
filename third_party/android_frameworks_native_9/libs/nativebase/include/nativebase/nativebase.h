/*
 * Copyright (C) 2017 The Android Open Source Project
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

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <sys/cdefs.h>
#include <system/graphics-base.h>
#include <cutils/native_handle.h>

__BEGIN_DECLS

#ifdef __cplusplus
#define ANDROID_NATIVE_UNSIGNED_CAST(x) static_cast<unsigned int>(x)
#else
#define ANDROID_NATIVE_UNSIGNED_CAST(x) ((unsigned int)(x))
#endif

#define ANDROID_NATIVE_MAKE_CONSTANT(a,b,c,d)  \
    ((ANDROID_NATIVE_UNSIGNED_CAST(a) << 24) | \
     (ANDROID_NATIVE_UNSIGNED_CAST(b) << 16) | \
     (ANDROID_NATIVE_UNSIGNED_CAST(c) <<  8) | \
     (ANDROID_NATIVE_UNSIGNED_CAST(d)))

#define ANDROID_NATIVE_BUFFER_MAGIC     ANDROID_NATIVE_MAKE_CONSTANT('_','b','f','r')


typedef struct android_native_base_t
{
    /* a magic value defined by the actual EGL native type */
    int magic;

    /* the sizeof() of the actual EGL native type */
    int version;

    void* reserved[4];

    /* reference-counting interface */
    void (*incRef)(struct android_native_base_t* base);
    void (*decRef)(struct android_native_base_t* base);
} android_native_base_t;

typedef struct android_native_rect_t
{
    int32_t left;
    int32_t top;
    int32_t right;
    int32_t bottom;
} android_native_rect_t;

typedef struct ANativeWindowBuffer
{
#ifdef __cplusplus
    ANativeWindowBuffer() {
        common.magic = ANDROID_NATIVE_BUFFER_MAGIC;
        common.version = sizeof(ANativeWindowBuffer);
        memset(common.reserved, 0, sizeof(common.reserved));
    }

    // Implement the methods that sp<ANativeWindowBuffer> expects so that it
    // can be used to automatically refcount ANativeWindowBuffer's.
    void incStrong(const void* /*id*/) const {
        common.incRef(const_cast<android_native_base_t*>(&common));
    }
    void decStrong(const void* /*id*/) const {
        common.decRef(const_cast<android_native_base_t*>(&common));
    }
#endif

    struct android_native_base_t common;

    int width;
    int height;
    int stride;
    int format;
    int usage_deprecated;
    uintptr_t layerCount;

    void* reserved[1];

    const native_handle_t* handle;
    uint64_t usage;

    // we needed extra space for storing the 64-bits usage flags
    // the number of slots to use from reserved_proc depends on the
    // architecture.
    void* reserved_proc[8 - (sizeof(uint64_t) / sizeof(void*))];
} ANativeWindowBuffer_t;

typedef struct ANativeWindowBuffer ANativeWindowBuffer;

// Old typedef for backwards compatibility.
typedef ANativeWindowBuffer_t android_native_buffer_t;

__END_DECLS
