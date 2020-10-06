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

#ifndef _LIBS_UTILS_THREAD_DEFS_H
#define _LIBS_UTILS_THREAD_DEFS_H

#include <stdint.h>
#include <sys/types.h>
#include <system/graphics.h>
#include <system/thread_defs.h>

// ---------------------------------------------------------------------------
// C API

#ifdef __cplusplus
extern "C" {
#endif

typedef void* android_thread_id_t;

typedef int (*android_thread_func_t)(void*);

#ifdef __cplusplus
} // extern "C"
#endif

// ---------------------------------------------------------------------------
// C++ API
#ifdef __cplusplus
namespace android {
// ---------------------------------------------------------------------------

typedef android_thread_id_t thread_id_t;
typedef android_thread_func_t thread_func_t;

enum {
    PRIORITY_LOWEST         = ANDROID_PRIORITY_LOWEST,
    PRIORITY_BACKGROUND     = ANDROID_PRIORITY_BACKGROUND,
    PRIORITY_NORMAL         = ANDROID_PRIORITY_NORMAL,
    PRIORITY_FOREGROUND     = ANDROID_PRIORITY_FOREGROUND,
    PRIORITY_DISPLAY        = ANDROID_PRIORITY_DISPLAY,
    PRIORITY_URGENT_DISPLAY = ANDROID_PRIORITY_URGENT_DISPLAY,
    PRIORITY_AUDIO          = ANDROID_PRIORITY_AUDIO,
    PRIORITY_URGENT_AUDIO   = ANDROID_PRIORITY_URGENT_AUDIO,
    PRIORITY_HIGHEST        = ANDROID_PRIORITY_HIGHEST,
    PRIORITY_DEFAULT        = ANDROID_PRIORITY_DEFAULT,
    PRIORITY_MORE_FAVORABLE = ANDROID_PRIORITY_MORE_FAVORABLE,
    PRIORITY_LESS_FAVORABLE = ANDROID_PRIORITY_LESS_FAVORABLE,
};

// ---------------------------------------------------------------------------
}; // namespace android
#endif  // __cplusplus
// ---------------------------------------------------------------------------


#endif // _LIBS_UTILS_THREAD_DEFS_H
