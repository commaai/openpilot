/*
 * Copyright (C) 2009 The Android Open Source Project
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

#ifndef _ANDROID_CUTILS_LOGD_H
#define _ANDROID_CUTILS_LOGD_H

/* the stable/frozen log-related definitions have been
 * moved to this header, which is exposed by the NDK
 */
#include <android/log.h>

/* the rest is only used internally by the system */
#if !defined(_WIN32)
#include <pthread.h>
#endif
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <log/uio.h>

#ifdef __cplusplus
extern "C" {
#endif

int __android_log_bwrite(int32_t tag, const void *payload, size_t len);
int __android_log_btwrite(int32_t tag, char type, const void *payload,
    size_t len);
int __android_log_bswrite(int32_t tag, const char *payload);

#ifdef __cplusplus
}
#endif

#endif /* _LOGD_H */
