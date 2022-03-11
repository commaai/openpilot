/*
 * Copyright (C) 2005-2014 The Android Open Source Project
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

#ifndef _LIBS_LOG_LOG_H
#define _LIBS_LOG_LOG_H

/* Too many in the ecosystem assume these are included */
#if !defined(_WIN32)
#include <pthread.h>
#endif
#include <stdint.h> /* uint16_t, int32_t */
#include <stdio.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <android/log.h>
#include <log/log_id.h>
#include <log/log_main.h>
#include <log/log_radio.h>
#include <log/log_read.h>
#include <log/log_safetynet.h>
#include <log/log_system.h>
#include <log/log_time.h>
#include <log/uio.h> /* helper to define iovec for portability */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * LOG_TAG is the local tag used for the following simplified
 * logging macros.  You can change this preprocessor definition
 * before using the other macros to change the tag.
 */

#ifndef LOG_TAG
#define LOG_TAG NULL
#endif

/*
 * Normally we strip the effects of ALOGV (VERBOSE messages),
 * LOG_FATAL and LOG_FATAL_IF (FATAL assert messages) from the
 * release builds be defining NDEBUG.  You can modify this (for
 * example with "#define LOG_NDEBUG 0" at the top of your source
 * file) to change that behavior.
 */

#ifndef LOG_NDEBUG
#ifdef NDEBUG
#define LOG_NDEBUG 1
#else
#define LOG_NDEBUG 0
#endif
#endif

/* --------------------------------------------------------------------- */

/*
 * This file uses ", ## __VA_ARGS__" zero-argument token pasting to
 * work around issues with debug-only syntax errors in assertions
 * that are missing format strings.  See commit
 * 19299904343daf191267564fe32e6cd5c165cd42
 */
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

/* --------------------------------------------------------------------- */

/*
 * Event logging.
 */

/*
 * The following should not be used directly.
 */

int __android_log_bwrite(int32_t tag, const void* payload, size_t len);
int __android_log_btwrite(int32_t tag, char type, const void* payload,
                          size_t len);
int __android_log_bswrite(int32_t tag, const char* payload);

int __android_log_stats_bwrite(int32_t tag, const void* payload, size_t len);

#define android_bWriteLog(tag, payload, len) \
  __android_log_bwrite(tag, payload, len)
#define android_btWriteLog(tag, type, payload, len) \
  __android_log_btwrite(tag, type, payload, len)

/*
 * Event log entry types.
 */
#ifndef __AndroidEventLogType_defined
#define __AndroidEventLogType_defined
typedef enum {
  /* Special markers for android_log_list_element type */
  EVENT_TYPE_LIST_STOP = '\n', /* declare end of list  */
  EVENT_TYPE_UNKNOWN = '?',    /* protocol error       */

  /* must match with declaration in java/android/android/util/EventLog.java */
  EVENT_TYPE_INT = 0,  /* int32_t */
  EVENT_TYPE_LONG = 1, /* int64_t */
  EVENT_TYPE_STRING = 2,
  EVENT_TYPE_LIST = 3,
  EVENT_TYPE_FLOAT = 4,
} AndroidEventLogType;
#endif
#define sizeof_AndroidEventLogType sizeof(typeof_AndroidEventLogType)
#define typeof_AndroidEventLogType unsigned char

#ifndef LOG_EVENT_INT
#define LOG_EVENT_INT(_tag, _value)                                          \
  {                                                                          \
    int intBuf = _value;                                                     \
    (void)android_btWriteLog(_tag, EVENT_TYPE_INT, &intBuf, sizeof(intBuf)); \
  }
#endif
#ifndef LOG_EVENT_LONG
#define LOG_EVENT_LONG(_tag, _value)                                            \
  {                                                                             \
    long long longBuf = _value;                                                 \
    (void)android_btWriteLog(_tag, EVENT_TYPE_LONG, &longBuf, sizeof(longBuf)); \
  }
#endif
#ifndef LOG_EVENT_FLOAT
#define LOG_EVENT_FLOAT(_tag, _value)                           \
  {                                                             \
    float floatBuf = _value;                                    \
    (void)android_btWriteLog(_tag, EVENT_TYPE_FLOAT, &floatBuf, \
                             sizeof(floatBuf));                 \
  }
#endif
#ifndef LOG_EVENT_STRING
#define LOG_EVENT_STRING(_tag, _value) \
  (void)__android_log_bswrite(_tag, _value);
#endif

#ifdef __linux__

#ifndef __ANDROID_USE_LIBLOG_CLOCK_INTERFACE
#ifndef __ANDROID_API__
#define __ANDROID_USE_LIBLOG_CLOCK_INTERFACE 1
#elif __ANDROID_API__ > 22 /* > Lollipop */
#define __ANDROID_USE_LIBLOG_CLOCK_INTERFACE 1
#else
#define __ANDROID_USE_LIBLOG_CLOCK_INTERFACE 0
#endif
#endif

#if __ANDROID_USE_LIBLOG_CLOCK_INTERFACE
clockid_t android_log_clockid(void);
#endif

#endif /* __linux__ */

/* --------------------------------------------------------------------- */

#ifndef __ANDROID_USE_LIBLOG_CLOSE_INTERFACE
#ifndef __ANDROID_API__
#define __ANDROID_USE_LIBLOG_CLOSE_INTERFACE 1
#elif __ANDROID_API__ > 18 /* > JellyBean */
#define __ANDROID_USE_LIBLOG_CLOSE_INTERFACE 1
#else
#define __ANDROID_USE_LIBLOG_CLOSE_INTERFACE 0
#endif
#endif

#if __ANDROID_USE_LIBLOG_CLOSE_INTERFACE
/*
 * Release any logger resources (a new log write will immediately re-acquire)
 *
 * May be used to clean up File descriptors after a Fork, the resources are
 * all O_CLOEXEC so wil self clean on exec().
 */
void __android_log_close(void);
#endif

#ifndef __ANDROID_USE_LIBLOG_RATELIMIT_INTERFACE
#ifndef __ANDROID_API__
#define __ANDROID_USE_LIBLOG_RATELIMIT_INTERFACE 1
#elif __ANDROID_API__ > 25 /* > OC */
#define __ANDROID_USE_LIBLOG_RATELIMIT_INTERFACE 1
#else
#define __ANDROID_USE_LIBLOG_RATELIMIT_INTERFACE 0
#endif
#endif

#if __ANDROID_USE_LIBLOG_RATELIMIT_INTERFACE

/*
 * if last is NULL, caller _must_ provide a consistent value for seconds.
 *
 * Return -1 if we can not acquire a lock, which below will permit the logging,
 * error on allowing a log message through.
 */
int __android_log_ratelimit(time_t seconds, time_t* last);

/*
 * Usage:
 *
 *   // Global default and state
 *   IF_ALOG_RATELIMIT() {
 *      ALOG*(...);
 *   }
 *
 *   // local state, 10 seconds ratelimit
 *   static time_t local_state;
 *   IF_ALOG_RATELIMIT_LOCAL(10, &local_state) {
 *     ALOG*(...);
 *   }
 */

#define IF_ALOG_RATELIMIT() if (__android_log_ratelimit(0, NULL) > 0)
#define IF_ALOG_RATELIMIT_LOCAL(seconds, state) \
  if (__android_log_ratelimit(seconds, state) > 0)

#else

/* No ratelimiting as API unsupported */
#define IF_ALOG_RATELIMIT() if (1)
#define IF_ALOG_RATELIMIT_LOCAL(...) if (1)

#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#ifdef __cplusplus
}
#endif

#endif /* _LIBS_LOG_LOG_H */
