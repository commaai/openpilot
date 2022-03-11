/*
 * Copyright (C) 2005-2017 The Android Open Source Project
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

#ifndef _LIBS_LOG_LOG_RADIO_H
#define _LIBS_LOG_LOG_RADIO_H

#include <android/log.h>
#include <log/log_id.h>

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

#ifndef __predict_false
#define __predict_false(exp) __builtin_expect((exp) != 0, 0)
#endif

/*
 * Simplified macro to send a verbose radio log message using current LOG_TAG.
 */
#ifndef RLOGV
#define __RLOGV(...)                                                         \
  ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_VERBOSE, LOG_TAG, \
                                 __VA_ARGS__))
#if LOG_NDEBUG
#define RLOGV(...)          \
  do {                      \
    if (0) {                \
      __RLOGV(__VA_ARGS__); \
    }                       \
  } while (0)
#else
#define RLOGV(...) __RLOGV(__VA_ARGS__)
#endif
#endif

#ifndef RLOGV_IF
#if LOG_NDEBUG
#define RLOGV_IF(cond, ...) ((void)0)
#else
#define RLOGV_IF(cond, ...)                                                \
  ((__predict_false(cond))                                                 \
       ? ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_VERBOSE, \
                                        LOG_TAG, __VA_ARGS__))             \
       : (void)0)
#endif
#endif

/*
 * Simplified macro to send a debug radio log message using  current LOG_TAG.
 */
#ifndef RLOGD
#define RLOGD(...)                                                         \
  ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_DEBUG, LOG_TAG, \
                                 __VA_ARGS__))
#endif

#ifndef RLOGD_IF
#define RLOGD_IF(cond, ...)                                              \
  ((__predict_false(cond))                                               \
       ? ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_DEBUG, \
                                        LOG_TAG, __VA_ARGS__))           \
       : (void)0)
#endif

/*
 * Simplified macro to send an info radio log message using  current LOG_TAG.
 */
#ifndef RLOGI
#define RLOGI(...)                                                        \
  ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_INFO, LOG_TAG, \
                                 __VA_ARGS__))
#endif

#ifndef RLOGI_IF
#define RLOGI_IF(cond, ...)                                             \
  ((__predict_false(cond))                                              \
       ? ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_INFO, \
                                        LOG_TAG, __VA_ARGS__))          \
       : (void)0)
#endif

/*
 * Simplified macro to send a warning radio log message using current LOG_TAG.
 */
#ifndef RLOGW
#define RLOGW(...)                                                        \
  ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_WARN, LOG_TAG, \
                                 __VA_ARGS__))
#endif

#ifndef RLOGW_IF
#define RLOGW_IF(cond, ...)                                             \
  ((__predict_false(cond))                                              \
       ? ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_WARN, \
                                        LOG_TAG, __VA_ARGS__))          \
       : (void)0)
#endif

/*
 * Simplified macro to send an error radio log message using current LOG_TAG.
 */
#ifndef RLOGE
#define RLOGE(...)                                                         \
  ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_ERROR, LOG_TAG, \
                                 __VA_ARGS__))
#endif

#ifndef RLOGE_IF
#define RLOGE_IF(cond, ...)                                              \
  ((__predict_false(cond))                                               \
       ? ((void)__android_log_buf_print(LOG_ID_RADIO, ANDROID_LOG_ERROR, \
                                        LOG_TAG, __VA_ARGS__))           \
       : (void)0)
#endif

#endif /* _LIBS_LOG_LOG_RADIO_H */
