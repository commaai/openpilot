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

#ifndef _LIBS_LOG_LOG_ID_H
#define _LIBS_LOG_LOG_ID_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef log_id_t_defined
#define log_id_t_defined
typedef enum log_id {
  LOG_ID_MIN = 0,

  LOG_ID_MAIN = 0,
  LOG_ID_RADIO = 1,
  LOG_ID_EVENTS = 2,
  LOG_ID_SYSTEM = 3,
  LOG_ID_CRASH = 4,
  LOG_ID_STATS = 5,
  LOG_ID_SECURITY = 6,
  LOG_ID_KERNEL = 7, /* place last, third-parties can not use it */

  LOG_ID_MAX
} log_id_t;
#endif
#define sizeof_log_id_t sizeof(typeof_log_id_t)
#define typeof_log_id_t unsigned char

/*
 * Send a simple string to the log.
 */
int __android_log_buf_write(int bufID, int prio, const char* tag,
                            const char* text);
int __android_log_buf_print(int bufID, int prio, const char* tag,
                            const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((__format__(printf, 4, 5)))
#endif
    ;

/*
 * log_id_t helpers
 */
log_id_t android_name_to_log_id(const char* logName);
const char* android_log_id_to_name(log_id_t log_id);

#ifdef __cplusplus
}
#endif

#endif /* _LIBS_LOG_LOG_ID_H */
