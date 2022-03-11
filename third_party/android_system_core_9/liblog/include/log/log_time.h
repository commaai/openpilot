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

#ifndef _LIBS_LOG_LOG_TIME_H
#define _LIBS_LOG_LOG_TIME_H

#include <stdint.h>
#include <time.h>

/* struct log_time is a wire-format variant of struct timespec */
#define NS_PER_SEC 1000000000ULL
#define US_PER_SEC 1000000ULL
#define MS_PER_SEC 1000ULL

#ifndef __struct_log_time_defined
#define __struct_log_time_defined

#define LOG_TIME_SEC(t) ((t)->tv_sec)
/* next power of two after NS_PER_SEC */
#define LOG_TIME_NSEC(t) ((t)->tv_nsec & (UINT32_MAX >> 2))

#ifdef __cplusplus

/*
 * NB: we did NOT define a copy constructor. This will result in structure
 * no longer being compatible with pass-by-value which is desired
 * efficient behavior. Also, pass-by-reference breaks C/C++ ABI.
 */
struct log_time {
 public:
  uint32_t tv_sec; /* good to Feb 5 2106 */
  uint32_t tv_nsec;

  static const uint32_t tv_sec_max = 0xFFFFFFFFUL;
  static const uint32_t tv_nsec_max = 999999999UL;

  log_time(const timespec& T)
      : tv_sec(static_cast<uint32_t>(T.tv_sec)),
        tv_nsec(static_cast<uint32_t>(T.tv_nsec)) {
  }
  explicit log_time(uint32_t sec, uint32_t nsec = 0)
      : tv_sec(sec), tv_nsec(nsec) {
  }
#ifdef _SYSTEM_CORE_INCLUDE_PRIVATE_ANDROID_LOGGER_H_
#define __struct_log_time_private_defined
  static const timespec EPOCH;
#endif
  log_time() {
  }
#ifdef __linux__
  explicit log_time(clockid_t id) {
    timespec T;
    clock_gettime(id, &T);
    tv_sec = static_cast<uint32_t>(T.tv_sec);
    tv_nsec = static_cast<uint32_t>(T.tv_nsec);
  }
#endif
  explicit log_time(const char* T) {
    const uint8_t* c = reinterpret_cast<const uint8_t*>(T);
    tv_sec = c[0] | (static_cast<uint32_t>(c[1]) << 8) |
             (static_cast<uint32_t>(c[2]) << 16) |
             (static_cast<uint32_t>(c[3]) << 24);
    tv_nsec = c[4] | (static_cast<uint32_t>(c[5]) << 8) |
              (static_cast<uint32_t>(c[6]) << 16) |
              (static_cast<uint32_t>(c[7]) << 24);
  }

  /* timespec */
  bool operator==(const timespec& T) const {
    return (tv_sec == static_cast<uint32_t>(T.tv_sec)) &&
           (tv_nsec == static_cast<uint32_t>(T.tv_nsec));
  }
  bool operator!=(const timespec& T) const {
    return !(*this == T);
  }
  bool operator<(const timespec& T) const {
    return (tv_sec < static_cast<uint32_t>(T.tv_sec)) ||
           ((tv_sec == static_cast<uint32_t>(T.tv_sec)) &&
            (tv_nsec < static_cast<uint32_t>(T.tv_nsec)));
  }
  bool operator>=(const timespec& T) const {
    return !(*this < T);
  }
  bool operator>(const timespec& T) const {
    return (tv_sec > static_cast<uint32_t>(T.tv_sec)) ||
           ((tv_sec == static_cast<uint32_t>(T.tv_sec)) &&
            (tv_nsec > static_cast<uint32_t>(T.tv_nsec)));
  }
  bool operator<=(const timespec& T) const {
    return !(*this > T);
  }

#ifdef _SYSTEM_CORE_INCLUDE_PRIVATE_ANDROID_LOGGER_H_
  log_time operator-=(const timespec& T);
  log_time operator-(const timespec& T) const {
    log_time local(*this);
    return local -= T;
  }
  log_time operator+=(const timespec& T);
  log_time operator+(const timespec& T) const {
    log_time local(*this);
    return local += T;
  }
#endif

  /* log_time */
  bool operator==(const log_time& T) const {
    return (tv_sec == T.tv_sec) && (tv_nsec == T.tv_nsec);
  }
  bool operator!=(const log_time& T) const {
    return !(*this == T);
  }
  bool operator<(const log_time& T) const {
    return (tv_sec < T.tv_sec) ||
           ((tv_sec == T.tv_sec) && (tv_nsec < T.tv_nsec));
  }
  bool operator>=(const log_time& T) const {
    return !(*this < T);
  }
  bool operator>(const log_time& T) const {
    return (tv_sec > T.tv_sec) ||
           ((tv_sec == T.tv_sec) && (tv_nsec > T.tv_nsec));
  }
  bool operator<=(const log_time& T) const {
    return !(*this > T);
  }

#ifdef _SYSTEM_CORE_INCLUDE_PRIVATE_ANDROID_LOGGER_H_
  log_time operator-=(const log_time& T);
  log_time operator-(const log_time& T) const {
    log_time local(*this);
    return local -= T;
  }
  log_time operator+=(const log_time& T);
  log_time operator+(const log_time& T) const {
    log_time local(*this);
    return local += T;
  }
#endif

  uint64_t nsec() const {
    return static_cast<uint64_t>(tv_sec) * NS_PER_SEC + tv_nsec;
  }
  uint64_t usec() const {
    return static_cast<uint64_t>(tv_sec) * US_PER_SEC +
           tv_nsec / (NS_PER_SEC / US_PER_SEC);
  }
  uint64_t msec() const {
    return static_cast<uint64_t>(tv_sec) * MS_PER_SEC +
           tv_nsec / (NS_PER_SEC / MS_PER_SEC);
  }

#ifdef _SYSTEM_CORE_INCLUDE_PRIVATE_ANDROID_LOGGER_H_
  static const char default_format[];

  /* Add %#q for the fraction of a second to the standard library functions */
  char* strptime(const char* s, const char* format = default_format);
#endif
} __attribute__((__packed__));

#else /* __cplusplus */

typedef struct log_time {
  uint32_t tv_sec;
  uint32_t tv_nsec;
} __attribute__((__packed__)) log_time;

#endif /* __cplusplus */

#endif /* __struct_log_time_defined */

#endif /* _LIBS_LOG_LOG_TIME_H */
