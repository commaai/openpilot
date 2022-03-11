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

#ifndef _LIBS_LOG_LOG_READ_H
#define _LIBS_LOG_LOG_READ_H

/* deal with possible sys/cdefs.h conflict with fcntl.h */
#ifdef __unused
#define __unused_defined __unused
#undef __unused
#endif

#include <fcntl.h> /* Pick up O_* macros */

/* restore definitions from above */
#ifdef __unused_defined
#define __unused __attribute__((__unused__))
#endif

#include <stdint.h>

#include <log/log_id.h>
#include <log/log_time.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Native log reading interface section. See logcat for sample code.
 *
 * The preferred API is an exec of logcat. Likely uses of this interface
 * are if native code suffers from exec or filtration being too costly,
 * access to raw information, or parsing is an issue.
 */

/*
 * The userspace structure for version 1 of the logger_entry ABI.
 */
#ifndef __struct_logger_entry_defined
#define __struct_logger_entry_defined
struct logger_entry {
  uint16_t len;   /* length of the payload */
  uint16_t __pad; /* no matter what, we get 2 bytes of padding */
  int32_t pid;    /* generating process's pid */
  int32_t tid;    /* generating process's tid */
  int32_t sec;    /* seconds since Epoch */
  int32_t nsec;   /* nanoseconds */
#ifndef __cplusplus
  char msg[0]; /* the entry's payload */
#endif
};
#endif

/*
 * The userspace structure for version 2 of the logger_entry ABI.
 */
#ifndef __struct_logger_entry_v2_defined
#define __struct_logger_entry_v2_defined
struct logger_entry_v2 {
  uint16_t len;      /* length of the payload */
  uint16_t hdr_size; /* sizeof(struct logger_entry_v2) */
  int32_t pid;       /* generating process's pid */
  int32_t tid;       /* generating process's tid */
  int32_t sec;       /* seconds since Epoch */
  int32_t nsec;      /* nanoseconds */
  uint32_t euid;     /* effective UID of logger */
#ifndef __cplusplus
  char msg[0]; /* the entry's payload */
#endif
} __attribute__((__packed__));
#endif

/*
 * The userspace structure for version 3 of the logger_entry ABI.
 */
#ifndef __struct_logger_entry_v3_defined
#define __struct_logger_entry_v3_defined
struct logger_entry_v3 {
  uint16_t len;      /* length of the payload */
  uint16_t hdr_size; /* sizeof(struct logger_entry_v3) */
  int32_t pid;       /* generating process's pid */
  int32_t tid;       /* generating process's tid */
  int32_t sec;       /* seconds since Epoch */
  int32_t nsec;      /* nanoseconds */
  uint32_t lid;      /* log id of the payload */
#ifndef __cplusplus
  char msg[0]; /* the entry's payload */
#endif
} __attribute__((__packed__));
#endif

/*
 * The userspace structure for version 4 of the logger_entry ABI.
 */
#ifndef __struct_logger_entry_v4_defined
#define __struct_logger_entry_v4_defined
struct logger_entry_v4 {
  uint16_t len;      /* length of the payload */
  uint16_t hdr_size; /* sizeof(struct logger_entry_v4) */
  int32_t pid;       /* generating process's pid */
  uint32_t tid;      /* generating process's tid */
  uint32_t sec;      /* seconds since Epoch */
  uint32_t nsec;     /* nanoseconds */
  uint32_t lid;      /* log id of the payload, bottom 4 bits currently */
  uint32_t uid;      /* generating process's uid */
#ifndef __cplusplus
  char msg[0]; /* the entry's payload */
#endif
};
#endif

/*
 * The maximum size of the log entry payload that can be
 * written to the logger. An attempt to write more than
 * this amount will result in a truncated log entry.
 */
#define LOGGER_ENTRY_MAX_PAYLOAD 4068

/*
 * The maximum size of a log entry which can be read.
 * An attempt to read less than this amount may result
 * in read() returning EINVAL.
 */
#define LOGGER_ENTRY_MAX_LEN (5 * 1024)

#ifndef __struct_log_msg_defined
#define __struct_log_msg_defined
struct log_msg {
  union {
    unsigned char buf[LOGGER_ENTRY_MAX_LEN + 1];
    struct logger_entry_v4 entry;
    struct logger_entry_v4 entry_v4;
    struct logger_entry_v3 entry_v3;
    struct logger_entry_v2 entry_v2;
    struct logger_entry entry_v1;
  } __attribute__((aligned(4)));
#ifdef __cplusplus
  /* Matching log_time operators */
  bool operator==(const log_msg& T) const {
    return (entry.sec == T.entry.sec) && (entry.nsec == T.entry.nsec);
  }
  bool operator!=(const log_msg& T) const {
    return !(*this == T);
  }
  bool operator<(const log_msg& T) const {
    return (entry.sec < T.entry.sec) ||
           ((entry.sec == T.entry.sec) && (entry.nsec < T.entry.nsec));
  }
  bool operator>=(const log_msg& T) const {
    return !(*this < T);
  }
  bool operator>(const log_msg& T) const {
    return (entry.sec > T.entry.sec) ||
           ((entry.sec == T.entry.sec) && (entry.nsec > T.entry.nsec));
  }
  bool operator<=(const log_msg& T) const {
    return !(*this > T);
  }
  uint64_t nsec() const {
    return static_cast<uint64_t>(entry.sec) * NS_PER_SEC + entry.nsec;
  }

  /* packet methods */
  log_id_t id() {
    return static_cast<log_id_t>(entry.lid);
  }
  char* msg() {
    unsigned short hdr_size = entry.hdr_size;
    if (!hdr_size) {
      hdr_size = sizeof(entry_v1);
    }
    if ((hdr_size < sizeof(entry_v1)) || (hdr_size > sizeof(entry))) {
      return NULL;
    }
    return reinterpret_cast<char*>(buf) + hdr_size;
  }
  unsigned int len() {
    return (entry.hdr_size ? entry.hdr_size
                           : static_cast<uint16_t>(sizeof(entry_v1))) +
           entry.len;
  }
#endif
};
#endif

#ifndef __ANDROID_USE_LIBLOG_READER_INTERFACE
#ifndef __ANDROID_API__
#define __ANDROID_USE_LIBLOG_READER_INTERFACE 3
#elif __ANDROID_API__ > 23 /* > Marshmallow */
#define __ANDROID_USE_LIBLOG_READER_INTERFACE 3
#elif __ANDROID_API__ > 22 /* > Lollipop */
#define __ANDROID_USE_LIBLOG_READER_INTERFACE 2
#elif __ANDROID_API__ > 19 /* > KitKat */
#define __ANDROID_USE_LIBLOG_READER_INTERFACE 1
#else
#define __ANDROID_USE_LIBLOG_READER_INTERFACE 0
#endif
#endif

#if __ANDROID_USE_LIBLOG_READER_INTERFACE

struct logger;

log_id_t android_logger_get_id(struct logger* logger);

int android_logger_clear(struct logger* logger);
long android_logger_get_log_size(struct logger* logger);
int android_logger_set_log_size(struct logger* logger, unsigned long size);
long android_logger_get_log_readable_size(struct logger* logger);
int android_logger_get_log_version(struct logger* logger);

struct logger_list;

#if __ANDROID_USE_LIBLOG_READER_INTERFACE > 1
ssize_t android_logger_get_statistics(struct logger_list* logger_list,
                                      char* buf, size_t len);
ssize_t android_logger_get_prune_list(struct logger_list* logger_list,
                                      char* buf, size_t len);
int android_logger_set_prune_list(struct logger_list* logger_list, char* buf,
                                  size_t len);
#endif

#define ANDROID_LOG_RDONLY O_RDONLY
#define ANDROID_LOG_WRONLY O_WRONLY
#define ANDROID_LOG_RDWR O_RDWR
#define ANDROID_LOG_ACCMODE O_ACCMODE
#ifndef O_NONBLOCK
#define ANDROID_LOG_NONBLOCK 0x00000800
#else
#define ANDROID_LOG_NONBLOCK O_NONBLOCK
#endif
#if __ANDROID_USE_LIBLOG_READER_INTERFACE > 2
#define ANDROID_LOG_WRAP 0x40000000 /* Block until buffer about to wrap */
#define ANDROID_LOG_WRAP_DEFAULT_TIMEOUT 7200 /* 2 hour default */
#endif
#if __ANDROID_USE_LIBLOG_READER_INTERFACE > 1
#define ANDROID_LOG_PSTORE 0x80000000
#endif

struct logger_list* android_logger_list_alloc(int mode, unsigned int tail,
                                              pid_t pid);
struct logger_list* android_logger_list_alloc_time(int mode, log_time start,
                                                   pid_t pid);
void android_logger_list_free(struct logger_list* logger_list);
/* In the purest sense, the following two are orthogonal interfaces */
int android_logger_list_read(struct logger_list* logger_list,
                             struct log_msg* log_msg);

/* Multiple log_id_t opens */
struct logger* android_logger_open(struct logger_list* logger_list, log_id_t id);
#define android_logger_close android_logger_free
/* Single log_id_t open */
struct logger_list* android_logger_list_open(log_id_t id, int mode,
                                             unsigned int tail, pid_t pid);
#define android_logger_list_close android_logger_list_free

#endif /* __ANDROID_USE_LIBLOG_READER_INTERFACE */

#ifdef __cplusplus
}
#endif

#endif /* _LIBS_LOG_LOG_H */
