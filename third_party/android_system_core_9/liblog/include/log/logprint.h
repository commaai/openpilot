/*
 * Copyright (C) 2006 The Android Open Source Project
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

#ifndef _LOGPRINT_H
#define _LOGPRINT_H

#include <pthread.h>

#include <android/log.h>
#include <log/event_tag_map.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  /* Verbs */
  FORMAT_OFF = 0,
  FORMAT_BRIEF,
  FORMAT_PROCESS,
  FORMAT_TAG,
  FORMAT_THREAD,
  FORMAT_RAW,
  FORMAT_TIME,
  FORMAT_THREADTIME,
  FORMAT_LONG,
  /* Adverbs. The following are modifiers to above format verbs */
  FORMAT_MODIFIER_COLOR,     /* converts priority to color */
  FORMAT_MODIFIER_TIME_USEC, /* switches from msec to usec time precision */
  FORMAT_MODIFIER_PRINTABLE, /* converts non-printable to printable escapes */
  FORMAT_MODIFIER_YEAR,      /* Adds year to date */
  FORMAT_MODIFIER_ZONE,      /* Adds zone to date, + UTC */
  FORMAT_MODIFIER_EPOCH,     /* Print time as seconds since Jan 1 1970 */
  FORMAT_MODIFIER_MONOTONIC, /* Print cpu time as seconds since start */
  FORMAT_MODIFIER_UID,       /* Adds uid */
  FORMAT_MODIFIER_DESCRIPT,  /* Adds descriptive */
  /* private, undocumented */
  FORMAT_MODIFIER_TIME_NSEC, /* switches from msec to nsec time precision */
} AndroidLogPrintFormat;

typedef struct AndroidLogFormat_t AndroidLogFormat;

typedef struct AndroidLogEntry_t {
  time_t tv_sec;
  long tv_nsec;
  android_LogPriority priority;
  int32_t uid;
  int32_t pid;
  int32_t tid;
  const char* tag;
  size_t tagLen;
  size_t messageLen;
  const char* message;
} AndroidLogEntry;

AndroidLogFormat* android_log_format_new();

void android_log_format_free(AndroidLogFormat* p_format);

/* currently returns 0 if format is a modifier, 1 if not */
int android_log_setPrintFormat(AndroidLogFormat* p_format,
                               AndroidLogPrintFormat format);

/**
 * Returns FORMAT_OFF on invalid string
 */
AndroidLogPrintFormat android_log_formatFromString(const char* s);

/**
 * filterExpression: a single filter expression
 * eg "AT:d"
 *
 * returns 0 on success and -1 on invalid expression
 *
 * Assumes single threaded execution
 *
 */

int android_log_addFilterRule(AndroidLogFormat* p_format,
                              const char* filterExpression);

/**
 * filterString: a whitespace-separated set of filter expressions
 * eg "AT:d *:i"
 *
 * returns 0 on success and -1 on invalid expression
 *
 * Assumes single threaded execution
 *
 */

int android_log_addFilterString(AndroidLogFormat* p_format,
                                const char* filterString);

/**
 * returns 1 if this log line should be printed based on its priority
 * and tag, and 0 if it should not
 */
int android_log_shouldPrintLine(AndroidLogFormat* p_format, const char* tag,
                                android_LogPriority pri);

/**
 * Splits a wire-format buffer into an AndroidLogEntry
 * entry allocated by caller. Pointers will point directly into buf
 *
 * Returns 0 on success and -1 on invalid wire format (entry will be
 * in unspecified state)
 */
int android_log_processLogBuffer(struct logger_entry* buf,
                                 AndroidLogEntry* entry);

/**
 * Like android_log_processLogBuffer, but for binary logs.
 *
 * If "map" is non-NULL, it will be used to convert the log tag number
 * into a string.
 */
int android_log_processBinaryLogBuffer(struct logger_entry* buf,
                                       AndroidLogEntry* entry,
                                       const EventTagMap* map, char* messageBuf,
                                       int messageBufLen);

/**
 * Formats a log message into a buffer
 *
 * Uses defaultBuffer if it can, otherwise malloc()'s a new buffer
 * If return value != defaultBuffer, caller must call free()
 * Returns NULL on malloc error
 */

char* android_log_formatLogLine(AndroidLogFormat* p_format, char* defaultBuffer,
                                size_t defaultBufferSize,
                                const AndroidLogEntry* p_line,
                                size_t* p_outLength);

/**
 * Either print or do not print log line, based on filter
 *
 * Assumes single threaded execution
 *
 */
int android_log_printLogLine(AndroidLogFormat* p_format, int fd,
                             const AndroidLogEntry* entry);

#ifdef __cplusplus
}
#endif

#endif /*_LOGPRINT_H*/
