/*
 * Copyright (C) 2012 The Android Open Source Project
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

#ifndef _LIBS_CUTILS_TRACE_H
#define _LIBS_CUTILS_TRACE_H

#include <inttypes.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <unistd.h>

#include <cutils/compiler.h>

__BEGIN_DECLS

/**
 * The ATRACE_TAG macro can be defined before including this header to trace
 * using one of the tags defined below.  It must be defined to one of the
 * following ATRACE_TAG_* macros.  The trace tag is used to filter tracing in
 * userland to avoid some of the runtime cost of tracing when it is not desired.
 *
 * Defining ATRACE_TAG to be ATRACE_TAG_ALWAYS will result in the tracing always
 * being enabled - this should ONLY be done for debug code, as userland tracing
 * has a performance cost even when the trace is not being recorded.  Defining
 * ATRACE_TAG to be ATRACE_TAG_NEVER or leaving ATRACE_TAG undefined will result
 * in the tracing always being disabled.
 *
 * ATRACE_TAG_HAL should be bitwise ORed with the relevant tags for tracing
 * within a hardware module.  For example a camera hardware module would set:
 * #define ATRACE_TAG  (ATRACE_TAG_CAMERA | ATRACE_TAG_HAL)
 *
 * Keep these in sync with frameworks/base/core/java/android/os/Trace.java.
 */
#define ATRACE_TAG_NEVER            0       // This tag is never enabled.
#define ATRACE_TAG_ALWAYS           (1<<0)  // This tag is always enabled.
#define ATRACE_TAG_GRAPHICS         (1<<1)
#define ATRACE_TAG_INPUT            (1<<2)
#define ATRACE_TAG_VIEW             (1<<3)
#define ATRACE_TAG_WEBVIEW          (1<<4)
#define ATRACE_TAG_WINDOW_MANAGER   (1<<5)
#define ATRACE_TAG_ACTIVITY_MANAGER (1<<6)
#define ATRACE_TAG_SYNC_MANAGER     (1<<7)
#define ATRACE_TAG_AUDIO            (1<<8)
#define ATRACE_TAG_VIDEO            (1<<9)
#define ATRACE_TAG_CAMERA           (1<<10)
#define ATRACE_TAG_HAL              (1<<11)
#define ATRACE_TAG_APP              (1<<12)
#define ATRACE_TAG_RESOURCES        (1<<13)
#define ATRACE_TAG_DALVIK           (1<<14)
#define ATRACE_TAG_RS               (1<<15)
#define ATRACE_TAG_BIONIC           (1<<16)
#define ATRACE_TAG_POWER            (1<<17)
#define ATRACE_TAG_LAST             ATRACE_TAG_POWER

// Reserved for initialization.
#define ATRACE_TAG_NOT_READY        (1LL<<63)

#define ATRACE_TAG_VALID_MASK ((ATRACE_TAG_LAST - 1) | ATRACE_TAG_LAST)

#ifndef ATRACE_TAG
#define ATRACE_TAG ATRACE_TAG_NEVER
#elif ATRACE_TAG > ATRACE_TAG_VALID_MASK
#error ATRACE_TAG must be defined to be one of the tags defined in cutils/trace.h
#endif

/**
 * Opens the trace file for writing and reads the property for initial tags.
 * The atrace.tags.enableflags property sets the tags to trace.
 * This function should not be explicitly called, the first call to any normal
 * trace function will cause it to be run safely.
 */
void atrace_setup();

/**
 * If tracing is ready, set atrace_enabled_tags to the system property
 * debug.atrace.tags.enableflags. Can be used as a sysprop change callback.
 */
void atrace_update_tags();

/**
 * Set whether the process is debuggable.  By default the process is not
 * considered debuggable.  If the process is not debuggable then application-
 * level tracing is not allowed unless the ro.debuggable system property is
 * set to '1'.
 */
void atrace_set_debuggable(bool debuggable);

/**
 * Set whether tracing is enabled for the current process.  This is used to
 * prevent tracing within the Zygote process.
 */
void atrace_set_tracing_enabled(bool enabled);

/**
 * Flag indicating whether setup has been completed, initialized to 0.
 * Nonzero indicates setup has completed.
 * Note: This does NOT indicate whether or not setup was successful.
 */
extern atomic_bool atrace_is_ready;

/**
 * Set of ATRACE_TAG flags to trace for, initialized to ATRACE_TAG_NOT_READY.
 * A value of zero indicates setup has failed.
 * Any other nonzero value indicates setup has succeeded, and tracing is on.
 */
extern uint64_t atrace_enabled_tags;

/**
 * Handle to the kernel's trace buffer, initialized to -1.
 * Any other value indicates setup has succeeded, and is a valid fd for tracing.
 */
extern int atrace_marker_fd;

/**
 * atrace_init readies the process for tracing by opening the trace_marker file.
 * Calling any trace function causes this to be run, so calling it is optional.
 * This can be explicitly run to avoid setup delay on first trace function.
 */
#define ATRACE_INIT() atrace_init()
static inline void atrace_init()
{
    if (CC_UNLIKELY(!atomic_load_explicit(&atrace_is_ready, memory_order_acquire))) {
        atrace_setup();
    }
}

/**
 * Get the mask of all tags currently enabled.
 * It can be used as a guard condition around more expensive trace calculations.
 * Every trace function calls this, which ensures atrace_init is run.
 */
#define ATRACE_GET_ENABLED_TAGS() atrace_get_enabled_tags()
static inline uint64_t atrace_get_enabled_tags()
{
    atrace_init();
    return atrace_enabled_tags;
}

/**
 * Test if a given tag is currently enabled.
 * Returns nonzero if the tag is enabled, otherwise zero.
 * It can be used as a guard condition around more expensive trace calculations.
 */
#define ATRACE_ENABLED() atrace_is_tag_enabled(ATRACE_TAG)
static inline uint64_t atrace_is_tag_enabled(uint64_t tag)
{
    return atrace_get_enabled_tags() & tag;
}

/**
 * Trace the beginning of a context.  name is used to identify the context.
 * This is often used to time function execution.
 */
#define ATRACE_BEGIN(name) atrace_begin(ATRACE_TAG, name)
static inline void atrace_begin(uint64_t tag, const char* name)
{
    if (CC_UNLIKELY(atrace_is_tag_enabled(tag))) {
        void atrace_begin_body(const char*);
        atrace_begin_body(name);
    }
}

/**
 * Trace the end of a context.
 * This should match up (and occur after) a corresponding ATRACE_BEGIN.
 */
#define ATRACE_END() atrace_end(ATRACE_TAG)
static inline void atrace_end(uint64_t tag)
{
    if (CC_UNLIKELY(atrace_is_tag_enabled(tag))) {
        char c = 'E';
        write(atrace_marker_fd, &c, 1);
    }
}

/**
 * Trace the beginning of an asynchronous event. Unlike ATRACE_BEGIN/ATRACE_END
 * contexts, asynchronous events do not need to be nested. The name describes
 * the event, and the cookie provides a unique identifier for distinguishing
 * simultaneous events. The name and cookie used to begin an event must be
 * used to end it.
 */
#define ATRACE_ASYNC_BEGIN(name, cookie) \
    atrace_async_begin(ATRACE_TAG, name, cookie)
static inline void atrace_async_begin(uint64_t tag, const char* name,
        int32_t cookie)
{
    if (CC_UNLIKELY(atrace_is_tag_enabled(tag))) {
        void atrace_async_begin_body(const char*, int32_t);
        atrace_async_begin_body(name, cookie);
    }
}

/**
 * Trace the end of an asynchronous event.
 * This should have a corresponding ATRACE_ASYNC_BEGIN.
 */
#define ATRACE_ASYNC_END(name, cookie) atrace_async_end(ATRACE_TAG, name, cookie)
static inline void atrace_async_end(uint64_t tag, const char* name, int32_t cookie)
{
    if (CC_UNLIKELY(atrace_is_tag_enabled(tag))) {
        void atrace_async_end_body(const char*, int32_t);
        atrace_async_end_body(name, cookie);
    }
}

/**
 * Traces an integer counter value.  name is used to identify the counter.
 * This can be used to track how a value changes over time.
 */
#define ATRACE_INT(name, value) atrace_int(ATRACE_TAG, name, value)
static inline void atrace_int(uint64_t tag, const char* name, int32_t value)
{
    if (CC_UNLIKELY(atrace_is_tag_enabled(tag))) {
        void atrace_int_body(const char*, int32_t);
        atrace_int_body(name, value);
    }
}

/**
 * Traces a 64-bit integer counter value.  name is used to identify the
 * counter. This can be used to track how a value changes over time.
 */
#define ATRACE_INT64(name, value) atrace_int64(ATRACE_TAG, name, value)
static inline void atrace_int64(uint64_t tag, const char* name, int64_t value)
{
    if (CC_UNLIKELY(atrace_is_tag_enabled(tag))) {
        void atrace_int64_body(const char*, int64_t);
        atrace_int64_body(name, value);
    }
}

__END_DECLS

#endif // _LIBS_CUTILS_TRACE_H
