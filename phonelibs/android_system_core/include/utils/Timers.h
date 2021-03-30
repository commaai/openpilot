/*
 * Copyright (C) 2005 The Android Open Source Project
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

//
// Timer functions.
//
#ifndef _LIBS_UTILS_TIMERS_H
#define _LIBS_UTILS_TIMERS_H

#include <stdint.h>
#include <sys/types.h>
#include <sys/time.h>

#include <utils/Compat.h>

// ------------------------------------------------------------------
// C API

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t nsecs_t;       // nano-seconds

static CONSTEXPR inline nsecs_t seconds_to_nanoseconds(nsecs_t secs)
{
    return secs*1000000000;
}

static CONSTEXPR inline nsecs_t milliseconds_to_nanoseconds(nsecs_t secs)
{
    return secs*1000000;
}

static CONSTEXPR inline nsecs_t microseconds_to_nanoseconds(nsecs_t secs)
{
    return secs*1000;
}

static CONSTEXPR inline nsecs_t nanoseconds_to_seconds(nsecs_t secs)
{
    return secs/1000000000;
}

static CONSTEXPR inline nsecs_t nanoseconds_to_milliseconds(nsecs_t secs)
{
    return secs/1000000;
}

static CONSTEXPR inline nsecs_t nanoseconds_to_microseconds(nsecs_t secs)
{
    return secs/1000;
}

static CONSTEXPR inline nsecs_t s2ns(nsecs_t v)  {return seconds_to_nanoseconds(v);}
static CONSTEXPR inline nsecs_t ms2ns(nsecs_t v) {return milliseconds_to_nanoseconds(v);}
static CONSTEXPR inline nsecs_t us2ns(nsecs_t v) {return microseconds_to_nanoseconds(v);}
static CONSTEXPR inline nsecs_t ns2s(nsecs_t v)  {return nanoseconds_to_seconds(v);}
static CONSTEXPR inline nsecs_t ns2ms(nsecs_t v) {return nanoseconds_to_milliseconds(v);}
static CONSTEXPR inline nsecs_t ns2us(nsecs_t v) {return nanoseconds_to_microseconds(v);}

static CONSTEXPR inline nsecs_t seconds(nsecs_t v)      { return s2ns(v); }
static CONSTEXPR inline nsecs_t milliseconds(nsecs_t v) { return ms2ns(v); }
static CONSTEXPR inline nsecs_t microseconds(nsecs_t v) { return us2ns(v); }

enum {
    SYSTEM_TIME_REALTIME = 0,  // system-wide realtime clock
    SYSTEM_TIME_MONOTONIC = 1, // monotonic time since unspecified starting point
    SYSTEM_TIME_PROCESS = 2,   // high-resolution per-process clock
    SYSTEM_TIME_THREAD = 3,    // high-resolution per-thread clock
    SYSTEM_TIME_BOOTTIME = 4   // same as SYSTEM_TIME_MONOTONIC, but including CPU suspend time
};

// return the system-time according to the specified clock
#ifdef __cplusplus
nsecs_t systemTime(int clock = SYSTEM_TIME_MONOTONIC);
#else
nsecs_t systemTime(int clock);
#endif // def __cplusplus

/**
 * Returns the number of milliseconds to wait between the reference time and the timeout time.
 * If the timeout is in the past relative to the reference time, returns 0.
 * If the timeout is more than INT_MAX milliseconds in the future relative to the reference time,
 * such as when timeoutTime == LLONG_MAX, returns -1 to indicate an infinite timeout delay.
 * Otherwise, returns the difference between the reference time and timeout time
 * rounded up to the next millisecond.
 */
int toMillisecondTimeoutDelay(nsecs_t referenceTime, nsecs_t timeoutTime);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _LIBS_UTILS_TIMERS_H
