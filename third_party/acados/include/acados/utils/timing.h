/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


#ifndef ACADOS_UTILS_TIMING_H_
#define ACADOS_UTILS_TIMING_H_

#include "acados/utils/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MEASURE_TIMINGS
#if (defined _WIN32 || defined _WIN64) && !(defined __MINGW32__ || defined __MINGW64__)

/* Use Windows QueryPerformanceCounter for timing. */
#include <Windows.h>

/** A structure for keeping internal timer data. */
typedef struct acados_timer_
{
    LARGE_INTEGER tic;
    LARGE_INTEGER toc;
    LARGE_INTEGER freq;
} acados_timer;

#elif defined(__APPLE__)

#include <mach/mach_time.h>

/** A structure for keeping internal timer data. */
typedef struct acados_timer_
{
    uint64_t tic;
    uint64_t toc;
    mach_timebase_info_data_t tinfo;
} acados_timer;

#elif defined(__MABX2__)

#include <brtenv.h>

typedef struct acados_timer_
{
    double time;
} acados_timer;

#else

/* Use POSIX clock_gettime() for timing on non-Windows machines. */
#include <time.h>

#if (__STDC_VERSION__ >= 199901L) && !(defined __MINGW32__ || defined __MINGW64__)  // C99 Mode

#include <sys/stat.h>
#include <sys/time.h>

typedef struct acados_timer_
{
    struct timeval tic;
    struct timeval toc;
} acados_timer;

#else  // ANSI C Mode

/** A structure for keeping internal timer data. */
typedef struct acados_timer_
{
    struct timespec tic;
    struct timespec toc;
} acados_timer;

#endif  // __STDC_VERSION__ >= 199901L

#endif  // (defined _WIN32 || defined _WIN64)

#else

// Dummy type when timings are off
typedef real_t acados_timer;

#endif  // MEASURE_TIMINGS

/** A function for measurement of the current time. */
void acados_tic(acados_timer* t);

/** A function which returns the elapsed time. */
real_t acados_toc(acados_timer* t);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_UTILS_TIMING_H_
