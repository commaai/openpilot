/*
 * Copyright (C) 2015 The Android Open Source Project
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


#ifndef ANDROID_NATIVE_TRACE_H
#define ANDROID_NATIVE_TRACE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns true if tracing is enabled. Use this signal to avoid expensive computation only necessary
 * when tracing is enabled.
 */
bool ATrace_isEnabled();

/**
 * Writes a tracing message to indicate that the given section of code has begun. This call must be
 * followed by a corresponding call to endSection() on the same thread.
 *
 * Note: At this time the vertical bar character '|' and newline character '\n' are used internally
 * by the tracing mechanism. If sectionName contains these characters they will be replaced with a
 * space character in the trace.
 */
void ATrace_beginSection(const char* sectionName);

/**
 * Writes a tracing message to indicate that a given section of code has ended. This call must be
 * preceeded by a corresponding call to beginSection(char*) on the same thread. Calling this method
 * will mark the end of the most recently begun section of code, so care must be taken to ensure
 * that beginSection / endSection pairs are properly nested and called from the same thread.
 */
void ATrace_endSection();

#ifdef __cplusplus
};
#endif

#endif // ANDROID_NATIVE_TRACE_H
