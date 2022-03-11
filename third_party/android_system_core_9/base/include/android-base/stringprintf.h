/*
 * Copyright (C) 2011 The Android Open Source Project
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

#ifndef ANDROID_BASE_STRINGPRINTF_H
#define ANDROID_BASE_STRINGPRINTF_H

#include <stdarg.h>
#include <string>

namespace android {
namespace base {

// These printf-like functions are implemented in terms of vsnprintf, so they
// use the same attribute for compile-time format string checking. On Windows,
// if the mingw version of vsnprintf is used, use `gnu_printf' which allows z
// in %zd and PRIu64 (and related) to be recognized by the compile-time
// checking.
#define ANDROID_BASE_FORMAT_ARCHETYPE __printf__
#ifdef __USE_MINGW_ANSI_STDIO
#if __USE_MINGW_ANSI_STDIO
#undef ANDROID_BASE_FORMAT_ARCHETYPE
#define ANDROID_BASE_FORMAT_ARCHETYPE gnu_printf
#endif
#endif

// Returns a string corresponding to printf-like formatting of the arguments.
std::string StringPrintf(const char* fmt, ...)
    __attribute__((__format__(ANDROID_BASE_FORMAT_ARCHETYPE, 1, 2)));

// Appends a printf-like formatting of the arguments to 'dst'.
void StringAppendF(std::string* dst, const char* fmt, ...)
    __attribute__((__format__(ANDROID_BASE_FORMAT_ARCHETYPE, 2, 3)));

// Appends a printf-like formatting of the arguments to 'dst'.
void StringAppendV(std::string* dst, const char* format, va_list ap)
    __attribute__((__format__(ANDROID_BASE_FORMAT_ARCHETYPE, 2, 0)));

#undef ANDROID_BASE_FORMAT_ARCHETYPE

}  // namespace base
}  // namespace android

#endif  // ANDROID_BASE_STRINGPRINTF_H
