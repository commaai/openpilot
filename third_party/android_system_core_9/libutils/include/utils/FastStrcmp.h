/*
 * Copyright (C) 2014-2016 The Android Open Source Project
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

#ifndef _ANDROID_UTILS_FASTSTRCMP_H__
#define _ANDROID_UTILS_FASTSTRCMP_H__

#include <ctype.h>
#include <string.h>

#ifndef __predict_true
#define __predict_true(exp) __builtin_expect((exp) != 0, 1)
#endif

#ifdef __cplusplus

// Optimized for instruction cache locality
//
// Template class fastcmp used to create more time-efficient str*cmp
// functions by pre-checking the first character before resorting
// to calling the underlying string function.  Profiled with a
// measurable speedup when used in hot code.  Usage is of the form:
//
//  fastcmp<strncmp>(str1, str2, len)
//
// NB: use fasticmp for the case insensitive str*cmp functions.
// NB: Returns boolean, do not use if expecting to check negative value.
//     Thus not semantically identical to the expected function behavior.

template <int (*cmp)(const char* l, const char* r, const size_t s)>
static inline int fastcmp(const char* l, const char* r, const size_t s) {
    const ssize_t n = s;  // To help reject negative sizes, treat like zero
    return __predict_true(n > 0) &&
           ((*l != *r) || (__predict_true(n > 1) && cmp(l + 1, r + 1, n - 1)));
}

template <int (*cmp)(const char* l, const char* r, const size_t s)>
static inline int fasticmp(const char* l, const char* r, const size_t s) {
    const ssize_t n = s;  // To help reject negative sizes, treat like zero
    return __predict_true(n > 0) &&
           ((tolower(*l) != tolower(*r)) || (__predict_true(n > 1) && cmp(l + 1, r + 1, n - 1)));
}

template <int (*cmp)(const void* l, const void* r, const size_t s)>
static inline int fastcmp(const void* lv, const void* rv, const size_t s) {
    const char* l = static_cast<const char*>(lv);
    const char* r = static_cast<const char*>(rv);
    const ssize_t n = s;  // To help reject negative sizes, treat like zero
    return __predict_true(n > 0) &&
           ((*l != *r) || (__predict_true(n > 1) && cmp(l + 1, r + 1, n - 1)));
}

template <int (*cmp)(const char* l, const char* r)>
static inline int fastcmp(const char* l, const char* r) {
    return (*l != *r) || (__predict_true(*l) && cmp(l + 1, r + 1));
}

template <int (*cmp)(const char* l, const char* r)>
static inline int fasticmp(const char* l, const char* r) {
    return (tolower(*l) != tolower(*r)) || (__predict_true(*l) && cmp(l + 1, r + 1));
}

#endif

#endif // _ANDROID_UTILS_FASTSTRCMP_H__
