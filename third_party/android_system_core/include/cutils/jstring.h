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

#ifndef __CUTILS_STRING16_H
#define __CUTILS_STRING16_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if __STDC_VERSION__ < 201112L && __cplusplus < 201103L
  typedef uint16_t char16_t;
#endif
  // otherwise char16_t is a keyword with the right semantics

extern char * strndup16to8 (const char16_t* s, size_t n);
extern size_t strnlen16to8 (const char16_t* s, size_t n);
extern char * strncpy16to8 (char *dest, const char16_t*s, size_t n);

extern char16_t * strdup8to16 (const char* s, size_t *out_len);
extern size_t strlen8to16 (const char* utf8Str);
extern char16_t * strcpy8to16 (char16_t *dest, const char*s, size_t *out_len);
extern char16_t * strcpylen8to16 (char16_t *dest, const char*s, int length,
    size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* __CUTILS_STRING16_H */
