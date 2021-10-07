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

#ifndef ANDROID_CUTILS_MEMORY_H
#define ANDROID_CUTILS_MEMORY_H

#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* size is given in bytes and must be multiple of 2 */
void android_memset16(uint16_t* dst, uint16_t value, size_t size);

/* size is given in bytes and must be multiple of 4 */
void android_memset32(uint32_t* dst, uint32_t value, size_t size);

#if defined(__GLIBC__) || defined(_WIN32)
/* Declaration of strlcpy() for platforms that don't already have it. */
size_t strlcpy(char *dst, const char *src, size_t size);
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ANDROID_CUTILS_MEMORY_H
