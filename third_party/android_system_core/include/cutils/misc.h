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

#ifndef __CUTILS_MISC_H
#define __CUTILS_MISC_H

#ifdef __cplusplus
extern "C" {
#endif

        /* Load an entire file into a malloc'd chunk of memory
         * that is length_of_file + 1 (null terminator).  If
         * sz is non-zero, return the size of the file via sz.
         * Returns 0 on failure.
         */
extern void *load_file(const char *fn, unsigned *sz);

        /* This is the range of UIDs (and GIDs) that are reserved
         * for assigning to applications.
         */
#define FIRST_APPLICATION_UID 10000
#define LAST_APPLICATION_UID 99999

#ifdef __cplusplus
}
#endif

#endif /* __CUTILS_MISC_H */ 
