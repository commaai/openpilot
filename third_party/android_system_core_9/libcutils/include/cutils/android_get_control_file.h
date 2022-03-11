/*
 * Copyright (C) 2016 The Android Open Source Project
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

#ifndef __CUTILS_ANDROID_GET_CONTROL_FILE_H
#define __CUTILS_ANDROID_GET_CONTROL_FILE_H

#define ANDROID_FILE_ENV_PREFIX "ANDROID_FILE_"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * android_get_control_file - simple helper function to get the file
 * descriptor of our init-managed file. `path' is the filename path as
 * given in init.rc. Returns -1 on error.
 */
int android_get_control_file(const char* path);

#ifdef __cplusplus
}
#endif

#endif /* __CUTILS_ANDROID_GET_CONTROL_FILE_H */
