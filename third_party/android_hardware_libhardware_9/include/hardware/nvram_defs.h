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

/*
 * This file contains data type definitions and constants that are useful to
 * code interacting with and implementing the NVRAM HAL, even though it doesn't
 * use the actual NVRAM HAL module interface. Keeping this in a separate file
 * simplifies inclusion in low-level code which can't easily include the heavier
 * hardware.h due to lacking standard headers.
 */

#ifndef ANDROID_HARDWARE_NVRAM_DEFS_H
#define ANDROID_HARDWARE_NVRAM_DEFS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/* Values returned by nvram_device methods. */
typedef uint32_t nvram_result_t;

const nvram_result_t NV_RESULT_SUCCESS = 0;
const nvram_result_t NV_RESULT_INTERNAL_ERROR = 1;
const nvram_result_t NV_RESULT_ACCESS_DENIED = 2;
const nvram_result_t NV_RESULT_INVALID_PARAMETER = 3;
const nvram_result_t NV_RESULT_SPACE_DOES_NOT_EXIST = 4;
const nvram_result_t NV_RESULT_SPACE_ALREADY_EXISTS = 5;
const nvram_result_t NV_RESULT_OPERATION_DISABLED = 6;

/* Values describing available access controls. */
typedef uint32_t nvram_control_t;

const nvram_control_t NV_CONTROL_PERSISTENT_WRITE_LOCK = 1;
const nvram_control_t NV_CONTROL_BOOT_WRITE_LOCK = 2;
const nvram_control_t NV_CONTROL_BOOT_READ_LOCK = 3;
const nvram_control_t NV_CONTROL_WRITE_AUTHORIZATION = 4;
const nvram_control_t NV_CONTROL_READ_AUTHORIZATION = 5;
const nvram_control_t NV_CONTROL_WRITE_EXTEND = 6;

const uint32_t NV_UNLIMITED_SPACES = 0xFFFFFFFF;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ANDROID_HARDWARE_NVRAM_DEFS_H
