/*
 * Copyright (C) 2014 The Android Open Source Project
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

#include <stdint.h>

#ifndef ANDROID_HARDWARE_HW_AUTH_TOKEN_H
#define ANDROID_HARDWARE_HW_AUTH_TOKEN_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

const uint8_t HW_AUTH_TOKEN_VERSION = 0;

typedef enum {
    HW_AUTH_NONE = 0,
    HW_AUTH_PASSWORD = 1 << 0,
    HW_AUTH_FINGERPRINT = 1 << 1,
    // Additional entries should be powers of 2.
    HW_AUTH_ANY = UINT32_MAX,
} hw_authenticator_type_t;

/**
 * Data format for an authentication record used to prove successful authentication.
 */
typedef struct __attribute__((__packed__)) {
    uint8_t version;  // Current version is 0
    uint64_t challenge;
    uint64_t user_id;             // secure user ID, not Android user ID
    uint64_t authenticator_id;    // secure authenticator ID
    uint32_t authenticator_type;  // hw_authenticator_type_t, in network order
    uint64_t timestamp;           // in network order
    uint8_t hmac[32];
} hw_auth_token_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ANDROID_HARDWARE_HW_AUTH_TOKEN_H
