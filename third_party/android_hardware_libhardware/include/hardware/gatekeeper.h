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

#ifndef ANDROID_HARDWARE_GATEKEEPER_H
#define ANDROID_HARDWARE_GATEKEEPER_H

#include <sys/cdefs.h>
#include <sys/types.h>
#include <hardware/hardware.h>

__BEGIN_DECLS

#define GATEKEEPER_HARDWARE_MODULE_ID "gatekeeper"

#define GATEKEEPER_MODULE_API_VERSION_0_1 HARDWARE_MODULE_API_VERSION(0, 1)

#define HARDWARE_GATEKEEPER "gatekeeper"

struct gatekeeper_module {
    /**
     * Comon methods of the gatekeeper module. This *must* be the first member of
     * gatekeeper_module as users of this structure will cast a hw_module_t to
     * a gatekeeper_module pointer in the appropriate context.
     */
    hw_module_t common;
};

struct gatekeeper_device {
    /**
     * Common methods of the gatekeeper device. As above, this must be the first
     * member of keymaster_device.
     */
    hw_device_t common;

    /**
     * Enrolls desired_password, which should be derived from a user selected pin or password,
     * with the authentication factor private key used only for enrolling authentication
     * factor data.
     *
     * If there was already a password enrolled, it should be provided in
     * current_password_handle, along with the current password in current_password
     * that should validate against current_password_handle.
     *
     * Parameters:
     * - dev: pointer to gatekeeper_device acquired via calls to gatekeeper_open
     * - uid: the Android user identifier
     *
     * - current_password_handle: the currently enrolled password handle the user
     *   wants to replace. May be null if there's no currently enrolled password.
     * - current_password_handle_length: the length in bytes of the buffer pointed
     *   at by current_password_handle. Must be 0 if current_password_handle is NULL.
     *
     * - current_password: the user's current password in plain text. If presented,
     *   it MUST verify against current_password_handle.
     * - current_password_length: the size in bytes of the buffer pointed at by
     *   current_password. Must be 0 if the current_password is NULL.
     *
     * - desired_password: the new password the user wishes to enroll in plain-text.
     *   Cannot be NULL.
     * - desired_password_length: the length in bytes of the buffer pointed at by
     *   desired_password.
     *
     * - enrolled_password_handle: on success, a buffer will be allocated with the
     *   new password handle referencing the password provided in desired_password.
     *   This buffer can be used on subsequent calls to enroll or verify.
     *   The caller is responsible for deallocating this buffer via a call to delete[]
     * - enrolled_password_handle_length: pointer to the length in bytes of the buffer allocated
     *   by this function and pointed to by *enrolled_password_handle_length.
     *
     * Returns:
     * - 0 on success
     * - An error code < 0 on failure, or
     * - A timeout value T > 0 if the call should not be re-attempted until T milliseconds
     *   have elapsed.
     *
     * On error, enrolled_password_handle will not be allocated.
     */
    int (*enroll)(const struct gatekeeper_device *dev, uint32_t uid,
            const uint8_t *current_password_handle, uint32_t current_password_handle_length,
            const uint8_t *current_password, uint32_t current_password_length,
            const uint8_t *desired_password, uint32_t desired_password_length,
            uint8_t **enrolled_password_handle, uint32_t *enrolled_password_handle_length);

    /**
     * Verifies provided_password matches enrolled_password_handle.
     *
     * Implementations of this module may retain the result of this call
     * to attest to the recency of authentication.
     *
     * On success, writes the address of a verification token to auth_token,
     * usable to attest password verification to other trusted services. Clients
     * may pass NULL for this value.
     *
     * Parameters:
     * - dev: pointer to gatekeeper_device acquired via calls to gatekeeper_open
     * - uid: the Android user identifier
     *
     * - challenge: An optional challenge to authenticate against, or 0. Used when a separate
     *              authenticator requests password verification, or for transactional
     *              password authentication.
     *
     * - enrolled_password_handle: the currently enrolled password handle that the
     *   user wishes to verify against.
     * - enrolled_password_handle_length: the length in bytes of the buffer pointed
     *   to by enrolled_password_handle
     *
     * - provided_password: the plaintext password to be verified against the
     *   enrolled_password_handle
     * - provided_password_length: the length in bytes of the buffer pointed to by
     *   provided_password
     *
     * - auth_token: on success, a buffer containing the authentication token
     *   resulting from this verification is assigned to *auth_token. The caller
     *   is responsible for deallocating this memory via a call to delete[]
     * - auth_token_length: on success, the length in bytes of the authentication
     *   token assigned to *auth_token will be assigned to *auth_token_length
     *
     * - request_reenroll: a request to the upper layers to re-enroll the verified
     *   password due to a version change. Not set if verification fails.
     *
     * Returns:
     * - 0 on success
     * - An error code < 0 on failure, or
     * - A timeout value T > 0 if the call should not be re-attempted until T milliseconds
     *   have elapsed.
     * On error, auth token will not be allocated
     */
    int (*verify)(const struct gatekeeper_device *dev, uint32_t uid, uint64_t challenge,
            const uint8_t *enrolled_password_handle, uint32_t enrolled_password_handle_length,
            const uint8_t *provided_password, uint32_t provided_password_length,
            uint8_t **auth_token, uint32_t *auth_token_length, bool *request_reenroll);

    /*
     * Deletes the enrolled_password_handle associated wth the uid. Once deleted
     * the user cannot be verified anymore.
     * This function is optional and should be set to NULL if it is not implemented.
     *
     * Parameters
     * - dev: pointer to gatekeeper_device acquired via calls to gatekeeper_open
     * - uid: the Android user identifier
     *
     * Returns:
     * - 0 on success
     * - An error code < 0 on failure
     */
    int (*delete_user)(const struct gatekeeper_device *dev,  uint32_t uid);

    /*
     * Deletes all the enrolled_password_handles for all uid's. Once called,
     * no users will be enrolled on the device.
     * This function is optional and should be set to NULL if it is not implemented.
     *
     * Parameters
     * - dev: pointer to gatekeeper_device acquired via calls to gatekeeper_open
     *
     * Returns:
     * - 0 on success
     * - An error code < 0 on failure
     */
    int (*delete_all_users)(const struct gatekeeper_device *dev);
};

typedef struct gatekeeper_device gatekeeper_device_t;

static inline int gatekeeper_open(const struct hw_module_t *module,
        gatekeeper_device_t **device) {
    return module->methods->open(module, HARDWARE_GATEKEEPER,
            (struct hw_device_t **) device);
}

static inline int gatekeeper_close(gatekeeper_device_t *device) {
    return device->common.close(&device->common);
}

__END_DECLS

#endif // ANDROID_HARDWARE_GATEKEEPER_H
