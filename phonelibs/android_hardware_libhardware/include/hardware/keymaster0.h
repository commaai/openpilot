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

#ifndef ANDROID_HARDWARE_KEYMASTER_0_H
#define ANDROID_HARDWARE_KEYMASTER_0_H

#include <hardware/keymaster_common.h>

__BEGIN_DECLS

/**
 * Keymaster0 device definition.
 */
struct keymaster0_device {
    /**
     * Common methods of the keymaster device.  This *must* be the first member of
     * keymaster0_device as users of this structure will cast a hw_device_t to
     * keymaster0_device pointer in contexts where it's known the hw_device_t references a
     * keymaster0_device.
     */
    struct hw_device_t common;

    /**
     * THIS IS DEPRECATED. Use the new "module_api_version" and "hal_api_version"
     * fields in the keymaster_module initialization instead.
     */
    uint32_t client_version;

    /**
     * See flags defined for keymaster0_device::flags in keymaster_common.h
     */
    uint32_t flags;

    void* context;

    /**
     * Generates a public and private key. The key-blob returned is opaque
     * and must subsequently provided for signing and verification.
     *
     * Returns: 0 on success or an error code less than 0.
     */
    int (*generate_keypair)(const struct keymaster0_device* dev,
            const keymaster_keypair_t key_type, const void* key_params,
            uint8_t** key_blob, size_t* key_blob_length);

    /**
     * Imports a public and private key pair. The imported keys will be in
     * PKCS#8 format with DER encoding (Java standard). The key-blob
     * returned is opaque and will be subsequently provided for signing
     * and verification.
     *
     * Returns: 0 on success or an error code less than 0.
     */
    int (*import_keypair)(const struct keymaster0_device* dev,
            const uint8_t* key, const size_t key_length,
            uint8_t** key_blob, size_t* key_blob_length);

    /**
     * Gets the public key part of a key pair. The public key must be in
     * X.509 format (Java standard) encoded byte array.
     *
     * Returns: 0 on success or an error code less than 0.
     * On error, x509_data should not be allocated.
     */
    int (*get_keypair_public)(const struct keymaster0_device* dev,
            const uint8_t* key_blob, const size_t key_blob_length,
            uint8_t** x509_data, size_t* x509_data_length);

    /**
     * Deletes the key pair associated with the key blob.
     *
     * This function is optional and should be set to NULL if it is not
     * implemented.
     *
     * Returns 0 on success or an error code less than 0.
     */
    int (*delete_keypair)(const struct keymaster0_device* dev,
            const uint8_t* key_blob, const size_t key_blob_length);

    /**
     * Deletes all keys in the hardware keystore. Used when keystore is
     * reset completely.
     *
     * This function is optional and should be set to NULL if it is not
     * implemented.
     *
     * Returns 0 on success or an error code less than 0.
     */
    int (*delete_all)(const struct keymaster0_device* dev);

    /**
     * Signs data using a key-blob generated before. This can use either
     * an asymmetric key or a secret key.
     *
     * Returns: 0 on success or an error code less than 0.
     */
    int (*sign_data)(const struct keymaster0_device* dev,
            const void* signing_params,
            const uint8_t* key_blob, const size_t key_blob_length,
            const uint8_t* data, const size_t data_length,
            uint8_t** signed_data, size_t* signed_data_length);

    /**
     * Verifies data signed with a key-blob. This can use either
     * an asymmetric key or a secret key.
     *
     * Returns: 0 on successful verification or an error code less than 0.
     */
    int (*verify_data)(const struct keymaster0_device* dev,
            const void* signing_params,
            const uint8_t* key_blob, const size_t key_blob_length,
            const uint8_t* signed_data, const size_t signed_data_length,
            const uint8_t* signature, const size_t signature_length);
};
typedef struct keymaster0_device keymaster0_device_t;


/* Convenience API for opening and closing keymaster devices */

static inline int keymaster0_open(const struct hw_module_t* module,
        keymaster0_device_t** device)
{
    int rc = module->methods->open(module, KEYSTORE_KEYMASTER,
            (struct hw_device_t**) device);

    return rc;
}

static inline int keymaster0_close(keymaster0_device_t* device)
{
    return device->common.close(&device->common);
}

__END_DECLS

#endif  // ANDROID_HARDWARE_KEYMASTER_0_H
