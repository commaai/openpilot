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

#ifndef ANDROID_HARDWARE_KEYMASTER_COMMON_H
#define ANDROID_HARDWARE_KEYMASTER_COMMON_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>

__BEGIN_DECLS

/**
 * The id of this module
 */
#define KEYSTORE_HARDWARE_MODULE_ID "keystore"

#define KEYSTORE_KEYMASTER "keymaster"


/**
 * Settings for "module_api_version" and "hal_api_version"
 * fields in the keymaster_module initialization.
 */

/**
 * Keymaster 0.X module version provide the same APIs, but later versions add more options
 * for algorithms and flags.
 */
#define KEYMASTER_MODULE_API_VERSION_0_2 HARDWARE_MODULE_API_VERSION(0, 2)
#define KEYMASTER_DEVICE_API_VERSION_0_2 HARDWARE_DEVICE_API_VERSION(0, 2)

#define KEYMASTER_MODULE_API_VERSION_0_3 HARDWARE_MODULE_API_VERSION(0, 3)
#define KEYMASTER_DEVICE_API_VERSION_0_3 HARDWARE_DEVICE_API_VERSION(0, 3)

/**
 * Keymaster 1.0 module version provides a completely different API, incompatible with 0.X.
 */
#define KEYMASTER_MODULE_API_VERSION_1_0 HARDWARE_MODULE_API_VERSION(1, 0)
#define KEYMASTER_DEVICE_API_VERSION_1_0 HARDWARE_DEVICE_API_VERSION(1, 0)

struct keystore_module {
    /**
     * Common methods of the keystore module.  This *must* be the first member of keystore_module as
     * users of this structure will cast a hw_module_t to keystore_module pointer in contexts where
     * it's known the hw_module_t references a keystore_module.
     */
    hw_module_t common;

    /* There are no keystore module methods other than the common ones. */
};

/**
 * Flags for keymaster0_device::flags
 */
enum {
    /*
     * Indicates this keymaster implementation does not have hardware that
     * keeps private keys out of user space.
     *
     * This should not be implemented on anything other than the default
     * implementation.
     */
    KEYMASTER_SOFTWARE_ONLY = 1 << 0,

    /*
     * This indicates that the key blobs returned via all the primitives
     * are sufficient to operate on their own without the trusted OS
     * querying userspace to retrieve some other data. Key blobs of
     * this type are normally returned encrypted with a
     * Key Encryption Key (KEK).
     *
     * This is currently used by "vold" to know whether the whole disk
     * encryption secret can be unwrapped without having some external
     * service started up beforehand since the "/data" partition will
     * be unavailable at that point.
     */
    KEYMASTER_BLOBS_ARE_STANDALONE = 1 << 1,

    /*
     * Indicates that the keymaster module supports DSA keys.
     */
    KEYMASTER_SUPPORTS_DSA = 1 << 2,

    /*
     * Indicates that the keymaster module supports EC keys.
     */
    KEYMASTER_SUPPORTS_EC = 1 << 3,
};

/**
 * Asymmetric key pair types.
 */
typedef enum {
    TYPE_RSA = 1,
    TYPE_DSA = 2,
    TYPE_EC = 3,
} keymaster_keypair_t;

/**
 * Parameters needed to generate an RSA key.
 */
typedef struct {
    uint32_t modulus_size;
    uint64_t public_exponent;
} keymaster_rsa_keygen_params_t;

/**
 * Parameters needed to generate a DSA key.
 */
typedef struct {
    uint32_t key_size;
    uint32_t generator_len;
    uint32_t prime_p_len;
    uint32_t prime_q_len;
    const uint8_t* generator;
    const uint8_t* prime_p;
    const uint8_t* prime_q;
} keymaster_dsa_keygen_params_t;

/**
 * Parameters needed to generate an EC key.
 *
 * Field size is the only parameter in version 2. The sizes correspond to these required curves:
 *
 * 192 = NIST P-192
 * 224 = NIST P-224
 * 256 = NIST P-256
 * 384 = NIST P-384
 * 521 = NIST P-521
 *
 * The parameters for these curves are available at: http://www.nsa.gov/ia/_files/nist-routines.pdf
 * in Chapter 4.
 */
typedef struct {
    uint32_t field_size;
} keymaster_ec_keygen_params_t;


/**
 * Digest type.
 */
typedef enum {
    DIGEST_NONE,
} keymaster_digest_algorithm_t;

/**
 * Type of padding used for RSA operations.
 */
typedef enum {
    PADDING_NONE,
} keymaster_rsa_padding_t;


typedef struct {
    keymaster_digest_algorithm_t digest_type;
} keymaster_dsa_sign_params_t;

typedef struct {
    keymaster_digest_algorithm_t digest_type;
} keymaster_ec_sign_params_t;

typedef struct {
    keymaster_digest_algorithm_t digest_type;
    keymaster_rsa_padding_t padding_type;
} keymaster_rsa_sign_params_t;

__END_DECLS

#endif  // ANDROID_HARDWARE_KEYMASTER_COMMON_H
