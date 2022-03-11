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

#ifndef ANDROID_HARDWARE_KEYMASTER_DEFS_H
#define ANDROID_HARDWARE_KEYMASTER_DEFS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * Authorization tags each have an associated type.  This enumeration facilitates tagging each with
 * a type, by using the high four bits (of an implied 32-bit unsigned enum value) to specify up to
 * 16 data types.  These values are ORed with tag IDs to generate the final tag ID values.
 */
typedef enum {
    KM_INVALID = 0 << 28, /* Invalid type, used to designate a tag as uninitialized */
    KM_ENUM = 1 << 28,
    KM_ENUM_REP = 2 << 28, /* Repeatable enumeration value. */
    KM_UINT = 3 << 28,
    KM_UINT_REP = 4 << 28, /* Repeatable integer value */
    KM_ULONG = 5 << 28,
    KM_DATE = 6 << 28,
    KM_BOOL = 7 << 28,
    KM_BIGNUM = 8 << 28,
    KM_BYTES = 9 << 28,
    KM_ULONG_REP = 10 << 28, /* Repeatable long value */
} keymaster_tag_type_t;

typedef enum {
    KM_TAG_INVALID = KM_INVALID | 0,

    /*
     * Tags that must be semantically enforced by hardware and software implementations.
     */

    /* Crypto parameters */
    KM_TAG_PURPOSE = KM_ENUM_REP | 1,    /* keymaster_purpose_t. */
    KM_TAG_ALGORITHM = KM_ENUM | 2,      /* keymaster_algorithm_t. */
    KM_TAG_KEY_SIZE = KM_UINT | 3,       /* Key size in bits. */
    KM_TAG_BLOCK_MODE = KM_ENUM_REP | 4, /* keymaster_block_mode_t. */
    KM_TAG_DIGEST = KM_ENUM_REP | 5,     /* keymaster_digest_t. */
    KM_TAG_PADDING = KM_ENUM_REP | 6,    /* keymaster_padding_t. */
    KM_TAG_CALLER_NONCE = KM_BOOL | 7,   /* Allow caller to specify nonce or IV. */
    KM_TAG_MIN_MAC_LENGTH = KM_UINT | 8, /* Minimum length of MAC or AEAD authentication tag in
                                          * bits. */
    KM_TAG_KDF = KM_ENUM_REP | 9,        /* keymaster_kdf_t (keymaster2) */
    KM_TAG_EC_CURVE = KM_ENUM | 10,      /* keymaster_ec_curve_t (keymaster2) */

    /* Algorithm-specific. */
    KM_TAG_RSA_PUBLIC_EXPONENT = KM_ULONG | 200,
    KM_TAG_ECIES_SINGLE_HASH_MODE = KM_BOOL | 201, /* Whether the ephemeral public key is fed into
                                                    * the KDF */
    KM_TAG_INCLUDE_UNIQUE_ID = KM_BOOL | 202,      /* If true, attestation certificates for this key
                                                    * will contain an application-scoped and
                                                    * time-bounded device-unique ID. (keymaster2) */

    /* Other hardware-enforced. */
    KM_TAG_BLOB_USAGE_REQUIREMENTS = KM_ENUM | 301, /* keymaster_key_blob_usage_requirements_t */
    KM_TAG_BOOTLOADER_ONLY = KM_BOOL | 302,         /* Usable only by bootloader */

    /*
     * Tags that should be semantically enforced by hardware if possible and will otherwise be
     * enforced by software (keystore).
     */

    /* Key validity period */
    KM_TAG_ACTIVE_DATETIME = KM_DATE | 400,             /* Start of validity */
    KM_TAG_ORIGINATION_EXPIRE_DATETIME = KM_DATE | 401, /* Date when new "messages" should no
                                                           longer be created. */
    KM_TAG_USAGE_EXPIRE_DATETIME = KM_DATE | 402,       /* Date when existing "messages" should no
                                                           longer be trusted. */
    KM_TAG_MIN_SECONDS_BETWEEN_OPS = KM_UINT | 403,     /* Minimum elapsed time between
                                                           cryptographic operations with the key. */
    KM_TAG_MAX_USES_PER_BOOT = KM_UINT | 404,           /* Number of times the key can be used per
                                                           boot. */

    /* User authentication */
    KM_TAG_ALL_USERS = KM_BOOL | 500,           /* Reserved for future use -- ignore */
    KM_TAG_USER_ID = KM_UINT | 501,             /* Reserved for future use -- ignore */
    KM_TAG_USER_SECURE_ID = KM_ULONG_REP | 502, /* Secure ID of authorized user or authenticator(s).
                                                   Disallowed if KM_TAG_ALL_USERS or
                                                   KM_TAG_NO_AUTH_REQUIRED is present. */
    KM_TAG_NO_AUTH_REQUIRED = KM_BOOL | 503,    /* If key is usable without authentication. */
    KM_TAG_USER_AUTH_TYPE = KM_ENUM | 504,      /* Bitmask of authenticator types allowed when
                                                 * KM_TAG_USER_SECURE_ID contains a secure user ID,
                                                 * rather than a secure authenticator ID.  Defined in
                                                 * hw_authenticator_type_t in hw_auth_token.h. */
    KM_TAG_AUTH_TIMEOUT = KM_UINT | 505,        /* Required freshness of user authentication for
                                                   private/secret key operations, in seconds.
                                                   Public key operations require no authentication.
                                                   If absent, authentication is required for every
                                                   use.  Authentication state is lost when the
                                                   device is powered off. */
    KM_TAG_ALLOW_WHILE_ON_BODY = KM_BOOL | 506, /* Allow key to be used after authentication timeout
                                                 * if device is still on-body (requires secure
                                                 * on-body sensor. */
    KM_TAG_UNLOCKED_DEVICE_REQUIRED = KM_BOOL | 508, /* Require the device screen to be unlocked if the
                                                      * key is used. */

    /* Application access control */
    KM_TAG_ALL_APPLICATIONS = KM_BOOL | 600, /* Specified to indicate key is usable by all
                                              * applications. */
    KM_TAG_APPLICATION_ID = KM_BYTES | 601,  /* Byte string identifying the authorized
                                              * application. */
    KM_TAG_EXPORTABLE = KM_BOOL | 602,       /* If true, private/secret key can be exported, but
                                              * only if all access control requirements for use are
                                              * met. (keymaster2) */

    /*
     * Semantically unenforceable tags, either because they have no specific meaning or because
     * they're informational only.
     */
    KM_TAG_APPLICATION_DATA = KM_BYTES | 700,      /* Data provided by authorized application. */
    KM_TAG_CREATION_DATETIME = KM_DATE | 701,      /* Key creation time */
    KM_TAG_ORIGIN = KM_ENUM | 702,                 /* keymaster_key_origin_t. */
    KM_TAG_ROLLBACK_RESISTANT = KM_BOOL | 703,     /* Whether key is rollback-resistant. */
    KM_TAG_ROOT_OF_TRUST = KM_BYTES | 704,         /* Root of trust ID. */
    KM_TAG_OS_VERSION = KM_UINT | 705,             /* Version of system (keymaster2) */
    KM_TAG_OS_PATCHLEVEL = KM_UINT | 706,          /* Patch level of system (keymaster2) */
    KM_TAG_UNIQUE_ID = KM_BYTES | 707,             /* Used to provide unique ID in attestation */
    KM_TAG_ATTESTATION_CHALLENGE = KM_BYTES | 708, /* Used to provide challenge in attestation */
    KM_TAG_ATTESTATION_APPLICATION_ID = KM_BYTES | 709, /* Used to identify the set of possible
                                                         * applications of which one has initiated
                                                         * a key attestation */
    KM_TAG_ATTESTATION_ID_BRAND = KM_BYTES | 710,  /* Used to provide the device's brand name to be
                                                      included in attestation */
    KM_TAG_ATTESTATION_ID_DEVICE = KM_BYTES | 711, /* Used to provide the device's device name to be
                                                      included in attestation */
    KM_TAG_ATTESTATION_ID_PRODUCT = KM_BYTES | 712, /* Used to provide the device's product name to
                                                       be included in attestation */
    KM_TAG_ATTESTATION_ID_SERIAL = KM_BYTES | 713, /* Used to provide the device's serial number to
                                                      be included in attestation */
    KM_TAG_ATTESTATION_ID_IMEI = KM_BYTES | 714,   /* Used to provide the device's IMEI to be
                                                      included in attestation */
    KM_TAG_ATTESTATION_ID_MEID = KM_BYTES | 715,   /* Used to provide the device's MEID to be
                                                      included in attestation */
    KM_TAG_ATTESTATION_ID_MANUFACTURER = KM_BYTES | 716, /* Used to provide the device's
                                                            manufacturer name to be included in
                                                            attestation */
    KM_TAG_ATTESTATION_ID_MODEL = KM_BYTES | 717,  /* Used to provide the device's model name to be
                                                      included in attestation */

    /* Tags used only to provide data to or receive data from operations */
    KM_TAG_ASSOCIATED_DATA = KM_BYTES | 1000, /* Used to provide associated data for AEAD modes. */
    KM_TAG_NONCE = KM_BYTES | 1001,           /* Nonce or Initialization Vector */
    KM_TAG_AUTH_TOKEN = KM_BYTES | 1002,      /* Authentication token that proves secure user
                                                 authentication has been performed.  Structure
                                                 defined in hw_auth_token_t in hw_auth_token.h. */
    KM_TAG_MAC_LENGTH = KM_UINT | 1003,       /* MAC or AEAD authentication tag length in
                                               * bits. */

    KM_TAG_RESET_SINCE_ID_ROTATION = KM_BOOL | 1004, /* Whether the device has beeen factory reset
                                                        since the last unique ID rotation.  Used for
                                                        key attestation. */
} keymaster_tag_t;

/**
 * Algorithms that may be provided by keymaster implementations.  Those that must be provided by all
 * implementations are tagged as "required".
 */
typedef enum {
    /* Asymmetric algorithms. */
    KM_ALGORITHM_RSA = 1,
    // KM_ALGORITHM_DSA = 2, -- Removed, do not re-use value 2.
    KM_ALGORITHM_EC = 3,

    /* Block ciphers algorithms */
    KM_ALGORITHM_AES = 32,
    KM_ALGORITHM_TRIPLE_DES = 33,

    /* MAC algorithms */
    KM_ALGORITHM_HMAC = 128,
} keymaster_algorithm_t;

/**
 * Symmetric block cipher modes provided by keymaster implementations.
 */
typedef enum {
    /* Unauthenticated modes, usable only for encryption/decryption and not generally recommended
     * except for compatibility with existing other protocols. */
    KM_MODE_ECB = 1,
    KM_MODE_CBC = 2,
    KM_MODE_CTR = 3,

    /* Authenticated modes, usable for encryption/decryption and signing/verification.  Recommended
     * over unauthenticated modes for all purposes. */
    KM_MODE_GCM = 32,
} keymaster_block_mode_t;

/**
 * Padding modes that may be applied to plaintext for encryption operations.  This list includes
 * padding modes for both symmetric and asymmetric algorithms.  Note that implementations should not
 * provide all possible combinations of algorithm and padding, only the
 * cryptographically-appropriate pairs.
 */
typedef enum {
    KM_PAD_NONE = 1, /* deprecated */
    KM_PAD_RSA_OAEP = 2,
    KM_PAD_RSA_PSS = 3,
    KM_PAD_RSA_PKCS1_1_5_ENCRYPT = 4,
    KM_PAD_RSA_PKCS1_1_5_SIGN = 5,
    KM_PAD_PKCS7 = 64,
} keymaster_padding_t;

/**
 * Digests provided by keymaster implementations.
 */
typedef enum {
    KM_DIGEST_NONE = 0,
    KM_DIGEST_MD5 = 1, /* Optional, may not be implemented in hardware, will be handled in software
                        * if needed. */
    KM_DIGEST_SHA1 = 2,
    KM_DIGEST_SHA_2_224 = 3,
    KM_DIGEST_SHA_2_256 = 4,
    KM_DIGEST_SHA_2_384 = 5,
    KM_DIGEST_SHA_2_512 = 6,
} keymaster_digest_t;

/*
 * Key derivation functions, mostly used in ECIES.
 */
typedef enum {
    /* Do not apply a key derivation function; use the raw agreed key */
    KM_KDF_NONE = 0,
    /* HKDF defined in RFC 5869 with SHA256 */
    KM_KDF_RFC5869_SHA256 = 1,
    /* KDF1 defined in ISO 18033-2 with SHA1 */
    KM_KDF_ISO18033_2_KDF1_SHA1 = 2,
    /* KDF1 defined in ISO 18033-2 with SHA256 */
    KM_KDF_ISO18033_2_KDF1_SHA256 = 3,
    /* KDF2 defined in ISO 18033-2 with SHA1 */
    KM_KDF_ISO18033_2_KDF2_SHA1 = 4,
    /* KDF2 defined in ISO 18033-2 with SHA256 */
    KM_KDF_ISO18033_2_KDF2_SHA256 = 5,
} keymaster_kdf_t;

/**
 * Supported EC curves, used in ECDSA/ECIES.
 */
typedef enum {
    KM_EC_CURVE_P_224 = 0,
    KM_EC_CURVE_P_256 = 1,
    KM_EC_CURVE_P_384 = 2,
    KM_EC_CURVE_P_521 = 3,
} keymaster_ec_curve_t;

/**
 * The origin of a key (or pair), i.e. where it was generated.  Note that KM_TAG_ORIGIN can be found
 * in either the hardware-enforced or software-enforced list for a key, indicating whether the key
 * is hardware or software-based.  Specifically, a key with KM_ORIGIN_GENERATED in the
 * hardware-enforced list is guaranteed never to have existed outide the secure hardware.
 */
typedef enum {
    KM_ORIGIN_GENERATED = 0, /* Generated in keymaster.  Should not exist outside the TEE. */
    KM_ORIGIN_DERIVED = 1,   /* Derived inside keymaster.  Likely exists off-device. */
    KM_ORIGIN_IMPORTED = 2,  /* Imported into keymaster.  Existed as cleartext in Android. */
    KM_ORIGIN_UNKNOWN = 3,   /* Keymaster did not record origin.  This value can only be seen on
                              * keys in a keymaster0 implementation.  The keymaster0 adapter uses
                              * this value to document the fact that it is unkown whether the key
                              * was generated inside or imported into keymaster. */
} keymaster_key_origin_t;

/**
 * Usability requirements of key blobs.  This defines what system functionality must be available
 * for the key to function.  For example, key "blobs" which are actually handles referencing
 * encrypted key material stored in the file system cannot be used until the file system is
 * available, and should have BLOB_REQUIRES_FILE_SYSTEM.  Other requirements entries will be added
 * as needed for implementations.
 */
typedef enum {
    KM_BLOB_STANDALONE = 0,
    KM_BLOB_REQUIRES_FILE_SYSTEM = 1,
} keymaster_key_blob_usage_requirements_t;

/**
 * Possible purposes of a key (or pair).
 */
typedef enum {
    KM_PURPOSE_ENCRYPT = 0,    /* Usable with RSA, EC and AES keys. */
    KM_PURPOSE_DECRYPT = 1,    /* Usable with RSA, EC and AES keys. */
    KM_PURPOSE_SIGN = 2,       /* Usable with RSA, EC and HMAC keys. */
    KM_PURPOSE_VERIFY = 3,     /* Usable with RSA, EC and HMAC keys. */
    KM_PURPOSE_DERIVE_KEY = 4, /* Usable with EC keys. */
    KM_PURPOSE_WRAP = 5,       /* Usable with wrapped keys. */

} keymaster_purpose_t;

typedef struct {
    const uint8_t* data;
    size_t data_length;
} keymaster_blob_t;

typedef struct {
    keymaster_tag_t tag;
    union {
        uint32_t enumerated;   /* KM_ENUM and KM_ENUM_REP */
        bool boolean;          /* KM_BOOL */
        uint32_t integer;      /* KM_INT and KM_INT_REP */
        uint64_t long_integer; /* KM_LONG */
        uint64_t date_time;    /* KM_DATE */
        keymaster_blob_t blob; /* KM_BIGNUM and KM_BYTES*/
    };
} keymaster_key_param_t;

typedef struct {
    keymaster_key_param_t* params; /* may be NULL if length == 0 */
    size_t length;
} keymaster_key_param_set_t;

/**
 * Parameters that define a key's characteristics, including authorized modes of usage and access
 * control restrictions.  The parameters are divided into two categories, those that are enforced by
 * secure hardware, and those that are not.  For a software-only keymaster implementation the
 * enforced array must NULL.  Hardware implementations must enforce everything in the enforced
 * array.
 */
typedef struct {
    keymaster_key_param_set_t hw_enforced;
    keymaster_key_param_set_t sw_enforced;
} keymaster_key_characteristics_t;

typedef struct {
    const uint8_t* key_material;
    size_t key_material_size;
} keymaster_key_blob_t;

typedef struct {
    keymaster_blob_t* entries;
    size_t entry_count;
} keymaster_cert_chain_t;

typedef enum {
    KM_VERIFIED_BOOT_VERIFIED = 0,    /* Full chain of trust extending from the bootloader to
                                       * verified partitions, including the bootloader, boot
                                       * partition, and all verified partitions*/
    KM_VERIFIED_BOOT_SELF_SIGNED = 1, /* The boot partition has been verified using the embedded
                                       * certificate, and the signature is valid. The bootloader
                                       * displays a warning and the fingerprint of the public
                                       * key before allowing the boot process to continue.*/
    KM_VERIFIED_BOOT_UNVERIFIED = 2,  /* The device may be freely modified. Device integrity is left
                                       * to the user to verify out-of-band. The bootloader
                                       * displays a warning to the user before allowing the boot
                                       * process to continue */
    KM_VERIFIED_BOOT_FAILED = 3,      /* The device failed verification. The bootloader displays a
                                       * warning and stops the boot process, so no keymaster
                                       * implementation should ever actually return this value,
                                       * since it should not run.  Included here only for
                                       * completeness. */
} keymaster_verified_boot_t;

typedef enum {
    KM_SECURITY_LEVEL_SOFTWARE = 0,
    KM_SECURITY_LEVEL_TRUSTED_ENVIRONMENT = 1,
} keymaster_security_level_t;

/**
 * Formats for key import and export.
 */
typedef enum {
    KM_KEY_FORMAT_X509 = 0,  /* for public key export */
    KM_KEY_FORMAT_PKCS8 = 1, /* for asymmetric key pair import */
    KM_KEY_FORMAT_RAW = 3,   /* for symmetric key import and export*/
} keymaster_key_format_t;

/**
 * The keymaster operation API consists of begin, update, finish and abort. This is the type of the
 * handle used to tie the sequence of calls together.  A 64-bit value is used because it's important
 * that handles not be predictable.  Implementations must use strong random numbers for handle
 * values.
 */
typedef uint64_t keymaster_operation_handle_t;

typedef enum {
    KM_ERROR_OK = 0,
    KM_ERROR_ROOT_OF_TRUST_ALREADY_SET = -1,
    KM_ERROR_UNSUPPORTED_PURPOSE = -2,
    KM_ERROR_INCOMPATIBLE_PURPOSE = -3,
    KM_ERROR_UNSUPPORTED_ALGORITHM = -4,
    KM_ERROR_INCOMPATIBLE_ALGORITHM = -5,
    KM_ERROR_UNSUPPORTED_KEY_SIZE = -6,
    KM_ERROR_UNSUPPORTED_BLOCK_MODE = -7,
    KM_ERROR_INCOMPATIBLE_BLOCK_MODE = -8,
    KM_ERROR_UNSUPPORTED_MAC_LENGTH = -9,
    KM_ERROR_UNSUPPORTED_PADDING_MODE = -10,
    KM_ERROR_INCOMPATIBLE_PADDING_MODE = -11,
    KM_ERROR_UNSUPPORTED_DIGEST = -12,
    KM_ERROR_INCOMPATIBLE_DIGEST = -13,
    KM_ERROR_INVALID_EXPIRATION_TIME = -14,
    KM_ERROR_INVALID_USER_ID = -15,
    KM_ERROR_INVALID_AUTHORIZATION_TIMEOUT = -16,
    KM_ERROR_UNSUPPORTED_KEY_FORMAT = -17,
    KM_ERROR_INCOMPATIBLE_KEY_FORMAT = -18,
    KM_ERROR_UNSUPPORTED_KEY_ENCRYPTION_ALGORITHM = -19,   /* For PKCS8 & PKCS12 */
    KM_ERROR_UNSUPPORTED_KEY_VERIFICATION_ALGORITHM = -20, /* For PKCS8 & PKCS12 */
    KM_ERROR_INVALID_INPUT_LENGTH = -21,
    KM_ERROR_KEY_EXPORT_OPTIONS_INVALID = -22,
    KM_ERROR_DELEGATION_NOT_ALLOWED = -23,
    KM_ERROR_KEY_NOT_YET_VALID = -24,
    KM_ERROR_KEY_EXPIRED = -25,
    KM_ERROR_KEY_USER_NOT_AUTHENTICATED = -26,
    KM_ERROR_OUTPUT_PARAMETER_NULL = -27,
    KM_ERROR_INVALID_OPERATION_HANDLE = -28,
    KM_ERROR_INSUFFICIENT_BUFFER_SPACE = -29,
    KM_ERROR_VERIFICATION_FAILED = -30,
    KM_ERROR_TOO_MANY_OPERATIONS = -31,
    KM_ERROR_UNEXPECTED_NULL_POINTER = -32,
    KM_ERROR_INVALID_KEY_BLOB = -33,
    KM_ERROR_IMPORTED_KEY_NOT_ENCRYPTED = -34,
    KM_ERROR_IMPORTED_KEY_DECRYPTION_FAILED = -35,
    KM_ERROR_IMPORTED_KEY_NOT_SIGNED = -36,
    KM_ERROR_IMPORTED_KEY_VERIFICATION_FAILED = -37,
    KM_ERROR_INVALID_ARGUMENT = -38,
    KM_ERROR_UNSUPPORTED_TAG = -39,
    KM_ERROR_INVALID_TAG = -40,
    KM_ERROR_MEMORY_ALLOCATION_FAILED = -41,
    KM_ERROR_IMPORT_PARAMETER_MISMATCH = -44,
    KM_ERROR_SECURE_HW_ACCESS_DENIED = -45,
    KM_ERROR_OPERATION_CANCELLED = -46,
    KM_ERROR_CONCURRENT_ACCESS_CONFLICT = -47,
    KM_ERROR_SECURE_HW_BUSY = -48,
    KM_ERROR_SECURE_HW_COMMUNICATION_FAILED = -49,
    KM_ERROR_UNSUPPORTED_EC_FIELD = -50,
    KM_ERROR_MISSING_NONCE = -51,
    KM_ERROR_INVALID_NONCE = -52,
    KM_ERROR_MISSING_MAC_LENGTH = -53,
    KM_ERROR_KEY_RATE_LIMIT_EXCEEDED = -54,
    KM_ERROR_CALLER_NONCE_PROHIBITED = -55,
    KM_ERROR_KEY_MAX_OPS_EXCEEDED = -56,
    KM_ERROR_INVALID_MAC_LENGTH = -57,
    KM_ERROR_MISSING_MIN_MAC_LENGTH = -58,
    KM_ERROR_UNSUPPORTED_MIN_MAC_LENGTH = -59,
    KM_ERROR_UNSUPPORTED_KDF = -60,
    KM_ERROR_UNSUPPORTED_EC_CURVE = -61,
    KM_ERROR_KEY_REQUIRES_UPGRADE = -62,
    KM_ERROR_ATTESTATION_CHALLENGE_MISSING = -63,
    KM_ERROR_KEYMASTER_NOT_CONFIGURED = -64,
    KM_ERROR_ATTESTATION_APPLICATION_ID_MISSING = -65,
    KM_ERROR_CANNOT_ATTEST_IDS = -66,
    KM_ERROR_DEVICE_LOCKED = -72,

    KM_ERROR_UNIMPLEMENTED = -100,
    KM_ERROR_VERSION_MISMATCH = -101,

    KM_ERROR_UNKNOWN_ERROR = -1000,
} keymaster_error_t;

/* Convenience functions for manipulating keymaster tag types */

static inline keymaster_tag_type_t keymaster_tag_get_type(keymaster_tag_t tag) {
    return (keymaster_tag_type_t)(tag & (0xF << 28));
}

static inline uint32_t keymaster_tag_mask_type(keymaster_tag_t tag) {
    return tag & 0x0FFFFFFF;
}

static inline bool keymaster_tag_type_repeatable(keymaster_tag_type_t type) {
    switch (type) {
    case KM_UINT_REP:
    case KM_ENUM_REP:
        return true;
    default:
        return false;
    }
}

static inline bool keymaster_tag_repeatable(keymaster_tag_t tag) {
    return keymaster_tag_type_repeatable(keymaster_tag_get_type(tag));
}

/* Convenience functions for manipulating keymaster_key_param_t structs */

inline keymaster_key_param_t keymaster_param_enum(keymaster_tag_t tag, uint32_t value) {
    // assert(keymaster_tag_get_type(tag) == KM_ENUM || keymaster_tag_get_type(tag) == KM_ENUM_REP);
    keymaster_key_param_t param;
    memset(&param, 0, sizeof(param));
    param.tag = tag;
    param.enumerated = value;
    return param;
}

inline keymaster_key_param_t keymaster_param_int(keymaster_tag_t tag, uint32_t value) {
    // assert(keymaster_tag_get_type(tag) == KM_INT || keymaster_tag_get_type(tag) == KM_INT_REP);
    keymaster_key_param_t param;
    memset(&param, 0, sizeof(param));
    param.tag = tag;
    param.integer = value;
    return param;
}

inline keymaster_key_param_t keymaster_param_long(keymaster_tag_t tag, uint64_t value) {
    // assert(keymaster_tag_get_type(tag) == KM_LONG);
    keymaster_key_param_t param;
    memset(&param, 0, sizeof(param));
    param.tag = tag;
    param.long_integer = value;
    return param;
}

inline keymaster_key_param_t keymaster_param_blob(keymaster_tag_t tag, const uint8_t* bytes,
                                                  size_t bytes_len) {
    // assert(keymaster_tag_get_type(tag) == KM_BYTES || keymaster_tag_get_type(tag) == KM_BIGNUM);
    keymaster_key_param_t param;
    memset(&param, 0, sizeof(param));
    param.tag = tag;
    param.blob.data = (uint8_t*)bytes;
    param.blob.data_length = bytes_len;
    return param;
}

inline keymaster_key_param_t keymaster_param_bool(keymaster_tag_t tag) {
    // assert(keymaster_tag_get_type(tag) == KM_BOOL);
    keymaster_key_param_t param;
    memset(&param, 0, sizeof(param));
    param.tag = tag;
    param.boolean = true;
    return param;
}

inline keymaster_key_param_t keymaster_param_date(keymaster_tag_t tag, uint64_t value) {
    // assert(keymaster_tag_get_type(tag) == KM_DATE);
    keymaster_key_param_t param;
    memset(&param, 0, sizeof(param));
    param.tag = tag;
    param.date_time = value;
    return param;
}

#define KEYMASTER_SIMPLE_COMPARE(a, b) (a < b) ? -1 : ((a > b) ? 1 : 0)
inline int keymaster_param_compare(const keymaster_key_param_t* a, const keymaster_key_param_t* b) {
    int retval = KEYMASTER_SIMPLE_COMPARE((uint32_t)a->tag, (uint32_t)b->tag);
    if (retval != 0)
        return retval;

    switch (keymaster_tag_get_type(a->tag)) {
    case KM_INVALID:
    case KM_BOOL:
        return 0;
    case KM_ENUM:
    case KM_ENUM_REP:
        return KEYMASTER_SIMPLE_COMPARE(a->enumerated, b->enumerated);
    case KM_UINT:
    case KM_UINT_REP:
        return KEYMASTER_SIMPLE_COMPARE(a->integer, b->integer);
    case KM_ULONG:
    case KM_ULONG_REP:
        return KEYMASTER_SIMPLE_COMPARE(a->long_integer, b->long_integer);
    case KM_DATE:
        return KEYMASTER_SIMPLE_COMPARE(a->date_time, b->date_time);
    case KM_BIGNUM:
    case KM_BYTES:
        // Handle the empty cases.
        if (a->blob.data_length != 0 && b->blob.data_length == 0)
            return -1;
        if (a->blob.data_length == 0 && b->blob.data_length == 0)
            return 0;
        if (a->blob.data_length == 0 && b->blob.data_length > 0)
            return 1;

        retval = memcmp(a->blob.data, b->blob.data, a->blob.data_length < b->blob.data_length
                                                        ? a->blob.data_length
                                                        : b->blob.data_length);
        if (retval != 0)
            return retval;
        else if (a->blob.data_length != b->blob.data_length) {
            // Equal up to the common length; longer one is larger.
            if (a->blob.data_length < b->blob.data_length)
                return -1;
            if (a->blob.data_length > b->blob.data_length)
                return 1;
        };
    }

    return 0;
}
#undef KEYMASTER_SIMPLE_COMPARE

inline void keymaster_free_param_values(keymaster_key_param_t* param, size_t param_count) {
    while (param_count > 0) {
        param_count--;
        switch (keymaster_tag_get_type(param->tag)) {
        case KM_BIGNUM:
        case KM_BYTES:
            free((void*)param->blob.data);
            param->blob.data = NULL;
            break;
        default:
            // NOP
            break;
        }
        ++param;
    }
}

inline void keymaster_free_param_set(keymaster_key_param_set_t* set) {
    if (set) {
        keymaster_free_param_values(set->params, set->length);
        free(set->params);
        set->params = NULL;
        set->length = 0;
    }
}

inline void keymaster_free_characteristics(keymaster_key_characteristics_t* characteristics) {
    if (characteristics) {
        keymaster_free_param_set(&characteristics->hw_enforced);
        keymaster_free_param_set(&characteristics->sw_enforced);
    }
}

inline void keymaster_free_cert_chain(keymaster_cert_chain_t* chain) {
    if (chain) {
        for (size_t i = 0; i < chain->entry_count; ++i) {
            free((uint8_t*)chain->entries[i].data);
            chain->entries[i].data = NULL;
            chain->entries[i].data_length = 0;
        }
        free(chain->entries);
        chain->entries = NULL;
        chain->entry_count = 0;
    }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ANDROID_HARDWARE_KEYMASTER_DEFS_H
