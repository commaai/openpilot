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

#ifndef ANDROID_HARDWARE_KEYMASTER2_H
#define ANDROID_HARDWARE_KEYMASTER2_H

#include <hardware/keymaster_common.h>
#include <hardware/keymaster_defs.h>

__BEGIN_DECLS

/**
 * Keymaster2 device definition
 */
struct keymaster2_device {
    /**
     * Common methods of the keymaster device.  This *must* be the first member of
     * keymaster_device as users of this structure will cast a hw_device_t to
     * keymaster_device pointer in contexts where it's known the hw_device_t references a
     * keymaster_device.
     */
    struct hw_device_t common;

    void* context;

    /**
     * See flags defined for keymaster0_devices::flags in keymaster_common.h.  Used only for
     * backward compatibility; keymaster2 hardware devices must set this to zero.
     */
    uint32_t flags;

    /**
     * Configures keymaster.  This method must be called once after the device is opened and before
     * it is used.  It's used to provide KM_TAG_OS_VERSION and KM_TAG_OS_PATCHLEVEL to keymaster.
     * Until this method is called, all other methods will return KM_ERROR_KEYMASTER_NOT_CONFIGURED.
     * The values provided by this method are only accepted by keymaster once per boot.  Subsequent
     * calls will return KM_ERROR_OK, but do nothing.
     *
     * If the keymaster implementation is in secure hardware and the OS version and patch level
     * values provided do not match the values provided to the secure hardware by the bootloader (or
     * if the bootloader did not provide values), then this method will return
     * KM_ERROR_INVALID_ARGUMENT, and all other methods will continue returning
     * KM_ERROR_KEYMASTER_NOT_CONFIGURED.
     */
    keymaster_error_t (*configure)(const struct keymaster2_device* dev,
                                   const keymaster_key_param_set_t* params);

    /**
     * Adds entropy to the RNG used by keymaster.  Entropy added through this method is guaranteed
     * not to be the only source of entropy used, and the mixing function is required to be secure,
     * in the sense that if the RNG is seeded (from any source) with any data the attacker cannot
     * predict (or control), then the RNG output is indistinguishable from random.  Thus, if the
     * entropy from any source is good, the output will be good.
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] data Random data to be mixed in.
     *
     * \param[in] data_length Length of \p data.
     */
    keymaster_error_t (*add_rng_entropy)(const struct keymaster2_device* dev, const uint8_t* data,
                                         size_t data_length);

    /**
     * Generates a key, or key pair, returning a key blob and/or a description of the key.
     *
     * Key generation parameters are defined as keymaster tag/value pairs, provided in \p params.
     * See keymaster_tag_t for the full list.  Some values that are always required for generation
     * of useful keys are:
     *
     * - KM_TAG_ALGORITHM;
     * - KM_TAG_PURPOSE; and
     * - (KM_TAG_USER_SECURE_ID and KM_TAG_USER_AUTH_TYPE) or KM_TAG_NO_AUTH_REQUIRED.
     *
     * KM_TAG_AUTH_TIMEOUT should generally be specified unless KM_TAG_NO_AUTH_REQUIRED is present,
     * or the user will have to authenticate for every use.
     *
     * KM_TAG_BLOCK_MODE, KM_TAG_PADDING, KM_TAG_MAC_LENGTH and KM_TAG_DIGEST must be specified for
     * algorithms that require them.
     *
     * The following tags may not be specified; their values will be provided by the implementation.
     *
     * - KM_TAG_ORIGIN,
     * - KM_TAG_ROLLBACK_RESISTANT,
     * - KM_TAG_CREATION_DATETIME
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] params Array of key generation param
     *
     * \param[out] key_blob returns the generated key. \p key_blob must not be NULL.  The caller
     * assumes ownership key_blob->key_material and must free() it.
     *
     * \param[out] characteristics returns the characteristics of the key that was, generated, if
     * non-NULL.  If non-NULL, the caller assumes ownership and must deallocate with
     * keymaster_free_characteristics().  Note that KM_TAG_ROOT_OF_TRUST, KM_TAG_APPLICATION_ID and
     * KM_TAG_APPLICATION_DATA are never returned.
     */
    keymaster_error_t (*generate_key)(const struct keymaster2_device* dev,
                                      const keymaster_key_param_set_t* params,
                                      keymaster_key_blob_t* key_blob,
                                      keymaster_key_characteristics_t* characteristics);

    /**
     * Returns the characteristics of the specified key, or KM_ERROR_INVALID_KEY_BLOB if the
     * key_blob is invalid (implementations must fully validate the integrity of the key).
     * client_id and app_data must be the ID and data provided when the key was generated or
     * imported, or empty if KM_TAG_APPLICATION_ID and/or KM_TAG_APPLICATION_DATA were not provided
     * during generation.  Those values are not included in the returned characteristics.  The
     * caller assumes ownership of the allocated characteristics object, which must be deallocated
     * with keymaster_free_characteristics().
     *
     * Note that KM_TAG_APPLICATION_ID and KM_TAG_APPLICATION_DATA are never returned.
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] key_blob The key to retreive characteristics from.
     *
     * \param[in] client_id The client ID data, or NULL if none associated.
     *
     * \param[in] app_id The app data, or NULL if none associated.
     *
     * \param[out] characteristics The key characteristics. Must not be NULL.  The caller assumes
     * ownership of the contents and must deallocate with keymaster_free_characteristics().
     */
    keymaster_error_t (*get_key_characteristics)(const struct keymaster2_device* dev,
                                                 const keymaster_key_blob_t* key_blob,
                                                 const keymaster_blob_t* client_id,
                                                 const keymaster_blob_t* app_data,
                                                 keymaster_key_characteristics_t* characteristics);

    /**
     * Imports a key, or key pair, returning a key blob and/or a description of the key.
     *
     * Most key import parameters are defined as keymaster tag/value pairs, provided in "params".
     * See keymaster_tag_t for the full list.  Values that are always required for import of useful
     * keys are:
     *
     * - KM_TAG_ALGORITHM;
     * - KM_TAG_PURPOSE; and
     * - (KM_TAG_USER_SECURE_ID and KM_TAG_USER_AUTH_TYPE) or KM_TAG_NO_AUTH_REQUIRED.
     *
     * KM_TAG_AUTH_TIMEOUT should generally be specified. If unspecified, the user will have to
     * authenticate for every use.
     *
     * The following tags will take default values if unspecified:
     *
     * - KM_TAG_KEY_SIZE will default to the size of the key provided.
     * - KM_TAG_RSA_PUBLIC_EXPONENT will default to the value in the key provided (for RSA keys)
     *
     * The following tags may not be specified; their values will be provided by the implementation.
     *
     * - KM_TAG_ORIGIN,
     * - KM_TAG_ROLLBACK_RESISTANT,
     * - KM_TAG_CREATION_DATETIME
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] params Parameters defining the imported key.
     *
     * \param[in] params_count The number of entries in \p params.
     *
     * \param[in] key_format specifies the format of the key data in key_data.
     *
     * \param[out] key_blob Used to return the opaque key blob.  Must be non-NULL.  The caller
     * assumes ownership of the contained key_material.
     *
     * \param[out] characteristics Used to return the characteristics of the imported key.  May be
     * NULL, in which case no characteristics will be returned.  If non-NULL, the caller assumes
     * ownership of the contents and must deallocate with keymaster_free_characteristics().  Note
     * that KM_TAG_APPLICATION_ID and KM_TAG_APPLICATION_DATA are never returned.
     */
    keymaster_error_t (*import_key)(const struct keymaster2_device* dev,
                                    const keymaster_key_param_set_t* params,
                                    keymaster_key_format_t key_format,
                                    const keymaster_blob_t* key_data,
                                    keymaster_key_blob_t* key_blob,
                                    keymaster_key_characteristics_t* characteristics);

    /**
     * Exports a public or symmetric key, returning a byte array in the specified format.
     *
     * Note that symmetric key export is allowed only if the key was created with KM_TAG_EXPORTABLE,
     * and only if all of the requirements for key usage (e.g. authentication) are met.
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] export_format The format to be used for exporting the key.
     *
     * \param[in] key_to_export The key to export.
     *
     * \param[in] client_id Client ID blob, which must match the blob provided in
     * KM_TAG_APPLICATION_ID during key generation (if any).
     *
     * \param[in] app_data Appliation data blob, which must match the blob provided in
     * KM_TAG_APPLICATION_DATA during key generation (if any).
     *
     * \param[out] export_data The exported key material.  The caller assumes ownership.
     */
    keymaster_error_t (*export_key)(const struct keymaster2_device* dev,
                                    keymaster_key_format_t export_format,
                                    const keymaster_key_blob_t* key_to_export,
                                    const keymaster_blob_t* client_id,
                                    const keymaster_blob_t* app_data,
                                    keymaster_blob_t* export_data);

    /**
     * Generates a signed X.509 certificate chain attesting to the presence of \p key_to_attest in
     * keymaster (TODO(swillden): Describe certificate contents in more detail).  The certificate
     * will contain an extension with OID 1.3.6.1.4.1.11129.2.1.17 and value defined in
     * <TODO:swillden -- insert link here> which contains the key description.
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] key_to_attest The keymaster key for which the attestation certificate will be
     * generated.
     *
     * \param[in] attest_params Parameters defining how to do the attestation.  At present the only
     * parameter is KM_TAG_ALGORITHM, which must be either KM_ALGORITHM_EC or KM_ALGORITHM_RSA.
     * This selects which of the provisioned attestation keys will be used to sign the certificate.
     *
     * \param[out] cert_chain An array of DER-encoded X.509 certificates. The first will be the
     * certificate for \p key_to_attest.  The remaining entries will chain back to the root.  The
     * caller takes ownership and must deallocate with keymaster_free_cert_chain.
     */
    keymaster_error_t (*attest_key)(const struct keymaster2_device* dev,
                                    const keymaster_key_blob_t* key_to_attest,
                                    const keymaster_key_param_set_t* attest_params,
                                    keymaster_cert_chain_t* cert_chain);

    /**
     * Upgrades an old key.  Keys can become "old" in two ways: Keymaster can be upgraded to a new
     * version, or the system can be updated to invalidate the OS version and/or patch level.  In
     * either case, attempts to use an old key will result in keymaster returning
     * KM_ERROR_KEY_REQUIRES_UPGRADE.  This method should then be called to upgrade the key.
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] key_to_upgrade The keymaster key to upgrade.
     *
     * \param[in] upgrade_params Parameters needed to complete the upgrade. In particular,
     * KM_TAG_APPLICATION_ID and KM_TAG_APPLICATION_DATA will be required if they were defined for
     * the key.
     *
     * \param[out] upgraded_key The upgraded key blob.
     */
    keymaster_error_t (*upgrade_key)(const struct keymaster2_device* dev,
                                     const keymaster_key_blob_t* key_to_upgrade,
                                     const keymaster_key_param_set_t* upgrade_params,
                                     keymaster_key_blob_t* upgraded_key);

    /**
     * Deletes the key, or key pair, associated with the key blob.  After calling this function it
     * will be impossible to use the key for any other operations.  May be applied to keys from
     * foreign roots of trust (keys not usable under the current root of trust).
     *
     * This function is optional and should be set to NULL if it is not implemented.
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] key The key to be deleted.
     */
    keymaster_error_t (*delete_key)(const struct keymaster2_device* dev,
                                    const keymaster_key_blob_t* key);

    /**
     * Deletes all keys in the hardware keystore. Used when keystore is reset completely.  After
     * calling this function it will be impossible to use any previously generated or imported key
     * blobs for any operations.
     *
     * This function is optional and should be set to NULL if it is not implemented.
     *
     * \param[in] dev The keymaster device structure.
     */
    keymaster_error_t (*delete_all_keys)(const struct keymaster2_device* dev);

    /**
     * Begins a cryptographic operation using the specified key.  If all is well, begin() will
     * return KM_ERROR_OK and create an operation handle which must be passed to subsequent calls to
     * update(), finish() or abort().
     *
     * It is critical that each call to begin() be paired with a subsequent call to finish() or
     * abort(), to allow the keymaster implementation to clean up any internal operation state.
     * Failure to do this may leak internal state space or other internal resources and may
     * eventually cause begin() to return KM_ERROR_TOO_MANY_OPERATIONS when it runs out of space for
     * operations.  Any result other than KM_ERROR_OK from begin(), update() or finish() implicitly
     * aborts the operation, in which case abort() need not be called (and will return
     * KM_ERROR_INVALID_OPERATION_HANDLE if called).
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] purpose The purpose of the operation, one of KM_PURPOSE_ENCRYPT,
     * KM_PURPOSE_DECRYPT, KM_PURPOSE_SIGN or KM_PURPOSE_VERIFY. Note that for AEAD modes,
     * encryption and decryption imply signing and verification, respectively, but should be
     * specified as KM_PURPOSE_ENCRYPT and KM_PURPOSE_DECRYPT.
     *
     * \param[in] key The key to be used for the operation. \p key must have a purpose compatible
     * with \p purpose and all of its usage requirements must be satisfied, or begin() will return
     * an appropriate error code.
     *
     * \param[in] in_params Additional parameters for the operation.  This is typically used to
     * provide authentication data, with KM_TAG_AUTH_TOKEN.  If KM_TAG_APPLICATION_ID or
     * KM_TAG_APPLICATION_DATA were provided during generation, they must be provided here, or the
     * operation will fail with KM_ERROR_INVALID_KEY_BLOB.  For operations that require a nonce or
     * IV, on keys that were generated with KM_TAG_CALLER_NONCE, in_params may contain a tag
     * KM_TAG_NONCE.
     *
     * \param[out] out_params Output parameters.  Used to return additional data from the operation
     * initialization, notably to return the IV or nonce from operations that generate an IV or
     * nonce.  The caller takes ownership of the output parameters array and must free it with
     * keymaster_free_param_set().  out_params may be set to NULL if no output parameters are
     * expected.  If out_params is NULL, and output paramaters are generated, begin() will return
     * KM_ERROR_OUTPUT_PARAMETER_NULL.
     *
     * \param[out] operation_handle The newly-created operation handle which must be passed to
     * update(), finish() or abort().  If operation_handle is NULL, begin() will return
     * KM_ERROR_OUTPUT_PARAMETER_NULL.
     */
    keymaster_error_t (*begin)(const struct keymaster2_device* dev, keymaster_purpose_t purpose,
                               const keymaster_key_blob_t* key,
                               const keymaster_key_param_set_t* in_params,
                               keymaster_key_param_set_t* out_params,
                               keymaster_operation_handle_t* operation_handle);

    /**
     * Provides data to, and possibly receives output from, an ongoing cryptographic operation begun
     * with begin().
     *
     * If operation_handle is invalid, update() will return KM_ERROR_INVALID_OPERATION_HANDLE.
     *
     * update() may not consume all of the data provided in the data buffer.  update() will return
     * the amount consumed in *data_consumed.  The caller should provide the unconsumed data in a
     * subsequent call.
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] operation_handle The operation handle returned by begin().
     *
     * \param[in] in_params Additional parameters for the operation.  For AEAD modes, this is used
     * to specify KM_TAG_ADDITIONAL_DATA.  Note that additional data may be provided in multiple
     * calls to update(), but only until input data has been provided.
     *
     * \param[in] input Data to be processed, per the parameters established in the call to begin().
     * Note that update() may or may not consume all of the data provided.  See \p input_consumed.
     *
     * \param[out] input_consumed Amount of data that was consumed by update().  If this is less
     * than the amount provided, the caller should provide the remainder in a subsequent call to
     * update().
     *
     * \param[out] out_params Output parameters.  Used to return additional data from the operation
     * The caller takes ownership of the output parameters array and must free it with
     * keymaster_free_param_set().  out_params may be set to NULL if no output parameters are
     * expected.  If out_params is NULL, and output paramaters are generated, begin() will return
     * KM_ERROR_OUTPUT_PARAMETER_NULL.
     *
     * \param[out] output The output data, if any.  The caller assumes ownership of the allocated
     * buffer.  output must not be NULL.
     *
     * Note that update() may not provide any output, in which case output->data_length will be
     * zero, and output->data may be either NULL or zero-length (so the caller should always free()
     * it).
     */
    keymaster_error_t (*update)(const struct keymaster2_device* dev,
                                keymaster_operation_handle_t operation_handle,
                                const keymaster_key_param_set_t* in_params,
                                const keymaster_blob_t* input, size_t* input_consumed,
                                keymaster_key_param_set_t* out_params, keymaster_blob_t* output);

    /**
     * Finalizes a cryptographic operation begun with begin() and invalidates \p operation_handle.
     *
     * \param[in] dev The keymaster device structure.
     *
     * \param[in] operation_handle The operation handle returned by begin().  This handle will be
     * invalidated.
     *
     * \param[in] in_params Additional parameters for the operation.  For AEAD modes, this is used
     * to specify KM_TAG_ADDITIONAL_DATA, but only if no input data was provided to update().
     *
     * \param[in] input Data to be processed, per the parameters established in the call to
     * begin(). finish() must consume all provided data or return KM_ERROR_INVALID_INPUT_LENGTH.
     *
     * \param[in] signature The signature to be verified if the purpose specified in the begin()
     * call was KM_PURPOSE_VERIFY.
     *
     * \param[out] output The output data, if any.  The caller assumes ownership of the allocated
     * buffer.
     *
     * If the operation being finished is a signature verification or an AEAD-mode decryption and
     * verification fails then finish() will return KM_ERROR_VERIFICATION_FAILED.
     */
    keymaster_error_t (*finish)(const struct keymaster2_device* dev,
                                keymaster_operation_handle_t operation_handle,
                                const keymaster_key_param_set_t* in_params,
                                const keymaster_blob_t* input, const keymaster_blob_t* signature,
                                keymaster_key_param_set_t* out_params, keymaster_blob_t* output);

    /**
     * Aborts a cryptographic operation begun with begin(), freeing all internal resources and
     * invalidating \p operation_handle.
     */
    keymaster_error_t (*abort)(const struct keymaster2_device* dev,
                               keymaster_operation_handle_t operation_handle);
};
typedef struct keymaster2_device keymaster2_device_t;

/* Convenience API for opening and closing keymaster devices */

static inline int keymaster2_open(const struct hw_module_t* module, keymaster2_device_t** device) {
    return module->methods->open(module, KEYSTORE_KEYMASTER, TO_HW_DEVICE_T_OPEN(device));
}

static inline int keymaster2_close(keymaster2_device_t* device) {
    return device->common.close(&device->common);
}

__END_DECLS

#endif  // ANDROID_HARDWARE_KEYMASTER2_H
