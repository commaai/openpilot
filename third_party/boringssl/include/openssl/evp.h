/* Copyright (C) 1995-1998 Eric Young (eay@cryptsoft.com)
 * All rights reserved.
 *
 * This package is an SSL implementation written
 * by Eric Young (eay@cryptsoft.com).
 * The implementation was written so as to conform with Netscapes SSL.
 *
 * This library is free for commercial and non-commercial use as long as
 * the following conditions are aheared to.  The following conditions
 * apply to all code found in this distribution, be it the RC4, RSA,
 * lhash, DES, etc., code; not just the SSL code.  The SSL documentation
 * included with this distribution is covered by the same copyright terms
 * except that the holder is Tim Hudson (tjh@cryptsoft.com).
 *
 * Copyright remains Eric Young's, and as such any Copyright notices in
 * the code are not to be removed.
 * If this package is used in a product, Eric Young should be given attribution
 * as the author of the parts of the library used.
 * This can be in the form of a textual message at program startup or
 * in documentation (online or textual) provided with the package.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    "This product includes cryptographic software written by
 *     Eric Young (eay@cryptsoft.com)"
 *    The word 'cryptographic' can be left out if the rouines from the library
 *    being used are not cryptographic related :-).
 * 4. If you include any Windows specific code (or a derivative thereof) from
 *    the apps directory (application code) you must include an acknowledgement:
 *    "This product includes software written by Tim Hudson (tjh@cryptsoft.com)"
 *
 * THIS SOFTWARE IS PROVIDED BY ERIC YOUNG ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * The licence and distribution terms for any publically available version or
 * derivative of this code cannot be changed.  i.e. this code cannot simply be
 * copied and put under another distribution licence
 * [including the GNU Public Licence.] */

#ifndef OPENSSL_HEADER_EVP_H
#define OPENSSL_HEADER_EVP_H

#include <openssl/base.h>

#include <openssl/thread.h>

/* OpenSSL included digest and cipher functions in this header so we include
 * them for users that still expect that.
 *
 * TODO(fork): clean up callers so that they include what they use. */
#include <openssl/aead.h>
#include <openssl/cipher.h>
#include <openssl/digest.h>
#include <openssl/obj.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* EVP abstracts over public/private key algorithms. */


/* Public key objects. */

/* EVP_PKEY_new creates a new, empty public-key object and returns it or NULL
 * on allocation failure. */
OPENSSL_EXPORT EVP_PKEY *EVP_PKEY_new(void);

/* EVP_PKEY_free frees all data referenced by |pkey| and then frees |pkey|
 * itself. */
OPENSSL_EXPORT void EVP_PKEY_free(EVP_PKEY *pkey);

/* EVP_PKEY_up_ref increments the reference count of |pkey| and returns it. */
OPENSSL_EXPORT EVP_PKEY *EVP_PKEY_up_ref(EVP_PKEY *pkey);

/* EVP_PKEY_is_opaque returns one if |pkey| is opaque. Opaque keys are backed by
 * custom implementations which do not expose key material and parameters. It is
 * an error to attempt to duplicate, export, or compare an opaque key. */
OPENSSL_EXPORT int EVP_PKEY_is_opaque(const EVP_PKEY *pkey);

/* EVP_PKEY_supports_digest returns one if |pkey| supports digests of
 * type |md|. This is intended for use with EVP_PKEYs backing custom
 * implementations which can't sign all digests. */
OPENSSL_EXPORT int EVP_PKEY_supports_digest(const EVP_PKEY *pkey,
                                            const EVP_MD *md);

/* EVP_PKEY_cmp compares |a| and |b| and returns one if they are equal, zero if
 * not and a negative number on error.
 *
 * WARNING: this differs from the traditional return value of a "cmp"
 * function. */
OPENSSL_EXPORT int EVP_PKEY_cmp(const EVP_PKEY *a, const EVP_PKEY *b);

/* EVP_PKEY_copy_parameters sets the parameters of |to| to equal the parameters
 * of |from|. It returns one on success and zero on error. */
OPENSSL_EXPORT int EVP_PKEY_copy_parameters(EVP_PKEY *to, const EVP_PKEY *from);

/* EVP_PKEY_missing_parameters returns one if |pkey| is missing needed
 * parameters or zero if not, or if the algorithm doesn't take parameters. */
OPENSSL_EXPORT int EVP_PKEY_missing_parameters(const EVP_PKEY *pkey);

/* EVP_PKEY_size returns the maximum size, in bytes, of a signature signed by
 * |pkey|. For an RSA key, this returns the number of bytes needed to represent
 * the modulus. For an EC key, this returns the maximum size of a DER-encoded
 * ECDSA signature. */
OPENSSL_EXPORT int EVP_PKEY_size(const EVP_PKEY *pkey);

/* EVP_PKEY_bits returns the "size", in bits, of |pkey|. For an RSA key, this
 * returns the bit length of the modulus. For an EC key, this returns the bit
 * length of the group order. */
OPENSSL_EXPORT int EVP_PKEY_bits(EVP_PKEY *pkey);

/* EVP_PKEY_id returns the type of |pkey|, which is one of the |EVP_PKEY_*|
 * values. */
OPENSSL_EXPORT int EVP_PKEY_id(const EVP_PKEY *pkey);

/* EVP_PKEY_type returns a canonicalised form of |NID|. For example,
 * |EVP_PKEY_RSA2| will be turned into |EVP_PKEY_RSA|. */
OPENSSL_EXPORT int EVP_PKEY_type(int nid);

/* Deprecated: EVP_PKEY_new_mac_key allocates a fresh |EVP_PKEY| of the given
 * type (e.g. |EVP_PKEY_HMAC|), sets |mac_key| as the MAC key and "generates" a
 * new key, suitable for signing. It returns the fresh |EVP_PKEY|, or NULL on
 * error. Use |HMAC_CTX| directly instead. */
OPENSSL_EXPORT EVP_PKEY *EVP_PKEY_new_mac_key(int type, ENGINE *engine,
                                              const uint8_t *mac_key,
                                              size_t mac_key_len);


/* Getting and setting concrete public key types.
 *
 * The following functions get and set the underlying public key in an
 * |EVP_PKEY| object. The |set1| functions take an additional reference to the
 * underlying key and return one on success or zero on error. The |assign|
 * functions adopt the caller's reference. The getters return a fresh reference
 * to the underlying object. */

OPENSSL_EXPORT int EVP_PKEY_set1_RSA(EVP_PKEY *pkey, RSA *key);
OPENSSL_EXPORT int EVP_PKEY_assign_RSA(EVP_PKEY *pkey, RSA *key);
OPENSSL_EXPORT RSA *EVP_PKEY_get1_RSA(EVP_PKEY *pkey);

OPENSSL_EXPORT int EVP_PKEY_set1_DSA(EVP_PKEY *pkey, struct dsa_st *key);
OPENSSL_EXPORT int EVP_PKEY_assign_DSA(EVP_PKEY *pkey, DSA *key);
OPENSSL_EXPORT struct dsa_st *EVP_PKEY_get1_DSA(EVP_PKEY *pkey);

OPENSSL_EXPORT int EVP_PKEY_set1_EC_KEY(EVP_PKEY *pkey, struct ec_key_st *key);
OPENSSL_EXPORT int EVP_PKEY_assign_EC_KEY(EVP_PKEY *pkey, EC_KEY *key);
OPENSSL_EXPORT struct ec_key_st *EVP_PKEY_get1_EC_KEY(EVP_PKEY *pkey);

OPENSSL_EXPORT int EVP_PKEY_set1_DH(EVP_PKEY *pkey, struct dh_st *key);
OPENSSL_EXPORT int EVP_PKEY_assign_DH(EVP_PKEY *pkey, DH *key);
OPENSSL_EXPORT struct dh_st *EVP_PKEY_get1_DH(EVP_PKEY *pkey);

#define EVP_PKEY_NONE NID_undef
#define EVP_PKEY_RSA NID_rsaEncryption
#define EVP_PKEY_RSA2 NID_rsa
#define EVP_PKEY_DSA NID_dsa
#define EVP_PKEY_DH NID_dhKeyAgreement
#define EVP_PKEY_DHX NID_dhpublicnumber
#define EVP_PKEY_EC NID_X9_62_id_ecPublicKey

/* Deprecated: Use |HMAC_CTX| directly instead. */
#define EVP_PKEY_HMAC NID_hmac

/* EVP_PKEY_assign sets the underlying key of |pkey| to |key|, which must be of
 * the given type. The |type| argument should be one of the |EVP_PKEY_*|
 * values. */
OPENSSL_EXPORT int EVP_PKEY_assign(EVP_PKEY *pkey, int type, void *key);

/* EVP_PKEY_set_type sets the type of |pkey| to |type|, which should be one of
 * the |EVP_PKEY_*| values. It returns one if sucessful or zero otherwise. If
 * |pkey| is NULL, it simply reports whether the type is known. */
OPENSSL_EXPORT int EVP_PKEY_set_type(EVP_PKEY *pkey, int type);

/* EVP_PKEY_cmp_parameters compares the parameters of |a| and |b|. It returns
 * one if they match, zero if not, or a negative number of on error.
 *
 * WARNING: the return value differs from the usual return value convention. */
OPENSSL_EXPORT int EVP_PKEY_cmp_parameters(const EVP_PKEY *a,
                                           const EVP_PKEY *b);


/* ASN.1 functions */

/* d2i_PrivateKey parses an ASN.1, DER-encoded, private key from |len| bytes at
 * |*inp|. If |out| is not NULL then, on exit, a pointer to the result is in
 * |*out|. If |*out| is already non-NULL on entry then the result is written
 * directly into |*out|, otherwise a fresh |EVP_PKEY| is allocated. On
 * successful exit, |*inp| is advanced past the DER structure. It returns the
 * result or NULL on error. */
OPENSSL_EXPORT EVP_PKEY *d2i_PrivateKey(int type, EVP_PKEY **out,
                                        const uint8_t **inp, long len);

/* d2i_AutoPrivateKey acts the same as |d2i_PrivateKey|, but detects the type
 * of the private key. */
OPENSSL_EXPORT EVP_PKEY *d2i_AutoPrivateKey(EVP_PKEY **out, const uint8_t **inp,
                                            long len);

/* i2d_PrivateKey marshals a private key from |key| to an ASN.1, DER
 * structure. If |outp| is not NULL then the result is written to |*outp| and
 * |*outp| is advanced just past the output. It returns the number of bytes in
 * the result, whether written or not, or a negative value on error. */
OPENSSL_EXPORT int i2d_PrivateKey(const EVP_PKEY *key, uint8_t **outp);

/* i2d_PublicKey marshals a public key from |key| to an ASN.1, DER
 * structure. If |outp| is not NULL then the result is written to |*outp| and
 * |*outp| is advanced just past the output. It returns the number of bytes in
 * the result, whether written or not, or a negative value on error. */
OPENSSL_EXPORT int i2d_PublicKey(EVP_PKEY *key, uint8_t **outp);


/* Signing */

/* EVP_DigestSignInit sets up |ctx| for a signing operation with |type| and
 * |pkey|. The |ctx| argument must have been initialised with
 * |EVP_MD_CTX_init|. If |pctx| is not NULL, the |EVP_PKEY_CTX| of the signing
 * operation will be written to |*pctx|; this can be used to set alternative
 * signing options.
 *
 * It returns one on success, or zero on error. */
OPENSSL_EXPORT int EVP_DigestSignInit(EVP_MD_CTX *ctx, EVP_PKEY_CTX **pctx,
                                      const EVP_MD *type, ENGINE *e,
                                      EVP_PKEY *pkey);

/* EVP_DigestSignUpdate appends |len| bytes from |data| to the data which will
 * be signed in |EVP_DigestSignFinal|. It returns one. */
OPENSSL_EXPORT int EVP_DigestSignUpdate(EVP_MD_CTX *ctx, const void *data,
                                        size_t len);

/* EVP_DigestSignFinal signs the data that has been included by one or more
 * calls to |EVP_DigestSignUpdate|. If |out_sig| is NULL then |*out_sig_len| is
 * set to the maximum number of output bytes. Otherwise, on entry,
 * |*out_sig_len| must contain the length of the |out_sig| buffer. If the call
 * is successful, the signature is written to |out_sig| and |*out_sig_len| is
 * set to its length.
 *
 * It returns one on success, or zero on error. */
OPENSSL_EXPORT int EVP_DigestSignFinal(EVP_MD_CTX *ctx, uint8_t *out_sig,
                                       size_t *out_sig_len);

/* EVP_DigestSignAlgorithm encodes the signing parameters of |ctx| as an
 * AlgorithmIdentifer and saves the result in |algor|.
 *
 * It returns one on success, or zero on error.
 *
 * TODO(davidben): This API should eventually lose the dependency on
 * crypto/asn1/. */
OPENSSL_EXPORT int EVP_DigestSignAlgorithm(EVP_MD_CTX *ctx, X509_ALGOR *algor);


/* Verifying */

/* EVP_DigestVerifyInit sets up |ctx| for a signature verification operation
 * with |type| and |pkey|. The |ctx| argument must have been initialised with
 * |EVP_MD_CTX_init|. If |pctx| is not NULL, the |EVP_PKEY_CTX| of the signing
 * operation will be written to |*pctx|; this can be used to set alternative
 * signing options.
 *
 * It returns one on success, or zero on error. */
OPENSSL_EXPORT int EVP_DigestVerifyInit(EVP_MD_CTX *ctx, EVP_PKEY_CTX **pctx,
                                        const EVP_MD *type, ENGINE *e,
                                        EVP_PKEY *pkey);

/* EVP_DigestVerifyInitFromAlgorithm sets up |ctx| for a signature verification
 * operation with public key |pkey| and parameters from |algor|. The |ctx|
 * argument must have been initialised with |EVP_MD_CTX_init|.
 *
 * It returns one on success, or zero on error.
 *
 * TODO(davidben): This API should eventually lose the dependency on
 * crypto/asn1/. */
OPENSSL_EXPORT int EVP_DigestVerifyInitFromAlgorithm(EVP_MD_CTX *ctx,
                                                     X509_ALGOR *algor,
                                                     EVP_PKEY *pkey);

/* EVP_DigestVerifyUpdate appends |len| bytes from |data| to the data which
 * will be verified by |EVP_DigestVerifyFinal|. It returns one. */
OPENSSL_EXPORT int EVP_DigestVerifyUpdate(EVP_MD_CTX *ctx, const void *data,
                                          size_t len);

/* EVP_DigestVerifyFinal verifies that |sig_len| bytes of |sig| are a valid
 * signature for the data that has been included by one or more calls to
 * |EVP_DigestVerifyUpdate|. It returns one on success and zero otherwise. */
OPENSSL_EXPORT int EVP_DigestVerifyFinal(EVP_MD_CTX *ctx, const uint8_t *sig,
                                         size_t sig_len);


/* Signing (old functions) */

/* EVP_SignInit_ex configures |ctx|, which must already have been initialised,
 * for a fresh signing operation using the hash function |type|. It returns one
 * on success and zero otherwise.
 *
 * (In order to initialise |ctx|, either obtain it initialised with
 * |EVP_MD_CTX_create|, or use |EVP_MD_CTX_init|.) */
OPENSSL_EXPORT int EVP_SignInit_ex(EVP_MD_CTX *ctx, const EVP_MD *type,
                                   ENGINE *impl);

/* EVP_SignInit is a deprecated version of |EVP_SignInit_ex|.
 *
 * TODO(fork): remove. */
OPENSSL_EXPORT int EVP_SignInit(EVP_MD_CTX *ctx, const EVP_MD *type);

/* EVP_SignUpdate appends |len| bytes from |data| to the data which will be
 * signed in |EVP_SignFinal|. */
OPENSSL_EXPORT int EVP_SignUpdate(EVP_MD_CTX *ctx, const void *data,
                                  size_t len);

/* EVP_SignFinal signs the data that has been included by one or more calls to
 * |EVP_SignUpdate|, using the key |pkey|, and writes it to |sig|. On entry,
 * |sig| must point to at least |EVP_PKEY_size(pkey)| bytes of space. The
 * actual size of the signature is written to |*out_sig_len|.
 *
 * It returns one on success and zero otherwise.
 *
 * It does not modify |ctx|, thus it's possible to continue to use |ctx| in
 * order to sign a longer message. */
OPENSSL_EXPORT int EVP_SignFinal(const EVP_MD_CTX *ctx, uint8_t *sig,
                                 unsigned int *out_sig_len, EVP_PKEY *pkey);


/* Verifying (old functions) */

/* EVP_VerifyInit_ex configures |ctx|, which must already have been
 * initialised, for a fresh signature verification operation using the hash
 * function |type|. It returns one on success and zero otherwise.
 *
 * (In order to initialise |ctx|, either obtain it initialised with
 * |EVP_MD_CTX_create|, or use |EVP_MD_CTX_init|.) */
OPENSSL_EXPORT int EVP_VerifyInit_ex(EVP_MD_CTX *ctx, const EVP_MD *type,
                                     ENGINE *impl);

/* EVP_VerifyInit is a deprecated version of |EVP_VerifyInit_ex|.
 *
 * TODO(fork): remove. */
OPENSSL_EXPORT int EVP_VerifyInit(EVP_MD_CTX *ctx, const EVP_MD *type);

/* EVP_VerifyUpdate appends |len| bytes from |data| to the data which will be
 * signed in |EVP_VerifyFinal|. */
OPENSSL_EXPORT int EVP_VerifyUpdate(EVP_MD_CTX *ctx, const void *data,
                                    size_t len);

/* EVP_VerifyFinal verifies that |sig_len| bytes of |sig| are a valid
 * signature, by |pkey|, for the data that has been included by one or more
 * calls to |EVP_VerifyUpdate|.
 *
 * It returns one on success and zero otherwise.
 *
 * It does not modify |ctx|, thus it's possible to continue to use |ctx| in
 * order to sign a longer message. */
OPENSSL_EXPORT int EVP_VerifyFinal(EVP_MD_CTX *ctx, const uint8_t *sig,
                                   size_t sig_len, EVP_PKEY *pkey);


/* Printing */

/* EVP_PKEY_print_public prints a textual representation of the public key in
 * |pkey| to |out|. Returns one on success or zero otherwise. */
OPENSSL_EXPORT int EVP_PKEY_print_public(BIO *out, const EVP_PKEY *pkey,
                                         int indent, ASN1_PCTX *pctx);

/* EVP_PKEY_print_public prints a textual representation of the private key in
 * |pkey| to |out|. Returns one on success or zero otherwise. */
OPENSSL_EXPORT int EVP_PKEY_print_private(BIO *out, const EVP_PKEY *pkey,
                                          int indent, ASN1_PCTX *pctx);

/* EVP_PKEY_print_public prints a textual representation of the parameters in
 * |pkey| to |out|. Returns one on success or zero otherwise. */
OPENSSL_EXPORT int EVP_PKEY_print_params(BIO *out, const EVP_PKEY *pkey,
                                         int indent, ASN1_PCTX *pctx);


/* Password stretching.
 *
 * Password stretching functions take a low-entropy password and apply a slow
 * function that results in a key suitable for use in symmetric
 * cryptography. */

/* PKCS5_PBKDF2_HMAC computes |iterations| iterations of PBKDF2 of |password|
 * and |salt|, using |digest|, and outputs |key_len| bytes to |out_key|. It
 * returns one on success and zero on error. */
OPENSSL_EXPORT int PKCS5_PBKDF2_HMAC(const char *password, size_t password_len,
                                     const uint8_t *salt, size_t salt_len,
                                     unsigned iterations, const EVP_MD *digest,
                                     size_t key_len, uint8_t *out_key);

/* PKCS5_PBKDF2_HMAC_SHA1 is the same as PKCS5_PBKDF2_HMAC, but with |digest|
 * fixed to |EVP_sha1|. */
OPENSSL_EXPORT int PKCS5_PBKDF2_HMAC_SHA1(const char *password,
                                          size_t password_len, const uint8_t *salt,
                                          size_t salt_len, unsigned iterations,
                                          size_t key_len, uint8_t *out_key);


/* Public key contexts.
 *
 * |EVP_PKEY_CTX| objects hold the context of an operation (e.g. signing or
 * encrypting) that uses a public key. */

/* EVP_PKEY_CTX_new allocates a fresh |EVP_PKEY_CTX| for use with |pkey|. It
 * returns the context or NULL on error. */
OPENSSL_EXPORT EVP_PKEY_CTX *EVP_PKEY_CTX_new(EVP_PKEY *pkey, ENGINE *e);

/* EVP_PKEY_CTX_new allocates a fresh |EVP_PKEY_CTX| for a key of type |id|
 * (e.g. |EVP_PKEY_HMAC|). This can be used for key generation where
 * |EVP_PKEY_CTX_new| can't be used because there isn't an |EVP_PKEY| to pass
 * it. It returns the context or NULL on error. */
OPENSSL_EXPORT EVP_PKEY_CTX *EVP_PKEY_CTX_new_id(int id, ENGINE *e);

/* EVP_KEY_CTX_free frees |ctx| and the data it owns. */
OPENSSL_EXPORT void EVP_PKEY_CTX_free(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_CTX_dup allocates a fresh |EVP_PKEY_CTX| and sets it equal to the
 * state of |ctx|. It returns the fresh |EVP_PKEY_CTX| or NULL on error. */
OPENSSL_EXPORT EVP_PKEY_CTX *EVP_PKEY_CTX_dup(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_CTX_get0_pkey returns the |EVP_PKEY| associated with |ctx|. */
OPENSSL_EXPORT EVP_PKEY *EVP_PKEY_CTX_get0_pkey(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_CTX_set_app_data sets an opaque pointer on |ctx|. */
OPENSSL_EXPORT void EVP_PKEY_CTX_set_app_data(EVP_PKEY_CTX *ctx, void *data);

/* EVP_PKEY_CTX_get_app_data returns the opaque pointer from |ctx| that was
 * previously set with |EVP_PKEY_CTX_set_app_data|, or NULL if none has been
 * set. */
OPENSSL_EXPORT void *EVP_PKEY_CTX_get_app_data(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_sign_init initialises an |EVP_PKEY_CTX| for a signing operation. It
 * should be called before |EVP_PKEY_sign|.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_sign_init(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_sign signs |data_len| bytes from |data| using |ctx|. If |sig| is
 * NULL, the maximum size of the signature is written to
 * |out_sig_len|. Otherwise, |*sig_len| must contain the number of bytes of
 * space available at |sig|. If sufficient, the signature will be written to
 * |sig| and |*sig_len| updated with the true length.
 *
 * WARNING: Setting |sig| to NULL only gives the maximum size of the
 * signature. The actual signature may be smaller.
 *
 * It returns one on success or zero on error. (Note: this differs from
 * OpenSSL, which can also return negative values to indicate an error. ) */
OPENSSL_EXPORT int EVP_PKEY_sign(EVP_PKEY_CTX *ctx, uint8_t *sig,
                                 size_t *sig_len, const uint8_t *data,
                                 size_t data_len);

/* EVP_PKEY_verify_init initialises an |EVP_PKEY_CTX| for a signature
 * verification operation. It should be called before |EVP_PKEY_verify|.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_verify_init(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_verify verifies that |sig_len| bytes from |sig| are a valid signature
 * for |data|.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_verify(EVP_PKEY_CTX *ctx, const uint8_t *sig,
                                   size_t sig_len, const uint8_t *data,
                                   size_t data_len);

/* EVP_PKEY_encrypt_init initialises an |EVP_PKEY_CTX| for an encryption
 * operation. It should be called before |EVP_PKEY_encrypt|.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_encrypt_init(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_encrypt encrypts |in_len| bytes from |in|. If |out| is NULL, the
 * maximum size of the ciphertext is written to |out_len|. Otherwise, |*out_len|
 * must contain the number of bytes of space available at |out|. If sufficient,
 * the ciphertext will be written to |out| and |*out_len| updated with the true
 * length.
 *
 * WARNING: Setting |out| to NULL only gives the maximum size of the
 * ciphertext. The actual ciphertext may be smaller.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_encrypt(EVP_PKEY_CTX *ctx, uint8_t *out,
                                    size_t *out_len, const uint8_t *in,
                                    size_t in_len);

/* EVP_PKEY_decrypt_init initialises an |EVP_PKEY_CTX| for a decryption
 * operation. It should be called before |EVP_PKEY_decrypt|.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_decrypt_init(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_decrypt decrypts |in_len| bytes from |in|. If |out| is NULL, the
 * maximum size of the plaintext is written to |out_len|. Otherwise, |*out_len|
 * must contain the number of bytes of space available at |out|. If sufficient,
 * the ciphertext will be written to |out| and |*out_len| updated with the true
 * length.
 *
 * WARNING: Setting |out| to NULL only gives the maximum size of the
 * plaintext. The actual plaintext may be smaller.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_decrypt(EVP_PKEY_CTX *ctx, uint8_t *out,
                                    size_t *out_len, const uint8_t *in,
                                    size_t in_len);

/* EVP_PKEY_derive_init initialises an |EVP_PKEY_CTX| for a key derivation
 * operation. It should be called before |EVP_PKEY_derive_set_peer| and
 * |EVP_PKEY_derive|.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_derive_init(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_derive_set_peer sets the peer's key to be used for key derivation
 * by |ctx| to |peer|. It should be called after |EVP_PKEY_derive_init|. (For
 * example, this is used to set the peer's key in (EC)DH.) It returns one on
 * success and zero on error. */
OPENSSL_EXPORT int EVP_PKEY_derive_set_peer(EVP_PKEY_CTX *ctx, EVP_PKEY *peer);

/* EVP_PKEY_derive derives a shared key between the two keys configured in
 * |ctx|. If |key| is non-NULL then, on entry, |out_key_len| must contain the
 * amount of space at |key|. If sufficient then the shared key will be written
 * to |key| and |*out_key_len| will be set to the length. If |key| is NULL then
 * |out_key_len| will be set to the maximum length.
 *
 * WARNING: Setting |out| to NULL only gives the maximum size of the key. The
 * actual key may be smaller.
 *
 * It returns one on success and zero on error. */
OPENSSL_EXPORT int EVP_PKEY_derive(EVP_PKEY_CTX *ctx, uint8_t *key,
                                   size_t *out_key_len);

/* EVP_PKEY_keygen_init initialises an |EVP_PKEY_CTX| for a key generation
 * operation. It should be called before |EVP_PKEY_keygen|.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_keygen_init(EVP_PKEY_CTX *ctx);

/* EVP_PKEY_keygen performs a key generation operation using the values from
 * |ctx| and sets |*ppkey| to a fresh |EVP_PKEY| containing the resulting key.
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_keygen(EVP_PKEY_CTX *ctx, EVP_PKEY **ppkey);


/* Generic control functions. */

/* EVP_PKEY_CTX_set_signature_md sets |md| as the digest to be used in a
 * signature operation. It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_set_signature_md(EVP_PKEY_CTX *ctx,
                                                 const EVP_MD *md);

/* EVP_PKEY_CTX_get_signature_md sets |*out_md| to the digest to be used in a
 * signature operation. It returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_get_signature_md(EVP_PKEY_CTX *ctx,
                                                 const EVP_MD **out_md);


/* RSA specific control functions. */

/* EVP_PKEY_CTX_set_rsa_padding sets the padding type to use. It should be one
 * of the |RSA_*_PADDING| values. Returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_set_rsa_padding(EVP_PKEY_CTX *ctx, int padding);

/* EVP_PKEY_CTX_get_rsa_padding sets |*out_padding| to the current padding
 * value, which is one of the |RSA_*_PADDING| values. Returns one on success or
 * zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_get_rsa_padding(EVP_PKEY_CTX *ctx,
                                                int *out_padding);

/* EVP_PKEY_CTX_set_rsa_pss_saltlen sets the length of the salt in a PSS-padded
 * signature. A value of -1 cause the salt to be the same length as the digest
 * in the signature. A value of -2 causes the salt to be the maximum length
 * that will fit. Otherwise the value gives the size of the salt in bytes.
 *
 * Returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_set_rsa_pss_saltlen(EVP_PKEY_CTX *ctx,
                                                    int salt_len);

/* EVP_PKEY_CTX_get_rsa_pss_saltlen sets |*out_salt_len| to the salt length of
 * a PSS-padded signature. See the documentation for
 * |EVP_PKEY_CTX_set_rsa_pss_saltlen| for details of the special values that it
 * can take.
 *
 * Returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_get_rsa_pss_saltlen(EVP_PKEY_CTX *ctx,
                                                    int *out_salt_len);

/* EVP_PKEY_CTX_set_rsa_keygen_bits sets the size of the desired RSA modulus,
 * in bits, for key generation. Returns one on success or zero on
 * error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_set_rsa_keygen_bits(EVP_PKEY_CTX *ctx,
                                                    int bits);

/* EVP_PKEY_CTX_set_rsa_keygen_pubexp sets |e| as the public exponent for key
 * generation. Returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_set_rsa_keygen_pubexp(EVP_PKEY_CTX *ctx,
                                                      BIGNUM *e);

/* EVP_PKEY_CTX_set_rsa_oaep_md sets |md| as the digest used in OAEP padding.
 * Returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_set_rsa_oaep_md(EVP_PKEY_CTX *ctx,
                                                const EVP_MD *md);

/* EVP_PKEY_CTX_get_rsa_oaep_md sets |*out_md| to the digest function used in
 * OAEP padding. Returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_get_rsa_oaep_md(EVP_PKEY_CTX *ctx,
                                                const EVP_MD **out_md);

/* EVP_PKEY_CTX_set_rsa_mgf1_md sets |md| as the digest used in MGF1. Returns
 * one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_set_rsa_mgf1_md(EVP_PKEY_CTX *ctx,
                                                const EVP_MD *md);

/* EVP_PKEY_CTX_get_rsa_mgf1_md sets |*out_md| to the digest function used in
 * MGF1. Returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_get_rsa_mgf1_md(EVP_PKEY_CTX *ctx,
                                                const EVP_MD **out_md);

/* EVP_PKEY_CTX_set0_rsa_oaep_label sets |label_len| bytes from |label| as the
 * label used in OAEP. DANGER: On success, this call takes ownership of |label|
 * and will call |OPENSSL_free| on it when |ctx| is destroyed.
 *
 * Returns one on success or zero on error. */
OPENSSL_EXPORT int EVP_PKEY_CTX_set0_rsa_oaep_label(EVP_PKEY_CTX *ctx,
                                                    const uint8_t *label,
                                                    size_t label_len);

/* EVP_PKEY_CTX_get0_rsa_oaep_label sets |*out_label| to point to the internal
 * buffer containing the OAEP label (which may be NULL) and returns the length
 * of the label or a negative value on error.
 *
 * WARNING: the return value differs from the usual return value convention. */
OPENSSL_EXPORT int EVP_PKEY_CTX_get0_rsa_oaep_label(EVP_PKEY_CTX *ctx,
                                                    const uint8_t **out_label);


/* Deprecated functions. */

/* EVP_PKEY_dup adds one to the reference count of |pkey| and returns
 * |pkey|.
 *
 * WARNING: this is a |_dup| function that doesn't actually duplicate! Use
 * |EVP_PKEY_up_ref| if you want to increment the reference count without
 * confusion. */
OPENSSL_EXPORT EVP_PKEY *EVP_PKEY_dup(EVP_PKEY *pkey);


/* Private functions */

/* OpenSSL_add_all_algorithms does nothing. */
OPENSSL_EXPORT void OpenSSL_add_all_algorithms(void);

/* OpenSSL_add_all_ciphers does nothing. */
OPENSSL_EXPORT void OpenSSL_add_all_ciphers(void);

/* OpenSSL_add_all_digests does nothing. */
OPENSSL_EXPORT void OpenSSL_add_all_digests(void);

/* EVP_cleanup does nothing. */
OPENSSL_EXPORT void EVP_cleanup(void);

/* EVP_PKEY_asn1_find returns the ASN.1 method table for the given |nid|, which
 * should be one of the |EVP_PKEY_*| values. It returns NULL if |nid| is
 * unknown. */
OPENSSL_EXPORT const EVP_PKEY_ASN1_METHOD *EVP_PKEY_asn1_find(ENGINE **pengine,
                                                              int nid);

/* TODO(fork): move to PEM? */
OPENSSL_EXPORT const EVP_PKEY_ASN1_METHOD *EVP_PKEY_asn1_find_str(
    ENGINE **pengine, const char *name, size_t len);

struct evp_pkey_st {
  CRYPTO_refcount_t references;

  /* type contains one of the EVP_PKEY_* values or NID_undef and determines
   * which element (if any) of the |pkey| union is valid. */
  int type;

  union {
    char *ptr;
    struct rsa_st *rsa; /* RSA */
    struct dsa_st *dsa; /* DSA */
    struct dh_st *dh; /* DH */
    struct ec_key_st *ec; /* ECC */
  } pkey;

  /* ameth contains a pointer to a method table that contains many ASN.1
   * methods for the key type. */
  const EVP_PKEY_ASN1_METHOD *ameth;
} /* EVP_PKEY */;


#if defined(__cplusplus)
}  /* extern C */
#endif

#define EVP_F_EVP_PKEY_derive_init 108
#define EVP_F_EVP_PKEY_encrypt 110
#define EVP_F_EVP_PKEY_encrypt_init 111
#define EVP_F_EVP_PKEY_get1_DH 112
#define EVP_F_EVP_PKEY_get1_EC_KEY 114
#define EVP_F_EVP_PKEY_get1_RSA 115
#define EVP_F_EVP_PKEY_keygen 116
#define EVP_F_EVP_PKEY_sign 120
#define EVP_F_EVP_PKEY_sign_init 121
#define EVP_F_EVP_PKEY_verify 122
#define EVP_F_EVP_PKEY_verify_init 123
#define EVP_F_d2i_AutoPrivateKey 125
#define EVP_F_d2i_PrivateKey 126
#define EVP_F_do_EC_KEY_print 127
#define EVP_F_do_sigver_init 129
#define EVP_F_eckey_param2type 130
#define EVP_F_eckey_param_decode 131
#define EVP_F_eckey_priv_decode 132
#define EVP_F_eckey_priv_encode 133
#define EVP_F_eckey_pub_decode 134
#define EVP_F_eckey_pub_encode 135
#define EVP_F_eckey_type2param 136
#define EVP_F_evp_pkey_ctx_new 137
#define EVP_F_hmac_signctx 138
#define EVP_F_i2d_PublicKey 139
#define EVP_F_old_ec_priv_decode 140
#define EVP_F_old_rsa_priv_decode 141
#define EVP_F_pkey_ec_ctrl 142
#define EVP_F_pkey_ec_derive 143
#define EVP_F_pkey_ec_keygen 144
#define EVP_F_pkey_ec_paramgen 145
#define EVP_F_pkey_ec_sign 146
#define EVP_F_pkey_rsa_ctrl 147
#define EVP_F_pkey_rsa_decrypt 148
#define EVP_F_pkey_rsa_encrypt 149
#define EVP_F_pkey_rsa_sign 150
#define EVP_F_rsa_algor_to_md 151
#define EVP_F_rsa_digest_verify_init_from_algorithm 152
#define EVP_F_rsa_mgf1_to_md 153
#define EVP_F_rsa_priv_decode 154
#define EVP_F_rsa_priv_encode 155
#define EVP_F_rsa_pss_to_ctx 156
#define EVP_F_rsa_pub_decode 157
#define EVP_F_pkey_hmac_ctrl 158
#define EVP_F_EVP_PKEY_CTX_get0_rsa_oaep_label 159
#define EVP_F_EVP_DigestSignAlgorithm 160
#define EVP_F_EVP_DigestVerifyInitFromAlgorithm 161
#define EVP_F_EVP_PKEY_CTX_ctrl 162
#define EVP_F_EVP_PKEY_CTX_dup 163
#define EVP_F_EVP_PKEY_copy_parameters 164
#define EVP_F_EVP_PKEY_decrypt 165
#define EVP_F_EVP_PKEY_decrypt_init 166
#define EVP_F_EVP_PKEY_derive 167
#define EVP_F_EVP_PKEY_derive_set_peer 168
#define EVP_F_EVP_PKEY_get1_DSA 169
#define EVP_F_EVP_PKEY_keygen_init 170
#define EVP_F_EVP_PKEY_new 171
#define EVP_F_EVP_PKEY_set_type 172
#define EVP_F_check_padding_md 173
#define EVP_F_do_dsa_print 174
#define EVP_F_do_rsa_print 175
#define EVP_F_dsa_param_decode 176
#define EVP_F_dsa_priv_decode 177
#define EVP_F_dsa_priv_encode 178
#define EVP_F_dsa_pub_decode 179
#define EVP_F_dsa_pub_encode 180
#define EVP_F_dsa_sig_print 181
#define EVP_F_old_dsa_priv_decode 182
#define EVP_R_BUFFER_TOO_SMALL 100
#define EVP_R_COMMAND_NOT_SUPPORTED 101
#define EVP_R_DIFFERENT_KEY_TYPES 104
#define EVP_R_DIFFERENT_PARAMETERS 105
#define EVP_R_EXPECTING_AN_EC_KEY_KEY 107
#define EVP_R_EXPECTING_A_DH_KEY 109
#define EVP_R_EXPECTING_A_DSA_KEY 110
#define EVP_R_ILLEGAL_OR_UNSUPPORTED_PADDING_MODE 111
#define EVP_R_INVALID_CURVE 112
#define EVP_R_INVALID_DIGEST_LENGTH 113
#define EVP_R_INVALID_DIGEST_TYPE 114
#define EVP_R_INVALID_KEYBITS 115
#define EVP_R_INVALID_MGF1_MD 116
#define EVP_R_INVALID_PADDING_MODE 118
#define EVP_R_INVALID_PSS_PARAMETERS 119
#define EVP_R_INVALID_SALT_LENGTH 121
#define EVP_R_INVALID_TRAILER 122
#define EVP_R_KEYS_NOT_SET 123
#define EVP_R_MISSING_PARAMETERS 124
#define EVP_R_NO_DEFAULT_DIGEST 125
#define EVP_R_NO_KEY_SET 126
#define EVP_R_NO_MDC2_SUPPORT 127
#define EVP_R_NO_NID_FOR_CURVE 128
#define EVP_R_NO_OPERATION_SET 129
#define EVP_R_NO_PARAMETERS_SET 130
#define EVP_R_OPERATION_NOT_SUPPORTED_FOR_THIS_KEYTYPE 131
#define EVP_R_OPERATON_NOT_INITIALIZED 132
#define EVP_R_UNKNOWN_DIGEST 133
#define EVP_R_UNKNOWN_MASK_DIGEST 134
#define EVP_R_UNSUPPORTED_ALGORITHM 138
#define EVP_R_UNSUPPORTED_MASK_ALGORITHM 139
#define EVP_R_UNSUPPORTED_MASK_PARAMETER 140
#define EVP_R_EXPECTING_AN_RSA_KEY 141
#define EVP_R_INVALID_OPERATION 142
#define EVP_R_DECODE_ERROR 143
#define EVP_R_INVALID_PSS_SALTLEN 144
#define EVP_R_UNKNOWN_PUBLIC_KEY_TYPE 145
#define EVP_R_CONTEXT_NOT_INITIALISED 146
#define EVP_R_DIGEST_AND_KEY_TYPE_NOT_SUPPORTED 147
#define EVP_R_WRONG_PUBLIC_KEY_TYPE 148
#define EVP_R_UNKNOWN_SIGNATURE_ALGORITHM 149
#define EVP_R_UNKNOWN_MESSAGE_DIGEST_ALGORITHM 150
#define EVP_R_BN_DECODE_ERROR 151
#define EVP_R_PARAMETER_ENCODING_ERROR 152
#define EVP_R_UNSUPPORTED_PUBLIC_KEY_TYPE 153
#define EVP_R_UNSUPPORTED_SIGNATURE_TYPE 154

#endif  /* OPENSSL_HEADER_EVP_H */
