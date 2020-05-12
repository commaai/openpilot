/* Written by Dr Stephen N Henson (steve@openssl.org) for the OpenSSL
 * project 1999.
 */
/* ====================================================================
 * Copyright (c) 1999 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.OpenSSL.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    licensing@OpenSSL.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.OpenSSL.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com). */


#ifndef OPENSSL_HEADER_PKCS8_H
#define OPENSSL_HEADER_PKCS8_H

#include <openssl/base.h>
#include <openssl/x509.h>


#if defined(__cplusplus)
extern "C" {
#endif


/* PKCS8_encrypt_pbe serializes and encrypts a PKCS8_PRIV_KEY_INFO with PBES1 as
 * defined in PKCS #5. Only pbeWithSHAAnd128BitRC4,
 * pbeWithSHAAnd3-KeyTripleDES-CBC and pbeWithSHA1And40BitRC2, defined in PKCS
 * #12, are supported. The |pass_raw_len| bytes pointed to by |pass_raw| are
 * used as the password. Note that any conversions from the password as
 * supplied in a text string (such as those specified in B.1 of PKCS #12) must
 * be performed by the caller.
 *
 * If |salt| is NULL, a random salt of |salt_len| bytes is generated. If
 * |salt_len| is zero, a default salt length is used instead.
 *
 * The resulting structure is stored in an X509_SIG which must be freed by the
 * caller.
 *
 * TODO(davidben): Really? An X509_SIG? OpenSSL probably did that because it has
 * the same structure as EncryptedPrivateKeyInfo. */
OPENSSL_EXPORT X509_SIG *PKCS8_encrypt_pbe(int pbe_nid,
                                           const uint8_t *pass_raw,
                                           size_t pass_raw_len,
                                           uint8_t *salt, size_t salt_len,
                                           int iterations,
                                           PKCS8_PRIV_KEY_INFO *p8inf);

/* PKCS8_decrypt_pbe decrypts and decodes a PKCS8_PRIV_KEY_INFO with PBES1 as
 * defined in PKCS #5. Only pbeWithSHAAnd128BitRC4,
 * pbeWithSHAAnd3-KeyTripleDES-CBC and pbeWithSHA1And40BitRC2, defined in PKCS
 * #12, are supported. The |pass_raw_len| bytes pointed to by |pass_raw| are
 * used as the password. Note that any conversions from the password as
 * supplied in a text string (such as those specified in B.1 of PKCS #12) must
 * be performed by the caller.
 *
 * The resulting structure must be freed by the caller. */
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *PKCS8_decrypt_pbe(X509_SIG *pkcs8,
                                                      const uint8_t *pass_raw,
                                                      size_t pass_raw_len);


/* Deprecated functions. */

/* PKCS8_encrypt calls PKCS8_encrypt_pbe after treating |pass| as an ASCII
 * string, appending U+0000, and converting to UCS-2. (So the empty password
 * encodes as two NUL bytes.) The |cipher| argument is ignored. */
OPENSSL_EXPORT X509_SIG *PKCS8_encrypt(int pbe_nid, const EVP_CIPHER *cipher,
                                       const char *pass, int pass_len,
                                       uint8_t *salt, size_t salt_len,
                                       int iterations,
                                       PKCS8_PRIV_KEY_INFO *p8inf);

/* PKCS8_decrypt calls PKCS8_decrypt_pbe after treating |pass| as an ASCII
 * string, appending U+0000, and converting to UCS-2. (So the empty password
 * encodes as two NUL bytes.) */
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *PKCS8_decrypt(X509_SIG *pkcs8,
                                                  const char *pass,
                                                  int pass_len);

/* PKCS12_get_key_and_certs parses a PKCS#12 structure from |in|, authenticates
 * and decrypts it using |password|, sets |*out_key| to the included private
 * key and appends the included certificates to |out_certs|. It returns one on
 * success and zero on error. The caller takes ownership of the outputs. */
OPENSSL_EXPORT int PKCS12_get_key_and_certs(EVP_PKEY **out_key,
                                            STACK_OF(X509) *out_certs,
                                            CBS *in, const char *password);


/* Deprecated functions. */

/* PKCS12_PBE_add does nothing. It exists for compatibility with OpenSSL. */
OPENSSL_EXPORT void PKCS12_PBE_add(void);

/* d2i_PKCS12 is a dummy function that copies |*ber_bytes| into a
 * |PKCS12| structure. The |out_p12| argument must be NULL. On exit,
 * |*ber_bytes| will be advanced by |ber_len|. It returns a fresh |PKCS12|
 * structure or NULL on error.
 *
 * Note: unlike other d2i functions, |d2i_PKCS12| will always consume |ber_len|
 * bytes.*/
OPENSSL_EXPORT PKCS12 *d2i_PKCS12(PKCS12 **out_p12, const uint8_t **ber_bytes,
                                  size_t ber_len);

/* d2i_PKCS12_bio acts like |d2i_PKCS12| but reads from a |BIO|. */
OPENSSL_EXPORT PKCS12* d2i_PKCS12_bio(BIO *bio, PKCS12 **out_p12);

/* d2i_PKCS12_fp acts like |d2i_PKCS12| but reads from a |FILE|. */
OPENSSL_EXPORT PKCS12* d2i_PKCS12_fp(FILE *fp, PKCS12 **out_p12);

/* PKCS12_parse calls |PKCS12_get_key_and_certs| on the ASN.1 data stored in
 * |p12|. The |out_pkey| and |out_cert| arguments must not be NULL and, on
 * successful exit, the private key and first certificate will be stored in
 * them. The |out_ca_certs| argument may be NULL but, if not, then any extra
 * certificates will be appended to |*out_ca_certs|. If |*out_ca_certs| is NULL
 * then it will be set to a freshly allocated stack containing the extra certs.
 *
 * It returns one on success and zero on error. */
OPENSSL_EXPORT int PKCS12_parse(const PKCS12 *p12, const char *password,
                                EVP_PKEY **out_pkey, X509 **out_cert,
                                STACK_OF(X509) **out_ca_certs);

/* PKCS12_free frees |p12| and its contents. */
OPENSSL_EXPORT void PKCS12_free(PKCS12 *p12);

#if defined(__cplusplus)
}  /* extern C */
#endif

#define PKCS8_F_EVP_PKCS82PKEY 100
#define PKCS8_F_EVP_PKEY2PKCS8 101
#define PKCS8_F_PKCS12_get_key_and_certs 102
#define PKCS8_F_PKCS12_handle_content_info 103
#define PKCS8_F_PKCS12_handle_content_infos 104
#define PKCS8_F_PKCS5_pbe2_set_iv 105
#define PKCS8_F_PKCS5_pbe_set 106
#define PKCS8_F_PKCS5_pbe_set0_algor 107
#define PKCS8_F_PKCS5_pbkdf2_set 108
#define PKCS8_F_PKCS8_decrypt 109
#define PKCS8_F_PKCS8_encrypt 110
#define PKCS8_F_PKCS8_encrypt_pbe 111
#define PKCS8_F_pbe_cipher_init 112
#define PKCS8_F_pbe_crypt 113
#define PKCS8_F_pkcs12_item_decrypt_d2i 114
#define PKCS8_F_pkcs12_item_i2d_encrypt 115
#define PKCS8_F_pkcs12_key_gen_raw 116
#define PKCS8_F_pkcs12_pbe_keyivgen 117
#define PKCS8_R_BAD_PKCS12_DATA 100
#define PKCS8_R_BAD_PKCS12_VERSION 101
#define PKCS8_R_CIPHER_HAS_NO_OBJECT_IDENTIFIER 102
#define PKCS8_R_CRYPT_ERROR 103
#define PKCS8_R_DECODE_ERROR 104
#define PKCS8_R_ENCODE_ERROR 105
#define PKCS8_R_ENCRYPT_ERROR 106
#define PKCS8_R_ERROR_SETTING_CIPHER_PARAMS 107
#define PKCS8_R_INCORRECT_PASSWORD 108
#define PKCS8_R_KEYGEN_FAILURE 109
#define PKCS8_R_KEY_GEN_ERROR 110
#define PKCS8_R_METHOD_NOT_SUPPORTED 111
#define PKCS8_R_MISSING_MAC 112
#define PKCS8_R_MULTIPLE_PRIVATE_KEYS_IN_PKCS12 113
#define PKCS8_R_PKCS12_PUBLIC_KEY_INTEGRITY_NOT_SUPPORTED 114
#define PKCS8_R_PKCS12_TOO_DEEPLY_NESTED 115
#define PKCS8_R_PRIVATE_KEY_DECODE_ERROR 116
#define PKCS8_R_PRIVATE_KEY_ENCODE_ERROR 117
#define PKCS8_R_TOO_LONG 118
#define PKCS8_R_UNKNOWN_ALGORITHM 119
#define PKCS8_R_UNKNOWN_CIPHER 120
#define PKCS8_R_UNKNOWN_CIPHER_ALGORITHM 121
#define PKCS8_R_UNKNOWN_DIGEST 122
#define PKCS8_R_UNKNOWN_HASH 123
#define PKCS8_R_UNSUPPORTED_PRIVATE_KEY_ALGORITHM 124

#endif  /* OPENSSL_HEADER_PKCS8_H */
