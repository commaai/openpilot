/* Originally written by Bodo Moeller for the OpenSSL project.
 * ====================================================================
 * Copyright (c) 1998-2005 The OpenSSL Project.  All rights reserved.
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
 *    for use in the OpenSSL Toolkit. (http://www.openssl.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@openssl.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.openssl.org/)"
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
 * Hudson (tjh@cryptsoft.com).
 *
 */
/* ====================================================================
 * Copyright 2002 Sun Microsystems, Inc. ALL RIGHTS RESERVED.
 *
 * Portions of the attached software ("Contribution") are developed by
 * SUN MICROSYSTEMS, INC., and are contributed to the OpenSSL project.
 *
 * The Contribution is licensed pursuant to the OpenSSL open source
 * license provided above.
 *
 * The elliptic curve binary polynomial software is originally written by
 * Sheueling Chang Shantz and Douglas Stebila of Sun Microsystems
 * Laboratories. */

#ifndef OPENSSL_HEADER_EC_KEY_H
#define OPENSSL_HEADER_EC_KEY_H

#include <openssl/base.h>

#include <openssl/ec.h>
#include <openssl/engine.h>
#include <openssl/ex_data.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* ec_key.h contains functions that handle elliptic-curve points that are
 * public/private keys. */


/* EC key objects. */

/* EC_KEY_new returns a fresh |EC_KEY| object or NULL on error. */
OPENSSL_EXPORT EC_KEY *EC_KEY_new(void);

/* EC_KEY_new_method acts the same as |EC_KEY_new|, but takes an explicit
 * |ENGINE|. */
OPENSSL_EXPORT EC_KEY *EC_KEY_new_method(const ENGINE *engine);

/* EC_KEY_new_by_curve_name returns a fresh EC_KEY for group specified by |nid|
 * or NULL on error. */
OPENSSL_EXPORT EC_KEY *EC_KEY_new_by_curve_name(int nid);

/* EC_KEY_free frees all the data owned by |key| and |key| itself. */
OPENSSL_EXPORT void EC_KEY_free(EC_KEY *key);

/* EC_KEY_copy sets |dst| equal to |src| and returns |dst| or NULL on error. */
OPENSSL_EXPORT EC_KEY *EC_KEY_copy(EC_KEY *dst, const EC_KEY *src);

/* EC_KEY_dup returns a fresh copy of |src| or NULL on error. */
OPENSSL_EXPORT EC_KEY *EC_KEY_dup(const EC_KEY *src);

/* EC_KEY_up_ref increases the reference count of |key|. It returns one on
 * success and zero otherwise. */
OPENSSL_EXPORT int EC_KEY_up_ref(EC_KEY *key);

/* EC_KEY_is_opaque returns one if |key| is opaque and doesn't expose its key
 * material. Otherwise it return zero. */
OPENSSL_EXPORT int EC_KEY_is_opaque(const EC_KEY *key);

/* EC_KEY_get0_group returns a pointer to the |EC_GROUP| object inside |key|. */
OPENSSL_EXPORT const EC_GROUP *EC_KEY_get0_group(const EC_KEY *key);

/* EC_KEY_set_group sets the |EC_GROUP| object that |key| will use to |group|.
 * It returns one on success and zero otherwise. */
OPENSSL_EXPORT int EC_KEY_set_group(EC_KEY *key, const EC_GROUP *group);

/* EC_KEY_get0_private_key returns a pointer to the private key inside |key|. */
OPENSSL_EXPORT const BIGNUM *EC_KEY_get0_private_key(const EC_KEY *key);

/* EC_KEY_set_private_key sets the private key of |key| to |priv|. It returns
 * one on success and zero otherwise. */
OPENSSL_EXPORT int EC_KEY_set_private_key(EC_KEY *key, const BIGNUM *prv);

/* EC_KEY_get0_public_key returns a pointer to the public key point inside
 * |key|. */
OPENSSL_EXPORT const EC_POINT *EC_KEY_get0_public_key(const EC_KEY *key);

/* EC_KEY_set_public_key sets the public key of |key| to |pub|, by copying it.
 * It returns one on success and zero otherwise. */
OPENSSL_EXPORT int EC_KEY_set_public_key(EC_KEY *key, const EC_POINT *pub);

#define EC_PKEY_NO_PARAMETERS 0x001
#define EC_PKEY_NO_PUBKEY 0x002

/* EC_KEY_get_enc_flags returns the encoding flags for |key|, which is a
 * bitwise-OR of |EC_PKEY_*| values. */
OPENSSL_EXPORT unsigned EC_KEY_get_enc_flags(const EC_KEY *key);

/* EC_KEY_set_enc_flags sets the encoding flags for |key|, which is a
 * bitwise-OR of |EC_PKEY_*| values. */
OPENSSL_EXPORT void EC_KEY_set_enc_flags(EC_KEY *key, unsigned flags);

/* EC_KEY_get_conv_form returns the conversation form that will be used by
 * |key|. */
OPENSSL_EXPORT point_conversion_form_t EC_KEY_get_conv_form(const EC_KEY *key);

/* EC_KEY_set_conv_form sets the conversion form to be used by |key|. */
OPENSSL_EXPORT void EC_KEY_set_conv_form(EC_KEY *key,
                                         point_conversion_form_t cform);

/* EC_KEY_precompute_mult precomputes multiplies of the generator of the
 * underlying group in order to speed up operations that calculate generator
 * multiples. If |ctx| is not NULL, it may be used. It returns one on success
 * and zero otherwise. */
OPENSSL_EXPORT int EC_KEY_precompute_mult(EC_KEY *key, BN_CTX *ctx);

/* EC_KEY_check_key performs several checks on |key| (possibly including an
 * expensive check that the public key is in the primary subgroup). It returns
 * one if all checks pass and zero otherwise. If it returns zero then detail
 * about the problem can be found on the error stack. */
OPENSSL_EXPORT int EC_KEY_check_key(const EC_KEY *key);

/* EC_KEY_set_public_key_affine_coordinates sets the public key in |key| to
 * (|x|, |y|). It returns one on success and zero otherwise. */
OPENSSL_EXPORT int EC_KEY_set_public_key_affine_coordinates(EC_KEY *key,
                                                            BIGNUM *x,
                                                            BIGNUM *y);


/* Key generation. */

/* EC_KEY_generate_key generates a random, private key, calculates the
 * corresponding public key and stores both in |key|. It returns one on success
 * or zero otherwise. */
OPENSSL_EXPORT int EC_KEY_generate_key(EC_KEY *key);


/* Serialisation. */

/* d2i_ECPrivateKey parses an ASN.1, DER-encoded, private key from |len| bytes
 * at |*inp|. If |out_key| is not NULL then, on exit, a pointer to the result
 * is in |*out_key|. If |*out_key| is already non-NULL on entry then the result
 * is written directly into |*out_key|, otherwise a fresh |EC_KEY| is
 * allocated. On successful exit, |*inp| is advanced past the DER structure. It
 * returns the result or NULL on error. */
OPENSSL_EXPORT EC_KEY *d2i_ECPrivateKey(EC_KEY **out_key, const uint8_t **inp,
                                        long len);

/* i2d_ECParameters marshals an EC private key from |key| to an ASN.1, DER
 * structure. If |outp| is not NULL then the result is written to |*outp| and
 * |*outp| is advanced just past the output. It returns the number of bytes in
 * the result, whether written or not, or a negative value on error. */
OPENSSL_EXPORT int i2d_ECPrivateKey(const EC_KEY *key, uint8_t **outp);

/* d2i_ECParameters parses an ASN.1, DER-encoded, set of EC parameters from
 * |len| bytes at |*inp|. If |out_key| is not NULL then, on exit, a pointer to
 * the result is in |*out_key|. If |*out_key| is already non-NULL on entry then
 * the result is written directly into |*out_key|, otherwise a fresh |EC_KEY|
 * is allocated. On successful exit, |*inp| is advanced past the DER structure.
 * It returns the result or NULL on error. */
OPENSSL_EXPORT EC_KEY *d2i_ECParameters(EC_KEY **out_key, const uint8_t **inp,
                                        long len);

/* i2d_ECParameters marshals EC parameters from |key| to an ASN.1, DER
 * structure. If |outp| is not NULL then the result is written to |*outp| and
 * |*outp| is advanced just past the output. It returns the number of bytes in
 * the result, whether written or not, or a negative value on error. */
OPENSSL_EXPORT int i2d_ECParameters(const EC_KEY *key, uint8_t **outp);

/* o2i_ECPublicKey parses an EC point from |len| bytes at |*inp| into
 * |*out_key|. Note that this differs from the d2i format in that |*out_key|
 * must be non-NULL. On successful exit, |*inp| is advanced past the DER
 * structure. It returns |*out_key| or NULL on error. */
OPENSSL_EXPORT EC_KEY *o2i_ECPublicKey(EC_KEY **out_key, const uint8_t **inp,
                                       long len);

/* i2o_ECPublicKey marshals an EC point from |key|. If |outp| is not NULL then
 * the result is written to |*outp| and |*outp| is advanced just past the
 * output. It returns the number of bytes in the result, whether written or
 * not, or a negative value on error. */
OPENSSL_EXPORT int i2o_ECPublicKey(const EC_KEY *key, unsigned char **outp);


/* ex_data functions.
 *
 * These functions are wrappers. See |ex_data.h| for details. */

OPENSSL_EXPORT int EC_KEY_get_ex_new_index(long argl, void *argp,
                                           CRYPTO_EX_new *new_func,
                                           CRYPTO_EX_dup *dup_func,
                                           CRYPTO_EX_free *free_func);
OPENSSL_EXPORT int EC_KEY_set_ex_data(EC_KEY *r, int idx, void *arg);
OPENSSL_EXPORT void *EC_KEY_get_ex_data(const EC_KEY *r, int idx);


/* ECDSA method. */

/* ECDSA_FLAG_OPAQUE specifies that this ECDSA_METHOD does not expose its key
 * material. This may be set if, for instance, it is wrapping some other crypto
 * API, like a platform key store. */
#define ECDSA_FLAG_OPAQUE 1

/* ecdsa_method_st is a structure of function pointers for implementing ECDSA.
 * See engine.h. */
struct ecdsa_method_st {
  struct openssl_method_common_st common;

  void *app_data;

  int (*init)(EC_KEY *key);
  int (*finish)(EC_KEY *key);

  /* group_order_size returns the number of bytes needed to represent the order
   * of the group. This is used to calculate the maximum size of an ECDSA
   * signature in |ECDSA_size|. */
  size_t (*group_order_size)(const EC_KEY *key);

  /* sign matches the arguments and behaviour of |ECDSA_sign|. */
  int (*sign)(const uint8_t *digest, size_t digest_len, uint8_t *sig,
              unsigned int *sig_len, EC_KEY *eckey);

  /* verify matches the arguments and behaviour of |ECDSA_verify|. */
  int (*verify)(const uint8_t *digest, size_t digest_len, const uint8_t *sig,
                size_t sig_len, EC_KEY *eckey);

  int flags;
};


/* Deprecated functions. */

/* EC_KEY_set_asn1_flag does nothing. */
OPENSSL_EXPORT void EC_KEY_set_asn1_flag(EC_KEY *key, int flag);


#if defined(__cplusplus)
}  /* extern C */
#endif

#endif  /* OPENSSL_HEADER_EC_KEY_H */
