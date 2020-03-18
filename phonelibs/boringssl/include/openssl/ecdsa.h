/* ====================================================================
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
 *    for use in the OpenSSL Toolkit. (http://www.OpenSSL.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@OpenSSL.org.
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

#ifndef OPENSSL_HEADER_ECDSA_H
#define OPENSSL_HEADER_ECDSA_H

#include <openssl/base.h>

#include <openssl/ec_key.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* ECDSA contains functions for signing and verifying with the Digital Signature
 * Algorithm over elliptic curves. */


/* Signing and verifing. */

/* ECDSA_sign signs |digest_len| bytes from |digest| with |key| and writes the
 * resulting signature to |sig|, which must have |ECDSA_size(key)| bytes of
 * space. On successful exit, |*sig_len| is set to the actual number of bytes
 * written. The |type| argument should be zero. It returns one on success and
 * zero otherwise. */
OPENSSL_EXPORT int ECDSA_sign(int type, const uint8_t *digest,
                              size_t digest_len, uint8_t *sig,
                              unsigned int *sig_len, EC_KEY *key);

/* ECDSA_verify verifies that |sig_len| bytes from |sig| constitute a valid
 * signature by |key| of |digest|. (The |type| argument should be zero.) It
 * returns one on success or zero if the signature is invalid or an error
 * occured. */
OPENSSL_EXPORT int ECDSA_verify(int type, const uint8_t *digest,
                                size_t digest_len, const uint8_t *sig,
                                size_t sig_len, EC_KEY *key);

/* ECDSA_size returns the maximum size of an ECDSA signature using |key|. It
 * returns zero on error. */
OPENSSL_EXPORT size_t ECDSA_size(const EC_KEY *key);


/* Low-level signing and verification.
 *
 * Low-level functions handle signatures as |ECDSA_SIG| structures which allow
 * the two values in an ECDSA signature to be handled separately. */

struct ecdsa_sig_st {
  BIGNUM *r;
  BIGNUM *s;
};

/* ECDSA_SIG_new returns a fresh |ECDSA_SIG| structure or NULL on error. */
OPENSSL_EXPORT ECDSA_SIG *ECDSA_SIG_new(void);

/* ECDSA_SIG_free frees |sig| its member |BIGNUM|s. */
OPENSSL_EXPORT void ECDSA_SIG_free(ECDSA_SIG *sig);

/* ECDSA_sign signs |digest_len| bytes from |digest| with |key| and returns the
 * resulting signature structure, or NULL on error. */
OPENSSL_EXPORT ECDSA_SIG *ECDSA_do_sign(const uint8_t *digest,
                                        size_t digest_len, EC_KEY *key);

/* ECDSA_verify verifies that |sig| constitutes a valid signature by |key| of
 * |digest|. It returns one on success or zero if the signature is invalid or
 * on error. */
OPENSSL_EXPORT int ECDSA_do_verify(const uint8_t *digest, size_t digest_len,
                                   const ECDSA_SIG *sig, EC_KEY *key);


/* Signing with precomputation.
 *
 * Parts of the ECDSA signature can be independent of the message to be signed
 * thus it's possible to precompute them and reduce the signing latency.
 *
 * TODO(fork): remove support for this as it cannot support safe-randomness. */

/* ECDSA_sign_setup precomputes parts of an ECDSA signing operation. It sets
 * |*kinv| and |*rp| to the precomputed values and uses the |ctx| argument, if
 * not NULL. It returns one on success and zero otherwise. */
OPENSSL_EXPORT int ECDSA_sign_setup(EC_KEY *eckey, BN_CTX *ctx, BIGNUM **kinv,
                                    BIGNUM **rp);

/* ECDSA_do_sign_ex is the same as |ECDSA_do_sign| but takes precomputed values
 * as generated by |ECDSA_sign_setup|. */
OPENSSL_EXPORT ECDSA_SIG *ECDSA_do_sign_ex(const uint8_t *digest,
                                           size_t digest_len,
                                           const BIGNUM *kinv, const BIGNUM *rp,
                                           EC_KEY *eckey);

/* ECDSA_sign_ex is the same as |ECDSA_sign| but takes precomputed values as
 * generated by |ECDSA_sign_setup|. */
OPENSSL_EXPORT int ECDSA_sign_ex(int type, const uint8_t *digest,
                                 size_t digest_len, uint8_t *sig,
                                 unsigned int *sig_len, const BIGNUM *kinv,
                                 const BIGNUM *rp, EC_KEY *eckey);


/* ASN.1 functions. */

/* d2i_ECDSA_SIG parses an ASN.1, DER-encoded, signature from |len| bytes at
 * |*inp|. If |out| is not NULL then, on exit, a pointer to the result is in
 * |*out|. If |*out| is already non-NULL on entry then the result is written
 * directly into |*out|, otherwise a fresh |ECDSA_SIG| is allocated. On
 * successful exit, |*inp| is advanced past the DER structure. It returns the
 * result or NULL on error. */
OPENSSL_EXPORT ECDSA_SIG *d2i_ECDSA_SIG(ECDSA_SIG **out, const uint8_t **inp,
                                        long len);

/* i2d_ECDSA_SIG marshals a signature from |sig| to an ASN.1, DER
 * structure. If |outp| is not NULL then the result is written to |*outp| and
 * |*outp| is advanced just past the output. It returns the number of bytes in
 * the result, whether written or not, or a negative value on error. */
OPENSSL_EXPORT int i2d_ECDSA_SIG(const ECDSA_SIG *sig, uint8_t **outp);


#if defined(__cplusplus)
}  /* extern C */
#endif

#define ECDSA_F_ECDSA_do_sign_ex 100
#define ECDSA_F_ECDSA_do_verify 101
#define ECDSA_F_ECDSA_sign_ex 102
#define ECDSA_F_digest_to_bn 103
#define ECDSA_F_ecdsa_sign_setup 104
#define ECDSA_R_BAD_SIGNATURE 100
#define ECDSA_R_MISSING_PARAMETERS 101
#define ECDSA_R_NEED_NEW_SETUP_VALUES 102
#define ECDSA_R_NOT_IMPLEMENTED 103
#define ECDSA_R_RANDOM_NUMBER_GENERATION_FAILED 104

#endif  /* OPENSSL_HEADER_ECDSA_H */
