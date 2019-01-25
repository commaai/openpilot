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

#ifndef OPENSSL_HEADER_HMAC_H
#define OPENSSL_HEADER_HMAC_H

#include <openssl/base.h>

#include <openssl/digest.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* HMAC contains functions for constructing PRFs from Merkle–Damgård hash
 * functions using HMAC. */


/* One-shot operation. */

/* HMAC calculates the HMAC of |data_len| bytes of |data|, using the given key
 * and hash function, and writes the result to |out|. On entry, |out| must
 * contain |EVP_MAX_MD_SIZE| bytes of space. The actual length of the result is
 * written to |*out_len|. It returns |out| or NULL on error. */
OPENSSL_EXPORT uint8_t *HMAC(const EVP_MD *evp_md, const void *key,
                             size_t key_len, const uint8_t *data,
                             size_t data_len, uint8_t *out,
                             unsigned int *out_len);


/* Incremental operation. */

/* HMAC_CTX_init initialises |ctx| for use in an HMAC operation. It's assumed
 * that HMAC_CTX objects will be allocated on the stack thus no allocation
 * function is provided. If needed, allocate |sizeof(HMAC_CTX)| and call
 * |HMAC_CTX_init| on it. */
OPENSSL_EXPORT void HMAC_CTX_init(HMAC_CTX *ctx);

/* HMAC_CTX_cleanup frees data owned by |ctx|. */
OPENSSL_EXPORT void HMAC_CTX_cleanup(HMAC_CTX *ctx);

/* HMAC_Init_ex sets up an initialised |HMAC_CTX| to use |md| as the hash
 * function and |key| as the key. For a non-initial call, |md| may be NULL, in
 * which case the previous hash function will be used. If the hash function has
 * not changed and |key| is NULL, |ctx| reuses the previous key. It returns one
 * on success or zero otherwise.
 *
 * WARNING: NULL and empty keys are ambiguous on non-initial calls. Passing NULL
 * |key| but repeating the previous |md| reuses the previous key rather than the
 * empty key. */
OPENSSL_EXPORT int HMAC_Init_ex(HMAC_CTX *ctx, const void *key, size_t key_len,
                                const EVP_MD *md, ENGINE *impl);

/* HMAC_Update hashes |data_len| bytes from |data| into the current HMAC
 * operation in |ctx|. It returns one. */
OPENSSL_EXPORT int HMAC_Update(HMAC_CTX *ctx, const uint8_t *data,
                               size_t data_len);

/* HMAC_Final completes the HMAC operation in |ctx| and writes the result to
 * |out| and the sets |*out_len| to the length of the result. On entry, |out|
 * must contain at least |EVP_MAX_MD_SIZE| bytes of space. It returns one on
 * success or zero on error. */
OPENSSL_EXPORT int HMAC_Final(HMAC_CTX *ctx, uint8_t *out,
                              unsigned int *out_len);


/* Utility functions. */

/* HMAC_size returns the size, in bytes, of the HMAC that will be produced by
 * |ctx|. On entry, |ctx| must have been setup with |HMAC_Init_ex|. */
OPENSSL_EXPORT size_t HMAC_size(const HMAC_CTX *ctx);

/* HMAC_CTX_copy_ex sets |dest| equal to |src|. On entry, |dest| must have been
 * initialised by calling |HMAC_CTX_init|. It returns one on success and zero
 * on error. */
OPENSSL_EXPORT int HMAC_CTX_copy_ex(HMAC_CTX *dest, const HMAC_CTX *src);


/* Deprecated functions. */

OPENSSL_EXPORT int HMAC_Init(HMAC_CTX *ctx, const void *key, int key_len,
                             const EVP_MD *md);

/* HMAC_CTX_copy calls |HMAC_CTX_init| on |dest| and then sets it equal to
 * |src|. On entry, |dest| must /not/ be initialised for an operation with
 * |HMAC_Init_ex|. It returns one on success and zero on error. */
OPENSSL_EXPORT int HMAC_CTX_copy(HMAC_CTX *dest, const HMAC_CTX *src);


/* Private functions */

#define HMAC_MAX_MD_CBLOCK 128 /* largest known is SHA512 */

struct hmac_ctx_st {
  const EVP_MD *md;
  EVP_MD_CTX md_ctx;
  EVP_MD_CTX i_ctx;
  EVP_MD_CTX o_ctx;
} /* HMAC_CTX */;


#if defined(__cplusplus)
}  /* extern C */
#endif

#endif  /* OPENSSL_HEADER_HMAC_H */
