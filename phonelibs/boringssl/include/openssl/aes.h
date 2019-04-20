/* ====================================================================
 * Copyright (c) 2002-2006 The OpenSSL Project.  All rights reserved.
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
 * ==================================================================== */

#ifndef OPENSSL_HEADER_AES_H
#define OPENSSL_HEADER_AES_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* Raw AES functions. */


#define AES_ENCRYPT 1
#define AES_DECRYPT 0

/* AES_MAXNR is the maximum number of AES rounds. */
#define AES_MAXNR 14

#define AES_BLOCK_SIZE 16

/* aes_key_st should be an opaque type, but EVP requires that the size be
 * known. */
struct aes_key_st {
  uint32_t rd_key[4 * (AES_MAXNR + 1)];
  unsigned rounds;
};
typedef struct aes_key_st AES_KEY;

/* AES_set_encrypt_key configures |aeskey| to encrypt with the |bits|-bit key,
 * |key|.
 *
 * WARNING: unlike other OpenSSL functions, this returns zero on success and a
 * negative number on error. */
OPENSSL_EXPORT int AES_set_encrypt_key(const uint8_t *key, unsigned bits,
                                       AES_KEY *aeskey);

/* AES_set_decrypt_key configures |aeskey| to decrypt with the |bits|-bit key,
 * |key|.
 *
 * WARNING: unlike other OpenSSL functions, this returns zero on success and a
 * negative number on error. */
OPENSSL_EXPORT int AES_set_decrypt_key(const uint8_t *key, unsigned bits,
                                       AES_KEY *aeskey);

/* AES_encrypt encrypts a single block from |in| to |out| with |key|. The |in|
 * and |out| pointers may overlap. */
OPENSSL_EXPORT void AES_encrypt(const uint8_t *in, uint8_t *out,
                                const AES_KEY *key);

/* AES_decrypt decrypts a single block from |in| to |out| with |key|. The |in|
 * and |out| pointers may overlap. */
OPENSSL_EXPORT void AES_decrypt(const uint8_t *in, uint8_t *out,
                                const AES_KEY *key);


/* Block cipher modes. */

/* AES_ctr128_encrypt encrypts (or decrypts, it's the same in CTR mode) |len|
 * bytes from |in| to |out|. The |num| parameter must be set to zero on the
 * first call and |ivec| will be incremented. */
OPENSSL_EXPORT void AES_ctr128_encrypt(const uint8_t *in, uint8_t *out,
                                       size_t len, const AES_KEY *key,
                                       uint8_t ivec[AES_BLOCK_SIZE],
                                       uint8_t ecount_buf[AES_BLOCK_SIZE],
                                       unsigned int *num);

/* AES_ecb_encrypt encrypts (or decrypts, if |enc| == |AES_DECRYPT|) a single,
 * 16 byte block from |in| to |out|. */
OPENSSL_EXPORT void AES_ecb_encrypt(const uint8_t *in, uint8_t *out,
                                    const AES_KEY *key, const int enc);

/* AES_cbc_encrypt encrypts (or decrypts, if |enc| == |AES_DECRYPT|) |len|
 * bytes from |in| to |out|. The length must be a multiple of the block size. */
OPENSSL_EXPORT void AES_cbc_encrypt(const uint8_t *in, uint8_t *out, size_t len,
                                    const AES_KEY *key, uint8_t *ivec,
                                    const int enc);

/* AES_ofb128_encrypt encrypts (or decrypts, it's the same in CTR mode) |len|
 * bytes from |in| to |out|. The |num| parameter must be set to zero on the
 * first call. */
OPENSSL_EXPORT void AES_ofb128_encrypt(const uint8_t *in, uint8_t *out,
                                       size_t len, const AES_KEY *key,
                                       uint8_t *ivec, int *num);

/* AES_cfb128_encrypt encrypts (or decrypts, if |enc| == |AES_DECRYPT|) |len|
 * bytes from |in| to |out|. The |num| parameter must be set to zero on the
 * first call. */
OPENSSL_EXPORT void AES_cfb128_encrypt(const uint8_t *in, uint8_t *out,
                                       size_t len, const AES_KEY *key,
                                       uint8_t *ivec, int *num, int enc);


/* Android compatibility section.
 *
 * These functions are declared, temporarily, for Android because
 * wpa_supplicant will take a little time to sync with upstream. Outside of
 * Android they'll have no definition. */

OPENSSL_EXPORT int AES_wrap_key(AES_KEY *key, const uint8_t *iv, uint8_t *out,
                                const uint8_t *in, unsigned in_len);
OPENSSL_EXPORT int AES_unwrap_key(AES_KEY *key, const uint8_t *iv, uint8_t *out,
                                  const uint8_t *in, unsigned in_len);


#if defined(__cplusplus)
}  /* extern C */
#endif

#endif  /* OPENSSL_HEADER_AES_H */
