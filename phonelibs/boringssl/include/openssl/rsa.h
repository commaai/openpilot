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

#ifndef OPENSSL_HEADER_RSA_H
#define OPENSSL_HEADER_RSA_H

#include <openssl/base.h>

#include <openssl/engine.h>
#include <openssl/ex_data.h>
#include <openssl/thread.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* rsa.h contains functions for handling encryption and signature using RSA. */


/* Allocation and destruction. */

/* RSA_new returns a new, empty RSA object or NULL on error. */
OPENSSL_EXPORT RSA *RSA_new(void);

/* RSA_new_method acts the same as |RSA_new| but takes an explicit |ENGINE|. */
OPENSSL_EXPORT RSA *RSA_new_method(const ENGINE *engine);

/* RSA_free decrements the reference count of |rsa| and frees it if the
 * reference count drops to zero. */
OPENSSL_EXPORT void RSA_free(RSA *rsa);

/* RSA_up_ref increments the reference count of |rsa|. */
OPENSSL_EXPORT int RSA_up_ref(RSA *rsa);


/* Key generation. */

/* RSA_generate_key_ex generates a new RSA key where the modulus has size
 * |bits| and the public exponent is |e|. If unsure, |RSA_F4| is a good value
 * for |e|. If |cb| is not NULL then it is called during the key generation
 * process. In addition to the calls documented for |BN_generate_prime_ex|, it
 * is called with event=2 when the n'th prime is rejected as unsuitable and
 * with event=3 when a suitable value for |p| is found.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int RSA_generate_key_ex(RSA *rsa, int bits, BIGNUM *e,
                                       BN_GENCB *cb);


/* Encryption / Decryption */

/* Padding types for encryption. */
#define RSA_PKCS1_PADDING 1
#define RSA_NO_PADDING 3
#define RSA_PKCS1_OAEP_PADDING 4
/* RSA_PKCS1_PSS_PADDING can only be used via the EVP interface. */
#define RSA_PKCS1_PSS_PADDING 6

/* RSA_encrypt encrypts |in_len| bytes from |in| to the public key from |rsa|
 * and writes, at most, |max_out| bytes of encrypted data to |out|. The
 * |max_out| argument must be, at least, |RSA_size| in order to ensure success.
 *
 * It returns 1 on success or zero on error.
 *
 * The |padding| argument must be one of the |RSA_*_PADDING| values. If in
 * doubt, |RSA_PKCS1_PADDING| is the most common but |RSA_PKCS1_OAEP_PADDING|
 * is the most secure. */
OPENSSL_EXPORT int RSA_encrypt(RSA *rsa, size_t *out_len, uint8_t *out,
                               size_t max_out, const uint8_t *in, size_t in_len,
                               int padding);

/* RSA_decrypt decrypts |in_len| bytes from |in| with the private key from
 * |rsa| and writes, at most, |max_out| bytes of plaintext to |out|. The
 * |max_out| argument must be, at least, |RSA_size| in order to ensure success.
 *
 * It returns 1 on success or zero on error.
 *
 * The |padding| argument must be one of the |RSA_*_PADDING| values. If in
 * doubt, |RSA_PKCS1_PADDING| is the most common but |RSA_PKCS1_OAEP_PADDING|
 * is the most secure. */
OPENSSL_EXPORT int RSA_decrypt(RSA *rsa, size_t *out_len, uint8_t *out,
                               size_t max_out, const uint8_t *in, size_t in_len,
                               int padding);

/* RSA_public_encrypt encrypts |flen| bytes from |from| to the public key in
 * |rsa| and writes the encrypted data to |to|. The |to| buffer must have at
 * least |RSA_size| bytes of space. It returns the number of bytes written, or
 * -1 on error. The |padding| argument must be one of the |RSA_*_PADDING|
 * values. If in doubt, |RSA_PKCS1_PADDING| is the most common but
 * |RSA_PKCS1_OAEP_PADDING| is the most secure.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. Use |RSA_encrypt| instead. */
OPENSSL_EXPORT int RSA_public_encrypt(int flen, const uint8_t *from,
                                      uint8_t *to, RSA *rsa, int padding);

/* RSA_private_decrypt decrypts |flen| bytes from |from| with the public key in
 * |rsa| and writes the plaintext to |to|. The |to| buffer must have at
 * least |RSA_size| bytes of space. It returns the number of bytes written, or
 * -1 on error. The |padding| argument must be one of the |RSA_*_PADDING|
 * values. If in doubt, |RSA_PKCS1_PADDING| is the most common but
 * |RSA_PKCS1_OAEP_PADDING| is the most secure.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. Use |RSA_decrypt| instead. */
OPENSSL_EXPORT int RSA_private_decrypt(int flen, const uint8_t *from,
                                       uint8_t *to, RSA *rsa, int padding);

/* RSA_message_index_PKCS1_type_2 performs the first step of a PKCS #1 padding
 * check for decryption. If the |from_len| bytes pointed to at |from| are a
 * valid PKCS #1 message, it returns one and sets |*out_index| to the start of
 * the unpadded message. The unpadded message is a suffix of the input and has
 * length |from_len - *out_index|. Otherwise, it returns zero and sets
 * |*out_index| to zero. This function runs in time independent of the input
 * data and is intended to be used directly to avoid Bleichenbacher's attack.
 *
 * WARNING: This function behaves differently from the usual OpenSSL convention
 * in that it does NOT put an error on the queue in the error case. */
OPENSSL_EXPORT int RSA_message_index_PKCS1_type_2(const uint8_t *from,
                                                  size_t from_len,
                                                  size_t *out_index);


/* Signing / Verification */

/* RSA_sign signs |in_len| bytes of digest from |in| with |rsa| and writes, at
 * most, |RSA_size(rsa)| bytes to |out|. On successful return, the actual
 * number of bytes written is written to |*out_len|.
 *
 * The |hash_nid| argument identifies the hash function used to calculate |in|
 * and is embedded in the resulting signature. For example, it might be
 * |NID_sha256|.
 *
 * It returns 1 on success and zero on error. */
OPENSSL_EXPORT int RSA_sign(int hash_nid, const uint8_t *in,
                            unsigned int in_len, uint8_t *out,
                            unsigned int *out_len, RSA *rsa);

/* RSA_sign_raw signs |in_len| bytes from |in| with the public key from |rsa|
 * and writes, at most, |max_out| bytes of signature data to |out|. The
 * |max_out| argument must be, at least, |RSA_size| in order to ensure success.
 *
 * It returns 1 on success or zero on error.
 *
 * The |padding| argument must be one of the |RSA_*_PADDING| values. If in
 * doubt, |RSA_PKCS1_PADDING| is the most common. */
OPENSSL_EXPORT int RSA_sign_raw(RSA *rsa, size_t *out_len, uint8_t *out,
                                size_t max_out, const uint8_t *in,
                                size_t in_len, int padding);

/* RSA_verify verifies that |sig_len| bytes from |sig| are a valid, PKCS#1
 * signature of |msg_len| bytes at |msg| by |rsa|.
 *
 * The |hash_nid| argument identifies the hash function used to calculate |in|
 * and is embedded in the resulting signature in order to prevent hash
 * confusion attacks. For example, it might be |NID_sha256|.
 *
 * It returns one if the signature is valid and zero otherwise.
 *
 * WARNING: this differs from the original, OpenSSL function which additionally
 * returned -1 on error. */
OPENSSL_EXPORT int RSA_verify(int hash_nid, const uint8_t *msg, size_t msg_len,
                              const uint8_t *sig, size_t sig_len, RSA *rsa);

/* RSA_verify_raw verifies |in_len| bytes of signature from |in| using the
 * public key from |rsa| and writes, at most, |max_out| bytes of plaintext to
 * |out|. The |max_out| argument must be, at least, |RSA_size| in order to
 * ensure success.
 *
 * It returns 1 on success or zero on error.
 *
 * The |padding| argument must be one of the |RSA_*_PADDING| values. If in
 * doubt, |RSA_PKCS1_PADDING| is the most common. */
OPENSSL_EXPORT int RSA_verify_raw(RSA *rsa, size_t *out_len, uint8_t *out,
                                  size_t max_out, const uint8_t *in,
                                  size_t in_len, int padding);

/* RSA_private_encrypt encrypts |flen| bytes from |from| with the private key in
 * |rsa| and writes the encrypted data to |to|. The |to| buffer must have at
 * least |RSA_size| bytes of space. It returns the number of bytes written, or
 * -1 on error. The |padding| argument must be one of the |RSA_*_PADDING|
 * values. If in doubt, |RSA_PKCS1_PADDING| is the most common.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. Use |RSA_sign_raw| instead. */
OPENSSL_EXPORT int RSA_private_encrypt(int flen, const uint8_t *from,
                                       uint8_t *to, RSA *rsa, int padding);

/* RSA_private_encrypt verifies |flen| bytes of signature from |from| using the
 * public key in |rsa| and writes the plaintext to |to|. The |to| buffer must
 * have at least |RSA_size| bytes of space. It returns the number of bytes
 * written, or -1 on error. The |padding| argument must be one of the
 * |RSA_*_PADDING| values. If in doubt, |RSA_PKCS1_PADDING| is the most common.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. Use |RSA_verify_raw| instead. */
OPENSSL_EXPORT int RSA_public_decrypt(int flen, const uint8_t *from,
                                      uint8_t *to, RSA *rsa, int padding);


/* Utility functions. */

/* RSA_size returns the number of bytes in the modulus, which is also the size
 * of a signature or encrypted value using |rsa|. */
OPENSSL_EXPORT unsigned RSA_size(const RSA *rsa);

/* RSA_is_opaque returns one if |rsa| is opaque and doesn't expose its key
 * material. Otherwise it returns zero. */
OPENSSL_EXPORT int RSA_is_opaque(const RSA *rsa);

/* RSA_supports_digest returns one if |rsa| supports signing digests
 * of type |md|. Otherwise it returns zero. */
OPENSSL_EXPORT int RSA_supports_digest(const RSA *rsa, const EVP_MD *md);

/* RSAPublicKey_dup allocates a fresh |RSA| and copies the private key from
 * |rsa| into it. It returns the fresh |RSA| object, or NULL on error. */
OPENSSL_EXPORT RSA *RSAPublicKey_dup(const RSA *rsa);

/* RSAPrivateKey_dup allocates a fresh |RSA| and copies the private key from
 * |rsa| into it. It returns the fresh |RSA| object, or NULL on error. */
OPENSSL_EXPORT RSA *RSAPrivateKey_dup(const RSA *rsa);

/* RSA_check_key performs basic validatity tests on |rsa|. It returns one if
 * they pass and zero otherwise. Opaque keys and public keys always pass. If it
 * returns zero then a more detailed error is available on the error queue. */
OPENSSL_EXPORT int RSA_check_key(const RSA *rsa);

/* RSA_recover_crt_params uses |rsa->n|, |rsa->d| and |rsa->e| in order to
 * calculate the two primes used and thus the precomputed, CRT values. These
 * values are set in the |p|, |q|, |dmp1|, |dmq1| and |iqmp| members of |rsa|,
 * which must be |NULL| on entry. It returns one on success and zero
 * otherwise. */
OPENSSL_EXPORT int RSA_recover_crt_params(RSA *rsa);

/* RSA_verify_PKCS1_PSS_mgf1 verifies that |EM| is a correct PSS padding of
 * |mHash|, where |mHash| is a digest produced by |Hash|. |EM| must point to
 * exactly |RSA_size(rsa)| bytes of data. The |mgf1Hash| argument specifies the
 * hash function for generating the mask. If NULL, |Hash| is used. The |sLen|
 * argument specifies the expected salt length in bytes. If |sLen| is -1 then
 * the salt length is the same as the hash length. If -2, then the salt length
 * is maximal and is taken from the size of |EM|.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int RSA_verify_PKCS1_PSS_mgf1(RSA *rsa, const uint8_t *mHash,
                                             const EVP_MD *Hash,
                                             const EVP_MD *mgf1Hash,
                                             const uint8_t *EM, int sLen);

/* RSA_padding_add_PKCS1_PSS_mgf1 writes a PSS padding of |mHash| to |EM|,
 * where |mHash| is a digest produced by |Hash|. |RSA_size(rsa)| bytes of
 * output will be written to |EM|. The |mgf1Hash| argument specifies the hash
 * function for generating the mask. If NULL, |Hash| is used. The |sLen|
 * argument specifies the expected salt length in bytes. If |sLen| is -1 then
 * the salt length is the same as the hash length. If -2, then the salt length
 * is maximal given the space in |EM|.
 *
 * It returns one on success or zero on error. */
OPENSSL_EXPORT int RSA_padding_add_PKCS1_PSS_mgf1(RSA *rsa, uint8_t *EM,
                                                  const uint8_t *mHash,
                                                  const EVP_MD *Hash,
                                                  const EVP_MD *mgf1Hash,
                                                  int sLen);


/* ASN.1 functions. */

/* d2i_RSAPublicKey parses an ASN.1, DER-encoded, RSA public key from |len|
 * bytes at |*inp|. If |out| is not NULL then, on exit, a pointer to the result
 * is in |*out|. If |*out| is already non-NULL on entry then the result is
 * written directly into |*out|, otherwise a fresh |RSA| is allocated. On
 * successful exit, |*inp| is advanced past the DER structure. It returns the
 * result or NULL on error. */
OPENSSL_EXPORT RSA *d2i_RSAPublicKey(RSA **out, const uint8_t **inp, long len);

/* i2d_RSAPublicKey marshals |in| to an ASN.1, DER structure. If |outp| is not
 * NULL then the result is written to |*outp| and |*outp| is advanced just past
 * the output. It returns the number of bytes in the result, whether written or
 * not, or a negative value on error. */
OPENSSL_EXPORT int i2d_RSAPublicKey(const RSA *in, uint8_t **outp);

/* d2i_RSAPrivateKey parses an ASN.1, DER-encoded, RSA private key from |len|
 * bytes at |*inp|. If |out| is not NULL then, on exit, a pointer to the result
 * is in |*out|. If |*out| is already non-NULL on entry then the result is
 * written directly into |*out|, otherwise a fresh |RSA| is allocated. On
 * successful exit, |*inp| is advanced past the DER structure. It returns the
 * result or NULL on error. */
OPENSSL_EXPORT RSA *d2i_RSAPrivateKey(RSA **out, const uint8_t **inp, long len);

/* i2d_RSAPrivateKey marshals |in| to an ASN.1, DER structure. If |outp| is not
 * NULL then the result is written to |*outp| and |*outp| is advanced just past
 * the output. It returns the number of bytes in the result, whether written or
 * not, or a negative value on error. */
OPENSSL_EXPORT int i2d_RSAPrivateKey(const RSA *in, uint8_t **outp);


/* ex_data functions.
 *
 * See |ex_data.h| for details. */

OPENSSL_EXPORT int RSA_get_ex_new_index(long argl, void *argp,
                                        CRYPTO_EX_new *new_func,
                                        CRYPTO_EX_dup *dup_func,
                                        CRYPTO_EX_free *free_func);
OPENSSL_EXPORT int RSA_set_ex_data(RSA *r, int idx, void *arg);
OPENSSL_EXPORT void *RSA_get_ex_data(const RSA *r, int idx);

/* RSA_FLAG_OPAQUE specifies that this RSA_METHOD does not expose its key
 * material. This may be set if, for instance, it is wrapping some other crypto
 * API, like a platform key store. */
#define RSA_FLAG_OPAQUE 1

/* RSA_FLAG_CACHE_PUBLIC causes a precomputed Montgomery context to be created,
 * on demand, for the public key operations. */
#define RSA_FLAG_CACHE_PUBLIC 2

/* RSA_FLAG_CACHE_PRIVATE causes a precomputed Montgomery context to be
 * created, on demand, for the private key operations. */
#define RSA_FLAG_CACHE_PRIVATE 4

/* RSA_FLAG_NO_BLINDING disables blinding of private operations. */
#define RSA_FLAG_NO_BLINDING 8

/* RSA_FLAG_EXT_PKEY means that private key operations will be handled by
 * |mod_exp| and that they do not depend on the private key components being
 * present: for example a key stored in external hardware. */
#define RSA_FLAG_EXT_PKEY 0x20

/* RSA_FLAG_SIGN_VER causes the |sign| and |verify| functions of |rsa_meth_st|
 * to be called when set. */
#define RSA_FLAG_SIGN_VER 0x40


/* RSA public exponent values. */

#define RSA_3 0x3
#define RSA_F4 0x10001


/* Deprecated functions. */

/* RSA_blinding_on returns one. */
OPENSSL_EXPORT int RSA_blinding_on(RSA *rsa, BN_CTX *ctx);


struct rsa_meth_st {
  struct openssl_method_common_st common;

  void *app_data;

  int (*init)(RSA *rsa);
  int (*finish)(RSA *rsa);

  /* size returns the size of the RSA modulus in bytes. */
  size_t (*size)(const RSA *rsa);

  int (*sign)(int type, const uint8_t *m, unsigned int m_length,
              uint8_t *sigret, unsigned int *siglen, const RSA *rsa);

  int (*verify)(int dtype, const uint8_t *m, unsigned int m_length,
                const uint8_t *sigbuf, unsigned int siglen, const RSA *rsa);


  /* These functions mirror the |RSA_*| functions of the same name. */
  int (*encrypt)(RSA *rsa, size_t *out_len, uint8_t *out, size_t max_out,
                 const uint8_t *in, size_t in_len, int padding);
  int (*sign_raw)(RSA *rsa, size_t *out_len, uint8_t *out, size_t max_out,
                  const uint8_t *in, size_t in_len, int padding);

  int (*decrypt)(RSA *rsa, size_t *out_len, uint8_t *out, size_t max_out,
                 const uint8_t *in, size_t in_len, int padding);
  int (*verify_raw)(RSA *rsa, size_t *out_len, uint8_t *out, size_t max_out,
                    const uint8_t *in, size_t in_len, int padding);

  /* private_transform takes a big-endian integer from |in|, calculates the
   * d'th power of it, modulo the RSA modulus and writes the result as a
   * big-endian integer to |out|. Both |in| and |out| are |len| bytes long and
   * |len| is always equal to |RSA_size(rsa)|. If the result of the transform
   * can be represented in fewer than |len| bytes, then |out| must be zero
   * padded on the left.
   *
   * It returns one on success and zero otherwise.
   *
   * RSA decrypt and sign operations will call this, thus an ENGINE might wish
   * to override it in order to avoid having to implement the padding
   * functionality demanded by those, higher level, operations. */
  int (*private_transform)(RSA *rsa, uint8_t *out, const uint8_t *in,
                           size_t len);

  int (*mod_exp)(BIGNUM *r0, const BIGNUM *I, RSA *rsa,
                 BN_CTX *ctx); /* Can be null */
  int (*bn_mod_exp)(BIGNUM *r, const BIGNUM *a, const BIGNUM *p,
                    const BIGNUM *m, BN_CTX *ctx,
                    BN_MONT_CTX *m_ctx);

  int flags;

  int (*keygen)(RSA *rsa, int bits, BIGNUM *e, BN_GENCB *cb);

  /* supports_digest returns one if |rsa| supports digests of type
   * |md|. If null, it is assumed that all digests are supported. */
  int (*supports_digest)(const RSA *rsa, const EVP_MD *md);
};


/* Private functions. */

typedef struct bn_blinding_st BN_BLINDING;

struct rsa_st {
  /* version is only used during ASN.1 (de)serialisation. */
  long version;
  RSA_METHOD *meth;

  BIGNUM *n;
  BIGNUM *e;
  BIGNUM *d;
  BIGNUM *p;
  BIGNUM *q;
  BIGNUM *dmp1;
  BIGNUM *dmq1;
  BIGNUM *iqmp;
  /* be careful using this if the RSA structure is shared */
  CRYPTO_EX_DATA ex_data;
  CRYPTO_refcount_t references;
  int flags;

  CRYPTO_MUTEX lock;

  /* Used to cache montgomery values. The creation of these values is protected
   * by |lock|. */
  BN_MONT_CTX *_method_mod_n;
  BN_MONT_CTX *_method_mod_p;
  BN_MONT_CTX *_method_mod_q;

  /* num_blindings contains the size of the |blindings| and |blindings_inuse|
   * arrays. This member and the |blindings_inuse| array are protected by
   * |lock|. */
  unsigned num_blindings;
  /* blindings is an array of BN_BLINDING structures that can be reserved by a
   * thread by locking |lock| and changing the corresponding element in
   * |blindings_inuse| from 0 to 1. */
  BN_BLINDING **blindings;
  unsigned char *blindings_inuse;
};


#if defined(__cplusplus)
}  /* extern C */
#endif

#define RSA_F_BN_BLINDING_convert_ex 100
#define RSA_F_BN_BLINDING_create_param 101
#define RSA_F_BN_BLINDING_invert_ex 102
#define RSA_F_BN_BLINDING_new 103
#define RSA_F_BN_BLINDING_update 104
#define RSA_F_RSA_check_key 105
#define RSA_F_RSA_new_method 106
#define RSA_F_RSA_padding_add_PKCS1_OAEP_mgf1 107
#define RSA_F_RSA_padding_add_PKCS1_PSS_mgf1 108
#define RSA_F_RSA_padding_add_PKCS1_type_1 109
#define RSA_F_RSA_padding_add_PKCS1_type_2 110
#define RSA_F_RSA_padding_add_none 111
#define RSA_F_RSA_padding_check_PKCS1_OAEP_mgf1 112
#define RSA_F_RSA_padding_check_PKCS1_type_1 113
#define RSA_F_RSA_padding_check_PKCS1_type_2 114
#define RSA_F_RSA_padding_check_none 115
#define RSA_F_RSA_recover_crt_params 116
#define RSA_F_RSA_sign 117
#define RSA_F_RSA_verify 118
#define RSA_F_RSA_verify_PKCS1_PSS_mgf1 119
#define RSA_F_decrypt 120
#define RSA_F_encrypt 121
#define RSA_F_keygen 122
#define RSA_F_pkcs1_prefixed_msg 123
#define RSA_F_private_transform 124
#define RSA_F_rsa_setup_blinding 125
#define RSA_F_sign_raw 126
#define RSA_F_verify_raw 127
#define RSA_R_BAD_E_VALUE 100
#define RSA_R_BAD_FIXED_HEADER_DECRYPT 101
#define RSA_R_BAD_PAD_BYTE_COUNT 102
#define RSA_R_BAD_RSA_PARAMETERS 103
#define RSA_R_BAD_SIGNATURE 104
#define RSA_R_BLOCK_TYPE_IS_NOT_01 105
#define RSA_R_BN_NOT_INITIALIZED 106
#define RSA_R_CRT_PARAMS_ALREADY_GIVEN 107
#define RSA_R_CRT_VALUES_INCORRECT 108
#define RSA_R_DATA_LEN_NOT_EQUAL_TO_MOD_LEN 109
#define RSA_R_DATA_TOO_LARGE 110
#define RSA_R_DATA_TOO_LARGE_FOR_KEY_SIZE 111
#define RSA_R_DATA_TOO_LARGE_FOR_MODULUS 112
#define RSA_R_DATA_TOO_SMALL 113
#define RSA_R_DATA_TOO_SMALL_FOR_KEY_SIZE 114
#define RSA_R_DIGEST_TOO_BIG_FOR_RSA_KEY 115
#define RSA_R_D_E_NOT_CONGRUENT_TO_1 116
#define RSA_R_EMPTY_PUBLIC_KEY 117
#define RSA_R_FIRST_OCTET_INVALID 118
#define RSA_R_INCONSISTENT_SET_OF_CRT_VALUES 119
#define RSA_R_INTERNAL_ERROR 120
#define RSA_R_INVALID_MESSAGE_LENGTH 121
#define RSA_R_KEY_SIZE_TOO_SMALL 122
#define RSA_R_LAST_OCTET_INVALID 123
#define RSA_R_MODULUS_TOO_LARGE 124
#define RSA_R_NO_PUBLIC_EXPONENT 125
#define RSA_R_NULL_BEFORE_BLOCK_MISSING 126
#define RSA_R_N_NOT_EQUAL_P_Q 127
#define RSA_R_OAEP_DECODING_ERROR 128
#define RSA_R_ONLY_ONE_OF_P_Q_GIVEN 129
#define RSA_R_OUTPUT_BUFFER_TOO_SMALL 130
#define RSA_R_PADDING_CHECK_FAILED 131
#define RSA_R_PKCS_DECODING_ERROR 132
#define RSA_R_SLEN_CHECK_FAILED 133
#define RSA_R_SLEN_RECOVERY_FAILED 134
#define RSA_R_TOO_LONG 135
#define RSA_R_TOO_MANY_ITERATIONS 136
#define RSA_R_UNKNOWN_ALGORITHM_TYPE 137
#define RSA_R_UNKNOWN_PADDING_TYPE 138
#define RSA_R_VALUE_MISSING 139
#define RSA_R_WRONG_SIGNATURE_LENGTH 140

#endif  /* OPENSSL_HEADER_RSA_H */
