/* Copyright (C) 1995-1997 Eric Young (eay@cryptsoft.com)
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
 * [including the GNU Public Licence.]
 */
/* ====================================================================
 * Copyright (c) 1998-2006 The OpenSSL Project.  All rights reserved.
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
 * The Contribution is licensed pursuant to the Eric Young open source
 * license provided above.
 *
 * The binary polynomial arithmetic software is originally written by
 * Sheueling Chang Shantz and Douglas Stebila of Sun Microsystems
 * Laboratories. */

#ifndef OPENSSL_HEADER_BN_H
#define OPENSSL_HEADER_BN_H

#include <openssl/base.h>
#include <openssl/thread.h>

#include <inttypes.h>  /* for PRIu64 and friends */
#include <stdio.h>  /* for FILE* */

#if defined(__cplusplus)
extern "C" {
#endif


/* BN provides support for working with arbitary sized integers. For example,
 * although the largest integer supported by the compiler might be 64 bits, BN
 * will allow you to work with numbers until you run out of memory. */


/* BN_ULONG is the native word size when working with big integers.
 *
 * Note: on some platforms, inttypes.h does not define print format macros in
 * C++ unless |__STDC_FORMAT_MACROS| defined. As this is a public header, bn.h
 * does not define |__STDC_FORMAT_MACROS| itself. C++ source files which use the
 * FMT macros must define it externally. */
#if defined(OPENSSL_64_BIT)
#define BN_ULONG uint64_t
#define BN_BITS2 64
#define BN_DEC_FMT1	"%" PRIu64
#define BN_DEC_FMT2	"%019" PRIu64
#define BN_HEX_FMT1	"%" PRIx64
#elif defined(OPENSSL_32_BIT)
#define BN_ULONG uint32_t
#define BN_BITS2 32
#define BN_DEC_FMT1	"%" PRIu32
#define BN_DEC_FMT2	"%09" PRIu32
#define BN_HEX_FMT1	"%" PRIx32
#else
#error "Must define either OPENSSL_32_BIT or OPENSSL_64_BIT"
#endif


/* Allocation and freeing. */

/* BN_new creates a new, allocated BIGNUM and initialises it. */
OPENSSL_EXPORT BIGNUM *BN_new(void);

/* BN_init initialises a stack allocated |BIGNUM|. */
OPENSSL_EXPORT void BN_init(BIGNUM *bn);

/* BN_free frees the data referenced by |bn| and, if |bn| was originally
 * allocated on the heap, frees |bn| also. */
OPENSSL_EXPORT void BN_free(BIGNUM *bn);

/* BN_clear_free erases and frees the data referenced by |bn| and, if |bn| was
 * originally allocated on the heap, frees |bn| also. */
OPENSSL_EXPORT void BN_clear_free(BIGNUM *bn);

/* BN_dup allocates a new BIGNUM and sets it equal to |src|. It returns the
 * allocated BIGNUM on success or NULL otherwise. */
OPENSSL_EXPORT BIGNUM *BN_dup(const BIGNUM *src);

/* BN_copy sets |dest| equal to |src| and returns |dest|. */
OPENSSL_EXPORT BIGNUM *BN_copy(BIGNUM *dest, const BIGNUM *src);

/* BN_clear sets |bn| to zero and erases the old data. */
OPENSSL_EXPORT void BN_clear(BIGNUM *bn);

/* BN_value_one returns a static BIGNUM with value 1. */
OPENSSL_EXPORT const BIGNUM *BN_value_one(void);

/* BN_with_flags initialises a stack allocated |BIGNUM| with pointers to the
 * contents of |in| but with |flags| ORed into the flags field.
 *
 * Note: the two BIGNUMs share state and so |out| should /not/ be passed to
 * |BN_free|. */
OPENSSL_EXPORT void BN_with_flags(BIGNUM *out, const BIGNUM *in, int flags);


/* Basic functions. */

/* BN_num_bits returns the minimum number of bits needed to represent the
 * absolute value of |bn|. */
OPENSSL_EXPORT unsigned BN_num_bits(const BIGNUM *bn);

/* BN_num_bytes returns the minimum number of bytes needed to represent the
 * absolute value of |bn|. */
OPENSSL_EXPORT unsigned BN_num_bytes(const BIGNUM *bn);

/* BN_zero sets |bn| to zero. */
OPENSSL_EXPORT void BN_zero(BIGNUM *bn);

/* BN_one sets |bn| to one. It returns one on success or zero on allocation
 * failure. */
OPENSSL_EXPORT int BN_one(BIGNUM *bn);

/* BN_set_word sets |bn| to |value|. It returns one on success or zero on
 * allocation failure. */
OPENSSL_EXPORT int BN_set_word(BIGNUM *bn, BN_ULONG value);

/* BN_set_negative sets the sign of |bn|. */
OPENSSL_EXPORT void BN_set_negative(BIGNUM *bn, int sign);

/* BN_is_negative returns one if |bn| is negative and zero otherwise. */
OPENSSL_EXPORT int BN_is_negative(const BIGNUM *bn);

/* BN_get_flags returns |bn->flags| & |flags|. */
OPENSSL_EXPORT int BN_get_flags(const BIGNUM *bn, int flags);

/* BN_set_flags sets |flags| on |bn|. */
OPENSSL_EXPORT void BN_set_flags(BIGNUM *bn, int flags);


/* Conversion functions. */

/* BN_bin2bn sets |*ret| to the value of |len| bytes from |in|, interpreted as
 * a big-endian number, and returns |ret|. If |ret| is NULL then a fresh
 * |BIGNUM| is allocated and returned. It returns NULL on allocation
 * failure. */
OPENSSL_EXPORT BIGNUM *BN_bin2bn(const uint8_t *in, size_t len, BIGNUM *ret);

/* BN_bn2bin serialises the absolute value of |in| to |out| as a big-endian
 * integer, which must have |BN_num_bytes| of space available. It returns the
 * number of bytes written. */
OPENSSL_EXPORT size_t BN_bn2bin(const BIGNUM *in, uint8_t *out);

/* BN_bn2bin_padded serialises the absolute value of |in| to |out| as a
 * big-endian integer. The integer is padded with leading zeros up to size
 * |len|. If |len| is smaller than |BN_num_bytes|, the function fails and
 * returns 0. Otherwise, it returns 1. */
OPENSSL_EXPORT int BN_bn2bin_padded(uint8_t *out, size_t len, const BIGNUM *in);

/* BN_bn2hex returns an allocated string that contains a NUL-terminated, hex
 * representation of |bn|. If |bn| is negative, the first char in the resulting
 * string will be '-'. Returns NULL on allocation failure. */
OPENSSL_EXPORT char *BN_bn2hex(const BIGNUM *bn);

/* BN_hex2bn parses the leading hex number from |in|, which may be proceeded by
 * a '-' to indicate a negative number and may contain trailing, non-hex data.
 * If |outp| is not NULL, it constructs a BIGNUM equal to the hex number and
 * stores it in |*outp|. If |*outp| is NULL then it allocates a new BIGNUM and
 * updates |*outp|. It returns the number of bytes of |in| processed or zero on
 * error. */
OPENSSL_EXPORT int BN_hex2bn(BIGNUM **outp, const char *in);

/* BN_bn2dec returns an allocated string that contains a NUL-terminated,
 * decimal representation of |bn|. If |bn| is negative, the first char in the
 * resulting string will be '-'. Returns NULL on allocation failure. */
OPENSSL_EXPORT char *BN_bn2dec(const BIGNUM *a);

/* BN_dec2bn parses the leading decimal number from |in|, which may be
 * proceeded by a '-' to indicate a negative number and may contain trailing,
 * non-decimal data. If |outp| is not NULL, it constructs a BIGNUM equal to the
 * decimal number and stores it in |*outp|. If |*outp| is NULL then it
 * allocates a new BIGNUM and updates |*outp|. It returns the number of bytes
 * of |in| processed or zero on error. */
OPENSSL_EXPORT int BN_dec2bn(BIGNUM **outp, const char *in);

/* BN_asc2bn acts like |BN_dec2bn| or |BN_hex2bn| depending on whether |in|
 * begins with "0X" or "0x" (indicating hex) or not (indicating decimal). A
 * leading '-' is still permitted and comes before the optional 0X/0x. It
 * returns one on success or zero on error. */
OPENSSL_EXPORT int BN_asc2bn(BIGNUM **outp, const char *in);

/* BN_print writes a hex encoding of |a| to |bio|. It returns one on success
 * and zero on error. */
OPENSSL_EXPORT int BN_print(BIO *bio, const BIGNUM *a);

/* BN_print_fp acts like |BIO_print|, but wraps |fp| in a |BIO| first. */
OPENSSL_EXPORT int BN_print_fp(FILE *fp, const BIGNUM *a);

/* BN_get_word returns the absolute value of |bn| as a single word. If |bn| is
 * too large to be represented as a single word, the maximum possible value
 * will be returned. */
OPENSSL_EXPORT BN_ULONG BN_get_word(const BIGNUM *bn);


/* Internal functions.
 *
 * These functions are useful for code that is doing low-level manipulations of
 * BIGNUM values. However, be sure that no other function in this file does
 * what you want before turning to these. */

/* bn_correct_top decrements |bn->top| until |bn->d[top-1]| is non-zero or
 * until |top| is zero. */
OPENSSL_EXPORT void bn_correct_top(BIGNUM *bn);

/* bn_wexpand ensures that |bn| has at least |words| works of space without
 * altering its value. It returns one on success or zero on allocation
 * failure. */
OPENSSL_EXPORT BIGNUM *bn_wexpand(BIGNUM *bn, unsigned words);


/* BIGNUM pools.
 *
 * Certain BIGNUM operations need to use many temporary variables and
 * allocating and freeing them can be quite slow. Thus such opertions typically
 * take a |BN_CTX| parameter, which contains a pool of |BIGNUMs|. The |ctx|
 * argument to a public function may be NULL, in which case a local |BN_CTX|
 * will be created just for the lifetime of that call.
 *
 * A function must call |BN_CTX_start| first. Then, |BN_CTX_get| may be called
 * repeatedly to obtain temporary |BIGNUM|s. All |BN_CTX_get| calls must be made
 * before calling any other functions that use the |ctx| as an argument.
 *
 * Finally, |BN_CTX_end| must be called before returning from the function.
 * When |BN_CTX_end| is called, the |BIGNUM| pointers obtained from
 * |BN_CTX_get| become invalid. */

/* BN_CTX_new returns a new, empty BN_CTX or NULL on allocation failure. */
OPENSSL_EXPORT BN_CTX *BN_CTX_new(void);

/* BN_CTX_free frees all BIGNUMs contained in |ctx| and then frees |ctx|
 * itself. */
OPENSSL_EXPORT void BN_CTX_free(BN_CTX *ctx);

/* BN_CTX_start "pushes" a new entry onto the |ctx| stack and allows future
 * calls to |BN_CTX_get|. */
OPENSSL_EXPORT void BN_CTX_start(BN_CTX *ctx);

/* BN_CTX_get returns a new |BIGNUM|, or NULL on allocation failure. Once
 * |BN_CTX_get| has returned NULL, all future calls will also return NULL until
 * |BN_CTX_end| is called. */
OPENSSL_EXPORT BIGNUM *BN_CTX_get(BN_CTX *ctx);

/* BN_CTX_end invalidates all |BIGNUM|s returned from |BN_CTX_get| since the
 * matching |BN_CTX_start| call. */
OPENSSL_EXPORT void BN_CTX_end(BN_CTX *ctx);


/* Simple arithmetic */

/* BN_add sets |r| = |a| + |b|, where |r| may be the same pointer as either |a|
 * or |b|. It returns one on success and zero on allocation failure. */
OPENSSL_EXPORT int BN_add(BIGNUM *r, const BIGNUM *a, const BIGNUM *b);

/* BN_uadd sets |r| = |a| + |b|, where |a| and |b| are non-negative and |r| may
 * be the same pointer as either |a| or |b|. It returns one on success and zero
 * on allocation failure. */
OPENSSL_EXPORT int BN_uadd(BIGNUM *r, const BIGNUM *a, const BIGNUM *b);

/* BN_add_word adds |w| to |a|. It returns one on success and zero otherwise. */
OPENSSL_EXPORT int BN_add_word(BIGNUM *a, BN_ULONG w);

/* BN_sub sets |r| = |a| - |b|, where |r| must be a distinct pointer from |a|
 * and |b|. It returns one on success and zero on allocation failure. */
OPENSSL_EXPORT int BN_sub(BIGNUM *r, const BIGNUM *a, const BIGNUM *b);

/* BN_usub sets |r| = |a| - |b|, where |a| and |b| are non-negative integers,
 * |b| < |a| and |r| must be a distinct pointer from |a| and |b|. It returns
 * one on success and zero on allocation failure. */
OPENSSL_EXPORT int BN_usub(BIGNUM *r, const BIGNUM *a, const BIGNUM *b);

/* BN_sub_word subtracts |w| from |a|. It returns one on success and zero on
 * allocation failure. */
OPENSSL_EXPORT int BN_sub_word(BIGNUM *a, BN_ULONG w);

/* BN_mul sets |r| = |a| * |b|, where |r| may be the same pointer as |a| or
 * |b|. Returns one on success and zero otherwise. */
OPENSSL_EXPORT int BN_mul(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
                          BN_CTX *ctx);

/* BN_mul_word sets |bn| = |bn| * |w|. It returns one on success or zero on
 * allocation failure. */
OPENSSL_EXPORT int BN_mul_word(BIGNUM *bn, BN_ULONG w);

/* BN_sqr sets |r| = |a|^2 (i.e. squares), where |r| may be the same pointer as
 * |a|. Returns one on success and zero otherwise. This is more efficient than
 * BN_mul(r, a, a, ctx). */
OPENSSL_EXPORT int BN_sqr(BIGNUM *r, const BIGNUM *a, BN_CTX *ctx);

/* BN_div divides |numerator| by |divisor| and places the result in |quotient|
 * and the remainder in |rem|. Either of |quotient| or |rem| may be NULL, in
 * which case the respective value is not returned. The result is rounded
 * towards zero; thus if |numerator| is negative, the remainder will be zero or
 * negative. It returns one on success or zero on error. */
OPENSSL_EXPORT int BN_div(BIGNUM *quotient, BIGNUM *rem,
                          const BIGNUM *numerator, const BIGNUM *divisor,
                          BN_CTX *ctx);

/* BN_div_word sets |numerator| = |numerator|/|divisor| and returns the
 * remainder or (BN_ULONG)-1 on error. */
OPENSSL_EXPORT BN_ULONG BN_div_word(BIGNUM *numerator, BN_ULONG divisor);

/* BN_sqrt sets |*out_sqrt| (which may be the same |BIGNUM| as |in|) to the
 * square root of |in|, using |ctx|. It returns one on success or zero on
 * error. Negative numbers and non-square numbers will result in an error with
 * appropriate errors on the error queue. */
OPENSSL_EXPORT int BN_sqrt(BIGNUM *out_sqrt, const BIGNUM *in, BN_CTX *ctx);


/* Comparison functions */

/* BN_cmp returns a value less than, equal to or greater than zero if |a| is
 * less than, equal to or greater than |b|, respectively. */
OPENSSL_EXPORT int BN_cmp(const BIGNUM *a, const BIGNUM *b);

/* BN_ucmp returns a value less than, equal to or greater than zero if the
 * absolute value of |a| is less than, equal to or greater than the absolute
 * value of |b|, respectively. */
OPENSSL_EXPORT int BN_ucmp(const BIGNUM *a, const BIGNUM *b);

/* BN_abs_is_word returns one if the absolute value of |bn| equals |w| and zero
 * otherwise. */
OPENSSL_EXPORT int BN_abs_is_word(const BIGNUM *bn, BN_ULONG w);

/* BN_is_zero returns one if |bn| is zero and zero otherwise. */
OPENSSL_EXPORT int BN_is_zero(const BIGNUM *bn);

/* BN_is_one returns one if |bn| equals one and zero otherwise. */
OPENSSL_EXPORT int BN_is_one(const BIGNUM *bn);

/* BN_is_word returns one if |bn| is exactly |w| and zero otherwise. */
OPENSSL_EXPORT int BN_is_word(const BIGNUM *bn, BN_ULONG w);

/* BN_is_odd returns one if |bn| is odd and zero otherwise. */
OPENSSL_EXPORT int BN_is_odd(const BIGNUM *bn);


/* Bitwise operations. */

/* BN_lshift sets |r| equal to |a| << n. The |a| and |r| arguments may be the
 * same |BIGNUM|. It returns one on success and zero on allocation failure. */
OPENSSL_EXPORT int BN_lshift(BIGNUM *r, const BIGNUM *a, int n);

/* BN_lshift1 sets |r| equal to |a| << 1, where |r| and |a| may be the same
 * pointer. It returns one on success and zero on allocation failure. */
OPENSSL_EXPORT int BN_lshift1(BIGNUM *r, const BIGNUM *a);

/* BN_rshift sets |r| equal to |a| >> n, where |r| and |a| may be the same
 * pointer. It returns one on success and zero on allocation failure. */
OPENSSL_EXPORT int BN_rshift(BIGNUM *r, const BIGNUM *a, int n);

/* BN_rshift1 sets |r| equal to |a| >> 1, where |r| and |a| may be the same
 * pointer. It returns one on success and zero on allocation failure. */
OPENSSL_EXPORT int BN_rshift1(BIGNUM *r, const BIGNUM *a);

/* BN_set_bit sets the |n|th, least-significant bit in |a|. For example, if |a|
 * is 2 then setting bit zero will make it 3. It returns one on success or zero
 * on allocation failure. */
OPENSSL_EXPORT int BN_set_bit(BIGNUM *a, int n);

/* BN_clear_bit clears the |n|th, least-significant bit in |a|. For example, if
 * |a| is 3, clearing bit zero will make it two. It returns one on success or
 * zero on allocation failure. */
OPENSSL_EXPORT int BN_clear_bit(BIGNUM *a, int n);

/* BN_is_bit_set returns the value of the |n|th, least-significant bit in |a|,
 * or zero if the bit doesn't exist. */
OPENSSL_EXPORT int BN_is_bit_set(const BIGNUM *a, int n);

/* BN_mask_bits truncates |a| so that it is only |n| bits long. It returns one
 * on success or zero if |n| is greater than the length of |a| already. */
OPENSSL_EXPORT int BN_mask_bits(BIGNUM *a, int n);


/* Modulo arithmetic. */

/* BN_mod_word returns |a| mod |w|. */
OPENSSL_EXPORT BN_ULONG BN_mod_word(const BIGNUM *a, BN_ULONG w);

/* BN_mod is a helper macro that calls |BN_div| and discards the quotient. */
#define BN_mod(rem, numerator, divisor, ctx) \
  BN_div(NULL, (rem), (numerator), (divisor), (ctx))

/* BN_nnmod is a non-negative modulo function. It acts like |BN_mod|, but 0 <=
 * |rem| < |divisor| is always true. It returns one on success and zero on
 * error. */
OPENSSL_EXPORT int BN_nnmod(BIGNUM *rem, const BIGNUM *numerator,
                            const BIGNUM *divisor, BN_CTX *ctx);

/* BN_mod_add sets |r| = |a| + |b| mod |m|. It returns one on success and zero
 * on error. */
OPENSSL_EXPORT int BN_mod_add(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
                              const BIGNUM *m, BN_CTX *ctx);

/* BN_mod_add_quick acts like |BN_mod_add| but requires that |a| and |b| be
 * non-negative and less than |m|. */
OPENSSL_EXPORT int BN_mod_add_quick(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
                                    const BIGNUM *m);

/* BN_mod_sub sets |r| = |a| - |b| mod |m|. It returns one on success and zero
 * on error. */
OPENSSL_EXPORT int BN_mod_sub(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
                              const BIGNUM *m, BN_CTX *ctx);

/* BN_mod_sub_quick acts like |BN_mod_sub| but requires that |a| and |b| be
 * non-negative and less than |m|. */
OPENSSL_EXPORT int BN_mod_sub_quick(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
                                    const BIGNUM *m);

/* BN_mod_mul sets |r| = |a|*|b| mod |m|. It returns one on success and zero
 * on error. */
OPENSSL_EXPORT int BN_mod_mul(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
                              const BIGNUM *m, BN_CTX *ctx);

/* BN_mod_mul sets |r| = |a|^2 mod |m|. It returns one on success and zero
 * on error. */
OPENSSL_EXPORT int BN_mod_sqr(BIGNUM *r, const BIGNUM *a, const BIGNUM *m,
                              BN_CTX *ctx);

/* BN_mod_lshift sets |r| = (|a| << n) mod |m|, where |r| and |a| may be the
 * same pointer. It returns one on success and zero on error. */
OPENSSL_EXPORT int BN_mod_lshift(BIGNUM *r, const BIGNUM *a, int n,
                                 const BIGNUM *m, BN_CTX *ctx);

/* BN_mod_lshift_quick acts like |BN_mod_lshift| but requires that |a| be
 * non-negative and less than |m|. */
OPENSSL_EXPORT int BN_mod_lshift_quick(BIGNUM *r, const BIGNUM *a, int n,
                                       const BIGNUM *m);

/* BN_mod_lshift1 sets |r| = (|a| << 1) mod |m|, where |r| and |a| may be the
 * same pointer. It returns one on success and zero on error. */
OPENSSL_EXPORT int BN_mod_lshift1(BIGNUM *r, const BIGNUM *a, const BIGNUM *m,
                                  BN_CTX *ctx);

/* BN_mod_lshift1_quick acts like |BN_mod_lshift1| but requires that |a| be
 * non-negative and less than |m|. */
OPENSSL_EXPORT int BN_mod_lshift1_quick(BIGNUM *r, const BIGNUM *a,
                                        const BIGNUM *m);

/* BN_mod_sqrt returns a |BIGNUM|, r, such that r^2 == a (mod p). */
OPENSSL_EXPORT BIGNUM *BN_mod_sqrt(BIGNUM *in, const BIGNUM *a, const BIGNUM *p,
                                   BN_CTX *ctx);


/* Random and prime number generation. */

/* BN_rand sets |rnd| to a random number of length |bits|. If |top| is zero, the
 * most-significant bit, if any, will be set. If |top| is one, the two most
 * significant bits, if any, will be set.
 *
 * If |top| is -1 then no extra action will be taken and |BN_num_bits(rnd)| may
 * not equal |bits| if the most significant bits randomly ended up as zeros.
 *
 * If |bottom| is non-zero, the least-significant bit, if any, will be set. The
 * function returns one on success or zero otherwise. */
OPENSSL_EXPORT int BN_rand(BIGNUM *rnd, int bits, int top, int bottom);

/* BN_pseudo_rand is an alias for |BN_rand|. */
OPENSSL_EXPORT int BN_pseudo_rand(BIGNUM *rnd, int bits, int top, int bottom);

/* BN_rand_range sets |rnd| to a random value [0..range). It returns one on
 * success and zero otherwise. */
OPENSSL_EXPORT int BN_rand_range(BIGNUM *rnd, const BIGNUM *range);

/* BN_pseudo_rand_range is an alias for BN_rand_range. */
OPENSSL_EXPORT int BN_pseudo_rand_range(BIGNUM *rnd, const BIGNUM *range);

/* BN_generate_dsa_nonce generates a random number 0 <= out < range. Unlike
 * BN_rand_range, it also includes the contents of |priv| and |message| in the
 * generation so that an RNG failure isn't fatal as long as |priv| remains
 * secret. This is intended for use in DSA and ECDSA where an RNG weakness
 * leads directly to private key exposure unless this function is used.
 * It returns one on success and zero on error. */
OPENSSL_EXPORT int BN_generate_dsa_nonce(BIGNUM *out, const BIGNUM *range,
                                         const BIGNUM *priv,
                                         const uint8_t *message,
                                         size_t message_len, BN_CTX *ctx);

/* BN_GENCB holds a callback function that is used by generation functions that
 * can take a very long time to complete. Use |BN_GENCB_set| to initialise a
 * |BN_GENCB| structure.
 *
 * The callback receives the address of that |BN_GENCB| structure as its last
 * argument and the user is free to put an arbitary pointer in |arg|. The other
 * arguments are set as follows:
 *   event=BN_GENCB_GENERATED, n=i:   after generating the i'th possible prime
 *                                    number.
 *   event=BN_GENCB_PRIME_TEST, n=-1: when finished trial division primality
 *                                    checks.
 *   event=BN_GENCB_PRIME_TEST, n=i:  when the i'th primality test has finished.
 *
 * The callback can return zero to abort the generation progress or one to
 * allow it to continue.
 *
 * When other code needs to call a BN generation function it will often take a
 * BN_GENCB argument and may call the function with other argument values. */
#define BN_GENCB_GENERATED 0
#define BN_GENCB_PRIME_TEST 1

struct bn_gencb_st {
  void *arg;        /* callback-specific data */
  int (*callback)(int event, int n, struct bn_gencb_st *);
};

/* BN_GENCB_set configures |callback| to call |f| and sets |callout->arg| to
 * |arg|. */
OPENSSL_EXPORT void BN_GENCB_set(BN_GENCB *callback,
                                 int (*f)(int event, int n,
                                          struct bn_gencb_st *),
                                 void *arg);

/* BN_GENCB_call calls |callback|, if not NULL, and returns the return value of
 * the callback, or 1 if |callback| is NULL. */
OPENSSL_EXPORT int BN_GENCB_call(BN_GENCB *callback, int event, int n);

/* BN_generate_prime_ex sets |ret| to a prime number of |bits| length. If safe
 * is non-zero then the prime will be such that (ret-1)/2 is also a prime.
 * (This is needed for Diffie-Hellman groups to ensure that the only subgroups
 * are of size 2 and (p-1)/2.).
 *
 * If |add| is not NULL, the prime will fulfill the condition |ret| % |add| ==
 * |rem| in order to suit a given generator. (If |rem| is NULL then |ret| %
 * |add| == 1.)
 *
 * If |cb| is not NULL, it will be called during processing to give an
 * indication of progress. See the comments for |BN_GENCB|. It returns one on
 * success and zero otherwise. */
OPENSSL_EXPORT int BN_generate_prime_ex(BIGNUM *ret, int bits, int safe,
                                        const BIGNUM *add, const BIGNUM *rem,
                                        BN_GENCB *cb);

/* BN_prime_checks is magic value that can be used as the |checks| argument to
 * the primality testing functions in order to automatically select a number of
 * Miller-Rabin checks that gives a false positive rate of ~2^{-80}. */
#define BN_prime_checks 0

/* BN_primality_test sets |*is_probably_prime| to one if |candidate| is
 * probably a prime number by the Miller-Rabin test or zero if it's certainly
 * not.
 *
 * If |do_trial_division| is non-zero then |candidate| will be tested against a
 * list of small primes before Miller-Rabin tests. The probability of this
 * function returning a false positive is 2^{2*checks}. If |checks| is
 * |BN_prime_checks| then a value that results in approximately 2^{-80} false
 * positive probability is used. If |cb| is not NULL then it is called during
 * the checking process. See the comment above |BN_GENCB|.
 *
 * The function returns one on success and zero on error.
 *
 * (If you are unsure whether you want |do_trial_division|, don't set it.) */
OPENSSL_EXPORT int BN_primality_test(int *is_probably_prime,
                                     const BIGNUM *candidate, int checks,
                                     BN_CTX *ctx, int do_trial_division,
                                     BN_GENCB *cb);

/* BN_is_prime_fasttest_ex returns one if |candidate| is probably a prime
 * number by the Miller-Rabin test, zero if it's certainly not and -1 on error.
 *
 * If |do_trial_division| is non-zero then |candidate| will be tested against a
 * list of small primes before Miller-Rabin tests. The probability of this
 * function returning one when |candidate| is composite is 2^{2*checks}. If
 * |checks| is |BN_prime_checks| then a value that results in approximately
 * 2^{-80} false positive probability is used. If |cb| is not NULL then it is
 * called during the checking process. See the comment above |BN_GENCB|.
 *
 * WARNING: deprecated. Use |BN_primality_test|. */
OPENSSL_EXPORT int BN_is_prime_fasttest_ex(const BIGNUM *candidate, int checks,
                                           BN_CTX *ctx, int do_trial_division,
                                           BN_GENCB *cb);

/* BN_is_prime_ex acts the same as |BN_is_prime_fasttest_ex| with
 * |do_trial_division| set to zero.
 *
 * WARNING: deprecated: Use |BN_primality_test|. */
OPENSSL_EXPORT int BN_is_prime_ex(const BIGNUM *candidate, int checks,
                                  BN_CTX *ctx, BN_GENCB *cb);


/* Number theory functions */

/* BN_gcd sets |r| = gcd(|a|, |b|). It returns one on success and zero
 * otherwise. */
OPENSSL_EXPORT int BN_gcd(BIGNUM *r, const BIGNUM *a, const BIGNUM *b,
                          BN_CTX *ctx);

/* BN_mod_inverse sets |out| equal to |a|^-1, mod |n|. If either of |a| or |n|
 * have |BN_FLG_CONSTTIME| set then the operation is performed in constant
 * time. If |out| is NULL, a fresh BIGNUM is allocated. It returns the result
 * or NULL on error. */
OPENSSL_EXPORT BIGNUM *BN_mod_inverse(BIGNUM *out, const BIGNUM *a,
                                      const BIGNUM *n, BN_CTX *ctx);

/* BN_kronecker returns the Kronecker symbol of |a| and |b| (which is -1, 0 or
 * 1), or -2 on error. */
OPENSSL_EXPORT int BN_kronecker(const BIGNUM *a, const BIGNUM *b, BN_CTX *ctx);


/* Montgomery arithmetic. */

/* BN_MONT_CTX contains the precomputed values needed to work in a specific
 * Montgomery domain. */

/* BN_MONT_CTX_new returns a fresh BN_MONT_CTX or NULL on allocation failure. */
OPENSSL_EXPORT BN_MONT_CTX *BN_MONT_CTX_new(void);

/* BN_MONT_CTX_init initialises a stack allocated |BN_MONT_CTX|. */
OPENSSL_EXPORT void BN_MONT_CTX_init(BN_MONT_CTX *mont);

/* BN_MONT_CTX_free frees the contexts of |mont| and, if it was originally
 * allocated with |BN_MONT_CTX_new|, |mont| itself. */
OPENSSL_EXPORT void BN_MONT_CTX_free(BN_MONT_CTX *mont);

/* BN_MONT_CTX_copy sets |to| equal to |from|. It returns |to| on success or
 * NULL on error. */
OPENSSL_EXPORT BN_MONT_CTX *BN_MONT_CTX_copy(BN_MONT_CTX *to,
                                             BN_MONT_CTX *from);

/* BN_MONT_CTX_set sets up a Montgomery context given the modulus, |mod|. It
 * returns one on success and zero on error. */
OPENSSL_EXPORT int BN_MONT_CTX_set(BN_MONT_CTX *mont, const BIGNUM *mod,
                                   BN_CTX *ctx);

/* BN_MONT_CTX_set_locked takes |lock| and checks whether |*pmont| is NULL. If
 * so, it creates a new |BN_MONT_CTX| and sets the modulus for it to |mod|. It
 * then stores it as |*pmont| and returns it, or NULL on error.
 *
 * If |*pmont| is already non-NULL then the existing value is returned. */
BN_MONT_CTX *BN_MONT_CTX_set_locked(BN_MONT_CTX **pmont, CRYPTO_MUTEX *lock,
                                    const BIGNUM *mod, BN_CTX *bn_ctx);

/* BN_to_montgomery sets |ret| equal to |a| in the Montgomery domain. It
 * returns one on success and zero on error. */
OPENSSL_EXPORT int BN_to_montgomery(BIGNUM *ret, const BIGNUM *a,
                                    const BN_MONT_CTX *mont, BN_CTX *ctx);

/* BN_from_montgomery sets |ret| equal to |a| * R^-1, i.e. translates values
 * out of the Montgomery domain. It returns one on success or zero on error. */
OPENSSL_EXPORT int BN_from_montgomery(BIGNUM *ret, const BIGNUM *a,
                                      const BN_MONT_CTX *mont, BN_CTX *ctx);

/* BN_mod_mul_montgomery set |r| equal to |a| * |b|, in the Montgomery domain.
 * Both |a| and |b| must already be in the Montgomery domain (by
 * |BN_to_montgomery|). It returns one on success or zero on error. */
OPENSSL_EXPORT int BN_mod_mul_montgomery(BIGNUM *r, const BIGNUM *a,
                                         const BIGNUM *b,
                                         const BN_MONT_CTX *mont, BN_CTX *ctx);


/* Exponentiation. */

/* BN_exp sets |r| equal to |a|^{|p|}. It does so with a square-and-multiply
 * algorithm that leaks side-channel information. It returns one on success or
 * zero otherwise. */
OPENSSL_EXPORT int BN_exp(BIGNUM *r, const BIGNUM *a, const BIGNUM *p,
                          BN_CTX *ctx);

/* BN_mod_exp sets |r| equal to |a|^{|p|} mod |m|. It does so with the best
 * algorithm for the values provided and can run in constant time if
 * |BN_FLG_CONSTTIME| is set for |p|. It returns one on success or zero
 * otherwise. */
OPENSSL_EXPORT int BN_mod_exp(BIGNUM *r, const BIGNUM *a, const BIGNUM *p,
                              const BIGNUM *m, BN_CTX *ctx);

OPENSSL_EXPORT int BN_mod_exp_mont(BIGNUM *r, const BIGNUM *a, const BIGNUM *p,
                                   const BIGNUM *m, BN_CTX *ctx,
                                   BN_MONT_CTX *m_ctx);

OPENSSL_EXPORT int BN_mod_exp_mont_consttime(BIGNUM *rr, const BIGNUM *a,
                                             const BIGNUM *p, const BIGNUM *m,
                                             BN_CTX *ctx, BN_MONT_CTX *in_mont);

OPENSSL_EXPORT int BN_mod_exp_mont_word(BIGNUM *r, BN_ULONG a, const BIGNUM *p,
                                        const BIGNUM *m, BN_CTX *ctx,
                                        BN_MONT_CTX *m_ctx);
OPENSSL_EXPORT int BN_mod_exp2_mont(BIGNUM *r, const BIGNUM *a1,
                                    const BIGNUM *p1, const BIGNUM *a2,
                                    const BIGNUM *p2, const BIGNUM *m,
                                    BN_CTX *ctx, BN_MONT_CTX *m_ctx);


/* Private functions */

struct bignum_st {
  BN_ULONG *d; /* Pointer to an array of 'BN_BITS2' bit chunks in little-endian
                  order. */
  int top;   /* Index of last used element in |d|, plus one. */
  int dmax;  /* Size of |d|, in words. */
  int neg;   /* one if the number is negative */
  int flags; /* bitmask of BN_FLG_* values */
};

struct bn_mont_ctx_st {
  BIGNUM RR; /* used to convert to montgomery form */
  BIGNUM N;  /* The modulus */
  BIGNUM Ni; /* R*(1/R mod N) - N*Ni = 1
              * (Ni is only stored for bignum algorithm) */
  BN_ULONG n0[2]; /* least significant word(s) of Ni;
                     (type changed with 0.9.9, was "BN_ULONG n0;" before) */
  int flags;
  int ri;    /* number of bits in R */
};

OPENSSL_EXPORT unsigned BN_num_bits_word(BN_ULONG l);

#define BN_FLG_MALLOCED 0x01
#define BN_FLG_STATIC_DATA 0x02
/* avoid leaking exponent information through timing, BN_mod_exp_mont() will
 * call BN_mod_exp_mont_consttime, BN_div() will call BN_div_no_branch,
 * BN_mod_inverse() will call BN_mod_inverse_no_branch. */
#define BN_FLG_CONSTTIME 0x04


/* Android compatibility section.
 *
 * These functions are declared, temporarily, for Android because
 * wpa_supplicant will take a little time to sync with upstream. Outside of
 * Android they'll have no definition. */

OPENSSL_EXPORT BIGNUM *get_rfc3526_prime_1536(BIGNUM *bn);


#if defined(__cplusplus)
}  /* extern C */
#endif

#define BN_F_BN_CTX_get 100
#define BN_F_BN_CTX_new 101
#define BN_F_BN_CTX_start 102
#define BN_F_BN_bn2dec 103
#define BN_F_BN_bn2hex 104
#define BN_F_BN_div 105
#define BN_F_BN_div_recp 106
#define BN_F_BN_exp 107
#define BN_F_BN_generate_dsa_nonce 108
#define BN_F_BN_generate_prime_ex 109
#define BN_F_BN_mod_exp2_mont 110
#define BN_F_BN_mod_exp_mont 111
#define BN_F_BN_mod_exp_mont_consttime 112
#define BN_F_BN_mod_exp_mont_word 113
#define BN_F_BN_mod_inverse 114
#define BN_F_BN_mod_inverse_no_branch 115
#define BN_F_BN_mod_lshift_quick 116
#define BN_F_BN_mod_sqrt 117
#define BN_F_BN_new 118
#define BN_F_BN_rand 119
#define BN_F_BN_rand_range 120
#define BN_F_BN_sqrt 121
#define BN_F_BN_usub 122
#define BN_F_bn_wexpand 123
#define BN_F_mod_exp_recp 124
#define BN_F_BN_lshift 125
#define BN_F_BN_rshift 126
#define BN_R_ARG2_LT_ARG3 100
#define BN_R_BAD_RECIPROCAL 101
#define BN_R_BIGNUM_TOO_LONG 102
#define BN_R_BITS_TOO_SMALL 103
#define BN_R_CALLED_WITH_EVEN_MODULUS 104
#define BN_R_DIV_BY_ZERO 105
#define BN_R_EXPAND_ON_STATIC_BIGNUM_DATA 106
#define BN_R_INPUT_NOT_REDUCED 107
#define BN_R_INVALID_RANGE 108
#define BN_R_NEGATIVE_NUMBER 109
#define BN_R_NOT_A_SQUARE 110
#define BN_R_NOT_INITIALIZED 111
#define BN_R_NO_INVERSE 112
#define BN_R_PRIVATE_KEY_TOO_LARGE 113
#define BN_R_P_IS_NOT_PRIME 114
#define BN_R_TOO_MANY_ITERATIONS 115
#define BN_R_TOO_MANY_TEMPORARY_VARIABLES 116

#endif  /* OPENSSL_HEADER_BN_H */
