/* Copyright (c) 2014, Google Inc.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#ifndef OPENSSL_HEADER_CRYPTO_H
#define OPENSSL_HEADER_CRYPTO_H

#include <openssl/base.h>

/* Upstream OpenSSL defines |OPENSSL_malloc|, etc., in crypto.h rather than
 * mem.h. */
#include <openssl/mem.h>


#if defined(__cplusplus)
extern "C" {
#endif


/* crypto.h contains functions for initializing the crypto library. */


/* CRYPTO_library_init initializes the crypto library. It must be called if the
 * library is built with BORINGSSL_NO_STATIC_INITIALIZER. Otherwise, it does
 * nothing and a static initializer is used instead. */
OPENSSL_EXPORT void CRYPTO_library_init(void);


/* Deprecated functions. */

#define OPENSSL_VERSION_TEXT "BoringSSL"

#define SSLEAY_VERSION 0

/* SSLeay_version is a compatibility function that returns the string
 * "BoringSSL". */
OPENSSL_EXPORT const char *SSLeay_version(int unused);

/* SSLeay is a compatibility function that returns OPENSSL_VERSION_NUMBER from
 * base.h. */
OPENSSL_EXPORT unsigned long SSLeay(void);


#if defined(__cplusplus)
}  /* extern C */
#endif

#define CRYPTO_F_CRYPTO_get_ex_new_index 100
#define CRYPTO_F_CRYPTO_set_ex_data 101
#define CRYPTO_F_get_class 102
#define CRYPTO_F_get_func_pointers 103

#endif  /* OPENSSL_HEADER_CRYPTO_H */
