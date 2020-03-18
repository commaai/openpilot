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

#ifndef OPENSSL_HEADER_RAND_H
#define OPENSSL_HEADER_RAND_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* Random number generation. */


/* RAND_bytes writes |len| bytes of random data to |buf| and returns one. */
OPENSSL_EXPORT int RAND_bytes(uint8_t *buf, size_t len);

/* RAND_cleanup frees any resources used by the RNG. This is not safe if other
 * threads might still be calling |RAND_bytes|. */
OPENSSL_EXPORT void RAND_cleanup(void);


/* Deprecated functions */

/* RAND_pseudo_bytes is a wrapper around |RAND_bytes|. */
OPENSSL_EXPORT int RAND_pseudo_bytes(uint8_t *buf, size_t len);

/* RAND_seed does nothing. */
OPENSSL_EXPORT void RAND_seed(const void *buf, int num);

/* RAND_load_file returns a nonnegative number. */
OPENSSL_EXPORT int RAND_load_file(const char *path, long num);

/* RAND_add does nothing. */
OPENSSL_EXPORT void RAND_add(const void *buf, int num, double entropy);

/* RAND_poll returns one. */
OPENSSL_EXPORT int RAND_poll(void);

/* RAND_status returns one. */
OPENSSL_EXPORT int RAND_status(void);


#if defined(__cplusplus)
}  /* extern C */
#endif

#endif  /* OPENSSL_HEADER_RAND_H */
