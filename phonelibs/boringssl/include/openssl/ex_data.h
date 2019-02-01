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
 * [including the GNU Public Licence.]
 */
/* ====================================================================
 * Copyright (c) 1998-2001 The OpenSSL Project.  All rights reserved.
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
 * Hudson (tjh@cryptsoft.com). */

#ifndef OPENSSL_HEADER_EX_DATA_H
#define OPENSSL_HEADER_EX_DATA_H

#include <openssl/base.h>

#include <openssl/stack.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* ex_data is a mechanism for associating arbitrary extra data with objects.
 * For each type of object that supports ex_data, different users can be
 * assigned indexes in which to store their data. Each index has callback
 * functions that are called when a new object of that type is created, freed
 * and duplicated. */


typedef struct crypto_ex_data_st CRYPTO_EX_DATA;


/* Type-specific functions.
 *
 * Each type that supports ex_data provides three functions: */

#if 0 /* Sample */

/* |TYPE_get_ex_new_index| allocates a new index for |TYPE|. See the
 * descriptions of the callback typedefs for details of when they are
 * called. Any of the callback arguments may be NULL. The |argl| and |argp|
 * arguments are opaque values that are passed to the callbacks. It returns the
 * new index or a negative number on error.
 *
 * TODO(fork): this should follow the standard calling convention. */
OPENSSL_EXPORT int TYPE_get_ex_new_index(long argl, void *argp,
                                         CRYPTO_EX_new *new_func,
                                         CRYPTO_EX_dup *dup_func,
                                         CRYPTO_EX_free *free_func);

/* |TYPE_set_ex_data| sets an extra data pointer on |t|. The |index| argument
 * should have been returned from a previous call to |TYPE_get_ex_new_index|. */
OPENSSL_EXPORT int TYPE_set_ex_data(TYPE *t, int index, void *arg);

/* |TYPE_get_ex_data| returns an extra data pointer for |t|, or NULL if no such
 * pointer exists. The |index| argument should have been returned from a
 * previous call to |TYPE_get_ex_new_index|. */
OPENSSL_EXPORT void *TYPE_get_ex_data(const TYPE *t, int index);

#endif /* Sample */


/* Callback types. */

/* CRYPTO_EX_new is the type of a callback function that is called whenever a
 * new object of a given class is created. For example, if this callback has
 * been passed to |SSL_get_ex_new_index| then it'll be called each time an SSL*
 * is created.
 *
 * The callback is passed the new object (i.e. the SSL*) in |parent|. The
 * arguments |argl| and |argp| contain opaque values that were given to
 * |CRYPTO_get_ex_new_index|. The callback should return one on success, but
 * the value is ignored.
 *
 * TODO(fork): the |ptr| argument is always NULL, no? */
typedef int CRYPTO_EX_new(void *parent, void *ptr, CRYPTO_EX_DATA *ad,
                          int index, long argl, void *argp);

/* CRYPTO_EX_free is a callback function that is called when an object of the
 * class is being destroyed. See |CRYPTO_EX_new| for a discussion of the
 * arguments.
 *
 * If |CRYPTO_get_ex_new_index| was called after the creation of objects of the
 * class that this applies to then, when those those objects are destroyed,
 * this callback will be called with a NULL value for |ptr|. */
typedef void CRYPTO_EX_free(void *parent, void *ptr, CRYPTO_EX_DATA *ad,
                            int index, long argl, void *argp);

/* CRYPTO_EX_dup is a callback function that is called when an object of the
 * class is being copied and thus the ex_data linked to it also needs to be
 * copied. On entry, |*from_d| points to the data for this index from the
 * original object. When the callback returns, |*from_d| will be set as the
 * data for this index in |to|.
 *
 * If |CRYPTO_get_ex_new_index| was called after the creation of objects of the
 * class that this applies to then, when those those objects are copies, this
 * callback will be called with a NULL value for |*from_d|. */
typedef int CRYPTO_EX_dup(CRYPTO_EX_DATA *to, const CRYPTO_EX_DATA *from,
                          void **from_d, int index, long argl, void *argp);


/* Deprecated functions. */

/* CRYPTO_cleanup_all_ex_data does nothing. */
OPENSSL_EXPORT void CRYPTO_cleanup_all_ex_data(void);

struct crypto_ex_data_st {
  STACK_OF(void) *sk;
};


#if defined(__cplusplus)
}  /* extern C */
#endif

#endif  /* OPENSSL_HEADER_EX_DATA_H */
