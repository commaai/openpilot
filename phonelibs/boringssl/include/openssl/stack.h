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

#ifndef OPENSSL_HEADER_STACK_H
#define OPENSSL_HEADER_STACK_H

#include <openssl/base.h>

#include <openssl/type_check.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* A stack, in OpenSSL, is an array of pointers. They are the most commonly
 * used collection object.
 *
 * This file defines macros for type safe use of the stack functions. A stack
 * of a specific type of object has type |STACK_OF(type)|. This can be defined
 * (once) with |DEFINE_STACK_OF(type)| and declared where needed with
 * |DECLARE_STACK_OF(type)|. For example:
 *
 *   struct foo {
 *     int bar;
 *   };
 *
 *   DEFINE_STACK_OF(struct foo);
 *
 * Although note that the stack will contain /pointers/ to |foo|.
 *
 * A macro will be defined for each of the sk_* functions below. For
 * STACK_OF(foo), the macros would be sk_foo_new, sk_foo_pop etc. */


/* stack_cmp_func is a comparison function that returns a value < 0, 0 or > 0
 * if |*a| is less than, equal to or greater than |*b|, respectively.  Note the
 * extra indirection - the function is given a pointer to a pointer to the
 * element. This differs from the usual qsort/bsearch comparison function. */
typedef int (*stack_cmp_func)(const void **a, const void **b);

/* stack_st contains an array of pointers. It is not designed to be used
 * directly, rather the wrapper macros should be used. */
typedef struct stack_st {
  /* num contains the number of valid pointers in |data|. */
  size_t num;
  void **data;
  /* sorted is non-zero if the values pointed to by |data| are in ascending
   * order, based on |comp|. */
  size_t sorted;
  /* num_alloc contains the number of pointers allocated in the buffer pointed
   * to by |data|, which may be larger than |num|. */
  size_t num_alloc;
  /* comp is an optional comparison function. */
  stack_cmp_func comp;
} _STACK;


#define STACK_OF(type) struct stack_st_##type

#define DEFINE_STACK_OF(type) \
STACK_OF(type) {\
  _STACK stack; \
}

#define DECLARE_STACK_OF(type) STACK_OF(type);

/* The make_macros.sh script in this directory parses the following lines and
 * generates the stack_macros.h file that contains macros for the following
 * types of stacks:
 *
 * STACK_OF:ACCESS_DESCRIPTION
 * STACK_OF:ASN1_ADB_TABLE
 * STACK_OF:ASN1_GENERALSTRING
 * STACK_OF:ASN1_INTEGER
 * STACK_OF:ASN1_OBJECT
 * STACK_OF:ASN1_STRING_TABLE
 * STACK_OF:ASN1_TYPE
 * STACK_OF:ASN1_VALUE
 * STACK_OF:BIO
 * STACK_OF:BY_DIR_ENTRY
 * STACK_OF:BY_DIR_HASH
 * STACK_OF:CONF_VALUE
 * STACK_OF:CRYPTO_EX_DATA_FUNCS
 * STACK_OF:DIST_POINT
 * STACK_OF:GENERAL_NAME
 * STACK_OF:GENERAL_NAMES
 * STACK_OF:GENERAL_SUBTREE
 * STACK_OF:MIME_HEADER
 * STACK_OF:PKCS7_SIGNER_INFO
 * STACK_OF:PKCS7_RECIP_INFO
 * STACK_OF:POLICYINFO
 * STACK_OF:POLICYQUALINFO
 * STACK_OF:POLICY_MAPPING
 * STACK_OF:SSL_COMP
 * STACK_OF:STACK_OF_X509_NAME_ENTRY
 * STACK_OF:SXNETID
 * STACK_OF:X509
 * STACK_OF:X509V3_EXT_METHOD
 * STACK_OF:X509_ALGOR
 * STACK_OF:X509_ATTRIBUTE
 * STACK_OF:X509_CRL
 * STACK_OF:X509_EXTENSION
 * STACK_OF:X509_INFO
 * STACK_OF:X509_LOOKUP
 * STACK_OF:X509_NAME
 * STACK_OF:X509_NAME_ENTRY
 * STACK_OF:X509_OBJECT
 * STACK_OF:X509_POLICY_DATA
 * STACK_OF:X509_POLICY_NODE
 * STACK_OF:X509_PURPOSE
 * STACK_OF:X509_REVOKED
 * STACK_OF:X509_TRUST
 * STACK_OF:X509_VERIFY_PARAM
 * STACK_OF:void
 *
 * Some stacks contain only const structures, so the stack should return const
 * pointers to retain type-checking.
 *
 * CONST_STACK_OF:SRTP_PROTECTION_PROFILE
 * CONST_STACK_OF:SSL_CIPHER */


/* Some stacks are special because, although we would like STACK_OF(char *),
 * that would actually be a stack of pointers to char*, but we just want to
 * point to the string directly. In this case we call them "special" and use
 * |DEFINE_SPECIAL_STACK_OF(type)| */
#define DEFINE_SPECIAL_STACK_OF(type, inner)             \
  STACK_OF(type) { _STACK special_stack; };              \
  OPENSSL_COMPILE_ASSERT(sizeof(type) == sizeof(void *), \
                         special_stack_of_non_pointer_##type);

typedef char *OPENSSL_STRING;

DEFINE_SPECIAL_STACK_OF(OPENSSL_STRING, char)
DEFINE_SPECIAL_STACK_OF(OPENSSL_BLOCK, uint8_t)

/* The make_macros.sh script in this directory parses the following lines and
 * generates the stack_macros.h file that contains macros for the following
 * types of stacks:
 *
 * SPECIAL_STACK_OF:OPENSSL_STRING
 * SPECIAL_STACK_OF:OPENSSL_BLOCK */

#define IN_STACK_H
#include <openssl/stack_macros.h>
#undef IN_STACK_H


/* These are the raw stack functions, you shouldn't be using them. Rather you
 * should be using the type stack macros implemented above. */

/* sk_new creates a new, empty stack with the given comparison function, which
 * may be zero. It returns the new stack or NULL on allocation failure. */
OPENSSL_EXPORT _STACK *sk_new(stack_cmp_func comp);

/* sk_new_null creates a new, empty stack. It returns the new stack or NULL on
 * allocation failure. */
OPENSSL_EXPORT _STACK *sk_new_null(void);

/* sk_num returns the number of elements in |s|. */
OPENSSL_EXPORT size_t sk_num(const _STACK *sk);

/* sk_zero resets |sk| to the empty state but does nothing to free the
 * individual elements themselves. */
OPENSSL_EXPORT void sk_zero(_STACK *sk);

/* sk_value returns the |i|th pointer in |sk|, or NULL if |i| is out of
 * range. */
OPENSSL_EXPORT void *sk_value(const _STACK *sk, size_t i);

/* sk_set sets the |i|th pointer in |sk| to |p| and returns |p|. If |i| is out
 * of range, it returns NULL. */
OPENSSL_EXPORT void *sk_set(_STACK *sk, size_t i, void *p);

/* sk_free frees the given stack and array of pointers, but does nothing to
 * free the individual elements. Also see |sk_pop_free|. */
OPENSSL_EXPORT void sk_free(_STACK *sk);

/* sk_pop_free calls |free_func| on each element in the stack and then frees
 * the stack itself. */
OPENSSL_EXPORT void sk_pop_free(_STACK *sk, void (*free_func)(void *));

/* sk_insert inserts |p| into the stack at index |where|, moving existing
 * elements if needed. It returns the length of the new stack, or zero on
 * error. */
OPENSSL_EXPORT size_t sk_insert(_STACK *sk, void *p, size_t where);

/* sk_delete removes the pointer at index |where|, moving other elements down
 * if needed. It returns the removed pointer, or NULL if |where| is out of
 * range. */
OPENSSL_EXPORT void *sk_delete(_STACK *sk, size_t where);

/* sk_delete_ptr removes, at most, one instance of |p| from the stack based on
 * pointer equality. If an instance of |p| is found then |p| is returned,
 * otherwise it returns NULL. */
OPENSSL_EXPORT void *sk_delete_ptr(_STACK *sk, void *p);

/* sk_find returns the first value in the stack equal to |p|. If a comparison
 * function has been set on the stack, then equality is defined by it and the
 * stack will be sorted if need be so that a binary search can be used.
 * Otherwise pointer equality is used. If a matching element is found, its
 * index is written to |*out_index| (if |out_index| is not NULL) and one is
 * returned. Otherwise zero is returned. */
OPENSSL_EXPORT int sk_find(_STACK *sk, size_t *out_index, void *p);

/* sk_shift removes and returns the first element in the stack, or returns NULL
 * if the stack is empty. */
OPENSSL_EXPORT void *sk_shift(_STACK *sk);

/* sk_push appends |p| to the stack and returns the length of the new stack, or
 * 0 on allocation failure. */
OPENSSL_EXPORT size_t sk_push(_STACK *sk, void *p);

/* sk_pop returns and removes the last element on the stack, or NULL if the
 * stack is empty. */
OPENSSL_EXPORT void *sk_pop(_STACK *sk);

/* sk_dup performs a shallow copy of a stack and returns the new stack, or NULL
 * on error. */
OPENSSL_EXPORT _STACK *sk_dup(const _STACK *sk);

/* sk_sort sorts the elements of |sk| into ascending order based on the
 * comparison function. The stack maintains a |sorted| flag and sorting an
 * already sorted stack is a no-op. */
OPENSSL_EXPORT void sk_sort(_STACK *sk);

/* sk_is_sorted returns one if |sk| is known to be sorted and zero
 * otherwise. */
OPENSSL_EXPORT int sk_is_sorted(const _STACK *sk);

/* sk_set_cmp_func sets the comparison function to be used by |sk| and returns
 * the previous one. */
OPENSSL_EXPORT stack_cmp_func sk_set_cmp_func(_STACK *sk, stack_cmp_func comp);

/* sk_deep_copy performs a copy of |sk| and of each of the non-NULL elements in
 * |sk| by using |copy_func|. If an error occurs, |free_func| is used to free
 * any copies already made and NULL is returned. */
OPENSSL_EXPORT _STACK *sk_deep_copy(const _STACK *sk,
                                    void *(*copy_func)(void *),
                                    void (*free_func)(void *));


#if defined(__cplusplus)
}  /* extern C */
#endif

#endif  /* OPENSSL_HEADER_STACK_H */
