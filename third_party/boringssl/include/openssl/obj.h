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

#ifndef OPENSSL_HEADER_OBJECTS_H
#define OPENSSL_HEADER_OBJECTS_H

#include <openssl/base.h>

#include <openssl/bytestring.h>
#include <openssl/obj_mac.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* The objects library deals with the registration and indexing of ASN.1 object
 * identifiers. These values are often written as a dotted sequence of numbers,
 * e.g. 1.2.840.113549.1.9.16.3.9.
 *
 * Internally, OpenSSL likes to deal with these values by numbering them with
 * numbers called "nids". OpenSSL has a large, built-in database of common
 * object identifiers and also has both short and long names for them.
 *
 * This library provides functions for translating between object identifiers,
 * nids, short names and long names.
 *
 * The nid values should not be used outside of a single process: they are not
 * stable identifiers. */


/* Basic operations. */

/* OBJ_dup returns a duplicate copy of |obj| or NULL on allocation failure. */
OPENSSL_EXPORT ASN1_OBJECT *OBJ_dup(const ASN1_OBJECT *obj);

/* OBJ_cmp returns a value less than, equal to or greater than zero if |a| is
 * less than, equal to or greater than |b|, respectively. */
OPENSSL_EXPORT int OBJ_cmp(const ASN1_OBJECT *a, const ASN1_OBJECT *b);


/* Looking up nids. */

/* OBJ_obj2nid returns the nid corresponding to |obj|, or |NID_undef| if no
 * such object is known. */
OPENSSL_EXPORT int OBJ_obj2nid(const ASN1_OBJECT *obj);

/* OBJ_cbs2nid returns the nid corresponding to the DER data in |cbs|, or
 * |NID_undef| if no such object is known. */
OPENSSL_EXPORT int OBJ_cbs2nid(const CBS *cbs);

/* OBJ_sn2nid returns the nid corresponding to |short_name|, or |NID_undef| if
 * no such short name is known. */
OPENSSL_EXPORT int OBJ_sn2nid(const char *short_name);

/* OBJ_ln2nid returns the nid corresponding to |long_name|, or |NID_undef| if
 * no such long name is known. */
OPENSSL_EXPORT int OBJ_ln2nid(const char *long_name);

/* OBJ_txt2nid returns the nid corresponding to |s|, which may be a short name,
 * long name, or an ASCII string containing a dotted sequence of numbers. It
 * returns the nid or NID_undef if unknown. */
OPENSSL_EXPORT int OBJ_txt2nid(const char *s);


/* Getting information about nids. */

/* OBJ_nid2obj returns the ASN1_OBJECT corresponding to |nid|, or NULL if |nid|
 * is unknown. */
OPENSSL_EXPORT const ASN1_OBJECT *OBJ_nid2obj(int nid);

/* OBJ_nid2sn returns the short name for |nid|, or NULL if |nid| is unknown. */
OPENSSL_EXPORT const char *OBJ_nid2sn(int nid);

/* OBJ_nid2sn returns the long name for |nid|, or NULL if |nid| is unknown. */
OPENSSL_EXPORT const char *OBJ_nid2ln(int nid);

/* OBJ_nid2cbs writes |nid| as an ASN.1 OBJECT IDENTIFIER to |out|. It returns
 * one on success or zero otherwise. */
OPENSSL_EXPORT int OBJ_nid2cbb(CBB *out, int nid);


/* Dealing with textual representations of object identifiers. */

/* OBJ_txt2obj returns an ASN1_OBJECT for the textual respresentation in |s|.
 * If |dont_search_names| is zero, then |s| will be matched against the long
 * and short names of a known objects to find a match. Otherwise |s| must
 * contain an ASCII string with a dotted sequence of numbers. The resulting
 * object need not be previously known. It returns a freshly allocated
 * |ASN1_OBJECT| or NULL on error. */
OPENSSL_EXPORT ASN1_OBJECT *OBJ_txt2obj(const char *s, int dont_search_names);

/* OBJ_obj2txt converts |obj| to a textual representation. If
 * |dont_return_name| is zero then |obj| will be matched against known objects
 * and the long (preferably) or short name will be used if found. Otherwise
 * |obj| will be converted into a dotted sequence of integers. If |out| is not
 * NULL, then at most |out_len| bytes of the textual form will be written
 * there. If |out_len| is at least one, then string written to |out| will
 * always be NUL terminated. It returns the number of characters that could
 * have been written, not including the final NUL, or -1 on error. */
OPENSSL_EXPORT int OBJ_obj2txt(char *out, int out_len, const ASN1_OBJECT *obj,
                               int dont_return_name);


/* Adding objects at runtime. */

/* OBJ_create adds a known object and returns the nid of the new object, or
 * NID_undef on error. */
OPENSSL_EXPORT int OBJ_create(const char *oid, const char *short_name,
                              const char *long_name);


/* Handling signature algorithm identifiers.
 *
 * Some NIDs (e.g. sha256WithRSAEncryption) specify both a digest algorithm and
 * a public key algorithm. The following functions map between pairs of digest
 * and public-key algorithms and the NIDs that specify their combination.
 *
 * Sometimes the combination NID leaves the digest unspecified (e.g.
 * rsassaPss). In these cases, the digest NID is |NID_undef|. */

/* OBJ_find_sigid_algs finds the digest and public-key NIDs that correspond to
 * the signing algorithm |sign_nid|. If successful, it sets |*out_digest_nid|
 * and |*out_pkey_nid| and returns one. Otherwise it returns zero. Any of
 * |out_digest_nid| or |out_pkey_nid| can be NULL if the caller doesn't need
 * that output value. */
OPENSSL_EXPORT int OBJ_find_sigid_algs(int sign_nid, int *out_digest_nid,
                                       int *out_pkey_nid);

/* OBJ_find_sigid_by_algs finds the signature NID that corresponds to the
 * combination of |digest_nid| and |pkey_nid|. If success, it sets
 * |*out_sign_nid| and returns one. Otherwise it returns zero. The
 * |out_sign_nid| argument can be NULL if the caller only wishes to learn
 * whether the combination is valid. */
OPENSSL_EXPORT int OBJ_find_sigid_by_algs(int *out_sign_nid, int digest_nid,
                                          int pkey_nid);


#if defined(__cplusplus)
}  /* extern C */
#endif

#define OBJ_F_OBJ_create 100
#define OBJ_F_OBJ_dup 101
#define OBJ_F_OBJ_nid2obj 102
#define OBJ_F_OBJ_txt2obj 103
#define OBJ_R_UNKNOWN_NID 100

#endif  /* OPENSSL_HEADER_OBJECTS_H */
