/* crypto/x509/x509.h */
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
 * Copyright 2002 Sun Microsystems, Inc. ALL RIGHTS RESERVED.
 * ECDH support in OpenSSL originally developed by 
 * SUN MICROSYSTEMS, INC., and contributed to the OpenSSL project.
 */

#ifndef HEADER_X509_H
#define HEADER_X509_H

#include <openssl/base.h>

#include <time.h>

#include <openssl/asn1.h>
#include <openssl/bio.h>
#include <openssl/cipher.h>
#include <openssl/dh.h>
#include <openssl/dsa.h>
#include <openssl/ecdh.h>
#include <openssl/ecdsa.h>
#include <openssl/ec.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>
#include <openssl/sha.h>
#include <openssl/stack.h>
#include <openssl/thread.h>

#ifdef  __cplusplus
extern "C" {
#endif


#define X509_FILETYPE_PEM	1
#define X509_FILETYPE_ASN1	2
#define X509_FILETYPE_DEFAULT	3

#define X509v3_KU_DIGITAL_SIGNATURE	0x0080
#define X509v3_KU_NON_REPUDIATION	0x0040
#define X509v3_KU_KEY_ENCIPHERMENT	0x0020
#define X509v3_KU_DATA_ENCIPHERMENT	0x0010
#define X509v3_KU_KEY_AGREEMENT		0x0008
#define X509v3_KU_KEY_CERT_SIGN		0x0004
#define X509v3_KU_CRL_SIGN		0x0002
#define X509v3_KU_ENCIPHER_ONLY		0x0001
#define X509v3_KU_DECIPHER_ONLY		0x8000
#define X509v3_KU_UNDEF			0xffff

typedef struct X509_objects_st
	{
	int nid;
	int (*a2i)(void);
	int (*i2a)(void);
	} X509_OBJECTS;

DECLARE_ASN1_SET_OF(X509_ALGOR)

typedef STACK_OF(X509_ALGOR) X509_ALGORS;

typedef struct X509_val_st
	{
	ASN1_TIME *notBefore;
	ASN1_TIME *notAfter;
	} X509_VAL;

struct X509_pubkey_st
	{
	X509_ALGOR *algor;
	ASN1_BIT_STRING *public_key;
	EVP_PKEY *pkey;
	};

typedef struct X509_sig_st
	{
	X509_ALGOR *algor;
	ASN1_OCTET_STRING *digest;
	} X509_SIG;

typedef struct X509_name_entry_st
	{
	ASN1_OBJECT *object;
	ASN1_STRING *value;
	int set;
	int size; 	/* temp variable */
	} X509_NAME_ENTRY;

DECLARE_STACK_OF(X509_NAME_ENTRY)
DECLARE_ASN1_SET_OF(X509_NAME_ENTRY)

/* we always keep X509_NAMEs in 2 forms. */
typedef struct X509_name_st
	{
	STACK_OF(X509_NAME_ENTRY) *entries;
	int modified;	/* true if 'bytes' needs to be built */
#ifndef OPENSSL_NO_BUFFER
	BUF_MEM *bytes;
#else
	char *bytes;
#endif
/*	unsigned long hash; Keep the hash around for lookups */
	unsigned char *canon_enc;
	int canon_enclen;
	} X509_NAME;

DECLARE_STACK_OF(X509_NAME)

#define X509_EX_V_NETSCAPE_HACK		0x8000
#define X509_EX_V_INIT			0x0001
typedef struct X509_extension_st
	{
	ASN1_OBJECT *object;
	ASN1_BOOLEAN critical;
	ASN1_OCTET_STRING *value;
	} X509_EXTENSION;

typedef STACK_OF(X509_EXTENSION) X509_EXTENSIONS;

DECLARE_STACK_OF(X509_EXTENSION)
DECLARE_ASN1_SET_OF(X509_EXTENSION)

/* a sequence of these are used */
typedef struct x509_attributes_st
	{
	ASN1_OBJECT *object;
	int single; /* 0 for a set, 1 for a single item (which is wrong) */
	union	{
		char		*ptr;
/* 0 */		STACK_OF(ASN1_TYPE) *set;
/* 1 */		ASN1_TYPE	*single;
		} value;
	} X509_ATTRIBUTE;

DECLARE_STACK_OF(X509_ATTRIBUTE)
DECLARE_ASN1_SET_OF(X509_ATTRIBUTE)


typedef struct X509_req_info_st
	{
	ASN1_ENCODING enc;
	ASN1_INTEGER *version;
	X509_NAME *subject;
	X509_PUBKEY *pubkey;
	/*  d=2 hl=2 l=  0 cons: cont: 00 */
	STACK_OF(X509_ATTRIBUTE) *attributes; /* [ 0 ] */
	} X509_REQ_INFO;

typedef struct X509_req_st
	{
	X509_REQ_INFO *req_info;
	X509_ALGOR *sig_alg;
	ASN1_BIT_STRING *signature;
	CRYPTO_refcount_t references;
	} X509_REQ;

typedef struct x509_cinf_st
	{
	ASN1_INTEGER *version;		/* [ 0 ] default of v1 */
	ASN1_INTEGER *serialNumber;
	X509_ALGOR *signature;
	X509_NAME *issuer;
	X509_VAL *validity;
	X509_NAME *subject;
	X509_PUBKEY *key;
	ASN1_BIT_STRING *issuerUID;		/* [ 1 ] optional in v2 */
	ASN1_BIT_STRING *subjectUID;		/* [ 2 ] optional in v2 */
	STACK_OF(X509_EXTENSION) *extensions;	/* [ 3 ] optional in v3 */
	ASN1_ENCODING enc;
	} X509_CINF;

/* This stuff is certificate "auxiliary info"
 * it contains details which are useful in certificate
 * stores and databases. When used this is tagged onto
 * the end of the certificate itself
 */

typedef struct x509_cert_aux_st
	{
	STACK_OF(ASN1_OBJECT) *trust;		/* trusted uses */
	STACK_OF(ASN1_OBJECT) *reject;		/* rejected uses */
	ASN1_UTF8STRING *alias;			/* "friendly name" */
	ASN1_OCTET_STRING *keyid;		/* key id of private key */
	STACK_OF(X509_ALGOR) *other;		/* other unspecified info */
	} X509_CERT_AUX;

struct x509_st
	{
	X509_CINF *cert_info;
	X509_ALGOR *sig_alg;
	ASN1_BIT_STRING *signature;
	int valid;
	CRYPTO_refcount_t references;
	char *name;
	CRYPTO_EX_DATA ex_data;
	/* These contain copies of various extension values */
	long ex_pathlen;
	long ex_pcpathlen;
	unsigned long ex_flags;
	unsigned long ex_kusage;
	unsigned long ex_xkusage;
	unsigned long ex_nscert;
	ASN1_OCTET_STRING *skid;
	AUTHORITY_KEYID *akid;
	X509_POLICY_CACHE *policy_cache;
	STACK_OF(DIST_POINT) *crldp;
	STACK_OF(GENERAL_NAME) *altname;
	NAME_CONSTRAINTS *nc;
	unsigned char sha1_hash[SHA_DIGEST_LENGTH];
	X509_CERT_AUX *aux;
	} /* X509 */;

DECLARE_STACK_OF(X509)
DECLARE_ASN1_SET_OF(X509)

/* This is used for a table of trust checking functions */

typedef struct x509_trust_st {
	int trust;
	int flags;
	int (*check_trust)(struct x509_trust_st *, X509 *, int);
	char *name;
	int arg1;
	void *arg2;
} X509_TRUST;

DECLARE_STACK_OF(X509_TRUST)

typedef struct x509_cert_pair_st {
	X509 *forward;
	X509 *reverse;
} X509_CERT_PAIR;

/* standard trust ids */

#define X509_TRUST_DEFAULT	-1	/* Only valid in purpose settings */

#define X509_TRUST_COMPAT	1
#define X509_TRUST_SSL_CLIENT	2
#define X509_TRUST_SSL_SERVER	3
#define X509_TRUST_EMAIL	4
#define X509_TRUST_OBJECT_SIGN	5
#define X509_TRUST_OCSP_SIGN	6
#define X509_TRUST_OCSP_REQUEST	7
#define X509_TRUST_TSA		8

/* Keep these up to date! */
#define X509_TRUST_MIN		1
#define X509_TRUST_MAX		8


/* trust_flags values */
#define	X509_TRUST_DYNAMIC 	1
#define	X509_TRUST_DYNAMIC_NAME	2

/* check_trust return codes */

#define X509_TRUST_TRUSTED	1
#define X509_TRUST_REJECTED	2
#define X509_TRUST_UNTRUSTED	3

/* Flags for X509_print_ex() */

#define	X509_FLAG_COMPAT		0
#define	X509_FLAG_NO_HEADER		1L
#define	X509_FLAG_NO_VERSION		(1L << 1)
#define	X509_FLAG_NO_SERIAL		(1L << 2)
#define	X509_FLAG_NO_SIGNAME		(1L << 3)
#define	X509_FLAG_NO_ISSUER		(1L << 4)
#define	X509_FLAG_NO_VALIDITY		(1L << 5)
#define	X509_FLAG_NO_SUBJECT		(1L << 6)
#define	X509_FLAG_NO_PUBKEY		(1L << 7)
#define	X509_FLAG_NO_EXTENSIONS		(1L << 8)
#define	X509_FLAG_NO_SIGDUMP		(1L << 9)
#define	X509_FLAG_NO_AUX		(1L << 10)
#define	X509_FLAG_NO_ATTRIBUTES		(1L << 11)
#define	X509_FLAG_NO_IDS		(1L << 12)

/* Flags specific to X509_NAME_print_ex() */	

/* The field separator information */

#define XN_FLAG_SEP_MASK	(0xf << 16)

#define XN_FLAG_COMPAT		0		/* Traditional SSLeay: use old X509_NAME_print */
#define XN_FLAG_SEP_COMMA_PLUS	(1 << 16)	/* RFC2253 ,+ */
#define XN_FLAG_SEP_CPLUS_SPC	(2 << 16)	/* ,+ spaced: more readable */
#define XN_FLAG_SEP_SPLUS_SPC	(3 << 16)	/* ;+ spaced */
#define XN_FLAG_SEP_MULTILINE	(4 << 16)	/* One line per field */

#define XN_FLAG_DN_REV		(1 << 20)	/* Reverse DN order */

/* How the field name is shown */

#define XN_FLAG_FN_MASK		(0x3 << 21)

#define XN_FLAG_FN_SN		0		/* Object short name */
#define XN_FLAG_FN_LN		(1 << 21)	/* Object long name */
#define XN_FLAG_FN_OID		(2 << 21)	/* Always use OIDs */
#define XN_FLAG_FN_NONE		(3 << 21)	/* No field names */

#define XN_FLAG_SPC_EQ		(1 << 23)	/* Put spaces round '=' */

/* This determines if we dump fields we don't recognise:
 * RFC2253 requires this.
 */

#define XN_FLAG_DUMP_UNKNOWN_FIELDS (1 << 24)

#define XN_FLAG_FN_ALIGN	(1 << 25)	/* Align field names to 20 characters */

/* Complete set of RFC2253 flags */

#define XN_FLAG_RFC2253 (ASN1_STRFLGS_RFC2253 | \
			XN_FLAG_SEP_COMMA_PLUS | \
			XN_FLAG_DN_REV | \
			XN_FLAG_FN_SN | \
			XN_FLAG_DUMP_UNKNOWN_FIELDS)

/* readable oneline form */

#define XN_FLAG_ONELINE (ASN1_STRFLGS_RFC2253 | \
			ASN1_STRFLGS_ESC_QUOTE | \
			XN_FLAG_SEP_CPLUS_SPC | \
			XN_FLAG_SPC_EQ | \
			XN_FLAG_FN_SN)

/* readable multiline form */

#define XN_FLAG_MULTILINE (ASN1_STRFLGS_ESC_CTRL | \
			ASN1_STRFLGS_ESC_MSB | \
			XN_FLAG_SEP_MULTILINE | \
			XN_FLAG_SPC_EQ | \
			XN_FLAG_FN_LN | \
			XN_FLAG_FN_ALIGN)

struct x509_revoked_st
	{
	ASN1_INTEGER *serialNumber;
	ASN1_TIME *revocationDate;
	STACK_OF(X509_EXTENSION) /* optional */ *extensions;
	/* Set up if indirect CRL */
	STACK_OF(GENERAL_NAME) *issuer;
	/* Revocation reason */
	int reason;
	int sequence; /* load sequence */
	};

DECLARE_STACK_OF(X509_REVOKED)
DECLARE_ASN1_SET_OF(X509_REVOKED)

typedef struct X509_crl_info_st
	{
	ASN1_INTEGER *version;
	X509_ALGOR *sig_alg;
	X509_NAME *issuer;
	ASN1_TIME *lastUpdate;
	ASN1_TIME *nextUpdate;
	STACK_OF(X509_REVOKED) *revoked;
	STACK_OF(X509_EXTENSION) /* [0] */ *extensions;
	ASN1_ENCODING enc;
	} X509_CRL_INFO;

struct X509_crl_st
	{
	/* actual signature */
	X509_CRL_INFO *crl;
	X509_ALGOR *sig_alg;
	ASN1_BIT_STRING *signature;
	CRYPTO_refcount_t references;
	int flags;
	/* Copies of various extensions */
	AUTHORITY_KEYID *akid;
	ISSUING_DIST_POINT *idp;
	/* Convenient breakdown of IDP */
	int idp_flags;
	int idp_reasons;
	/* CRL and base CRL numbers for delta processing */
	ASN1_INTEGER *crl_number;
	ASN1_INTEGER *base_crl_number;
	unsigned char sha1_hash[SHA_DIGEST_LENGTH];
	STACK_OF(GENERAL_NAMES) *issuers;
	const X509_CRL_METHOD *meth;
	void *meth_data;
	} /* X509_CRL */;

DECLARE_STACK_OF(X509_CRL)
DECLARE_ASN1_SET_OF(X509_CRL)

typedef struct private_key_st
	{
	int version;
	/* The PKCS#8 data types */
	X509_ALGOR *enc_algor;
	ASN1_OCTET_STRING *enc_pkey;	/* encrypted pub key */

	/* When decrypted, the following will not be NULL */
	EVP_PKEY *dec_pkey;

	/* used to encrypt and decrypt */
	int key_length;
	char *key_data;
	int key_free;	/* true if we should auto free key_data */

	/* expanded version of 'enc_algor' */
	EVP_CIPHER_INFO cipher;
	} X509_PKEY;

#ifndef OPENSSL_NO_EVP
typedef struct X509_info_st
	{
	X509 *x509;
	X509_CRL *crl;
	X509_PKEY *x_pkey;

	EVP_CIPHER_INFO enc_cipher;
	int enc_len;
	char *enc_data;

	} X509_INFO;

DECLARE_STACK_OF(X509_INFO)
#endif

/* The next 2 structures and their 8 routines were sent to me by
 * Pat Richard <patr@x509.com> and are used to manipulate
 * Netscapes spki structures - useful if you are writing a CA web page
 */
typedef struct Netscape_spkac_st
	{
	X509_PUBKEY *pubkey;
	ASN1_IA5STRING *challenge;	/* challenge sent in atlas >= PR2 */
	} NETSCAPE_SPKAC;

typedef struct Netscape_spki_st
	{
	NETSCAPE_SPKAC *spkac;	/* signed public key and challenge */
	X509_ALGOR *sig_algor;
	ASN1_BIT_STRING *signature;
	} NETSCAPE_SPKI;

/* Netscape certificate sequence structure */
typedef struct Netscape_certificate_sequence
	{
	ASN1_OBJECT *type;
	STACK_OF(X509) *certs;
	} NETSCAPE_CERT_SEQUENCE;

/* Unused (and iv length is wrong)
typedef struct CBCParameter_st
	{
	unsigned char iv[8];
	} CBC_PARAM;
*/

/* Password based encryption structure */

typedef struct PBEPARAM_st {
ASN1_OCTET_STRING *salt;
ASN1_INTEGER *iter;
} PBEPARAM;

/* Password based encryption V2 structures */

typedef struct PBE2PARAM_st {
X509_ALGOR *keyfunc;
X509_ALGOR *encryption;
} PBE2PARAM;

typedef struct PBKDF2PARAM_st {
ASN1_TYPE *salt;	/* Usually OCTET STRING but could be anything */
ASN1_INTEGER *iter;
ASN1_INTEGER *keylength;
X509_ALGOR *prf;
} PBKDF2PARAM;


/* PKCS#8 private key info structure */

struct pkcs8_priv_key_info_st
        {
        int broken;     /* Flag for various broken formats */
#define PKCS8_OK		0
#define PKCS8_NO_OCTET		1
#define PKCS8_EMBEDDED_PARAM	2
#define PKCS8_NS_DB		3
#define PKCS8_NEG_PRIVKEY	4
        ASN1_INTEGER *version;
        X509_ALGOR *pkeyalg;
        ASN1_TYPE *pkey; /* Should be OCTET STRING but some are broken */
        STACK_OF(X509_ATTRIBUTE) *attributes;
        };

#ifdef  __cplusplus
}
#endif

#include <openssl/x509_vfy.h>

#ifdef  __cplusplus
extern "C" {
#endif

#define X509_EXT_PACK_UNKNOWN	1
#define X509_EXT_PACK_STRING	2

#define		X509_get_version(x) ASN1_INTEGER_get((x)->cert_info->version)
/* #define	X509_get_serialNumber(x) ((x)->cert_info->serialNumber) */
#define		X509_get_notBefore(x) ((x)->cert_info->validity->notBefore)
#define		X509_get_notAfter(x) ((x)->cert_info->validity->notAfter)
#define		X509_get_cert_info(x) ((x)->cert_info)
#define		X509_extract_key(x)	X509_get_pubkey(x) /*****/
#define		X509_REQ_get_version(x) ASN1_INTEGER_get((x)->req_info->version)
#define		X509_REQ_get_subject_name(x) ((x)->req_info->subject)
#define		X509_REQ_extract_key(a)	X509_REQ_get_pubkey(a)
#define		X509_name_cmp(a,b)	X509_NAME_cmp((a),(b))
#define		X509_get_signature_type(x) EVP_PKEY_type(OBJ_obj2nid((x)->sig_alg->algorithm))

#define		X509_CRL_get_version(x) ASN1_INTEGER_get((x)->crl->version)
#define 	X509_CRL_get_lastUpdate(x) ((x)->crl->lastUpdate)
#define 	X509_CRL_get_nextUpdate(x) ((x)->crl->nextUpdate)
#define		X509_CRL_get_issuer(x) ((x)->crl->issuer)
#define		X509_CRL_get_REVOKED(x) ((x)->crl->revoked)

#define		X509_CINF_set_modified(c) ((c)->enc.modified = 1)
#define		X509_CINF_get_issuer(c) (&(c)->issuer)
#define		X509_CINF_get_extensions(c) ((c)->extensions)
#define		X509_CINF_get_signature(c) ((c)->signature)

OPENSSL_EXPORT void X509_CRL_set_default_method(const X509_CRL_METHOD *meth);
OPENSSL_EXPORT X509_CRL_METHOD *X509_CRL_METHOD_new(
	int (*crl_init)(X509_CRL *crl),
	int (*crl_free)(X509_CRL *crl),
	int (*crl_lookup)(X509_CRL *crl, X509_REVOKED **ret,
				ASN1_INTEGER *ser, X509_NAME *issuer),
	int (*crl_verify)(X509_CRL *crl, EVP_PKEY *pk));
OPENSSL_EXPORT void X509_CRL_METHOD_free(X509_CRL_METHOD *m);

OPENSSL_EXPORT void X509_CRL_set_meth_data(X509_CRL *crl, void *dat);
OPENSSL_EXPORT void *X509_CRL_get_meth_data(X509_CRL *crl);

/* This one is only used so that a binary form can output, as in
 * i2d_X509_NAME(X509_get_X509_PUBKEY(x),&buf) */
#define 	X509_get_X509_PUBKEY(x) ((x)->cert_info->key)


OPENSSL_EXPORT const char *X509_verify_cert_error_string(long n);

#ifndef OPENSSL_NO_EVP
OPENSSL_EXPORT int X509_verify(X509 *a, EVP_PKEY *r);

OPENSSL_EXPORT int X509_REQ_verify(X509_REQ *a, EVP_PKEY *r);
OPENSSL_EXPORT int X509_CRL_verify(X509_CRL *a, EVP_PKEY *r);
OPENSSL_EXPORT int NETSCAPE_SPKI_verify(NETSCAPE_SPKI *a, EVP_PKEY *r);

OPENSSL_EXPORT NETSCAPE_SPKI * NETSCAPE_SPKI_b64_decode(const char *str, int len);
OPENSSL_EXPORT char * NETSCAPE_SPKI_b64_encode(NETSCAPE_SPKI *x);
OPENSSL_EXPORT EVP_PKEY *NETSCAPE_SPKI_get_pubkey(NETSCAPE_SPKI *x);
OPENSSL_EXPORT int NETSCAPE_SPKI_set_pubkey(NETSCAPE_SPKI *x, EVP_PKEY *pkey);

OPENSSL_EXPORT int NETSCAPE_SPKI_print(BIO *out, NETSCAPE_SPKI *spki);

OPENSSL_EXPORT int X509_signature_dump(BIO *bp,const ASN1_STRING *sig, int indent);
OPENSSL_EXPORT int X509_signature_print(BIO *bp,X509_ALGOR *alg, ASN1_STRING *sig);

OPENSSL_EXPORT int X509_sign(X509 *x, EVP_PKEY *pkey, const EVP_MD *md);
OPENSSL_EXPORT int X509_sign_ctx(X509 *x, EVP_MD_CTX *ctx);
/* int X509_http_nbio(OCSP_REQ_CTX *rctx, X509 **pcert); */
OPENSSL_EXPORT int X509_REQ_sign(X509_REQ *x, EVP_PKEY *pkey, const EVP_MD *md);
OPENSSL_EXPORT int X509_REQ_sign_ctx(X509_REQ *x, EVP_MD_CTX *ctx);
OPENSSL_EXPORT int X509_CRL_sign(X509_CRL *x, EVP_PKEY *pkey, const EVP_MD *md);
OPENSSL_EXPORT int X509_CRL_sign_ctx(X509_CRL *x, EVP_MD_CTX *ctx);
/* int X509_CRL_http_nbio(OCSP_REQ_CTX *rctx, X509_CRL **pcrl); */
OPENSSL_EXPORT int NETSCAPE_SPKI_sign(NETSCAPE_SPKI *x, EVP_PKEY *pkey, const EVP_MD *md);

OPENSSL_EXPORT int X509_pubkey_digest(const X509 *data,const EVP_MD *type,
		unsigned char *md, unsigned int *len);
OPENSSL_EXPORT int X509_digest(const X509 *data,const EVP_MD *type,
		unsigned char *md, unsigned int *len);
OPENSSL_EXPORT int X509_CRL_digest(const X509_CRL *data,const EVP_MD *type,
		unsigned char *md, unsigned int *len);
OPENSSL_EXPORT int X509_REQ_digest(const X509_REQ *data,const EVP_MD *type,
		unsigned char *md, unsigned int *len);
OPENSSL_EXPORT int X509_NAME_digest(const X509_NAME *data,const EVP_MD *type,
		unsigned char *md, unsigned int *len);
#endif

#ifndef OPENSSL_NO_FP_API
OPENSSL_EXPORT X509 *d2i_X509_fp(FILE *fp, X509 **x509);
OPENSSL_EXPORT int i2d_X509_fp(FILE *fp,X509 *x509);
OPENSSL_EXPORT X509_CRL *d2i_X509_CRL_fp(FILE *fp,X509_CRL **crl);
OPENSSL_EXPORT int i2d_X509_CRL_fp(FILE *fp,X509_CRL *crl);
OPENSSL_EXPORT X509_REQ *d2i_X509_REQ_fp(FILE *fp,X509_REQ **req);
OPENSSL_EXPORT int i2d_X509_REQ_fp(FILE *fp,X509_REQ *req);
OPENSSL_EXPORT RSA *d2i_RSAPrivateKey_fp(FILE *fp,RSA **rsa);
OPENSSL_EXPORT int i2d_RSAPrivateKey_fp(FILE *fp,RSA *rsa);
OPENSSL_EXPORT RSA *d2i_RSAPublicKey_fp(FILE *fp,RSA **rsa);
OPENSSL_EXPORT int i2d_RSAPublicKey_fp(FILE *fp,RSA *rsa);
OPENSSL_EXPORT RSA *d2i_RSA_PUBKEY_fp(FILE *fp,RSA **rsa);
OPENSSL_EXPORT int i2d_RSA_PUBKEY_fp(FILE *fp,RSA *rsa);
#ifndef OPENSSL_NO_DSA
OPENSSL_EXPORT DSA *d2i_DSA_PUBKEY_fp(FILE *fp, DSA **dsa);
OPENSSL_EXPORT int i2d_DSA_PUBKEY_fp(FILE *fp, DSA *dsa);
OPENSSL_EXPORT DSA *d2i_DSAPrivateKey_fp(FILE *fp, DSA **dsa);
OPENSSL_EXPORT int i2d_DSAPrivateKey_fp(FILE *fp, DSA *dsa);
#endif
OPENSSL_EXPORT EC_KEY *d2i_EC_PUBKEY_fp(FILE *fp, EC_KEY **eckey);
OPENSSL_EXPORT int   i2d_EC_PUBKEY_fp(FILE *fp, EC_KEY *eckey);
OPENSSL_EXPORT EC_KEY *d2i_ECPrivateKey_fp(FILE *fp, EC_KEY **eckey);
OPENSSL_EXPORT int   i2d_ECPrivateKey_fp(FILE *fp, EC_KEY *eckey);
OPENSSL_EXPORT X509_SIG *d2i_PKCS8_fp(FILE *fp,X509_SIG **p8);
OPENSSL_EXPORT int i2d_PKCS8_fp(FILE *fp,X509_SIG *p8);
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *d2i_PKCS8_PRIV_KEY_INFO_fp(FILE *fp,
						PKCS8_PRIV_KEY_INFO **p8inf);
OPENSSL_EXPORT int i2d_PKCS8_PRIV_KEY_INFO_fp(FILE *fp,PKCS8_PRIV_KEY_INFO *p8inf);
OPENSSL_EXPORT int i2d_PKCS8PrivateKeyInfo_fp(FILE *fp, EVP_PKEY *key);
OPENSSL_EXPORT int i2d_PrivateKey_fp(FILE *fp, EVP_PKEY *pkey);
OPENSSL_EXPORT EVP_PKEY *d2i_PrivateKey_fp(FILE *fp, EVP_PKEY **a);
OPENSSL_EXPORT int i2d_PUBKEY_fp(FILE *fp, EVP_PKEY *pkey);
OPENSSL_EXPORT EVP_PKEY *d2i_PUBKEY_fp(FILE *fp, EVP_PKEY **a);
#endif

OPENSSL_EXPORT X509 *d2i_X509_bio(BIO *bp,X509 **x509);
OPENSSL_EXPORT int i2d_X509_bio(BIO *bp,X509 *x509);
OPENSSL_EXPORT X509_CRL *d2i_X509_CRL_bio(BIO *bp,X509_CRL **crl);
OPENSSL_EXPORT int i2d_X509_CRL_bio(BIO *bp,X509_CRL *crl);
OPENSSL_EXPORT X509_REQ *d2i_X509_REQ_bio(BIO *bp,X509_REQ **req);
OPENSSL_EXPORT int i2d_X509_REQ_bio(BIO *bp,X509_REQ *req);
OPENSSL_EXPORT RSA *d2i_RSAPrivateKey_bio(BIO *bp,RSA **rsa);
OPENSSL_EXPORT int i2d_RSAPrivateKey_bio(BIO *bp,RSA *rsa);
OPENSSL_EXPORT RSA *d2i_RSAPublicKey_bio(BIO *bp,RSA **rsa);
OPENSSL_EXPORT int i2d_RSAPublicKey_bio(BIO *bp,RSA *rsa);
OPENSSL_EXPORT RSA *d2i_RSA_PUBKEY_bio(BIO *bp,RSA **rsa);
OPENSSL_EXPORT int i2d_RSA_PUBKEY_bio(BIO *bp,RSA *rsa);
#ifndef OPENSSL_NO_DSA
OPENSSL_EXPORT DSA *d2i_DSA_PUBKEY_bio(BIO *bp, DSA **dsa);
OPENSSL_EXPORT int i2d_DSA_PUBKEY_bio(BIO *bp, DSA *dsa);
OPENSSL_EXPORT DSA *d2i_DSAPrivateKey_bio(BIO *bp, DSA **dsa);
OPENSSL_EXPORT int i2d_DSAPrivateKey_bio(BIO *bp, DSA *dsa);
#endif
OPENSSL_EXPORT EC_KEY *d2i_EC_PUBKEY_bio(BIO *bp, EC_KEY **eckey);
OPENSSL_EXPORT int   i2d_EC_PUBKEY_bio(BIO *bp, EC_KEY *eckey);
OPENSSL_EXPORT EC_KEY *d2i_ECPrivateKey_bio(BIO *bp, EC_KEY **eckey);
OPENSSL_EXPORT int   i2d_ECPrivateKey_bio(BIO *bp, EC_KEY *eckey);
OPENSSL_EXPORT X509_SIG *d2i_PKCS8_bio(BIO *bp,X509_SIG **p8);
OPENSSL_EXPORT int i2d_PKCS8_bio(BIO *bp,X509_SIG *p8);
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *d2i_PKCS8_PRIV_KEY_INFO_bio(BIO *bp,
						PKCS8_PRIV_KEY_INFO **p8inf);
OPENSSL_EXPORT int i2d_PKCS8_PRIV_KEY_INFO_bio(BIO *bp,PKCS8_PRIV_KEY_INFO *p8inf);
OPENSSL_EXPORT int i2d_PKCS8PrivateKeyInfo_bio(BIO *bp, EVP_PKEY *key);
OPENSSL_EXPORT int i2d_PrivateKey_bio(BIO *bp, EVP_PKEY *pkey);
OPENSSL_EXPORT EVP_PKEY *d2i_PrivateKey_bio(BIO *bp, EVP_PKEY **a);
OPENSSL_EXPORT int i2d_PUBKEY_bio(BIO *bp, EVP_PKEY *pkey);
OPENSSL_EXPORT EVP_PKEY *d2i_PUBKEY_bio(BIO *bp, EVP_PKEY **a);

OPENSSL_EXPORT X509 *X509_dup(X509 *x509);
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_dup(X509_ATTRIBUTE *xa);
OPENSSL_EXPORT X509_EXTENSION *X509_EXTENSION_dup(X509_EXTENSION *ex);
OPENSSL_EXPORT X509_CRL *X509_CRL_dup(X509_CRL *crl);
OPENSSL_EXPORT X509_REVOKED *X509_REVOKED_dup(X509_REVOKED *rev);
OPENSSL_EXPORT X509_REQ *X509_REQ_dup(X509_REQ *req);
OPENSSL_EXPORT X509_ALGOR *X509_ALGOR_dup(X509_ALGOR *xn);
OPENSSL_EXPORT int X509_ALGOR_set0(X509_ALGOR *alg, const ASN1_OBJECT *aobj, int ptype, void *pval);
OPENSSL_EXPORT void X509_ALGOR_get0(ASN1_OBJECT **paobj, int *pptype, void **ppval,
						X509_ALGOR *algor);
OPENSSL_EXPORT void X509_ALGOR_set_md(X509_ALGOR *alg, const EVP_MD *md);
OPENSSL_EXPORT int X509_ALGOR_cmp(const X509_ALGOR *a, const X509_ALGOR *b);

OPENSSL_EXPORT X509_NAME *X509_NAME_dup(X509_NAME *xn);
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_ENTRY_dup(X509_NAME_ENTRY *ne);

OPENSSL_EXPORT int		X509_cmp_time(const ASN1_TIME *s, time_t *t);
OPENSSL_EXPORT int		X509_cmp_current_time(const ASN1_TIME *s);
OPENSSL_EXPORT ASN1_TIME *	X509_time_adj(ASN1_TIME *s, long adj, time_t *t);
OPENSSL_EXPORT ASN1_TIME *	X509_time_adj_ex(ASN1_TIME *s, int offset_day, long offset_sec, time_t *t);
OPENSSL_EXPORT ASN1_TIME *	X509_gmtime_adj(ASN1_TIME *s, long adj);

OPENSSL_EXPORT const char *	X509_get_default_cert_area(void );
OPENSSL_EXPORT const char *	X509_get_default_cert_dir(void );
OPENSSL_EXPORT const char *	X509_get_default_cert_file(void );
OPENSSL_EXPORT const char *	X509_get_default_cert_dir_env(void );
OPENSSL_EXPORT const char *	X509_get_default_cert_file_env(void );
OPENSSL_EXPORT const char *	X509_get_default_private_dir(void );

OPENSSL_EXPORT X509_REQ *	X509_to_X509_REQ(X509 *x, EVP_PKEY *pkey, const EVP_MD *md);
OPENSSL_EXPORT X509 *		X509_REQ_to_X509(X509_REQ *r, int days,EVP_PKEY *pkey);

DECLARE_ASN1_ENCODE_FUNCTIONS(X509_ALGORS, X509_ALGORS, X509_ALGORS)
DECLARE_ASN1_FUNCTIONS(X509_VAL)

DECLARE_ASN1_FUNCTIONS(X509_PUBKEY)

OPENSSL_EXPORT int		X509_PUBKEY_set(X509_PUBKEY **x, EVP_PKEY *pkey);
OPENSSL_EXPORT EVP_PKEY *	X509_PUBKEY_get(X509_PUBKEY *key);
OPENSSL_EXPORT int		i2d_PUBKEY(const EVP_PKEY *a,unsigned char **pp);
OPENSSL_EXPORT EVP_PKEY *	d2i_PUBKEY(EVP_PKEY **a,const unsigned char **pp,
			long length);
OPENSSL_EXPORT int		i2d_RSA_PUBKEY(const RSA *a,unsigned char **pp);
OPENSSL_EXPORT RSA *		d2i_RSA_PUBKEY(RSA **a,const unsigned char **pp,
			long length);
#ifndef OPENSSL_NO_DSA
OPENSSL_EXPORT int		i2d_DSA_PUBKEY(const DSA *a,unsigned char **pp);
OPENSSL_EXPORT DSA *		d2i_DSA_PUBKEY(DSA **a,const unsigned char **pp,
			long length);
#endif
OPENSSL_EXPORT int		i2d_EC_PUBKEY(const EC_KEY *a, unsigned char **pp);
OPENSSL_EXPORT EC_KEY 		*d2i_EC_PUBKEY(EC_KEY **a, const unsigned char **pp,
			long length);

DECLARE_ASN1_FUNCTIONS(X509_SIG)
DECLARE_ASN1_FUNCTIONS(X509_REQ_INFO)
DECLARE_ASN1_FUNCTIONS(X509_REQ)

DECLARE_ASN1_FUNCTIONS(X509_ATTRIBUTE)
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_create(int nid, int atrtype, void *value);

DECLARE_ASN1_FUNCTIONS(X509_EXTENSION)
DECLARE_ASN1_ENCODE_FUNCTIONS(X509_EXTENSIONS, X509_EXTENSIONS, X509_EXTENSIONS)

DECLARE_ASN1_FUNCTIONS(X509_NAME_ENTRY)

DECLARE_ASN1_FUNCTIONS(X509_NAME)

OPENSSL_EXPORT int		X509_NAME_set(X509_NAME **xn, X509_NAME *name);

DECLARE_ASN1_FUNCTIONS(X509_CINF)

DECLARE_ASN1_FUNCTIONS(X509)
DECLARE_ASN1_FUNCTIONS(X509_CERT_AUX)

DECLARE_ASN1_FUNCTIONS(X509_CERT_PAIR)

/* X509_up_ref adds one to the reference count of |x| and returns
 * |x|. */
OPENSSL_EXPORT X509 *X509_up_ref(X509 *x);

OPENSSL_EXPORT int X509_get_ex_new_index(long argl, void *argp, CRYPTO_EX_new *new_func,
	     CRYPTO_EX_dup *dup_func, CRYPTO_EX_free *free_func);
OPENSSL_EXPORT int X509_set_ex_data(X509 *r, int idx, void *arg);
OPENSSL_EXPORT void *X509_get_ex_data(X509 *r, int idx);
OPENSSL_EXPORT int		i2d_X509_AUX(X509 *a,unsigned char **pp);
OPENSSL_EXPORT X509 *		d2i_X509_AUX(X509 **a,const unsigned char **pp,long length);

OPENSSL_EXPORT void X509_get0_signature(ASN1_BIT_STRING **psig, X509_ALGOR **palg,
								const X509 *x);
OPENSSL_EXPORT int X509_get_signature_nid(const X509 *x);

OPENSSL_EXPORT int X509_alias_set1(X509 *x, unsigned char *name, int len);
OPENSSL_EXPORT int X509_keyid_set1(X509 *x, unsigned char *id, int len);
OPENSSL_EXPORT unsigned char * X509_alias_get0(X509 *x, int *len);
OPENSSL_EXPORT unsigned char * X509_keyid_get0(X509 *x, int *len);
OPENSSL_EXPORT int (*X509_TRUST_set_default(int (*trust)(int , X509 *, int)))(int, X509 *, int);
OPENSSL_EXPORT int X509_TRUST_set(int *t, int trust);
OPENSSL_EXPORT int X509_add1_trust_object(X509 *x, ASN1_OBJECT *obj);
OPENSSL_EXPORT int X509_add1_reject_object(X509 *x, ASN1_OBJECT *obj);
OPENSSL_EXPORT void X509_trust_clear(X509 *x);
OPENSSL_EXPORT void X509_reject_clear(X509 *x);

DECLARE_ASN1_FUNCTIONS(X509_REVOKED)
DECLARE_ASN1_FUNCTIONS(X509_CRL_INFO)
DECLARE_ASN1_FUNCTIONS(X509_CRL)

OPENSSL_EXPORT int X509_CRL_add0_revoked(X509_CRL *crl, X509_REVOKED *rev);
OPENSSL_EXPORT int X509_CRL_get0_by_serial(X509_CRL *crl,
		X509_REVOKED **ret, ASN1_INTEGER *serial);
OPENSSL_EXPORT int X509_CRL_get0_by_cert(X509_CRL *crl, X509_REVOKED **ret, X509 *x);

OPENSSL_EXPORT X509_PKEY *	X509_PKEY_new(void );
OPENSSL_EXPORT void		X509_PKEY_free(X509_PKEY *a);

DECLARE_ASN1_FUNCTIONS(NETSCAPE_SPKI)
DECLARE_ASN1_FUNCTIONS(NETSCAPE_SPKAC)
DECLARE_ASN1_FUNCTIONS(NETSCAPE_CERT_SEQUENCE)

#ifndef OPENSSL_NO_EVP
OPENSSL_EXPORT X509_INFO *	X509_INFO_new(void);
OPENSSL_EXPORT void		X509_INFO_free(X509_INFO *a);
OPENSSL_EXPORT char *		X509_NAME_oneline(X509_NAME *a,char *buf,int size);

OPENSSL_EXPORT int ASN1_digest(i2d_of_void *i2d,const EVP_MD *type,char *data,
		unsigned char *md,unsigned int *len);

OPENSSL_EXPORT int ASN1_item_digest(const ASN1_ITEM *it,const EVP_MD *type,void *data,
	unsigned char *md,unsigned int *len);

OPENSSL_EXPORT int ASN1_item_verify(const ASN1_ITEM *it, X509_ALGOR *algor1,
	ASN1_BIT_STRING *signature,void *data,EVP_PKEY *pkey);

OPENSSL_EXPORT int ASN1_item_sign(const ASN1_ITEM *it, X509_ALGOR *algor1, X509_ALGOR *algor2,
	ASN1_BIT_STRING *signature,
	void *data, EVP_PKEY *pkey, const EVP_MD *type);
OPENSSL_EXPORT int ASN1_item_sign_ctx(const ASN1_ITEM *it,
		X509_ALGOR *algor1, X509_ALGOR *algor2,
	     	ASN1_BIT_STRING *signature, void *asn, EVP_MD_CTX *ctx);
#endif

OPENSSL_EXPORT int 		X509_set_version(X509 *x,long version);
OPENSSL_EXPORT int 		X509_set_serialNumber(X509 *x, ASN1_INTEGER *serial);
OPENSSL_EXPORT ASN1_INTEGER *	X509_get_serialNumber(X509 *x);
OPENSSL_EXPORT int 		X509_set_issuer_name(X509 *x, X509_NAME *name);
OPENSSL_EXPORT X509_NAME *	X509_get_issuer_name(X509 *a);
OPENSSL_EXPORT int 		X509_set_subject_name(X509 *x, X509_NAME *name);
OPENSSL_EXPORT X509_NAME *	X509_get_subject_name(X509 *a);
OPENSSL_EXPORT int 		X509_set_notBefore(X509 *x, const ASN1_TIME *tm);
OPENSSL_EXPORT int 		X509_set_notAfter(X509 *x, const ASN1_TIME *tm);
OPENSSL_EXPORT int 		X509_set_pubkey(X509 *x, EVP_PKEY *pkey);
OPENSSL_EXPORT EVP_PKEY *	X509_get_pubkey(X509 *x);
OPENSSL_EXPORT ASN1_BIT_STRING * X509_get0_pubkey_bitstr(const X509 *x);
OPENSSL_EXPORT int		X509_certificate_type(X509 *x,EVP_PKEY *pubkey /* optional */);

OPENSSL_EXPORT int		X509_REQ_set_version(X509_REQ *x,long version);
OPENSSL_EXPORT int		X509_REQ_set_subject_name(X509_REQ *req,X509_NAME *name);
OPENSSL_EXPORT int		X509_REQ_set_pubkey(X509_REQ *x, EVP_PKEY *pkey);
OPENSSL_EXPORT EVP_PKEY *	X509_REQ_get_pubkey(X509_REQ *req);
OPENSSL_EXPORT int		X509_REQ_extension_nid(int nid);
OPENSSL_EXPORT const int *	X509_REQ_get_extension_nids(void);
OPENSSL_EXPORT void		X509_REQ_set_extension_nids(const int *nids);
OPENSSL_EXPORT STACK_OF(X509_EXTENSION) *X509_REQ_get_extensions(X509_REQ *req);
OPENSSL_EXPORT int X509_REQ_add_extensions_nid(X509_REQ *req, STACK_OF(X509_EXTENSION) *exts,
				int nid);
OPENSSL_EXPORT int X509_REQ_add_extensions(X509_REQ *req, STACK_OF(X509_EXTENSION) *exts);
OPENSSL_EXPORT int X509_REQ_get_attr_count(const X509_REQ *req);
OPENSSL_EXPORT int X509_REQ_get_attr_by_NID(const X509_REQ *req, int nid,
			  int lastpos);
OPENSSL_EXPORT int X509_REQ_get_attr_by_OBJ(const X509_REQ *req, ASN1_OBJECT *obj,
			  int lastpos);
OPENSSL_EXPORT X509_ATTRIBUTE *X509_REQ_get_attr(const X509_REQ *req, int loc);
OPENSSL_EXPORT X509_ATTRIBUTE *X509_REQ_delete_attr(X509_REQ *req, int loc);
OPENSSL_EXPORT int X509_REQ_add1_attr(X509_REQ *req, X509_ATTRIBUTE *attr);
OPENSSL_EXPORT int X509_REQ_add1_attr_by_OBJ(X509_REQ *req,
			const ASN1_OBJECT *obj, int type,
			const unsigned char *bytes, int len);
OPENSSL_EXPORT int X509_REQ_add1_attr_by_NID(X509_REQ *req,
			int nid, int type,
			const unsigned char *bytes, int len);
OPENSSL_EXPORT int X509_REQ_add1_attr_by_txt(X509_REQ *req,
			const char *attrname, int type,
			const unsigned char *bytes, int len);

OPENSSL_EXPORT int X509_CRL_set_version(X509_CRL *x, long version);
OPENSSL_EXPORT int X509_CRL_set_issuer_name(X509_CRL *x, X509_NAME *name);
OPENSSL_EXPORT int X509_CRL_set_lastUpdate(X509_CRL *x, const ASN1_TIME *tm);
OPENSSL_EXPORT int X509_CRL_set_nextUpdate(X509_CRL *x, const ASN1_TIME *tm);
OPENSSL_EXPORT int X509_CRL_sort(X509_CRL *crl);

OPENSSL_EXPORT int X509_REVOKED_set_serialNumber(X509_REVOKED *x, ASN1_INTEGER *serial);
OPENSSL_EXPORT int X509_REVOKED_set_revocationDate(X509_REVOKED *r, ASN1_TIME *tm);

OPENSSL_EXPORT X509_CRL *X509_CRL_diff(X509_CRL *base, X509_CRL *newer,
			EVP_PKEY *skey, const EVP_MD *md, unsigned int flags);

OPENSSL_EXPORT int		X509_REQ_check_private_key(X509_REQ *x509,EVP_PKEY *pkey);

OPENSSL_EXPORT int		X509_check_private_key(X509 *x509,EVP_PKEY *pkey);
OPENSSL_EXPORT int 		X509_chain_check_suiteb(int *perror_depth,
						X509 *x, STACK_OF(X509) *chain,
						unsigned long flags);
OPENSSL_EXPORT int 		X509_CRL_check_suiteb(X509_CRL *crl, EVP_PKEY *pk,
						unsigned long flags);
OPENSSL_EXPORT STACK_OF(X509) *X509_chain_up_ref(STACK_OF(X509) *chain);

OPENSSL_EXPORT int		X509_issuer_and_serial_cmp(const X509 *a, const X509 *b);
OPENSSL_EXPORT unsigned long	X509_issuer_and_serial_hash(X509 *a);

OPENSSL_EXPORT int		X509_issuer_name_cmp(const X509 *a, const X509 *b);
OPENSSL_EXPORT unsigned long	X509_issuer_name_hash(X509 *a);

OPENSSL_EXPORT int		X509_subject_name_cmp(const X509 *a, const X509 *b);
OPENSSL_EXPORT unsigned long	X509_subject_name_hash(X509 *x);

OPENSSL_EXPORT unsigned long	X509_issuer_name_hash_old(X509 *a);
OPENSSL_EXPORT unsigned long	X509_subject_name_hash_old(X509 *x);

OPENSSL_EXPORT int		X509_cmp(const X509 *a, const X509 *b);
OPENSSL_EXPORT int		X509_NAME_cmp(const X509_NAME *a, const X509_NAME *b);
OPENSSL_EXPORT unsigned long	X509_NAME_hash(X509_NAME *x);
OPENSSL_EXPORT unsigned long	X509_NAME_hash_old(X509_NAME *x);

OPENSSL_EXPORT int		X509_CRL_cmp(const X509_CRL *a, const X509_CRL *b);
OPENSSL_EXPORT int		X509_CRL_match(const X509_CRL *a, const X509_CRL *b);
#ifndef OPENSSL_NO_FP_API
OPENSSL_EXPORT int		X509_print_ex_fp(FILE *bp,X509 *x, unsigned long nmflag, unsigned long cflag);
OPENSSL_EXPORT int		X509_print_fp(FILE *bp,X509 *x);
OPENSSL_EXPORT int		X509_CRL_print_fp(FILE *bp,X509_CRL *x);
OPENSSL_EXPORT int		X509_REQ_print_fp(FILE *bp,X509_REQ *req);
OPENSSL_EXPORT int X509_NAME_print_ex_fp(FILE *fp, X509_NAME *nm, int indent, unsigned long flags);
#endif

OPENSSL_EXPORT int		X509_NAME_print(BIO *bp, X509_NAME *name, int obase);
OPENSSL_EXPORT int X509_NAME_print_ex(BIO *out, X509_NAME *nm, int indent, unsigned long flags);
OPENSSL_EXPORT int		X509_print_ex(BIO *bp,X509 *x, unsigned long nmflag, unsigned long cflag);
OPENSSL_EXPORT int		X509_print(BIO *bp,X509 *x);
OPENSSL_EXPORT int		X509_ocspid_print(BIO *bp,X509 *x);
OPENSSL_EXPORT int		X509_CERT_AUX_print(BIO *bp,X509_CERT_AUX *x, int indent);
OPENSSL_EXPORT int		X509_CRL_print(BIO *bp,X509_CRL *x);
OPENSSL_EXPORT int		X509_REQ_print_ex(BIO *bp, X509_REQ *x, unsigned long nmflag, unsigned long cflag);
OPENSSL_EXPORT int		X509_REQ_print(BIO *bp,X509_REQ *req);

OPENSSL_EXPORT int 		X509_NAME_entry_count(X509_NAME *name);
OPENSSL_EXPORT int 		X509_NAME_get_text_by_NID(X509_NAME *name, int nid,
			char *buf,int len);
OPENSSL_EXPORT int		X509_NAME_get_text_by_OBJ(X509_NAME *name, const ASN1_OBJECT *obj,
			char *buf,int len);

/* NOTE: you should be passsing -1, not 0 as lastpos.  The functions that use
 * lastpos, search after that position on. */
OPENSSL_EXPORT int 		X509_NAME_get_index_by_NID(X509_NAME *name,int nid,int lastpos);
OPENSSL_EXPORT int 		X509_NAME_get_index_by_OBJ(X509_NAME *name, const ASN1_OBJECT *obj,
			int lastpos);
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_get_entry(X509_NAME *name, int loc);
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_delete_entry(X509_NAME *name, int loc);
OPENSSL_EXPORT int 		X509_NAME_add_entry(X509_NAME *name,X509_NAME_ENTRY *ne,
			int loc, int set);
OPENSSL_EXPORT int X509_NAME_add_entry_by_OBJ(X509_NAME *name, ASN1_OBJECT *obj, int type,
			unsigned char *bytes, int len, int loc, int set);
OPENSSL_EXPORT int X509_NAME_add_entry_by_NID(X509_NAME *name, int nid, int type,
			unsigned char *bytes, int len, int loc, int set);
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_ENTRY_create_by_txt(X509_NAME_ENTRY **ne,
		const char *field, int type, const unsigned char *bytes, int len);
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_ENTRY_create_by_NID(X509_NAME_ENTRY **ne, int nid,
			int type,unsigned char *bytes, int len);
OPENSSL_EXPORT int X509_NAME_add_entry_by_txt(X509_NAME *name, const char *field, int type,
			const unsigned char *bytes, int len, int loc, int set);
OPENSSL_EXPORT X509_NAME_ENTRY *X509_NAME_ENTRY_create_by_OBJ(X509_NAME_ENTRY **ne,
			const ASN1_OBJECT *obj, int type,const unsigned char *bytes,
			int len);
OPENSSL_EXPORT int 		X509_NAME_ENTRY_set_object(X509_NAME_ENTRY *ne,
			const ASN1_OBJECT *obj);
OPENSSL_EXPORT int 		X509_NAME_ENTRY_set_data(X509_NAME_ENTRY *ne, int type,
			const unsigned char *bytes, int len);
OPENSSL_EXPORT ASN1_OBJECT *	X509_NAME_ENTRY_get_object(X509_NAME_ENTRY *ne);
OPENSSL_EXPORT ASN1_STRING *	X509_NAME_ENTRY_get_data(X509_NAME_ENTRY *ne);

OPENSSL_EXPORT int		X509v3_get_ext_count(const STACK_OF(X509_EXTENSION) *x);
OPENSSL_EXPORT int		X509v3_get_ext_by_NID(const STACK_OF(X509_EXTENSION) *x,
				      int nid, int lastpos);
OPENSSL_EXPORT int		X509v3_get_ext_by_OBJ(const STACK_OF(X509_EXTENSION) *x,
				      const ASN1_OBJECT *obj,int lastpos);
OPENSSL_EXPORT int		X509v3_get_ext_by_critical(const STACK_OF(X509_EXTENSION) *x,
					   int crit, int lastpos);
OPENSSL_EXPORT X509_EXTENSION *X509v3_get_ext(const STACK_OF(X509_EXTENSION) *x, int loc);
OPENSSL_EXPORT X509_EXTENSION *X509v3_delete_ext(STACK_OF(X509_EXTENSION) *x, int loc);
OPENSSL_EXPORT STACK_OF(X509_EXTENSION) *X509v3_add_ext(STACK_OF(X509_EXTENSION) **x,
					 X509_EXTENSION *ex, int loc);

OPENSSL_EXPORT int		X509_get_ext_count(X509 *x);
OPENSSL_EXPORT int		X509_get_ext_by_NID(X509 *x, int nid, int lastpos);
OPENSSL_EXPORT int		X509_get_ext_by_OBJ(X509 *x,ASN1_OBJECT *obj,int lastpos);
OPENSSL_EXPORT int		X509_get_ext_by_critical(X509 *x, int crit, int lastpos);
OPENSSL_EXPORT X509_EXTENSION *X509_get_ext(X509 *x, int loc);
OPENSSL_EXPORT X509_EXTENSION *X509_delete_ext(X509 *x, int loc);
OPENSSL_EXPORT int		X509_add_ext(X509 *x, X509_EXTENSION *ex, int loc);
OPENSSL_EXPORT void	*	X509_get_ext_d2i(X509 *x, int nid, int *crit, int *idx);
OPENSSL_EXPORT int		X509_add1_ext_i2d(X509 *x, int nid, void *value, int crit,
							unsigned long flags);

OPENSSL_EXPORT int		X509_CRL_get_ext_count(X509_CRL *x);
OPENSSL_EXPORT int		X509_CRL_get_ext_by_NID(X509_CRL *x, int nid, int lastpos);
OPENSSL_EXPORT int		X509_CRL_get_ext_by_OBJ(X509_CRL *x,ASN1_OBJECT *obj,int lastpos);
OPENSSL_EXPORT int		X509_CRL_get_ext_by_critical(X509_CRL *x, int crit, int lastpos);
OPENSSL_EXPORT X509_EXTENSION *X509_CRL_get_ext(X509_CRL *x, int loc);
OPENSSL_EXPORT X509_EXTENSION *X509_CRL_delete_ext(X509_CRL *x, int loc);
OPENSSL_EXPORT int		X509_CRL_add_ext(X509_CRL *x, X509_EXTENSION *ex, int loc);
OPENSSL_EXPORT void	*	X509_CRL_get_ext_d2i(X509_CRL *x, int nid, int *crit, int *idx);
OPENSSL_EXPORT int		X509_CRL_add1_ext_i2d(X509_CRL *x, int nid, void *value, int crit,
							unsigned long flags);

OPENSSL_EXPORT int		X509_REVOKED_get_ext_count(X509_REVOKED *x);
OPENSSL_EXPORT int		X509_REVOKED_get_ext_by_NID(X509_REVOKED *x, int nid, int lastpos);
OPENSSL_EXPORT int		X509_REVOKED_get_ext_by_OBJ(X509_REVOKED *x,ASN1_OBJECT *obj,int lastpos);
OPENSSL_EXPORT int		X509_REVOKED_get_ext_by_critical(X509_REVOKED *x, int crit, int lastpos);
OPENSSL_EXPORT X509_EXTENSION *X509_REVOKED_get_ext(X509_REVOKED *x, int loc);
OPENSSL_EXPORT X509_EXTENSION *X509_REVOKED_delete_ext(X509_REVOKED *x, int loc);
OPENSSL_EXPORT int		X509_REVOKED_add_ext(X509_REVOKED *x, X509_EXTENSION *ex, int loc);
OPENSSL_EXPORT void	*	X509_REVOKED_get_ext_d2i(X509_REVOKED *x, int nid, int *crit, int *idx);
OPENSSL_EXPORT int		X509_REVOKED_add1_ext_i2d(X509_REVOKED *x, int nid, void *value, int crit,
							unsigned long flags);

OPENSSL_EXPORT X509_EXTENSION *X509_EXTENSION_create_by_NID(X509_EXTENSION **ex,
			int nid, int crit, ASN1_OCTET_STRING *data);
OPENSSL_EXPORT X509_EXTENSION *X509_EXTENSION_create_by_OBJ(X509_EXTENSION **ex,
			const ASN1_OBJECT *obj,int crit,ASN1_OCTET_STRING *data);
OPENSSL_EXPORT int		X509_EXTENSION_set_object(X509_EXTENSION *ex,const ASN1_OBJECT *obj);
OPENSSL_EXPORT int		X509_EXTENSION_set_critical(X509_EXTENSION *ex, int crit);
OPENSSL_EXPORT int		X509_EXTENSION_set_data(X509_EXTENSION *ex,
			ASN1_OCTET_STRING *data);
OPENSSL_EXPORT ASN1_OBJECT *	X509_EXTENSION_get_object(X509_EXTENSION *ex);
OPENSSL_EXPORT ASN1_OCTET_STRING *X509_EXTENSION_get_data(X509_EXTENSION *ne);
OPENSSL_EXPORT int		X509_EXTENSION_get_critical(X509_EXTENSION *ex);

OPENSSL_EXPORT int X509at_get_attr_count(const STACK_OF(X509_ATTRIBUTE) *x);
OPENSSL_EXPORT int X509at_get_attr_by_NID(const STACK_OF(X509_ATTRIBUTE) *x, int nid,
			  int lastpos);
OPENSSL_EXPORT int X509at_get_attr_by_OBJ(const STACK_OF(X509_ATTRIBUTE) *sk, const ASN1_OBJECT *obj,
			  int lastpos);
OPENSSL_EXPORT X509_ATTRIBUTE *X509at_get_attr(const STACK_OF(X509_ATTRIBUTE) *x, int loc);
OPENSSL_EXPORT X509_ATTRIBUTE *X509at_delete_attr(STACK_OF(X509_ATTRIBUTE) *x, int loc);
OPENSSL_EXPORT STACK_OF(X509_ATTRIBUTE) *X509at_add1_attr(STACK_OF(X509_ATTRIBUTE) **x,
					 X509_ATTRIBUTE *attr);
OPENSSL_EXPORT STACK_OF(X509_ATTRIBUTE) *X509at_add1_attr_by_OBJ(STACK_OF(X509_ATTRIBUTE) **x,
			const ASN1_OBJECT *obj, int type,
			const unsigned char *bytes, int len);
OPENSSL_EXPORT STACK_OF(X509_ATTRIBUTE) *X509at_add1_attr_by_NID(STACK_OF(X509_ATTRIBUTE) **x,
			int nid, int type,
			const unsigned char *bytes, int len);
OPENSSL_EXPORT STACK_OF(X509_ATTRIBUTE) *X509at_add1_attr_by_txt(STACK_OF(X509_ATTRIBUTE) **x,
			const char *attrname, int type,
			const unsigned char *bytes, int len);
OPENSSL_EXPORT void *X509at_get0_data_by_OBJ(STACK_OF(X509_ATTRIBUTE) *x,
				ASN1_OBJECT *obj, int lastpos, int type);
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_create_by_NID(X509_ATTRIBUTE **attr, int nid,
	     int atrtype, const void *data, int len);
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_create_by_OBJ(X509_ATTRIBUTE **attr,
	     const ASN1_OBJECT *obj, int atrtype, const void *data, int len);
OPENSSL_EXPORT X509_ATTRIBUTE *X509_ATTRIBUTE_create_by_txt(X509_ATTRIBUTE **attr,
		const char *atrname, int type, const unsigned char *bytes, int len);
OPENSSL_EXPORT int X509_ATTRIBUTE_set1_object(X509_ATTRIBUTE *attr, const ASN1_OBJECT *obj);
OPENSSL_EXPORT int X509_ATTRIBUTE_set1_data(X509_ATTRIBUTE *attr, int attrtype, const void *data, int len);
OPENSSL_EXPORT void *X509_ATTRIBUTE_get0_data(X509_ATTRIBUTE *attr, int idx,
					int atrtype, void *data);
OPENSSL_EXPORT int X509_ATTRIBUTE_count(X509_ATTRIBUTE *attr);
OPENSSL_EXPORT ASN1_OBJECT *X509_ATTRIBUTE_get0_object(X509_ATTRIBUTE *attr);
OPENSSL_EXPORT ASN1_TYPE *X509_ATTRIBUTE_get0_type(X509_ATTRIBUTE *attr, int idx);

OPENSSL_EXPORT int EVP_PKEY_get_attr_count(const EVP_PKEY *key);
OPENSSL_EXPORT int EVP_PKEY_get_attr_by_NID(const EVP_PKEY *key, int nid,
			  int lastpos);
OPENSSL_EXPORT int EVP_PKEY_get_attr_by_OBJ(const EVP_PKEY *key, ASN1_OBJECT *obj,
			  int lastpos);
OPENSSL_EXPORT X509_ATTRIBUTE *EVP_PKEY_get_attr(const EVP_PKEY *key, int loc);
OPENSSL_EXPORT X509_ATTRIBUTE *EVP_PKEY_delete_attr(EVP_PKEY *key, int loc);
OPENSSL_EXPORT int EVP_PKEY_add1_attr(EVP_PKEY *key, X509_ATTRIBUTE *attr);
OPENSSL_EXPORT int EVP_PKEY_add1_attr_by_OBJ(EVP_PKEY *key,
			const ASN1_OBJECT *obj, int type,
			const unsigned char *bytes, int len);
OPENSSL_EXPORT int EVP_PKEY_add1_attr_by_NID(EVP_PKEY *key,
			int nid, int type,
			const unsigned char *bytes, int len);
OPENSSL_EXPORT int EVP_PKEY_add1_attr_by_txt(EVP_PKEY *key,
			const char *attrname, int type,
			const unsigned char *bytes, int len);

OPENSSL_EXPORT int		X509_verify_cert(X509_STORE_CTX *ctx);

/* lookup a cert from a X509 STACK */
OPENSSL_EXPORT X509 *X509_find_by_issuer_and_serial(STACK_OF(X509) *sk,X509_NAME *name,
				     ASN1_INTEGER *serial);
OPENSSL_EXPORT X509 *X509_find_by_subject(STACK_OF(X509) *sk,X509_NAME *name);

DECLARE_ASN1_FUNCTIONS(PBEPARAM)
DECLARE_ASN1_FUNCTIONS(PBE2PARAM)
DECLARE_ASN1_FUNCTIONS(PBKDF2PARAM)

OPENSSL_EXPORT int PKCS5_pbe_set0_algor(X509_ALGOR *algor, int alg, int iter,
				const unsigned char *salt, int saltlen);

OPENSSL_EXPORT X509_ALGOR *PKCS5_pbe_set(int alg, int iter,
				const unsigned char *salt, int saltlen);
OPENSSL_EXPORT X509_ALGOR *PKCS5_pbe2_set(const EVP_CIPHER *cipher, int iter,
					 unsigned char *salt, int saltlen);
OPENSSL_EXPORT X509_ALGOR *PKCS5_pbe2_set_iv(const EVP_CIPHER *cipher, int iter,
				 unsigned char *salt, int saltlen,
				 unsigned char *aiv, int prf_nid);

OPENSSL_EXPORT X509_ALGOR *PKCS5_pbkdf2_set(int iter, unsigned char *salt, int saltlen,
				int prf_nid, int keylen);

/* PKCS#8 utilities */

DECLARE_ASN1_FUNCTIONS(PKCS8_PRIV_KEY_INFO)

OPENSSL_EXPORT EVP_PKEY *EVP_PKCS82PKEY(PKCS8_PRIV_KEY_INFO *p8);
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *EVP_PKEY2PKCS8(EVP_PKEY *pkey);
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *EVP_PKEY2PKCS8_broken(EVP_PKEY *pkey, int broken);
OPENSSL_EXPORT PKCS8_PRIV_KEY_INFO *PKCS8_set_broken(PKCS8_PRIV_KEY_INFO *p8, int broken);

OPENSSL_EXPORT int PKCS8_pkey_set0(PKCS8_PRIV_KEY_INFO *priv, ASN1_OBJECT *aobj,
			int version, int ptype, void *pval,
				unsigned char *penc, int penclen);
OPENSSL_EXPORT int PKCS8_pkey_get0(ASN1_OBJECT **ppkalg,
		const unsigned char **pk, int *ppklen,
		X509_ALGOR **pa,
		PKCS8_PRIV_KEY_INFO *p8);

OPENSSL_EXPORT int X509_PUBKEY_set0_param(X509_PUBKEY *pub, const ASN1_OBJECT *aobj,
					int ptype, void *pval,
					unsigned char *penc, int penclen);
OPENSSL_EXPORT int X509_PUBKEY_get0_param(ASN1_OBJECT **ppkalg,
		const unsigned char **pk, int *ppklen,
		X509_ALGOR **pa,
		X509_PUBKEY *pub);

OPENSSL_EXPORT int X509_check_trust(X509 *x, int id, int flags);
OPENSSL_EXPORT int X509_TRUST_get_count(void);
OPENSSL_EXPORT X509_TRUST * X509_TRUST_get0(int idx);
OPENSSL_EXPORT int X509_TRUST_get_by_id(int id);
OPENSSL_EXPORT int X509_TRUST_add(int id, int flags, int (*ck)(X509_TRUST *, X509 *, int),
					char *name, int arg1, void *arg2);
OPENSSL_EXPORT void X509_TRUST_cleanup(void);
OPENSSL_EXPORT int X509_TRUST_get_flags(X509_TRUST *xp);
OPENSSL_EXPORT char *X509_TRUST_get0_name(X509_TRUST *xp);
OPENSSL_EXPORT int X509_TRUST_get_trust(X509_TRUST *xp);

/* PKCS7_get_certificates parses a PKCS#7, SignedData structure from |cbs| and
 * appends the included certificates to |out_certs|. It returns one on success
 * and zero on error. */
OPENSSL_EXPORT int PKCS7_get_certificates(STACK_OF(X509) *out_certs, CBS *cbs);

/* PKCS7_bundle_certificates appends a PKCS#7, SignedData structure containing
 * |certs| to |out|. It returns one on success and zero on error. */
OPENSSL_EXPORT int PKCS7_bundle_certificates(
    CBB *out, const STACK_OF(X509) *certs);

/* PKCS7_get_CRLs parses a PKCS#7, SignedData structure from |cbs| and appends
 * the included CRLs to |out_crls|. It returns one on success and zero on
 * error. */
OPENSSL_EXPORT int PKCS7_get_CRLs(STACK_OF(X509_CRL) *out_crls, CBS *cbs);

/* PKCS7_bundle_CRLs appends a PKCS#7, SignedData structure containing
 * |crls| to |out|. It returns one on success and zero on error. */
OPENSSL_EXPORT int PKCS7_bundle_CRLs(CBB *out, const STACK_OF(X509_CRL) *crls);

/* PKCS7_get_PEM_certificates reads a PEM-encoded, PKCS#7, SignedData structure
 * from |pem_bio| and appends the included certificates to |out_certs|. It
 * returns one on success and zero on error. */
OPENSSL_EXPORT int PKCS7_get_PEM_certificates(STACK_OF(X509) *out_certs,
                                              BIO *pem_bio);

/* PKCS7_get_PEM_CRLs reads a PEM-encoded, PKCS#7, SignedData structure from
 * |pem_bio| and appends the included CRLs to |out_crls|. It returns one on
 * success and zero on error. */
OPENSSL_EXPORT int PKCS7_get_PEM_CRLs(STACK_OF(X509_CRL) *out_crls,
                                      BIO *pem_bio);

/* EVP_PK values indicate the algorithm of the public key in a certificate. */

#define EVP_PK_RSA	0x0001
#define EVP_PK_DSA	0x0002
#define EVP_PK_DH	0x0004
#define EVP_PK_EC	0x0008

/* EVP_PKS values indicate the algorithm used to sign a certificate. */

#define EVP_PKS_RSA 0x0100
#define EVP_PKS_DSA 0x0200
#define EVP_PKS_EC 0x0400

/* EVP_PKT values are flags that define what public-key operations can be
 * performed with the public key from a certificate. */

/* EVP_PKT_SIGN indicates that the public key can be used for signing. */
#define EVP_PKT_SIGN 0x0010
/* EVP_PKT_ENC indicates that a session key can be encrypted to the public
 * key. */
#define EVP_PKT_ENC 0x0020
/* EVP_PKT_EXCH indicates that key-agreement can be performed. */
#define EVP_PKT_EXCH 0x0040
/* EVP_PKT_EXP indicates that key is weak (i.e. "export"). */
#define EVP_PKT_EXP 0x1000


#ifdef  __cplusplus
}
#endif

#define X509_F_ASN1_digest 100
#define X509_F_ASN1_item_sign_ctx 101
#define X509_F_ASN1_item_verify 102
#define X509_F_NETSCAPE_SPKI_b64_decode 103
#define X509_F_NETSCAPE_SPKI_b64_encode 104
#define X509_F_PKCS7_get_certificates 105
#define X509_F_X509_ATTRIBUTE_create_by_NID 106
#define X509_F_X509_ATTRIBUTE_create_by_OBJ 107
#define X509_F_X509_ATTRIBUTE_create_by_txt 108
#define X509_F_X509_ATTRIBUTE_get0_data 109
#define X509_F_X509_ATTRIBUTE_set1_data 110
#define X509_F_X509_CRL_add0_revoked 111
#define X509_F_X509_CRL_diff 112
#define X509_F_X509_CRL_print_fp 113
#define X509_F_X509_EXTENSION_create_by_NID 114
#define X509_F_X509_EXTENSION_create_by_OBJ 115
#define X509_F_X509_INFO_new 116
#define X509_F_X509_NAME_ENTRY_create_by_NID 117
#define X509_F_X509_NAME_ENTRY_create_by_txt 118
#define X509_F_X509_NAME_ENTRY_set_object 119
#define X509_F_X509_NAME_add_entry 120
#define X509_F_X509_NAME_oneline 121
#define X509_F_X509_NAME_print 122
#define X509_F_X509_PKEY_new 123
#define X509_F_X509_PUBKEY_get 124
#define X509_F_X509_PUBKEY_set 125
#define X509_F_X509_REQ_check_private_key 126
#define X509_F_X509_REQ_to_X509 127
#define X509_F_X509_STORE_CTX_get1_issuer 128
#define X509_F_X509_STORE_CTX_init 129
#define X509_F_X509_STORE_CTX_new 130
#define X509_F_X509_STORE_CTX_purpose_inherit 131
#define X509_F_X509_STORE_add_cert 132
#define X509_F_X509_STORE_add_crl 133
#define X509_F_X509_TRUST_add 134
#define X509_F_X509_TRUST_set 135
#define X509_F_X509_check_private_key 136
#define X509_F_X509_get_pubkey_parameters 137
#define X509_F_X509_load_cert_crl_file 138
#define X509_F_X509_load_cert_file 139
#define X509_F_X509_load_crl_file 140
#define X509_F_X509_print_ex_fp 141
#define X509_F_X509_to_X509_REQ 142
#define X509_F_X509_verify_cert 143
#define X509_F_X509at_add1_attr 144
#define X509_F_X509v3_add_ext 145
#define X509_F_add_cert_dir 146
#define X509_F_by_file_ctrl 147
#define X509_F_check_policy 148
#define X509_F_dir_ctrl 149
#define X509_F_get_cert_by_subject 150
#define X509_F_i2d_DSA_PUBKEY 151
#define X509_F_i2d_EC_PUBKEY 152
#define X509_F_i2d_RSA_PUBKEY 153
#define X509_F_x509_name_encode 154
#define X509_F_x509_name_ex_d2i 155
#define X509_F_x509_name_ex_new 156
#define X509_F_pkcs7_parse_header 157
#define X509_F_PKCS7_get_CRLs 158
#define X509_R_AKID_MISMATCH 100
#define X509_R_BAD_PKCS7_VERSION 101
#define X509_R_BAD_X509_FILETYPE 102
#define X509_R_BASE64_DECODE_ERROR 103
#define X509_R_CANT_CHECK_DH_KEY 104
#define X509_R_CERT_ALREADY_IN_HASH_TABLE 105
#define X509_R_CRL_ALREADY_DELTA 106
#define X509_R_CRL_VERIFY_FAILURE 107
#define X509_R_IDP_MISMATCH 108
#define X509_R_INVALID_BIT_STRING_BITS_LEFT 109
#define X509_R_INVALID_DIRECTORY 110
#define X509_R_INVALID_FIELD_NAME 111
#define X509_R_INVALID_TRUST 112
#define X509_R_ISSUER_MISMATCH 113
#define X509_R_KEY_TYPE_MISMATCH 114
#define X509_R_KEY_VALUES_MISMATCH 115
#define X509_R_LOADING_CERT_DIR 116
#define X509_R_LOADING_DEFAULTS 117
#define X509_R_METHOD_NOT_SUPPORTED 118
#define X509_R_NEWER_CRL_NOT_NEWER 119
#define X509_R_NOT_PKCS7_SIGNED_DATA 120
#define X509_R_NO_CERTIFICATES_INCLUDED 121
#define X509_R_NO_CERT_SET_FOR_US_TO_VERIFY 122
#define X509_R_NO_CRL_NUMBER 123
#define X509_R_PUBLIC_KEY_DECODE_ERROR 124
#define X509_R_PUBLIC_KEY_ENCODE_ERROR 125
#define X509_R_SHOULD_RETRY 126
#define X509_R_UNABLE_TO_FIND_PARAMETERS_IN_CHAIN 127
#define X509_R_UNABLE_TO_GET_CERTS_PUBLIC_KEY 128
#define X509_R_UNKNOWN_KEY_TYPE 129
#define X509_R_UNKNOWN_NID 130
#define X509_R_UNKNOWN_PURPOSE_ID 131
#define X509_R_UNKNOWN_TRUST_ID 132
#define X509_R_UNSUPPORTED_ALGORITHM 133
#define X509_R_WRONG_LOOKUP_TYPE 134
#define X509_R_WRONG_TYPE 135
#define X509_R_NO_CRLS_INCLUDED 136

#endif
