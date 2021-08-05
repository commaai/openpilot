/* ssl/ssl.h */
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
 * Copyright (c) 1998-2007 The OpenSSL Project.  All rights reserved.
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
 * ECC cipher suite support in OpenSSL originally developed by
 * SUN MICROSYSTEMS, INC., and contributed to the OpenSSL project.
 */
/* ====================================================================
 * Copyright 2005 Nokia. All rights reserved.
 *
 * The portions of the attached software ("Contribution") is developed by
 * Nokia Corporation and is licensed pursuant to the OpenSSL open source
 * license.
 *
 * The Contribution, originally written by Mika Kousa and Pasi Eronen of
 * Nokia Corporation, consists of the "PSK" (Pre-Shared Key) ciphersuites
 * support (see RFC 4279) to OpenSSL.
 *
 * No patent licenses or other rights except those expressly stated in
 * the OpenSSL open source license shall be deemed granted or received
 * expressly, by implication, estoppel, or otherwise.
 *
 * No assurances are provided by Nokia that the Contribution does not
 * infringe the patent or other intellectual property rights of any third
 * party or that the license provides you with all the necessary rights
 * to make use of the Contribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. IN
 * ADDITION TO THE DISCLAIMERS INCLUDED IN THE LICENSE, NOKIA
 * SPECIFICALLY DISCLAIMS ANY LIABILITY FOR CLAIMS BROUGHT BY YOU OR ANY
 * OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS OR
 * OTHERWISE.
 */

#ifndef OPENSSL_HEADER_SSL_H
#define OPENSSL_HEADER_SSL_H

#include <openssl/base.h>

#include <openssl/bio.h>
#include <openssl/buf.h>
#include <openssl/hmac.h>
#include <openssl/lhash.h>
#include <openssl/pem.h>
#include <openssl/thread.h>
#include <openssl/x509.h>

#if !defined(OPENSSL_WINDOWS)
#include <sys/time.h>
#endif

/* wpa_supplicant expects to get the version functions from ssl.h */
#include <openssl/crypto.h>

/* Forward-declare struct timeval. On Windows, it is defined in winsock2.h and
 * Windows headers define too many macros to be included in public headers.
 * However, only a forward declaration is needed. */
struct timeval;

#if defined(__cplusplus)
extern "C" {
#endif


/* SSL implementation. */


/* Initialization. */

/* SSL_library_init initializes the crypto and SSL libraries and returns one. */
OPENSSL_EXPORT int SSL_library_init(void);


/* Cipher suites. */

/* An SSL_CIPHER represents a cipher suite. */
typedef struct ssl_cipher_st {
  /* name is the OpenSSL name for the cipher. */
  const char *name;
  /* id is the cipher suite value bitwise OR-d with 0x03000000. */
  uint32_t id;

  /* The following are internal fields. See ssl/internal.h for their values. */

  uint32_t algorithm_mkey;
  uint32_t algorithm_auth;
  uint32_t algorithm_enc;
  uint32_t algorithm_mac;
  uint32_t algorithm_ssl;
  uint32_t algo_strength;

  /* algorithm2 contains extra flags. See ssl/internal.h. */
  uint32_t algorithm2;

  /* strength_bits is the strength of the cipher in bits. */
  int strength_bits;
  /* alg_bits is the number of bits of key material used by the algorithm. */
  int alg_bits;
} SSL_CIPHER;

DECLARE_STACK_OF(SSL_CIPHER)

/* SSL_get_cipher_by_value returns the structure representing a TLS cipher
 * suite based on its assigned number, or NULL if unknown. See
 * https://www.iana.org/assignments/tls-parameters/tls-parameters.xhtml#tls-parameters-4. */
OPENSSL_EXPORT const SSL_CIPHER *SSL_get_cipher_by_value(uint16_t value);

/* SSL_CIPHER_get_id returns |cipher|'s id. It may be cast to a |uint16_t| to
 * get the cipher suite value. */
OPENSSL_EXPORT uint32_t SSL_CIPHER_get_id(const SSL_CIPHER *cipher);

/* SSL_CIPHER_is_AES returns one if |cipher| uses AES (either GCM or CBC
 * mode). */
OPENSSL_EXPORT int SSL_CIPHER_is_AES(const SSL_CIPHER *cipher);

/* SSL_CIPHER_has_MD5_HMAC returns one if |cipher| uses HMAC-MD5. */
OPENSSL_EXPORT int SSL_CIPHER_has_MD5_HMAC(const SSL_CIPHER *cipher);

/* SSL_CIPHER_is_AESGCM returns one if |cipher| uses AES-GCM. */
OPENSSL_EXPORT int SSL_CIPHER_is_AESGCM(const SSL_CIPHER *cipher);

/* SSL_CIPHER_is_CHACHA20POLY1305 returns one if |cipher| uses
 * CHACHA20_POLY1305. */
OPENSSL_EXPORT int SSL_CIPHER_is_CHACHA20POLY1305(const SSL_CIPHER *cipher);

/* SSL_CIPHER_get_name returns the OpenSSL name of |cipher|. */
OPENSSL_EXPORT const char *SSL_CIPHER_get_name(const SSL_CIPHER *cipher);

/* SSL_CIPHER_get_kx_name returns a string that describes the key-exchange
 * method used by |cipher|. For example, "ECDHE_ECDSA". */
OPENSSL_EXPORT const char *SSL_CIPHER_get_kx_name(const SSL_CIPHER *cipher);

/* SSL_CIPHER_get_rfc_name returns a newly-allocated string with the standard
 * name for |cipher| or NULL on error. For example,
 * "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256". The caller is responsible for
 * calling |OPENSSL_free| on the result. */
OPENSSL_EXPORT char *SSL_CIPHER_get_rfc_name(const SSL_CIPHER *cipher);

/* SSL_CIPHER_get_bits returns the strength, in bits, of |cipher|. If
 * |out_alg_bits| is not NULL, it writes the number of bits consumed by the
 * symmetric algorithm to |*out_alg_bits|. */
OPENSSL_EXPORT int SSL_CIPHER_get_bits(const SSL_CIPHER *cipher,
                                       int *out_alg_bits);


/* SSL contexts. */

/* An SSL_METHOD selects whether to use TLS or DTLS. */
typedef struct ssl_method_st SSL_METHOD;

/* TLS_method is the |SSL_METHOD| used for TLS (and SSLv3) connections. */
OPENSSL_EXPORT const SSL_METHOD *TLS_method(void);

/* DTLS_method is the |SSL_METHOD| used for DTLS connections. */
OPENSSL_EXPORT const SSL_METHOD *DTLS_method(void);

/* SSL_CTX_new returns a newly-allocated |SSL_CTX| with default settings or NULL
 * on error. An |SSL_CTX| manages shared state and configuration between
 * multiple TLS or DTLS connections. */
OPENSSL_EXPORT SSL_CTX *SSL_CTX_new(const SSL_METHOD *method);

/* SSL_CTX_free releases memory associated with |ctx|. */
OPENSSL_EXPORT void SSL_CTX_free(SSL_CTX *ctx);


/* SSL connections. */

/* SSL_new returns a newly-allocated |SSL| using |ctx| or NULL on error. An
 * |SSL| object represents a single TLS or DTLS connection. It inherits settings
 * from |ctx| at the time of creation. Settings may also be individually
 * configured on the connection.
 *
 * On creation, an |SSL| is not configured to be either a client or server. Call
 * |SSL_set_connect_state| or |SSL_set_accept_state| to set this. */
OPENSSL_EXPORT SSL *SSL_new(SSL_CTX *ctx);

/* SSL_free releases memory associated with |ssl|. */
OPENSSL_EXPORT void SSL_free(SSL *ssl);

/* SSL_set_connect_state configures |ssl| to be a client. */
OPENSSL_EXPORT void SSL_set_connect_state(SSL *ssl);

/* SSL_set_accept_state configures |ssl| to be a server. */
OPENSSL_EXPORT void SSL_set_accept_state(SSL *ssl);


/* Protocol versions. */

#define SSL3_VERSION_MAJOR 0x03

#define SSL3_VERSION 0x0300
#define TLS1_VERSION 0x0301
#define TLS1_1_VERSION 0x0302
#define TLS1_2_VERSION 0x0303

#define DTLS1_VERSION 0xfeff
#define DTLS1_2_VERSION 0xfefd

/* SSL_CTX_set_min_version sets the minimum protocol version for |ctx| to
 * |version|. */
OPENSSL_EXPORT void SSL_CTX_set_min_version(SSL_CTX *ctx, uint16_t version);

/* SSL_CTX_set_max_version sets the maximum protocol version for |ctx| to
 * |version|. */
OPENSSL_EXPORT void SSL_CTX_set_max_version(SSL_CTX *ctx, uint16_t version);

/* SSL_set_min_version sets the minimum protocol version for |ssl| to
 * |version|. */
OPENSSL_EXPORT void SSL_set_min_version(SSL *ssl, uint16_t version);

/* SSL_set_max_version sets the maximum protocol version for |ssl| to
 * |version|. */
OPENSSL_EXPORT void SSL_set_max_version(SSL *ssl, uint16_t version);


/* Options.
 *
 * Options configure protocol behavior. */

/* SSL_OP_LEGACY_SERVER_CONNECT allows initial connections to servers that don't
 * support the renegotiation_info extension (RFC 5746). It is on by default. */
#define SSL_OP_LEGACY_SERVER_CONNECT 0x00000004L

/* SSL_OP_MICROSOFT_BIG_SSLV3_BUFFER allows for record sizes |SSL3_RT_MAX_EXTRA|
 * bytes above the maximum record size. */
#define SSL_OP_MICROSOFT_BIG_SSLV3_BUFFER 0x00000020L

/* SSL_OP_TLS_D5_BUG accepts an RSAClientKeyExchange in TLS encoded as in SSL3
 * (i.e. without a length prefix). */
#define SSL_OP_TLS_D5_BUG 0x00000100L

/* SSL_OP_ALL enables the above bug workarounds that are enabled by many
 * consumers.
 * TODO(davidben): Determine which of the remaining may be removed now. */
#define SSL_OP_ALL 0x00000BFFL

/* SSL_OP_NO_QUERY_MTU, in DTLS, disables querying the MTU from the underlying
 * |BIO|. Instead, the MTU is configured with |SSL_set_mtu|. */
#define SSL_OP_NO_QUERY_MTU 0x00001000L

/* SSL_OP_NO_TICKET disables session ticket support (RFC 4507). */
#define SSL_OP_NO_TICKET 0x00004000L

/* SSL_OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION permits unsafe legacy renegotiation
 * without renegotiation_info (RFC 5746) support. */
#define SSL_OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION 0x00040000L

/* SSL_OP_CIPHER_SERVER_PREFERENCE configures servers to select ciphers and
 * ECDHE curves according to the server's preferences instead of the
 * client's. */
#define SSL_OP_CIPHER_SERVER_PREFERENCE 0x00400000L

/* The following flags toggle individual protocol versions. This is deprecated.
 * Use |SSL_CTX_set_min_version| and |SSL_CTX_set_max_version| instead. */
#define SSL_OP_NO_SSLv3 0x02000000L
#define SSL_OP_NO_TLSv1 0x04000000L
#define SSL_OP_NO_TLSv1_2 0x08000000L
#define SSL_OP_NO_TLSv1_1 0x10000000L
#define SSL_OP_NO_DTLSv1 SSL_OP_NO_TLSv1
#define SSL_OP_NO_DTLSv1_2 SSL_OP_NO_TLSv1_2

/* The following flags do nothing and are included only to make it easier to
 * compile code with BoringSSL. */
#define SSL_OP_DONT_INSERT_EMPTY_FRAGMENTS 0
#define SSL_OP_MICROSOFT_SESS_ID_BUG 0
#define SSL_OP_NETSCAPE_CHALLENGE_BUG 0
#define SSL_OP_NO_COMPRESSION 0
#define SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION 0
#define SSL_OP_NO_SSLv2 0
#define SSL_OP_SINGLE_DH_USE 0
#define SSL_OP_SINGLE_ECDH_USE 0
#define SSL_OP_SSLREF2_REUSE_CERT_TYPE_BUG 0
#define SSL_OP_TLS_BLOCK_PADDING_BUG 0
#define SSL_OP_TLS_ROLLBACK_BUG 0

/* SSL_CTX_set_options enables all options set in |options| (which should be one
 * or more of the |SSL_OP_*| values, ORed together) in |ctx|. It returns a
 * bitmask representing the resulting enabled options. */
OPENSSL_EXPORT uint32_t SSL_CTX_set_options(SSL_CTX *ctx, uint32_t options);

/* SSL_CTX_clear_options disables all options set in |options| (which should be
 * one or more of the |SSL_OP_*| values, ORed together) in |ctx|. It returns a
 * bitmask representing the resulting enabled options. */
OPENSSL_EXPORT uint32_t SSL_CTX_clear_options(SSL_CTX *ctx, uint32_t options);

/* SSL_CTX_get_options returns a bitmask of |SSL_OP_*| values that represent all
 * the options enabled for |ctx|. */
OPENSSL_EXPORT uint32_t SSL_CTX_get_options(const SSL_CTX *ctx);

/* SSL_set_options enables all options set in |options| (which should be one or
 * more of the |SSL_OP_*| values, ORed together) in |ssl|. It returns a bitmask
 * representing the resulting enabled options. */
OPENSSL_EXPORT uint32_t SSL_set_options(SSL *ssl, uint32_t options);

/* SSL_clear_options disables all options set in |options| (which should be one
 * or more of the |SSL_OP_*| values, ORed together) in |ssl|. It returns a
 * bitmask representing the resulting enabled options. */
OPENSSL_EXPORT uint32_t SSL_clear_options(SSL *ssl, uint32_t options);

/* SSL_get_options returns a bitmask of |SSL_OP_*| values that represent all the
 * options enabled for |ssl|. */
OPENSSL_EXPORT uint32_t SSL_get_options(const SSL *ssl);


/* Modes.
 *
 * Modes configure API behavior. */

/* SSL_MODE_ENABLE_PARTIAL_WRITE allows |SSL_write| to complete with a partial
 * result when the only part of the input was written in a single record. */
#define SSL_MODE_ENABLE_PARTIAL_WRITE 0x00000001L

/* SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER allows retrying an incomplete |SSL_write|
 * with a different buffer. However, |SSL_write| still assumes the buffer
 * contents are unchanged. This is not the default to avoid the misconception
 * that non-blocking |SSL_write| behaves like non-blocking |write|. */
#define SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER 0x00000002L

/* SSL_MODE_NO_AUTO_CHAIN disables automatically building a certificate chain
 * before sending certificates to the peer.
 * TODO(davidben): Remove this behavior. https://crbug.com/486295. */
#define SSL_MODE_NO_AUTO_CHAIN 0x00000008L

/* SSL_MODE_ENABLE_FALSE_START allows clients to send application data before
 * receipt of CCS and Finished. This mode enables full-handshakes to 'complete'
 * in one RTT. See draft-bmoeller-tls-falsestart-01. */
#define SSL_MODE_ENABLE_FALSE_START 0x00000080L

/* Deprecated: SSL_MODE_HANDSHAKE_CUTTHROUGH is the same as
 * SSL_MODE_ENABLE_FALSE_START. */
#define SSL_MODE_HANDSHAKE_CUTTHROUGH SSL_MODE_ENABLE_FALSE_START

/* SSL_MODE_CBC_RECORD_SPLITTING causes multi-byte CBC records in SSL 3.0 and
 * TLS 1.0 to be split in two: the first record will contain a single byte and
 * the second will contain the remainder. This effectively randomises the IV and
 * prevents BEAST attacks. */
#define SSL_MODE_CBC_RECORD_SPLITTING 0x00000100L

/* SSL_MODE_NO_SESSION_CREATION will cause any attempts to create a session to
 * fail with SSL_R_SESSION_MAY_NOT_BE_CREATED. This can be used to enforce that
 * session resumption is used for a given SSL*. */
#define SSL_MODE_NO_SESSION_CREATION 0x00000200L

/* SSL_MODE_SEND_FALLBACK_SCSV sends TLS_FALLBACK_SCSV in the ClientHello.
 * To be set only by applications that reconnect with a downgraded protocol
 * version; see https://tools.ietf.org/html/draft-ietf-tls-downgrade-scsv-05
 * for details.
 *
 * DO NOT ENABLE THIS if your application attempts a normal handshake. Only use
 * this in explicit fallback retries, following the guidance in
 * draft-ietf-tls-downgrade-scsv-05. */
#define SSL_MODE_SEND_FALLBACK_SCSV 0x00000400L

/* The following flags do nothing and are included only to make it easier to
 * compile code with BoringSSL. */
#define SSL_MODE_AUTO_RETRY 0
#define SSL_MODE_RELEASE_BUFFERS 0
#define SSL_MODE_SEND_CLIENTHELLO_TIME 0
#define SSL_MODE_SEND_SERVERHELLO_TIME 0

/* SSL_CTX_set_mode enables all modes set in |mode| (which should be one or more
 * of the |SSL_MODE_*| values, ORed together) in |ctx|. It returns a bitmask
 * representing the resulting enabled modes. */
OPENSSL_EXPORT uint32_t SSL_CTX_set_mode(SSL_CTX *ctx, uint32_t mode);

/* SSL_CTX_clear_mode disables all modes set in |mode| (which should be one or
 * more of the |SSL_MODE_*| values, ORed together) in |ctx|. It returns a
 * bitmask representing the resulting enabled modes. */
OPENSSL_EXPORT uint32_t SSL_CTX_clear_mode(SSL_CTX *ctx, uint32_t mode);

/* SSL_CTX_get_mode returns a bitmask of |SSL_MODE_*| values that represent all
 * the modes enabled for |ssl|. */
OPENSSL_EXPORT uint32_t SSL_CTX_get_mode(const SSL_CTX *ctx);

/* SSL_set_mode enables all modes set in |mode| (which should be one or more of
 * the |SSL_MODE_*| values, ORed together) in |ssl|. It returns a bitmask
 * representing the resulting enabled modes. */
OPENSSL_EXPORT uint32_t SSL_set_mode(SSL *ssl, uint32_t mode);

/* SSL_clear_mode disables all modes set in |mode| (which should be one or more
 * of the |SSL_MODE_*| values, ORed together) in |ssl|. It returns a bitmask
 * representing the resulting enabled modes. */
OPENSSL_EXPORT uint32_t SSL_clear_mode(SSL *ssl, uint32_t mode);

/* SSL_get_mode returns a bitmask of |SSL_MODE_*| values that represent all the
 * modes enabled for |ssl|. */
OPENSSL_EXPORT uint32_t SSL_get_mode(const SSL *ssl);


/* Connection information. */

/* SSL_get_tls_unique writes at most |max_out| bytes of the tls-unique value
 * for |ssl| to |out| and sets |*out_len| to the number of bytes written. It
 * returns one on success or zero on error. In general |max_out| should be at
 * least 12.
 *
 * This function will always fail if the initial handshake has not completed.
 * The tls-unique value will change after a renegotiation but, since
 * renegotiations can be initiated by the server at any point, the higher-level
 * protocol must either leave them disabled or define states in which the
 * tls-unique value can be read.
 *
 * The tls-unique value is defined by
 * https://tools.ietf.org/html/rfc5929#section-3.1. Due to a weakness in the
 * TLS protocol, tls-unique is broken for resumed connections unless the
 * Extended Master Secret extension is negotiated. Thus this function will
 * return zero if |ssl| performed session resumption unless EMS was used when
 * negotiating the original session. */
OPENSSL_EXPORT int SSL_get_tls_unique(const SSL *ssl, uint8_t *out,
                                      size_t *out_len, size_t max_out);


/* Underdocumented functions.
 *
 * Functions below here haven't been touched up and may be underdocumented. */

/* SSLeay version number for ASN.1 encoding of the session information */
/* Version 0 - initial version
 * Version 1 - added the optional peer certificate. */
#define SSL_SESSION_ASN1_VERSION 0x0001

#define SSL_MAX_SSL_SESSION_ID_LENGTH 32
#define SSL_MAX_SID_CTX_LENGTH 32
#define SSL_MAX_MASTER_KEY_LENGTH 48

/* These are used to specify which ciphers to use and not to use */

#define SSL_TXT_MEDIUM "MEDIUM"
#define SSL_TXT_HIGH "HIGH"
#define SSL_TXT_FIPS "FIPS"

#define SSL_TXT_kRSA "kRSA"
#define SSL_TXT_kDHE "kDHE"
#define SSL_TXT_kEDH "kEDH" /* same as "kDHE" */
#define SSL_TXT_kECDHE "kECDHE"
#define SSL_TXT_kEECDH "kEECDH" /* same as "kECDHE" */
#define SSL_TXT_kPSK "kPSK"

#define SSL_TXT_aRSA "aRSA"
#define SSL_TXT_aECDSA "aECDSA"
#define SSL_TXT_aPSK "aPSK"

#define SSL_TXT_DH "DH"
#define SSL_TXT_DHE "DHE" /* same as "kDHE" */
#define SSL_TXT_EDH "EDH" /* same as "DHE" */
#define SSL_TXT_RSA "RSA"
#define SSL_TXT_ECDH "ECDH"
#define SSL_TXT_ECDHE "ECDHE" /* same as "kECDHE" */
#define SSL_TXT_EECDH "EECDH" /* same as "ECDHE" */
#define SSL_TXT_ECDSA "ECDSA"
#define SSL_TXT_PSK "PSK"

#define SSL_TXT_3DES "3DES"
#define SSL_TXT_RC4 "RC4"
#define SSL_TXT_AES128 "AES128"
#define SSL_TXT_AES256 "AES256"
#define SSL_TXT_AES "AES"
#define SSL_TXT_AES_GCM "AESGCM"
#define SSL_TXT_CHACHA20 "CHACHA20"

#define SSL_TXT_MD5 "MD5"
#define SSL_TXT_SHA1 "SHA1"
#define SSL_TXT_SHA "SHA" /* same as "SHA1" */
#define SSL_TXT_SHA256 "SHA256"
#define SSL_TXT_SHA384 "SHA384"

#define SSL_TXT_SSLV3 "SSLv3"
#define SSL_TXT_TLSV1 "TLSv1"
#define SSL_TXT_TLSV1_1 "TLSv1.1"
#define SSL_TXT_TLSV1_2 "TLSv1.2"

#define SSL_TXT_ALL "ALL"

/* COMPLEMENTOF* definitions. These identifiers are used to (de-select) ciphers
 * normally not being used.
 *
 * Example: "RC4" will activate all ciphers using RC4 including ciphers without
 * authentication, which would normally disabled by DEFAULT (due the "!ADH"
 * being part of default). Therefore "RC4:!COMPLEMENTOFDEFAULT" will make sure
 * that it is also disabled in the specific selection. COMPLEMENTOF*
 * identifiers are portable between version, as adjustments to the default
 * cipher setup will also be included here.
 *
 * COMPLEMENTOFDEFAULT does not experience the same special treatment that
 * DEFAULT gets, as only selection is being done and no sorting as needed for
 * DEFAULT. */
#define SSL_TXT_CMPDEF "COMPLEMENTOFDEFAULT"

/* The following cipher list is used by default. It also is substituted when an
 * application-defined cipher list string starts with 'DEFAULT'. */
#define SSL_DEFAULT_CIPHER_LIST "ALL"

/* As of OpenSSL 1.0.0, ssl_create_cipher_list() in ssl/ssl_ciph.c always
 * starts with a reasonable order, and all we have to do for DEFAULT is
 * throwing out anonymous and unencrypted ciphersuites! (The latter are not
 * actually enabled by ALL, but "ALL:RSA" would enable some of them.) */

/* Used in SSL_set_shutdown()/SSL_get_shutdown(); */
#define SSL_SENT_SHUTDOWN 1
#define SSL_RECEIVED_SHUTDOWN 2

#define SSL_FILETYPE_ASN1 X509_FILETYPE_ASN1
#define SSL_FILETYPE_PEM X509_FILETYPE_PEM

typedef struct ssl_protocol_method_st SSL_PROTOCOL_METHOD;
typedef struct ssl_session_st SSL_SESSION;
typedef struct tls_sigalgs_st TLS_SIGALGS;
typedef struct ssl_conf_ctx_st SSL_CONF_CTX;
typedef struct ssl3_enc_method SSL3_ENC_METHOD;

/* SRTP protection profiles for use with the use_srtp extension (RFC 5764). */
typedef struct srtp_protection_profile_st {
  const char *name;
  unsigned long id;
} SRTP_PROTECTION_PROFILE;

DECLARE_STACK_OF(SRTP_PROTECTION_PROFILE)

/* An SSL_SESSION represents an SSL session that may be resumed in an
 * abbreviated handshake. */
struct ssl_session_st {
  int ssl_version; /* what ssl version session info is being kept in here? */

  int master_key_length;
  uint8_t master_key[SSL_MAX_MASTER_KEY_LENGTH];
  /* session_id - valid? */
  unsigned int session_id_length;
  uint8_t session_id[SSL_MAX_SSL_SESSION_ID_LENGTH];
  /* this is used to determine whether the session is being reused in
   * the appropriate context. It is up to the application to set this,
   * via SSL_new */
  unsigned int sid_ctx_length;
  uint8_t sid_ctx[SSL_MAX_SID_CTX_LENGTH];

  char *psk_identity;
  /* Used to indicate that session resumption is not allowed. Applications can
   * also set this bit for a new session via not_resumable_session_cb to
   * disable session caching and tickets. */
  int not_resumable;

  /* The cert is the certificate used to establish this connection */
  struct sess_cert_st /* SESS_CERT */ *sess_cert;

  /* This is the cert for the other end. On clients, it will be the same as
   * sess_cert->peer_key->x509 (the latter is not enough as sess_cert is not
   * retained in the external representation of sessions, see ssl_asn1.c). */
  X509 *peer;
  /* when app_verify_callback accepts a session where the peer's certificate is
   * not ok, we must remember the error for session reuse: */
  long verify_result; /* only for servers */

  CRYPTO_refcount_t references;
  long timeout;
  long time;

  const SSL_CIPHER *cipher;

  CRYPTO_EX_DATA ex_data; /* application specific data */

  /* These are used to make removal of session-ids more efficient and to
   * implement a maximum cache size. */
  SSL_SESSION *prev, *next;
  char *tlsext_hostname;
  /* RFC4507 info */
  uint8_t *tlsext_tick;               /* Session ticket */
  size_t tlsext_ticklen;              /* Session ticket length */
  uint32_t tlsext_tick_lifetime_hint; /* Session lifetime hint in seconds */

  size_t tlsext_signed_cert_timestamp_list_length;
  uint8_t *tlsext_signed_cert_timestamp_list; /* Server's list. */

  /* The OCSP response that came with the session. */
  size_t ocsp_response_length;
  uint8_t *ocsp_response;

  char peer_sha256_valid; /* Non-zero if peer_sha256 is valid */
  uint8_t
      peer_sha256[SHA256_DIGEST_LENGTH]; /* SHA256 of peer certificate */

  /* original_handshake_hash contains the handshake hash (either SHA-1+MD5 or
   * SHA-2, depending on TLS version) for the original, full handshake that
   * created a session. This is used by Channel IDs during resumption. */
  uint8_t original_handshake_hash[EVP_MAX_MD_SIZE];
  unsigned int original_handshake_hash_len;

  /* extended_master_secret is true if the master secret in this session was
   * generated using EMS and thus isn't vulnerable to the Triple Handshake
   * attack. */
  char extended_master_secret;
};


/* Cert related flags */
/* Many implementations ignore some aspects of the TLS standards such as
 * enforcing certifcate chain algorithms. When this is set we enforce them. */
#define SSL_CERT_FLAG_TLS_STRICT 0x00000001L

/* Flags for building certificate chains */
/* Treat any existing certificates as untrusted CAs */
#define SSL_BUILD_CHAIN_FLAG_UNTRUSTED 0x1
/* Don't include root CA in chain */
#define SSL_BUILD_CHAIN_FLAG_NO_ROOT 0x2
/* Just check certificates already there */
#define SSL_BUILD_CHAIN_FLAG_CHECK 0x4
/* Ignore verification errors */
#define SSL_BUILD_CHAIN_FLAG_IGNORE_ERROR 0x8
/* Clear verification errors from queue */
#define SSL_BUILD_CHAIN_FLAG_CLEAR_ERROR 0x10

/* SSL_set_mtu sets the |ssl|'s MTU in DTLS to |mtu|. It returns one on success
 * and zero on failure. */
OPENSSL_EXPORT int SSL_set_mtu(SSL *ssl, unsigned mtu);

/* SSL_get_secure_renegotiation_support returns one if the peer supports secure
 * renegotiation (RFC 5746) and zero otherwise. */
OPENSSL_EXPORT int SSL_get_secure_renegotiation_support(const SSL *ssl);

/* SSL_CTX_set_msg_callback installs |cb| as the message callback for |ctx|.
 * This callback will be called when sending or receiving low-level record
 * headers, complete handshake messages, ChangeCipherSpec, and alerts.
 * |write_p| is one for outgoing messages and zero for incoming messages.
 *
 * For each record header, |cb| is called with |version| = 0 and |content_type|
 * = |SSL3_RT_HEADER|. The |len| bytes from |buf| contain the header. Note that
 * this does not include the record body. If the record is sealed, the length
 * in the header is the length of the ciphertext.
 *
 * For each handshake message, ChangeCipherSpec, and alert, |version| is the
 * protocol version and |content_type| is the corresponding record type. The
 * |len| bytes from |buf| contain the handshake message, one-byte
 * ChangeCipherSpec body, and two-byte alert, respectively. */
OPENSSL_EXPORT void SSL_CTX_set_msg_callback(
    SSL_CTX *ctx, void (*cb)(int write_p, int version, int content_type,
                             const void *buf, size_t len, SSL *ssl, void *arg));

/* SSL_CTX_set_msg_callback_arg sets the |arg| parameter of the message
 * callback. */
OPENSSL_EXPORT void SSL_CTX_set_msg_callback_arg(SSL_CTX *ctx, void *arg);

/* SSL_set_msg_callback installs |cb| as the message callback of |ssl|. See
 * |SSL_CTX_set_msg_callback| for when this callback is called. */
OPENSSL_EXPORT void SSL_set_msg_callback(
    SSL *ssl, void (*cb)(int write_p, int version, int content_type,
                         const void *buf, size_t len, SSL *ssl, void *arg));

/* SSL_set_msg_callback_arg sets the |arg| parameter of the message callback. */
OPENSSL_EXPORT void SSL_set_msg_callback_arg(SSL *ssl, void *arg);

/* SSL_CTX_set_keylog_bio sets configures all SSL objects attached to |ctx| to
 * log session material to |keylog_bio|. This is intended for debugging use
 * with tools like Wireshark. |ctx| takes ownership of |keylog_bio|.
 *
 * The format is described in
 * https://developer.mozilla.org/en-US/docs/Mozilla/Projects/NSS/Key_Log_Format. */
OPENSSL_EXPORT void SSL_CTX_set_keylog_bio(SSL_CTX *ctx, BIO *keylog_bio);


struct ssl_aead_ctx_st;
typedef struct ssl_aead_ctx_st SSL_AEAD_CTX;

#define SSL_MAX_CERT_LIST_DEFAULT 1024 * 100 /* 100k max cert list */

#define SSL_SESSION_CACHE_MAX_SIZE_DEFAULT (1024 * 20)

#define SSL_DEFAULT_SESSION_TIMEOUT (2 * 60 * 60)

/* This callback type is used inside SSL_CTX, SSL, and in the functions that
 * set them. It is used to override the generation of SSL/TLS session IDs in a
 * server. Return value should be zero on an error, non-zero to proceed. Also,
 * callbacks should themselves check if the id they generate is unique
 * otherwise the SSL handshake will fail with an error - callbacks can do this
 * using the 'ssl' value they're passed by;
 *      SSL_has_matching_session_id(ssl, id, *id_len)
 * The length value passed in is set at the maximum size the session ID can be.
 * In SSLv2 this is 16 bytes, whereas SSLv3/TLSv1 it is 32 bytes. The callback
 * can alter this length to be less if desired, but under SSLv2 session IDs are
 * supposed to be fixed at 16 bytes so the id will be padded after the callback
 * returns in this case. It is also an error for the callback to set the size
 * to zero. */
typedef int (*GEN_SESSION_CB)(const SSL *ssl, uint8_t *id,
                              unsigned int *id_len);

/* ssl_early_callback_ctx is passed to certain callbacks that are called very
 * early on during the server handshake. At this point, much of the SSL* hasn't
 * been filled out and only the ClientHello can be depended on. */
struct ssl_early_callback_ctx {
  SSL *ssl;
  const uint8_t *client_hello;
  size_t client_hello_len;
  const uint8_t *session_id;
  size_t session_id_len;
  const uint8_t *cipher_suites;
  size_t cipher_suites_len;
  const uint8_t *compression_methods;
  size_t compression_methods_len;
  const uint8_t *extensions;
  size_t extensions_len;
};

/* SSL_early_callback_ctx_extension_get searches the extensions in |ctx| for an
 * extension of the given type. If not found, it returns zero. Otherwise it
 * sets |out_data| to point to the extension contents (not including the type
 * and length bytes), sets |out_len| to the length of the extension contents
 * and returns one. */
OPENSSL_EXPORT char SSL_early_callback_ctx_extension_get(
    const struct ssl_early_callback_ctx *ctx, uint16_t extension_type,
    const uint8_t **out_data, size_t *out_len);

typedef struct ssl_comp_st SSL_COMP;

struct ssl_comp_st {
  int id;
  const char *name;
  char *method;
};

DECLARE_STACK_OF(SSL_COMP)
DECLARE_LHASH_OF(SSL_SESSION)

/* ssl_cipher_preference_list_st contains a list of SSL_CIPHERs with
 * equal-preference groups. For TLS clients, the groups are moot because the
 * server picks the cipher and groups cannot be expressed on the wire. However,
 * for servers, the equal-preference groups allow the client's preferences to
 * be partially respected. (This only has an effect with
 * SSL_OP_CIPHER_SERVER_PREFERENCE).
 *
 * The equal-preference groups are expressed by grouping SSL_CIPHERs together.
 * All elements of a group have the same priority: no ordering is expressed
 * within a group.
 *
 * The values in |ciphers| are in one-to-one correspondence with
 * |in_group_flags|. (That is, sk_SSL_CIPHER_num(ciphers) is the number of
 * bytes in |in_group_flags|.) The bytes in |in_group_flags| are either 1, to
 * indicate that the corresponding SSL_CIPHER is not the last element of a
 * group, or 0 to indicate that it is.
 *
 * For example, if |in_group_flags| contains all zeros then that indicates a
 * traditional, fully-ordered preference. Every SSL_CIPHER is the last element
 * of the group (i.e. they are all in a one-element group).
 *
 * For a more complex example, consider:
 *   ciphers:        A  B  C  D  E  F
 *   in_group_flags: 1  1  0  0  1  0
 *
 * That would express the following, order:
 *
 *    A         E
 *    B -> D -> F
 *    C
 */
struct ssl_cipher_preference_list_st {
  STACK_OF(SSL_CIPHER) *ciphers;
  uint8_t *in_group_flags;
};

struct ssl_ctx_st {
  const SSL_PROTOCOL_METHOD *method;

  /* lock is used to protect various operations on this object. */
  CRYPTO_MUTEX lock;

  /* max_version is the maximum acceptable protocol version. If zero, the
   * maximum supported version, currently (D)TLS 1.2, is used. */
  uint16_t max_version;

  /* min_version is the minimum acceptable protocl version. If zero, the
   * minimum supported version, currently SSL 3.0 and DTLS 1.0, is used */
  uint16_t min_version;

  struct ssl_cipher_preference_list_st *cipher_list;
  /* same as above but sorted for lookup */
  STACK_OF(SSL_CIPHER) *cipher_list_by_id;
  /* cipher_list_tls11 is the list of ciphers when TLS 1.1 or greater is in
   * use. This only applies to server connections as, for clients, the version
   * number is known at connect time and so the cipher list can be set then. */
  struct ssl_cipher_preference_list_st *cipher_list_tls11;

  X509_STORE *cert_store;
  LHASH_OF(SSL_SESSION) *sessions;
  /* Most session-ids that will be cached, default is
   * SSL_SESSION_CACHE_MAX_SIZE_DEFAULT. 0 is unlimited. */
  unsigned long session_cache_size;
  SSL_SESSION *session_cache_head;
  SSL_SESSION *session_cache_tail;

  /* handshakes_since_cache_flush is the number of successful handshakes since
   * the last cache flush. */
  int handshakes_since_cache_flush;

  /* This can have one of 2 values, ored together,
   * SSL_SESS_CACHE_CLIENT,
   * SSL_SESS_CACHE_SERVER,
   * Default is SSL_SESSION_CACHE_SERVER, which means only
   * SSL_accept which cache SSL_SESSIONS. */
  int session_cache_mode;

  /* If timeout is not 0, it is the default timeout value set when SSL_new() is
   * called.  This has been put in to make life easier to set things up */
  long session_timeout;

  /* If this callback is not null, it will be called each time a session id is
   * added to the cache.  If this function returns 1, it means that the
   * callback will do a SSL_SESSION_free() when it has finished using it.
   * Otherwise, on 0, it means the callback has finished with it. If
   * remove_session_cb is not null, it will be called when a session-id is
   * removed from the cache.  After the call, OpenSSL will SSL_SESSION_free()
   * it. */
  int (*new_session_cb)(SSL *ssl, SSL_SESSION *sess);
  void (*remove_session_cb)(SSL_CTX *ctx, SSL_SESSION *sess);
  SSL_SESSION *(*get_session_cb)(SSL *ssl, uint8_t *data, int len,
                                 int *copy);

  CRYPTO_refcount_t references;

  /* if defined, these override the X509_verify_cert() calls */
  int (*app_verify_callback)(X509_STORE_CTX *, void *);
  void *app_verify_arg;
  /* before OpenSSL 0.9.7, 'app_verify_arg' was ignored ('app_verify_callback'
   * was called with just one argument) */

  /* Default password callback. */
  pem_password_cb *default_passwd_callback;

  /* Default password callback user data. */
  void *default_passwd_callback_userdata;

  /* get client cert callback */
  int (*client_cert_cb)(SSL *ssl, X509 **x509, EVP_PKEY **pkey);

  /* get channel id callback */
  void (*channel_id_cb)(SSL *ssl, EVP_PKEY **pkey);

  CRYPTO_EX_DATA ex_data;

  STACK_OF(X509) *extra_certs;


  /* Default values used when no per-SSL value is defined follow */

  void (*info_callback)(const SSL *ssl, int type,
                        int val); /* used if SSL's info_callback is NULL */

  /* what we put in client cert requests */
  STACK_OF(X509_NAME) *client_CA;


  /* Default values to use in SSL structures follow (these are copied by
   * SSL_new) */

  uint32_t options;
  uint32_t mode;
  uint32_t max_cert_list;

  struct cert_st /* CERT */ *cert;

  /* callback that allows applications to peek at protocol messages */
  void (*msg_callback)(int write_p, int version, int content_type,
                       const void *buf, size_t len, SSL *ssl, void *arg);
  void *msg_callback_arg;

  int verify_mode;
  unsigned int sid_ctx_length;
  uint8_t sid_ctx[SSL_MAX_SID_CTX_LENGTH];
  int (*default_verify_callback)(
      int ok, X509_STORE_CTX *ctx); /* called 'verify_callback' in the SSL */

  /* Default generate session ID callback. */
  GEN_SESSION_CB generate_session_id;

  X509_VERIFY_PARAM *param;

  /* select_certificate_cb is called before most ClientHello processing and
   * before the decision whether to resume a session is made. It may return one
   * to continue the handshake or zero to cause the handshake loop to return
   * with an error and cause SSL_get_error to return
   * SSL_ERROR_PENDING_CERTIFICATE. Note: when the handshake loop is resumed, it
   * will not call the callback a second time. */
  int (*select_certificate_cb)(const struct ssl_early_callback_ctx *);

  /* dos_protection_cb is called once the resumption decision for a ClientHello
   * has been made. It returns one to continue the handshake or zero to
   * abort. */
  int (*dos_protection_cb) (const struct ssl_early_callback_ctx *);

  /* quiet_shutdown is true if the connection should not send a close_notify on
   * shutdown. */
  int quiet_shutdown;

  /* Maximum amount of data to send in one fragment. actual record size can be
   * more than this due to padding and MAC overheads. */
  uint16_t max_send_fragment;

  /* TLS extensions servername callback */
  int (*tlsext_servername_callback)(SSL *, int *, void *);
  void *tlsext_servername_arg;
  /* RFC 4507 session ticket keys */
  uint8_t tlsext_tick_key_name[16];
  uint8_t tlsext_tick_hmac_key[16];
  uint8_t tlsext_tick_aes_key[16];
  /* Callback to support customisation of ticket key setting */
  int (*tlsext_ticket_key_cb)(SSL *ssl, uint8_t *name, uint8_t *iv,
                              EVP_CIPHER_CTX *ectx, HMAC_CTX *hctx, int enc);

  /* Server-only: psk_identity_hint is the default identity hint to send in
   * PSK-based key exchanges. */
  char *psk_identity_hint;

  unsigned int (*psk_client_callback)(SSL *ssl, const char *hint,
                                      char *identity,
                                      unsigned int max_identity_len,
                                      uint8_t *psk, unsigned int max_psk_len);
  unsigned int (*psk_server_callback)(SSL *ssl, const char *identity,
                                      uint8_t *psk, unsigned int max_psk_len);


  /* retain_only_sha256_of_client_certs is true if we should compute the SHA256
   * hash of the peer's certifiate and then discard it to save memory and
   * session space. Only effective on the server side. */
  char retain_only_sha256_of_client_certs;

  /* Next protocol negotiation information */
  /* (for experimental NPN extension). */

  /* For a server, this contains a callback function by which the set of
   * advertised protocols can be provided. */
  int (*next_protos_advertised_cb)(SSL *s, const uint8_t **buf,
                                   unsigned int *len, void *arg);
  void *next_protos_advertised_cb_arg;
  /* For a client, this contains a callback function that selects the
   * next protocol from the list provided by the server. */
  int (*next_proto_select_cb)(SSL *s, uint8_t **out, uint8_t *outlen,
                              const uint8_t *in, unsigned int inlen, void *arg);
  void *next_proto_select_cb_arg;

  /* ALPN information
   * (we are in the process of transitioning from NPN to ALPN.) */

  /* For a server, this contains a callback function that allows the
   * server to select the protocol for the connection.
   *   out: on successful return, this must point to the raw protocol
   *        name (without the length prefix).
   *   outlen: on successful return, this contains the length of |*out|.
   *   in: points to the client's list of supported protocols in
   *       wire-format.
   *   inlen: the length of |in|. */
  int (*alpn_select_cb)(SSL *s, const uint8_t **out, uint8_t *outlen,
                        const uint8_t *in, unsigned int inlen, void *arg);
  void *alpn_select_cb_arg;

  /* For a client, this contains the list of supported protocols in wire
   * format. */
  uint8_t *alpn_client_proto_list;
  unsigned alpn_client_proto_list_len;

  /* SRTP profiles we are willing to do from RFC 5764 */
  STACK_OF(SRTP_PROTECTION_PROFILE) *srtp_profiles;

  /* EC extension values inherited by SSL structure */
  size_t tlsext_ecpointformatlist_length;
  uint8_t *tlsext_ecpointformatlist;
  size_t tlsext_ellipticcurvelist_length;
  uint16_t *tlsext_ellipticcurvelist;

  /* If true, a client will advertise the Channel ID extension and a server
   * will echo it. */
  char tlsext_channel_id_enabled;
  /* tlsext_channel_id_enabled_new is a hack to support both old and new
   * ChannelID signatures. It indicates that a client should advertise the new
   * ChannelID extension number. */
  char tlsext_channel_id_enabled_new;
  /* The client's Channel ID private key. */
  EVP_PKEY *tlsext_channel_id_private;

  /* If true, a client will request certificate timestamps. */
  char signed_cert_timestamps_enabled;

  /* If true, a client will request a stapled OCSP response. */
  char ocsp_stapling_enabled;

  /* If not NULL, session key material will be logged to this BIO for debugging
   * purposes. The format matches NSS's and is readable by Wireshark. */
  BIO *keylog_bio;

  /* current_time_cb, if not NULL, is the function to use to get the current
   * time. It sets |*out_clock| to the current time. */
  void (*current_time_cb)(const SSL *ssl, struct timeval *out_clock);
};

OPENSSL_EXPORT LHASH_OF(SSL_SESSION) *SSL_CTX_sessions(SSL_CTX *ctx);

/* SSL_CTX_sess_number returns the number of sessions in |ctx|'s internal
 * session cache. */
OPENSSL_EXPORT size_t SSL_CTX_sess_number(const SSL_CTX *ctx);

OPENSSL_EXPORT void SSL_CTX_sess_set_new_cb(
    SSL_CTX *ctx, int (*new_session_cb)(SSL *ssl, SSL_SESSION *sess));
OPENSSL_EXPORT int (*SSL_CTX_sess_get_new_cb(SSL_CTX *ctx))(SSL *ssl,
                                                            SSL_SESSION *sess);
OPENSSL_EXPORT void SSL_CTX_sess_set_remove_cb(
    SSL_CTX *ctx,
    void (*remove_session_cb)(SSL_CTX *ctx, SSL_SESSION *sess));
OPENSSL_EXPORT void (*SSL_CTX_sess_get_remove_cb(SSL_CTX *ctx))(
    SSL_CTX *ctx, SSL_SESSION *sess);
OPENSSL_EXPORT void SSL_CTX_sess_set_get_cb(
    SSL_CTX *ctx,
    SSL_SESSION *(*get_session_cb)(SSL *ssl, uint8_t *data, int len,
                                   int *copy));
OPENSSL_EXPORT SSL_SESSION *(*SSL_CTX_sess_get_get_cb(SSL_CTX *ctx))(
    SSL *ssl, uint8_t *data, int len, int *copy);
/* SSL_magic_pending_session_ptr returns a magic SSL_SESSION* which indicates
 * that the session isn't currently unavailable. SSL_get_error will then return
 * SSL_ERROR_PENDING_SESSION and the handshake can be retried later when the
 * lookup has completed. */
OPENSSL_EXPORT SSL_SESSION *SSL_magic_pending_session_ptr(void);
OPENSSL_EXPORT void SSL_CTX_set_info_callback(SSL_CTX *ctx,
                                              void (*cb)(const SSL *ssl,
                                                         int type, int val));
OPENSSL_EXPORT void (*SSL_CTX_get_info_callback(SSL_CTX *ctx))(const SSL *ssl,
                                                               int type,
                                                               int val);
OPENSSL_EXPORT void SSL_CTX_set_client_cert_cb(
    SSL_CTX *ctx,
    int (*client_cert_cb)(SSL *ssl, X509 **x509, EVP_PKEY **pkey));
OPENSSL_EXPORT int (*SSL_CTX_get_client_cert_cb(SSL_CTX *ctx))(SSL *ssl,
                                                               X509 **x509,
                                                               EVP_PKEY **pkey);
OPENSSL_EXPORT void SSL_CTX_set_channel_id_cb(
    SSL_CTX *ctx, void (*channel_id_cb)(SSL *ssl, EVP_PKEY **pkey));
OPENSSL_EXPORT void (*SSL_CTX_get_channel_id_cb(SSL_CTX *ctx))(SSL *ssl,
                                                               EVP_PKEY **pkey);

/* SSL_enable_signed_cert_timestamps causes |ssl| (which must be the client end
 * of a connection) to request SCTs from the server. See
 * https://tools.ietf.org/html/rfc6962. It returns one. */
OPENSSL_EXPORT int SSL_enable_signed_cert_timestamps(SSL *ssl);

/* SSL_CTX_enable_signed_cert_timestamps enables SCT requests on all client SSL
 * objects created from |ctx|. */
OPENSSL_EXPORT void SSL_CTX_enable_signed_cert_timestamps(SSL_CTX *ctx);

/* SSL_enable_ocsp_stapling causes |ssl| (which must be the client end of a
 * connection) to request a stapled OCSP response from the server. It returns
 * one. */
OPENSSL_EXPORT int SSL_enable_ocsp_stapling(SSL *ssl);

/* SSL_CTX_enable_ocsp_stapling enables OCSP stapling on all client SSL objects
 * created from |ctx|. */
OPENSSL_EXPORT void SSL_CTX_enable_ocsp_stapling(SSL_CTX *ctx);

/* SSL_get0_signed_cert_timestamp_list sets |*out| and |*out_len| to point to
 * |*out_len| bytes of SCT information from the server. This is only valid if
 * |ssl| is a client. The SCT information is a SignedCertificateTimestampList
 * (including the two leading length bytes).
 * See https://tools.ietf.org/html/rfc6962#section-3.3
 * If no SCT was received then |*out_len| will be zero on return.
 *
 * WARNING: the returned data is not guaranteed to be well formed. */
OPENSSL_EXPORT void SSL_get0_signed_cert_timestamp_list(const SSL *ssl,
                                                        const uint8_t **out,
                                                        size_t *out_len);

/* SSL_get0_ocsp_response sets |*out| and |*out_len| to point to |*out_len|
 * bytes of an OCSP response from the server. This is the DER encoding of an
 * OCSPResponse type as defined in RFC 2560.
 *
 * WARNING: the returned data is not guaranteed to be well formed. */
OPENSSL_EXPORT void SSL_get0_ocsp_response(const SSL *ssl, const uint8_t **out,
                                           size_t *out_len);

OPENSSL_EXPORT void SSL_CTX_set_next_protos_advertised_cb(
    SSL_CTX *s,
    int (*cb)(SSL *ssl, const uint8_t **out, unsigned int *outlen, void *arg),
    void *arg);
OPENSSL_EXPORT void SSL_CTX_set_next_proto_select_cb(
    SSL_CTX *s, int (*cb)(SSL *ssl, uint8_t **out, uint8_t *outlen,
                          const uint8_t *in, unsigned int inlen, void *arg),
    void *arg);
OPENSSL_EXPORT void SSL_get0_next_proto_negotiated(const SSL *s,
                                                   const uint8_t **data,
                                                   unsigned *len);

OPENSSL_EXPORT int SSL_select_next_proto(uint8_t **out, uint8_t *outlen,
                                         const uint8_t *in, unsigned int inlen,
                                         const uint8_t *client,
                                         unsigned int client_len);

#define OPENSSL_NPN_UNSUPPORTED 0
#define OPENSSL_NPN_NEGOTIATED 1
#define OPENSSL_NPN_NO_OVERLAP 2

/* SSL_CTX_set_alpn_protos sets the ALPN protocol list on |ctx| to |protos|.
 * |protos| must be in wire-format (i.e. a series of non-empty, 8-bit
 * length-prefixed strings). It returns zero on success and one on failure.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. */
OPENSSL_EXPORT int SSL_CTX_set_alpn_protos(SSL_CTX *ctx, const uint8_t *protos,
                                           unsigned protos_len);

/* SSL_set_alpn_protos sets the ALPN protocol list on |ssl| to |protos|.
 * |protos| must be in wire-format (i.e. a series of non-empty, 8-bit
 * length-prefixed strings). It returns zero on success and one on failure.
 *
 * WARNING: this function is dangerous because it breaks the usual return value
 * convention. */
OPENSSL_EXPORT int SSL_set_alpn_protos(SSL *ssl, const uint8_t *protos,
                                       unsigned protos_len);

OPENSSL_EXPORT void SSL_CTX_set_alpn_select_cb(
    SSL_CTX *ctx, int (*cb)(SSL *ssl, const uint8_t **out, uint8_t *outlen,
                            const uint8_t *in, unsigned int inlen, void *arg),
    void *arg);
OPENSSL_EXPORT void SSL_get0_alpn_selected(const SSL *ssl, const uint8_t **data,
                                           unsigned *len);

/* SSL_enable_fastradio_padding controls whether fastradio padding is enabled
 * on |ssl|. If it is, ClientHello messages are padded to 1024 bytes. This
 * causes 3G radios to switch to DCH mode (high data rate). */
OPENSSL_EXPORT void SSL_enable_fastradio_padding(SSL *ssl, char on_off);

/* SSL_set_reject_peer_renegotiations controls whether renegotiation attempts by
 * the peer are rejected. It may be set at any point in a connection's lifetime
 * to control future renegotiations programmatically. By default, renegotiations
 * are rejected. (Renegotiations requested by a client are always rejected.) */
OPENSSL_EXPORT void SSL_set_reject_peer_renegotiations(SSL *ssl, int reject);

/* the maximum length of the buffer given to callbacks containing the resulting
 * identity/psk */
#define PSK_MAX_IDENTITY_LEN 128
#define PSK_MAX_PSK_LEN 256
OPENSSL_EXPORT void SSL_CTX_set_psk_client_callback(
    SSL_CTX *ctx,
    unsigned int (*psk_client_callback)(
        SSL *ssl, const char *hint, char *identity,
        unsigned int max_identity_len, uint8_t *psk, unsigned int max_psk_len));
OPENSSL_EXPORT void SSL_set_psk_client_callback(
    SSL *ssl, unsigned int (*psk_client_callback)(SSL *ssl, const char *hint,
                                                  char *identity,
                                                  unsigned int max_identity_len,
                                                  uint8_t *psk,
                                                  unsigned int max_psk_len));
OPENSSL_EXPORT void SSL_CTX_set_psk_server_callback(
    SSL_CTX *ctx,
    unsigned int (*psk_server_callback)(SSL *ssl, const char *identity,
                                        uint8_t *psk,
                                        unsigned int max_psk_len));
OPENSSL_EXPORT void SSL_set_psk_server_callback(
    SSL *ssl,
    unsigned int (*psk_server_callback)(SSL *ssl, const char *identity,
                                        uint8_t *psk,
                                        unsigned int max_psk_len));
OPENSSL_EXPORT int SSL_CTX_use_psk_identity_hint(SSL_CTX *ctx,
                                                 const char *identity_hint);
OPENSSL_EXPORT int SSL_use_psk_identity_hint(SSL *s, const char *identity_hint);
OPENSSL_EXPORT const char *SSL_get_psk_identity_hint(const SSL *s);
OPENSSL_EXPORT const char *SSL_get_psk_identity(const SSL *s);

#define SSL_NOTHING 1
#define SSL_WRITING 2
#define SSL_READING 3
#define SSL_X509_LOOKUP 4
#define SSL_CHANNEL_ID_LOOKUP 5
#define SSL_PENDING_SESSION 7
#define SSL_CERTIFICATE_SELECTION_PENDING 8

/* These will only be used when doing non-blocking IO */
#define SSL_want_nothing(s) (SSL_want(s) == SSL_NOTHING)
#define SSL_want_read(s) (SSL_want(s) == SSL_READING)
#define SSL_want_write(s) (SSL_want(s) == SSL_WRITING)
#define SSL_want_x509_lookup(s) (SSL_want(s) == SSL_X509_LOOKUP)
#define SSL_want_channel_id_lookup(s) (SSL_want(s) == SSL_CHANNEL_ID_LOOKUP)
#define SSL_want_session(s) (SSL_want(s) == SSL_PENDING_SESSION)
#define SSL_want_certificate(s) \
  (SSL_want(s) == SSL_CERTIFICATE_SELECTION_PENDING)

struct ssl_st {
  /* version is the protocol version. */
  int version;

  /* method is the method table corresponding to the current protocol (DTLS or
   * TLS). */
  const SSL_PROTOCOL_METHOD *method;

  /* enc_method is the method table corresponding to the current protocol
   * version. */
  const SSL3_ENC_METHOD *enc_method;

  /* max_version is the maximum acceptable protocol version. If zero, the
   * maximum supported version, currently (D)TLS 1.2, is used. */
  uint16_t max_version;

  /* min_version is the minimum acceptable protocl version. If zero, the
   * minimum supported version, currently SSL 3.0 and DTLS 1.0, is used */
  uint16_t min_version;

  /* There are 2 BIO's even though they are normally both the same. This is so
   * data can be read and written to different handlers */

  BIO *rbio; /* used by SSL_read */
  BIO *wbio; /* used by SSL_write */
  BIO *bbio; /* used during session-id reuse to concatenate
              * messages */

  /* This holds a variable that indicates what we were doing when a 0 or -1 is
   * returned.  This is needed for non-blocking IO so we know what request
   * needs re-doing when in SSL_accept or SSL_connect */
  int rwstate;

  /* true when we are actually in SSL_accept() or SSL_connect() */
  int in_handshake;
  int (*handshake_func)(SSL *);

  /* Imagine that here's a boolean member "init" that is switched as soon as
   * SSL_set_{accept/connect}_state is called for the first time, so that
   * "state" and "handshake_func" are properly initialized.  But as
   * handshake_func is == 0 until then, we use this test instead of an "init"
   * member. */

  /* server is true iff the this SSL* is the server half. Note: before the SSL*
   * is initialized by either SSL_set_accept_state or SSL_set_connect_state,
   * the side is not determined. In this state, server is always false. */
  int server;

  /* quiet_shutdown is true if the connection should not send a close_notify on
   * shutdown. */
  int quiet_shutdown;

  int shutdown; /* we have shut things down, 0x01 sent, 0x02
                 * for received */
  int state;    /* where we are */
  int rstate;   /* where we are when reading */

  BUF_MEM *init_buf; /* buffer used during init */
  uint8_t *init_msg; /* pointer to handshake message body, set by
                        ssl3_get_message() */
  int init_num;      /* amount read/written */
  int init_off;      /* amount read/written */

  /* used internally to point at a raw packet */
  uint8_t *packet;
  unsigned int packet_length;

  struct ssl3_state_st *s3;  /* SSLv3 variables */
  struct dtls1_state_st *d1; /* DTLSv1 variables */

  /* callback that allows applications to peek at protocol messages */
  void (*msg_callback)(int write_p, int version, int content_type,
                       const void *buf, size_t len, SSL *ssl, void *arg);
  void *msg_callback_arg;

  int hit; /* reusing a previous session */

  X509_VERIFY_PARAM *param;

  /* crypto */
  struct ssl_cipher_preference_list_st *cipher_list;
  STACK_OF(SSL_CIPHER) *cipher_list_by_id;

  SSL_AEAD_CTX *aead_read_ctx;
  SSL_AEAD_CTX *aead_write_ctx;

  /* session info */

  /* client cert? */
  /* This is used to hold the server certificate used */
  struct cert_st /* CERT */ *cert;

  /* the session_id_context is used to ensure sessions are only reused
   * in the appropriate context */
  unsigned int sid_ctx_length;
  uint8_t sid_ctx[SSL_MAX_SID_CTX_LENGTH];

  /* This can also be in the session once a session is established */
  SSL_SESSION *session;

  /* Default generate session ID callback. */
  GEN_SESSION_CB generate_session_id;

  /* Used in SSL2 and SSL3 */
  int verify_mode; /* 0 don't care about verify failure.
                    * 1 fail if verify fails */
  int (*verify_callback)(int ok,
                         X509_STORE_CTX *ctx); /* fail if callback returns 0 */

  void (*info_callback)(const SSL *ssl, int type,
                        int val); /* optional informational callback */

  /* Server-only: psk_identity_hint is the identity hint to send in
   * PSK-based key exchanges. */
  char *psk_identity_hint;

  unsigned int (*psk_client_callback)(SSL *ssl, const char *hint,
                                      char *identity,
                                      unsigned int max_identity_len,
                                      uint8_t *psk, unsigned int max_psk_len);
  unsigned int (*psk_server_callback)(SSL *ssl, const char *identity,
                                      uint8_t *psk, unsigned int max_psk_len);

  SSL_CTX *ctx;

  /* extra application data */
  long verify_result;
  CRYPTO_EX_DATA ex_data;

  /* for server side, keep the list of CA_dn we can use */
  STACK_OF(X509_NAME) *client_CA;

  uint32_t options; /* protocol behaviour */
  uint32_t mode;    /* API behaviour */
  uint32_t max_cert_list;
  int client_version; /* what was passed, used for
                       * SSLv3/TLS rollback check */
  uint16_t max_send_fragment;
  char *tlsext_hostname;
  /* should_ack_sni is true if the SNI extension should be acked. This is
   * only used by a server. */
  char should_ack_sni;
  /* RFC4507 session ticket expected to be received or sent */
  int tlsext_ticket_expected;
  size_t tlsext_ecpointformatlist_length;
  uint8_t *tlsext_ecpointformatlist; /* our list */
  size_t tlsext_ellipticcurvelist_length;
  uint16_t *tlsext_ellipticcurvelist; /* our list */

  SSL_CTX *initial_ctx; /* initial ctx, used to store sessions */

  /* Next protocol negotiation. For the client, this is the protocol that we
   * sent in NextProtocol and is set when handling ServerHello extensions.
   *
   * For a server, this is the client's selected_protocol from NextProtocol and
   * is set when handling the NextProtocol message, before the Finished
   * message. */
  uint8_t *next_proto_negotiated;
  size_t next_proto_negotiated_len;

  /* srtp_profiles is the list of configured SRTP protection profiles for
   * DTLS-SRTP. */
  STACK_OF(SRTP_PROTECTION_PROFILE) *srtp_profiles;

  /* srtp_profile is the selected SRTP protection profile for
   * DTLS-SRTP. */
  const SRTP_PROTECTION_PROFILE *srtp_profile;

  /* Copied from the SSL_CTX. For a server, means that we'll accept Channel IDs
   * from clients. For a client, means that we'll advertise support. */
  char tlsext_channel_id_enabled;
  /* The client's Channel ID private key. */
  EVP_PKEY *tlsext_channel_id_private;

  /* Enable signed certificate time stamps. Currently client only. */
  char signed_cert_timestamps_enabled;

  /* Enable OCSP stapling. Currently client only.
   * TODO(davidben): Add a server-side implementation when it becomes
   * necesary. */
  char ocsp_stapling_enabled;

  /* For a client, this contains the list of supported protocols in wire
   * format. */
  uint8_t *alpn_client_proto_list;
  unsigned alpn_client_proto_list_len;

  /* fastradio_padding, if true, causes ClientHellos to be padded to 1024
   * bytes. This ensures that the cellular radio is fast forwarded to DCH (high
   * data rate) state in 3G networks. */
  char fastradio_padding;

  /* accept_peer_renegotiations, if one, accepts renegotiation attempts from the
   * peer. Otherwise, they will be rejected with a fatal error. */
  char accept_peer_renegotiations;

  /* These fields are always NULL and exist only to keep wpa_supplicant happy
   * about the change to EVP_AEAD. They are only needed for EAP-FAST, which we
   * don't support. */
  EVP_CIPHER_CTX *enc_read_ctx;
  EVP_MD_CTX *read_hash;
};

/* compatibility */
#define SSL_set_app_data(s, arg) (SSL_set_ex_data(s, 0, (char *)arg))
#define SSL_get_app_data(s) (SSL_get_ex_data(s, 0))
#define SSL_SESSION_set_app_data(s, a) \
  (SSL_SESSION_set_ex_data(s, 0, (char *)a))
#define SSL_SESSION_get_app_data(s) (SSL_SESSION_get_ex_data(s, 0))
#define SSL_CTX_get_app_data(ctx) (SSL_CTX_get_ex_data(ctx, 0))
#define SSL_CTX_set_app_data(ctx, arg) \
  (SSL_CTX_set_ex_data(ctx, 0, (char *)arg))

/* The following are the possible values for ssl->state are are used to
 * indicate where we are up to in the SSL connection establishment. The macros
 * that follow are about the only things you should need to use and even then,
 * only when using non-blocking IO. It can also be useful to work out where you
 * were when the connection failed */

#define SSL_ST_CONNECT 0x1000
#define SSL_ST_ACCEPT 0x2000
#define SSL_ST_MASK 0x0FFF
#define SSL_ST_INIT (SSL_ST_CONNECT | SSL_ST_ACCEPT)
#define SSL_ST_OK 0x03
#define SSL_ST_RENEGOTIATE (0x04 | SSL_ST_INIT)

#define SSL_CB_LOOP 0x01
#define SSL_CB_EXIT 0x02
#define SSL_CB_READ 0x04
#define SSL_CB_WRITE 0x08
#define SSL_CB_ALERT 0x4000 /* used in callback */
#define SSL_CB_READ_ALERT (SSL_CB_ALERT | SSL_CB_READ)
#define SSL_CB_WRITE_ALERT (SSL_CB_ALERT | SSL_CB_WRITE)
#define SSL_CB_ACCEPT_LOOP (SSL_ST_ACCEPT | SSL_CB_LOOP)
#define SSL_CB_ACCEPT_EXIT (SSL_ST_ACCEPT | SSL_CB_EXIT)
#define SSL_CB_CONNECT_LOOP (SSL_ST_CONNECT | SSL_CB_LOOP)
#define SSL_CB_CONNECT_EXIT (SSL_ST_CONNECT | SSL_CB_EXIT)
#define SSL_CB_HANDSHAKE_START 0x10
#define SSL_CB_HANDSHAKE_DONE 0x20

/* Is the SSL_connection established? */
#define SSL_get_state(a) SSL_state(a)
#define SSL_is_init_finished(a) (SSL_state(a) == SSL_ST_OK)
#define SSL_in_init(a) (SSL_state(a) & SSL_ST_INIT)
#define SSL_in_connect_init(a) (SSL_state(a) & SSL_ST_CONNECT)
#define SSL_in_accept_init(a) (SSL_state(a) & SSL_ST_ACCEPT)

/* SSL_in_false_start returns one if |s| has a pending unfinished handshake that
 * is in False Start. |SSL_write| may be called at this point without waiting
 * for the peer, but |SSL_read| will require the handshake to be completed. */
OPENSSL_EXPORT int SSL_in_false_start(const SSL *s);

/* The following 2 states are kept in ssl->rstate when reads fail,
 * you should not need these */
#define SSL_ST_READ_HEADER 0xF0
#define SSL_ST_READ_BODY 0xF1
#define SSL_ST_READ_DONE 0xF2

/* Obtain latest Finished message
 *   -- that we sent (SSL_get_finished)
 *   -- that we expected from peer (SSL_get_peer_finished).
 * Returns length (0 == no Finished so far), copies up to 'count' bytes. */
OPENSSL_EXPORT size_t SSL_get_finished(const SSL *s, void *buf, size_t count);
OPENSSL_EXPORT size_t SSL_get_peer_finished(const SSL *s, void *buf, size_t count);

/* use either SSL_VERIFY_NONE or SSL_VERIFY_PEER, the last 3 options
 * are 'ored' with SSL_VERIFY_PEER if they are desired */
#define SSL_VERIFY_NONE 0x00
#define SSL_VERIFY_PEER 0x01
#define SSL_VERIFY_FAIL_IF_NO_PEER_CERT 0x02
/* SSL_VERIFY_CLIENT_ONCE does nothing. */
#define SSL_VERIFY_CLIENT_ONCE 0x04
#define SSL_VERIFY_PEER_IF_NO_OBC 0x08

#define OpenSSL_add_ssl_algorithms() SSL_library_init()
#define SSLeay_add_ssl_algorithms() SSL_library_init()

/* For backward compatibility */
#define SSL_get_cipher(s) SSL_CIPHER_get_name(SSL_get_current_cipher(s))
#define SSL_get_cipher_bits(s, np) \
  SSL_CIPHER_get_bits(SSL_get_current_cipher(s), np)
#define SSL_get_cipher_version(s) \
  SSL_CIPHER_get_version(SSL_get_current_cipher(s))
#define SSL_get_cipher_name(s) SSL_CIPHER_get_name(SSL_get_current_cipher(s))
#define SSL_get_time(a) SSL_SESSION_get_time(a)
#define SSL_set_time(a, b) SSL_SESSION_set_time((a), (b))
#define SSL_get_timeout(a) SSL_SESSION_get_timeout(a)
#define SSL_set_timeout(a, b) SSL_SESSION_set_timeout((a), (b))

#define d2i_SSL_SESSION_bio(bp, s_id) \
  ASN1_d2i_bio_of(SSL_SESSION, SSL_SESSION_new, d2i_SSL_SESSION, bp, s_id)
#define i2d_SSL_SESSION_bio(bp, s_id) \
  ASN1_i2d_bio_of(SSL_SESSION, i2d_SSL_SESSION, bp, s_id)

DECLARE_PEM_rw(SSL_SESSION, SSL_SESSION)

/* make_errors.go reserves error codes above 1000 for manually-assigned errors.
 * This value must be kept in sync with reservedReasonCode in make_errors.h */
#define SSL_AD_REASON_OFFSET \
  1000 /* offset to get SSL_R_... value from SSL_AD_... */

/* These alert types are for SSLv3 and TLSv1 */
#define SSL_AD_CLOSE_NOTIFY SSL3_AD_CLOSE_NOTIFY
#define SSL_AD_UNEXPECTED_MESSAGE SSL3_AD_UNEXPECTED_MESSAGE /* fatal */
#define SSL_AD_BAD_RECORD_MAC SSL3_AD_BAD_RECORD_MAC         /* fatal */
#define SSL_AD_DECRYPTION_FAILED TLS1_AD_DECRYPTION_FAILED
#define SSL_AD_RECORD_OVERFLOW TLS1_AD_RECORD_OVERFLOW
#define SSL_AD_DECOMPRESSION_FAILURE SSL3_AD_DECOMPRESSION_FAILURE /* fatal */
#define SSL_AD_HANDSHAKE_FAILURE SSL3_AD_HANDSHAKE_FAILURE         /* fatal */
#define SSL_AD_NO_CERTIFICATE SSL3_AD_NO_CERTIFICATE /* Not for TLS */
#define SSL_AD_BAD_CERTIFICATE SSL3_AD_BAD_CERTIFICATE
#define SSL_AD_UNSUPPORTED_CERTIFICATE SSL3_AD_UNSUPPORTED_CERTIFICATE
#define SSL_AD_CERTIFICATE_REVOKED SSL3_AD_CERTIFICATE_REVOKED
#define SSL_AD_CERTIFICATE_EXPIRED SSL3_AD_CERTIFICATE_EXPIRED
#define SSL_AD_CERTIFICATE_UNKNOWN SSL3_AD_CERTIFICATE_UNKNOWN
#define SSL_AD_ILLEGAL_PARAMETER SSL3_AD_ILLEGAL_PARAMETER /* fatal */
#define SSL_AD_UNKNOWN_CA TLS1_AD_UNKNOWN_CA               /* fatal */
#define SSL_AD_ACCESS_DENIED TLS1_AD_ACCESS_DENIED         /* fatal */
#define SSL_AD_DECODE_ERROR TLS1_AD_DECODE_ERROR           /* fatal */
#define SSL_AD_DECRYPT_ERROR TLS1_AD_DECRYPT_ERROR
#define SSL_AD_EXPORT_RESTRICTION TLS1_AD_EXPORT_RESTRICTION       /* fatal */
#define SSL_AD_PROTOCOL_VERSION TLS1_AD_PROTOCOL_VERSION           /* fatal */
#define SSL_AD_INSUFFICIENT_SECURITY TLS1_AD_INSUFFICIENT_SECURITY /* fatal */
#define SSL_AD_INTERNAL_ERROR TLS1_AD_INTERNAL_ERROR               /* fatal */
#define SSL_AD_USER_CANCELLED TLS1_AD_USER_CANCELLED
#define SSL_AD_NO_RENEGOTIATION TLS1_AD_NO_RENEGOTIATION
#define SSL_AD_UNSUPPORTED_EXTENSION TLS1_AD_UNSUPPORTED_EXTENSION
#define SSL_AD_CERTIFICATE_UNOBTAINABLE TLS1_AD_CERTIFICATE_UNOBTAINABLE
#define SSL_AD_UNRECOGNIZED_NAME TLS1_AD_UNRECOGNIZED_NAME
#define SSL_AD_BAD_CERTIFICATE_STATUS_RESPONSE \
  TLS1_AD_BAD_CERTIFICATE_STATUS_RESPONSE
#define SSL_AD_BAD_CERTIFICATE_HASH_VALUE TLS1_AD_BAD_CERTIFICATE_HASH_VALUE
#define SSL_AD_UNKNOWN_PSK_IDENTITY TLS1_AD_UNKNOWN_PSK_IDENTITY     /* fatal */
#define SSL_AD_INAPPROPRIATE_FALLBACK SSL3_AD_INAPPROPRIATE_FALLBACK /* fatal */

#define SSL_ERROR_NONE 0
#define SSL_ERROR_SSL 1
#define SSL_ERROR_WANT_READ 2
#define SSL_ERROR_WANT_WRITE 3
#define SSL_ERROR_WANT_X509_LOOKUP 4
#define SSL_ERROR_SYSCALL 5 /* look at error stack/return value/errno */
#define SSL_ERROR_ZERO_RETURN 6
#define SSL_ERROR_WANT_CONNECT 7
#define SSL_ERROR_WANT_ACCEPT 8
#define SSL_ERROR_WANT_CHANNEL_ID_LOOKUP 9
#define SSL_ERROR_PENDING_SESSION 11
#define SSL_ERROR_PENDING_CERTIFICATE 12

#define SSL_CTRL_EXTRA_CHAIN_CERT 14

/* see tls1.h for macros based on these */
#define SSL_CTRL_GET_TLSEXT_TICKET_KEYS 58
#define SSL_CTRL_SET_TLSEXT_TICKET_KEYS 59

#define SSL_CTRL_SET_SRP_ARG 78
#define SSL_CTRL_SET_TLS_EXT_SRP_USERNAME 79
#define SSL_CTRL_SET_TLS_EXT_SRP_STRENGTH 80
#define SSL_CTRL_SET_TLS_EXT_SRP_PASSWORD 81

#define SSL_CTRL_GET_EXTRA_CHAIN_CERTS 82
#define SSL_CTRL_CLEAR_EXTRA_CHAIN_CERTS 83

#define SSL_CTRL_CHAIN 88
#define SSL_CTRL_CHAIN_CERT 89

#define SSL_CTRL_GET_CURVES 90
#define SSL_CTRL_SET_CURVES 91
#define SSL_CTRL_SET_CURVES_LIST 92
#define SSL_CTRL_SET_SIGALGS 97
#define SSL_CTRL_SET_SIGALGS_LIST 98
#define SSL_CTRL_SET_CLIENT_SIGALGS 101
#define SSL_CTRL_SET_CLIENT_SIGALGS_LIST 102
#define SSL_CTRL_GET_CLIENT_CERT_TYPES 103
#define SSL_CTRL_SET_CLIENT_CERT_TYPES 104
#define SSL_CTRL_BUILD_CERT_CHAIN 105
#define SSL_CTRL_SET_VERIFY_CERT_STORE 106
#define SSL_CTRL_SET_CHAIN_CERT_STORE 107
#define SSL_CTRL_GET_SERVER_TMP_KEY 109
#define SSL_CTRL_GET_EC_POINT_FORMATS 111

#define SSL_CTRL_GET_CHAIN_CERTS 115
#define SSL_CTRL_SELECT_CURRENT_CERT 116

/* DTLSv1_get_timeout queries the next DTLS handshake timeout. If there is a
 * timeout in progress, it sets |*out| to the time remaining and returns one.
 * Otherwise, it returns zero.
 *
 * When the timeout expires, call |DTLSv1_handle_timeout| to handle the
 * retransmit behavior.
 *
 * NOTE: This function must be queried again whenever the handshake state
 * machine changes, including when |DTLSv1_handle_timeout| is called. */
OPENSSL_EXPORT int DTLSv1_get_timeout(const SSL *ssl, struct timeval *out);

/* DTLSv1_handle_timeout is called when a DTLS handshake timeout expires. If no
 * timeout had expired, it returns 0. Otherwise, it retransmits the previous
 * flight of handshake messages and returns 1. If too many timeouts had expired
 * without progress or an error occurs, it returns -1.
 *
 * NOTE: The caller's external timer should be compatible with the one |ssl|
 * queries within some fudge factor. Otherwise, the call will be a no-op, but
 * |DTLSv1_get_timeout| will return an updated timeout.
 *
 * WARNING: This function breaks the usual return value convention. */
OPENSSL_EXPORT int DTLSv1_handle_timeout(SSL *ssl);

/* SSL_session_reused returns one if |ssl| performed an abbreviated handshake
 * and zero otherwise.
 *
 * TODO(davidben): Hammer down the semantics of this API while a handshake,
 * initial or renego, is in progress. */
OPENSSL_EXPORT int SSL_session_reused(const SSL *ssl);

/* SSL_total_renegotiations returns the total number of renegotiation handshakes
 * peformed by |ssl|. This includes the pending renegotiation, if any. */
OPENSSL_EXPORT int SSL_total_renegotiations(const SSL *ssl);

/* SSL_CTX_set_tmp_dh configures |ctx| to use the group from |dh| as the group
 * for DHE. Only the group is used, so |dh| needn't have a keypair. It returns
 * one on success and zero on error. */
OPENSSL_EXPORT int SSL_CTX_set_tmp_dh(SSL_CTX *ctx, const DH *dh);

/* SSL_set_tmp_dh configures |ssl| to use the group from |dh| as the group for
 * DHE. Only the group is used, so |dh| needn't have a keypair. It returns one
 * on success and zero on error. */
OPENSSL_EXPORT int SSL_set_tmp_dh(SSL *ssl, const DH *dh);

/* SSL_CTX_set_tmp_ecdh configures |ctx| to use the curve from |ecdh| as the
 * curve for ephemeral ECDH keys. For historical reasons, this API expects an
 * |EC_KEY|, but only the curve is used. It returns one on success and zero on
 * error. If unset, an appropriate curve will be chosen automatically. (This is
 * recommended.) */
OPENSSL_EXPORT int SSL_CTX_set_tmp_ecdh(SSL_CTX *ctx, const EC_KEY *ec_key);

/* SSL_set_tmp_ecdh configures |ssl| to use the curve from |ecdh| as the curve
 * for ephemeral ECDH keys. For historical reasons, this API expects an
 * |EC_KEY|, but only the curve is used. It returns one on success and zero on
 * error. If unset, an appropriate curve will be chosen automatically. (This is
 * recommended.) */
OPENSSL_EXPORT int SSL_set_tmp_ecdh(SSL *ssl, const EC_KEY *ec_key);

/* SSL_CTX_enable_tls_channel_id either configures a TLS server to accept TLS
 * client IDs from clients, or configures a client to send TLS client IDs to
 * a server. It returns one. */
OPENSSL_EXPORT int SSL_CTX_enable_tls_channel_id(SSL_CTX *ctx);

/* SSL_enable_tls_channel_id either configures a TLS server to accept TLS
 * client IDs from clients, or configure a client to send TLS client IDs to
 * server. It returns one. */
OPENSSL_EXPORT int SSL_enable_tls_channel_id(SSL *ssl);

/* SSL_CTX_set1_tls_channel_id configures a TLS client to send a TLS Channel ID
 * to compatible servers. |private_key| must be a P-256 EC key. It returns one
 * on success and zero on error. */
OPENSSL_EXPORT int SSL_CTX_set1_tls_channel_id(SSL_CTX *ctx,
                                               EVP_PKEY *private_key);

/* SSL_set1_tls_channel_id configures a TLS client to send a TLS Channel ID to
 * compatible servers. |private_key| must be a P-256 EC key. It returns one on
 * success and zero on error. */
OPENSSL_EXPORT int SSL_set1_tls_channel_id(SSL *ssl, EVP_PKEY *private_key);

/* SSL_get_tls_channel_id gets the client's TLS Channel ID from a server SSL*
 * and copies up to the first |max_out| bytes into |out|. The Channel ID
 * consists of the client's P-256 public key as an (x,y) pair where each is a
 * 32-byte, big-endian field element. It returns 0 if the client didn't offer a
 * Channel ID and the length of the complete Channel ID otherwise. */
OPENSSL_EXPORT size_t SSL_get_tls_channel_id(SSL *ssl, uint8_t *out,
                                             size_t max_out);

#define SSL_CTX_add_extra_chain_cert(ctx, x509) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_EXTRA_CHAIN_CERT, 0, (char *)x509)
#define SSL_CTX_get_extra_chain_certs(ctx, px509) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_GET_EXTRA_CHAIN_CERTS, 0, px509)
#define SSL_CTX_get_extra_chain_certs_only(ctx, px509) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_GET_EXTRA_CHAIN_CERTS, 1, px509)
#define SSL_CTX_clear_extra_chain_certs(ctx) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_CLEAR_EXTRA_CHAIN_CERTS, 0, NULL)

#define SSL_CTX_set0_chain(ctx, sk) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_CHAIN, 0, (char *)sk)
#define SSL_CTX_set1_chain(ctx, sk) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_CHAIN, 1, (char *)sk)
#define SSL_CTX_add0_chain_cert(ctx, x509) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_CHAIN_CERT, 0, (char *)x509)
#define SSL_CTX_add1_chain_cert(ctx, x509) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_CHAIN_CERT, 1, (char *)x509)
#define SSL_CTX_get0_chain_certs(ctx, px509) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_GET_CHAIN_CERTS, 0, px509)
#define SSL_CTX_clear_chain_certs(ctx) SSL_CTX_set0_chain(ctx, NULL)
#define SSL_CTX_build_cert_chain(ctx, flags) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_BUILD_CERT_CHAIN, flags, NULL)
#define SSL_CTX_select_current_cert(ctx, x509) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SELECT_CURRENT_CERT, 0, (char *)x509)

#define SSL_CTX_set0_verify_cert_store(ctx, st) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_VERIFY_CERT_STORE, 0, (char *)st)
#define SSL_CTX_set1_verify_cert_store(ctx, st) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_VERIFY_CERT_STORE, 1, (char *)st)
#define SSL_CTX_set0_chain_cert_store(ctx, st) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_CHAIN_CERT_STORE, 0, (char *)st)
#define SSL_CTX_set1_chain_cert_store(ctx, st) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_CHAIN_CERT_STORE, 1, (char *)st)

#define SSL_set0_chain(ctx, sk) SSL_ctrl(ctx, SSL_CTRL_CHAIN, 0, (char *)sk)
#define SSL_set1_chain(ctx, sk) SSL_ctrl(ctx, SSL_CTRL_CHAIN, 1, (char *)sk)
#define SSL_add0_chain_cert(ctx, x509) \
  SSL_ctrl(ctx, SSL_CTRL_CHAIN_CERT, 0, (char *)x509)
#define SSL_add1_chain_cert(ctx, x509) \
  SSL_ctrl(ctx, SSL_CTRL_CHAIN_CERT, 1, (char *)x509)
#define SSL_get0_chain_certs(ctx, px509) \
  SSL_ctrl(ctx, SSL_CTRL_GET_CHAIN_CERTS, 0, px509)
#define SSL_clear_chain_certs(ctx) SSL_set0_chain(ctx, NULL)
#define SSL_build_cert_chain(s, flags) \
  SSL_ctrl(s, SSL_CTRL_BUILD_CERT_CHAIN, flags, NULL)
#define SSL_select_current_cert(ctx, x509) \
  SSL_ctrl(ctx, SSL_CTRL_SELECT_CURRENT_CERT, 0, (char *)x509)

#define SSL_set0_verify_cert_store(s, st) \
  SSL_ctrl(s, SSL_CTRL_SET_VERIFY_CERT_STORE, 0, (char *)st)
#define SSL_set1_verify_cert_store(s, st) \
  SSL_ctrl(s, SSL_CTRL_SET_VERIFY_CERT_STORE, 1, (char *)st)
#define SSL_set0_chain_cert_store(s, st) \
  SSL_ctrl(s, SSL_CTRL_SET_CHAIN_CERT_STORE, 0, (char *)st)
#define SSL_set1_chain_cert_store(s, st) \
  SSL_ctrl(s, SSL_CTRL_SET_CHAIN_CERT_STORE, 1, (char *)st)

#define SSL_get1_curves(ctx, s) SSL_ctrl(ctx, SSL_CTRL_GET_CURVES, 0, (char *)s)
#define SSL_CTX_set1_curves(ctx, clist, clistlen) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_CURVES, clistlen, (char *)clist)
#define SSL_CTX_set1_curves_list(ctx, s) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_CURVES_LIST, 0, (char *)s)
#define SSL_set1_curves(ctx, clist, clistlen) \
  SSL_ctrl(ctx, SSL_CTRL_SET_CURVES, clistlen, (char *)clist)
#define SSL_set1_curves_list(ctx, s) \
  SSL_ctrl(ctx, SSL_CTRL_SET_CURVES_LIST, 0, (char *)s)

#define SSL_CTX_set1_sigalgs(ctx, slist, slistlen) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_SIGALGS, slistlen, (int *)slist)
#define SSL_CTX_set1_sigalgs_list(ctx, s) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_SIGALGS_LIST, 0, (char *)s)
#define SSL_set1_sigalgs(ctx, slist, slistlen) \
  SSL_ctrl(ctx, SSL_CTRL_SET_SIGALGS, clistlen, (int *)slist)
#define SSL_set1_sigalgs_list(ctx, s) \
  SSL_ctrl(ctx, SSL_CTRL_SET_SIGALGS_LIST, 0, (char *)s)

#define SSL_CTX_set1_client_sigalgs(ctx, slist, slistlen) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_CLIENT_SIGALGS, slistlen, (int *)slist)
#define SSL_CTX_set1_client_sigalgs_list(ctx, s) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_CLIENT_SIGALGS_LIST, 0, (char *)s)
#define SSL_set1_client_sigalgs(ctx, slist, slistlen) \
  SSL_ctrl(ctx, SSL_CTRL_SET_CLIENT_SIGALGS, clistlen, (int *)slist)
#define SSL_set1_client_sigalgs_list(ctx, s) \
  SSL_ctrl(ctx, SSL_CTRL_SET_CLIENT_SIGALGS_LIST, 0, (char *)s)

#define SSL_get0_certificate_types(s, clist) \
  SSL_ctrl(s, SSL_CTRL_GET_CLIENT_CERT_TYPES, 0, (char *)clist)

#define SSL_CTX_set1_client_certificate_types(ctx, clist, clistlen) \
  SSL_CTX_ctrl(ctx, SSL_CTRL_SET_CLIENT_CERT_TYPES, clistlen, (char *)clist)
#define SSL_set1_client_certificate_types(s, clist, clistlen) \
  SSL_ctrl(s, SSL_CTRL_SET_CLIENT_CERT_TYPES, clistlen, (char *)clist)

#define SSL_get_server_tmp_key(s, pk) \
  SSL_ctrl(s, SSL_CTRL_GET_SERVER_TMP_KEY, 0, pk)

#define SSL_get0_ec_point_formats(s, plst) \
  SSL_ctrl(s, SSL_CTRL_GET_EC_POINT_FORMATS, 0, (char *)plst)

OPENSSL_EXPORT int SSL_CTX_set_cipher_list(SSL_CTX *, const char *str);
OPENSSL_EXPORT int SSL_CTX_set_cipher_list_tls11(SSL_CTX *, const char *str);
OPENSSL_EXPORT long SSL_CTX_set_timeout(SSL_CTX *ctx, long t);
OPENSSL_EXPORT long SSL_CTX_get_timeout(const SSL_CTX *ctx);
OPENSSL_EXPORT X509_STORE *SSL_CTX_get_cert_store(const SSL_CTX *);
OPENSSL_EXPORT void SSL_CTX_set_cert_store(SSL_CTX *, X509_STORE *);
OPENSSL_EXPORT int SSL_want(const SSL *s);

OPENSSL_EXPORT void SSL_CTX_flush_sessions(SSL_CTX *ctx, long tm);

/* SSL_get_current_cipher returns the cipher used in the current outgoing
 * connection state, or NULL if the null cipher is active. */
OPENSSL_EXPORT const SSL_CIPHER *SSL_get_current_cipher(const SSL *s);

OPENSSL_EXPORT int SSL_get_fd(const SSL *s);
OPENSSL_EXPORT int SSL_get_rfd(const SSL *s);
OPENSSL_EXPORT int SSL_get_wfd(const SSL *s);
OPENSSL_EXPORT const char *SSL_get_cipher_list(const SSL *s, int n);
OPENSSL_EXPORT int SSL_pending(const SSL *s);
OPENSSL_EXPORT int SSL_set_fd(SSL *s, int fd);
OPENSSL_EXPORT int SSL_set_rfd(SSL *s, int fd);
OPENSSL_EXPORT int SSL_set_wfd(SSL *s, int fd);
OPENSSL_EXPORT void SSL_set_bio(SSL *s, BIO *rbio, BIO *wbio);
OPENSSL_EXPORT BIO *SSL_get_rbio(const SSL *s);
OPENSSL_EXPORT BIO *SSL_get_wbio(const SSL *s);
OPENSSL_EXPORT int SSL_set_cipher_list(SSL *s, const char *str);
OPENSSL_EXPORT int SSL_get_verify_mode(const SSL *s);
OPENSSL_EXPORT int SSL_get_verify_depth(const SSL *s);
OPENSSL_EXPORT int (*SSL_get_verify_callback(const SSL *s))(int,
                                                            X509_STORE_CTX *);
OPENSSL_EXPORT void SSL_set_verify(SSL *s, int mode,
                                   int (*callback)(int ok,
                                                   X509_STORE_CTX *ctx));
OPENSSL_EXPORT void SSL_set_verify_depth(SSL *s, int depth);
OPENSSL_EXPORT void SSL_set_cert_cb(SSL *s, int (*cb)(SSL *ssl, void *arg),
                                    void *arg);
OPENSSL_EXPORT int SSL_use_RSAPrivateKey(SSL *ssl, RSA *rsa);
OPENSSL_EXPORT int SSL_use_RSAPrivateKey_ASN1(SSL *ssl, uint8_t *d, long len);
OPENSSL_EXPORT int SSL_use_PrivateKey(SSL *ssl, EVP_PKEY *pkey);
OPENSSL_EXPORT int SSL_use_PrivateKey_ASN1(int pk, SSL *ssl, const uint8_t *d,
                                           long len);
OPENSSL_EXPORT int SSL_use_certificate(SSL *ssl, X509 *x);
OPENSSL_EXPORT int SSL_use_certificate_ASN1(SSL *ssl, const uint8_t *d,
                                            int len);

OPENSSL_EXPORT int SSL_use_RSAPrivateKey_file(SSL *ssl, const char *file,
                                              int type);
OPENSSL_EXPORT int SSL_use_PrivateKey_file(SSL *ssl, const char *file,
                                           int type);
OPENSSL_EXPORT int SSL_use_certificate_file(SSL *ssl, const char *file,
                                            int type);
OPENSSL_EXPORT int SSL_CTX_use_RSAPrivateKey_file(SSL_CTX *ctx,
                                                  const char *file, int type);
OPENSSL_EXPORT int SSL_CTX_use_PrivateKey_file(SSL_CTX *ctx, const char *file,
                                               int type);
OPENSSL_EXPORT int SSL_CTX_use_certificate_file(SSL_CTX *ctx, const char *file,
                                                int type);
OPENSSL_EXPORT int SSL_CTX_use_certificate_chain_file(
    SSL_CTX *ctx, const char *file); /* PEM type */
OPENSSL_EXPORT STACK_OF(X509_NAME) *SSL_load_client_CA_file(const char *file);
OPENSSL_EXPORT int SSL_add_file_cert_subjects_to_stack(STACK_OF(X509_NAME) *
                                                           stackCAs,
                                                       const char *file);
OPENSSL_EXPORT int SSL_add_dir_cert_subjects_to_stack(STACK_OF(X509_NAME) *
                                                          stackCAs,
                                                      const char *dir);

/* SSL_load_error_strings does nothing. */
OPENSSL_EXPORT void SSL_load_error_strings(void);

OPENSSL_EXPORT const char *SSL_state_string(const SSL *s);
OPENSSL_EXPORT const char *SSL_rstate_string(const SSL *s);
OPENSSL_EXPORT const char *SSL_state_string_long(const SSL *s);
OPENSSL_EXPORT const char *SSL_rstate_string_long(const SSL *s);
OPENSSL_EXPORT long SSL_SESSION_get_time(const SSL_SESSION *s);
OPENSSL_EXPORT long SSL_SESSION_set_time(SSL_SESSION *s, long t);
OPENSSL_EXPORT long SSL_SESSION_get_timeout(const SSL_SESSION *s);
OPENSSL_EXPORT long SSL_SESSION_set_timeout(SSL_SESSION *s, long t);
OPENSSL_EXPORT X509 *SSL_SESSION_get0_peer(SSL_SESSION *s);
OPENSSL_EXPORT int SSL_SESSION_set1_id_context(SSL_SESSION *s,
                                               const uint8_t *sid_ctx,
                                               unsigned int sid_ctx_len);

OPENSSL_EXPORT SSL_SESSION *SSL_SESSION_new(void);
OPENSSL_EXPORT const uint8_t *SSL_SESSION_get_id(const SSL_SESSION *s,
                                                 unsigned int *len);
OPENSSL_EXPORT int SSL_SESSION_print_fp(FILE *fp, const SSL_SESSION *ses);
OPENSSL_EXPORT int SSL_SESSION_print(BIO *fp, const SSL_SESSION *ses);

/* SSL_SESSION_up_ref, if |session| is not NULL, increments the reference count
 * of |session|. It then returns |session|. */
OPENSSL_EXPORT SSL_SESSION *SSL_SESSION_up_ref(SSL_SESSION *session);

/* SSL_SESSION_free decrements the reference count of |session|. If it reaches
 * zero, all data referenced by |session| and |session| itself are released. */
OPENSSL_EXPORT void SSL_SESSION_free(SSL_SESSION *session);

OPENSSL_EXPORT int SSL_set_session(SSL *to, SSL_SESSION *session);
OPENSSL_EXPORT int SSL_CTX_add_session(SSL_CTX *s, SSL_SESSION *c);
OPENSSL_EXPORT int SSL_CTX_remove_session(SSL_CTX *, SSL_SESSION *c);
OPENSSL_EXPORT int SSL_CTX_set_generate_session_id(SSL_CTX *, GEN_SESSION_CB);
OPENSSL_EXPORT int SSL_set_generate_session_id(SSL *, GEN_SESSION_CB);
OPENSSL_EXPORT int SSL_has_matching_session_id(const SSL *ssl,
                                               const uint8_t *id,
                                               unsigned int id_len);

/* SSL_SESSION_to_bytes serializes |in| into a newly allocated buffer and sets
 * |*out_data| to that buffer and |*out_len| to its length. The caller takes
 * ownership of the buffer and must call |OPENSSL_free| when done. It returns
 * one on success and zero on error. */
OPENSSL_EXPORT int SSL_SESSION_to_bytes(SSL_SESSION *in, uint8_t **out_data,
                                        size_t *out_len);

/* SSL_SESSION_to_bytes_for_ticket serializes |in|, but excludes the session ID
 * which is not necessary in a session ticket. */
OPENSSL_EXPORT int SSL_SESSION_to_bytes_for_ticket(SSL_SESSION *in,
                                                   uint8_t **out_data,
                                                   size_t *out_len);

/* Deprecated: i2d_SSL_SESSION serializes |in| to the bytes pointed to by
 * |*pp|. On success, it returns the number of bytes written and advances |*pp|
 * by that many bytes. On failure, it returns -1. If |pp| is NULL, no bytes are
 * written and only the length is returned.
 *
 * Use SSL_SESSION_to_bytes instead. */
OPENSSL_EXPORT int i2d_SSL_SESSION(SSL_SESSION *in, uint8_t **pp);

/* d2i_SSL_SESSION deserializes a serialized buffer contained in the |length|
 * bytes pointed to by |*pp|. It returns the new SSL_SESSION and advances |*pp|
 * by the number of bytes consumed on success and NULL on failure. If |a| is
 * NULL, the caller takes ownership of the new session and must call
 * |SSL_SESSION_free| when done.
 *
 * If |a| and |*a| are not NULL, the SSL_SESSION at |*a| is overridden with the
 * deserialized session rather than allocating a new one. In addition, |a| is
 * not NULL, but |*a| is, |*a| is set to the new SSL_SESSION.
 *
 * Passing a value other than NULL to |a| is deprecated. */
OPENSSL_EXPORT SSL_SESSION *d2i_SSL_SESSION(SSL_SESSION **a, const uint8_t **pp,
                                            long length);

OPENSSL_EXPORT X509 *SSL_get_peer_certificate(const SSL *s);

OPENSSL_EXPORT STACK_OF(X509) *SSL_get_peer_cert_chain(const SSL *s);

OPENSSL_EXPORT int SSL_CTX_get_verify_mode(const SSL_CTX *ctx);
OPENSSL_EXPORT int SSL_CTX_get_verify_depth(const SSL_CTX *ctx);
OPENSSL_EXPORT int (*SSL_CTX_get_verify_callback(const SSL_CTX *ctx))(
    int, X509_STORE_CTX *);
OPENSSL_EXPORT void SSL_CTX_set_verify(SSL_CTX *ctx, int mode,
                                       int (*callback)(int, X509_STORE_CTX *));
OPENSSL_EXPORT void SSL_CTX_set_verify_depth(SSL_CTX *ctx, int depth);
OPENSSL_EXPORT void SSL_CTX_set_cert_verify_callback(
    SSL_CTX *ctx, int (*cb)(X509_STORE_CTX *, void *), void *arg);
OPENSSL_EXPORT void SSL_CTX_set_cert_cb(SSL_CTX *c,
                                        int (*cb)(SSL *ssl, void *arg),
                                        void *arg);
OPENSSL_EXPORT int SSL_CTX_use_RSAPrivateKey(SSL_CTX *ctx, RSA *rsa);
OPENSSL_EXPORT int SSL_CTX_use_RSAPrivateKey_ASN1(SSL_CTX *ctx,
                                                  const uint8_t *d, long len);
OPENSSL_EXPORT int SSL_CTX_use_PrivateKey(SSL_CTX *ctx, EVP_PKEY *pkey);
OPENSSL_EXPORT int SSL_CTX_use_PrivateKey_ASN1(int pk, SSL_CTX *ctx,
                                               const uint8_t *d, long len);
OPENSSL_EXPORT int SSL_CTX_use_certificate(SSL_CTX *ctx, X509 *x);
OPENSSL_EXPORT int SSL_CTX_use_certificate_ASN1(SSL_CTX *ctx, int len,
                                                const uint8_t *d);

OPENSSL_EXPORT void SSL_CTX_set_default_passwd_cb(SSL_CTX *ctx,
                                                  pem_password_cb *cb);
OPENSSL_EXPORT void SSL_CTX_set_default_passwd_cb_userdata(SSL_CTX *ctx,
                                                           void *u);

OPENSSL_EXPORT int SSL_CTX_check_private_key(const SSL_CTX *ctx);
OPENSSL_EXPORT int SSL_check_private_key(const SSL *ctx);

OPENSSL_EXPORT int SSL_CTX_set_session_id_context(SSL_CTX *ctx,
                                                  const uint8_t *sid_ctx,
                                                  unsigned int sid_ctx_len);

OPENSSL_EXPORT int SSL_set_session_id_context(SSL *ssl, const uint8_t *sid_ctx,
                                              unsigned int sid_ctx_len);

OPENSSL_EXPORT int SSL_CTX_set_purpose(SSL_CTX *s, int purpose);
OPENSSL_EXPORT int SSL_set_purpose(SSL *s, int purpose);
OPENSSL_EXPORT int SSL_CTX_set_trust(SSL_CTX *s, int trust);
OPENSSL_EXPORT int SSL_set_trust(SSL *s, int trust);

OPENSSL_EXPORT int SSL_CTX_set1_param(SSL_CTX *ctx, X509_VERIFY_PARAM *vpm);
OPENSSL_EXPORT int SSL_set1_param(SSL *ssl, X509_VERIFY_PARAM *vpm);

OPENSSL_EXPORT X509_VERIFY_PARAM *SSL_CTX_get0_param(SSL_CTX *ctx);
OPENSSL_EXPORT X509_VERIFY_PARAM *SSL_get0_param(SSL *ssl);

OPENSSL_EXPORT void SSL_certs_clear(SSL *s);
OPENSSL_EXPORT int SSL_accept(SSL *ssl);
OPENSSL_EXPORT int SSL_connect(SSL *ssl);
OPENSSL_EXPORT int SSL_read(SSL *ssl, void *buf, int num);
OPENSSL_EXPORT int SSL_peek(SSL *ssl, void *buf, int num);
OPENSSL_EXPORT int SSL_write(SSL *ssl, const void *buf, int num);
OPENSSL_EXPORT long SSL_ctrl(SSL *ssl, int cmd, long larg, void *parg);
OPENSSL_EXPORT long SSL_CTX_ctrl(SSL_CTX *ctx, int cmd, long larg, void *parg);

OPENSSL_EXPORT int SSL_get_error(const SSL *s, int ret_code);
/* SSL_get_version returns a string describing the TLS version used by |s|. For
 * example, "TLSv1.2" or "SSLv3". */
OPENSSL_EXPORT const char *SSL_get_version(const SSL *s);
/* SSL_SESSION_get_version returns a string describing the TLS version used by
 * |sess|. For example, "TLSv1.2" or "SSLv3". */
OPENSSL_EXPORT const char *SSL_SESSION_get_version(const SSL_SESSION *sess);

OPENSSL_EXPORT STACK_OF(SSL_CIPHER) *SSL_get_ciphers(const SSL *s);

OPENSSL_EXPORT int SSL_do_handshake(SSL *s);

/* SSL_renegotiate_pending returns one if |ssl| is in the middle of a
 * renegotiation. */
OPENSSL_EXPORT int SSL_renegotiate_pending(SSL *ssl);

OPENSSL_EXPORT int SSL_shutdown(SSL *s);

OPENSSL_EXPORT const char *SSL_alert_type_string_long(int value);
OPENSSL_EXPORT const char *SSL_alert_type_string(int value);
OPENSSL_EXPORT const char *SSL_alert_desc_string_long(int value);
OPENSSL_EXPORT const char *SSL_alert_desc_string(int value);

OPENSSL_EXPORT void SSL_set_client_CA_list(SSL *s,
                                           STACK_OF(X509_NAME) *name_list);
OPENSSL_EXPORT void SSL_CTX_set_client_CA_list(SSL_CTX *ctx,
                                               STACK_OF(X509_NAME) *name_list);
OPENSSL_EXPORT STACK_OF(X509_NAME) *SSL_get_client_CA_list(const SSL *s);
OPENSSL_EXPORT STACK_OF(X509_NAME) *
    SSL_CTX_get_client_CA_list(const SSL_CTX *s);
OPENSSL_EXPORT int SSL_add_client_CA(SSL *ssl, X509 *x);
OPENSSL_EXPORT int SSL_CTX_add_client_CA(SSL_CTX *ctx, X509 *x);

OPENSSL_EXPORT long SSL_get_default_timeout(const SSL *s);

OPENSSL_EXPORT STACK_OF(X509_NAME) *SSL_dup_CA_list(STACK_OF(X509_NAME) *sk);

OPENSSL_EXPORT X509 *SSL_get_certificate(const SSL *ssl);
OPENSSL_EXPORT EVP_PKEY *SSL_get_privatekey(const SSL *ssl);

OPENSSL_EXPORT X509 *SSL_CTX_get0_certificate(const SSL_CTX *ctx);
OPENSSL_EXPORT EVP_PKEY *SSL_CTX_get0_privatekey(const SSL_CTX *ctx);

OPENSSL_EXPORT void SSL_CTX_set_quiet_shutdown(SSL_CTX *ctx, int mode);
OPENSSL_EXPORT int SSL_CTX_get_quiet_shutdown(const SSL_CTX *ctx);
OPENSSL_EXPORT void SSL_set_quiet_shutdown(SSL *ssl, int mode);
OPENSSL_EXPORT int SSL_get_quiet_shutdown(const SSL *ssl);
OPENSSL_EXPORT void SSL_set_shutdown(SSL *ssl, int mode);
OPENSSL_EXPORT int SSL_get_shutdown(const SSL *ssl);
OPENSSL_EXPORT int SSL_version(const SSL *ssl);
OPENSSL_EXPORT int SSL_CTX_set_default_verify_paths(SSL_CTX *ctx);
OPENSSL_EXPORT int SSL_CTX_load_verify_locations(SSL_CTX *ctx,
                                                 const char *CAfile,
                                                 const char *CApath);
#define SSL_get0_session SSL_get_session /* just peek at pointer */
OPENSSL_EXPORT SSL_SESSION *SSL_get_session(const SSL *ssl);
OPENSSL_EXPORT SSL_SESSION *SSL_get1_session(
    SSL *ssl); /* obtain a reference count */
OPENSSL_EXPORT SSL_CTX *SSL_get_SSL_CTX(const SSL *ssl);
OPENSSL_EXPORT SSL_CTX *SSL_set_SSL_CTX(SSL *ssl, SSL_CTX *ctx);
OPENSSL_EXPORT void SSL_set_info_callback(SSL *ssl,
                                          void (*cb)(const SSL *ssl, int type,
                                                     int val));
OPENSSL_EXPORT void (*SSL_get_info_callback(const SSL *ssl))(const SSL *ssl,
                                                             int type, int val);
OPENSSL_EXPORT int SSL_state(const SSL *ssl);

OPENSSL_EXPORT void SSL_set_verify_result(SSL *ssl, long v);
OPENSSL_EXPORT long SSL_get_verify_result(const SSL *ssl);

OPENSSL_EXPORT int SSL_set_ex_data(SSL *ssl, int idx, void *data);
OPENSSL_EXPORT void *SSL_get_ex_data(const SSL *ssl, int idx);
OPENSSL_EXPORT int SSL_get_ex_new_index(long argl, void *argp,
                                        CRYPTO_EX_new *new_func,
                                        CRYPTO_EX_dup *dup_func,
                                        CRYPTO_EX_free *free_func);

OPENSSL_EXPORT int SSL_SESSION_set_ex_data(SSL_SESSION *ss, int idx,
                                           void *data);
OPENSSL_EXPORT void *SSL_SESSION_get_ex_data(const SSL_SESSION *ss, int idx);
OPENSSL_EXPORT int SSL_SESSION_get_ex_new_index(long argl, void *argp,
                                                CRYPTO_EX_new *new_func,
                                                CRYPTO_EX_dup *dup_func,
                                                CRYPTO_EX_free *free_func);

OPENSSL_EXPORT int SSL_CTX_set_ex_data(SSL_CTX *ssl, int idx, void *data);
OPENSSL_EXPORT void *SSL_CTX_get_ex_data(const SSL_CTX *ssl, int idx);
OPENSSL_EXPORT int SSL_CTX_get_ex_new_index(long argl, void *argp,
                                            CRYPTO_EX_new *new_func,
                                            CRYPTO_EX_dup *dup_func,
                                            CRYPTO_EX_free *free_func);

OPENSSL_EXPORT int SSL_get_ex_data_X509_STORE_CTX_idx(void);

/* SSL_CTX_sess_set_cache_size sets the maximum size of |ctx|'s session cache to
 * |size|. It returns the previous value. */
OPENSSL_EXPORT unsigned long SSL_CTX_sess_set_cache_size(SSL_CTX *ctx,
                                                         unsigned long size);

/* SSL_CTX_sess_get_cache_size returns the maximum size of |ctx|'s session
 * cache. */
OPENSSL_EXPORT unsigned long SSL_CTX_sess_get_cache_size(const SSL_CTX *ctx);

/* SSL_SESS_CACHE_* are the possible session cache mode bits.
 * TODO(davidben): Document. */
#define SSL_SESS_CACHE_OFF 0x0000
#define SSL_SESS_CACHE_CLIENT 0x0001
#define SSL_SESS_CACHE_SERVER 0x0002
#define SSL_SESS_CACHE_BOTH (SSL_SESS_CACHE_CLIENT | SSL_SESS_CACHE_SERVER)
#define SSL_SESS_CACHE_NO_AUTO_CLEAR 0x0080
#define SSL_SESS_CACHE_NO_INTERNAL_LOOKUP 0x0100
#define SSL_SESS_CACHE_NO_INTERNAL_STORE 0x0200
#define SSL_SESS_CACHE_NO_INTERNAL \
  (SSL_SESS_CACHE_NO_INTERNAL_LOOKUP | SSL_SESS_CACHE_NO_INTERNAL_STORE)

/* SSL_CTX_set_session_cache_mode sets the session cache mode bits for |ctx| to
 * |mode|. It returns the previous value. */
OPENSSL_EXPORT int SSL_CTX_set_session_cache_mode(SSL_CTX *ctx, int mode);

/* SSL_CTX_get_session_cache_mode returns the session cache mode bits for
 * |ctx| */
OPENSSL_EXPORT int SSL_CTX_get_session_cache_mode(const SSL_CTX *ctx);

/* SSL_CTX_get_max_cert_list returns the maximum length, in bytes, of a peer
 * certificate chain accepted by |ctx|. */
OPENSSL_EXPORT size_t SSL_CTX_get_max_cert_list(const SSL_CTX *ctx);

/* SSL_CTX_set_max_cert_list sets the maximum length, in bytes, of a peer
 * certificate chain to |max_cert_list|. This affects how much memory may be
 * consumed during the handshake. */
OPENSSL_EXPORT void SSL_CTX_set_max_cert_list(SSL_CTX *ctx,
                                              size_t max_cert_list);

/* SSL_get_max_cert_list returns the maximum length, in bytes, of a peer
 * certificate chain accepted by |ssl|. */
OPENSSL_EXPORT size_t SSL_get_max_cert_list(const SSL *ssl);

/* SSL_set_max_cert_list sets the maximum length, in bytes, of a peer
 * certificate chain to |max_cert_list|. This affects how much memory may be
 * consumed during the handshake. */
OPENSSL_EXPORT void SSL_set_max_cert_list(SSL *ssl, size_t max_cert_list);

/* SSL_CTX_set_max_send_fragment sets the maximum length, in bytes, of records
 * sent by |ctx|. Beyond this length, handshake messages and application data
 * will be split into multiple records. */
OPENSSL_EXPORT void SSL_CTX_set_max_send_fragment(SSL_CTX *ctx,
                                                  size_t max_send_fragment);

/* SSL_set_max_send_fragment sets the maximum length, in bytes, of records
 * sent by |ssl|. Beyond this length, handshake messages and application data
 * will be split into multiple records. */
OPENSSL_EXPORT void SSL_set_max_send_fragment(SSL *ssl,
                                              size_t max_send_fragment);

/* SSL_CTX_set_tmp_dh_callback configures |ctx| to use |callback| to determine
 * the group for DHE ciphers. |callback| should ignore |is_export| and
 * |keylength| and return a |DH| of the selected group or NULL on error. Only
 * the parameters are used, so the |DH| needn't have a generated keypair.
 *
 * WARNING: The caller does not take ownership of the resulting |DH|, so
 * |callback| must save and release the object elsewhere. */
OPENSSL_EXPORT void SSL_CTX_set_tmp_dh_callback(
    SSL_CTX *ctx, DH *(*callback)(SSL *ssl, int is_export, int keylength));

/* SSL_set_tmp_dh_callback configures |ssl| to use |callback| to determine the
 * group for DHE ciphers. |callback| should ignore |is_export| and |keylength|
 * and return a |DH| of the selected group or NULL on error. Only the
 * parameters are used, so the |DH| needn't have a generated keypair.
 *
 * WARNING: The caller does not take ownership of the resulting |DH|, so
 * |callback| must save and release the object elsewhere. */
OPENSSL_EXPORT void SSL_set_tmp_dh_callback(SSL *ssl,
                                            DH *(*dh)(SSL *ssl, int is_export,
                                                      int keylength));

/* SSL_CTX_set_tmp_ecdh_callback configures |ctx| to use |callback| to determine
 * the curve for ephemeral ECDH keys. |callback| should ignore |is_export| and
 * |keylength| and return an |EC_KEY| of the selected curve or NULL on
 * error. Only the curve is used, so the |EC_KEY| needn't have a generated
 * keypair.
 *
 * If the callback is unset, an appropriate curve will be chosen automatically.
 * (This is recommended.)
 *
 * WARNING: The caller does not take ownership of the resulting |EC_KEY|, so
 * |callback| must save and release the object elsewhere. */
OPENSSL_EXPORT void SSL_CTX_set_tmp_ecdh_callback(
    SSL_CTX *ctx, EC_KEY *(*callback)(SSL *ssl, int is_export, int keylength));

/* SSL_set_tmp_ecdh_callback configures |ssl| to use |callback| to determine the
 * curve for ephemeral ECDH keys. |callback| should ignore |is_export| and
 * |keylength| and return an |EC_KEY| of the selected curve or NULL on
 * error. Only the curve is used, so the |EC_KEY| needn't have a generated
 * keypair.
 *
 * If the callback is unset, an appropriate curve will be chosen automatically.
 * (This is recommended.)
 *
 * WARNING: The caller does not take ownership of the resulting |EC_KEY|, so
 * |callback| must save and release the object elsewhere. */
OPENSSL_EXPORT void SSL_set_tmp_ecdh_callback(
    SSL *ssl, EC_KEY *(*callback)(SSL *ssl, int is_export, int keylength));

OPENSSL_EXPORT const void *SSL_get_current_compression(SSL *s);
OPENSSL_EXPORT const void *SSL_get_current_expansion(SSL *s);

OPENSSL_EXPORT int SSL_cache_hit(SSL *s);
OPENSSL_EXPORT int SSL_is_server(SSL *s);

/* SSL_CTX_set_dos_protection_cb sets a callback that is called once the
 * resumption decision for a ClientHello has been made. It can return 1 to
 * allow the handshake to continue or zero to cause the handshake to abort. */
OPENSSL_EXPORT void SSL_CTX_set_dos_protection_cb(
    SSL_CTX *ctx, int (*cb)(const struct ssl_early_callback_ctx *));

/* SSL_get_structure_sizes returns the sizes of the SSL, SSL_CTX and
 * SSL_SESSION structures so that a test can ensure that outside code agrees on
 * these values. */
OPENSSL_EXPORT void SSL_get_structure_sizes(size_t *ssl_size,
                                            size_t *ssl_ctx_size,
                                            size_t *ssl_session_size);

OPENSSL_EXPORT void ERR_load_SSL_strings(void);

/* SSL_get_rc4_state sets |*read_key| and |*write_key| to the RC4 states for
 * the read and write directions. It returns one on success or zero if |ssl|
 * isn't using an RC4-based cipher suite. */
OPENSSL_EXPORT int SSL_get_rc4_state(const SSL *ssl, const RC4_KEY **read_key,
                                     const RC4_KEY **write_key);


/* Deprecated functions. */

/* SSL_CIPHER_description writes a description of |cipher| into |buf| and
 * returns |buf|. If |buf| is NULL, it returns a newly allocated string, to be
 * freed with |OPENSSL_free|, or NULL on error.
 *
 * The description includes a trailing newline and has the form:
 * AES128-SHA              SSLv3 Kx=RSA      Au=RSA  Enc=AES(128)  Mac=SHA1
 *
 * Consider |SSL_CIPHER_get_name| or |SSL_CIPHER_get_rfc_name| instead. */
OPENSSL_EXPORT const char *SSL_CIPHER_description(const SSL_CIPHER *cipher,
                                                  char *buf, int len);

/* SSL_CIPHER_get_version returns the string "TLSv1/SSLv3". */
OPENSSL_EXPORT const char *SSL_CIPHER_get_version(const SSL_CIPHER *cipher);

/* SSL_COMP_get_compression_methods returns NULL. */
OPENSSL_EXPORT void *SSL_COMP_get_compression_methods(void);

/* SSL_COMP_add_compression_method returns one. */
OPENSSL_EXPORT int SSL_COMP_add_compression_method(int id, void *cm);

/* SSL_COMP_get_name returns NULL. */
OPENSSL_EXPORT const char *SSL_COMP_get_name(const void *comp);

/* SSLv23_method calls |TLS_method|. */
OPENSSL_EXPORT const SSL_METHOD *SSLv23_method(void);

/* Version-specific methods behave exactly like |TLS_method| and |DTLS_method|
 * except they also call |SSL_CTX_set_min_version| and |SSL_CTX_set_max_version|
 * to lock connections to that protocol version. */
OPENSSL_EXPORT const SSL_METHOD *SSLv3_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_1_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_2_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_2_method(void);

/* Client- and server-specific methods call their corresponding generic
 * methods. */
OPENSSL_EXPORT const SSL_METHOD *SSLv23_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *SSLv23_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *SSLv3_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *SSLv3_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_1_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_1_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_2_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *TLSv1_2_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLS_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLS_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_client_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_2_server_method(void);
OPENSSL_EXPORT const SSL_METHOD *DTLSv1_2_client_method(void);

/* SSL_clear resets |ssl| to allow another connection and returns one on success
 * or zero on failure. It returns most configuration state but releases memory
 * associated with the current connection.
 *
 * Free |ssl| and create a new one instead. */
OPENSSL_EXPORT int SSL_clear(SSL *ssl);

/* SSL_CTX_set_tmp_rsa_callback does nothing. */
OPENSSL_EXPORT void SSL_CTX_set_tmp_rsa_callback(
    SSL_CTX *ctx, RSA *(*cb)(SSL *ssl, int is_export, int keylength));

/* SSL_set_tmp_rsa_callback does nothing. */
OPENSSL_EXPORT void SSL_set_tmp_rsa_callback(SSL *ssl,
                                             RSA *(*cb)(SSL *ssl, int is_export,
                                                        int keylength));

/* SSL_CTX_sess_connect returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_connect(const SSL_CTX *ctx);

/* SSL_CTX_sess_connect_good returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_connect_good(const SSL_CTX *ctx);

/* SSL_CTX_sess_connect_renegotiate returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_connect_renegotiate(const SSL_CTX *ctx);

/* SSL_CTX_sess_accept returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_accept(const SSL_CTX *ctx);

/* SSL_CTX_sess_accept_renegotiate returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_accept_renegotiate(const SSL_CTX *ctx);

/* SSL_CTX_sess_accept_good returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_accept_good(const SSL_CTX *ctx);

/* SSL_CTX_sess_hits returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_hits(const SSL_CTX *ctx);

/* SSL_CTX_sess_cb_hits returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_cb_hits(const SSL_CTX *ctx);

/* SSL_CTX_sess_misses returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_misses(const SSL_CTX *ctx);

/* SSL_CTX_sess_timeouts returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_timeouts(const SSL_CTX *ctx);

/* SSL_CTX_sess_cache_full returns zero. */
OPENSSL_EXPORT int SSL_CTX_sess_cache_full(const SSL_CTX *ctx);

/* SSL_cutthrough_complete calls |SSL_in_false_start|. */
OPENSSL_EXPORT int SSL_cutthrough_complete(const SSL *s);

/* SSL_num_renegotiations calls |SSL_total_renegotiations|. */
OPENSSL_EXPORT int SSL_num_renegotiations(const SSL *ssl);

/* SSL_CTX_need_tmp_RSA returns zero. */
OPENSSL_EXPORT int SSL_CTX_need_tmp_RSA(const SSL_CTX *ctx);

/* SSL_need_tmp_RSA returns zero. */
OPENSSL_EXPORT int SSL_need_tmp_RSA(const SSL *ssl);

/* SSL_CTX_set_tmp_rsa returns one. */
OPENSSL_EXPORT int SSL_CTX_set_tmp_rsa(SSL_CTX *ctx, const RSA *rsa);

/* SSL_set_tmp_rsa returns one. */
OPENSSL_EXPORT int SSL_set_tmp_rsa(SSL *ssl, const RSA *rsa);

/* SSL_CTX_get_read_head returns zero. */
OPENSSL_EXPORT int SSL_CTX_get_read_ahead(const SSL_CTX *ctx);

/* SSL_CTX_set_read_ahead does nothing. */
OPENSSL_EXPORT void SSL_CTX_set_read_ahead(SSL_CTX *ctx, int yes);

/* SSL_get_read_head returns zero. */
OPENSSL_EXPORT int SSL_get_read_ahead(const SSL *s);

/* SSL_set_read_ahead does nothing. */
OPENSSL_EXPORT void SSL_set_read_ahead(SSL *s, int yes);

/* SSL_renegotiate put an error on the error queue and returns zero. */
OPENSSL_EXPORT int SSL_renegotiate(SSL *ssl);

/* SSL_set_state does nothing. */
OPENSSL_EXPORT void SSL_set_state(SSL *ssl, int state);


/* Android compatibility section.
 *
 * These functions are declared, temporarily, for Android because
 * wpa_supplicant will take a little time to sync with upstream. Outside of
 * Android they'll have no definition. */

#define SSL_F_SSL_SET_SESSION_TICKET_EXT doesnt_exist

OPENSSL_EXPORT int SSL_set_session_ticket_ext(SSL *s, void *ext_data,
                                              int ext_len);
OPENSSL_EXPORT int SSL_set_session_secret_cb(SSL *s, void *cb, void *arg);
OPENSSL_EXPORT int SSL_set_session_ticket_ext_cb(SSL *s, void *cb, void *arg);
OPENSSL_EXPORT int SSL_set_ssl_method(SSL *s, const SSL_METHOD *method);

#define OPENSSL_VERSION_TEXT "BoringSSL"

#define SSLEAY_VERSION 0

/* SSLeay_version is a compatibility function that returns the string
 * "BoringSSL". */
OPENSSL_EXPORT const char *SSLeay_version(int unused);


/* Preprocessor compatibility section.
 *
 * Historically, a number of APIs were implemented in OpenSSL as macros and
 * constants to 'ctrl' functions. To avoid breaking #ifdefs in consumers, this
 * section defines a number of legacy macros. */

#define SSL_CTRL_NEED_TMP_RSA doesnt_exist
#define SSL_CTRL_SET_TMP_RSA doesnt_exist
#define SSL_CTRL_SET_TMP_DH doesnt_exist
#define SSL_CTRL_SET_TMP_ECDH doesnt_exist
#define SSL_CTRL_SET_TMP_RSA_CB doesnt_exist
#define SSL_CTRL_SET_TMP_DH_CB doesnt_exist
#define SSL_CTRL_SET_TMP_ECDH_CB doesnt_exist
#define SSL_CTRL_GET_SESSION_REUSED doesnt_exist
#define SSL_CTRL_GET_NUM_RENEGOTIATIONS doesnt_exist
#define SSL_CTRL_GET_TOTAL_RENEGOTIATIONS doesnt_exist
#define SSL_CTRL_SET_MSG_CALLBACK doesnt_exist
#define SSL_CTRL_SET_MSG_CALLBACK_ARG doesnt_exist
#define SSL_CTRL_SET_MTU doesnt_exist
#define SSL_CTRL_SESS_NUMBER doesnt_exist
#define SSL_CTRL_OPTIONS doesnt_exist
#define SSL_CTRL_MODE doesnt_exist
#define SSL_CTRL_GET_READ_AHEAD doesnt_exist
#define SSL_CTRL_SET_READ_AHEAD doesnt_exist
#define SSL_CTRL_SET_SESS_CACHE_SIZE doesnt_exist
#define SSL_CTRL_GET_SESS_CACHE_SIZE doesnt_exist
#define SSL_CTRL_SET_SESS_CACHE_MODE doesnt_exist
#define SSL_CTRL_GET_SESS_CACHE_MODE doesnt_exist
#define SSL_CTRL_GET_MAX_CERT_LIST doesnt_exist
#define SSL_CTRL_SET_MAX_CERT_LIST doesnt_exist
#define SSL_CTRL_SET_MAX_SEND_FRAGMENT doesnt_exist
#define SSL_CTRL_SET_TLSEXT_SERVERNAME_CB doesnt_exist
#define SSL_CTRL_SET_TLSEXT_SERVERNAME_ARG doesnt_exist
#define SSL_CTRL_SET_TLSEXT_HOSTNAME doesnt_exist
#define SSL_CTRL_SET_TLSEXT_TICKET_KEY_CB doesnt_exist
#define DTLS_CTRL_GET_TIMEOUT doesnt_exist
#define DTLS_CTRL_HANDLE_TIMEOUT doesnt_exist
#define SSL_CTRL_GET_RI_SUPPORT doesnt_exist
#define SSL_CTRL_CLEAR_OPTIONS doesnt_exist
#define SSL_CTRL_CLEAR_MODE doesnt_exist
#define SSL_CTRL_CHANNEL_ID doesnt_exist
#define SSL_CTRL_GET_CHANNEL_ID doesnt_exist
#define SSL_CTRL_SET_CHANNEL_ID doesnt_exist

#define SSL_CTX_need_tmp_RSA SSL_CTX_need_tmp_RSA
#define SSL_need_tmp_RSA SSL_need_tmp_RSA
#define SSL_CTX_set_tmp_rsa SSL_CTX_set_tmp_rsa
#define SSL_set_tmp_rsa SSL_set_tmp_rsa
#define SSL_CTX_set_tmp_dh SSL_CTX_set_tmp_dh
#define SSL_set_tmp_dh SSL_set_tmp_dh
#define SSL_CTX_set_tmp_ecdh SSL_CTX_set_tmp_ecdh
#define SSL_set_tmp_ecdh SSL_set_tmp_ecdh
#define SSL_session_reused SSL_session_reused
#define SSL_num_renegotiations SSL_num_renegotiations
#define SSL_total_renegotiations SSL_total_renegotiations
#define SSL_CTX_set_msg_callback_arg SSL_CTX_set_msg_callback_arg
#define SSL_set_msg_callback_arg SSL_set_msg_callback_arg
#define SSL_set_mtu SSL_set_mtu
#define SSL_CTX_sess_number SSL_CTX_sess_number
#define SSL_CTX_get_options SSL_CTX_get_options
#define SSL_CTX_set_options SSL_CTX_set_options
#define SSL_get_options SSL_get_options
#define SSL_set_options SSL_set_options
#define SSL_CTX_get_mode SSL_CTX_get_mode
#define SSL_CTX_set_mode SSL_CTX_set_mode
#define SSL_get_mode SSL_get_mode
#define SSL_set_mode SSL_set_mode
#define SSL_CTX_get_read_ahead SSL_CTX_get_read_ahead
#define SSL_CTX_set_read_ahead SSL_CTX_set_read_ahead
#define SSL_CTX_sess_set_cache_size SSL_CTX_sess_set_cache_size
#define SSL_CTX_sess_get_cache_size SSL_CTX_sess_get_cache_size
#define SSL_CTX_set_session_cache_mode SSL_CTX_set_session_cache_mode
#define SSL_CTX_get_session_cache_mode SSL_CTX_get_session_cache_mode
#define SSL_CTX_get_max_cert_list SSL_CTX_get_max_cert_list
#define SSL_get_max_cert_list SSL_get_max_cert_list
#define SSL_CTX_set_max_cert_list SSL_CTX_set_max_cert_list
#define SSL_set_max_cert_list SSL_set_max_cert_list
#define SSL_CTX_set_max_send_fragment SSL_CTX_set_max_send_fragment
#define SSL_set_max_send_fragment SSL_set_max_send_fragment
#define SSL_CTX_set_tlsext_servername_callback \
    SSL_CTX_set_tlsext_servername_callback
#define SSL_CTX_set_tlsext_servername_arg SSL_CTX_set_tlsext_servername_arg
#define SSL_set_tlsext_host_name SSL_set_tlsext_host_name
#define SSL_CTX_set_tlsext_ticket_key_cb SSL_CTX_set_tlsext_ticket_key_cb
#define DTLSv1_get_timeout DTLSv1_get_timeout
#define DTLSv1_handle_timeout DTLSv1_handle_timeout
#define SSL_get_secure_renegotiation_support \
    SSL_get_secure_renegotiation_support
#define SSL_CTX_clear_options SSL_CTX_clear_options
#define SSL_clear_options SSL_clear_options
#define SSL_CTX_clear_mode SSL_CTX_clear_mode
#define SSL_clear_mode SSL_clear_mode
#define SSL_CTX_enable_tls_channel_id SSL_CTX_enable_tls_channel_id
#define SSL_enable_tls_channel_id SSL_enable_tls_channel_id
#define SSL_set1_tls_channel_id SSL_set1_tls_channel_id
#define SSL_CTX_set1_tls_channel_id SSL_CTX_set1_tls_channel_id
#define SSL_get_tls_channel_id SSL_get_tls_channel_id


#if defined(__cplusplus)
} /* extern C */
#endif


/* Library consumers assume these headers are included by ssl.h, but they depend
 * on ssl.h, so include them after all declarations.
 *
 * TODO(davidben): The separation between ssl.h and these version-specific
 * headers introduces circular dependencies and is inconsistent. The function
 * declarations should move to ssl.h. Many of the constants can probably be
 * pruned or unexported. */
#include <openssl/ssl2.h>
#include <openssl/ssl3.h>
#include <openssl/tls1.h> /* This is mostly sslv3 with a few tweaks */
#include <openssl/ssl23.h>
#include <openssl/srtp.h>  /* Support for the use_srtp extension */


/* BEGIN ERROR CODES */
/* The following lines are auto generated by the script make_errors.go. Any
 * changes made after this point may be overwritten when the script is next run.
 */
#define SSL_F_SSL_CTX_check_private_key 100
#define SSL_F_SSL_CTX_new 101
#define SSL_F_SSL_CTX_set_cipher_list 102
#define SSL_F_SSL_CTX_set_cipher_list_tls11 103
#define SSL_F_SSL_CTX_set_session_id_context 104
#define SSL_F_SSL_CTX_use_PrivateKey 105
#define SSL_F_SSL_CTX_use_PrivateKey_ASN1 106
#define SSL_F_SSL_CTX_use_PrivateKey_file 107
#define SSL_F_SSL_CTX_use_RSAPrivateKey 108
#define SSL_F_SSL_CTX_use_RSAPrivateKey_ASN1 109
#define SSL_F_SSL_CTX_use_RSAPrivateKey_file 110
#define SSL_F_SSL_CTX_use_certificate 111
#define SSL_F_SSL_CTX_use_certificate_ASN1 112
#define SSL_F_SSL_CTX_use_certificate_chain_file 113
#define SSL_F_SSL_CTX_use_certificate_file 114
#define SSL_F_SSL_CTX_use_psk_identity_hint 115
#define SSL_F_SSL_SESSION_new 116
#define SSL_F_SSL_SESSION_print_fp 117
#define SSL_F_SSL_SESSION_set1_id_context 118
#define SSL_F_SSL_SESSION_to_bytes_full 119
#define SSL_F_SSL_accept 120
#define SSL_F_SSL_add_dir_cert_subjects_to_stack 121
#define SSL_F_SSL_add_file_cert_subjects_to_stack 122
#define SSL_F_SSL_check_private_key 123
#define SSL_F_SSL_clear 124
#define SSL_F_SSL_connect 125
#define SSL_F_SSL_do_handshake 126
#define SSL_F_SSL_load_client_CA_file 127
#define SSL_F_SSL_new 128
#define SSL_F_SSL_peek 129
#define SSL_F_SSL_read 130
#define SSL_F_SSL_renegotiate 131
#define SSL_F_SSL_set_cipher_list 132
#define SSL_F_SSL_set_fd 133
#define SSL_F_SSL_set_rfd 134
#define SSL_F_SSL_set_session_id_context 135
#define SSL_F_SSL_set_wfd 136
#define SSL_F_SSL_shutdown 137
#define SSL_F_SSL_use_PrivateKey 138
#define SSL_F_SSL_use_PrivateKey_ASN1 139
#define SSL_F_SSL_use_PrivateKey_file 140
#define SSL_F_SSL_use_RSAPrivateKey 141
#define SSL_F_SSL_use_RSAPrivateKey_ASN1 142
#define SSL_F_SSL_use_RSAPrivateKey_file 143
#define SSL_F_SSL_use_certificate 144
#define SSL_F_SSL_use_certificate_ASN1 145
#define SSL_F_SSL_use_certificate_file 146
#define SSL_F_SSL_use_psk_identity_hint 147
#define SSL_F_SSL_write 148
#define SSL_F_d2i_SSL_SESSION 149
#define SSL_F_d2i_SSL_SESSION_get_octet_string 150
#define SSL_F_d2i_SSL_SESSION_get_string 151
#define SSL_F_do_ssl3_write 152
#define SSL_F_dtls1_accept 153
#define SSL_F_dtls1_buffer_record 154
#define SSL_F_dtls1_check_timeout_num 155
#define SSL_F_dtls1_connect 156
#define SSL_F_dtls1_do_write 157
#define SSL_F_dtls1_get_hello_verify 158
#define SSL_F_dtls1_get_message 159
#define SSL_F_dtls1_get_message_fragment 160
#define SSL_F_dtls1_preprocess_fragment 161
#define SSL_F_dtls1_process_record 162
#define SSL_F_dtls1_read_bytes 163
#define SSL_F_dtls1_send_hello_verify_request 164
#define SSL_F_dtls1_write_app_data 165
#define SSL_F_i2d_SSL_SESSION 166
#define SSL_F_ssl3_accept 167
#define SSL_F_ssl3_cert_verify_hash 169
#define SSL_F_ssl3_check_cert_and_algorithm 170
#define SSL_F_ssl3_connect 171
#define SSL_F_ssl3_ctrl 172
#define SSL_F_ssl3_ctx_ctrl 173
#define SSL_F_ssl3_digest_cached_records 174
#define SSL_F_ssl3_do_change_cipher_spec 175
#define SSL_F_ssl3_expect_change_cipher_spec 176
#define SSL_F_ssl3_get_cert_status 177
#define SSL_F_ssl3_get_cert_verify 178
#define SSL_F_ssl3_get_certificate_request 179
#define SSL_F_ssl3_get_channel_id 180
#define SSL_F_ssl3_get_client_certificate 181
#define SSL_F_ssl3_get_client_hello 182
#define SSL_F_ssl3_get_client_key_exchange 183
#define SSL_F_ssl3_get_finished 184
#define SSL_F_ssl3_get_initial_bytes 185
#define SSL_F_ssl3_get_message 186
#define SSL_F_ssl3_get_new_session_ticket 187
#define SSL_F_ssl3_get_next_proto 188
#define SSL_F_ssl3_get_record 189
#define SSL_F_ssl3_get_server_certificate 190
#define SSL_F_ssl3_get_server_done 191
#define SSL_F_ssl3_get_server_hello 192
#define SSL_F_ssl3_get_server_key_exchange 193
#define SSL_F_ssl3_get_v2_client_hello 194
#define SSL_F_ssl3_handshake_mac 195
#define SSL_F_ssl3_prf 196
#define SSL_F_ssl3_read_bytes 197
#define SSL_F_ssl3_read_n 198
#define SSL_F_ssl3_send_cert_verify 199
#define SSL_F_ssl3_send_certificate_request 200
#define SSL_F_ssl3_send_channel_id 201
#define SSL_F_ssl3_send_client_certificate 202
#define SSL_F_ssl3_send_client_hello 203
#define SSL_F_ssl3_send_client_key_exchange 204
#define SSL_F_ssl3_send_server_certificate 205
#define SSL_F_ssl3_send_server_hello 206
#define SSL_F_ssl3_send_server_key_exchange 207
#define SSL_F_ssl3_setup_read_buffer 208
#define SSL_F_ssl3_setup_write_buffer 209
#define SSL_F_ssl3_write_bytes 210
#define SSL_F_ssl3_write_pending 211
#define SSL_F_ssl_add_cert_chain 212
#define SSL_F_ssl_add_cert_to_buf 213
#define SSL_F_ssl_add_clienthello_renegotiate_ext 214
#define SSL_F_ssl_add_clienthello_tlsext 215
#define SSL_F_ssl_add_clienthello_use_srtp_ext 216
#define SSL_F_ssl_add_serverhello_renegotiate_ext 217
#define SSL_F_ssl_add_serverhello_tlsext 218
#define SSL_F_ssl_add_serverhello_use_srtp_ext 219
#define SSL_F_ssl_build_cert_chain 220
#define SSL_F_ssl_bytes_to_cipher_list 221
#define SSL_F_ssl_cert_dup 222
#define SSL_F_ssl_cert_inst 223
#define SSL_F_ssl_cert_new 224
#define SSL_F_ssl_check_serverhello_tlsext 225
#define SSL_F_ssl_check_srvr_ecc_cert_and_alg 226
#define SSL_F_ssl_cipher_process_rulestr 227
#define SSL_F_ssl_cipher_strength_sort 228
#define SSL_F_ssl_create_cipher_list 229
#define SSL_F_ssl_ctx_log_master_secret 230
#define SSL_F_ssl_ctx_log_rsa_client_key_exchange 231
#define SSL_F_ssl_ctx_make_profiles 232
#define SSL_F_ssl_get_new_session 233
#define SSL_F_ssl_get_prev_session 234
#define SSL_F_ssl_get_server_cert_index 235
#define SSL_F_ssl_get_sign_pkey 236
#define SSL_F_ssl_init_wbio_buffer 237
#define SSL_F_ssl_parse_clienthello_renegotiate_ext 238
#define SSL_F_ssl_parse_clienthello_tlsext 239
#define SSL_F_ssl_parse_clienthello_use_srtp_ext 240
#define SSL_F_ssl_parse_serverhello_renegotiate_ext 241
#define SSL_F_ssl_parse_serverhello_tlsext 242
#define SSL_F_ssl_parse_serverhello_use_srtp_ext 243
#define SSL_F_ssl_scan_clienthello_tlsext 244
#define SSL_F_ssl_scan_serverhello_tlsext 245
#define SSL_F_ssl_sess_cert_new 246
#define SSL_F_ssl_set_cert 247
#define SSL_F_ssl_set_pkey 248
#define SSL_F_ssl_verify_cert_chain 252
#define SSL_F_tls12_check_peer_sigalg 253
#define SSL_F_tls1_aead_ctx_init 254
#define SSL_F_tls1_cert_verify_mac 255
#define SSL_F_tls1_change_cipher_state 256
#define SSL_F_tls1_change_cipher_state_aead 257
#define SSL_F_tls1_check_duplicate_extensions 258
#define SSL_F_tls1_enc 259
#define SSL_F_tls1_export_keying_material 260
#define SSL_F_tls1_prf 261
#define SSL_F_tls1_setup_key_block 262
#define SSL_F_dtls1_get_buffered_message 263
#define SSL_F_dtls1_process_fragment 264
#define SSL_F_dtls1_hm_fragment_new 265
#define SSL_F_ssl3_seal_record 266
#define SSL_F_ssl3_record_sequence_update 267
#define SSL_F_SSL_CTX_set_tmp_dh 268
#define SSL_F_SSL_CTX_set_tmp_ecdh 269
#define SSL_F_SSL_set_tmp_dh 270
#define SSL_F_SSL_set_tmp_ecdh 271
#define SSL_F_SSL_CTX_set1_tls_channel_id 272
#define SSL_F_SSL_set1_tls_channel_id 273
#define SSL_F_SSL_set_tlsext_host_name 274
#define SSL_F_ssl3_output_cert_chain 275
#define SSL_F_SSL_AEAD_CTX_new 276
#define SSL_F_SSL_AEAD_CTX_open 277
#define SSL_F_SSL_AEAD_CTX_seal 278
#define SSL_F_dtls1_seal_record 279
#define SSL_R_APP_DATA_IN_HANDSHAKE 100
#define SSL_R_ATTEMPT_TO_REUSE_SESSION_IN_DIFFERENT_CONTEXT 101
#define SSL_R_BAD_ALERT 102
#define SSL_R_BAD_CHANGE_CIPHER_SPEC 103
#define SSL_R_BAD_DATA_RETURNED_BY_CALLBACK 104
#define SSL_R_BAD_DH_P_LENGTH 105
#define SSL_R_BAD_DIGEST_LENGTH 106
#define SSL_R_BAD_ECC_CERT 107
#define SSL_R_BAD_ECPOINT 108
#define SSL_R_BAD_HANDSHAKE_LENGTH 109
#define SSL_R_BAD_HANDSHAKE_RECORD 110
#define SSL_R_BAD_HELLO_REQUEST 111
#define SSL_R_BAD_LENGTH 112
#define SSL_R_BAD_PACKET_LENGTH 113
#define SSL_R_BAD_RSA_ENCRYPT 114
#define SSL_R_BAD_SIGNATURE 115
#define SSL_R_BAD_SRTP_MKI_VALUE 116
#define SSL_R_BAD_SRTP_PROTECTION_PROFILE_LIST 117
#define SSL_R_BAD_SSL_FILETYPE 118
#define SSL_R_BAD_WRITE_RETRY 119
#define SSL_R_BIO_NOT_SET 120
#define SSL_R_BN_LIB 121
#define SSL_R_CANNOT_SERIALIZE_PUBLIC_KEY 122
#define SSL_R_CA_DN_LENGTH_MISMATCH 123
#define SSL_R_CA_DN_TOO_LONG 124
#define SSL_R_CCS_RECEIVED_EARLY 125
#define SSL_R_CERTIFICATE_VERIFY_FAILED 126
#define SSL_R_CERT_CB_ERROR 127
#define SSL_R_CERT_LENGTH_MISMATCH 128
#define SSL_R_CHANNEL_ID_NOT_P256 129
#define SSL_R_CHANNEL_ID_SIGNATURE_INVALID 130
#define SSL_R_CIPHER_CODE_WRONG_LENGTH 131
#define SSL_R_CIPHER_OR_HASH_UNAVAILABLE 132
#define SSL_R_CLIENTHELLO_PARSE_FAILED 133
#define SSL_R_CLIENTHELLO_TLSEXT 134
#define SSL_R_CONNECTION_REJECTED 135
#define SSL_R_CONNECTION_TYPE_NOT_SET 136
#define SSL_R_COOKIE_MISMATCH 137
#define SSL_R_D2I_ECDSA_SIG 138
#define SSL_R_DATA_BETWEEN_CCS_AND_FINISHED 139
#define SSL_R_DATA_LENGTH_TOO_LONG 140
#define SSL_R_DECODE_ERROR 141
#define SSL_R_DECRYPTION_FAILED 142
#define SSL_R_DECRYPTION_FAILED_OR_BAD_RECORD_MAC 143
#define SSL_R_DH_PUBLIC_VALUE_LENGTH_IS_WRONG 144
#define SSL_R_DIGEST_CHECK_FAILED 145
#define SSL_R_DTLS_MESSAGE_TOO_BIG 146
#define SSL_R_ECC_CERT_NOT_FOR_SIGNING 147
#define SSL_R_EMPTY_SRTP_PROTECTION_PROFILE_LIST 148
#define SSL_R_ENCRYPTED_LENGTH_TOO_LONG 149
#define SSL_R_ERROR_IN_RECEIVED_CIPHER_LIST 150
#define SSL_R_EVP_DIGESTSIGNFINAL_FAILED 151
#define SSL_R_EVP_DIGESTSIGNINIT_FAILED 152
#define SSL_R_EXCESSIVE_MESSAGE_SIZE 153
#define SSL_R_EXTRA_DATA_IN_MESSAGE 154
#define SSL_R_GOT_A_FIN_BEFORE_A_CCS 155
#define SSL_R_GOT_CHANNEL_ID_BEFORE_A_CCS 156
#define SSL_R_GOT_NEXT_PROTO_BEFORE_A_CCS 157
#define SSL_R_GOT_NEXT_PROTO_WITHOUT_EXTENSION 158
#define SSL_R_HANDSHAKE_FAILURE_ON_CLIENT_HELLO 159
#define SSL_R_HANDSHAKE_RECORD_BEFORE_CCS 160
#define SSL_R_HTTPS_PROXY_REQUEST 161
#define SSL_R_HTTP_REQUEST 162
#define SSL_R_INAPPROPRIATE_FALLBACK 163
#define SSL_R_INVALID_COMMAND 164
#define SSL_R_INVALID_MESSAGE 165
#define SSL_R_INVALID_SSL_SESSION 166
#define SSL_R_INVALID_TICKET_KEYS_LENGTH 167
#define SSL_R_LENGTH_MISMATCH 168
#define SSL_R_LIBRARY_HAS_NO_CIPHERS 169
#define SSL_R_MISSING_DH_KEY 170
#define SSL_R_MISSING_ECDSA_SIGNING_CERT 171
#define SSL_R_MISSING_RSA_CERTIFICATE 172
#define SSL_R_MISSING_RSA_ENCRYPTING_CERT 173
#define SSL_R_MISSING_RSA_SIGNING_CERT 174
#define SSL_R_MISSING_TMP_DH_KEY 175
#define SSL_R_MISSING_TMP_ECDH_KEY 176
#define SSL_R_MIXED_SPECIAL_OPERATOR_WITH_GROUPS 177
#define SSL_R_MTU_TOO_SMALL 178
#define SSL_R_NESTED_GROUP 179
#define SSL_R_NO_CERTIFICATES_RETURNED 180
#define SSL_R_NO_CERTIFICATE_ASSIGNED 181
#define SSL_R_NO_CERTIFICATE_SET 182
#define SSL_R_NO_CIPHERS_AVAILABLE 183
#define SSL_R_NO_CIPHERS_PASSED 184
#define SSL_R_NO_CIPHERS_SPECIFIED 185
#define SSL_R_NO_CIPHER_MATCH 186
#define SSL_R_NO_COMPRESSION_SPECIFIED 187
#define SSL_R_NO_METHOD_SPECIFIED 188
#define SSL_R_NO_P256_SUPPORT 189
#define SSL_R_NO_PRIVATE_KEY_ASSIGNED 190
#define SSL_R_NO_RENEGOTIATION 191
#define SSL_R_NO_REQUIRED_DIGEST 192
#define SSL_R_NO_SHARED_CIPHER 193
#define SSL_R_NO_SHARED_SIGATURE_ALGORITHMS 194
#define SSL_R_NO_SRTP_PROFILES 195
#define SSL_R_NULL_SSL_CTX 196
#define SSL_R_NULL_SSL_METHOD_PASSED 197
#define SSL_R_OLD_SESSION_CIPHER_NOT_RETURNED 198
#define SSL_R_PACKET_LENGTH_TOO_LONG 199
#define SSL_R_PARSE_TLSEXT 200
#define SSL_R_PATH_TOO_LONG 201
#define SSL_R_PEER_DID_NOT_RETURN_A_CERTIFICATE 202
#define SSL_R_PEER_ERROR_UNSUPPORTED_CERTIFICATE_TYPE 203
#define SSL_R_PROTOCOL_IS_SHUTDOWN 204
#define SSL_R_PSK_IDENTITY_NOT_FOUND 205
#define SSL_R_PSK_NO_CLIENT_CB 206
#define SSL_R_PSK_NO_SERVER_CB 207
#define SSL_R_READ_BIO_NOT_SET 208
#define SSL_R_READ_TIMEOUT_EXPIRED 209
#define SSL_R_RECORD_LENGTH_MISMATCH 210
#define SSL_R_RECORD_TOO_LARGE 211
#define SSL_R_RENEGOTIATE_EXT_TOO_LONG 212
#define SSL_R_RENEGOTIATION_ENCODING_ERR 213
#define SSL_R_RENEGOTIATION_MISMATCH 214
#define SSL_R_REQUIRED_CIPHER_MISSING 215
#define SSL_R_SCSV_RECEIVED_WHEN_RENEGOTIATING 216
#define SSL_R_SERVERHELLO_TLSEXT 217
#define SSL_R_SESSION_ID_CONTEXT_UNINITIALIZED 218
#define SSL_R_SESSION_MAY_NOT_BE_CREATED 219
#define SSL_R_SIGNATURE_ALGORITHMS_ERROR 220
#define SSL_R_SRTP_COULD_NOT_ALLOCATE_PROFILES 221
#define SSL_R_SRTP_PROTECTION_PROFILE_LIST_TOO_LONG 222
#define SSL_R_SRTP_UNKNOWN_PROTECTION_PROFILE 223
#define SSL_R_SSL3_EXT_INVALID_SERVERNAME 224
#define SSL_R_SSL3_EXT_INVALID_SERVERNAME_TYPE 225
#define SSL_R_SSL_CTX_HAS_NO_DEFAULT_SSL_VERSION 226
#define SSL_R_SSL_HANDSHAKE_FAILURE 227
#define SSL_R_SSL_SESSION_ID_CALLBACK_FAILED 228
#define SSL_R_SSL_SESSION_ID_CONFLICT 229
#define SSL_R_SSL_SESSION_ID_CONTEXT_TOO_LONG 230
#define SSL_R_SSL_SESSION_ID_HAS_BAD_LENGTH 231
#define SSL_R_TLS_CLIENT_CERT_REQ_WITH_ANON_CIPHER 232
#define SSL_R_TLS_ILLEGAL_EXPORTER_LABEL 233
#define SSL_R_TLS_INVALID_ECPOINTFORMAT_LIST 234
#define SSL_R_TLS_PEER_DID_NOT_RESPOND_WITH_CERTIFICATE_LIST 235
#define SSL_R_TLS_RSA_ENCRYPTED_VALUE_LENGTH_IS_WRONG 236
#define SSL_R_TOO_MANY_EMPTY_FRAGMENTS 237
#define SSL_R_UNABLE_TO_FIND_ECDH_PARAMETERS 238
#define SSL_R_UNABLE_TO_FIND_PUBLIC_KEY_PARAMETERS 239
#define SSL_R_UNEXPECTED_GROUP_CLOSE 240
#define SSL_R_UNEXPECTED_MESSAGE 241
#define SSL_R_UNEXPECTED_OPERATOR_IN_GROUP 242
#define SSL_R_UNEXPECTED_RECORD 243
#define SSL_R_UNINITIALIZED 244
#define SSL_R_UNKNOWN_ALERT_TYPE 245
#define SSL_R_UNKNOWN_CERTIFICATE_TYPE 246
#define SSL_R_UNKNOWN_CIPHER_RETURNED 247
#define SSL_R_UNKNOWN_CIPHER_TYPE 248
#define SSL_R_UNKNOWN_DIGEST 249
#define SSL_R_UNKNOWN_KEY_EXCHANGE_TYPE 250
#define SSL_R_UNKNOWN_PROTOCOL 251
#define SSL_R_UNKNOWN_SSL_VERSION 252
#define SSL_R_UNKNOWN_STATE 253
#define SSL_R_UNPROCESSED_HANDSHAKE_DATA 254
#define SSL_R_UNSAFE_LEGACY_RENEGOTIATION_DISABLED 255
#define SSL_R_UNSUPPORTED_CIPHER 256
#define SSL_R_UNSUPPORTED_COMPRESSION_ALGORITHM 257
#define SSL_R_UNSUPPORTED_ELLIPTIC_CURVE 258
#define SSL_R_UNSUPPORTED_PROTOCOL 259
#define SSL_R_UNSUPPORTED_SSL_VERSION 260
#define SSL_R_USE_SRTP_NOT_NEGOTIATED 261
#define SSL_R_WRONG_CERTIFICATE_TYPE 262
#define SSL_R_WRONG_CIPHER_RETURNED 263
#define SSL_R_WRONG_CURVE 264
#define SSL_R_WRONG_MESSAGE_TYPE 265
#define SSL_R_WRONG_SIGNATURE_TYPE 266
#define SSL_R_WRONG_SSL_VERSION 267
#define SSL_R_WRONG_VERSION_NUMBER 268
#define SSL_R_X509_LIB 269
#define SSL_R_X509_VERIFICATION_SETUP_PROBLEMS 270
#define SSL_R_FRAGMENT_MISMATCH 271
#define SSL_R_BUFFER_TOO_SMALL 272
#define SSL_R_OLD_SESSION_VERSION_NOT_RETURNED 273
#define SSL_R_OUTPUT_ALIASES_INPUT 274
#define SSL_R_RESUMED_EMS_SESSION_WITHOUT_EMS_EXTENSION 275
#define SSL_R_EMS_STATE_INCONSISTENT 276
#define SSL_R_RESUMED_NON_EMS_SESSION_WITH_EMS_EXTENSION 277
#define SSL_R_SSLV3_ALERT_CLOSE_NOTIFY 1000
#define SSL_R_SSLV3_ALERT_UNEXPECTED_MESSAGE 1010
#define SSL_R_SSLV3_ALERT_BAD_RECORD_MAC 1020
#define SSL_R_TLSV1_ALERT_DECRYPTION_FAILED 1021
#define SSL_R_TLSV1_ALERT_RECORD_OVERFLOW 1022
#define SSL_R_SSLV3_ALERT_DECOMPRESSION_FAILURE 1030
#define SSL_R_SSLV3_ALERT_HANDSHAKE_FAILURE 1040
#define SSL_R_SSLV3_ALERT_NO_CERTIFICATE 1041
#define SSL_R_SSLV3_ALERT_BAD_CERTIFICATE 1042
#define SSL_R_SSLV3_ALERT_UNSUPPORTED_CERTIFICATE 1043
#define SSL_R_SSLV3_ALERT_CERTIFICATE_REVOKED 1044
#define SSL_R_SSLV3_ALERT_CERTIFICATE_EXPIRED 1045
#define SSL_R_SSLV3_ALERT_CERTIFICATE_UNKNOWN 1046
#define SSL_R_SSLV3_ALERT_ILLEGAL_PARAMETER 1047
#define SSL_R_TLSV1_ALERT_UNKNOWN_CA 1048
#define SSL_R_TLSV1_ALERT_ACCESS_DENIED 1049
#define SSL_R_TLSV1_ALERT_DECODE_ERROR 1050
#define SSL_R_TLSV1_ALERT_DECRYPT_ERROR 1051
#define SSL_R_TLSV1_ALERT_EXPORT_RESTRICTION 1060
#define SSL_R_TLSV1_ALERT_PROTOCOL_VERSION 1070
#define SSL_R_TLSV1_ALERT_INSUFFICIENT_SECURITY 1071
#define SSL_R_TLSV1_ALERT_INTERNAL_ERROR 1080
#define SSL_R_TLSV1_ALERT_INAPPROPRIATE_FALLBACK 1086
#define SSL_R_TLSV1_ALERT_USER_CANCELLED 1090
#define SSL_R_TLSV1_ALERT_NO_RENEGOTIATION 1100
#define SSL_R_TLSV1_UNSUPPORTED_EXTENSION 1110
#define SSL_R_TLSV1_CERTIFICATE_UNOBTAINABLE 1111
#define SSL_R_TLSV1_UNRECOGNIZED_NAME 1112
#define SSL_R_TLSV1_BAD_CERTIFICATE_STATUS_RESPONSE 1113
#define SSL_R_TLSV1_BAD_CERTIFICATE_HASH_VALUE 1114

#endif /* OPENSSL_HEADER_SSL_H */
