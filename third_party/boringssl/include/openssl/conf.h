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

#ifndef OPENSSL_HEADER_CONF_H
#define OPENSSL_HEADER_CONF_H

#include <openssl/base.h>

#include <openssl/stack.h>
#include <openssl/lhash.h>

#if defined(__cplusplus)
extern "C" {
#endif


/* Config files look like:
 *
 *   # Comment
 *
 *   # This key is in the default section.
 *   key=value
 *
 *   [section_name]
 *   key2=value2
 *
 * Config files are representated by a |CONF|. */

struct conf_value_st {
  char *section;
  char *name;
  char *value;
};

struct conf_st {
  LHASH_OF(CONF_VALUE) *data;
};


/* NCONF_new returns a fresh, empty |CONF|, or NULL on error. The |method|
 * argument must be NULL. */
CONF *NCONF_new(void *method);

/* NCONF_free frees all the data owned by |conf| and then |conf| itself. */
void NCONF_free(CONF *conf);

/* NCONF_load parses the file named |filename| and adds the values found to
 * |conf|. It returns one on success and zero on error. In the event of an
 * error, if |out_error_line| is not NULL, |*out_error_line| is set to the
 * number of the line that contained the error. */
int NCONF_load(CONF *conf, const char *filename, long *out_error_line);

/* NCONF_load_bio acts like |NCONF_load| but reads from |bio| rather than from
 * a named file. */
int NCONF_load_bio(CONF *conf, BIO *bio, long *out_error_line);

/* NCONF_get_section returns a stack of values for a given section in |conf|.
 * If |section| is NULL, the default section is returned. It returns NULL on
 * error. */
STACK_OF(CONF_VALUE) *NCONF_get_section(const CONF *conf, const char *section);

/* NCONF_get_string returns the value of the key |name|, in section |section|.
 * The |section| argument may be NULL to indicate the default section. It
 * returns the value or NULL on error. */
const char *NCONF_get_string(const CONF *conf, const char *section,
                             const char *name);


/* Utility functions */

/* CONF_parse_list takes a list separated by 'sep' and calls |list_cb| giving
 * the start and length of each member, optionally stripping leading and
 * trailing whitespace. This can be used to parse comma separated lists for
 * example. If |list_cb| returns <= 0, then the iteration is halted and that
 * value is returned immediately. Otherwise it returns one. Note that |list_cb|
 * may be called on an empty member. */
int CONF_parse_list(const char *list, char sep, int remove_whitespace,
                    int (*list_cb)(const char *elem, int len, void *usr),
                    void *arg);

#if defined(__cplusplus)
}  /* extern C */
#endif

#define CONF_F_CONF_parse_list 100
#define CONF_F_NCONF_load 101
#define CONF_F_def_load_bio 102
#define CONF_F_str_copy 103
#define CONF_R_LIST_CANNOT_BE_NULL 100
#define CONF_R_MISSING_CLOSE_SQUARE_BRACKET 101
#define CONF_R_MISSING_EQUAL_SIGN 102
#define CONF_R_NO_CLOSE_BRACE 103
#define CONF_R_UNABLE_TO_CREATE_NEW_SECTION 104
#define CONF_R_VARIABLE_HAS_NO_VALUE 105

#endif  /* OPENSSL_HEADER_THREAD_H */
