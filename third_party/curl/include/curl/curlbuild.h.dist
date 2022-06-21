#ifndef __CURL_CURLBUILD_H
#define __CURL_CURLBUILD_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2013, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at http://curl.haxx.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

/* ================================================================ */
/*               NOTES FOR CONFIGURE CAPABLE SYSTEMS                */
/* ================================================================ */

/*
 * NOTE 1:
 * -------
 *
 * See file include/curl/curlbuild.h.in, run configure, and forget
 * that this file exists it is only used for non-configure systems.
 * But you can keep reading if you want ;-)
 *
 */

/* ================================================================ */
/*                 NOTES FOR NON-CONFIGURE SYSTEMS                  */
/* ================================================================ */

/*
 * NOTE 1:
 * -------
 *
 * Nothing in this file is intended to be modified or adjusted by the
 * curl library user nor by the curl library builder.
 *
 * If you think that something actually needs to be changed, adjusted
 * or fixed in this file, then, report it on the libcurl development
 * mailing list: http://cool.haxx.se/mailman/listinfo/curl-library/
 *
 * Try to keep one section per platform, compiler and architecture,
 * otherwise, if an existing section is reused for a different one and
 * later on the original is adjusted, probably the piggybacking one can
 * be adversely changed.
 *
 * In order to differentiate between platforms/compilers/architectures
 * use only compiler built in predefined preprocessor symbols.
 *
 * This header file shall only export symbols which are 'curl' or 'CURL'
 * prefixed, otherwise public name space would be polluted.
 *
 * NOTE 2:
 * -------
 *
 * For any given platform/compiler curl_off_t must be typedef'ed to a
 * 64-bit wide signed integral data type. The width of this data type
 * must remain constant and independent of any possible large file
 * support settings.
 *
 * As an exception to the above, curl_off_t shall be typedef'ed to a
 * 32-bit wide signed integral data type if there is no 64-bit type.
 *
 * As a general rule, curl_off_t shall not be mapped to off_t. This
 * rule shall only be violated if off_t is the only 64-bit data type
 * available and the size of off_t is independent of large file support
 * settings. Keep your build on the safe side avoiding an off_t gating.
 * If you have a 64-bit off_t then take for sure that another 64-bit
 * data type exists, dig deeper and you will find it.
 *
 * NOTE 3:
 * -------
 *
 * Right now you might be staring at file include/curl/curlbuild.h.dist or
 * at file include/curl/curlbuild.h, this is due to the following reason:
 * file include/curl/curlbuild.h.dist is renamed to include/curl/curlbuild.h
 * when the libcurl source code distribution archive file is created.
 *
 * File include/curl/curlbuild.h.dist is not included in the distribution
 * archive. File include/curl/curlbuild.h is not present in the git tree.
 *
 * The distributed include/curl/curlbuild.h file is only intended to be used
 * on systems which can not run the also distributed configure script.
 *
 * On systems capable of running the configure script, the configure process
 * will overwrite the distributed include/curl/curlbuild.h file with one that
 * is suitable and specific to the library being configured and built, which
 * is generated from the include/curl/curlbuild.h.in template file.
 *
 * If you check out from git on a non-configure platform, you must run the
 * appropriate buildconf* script to set up curlbuild.h and other local files.
 *
 */

/* ================================================================ */
/*  DEFINITION OF THESE SYMBOLS SHALL NOT TAKE PLACE ANYWHERE ELSE  */
/* ================================================================ */

#ifdef CURL_SIZEOF_LONG
#  error "CURL_SIZEOF_LONG shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_SIZEOF_LONG_already_defined
#endif

#ifdef CURL_TYPEOF_CURL_SOCKLEN_T
#  error "CURL_TYPEOF_CURL_SOCKLEN_T shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_TYPEOF_CURL_SOCKLEN_T_already_defined
#endif

#ifdef CURL_SIZEOF_CURL_SOCKLEN_T
#  error "CURL_SIZEOF_CURL_SOCKLEN_T shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_SIZEOF_CURL_SOCKLEN_T_already_defined
#endif

#ifdef CURL_TYPEOF_CURL_OFF_T
#  error "CURL_TYPEOF_CURL_OFF_T shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_TYPEOF_CURL_OFF_T_already_defined
#endif

#ifdef CURL_FORMAT_CURL_OFF_T
#  error "CURL_FORMAT_CURL_OFF_T shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_FORMAT_CURL_OFF_T_already_defined
#endif

#ifdef CURL_FORMAT_CURL_OFF_TU
#  error "CURL_FORMAT_CURL_OFF_TU shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_FORMAT_CURL_OFF_TU_already_defined
#endif

#ifdef CURL_FORMAT_OFF_T
#  error "CURL_FORMAT_OFF_T shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_FORMAT_OFF_T_already_defined
#endif

#ifdef CURL_SIZEOF_CURL_OFF_T
#  error "CURL_SIZEOF_CURL_OFF_T shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_SIZEOF_CURL_OFF_T_already_defined
#endif

#ifdef CURL_SUFFIX_CURL_OFF_T
#  error "CURL_SUFFIX_CURL_OFF_T shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_SUFFIX_CURL_OFF_T_already_defined
#endif

#ifdef CURL_SUFFIX_CURL_OFF_TU
#  error "CURL_SUFFIX_CURL_OFF_TU shall not be defined except in curlbuild.h"
   Error Compilation_aborted_CURL_SUFFIX_CURL_OFF_TU_already_defined
#endif

/* ================================================================ */
/*    EXTERNAL INTERFACE SETTINGS FOR NON-CONFIGURE SYSTEMS ONLY    */
/* ================================================================ */

#if defined(__DJGPP__) || defined(__GO32__)
#  if defined(__DJGPP__) && (__DJGPP__ > 1)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_FORMAT_OFF_T          "%lld"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  else
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_FORMAT_OFF_T          "%ld"
#    define CURL_SIZEOF_CURL_OFF_T     4
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__SALFORDC__)
#  define CURL_SIZEOF_LONG           4
#  define CURL_TYPEOF_CURL_OFF_T     long
#  define CURL_FORMAT_CURL_OFF_T     "ld"
#  define CURL_FORMAT_CURL_OFF_TU    "lu"
#  define CURL_FORMAT_OFF_T          "%ld"
#  define CURL_SIZEOF_CURL_OFF_T     4
#  define CURL_SUFFIX_CURL_OFF_T     L
#  define CURL_SUFFIX_CURL_OFF_TU    UL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__BORLANDC__)
#  if (__BORLANDC__ < 0x520)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_FORMAT_OFF_T          "%ld"
#    define CURL_SIZEOF_CURL_OFF_T     4
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  else
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "I64d"
#    define CURL_FORMAT_CURL_OFF_TU    "I64u"
#    define CURL_FORMAT_OFF_T          "%I64d"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     i64
#    define CURL_SUFFIX_CURL_OFF_TU    ui64
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__TURBOC__)
#  define CURL_SIZEOF_LONG           4
#  define CURL_TYPEOF_CURL_OFF_T     long
#  define CURL_FORMAT_CURL_OFF_T     "ld"
#  define CURL_FORMAT_CURL_OFF_TU    "lu"
#  define CURL_FORMAT_OFF_T          "%ld"
#  define CURL_SIZEOF_CURL_OFF_T     4
#  define CURL_SUFFIX_CURL_OFF_T     L
#  define CURL_SUFFIX_CURL_OFF_TU    UL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__WATCOMC__)
#  if defined(__386__)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "I64d"
#    define CURL_FORMAT_CURL_OFF_TU    "I64u"
#    define CURL_FORMAT_OFF_T          "%I64d"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     i64
#    define CURL_SUFFIX_CURL_OFF_TU    ui64
#  else
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_FORMAT_OFF_T          "%ld"
#    define CURL_SIZEOF_CURL_OFF_T     4
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__POCC__)
#  if (__POCC__ < 280)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_FORMAT_OFF_T          "%ld"
#    define CURL_SIZEOF_CURL_OFF_T     4
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  elif defined(_MSC_VER)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "I64d"
#    define CURL_FORMAT_CURL_OFF_TU    "I64u"
#    define CURL_FORMAT_OFF_T          "%I64d"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     i64
#    define CURL_SUFFIX_CURL_OFF_TU    ui64
#  else
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_FORMAT_OFF_T          "%lld"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__LCC__)
#  define CURL_SIZEOF_LONG           4
#  define CURL_TYPEOF_CURL_OFF_T     long
#  define CURL_FORMAT_CURL_OFF_T     "ld"
#  define CURL_FORMAT_CURL_OFF_TU    "lu"
#  define CURL_FORMAT_OFF_T          "%ld"
#  define CURL_SIZEOF_CURL_OFF_T     4
#  define CURL_SUFFIX_CURL_OFF_T     L
#  define CURL_SUFFIX_CURL_OFF_TU    UL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__SYMBIAN32__)
#  if defined(__EABI__)  /* Treat all ARM compilers equally */
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_FORMAT_OFF_T          "%lld"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  elif defined(__CW32__)
#    pragma longlong on
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_FORMAT_OFF_T          "%lld"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  elif defined(__VC32__)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_FORMAT_OFF_T          "%lld"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T unsigned int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__MWERKS__)
#  define CURL_SIZEOF_LONG           4
#  define CURL_TYPEOF_CURL_OFF_T     long long
#  define CURL_FORMAT_CURL_OFF_T     "lld"
#  define CURL_FORMAT_CURL_OFF_TU    "llu"
#  define CURL_FORMAT_OFF_T          "%lld"
#  define CURL_SIZEOF_CURL_OFF_T     8
#  define CURL_SUFFIX_CURL_OFF_T     LL
#  define CURL_SUFFIX_CURL_OFF_TU    ULL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(_WIN32_WCE)
#  define CURL_SIZEOF_LONG           4
#  define CURL_TYPEOF_CURL_OFF_T     __int64
#  define CURL_FORMAT_CURL_OFF_T     "I64d"
#  define CURL_FORMAT_CURL_OFF_TU    "I64u"
#  define CURL_FORMAT_OFF_T          "%I64d"
#  define CURL_SIZEOF_CURL_OFF_T     8
#  define CURL_SUFFIX_CURL_OFF_T     i64
#  define CURL_SUFFIX_CURL_OFF_TU    ui64
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__MINGW32__)
#  define CURL_SIZEOF_LONG           4
#  define CURL_TYPEOF_CURL_OFF_T     long long
#  define CURL_FORMAT_CURL_OFF_T     "I64d"
#  define CURL_FORMAT_CURL_OFF_TU    "I64u"
#  define CURL_FORMAT_OFF_T          "%I64d"
#  define CURL_SIZEOF_CURL_OFF_T     8
#  define CURL_SUFFIX_CURL_OFF_T     LL
#  define CURL_SUFFIX_CURL_OFF_TU    ULL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__VMS)
#  if defined(__VAX)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_FORMAT_OFF_T          "%ld"
#    define CURL_SIZEOF_CURL_OFF_T     4
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  else
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_FORMAT_OFF_T          "%lld"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T unsigned int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

#elif defined(__OS400__)
#  if defined(__ILEC400__)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_FORMAT_OFF_T          "%lld"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#    define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#    define CURL_SIZEOF_CURL_SOCKLEN_T 4
#    define CURL_PULL_SYS_TYPES_H      1
#    define CURL_PULL_SYS_SOCKET_H     1
#  endif

#elif defined(__MVS__)
#  if defined(__IBMC__) || defined(__IBMCPP__)
#    if defined(_ILP32)
#      define CURL_SIZEOF_LONG           4
#    elif defined(_LP64)
#      define CURL_SIZEOF_LONG           8
#    endif
#    if defined(_LONG_LONG)
#      define CURL_TYPEOF_CURL_OFF_T     long long
#      define CURL_FORMAT_CURL_OFF_T     "lld"
#      define CURL_FORMAT_CURL_OFF_TU    "llu"
#      define CURL_FORMAT_OFF_T          "%lld"
#      define CURL_SIZEOF_CURL_OFF_T     8
#      define CURL_SUFFIX_CURL_OFF_T     LL
#      define CURL_SUFFIX_CURL_OFF_TU    ULL
#    elif defined(_LP64)
#      define CURL_TYPEOF_CURL_OFF_T     long
#      define CURL_FORMAT_CURL_OFF_T     "ld"
#      define CURL_FORMAT_CURL_OFF_TU    "lu"
#      define CURL_FORMAT_OFF_T          "%ld"
#      define CURL_SIZEOF_CURL_OFF_T     8
#      define CURL_SUFFIX_CURL_OFF_T     L
#      define CURL_SUFFIX_CURL_OFF_TU    UL
#    else
#      define CURL_TYPEOF_CURL_OFF_T     long
#      define CURL_FORMAT_CURL_OFF_T     "ld"
#      define CURL_FORMAT_CURL_OFF_TU    "lu"
#      define CURL_FORMAT_OFF_T          "%ld"
#      define CURL_SIZEOF_CURL_OFF_T     4
#      define CURL_SUFFIX_CURL_OFF_T     L
#      define CURL_SUFFIX_CURL_OFF_TU    UL
#    endif
#    define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#    define CURL_SIZEOF_CURL_SOCKLEN_T 4
#    define CURL_PULL_SYS_TYPES_H      1
#    define CURL_PULL_SYS_SOCKET_H     1
#  endif

#elif defined(__370__)
#  if defined(__IBMC__) || defined(__IBMCPP__)
#    if defined(_ILP32)
#      define CURL_SIZEOF_LONG           4
#    elif defined(_LP64)
#      define CURL_SIZEOF_LONG           8
#    endif
#    if defined(_LONG_LONG)
#      define CURL_TYPEOF_CURL_OFF_T     long long
#      define CURL_FORMAT_CURL_OFF_T     "lld"
#      define CURL_FORMAT_CURL_OFF_TU    "llu"
#      define CURL_FORMAT_OFF_T          "%lld"
#      define CURL_SIZEOF_CURL_OFF_T     8
#      define CURL_SUFFIX_CURL_OFF_T     LL
#      define CURL_SUFFIX_CURL_OFF_TU    ULL
#    elif defined(_LP64)
#      define CURL_TYPEOF_CURL_OFF_T     long
#      define CURL_FORMAT_CURL_OFF_T     "ld"
#      define CURL_FORMAT_CURL_OFF_TU    "lu"
#      define CURL_FORMAT_OFF_T          "%ld"
#      define CURL_SIZEOF_CURL_OFF_T     8
#      define CURL_SUFFIX_CURL_OFF_T     L
#      define CURL_SUFFIX_CURL_OFF_TU    UL
#    else
#      define CURL_TYPEOF_CURL_OFF_T     long
#      define CURL_FORMAT_CURL_OFF_T     "ld"
#      define CURL_FORMAT_CURL_OFF_TU    "lu"
#      define CURL_FORMAT_OFF_T          "%ld"
#      define CURL_SIZEOF_CURL_OFF_T     4
#      define CURL_SUFFIX_CURL_OFF_T     L
#      define CURL_SUFFIX_CURL_OFF_TU    UL
#    endif
#    define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#    define CURL_SIZEOF_CURL_SOCKLEN_T 4
#    define CURL_PULL_SYS_TYPES_H      1
#    define CURL_PULL_SYS_SOCKET_H     1
#  endif

#elif defined(TPF)
#  define CURL_SIZEOF_LONG           8
#  define CURL_TYPEOF_CURL_OFF_T     long
#  define CURL_FORMAT_CURL_OFF_T     "ld"
#  define CURL_FORMAT_CURL_OFF_TU    "lu"
#  define CURL_FORMAT_OFF_T          "%ld"
#  define CURL_SIZEOF_CURL_OFF_T     8
#  define CURL_SUFFIX_CURL_OFF_T     L
#  define CURL_SUFFIX_CURL_OFF_TU    UL
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

/* ===================================== */
/*    KEEP MSVC THE PENULTIMATE ENTRY    */
/* ===================================== */

#elif defined(_MSC_VER)
#  if (_MSC_VER >= 900) && (_INTEGRAL_MAX_BITS >= 64)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     __int64
#    define CURL_FORMAT_CURL_OFF_T     "I64d"
#    define CURL_FORMAT_CURL_OFF_TU    "I64u"
#    define CURL_FORMAT_OFF_T          "%I64d"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     i64
#    define CURL_SUFFIX_CURL_OFF_TU    ui64
#  else
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_FORMAT_OFF_T          "%ld"
#    define CURL_SIZEOF_CURL_OFF_T     4
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T int
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4

/* ===================================== */
/*    KEEP GENERIC GCC THE LAST ENTRY    */
/* ===================================== */

#elif defined(__GNUC__)
#  if defined(__ILP32__) || \
      defined(__i386__) || defined(__ppc__) || defined(__arm__) || defined(__sparc__)
#    define CURL_SIZEOF_LONG           4
#    define CURL_TYPEOF_CURL_OFF_T     long long
#    define CURL_FORMAT_CURL_OFF_T     "lld"
#    define CURL_FORMAT_CURL_OFF_TU    "llu"
#    define CURL_FORMAT_OFF_T          "%lld"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     LL
#    define CURL_SUFFIX_CURL_OFF_TU    ULL
#  elif defined(__LP64__) || \
        defined(__x86_64__) || defined(__ppc64__) || defined(__sparc64__)
#    define CURL_SIZEOF_LONG           8
#    define CURL_TYPEOF_CURL_OFF_T     long
#    define CURL_FORMAT_CURL_OFF_T     "ld"
#    define CURL_FORMAT_CURL_OFF_TU    "lu"
#    define CURL_FORMAT_OFF_T          "%ld"
#    define CURL_SIZEOF_CURL_OFF_T     8
#    define CURL_SUFFIX_CURL_OFF_T     L
#    define CURL_SUFFIX_CURL_OFF_TU    UL
#  endif
#  define CURL_TYPEOF_CURL_SOCKLEN_T socklen_t
#  define CURL_SIZEOF_CURL_SOCKLEN_T 4
#  define CURL_PULL_SYS_TYPES_H      1
#  define CURL_PULL_SYS_SOCKET_H     1

#else
#  error "Unknown non-configure build target!"
   Error Compilation_aborted_Unknown_non_configure_build_target
#endif

/* CURL_PULL_SYS_TYPES_H is defined above when inclusion of header file  */
/* sys/types.h is required here to properly make type definitions below. */
#ifdef CURL_PULL_SYS_TYPES_H
#  include <sys/types.h>
#endif

/* CURL_PULL_SYS_SOCKET_H is defined above when inclusion of header file  */
/* sys/socket.h is required here to properly make type definitions below. */
#ifdef CURL_PULL_SYS_SOCKET_H
#  include <sys/socket.h>
#endif

/* Data type definition of curl_socklen_t. */

#ifdef CURL_TYPEOF_CURL_SOCKLEN_T
  typedef CURL_TYPEOF_CURL_SOCKLEN_T curl_socklen_t;
#endif

/* Data type definition of curl_off_t. */

#ifdef CURL_TYPEOF_CURL_OFF_T
  typedef CURL_TYPEOF_CURL_OFF_T curl_off_t;
#endif

#endif /* __CURL_CURLBUILD_H */
