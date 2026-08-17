/**
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef AEESTDDEF_H
#define AEESTDDEF_H
/*
=======================================================================

FILE:         AEEStdDef.h

DESCRIPTION:  definition of basic types, constants,
                 preprocessor macros

=======================================================================
*/

#include <stdint.h>

#if defined(COMDEF_H) /* guards against a known re-definer */
#define _BOOLEAN_DEFINED
#define _UINT32_DEFINED
#define _UINT16_DEFINED
#define _UINT8_DEFINED
#define _INT32_DEFINED
#define _INT16_DEFINED
#define _INT8_DEFINED
#define _UINT64_DEFINED
#define _INT64_DEFINED
#define _BYTE_DEFINED
#endif /* #if !defined(COMDEF_H) */

/* -----------------------------------------------------------------------
** Standard Types
** ----------------------------------------------------------------------- */

/* The following definitions are the same accross platforms.  This first
** group are the sanctioned types.
*/
#ifndef _BOOLEAN_DEFINED
typedef  unsigned char      boolean;     /* Boolean value type. */
#define _BOOLEAN_DEFINED
#endif

#ifndef _UINT32_DEFINED
typedef  uint32_t           uint32;      /* Unsigned 32 bit value */
#define _UINT32_DEFINED
#endif

#ifndef _UINT16_DEFINED
typedef  unsigned short     uint16;      /* Unsigned 16 bit value */
#define _UINT16_DEFINED
#endif

#ifndef _UINT8_DEFINED
typedef  unsigned char      uint8;       /* Unsigned 8  bit value */
#define _UINT8_DEFINED
#endif

#ifndef _INT32_DEFINED
typedef  int32_t            int32;       /* Signed 32 bit value */
#define _INT32_DEFINED
#endif

#ifndef _INT16_DEFINED
typedef  signed short       int16;       /* Signed 16 bit value */
#define _INT16_DEFINED
#endif

#ifndef _INT8_DEFINED
typedef  signed char        int8;        /* Signed 8  bit value */
#define _INT8_DEFINED
#endif

#if defined(__GNUC__)
#define __int64 long long
#endif

#ifndef _UINT64_DEFINED
typedef  unsigned __int64   uint64;      /* Unsigned 64 bit value */
#define _UINT64_DEFINED
#endif

#ifndef _INT64_DEFINED
typedef  __int64            int64;       /* Signed 64 bit value */
#define _INT64_DEFINED
#endif

#ifndef _BYTE_DEFINED
typedef  unsigned char      byte;        /* byte type */
#define  _BYTE_DEFINED
#endif


#ifndef _AEEUID_DEFINED
typedef uint32             AEEUID;
#define _AEEUID_DEFINED
#endif

#ifndef _AEEIID_DEFINED
typedef uint32             AEEIID;
#define _AEEIID_DEFINED
#endif

#ifndef _AEECLSID_DEFINED
typedef uint32             AEECLSID;
#define _AEECLSID_DEFINED
#endif

#ifndef _AEEPRIVID_DEFINED
typedef uint32             AEEPRIVID;
#define _AEEPRIVID_DEFINED
#endif

#ifndef _AECHAR_DEFINED
typedef uint16             AECHAR;
#define _AECHAR_DEFINED
#endif

#ifndef _AEERESULT_DEFINED
typedef int                AEEResult;
#define _AEERESULT_DEFINED
#endif

/* -----------------------------------------------------------------------
** Function Calling Conventions
** ----------------------------------------------------------------------- */

#ifndef CDECL
#ifdef _MSC_VER
#define CDECL __cdecl
#else
#define CDECL
#endif /* _MSC_VER */
#endif /* CDECL */

/* -----------------------------------------------------------------------
** Constants
** ----------------------------------------------------------------------- */

#ifndef TRUE
#define TRUE   1   /* Boolean true value. */
#endif

#ifndef FALSE
#define FALSE  0   /* Boolean false value. */
#endif

#ifndef NULL
#define NULL  0
#endif

#ifndef MIN_INT8
#define MIN_INT8 -128
#endif
#ifndef MIN_INT16
#define MIN_INT16 -32768
#endif
#ifndef MIN_INT32
#define MIN_INT32 (~0x7fffffff)   /* -2147483648 is unsigned */
#endif
#ifndef MIN_INT64
#define MIN_INT64 (~0x7fffffffffffffffLL) /* -9223372036854775808 is unsigned */
#endif

#ifndef MAX_INT8
#define MAX_INT8 127
#endif
#ifndef MAX_INT16
#define MAX_INT16 32767
#endif
#ifndef MAX_INT32
#define MAX_INT32 2147483647
#endif
#ifndef MAX_INT64
#define MAX_INT64 9223372036854775807LL
#endif

#ifndef MAX_UINT8
#define MAX_UINT8 255
#endif
#ifndef MAX_UINT16
#define MAX_UINT16 65535
#endif
#ifndef MAX_UINT32
#define MAX_UINT32 4294967295u
#endif
#ifndef MAX_UINT64
#define MAX_UINT64 18446744073709551615uLL
#endif

#ifndef MIN_AECHAR
#define MIN_AECHAR 0
#endif
#ifndef MAX_AECHAR
#define MAX_AECHAR 65535
#endif


/* -----------------------------------------------------------------------
** Preprocessor helpers
** ----------------------------------------------------------------------- */
#define __STR__(x) #x
#define __TOSTR__(x) __STR__(x)
#define __FILE_LINE__ __FILE__ ":" __TOSTR__(__LINE__)

/* -----------------------------------------------------------------------
** Types for code generated from IDL
** ----------------------------------------------------------------------- */

#ifndef __QIDL_WCHAR_T_DEFINED__
#define __QIDL_WCHAR_T_DEFINED__
typedef uint16 _wchar_t;
#endif

/* __STRING_OBJECT__ will be deprecated in the future */
#if !defined(__QIDL_STRING_OBJECT_DEFINED__) && !defined(__STRING_OBJECT__)
#define __QIDL_STRING_OBJECT_DEFINED__
#define __STRING_OBJECT__
typedef struct _cstring_s {
   char* data;
   int dataLen;
   int dataLenReq;
} _cstring_t;

typedef struct _wstring_s {
   _wchar_t* data;
   int dataLen;
   int dataLenReq;
} _wstring_t;
#endif /* __QIDL_STRING_OBJECT_DEFINED__ */

/*
=======================================================================
  DATA STRUCTURES DOCUMENTATION
=======================================================================

boolean

Description:
    This type is used to express boolean values (TRUE or FALSE).

Definition:
    typedef  unsigned char boolean

See Also:
   byte
   int8
   int16
   int32
   int64
   uint32
   uint16
   uint8
   uint64

=======================================================================

uint32

Description:
    This is a 32-bit unsigned integer.

Definition:
    typedef  uint32_t uint32

See Also:
   boolean
   byte
   int8
   int16
   int32
   int64
   uint8
   uint16
   uint64

=======================================================================

uint16

Description:
    This is a 16-bit unsigned integer.

Definition:
    typedef  unsigned short     uint16

See Also:
   boolean
   byte
   int8
   int16
   int32
   int64
   uint8
   uint32
   uint64

=======================================================================

uint8

Description:
    This is an 8-bit unsigned integer.

Definition:
    typedef  unsigned char      uint8

See Also:
   boolean
   byte
   int8
   int16
   int32
   int64
   uint16
   uint32
   uint64

=======================================================================

int32

Description:
    This is a 32-bit signed integer.

Definition:
    typedef  int32_t int32

See Also:
   boolean
   byte
   int8
   int16
   int64
   uint8
   uint16
   uint32
   uint64

=======================================================================

int16

Description:
    This is a 16-bit signed integer.

Definition:
    typedef  signed short       int16

See Also:
   boolean
   byte
   int8
   int32
   int64
   uint8
   uint16
   uint32
   uint64

=======================================================================

int8

Description:
    This is an 8-bit signed integer.

Definition:
    typedef  signed char        int8

See Also:
   boolean
   byte
   int16
   int32
   int64
   uint8
   uint16
   uint32
   uint64

=======================================================================

uint64

Description:
    This is a 64-bit unsigned integer.

Definition:
    typedef  unsigned __int64   uint64

See Also:
   boolean
   byte
   int8
   int16
   int32
   int64
   uint8
   uint16
   uint32

=======================================================================

int64

Description:
    This is a 64-bit signed integer.

Definition:
    typedef  __int64            int64

See Also:
   boolean
   byte
   int8
   int16
   int32
   uint8
   uint16
   uint32
   uint64

=======================================================================

byte

Description:
    This is a byte.

Definition:
    typedef  unsigned char     byte

See Also:
   boolean
   int8
   int16
   int32
   int64
   uint8
   uint16
   uint32
   uint64

=======================================================================

AEEUID

Description:
   This is a BREW unique ID.  Used to express unique types, interfaces, classes
     groups and privileges.  The BREW ClassID Generator generates
     unique IDs that can be used anywhere you need a new AEEIID, AEECLSID,
     or AEEPRIVID.

Definition:
    typedef uint32             AEEUID

See Also:
   AECHAR
   AEECLSID
   AEEIID
   AEEPRIVID
   AEEResult

=======================================================================

AEEIID

Description:
   This is an interface ID type, used to denote a BREW interface. It is a special case
     of AEEUID.

Definition:
    typedef uint32             AEEIID

See Also:
   AECHAR
   AEECLSID
   AEEPRIVID
   AEEResult
   AEEUID

=======================================================================

AEECLSID

Description:
   This is a classe ID type, used to denote a BREW class. It is a special case
     of AEEUID.

Definition:
    typedef uint32             AEECLSID

See Also:
   AECHAR
   AEECLSIDs
   AEEIID
   AEEPRIVID
   AEEResult
   AEEUID

=======================================================================

AEEPRIVID

Description:
   This is a privilege ID type, used to express a privilege.  It is a special case
     of AEEUID.

Definition:
    typedef uint32             AEEPRIVID

See Also:
   AECHAR
   AEECLSID
   AEEIID
   AEEResult
   AEEUID

=======================================================================

AECHAR

Description:
   This is a 16-bit character type.

Definition:
   typedef uint16             AECHAR

See Also:
   AEEPRIVID
   AEECLSID
   AEEIID
   AEEResult
   AEEUID

=======================================================================

AEEResult

Description:
   This is the standard result type.

Definition:
   typedef int                AEEResult

See Also:
   AECHAR
   AEEPRIVID
   AEECLSID
   AEEIID
   AEEUID

=======================================================================

_wchar_t

Description:
   This is a 16-bit character type corresponding to the IDL 'wchar'
   type.

Definition:
   typedef uint16             _wchar_t

See Also:
   _cstring_t
   _wstring_t

=======================================================================

_cstring_t

Description:
   This structure is used to represent an IDL string when used inside a
   sequence or union.

Definition:
   typedef struct _cstring_s {
      char* data;
      int dataLen;
      int dataLenReq;
   } _cstring_t;

Members:
   data       : A pointer to the NULL-terminated string.
   dataLen    : The size, in chars, of the buffer pointed to by 'data',
                including the NULL terminator.  This member is only used
                when the structure is part of an rout or inrout
                parameter, but must be supplied by the caller as an
                input in these cases.
   dataLenReq : The size that would have been required to store the
                entire result string.  This member is only used when the
                structure is part of an rout or inrout parameter, when
                it is an output value set by the callee.  The length of
                the returned string (including the NULL terminator)
                after a call is the minimum of dataLen and dataLenReq.

See Also:
   _wchar_t
   _wstring_t

=======================================================================

_wstring_t

Description:
   This structure is used to represent an IDL wstring when used inside a
   sequence or union.

Definition:
   typedef struct _wstring_s {
      _wchar_t* data;
      int dataLen;
      int dataLenReq;
   } _wstring_t;

Members:
   data       : A pointer to the NULL-terminated wide string.
   dataLen    : The size, in 16-bit characters, of the buffer pointed to
                by 'data', including the NULL terminator.  This member
                is only used when the structure is part of an rout or
                inrout parameter, but must be supplied by the caller as
                an input in these cases.
   dataLenReq : The number of 16-bit characters that would have been
                required to store the entire result string.  This member
                is only used when the structure is part of an rout or
                inrout parameter, when it is an output value set by the
                callee.  The length of the returned wstring (including
                the NULL terminator) after a call is the minimum of
                dataLen and dataLenReq.

See Also:
   _cstring_t
   _wchar_t

=======================================================================
CONSTANTS DOCUMENTATION
=======================================================================

TRUE

Description:
   TRUE is the boolean "true."

Definition:

   #define TRUE    1

See Also:
   FALSE
   NULL

=======================================================================

FALSE

Description:
   FALSE is the boolean "false."

Definition:

   #define FALSE   0

See Also:
   NULL
   TRUE

=======================================================================

NULL

Description:
   NULL is the null value, usually used to test a pointer.

Definition:

   #define NULL   0

See Also:
   FALSE
   TRUE

=======================================================================

MIN_INT8

Description:
  MIN_INT8 is the minimum signed 8-bit integer value.

Definition:

   #define MIN_INT8  -128

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MIN_INT16

Description:
  MIN_INT16 is the minimum signed 16-bit integer value

Definition:

   #define MIN_INT16  -32768

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT32
   MIN_INT64

=======================================================================

MIN_INT32

Description:
   MIN_INT32 is the minimum signed 32-bit integer value.

Definition:

   #define MIN_INT32 (~0x7fffffff)

Comments:
    Brew MP uses (~0x7fffffff), because -2147483648 is treated as unsigned by compilers.

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT64

=======================================================================

MIN_INT64

Description:
   MIN_INT64 is the minimum signed 64-bit integer value.

Definition:

   #define MIN_INT64 (~0x7fffffffffffffffll)

Comments:
    Brew MP uses (~0x7fffffffffffffffll), because -9223372036854775808 is
       treated as unsigned by compilers.

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT32

=======================================================================

MAX_INT8

Description:
   MAX_INT8 is the maximum signed 8-bit integer value

Definition:

   #define MAX_INT8  127

See Also:
   MAX_AECHAR
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MAX_INT16

Description:
   MAX_INT16 is the maximum signed 16-bit integer value.

Definition:

   #define MAX_INT16  32767

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MAX_INT32

Description:
   MAX_INT32 is the maximum signed 32-bit integer value.

Definition:

   #define MAX_INT32  2147483647

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MAX_INT64

Description:
   MAX_INT64 is the maximum signed 64-bit integer value.

Definition:

   #define MAX_INT64  9223372036854775807ll

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MAX_UINT8

Description:
   MAX_UINT8 is the maximum unsigned 8-bit integer value.

Definition:

   #define MAX_UINT8  255

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MAX_UINT16

Description:
   MAX_UINT16 is the maximum unsigned 16-bit integer value.

Definition:

   #define MAX_UINT16  65535

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT32
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MAX_UINT32

Description:
   MAX_UINT32 is the maximum unsigned 32-bit integer value.

Definition:

   #define MAX_UINT32  4294967295u

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT64
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MAX_UINT64

Description:
   MAX_UINT64 is the maximum unsigned 64-bit integer value.

Definition:

   #define MAX_UINT64  18446744073709551615ull

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MIN_AECHAR
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MIN_AECHAR

Description:
  MIN_AECHAR is the minimum AECHAR value.

Definition:

   #define MIN_AECHAR  0

See Also:
   MAX_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================

MAX_AECHAR

Description:
  MAX_AECHAR is the maximum AECHAR value.

Definition:

   #define MAX_AECHAR  65535

See Also:
   MIN_AECHAR
   MAX_INT8
   MAX_INT16
   MAX_INT32
   MAX_INT64
   MAX_UINT8
   MAX_UINT16
   MAX_UINT32
   MAX_UINT64
   MIN_INT8
   MIN_INT16
   MIN_INT32
   MIN_INT64

=======================================================================
MACROS DOCUMENTATION
=======================================================================

__STR__()

Description:
   The __STR__() makes a token into a string, used to string-ize things already
      defined.

Definition:

   #define __STR__(x)  #x

Parameters:
    x: token to make into a string

See Also:
   __TOSTR__()
   __FILE_LINE__

=======================================================================

__TOSTR__()

Description:
   The __TOSTR__() makes a token's value into a string, used to string-ize things
    already defined, used with __STR__.

Definition:

   #define __TOSTR__(x)  __STR__(x)

Parameters:
    x: token to evaluate and string-ize

Evaluation Value:
    the token's replacement as a string

See Also:
   __FILE_LINE__
   __STR__()

=======================================================================

__FILE_LINE__

Description:
   The compiler's __FILE__ (a string) and __LINE__ (an integer) are pasted
     together as a single string with a ":" between.

Definition:

   #define __FILE_LINE__ __FILE__ ":"  __TOSTR__(__LINE__)

Evaluation Value:
   __FILE__":""__LINE__"

See Also:
   __STR__()
   __TOSTR__()

=======================================================================
*/

#endif /* #ifndef AEESTDDEF_H */

