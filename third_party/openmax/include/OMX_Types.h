/*
 * Copyright (c) 2008 The Khronos Group Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

/** OMX_Types.h - OpenMax IL version 1.1.2
 *  The OMX_Types header file contains the primitive type definitions used by
 *  the core, the application and the component.  This file may need to be
 *  modified to be used on systems that do not have "char" set to 8 bits, 
 *  "short" set to 16 bits and "long" set to 32 bits.
 */

#ifndef OMX_Types_h
#define OMX_Types_h

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** The OMX_API and OMX_APIENTRY are platform specific definitions used
 *  to declare OMX function prototypes.  They are modified to meet the
 *  requirements for a particular platform */
#ifdef __SYMBIAN32__
#   ifdef __OMX_EXPORTS
#       define OMX_API __declspec(dllexport)
#   else
#       ifdef _WIN32
#           define OMX_API __declspec(dllexport)
#       else
#           define OMX_API __declspec(dllimport)
#       endif
#   endif
#else
#   ifdef _WIN32
#      ifdef __OMX_EXPORTS
#          define OMX_API __declspec(dllexport)
#      else
#          define OMX_API __declspec(dllimport)
#      endif
#   else
#      ifdef __OMX_EXPORTS
#          define OMX_API
#      else
#          define OMX_API extern
#      endif
#   endif
#endif

#ifndef OMX_APIENTRY
#define OMX_APIENTRY
#endif

/** OMX_IN is used to identify inputs to an OMX function.  This designation
    will also be used in the case of a pointer that points to a parameter
    that is used as an output. */
#ifndef OMX_IN
#define OMX_IN
#endif

/** OMX_OUT is used to identify outputs from an OMX function.  This
    designation will also be used in the case of a pointer that points
    to a parameter that is used as an input. */
#ifndef OMX_OUT
#define OMX_OUT
#endif


/** OMX_INOUT is used to identify parameters that may be either inputs or
    outputs from an OMX function at the same time.  This designation will
    also be used in the case of a pointer that  points to a parameter that
    is used both as an input and an output. */
#ifndef OMX_INOUT
#define OMX_INOUT
#endif

/** OMX_ALL is used to as a wildcard to select all entities of the same type
 *  when specifying the index, or referring to a object by an index.  (i.e.
 *  use OMX_ALL to indicate all N channels). When used as a port index
 *  for a config or parameter this OMX_ALL denotes that the config or
 *  parameter applies to the entire component not just one port. */
#define OMX_ALL 0xFFFFFFFF

/** In the following we define groups that help building doxygen documentation */

/** @defgroup core OpenMAX IL core
 * Functions and structure related to the OMX IL core
 */
 
 /** @defgroup comp OpenMAX IL component
 * Functions and structure related to the OMX IL component
 */
 
/** @defgroup rpm Resource and Policy Management
 * Structures for resource and policy management of components
 */

/** @defgroup buf Buffer Management
 * Buffer handling functions and structures
 */
  
/** @defgroup tun Tunneling
 * @ingroup core comp
 * Structures and functions to manage tunnels among component ports
 */
 
/** @defgroup cp Content Pipes
 *  @ingroup core
 */
 
 /** @defgroup metadata Metadata handling
  * 
  */ 

/** OMX_U8 is an 8 bit unsigned quantity that is byte aligned */
typedef unsigned char OMX_U8;

/** OMX_S8 is an 8 bit signed quantity that is byte aligned */
typedef signed char OMX_S8;

/** OMX_U16 is a 16 bit unsigned quantity that is 16 bit word aligned */
typedef unsigned short OMX_U16;

/** OMX_S16 is a 16 bit signed quantity that is 16 bit word aligned */
typedef signed short OMX_S16;

/** OMX_U32 is a 32 bit unsigned quantity that is 32 bit word aligned */
typedef unsigned int OMX_U32;

/** OMX_S32 is a 32 bit signed quantity that is 32 bit word aligned */
typedef signed int OMX_S32;


/* Users with compilers that cannot accept the "long long" designation should
   define the OMX_SKIP64BIT macro.  It should be noted that this may cause
   some components to fail to compile if the component was written to require
   64 bit integral types.  However, these components would NOT compile anyway
   since the compiler does not support the way the component was written.
*/
#ifndef OMX_SKIP64BIT
#ifdef __SYMBIAN32__
/** OMX_U64 is a 64 bit unsigned quantity that is 64 bit word aligned */
typedef unsigned long long OMX_U64;

/** OMX_S64 is a 64 bit signed quantity that is 64 bit word aligned */
typedef signed long long OMX_S64;

#elif defined(WIN32)

/** OMX_U64 is a 64 bit unsigned quantity that is 64 bit word aligned */
typedef unsigned __int64  OMX_U64;

/** OMX_S64 is a 64 bit signed quantity that is 64 bit word aligned */
typedef signed   __int64  OMX_S64;

#else /* WIN32 */

/** OMX_U64 is a 64 bit unsigned quantity that is 64 bit word aligned */
typedef unsigned long long OMX_U64;

/** OMX_S64 is a 64 bit signed quantity that is 64 bit word aligned */
typedef signed long long OMX_S64;

#endif /* WIN32 */
#endif


/** The OMX_BOOL type is intended to be used to represent a true or a false
    value when passing parameters to and from the OMX core and components.  The
    OMX_BOOL is a 32 bit quantity and is aligned on a 32 bit word boundary.
 */
typedef enum OMX_BOOL {
    OMX_FALSE = 0,
    OMX_TRUE = !OMX_FALSE,
    OMX_BOOL_MAX = 0x7FFFFFFF
} OMX_BOOL;
 
#ifdef OMX_ANDROID_COMPILE_AS_32BIT_ON_64BIT_PLATFORMS

typedef OMX_U32 OMX_PTR;
typedef OMX_PTR OMX_STRING;
typedef OMX_PTR OMX_BYTE;

#else

/** The OMX_PTR type is intended to be used to pass pointers between the OMX
    applications and the OMX Core and components.  This is a 32 bit pointer and
    is aligned on a 32 bit boundary.
 */
typedef void* OMX_PTR;

/** The OMX_STRING type is intended to be used to pass "C" type strings between
    the application and the core and component.  The OMX_STRING type is a 32
    bit pointer to a zero terminated string.  The  pointer is word aligned and
    the string is byte aligned.
 */
typedef char* OMX_STRING;

/** The OMX_BYTE type is intended to be used to pass arrays of bytes such as
    buffers between the application and the component and core.  The OMX_BYTE
    type is a 32 bit pointer to a zero terminated string.  The  pointer is word
    aligned and the string is byte aligned.
 */
typedef unsigned char* OMX_BYTE;

/** OMX_UUIDTYPE is a very long unique identifier to uniquely identify
    at runtime.  This identifier should be generated by a component in a way
    that guarantees that every instance of the identifier running on the system
    is unique. */


#endif

typedef unsigned char OMX_UUIDTYPE[128];

/** The OMX_DIRTYPE enumeration is used to indicate if a port is an input or
    an output port.  This enumeration is common across all component types.
 */
typedef enum OMX_DIRTYPE
{
    OMX_DirInput,              /**< Port is an input port */
    OMX_DirOutput,             /**< Port is an output port */
    OMX_DirMax = 0x7FFFFFFF
} OMX_DIRTYPE;

/** The OMX_ENDIANTYPE enumeration is used to indicate the bit ordering
    for numerical data (i.e. big endian, or little endian).
 */
typedef enum OMX_ENDIANTYPE
{
    OMX_EndianBig, /**< big endian */
    OMX_EndianLittle, /**< little endian */
    OMX_EndianMax = 0x7FFFFFFF
} OMX_ENDIANTYPE;


/** The OMX_NUMERICALDATATYPE enumeration is used to indicate if data
    is signed or unsigned
 */
typedef enum OMX_NUMERICALDATATYPE
{
    OMX_NumericalDataSigned, /**< signed data */
    OMX_NumericalDataUnsigned, /**< unsigned data */
    OMX_NumercialDataMax = 0x7FFFFFFF
} OMX_NUMERICALDATATYPE;


/** Unsigned bounded value type */
typedef struct OMX_BU32 {
    OMX_U32 nValue; /**< actual value */
    OMX_U32 nMin;   /**< minimum for value (i.e. nValue >= nMin) */
    OMX_U32 nMax;   /**< maximum for value (i.e. nValue <= nMax) */
} OMX_BU32;


/** Signed bounded value type */
typedef struct OMX_BS32 {
    OMX_S32 nValue; /**< actual value */
    OMX_S32 nMin;   /**< minimum for value (i.e. nValue >= nMin) */
    OMX_S32 nMax;   /**< maximum for value (i.e. nValue <= nMax) */
} OMX_BS32;


/** Structure representing some time or duration in microseconds. This structure
  *  must be interpreted as a signed 64 bit value. The quantity is signed to accommodate
  *  negative deltas and preroll scenarios. The quantity is represented in microseconds
  *  to accomodate high resolution timestamps (e.g. DVD presentation timestamps based
  *  on a 90kHz clock) and to allow more accurate and synchronized delivery (e.g.
  *  individual audio samples delivered at 192 kHz). The quantity is 64 bit to 
  *  accommodate a large dynamic range (signed 32 bit values would allow only for plus
  *  or minus 35 minutes).
  *
  *  Implementations with limited precision may convert the signed 64 bit value to
  *  a signed 32 bit value internally but risk loss of precision.
  */
#ifndef OMX_SKIP64BIT
typedef OMX_S64 OMX_TICKS;
#else
typedef struct OMX_TICKS
{
    OMX_U32 nLowPart;    /** low bits of the signed 64 bit tick value */
    OMX_U32 nHighPart;   /** high bits of the signed 64 bit tick value */
} OMX_TICKS;
#endif
#define OMX_TICKS_PER_SECOND 1000000

/** Define the public interface for the OMX Handle.  The core will not use
    this value internally, but the application should only use this value.
 */
typedef void* OMX_HANDLETYPE;

typedef struct OMX_MARKTYPE
{
    OMX_HANDLETYPE hMarkTargetComponent;   /**< The component that will
                                                generate a mark event upon
                                                processing the mark. */
    OMX_PTR pMarkData;   /**< Application specific data associated with 
                              the mark sent on a mark event to disambiguate
                              this mark from others. */
} OMX_MARKTYPE;


/** OMX_NATIVE_DEVICETYPE is used to map a OMX video port to the
 *  platform & operating specific object used to reference the display 
 *  or can be used by a audio port for native audio rendering */
typedef void* OMX_NATIVE_DEVICETYPE;

/** OMX_NATIVE_WINDOWTYPE is used to map a OMX video port to the
 *  platform & operating specific object used to reference the window */
typedef void* OMX_NATIVE_WINDOWTYPE;

/** The OMX_VERSIONTYPE union is used to specify the version for
    a structure or component.  For a component, the version is entirely
    specified by the component vendor.  Components doing the same function
    from different vendors may or may not have the same version.  For
    structures, the version shall be set by the entity that allocates the
    structure.  For structures specified in the OMX 1.1 specification, the
    value of the version shall be set to 1.1.0.0 in all cases.  Access to the
    OMX_VERSIONTYPE can be by a single 32 bit access (e.g. by nVersion) or
    by accessing one of the structure elements to, for example, check only
    the Major revision.
 */
typedef union OMX_VERSIONTYPE
{
    struct
    {
        OMX_U8 nVersionMajor;   /**< Major version accessor element */
        OMX_U8 nVersionMinor;   /**< Minor version accessor element */
        OMX_U8 nRevision;       /**< Revision version accessor element */
        OMX_U8 nStep;           /**< Step version accessor element */
    } s;
    OMX_U32 nVersion;           /**< 32 bit value to make accessing the
                                    version easily done in a single word
                                    size copy/compare operation */
} OMX_VERSIONTYPE;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
/* File EOF */
