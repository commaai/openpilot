/*  =========================================================================
    czmq_prelude.h - CZMQ environment

    Copyright (c) the Contributors as noted in the AUTHORS file.
    This file is part of CZMQ, the high-level C binding for 0MQ:
    http://czmq.zeromq.org.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
    =========================================================================
*/

#ifndef __CZMQ_PRELUDE_H_INCLUDED__
#define __CZMQ_PRELUDE_H_INCLUDED__

//- Establish the compiler and computer system ------------------------------
/*
 *  Defines zero or more of these symbols, for use in any non-portable
 *  code:
 *
 *  __WINDOWS__         Microsoft C/C++ with Windows calls
 *  __MSDOS__           System is MS-DOS (set if __WINDOWS__ set)
 *  __VMS__             System is VAX/VMS or Alpha/OpenVMS
 *  __UNIX__            System is UNIX
 *  __OS2__             System is OS/2
 *
 *  __IS_32BIT__        OS/compiler is 32 bits
 *  __IS_64BIT__        OS/compiler is 64 bits
 *
 *  When __UNIX__ is defined, we also define exactly one of these:
 *
 *  __UTYPE_AUX         Apple AUX
 *  __UTYPE_BEOS        BeOS
 *  __UTYPE_BSDOS       BSD/OS
 *  __UTYPE_DECALPHA    Digital UNIX (Alpha)
 *  __UTYPE_IBMAIX      IBM RS/6000 AIX
 *  __UTYPE_FREEBSD     FreeBSD
 *  __UTYPE_HPUX        HP/UX
 *  __UTYPE_ANDROID     Android
 *  __UTYPE_LINUX       Linux
 *  __UTYPE_GNU         GNU/Hurd
 *  __UTYPE_MIPS        MIPS (BSD 4.3/System V mixture)
 *  __UTYPE_NETBSD      NetBSD
 *  __UTYPE_NEXT        NeXT
 *  __UTYPE_OPENBSD     OpenBSD
 *  __UTYPE_OSX         Apple Macintosh OS X
 *  __UTYPE_IOS         Apple iOS
 *  __UTYPE_QNX         QNX
 *  __UTYPE_IRIX        Silicon Graphics IRIX
 *  __UTYPE_SINIX       SINIX-N (Siemens-Nixdorf Unix)
 *  __UTYPE_SUNOS       SunOS
 *  __UTYPE_SUNSOLARIS  Sun Solaris
 *  __UTYPE_UNIXWARE    SCO UnixWare
 *                      ... these are the ones I know about so far.
 *  __UTYPE_GENERIC     Any other UNIX
 *
 *  When __VMS__ is defined, we may define one or more of these:
 *
 *  __VMS_XOPEN         Supports XOPEN functions
 */

#if (defined (__64BIT__) || defined (__x86_64__))
#    define __IS_64BIT__                //  May have 64-bit OS/compiler
#else
#    define __IS_32BIT__                //  Else assume 32-bit OS/compiler
#endif

#if (defined WIN32 || defined _WIN32)
#   undef __WINDOWS__
#   define __WINDOWS__
#   undef __MSDOS__
#   define __MSDOS__
#endif

#if (defined WINDOWS || defined _WINDOWS || defined __WINDOWS__)
#   undef __WINDOWS__
#   define __WINDOWS__
#   undef __MSDOS__
#   define __MSDOS__
//  Stop cheeky warnings about "deprecated" functions like fopen
#   if _MSC_VER >= 1500
#       undef  _CRT_SECURE_NO_DEPRECATE
#       define _CRT_SECURE_NO_DEPRECATE
#       pragma warning(disable: 4996)
#   endif
#endif

//  MSDOS               Microsoft C
//  _MSC_VER            Microsoft C
#if (defined (MSDOS) || defined (_MSC_VER))
#   undef __MSDOS__
#   define __MSDOS__
#   if (defined (_DEBUG) && !defined (DEBUG))
#       define DEBUG
#   endif
#endif

#if (defined (__EMX__) && defined (__i386__))
#   undef __OS2__
#   define __OS2__
#endif

//  VMS                 VAX C (VAX/VMS)
//  __VMS               Dec C (Alpha/OpenVMS)
//  __vax__             gcc
#if (defined (VMS) || defined (__VMS) || defined (__vax__))
#   undef __VMS__
#   define __VMS__
#   if (__VMS_VER >= 70000000)
#       define __VMS_XOPEN
#   endif
#endif

//  Try to define a __UTYPE_xxx symbol...
//  unix                SunOS at least
//  __unix__            gcc
//  _POSIX_SOURCE is various UNIX systems, maybe also VAX/VMS
#if (defined (unix) || defined (__unix__) || defined (_POSIX_SOURCE))
#   if (!defined (__VMS__))
#       undef __UNIX__
#       define __UNIX__
#       if (defined (__alpha))          //  Digital UNIX is 64-bit
#           undef  __IS_32BIT__
#           define __IS_64BIT__
#           define __UTYPE_DECALPHA
#       endif
#   endif
#endif

#if (defined (_AUX))
#   define __UTYPE_AUX
#   define __UNIX__
#elif (defined (__BEOS__))
#   define __UTYPE_BEOS
#   define __UNIX__
#elif (defined (__hpux))
#   define __UTYPE_HPUX
#   define __UNIX__
#   define _INCLUDE_HPUX_SOURCE
#   define _INCLUDE_XOPEN_SOURCE
#   define _INCLUDE_POSIX_SOURCE
#elif (defined (_AIX) || defined (AIX))
#   define __UTYPE_IBMAIX
#   define __UNIX__
#elif (defined (BSD) || defined (bsd))
#   define __UTYPE_BSDOS
#   define __UNIX__
#elif (defined (__ANDROID__))
#   define __UTYPE_ANDROID
#   define __UNIX__
#elif (defined (LINUX) || defined (linux) || defined (__linux__))
#   define __UTYPE_LINUX
#   define __UNIX__
#   ifndef __NO_CTYPE
#   define __NO_CTYPE                   //  Suppress warnings on tolower()
#   endif
#   ifndef _DEFAULT_SOURCE
#   define _DEFAULT_SOURCE                  //  Include stuff from 4.3 BSD Unix
#   endif
#elif (defined (__GNU__))
#   define __UTYPE_GNU
#   define __UNIX__
#elif (defined (Mips))
#   define __UTYPE_MIPS
#   define __UNIX__
#elif (defined (FreeBSD) || defined (__FreeBSD__))
#   define __UTYPE_FREEBSD
#   define __UNIX__
#elif (defined (NetBSD) || defined (__NetBSD__))
#   define __UTYPE_NETBSD
#   define __UNIX__
#elif (defined (OpenBSD) || defined (__OpenBSD__))
#   define __UTYPE_OPENBSD
#   define __UNIX__
#elif (defined (APPLE) || defined (__APPLE__))
#   include <TargetConditionals.h>
#   define __UNIX__
#   if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
#      define __UTYPE_IOS
#   else
#      define __UTYPE_OSX
#   endif
#elif (defined (NeXT))
#   define __UTYPE_NEXT
#   define __UNIX__
#elif (defined (__QNX__))
#   define __UTYPE_QNX
#   define __UNIX__
#elif (defined (sgi))
#   define __UTYPE_IRIX
#   define __UNIX__
#elif (defined (sinix))
#   define __UTYPE_SINIX
#   define __UNIX__
#elif (defined (SOLARIS) || defined (__SRV4))
#   define __UTYPE_SUNSOLARIS
#   define __UNIX__
#elif (defined (SUNOS) || defined (SUN) || defined (sun))
#   define __UTYPE_SUNOS
#   define __UNIX__
#elif (defined (__USLC__) || defined (UnixWare))
#   define __UTYPE_UNIXWARE
#   define __UNIX__
#elif (defined (__CYGWIN__))
#   define __UTYPE_CYGWIN
#   define __UNIX__
#elif (defined (__UNIX__))
#   define __UTYPE_GENERIC
#endif

//- Always include ZeroMQ headers -------------------------------------------

#include "zmq.h"
#if (ZMQ_VERSION < ZMQ_MAKE_VERSION (4, 2, 0))
#   include "zmq_utils.h"
#endif

//- Standard ANSI include files ---------------------------------------------

#include <ctype.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <signal.h>
#include <setjmp.h>
#include <assert.h>

//- System-specific include files -------------------------------------------

#if (defined (__MSDOS__))
#   if (defined (__WINDOWS__))
#       if (_WIN32_WINNT < 0x0501)
#           undef _WIN32_WINNT
#           define _WIN32_WINNT 0x0501
#       endif
#       if (!defined (FD_SETSIZE))
#           define FD_SETSIZE 1024      //  Max. filehandles/sockets
#       endif
#       include <direct.h>
#       include <winsock2.h>
#       include <windows.h>
#       include <process.h>
#       include <ws2tcpip.h>            //  For getnameinfo ()
#       include <iphlpapi.h>            //  For GetAdaptersAddresses ()
#   endif
#   include <malloc.h>
#   include <dos.h>
#   include <io.h>
#   include <fcntl.h>
#   include <sys/types.h>
#   include <sys/stat.h>
#   include <sys/utime.h>
#   include <share.h>
#endif

#if (defined (__UNIX__))
#   include <fcntl.h>
#   include <netdb.h>
#   include <unistd.h>
#   include <pthread.h>
#   include <dirent.h>
#   include <pwd.h>
#   include <grp.h>
#   include <utime.h>
#   include <inttypes.h>
#   include <syslog.h>
#   include <sys/types.h>
#   include <sys/param.h>
#   include <sys/socket.h>
#   include <sys/time.h>
#   include <sys/stat.h>
#   include <sys/ioctl.h>
#   include <sys/file.h>
#   include <sys/wait.h>
#   include <sys/un.h>
#   include <sys/uio.h>             //  Let CZMQ build with libzmq/3.x
#   include <netinet/in.h>          //  Must come before arpa/inet.h
#   if (!defined (__UTYPE_ANDROID)) && (!defined (__UTYPE_IBMAIX)) \
    && (!defined (__UTYPE_HPUX))
#       include <ifaddrs.h>
#   endif
#   if defined (__UTYPE_SUNSOLARIS) || defined (__UTYPE_SUNOS)
#       include <sys/sockio.h>
#   endif
#   if (!defined (__UTYPE_BEOS))
#       include <arpa/inet.h>
#       if (!defined (TCP_NODELAY))
#           include <netinet/tcp.h>
#       endif
#   endif
#   if (defined (__UTYPE_IBMAIX) || defined(__UTYPE_QNX))
#       include <sys/select.h>
#   endif
#   if (defined (__UTYPE_BEOS))
#       include <NetKit.h>
#   endif
#   if ((defined (_XOPEN_REALTIME) && (_XOPEN_REALTIME >= 1)) \
     || (defined (_POSIX_VERSION)  && (_POSIX_VERSION  >= 199309L)))
#       include <sched.h>
#   endif
#   if (defined (__UTYPE_OSX) || defined (__UTYPE_IOS))
#       include <mach/clock.h>
#       include <mach/mach.h>           //  For monotonic clocks
#   endif
#   if (defined (__UTYPE_OSX))
#       include <crt_externs.h>         //  For _NSGetEnviron()
#   endif
#   if (defined (__UTYPE_ANDROID))
#       include <android/log.h>
#   endif
#   if (defined (__UTYPE_LINUX) && defined (HAVE_LIBSYSTEMD))
#       include <systemd/sd-daemon.h>
#   endif
#endif

#if (defined (__VMS__))
#   if (!defined (vaxc))
#       include <fcntl.h>               //  Not provided by Vax C
#   endif
#   include <netdb.h>
#   include <unistd.h>
#   include <pthread.h>
#   include <unixio.h>
#   include <unixlib.h>
#   include <types.h>
#   include <file.h>
#   include <socket.h>
#   include <dirent.h>
#   include <time.h>
#   include <pwd.h>
#   include <stat.h>
#   include <in.h>
#   include <inet.h>
#endif

#if (defined (__OS2__))
#   include <sys/types.h>               //  Required near top
#   include <fcntl.h>
#   include <malloc.h>
#   include <netdb.h>
#   include <unistd.h>
#   include <pthread.h>
#   include <dirent.h>
#   include <pwd.h>
#   include <grp.h>
#   include <io.h>
#   include <process.h>
#   include <sys/param.h>
#   include <sys/socket.h>
#   include <sys/select.h>
#   include <sys/time.h>
#   include <sys/stat.h>
#   include <sys/ioctl.h>
#   include <sys/file.h>
#   include <sys/wait.h>
#   include <netinet/in.h>              //  Must come before arpa/inet.h
#   include <arpa/inet.h>
#   include <utime.h>
#   if (!defined (TCP_NODELAY))
#       include <netinet/tcp.h>
#   endif
#endif

//  Add missing defines for non-POSIX systems
#ifndef S_IRUSR
#   define S_IRUSR S_IREAD
#endif
#ifndef S_IWUSR
#   define S_IWUSR S_IWRITE
#endif
#ifndef S_ISDIR
#   define S_ISDIR(m) (((m) & S_IFDIR) != 0)
#endif
#ifndef S_ISREG
#   define S_ISREG(m) (((m) & S_IFREG) != 0)
#endif


//- Check compiler data type sizes ------------------------------------------

#if (UCHAR_MAX != 0xFF)
#   error "Cannot compile: must change definition of 'byte'."
#endif
#if (USHRT_MAX != 0xFFFFU)
#    error "Cannot compile: must change definition of 'dbyte'."
#endif
#if (UINT_MAX != 0xFFFFFFFFU)
#    error "Cannot compile: must change definition of 'qbyte'."
#endif

//- Data types --------------------------------------------------------------

typedef unsigned char   byte;           //  Single unsigned byte = 8 bits
typedef unsigned short  dbyte;          //  Double byte = 16 bits
typedef unsigned int    qbyte;          //  Quad byte = 32 bits
typedef struct sockaddr_in  inaddr_t;   //  Internet socket address structure
typedef struct sockaddr_in6 in6addr_t;  //  Internet 6 socket address structure

// Common structure to hold inaddr_t and in6addr_t with length
typedef struct {
    union {
        inaddr_t __addr;          //  IPv4 address
        in6addr_t __addr6;        //  IPv6 address
    } __inaddr_u;
#define ipv4addr   __inaddr_u.__addr
#define ipv6addr   __inaddr_u.__addr6
    int inaddrlen;
} inaddr_storage_t;

//- Inevitable macros -------------------------------------------------------

#define streq(s1,s2)    (!strcmp ((s1), (s2)))
#define strneq(s1,s2)   (strcmp ((s1), (s2)))

//  Provide random number from 0..(num-1)
#if (defined (__WINDOWS__)) || (defined (__UTYPE_IBMAIX)) \
 || (defined (__UTYPE_HPUX)) || (defined (__UTYPE_SUNOS))
#   define randof(num)  (int) ((float) (num) * rand () / (RAND_MAX + 1.0))
#else
#   define randof(num)  (int) ((float) (num) * random () / (RAND_MAX + 1.0))
#endif

// Windows MSVS doesn't have stdbool
#if (defined (_MSC_VER))
#   if (!defined (__cplusplus) && (!defined (true)))
#       define true 1
#       define false 0
        typedef char bool;
#   endif
#else
#   include <stdbool.h>
#endif

//- A number of POSIX and C99 keywords and data types -----------------------
//  CZMQ uses uint for array indices; equivalent to unsigned int, but more
//  convenient in code. We define it in czmq_prelude.h on systems that do
//  not define it by default.

#if (defined (__WINDOWS__))
#   if (!defined (__cplusplus) && (!defined (inline)))
#       define inline __inline
#   endif
#   define strtoull _strtoui64
#   define atoll _atoi64
#   define srandom srand
#   define TIMEZONE _timezone
#   if (!defined (__MINGW32__))
#       define snprintf _snprintf
#       define vsnprintf _vsnprintf
#   endif
    typedef unsigned long ulong;
    typedef unsigned int  uint;
#   if (!defined (__MINGW32__))
    typedef int mode_t;
#     if !defined (_SSIZE_T_DEFINED)
typedef intptr_t ssize_t;
#       define _SSIZE_T_DEFINED
#     endif
#   endif
#   if ((!defined (__MINGW32__) \
    || (defined (__MINGW32__) && defined (__IS_64BIT__))) \
    && !defined (ZMQ_DEFINED_STDINT))
    typedef __int8 int8_t;
    typedef __int16 int16_t;
    typedef __int32 int32_t;
    typedef __int64 int64_t;
    typedef unsigned __int8 uint8_t;
    typedef unsigned __int16 uint16_t;
    typedef unsigned __int32 uint32_t;
    typedef unsigned __int64 uint64_t;
#   endif
    typedef uint32_t in_addr_t;
#   if (!defined (PRId8))
#       define PRId8    "d"
#   endif
#   if (!defined (PRId16))
#       define PRId16   "d"
#   endif
#   if (!defined (PRId32))
#       define PRId32   "d"
#   endif
#   if (!defined (PRId64))
#       define PRId64   "I64d"
#   endif
#   if (!defined (PRIu8))
#       define PRIu8    "u"
#   endif
#   if (!defined (PRIu16))
#       define PRIu16   "u"
#   endif
#   if (!defined (PRIu32))
#       define PRIu32   "u"
#   endif
#   if (!defined (PRIu64))
#       define PRIu64   "I64u"
#   endif
#   if (!defined (va_copy))
    //  MSVC does not support C99's va_copy so we use a regular assignment
#       define va_copy(dest,src) (dest) = (src)
#   endif
#elif (defined (__UTYPE_OSX))
    typedef unsigned long ulong;
    typedef unsigned int uint;
    //  This fixes header-order dependence problem with some Linux versions
#elif (defined (__UTYPE_LINUX))
#   if (__STDC_VERSION__ >= 199901L && !defined (__USE_MISC))
    typedef unsigned int uint;
#   endif
#endif

//- Non-portable declaration specifiers -------------------------------------

//  For thread-local storage
#if defined (__WINDOWS__)
#   define CZMQ_THREADLS __declspec(thread)
#else
#   define CZMQ_THREADLS __thread
#endif

//  Replacement for malloc() which asserts if we run out of heap, and
//  which zeroes the allocated block.
static inline void *
safe_malloc (size_t size, const char *file, unsigned line)
{
//     printf ("%s:%u %08d\n", file, line, (int) size);
    void *mem = calloc (1, size);
    if (mem == NULL) {
        fprintf (stderr, "FATAL ERROR at %s:%u\n", file, line);
        fprintf (stderr, "OUT OF MEMORY (malloc returned NULL)\n");
        fflush (stderr);
        abort ();
    }
    return mem;
}

//  Define _ZMALLOC_DEBUG if you need to trace memory leaks using e.g. mtrace,
//  otherwise all allocations will claim to come from czmq_prelude.h. For best
//  results, compile all classes so you see dangling object allocations.
//  _ZMALLOC_PEDANTIC does the same thing, but its intention is to propagate
//  out of memory condition back up the call stack.
#if defined _ZMALLOC_DEBUG || _ZMALLOC_PEDANTIC
#   define zmalloc(size) calloc(1,(size))
#else
#   define zmalloc(size) safe_malloc((size), __FILE__, __LINE__)
#endif

//  GCC supports validating format strings for functions that act like printf
#if defined (__GNUC__) && (__GNUC__ >= 2)
#   define CHECK_PRINTF(a)   __attribute__((format (printf, a, a + 1)))
#else
#   define CHECK_PRINTF(a)
#endif

//  Lets us write code that compiles both on Windows and normal platforms
#if !defined (__WINDOWS__)
typedef int SOCKET;
#   define closesocket      close
#   define INVALID_SOCKET   -1
#   define SOCKET_ERROR     -1
#   define O_BINARY         0
#endif

//- Include non-portable header files based on platform.h -------------------

#if defined (HAVE_LINUX_WIRELESS_H)
#   include <linux/wireless.h>
//  This would normally come from net/if.h
unsigned int if_nametoindex (const char *ifname);
#else
#   if defined (HAVE_NET_IF_H)
#       include <net/if.h>
#   endif
#   if defined (HAVE_NET_IF_MEDIA_H)
#       include <net/if_media.h>
#   endif
#endif

#if defined (__WINDOWS__) && !defined (HAVE_UUID)
#   define HAVE_UUID 1
#endif
#if defined (__UTYPE_OSX) && !defined (HAVE_UUID)
#   define HAVE_UUID 1
#endif
#if defined (HAVE_UUID)
#   if defined (__UTYPE_FREEBSD) || defined (__UTYPE_NETBSD)
#       include <uuid.h>
#   elif defined __UTYPE_HPUX
#       include <dce/uuid.h>
#   elif defined (__UNIX__)
#       include <uuid/uuid.h>
#   endif
#endif

//  ZMQ compatibility macros

#if ZMQ_VERSION_MAJOR == 4
#   define ZMQ_POLL_MSEC    1           //  zmq_poll is msec

#elif ZMQ_VERSION_MAJOR == 3
#   define ZMQ_POLL_MSEC    1           //  zmq_poll is msec
#   if  ZMQ_VERSION_MINOR < 2
#       define zmq_ctx_new  zmq_init
#   endif
#   define zmq_ctx_term     zmq_term

#elif ZMQ_VERSION_MAJOR == 2
#   define ZMQ_POLL_MSEC    1000        //  zmq_poll is usec
#   define zmq_sendmsg      zmq_send    //  Smooth out 2.x changes
#   define zmq_recvmsg      zmq_recv
#   define zmq_ctx_new      zmq_init
#   define zmq_ctx_term     zmq_term
#   define zmq_msg_send(m,s,f)  zmq_sendmsg ((s),(m),(f))
#   define zmq_msg_recv(m,s,f)  zmq_recvmsg ((s),(m),(f))
    //  Older libzmq APIs may be missing some aspects of libzmq v3.0
#   ifndef ZMQ_ROUTER
#       define ZMQ_ROUTER       ZMQ_XREP
#   endif
#   ifndef ZMQ_DEALER
#       define ZMQ_DEALER       ZMQ_XREQ
#   endif
#   ifndef ZMQ_DONTWAIT
#       define ZMQ_DONTWAIT     ZMQ_NOBLOCK
#   endif
#   ifndef ZMQ_XSUB
#       error "please upgrade your libzmq from http://zeromq.org"
#   endif
#   if  ZMQ_VERSION_MINOR == 0 \
    || (ZMQ_VERSION_MINOR == 1 && ZMQ_VERSION_PATCH < 7)
#       error "CZMQ requires at least libzmq/2.1.7 stable"
#   endif
#endif

#endif
