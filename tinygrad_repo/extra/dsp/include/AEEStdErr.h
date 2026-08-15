/*
 * Copyright (c) 2005-2007, 2012-2013, 2019-2020 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef AEESTDERR_H
#define AEESTDERR_H
//
// Basic Error Codes
//
//
#if defined(__hexagon__)
	#define AEE_EOFFSET               0x80000400
#else
	#define AEE_EOFFSET               0x00000000
#endif
/** @defgroup stdbasicerror Basic error codes
 *  @{
 */
#define AEE_SUCCESS                   0                      ///< No error
#define AEE_EUNKNOWN                  -1                     ///< Unknown error (should not use this)

#define AEE_EFAILED                   (AEE_EOFFSET + 0x001)  ///< General failure
#define AEE_ENOMEMORY                 (AEE_EOFFSET + 0x002)  ///< Memory allocation failed because of insufficient RAM
#define AEE_ECLASSNOTSUPPORT          (AEE_EOFFSET + 0x003)  ///< Specified class unsupported
#define AEE_EVERSIONNOTSUPPORT        (AEE_EOFFSET + 0x004)  ///< Version not supported
#define AEE_EALREADYLOADED            (AEE_EOFFSET + 0x005)  ///< Object already loaded
#define AEE_EUNABLETOLOAD             (AEE_EOFFSET + 0x006)  ///< Unable to load object/applet
#define AEE_EUNABLETOUNLOAD           (AEE_EOFFSET + 0x007)  ///< Unable to unload
                                                                    ///< object/applet
#define AEE_EALARMPENDING             (AEE_EOFFSET + 0x008)  ///< Alarm is pending
#define AEE_EINVALIDTIME              (AEE_EOFFSET + 0x009)  ///< Invalid time
#define AEE_EBADCLASS                 (AEE_EOFFSET + 0x00A)  ///< NULL class object
#define AEE_EBADMETRIC                (AEE_EOFFSET + 0x00B)  ///< Invalid metric specified
#define AEE_EEXPIRED                  (AEE_EOFFSET + 0x00C)  ///< App/Component Expired
#define AEE_EBADSTATE                 (AEE_EOFFSET + 0x00D)  ///< Process or thread is not in expected state
#define AEE_EBADPARM                  (AEE_EOFFSET + 0x00E)  ///< Invalid parameter
#define AEE_ESCHEMENOTSUPPORTED       (AEE_EOFFSET + 0x00F)  ///< Invalid URL scheme
#define AEE_EBADITEM                  (AEE_EOFFSET + 0x010)  ///< Value out of range
#define AEE_EINVALIDFORMAT            (AEE_EOFFSET + 0x011)  ///< Invalid format
#define AEE_EINCOMPLETEITEM           (AEE_EOFFSET + 0x012)  ///< Incomplete item, like length of a string is less that expected
#define AEE_ENOPERSISTMEMORY          (AEE_EOFFSET + 0x013)  ///< Insufficient flash
#define AEE_EUNSUPPORTED              (AEE_EOFFSET + 0x014)  ///< API not implemented
#define AEE_EPRIVLEVEL                (AEE_EOFFSET + 0x015)  ///< Privileges are insufficient
                                                                    ///< for this operation
#define AEE_ERESOURCENOTFOUND         (AEE_EOFFSET + 0x016)  ///< Unable to find specified
                                                                    ///< resource
#define AEE_EREENTERED                (AEE_EOFFSET + 0x017)  ///< Non re-entrant API
                                                                    ///< re-entered
#define AEE_EBADTASK                  (AEE_EOFFSET + 0x018)  ///< API called in wrong task
                                                                    ///< context
#define AEE_EALLOCATED                (AEE_EOFFSET + 0x019)  ///< App/Module left memory
                                                                    ///< allocated when released.
#define AEE_EALREADY                  (AEE_EOFFSET + 0x01A)  ///< Operation is already in
                                                                    ///< progress
#define AEE_EADSAUTHBAD               (AEE_EOFFSET + 0x01B)  ///< ADS mutual authorization
                                                                    ///< failed
#define AEE_ENEEDSERVICEPROG          (AEE_EOFFSET + 0x01C)  ///< Need service programming
#define AEE_EMEMPTR                   (AEE_EOFFSET + 0x01D)  ///< bad memory pointer, expected to be NULL
#define AEE_EHEAP                     (AEE_EOFFSET + 0x01E)  ///< An internal heap error was detected
#define AEE_EIDLE                     (AEE_EOFFSET + 0x01F)  ///< Context (system, interface,
                                                                    ///< etc.) is idle
#define AEE_EITEMBUSY                 (AEE_EOFFSET + 0x020)  ///< Context (system, interface,
                                                                    ///< etc.) is busy
#define AEE_EBADSID                   (AEE_EOFFSET + 0x021)  ///< Invalid subscriber ID
#define AEE_ENOTYPE                   (AEE_EOFFSET + 0x022)  ///< No type detected/found
#define AEE_ENEEDMORE                 (AEE_EOFFSET + 0x023)  ///< Need more data/info
#define AEE_EADSCAPS                  (AEE_EOFFSET + 0x024)  ///< ADS Capabilities do not
                                                                    ///< match those required for phone
#define AEE_EBADSHUTDOWN              (AEE_EOFFSET + 0x025)  ///< App failed to close properly
#define AEE_EBUFFERTOOSMALL           (AEE_EOFFSET + 0x026)  ///< Destination buffer given is
                                                                    ///< too small
                                                                    ///< or service exists or is
                                                                    ///< valid
#define AEE_EACKPENDING               (AEE_EOFFSET + 0x028)  ///< ACK pending on application
#define AEE_ENOTOWNER                 (AEE_EOFFSET + 0x029)  ///< Not an owner authorized to
                                                                    ///< perform the operation
#define AEE_EINVALIDITEM              (AEE_EOFFSET + 0x02A)  ///< Current item is invalid, it can be a switch case or a pointer to memory
#define AEE_ENOTALLOWED               (AEE_EOFFSET + 0x02B)  ///< Not allowed to perform the
                                                                    ///< operation
#define AEE_EBADHANDLE                (AEE_EOFFSET + 0x02C)  ///< Invalid/Wrong handle
#define AEE_EINVHANDLE                (AEE_EOFFSET + 0x02C)  ///< Invalid handle - adding here as its defined in vendor AEEStdErr.h - needed to check valid handle in stub.c
#define AEE_EOUTOFHANDLES             (AEE_EOFFSET + 0x02D)  ///< Out of handles (Handle list is already full)
//Hole here
#define AEE_ENOMORE                   (AEE_EOFFSET + 0x02F)  ///< No more items available --
                                                                    ///< reached end
#define AEE_ECPUEXCEPTION             (AEE_EOFFSET + 0x030)  ///< A CPU exception occurred
#define AEE_EREADONLY                 (AEE_EOFFSET + 0x031)  ///< Cannot change read-only
                                                                    ///< object or parameter ( Parameter is in protected mode)
#define AEE_ERPC                      (AEE_EOFFSET + 0x200)  ///< Error due to fastrpc implementation
#define AEE_EFILE                     (AEE_EOFFSET + 0x201)  ///<File handling related error
//NOTE: Used in both HLOS and DSP.
#define AEE_ENOSUCH                   (39)                   ///< No such name, port, socket
#define AEE_EINTERRUPTED              (46)                   ///< Waitable call is interrupted,
                                                                   ///< the user should return to the HLOS and retry the call
#define AEE_ECONNRESET                (104)                  ///< Connection reset by peer
#define AEE_EWOULDBLOCK               (516)                  ///< Operation would block if not
                                                                    ///< non-blocking; wait and try
                                                                    ///< again
/**
 * @}
 */

/** @defgroup sigverifyerror Sigverify error codes
 *  @{
 */

#define AEE_EINVALIDMSG	              (AEE_EOFFSET + 0x032)     ///<  Invalid SMD message from APPS
#define AEE_EINVALIDTHREAD            (AEE_EOFFSET + 0x033)     ///<  Invalid thread
#define AEE_EINVALIDPROCESS           (AEE_EOFFSET + 0x034)     ///<  Invalid Process
#define AEE_EINVALIDFILENAME          (AEE_EOFFSET + 0x035)     ///<  Invalid filename
#define AEE_EINVALIDDIGESTSIZE        (AEE_EOFFSET + 0x036)     ///<  Invalid digest size
#define AEE_EINVALIDSEGS              (AEE_EOFFSET + 0x037)     ///<  Invalid segments
#define AEE_EINVALIDSIGNATURE         (AEE_EOFFSET + 0x038)     ///<  Invalid signature
#define AEE_EINVALIDDOMAIN            (AEE_EOFFSET + 0x039)     ///<  Invalid DSP domain
#define AEE_EINVALIDFD                (AEE_EOFFSET + 0x03A)     ///<  Invalid file descriptor
#define AEE_EINVALIDDEVICE            (AEE_EOFFSET + 0x03B)     ///<  Invalid Device or Device node open failed for the domain
#define AEE_EINVALIDMODE              (AEE_EOFFSET + 0x03C)     ///<  Invalid Mode
#define AEE_EINVALIDPROCNAME          (AEE_EOFFSET + 0x03D)     ///<  Invalid Process name
#define AEE_ENOSUCHMOD                (AEE_EOFFSET + 0x03E)     ///<  No such module
#define AEE_ENOSUCHINSTANCE           (AEE_EOFFSET + 0x03F)     ///<  No instance in the list lookup
#define AEE_ENOSUCHTHREAD             (AEE_EOFFSET + 0x040)     ///<  No such thread
#define AEE_ENOSUCHPROCESS            (AEE_EOFFSET + 0x041)     ///<  No such process
#define AEE_ENOSUCHSYMBOL             (AEE_EOFFSET + 0x042)     ///<  No such symbol( dlsym for the symbol failed)
#define AEE_ENOSUCHDEVICE             (AEE_EOFFSET + 0x043)     ///<  No such device
#define AEE_ENOSUCHPROP               (AEE_EOFFSET + 0x044)     ///<  No such dal property
#define AEE_ENOSUCHFILE               (AEE_EOFFSET + 0x045)     ///<  No such file found
#define AEE_ENOSUCHHANDLE             (AEE_EOFFSET + 0x046)     ///<  No such handle
#define AEE_ENOSUCHSTREAM             (AEE_EOFFSET + 0x047)     ///<  No such stream
#define AEE_ENOSUCHMAP                (AEE_EOFFSET + 0x048)     ///<  No mapping exists for this address on DSP
#define AEE_ENOSUCHREGISTER           (AEE_EOFFSET + 0x049)     ///<  No such register
#define AEE_ENOSUCHCLIENT             (AEE_EOFFSET + 0x04A)     ///<  No such QDI client
#define AEE_EBADDOMAIN                (AEE_EOFFSET + 0x04B)     ///<  Bad domain (not initialized)
#define AEE_EBADOFFSET                (AEE_EOFFSET + 0x04C)     ///<  Bad buffer/page/heap offset
#define AEE_EBADSIZE                  (AEE_EOFFSET + 0x04D)     ///<  Bad buffer/page/heap size
#define AEE_EBADPERMS                 (AEE_EOFFSET + 0x04E)     ///<  Bad FILE/MAP/MEM permissions
#define AEE_EBADFD                    (AEE_EOFFSET + 0x04F)     ///<  Bad file descriptor
#define AEE_EBADPID                   (AEE_EOFFSET + 0x050)     ///<  Bad PID from HLOS
#define AEE_EBADTID                   (AEE_EOFFSET + 0x051)     ///<  Bad TID
#define AEE_EBADELF                   (AEE_EOFFSET + 0x052)     ///<  Bad elf file
#define AEE_EBADASID                  (AEE_EOFFSET + 0x053)     ///<  Bad asid
#define AEE_EBADCONTEXT               (AEE_EOFFSET + 0x054)     ///<  Bad context
#define AEE_EBADMEMALIGN              (AEE_EOFFSET + 0x055)     ///<  Bad memory alignment
#define AEE_EIOCTL                    (AEE_EOFFSET + 0x056)     ///<  ioctl call failed
#define AEE_EFOPEN                    (AEE_EOFFSET + 0x057)     ///<  file open error or device node open failed for DSP domain
#define AEE_EFGETS                    (AEE_EOFFSET + 0x058)     ///<  file get string error
#define AEE_EFFLUSH                   (AEE_EOFFSET + 0x059)     ///<  file flush error
#define AEE_EFCLOSE                   (AEE_EOFFSET + 0x05A)     ///<  file close error
#define AEE_EEOF                      (AEE_EOFFSET + 0x05B)     ///<  File EOF reached
#define AEE_EFREAD                    (AEE_EOFFSET + 0x05C)     ///<  file read failed
#define AEE_EFWRITE                   (AEE_EOFFSET + 0x05D)     ///<  file write failed
#define AEE_EFGETPOS                  (AEE_EOFFSET + 0x05E)     ///<  file get position failed
#define AEE_EFSETPOS                  (AEE_EOFFSET + 0x05F)     ///<  file set position failed
#define AEE_EFTELL                    (AEE_EOFFSET + 0x060)     ///<  file tell position failed
#define AEE_EFSEEK                    (AEE_EOFFSET + 0x061)     ///<  file seek failed
#define AEE_EFLEN                     (AEE_EOFFSET + 0x062)     ///<  file len greater than expected
#define AEE_EGETENV                   (AEE_EOFFSET + 0x063)     ///<  apps_std get enviroment failed
#define AEE_ESETENV                   (AEE_EOFFSET + 0x064)     ///<  apps_std set enviroment failed
#define AEE_EMMAP                     (AEE_EOFFSET + 0x065)     ///<  mmap failed
#define AEE_EIONMAP                   (AEE_EOFFSET + 0x066)     ///<  ion map failed
#define AEE_EIONALLOC                 (AEE_EOFFSET + 0x067)     ///<  ion alloc failed
#define AEE_ENORPCMEMORY              (AEE_EOFFSET + 0x068)     ///<  ION memory allocation failed
#define AEE_ENOROOTOFTRUST            (AEE_EOFFSET + 0x069)     ///<  No root of trust for sigverify
#define AEE_ENOTLOCKED                (AEE_EOFFSET + 0x06A)     ///<  Unlock failed, not locked before
#define AEE_ENOTINITIALIZED           (AEE_EOFFSET + 0x06B)     ///<  Not initialized
#define AEE_EUNSUPPORTEDAPI           (AEE_EOFFSET + 0x06C)     ///<  unsupported API/request ID
#define AEE_EUNPACK                   (AEE_EOFFSET + 0x06D)     ///<  unpacking command failed
#define AEE_EPOLL                     (AEE_EOFFSET + 0x06E)     ///<  error while polling for event
#define AEE_EEVENTREAD                (AEE_EOFFSET + 0x06F)     ///<  event read failed
#define AEE_EMAXBUFS                  (AEE_EOFFSET + 0x070)     ///<  Maximum buffers
#define AEE_EINVARGS                  (AEE_EOFFSET + 0x071)     ///<  Invalid Arguments
#define AEE_ECONNREFUSED              (AEE_EOFFSET + 0x072)     ///<  Connection refused to DSP
#define AEE_EUNSIGNEDMOD              (AEE_EOFFSET + 0x081)     ///<  test-sig not found, Unsigned shared object
#define AEE_EINVALIDHASH              (AEE_EOFFSET + 0x082)     ///<  test-sig not found, Invalid hash object
#define AEE_EBADVA                    (AEE_EOFFSET + 0x083)     ///<  Bad VA address
#define AEE_ENOSUCHJOB                (AEE_EOFFSET + 0x084)     ///<  No such job
#define AEE_ENOSUCHGROUP              (AEE_EOFFSET + 0x084)     ///<  No such static pd group
#define AEE_EBADMAPREFCNT             (AEE_EOFFSET + 0x085)     ///<  Bad map reference count
#define AEE_EBADPAGECNT               (AEE_EOFFSET + 0x086)     ///<  Bad page count
#define AEE_EMAPALREADYPRESENT        (AEE_EOFFSET + 0x087)     ///<  Map already present
#define AEE_ENOFREESECTION            (AEE_EOFFSET + 0x088)     ///<  No more free sections available
#define AEE_U2GCLIENT_OPEN            (AEE_EOFFSET + 0x089)     ///<  u2g client open failed

/**
 * @}
 */

/** @defgroup smderror SMD error codes
 *  @{
 */


#if defined(__hexagon__)
	#define AEE_EGLINK_OFFSET         (AEE_EOFFSET + 0x100)     ///<  SMD errors offset
	#define AEE_EGLINKBADPACKET       (AEE_EOFFSET + 0x101)     ///<  SMD invalid packet size
	#define AEE_EGLINKALREADYOPEN     (AEE_EOFFSET + 0x102)     ///<  SMD port is already open
	#define AEE_EGLINKOPENFAILED      (AEE_EOFFSET + 0x103)     ///<  SMD port open failed
	#define AEE_EGLINKWRITE           (AEE_EOFFSET + 0x104)     ///<  SMD port write failed
	#define AEE_EGLINKREGISTER        (AEE_EOFFSET + 0x105)     ///<  SMD port register callback failed
#else
	#define AEE_ESMD_OFFSET           (AEE_EOFFSET + 0x100)     ///<  SMD errors offset
	#define AEE_ESMDBADPACKET         (AEE_EOFFSET + 0x101)     ///<  SMD invalid packet size
	#define AEE_ESMDALREADYOPEN       (AEE_EOFFSET + 0x102)     ///<  SMD port is already open
	#define AEE_ESMDOPENFAILED        (AEE_EOFFSET + 0x103)     ///<  SMD port open failed
#endif
 /**
 * @}
 */

 /** @defgroup dalerror DAL error codes
 *  @{
 */


#define AEE_EDAL_OFFSET               (AEE_EOFFSET + 0x120)     ///<  Dal error offset
#define AEE_EDALDEVATTACH             (AEE_EOFFSET + 0x121)     ///<  DAL attach error
#define AEE_EDALINTREGISTER           (AEE_EOFFSET + 0x122)     ///<  DAL interrupt register error
#define AEE_EDALINTUNREGISTER         (AEE_EOFFSET + 0x123)     ///<  Dal interrupt unregister error
#define AEE_EDALGETPROP               (AEE_EOFFSET + 0x124)     ///<  Dal get property
#define AEE_EDALGETVAL                (AEE_EOFFSET + 0x125)     ///<  Dal get property value
#define AEE_EDCVSREQUEST              (AEE_EOFFSET + 0x126)     ///<  Dal get property value

 /**
 * @}
 */

 /** @defgroup qurterror QURT error codes
 *  @{
 */

#define AEE_EQURT_OFFSET              (AEE_EOFFSET + 0x140)     ///<  QURT error offset
#define AEE_EQURTREGIONCREATE         (AEE_EOFFSET + 0x141)     ///<  QURT region create failed
#define AEE_EQURTCACHECLEAN	          (AEE_EOFFSET + 0x142)     ///<  QURT cache clean failed
#define AEE_EQURTREGIONGETATTR        (AEE_EOFFSET + 0x143)     ///<  QURT region get attribute failed
#define AEE_EQURTBADREGIONPERMS       (AEE_EOFFSET + 0x144)     ///<  QURT bad permissions for region
#define AEE_EQURTMEMPOOLADD	          (AEE_EOFFSET + 0x145)     ///<  QURT Add to memory pool failed
#define AEE_EQURTREGISTERDEV          (AEE_EOFFSET + 0x146)     ///<  QURT register device failed
#define AEE_EQURTMEMPOOLCREATE        (AEE_EOFFSET + 0x147)     ///<  QURT create memory pool failed
#define AEE_EQURTGETVA                (AEE_EOFFSET + 0x148)     ///<  QURT get VA failed
#define AEE_EQURTREGIONDELETE         (AEE_EOFFSET + 0x149)     ///<  QURT region delete failed
#define AEE_EQURTMEMPOOLATTACH        (AEE_EOFFSET + 0x14A)     ///<  QURT memory pool attach failed
#define AEE_EQURTTHREADCREATE         (AEE_EOFFSET + 0x14B)     ///<  QURT thread create failed
#define AEE_EQURTCOPYTOUSER           (AEE_EOFFSET + 0x14C)     ///<  QURT copy to user memory failed
#define AEE_EQURTMEMMAPCREATE         (AEE_EOFFSET + 0x14D)     ///<  QURT map create failed
#define AEE_EQURTINVHANDLE            (AEE_EOFFSET + 0x14E)     ///<  QURT Invalid client handle
#define AEE_EQURTBADASID              (AEE_EOFFSET + 0x14F)     ///<  QURT Bad ASIC from QURT
#define AEE_EQURTOPENFAILED           (AEE_EOFFSET + 0x150)     ///<  QURT QDI open failed
#define AEE_EQURTCOPYFROMUSER         (AEE_EOFFSET + 0x151)     ///<  QURT Copy from user failed
#define AEE_EQURTLINELOCK             (AEE_EOFFSET + 0x152)     ///<  QURT Line lock failed
#define AEE_EQURTQDIDEFMETHOD         (AEE_EOFFSET + 0x153)     ///<  QURT QDI default method failed
#define AEE_EQURTCREATEHANDLE         (AEE_EOFFSET + 0x154)     ///<  QURT create handle from obj failed
#define AEE_EQURTWRITABLEMEM          (AEE_EOFFSET + 0x155)     ///<  QURT CPZ migration writable mem
#define AEE_EQURTTHREADCREATEDEF      (AEE_EOFFSET + 0x156)     ///<  QURT thread create def
#define AEE_EQURTLOOKUPVA             (AEE_EOFFSET + 0x157)     ///<  QURT lookup VA
#define AEE_EQURTLOOKUPPA             (AEE_EOFFSET + 0x158)     ///<  QURT lookup PA
#define AEE_EQURTMIGRATESECURE        (AEE_EOFFSET + 0x159)     ///<  QURT CPZ migration failure
#define AEE_EQURTQDIOPEN              (AEE_EOFFSET + 0X160)     ///<  QURT QDI open failure
#define AEE_EQURTMAPREMOVE            (AEE_EOFFSET + 0X161)     ///<  QURT map remove failure
#define AEE_EQURTQDICLOSE             (AEE_EOFFSET + 0X162)     ///<  QURT QDI close failed
#define AEE_EQURTWAIT                 (AEE_EOFFSET + 0X163)     ///<  QURT Futex wait failed

 /**
 * @}
 */

  /** @defgroup mmpmerr MMPM error codes
 *  @{
 */

#define AEE_EMMPM_OFFSET              (AEE_EOFFSET + 0x170)     ///<  MMPM errors offset
#define AEE_EMMPMREQUEST              (AEE_EOFFSET + 0x171)     ///<  MMPM Power request to failed
#define AEE_EMMPMRELEASE              (AEE_EOFFSET + 0x172)     ///<  MMPM Release request failed
#define AEE_EMMPMSETPARAM             (AEE_EOFFSET + 0x173)     ///<  MMPM set param request failed
#define AEE_EMMPMREGISTER             (AEE_EOFFSET + 0x174)     ///<  MMPM Register request failed
#define AEE_EMMPMGETINFO              (AEE_EOFFSET + 0x175)     ///<  MMPM Get info failed
#define AEE_EMAX_MMPM_CLIENTS         (AEE_EOFFSET + 0x176)     ///<  MMPM Reached maximum clients per PD(HAP_MAX_CLIENTS)
#define AEE_EDCVSREGISTER             (AEE_EOFFSET + 0x177)     ///<  ADSP DCVS client registration failed
#define AEE_PDRREGFAIL                (AEE_EOFFSET + 0x178)     ///<  Error Callback Services Registration failed for PD

 /**
 * @}
 */

#define AEE_DEFAULT_PROCESS           (AEE_EOFFSET + 0x180)     ///<  Default process in Guest OS is not present
#define AEE_ENULLCONTEXT              (AEE_EOFFSET + 0x181)     ///<  User NULL context vote
#define AEE_EINVALIDJOB               (AEE_EOFFSET + 0x182)     ///<  AsyncRPC Invalid job
#define AEE_EBUSY                     (AEE_EOFFSET + 0x183)     ///<  AsyncRPC Pending job

 /** @defgroup heaperror Heap error codes
 *  @{
 */

#define E_APPS_BUSY_RETRY_LATER       (AEE_EOFFSET + 0x190)     ///<  Retry because the apps is busy
#define E_HLOS_CAP_REACHED            (AEE_EOFFSET + 0x191)     ///<  cannot allocate any more hlos mem
#define E_DPOOL_CAP_REACHED           (AEE_EOFFSET + 0x192)     ///<  cannot allocate any more physpool mem
#define E_NO_MORE_FREE_SECTIONS       (AEE_EOFFSET + 0x193)     ///<  No more free sections available to grow heap

 /**
 * @}
 */

#endif /* #ifndef AEESTDERR_H */

