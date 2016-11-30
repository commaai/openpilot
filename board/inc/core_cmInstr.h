/**************************************************************************//**
 * @file     core_cmInstr.h
 * @brief    CMSIS Cortex-M Core Instruction Access Header File
 * @version  V2.10
 * @date     19. July 2011
 *
 * @note
 * Copyright (C) 2009-2011 ARM Limited. All rights reserved.
 *
 * @par
 * ARM Limited (ARM) is supplying this software for use with Cortex-M 
 * processor based microcontrollers.  This file can be freely distributed 
 * within development tools that are supporting such ARM based processors. 
 *
 * @par
 * THIS SOFTWARE IS PROVIDED "AS IS".  NO WARRANTIES, WHETHER EXPRESS, IMPLIED
 * OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
 * ARM SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR
 * CONSEQUENTIAL DAMAGES, FOR ANY REASON WHATSOEVER.
 *
 ******************************************************************************/

#ifndef __CORE_CMINSTR_H
#define __CORE_CMINSTR_H


/* ##########################  Core Instruction Access  ######################### */
/** \defgroup CMSIS_Core_InstructionInterface CMSIS Core Instruction Interface
  Access to dedicated instructions
  @{
*/

#if   defined ( __CC_ARM ) /*------------------RealView Compiler -----------------*/
/* ARM armcc specific functions */

#if (__ARMCC_VERSION < 400677)
  #error "Please use ARM Compiler Toolchain V4.0.677 or later!"
#endif


/** \brief  No Operation

    No Operation does nothing. This instruction can be used for code alignment purposes.
 */
#define __NOP                             __nop


/** \brief  Wait For Interrupt

    Wait For Interrupt is a hint instruction that suspends execution
    until one of a number of events occurs.
 */
#define __WFI                             __wfi


/** \brief  Wait For Event

    Wait For Event is a hint instruction that permits the processor to enter
    a low-power state until one of a number of events occurs.
 */
#define __WFE                             __wfe


/** \brief  Send Event

    Send Event is a hint instruction. It causes an event to be signaled to the CPU.
 */
#define __SEV                             __sev


/** \brief  Instruction Synchronization Barrier

    Instruction Synchronization Barrier flushes the pipeline in the processor, 
    so that all instructions following the ISB are fetched from cache or 
    memory, after the instruction has been completed.
 */
#define __ISB()                           __isb(0xF)


/** \brief  Data Synchronization Barrier

    This function acts as a special kind of Data Memory Barrier. 
    It completes when all explicit memory accesses before this instruction complete.
 */
#define __DSB()                           __dsb(0xF)


/** \brief  Data Memory Barrier

    This function ensures the apparent order of the explicit memory operations before 
    and after the instruction, without ensuring their completion.
 */
#define __DMB()                           __dmb(0xF)


/** \brief  Reverse byte order (32 bit)

    This function reverses the byte order in integer value.

    \param [in]    value  Value to reverse
    \return               Reversed value
 */
#define __REV                             __rev


/** \brief  Reverse byte order (16 bit)

    This function reverses the byte order in two unsigned short values.

    \param [in]    value  Value to reverse
    \return               Reversed value
 */
static __INLINE __ASM uint32_t __REV16(uint32_t value)
{
  rev16 r0, r0
  bx lr
}


/** \brief  Reverse byte order in signed short value

    This function reverses the byte order in a signed short value with sign extension to integer.

    \param [in]    value  Value to reverse
    \return               Reversed value
 */
static __INLINE __ASM int32_t __REVSH(int32_t value)
{
  revsh r0, r0
  bx lr
}


#if       (__CORTEX_M >= 0x03)

/** \brief  Reverse bit order of value

    This function reverses the bit order of the given value.

    \param [in]    value  Value to reverse
    \return               Reversed value
 */
#define __RBIT                            __rbit


/** \brief  LDR Exclusive (8 bit)

    This function performs a exclusive LDR command for 8 bit value.

    \param [in]    ptr  Pointer to data
    \return             value of type uint8_t at (*ptr)
 */
#define __LDREXB(ptr)                     ((uint8_t ) __ldrex(ptr))


/** \brief  LDR Exclusive (16 bit)

    This function performs a exclusive LDR command for 16 bit values.

    \param [in]    ptr  Pointer to data
    \return        value of type uint16_t at (*ptr)
 */
#define __LDREXH(ptr)                     ((uint16_t) __ldrex(ptr))


/** \brief  LDR Exclusive (32 bit)

    This function performs a exclusive LDR command for 32 bit values.

    \param [in]    ptr  Pointer to data
    \return        value of type uint32_t at (*ptr)
 */
#define __LDREXW(ptr)                     ((uint32_t ) __ldrex(ptr))


/** \brief  STR Exclusive (8 bit)

    This function performs a exclusive STR command for 8 bit values.

    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
 */
#define __STREXB(value, ptr)              __strex(value, ptr)


/** \brief  STR Exclusive (16 bit)

    This function performs a exclusive STR command for 16 bit values.

    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
 */
#define __STREXH(value, ptr)              __strex(value, ptr)


/** \brief  STR Exclusive (32 bit)

    This function performs a exclusive STR command for 32 bit values.

    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
 */
#define __STREXW(value, ptr)              __strex(value, ptr)


/** \brief  Remove the exclusive lock

    This function removes the exclusive lock which is created by LDREX.

 */
#define __CLREX                           __clrex


/** \brief  Signed Saturate

    This function saturates a signed value.

    \param [in]  value  Value to be saturated
    \param [in]    sat  Bit position to saturate to (1..32)
    \return             Saturated value
 */
#define __SSAT                            __ssat


/** \brief  Unsigned Saturate

    This function saturates an unsigned value.

    \param [in]  value  Value to be saturated
    \param [in]    sat  Bit position to saturate to (0..31)
    \return             Saturated value
 */
#define __USAT                            __usat


/** \brief  Count leading zeros

    This function counts the number of leading zeros of a data value.

    \param [in]  value  Value to count the leading zeros
    \return             number of leading zeros in value
 */
#define __CLZ                             __clz 

#endif /* (__CORTEX_M >= 0x03) */



#elif defined ( __ICCARM__ ) /*------------------ ICC Compiler -------------------*/
/* IAR iccarm specific functions */

#include <cmsis_iar.h>


#elif defined ( __GNUC__ ) /*------------------ GNU Compiler ---------------------*/
/* GNU gcc specific functions */

/** \brief  No Operation

    No Operation does nothing. This instruction can be used for code alignment purposes.
 */
__attribute__( ( always_inline ) ) static __INLINE void __NOP(void)
{
  __ASM volatile ("nop");
}


/** \brief  Wait For Interrupt

    Wait For Interrupt is a hint instruction that suspends execution
    until one of a number of events occurs.
 */
__attribute__( ( always_inline ) ) static __INLINE void __WFI(void)
{
  __ASM volatile ("wfi");
}


/** \brief  Wait For Event

    Wait For Event is a hint instruction that permits the processor to enter
    a low-power state until one of a number of events occurs.
 */
__attribute__( ( always_inline ) ) static __INLINE void __WFE(void)
{
  __ASM volatile ("wfe");
}


/** \brief  Send Event

    Send Event is a hint instruction. It causes an event to be signaled to the CPU.
 */
__attribute__( ( always_inline ) ) static __INLINE void __SEV(void)
{
  __ASM volatile ("sev");
}


/** \brief  Instruction Synchronization Barrier

    Instruction Synchronization Barrier flushes the pipeline in the processor, 
    so that all instructions following the ISB are fetched from cache or 
    memory, after the instruction has been completed.
 */
__attribute__( ( always_inline ) ) static __INLINE void __ISB(void)
{
  __ASM volatile ("isb");
}


/** \brief  Data Synchronization Barrier

    This function acts as a special kind of Data Memory Barrier. 
    It completes when all explicit memory accesses before this instruction complete.
 */
__attribute__( ( always_inline ) ) static __INLINE void __DSB(void)
{
  __ASM volatile ("dsb");
}


/** \brief  Data Memory Barrier

    This function ensures the apparent order of the explicit memory operations before 
    and after the instruction, without ensuring their completion.
 */
__attribute__( ( always_inline ) ) static __INLINE void __DMB(void)
{
  __ASM volatile ("dmb");
}


/** \brief  Reverse byte order (32 bit)

    This function reverses the byte order in integer value.

    \param [in]    value  Value to reverse
    \return               Reversed value
 */
__attribute__( ( always_inline ) ) static __INLINE uint32_t __REV(uint32_t value)
{
  uint32_t result;
  
  __ASM volatile ("rev %0, %1" : "=r" (result) : "r" (value) );
  return(result);
}


/** \brief  Reverse byte order (16 bit)

    This function reverses the byte order in two unsigned short values.

    \param [in]    value  Value to reverse
    \return               Reversed value
 */
__attribute__( ( always_inline ) ) static __INLINE uint32_t __REV16(uint32_t value)
{
  uint32_t result;
  
  __ASM volatile ("rev16 %0, %1" : "=r" (result) : "r" (value) );
  return(result);
}


/** \brief  Reverse byte order in signed short value

    This function reverses the byte order in a signed short value with sign extension to integer.

    \param [in]    value  Value to reverse
    \return               Reversed value
 */
__attribute__( ( always_inline ) ) static __INLINE int32_t __REVSH(int32_t value)
{
  uint32_t result;
  
  __ASM volatile ("revsh %0, %1" : "=r" (result) : "r" (value) );
  return((int32_t)result);
}


#if       (__CORTEX_M >= 0x03)

/** \brief  Reverse bit order of value

    This function reverses the bit order of the given value.

    \param [in]    value  Value to reverse
    \return               Reversed value
 */
__attribute__( ( always_inline ) ) static __INLINE uint32_t __RBIT(uint32_t value)
{
  uint32_t result;
  
   __ASM volatile ("rbit %0, %1" : "=r" (result) : "r" (value) );
   return(result);
}


/** \brief  LDR Exclusive (8 bit)

    This function performs a exclusive LDR command for 8 bit value.

    \param [in]    ptr  Pointer to data
    \return             value of type uint8_t at (*ptr)
 */
__attribute__( ( always_inline ) ) static __INLINE uint8_t __LDREXB(volatile uint8_t *addr)
{
    uint8_t result;
  
   __ASM volatile ("ldrexb %0, [%1]" : "=r" (result) : "r" (addr) );
   return(result);
}


/** \brief  LDR Exclusive (16 bit)

    This function performs a exclusive LDR command for 16 bit values.

    \param [in]    ptr  Pointer to data
    \return        value of type uint16_t at (*ptr)
 */
__attribute__( ( always_inline ) ) static __INLINE uint16_t __LDREXH(volatile uint16_t *addr)
{
    uint16_t result;
  
   __ASM volatile ("ldrexh %0, [%1]" : "=r" (result) : "r" (addr) );
   return(result);
}


/** \brief  LDR Exclusive (32 bit)

    This function performs a exclusive LDR command for 32 bit values.

    \param [in]    ptr  Pointer to data
    \return        value of type uint32_t at (*ptr)
 */
__attribute__( ( always_inline ) ) static __INLINE uint32_t __LDREXW(volatile uint32_t *addr)
{
    uint32_t result;
  
   __ASM volatile ("ldrex %0, [%1]" : "=r" (result) : "r" (addr) );
   return(result);
}


/** \brief  STR Exclusive (8 bit)

    This function performs a exclusive STR command for 8 bit values.

    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
 */
__attribute__( ( always_inline ) ) static __INLINE uint32_t __STREXB(uint8_t value, volatile uint8_t *addr)
{
   uint32_t result;
  
   __ASM volatile ("strexb %0, %2, [%1]" : "=r" (result) : "r" (addr), "r" (value) );
   return(result);
}


/** \brief  STR Exclusive (16 bit)

    This function performs a exclusive STR command for 16 bit values.

    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
 */
__attribute__( ( always_inline ) ) static __INLINE uint32_t __STREXH(uint16_t value, volatile uint16_t *addr)
{
   uint32_t result;
  
   __ASM volatile ("strexh %0, %2, [%1]" : "=r" (result) : "r" (addr), "r" (value) );
   return(result);
}


/** \brief  STR Exclusive (32 bit)

    This function performs a exclusive STR command for 32 bit values.

    \param [in]  value  Value to store
    \param [in]    ptr  Pointer to location
    \return          0  Function succeeded
    \return          1  Function failed
 */
__attribute__( ( always_inline ) ) static __INLINE uint32_t __STREXW(uint32_t value, volatile uint32_t *addr)
{
   uint32_t result;
  
   __ASM volatile ("strex %0, %2, [%1]" : "=r" (result) : "r" (addr), "r" (value) );
   return(result);
}


/** \brief  Remove the exclusive lock

    This function removes the exclusive lock which is created by LDREX.

 */
__attribute__( ( always_inline ) ) static __INLINE void __CLREX(void)
{
  __ASM volatile ("clrex");
}


/** \brief  Signed Saturate

    This function saturates a signed value.

    \param [in]  value  Value to be saturated
    \param [in]    sat  Bit position to saturate to (1..32)
    \return             Saturated value
 */
#define __SSAT(ARG1,ARG2) \
({                          \
  uint32_t __RES, __ARG1 = (ARG1); \
  __ASM ("ssat %0, %1, %2" : "=r" (__RES) :  "I" (ARG2), "r" (__ARG1) ); \
  __RES; \
 })


/** \brief  Unsigned Saturate

    This function saturates an unsigned value.

    \param [in]  value  Value to be saturated
    \param [in]    sat  Bit position to saturate to (0..31)
    \return             Saturated value
 */
#define __USAT(ARG1,ARG2) \
({                          \
  uint32_t __RES, __ARG1 = (ARG1); \
  __ASM ("usat %0, %1, %2" : "=r" (__RES) :  "I" (ARG2), "r" (__ARG1) ); \
  __RES; \
 })


/** \brief  Count leading zeros

    This function counts the number of leading zeros of a data value.

    \param [in]  value  Value to count the leading zeros
    \return             number of leading zeros in value
 */
__attribute__( ( always_inline ) ) static __INLINE uint8_t __CLZ(uint32_t value)
{
  uint8_t result;
  
  __ASM volatile ("clz %0, %1" : "=r" (result) : "r" (value) );
  return(result);
}

#endif /* (__CORTEX_M >= 0x03) */




#elif defined ( __TASKING__ ) /*------------------ TASKING Compiler --------------*/
/* TASKING carm specific functions */

/*
 * The CMSIS functions have been implemented as intrinsics in the compiler.
 * Please use "carm -?i" to get an up to date list of all intrinsics,
 * Including the CMSIS ones.
 */

#endif

/*@}*/ /* end of group CMSIS_Core_InstructionInterface */

#endif /* __CORE_CMINSTR_H */
