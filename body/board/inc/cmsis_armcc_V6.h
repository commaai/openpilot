/**************************************************************************//**
 * @file     cmsis_armcc_V6.h
 * @brief    CMSIS Cortex-M Core Function/Instruction Header File
 * @version  V4.30
 * @date     20. October 2015
 ******************************************************************************/
/* Copyright (c) 2009 - 2015 ARM LIMITED

   All rights reserved.
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
   - Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   - Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   - Neither the name of ARM nor the names of its contributors may be used
     to endorse or promote products derived from this software without
     specific prior written permission.
   *
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS AND CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
   ---------------------------------------------------------------------------*/


#ifndef __CMSIS_ARMCC_V6_H
#define __CMSIS_ARMCC_V6_H


/* ###########################  Core Function Access  ########################### */
/** \ingroup  CMSIS_Core_FunctionInterface
    \defgroup CMSIS_Core_RegAccFunctions CMSIS Core Register Access Functions
  @{
 */

/**
  \brief   Enable IRQ Interrupts
  \details Enables IRQ interrupts by clearing the I-bit in the CPSR.
           Can only be executed in Privileged modes.
 */
__attribute__((always_inline)) __STATIC_INLINE void __enable_irq(void)
{
  __ASM volatile ("cpsie i" : : : "memory");
}


/**
  \brief   Disable IRQ Interrupts
  \details Disables IRQ interrupts by setting the I-bit in the CPSR.
           Can only be executed in Privileged modes.
 */
__attribute__((always_inline)) __STATIC_INLINE void __disable_irq(void)
{
  __ASM volatile ("cpsid i" : : : "memory");
}


/**
  \brief   Get Control Register
  \details Returns the content of the Control Register.
  \return               Control Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_CONTROL(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, control" : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get Control Register (non-secure)
  \details Returns the content of the non-secure Control Register when in secure mode.
  \return               non-secure Control Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_CONTROL_NS(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, control_ns" : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Set Control Register
  \details Writes the given value to the Control Register.
  \param [in]    control  Control Register value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __set_CONTROL(uint32_t control)
{
  __ASM volatile ("MSR control, %0" : : "r" (control) : "memory");
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Set Control Register (non-secure)
  \details Writes the given value to the non-secure Control Register when in secure state.
  \param [in]    control  Control Register value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_CONTROL_NS(uint32_t control)
{
  __ASM volatile ("MSR control_ns, %0" : : "r" (control) : "memory");
}
#endif


/**
  \brief   Get IPSR Register
  \details Returns the content of the IPSR Register.
  \return               IPSR Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_IPSR(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, ipsr" : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get IPSR Register (non-secure)
  \details Returns the content of the non-secure IPSR Register when in secure state.
  \return               IPSR Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_IPSR_NS(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, ipsr_ns" : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Get APSR Register
  \details Returns the content of the APSR Register.
  \return               APSR Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_APSR(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, apsr" : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get APSR Register (non-secure)
  \details Returns the content of the non-secure APSR Register when in secure state.
  \return               APSR Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_APSR_NS(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, apsr_ns" : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Get xPSR Register
  \details Returns the content of the xPSR Register.
  \return               xPSR Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_xPSR(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, xpsr" : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get xPSR Register (non-secure)
  \details Returns the content of the non-secure xPSR Register when in secure state.
  \return               xPSR Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_xPSR_NS(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, xpsr_ns" : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Get Process Stack Pointer
  \details Returns the current value of the Process Stack Pointer (PSP).
  \return               PSP Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_PSP(void)
{
  register uint32_t result;

  __ASM volatile ("MRS %0, psp"  : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get Process Stack Pointer (non-secure)
  \details Returns the current value of the non-secure Process Stack Pointer (PSP) when in secure state.
  \return               PSP Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_PSP_NS(void)
{
  register uint32_t result;

  __ASM volatile ("MRS %0, psp_ns"  : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Set Process Stack Pointer
  \details Assigns the given value to the Process Stack Pointer (PSP).
  \param [in]    topOfProcStack  Process Stack Pointer value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __set_PSP(uint32_t topOfProcStack)
{
  __ASM volatile ("MSR psp, %0" : : "r" (topOfProcStack) : "sp");
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Set Process Stack Pointer (non-secure)
  \details Assigns the given value to the non-secure Process Stack Pointer (PSP) when in secure state.
  \param [in]    topOfProcStack  Process Stack Pointer value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_PSP_NS(uint32_t topOfProcStack)
{
  __ASM volatile ("MSR psp_ns, %0" : : "r" (topOfProcStack) : "sp");
}
#endif


/**
  \brief   Get Main Stack Pointer
  \details Returns the current value of the Main Stack Pointer (MSP).
  \return               MSP Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_MSP(void)
{
  register uint32_t result;

  __ASM volatile ("MRS %0, msp" : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get Main Stack Pointer (non-secure)
  \details Returns the current value of the non-secure Main Stack Pointer (MSP) when in secure state.
  \return               MSP Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_MSP_NS(void)
{
  register uint32_t result;

  __ASM volatile ("MRS %0, msp_ns" : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Set Main Stack Pointer
  \details Assigns the given value to the Main Stack Pointer (MSP).
  \param [in]    topOfMainStack  Main Stack Pointer value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __set_MSP(uint32_t topOfMainStack)
{
  __ASM volatile ("MSR msp, %0" : : "r" (topOfMainStack) : "sp");
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Set Main Stack Pointer (non-secure)
  \details Assigns the given value to the non-secure Main Stack Pointer (MSP) when in secure state.
  \param [in]    topOfMainStack  Main Stack Pointer value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_MSP_NS(uint32_t topOfMainStack)
{
  __ASM volatile ("MSR msp_ns, %0" : : "r" (topOfMainStack) : "sp");
}
#endif


/**
  \brief   Get Priority Mask
  \details Returns the current state of the priority mask bit from the Priority Mask Register.
  \return               Priority Mask value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_PRIMASK(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, primask" : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get Priority Mask (non-secure)
  \details Returns the current state of the non-secure priority mask bit from the Priority Mask Register when in secure state.
  \return               Priority Mask value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_PRIMASK_NS(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, primask_ns" : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Set Priority Mask
  \details Assigns the given value to the Priority Mask Register.
  \param [in]    priMask  Priority Mask
 */
__attribute__((always_inline)) __STATIC_INLINE void __set_PRIMASK(uint32_t priMask)
{
  __ASM volatile ("MSR primask, %0" : : "r" (priMask) : "memory");
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Set Priority Mask (non-secure)
  \details Assigns the given value to the non-secure Priority Mask Register when in secure state.
  \param [in]    priMask  Priority Mask
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_PRIMASK_NS(uint32_t priMask)
{
  __ASM volatile ("MSR primask_ns, %0" : : "r" (priMask) : "memory");
}
#endif


#if ((__ARM_ARCH_7M__ == 1U) || (__ARM_ARCH_7EM__ == 1U) || (__ARM_ARCH_8M__ == 1U))  /* ToDo:  ARMCC_V6: check if this is ok for cortex >=3 */

/**
  \brief   Enable FIQ
  \details Enables FIQ interrupts by clearing the F-bit in the CPSR.
           Can only be executed in Privileged modes.
 */
__attribute__((always_inline)) __STATIC_INLINE void __enable_fault_irq(void)
{
  __ASM volatile ("cpsie f" : : : "memory");
}


/**
  \brief   Disable FIQ
  \details Disables FIQ interrupts by setting the F-bit in the CPSR.
           Can only be executed in Privileged modes.
 */
__attribute__((always_inline)) __STATIC_INLINE void __disable_fault_irq(void)
{
  __ASM volatile ("cpsid f" : : : "memory");
}


/**
  \brief   Get Base Priority
  \details Returns the current value of the Base Priority register.
  \return               Base Priority register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_BASEPRI(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, basepri" : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get Base Priority (non-secure)
  \details Returns the current value of the non-secure Base Priority register when in secure state.
  \return               Base Priority register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_BASEPRI_NS(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, basepri_ns" : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Set Base Priority
  \details Assigns the given value to the Base Priority register.
  \param [in]    basePri  Base Priority value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __set_BASEPRI(uint32_t value)
{
  __ASM volatile ("MSR basepri, %0" : : "r" (value) : "memory");
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Set Base Priority (non-secure)
  \details Assigns the given value to the non-secure Base Priority register when in secure state.
  \param [in]    basePri  Base Priority value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_BASEPRI_NS(uint32_t value)
{
  __ASM volatile ("MSR basepri_ns, %0" : : "r" (value) : "memory");
}
#endif


/**
  \brief   Set Base Priority with condition
  \details Assigns the given value to the Base Priority register only if BASEPRI masking is disabled,
           or the new value increases the BASEPRI priority level.
  \param [in]    basePri  Base Priority value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __set_BASEPRI_MAX(uint32_t value)
{
  __ASM volatile ("MSR basepri_max, %0" : : "r" (value) : "memory");
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Set Base Priority with condition (non_secure)
  \details Assigns the given value to the non-secure Base Priority register when in secure state only if BASEPRI masking is disabled,
	       or the new value increases the BASEPRI priority level.
  \param [in]    basePri  Base Priority value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_BASEPRI_MAX_NS(uint32_t value)
{
  __ASM volatile ("MSR basepri_max_ns, %0" : : "r" (value) : "memory");
}
#endif


/**
  \brief   Get Fault Mask
  \details Returns the current value of the Fault Mask register.
  \return               Fault Mask register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_FAULTMASK(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, faultmask" : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get Fault Mask (non-secure)
  \details Returns the current value of the non-secure Fault Mask register when in secure state.
  \return               Fault Mask register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_FAULTMASK_NS(void)
{
  uint32_t result;

  __ASM volatile ("MRS %0, faultmask_ns" : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Set Fault Mask
  \details Assigns the given value to the Fault Mask register.
  \param [in]    faultMask  Fault Mask value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __set_FAULTMASK(uint32_t faultMask)
{
  __ASM volatile ("MSR faultmask, %0" : : "r" (faultMask) : "memory");
}


#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Set Fault Mask (non-secure)
  \details Assigns the given value to the non-secure Fault Mask register when in secure state.
  \param [in]    faultMask  Fault Mask value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_FAULTMASK_NS(uint32_t faultMask)
{
  __ASM volatile ("MSR faultmask_ns, %0" : : "r" (faultMask) : "memory");
}
#endif


#endif /* ((__ARM_ARCH_7M__ == 1U) || (__ARM_ARCH_8M__ == 1U)) */


#if (__ARM_ARCH_8M__ == 1U)

/**
  \brief   Get Process Stack Pointer Limit
  \details Returns the current value of the Process Stack Pointer Limit (PSPLIM).
  \return               PSPLIM Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_PSPLIM(void)
{
  register uint32_t result;

  __ASM volatile ("MRS %0, psplim"  : "=r" (result) );
  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U) && (__ARM_ARCH_PROFILE == 'M')     /* ToDo:  ARMCC_V6: check predefined macro for mainline */
/**
  \brief   Get Process Stack Pointer Limit (non-secure)
  \details Returns the current value of the non-secure Process Stack Pointer Limit (PSPLIM) when in secure state.
  \return               PSPLIM Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_PSPLIM_NS(void)
{
  register uint32_t result;

  __ASM volatile ("MRS %0, psplim_ns"  : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Set Process Stack Pointer Limit
  \details Assigns the given value to the Process Stack Pointer Limit (PSPLIM).
  \param [in]    ProcStackPtrLimit  Process Stack Pointer Limit value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __set_PSPLIM(uint32_t ProcStackPtrLimit)
{
  __ASM volatile ("MSR psplim, %0" : : "r" (ProcStackPtrLimit));
}


#if  (__ARM_FEATURE_CMSE == 3U) && (__ARM_ARCH_PROFILE == 'M')     /* ToDo:  ARMCC_V6: check predefined macro for mainline */
/**
  \brief   Set Process Stack Pointer (non-secure)
  \details Assigns the given value to the non-secure Process Stack Pointer Limit (PSPLIM) when in secure state.
  \param [in]    ProcStackPtrLimit  Process Stack Pointer Limit value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_PSPLIM_NS(uint32_t ProcStackPtrLimit)
{
  __ASM volatile ("MSR psplim_ns, %0\n" : : "r" (ProcStackPtrLimit));
}
#endif


/**
  \brief   Get Main Stack Pointer Limit
  \details Returns the current value of the Main Stack Pointer Limit (MSPLIM).
  \return               MSPLIM Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_MSPLIM(void)
{
  register uint32_t result;

  __ASM volatile ("MRS %0, msplim" : "=r" (result) );

  return(result);
}


#if  (__ARM_FEATURE_CMSE == 3U) && (__ARM_ARCH_PROFILE == 'M')     /* ToDo:  ARMCC_V6: check predefined macro for mainline */
/**
  \brief   Get Main Stack Pointer Limit (non-secure)
  \details Returns the current value of the non-secure Main Stack Pointer Limit(MSPLIM) when in secure state.
  \return               MSPLIM Register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_MSPLIM_NS(void)
{
  register uint32_t result;

  __ASM volatile ("MRS %0, msplim_ns" : "=r" (result) );
  return(result);
}
#endif


/**
  \brief   Set Main Stack Pointer Limit
  \details Assigns the given value to the Main Stack Pointer Limit (MSPLIM).
  \param [in]    MainStackPtrLimit  Main Stack Pointer Limit value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __set_MSPLIM(uint32_t MainStackPtrLimit)
{
  __ASM volatile ("MSR msplim, %0" : : "r" (MainStackPtrLimit));
}


#if  (__ARM_FEATURE_CMSE == 3U) && (__ARM_ARCH_PROFILE == 'M')     /* ToDo:  ARMCC_V6: check predefined macro for mainline */
/**
  \brief   Set Main Stack Pointer Limit (non-secure)
  \details Assigns the given value to the non-secure Main Stack Pointer Limit (MSPLIM) when in secure state.
  \param [in]    MainStackPtrLimit  Main Stack Pointer value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_MSPLIM_NS(uint32_t MainStackPtrLimit)
{
  __ASM volatile ("MSR msplim_ns, %0" : : "r" (MainStackPtrLimit));
}
#endif

#endif /* (__ARM_ARCH_8M__ == 1U) */


#if ((__ARM_ARCH_7EM__ == 1U) || (__ARM_ARCH_8M__ == 1U))  /* ToDo:  ARMCC_V6: check if this is ok for cortex >=4 */

/**
  \brief   Get FPSCR
  \details eturns the current value of the Floating Point Status/Control register.
  \return               Floating Point Status/Control register value
 */
#define __get_FPSCR      __builtin_arm_get_fpscr
#if 0
__attribute__((always_inline)) __STATIC_INLINE uint32_t __get_FPSCR(void)
{
#if (__FPU_PRESENT == 1U) && (__FPU_USED == 1U)
  uint32_t result;

  __ASM volatile ("");                                 /* Empty asm statement works as a scheduling barrier */
  __ASM volatile ("VMRS %0, fpscr" : "=r" (result) );
  __ASM volatile ("");
  return(result);
#else
   return(0);
#endif
}
#endif

#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get FPSCR (non-secure)
  \details Returns the current value of the non-secure Floating Point Status/Control register when in secure state.
  \return               Floating Point Status/Control register value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __TZ_get_FPSCR_NS(void)
{
#if (__FPU_PRESENT == 1U) && (__FPU_USED == 1U)
  uint32_t result;

  __ASM volatile ("");                                 /* Empty asm statement works as a scheduling barrier */
  __ASM volatile ("VMRS %0, fpscr_ns" : "=r" (result) );
  __ASM volatile ("");
  return(result);
#else
   return(0);
#endif
}
#endif


/**
  \brief   Set FPSCR
  \details Assigns the given value to the Floating Point Status/Control register.
  \param [in]    fpscr  Floating Point Status/Control value to set
 */
#define __set_FPSCR      __builtin_arm_set_fpscr
#if 0
__attribute__((always_inline)) __STATIC_INLINE void __set_FPSCR(uint32_t fpscr)
{
#if (__FPU_PRESENT == 1U) && (__FPU_USED == 1U)
  __ASM volatile ("");                                 /* Empty asm statement works as a scheduling barrier */
  __ASM volatile ("VMSR fpscr, %0" : : "r" (fpscr) : "vfpcc");
  __ASM volatile ("");
#endif
}
#endif

#if  (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Set FPSCR (non-secure)
  \details Assigns the given value to the non-secure Floating Point Status/Control register when in secure state.
  \param [in]    fpscr  Floating Point Status/Control value to set
 */
__attribute__((always_inline)) __STATIC_INLINE void __TZ_set_FPSCR_NS(uint32_t fpscr)
{
#if (__FPU_PRESENT == 1U) && (__FPU_USED == 1U)
  __ASM volatile ("");                                 /* Empty asm statement works as a scheduling barrier */
  __ASM volatile ("VMSR fpscr_ns, %0" : : "r" (fpscr) : "vfpcc");
  __ASM volatile ("");
#endif
}
#endif

#endif /* ((__ARM_ARCH_7EM__ == 1U) || (__ARM_ARCH_8M__ == 1U)) */



/*@} end of CMSIS_Core_RegAccFunctions */


/* ##########################  Core Instruction Access  ######################### */
/** \defgroup CMSIS_Core_InstructionInterface CMSIS Core Instruction Interface
  Access to dedicated instructions
  @{
*/

/* Define macros for porting to both thumb1 and thumb2.
 * For thumb1, use low register (r0-r7), specified by constraint "l"
 * Otherwise, use general registers, specified by constraint "r" */
#if defined (__thumb__) && !defined (__thumb2__)
#define __CMSIS_GCC_OUT_REG(r) "=l" (r)
#define __CMSIS_GCC_USE_REG(r) "l" (r)
#else
#define __CMSIS_GCC_OUT_REG(r) "=r" (r)
#define __CMSIS_GCC_USE_REG(r) "r" (r)
#endif

/**
  \brief   No Operation
  \details No Operation does nothing. This instruction can be used for code alignment purposes.
 */
#define __NOP          __builtin_arm_nop

/**
  \brief   Wait For Interrupt
  \details Wait For Interrupt is a hint instruction that suspends execution until one of a number of events occurs.
 */
#define __WFI          __builtin_arm_wfi


/**
  \brief   Wait For Event
  \details Wait For Event is a hint instruction that permits the processor to enter
           a low-power state until one of a number of events occurs.
 */
#define __WFE          __builtin_arm_wfe


/**
  \brief   Send Event
  \details Send Event is a hint instruction. It causes an event to be signaled to the CPU.
 */
#define __SEV          __builtin_arm_sev


/**
  \brief   Instruction Synchronization Barrier
  \details Instruction Synchronization Barrier flushes the pipeline in the processor,
           so that all instructions following the ISB are fetched from cache or memory,
           after the instruction has been completed.
 */
#define __ISB()        __builtin_arm_isb(0xF);

/**
  \brief   Data Synchronization Barrier
  \details Acts as a special kind of Data Memory Barrier.
           It completes when all explicit memory accesses before this instruction complete.
 */
#define __DSB()        __builtin_arm_dsb(0xF);


/**
  \brief   Data Memory Barrier
  \details Ensures the apparent order of the explicit memory operations before
           and after the instruction, without ensuring their completion.
 */
#define __DMB()        __builtin_arm_dmb(0xF);


/**
  \brief   Reverse byte order (32 bit)
  \details Reverses the byte order in integer value.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __REV          __builtin_bswap32


/**
  \brief   Reverse byte order (16 bit)
  \details Reverses the byte order in two unsigned short values.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
#define __REV16          __builtin_bswap16                           /* ToDo:  ARMCC_V6: check if __builtin_bswap16 could be used */
#if 0
__attribute__((always_inline)) __STATIC_INLINE uint32_t __REV16(uint32_t value)
{
  uint32_t result;

  __ASM volatile ("rev16 %0, %1" : __CMSIS_GCC_OUT_REG (result) : __CMSIS_GCC_USE_REG (value) );
  return(result);
}
#endif


/**
  \brief   Reverse byte order in signed short value
  \details Reverses the byte order in a signed short value with sign extension to integer.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
                                                          /* ToDo:  ARMCC_V6: check if __builtin_bswap16 could be used */
__attribute__((always_inline)) __STATIC_INLINE int32_t __REVSH(int32_t value)
{
  int32_t result;

  __ASM volatile ("revsh %0, %1" : __CMSIS_GCC_OUT_REG (result) : __CMSIS_GCC_USE_REG (value) );
  return(result);
}


/**
  \brief   Rotate Right in unsigned value (32 bit)
  \details Rotate Right (immediate) provides the value of the contents of a register rotated by a variable number of bits.
  \param [in]    op1  Value to rotate
  \param [in]    op2  Number of Bits to rotate
  \return               Rotated value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __ROR(uint32_t op1, uint32_t op2)
{
  return (op1 >> op2) | (op1 << (32U - op2));
}


/**
  \brief   Breakpoint
  \details Causes the processor to enter Debug state.
            Debug tools can use this to investigate system state when the instruction at a particular address is reached.
    \param [in]    value  is ignored by the processor.
                   If required, a debugger can use it to store additional information about the breakpoint.
 */
#define __BKPT(value)                       __ASM volatile ("bkpt "#value)


/**
  \brief   Reverse bit order of value
  \details Reverses the bit order of the given value.
  \param [in]    value  Value to reverse
  \return               Reversed value
 */
                                                          /* ToDo:  ARMCC_V6: check if __builtin_arm_rbit is supported */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __RBIT(uint32_t value)
{
  uint32_t result;

#if ((__ARM_ARCH_7M__ == 1U) || (__ARM_ARCH_7EM__ == 1U) || (__ARM_ARCH_8M__ == 1U))  /* ToDo:  ARMCC_V6: check if this is ok for cortex >=3 */
   __ASM volatile ("rbit %0, %1" : "=r" (result) : "r" (value) );
#else
  int32_t s = 4 /*sizeof(v)*/ * 8 - 1; /* extra shift needed at end */

  result = value;                      /* r will be reversed bits of v; first get LSB of v */
  for (value >>= 1U; value; value >>= 1U)
  {
    result <<= 1U;
    result |= value & 1U;
    s--;
  }
  result <<= s;                        /* shift when v's highest bits are zero */
#endif
  return(result);
}


/**
  \brief   Count leading zeros
  \details Counts the number of leading zeros of a data value.
  \param [in]  value  Value to count the leading zeros
  \return             number of leading zeros in value
 */
#define __CLZ             __builtin_clz


#if ((__ARM_ARCH_7M__ == 1U) || (__ARM_ARCH_7EM__ == 1U) || (__ARM_ARCH_8M__ == 1U))  /* ToDo:  ARMCC_V6: check if this is ok for cortex >=3 */

/**
  \brief   LDR Exclusive (8 bit)
  \details Executes a exclusive LDR instruction for 8 bit value.
  \param [in]    ptr  Pointer to data
  \return             value of type uint8_t at (*ptr)
 */
#define __LDREXB        (uint8_t)__builtin_arm_ldrex


/**
  \brief   LDR Exclusive (16 bit)
  \details Executes a exclusive LDR instruction for 16 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint16_t at (*ptr)
 */
#define __LDREXH        (uint16_t)__builtin_arm_ldrex


/**
  \brief   LDR Exclusive (32 bit)
  \details Executes a exclusive LDR instruction for 32 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint32_t at (*ptr)
 */
#define __LDREXW        (uint32_t)__builtin_arm_ldrex


/**
  \brief   STR Exclusive (8 bit)
  \details Executes a exclusive STR instruction for 8 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#define __STREXB        (uint32_t)__builtin_arm_strex


/**
  \brief   STR Exclusive (16 bit)
  \details Executes a exclusive STR instruction for 16 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#define __STREXH        (uint32_t)__builtin_arm_strex


/**
  \brief   STR Exclusive (32 bit)
  \details Executes a exclusive STR instruction for 32 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#define __STREXW        (uint32_t)__builtin_arm_strex


/**
  \brief   Remove the exclusive lock
  \details Removes the exclusive lock which is created by LDREX.
 */
#define __CLREX             __builtin_arm_clrex


/**
  \brief   Signed Saturate
  \details Saturates a signed value.
  \param [in]  value  Value to be saturated
  \param [in]    sat  Bit position to saturate to (1..32)
  \return             Saturated value
 */
/*#define __SSAT             __builtin_arm_ssat*/
#define __SSAT(ARG1,ARG2) \
({                          \
  int32_t __RES, __ARG1 = (ARG1); \
  __ASM ("ssat %0, %1, %2" : "=r" (__RES) :  "I" (ARG2), "r" (__ARG1) ); \
  __RES; \
 })


/**
  \brief   Unsigned Saturate
  \details Saturates an unsigned value.
  \param [in]  value  Value to be saturated
  \param [in]    sat  Bit position to saturate to (0..31)
  \return             Saturated value
 */
#define __USAT             __builtin_arm_usat
#if 0
#define __USAT(ARG1,ARG2) \
({                          \
  uint32_t __RES, __ARG1 = (ARG1); \
  __ASM ("usat %0, %1, %2" : "=r" (__RES) :  "I" (ARG2), "r" (__ARG1) ); \
  __RES; \
 })
#endif


/**
  \brief   Rotate Right with Extend (32 bit)
  \details Moves each bit of a bitstring right by one bit.
           The carry input is shifted in at the left end of the bitstring.
  \param [in]    value  Value to rotate
  \return               Rotated value
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __RRX(uint32_t value)
{
  uint32_t result;

  __ASM volatile ("rrx %0, %1" : __CMSIS_GCC_OUT_REG (result) : __CMSIS_GCC_USE_REG (value) );
  return(result);
}


/**
  \brief   LDRT Unprivileged (8 bit)
  \details Executes a Unprivileged LDRT instruction for 8 bit value.
  \param [in]    ptr  Pointer to data
  \return             value of type uint8_t at (*ptr)
 */
__attribute__((always_inline)) __STATIC_INLINE uint8_t __LDRBT(volatile uint8_t *ptr)
{
    uint32_t result;

   __ASM volatile ("ldrbt %0, %1" : "=r" (result) : "Q" (*ptr) );
   return ((uint8_t) result);    /* Add explicit type cast here */
}


/**
  \brief   LDRT Unprivileged (16 bit)
  \details Executes a Unprivileged LDRT instruction for 16 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint16_t at (*ptr)
 */
__attribute__((always_inline)) __STATIC_INLINE uint16_t __LDRHT(volatile uint16_t *ptr)
{
    uint32_t result;

   __ASM volatile ("ldrht %0, %1" : "=r" (result) : "Q" (*ptr) );
   return ((uint16_t) result);    /* Add explicit type cast here */
}


/**
  \brief   LDRT Unprivileged (32 bit)
  \details Executes a Unprivileged LDRT instruction for 32 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint32_t at (*ptr)
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __LDRT(volatile uint32_t *ptr)
{
    uint32_t result;

   __ASM volatile ("ldrt %0, %1" : "=r" (result) : "Q" (*ptr) );
   return(result);
}


/**
  \brief   STRT Unprivileged (8 bit)
  \details Executes a Unprivileged STRT instruction for 8 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
 */
__attribute__((always_inline)) __STATIC_INLINE void __STRBT(uint8_t value, volatile uint8_t *ptr)
{
   __ASM volatile ("strbt %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) );
}


/**
  \brief   STRT Unprivileged (16 bit)
  \details Executes a Unprivileged STRT instruction for 16 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
 */
__attribute__((always_inline)) __STATIC_INLINE void __STRHT(uint16_t value, volatile uint16_t *ptr)
{
   __ASM volatile ("strht %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) );
}


/**
  \brief   STRT Unprivileged (32 bit)
  \details Executes a Unprivileged STRT instruction for 32 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
 */
__attribute__((always_inline)) __STATIC_INLINE void __STRT(uint32_t value, volatile uint32_t *ptr)
{
   __ASM volatile ("strt %1, %0" : "=Q" (*ptr) : "r" (value) );
}

#endif /* ((__ARM_ARCH_7M__ == 1U) || (__ARM_ARCH_7EM__ == 1U) || (__ARM_ARCH_8M__ == 1U)) */


#if (__ARM_ARCH_8M__ == 1U)

/**
  \brief   Load-Acquire (8 bit)
  \details Executes a LDAB instruction for 8 bit value.
  \param [in]    ptr  Pointer to data
  \return             value of type uint8_t at (*ptr)
 */
__attribute__((always_inline)) __STATIC_INLINE uint8_t __LDAB(volatile uint8_t *ptr)
{
    uint32_t result;

   __ASM volatile ("ldab %0, %1" : "=r" (result) : "Q" (*ptr) );
   return ((uint8_t) result);
}


/**
  \brief   Load-Acquire (16 bit)
  \details Executes a LDAH instruction for 16 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint16_t at (*ptr)
 */
__attribute__((always_inline)) __STATIC_INLINE uint16_t __LDAH(volatile uint16_t *ptr)
{
    uint32_t result;

   __ASM volatile ("ldah %0, %1" : "=r" (result) : "Q" (*ptr) );
   return ((uint16_t) result);
}


/**
  \brief   Load-Acquire (32 bit)
  \details Executes a LDA instruction for 32 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint32_t at (*ptr)
 */
__attribute__((always_inline)) __STATIC_INLINE uint32_t __LDA(volatile uint32_t *ptr)
{
    uint32_t result;

   __ASM volatile ("lda %0, %1" : "=r" (result) : "Q" (*ptr) );
   return(result);
}


/**
  \brief   Store-Release (8 bit)
  \details Executes a STLB instruction for 8 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
 */
__attribute__((always_inline)) __STATIC_INLINE void __STLB(uint8_t value, volatile uint8_t *ptr)
{
   __ASM volatile ("stlb %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) );
}


/**
  \brief   Store-Release (16 bit)
  \details Executes a STLH instruction for 16 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
 */
__attribute__((always_inline)) __STATIC_INLINE void __STLH(uint16_t value, volatile uint16_t *ptr)
{
   __ASM volatile ("stlh %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) );
}


/**
  \brief   Store-Release (32 bit)
  \details Executes a STL instruction for 32 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
 */
__attribute__((always_inline)) __STATIC_INLINE void __STL(uint32_t value, volatile uint32_t *ptr)
{
   __ASM volatile ("stl %1, %0" : "=Q" (*ptr) : "r" ((uint32_t)value) );
}


/**
  \brief   Load-Acquire Exclusive (8 bit)
  \details Executes a LDAB exclusive instruction for 8 bit value.
  \param [in]    ptr  Pointer to data
  \return             value of type uint8_t at (*ptr)
 */
#define     __LDAEXB                 (uint8_t)__builtin_arm_ldaex


/**
  \brief   Load-Acquire Exclusive (16 bit)
  \details Executes a LDAH exclusive instruction for 16 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint16_t at (*ptr)
 */
#define     __LDAEXH                 (uint16_t)__builtin_arm_ldaex


/**
  \brief   Load-Acquire Exclusive (32 bit)
  \details Executes a LDA exclusive instruction for 32 bit values.
  \param [in]    ptr  Pointer to data
  \return        value of type uint32_t at (*ptr)
 */
#define     __LDAEX                  (uint32_t)__builtin_arm_ldaex


/**
  \brief   Store-Release Exclusive (8 bit)
  \details Executes a STLB exclusive instruction for 8 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#define     __STLEXB                 (uint32_t)__builtin_arm_stlex


/**
  \brief   Store-Release Exclusive (16 bit)
  \details Executes a STLH exclusive instruction for 16 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#define     __STLEXH                 (uint32_t)__builtin_arm_stlex


/**
  \brief   Store-Release Exclusive (32 bit)
  \details Executes a STL exclusive instruction for 32 bit values.
  \param [in]  value  Value to store
  \param [in]    ptr  Pointer to location
  \return          0  Function succeeded
  \return          1  Function failed
 */
#define     __STLEX                  (uint32_t)__builtin_arm_stlex

#endif /* (__ARM_ARCH_8M__ == 1U) */

/*@}*/ /* end of group CMSIS_Core_InstructionInterface */


/* ###################  Compiler specific Intrinsics  ########################### */
/** \defgroup CMSIS_SIMD_intrinsics CMSIS SIMD Intrinsics
  Access to dedicated SIMD instructions
  @{
*/

#if (__ARM_FEATURE_DSP == 1U)        /* ToDo:  ARMCC_V6: This should be ARCH >= ARMv7-M + SIMD */

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __QADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SHADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UQADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UHADD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhadd8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}


__attribute__((always_inline)) __STATIC_INLINE uint32_t __SSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("ssub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __QSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qsub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SHSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shsub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __USUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("usub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UQSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqsub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UHSUB8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhsub8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}


__attribute__((always_inline)) __STATIC_INLINE uint32_t __SADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __QADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SHADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UQADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UHADD16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhadd16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("ssub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __QSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qsub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SHSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shsub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __USUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("usub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UQSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqsub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UHSUB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhsub16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __QASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SHASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UQASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UHASX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhasx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("ssax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __QSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("qsax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SHSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("shsax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __USAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("usax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UQSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uqsax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UHSAX(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uhsax %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __USAD8(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("usad8 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __USADA8(uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("usada8 %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

#define __SSAT16(ARG1,ARG2) \
({                          \
  uint32_t __RES, __ARG1 = (ARG1); \
  __ASM ("ssat16 %0, %1, %2" : "=r" (__RES) :  "I" (ARG2), "r" (__ARG1) ); \
  __RES; \
 })

#define __USAT16(ARG1,ARG2) \
({                          \
  uint32_t __RES, __ARG1 = (ARG1); \
  __ASM ("usat16 %0, %1, %2" : "=r" (__RES) :  "I" (ARG2), "r" (__ARG1) ); \
  __RES; \
 })

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UXTB16(uint32_t op1)
{
  uint32_t result;

  __ASM volatile ("uxtb16 %0, %1" : "=r" (result) : "r" (op1));
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __UXTAB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("uxtab16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SXTB16(uint32_t op1)
{
  uint32_t result;

  __ASM volatile ("sxtb16 %0, %1" : "=r" (result) : "r" (op1));
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SXTAB16(uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sxtab16 %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SMUAD  (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("smuad %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SMUADX (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("smuadx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SMLAD (uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("smlad %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SMLADX (uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("smladx %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint64_t __SMLALD (uint32_t op1, uint32_t op2, uint64_t acc)
{
  union llreg_u{
    uint32_t w32[2];
    uint64_t w64;
  } llr;
  llr.w64 = acc;

#ifndef __ARMEB__   /* Little endian */
  __ASM volatile ("smlald %0, %1, %2, %3" : "=r" (llr.w32[0]), "=r" (llr.w32[1]): "r" (op1), "r" (op2) , "0" (llr.w32[0]), "1" (llr.w32[1]) );
#else               /* Big endian */
  __ASM volatile ("smlald %0, %1, %2, %3" : "=r" (llr.w32[1]), "=r" (llr.w32[0]): "r" (op1), "r" (op2) , "0" (llr.w32[1]), "1" (llr.w32[0]) );
#endif

  return(llr.w64);
}

__attribute__((always_inline)) __STATIC_INLINE uint64_t __SMLALDX (uint32_t op1, uint32_t op2, uint64_t acc)
{
  union llreg_u{
    uint32_t w32[2];
    uint64_t w64;
  } llr;
  llr.w64 = acc;

#ifndef __ARMEB__   /* Little endian */
  __ASM volatile ("smlaldx %0, %1, %2, %3" : "=r" (llr.w32[0]), "=r" (llr.w32[1]): "r" (op1), "r" (op2) , "0" (llr.w32[0]), "1" (llr.w32[1]) );
#else               /* Big endian */
  __ASM volatile ("smlaldx %0, %1, %2, %3" : "=r" (llr.w32[1]), "=r" (llr.w32[0]): "r" (op1), "r" (op2) , "0" (llr.w32[1]), "1" (llr.w32[0]) );
#endif

  return(llr.w64);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SMUSD  (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("smusd %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SMUSDX (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("smusdx %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SMLSD (uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("smlsd %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SMLSDX (uint32_t op1, uint32_t op2, uint32_t op3)
{
  uint32_t result;

  __ASM volatile ("smlsdx %0, %1, %2, %3" : "=r" (result) : "r" (op1), "r" (op2), "r" (op3) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE uint64_t __SMLSLD (uint32_t op1, uint32_t op2, uint64_t acc)
{
  union llreg_u{
    uint32_t w32[2];
    uint64_t w64;
  } llr;
  llr.w64 = acc;

#ifndef __ARMEB__   /* Little endian */
  __ASM volatile ("smlsld %0, %1, %2, %3" : "=r" (llr.w32[0]), "=r" (llr.w32[1]): "r" (op1), "r" (op2) , "0" (llr.w32[0]), "1" (llr.w32[1]) );
#else               /* Big endian */
  __ASM volatile ("smlsld %0, %1, %2, %3" : "=r" (llr.w32[1]), "=r" (llr.w32[0]): "r" (op1), "r" (op2) , "0" (llr.w32[1]), "1" (llr.w32[0]) );
#endif

  return(llr.w64);
}

__attribute__((always_inline)) __STATIC_INLINE uint64_t __SMLSLDX (uint32_t op1, uint32_t op2, uint64_t acc)
{
  union llreg_u{
    uint32_t w32[2];
    uint64_t w64;
  } llr;
  llr.w64 = acc;

#ifndef __ARMEB__   /* Little endian */
  __ASM volatile ("smlsldx %0, %1, %2, %3" : "=r" (llr.w32[0]), "=r" (llr.w32[1]): "r" (op1), "r" (op2) , "0" (llr.w32[0]), "1" (llr.w32[1]) );
#else               /* Big endian */
  __ASM volatile ("smlsldx %0, %1, %2, %3" : "=r" (llr.w32[1]), "=r" (llr.w32[0]): "r" (op1), "r" (op2) , "0" (llr.w32[1]), "1" (llr.w32[0]) );
#endif

  return(llr.w64);
}

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SEL  (uint32_t op1, uint32_t op2)
{
  uint32_t result;

  __ASM volatile ("sel %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE  int32_t __QADD( int32_t op1,  int32_t op2)
{
  int32_t result;

  __ASM volatile ("qadd %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

__attribute__((always_inline)) __STATIC_INLINE  int32_t __QSUB( int32_t op1,  int32_t op2)
{
  int32_t result;

  __ASM volatile ("qsub %0, %1, %2" : "=r" (result) : "r" (op1), "r" (op2) );
  return(result);
}

#define __PKHBT(ARG1,ARG2,ARG3) \
({                          \
  uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2); \
  __ASM ("pkhbt %0, %1, %2, lsl %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  ); \
  __RES; \
 })

#define __PKHTB(ARG1,ARG2,ARG3) \
({                          \
  uint32_t __RES, __ARG1 = (ARG1), __ARG2 = (ARG2); \
  if (ARG3 == 0) \
    __ASM ("pkhtb %0, %1, %2" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2)  ); \
  else \
    __ASM ("pkhtb %0, %1, %2, asr %3" : "=r" (__RES) :  "r" (__ARG1), "r" (__ARG2), "I" (ARG3)  ); \
  __RES; \
 })

__attribute__((always_inline)) __STATIC_INLINE uint32_t __SMMLA (int32_t op1, int32_t op2, int32_t op3)
{
 int32_t result;

 __ASM volatile ("smmla %0, %1, %2, %3" : "=r" (result): "r"  (op1), "r" (op2), "r" (op3) );
 return(result);
}

#endif /* (__ARM_FEATURE_DSP == 1U) */
/*@} end of group CMSIS_SIMD_intrinsics */


#endif /* __CMSIS_ARMCC_V6_H */
