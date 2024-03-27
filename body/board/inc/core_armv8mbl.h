/**************************************************************************//**
 * @file     core_armv8mbl.h
 * @brief    CMSIS Armv8-M Baseline Core Peripheral Access Layer Header File
 * @version  V5.0.7
 * @date     22. June 2018
 ******************************************************************************/
/*
 * Copyright (c) 2009-2018 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if   defined ( __ICCARM__ )
  #pragma system_include         /* treat file as system include file for MISRA check */
#elif defined (__clang__)
  #pragma clang system_header   /* treat file as system include file */
#endif

#ifndef __CORE_ARMV8MBL_H_GENERIC
#define __CORE_ARMV8MBL_H_GENERIC

#include <stdint.h>

#ifdef __cplusplus
 extern "C" {
#endif

/**
  \page CMSIS_MISRA_Exceptions  MISRA-C:2004 Compliance Exceptions
  CMSIS violates the following MISRA-C:2004 rules:

   \li Required Rule 8.5, object/function definition in header file.<br>
     Function definitions in header files are used to allow 'inlining'.

   \li Required Rule 18.4, declaration of union type or object of union type: '{...}'.<br>
     Unions are used for effective representation of core registers.

   \li Advisory Rule 19.7, Function-like macro defined.<br>
     Function-like macros are used to allow more efficient code.
 */


/*******************************************************************************
 *                 CMSIS definitions
 ******************************************************************************/
/**
  \ingroup Cortex_ARMv8MBL
  @{
 */

#include "cmsis_version.h"

/*  CMSIS definitions */
#define __ARMv8MBL_CMSIS_VERSION_MAIN  (__CM_CMSIS_VERSION_MAIN)                   /*!< \deprecated [31:16] CMSIS HAL main version */
#define __ARMv8MBL_CMSIS_VERSION_SUB   (__CM_CMSIS_VERSION_SUB)                    /*!< \deprecated [15:0]  CMSIS HAL sub version */
#define __ARMv8MBL_CMSIS_VERSION       ((__ARMv8MBL_CMSIS_VERSION_MAIN << 16U) | \
                                         __ARMv8MBL_CMSIS_VERSION_SUB           )  /*!< \deprecated CMSIS HAL version number */

#define __CORTEX_M                     ( 2U)                                            /*!< Cortex-M Core */

/** __FPU_USED indicates whether an FPU is used or not.
    This core does not support an FPU at all
*/
#define __FPU_USED       0U

#if defined ( __CC_ARM )
  #if defined __TARGET_FPU_VFP
    #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
  #endif

#elif defined (__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)
  #if defined __ARM_PCS_VFP
    #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
  #endif

#elif defined ( __GNUC__ )
  #if defined (__VFP_FP__) && !defined(__SOFTFP__)
    #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
  #endif

#elif defined ( __ICCARM__ )
  #if defined __ARMVFP__
    #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
  #endif

#elif defined ( __TI_ARM__ )
  #if defined __TI_VFP_SUPPORT__
    #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
  #endif

#elif defined ( __TASKING__ )
  #if defined __FPU_VFP__
    #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
  #endif

#elif defined ( __CSMC__ )
  #if ( __CSMC__ & 0x400U)
    #error "Compiler generates FPU instructions for a device without an FPU (check __FPU_PRESENT)"
  #endif

#endif

#include "cmsis_compiler.h"               /* CMSIS compiler specific defines */


#ifdef __cplusplus
}
#endif

#endif /* __CORE_ARMV8MBL_H_GENERIC */

#ifndef __CMSIS_GENERIC

#ifndef __CORE_ARMV8MBL_H_DEPENDANT
#define __CORE_ARMV8MBL_H_DEPENDANT

#ifdef __cplusplus
 extern "C" {
#endif

/* check device defines and use defaults */
#if defined __CHECK_DEVICE_DEFINES
  #ifndef __ARMv8MBL_REV
    #define __ARMv8MBL_REV               0x0000U
    #warning "__ARMv8MBL_REV not defined in device header file; using default!"
  #endif

  #ifndef __FPU_PRESENT
    #define __FPU_PRESENT             0U
    #warning "__FPU_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __MPU_PRESENT
    #define __MPU_PRESENT             0U
    #warning "__MPU_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __SAUREGION_PRESENT
    #define __SAUREGION_PRESENT       0U
    #warning "__SAUREGION_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __VTOR_PRESENT
    #define __VTOR_PRESENT            0U
    #warning "__VTOR_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __NVIC_PRIO_BITS
    #define __NVIC_PRIO_BITS          2U
    #warning "__NVIC_PRIO_BITS not defined in device header file; using default!"
  #endif

  #ifndef __Vendor_SysTickConfig
    #define __Vendor_SysTickConfig    0U
    #warning "__Vendor_SysTickConfig not defined in device header file; using default!"
  #endif

  #ifndef __ETM_PRESENT
    #define __ETM_PRESENT             0U
    #warning "__ETM_PRESENT not defined in device header file; using default!"
  #endif

  #ifndef __MTB_PRESENT
    #define __MTB_PRESENT             0U
    #warning "__MTB_PRESENT not defined in device header file; using default!"
  #endif

#endif

/* IO definitions (access restrictions to peripheral registers) */
/**
    \defgroup CMSIS_glob_defs CMSIS Global Defines

    <strong>IO Type Qualifiers</strong> are used
    \li to specify the access to peripheral variables.
    \li for automatic generation of peripheral register debug information.
*/
#ifdef __cplusplus
  #define   __I     volatile             /*!< Defines 'read only' permissions */
#else
  #define   __I     volatile const       /*!< Defines 'read only' permissions */
#endif
#define     __O     volatile             /*!< Defines 'write only' permissions */
#define     __IO    volatile             /*!< Defines 'read / write' permissions */

/* following defines should be used for structure members */
#define     __IM     volatile const      /*! Defines 'read only' structure member permissions */
#define     __OM     volatile            /*! Defines 'write only' structure member permissions */
#define     __IOM    volatile            /*! Defines 'read / write' structure member permissions */

/*@} end of group ARMv8MBL */



/*******************************************************************************
 *                 Register Abstraction
  Core Register contain:
  - Core Register
  - Core NVIC Register
  - Core SCB Register
  - Core SysTick Register
  - Core Debug Register
  - Core MPU Register
  - Core SAU Register
 ******************************************************************************/
/**
  \defgroup CMSIS_core_register Defines and Type Definitions
  \brief Type definitions and defines for Cortex-M processor based devices.
*/

/**
  \ingroup    CMSIS_core_register
  \defgroup   CMSIS_CORE  Status and Control Registers
  \brief      Core Register type definitions.
  @{
 */

/**
  \brief  Union type to access the Application Program Status Register (APSR).
 */
typedef union
{
  struct
  {
    uint32_t _reserved0:28;              /*!< bit:  0..27  Reserved */
    uint32_t V:1;                        /*!< bit:     28  Overflow condition code flag */
    uint32_t C:1;                        /*!< bit:     29  Carry condition code flag */
    uint32_t Z:1;                        /*!< bit:     30  Zero condition code flag */
    uint32_t N:1;                        /*!< bit:     31  Negative condition code flag */
  } b;                                   /*!< Structure used for bit  access */
  uint32_t w;                            /*!< Type      used for word access */
} APSR_Type;

/* APSR Register Definitions */
#define APSR_N_Pos                         31U                                            /*!< APSR: N Position */
#define APSR_N_Msk                         (1UL << APSR_N_Pos)                            /*!< APSR: N Mask */

#define APSR_Z_Pos                         30U                                            /*!< APSR: Z Position */
#define APSR_Z_Msk                         (1UL << APSR_Z_Pos)                            /*!< APSR: Z Mask */

#define APSR_C_Pos                         29U                                            /*!< APSR: C Position */
#define APSR_C_Msk                         (1UL << APSR_C_Pos)                            /*!< APSR: C Mask */

#define APSR_V_Pos                         28U                                            /*!< APSR: V Position */
#define APSR_V_Msk                         (1UL << APSR_V_Pos)                            /*!< APSR: V Mask */


/**
  \brief  Union type to access the Interrupt Program Status Register (IPSR).
 */
typedef union
{
  struct
  {
    uint32_t ISR:9;                      /*!< bit:  0.. 8  Exception number */
    uint32_t _reserved0:23;              /*!< bit:  9..31  Reserved */
  } b;                                   /*!< Structure used for bit  access */
  uint32_t w;                            /*!< Type      used for word access */
} IPSR_Type;

/* IPSR Register Definitions */
#define IPSR_ISR_Pos                        0U                                            /*!< IPSR: ISR Position */
#define IPSR_ISR_Msk                       (0x1FFUL /*<< IPSR_ISR_Pos*/)                  /*!< IPSR: ISR Mask */


/**
  \brief  Union type to access the Special-Purpose Program Status Registers (xPSR).
 */
typedef union
{
  struct
  {
    uint32_t ISR:9;                      /*!< bit:  0.. 8  Exception number */
    uint32_t _reserved0:15;              /*!< bit:  9..23  Reserved */
    uint32_t T:1;                        /*!< bit:     24  Thumb bit        (read 0) */
    uint32_t _reserved1:3;               /*!< bit: 25..27  Reserved */
    uint32_t V:1;                        /*!< bit:     28  Overflow condition code flag */
    uint32_t C:1;                        /*!< bit:     29  Carry condition code flag */
    uint32_t Z:1;                        /*!< bit:     30  Zero condition code flag */
    uint32_t N:1;                        /*!< bit:     31  Negative condition code flag */
  } b;                                   /*!< Structure used for bit  access */
  uint32_t w;                            /*!< Type      used for word access */
} xPSR_Type;

/* xPSR Register Definitions */
#define xPSR_N_Pos                         31U                                            /*!< xPSR: N Position */
#define xPSR_N_Msk                         (1UL << xPSR_N_Pos)                            /*!< xPSR: N Mask */

#define xPSR_Z_Pos                         30U                                            /*!< xPSR: Z Position */
#define xPSR_Z_Msk                         (1UL << xPSR_Z_Pos)                            /*!< xPSR: Z Mask */

#define xPSR_C_Pos                         29U                                            /*!< xPSR: C Position */
#define xPSR_C_Msk                         (1UL << xPSR_C_Pos)                            /*!< xPSR: C Mask */

#define xPSR_V_Pos                         28U                                            /*!< xPSR: V Position */
#define xPSR_V_Msk                         (1UL << xPSR_V_Pos)                            /*!< xPSR: V Mask */

#define xPSR_T_Pos                         24U                                            /*!< xPSR: T Position */
#define xPSR_T_Msk                         (1UL << xPSR_T_Pos)                            /*!< xPSR: T Mask */

#define xPSR_ISR_Pos                        0U                                            /*!< xPSR: ISR Position */
#define xPSR_ISR_Msk                       (0x1FFUL /*<< xPSR_ISR_Pos*/)                  /*!< xPSR: ISR Mask */


/**
  \brief  Union type to access the Control Registers (CONTROL).
 */
typedef union
{
  struct
  {
    uint32_t nPRIV:1;                    /*!< bit:      0  Execution privilege in Thread mode */
    uint32_t SPSEL:1;                    /*!< bit:      1  Stack-pointer select */
    uint32_t _reserved1:30;              /*!< bit:  2..31  Reserved */
  } b;                                   /*!< Structure used for bit  access */
  uint32_t w;                            /*!< Type      used for word access */
} CONTROL_Type;

/* CONTROL Register Definitions */
#define CONTROL_SPSEL_Pos                   1U                                            /*!< CONTROL: SPSEL Position */
#define CONTROL_SPSEL_Msk                  (1UL << CONTROL_SPSEL_Pos)                     /*!< CONTROL: SPSEL Mask */

#define CONTROL_nPRIV_Pos                   0U                                            /*!< CONTROL: nPRIV Position */
#define CONTROL_nPRIV_Msk                  (1UL /*<< CONTROL_nPRIV_Pos*/)                 /*!< CONTROL: nPRIV Mask */

/*@} end of group CMSIS_CORE */


/**
  \ingroup    CMSIS_core_register
  \defgroup   CMSIS_NVIC  Nested Vectored Interrupt Controller (NVIC)
  \brief      Type definitions for the NVIC Registers
  @{
 */

/**
  \brief  Structure type to access the Nested Vectored Interrupt Controller (NVIC).
 */
typedef struct
{
  __IOM uint32_t ISER[16U];              /*!< Offset: 0x000 (R/W)  Interrupt Set Enable Register */
        uint32_t RESERVED0[16U];
  __IOM uint32_t ICER[16U];              /*!< Offset: 0x080 (R/W)  Interrupt Clear Enable Register */
        uint32_t RSERVED1[16U];
  __IOM uint32_t ISPR[16U];              /*!< Offset: 0x100 (R/W)  Interrupt Set Pending Register */
        uint32_t RESERVED2[16U];
  __IOM uint32_t ICPR[16U];              /*!< Offset: 0x180 (R/W)  Interrupt Clear Pending Register */
        uint32_t RESERVED3[16U];
  __IOM uint32_t IABR[16U];              /*!< Offset: 0x200 (R/W)  Interrupt Active bit Register */
        uint32_t RESERVED4[16U];
  __IOM uint32_t ITNS[16U];              /*!< Offset: 0x280 (R/W)  Interrupt Non-Secure State Register */
        uint32_t RESERVED5[16U];
  __IOM uint32_t IPR[124U];              /*!< Offset: 0x300 (R/W)  Interrupt Priority Register */
}  NVIC_Type;

/*@} end of group CMSIS_NVIC */


/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_SCB     System Control Block (SCB)
  \brief    Type definitions for the System Control Block Registers
  @{
 */

/**
  \brief  Structure type to access the System Control Block (SCB).
 */
typedef struct
{
  __IM  uint32_t CPUID;                  /*!< Offset: 0x000 (R/ )  CPUID Base Register */
  __IOM uint32_t ICSR;                   /*!< Offset: 0x004 (R/W)  Interrupt Control and State Register */
#if defined (__VTOR_PRESENT) && (__VTOR_PRESENT == 1U)
  __IOM uint32_t VTOR;                   /*!< Offset: 0x008 (R/W)  Vector Table Offset Register */
#else
        uint32_t RESERVED0;
#endif
  __IOM uint32_t AIRCR;                  /*!< Offset: 0x00C (R/W)  Application Interrupt and Reset Control Register */
  __IOM uint32_t SCR;                    /*!< Offset: 0x010 (R/W)  System Control Register */
  __IOM uint32_t CCR;                    /*!< Offset: 0x014 (R/W)  Configuration Control Register */
        uint32_t RESERVED1;
  __IOM uint32_t SHPR[2U];               /*!< Offset: 0x01C (R/W)  System Handlers Priority Registers. [0] is RESERVED */
  __IOM uint32_t SHCSR;                  /*!< Offset: 0x024 (R/W)  System Handler Control and State Register */
} SCB_Type;

/* SCB CPUID Register Definitions */
#define SCB_CPUID_IMPLEMENTER_Pos          24U                                            /*!< SCB CPUID: IMPLEMENTER Position */
#define SCB_CPUID_IMPLEMENTER_Msk          (0xFFUL << SCB_CPUID_IMPLEMENTER_Pos)          /*!< SCB CPUID: IMPLEMENTER Mask */

#define SCB_CPUID_VARIANT_Pos              20U                                            /*!< SCB CPUID: VARIANT Position */
#define SCB_CPUID_VARIANT_Msk              (0xFUL << SCB_CPUID_VARIANT_Pos)               /*!< SCB CPUID: VARIANT Mask */

#define SCB_CPUID_ARCHITECTURE_Pos         16U                                            /*!< SCB CPUID: ARCHITECTURE Position */
#define SCB_CPUID_ARCHITECTURE_Msk         (0xFUL << SCB_CPUID_ARCHITECTURE_Pos)          /*!< SCB CPUID: ARCHITECTURE Mask */

#define SCB_CPUID_PARTNO_Pos                4U                                            /*!< SCB CPUID: PARTNO Position */
#define SCB_CPUID_PARTNO_Msk               (0xFFFUL << SCB_CPUID_PARTNO_Pos)              /*!< SCB CPUID: PARTNO Mask */

#define SCB_CPUID_REVISION_Pos              0U                                            /*!< SCB CPUID: REVISION Position */
#define SCB_CPUID_REVISION_Msk             (0xFUL /*<< SCB_CPUID_REVISION_Pos*/)          /*!< SCB CPUID: REVISION Mask */

/* SCB Interrupt Control State Register Definitions */
#define SCB_ICSR_PENDNMISET_Pos            31U                                            /*!< SCB ICSR: PENDNMISET Position */
#define SCB_ICSR_PENDNMISET_Msk            (1UL << SCB_ICSR_PENDNMISET_Pos)               /*!< SCB ICSR: PENDNMISET Mask */

#define SCB_ICSR_NMIPENDSET_Pos            SCB_ICSR_PENDNMISET_Pos                        /*!< SCB ICSR: NMIPENDSET Position, backward compatibility */
#define SCB_ICSR_NMIPENDSET_Msk            SCB_ICSR_PENDNMISET_Msk                        /*!< SCB ICSR: NMIPENDSET Mask, backward compatibility */

#define SCB_ICSR_PENDNMICLR_Pos            30U                                            /*!< SCB ICSR: PENDNMICLR Position */
#define SCB_ICSR_PENDNMICLR_Msk            (1UL << SCB_ICSR_PENDNMICLR_Pos)               /*!< SCB ICSR: PENDNMICLR Mask */

#define SCB_ICSR_PENDSVSET_Pos             28U                                            /*!< SCB ICSR: PENDSVSET Position */
#define SCB_ICSR_PENDSVSET_Msk             (1UL << SCB_ICSR_PENDSVSET_Pos)                /*!< SCB ICSR: PENDSVSET Mask */

#define SCB_ICSR_PENDSVCLR_Pos             27U                                            /*!< SCB ICSR: PENDSVCLR Position */
#define SCB_ICSR_PENDSVCLR_Msk             (1UL << SCB_ICSR_PENDSVCLR_Pos)                /*!< SCB ICSR: PENDSVCLR Mask */

#define SCB_ICSR_PENDSTSET_Pos             26U                                            /*!< SCB ICSR: PENDSTSET Position */
#define SCB_ICSR_PENDSTSET_Msk             (1UL << SCB_ICSR_PENDSTSET_Pos)                /*!< SCB ICSR: PENDSTSET Mask */

#define SCB_ICSR_PENDSTCLR_Pos             25U                                            /*!< SCB ICSR: PENDSTCLR Position */
#define SCB_ICSR_PENDSTCLR_Msk             (1UL << SCB_ICSR_PENDSTCLR_Pos)                /*!< SCB ICSR: PENDSTCLR Mask */

#define SCB_ICSR_STTNS_Pos                 24U                                            /*!< SCB ICSR: STTNS Position (Security Extension) */
#define SCB_ICSR_STTNS_Msk                 (1UL << SCB_ICSR_STTNS_Pos)                    /*!< SCB ICSR: STTNS Mask (Security Extension) */

#define SCB_ICSR_ISRPREEMPT_Pos            23U                                            /*!< SCB ICSR: ISRPREEMPT Position */
#define SCB_ICSR_ISRPREEMPT_Msk            (1UL << SCB_ICSR_ISRPREEMPT_Pos)               /*!< SCB ICSR: ISRPREEMPT Mask */

#define SCB_ICSR_ISRPENDING_Pos            22U                                            /*!< SCB ICSR: ISRPENDING Position */
#define SCB_ICSR_ISRPENDING_Msk            (1UL << SCB_ICSR_ISRPENDING_Pos)               /*!< SCB ICSR: ISRPENDING Mask */

#define SCB_ICSR_VECTPENDING_Pos           12U                                            /*!< SCB ICSR: VECTPENDING Position */
#define SCB_ICSR_VECTPENDING_Msk           (0x1FFUL << SCB_ICSR_VECTPENDING_Pos)          /*!< SCB ICSR: VECTPENDING Mask */

#define SCB_ICSR_RETTOBASE_Pos             11U                                            /*!< SCB ICSR: RETTOBASE Position */
#define SCB_ICSR_RETTOBASE_Msk             (1UL << SCB_ICSR_RETTOBASE_Pos)                /*!< SCB ICSR: RETTOBASE Mask */

#define SCB_ICSR_VECTACTIVE_Pos             0U                                            /*!< SCB ICSR: VECTACTIVE Position */
#define SCB_ICSR_VECTACTIVE_Msk            (0x1FFUL /*<< SCB_ICSR_VECTACTIVE_Pos*/)       /*!< SCB ICSR: VECTACTIVE Mask */

#if defined (__VTOR_PRESENT) && (__VTOR_PRESENT == 1U)
/* SCB Vector Table Offset Register Definitions */
#define SCB_VTOR_TBLOFF_Pos                 7U                                            /*!< SCB VTOR: TBLOFF Position */
#define SCB_VTOR_TBLOFF_Msk                (0x1FFFFFFUL << SCB_VTOR_TBLOFF_Pos)           /*!< SCB VTOR: TBLOFF Mask */
#endif

/* SCB Application Interrupt and Reset Control Register Definitions */
#define SCB_AIRCR_VECTKEY_Pos              16U                                            /*!< SCB AIRCR: VECTKEY Position */
#define SCB_AIRCR_VECTKEY_Msk              (0xFFFFUL << SCB_AIRCR_VECTKEY_Pos)            /*!< SCB AIRCR: VECTKEY Mask */

#define SCB_AIRCR_VECTKEYSTAT_Pos          16U                                            /*!< SCB AIRCR: VECTKEYSTAT Position */
#define SCB_AIRCR_VECTKEYSTAT_Msk          (0xFFFFUL << SCB_AIRCR_VECTKEYSTAT_Pos)        /*!< SCB AIRCR: VECTKEYSTAT Mask */

#define SCB_AIRCR_ENDIANESS_Pos            15U                                            /*!< SCB AIRCR: ENDIANESS Position */
#define SCB_AIRCR_ENDIANESS_Msk            (1UL << SCB_AIRCR_ENDIANESS_Pos)               /*!< SCB AIRCR: ENDIANESS Mask */

#define SCB_AIRCR_PRIS_Pos                 14U                                            /*!< SCB AIRCR: PRIS Position */
#define SCB_AIRCR_PRIS_Msk                 (1UL << SCB_AIRCR_PRIS_Pos)                    /*!< SCB AIRCR: PRIS Mask */

#define SCB_AIRCR_BFHFNMINS_Pos            13U                                            /*!< SCB AIRCR: BFHFNMINS Position */
#define SCB_AIRCR_BFHFNMINS_Msk            (1UL << SCB_AIRCR_BFHFNMINS_Pos)               /*!< SCB AIRCR: BFHFNMINS Mask */

#define SCB_AIRCR_SYSRESETREQS_Pos          3U                                            /*!< SCB AIRCR: SYSRESETREQS Position */
#define SCB_AIRCR_SYSRESETREQS_Msk         (1UL << SCB_AIRCR_SYSRESETREQS_Pos)            /*!< SCB AIRCR: SYSRESETREQS Mask */

#define SCB_AIRCR_SYSRESETREQ_Pos           2U                                            /*!< SCB AIRCR: SYSRESETREQ Position */
#define SCB_AIRCR_SYSRESETREQ_Msk          (1UL << SCB_AIRCR_SYSRESETREQ_Pos)             /*!< SCB AIRCR: SYSRESETREQ Mask */

#define SCB_AIRCR_VECTCLRACTIVE_Pos         1U                                            /*!< SCB AIRCR: VECTCLRACTIVE Position */
#define SCB_AIRCR_VECTCLRACTIVE_Msk        (1UL << SCB_AIRCR_VECTCLRACTIVE_Pos)           /*!< SCB AIRCR: VECTCLRACTIVE Mask */

/* SCB System Control Register Definitions */
#define SCB_SCR_SEVONPEND_Pos               4U                                            /*!< SCB SCR: SEVONPEND Position */
#define SCB_SCR_SEVONPEND_Msk              (1UL << SCB_SCR_SEVONPEND_Pos)                 /*!< SCB SCR: SEVONPEND Mask */

#define SCB_SCR_SLEEPDEEPS_Pos              3U                                            /*!< SCB SCR: SLEEPDEEPS Position */
#define SCB_SCR_SLEEPDEEPS_Msk             (1UL << SCB_SCR_SLEEPDEEPS_Pos)                /*!< SCB SCR: SLEEPDEEPS Mask */

#define SCB_SCR_SLEEPDEEP_Pos               2U                                            /*!< SCB SCR: SLEEPDEEP Position */
#define SCB_SCR_SLEEPDEEP_Msk              (1UL << SCB_SCR_SLEEPDEEP_Pos)                 /*!< SCB SCR: SLEEPDEEP Mask */

#define SCB_SCR_SLEEPONEXIT_Pos             1U                                            /*!< SCB SCR: SLEEPONEXIT Position */
#define SCB_SCR_SLEEPONEXIT_Msk            (1UL << SCB_SCR_SLEEPONEXIT_Pos)               /*!< SCB SCR: SLEEPONEXIT Mask */

/* SCB Configuration Control Register Definitions */
#define SCB_CCR_BP_Pos                     18U                                            /*!< SCB CCR: BP Position */
#define SCB_CCR_BP_Msk                     (1UL << SCB_CCR_BP_Pos)                        /*!< SCB CCR: BP Mask */

#define SCB_CCR_IC_Pos                     17U                                            /*!< SCB CCR: IC Position */
#define SCB_CCR_IC_Msk                     (1UL << SCB_CCR_IC_Pos)                        /*!< SCB CCR: IC Mask */

#define SCB_CCR_DC_Pos                     16U                                            /*!< SCB CCR: DC Position */
#define SCB_CCR_DC_Msk                     (1UL << SCB_CCR_DC_Pos)                        /*!< SCB CCR: DC Mask */

#define SCB_CCR_STKOFHFNMIGN_Pos           10U                                            /*!< SCB CCR: STKOFHFNMIGN Position */
#define SCB_CCR_STKOFHFNMIGN_Msk           (1UL << SCB_CCR_STKOFHFNMIGN_Pos)              /*!< SCB CCR: STKOFHFNMIGN Mask */

#define SCB_CCR_BFHFNMIGN_Pos               8U                                            /*!< SCB CCR: BFHFNMIGN Position */
#define SCB_CCR_BFHFNMIGN_Msk              (1UL << SCB_CCR_BFHFNMIGN_Pos)                 /*!< SCB CCR: BFHFNMIGN Mask */

#define SCB_CCR_DIV_0_TRP_Pos               4U                                            /*!< SCB CCR: DIV_0_TRP Position */
#define SCB_CCR_DIV_0_TRP_Msk              (1UL << SCB_CCR_DIV_0_TRP_Pos)                 /*!< SCB CCR: DIV_0_TRP Mask */

#define SCB_CCR_UNALIGN_TRP_Pos             3U                                            /*!< SCB CCR: UNALIGN_TRP Position */
#define SCB_CCR_UNALIGN_TRP_Msk            (1UL << SCB_CCR_UNALIGN_TRP_Pos)               /*!< SCB CCR: UNALIGN_TRP Mask */

#define SCB_CCR_USERSETMPEND_Pos            1U                                            /*!< SCB CCR: USERSETMPEND Position */
#define SCB_CCR_USERSETMPEND_Msk           (1UL << SCB_CCR_USERSETMPEND_Pos)              /*!< SCB CCR: USERSETMPEND Mask */

/* SCB System Handler Control and State Register Definitions */
#define SCB_SHCSR_HARDFAULTPENDED_Pos      21U                                            /*!< SCB SHCSR: HARDFAULTPENDED Position */
#define SCB_SHCSR_HARDFAULTPENDED_Msk      (1UL << SCB_SHCSR_HARDFAULTPENDED_Pos)         /*!< SCB SHCSR: HARDFAULTPENDED Mask */

#define SCB_SHCSR_SVCALLPENDED_Pos         15U                                            /*!< SCB SHCSR: SVCALLPENDED Position */
#define SCB_SHCSR_SVCALLPENDED_Msk         (1UL << SCB_SHCSR_SVCALLPENDED_Pos)            /*!< SCB SHCSR: SVCALLPENDED Mask */

#define SCB_SHCSR_SYSTICKACT_Pos           11U                                            /*!< SCB SHCSR: SYSTICKACT Position */
#define SCB_SHCSR_SYSTICKACT_Msk           (1UL << SCB_SHCSR_SYSTICKACT_Pos)              /*!< SCB SHCSR: SYSTICKACT Mask */

#define SCB_SHCSR_PENDSVACT_Pos            10U                                            /*!< SCB SHCSR: PENDSVACT Position */
#define SCB_SHCSR_PENDSVACT_Msk            (1UL << SCB_SHCSR_PENDSVACT_Pos)               /*!< SCB SHCSR: PENDSVACT Mask */

#define SCB_SHCSR_SVCALLACT_Pos             7U                                            /*!< SCB SHCSR: SVCALLACT Position */
#define SCB_SHCSR_SVCALLACT_Msk            (1UL << SCB_SHCSR_SVCALLACT_Pos)               /*!< SCB SHCSR: SVCALLACT Mask */

#define SCB_SHCSR_NMIACT_Pos                5U                                            /*!< SCB SHCSR: NMIACT Position */
#define SCB_SHCSR_NMIACT_Msk               (1UL << SCB_SHCSR_NMIACT_Pos)                  /*!< SCB SHCSR: NMIACT Mask */

#define SCB_SHCSR_HARDFAULTACT_Pos          2U                                            /*!< SCB SHCSR: HARDFAULTACT Position */
#define SCB_SHCSR_HARDFAULTACT_Msk         (1UL << SCB_SHCSR_HARDFAULTACT_Pos)            /*!< SCB SHCSR: HARDFAULTACT Mask */

/*@} end of group CMSIS_SCB */


/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_SysTick     System Tick Timer (SysTick)
  \brief    Type definitions for the System Timer Registers.
  @{
 */

/**
  \brief  Structure type to access the System Timer (SysTick).
 */
typedef struct
{
  __IOM uint32_t CTRL;                   /*!< Offset: 0x000 (R/W)  SysTick Control and Status Register */
  __IOM uint32_t LOAD;                   /*!< Offset: 0x004 (R/W)  SysTick Reload Value Register */
  __IOM uint32_t VAL;                    /*!< Offset: 0x008 (R/W)  SysTick Current Value Register */
  __IM  uint32_t CALIB;                  /*!< Offset: 0x00C (R/ )  SysTick Calibration Register */
} SysTick_Type;

/* SysTick Control / Status Register Definitions */
#define SysTick_CTRL_COUNTFLAG_Pos         16U                                            /*!< SysTick CTRL: COUNTFLAG Position */
#define SysTick_CTRL_COUNTFLAG_Msk         (1UL << SysTick_CTRL_COUNTFLAG_Pos)            /*!< SysTick CTRL: COUNTFLAG Mask */

#define SysTick_CTRL_CLKSOURCE_Pos          2U                                            /*!< SysTick CTRL: CLKSOURCE Position */
#define SysTick_CTRL_CLKSOURCE_Msk         (1UL << SysTick_CTRL_CLKSOURCE_Pos)            /*!< SysTick CTRL: CLKSOURCE Mask */

#define SysTick_CTRL_TICKINT_Pos            1U                                            /*!< SysTick CTRL: TICKINT Position */
#define SysTick_CTRL_TICKINT_Msk           (1UL << SysTick_CTRL_TICKINT_Pos)              /*!< SysTick CTRL: TICKINT Mask */

#define SysTick_CTRL_ENABLE_Pos             0U                                            /*!< SysTick CTRL: ENABLE Position */
#define SysTick_CTRL_ENABLE_Msk            (1UL /*<< SysTick_CTRL_ENABLE_Pos*/)           /*!< SysTick CTRL: ENABLE Mask */

/* SysTick Reload Register Definitions */
#define SysTick_LOAD_RELOAD_Pos             0U                                            /*!< SysTick LOAD: RELOAD Position */
#define SysTick_LOAD_RELOAD_Msk            (0xFFFFFFUL /*<< SysTick_LOAD_RELOAD_Pos*/)    /*!< SysTick LOAD: RELOAD Mask */

/* SysTick Current Register Definitions */
#define SysTick_VAL_CURRENT_Pos             0U                                            /*!< SysTick VAL: CURRENT Position */
#define SysTick_VAL_CURRENT_Msk            (0xFFFFFFUL /*<< SysTick_VAL_CURRENT_Pos*/)    /*!< SysTick VAL: CURRENT Mask */

/* SysTick Calibration Register Definitions */
#define SysTick_CALIB_NOREF_Pos            31U                                            /*!< SysTick CALIB: NOREF Position */
#define SysTick_CALIB_NOREF_Msk            (1UL << SysTick_CALIB_NOREF_Pos)               /*!< SysTick CALIB: NOREF Mask */

#define SysTick_CALIB_SKEW_Pos             30U                                            /*!< SysTick CALIB: SKEW Position */
#define SysTick_CALIB_SKEW_Msk             (1UL << SysTick_CALIB_SKEW_Pos)                /*!< SysTick CALIB: SKEW Mask */

#define SysTick_CALIB_TENMS_Pos             0U                                            /*!< SysTick CALIB: TENMS Position */
#define SysTick_CALIB_TENMS_Msk            (0xFFFFFFUL /*<< SysTick_CALIB_TENMS_Pos*/)    /*!< SysTick CALIB: TENMS Mask */

/*@} end of group CMSIS_SysTick */


/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_DWT     Data Watchpoint and Trace (DWT)
  \brief    Type definitions for the Data Watchpoint and Trace (DWT)
  @{
 */

/**
  \brief  Structure type to access the Data Watchpoint and Trace Register (DWT).
 */
typedef struct
{
  __IOM uint32_t CTRL;                   /*!< Offset: 0x000 (R/W)  Control Register */
        uint32_t RESERVED0[6U];
  __IM  uint32_t PCSR;                   /*!< Offset: 0x01C (R/ )  Program Counter Sample Register */
  __IOM uint32_t COMP0;                  /*!< Offset: 0x020 (R/W)  Comparator Register 0 */
        uint32_t RESERVED1[1U];
  __IOM uint32_t FUNCTION0;              /*!< Offset: 0x028 (R/W)  Function Register 0 */
        uint32_t RESERVED2[1U];
  __IOM uint32_t COMP1;                  /*!< Offset: 0x030 (R/W)  Comparator Register 1 */
        uint32_t RESERVED3[1U];
  __IOM uint32_t FUNCTION1;              /*!< Offset: 0x038 (R/W)  Function Register 1 */
        uint32_t RESERVED4[1U];
  __IOM uint32_t COMP2;                  /*!< Offset: 0x040 (R/W)  Comparator Register 2 */
        uint32_t RESERVED5[1U];
  __IOM uint32_t FUNCTION2;              /*!< Offset: 0x048 (R/W)  Function Register 2 */
        uint32_t RESERVED6[1U];
  __IOM uint32_t COMP3;                  /*!< Offset: 0x050 (R/W)  Comparator Register 3 */
        uint32_t RESERVED7[1U];
  __IOM uint32_t FUNCTION3;              /*!< Offset: 0x058 (R/W)  Function Register 3 */
        uint32_t RESERVED8[1U];
  __IOM uint32_t COMP4;                  /*!< Offset: 0x060 (R/W)  Comparator Register 4 */
        uint32_t RESERVED9[1U];
  __IOM uint32_t FUNCTION4;              /*!< Offset: 0x068 (R/W)  Function Register 4 */
        uint32_t RESERVED10[1U];
  __IOM uint32_t COMP5;                  /*!< Offset: 0x070 (R/W)  Comparator Register 5 */
        uint32_t RESERVED11[1U];
  __IOM uint32_t FUNCTION5;              /*!< Offset: 0x078 (R/W)  Function Register 5 */
        uint32_t RESERVED12[1U];
  __IOM uint32_t COMP6;                  /*!< Offset: 0x080 (R/W)  Comparator Register 6 */
        uint32_t RESERVED13[1U];
  __IOM uint32_t FUNCTION6;              /*!< Offset: 0x088 (R/W)  Function Register 6 */
        uint32_t RESERVED14[1U];
  __IOM uint32_t COMP7;                  /*!< Offset: 0x090 (R/W)  Comparator Register 7 */
        uint32_t RESERVED15[1U];
  __IOM uint32_t FUNCTION7;              /*!< Offset: 0x098 (R/W)  Function Register 7 */
        uint32_t RESERVED16[1U];
  __IOM uint32_t COMP8;                  /*!< Offset: 0x0A0 (R/W)  Comparator Register 8 */
        uint32_t RESERVED17[1U];
  __IOM uint32_t FUNCTION8;              /*!< Offset: 0x0A8 (R/W)  Function Register 8 */
        uint32_t RESERVED18[1U];
  __IOM uint32_t COMP9;                  /*!< Offset: 0x0B0 (R/W)  Comparator Register 9 */
        uint32_t RESERVED19[1U];
  __IOM uint32_t FUNCTION9;              /*!< Offset: 0x0B8 (R/W)  Function Register 9 */
        uint32_t RESERVED20[1U];
  __IOM uint32_t COMP10;                 /*!< Offset: 0x0C0 (R/W)  Comparator Register 10 */
        uint32_t RESERVED21[1U];
  __IOM uint32_t FUNCTION10;             /*!< Offset: 0x0C8 (R/W)  Function Register 10 */
        uint32_t RESERVED22[1U];
  __IOM uint32_t COMP11;                 /*!< Offset: 0x0D0 (R/W)  Comparator Register 11 */
        uint32_t RESERVED23[1U];
  __IOM uint32_t FUNCTION11;             /*!< Offset: 0x0D8 (R/W)  Function Register 11 */
        uint32_t RESERVED24[1U];
  __IOM uint32_t COMP12;                 /*!< Offset: 0x0E0 (R/W)  Comparator Register 12 */
        uint32_t RESERVED25[1U];
  __IOM uint32_t FUNCTION12;             /*!< Offset: 0x0E8 (R/W)  Function Register 12 */
        uint32_t RESERVED26[1U];
  __IOM uint32_t COMP13;                 /*!< Offset: 0x0F0 (R/W)  Comparator Register 13 */
        uint32_t RESERVED27[1U];
  __IOM uint32_t FUNCTION13;             /*!< Offset: 0x0F8 (R/W)  Function Register 13 */
        uint32_t RESERVED28[1U];
  __IOM uint32_t COMP14;                 /*!< Offset: 0x100 (R/W)  Comparator Register 14 */
        uint32_t RESERVED29[1U];
  __IOM uint32_t FUNCTION14;             /*!< Offset: 0x108 (R/W)  Function Register 14 */
        uint32_t RESERVED30[1U];
  __IOM uint32_t COMP15;                 /*!< Offset: 0x110 (R/W)  Comparator Register 15 */
        uint32_t RESERVED31[1U];
  __IOM uint32_t FUNCTION15;             /*!< Offset: 0x118 (R/W)  Function Register 15 */
} DWT_Type;

/* DWT Control Register Definitions */
#define DWT_CTRL_NUMCOMP_Pos               28U                                         /*!< DWT CTRL: NUMCOMP Position */
#define DWT_CTRL_NUMCOMP_Msk               (0xFUL << DWT_CTRL_NUMCOMP_Pos)             /*!< DWT CTRL: NUMCOMP Mask */

#define DWT_CTRL_NOTRCPKT_Pos              27U                                         /*!< DWT CTRL: NOTRCPKT Position */
#define DWT_CTRL_NOTRCPKT_Msk              (0x1UL << DWT_CTRL_NOTRCPKT_Pos)            /*!< DWT CTRL: NOTRCPKT Mask */

#define DWT_CTRL_NOEXTTRIG_Pos             26U                                         /*!< DWT CTRL: NOEXTTRIG Position */
#define DWT_CTRL_NOEXTTRIG_Msk             (0x1UL << DWT_CTRL_NOEXTTRIG_Pos)           /*!< DWT CTRL: NOEXTTRIG Mask */

#define DWT_CTRL_NOCYCCNT_Pos              25U                                         /*!< DWT CTRL: NOCYCCNT Position */
#define DWT_CTRL_NOCYCCNT_Msk              (0x1UL << DWT_CTRL_NOCYCCNT_Pos)            /*!< DWT CTRL: NOCYCCNT Mask */

#define DWT_CTRL_NOPRFCNT_Pos              24U                                         /*!< DWT CTRL: NOPRFCNT Position */
#define DWT_CTRL_NOPRFCNT_Msk              (0x1UL << DWT_CTRL_NOPRFCNT_Pos)            /*!< DWT CTRL: NOPRFCNT Mask */

/* DWT Comparator Function Register Definitions */
#define DWT_FUNCTION_ID_Pos                27U                                         /*!< DWT FUNCTION: ID Position */
#define DWT_FUNCTION_ID_Msk                (0x1FUL << DWT_FUNCTION_ID_Pos)             /*!< DWT FUNCTION: ID Mask */

#define DWT_FUNCTION_MATCHED_Pos           24U                                         /*!< DWT FUNCTION: MATCHED Position */
#define DWT_FUNCTION_MATCHED_Msk           (0x1UL << DWT_FUNCTION_MATCHED_Pos)         /*!< DWT FUNCTION: MATCHED Mask */

#define DWT_FUNCTION_DATAVSIZE_Pos         10U                                         /*!< DWT FUNCTION: DATAVSIZE Position */
#define DWT_FUNCTION_DATAVSIZE_Msk         (0x3UL << DWT_FUNCTION_DATAVSIZE_Pos)       /*!< DWT FUNCTION: DATAVSIZE Mask */

#define DWT_FUNCTION_ACTION_Pos             4U                                         /*!< DWT FUNCTION: ACTION Position */
#define DWT_FUNCTION_ACTION_Msk            (0x3UL << DWT_FUNCTION_ACTION_Pos)          /*!< DWT FUNCTION: ACTION Mask */

#define DWT_FUNCTION_MATCH_Pos              0U                                         /*!< DWT FUNCTION: MATCH Position */
#define DWT_FUNCTION_MATCH_Msk             (0xFUL /*<< DWT_FUNCTION_MATCH_Pos*/)       /*!< DWT FUNCTION: MATCH Mask */

/*@}*/ /* end of group CMSIS_DWT */


/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_TPI     Trace Port Interface (TPI)
  \brief    Type definitions for the Trace Port Interface (TPI)
  @{
 */

/**
  \brief  Structure type to access the Trace Port Interface Register (TPI).
 */
typedef struct
{
  __IM  uint32_t SSPSR;                  /*!< Offset: 0x000 (R/ )  Supported Parallel Port Sizes Register */
  __IOM uint32_t CSPSR;                  /*!< Offset: 0x004 (R/W)  Current Parallel Port Sizes Register */
        uint32_t RESERVED0[2U];
  __IOM uint32_t ACPR;                   /*!< Offset: 0x010 (R/W)  Asynchronous Clock Prescaler Register */
        uint32_t RESERVED1[55U];
  __IOM uint32_t SPPR;                   /*!< Offset: 0x0F0 (R/W)  Selected Pin Protocol Register */
        uint32_t RESERVED2[131U];
  __IM  uint32_t FFSR;                   /*!< Offset: 0x300 (R/ )  Formatter and Flush Status Register */
  __IOM uint32_t FFCR;                   /*!< Offset: 0x304 (R/W)  Formatter and Flush Control Register */
  __IOM uint32_t PSCR;                   /*!< Offset: 0x308 (R/W)  Periodic Synchronization Control Register */
        uint32_t RESERVED3[809U];
  __OM  uint32_t LAR;                    /*!< Offset: 0xFB0 ( /W)  Software Lock Access Register */
  __IM  uint32_t LSR;                    /*!< Offset: 0xFB4 (R/ )  Software Lock Status Register */
        uint32_t RESERVED4[4U];
  __IM  uint32_t TYPE;                   /*!< Offset: 0xFC8 (R/ )  Device Identifier Register */
  __IM  uint32_t DEVTYPE;                /*!< Offset: 0xFCC (R/ )  Device Type Register */
} TPI_Type;

/* TPI Asynchronous Clock Prescaler Register Definitions */
#define TPI_ACPR_SWOSCALER_Pos              0U                                         /*!< TPI ACPR: SWOSCALER Position */
#define TPI_ACPR_SWOSCALER_Msk             (0xFFFFUL /*<< TPI_ACPR_SWOSCALER_Pos*/)    /*!< TPI ACPR: SWOSCALER Mask */

/* TPI Selected Pin Protocol Register Definitions */
#define TPI_SPPR_TXMODE_Pos                 0U                                         /*!< TPI SPPR: TXMODE Position */
#define TPI_SPPR_TXMODE_Msk                (0x3UL /*<< TPI_SPPR_TXMODE_Pos*/)          /*!< TPI SPPR: TXMODE Mask */

/* TPI Formatter and Flush Status Register Definitions */
#define TPI_FFSR_FtNonStop_Pos              3U                                         /*!< TPI FFSR: FtNonStop Position */
#define TPI_FFSR_FtNonStop_Msk             (0x1UL << TPI_FFSR_FtNonStop_Pos)           /*!< TPI FFSR: FtNonStop Mask */

#define TPI_FFSR_TCPresent_Pos              2U                                         /*!< TPI FFSR: TCPresent Position */
#define TPI_FFSR_TCPresent_Msk             (0x1UL << TPI_FFSR_TCPresent_Pos)           /*!< TPI FFSR: TCPresent Mask */

#define TPI_FFSR_FtStopped_Pos              1U                                         /*!< TPI FFSR: FtStopped Position */
#define TPI_FFSR_FtStopped_Msk             (0x1UL << TPI_FFSR_FtStopped_Pos)           /*!< TPI FFSR: FtStopped Mask */

#define TPI_FFSR_FlInProg_Pos               0U                                         /*!< TPI FFSR: FlInProg Position */
#define TPI_FFSR_FlInProg_Msk              (0x1UL /*<< TPI_FFSR_FlInProg_Pos*/)        /*!< TPI FFSR: FlInProg Mask */

/* TPI Formatter and Flush Control Register Definitions */
#define TPI_FFCR_TrigIn_Pos                 8U                                         /*!< TPI FFCR: TrigIn Position */
#define TPI_FFCR_TrigIn_Msk                (0x1UL << TPI_FFCR_TrigIn_Pos)              /*!< TPI FFCR: TrigIn Mask */

#define TPI_FFCR_FOnMan_Pos                 6U                                         /*!< TPI FFCR: FOnMan Position */
#define TPI_FFCR_FOnMan_Msk                (0x1UL << TPI_FFCR_FOnMan_Pos)              /*!< TPI FFCR: FOnMan Mask */

#define TPI_FFCR_EnFCont_Pos                1U                                         /*!< TPI FFCR: EnFCont Position */
#define TPI_FFCR_EnFCont_Msk               (0x1UL << TPI_FFCR_EnFCont_Pos)             /*!< TPI FFCR: EnFCont Mask */

/* TPI Periodic Synchronization Control Register Definitions */
#define TPI_PSCR_PSCount_Pos                0U                                         /*!< TPI PSCR: PSCount Position */
#define TPI_PSCR_PSCount_Msk               (0x1FUL /*<< TPI_PSCR_PSCount_Pos*/)        /*!< TPI PSCR: TPSCount Mask */

/* TPI Software Lock Status Register Definitions */
#define TPI_LSR_nTT_Pos                     1U                                         /*!< TPI LSR: Not thirty-two bit. Position */
#define TPI_LSR_nTT_Msk                    (0x1UL << TPI_LSR_nTT_Pos)                  /*!< TPI LSR: Not thirty-two bit. Mask */

#define TPI_LSR_SLK_Pos                     1U                                         /*!< TPI LSR: Software Lock status Position */
#define TPI_LSR_SLK_Msk                    (0x1UL << TPI_LSR_SLK_Pos)                  /*!< TPI LSR: Software Lock status Mask */

#define TPI_LSR_SLI_Pos                     0U                                         /*!< TPI LSR: Software Lock implemented Position */
#define TPI_LSR_SLI_Msk                    (0x1UL /*<< TPI_LSR_SLI_Pos*/)              /*!< TPI LSR: Software Lock implemented Mask */

/* TPI DEVID Register Definitions */
#define TPI_DEVID_NRZVALID_Pos             11U                                         /*!< TPI DEVID: NRZVALID Position */
#define TPI_DEVID_NRZVALID_Msk             (0x1UL << TPI_DEVID_NRZVALID_Pos)           /*!< TPI DEVID: NRZVALID Mask */

#define TPI_DEVID_MANCVALID_Pos            10U                                         /*!< TPI DEVID: MANCVALID Position */
#define TPI_DEVID_MANCVALID_Msk            (0x1UL << TPI_DEVID_MANCVALID_Pos)          /*!< TPI DEVID: MANCVALID Mask */

#define TPI_DEVID_PTINVALID_Pos             9U                                         /*!< TPI DEVID: PTINVALID Position */
#define TPI_DEVID_PTINVALID_Msk            (0x1UL << TPI_DEVID_PTINVALID_Pos)          /*!< TPI DEVID: PTINVALID Mask */

#define TPI_DEVID_FIFOSZ_Pos                6U                                         /*!< TPI DEVID: FIFO depth Position */
#define TPI_DEVID_FIFOSZ_Msk               (0x7UL << TPI_DEVID_FIFOSZ_Pos)             /*!< TPI DEVID: FIFO depth Mask */

/* TPI DEVTYPE Register Definitions */
#define TPI_DEVTYPE_SubType_Pos             4U                                         /*!< TPI DEVTYPE: SubType Position */
#define TPI_DEVTYPE_SubType_Msk            (0xFUL /*<< TPI_DEVTYPE_SubType_Pos*/)      /*!< TPI DEVTYPE: SubType Mask */

#define TPI_DEVTYPE_MajorType_Pos           0U                                         /*!< TPI DEVTYPE: MajorType Position */
#define TPI_DEVTYPE_MajorType_Msk          (0xFUL << TPI_DEVTYPE_MajorType_Pos)        /*!< TPI DEVTYPE: MajorType Mask */

/*@}*/ /* end of group CMSIS_TPI */


#if defined (__MPU_PRESENT) && (__MPU_PRESENT == 1U)
/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_MPU     Memory Protection Unit (MPU)
  \brief    Type definitions for the Memory Protection Unit (MPU)
  @{
 */

/**
  \brief  Structure type to access the Memory Protection Unit (MPU).
 */
typedef struct
{
  __IM  uint32_t TYPE;                   /*!< Offset: 0x000 (R/ )  MPU Type Register */
  __IOM uint32_t CTRL;                   /*!< Offset: 0x004 (R/W)  MPU Control Register */
  __IOM uint32_t RNR;                    /*!< Offset: 0x008 (R/W)  MPU Region Number Register */
  __IOM uint32_t RBAR;                   /*!< Offset: 0x00C (R/W)  MPU Region Base Address Register */
  __IOM uint32_t RLAR;                   /*!< Offset: 0x010 (R/W)  MPU Region Limit Address Register */
        uint32_t RESERVED0[7U];
  union {
  __IOM uint32_t MAIR[2];
  struct {
  __IOM uint32_t MAIR0;                  /*!< Offset: 0x030 (R/W)  MPU Memory Attribute Indirection Register 0 */
  __IOM uint32_t MAIR1;                  /*!< Offset: 0x034 (R/W)  MPU Memory Attribute Indirection Register 1 */
  };
  };
} MPU_Type;

#define MPU_TYPE_RALIASES                  1U

/* MPU Type Register Definitions */
#define MPU_TYPE_IREGION_Pos               16U                                            /*!< MPU TYPE: IREGION Position */
#define MPU_TYPE_IREGION_Msk               (0xFFUL << MPU_TYPE_IREGION_Pos)               /*!< MPU TYPE: IREGION Mask */

#define MPU_TYPE_DREGION_Pos                8U                                            /*!< MPU TYPE: DREGION Position */
#define MPU_TYPE_DREGION_Msk               (0xFFUL << MPU_TYPE_DREGION_Pos)               /*!< MPU TYPE: DREGION Mask */

#define MPU_TYPE_SEPARATE_Pos               0U                                            /*!< MPU TYPE: SEPARATE Position */
#define MPU_TYPE_SEPARATE_Msk              (1UL /*<< MPU_TYPE_SEPARATE_Pos*/)             /*!< MPU TYPE: SEPARATE Mask */

/* MPU Control Register Definitions */
#define MPU_CTRL_PRIVDEFENA_Pos             2U                                            /*!< MPU CTRL: PRIVDEFENA Position */
#define MPU_CTRL_PRIVDEFENA_Msk            (1UL << MPU_CTRL_PRIVDEFENA_Pos)               /*!< MPU CTRL: PRIVDEFENA Mask */

#define MPU_CTRL_HFNMIENA_Pos               1U                                            /*!< MPU CTRL: HFNMIENA Position */
#define MPU_CTRL_HFNMIENA_Msk              (1UL << MPU_CTRL_HFNMIENA_Pos)                 /*!< MPU CTRL: HFNMIENA Mask */

#define MPU_CTRL_ENABLE_Pos                 0U                                            /*!< MPU CTRL: ENABLE Position */
#define MPU_CTRL_ENABLE_Msk                (1UL /*<< MPU_CTRL_ENABLE_Pos*/)               /*!< MPU CTRL: ENABLE Mask */

/* MPU Region Number Register Definitions */
#define MPU_RNR_REGION_Pos                  0U                                            /*!< MPU RNR: REGION Position */
#define MPU_RNR_REGION_Msk                 (0xFFUL /*<< MPU_RNR_REGION_Pos*/)             /*!< MPU RNR: REGION Mask */

/* MPU Region Base Address Register Definitions */
#define MPU_RBAR_BASE_Pos                   5U                                            /*!< MPU RBAR: BASE Position */
#define MPU_RBAR_BASE_Msk                  (0x7FFFFFFUL << MPU_RBAR_BASE_Pos)             /*!< MPU RBAR: BASE Mask */

#define MPU_RBAR_SH_Pos                     3U                                            /*!< MPU RBAR: SH Position */
#define MPU_RBAR_SH_Msk                    (0x3UL << MPU_RBAR_SH_Pos)                     /*!< MPU RBAR: SH Mask */

#define MPU_RBAR_AP_Pos                     1U                                            /*!< MPU RBAR: AP Position */
#define MPU_RBAR_AP_Msk                    (0x3UL << MPU_RBAR_AP_Pos)                     /*!< MPU RBAR: AP Mask */

#define MPU_RBAR_XN_Pos                     0U                                            /*!< MPU RBAR: XN Position */
#define MPU_RBAR_XN_Msk                    (01UL /*<< MPU_RBAR_XN_Pos*/)                  /*!< MPU RBAR: XN Mask */

/* MPU Region Limit Address Register Definitions */
#define MPU_RLAR_LIMIT_Pos                  5U                                            /*!< MPU RLAR: LIMIT Position */
#define MPU_RLAR_LIMIT_Msk                 (0x7FFFFFFUL << MPU_RLAR_LIMIT_Pos)            /*!< MPU RLAR: LIMIT Mask */

#define MPU_RLAR_AttrIndx_Pos               1U                                            /*!< MPU RLAR: AttrIndx Position */
#define MPU_RLAR_AttrIndx_Msk              (0x7UL << MPU_RLAR_AttrIndx_Pos)               /*!< MPU RLAR: AttrIndx Mask */

#define MPU_RLAR_EN_Pos                     0U                                            /*!< MPU RLAR: EN Position */
#define MPU_RLAR_EN_Msk                    (1UL /*<< MPU_RLAR_EN_Pos*/)                   /*!< MPU RLAR: EN Mask */

/* MPU Memory Attribute Indirection Register 0 Definitions */
#define MPU_MAIR0_Attr3_Pos                24U                                            /*!< MPU MAIR0: Attr3 Position */
#define MPU_MAIR0_Attr3_Msk                (0xFFUL << MPU_MAIR0_Attr3_Pos)                /*!< MPU MAIR0: Attr3 Mask */

#define MPU_MAIR0_Attr2_Pos                16U                                            /*!< MPU MAIR0: Attr2 Position */
#define MPU_MAIR0_Attr2_Msk                (0xFFUL << MPU_MAIR0_Attr2_Pos)                /*!< MPU MAIR0: Attr2 Mask */

#define MPU_MAIR0_Attr1_Pos                 8U                                            /*!< MPU MAIR0: Attr1 Position */
#define MPU_MAIR0_Attr1_Msk                (0xFFUL << MPU_MAIR0_Attr1_Pos)                /*!< MPU MAIR0: Attr1 Mask */

#define MPU_MAIR0_Attr0_Pos                 0U                                            /*!< MPU MAIR0: Attr0 Position */
#define MPU_MAIR0_Attr0_Msk                (0xFFUL /*<< MPU_MAIR0_Attr0_Pos*/)            /*!< MPU MAIR0: Attr0 Mask */

/* MPU Memory Attribute Indirection Register 1 Definitions */
#define MPU_MAIR1_Attr7_Pos                24U                                            /*!< MPU MAIR1: Attr7 Position */
#define MPU_MAIR1_Attr7_Msk                (0xFFUL << MPU_MAIR1_Attr7_Pos)                /*!< MPU MAIR1: Attr7 Mask */

#define MPU_MAIR1_Attr6_Pos                16U                                            /*!< MPU MAIR1: Attr6 Position */
#define MPU_MAIR1_Attr6_Msk                (0xFFUL << MPU_MAIR1_Attr6_Pos)                /*!< MPU MAIR1: Attr6 Mask */

#define MPU_MAIR1_Attr5_Pos                 8U                                            /*!< MPU MAIR1: Attr5 Position */
#define MPU_MAIR1_Attr5_Msk                (0xFFUL << MPU_MAIR1_Attr5_Pos)                /*!< MPU MAIR1: Attr5 Mask */

#define MPU_MAIR1_Attr4_Pos                 0U                                            /*!< MPU MAIR1: Attr4 Position */
#define MPU_MAIR1_Attr4_Msk                (0xFFUL /*<< MPU_MAIR1_Attr4_Pos*/)            /*!< MPU MAIR1: Attr4 Mask */

/*@} end of group CMSIS_MPU */
#endif


#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_SAU     Security Attribution Unit (SAU)
  \brief    Type definitions for the Security Attribution Unit (SAU)
  @{
 */

/**
  \brief  Structure type to access the Security Attribution Unit (SAU).
 */
typedef struct
{
  __IOM uint32_t CTRL;                   /*!< Offset: 0x000 (R/W)  SAU Control Register */
  __IM  uint32_t TYPE;                   /*!< Offset: 0x004 (R/ )  SAU Type Register */
#if defined (__SAUREGION_PRESENT) && (__SAUREGION_PRESENT == 1U)
  __IOM uint32_t RNR;                    /*!< Offset: 0x008 (R/W)  SAU Region Number Register */
  __IOM uint32_t RBAR;                   /*!< Offset: 0x00C (R/W)  SAU Region Base Address Register */
  __IOM uint32_t RLAR;                   /*!< Offset: 0x010 (R/W)  SAU Region Limit Address Register */
#endif
} SAU_Type;

/* SAU Control Register Definitions */
#define SAU_CTRL_ALLNS_Pos                  1U                                            /*!< SAU CTRL: ALLNS Position */
#define SAU_CTRL_ALLNS_Msk                 (1UL << SAU_CTRL_ALLNS_Pos)                    /*!< SAU CTRL: ALLNS Mask */

#define SAU_CTRL_ENABLE_Pos                 0U                                            /*!< SAU CTRL: ENABLE Position */
#define SAU_CTRL_ENABLE_Msk                (1UL /*<< SAU_CTRL_ENABLE_Pos*/)               /*!< SAU CTRL: ENABLE Mask */

/* SAU Type Register Definitions */
#define SAU_TYPE_SREGION_Pos                0U                                            /*!< SAU TYPE: SREGION Position */
#define SAU_TYPE_SREGION_Msk               (0xFFUL /*<< SAU_TYPE_SREGION_Pos*/)           /*!< SAU TYPE: SREGION Mask */

#if defined (__SAUREGION_PRESENT) && (__SAUREGION_PRESENT == 1U)
/* SAU Region Number Register Definitions */
#define SAU_RNR_REGION_Pos                  0U                                            /*!< SAU RNR: REGION Position */
#define SAU_RNR_REGION_Msk                 (0xFFUL /*<< SAU_RNR_REGION_Pos*/)             /*!< SAU RNR: REGION Mask */

/* SAU Region Base Address Register Definitions */
#define SAU_RBAR_BADDR_Pos                  5U                                            /*!< SAU RBAR: BADDR Position */
#define SAU_RBAR_BADDR_Msk                 (0x7FFFFFFUL << SAU_RBAR_BADDR_Pos)            /*!< SAU RBAR: BADDR Mask */

/* SAU Region Limit Address Register Definitions */
#define SAU_RLAR_LADDR_Pos                  5U                                            /*!< SAU RLAR: LADDR Position */
#define SAU_RLAR_LADDR_Msk                 (0x7FFFFFFUL << SAU_RLAR_LADDR_Pos)            /*!< SAU RLAR: LADDR Mask */

#define SAU_RLAR_NSC_Pos                    1U                                            /*!< SAU RLAR: NSC Position */
#define SAU_RLAR_NSC_Msk                   (1UL << SAU_RLAR_NSC_Pos)                      /*!< SAU RLAR: NSC Mask */

#define SAU_RLAR_ENABLE_Pos                 0U                                            /*!< SAU RLAR: ENABLE Position */
#define SAU_RLAR_ENABLE_Msk                (1UL /*<< SAU_RLAR_ENABLE_Pos*/)               /*!< SAU RLAR: ENABLE Mask */

#endif /* defined (__SAUREGION_PRESENT) && (__SAUREGION_PRESENT == 1U) */

/*@} end of group CMSIS_SAU */
#endif /* defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U) */


/**
  \ingroup  CMSIS_core_register
  \defgroup CMSIS_CoreDebug       Core Debug Registers (CoreDebug)
  \brief    Type definitions for the Core Debug Registers
  @{
 */

/**
  \brief  Structure type to access the Core Debug Register (CoreDebug).
 */
typedef struct
{
  __IOM uint32_t DHCSR;                  /*!< Offset: 0x000 (R/W)  Debug Halting Control and Status Register */
  __OM  uint32_t DCRSR;                  /*!< Offset: 0x004 ( /W)  Debug Core Register Selector Register */
  __IOM uint32_t DCRDR;                  /*!< Offset: 0x008 (R/W)  Debug Core Register Data Register */
  __IOM uint32_t DEMCR;                  /*!< Offset: 0x00C (R/W)  Debug Exception and Monitor Control Register */
        uint32_t RESERVED4[1U];
  __IOM uint32_t DAUTHCTRL;              /*!< Offset: 0x014 (R/W)  Debug Authentication Control Register */
  __IOM uint32_t DSCSR;                  /*!< Offset: 0x018 (R/W)  Debug Security Control and Status Register */
} CoreDebug_Type;

/* Debug Halting Control and Status Register Definitions */
#define CoreDebug_DHCSR_DBGKEY_Pos         16U                                            /*!< CoreDebug DHCSR: DBGKEY Position */
#define CoreDebug_DHCSR_DBGKEY_Msk         (0xFFFFUL << CoreDebug_DHCSR_DBGKEY_Pos)       /*!< CoreDebug DHCSR: DBGKEY Mask */

#define CoreDebug_DHCSR_S_RESTART_ST_Pos   26U                                            /*!< CoreDebug DHCSR: S_RESTART_ST Position */
#define CoreDebug_DHCSR_S_RESTART_ST_Msk   (1UL << CoreDebug_DHCSR_S_RESTART_ST_Pos)      /*!< CoreDebug DHCSR: S_RESTART_ST Mask */

#define CoreDebug_DHCSR_S_RESET_ST_Pos     25U                                            /*!< CoreDebug DHCSR: S_RESET_ST Position */
#define CoreDebug_DHCSR_S_RESET_ST_Msk     (1UL << CoreDebug_DHCSR_S_RESET_ST_Pos)        /*!< CoreDebug DHCSR: S_RESET_ST Mask */

#define CoreDebug_DHCSR_S_RETIRE_ST_Pos    24U                                            /*!< CoreDebug DHCSR: S_RETIRE_ST Position */
#define CoreDebug_DHCSR_S_RETIRE_ST_Msk    (1UL << CoreDebug_DHCSR_S_RETIRE_ST_Pos)       /*!< CoreDebug DHCSR: S_RETIRE_ST Mask */

#define CoreDebug_DHCSR_S_LOCKUP_Pos       19U                                            /*!< CoreDebug DHCSR: S_LOCKUP Position */
#define CoreDebug_DHCSR_S_LOCKUP_Msk       (1UL << CoreDebug_DHCSR_S_LOCKUP_Pos)          /*!< CoreDebug DHCSR: S_LOCKUP Mask */

#define CoreDebug_DHCSR_S_SLEEP_Pos        18U                                            /*!< CoreDebug DHCSR: S_SLEEP Position */
#define CoreDebug_DHCSR_S_SLEEP_Msk        (1UL << CoreDebug_DHCSR_S_SLEEP_Pos)           /*!< CoreDebug DHCSR: S_SLEEP Mask */

#define CoreDebug_DHCSR_S_HALT_Pos         17U                                            /*!< CoreDebug DHCSR: S_HALT Position */
#define CoreDebug_DHCSR_S_HALT_Msk         (1UL << CoreDebug_DHCSR_S_HALT_Pos)            /*!< CoreDebug DHCSR: S_HALT Mask */

#define CoreDebug_DHCSR_S_REGRDY_Pos       16U                                            /*!< CoreDebug DHCSR: S_REGRDY Position */
#define CoreDebug_DHCSR_S_REGRDY_Msk       (1UL << CoreDebug_DHCSR_S_REGRDY_Pos)          /*!< CoreDebug DHCSR: S_REGRDY Mask */

#define CoreDebug_DHCSR_C_MASKINTS_Pos      3U                                            /*!< CoreDebug DHCSR: C_MASKINTS Position */
#define CoreDebug_DHCSR_C_MASKINTS_Msk     (1UL << CoreDebug_DHCSR_C_MASKINTS_Pos)        /*!< CoreDebug DHCSR: C_MASKINTS Mask */

#define CoreDebug_DHCSR_C_STEP_Pos          2U                                            /*!< CoreDebug DHCSR: C_STEP Position */
#define CoreDebug_DHCSR_C_STEP_Msk         (1UL << CoreDebug_DHCSR_C_STEP_Pos)            /*!< CoreDebug DHCSR: C_STEP Mask */

#define CoreDebug_DHCSR_C_HALT_Pos          1U                                            /*!< CoreDebug DHCSR: C_HALT Position */
#define CoreDebug_DHCSR_C_HALT_Msk         (1UL << CoreDebug_DHCSR_C_HALT_Pos)            /*!< CoreDebug DHCSR: C_HALT Mask */

#define CoreDebug_DHCSR_C_DEBUGEN_Pos       0U                                            /*!< CoreDebug DHCSR: C_DEBUGEN Position */
#define CoreDebug_DHCSR_C_DEBUGEN_Msk      (1UL /*<< CoreDebug_DHCSR_C_DEBUGEN_Pos*/)     /*!< CoreDebug DHCSR: C_DEBUGEN Mask */

/* Debug Core Register Selector Register Definitions */
#define CoreDebug_DCRSR_REGWnR_Pos         16U                                            /*!< CoreDebug DCRSR: REGWnR Position */
#define CoreDebug_DCRSR_REGWnR_Msk         (1UL << CoreDebug_DCRSR_REGWnR_Pos)            /*!< CoreDebug DCRSR: REGWnR Mask */

#define CoreDebug_DCRSR_REGSEL_Pos          0U                                            /*!< CoreDebug DCRSR: REGSEL Position */
#define CoreDebug_DCRSR_REGSEL_Msk         (0x1FUL /*<< CoreDebug_DCRSR_REGSEL_Pos*/)     /*!< CoreDebug DCRSR: REGSEL Mask */

/* Debug Exception and Monitor Control Register */
#define CoreDebug_DEMCR_DWTENA_Pos         24U                                            /*!< CoreDebug DEMCR: DWTENA Position */
#define CoreDebug_DEMCR_DWTENA_Msk         (1UL << CoreDebug_DEMCR_DWTENA_Pos)            /*!< CoreDebug DEMCR: DWTENA Mask */

#define CoreDebug_DEMCR_VC_HARDERR_Pos     10U                                            /*!< CoreDebug DEMCR: VC_HARDERR Position */
#define CoreDebug_DEMCR_VC_HARDERR_Msk     (1UL << CoreDebug_DEMCR_VC_HARDERR_Pos)        /*!< CoreDebug DEMCR: VC_HARDERR Mask */

#define CoreDebug_DEMCR_VC_CORERESET_Pos    0U                                            /*!< CoreDebug DEMCR: VC_CORERESET Position */
#define CoreDebug_DEMCR_VC_CORERESET_Msk   (1UL /*<< CoreDebug_DEMCR_VC_CORERESET_Pos*/)  /*!< CoreDebug DEMCR: VC_CORERESET Mask */

/* Debug Authentication Control Register Definitions */
#define CoreDebug_DAUTHCTRL_INTSPNIDEN_Pos  3U                                            /*!< CoreDebug DAUTHCTRL: INTSPNIDEN, Position */
#define CoreDebug_DAUTHCTRL_INTSPNIDEN_Msk (1UL << CoreDebug_DAUTHCTRL_INTSPNIDEN_Pos)    /*!< CoreDebug DAUTHCTRL: INTSPNIDEN, Mask */

#define CoreDebug_DAUTHCTRL_SPNIDENSEL_Pos  2U                                            /*!< CoreDebug DAUTHCTRL: SPNIDENSEL Position */
#define CoreDebug_DAUTHCTRL_SPNIDENSEL_Msk (1UL << CoreDebug_DAUTHCTRL_SPNIDENSEL_Pos)    /*!< CoreDebug DAUTHCTRL: SPNIDENSEL Mask */

#define CoreDebug_DAUTHCTRL_INTSPIDEN_Pos   1U                                            /*!< CoreDebug DAUTHCTRL: INTSPIDEN Position */
#define CoreDebug_DAUTHCTRL_INTSPIDEN_Msk  (1UL << CoreDebug_DAUTHCTRL_INTSPIDEN_Pos)     /*!< CoreDebug DAUTHCTRL: INTSPIDEN Mask */

#define CoreDebug_DAUTHCTRL_SPIDENSEL_Pos   0U                                            /*!< CoreDebug DAUTHCTRL: SPIDENSEL Position */
#define CoreDebug_DAUTHCTRL_SPIDENSEL_Msk  (1UL /*<< CoreDebug_DAUTHCTRL_SPIDENSEL_Pos*/) /*!< CoreDebug DAUTHCTRL: SPIDENSEL Mask */

/* Debug Security Control and Status Register Definitions */
#define CoreDebug_DSCSR_CDS_Pos            16U                                            /*!< CoreDebug DSCSR: CDS Position */
#define CoreDebug_DSCSR_CDS_Msk            (1UL << CoreDebug_DSCSR_CDS_Pos)               /*!< CoreDebug DSCSR: CDS Mask */

#define CoreDebug_DSCSR_SBRSEL_Pos          1U                                            /*!< CoreDebug DSCSR: SBRSEL Position */
#define CoreDebug_DSCSR_SBRSEL_Msk         (1UL << CoreDebug_DSCSR_SBRSEL_Pos)            /*!< CoreDebug DSCSR: SBRSEL Mask */

#define CoreDebug_DSCSR_SBRSELEN_Pos        0U                                            /*!< CoreDebug DSCSR: SBRSELEN Position */
#define CoreDebug_DSCSR_SBRSELEN_Msk       (1UL /*<< CoreDebug_DSCSR_SBRSELEN_Pos*/)      /*!< CoreDebug DSCSR: SBRSELEN Mask */

/*@} end of group CMSIS_CoreDebug */


/**
  \ingroup    CMSIS_core_register
  \defgroup   CMSIS_core_bitfield     Core register bit field macros
  \brief      Macros for use with bit field definitions (xxx_Pos, xxx_Msk).
  @{
 */

/**
  \brief   Mask and shift a bit field value for use in a register bit range.
  \param[in] field  Name of the register bit field.
  \param[in] value  Value of the bit field. This parameter is interpreted as an uint32_t type.
  \return           Masked and shifted value.
*/
#define _VAL2FLD(field, value)    (((uint32_t)(value) << field ## _Pos) & field ## _Msk)

/**
  \brief     Mask and shift a register value to extract a bit filed value.
  \param[in] field  Name of the register bit field.
  \param[in] value  Value of register. This parameter is interpreted as an uint32_t type.
  \return           Masked and shifted bit field value.
*/
#define _FLD2VAL(field, value)    (((uint32_t)(value) & field ## _Msk) >> field ## _Pos)

/*@} end of group CMSIS_core_bitfield */


/**
  \ingroup    CMSIS_core_register
  \defgroup   CMSIS_core_base     Core Definitions
  \brief      Definitions for base addresses, unions, and structures.
  @{
 */

/* Memory mapping of Core Hardware */
  #define SCS_BASE            (0xE000E000UL)                             /*!< System Control Space Base Address */
  #define DWT_BASE            (0xE0001000UL)                             /*!< DWT Base Address */
  #define TPI_BASE            (0xE0040000UL)                             /*!< TPI Base Address */
  #define CoreDebug_BASE      (0xE000EDF0UL)                             /*!< Core Debug Base Address */
  #define SysTick_BASE        (SCS_BASE +  0x0010UL)                     /*!< SysTick Base Address */
  #define NVIC_BASE           (SCS_BASE +  0x0100UL)                     /*!< NVIC Base Address */
  #define SCB_BASE            (SCS_BASE +  0x0D00UL)                     /*!< System Control Block Base Address */


  #define SCB                 ((SCB_Type       *)     SCB_BASE         ) /*!< SCB configuration struct */
  #define SysTick             ((SysTick_Type   *)     SysTick_BASE     ) /*!< SysTick configuration struct */
  #define NVIC                ((NVIC_Type      *)     NVIC_BASE        ) /*!< NVIC configuration struct */
  #define DWT                 ((DWT_Type       *)     DWT_BASE         ) /*!< DWT configuration struct */
  #define TPI                 ((TPI_Type       *)     TPI_BASE         ) /*!< TPI configuration struct */
  #define CoreDebug           ((CoreDebug_Type *)     CoreDebug_BASE   ) /*!< Core Debug configuration struct */

  #if defined (__MPU_PRESENT) && (__MPU_PRESENT == 1U)
    #define MPU_BASE          (SCS_BASE +  0x0D90UL)                     /*!< Memory Protection Unit */
    #define MPU               ((MPU_Type       *)     MPU_BASE         ) /*!< Memory Protection Unit */
  #endif

  #if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
    #define SAU_BASE          (SCS_BASE +  0x0DD0UL)                     /*!< Security Attribution Unit */
    #define SAU               ((SAU_Type       *)     SAU_BASE         ) /*!< Security Attribution Unit */
  #endif

#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
  #define SCS_BASE_NS         (0xE002E000UL)                             /*!< System Control Space Base Address (non-secure address space) */
  #define CoreDebug_BASE_NS   (0xE002EDF0UL)                             /*!< Core Debug Base Address           (non-secure address space) */
  #define SysTick_BASE_NS     (SCS_BASE_NS +  0x0010UL)                  /*!< SysTick Base Address              (non-secure address space) */
  #define NVIC_BASE_NS        (SCS_BASE_NS +  0x0100UL)                  /*!< NVIC Base Address                 (non-secure address space) */
  #define SCB_BASE_NS         (SCS_BASE_NS +  0x0D00UL)                  /*!< System Control Block Base Address (non-secure address space) */

  #define SCB_NS              ((SCB_Type       *)     SCB_BASE_NS      ) /*!< SCB configuration struct          (non-secure address space) */
  #define SysTick_NS          ((SysTick_Type   *)     SysTick_BASE_NS  ) /*!< SysTick configuration struct      (non-secure address space) */
  #define NVIC_NS             ((NVIC_Type      *)     NVIC_BASE_NS     ) /*!< NVIC configuration struct         (non-secure address space) */
  #define CoreDebug_NS        ((CoreDebug_Type *)     CoreDebug_BASE_NS) /*!< Core Debug configuration struct   (non-secure address space) */

  #if defined (__MPU_PRESENT) && (__MPU_PRESENT == 1U)
    #define MPU_BASE_NS       (SCS_BASE_NS +  0x0D90UL)                  /*!< Memory Protection Unit            (non-secure address space) */
    #define MPU_NS            ((MPU_Type       *)     MPU_BASE_NS      ) /*!< Memory Protection Unit            (non-secure address space) */
  #endif

#endif /* defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U) */
/*@} */



/*******************************************************************************
 *                Hardware Abstraction Layer
  Core Function Interface contains:
  - Core NVIC Functions
  - Core SysTick Functions
  - Core Register Access Functions
 ******************************************************************************/
/**
  \defgroup CMSIS_Core_FunctionInterface Functions and Instructions Reference
*/



/* ##########################   NVIC functions  #################################### */
/**
  \ingroup  CMSIS_Core_FunctionInterface
  \defgroup CMSIS_Core_NVICFunctions NVIC Functions
  \brief    Functions that manage interrupts and exceptions via the NVIC.
  @{
 */

#ifdef CMSIS_NVIC_VIRTUAL
  #ifndef CMSIS_NVIC_VIRTUAL_HEADER_FILE
    #define CMSIS_NVIC_VIRTUAL_HEADER_FILE "cmsis_nvic_virtual.h"
  #endif
  #include CMSIS_NVIC_VIRTUAL_HEADER_FILE
#else
  #define NVIC_SetPriorityGrouping    __NVIC_SetPriorityGrouping
  #define NVIC_GetPriorityGrouping    __NVIC_GetPriorityGrouping
  #define NVIC_EnableIRQ              __NVIC_EnableIRQ
  #define NVIC_GetEnableIRQ           __NVIC_GetEnableIRQ
  #define NVIC_DisableIRQ             __NVIC_DisableIRQ
  #define NVIC_GetPendingIRQ          __NVIC_GetPendingIRQ
  #define NVIC_SetPendingIRQ          __NVIC_SetPendingIRQ
  #define NVIC_ClearPendingIRQ        __NVIC_ClearPendingIRQ
  #define NVIC_GetActive              __NVIC_GetActive
  #define NVIC_SetPriority            __NVIC_SetPriority
  #define NVIC_GetPriority            __NVIC_GetPriority
  #define NVIC_SystemReset            __NVIC_SystemReset
#endif /* CMSIS_NVIC_VIRTUAL */

#ifdef CMSIS_VECTAB_VIRTUAL
  #ifndef CMSIS_VECTAB_VIRTUAL_HEADER_FILE
    #define CMSIS_VECTAB_VIRTUAL_HEADER_FILE "cmsis_vectab_virtual.h"
  #endif
  #include CMSIS_VECTAB_VIRTUAL_HEADER_FILE
#else
  #define NVIC_SetVector              __NVIC_SetVector
  #define NVIC_GetVector              __NVIC_GetVector
#endif  /* (CMSIS_VECTAB_VIRTUAL) */

#define NVIC_USER_IRQ_OFFSET          16


/* Special LR values for Secure/Non-Secure call handling and exception handling                                               */

/* Function Return Payload (from ARMv8-M Architecture Reference Manual) LR value on entry from Secure BLXNS                   */
#define FNC_RETURN                 (0xFEFFFFFFUL)     /* bit [0] ignored when processing a branch                             */

/* The following EXC_RETURN mask values are used to evaluate the LR on exception entry */
#define EXC_RETURN_PREFIX          (0xFF000000UL)     /* bits [31:24] set to indicate an EXC_RETURN value                     */
#define EXC_RETURN_S               (0x00000040UL)     /* bit [6] stack used to push registers: 0=Non-secure 1=Secure          */
#define EXC_RETURN_DCRS            (0x00000020UL)     /* bit [5] stacking rules for called registers: 0=skipped 1=saved       */
#define EXC_RETURN_FTYPE           (0x00000010UL)     /* bit [4] allocate stack for floating-point context: 0=done 1=skipped  */
#define EXC_RETURN_MODE            (0x00000008UL)     /* bit [3] processor mode for return: 0=Handler mode 1=Thread mode      */
#define EXC_RETURN_SPSEL           (0x00000002UL)     /* bit [1] stack pointer used to restore context: 0=MSP 1=PSP           */
#define EXC_RETURN_ES              (0x00000001UL)     /* bit [0] security state exception was taken to: 0=Non-secure 1=Secure */

/* Integrity Signature (from ARMv8-M Architecture Reference Manual) for exception context stacking                            */
#if defined (__FPU_PRESENT) && (__FPU_PRESENT == 1U)  /* Value for processors with floating-point extension:                  */
#define EXC_INTEGRITY_SIGNATURE     (0xFEFA125AUL)     /* bit [0] SFTC must match LR bit[4] EXC_RETURN_FTYPE                   */
#else
#define EXC_INTEGRITY_SIGNATURE     (0xFEFA125BUL)     /* Value for processors without floating-point extension                */
#endif


/* Interrupt Priorities are WORD accessible only under Armv6-M                  */
/* The following MACROS handle generation of the register offset and byte masks */
#define _BIT_SHIFT(IRQn)         (  ((((uint32_t)(int32_t)(IRQn))         )      &  0x03UL) * 8UL)
#define _SHP_IDX(IRQn)           ( (((((uint32_t)(int32_t)(IRQn)) & 0x0FUL)-8UL) >>    2UL)      )
#define _IP_IDX(IRQn)            (   (((uint32_t)(int32_t)(IRQn))                >>    2UL)      )

#define __NVIC_SetPriorityGrouping(X) (void)(X)
#define __NVIC_GetPriorityGrouping()  (0U)

/**
  \brief   Enable Interrupt
  \details Enables a device specific interrupt in the NVIC interrupt controller.
  \param [in]      IRQn  Device specific interrupt number.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE void __NVIC_EnableIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC->ISER[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}


/**
  \brief   Get Interrupt Enable status
  \details Returns a device specific interrupt enable status from the NVIC interrupt controller.
  \param [in]      IRQn  Device specific interrupt number.
  \return             0  Interrupt is not enabled.
  \return             1  Interrupt is enabled.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE uint32_t __NVIC_GetEnableIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((NVIC->ISER[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}


/**
  \brief   Disable Interrupt
  \details Disables a device specific interrupt in the NVIC interrupt controller.
  \param [in]      IRQn  Device specific interrupt number.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE void __NVIC_DisableIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC->ICER[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
    __DSB();
    __ISB();
  }
}


/**
  \brief   Get Pending Interrupt
  \details Reads the NVIC pending register and returns the pending bit for the specified device specific interrupt.
  \param [in]      IRQn  Device specific interrupt number.
  \return             0  Interrupt status is not pending.
  \return             1  Interrupt status is pending.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE uint32_t __NVIC_GetPendingIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((NVIC->ISPR[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}


/**
  \brief   Set Pending Interrupt
  \details Sets the pending bit of a device specific interrupt in the NVIC pending register.
  \param [in]      IRQn  Device specific interrupt number.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE void __NVIC_SetPendingIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC->ISPR[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}


/**
  \brief   Clear Pending Interrupt
  \details Clears the pending bit of a device specific interrupt in the NVIC pending register.
  \param [in]      IRQn  Device specific interrupt number.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE void __NVIC_ClearPendingIRQ(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC->ICPR[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}


/**
  \brief   Get Active Interrupt
  \details Reads the active register in the NVIC and returns the active bit for the device specific interrupt.
  \param [in]      IRQn  Device specific interrupt number.
  \return             0  Interrupt status is not active.
  \return             1  Interrupt status is active.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE uint32_t __NVIC_GetActive(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((NVIC->IABR[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}


#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Get Interrupt Target State
  \details Reads the interrupt target field in the NVIC and returns the interrupt target bit for the device specific interrupt.
  \param [in]      IRQn  Device specific interrupt number.
  \return             0  if interrupt is assigned to Secure
  \return             1  if interrupt is assigned to Non Secure
  \note    IRQn must not be negative.
 */
__STATIC_INLINE uint32_t NVIC_GetTargetState(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((NVIC->ITNS[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}


/**
  \brief   Set Interrupt Target State
  \details Sets the interrupt target field in the NVIC and returns the interrupt target bit for the device specific interrupt.
  \param [in]      IRQn  Device specific interrupt number.
  \return             0  if interrupt is assigned to Secure
                      1  if interrupt is assigned to Non Secure
  \note    IRQn must not be negative.
 */
__STATIC_INLINE uint32_t NVIC_SetTargetState(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC->ITNS[(((uint32_t)IRQn) >> 5UL)] |=  ((uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL)));
    return((uint32_t)(((NVIC->ITNS[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}


/**
  \brief   Clear Interrupt Target State
  \details Clears the interrupt target field in the NVIC and returns the interrupt target bit for the device specific interrupt.
  \param [in]      IRQn  Device specific interrupt number.
  \return             0  if interrupt is assigned to Secure
                      1  if interrupt is assigned to Non Secure
  \note    IRQn must not be negative.
 */
__STATIC_INLINE uint32_t NVIC_ClearTargetState(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC->ITNS[(((uint32_t)IRQn) >> 5UL)] &= ~((uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL)));
    return((uint32_t)(((NVIC->ITNS[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}
#endif /* defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U) */


/**
  \brief   Set Interrupt Priority
  \details Sets the priority of a device specific interrupt or a processor exception.
           The interrupt number can be positive to specify a device specific interrupt,
           or negative to specify a processor exception.
  \param [in]      IRQn  Interrupt number.
  \param [in]  priority  Priority to set.
  \note    The priority cannot be set for every processor exception.
 */
__STATIC_INLINE void __NVIC_SetPriority(IRQn_Type IRQn, uint32_t priority)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC->IPR[_IP_IDX(IRQn)]  = ((uint32_t)(NVIC->IPR[_IP_IDX(IRQn)]  & ~(0xFFUL << _BIT_SHIFT(IRQn))) |
       (((priority << (8U - __NVIC_PRIO_BITS)) & (uint32_t)0xFFUL) << _BIT_SHIFT(IRQn)));
  }
  else
  {
    SCB->SHPR[_SHP_IDX(IRQn)] = ((uint32_t)(SCB->SHPR[_SHP_IDX(IRQn)] & ~(0xFFUL << _BIT_SHIFT(IRQn))) |
       (((priority << (8U - __NVIC_PRIO_BITS)) & (uint32_t)0xFFUL) << _BIT_SHIFT(IRQn)));
  }
}


/**
  \brief   Get Interrupt Priority
  \details Reads the priority of a device specific interrupt or a processor exception.
           The interrupt number can be positive to specify a device specific interrupt,
           or negative to specify a processor exception.
  \param [in]   IRQn  Interrupt number.
  \return             Interrupt Priority.
                      Value is aligned automatically to the implemented priority bits of the microcontroller.
 */
__STATIC_INLINE uint32_t __NVIC_GetPriority(IRQn_Type IRQn)
{

  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((NVIC->IPR[ _IP_IDX(IRQn)] >> _BIT_SHIFT(IRQn) ) & (uint32_t)0xFFUL) >> (8U - __NVIC_PRIO_BITS)));
  }
  else
  {
    return((uint32_t)(((SCB->SHPR[_SHP_IDX(IRQn)] >> _BIT_SHIFT(IRQn) ) & (uint32_t)0xFFUL) >> (8U - __NVIC_PRIO_BITS)));
  }
}


/**
  \brief   Encode Priority
  \details Encodes the priority for an interrupt with the given priority group,
           preemptive priority value, and subpriority value.
           In case of a conflict between priority grouping and available
           priority bits (__NVIC_PRIO_BITS), the smallest possible priority group is set.
  \param [in]     PriorityGroup  Used priority group.
  \param [in]   PreemptPriority  Preemptive priority value (starting from 0).
  \param [in]       SubPriority  Subpriority value (starting from 0).
  \return                        Encoded priority. Value can be used in the function \ref NVIC_SetPriority().
 */
__STATIC_INLINE uint32_t NVIC_EncodePriority (uint32_t PriorityGroup, uint32_t PreemptPriority, uint32_t SubPriority)
{
  uint32_t PriorityGroupTmp = (PriorityGroup & (uint32_t)0x07UL);   /* only values 0..7 are used          */
  uint32_t PreemptPriorityBits;
  uint32_t SubPriorityBits;

  PreemptPriorityBits = ((7UL - PriorityGroupTmp) > (uint32_t)(__NVIC_PRIO_BITS)) ? (uint32_t)(__NVIC_PRIO_BITS) : (uint32_t)(7UL - PriorityGroupTmp);
  SubPriorityBits     = ((PriorityGroupTmp + (uint32_t)(__NVIC_PRIO_BITS)) < (uint32_t)7UL) ? (uint32_t)0UL : (uint32_t)((PriorityGroupTmp - 7UL) + (uint32_t)(__NVIC_PRIO_BITS));

  return (
           ((PreemptPriority & (uint32_t)((1UL << (PreemptPriorityBits)) - 1UL)) << SubPriorityBits) |
           ((SubPriority     & (uint32_t)((1UL << (SubPriorityBits    )) - 1UL)))
         );
}


/**
  \brief   Decode Priority
  \details Decodes an interrupt priority value with a given priority group to
           preemptive priority value and subpriority value.
           In case of a conflict between priority grouping and available
           priority bits (__NVIC_PRIO_BITS) the smallest possible priority group is set.
  \param [in]         Priority   Priority value, which can be retrieved with the function \ref NVIC_GetPriority().
  \param [in]     PriorityGroup  Used priority group.
  \param [out] pPreemptPriority  Preemptive priority value (starting from 0).
  \param [out]     pSubPriority  Subpriority value (starting from 0).
 */
__STATIC_INLINE void NVIC_DecodePriority (uint32_t Priority, uint32_t PriorityGroup, uint32_t* const pPreemptPriority, uint32_t* const pSubPriority)
{
  uint32_t PriorityGroupTmp = (PriorityGroup & (uint32_t)0x07UL);   /* only values 0..7 are used          */
  uint32_t PreemptPriorityBits;
  uint32_t SubPriorityBits;

  PreemptPriorityBits = ((7UL - PriorityGroupTmp) > (uint32_t)(__NVIC_PRIO_BITS)) ? (uint32_t)(__NVIC_PRIO_BITS) : (uint32_t)(7UL - PriorityGroupTmp);
  SubPriorityBits     = ((PriorityGroupTmp + (uint32_t)(__NVIC_PRIO_BITS)) < (uint32_t)7UL) ? (uint32_t)0UL : (uint32_t)((PriorityGroupTmp - 7UL) + (uint32_t)(__NVIC_PRIO_BITS));

  *pPreemptPriority = (Priority >> SubPriorityBits) & (uint32_t)((1UL << (PreemptPriorityBits)) - 1UL);
  *pSubPriority     = (Priority                   ) & (uint32_t)((1UL << (SubPriorityBits    )) - 1UL);
}


/**
  \brief   Set Interrupt Vector
  \details Sets an interrupt vector in SRAM based interrupt vector table.
           The interrupt number can be positive to specify a device specific interrupt,
           or negative to specify a processor exception.
           VTOR must been relocated to SRAM before.
           If VTOR is not present address 0 must be mapped to SRAM.
  \param [in]   IRQn      Interrupt number
  \param [in]   vector    Address of interrupt handler function
 */
__STATIC_INLINE void __NVIC_SetVector(IRQn_Type IRQn, uint32_t vector)
{
#if defined (__VTOR_PRESENT) && (__VTOR_PRESENT == 1U)
  uint32_t *vectors = (uint32_t *)SCB->VTOR;
#else
  uint32_t *vectors = (uint32_t *)0x0U;
#endif
  vectors[(int32_t)IRQn + NVIC_USER_IRQ_OFFSET] = vector;
}


/**
  \brief   Get Interrupt Vector
  \details Reads an interrupt vector from interrupt vector table.
           The interrupt number can be positive to specify a device specific interrupt,
           or negative to specify a processor exception.
  \param [in]   IRQn      Interrupt number.
  \return                 Address of interrupt handler function
 */
__STATIC_INLINE uint32_t __NVIC_GetVector(IRQn_Type IRQn)
{
#if defined (__VTOR_PRESENT) && (__VTOR_PRESENT == 1U)
  uint32_t *vectors = (uint32_t *)SCB->VTOR;
#else
  uint32_t *vectors = (uint32_t *)0x0U;
#endif
  return vectors[(int32_t)IRQn + NVIC_USER_IRQ_OFFSET];
}


/**
  \brief   System Reset
  \details Initiates a system reset request to reset the MCU.
 */
__NO_RETURN __STATIC_INLINE void __NVIC_SystemReset(void)
{
  __DSB();                                                          /* Ensure all outstanding memory accesses included
                                                                       buffered write are completed before reset */
  SCB->AIRCR  = ((0x5FAUL << SCB_AIRCR_VECTKEY_Pos) |
                 SCB_AIRCR_SYSRESETREQ_Msk);
  __DSB();                                                          /* Ensure completion of memory access */

  for(;;)                                                           /* wait until reset */
  {
    __NOP();
  }
}

#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   Enable Interrupt (non-secure)
  \details Enables a device specific interrupt in the non-secure NVIC interrupt controller when in secure state.
  \param [in]      IRQn  Device specific interrupt number.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE void TZ_NVIC_EnableIRQ_NS(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC_NS->ISER[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}


/**
  \brief   Get Interrupt Enable status (non-secure)
  \details Returns a device specific interrupt enable status from the non-secure NVIC interrupt controller when in secure state.
  \param [in]      IRQn  Device specific interrupt number.
  \return             0  Interrupt is not enabled.
  \return             1  Interrupt is enabled.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE uint32_t TZ_NVIC_GetEnableIRQ_NS(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((NVIC_NS->ISER[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}


/**
  \brief   Disable Interrupt (non-secure)
  \details Disables a device specific interrupt in the non-secure NVIC interrupt controller when in secure state.
  \param [in]      IRQn  Device specific interrupt number.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE void TZ_NVIC_DisableIRQ_NS(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC_NS->ICER[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}


/**
  \brief   Get Pending Interrupt (non-secure)
  \details Reads the NVIC pending register in the non-secure NVIC when in secure state and returns the pending bit for the specified device specific interrupt.
  \param [in]      IRQn  Device specific interrupt number.
  \return             0  Interrupt status is not pending.
  \return             1  Interrupt status is pending.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE uint32_t TZ_NVIC_GetPendingIRQ_NS(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((NVIC_NS->ISPR[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}


/**
  \brief   Set Pending Interrupt (non-secure)
  \details Sets the pending bit of a device specific interrupt in the non-secure NVIC pending register when in secure state.
  \param [in]      IRQn  Device specific interrupt number.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE void TZ_NVIC_SetPendingIRQ_NS(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC_NS->ISPR[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}


/**
  \brief   Clear Pending Interrupt (non-secure)
  \details Clears the pending bit of a device specific interrupt in the non-secure NVIC pending register when in secure state.
  \param [in]      IRQn  Device specific interrupt number.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE void TZ_NVIC_ClearPendingIRQ_NS(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC_NS->ICPR[(((uint32_t)IRQn) >> 5UL)] = (uint32_t)(1UL << (((uint32_t)IRQn) & 0x1FUL));
  }
}


/**
  \brief   Get Active Interrupt (non-secure)
  \details Reads the active register in non-secure NVIC when in secure state and returns the active bit for the device specific interrupt.
  \param [in]      IRQn  Device specific interrupt number.
  \return             0  Interrupt status is not active.
  \return             1  Interrupt status is active.
  \note    IRQn must not be negative.
 */
__STATIC_INLINE uint32_t TZ_NVIC_GetActive_NS(IRQn_Type IRQn)
{
  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((NVIC_NS->IABR[(((uint32_t)IRQn) >> 5UL)] & (1UL << (((uint32_t)IRQn) & 0x1FUL))) != 0UL) ? 1UL : 0UL));
  }
  else
  {
    return(0U);
  }
}


/**
  \brief   Set Interrupt Priority (non-secure)
  \details Sets the priority of a non-secure device specific interrupt or a non-secure processor exception when in secure state.
           The interrupt number can be positive to specify a device specific interrupt,
           or negative to specify a processor exception.
  \param [in]      IRQn  Interrupt number.
  \param [in]  priority  Priority to set.
  \note    The priority cannot be set for every non-secure processor exception.
 */
__STATIC_INLINE void TZ_NVIC_SetPriority_NS(IRQn_Type IRQn, uint32_t priority)
{
  if ((int32_t)(IRQn) >= 0)
  {
    NVIC_NS->IPR[_IP_IDX(IRQn)]  = ((uint32_t)(NVIC_NS->IPR[_IP_IDX(IRQn)]  & ~(0xFFUL << _BIT_SHIFT(IRQn))) |
       (((priority << (8U - __NVIC_PRIO_BITS)) & (uint32_t)0xFFUL) << _BIT_SHIFT(IRQn)));
  }
  else
  {
    SCB_NS->SHPR[_SHP_IDX(IRQn)] = ((uint32_t)(SCB_NS->SHPR[_SHP_IDX(IRQn)] & ~(0xFFUL << _BIT_SHIFT(IRQn))) |
       (((priority << (8U - __NVIC_PRIO_BITS)) & (uint32_t)0xFFUL) << _BIT_SHIFT(IRQn)));
  }
}


/**
  \brief   Get Interrupt Priority (non-secure)
  \details Reads the priority of a non-secure device specific interrupt or a non-secure processor exception when in secure state.
           The interrupt number can be positive to specify a device specific interrupt,
           or negative to specify a processor exception.
  \param [in]   IRQn  Interrupt number.
  \return             Interrupt Priority. Value is aligned automatically to the implemented priority bits of the microcontroller.
 */
__STATIC_INLINE uint32_t TZ_NVIC_GetPriority_NS(IRQn_Type IRQn)
{

  if ((int32_t)(IRQn) >= 0)
  {
    return((uint32_t)(((NVIC_NS->IPR[ _IP_IDX(IRQn)] >> _BIT_SHIFT(IRQn) ) & (uint32_t)0xFFUL) >> (8U - __NVIC_PRIO_BITS)));
  }
  else
  {
    return((uint32_t)(((SCB_NS->SHPR[_SHP_IDX(IRQn)] >> _BIT_SHIFT(IRQn) ) & (uint32_t)0xFFUL) >> (8U - __NVIC_PRIO_BITS)));
  }
}
#endif /*  defined (__ARM_FEATURE_CMSE) &&(__ARM_FEATURE_CMSE == 3U) */

/*@} end of CMSIS_Core_NVICFunctions */

/* ##########################  MPU functions  #################################### */

#if defined (__MPU_PRESENT) && (__MPU_PRESENT == 1U)

#include "mpu_armv8.h"

#endif

/* ##########################  FPU functions  #################################### */
/**
  \ingroup  CMSIS_Core_FunctionInterface
  \defgroup CMSIS_Core_FpuFunctions FPU Functions
  \brief    Function that provides FPU type.
  @{
 */

/**
  \brief   get FPU type
  \details returns the FPU type
  \returns
   - \b  0: No FPU
   - \b  1: Single precision FPU
   - \b  2: Double + Single precision FPU
 */
__STATIC_INLINE uint32_t SCB_GetFPUType(void)
{
    return 0U;           /* No FPU */
}


/*@} end of CMSIS_Core_FpuFunctions */



/* ##########################   SAU functions  #################################### */
/**
  \ingroup  CMSIS_Core_FunctionInterface
  \defgroup CMSIS_Core_SAUFunctions SAU Functions
  \brief    Functions that configure the SAU.
  @{
 */

#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)

/**
  \brief   Enable SAU
  \details Enables the Security Attribution Unit (SAU).
 */
__STATIC_INLINE void TZ_SAU_Enable(void)
{
    SAU->CTRL |=  (SAU_CTRL_ENABLE_Msk);
}



/**
  \brief   Disable SAU
  \details Disables the Security Attribution Unit (SAU).
 */
__STATIC_INLINE void TZ_SAU_Disable(void)
{
    SAU->CTRL &= ~(SAU_CTRL_ENABLE_Msk);
}

#endif /* defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U) */

/*@} end of CMSIS_Core_SAUFunctions */




/* ##################################    SysTick function  ############################################ */
/**
  \ingroup  CMSIS_Core_FunctionInterface
  \defgroup CMSIS_Core_SysTickFunctions SysTick Functions
  \brief    Functions that configure the System.
  @{
 */

#if defined (__Vendor_SysTickConfig) && (__Vendor_SysTickConfig == 0U)

/**
  \brief   System Tick Configuration
  \details Initializes the System Timer and its interrupt, and starts the System Tick Timer.
           Counter is in free running mode to generate periodic interrupts.
  \param [in]  ticks  Number of ticks between two interrupts.
  \return          0  Function succeeded.
  \return          1  Function failed.
  \note    When the variable <b>__Vendor_SysTickConfig</b> is set to 1, then the
           function <b>SysTick_Config</b> is not included. In this case, the file <b><i>device</i>.h</b>
           must contain a vendor-specific implementation of this function.
 */
__STATIC_INLINE uint32_t SysTick_Config(uint32_t ticks)
{
  if ((ticks - 1UL) > SysTick_LOAD_RELOAD_Msk)
  {
    return (1UL);                                                   /* Reload value impossible */
  }

  SysTick->LOAD  = (uint32_t)(ticks - 1UL);                         /* set reload register */
  NVIC_SetPriority (SysTick_IRQn, (1UL << __NVIC_PRIO_BITS) - 1UL); /* set Priority for Systick Interrupt */
  SysTick->VAL   = 0UL;                                             /* Load the SysTick Counter Value */
  SysTick->CTRL  = SysTick_CTRL_CLKSOURCE_Msk |
                   SysTick_CTRL_TICKINT_Msk   |
                   SysTick_CTRL_ENABLE_Msk;                         /* Enable SysTick IRQ and SysTick Timer */
  return (0UL);                                                     /* Function successful */
}

#if defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U)
/**
  \brief   System Tick Configuration (non-secure)
  \details Initializes the non-secure System Timer and its interrupt when in secure state, and starts the System Tick Timer.
           Counter is in free running mode to generate periodic interrupts.
  \param [in]  ticks  Number of ticks between two interrupts.
  \return          0  Function succeeded.
  \return          1  Function failed.
  \note    When the variable <b>__Vendor_SysTickConfig</b> is set to 1, then the
           function <b>TZ_SysTick_Config_NS</b> is not included. In this case, the file <b><i>device</i>.h</b>
           must contain a vendor-specific implementation of this function.

 */
__STATIC_INLINE uint32_t TZ_SysTick_Config_NS(uint32_t ticks)
{
  if ((ticks - 1UL) > SysTick_LOAD_RELOAD_Msk)
  {
    return (1UL);                                                         /* Reload value impossible */
  }

  SysTick_NS->LOAD  = (uint32_t)(ticks - 1UL);                            /* set reload register */
  TZ_NVIC_SetPriority_NS (SysTick_IRQn, (1UL << __NVIC_PRIO_BITS) - 1UL); /* set Priority for Systick Interrupt */
  SysTick_NS->VAL   = 0UL;                                                /* Load the SysTick Counter Value */
  SysTick_NS->CTRL  = SysTick_CTRL_CLKSOURCE_Msk |
                      SysTick_CTRL_TICKINT_Msk   |
                      SysTick_CTRL_ENABLE_Msk;                            /* Enable SysTick IRQ and SysTick Timer */
  return (0UL);                                                           /* Function successful */
}
#endif /* defined (__ARM_FEATURE_CMSE) && (__ARM_FEATURE_CMSE == 3U) */

#endif

/*@} end of CMSIS_Core_SysTickFunctions */




#ifdef __cplusplus
}
#endif

#endif /* __CORE_ARMV8MBL_H_DEPENDANT */

#endif /* __CMSIS_GENERIC */
