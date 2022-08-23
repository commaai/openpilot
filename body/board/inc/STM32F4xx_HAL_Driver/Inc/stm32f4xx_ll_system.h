/**
  ******************************************************************************
  * @file    stm32f4xx_ll_system.h
  * @author  MCD Application Team
  * @brief   Header file of SYSTEM LL module.
  @verbatim
  ==============================================================================
                     ##### How to use this driver #####
  ==============================================================================
    [..]
    The LL SYSTEM driver contains a set of generic APIs that can be
    used by user:
      (+) Some of the FLASH features need to be handled in the SYSTEM file.
      (+) Access to DBGCMU registers
      (+) Access to SYSCFG registers

  @endverbatim
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM32F4xx_LL_SYSTEM_H
#define __STM32F4xx_LL_SYSTEM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (FLASH) || defined (SYSCFG) || defined (DBGMCU)

/** @defgroup SYSTEM_LL SYSTEM
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/

/* Private constants ---------------------------------------------------------*/
/** @defgroup SYSTEM_LL_Private_Constants SYSTEM Private Constants
  * @{
  */

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/

/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
/** @defgroup SYSTEM_LL_Exported_Constants SYSTEM Exported Constants
  * @{
  */

/** @defgroup SYSTEM_LL_EC_REMAP SYSCFG REMAP
* @{
*/
#define LL_SYSCFG_REMAP_FLASH              (uint32_t)0x00000000                                  /*!< Main Flash memory mapped at 0x00000000              */
#define LL_SYSCFG_REMAP_SYSTEMFLASH        SYSCFG_MEMRMP_MEM_MODE_0                              /*!< System Flash memory mapped at 0x00000000            */
#if defined(FSMC_Bank1)
#define LL_SYSCFG_REMAP_FSMC               SYSCFG_MEMRMP_MEM_MODE_1                              /*!< FSMC(NOR/PSRAM 1 and 2) mapped at 0x00000000        */
#endif /* FSMC_Bank1 */
#if defined(FMC_Bank1)
#define LL_SYSCFG_REMAP_FMC                SYSCFG_MEMRMP_MEM_MODE_1                              /*!< FMC(NOR/PSRAM 1 and 2) mapped at 0x00000000         */
#define LL_SYSCFG_REMAP_SDRAM              SYSCFG_MEMRMP_MEM_MODE_2                              /*!< FMC/SDRAM mapped at 0x00000000                      */
#endif /* FMC_Bank1 */
#define LL_SYSCFG_REMAP_SRAM               (SYSCFG_MEMRMP_MEM_MODE_1 | SYSCFG_MEMRMP_MEM_MODE_0) /*!< SRAM1 mapped at 0x00000000                          */

/**
  * @}
  */

#if defined(SYSCFG_PMC_MII_RMII_SEL)
 /** @defgroup SYSTEM_LL_EC_PMC SYSCFG PMC
* @{
*/
#define LL_SYSCFG_PMC_ETHMII               (uint32_t)0x00000000                                /*!< ETH Media MII interface */
#define LL_SYSCFG_PMC_ETHRMII              (uint32_t)SYSCFG_PMC_MII_RMII_SEL                   /*!< ETH Media RMII interface */

/**
  * @}
  */
#endif /* SYSCFG_PMC_MII_RMII_SEL */



#if defined(SYSCFG_MEMRMP_UFB_MODE)
/** @defgroup SYSTEM_LL_EC_BANKMODE SYSCFG BANK MODE
  * @{
  */
#define LL_SYSCFG_BANKMODE_BANK1          (uint32_t)0x00000000       /*!< Flash Bank 1 base address mapped at 0x0800 0000 (AXI) and 0x0020 0000 (TCM)
                                                                      and Flash Bank 2 base address mapped at 0x0810 0000 (AXI) and 0x0030 0000 (TCM)*/
#define LL_SYSCFG_BANKMODE_BANK2          SYSCFG_MEMRMP_UFB_MODE     /*!< Flash Bank 2 base address mapped at 0x0800 0000 (AXI) and 0x0020 0000(TCM)
                                                                      and Flash Bank 1 base address mapped at 0x0810 0000 (AXI) and 0x0030 0000(TCM) */
/**
  * @}
  */
#endif /* SYSCFG_MEMRMP_UFB_MODE */
/** @defgroup SYSTEM_LL_EC_I2C_FASTMODEPLUS SYSCFG I2C FASTMODEPLUS
  * @{
  */ 
#if defined(SYSCFG_CFGR_FMPI2C1_SCL)
#define LL_SYSCFG_I2C_FASTMODEPLUS_SCL         SYSCFG_CFGR_FMPI2C1_SCL   /*!< Enable Fast Mode Plus on FMPI2C_SCL pin */
#define LL_SYSCFG_I2C_FASTMODEPLUS_SDA         SYSCFG_CFGR_FMPI2C1_SDA   /*!< Enable Fast Mode Plus on FMPI2C_SDA pin*/
#endif /* SYSCFG_CFGR_FMPI2C1_SCL */
/**
  * @}
  */

/** @defgroup SYSTEM_LL_EC_EXTI_PORT SYSCFG EXTI PORT
  * @{
  */
#define LL_SYSCFG_EXTI_PORTA               (uint32_t)0               /*!< EXTI PORT A                        */
#define LL_SYSCFG_EXTI_PORTB               (uint32_t)1               /*!< EXTI PORT B                        */
#define LL_SYSCFG_EXTI_PORTC               (uint32_t)2               /*!< EXTI PORT C                        */
#define LL_SYSCFG_EXTI_PORTD               (uint32_t)3               /*!< EXTI PORT D                        */
#define LL_SYSCFG_EXTI_PORTE               (uint32_t)4               /*!< EXTI PORT E                        */
#if defined(GPIOF)
#define LL_SYSCFG_EXTI_PORTF               (uint32_t)5               /*!< EXTI PORT F                        */
#endif /* GPIOF */
#if defined(GPIOG)
#define LL_SYSCFG_EXTI_PORTG               (uint32_t)6               /*!< EXTI PORT G                        */
#endif /* GPIOG */
#define LL_SYSCFG_EXTI_PORTH               (uint32_t)7               /*!< EXTI PORT H                        */
#if defined(GPIOI)
#define LL_SYSCFG_EXTI_PORTI               (uint32_t)8               /*!< EXTI PORT I                        */
#endif /* GPIOI */
#if defined(GPIOJ)
#define LL_SYSCFG_EXTI_PORTJ               (uint32_t)9               /*!< EXTI PORT J                        */
#endif /* GPIOJ */
#if defined(GPIOK)
#define LL_SYSCFG_EXTI_PORTK               (uint32_t)10              /*!< EXTI PORT k                        */
#endif /* GPIOK */
/**
  * @}
  */

/** @defgroup SYSTEM_LL_EC_EXTI_LINE SYSCFG EXTI LINE
  * @{
  */
#define LL_SYSCFG_EXTI_LINE0               (uint32_t)(0x000FU << 16 | 0)  /*!< EXTI_POSITION_0  | EXTICR[0] */
#define LL_SYSCFG_EXTI_LINE1               (uint32_t)(0x00F0U << 16 | 0)  /*!< EXTI_POSITION_4  | EXTICR[0] */
#define LL_SYSCFG_EXTI_LINE2               (uint32_t)(0x0F00U << 16 | 0)  /*!< EXTI_POSITION_8  | EXTICR[0] */
#define LL_SYSCFG_EXTI_LINE3               (uint32_t)(0xF000U << 16 | 0)  /*!< EXTI_POSITION_12 | EXTICR[0] */
#define LL_SYSCFG_EXTI_LINE4               (uint32_t)(0x000FU << 16 | 1)  /*!< EXTI_POSITION_0  | EXTICR[1] */
#define LL_SYSCFG_EXTI_LINE5               (uint32_t)(0x00F0U << 16 | 1)  /*!< EXTI_POSITION_4  | EXTICR[1] */
#define LL_SYSCFG_EXTI_LINE6               (uint32_t)(0x0F00U << 16 | 1)  /*!< EXTI_POSITION_8  | EXTICR[1] */
#define LL_SYSCFG_EXTI_LINE7               (uint32_t)(0xF000U << 16 | 1)  /*!< EXTI_POSITION_12 | EXTICR[1] */
#define LL_SYSCFG_EXTI_LINE8               (uint32_t)(0x000FU << 16 | 2)  /*!< EXTI_POSITION_0  | EXTICR[2] */
#define LL_SYSCFG_EXTI_LINE9               (uint32_t)(0x00F0U << 16 | 2)  /*!< EXTI_POSITION_4  | EXTICR[2] */
#define LL_SYSCFG_EXTI_LINE10              (uint32_t)(0x0F00U << 16 | 2)  /*!< EXTI_POSITION_8  | EXTICR[2] */
#define LL_SYSCFG_EXTI_LINE11              (uint32_t)(0xF000U << 16 | 2)  /*!< EXTI_POSITION_12 | EXTICR[2] */
#define LL_SYSCFG_EXTI_LINE12              (uint32_t)(0x000FU << 16 | 3)  /*!< EXTI_POSITION_0  | EXTICR[3] */
#define LL_SYSCFG_EXTI_LINE13              (uint32_t)(0x00F0U << 16 | 3)  /*!< EXTI_POSITION_4  | EXTICR[3] */
#define LL_SYSCFG_EXTI_LINE14              (uint32_t)(0x0F00U << 16 | 3)  /*!< EXTI_POSITION_8  | EXTICR[3] */
#define LL_SYSCFG_EXTI_LINE15              (uint32_t)(0xF000U << 16 | 3)  /*!< EXTI_POSITION_12 | EXTICR[3] */
/**
  * @}
  */

/** @defgroup SYSTEM_LL_EC_TIMBREAK SYSCFG TIMER BREAK
  * @{
  */
#if defined(SYSCFG_CFGR2_LOCKUP_LOCK)
#define LL_SYSCFG_TIMBREAK_LOCKUP          SYSCFG_CFGR2_LOCKUP_LOCK   /*!< Enables and locks the LOCKUP output of CortexM4 
                                                                      with Break Input of TIM1/8                                    */
#define LL_SYSCFG_TIMBREAK_PVD             SYSCFG_CFGR2_PVD_LOCK      /*!< Enables and locks the PVD connection with TIM1/8 Break Input 
                                                                      and also the PVDE and PLS bits of the Power Control Interface  */
#endif /* SYSCFG_CFGR2_CLL */
/**
  * @}
  */

#if defined(SYSCFG_MCHDLYCR_BSCKSEL)
/** @defgroup SYSTEM_LL_DFSDM_BitStream_ClockSource SYSCFG MCHDLY BCKKSEL
  * @{
  */
#define LL_SYSCFG_BITSTREAM_CLOCK_TIM2OC1          (uint32_t)0x00000000
#define LL_SYSCFG_BITSTREAM_CLOCK_DFSDM2           SYSCFG_MCHDLYCR_BSCKSEL
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM_MCHDLYEN              SYSCFG MCHDLY MCHDLYEN
  * @{
  */  
#define LL_SYSCFG_DFSDM1_MCHDLYEN                  SYSCFG_MCHDLYCR_MCHDLY1EN
#define LL_SYSCFG_DFSDM2_MCHDLYEN                  SYSCFG_MCHDLYCR_MCHDLY2EN
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM_DataIn0_Source       SYSCFG MCHDLY DFSDMD0SEL
  * @{
  */
#define LL_SYSCFG_DFSDM1_DataIn0                   SYSCFG_MCHDLYCR_DFSDM1D0SEL
#define LL_SYSCFG_DFSDM2_DataIn0                   SYSCFG_MCHDLYCR_DFSDM2D0SEL

#define LL_SYSCFG_DFSDM1_DataIn0_PAD               (uint32_t)((SYSCFG_MCHDLYCR_DFSDM1D0SEL << 16) | 0x00000000)
#define LL_SYSCFG_DFSDM1_DataIn0_DM                (uint32_t)((SYSCFG_MCHDLYCR_DFSDM1D0SEL << 16) | SYSCFG_MCHDLYCR_DFSDM1D0SEL)
#define LL_SYSCFG_DFSDM2_DataIn0_PAD               (uint32_t)((SYSCFG_MCHDLYCR_DFSDM2D0SEL << 16) | 0x00000000)
#define LL_SYSCFG_DFSDM2_DataIn0_DM                (uint32_t)((SYSCFG_MCHDLYCR_DFSDM2D0SEL << 16) | SYSCFG_MCHDLYCR_DFSDM2D0SEL)
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM_DataIn2_Source       SYSCFG MCHDLY DFSDMD2SEL
  * @{
  */   
#define LL_SYSCFG_DFSDM1_DataIn2                   SYSCFG_MCHDLYCR_DFSDM1D2SEL
#define LL_SYSCFG_DFSDM2_DataIn2                   SYSCFG_MCHDLYCR_DFSDM2D2SEL

#define LL_SYSCFG_DFSDM1_DataIn2_PAD               (uint32_t)((SYSCFG_MCHDLYCR_DFSDM1D2SEL << 16) | 0x00000000)
#define LL_SYSCFG_DFSDM1_DataIn2_DM                (uint32_t)((SYSCFG_MCHDLYCR_DFSDM1D2SEL << 16) | SYSCFG_MCHDLYCR_DFSDM1D2SEL)
#define LL_SYSCFG_DFSDM2_DataIn2_PAD               (uint32_t)((SYSCFG_MCHDLYCR_DFSDM2D2SEL << 16) | 0x00000000)
#define LL_SYSCFG_DFSDM2_DataIn2_DM                (uint32_t)((SYSCFG_MCHDLYCR_DFSDM2D2SEL << 16) | SYSCFG_MCHDLYCR_DFSDM2D2SEL)
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM1_TIM4OC2_BitstreamDistribution  SYSCFG MCHDLY DFSDM1CK02SEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM1_TIM4OC2_CLKIN0           (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM1_TIM4OC2_CLKIN2           SYSCFG_MCHDLYCR_DFSDM1CK02SEL
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM1_TIM4OC1_BitstreamDistribution  SYSCFG MCHDLY DFSDM1CK13SEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM1_TIM4OC1_CLKIN1           (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM1_TIM4OC1_CLKIN3           SYSCFG_MCHDLYCR_DFSDM1CK13SEL
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM1_CLKIN_SourceSelection SYSCFG MCHDLY DFSDMCFG
  * @{
  */
#define LL_SYSCFG_DFSDM1_CKIN_PAD                 (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM1_CKIN_DM                  SYSCFG_MCHDLYCR_DFSDM1CFG
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM1_CLKOUT_SourceSelection SYSCFG MCHDLY DFSDM1CKOSEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM1_CKOUT                    (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM1_CKOUT_M27                SYSCFG_MCHDLYCR_DFSDM1CKOSEL
/**
  * @}
  */

/** @defgroup SYSTEM_LL_DFSDM2_DataIn4_SourceSelection SYSCFG MCHDLY DFSDM2D4SEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM2_DataIn4_PAD              (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM2_DataIn4_DM               SYSCFG_MCHDLYCR_DFSDM2D4SEL
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM2_DataIn6_SourceSelection SYSCFG MCHDLY DFSDM2D6SEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM2_DataIn6_PAD              (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM2_DataIn6_DM               SYSCFG_MCHDLYCR_DFSDM2D6SEL
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM2_TIM3OC4_BitstreamDistribution  SYSCFG MCHDLY DFSDM2CK04SEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM2_TIM3OC4_CLKIN0           (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM2_TIM3OC4_CLKIN4           SYSCFG_MCHDLYCR_DFSDM2CK04SEL
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM2_TIM3OC3_BitstreamDistribution  SYSCFG MCHDLY DFSDM2CK15SEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM2_TIM3OC3_CLKIN1           (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM2_TIM3OC3_CLKIN5           SYSCFG_MCHDLYCR_DFSDM2CK15SEL
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM2_TIM3OC2_BitstreamDistribution  SYSCFG MCHDLY DFSDM2CK26SEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM2_TIM3OC2_CLKIN2           (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM2_TIM3OC2_CLKIN6           SYSCFG_MCHDLYCR_DFSDM2CK26SEL
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM2_TIM3OC1_BitstreamDistribution  SYSCFG MCHDLY DFSDM2CK37SEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM2_TIM3OC1_CLKIN3           (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM2_TIM3OC1_CLKIN7           SYSCFG_MCHDLYCR_DFSDM2CK37SEL
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM2_CLKIN_SourceSelection SYSCFG MCHDLY DFSDM2CFG
  * @{
  */ 
#define LL_SYSCFG_DFSDM2_CKIN_PAD                 (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM2_CKIN_DM                  SYSCFG_MCHDLYCR_DFSDM2CFG
/**
  * @}
  */
/** @defgroup SYSTEM_LL_DFSDM2_CLKOUT_SourceSelection SYSCFG MCHDLY DFSDM2CKOSEL
  * @{
  */ 
#define LL_SYSCFG_DFSDM2_CKOUT                    (uint32_t)0x00000000
#define LL_SYSCFG_DFSDM2_CKOUT_M27                SYSCFG_MCHDLYCR_DFSDM2CKOSEL
/**
  * @}
  */ 
#endif /* SYSCFG_MCHDLYCR_BSCKSEL */  

/** @defgroup SYSTEM_LL_EC_TRACE DBGMCU TRACE Pin Assignment
  * @{
  */
#define LL_DBGMCU_TRACE_NONE               0x00000000U                                     /*!< TRACE pins not assigned (default state) */
#define LL_DBGMCU_TRACE_ASYNCH             DBGMCU_CR_TRACE_IOEN                            /*!< TRACE pin assignment for Asynchronous Mode */
#define LL_DBGMCU_TRACE_SYNCH_SIZE1        (DBGMCU_CR_TRACE_IOEN | DBGMCU_CR_TRACE_MODE_0) /*!< TRACE pin assignment for Synchronous Mode with a TRACEDATA size of 1 */
#define LL_DBGMCU_TRACE_SYNCH_SIZE2        (DBGMCU_CR_TRACE_IOEN | DBGMCU_CR_TRACE_MODE_1) /*!< TRACE pin assignment for Synchronous Mode with a TRACEDATA size of 2 */
#define LL_DBGMCU_TRACE_SYNCH_SIZE4        (DBGMCU_CR_TRACE_IOEN | DBGMCU_CR_TRACE_MODE)   /*!< TRACE pin assignment for Synchronous Mode with a TRACEDATA size of 4 */
/**
  * @}
  */

/** @defgroup SYSTEM_LL_EC_APB1_GRP1_STOP_IP DBGMCU APB1 GRP1 STOP IP
  * @{
  */
#if defined(DBGMCU_APB1_FZ_DBG_TIM2_STOP)
#define LL_DBGMCU_APB1_GRP1_TIM2_STOP      DBGMCU_APB1_FZ_DBG_TIM2_STOP          /*!< TIM2 counter stopped when core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_TIM2_STOP */
#if defined(DBGMCU_APB1_FZ_DBG_TIM3_STOP)
#define LL_DBGMCU_APB1_GRP1_TIM3_STOP      DBGMCU_APB1_FZ_DBG_TIM3_STOP          /*!< TIM3 counter stopped when core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_TIM3_STOP */
#if defined(DBGMCU_APB1_FZ_DBG_TIM4_STOP)
#define LL_DBGMCU_APB1_GRP1_TIM4_STOP      DBGMCU_APB1_FZ_DBG_TIM4_STOP          /*!< TIM4 counter stopped when core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_TIM4_STOP */
#define LL_DBGMCU_APB1_GRP1_TIM5_STOP      DBGMCU_APB1_FZ_DBG_TIM5_STOP          /*!< TIM5 counter stopped when core is halted */
#if defined(DBGMCU_APB1_FZ_DBG_TIM6_STOP)
#define LL_DBGMCU_APB1_GRP1_TIM6_STOP      DBGMCU_APB1_FZ_DBG_TIM6_STOP          /*!< TIM6 counter stopped when core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_TIM6_STOP */
#if defined(DBGMCU_APB1_FZ_DBG_TIM7_STOP)
#define LL_DBGMCU_APB1_GRP1_TIM7_STOP      DBGMCU_APB1_FZ_DBG_TIM7_STOP          /*!< TIM7 counter stopped when core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_TIM7_STOP */
#if defined(DBGMCU_APB1_FZ_DBG_TIM12_STOP)
#define LL_DBGMCU_APB1_GRP1_TIM12_STOP     DBGMCU_APB1_FZ_DBG_TIM12_STOP         /*!< TIM12 counter stopped when core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_TIM12_STOP */
#if defined(DBGMCU_APB1_FZ_DBG_TIM13_STOP)
#define LL_DBGMCU_APB1_GRP1_TIM13_STOP     DBGMCU_APB1_FZ_DBG_TIM13_STOP         /*!< TIM13 counter stopped when core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_TIM13_STOP */
#if defined(DBGMCU_APB1_FZ_DBG_TIM14_STOP)
#define LL_DBGMCU_APB1_GRP1_TIM14_STOP     DBGMCU_APB1_FZ_DBG_TIM14_STOP         /*!< TIM14 counter stopped when core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_TIM14_STOP */
#if defined(DBGMCU_APB1_FZ_DBG_LPTIM_STOP)
#define LL_DBGMCU_APB1_GRP1_LPTIM_STOP     DBGMCU_APB1_FZ_DBG_LPTIM_STOP         /*!< LPTIM counter stopped when core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_LPTIM_STOP */
#define LL_DBGMCU_APB1_GRP1_RTC_STOP       DBGMCU_APB1_FZ_DBG_RTC_STOP           /*!< RTC counter stopped when core is halted */
#define LL_DBGMCU_APB1_GRP1_WWDG_STOP      DBGMCU_APB1_FZ_DBG_WWDG_STOP          /*!< Debug Window Watchdog stopped when Core is halted */
#define LL_DBGMCU_APB1_GRP1_IWDG_STOP      DBGMCU_APB1_FZ_DBG_IWDG_STOP          /*!< Debug Independent Watchdog stopped when Core is halted */
#define LL_DBGMCU_APB1_GRP1_I2C1_STOP      DBGMCU_APB1_FZ_DBG_I2C1_SMBUS_TIMEOUT /*!< I2C1 SMBUS timeout mode stopped when Core is halted */
#define LL_DBGMCU_APB1_GRP1_I2C2_STOP      DBGMCU_APB1_FZ_DBG_I2C2_SMBUS_TIMEOUT /*!< I2C2 SMBUS timeout mode stopped when Core is halted */
#if defined(DBGMCU_APB1_FZ_DBG_I2C3_SMBUS_TIMEOUT)
#define LL_DBGMCU_APB1_GRP1_I2C3_STOP      DBGMCU_APB1_FZ_DBG_I2C3_SMBUS_TIMEOUT /*!< I2C3 SMBUS timeout mode stopped when Core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_I2C3_SMBUS_TIMEOUT */
#if defined(DBGMCU_APB1_FZ_DBG_I2C4_SMBUS_TIMEOUT)
#define LL_DBGMCU_APB1_GRP1_I2C4_STOP      DBGMCU_APB1_FZ_DBG_I2C4_SMBUS_TIMEOUT /*!< I2C4 SMBUS timeout mode stopped when Core is halted */
#endif /* DBGMCU_APB1_FZ_DBG_I2C4_SMBUS_TIMEOUT */
#if defined(DBGMCU_APB1_FZ_DBG_CAN1_STOP)
#define LL_DBGMCU_APB1_GRP1_CAN1_STOP      DBGMCU_APB1_FZ_DBG_CAN1_STOP          /*!< CAN1 debug stopped when Core is halted  */
#endif /* DBGMCU_APB1_FZ_DBG_CAN1_STOP */
#if defined(DBGMCU_APB1_FZ_DBG_CAN2_STOP)
#define LL_DBGMCU_APB1_GRP1_CAN2_STOP      DBGMCU_APB1_FZ_DBG_CAN2_STOP          /*!< CAN2 debug stopped when Core is halted  */
#endif /* DBGMCU_APB1_FZ_DBG_CAN2_STOP */
#if defined(DBGMCU_APB1_FZ_DBG_CAN3_STOP)
#define LL_DBGMCU_APB1_GRP1_CAN3_STOP      DBGMCU_APB1_FZ_DBG_CAN3_STOP          /*!< CAN3 debug stopped when Core is halted  */
#endif /* DBGMCU_APB1_FZ_DBG_CAN3_STOP */
/**
  * @}
  */

/** @defgroup SYSTEM_LL_EC_APB2_GRP1_STOP_IP DBGMCU APB2 GRP1 STOP IP
  * @{
  */
#define LL_DBGMCU_APB2_GRP1_TIM1_STOP      DBGMCU_APB2_FZ_DBG_TIM1_STOP   /*!< TIM1 counter stopped when core is halted */
#if defined(DBGMCU_APB2_FZ_DBG_TIM8_STOP)
#define LL_DBGMCU_APB2_GRP1_TIM8_STOP      DBGMCU_APB2_FZ_DBG_TIM8_STOP   /*!< TIM8 counter stopped when core is halted */
#endif /* DBGMCU_APB2_FZ_DBG_TIM8_STOP */
#define LL_DBGMCU_APB2_GRP1_TIM9_STOP      DBGMCU_APB2_FZ_DBG_TIM9_STOP   /*!< TIM9 counter stopped when core is halted */
#if defined(DBGMCU_APB2_FZ_DBG_TIM10_STOP)
#define LL_DBGMCU_APB2_GRP1_TIM10_STOP     DBGMCU_APB2_FZ_DBG_TIM10_STOP   /*!< TIM10 counter stopped when core is halted */
#endif /* DBGMCU_APB2_FZ_DBG_TIM10_STOP */
#define LL_DBGMCU_APB2_GRP1_TIM11_STOP     DBGMCU_APB2_FZ_DBG_TIM11_STOP   /*!< TIM11 counter stopped when core is halted */
/**
  * @}
  */

/** @defgroup SYSTEM_LL_EC_LATENCY FLASH LATENCY
  * @{
  */
#define LL_FLASH_LATENCY_0                 FLASH_ACR_LATENCY_0WS   /*!< FLASH Zero wait state */
#define LL_FLASH_LATENCY_1                 FLASH_ACR_LATENCY_1WS   /*!< FLASH One wait state */
#define LL_FLASH_LATENCY_2                 FLASH_ACR_LATENCY_2WS   /*!< FLASH Two wait states */
#define LL_FLASH_LATENCY_3                 FLASH_ACR_LATENCY_3WS   /*!< FLASH Three wait states */
#define LL_FLASH_LATENCY_4                 FLASH_ACR_LATENCY_4WS   /*!< FLASH Four wait states */
#define LL_FLASH_LATENCY_5                 FLASH_ACR_LATENCY_5WS   /*!< FLASH five wait state */
#define LL_FLASH_LATENCY_6                 FLASH_ACR_LATENCY_6WS   /*!< FLASH six wait state */
#define LL_FLASH_LATENCY_7                 FLASH_ACR_LATENCY_7WS   /*!< FLASH seven wait states */
#define LL_FLASH_LATENCY_8                 FLASH_ACR_LATENCY_8WS   /*!< FLASH eight wait states */
#define LL_FLASH_LATENCY_9                 FLASH_ACR_LATENCY_9WS   /*!< FLASH nine wait states */
#define LL_FLASH_LATENCY_10                FLASH_ACR_LATENCY_10WS   /*!< FLASH ten wait states */
#define LL_FLASH_LATENCY_11                FLASH_ACR_LATENCY_11WS   /*!< FLASH eleven wait states */
#define LL_FLASH_LATENCY_12                FLASH_ACR_LATENCY_12WS   /*!< FLASH twelve wait states */
#define LL_FLASH_LATENCY_13                FLASH_ACR_LATENCY_13WS   /*!< FLASH thirteen wait states */
#define LL_FLASH_LATENCY_14                FLASH_ACR_LATENCY_14WS   /*!< FLASH fourteen wait states */
#define LL_FLASH_LATENCY_15                FLASH_ACR_LATENCY_15WS   /*!< FLASH fifteen wait states */
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @defgroup SYSTEM_LL_Exported_Functions SYSTEM Exported Functions
  * @{
  */

/** @defgroup SYSTEM_LL_EF_SYSCFG SYSCFG
  * @{
  */
/**
  * @brief  Set memory mapping at address 0x00000000
  * @rmtoll SYSCFG_MEMRMP MEM_MODE      LL_SYSCFG_SetRemapMemory
  * @param  Memory This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_REMAP_FLASH
  *         @arg @ref LL_SYSCFG_REMAP_SYSTEMFLASH
  *         @arg @ref LL_SYSCFG_REMAP_SRAM
  *         @arg @ref LL_SYSCFG_REMAP_FSMC (*)
  *         @arg @ref LL_SYSCFG_REMAP_FMC (*)
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_SetRemapMemory(uint32_t Memory)
{
  MODIFY_REG(SYSCFG->MEMRMP, SYSCFG_MEMRMP_MEM_MODE, Memory);
}

/**
  * @brief  Get memory mapping at address 0x00000000
  * @rmtoll SYSCFG_MEMRMP MEM_MODE      LL_SYSCFG_GetRemapMemory
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_REMAP_FLASH
  *         @arg @ref LL_SYSCFG_REMAP_SYSTEMFLASH
  *         @arg @ref LL_SYSCFG_REMAP_SRAM
  *         @arg @ref LL_SYSCFG_REMAP_FSMC (*)
  *         @arg @ref LL_SYSCFG_REMAP_FMC (*)
  */
__STATIC_INLINE uint32_t LL_SYSCFG_GetRemapMemory(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MEMRMP, SYSCFG_MEMRMP_MEM_MODE));
}

#if defined(SYSCFG_MEMRMP_SWP_FMC)
/**
  * @brief  Enables the FMC Memory Mapping Swapping
  * @rmtoll SYSCFG_MEMRMP SWP_FMC      LL_SYSCFG_EnableFMCMemorySwapping
  * @note   SDRAM is accessible at 0x60000000 and NOR/RAM 
  *         is accessible at 0xC0000000   
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_EnableFMCMemorySwapping(void)
{
  SET_BIT(SYSCFG->MEMRMP, SYSCFG_MEMRMP_SWP_FMC_0);
}

/**
  * @brief  Disables the FMC Memory Mapping Swapping
  * @rmtoll SYSCFG_MEMRMP SWP_FMC      LL_SYSCFG_DisableFMCMemorySwapping
  * @note   SDRAM is accessible at 0xC0000000 (default mapping)  
  *         and NOR/RAM is accessible at 0x60000000 (default mapping)
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DisableFMCMemorySwapping(void)
{
  CLEAR_BIT(SYSCFG->MEMRMP, SYSCFG_MEMRMP_SWP_FMC);
}

#endif /* SYSCFG_MEMRMP_SWP_FMC */
/**
  * @brief  Enables the Compensation cell Power Down
  * @rmtoll SYSCFG_CMPCR CMP_PD      LL_SYSCFG_EnableCompensationCell
  * @note   The I/O compensation cell can be used only when the device supply
  *         voltage ranges from 2.4 to 3.6 V
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_EnableCompensationCell(void)
{
  SET_BIT(SYSCFG->CMPCR, SYSCFG_CMPCR_CMP_PD);
}

/**
  * @brief  Disables the Compensation cell Power Down
  * @rmtoll SYSCFG_CMPCR CMP_PD      LL_SYSCFG_DisableCompensationCell
  * @note   The I/O compensation cell can be used only when the device supply
  *         voltage ranges from 2.4 to 3.6 V
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DisableCompensationCell(void)
{
  CLEAR_BIT(SYSCFG->CMPCR, SYSCFG_CMPCR_CMP_PD);
}

/**
  * @brief  Get Compensation Cell ready Flag
  * @rmtoll SYSCFG_CMPCR READY  LL_SYSCFG_IsActiveFlag_CMPCR
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SYSCFG_IsActiveFlag_CMPCR(void)
{
  return (READ_BIT(SYSCFG->CMPCR, SYSCFG_CMPCR_READY) == (SYSCFG_CMPCR_READY));
}

#if defined(SYSCFG_PMC_MII_RMII_SEL)
/**
  * @brief  Select Ethernet PHY interface 
  * @rmtoll SYSCFG_PMC MII_RMII_SEL       LL_SYSCFG_SetPHYInterface
  * @param  Interface This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_PMC_ETHMII
  *         @arg @ref LL_SYSCFG_PMC_ETHRMII
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_SetPHYInterface(uint32_t Interface)
{
  MODIFY_REG(SYSCFG->PMC, SYSCFG_PMC_MII_RMII_SEL, Interface);
}

/**
  * @brief  Get Ethernet PHY interface 
  * @rmtoll SYSCFG_PMC MII_RMII_SEL       LL_SYSCFG_GetPHYInterface
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_PMC_ETHMII
  *         @arg @ref LL_SYSCFG_PMC_ETHRMII
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_GetPHYInterface(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->PMC, SYSCFG_PMC_MII_RMII_SEL));
}
#endif /* SYSCFG_PMC_MII_RMII_SEL */
 


#if defined(SYSCFG_MEMRMP_UFB_MODE)
/**
  * @brief  Select Flash bank mode (Bank flashed at 0x08000000)
  * @rmtoll SYSCFG_MEMRMP UFB_MODE       LL_SYSCFG_SetFlashBankMode
  * @param  Bank This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_BANKMODE_BANK1
  *         @arg @ref LL_SYSCFG_BANKMODE_BANK2
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_SetFlashBankMode(uint32_t Bank)
{ 
  MODIFY_REG(SYSCFG->MEMRMP, SYSCFG_MEMRMP_UFB_MODE, Bank);
}

/**
  * @brief  Get Flash bank mode (Bank flashed at 0x08000000)
  * @rmtoll SYSCFG_MEMRMP UFB_MODE       LL_SYSCFG_GetFlashBankMode
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_BANKMODE_BANK1
  *         @arg @ref LL_SYSCFG_BANKMODE_BANK2
  */
__STATIC_INLINE uint32_t LL_SYSCFG_GetFlashBankMode(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MEMRMP, SYSCFG_MEMRMP_UFB_MODE));
}
#endif /* SYSCFG_MEMRMP_UFB_MODE */

#if defined(SYSCFG_CFGR_FMPI2C1_SCL)
/**
  * @brief  Enable the I2C fast mode plus driving capability.
  * @rmtoll SYSCFG_CFGR FMPI2C1_SCL   LL_SYSCFG_EnableFastModePlus\n
  *         SYSCFG_CFGR FMPI2C1_SDA   LL_SYSCFG_EnableFastModePlus
  * @param  ConfigFastModePlus This parameter can be a combination of the following values:
  *         @arg @ref LL_SYSCFG_I2C_FASTMODEPLUS_SCL
  *         @arg @ref LL_SYSCFG_I2C_FASTMODEPLUS_SDA
  *         (*) value not defined in all devices
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_EnableFastModePlus(uint32_t ConfigFastModePlus)
{
  SET_BIT(SYSCFG->CFGR, ConfigFastModePlus);
}

/**
  * @brief  Disable the I2C fast mode plus driving capability.
  * @rmtoll SYSCFG_CFGR FMPI2C1_SCL  LL_SYSCFG_DisableFastModePlus\n
  *         SYSCFG_CFGR FMPI2C1_SDA  LL_SYSCFG_DisableFastModePlus\n
  * @param  ConfigFastModePlus This parameter can be a combination of the following values:
  *         @arg @ref LL_SYSCFG_I2C_FASTMODEPLUS_SCL
  *         @arg @ref LL_SYSCFG_I2C_FASTMODEPLUS_SDA
  *         (*) value not defined in all devices
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DisableFastModePlus(uint32_t ConfigFastModePlus)
{
  CLEAR_BIT(SYSCFG->CFGR, ConfigFastModePlus);
}
#endif /* SYSCFG_CFGR_FMPI2C1_SCL */

/**
  * @brief  Configure source input for the EXTI external interrupt.
  * @rmtoll SYSCFG_EXTICR1 EXTIx         LL_SYSCFG_SetEXTISource\n
  *         SYSCFG_EXTICR2 EXTIx         LL_SYSCFG_SetEXTISource\n
  *         SYSCFG_EXTICR3 EXTIx         LL_SYSCFG_SetEXTISource\n
  *         SYSCFG_EXTICR4 EXTIx         LL_SYSCFG_SetEXTISource
  * @param  Port This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_EXTI_PORTA
  *         @arg @ref LL_SYSCFG_EXTI_PORTB
  *         @arg @ref LL_SYSCFG_EXTI_PORTC
  *         @arg @ref LL_SYSCFG_EXTI_PORTD
  *         @arg @ref LL_SYSCFG_EXTI_PORTE
  *         @arg @ref LL_SYSCFG_EXTI_PORTF (*)
  *         @arg @ref LL_SYSCFG_EXTI_PORTG (*)
  *         @arg @ref LL_SYSCFG_EXTI_PORTH
  *
  *         (*) value not defined in all devices
  * @param  Line This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_EXTI_LINE0
  *         @arg @ref LL_SYSCFG_EXTI_LINE1
  *         @arg @ref LL_SYSCFG_EXTI_LINE2
  *         @arg @ref LL_SYSCFG_EXTI_LINE3
  *         @arg @ref LL_SYSCFG_EXTI_LINE4
  *         @arg @ref LL_SYSCFG_EXTI_LINE5
  *         @arg @ref LL_SYSCFG_EXTI_LINE6
  *         @arg @ref LL_SYSCFG_EXTI_LINE7
  *         @arg @ref LL_SYSCFG_EXTI_LINE8
  *         @arg @ref LL_SYSCFG_EXTI_LINE9
  *         @arg @ref LL_SYSCFG_EXTI_LINE10
  *         @arg @ref LL_SYSCFG_EXTI_LINE11
  *         @arg @ref LL_SYSCFG_EXTI_LINE12
  *         @arg @ref LL_SYSCFG_EXTI_LINE13
  *         @arg @ref LL_SYSCFG_EXTI_LINE14
  *         @arg @ref LL_SYSCFG_EXTI_LINE15
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_SetEXTISource(uint32_t Port, uint32_t Line)
{
  MODIFY_REG(SYSCFG->EXTICR[Line & 0xFF], (Line >> 16), Port << POSITION_VAL((Line >> 16)));
}

/**
  * @brief  Get the configured defined for specific EXTI Line
  * @rmtoll SYSCFG_EXTICR1 EXTIx         LL_SYSCFG_GetEXTISource\n
  *         SYSCFG_EXTICR2 EXTIx         LL_SYSCFG_GetEXTISource\n
  *         SYSCFG_EXTICR3 EXTIx         LL_SYSCFG_GetEXTISource\n
  *         SYSCFG_EXTICR4 EXTIx         LL_SYSCFG_GetEXTISource
  * @param  Line This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_EXTI_LINE0
  *         @arg @ref LL_SYSCFG_EXTI_LINE1
  *         @arg @ref LL_SYSCFG_EXTI_LINE2
  *         @arg @ref LL_SYSCFG_EXTI_LINE3
  *         @arg @ref LL_SYSCFG_EXTI_LINE4
  *         @arg @ref LL_SYSCFG_EXTI_LINE5
  *         @arg @ref LL_SYSCFG_EXTI_LINE6
  *         @arg @ref LL_SYSCFG_EXTI_LINE7
  *         @arg @ref LL_SYSCFG_EXTI_LINE8
  *         @arg @ref LL_SYSCFG_EXTI_LINE9
  *         @arg @ref LL_SYSCFG_EXTI_LINE10
  *         @arg @ref LL_SYSCFG_EXTI_LINE11
  *         @arg @ref LL_SYSCFG_EXTI_LINE12
  *         @arg @ref LL_SYSCFG_EXTI_LINE13
  *         @arg @ref LL_SYSCFG_EXTI_LINE14
  *         @arg @ref LL_SYSCFG_EXTI_LINE15
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_EXTI_PORTA
  *         @arg @ref LL_SYSCFG_EXTI_PORTB
  *         @arg @ref LL_SYSCFG_EXTI_PORTC
  *         @arg @ref LL_SYSCFG_EXTI_PORTD
  *         @arg @ref LL_SYSCFG_EXTI_PORTE
  *         @arg @ref LL_SYSCFG_EXTI_PORTF (*)
  *         @arg @ref LL_SYSCFG_EXTI_PORTG (*)
  *         @arg @ref LL_SYSCFG_EXTI_PORTH
  *         (*) value not defined in all devices
  */
__STATIC_INLINE uint32_t LL_SYSCFG_GetEXTISource(uint32_t Line)
{
  return (uint32_t)(READ_BIT(SYSCFG->EXTICR[Line & 0xFF], (Line >> 16)) >> POSITION_VAL(Line >> 16));
}

#if defined(SYSCFG_CFGR2_LOCKUP_LOCK)
/**
  * @brief  Set connections to TIM1/8 break inputs
  * @rmtoll SYSCFG_CFGR2 LockUp Lock           LL_SYSCFG_SetTIMBreakInputs \n
  *         SYSCFG_CFGR2 PVD Lock              LL_SYSCFG_SetTIMBreakInputs
  * @param  Break This parameter can be a combination of the following values:
  *         @arg @ref LL_SYSCFG_TIMBREAK_LOCKUP
  *         @arg @ref LL_SYSCFG_TIMBREAK_PVD
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_SetTIMBreakInputs(uint32_t Break)
{
  MODIFY_REG(SYSCFG->CFGR2, SYSCFG_CFGR2_LOCKUP_LOCK | SYSCFG_CFGR2_PVD_LOCK, Break);
}

/**
  * @brief  Get connections to TIM1/8 Break inputs
  * @rmtoll SYSCFG_CFGR2 LockUp Lock           LL_SYSCFG_SetTIMBreakInputs \n
  *         SYSCFG_CFGR2 PVD Lock              LL_SYSCFG_SetTIMBreakInputs
  * @retval Returned value can be can be a combination of the following values:
  *         @arg @ref LL_SYSCFG_TIMBREAK_LOCKUP
  *         @arg @ref LL_SYSCFG_TIMBREAK_PVD
  */
__STATIC_INLINE uint32_t LL_SYSCFG_GetTIMBreakInputs(void)
{   
  return (uint32_t)(READ_BIT(SYSCFG->CFGR2, SYSCFG_CFGR2_LOCKUP_LOCK | SYSCFG_CFGR2_PVD_LOCK));
}
#endif /* SYSCFG_CFGR2_LOCKUP_LOCK */
#if defined(SYSCFG_MCHDLYCR_BSCKSEL)
/**
  * @brief  Select the DFSDM2 or TIM2_OC1 as clock source for the bitstream clock.
  * @rmtoll SYSCFG_MCHDLYCR BSCKSEL        LL_SYSCFG_DFSDM_SetBitstreamClockSourceSelection
  * @param  ClockSource This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_BITSTREAM_CLOCK_DFSDM2
  *         @arg @ref LL_SYSCFG_BITSTREAM_CLOCK_TIM2OC1
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM_SetBitstreamClockSourceSelection(uint32_t ClockSource)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_BSCKSEL, ClockSource);
}
/**
  * @brief  Get the DFSDM2 or TIM2_OC1 as clock source for the bitstream clock.
  * @rmtoll SYSCFG_MCHDLYCR BSCKSEL       LL_SYSCFG_DFSDM_GetBitstreamClockSourceSelection
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_BITSTREAM_CLOCK_DFSDM2
  *         @arg @ref LL_SYSCFG_BITSTREAM_CLOCK_TIM2OC1
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM_GetBitstreamClockSourceSelection(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_BSCKSEL));
}
/**
  * @brief  Enables the DFSDM1 or DFSDM2 Delay clock
  * @rmtoll SYSCFG_MCHDLYCR MCHDLYEN      LL_SYSCFG_DFSDM_EnableDelayClock
  * @param MCHDLY This paramater can be one of the following values
  *         @arg @ref LL_SYSCFG_DFSDM1_MCHDLYEN
  *         @arg @ref LL_SYSCFG_DFSDM2_MCHDLYEN
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM_EnableDelayClock(uint32_t MCHDLY)
{
  SET_BIT(SYSCFG->MCHDLYCR, MCHDLY);
}

/**
  * @brief  Disables the DFSDM1 or the DFSDM2 Delay clock
  * @rmtoll SYSCFG_MCHDLYCR MCHDLY1EN      LL_SYSCFG_DFSDM1_DisableDelayClock
  * @param MCHDLY This paramater can be one of the following values
  *         @arg @ref LL_SYSCFG_DFSDM1_MCHDLYEN
  *         @arg @ref LL_SYSCFG_DFSDM2_MCHDLYEN
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM_DisableDelayClock(uint32_t MCHDLY)
{
  CLEAR_BIT(SYSCFG->MCHDLYCR, MCHDLY);
}

/**
  * @brief  Select the source for DFSDM1 or DFSDM2 DatIn0 
  * @rmtoll SYSCFG_MCHDLYCR DFSDMD0SEL        LL_SYSCFG_DFSDM_SetDataIn0Source
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn0_PAD
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn0_DM
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn0_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn0_DM
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM_SetDataIn0Source(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, (Source >> 16), (Source & 0x0000FFFF));
}
/**
  * @brief  Get the source for DFSDM1 or DFSDM2 DatIn0.
  * @rmtoll SYSCFG_MCHDLYCR DFSDMD0SEL       LL_SYSCFG_DFSDM_GetDataIn0Source
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn0
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn0
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn0_PAD
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn0_DM
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn0_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn0_DM
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM_GetDataIn0Source(uint32_t Source)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, Source));
}
/**
  * @brief  Select the source for DFSDM1 or DFSDM2 DatIn2 
  * @rmtoll SYSCFG_MCHDLYCR DFSDMD2SEL        LL_SYSCFG_DFSDM_SetDataIn2Source
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn2_PAD
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn2_DM
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn2_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn2_DM
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM_SetDataIn2Source(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, (Source >> 16), (Source & 0x0000FFFF));
}
/**
  * @brief  Get the source for DFSDM1 or DFSDM2 DatIn2.
  * @rmtoll SYSCFG_MCHDLYCR DFSDMD2SEL       LL_SYSCFG_DFSDM_GetDataIn2Source
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn2
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn2
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn2_PAD
  *         @arg @ref LL_SYSCFG_DFSDM1_DataIn2_DM
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn2_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn2_DM
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM_GetDataIn2Source(uint32_t Source)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, Source));
}

/**
  * @brief  Select the distribution of the bitsream lock gated by TIM4 OC2 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM1CK02SEL        LL_SYSCFG_DFSDM1_SetTIM4OC2BitStreamDistribution
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_TIM4OC2_CLKIN0
  *         @arg @ref LL_SYSCFG_DFSDM1_TIM4OC2_CLKIN2
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM1_SetTIM4OC2BitStreamDistribution(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM1CK02SEL, Source);
}
/**
  * @brief  Get the distribution of the bitsream lock gated by TIM4 OC2 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM1D2SEL       LL_SYSCFG_DFSDM1_GetTIM4OC2BitStreamDistribution
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_TIM4OC2_CLKIN0
  *         @arg @ref LL_SYSCFG_DFSDM1_TIM4OC2_CLKIN2
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM1_GetTIM4OC2BitStreamDistribution(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM1CK02SEL));
}

/**
  * @brief  Select the distribution of the bitsream lock gated by TIM4 OC1 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM1CK13SEL        LL_SYSCFG_DFSDM1_SetTIM4OC1BitStreamDistribution
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_TIM4OC1_CLKIN1
  *         @arg @ref LL_SYSCFG_DFSDM1_TIM4OC1_CLKIN3
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM1_SetTIM4OC1BitStreamDistribution(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM1CK13SEL, Source);
}
/**
  * @brief  Get the distribution of the bitsream lock gated by TIM4 OC1 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM1D2SEL       LL_SYSCFG_DFSDM1_GetTIM4OC1BitStreamDistribution
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_TIM4OC1_CLKIN1
  *         @arg @ref LL_SYSCFG_DFSDM1_TIM4OC1_CLKIN3
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM1_GetTIM4OC1BitStreamDistribution(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM1CK13SEL));
}

/**
  * @brief  Select the DFSDM1 Clock In 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM1CFG        LL_SYSCFG_DFSDM1_SetClockInSourceSelection
  * @param  ClockSource This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_CKIN_PAD
  *         @arg @ref LL_SYSCFG_DFSDM1_CKIN_DM
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM1_SetClockInSourceSelection(uint32_t ClockSource)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM1CFG, ClockSource);
}
/**
  * @brief  GET the DFSDM1 Clock In
  * @rmtoll SYSCFG_MCHDLYCR DFSDM1CFG       LL_SYSCFG_DFSDM1_GetClockInSourceSelection
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_CKIN_PAD
  *         @arg @ref LL_SYSCFG_DFSDM1_CKIN_DM
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM1_GetClockInSourceSelection(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM1CFG));
}

/**
  * @brief  Select the DFSDM1 Clock Out 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM1CKOSEL        LL_SYSCFG_DFSDM1_SetClockOutSourceSelection
  * @param  ClockSource This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_CKOUT
  *         @arg @ref LL_SYSCFG_DFSDM1_CKOUT_M27
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM1_SetClockOutSourceSelection(uint32_t ClockSource)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM1CKOSEL, ClockSource);
}
/**
  * @brief  GET the DFSDM1 Clock Out
  * @rmtoll SYSCFG_MCHDLYCR DFSDM1CKOSEL       LL_SYSCFG_DFSDM1_GetClockOutSourceSelection
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM1_CKOUT
  *         @arg @ref LL_SYSCFG_DFSDM1_CKOUT_M27
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM1_GetClockOutSourceSelection(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM1CKOSEL));
}

/**
  * @brief  Enables the DFSDM2 Delay clock
  * @rmtoll SYSCFG_MCHDLYCR MCHDLY2EN      LL_SYSCFG_DFSDM2_EnableDelayClock
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_EnableDelayClock(void)
{
  SET_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_MCHDLY2EN);
}

/**
  * @brief  Disables the DFSDM2 Delay clock
  * @rmtoll SYSCFG_MCHDLYCR MCHDLY2EN      LL_SYSCFG_DFSDM2_DisableDelayClock
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_DisableDelayClock(void)
{
  CLEAR_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_MCHDLY2EN);
}
/**
  * @brief  Select the source for DFSDM2 DatIn0 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2D0SEL        LL_SYSCFG_DFSDM2_SetDataIn0Source
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn0_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn0_DM
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetDataIn0Source(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2D0SEL, Source);
}
/**
  * @brief  Get the source for DFSDM2 DatIn0.
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2D0SEL       LL_SYSCFG_DFSDM2_GetDataIn0Source
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn0_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn0_DM
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetDataIn0Source(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2D0SEL));
}

/**
  * @brief  Select the source for DFSDM2 DatIn2 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2D2SEL        LL_SYSCFG_DFSDM2_SetDataIn2Source
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn2_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn2_DM
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetDataIn2Source(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2D2SEL, Source);
}
/**
  * @brief  Get the source for DFSDM2 DatIn2.
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2D2SEL       LL_SYSCFG_DFSDM2_GetDataIn2Source
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn2_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn2_DM
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetDataIn2Source(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2D2SEL));
}

/**
  * @brief  Select the source for DFSDM2 DatIn4 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2D4SEL        LL_SYSCFG_DFSDM2_SetDataIn4Source
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn4_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn4_DM
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetDataIn4Source(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2D4SEL, Source);
}
/**
  * @brief  Get the source for DFSDM2 DatIn4.
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2D4SEL       LL_SYSCFG_DFSDM2_GetDataIn4Source
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn4_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn4_DM
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetDataIn4Source(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2D4SEL));
}

/**
  * @brief  Select the source for DFSDM2 DatIn6 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2D6SEL        LL_SYSCFG_DFSDM2_SetDataIn6Source
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn6_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn6_DM
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetDataIn6Source(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2D6SEL, Source);
}
/**
  * @brief  Get the source for DFSDM2 DatIn6.
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2D6SEL       LL_SYSCFG_DFSDM2_GetDataIn6Source
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn6_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_DataIn6_DM
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetDataIn6Source(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2D6SEL));
}

/**
  * @brief  Select the distribution of the bitsream lock gated by TIM3 OC4 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CK04SEL        LL_SYSCFG_DFSDM2_SetTIM3OC4BitStreamDistribution
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC4_CLKIN0
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC4_CLKIN4
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetTIM3OC4BitStreamDistribution(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CK04SEL, Source);
}
/**
  * @brief  Get the distribution of the bitsream lock gated by TIM3 OC4 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CK04SEL       LL_SYSCFG_DFSDM2_GetTIM3OC4BitStreamDistribution
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC4_CLKIN0
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC4_CLKIN4
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetTIM3OC4BitStreamDistribution(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CK04SEL));
}

/**
  * @brief  Select the distribution of the bitsream lock gated by TIM3 OC3 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CK15SEL        LL_SYSCFG_DFSDM2_SetTIM3OC3BitStreamDistribution
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC3_CLKIN1
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC3_CLKIN5
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetTIM3OC3BitStreamDistribution(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CK15SEL, Source);
}
/**
  * @brief  Get the distribution of the bitsream lock gated by TIM3 OC4 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CK04SEL       LL_SYSCFG_DFSDM2_GetTIM3OC3BitStreamDistribution
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC3_CLKIN1
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC3_CLKIN5
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetTIM3OC3BitStreamDistribution(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CK15SEL));
}

/**
  * @brief  Select the distribution of the bitsream lock gated by TIM3 OC2 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CK26SEL        LL_SYSCFG_DFSDM2_SetTIM3OC2BitStreamDistribution
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC2_CLKIN2
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC2_CLKIN6
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetTIM3OC2BitStreamDistribution(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CK26SEL, Source);
}
/**
  * @brief  Get the distribution of the bitsream lock gated by TIM3 OC2 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CK04SEL       LL_SYSCFG_DFSDM2_GetTIM3OC2BitStreamDistribution
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC2_CLKIN2
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC2_CLKIN6
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetTIM3OC2BitStreamDistribution(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CK26SEL));
}

/**
  * @brief  Select the distribution of the bitsream lock gated by TIM3 OC1 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CK37SEL        LL_SYSCFG_DFSDM2_SetTIM3OC1BitStreamDistribution
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC1_CLKIN3
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC1_CLKIN7
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetTIM3OC1BitStreamDistribution(uint32_t Source)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CK37SEL, Source);
}
/**
  * @brief  Get the distribution of the bitsream lock gated by TIM3 OC1 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CK37SEL       LL_SYSCFG_DFSDM2_GetTIM3OC1BitStreamDistribution
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC1_CLKIN3
  *         @arg @ref LL_SYSCFG_DFSDM2_TIM3OC1_CLKIN7
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetTIM3OC1BitStreamDistribution(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CK37SEL));
}

/**
  * @brief  Select the DFSDM2 Clock In 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CFG        LL_SYSCFG_DFSDM2_SetClockInSourceSelection
  * @param  ClockSource This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_CKIN_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_CKIN_DM
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetClockInSourceSelection(uint32_t ClockSource)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CFG, ClockSource);
}
/**
  * @brief  GET the DFSDM2 Clock In
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CFG       LL_SYSCFG_DFSDM2_GetClockInSourceSelection
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_CKIN_PAD
  *         @arg @ref LL_SYSCFG_DFSDM2_CKIN_DM
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetClockInSourceSelection(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CFG));
}

/**
  * @brief  Select the DFSDM2 Clock Out 
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CKOSEL        LL_SYSCFG_DFSDM2_SetClockOutSourceSelection
  * @param  ClockSource This parameter can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_CKOUT
  *         @arg @ref LL_SYSCFG_DFSDM2_CKOUT_M27
  * @retval None
  */
__STATIC_INLINE void LL_SYSCFG_DFSDM2_SetClockOutSourceSelection(uint32_t ClockSource)
{
  MODIFY_REG(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CKOSEL, ClockSource);
}
/**
  * @brief  GET the DFSDM2 Clock Out
  * @rmtoll SYSCFG_MCHDLYCR DFSDM2CKOSEL       LL_SYSCFG_DFSDM2_GetClockOutSourceSelection
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SYSCFG_DFSDM2_CKOUT
  *         @arg @ref LL_SYSCFG_DFSDM2_CKOUT_M27
  * @retval None
  */
__STATIC_INLINE uint32_t LL_SYSCFG_DFSDM2_GetClockOutSourceSelection(void)
{
  return (uint32_t)(READ_BIT(SYSCFG->MCHDLYCR, SYSCFG_MCHDLYCR_DFSDM2CKOSEL));
}

#endif /* SYSCFG_MCHDLYCR_BSCKSEL */
/**
  * @}
  */


/** @defgroup SYSTEM_LL_EF_DBGMCU DBGMCU
  * @{
  */

/**
  * @brief  Return the device identifier
  * @note For STM32F405/407xx and STM32F415/417xx devices, the device ID is 0x413
  * @note For STM32F42xxx and STM32F43xxx devices, the device ID is 0x419
  * @note For STM32F401xx devices, the device ID is 0x423
  * @note For STM32F401xx devices, the device ID is 0x433
  * @note For STM32F411xx devices, the device ID is 0x431
  * @note For STM32F410xx devices, the device ID is 0x458
  * @note For STM32F412xx devices, the device ID is 0x441
  * @note For STM32F413xx and STM32423xx devices, the device ID is 0x463
  * @note For STM32F446xx devices, the device ID is 0x421
  * @note For STM32F469xx and STM32F479xx devices, the device ID is 0x434
  * @rmtoll DBGMCU_IDCODE DEV_ID        LL_DBGMCU_GetDeviceID
  * @retval Values between Min_Data=0x00 and Max_Data=0xFFF
  */
__STATIC_INLINE uint32_t LL_DBGMCU_GetDeviceID(void)
{
  return (uint32_t)(READ_BIT(DBGMCU->IDCODE, DBGMCU_IDCODE_DEV_ID));
}

/**
  * @brief  Return the device revision identifier
  * @note This field indicates the revision of the device.
          For example, it is read as RevA -> 0x1000, Cat 2 revZ -> 0x1001, rev1 -> 0x1003, rev2 ->0x1007, revY -> 0x100F for STM32F405/407xx and STM32F415/417xx devices
          For example, it is read as RevA -> 0x1000, Cat 2 revY -> 0x1003, rev1 -> 0x1007, rev3 ->0x2001 for STM32F42xxx and STM32F43xxx devices
          For example, it is read as RevZ -> 0x1000, Cat 2 revA -> 0x1001 for STM32F401xB/C devices
          For example, it is read as RevA -> 0x1000, Cat 2 revZ -> 0x1001 for STM32F401xD/E devices
          For example, it is read as RevA -> 0x1000 for STM32F411xx,STM32F413/423xx,STM32F469/423xx, STM32F446xx and STM32F410xx devices
          For example, it is read as RevZ -> 0x1001, Cat 2 revB -> 0x2000, revC -> 0x3000 for STM32F412xx devices
  * @rmtoll DBGMCU_IDCODE REV_ID        LL_DBGMCU_GetRevisionID
  * @retval Values between Min_Data=0x00 and Max_Data=0xFFFF
  */
__STATIC_INLINE uint32_t LL_DBGMCU_GetRevisionID(void)
{
  return (uint32_t)(READ_BIT(DBGMCU->IDCODE, DBGMCU_IDCODE_REV_ID) >> DBGMCU_IDCODE_REV_ID_Pos);
}

/**
  * @brief  Enable the Debug Module during SLEEP mode
  * @rmtoll DBGMCU_CR    DBG_SLEEP     LL_DBGMCU_EnableDBGSleepMode
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_EnableDBGSleepMode(void)
{
  SET_BIT(DBGMCU->CR, DBGMCU_CR_DBG_SLEEP);
}

/**
  * @brief  Disable the Debug Module during SLEEP mode
  * @rmtoll DBGMCU_CR    DBG_SLEEP     LL_DBGMCU_DisableDBGSleepMode
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_DisableDBGSleepMode(void)
{
  CLEAR_BIT(DBGMCU->CR, DBGMCU_CR_DBG_SLEEP);
}

/**
  * @brief  Enable the Debug Module during STOP mode
  * @rmtoll DBGMCU_CR    DBG_STOP      LL_DBGMCU_EnableDBGStopMode
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_EnableDBGStopMode(void)
{
  SET_BIT(DBGMCU->CR, DBGMCU_CR_DBG_STOP);
}

/**
  * @brief  Disable the Debug Module during STOP mode
  * @rmtoll DBGMCU_CR    DBG_STOP      LL_DBGMCU_DisableDBGStopMode
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_DisableDBGStopMode(void)
{
  CLEAR_BIT(DBGMCU->CR, DBGMCU_CR_DBG_STOP);
}

/**
  * @brief  Enable the Debug Module during STANDBY mode
  * @rmtoll DBGMCU_CR    DBG_STANDBY   LL_DBGMCU_EnableDBGStandbyMode
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_EnableDBGStandbyMode(void)
{
  SET_BIT(DBGMCU->CR, DBGMCU_CR_DBG_STANDBY);
}

/**
  * @brief  Disable the Debug Module during STANDBY mode
  * @rmtoll DBGMCU_CR    DBG_STANDBY   LL_DBGMCU_DisableDBGStandbyMode
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_DisableDBGStandbyMode(void)
{
  CLEAR_BIT(DBGMCU->CR, DBGMCU_CR_DBG_STANDBY);
}

/**
  * @brief  Set Trace pin assignment control
  * @rmtoll DBGMCU_CR    TRACE_IOEN    LL_DBGMCU_SetTracePinAssignment\n
  *         DBGMCU_CR    TRACE_MODE    LL_DBGMCU_SetTracePinAssignment
  * @param  PinAssignment This parameter can be one of the following values:
  *         @arg @ref LL_DBGMCU_TRACE_NONE
  *         @arg @ref LL_DBGMCU_TRACE_ASYNCH
  *         @arg @ref LL_DBGMCU_TRACE_SYNCH_SIZE1
  *         @arg @ref LL_DBGMCU_TRACE_SYNCH_SIZE2
  *         @arg @ref LL_DBGMCU_TRACE_SYNCH_SIZE4
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_SetTracePinAssignment(uint32_t PinAssignment)
{
  MODIFY_REG(DBGMCU->CR, DBGMCU_CR_TRACE_IOEN | DBGMCU_CR_TRACE_MODE, PinAssignment);
}

/**
  * @brief  Get Trace pin assignment control
  * @rmtoll DBGMCU_CR    TRACE_IOEN    LL_DBGMCU_GetTracePinAssignment\n
  *         DBGMCU_CR    TRACE_MODE    LL_DBGMCU_GetTracePinAssignment
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_DBGMCU_TRACE_NONE
  *         @arg @ref LL_DBGMCU_TRACE_ASYNCH
  *         @arg @ref LL_DBGMCU_TRACE_SYNCH_SIZE1
  *         @arg @ref LL_DBGMCU_TRACE_SYNCH_SIZE2
  *         @arg @ref LL_DBGMCU_TRACE_SYNCH_SIZE4
  */
__STATIC_INLINE uint32_t LL_DBGMCU_GetTracePinAssignment(void)
{
  return (uint32_t)(READ_BIT(DBGMCU->CR, DBGMCU_CR_TRACE_IOEN | DBGMCU_CR_TRACE_MODE));
}

/**
  * @brief  Freeze APB1 peripherals (group1 peripherals)
  * @rmtoll DBGMCU_APB1_FZ      DBG_TIM2_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM3_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM4_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM5_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM6_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM7_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM12_STOP          LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM13_STOP          LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM14_STOP          LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_LPTIM_STOP          LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_RTC_STOP            LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_WWDG_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_IWDG_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_I2C1_SMBUS_TIMEOUT  LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_I2C2_SMBUS_TIMEOUT  LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_I2C3_SMBUS_TIMEOUT  LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_I2C4_SMBUS_TIMEOUT  LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_CAN1_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_CAN2_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_CAN3_STOP           LL_DBGMCU_APB1_GRP1_FreezePeriph
  * @param  Periphs This parameter can be a combination of the following values:
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM2_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM3_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM4_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM5_STOP 
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM6_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM7_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM12_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM13_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM14_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_LPTIM_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_RTC_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_WWDG_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_IWDG_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_I2C1_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_I2C2_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_I2C3_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_I2C4_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_CAN1_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_CAN2_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_CAN3_STOP (*)
  *         
  *         (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_APB1_GRP1_FreezePeriph(uint32_t Periphs)
{
  SET_BIT(DBGMCU->APB1FZ, Periphs);
}

/**
  * @brief  Unfreeze APB1 peripherals (group1 peripherals)
  * @rmtoll DBGMCU_APB1_FZ      DBG_TIM2_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM3_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM4_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM5_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM6_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM7_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM12_STOP          LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM13_STOP          LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_TIM14_STOP          LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_LPTIM_STOP         LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_RTC_STOP            LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_WWDG_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_IWDG_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_I2C1_SMBUS_TIMEOUT  LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_I2C2_SMBUS_TIMEOUT  LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_I2C3_SMBUS_TIMEOUT  LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_I2C4_SMBUS_TIMEOUT  LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_CAN1_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_CAN2_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB1_FZ      DBG_CAN3_STOP           LL_DBGMCU_APB1_GRP1_UnFreezePeriph
  * @param  Periphs This parameter can be a combination of the following values:
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM2_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM3_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM4_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM5_STOP 
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM6_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM7_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM12_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM13_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_TIM14_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_LPTIM_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_RTC_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_WWDG_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_IWDG_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_I2C1_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_I2C2_STOP
  *         @arg @ref LL_DBGMCU_APB1_GRP1_I2C3_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_I2C4_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_CAN1_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_CAN2_STOP (*)
  *         @arg @ref LL_DBGMCU_APB1_GRP1_CAN3_STOP (*)
  *         
  *         (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_APB1_GRP1_UnFreezePeriph(uint32_t Periphs)
{
  CLEAR_BIT(DBGMCU->APB1FZ, Periphs);
}

/**
  * @brief  Freeze APB2 peripherals
  * @rmtoll DBGMCU_APB2_FZ      DBG_TIM1_STOP    LL_DBGMCU_APB2_GRP1_FreezePeriph\n
  *         DBGMCU_APB2_FZ      DBG_TIM8_STOP    LL_DBGMCU_APB2_GRP1_FreezePeriph\n
  *         DBGMCU_APB2_FZ      DBG_TIM9_STOP    LL_DBGMCU_APB2_GRP1_FreezePeriph\n
  *         DBGMCU_APB2_FZ      DBG_TIM10_STOP   LL_DBGMCU_APB2_GRP1_FreezePeriph\n
  *         DBGMCU_APB2_FZ      DBG_TIM11_STOP   LL_DBGMCU_APB2_GRP1_FreezePeriph
  * @param  Periphs This parameter can be a combination of the following values:
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM1_STOP
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM8_STOP (*)
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM9_STOP (*)
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM10_STOP (*)
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM11_STOP (*)
  *
  *         (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_APB2_GRP1_FreezePeriph(uint32_t Periphs)
{
  SET_BIT(DBGMCU->APB2FZ, Periphs);
}

/**
  * @brief  Unfreeze APB2 peripherals
  * @rmtoll DBGMCU_APB2_FZ      DBG_TIM1_STOP    LL_DBGMCU_APB2_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB2_FZ      DBG_TIM8_STOP    LL_DBGMCU_APB2_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB2_FZ      DBG_TIM9_STOP    LL_DBGMCU_APB2_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB2_FZ      DBG_TIM10_STOP   LL_DBGMCU_APB2_GRP1_UnFreezePeriph\n
  *         DBGMCU_APB2_FZ      DBG_TIM11_STOP   LL_DBGMCU_APB2_GRP1_UnFreezePeriph
  * @param  Periphs This parameter can be a combination of the following values:
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM1_STOP
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM8_STOP (*)
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM9_STOP (*)
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM10_STOP (*)
  *         @arg @ref LL_DBGMCU_APB2_GRP1_TIM11_STOP (*)
  *
  *         (*) value not defined in all devices.
  * @retval None
  */
__STATIC_INLINE void LL_DBGMCU_APB2_GRP1_UnFreezePeriph(uint32_t Periphs)
{
  CLEAR_BIT(DBGMCU->APB2FZ, Periphs);
}
/**
  * @}
  */

/** @defgroup SYSTEM_LL_EF_FLASH FLASH
  * @{
  */

/**
  * @brief  Set FLASH Latency
  * @rmtoll FLASH_ACR    LATENCY       LL_FLASH_SetLatency
  * @param  Latency This parameter can be one of the following values:
  *         @arg @ref LL_FLASH_LATENCY_0
  *         @arg @ref LL_FLASH_LATENCY_1
  *         @arg @ref LL_FLASH_LATENCY_2
  *         @arg @ref LL_FLASH_LATENCY_3
  *         @arg @ref LL_FLASH_LATENCY_4
  *         @arg @ref LL_FLASH_LATENCY_5
  *         @arg @ref LL_FLASH_LATENCY_6
  *         @arg @ref LL_FLASH_LATENCY_7
  *         @arg @ref LL_FLASH_LATENCY_8
  *         @arg @ref LL_FLASH_LATENCY_9
  *         @arg @ref LL_FLASH_LATENCY_10
  *         @arg @ref LL_FLASH_LATENCY_11
  *         @arg @ref LL_FLASH_LATENCY_12
  *         @arg @ref LL_FLASH_LATENCY_13
  *         @arg @ref LL_FLASH_LATENCY_14
  *         @arg @ref LL_FLASH_LATENCY_15
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_SetLatency(uint32_t Latency)
{
  MODIFY_REG(FLASH->ACR, FLASH_ACR_LATENCY, Latency);
}

/**
  * @brief  Get FLASH Latency
  * @rmtoll FLASH_ACR    LATENCY       LL_FLASH_GetLatency
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_FLASH_LATENCY_0
  *         @arg @ref LL_FLASH_LATENCY_1
  *         @arg @ref LL_FLASH_LATENCY_2
  *         @arg @ref LL_FLASH_LATENCY_3
  *         @arg @ref LL_FLASH_LATENCY_4
  *         @arg @ref LL_FLASH_LATENCY_5
  *         @arg @ref LL_FLASH_LATENCY_6
  *         @arg @ref LL_FLASH_LATENCY_7
  *         @arg @ref LL_FLASH_LATENCY_8
  *         @arg @ref LL_FLASH_LATENCY_9
  *         @arg @ref LL_FLASH_LATENCY_10
  *         @arg @ref LL_FLASH_LATENCY_11
  *         @arg @ref LL_FLASH_LATENCY_12
  *         @arg @ref LL_FLASH_LATENCY_13
  *         @arg @ref LL_FLASH_LATENCY_14
  *         @arg @ref LL_FLASH_LATENCY_15
  */
__STATIC_INLINE uint32_t LL_FLASH_GetLatency(void)
{
  return (uint32_t)(READ_BIT(FLASH->ACR, FLASH_ACR_LATENCY));
}

/**
  * @brief  Enable Prefetch
  * @rmtoll FLASH_ACR    PRFTEN        LL_FLASH_EnablePrefetch
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_EnablePrefetch(void)
{
  SET_BIT(FLASH->ACR, FLASH_ACR_PRFTEN);
}

/**
  * @brief  Disable Prefetch
  * @rmtoll FLASH_ACR    PRFTEN        LL_FLASH_DisablePrefetch
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_DisablePrefetch(void)
{
  CLEAR_BIT(FLASH->ACR, FLASH_ACR_PRFTEN);
}

/**
  * @brief  Check if Prefetch buffer is enabled
  * @rmtoll FLASH_ACR    PRFTEN        LL_FLASH_IsPrefetchEnabled
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FLASH_IsPrefetchEnabled(void)
{
  return (READ_BIT(FLASH->ACR, FLASH_ACR_PRFTEN) == (FLASH_ACR_PRFTEN));
}

/**
  * @brief  Enable Instruction cache
  * @rmtoll FLASH_ACR    ICEN          LL_FLASH_EnableInstCache
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_EnableInstCache(void)
{
  SET_BIT(FLASH->ACR, FLASH_ACR_ICEN);
}

/**
  * @brief  Disable Instruction cache
  * @rmtoll FLASH_ACR    ICEN          LL_FLASH_DisableInstCache
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_DisableInstCache(void)
{
  CLEAR_BIT(FLASH->ACR, FLASH_ACR_ICEN);
}

/**
  * @brief  Enable Data cache
  * @rmtoll FLASH_ACR    DCEN          LL_FLASH_EnableDataCache
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_EnableDataCache(void)
{
  SET_BIT(FLASH->ACR, FLASH_ACR_DCEN);
}

/**
  * @brief  Disable Data cache
  * @rmtoll FLASH_ACR    DCEN          LL_FLASH_DisableDataCache
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_DisableDataCache(void)
{
  CLEAR_BIT(FLASH->ACR, FLASH_ACR_DCEN);
}

/**
  * @brief  Enable Instruction cache reset
  * @note  bit can be written only when the instruction cache is disabled
  * @rmtoll FLASH_ACR    ICRST         LL_FLASH_EnableInstCacheReset
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_EnableInstCacheReset(void)
{
  SET_BIT(FLASH->ACR, FLASH_ACR_ICRST);
}

/**
  * @brief  Disable Instruction cache reset
  * @rmtoll FLASH_ACR    ICRST         LL_FLASH_DisableInstCacheReset
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_DisableInstCacheReset(void)
{
  CLEAR_BIT(FLASH->ACR, FLASH_ACR_ICRST);
}

/**
  * @brief  Enable Data cache reset
  * @note bit can be written only when the data cache is disabled
  * @rmtoll FLASH_ACR    DCRST         LL_FLASH_EnableDataCacheReset
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_EnableDataCacheReset(void)
{
  SET_BIT(FLASH->ACR, FLASH_ACR_DCRST);
}

/**
  * @brief  Disable Data cache reset
  * @rmtoll FLASH_ACR    DCRST         LL_FLASH_DisableDataCacheReset
  * @retval None
  */
__STATIC_INLINE void LL_FLASH_DisableDataCacheReset(void)
{
  CLEAR_BIT(FLASH->ACR, FLASH_ACR_DCRST);
}


/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#endif /* defined (FLASH) || defined (SYSCFG) || defined (DBGMCU) */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_LL_SYSTEM_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
