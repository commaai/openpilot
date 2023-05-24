/**
  ******************************************************************************
  * @file    stm32f4xx_hal.h
  * @author  MCD Application Team
  * @brief   This file contains all the functions prototypes for the HAL 
  *          module driver.
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
#ifndef __STM32F4xx_HAL_H
#define __STM32F4xx_HAL_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_conf.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup HAL
  * @{
  */ 

/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/

/** @defgroup HAL_Exported_Constants HAL Exported Constants
  * @{
  */

/** @defgroup HAL_TICK_FREQ Tick Frequency
  * @{
  */
typedef enum
{
  HAL_TICK_FREQ_10HZ         = 100U,
  HAL_TICK_FREQ_100HZ        = 10U,
  HAL_TICK_FREQ_1KHZ         = 1U,
  HAL_TICK_FREQ_DEFAULT      = HAL_TICK_FREQ_1KHZ
} HAL_TickFreqTypeDef;
/**
  * @}
  */

/**
  * @}
  */
   
/* Exported macro ------------------------------------------------------------*/
/** @defgroup HAL_Exported_Macros HAL Exported Macros
  * @{
  */

/** @brief  Freeze/Unfreeze Peripherals in Debug mode 
  */
#define __HAL_DBGMCU_FREEZE_TIM2()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM2_STOP))
#define __HAL_DBGMCU_FREEZE_TIM3()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM3_STOP))
#define __HAL_DBGMCU_FREEZE_TIM4()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM4_STOP))
#define __HAL_DBGMCU_FREEZE_TIM5()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM5_STOP))
#define __HAL_DBGMCU_FREEZE_TIM6()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM6_STOP))
#define __HAL_DBGMCU_FREEZE_TIM7()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM7_STOP))
#define __HAL_DBGMCU_FREEZE_TIM12()          (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM12_STOP))
#define __HAL_DBGMCU_FREEZE_TIM13()          (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM13_STOP))
#define __HAL_DBGMCU_FREEZE_TIM14()          (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM14_STOP))
#define __HAL_DBGMCU_FREEZE_RTC()            (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_RTC_STOP))
#define __HAL_DBGMCU_FREEZE_WWDG()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_WWDG_STOP))
#define __HAL_DBGMCU_FREEZE_IWDG()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_IWDG_STOP))
#define __HAL_DBGMCU_FREEZE_I2C1_TIMEOUT()   (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_I2C1_SMBUS_TIMEOUT))
#define __HAL_DBGMCU_FREEZE_I2C2_TIMEOUT()   (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_I2C2_SMBUS_TIMEOUT))
#define __HAL_DBGMCU_FREEZE_I2C3_TIMEOUT()   (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_I2C3_SMBUS_TIMEOUT))
#define __HAL_DBGMCU_FREEZE_CAN1()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_CAN1_STOP))
#define __HAL_DBGMCU_FREEZE_CAN2()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_CAN2_STOP))
#define __HAL_DBGMCU_FREEZE_TIM1()           (DBGMCU->APB2FZ |= (DBGMCU_APB2_FZ_DBG_TIM1_STOP))
#define __HAL_DBGMCU_FREEZE_TIM8()           (DBGMCU->APB2FZ |= (DBGMCU_APB2_FZ_DBG_TIM8_STOP))
#define __HAL_DBGMCU_FREEZE_TIM9()           (DBGMCU->APB2FZ |= (DBGMCU_APB2_FZ_DBG_TIM9_STOP))
#define __HAL_DBGMCU_FREEZE_TIM10()          (DBGMCU->APB2FZ |= (DBGMCU_APB2_FZ_DBG_TIM10_STOP))
#define __HAL_DBGMCU_FREEZE_TIM11()          (DBGMCU->APB2FZ |= (DBGMCU_APB2_FZ_DBG_TIM11_STOP))

#define __HAL_DBGMCU_UNFREEZE_TIM2()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_TIM2_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM3()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_TIM3_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM4()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_TIM4_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM5()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_TIM5_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM6()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_TIM6_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM7()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_TIM7_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM12()          (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_TIM12_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM13()          (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_TIM13_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM14()          (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_TIM14_STOP))
#define __HAL_DBGMCU_UNFREEZE_RTC()            (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_RTC_STOP))
#define __HAL_DBGMCU_UNFREEZE_WWDG()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_WWDG_STOP))
#define __HAL_DBGMCU_UNFREEZE_IWDG()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_IWDG_STOP))
#define __HAL_DBGMCU_UNFREEZE_I2C1_TIMEOUT()   (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_I2C1_SMBUS_TIMEOUT))
#define __HAL_DBGMCU_UNFREEZE_I2C2_TIMEOUT()   (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_I2C2_SMBUS_TIMEOUT))
#define __HAL_DBGMCU_UNFREEZE_I2C3_TIMEOUT()   (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_I2C3_SMBUS_TIMEOUT))
#define __HAL_DBGMCU_UNFREEZE_CAN1()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_CAN1_STOP))
#define __HAL_DBGMCU_UNFREEZE_CAN2()           (DBGMCU->APB1FZ &= ~(DBGMCU_APB1_FZ_DBG_CAN2_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM1()           (DBGMCU->APB2FZ &= ~(DBGMCU_APB2_FZ_DBG_TIM1_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM8()           (DBGMCU->APB2FZ &= ~(DBGMCU_APB2_FZ_DBG_TIM8_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM9()           (DBGMCU->APB2FZ &= ~(DBGMCU_APB2_FZ_DBG_TIM9_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM10()          (DBGMCU->APB2FZ &= ~(DBGMCU_APB2_FZ_DBG_TIM10_STOP))
#define __HAL_DBGMCU_UNFREEZE_TIM11()          (DBGMCU->APB2FZ &= ~(DBGMCU_APB2_FZ_DBG_TIM11_STOP))

/** @brief  Main Flash memory mapped at 0x00000000
  */
#define __HAL_SYSCFG_REMAPMEMORY_FLASH()             (SYSCFG->MEMRMP &= ~(SYSCFG_MEMRMP_MEM_MODE))

/** @brief  System Flash memory mapped at 0x00000000
  */
#define __HAL_SYSCFG_REMAPMEMORY_SYSTEMFLASH()       do {SYSCFG->MEMRMP &= ~(SYSCFG_MEMRMP_MEM_MODE);\
                                                         SYSCFG->MEMRMP |= SYSCFG_MEMRMP_MEM_MODE_0;\
                                                        }while(0);

/** @brief  Embedded SRAM mapped at 0x00000000
  */
#define __HAL_SYSCFG_REMAPMEMORY_SRAM()       do {SYSCFG->MEMRMP &= ~(SYSCFG_MEMRMP_MEM_MODE);\
                                                  SYSCFG->MEMRMP |= (SYSCFG_MEMRMP_MEM_MODE_0 | SYSCFG_MEMRMP_MEM_MODE_1);\
                                                 }while(0);

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx)|| defined(STM32F417xx)
/** @brief  FSMC Bank1 (NOR/PSRAM 1 and 2) mapped at 0x00000000
  */
#define __HAL_SYSCFG_REMAPMEMORY_FSMC()       do {SYSCFG->MEMRMP &= ~(SYSCFG_MEMRMP_MEM_MODE);\
                                                  SYSCFG->MEMRMP |= (SYSCFG_MEMRMP_MEM_MODE_1);\
                                                 }while(0);
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
/** @brief  FMC Bank1 (NOR/PSRAM 1 and 2) mapped at 0x00000000
  */
#define __HAL_SYSCFG_REMAPMEMORY_FMC()       do {SYSCFG->MEMRMP &= ~(SYSCFG_MEMRMP_MEM_MODE);\
                                                 SYSCFG->MEMRMP |= (SYSCFG_MEMRMP_MEM_MODE_1);\
                                                }while(0);

/** @brief  FMC/SDRAM Bank 1 and 2 mapped at 0x00000000
  */
#define __HAL_SYSCFG_REMAPMEMORY_FMC_SDRAM()       do {SYSCFG->MEMRMP &= ~(SYSCFG_MEMRMP_MEM_MODE);\
                                                       SYSCFG->MEMRMP |= (SYSCFG_MEMRMP_MEM_MODE_2);\
                                                      }while(0);
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */ 

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F413xx) || defined(STM32F423xx)
/** @defgroup Cortex_Lockup_Enable Cortex Lockup Enable
  * @{
  */
/** @brief  SYSCFG Break Lockup lock
  *         Enables and locks the connection of Cortex-M4 LOCKUP (Hardfault) output to TIM1/8 input
  * @note   The selected configuration is locked and can be unlocked by system reset
  */
#define __HAL_SYSCFG_BREAK_PVD_LOCK()      do {SYSCFG->CFGR2 &= ~(SYSCFG_CFGR2_PVD_LOCK); \
                                               SYSCFG->CFGR2 |= SYSCFG_CFGR2_PVD_LOCK;    \
                                              }while(0)
/**
 * @}
 */

/** @defgroup PVD_Lock_Enable PVD Lock
  * @{
  */
/** @brief  SYSCFG Break PVD lock
  *         Enables and locks the PVD connection with Timer1/8 Break Input, , as well as the PVDE and PLS[2:0] in the PWR_CR register
  * @note   The selected configuration is locked and can be unlocked by system reset
  */
#define __HAL_SYSCFG_BREAK_LOCKUP_LOCK()     do {SYSCFG->CFGR2 &= ~(SYSCFG_CFGR2_LOCKUP_LOCK); \
                                                 SYSCFG->CFGR2 |= SYSCFG_CFGR2_LOCKUP_LOCK;    \
                                                }while(0)
/**
 * @}
 */
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx || STM32F413xx || STM32F423xx */
/**
  * @}
  */

/** @defgroup HAL_Private_Macros HAL Private Macros
  * @{
  */
#define IS_TICKFREQ(FREQ) (((FREQ) == HAL_TICK_FREQ_10HZ)  || \
                           ((FREQ) == HAL_TICK_FREQ_100HZ) || \
                           ((FREQ) == HAL_TICK_FREQ_1KHZ))
/**
  * @}
  */

/* Exported variables --------------------------------------------------------*/

/** @addtogroup HAL_Exported_Variables
  * @{
  */
extern __IO uint32_t uwTick;
extern uint32_t uwTickPrio;
extern HAL_TickFreqTypeDef uwTickFreq;
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup HAL_Exported_Functions
  * @{
  */
/** @addtogroup HAL_Exported_Functions_Group1
  * @{
  */
/* Initialization and Configuration functions  ******************************/
HAL_StatusTypeDef HAL_Init(void);
HAL_StatusTypeDef HAL_DeInit(void);
void HAL_MspInit(void);
void HAL_MspDeInit(void);
HAL_StatusTypeDef HAL_InitTick (uint32_t TickPriority);
/**
  * @}
  */

/** @addtogroup HAL_Exported_Functions_Group2
  * @{
  */
/* Peripheral Control functions  ************************************************/
void HAL_IncTick(void);
void HAL_Delay(uint32_t Delay);
uint32_t HAL_GetTick(void);
uint32_t HAL_GetTickPrio(void);
HAL_StatusTypeDef HAL_SetTickFreq(HAL_TickFreqTypeDef Freq);
HAL_TickFreqTypeDef HAL_GetTickFreq(void);
void HAL_SuspendTick(void);
void HAL_ResumeTick(void);
uint32_t HAL_GetHalVersion(void);
uint32_t HAL_GetREVID(void);
uint32_t HAL_GetDEVID(void);
void HAL_DBGMCU_EnableDBGSleepMode(void);
void HAL_DBGMCU_DisableDBGSleepMode(void);
void HAL_DBGMCU_EnableDBGStopMode(void);
void HAL_DBGMCU_DisableDBGStopMode(void);
void HAL_DBGMCU_EnableDBGStandbyMode(void);
void HAL_DBGMCU_DisableDBGStandbyMode(void);
void HAL_EnableCompensationCell(void);
void HAL_DisableCompensationCell(void);
uint32_t HAL_GetUIDw0(void);
uint32_t HAL_GetUIDw1(void);
uint32_t HAL_GetUIDw2(void);
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
void HAL_EnableMemorySwappingBank(void);
void HAL_DisableMemorySwappingBank(void);
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */ 
/**
  * @}
  */

/**
  * @}
  */
/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/** @defgroup HAL_Private_Variables HAL Private Variables
  * @{
  */
/**
  * @}
  */
/* Private constants ---------------------------------------------------------*/
/** @defgroup HAL_Private_Constants HAL Private Constants
  * @{
  */
/**
  * @}
  */
/* Private macros ------------------------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/**
  * @}
  */

/**
  * @}
  */ 
  
#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
