/**
  ******************************************************************************
  * @file    stm32f4xx_ll_utils.h
  * @author  MCD Application Team
  * @brief   Header file of UTILS LL module.
  @verbatim
  ==============================================================================
                     ##### How to use this driver #####
  ==============================================================================
    [..]
    The LL UTILS driver contains a set of generic APIs that can be
    used by user:
      (+) Device electronic signature
      (+) Timing functions
      (+) PLL configuration functions

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
#ifndef __STM32F4xx_LL_UTILS_H
#define __STM32F4xx_LL_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

/** @defgroup UTILS_LL UTILS
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/

/* Private constants ---------------------------------------------------------*/
/** @defgroup UTILS_LL_Private_Constants UTILS Private Constants
  * @{
  */

/* Max delay can be used in LL_mDelay */
#define LL_MAX_DELAY                  0xFFFFFFFFU

/**
 * @brief Unique device ID register base address
 */
#define UID_BASE_ADDRESS              UID_BASE

/**
 * @brief Flash size data register base address
 */
#define FLASHSIZE_BASE_ADDRESS        FLASHSIZE_BASE

/**
 * @brief Package data register base address
 */
#define PACKAGE_BASE_ADDRESS          PACKAGE_BASE

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup UTILS_LL_Private_Macros UTILS Private Macros
  * @{
  */
/**
  * @}
  */
/* Exported types ------------------------------------------------------------*/
/** @defgroup UTILS_LL_ES_INIT UTILS Exported structures
  * @{
  */
/**
  * @brief  UTILS PLL structure definition
  */
typedef struct
{
  uint32_t PLLM;   /*!< Division factor for PLL VCO input clock.
                        This parameter can be a value of @ref RCC_LL_EC_PLLM_DIV

                        This feature can be modified afterwards using unitary function
                        @ref LL_RCC_PLL_ConfigDomain_SYS(). */

  uint32_t PLLN;   /*!< Multiplication factor for PLL VCO output clock.
                        This parameter must be a number between Min_Data = @ref RCC_PLLN_MIN_VALUE
                        and Max_Data = @ref RCC_PLLN_MIN_VALUE

                        This feature can be modified afterwards using unitary function
                        @ref LL_RCC_PLL_ConfigDomain_SYS(). */

  uint32_t PLLP;   /*!< Division for the main system clock.
                        This parameter can be a value of @ref RCC_LL_EC_PLLP_DIV

                        This feature can be modified afterwards using unitary function
                        @ref LL_RCC_PLL_ConfigDomain_SYS(). */
} LL_UTILS_PLLInitTypeDef;

/**
  * @brief  UTILS System, AHB and APB buses clock configuration structure definition
  */
typedef struct
{
  uint32_t AHBCLKDivider;         /*!< The AHB clock (HCLK) divider. This clock is derived from the system clock (SYSCLK).
                                       This parameter can be a value of @ref RCC_LL_EC_SYSCLK_DIV

                                       This feature can be modified afterwards using unitary function
                                       @ref LL_RCC_SetAHBPrescaler(). */

  uint32_t APB1CLKDivider;        /*!< The APB1 clock (PCLK1) divider. This clock is derived from the AHB clock (HCLK).
                                       This parameter can be a value of @ref RCC_LL_EC_APB1_DIV

                                       This feature can be modified afterwards using unitary function
                                       @ref LL_RCC_SetAPB1Prescaler(). */

  uint32_t APB2CLKDivider;        /*!< The APB2 clock (PCLK2) divider. This clock is derived from the AHB clock (HCLK).
                                       This parameter can be a value of @ref RCC_LL_EC_APB2_DIV

                                       This feature can be modified afterwards using unitary function
                                       @ref LL_RCC_SetAPB2Prescaler(). */

} LL_UTILS_ClkInitTypeDef;

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup UTILS_LL_Exported_Constants UTILS Exported Constants
  * @{
  */

/** @defgroup UTILS_EC_HSE_BYPASS HSE Bypass activation
  * @{
  */
#define LL_UTILS_HSEBYPASS_OFF        0x00000000U       /*!< HSE Bypass is not enabled                */
#define LL_UTILS_HSEBYPASS_ON         0x00000001U       /*!< HSE Bypass is enabled                    */
/**
  * @}
  */

/** @defgroup UTILS_EC_PACKAGETYPE PACKAGE TYPE
  * @{
  */
#define LL_UTILS_PACKAGETYPE_WLCSP36_UFQFPN48_LQFP64                        0x00000000U /*!< WLCSP36 or UFQFPN48 or LQFP64 package type                         */
#define LL_UTILS_PACKAGETYPE_WLCSP168_FBGA169_LQFP100_LQFP64_UFQFPN48       0x00000100U /*!< WLCSP168 or FBGA169 or LQFP100 or LQFP64 or UFQFPN48 package type  */
#define LL_UTILS_PACKAGETYPE_WLCSP64_WLCSP81_LQFP176_UFBGA176               0x00000200U /*!< WLCSP64 or WLCSP81 or LQFP176 or UFBGA176 package type             */
#define LL_UTILS_PACKAGETYPE_LQFP144_UFBGA144_UFBGA144_UFBGA100             0x00000300U /*!< LQFP144 or UFBGA144 or UFBGA144 or UFBGA100 package type           */
#define LL_UTILS_PACKAGETYPE_LQFP100_LQFP208_TFBGA216                       0x00000400U /*!< LQFP100 or LQFP208 or TFBGA216 package type                        */
#define LL_UTILS_PACKAGETYPE_LQFP208_TFBGA216                               0x00000500U /*!< LQFP208 or TFBGA216 package type                                   */
#define LL_UTILS_PACKAGETYPE_TQFP64_UFBGA144_LQFP144                        0x00000700U /*!< TQFP64 or UFBGA144 or LQFP144 package type                         */
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @defgroup UTILS_LL_Exported_Functions UTILS Exported Functions
  * @{
  */

/** @defgroup UTILS_EF_DEVICE_ELECTRONIC_SIGNATURE DEVICE ELECTRONIC SIGNATURE
  * @{
  */

/**
  * @brief  Get Word0 of the unique device identifier (UID based on 96 bits)
  * @retval UID[31:0]
  */
__STATIC_INLINE uint32_t LL_GetUID_Word0(void)
{
  return (uint32_t)(READ_REG(*((uint32_t *)UID_BASE_ADDRESS)));
}

/**
  * @brief  Get Word1 of the unique device identifier (UID based on 96 bits)
  * @retval UID[63:32]
  */
__STATIC_INLINE uint32_t LL_GetUID_Word1(void)
{
  return (uint32_t)(READ_REG(*((uint32_t *)(UID_BASE_ADDRESS + 4U))));
}

/**
  * @brief  Get Word2 of the unique device identifier (UID based on 96 bits)
  * @retval UID[95:64]
  */
__STATIC_INLINE uint32_t LL_GetUID_Word2(void)
{
  return (uint32_t)(READ_REG(*((uint32_t *)(UID_BASE_ADDRESS + 8U))));
}

/**
  * @brief  Get Flash memory size
  * @note   This bitfield indicates the size of the device Flash memory expressed in
  *         Kbytes. As an example, 0x040 corresponds to 64 Kbytes.
  * @retval FLASH_SIZE[15:0]: Flash memory size
  */
__STATIC_INLINE uint32_t LL_GetFlashSize(void)
{
  return (uint32_t)(READ_REG(*((uint32_t *)FLASHSIZE_BASE_ADDRESS)) & 0xFFFF);
}

/**
  * @brief  Get Package type
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_UTILS_PACKAGETYPE_WLCSP36_UFQFPN48_LQFP64 (*)
  *         @arg @ref LL_UTILS_PACKAGETYPE_WLCSP168_FBGA169_LQFP100_LQFP64_UFQFPN48 (*)
  *         @arg @ref LL_UTILS_PACKAGETYPE_WLCSP64_WLCSP81_LQFP176_UFBGA176 (*)
  *         @arg @ref LL_UTILS_PACKAGETYPE_LQFP144_UFBGA144_UFBGA144_UFBGA100 (*)
  *         @arg @ref LL_UTILS_PACKAGETYPE_LQFP100_LQFP208_TFBGA216 (*)
  *         @arg @ref LL_UTILS_PACKAGETYPE_LQFP208_TFBGA216 (*)
  *         @arg @ref LL_UTILS_PACKAGETYPE_TQFP64_UFBGA144_LQFP144 (*)
  * 
  *         (*) value not defined in all devices.
  */
__STATIC_INLINE uint32_t LL_GetPackageType(void)
{
  return (uint32_t)(READ_REG(*((uint32_t *)PACKAGE_BASE_ADDRESS)) & 0x0700U);
}

/**
  * @}
  */

/** @defgroup UTILS_LL_EF_DELAY DELAY
  * @{
  */

/**
  * @brief  This function configures the Cortex-M SysTick source of the time base.
  * @param  HCLKFrequency HCLK frequency in Hz (can be calculated thanks to RCC helper macro)
  * @note   When a RTOS is used, it is recommended to avoid changing the SysTick
  *         configuration by calling this function, for a delay use rather osDelay RTOS service.
  * @param  Ticks Number of ticks
  * @retval None
  */
__STATIC_INLINE void LL_InitTick(uint32_t HCLKFrequency, uint32_t Ticks)
{
  /* Configure the SysTick to have interrupt in 1ms time base */
  SysTick->LOAD  = (uint32_t)((HCLKFrequency / Ticks) - 1UL);  /* set reload register */
  SysTick->VAL   = 0UL;                                       /* Load the SysTick Counter Value */
  SysTick->CTRL  = SysTick_CTRL_CLKSOURCE_Msk |
                   SysTick_CTRL_ENABLE_Msk;                   /* Enable the Systick Timer */
}

void        LL_Init1msTick(uint32_t HCLKFrequency);
void        LL_mDelay(uint32_t Delay);

/**
  * @}
  */

/** @defgroup UTILS_EF_SYSTEM SYSTEM
  * @{
  */

void        LL_SetSystemCoreClock(uint32_t HCLKFrequency);
ErrorStatus LL_SetFlashLatency(uint32_t HCLK_Frequency);
ErrorStatus LL_PLL_ConfigSystemClock_HSI(LL_UTILS_PLLInitTypeDef *UTILS_PLLInitStruct,
                                         LL_UTILS_ClkInitTypeDef *UTILS_ClkInitStruct);
ErrorStatus LL_PLL_ConfigSystemClock_HSE(uint32_t HSEFrequency, uint32_t HSEBypass,
                                         LL_UTILS_PLLInitTypeDef *UTILS_PLLInitStruct, LL_UTILS_ClkInitTypeDef *UTILS_ClkInitStruct);

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_LL_UTILS_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
