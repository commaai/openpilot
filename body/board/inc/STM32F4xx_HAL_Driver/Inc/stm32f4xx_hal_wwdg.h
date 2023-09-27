/**
  ******************************************************************************
  * @file    stm32f4xx_hal_wwdg.h
  * @author  MCD Application Team
  * @brief   Header file of WWDG HAL module.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2016 STMicroelectronics.
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
#ifndef STM32F4xx_HAL_WWDG_H
#define STM32F4xx_HAL_WWDG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup WWDG
  * @{
  */

/* Exported types ------------------------------------------------------------*/

/** @defgroup WWDG_Exported_Types WWDG Exported Types
  * @{
  */

/**
  * @brief  WWDG Init structure definition
  */
typedef struct
{
  uint32_t Prescaler;     /*!< Specifies the prescaler value of the WWDG.
                               This parameter can be a value of @ref WWDG_Prescaler */

  uint32_t Window;        /*!< Specifies the WWDG window value to be compared to the downcounter.
                               This parameter must be a number Min_Data = 0x40 and Max_Data = 0x7F */

  uint32_t Counter;       /*!< Specifies the WWDG free-running downcounter  value.
                               This parameter must be a number between Min_Data = 0x40 and Max_Data = 0x7F */

  uint32_t EWIMode ;      /*!< Specifies if WWDG Early Wakeup Interrupt is enable or not.
                               This parameter can be a value of @ref WWDG_EWI_Mode */

} WWDG_InitTypeDef;

/**
  * @brief  WWDG handle Structure definition
  */
#if (USE_HAL_WWDG_REGISTER_CALLBACKS == 1)
typedef struct __WWDG_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_WWDG_REGISTER_CALLBACKS */
{
  WWDG_TypeDef      *Instance;  /*!< Register base address */

  WWDG_InitTypeDef  Init;       /*!< WWDG required parameters */

#if (USE_HAL_WWDG_REGISTER_CALLBACKS == 1)
  void (* EwiCallback)(struct __WWDG_HandleTypeDef *hwwdg);                  /*!< WWDG Early WakeUp Interrupt callback */

  void (* MspInitCallback)(struct __WWDG_HandleTypeDef *hwwdg);              /*!< WWDG Msp Init callback */
#endif /* USE_HAL_WWDG_REGISTER_CALLBACKS */
} WWDG_HandleTypeDef;

#if (USE_HAL_WWDG_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL WWDG common Callback ID enumeration definition
  */
typedef enum
{
  HAL_WWDG_EWI_CB_ID          = 0x00U,    /*!< WWDG EWI callback ID */
  HAL_WWDG_MSPINIT_CB_ID      = 0x01U,    /*!< WWDG MspInit callback ID */
} HAL_WWDG_CallbackIDTypeDef;

/**
  * @brief  HAL WWDG Callback pointer definition
  */
typedef void (*pWWDG_CallbackTypeDef)(WWDG_HandleTypeDef *hppp);  /*!< pointer to a WWDG common callback functions */

#endif /* USE_HAL_WWDG_REGISTER_CALLBACKS */
/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/

/** @defgroup WWDG_Exported_Constants WWDG Exported Constants
  * @{
  */

/** @defgroup WWDG_Interrupt_definition WWDG Interrupt definition
  * @{
  */
#define WWDG_IT_EWI                         WWDG_CFR_EWI  /*!< Early wakeup interrupt */
/**
  * @}
  */

/** @defgroup WWDG_Flag_definition WWDG Flag definition
  * @brief WWDG Flag definition
  * @{
  */
#define WWDG_FLAG_EWIF                      WWDG_SR_EWIF  /*!< Early wakeup interrupt flag */
/**
  * @}
  */

/** @defgroup WWDG_Prescaler WWDG Prescaler
  * @{
  */
#define WWDG_PRESCALER_1                    0x00000000u                              /*!< WWDG counter clock = (PCLK1/4096)/1 */
#define WWDG_PRESCALER_2                    WWDG_CFR_WDGTB_0                         /*!< WWDG counter clock = (PCLK1/4096)/2 */
#define WWDG_PRESCALER_4                    WWDG_CFR_WDGTB_1                         /*!< WWDG counter clock = (PCLK1/4096)/4 */
#define WWDG_PRESCALER_8                    (WWDG_CFR_WDGTB_1 | WWDG_CFR_WDGTB_0)    /*!< WWDG counter clock = (PCLK1/4096)/8 */
/**
  * @}
  */

/** @defgroup WWDG_EWI_Mode WWDG Early Wakeup Interrupt Mode
  * @{
  */
#define WWDG_EWI_DISABLE                    0x00000000u       /*!< EWI Disable */
#define WWDG_EWI_ENABLE                     WWDG_CFR_EWI      /*!< EWI Enable */
/**
  * @}
  */

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/

/** @defgroup WWDG_Private_Macros WWDG Private Macros
  * @{
  */
#define IS_WWDG_PRESCALER(__PRESCALER__)    (((__PRESCALER__) == WWDG_PRESCALER_1)  || \
                                             ((__PRESCALER__) == WWDG_PRESCALER_2)  || \
                                             ((__PRESCALER__) == WWDG_PRESCALER_4)  || \
                                             ((__PRESCALER__) == WWDG_PRESCALER_8))

#define IS_WWDG_WINDOW(__WINDOW__)          (((__WINDOW__) >= WWDG_CFR_W_6) && ((__WINDOW__) <= WWDG_CFR_W))

#define IS_WWDG_COUNTER(__COUNTER__)        (((__COUNTER__) >= WWDG_CR_T_6) && ((__COUNTER__) <= WWDG_CR_T))

#define IS_WWDG_EWI_MODE(__MODE__)          (((__MODE__) == WWDG_EWI_ENABLE) || \
                                             ((__MODE__) == WWDG_EWI_DISABLE))
/**
  * @}
  */


/* Exported macros ------------------------------------------------------------*/

/** @defgroup WWDG_Exported_Macros WWDG Exported Macros
  * @{
  */

/**
  * @brief  Enable the WWDG peripheral.
  * @param  __HANDLE__  WWDG handle
  * @retval None
  */
#define __HAL_WWDG_ENABLE(__HANDLE__)                         SET_BIT((__HANDLE__)->Instance->CR, WWDG_CR_WDGA)

/**
  * @brief  Enable the WWDG early wakeup interrupt.
  * @param  __HANDLE__: WWDG handle
  * @param  __INTERRUPT__  specifies the interrupt to enable.
  *         This parameter can be one of the following values:
  *            @arg WWDG_IT_EWI: Early wakeup interrupt
  * @note   Once enabled this interrupt cannot be disabled except by a system reset.
  * @retval None
  */
#define __HAL_WWDG_ENABLE_IT(__HANDLE__, __INTERRUPT__)       SET_BIT((__HANDLE__)->Instance->CFR, (__INTERRUPT__))

/**
  * @brief  Check whether the selected WWDG interrupt has occurred or not.
  * @param  __HANDLE__  WWDG handle
  * @param  __INTERRUPT__  specifies the it to check.
  *        This parameter can be one of the following values:
  *            @arg WWDG_FLAG_EWIF: Early wakeup interrupt IT
  * @retval The new state of WWDG_FLAG (SET or RESET).
  */
#define __HAL_WWDG_GET_IT(__HANDLE__, __INTERRUPT__)        __HAL_WWDG_GET_FLAG((__HANDLE__),(__INTERRUPT__))

/** @brief  Clear the WWDG interrupt pending bits.
  *         bits to clear the selected interrupt pending bits.
  * @param  __HANDLE__  WWDG handle
  * @param  __INTERRUPT__  specifies the interrupt pending bit to clear.
  *         This parameter can be one of the following values:
  *            @arg WWDG_FLAG_EWIF: Early wakeup interrupt flag
  */
#define __HAL_WWDG_CLEAR_IT(__HANDLE__, __INTERRUPT__)      __HAL_WWDG_CLEAR_FLAG((__HANDLE__), (__INTERRUPT__))

/**
  * @brief  Check whether the specified WWDG flag is set or not.
  * @param  __HANDLE__  WWDG handle
  * @param  __FLAG__  specifies the flag to check.
  *         This parameter can be one of the following values:
  *            @arg WWDG_FLAG_EWIF: Early wakeup interrupt flag
  * @retval The new state of WWDG_FLAG (SET or RESET).
  */
#define __HAL_WWDG_GET_FLAG(__HANDLE__, __FLAG__)           (((__HANDLE__)->Instance->SR & (__FLAG__)) == (__FLAG__))

/**
  * @brief  Clear the WWDG's pending flags.
  * @param  __HANDLE__  WWDG handle
  * @param  __FLAG__  specifies the flag to clear.
  *         This parameter can be one of the following values:
  *            @arg WWDG_FLAG_EWIF: Early wakeup interrupt flag
  * @retval None
  */
#define __HAL_WWDG_CLEAR_FLAG(__HANDLE__, __FLAG__)         ((__HANDLE__)->Instance->SR = ~(__FLAG__))

/** @brief  Check whether the specified WWDG interrupt source is enabled or not.
  * @param  __HANDLE__  WWDG Handle.
  * @param  __INTERRUPT__  specifies the WWDG interrupt source to check.
  *         This parameter can be one of the following values:
  *            @arg WWDG_IT_EWI: Early Wakeup Interrupt
  * @retval state of __INTERRUPT__ (TRUE or FALSE).
  */
#define __HAL_WWDG_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) (((__HANDLE__)->Instance->CFR\
                                                              & (__INTERRUPT__)) == (__INTERRUPT__))

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/

/** @addtogroup WWDG_Exported_Functions
  * @{
  */

/** @addtogroup WWDG_Exported_Functions_Group1
  * @{
  */
/* Initialization/de-initialization functions  **********************************/
HAL_StatusTypeDef     HAL_WWDG_Init(WWDG_HandleTypeDef *hwwdg);
void                  HAL_WWDG_MspInit(WWDG_HandleTypeDef *hwwdg);
/* Callbacks Register/UnRegister functions  ***********************************/
#if (USE_HAL_WWDG_REGISTER_CALLBACKS == 1)
HAL_StatusTypeDef     HAL_WWDG_RegisterCallback(WWDG_HandleTypeDef *hwwdg, HAL_WWDG_CallbackIDTypeDef CallbackID,
                                                pWWDG_CallbackTypeDef pCallback);
HAL_StatusTypeDef     HAL_WWDG_UnRegisterCallback(WWDG_HandleTypeDef *hwwdg, HAL_WWDG_CallbackIDTypeDef CallbackID);
#endif /* USE_HAL_WWDG_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @addtogroup WWDG_Exported_Functions_Group2
  * @{
  */
/* I/O operation functions ******************************************************/
HAL_StatusTypeDef     HAL_WWDG_Refresh(WWDG_HandleTypeDef *hwwdg);
void                  HAL_WWDG_IRQHandler(WWDG_HandleTypeDef *hwwdg);
void                  HAL_WWDG_EarlyWakeupCallback(WWDG_HandleTypeDef *hwwdg);
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

#endif /* STM32F4xx_HAL_WWDG_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
