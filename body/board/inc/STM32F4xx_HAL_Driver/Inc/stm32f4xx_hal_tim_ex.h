/**
  ******************************************************************************
  * @file    stm32f4xx_hal_tim_ex.h
  * @author  MCD Application Team
  * @brief   Header file of TIM HAL Extended module.
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
#ifndef STM32F4xx_HAL_TIM_EX_H
#define STM32F4xx_HAL_TIM_EX_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup TIMEx
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup TIMEx_Exported_Types TIM Extended Exported Types
  * @{
  */

/**
  * @brief  TIM Hall sensor Configuration Structure definition
  */

typedef struct
{
  uint32_t IC1Polarity;         /*!< Specifies the active edge of the input signal.
                                     This parameter can be a value of @ref TIM_Input_Capture_Polarity */

  uint32_t IC1Prescaler;        /*!< Specifies the Input Capture Prescaler.
                                     This parameter can be a value of @ref TIM_Input_Capture_Prescaler */

  uint32_t IC1Filter;           /*!< Specifies the input capture filter.
                                     This parameter can be a number between Min_Data = 0x0 and Max_Data = 0xF */

  uint32_t Commutation_Delay;   /*!< Specifies the pulse value to be loaded into the Capture Compare Register.
                                     This parameter can be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF */
} TIM_HallSensor_InitTypeDef;
/**
  * @}
  */
/* End of exported types -----------------------------------------------------*/

/* Exported constants --------------------------------------------------------*/
/** @defgroup TIMEx_Exported_Constants TIM Extended Exported Constants
  * @{
  */

/** @defgroup TIMEx_Remap TIM Extended Remapping
  * @{
  */
#if defined (TIM2)
#if defined(TIM8)
#define TIM_TIM2_TIM8_TRGO                     0x00000000U                              /*!< TIM2 ITR1 is connected to TIM8 TRGO */
#else
#define TIM_TIM2_ETH_PTP                       TIM_OR_ITR1_RMP_0                        /*!< TIM2 ITR1 is connected to PTP trigger output */
#endif /*  TIM8 */
#define TIM_TIM2_USBFS_SOF                     TIM_OR_ITR1_RMP_1                        /*!< TIM2 ITR1 is connected to OTG FS SOF */
#define TIM_TIM2_USBHS_SOF                     (TIM_OR_ITR1_RMP_1 | TIM_OR_ITR1_RMP_0)  /*!< TIM2 ITR1 is connected to OTG HS SOF */
#endif /* TIM2 */

#define TIM_TIM5_GPIO                          0x00000000U                              /*!< TIM5 TI4 is connected to GPIO */
#define TIM_TIM5_LSI                           TIM_OR_TI4_RMP_0                         /*!< TIM5 TI4 is connected to LSI */
#define TIM_TIM5_LSE                           TIM_OR_TI4_RMP_1                         /*!< TIM5 TI4 is connected to LSE */
#define TIM_TIM5_RTC                           (TIM_OR_TI4_RMP_1 | TIM_OR_TI4_RMP_0)    /*!< TIM5 TI4 is connected to the RTC wakeup interrupt */

#define TIM_TIM11_GPIO                         0x00000000U                              /*!< TIM11 TI1 is connected to GPIO */
#define TIM_TIM11_HSE                          TIM_OR_TI1_RMP_1                         /*!< TIM11 TI1 is connected to HSE_RTC clock */
#if defined(SPDIFRX)
#define TIM_TIM11_SPDIFRX                      TIM_OR_TI1_RMP_0                         /*!< TIM11 TI1 is connected to SPDIFRX_FRAME_SYNC */
#endif /* SPDIFRX*/

#if defined(LPTIM_OR_TIM1_ITR2_RMP) && defined(LPTIM_OR_TIM5_ITR1_RMP) && defined(LPTIM_OR_TIM5_ITR1_RMP)
#define LPTIM_REMAP_MASK                       0x10000000U

#define TIM_TIM9_TIM3_TRGO                     LPTIM_REMAP_MASK                             /*!< TIM9 ITR1 is connected to TIM3 TRGO */
#define TIM_TIM9_LPTIM                         (LPTIM_REMAP_MASK | LPTIM_OR_TIM9_ITR1_RMP)  /*!< TIM9 ITR1 is connected to LPTIM1 output */

#define TIM_TIM5_TIM3_TRGO                     LPTIM_REMAP_MASK                             /*!< TIM5 ITR1 is connected to TIM3 TRGO */
#define TIM_TIM5_LPTIM                         (LPTIM_REMAP_MASK | LPTIM_OR_TIM5_ITR1_RMP)  /*!< TIM5 ITR1 is connected to LPTIM1 output */

#define TIM_TIM1_TIM3_TRGO                     LPTIM_REMAP_MASK                             /*!< TIM1 ITR2 is connected to TIM3 TRGO */
#define TIM_TIM1_LPTIM                         (LPTIM_REMAP_MASK | LPTIM_OR_TIM1_ITR2_RMP)  /*!< TIM1 ITR2 is connected to LPTIM1 output */
#endif /* LPTIM_OR_TIM1_ITR2_RMP &&  LPTIM_OR_TIM5_ITR1_RMP && LPTIM_OR_TIM5_ITR1_RMP */
/**
  * @}
  */

/**
  * @}
  */
/* End of exported constants -------------------------------------------------*/

/* Exported macro ------------------------------------------------------------*/
/** @defgroup TIMEx_Exported_Macros TIM Extended Exported Macros
  * @{
  */

/**
  * @}
  */
/* End of exported macro -----------------------------------------------------*/

/* Private macro -------------------------------------------------------------*/
/** @defgroup TIMEx_Private_Macros TIM Extended Private Macros
  * @{
  */
#if defined(SPDIFRX)
#define IS_TIM_REMAP(INSTANCE, TIM_REMAP)                                 \
  ((((INSTANCE) == TIM2)  && (((TIM_REMAP) == TIM_TIM2_TIM8_TRGO)      || \
                              ((TIM_REMAP) == TIM_TIM2_USBFS_SOF)      || \
                              ((TIM_REMAP) == TIM_TIM2_USBHS_SOF)))    || \
   (((INSTANCE) == TIM5)  && (((TIM_REMAP) == TIM_TIM5_GPIO)           || \
                              ((TIM_REMAP) == TIM_TIM5_LSI)            || \
                              ((TIM_REMAP) == TIM_TIM5_LSE)            || \
                              ((TIM_REMAP) == TIM_TIM5_RTC)))          || \
   (((INSTANCE) == TIM11) && (((TIM_REMAP) == TIM_TIM11_GPIO)          || \
                              ((TIM_REMAP) == TIM_TIM11_SPDIFRX)       || \
                              ((TIM_REMAP) == TIM_TIM11_HSE))))
#elif defined(TIM2)
#if defined(LPTIM_OR_TIM1_ITR2_RMP) && defined(LPTIM_OR_TIM5_ITR1_RMP) && defined(LPTIM_OR_TIM5_ITR1_RMP)
#define IS_TIM_REMAP(INSTANCE, TIM_REMAP)                                 \
  ((((INSTANCE) == TIM2)  && (((TIM_REMAP) == TIM_TIM2_TIM8_TRGO)      || \
                              ((TIM_REMAP) == TIM_TIM2_USBFS_SOF)      || \
                              ((TIM_REMAP) == TIM_TIM2_USBHS_SOF)))    || \
   (((INSTANCE) == TIM5)  && (((TIM_REMAP) == TIM_TIM5_GPIO)           || \
                              ((TIM_REMAP) == TIM_TIM5_LSI)            || \
                              ((TIM_REMAP) == TIM_TIM5_LSE)            || \
                              ((TIM_REMAP) == TIM_TIM5_RTC)))          || \
   (((INSTANCE) == TIM11) && (((TIM_REMAP) == TIM_TIM11_GPIO)          || \
                              ((TIM_REMAP) == TIM_TIM11_HSE)))         || \
   (((INSTANCE) == TIM1)  && (((TIM_REMAP) == TIM_TIM1_TIM3_TRGO)      || \
                              ((TIM_REMAP) == TIM_TIM1_LPTIM)))        || \
   (((INSTANCE) == TIM5)  && (((TIM_REMAP) == TIM_TIM5_TIM3_TRGO)      || \
                              ((TIM_REMAP) == TIM_TIM5_LPTIM)))        || \
   (((INSTANCE) == TIM9)  && (((TIM_REMAP) == TIM_TIM9_TIM3_TRGO)      || \
                              ((TIM_REMAP) == TIM_TIM9_LPTIM))))
#elif defined(TIM8)
#define IS_TIM_REMAP(INSTANCE, TIM_REMAP)                                 \
  ((((INSTANCE) == TIM2)  && (((TIM_REMAP) == TIM_TIM2_TIM8_TRGO)      || \
                              ((TIM_REMAP) == TIM_TIM2_USBFS_SOF)      || \
                              ((TIM_REMAP) == TIM_TIM2_USBHS_SOF)))    || \
   (((INSTANCE) == TIM5)  && (((TIM_REMAP) == TIM_TIM5_GPIO)           || \
                              ((TIM_REMAP) == TIM_TIM5_LSI)            || \
                              ((TIM_REMAP) == TIM_TIM5_LSE)            || \
                              ((TIM_REMAP) == TIM_TIM5_RTC)))          || \
   (((INSTANCE) == TIM11) && (((TIM_REMAP) == TIM_TIM11_GPIO)          || \
                              ((TIM_REMAP) == TIM_TIM11_HSE))))
#else
#define IS_TIM_REMAP(INSTANCE, TIM_REMAP)                                 \
  ((((INSTANCE) == TIM2)  && (((TIM_REMAP) == TIM_TIM2_ETH_PTP)        || \
                              ((TIM_REMAP) == TIM_TIM2_USBFS_SOF)      || \
                              ((TIM_REMAP) == TIM_TIM2_USBHS_SOF)))    || \
   (((INSTANCE) == TIM5)  && (((TIM_REMAP) == TIM_TIM5_GPIO)           || \
                              ((TIM_REMAP) == TIM_TIM5_LSI)            || \
                              ((TIM_REMAP) == TIM_TIM5_LSE)            || \
                              ((TIM_REMAP) == TIM_TIM5_RTC)))          || \
   (((INSTANCE) == TIM11) && (((TIM_REMAP) == TIM_TIM11_GPIO)          || \
                              ((TIM_REMAP) == TIM_TIM11_HSE))))
#endif /* LPTIM_OR_TIM1_ITR2_RMP &&  LPTIM_OR_TIM5_ITR1_RMP && LPTIM_OR_TIM5_ITR1_RMP */
#else
#define IS_TIM_REMAP(INSTANCE, TIM_REMAP)                                 \
  ((((INSTANCE) == TIM5)  && (((TIM_REMAP) == TIM_TIM5_GPIO)           || \
                              ((TIM_REMAP) == TIM_TIM5_LSI)            || \
                              ((TIM_REMAP) == TIM_TIM5_LSE)            || \
                              ((TIM_REMAP) == TIM_TIM5_RTC)))          || \
   (((INSTANCE) == TIM11) && (((TIM_REMAP) == TIM_TIM11_GPIO)          || \
                              ((TIM_REMAP) == TIM_TIM11_HSE))))
#endif /* SPDIFRX */

/**
  * @}
  */
/* End of private macro ------------------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup TIMEx_Exported_Functions TIM Extended Exported Functions
  * @{
  */

/** @addtogroup TIMEx_Exported_Functions_Group1 Extended Timer Hall Sensor functions
  *  @brief    Timer Hall Sensor functions
  * @{
  */
/*  Timer Hall Sensor functions  **********************************************/
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Init(TIM_HandleTypeDef *htim, TIM_HallSensor_InitTypeDef *sConfig);
HAL_StatusTypeDef HAL_TIMEx_HallSensor_DeInit(TIM_HandleTypeDef *htim);

void HAL_TIMEx_HallSensor_MspInit(TIM_HandleTypeDef *htim);
void HAL_TIMEx_HallSensor_MspDeInit(TIM_HandleTypeDef *htim);

/* Blocking mode: Polling */
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Start(TIM_HandleTypeDef *htim);
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Stop(TIM_HandleTypeDef *htim);
/* Non-Blocking mode: Interrupt */
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Start_IT(TIM_HandleTypeDef *htim);
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Stop_IT(TIM_HandleTypeDef *htim);
/* Non-Blocking mode: DMA */
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Start_DMA(TIM_HandleTypeDef *htim, uint32_t *pData, uint16_t Length);
HAL_StatusTypeDef HAL_TIMEx_HallSensor_Stop_DMA(TIM_HandleTypeDef *htim);
/**
  * @}
  */

/** @addtogroup TIMEx_Exported_Functions_Group2 Extended Timer Complementary Output Compare functions
  *  @brief   Timer Complementary Output Compare functions
  * @{
  */
/*  Timer Complementary Output Compare functions  *****************************/
/* Blocking mode: Polling */
HAL_StatusTypeDef HAL_TIMEx_OCN_Start(TIM_HandleTypeDef *htim, uint32_t Channel);
HAL_StatusTypeDef HAL_TIMEx_OCN_Stop(TIM_HandleTypeDef *htim, uint32_t Channel);

/* Non-Blocking mode: Interrupt */
HAL_StatusTypeDef HAL_TIMEx_OCN_Start_IT(TIM_HandleTypeDef *htim, uint32_t Channel);
HAL_StatusTypeDef HAL_TIMEx_OCN_Stop_IT(TIM_HandleTypeDef *htim, uint32_t Channel);

/* Non-Blocking mode: DMA */
HAL_StatusTypeDef HAL_TIMEx_OCN_Start_DMA(TIM_HandleTypeDef *htim, uint32_t Channel, uint32_t *pData, uint16_t Length);
HAL_StatusTypeDef HAL_TIMEx_OCN_Stop_DMA(TIM_HandleTypeDef *htim, uint32_t Channel);
/**
  * @}
  */

/** @addtogroup TIMEx_Exported_Functions_Group3 Extended Timer Complementary PWM functions
  *  @brief    Timer Complementary PWM functions
  * @{
  */
/*  Timer Complementary PWM functions  ****************************************/
/* Blocking mode: Polling */
HAL_StatusTypeDef HAL_TIMEx_PWMN_Start(TIM_HandleTypeDef *htim, uint32_t Channel);
HAL_StatusTypeDef HAL_TIMEx_PWMN_Stop(TIM_HandleTypeDef *htim, uint32_t Channel);

/* Non-Blocking mode: Interrupt */
HAL_StatusTypeDef HAL_TIMEx_PWMN_Start_IT(TIM_HandleTypeDef *htim, uint32_t Channel);
HAL_StatusTypeDef HAL_TIMEx_PWMN_Stop_IT(TIM_HandleTypeDef *htim, uint32_t Channel);
/* Non-Blocking mode: DMA */
HAL_StatusTypeDef HAL_TIMEx_PWMN_Start_DMA(TIM_HandleTypeDef *htim, uint32_t Channel, uint32_t *pData, uint16_t Length);
HAL_StatusTypeDef HAL_TIMEx_PWMN_Stop_DMA(TIM_HandleTypeDef *htim, uint32_t Channel);
/**
  * @}
  */

/** @addtogroup TIMEx_Exported_Functions_Group4 Extended Timer Complementary One Pulse functions
  *  @brief    Timer Complementary One Pulse functions
  * @{
  */
/*  Timer Complementary One Pulse functions  **********************************/
/* Blocking mode: Polling */
HAL_StatusTypeDef HAL_TIMEx_OnePulseN_Start(TIM_HandleTypeDef *htim, uint32_t OutputChannel);
HAL_StatusTypeDef HAL_TIMEx_OnePulseN_Stop(TIM_HandleTypeDef *htim, uint32_t OutputChannel);

/* Non-Blocking mode: Interrupt */
HAL_StatusTypeDef HAL_TIMEx_OnePulseN_Start_IT(TIM_HandleTypeDef *htim, uint32_t OutputChannel);
HAL_StatusTypeDef HAL_TIMEx_OnePulseN_Stop_IT(TIM_HandleTypeDef *htim, uint32_t OutputChannel);
/**
  * @}
  */

/** @addtogroup TIMEx_Exported_Functions_Group5 Extended Peripheral Control functions
  *  @brief    Peripheral Control functions
  * @{
  */
/* Extended Control functions  ************************************************/
HAL_StatusTypeDef HAL_TIMEx_ConfigCommutEvent(TIM_HandleTypeDef *htim, uint32_t  InputTrigger,
                                              uint32_t  CommutationSource);
HAL_StatusTypeDef HAL_TIMEx_ConfigCommutEvent_IT(TIM_HandleTypeDef *htim, uint32_t  InputTrigger,
                                                 uint32_t  CommutationSource);
HAL_StatusTypeDef HAL_TIMEx_ConfigCommutEvent_DMA(TIM_HandleTypeDef *htim, uint32_t  InputTrigger,
                                                  uint32_t  CommutationSource);
HAL_StatusTypeDef HAL_TIMEx_MasterConfigSynchronization(TIM_HandleTypeDef *htim,
                                                        TIM_MasterConfigTypeDef *sMasterConfig);
HAL_StatusTypeDef HAL_TIMEx_ConfigBreakDeadTime(TIM_HandleTypeDef *htim,
                                                TIM_BreakDeadTimeConfigTypeDef *sBreakDeadTimeConfig);
HAL_StatusTypeDef HAL_TIMEx_RemapConfig(TIM_HandleTypeDef *htim, uint32_t Remap);
/**
  * @}
  */

/** @addtogroup TIMEx_Exported_Functions_Group6 Extended Callbacks functions
  * @brief    Extended Callbacks functions
  * @{
  */
/* Extended Callback **********************************************************/
void HAL_TIMEx_CommutCallback(TIM_HandleTypeDef *htim);
void HAL_TIMEx_CommutHalfCpltCallback(TIM_HandleTypeDef *htim);
void HAL_TIMEx_BreakCallback(TIM_HandleTypeDef *htim);
/**
  * @}
  */

/** @addtogroup TIMEx_Exported_Functions_Group7 Extended Peripheral State functions
  * @brief    Extended Peripheral State functions
  * @{
  */
/* Extended Peripheral State functions  ***************************************/
HAL_TIM_StateTypeDef HAL_TIMEx_HallSensor_GetState(TIM_HandleTypeDef *htim);
HAL_TIM_ChannelStateTypeDef HAL_TIMEx_GetChannelNState(TIM_HandleTypeDef *htim,  uint32_t ChannelN);
/**
  * @}
  */

/**
  * @}
  */
/* End of exported functions -------------------------------------------------*/

/* Private functions----------------------------------------------------------*/
/** @addtogroup TIMEx_Private_Functions TIMEx Private Functions
  * @{
  */
void TIMEx_DMACommutationCplt(DMA_HandleTypeDef *hdma);
void TIMEx_DMACommutationHalfCplt(DMA_HandleTypeDef *hdma);
/**
  * @}
  */
/* End of private functions --------------------------------------------------*/

/**
  * @}
  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif


#endif /* STM32F4xx_HAL_TIM_EX_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
