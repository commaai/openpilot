/**
  ******************************************************************************
  * @file    stm32f4xx_ll_lptim.h
  * @author  MCD Application Team
  * @brief   Header file of LPTIM LL module.
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
#ifndef STM32F4xx_LL_LPTIM_H
#define STM32F4xx_LL_LPTIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (LPTIM1)

/** @defgroup LPTIM_LL LPTIM
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/

/* Private constants ---------------------------------------------------------*/

/* Private macros ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup LPTIM_LL_Private_Macros LPTIM Private Macros
  * @{
  */
/**
  * @}
  */
#endif /*USE_FULL_LL_DRIVER*/

/* Exported types ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup LPTIM_LL_ES_INIT LPTIM Exported Init structure
  * @{
  */

/**
  * @brief  LPTIM Init structure definition
  */
typedef struct
{
  uint32_t ClockSource;    /*!< Specifies the source of the clock used by the LPTIM instance.
                                This parameter can be a value of @ref LPTIM_LL_EC_CLK_SOURCE.

                                This feature can be modified afterwards using unitary
                                function @ref LL_LPTIM_SetClockSource().*/

  uint32_t Prescaler;      /*!< Specifies the prescaler division ratio.
                                This parameter can be a value of @ref LPTIM_LL_EC_PRESCALER.

                                This feature can be modified afterwards using using unitary
                                function @ref LL_LPTIM_SetPrescaler().*/

  uint32_t Waveform;       /*!< Specifies the waveform shape.
                                This parameter can be a value of @ref LPTIM_LL_EC_OUTPUT_WAVEFORM.

                                This feature can be modified afterwards using unitary
                                function @ref LL_LPTIM_ConfigOutput().*/

  uint32_t Polarity;       /*!< Specifies waveform polarity.
                                This parameter can be a value of @ref LPTIM_LL_EC_OUTPUT_POLARITY.

                                This feature can be modified afterwards using unitary
                                function @ref LL_LPTIM_ConfigOutput().*/
} LL_LPTIM_InitTypeDef;

/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/* Exported constants --------------------------------------------------------*/
/** @defgroup LPTIM_LL_Exported_Constants LPTIM Exported Constants
  * @{
  */

/** @defgroup LPTIM_LL_EC_GET_FLAG Get Flags Defines
  * @brief    Flags defines which can be used with LL_LPTIM_ReadReg function
  * @{
  */
#define LL_LPTIM_ISR_CMPM                     LPTIM_ISR_CMPM     /*!< Compare match */
#define LL_LPTIM_ISR_CMPOK                    LPTIM_ISR_CMPOK    /*!< Compare register update OK */
#define LL_LPTIM_ISR_ARRM                     LPTIM_ISR_ARRM     /*!< Autoreload match */
#define LL_LPTIM_ISR_EXTTRIG                  LPTIM_ISR_EXTTRIG  /*!< External trigger edge event */
#define LL_LPTIM_ISR_ARROK                    LPTIM_ISR_ARROK    /*!< Autoreload register update OK */
#define LL_LPTIM_ISR_UP                       LPTIM_ISR_UP       /*!< Counter direction change down to up */
#define LL_LPTIM_ISR_DOWN                     LPTIM_ISR_DOWN     /*!< Counter direction change up to down */
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_IT IT Defines
  * @brief    IT defines which can be used with LL_LPTIM_ReadReg and  LL_LPTIM_WriteReg functions
  * @{
  */
#define LL_LPTIM_IER_CMPMIE                   LPTIM_IER_CMPMIE     /*!< Compare match */
#define LL_LPTIM_IER_CMPOKIE                  LPTIM_IER_CMPOKIE    /*!< Compare register update OK */
#define LL_LPTIM_IER_ARRMIE                   LPTIM_IER_ARRMIE     /*!< Autoreload match */
#define LL_LPTIM_IER_EXTTRIGIE                LPTIM_IER_EXTTRIGIE  /*!< External trigger edge event */
#define LL_LPTIM_IER_ARROKIE                  LPTIM_IER_ARROKIE    /*!< Autoreload register update OK */
#define LL_LPTIM_IER_UPIE                     LPTIM_IER_UPIE       /*!< Counter direction change down to up */
#define LL_LPTIM_IER_DOWNIE                   LPTIM_IER_DOWNIE     /*!< Counter direction change up to down */
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_OPERATING_MODE Operating Mode
  * @{
  */
#define LL_LPTIM_OPERATING_MODE_CONTINUOUS    LPTIM_CR_CNTSTRT /*!<LP Timer starts in continuous mode*/
#define LL_LPTIM_OPERATING_MODE_ONESHOT       LPTIM_CR_SNGSTRT /*!<LP Tilmer starts in single mode*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_UPDATE_MODE Update Mode
  * @{
  */
#define LL_LPTIM_UPDATE_MODE_IMMEDIATE        0x00000000U        /*!<Preload is disabled: registers are updated after each APB bus write access*/
#define LL_LPTIM_UPDATE_MODE_ENDOFPERIOD      LPTIM_CFGR_PRELOAD /*!<preload is enabled: registers are updated at the end of the current LPTIM period*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_COUNTER_MODE Counter Mode
  * @{
  */
#define LL_LPTIM_COUNTER_MODE_INTERNAL        0x00000000U          /*!<The counter is incremented following each internal clock pulse*/
#define LL_LPTIM_COUNTER_MODE_EXTERNAL        LPTIM_CFGR_COUNTMODE /*!<The counter is incremented following each valid clock pulse on the LPTIM external Input1*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_OUTPUT_WAVEFORM Output Waveform Type
  * @{
  */
#define LL_LPTIM_OUTPUT_WAVEFORM_PWM          0x00000000U     /*!<LPTIM  generates either a PWM waveform or a One pulse waveform depending on chosen operating mode CONTINUOUS or SINGLE*/
#define LL_LPTIM_OUTPUT_WAVEFORM_SETONCE      LPTIM_CFGR_WAVE /*!<LPTIM  generates a Set Once waveform*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_OUTPUT_POLARITY Output Polarity
  * @{
  */
#define LL_LPTIM_OUTPUT_POLARITY_REGULAR      0x00000000U             /*!<The LPTIM output reflects the compare results between LPTIMx_ARR and LPTIMx_CMP registers*/
#define LL_LPTIM_OUTPUT_POLARITY_INVERSE      LPTIM_CFGR_WAVPOL       /*!<The LPTIM output reflects the inverse of the compare results between LPTIMx_ARR and LPTIMx_CMP registers*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_PRESCALER Prescaler Value
  * @{
  */
#define LL_LPTIM_PRESCALER_DIV1               0x00000000U                               /*!<Prescaler division factor is set to 1*/
#define LL_LPTIM_PRESCALER_DIV2               LPTIM_CFGR_PRESC_0                        /*!<Prescaler division factor is set to 2*/
#define LL_LPTIM_PRESCALER_DIV4               LPTIM_CFGR_PRESC_1                        /*!<Prescaler division factor is set to 4*/
#define LL_LPTIM_PRESCALER_DIV8               (LPTIM_CFGR_PRESC_1 | LPTIM_CFGR_PRESC_0) /*!<Prescaler division factor is set to 8*/
#define LL_LPTIM_PRESCALER_DIV16              LPTIM_CFGR_PRESC_2                        /*!<Prescaler division factor is set to 16*/
#define LL_LPTIM_PRESCALER_DIV32              (LPTIM_CFGR_PRESC_2 | LPTIM_CFGR_PRESC_0) /*!<Prescaler division factor is set to 32*/
#define LL_LPTIM_PRESCALER_DIV64              (LPTIM_CFGR_PRESC_2 | LPTIM_CFGR_PRESC_1) /*!<Prescaler division factor is set to 64*/
#define LL_LPTIM_PRESCALER_DIV128             LPTIM_CFGR_PRESC                          /*!<Prescaler division factor is set to 128*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_TRIG_SOURCE Trigger Source
  * @{
  */
#define LL_LPTIM_TRIG_SOURCE_GPIO             0x00000000U                                                          /*!<External input trigger is connected to TIMx_ETR input*/
#define LL_LPTIM_TRIG_SOURCE_RTCALARMA        LPTIM_CFGR_TRIGSEL_0                                                 /*!<External input trigger is connected to RTC Alarm A*/
#define LL_LPTIM_TRIG_SOURCE_RTCALARMB        LPTIM_CFGR_TRIGSEL_1                                                 /*!<External input trigger is connected to RTC Alarm B*/
#define LL_LPTIM_TRIG_SOURCE_RTCTAMP1         (LPTIM_CFGR_TRIGSEL_1 | LPTIM_CFGR_TRIGSEL_0)                        /*!<External input trigger is connected to RTC Tamper 1*/
#define LL_LPTIM_TRIG_SOURCE_TIM1_TRGO        LPTIM_CFGR_TRIGSEL_2                                                 /*!<External input trigger is connected to TIM1*/
#define LL_LPTIM_TRIG_SOURCE_TIM5_TRGO        (LPTIM_CFGR_TRIGSEL_2 | LPTIM_CFGR_TRIGSEL_0)                        /*!<External input trigger is connected to TIM5*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_TRIG_FILTER Trigger Filter
  * @{
  */
#define LL_LPTIM_TRIG_FILTER_NONE             0x00000000U         /*!<Any trigger active level change is considered as a valid trigger*/
#define LL_LPTIM_TRIG_FILTER_2                LPTIM_CFGR_TRGFLT_0 /*!<Trigger active level change must be stable for at least 2 clock periods before it is considered as valid trigger*/
#define LL_LPTIM_TRIG_FILTER_4                LPTIM_CFGR_TRGFLT_1 /*!<Trigger active level change must be stable for at least 4 clock periods before it is considered as valid trigger*/
#define LL_LPTIM_TRIG_FILTER_8                LPTIM_CFGR_TRGFLT   /*!<Trigger active level change must be stable for at least 8 clock periods before it is considered as valid trigger*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_TRIG_POLARITY Trigger Polarity
  * @{
  */
#define LL_LPTIM_TRIG_POLARITY_RISING         LPTIM_CFGR_TRIGEN_0 /*!<LPTIM counter starts when a rising edge is detected*/
#define LL_LPTIM_TRIG_POLARITY_FALLING        LPTIM_CFGR_TRIGEN_1 /*!<LPTIM counter starts when a falling edge is detected*/
#define LL_LPTIM_TRIG_POLARITY_RISING_FALLING LPTIM_CFGR_TRIGEN   /*!<LPTIM counter starts when a rising or a falling edge is detected*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_CLK_SOURCE Clock Source
  * @{
  */
#define LL_LPTIM_CLK_SOURCE_INTERNAL          0x00000000U      /*!<LPTIM is clocked by internal clock source (APB clock or any of the embedded oscillators)*/
#define LL_LPTIM_CLK_SOURCE_EXTERNAL          LPTIM_CFGR_CKSEL /*!<LPTIM is clocked by an external clock source through the LPTIM external Input1*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_CLK_FILTER Clock Filter
  * @{
  */
#define LL_LPTIM_CLK_FILTER_NONE              0x00000000U        /*!<Any external clock signal level change is considered as a valid transition*/
#define LL_LPTIM_CLK_FILTER_2                 LPTIM_CFGR_CKFLT_0 /*!<External clock signal level change must be stable for at least 2 clock periods before it is considered as valid transition*/
#define LL_LPTIM_CLK_FILTER_4                 LPTIM_CFGR_CKFLT_1 /*!<External clock signal level change must be stable for at least 4 clock periods before it is considered as valid transition*/
#define LL_LPTIM_CLK_FILTER_8                 LPTIM_CFGR_CKFLT   /*!<External clock signal level change must be stable for at least 8 clock periods before it is considered as valid transition*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_CLK_POLARITY Clock Polarity
  * @{
  */
#define LL_LPTIM_CLK_POLARITY_RISING          0x00000000U        /*!< The rising edge is the active edge used for counting*/
#define LL_LPTIM_CLK_POLARITY_FALLING         LPTIM_CFGR_CKPOL_0 /*!< The falling edge is the active edge used for counting*/
#define LL_LPTIM_CLK_POLARITY_RISING_FALLING  LPTIM_CFGR_CKPOL_1 /*!< Both edges are active edges*/
/**
  * @}
  */

/** @defgroup LPTIM_LL_EC_ENCODER_MODE Encoder Mode
  * @{
  */
#define LL_LPTIM_ENCODER_MODE_RISING          0x00000000U        /*!< The rising edge is the active edge used for counting*/
#define LL_LPTIM_ENCODER_MODE_FALLING         LPTIM_CFGR_CKPOL_0 /*!< The falling edge is the active edge used for counting*/
#define LL_LPTIM_ENCODER_MODE_RISING_FALLING  LPTIM_CFGR_CKPOL_1 /*!< Both edges are active edges*/
/**
  * @}
  */

/** @defgroup LPTIM_EC_INPUT1_SRC Input1 Source
  * @{
  */
#define LL_LPTIM_INPUT1_SRC_PAD_AF       0x00000000U
#define LL_LPTIM_INPUT1_SRC_PAD_PA4      LPTIM_OR_OR_0
#define LL_LPTIM_INPUT1_SRC_PAD_PB9      LPTIM_OR_OR_1
#define LL_LPTIM_INPUT1_SRC_TIM_DAC      LPTIM_OR_OR
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup LPTIM_LL_Exported_Macros LPTIM Exported Macros
  * @{
  */

/** @defgroup LPTIM_LL_EM_WRITE_READ Common Write and read registers Macros
  * @{
  */

/**
  * @brief  Write a value in LPTIM register
  * @param  __INSTANCE__ LPTIM Instance
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_LPTIM_WriteReg(__INSTANCE__, __REG__, __VALUE__) WRITE_REG((__INSTANCE__)->__REG__, (__VALUE__))

/**
  * @brief  Read a value in LPTIM register
  * @param  __INSTANCE__ LPTIM Instance
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_LPTIM_ReadReg(__INSTANCE__, __REG__) READ_REG((__INSTANCE__)->__REG__)
/**
  * @}
  */

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup LPTIM_LL_Exported_Functions LPTIM Exported Functions
  * @{
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup LPTIM_LL_EF_Init Initialisation and deinitialisation functions
  * @{
  */

ErrorStatus LL_LPTIM_DeInit(LPTIM_TypeDef *LPTIMx);
void LL_LPTIM_StructInit(LL_LPTIM_InitTypeDef *LPTIM_InitStruct);
ErrorStatus LL_LPTIM_Init(LPTIM_TypeDef *LPTIMx, LL_LPTIM_InitTypeDef *LPTIM_InitStruct);
void LL_LPTIM_Disable(LPTIM_TypeDef *LPTIMx);
/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/** @defgroup LPTIM_LL_EF_LPTIM_Configuration LPTIM Configuration
  * @{
  */

/**
  * @brief  Enable the LPTIM instance
  * @note After setting the ENABLE bit, a delay of two counter clock is needed
  *       before the LPTIM instance is actually enabled.
  * @rmtoll CR           ENABLE        LL_LPTIM_Enable
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_Enable(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->CR, LPTIM_CR_ENABLE);
}

/**
  * @brief  Indicates whether the LPTIM instance is enabled.
  * @rmtoll CR           ENABLE        LL_LPTIM_IsEnabled
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabled(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->CR, LPTIM_CR_ENABLE) == LPTIM_CR_ENABLE) ? 1UL : 0UL));
}

/**
  * @brief  Starts the LPTIM counter in the desired mode.
  * @note LPTIM instance must be enabled before starting the counter.
  * @note It is possible to change on the fly from One Shot mode to
  *       Continuous mode.
  * @rmtoll CR           CNTSTRT       LL_LPTIM_StartCounter\n
  *         CR           SNGSTRT       LL_LPTIM_StartCounter
  * @param  LPTIMx Low-Power Timer instance
  * @param  OperatingMode This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_OPERATING_MODE_CONTINUOUS
  *         @arg @ref LL_LPTIM_OPERATING_MODE_ONESHOT
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_StartCounter(LPTIM_TypeDef *LPTIMx, uint32_t OperatingMode)
{
  MODIFY_REG(LPTIMx->CR, LPTIM_CR_CNTSTRT | LPTIM_CR_SNGSTRT, OperatingMode);
}

/**
  * @brief  Set the LPTIM registers update mode (enable/disable register preload)
  * @note This function must be called when the LPTIM instance is disabled.
  * @rmtoll CFGR         PRELOAD       LL_LPTIM_SetUpdateMode
  * @param  LPTIMx Low-Power Timer instance
  * @param  UpdateMode This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_UPDATE_MODE_IMMEDIATE
  *         @arg @ref LL_LPTIM_UPDATE_MODE_ENDOFPERIOD
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetUpdateMode(LPTIM_TypeDef *LPTIMx, uint32_t UpdateMode)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_PRELOAD, UpdateMode);
}

/**
  * @brief  Get the LPTIM registers update mode
  * @rmtoll CFGR         PRELOAD       LL_LPTIM_GetUpdateMode
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_UPDATE_MODE_IMMEDIATE
  *         @arg @ref LL_LPTIM_UPDATE_MODE_ENDOFPERIOD
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetUpdateMode(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_PRELOAD));
}

/**
  * @brief  Set the auto reload value
  * @note The LPTIMx_ARR register content must only be modified when the LPTIM is enabled
  * @note After a write to the LPTIMx_ARR register a new write operation to the
  *       same register can only be performed when the previous write operation
  *       is completed. Any successive write before  the ARROK flag is set, will
  *       lead to unpredictable results.
  * @note autoreload value be strictly greater than the compare value.
  * @rmtoll ARR          ARR           LL_LPTIM_SetAutoReload
  * @param  LPTIMx Low-Power Timer instance
  * @param  AutoReload Value between Min_Data=0x00 and Max_Data=0xFFFF
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetAutoReload(LPTIM_TypeDef *LPTIMx, uint32_t AutoReload)
{
  MODIFY_REG(LPTIMx->ARR, LPTIM_ARR_ARR, AutoReload);
}

/**
  * @brief  Get actual auto reload value
  * @rmtoll ARR          ARR           LL_LPTIM_GetAutoReload
  * @param  LPTIMx Low-Power Timer instance
  * @retval AutoReload Value between Min_Data=0x00 and Max_Data=0xFFFF
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetAutoReload(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->ARR, LPTIM_ARR_ARR));
}

/**
  * @brief  Set the compare value
  * @note After a write to the LPTIMx_CMP register a new write operation to the
  *       same register can only be performed when the previous write operation
  *       is completed. Any successive write before the CMPOK flag is set, will
  *       lead to unpredictable results.
  * @rmtoll CMP          CMP           LL_LPTIM_SetCompare
  * @param  LPTIMx Low-Power Timer instance
  * @param  CompareValue Value between Min_Data=0x00 and Max_Data=0xFFFF
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetCompare(LPTIM_TypeDef *LPTIMx, uint32_t CompareValue)
{
  MODIFY_REG(LPTIMx->CMP, LPTIM_CMP_CMP, CompareValue);
}

/**
  * @brief  Get actual compare value
  * @rmtoll CMP          CMP           LL_LPTIM_GetCompare
  * @param  LPTIMx Low-Power Timer instance
  * @retval CompareValue Value between Min_Data=0x00 and Max_Data=0xFFFF
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetCompare(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CMP, LPTIM_CMP_CMP));
}

/**
  * @brief  Get actual counter value
  * @note When the LPTIM instance is running with an asynchronous clock, reading
  *       the LPTIMx_CNT register may return unreliable values. So in this case
  *       it is necessary to perform two consecutive read accesses and verify
  *       that the two returned values are identical.
  * @rmtoll CNT          CNT           LL_LPTIM_GetCounter
  * @param  LPTIMx Low-Power Timer instance
  * @retval Counter value
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetCounter(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CNT, LPTIM_CNT_CNT));
}

/**
  * @brief  Set the counter mode (selection of the LPTIM counter clock source).
  * @note The counter mode can be set only when the LPTIM instance is disabled.
  * @rmtoll CFGR         COUNTMODE     LL_LPTIM_SetCounterMode
  * @param  LPTIMx Low-Power Timer instance
  * @param  CounterMode This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_COUNTER_MODE_INTERNAL
  *         @arg @ref LL_LPTIM_COUNTER_MODE_EXTERNAL
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetCounterMode(LPTIM_TypeDef *LPTIMx, uint32_t CounterMode)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_COUNTMODE, CounterMode);
}

/**
  * @brief  Get the counter mode
  * @rmtoll CFGR         COUNTMODE     LL_LPTIM_GetCounterMode
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_COUNTER_MODE_INTERNAL
  *         @arg @ref LL_LPTIM_COUNTER_MODE_EXTERNAL
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetCounterMode(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_COUNTMODE));
}

/**
  * @brief  Configure the LPTIM instance output (LPTIMx_OUT)
  * @note This function must be called when the LPTIM instance is disabled.
  * @note Regarding the LPTIM output polarity the change takes effect
  *       immediately, so the output default value will change immediately after
  *       the polarity is re-configured, even before the timer is enabled.
  * @rmtoll CFGR         WAVE          LL_LPTIM_ConfigOutput\n
  *         CFGR         WAVPOL        LL_LPTIM_ConfigOutput
  * @param  LPTIMx Low-Power Timer instance
  * @param  Waveform This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_OUTPUT_WAVEFORM_PWM
  *         @arg @ref LL_LPTIM_OUTPUT_WAVEFORM_SETONCE
  * @param  Polarity This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_OUTPUT_POLARITY_REGULAR
  *         @arg @ref LL_LPTIM_OUTPUT_POLARITY_INVERSE
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ConfigOutput(LPTIM_TypeDef *LPTIMx, uint32_t Waveform, uint32_t Polarity)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_WAVE | LPTIM_CFGR_WAVPOL, Waveform | Polarity);
}

/**
  * @brief  Set  waveform shape
  * @rmtoll CFGR         WAVE          LL_LPTIM_SetWaveform
  * @param  LPTIMx Low-Power Timer instance
  * @param  Waveform This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_OUTPUT_WAVEFORM_PWM
  *         @arg @ref LL_LPTIM_OUTPUT_WAVEFORM_SETONCE
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetWaveform(LPTIM_TypeDef *LPTIMx, uint32_t Waveform)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_WAVE, Waveform);
}

/**
  * @brief  Get actual waveform shape
  * @rmtoll CFGR         WAVE          LL_LPTIM_GetWaveform
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_OUTPUT_WAVEFORM_PWM
  *         @arg @ref LL_LPTIM_OUTPUT_WAVEFORM_SETONCE
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetWaveform(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_WAVE));
}

/**
  * @brief  Set  output polarity
  * @rmtoll CFGR         WAVPOL        LL_LPTIM_SetPolarity
  * @param  LPTIMx Low-Power Timer instance
  * @param  Polarity This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_OUTPUT_POLARITY_REGULAR
  *         @arg @ref LL_LPTIM_OUTPUT_POLARITY_INVERSE
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetPolarity(LPTIM_TypeDef *LPTIMx, uint32_t Polarity)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_WAVPOL, Polarity);
}

/**
  * @brief  Get actual output polarity
  * @rmtoll CFGR         WAVPOL        LL_LPTIM_GetPolarity
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_OUTPUT_POLARITY_REGULAR
  *         @arg @ref LL_LPTIM_OUTPUT_POLARITY_INVERSE
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetPolarity(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_WAVPOL));
}

/**
  * @brief  Set actual prescaler division ratio.
  * @note This function must be called when the LPTIM instance is disabled.
  * @note When the LPTIM is configured to be clocked by an internal clock source
  *       and the LPTIM counter is configured to be updated by active edges
  *       detected on the LPTIM external Input1, the internal clock provided to
  *       the LPTIM must be not be prescaled.
  * @rmtoll CFGR         PRESC         LL_LPTIM_SetPrescaler
  * @param  LPTIMx Low-Power Timer instance
  * @param  Prescaler This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_PRESCALER_DIV1
  *         @arg @ref LL_LPTIM_PRESCALER_DIV2
  *         @arg @ref LL_LPTIM_PRESCALER_DIV4
  *         @arg @ref LL_LPTIM_PRESCALER_DIV8
  *         @arg @ref LL_LPTIM_PRESCALER_DIV16
  *         @arg @ref LL_LPTIM_PRESCALER_DIV32
  *         @arg @ref LL_LPTIM_PRESCALER_DIV64
  *         @arg @ref LL_LPTIM_PRESCALER_DIV128
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetPrescaler(LPTIM_TypeDef *LPTIMx, uint32_t Prescaler)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_PRESC, Prescaler);
}

/**
  * @brief  Get actual prescaler division ratio.
  * @rmtoll CFGR         PRESC         LL_LPTIM_GetPrescaler
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_PRESCALER_DIV1
  *         @arg @ref LL_LPTIM_PRESCALER_DIV2
  *         @arg @ref LL_LPTIM_PRESCALER_DIV4
  *         @arg @ref LL_LPTIM_PRESCALER_DIV8
  *         @arg @ref LL_LPTIM_PRESCALER_DIV16
  *         @arg @ref LL_LPTIM_PRESCALER_DIV32
  *         @arg @ref LL_LPTIM_PRESCALER_DIV64
  *         @arg @ref LL_LPTIM_PRESCALER_DIV128
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetPrescaler(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_PRESC));
}

/**
  * @brief  Set LPTIM input 1 source (default GPIO).
  * @rmtoll OR      OR       LL_LPTIM_SetInput1Src
  * @param  LPTIMx Low-Power Timer instance
  * @param  Src This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_INPUT1_SRC_PAD_AF
  *         @arg @ref LL_LPTIM_INPUT1_SRC_PAD_PA4
  *         @arg @ref LL_LPTIM_INPUT1_SRC_PAD_PB9
  *         @arg @ref LL_LPTIM_INPUT1_SRC_TIM_DAC
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetInput1Src(LPTIM_TypeDef *LPTIMx, uint32_t Src)
{
  MODIFY_REG(LPTIMx->OR, LPTIM_OR_OR, Src);
}

/**
  * @}
  */

/** @defgroup LPTIM_LL_EF_Trigger_Configuration Trigger Configuration
  * @{
  */

/**
  * @brief  Enable the timeout function
  * @note This function must be called when the LPTIM instance is disabled.
  * @note The first trigger event will start the timer, any successive trigger
  *       event will reset the counter and the timer will restart.
  * @note The timeout value corresponds to the compare value; if no trigger
  *       occurs within the expected time frame, the MCU is waked-up by the
  *       compare match event.
  * @rmtoll CFGR         TIMOUT        LL_LPTIM_EnableTimeout
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_EnableTimeout(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->CFGR, LPTIM_CFGR_TIMOUT);
}

/**
  * @brief  Disable the timeout function
  * @note This function must be called when the LPTIM instance is disabled.
  * @note A trigger event arriving when the timer is already started will be
  *       ignored.
  * @rmtoll CFGR         TIMOUT        LL_LPTIM_DisableTimeout
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_DisableTimeout(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->CFGR, LPTIM_CFGR_TIMOUT);
}

/**
  * @brief  Indicate whether the timeout function is enabled.
  * @rmtoll CFGR         TIMOUT        LL_LPTIM_IsEnabledTimeout
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabledTimeout(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_TIMOUT) == LPTIM_CFGR_TIMOUT) ? 1UL : 0UL));
}

/**
  * @brief  Start the LPTIM counter
  * @note This function must be called when the LPTIM instance is disabled.
  * @rmtoll CFGR         TRIGEN        LL_LPTIM_TrigSw
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_TrigSw(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->CFGR, LPTIM_CFGR_TRIGEN);
}

/**
  * @brief  Configure the external trigger used as a trigger event for the LPTIM.
  * @note This function must be called when the LPTIM instance is disabled.
  * @note An internal clock source must be present when a digital filter is
  *       required for the trigger.
  * @rmtoll CFGR         TRIGSEL       LL_LPTIM_ConfigTrigger\n
  *         CFGR         TRGFLT        LL_LPTIM_ConfigTrigger\n
  *         CFGR         TRIGEN        LL_LPTIM_ConfigTrigger
  * @param  LPTIMx Low-Power Timer instance
  * @param  Source This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_GPIO
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_RTCALARMA
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_RTCALARMB
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_RTCTAMP1
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_TIM1_TRGO
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_TIM5_TRGO
  * @param  Filter This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_TRIG_FILTER_NONE
  *         @arg @ref LL_LPTIM_TRIG_FILTER_2
  *         @arg @ref LL_LPTIM_TRIG_FILTER_4
  *         @arg @ref LL_LPTIM_TRIG_FILTER_8
  * @param  Polarity This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_TRIG_POLARITY_RISING
  *         @arg @ref LL_LPTIM_TRIG_POLARITY_FALLING
  *         @arg @ref LL_LPTIM_TRIG_POLARITY_RISING_FALLING
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ConfigTrigger(LPTIM_TypeDef *LPTIMx, uint32_t Source, uint32_t Filter, uint32_t Polarity)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_TRIGSEL | LPTIM_CFGR_TRGFLT | LPTIM_CFGR_TRIGEN, Source | Filter | Polarity);
}

/**
  * @brief  Get actual external trigger source.
  * @rmtoll CFGR         TRIGSEL       LL_LPTIM_GetTriggerSource
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_GPIO
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_RTCALARMA
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_RTCALARMB
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_RTCTAMP1
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_TIM1_TRGO
  *         @arg @ref LL_LPTIM_TRIG_SOURCE_TIM5_TRGO
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetTriggerSource(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_TRIGSEL));
}

/**
  * @brief  Get actual external trigger filter.
  * @rmtoll CFGR         TRGFLT        LL_LPTIM_GetTriggerFilter
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_TRIG_FILTER_NONE
  *         @arg @ref LL_LPTIM_TRIG_FILTER_2
  *         @arg @ref LL_LPTIM_TRIG_FILTER_4
  *         @arg @ref LL_LPTIM_TRIG_FILTER_8
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetTriggerFilter(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_TRGFLT));
}

/**
  * @brief  Get actual external trigger polarity.
  * @rmtoll CFGR         TRIGEN        LL_LPTIM_GetTriggerPolarity
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_TRIG_POLARITY_RISING
  *         @arg @ref LL_LPTIM_TRIG_POLARITY_FALLING
  *         @arg @ref LL_LPTIM_TRIG_POLARITY_RISING_FALLING
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetTriggerPolarity(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_TRIGEN));
}

/**
  * @}
  */

/** @defgroup LPTIM_LL_EF_Clock_Configuration Clock Configuration
  * @{
  */

/**
  * @brief  Set the source of the clock used by the LPTIM instance.
  * @note This function must be called when the LPTIM instance is disabled.
  * @rmtoll CFGR         CKSEL         LL_LPTIM_SetClockSource
  * @param  LPTIMx Low-Power Timer instance
  * @param  ClockSource This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_CLK_SOURCE_INTERNAL
  *         @arg @ref LL_LPTIM_CLK_SOURCE_EXTERNAL
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetClockSource(LPTIM_TypeDef *LPTIMx, uint32_t ClockSource)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_CKSEL, ClockSource);
}

/**
  * @brief  Get actual LPTIM instance clock source.
  * @rmtoll CFGR         CKSEL         LL_LPTIM_GetClockSource
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_CLK_SOURCE_INTERNAL
  *         @arg @ref LL_LPTIM_CLK_SOURCE_EXTERNAL
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetClockSource(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_CKSEL));
}

/**
  * @brief  Configure the active edge or edges used by the counter when
            the LPTIM is clocked by an external clock source.
  * @note This function must be called when the LPTIM instance is disabled.
  * @note When both external clock signal edges are considered active ones,
  *       the LPTIM must also be clocked by an internal clock source with a
  *       frequency equal to at least four times the external clock frequency.
  * @note An internal clock source must be present when a digital filter is
  *       required for external clock.
  * @rmtoll CFGR         CKFLT         LL_LPTIM_ConfigClock\n
  *         CFGR         CKPOL         LL_LPTIM_ConfigClock
  * @param  LPTIMx Low-Power Timer instance
  * @param  ClockFilter This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_CLK_FILTER_NONE
  *         @arg @ref LL_LPTIM_CLK_FILTER_2
  *         @arg @ref LL_LPTIM_CLK_FILTER_4
  *         @arg @ref LL_LPTIM_CLK_FILTER_8
  * @param  ClockPolarity This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_CLK_POLARITY_RISING
  *         @arg @ref LL_LPTIM_CLK_POLARITY_FALLING
  *         @arg @ref LL_LPTIM_CLK_POLARITY_RISING_FALLING
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ConfigClock(LPTIM_TypeDef *LPTIMx, uint32_t ClockFilter, uint32_t ClockPolarity)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_CKFLT | LPTIM_CFGR_CKPOL, ClockFilter | ClockPolarity);
}

/**
  * @brief  Get actual clock polarity
  * @rmtoll CFGR         CKPOL         LL_LPTIM_GetClockPolarity
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_CLK_POLARITY_RISING
  *         @arg @ref LL_LPTIM_CLK_POLARITY_FALLING
  *         @arg @ref LL_LPTIM_CLK_POLARITY_RISING_FALLING
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetClockPolarity(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_CKPOL));
}

/**
  * @brief  Get actual clock digital filter
  * @rmtoll CFGR         CKFLT         LL_LPTIM_GetClockFilter
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_CLK_FILTER_NONE
  *         @arg @ref LL_LPTIM_CLK_FILTER_2
  *         @arg @ref LL_LPTIM_CLK_FILTER_4
  *         @arg @ref LL_LPTIM_CLK_FILTER_8
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetClockFilter(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_CKFLT));
}

/**
  * @}
  */

/** @defgroup LPTIM_LL_EF_Encoder_Mode Encoder Mode
  * @{
  */

/**
  * @brief  Configure the encoder mode.
  * @note This function must be called when the LPTIM instance is disabled.
  * @rmtoll CFGR         CKPOL         LL_LPTIM_SetEncoderMode
  * @param  LPTIMx Low-Power Timer instance
  * @param  EncoderMode This parameter can be one of the following values:
  *         @arg @ref LL_LPTIM_ENCODER_MODE_RISING
  *         @arg @ref LL_LPTIM_ENCODER_MODE_FALLING
  *         @arg @ref LL_LPTIM_ENCODER_MODE_RISING_FALLING
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_SetEncoderMode(LPTIM_TypeDef *LPTIMx, uint32_t EncoderMode)
{
  MODIFY_REG(LPTIMx->CFGR, LPTIM_CFGR_CKPOL, EncoderMode);
}

/**
  * @brief  Get actual encoder mode.
  * @rmtoll CFGR         CKPOL         LL_LPTIM_GetEncoderMode
  * @param  LPTIMx Low-Power Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_LPTIM_ENCODER_MODE_RISING
  *         @arg @ref LL_LPTIM_ENCODER_MODE_FALLING
  *         @arg @ref LL_LPTIM_ENCODER_MODE_RISING_FALLING
  */
__STATIC_INLINE uint32_t LL_LPTIM_GetEncoderMode(LPTIM_TypeDef *LPTIMx)
{
  return (uint32_t)(READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_CKPOL));
}

/**
  * @brief  Enable the encoder mode
  * @note This function must be called when the LPTIM instance is disabled.
  * @note In this mode the LPTIM instance must be clocked by an internal clock
  *       source. Also, the prescaler division ratio must be equal to 1.
  * @note LPTIM instance must be configured in continuous mode prior enabling
  *       the encoder mode.
  * @rmtoll CFGR         ENC           LL_LPTIM_EnableEncoderMode
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_EnableEncoderMode(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->CFGR, LPTIM_CFGR_ENC);
}

/**
  * @brief  Disable the encoder mode
  * @note This function must be called when the LPTIM instance is disabled.
  * @rmtoll CFGR         ENC           LL_LPTIM_DisableEncoderMode
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_DisableEncoderMode(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->CFGR, LPTIM_CFGR_ENC);
}

/**
  * @brief  Indicates whether the LPTIM operates in encoder mode.
  * @rmtoll CFGR         ENC           LL_LPTIM_IsEnabledEncoderMode
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabledEncoderMode(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->CFGR, LPTIM_CFGR_ENC) == LPTIM_CFGR_ENC) ? 1UL : 0UL));
}

/**
  * @}
  */

/** @defgroup LPTIM_LL_EF_FLAG_Management FLAG Management
  * @{
  */

/**
  * @brief  Clear the compare match flag (CMPMCF)
  * @rmtoll ICR          CMPMCF        LL_LPTIM_ClearFLAG_CMPM
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ClearFLAG_CMPM(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->ICR, LPTIM_ICR_CMPMCF);
}

/**
  * @brief  Inform application whether a compare match interrupt has occurred.
  * @rmtoll ISR          CMPM          LL_LPTIM_IsActiveFlag_CMPM
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsActiveFlag_CMPM(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->ISR, LPTIM_ISR_CMPM) == LPTIM_ISR_CMPM) ? 1UL : 0UL));
}

/**
  * @brief  Clear the autoreload match flag (ARRMCF)
  * @rmtoll ICR          ARRMCF        LL_LPTIM_ClearFLAG_ARRM
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ClearFLAG_ARRM(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->ICR, LPTIM_ICR_ARRMCF);
}

/**
  * @brief  Inform application whether a autoreload match interrupt has occurred.
  * @rmtoll ISR          ARRM          LL_LPTIM_IsActiveFlag_ARRM
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsActiveFlag_ARRM(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->ISR, LPTIM_ISR_ARRM) == LPTIM_ISR_ARRM) ? 1UL : 0UL));
}

/**
  * @brief  Clear the external trigger valid edge flag(EXTTRIGCF).
  * @rmtoll ICR          EXTTRIGCF     LL_LPTIM_ClearFlag_EXTTRIG
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ClearFlag_EXTTRIG(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->ICR, LPTIM_ICR_EXTTRIGCF);
}

/**
  * @brief  Inform application whether a valid edge on the selected external trigger input has occurred.
  * @rmtoll ISR          EXTTRIG       LL_LPTIM_IsActiveFlag_EXTTRIG
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsActiveFlag_EXTTRIG(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->ISR, LPTIM_ISR_EXTTRIG) == LPTIM_ISR_EXTTRIG) ? 1UL : 0UL));
}

/**
  * @brief  Clear the compare register update interrupt flag (CMPOKCF).
  * @rmtoll ICR          CMPOKCF       LL_LPTIM_ClearFlag_CMPOK
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ClearFlag_CMPOK(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->ICR, LPTIM_ICR_CMPOKCF);
}

/**
  * @brief  Informs application whether the APB bus write operation to the LPTIMx_CMP register has been successfully
            completed. If so, a new one can be initiated.
  * @rmtoll ISR          CMPOK         LL_LPTIM_IsActiveFlag_CMPOK
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsActiveFlag_CMPOK(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->ISR, LPTIM_ISR_CMPOK) == LPTIM_ISR_CMPOK) ? 1UL : 0UL));
}

/**
  * @brief  Clear the autoreload register update interrupt flag (ARROKCF).
  * @rmtoll ICR          ARROKCF       LL_LPTIM_ClearFlag_ARROK
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ClearFlag_ARROK(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->ICR, LPTIM_ICR_ARROKCF);
}

/**
  * @brief  Informs application whether the APB bus write operation to the LPTIMx_ARR register has been successfully
            completed. If so, a new one can be initiated.
  * @rmtoll ISR          ARROK         LL_LPTIM_IsActiveFlag_ARROK
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsActiveFlag_ARROK(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->ISR, LPTIM_ISR_ARROK) == LPTIM_ISR_ARROK) ? 1UL : 0UL));
}

/**
  * @brief  Clear the counter direction change to up interrupt flag (UPCF).
  * @rmtoll ICR          UPCF          LL_LPTIM_ClearFlag_UP
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ClearFlag_UP(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->ICR, LPTIM_ICR_UPCF);
}

/**
  * @brief  Informs the application whether the counter direction has changed from down to up (when the LPTIM instance
            operates in encoder mode).
  * @rmtoll ISR          UP            LL_LPTIM_IsActiveFlag_UP
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsActiveFlag_UP(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->ISR, LPTIM_ISR_UP) == LPTIM_ISR_UP) ? 1UL : 0UL));
}

/**
  * @brief  Clear the counter direction change to down interrupt flag (DOWNCF).
  * @rmtoll ICR          DOWNCF        LL_LPTIM_ClearFlag_DOWN
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_ClearFlag_DOWN(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->ICR, LPTIM_ICR_DOWNCF);
}

/**
  * @brief  Informs the application whether the counter direction has changed from up to down (when the LPTIM instance
            operates in encoder mode).
  * @rmtoll ISR          DOWN          LL_LPTIM_IsActiveFlag_DOWN
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsActiveFlag_DOWN(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->ISR, LPTIM_ISR_DOWN) == LPTIM_ISR_DOWN) ? 1UL : 0UL));
}

/**
  * @}
  */

/** @defgroup LPTIM_LL_EF_IT_Management Interrupt Management
  * @{
  */

/**
  * @brief  Enable compare match interrupt (CMPMIE).
  * @rmtoll IER          CMPMIE        LL_LPTIM_EnableIT_CMPM
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_EnableIT_CMPM(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->IER, LPTIM_IER_CMPMIE);
}

/**
  * @brief  Disable compare match interrupt (CMPMIE).
  * @rmtoll IER          CMPMIE        LL_LPTIM_DisableIT_CMPM
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_DisableIT_CMPM(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->IER, LPTIM_IER_CMPMIE);
}

/**
  * @brief  Indicates whether the compare match interrupt (CMPMIE) is enabled.
  * @rmtoll IER          CMPMIE        LL_LPTIM_IsEnabledIT_CMPM
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabledIT_CMPM(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->IER, LPTIM_IER_CMPMIE) == LPTIM_IER_CMPMIE) ? 1UL : 0UL));
}

/**
  * @brief  Enable autoreload match interrupt (ARRMIE).
  * @rmtoll IER          ARRMIE        LL_LPTIM_EnableIT_ARRM
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_EnableIT_ARRM(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->IER, LPTIM_IER_ARRMIE);
}

/**
  * @brief  Disable autoreload match interrupt (ARRMIE).
  * @rmtoll IER          ARRMIE        LL_LPTIM_DisableIT_ARRM
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_DisableIT_ARRM(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->IER, LPTIM_IER_ARRMIE);
}

/**
  * @brief  Indicates whether the autoreload match interrupt (ARRMIE) is enabled.
  * @rmtoll IER          ARRMIE        LL_LPTIM_IsEnabledIT_ARRM
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabledIT_ARRM(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->IER, LPTIM_IER_ARRMIE) == LPTIM_IER_ARRMIE) ? 1UL : 0UL));
}

/**
  * @brief  Enable external trigger valid edge interrupt (EXTTRIGIE).
  * @rmtoll IER          EXTTRIGIE     LL_LPTIM_EnableIT_EXTTRIG
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_EnableIT_EXTTRIG(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->IER, LPTIM_IER_EXTTRIGIE);
}

/**
  * @brief  Disable external trigger valid edge interrupt (EXTTRIGIE).
  * @rmtoll IER          EXTTRIGIE     LL_LPTIM_DisableIT_EXTTRIG
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_DisableIT_EXTTRIG(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->IER, LPTIM_IER_EXTTRIGIE);
}

/**
  * @brief  Indicates external trigger valid edge interrupt (EXTTRIGIE) is enabled.
  * @rmtoll IER          EXTTRIGIE     LL_LPTIM_IsEnabledIT_EXTTRIG
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabledIT_EXTTRIG(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->IER, LPTIM_IER_EXTTRIGIE) == LPTIM_IER_EXTTRIGIE) ? 1UL : 0UL));
}

/**
  * @brief  Enable compare register write completed interrupt (CMPOKIE).
  * @rmtoll IER          CMPOKIE       LL_LPTIM_EnableIT_CMPOK
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_EnableIT_CMPOK(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->IER, LPTIM_IER_CMPOKIE);
}

/**
  * @brief  Disable compare register write completed interrupt (CMPOKIE).
  * @rmtoll IER          CMPOKIE       LL_LPTIM_DisableIT_CMPOK
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_DisableIT_CMPOK(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->IER, LPTIM_IER_CMPOKIE);
}

/**
  * @brief  Indicates whether the compare register write completed interrupt (CMPOKIE) is enabled.
  * @rmtoll IER          CMPOKIE       LL_LPTIM_IsEnabledIT_CMPOK
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabledIT_CMPOK(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->IER, LPTIM_IER_CMPOKIE) == LPTIM_IER_CMPOKIE) ? 1UL : 0UL));
}

/**
  * @brief  Enable autoreload register write completed interrupt (ARROKIE).
  * @rmtoll IER         ARROKIE       LL_LPTIM_EnableIT_ARROK
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_EnableIT_ARROK(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->IER, LPTIM_IER_ARROKIE);
}

/**
  * @brief  Disable autoreload register write completed interrupt (ARROKIE).
  * @rmtoll IER         ARROKIE       LL_LPTIM_DisableIT_ARROK
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_DisableIT_ARROK(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->IER, LPTIM_IER_ARROKIE);
}

/**
  * @brief  Indicates whether the autoreload register write completed interrupt (ARROKIE) is enabled.
  * @rmtoll IER         ARROKIE       LL_LPTIM_IsEnabledIT_ARROK
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit(1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabledIT_ARROK(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->IER, LPTIM_IER_ARROKIE) == LPTIM_IER_ARROKIE) ? 1UL : 0UL));
}

/**
  * @brief  Enable direction change to up interrupt (UPIE).
  * @rmtoll IER         UPIE          LL_LPTIM_EnableIT_UP
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_EnableIT_UP(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->IER, LPTIM_IER_UPIE);
}

/**
  * @brief  Disable direction change to up interrupt (UPIE).
  * @rmtoll IER         UPIE          LL_LPTIM_DisableIT_UP
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_DisableIT_UP(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->IER, LPTIM_IER_UPIE);
}

/**
  * @brief  Indicates whether the direction change to up interrupt (UPIE) is enabled.
  * @rmtoll IER         UPIE          LL_LPTIM_IsEnabledIT_UP
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit(1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabledIT_UP(LPTIM_TypeDef *LPTIMx)
{
  return (((READ_BIT(LPTIMx->IER, LPTIM_IER_UPIE) == LPTIM_IER_UPIE) ? 1UL : 0UL));
}

/**
  * @brief  Enable direction change to down interrupt (DOWNIE).
  * @rmtoll IER         DOWNIE        LL_LPTIM_EnableIT_DOWN
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_EnableIT_DOWN(LPTIM_TypeDef *LPTIMx)
{
  SET_BIT(LPTIMx->IER, LPTIM_IER_DOWNIE);
}

/**
  * @brief  Disable direction change to down interrupt (DOWNIE).
  * @rmtoll IER         DOWNIE        LL_LPTIM_DisableIT_DOWN
  * @param  LPTIMx Low-Power Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_LPTIM_DisableIT_DOWN(LPTIM_TypeDef *LPTIMx)
{
  CLEAR_BIT(LPTIMx->IER, LPTIM_IER_DOWNIE);
}

/**
  * @brief  Indicates whether the direction change to down interrupt (DOWNIE) is enabled.
  * @rmtoll IER         DOWNIE        LL_LPTIM_IsEnabledIT_DOWN
  * @param  LPTIMx Low-Power Timer instance
  * @retval State of bit(1 or 0).
  */
__STATIC_INLINE uint32_t LL_LPTIM_IsEnabledIT_DOWN(LPTIM_TypeDef *LPTIMx)
{
  return ((READ_BIT(LPTIMx->IER, LPTIM_IER_DOWNIE) == LPTIM_IER_DOWNIE) ? 1UL : 0UL);
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

#endif /* LPTIM1 */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_LL_LPTIM_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
