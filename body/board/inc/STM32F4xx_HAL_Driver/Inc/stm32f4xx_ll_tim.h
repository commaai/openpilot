/**
  ******************************************************************************
  * @file    stm32f4xx_ll_tim.h
  * @author  MCD Application Team
  * @brief   Header file of TIM LL module.
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
#ifndef __STM32F4xx_LL_TIM_H
#define __STM32F4xx_LL_TIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (TIM1) || defined (TIM2) || defined (TIM3) || defined (TIM4) || defined (TIM5) || defined (TIM6) || defined (TIM7) || defined (TIM8) || defined (TIM9) || defined (TIM10) || defined (TIM11) || defined (TIM12) || defined (TIM13) || defined (TIM14)

/** @defgroup TIM_LL TIM
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/** @defgroup TIM_LL_Private_Variables TIM Private Variables
  * @{
  */
static const uint8_t OFFSET_TAB_CCMRx[] =
{
  0x00U,   /* 0: TIMx_CH1  */
  0x00U,   /* 1: TIMx_CH1N */
  0x00U,   /* 2: TIMx_CH2  */
  0x00U,   /* 3: TIMx_CH2N */
  0x04U,   /* 4: TIMx_CH3  */
  0x04U,   /* 5: TIMx_CH3N */
  0x04U    /* 6: TIMx_CH4  */
};

static const uint8_t SHIFT_TAB_OCxx[] =
{
  0U,            /* 0: OC1M, OC1FE, OC1PE */
  0U,            /* 1: - NA */
  8U,            /* 2: OC2M, OC2FE, OC2PE */
  0U,            /* 3: - NA */
  0U,            /* 4: OC3M, OC3FE, OC3PE */
  0U,            /* 5: - NA */
  8U             /* 6: OC4M, OC4FE, OC4PE */
};

static const uint8_t SHIFT_TAB_ICxx[] =
{
  0U,            /* 0: CC1S, IC1PSC, IC1F */
  0U,            /* 1: - NA */
  8U,            /* 2: CC2S, IC2PSC, IC2F */
  0U,            /* 3: - NA */
  0U,            /* 4: CC3S, IC3PSC, IC3F */
  0U,            /* 5: - NA */
  8U             /* 6: CC4S, IC4PSC, IC4F */
};

static const uint8_t SHIFT_TAB_CCxP[] =
{
  0U,            /* 0: CC1P */
  2U,            /* 1: CC1NP */
  4U,            /* 2: CC2P */
  6U,            /* 3: CC2NP */
  8U,            /* 4: CC3P */
  10U,           /* 5: CC3NP */
  12U            /* 6: CC4P */
};

static const uint8_t SHIFT_TAB_OISx[] =
{
  0U,            /* 0: OIS1 */
  1U,            /* 1: OIS1N */
  2U,            /* 2: OIS2 */
  3U,            /* 3: OIS2N */
  4U,            /* 4: OIS3 */
  5U,            /* 5: OIS3N */
  6U             /* 6: OIS4 */
};
/**
  * @}
  */

/* Private constants ---------------------------------------------------------*/
/** @defgroup TIM_LL_Private_Constants TIM Private Constants
  * @{
  */


/* Remap mask definitions */
#define TIMx_OR_RMP_SHIFT  16U
#define TIMx_OR_RMP_MASK   0x0000FFFFU
#define TIM2_OR_RMP_MASK   (TIM_OR_ITR1_RMP << TIMx_OR_RMP_SHIFT)
#define TIM5_OR_RMP_MASK   (TIM_OR_TI4_RMP << TIMx_OR_RMP_SHIFT)
#define TIM11_OR_RMP_MASK  (TIM_OR_TI1_RMP << TIMx_OR_RMP_SHIFT)

/* Mask used to set the TDG[x:0] of the DTG bits of the TIMx_BDTR register */
#define DT_DELAY_1 ((uint8_t)0x7F)
#define DT_DELAY_2 ((uint8_t)0x3F)
#define DT_DELAY_3 ((uint8_t)0x1F)
#define DT_DELAY_4 ((uint8_t)0x1F)

/* Mask used to set the DTG[7:5] bits of the DTG bits of the TIMx_BDTR register */
#define DT_RANGE_1 ((uint8_t)0x00)
#define DT_RANGE_2 ((uint8_t)0x80)
#define DT_RANGE_3 ((uint8_t)0xC0)
#define DT_RANGE_4 ((uint8_t)0xE0)


/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup TIM_LL_Private_Macros TIM Private Macros
  * @{
  */
/** @brief  Convert channel id into channel index.
  * @param  __CHANNEL__ This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH1N
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH2N
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH3N
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval none
  */
#define TIM_GET_CHANNEL_INDEX( __CHANNEL__) \
  (((__CHANNEL__) == LL_TIM_CHANNEL_CH1) ? 0U :\
   ((__CHANNEL__) == LL_TIM_CHANNEL_CH1N) ? 1U :\
   ((__CHANNEL__) == LL_TIM_CHANNEL_CH2) ? 2U :\
   ((__CHANNEL__) == LL_TIM_CHANNEL_CH2N) ? 3U :\
   ((__CHANNEL__) == LL_TIM_CHANNEL_CH3) ? 4U :\
   ((__CHANNEL__) == LL_TIM_CHANNEL_CH3N) ? 5U : 6U)

/** @brief  Calculate the deadtime sampling period(in ps).
  * @param  __TIMCLK__ timer input clock frequency (in Hz).
  * @param  __CKD__ This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV1
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV2
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV4
  * @retval none
  */
#define TIM_CALC_DTS(__TIMCLK__, __CKD__)                                                        \
  (((__CKD__) == LL_TIM_CLOCKDIVISION_DIV1) ? ((uint64_t)1000000000000U/(__TIMCLK__))         : \
   ((__CKD__) == LL_TIM_CLOCKDIVISION_DIV2) ? ((uint64_t)1000000000000U/((__TIMCLK__) >> 1U)) : \
   ((uint64_t)1000000000000U/((__TIMCLK__) >> 2U)))
/**
  * @}
  */


/* Exported types ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup TIM_LL_ES_INIT TIM Exported Init structure
  * @{
  */

/**
  * @brief  TIM Time Base configuration structure definition.
  */
typedef struct
{
  uint16_t Prescaler;         /*!< Specifies the prescaler value used to divide the TIM clock.
                                   This parameter can be a number between Min_Data=0x0000 and Max_Data=0xFFFF.

                                   This feature can be modified afterwards using unitary function
                                   @ref LL_TIM_SetPrescaler().*/

  uint32_t CounterMode;       /*!< Specifies the counter mode.
                                   This parameter can be a value of @ref TIM_LL_EC_COUNTERMODE.

                                   This feature can be modified afterwards using unitary function
                                   @ref LL_TIM_SetCounterMode().*/

  uint32_t Autoreload;        /*!< Specifies the auto reload value to be loaded into the active
                                   Auto-Reload Register at the next update event.
                                   This parameter must be a number between Min_Data=0x0000 and Max_Data=0xFFFF.
                                   Some timer instances may support 32 bits counters. In that case this parameter must
                                   be a number between 0x0000 and 0xFFFFFFFF.

                                   This feature can be modified afterwards using unitary function
                                   @ref LL_TIM_SetAutoReload().*/

  uint32_t ClockDivision;     /*!< Specifies the clock division.
                                   This parameter can be a value of @ref TIM_LL_EC_CLOCKDIVISION.

                                   This feature can be modified afterwards using unitary function
                                   @ref LL_TIM_SetClockDivision().*/

  uint32_t RepetitionCounter;  /*!< Specifies the repetition counter value. Each time the RCR downcounter
                                   reaches zero, an update event is generated and counting restarts
                                   from the RCR value (N).
                                   This means in PWM mode that (N+1) corresponds to:
                                      - the number of PWM periods in edge-aligned mode
                                      - the number of half PWM period in center-aligned mode
                                   GP timers: this parameter must be a number between Min_Data = 0x00 and
                                   Max_Data = 0xFF.
                                   Advanced timers: this parameter must be a number between Min_Data = 0x0000 and
                                   Max_Data = 0xFFFF.

                                   This feature can be modified afterwards using unitary function
                                   @ref LL_TIM_SetRepetitionCounter().*/
} LL_TIM_InitTypeDef;

/**
  * @brief  TIM Output Compare configuration structure definition.
  */
typedef struct
{
  uint32_t OCMode;        /*!< Specifies the output mode.
                               This parameter can be a value of @ref TIM_LL_EC_OCMODE.

                               This feature can be modified afterwards using unitary function
                               @ref LL_TIM_OC_SetMode().*/

  uint32_t OCState;       /*!< Specifies the TIM Output Compare state.
                               This parameter can be a value of @ref TIM_LL_EC_OCSTATE.

                               This feature can be modified afterwards using unitary functions
                               @ref LL_TIM_CC_EnableChannel() or @ref LL_TIM_CC_DisableChannel().*/

  uint32_t OCNState;      /*!< Specifies the TIM complementary Output Compare state.
                               This parameter can be a value of @ref TIM_LL_EC_OCSTATE.

                               This feature can be modified afterwards using unitary functions
                               @ref LL_TIM_CC_EnableChannel() or @ref LL_TIM_CC_DisableChannel().*/

  uint32_t CompareValue;  /*!< Specifies the Compare value to be loaded into the Capture Compare Register.
                               This parameter can be a number between Min_Data=0x0000 and Max_Data=0xFFFF.

                               This feature can be modified afterwards using unitary function
                               LL_TIM_OC_SetCompareCHx (x=1..6).*/

  uint32_t OCPolarity;    /*!< Specifies the output polarity.
                               This parameter can be a value of @ref TIM_LL_EC_OCPOLARITY.

                               This feature can be modified afterwards using unitary function
                               @ref LL_TIM_OC_SetPolarity().*/

  uint32_t OCNPolarity;   /*!< Specifies the complementary output polarity.
                               This parameter can be a value of @ref TIM_LL_EC_OCPOLARITY.

                               This feature can be modified afterwards using unitary function
                               @ref LL_TIM_OC_SetPolarity().*/


  uint32_t OCIdleState;   /*!< Specifies the TIM Output Compare pin state during Idle state.
                               This parameter can be a value of @ref TIM_LL_EC_OCIDLESTATE.

                               This feature can be modified afterwards using unitary function
                               @ref LL_TIM_OC_SetIdleState().*/

  uint32_t OCNIdleState;  /*!< Specifies the TIM Output Compare pin state during Idle state.
                               This parameter can be a value of @ref TIM_LL_EC_OCIDLESTATE.

                               This feature can be modified afterwards using unitary function
                               @ref LL_TIM_OC_SetIdleState().*/
} LL_TIM_OC_InitTypeDef;

/**
  * @brief  TIM Input Capture configuration structure definition.
  */

typedef struct
{

  uint32_t ICPolarity;    /*!< Specifies the active edge of the input signal.
                               This parameter can be a value of @ref TIM_LL_EC_IC_POLARITY.

                               This feature can be modified afterwards using unitary function
                               @ref LL_TIM_IC_SetPolarity().*/

  uint32_t ICActiveInput; /*!< Specifies the input.
                               This parameter can be a value of @ref TIM_LL_EC_ACTIVEINPUT.

                               This feature can be modified afterwards using unitary function
                               @ref LL_TIM_IC_SetActiveInput().*/

  uint32_t ICPrescaler;   /*!< Specifies the Input Capture Prescaler.
                               This parameter can be a value of @ref TIM_LL_EC_ICPSC.

                               This feature can be modified afterwards using unitary function
                               @ref LL_TIM_IC_SetPrescaler().*/

  uint32_t ICFilter;      /*!< Specifies the input capture filter.
                               This parameter can be a value of @ref TIM_LL_EC_IC_FILTER.

                               This feature can be modified afterwards using unitary function
                               @ref LL_TIM_IC_SetFilter().*/
} LL_TIM_IC_InitTypeDef;


/**
  * @brief  TIM Encoder interface configuration structure definition.
  */
typedef struct
{
  uint32_t EncoderMode;     /*!< Specifies the encoder resolution (x2 or x4).
                                 This parameter can be a value of @ref TIM_LL_EC_ENCODERMODE.

                                 This feature can be modified afterwards using unitary function
                                 @ref LL_TIM_SetEncoderMode().*/

  uint32_t IC1Polarity;     /*!< Specifies the active edge of TI1 input.
                                 This parameter can be a value of @ref TIM_LL_EC_IC_POLARITY.

                                 This feature can be modified afterwards using unitary function
                                 @ref LL_TIM_IC_SetPolarity().*/

  uint32_t IC1ActiveInput;  /*!< Specifies the TI1 input source
                                 This parameter can be a value of @ref TIM_LL_EC_ACTIVEINPUT.

                                 This feature can be modified afterwards using unitary function
                                 @ref LL_TIM_IC_SetActiveInput().*/

  uint32_t IC1Prescaler;    /*!< Specifies the TI1 input prescaler value.
                                 This parameter can be a value of @ref TIM_LL_EC_ICPSC.

                                 This feature can be modified afterwards using unitary function
                                 @ref LL_TIM_IC_SetPrescaler().*/

  uint32_t IC1Filter;       /*!< Specifies the TI1 input filter.
                                 This parameter can be a value of @ref TIM_LL_EC_IC_FILTER.

                                 This feature can be modified afterwards using unitary function
                                 @ref LL_TIM_IC_SetFilter().*/

  uint32_t IC2Polarity;      /*!< Specifies the active edge of TI2 input.
                                 This parameter can be a value of @ref TIM_LL_EC_IC_POLARITY.

                                 This feature can be modified afterwards using unitary function
                                 @ref LL_TIM_IC_SetPolarity().*/

  uint32_t IC2ActiveInput;  /*!< Specifies the TI2 input source
                                 This parameter can be a value of @ref TIM_LL_EC_ACTIVEINPUT.

                                 This feature can be modified afterwards using unitary function
                                 @ref LL_TIM_IC_SetActiveInput().*/

  uint32_t IC2Prescaler;    /*!< Specifies the TI2 input prescaler value.
                                 This parameter can be a value of @ref TIM_LL_EC_ICPSC.

                                 This feature can be modified afterwards using unitary function
                                 @ref LL_TIM_IC_SetPrescaler().*/

  uint32_t IC2Filter;       /*!< Specifies the TI2 input filter.
                                 This parameter can be a value of @ref TIM_LL_EC_IC_FILTER.

                                 This feature can be modified afterwards using unitary function
                                 @ref LL_TIM_IC_SetFilter().*/

} LL_TIM_ENCODER_InitTypeDef;

/**
  * @brief  TIM Hall sensor interface configuration structure definition.
  */
typedef struct
{

  uint32_t IC1Polarity;        /*!< Specifies the active edge of TI1 input.
                                    This parameter can be a value of @ref TIM_LL_EC_IC_POLARITY.

                                    This feature can be modified afterwards using unitary function
                                    @ref LL_TIM_IC_SetPolarity().*/

  uint32_t IC1Prescaler;       /*!< Specifies the TI1 input prescaler value.
                                    Prescaler must be set to get a maximum counter period longer than the
                                    time interval between 2 consecutive changes on the Hall inputs.
                                    This parameter can be a value of @ref TIM_LL_EC_ICPSC.

                                    This feature can be modified afterwards using unitary function
                                    @ref LL_TIM_IC_SetPrescaler().*/

  uint32_t IC1Filter;          /*!< Specifies the TI1 input filter.
                                    This parameter can be a value of
                                    @ref TIM_LL_EC_IC_FILTER.

                                    This feature can be modified afterwards using unitary function
                                    @ref LL_TIM_IC_SetFilter().*/

  uint32_t CommutationDelay;   /*!< Specifies the compare value to be loaded into the Capture Compare Register.
                                    A positive pulse (TRGO event) is generated with a programmable delay every time
                                    a change occurs on the Hall inputs.
                                    This parameter can be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF.

                                    This feature can be modified afterwards using unitary function
                                    @ref LL_TIM_OC_SetCompareCH2().*/
} LL_TIM_HALLSENSOR_InitTypeDef;

/**
  * @brief  BDTR (Break and Dead Time) structure definition
  */
typedef struct
{
  uint32_t OSSRState;            /*!< Specifies the Off-State selection used in Run mode.
                                      This parameter can be a value of @ref TIM_LL_EC_OSSR

                                      This feature can be modified afterwards using unitary function
                                      @ref LL_TIM_SetOffStates()

                                      @note This bit-field cannot be modified as long as LOCK level 2 has been
                                       programmed. */

  uint32_t OSSIState;            /*!< Specifies the Off-State used in Idle state.
                                      This parameter can be a value of @ref TIM_LL_EC_OSSI

                                      This feature can be modified afterwards using unitary function
                                      @ref LL_TIM_SetOffStates()

                                      @note This bit-field cannot be modified as long as LOCK level 2 has been
                                      programmed. */

  uint32_t LockLevel;            /*!< Specifies the LOCK level parameters.
                                      This parameter can be a value of @ref TIM_LL_EC_LOCKLEVEL

                                      @note The LOCK bits can be written only once after the reset. Once the TIMx_BDTR
                                      register has been written, their content is frozen until the next reset.*/

  uint8_t DeadTime;              /*!< Specifies the delay time between the switching-off and the
                                      switching-on of the outputs.
                                      This parameter can be a number between Min_Data = 0x00 and Max_Data = 0xFF.

                                      This feature can be modified afterwards using unitary function
                                      @ref LL_TIM_OC_SetDeadTime()

                                      @note This bit-field can not be modified as long as LOCK level 1, 2 or 3 has been
                                       programmed. */

  uint16_t BreakState;           /*!< Specifies whether the TIM Break input is enabled or not.
                                      This parameter can be a value of @ref TIM_LL_EC_BREAK_ENABLE

                                      This feature can be modified afterwards using unitary functions
                                      @ref LL_TIM_EnableBRK() or @ref LL_TIM_DisableBRK()

                                      @note This bit-field can not be modified as long as LOCK level 1 has been
                                      programmed. */

  uint32_t BreakPolarity;        /*!< Specifies the TIM Break Input pin polarity.
                                      This parameter can be a value of @ref TIM_LL_EC_BREAK_POLARITY

                                      This feature can be modified afterwards using unitary function
                                      @ref LL_TIM_ConfigBRK()

                                      @note This bit-field can not be modified as long as LOCK level 1 has been
                                      programmed. */

  uint32_t AutomaticOutput;      /*!< Specifies whether the TIM Automatic Output feature is enabled or not.
                                      This parameter can be a value of @ref TIM_LL_EC_AUTOMATICOUTPUT_ENABLE

                                      This feature can be modified afterwards using unitary functions
                                      @ref LL_TIM_EnableAutomaticOutput() or @ref LL_TIM_DisableAutomaticOutput()

                                      @note This bit-field can not be modified as long as LOCK level 1 has been
                                      programmed. */
} LL_TIM_BDTR_InitTypeDef;

/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/* Exported constants --------------------------------------------------------*/
/** @defgroup TIM_LL_Exported_Constants TIM Exported Constants
  * @{
  */

/** @defgroup TIM_LL_EC_GET_FLAG Get Flags Defines
  * @brief    Flags defines which can be used with LL_TIM_ReadReg function.
  * @{
  */
#define LL_TIM_SR_UIF                          TIM_SR_UIF           /*!< Update interrupt flag */
#define LL_TIM_SR_CC1IF                        TIM_SR_CC1IF         /*!< Capture/compare 1 interrupt flag */
#define LL_TIM_SR_CC2IF                        TIM_SR_CC2IF         /*!< Capture/compare 2 interrupt flag */
#define LL_TIM_SR_CC3IF                        TIM_SR_CC3IF         /*!< Capture/compare 3 interrupt flag */
#define LL_TIM_SR_CC4IF                        TIM_SR_CC4IF         /*!< Capture/compare 4 interrupt flag */
#define LL_TIM_SR_COMIF                        TIM_SR_COMIF         /*!< COM interrupt flag */
#define LL_TIM_SR_TIF                          TIM_SR_TIF           /*!< Trigger interrupt flag */
#define LL_TIM_SR_BIF                          TIM_SR_BIF           /*!< Break interrupt flag */
#define LL_TIM_SR_CC1OF                        TIM_SR_CC1OF         /*!< Capture/Compare 1 overcapture flag */
#define LL_TIM_SR_CC2OF                        TIM_SR_CC2OF         /*!< Capture/Compare 2 overcapture flag */
#define LL_TIM_SR_CC3OF                        TIM_SR_CC3OF         /*!< Capture/Compare 3 overcapture flag */
#define LL_TIM_SR_CC4OF                        TIM_SR_CC4OF         /*!< Capture/Compare 4 overcapture flag */
/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup TIM_LL_EC_BREAK_ENABLE Break Enable
  * @{
  */
#define LL_TIM_BREAK_DISABLE            0x00000000U             /*!< Break function disabled */
#define LL_TIM_BREAK_ENABLE             TIM_BDTR_BKE            /*!< Break function enabled */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_AUTOMATICOUTPUT_ENABLE Automatic output enable
  * @{
  */
#define LL_TIM_AUTOMATICOUTPUT_DISABLE         0x00000000U             /*!< MOE can be set only by software */
#define LL_TIM_AUTOMATICOUTPUT_ENABLE          TIM_BDTR_AOE            /*!< MOE can be set by software or automatically at the next update event */
/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/** @defgroup TIM_LL_EC_IT IT Defines
  * @brief    IT defines which can be used with LL_TIM_ReadReg and  LL_TIM_WriteReg functions.
  * @{
  */
#define LL_TIM_DIER_UIE                        TIM_DIER_UIE         /*!< Update interrupt enable */
#define LL_TIM_DIER_CC1IE                      TIM_DIER_CC1IE       /*!< Capture/compare 1 interrupt enable */
#define LL_TIM_DIER_CC2IE                      TIM_DIER_CC2IE       /*!< Capture/compare 2 interrupt enable */
#define LL_TIM_DIER_CC3IE                      TIM_DIER_CC3IE       /*!< Capture/compare 3 interrupt enable */
#define LL_TIM_DIER_CC4IE                      TIM_DIER_CC4IE       /*!< Capture/compare 4 interrupt enable */
#define LL_TIM_DIER_COMIE                      TIM_DIER_COMIE       /*!< COM interrupt enable */
#define LL_TIM_DIER_TIE                        TIM_DIER_TIE         /*!< Trigger interrupt enable */
#define LL_TIM_DIER_BIE                        TIM_DIER_BIE         /*!< Break interrupt enable */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_UPDATESOURCE Update Source
  * @{
  */
#define LL_TIM_UPDATESOURCE_REGULAR            0x00000000U          /*!< Counter overflow/underflow, Setting the UG bit or Update generation through the slave mode controller generates an update request */
#define LL_TIM_UPDATESOURCE_COUNTER            TIM_CR1_URS          /*!< Only counter overflow/underflow generates an update request */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_ONEPULSEMODE One Pulse Mode
  * @{
  */
#define LL_TIM_ONEPULSEMODE_SINGLE             TIM_CR1_OPM          /*!< Counter stops counting at the next update event */
#define LL_TIM_ONEPULSEMODE_REPETITIVE         0x00000000U          /*!< Counter is not stopped at update event */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_COUNTERMODE Counter Mode
  * @{
  */
#define LL_TIM_COUNTERMODE_UP                  0x00000000U          /*!<Counter used as upcounter */
#define LL_TIM_COUNTERMODE_DOWN                TIM_CR1_DIR          /*!< Counter used as downcounter */
#define LL_TIM_COUNTERMODE_CENTER_DOWN         TIM_CR1_CMS_0        /*!< The counter counts up and down alternatively. Output compare interrupt flags of output channels  are set only when the counter is counting down. */
#define LL_TIM_COUNTERMODE_CENTER_UP           TIM_CR1_CMS_1        /*!<The counter counts up and down alternatively. Output compare interrupt flags of output channels  are set only when the counter is counting up */
#define LL_TIM_COUNTERMODE_CENTER_UP_DOWN      TIM_CR1_CMS          /*!< The counter counts up and down alternatively. Output compare interrupt flags of output channels  are set only when the counter is counting up or down. */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_CLOCKDIVISION Clock Division
  * @{
  */
#define LL_TIM_CLOCKDIVISION_DIV1              0x00000000U          /*!< tDTS=tCK_INT */
#define LL_TIM_CLOCKDIVISION_DIV2              TIM_CR1_CKD_0        /*!< tDTS=2*tCK_INT */
#define LL_TIM_CLOCKDIVISION_DIV4              TIM_CR1_CKD_1        /*!< tDTS=4*tCK_INT */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_COUNTERDIRECTION Counter Direction
  * @{
  */
#define LL_TIM_COUNTERDIRECTION_UP             0x00000000U          /*!< Timer counter counts up */
#define LL_TIM_COUNTERDIRECTION_DOWN           TIM_CR1_DIR          /*!< Timer counter counts down */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_CCUPDATESOURCE Capture Compare  Update Source
  * @{
  */
#define LL_TIM_CCUPDATESOURCE_COMG_ONLY        0x00000000U          /*!< Capture/compare control bits are updated by setting the COMG bit only */
#define LL_TIM_CCUPDATESOURCE_COMG_AND_TRGI    TIM_CR2_CCUS         /*!< Capture/compare control bits are updated by setting the COMG bit or when a rising edge occurs on trigger input (TRGI) */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_CCDMAREQUEST Capture Compare DMA Request
  * @{
  */
#define LL_TIM_CCDMAREQUEST_CC                 0x00000000U          /*!< CCx DMA request sent when CCx event occurs */
#define LL_TIM_CCDMAREQUEST_UPDATE             TIM_CR2_CCDS         /*!< CCx DMA requests sent when update event occurs */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_LOCKLEVEL Lock Level
  * @{
  */
#define LL_TIM_LOCKLEVEL_OFF                   0x00000000U          /*!< LOCK OFF - No bit is write protected */
#define LL_TIM_LOCKLEVEL_1                     TIM_BDTR_LOCK_0      /*!< LOCK Level 1 */
#define LL_TIM_LOCKLEVEL_2                     TIM_BDTR_LOCK_1      /*!< LOCK Level 2 */
#define LL_TIM_LOCKLEVEL_3                     TIM_BDTR_LOCK        /*!< LOCK Level 3 */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_CHANNEL Channel
  * @{
  */
#define LL_TIM_CHANNEL_CH1                     TIM_CCER_CC1E     /*!< Timer input/output channel 1 */
#define LL_TIM_CHANNEL_CH1N                    TIM_CCER_CC1NE    /*!< Timer complementary output channel 1 */
#define LL_TIM_CHANNEL_CH2                     TIM_CCER_CC2E     /*!< Timer input/output channel 2 */
#define LL_TIM_CHANNEL_CH2N                    TIM_CCER_CC2NE    /*!< Timer complementary output channel 2 */
#define LL_TIM_CHANNEL_CH3                     TIM_CCER_CC3E     /*!< Timer input/output channel 3 */
#define LL_TIM_CHANNEL_CH3N                    TIM_CCER_CC3NE    /*!< Timer complementary output channel 3 */
#define LL_TIM_CHANNEL_CH4                     TIM_CCER_CC4E     /*!< Timer input/output channel 4 */
/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup TIM_LL_EC_OCSTATE Output Configuration State
  * @{
  */
#define LL_TIM_OCSTATE_DISABLE                 0x00000000U             /*!< OCx is not active */
#define LL_TIM_OCSTATE_ENABLE                  TIM_CCER_CC1E           /*!< OCx signal is output on the corresponding output pin */
/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/** @defgroup TIM_LL_EC_OCMODE Output Configuration Mode
  * @{
  */
#define LL_TIM_OCMODE_FROZEN                   0x00000000U                                              /*!<The comparison between the output compare register TIMx_CCRy and the counter TIMx_CNT has no effect on the output channel level */
#define LL_TIM_OCMODE_ACTIVE                   TIM_CCMR1_OC1M_0                                         /*!<OCyREF is forced high on compare match*/
#define LL_TIM_OCMODE_INACTIVE                 TIM_CCMR1_OC1M_1                                         /*!<OCyREF is forced low on compare match*/
#define LL_TIM_OCMODE_TOGGLE                   (TIM_CCMR1_OC1M_1 | TIM_CCMR1_OC1M_0)                    /*!<OCyREF toggles on compare match*/
#define LL_TIM_OCMODE_FORCED_INACTIVE          TIM_CCMR1_OC1M_2                                         /*!<OCyREF is forced low*/
#define LL_TIM_OCMODE_FORCED_ACTIVE            (TIM_CCMR1_OC1M_2 | TIM_CCMR1_OC1M_0)                    /*!<OCyREF is forced high*/
#define LL_TIM_OCMODE_PWM1                     (TIM_CCMR1_OC1M_2 | TIM_CCMR1_OC1M_1)                    /*!<In upcounting, channel y is active as long as TIMx_CNT<TIMx_CCRy else inactive.  In downcounting, channel y is inactive as long as TIMx_CNT>TIMx_CCRy else active.*/
#define LL_TIM_OCMODE_PWM2                     (TIM_CCMR1_OC1M_2 | TIM_CCMR1_OC1M_1 | TIM_CCMR1_OC1M_0) /*!<In upcounting, channel y is inactive as long as TIMx_CNT<TIMx_CCRy else active.  In downcounting, channel y is active as long as TIMx_CNT>TIMx_CCRy else inactive*/
/**
  * @}
  */

/** @defgroup TIM_LL_EC_OCPOLARITY Output Configuration Polarity
  * @{
  */
#define LL_TIM_OCPOLARITY_HIGH                 0x00000000U                 /*!< OCxactive high*/
#define LL_TIM_OCPOLARITY_LOW                  TIM_CCER_CC1P               /*!< OCxactive low*/
/**
  * @}
  */

/** @defgroup TIM_LL_EC_OCIDLESTATE Output Configuration Idle State
  * @{
  */
#define LL_TIM_OCIDLESTATE_LOW                 0x00000000U             /*!<OCx=0 (after a dead-time if OC is implemented) when MOE=0*/
#define LL_TIM_OCIDLESTATE_HIGH                TIM_CR2_OIS1            /*!<OCx=1 (after a dead-time if OC is implemented) when MOE=0*/
/**
  * @}
  */


/** @defgroup TIM_LL_EC_ACTIVEINPUT Active Input Selection
  * @{
  */
#define LL_TIM_ACTIVEINPUT_DIRECTTI            (TIM_CCMR1_CC1S_0 << 16U) /*!< ICx is mapped on TIx */
#define LL_TIM_ACTIVEINPUT_INDIRECTTI          (TIM_CCMR1_CC1S_1 << 16U) /*!< ICx is mapped on TIy */
#define LL_TIM_ACTIVEINPUT_TRC                 (TIM_CCMR1_CC1S << 16U)   /*!< ICx is mapped on TRC */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_ICPSC Input Configuration Prescaler
  * @{
  */
#define LL_TIM_ICPSC_DIV1                      0x00000000U                    /*!< No prescaler, capture is done each time an edge is detected on the capture input */
#define LL_TIM_ICPSC_DIV2                      (TIM_CCMR1_IC1PSC_0 << 16U)    /*!< Capture is done once every 2 events */
#define LL_TIM_ICPSC_DIV4                      (TIM_CCMR1_IC1PSC_1 << 16U)    /*!< Capture is done once every 4 events */
#define LL_TIM_ICPSC_DIV8                      (TIM_CCMR1_IC1PSC << 16U)      /*!< Capture is done once every 8 events */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_IC_FILTER Input Configuration Filter
  * @{
  */
#define LL_TIM_IC_FILTER_FDIV1                 0x00000000U                                                        /*!< No filter, sampling is done at fDTS */
#define LL_TIM_IC_FILTER_FDIV1_N2              (TIM_CCMR1_IC1F_0 << 16U)                                          /*!< fSAMPLING=fCK_INT, N=2 */
#define LL_TIM_IC_FILTER_FDIV1_N4              (TIM_CCMR1_IC1F_1 << 16U)                                          /*!< fSAMPLING=fCK_INT, N=4 */
#define LL_TIM_IC_FILTER_FDIV1_N8              ((TIM_CCMR1_IC1F_1 | TIM_CCMR1_IC1F_0) << 16U)                     /*!< fSAMPLING=fCK_INT, N=8 */
#define LL_TIM_IC_FILTER_FDIV2_N6              (TIM_CCMR1_IC1F_2 << 16U)                                          /*!< fSAMPLING=fDTS/2, N=6 */
#define LL_TIM_IC_FILTER_FDIV2_N8              ((TIM_CCMR1_IC1F_2 | TIM_CCMR1_IC1F_0) << 16U)                     /*!< fSAMPLING=fDTS/2, N=8 */
#define LL_TIM_IC_FILTER_FDIV4_N6              ((TIM_CCMR1_IC1F_2 | TIM_CCMR1_IC1F_1) << 16U)                     /*!< fSAMPLING=fDTS/4, N=6 */
#define LL_TIM_IC_FILTER_FDIV4_N8              ((TIM_CCMR1_IC1F_2 | TIM_CCMR1_IC1F_1 | TIM_CCMR1_IC1F_0) << 16U)  /*!< fSAMPLING=fDTS/4, N=8 */
#define LL_TIM_IC_FILTER_FDIV8_N6              (TIM_CCMR1_IC1F_3 << 16U)                                          /*!< fSAMPLING=fDTS/8, N=6 */
#define LL_TIM_IC_FILTER_FDIV8_N8              ((TIM_CCMR1_IC1F_3 | TIM_CCMR1_IC1F_0) << 16U)                     /*!< fSAMPLING=fDTS/8, N=8 */
#define LL_TIM_IC_FILTER_FDIV16_N5             ((TIM_CCMR1_IC1F_3 | TIM_CCMR1_IC1F_1) << 16U)                     /*!< fSAMPLING=fDTS/16, N=5 */
#define LL_TIM_IC_FILTER_FDIV16_N6             ((TIM_CCMR1_IC1F_3 | TIM_CCMR1_IC1F_1 | TIM_CCMR1_IC1F_0) << 16U)  /*!< fSAMPLING=fDTS/16, N=6 */
#define LL_TIM_IC_FILTER_FDIV16_N8             ((TIM_CCMR1_IC1F_3 | TIM_CCMR1_IC1F_2) << 16U)                     /*!< fSAMPLING=fDTS/16, N=8 */
#define LL_TIM_IC_FILTER_FDIV32_N5             ((TIM_CCMR1_IC1F_3 | TIM_CCMR1_IC1F_2 | TIM_CCMR1_IC1F_0) << 16U)  /*!< fSAMPLING=fDTS/32, N=5 */
#define LL_TIM_IC_FILTER_FDIV32_N6             ((TIM_CCMR1_IC1F_3 | TIM_CCMR1_IC1F_2 | TIM_CCMR1_IC1F_1) << 16U)  /*!< fSAMPLING=fDTS/32, N=6 */
#define LL_TIM_IC_FILTER_FDIV32_N8             (TIM_CCMR1_IC1F << 16U)                                            /*!< fSAMPLING=fDTS/32, N=8 */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_IC_POLARITY Input Configuration Polarity
  * @{
  */
#define LL_TIM_IC_POLARITY_RISING              0x00000000U                      /*!< The circuit is sensitive to TIxFP1 rising edge, TIxFP1 is not inverted */
#define LL_TIM_IC_POLARITY_FALLING             TIM_CCER_CC1P                    /*!< The circuit is sensitive to TIxFP1 falling edge, TIxFP1 is inverted */
#define LL_TIM_IC_POLARITY_BOTHEDGE            (TIM_CCER_CC1P | TIM_CCER_CC1NP) /*!< The circuit is sensitive to both TIxFP1 rising and falling edges, TIxFP1 is not inverted */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_CLOCKSOURCE Clock Source
  * @{
  */
#define LL_TIM_CLOCKSOURCE_INTERNAL            0x00000000U                                          /*!< The timer is clocked by the internal clock provided from the RCC */
#define LL_TIM_CLOCKSOURCE_EXT_MODE1           (TIM_SMCR_SMS_2 | TIM_SMCR_SMS_1 | TIM_SMCR_SMS_0)   /*!< Counter counts at each rising or falling edge on a selected input*/
#define LL_TIM_CLOCKSOURCE_EXT_MODE2           TIM_SMCR_ECE                                         /*!< Counter counts at each rising or falling edge on the external trigger input ETR */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_ENCODERMODE Encoder Mode
  * @{
  */
#define LL_TIM_ENCODERMODE_X2_TI1                     TIM_SMCR_SMS_0                                                     /*!< Quadrature encoder mode 1, x2 mode - Counter counts up/down on TI1FP1 edge depending on TI2FP2 level */
#define LL_TIM_ENCODERMODE_X2_TI2                     TIM_SMCR_SMS_1                                                     /*!< Quadrature encoder mode 2, x2 mode - Counter counts up/down on TI2FP2 edge depending on TI1FP1 level */
#define LL_TIM_ENCODERMODE_X4_TI12                   (TIM_SMCR_SMS_1 | TIM_SMCR_SMS_0)                                   /*!< Quadrature encoder mode 3, x4 mode - Counter counts up/down on both TI1FP1 and TI2FP2 edges depending on the level of the other input */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_TRGO Trigger Output
  * @{
  */
#define LL_TIM_TRGO_RESET                      0x00000000U                                     /*!< UG bit from the TIMx_EGR register is used as trigger output */
#define LL_TIM_TRGO_ENABLE                     TIM_CR2_MMS_0                                   /*!< Counter Enable signal (CNT_EN) is used as trigger output */
#define LL_TIM_TRGO_UPDATE                     TIM_CR2_MMS_1                                   /*!< Update event is used as trigger output */
#define LL_TIM_TRGO_CC1IF                      (TIM_CR2_MMS_1 | TIM_CR2_MMS_0)                 /*!< CC1 capture or a compare match is used as trigger output */
#define LL_TIM_TRGO_OC1REF                     TIM_CR2_MMS_2                                   /*!< OC1REF signal is used as trigger output */
#define LL_TIM_TRGO_OC2REF                     (TIM_CR2_MMS_2 | TIM_CR2_MMS_0)                 /*!< OC2REF signal is used as trigger output */
#define LL_TIM_TRGO_OC3REF                     (TIM_CR2_MMS_2 | TIM_CR2_MMS_1)                 /*!< OC3REF signal is used as trigger output */
#define LL_TIM_TRGO_OC4REF                     (TIM_CR2_MMS_2 | TIM_CR2_MMS_1 | TIM_CR2_MMS_0) /*!< OC4REF signal is used as trigger output */
/**
  * @}
  */


/** @defgroup TIM_LL_EC_SLAVEMODE Slave Mode
  * @{
  */
#define LL_TIM_SLAVEMODE_DISABLED              0x00000000U                         /*!< Slave mode disabled */
#define LL_TIM_SLAVEMODE_RESET                 TIM_SMCR_SMS_2                      /*!< Reset Mode - Rising edge of the selected trigger input (TRGI) reinitializes the counter */
#define LL_TIM_SLAVEMODE_GATED                 (TIM_SMCR_SMS_2 | TIM_SMCR_SMS_0)   /*!< Gated Mode - The counter clock is enabled when the trigger input (TRGI) is high */
#define LL_TIM_SLAVEMODE_TRIGGER               (TIM_SMCR_SMS_2 | TIM_SMCR_SMS_1)   /*!< Trigger Mode - The counter starts at a rising edge of the trigger TRGI */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_TS Trigger Selection
  * @{
  */
#define LL_TIM_TS_ITR0                         0x00000000U                                                     /*!< Internal Trigger 0 (ITR0) is used as trigger input */
#define LL_TIM_TS_ITR1                         TIM_SMCR_TS_0                                                   /*!< Internal Trigger 1 (ITR1) is used as trigger input */
#define LL_TIM_TS_ITR2                         TIM_SMCR_TS_1                                                   /*!< Internal Trigger 2 (ITR2) is used as trigger input */
#define LL_TIM_TS_ITR3                         (TIM_SMCR_TS_0 | TIM_SMCR_TS_1)                                 /*!< Internal Trigger 3 (ITR3) is used as trigger input */
#define LL_TIM_TS_TI1F_ED                      TIM_SMCR_TS_2                                                   /*!< TI1 Edge Detector (TI1F_ED) is used as trigger input */
#define LL_TIM_TS_TI1FP1                       (TIM_SMCR_TS_2 | TIM_SMCR_TS_0)                                 /*!< Filtered Timer Input 1 (TI1FP1) is used as trigger input */
#define LL_TIM_TS_TI2FP2                       (TIM_SMCR_TS_2 | TIM_SMCR_TS_1)                                 /*!< Filtered Timer Input 2 (TI12P2) is used as trigger input */
#define LL_TIM_TS_ETRF                         (TIM_SMCR_TS_2 | TIM_SMCR_TS_1 | TIM_SMCR_TS_0)                 /*!< Filtered external Trigger (ETRF) is used as trigger input */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_ETR_POLARITY External Trigger Polarity
  * @{
  */
#define LL_TIM_ETR_POLARITY_NONINVERTED        0x00000000U             /*!< ETR is non-inverted, active at high level or rising edge */
#define LL_TIM_ETR_POLARITY_INVERTED           TIM_SMCR_ETP            /*!< ETR is inverted, active at low level or falling edge */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_ETR_PRESCALER External Trigger Prescaler
  * @{
  */
#define LL_TIM_ETR_PRESCALER_DIV1              0x00000000U             /*!< ETR prescaler OFF */
#define LL_TIM_ETR_PRESCALER_DIV2              TIM_SMCR_ETPS_0         /*!< ETR frequency is divided by 2 */
#define LL_TIM_ETR_PRESCALER_DIV4              TIM_SMCR_ETPS_1         /*!< ETR frequency is divided by 4 */
#define LL_TIM_ETR_PRESCALER_DIV8              TIM_SMCR_ETPS           /*!< ETR frequency is divided by 8 */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_ETR_FILTER External Trigger Filter
  * @{
  */
#define LL_TIM_ETR_FILTER_FDIV1                0x00000000U                                          /*!< No filter, sampling is done at fDTS */
#define LL_TIM_ETR_FILTER_FDIV1_N2             TIM_SMCR_ETF_0                                       /*!< fSAMPLING=fCK_INT, N=2 */
#define LL_TIM_ETR_FILTER_FDIV1_N4             TIM_SMCR_ETF_1                                       /*!< fSAMPLING=fCK_INT, N=4 */
#define LL_TIM_ETR_FILTER_FDIV1_N8             (TIM_SMCR_ETF_1 | TIM_SMCR_ETF_0)                    /*!< fSAMPLING=fCK_INT, N=8 */
#define LL_TIM_ETR_FILTER_FDIV2_N6             TIM_SMCR_ETF_2                                       /*!< fSAMPLING=fDTS/2, N=6 */
#define LL_TIM_ETR_FILTER_FDIV2_N8             (TIM_SMCR_ETF_2 | TIM_SMCR_ETF_0)                    /*!< fSAMPLING=fDTS/2, N=8 */
#define LL_TIM_ETR_FILTER_FDIV4_N6             (TIM_SMCR_ETF_2 | TIM_SMCR_ETF_1)                    /*!< fSAMPLING=fDTS/4, N=6 */
#define LL_TIM_ETR_FILTER_FDIV4_N8             (TIM_SMCR_ETF_2 | TIM_SMCR_ETF_1 | TIM_SMCR_ETF_0)   /*!< fSAMPLING=fDTS/4, N=8 */
#define LL_TIM_ETR_FILTER_FDIV8_N6             TIM_SMCR_ETF_3                                       /*!< fSAMPLING=fDTS/8, N=8 */
#define LL_TIM_ETR_FILTER_FDIV8_N8             (TIM_SMCR_ETF_3 | TIM_SMCR_ETF_0)                    /*!< fSAMPLING=fDTS/16, N=5 */
#define LL_TIM_ETR_FILTER_FDIV16_N5            (TIM_SMCR_ETF_3 | TIM_SMCR_ETF_1)                    /*!< fSAMPLING=fDTS/16, N=6 */
#define LL_TIM_ETR_FILTER_FDIV16_N6            (TIM_SMCR_ETF_3 | TIM_SMCR_ETF_1 | TIM_SMCR_ETF_0)   /*!< fSAMPLING=fDTS/16, N=8 */
#define LL_TIM_ETR_FILTER_FDIV16_N8            (TIM_SMCR_ETF_3 | TIM_SMCR_ETF_2)                    /*!< fSAMPLING=fDTS/16, N=5 */
#define LL_TIM_ETR_FILTER_FDIV32_N5            (TIM_SMCR_ETF_3 | TIM_SMCR_ETF_2 | TIM_SMCR_ETF_0)   /*!< fSAMPLING=fDTS/32, N=5 */
#define LL_TIM_ETR_FILTER_FDIV32_N6            (TIM_SMCR_ETF_3 | TIM_SMCR_ETF_2 | TIM_SMCR_ETF_1)   /*!< fSAMPLING=fDTS/32, N=6 */
#define LL_TIM_ETR_FILTER_FDIV32_N8            TIM_SMCR_ETF                                         /*!< fSAMPLING=fDTS/32, N=8 */
/**
  * @}
  */


/** @defgroup TIM_LL_EC_BREAK_POLARITY break polarity
  * @{
  */
#define LL_TIM_BREAK_POLARITY_LOW              0x00000000U               /*!< Break input BRK is active low */
#define LL_TIM_BREAK_POLARITY_HIGH             TIM_BDTR_BKP              /*!< Break input BRK is active high */
/**
  * @}
  */




/** @defgroup TIM_LL_EC_OSSI OSSI
  * @{
  */
#define LL_TIM_OSSI_DISABLE                    0x00000000U             /*!< When inactive, OCx/OCxN outputs are disabled */
#define LL_TIM_OSSI_ENABLE                     TIM_BDTR_OSSI           /*!< When inactive, OxC/OCxN outputs are first forced with their inactive level then forced to their idle level after the deadtime */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_OSSR OSSR
  * @{
  */
#define LL_TIM_OSSR_DISABLE                    0x00000000U             /*!< When inactive, OCx/OCxN outputs are disabled */
#define LL_TIM_OSSR_ENABLE                     TIM_BDTR_OSSR           /*!< When inactive, OC/OCN outputs are enabled with their inactive level as soon as CCxE=1 or CCxNE=1 */
/**
  * @}
  */


/** @defgroup TIM_LL_EC_DMABURST_BASEADDR DMA Burst Base Address
  * @{
  */
#define LL_TIM_DMABURST_BASEADDR_CR1           0x00000000U                                                      /*!< TIMx_CR1 register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_CR2           TIM_DCR_DBA_0                                                    /*!< TIMx_CR2 register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_SMCR          TIM_DCR_DBA_1                                                    /*!< TIMx_SMCR register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_DIER          (TIM_DCR_DBA_1 |  TIM_DCR_DBA_0)                                 /*!< TIMx_DIER register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_SR            TIM_DCR_DBA_2                                                    /*!< TIMx_SR register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_EGR           (TIM_DCR_DBA_2 | TIM_DCR_DBA_0)                                  /*!< TIMx_EGR register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_CCMR1         (TIM_DCR_DBA_2 | TIM_DCR_DBA_1)                                  /*!< TIMx_CCMR1 register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_CCMR2         (TIM_DCR_DBA_2 | TIM_DCR_DBA_1 | TIM_DCR_DBA_0)                  /*!< TIMx_CCMR2 register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_CCER          TIM_DCR_DBA_3                                                    /*!< TIMx_CCER register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_CNT           (TIM_DCR_DBA_3 | TIM_DCR_DBA_0)                                  /*!< TIMx_CNT register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_PSC           (TIM_DCR_DBA_3 | TIM_DCR_DBA_1)                                  /*!< TIMx_PSC register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_ARR           (TIM_DCR_DBA_3 | TIM_DCR_DBA_1 | TIM_DCR_DBA_0)                  /*!< TIMx_ARR register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_RCR           (TIM_DCR_DBA_3 | TIM_DCR_DBA_2)                                  /*!< TIMx_RCR register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_CCR1          (TIM_DCR_DBA_3 | TIM_DCR_DBA_2 | TIM_DCR_DBA_0)                  /*!< TIMx_CCR1 register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_CCR2          (TIM_DCR_DBA_3 | TIM_DCR_DBA_2 | TIM_DCR_DBA_1)                  /*!< TIMx_CCR2 register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_CCR3          (TIM_DCR_DBA_3 | TIM_DCR_DBA_2 | TIM_DCR_DBA_1 | TIM_DCR_DBA_0)  /*!< TIMx_CCR3 register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_CCR4          TIM_DCR_DBA_4                                                    /*!< TIMx_CCR4 register is the DMA base address for DMA burst */
#define LL_TIM_DMABURST_BASEADDR_BDTR          (TIM_DCR_DBA_4 | TIM_DCR_DBA_0)                                  /*!< TIMx_BDTR register is the DMA base address for DMA burst */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_DMABURST_LENGTH DMA Burst Length
  * @{
  */
#define LL_TIM_DMABURST_LENGTH_1TRANSFER       0x00000000U                                                     /*!< Transfer is done to 1 register starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_2TRANSFERS      TIM_DCR_DBL_0                                                   /*!< Transfer is done to 2 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_3TRANSFERS      TIM_DCR_DBL_1                                                   /*!< Transfer is done to 3 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_4TRANSFERS      (TIM_DCR_DBL_1 |  TIM_DCR_DBL_0)                                /*!< Transfer is done to 4 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_5TRANSFERS      TIM_DCR_DBL_2                                                   /*!< Transfer is done to 5 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_6TRANSFERS      (TIM_DCR_DBL_2 | TIM_DCR_DBL_0)                                 /*!< Transfer is done to 6 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_7TRANSFERS      (TIM_DCR_DBL_2 | TIM_DCR_DBL_1)                                 /*!< Transfer is done to 7 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_8TRANSFERS      (TIM_DCR_DBL_2 | TIM_DCR_DBL_1 | TIM_DCR_DBL_0)                 /*!< Transfer is done to 1 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_9TRANSFERS      TIM_DCR_DBL_3                                                   /*!< Transfer is done to 9 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_10TRANSFERS     (TIM_DCR_DBL_3 | TIM_DCR_DBL_0)                                 /*!< Transfer is done to 10 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_11TRANSFERS     (TIM_DCR_DBL_3 | TIM_DCR_DBL_1)                                 /*!< Transfer is done to 11 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_12TRANSFERS     (TIM_DCR_DBL_3 | TIM_DCR_DBL_1 | TIM_DCR_DBL_0)                 /*!< Transfer is done to 12 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_13TRANSFERS     (TIM_DCR_DBL_3 | TIM_DCR_DBL_2)                                 /*!< Transfer is done to 13 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_14TRANSFERS     (TIM_DCR_DBL_3 | TIM_DCR_DBL_2 | TIM_DCR_DBL_0)                 /*!< Transfer is done to 14 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_15TRANSFERS     (TIM_DCR_DBL_3 | TIM_DCR_DBL_2 | TIM_DCR_DBL_1)                 /*!< Transfer is done to 15 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_16TRANSFERS     (TIM_DCR_DBL_3 | TIM_DCR_DBL_2 | TIM_DCR_DBL_1 | TIM_DCR_DBL_0) /*!< Transfer is done to 16 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_17TRANSFERS     TIM_DCR_DBL_4                                                   /*!< Transfer is done to 17 registers starting from the DMA burst base address */
#define LL_TIM_DMABURST_LENGTH_18TRANSFERS     (TIM_DCR_DBL_4 |  TIM_DCR_DBL_0)                                /*!< Transfer is done to 18 registers starting from the DMA burst base address */
/**
  * @}
  */


/** @defgroup TIM_LL_EC_TIM2_ITR1_RMP_TIM8  TIM2 Internal Trigger1 Remap TIM8
  * @{
  */
#define LL_TIM_TIM2_ITR1_RMP_TIM8_TRGO    TIM2_OR_RMP_MASK                        /*!< TIM2_ITR1 is connected to TIM8_TRGO */
#define LL_TIM_TIM2_ITR1_RMP_ETH_PTP      (TIM_OR_ITR1_RMP_0 | TIM2_OR_RMP_MASK)  /*!< TIM2_ITR1 is connected to ETH_PTP */
#define LL_TIM_TIM2_ITR1_RMP_OTG_FS_SOF   (TIM_OR_ITR1_RMP_1 | TIM2_OR_RMP_MASK)  /*!< TIM2_ITR1 is connected to OTG_FS SOF */
#define LL_TIM_TIM2_ITR1_RMP_OTG_HS_SOF   (TIM_OR_ITR1_RMP | TIM2_OR_RMP_MASK)    /*!< TIM2_ITR1 is connected to OTG_HS SOF */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_TIM5_TI4_RMP  TIM5 External Input Ch4 Remap
  * @{
  */
#define LL_TIM_TIM5_TI4_RMP_GPIO        TIM5_OR_RMP_MASK                         /*!< TIM5 channel 4 is connected to GPIO */
#define LL_TIM_TIM5_TI4_RMP_LSI         (TIM_OR_TI4_RMP_0 | TIM5_OR_RMP_MASK)    /*!< TIM5 channel 4 is connected to LSI internal clock */
#define LL_TIM_TIM5_TI4_RMP_LSE         (TIM_OR_TI4_RMP_1 | TIM5_OR_RMP_MASK)    /*!< TIM5 channel 4 is connected to LSE */
#define LL_TIM_TIM5_TI4_RMP_RTC         (TIM_OR_TI4_RMP | TIM5_OR_RMP_MASK)      /*!< TIM5 channel 4 is connected to RTC wakeup interrupt */
/**
  * @}
  */

/** @defgroup TIM_LL_EC_TIM11_TI1_RMP  TIM11 External Input Capture 1 Remap
  * @{
  */
#define LL_TIM_TIM11_TI1_RMP_GPIO        TIM11_OR_RMP_MASK                          /*!< TIM11 channel 1 is connected to GPIO */
#if defined(SPDIFRX)
#define LL_TIM_TIM11_TI1_RMP_SPDIFRX     (TIM_OR_TI1_RMP_0 | TIM11_OR_RMP_MASK)     /*!< TIM11 channel 1 is connected to SPDIFRX */

/* Legacy define */
#define  LL_TIM_TIM11_TI1_RMP_GPIO1      LL_TIM_TIM11_TI1_RMP_SPDIFRX               /*!< Legacy define for LL_TIM_TIM11_TI1_RMP_SPDIFRX */

#else
#define LL_TIM_TIM11_TI1_RMP_GPIO1       (TIM_OR_TI1_RMP_0 | TIM11_OR_RMP_MASK)     /*!< TIM11 channel 1 is connected to GPIO */
#endif /* SPDIFRX */
#define LL_TIM_TIM11_TI1_RMP_GPIO2       (TIM_OR_TI1_RMP   | TIM11_OR_RMP_MASK)     /*!< TIM11 channel 1 is connected to GPIO */
#define LL_TIM_TIM11_TI1_RMP_HSE_RTC     (TIM_OR_TI1_RMP_1 | TIM11_OR_RMP_MASK)     /*!< TIM11 channel 1 is connected to HSE_RTC */
/**
  * @}
  */
#if defined(LPTIM_OR_TIM1_ITR2_RMP) && defined(LPTIM_OR_TIM5_ITR1_RMP) && defined(LPTIM_OR_TIM9_ITR1_RMP)

#define LL_TIM_LPTIM_REMAP_MASK           0x10000000U

#define LL_TIM_TIM9_ITR1_RMP_TIM3_TRGO    LL_TIM_LPTIM_REMAP_MASK                              /*!< TIM9_ITR1 is connected to TIM3 TRGO */
#define LL_TIM_TIM9_ITR1_RMP_LPTIM       (LL_TIM_LPTIM_REMAP_MASK | LPTIM_OR_TIM9_ITR1_RMP)    /*!< TIM9_ITR1 is connected to LPTIM1 output */

#define LL_TIM_TIM5_ITR1_RMP_TIM3_TRGO    LL_TIM_LPTIM_REMAP_MASK                              /*!< TIM5_ITR1 is connected to TIM3 TRGO */
#define LL_TIM_TIM5_ITR1_RMP_LPTIM       (LL_TIM_LPTIM_REMAP_MASK | LPTIM_OR_TIM5_ITR1_RMP)    /*!< TIM5_ITR1 is connected to LPTIM1 output */

#define LL_TIM_TIM1_ITR2_RMP_TIM3_TRGO    LL_TIM_LPTIM_REMAP_MASK                              /*!< TIM1_ITR2 is connected to TIM3 TRGO */
#define LL_TIM_TIM1_ITR2_RMP_LPTIM       (LL_TIM_LPTIM_REMAP_MASK | LPTIM_OR_TIM1_ITR2_RMP)    /*!< TIM1_ITR2 is connected to LPTIM1 output */

#endif /* LPTIM_OR_TIM1_ITR2_RMP &&  LPTIM_OR_TIM5_ITR1_RMP && LPTIM_OR_TIM9_ITR1_RMP */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup TIM_LL_Exported_Macros TIM Exported Macros
  * @{
  */

/** @defgroup TIM_LL_EM_WRITE_READ Common Write and read registers Macros
  * @{
  */
/**
  * @brief  Write a value in TIM register.
  * @param  __INSTANCE__ TIM Instance
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_TIM_WriteReg(__INSTANCE__, __REG__, __VALUE__) WRITE_REG((__INSTANCE__)->__REG__, (__VALUE__))

/**
  * @brief  Read a value in TIM register.
  * @param  __INSTANCE__ TIM Instance
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_TIM_ReadReg(__INSTANCE__, __REG__) READ_REG((__INSTANCE__)->__REG__)
/**
  * @}
  */

/** @defgroup TIM_LL_EM_Exported_Macros Exported_Macros
  * @{
  */

/**
  * @brief  HELPER macro calculating DTG[0:7] in the TIMx_BDTR register to achieve the requested dead time duration.
  * @note ex: @ref __LL_TIM_CALC_DEADTIME (80000000, @ref LL_TIM_GetClockDivision (), 120);
  * @param  __TIMCLK__ timer input clock frequency (in Hz)
  * @param  __CKD__ This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV1
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV2
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV4
  * @param  __DT__ deadtime duration (in ns)
  * @retval DTG[0:7]
  */
#define __LL_TIM_CALC_DEADTIME(__TIMCLK__, __CKD__, __DT__)  \
  ( (((uint64_t)((__DT__)*1000U)) < ((DT_DELAY_1+1U) * TIM_CALC_DTS((__TIMCLK__), (__CKD__))))    ?  \
    (uint8_t)(((uint64_t)((__DT__)*1000U) / TIM_CALC_DTS((__TIMCLK__), (__CKD__)))  & DT_DELAY_1) :      \
    (((uint64_t)((__DT__)*1000U)) < ((64U + (DT_DELAY_2+1U)) * 2U * TIM_CALC_DTS((__TIMCLK__), (__CKD__))))  ?  \
    (uint8_t)(DT_RANGE_2 | ((uint8_t)((uint8_t)((((uint64_t)((__DT__)*1000U))/ TIM_CALC_DTS((__TIMCLK__),   \
                                                 (__CKD__))) >> 1U) - (uint8_t) 64) & DT_DELAY_2)) :\
    (((uint64_t)((__DT__)*1000U)) < ((32U + (DT_DELAY_3+1U)) * 8U * TIM_CALC_DTS((__TIMCLK__), (__CKD__))))  ?  \
    (uint8_t)(DT_RANGE_3 | ((uint8_t)((uint8_t)(((((uint64_t)(__DT__)*1000U))/ TIM_CALC_DTS((__TIMCLK__),  \
                                                 (__CKD__))) >> 3U) - (uint8_t) 32) & DT_DELAY_3)) :\
    (((uint64_t)((__DT__)*1000U)) < ((32U + (DT_DELAY_4+1U)) * 16U * TIM_CALC_DTS((__TIMCLK__), (__CKD__)))) ?  \
    (uint8_t)(DT_RANGE_4 | ((uint8_t)((uint8_t)(((((uint64_t)(__DT__)*1000U))/ TIM_CALC_DTS((__TIMCLK__),  \
                                                 (__CKD__))) >> 4U) - (uint8_t) 32) & DT_DELAY_4)) :\
    0U)

/**
  * @brief  HELPER macro calculating the prescaler value to achieve the required counter clock frequency.
  * @note ex: @ref __LL_TIM_CALC_PSC (80000000, 1000000);
  * @param  __TIMCLK__ timer input clock frequency (in Hz)
  * @param  __CNTCLK__ counter clock frequency (in Hz)
  * @retval Prescaler value  (between Min_Data=0 and Max_Data=65535)
  */
#define __LL_TIM_CALC_PSC(__TIMCLK__, __CNTCLK__)   \
  (((__TIMCLK__) >= (__CNTCLK__)) ? (uint32_t)(((__TIMCLK__)/(__CNTCLK__)) - 1U) : 0U)

/**
  * @brief  HELPER macro calculating the auto-reload value to achieve the required output signal frequency.
  * @note ex: @ref __LL_TIM_CALC_ARR (1000000, @ref LL_TIM_GetPrescaler (), 10000);
  * @param  __TIMCLK__ timer input clock frequency (in Hz)
  * @param  __PSC__ prescaler
  * @param  __FREQ__ output signal frequency (in Hz)
  * @retval  Auto-reload value  (between Min_Data=0 and Max_Data=65535)
  */
#define __LL_TIM_CALC_ARR(__TIMCLK__, __PSC__, __FREQ__) \
  ((((__TIMCLK__)/((__PSC__) + 1U)) >= (__FREQ__)) ? (((__TIMCLK__)/((__FREQ__) * ((__PSC__) + 1U))) - 1U) : 0U)

/**
  * @brief  HELPER macro calculating the compare value required to achieve the required timer output compare
  *         active/inactive delay.
  * @note ex: @ref __LL_TIM_CALC_DELAY (1000000, @ref LL_TIM_GetPrescaler (), 10);
  * @param  __TIMCLK__ timer input clock frequency (in Hz)
  * @param  __PSC__ prescaler
  * @param  __DELAY__ timer output compare active/inactive delay (in us)
  * @retval Compare value  (between Min_Data=0 and Max_Data=65535)
  */
#define __LL_TIM_CALC_DELAY(__TIMCLK__, __PSC__, __DELAY__)  \
  ((uint32_t)(((uint64_t)(__TIMCLK__) * (uint64_t)(__DELAY__)) \
              / ((uint64_t)1000000U * (uint64_t)((__PSC__) + 1U))))

/**
  * @brief  HELPER macro calculating the auto-reload value to achieve the required pulse duration
  *         (when the timer operates in one pulse mode).
  * @note ex: @ref __LL_TIM_CALC_PULSE (1000000, @ref LL_TIM_GetPrescaler (), 10, 20);
  * @param  __TIMCLK__ timer input clock frequency (in Hz)
  * @param  __PSC__ prescaler
  * @param  __DELAY__ timer output compare active/inactive delay (in us)
  * @param  __PULSE__ pulse duration (in us)
  * @retval Auto-reload value  (between Min_Data=0 and Max_Data=65535)
  */
#define __LL_TIM_CALC_PULSE(__TIMCLK__, __PSC__, __DELAY__, __PULSE__)  \
  ((uint32_t)(__LL_TIM_CALC_DELAY((__TIMCLK__), (__PSC__), (__PULSE__)) \
              + __LL_TIM_CALC_DELAY((__TIMCLK__), (__PSC__), (__DELAY__))))

/**
  * @brief  HELPER macro retrieving the ratio of the input capture prescaler
  * @note ex: @ref __LL_TIM_GET_ICPSC_RATIO (@ref LL_TIM_IC_GetPrescaler ());
  * @param  __ICPSC__ This parameter can be one of the following values:
  *         @arg @ref LL_TIM_ICPSC_DIV1
  *         @arg @ref LL_TIM_ICPSC_DIV2
  *         @arg @ref LL_TIM_ICPSC_DIV4
  *         @arg @ref LL_TIM_ICPSC_DIV8
  * @retval Input capture prescaler ratio (1, 2, 4 or 8)
  */
#define __LL_TIM_GET_ICPSC_RATIO(__ICPSC__)  \
  ((uint32_t)(0x01U << (((__ICPSC__) >> 16U) >> TIM_CCMR1_IC1PSC_Pos)))


/**
  * @}
  */


/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup TIM_LL_Exported_Functions TIM Exported Functions
  * @{
  */

/** @defgroup TIM_LL_EF_Time_Base Time Base configuration
  * @{
  */
/**
  * @brief  Enable timer counter.
  * @rmtoll CR1          CEN           LL_TIM_EnableCounter
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableCounter(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->CR1, TIM_CR1_CEN);
}

/**
  * @brief  Disable timer counter.
  * @rmtoll CR1          CEN           LL_TIM_DisableCounter
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableCounter(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->CR1, TIM_CR1_CEN);
}

/**
  * @brief  Indicates whether the timer counter is enabled.
  * @rmtoll CR1          CEN           LL_TIM_IsEnabledCounter
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledCounter(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->CR1, TIM_CR1_CEN) == (TIM_CR1_CEN)) ? 1UL : 0UL);
}

/**
  * @brief  Enable update event generation.
  * @rmtoll CR1          UDIS          LL_TIM_EnableUpdateEvent
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableUpdateEvent(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->CR1, TIM_CR1_UDIS);
}

/**
  * @brief  Disable update event generation.
  * @rmtoll CR1          UDIS          LL_TIM_DisableUpdateEvent
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableUpdateEvent(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->CR1, TIM_CR1_UDIS);
}

/**
  * @brief  Indicates whether update event generation is enabled.
  * @rmtoll CR1          UDIS          LL_TIM_IsEnabledUpdateEvent
  * @param  TIMx Timer instance
  * @retval Inverted state of bit (0 or 1).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledUpdateEvent(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->CR1, TIM_CR1_UDIS) == (uint32_t)RESET) ? 1UL : 0UL);
}

/**
  * @brief  Set update event source
  * @note Update event source set to LL_TIM_UPDATESOURCE_REGULAR: any of the following events
  *       generate an update interrupt or DMA request if enabled:
  *        - Counter overflow/underflow
  *        - Setting the UG bit
  *        - Update generation through the slave mode controller
  * @note Update event source set to LL_TIM_UPDATESOURCE_COUNTER: only counter
  *       overflow/underflow generates an update interrupt or DMA request if enabled.
  * @rmtoll CR1          URS           LL_TIM_SetUpdateSource
  * @param  TIMx Timer instance
  * @param  UpdateSource This parameter can be one of the following values:
  *         @arg @ref LL_TIM_UPDATESOURCE_REGULAR
  *         @arg @ref LL_TIM_UPDATESOURCE_COUNTER
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetUpdateSource(TIM_TypeDef *TIMx, uint32_t UpdateSource)
{
  MODIFY_REG(TIMx->CR1, TIM_CR1_URS, UpdateSource);
}

/**
  * @brief  Get actual event update source
  * @rmtoll CR1          URS           LL_TIM_GetUpdateSource
  * @param  TIMx Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_UPDATESOURCE_REGULAR
  *         @arg @ref LL_TIM_UPDATESOURCE_COUNTER
  */
__STATIC_INLINE uint32_t LL_TIM_GetUpdateSource(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_BIT(TIMx->CR1, TIM_CR1_URS));
}

/**
  * @brief  Set one pulse mode (one shot v.s. repetitive).
  * @rmtoll CR1          OPM           LL_TIM_SetOnePulseMode
  * @param  TIMx Timer instance
  * @param  OnePulseMode This parameter can be one of the following values:
  *         @arg @ref LL_TIM_ONEPULSEMODE_SINGLE
  *         @arg @ref LL_TIM_ONEPULSEMODE_REPETITIVE
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetOnePulseMode(TIM_TypeDef *TIMx, uint32_t OnePulseMode)
{
  MODIFY_REG(TIMx->CR1, TIM_CR1_OPM, OnePulseMode);
}

/**
  * @brief  Get actual one pulse mode.
  * @rmtoll CR1          OPM           LL_TIM_GetOnePulseMode
  * @param  TIMx Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_ONEPULSEMODE_SINGLE
  *         @arg @ref LL_TIM_ONEPULSEMODE_REPETITIVE
  */
__STATIC_INLINE uint32_t LL_TIM_GetOnePulseMode(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_BIT(TIMx->CR1, TIM_CR1_OPM));
}

/**
  * @brief  Set the timer counter counting mode.
  * @note Macro IS_TIM_COUNTER_MODE_SELECT_INSTANCE(TIMx) can be used to
  *       check whether or not the counter mode selection feature is supported
  *       by a timer instance.
  * @note Switching from Center Aligned counter mode to Edge counter mode (or reverse)
  *       requires a timer reset to avoid unexpected direction
  *       due to DIR bit readonly in center aligned mode.
  * @rmtoll CR1          DIR           LL_TIM_SetCounterMode\n
  *         CR1          CMS           LL_TIM_SetCounterMode
  * @param  TIMx Timer instance
  * @param  CounterMode This parameter can be one of the following values:
  *         @arg @ref LL_TIM_COUNTERMODE_UP
  *         @arg @ref LL_TIM_COUNTERMODE_DOWN
  *         @arg @ref LL_TIM_COUNTERMODE_CENTER_UP
  *         @arg @ref LL_TIM_COUNTERMODE_CENTER_DOWN
  *         @arg @ref LL_TIM_COUNTERMODE_CENTER_UP_DOWN
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetCounterMode(TIM_TypeDef *TIMx, uint32_t CounterMode)
{
  MODIFY_REG(TIMx->CR1, (TIM_CR1_DIR | TIM_CR1_CMS), CounterMode);
}

/**
  * @brief  Get actual counter mode.
  * @note Macro IS_TIM_COUNTER_MODE_SELECT_INSTANCE(TIMx) can be used to
  *       check whether or not the counter mode selection feature is supported
  *       by a timer instance.
  * @rmtoll CR1          DIR           LL_TIM_GetCounterMode\n
  *         CR1          CMS           LL_TIM_GetCounterMode
  * @param  TIMx Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_COUNTERMODE_UP
  *         @arg @ref LL_TIM_COUNTERMODE_DOWN
  *         @arg @ref LL_TIM_COUNTERMODE_CENTER_UP
  *         @arg @ref LL_TIM_COUNTERMODE_CENTER_DOWN
  *         @arg @ref LL_TIM_COUNTERMODE_CENTER_UP_DOWN
  */
__STATIC_INLINE uint32_t LL_TIM_GetCounterMode(TIM_TypeDef *TIMx)
{
  uint32_t counter_mode;

  counter_mode = (uint32_t)(READ_BIT(TIMx->CR1, TIM_CR1_CMS));

  if (counter_mode == 0U)
  {
    counter_mode = (uint32_t)(READ_BIT(TIMx->CR1, TIM_CR1_DIR));
  }

  return counter_mode;
}

/**
  * @brief  Enable auto-reload (ARR) preload.
  * @rmtoll CR1          ARPE          LL_TIM_EnableARRPreload
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableARRPreload(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->CR1, TIM_CR1_ARPE);
}

/**
  * @brief  Disable auto-reload (ARR) preload.
  * @rmtoll CR1          ARPE          LL_TIM_DisableARRPreload
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableARRPreload(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->CR1, TIM_CR1_ARPE);
}

/**
  * @brief  Indicates whether auto-reload (ARR) preload is enabled.
  * @rmtoll CR1          ARPE          LL_TIM_IsEnabledARRPreload
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledARRPreload(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->CR1, TIM_CR1_ARPE) == (TIM_CR1_ARPE)) ? 1UL : 0UL);
}

/**
  * @brief  Set the division ratio between the timer clock  and the sampling clock used by the dead-time generators
  *         (when supported) and the digital filters.
  * @note Macro IS_TIM_CLOCK_DIVISION_INSTANCE(TIMx) can be used to check
  *       whether or not the clock division feature is supported by the timer
  *       instance.
  * @rmtoll CR1          CKD           LL_TIM_SetClockDivision
  * @param  TIMx Timer instance
  * @param  ClockDivision This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV1
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV2
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV4
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetClockDivision(TIM_TypeDef *TIMx, uint32_t ClockDivision)
{
  MODIFY_REG(TIMx->CR1, TIM_CR1_CKD, ClockDivision);
}

/**
  * @brief  Get the actual division ratio between the timer clock  and the sampling clock used by the dead-time
  *         generators (when supported) and the digital filters.
  * @note Macro IS_TIM_CLOCK_DIVISION_INSTANCE(TIMx) can be used to check
  *       whether or not the clock division feature is supported by the timer
  *       instance.
  * @rmtoll CR1          CKD           LL_TIM_GetClockDivision
  * @param  TIMx Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV1
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV2
  *         @arg @ref LL_TIM_CLOCKDIVISION_DIV4
  */
__STATIC_INLINE uint32_t LL_TIM_GetClockDivision(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_BIT(TIMx->CR1, TIM_CR1_CKD));
}

/**
  * @brief  Set the counter value.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @rmtoll CNT          CNT           LL_TIM_SetCounter
  * @param  TIMx Timer instance
  * @param  Counter Counter value (between Min_Data=0 and Max_Data=0xFFFF or 0xFFFFFFFF)
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetCounter(TIM_TypeDef *TIMx, uint32_t Counter)
{
  WRITE_REG(TIMx->CNT, Counter);
}

/**
  * @brief  Get the counter value.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @rmtoll CNT          CNT           LL_TIM_GetCounter
  * @param  TIMx Timer instance
  * @retval Counter value (between Min_Data=0 and Max_Data=0xFFFF or 0xFFFFFFFF)
  */
__STATIC_INLINE uint32_t LL_TIM_GetCounter(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->CNT));
}

/**
  * @brief  Get the current direction of the counter
  * @rmtoll CR1          DIR           LL_TIM_GetDirection
  * @param  TIMx Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_COUNTERDIRECTION_UP
  *         @arg @ref LL_TIM_COUNTERDIRECTION_DOWN
  */
__STATIC_INLINE uint32_t LL_TIM_GetDirection(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_BIT(TIMx->CR1, TIM_CR1_DIR));
}

/**
  * @brief  Set the prescaler value.
  * @note The counter clock frequency CK_CNT is equal to fCK_PSC / (PSC[15:0] + 1).
  * @note The prescaler can be changed on the fly as this control register is buffered. The new
  *       prescaler ratio is taken into account at the next update event.
  * @note Helper macro @ref __LL_TIM_CALC_PSC can be used to calculate the Prescaler parameter
  * @rmtoll PSC          PSC           LL_TIM_SetPrescaler
  * @param  TIMx Timer instance
  * @param  Prescaler between Min_Data=0 and Max_Data=65535
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetPrescaler(TIM_TypeDef *TIMx, uint32_t Prescaler)
{
  WRITE_REG(TIMx->PSC, Prescaler);
}

/**
  * @brief  Get the prescaler value.
  * @rmtoll PSC          PSC           LL_TIM_GetPrescaler
  * @param  TIMx Timer instance
  * @retval  Prescaler value between Min_Data=0 and Max_Data=65535
  */
__STATIC_INLINE uint32_t LL_TIM_GetPrescaler(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->PSC));
}

/**
  * @brief  Set the auto-reload value.
  * @note The counter is blocked while the auto-reload value is null.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Helper macro @ref __LL_TIM_CALC_ARR can be used to calculate the AutoReload parameter
  * @rmtoll ARR          ARR           LL_TIM_SetAutoReload
  * @param  TIMx Timer instance
  * @param  AutoReload between Min_Data=0 and Max_Data=65535
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetAutoReload(TIM_TypeDef *TIMx, uint32_t AutoReload)
{
  WRITE_REG(TIMx->ARR, AutoReload);
}

/**
  * @brief  Get the auto-reload value.
  * @rmtoll ARR          ARR           LL_TIM_GetAutoReload
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @param  TIMx Timer instance
  * @retval Auto-reload value
  */
__STATIC_INLINE uint32_t LL_TIM_GetAutoReload(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->ARR));
}

/**
  * @brief  Set the repetition counter value.
  * @note Macro IS_TIM_REPETITION_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a repetition counter.
  * @rmtoll RCR          REP           LL_TIM_SetRepetitionCounter
  * @param  TIMx Timer instance
  * @param  RepetitionCounter between Min_Data=0 and Max_Data=255 or 65535 for advanced timer.
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetRepetitionCounter(TIM_TypeDef *TIMx, uint32_t RepetitionCounter)
{
  WRITE_REG(TIMx->RCR, RepetitionCounter);
}

/**
  * @brief  Get the repetition counter value.
  * @note Macro IS_TIM_REPETITION_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a repetition counter.
  * @rmtoll RCR          REP           LL_TIM_GetRepetitionCounter
  * @param  TIMx Timer instance
  * @retval Repetition counter value
  */
__STATIC_INLINE uint32_t LL_TIM_GetRepetitionCounter(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->RCR));
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_Capture_Compare Capture Compare configuration
  * @{
  */
/**
  * @brief  Enable  the capture/compare control bits (CCxE, CCxNE and OCxM) preload.
  * @note CCxE, CCxNE and OCxM bits are preloaded, after having been written,
  *       they are updated only when a commutation event (COM) occurs.
  * @note Only on channels that have a complementary output.
  * @note Macro IS_TIM_COMMUTATION_EVENT_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance is able to generate a commutation event.
  * @rmtoll CR2          CCPC          LL_TIM_CC_EnablePreload
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_CC_EnablePreload(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->CR2, TIM_CR2_CCPC);
}

/**
  * @brief  Disable  the capture/compare control bits (CCxE, CCxNE and OCxM) preload.
  * @note Macro IS_TIM_COMMUTATION_EVENT_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance is able to generate a commutation event.
  * @rmtoll CR2          CCPC          LL_TIM_CC_DisablePreload
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_CC_DisablePreload(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->CR2, TIM_CR2_CCPC);
}

/**
  * @brief  Set the updated source of the capture/compare control bits (CCxE, CCxNE and OCxM).
  * @note Macro IS_TIM_COMMUTATION_EVENT_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance is able to generate a commutation event.
  * @rmtoll CR2          CCUS          LL_TIM_CC_SetUpdate
  * @param  TIMx Timer instance
  * @param  CCUpdateSource This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CCUPDATESOURCE_COMG_ONLY
  *         @arg @ref LL_TIM_CCUPDATESOURCE_COMG_AND_TRGI
  * @retval None
  */
__STATIC_INLINE void LL_TIM_CC_SetUpdate(TIM_TypeDef *TIMx, uint32_t CCUpdateSource)
{
  MODIFY_REG(TIMx->CR2, TIM_CR2_CCUS, CCUpdateSource);
}

/**
  * @brief  Set the trigger of the capture/compare DMA request.
  * @rmtoll CR2          CCDS          LL_TIM_CC_SetDMAReqTrigger
  * @param  TIMx Timer instance
  * @param  DMAReqTrigger This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CCDMAREQUEST_CC
  *         @arg @ref LL_TIM_CCDMAREQUEST_UPDATE
  * @retval None
  */
__STATIC_INLINE void LL_TIM_CC_SetDMAReqTrigger(TIM_TypeDef *TIMx, uint32_t DMAReqTrigger)
{
  MODIFY_REG(TIMx->CR2, TIM_CR2_CCDS, DMAReqTrigger);
}

/**
  * @brief  Get actual trigger of the capture/compare DMA request.
  * @rmtoll CR2          CCDS          LL_TIM_CC_GetDMAReqTrigger
  * @param  TIMx Timer instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_CCDMAREQUEST_CC
  *         @arg @ref LL_TIM_CCDMAREQUEST_UPDATE
  */
__STATIC_INLINE uint32_t LL_TIM_CC_GetDMAReqTrigger(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_BIT(TIMx->CR2, TIM_CR2_CCDS));
}

/**
  * @brief  Set the lock level to freeze the
  *         configuration of several capture/compare parameters.
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       the lock mechanism is supported by a timer instance.
  * @rmtoll BDTR         LOCK          LL_TIM_CC_SetLockLevel
  * @param  TIMx Timer instance
  * @param  LockLevel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_LOCKLEVEL_OFF
  *         @arg @ref LL_TIM_LOCKLEVEL_1
  *         @arg @ref LL_TIM_LOCKLEVEL_2
  *         @arg @ref LL_TIM_LOCKLEVEL_3
  * @retval None
  */
__STATIC_INLINE void LL_TIM_CC_SetLockLevel(TIM_TypeDef *TIMx, uint32_t LockLevel)
{
  MODIFY_REG(TIMx->BDTR, TIM_BDTR_LOCK, LockLevel);
}

/**
  * @brief  Enable capture/compare channels.
  * @rmtoll CCER         CC1E          LL_TIM_CC_EnableChannel\n
  *         CCER         CC1NE         LL_TIM_CC_EnableChannel\n
  *         CCER         CC2E          LL_TIM_CC_EnableChannel\n
  *         CCER         CC2NE         LL_TIM_CC_EnableChannel\n
  *         CCER         CC3E          LL_TIM_CC_EnableChannel\n
  *         CCER         CC3NE         LL_TIM_CC_EnableChannel\n
  *         CCER         CC4E          LL_TIM_CC_EnableChannel
  * @param  TIMx Timer instance
  * @param  Channels This parameter can be a combination of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH1N
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH2N
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH3N
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval None
  */
__STATIC_INLINE void LL_TIM_CC_EnableChannel(TIM_TypeDef *TIMx, uint32_t Channels)
{
  SET_BIT(TIMx->CCER, Channels);
}

/**
  * @brief  Disable capture/compare channels.
  * @rmtoll CCER         CC1E          LL_TIM_CC_DisableChannel\n
  *         CCER         CC1NE         LL_TIM_CC_DisableChannel\n
  *         CCER         CC2E          LL_TIM_CC_DisableChannel\n
  *         CCER         CC2NE         LL_TIM_CC_DisableChannel\n
  *         CCER         CC3E          LL_TIM_CC_DisableChannel\n
  *         CCER         CC3NE         LL_TIM_CC_DisableChannel\n
  *         CCER         CC4E          LL_TIM_CC_DisableChannel
  * @param  TIMx Timer instance
  * @param  Channels This parameter can be a combination of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH1N
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH2N
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH3N
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval None
  */
__STATIC_INLINE void LL_TIM_CC_DisableChannel(TIM_TypeDef *TIMx, uint32_t Channels)
{
  CLEAR_BIT(TIMx->CCER, Channels);
}

/**
  * @brief  Indicate whether channel(s) is(are) enabled.
  * @rmtoll CCER         CC1E          LL_TIM_CC_IsEnabledChannel\n
  *         CCER         CC1NE         LL_TIM_CC_IsEnabledChannel\n
  *         CCER         CC2E          LL_TIM_CC_IsEnabledChannel\n
  *         CCER         CC2NE         LL_TIM_CC_IsEnabledChannel\n
  *         CCER         CC3E          LL_TIM_CC_IsEnabledChannel\n
  *         CCER         CC3NE         LL_TIM_CC_IsEnabledChannel\n
  *         CCER         CC4E          LL_TIM_CC_IsEnabledChannel
  * @param  TIMx Timer instance
  * @param  Channels This parameter can be a combination of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH1N
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH2N
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH3N
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_CC_IsEnabledChannel(TIM_TypeDef *TIMx, uint32_t Channels)
{
  return ((READ_BIT(TIMx->CCER, Channels) == (Channels)) ? 1UL : 0UL);
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_Output_Channel Output channel configuration
  * @{
  */
/**
  * @brief  Configure an output channel.
  * @rmtoll CCMR1        CC1S          LL_TIM_OC_ConfigOutput\n
  *         CCMR1        CC2S          LL_TIM_OC_ConfigOutput\n
  *         CCMR2        CC3S          LL_TIM_OC_ConfigOutput\n
  *         CCMR2        CC4S          LL_TIM_OC_ConfigOutput\n
  *         CCER         CC1P          LL_TIM_OC_ConfigOutput\n
  *         CCER         CC2P          LL_TIM_OC_ConfigOutput\n
  *         CCER         CC3P          LL_TIM_OC_ConfigOutput\n
  *         CCER         CC4P          LL_TIM_OC_ConfigOutput\n
  *         CR2          OIS1          LL_TIM_OC_ConfigOutput\n
  *         CR2          OIS2          LL_TIM_OC_ConfigOutput\n
  *         CR2          OIS3          LL_TIM_OC_ConfigOutput\n
  *         CR2          OIS4          LL_TIM_OC_ConfigOutput
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @param  Configuration This parameter must be a combination of all the following values:
  *         @arg @ref LL_TIM_OCPOLARITY_HIGH or @ref LL_TIM_OCPOLARITY_LOW
  *         @arg @ref LL_TIM_OCIDLESTATE_LOW or @ref LL_TIM_OCIDLESTATE_HIGH
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_ConfigOutput(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t Configuration)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  CLEAR_BIT(*pReg, (TIM_CCMR1_CC1S << SHIFT_TAB_OCxx[iChannel]));
  MODIFY_REG(TIMx->CCER, (TIM_CCER_CC1P << SHIFT_TAB_CCxP[iChannel]),
             (Configuration & TIM_CCER_CC1P) << SHIFT_TAB_CCxP[iChannel]);
  MODIFY_REG(TIMx->CR2, (TIM_CR2_OIS1 << SHIFT_TAB_OISx[iChannel]),
             (Configuration & TIM_CR2_OIS1) << SHIFT_TAB_OISx[iChannel]);
}

/**
  * @brief  Define the behavior of the output reference signal OCxREF from which
  *         OCx and OCxN (when relevant) are derived.
  * @rmtoll CCMR1        OC1M          LL_TIM_OC_SetMode\n
  *         CCMR1        OC2M          LL_TIM_OC_SetMode\n
  *         CCMR2        OC3M          LL_TIM_OC_SetMode\n
  *         CCMR2        OC4M          LL_TIM_OC_SetMode
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @param  Mode This parameter can be one of the following values:
  *         @arg @ref LL_TIM_OCMODE_FROZEN
  *         @arg @ref LL_TIM_OCMODE_ACTIVE
  *         @arg @ref LL_TIM_OCMODE_INACTIVE
  *         @arg @ref LL_TIM_OCMODE_TOGGLE
  *         @arg @ref LL_TIM_OCMODE_FORCED_INACTIVE
  *         @arg @ref LL_TIM_OCMODE_FORCED_ACTIVE
  *         @arg @ref LL_TIM_OCMODE_PWM1
  *         @arg @ref LL_TIM_OCMODE_PWM2
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_SetMode(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t Mode)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  MODIFY_REG(*pReg, ((TIM_CCMR1_OC1M  | TIM_CCMR1_CC1S) << SHIFT_TAB_OCxx[iChannel]), Mode << SHIFT_TAB_OCxx[iChannel]);
}

/**
  * @brief  Get the output compare mode of an output channel.
  * @rmtoll CCMR1        OC1M          LL_TIM_OC_GetMode\n
  *         CCMR1        OC2M          LL_TIM_OC_GetMode\n
  *         CCMR2        OC3M          LL_TIM_OC_GetMode\n
  *         CCMR2        OC4M          LL_TIM_OC_GetMode
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_OCMODE_FROZEN
  *         @arg @ref LL_TIM_OCMODE_ACTIVE
  *         @arg @ref LL_TIM_OCMODE_INACTIVE
  *         @arg @ref LL_TIM_OCMODE_TOGGLE
  *         @arg @ref LL_TIM_OCMODE_FORCED_INACTIVE
  *         @arg @ref LL_TIM_OCMODE_FORCED_ACTIVE
  *         @arg @ref LL_TIM_OCMODE_PWM1
  *         @arg @ref LL_TIM_OCMODE_PWM2
  */
__STATIC_INLINE uint32_t LL_TIM_OC_GetMode(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  const __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  return (READ_BIT(*pReg, ((TIM_CCMR1_OC1M | TIM_CCMR1_CC1S) << SHIFT_TAB_OCxx[iChannel])) >> SHIFT_TAB_OCxx[iChannel]);
}

/**
  * @brief  Set the polarity of an output channel.
  * @rmtoll CCER         CC1P          LL_TIM_OC_SetPolarity\n
  *         CCER         CC1NP         LL_TIM_OC_SetPolarity\n
  *         CCER         CC2P          LL_TIM_OC_SetPolarity\n
  *         CCER         CC2NP         LL_TIM_OC_SetPolarity\n
  *         CCER         CC3P          LL_TIM_OC_SetPolarity\n
  *         CCER         CC3NP         LL_TIM_OC_SetPolarity\n
  *         CCER         CC4P          LL_TIM_OC_SetPolarity
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH1N
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH2N
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH3N
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @param  Polarity This parameter can be one of the following values:
  *         @arg @ref LL_TIM_OCPOLARITY_HIGH
  *         @arg @ref LL_TIM_OCPOLARITY_LOW
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_SetPolarity(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t Polarity)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  MODIFY_REG(TIMx->CCER, (TIM_CCER_CC1P << SHIFT_TAB_CCxP[iChannel]),  Polarity << SHIFT_TAB_CCxP[iChannel]);
}

/**
  * @brief  Get the polarity of an output channel.
  * @rmtoll CCER         CC1P          LL_TIM_OC_GetPolarity\n
  *         CCER         CC1NP         LL_TIM_OC_GetPolarity\n
  *         CCER         CC2P          LL_TIM_OC_GetPolarity\n
  *         CCER         CC2NP         LL_TIM_OC_GetPolarity\n
  *         CCER         CC3P          LL_TIM_OC_GetPolarity\n
  *         CCER         CC3NP         LL_TIM_OC_GetPolarity\n
  *         CCER         CC4P          LL_TIM_OC_GetPolarity
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH1N
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH2N
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH3N
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_OCPOLARITY_HIGH
  *         @arg @ref LL_TIM_OCPOLARITY_LOW
  */
__STATIC_INLINE uint32_t LL_TIM_OC_GetPolarity(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  return (READ_BIT(TIMx->CCER, (TIM_CCER_CC1P << SHIFT_TAB_CCxP[iChannel])) >> SHIFT_TAB_CCxP[iChannel]);
}

/**
  * @brief  Set the IDLE state of an output channel
  * @note This function is significant only for the timer instances
  *       supporting the break feature. Macro IS_TIM_BREAK_INSTANCE(TIMx)
  *       can be used to check whether or not a timer instance provides
  *       a break input.
  * @rmtoll CR2         OIS1          LL_TIM_OC_SetIdleState\n
  *         CR2         OIS1N         LL_TIM_OC_SetIdleState\n
  *         CR2         OIS2          LL_TIM_OC_SetIdleState\n
  *         CR2         OIS2N         LL_TIM_OC_SetIdleState\n
  *         CR2         OIS3          LL_TIM_OC_SetIdleState\n
  *         CR2         OIS3N         LL_TIM_OC_SetIdleState\n
  *         CR2         OIS4          LL_TIM_OC_SetIdleState
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH1N
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH2N
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH3N
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @param  IdleState This parameter can be one of the following values:
  *         @arg @ref LL_TIM_OCIDLESTATE_LOW
  *         @arg @ref LL_TIM_OCIDLESTATE_HIGH
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_SetIdleState(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t IdleState)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  MODIFY_REG(TIMx->CR2, (TIM_CR2_OIS1 << SHIFT_TAB_OISx[iChannel]),  IdleState << SHIFT_TAB_OISx[iChannel]);
}

/**
  * @brief  Get the IDLE state of an output channel
  * @rmtoll CR2         OIS1          LL_TIM_OC_GetIdleState\n
  *         CR2         OIS1N         LL_TIM_OC_GetIdleState\n
  *         CR2         OIS2          LL_TIM_OC_GetIdleState\n
  *         CR2         OIS2N         LL_TIM_OC_GetIdleState\n
  *         CR2         OIS3          LL_TIM_OC_GetIdleState\n
  *         CR2         OIS3N         LL_TIM_OC_GetIdleState\n
  *         CR2         OIS4          LL_TIM_OC_GetIdleState
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH1N
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH2N
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH3N
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_OCIDLESTATE_LOW
  *         @arg @ref LL_TIM_OCIDLESTATE_HIGH
  */
__STATIC_INLINE uint32_t LL_TIM_OC_GetIdleState(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  return (READ_BIT(TIMx->CR2, (TIM_CR2_OIS1 << SHIFT_TAB_OISx[iChannel])) >> SHIFT_TAB_OISx[iChannel]);
}

/**
  * @brief  Enable fast mode for the output channel.
  * @note Acts only if the channel is configured in PWM1 or PWM2 mode.
  * @rmtoll CCMR1        OC1FE          LL_TIM_OC_EnableFast\n
  *         CCMR1        OC2FE          LL_TIM_OC_EnableFast\n
  *         CCMR2        OC3FE          LL_TIM_OC_EnableFast\n
  *         CCMR2        OC4FE          LL_TIM_OC_EnableFast
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_EnableFast(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  SET_BIT(*pReg, (TIM_CCMR1_OC1FE << SHIFT_TAB_OCxx[iChannel]));

}

/**
  * @brief  Disable fast mode for the output channel.
  * @rmtoll CCMR1        OC1FE          LL_TIM_OC_DisableFast\n
  *         CCMR1        OC2FE          LL_TIM_OC_DisableFast\n
  *         CCMR2        OC3FE          LL_TIM_OC_DisableFast\n
  *         CCMR2        OC4FE          LL_TIM_OC_DisableFast
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_DisableFast(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  CLEAR_BIT(*pReg, (TIM_CCMR1_OC1FE << SHIFT_TAB_OCxx[iChannel]));

}

/**
  * @brief  Indicates whether fast mode is enabled for the output channel.
  * @rmtoll CCMR1        OC1FE          LL_TIM_OC_IsEnabledFast\n
  *         CCMR1        OC2FE          LL_TIM_OC_IsEnabledFast\n
  *         CCMR2        OC3FE          LL_TIM_OC_IsEnabledFast\n
  *         CCMR2        OC4FE          LL_TIM_OC_IsEnabledFast\n
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_OC_IsEnabledFast(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  const __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  uint32_t bitfield = TIM_CCMR1_OC1FE << SHIFT_TAB_OCxx[iChannel];
  return ((READ_BIT(*pReg, bitfield) == bitfield) ? 1UL : 0UL);
}

/**
  * @brief  Enable compare register (TIMx_CCRx) preload for the output channel.
  * @rmtoll CCMR1        OC1PE          LL_TIM_OC_EnablePreload\n
  *         CCMR1        OC2PE          LL_TIM_OC_EnablePreload\n
  *         CCMR2        OC3PE          LL_TIM_OC_EnablePreload\n
  *         CCMR2        OC4PE          LL_TIM_OC_EnablePreload
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_EnablePreload(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  SET_BIT(*pReg, (TIM_CCMR1_OC1PE << SHIFT_TAB_OCxx[iChannel]));
}

/**
  * @brief  Disable compare register (TIMx_CCRx) preload for the output channel.
  * @rmtoll CCMR1        OC1PE          LL_TIM_OC_DisablePreload\n
  *         CCMR1        OC2PE          LL_TIM_OC_DisablePreload\n
  *         CCMR2        OC3PE          LL_TIM_OC_DisablePreload\n
  *         CCMR2        OC4PE          LL_TIM_OC_DisablePreload
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_DisablePreload(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  CLEAR_BIT(*pReg, (TIM_CCMR1_OC1PE << SHIFT_TAB_OCxx[iChannel]));
}

/**
  * @brief  Indicates whether compare register (TIMx_CCRx) preload is enabled for the output channel.
  * @rmtoll CCMR1        OC1PE          LL_TIM_OC_IsEnabledPreload\n
  *         CCMR1        OC2PE          LL_TIM_OC_IsEnabledPreload\n
  *         CCMR2        OC3PE          LL_TIM_OC_IsEnabledPreload\n
  *         CCMR2        OC4PE          LL_TIM_OC_IsEnabledPreload\n
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_OC_IsEnabledPreload(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  const __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  uint32_t bitfield = TIM_CCMR1_OC1PE << SHIFT_TAB_OCxx[iChannel];
  return ((READ_BIT(*pReg, bitfield) == bitfield) ? 1UL : 0UL);
}

/**
  * @brief  Enable clearing the output channel on an external event.
  * @note This function can only be used in Output compare and PWM modes. It does not work in Forced mode.
  * @note Macro IS_TIM_OCXREF_CLEAR_INSTANCE(TIMx) can be used to check whether
  *       or not a timer instance can clear the OCxREF signal on an external event.
  * @rmtoll CCMR1        OC1CE          LL_TIM_OC_EnableClear\n
  *         CCMR1        OC2CE          LL_TIM_OC_EnableClear\n
  *         CCMR2        OC3CE          LL_TIM_OC_EnableClear\n
  *         CCMR2        OC4CE          LL_TIM_OC_EnableClear
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_EnableClear(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  SET_BIT(*pReg, (TIM_CCMR1_OC1CE << SHIFT_TAB_OCxx[iChannel]));
}

/**
  * @brief  Disable clearing the output channel on an external event.
  * @note Macro IS_TIM_OCXREF_CLEAR_INSTANCE(TIMx) can be used to check whether
  *       or not a timer instance can clear the OCxREF signal on an external event.
  * @rmtoll CCMR1        OC1CE          LL_TIM_OC_DisableClear\n
  *         CCMR1        OC2CE          LL_TIM_OC_DisableClear\n
  *         CCMR2        OC3CE          LL_TIM_OC_DisableClear\n
  *         CCMR2        OC4CE          LL_TIM_OC_DisableClear
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_DisableClear(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  CLEAR_BIT(*pReg, (TIM_CCMR1_OC1CE << SHIFT_TAB_OCxx[iChannel]));
}

/**
  * @brief  Indicates clearing the output channel on an external event is enabled for the output channel.
  * @note This function enables clearing the output channel on an external event.
  * @note This function can only be used in Output compare and PWM modes. It does not work in Forced mode.
  * @note Macro IS_TIM_OCXREF_CLEAR_INSTANCE(TIMx) can be used to check whether
  *       or not a timer instance can clear the OCxREF signal on an external event.
  * @rmtoll CCMR1        OC1CE          LL_TIM_OC_IsEnabledClear\n
  *         CCMR1        OC2CE          LL_TIM_OC_IsEnabledClear\n
  *         CCMR2        OC3CE          LL_TIM_OC_IsEnabledClear\n
  *         CCMR2        OC4CE          LL_TIM_OC_IsEnabledClear\n
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_OC_IsEnabledClear(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  const __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  uint32_t bitfield = TIM_CCMR1_OC1CE << SHIFT_TAB_OCxx[iChannel];
  return ((READ_BIT(*pReg, bitfield) == bitfield) ? 1UL : 0UL);
}

/**
  * @brief  Set the dead-time delay (delay inserted between the rising edge of the OCxREF signal and the rising edge of
  *         the Ocx and OCxN signals).
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       dead-time insertion feature is supported by a timer instance.
  * @note Helper macro @ref __LL_TIM_CALC_DEADTIME can be used to calculate the DeadTime parameter
  * @rmtoll BDTR         DTG           LL_TIM_OC_SetDeadTime
  * @param  TIMx Timer instance
  * @param  DeadTime between Min_Data=0 and Max_Data=255
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_SetDeadTime(TIM_TypeDef *TIMx, uint32_t DeadTime)
{
  MODIFY_REG(TIMx->BDTR, TIM_BDTR_DTG, DeadTime);
}

/**
  * @brief  Set compare value for output channel 1 (TIMx_CCR1).
  * @note In 32-bit timer implementations compare value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC1_INSTANCE(TIMx) can be used to check whether or not
  *       output channel 1 is supported by a timer instance.
  * @rmtoll CCR1         CCR1          LL_TIM_OC_SetCompareCH1
  * @param  TIMx Timer instance
  * @param  CompareValue between Min_Data=0 and Max_Data=65535
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_SetCompareCH1(TIM_TypeDef *TIMx, uint32_t CompareValue)
{
  WRITE_REG(TIMx->CCR1, CompareValue);
}

/**
  * @brief  Set compare value for output channel 2 (TIMx_CCR2).
  * @note In 32-bit timer implementations compare value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC2_INSTANCE(TIMx) can be used to check whether or not
  *       output channel 2 is supported by a timer instance.
  * @rmtoll CCR2         CCR2          LL_TIM_OC_SetCompareCH2
  * @param  TIMx Timer instance
  * @param  CompareValue between Min_Data=0 and Max_Data=65535
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_SetCompareCH2(TIM_TypeDef *TIMx, uint32_t CompareValue)
{
  WRITE_REG(TIMx->CCR2, CompareValue);
}

/**
  * @brief  Set compare value for output channel 3 (TIMx_CCR3).
  * @note In 32-bit timer implementations compare value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC3_INSTANCE(TIMx) can be used to check whether or not
  *       output channel is supported by a timer instance.
  * @rmtoll CCR3         CCR3          LL_TIM_OC_SetCompareCH3
  * @param  TIMx Timer instance
  * @param  CompareValue between Min_Data=0 and Max_Data=65535
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_SetCompareCH3(TIM_TypeDef *TIMx, uint32_t CompareValue)
{
  WRITE_REG(TIMx->CCR3, CompareValue);
}

/**
  * @brief  Set compare value for output channel 4 (TIMx_CCR4).
  * @note In 32-bit timer implementations compare value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC4_INSTANCE(TIMx) can be used to check whether or not
  *       output channel 4 is supported by a timer instance.
  * @rmtoll CCR4         CCR4          LL_TIM_OC_SetCompareCH4
  * @param  TIMx Timer instance
  * @param  CompareValue between Min_Data=0 and Max_Data=65535
  * @retval None
  */
__STATIC_INLINE void LL_TIM_OC_SetCompareCH4(TIM_TypeDef *TIMx, uint32_t CompareValue)
{
  WRITE_REG(TIMx->CCR4, CompareValue);
}

/**
  * @brief  Get compare value (TIMx_CCR1) set for  output channel 1.
  * @note In 32-bit timer implementations returned compare value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC1_INSTANCE(TIMx) can be used to check whether or not
  *       output channel 1 is supported by a timer instance.
  * @rmtoll CCR1         CCR1          LL_TIM_OC_GetCompareCH1
  * @param  TIMx Timer instance
  * @retval CompareValue (between Min_Data=0 and Max_Data=65535)
  */
__STATIC_INLINE uint32_t LL_TIM_OC_GetCompareCH1(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->CCR1));
}

/**
  * @brief  Get compare value (TIMx_CCR2) set for  output channel 2.
  * @note In 32-bit timer implementations returned compare value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC2_INSTANCE(TIMx) can be used to check whether or not
  *       output channel 2 is supported by a timer instance.
  * @rmtoll CCR2         CCR2          LL_TIM_OC_GetCompareCH2
  * @param  TIMx Timer instance
  * @retval CompareValue (between Min_Data=0 and Max_Data=65535)
  */
__STATIC_INLINE uint32_t LL_TIM_OC_GetCompareCH2(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->CCR2));
}

/**
  * @brief  Get compare value (TIMx_CCR3) set for  output channel 3.
  * @note In 32-bit timer implementations returned compare value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC3_INSTANCE(TIMx) can be used to check whether or not
  *       output channel 3 is supported by a timer instance.
  * @rmtoll CCR3         CCR3          LL_TIM_OC_GetCompareCH3
  * @param  TIMx Timer instance
  * @retval CompareValue (between Min_Data=0 and Max_Data=65535)
  */
__STATIC_INLINE uint32_t LL_TIM_OC_GetCompareCH3(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->CCR3));
}

/**
  * @brief  Get compare value (TIMx_CCR4) set for  output channel 4.
  * @note In 32-bit timer implementations returned compare value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC4_INSTANCE(TIMx) can be used to check whether or not
  *       output channel 4 is supported by a timer instance.
  * @rmtoll CCR4         CCR4          LL_TIM_OC_GetCompareCH4
  * @param  TIMx Timer instance
  * @retval CompareValue (between Min_Data=0 and Max_Data=65535)
  */
__STATIC_INLINE uint32_t LL_TIM_OC_GetCompareCH4(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->CCR4));
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_Input_Channel Input channel configuration
  * @{
  */
/**
  * @brief  Configure input channel.
  * @rmtoll CCMR1        CC1S          LL_TIM_IC_Config\n
  *         CCMR1        IC1PSC        LL_TIM_IC_Config\n
  *         CCMR1        IC1F          LL_TIM_IC_Config\n
  *         CCMR1        CC2S          LL_TIM_IC_Config\n
  *         CCMR1        IC2PSC        LL_TIM_IC_Config\n
  *         CCMR1        IC2F          LL_TIM_IC_Config\n
  *         CCMR2        CC3S          LL_TIM_IC_Config\n
  *         CCMR2        IC3PSC        LL_TIM_IC_Config\n
  *         CCMR2        IC3F          LL_TIM_IC_Config\n
  *         CCMR2        CC4S          LL_TIM_IC_Config\n
  *         CCMR2        IC4PSC        LL_TIM_IC_Config\n
  *         CCMR2        IC4F          LL_TIM_IC_Config\n
  *         CCER         CC1P          LL_TIM_IC_Config\n
  *         CCER         CC1NP         LL_TIM_IC_Config\n
  *         CCER         CC2P          LL_TIM_IC_Config\n
  *         CCER         CC2NP         LL_TIM_IC_Config\n
  *         CCER         CC3P          LL_TIM_IC_Config\n
  *         CCER         CC3NP         LL_TIM_IC_Config\n
  *         CCER         CC4P          LL_TIM_IC_Config\n
  *         CCER         CC4NP         LL_TIM_IC_Config
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @param  Configuration This parameter must be a combination of all the following values:
  *         @arg @ref LL_TIM_ACTIVEINPUT_DIRECTTI or @ref LL_TIM_ACTIVEINPUT_INDIRECTTI or @ref LL_TIM_ACTIVEINPUT_TRC
  *         @arg @ref LL_TIM_ICPSC_DIV1 or ... or @ref LL_TIM_ICPSC_DIV8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV1 or ... or @ref LL_TIM_IC_FILTER_FDIV32_N8
  *         @arg @ref LL_TIM_IC_POLARITY_RISING or @ref LL_TIM_IC_POLARITY_FALLING or @ref LL_TIM_IC_POLARITY_BOTHEDGE
  * @retval None
  */
__STATIC_INLINE void LL_TIM_IC_Config(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t Configuration)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  MODIFY_REG(*pReg, ((TIM_CCMR1_IC1F | TIM_CCMR1_IC1PSC | TIM_CCMR1_CC1S) << SHIFT_TAB_ICxx[iChannel]),
             ((Configuration >> 16U) & (TIM_CCMR1_IC1F | TIM_CCMR1_IC1PSC | TIM_CCMR1_CC1S))                \
             << SHIFT_TAB_ICxx[iChannel]);
  MODIFY_REG(TIMx->CCER, ((TIM_CCER_CC1NP | TIM_CCER_CC1P) << SHIFT_TAB_CCxP[iChannel]),
             (Configuration & (TIM_CCER_CC1NP | TIM_CCER_CC1P)) << SHIFT_TAB_CCxP[iChannel]);
}

/**
  * @brief  Set the active input.
  * @rmtoll CCMR1        CC1S          LL_TIM_IC_SetActiveInput\n
  *         CCMR1        CC2S          LL_TIM_IC_SetActiveInput\n
  *         CCMR2        CC3S          LL_TIM_IC_SetActiveInput\n
  *         CCMR2        CC4S          LL_TIM_IC_SetActiveInput
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @param  ICActiveInput This parameter can be one of the following values:
  *         @arg @ref LL_TIM_ACTIVEINPUT_DIRECTTI
  *         @arg @ref LL_TIM_ACTIVEINPUT_INDIRECTTI
  *         @arg @ref LL_TIM_ACTIVEINPUT_TRC
  * @retval None
  */
__STATIC_INLINE void LL_TIM_IC_SetActiveInput(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t ICActiveInput)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  MODIFY_REG(*pReg, ((TIM_CCMR1_CC1S) << SHIFT_TAB_ICxx[iChannel]), (ICActiveInput >> 16U) << SHIFT_TAB_ICxx[iChannel]);
}

/**
  * @brief  Get the current active input.
  * @rmtoll CCMR1        CC1S          LL_TIM_IC_GetActiveInput\n
  *         CCMR1        CC2S          LL_TIM_IC_GetActiveInput\n
  *         CCMR2        CC3S          LL_TIM_IC_GetActiveInput\n
  *         CCMR2        CC4S          LL_TIM_IC_GetActiveInput
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_ACTIVEINPUT_DIRECTTI
  *         @arg @ref LL_TIM_ACTIVEINPUT_INDIRECTTI
  *         @arg @ref LL_TIM_ACTIVEINPUT_TRC
  */
__STATIC_INLINE uint32_t LL_TIM_IC_GetActiveInput(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  const __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  return ((READ_BIT(*pReg, ((TIM_CCMR1_CC1S) << SHIFT_TAB_ICxx[iChannel])) >> SHIFT_TAB_ICxx[iChannel]) << 16U);
}

/**
  * @brief  Set the prescaler of input channel.
  * @rmtoll CCMR1        IC1PSC        LL_TIM_IC_SetPrescaler\n
  *         CCMR1        IC2PSC        LL_TIM_IC_SetPrescaler\n
  *         CCMR2        IC3PSC        LL_TIM_IC_SetPrescaler\n
  *         CCMR2        IC4PSC        LL_TIM_IC_SetPrescaler
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @param  ICPrescaler This parameter can be one of the following values:
  *         @arg @ref LL_TIM_ICPSC_DIV1
  *         @arg @ref LL_TIM_ICPSC_DIV2
  *         @arg @ref LL_TIM_ICPSC_DIV4
  *         @arg @ref LL_TIM_ICPSC_DIV8
  * @retval None
  */
__STATIC_INLINE void LL_TIM_IC_SetPrescaler(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t ICPrescaler)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  MODIFY_REG(*pReg, ((TIM_CCMR1_IC1PSC) << SHIFT_TAB_ICxx[iChannel]), (ICPrescaler >> 16U) << SHIFT_TAB_ICxx[iChannel]);
}

/**
  * @brief  Get the current prescaler value acting on an  input channel.
  * @rmtoll CCMR1        IC1PSC        LL_TIM_IC_GetPrescaler\n
  *         CCMR1        IC2PSC        LL_TIM_IC_GetPrescaler\n
  *         CCMR2        IC3PSC        LL_TIM_IC_GetPrescaler\n
  *         CCMR2        IC4PSC        LL_TIM_IC_GetPrescaler
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_ICPSC_DIV1
  *         @arg @ref LL_TIM_ICPSC_DIV2
  *         @arg @ref LL_TIM_ICPSC_DIV4
  *         @arg @ref LL_TIM_ICPSC_DIV8
  */
__STATIC_INLINE uint32_t LL_TIM_IC_GetPrescaler(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  const __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  return ((READ_BIT(*pReg, ((TIM_CCMR1_IC1PSC) << SHIFT_TAB_ICxx[iChannel])) >> SHIFT_TAB_ICxx[iChannel]) << 16U);
}

/**
  * @brief  Set the input filter duration.
  * @rmtoll CCMR1        IC1F          LL_TIM_IC_SetFilter\n
  *         CCMR1        IC2F          LL_TIM_IC_SetFilter\n
  *         CCMR2        IC3F          LL_TIM_IC_SetFilter\n
  *         CCMR2        IC4F          LL_TIM_IC_SetFilter
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @param  ICFilter This parameter can be one of the following values:
  *         @arg @ref LL_TIM_IC_FILTER_FDIV1
  *         @arg @ref LL_TIM_IC_FILTER_FDIV1_N2
  *         @arg @ref LL_TIM_IC_FILTER_FDIV1_N4
  *         @arg @ref LL_TIM_IC_FILTER_FDIV1_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV2_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV2_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV4_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV4_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV8_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV8_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV16_N5
  *         @arg @ref LL_TIM_IC_FILTER_FDIV16_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV16_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV32_N5
  *         @arg @ref LL_TIM_IC_FILTER_FDIV32_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV32_N8
  * @retval None
  */
__STATIC_INLINE void LL_TIM_IC_SetFilter(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t ICFilter)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  MODIFY_REG(*pReg, ((TIM_CCMR1_IC1F) << SHIFT_TAB_ICxx[iChannel]), (ICFilter >> 16U) << SHIFT_TAB_ICxx[iChannel]);
}

/**
  * @brief  Get the input filter duration.
  * @rmtoll CCMR1        IC1F          LL_TIM_IC_GetFilter\n
  *         CCMR1        IC2F          LL_TIM_IC_GetFilter\n
  *         CCMR2        IC3F          LL_TIM_IC_GetFilter\n
  *         CCMR2        IC4F          LL_TIM_IC_GetFilter
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_IC_FILTER_FDIV1
  *         @arg @ref LL_TIM_IC_FILTER_FDIV1_N2
  *         @arg @ref LL_TIM_IC_FILTER_FDIV1_N4
  *         @arg @ref LL_TIM_IC_FILTER_FDIV1_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV2_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV2_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV4_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV4_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV8_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV8_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV16_N5
  *         @arg @ref LL_TIM_IC_FILTER_FDIV16_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV16_N8
  *         @arg @ref LL_TIM_IC_FILTER_FDIV32_N5
  *         @arg @ref LL_TIM_IC_FILTER_FDIV32_N6
  *         @arg @ref LL_TIM_IC_FILTER_FDIV32_N8
  */
__STATIC_INLINE uint32_t LL_TIM_IC_GetFilter(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  const __IO uint32_t *pReg = (__IO uint32_t *)((uint32_t)((uint32_t)(&TIMx->CCMR1) + OFFSET_TAB_CCMRx[iChannel]));
  return ((READ_BIT(*pReg, ((TIM_CCMR1_IC1F) << SHIFT_TAB_ICxx[iChannel])) >> SHIFT_TAB_ICxx[iChannel]) << 16U);
}

/**
  * @brief  Set the input channel polarity.
  * @rmtoll CCER         CC1P          LL_TIM_IC_SetPolarity\n
  *         CCER         CC1NP         LL_TIM_IC_SetPolarity\n
  *         CCER         CC2P          LL_TIM_IC_SetPolarity\n
  *         CCER         CC2NP         LL_TIM_IC_SetPolarity\n
  *         CCER         CC3P          LL_TIM_IC_SetPolarity\n
  *         CCER         CC3NP         LL_TIM_IC_SetPolarity\n
  *         CCER         CC4P          LL_TIM_IC_SetPolarity\n
  *         CCER         CC4NP         LL_TIM_IC_SetPolarity
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @param  ICPolarity This parameter can be one of the following values:
  *         @arg @ref LL_TIM_IC_POLARITY_RISING
  *         @arg @ref LL_TIM_IC_POLARITY_FALLING
  *         @arg @ref LL_TIM_IC_POLARITY_BOTHEDGE
  * @retval None
  */
__STATIC_INLINE void LL_TIM_IC_SetPolarity(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t ICPolarity)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  MODIFY_REG(TIMx->CCER, ((TIM_CCER_CC1NP | TIM_CCER_CC1P) << SHIFT_TAB_CCxP[iChannel]),
             ICPolarity << SHIFT_TAB_CCxP[iChannel]);
}

/**
  * @brief  Get the current input channel polarity.
  * @rmtoll CCER         CC1P          LL_TIM_IC_GetPolarity\n
  *         CCER         CC1NP         LL_TIM_IC_GetPolarity\n
  *         CCER         CC2P          LL_TIM_IC_GetPolarity\n
  *         CCER         CC2NP         LL_TIM_IC_GetPolarity\n
  *         CCER         CC3P          LL_TIM_IC_GetPolarity\n
  *         CCER         CC3NP         LL_TIM_IC_GetPolarity\n
  *         CCER         CC4P          LL_TIM_IC_GetPolarity\n
  *         CCER         CC4NP         LL_TIM_IC_GetPolarity
  * @param  TIMx Timer instance
  * @param  Channel This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CHANNEL_CH1
  *         @arg @ref LL_TIM_CHANNEL_CH2
  *         @arg @ref LL_TIM_CHANNEL_CH3
  *         @arg @ref LL_TIM_CHANNEL_CH4
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_TIM_IC_POLARITY_RISING
  *         @arg @ref LL_TIM_IC_POLARITY_FALLING
  *         @arg @ref LL_TIM_IC_POLARITY_BOTHEDGE
  */
__STATIC_INLINE uint32_t LL_TIM_IC_GetPolarity(TIM_TypeDef *TIMx, uint32_t Channel)
{
  uint8_t iChannel = TIM_GET_CHANNEL_INDEX(Channel);
  return (READ_BIT(TIMx->CCER, ((TIM_CCER_CC1NP | TIM_CCER_CC1P) << SHIFT_TAB_CCxP[iChannel])) >>
          SHIFT_TAB_CCxP[iChannel]);
}

/**
  * @brief  Connect the TIMx_CH1, CH2 and CH3 pins  to the TI1 input (XOR combination).
  * @note Macro IS_TIM_XOR_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides an XOR input.
  * @rmtoll CR2          TI1S          LL_TIM_IC_EnableXORCombination
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_IC_EnableXORCombination(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->CR2, TIM_CR2_TI1S);
}

/**
  * @brief  Disconnect the TIMx_CH1, CH2 and CH3 pins  from the TI1 input.
  * @note Macro IS_TIM_XOR_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides an XOR input.
  * @rmtoll CR2          TI1S          LL_TIM_IC_DisableXORCombination
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_IC_DisableXORCombination(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->CR2, TIM_CR2_TI1S);
}

/**
  * @brief  Indicates whether the TIMx_CH1, CH2 and CH3 pins are connectected to the TI1 input.
  * @note Macro IS_TIM_XOR_INSTANCE(TIMx) can be used to check whether or not
  * a timer instance provides an XOR input.
  * @rmtoll CR2          TI1S          LL_TIM_IC_IsEnabledXORCombination
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IC_IsEnabledXORCombination(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->CR2, TIM_CR2_TI1S) == (TIM_CR2_TI1S)) ? 1UL : 0UL);
}

/**
  * @brief  Get captured value for input channel 1.
  * @note In 32-bit timer implementations returned captured value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC1_INSTANCE(TIMx) can be used to check whether or not
  *       input channel 1 is supported by a timer instance.
  * @rmtoll CCR1         CCR1          LL_TIM_IC_GetCaptureCH1
  * @param  TIMx Timer instance
  * @retval CapturedValue (between Min_Data=0 and Max_Data=65535)
  */
__STATIC_INLINE uint32_t LL_TIM_IC_GetCaptureCH1(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->CCR1));
}

/**
  * @brief  Get captured value for input channel 2.
  * @note In 32-bit timer implementations returned captured value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC2_INSTANCE(TIMx) can be used to check whether or not
  *       input channel 2 is supported by a timer instance.
  * @rmtoll CCR2         CCR2          LL_TIM_IC_GetCaptureCH2
  * @param  TIMx Timer instance
  * @retval CapturedValue (between Min_Data=0 and Max_Data=65535)
  */
__STATIC_INLINE uint32_t LL_TIM_IC_GetCaptureCH2(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->CCR2));
}

/**
  * @brief  Get captured value for input channel 3.
  * @note In 32-bit timer implementations returned captured value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC3_INSTANCE(TIMx) can be used to check whether or not
  *       input channel 3 is supported by a timer instance.
  * @rmtoll CCR3         CCR3          LL_TIM_IC_GetCaptureCH3
  * @param  TIMx Timer instance
  * @retval CapturedValue (between Min_Data=0 and Max_Data=65535)
  */
__STATIC_INLINE uint32_t LL_TIM_IC_GetCaptureCH3(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->CCR3));
}

/**
  * @brief  Get captured value for input channel 4.
  * @note In 32-bit timer implementations returned captured value can be between 0x00000000 and 0xFFFFFFFF.
  * @note Macro IS_TIM_32B_COUNTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports a 32 bits counter.
  * @note Macro IS_TIM_CC4_INSTANCE(TIMx) can be used to check whether or not
  *       input channel 4 is supported by a timer instance.
  * @rmtoll CCR4         CCR4          LL_TIM_IC_GetCaptureCH4
  * @param  TIMx Timer instance
  * @retval CapturedValue (between Min_Data=0 and Max_Data=65535)
  */
__STATIC_INLINE uint32_t LL_TIM_IC_GetCaptureCH4(TIM_TypeDef *TIMx)
{
  return (uint32_t)(READ_REG(TIMx->CCR4));
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_Clock_Selection Counter clock selection
  * @{
  */
/**
  * @brief  Enable external clock mode 2.
  * @note When external clock mode 2 is enabled the counter is clocked by any active edge on the ETRF signal.
  * @note Macro IS_TIM_CLOCKSOURCE_ETRMODE2_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports external clock mode2.
  * @rmtoll SMCR         ECE           LL_TIM_EnableExternalClock
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableExternalClock(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->SMCR, TIM_SMCR_ECE);
}

/**
  * @brief  Disable external clock mode 2.
  * @note Macro IS_TIM_CLOCKSOURCE_ETRMODE2_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports external clock mode2.
  * @rmtoll SMCR         ECE           LL_TIM_DisableExternalClock
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableExternalClock(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->SMCR, TIM_SMCR_ECE);
}

/**
  * @brief  Indicate whether external clock mode 2 is enabled.
  * @note Macro IS_TIM_CLOCKSOURCE_ETRMODE2_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports external clock mode2.
  * @rmtoll SMCR         ECE           LL_TIM_IsEnabledExternalClock
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledExternalClock(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SMCR, TIM_SMCR_ECE) == (TIM_SMCR_ECE)) ? 1UL : 0UL);
}

/**
  * @brief  Set the clock source of the counter clock.
  * @note when selected clock source is external clock mode 1, the timer input
  *       the external clock is applied is selected by calling the @ref LL_TIM_SetTriggerInput()
  *       function. This timer input must be configured by calling
  *       the @ref LL_TIM_IC_Config() function.
  * @note Macro IS_TIM_CLOCKSOURCE_ETRMODE1_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports external clock mode1.
  * @note Macro IS_TIM_CLOCKSOURCE_ETRMODE2_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports external clock mode2.
  * @rmtoll SMCR         SMS           LL_TIM_SetClockSource\n
  *         SMCR         ECE           LL_TIM_SetClockSource
  * @param  TIMx Timer instance
  * @param  ClockSource This parameter can be one of the following values:
  *         @arg @ref LL_TIM_CLOCKSOURCE_INTERNAL
  *         @arg @ref LL_TIM_CLOCKSOURCE_EXT_MODE1
  *         @arg @ref LL_TIM_CLOCKSOURCE_EXT_MODE2
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetClockSource(TIM_TypeDef *TIMx, uint32_t ClockSource)
{
  MODIFY_REG(TIMx->SMCR, TIM_SMCR_SMS | TIM_SMCR_ECE, ClockSource);
}

/**
  * @brief  Set the encoder interface mode.
  * @note Macro IS_TIM_ENCODER_INTERFACE_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance supports the encoder mode.
  * @rmtoll SMCR         SMS           LL_TIM_SetEncoderMode
  * @param  TIMx Timer instance
  * @param  EncoderMode This parameter can be one of the following values:
  *         @arg @ref LL_TIM_ENCODERMODE_X2_TI1
  *         @arg @ref LL_TIM_ENCODERMODE_X2_TI2
  *         @arg @ref LL_TIM_ENCODERMODE_X4_TI12
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetEncoderMode(TIM_TypeDef *TIMx, uint32_t EncoderMode)
{
  MODIFY_REG(TIMx->SMCR, TIM_SMCR_SMS, EncoderMode);
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_Timer_Synchronization Timer synchronisation configuration
  * @{
  */
/**
  * @brief  Set the trigger output (TRGO) used for timer synchronization .
  * @note Macro IS_TIM_MASTER_INSTANCE(TIMx) can be used to check
  *       whether or not a timer instance can operate as a master timer.
  * @rmtoll CR2          MMS           LL_TIM_SetTriggerOutput
  * @param  TIMx Timer instance
  * @param  TimerSynchronization This parameter can be one of the following values:
  *         @arg @ref LL_TIM_TRGO_RESET
  *         @arg @ref LL_TIM_TRGO_ENABLE
  *         @arg @ref LL_TIM_TRGO_UPDATE
  *         @arg @ref LL_TIM_TRGO_CC1IF
  *         @arg @ref LL_TIM_TRGO_OC1REF
  *         @arg @ref LL_TIM_TRGO_OC2REF
  *         @arg @ref LL_TIM_TRGO_OC3REF
  *         @arg @ref LL_TIM_TRGO_OC4REF
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetTriggerOutput(TIM_TypeDef *TIMx, uint32_t TimerSynchronization)
{
  MODIFY_REG(TIMx->CR2, TIM_CR2_MMS, TimerSynchronization);
}

/**
  * @brief  Set the synchronization mode of a slave timer.
  * @note Macro IS_TIM_SLAVE_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance can operate as a slave timer.
  * @rmtoll SMCR         SMS           LL_TIM_SetSlaveMode
  * @param  TIMx Timer instance
  * @param  SlaveMode This parameter can be one of the following values:
  *         @arg @ref LL_TIM_SLAVEMODE_DISABLED
  *         @arg @ref LL_TIM_SLAVEMODE_RESET
  *         @arg @ref LL_TIM_SLAVEMODE_GATED
  *         @arg @ref LL_TIM_SLAVEMODE_TRIGGER
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetSlaveMode(TIM_TypeDef *TIMx, uint32_t SlaveMode)
{
  MODIFY_REG(TIMx->SMCR, TIM_SMCR_SMS, SlaveMode);
}

/**
  * @brief  Set the selects the trigger input to be used to synchronize the counter.
  * @note Macro IS_TIM_SLAVE_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance can operate as a slave timer.
  * @rmtoll SMCR         TS            LL_TIM_SetTriggerInput
  * @param  TIMx Timer instance
  * @param  TriggerInput This parameter can be one of the following values:
  *         @arg @ref LL_TIM_TS_ITR0
  *         @arg @ref LL_TIM_TS_ITR1
  *         @arg @ref LL_TIM_TS_ITR2
  *         @arg @ref LL_TIM_TS_ITR3
  *         @arg @ref LL_TIM_TS_TI1F_ED
  *         @arg @ref LL_TIM_TS_TI1FP1
  *         @arg @ref LL_TIM_TS_TI2FP2
  *         @arg @ref LL_TIM_TS_ETRF
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetTriggerInput(TIM_TypeDef *TIMx, uint32_t TriggerInput)
{
  MODIFY_REG(TIMx->SMCR, TIM_SMCR_TS, TriggerInput);
}

/**
  * @brief  Enable the Master/Slave mode.
  * @note Macro IS_TIM_SLAVE_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance can operate as a slave timer.
  * @rmtoll SMCR         MSM           LL_TIM_EnableMasterSlaveMode
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableMasterSlaveMode(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->SMCR, TIM_SMCR_MSM);
}

/**
  * @brief  Disable the Master/Slave mode.
  * @note Macro IS_TIM_SLAVE_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance can operate as a slave timer.
  * @rmtoll SMCR         MSM           LL_TIM_DisableMasterSlaveMode
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableMasterSlaveMode(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->SMCR, TIM_SMCR_MSM);
}

/**
  * @brief Indicates whether the Master/Slave mode is enabled.
  * @note Macro IS_TIM_SLAVE_INSTANCE(TIMx) can be used to check whether or not
  * a timer instance can operate as a slave timer.
  * @rmtoll SMCR         MSM           LL_TIM_IsEnabledMasterSlaveMode
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledMasterSlaveMode(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SMCR, TIM_SMCR_MSM) == (TIM_SMCR_MSM)) ? 1UL : 0UL);
}

/**
  * @brief  Configure the external trigger (ETR) input.
  * @note Macro IS_TIM_ETR_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides an external trigger input.
  * @rmtoll SMCR         ETP           LL_TIM_ConfigETR\n
  *         SMCR         ETPS          LL_TIM_ConfigETR\n
  *         SMCR         ETF           LL_TIM_ConfigETR
  * @param  TIMx Timer instance
  * @param  ETRPolarity This parameter can be one of the following values:
  *         @arg @ref LL_TIM_ETR_POLARITY_NONINVERTED
  *         @arg @ref LL_TIM_ETR_POLARITY_INVERTED
  * @param  ETRPrescaler This parameter can be one of the following values:
  *         @arg @ref LL_TIM_ETR_PRESCALER_DIV1
  *         @arg @ref LL_TIM_ETR_PRESCALER_DIV2
  *         @arg @ref LL_TIM_ETR_PRESCALER_DIV4
  *         @arg @ref LL_TIM_ETR_PRESCALER_DIV8
  * @param  ETRFilter This parameter can be one of the following values:
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV1
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV1_N2
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV1_N4
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV1_N8
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV2_N6
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV2_N8
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV4_N6
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV4_N8
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV8_N6
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV8_N8
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV16_N5
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV16_N6
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV16_N8
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV32_N5
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV32_N6
  *         @arg @ref LL_TIM_ETR_FILTER_FDIV32_N8
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ConfigETR(TIM_TypeDef *TIMx, uint32_t ETRPolarity, uint32_t ETRPrescaler,
                                      uint32_t ETRFilter)
{
  MODIFY_REG(TIMx->SMCR, TIM_SMCR_ETP | TIM_SMCR_ETPS | TIM_SMCR_ETF, ETRPolarity | ETRPrescaler | ETRFilter);
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_Break_Function Break function configuration
  * @{
  */
/**
  * @brief  Enable the break function.
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @rmtoll BDTR         BKE           LL_TIM_EnableBRK
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableBRK(TIM_TypeDef *TIMx)
{
  __IO uint32_t tmpreg;
  SET_BIT(TIMx->BDTR, TIM_BDTR_BKE);
  /* Note: Any write operation to this bit takes a delay of 1 APB clock cycle to become effective. */
  tmpreg = READ_REG(TIMx->BDTR);
  (void)(tmpreg);
}

/**
  * @brief  Disable the break function.
  * @rmtoll BDTR         BKE           LL_TIM_DisableBRK
  * @param  TIMx Timer instance
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableBRK(TIM_TypeDef *TIMx)
{
  __IO uint32_t tmpreg;
  CLEAR_BIT(TIMx->BDTR, TIM_BDTR_BKE);
  /* Note: Any write operation to this bit takes a delay of 1 APB clock cycle to become effective. */
  tmpreg = READ_REG(TIMx->BDTR);
  (void)(tmpreg);
}

/**
  * @brief  Configure the break input.
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @rmtoll BDTR         BKP           LL_TIM_ConfigBRK
  * @param  TIMx Timer instance
  * @param  BreakPolarity This parameter can be one of the following values:
  *         @arg @ref LL_TIM_BREAK_POLARITY_LOW
  *         @arg @ref LL_TIM_BREAK_POLARITY_HIGH
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ConfigBRK(TIM_TypeDef *TIMx, uint32_t BreakPolarity)
{
  __IO uint32_t tmpreg;
  MODIFY_REG(TIMx->BDTR, TIM_BDTR_BKP, BreakPolarity);
  /* Note: Any write operation to BKP bit takes a delay of 1 APB clock cycle to become effective. */
  tmpreg = READ_REG(TIMx->BDTR);
  (void)(tmpreg);
}

/**
  * @brief  Select the outputs off state (enabled v.s. disabled) in Idle and Run modes.
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @rmtoll BDTR         OSSI          LL_TIM_SetOffStates\n
  *         BDTR         OSSR          LL_TIM_SetOffStates
  * @param  TIMx Timer instance
  * @param  OffStateIdle This parameter can be one of the following values:
  *         @arg @ref LL_TIM_OSSI_DISABLE
  *         @arg @ref LL_TIM_OSSI_ENABLE
  * @param  OffStateRun This parameter can be one of the following values:
  *         @arg @ref LL_TIM_OSSR_DISABLE
  *         @arg @ref LL_TIM_OSSR_ENABLE
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetOffStates(TIM_TypeDef *TIMx, uint32_t OffStateIdle, uint32_t OffStateRun)
{
  MODIFY_REG(TIMx->BDTR, TIM_BDTR_OSSI | TIM_BDTR_OSSR, OffStateIdle | OffStateRun);
}

/**
  * @brief  Enable automatic output (MOE can be set by software or automatically when a break input is active).
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @rmtoll BDTR         AOE           LL_TIM_EnableAutomaticOutput
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableAutomaticOutput(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->BDTR, TIM_BDTR_AOE);
}

/**
  * @brief  Disable automatic output (MOE can be set only by software).
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @rmtoll BDTR         AOE           LL_TIM_DisableAutomaticOutput
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableAutomaticOutput(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->BDTR, TIM_BDTR_AOE);
}

/**
  * @brief  Indicate whether automatic output is enabled.
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @rmtoll BDTR         AOE           LL_TIM_IsEnabledAutomaticOutput
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledAutomaticOutput(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->BDTR, TIM_BDTR_AOE) == (TIM_BDTR_AOE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable the outputs (set the MOE bit in TIMx_BDTR register).
  * @note The MOE bit in TIMx_BDTR register allows to enable /disable the outputs by
  *       software and is reset in case of break or break2 event
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @rmtoll BDTR         MOE           LL_TIM_EnableAllOutputs
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableAllOutputs(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->BDTR, TIM_BDTR_MOE);
}

/**
  * @brief  Disable the outputs (reset the MOE bit in TIMx_BDTR register).
  * @note The MOE bit in TIMx_BDTR register allows to enable /disable the outputs by
  *       software and is reset in case of break or break2 event.
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @rmtoll BDTR         MOE           LL_TIM_DisableAllOutputs
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableAllOutputs(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->BDTR, TIM_BDTR_MOE);
}

/**
  * @brief  Indicates whether outputs are enabled.
  * @note Macro IS_TIM_BREAK_INSTANCE(TIMx) can be used to check whether or not
  *       a timer instance provides a break input.
  * @rmtoll BDTR         MOE           LL_TIM_IsEnabledAllOutputs
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledAllOutputs(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->BDTR, TIM_BDTR_MOE) == (TIM_BDTR_MOE)) ? 1UL : 0UL);
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_DMA_Burst_Mode DMA burst mode configuration
  * @{
  */
/**
  * @brief  Configures the timer DMA burst feature.
  * @note Macro IS_TIM_DMABURST_INSTANCE(TIMx) can be used to check whether or
  *       not a timer instance supports the DMA burst mode.
  * @rmtoll DCR          DBL           LL_TIM_ConfigDMABurst\n
  *         DCR          DBA           LL_TIM_ConfigDMABurst
  * @param  TIMx Timer instance
  * @param  DMABurstBaseAddress This parameter can be one of the following values:
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CR1
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CR2
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_SMCR
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_DIER
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_SR
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_EGR
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CCMR1
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CCMR2
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CCER
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CNT
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_PSC
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_ARR
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_RCR
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CCR1
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CCR2
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CCR3
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_CCR4
  *         @arg @ref LL_TIM_DMABURST_BASEADDR_BDTR
  * @param  DMABurstLength This parameter can be one of the following values:
  *         @arg @ref LL_TIM_DMABURST_LENGTH_1TRANSFER
  *         @arg @ref LL_TIM_DMABURST_LENGTH_2TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_3TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_4TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_5TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_6TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_7TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_8TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_9TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_10TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_11TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_12TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_13TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_14TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_15TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_16TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_17TRANSFERS
  *         @arg @ref LL_TIM_DMABURST_LENGTH_18TRANSFERS
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ConfigDMABurst(TIM_TypeDef *TIMx, uint32_t DMABurstBaseAddress, uint32_t DMABurstLength)
{
  MODIFY_REG(TIMx->DCR, (TIM_DCR_DBL | TIM_DCR_DBA), (DMABurstBaseAddress | DMABurstLength));
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_Timer_Inputs_Remapping Timer input remapping
  * @{
  */
/**
  * @brief  Remap TIM inputs (input channel, internal/external triggers).
  * @note Macro IS_TIM_REMAP_INSTANCE(TIMx) can be used to check whether or not
  *       a some timer inputs can be remapped.
  * @rmtoll TIM1_OR     ITR2_RMP          LL_TIM_SetRemap\n
  *         TIM2_OR     ITR1_RMP          LL_TIM_SetRemap\n
  *         TIM5_OR     ITR1_RMP          LL_TIM_SetRemap\n
  *         TIM5_OR     TI4_RMP           LL_TIM_SetRemap\n
  *         TIM9_OR     ITR1_RMP          LL_TIM_SetRemap\n
  *         TIM11_OR    TI1_RMP           LL_TIM_SetRemap\n
  *         LPTIM1_OR   OR                LL_TIM_SetRemap
  * @param  TIMx Timer instance
  * @param  Remap Remap param depends on the TIMx. Description available only
  *         in CHM version of the User Manual (not in .pdf).
  *         Otherwise see Reference Manual description of OR registers.
  *
  *         Below description summarizes "Timer Instance" and "Remap" param combinations:
  *
  *         TIM1: one of the following values
  *
  *            ITR2_RMP can be one of the following values
  *            @arg @ref LL_TIM_TIM1_ITR2_RMP_TIM3_TRGO (*)
  *            @arg @ref LL_TIM_TIM1_ITR2_RMP_LPTIM (*)
  *
  *         TIM2: one of the following values
  *
  *            ITR1_RMP can be one of the following values
  *            @arg @ref LL_TIM_TIM2_ITR1_RMP_TIM8_TRGO
  *            @arg @ref LL_TIM_TIM2_ITR1_RMP_OTG_FS_SOF
  *            @arg @ref LL_TIM_TIM2_ITR1_RMP_OTG_HS_SOF
  *
  *         TIM5: one of the following values
  *
  *            @arg @ref LL_TIM_TIM5_TI4_RMP_GPIO
  *            @arg @ref LL_TIM_TIM5_TI4_RMP_LSI
  *            @arg @ref LL_TIM_TIM5_TI4_RMP_LSE
  *            @arg @ref LL_TIM_TIM5_TI4_RMP_RTC
  *            @arg @ref LL_TIM_TIM5_ITR1_RMP_TIM3_TRGO (*)
  *            @arg @ref LL_TIM_TIM5_ITR1_RMP_LPTIM (*)
  *
  *         TIM9: one of the following values
  *
  *            ITR1_RMP can be one of the following values
  *            @arg @ref LL_TIM_TIM9_ITR1_RMP_TIM3_TRGO (*)
  *            @arg @ref LL_TIM_TIM9_ITR1_RMP_LPTIM (*)
  *
  *         TIM11: one of the following values
  *
  *            @arg @ref LL_TIM_TIM11_TI1_RMP_GPIO
  *            @arg @ref LL_TIM_TIM11_TI1_RMP_GPIO1 (*)
  *            @arg @ref LL_TIM_TIM11_TI1_RMP_HSE_RTC
  *            @arg @ref LL_TIM_TIM11_TI1_RMP_GPIO2
  *            @arg @ref LL_TIM_TIM11_TI1_RMP_SPDIFRX (*)
  *
  *         (*)  Value not defined in all devices. \n
  *
  * @retval None
  */
__STATIC_INLINE void LL_TIM_SetRemap(TIM_TypeDef *TIMx, uint32_t Remap)
{
#if defined(LPTIM_OR_TIM1_ITR2_RMP) && defined(LPTIM_OR_TIM5_ITR1_RMP) && defined(LPTIM_OR_TIM9_ITR1_RMP)
  if ((Remap & LL_TIM_LPTIM_REMAP_MASK) == LL_TIM_LPTIM_REMAP_MASK)
  {
    /* Connect TIMx internal trigger to LPTIM1 output */
    SET_BIT(RCC->APB1ENR, RCC_APB1ENR_LPTIM1EN);
    MODIFY_REG(LPTIM1->OR,
               (LPTIM_OR_TIM1_ITR2_RMP | LPTIM_OR_TIM5_ITR1_RMP | LPTIM_OR_TIM9_ITR1_RMP),
               Remap & ~(LL_TIM_LPTIM_REMAP_MASK));
  }
  else
  {
    MODIFY_REG(TIMx->OR, (Remap >> TIMx_OR_RMP_SHIFT), (Remap & TIMx_OR_RMP_MASK));
  }
#else
  MODIFY_REG(TIMx->OR, (Remap >> TIMx_OR_RMP_SHIFT), (Remap & TIMx_OR_RMP_MASK));
#endif /* LPTIM_OR_TIM1_ITR2_RMP &&  LPTIM_OR_TIM5_ITR1_RMP && LPTIM_OR_TIM9_ITR1_RMP */
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_FLAG_Management FLAG-Management
  * @{
  */
/**
  * @brief  Clear the update interrupt flag (UIF).
  * @rmtoll SR           UIF           LL_TIM_ClearFlag_UPDATE
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_UPDATE(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_UIF));
}

/**
  * @brief  Indicate whether update interrupt flag (UIF) is set (update interrupt is pending).
  * @rmtoll SR           UIF           LL_TIM_IsActiveFlag_UPDATE
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_UPDATE(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_UIF) == (TIM_SR_UIF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the Capture/Compare 1 interrupt flag (CC1F).
  * @rmtoll SR           CC1IF         LL_TIM_ClearFlag_CC1
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_CC1(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_CC1IF));
}

/**
  * @brief  Indicate whether Capture/Compare 1 interrupt flag (CC1F) is set (Capture/Compare 1 interrupt is pending).
  * @rmtoll SR           CC1IF         LL_TIM_IsActiveFlag_CC1
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_CC1(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_CC1IF) == (TIM_SR_CC1IF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the Capture/Compare 2 interrupt flag (CC2F).
  * @rmtoll SR           CC2IF         LL_TIM_ClearFlag_CC2
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_CC2(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_CC2IF));
}

/**
  * @brief  Indicate whether Capture/Compare 2 interrupt flag (CC2F) is set (Capture/Compare 2 interrupt is pending).
  * @rmtoll SR           CC2IF         LL_TIM_IsActiveFlag_CC2
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_CC2(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_CC2IF) == (TIM_SR_CC2IF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the Capture/Compare 3 interrupt flag (CC3F).
  * @rmtoll SR           CC3IF         LL_TIM_ClearFlag_CC3
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_CC3(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_CC3IF));
}

/**
  * @brief  Indicate whether Capture/Compare 3 interrupt flag (CC3F) is set (Capture/Compare 3 interrupt is pending).
  * @rmtoll SR           CC3IF         LL_TIM_IsActiveFlag_CC3
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_CC3(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_CC3IF) == (TIM_SR_CC3IF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the Capture/Compare 4 interrupt flag (CC4F).
  * @rmtoll SR           CC4IF         LL_TIM_ClearFlag_CC4
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_CC4(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_CC4IF));
}

/**
  * @brief  Indicate whether Capture/Compare 4 interrupt flag (CC4F) is set (Capture/Compare 4 interrupt is pending).
  * @rmtoll SR           CC4IF         LL_TIM_IsActiveFlag_CC4
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_CC4(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_CC4IF) == (TIM_SR_CC4IF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the commutation interrupt flag (COMIF).
  * @rmtoll SR           COMIF         LL_TIM_ClearFlag_COM
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_COM(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_COMIF));
}

/**
  * @brief  Indicate whether commutation interrupt flag (COMIF) is set (commutation interrupt is pending).
  * @rmtoll SR           COMIF         LL_TIM_IsActiveFlag_COM
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_COM(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_COMIF) == (TIM_SR_COMIF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the trigger interrupt flag (TIF).
  * @rmtoll SR           TIF           LL_TIM_ClearFlag_TRIG
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_TRIG(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_TIF));
}

/**
  * @brief  Indicate whether trigger interrupt flag (TIF) is set (trigger interrupt is pending).
  * @rmtoll SR           TIF           LL_TIM_IsActiveFlag_TRIG
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_TRIG(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_TIF) == (TIM_SR_TIF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the break interrupt flag (BIF).
  * @rmtoll SR           BIF           LL_TIM_ClearFlag_BRK
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_BRK(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_BIF));
}

/**
  * @brief  Indicate whether break interrupt flag (BIF) is set (break interrupt is pending).
  * @rmtoll SR           BIF           LL_TIM_IsActiveFlag_BRK
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_BRK(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_BIF) == (TIM_SR_BIF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the Capture/Compare 1 over-capture interrupt flag (CC1OF).
  * @rmtoll SR           CC1OF         LL_TIM_ClearFlag_CC1OVR
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_CC1OVR(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_CC1OF));
}

/**
  * @brief  Indicate whether Capture/Compare 1 over-capture interrupt flag (CC1OF) is set
  *         (Capture/Compare 1 interrupt is pending).
  * @rmtoll SR           CC1OF         LL_TIM_IsActiveFlag_CC1OVR
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_CC1OVR(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_CC1OF) == (TIM_SR_CC1OF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the Capture/Compare 2 over-capture interrupt flag (CC2OF).
  * @rmtoll SR           CC2OF         LL_TIM_ClearFlag_CC2OVR
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_CC2OVR(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_CC2OF));
}

/**
  * @brief  Indicate whether Capture/Compare 2 over-capture interrupt flag (CC2OF) is set
  *         (Capture/Compare 2 over-capture interrupt is pending).
  * @rmtoll SR           CC2OF         LL_TIM_IsActiveFlag_CC2OVR
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_CC2OVR(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_CC2OF) == (TIM_SR_CC2OF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the Capture/Compare 3 over-capture interrupt flag (CC3OF).
  * @rmtoll SR           CC3OF         LL_TIM_ClearFlag_CC3OVR
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_CC3OVR(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_CC3OF));
}

/**
  * @brief  Indicate whether Capture/Compare 3 over-capture interrupt flag (CC3OF) is set
  *         (Capture/Compare 3 over-capture interrupt is pending).
  * @rmtoll SR           CC3OF         LL_TIM_IsActiveFlag_CC3OVR
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_CC3OVR(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_CC3OF) == (TIM_SR_CC3OF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear the Capture/Compare 4 over-capture interrupt flag (CC4OF).
  * @rmtoll SR           CC4OF         LL_TIM_ClearFlag_CC4OVR
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_ClearFlag_CC4OVR(TIM_TypeDef *TIMx)
{
  WRITE_REG(TIMx->SR, ~(TIM_SR_CC4OF));
}

/**
  * @brief  Indicate whether Capture/Compare 4 over-capture interrupt flag (CC4OF) is set
  *         (Capture/Compare 4 over-capture interrupt is pending).
  * @rmtoll SR           CC4OF         LL_TIM_IsActiveFlag_CC4OVR
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsActiveFlag_CC4OVR(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->SR, TIM_SR_CC4OF) == (TIM_SR_CC4OF)) ? 1UL : 0UL);
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_IT_Management IT-Management
  * @{
  */
/**
  * @brief  Enable update interrupt (UIE).
  * @rmtoll DIER         UIE           LL_TIM_EnableIT_UPDATE
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableIT_UPDATE(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_UIE);
}

/**
  * @brief  Disable update interrupt (UIE).
  * @rmtoll DIER         UIE           LL_TIM_DisableIT_UPDATE
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableIT_UPDATE(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_UIE);
}

/**
  * @brief  Indicates whether the update interrupt (UIE) is enabled.
  * @rmtoll DIER         UIE           LL_TIM_IsEnabledIT_UPDATE
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledIT_UPDATE(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_UIE) == (TIM_DIER_UIE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable capture/compare 1 interrupt (CC1IE).
  * @rmtoll DIER         CC1IE         LL_TIM_EnableIT_CC1
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableIT_CC1(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_CC1IE);
}

/**
  * @brief  Disable capture/compare 1  interrupt (CC1IE).
  * @rmtoll DIER         CC1IE         LL_TIM_DisableIT_CC1
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableIT_CC1(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_CC1IE);
}

/**
  * @brief  Indicates whether the capture/compare 1 interrupt (CC1IE) is enabled.
  * @rmtoll DIER         CC1IE         LL_TIM_IsEnabledIT_CC1
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledIT_CC1(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_CC1IE) == (TIM_DIER_CC1IE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable capture/compare 2 interrupt (CC2IE).
  * @rmtoll DIER         CC2IE         LL_TIM_EnableIT_CC2
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableIT_CC2(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_CC2IE);
}

/**
  * @brief  Disable capture/compare 2  interrupt (CC2IE).
  * @rmtoll DIER         CC2IE         LL_TIM_DisableIT_CC2
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableIT_CC2(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_CC2IE);
}

/**
  * @brief  Indicates whether the capture/compare 2 interrupt (CC2IE) is enabled.
  * @rmtoll DIER         CC2IE         LL_TIM_IsEnabledIT_CC2
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledIT_CC2(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_CC2IE) == (TIM_DIER_CC2IE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable capture/compare 3 interrupt (CC3IE).
  * @rmtoll DIER         CC3IE         LL_TIM_EnableIT_CC3
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableIT_CC3(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_CC3IE);
}

/**
  * @brief  Disable capture/compare 3  interrupt (CC3IE).
  * @rmtoll DIER         CC3IE         LL_TIM_DisableIT_CC3
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableIT_CC3(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_CC3IE);
}

/**
  * @brief  Indicates whether the capture/compare 3 interrupt (CC3IE) is enabled.
  * @rmtoll DIER         CC3IE         LL_TIM_IsEnabledIT_CC3
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledIT_CC3(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_CC3IE) == (TIM_DIER_CC3IE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable capture/compare 4 interrupt (CC4IE).
  * @rmtoll DIER         CC4IE         LL_TIM_EnableIT_CC4
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableIT_CC4(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_CC4IE);
}

/**
  * @brief  Disable capture/compare 4  interrupt (CC4IE).
  * @rmtoll DIER         CC4IE         LL_TIM_DisableIT_CC4
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableIT_CC4(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_CC4IE);
}

/**
  * @brief  Indicates whether the capture/compare 4 interrupt (CC4IE) is enabled.
  * @rmtoll DIER         CC4IE         LL_TIM_IsEnabledIT_CC4
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledIT_CC4(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_CC4IE) == (TIM_DIER_CC4IE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable commutation interrupt (COMIE).
  * @rmtoll DIER         COMIE         LL_TIM_EnableIT_COM
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableIT_COM(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_COMIE);
}

/**
  * @brief  Disable commutation interrupt (COMIE).
  * @rmtoll DIER         COMIE         LL_TIM_DisableIT_COM
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableIT_COM(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_COMIE);
}

/**
  * @brief  Indicates whether the commutation interrupt (COMIE) is enabled.
  * @rmtoll DIER         COMIE         LL_TIM_IsEnabledIT_COM
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledIT_COM(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_COMIE) == (TIM_DIER_COMIE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable trigger interrupt (TIE).
  * @rmtoll DIER         TIE           LL_TIM_EnableIT_TRIG
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableIT_TRIG(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_TIE);
}

/**
  * @brief  Disable trigger interrupt (TIE).
  * @rmtoll DIER         TIE           LL_TIM_DisableIT_TRIG
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableIT_TRIG(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_TIE);
}

/**
  * @brief  Indicates whether the trigger interrupt (TIE) is enabled.
  * @rmtoll DIER         TIE           LL_TIM_IsEnabledIT_TRIG
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledIT_TRIG(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_TIE) == (TIM_DIER_TIE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable break interrupt (BIE).
  * @rmtoll DIER         BIE           LL_TIM_EnableIT_BRK
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableIT_BRK(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_BIE);
}

/**
  * @brief  Disable break interrupt (BIE).
  * @rmtoll DIER         BIE           LL_TIM_DisableIT_BRK
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableIT_BRK(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_BIE);
}

/**
  * @brief  Indicates whether the break interrupt (BIE) is enabled.
  * @rmtoll DIER         BIE           LL_TIM_IsEnabledIT_BRK
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledIT_BRK(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_BIE) == (TIM_DIER_BIE)) ? 1UL : 0UL);
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_DMA_Management DMA Management
  * @{
  */
/**
  * @brief  Enable update DMA request (UDE).
  * @rmtoll DIER         UDE           LL_TIM_EnableDMAReq_UPDATE
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableDMAReq_UPDATE(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_UDE);
}

/**
  * @brief  Disable update DMA request (UDE).
  * @rmtoll DIER         UDE           LL_TIM_DisableDMAReq_UPDATE
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableDMAReq_UPDATE(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_UDE);
}

/**
  * @brief  Indicates whether the update DMA request  (UDE) is enabled.
  * @rmtoll DIER         UDE           LL_TIM_IsEnabledDMAReq_UPDATE
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledDMAReq_UPDATE(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_UDE) == (TIM_DIER_UDE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable capture/compare 1 DMA request (CC1DE).
  * @rmtoll DIER         CC1DE         LL_TIM_EnableDMAReq_CC1
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableDMAReq_CC1(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_CC1DE);
}

/**
  * @brief  Disable capture/compare 1  DMA request (CC1DE).
  * @rmtoll DIER         CC1DE         LL_TIM_DisableDMAReq_CC1
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableDMAReq_CC1(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_CC1DE);
}

/**
  * @brief  Indicates whether the capture/compare 1 DMA request (CC1DE) is enabled.
  * @rmtoll DIER         CC1DE         LL_TIM_IsEnabledDMAReq_CC1
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledDMAReq_CC1(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_CC1DE) == (TIM_DIER_CC1DE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable capture/compare 2 DMA request (CC2DE).
  * @rmtoll DIER         CC2DE         LL_TIM_EnableDMAReq_CC2
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableDMAReq_CC2(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_CC2DE);
}

/**
  * @brief  Disable capture/compare 2  DMA request (CC2DE).
  * @rmtoll DIER         CC2DE         LL_TIM_DisableDMAReq_CC2
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableDMAReq_CC2(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_CC2DE);
}

/**
  * @brief  Indicates whether the capture/compare 2 DMA request (CC2DE) is enabled.
  * @rmtoll DIER         CC2DE         LL_TIM_IsEnabledDMAReq_CC2
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledDMAReq_CC2(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_CC2DE) == (TIM_DIER_CC2DE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable capture/compare 3 DMA request (CC3DE).
  * @rmtoll DIER         CC3DE         LL_TIM_EnableDMAReq_CC3
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableDMAReq_CC3(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_CC3DE);
}

/**
  * @brief  Disable capture/compare 3  DMA request (CC3DE).
  * @rmtoll DIER         CC3DE         LL_TIM_DisableDMAReq_CC3
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableDMAReq_CC3(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_CC3DE);
}

/**
  * @brief  Indicates whether the capture/compare 3 DMA request (CC3DE) is enabled.
  * @rmtoll DIER         CC3DE         LL_TIM_IsEnabledDMAReq_CC3
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledDMAReq_CC3(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_CC3DE) == (TIM_DIER_CC3DE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable capture/compare 4 DMA request (CC4DE).
  * @rmtoll DIER         CC4DE         LL_TIM_EnableDMAReq_CC4
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableDMAReq_CC4(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_CC4DE);
}

/**
  * @brief  Disable capture/compare 4  DMA request (CC4DE).
  * @rmtoll DIER         CC4DE         LL_TIM_DisableDMAReq_CC4
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableDMAReq_CC4(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_CC4DE);
}

/**
  * @brief  Indicates whether the capture/compare 4 DMA request (CC4DE) is enabled.
  * @rmtoll DIER         CC4DE         LL_TIM_IsEnabledDMAReq_CC4
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledDMAReq_CC4(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_CC4DE) == (TIM_DIER_CC4DE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable commutation DMA request (COMDE).
  * @rmtoll DIER         COMDE         LL_TIM_EnableDMAReq_COM
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableDMAReq_COM(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_COMDE);
}

/**
  * @brief  Disable commutation DMA request (COMDE).
  * @rmtoll DIER         COMDE         LL_TIM_DisableDMAReq_COM
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableDMAReq_COM(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_COMDE);
}

/**
  * @brief  Indicates whether the commutation DMA request (COMDE) is enabled.
  * @rmtoll DIER         COMDE         LL_TIM_IsEnabledDMAReq_COM
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledDMAReq_COM(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_COMDE) == (TIM_DIER_COMDE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable trigger interrupt (TDE).
  * @rmtoll DIER         TDE           LL_TIM_EnableDMAReq_TRIG
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_EnableDMAReq_TRIG(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->DIER, TIM_DIER_TDE);
}

/**
  * @brief  Disable trigger interrupt (TDE).
  * @rmtoll DIER         TDE           LL_TIM_DisableDMAReq_TRIG
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_DisableDMAReq_TRIG(TIM_TypeDef *TIMx)
{
  CLEAR_BIT(TIMx->DIER, TIM_DIER_TDE);
}

/**
  * @brief  Indicates whether the trigger interrupt (TDE) is enabled.
  * @rmtoll DIER         TDE           LL_TIM_IsEnabledDMAReq_TRIG
  * @param  TIMx Timer instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_TIM_IsEnabledDMAReq_TRIG(TIM_TypeDef *TIMx)
{
  return ((READ_BIT(TIMx->DIER, TIM_DIER_TDE) == (TIM_DIER_TDE)) ? 1UL : 0UL);
}

/**
  * @}
  */

/** @defgroup TIM_LL_EF_EVENT_Management EVENT-Management
  * @{
  */
/**
  * @brief  Generate an update event.
  * @rmtoll EGR          UG            LL_TIM_GenerateEvent_UPDATE
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_GenerateEvent_UPDATE(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->EGR, TIM_EGR_UG);
}

/**
  * @brief  Generate Capture/Compare 1 event.
  * @rmtoll EGR          CC1G          LL_TIM_GenerateEvent_CC1
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_GenerateEvent_CC1(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->EGR, TIM_EGR_CC1G);
}

/**
  * @brief  Generate Capture/Compare 2 event.
  * @rmtoll EGR          CC2G          LL_TIM_GenerateEvent_CC2
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_GenerateEvent_CC2(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->EGR, TIM_EGR_CC2G);
}

/**
  * @brief  Generate Capture/Compare 3 event.
  * @rmtoll EGR          CC3G          LL_TIM_GenerateEvent_CC3
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_GenerateEvent_CC3(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->EGR, TIM_EGR_CC3G);
}

/**
  * @brief  Generate Capture/Compare 4 event.
  * @rmtoll EGR          CC4G          LL_TIM_GenerateEvent_CC4
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_GenerateEvent_CC4(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->EGR, TIM_EGR_CC4G);
}

/**
  * @brief  Generate commutation event.
  * @rmtoll EGR          COMG          LL_TIM_GenerateEvent_COM
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_GenerateEvent_COM(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->EGR, TIM_EGR_COMG);
}

/**
  * @brief  Generate trigger event.
  * @rmtoll EGR          TG            LL_TIM_GenerateEvent_TRIG
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_GenerateEvent_TRIG(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->EGR, TIM_EGR_TG);
}

/**
  * @brief  Generate break event.
  * @rmtoll EGR          BG            LL_TIM_GenerateEvent_BRK
  * @param  TIMx Timer instance
  * @retval None
  */
__STATIC_INLINE void LL_TIM_GenerateEvent_BRK(TIM_TypeDef *TIMx)
{
  SET_BIT(TIMx->EGR, TIM_EGR_BG);
}

/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup TIM_LL_EF_Init Initialisation and deinitialisation functions
  * @{
  */

ErrorStatus LL_TIM_DeInit(TIM_TypeDef *TIMx);
void LL_TIM_StructInit(LL_TIM_InitTypeDef *TIM_InitStruct);
ErrorStatus LL_TIM_Init(TIM_TypeDef *TIMx, LL_TIM_InitTypeDef *TIM_InitStruct);
void LL_TIM_OC_StructInit(LL_TIM_OC_InitTypeDef *TIM_OC_InitStruct);
ErrorStatus LL_TIM_OC_Init(TIM_TypeDef *TIMx, uint32_t Channel, LL_TIM_OC_InitTypeDef *TIM_OC_InitStruct);
void LL_TIM_IC_StructInit(LL_TIM_IC_InitTypeDef *TIM_ICInitStruct);
ErrorStatus LL_TIM_IC_Init(TIM_TypeDef *TIMx, uint32_t Channel, LL_TIM_IC_InitTypeDef *TIM_IC_InitStruct);
void LL_TIM_ENCODER_StructInit(LL_TIM_ENCODER_InitTypeDef *TIM_EncoderInitStruct);
ErrorStatus LL_TIM_ENCODER_Init(TIM_TypeDef *TIMx, LL_TIM_ENCODER_InitTypeDef *TIM_EncoderInitStruct);
void LL_TIM_HALLSENSOR_StructInit(LL_TIM_HALLSENSOR_InitTypeDef *TIM_HallSensorInitStruct);
ErrorStatus LL_TIM_HALLSENSOR_Init(TIM_TypeDef *TIMx, LL_TIM_HALLSENSOR_InitTypeDef *TIM_HallSensorInitStruct);
void LL_TIM_BDTR_StructInit(LL_TIM_BDTR_InitTypeDef *TIM_BDTRInitStruct);
ErrorStatus LL_TIM_BDTR_Init(TIM_TypeDef *TIMx, LL_TIM_BDTR_InitTypeDef *TIM_BDTRInitStruct);
/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/**
  * @}
  */

/**
  * @}
  */

#endif /* TIM1 || TIM2 || TIM3 || TIM4 || TIM5 || TIM6 || TIM7 || TIM8 || TIM9 || TIM10 || TIM11 || TIM12 || TIM13 || TIM14 */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_LL_TIM_H */
/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
