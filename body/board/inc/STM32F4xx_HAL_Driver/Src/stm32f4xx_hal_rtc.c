/**
  ******************************************************************************
  * @file    stm32f4xx_hal_rtc.c
  * @author  MCD Application Team
  * @brief   RTC HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the Real Time Clock (RTC) peripheral:
  *           + Initialization and de-initialization functions
  *           + RTC Time and Date functions
  *           + RTC Alarm functions
  *           + Peripheral Control functions
  *           + Peripheral State functions
  *
  @verbatim
  ==============================================================================
              ##### Backup Domain Operating Condition #####
  ==============================================================================
  [..] The real-time clock (RTC), the RTC backup registers, and the backup
       SRAM (BKP SRAM) can be powered from the VBAT voltage when the main
       VDD supply is powered off.
       To retain the content of the RTC backup registers, backup SRAM, and supply
       the RTC when VDD is turned off, VBAT pin can be connected to an optional
       standby voltage supplied by a battery or by another source.

  [..] To allow the RTC operating even when the main digital supply (VDD) is turned
       off, the VBAT pin powers the following blocks:
    (#) The RTC
    (#) The LSE oscillator
    (#) The backup SRAM when the low power backup regulator is enabled
    (#) PC13 to PC15 I/Os, plus PI8 I/O (when available)

  [..] When the backup domain is supplied by VDD (analog switch connected to VDD),
       the following pins are available:
    (#) PC14 and PC15 can be used as either GPIO or LSE pins
    (#) PC13 can be used as a GPIO or as the RTC_AF1 pin
    (#) PI8 can be used as a GPIO or as the RTC_AF2 pin

  [..] When the backup domain is supplied by VBAT (analog switch connected to VBAT
       because VDD is not present), the following pins are available:
    (#) PC14 and PC15 can be used as LSE pins only
    (#) PC13 can be used as the RTC_AF1 pin
    (#) PI8 can be used as the RTC_AF2 pin

                   ##### Backup Domain Reset #####
  ==================================================================
  [..] The backup domain reset sets all RTC registers and the RCC_BDCR register
       to their reset values. The BKPSRAM is not affected by this reset. The only
       way to reset the BKPSRAM is through the Flash interface by requesting
       a protection level change from 1 to 0.
  [..] A backup domain reset is generated when one of the following events occurs:
    (#) Software reset, triggered by setting the BDRST bit in the
        RCC Backup domain control register (RCC_BDCR).
    (#) VDD or VBAT power on, if both supplies have previously been powered off.

                   ##### Backup Domain Access #####
  ==================================================================
  [..] After reset, the backup domain (RTC registers, RTC backup data
       registers and backup SRAM) is protected against possible unwanted write
       accesses.
  [..] To enable access to the RTC Domain and RTC registers, proceed as follows:
    (+) Enable the Power Controller (PWR) APB1 interface clock using the
        __HAL_RCC_PWR_CLK_ENABLE() function.
    (+) Enable access to RTC domain using the HAL_PWR_EnableBkUpAccess() function.
    (+) Select the RTC clock source using the __HAL_RCC_RTC_CONFIG() function.
    (+) Enable RTC Clock using the __HAL_RCC_RTC_ENABLE() function.


                  ##### How to use this driver #####
  ==================================================================
  [..]
    (+) Enable the RTC domain access (see description in the section above).
    (+) Configure the RTC Prescaler (Asynchronous and Synchronous) and RTC hour
        format using the HAL_RTC_Init() function.

  *** Time and Date configuration ***
  ===================================
  [..]
    (+) To configure the RTC Calendar (Time and Date) use the HAL_RTC_SetTime()
        and HAL_RTC_SetDate() functions.
    (+) To read the RTC Calendar, use the HAL_RTC_GetTime() and HAL_RTC_GetDate() functions.

  *** Alarm configuration ***
  ===========================
  [..]
    (+) To configure the RTC Alarm use the HAL_RTC_SetAlarm() function.
        You can also configure the RTC Alarm with interrupt mode using the HAL_RTC_SetAlarm_IT() function.
    (+) To read the RTC Alarm, use the HAL_RTC_GetAlarm() function.

                  ##### RTC and low power modes #####
  ==================================================================
  [..] The MCU can be woken up from a low power mode by an RTC alternate
       function.
  [..] The RTC alternate functions are the RTC alarms (Alarm A and Alarm B),
       RTC wake-up, RTC tamper event detection and RTC time stamp event detection.
       These RTC alternate functions can wake up the system from the Stop and
       Standby low power modes.
  [..] The system can also wake up from low power modes without depending
       on an external interrupt (Auto-wake-up mode), by using the RTC alarm
       or the RTC wake-up events.
  [..] The RTC provides a programmable time base for waking up from the
       Stop or Standby mode at regular intervals.
       Wake-up from STOP and STANDBY modes is possible only when the RTC clock source
       is LSE or LSI.

  *** Callback registration ***
  =============================================

  [..]
  The compilation define  USE_HAL_RTC_REGISTER_CALLBACKS when set to 1
  allows the user to configure dynamically the driver callbacks.
  Use Function HAL_RTC_RegisterCallback() to register an interrupt callback.

  [..]
  Function HAL_RTC_RegisterCallback() allows to register following callbacks:
    (+) AlarmAEventCallback          : RTC Alarm A Event callback.
    (+) AlarmBEventCallback          : RTC Alarm B Event callback.
    (+) TimeStampEventCallback       : RTC TimeStamp Event callback.
    (+) WakeUpTimerEventCallback     : RTC WakeUpTimer Event callback.
    (+) Tamper1EventCallback         : RTC Tamper 1 Event callback.
    (+) Tamper2EventCallback         : RTC Tamper 2 Event callback.
    (+) MspInitCallback              : RTC MspInit callback.
    (+) MspDeInitCallback            : RTC MspDeInit callback.
  [..]
  This function takes as parameters the HAL peripheral handle, the Callback ID
  and a pointer to the user callback function.

  [..]
  Use function HAL_RTC_UnRegisterCallback() to reset a callback to the default
  weak function.
  HAL_RTC_UnRegisterCallback() takes as parameters the HAL peripheral handle,
  and the Callback ID.
  This function allows to reset following callbacks:
    (+) AlarmAEventCallback          : RTC Alarm A Event callback.
    (+) AlarmBEventCallback          : RTC Alarm B Event callback.
    (+) TimeStampEventCallback       : RTC TimeStamp Event callback.
    (+) WakeUpTimerEventCallback     : RTC WakeUpTimer Event callback.
    (+) Tamper1EventCallback         : RTC Tamper 1 Event callback.
    (+) Tamper2EventCallback         : RTC Tamper 2 Event callback.
    (+) MspInitCallback              : RTC MspInit callback.
    (+) MspDeInitCallback            : RTC MspDeInit callback.

  [..]
  By default, after the HAL_RTC_Init() and when the state is HAL_RTC_STATE_RESET,
  all callbacks are set to the corresponding weak functions :
  examples AlarmAEventCallback(), WakeUpTimerEventCallback().
  Exception done for MspInit and MspDeInit callbacks that are reset to the legacy weak function
  in the HAL_RTC_Init()/HAL_RTC_DeInit() only when these callbacks are null
  (not registered beforehand).
  If not, MspInit or MspDeInit are not null, HAL_RTC_Init()/HAL_RTC_DeInit()
  keep and use the user MspInit/MspDeInit callbacks (registered beforehand)

  [..]
  Callbacks can be registered/unregistered in HAL_RTC_STATE_READY state only.
  Exception done MspInit/MspDeInit that can be registered/unregistered
  in HAL_RTC_STATE_READY or HAL_RTC_STATE_RESET state,
  thus registered (user) MspInit/DeInit callbacks can be used during the Init/DeInit.
  In that case first register the MspInit/MspDeInit user callbacks
  using HAL_RTC_RegisterCallback() before calling HAL_RTC_DeInit()
  or HAL_RTC_Init() function.

  [..]
  When The compilation define USE_HAL_RTC_REGISTER_CALLBACKS is set to 0 or
  not defined, the callback registration feature is not available and all callbacks
  are set to the corresponding weak functions.
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

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup RTC RTC
  * @brief RTC HAL module driver
  * @{
  */

#ifdef HAL_RTC_MODULE_ENABLED

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Exported functions --------------------------------------------------------*/

/** @defgroup RTC_Exported_Functions RTC Exported Functions
  * @{
  */

/** @defgroup RTC_Exported_Functions_Group1 Initialization and de-initialization functions
 *  @brief    Initialization and Configuration functions
 *
@verbatim
 ===============================================================================
              ##### Initialization and de-initialization functions #####
 ===============================================================================
   [..] This section provides functions allowing to initialize and configure the
         RTC Prescaler (Synchronous and Asynchronous), RTC Hour format, disable
         RTC registers Write protection, enter and exit the RTC initialization mode,
         RTC registers synchronization check and reference clock detection enable.
         (#) The RTC Prescaler is programmed to generate the RTC 1Hz time base.
             It is split into 2 programmable prescalers to minimize power consumption.
             (++) A 7-bit asynchronous prescaler and a 13-bit synchronous prescaler.
             (++) When both prescalers are used, it is recommended to configure the
                 asynchronous prescaler to a high value to minimize power consumption.
         (#) All RTC registers are Write protected. Writing to the RTC registers
             is enabled by writing a key into the Write Protection register, RTC_WPR.
         (#) To configure the RTC Calendar, user application should enter
             initialization mode. In this mode, the calendar counter is stopped
             and its value can be updated. When the initialization sequence is
             complete, the calendar restarts counting after 4 RTCCLK cycles.
         (#) To read the calendar through the shadow registers after Calendar
             initialization, calendar update or after wake-up from low power modes
             the software must first clear the RSF flag. The software must then
             wait until it is set again before reading the calendar, which means
             that the calendar registers have been correctly copied into the
             RTC_TR and RTC_DR shadow registers.The HAL_RTC_WaitForSynchro() function
             implements the above software sequence (RSF clear and RSF check).

@endverbatim
  * @{
  */

/**
  * @brief  Initializes the RTC peripheral
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_Init(RTC_HandleTypeDef *hrtc)
{
  /* Check the RTC peripheral state */
  if(hrtc == NULL)
  {
     return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_RTC_HOUR_FORMAT(hrtc->Init.HourFormat));
  assert_param(IS_RTC_ASYNCH_PREDIV(hrtc->Init.AsynchPrediv));
  assert_param(IS_RTC_SYNCH_PREDIV(hrtc->Init.SynchPrediv));
  assert_param (IS_RTC_OUTPUT(hrtc->Init.OutPut));
  assert_param (IS_RTC_OUTPUT_POL(hrtc->Init.OutPutPolarity));
  assert_param(IS_RTC_OUTPUT_TYPE(hrtc->Init.OutPutType));

#if (USE_HAL_RTC_REGISTER_CALLBACKS == 1)
  if(hrtc->State == HAL_RTC_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hrtc->Lock = HAL_UNLOCKED;

    hrtc->AlarmAEventCallback          =  HAL_RTC_AlarmAEventCallback;        /* Legacy weak AlarmAEventCallback      */
    hrtc->AlarmBEventCallback          =  HAL_RTCEx_AlarmBEventCallback;      /* Legacy weak AlarmBEventCallback      */
    hrtc->TimeStampEventCallback       =  HAL_RTCEx_TimeStampEventCallback;   /* Legacy weak TimeStampEventCallback   */
    hrtc->WakeUpTimerEventCallback     =  HAL_RTCEx_WakeUpTimerEventCallback; /* Legacy weak WakeUpTimerEventCallback */
    hrtc->Tamper1EventCallback         =  HAL_RTCEx_Tamper1EventCallback;     /* Legacy weak Tamper1EventCallback     */
    hrtc->Tamper2EventCallback         =  HAL_RTCEx_Tamper2EventCallback;     /* Legacy weak Tamper2EventCallback     */

    if(hrtc->MspInitCallback == NULL)
    {
      hrtc->MspInitCallback = HAL_RTC_MspInit;
    }
    /* Init the low level hardware */
    hrtc->MspInitCallback(hrtc);

    if(hrtc->MspDeInitCallback == NULL)
    {
      hrtc->MspDeInitCallback = HAL_RTC_MspDeInit;
    }
  }
#else
  if(hrtc->State == HAL_RTC_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hrtc->Lock = HAL_UNLOCKED;

    /* Initialize RTC MSP */
    HAL_RTC_MspInit(hrtc);
  }
#endif /* (USE_HAL_RTC_REGISTER_CALLBACKS) */

  /* Set RTC state */
  hrtc->State = HAL_RTC_STATE_BUSY;

  /* Disable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);

  /* Set Initialization mode */
  if(RTC_EnterInitMode(hrtc) != HAL_OK)
  {
    /* Enable the write protection for RTC registers */
    __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

    /* Set RTC state */
    hrtc->State = HAL_RTC_STATE_ERROR;

    return HAL_ERROR;
  }
  else
  {
    /* Clear RTC_CR FMT, OSEL and POL Bits */
    hrtc->Instance->CR &= ((uint32_t)~(RTC_CR_FMT | RTC_CR_OSEL | RTC_CR_POL));
    /* Set RTC_CR register */
    hrtc->Instance->CR |= (uint32_t)(hrtc->Init.HourFormat | hrtc->Init.OutPut | hrtc->Init.OutPutPolarity);

    /* Configure the RTC PRER */
    hrtc->Instance->PRER = (uint32_t)(hrtc->Init.SynchPrediv);
    hrtc->Instance->PRER |= (uint32_t)(hrtc->Init.AsynchPrediv << 16U);

    /* Exit Initialization mode */
    hrtc->Instance->ISR &= (uint32_t)~RTC_ISR_INIT;

    /* If  CR_BYPSHAD bit = 0, wait for synchro else this check is not needed */
    if((hrtc->Instance->CR & RTC_CR_BYPSHAD) == RESET)
    {
      if(HAL_RTC_WaitForSynchro(hrtc) != HAL_OK)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_ERROR;

        return HAL_ERROR;
      }
    }

    hrtc->Instance->TAFCR &= (uint32_t)~RTC_TAFCR_ALARMOUTTYPE;
    hrtc->Instance->TAFCR |= (uint32_t)(hrtc->Init.OutPutType);

    /* Enable the write protection for RTC registers */
    __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

    /* Set RTC state */
    hrtc->State = HAL_RTC_STATE_READY;

    return HAL_OK;
  }
}

/**
  * @brief  DeInitializes the RTC peripheral
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @note   This function doesn't reset the RTC Backup Data registers.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_DeInit(RTC_HandleTypeDef *hrtc)
{
  uint32_t tickstart = 0U;

  /* Set RTC state */
  hrtc->State = HAL_RTC_STATE_BUSY;

  /* Disable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);

  /* Set Initialization mode */
  if(RTC_EnterInitMode(hrtc) != HAL_OK)
  {
    /* Enable the write protection for RTC registers */
    __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

    /* Set RTC state */
    hrtc->State = HAL_RTC_STATE_ERROR;

    return HAL_ERROR;
  }
  else
  {
    /* Reset TR, DR and CR registers */
    hrtc->Instance->TR = 0x00000000U;
    hrtc->Instance->DR = 0x00002101U;
    /* Reset All CR bits except CR[2:0] */
    hrtc->Instance->CR &= 0x00000007U;

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait till WUTWF flag is set and if Time out is reached exit */
    while(((hrtc->Instance->ISR) & RTC_ISR_WUTWF) == (uint32_t)RESET)
    {
      if((HAL_GetTick() - tickstart ) > RTC_TIMEOUT_VALUE)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        /* Set RTC state */
        hrtc->State = HAL_RTC_STATE_TIMEOUT;

        return HAL_TIMEOUT;
      }
    }

    /* Reset all RTC CR register bits */
    hrtc->Instance->CR &= 0x00000000U;
    hrtc->Instance->WUTR = 0x0000FFFFU;
    hrtc->Instance->PRER = 0x007F00FFU;
    hrtc->Instance->CALIBR = 0x00000000U;
    hrtc->Instance->ALRMAR = 0x00000000U;
    hrtc->Instance->ALRMBR = 0x00000000U;
    hrtc->Instance->SHIFTR = 0x00000000U;
    hrtc->Instance->CALR = 0x00000000U;
    hrtc->Instance->ALRMASSR = 0x00000000U;
    hrtc->Instance->ALRMBSSR = 0x00000000U;

    /* Reset ISR register and exit initialization mode */
    hrtc->Instance->ISR = 0x00000000U;

    /* Reset Tamper and alternate functions configuration register */
    hrtc->Instance->TAFCR = 0x00000000U;

    /* If  RTC_CR_BYPSHAD bit = 0, wait for synchro else this check is not needed */
    if((hrtc->Instance->CR & RTC_CR_BYPSHAD) == RESET)
    {
      if(HAL_RTC_WaitForSynchro(hrtc) != HAL_OK)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_ERROR;

        return HAL_ERROR;
      }
    }
  }

  /* Enable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

#if (USE_HAL_RTC_REGISTER_CALLBACKS == 1)
  if(hrtc->MspDeInitCallback == NULL)
  {
    hrtc->MspDeInitCallback = HAL_RTC_MspDeInit;
  }

  /* DeInit the low level hardware: CLOCK, NVIC.*/
  hrtc->MspDeInitCallback(hrtc);

#else
  /* De-Initialize RTC MSP */
  HAL_RTC_MspDeInit(hrtc);
#endif /* (USE_HAL_RTC_REGISTER_CALLBACKS) */

  hrtc->State = HAL_RTC_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(hrtc);

  return HAL_OK;
}

#if (USE_HAL_RTC_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User RTC Callback
  *         To be used instead of the weak predefined callback
  * @param  hrtc RTC handle
  * @param  CallbackID ID of the callback to be registered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_RTC_ALARM_A_EVENT_CB_ID          Alarm A Event Callback ID
  *          @arg @ref HAL_RTC_ALARM_B_EVENT_CB_ID          Alarm B Event Callback ID
  *          @arg @ref HAL_RTC_TIMESTAMP_EVENT_CB_ID        TimeStamp Event Callback ID
  *          @arg @ref HAL_RTC_WAKEUPTIMER_EVENT_CB_ID      Wake-Up Timer Event Callback ID
  *          @arg @ref HAL_RTC_TAMPER1_EVENT_CB_ID          Tamper 1 Callback ID
  *          @arg @ref HAL_RTC_TAMPER2_EVENT_CB_ID          Tamper 2 Callback ID
  *          @arg @ref HAL_RTC_MSPINIT_CB_ID                Msp Init callback ID
  *          @arg @ref HAL_RTC_MSPDEINIT_CB_ID              Msp DeInit callback ID
  * @param  pCallback pointer to the Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_RegisterCallback(RTC_HandleTypeDef *hrtc, HAL_RTC_CallbackIDTypeDef CallbackID, pRTC_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if(pCallback == NULL)
  {
    return HAL_ERROR;
  }

  /* Process locked */
  __HAL_LOCK(hrtc);

  if(HAL_RTC_STATE_READY == hrtc->State)
  {
    switch (CallbackID)
    {
    case HAL_RTC_ALARM_A_EVENT_CB_ID :
      hrtc->AlarmAEventCallback = pCallback;
      break;

    case HAL_RTC_ALARM_B_EVENT_CB_ID :
      hrtc->AlarmBEventCallback = pCallback;
      break;

    case HAL_RTC_TIMESTAMP_EVENT_CB_ID :
      hrtc->TimeStampEventCallback = pCallback;
      break;

    case HAL_RTC_WAKEUPTIMER_EVENT_CB_ID :
      hrtc->WakeUpTimerEventCallback = pCallback;
      break;

    case HAL_RTC_TAMPER1_EVENT_CB_ID :
      hrtc->Tamper1EventCallback = pCallback;
      break;

    case HAL_RTC_TAMPER2_EVENT_CB_ID :
      hrtc->Tamper2EventCallback = pCallback;
      break;

   case HAL_RTC_MSPINIT_CB_ID :
      hrtc->MspInitCallback = pCallback;
      break;

   case HAL_RTC_MSPDEINIT_CB_ID :
      hrtc->MspDeInitCallback = pCallback;
      break;

    default :
     /* Return error status */
      status =  HAL_ERROR;
      break;
    }
  }
  else if(HAL_RTC_STATE_RESET == hrtc->State)
  {
    switch (CallbackID)
    {
    case HAL_RTC_MSPINIT_CB_ID :
      hrtc->MspInitCallback = pCallback;
      break;

   case HAL_RTC_MSPDEINIT_CB_ID :
      hrtc->MspDeInitCallback = pCallback;
      break;

    default :
     /* Return error status */
      status =  HAL_ERROR;
      break;
    }
  }
  else
  {
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hrtc);

  return status;
}

/**
  * @brief  Unregister an RTC Callback
  *         RTC callabck is redirected to the weak predefined callback
  * @param  hrtc RTC handle
  * @param  CallbackID ID of the callback to be unregistered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_RTC_ALARM_A_EVENT_CB_ID          Alarm A Event Callback ID
  *          @arg @ref HAL_RTC_ALARM_B_EVENT_CB_ID          Alarm B Event Callback ID
  *          @arg @ref HAL_RTC_TIMESTAMP_EVENT_CB_ID        TimeStamp Event Callback ID
  *          @arg @ref HAL_RTC_WAKEUPTIMER_EVENT_CB_ID      Wake-Up Timer Event Callback ID
  *          @arg @ref HAL_RTC_TAMPER1_EVENT_CB_ID          Tamper 1 Callback ID
  *          @arg @ref HAL_RTC_TAMPER2_EVENT_CB_ID          Tamper 2 Callback ID
  *          @arg @ref HAL_RTC_MSPINIT_CB_ID Msp Init callback ID
  *          @arg @ref HAL_RTC_MSPDEINIT_CB_ID Msp DeInit callback ID
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_UnRegisterCallback(RTC_HandleTypeDef *hrtc, HAL_RTC_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hrtc);

  if(HAL_RTC_STATE_READY == hrtc->State)
  {
    switch (CallbackID)
    {
    case HAL_RTC_ALARM_A_EVENT_CB_ID :
      hrtc->AlarmAEventCallback = HAL_RTC_AlarmAEventCallback;         /* Legacy weak AlarmAEventCallback    */
      break;

    case HAL_RTC_ALARM_B_EVENT_CB_ID :
      hrtc->AlarmBEventCallback = HAL_RTCEx_AlarmBEventCallback;          /* Legacy weak AlarmBEventCallback */
      break;

    case HAL_RTC_TIMESTAMP_EVENT_CB_ID :
      hrtc->TimeStampEventCallback = HAL_RTCEx_TimeStampEventCallback;    /* Legacy weak TimeStampEventCallback    */
      break;

    case HAL_RTC_WAKEUPTIMER_EVENT_CB_ID :
      hrtc->WakeUpTimerEventCallback = HAL_RTCEx_WakeUpTimerEventCallback; /* Legacy weak WakeUpTimerEventCallback */
      break;

    case HAL_RTC_TAMPER1_EVENT_CB_ID :
      hrtc->Tamper1EventCallback = HAL_RTCEx_Tamper1EventCallback;         /* Legacy weak Tamper1EventCallback   */
      break;

    case HAL_RTC_TAMPER2_EVENT_CB_ID :
      hrtc->Tamper2EventCallback = HAL_RTCEx_Tamper2EventCallback;         /* Legacy weak Tamper2EventCallback         */
      break;

    case HAL_RTC_MSPINIT_CB_ID :
      hrtc->MspInitCallback = HAL_RTC_MspInit;
      break;

    case HAL_RTC_MSPDEINIT_CB_ID :
      hrtc->MspDeInitCallback = HAL_RTC_MspDeInit;
      break;

    default :
     /* Return error status */
      status =  HAL_ERROR;
      break;
    }
  }
  else if(HAL_RTC_STATE_RESET == hrtc->State)
  {
    switch (CallbackID)
    {
    case HAL_RTC_MSPINIT_CB_ID :
      hrtc->MspInitCallback = HAL_RTC_MspInit;
      break;

    case HAL_RTC_MSPDEINIT_CB_ID :
      hrtc->MspDeInitCallback = HAL_RTC_MspDeInit;
      break;

    default :
     /* Return error status */
      status =  HAL_ERROR;
      break;
    }
  }
  else
  {
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hrtc);

  return status;
}
#endif /* USE_HAL_RTC_REGISTER_CALLBACKS */

/**
  * @brief  Initializes the RTC MSP.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @retval None
  */
__weak void HAL_RTC_MspInit(RTC_HandleTypeDef* hrtc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hrtc);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_RTC_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitializes the RTC MSP.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @retval None
  */
__weak void HAL_RTC_MspDeInit(RTC_HandleTypeDef* hrtc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hrtc);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_RTC_MspDeInit could be implemented in the user file
   */
}

/**
  * @}
  */

/** @defgroup RTC_Exported_Functions_Group2 RTC Time and Date functions
 *  @brief   RTC Time and Date functions
 *
@verbatim
 ===============================================================================
                 ##### RTC Time and Date functions #####
 ===============================================================================

 [..] This section provides functions allowing to configure Time and Date features

@endverbatim
  * @{
  */

/**
  * @brief  Sets RTC current time.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @param  sTime Pointer to Time structure
  * @param  Format Specifies the format of the entered parameters.
  *          This parameter can be one of the following values:
  *            @arg RTC_FORMAT_BIN: Binary data format
  *            @arg RTC_FORMAT_BCD: BCD data format
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_SetTime(RTC_HandleTypeDef *hrtc, RTC_TimeTypeDef *sTime, uint32_t Format)
{
  uint32_t tmpreg = 0U;

 /* Check the parameters */
  assert_param(IS_RTC_FORMAT(Format));
  assert_param(IS_RTC_DAYLIGHT_SAVING(sTime->DayLightSaving));
  assert_param(IS_RTC_STORE_OPERATION(sTime->StoreOperation));

  /* Process Locked */
  __HAL_LOCK(hrtc);

  hrtc->State = HAL_RTC_STATE_BUSY;

  if(Format == RTC_FORMAT_BIN)
  {
    if((hrtc->Instance->CR & RTC_CR_FMT) != (uint32_t)RESET)
    {
      assert_param(IS_RTC_HOUR12(sTime->Hours));
      assert_param(IS_RTC_HOURFORMAT12(sTime->TimeFormat));
    }
    else
    {
      sTime->TimeFormat = 0x00U;
      assert_param(IS_RTC_HOUR24(sTime->Hours));
    }
    assert_param(IS_RTC_MINUTES(sTime->Minutes));
    assert_param(IS_RTC_SECONDS(sTime->Seconds));

    tmpreg = (uint32_t)(((uint32_t)RTC_ByteToBcd2(sTime->Hours) << 16U) | \
                        ((uint32_t)RTC_ByteToBcd2(sTime->Minutes) << 8U) | \
                        ((uint32_t)RTC_ByteToBcd2(sTime->Seconds)) | \
                        (((uint32_t)sTime->TimeFormat) << 16U));
  }
  else
  {
    if((hrtc->Instance->CR & RTC_CR_FMT) != (uint32_t)RESET)
    {
      assert_param(IS_RTC_HOUR12(RTC_Bcd2ToByte(sTime->Hours)));
      assert_param(IS_RTC_HOURFORMAT12(sTime->TimeFormat));
    }
    else
    {
      sTime->TimeFormat = 0x00U;
      assert_param(IS_RTC_HOUR24(RTC_Bcd2ToByte(sTime->Hours)));
    }
    assert_param(IS_RTC_MINUTES(RTC_Bcd2ToByte(sTime->Minutes)));
    assert_param(IS_RTC_SECONDS(RTC_Bcd2ToByte(sTime->Seconds)));
    tmpreg = (((uint32_t)(sTime->Hours) << 16U) | \
              ((uint32_t)(sTime->Minutes) << 8U) | \
              ((uint32_t)sTime->Seconds) | \
              ((uint32_t)(sTime->TimeFormat) << 16U));
  }

  /* Disable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);

  /* Set Initialization mode */
  if(RTC_EnterInitMode(hrtc) != HAL_OK)
  {
    /* Enable the write protection for RTC registers */
    __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

    /* Set RTC state */
    hrtc->State = HAL_RTC_STATE_ERROR;

    /* Process Unlocked */
    __HAL_UNLOCK(hrtc);

    return HAL_ERROR;
  }
  else
  {
    /* Set the RTC_TR register */
    hrtc->Instance->TR = (uint32_t)(tmpreg & RTC_TR_RESERVED_MASK);

    /* This interface is deprecated. To manage Daylight Saving Time, please use HAL_RTC_DST_xxx functions */
    hrtc->Instance->CR &= (uint32_t)~RTC_CR_BCK;

    /* This interface is deprecated. To manage Daylight Saving Time, please use HAL_RTC_DST_xxx functions */
    hrtc->Instance->CR |= (uint32_t)(sTime->DayLightSaving | sTime->StoreOperation);

    /* Exit Initialization mode */
    hrtc->Instance->ISR &= (uint32_t)~RTC_ISR_INIT;

    /* If  CR_BYPSHAD bit = 0, wait for synchro else this check is not needed */
    if((hrtc->Instance->CR & RTC_CR_BYPSHAD) == RESET)
    {
      if(HAL_RTC_WaitForSynchro(hrtc) != HAL_OK)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_ERROR;

        /* Process Unlocked */
        __HAL_UNLOCK(hrtc);

        return HAL_ERROR;
      }
    }

    /* Enable the write protection for RTC registers */
    __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

   hrtc->State = HAL_RTC_STATE_READY;

   __HAL_UNLOCK(hrtc);

   return HAL_OK;
  }
}

/**
  * @brief  Gets RTC current time.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @param  sTime Pointer to Time structure
  * @param  Format Specifies the format of the entered parameters.
  *          This parameter can be one of the following values:
  *            @arg RTC_FORMAT_BIN: Binary data format
  *            @arg RTC_FORMAT_BCD: BCD data format
  * @note  You can use SubSeconds and SecondFraction (sTime structure fields returned) to convert SubSeconds
  *        value in second fraction ratio with time unit following generic formula:
  *        Second fraction ratio * time_unit= [(SecondFraction-SubSeconds)/(SecondFraction+1)] * time_unit
  *        This conversion can be performed only if no shift operation is pending (ie. SHFP=0) when PREDIV_S >= SS
  * @note You must call HAL_RTC_GetDate() after HAL_RTC_GetTime() to unlock the values
  *        in the higher-order calendar shadow registers to ensure consistency between the time and date values.
  *        Reading RTC current time locks the values in calendar shadow registers until current date is read.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_GetTime(RTC_HandleTypeDef *hrtc, RTC_TimeTypeDef *sTime, uint32_t Format)
{
  uint32_t tmpreg = 0U;

  /* Check the parameters */
  assert_param(IS_RTC_FORMAT(Format));

  /* Get subseconds structure field from the corresponding register */
  sTime->SubSeconds = (uint32_t)(hrtc->Instance->SSR);

  /* Get SecondFraction structure field from the corresponding register field*/
  sTime->SecondFraction = (uint32_t)(hrtc->Instance->PRER & RTC_PRER_PREDIV_S);

  /* Get the TR register */
  tmpreg = (uint32_t)(hrtc->Instance->TR & RTC_TR_RESERVED_MASK);

  /* Fill the structure fields with the read parameters */
  sTime->Hours = (uint8_t)((tmpreg & (RTC_TR_HT | RTC_TR_HU)) >> 16U);
  sTime->Minutes = (uint8_t)((tmpreg & (RTC_TR_MNT | RTC_TR_MNU)) >> 8U);
  sTime->Seconds = (uint8_t)(tmpreg & (RTC_TR_ST | RTC_TR_SU));
  sTime->TimeFormat = (uint8_t)((tmpreg & (RTC_TR_PM)) >> 16U);

  /* Check the input parameters format */
  if(Format == RTC_FORMAT_BIN)
  {
    /* Convert the time structure parameters to Binary format */
    sTime->Hours = (uint8_t)RTC_Bcd2ToByte(sTime->Hours);
    sTime->Minutes = (uint8_t)RTC_Bcd2ToByte(sTime->Minutes);
    sTime->Seconds = (uint8_t)RTC_Bcd2ToByte(sTime->Seconds);
  }

  return HAL_OK;
}

/**
  * @brief  Sets RTC current date.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @param  sDate Pointer to date structure
  * @param  Format specifies the format of the entered parameters.
  *          This parameter can be one of the following values:
  *            @arg RTC_FORMAT_BIN: Binary data format
  *            @arg RTC_FORMAT_BCD: BCD data format
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_SetDate(RTC_HandleTypeDef *hrtc, RTC_DateTypeDef *sDate, uint32_t Format)
{
  uint32_t datetmpreg = 0U;

 /* Check the parameters */
  assert_param(IS_RTC_FORMAT(Format));

 /* Process Locked */
 __HAL_LOCK(hrtc);

  hrtc->State = HAL_RTC_STATE_BUSY;

  if((Format == RTC_FORMAT_BIN) && ((sDate->Month & 0x10U) == 0x10U))
  {
    sDate->Month = (uint8_t)((sDate->Month & (uint8_t)~(0x10U)) + (uint8_t)0x0AU);
  }

  assert_param(IS_RTC_WEEKDAY(sDate->WeekDay));

  if(Format == RTC_FORMAT_BIN)
  {
    assert_param(IS_RTC_YEAR(sDate->Year));
    assert_param(IS_RTC_MONTH(sDate->Month));
    assert_param(IS_RTC_DATE(sDate->Date));

   datetmpreg = (((uint32_t)RTC_ByteToBcd2(sDate->Year) << 16U) | \
                 ((uint32_t)RTC_ByteToBcd2(sDate->Month) << 8U) | \
                 ((uint32_t)RTC_ByteToBcd2(sDate->Date)) | \
                 ((uint32_t)sDate->WeekDay << 13U));
  }
  else
  {
    assert_param(IS_RTC_YEAR(RTC_Bcd2ToByte(sDate->Year)));
    assert_param(IS_RTC_MONTH(RTC_Bcd2ToByte(sDate->Month)));
    assert_param(IS_RTC_DATE(RTC_Bcd2ToByte(sDate->Date)));

    datetmpreg = ((((uint32_t)sDate->Year) << 16U) | \
                  (((uint32_t)sDate->Month) << 8U) | \
                  ((uint32_t)sDate->Date) | \
                  (((uint32_t)sDate->WeekDay) << 13U));
  }

  /* Disable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);

  /* Set Initialization mode */
  if(RTC_EnterInitMode(hrtc) != HAL_OK)
  {
    /* Enable the write protection for RTC registers */
    __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

    /* Set RTC state*/
    hrtc->State = HAL_RTC_STATE_ERROR;

    /* Process Unlocked */
    __HAL_UNLOCK(hrtc);

    return HAL_ERROR;
  }
  else
  {
    /* Set the RTC_DR register */
    hrtc->Instance->DR = (uint32_t)(datetmpreg & RTC_DR_RESERVED_MASK);

    /* Exit Initialization mode */
    hrtc->Instance->ISR &= (uint32_t)~RTC_ISR_INIT;

    /* If  CR_BYPSHAD bit = 0, wait for synchro else this check is not needed */
    if((hrtc->Instance->CR & RTC_CR_BYPSHAD) == RESET)
    {
      if(HAL_RTC_WaitForSynchro(hrtc) != HAL_OK)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_ERROR;

        /* Process Unlocked */
        __HAL_UNLOCK(hrtc);

        return HAL_ERROR;
      }
    }

    /* Enable the write protection for RTC registers */
    __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

    hrtc->State = HAL_RTC_STATE_READY ;

    /* Process Unlocked */
    __HAL_UNLOCK(hrtc);

    return HAL_OK;
  }
}

/**
  * @brief  Gets RTC current date.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @param  sDate Pointer to Date structure
  * @param  Format Specifies the format of the entered parameters.
  *          This parameter can be one of the following values:
  *            @arg RTC_FORMAT_BIN:  Binary data format
  *            @arg RTC_FORMAT_BCD:  BCD data format
  * @note You must call HAL_RTC_GetDate() after HAL_RTC_GetTime() to unlock the values
  * in the higher-order calendar shadow registers to ensure consistency between the time and date values.
  * Reading RTC current time locks the values in calendar shadow registers until Current date is read.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_GetDate(RTC_HandleTypeDef *hrtc, RTC_DateTypeDef *sDate, uint32_t Format)
{
  uint32_t datetmpreg = 0U;

  /* Check the parameters */
  assert_param(IS_RTC_FORMAT(Format));

  /* Get the DR register */
  datetmpreg = (uint32_t)(hrtc->Instance->DR & RTC_DR_RESERVED_MASK);

  /* Fill the structure fields with the read parameters */
  sDate->Year = (uint8_t)((datetmpreg & (RTC_DR_YT | RTC_DR_YU)) >> 16U);
  sDate->Month = (uint8_t)((datetmpreg & (RTC_DR_MT | RTC_DR_MU)) >> 8U);
  sDate->Date = (uint8_t)(datetmpreg & (RTC_DR_DT | RTC_DR_DU));
  sDate->WeekDay = (uint8_t)((datetmpreg & (RTC_DR_WDU)) >> 13U);

  /* Check the input parameters format */
  if(Format == RTC_FORMAT_BIN)
  {
    /* Convert the date structure parameters to Binary format */
    sDate->Year = (uint8_t)RTC_Bcd2ToByte(sDate->Year);
    sDate->Month = (uint8_t)RTC_Bcd2ToByte(sDate->Month);
    sDate->Date = (uint8_t)RTC_Bcd2ToByte(sDate->Date);
  }
  return HAL_OK;
}

/**
  * @}
  */

/** @defgroup RTC_Exported_Functions_Group3 RTC Alarm functions
 *  @brief   RTC Alarm functions
 *
@verbatim
 ===============================================================================
                 ##### RTC Alarm functions #####
 ===============================================================================

 [..] This section provides functions allowing to configure Alarm feature

@endverbatim
  * @{
  */
/**
  * @brief  Sets the specified RTC Alarm.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @param  sAlarm Pointer to Alarm structure
  * @param  Format Specifies the format of the entered parameters.
  *          This parameter can be one of the following values:
  *             @arg RTC_FORMAT_BIN: Binary data format
  *             @arg RTC_FORMAT_BCD: BCD data format
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_SetAlarm(RTC_HandleTypeDef *hrtc, RTC_AlarmTypeDef *sAlarm, uint32_t Format)
{
  uint32_t tickstart = 0U;
  uint32_t tmpreg = 0U, subsecondtmpreg = 0U;

  /* Check the parameters */
  assert_param(IS_RTC_FORMAT(Format));
  assert_param(IS_RTC_ALARM(sAlarm->Alarm));
  assert_param(IS_RTC_ALARM_MASK(sAlarm->AlarmMask));
  assert_param(IS_RTC_ALARM_DATE_WEEKDAY_SEL(sAlarm->AlarmDateWeekDaySel));
  assert_param(IS_RTC_ALARM_SUB_SECOND_VALUE(sAlarm->AlarmTime.SubSeconds));
  assert_param(IS_RTC_ALARM_SUB_SECOND_MASK(sAlarm->AlarmSubSecondMask));

  /* Process Locked */
  __HAL_LOCK(hrtc);

  hrtc->State = HAL_RTC_STATE_BUSY;

  if(Format == RTC_FORMAT_BIN)
  {
    if((hrtc->Instance->CR & RTC_CR_FMT) != (uint32_t)RESET)
    {
      assert_param(IS_RTC_HOUR12(sAlarm->AlarmTime.Hours));
      assert_param(IS_RTC_HOURFORMAT12(sAlarm->AlarmTime.TimeFormat));
    }
    else
    {
      sAlarm->AlarmTime.TimeFormat = 0x00U;
      assert_param(IS_RTC_HOUR24(sAlarm->AlarmTime.Hours));
    }
    assert_param(IS_RTC_MINUTES(sAlarm->AlarmTime.Minutes));
    assert_param(IS_RTC_SECONDS(sAlarm->AlarmTime.Seconds));

    if(sAlarm->AlarmDateWeekDaySel == RTC_ALARMDATEWEEKDAYSEL_DATE)
    {
      assert_param(IS_RTC_ALARM_DATE_WEEKDAY_DATE(sAlarm->AlarmDateWeekDay));
    }
    else
    {
      assert_param(IS_RTC_ALARM_DATE_WEEKDAY_WEEKDAY(sAlarm->AlarmDateWeekDay));
    }

    tmpreg = (((uint32_t)RTC_ByteToBcd2(sAlarm->AlarmTime.Hours) << 16U) | \
              ((uint32_t)RTC_ByteToBcd2(sAlarm->AlarmTime.Minutes) << 8U) | \
              ((uint32_t)RTC_ByteToBcd2(sAlarm->AlarmTime.Seconds)) | \
              ((uint32_t)(sAlarm->AlarmTime.TimeFormat) << 16U) | \
              ((uint32_t)RTC_ByteToBcd2(sAlarm->AlarmDateWeekDay) << 24U) | \
              ((uint32_t)sAlarm->AlarmDateWeekDaySel) | \
              ((uint32_t)sAlarm->AlarmMask));
  }
  else
  {
    if((hrtc->Instance->CR & RTC_CR_FMT) != (uint32_t)RESET)
    {
      assert_param(IS_RTC_HOUR12(RTC_Bcd2ToByte(sAlarm->AlarmTime.Hours)));
      assert_param(IS_RTC_HOURFORMAT12(sAlarm->AlarmTime.TimeFormat));
    }
    else
    {
      sAlarm->AlarmTime.TimeFormat = 0x00U;
      assert_param(IS_RTC_HOUR24(RTC_Bcd2ToByte(sAlarm->AlarmTime.Hours)));
    }

    assert_param(IS_RTC_MINUTES(RTC_Bcd2ToByte(sAlarm->AlarmTime.Minutes)));
    assert_param(IS_RTC_SECONDS(RTC_Bcd2ToByte(sAlarm->AlarmTime.Seconds)));

    if(sAlarm->AlarmDateWeekDaySel == RTC_ALARMDATEWEEKDAYSEL_DATE)
    {
      assert_param(IS_RTC_ALARM_DATE_WEEKDAY_DATE(RTC_Bcd2ToByte(sAlarm->AlarmDateWeekDay)));
    }
    else
    {
      assert_param(IS_RTC_ALARM_DATE_WEEKDAY_WEEKDAY(RTC_Bcd2ToByte(sAlarm->AlarmDateWeekDay)));
    }

    tmpreg = (((uint32_t)(sAlarm->AlarmTime.Hours) << 16U) | \
              ((uint32_t)(sAlarm->AlarmTime.Minutes) << 8U) | \
              ((uint32_t) sAlarm->AlarmTime.Seconds) | \
              ((uint32_t)(sAlarm->AlarmTime.TimeFormat) << 16U) | \
              ((uint32_t)(sAlarm->AlarmDateWeekDay) << 24U) | \
              ((uint32_t)sAlarm->AlarmDateWeekDaySel) | \
              ((uint32_t)sAlarm->AlarmMask));
  }

  /* Configure the Alarm A or Alarm B Sub Second registers */
  subsecondtmpreg = (uint32_t)((uint32_t)(sAlarm->AlarmTime.SubSeconds) | (uint32_t)(sAlarm->AlarmSubSecondMask));

  /* Disable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);

  /* Configure the Alarm register */
  if(sAlarm->Alarm == RTC_ALARM_A)
  {
    /* Disable the Alarm A interrupt */
    __HAL_RTC_ALARMA_DISABLE(hrtc);

    /* In case of interrupt mode is used, the interrupt source must disabled */
    __HAL_RTC_ALARM_DISABLE_IT(hrtc, RTC_IT_ALRA);

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait till RTC ALRAWF flag is set and if Time out is reached exit */
    while(__HAL_RTC_ALARM_GET_FLAG(hrtc, RTC_FLAG_ALRAWF) == RESET)
    {
      if((HAL_GetTick() - tickstart ) > RTC_TIMEOUT_VALUE)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_TIMEOUT;

        /* Process Unlocked */
        __HAL_UNLOCK(hrtc);

        return HAL_TIMEOUT;
      }
    }

    hrtc->Instance->ALRMAR = (uint32_t)tmpreg;
    /* Configure the Alarm A Sub Second register */
    hrtc->Instance->ALRMASSR = subsecondtmpreg;
    /* Configure the Alarm state: Enable Alarm */
    __HAL_RTC_ALARMA_ENABLE(hrtc);
  }
  else
  {
    /* Disable the Alarm B interrupt */
    __HAL_RTC_ALARMB_DISABLE(hrtc);

    /* In case of interrupt mode is used, the interrupt source must disabled */
    __HAL_RTC_ALARM_DISABLE_IT(hrtc, RTC_IT_ALRB);

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait till RTC ALRBWF flag is set and if Time out is reached exit */
    while(__HAL_RTC_ALARM_GET_FLAG(hrtc, RTC_FLAG_ALRBWF) == RESET)
    {
      if((HAL_GetTick() - tickstart ) > RTC_TIMEOUT_VALUE)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_TIMEOUT;

        /* Process Unlocked */
        __HAL_UNLOCK(hrtc);

        return HAL_TIMEOUT;
      }
    }

    hrtc->Instance->ALRMBR = (uint32_t)tmpreg;
    /* Configure the Alarm B Sub Second register */
    hrtc->Instance->ALRMBSSR = subsecondtmpreg;
    /* Configure the Alarm state: Enable Alarm */
    __HAL_RTC_ALARMB_ENABLE(hrtc);
  }

  /* Enable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

  /* Change RTC state */
  hrtc->State = HAL_RTC_STATE_READY;

  /* Process Unlocked */
  __HAL_UNLOCK(hrtc);

  return HAL_OK;
}

/**
  * @brief  Sets the specified RTC Alarm with Interrupt
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @param  sAlarm Pointer to Alarm structure
  * @param  Format Specifies the format of the entered parameters.
  *          This parameter can be one of the following values:
  *             @arg RTC_FORMAT_BIN: Binary data format
  *             @arg RTC_FORMAT_BCD: BCD data format
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_SetAlarm_IT(RTC_HandleTypeDef *hrtc, RTC_AlarmTypeDef *sAlarm, uint32_t Format)
{
  uint32_t tmpreg = 0U, subsecondtmpreg = 0U;
  __IO uint32_t count = RTC_TIMEOUT_VALUE  * (SystemCoreClock / 32U / 1000U) ;

  /* Check the parameters */
  assert_param(IS_RTC_FORMAT(Format));
  assert_param(IS_RTC_ALARM(sAlarm->Alarm));
  assert_param(IS_RTC_ALARM_MASK(sAlarm->AlarmMask));
  assert_param(IS_RTC_ALARM_DATE_WEEKDAY_SEL(sAlarm->AlarmDateWeekDaySel));
  assert_param(IS_RTC_ALARM_SUB_SECOND_VALUE(sAlarm->AlarmTime.SubSeconds));
  assert_param(IS_RTC_ALARM_SUB_SECOND_MASK(sAlarm->AlarmSubSecondMask));

  /* Process Locked */
  __HAL_LOCK(hrtc);

  hrtc->State = HAL_RTC_STATE_BUSY;

  if(Format == RTC_FORMAT_BIN)
  {
    if((hrtc->Instance->CR & RTC_CR_FMT) != (uint32_t)RESET)
    {
      assert_param(IS_RTC_HOUR12(sAlarm->AlarmTime.Hours));
      assert_param(IS_RTC_HOURFORMAT12(sAlarm->AlarmTime.TimeFormat));
    }
    else
    {
      sAlarm->AlarmTime.TimeFormat = 0x00U;
      assert_param(IS_RTC_HOUR24(sAlarm->AlarmTime.Hours));
    }
    assert_param(IS_RTC_MINUTES(sAlarm->AlarmTime.Minutes));
    assert_param(IS_RTC_SECONDS(sAlarm->AlarmTime.Seconds));

    if(sAlarm->AlarmDateWeekDaySel == RTC_ALARMDATEWEEKDAYSEL_DATE)
    {
      assert_param(IS_RTC_ALARM_DATE_WEEKDAY_DATE(sAlarm->AlarmDateWeekDay));
    }
    else
    {
      assert_param(IS_RTC_ALARM_DATE_WEEKDAY_WEEKDAY(sAlarm->AlarmDateWeekDay));
    }
    tmpreg = (((uint32_t)RTC_ByteToBcd2(sAlarm->AlarmTime.Hours) << 16U) | \
              ((uint32_t)RTC_ByteToBcd2(sAlarm->AlarmTime.Minutes) << 8U) | \
              ((uint32_t)RTC_ByteToBcd2(sAlarm->AlarmTime.Seconds)) | \
              ((uint32_t)(sAlarm->AlarmTime.TimeFormat) << 16U) | \
              ((uint32_t)RTC_ByteToBcd2(sAlarm->AlarmDateWeekDay) << 24U) | \
              ((uint32_t)sAlarm->AlarmDateWeekDaySel) | \
              ((uint32_t)sAlarm->AlarmMask));
  }
  else
  {
    if((hrtc->Instance->CR & RTC_CR_FMT) != (uint32_t)RESET)
    {
      assert_param(IS_RTC_HOUR12(RTC_Bcd2ToByte(sAlarm->AlarmTime.Hours)));
      assert_param(IS_RTC_HOURFORMAT12(sAlarm->AlarmTime.TimeFormat));
    }
    else
    {
      sAlarm->AlarmTime.TimeFormat = 0x00U;
      assert_param(IS_RTC_HOUR24(RTC_Bcd2ToByte(sAlarm->AlarmTime.Hours)));
    }

    assert_param(IS_RTC_MINUTES(RTC_Bcd2ToByte(sAlarm->AlarmTime.Minutes)));
    assert_param(IS_RTC_SECONDS(RTC_Bcd2ToByte(sAlarm->AlarmTime.Seconds)));

    if(sAlarm->AlarmDateWeekDaySel == RTC_ALARMDATEWEEKDAYSEL_DATE)
    {
      assert_param(IS_RTC_ALARM_DATE_WEEKDAY_DATE(RTC_Bcd2ToByte(sAlarm->AlarmDateWeekDay)));
    }
    else
    {
      assert_param(IS_RTC_ALARM_DATE_WEEKDAY_WEEKDAY(RTC_Bcd2ToByte(sAlarm->AlarmDateWeekDay)));
    }
    tmpreg = (((uint32_t)(sAlarm->AlarmTime.Hours) << 16U) | \
              ((uint32_t)(sAlarm->AlarmTime.Minutes) << 8U) | \
              ((uint32_t) sAlarm->AlarmTime.Seconds) | \
              ((uint32_t)(sAlarm->AlarmTime.TimeFormat) << 16U) | \
              ((uint32_t)(sAlarm->AlarmDateWeekDay) << 24U) | \
              ((uint32_t)sAlarm->AlarmDateWeekDaySel) | \
              ((uint32_t)sAlarm->AlarmMask));
  }
  /* Configure the Alarm A or Alarm B Sub Second registers */
  subsecondtmpreg = (uint32_t)((uint32_t)(sAlarm->AlarmTime.SubSeconds) | (uint32_t)(sAlarm->AlarmSubSecondMask));

  /* Disable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);

  /* Configure the Alarm register */
  if(sAlarm->Alarm == RTC_ALARM_A)
  {
    /* Disable the Alarm A interrupt */
    __HAL_RTC_ALARMA_DISABLE(hrtc);

    /* Clear flag alarm A */
    __HAL_RTC_ALARM_CLEAR_FLAG(hrtc, RTC_FLAG_ALRAF);

    /* Wait till RTC ALRAWF flag is set and if Time out is reached exit */
    do
    {
      if (count-- == 0U)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_TIMEOUT;

        /* Process Unlocked */
        __HAL_UNLOCK(hrtc);

        return HAL_TIMEOUT;
      }
    }
    while (__HAL_RTC_ALARM_GET_FLAG(hrtc, RTC_FLAG_ALRAWF) == RESET);

    hrtc->Instance->ALRMAR = (uint32_t)tmpreg;
    /* Configure the Alarm A Sub Second register */
    hrtc->Instance->ALRMASSR = subsecondtmpreg;
    /* Configure the Alarm state: Enable Alarm */
    __HAL_RTC_ALARMA_ENABLE(hrtc);
    /* Configure the Alarm interrupt */
    __HAL_RTC_ALARM_ENABLE_IT(hrtc,RTC_IT_ALRA);
  }
  else
  {
    /* Disable the Alarm B interrupt */
    __HAL_RTC_ALARMB_DISABLE(hrtc);

    /* Clear flag alarm B */
    __HAL_RTC_ALARM_CLEAR_FLAG(hrtc, RTC_FLAG_ALRBF);

    /* Wait till RTC ALRBWF flag is set and if Time out is reached exit */
    do
    {
      if (count-- == 0U)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_TIMEOUT;

        /* Process Unlocked */
        __HAL_UNLOCK(hrtc);

        return HAL_TIMEOUT;
      }
    }
    while (__HAL_RTC_ALARM_GET_FLAG(hrtc, RTC_FLAG_ALRBWF) == RESET);

    hrtc->Instance->ALRMBR = (uint32_t)tmpreg;
    /* Configure the Alarm B Sub Second register */
    hrtc->Instance->ALRMBSSR = subsecondtmpreg;
    /* Configure the Alarm state: Enable Alarm */
    __HAL_RTC_ALARMB_ENABLE(hrtc);
    /* Configure the Alarm interrupt */
    __HAL_RTC_ALARM_ENABLE_IT(hrtc, RTC_IT_ALRB);
  }

  /* RTC Alarm Interrupt Configuration: EXTI configuration */
  __HAL_RTC_ALARM_EXTI_ENABLE_IT();

  EXTI->RTSR |= RTC_EXTI_LINE_ALARM_EVENT;

  /* Enable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

  hrtc->State = HAL_RTC_STATE_READY;

  /* Process Unlocked */
  __HAL_UNLOCK(hrtc);

  return HAL_OK;
}

/**
  * @brief  Deactivate the specified RTC Alarm
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @param  Alarm Specifies the Alarm.
  *          This parameter can be one of the following values:
  *            @arg RTC_ALARM_A:  AlarmA
  *            @arg RTC_ALARM_B:  AlarmB
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_DeactivateAlarm(RTC_HandleTypeDef *hrtc, uint32_t Alarm)
{
  uint32_t tickstart = 0U;

  /* Check the parameters */
  assert_param(IS_RTC_ALARM(Alarm));

  /* Process Locked */
  __HAL_LOCK(hrtc);

  hrtc->State = HAL_RTC_STATE_BUSY;

  /* Disable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);

  if(Alarm == RTC_ALARM_A)
  {
    /* AlarmA */
    __HAL_RTC_ALARMA_DISABLE(hrtc);

    /* In case of interrupt mode is used, the interrupt source must disabled */
    __HAL_RTC_ALARM_DISABLE_IT(hrtc, RTC_IT_ALRA);

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait till RTC ALRxWF flag is set and if Time out is reached exit */
    while(__HAL_RTC_ALARM_GET_FLAG(hrtc, RTC_FLAG_ALRAWF) == RESET)
    {
      if((HAL_GetTick() - tickstart ) > RTC_TIMEOUT_VALUE)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_TIMEOUT;

        /* Process Unlocked */
        __HAL_UNLOCK(hrtc);

        return HAL_TIMEOUT;
      }
    }
  }
  else
  {
    /* AlarmB */
    __HAL_RTC_ALARMB_DISABLE(hrtc);

    /* In case of interrupt mode is used, the interrupt source must disabled */
    __HAL_RTC_ALARM_DISABLE_IT(hrtc,RTC_IT_ALRB);

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait till RTC ALRxWF flag is set and if Time out is reached exit */
    while(__HAL_RTC_ALARM_GET_FLAG(hrtc, RTC_FLAG_ALRBWF) == RESET)
    {
      if((HAL_GetTick() - tickstart ) > RTC_TIMEOUT_VALUE)
      {
        /* Enable the write protection for RTC registers */
        __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

        hrtc->State = HAL_RTC_STATE_TIMEOUT;

        /* Process Unlocked */
        __HAL_UNLOCK(hrtc);

        return HAL_TIMEOUT;
      }
    }
  }
  /* Enable the write protection for RTC registers */
  __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

  hrtc->State = HAL_RTC_STATE_READY;

  /* Process Unlocked */
  __HAL_UNLOCK(hrtc);

  return HAL_OK;
}

/**
  * @brief  Gets the RTC Alarm value and masks.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @param  sAlarm Pointer to Date structure
  * @param  Alarm Specifies the Alarm.
  *          This parameter can be one of the following values:
  *             @arg RTC_ALARM_A: AlarmA
  *             @arg RTC_ALARM_B: AlarmB
  * @param  Format Specifies the format of the entered parameters.
  *          This parameter can be one of the following values:
  *             @arg RTC_FORMAT_BIN: Binary data format
  *             @arg RTC_FORMAT_BCD: BCD data format
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_GetAlarm(RTC_HandleTypeDef *hrtc, RTC_AlarmTypeDef *sAlarm, uint32_t Alarm, uint32_t Format)
{
  uint32_t tmpreg = 0U, subsecondtmpreg = 0U;

  /* Check the parameters */
  assert_param(IS_RTC_FORMAT(Format));
  assert_param(IS_RTC_ALARM(Alarm));

  if(Alarm == RTC_ALARM_A)
  {
    /* AlarmA */
    sAlarm->Alarm = RTC_ALARM_A;

    tmpreg = (uint32_t)(hrtc->Instance->ALRMAR);
    subsecondtmpreg = (uint32_t)((hrtc->Instance->ALRMASSR ) & RTC_ALRMASSR_SS);
  }
  else
  {
    sAlarm->Alarm = RTC_ALARM_B;

    tmpreg = (uint32_t)(hrtc->Instance->ALRMBR);
    subsecondtmpreg = (uint32_t)((hrtc->Instance->ALRMBSSR) & RTC_ALRMBSSR_SS);
  }

  /* Fill the structure with the read parameters */
  sAlarm->AlarmTime.Hours = (uint32_t)((tmpreg & (RTC_ALRMAR_HT | RTC_ALRMAR_HU)) >> 16U);
  sAlarm->AlarmTime.Minutes = (uint32_t)((tmpreg & (RTC_ALRMAR_MNT | RTC_ALRMAR_MNU)) >> 8U);
  sAlarm->AlarmTime.Seconds = (uint32_t)(tmpreg & (RTC_ALRMAR_ST | RTC_ALRMAR_SU));
  sAlarm->AlarmTime.TimeFormat = (uint32_t)((tmpreg & RTC_ALRMAR_PM) >> 16U);
  sAlarm->AlarmTime.SubSeconds = (uint32_t) subsecondtmpreg;
  sAlarm->AlarmDateWeekDay = (uint32_t)((tmpreg & (RTC_ALRMAR_DT | RTC_ALRMAR_DU)) >> 24U);
  sAlarm->AlarmDateWeekDaySel = (uint32_t)(tmpreg & RTC_ALRMAR_WDSEL);
  sAlarm->AlarmMask = (uint32_t)(tmpreg & RTC_ALARMMASK_ALL);

  if(Format == RTC_FORMAT_BIN)
  {
    sAlarm->AlarmTime.Hours = RTC_Bcd2ToByte(sAlarm->AlarmTime.Hours);
    sAlarm->AlarmTime.Minutes = RTC_Bcd2ToByte(sAlarm->AlarmTime.Minutes);
    sAlarm->AlarmTime.Seconds = RTC_Bcd2ToByte(sAlarm->AlarmTime.Seconds);
    sAlarm->AlarmDateWeekDay = RTC_Bcd2ToByte(sAlarm->AlarmDateWeekDay);
  }

  return HAL_OK;
}

/**
  * @brief  This function handles Alarm interrupt request.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @retval None
  */
void HAL_RTC_AlarmIRQHandler(RTC_HandleTypeDef* hrtc)
{
  /* Get the AlarmA interrupt source enable status */
  if(__HAL_RTC_ALARM_GET_IT_SOURCE(hrtc, RTC_IT_ALRA) != (uint32_t)RESET)
  {
    /* Get the pending status of the AlarmA Interrupt */
    if(__HAL_RTC_ALARM_GET_FLAG(hrtc, RTC_FLAG_ALRAF) != (uint32_t)RESET)
    {
      /* AlarmA callback */
    #if (USE_HAL_RTC_REGISTER_CALLBACKS == 1)
      hrtc->AlarmAEventCallback(hrtc);
    #else
      HAL_RTC_AlarmAEventCallback(hrtc);
    #endif /* USE_HAL_RTC_REGISTER_CALLBACKS */

      /* Clear the AlarmA interrupt pending bit */
      __HAL_RTC_ALARM_CLEAR_FLAG(hrtc,RTC_FLAG_ALRAF);
    }
  }

  /* Get the AlarmB interrupt source enable status */
  if(__HAL_RTC_ALARM_GET_IT_SOURCE(hrtc, RTC_IT_ALRB) != (uint32_t)RESET)
  {
    /* Get the pending status of the AlarmB Interrupt */
    if(__HAL_RTC_ALARM_GET_FLAG(hrtc, RTC_FLAG_ALRBF) != (uint32_t)RESET)
    {
      /* AlarmB callback */
    #if (USE_HAL_RTC_REGISTER_CALLBACKS == 1)
      hrtc->AlarmBEventCallback(hrtc);
    #else
      HAL_RTCEx_AlarmBEventCallback(hrtc);
    #endif /* USE_HAL_RTC_REGISTER_CALLBACKS */

      /* Clear the AlarmB interrupt pending bit */
      __HAL_RTC_ALARM_CLEAR_FLAG(hrtc,RTC_FLAG_ALRBF);
    }
  }

  /* Clear the EXTI's line Flag for RTC Alarm */
  __HAL_RTC_ALARM_EXTI_CLEAR_FLAG();

  /* Change RTC state */
  hrtc->State = HAL_RTC_STATE_READY;
}

/**
  * @brief  Alarm A callback.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @retval None
  */
__weak void HAL_RTC_AlarmAEventCallback(RTC_HandleTypeDef *hrtc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hrtc);
  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_RTC_AlarmAEventCallback could be implemented in the user file
   */
}

/**
  * @brief  This function handles AlarmA Polling request.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_PollForAlarmAEvent(RTC_HandleTypeDef *hrtc, uint32_t Timeout)
{
  uint32_t tickstart = 0U;

    /* Get tick */
    tickstart = HAL_GetTick();

  while(__HAL_RTC_ALARM_GET_FLAG(hrtc, RTC_FLAG_ALRAF) == RESET)
  {
    if(Timeout != HAL_MAX_DELAY)
    {
      if((Timeout == 0U)||((HAL_GetTick() - tickstart ) > Timeout))
      {
        hrtc->State = HAL_RTC_STATE_TIMEOUT;
        return HAL_TIMEOUT;
      }
    }
  }

  /* Clear the Alarm interrupt pending bit */
  __HAL_RTC_ALARM_CLEAR_FLAG(hrtc, RTC_FLAG_ALRAF);

  /* Change RTC state */
  hrtc->State = HAL_RTC_STATE_READY;

  return HAL_OK;
}

/**
  * @}
  */

/** @defgroup RTC_Exported_Functions_Group4 Peripheral Control functions
 *  @brief   Peripheral Control functions
 *
@verbatim
 ===============================================================================
                     ##### Peripheral Control functions #####
 ===============================================================================
    [..]
    This subsection provides functions allowing to
      (+) Wait for RTC Time and Date Synchronization

@endverbatim
  * @{
  */

/**
  * @brief  Waits until the RTC Time and Date registers (RTC_TR and RTC_DR) are
  *         synchronized with RTC APB clock.
  * @note   The RTC Resynchronization mode is write protected, use the
  *         __HAL_RTC_WRITEPROTECTION_DISABLE() before calling this function.
  * @note   To read the calendar through the shadow registers after Calendar
  *         initialization, calendar update or after wake-up from low power modes
  *         the software must first clear the RSF flag.
  *         The software must then wait until it is set again before reading
  *         the calendar, which means that the calendar registers have been
  *         correctly copied into the RTC_TR and RTC_DR shadow registers.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RTC_WaitForSynchro(RTC_HandleTypeDef* hrtc)
{
  uint32_t tickstart = 0U;

  /* Clear RSF flag */
  hrtc->Instance->ISR &= (uint32_t)RTC_RSF_MASK;

    /* Get tick */
    tickstart = HAL_GetTick();

  /* Wait the registers to be synchronised */
  while((hrtc->Instance->ISR & RTC_ISR_RSF) == (uint32_t)RESET)
  {
    if((HAL_GetTick() - tickstart ) > RTC_TIMEOUT_VALUE)
    {
      return HAL_TIMEOUT;
    }
  }

  return HAL_OK;
}

/**
  * @}
  */

/** @defgroup RTC_Exported_Functions_Group5 Peripheral State functions
 *  @brief   Peripheral State functions
 *
@verbatim
 ===============================================================================
                     ##### Peripheral State functions #####
 ===============================================================================
    [..]
    This subsection provides functions allowing to
      (+) Get RTC state

@endverbatim
  * @{
  */
/**
  * @brief  Returns the RTC state.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @retval HAL state
  */
HAL_RTCStateTypeDef HAL_RTC_GetState(RTC_HandleTypeDef* hrtc)
{
  return hrtc->State;
}

/**
  * @brief  Daylight Saving Time, Add one hour to the calendar in one single operation
  *         without going through the initialization procedure.
  * @param  hrtc RTC handle
  * @retval None
  */
void HAL_RTC_DST_Add1Hour(RTC_HandleTypeDef *hrtc)
{
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);
  SET_BIT(hrtc->Instance->CR, RTC_CR_ADD1H);
  __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);
}

/**
  * @brief  Daylight Saving Time, Substract one hour from the calendar in one
  *         single operation without going through the initialization procedure.
  * @param  hrtc RTC handle
  * @retval None
  */
void HAL_RTC_DST_Sub1Hour(RTC_HandleTypeDef *hrtc)
{
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);
  SET_BIT(hrtc->Instance->CR, RTC_CR_SUB1H);
  __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);
}

/**
  * @brief  Daylight Saving Time, Set the store operation bit.
  * @note   It can be used by the software in order to memorize the DST status.
  * @param  hrtc RTC handle
  * @retval None
  */
void HAL_RTC_DST_SetStoreOperation(RTC_HandleTypeDef *hrtc)
{
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);
  SET_BIT(hrtc->Instance->CR, RTC_CR_BKP);
  __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);
}

/**
  * @brief  Daylight Saving Time, Clear the store operation bit.
  * @param  hrtc RTC handle
  * @retval None
  */
void HAL_RTC_DST_ClearStoreOperation(RTC_HandleTypeDef *hrtc)
{
  __HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);
  CLEAR_BIT(hrtc->Instance->CR, RTC_CR_BKP);
  __HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);
}

/**
  * @brief  Daylight Saving Time, Read the store operation bit.
  * @param  hrtc RTC handle
  * @retval operation see RTC_StoreOperation_Definitions
  */
uint32_t HAL_RTC_DST_ReadStoreOperation(RTC_HandleTypeDef *hrtc)
{
  return READ_BIT(hrtc->Instance->CR, RTC_CR_BKP);
}

/**
  * @}
  */

/**
  * @brief  Enters the RTC Initialization mode.
  * @note   The RTC Initialization mode is write protected, use the
  *         __HAL_RTC_WRITEPROTECTION_DISABLE() before calling this function.
  * @param  hrtc pointer to a RTC_HandleTypeDef structure that contains
  *                the configuration information for RTC.
  * @retval HAL status
  */
HAL_StatusTypeDef RTC_EnterInitMode(RTC_HandleTypeDef* hrtc)
{
  uint32_t tickstart = 0U;

  /* Check if the Initialization mode is set */
  if((hrtc->Instance->ISR & RTC_ISR_INITF) == (uint32_t)RESET)
  {
    /* Set the Initialization mode */
    hrtc->Instance->ISR = (uint32_t)RTC_INIT_MASK;

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait till RTC is in INIT state and if Time out is reached exit */
    while((hrtc->Instance->ISR & RTC_ISR_INITF) == (uint32_t)RESET)
    {
      if((HAL_GetTick() - tickstart ) > RTC_TIMEOUT_VALUE)
      {
        return HAL_TIMEOUT;
      }
    }
  }

  return HAL_OK;
}


/**
  * @brief  Converts a 2 digit decimal to BCD format.
  * @param  Value Byte to be converted
  * @retval Converted byte
  */
uint8_t RTC_ByteToBcd2(uint8_t Value)
{
  uint32_t bcdhigh = 0U;

  while(Value >= 10U)
  {
    bcdhigh++;
    Value -= 10U;
  }

  return  ((uint8_t)(bcdhigh << 4U) | Value);
}

/**
  * @brief  Converts from 2 digit BCD to Binary.
  * @param  Value BCD value to be converted
  * @retval Converted word
  */
uint8_t RTC_Bcd2ToByte(uint8_t Value)
{
  uint32_t tmp = 0U;
  tmp = ((uint8_t)(Value & (uint8_t)0xF0) >> (uint8_t)0x4) * 10;
  return (tmp + (Value & (uint8_t)0x0F));
}

/**
  * @}
  */

#endif /* HAL_RTC_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
