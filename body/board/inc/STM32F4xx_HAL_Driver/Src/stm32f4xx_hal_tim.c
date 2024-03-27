/**
  ******************************************************************************
  * @file    stm32f4xx_hal_tim.c
  * @author  MCD Application Team
  * @brief   TIM HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the Timer (TIM) peripheral:
  *           + TIM Time Base Initialization
  *           + TIM Time Base Start
  *           + TIM Time Base Start Interruption
  *           + TIM Time Base Start DMA
  *           + TIM Output Compare/PWM Initialization
  *           + TIM Output Compare/PWM Channel Configuration
  *           + TIM Output Compare/PWM  Start
  *           + TIM Output Compare/PWM  Start Interruption
  *           + TIM Output Compare/PWM Start DMA
  *           + TIM Input Capture Initialization
  *           + TIM Input Capture Channel Configuration
  *           + TIM Input Capture Start
  *           + TIM Input Capture Start Interruption
  *           + TIM Input Capture Start DMA
  *           + TIM One Pulse Initialization
  *           + TIM One Pulse Channel Configuration
  *           + TIM One Pulse Start
  *           + TIM Encoder Interface Initialization
  *           + TIM Encoder Interface Start
  *           + TIM Encoder Interface Start Interruption
  *           + TIM Encoder Interface Start DMA
  *           + Commutation Event configuration with Interruption and DMA
  *           + TIM OCRef clear configuration
  *           + TIM External Clock configuration
  @verbatim
  ==============================================================================
                      ##### TIMER Generic features #####
  ==============================================================================
  [..] The Timer features include:
       (#) 16-bit up, down, up/down auto-reload counter.
       (#) 16-bit programmable prescaler allowing dividing (also on the fly) the
           counter clock frequency either by any factor between 1 and 65536.
       (#) Up to 4 independent channels for:
           (++) Input Capture
           (++) Output Compare
           (++) PWM generation (Edge and Center-aligned Mode)
           (++) One-pulse mode output
       (#) Synchronization circuit to control the timer with external signals and to interconnect
            several timers together.
       (#) Supports incremental encoder for positioning purposes

            ##### How to use this driver #####
  ==============================================================================
    [..]
     (#) Initialize the TIM low level resources by implementing the following functions
         depending on the selected feature:
           (++) Time Base : HAL_TIM_Base_MspInit()
           (++) Input Capture : HAL_TIM_IC_MspInit()
           (++) Output Compare : HAL_TIM_OC_MspInit()
           (++) PWM generation : HAL_TIM_PWM_MspInit()
           (++) One-pulse mode output : HAL_TIM_OnePulse_MspInit()
           (++) Encoder mode output : HAL_TIM_Encoder_MspInit()

     (#) Initialize the TIM low level resources :
        (##) Enable the TIM interface clock using __HAL_RCC_TIMx_CLK_ENABLE();
        (##) TIM pins configuration
            (+++) Enable the clock for the TIM GPIOs using the following function:
             __HAL_RCC_GPIOx_CLK_ENABLE();
            (+++) Configure these TIM pins in Alternate function mode using HAL_GPIO_Init();

     (#) The external Clock can be configured, if needed (the default clock is the
         internal clock from the APBx), using the following function:
         HAL_TIM_ConfigClockSource, the clock configuration should be done before
         any start function.

     (#) Configure the TIM in the desired functioning mode using one of the
       Initialization function of this driver:
       (++) HAL_TIM_Base_Init: to use the Timer to generate a simple time base
       (++) HAL_TIM_OC_Init and HAL_TIM_OC_ConfigChannel: to use the Timer to generate an
            Output Compare signal.
       (++) HAL_TIM_PWM_Init and HAL_TIM_PWM_ConfigChannel: to use the Timer to generate a
            PWM signal.
       (++) HAL_TIM_IC_Init and HAL_TIM_IC_ConfigChannel: to use the Timer to measure an
            external signal.
       (++) HAL_TIM_OnePulse_Init and HAL_TIM_OnePulse_ConfigChannel: to use the Timer
            in One Pulse Mode.
       (++) HAL_TIM_Encoder_Init: to use the Timer Encoder Interface.

     (#) Activate the TIM peripheral using one of the start functions depending from the feature used:
           (++) Time Base : HAL_TIM_Base_Start(), HAL_TIM_Base_Start_DMA(), HAL_TIM_Base_Start_IT()
           (++) Input Capture :  HAL_TIM_IC_Start(), HAL_TIM_IC_Start_DMA(), HAL_TIM_IC_Start_IT()
           (++) Output Compare : HAL_TIM_OC_Start(), HAL_TIM_OC_Start_DMA(), HAL_TIM_OC_Start_IT()
           (++) PWM generation : HAL_TIM_PWM_Start(), HAL_TIM_PWM_Start_DMA(), HAL_TIM_PWM_Start_IT()
           (++) One-pulse mode output : HAL_TIM_OnePulse_Start(), HAL_TIM_OnePulse_Start_IT()
           (++) Encoder mode output : HAL_TIM_Encoder_Start(), HAL_TIM_Encoder_Start_DMA(), HAL_TIM_Encoder_Start_IT().

     (#) The DMA Burst is managed with the two following functions:
         HAL_TIM_DMABurst_WriteStart()
         HAL_TIM_DMABurst_ReadStart()

    *** Callback registration ***
  =============================================

  [..]
  The compilation define  USE_HAL_TIM_REGISTER_CALLBACKS when set to 1
  allows the user to configure dynamically the driver callbacks.

  [..]
  Use Function HAL_TIM_RegisterCallback() to register a callback.
  HAL_TIM_RegisterCallback() takes as parameters the HAL peripheral handle,
  the Callback ID and a pointer to the user callback function.

  [..]
  Use function HAL_TIM_UnRegisterCallback() to reset a callback to the default
  weak function.
  HAL_TIM_UnRegisterCallback takes as parameters the HAL peripheral handle,
  and the Callback ID.

  [..]
  These functions allow to register/unregister following callbacks:
    (+) Base_MspInitCallback              : TIM Base Msp Init Callback.
    (+) Base_MspDeInitCallback            : TIM Base Msp DeInit Callback.
    (+) IC_MspInitCallback                : TIM IC Msp Init Callback.
    (+) IC_MspDeInitCallback              : TIM IC Msp DeInit Callback.
    (+) OC_MspInitCallback                : TIM OC Msp Init Callback.
    (+) OC_MspDeInitCallback              : TIM OC Msp DeInit Callback.
    (+) PWM_MspInitCallback               : TIM PWM Msp Init Callback.
    (+) PWM_MspDeInitCallback             : TIM PWM Msp DeInit Callback.
    (+) OnePulse_MspInitCallback          : TIM One Pulse Msp Init Callback.
    (+) OnePulse_MspDeInitCallback        : TIM One Pulse Msp DeInit Callback.
    (+) Encoder_MspInitCallback           : TIM Encoder Msp Init Callback.
    (+) Encoder_MspDeInitCallback         : TIM Encoder Msp DeInit Callback.
    (+) HallSensor_MspInitCallback        : TIM Hall Sensor Msp Init Callback.
    (+) HallSensor_MspDeInitCallback      : TIM Hall Sensor Msp DeInit Callback.
    (+) PeriodElapsedCallback             : TIM Period Elapsed Callback.
    (+) PeriodElapsedHalfCpltCallback     : TIM Period Elapsed half complete Callback.
    (+) TriggerCallback                   : TIM Trigger Callback.
    (+) TriggerHalfCpltCallback           : TIM Trigger half complete Callback.
    (+) IC_CaptureCallback                : TIM Input Capture Callback.
    (+) IC_CaptureHalfCpltCallback        : TIM Input Capture half complete Callback.
    (+) OC_DelayElapsedCallback           : TIM Output Compare Delay Elapsed Callback.
    (+) PWM_PulseFinishedCallback         : TIM PWM Pulse Finished Callback.
    (+) PWM_PulseFinishedHalfCpltCallback : TIM PWM Pulse Finished half complete Callback.
    (+) ErrorCallback                     : TIM Error Callback.
    (+) CommutationCallback               : TIM Commutation Callback.
    (+) CommutationHalfCpltCallback       : TIM Commutation half complete Callback.
    (+) BreakCallback                     : TIM Break Callback.

  [..]
By default, after the Init and when the state is HAL_TIM_STATE_RESET
all interrupt callbacks are set to the corresponding weak functions:
  examples HAL_TIM_TriggerCallback(), HAL_TIM_ErrorCallback().

  [..]
  Exception done for MspInit and MspDeInit functions that are reset to the legacy weak
  functionalities in the Init / DeInit only when these callbacks are null
  (not registered beforehand). If not, MspInit or MspDeInit are not null, the Init / DeInit
    keep and use the user MspInit / MspDeInit callbacks(registered beforehand)

  [..]
    Callbacks can be registered / unregistered in HAL_TIM_STATE_READY state only.
    Exception done MspInit / MspDeInit that can be registered / unregistered
    in HAL_TIM_STATE_READY or HAL_TIM_STATE_RESET state,
    thus registered(user) MspInit / DeInit callbacks can be used during the Init / DeInit.
  In that case first register the MspInit/MspDeInit user callbacks
      using HAL_TIM_RegisterCallback() before calling DeInit or Init function.

  [..]
      When The compilation define USE_HAL_TIM_REGISTER_CALLBACKS is set to 0 or
      not defined, the callback registration feature is not available and all callbacks
      are set to the corresponding weak functions.

  @endverbatim
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

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup TIM TIM
  * @brief TIM HAL module driver
  * @{
  */

#ifdef HAL_TIM_MODULE_ENABLED

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/** @addtogroup TIM_Private_Functions
  * @{
  */
static void TIM_OC1_SetConfig(TIM_TypeDef *TIMx, TIM_OC_InitTypeDef *OC_Config);
static void TIM_OC3_SetConfig(TIM_TypeDef *TIMx, TIM_OC_InitTypeDef *OC_Config);
static void TIM_OC4_SetConfig(TIM_TypeDef *TIMx, TIM_OC_InitTypeDef *OC_Config);
static void TIM_TI1_ConfigInputStage(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICFilter);
static void TIM_TI2_SetConfig(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICSelection,
                              uint32_t TIM_ICFilter);
static void TIM_TI2_ConfigInputStage(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICFilter);
static void TIM_TI3_SetConfig(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICSelection,
                              uint32_t TIM_ICFilter);
static void TIM_TI4_SetConfig(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICSelection,
                              uint32_t TIM_ICFilter);
static void TIM_ITRx_SetConfig(TIM_TypeDef *TIMx, uint32_t InputTriggerSource);
static void TIM_DMAPeriodElapsedCplt(DMA_HandleTypeDef *hdma);
static void TIM_DMAPeriodElapsedHalfCplt(DMA_HandleTypeDef *hdma);
static void TIM_DMADelayPulseCplt(DMA_HandleTypeDef *hdma);
static void TIM_DMATriggerCplt(DMA_HandleTypeDef *hdma);
static void TIM_DMATriggerHalfCplt(DMA_HandleTypeDef *hdma);
static HAL_StatusTypeDef TIM_SlaveTimer_SetConfig(TIM_HandleTypeDef *htim,
                                                  TIM_SlaveConfigTypeDef *sSlaveConfig);
/**
  * @}
  */
/* Exported functions --------------------------------------------------------*/

/** @defgroup TIM_Exported_Functions TIM Exported Functions
  * @{
  */

/** @defgroup TIM_Exported_Functions_Group1 TIM Time Base functions
  *  @brief    Time Base functions
  *
@verbatim
  ==============================================================================
              ##### Time Base functions #####
  ==============================================================================
  [..]
    This section provides functions allowing to:
    (+) Initialize and configure the TIM base.
    (+) De-initialize the TIM base.
    (+) Start the Time Base.
    (+) Stop the Time Base.
    (+) Start the Time Base and enable interrupt.
    (+) Stop the Time Base and disable interrupt.
    (+) Start the Time Base and enable DMA transfer.
    (+) Stop the Time Base and disable DMA transfer.

@endverbatim
  * @{
  */
/**
  * @brief  Initializes the TIM Time base Unit according to the specified
  *         parameters in the TIM_HandleTypeDef and initialize the associated handle.
  * @note   Switching from Center Aligned counter mode to Edge counter mode (or reverse)
  *         requires a timer reset to avoid unexpected direction
  *         due to DIR bit readonly in center aligned mode.
  *         Ex: call @ref HAL_TIM_Base_DeInit() before HAL_TIM_Base_Init()
  * @param  htim TIM Base handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Base_Init(TIM_HandleTypeDef *htim)
{
  /* Check the TIM handle allocation */
  if (htim == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));
  assert_param(IS_TIM_COUNTER_MODE(htim->Init.CounterMode));
  assert_param(IS_TIM_CLOCKDIVISION_DIV(htim->Init.ClockDivision));
  assert_param(IS_TIM_AUTORELOAD_PRELOAD(htim->Init.AutoReloadPreload));

  if (htim->State == HAL_TIM_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    htim->Lock = HAL_UNLOCKED;

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
    /* Reset interrupt callbacks to legacy weak callbacks */
    TIM_ResetCallback(htim);

    if (htim->Base_MspInitCallback == NULL)
    {
      htim->Base_MspInitCallback = HAL_TIM_Base_MspInit;
    }
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    htim->Base_MspInitCallback(htim);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    HAL_TIM_Base_MspInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
  }

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_BUSY;

  /* Set the Time Base configuration */
  TIM_Base_SetConfig(htim->Instance, &htim->Init);

  /* Initialize the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_READY;

  /* Initialize the TIM channels state */
  TIM_CHANNEL_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_READY);

  /* Initialize the TIM state*/
  htim->State = HAL_TIM_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  DeInitializes the TIM Base peripheral
  * @param  htim TIM Base handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Base_DeInit(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  htim->State = HAL_TIM_STATE_BUSY;

  /* Disable the TIM Peripheral Clock */
  __HAL_TIM_DISABLE(htim);

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  if (htim->Base_MspDeInitCallback == NULL)
  {
    htim->Base_MspDeInitCallback = HAL_TIM_Base_MspDeInit;
  }
  /* DeInit the low level hardware */
  htim->Base_MspDeInitCallback(htim);
#else
  /* DeInit the low level hardware: GPIO, CLOCK, NVIC */
  HAL_TIM_Base_MspDeInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  /* Change the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_RESET;

  /* Change the TIM channels state */
  TIM_CHANNEL_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_RESET);

  /* Change TIM state */
  htim->State = HAL_TIM_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(htim);

  return HAL_OK;
}

/**
  * @brief  Initializes the TIM Base MSP.
  * @param  htim TIM Base handle
  * @retval None
  */
__weak void HAL_TIM_Base_MspInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_Base_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitializes TIM Base MSP.
  * @param  htim TIM Base handle
  * @retval None
  */
__weak void HAL_TIM_Base_MspDeInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_Base_MspDeInit could be implemented in the user file
   */
}


/**
  * @brief  Starts the TIM Base generation.
  * @param  htim TIM Base handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Base_Start(TIM_HandleTypeDef *htim)
{
  uint32_t tmpsmcr;

  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  /* Check the TIM state */
  if (htim->State != HAL_TIM_STATE_READY)
  {
    return HAL_ERROR;
  }

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_BUSY;

  /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM Base generation.
  * @param  htim TIM Base handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Base_Stop(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_READY;

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Starts the TIM Base generation in interrupt mode.
  * @param  htim TIM Base handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Base_Start_IT(TIM_HandleTypeDef *htim)
{
  uint32_t tmpsmcr;

  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  /* Check the TIM state */
  if (htim->State != HAL_TIM_STATE_READY)
  {
    return HAL_ERROR;
  }

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_BUSY;

  /* Enable the TIM Update interrupt */
  __HAL_TIM_ENABLE_IT(htim, TIM_IT_UPDATE);

  /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM Base generation in interrupt mode.
  * @param  htim TIM Base handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Base_Stop_IT(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  /* Disable the TIM Update interrupt */
  __HAL_TIM_DISABLE_IT(htim, TIM_IT_UPDATE);

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_READY;

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Starts the TIM Base generation in DMA mode.
  * @param  htim TIM Base handle
  * @param  pData The source Buffer address.
  * @param  Length The length of data to be transferred from memory to peripheral.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Base_Start_DMA(TIM_HandleTypeDef *htim, uint32_t *pData, uint16_t Length)
{
  uint32_t tmpsmcr;

  /* Check the parameters */
  assert_param(IS_TIM_DMA_INSTANCE(htim->Instance));

  /* Set the TIM state */
  if (htim->State == HAL_TIM_STATE_BUSY)
  {
    return HAL_BUSY;
  }
  else if (htim->State == HAL_TIM_STATE_READY)
  {
    if ((pData == NULL) && (Length > 0U))
    {
      return HAL_ERROR;
    }
    else
    {
      htim->State = HAL_TIM_STATE_BUSY;
    }
  }
  else
  {
    return HAL_ERROR;
  }

  /* Set the DMA Period elapsed callbacks */
  htim->hdma[TIM_DMA_ID_UPDATE]->XferCpltCallback = TIM_DMAPeriodElapsedCplt;
  htim->hdma[TIM_DMA_ID_UPDATE]->XferHalfCpltCallback = TIM_DMAPeriodElapsedHalfCplt;

  /* Set the DMA error callback */
  htim->hdma[TIM_DMA_ID_UPDATE]->XferErrorCallback = TIM_DMAError ;

  /* Enable the DMA stream */
  if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_UPDATE], (uint32_t)pData, (uint32_t)&htim->Instance->ARR,
                       Length) != HAL_OK)
  {
    /* Return error status */
    return HAL_ERROR;
  }

  /* Enable the TIM Update DMA request */
  __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_UPDATE);

  /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM Base generation in DMA mode.
  * @param  htim TIM Base handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Base_Stop_DMA(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_DMA_INSTANCE(htim->Instance));

  /* Disable the TIM Update DMA request */
  __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_UPDATE);

  (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_UPDATE]);

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_READY;

  /* Return function status */
  return HAL_OK;
}

/**
  * @}
  */

/** @defgroup TIM_Exported_Functions_Group2 TIM Output Compare functions
  *  @brief    TIM Output Compare functions
  *
@verbatim
  ==============================================================================
                  ##### TIM Output Compare functions #####
  ==============================================================================
  [..]
    This section provides functions allowing to:
    (+) Initialize and configure the TIM Output Compare.
    (+) De-initialize the TIM Output Compare.
    (+) Start the TIM Output Compare.
    (+) Stop the TIM Output Compare.
    (+) Start the TIM Output Compare and enable interrupt.
    (+) Stop the TIM Output Compare and disable interrupt.
    (+) Start the TIM Output Compare and enable DMA transfer.
    (+) Stop the TIM Output Compare and disable DMA transfer.

@endverbatim
  * @{
  */
/**
  * @brief  Initializes the TIM Output Compare according to the specified
  *         parameters in the TIM_HandleTypeDef and initializes the associated handle.
  * @note   Switching from Center Aligned counter mode to Edge counter mode (or reverse)
  *         requires a timer reset to avoid unexpected direction
  *         due to DIR bit readonly in center aligned mode.
  *         Ex: call @ref HAL_TIM_OC_DeInit() before HAL_TIM_OC_Init()
  * @param  htim TIM Output Compare handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OC_Init(TIM_HandleTypeDef *htim)
{
  /* Check the TIM handle allocation */
  if (htim == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));
  assert_param(IS_TIM_COUNTER_MODE(htim->Init.CounterMode));
  assert_param(IS_TIM_CLOCKDIVISION_DIV(htim->Init.ClockDivision));
  assert_param(IS_TIM_AUTORELOAD_PRELOAD(htim->Init.AutoReloadPreload));

  if (htim->State == HAL_TIM_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    htim->Lock = HAL_UNLOCKED;

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
    /* Reset interrupt callbacks to legacy weak callbacks */
    TIM_ResetCallback(htim);

    if (htim->OC_MspInitCallback == NULL)
    {
      htim->OC_MspInitCallback = HAL_TIM_OC_MspInit;
    }
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    htim->OC_MspInitCallback(htim);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC and DMA */
    HAL_TIM_OC_MspInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
  }

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_BUSY;

  /* Init the base time for the Output Compare */
  TIM_Base_SetConfig(htim->Instance,  &htim->Init);

  /* Initialize the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_READY;

  /* Initialize the TIM channels state */
  TIM_CHANNEL_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_READY);

  /* Initialize the TIM state*/
  htim->State = HAL_TIM_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  DeInitializes the TIM peripheral
  * @param  htim TIM Output Compare handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OC_DeInit(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  htim->State = HAL_TIM_STATE_BUSY;

  /* Disable the TIM Peripheral Clock */
  __HAL_TIM_DISABLE(htim);

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  if (htim->OC_MspDeInitCallback == NULL)
  {
    htim->OC_MspDeInitCallback = HAL_TIM_OC_MspDeInit;
  }
  /* DeInit the low level hardware */
  htim->OC_MspDeInitCallback(htim);
#else
  /* DeInit the low level hardware: GPIO, CLOCK, NVIC and DMA */
  HAL_TIM_OC_MspDeInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  /* Change the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_RESET;

  /* Change the TIM channels state */
  TIM_CHANNEL_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_RESET);

  /* Change TIM state */
  htim->State = HAL_TIM_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(htim);

  return HAL_OK;
}

/**
  * @brief  Initializes the TIM Output Compare MSP.
  * @param  htim TIM Output Compare handle
  * @retval None
  */
__weak void HAL_TIM_OC_MspInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_OC_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitializes TIM Output Compare MSP.
  * @param  htim TIM Output Compare handle
  * @retval None
  */
__weak void HAL_TIM_OC_MspDeInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_OC_MspDeInit could be implemented in the user file
   */
}

/**
  * @brief  Starts the TIM Output Compare signal generation.
  * @param  htim TIM Output Compare handle
  * @param  Channel TIM Channel to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OC_Start(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  uint32_t tmpsmcr;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Check the TIM channel state */
  if (TIM_CHANNEL_STATE_GET(htim, Channel) != HAL_TIM_CHANNEL_STATE_READY)
  {
    return HAL_ERROR;
  }

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);

  /* Enable the Output compare channel */
  TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_ENABLE);

  if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
  {
    /* Enable the main output */
    __HAL_TIM_MOE_ENABLE(htim);
  }

  /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM Output Compare signal generation.
  * @param  htim TIM Output Compare handle
  * @param  Channel TIM Channel to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OC_Stop(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Disable the Output compare channel */
  TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_DISABLE);

  if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
  {
    /* Disable the Main Output */
    __HAL_TIM_MOE_DISABLE(htim);
  }

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Starts the TIM Output Compare signal generation in interrupt mode.
  * @param  htim TIM Output Compare handle
  * @param  Channel TIM Channel to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OC_Start_IT(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;
  uint32_t tmpsmcr;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Check the TIM channel state */
  if (TIM_CHANNEL_STATE_GET(htim, Channel) != HAL_TIM_CHANNEL_STATE_READY)
  {
    return HAL_ERROR;
  }

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Enable the TIM Capture/Compare 1 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Enable the TIM Capture/Compare 2 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC2);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Enable the TIM Capture/Compare 3 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC3);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Enable the TIM Capture/Compare 4 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC4);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Enable the Output compare channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_ENABLE);

    if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
    {
      /* Enable the main output */
      __HAL_TIM_MOE_ENABLE(htim);
    }

    /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
    if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
    {
      tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
      if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
      {
        __HAL_TIM_ENABLE(htim);
      }
    }
    else
    {
      __HAL_TIM_ENABLE(htim);
    }
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Stops the TIM Output Compare signal generation in interrupt mode.
  * @param  htim TIM Output Compare handle
  * @param  Channel TIM Channel to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OC_Stop_IT(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Disable the TIM Capture/Compare 1 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Disable the TIM Capture/Compare 2 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC2);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Disable the TIM Capture/Compare 3 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC3);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Disable the TIM Capture/Compare 4 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC4);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Disable the Output compare channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_DISABLE);

    if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
    {
      /* Disable the Main Output */
      __HAL_TIM_MOE_DISABLE(htim);
    }

    /* Disable the Peripheral */
    __HAL_TIM_DISABLE(htim);

    /* Set the TIM channel state */
    TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Starts the TIM Output Compare signal generation in DMA mode.
  * @param  htim TIM Output Compare handle
  * @param  Channel TIM Channel to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @param  pData The source Buffer address.
  * @param  Length The length of data to be transferred from memory to TIM peripheral
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OC_Start_DMA(TIM_HandleTypeDef *htim, uint32_t Channel, uint32_t *pData, uint16_t Length)
{
  HAL_StatusTypeDef status = HAL_OK;
  uint32_t tmpsmcr;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Set the TIM channel state */
  if (TIM_CHANNEL_STATE_GET(htim, Channel) == HAL_TIM_CHANNEL_STATE_BUSY)
  {
    return HAL_BUSY;
  }
  else if (TIM_CHANNEL_STATE_GET(htim, Channel) == HAL_TIM_CHANNEL_STATE_READY)
  {
    if ((pData == NULL) && (Length > 0U))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }
  else
  {
    return HAL_ERROR;
  }

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC1]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC1]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC1]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC1], (uint32_t)pData, (uint32_t)&htim->Instance->CCR1,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }

      /* Enable the TIM Capture/Compare 1 DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC2]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC2]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC2]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC2], (uint32_t)pData, (uint32_t)&htim->Instance->CCR2,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }

      /* Enable the TIM Capture/Compare 2 DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC2);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC3]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC3]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC3]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC3], (uint32_t)pData, (uint32_t)&htim->Instance->CCR3,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Capture/Compare 3 DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC3);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC4]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC4]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC4]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC4], (uint32_t)pData, (uint32_t)&htim->Instance->CCR4,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Capture/Compare 4 DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC4);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Enable the Output compare channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_ENABLE);

    if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
    {
      /* Enable the main output */
      __HAL_TIM_MOE_ENABLE(htim);
    }

    /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
    if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
    {
      tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
      if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
      {
        __HAL_TIM_ENABLE(htim);
      }
    }
    else
    {
      __HAL_TIM_ENABLE(htim);
    }
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Stops the TIM Output Compare signal generation in DMA mode.
  * @param  htim TIM Output Compare handle
  * @param  Channel TIM Channel to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OC_Stop_DMA(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Disable the TIM Capture/Compare 1 DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC1);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC1]);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Disable the TIM Capture/Compare 2 DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC2);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC2]);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Disable the TIM Capture/Compare 3 DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC3);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC3]);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Disable the TIM Capture/Compare 4 interrupt */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC4);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC4]);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Disable the Output compare channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_DISABLE);

    if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
    {
      /* Disable the Main Output */
      __HAL_TIM_MOE_DISABLE(htim);
    }

    /* Disable the Peripheral */
    __HAL_TIM_DISABLE(htim);

    /* Set the TIM channel state */
    TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  }

  /* Return function status */
  return status;
}

/**
  * @}
  */

/** @defgroup TIM_Exported_Functions_Group3 TIM PWM functions
  *  @brief    TIM PWM functions
  *
@verbatim
  ==============================================================================
                          ##### TIM PWM functions #####
  ==============================================================================
  [..]
    This section provides functions allowing to:
    (+) Initialize and configure the TIM PWM.
    (+) De-initialize the TIM PWM.
    (+) Start the TIM PWM.
    (+) Stop the TIM PWM.
    (+) Start the TIM PWM and enable interrupt.
    (+) Stop the TIM PWM and disable interrupt.
    (+) Start the TIM PWM and enable DMA transfer.
    (+) Stop the TIM PWM and disable DMA transfer.

@endverbatim
  * @{
  */
/**
  * @brief  Initializes the TIM PWM Time Base according to the specified
  *         parameters in the TIM_HandleTypeDef and initializes the associated handle.
  * @note   Switching from Center Aligned counter mode to Edge counter mode (or reverse)
  *         requires a timer reset to avoid unexpected direction
  *         due to DIR bit readonly in center aligned mode.
  *         Ex: call @ref HAL_TIM_PWM_DeInit() before HAL_TIM_PWM_Init()
  * @param  htim TIM PWM handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_PWM_Init(TIM_HandleTypeDef *htim)
{
  /* Check the TIM handle allocation */
  if (htim == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));
  assert_param(IS_TIM_COUNTER_MODE(htim->Init.CounterMode));
  assert_param(IS_TIM_CLOCKDIVISION_DIV(htim->Init.ClockDivision));
  assert_param(IS_TIM_AUTORELOAD_PRELOAD(htim->Init.AutoReloadPreload));

  if (htim->State == HAL_TIM_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    htim->Lock = HAL_UNLOCKED;

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
    /* Reset interrupt callbacks to legacy weak callbacks */
    TIM_ResetCallback(htim);

    if (htim->PWM_MspInitCallback == NULL)
    {
      htim->PWM_MspInitCallback = HAL_TIM_PWM_MspInit;
    }
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    htim->PWM_MspInitCallback(htim);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC and DMA */
    HAL_TIM_PWM_MspInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
  }

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_BUSY;

  /* Init the base time for the PWM */
  TIM_Base_SetConfig(htim->Instance, &htim->Init);

  /* Initialize the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_READY;

  /* Initialize the TIM channels state */
  TIM_CHANNEL_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_READY);

  /* Initialize the TIM state*/
  htim->State = HAL_TIM_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  DeInitializes the TIM peripheral
  * @param  htim TIM PWM handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_PWM_DeInit(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  htim->State = HAL_TIM_STATE_BUSY;

  /* Disable the TIM Peripheral Clock */
  __HAL_TIM_DISABLE(htim);

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  if (htim->PWM_MspDeInitCallback == NULL)
  {
    htim->PWM_MspDeInitCallback = HAL_TIM_PWM_MspDeInit;
  }
  /* DeInit the low level hardware */
  htim->PWM_MspDeInitCallback(htim);
#else
  /* DeInit the low level hardware: GPIO, CLOCK, NVIC and DMA */
  HAL_TIM_PWM_MspDeInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  /* Change the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_RESET;

  /* Change the TIM channels state */
  TIM_CHANNEL_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_RESET);

  /* Change TIM state */
  htim->State = HAL_TIM_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(htim);

  return HAL_OK;
}

/**
  * @brief  Initializes the TIM PWM MSP.
  * @param  htim TIM PWM handle
  * @retval None
  */
__weak void HAL_TIM_PWM_MspInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_PWM_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitializes TIM PWM MSP.
  * @param  htim TIM PWM handle
  * @retval None
  */
__weak void HAL_TIM_PWM_MspDeInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_PWM_MspDeInit could be implemented in the user file
   */
}

/**
  * @brief  Starts the PWM signal generation.
  * @param  htim TIM handle
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_PWM_Start(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  uint32_t tmpsmcr;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Check the TIM channel state */
  if (TIM_CHANNEL_STATE_GET(htim, Channel) != HAL_TIM_CHANNEL_STATE_READY)
  {
    return HAL_ERROR;
  }

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);

  /* Enable the Capture compare channel */
  TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_ENABLE);

  if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
  {
    /* Enable the main output */
    __HAL_TIM_MOE_ENABLE(htim);
  }

  /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the PWM signal generation.
  * @param  htim TIM PWM handle
  * @param  Channel TIM Channels to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_PWM_Stop(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Disable the Capture compare channel */
  TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_DISABLE);

  if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
  {
    /* Disable the Main Output */
    __HAL_TIM_MOE_DISABLE(htim);
  }

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Starts the PWM signal generation in interrupt mode.
  * @param  htim TIM PWM handle
  * @param  Channel TIM Channel to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_PWM_Start_IT(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;
  uint32_t tmpsmcr;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Check the TIM channel state */
  if (TIM_CHANNEL_STATE_GET(htim, Channel) != HAL_TIM_CHANNEL_STATE_READY)
  {
    return HAL_ERROR;
  }

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Enable the TIM Capture/Compare 1 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Enable the TIM Capture/Compare 2 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC2);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Enable the TIM Capture/Compare 3 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC3);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Enable the TIM Capture/Compare 4 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC4);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Enable the Capture compare channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_ENABLE);

    if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
    {
      /* Enable the main output */
      __HAL_TIM_MOE_ENABLE(htim);
    }

    /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
    if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
    {
      tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
      if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
      {
        __HAL_TIM_ENABLE(htim);
      }
    }
    else
    {
      __HAL_TIM_ENABLE(htim);
    }
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Stops the PWM signal generation in interrupt mode.
  * @param  htim TIM PWM handle
  * @param  Channel TIM Channels to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_PWM_Stop_IT(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Disable the TIM Capture/Compare 1 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Disable the TIM Capture/Compare 2 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC2);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Disable the TIM Capture/Compare 3 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC3);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Disable the TIM Capture/Compare 4 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC4);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Disable the Capture compare channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_DISABLE);

    if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
    {
      /* Disable the Main Output */
      __HAL_TIM_MOE_DISABLE(htim);
    }

    /* Disable the Peripheral */
    __HAL_TIM_DISABLE(htim);

    /* Set the TIM channel state */
    TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Starts the TIM PWM signal generation in DMA mode.
  * @param  htim TIM PWM handle
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @param  pData The source Buffer address.
  * @param  Length The length of data to be transferred from memory to TIM peripheral
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_PWM_Start_DMA(TIM_HandleTypeDef *htim, uint32_t Channel, uint32_t *pData, uint16_t Length)
{
  HAL_StatusTypeDef status = HAL_OK;
  uint32_t tmpsmcr;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Set the TIM channel state */
  if (TIM_CHANNEL_STATE_GET(htim, Channel) == HAL_TIM_CHANNEL_STATE_BUSY)
  {
    return HAL_BUSY;
  }
  else if (TIM_CHANNEL_STATE_GET(htim, Channel) == HAL_TIM_CHANNEL_STATE_READY)
  {
    if ((pData == NULL) && (Length > 0U))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }
  else
  {
    return HAL_ERROR;
  }

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC1]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC1]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC1]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC1], (uint32_t)pData, (uint32_t)&htim->Instance->CCR1,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }

      /* Enable the TIM Capture/Compare 1 DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC2]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC2]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC2]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC2], (uint32_t)pData, (uint32_t)&htim->Instance->CCR2,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Capture/Compare 2 DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC2);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC3]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC3]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC3]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC3], (uint32_t)pData, (uint32_t)&htim->Instance->CCR3,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Output Capture/Compare 3 request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC3);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC4]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC4]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC4]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC4], (uint32_t)pData, (uint32_t)&htim->Instance->CCR4,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Capture/Compare 4 DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC4);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Enable the Capture compare channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_ENABLE);

    if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
    {
      /* Enable the main output */
      __HAL_TIM_MOE_ENABLE(htim);
    }

    /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
    if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
    {
      tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
      if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
      {
        __HAL_TIM_ENABLE(htim);
      }
    }
    else
    {
      __HAL_TIM_ENABLE(htim);
    }
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Stops the TIM PWM signal generation in DMA mode.
  * @param  htim TIM PWM handle
  * @param  Channel TIM Channels to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_PWM_Stop_DMA(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Disable the TIM Capture/Compare 1 DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC1);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC1]);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Disable the TIM Capture/Compare 2 DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC2);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC2]);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Disable the TIM Capture/Compare 3 DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC3);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC3]);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Disable the TIM Capture/Compare 4 interrupt */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC4);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC4]);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Disable the Capture compare channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_DISABLE);

    if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
    {
      /* Disable the Main Output */
      __HAL_TIM_MOE_DISABLE(htim);
    }

    /* Disable the Peripheral */
    __HAL_TIM_DISABLE(htim);

    /* Set the TIM channel state */
    TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  }

  /* Return function status */
  return status;
}

/**
  * @}
  */

/** @defgroup TIM_Exported_Functions_Group4 TIM Input Capture functions
  *  @brief    TIM Input Capture functions
  *
@verbatim
  ==============================================================================
              ##### TIM Input Capture functions #####
  ==============================================================================
 [..]
   This section provides functions allowing to:
   (+) Initialize and configure the TIM Input Capture.
   (+) De-initialize the TIM Input Capture.
   (+) Start the TIM Input Capture.
   (+) Stop the TIM Input Capture.
   (+) Start the TIM Input Capture and enable interrupt.
   (+) Stop the TIM Input Capture and disable interrupt.
   (+) Start the TIM Input Capture and enable DMA transfer.
   (+) Stop the TIM Input Capture and disable DMA transfer.

@endverbatim
  * @{
  */
/**
  * @brief  Initializes the TIM Input Capture Time base according to the specified
  *         parameters in the TIM_HandleTypeDef and initializes the associated handle.
  * @note   Switching from Center Aligned counter mode to Edge counter mode (or reverse)
  *         requires a timer reset to avoid unexpected direction
  *         due to DIR bit readonly in center aligned mode.
  *         Ex: call @ref HAL_TIM_IC_DeInit() before HAL_TIM_IC_Init()
  * @param  htim TIM Input Capture handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_IC_Init(TIM_HandleTypeDef *htim)
{
  /* Check the TIM handle allocation */
  if (htim == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));
  assert_param(IS_TIM_COUNTER_MODE(htim->Init.CounterMode));
  assert_param(IS_TIM_CLOCKDIVISION_DIV(htim->Init.ClockDivision));
  assert_param(IS_TIM_AUTORELOAD_PRELOAD(htim->Init.AutoReloadPreload));

  if (htim->State == HAL_TIM_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    htim->Lock = HAL_UNLOCKED;

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
    /* Reset interrupt callbacks to legacy weak callbacks */
    TIM_ResetCallback(htim);

    if (htim->IC_MspInitCallback == NULL)
    {
      htim->IC_MspInitCallback = HAL_TIM_IC_MspInit;
    }
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    htim->IC_MspInitCallback(htim);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC and DMA */
    HAL_TIM_IC_MspInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
  }

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_BUSY;

  /* Init the base time for the input capture */
  TIM_Base_SetConfig(htim->Instance, &htim->Init);

  /* Initialize the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_READY;

  /* Initialize the TIM channels state */
  TIM_CHANNEL_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_READY);

  /* Initialize the TIM state*/
  htim->State = HAL_TIM_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  DeInitializes the TIM peripheral
  * @param  htim TIM Input Capture handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_IC_DeInit(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  htim->State = HAL_TIM_STATE_BUSY;

  /* Disable the TIM Peripheral Clock */
  __HAL_TIM_DISABLE(htim);

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  if (htim->IC_MspDeInitCallback == NULL)
  {
    htim->IC_MspDeInitCallback = HAL_TIM_IC_MspDeInit;
  }
  /* DeInit the low level hardware */
  htim->IC_MspDeInitCallback(htim);
#else
  /* DeInit the low level hardware: GPIO, CLOCK, NVIC and DMA */
  HAL_TIM_IC_MspDeInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  /* Change the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_RESET;

  /* Change the TIM channels state */
  TIM_CHANNEL_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET_ALL(htim, HAL_TIM_CHANNEL_STATE_RESET);

  /* Change TIM state */
  htim->State = HAL_TIM_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(htim);

  return HAL_OK;
}

/**
  * @brief  Initializes the TIM Input Capture MSP.
  * @param  htim TIM Input Capture handle
  * @retval None
  */
__weak void HAL_TIM_IC_MspInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_IC_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitializes TIM Input Capture MSP.
  * @param  htim TIM handle
  * @retval None
  */
__weak void HAL_TIM_IC_MspDeInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_IC_MspDeInit could be implemented in the user file
   */
}

/**
  * @brief  Starts the TIM Input Capture measurement.
  * @param  htim TIM Input Capture handle
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_IC_Start(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  uint32_t tmpsmcr;
  HAL_TIM_ChannelStateTypeDef channel_state = TIM_CHANNEL_STATE_GET(htim, Channel);
  HAL_TIM_ChannelStateTypeDef complementary_channel_state = TIM_CHANNEL_N_STATE_GET(htim, Channel);

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Check the TIM channel state */
  if ((channel_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_state != HAL_TIM_CHANNEL_STATE_READY))
  {
    return HAL_ERROR;
  }

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);

  /* Enable the Input Capture channel */
  TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_ENABLE);

  /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM Input Capture measurement.
  * @param  htim TIM Input Capture handle
  * @param  Channel TIM Channels to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_IC_Stop(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Disable the Input Capture channel */
  TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_DISABLE);

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Starts the TIM Input Capture measurement in interrupt mode.
  * @param  htim TIM Input Capture handle
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_IC_Start_IT(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;
  uint32_t tmpsmcr;

  HAL_TIM_ChannelStateTypeDef channel_state = TIM_CHANNEL_STATE_GET(htim, Channel);
  HAL_TIM_ChannelStateTypeDef complementary_channel_state = TIM_CHANNEL_N_STATE_GET(htim, Channel);

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  /* Check the TIM channel state */
  if ((channel_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_state != HAL_TIM_CHANNEL_STATE_READY))
  {
    return HAL_ERROR;
  }

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Enable the TIM Capture/Compare 1 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Enable the TIM Capture/Compare 2 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC2);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Enable the TIM Capture/Compare 3 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC3);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Enable the TIM Capture/Compare 4 interrupt */
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC4);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Enable the Input Capture channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_ENABLE);

    /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
    if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
    {
      tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
      if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
      {
        __HAL_TIM_ENABLE(htim);
      }
    }
    else
    {
      __HAL_TIM_ENABLE(htim);
    }
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Stops the TIM Input Capture measurement in interrupt mode.
  * @param  htim TIM Input Capture handle
  * @param  Channel TIM Channels to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_IC_Stop_IT(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Disable the TIM Capture/Compare 1 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Disable the TIM Capture/Compare 2 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC2);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Disable the TIM Capture/Compare 3 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC3);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Disable the TIM Capture/Compare 4 interrupt */
      __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC4);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Disable the Input Capture channel */
    TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_DISABLE);

    /* Disable the Peripheral */
    __HAL_TIM_DISABLE(htim);

    /* Set the TIM channel state */
    TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Starts the TIM Input Capture measurement in DMA mode.
  * @param  htim TIM Input Capture handle
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @param  pData The destination Buffer address.
  * @param  Length The length of data to be transferred from TIM peripheral to memory.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_IC_Start_DMA(TIM_HandleTypeDef *htim, uint32_t Channel, uint32_t *pData, uint16_t Length)
{
  HAL_StatusTypeDef status = HAL_OK;
  uint32_t tmpsmcr;

  HAL_TIM_ChannelStateTypeDef channel_state = TIM_CHANNEL_STATE_GET(htim, Channel);
  HAL_TIM_ChannelStateTypeDef complementary_channel_state = TIM_CHANNEL_N_STATE_GET(htim, Channel);

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));
  assert_param(IS_TIM_DMA_CC_INSTANCE(htim->Instance));

  /* Set the TIM channel state */
  if ((channel_state == HAL_TIM_CHANNEL_STATE_BUSY)
      || (complementary_channel_state == HAL_TIM_CHANNEL_STATE_BUSY))
  {
    return HAL_BUSY;
  }
  else if ((channel_state == HAL_TIM_CHANNEL_STATE_READY)
           && (complementary_channel_state == HAL_TIM_CHANNEL_STATE_READY))
  {
    if ((pData == NULL) && (Length > 0U))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }
  else
  {
    return HAL_ERROR;
  }

  /* Enable the Input Capture channel */
  TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_ENABLE);

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC1]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC1]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC1]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC1], (uint32_t)&htim->Instance->CCR1, (uint32_t)pData,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Capture/Compare 1 DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC2]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC2]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC2]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC2], (uint32_t)&htim->Instance->CCR2, (uint32_t)pData,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Capture/Compare 2  DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC2);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC3]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC3]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC3]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC3], (uint32_t)&htim->Instance->CCR3, (uint32_t)pData,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Capture/Compare 3  DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC3);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC4]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC4]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC4]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC4], (uint32_t)&htim->Instance->CCR4, (uint32_t)pData,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Capture/Compare 4  DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC4);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  /* Enable the Peripheral, except in trigger mode where enable is automatically done with trigger */
  if (IS_TIM_SLAVE_INSTANCE(htim->Instance))
  {
    tmpsmcr = htim->Instance->SMCR & TIM_SMCR_SMS;
    if (!IS_TIM_SLAVEMODE_TRIGGER_ENABLED(tmpsmcr))
    {
      __HAL_TIM_ENABLE(htim);
    }
  }
  else
  {
    __HAL_TIM_ENABLE(htim);
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Stops the TIM Input Capture measurement in DMA mode.
  * @param  htim TIM Input Capture handle
  * @param  Channel TIM Channels to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_IC_Stop_DMA(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));
  assert_param(IS_TIM_DMA_CC_INSTANCE(htim->Instance));

  /* Disable the Input Capture channel */
  TIM_CCxChannelCmd(htim->Instance, Channel, TIM_CCx_DISABLE);

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Disable the TIM Capture/Compare 1 DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC1);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC1]);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Disable the TIM Capture/Compare 2 DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC2);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC2]);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Disable the TIM Capture/Compare 3  DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC3);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC3]);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Disable the TIM Capture/Compare 4  DMA request */
      __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC4);
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC4]);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Disable the Peripheral */
    __HAL_TIM_DISABLE(htim);

    /* Set the TIM channel state */
    TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  }

  /* Return function status */
  return status;
}
/**
  * @}
  */

/** @defgroup TIM_Exported_Functions_Group5 TIM One Pulse functions
  *  @brief    TIM One Pulse functions
  *
@verbatim
  ==============================================================================
                        ##### TIM One Pulse functions #####
  ==============================================================================
  [..]
    This section provides functions allowing to:
    (+) Initialize and configure the TIM One Pulse.
    (+) De-initialize the TIM One Pulse.
    (+) Start the TIM One Pulse.
    (+) Stop the TIM One Pulse.
    (+) Start the TIM One Pulse and enable interrupt.
    (+) Stop the TIM One Pulse and disable interrupt.
    (+) Start the TIM One Pulse and enable DMA transfer.
    (+) Stop the TIM One Pulse and disable DMA transfer.

@endverbatim
  * @{
  */
/**
  * @brief  Initializes the TIM One Pulse Time Base according to the specified
  *         parameters in the TIM_HandleTypeDef and initializes the associated handle.
  * @note   Switching from Center Aligned counter mode to Edge counter mode (or reverse)
  *         requires a timer reset to avoid unexpected direction
  *         due to DIR bit readonly in center aligned mode.
  *         Ex: call @ref HAL_TIM_OnePulse_DeInit() before HAL_TIM_OnePulse_Init()
  * @note   When the timer instance is initialized in One Pulse mode, timer
  *         channels 1 and channel 2 are reserved and cannot be used for other
  *         purpose.
  * @param  htim TIM One Pulse handle
  * @param  OnePulseMode Select the One pulse mode.
  *         This parameter can be one of the following values:
  *            @arg TIM_OPMODE_SINGLE: Only one pulse will be generated.
  *            @arg TIM_OPMODE_REPETITIVE: Repetitive pulses will be generated.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OnePulse_Init(TIM_HandleTypeDef *htim, uint32_t OnePulseMode)
{
  /* Check the TIM handle allocation */
  if (htim == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));
  assert_param(IS_TIM_COUNTER_MODE(htim->Init.CounterMode));
  assert_param(IS_TIM_CLOCKDIVISION_DIV(htim->Init.ClockDivision));
  assert_param(IS_TIM_OPM_MODE(OnePulseMode));
  assert_param(IS_TIM_AUTORELOAD_PRELOAD(htim->Init.AutoReloadPreload));

  if (htim->State == HAL_TIM_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    htim->Lock = HAL_UNLOCKED;

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
    /* Reset interrupt callbacks to legacy weak callbacks */
    TIM_ResetCallback(htim);

    if (htim->OnePulse_MspInitCallback == NULL)
    {
      htim->OnePulse_MspInitCallback = HAL_TIM_OnePulse_MspInit;
    }
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    htim->OnePulse_MspInitCallback(htim);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC and DMA */
    HAL_TIM_OnePulse_MspInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
  }

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_BUSY;

  /* Configure the Time base in the One Pulse Mode */
  TIM_Base_SetConfig(htim->Instance, &htim->Init);

  /* Reset the OPM Bit */
  htim->Instance->CR1 &= ~TIM_CR1_OPM;

  /* Configure the OPM Mode */
  htim->Instance->CR1 |= OnePulseMode;

  /* Initialize the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_READY;

  /* Initialize the TIM channels state */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);

  /* Initialize the TIM state*/
  htim->State = HAL_TIM_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  DeInitializes the TIM One Pulse
  * @param  htim TIM One Pulse handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OnePulse_DeInit(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  htim->State = HAL_TIM_STATE_BUSY;

  /* Disable the TIM Peripheral Clock */
  __HAL_TIM_DISABLE(htim);

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  if (htim->OnePulse_MspDeInitCallback == NULL)
  {
    htim->OnePulse_MspDeInitCallback = HAL_TIM_OnePulse_MspDeInit;
  }
  /* DeInit the low level hardware */
  htim->OnePulse_MspDeInitCallback(htim);
#else
  /* DeInit the low level hardware: GPIO, CLOCK, NVIC */
  HAL_TIM_OnePulse_MspDeInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  /* Change the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_RESET;

  /* Set the TIM channel state */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_RESET);

  /* Change TIM state */
  htim->State = HAL_TIM_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(htim);

  return HAL_OK;
}

/**
  * @brief  Initializes the TIM One Pulse MSP.
  * @param  htim TIM One Pulse handle
  * @retval None
  */
__weak void HAL_TIM_OnePulse_MspInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_OnePulse_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitializes TIM One Pulse MSP.
  * @param  htim TIM One Pulse handle
  * @retval None
  */
__weak void HAL_TIM_OnePulse_MspDeInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_OnePulse_MspDeInit could be implemented in the user file
   */
}

/**
  * @brief  Starts the TIM One Pulse signal generation.
  * @note Though OutputChannel parameter is deprecated and ignored by the function
  *        it has been kept to avoid HAL_TIM API compatibility break.
  * @note The pulse output channel is determined when calling
  *       @ref HAL_TIM_OnePulse_ConfigChannel().
  * @param  htim TIM One Pulse handle
  * @param  OutputChannel See note above
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OnePulse_Start(TIM_HandleTypeDef *htim, uint32_t OutputChannel)
{
  HAL_TIM_ChannelStateTypeDef channel_1_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef channel_2_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_2);
  HAL_TIM_ChannelStateTypeDef complementary_channel_1_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef complementary_channel_2_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_2);

  /* Prevent unused argument(s) compilation warning */
  UNUSED(OutputChannel);

  /* Check the TIM channels state */
  if ((channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
      || (channel_2_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_2_state != HAL_TIM_CHANNEL_STATE_READY))
  {
    return HAL_ERROR;
  }

  /* Set the TIM channels state */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);

  /* Enable the Capture compare and the Input Capture channels
    (in the OPM Mode the two possible channels that can be used are TIM_CHANNEL_1 and TIM_CHANNEL_2)
    if TIM_CHANNEL_1 is used as output, the TIM_CHANNEL_2 will be used as input and
    if TIM_CHANNEL_1 is used as input, the TIM_CHANNEL_2 will be used as output
    whatever the combination, the TIM_CHANNEL_1 and TIM_CHANNEL_2 should be enabled together

    No need to enable the counter, it's enabled automatically by hardware
    (the counter starts in response to a stimulus and generate a pulse */

  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_ENABLE);

  if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
  {
    /* Enable the main output */
    __HAL_TIM_MOE_ENABLE(htim);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM One Pulse signal generation.
  * @note Though OutputChannel parameter is deprecated and ignored by the function
  *        it has been kept to avoid HAL_TIM API compatibility break.
  * @note The pulse output channel is determined when calling
  *       @ref HAL_TIM_OnePulse_ConfigChannel().
  * @param  htim TIM One Pulse handle
  * @param  OutputChannel See note above
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OnePulse_Stop(TIM_HandleTypeDef *htim, uint32_t OutputChannel)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(OutputChannel);

  /* Disable the Capture compare and the Input Capture channels
  (in the OPM Mode the two possible channels that can be used are TIM_CHANNEL_1 and TIM_CHANNEL_2)
  if TIM_CHANNEL_1 is used as output, the TIM_CHANNEL_2 will be used as input and
  if TIM_CHANNEL_1 is used as input, the TIM_CHANNEL_2 will be used as output
  whatever the combination, the TIM_CHANNEL_1 and TIM_CHANNEL_2 should be disabled together */

  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_DISABLE);

  if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
  {
    /* Disable the Main Output */
    __HAL_TIM_MOE_DISABLE(htim);
  }

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM channels state */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Starts the TIM One Pulse signal generation in interrupt mode.
  * @note Though OutputChannel parameter is deprecated and ignored by the function
  *        it has been kept to avoid HAL_TIM API compatibility break.
  * @note The pulse output channel is determined when calling
  *       @ref HAL_TIM_OnePulse_ConfigChannel().
  * @param  htim TIM One Pulse handle
  * @param  OutputChannel See note above
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OnePulse_Start_IT(TIM_HandleTypeDef *htim, uint32_t OutputChannel)
{
  HAL_TIM_ChannelStateTypeDef channel_1_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef channel_2_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_2);
  HAL_TIM_ChannelStateTypeDef complementary_channel_1_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef complementary_channel_2_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_2);

  /* Prevent unused argument(s) compilation warning */
  UNUSED(OutputChannel);

  /* Check the TIM channels state */
  if ((channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
      || (channel_2_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
      || (complementary_channel_2_state != HAL_TIM_CHANNEL_STATE_READY))
  {
    return HAL_ERROR;
  }

  /* Set the TIM channels state */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);

  /* Enable the Capture compare and the Input Capture channels
    (in the OPM Mode the two possible channels that can be used are TIM_CHANNEL_1 and TIM_CHANNEL_2)
    if TIM_CHANNEL_1 is used as output, the TIM_CHANNEL_2 will be used as input and
    if TIM_CHANNEL_1 is used as input, the TIM_CHANNEL_2 will be used as output
    whatever the combination, the TIM_CHANNEL_1 and TIM_CHANNEL_2 should be enabled together

    No need to enable the counter, it's enabled automatically by hardware
    (the counter starts in response to a stimulus and generate a pulse */

  /* Enable the TIM Capture/Compare 1 interrupt */
  __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC1);

  /* Enable the TIM Capture/Compare 2 interrupt */
  __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC2);

  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_ENABLE);

  if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
  {
    /* Enable the main output */
    __HAL_TIM_MOE_ENABLE(htim);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM One Pulse signal generation in interrupt mode.
  * @note Though OutputChannel parameter is deprecated and ignored by the function
  *        it has been kept to avoid HAL_TIM API compatibility break.
  * @note The pulse output channel is determined when calling
  *       @ref HAL_TIM_OnePulse_ConfigChannel().
  * @param  htim TIM One Pulse handle
  * @param  OutputChannel See note above
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OnePulse_Stop_IT(TIM_HandleTypeDef *htim, uint32_t OutputChannel)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(OutputChannel);

  /* Disable the TIM Capture/Compare 1 interrupt */
  __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC1);

  /* Disable the TIM Capture/Compare 2 interrupt */
  __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC2);

  /* Disable the Capture compare and the Input Capture channels
  (in the OPM Mode the two possible channels that can be used are TIM_CHANNEL_1 and TIM_CHANNEL_2)
  if TIM_CHANNEL_1 is used as output, the TIM_CHANNEL_2 will be used as input and
  if TIM_CHANNEL_1 is used as input, the TIM_CHANNEL_2 will be used as output
  whatever the combination, the TIM_CHANNEL_1 and TIM_CHANNEL_2 should be disabled together */
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);
  TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_DISABLE);

  if (IS_TIM_BREAK_INSTANCE(htim->Instance) != RESET)
  {
    /* Disable the Main Output */
    __HAL_TIM_MOE_DISABLE(htim);
  }

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM channels state */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);

  /* Return function status */
  return HAL_OK;
}

/**
  * @}
  */

/** @defgroup TIM_Exported_Functions_Group6 TIM Encoder functions
  *  @brief    TIM Encoder functions
  *
@verbatim
  ==============================================================================
                          ##### TIM Encoder functions #####
  ==============================================================================
  [..]
    This section provides functions allowing to:
    (+) Initialize and configure the TIM Encoder.
    (+) De-initialize the TIM Encoder.
    (+) Start the TIM Encoder.
    (+) Stop the TIM Encoder.
    (+) Start the TIM Encoder and enable interrupt.
    (+) Stop the TIM Encoder and disable interrupt.
    (+) Start the TIM Encoder and enable DMA transfer.
    (+) Stop the TIM Encoder and disable DMA transfer.

@endverbatim
  * @{
  */
/**
  * @brief  Initializes the TIM Encoder Interface and initialize the associated handle.
  * @note   Switching from Center Aligned counter mode to Edge counter mode (or reverse)
  *         requires a timer reset to avoid unexpected direction
  *         due to DIR bit readonly in center aligned mode.
  *         Ex: call @ref HAL_TIM_Encoder_DeInit() before HAL_TIM_Encoder_Init()
  * @note   Encoder mode and External clock mode 2 are not compatible and must not be selected together
  *         Ex: A call for @ref HAL_TIM_Encoder_Init will erase the settings of @ref HAL_TIM_ConfigClockSource
  *         using TIM_CLOCKSOURCE_ETRMODE2 and vice versa
  * @note   When the timer instance is initialized in Encoder mode, timer
  *         channels 1 and channel 2 are reserved and cannot be used for other
  *         purpose.
  * @param  htim TIM Encoder Interface handle
  * @param  sConfig TIM Encoder Interface configuration structure
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Encoder_Init(TIM_HandleTypeDef *htim,  TIM_Encoder_InitTypeDef *sConfig)
{
  uint32_t tmpsmcr;
  uint32_t tmpccmr1;
  uint32_t tmpccer;

  /* Check the TIM handle allocation */
  if (htim == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_TIM_ENCODER_INTERFACE_INSTANCE(htim->Instance));
  assert_param(IS_TIM_COUNTER_MODE(htim->Init.CounterMode));
  assert_param(IS_TIM_CLOCKDIVISION_DIV(htim->Init.ClockDivision));
  assert_param(IS_TIM_AUTORELOAD_PRELOAD(htim->Init.AutoReloadPreload));
  assert_param(IS_TIM_ENCODER_MODE(sConfig->EncoderMode));
  assert_param(IS_TIM_IC_SELECTION(sConfig->IC1Selection));
  assert_param(IS_TIM_IC_SELECTION(sConfig->IC2Selection));
  assert_param(IS_TIM_ENCODERINPUT_POLARITY(sConfig->IC1Polarity));
  assert_param(IS_TIM_ENCODERINPUT_POLARITY(sConfig->IC2Polarity));
  assert_param(IS_TIM_IC_PRESCALER(sConfig->IC1Prescaler));
  assert_param(IS_TIM_IC_PRESCALER(sConfig->IC2Prescaler));
  assert_param(IS_TIM_IC_FILTER(sConfig->IC1Filter));
  assert_param(IS_TIM_IC_FILTER(sConfig->IC2Filter));

  if (htim->State == HAL_TIM_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    htim->Lock = HAL_UNLOCKED;

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
    /* Reset interrupt callbacks to legacy weak callbacks */
    TIM_ResetCallback(htim);

    if (htim->Encoder_MspInitCallback == NULL)
    {
      htim->Encoder_MspInitCallback = HAL_TIM_Encoder_MspInit;
    }
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    htim->Encoder_MspInitCallback(htim);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC and DMA */
    HAL_TIM_Encoder_MspInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
  }

  /* Set the TIM state */
  htim->State = HAL_TIM_STATE_BUSY;

  /* Reset the SMS and ECE bits */
  htim->Instance->SMCR &= ~(TIM_SMCR_SMS | TIM_SMCR_ECE);

  /* Configure the Time base in the Encoder Mode */
  TIM_Base_SetConfig(htim->Instance, &htim->Init);

  /* Get the TIMx SMCR register value */
  tmpsmcr = htim->Instance->SMCR;

  /* Get the TIMx CCMR1 register value */
  tmpccmr1 = htim->Instance->CCMR1;

  /* Get the TIMx CCER register value */
  tmpccer = htim->Instance->CCER;

  /* Set the encoder Mode */
  tmpsmcr |= sConfig->EncoderMode;

  /* Select the Capture Compare 1 and the Capture Compare 2 as input */
  tmpccmr1 &= ~(TIM_CCMR1_CC1S | TIM_CCMR1_CC2S);
  tmpccmr1 |= (sConfig->IC1Selection | (sConfig->IC2Selection << 8U));

  /* Set the Capture Compare 1 and the Capture Compare 2 prescalers and filters */
  tmpccmr1 &= ~(TIM_CCMR1_IC1PSC | TIM_CCMR1_IC2PSC);
  tmpccmr1 &= ~(TIM_CCMR1_IC1F | TIM_CCMR1_IC2F);
  tmpccmr1 |= sConfig->IC1Prescaler | (sConfig->IC2Prescaler << 8U);
  tmpccmr1 |= (sConfig->IC1Filter << 4U) | (sConfig->IC2Filter << 12U);

  /* Set the TI1 and the TI2 Polarities */
  tmpccer &= ~(TIM_CCER_CC1P | TIM_CCER_CC2P);
  tmpccer &= ~(TIM_CCER_CC1NP | TIM_CCER_CC2NP);
  tmpccer |= sConfig->IC1Polarity | (sConfig->IC2Polarity << 4U);

  /* Write to TIMx SMCR */
  htim->Instance->SMCR = tmpsmcr;

  /* Write to TIMx CCMR1 */
  htim->Instance->CCMR1 = tmpccmr1;

  /* Write to TIMx CCER */
  htim->Instance->CCER = tmpccer;

  /* Initialize the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_READY;

  /* Set the TIM channels state */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);

  /* Initialize the TIM state*/
  htim->State = HAL_TIM_STATE_READY;

  return HAL_OK;
}


/**
  * @brief  DeInitializes the TIM Encoder interface
  * @param  htim TIM Encoder Interface handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Encoder_DeInit(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));

  htim->State = HAL_TIM_STATE_BUSY;

  /* Disable the TIM Peripheral Clock */
  __HAL_TIM_DISABLE(htim);

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  if (htim->Encoder_MspDeInitCallback == NULL)
  {
    htim->Encoder_MspDeInitCallback = HAL_TIM_Encoder_MspDeInit;
  }
  /* DeInit the low level hardware */
  htim->Encoder_MspDeInitCallback(htim);
#else
  /* DeInit the low level hardware: GPIO, CLOCK, NVIC */
  HAL_TIM_Encoder_MspDeInit(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  /* Change the DMA burst operation state */
  htim->DMABurstState = HAL_DMA_BURST_STATE_RESET;

  /* Set the TIM channels state */
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_RESET);
  TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_RESET);

  /* Change TIM state */
  htim->State = HAL_TIM_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(htim);

  return HAL_OK;
}

/**
  * @brief  Initializes the TIM Encoder Interface MSP.
  * @param  htim TIM Encoder Interface handle
  * @retval None
  */
__weak void HAL_TIM_Encoder_MspInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_Encoder_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitializes TIM Encoder Interface MSP.
  * @param  htim TIM Encoder Interface handle
  * @retval None
  */
__weak void HAL_TIM_Encoder_MspDeInit(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_Encoder_MspDeInit could be implemented in the user file
   */
}

/**
  * @brief  Starts the TIM Encoder Interface.
  * @param  htim TIM Encoder Interface handle
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_ALL: TIM Channel 1 and TIM Channel 2 are selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Encoder_Start(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_TIM_ChannelStateTypeDef channel_1_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef channel_2_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_2);
  HAL_TIM_ChannelStateTypeDef complementary_channel_1_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef complementary_channel_2_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_2);

  /* Check the parameters */
  assert_param(IS_TIM_ENCODER_INTERFACE_INSTANCE(htim->Instance));

  /* Set the TIM channel(s) state */
  if (Channel == TIM_CHANNEL_1)
  {
    if ((channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
        || (complementary_channel_1_state != HAL_TIM_CHANNEL_STATE_READY))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }
  else if (Channel == TIM_CHANNEL_2)
  {
    if ((channel_2_state != HAL_TIM_CHANNEL_STATE_READY)
        || (complementary_channel_2_state != HAL_TIM_CHANNEL_STATE_READY))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }
  else
  {
    if ((channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
        || (channel_2_state != HAL_TIM_CHANNEL_STATE_READY)
        || (complementary_channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
        || (complementary_channel_2_state != HAL_TIM_CHANNEL_STATE_READY))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }

  /* Enable the encoder interface channels */
  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);
      break;
    }

    case TIM_CHANNEL_2:
    {
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_ENABLE);
      break;
    }

    default :
    {
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_ENABLE);
      break;
    }
  }
  /* Enable the Peripheral */
  __HAL_TIM_ENABLE(htim);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM Encoder Interface.
  * @param  htim TIM Encoder Interface handle
  * @param  Channel TIM Channels to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_ALL: TIM Channel 1 and TIM Channel 2 are selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Encoder_Stop(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  /* Check the parameters */
  assert_param(IS_TIM_ENCODER_INTERFACE_INSTANCE(htim->Instance));

  /* Disable the Input Capture channels 1 and 2
    (in the EncoderInterface the two possible channels that can be used are TIM_CHANNEL_1 and TIM_CHANNEL_2) */
  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);
      break;
    }

    case TIM_CHANNEL_2:
    {
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_DISABLE);
      break;
    }

    default :
    {
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_DISABLE);
      break;
    }
  }

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM channel(s) state */
  if ((Channel == TIM_CHANNEL_1) || (Channel == TIM_CHANNEL_2))
  {
    TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  }
  else
  {
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Starts the TIM Encoder Interface in interrupt mode.
  * @param  htim TIM Encoder Interface handle
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_ALL: TIM Channel 1 and TIM Channel 2 are selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Encoder_Start_IT(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  HAL_TIM_ChannelStateTypeDef channel_1_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef channel_2_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_2);
  HAL_TIM_ChannelStateTypeDef complementary_channel_1_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef complementary_channel_2_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_2);

  /* Check the parameters */
  assert_param(IS_TIM_ENCODER_INTERFACE_INSTANCE(htim->Instance));

  /* Set the TIM channel(s) state */
  if (Channel == TIM_CHANNEL_1)
  {
    if ((channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
        || (complementary_channel_1_state != HAL_TIM_CHANNEL_STATE_READY))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }
  else if (Channel == TIM_CHANNEL_2)
  {
    if ((channel_2_state != HAL_TIM_CHANNEL_STATE_READY)
        || (complementary_channel_2_state != HAL_TIM_CHANNEL_STATE_READY))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }
  else
  {
    if ((channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
        || (channel_2_state != HAL_TIM_CHANNEL_STATE_READY)
        || (complementary_channel_1_state != HAL_TIM_CHANNEL_STATE_READY)
        || (complementary_channel_2_state != HAL_TIM_CHANNEL_STATE_READY))
    {
      return HAL_ERROR;
    }
    else
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
    }
  }

  /* Enable the encoder interface channels */
  /* Enable the capture compare Interrupts 1 and/or 2 */
  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC1);
      break;
    }

    case TIM_CHANNEL_2:
    {
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_ENABLE);
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC2);
      break;
    }

    default :
    {
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_ENABLE);
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC1);
      __HAL_TIM_ENABLE_IT(htim, TIM_IT_CC2);
      break;
    }
  }

  /* Enable the Peripheral */
  __HAL_TIM_ENABLE(htim);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM Encoder Interface in interrupt mode.
  * @param  htim TIM Encoder Interface handle
  * @param  Channel TIM Channels to be disabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_ALL: TIM Channel 1 and TIM Channel 2 are selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Encoder_Stop_IT(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  /* Check the parameters */
  assert_param(IS_TIM_ENCODER_INTERFACE_INSTANCE(htim->Instance));

  /* Disable the Input Capture channels 1 and 2
    (in the EncoderInterface the two possible channels that can be used are TIM_CHANNEL_1 and TIM_CHANNEL_2) */
  if (Channel == TIM_CHANNEL_1)
  {
    TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);

    /* Disable the capture compare Interrupts 1 */
    __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC1);
  }
  else if (Channel == TIM_CHANNEL_2)
  {
    TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_DISABLE);

    /* Disable the capture compare Interrupts 2 */
    __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC2);
  }
  else
  {
    TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);
    TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_DISABLE);

    /* Disable the capture compare Interrupts 1 and 2 */
    __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC1);
    __HAL_TIM_DISABLE_IT(htim, TIM_IT_CC2);
  }

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM channel(s) state */
  if ((Channel == TIM_CHANNEL_1) || (Channel == TIM_CHANNEL_2))
  {
    TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  }
  else
  {
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Starts the TIM Encoder Interface in DMA mode.
  * @param  htim TIM Encoder Interface handle
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_ALL: TIM Channel 1 and TIM Channel 2 are selected
  * @param  pData1 The destination Buffer address for IC1.
  * @param  pData2 The destination Buffer address for IC2.
  * @param  Length The length of data to be transferred from TIM peripheral to memory.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Encoder_Start_DMA(TIM_HandleTypeDef *htim, uint32_t Channel, uint32_t *pData1,
                                            uint32_t *pData2, uint16_t Length)
{
  HAL_TIM_ChannelStateTypeDef channel_1_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef channel_2_state = TIM_CHANNEL_STATE_GET(htim, TIM_CHANNEL_2);
  HAL_TIM_ChannelStateTypeDef complementary_channel_1_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_1);
  HAL_TIM_ChannelStateTypeDef complementary_channel_2_state = TIM_CHANNEL_N_STATE_GET(htim, TIM_CHANNEL_2);

  /* Check the parameters */
  assert_param(IS_TIM_ENCODER_INTERFACE_INSTANCE(htim->Instance));

  /* Set the TIM channel(s) state */
  if (Channel == TIM_CHANNEL_1)
  {
    if ((channel_1_state == HAL_TIM_CHANNEL_STATE_BUSY)
        || (complementary_channel_1_state == HAL_TIM_CHANNEL_STATE_BUSY))
    {
      return HAL_BUSY;
    }
    else if ((channel_1_state == HAL_TIM_CHANNEL_STATE_READY)
             && (complementary_channel_1_state == HAL_TIM_CHANNEL_STATE_READY))
    {
      if ((pData1 == NULL) && (Length > 0U))
      {
        return HAL_ERROR;
      }
      else
      {
        TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
        TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
      }
    }
    else
    {
      return HAL_ERROR;
    }
  }
  else if (Channel == TIM_CHANNEL_2)
  {
    if ((channel_2_state == HAL_TIM_CHANNEL_STATE_BUSY)
        || (complementary_channel_2_state == HAL_TIM_CHANNEL_STATE_BUSY))
    {
      return HAL_BUSY;
    }
    else if ((channel_2_state == HAL_TIM_CHANNEL_STATE_READY)
             && (complementary_channel_2_state == HAL_TIM_CHANNEL_STATE_READY))
    {
      if ((pData2 == NULL) && (Length > 0U))
      {
        return HAL_ERROR;
      }
      else
      {
        TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
        TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
      }
    }
    else
    {
      return HAL_ERROR;
    }
  }
  else
  {
    if ((channel_1_state == HAL_TIM_CHANNEL_STATE_BUSY)
        || (channel_2_state == HAL_TIM_CHANNEL_STATE_BUSY)
        || (complementary_channel_1_state == HAL_TIM_CHANNEL_STATE_BUSY)
        || (complementary_channel_2_state == HAL_TIM_CHANNEL_STATE_BUSY))
    {
      return HAL_BUSY;
    }
    else if ((channel_1_state == HAL_TIM_CHANNEL_STATE_READY)
             && (channel_2_state == HAL_TIM_CHANNEL_STATE_READY)
             && (complementary_channel_1_state == HAL_TIM_CHANNEL_STATE_READY)
             && (complementary_channel_2_state == HAL_TIM_CHANNEL_STATE_READY))
    {
      if ((((pData1 == NULL) || (pData2 == NULL))) && (Length > 0U))
      {
        return HAL_ERROR;
      }
      else
      {
        TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
        TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
        TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_BUSY);
        TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_BUSY);
      }
    }
    else
    {
      return HAL_ERROR;
    }
  }

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC1]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC1]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC1]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC1], (uint32_t)&htim->Instance->CCR1, (uint32_t)pData1,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Input Capture DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC1);

      /* Enable the Capture compare channel */
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);

      /* Enable the Peripheral */
      __HAL_TIM_ENABLE(htim);

      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC2]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC2]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC2]->XferErrorCallback = TIM_DMAError;
      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC2], (uint32_t)&htim->Instance->CCR2, (uint32_t)pData2,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      /* Enable the TIM Input Capture  DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC2);

      /* Enable the Capture compare channel */
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_ENABLE);

      /* Enable the Peripheral */
      __HAL_TIM_ENABLE(htim);

      break;
    }

    default:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC1]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC1]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC1]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC1], (uint32_t)&htim->Instance->CCR1, (uint32_t)pData1,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }

      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC2]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC2]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC2]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC2], (uint32_t)&htim->Instance->CCR2, (uint32_t)pData2,
                           Length) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }

      /* Enable the TIM Input Capture  DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC1);
      /* Enable the TIM Input Capture  DMA request */
      __HAL_TIM_ENABLE_DMA(htim, TIM_DMA_CC2);

      /* Enable the Capture compare channel */
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_ENABLE);
      TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_ENABLE);

      /* Enable the Peripheral */
      __HAL_TIM_ENABLE(htim);

      break;
    }
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Stops the TIM Encoder Interface in DMA mode.
  * @param  htim TIM Encoder Interface handle
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_ALL: TIM Channel 1 and TIM Channel 2 are selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_Encoder_Stop_DMA(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  /* Check the parameters */
  assert_param(IS_TIM_ENCODER_INTERFACE_INSTANCE(htim->Instance));

  /* Disable the Input Capture channels 1 and 2
    (in the EncoderInterface the two possible channels that can be used are TIM_CHANNEL_1 and TIM_CHANNEL_2) */
  if (Channel == TIM_CHANNEL_1)
  {
    TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);

    /* Disable the capture compare DMA Request 1 */
    __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC1);
    (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC1]);
  }
  else if (Channel == TIM_CHANNEL_2)
  {
    TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_DISABLE);

    /* Disable the capture compare DMA Request 2 */
    __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC2);
    (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC2]);
  }
  else
  {
    TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_1, TIM_CCx_DISABLE);
    TIM_CCxChannelCmd(htim->Instance, TIM_CHANNEL_2, TIM_CCx_DISABLE);

    /* Disable the capture compare DMA Request 1 and 2 */
    __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC1);
    __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_CC2);
    (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC1]);
    (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC2]);
  }

  /* Disable the Peripheral */
  __HAL_TIM_DISABLE(htim);

  /* Set the TIM channel(s) state */
  if ((Channel == TIM_CHANNEL_1) || (Channel == TIM_CHANNEL_2))
  {
    TIM_CHANNEL_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, Channel, HAL_TIM_CHANNEL_STATE_READY);
  }
  else
  {
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
    TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @}
  */
/** @defgroup TIM_Exported_Functions_Group7 TIM IRQ handler management
  *  @brief    TIM IRQ handler management
  *
@verbatim
  ==============================================================================
                        ##### IRQ handler management #####
  ==============================================================================
  [..]
    This section provides Timer IRQ handler function.

@endverbatim
  * @{
  */
/**
  * @brief  This function handles TIM interrupts requests.
  * @param  htim TIM  handle
  * @retval None
  */
void HAL_TIM_IRQHandler(TIM_HandleTypeDef *htim)
{
  /* Capture compare 1 event */
  if (__HAL_TIM_GET_FLAG(htim, TIM_FLAG_CC1) != RESET)
  {
    if (__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_CC1) != RESET)
    {
      {
        __HAL_TIM_CLEAR_IT(htim, TIM_IT_CC1);
        htim->Channel = HAL_TIM_ACTIVE_CHANNEL_1;

        /* Input capture event */
        if ((htim->Instance->CCMR1 & TIM_CCMR1_CC1S) != 0x00U)
        {
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
          htim->IC_CaptureCallback(htim);
#else
          HAL_TIM_IC_CaptureCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
        }
        /* Output compare event */
        else
        {
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
          htim->OC_DelayElapsedCallback(htim);
          htim->PWM_PulseFinishedCallback(htim);
#else
          HAL_TIM_OC_DelayElapsedCallback(htim);
          HAL_TIM_PWM_PulseFinishedCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
        }
        htim->Channel = HAL_TIM_ACTIVE_CHANNEL_CLEARED;
      }
    }
  }
  /* Capture compare 2 event */
  if (__HAL_TIM_GET_FLAG(htim, TIM_FLAG_CC2) != RESET)
  {
    if (__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_CC2) != RESET)
    {
      __HAL_TIM_CLEAR_IT(htim, TIM_IT_CC2);
      htim->Channel = HAL_TIM_ACTIVE_CHANNEL_2;
      /* Input capture event */
      if ((htim->Instance->CCMR1 & TIM_CCMR1_CC2S) != 0x00U)
      {
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
        htim->IC_CaptureCallback(htim);
#else
        HAL_TIM_IC_CaptureCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
      }
      /* Output compare event */
      else
      {
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
        htim->OC_DelayElapsedCallback(htim);
        htim->PWM_PulseFinishedCallback(htim);
#else
        HAL_TIM_OC_DelayElapsedCallback(htim);
        HAL_TIM_PWM_PulseFinishedCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
      }
      htim->Channel = HAL_TIM_ACTIVE_CHANNEL_CLEARED;
    }
  }
  /* Capture compare 3 event */
  if (__HAL_TIM_GET_FLAG(htim, TIM_FLAG_CC3) != RESET)
  {
    if (__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_CC3) != RESET)
    {
      __HAL_TIM_CLEAR_IT(htim, TIM_IT_CC3);
      htim->Channel = HAL_TIM_ACTIVE_CHANNEL_3;
      /* Input capture event */
      if ((htim->Instance->CCMR2 & TIM_CCMR2_CC3S) != 0x00U)
      {
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
        htim->IC_CaptureCallback(htim);
#else
        HAL_TIM_IC_CaptureCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
      }
      /* Output compare event */
      else
      {
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
        htim->OC_DelayElapsedCallback(htim);
        htim->PWM_PulseFinishedCallback(htim);
#else
        HAL_TIM_OC_DelayElapsedCallback(htim);
        HAL_TIM_PWM_PulseFinishedCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
      }
      htim->Channel = HAL_TIM_ACTIVE_CHANNEL_CLEARED;
    }
  }
  /* Capture compare 4 event */
  if (__HAL_TIM_GET_FLAG(htim, TIM_FLAG_CC4) != RESET)
  {
    if (__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_CC4) != RESET)
    {
      __HAL_TIM_CLEAR_IT(htim, TIM_IT_CC4);
      htim->Channel = HAL_TIM_ACTIVE_CHANNEL_4;
      /* Input capture event */
      if ((htim->Instance->CCMR2 & TIM_CCMR2_CC4S) != 0x00U)
      {
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
        htim->IC_CaptureCallback(htim);
#else
        HAL_TIM_IC_CaptureCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
      }
      /* Output compare event */
      else
      {
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
        htim->OC_DelayElapsedCallback(htim);
        htim->PWM_PulseFinishedCallback(htim);
#else
        HAL_TIM_OC_DelayElapsedCallback(htim);
        HAL_TIM_PWM_PulseFinishedCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
      }
      htim->Channel = HAL_TIM_ACTIVE_CHANNEL_CLEARED;
    }
  }
  /* TIM Update event */
  if (__HAL_TIM_GET_FLAG(htim, TIM_FLAG_UPDATE) != RESET)
  {
    if (__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_UPDATE) != RESET)
    {
      __HAL_TIM_CLEAR_IT(htim, TIM_IT_UPDATE);
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
      htim->PeriodElapsedCallback(htim);
#else
      HAL_TIM_PeriodElapsedCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
    }
  }
  /* TIM Break input event */
  if (__HAL_TIM_GET_FLAG(htim, TIM_FLAG_BREAK) != RESET)
  {
    if (__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_BREAK) != RESET)
    {
      __HAL_TIM_CLEAR_IT(htim, TIM_IT_BREAK);
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
      htim->BreakCallback(htim);
#else
      HAL_TIMEx_BreakCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
    }
  }
  /* TIM Trigger detection event */
  if (__HAL_TIM_GET_FLAG(htim, TIM_FLAG_TRIGGER) != RESET)
  {
    if (__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_TRIGGER) != RESET)
    {
      __HAL_TIM_CLEAR_IT(htim, TIM_IT_TRIGGER);
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
      htim->TriggerCallback(htim);
#else
      HAL_TIM_TriggerCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
    }
  }
  /* TIM commutation event */
  if (__HAL_TIM_GET_FLAG(htim, TIM_FLAG_COM) != RESET)
  {
    if (__HAL_TIM_GET_IT_SOURCE(htim, TIM_IT_COM) != RESET)
    {
      __HAL_TIM_CLEAR_IT(htim, TIM_FLAG_COM);
#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
      htim->CommutationCallback(htim);
#else
      HAL_TIMEx_CommutCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
    }
  }
}

/**
  * @}
  */

/** @defgroup TIM_Exported_Functions_Group8 TIM Peripheral Control functions
  *  @brief    TIM Peripheral Control functions
  *
@verbatim
  ==============================================================================
                   ##### Peripheral Control functions #####
  ==============================================================================
 [..]
   This section provides functions allowing to:
      (+) Configure The Input Output channels for OC, PWM, IC or One Pulse mode.
      (+) Configure External Clock source.
      (+) Configure Complementary channels, break features and dead time.
      (+) Configure Master and the Slave synchronization.
      (+) Configure the DMA Burst Mode.

@endverbatim
  * @{
  */

/**
  * @brief  Initializes the TIM Output Compare Channels according to the specified
  *         parameters in the TIM_OC_InitTypeDef.
  * @param  htim TIM Output Compare handle
  * @param  sConfig TIM Output Compare configuration structure
  * @param  Channel TIM Channels to configure
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OC_ConfigChannel(TIM_HandleTypeDef *htim,
                                           TIM_OC_InitTypeDef *sConfig,
                                           uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_CHANNELS(Channel));
  assert_param(IS_TIM_OC_MODE(sConfig->OCMode));
  assert_param(IS_TIM_OC_POLARITY(sConfig->OCPolarity));

  /* Process Locked */
  __HAL_LOCK(htim);

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC1_INSTANCE(htim->Instance));

      /* Configure the TIM Channel 1 in Output Compare */
      TIM_OC1_SetConfig(htim->Instance, sConfig);
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC2_INSTANCE(htim->Instance));

      /* Configure the TIM Channel 2 in Output Compare */
      TIM_OC2_SetConfig(htim->Instance, sConfig);
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC3_INSTANCE(htim->Instance));

      /* Configure the TIM Channel 3 in Output Compare */
      TIM_OC3_SetConfig(htim->Instance, sConfig);
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC4_INSTANCE(htim->Instance));

      /* Configure the TIM Channel 4 in Output Compare */
      TIM_OC4_SetConfig(htim->Instance, sConfig);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  __HAL_UNLOCK(htim);

  return status;
}

/**
  * @brief  Initializes the TIM Input Capture Channels according to the specified
  *         parameters in the TIM_IC_InitTypeDef.
  * @param  htim TIM IC handle
  * @param  sConfig TIM Input Capture configuration structure
  * @param  Channel TIM Channel to configure
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_IC_ConfigChannel(TIM_HandleTypeDef *htim, TIM_IC_InitTypeDef *sConfig, uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_CC1_INSTANCE(htim->Instance));
  assert_param(IS_TIM_IC_POLARITY(sConfig->ICPolarity));
  assert_param(IS_TIM_IC_SELECTION(sConfig->ICSelection));
  assert_param(IS_TIM_IC_PRESCALER(sConfig->ICPrescaler));
  assert_param(IS_TIM_IC_FILTER(sConfig->ICFilter));

  /* Process Locked */
  __HAL_LOCK(htim);

  if (Channel == TIM_CHANNEL_1)
  {
    /* TI1 Configuration */
    TIM_TI1_SetConfig(htim->Instance,
                      sConfig->ICPolarity,
                      sConfig->ICSelection,
                      sConfig->ICFilter);

    /* Reset the IC1PSC Bits */
    htim->Instance->CCMR1 &= ~TIM_CCMR1_IC1PSC;

    /* Set the IC1PSC value */
    htim->Instance->CCMR1 |= sConfig->ICPrescaler;
  }
  else if (Channel == TIM_CHANNEL_2)
  {
    /* TI2 Configuration */
    assert_param(IS_TIM_CC2_INSTANCE(htim->Instance));

    TIM_TI2_SetConfig(htim->Instance,
                      sConfig->ICPolarity,
                      sConfig->ICSelection,
                      sConfig->ICFilter);

    /* Reset the IC2PSC Bits */
    htim->Instance->CCMR1 &= ~TIM_CCMR1_IC2PSC;

    /* Set the IC2PSC value */
    htim->Instance->CCMR1 |= (sConfig->ICPrescaler << 8U);
  }
  else if (Channel == TIM_CHANNEL_3)
  {
    /* TI3 Configuration */
    assert_param(IS_TIM_CC3_INSTANCE(htim->Instance));

    TIM_TI3_SetConfig(htim->Instance,
                      sConfig->ICPolarity,
                      sConfig->ICSelection,
                      sConfig->ICFilter);

    /* Reset the IC3PSC Bits */
    htim->Instance->CCMR2 &= ~TIM_CCMR2_IC3PSC;

    /* Set the IC3PSC value */
    htim->Instance->CCMR2 |= sConfig->ICPrescaler;
  }
  else if (Channel == TIM_CHANNEL_4)
  {
    /* TI4 Configuration */
    assert_param(IS_TIM_CC4_INSTANCE(htim->Instance));

    TIM_TI4_SetConfig(htim->Instance,
                      sConfig->ICPolarity,
                      sConfig->ICSelection,
                      sConfig->ICFilter);

    /* Reset the IC4PSC Bits */
    htim->Instance->CCMR2 &= ~TIM_CCMR2_IC4PSC;

    /* Set the IC4PSC value */
    htim->Instance->CCMR2 |= (sConfig->ICPrescaler << 8U);
  }
  else
  {
    status = HAL_ERROR;
  }

  __HAL_UNLOCK(htim);

  return status;
}

/**
  * @brief  Initializes the TIM PWM  channels according to the specified
  *         parameters in the TIM_OC_InitTypeDef.
  * @param  htim TIM PWM handle
  * @param  sConfig TIM PWM configuration structure
  * @param  Channel TIM Channels to be configured
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_PWM_ConfigChannel(TIM_HandleTypeDef *htim,
                                            TIM_OC_InitTypeDef *sConfig,
                                            uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_CHANNELS(Channel));
  assert_param(IS_TIM_PWM_MODE(sConfig->OCMode));
  assert_param(IS_TIM_OC_POLARITY(sConfig->OCPolarity));
  assert_param(IS_TIM_FAST_STATE(sConfig->OCFastMode));

  /* Process Locked */
  __HAL_LOCK(htim);

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC1_INSTANCE(htim->Instance));

      /* Configure the Channel 1 in PWM mode */
      TIM_OC1_SetConfig(htim->Instance, sConfig);

      /* Set the Preload enable bit for channel1 */
      htim->Instance->CCMR1 |= TIM_CCMR1_OC1PE;

      /* Configure the Output Fast mode */
      htim->Instance->CCMR1 &= ~TIM_CCMR1_OC1FE;
      htim->Instance->CCMR1 |= sConfig->OCFastMode;
      break;
    }

    case TIM_CHANNEL_2:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC2_INSTANCE(htim->Instance));

      /* Configure the Channel 2 in PWM mode */
      TIM_OC2_SetConfig(htim->Instance, sConfig);

      /* Set the Preload enable bit for channel2 */
      htim->Instance->CCMR1 |= TIM_CCMR1_OC2PE;

      /* Configure the Output Fast mode */
      htim->Instance->CCMR1 &= ~TIM_CCMR1_OC2FE;
      htim->Instance->CCMR1 |= sConfig->OCFastMode << 8U;
      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC3_INSTANCE(htim->Instance));

      /* Configure the Channel 3 in PWM mode */
      TIM_OC3_SetConfig(htim->Instance, sConfig);

      /* Set the Preload enable bit for channel3 */
      htim->Instance->CCMR2 |= TIM_CCMR2_OC3PE;

      /* Configure the Output Fast mode */
      htim->Instance->CCMR2 &= ~TIM_CCMR2_OC3FE;
      htim->Instance->CCMR2 |= sConfig->OCFastMode;
      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC4_INSTANCE(htim->Instance));

      /* Configure the Channel 4 in PWM mode */
      TIM_OC4_SetConfig(htim->Instance, sConfig);

      /* Set the Preload enable bit for channel4 */
      htim->Instance->CCMR2 |= TIM_CCMR2_OC4PE;

      /* Configure the Output Fast mode */
      htim->Instance->CCMR2 &= ~TIM_CCMR2_OC4FE;
      htim->Instance->CCMR2 |= sConfig->OCFastMode << 8U;
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  __HAL_UNLOCK(htim);

  return status;
}

/**
  * @brief  Initializes the TIM One Pulse Channels according to the specified
  *         parameters in the TIM_OnePulse_InitTypeDef.
  * @param  htim TIM One Pulse handle
  * @param  sConfig TIM One Pulse configuration structure
  * @param  OutputChannel TIM output channel to configure
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  * @param  InputChannel TIM input Channel to configure
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  * @note  To output a waveform with a minimum delay user can enable the fast
  *        mode by calling the @ref __HAL_TIM_ENABLE_OCxFAST macro. Then CCx
  *        output is forced in response to the edge detection on TIx input,
  *        without taking in account the comparison.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_OnePulse_ConfigChannel(TIM_HandleTypeDef *htim,  TIM_OnePulse_InitTypeDef *sConfig,
                                                 uint32_t OutputChannel,  uint32_t InputChannel)
{
  HAL_StatusTypeDef status = HAL_OK;
  TIM_OC_InitTypeDef temp1;

  /* Check the parameters */
  assert_param(IS_TIM_OPM_CHANNELS(OutputChannel));
  assert_param(IS_TIM_OPM_CHANNELS(InputChannel));

  if (OutputChannel != InputChannel)
  {
    /* Process Locked */
    __HAL_LOCK(htim);

    htim->State = HAL_TIM_STATE_BUSY;

    /* Extract the Output compare configuration from sConfig structure */
    temp1.OCMode = sConfig->OCMode;
    temp1.Pulse = sConfig->Pulse;
    temp1.OCPolarity = sConfig->OCPolarity;
    temp1.OCNPolarity = sConfig->OCNPolarity;
    temp1.OCIdleState = sConfig->OCIdleState;
    temp1.OCNIdleState = sConfig->OCNIdleState;

    switch (OutputChannel)
    {
      case TIM_CHANNEL_1:
      {
        assert_param(IS_TIM_CC1_INSTANCE(htim->Instance));

        TIM_OC1_SetConfig(htim->Instance, &temp1);
        break;
      }

      case TIM_CHANNEL_2:
      {
        assert_param(IS_TIM_CC2_INSTANCE(htim->Instance));

        TIM_OC2_SetConfig(htim->Instance, &temp1);
        break;
      }

      default:
        status = HAL_ERROR;
        break;
    }

    if (status == HAL_OK)
    {
      switch (InputChannel)
      {
        case TIM_CHANNEL_1:
        {
          assert_param(IS_TIM_CC1_INSTANCE(htim->Instance));

          TIM_TI1_SetConfig(htim->Instance, sConfig->ICPolarity,
                            sConfig->ICSelection, sConfig->ICFilter);

          /* Reset the IC1PSC Bits */
          htim->Instance->CCMR1 &= ~TIM_CCMR1_IC1PSC;

          /* Select the Trigger source */
          htim->Instance->SMCR &= ~TIM_SMCR_TS;
          htim->Instance->SMCR |= TIM_TS_TI1FP1;

          /* Select the Slave Mode */
          htim->Instance->SMCR &= ~TIM_SMCR_SMS;
          htim->Instance->SMCR |= TIM_SLAVEMODE_TRIGGER;
          break;
        }

        case TIM_CHANNEL_2:
        {
          assert_param(IS_TIM_CC2_INSTANCE(htim->Instance));

          TIM_TI2_SetConfig(htim->Instance, sConfig->ICPolarity,
                            sConfig->ICSelection, sConfig->ICFilter);

          /* Reset the IC2PSC Bits */
          htim->Instance->CCMR1 &= ~TIM_CCMR1_IC2PSC;

          /* Select the Trigger source */
          htim->Instance->SMCR &= ~TIM_SMCR_TS;
          htim->Instance->SMCR |= TIM_TS_TI2FP2;

          /* Select the Slave Mode */
          htim->Instance->SMCR &= ~TIM_SMCR_SMS;
          htim->Instance->SMCR |= TIM_SLAVEMODE_TRIGGER;
          break;
        }

        default:
          status = HAL_ERROR;
          break;
      }
    }

    htim->State = HAL_TIM_STATE_READY;

    __HAL_UNLOCK(htim);

    return status;
  }
  else
  {
    return HAL_ERROR;
  }
}

/**
  * @brief  Configure the DMA Burst to transfer Data from the memory to the TIM peripheral
  * @param  htim TIM handle
  * @param  BurstBaseAddress TIM Base address from where the DMA  will start the Data write
  *         This parameter can be one of the following values:
  *            @arg TIM_DMABASE_CR1
  *            @arg TIM_DMABASE_CR2
  *            @arg TIM_DMABASE_SMCR
  *            @arg TIM_DMABASE_DIER
  *            @arg TIM_DMABASE_SR
  *            @arg TIM_DMABASE_EGR
  *            @arg TIM_DMABASE_CCMR1
  *            @arg TIM_DMABASE_CCMR2
  *            @arg TIM_DMABASE_CCER
  *            @arg TIM_DMABASE_CNT
  *            @arg TIM_DMABASE_PSC
  *            @arg TIM_DMABASE_ARR
  *            @arg TIM_DMABASE_RCR
  *            @arg TIM_DMABASE_CCR1
  *            @arg TIM_DMABASE_CCR2
  *            @arg TIM_DMABASE_CCR3
  *            @arg TIM_DMABASE_CCR4
  *            @arg TIM_DMABASE_BDTR
  * @param  BurstRequestSrc TIM DMA Request sources
  *         This parameter can be one of the following values:
  *            @arg TIM_DMA_UPDATE: TIM update Interrupt source
  *            @arg TIM_DMA_CC1: TIM Capture Compare 1 DMA source
  *            @arg TIM_DMA_CC2: TIM Capture Compare 2 DMA source
  *            @arg TIM_DMA_CC3: TIM Capture Compare 3 DMA source
  *            @arg TIM_DMA_CC4: TIM Capture Compare 4 DMA source
  *            @arg TIM_DMA_COM: TIM Commutation DMA source
  *            @arg TIM_DMA_TRIGGER: TIM Trigger DMA source
  * @param  BurstBuffer The Buffer address.
  * @param  BurstLength DMA Burst length. This parameter can be one value
  *         between: TIM_DMABURSTLENGTH_1TRANSFER and TIM_DMABURSTLENGTH_18TRANSFERS.
  * @note   This function should be used only when BurstLength is equal to DMA data transfer length.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_DMABurst_WriteStart(TIM_HandleTypeDef *htim, uint32_t BurstBaseAddress,
                                              uint32_t BurstRequestSrc, uint32_t *BurstBuffer, uint32_t  BurstLength)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (status == HAL_OK)
  {
    status = HAL_TIM_DMABurst_MultiWriteStart(htim, BurstBaseAddress, BurstRequestSrc, BurstBuffer, BurstLength,
                                              ((BurstLength) >> 8U) + 1U);
  }


  return status;
}

/**
  * @brief  Configure the DMA Burst to transfer multiple Data from the memory to the TIM peripheral
  * @param  htim TIM handle
  * @param  BurstBaseAddress TIM Base address from where the DMA will start the Data write
  *         This parameter can be one of the following values:
  *            @arg TIM_DMABASE_CR1
  *            @arg TIM_DMABASE_CR2
  *            @arg TIM_DMABASE_SMCR
  *            @arg TIM_DMABASE_DIER
  *            @arg TIM_DMABASE_SR
  *            @arg TIM_DMABASE_EGR
  *            @arg TIM_DMABASE_CCMR1
  *            @arg TIM_DMABASE_CCMR2
  *            @arg TIM_DMABASE_CCER
  *            @arg TIM_DMABASE_CNT
  *            @arg TIM_DMABASE_PSC
  *            @arg TIM_DMABASE_ARR
  *            @arg TIM_DMABASE_RCR
  *            @arg TIM_DMABASE_CCR1
  *            @arg TIM_DMABASE_CCR2
  *            @arg TIM_DMABASE_CCR3
  *            @arg TIM_DMABASE_CCR4
  *            @arg TIM_DMABASE_BDTR
  * @param  BurstRequestSrc TIM DMA Request sources
  *         This parameter can be one of the following values:
  *            @arg TIM_DMA_UPDATE: TIM update Interrupt source
  *            @arg TIM_DMA_CC1: TIM Capture Compare 1 DMA source
  *            @arg TIM_DMA_CC2: TIM Capture Compare 2 DMA source
  *            @arg TIM_DMA_CC3: TIM Capture Compare 3 DMA source
  *            @arg TIM_DMA_CC4: TIM Capture Compare 4 DMA source
  *            @arg TIM_DMA_COM: TIM Commutation DMA source
  *            @arg TIM_DMA_TRIGGER: TIM Trigger DMA source
  * @param  BurstBuffer The Buffer address.
  * @param  BurstLength DMA Burst length. This parameter can be one value
  *         between: TIM_DMABURSTLENGTH_1TRANSFER and TIM_DMABURSTLENGTH_18TRANSFERS.
  * @param  DataLength Data length. This parameter can be one value
  *         between 1 and 0xFFFF.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_DMABurst_MultiWriteStart(TIM_HandleTypeDef *htim, uint32_t BurstBaseAddress,
                                                   uint32_t BurstRequestSrc, uint32_t *BurstBuffer,
                                                   uint32_t  BurstLength,  uint32_t  DataLength)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_DMABURST_INSTANCE(htim->Instance));
  assert_param(IS_TIM_DMA_BASE(BurstBaseAddress));
  assert_param(IS_TIM_DMA_SOURCE(BurstRequestSrc));
  assert_param(IS_TIM_DMA_LENGTH(BurstLength));
  assert_param(IS_TIM_DMA_DATA_LENGTH(DataLength));

  if (htim->DMABurstState == HAL_DMA_BURST_STATE_BUSY)
  {
    return HAL_BUSY;
  }
  else if (htim->DMABurstState == HAL_DMA_BURST_STATE_READY)
  {
    if ((BurstBuffer == NULL) && (BurstLength > 0U))
    {
      return HAL_ERROR;
    }
    else
    {
      htim->DMABurstState = HAL_DMA_BURST_STATE_BUSY;
    }
  }
  else
  {
    /* nothing to do */
  }

  switch (BurstRequestSrc)
  {
    case TIM_DMA_UPDATE:
    {
      /* Set the DMA Period elapsed callbacks */
      htim->hdma[TIM_DMA_ID_UPDATE]->XferCpltCallback = TIM_DMAPeriodElapsedCplt;
      htim->hdma[TIM_DMA_ID_UPDATE]->XferHalfCpltCallback = TIM_DMAPeriodElapsedHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_UPDATE]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_UPDATE], (uint32_t)BurstBuffer,
                           (uint32_t)&htim->Instance->DMAR, DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_CC1:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC1]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC1]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC1]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC1], (uint32_t)BurstBuffer,
                           (uint32_t)&htim->Instance->DMAR, DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_CC2:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC2]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC2]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC2]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC2], (uint32_t)BurstBuffer,
                           (uint32_t)&htim->Instance->DMAR, DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_CC3:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC3]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC3]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC3]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC3], (uint32_t)BurstBuffer,
                           (uint32_t)&htim->Instance->DMAR, DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_CC4:
    {
      /* Set the DMA compare callbacks */
      htim->hdma[TIM_DMA_ID_CC4]->XferCpltCallback = TIM_DMADelayPulseCplt;
      htim->hdma[TIM_DMA_ID_CC4]->XferHalfCpltCallback = TIM_DMADelayPulseHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC4]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC4], (uint32_t)BurstBuffer,
                           (uint32_t)&htim->Instance->DMAR, DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_COM:
    {
      /* Set the DMA commutation callbacks */
      htim->hdma[TIM_DMA_ID_COMMUTATION]->XferCpltCallback =  TIMEx_DMACommutationCplt;
      htim->hdma[TIM_DMA_ID_COMMUTATION]->XferHalfCpltCallback =  TIMEx_DMACommutationHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_COMMUTATION]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_COMMUTATION], (uint32_t)BurstBuffer,
                           (uint32_t)&htim->Instance->DMAR, DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_TRIGGER:
    {
      /* Set the DMA trigger callbacks */
      htim->hdma[TIM_DMA_ID_TRIGGER]->XferCpltCallback = TIM_DMATriggerCplt;
      htim->hdma[TIM_DMA_ID_TRIGGER]->XferHalfCpltCallback = TIM_DMATriggerHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_TRIGGER]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_TRIGGER], (uint32_t)BurstBuffer,
                           (uint32_t)&htim->Instance->DMAR, DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Configure the DMA Burst Mode */
    htim->Instance->DCR = (BurstBaseAddress | BurstLength);
    /* Enable the TIM DMA Request */
    __HAL_TIM_ENABLE_DMA(htim, BurstRequestSrc);
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Stops the TIM DMA Burst mode
  * @param  htim TIM handle
  * @param  BurstRequestSrc TIM DMA Request sources to disable
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_DMABurst_WriteStop(TIM_HandleTypeDef *htim, uint32_t BurstRequestSrc)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_DMA_SOURCE(BurstRequestSrc));

  /* Abort the DMA transfer (at least disable the DMA stream) */
  switch (BurstRequestSrc)
  {
    case TIM_DMA_UPDATE:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_UPDATE]);
      break;
    }
    case TIM_DMA_CC1:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC1]);
      break;
    }
    case TIM_DMA_CC2:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC2]);
      break;
    }
    case TIM_DMA_CC3:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC3]);
      break;
    }
    case TIM_DMA_CC4:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC4]);
      break;
    }
    case TIM_DMA_COM:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_COMMUTATION]);
      break;
    }
    case TIM_DMA_TRIGGER:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_TRIGGER]);
      break;
    }
    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Disable the TIM Update DMA request */
    __HAL_TIM_DISABLE_DMA(htim, BurstRequestSrc);

    /* Change the DMA burst operation state */
    htim->DMABurstState = HAL_DMA_BURST_STATE_READY;
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Configure the DMA Burst to transfer Data from the TIM peripheral to the memory
  * @param  htim TIM handle
  * @param  BurstBaseAddress TIM Base address from where the DMA  will start the Data read
  *         This parameter can be one of the following values:
  *            @arg TIM_DMABASE_CR1
  *            @arg TIM_DMABASE_CR2
  *            @arg TIM_DMABASE_SMCR
  *            @arg TIM_DMABASE_DIER
  *            @arg TIM_DMABASE_SR
  *            @arg TIM_DMABASE_EGR
  *            @arg TIM_DMABASE_CCMR1
  *            @arg TIM_DMABASE_CCMR2
  *            @arg TIM_DMABASE_CCER
  *            @arg TIM_DMABASE_CNT
  *            @arg TIM_DMABASE_PSC
  *            @arg TIM_DMABASE_ARR
  *            @arg TIM_DMABASE_RCR
  *            @arg TIM_DMABASE_CCR1
  *            @arg TIM_DMABASE_CCR2
  *            @arg TIM_DMABASE_CCR3
  *            @arg TIM_DMABASE_CCR4
  *            @arg TIM_DMABASE_BDTR
  * @param  BurstRequestSrc TIM DMA Request sources
  *         This parameter can be one of the following values:
  *            @arg TIM_DMA_UPDATE: TIM update Interrupt source
  *            @arg TIM_DMA_CC1: TIM Capture Compare 1 DMA source
  *            @arg TIM_DMA_CC2: TIM Capture Compare 2 DMA source
  *            @arg TIM_DMA_CC3: TIM Capture Compare 3 DMA source
  *            @arg TIM_DMA_CC4: TIM Capture Compare 4 DMA source
  *            @arg TIM_DMA_COM: TIM Commutation DMA source
  *            @arg TIM_DMA_TRIGGER: TIM Trigger DMA source
  * @param  BurstBuffer The Buffer address.
  * @param  BurstLength DMA Burst length. This parameter can be one value
  *         between: TIM_DMABURSTLENGTH_1TRANSFER and TIM_DMABURSTLENGTH_18TRANSFERS.
  * @note   This function should be used only when BurstLength is equal to DMA data transfer length.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_DMABurst_ReadStart(TIM_HandleTypeDef *htim, uint32_t BurstBaseAddress,
                                             uint32_t BurstRequestSrc, uint32_t  *BurstBuffer, uint32_t  BurstLength)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (status == HAL_OK)
  {
    status = HAL_TIM_DMABurst_MultiReadStart(htim, BurstBaseAddress, BurstRequestSrc, BurstBuffer, BurstLength,
                                             ((BurstLength) >> 8U) + 1U);
  }

  return status;
}

/**
  * @brief  Configure the DMA Burst to transfer Data from the TIM peripheral to the memory
  * @param  htim TIM handle
  * @param  BurstBaseAddress TIM Base address from where the DMA  will start the Data read
  *         This parameter can be one of the following values:
  *            @arg TIM_DMABASE_CR1
  *            @arg TIM_DMABASE_CR2
  *            @arg TIM_DMABASE_SMCR
  *            @arg TIM_DMABASE_DIER
  *            @arg TIM_DMABASE_SR
  *            @arg TIM_DMABASE_EGR
  *            @arg TIM_DMABASE_CCMR1
  *            @arg TIM_DMABASE_CCMR2
  *            @arg TIM_DMABASE_CCER
  *            @arg TIM_DMABASE_CNT
  *            @arg TIM_DMABASE_PSC
  *            @arg TIM_DMABASE_ARR
  *            @arg TIM_DMABASE_RCR
  *            @arg TIM_DMABASE_CCR1
  *            @arg TIM_DMABASE_CCR2
  *            @arg TIM_DMABASE_CCR3
  *            @arg TIM_DMABASE_CCR4
  *            @arg TIM_DMABASE_BDTR
  * @param  BurstRequestSrc TIM DMA Request sources
  *         This parameter can be one of the following values:
  *            @arg TIM_DMA_UPDATE: TIM update Interrupt source
  *            @arg TIM_DMA_CC1: TIM Capture Compare 1 DMA source
  *            @arg TIM_DMA_CC2: TIM Capture Compare 2 DMA source
  *            @arg TIM_DMA_CC3: TIM Capture Compare 3 DMA source
  *            @arg TIM_DMA_CC4: TIM Capture Compare 4 DMA source
  *            @arg TIM_DMA_COM: TIM Commutation DMA source
  *            @arg TIM_DMA_TRIGGER: TIM Trigger DMA source
  * @param  BurstBuffer The Buffer address.
  * @param  BurstLength DMA Burst length. This parameter can be one value
  *         between: TIM_DMABURSTLENGTH_1TRANSFER and TIM_DMABURSTLENGTH_18TRANSFERS.
  * @param  DataLength Data length. This parameter can be one value
  *         between 1 and 0xFFFF.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_DMABurst_MultiReadStart(TIM_HandleTypeDef *htim, uint32_t BurstBaseAddress,
                                                  uint32_t BurstRequestSrc, uint32_t  *BurstBuffer,
                                                  uint32_t  BurstLength, uint32_t  DataLength)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_DMABURST_INSTANCE(htim->Instance));
  assert_param(IS_TIM_DMA_BASE(BurstBaseAddress));
  assert_param(IS_TIM_DMA_SOURCE(BurstRequestSrc));
  assert_param(IS_TIM_DMA_LENGTH(BurstLength));
  assert_param(IS_TIM_DMA_DATA_LENGTH(DataLength));

  if (htim->DMABurstState == HAL_DMA_BURST_STATE_BUSY)
  {
    return HAL_BUSY;
  }
  else if (htim->DMABurstState == HAL_DMA_BURST_STATE_READY)
  {
    if ((BurstBuffer == NULL) && (BurstLength > 0U))
    {
      return HAL_ERROR;
    }
    else
    {
      htim->DMABurstState = HAL_DMA_BURST_STATE_BUSY;
    }
  }
  else
  {
    /* nothing to do */
  }
  switch (BurstRequestSrc)
  {
    case TIM_DMA_UPDATE:
    {
      /* Set the DMA Period elapsed callbacks */
      htim->hdma[TIM_DMA_ID_UPDATE]->XferCpltCallback = TIM_DMAPeriodElapsedCplt;
      htim->hdma[TIM_DMA_ID_UPDATE]->XferHalfCpltCallback = TIM_DMAPeriodElapsedHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_UPDATE]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_UPDATE], (uint32_t)&htim->Instance->DMAR, (uint32_t)BurstBuffer,
                           DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_CC1:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC1]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC1]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC1]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC1], (uint32_t)&htim->Instance->DMAR, (uint32_t)BurstBuffer,
                           DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_CC2:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC2]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC2]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC2]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC2], (uint32_t)&htim->Instance->DMAR, (uint32_t)BurstBuffer,
                           DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_CC3:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC3]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC3]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC3]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC3], (uint32_t)&htim->Instance->DMAR, (uint32_t)BurstBuffer,
                           DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_CC4:
    {
      /* Set the DMA capture callbacks */
      htim->hdma[TIM_DMA_ID_CC4]->XferCpltCallback = TIM_DMACaptureCplt;
      htim->hdma[TIM_DMA_ID_CC4]->XferHalfCpltCallback = TIM_DMACaptureHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_CC4]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_CC4], (uint32_t)&htim->Instance->DMAR, (uint32_t)BurstBuffer,
                           DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_COM:
    {
      /* Set the DMA commutation callbacks */
      htim->hdma[TIM_DMA_ID_COMMUTATION]->XferCpltCallback =  TIMEx_DMACommutationCplt;
      htim->hdma[TIM_DMA_ID_COMMUTATION]->XferHalfCpltCallback =  TIMEx_DMACommutationHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_COMMUTATION]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_COMMUTATION], (uint32_t)&htim->Instance->DMAR, (uint32_t)BurstBuffer,
                           DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    case TIM_DMA_TRIGGER:
    {
      /* Set the DMA trigger callbacks */
      htim->hdma[TIM_DMA_ID_TRIGGER]->XferCpltCallback = TIM_DMATriggerCplt;
      htim->hdma[TIM_DMA_ID_TRIGGER]->XferHalfCpltCallback = TIM_DMATriggerHalfCplt;

      /* Set the DMA error callback */
      htim->hdma[TIM_DMA_ID_TRIGGER]->XferErrorCallback = TIM_DMAError ;

      /* Enable the DMA stream */
      if (HAL_DMA_Start_IT(htim->hdma[TIM_DMA_ID_TRIGGER], (uint32_t)&htim->Instance->DMAR, (uint32_t)BurstBuffer,
                           DataLength) != HAL_OK)
      {
        /* Return error status */
        return HAL_ERROR;
      }
      break;
    }
    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Configure the DMA Burst Mode */
    htim->Instance->DCR = (BurstBaseAddress | BurstLength);

    /* Enable the TIM DMA Request */
    __HAL_TIM_ENABLE_DMA(htim, BurstRequestSrc);
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Stop the DMA burst reading
  * @param  htim TIM handle
  * @param  BurstRequestSrc TIM DMA Request sources to disable.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_DMABurst_ReadStop(TIM_HandleTypeDef *htim, uint32_t BurstRequestSrc)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_DMA_SOURCE(BurstRequestSrc));

  /* Abort the DMA transfer (at least disable the DMA stream) */
  switch (BurstRequestSrc)
  {
    case TIM_DMA_UPDATE:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_UPDATE]);
      break;
    }
    case TIM_DMA_CC1:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC1]);
      break;
    }
    case TIM_DMA_CC2:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC2]);
      break;
    }
    case TIM_DMA_CC3:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC3]);
      break;
    }
    case TIM_DMA_CC4:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_CC4]);
      break;
    }
    case TIM_DMA_COM:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_COMMUTATION]);
      break;
    }
    case TIM_DMA_TRIGGER:
    {
      (void)HAL_DMA_Abort_IT(htim->hdma[TIM_DMA_ID_TRIGGER]);
      break;
    }
    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    /* Disable the TIM Update DMA request */
    __HAL_TIM_DISABLE_DMA(htim, BurstRequestSrc);

    /* Change the DMA burst operation state */
    htim->DMABurstState = HAL_DMA_BURST_STATE_READY;
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Generate a software event
  * @param  htim TIM handle
  * @param  EventSource specifies the event source.
  *          This parameter can be one of the following values:
  *            @arg TIM_EVENTSOURCE_UPDATE: Timer update Event source
  *            @arg TIM_EVENTSOURCE_CC1: Timer Capture Compare 1 Event source
  *            @arg TIM_EVENTSOURCE_CC2: Timer Capture Compare 2 Event source
  *            @arg TIM_EVENTSOURCE_CC3: Timer Capture Compare 3 Event source
  *            @arg TIM_EVENTSOURCE_CC4: Timer Capture Compare 4 Event source
  *            @arg TIM_EVENTSOURCE_COM: Timer COM event source
  *            @arg TIM_EVENTSOURCE_TRIGGER: Timer Trigger Event source
  *            @arg TIM_EVENTSOURCE_BREAK: Timer Break event source
  * @note   Basic timers can only generate an update event.
  * @note   TIM_EVENTSOURCE_COM is relevant only with advanced timer instances.
  * @note   TIM_EVENTSOURCE_BREAK are relevant only for timer instances
  *         supporting a break input.
  * @retval HAL status
  */

HAL_StatusTypeDef HAL_TIM_GenerateEvent(TIM_HandleTypeDef *htim, uint32_t EventSource)
{
  /* Check the parameters */
  assert_param(IS_TIM_INSTANCE(htim->Instance));
  assert_param(IS_TIM_EVENT_SOURCE(EventSource));

  /* Process Locked */
  __HAL_LOCK(htim);

  /* Change the TIM state */
  htim->State = HAL_TIM_STATE_BUSY;

  /* Set the event sources */
  htim->Instance->EGR = EventSource;

  /* Change the TIM state */
  htim->State = HAL_TIM_STATE_READY;

  __HAL_UNLOCK(htim);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Configures the OCRef clear feature
  * @param  htim TIM handle
  * @param  sClearInputConfig pointer to a TIM_ClearInputConfigTypeDef structure that
  *         contains the OCREF clear feature and parameters for the TIM peripheral.
  * @param  Channel specifies the TIM Channel
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1
  *            @arg TIM_CHANNEL_2: TIM Channel 2
  *            @arg TIM_CHANNEL_3: TIM Channel 3
  *            @arg TIM_CHANNEL_4: TIM Channel 4
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_ConfigOCrefClear(TIM_HandleTypeDef *htim,
                                           TIM_ClearInputConfigTypeDef *sClearInputConfig,
                                           uint32_t Channel)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_TIM_OCXREF_CLEAR_INSTANCE(htim->Instance));
  assert_param(IS_TIM_CLEARINPUT_SOURCE(sClearInputConfig->ClearInputSource));

  /* Process Locked */
  __HAL_LOCK(htim);

  htim->State = HAL_TIM_STATE_BUSY;

  switch (sClearInputConfig->ClearInputSource)
  {
    case TIM_CLEARINPUTSOURCE_NONE:
    {
      /* Clear the OCREF clear selection bit and the the ETR Bits */
      CLEAR_BIT(htim->Instance->SMCR, (TIM_SMCR_ETF | TIM_SMCR_ETPS | TIM_SMCR_ECE | TIM_SMCR_ETP));
      break;
    }

    case TIM_CLEARINPUTSOURCE_ETR:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CLEARINPUT_POLARITY(sClearInputConfig->ClearInputPolarity));
      assert_param(IS_TIM_CLEARINPUT_PRESCALER(sClearInputConfig->ClearInputPrescaler));
      assert_param(IS_TIM_CLEARINPUT_FILTER(sClearInputConfig->ClearInputFilter));

      /* When OCRef clear feature is used with ETR source, ETR prescaler must be off */
      if (sClearInputConfig->ClearInputPrescaler != TIM_CLEARINPUTPRESCALER_DIV1)
      {
        htim->State = HAL_TIM_STATE_READY;
        __HAL_UNLOCK(htim);
        return HAL_ERROR;
      }

      TIM_ETR_SetConfig(htim->Instance,
                        sClearInputConfig->ClearInputPrescaler,
                        sClearInputConfig->ClearInputPolarity,
                        sClearInputConfig->ClearInputFilter);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    switch (Channel)
    {
      case TIM_CHANNEL_1:
      {
        if (sClearInputConfig->ClearInputState != (uint32_t)DISABLE)
        {
          /* Enable the OCREF clear feature for Channel 1 */
          SET_BIT(htim->Instance->CCMR1, TIM_CCMR1_OC1CE);
        }
        else
        {
          /* Disable the OCREF clear feature for Channel 1 */
          CLEAR_BIT(htim->Instance->CCMR1, TIM_CCMR1_OC1CE);
        }
        break;
      }
      case TIM_CHANNEL_2:
      {
        if (sClearInputConfig->ClearInputState != (uint32_t)DISABLE)
        {
          /* Enable the OCREF clear feature for Channel 2 */
          SET_BIT(htim->Instance->CCMR1, TIM_CCMR1_OC2CE);
        }
        else
        {
          /* Disable the OCREF clear feature for Channel 2 */
          CLEAR_BIT(htim->Instance->CCMR1, TIM_CCMR1_OC2CE);
        }
        break;
      }
      case TIM_CHANNEL_3:
      {
        if (sClearInputConfig->ClearInputState != (uint32_t)DISABLE)
        {
          /* Enable the OCREF clear feature for Channel 3 */
          SET_BIT(htim->Instance->CCMR2, TIM_CCMR2_OC3CE);
        }
        else
        {
          /* Disable the OCREF clear feature for Channel 3 */
          CLEAR_BIT(htim->Instance->CCMR2, TIM_CCMR2_OC3CE);
        }
        break;
      }
      case TIM_CHANNEL_4:
      {
        if (sClearInputConfig->ClearInputState != (uint32_t)DISABLE)
        {
          /* Enable the OCREF clear feature for Channel 4 */
          SET_BIT(htim->Instance->CCMR2, TIM_CCMR2_OC4CE);
        }
        else
        {
          /* Disable the OCREF clear feature for Channel 4 */
          CLEAR_BIT(htim->Instance->CCMR2, TIM_CCMR2_OC4CE);
        }
        break;
      }
      default:
        break;
    }
  }

  htim->State = HAL_TIM_STATE_READY;

  __HAL_UNLOCK(htim);

  return status;
}

/**
  * @brief   Configures the clock source to be used
  * @param  htim TIM handle
  * @param  sClockSourceConfig pointer to a TIM_ClockConfigTypeDef structure that
  *         contains the clock source information for the TIM peripheral.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_ConfigClockSource(TIM_HandleTypeDef *htim, TIM_ClockConfigTypeDef *sClockSourceConfig)
{
  HAL_StatusTypeDef status = HAL_OK;
  uint32_t tmpsmcr;

  /* Process Locked */
  __HAL_LOCK(htim);

  htim->State = HAL_TIM_STATE_BUSY;

  /* Check the parameters */
  assert_param(IS_TIM_CLOCKSOURCE(sClockSourceConfig->ClockSource));

  /* Reset the SMS, TS, ECE, ETPS and ETRF bits */
  tmpsmcr = htim->Instance->SMCR;
  tmpsmcr &= ~(TIM_SMCR_SMS | TIM_SMCR_TS);
  tmpsmcr &= ~(TIM_SMCR_ETF | TIM_SMCR_ETPS | TIM_SMCR_ECE | TIM_SMCR_ETP);
  htim->Instance->SMCR = tmpsmcr;

  switch (sClockSourceConfig->ClockSource)
  {
    case TIM_CLOCKSOURCE_INTERNAL:
    {
      assert_param(IS_TIM_INSTANCE(htim->Instance));
      break;
    }

    case TIM_CLOCKSOURCE_ETRMODE1:
    {
      /* Check whether or not the timer instance supports external trigger input mode 1 (ETRF)*/
      assert_param(IS_TIM_CLOCKSOURCE_ETRMODE1_INSTANCE(htim->Instance));

      /* Check ETR input conditioning related parameters */
      assert_param(IS_TIM_CLOCKPRESCALER(sClockSourceConfig->ClockPrescaler));
      assert_param(IS_TIM_CLOCKPOLARITY(sClockSourceConfig->ClockPolarity));
      assert_param(IS_TIM_CLOCKFILTER(sClockSourceConfig->ClockFilter));

      /* Configure the ETR Clock source */
      TIM_ETR_SetConfig(htim->Instance,
                        sClockSourceConfig->ClockPrescaler,
                        sClockSourceConfig->ClockPolarity,
                        sClockSourceConfig->ClockFilter);

      /* Select the External clock mode1 and the ETRF trigger */
      tmpsmcr = htim->Instance->SMCR;
      tmpsmcr |= (TIM_SLAVEMODE_EXTERNAL1 | TIM_CLOCKSOURCE_ETRMODE1);
      /* Write to TIMx SMCR */
      htim->Instance->SMCR = tmpsmcr;
      break;
    }

    case TIM_CLOCKSOURCE_ETRMODE2:
    {
      /* Check whether or not the timer instance supports external trigger input mode 2 (ETRF)*/
      assert_param(IS_TIM_CLOCKSOURCE_ETRMODE2_INSTANCE(htim->Instance));

      /* Check ETR input conditioning related parameters */
      assert_param(IS_TIM_CLOCKPRESCALER(sClockSourceConfig->ClockPrescaler));
      assert_param(IS_TIM_CLOCKPOLARITY(sClockSourceConfig->ClockPolarity));
      assert_param(IS_TIM_CLOCKFILTER(sClockSourceConfig->ClockFilter));

      /* Configure the ETR Clock source */
      TIM_ETR_SetConfig(htim->Instance,
                        sClockSourceConfig->ClockPrescaler,
                        sClockSourceConfig->ClockPolarity,
                        sClockSourceConfig->ClockFilter);
      /* Enable the External clock mode2 */
      htim->Instance->SMCR |= TIM_SMCR_ECE;
      break;
    }

    case TIM_CLOCKSOURCE_TI1:
    {
      /* Check whether or not the timer instance supports external clock mode 1 */
      assert_param(IS_TIM_CLOCKSOURCE_TIX_INSTANCE(htim->Instance));

      /* Check TI1 input conditioning related parameters */
      assert_param(IS_TIM_CLOCKPOLARITY(sClockSourceConfig->ClockPolarity));
      assert_param(IS_TIM_CLOCKFILTER(sClockSourceConfig->ClockFilter));

      TIM_TI1_ConfigInputStage(htim->Instance,
                               sClockSourceConfig->ClockPolarity,
                               sClockSourceConfig->ClockFilter);
      TIM_ITRx_SetConfig(htim->Instance, TIM_CLOCKSOURCE_TI1);
      break;
    }

    case TIM_CLOCKSOURCE_TI2:
    {
      /* Check whether or not the timer instance supports external clock mode 1 (ETRF)*/
      assert_param(IS_TIM_CLOCKSOURCE_TIX_INSTANCE(htim->Instance));

      /* Check TI2 input conditioning related parameters */
      assert_param(IS_TIM_CLOCKPOLARITY(sClockSourceConfig->ClockPolarity));
      assert_param(IS_TIM_CLOCKFILTER(sClockSourceConfig->ClockFilter));

      TIM_TI2_ConfigInputStage(htim->Instance,
                               sClockSourceConfig->ClockPolarity,
                               sClockSourceConfig->ClockFilter);
      TIM_ITRx_SetConfig(htim->Instance, TIM_CLOCKSOURCE_TI2);
      break;
    }

    case TIM_CLOCKSOURCE_TI1ED:
    {
      /* Check whether or not the timer instance supports external clock mode 1 */
      assert_param(IS_TIM_CLOCKSOURCE_TIX_INSTANCE(htim->Instance));

      /* Check TI1 input conditioning related parameters */
      assert_param(IS_TIM_CLOCKPOLARITY(sClockSourceConfig->ClockPolarity));
      assert_param(IS_TIM_CLOCKFILTER(sClockSourceConfig->ClockFilter));

      TIM_TI1_ConfigInputStage(htim->Instance,
                               sClockSourceConfig->ClockPolarity,
                               sClockSourceConfig->ClockFilter);
      TIM_ITRx_SetConfig(htim->Instance, TIM_CLOCKSOURCE_TI1ED);
      break;
    }

    case TIM_CLOCKSOURCE_ITR0:
    case TIM_CLOCKSOURCE_ITR1:
    case TIM_CLOCKSOURCE_ITR2:
    case TIM_CLOCKSOURCE_ITR3:
    {
      /* Check whether or not the timer instance supports internal trigger input */
      assert_param(IS_TIM_CLOCKSOURCE_ITRX_INSTANCE(htim->Instance));

      TIM_ITRx_SetConfig(htim->Instance, sClockSourceConfig->ClockSource);
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }
  htim->State = HAL_TIM_STATE_READY;

  __HAL_UNLOCK(htim);

  return status;
}

/**
  * @brief  Selects the signal connected to the TI1 input: direct from CH1_input
  *         or a XOR combination between CH1_input, CH2_input & CH3_input
  * @param  htim TIM handle.
  * @param  TI1_Selection Indicate whether or not channel 1 is connected to the
  *         output of a XOR gate.
  *          This parameter can be one of the following values:
  *            @arg TIM_TI1SELECTION_CH1: The TIMx_CH1 pin is connected to TI1 input
  *            @arg TIM_TI1SELECTION_XORCOMBINATION: The TIMx_CH1, CH2 and CH3
  *            pins are connected to the TI1 input (XOR combination)
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_ConfigTI1Input(TIM_HandleTypeDef *htim, uint32_t TI1_Selection)
{
  uint32_t tmpcr2;

  /* Check the parameters */
  assert_param(IS_TIM_XOR_INSTANCE(htim->Instance));
  assert_param(IS_TIM_TI1SELECTION(TI1_Selection));

  /* Get the TIMx CR2 register value */
  tmpcr2 = htim->Instance->CR2;

  /* Reset the TI1 selection */
  tmpcr2 &= ~TIM_CR2_TI1S;

  /* Set the TI1 selection */
  tmpcr2 |= TI1_Selection;

  /* Write to TIMxCR2 */
  htim->Instance->CR2 = tmpcr2;

  return HAL_OK;
}

/**
  * @brief  Configures the TIM in Slave mode
  * @param  htim TIM handle.
  * @param  sSlaveConfig pointer to a TIM_SlaveConfigTypeDef structure that
  *         contains the selected trigger (internal trigger input, filtered
  *         timer input or external trigger input) and the Slave mode
  *         (Disable, Reset, Gated, Trigger, External clock mode 1).
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_SlaveConfigSynchro(TIM_HandleTypeDef *htim, TIM_SlaveConfigTypeDef *sSlaveConfig)
{
  /* Check the parameters */
  assert_param(IS_TIM_SLAVE_INSTANCE(htim->Instance));
  assert_param(IS_TIM_SLAVE_MODE(sSlaveConfig->SlaveMode));
  assert_param(IS_TIM_TRIGGER_SELECTION(sSlaveConfig->InputTrigger));

  __HAL_LOCK(htim);

  htim->State = HAL_TIM_STATE_BUSY;

  if (TIM_SlaveTimer_SetConfig(htim, sSlaveConfig) != HAL_OK)
  {
    htim->State = HAL_TIM_STATE_READY;
    __HAL_UNLOCK(htim);
    return HAL_ERROR;
  }

  /* Disable Trigger Interrupt */
  __HAL_TIM_DISABLE_IT(htim, TIM_IT_TRIGGER);

  /* Disable Trigger DMA request */
  __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_TRIGGER);

  htim->State = HAL_TIM_STATE_READY;

  __HAL_UNLOCK(htim);

  return HAL_OK;
}

/**
  * @brief  Configures the TIM in Slave mode in interrupt mode
  * @param  htim TIM handle.
  * @param  sSlaveConfig pointer to a TIM_SlaveConfigTypeDef structure that
  *         contains the selected trigger (internal trigger input, filtered
  *         timer input or external trigger input) and the Slave mode
  *         (Disable, Reset, Gated, Trigger, External clock mode 1).
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_TIM_SlaveConfigSynchro_IT(TIM_HandleTypeDef *htim,
                                                TIM_SlaveConfigTypeDef *sSlaveConfig)
{
  /* Check the parameters */
  assert_param(IS_TIM_SLAVE_INSTANCE(htim->Instance));
  assert_param(IS_TIM_SLAVE_MODE(sSlaveConfig->SlaveMode));
  assert_param(IS_TIM_TRIGGER_SELECTION(sSlaveConfig->InputTrigger));

  __HAL_LOCK(htim);

  htim->State = HAL_TIM_STATE_BUSY;

  if (TIM_SlaveTimer_SetConfig(htim, sSlaveConfig) != HAL_OK)
  {
    htim->State = HAL_TIM_STATE_READY;
    __HAL_UNLOCK(htim);
    return HAL_ERROR;
  }

  /* Enable Trigger Interrupt */
  __HAL_TIM_ENABLE_IT(htim, TIM_IT_TRIGGER);

  /* Disable Trigger DMA request */
  __HAL_TIM_DISABLE_DMA(htim, TIM_DMA_TRIGGER);

  htim->State = HAL_TIM_STATE_READY;

  __HAL_UNLOCK(htim);

  return HAL_OK;
}

/**
  * @brief  Read the captured value from Capture Compare unit
  * @param  htim TIM handle.
  * @param  Channel TIM Channels to be enabled
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1 selected
  *            @arg TIM_CHANNEL_2: TIM Channel 2 selected
  *            @arg TIM_CHANNEL_3: TIM Channel 3 selected
  *            @arg TIM_CHANNEL_4: TIM Channel 4 selected
  * @retval Captured value
  */
uint32_t HAL_TIM_ReadCapturedValue(TIM_HandleTypeDef *htim, uint32_t Channel)
{
  uint32_t tmpreg = 0U;

  switch (Channel)
  {
    case TIM_CHANNEL_1:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC1_INSTANCE(htim->Instance));

      /* Return the capture 1 value */
      tmpreg =  htim->Instance->CCR1;

      break;
    }
    case TIM_CHANNEL_2:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC2_INSTANCE(htim->Instance));

      /* Return the capture 2 value */
      tmpreg =   htim->Instance->CCR2;

      break;
    }

    case TIM_CHANNEL_3:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC3_INSTANCE(htim->Instance));

      /* Return the capture 3 value */
      tmpreg =   htim->Instance->CCR3;

      break;
    }

    case TIM_CHANNEL_4:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC4_INSTANCE(htim->Instance));

      /* Return the capture 4 value */
      tmpreg =   htim->Instance->CCR4;

      break;
    }

    default:
      break;
  }

  return tmpreg;
}

/**
  * @}
  */

/** @defgroup TIM_Exported_Functions_Group9 TIM Callbacks functions
  *  @brief    TIM Callbacks functions
  *
@verbatim
  ==============================================================================
                        ##### TIM Callbacks functions #####
  ==============================================================================
 [..]
   This section provides TIM callback functions:
   (+) TIM Period elapsed callback
   (+) TIM Output Compare callback
   (+) TIM Input capture callback
   (+) TIM Trigger callback
   (+) TIM Error callback

@endverbatim
  * @{
  */

/**
  * @brief  Period elapsed callback in non-blocking mode
  * @param  htim TIM handle
  * @retval None
  */
__weak void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_PeriodElapsedCallback could be implemented in the user file
   */
}

/**
  * @brief  Period elapsed half complete callback in non-blocking mode
  * @param  htim TIM handle
  * @retval None
  */
__weak void HAL_TIM_PeriodElapsedHalfCpltCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_PeriodElapsedHalfCpltCallback could be implemented in the user file
   */
}

/**
  * @brief  Output Compare callback in non-blocking mode
  * @param  htim TIM OC handle
  * @retval None
  */
__weak void HAL_TIM_OC_DelayElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_OC_DelayElapsedCallback could be implemented in the user file
   */
}

/**
  * @brief  Input Capture callback in non-blocking mode
  * @param  htim TIM IC handle
  * @retval None
  */
__weak void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_IC_CaptureCallback could be implemented in the user file
   */
}

/**
  * @brief  Input Capture half complete callback in non-blocking mode
  * @param  htim TIM IC handle
  * @retval None
  */
__weak void HAL_TIM_IC_CaptureHalfCpltCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_IC_CaptureHalfCpltCallback could be implemented in the user file
   */
}

/**
  * @brief  PWM Pulse finished callback in non-blocking mode
  * @param  htim TIM handle
  * @retval None
  */
__weak void HAL_TIM_PWM_PulseFinishedCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_PWM_PulseFinishedCallback could be implemented in the user file
   */
}

/**
  * @brief  PWM Pulse finished half complete callback in non-blocking mode
  * @param  htim TIM handle
  * @retval None
  */
__weak void HAL_TIM_PWM_PulseFinishedHalfCpltCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_PWM_PulseFinishedHalfCpltCallback could be implemented in the user file
   */
}

/**
  * @brief  Hall Trigger detection callback in non-blocking mode
  * @param  htim TIM handle
  * @retval None
  */
__weak void HAL_TIM_TriggerCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_TriggerCallback could be implemented in the user file
   */
}

/**
  * @brief  Hall Trigger detection half complete callback in non-blocking mode
  * @param  htim TIM handle
  * @retval None
  */
__weak void HAL_TIM_TriggerHalfCpltCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_TriggerHalfCpltCallback could be implemented in the user file
   */
}

/**
  * @brief  Timer error callback in non-blocking mode
  * @param  htim TIM handle
  * @retval None
  */
__weak void HAL_TIM_ErrorCallback(TIM_HandleTypeDef *htim)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(htim);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_TIM_ErrorCallback could be implemented in the user file
   */
}

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User TIM callback to be used instead of the weak predefined callback
  * @param htim tim handle
  * @param CallbackID ID of the callback to be registered
  *        This parameter can be one of the following values:
  *          @arg @ref HAL_TIM_BASE_MSPINIT_CB_ID Base MspInit Callback ID
  *          @arg @ref HAL_TIM_BASE_MSPDEINIT_CB_ID Base MspDeInit Callback ID
  *          @arg @ref HAL_TIM_IC_MSPINIT_CB_ID IC MspInit Callback ID
  *          @arg @ref HAL_TIM_IC_MSPDEINIT_CB_ID IC MspDeInit Callback ID
  *          @arg @ref HAL_TIM_OC_MSPINIT_CB_ID OC MspInit Callback ID
  *          @arg @ref HAL_TIM_OC_MSPDEINIT_CB_ID OC MspDeInit Callback ID
  *          @arg @ref HAL_TIM_PWM_MSPINIT_CB_ID PWM MspInit Callback ID
  *          @arg @ref HAL_TIM_PWM_MSPDEINIT_CB_ID PWM MspDeInit Callback ID
  *          @arg @ref HAL_TIM_ONE_PULSE_MSPINIT_CB_ID One Pulse MspInit Callback ID
  *          @arg @ref HAL_TIM_ONE_PULSE_MSPDEINIT_CB_ID One Pulse MspDeInit Callback ID
  *          @arg @ref HAL_TIM_ENCODER_MSPINIT_CB_ID Encoder MspInit Callback ID
  *          @arg @ref HAL_TIM_ENCODER_MSPDEINIT_CB_ID Encoder MspDeInit Callback ID
  *          @arg @ref HAL_TIM_HALL_SENSOR_MSPINIT_CB_ID Hall Sensor MspInit Callback ID
  *          @arg @ref HAL_TIM_HALL_SENSOR_MSPDEINIT_CB_ID Hall Sensor MspDeInit Callback ID
  *          @arg @ref HAL_TIM_PERIOD_ELAPSED_CB_ID Period Elapsed Callback ID
  *          @arg @ref HAL_TIM_PERIOD_ELAPSED_HALF_CB_ID Period Elapsed half complete Callback ID
  *          @arg @ref HAL_TIM_TRIGGER_CB_ID Trigger Callback ID
  *          @arg @ref HAL_TIM_TRIGGER_HALF_CB_ID Trigger half complete Callback ID
  *          @arg @ref HAL_TIM_IC_CAPTURE_CB_ID Input Capture Callback ID
  *          @arg @ref HAL_TIM_IC_CAPTURE_HALF_CB_ID Input Capture half complete Callback ID
  *          @arg @ref HAL_TIM_OC_DELAY_ELAPSED_CB_ID Output Compare Delay Elapsed Callback ID
  *          @arg @ref HAL_TIM_PWM_PULSE_FINISHED_CB_ID PWM Pulse Finished Callback ID
  *          @arg @ref HAL_TIM_PWM_PULSE_FINISHED_HALF_CB_ID PWM Pulse Finished half complete Callback ID
  *          @arg @ref HAL_TIM_ERROR_CB_ID Error Callback ID
  *          @arg @ref HAL_TIM_COMMUTATION_CB_ID Commutation Callback ID
  *          @arg @ref HAL_TIM_COMMUTATION_HALF_CB_ID Commutation half complete Callback ID
  *          @arg @ref HAL_TIM_BREAK_CB_ID Break Callback ID
  *          @param pCallback pointer to the callback function
  *          @retval status
  */
HAL_StatusTypeDef HAL_TIM_RegisterCallback(TIM_HandleTypeDef *htim, HAL_TIM_CallbackIDTypeDef CallbackID,
                                           pTIM_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(htim);

  if (htim->State == HAL_TIM_STATE_READY)
  {
    switch (CallbackID)
    {
      case HAL_TIM_BASE_MSPINIT_CB_ID :
        htim->Base_MspInitCallback                 = pCallback;
        break;

      case HAL_TIM_BASE_MSPDEINIT_CB_ID :
        htim->Base_MspDeInitCallback               = pCallback;
        break;

      case HAL_TIM_IC_MSPINIT_CB_ID :
        htim->IC_MspInitCallback                   = pCallback;
        break;

      case HAL_TIM_IC_MSPDEINIT_CB_ID :
        htim->IC_MspDeInitCallback                 = pCallback;
        break;

      case HAL_TIM_OC_MSPINIT_CB_ID :
        htim->OC_MspInitCallback                   = pCallback;
        break;

      case HAL_TIM_OC_MSPDEINIT_CB_ID :
        htim->OC_MspDeInitCallback                 = pCallback;
        break;

      case HAL_TIM_PWM_MSPINIT_CB_ID :
        htim->PWM_MspInitCallback                  = pCallback;
        break;

      case HAL_TIM_PWM_MSPDEINIT_CB_ID :
        htim->PWM_MspDeInitCallback                = pCallback;
        break;

      case HAL_TIM_ONE_PULSE_MSPINIT_CB_ID :
        htim->OnePulse_MspInitCallback             = pCallback;
        break;

      case HAL_TIM_ONE_PULSE_MSPDEINIT_CB_ID :
        htim->OnePulse_MspDeInitCallback           = pCallback;
        break;

      case HAL_TIM_ENCODER_MSPINIT_CB_ID :
        htim->Encoder_MspInitCallback              = pCallback;
        break;

      case HAL_TIM_ENCODER_MSPDEINIT_CB_ID :
        htim->Encoder_MspDeInitCallback            = pCallback;
        break;

      case HAL_TIM_HALL_SENSOR_MSPINIT_CB_ID :
        htim->HallSensor_MspInitCallback           = pCallback;
        break;

      case HAL_TIM_HALL_SENSOR_MSPDEINIT_CB_ID :
        htim->HallSensor_MspDeInitCallback         = pCallback;
        break;

      case HAL_TIM_PERIOD_ELAPSED_CB_ID :
        htim->PeriodElapsedCallback                = pCallback;
        break;

      case HAL_TIM_PERIOD_ELAPSED_HALF_CB_ID :
        htim->PeriodElapsedHalfCpltCallback        = pCallback;
        break;

      case HAL_TIM_TRIGGER_CB_ID :
        htim->TriggerCallback                      = pCallback;
        break;

      case HAL_TIM_TRIGGER_HALF_CB_ID :
        htim->TriggerHalfCpltCallback              = pCallback;
        break;

      case HAL_TIM_IC_CAPTURE_CB_ID :
        htim->IC_CaptureCallback                   = pCallback;
        break;

      case HAL_TIM_IC_CAPTURE_HALF_CB_ID :
        htim->IC_CaptureHalfCpltCallback           = pCallback;
        break;

      case HAL_TIM_OC_DELAY_ELAPSED_CB_ID :
        htim->OC_DelayElapsedCallback              = pCallback;
        break;

      case HAL_TIM_PWM_PULSE_FINISHED_CB_ID :
        htim->PWM_PulseFinishedCallback            = pCallback;
        break;

      case HAL_TIM_PWM_PULSE_FINISHED_HALF_CB_ID :
        htim->PWM_PulseFinishedHalfCpltCallback    = pCallback;
        break;

      case HAL_TIM_ERROR_CB_ID :
        htim->ErrorCallback                        = pCallback;
        break;

      case HAL_TIM_COMMUTATION_CB_ID :
        htim->CommutationCallback                  = pCallback;
        break;

      case HAL_TIM_COMMUTATION_HALF_CB_ID :
        htim->CommutationHalfCpltCallback          = pCallback;
        break;

      case HAL_TIM_BREAK_CB_ID :
        htim->BreakCallback                        = pCallback;
        break;

      default :
        /* Return error status */
        status = HAL_ERROR;
        break;
    }
  }
  else if (htim->State == HAL_TIM_STATE_RESET)
  {
    switch (CallbackID)
    {
      case HAL_TIM_BASE_MSPINIT_CB_ID :
        htim->Base_MspInitCallback         = pCallback;
        break;

      case HAL_TIM_BASE_MSPDEINIT_CB_ID :
        htim->Base_MspDeInitCallback       = pCallback;
        break;

      case HAL_TIM_IC_MSPINIT_CB_ID :
        htim->IC_MspInitCallback           = pCallback;
        break;

      case HAL_TIM_IC_MSPDEINIT_CB_ID :
        htim->IC_MspDeInitCallback         = pCallback;
        break;

      case HAL_TIM_OC_MSPINIT_CB_ID :
        htim->OC_MspInitCallback           = pCallback;
        break;

      case HAL_TIM_OC_MSPDEINIT_CB_ID :
        htim->OC_MspDeInitCallback         = pCallback;
        break;

      case HAL_TIM_PWM_MSPINIT_CB_ID :
        htim->PWM_MspInitCallback          = pCallback;
        break;

      case HAL_TIM_PWM_MSPDEINIT_CB_ID :
        htim->PWM_MspDeInitCallback        = pCallback;
        break;

      case HAL_TIM_ONE_PULSE_MSPINIT_CB_ID :
        htim->OnePulse_MspInitCallback     = pCallback;
        break;

      case HAL_TIM_ONE_PULSE_MSPDEINIT_CB_ID :
        htim->OnePulse_MspDeInitCallback   = pCallback;
        break;

      case HAL_TIM_ENCODER_MSPINIT_CB_ID :
        htim->Encoder_MspInitCallback      = pCallback;
        break;

      case HAL_TIM_ENCODER_MSPDEINIT_CB_ID :
        htim->Encoder_MspDeInitCallback    = pCallback;
        break;

      case HAL_TIM_HALL_SENSOR_MSPINIT_CB_ID :
        htim->HallSensor_MspInitCallback   = pCallback;
        break;

      case HAL_TIM_HALL_SENSOR_MSPDEINIT_CB_ID :
        htim->HallSensor_MspDeInitCallback = pCallback;
        break;

      default :
        /* Return error status */
        status = HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Return error status */
    status = HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(htim);

  return status;
}

/**
  * @brief  Unregister a TIM callback
  *         TIM callback is redirected to the weak predefined callback
  * @param htim tim handle
  * @param CallbackID ID of the callback to be unregistered
  *        This parameter can be one of the following values:
  *          @arg @ref HAL_TIM_BASE_MSPINIT_CB_ID Base MspInit Callback ID
  *          @arg @ref HAL_TIM_BASE_MSPDEINIT_CB_ID Base MspDeInit Callback ID
  *          @arg @ref HAL_TIM_IC_MSPINIT_CB_ID IC MspInit Callback ID
  *          @arg @ref HAL_TIM_IC_MSPDEINIT_CB_ID IC MspDeInit Callback ID
  *          @arg @ref HAL_TIM_OC_MSPINIT_CB_ID OC MspInit Callback ID
  *          @arg @ref HAL_TIM_OC_MSPDEINIT_CB_ID OC MspDeInit Callback ID
  *          @arg @ref HAL_TIM_PWM_MSPINIT_CB_ID PWM MspInit Callback ID
  *          @arg @ref HAL_TIM_PWM_MSPDEINIT_CB_ID PWM MspDeInit Callback ID
  *          @arg @ref HAL_TIM_ONE_PULSE_MSPINIT_CB_ID One Pulse MspInit Callback ID
  *          @arg @ref HAL_TIM_ONE_PULSE_MSPDEINIT_CB_ID One Pulse MspDeInit Callback ID
  *          @arg @ref HAL_TIM_ENCODER_MSPINIT_CB_ID Encoder MspInit Callback ID
  *          @arg @ref HAL_TIM_ENCODER_MSPDEINIT_CB_ID Encoder MspDeInit Callback ID
  *          @arg @ref HAL_TIM_HALL_SENSOR_MSPINIT_CB_ID Hall Sensor MspInit Callback ID
  *          @arg @ref HAL_TIM_HALL_SENSOR_MSPDEINIT_CB_ID Hall Sensor MspDeInit Callback ID
  *          @arg @ref HAL_TIM_PERIOD_ELAPSED_CB_ID Period Elapsed Callback ID
  *          @arg @ref HAL_TIM_PERIOD_ELAPSED_HALF_CB_ID Period Elapsed half complete Callback ID
  *          @arg @ref HAL_TIM_TRIGGER_CB_ID Trigger Callback ID
  *          @arg @ref HAL_TIM_TRIGGER_HALF_CB_ID Trigger half complete Callback ID
  *          @arg @ref HAL_TIM_IC_CAPTURE_CB_ID Input Capture Callback ID
  *          @arg @ref HAL_TIM_IC_CAPTURE_HALF_CB_ID Input Capture half complete Callback ID
  *          @arg @ref HAL_TIM_OC_DELAY_ELAPSED_CB_ID Output Compare Delay Elapsed Callback ID
  *          @arg @ref HAL_TIM_PWM_PULSE_FINISHED_CB_ID PWM Pulse Finished Callback ID
  *          @arg @ref HAL_TIM_PWM_PULSE_FINISHED_HALF_CB_ID PWM Pulse Finished half complete Callback ID
  *          @arg @ref HAL_TIM_ERROR_CB_ID Error Callback ID
  *          @arg @ref HAL_TIM_COMMUTATION_CB_ID Commutation Callback ID
  *          @arg @ref HAL_TIM_COMMUTATION_HALF_CB_ID Commutation half complete Callback ID
  *          @arg @ref HAL_TIM_BREAK_CB_ID Break Callback ID
  *          @retval status
  */
HAL_StatusTypeDef HAL_TIM_UnRegisterCallback(TIM_HandleTypeDef *htim, HAL_TIM_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(htim);

  if (htim->State == HAL_TIM_STATE_READY)
  {
    switch (CallbackID)
    {
      case HAL_TIM_BASE_MSPINIT_CB_ID :
        /* Legacy weak Base MspInit Callback */
        htim->Base_MspInitCallback              = HAL_TIM_Base_MspInit;
        break;

      case HAL_TIM_BASE_MSPDEINIT_CB_ID :
        /* Legacy weak Base Msp DeInit Callback */
        htim->Base_MspDeInitCallback            = HAL_TIM_Base_MspDeInit;
        break;

      case HAL_TIM_IC_MSPINIT_CB_ID :
        /* Legacy weak IC Msp Init Callback */
        htim->IC_MspInitCallback                = HAL_TIM_IC_MspInit;
        break;

      case HAL_TIM_IC_MSPDEINIT_CB_ID :
        /* Legacy weak IC Msp DeInit Callback */
        htim->IC_MspDeInitCallback              = HAL_TIM_IC_MspDeInit;
        break;

      case HAL_TIM_OC_MSPINIT_CB_ID :
        /* Legacy weak OC Msp Init Callback */
        htim->OC_MspInitCallback                = HAL_TIM_OC_MspInit;
        break;

      case HAL_TIM_OC_MSPDEINIT_CB_ID :
        /* Legacy weak OC Msp DeInit Callback */
        htim->OC_MspDeInitCallback              = HAL_TIM_OC_MspDeInit;
        break;

      case HAL_TIM_PWM_MSPINIT_CB_ID :
        /* Legacy weak PWM Msp Init Callback */
        htim->PWM_MspInitCallback               = HAL_TIM_PWM_MspInit;
        break;

      case HAL_TIM_PWM_MSPDEINIT_CB_ID :
        /* Legacy weak PWM Msp DeInit Callback */
        htim->PWM_MspDeInitCallback             = HAL_TIM_PWM_MspDeInit;
        break;

      case HAL_TIM_ONE_PULSE_MSPINIT_CB_ID :
        /* Legacy weak One Pulse Msp Init Callback */
        htim->OnePulse_MspInitCallback          = HAL_TIM_OnePulse_MspInit;
        break;

      case HAL_TIM_ONE_PULSE_MSPDEINIT_CB_ID :
        /* Legacy weak One Pulse Msp DeInit Callback */
        htim->OnePulse_MspDeInitCallback        = HAL_TIM_OnePulse_MspDeInit;
        break;

      case HAL_TIM_ENCODER_MSPINIT_CB_ID :
        /* Legacy weak Encoder Msp Init Callback */
        htim->Encoder_MspInitCallback           = HAL_TIM_Encoder_MspInit;
        break;

      case HAL_TIM_ENCODER_MSPDEINIT_CB_ID :
        /* Legacy weak Encoder Msp DeInit Callback */
        htim->Encoder_MspDeInitCallback         = HAL_TIM_Encoder_MspDeInit;
        break;

      case HAL_TIM_HALL_SENSOR_MSPINIT_CB_ID :
        /* Legacy weak Hall Sensor Msp Init Callback */
        htim->HallSensor_MspInitCallback        = HAL_TIMEx_HallSensor_MspInit;
        break;

      case HAL_TIM_HALL_SENSOR_MSPDEINIT_CB_ID :
        /* Legacy weak Hall Sensor Msp DeInit Callback */
        htim->HallSensor_MspDeInitCallback      = HAL_TIMEx_HallSensor_MspDeInit;
        break;

      case HAL_TIM_PERIOD_ELAPSED_CB_ID :
        /* Legacy weak Period Elapsed Callback */
        htim->PeriodElapsedCallback             = HAL_TIM_PeriodElapsedCallback;
        break;

      case HAL_TIM_PERIOD_ELAPSED_HALF_CB_ID :
        /* Legacy weak Period Elapsed half complete Callback */
        htim->PeriodElapsedHalfCpltCallback     = HAL_TIM_PeriodElapsedHalfCpltCallback;
        break;

      case HAL_TIM_TRIGGER_CB_ID :
        /* Legacy weak Trigger Callback */
        htim->TriggerCallback                   = HAL_TIM_TriggerCallback;
        break;

      case HAL_TIM_TRIGGER_HALF_CB_ID :
        /* Legacy weak Trigger half complete Callback */
        htim->TriggerHalfCpltCallback           = HAL_TIM_TriggerHalfCpltCallback;
        break;

      case HAL_TIM_IC_CAPTURE_CB_ID :
        /* Legacy weak IC Capture Callback */
        htim->IC_CaptureCallback                = HAL_TIM_IC_CaptureCallback;
        break;

      case HAL_TIM_IC_CAPTURE_HALF_CB_ID :
        /* Legacy weak IC Capture half complete Callback */
        htim->IC_CaptureHalfCpltCallback        = HAL_TIM_IC_CaptureHalfCpltCallback;
        break;

      case HAL_TIM_OC_DELAY_ELAPSED_CB_ID :
        /* Legacy weak OC Delay Elapsed Callback */
        htim->OC_DelayElapsedCallback           = HAL_TIM_OC_DelayElapsedCallback;
        break;

      case HAL_TIM_PWM_PULSE_FINISHED_CB_ID :
        /* Legacy weak PWM Pulse Finished Callback */
        htim->PWM_PulseFinishedCallback         = HAL_TIM_PWM_PulseFinishedCallback;
        break;

      case HAL_TIM_PWM_PULSE_FINISHED_HALF_CB_ID :
        /* Legacy weak PWM Pulse Finished half complete Callback */
        htim->PWM_PulseFinishedHalfCpltCallback = HAL_TIM_PWM_PulseFinishedHalfCpltCallback;
        break;

      case HAL_TIM_ERROR_CB_ID :
        /* Legacy weak Error Callback */
        htim->ErrorCallback                     = HAL_TIM_ErrorCallback;
        break;

      case HAL_TIM_COMMUTATION_CB_ID :
        /* Legacy weak Commutation Callback */
        htim->CommutationCallback               = HAL_TIMEx_CommutCallback;
        break;

      case HAL_TIM_COMMUTATION_HALF_CB_ID :
        /* Legacy weak Commutation half complete Callback */
        htim->CommutationHalfCpltCallback       = HAL_TIMEx_CommutHalfCpltCallback;
        break;

      case HAL_TIM_BREAK_CB_ID :
        /* Legacy weak Break Callback */
        htim->BreakCallback                     = HAL_TIMEx_BreakCallback;
        break;

      default :
        /* Return error status */
        status = HAL_ERROR;
        break;
    }
  }
  else if (htim->State == HAL_TIM_STATE_RESET)
  {
    switch (CallbackID)
    {
      case HAL_TIM_BASE_MSPINIT_CB_ID :
        /* Legacy weak Base MspInit Callback */
        htim->Base_MspInitCallback         = HAL_TIM_Base_MspInit;
        break;

      case HAL_TIM_BASE_MSPDEINIT_CB_ID :
        /* Legacy weak Base Msp DeInit Callback */
        htim->Base_MspDeInitCallback       = HAL_TIM_Base_MspDeInit;
        break;

      case HAL_TIM_IC_MSPINIT_CB_ID :
        /* Legacy weak IC Msp Init Callback */
        htim->IC_MspInitCallback           = HAL_TIM_IC_MspInit;
        break;

      case HAL_TIM_IC_MSPDEINIT_CB_ID :
        /* Legacy weak IC Msp DeInit Callback */
        htim->IC_MspDeInitCallback         = HAL_TIM_IC_MspDeInit;
        break;

      case HAL_TIM_OC_MSPINIT_CB_ID :
        /* Legacy weak OC Msp Init Callback */
        htim->OC_MspInitCallback           = HAL_TIM_OC_MspInit;
        break;

      case HAL_TIM_OC_MSPDEINIT_CB_ID :
        /* Legacy weak OC Msp DeInit Callback */
        htim->OC_MspDeInitCallback         = HAL_TIM_OC_MspDeInit;
        break;

      case HAL_TIM_PWM_MSPINIT_CB_ID :
        /* Legacy weak PWM Msp Init Callback */
        htim->PWM_MspInitCallback          = HAL_TIM_PWM_MspInit;
        break;

      case HAL_TIM_PWM_MSPDEINIT_CB_ID :
        /* Legacy weak PWM Msp DeInit Callback */
        htim->PWM_MspDeInitCallback        = HAL_TIM_PWM_MspDeInit;
        break;

      case HAL_TIM_ONE_PULSE_MSPINIT_CB_ID :
        /* Legacy weak One Pulse Msp Init Callback */
        htim->OnePulse_MspInitCallback     = HAL_TIM_OnePulse_MspInit;
        break;

      case HAL_TIM_ONE_PULSE_MSPDEINIT_CB_ID :
        /* Legacy weak One Pulse Msp DeInit Callback */
        htim->OnePulse_MspDeInitCallback   = HAL_TIM_OnePulse_MspDeInit;
        break;

      case HAL_TIM_ENCODER_MSPINIT_CB_ID :
        /* Legacy weak Encoder Msp Init Callback */
        htim->Encoder_MspInitCallback      = HAL_TIM_Encoder_MspInit;
        break;

      case HAL_TIM_ENCODER_MSPDEINIT_CB_ID :
        /* Legacy weak Encoder Msp DeInit Callback */
        htim->Encoder_MspDeInitCallback    = HAL_TIM_Encoder_MspDeInit;
        break;

      case HAL_TIM_HALL_SENSOR_MSPINIT_CB_ID :
        /* Legacy weak Hall Sensor Msp Init Callback */
        htim->HallSensor_MspInitCallback   = HAL_TIMEx_HallSensor_MspInit;
        break;

      case HAL_TIM_HALL_SENSOR_MSPDEINIT_CB_ID :
        /* Legacy weak Hall Sensor Msp DeInit Callback */
        htim->HallSensor_MspDeInitCallback = HAL_TIMEx_HallSensor_MspDeInit;
        break;

      default :
        /* Return error status */
        status = HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Return error status */
    status = HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(htim);

  return status;
}
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup TIM_Exported_Functions_Group10 TIM Peripheral State functions
  *  @brief   TIM Peripheral State functions
  *
@verbatim
  ==============================================================================
                        ##### Peripheral State functions #####
  ==============================================================================
    [..]
    This subsection permits to get in run-time the status of the peripheral
    and the data flow.

@endverbatim
  * @{
  */

/**
  * @brief  Return the TIM Base handle state.
  * @param  htim TIM Base handle
  * @retval HAL state
  */
HAL_TIM_StateTypeDef HAL_TIM_Base_GetState(TIM_HandleTypeDef *htim)
{
  return htim->State;
}

/**
  * @brief  Return the TIM OC handle state.
  * @param  htim TIM Output Compare handle
  * @retval HAL state
  */
HAL_TIM_StateTypeDef HAL_TIM_OC_GetState(TIM_HandleTypeDef *htim)
{
  return htim->State;
}

/**
  * @brief  Return the TIM PWM handle state.
  * @param  htim TIM handle
  * @retval HAL state
  */
HAL_TIM_StateTypeDef HAL_TIM_PWM_GetState(TIM_HandleTypeDef *htim)
{
  return htim->State;
}

/**
  * @brief  Return the TIM Input Capture handle state.
  * @param  htim TIM IC handle
  * @retval HAL state
  */
HAL_TIM_StateTypeDef HAL_TIM_IC_GetState(TIM_HandleTypeDef *htim)
{
  return htim->State;
}

/**
  * @brief  Return the TIM One Pulse Mode handle state.
  * @param  htim TIM OPM handle
  * @retval HAL state
  */
HAL_TIM_StateTypeDef HAL_TIM_OnePulse_GetState(TIM_HandleTypeDef *htim)
{
  return htim->State;
}

/**
  * @brief  Return the TIM Encoder Mode handle state.
  * @param  htim TIM Encoder Interface handle
  * @retval HAL state
  */
HAL_TIM_StateTypeDef HAL_TIM_Encoder_GetState(TIM_HandleTypeDef *htim)
{
  return htim->State;
}

/**
  * @brief  Return the TIM Encoder Mode handle state.
  * @param  htim TIM handle
  * @retval Active channel
  */
HAL_TIM_ActiveChannel HAL_TIM_GetActiveChannel(TIM_HandleTypeDef *htim)
{
  return htim->Channel;
}

/**
  * @brief  Return actual state of the TIM channel.
  * @param  htim TIM handle
  * @param  Channel TIM Channel
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1
  *            @arg TIM_CHANNEL_2: TIM Channel 2
  *            @arg TIM_CHANNEL_3: TIM Channel 3
  *            @arg TIM_CHANNEL_4: TIM Channel 4
  *            @arg TIM_CHANNEL_5: TIM Channel 5
  *            @arg TIM_CHANNEL_6: TIM Channel 6
  * @retval TIM Channel state
  */
HAL_TIM_ChannelStateTypeDef HAL_TIM_GetChannelState(TIM_HandleTypeDef *htim,  uint32_t Channel)
{
  HAL_TIM_ChannelStateTypeDef channel_state;

  /* Check the parameters */
  assert_param(IS_TIM_CCX_INSTANCE(htim->Instance, Channel));

  channel_state = TIM_CHANNEL_STATE_GET(htim, Channel);

  return channel_state;
}

/**
  * @brief  Return actual state of a DMA burst operation.
  * @param  htim TIM handle
  * @retval DMA burst state
  */
HAL_TIM_DMABurstStateTypeDef HAL_TIM_DMABurstState(TIM_HandleTypeDef *htim)
{
  /* Check the parameters */
  assert_param(IS_TIM_DMABURST_INSTANCE(htim->Instance));

  return htim->DMABurstState;
}

/**
  * @}
  */

/**
  * @}
  */

/** @defgroup TIM_Private_Functions TIM Private Functions
  * @{
  */

/**
  * @brief  TIM DMA error callback
  * @param  hdma pointer to DMA handle.
  * @retval None
  */
void TIM_DMAError(DMA_HandleTypeDef *hdma)
{
  TIM_HandleTypeDef *htim = (TIM_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (hdma == htim->hdma[TIM_DMA_ID_CC1])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_1;
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC2])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_2;
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC3])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_3;
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_3, HAL_TIM_CHANNEL_STATE_READY);
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC4])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_4;
    TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_4, HAL_TIM_CHANNEL_STATE_READY);
  }
  else
  {
    htim->State = HAL_TIM_STATE_READY;
  }

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  htim->ErrorCallback(htim);
#else
  HAL_TIM_ErrorCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  htim->Channel = HAL_TIM_ACTIVE_CHANNEL_CLEARED;
}

/**
  * @brief  TIM DMA Delay Pulse complete callback.
  * @param  hdma pointer to DMA handle.
  * @retval None
  */
static void TIM_DMADelayPulseCplt(DMA_HandleTypeDef *hdma)
{
  TIM_HandleTypeDef *htim = (TIM_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (hdma == htim->hdma[TIM_DMA_ID_CC1])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_1;

    if (hdma->Init.Mode == DMA_NORMAL)
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
    }
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC2])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_2;

    if (hdma->Init.Mode == DMA_NORMAL)
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
    }
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC3])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_3;

    if (hdma->Init.Mode == DMA_NORMAL)
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_3, HAL_TIM_CHANNEL_STATE_READY);
    }
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC4])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_4;

    if (hdma->Init.Mode == DMA_NORMAL)
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_4, HAL_TIM_CHANNEL_STATE_READY);
    }
  }
  else
  {
    /* nothing to do */
  }

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  htim->PWM_PulseFinishedCallback(htim);
#else
  HAL_TIM_PWM_PulseFinishedCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  htim->Channel = HAL_TIM_ACTIVE_CHANNEL_CLEARED;
}

/**
  * @brief  TIM DMA Delay Pulse half complete callback.
  * @param  hdma pointer to DMA handle.
  * @retval None
  */
void TIM_DMADelayPulseHalfCplt(DMA_HandleTypeDef *hdma)
{
  TIM_HandleTypeDef *htim = (TIM_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (hdma == htim->hdma[TIM_DMA_ID_CC1])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_1;
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC2])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_2;
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC3])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_3;
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC4])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_4;
  }
  else
  {
    /* nothing to do */
  }

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  htim->PWM_PulseFinishedHalfCpltCallback(htim);
#else
  HAL_TIM_PWM_PulseFinishedHalfCpltCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  htim->Channel = HAL_TIM_ACTIVE_CHANNEL_CLEARED;
}

/**
  * @brief  TIM DMA Capture complete callback.
  * @param  hdma pointer to DMA handle.
  * @retval None
  */
void TIM_DMACaptureCplt(DMA_HandleTypeDef *hdma)
{
  TIM_HandleTypeDef *htim = (TIM_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (hdma == htim->hdma[TIM_DMA_ID_CC1])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_1;

    if (hdma->Init.Mode == DMA_NORMAL)
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_1, HAL_TIM_CHANNEL_STATE_READY);
    }
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC2])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_2;

    if (hdma->Init.Mode == DMA_NORMAL)
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_2, HAL_TIM_CHANNEL_STATE_READY);
    }
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC3])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_3;

    if (hdma->Init.Mode == DMA_NORMAL)
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_3, HAL_TIM_CHANNEL_STATE_READY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_3, HAL_TIM_CHANNEL_STATE_READY);
    }
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC4])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_4;

    if (hdma->Init.Mode == DMA_NORMAL)
    {
      TIM_CHANNEL_STATE_SET(htim, TIM_CHANNEL_4, HAL_TIM_CHANNEL_STATE_READY);
      TIM_CHANNEL_N_STATE_SET(htim, TIM_CHANNEL_4, HAL_TIM_CHANNEL_STATE_READY);
    }
  }
  else
  {
    /* nothing to do */
  }

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  htim->IC_CaptureCallback(htim);
#else
  HAL_TIM_IC_CaptureCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  htim->Channel = HAL_TIM_ACTIVE_CHANNEL_CLEARED;
}

/**
  * @brief  TIM DMA Capture half complete callback.
  * @param  hdma pointer to DMA handle.
  * @retval None
  */
void TIM_DMACaptureHalfCplt(DMA_HandleTypeDef *hdma)
{
  TIM_HandleTypeDef *htim = (TIM_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (hdma == htim->hdma[TIM_DMA_ID_CC1])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_1;
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC2])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_2;
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC3])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_3;
  }
  else if (hdma == htim->hdma[TIM_DMA_ID_CC4])
  {
    htim->Channel = HAL_TIM_ACTIVE_CHANNEL_4;
  }
  else
  {
    /* nothing to do */
  }

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  htim->IC_CaptureHalfCpltCallback(htim);
#else
  HAL_TIM_IC_CaptureHalfCpltCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

  htim->Channel = HAL_TIM_ACTIVE_CHANNEL_CLEARED;
}

/**
  * @brief  TIM DMA Period Elapse complete callback.
  * @param  hdma pointer to DMA handle.
  * @retval None
  */
static void TIM_DMAPeriodElapsedCplt(DMA_HandleTypeDef *hdma)
{
  TIM_HandleTypeDef *htim = (TIM_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (htim->hdma[TIM_DMA_ID_UPDATE]->Init.Mode == DMA_NORMAL)
  {
    htim->State = HAL_TIM_STATE_READY;
  }

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  htim->PeriodElapsedCallback(htim);
#else
  HAL_TIM_PeriodElapsedCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
}

/**
  * @brief  TIM DMA Period Elapse half complete callback.
  * @param  hdma pointer to DMA handle.
  * @retval None
  */
static void TIM_DMAPeriodElapsedHalfCplt(DMA_HandleTypeDef *hdma)
{
  TIM_HandleTypeDef *htim = (TIM_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  htim->PeriodElapsedHalfCpltCallback(htim);
#else
  HAL_TIM_PeriodElapsedHalfCpltCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
}

/**
  * @brief  TIM DMA Trigger callback.
  * @param  hdma pointer to DMA handle.
  * @retval None
  */
static void TIM_DMATriggerCplt(DMA_HandleTypeDef *hdma)
{
  TIM_HandleTypeDef *htim = (TIM_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (htim->hdma[TIM_DMA_ID_TRIGGER]->Init.Mode == DMA_NORMAL)
  {
    htim->State = HAL_TIM_STATE_READY;
  }

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  htim->TriggerCallback(htim);
#else
  HAL_TIM_TriggerCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
}

/**
  * @brief  TIM DMA Trigger half complete callback.
  * @param  hdma pointer to DMA handle.
  * @retval None
  */
static void TIM_DMATriggerHalfCplt(DMA_HandleTypeDef *hdma)
{
  TIM_HandleTypeDef *htim = (TIM_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
  htim->TriggerHalfCpltCallback(htim);
#else
  HAL_TIM_TriggerHalfCpltCallback(htim);
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */
}

/**
  * @brief  Time Base configuration
  * @param  TIMx TIM peripheral
  * @param  Structure TIM Base configuration structure
  * @retval None
  */
void TIM_Base_SetConfig(TIM_TypeDef *TIMx, TIM_Base_InitTypeDef *Structure)
{
  uint32_t tmpcr1;
  tmpcr1 = TIMx->CR1;

  /* Set TIM Time Base Unit parameters ---------------------------------------*/
  if (IS_TIM_COUNTER_MODE_SELECT_INSTANCE(TIMx))
  {
    /* Select the Counter Mode */
    tmpcr1 &= ~(TIM_CR1_DIR | TIM_CR1_CMS);
    tmpcr1 |= Structure->CounterMode;
  }

  if (IS_TIM_CLOCK_DIVISION_INSTANCE(TIMx))
  {
    /* Set the clock division */
    tmpcr1 &= ~TIM_CR1_CKD;
    tmpcr1 |= (uint32_t)Structure->ClockDivision;
  }

  /* Set the auto-reload preload */
  MODIFY_REG(tmpcr1, TIM_CR1_ARPE, Structure->AutoReloadPreload);

  TIMx->CR1 = tmpcr1;

  /* Set the Autoreload value */
  TIMx->ARR = (uint32_t)Structure->Period ;

  /* Set the Prescaler value */
  TIMx->PSC = Structure->Prescaler;

  if (IS_TIM_REPETITION_COUNTER_INSTANCE(TIMx))
  {
    /* Set the Repetition Counter value */
    TIMx->RCR = Structure->RepetitionCounter;
  }

  /* Generate an update event to reload the Prescaler
     and the repetition counter (only for advanced timer) value immediately */
  TIMx->EGR = TIM_EGR_UG;
}

/**
  * @brief  Timer Output Compare 1 configuration
  * @param  TIMx to select the TIM peripheral
  * @param  OC_Config The output configuration structure
  * @retval None
  */
static void TIM_OC1_SetConfig(TIM_TypeDef *TIMx, TIM_OC_InitTypeDef *OC_Config)
{
  uint32_t tmpccmrx;
  uint32_t tmpccer;
  uint32_t tmpcr2;

  /* Disable the Channel 1: Reset the CC1E Bit */
  TIMx->CCER &= ~TIM_CCER_CC1E;

  /* Get the TIMx CCER register value */
  tmpccer = TIMx->CCER;
  /* Get the TIMx CR2 register value */
  tmpcr2 =  TIMx->CR2;

  /* Get the TIMx CCMR1 register value */
  tmpccmrx = TIMx->CCMR1;

  /* Reset the Output Compare Mode Bits */
  tmpccmrx &= ~TIM_CCMR1_OC1M;
  tmpccmrx &= ~TIM_CCMR1_CC1S;
  /* Select the Output Compare Mode */
  tmpccmrx |= OC_Config->OCMode;

  /* Reset the Output Polarity level */
  tmpccer &= ~TIM_CCER_CC1P;
  /* Set the Output Compare Polarity */
  tmpccer |= OC_Config->OCPolarity;

  if (IS_TIM_CCXN_INSTANCE(TIMx, TIM_CHANNEL_1))
  {
    /* Check parameters */
    assert_param(IS_TIM_OCN_POLARITY(OC_Config->OCNPolarity));

    /* Reset the Output N Polarity level */
    tmpccer &= ~TIM_CCER_CC1NP;
    /* Set the Output N Polarity */
    tmpccer |= OC_Config->OCNPolarity;
    /* Reset the Output N State */
    tmpccer &= ~TIM_CCER_CC1NE;
  }

  if (IS_TIM_BREAK_INSTANCE(TIMx))
  {
    /* Check parameters */
    assert_param(IS_TIM_OCNIDLE_STATE(OC_Config->OCNIdleState));
    assert_param(IS_TIM_OCIDLE_STATE(OC_Config->OCIdleState));

    /* Reset the Output Compare and Output Compare N IDLE State */
    tmpcr2 &= ~TIM_CR2_OIS1;
    tmpcr2 &= ~TIM_CR2_OIS1N;
    /* Set the Output Idle state */
    tmpcr2 |= OC_Config->OCIdleState;
    /* Set the Output N Idle state */
    tmpcr2 |= OC_Config->OCNIdleState;
  }

  /* Write to TIMx CR2 */
  TIMx->CR2 = tmpcr2;

  /* Write to TIMx CCMR1 */
  TIMx->CCMR1 = tmpccmrx;

  /* Set the Capture Compare Register value */
  TIMx->CCR1 = OC_Config->Pulse;

  /* Write to TIMx CCER */
  TIMx->CCER = tmpccer;
}

/**
  * @brief  Timer Output Compare 2 configuration
  * @param  TIMx to select the TIM peripheral
  * @param  OC_Config The output configuration structure
  * @retval None
  */
void TIM_OC2_SetConfig(TIM_TypeDef *TIMx, TIM_OC_InitTypeDef *OC_Config)
{
  uint32_t tmpccmrx;
  uint32_t tmpccer;
  uint32_t tmpcr2;

  /* Disable the Channel 2: Reset the CC2E Bit */
  TIMx->CCER &= ~TIM_CCER_CC2E;

  /* Get the TIMx CCER register value */
  tmpccer = TIMx->CCER;
  /* Get the TIMx CR2 register value */
  tmpcr2 =  TIMx->CR2;

  /* Get the TIMx CCMR1 register value */
  tmpccmrx = TIMx->CCMR1;

  /* Reset the Output Compare mode and Capture/Compare selection Bits */
  tmpccmrx &= ~TIM_CCMR1_OC2M;
  tmpccmrx &= ~TIM_CCMR1_CC2S;

  /* Select the Output Compare Mode */
  tmpccmrx |= (OC_Config->OCMode << 8U);

  /* Reset the Output Polarity level */
  tmpccer &= ~TIM_CCER_CC2P;
  /* Set the Output Compare Polarity */
  tmpccer |= (OC_Config->OCPolarity << 4U);

  if (IS_TIM_CCXN_INSTANCE(TIMx, TIM_CHANNEL_2))
  {
    assert_param(IS_TIM_OCN_POLARITY(OC_Config->OCNPolarity));

    /* Reset the Output N Polarity level */
    tmpccer &= ~TIM_CCER_CC2NP;
    /* Set the Output N Polarity */
    tmpccer |= (OC_Config->OCNPolarity << 4U);
    /* Reset the Output N State */
    tmpccer &= ~TIM_CCER_CC2NE;

  }

  if (IS_TIM_BREAK_INSTANCE(TIMx))
  {
    /* Check parameters */
    assert_param(IS_TIM_OCNIDLE_STATE(OC_Config->OCNIdleState));
    assert_param(IS_TIM_OCIDLE_STATE(OC_Config->OCIdleState));

    /* Reset the Output Compare and Output Compare N IDLE State */
    tmpcr2 &= ~TIM_CR2_OIS2;
    tmpcr2 &= ~TIM_CR2_OIS2N;
    /* Set the Output Idle state */
    tmpcr2 |= (OC_Config->OCIdleState << 2U);
    /* Set the Output N Idle state */
    tmpcr2 |= (OC_Config->OCNIdleState << 2U);
  }

  /* Write to TIMx CR2 */
  TIMx->CR2 = tmpcr2;

  /* Write to TIMx CCMR1 */
  TIMx->CCMR1 = tmpccmrx;

  /* Set the Capture Compare Register value */
  TIMx->CCR2 = OC_Config->Pulse;

  /* Write to TIMx CCER */
  TIMx->CCER = tmpccer;
}

/**
  * @brief  Timer Output Compare 3 configuration
  * @param  TIMx to select the TIM peripheral
  * @param  OC_Config The output configuration structure
  * @retval None
  */
static void TIM_OC3_SetConfig(TIM_TypeDef *TIMx, TIM_OC_InitTypeDef *OC_Config)
{
  uint32_t tmpccmrx;
  uint32_t tmpccer;
  uint32_t tmpcr2;

  /* Disable the Channel 3: Reset the CC2E Bit */
  TIMx->CCER &= ~TIM_CCER_CC3E;

  /* Get the TIMx CCER register value */
  tmpccer = TIMx->CCER;
  /* Get the TIMx CR2 register value */
  tmpcr2 =  TIMx->CR2;

  /* Get the TIMx CCMR2 register value */
  tmpccmrx = TIMx->CCMR2;

  /* Reset the Output Compare mode and Capture/Compare selection Bits */
  tmpccmrx &= ~TIM_CCMR2_OC3M;
  tmpccmrx &= ~TIM_CCMR2_CC3S;
  /* Select the Output Compare Mode */
  tmpccmrx |= OC_Config->OCMode;

  /* Reset the Output Polarity level */
  tmpccer &= ~TIM_CCER_CC3P;
  /* Set the Output Compare Polarity */
  tmpccer |= (OC_Config->OCPolarity << 8U);

  if (IS_TIM_CCXN_INSTANCE(TIMx, TIM_CHANNEL_3))
  {
    assert_param(IS_TIM_OCN_POLARITY(OC_Config->OCNPolarity));

    /* Reset the Output N Polarity level */
    tmpccer &= ~TIM_CCER_CC3NP;
    /* Set the Output N Polarity */
    tmpccer |= (OC_Config->OCNPolarity << 8U);
    /* Reset the Output N State */
    tmpccer &= ~TIM_CCER_CC3NE;
  }

  if (IS_TIM_BREAK_INSTANCE(TIMx))
  {
    /* Check parameters */
    assert_param(IS_TIM_OCNIDLE_STATE(OC_Config->OCNIdleState));
    assert_param(IS_TIM_OCIDLE_STATE(OC_Config->OCIdleState));

    /* Reset the Output Compare and Output Compare N IDLE State */
    tmpcr2 &= ~TIM_CR2_OIS3;
    tmpcr2 &= ~TIM_CR2_OIS3N;
    /* Set the Output Idle state */
    tmpcr2 |= (OC_Config->OCIdleState << 4U);
    /* Set the Output N Idle state */
    tmpcr2 |= (OC_Config->OCNIdleState << 4U);
  }

  /* Write to TIMx CR2 */
  TIMx->CR2 = tmpcr2;

  /* Write to TIMx CCMR2 */
  TIMx->CCMR2 = tmpccmrx;

  /* Set the Capture Compare Register value */
  TIMx->CCR3 = OC_Config->Pulse;

  /* Write to TIMx CCER */
  TIMx->CCER = tmpccer;
}

/**
  * @brief  Timer Output Compare 4 configuration
  * @param  TIMx to select the TIM peripheral
  * @param  OC_Config The output configuration structure
  * @retval None
  */
static void TIM_OC4_SetConfig(TIM_TypeDef *TIMx, TIM_OC_InitTypeDef *OC_Config)
{
  uint32_t tmpccmrx;
  uint32_t tmpccer;
  uint32_t tmpcr2;

  /* Disable the Channel 4: Reset the CC4E Bit */
  TIMx->CCER &= ~TIM_CCER_CC4E;

  /* Get the TIMx CCER register value */
  tmpccer = TIMx->CCER;
  /* Get the TIMx CR2 register value */
  tmpcr2 =  TIMx->CR2;

  /* Get the TIMx CCMR2 register value */
  tmpccmrx = TIMx->CCMR2;

  /* Reset the Output Compare mode and Capture/Compare selection Bits */
  tmpccmrx &= ~TIM_CCMR2_OC4M;
  tmpccmrx &= ~TIM_CCMR2_CC4S;

  /* Select the Output Compare Mode */
  tmpccmrx |= (OC_Config->OCMode << 8U);

  /* Reset the Output Polarity level */
  tmpccer &= ~TIM_CCER_CC4P;
  /* Set the Output Compare Polarity */
  tmpccer |= (OC_Config->OCPolarity << 12U);

  if (IS_TIM_BREAK_INSTANCE(TIMx))
  {
    /* Check parameters */
    assert_param(IS_TIM_OCIDLE_STATE(OC_Config->OCIdleState));

    /* Reset the Output Compare IDLE State */
    tmpcr2 &= ~TIM_CR2_OIS4;

    /* Set the Output Idle state */
    tmpcr2 |= (OC_Config->OCIdleState << 6U);
  }

  /* Write to TIMx CR2 */
  TIMx->CR2 = tmpcr2;

  /* Write to TIMx CCMR2 */
  TIMx->CCMR2 = tmpccmrx;

  /* Set the Capture Compare Register value */
  TIMx->CCR4 = OC_Config->Pulse;

  /* Write to TIMx CCER */
  TIMx->CCER = tmpccer;
}

/**
  * @brief  Slave Timer configuration function
  * @param  htim TIM handle
  * @param  sSlaveConfig Slave timer configuration
  * @retval None
  */
static HAL_StatusTypeDef TIM_SlaveTimer_SetConfig(TIM_HandleTypeDef *htim,
                                                  TIM_SlaveConfigTypeDef *sSlaveConfig)
{
  HAL_StatusTypeDef status = HAL_OK;
  uint32_t tmpsmcr;
  uint32_t tmpccmr1;
  uint32_t tmpccer;

  /* Get the TIMx SMCR register value */
  tmpsmcr = htim->Instance->SMCR;

  /* Reset the Trigger Selection Bits */
  tmpsmcr &= ~TIM_SMCR_TS;
  /* Set the Input Trigger source */
  tmpsmcr |= sSlaveConfig->InputTrigger;

  /* Reset the slave mode Bits */
  tmpsmcr &= ~TIM_SMCR_SMS;
  /* Set the slave mode */
  tmpsmcr |= sSlaveConfig->SlaveMode;

  /* Write to TIMx SMCR */
  htim->Instance->SMCR = tmpsmcr;

  /* Configure the trigger prescaler, filter, and polarity */
  switch (sSlaveConfig->InputTrigger)
  {
    case TIM_TS_ETRF:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CLOCKSOURCE_ETRMODE1_INSTANCE(htim->Instance));
      assert_param(IS_TIM_TRIGGERPRESCALER(sSlaveConfig->TriggerPrescaler));
      assert_param(IS_TIM_TRIGGERPOLARITY(sSlaveConfig->TriggerPolarity));
      assert_param(IS_TIM_TRIGGERFILTER(sSlaveConfig->TriggerFilter));
      /* Configure the ETR Trigger source */
      TIM_ETR_SetConfig(htim->Instance,
                        sSlaveConfig->TriggerPrescaler,
                        sSlaveConfig->TriggerPolarity,
                        sSlaveConfig->TriggerFilter);
      break;
    }

    case TIM_TS_TI1F_ED:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC1_INSTANCE(htim->Instance));
      assert_param(IS_TIM_TRIGGERFILTER(sSlaveConfig->TriggerFilter));

      if (sSlaveConfig->SlaveMode == TIM_SLAVEMODE_GATED)
      {
        return HAL_ERROR;
      }

      /* Disable the Channel 1: Reset the CC1E Bit */
      tmpccer = htim->Instance->CCER;
      htim->Instance->CCER &= ~TIM_CCER_CC1E;
      tmpccmr1 = htim->Instance->CCMR1;

      /* Set the filter */
      tmpccmr1 &= ~TIM_CCMR1_IC1F;
      tmpccmr1 |= ((sSlaveConfig->TriggerFilter) << 4U);

      /* Write to TIMx CCMR1 and CCER registers */
      htim->Instance->CCMR1 = tmpccmr1;
      htim->Instance->CCER = tmpccer;
      break;
    }

    case TIM_TS_TI1FP1:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC1_INSTANCE(htim->Instance));
      assert_param(IS_TIM_TRIGGERPOLARITY(sSlaveConfig->TriggerPolarity));
      assert_param(IS_TIM_TRIGGERFILTER(sSlaveConfig->TriggerFilter));

      /* Configure TI1 Filter and Polarity */
      TIM_TI1_ConfigInputStage(htim->Instance,
                               sSlaveConfig->TriggerPolarity,
                               sSlaveConfig->TriggerFilter);
      break;
    }

    case TIM_TS_TI2FP2:
    {
      /* Check the parameters */
      assert_param(IS_TIM_CC2_INSTANCE(htim->Instance));
      assert_param(IS_TIM_TRIGGERPOLARITY(sSlaveConfig->TriggerPolarity));
      assert_param(IS_TIM_TRIGGERFILTER(sSlaveConfig->TriggerFilter));

      /* Configure TI2 Filter and Polarity */
      TIM_TI2_ConfigInputStage(htim->Instance,
                               sSlaveConfig->TriggerPolarity,
                               sSlaveConfig->TriggerFilter);
      break;
    }

    case TIM_TS_ITR0:
    case TIM_TS_ITR1:
    case TIM_TS_ITR2:
    case TIM_TS_ITR3:
    {
      /* Check the parameter */
      assert_param(IS_TIM_CC2_INSTANCE(htim->Instance));
      break;
    }

    default:
      status = HAL_ERROR;
      break;
  }

  return status;
}

/**
  * @brief  Configure the TI1 as Input.
  * @param  TIMx to select the TIM peripheral.
  * @param  TIM_ICPolarity The Input Polarity.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICPOLARITY_RISING
  *            @arg TIM_ICPOLARITY_FALLING
  *            @arg TIM_ICPOLARITY_BOTHEDGE
  * @param  TIM_ICSelection specifies the input to be used.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICSELECTION_DIRECTTI: TIM Input 1 is selected to be connected to IC1.
  *            @arg TIM_ICSELECTION_INDIRECTTI: TIM Input 1 is selected to be connected to IC2.
  *            @arg TIM_ICSELECTION_TRC: TIM Input 1 is selected to be connected to TRC.
  * @param  TIM_ICFilter Specifies the Input Capture Filter.
  *          This parameter must be a value between 0x00 and 0x0F.
  * @retval None
  * @note TIM_ICFilter and TIM_ICPolarity are not used in INDIRECT mode as TI2FP1
  *       (on channel2 path) is used as the input signal. Therefore CCMR1 must be
  *        protected against un-initialized filter and polarity values.
  */
void TIM_TI1_SetConfig(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICSelection,
                       uint32_t TIM_ICFilter)
{
  uint32_t tmpccmr1;
  uint32_t tmpccer;

  /* Disable the Channel 1: Reset the CC1E Bit */
  TIMx->CCER &= ~TIM_CCER_CC1E;
  tmpccmr1 = TIMx->CCMR1;
  tmpccer = TIMx->CCER;

  /* Select the Input */
  if (IS_TIM_CC2_INSTANCE(TIMx) != RESET)
  {
    tmpccmr1 &= ~TIM_CCMR1_CC1S;
    tmpccmr1 |= TIM_ICSelection;
  }
  else
  {
    tmpccmr1 |= TIM_CCMR1_CC1S_0;
  }

  /* Set the filter */
  tmpccmr1 &= ~TIM_CCMR1_IC1F;
  tmpccmr1 |= ((TIM_ICFilter << 4U) & TIM_CCMR1_IC1F);

  /* Select the Polarity and set the CC1E Bit */
  tmpccer &= ~(TIM_CCER_CC1P | TIM_CCER_CC1NP);
  tmpccer |= (TIM_ICPolarity & (TIM_CCER_CC1P | TIM_CCER_CC1NP));

  /* Write to TIMx CCMR1 and CCER registers */
  TIMx->CCMR1 = tmpccmr1;
  TIMx->CCER = tmpccer;
}

/**
  * @brief  Configure the Polarity and Filter for TI1.
  * @param  TIMx to select the TIM peripheral.
  * @param  TIM_ICPolarity The Input Polarity.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICPOLARITY_RISING
  *            @arg TIM_ICPOLARITY_FALLING
  *            @arg TIM_ICPOLARITY_BOTHEDGE
  * @param  TIM_ICFilter Specifies the Input Capture Filter.
  *          This parameter must be a value between 0x00 and 0x0F.
  * @retval None
  */
static void TIM_TI1_ConfigInputStage(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICFilter)
{
  uint32_t tmpccmr1;
  uint32_t tmpccer;

  /* Disable the Channel 1: Reset the CC1E Bit */
  tmpccer = TIMx->CCER;
  TIMx->CCER &= ~TIM_CCER_CC1E;
  tmpccmr1 = TIMx->CCMR1;

  /* Set the filter */
  tmpccmr1 &= ~TIM_CCMR1_IC1F;
  tmpccmr1 |= (TIM_ICFilter << 4U);

  /* Select the Polarity and set the CC1E Bit */
  tmpccer &= ~(TIM_CCER_CC1P | TIM_CCER_CC1NP);
  tmpccer |= TIM_ICPolarity;

  /* Write to TIMx CCMR1 and CCER registers */
  TIMx->CCMR1 = tmpccmr1;
  TIMx->CCER = tmpccer;
}

/**
  * @brief  Configure the TI2 as Input.
  * @param  TIMx to select the TIM peripheral
  * @param  TIM_ICPolarity The Input Polarity.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICPOLARITY_RISING
  *            @arg TIM_ICPOLARITY_FALLING
  *            @arg TIM_ICPOLARITY_BOTHEDGE
  * @param  TIM_ICSelection specifies the input to be used.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICSELECTION_DIRECTTI: TIM Input 2 is selected to be connected to IC2.
  *            @arg TIM_ICSELECTION_INDIRECTTI: TIM Input 2 is selected to be connected to IC1.
  *            @arg TIM_ICSELECTION_TRC: TIM Input 2 is selected to be connected to TRC.
  * @param  TIM_ICFilter Specifies the Input Capture Filter.
  *          This parameter must be a value between 0x00 and 0x0F.
  * @retval None
  * @note TIM_ICFilter and TIM_ICPolarity are not used in INDIRECT mode as TI1FP2
  *       (on channel1 path) is used as the input signal. Therefore CCMR1 must be
  *        protected against un-initialized filter and polarity values.
  */
static void TIM_TI2_SetConfig(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICSelection,
                              uint32_t TIM_ICFilter)
{
  uint32_t tmpccmr1;
  uint32_t tmpccer;

  /* Disable the Channel 2: Reset the CC2E Bit */
  TIMx->CCER &= ~TIM_CCER_CC2E;
  tmpccmr1 = TIMx->CCMR1;
  tmpccer = TIMx->CCER;

  /* Select the Input */
  tmpccmr1 &= ~TIM_CCMR1_CC2S;
  tmpccmr1 |= (TIM_ICSelection << 8U);

  /* Set the filter */
  tmpccmr1 &= ~TIM_CCMR1_IC2F;
  tmpccmr1 |= ((TIM_ICFilter << 12U) & TIM_CCMR1_IC2F);

  /* Select the Polarity and set the CC2E Bit */
  tmpccer &= ~(TIM_CCER_CC2P | TIM_CCER_CC2NP);
  tmpccer |= ((TIM_ICPolarity << 4U) & (TIM_CCER_CC2P | TIM_CCER_CC2NP));

  /* Write to TIMx CCMR1 and CCER registers */
  TIMx->CCMR1 = tmpccmr1 ;
  TIMx->CCER = tmpccer;
}

/**
  * @brief  Configure the Polarity and Filter for TI2.
  * @param  TIMx to select the TIM peripheral.
  * @param  TIM_ICPolarity The Input Polarity.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICPOLARITY_RISING
  *            @arg TIM_ICPOLARITY_FALLING
  *            @arg TIM_ICPOLARITY_BOTHEDGE
  * @param  TIM_ICFilter Specifies the Input Capture Filter.
  *          This parameter must be a value between 0x00 and 0x0F.
  * @retval None
  */
static void TIM_TI2_ConfigInputStage(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICFilter)
{
  uint32_t tmpccmr1;
  uint32_t tmpccer;

  /* Disable the Channel 2: Reset the CC2E Bit */
  TIMx->CCER &= ~TIM_CCER_CC2E;
  tmpccmr1 = TIMx->CCMR1;
  tmpccer = TIMx->CCER;

  /* Set the filter */
  tmpccmr1 &= ~TIM_CCMR1_IC2F;
  tmpccmr1 |= (TIM_ICFilter << 12U);

  /* Select the Polarity and set the CC2E Bit */
  tmpccer &= ~(TIM_CCER_CC2P | TIM_CCER_CC2NP);
  tmpccer |= (TIM_ICPolarity << 4U);

  /* Write to TIMx CCMR1 and CCER registers */
  TIMx->CCMR1 = tmpccmr1 ;
  TIMx->CCER = tmpccer;
}

/**
  * @brief  Configure the TI3 as Input.
  * @param  TIMx to select the TIM peripheral
  * @param  TIM_ICPolarity The Input Polarity.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICPOLARITY_RISING
  *            @arg TIM_ICPOLARITY_FALLING
  *            @arg TIM_ICPOLARITY_BOTHEDGE
  * @param  TIM_ICSelection specifies the input to be used.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICSELECTION_DIRECTTI: TIM Input 3 is selected to be connected to IC3.
  *            @arg TIM_ICSELECTION_INDIRECTTI: TIM Input 3 is selected to be connected to IC4.
  *            @arg TIM_ICSELECTION_TRC: TIM Input 3 is selected to be connected to TRC.
  * @param  TIM_ICFilter Specifies the Input Capture Filter.
  *          This parameter must be a value between 0x00 and 0x0F.
  * @retval None
  * @note TIM_ICFilter and TIM_ICPolarity are not used in INDIRECT mode as TI3FP4
  *       (on channel1 path) is used as the input signal. Therefore CCMR2 must be
  *        protected against un-initialized filter and polarity values.
  */
static void TIM_TI3_SetConfig(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICSelection,
                              uint32_t TIM_ICFilter)
{
  uint32_t tmpccmr2;
  uint32_t tmpccer;

  /* Disable the Channel 3: Reset the CC3E Bit */
  TIMx->CCER &= ~TIM_CCER_CC3E;
  tmpccmr2 = TIMx->CCMR2;
  tmpccer = TIMx->CCER;

  /* Select the Input */
  tmpccmr2 &= ~TIM_CCMR2_CC3S;
  tmpccmr2 |= TIM_ICSelection;

  /* Set the filter */
  tmpccmr2 &= ~TIM_CCMR2_IC3F;
  tmpccmr2 |= ((TIM_ICFilter << 4U) & TIM_CCMR2_IC3F);

  /* Select the Polarity and set the CC3E Bit */
  tmpccer &= ~(TIM_CCER_CC3P | TIM_CCER_CC3NP);
  tmpccer |= ((TIM_ICPolarity << 8U) & (TIM_CCER_CC3P | TIM_CCER_CC3NP));

  /* Write to TIMx CCMR2 and CCER registers */
  TIMx->CCMR2 = tmpccmr2;
  TIMx->CCER = tmpccer;
}

/**
  * @brief  Configure the TI4 as Input.
  * @param  TIMx to select the TIM peripheral
  * @param  TIM_ICPolarity The Input Polarity.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICPOLARITY_RISING
  *            @arg TIM_ICPOLARITY_FALLING
  *            @arg TIM_ICPOLARITY_BOTHEDGE
  * @param  TIM_ICSelection specifies the input to be used.
  *          This parameter can be one of the following values:
  *            @arg TIM_ICSELECTION_DIRECTTI: TIM Input 4 is selected to be connected to IC4.
  *            @arg TIM_ICSELECTION_INDIRECTTI: TIM Input 4 is selected to be connected to IC3.
  *            @arg TIM_ICSELECTION_TRC: TIM Input 4 is selected to be connected to TRC.
  * @param  TIM_ICFilter Specifies the Input Capture Filter.
  *          This parameter must be a value between 0x00 and 0x0F.
  * @note TIM_ICFilter and TIM_ICPolarity are not used in INDIRECT mode as TI4FP3
  *       (on channel1 path) is used as the input signal. Therefore CCMR2 must be
  *        protected against un-initialized filter and polarity values.
  * @retval None
  */
static void TIM_TI4_SetConfig(TIM_TypeDef *TIMx, uint32_t TIM_ICPolarity, uint32_t TIM_ICSelection,
                              uint32_t TIM_ICFilter)
{
  uint32_t tmpccmr2;
  uint32_t tmpccer;

  /* Disable the Channel 4: Reset the CC4E Bit */
  TIMx->CCER &= ~TIM_CCER_CC4E;
  tmpccmr2 = TIMx->CCMR2;
  tmpccer = TIMx->CCER;

  /* Select the Input */
  tmpccmr2 &= ~TIM_CCMR2_CC4S;
  tmpccmr2 |= (TIM_ICSelection << 8U);

  /* Set the filter */
  tmpccmr2 &= ~TIM_CCMR2_IC4F;
  tmpccmr2 |= ((TIM_ICFilter << 12U) & TIM_CCMR2_IC4F);

  /* Select the Polarity and set the CC4E Bit */
  tmpccer &= ~(TIM_CCER_CC4P | TIM_CCER_CC4NP);
  tmpccer |= ((TIM_ICPolarity << 12U) & (TIM_CCER_CC4P | TIM_CCER_CC4NP));

  /* Write to TIMx CCMR2 and CCER registers */
  TIMx->CCMR2 = tmpccmr2;
  TIMx->CCER = tmpccer ;
}

/**
  * @brief  Selects the Input Trigger source
  * @param  TIMx to select the TIM peripheral
  * @param  InputTriggerSource The Input Trigger source.
  *          This parameter can be one of the following values:
  *            @arg TIM_TS_ITR0: Internal Trigger 0
  *            @arg TIM_TS_ITR1: Internal Trigger 1
  *            @arg TIM_TS_ITR2: Internal Trigger 2
  *            @arg TIM_TS_ITR3: Internal Trigger 3
  *            @arg TIM_TS_TI1F_ED: TI1 Edge Detector
  *            @arg TIM_TS_TI1FP1: Filtered Timer Input 1
  *            @arg TIM_TS_TI2FP2: Filtered Timer Input 2
  *            @arg TIM_TS_ETRF: External Trigger input
  * @retval None
  */
static void TIM_ITRx_SetConfig(TIM_TypeDef *TIMx, uint32_t InputTriggerSource)
{
  uint32_t tmpsmcr;

  /* Get the TIMx SMCR register value */
  tmpsmcr = TIMx->SMCR;
  /* Reset the TS Bits */
  tmpsmcr &= ~TIM_SMCR_TS;
  /* Set the Input Trigger source and the slave mode*/
  tmpsmcr |= (InputTriggerSource | TIM_SLAVEMODE_EXTERNAL1);
  /* Write to TIMx SMCR */
  TIMx->SMCR = tmpsmcr;
}
/**
  * @brief  Configures the TIMx External Trigger (ETR).
  * @param  TIMx to select the TIM peripheral
  * @param  TIM_ExtTRGPrescaler The external Trigger Prescaler.
  *          This parameter can be one of the following values:
  *            @arg TIM_ETRPRESCALER_DIV1: ETRP Prescaler OFF.
  *            @arg TIM_ETRPRESCALER_DIV2: ETRP frequency divided by 2.
  *            @arg TIM_ETRPRESCALER_DIV4: ETRP frequency divided by 4.
  *            @arg TIM_ETRPRESCALER_DIV8: ETRP frequency divided by 8.
  * @param  TIM_ExtTRGPolarity The external Trigger Polarity.
  *          This parameter can be one of the following values:
  *            @arg TIM_ETRPOLARITY_INVERTED: active low or falling edge active.
  *            @arg TIM_ETRPOLARITY_NONINVERTED: active high or rising edge active.
  * @param  ExtTRGFilter External Trigger Filter.
  *          This parameter must be a value between 0x00 and 0x0F
  * @retval None
  */
void TIM_ETR_SetConfig(TIM_TypeDef *TIMx, uint32_t TIM_ExtTRGPrescaler,
                       uint32_t TIM_ExtTRGPolarity, uint32_t ExtTRGFilter)
{
  uint32_t tmpsmcr;

  tmpsmcr = TIMx->SMCR;

  /* Reset the ETR Bits */
  tmpsmcr &= ~(TIM_SMCR_ETF | TIM_SMCR_ETPS | TIM_SMCR_ECE | TIM_SMCR_ETP);

  /* Set the Prescaler, the Filter value and the Polarity */
  tmpsmcr |= (uint32_t)(TIM_ExtTRGPrescaler | (TIM_ExtTRGPolarity | (ExtTRGFilter << 8U)));

  /* Write to TIMx SMCR */
  TIMx->SMCR = tmpsmcr;
}

/**
  * @brief  Enables or disables the TIM Capture Compare Channel x.
  * @param  TIMx to select the TIM peripheral
  * @param  Channel specifies the TIM Channel
  *          This parameter can be one of the following values:
  *            @arg TIM_CHANNEL_1: TIM Channel 1
  *            @arg TIM_CHANNEL_2: TIM Channel 2
  *            @arg TIM_CHANNEL_3: TIM Channel 3
  *            @arg TIM_CHANNEL_4: TIM Channel 4
  * @param  ChannelState specifies the TIM Channel CCxE bit new state.
  *          This parameter can be: TIM_CCx_ENABLE or TIM_CCx_DISABLE.
  * @retval None
  */
void TIM_CCxChannelCmd(TIM_TypeDef *TIMx, uint32_t Channel, uint32_t ChannelState)
{
  uint32_t tmp;

  /* Check the parameters */
  assert_param(IS_TIM_CC1_INSTANCE(TIMx));
  assert_param(IS_TIM_CHANNELS(Channel));

  tmp = TIM_CCER_CC1E << (Channel & 0x1FU); /* 0x1FU = 31 bits max shift */

  /* Reset the CCxE Bit */
  TIMx->CCER &= ~tmp;

  /* Set or reset the CCxE Bit */
  TIMx->CCER |= (uint32_t)(ChannelState << (Channel & 0x1FU)); /* 0x1FU = 31 bits max shift */
}

#if (USE_HAL_TIM_REGISTER_CALLBACKS == 1)
/**
  * @brief  Reset interrupt callbacks to the legacy weak callbacks.
  * @param  htim pointer to a TIM_HandleTypeDef structure that contains
  *                the configuration information for TIM module.
  * @retval None
  */
void TIM_ResetCallback(TIM_HandleTypeDef *htim)
{
  /* Reset the TIM callback to the legacy weak callbacks */
  htim->PeriodElapsedCallback             = HAL_TIM_PeriodElapsedCallback;
  htim->PeriodElapsedHalfCpltCallback     = HAL_TIM_PeriodElapsedHalfCpltCallback;
  htim->TriggerCallback                   = HAL_TIM_TriggerCallback;
  htim->TriggerHalfCpltCallback           = HAL_TIM_TriggerHalfCpltCallback;
  htim->IC_CaptureCallback                = HAL_TIM_IC_CaptureCallback;
  htim->IC_CaptureHalfCpltCallback        = HAL_TIM_IC_CaptureHalfCpltCallback;
  htim->OC_DelayElapsedCallback           = HAL_TIM_OC_DelayElapsedCallback;
  htim->PWM_PulseFinishedCallback         = HAL_TIM_PWM_PulseFinishedCallback;
  htim->PWM_PulseFinishedHalfCpltCallback = HAL_TIM_PWM_PulseFinishedHalfCpltCallback;
  htim->ErrorCallback                     = HAL_TIM_ErrorCallback;
  htim->CommutationCallback               = HAL_TIMEx_CommutCallback;
  htim->CommutationHalfCpltCallback       = HAL_TIMEx_CommutHalfCpltCallback;
  htim->BreakCallback                     = HAL_TIMEx_BreakCallback;
}
#endif /* USE_HAL_TIM_REGISTER_CALLBACKS */

/**
  * @}
  */

#endif /* HAL_TIM_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */
/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
