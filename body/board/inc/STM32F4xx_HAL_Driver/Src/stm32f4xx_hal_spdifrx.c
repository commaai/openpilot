/**
  ******************************************************************************
  * @file    stm32f4xx_hal_spdifrx.c
  * @author  MCD Application Team
  * @brief   This file provides firmware functions to manage the following
  *          functionalities of the SPDIFRX audio interface:
  *           + Initialization and Configuration
  *           + Data transfers functions
  *           + DMA transfers management
  *           + Interrupts and flags management
  @verbatim
 ===============================================================================
                  ##### How to use this driver #####
 ===============================================================================
 [..]
    The SPDIFRX HAL driver can be used as follow:

    (#) Declare SPDIFRX_HandleTypeDef handle structure.
    (#) Initialize the SPDIFRX low level resources by implement the HAL_SPDIFRX_MspInit() API:
        (##) Enable the SPDIFRX interface clock.
        (##) SPDIFRX pins configuration:
            (+++) Enable the clock for the SPDIFRX GPIOs.
            (+++) Configure these SPDIFRX pins as alternate function pull-up.
        (##) NVIC configuration if you need to use interrupt process (HAL_SPDIFRX_ReceiveControlFlow_IT() and HAL_SPDIFRX_ReceiveDataFlow_IT() API's).
            (+++) Configure the SPDIFRX interrupt priority.
            (+++) Enable the NVIC SPDIFRX IRQ handle.
        (##) DMA Configuration if you need to use DMA process (HAL_SPDIFRX_ReceiveDataFlow_DMA() and HAL_SPDIFRX_ReceiveControlFlow_DMA() API's).
            (+++) Declare a DMA handle structure for the reception of the Data Flow channel.
            (+++) Declare a DMA handle structure for the reception of the Control Flow channel.
            (+++) Enable the DMAx interface clock.
            (+++) Configure the declared DMA handle structure CtrlRx/DataRx with the required parameters.
            (+++) Configure the DMA Channel.
            (+++) Associate the initialized DMA handle to the SPDIFRX DMA CtrlRx/DataRx handle.
            (+++) Configure the priority and enable the NVIC for the transfer complete interrupt on the
                DMA CtrlRx/DataRx channel.

   (#) Program the input selection, re-tries number, wait for activity, channel status selection, data format, stereo mode and masking of user bits
       using HAL_SPDIFRX_Init() function.

   -@- The specific SPDIFRX interrupts (RXNE/CSRNE and Error Interrupts) will be managed using the macros
       __SPDIFRX_ENABLE_IT() and __SPDIFRX_DISABLE_IT() inside the receive process.
   -@- Make sure that ck_spdif clock is configured.

   (#) Three operation modes are available within this driver :

   *** Polling mode for reception operation (for debug purpose) ***
   ================================================================
   [..]
     (+) Receive data flow in blocking mode using HAL_SPDIFRX_ReceiveDataFlow()
     (+) Receive control flow of data in blocking mode using HAL_SPDIFRX_ReceiveControlFlow()

   *** Interrupt mode for reception operation ***
   =========================================
   [..]
     (+) Receive an amount of data (Data Flow) in non blocking mode using HAL_SPDIFRX_ReceiveDataFlow_IT()
     (+) Receive an amount of data (Control Flow) in non blocking mode using HAL_SPDIFRX_ReceiveControlFlow_IT()
     (+) At reception end of half transfer HAL_SPDIFRX_RxHalfCpltCallback is executed and user can
         add his own code by customization of function pointer HAL_SPDIFRX_RxHalfCpltCallback
     (+) At reception end of transfer HAL_SPDIFRX_RxCpltCallback is executed and user can
         add his own code by customization of function pointer HAL_SPDIFRX_RxCpltCallback
     (+) In case of transfer Error, HAL_SPDIFRX_ErrorCallback() function is executed and user can
         add his own code by customization of function pointer HAL_SPDIFRX_ErrorCallback

   *** DMA mode for reception operation ***
   ========================================
   [..]
     (+) Receive an amount of data (Data Flow) in non blocking mode (DMA) using HAL_SPDIFRX_ReceiveDataFlow_DMA()
     (+) Receive an amount of data (Control Flow) in non blocking mode (DMA) using HAL_SPDIFRX_ReceiveControlFlow_DMA()
     (+) At reception end of half transfer HAL_SPDIFRX_RxHalfCpltCallback is executed and user can
         add his own code by customization of function pointer HAL_SPDIFRX_RxHalfCpltCallback
     (+) At reception end of transfer HAL_SPDIFRX_RxCpltCallback is executed and user can
         add his own code by customization of function pointer HAL_SPDIFRX_RxCpltCallback
     (+) In case of transfer Error, HAL_SPDIFRX_ErrorCallback() function is executed and user can
         add his own code by customization of function pointer HAL_SPDIFRX_ErrorCallback
     (+) Stop the DMA Transfer using HAL_SPDIFRX_DMAStop()

   *** SPDIFRX HAL driver macros list ***
   =============================================
   [..]
     Below the list of most used macros in SPDIFRX HAL driver.
      (+) __HAL_SPDIFRX_IDLE: Disable the specified SPDIFRX peripheral (IDEL State)
      (+) __HAL_SPDIFRX_SYNC: Enable the synchronization state of the specified SPDIFRX peripheral (SYNC State)
      (+) __HAL_SPDIFRX_RCV: Enable the receive state of the specified SPDIFRX peripheral (RCV State)
      (+) __HAL_SPDIFRX_ENABLE_IT : Enable the specified SPDIFRX interrupts
      (+) __HAL_SPDIFRX_DISABLE_IT : Disable the specified SPDIFRX interrupts
      (+) __HAL_SPDIFRX_GET_FLAG: Check whether the specified SPDIFRX flag is set or not.

   [..]
      (@) You can refer to the SPDIFRX HAL driver header file for more useful macros

  *** Callback registration ***
  =============================================

  The compilation define  USE_HAL_SPDIFRX_REGISTER_CALLBACKS when set to 1
  allows the user to configure dynamically the driver callbacks.
  Use HAL_SPDIFRX_RegisterCallback() function to register an interrupt callback.

  The HAL_SPDIFRX_RegisterCallback() function allows to register the following callbacks:
    (+) RxHalfCpltCallback  : SPDIFRX Data flow half completed callback.
    (+) RxCpltCallback      : SPDIFRX Data flow completed callback.
    (+) CxHalfCpltCallback  : SPDIFRX Control flow half completed callback.
    (+) CxCpltCallback      : SPDIFRX Control flow completed callback.
    (+) ErrorCallback       : SPDIFRX error callback.
    (+) MspInitCallback     : SPDIFRX MspInit.
    (+) MspDeInitCallback   : SPDIFRX MspDeInit.
  This function takes as parameters the HAL peripheral handle, the Callback ID
  and a pointer to the user callback function.

  Use HAL_SPDIFRX_UnRegisterCallback() function to reset a callback to the default
  weak function.
  The HAL_SPDIFRX_UnRegisterCallback() function takes as parameters the HAL peripheral handle,
  and the Callback ID.
  This function allows to reset the following callbacks:
    (+) RxHalfCpltCallback  : SPDIFRX Data flow half completed callback.
    (+) RxCpltCallback      : SPDIFRX Data flow completed callback.
    (+) CxHalfCpltCallback  : SPDIFRX Control flow half completed callback.
    (+) CxCpltCallback      : SPDIFRX Control flow completed callback.
    (+) ErrorCallback       : SPDIFRX error callback.
    (+) MspInitCallback     : SPDIFRX MspInit.
    (+) MspDeInitCallback   : SPDIFRX MspDeInit.

  By default, after the HAL_SPDIFRX_Init() and when the state is HAL_SPDIFRX_STATE_RESET
  all callbacks are set to the corresponding weak functions :
  HAL_SPDIFRX_RxHalfCpltCallback() , HAL_SPDIFRX_RxCpltCallback(), HAL_SPDIFRX_CxHalfCpltCallback(),
  HAL_SPDIFRX_CxCpltCallback() and HAL_SPDIFRX_ErrorCallback()
  Exception done for MspInit and MspDeInit functions that are
  reset to the legacy weak function in the HAL_SPDIFRX_Init()/ HAL_SPDIFRX_DeInit() only when
  these callbacks pointers are NULL (not registered beforehand).
  If not, MspInit or MspDeInit callbacks pointers are not null, the HAL_SPDIFRX_Init() / HAL_SPDIFRX_DeInit()
  keep and use the user MspInit/MspDeInit functions (registered beforehand)

  Callbacks can be registered/unregistered in HAL_SPDIFRX_STATE_READY state only.
  Exception done MspInit/MspDeInit callbacks that can be registered/unregistered
  in HAL_SPDIFRX_STATE_READY or HAL_SPDIFRX_STATE_RESET state,
  thus registered (user) MspInit/DeInit callbacks can be used during the Init/DeInit.
  In that case first register the MspInit/MspDeInit user callbacks
  using HAL_SPDIFRX_RegisterCallback() before calling HAL_SPDIFRX_DeInit()
  or HAL_SPDIFRX_Init() function.

  When The compilation define USE_HAL_SPDIFRX_REGISTER_CALLBACKS is set to 0 or
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

/** @defgroup SPDIFRX SPDIFRX
  * @brief SPDIFRX HAL module driver
  * @{
  */

#ifdef HAL_SPDIFRX_MODULE_ENABLED
#if defined (SPDIFRX)
#if defined(STM32F446xx)
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define SPDIFRX_TIMEOUT_VALUE  0xFFFFU

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/** @addtogroup SPDIFRX_Private_Functions
  * @{
  */
static void  SPDIFRX_DMARxCplt(DMA_HandleTypeDef *hdma);
static void  SPDIFRX_DMARxHalfCplt(DMA_HandleTypeDef *hdma);
static void  SPDIFRX_DMACxCplt(DMA_HandleTypeDef *hdma);
static void  SPDIFRX_DMACxHalfCplt(DMA_HandleTypeDef *hdma);
static void  SPDIFRX_DMAError(DMA_HandleTypeDef *hdma);
static void  SPDIFRX_ReceiveControlFlow_IT(SPDIFRX_HandleTypeDef *hspdif);
static void  SPDIFRX_ReceiveDataFlow_IT(SPDIFRX_HandleTypeDef *hspdif);
static HAL_StatusTypeDef  SPDIFRX_WaitOnFlagUntilTimeout(SPDIFRX_HandleTypeDef *hspdif, uint32_t Flag, FlagStatus Status, uint32_t Timeout, uint32_t tickstart);
/**
  * @}
  */
/* Exported functions ---------------------------------------------------------*/

/** @defgroup SPDIFRX_Exported_Functions SPDIFRX Exported Functions
  * @{
  */

/** @defgroup  SPDIFRX_Exported_Functions_Group1 Initialization and de-initialization functions
  *  @brief    Initialization and Configuration functions
  *
  @verbatim
  ===============================================================================
  ##### Initialization and de-initialization functions #####
  ===============================================================================
  [..]  This subsection provides a set of functions allowing to initialize and
        de-initialize the SPDIFRX peripheral:

  (+) User must Implement HAL_SPDIFRX_MspInit() function in which he configures
      all related peripherals resources (CLOCK, GPIO, DMA, IT and NVIC ).

  (+) Call the function HAL_SPDIFRX_Init() to configure the SPDIFRX peripheral with
      the selected configuration:
  (++) Input Selection (IN0, IN1,...)
  (++) Maximum allowed re-tries during synchronization phase
  (++) Wait for activity on SPDIF selected input
  (++) Channel status selection (from channel A or B)
  (++) Data format (LSB, MSB, ...)
  (++) Stereo mode
  (++) User bits masking (PT,C,U,V,...)

  (+) Call the function HAL_SPDIFRX_DeInit() to restore the default configuration
      of the selected SPDIFRXx peripheral.
  @endverbatim
  * @{
  */

/**
  * @brief Initializes the SPDIFRX according to the specified parameters
  *        in the SPDIFRX_InitTypeDef and create the associated handle.
  * @param hspdif SPDIFRX handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_Init(SPDIFRX_HandleTypeDef *hspdif)
{
  uint32_t tmpreg;

  /* Check the SPDIFRX handle allocation */
  if(hspdif == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the SPDIFRX parameters */
  assert_param(IS_STEREO_MODE(hspdif->Init.StereoMode));
  assert_param(IS_SPDIFRX_INPUT_SELECT(hspdif->Init.InputSelection));
  assert_param(IS_SPDIFRX_MAX_RETRIES(hspdif->Init.Retries));
  assert_param(IS_SPDIFRX_WAIT_FOR_ACTIVITY(hspdif->Init.WaitForActivity));
  assert_param(IS_SPDIFRX_CHANNEL(hspdif->Init.ChannelSelection));
  assert_param(IS_SPDIFRX_DATA_FORMAT(hspdif->Init.DataFormat));
  assert_param(IS_PREAMBLE_TYPE_MASK(hspdif->Init.PreambleTypeMask));
  assert_param(IS_CHANNEL_STATUS_MASK(hspdif->Init.ChannelStatusMask));
  assert_param(IS_VALIDITY_MASK(hspdif->Init.ValidityBitMask));
  assert_param(IS_PARITY_ERROR_MASK(hspdif->Init.ParityErrorMask));

#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
  if(hspdif->State == HAL_SPDIFRX_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hspdif->Lock = HAL_UNLOCKED;

    hspdif->RxHalfCpltCallback  = HAL_SPDIFRX_RxHalfCpltCallback; /* Legacy weak RxHalfCpltCallback */
    hspdif->RxCpltCallback      = HAL_SPDIFRX_RxCpltCallback;     /* Legacy weak RxCpltCallback     */
    hspdif->CxHalfCpltCallback  = HAL_SPDIFRX_CxHalfCpltCallback; /* Legacy weak CxHalfCpltCallback */
    hspdif->CxCpltCallback      = HAL_SPDIFRX_CxCpltCallback;     /* Legacy weak CxCpltCallback     */
    hspdif->ErrorCallback       = HAL_SPDIFRX_ErrorCallback;      /* Legacy weak ErrorCallback      */

    if(hspdif->MspInitCallback == NULL)
    {
      hspdif->MspInitCallback = HAL_SPDIFRX_MspInit; /* Legacy weak MspInit  */
    }

    /* Init the low level hardware */
    hspdif->MspInitCallback(hspdif);
  }
#else
  if(hspdif->State == HAL_SPDIFRX_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hspdif->Lock = HAL_UNLOCKED;
    /* Init the low level hardware : GPIO, CLOCK, CORTEX...etc */
    HAL_SPDIFRX_MspInit(hspdif);
  }
#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */

  /* SPDIFRX peripheral state is BUSY */
  hspdif->State = HAL_SPDIFRX_STATE_BUSY;

  /* Disable SPDIFRX interface (IDLE State) */
  __HAL_SPDIFRX_IDLE(hspdif);

  /* Reset the old SPDIFRX CR configuration */
  tmpreg = hspdif->Instance->CR;

  tmpreg &= ~(SPDIFRX_CR_RXSTEO  | SPDIFRX_CR_DRFMT  | SPDIFRX_CR_PMSK |
              SPDIFRX_CR_VMSK | SPDIFRX_CR_CUMSK | SPDIFRX_CR_PTMSK  |
              SPDIFRX_CR_CHSEL | SPDIFRX_CR_NBTR | SPDIFRX_CR_WFA |
              SPDIFRX_CR_INSEL);

  /* Sets the new configuration of the SPDIFRX peripheral */
  tmpreg |= (hspdif->Init.StereoMode |
             hspdif->Init.InputSelection |
             hspdif->Init.Retries |
             hspdif->Init.WaitForActivity |
             hspdif->Init.ChannelSelection |
             hspdif->Init.DataFormat |
             hspdif->Init.PreambleTypeMask |
             hspdif->Init.ChannelStatusMask |
             hspdif->Init.ValidityBitMask |
             hspdif->Init.ParityErrorMask
             );


  hspdif->Instance->CR = tmpreg;

  hspdif->ErrorCode = HAL_SPDIFRX_ERROR_NONE;

  /* SPDIFRX peripheral state is READY*/
  hspdif->State = HAL_SPDIFRX_STATE_READY;

  return HAL_OK;
}

/**
  * @brief DeInitializes the SPDIFRX peripheral
  * @param hspdif SPDIFRX handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_DeInit(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Check the SPDIFRX handle allocation */
  if(hspdif == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_SPDIFRX_ALL_INSTANCE(hspdif->Instance));

  hspdif->State = HAL_SPDIFRX_STATE_BUSY;

  /* Disable SPDIFRX interface (IDLE state) */
  __HAL_SPDIFRX_IDLE(hspdif);

#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
  if(hspdif->MspDeInitCallback == NULL)
  {
    hspdif->MspDeInitCallback = HAL_SPDIFRX_MspDeInit; /* Legacy weak MspDeInit  */
  }

  /* DeInit the low level hardware */
  hspdif->MspDeInitCallback(hspdif);
#else
  /* DeInit the low level hardware: GPIO, CLOCK, NVIC... */
  HAL_SPDIFRX_MspDeInit(hspdif);
#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */

  hspdif->ErrorCode = HAL_SPDIFRX_ERROR_NONE;

  /* SPDIFRX peripheral state is RESET*/
  hspdif->State = HAL_SPDIFRX_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(hspdif);

  return HAL_OK;
}

/**
  * @brief SPDIFRX MSP Init
  * @param hspdif SPDIFRX handle
  * @retval None
  */
__weak void HAL_SPDIFRX_MspInit(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hspdif);

  /* NOTE : This function Should not be modified, when the callback is needed,
  the HAL_SPDIFRX_MspInit could be implemented in the user file
  */
}

/**
  * @brief SPDIFRX MSP DeInit
  * @param hspdif SPDIFRX handle
  * @retval None
  */
__weak void HAL_SPDIFRX_MspDeInit(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hspdif);

  /* NOTE : This function Should not be modified, when the callback is needed,
  the HAL_SPDIFRX_MspDeInit could be implemented in the user file
  */
}

#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User SPDIFRX Callback
  *         To be used instead of the weak predefined callback
  * @param  hspdif SPDIFRX handle
  * @param  CallbackID ID of the callback to be registered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_SPDIFRX_RX_HALF_CB_ID    SPDIFRX Data flow half completed callback ID
  *          @arg @ref HAL_SPDIFRX_RX_CPLT_CB_ID    SPDIFRX Data flow completed callback ID
  *          @arg @ref HAL_SPDIFRX_CX_HALF_CB_ID    SPDIFRX Control flow half completed callback ID
  *          @arg @ref HAL_SPDIFRX_CX_CPLT_CB_ID    SPDIFRX Control flow completed callback ID
  *          @arg @ref HAL_SPDIFRX_ERROR_CB_ID      SPDIFRX error callback ID
  *          @arg @ref HAL_SPDIFRX_MSPINIT_CB_ID    MspInit callback ID
  *          @arg @ref HAL_SPDIFRX_MSPDEINIT_CB_ID  MspDeInit callback ID
  * @param  pCallback pointer to the Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_RegisterCallback(SPDIFRX_HandleTypeDef *hspdif, HAL_SPDIFRX_CallbackIDTypeDef CallbackID, pSPDIFRX_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if(pCallback == NULL)
  {
    /* Update the error code */
    hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_INVALID_CALLBACK;
    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hspdif);

  if(HAL_SPDIFRX_STATE_READY == hspdif->State)
  {
    switch (CallbackID)
    {
      case HAL_SPDIFRX_RX_HALF_CB_ID :
        hspdif->RxHalfCpltCallback = pCallback;
        break;

      case HAL_SPDIFRX_RX_CPLT_CB_ID :
        hspdif->RxCpltCallback = pCallback;
        break;

      case HAL_SPDIFRX_CX_HALF_CB_ID :
        hspdif->CxHalfCpltCallback = pCallback;
        break;

      case HAL_SPDIFRX_CX_CPLT_CB_ID :
        hspdif->CxCpltCallback = pCallback;
        break;

      case HAL_SPDIFRX_ERROR_CB_ID :
        hspdif->ErrorCallback = pCallback;
        break;

      case HAL_SPDIFRX_MSPINIT_CB_ID :
        hspdif->MspInitCallback = pCallback;
        break;

      case HAL_SPDIFRX_MSPDEINIT_CB_ID :
        hspdif->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if(HAL_SPDIFRX_STATE_RESET == hspdif->State)
  {
    switch (CallbackID)
    {
      case HAL_SPDIFRX_MSPINIT_CB_ID :
        hspdif->MspInitCallback = pCallback;
        break;

      case HAL_SPDIFRX_MSPDEINIT_CB_ID :
        hspdif->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_INVALID_CALLBACK;
       /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hspdif);
  return status;
}

/**
  * @brief  Unregister a SPDIFRX Callback
  *         SPDIFRX callabck is redirected to the weak predefined callback
  * @param  hspdif SPDIFRX handle
  * @param  CallbackID ID of the callback to be unregistered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_SPDIFRX_RX_HALF_CB_ID    SPDIFRX Data flow half completed callback ID
  *          @arg @ref HAL_SPDIFRX_RX_CPLT_CB_ID    SPDIFRX Data flow completed callback ID
  *          @arg @ref HAL_SPDIFRX_CX_HALF_CB_ID    SPDIFRX Control flow half completed callback ID
  *          @arg @ref HAL_SPDIFRX_CX_CPLT_CB_ID    SPDIFRX Control flow completed callback ID
  *          @arg @ref HAL_SPDIFRX_ERROR_CB_ID      SPDIFRX error callback ID
  *          @arg @ref HAL_SPDIFRX_MSPINIT_CB_ID    MspInit callback ID
  *          @arg @ref HAL_SPDIFRX_MSPDEINIT_CB_ID  MspDeInit callback ID
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_UnRegisterCallback(SPDIFRX_HandleTypeDef *hspdif, HAL_SPDIFRX_CallbackIDTypeDef CallbackID)
{
HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hspdif);

  if(HAL_SPDIFRX_STATE_READY == hspdif->State)
  {
    switch (CallbackID)
    {
      case HAL_SPDIFRX_RX_HALF_CB_ID :
        hspdif->RxHalfCpltCallback = HAL_SPDIFRX_RxHalfCpltCallback;
        break;

      case HAL_SPDIFRX_RX_CPLT_CB_ID :
        hspdif->RxCpltCallback = HAL_SPDIFRX_RxCpltCallback;
        break;

      case HAL_SPDIFRX_CX_HALF_CB_ID :
        hspdif->CxHalfCpltCallback = HAL_SPDIFRX_CxHalfCpltCallback;
        break;

      case HAL_SPDIFRX_CX_CPLT_CB_ID :
        hspdif->CxCpltCallback = HAL_SPDIFRX_CxCpltCallback;
        break;

      case HAL_SPDIFRX_ERROR_CB_ID :
        hspdif->ErrorCallback = HAL_SPDIFRX_ErrorCallback;
        break;

      default :
        /* Update the error code */
        hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_INVALID_CALLBACK;
       /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if(HAL_SPDIFRX_STATE_RESET == hspdif->State)
  {
    switch (CallbackID)
    {
      case HAL_SPDIFRX_MSPINIT_CB_ID :
        hspdif->MspInitCallback = HAL_SPDIFRX_MspInit;  /* Legacy weak MspInit  */
        break;

      case HAL_SPDIFRX_MSPDEINIT_CB_ID :
        hspdif->MspDeInitCallback = HAL_SPDIFRX_MspDeInit;  /* Legacy weak MspInit  */
        break;

      default :
        /* Update the error code */
        hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hspdif);
  return status;
}

#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */

/**
  * @brief Set the SPDIFRX  data format according to the specified parameters in the SPDIFRX_InitTypeDef.
  * @param hspdif SPDIFRX handle
  * @param sDataFormat SPDIFRX data format
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_SetDataFormat(SPDIFRX_HandleTypeDef *hspdif, SPDIFRX_SetDataFormatTypeDef sDataFormat)
{
  uint32_t tmpreg;

  /* Check the SPDIFRX handle allocation */
  if(hspdif == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the SPDIFRX parameters */
  assert_param(IS_STEREO_MODE(sDataFormat.StereoMode));
  assert_param(IS_SPDIFRX_DATA_FORMAT(sDataFormat.DataFormat));
  assert_param(IS_PREAMBLE_TYPE_MASK(sDataFormat.PreambleTypeMask));
  assert_param(IS_CHANNEL_STATUS_MASK(sDataFormat.ChannelStatusMask));
  assert_param(IS_VALIDITY_MASK(sDataFormat.ValidityBitMask));
  assert_param(IS_PARITY_ERROR_MASK(sDataFormat.ParityErrorMask));

  /* Reset the old SPDIFRX CR configuration */
  tmpreg = hspdif->Instance->CR;

  if(((tmpreg & SPDIFRX_STATE_RCV) == SPDIFRX_STATE_RCV) &&
     (((tmpreg & SPDIFRX_CR_DRFMT) != sDataFormat.DataFormat) ||
      ((tmpreg & SPDIFRX_CR_RXSTEO) != sDataFormat.StereoMode)))
  {
    return HAL_ERROR;
  }

  tmpreg &= ~(SPDIFRX_CR_RXSTEO  | SPDIFRX_CR_DRFMT  | SPDIFRX_CR_PMSK |
              SPDIFRX_CR_VMSK | SPDIFRX_CR_CUMSK | SPDIFRX_CR_PTMSK);

  /* Configure the new data format */
  tmpreg |= (sDataFormat.StereoMode |
             sDataFormat.DataFormat |
             sDataFormat.PreambleTypeMask |
             sDataFormat.ChannelStatusMask |
             sDataFormat.ValidityBitMask |
             sDataFormat.ParityErrorMask);

  hspdif->Instance->CR = tmpreg;

  return HAL_OK;
}

/**
  * @}
  */

/** @defgroup SPDIFRX_Exported_Functions_Group2 IO operation functions
  *  @brief Data transfers functions
  *
@verbatim
===============================================================================
                     ##### IO operation functions #####
===============================================================================
    [..]
    This subsection provides a set of functions allowing to manage the SPDIFRX data
    transfers.

    (#) There is two mode of transfer:
        (++) Blocking mode : The communication is performed in the polling mode.
             The status of all data processing is returned by the same function
             after finishing transfer.
        (++) No-Blocking mode : The communication is performed using Interrupts
             or DMA. These functions return the status of the transfer start-up.
             The end of the data processing will be indicated through the
             dedicated SPDIFRX IRQ when using Interrupt mode or the DMA IRQ when
             using DMA mode.

    (#) Blocking mode functions are :
        (++) HAL_SPDIFRX_ReceiveDataFlow()
        (++) HAL_SPDIFRX_ReceiveControlFlow()
                (+@) Do not use blocking mode to receive both control and data flow at the same time.

    (#) No-Blocking mode functions with Interrupt are :
        (++) HAL_SPDIFRX_ReceiveControlFlow_IT()
        (++) HAL_SPDIFRX_ReceiveDataFlow_IT()

    (#) No-Blocking mode functions with DMA are :
        (++) HAL_SPDIFRX_ReceiveControlFlow_DMA()
        (++) HAL_SPDIFRX_ReceiveDataFlow_DMA()

    (#) A set of Transfer Complete Callbacks are provided in No_Blocking mode:
        (++) HAL_SPDIFRX_RxCpltCallback()
        (++) HAL_SPDIFRX_CxCpltCallback()

@endverbatim
* @{
*/

/**
  * @brief  Receives an amount of data (Data Flow) in blocking mode.
  * @param  hspdif pointer to SPDIFRX_HandleTypeDef structure that contains
  *                 the configuration information for SPDIFRX module.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be received
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_ReceiveDataFlow(SPDIFRX_HandleTypeDef *hspdif, uint32_t *pData, uint16_t Size, uint32_t Timeout)
{
  uint32_t tickstart;
  uint16_t sizeCounter = Size;
  uint32_t *pTmpBuf = pData;

  if((pData == NULL ) || (Size == 0U))
  {
    return  HAL_ERROR;
  }

  if(hspdif->State == HAL_SPDIFRX_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hspdif);

    hspdif->State = HAL_SPDIFRX_STATE_BUSY;

    /* Start synchronisation */
    __HAL_SPDIFRX_SYNC(hspdif);

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait until SYNCD flag is set */
    if(SPDIFRX_WaitOnFlagUntilTimeout(hspdif, SPDIFRX_FLAG_SYNCD, RESET, Timeout, tickstart) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }

    /* Start reception */
    __HAL_SPDIFRX_RCV(hspdif);

    /* Receive data flow */
    while(sizeCounter > 0U)
    {
      /* Get tick */
      tickstart = HAL_GetTick();

      /* Wait until RXNE flag is set */
      if(SPDIFRX_WaitOnFlagUntilTimeout(hspdif, SPDIFRX_FLAG_RXNE, RESET, Timeout, tickstart) != HAL_OK)
      {
        return HAL_TIMEOUT;
      }

      (*pTmpBuf) = hspdif->Instance->DR;
      pTmpBuf++;
      sizeCounter--;
    }

    /* SPDIFRX ready */
    hspdif->State = HAL_SPDIFRX_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hspdif);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Receives an amount of data (Control Flow) in blocking mode.
  * @param  hspdif pointer to a SPDIFRX_HandleTypeDef structure that contains
  *                 the configuration information for SPDIFRX module.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be received
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_ReceiveControlFlow(SPDIFRX_HandleTypeDef *hspdif, uint32_t *pData, uint16_t Size, uint32_t Timeout)
{
  uint32_t tickstart;
  uint16_t sizeCounter = Size;
  uint32_t *pTmpBuf = pData;

  if((pData == NULL ) || (Size == 0U))
  {
    return  HAL_ERROR;
  }

  if(hspdif->State == HAL_SPDIFRX_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hspdif);

    hspdif->State = HAL_SPDIFRX_STATE_BUSY;

    /* Start synchronization */
    __HAL_SPDIFRX_SYNC(hspdif);

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait until SYNCD flag is set */
    if(SPDIFRX_WaitOnFlagUntilTimeout(hspdif, SPDIFRX_FLAG_SYNCD, RESET, Timeout, tickstart) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }

    /* Start reception */
    __HAL_SPDIFRX_RCV(hspdif);

    /* Receive control flow */
    while(sizeCounter > 0U)
    {
      /* Get tick */
      tickstart = HAL_GetTick();

      /* Wait until CSRNE flag is set */
      if(SPDIFRX_WaitOnFlagUntilTimeout(hspdif, SPDIFRX_FLAG_CSRNE, RESET, Timeout, tickstart) != HAL_OK)
      {
        return HAL_TIMEOUT;
      }

      (*pTmpBuf) = hspdif->Instance->CSR;
      pTmpBuf++;
      sizeCounter--;
    }

    /* SPDIFRX ready */
    hspdif->State = HAL_SPDIFRX_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hspdif);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Receive an amount of data (Data Flow) in non-blocking mode with Interrupt
  * @param hspdif SPDIFRX handle
  * @param pData a 32-bit pointer to the Receive data buffer.
  * @param Size number of data sample to be received .
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_ReceiveDataFlow_IT(SPDIFRX_HandleTypeDef *hspdif, uint32_t *pData, uint16_t Size)
{
  uint32_t count = SPDIFRX_TIMEOUT_VALUE * (SystemCoreClock / 24U / 1000U);

  const HAL_SPDIFRX_StateTypeDef tempState = hspdif->State;

  if((tempState == HAL_SPDIFRX_STATE_READY) || (tempState == HAL_SPDIFRX_STATE_BUSY_CX))
  {
    if((pData == NULL) || (Size == 0U))
    {
      return HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hspdif);

    hspdif->pRxBuffPtr = pData;
    hspdif->RxXferSize = Size;
    hspdif->RxXferCount = Size;

    hspdif->ErrorCode = HAL_SPDIFRX_ERROR_NONE;

    /* Check if a receive process is ongoing or not */
    hspdif->State = HAL_SPDIFRX_STATE_BUSY_RX;

    /* Enable the SPDIFRX  PE Error Interrupt */
    __HAL_SPDIFRX_ENABLE_IT(hspdif, SPDIFRX_IT_PERRIE);

    /* Enable the SPDIFRX  OVR Error Interrupt */
    __HAL_SPDIFRX_ENABLE_IT(hspdif, SPDIFRX_IT_OVRIE);

    /* Enable the SPDIFRX RXNE interrupt */
    __HAL_SPDIFRX_ENABLE_IT(hspdif, SPDIFRX_IT_RXNE);

    if((SPDIFRX->CR & SPDIFRX_CR_SPDIFEN) != SPDIFRX_STATE_RCV)
    {
      /* Start synchronization */
      __HAL_SPDIFRX_SYNC(hspdif);

      /* Wait until SYNCD flag is set */
      do
      {
        if (count == 0U)
        {
          /* Disable TXE, RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts for the interrupt process */
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_RXNE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_CSRNE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_PERRIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_OVRIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SBLKIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SYNCDIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_IFEIE);

          hspdif->State= HAL_SPDIFRX_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hspdif);

          return HAL_TIMEOUT;
        }
        count--;
      } while (__HAL_SPDIFRX_GET_FLAG(hspdif, SPDIFRX_FLAG_SYNCD) == RESET);

      /* Start reception */
      __HAL_SPDIFRX_RCV(hspdif);
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hspdif);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Receive an amount of data (Control Flow) with Interrupt
  * @param hspdif SPDIFRX handle
  * @param pData a 32-bit pointer to the Receive data buffer.
  * @param Size number of data sample (Control Flow) to be received
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_ReceiveControlFlow_IT(SPDIFRX_HandleTypeDef *hspdif, uint32_t *pData, uint16_t Size)
{
  uint32_t count = SPDIFRX_TIMEOUT_VALUE * (SystemCoreClock / 24U / 1000U);

  const HAL_SPDIFRX_StateTypeDef tempState = hspdif->State;

  if((tempState == HAL_SPDIFRX_STATE_READY) || (tempState == HAL_SPDIFRX_STATE_BUSY_RX))
  {
    if((pData == NULL ) || (Size == 0U))
    {
      return HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hspdif);

    hspdif->pCsBuffPtr = pData;
    hspdif->CsXferSize = Size;
    hspdif->CsXferCount = Size;

    hspdif->ErrorCode = HAL_SPDIFRX_ERROR_NONE;

    /* Check if a receive process is ongoing or not */
    hspdif->State = HAL_SPDIFRX_STATE_BUSY_CX;

    /* Enable the SPDIFRX PE Error Interrupt */
    __HAL_SPDIFRX_ENABLE_IT(hspdif, SPDIFRX_IT_PERRIE);

    /* Enable the SPDIFRX OVR Error Interrupt */
    __HAL_SPDIFRX_ENABLE_IT(hspdif, SPDIFRX_IT_OVRIE);

    /* Enable the SPDIFRX CSRNE interrupt */
    __HAL_SPDIFRX_ENABLE_IT(hspdif, SPDIFRX_IT_CSRNE);

    if((SPDIFRX->CR & SPDIFRX_CR_SPDIFEN) != SPDIFRX_STATE_RCV)
    {
      /* Start synchronization */
      __HAL_SPDIFRX_SYNC(hspdif);

      /* Wait until SYNCD flag is set */
      do
      {
        if (count == 0U)
        {
          /* Disable TXE, RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts for the interrupt process */
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_RXNE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_CSRNE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_PERRIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_OVRIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SBLKIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SYNCDIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_IFEIE);

          hspdif->State= HAL_SPDIFRX_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hspdif);

          return HAL_TIMEOUT;
        }
        count--;
      } while (__HAL_SPDIFRX_GET_FLAG(hspdif, SPDIFRX_FLAG_SYNCD) == RESET);

      /* Start reception */
      __HAL_SPDIFRX_RCV(hspdif);
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hspdif);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Receive an amount of data (Data Flow) mode with DMA
  * @param hspdif SPDIFRX handle
  * @param pData a 32-bit pointer to the Receive data buffer.
  * @param Size number of data sample to be received
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_ReceiveDataFlow_DMA(SPDIFRX_HandleTypeDef *hspdif, uint32_t *pData, uint16_t Size)
{
  uint32_t count = SPDIFRX_TIMEOUT_VALUE * (SystemCoreClock / 24U / 1000U);

  const HAL_SPDIFRX_StateTypeDef tempState = hspdif->State;

  if((pData == NULL) || (Size == 0U))
  {
    return  HAL_ERROR;
  }

  if((tempState == HAL_SPDIFRX_STATE_READY) || (tempState == HAL_SPDIFRX_STATE_BUSY_CX))
  {
    /* Process Locked */
    __HAL_LOCK(hspdif);

    hspdif->pRxBuffPtr = pData;
    hspdif->RxXferSize = Size;
    hspdif->RxXferCount = Size;

    hspdif->ErrorCode = HAL_SPDIFRX_ERROR_NONE;
    hspdif->State = HAL_SPDIFRX_STATE_BUSY_RX;

    /* Set the SPDIFRX Rx DMA Half transfer complete callback */
    hspdif->hdmaDrRx->XferHalfCpltCallback = SPDIFRX_DMARxHalfCplt;

    /* Set the SPDIFRX Rx DMA transfer complete callback */
    hspdif->hdmaDrRx->XferCpltCallback = SPDIFRX_DMARxCplt;

    /* Set the DMA error callback */
    hspdif->hdmaDrRx->XferErrorCallback = SPDIFRX_DMAError;

    /* Enable the DMA request */
    if(HAL_DMA_Start_IT(hspdif->hdmaDrRx, (uint32_t)&hspdif->Instance->DR, (uint32_t)hspdif->pRxBuffPtr, Size) != HAL_OK)
    {
      /* Set SPDIFRX error */
      hspdif->ErrorCode = HAL_SPDIFRX_ERROR_DMA;

      /* Set SPDIFRX state */
      hspdif->State = HAL_SPDIFRX_STATE_ERROR;

      /* Process Unlocked */
      __HAL_UNLOCK(hspdif);

      return HAL_ERROR;
    }

    /* Enable RXDMAEN bit in SPDIFRX CR register for data flow reception*/
    hspdif->Instance->CR |= SPDIFRX_CR_RXDMAEN;

    if((SPDIFRX->CR & SPDIFRX_CR_SPDIFEN) != SPDIFRX_STATE_RCV)
    {
      /* Start synchronization */
      __HAL_SPDIFRX_SYNC(hspdif);

      /* Wait until SYNCD flag is set */
      do
      {
        if (count == 0U)
        {
          /* Disable TXE, RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts for the interrupt process */
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_RXNE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_CSRNE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_PERRIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_OVRIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SBLKIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SYNCDIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_IFEIE);

          hspdif->State= HAL_SPDIFRX_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hspdif);

          return HAL_TIMEOUT;
        }
        count--;
      } while (__HAL_SPDIFRX_GET_FLAG(hspdif, SPDIFRX_FLAG_SYNCD) == RESET);

      /* Start reception */
      __HAL_SPDIFRX_RCV(hspdif);
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hspdif);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Receive an amount of data (Control Flow) with DMA
  * @param hspdif SPDIFRX handle
  * @param pData a 32-bit pointer to the Receive data buffer.
  * @param Size number of data (Control Flow) sample to be received
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SPDIFRX_ReceiveControlFlow_DMA(SPDIFRX_HandleTypeDef *hspdif, uint32_t *pData, uint16_t Size)
{
  uint32_t count = SPDIFRX_TIMEOUT_VALUE * (SystemCoreClock / 24U / 1000U);

  const HAL_SPDIFRX_StateTypeDef tempState = hspdif->State;

  if((pData == NULL) || (Size == 0U))
  {
    return  HAL_ERROR;
  }

  if((tempState == HAL_SPDIFRX_STATE_READY) || (tempState == HAL_SPDIFRX_STATE_BUSY_RX))
  {
    hspdif->pCsBuffPtr = pData;
    hspdif->CsXferSize = Size;
    hspdif->CsXferCount = Size;

    /* Process Locked */
    __HAL_LOCK(hspdif);

    hspdif->ErrorCode = HAL_SPDIFRX_ERROR_NONE;
    hspdif->State = HAL_SPDIFRX_STATE_BUSY_CX;

    /* Set the SPDIFRX Rx DMA Half transfer complete callback */
    hspdif->hdmaCsRx->XferHalfCpltCallback = SPDIFRX_DMACxHalfCplt;

    /* Set the SPDIFRX Rx DMA transfer complete callback */
    hspdif->hdmaCsRx->XferCpltCallback = SPDIFRX_DMACxCplt;

    /* Set the DMA error callback */
    hspdif->hdmaCsRx->XferErrorCallback = SPDIFRX_DMAError;

    /* Enable the DMA request */
    if(HAL_DMA_Start_IT(hspdif->hdmaCsRx, (uint32_t)&hspdif->Instance->CSR, (uint32_t)hspdif->pCsBuffPtr, Size) != HAL_OK)
    {
      /* Set SPDIFRX error */
      hspdif->ErrorCode = HAL_SPDIFRX_ERROR_DMA;

      /* Set SPDIFRX state */
      hspdif->State = HAL_SPDIFRX_STATE_ERROR;

      /* Process Unlocked */
      __HAL_UNLOCK(hspdif);

      return HAL_ERROR;
    }

    /* Enable CBDMAEN bit in SPDIFRX CR register for control flow reception*/
    hspdif->Instance->CR |= SPDIFRX_CR_CBDMAEN;

    if((SPDIFRX->CR & SPDIFRX_CR_SPDIFEN) != SPDIFRX_STATE_RCV)
    {
      /* Start synchronization */
      __HAL_SPDIFRX_SYNC(hspdif);

      /* Wait until SYNCD flag is set */
      do
      {
        if (count == 0U)
        {
          /* Disable TXE, RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts for the interrupt process */
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_RXNE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_CSRNE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_PERRIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_OVRIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SBLKIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SYNCDIE);
          __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_IFEIE);

          hspdif->State= HAL_SPDIFRX_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hspdif);

          return HAL_TIMEOUT;
        }
        count--;
      } while (__HAL_SPDIFRX_GET_FLAG(hspdif, SPDIFRX_FLAG_SYNCD) == RESET);

      /* Start reception */
      __HAL_SPDIFRX_RCV(hspdif);
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hspdif);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief stop the audio stream receive from the Media.
  * @param hspdif SPDIFRX handle
  * @retval None
  */
HAL_StatusTypeDef HAL_SPDIFRX_DMAStop(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Process Locked */
  __HAL_LOCK(hspdif);

  /* Disable the SPDIFRX DMA requests */
  hspdif->Instance->CR &= (uint16_t)(~SPDIFRX_CR_RXDMAEN);
  hspdif->Instance->CR &= (uint16_t)(~SPDIFRX_CR_CBDMAEN);

  /* Disable the SPDIFRX DMA channel */
  __HAL_DMA_DISABLE(hspdif->hdmaDrRx);
  __HAL_DMA_DISABLE(hspdif->hdmaCsRx);

  /* Disable SPDIFRX peripheral */
  __HAL_SPDIFRX_IDLE(hspdif);

  hspdif->State = HAL_SPDIFRX_STATE_READY;

  /* Process Unlocked */
  __HAL_UNLOCK(hspdif);

  return HAL_OK;
}

/**
  * @brief  This function handles SPDIFRX interrupt request.
  * @param  hspdif SPDIFRX handle
  * @retval HAL status
  */
void HAL_SPDIFRX_IRQHandler(SPDIFRX_HandleTypeDef *hspdif)
{
  uint32_t itFlag   = hspdif->Instance->SR;
  uint32_t itSource = hspdif->Instance->IMR;

  /* SPDIFRX in mode Data Flow Reception */
  if(((itFlag & SPDIFRX_FLAG_RXNE) == SPDIFRX_FLAG_RXNE) && ((itSource &  SPDIFRX_IT_RXNE) == SPDIFRX_IT_RXNE))
  {
    __HAL_SPDIFRX_CLEAR_IT(hspdif, SPDIFRX_IT_RXNE);
    SPDIFRX_ReceiveDataFlow_IT(hspdif);
  }

  /* SPDIFRX in mode Control Flow Reception */
  if(((itFlag & SPDIFRX_FLAG_CSRNE) == SPDIFRX_FLAG_CSRNE) && ((itSource &  SPDIFRX_IT_CSRNE) == SPDIFRX_IT_CSRNE))
  {
    __HAL_SPDIFRX_CLEAR_IT(hspdif, SPDIFRX_IT_CSRNE);
    SPDIFRX_ReceiveControlFlow_IT(hspdif);
  }

  /* SPDIFRX Overrun error interrupt occurred */
  if(((itFlag & SPDIFRX_FLAG_OVR) == SPDIFRX_FLAG_OVR) && ((itSource &  SPDIFRX_IT_OVRIE) == SPDIFRX_IT_OVRIE))
  {
    __HAL_SPDIFRX_CLEAR_IT(hspdif, SPDIFRX_IT_OVRIE);

    /* Change the SPDIFRX error code */
    hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_OVR;

    /* the transfer is not stopped */
    HAL_SPDIFRX_ErrorCallback(hspdif);
  }

  /* SPDIFRX Parity error interrupt occurred */
  if(((itFlag & SPDIFRX_FLAG_PERR) == SPDIFRX_FLAG_PERR) && ((itSource &  SPDIFRX_IT_PERRIE) == SPDIFRX_IT_PERRIE))
  {
    __HAL_SPDIFRX_CLEAR_IT(hspdif, SPDIFRX_IT_PERRIE);

    /* Change the SPDIFRX error code */
    hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_PE;

    /* the transfer is not stopped */
    HAL_SPDIFRX_ErrorCallback(hspdif);
  }
}

/**
  * @brief Rx Transfer (Data flow) half completed callbacks
  * @param hspdif SPDIFRX handle
  * @retval None
  */
__weak void HAL_SPDIFRX_RxHalfCpltCallback(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hspdif);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_SPDIFRX_RxCpltCallback could be implemented in the user file
  */
}

/**
  * @brief Rx Transfer (Data flow) completed callbacks
  * @param hspdif SPDIFRX handle
  * @retval None
  */
__weak void HAL_SPDIFRX_RxCpltCallback(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hspdif);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_SPDIFRX_RxCpltCallback could be implemented in the user file
  */
}

/**
  * @brief Rx (Control flow) Transfer half completed callbacks
  * @param hspdif SPDIFRX handle
  * @retval None
  */
__weak void HAL_SPDIFRX_CxHalfCpltCallback(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hspdif);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_SPDIFRX_RxCpltCallback could be implemented in the user file
  */
}

/**
  * @brief Rx Transfer (Control flow) completed callbacks
  * @param hspdif SPDIFRX handle
  * @retval None
  */
__weak void HAL_SPDIFRX_CxCpltCallback(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hspdif);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_SPDIFRX_RxCpltCallback could be implemented in the user file
  */
}

/**
  * @brief SPDIFRX error callbacks
  * @param hspdif SPDIFRX handle
  * @retval None
  */
__weak void HAL_SPDIFRX_ErrorCallback(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hspdif);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_SPDIFRX_ErrorCallback could be implemented in the user file
  */
}

/**
  * @}
  */

/** @defgroup SPDIFRX_Exported_Functions_Group3 Peripheral State and Errors functions
  *  @brief   Peripheral State functions
  *
@verbatim
===============================================================================
##### Peripheral State and Errors functions #####
===============================================================================
[..]
This subsection permit to get in run-time the status of the peripheral
and the data flow.

@endverbatim
  * @{
  */

/**
  * @brief  Return the SPDIFRX state
  * @param  hspdif SPDIFRX handle
  * @retval HAL state
  */
HAL_SPDIFRX_StateTypeDef HAL_SPDIFRX_GetState(SPDIFRX_HandleTypeDef const * const hspdif)
{
  return hspdif->State;
}

/**
  * @brief  Return the SPDIFRX error code
  * @param  hspdif SPDIFRX handle
  * @retval SPDIFRX Error Code
  */
uint32_t HAL_SPDIFRX_GetError(SPDIFRX_HandleTypeDef const * const hspdif)
{
  return hspdif->ErrorCode;
}

/**
  * @}
  */

/**
  * @brief DMA SPDIFRX receive process (Data flow) complete callback
  * @param hdma DMA handle
  * @retval None
  */
static void SPDIFRX_DMARxCplt(DMA_HandleTypeDef *hdma)
{
  SPDIFRX_HandleTypeDef* hspdif = ( SPDIFRX_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;

  /* Disable Rx DMA Request */
  if(hdma->Init.Mode != DMA_CIRCULAR)
  {
    hspdif->Instance->CR &= (uint16_t)(~SPDIFRX_CR_RXDMAEN);
    hspdif->RxXferCount = 0;
    hspdif->State = HAL_SPDIFRX_STATE_READY;
  }
#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
  hspdif->RxCpltCallback(hspdif);
#else
  HAL_SPDIFRX_RxCpltCallback(hspdif);
#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */
}

/**
  * @brief DMA SPDIFRX receive process (Data flow) half complete callback
  * @param hdma DMA handle
  * @retval None
  */
static void SPDIFRX_DMARxHalfCplt(DMA_HandleTypeDef *hdma)
{
  SPDIFRX_HandleTypeDef* hspdif = (SPDIFRX_HandleTypeDef*)((DMA_HandleTypeDef*)hdma)->Parent;

#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
  hspdif->RxHalfCpltCallback(hspdif);
#else
  HAL_SPDIFRX_RxHalfCpltCallback(hspdif);
#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */
}


/**
  * @brief DMA SPDIFRX receive process (Control flow) complete callback
  * @param hdma DMA handle
  * @retval None
  */
static void SPDIFRX_DMACxCplt(DMA_HandleTypeDef *hdma)
{
  SPDIFRX_HandleTypeDef* hspdif = ( SPDIFRX_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;

  /* Disable Cb DMA Request */
  hspdif->Instance->CR &= (uint16_t)(~SPDIFRX_CR_CBDMAEN);
  hspdif->CsXferCount = 0;

  hspdif->State = HAL_SPDIFRX_STATE_READY;
#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
  hspdif->CxCpltCallback(hspdif);
#else
  HAL_SPDIFRX_CxCpltCallback(hspdif);
#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */
}

/**
  * @brief DMA SPDIFRX receive process (Control flow) half complete callback
  * @param hdma DMA handle
  * @retval None
  */
static void SPDIFRX_DMACxHalfCplt(DMA_HandleTypeDef *hdma)
{
  SPDIFRX_HandleTypeDef* hspdif = (SPDIFRX_HandleTypeDef*)((DMA_HandleTypeDef*)hdma)->Parent;

#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
  hspdif->CxHalfCpltCallback(hspdif);
#else
  HAL_SPDIFRX_CxHalfCpltCallback(hspdif);
#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */
}

/**
  * @brief DMA SPDIFRX communication error callback
  * @param hdma DMA handle
  * @retval None
  */
static void SPDIFRX_DMAError(DMA_HandleTypeDef *hdma)
{
  SPDIFRX_HandleTypeDef* hspdif = ( SPDIFRX_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;

  /* Disable Rx and Cb DMA Request */
  hspdif->Instance->CR &= (uint16_t)(~(SPDIFRX_CR_RXDMAEN | SPDIFRX_CR_CBDMAEN));
  hspdif->RxXferCount = 0;

  hspdif->State= HAL_SPDIFRX_STATE_READY;

  /* Set the error code and execute error callback*/
  hspdif->ErrorCode |= HAL_SPDIFRX_ERROR_DMA;

#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
  /* The transfer is not stopped */
  hspdif->ErrorCallback(hspdif);
#else
  /* The transfer is not stopped */
  HAL_SPDIFRX_ErrorCallback(hspdif);
#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */
}

/**
  * @brief Receive an amount of data (Data Flow) with Interrupt
  * @param hspdif SPDIFRX handle
  * @retval None
  */
static void SPDIFRX_ReceiveDataFlow_IT(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Receive data */
  (*hspdif->pRxBuffPtr) = hspdif->Instance->DR;
  hspdif->pRxBuffPtr++;
  hspdif->RxXferCount--;

  if(hspdif->RxXferCount == 0U)
  {
    /* Disable RXNE/PE and OVR interrupts */
    __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_OVRIE | SPDIFRX_IT_PERRIE | SPDIFRX_IT_RXNE);

    hspdif->State = HAL_SPDIFRX_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hspdif);

#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
  hspdif->RxCpltCallback(hspdif);
#else
  HAL_SPDIFRX_RxCpltCallback(hspdif);
#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */
  }
}

/**
  * @brief Receive an amount of data (Control Flow) with Interrupt
  * @param hspdif SPDIFRX handle
  * @retval None
  */
static void SPDIFRX_ReceiveControlFlow_IT(SPDIFRX_HandleTypeDef *hspdif)
{
  /* Receive data */
  (*hspdif->pCsBuffPtr) = hspdif->Instance->CSR;
  hspdif->pCsBuffPtr++;
  hspdif->CsXferCount--;

  if(hspdif->CsXferCount == 0U)
  {
    /* Disable CSRNE interrupt */
    __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_CSRNE);

    hspdif->State = HAL_SPDIFRX_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hspdif);

#if (USE_HAL_SPDIFRX_REGISTER_CALLBACKS == 1)
  hspdif->CxCpltCallback(hspdif);
#else
  HAL_SPDIFRX_CxCpltCallback(hspdif);
#endif /* USE_HAL_SPDIFRX_REGISTER_CALLBACKS */
  }
}

/**
  * @brief This function handles SPDIFRX Communication Timeout.
  * @param hspdif SPDIFRX handle
  * @param Flag Flag checked
  * @param Status Value of the flag expected
  * @param Timeout Duration of the timeout
  * @param tickstart Tick start value
  * @retval HAL status
  */
static HAL_StatusTypeDef SPDIFRX_WaitOnFlagUntilTimeout(SPDIFRX_HandleTypeDef *hspdif, uint32_t Flag, FlagStatus Status, uint32_t Timeout, uint32_t tickstart)
{
  /* Wait until flag is set */
  while(__HAL_SPDIFRX_GET_FLAG(hspdif, Flag) == Status)
  {
    /* Check for the Timeout */
    if(Timeout != HAL_MAX_DELAY)
    {
      if(((HAL_GetTick() - tickstart ) > Timeout) || (Timeout == 0U))
      {
        /* Disable TXE, RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts for the interrupt process */
        __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_RXNE);
        __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_CSRNE);
        __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_PERRIE);
        __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_OVRIE);
        __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SBLKIE);
        __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_SYNCDIE);
        __HAL_SPDIFRX_DISABLE_IT(hspdif, SPDIFRX_IT_IFEIE);

        hspdif->State= HAL_SPDIFRX_STATE_READY;

        /* Process Unlocked */
        __HAL_UNLOCK(hspdif);

        return HAL_TIMEOUT;
      }
    }
  }

  return HAL_OK;
}

/**
  * @}
  */
#endif /* STM32F446xx */

#endif /* SPDIFRX */
#endif /* HAL_SPDIFRX_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
