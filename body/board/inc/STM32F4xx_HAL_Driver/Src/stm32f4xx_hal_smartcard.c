/**
  ******************************************************************************
  * @file    stm32f4xx_hal_smartcard.c
  * @author  MCD Application Team
  * @brief   SMARTCARD HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the SMARTCARD peripheral:
  *           + Initialization and de-initialization functions
  *           + IO operation functions
  *           + Peripheral Control functions
  *           + Peripheral State and Error functions
  *
  @verbatim
  ==============================================================================
                     ##### How to use this driver #####
  ==============================================================================
    [..]
      The SMARTCARD HAL driver can be used as follows:

    (#) Declare a SMARTCARD_HandleTypeDef handle structure.
    (#) Initialize the SMARTCARD low level resources by implementing the HAL_SMARTCARD_MspInit() API:
        (##) Enable the interface clock of the USARTx associated to the SMARTCARD.
        (##) SMARTCARD pins configuration:
            (+++) Enable the clock for the SMARTCARD GPIOs.
            (+++) Configure SMARTCARD pins as alternate function pull-up.
        (##) NVIC configuration if you need to use interrupt process (HAL_SMARTCARD_Transmit_IT()
             and HAL_SMARTCARD_Receive_IT() APIs):
            (+++) Configure the USARTx interrupt priority.
            (+++) Enable the NVIC USART IRQ handle.
        (##) DMA Configuration if you need to use DMA process (HAL_SMARTCARD_Transmit_DMA()
             and HAL_SMARTCARD_Receive_DMA() APIs):
            (+++) Declare a DMA handle structure for the Tx/Rx stream.
            (+++) Enable the DMAx interface clock.
            (+++) Configure the declared DMA handle structure with the required Tx/Rx parameters.
            (+++) Configure the DMA Tx/Rx stream.
            (+++) Associate the initialized DMA handle to the SMARTCARD DMA Tx/Rx handle.
            (+++) Configure the priority and enable the NVIC for the transfer complete interrupt on the DMA Tx/Rx stream.
            (+++) Configure the USARTx interrupt priority and enable the NVIC USART IRQ handle
                  (used for last byte sending completion detection in DMA non circular mode)

    (#) Program the Baud Rate, Word Length , Stop Bit, Parity, Hardware
        flow control and Mode(Receiver/Transmitter) in the SMARTCARD Init structure.

    (#) Initialize the SMARTCARD registers by calling the HAL_SMARTCARD_Init() API:
        (++) These APIs configure also the low level Hardware GPIO, CLOCK, CORTEX...etc)
             by calling the customized HAL_SMARTCARD_MspInit() API.
    [..]
    (@) The specific SMARTCARD interrupts (Transmission complete interrupt,
        RXNE interrupt and Error Interrupts) will be managed using the macros
        __HAL_SMARTCARD_ENABLE_IT() and __HAL_SMARTCARD_DISABLE_IT() inside the transmit and receive process.

    [..]
    Three operation modes are available within this driver :

    *** Polling mode IO operation ***
    =================================
    [..]
      (+) Send an amount of data in blocking mode using HAL_SMARTCARD_Transmit()
      (+) Receive an amount of data in blocking mode using HAL_SMARTCARD_Receive()

    *** Interrupt mode IO operation ***
    ===================================
    [..]
      (+) Send an amount of data in non blocking mode using HAL_SMARTCARD_Transmit_IT()
      (+) At transmission end of transfer HAL_SMARTCARD_TxCpltCallback is executed and user can
          add his own code by customization of function pointer HAL_SMARTCARD_TxCpltCallback
      (+) Receive an amount of data in non blocking mode using HAL_SMARTCARD_Receive_IT()
      (+) At reception end of transfer HAL_SMARTCARD_RxCpltCallback is executed and user can
          add his own code by customization of function pointer HAL_SMARTCARD_RxCpltCallback
      (+) In case of transfer Error, HAL_SMARTCARD_ErrorCallback() function is executed and user can
          add his own code by customization of function pointer HAL_SMARTCARD_ErrorCallback

    *** DMA mode IO operation ***
    ==============================
    [..]
      (+) Send an amount of data in non blocking mode (DMA) using HAL_SMARTCARD_Transmit_DMA()
      (+) At transmission end of transfer HAL_SMARTCARD_TxCpltCallback is executed and user can
          add his own code by customization of function pointer HAL_SMARTCARD_TxCpltCallback
      (+) Receive an amount of data in non blocking mode (DMA) using HAL_SMARTCARD_Receive_DMA()
      (+) At reception end of transfer HAL_SMARTCARD_RxCpltCallback is executed and user can
          add his own code by customization of function pointer HAL_SMARTCARD_RxCpltCallback
      (+) In case of transfer Error, HAL_SMARTCARD_ErrorCallback() function is executed and user can
          add his own code by customization of function pointer HAL_SMARTCARD_ErrorCallback

    *** SMARTCARD HAL driver macros list ***
    ========================================
    [..]
      Below the list of most used macros in SMARTCARD HAL driver.

      (+) __HAL_SMARTCARD_ENABLE: Enable the SMARTCARD peripheral
      (+) __HAL_SMARTCARD_DISABLE: Disable the SMARTCARD peripheral
      (+) __HAL_SMARTCARD_GET_FLAG : Check whether the specified SMARTCARD flag is set or not
      (+) __HAL_SMARTCARD_CLEAR_FLAG : Clear the specified SMARTCARD pending flag
      (+) __HAL_SMARTCARD_ENABLE_IT: Enable the specified SMARTCARD interrupt
      (+) __HAL_SMARTCARD_DISABLE_IT: Disable the specified SMARTCARD interrupt

    [..]
      (@) You can refer to the SMARTCARD HAL driver header file for more useful macros

    ##### Callback registration #####
    ==================================

    [..]
    The compilation define USE_HAL_SMARTCARD_REGISTER_CALLBACKS when set to 1
    allows the user to configure dynamically the driver callbacks.

    [..]
    Use Function HAL_SMARTCARD_RegisterCallback() to register a user callback.
    Function HAL_SMARTCARD_RegisterCallback() allows to register following callbacks:
    (+) TxCpltCallback            : Tx Complete Callback.
    (+) RxCpltCallback            : Rx Complete Callback.
    (+) ErrorCallback             : Error Callback.
    (+) AbortCpltCallback         : Abort Complete Callback.
    (+) AbortTransmitCpltCallback : Abort Transmit Complete Callback.
    (+) AbortReceiveCpltCallback  : Abort Receive Complete Callback.
    (+) MspInitCallback           : SMARTCARD MspInit.
    (+) MspDeInitCallback         : SMARTCARD MspDeInit.
    This function takes as parameters the HAL peripheral handle, the Callback ID
    and a pointer to the user callback function.

    [..]
    Use function HAL_SMARTCARD_UnRegisterCallback() to reset a callback to the default
    weak (surcharged) function.
    HAL_SMARTCARD_UnRegisterCallback() takes as parameters the HAL peripheral handle,
    and the Callback ID.
    This function allows to reset following callbacks:
    (+) TxCpltCallback            : Tx Complete Callback.
    (+) RxCpltCallback            : Rx Complete Callback.
    (+) ErrorCallback             : Error Callback.
    (+) AbortCpltCallback         : Abort Complete Callback.
    (+) AbortTransmitCpltCallback : Abort Transmit Complete Callback.
    (+) AbortReceiveCpltCallback  : Abort Receive Complete Callback.
    (+) MspInitCallback           : SMARTCARD MspInit.
    (+) MspDeInitCallback         : SMARTCARD MspDeInit.

    [..]
    By default, after the HAL_SMARTCARD_Init() and when the state is HAL_SMARTCARD_STATE_RESET
    all callbacks are set to the corresponding weak (surcharged) functions:
    examples HAL_SMARTCARD_TxCpltCallback(), HAL_SMARTCARD_RxCpltCallback().
    Exception done for MspInit and MspDeInit functions that are respectively
    reset to the legacy weak (surcharged) functions in the HAL_SMARTCARD_Init()
    and HAL_SMARTCARD_DeInit() only when these callbacks are null (not registered beforehand).
    If not, MspInit or MspDeInit are not null, the HAL_SMARTCARD_Init() and HAL_SMARTCARD_DeInit()
    keep and use the user MspInit/MspDeInit callbacks (registered beforehand).

    [..]
    Callbacks can be registered/unregistered in HAL_SMARTCARD_STATE_READY state only.
    Exception done MspInit/MspDeInit that can be registered/unregistered
    in HAL_SMARTCARD_STATE_READY or HAL_SMARTCARD_STATE_RESET state, thus registered (user)
    MspInit/DeInit callbacks can be used during the Init/DeInit.
    In that case first register the MspInit/MspDeInit user callbacks
    using HAL_SMARTCARD_RegisterCallback() before calling HAL_SMARTCARD_DeInit()
    or HAL_SMARTCARD_Init() function.

    [..]
    When The compilation define USE_HAL_SMARTCARD_REGISTER_CALLBACKS is set to 0 or
    not defined, the callback registration feature is not available
    and weak (surcharged) callbacks are used.

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

/** @defgroup SMARTCARD SMARTCARD
  * @brief HAL SMARTCARD module driver
  * @{
  */
#ifdef HAL_SMARTCARD_MODULE_ENABLED
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @addtogroup SMARTCARD_Private_Constants
  * @{
  */
/**
  * @}
  */

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/** @addtogroup SMARTCARD_Private_Functions
  * @{
  */
#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
void SMARTCARD_InitCallbacksToDefault(SMARTCARD_HandleTypeDef *hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACKS */
static void SMARTCARD_EndTxTransfer(SMARTCARD_HandleTypeDef *hsc);
static void SMARTCARD_EndRxTransfer(SMARTCARD_HandleTypeDef *hsc);
static void SMARTCARD_SetConfig (SMARTCARD_HandleTypeDef *hsc);
static HAL_StatusTypeDef SMARTCARD_Transmit_IT(SMARTCARD_HandleTypeDef *hsc);
static HAL_StatusTypeDef SMARTCARD_EndTransmit_IT(SMARTCARD_HandleTypeDef *hsc);
static HAL_StatusTypeDef SMARTCARD_Receive_IT(SMARTCARD_HandleTypeDef *hsc);
static void SMARTCARD_DMATransmitCplt(DMA_HandleTypeDef *hdma);
static void SMARTCARD_DMAReceiveCplt(DMA_HandleTypeDef *hdma);
static void SMARTCARD_DMAError(DMA_HandleTypeDef *hdma);
static void SMARTCARD_DMAAbortOnError(DMA_HandleTypeDef *hdma);
static void SMARTCARD_DMATxAbortCallback(DMA_HandleTypeDef *hdma);
static void SMARTCARD_DMARxAbortCallback(DMA_HandleTypeDef *hdma);
static void SMARTCARD_DMATxOnlyAbortCallback(DMA_HandleTypeDef *hdma);
static void SMARTCARD_DMARxOnlyAbortCallback(DMA_HandleTypeDef *hdma);
static HAL_StatusTypeDef SMARTCARD_WaitOnFlagUntilTimeout(SMARTCARD_HandleTypeDef *hsc, uint32_t Flag, FlagStatus Status, uint32_t Tickstart, uint32_t Timeout);
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup SMARTCARD_Exported_Functions SMARTCARD Exported Functions
  * @{
  */

/** @defgroup SMARTCARD_Exported_Functions_Group1 SmartCard Initialization and de-initialization functions
  *  @brief    Initialization and Configuration functions
  *
@verbatim
  ==============================================================================
              ##### Initialization and Configuration functions #####
  ==============================================================================
  [..]
  This subsection provides a set of functions allowing to initialize the USART
  in Smartcard mode.
  [..]
  The Smartcard interface is designed to support asynchronous protocol Smartcards as
  defined in the ISO 7816-3 standard.
  [..]
  The USART can provide a clock to the smartcard through the SCLK output.
  In smartcard mode, SCLK is not associated to the communication but is simply derived
  from the internal peripheral input clock through a 5-bit prescaler.
  [..]
  (+) For the Smartcard mode only these parameters can be configured:
      (++) Baud Rate
      (++) Word Length => Should be 9 bits (8 bits + parity)
      (++) Stop Bit
      (++) Parity: => Should be enabled
      (++) USART polarity
      (++) USART phase
      (++) USART LastBit
      (++) Receiver/transmitter modes
      (++) Prescaler
      (++) GuardTime
      (++) NACKState: The Smartcard NACK state

     (+) Recommended SmartCard interface configuration to get the Answer to Reset from the Card:
        (++) Word Length = 9 Bits
        (++) 1.5 Stop Bit
        (++) Even parity
        (++) BaudRate = 12096 baud
        (++) Tx and Rx enabled
  [..]
  Please refer to the ISO 7816-3 specification for more details.

  [..]
   (@) It is also possible to choose 0.5 stop bit for receiving but it is recommended
       to use 1.5 stop bits for both transmitting and receiving to avoid switching
       between the two configurations.
  [..]
    The HAL_SMARTCARD_Init() function follows the USART  SmartCard configuration
    procedures (details for the procedures are available in reference manual
    (RM0430 for STM32F4X3xx MCUs and RM0402 for STM32F412xx MCUs
     RM0383 for STM32F411xC/E MCUs and RM0401 for STM32F410xx MCUs
     RM0090 for STM32F4X5xx/STM32F4X7xx/STM32F429xx/STM32F439xx MCUs
     RM0390 for STM32F446xx MCUs and RM0386 for STM32F469xx/STM32F479xx MCUs)).

@endverbatim

  The SMARTCARD frame format is given in the following table:
       +-------------------------------------------------------------+
       |   M bit |  PCE bit  |        SMARTCARD frame                |
       |---------------------|---------------------------------------|
       |    1    |    1      |    | SB | 8 bit data | PB | STB |     |
       +-------------------------------------------------------------+
  * @{
  */

/**
  * @brief  Initializes the SmartCard mode according to the specified
  *         parameters in the SMARTCARD_InitTypeDef and create the associated handle.
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_Init(SMARTCARD_HandleTypeDef *hsc)
{
  /* Check the SMARTCARD handle allocation */
  if(hsc == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_SMARTCARD_INSTANCE(hsc->Instance));
  assert_param(IS_SMARTCARD_NACK_STATE(hsc->Init.NACKState));

  if(hsc->gState == HAL_SMARTCARD_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hsc->Lock = HAL_UNLOCKED;

#if USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1
    SMARTCARD_InitCallbacksToDefault(hsc);

    if (hsc->MspInitCallback == NULL)
    {
      hsc->MspInitCallback = HAL_SMARTCARD_MspInit;
    }

    /* Init the low level hardware */
    hsc->MspInitCallback(hsc);
#else
    /* Init the low level hardware : GPIO, CLOCK */
    HAL_SMARTCARD_MspInit(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACKS */
  }

  hsc->gState = HAL_SMARTCARD_STATE_BUSY;

  /* Set the Prescaler */
  MODIFY_REG(hsc->Instance->GTPR, USART_GTPR_PSC, hsc->Init.Prescaler);

  /* Set the Guard Time */
  MODIFY_REG(hsc->Instance->GTPR, USART_GTPR_GT, ((hsc->Init.GuardTime)<<8U));

  /* Set the Smartcard Communication parameters */
  SMARTCARD_SetConfig(hsc);

  /* In SmartCard mode, the following bits must be kept cleared:
  - LINEN bit in the USART_CR2 register
  - HDSEL and IREN bits in the USART_CR3 register.*/
  CLEAR_BIT(hsc->Instance->CR2, USART_CR2_LINEN);
  CLEAR_BIT(hsc->Instance->CR3, (USART_CR3_IREN | USART_CR3_HDSEL));

  /* Enable the SMARTCARD Parity Error Interrupt */
  SET_BIT(hsc->Instance->CR1, USART_CR1_PEIE);

  /* Enable the SMARTCARD Framing Error Interrupt */
  SET_BIT(hsc->Instance->CR3, USART_CR3_EIE);

  /* Enable the Peripheral */
  __HAL_SMARTCARD_ENABLE(hsc);

  /* Configure the Smartcard NACK state */
  MODIFY_REG(hsc->Instance->CR3, USART_CR3_NACK, hsc->Init.NACKState);

  /* Enable the SC mode by setting the SCEN bit in the CR3 register */
  hsc->Instance->CR3 |= (USART_CR3_SCEN);

  /* Initialize the SMARTCARD state*/
  hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;
  hsc->gState= HAL_SMARTCARD_STATE_READY;
  hsc->RxState= HAL_SMARTCARD_STATE_READY;

  return HAL_OK;
}

/**
  * @brief DeInitializes the USART SmartCard peripheral
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_DeInit(SMARTCARD_HandleTypeDef *hsc)
{
  /* Check the SMARTCARD handle allocation */
  if(hsc == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_SMARTCARD_INSTANCE(hsc->Instance));

  hsc->gState = HAL_SMARTCARD_STATE_BUSY;

  /* Disable the Peripheral */
  __HAL_SMARTCARD_DISABLE(hsc);

  /* DeInit the low level hardware */
#if USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1
  if (hsc->MspDeInitCallback == NULL)
  {
    hsc->MspDeInitCallback = HAL_SMARTCARD_MspDeInit;
  }
  /* DeInit the low level hardware */
  hsc->MspDeInitCallback(hsc);
#else
  HAL_SMARTCARD_MspDeInit(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACKS */

  hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;
  hsc->gState = HAL_SMARTCARD_STATE_RESET;
  hsc->RxState = HAL_SMARTCARD_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(hsc);

  return HAL_OK;
}

/**
  * @brief  SMARTCARD MSP Init
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval None
  */
__weak void HAL_SMARTCARD_MspInit(SMARTCARD_HandleTypeDef *hsc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMARTCARD_MspInit can be implemented in the user file
   */
}

/**
  * @brief SMARTCARD MSP DeInit
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval None
  */
__weak void HAL_SMARTCARD_MspDeInit(SMARTCARD_HandleTypeDef *hsc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMARTCARD_MspDeInit can be implemented in the user file
   */
}

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User SMARTCARD Callback
  *         To be used instead of the weak predefined callback
  * @param  hsc smartcard handle
  * @param  CallbackID ID of the callback to be registered
  *         This parameter can be one of the following values:
  *           @arg @ref HAL_SMARTCARD_TX_COMPLETE_CB_ID Tx Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_RX_COMPLETE_CB_ID Rx Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_ERROR_CB_ID Error Callback ID
  *           @arg @ref HAL_SMARTCARD_ABORT_COMPLETE_CB_ID Abort Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_ABORT_TRANSMIT_COMPLETE_CB_ID Abort Transmit Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_ABORT_RECEIVE_COMPLETE_CB_ID Abort Receive Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_MSPINIT_CB_ID MspInit Callback ID
  *           @arg @ref HAL_SMARTCARD_MSPDEINIT_CB_ID MspDeInit Callback ID
  * @param  pCallback pointer to the Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_RegisterCallback(SMARTCARD_HandleTypeDef *hsc, HAL_SMARTCARD_CallbackIDTypeDef CallbackID, pSMARTCARD_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hsc->ErrorCode |= HAL_SMARTCARD_ERROR_INVALID_CALLBACK;

    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hsc);

  if (hsc->gState == HAL_SMARTCARD_STATE_READY)
  {
    switch (CallbackID)
    {

      case HAL_SMARTCARD_TX_COMPLETE_CB_ID :
        hsc->TxCpltCallback = pCallback;
        break;

      case HAL_SMARTCARD_RX_COMPLETE_CB_ID :
        hsc->RxCpltCallback = pCallback;
        break;

      case HAL_SMARTCARD_ERROR_CB_ID :
        hsc->ErrorCallback = pCallback;
        break;

      case HAL_SMARTCARD_ABORT_COMPLETE_CB_ID :
        hsc->AbortCpltCallback = pCallback;
        break;

      case HAL_SMARTCARD_ABORT_TRANSMIT_COMPLETE_CB_ID :
        hsc->AbortTransmitCpltCallback = pCallback;
        break;

      case HAL_SMARTCARD_ABORT_RECEIVE_COMPLETE_CB_ID :
        hsc->AbortReceiveCpltCallback = pCallback;
        break;


      case HAL_SMARTCARD_MSPINIT_CB_ID :
        hsc->MspInitCallback = pCallback;
        break;

      case HAL_SMARTCARD_MSPDEINIT_CB_ID :
        hsc->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hsc->ErrorCode |= HAL_SMARTCARD_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (hsc->gState == HAL_SMARTCARD_STATE_RESET)
  {
    switch (CallbackID)
    {
      case HAL_SMARTCARD_MSPINIT_CB_ID :
        hsc->MspInitCallback = pCallback;
        break;

      case HAL_SMARTCARD_MSPDEINIT_CB_ID :
        hsc->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hsc->ErrorCode |= HAL_SMARTCARD_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hsc->ErrorCode |= HAL_SMARTCARD_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hsc);

  return status;
}

/**
  * @brief  Unregister an SMARTCARD callback
  *         SMARTCARD callback is redirected to the weak predefined callback
  * @param  hsc smartcard handle
  * @param  CallbackID ID of the callback to be unregistered
  *         This parameter can be one of the following values:
  *           @arg @ref HAL_SMARTCARD_TX_COMPLETE_CB_ID Tx Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_RX_COMPLETE_CB_ID Rx Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_ERROR_CB_ID Error Callback ID
  *           @arg @ref HAL_SMARTCARD_ABORT_COMPLETE_CB_ID Abort Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_ABORT_TRANSMIT_COMPLETE_CB_ID Abort Transmit Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_ABORT_RECEIVE_COMPLETE_CB_ID Abort Receive Complete Callback ID
  *           @arg @ref HAL_SMARTCARD_MSPINIT_CB_ID MspInit Callback ID
  *           @arg @ref HAL_SMARTCARD_MSPDEINIT_CB_ID MspDeInit Callback ID
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_UnRegisterCallback(SMARTCARD_HandleTypeDef *hsc, HAL_SMARTCARD_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hsc);

  if (HAL_SMARTCARD_STATE_READY == hsc->gState)
  {
    switch (CallbackID)
    {
      case HAL_SMARTCARD_TX_COMPLETE_CB_ID :
        hsc->TxCpltCallback = HAL_SMARTCARD_TxCpltCallback;                       /* Legacy weak TxCpltCallback            */
        break;

      case HAL_SMARTCARD_RX_COMPLETE_CB_ID :
        hsc->RxCpltCallback = HAL_SMARTCARD_RxCpltCallback;                       /* Legacy weak RxCpltCallback            */
        break;

      case HAL_SMARTCARD_ERROR_CB_ID :
        hsc->ErrorCallback = HAL_SMARTCARD_ErrorCallback;                         /* Legacy weak ErrorCallback             */
        break;

      case HAL_SMARTCARD_ABORT_COMPLETE_CB_ID :
        hsc->AbortCpltCallback = HAL_SMARTCARD_AbortCpltCallback;                 /* Legacy weak AbortCpltCallback         */
        break;

      case HAL_SMARTCARD_ABORT_TRANSMIT_COMPLETE_CB_ID :
        hsc->AbortTransmitCpltCallback = HAL_SMARTCARD_AbortTransmitCpltCallback; /* Legacy weak AbortTransmitCpltCallback */
        break;

      case HAL_SMARTCARD_ABORT_RECEIVE_COMPLETE_CB_ID :
        hsc->AbortReceiveCpltCallback = HAL_SMARTCARD_AbortReceiveCpltCallback;   /* Legacy weak AbortReceiveCpltCallback  */
        break;


      case HAL_SMARTCARD_MSPINIT_CB_ID :
        hsc->MspInitCallback = HAL_SMARTCARD_MspInit;                             /* Legacy weak MspInitCallback           */
        break;

      case HAL_SMARTCARD_MSPDEINIT_CB_ID :
        hsc->MspDeInitCallback = HAL_SMARTCARD_MspDeInit;                         /* Legacy weak MspDeInitCallback         */
        break;

      default :
        /* Update the error code */
        hsc->ErrorCode |= HAL_SMARTCARD_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_SMARTCARD_STATE_RESET == hsc->gState)
  {
    switch (CallbackID)
    {
      case HAL_SMARTCARD_MSPINIT_CB_ID :
        hsc->MspInitCallback = HAL_SMARTCARD_MspInit;
        break;

      case HAL_SMARTCARD_MSPDEINIT_CB_ID :
        hsc->MspDeInitCallback = HAL_SMARTCARD_MspDeInit;
        break;

      default :
        /* Update the error code */
        hsc->ErrorCode |= HAL_SMARTCARD_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hsc->ErrorCode |= HAL_SMARTCARD_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hsc);

  return status;
}
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup SMARTCARD_Exported_Functions_Group2 IO operation functions
  * @brief    SMARTCARD Transmit and Receive functions
  *
@verbatim
 ===============================================================================
                      ##### IO operation functions #####
 ===============================================================================
 [..]
   This subsection provides a set of functions allowing to manage the SMARTCARD data transfers.

 [..]
    (#) Smartcard is a single wire half duplex communication protocol.
    The Smartcard interface is designed to support asynchronous protocol Smartcards as
    defined in the ISO 7816-3 standard.
    (#) The USART should be configured as:
       (++) 8 bits plus parity: where M=1 and PCE=1 in the USART_CR1 register
       (++) 1.5 stop bits when transmitting and receiving: where STOP=11 in the USART_CR2 register.

    (#) There are two modes of transfer:
       (++) Blocking mode: The communication is performed in polling mode.
            The HAL status of all data processing is returned by the same function
            after finishing transfer.
       (++) Non Blocking mode: The communication is performed using Interrupts
           or DMA, These APIs return the HAL status.
           The end of the data processing will be indicated through the
           dedicated SMARTCARD IRQ when using Interrupt mode or the DMA IRQ when
           using DMA mode.
           The HAL_SMARTCARD_TxCpltCallback(), HAL_SMARTCARD_RxCpltCallback() user callbacks
           will be executed respectively at the end of the Transmit or Receive process
           The HAL_SMARTCARD_ErrorCallback() user callback will be executed when a communication error is detected

    (#) Blocking mode APIs are :
        (++) HAL_SMARTCARD_Transmit()
        (++) HAL_SMARTCARD_Receive()

    (#) Non Blocking mode APIs with Interrupt are :
        (++) HAL_SMARTCARD_Transmit_IT()
        (++) HAL_SMARTCARD_Receive_IT()
        (++) HAL_SMARTCARD_IRQHandler()

    (#) Non Blocking mode functions with DMA are :
        (++) HAL_SMARTCARD_Transmit_DMA()
        (++) HAL_SMARTCARD_Receive_DMA()

    (#) A set of Transfer Complete Callbacks are provided in non Blocking mode:
        (++) HAL_SMARTCARD_TxCpltCallback()
        (++) HAL_SMARTCARD_RxCpltCallback()
        (++) HAL_SMARTCARD_ErrorCallback()

    (#) Non-Blocking mode transfers could be aborted using Abort API's :
        (+) HAL_SMARTCARD_Abort()
        (+) HAL_SMARTCARD_AbortTransmit()
        (+) HAL_SMARTCARD_AbortReceive()
        (+) HAL_SMARTCARD_Abort_IT()
        (+) HAL_SMARTCARD_AbortTransmit_IT()
        (+) HAL_SMARTCARD_AbortReceive_IT()

    (#) For Abort services based on interrupts (HAL_SMARTCARD_Abortxxx_IT), a set of Abort Complete Callbacks are provided:
        (+) HAL_SMARTCARD_AbortCpltCallback()
        (+) HAL_SMARTCARD_AbortTransmitCpltCallback()
        (+) HAL_SMARTCARD_AbortReceiveCpltCallback()

    (#) In Non-Blocking mode transfers, possible errors are split into 2 categories.
        Errors are handled as follows :
       (+) Error is considered as Recoverable and non blocking : Transfer could go till end, but error severity is
           to be evaluated by user : this concerns Frame Error, Parity Error or Noise Error in Interrupt mode reception .
           Received character is then retrieved and stored in Rx buffer, Error code is set to allow user to identify error type,
           and HAL_SMARTCARD_ErrorCallback() user callback is executed. Transfer is kept ongoing on SMARTCARD side.
           If user wants to abort it, Abort services should be called by user.
       (+) Error is considered as Blocking : Transfer could not be completed properly and is aborted.
           This concerns Frame Error in Interrupt mode transmission, Overrun Error in Interrupt mode reception and all errors in DMA mode.
           Error code is set to allow user to identify error type, and HAL_SMARTCARD_ErrorCallback() user callback is executed.

@endverbatim
  * @{
  */

/**
  * @brief Send an amount of data in blocking mode
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @param  pData  Pointer to data buffer
  * @param  Size   Amount of data to be sent
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_Transmit(SMARTCARD_HandleTypeDef *hsc, uint8_t *pData, uint16_t Size, uint32_t Timeout)
{
  uint8_t *tmp = pData;
  uint32_t tickstart = 0U;

  if(hsc->gState == HAL_SMARTCARD_STATE_READY)
  {
    if((pData == NULL) || (Size == 0U))
    {
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hsc);

    hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;
    hsc->gState = HAL_SMARTCARD_STATE_BUSY_TX;

    /* Init tickstart for timeout management */
    tickstart = HAL_GetTick();

    hsc->TxXferSize = Size;
    hsc->TxXferCount = Size;
    while(hsc->TxXferCount > 0U)
    {
      hsc->TxXferCount--;
      if(SMARTCARD_WaitOnFlagUntilTimeout(hsc, SMARTCARD_FLAG_TXE, RESET, tickstart, Timeout) != HAL_OK)
      {
        return HAL_TIMEOUT;
      }
      hsc->Instance->DR = (uint8_t)(*tmp & 0xFFU);
      tmp++;
    }

    if(SMARTCARD_WaitOnFlagUntilTimeout(hsc, SMARTCARD_FLAG_TC, RESET, tickstart, Timeout) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }

	/* At end of Tx process, restore hsc->gState to Ready */
    hsc->gState = HAL_SMARTCARD_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hsc);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Receive an amount of data in blocking mode
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @param  pData  Pointer to data buffer
  * @param  Size   Amount of data to be received
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_Receive(SMARTCARD_HandleTypeDef *hsc, uint8_t *pData, uint16_t Size, uint32_t Timeout)
{
  uint8_t  *tmp = pData;
  uint32_t tickstart = 0U;

  if(hsc->RxState == HAL_SMARTCARD_STATE_READY)
  {
    if((pData == NULL) || (Size == 0U))
    {
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hsc);

    hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;
    hsc->RxState = HAL_SMARTCARD_STATE_BUSY_RX;

    /* Init tickstart for timeout management */
    tickstart = HAL_GetTick();

    hsc->RxXferSize = Size;
    hsc->RxXferCount = Size;

    /* Check the remain data to be received */
    while(hsc->RxXferCount > 0U)
    {
      hsc->RxXferCount--;
      if(SMARTCARD_WaitOnFlagUntilTimeout(hsc, SMARTCARD_FLAG_RXNE, RESET, tickstart, Timeout) != HAL_OK)
      {
        return HAL_TIMEOUT;
      }
      *tmp = (uint8_t)(hsc->Instance->DR & (uint8_t)0xFFU);
      tmp++;
    }

    /* At end of Rx process, restore hsc->RxState to Ready */
    hsc->RxState = HAL_SMARTCARD_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hsc);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Send an amount of data in non blocking mode
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @param  pData  Pointer to data buffer
  * @param  Size   Amount of data to be sent
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_Transmit_IT(SMARTCARD_HandleTypeDef *hsc, uint8_t *pData, uint16_t Size)
{
  /* Check that a Tx process is not already ongoing */
  if(hsc->gState == HAL_SMARTCARD_STATE_READY)
  {
    if((pData == NULL) || (Size == 0U))
    {
      return HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hsc);

    hsc->pTxBuffPtr = pData;
    hsc->TxXferSize = Size;
    hsc->TxXferCount = Size;

    hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;
    hsc->gState = HAL_SMARTCARD_STATE_BUSY_TX;

    /* Process Unlocked */
    __HAL_UNLOCK(hsc);

    /* Enable the SMARTCARD Parity Error Interrupt */
    SET_BIT(hsc->Instance->CR1, USART_CR1_PEIE);

    /* Disable the SMARTCARD Error Interrupt: (Frame error, noise error, overrun error) */
    CLEAR_BIT(hsc->Instance->CR3, USART_CR3_EIE);

    /* Enable the SMARTCARD Transmit data register empty Interrupt */
    SET_BIT(hsc->Instance->CR1, USART_CR1_TXEIE);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Receive an amount of data in non blocking mode
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @param  pData  Pointer to data buffer
  * @param  Size   Amount of data to be received
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_Receive_IT(SMARTCARD_HandleTypeDef *hsc, uint8_t *pData, uint16_t Size)
{
  /* Check that a Rx process is not already ongoing */
  if(hsc->RxState == HAL_SMARTCARD_STATE_READY)
  {
    if((pData == NULL) || (Size == 0U))
    {
      return HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hsc);

    hsc->pRxBuffPtr = pData;
    hsc->RxXferSize = Size;
    hsc->RxXferCount = Size;

    hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;
    hsc->RxState = HAL_SMARTCARD_STATE_BUSY_RX;

    /* Process Unlocked */
    __HAL_UNLOCK(hsc);

    /* Enable the SMARTCARD Parity Error and Data Register not empty Interrupts */
    SET_BIT(hsc->Instance->CR1, USART_CR1_PEIE| USART_CR1_RXNEIE);

    /* Enable the SMARTCARD Error Interrupt: (Frame error, noise error, overrun error) */
    SET_BIT(hsc->Instance->CR3, USART_CR3_EIE);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Send an amount of data in non blocking mode
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @param  pData  Pointer to data buffer
  * @param  Size   Amount of data to be sent
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_Transmit_DMA(SMARTCARD_HandleTypeDef *hsc, uint8_t *pData, uint16_t Size)
{
  uint32_t *tmp;

  /* Check that a Tx process is not already ongoing */
  if(hsc->gState == HAL_SMARTCARD_STATE_READY)
  {
    if((pData == NULL) || (Size == 0U))
    {
      return HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hsc);

    hsc->pTxBuffPtr = pData;
    hsc->TxXferSize = Size;
    hsc->TxXferCount = Size;

    hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;
    hsc->gState = HAL_SMARTCARD_STATE_BUSY_TX;

    /* Set the SMARTCARD DMA transfer complete callback */
    hsc->hdmatx->XferCpltCallback = SMARTCARD_DMATransmitCplt;

    /* Set the DMA error callback */
    hsc->hdmatx->XferErrorCallback = SMARTCARD_DMAError;

    /* Set the DMA abort callback */
    hsc->hdmatx->XferAbortCallback = NULL;

    /* Enable the SMARTCARD transmit DMA stream */
    tmp = (uint32_t*)&pData;
    HAL_DMA_Start_IT(hsc->hdmatx, *(uint32_t*)tmp, (uint32_t)&hsc->Instance->DR, Size);

     /* Clear the TC flag in the SR register by writing 0 to it */
    __HAL_SMARTCARD_CLEAR_FLAG(hsc, SMARTCARD_FLAG_TC);

    /* Process Unlocked */
    __HAL_UNLOCK(hsc);

    /* Enable the DMA transfer for transmit request by setting the DMAT bit
    in the SMARTCARD CR3 register */
    SET_BIT(hsc->Instance->CR3, USART_CR3_DMAT);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Receive an amount of data in non blocking mode
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @param  pData  Pointer to data buffer
  * @param  Size   Amount of data to be received
  * @note   When the SMARTCARD parity is enabled (PCE = 1) the data received contain the parity bit.s
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMARTCARD_Receive_DMA(SMARTCARD_HandleTypeDef *hsc, uint8_t *pData, uint16_t Size)
{
  uint32_t *tmp;

  /* Check that a Rx process is not already ongoing */
  if(hsc->RxState == HAL_SMARTCARD_STATE_READY)
  {
    if((pData == NULL) || (Size == 0U))
    {
      return HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hsc);

    hsc->pRxBuffPtr = pData;
    hsc->RxXferSize = Size;

    hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;
    hsc->RxState = HAL_SMARTCARD_STATE_BUSY_RX;

    /* Set the SMARTCARD DMA transfer complete callback */
    hsc->hdmarx->XferCpltCallback = SMARTCARD_DMAReceiveCplt;

    /* Set the DMA error callback */
    hsc->hdmarx->XferErrorCallback = SMARTCARD_DMAError;

    /* Set the DMA abort callback */
    hsc->hdmatx->XferAbortCallback = NULL;

    /* Enable the DMA stream */
    tmp = (uint32_t*)&pData;
    HAL_DMA_Start_IT(hsc->hdmarx, (uint32_t)&hsc->Instance->DR, *(uint32_t*)tmp, Size);

    /* Clear the Overrun flag just before enabling the DMA Rx request: can be mandatory for the second transfer */
    __HAL_SMARTCARD_CLEAR_OREFLAG(hsc);

    /* Process Unlocked */
    __HAL_UNLOCK(hsc);

    /* Enable the SMARTCARD Parity Error Interrupt */
    SET_BIT(hsc->Instance->CR1, USART_CR1_PEIE);

    /* Enable the SMARTCARD Error Interrupt: (Frame error, noise error, overrun error) */
    SET_BIT(hsc->Instance->CR3, USART_CR3_EIE);

    /* Enable the DMA transfer for the receiver request by setting the DMAR bit
    in the SMARTCARD CR3 register */
    SET_BIT(hsc->Instance->CR3, USART_CR3_DMAR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Abort ongoing transfers (blocking mode).
  * @param  hsc SMARTCARD handle.
  * @note   This procedure could be used for aborting any ongoing transfer started in Interrupt or DMA mode.
  *         This procedure performs following operations :
  *           - Disable PPP Interrupts
  *           - Disable the DMA transfer in the peripheral register (if enabled)
  *           - Abort DMA transfer by calling HAL_DMA_Abort (in case of transfer in DMA mode)
  *           - Set handle State to READY
  * @note   This procedure is executed in blocking mode : when exiting function, Abort is considered as completed.
  * @retval HAL status
*/
HAL_StatusTypeDef HAL_SMARTCARD_Abort(SMARTCARD_HandleTypeDef *hsc)
{
  /* Disable TXEIE, TCIE, RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_RXNEIE | USART_CR1_PEIE | USART_CR1_TXEIE | USART_CR1_TCIE));
  CLEAR_BIT(hsc->Instance->CR3, USART_CR3_EIE);

  /* Disable the SMARTCARD DMA Tx request if enabled */
  if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAT))
  {
    CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAT);

    /* Abort the SMARTCARD DMA Tx channel : use blocking DMA Abort API (no callback) */
    if(hsc->hdmatx != NULL)
    {
      /* Set the SMARTCARD DMA Abort callback to Null.
         No call back execution at end of DMA abort procedure */
      hsc->hdmatx->XferAbortCallback = NULL;

      HAL_DMA_Abort(hsc->hdmatx);
    }
  }

  /* Disable the SMARTCARD DMA Rx request if enabled */
  if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAR))
  {
    CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAR);

    /* Abort the SMARTCARD DMA Rx channel : use blocking DMA Abort API (no callback) */
    if(hsc->hdmarx != NULL)
    {
      /* Set the SMARTCARD DMA Abort callback to Null.
         No call back execution at end of DMA abort procedure */
      hsc->hdmarx->XferAbortCallback = NULL;

      HAL_DMA_Abort(hsc->hdmarx);
    }
  }

  /* Reset Tx and Rx transfer counters */
  hsc->TxXferCount = 0x00U;
  hsc->RxXferCount = 0x00U;

  /* Reset ErrorCode */
  hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;

  /* Restore hsc->RxState and hsc->gState to Ready */
  hsc->RxState = HAL_SMARTCARD_STATE_READY;
  hsc->gState = HAL_SMARTCARD_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  Abort ongoing Transmit transfer (blocking mode).
  * @param  hsc SMARTCARD handle.
  * @note   This procedure could be used for aborting any ongoing transfer started in Interrupt or DMA mode.
  *         This procedure performs following operations :
  *           - Disable SMARTCARD Interrupts (Tx)
  *           - Disable the DMA transfer in the peripheral register (if enabled)
  *           - Abort DMA transfer by calling HAL_DMA_Abort (in case of transfer in DMA mode)
  *           - Set handle State to READY
  * @note   This procedure is executed in blocking mode : when exiting function, Abort is considered as completed.
  * @retval HAL status
*/
HAL_StatusTypeDef HAL_SMARTCARD_AbortTransmit(SMARTCARD_HandleTypeDef *hsc)
{
  /* Disable TXEIE and TCIE interrupts */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_TXEIE | USART_CR1_TCIE));

  /* Disable the SMARTCARD DMA Tx request if enabled */
  if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAT))
  {
    CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAT);

    /* Abort the SMARTCARD DMA Tx channel : use blocking DMA Abort API (no callback) */
    if(hsc->hdmatx != NULL)
    {
      /* Set the SMARTCARD DMA Abort callback to Null.
         No call back execution at end of DMA abort procedure */
      hsc->hdmatx->XferAbortCallback = NULL;

      HAL_DMA_Abort(hsc->hdmatx);
    }
  }

  /* Reset Tx transfer counter */
  hsc->TxXferCount = 0x00U;

  /* Restore hsc->gState to Ready */
  hsc->gState = HAL_SMARTCARD_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  Abort ongoing Receive transfer (blocking mode).
  * @param  hsc SMARTCARD handle.
  * @note   This procedure could be used for aborting any ongoing transfer started in Interrupt or DMA mode.
  *         This procedure performs following operations :
  *           - Disable PPP Interrupts
  *           - Disable the DMA transfer in the peripheral register (if enabled)
  *           - Abort DMA transfer by calling HAL_DMA_Abort (in case of transfer in DMA mode)
  *           - Set handle State to READY
  * @note   This procedure is executed in blocking mode : when exiting function, Abort is considered as completed.
  * @retval HAL status
*/
HAL_StatusTypeDef HAL_SMARTCARD_AbortReceive(SMARTCARD_HandleTypeDef *hsc)
{
  /* Disable RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_RXNEIE | USART_CR1_PEIE));
  CLEAR_BIT(hsc->Instance->CR3, USART_CR3_EIE);

  /* Disable the SMARTCARD DMA Rx request if enabled */
  if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAR))
  {
    CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAR);

    /* Abort the SMARTCARD DMA Rx channel : use blocking DMA Abort API (no callback) */
    if(hsc->hdmarx != NULL)
    {
      /* Set the SMARTCARD DMA Abort callback to Null.
         No call back execution at end of DMA abort procedure */
      hsc->hdmarx->XferAbortCallback = NULL;

      HAL_DMA_Abort(hsc->hdmarx);
    }
  }

  /* Reset Rx transfer counter */
  hsc->RxXferCount = 0x00U;

  /* Restore hsc->RxState to Ready */
  hsc->RxState = HAL_SMARTCARD_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  Abort ongoing transfers (Interrupt mode).
  * @param  hsc SMARTCARD handle.
  * @note   This procedure could be used for aborting any ongoing transfer started in Interrupt or DMA mode.
  *         This procedure performs following operations :
  *           - Disable PPP Interrupts
  *           - Disable the DMA transfer in the peripheral register (if enabled)
  *           - Abort DMA transfer by calling HAL_DMA_Abort_IT (in case of transfer in DMA mode)
  *           - Set handle State to READY
  *           - At abort completion, call user abort complete callback
  * @note   This procedure is executed in Interrupt mode, meaning that abort procedure could be
  *         considered as completed only when user abort complete callback is executed (not when exiting function).
  * @retval HAL status
*/
HAL_StatusTypeDef HAL_SMARTCARD_Abort_IT(SMARTCARD_HandleTypeDef *hsc)
{
  uint32_t AbortCplt = 0x01U;

  /* Disable TXEIE, TCIE, RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_RXNEIE | USART_CR1_PEIE | USART_CR1_TXEIE | USART_CR1_TCIE));
  CLEAR_BIT(hsc->Instance->CR3, USART_CR3_EIE);

  /* If DMA Tx and/or DMA Rx Handles are associated to SMARTCARD Handle, DMA Abort complete callbacks should be initialised
     before any call to DMA Abort functions */
  /* DMA Tx Handle is valid */
  if(hsc->hdmatx != NULL)
  {
    /* Set DMA Abort Complete callback if SMARTCARD DMA Tx request if enabled.
       Otherwise, set it to NULL */
    if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAT))
    {
      hsc->hdmatx->XferAbortCallback = SMARTCARD_DMATxAbortCallback;
    }
    else
    {
      hsc->hdmatx->XferAbortCallback = NULL;
    }
  }
  /* DMA Rx Handle is valid */
  if(hsc->hdmarx != NULL)
  {
    /* Set DMA Abort Complete callback if SMARTCARD DMA Rx request if enabled.
       Otherwise, set it to NULL */
    if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAR))
    {
      hsc->hdmarx->XferAbortCallback = SMARTCARD_DMARxAbortCallback;
    }
    else
    {
      hsc->hdmarx->XferAbortCallback = NULL;
    }
  }

  /* Disable the SMARTCARD DMA Tx request if enabled */
  if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAT))
  {
    /* Disable DMA Tx at SMARTCARD level */
    CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAT);

    /* Abort the SMARTCARD DMA Tx channel : use non blocking DMA Abort API (callback) */
    if(hsc->hdmatx != NULL)
    {
      /* SMARTCARD Tx DMA Abort callback has already been initialised :
         will lead to call HAL_SMARTCARD_AbortCpltCallback() at end of DMA abort procedure */

      /* Abort DMA TX */
      if(HAL_DMA_Abort_IT(hsc->hdmatx) != HAL_OK)
      {
        hsc->hdmatx->XferAbortCallback = NULL;
      }
      else
      {
        AbortCplt = 0x00U;
      }
    }
  }

  /* Disable the SMARTCARD DMA Rx request if enabled */
  if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAR))
  {
    CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAR);

    /* Abort the SMARTCARD DMA Rx channel : use non blocking DMA Abort API (callback) */
    if(hsc->hdmarx != NULL)
    {
      /* SMARTCARD Rx DMA Abort callback has already been initialised :
         will lead to call HAL_SMARTCARD_AbortCpltCallback() at end of DMA abort procedure */

      /* Abort DMA RX */
      if(HAL_DMA_Abort_IT(hsc->hdmarx) != HAL_OK)
      {
        hsc->hdmarx->XferAbortCallback = NULL;
        AbortCplt = 0x01U;
      }
      else
      {
        AbortCplt = 0x00U;
      }
    }
  }

  /* if no DMA abort complete callback execution is required => call user Abort Complete callback */
  if(AbortCplt == 0x01U)
  {
    /* Reset Tx and Rx transfer counters */
    hsc->TxXferCount = 0x00U;
    hsc->RxXferCount = 0x00U;

    /* Reset ErrorCode */
    hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;

    /* Restore hsc->gState and hsc->RxState to Ready */
    hsc->gState  = HAL_SMARTCARD_STATE_READY;
    hsc->RxState = HAL_SMARTCARD_STATE_READY;

    /* As no DMA to be aborted, call directly user Abort complete callback */
#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
    /* Call registered Abort complete callback */
    hsc->AbortCpltCallback(hsc);
#else
    /* Call legacy weak Abort complete callback */
    HAL_SMARTCARD_AbortCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
  }
  return HAL_OK;
}

/**
  * @brief  Abort ongoing Transmit transfer (Interrupt mode).
  * @param  hsc SMARTCARD handle.
  * @note   This procedure could be used for aborting any ongoing transfer started in Interrupt or DMA mode.
  *         This procedure performs following operations :
  *           - Disable SMARTCARD Interrupts (Tx)
  *           - Disable the DMA transfer in the peripheral register (if enabled)
  *           - Abort DMA transfer by calling HAL_DMA_Abort_IT (in case of transfer in DMA mode)
  *           - Set handle State to READY
  *           - At abort completion, call user abort complete callback
  * @note   This procedure is executed in Interrupt mode, meaning that abort procedure could be
  *         considered as completed only when user abort complete callback is executed (not when exiting function).
  * @retval HAL status
*/
HAL_StatusTypeDef HAL_SMARTCARD_AbortTransmit_IT(SMARTCARD_HandleTypeDef *hsc)
{
  /* Disable TXEIE and TCIE interrupts */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_TXEIE | USART_CR1_TCIE));

  /* Disable the SMARTCARD DMA Tx request if enabled */
  if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAT))
  {
    CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAT);

    /* Abort the SMARTCARD DMA Tx channel : use blocking DMA Abort API (no callback) */
    if(hsc->hdmatx != NULL)
    {
      /* Set the SMARTCARD DMA Abort callback :
         will lead to call HAL_SMARTCARD_AbortCpltCallback() at end of DMA abort procedure */
      hsc->hdmatx->XferAbortCallback = SMARTCARD_DMATxOnlyAbortCallback;

      /* Abort DMA TX */
      if(HAL_DMA_Abort_IT(hsc->hdmatx) != HAL_OK)
      {
        /* Call Directly hsc->hdmatx->XferAbortCallback function in case of error */
        hsc->hdmatx->XferAbortCallback(hsc->hdmatx);
      }
    }
    else
    {
      /* Reset Tx transfer counter */
      hsc->TxXferCount = 0x00U;

      /* Restore hsc->gState to Ready */
      hsc->gState = HAL_SMARTCARD_STATE_READY;

      /* As no DMA to be aborted, call directly user Abort complete callback */
#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
      /* Call registered Abort Transmit Complete Callback */
      hsc->AbortTransmitCpltCallback(hsc);
#else
      /* Call legacy weak Abort Transmit Complete Callback */
      HAL_SMARTCARD_AbortTransmitCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
    }
  }
  else
  {
    /* Reset Tx transfer counter */
    hsc->TxXferCount = 0x00U;

    /* Restore hsc->gState to Ready */
    hsc->gState = HAL_SMARTCARD_STATE_READY;

    /* As no DMA to be aborted, call directly user Abort complete callback */
#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
    /* Call registered Abort Transmit Complete Callback */
    hsc->AbortTransmitCpltCallback(hsc);
#else
    /* Call legacy weak Abort Transmit Complete Callback */
    HAL_SMARTCARD_AbortTransmitCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
  }

  return HAL_OK;
}

/**
  * @brief  Abort ongoing Receive transfer (Interrupt mode).
  * @param  hsc SMARTCARD handle.
  * @note   This procedure could be used for aborting any ongoing transfer started in Interrupt or DMA mode.
  *         This procedure performs following operations :
  *           - Disable SMARTCARD Interrupts (Rx)
  *           - Disable the DMA transfer in the peripheral register (if enabled)
  *           - Abort DMA transfer by calling HAL_DMA_Abort_IT (in case of transfer in DMA mode)
  *           - Set handle State to READY
  *           - At abort completion, call user abort complete callback
  * @note   This procedure is executed in Interrupt mode, meaning that abort procedure could be
  *         considered as completed only when user abort complete callback is executed (not when exiting function).
  * @retval HAL status
*/
HAL_StatusTypeDef HAL_SMARTCARD_AbortReceive_IT(SMARTCARD_HandleTypeDef *hsc)
{
  /* Disable RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_RXNEIE | USART_CR1_PEIE));
  CLEAR_BIT(hsc->Instance->CR3, USART_CR3_EIE);

  /* Disable the SMARTCARD DMA Rx request if enabled */
  if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAR))
  {
    CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAR);

    /* Abort the SMARTCARD DMA Rx channel : use blocking DMA Abort API (no callback) */
    if(hsc->hdmarx != NULL)
    {
      /* Set the SMARTCARD DMA Abort callback :
         will lead to call HAL_SMARTCARD_AbortCpltCallback() at end of DMA abort procedure */
      hsc->hdmarx->XferAbortCallback = SMARTCARD_DMARxOnlyAbortCallback;

      /* Abort DMA RX */
      if(HAL_DMA_Abort_IT(hsc->hdmarx) != HAL_OK)
      {
        /* Call Directly hsc->hdmarx->XferAbortCallback function in case of error */
        hsc->hdmarx->XferAbortCallback(hsc->hdmarx);
      }
    }
    else
    {
      /* Reset Rx transfer counter */
      hsc->RxXferCount = 0x00U;

      /* Restore hsc->RxState to Ready */
      hsc->RxState = HAL_SMARTCARD_STATE_READY;

      /* As no DMA to be aborted, call directly user Abort complete callback */
#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
      /* Call registered Abort Receive Complete Callback */
      hsc->AbortReceiveCpltCallback(hsc);
#else
      /* Call legacy weak Abort Receive Complete Callback */
      HAL_SMARTCARD_AbortReceiveCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
    }
  }
  else
  {
    /* Reset Rx transfer counter */
    hsc->RxXferCount = 0x00U;

    /* Restore hsc->RxState to Ready */
    hsc->RxState = HAL_SMARTCARD_STATE_READY;

    /* As no DMA to be aborted, call directly user Abort complete callback */
#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
    /* Call registered Abort Receive Complete Callback */
    hsc->AbortReceiveCpltCallback(hsc);
#else
    /* Call legacy weak Abort Receive Complete Callback */
    HAL_SMARTCARD_AbortReceiveCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
  }

  return HAL_OK;
}

/**
  * @brief This function handles SMARTCARD interrupt request.
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval None
  */
void HAL_SMARTCARD_IRQHandler(SMARTCARD_HandleTypeDef *hsc)
{
  uint32_t isrflags   = READ_REG(hsc->Instance->SR);
  uint32_t cr1its     = READ_REG(hsc->Instance->CR1);
  uint32_t cr3its     = READ_REG(hsc->Instance->CR3);
  uint32_t dmarequest = 0x00U;
  uint32_t errorflags = 0x00U;

  /* If no error occurs */
  errorflags = (isrflags & (uint32_t)(USART_SR_PE | USART_SR_FE | USART_SR_ORE | USART_SR_NE));
  if(errorflags == RESET)
  {
    /* SMARTCARD in mode Receiver -------------------------------------------------*/
    if(((isrflags & USART_SR_RXNE) != RESET) && ((cr1its & USART_CR1_RXNEIE) != RESET))
    {
      SMARTCARD_Receive_IT(hsc);
      return;
    }
  }

  /* If some errors occur */
  if((errorflags != RESET) && (((cr3its & USART_CR3_EIE) != RESET) || ((cr1its & (USART_CR1_RXNEIE | USART_CR1_PEIE)) != RESET)))
  {
    /* SMARTCARD parity error interrupt occurred ---------------------------*/
    if(((isrflags & SMARTCARD_FLAG_PE) != RESET) && ((cr1its & USART_CR1_PEIE) != RESET))
    {
      hsc->ErrorCode |= HAL_SMARTCARD_ERROR_PE;
    }

    /* SMARTCARD frame error interrupt occurred ----------------------------*/
    if(((isrflags & SMARTCARD_FLAG_FE) != RESET) && ((cr3its & USART_CR3_EIE) != RESET))
    {
      hsc->ErrorCode |= HAL_SMARTCARD_ERROR_FE;
    }

    /* SMARTCARD noise error interrupt occurred ----------------------------*/
    if(((isrflags & SMARTCARD_FLAG_NE) != RESET) && ((cr3its & USART_CR3_EIE) != RESET))
    {
      hsc->ErrorCode |= HAL_SMARTCARD_ERROR_NE;
    }

    /* SMARTCARD Over-Run interrupt occurred -------------------------------*/
    if(((isrflags & SMARTCARD_FLAG_ORE) != RESET) && (((cr1its & USART_CR1_RXNEIE) != RESET) || ((cr3its & USART_CR3_EIE) != RESET)))
    {
      hsc->ErrorCode |= HAL_SMARTCARD_ERROR_ORE;
    }
    /* Call the Error call Back in case of Errors --------------------------*/
    if(hsc->ErrorCode != HAL_SMARTCARD_ERROR_NONE)
    {
      /* SMARTCARD in mode Receiver ----------------------------------------*/
      if(((isrflags & USART_SR_RXNE) != RESET) && ((cr1its & USART_CR1_RXNEIE) != RESET))
      {
        SMARTCARD_Receive_IT(hsc);
      }

      /* If Overrun error occurs, or if any error occurs in DMA mode reception,
         consider error as blocking */
      dmarequest = HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAR);
      if(((hsc->ErrorCode & HAL_SMARTCARD_ERROR_ORE) != RESET) || dmarequest)
      {
        /* Blocking error : transfer is aborted
          Set the SMARTCARD state ready to be able to start again the process,
          Disable Rx Interrupts, and disable Rx DMA request, if ongoing */
        SMARTCARD_EndRxTransfer(hsc);
        /* Disable the SMARTCARD DMA Rx request if enabled */
        if(HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAR))
        {
          CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAR);

          /* Abort the SMARTCARD DMA Rx channel */
          if(hsc->hdmarx != NULL)
          {
            /* Set the SMARTCARD DMA Abort callback :
              will lead to call HAL_SMARTCARD_ErrorCallback() at end of DMA abort procedure */
            hsc->hdmarx->XferAbortCallback = SMARTCARD_DMAAbortOnError;

           if(HAL_DMA_Abort_IT(hsc->hdmarx) != HAL_OK)
            {
              /* Call Directly XferAbortCallback function in case of error */
              hsc->hdmarx->XferAbortCallback(hsc->hdmarx);
            }
          }
          else
          {
#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
            /* Call registered user error callback */
            hsc->ErrorCallback(hsc);
#else
            /* Call legacy weak user error callback */
            HAL_SMARTCARD_ErrorCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
          }
        }
        else
        {
#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
          /* Call registered user error callback */
          hsc->ErrorCallback(hsc);
#else
          /* Call legacy weak user error callback */
          HAL_SMARTCARD_ErrorCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
        }
      }
      else
      {
        /* Non Blocking error : transfer could go on.
           Error is notified to user through user error callback */
#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
        /* Call registered user error callback */
        hsc->ErrorCallback(hsc);
#else
        /* Call legacy weak user error callback */
        HAL_SMARTCARD_ErrorCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
        hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;
      }
    }
    return;
  } /* End if some error occurs */

  /* SMARTCARD in mode Transmitter ------------------------------------------*/
  if(((isrflags & SMARTCARD_FLAG_TXE) != RESET) && ((cr1its & USART_CR1_TXEIE) != RESET))
  {
    SMARTCARD_Transmit_IT(hsc);
    return;
  }

  /* SMARTCARD in mode Transmitter (transmission end) -----------------------*/
  if(((isrflags & SMARTCARD_FLAG_TC) != RESET) && ((cr1its & USART_CR1_TCIE) != RESET))
  {
    SMARTCARD_EndTransmit_IT(hsc);
    return;
  }
}

/**
  * @brief Tx Transfer completed callbacks
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval None
  */
__weak void HAL_SMARTCARD_TxCpltCallback(SMARTCARD_HandleTypeDef *hsc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMARTCARD_TxCpltCallback can be implemented in the user file.
   */
}

/**
  * @brief Rx Transfer completed callback
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval None
  */
__weak void HAL_SMARTCARD_RxCpltCallback(SMARTCARD_HandleTypeDef *hsc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMARTCARD_RxCpltCallback can be implemented in the user file.
   */
}

/**
  * @brief SMARTCARD error callback
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval None
  */
__weak void HAL_SMARTCARD_ErrorCallback(SMARTCARD_HandleTypeDef *hsc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMARTCARD_ErrorCallback can be implemented in the user file.
   */
}

/**
  * @brief  SMARTCARD Abort Complete callback.
  * @param  hsc SMARTCARD handle.
  * @retval None
  */
__weak void HAL_SMARTCARD_AbortCpltCallback (SMARTCARD_HandleTypeDef *hsc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMARTCARD_AbortCpltCallback can be implemented in the user file.
   */
}

/**
  * @brief  SMARTCARD Abort Transmit Complete callback.
  * @param  hsc SMARTCARD handle.
  * @retval None
  */
__weak void HAL_SMARTCARD_AbortTransmitCpltCallback (SMARTCARD_HandleTypeDef *hsc)
{
    /* Prevent unused argument(s) compilation warning */
    UNUSED(hsc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMARTCARD_AbortTransmitCpltCallback can be implemented in the user file.
   */
}

/**
  * @brief  SMARTCARD Abort Receive Complete callback.
  * @param  hsc SMARTCARD handle.
  * @retval None
  */
__weak void HAL_SMARTCARD_AbortReceiveCpltCallback (SMARTCARD_HandleTypeDef *hsc)
{
    /* Prevent unused argument(s) compilation warning */
    UNUSED(hsc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMARTCARD_AbortReceiveCpltCallback can be implemented in the user file.
   */
}

/**
  * @}
  */

/** @defgroup SMARTCARD_Exported_Functions_Group3 Peripheral State and Errors functions
  *  @brief   SMARTCARD State and Errors functions
  *
@verbatim
 ===============================================================================
                ##### Peripheral State and Errors functions #####
 ===============================================================================
    [..]
    This subsection provides a set of functions allowing to control the SmartCard.
     (+) HAL_SMARTCARD_GetState() API can be helpful to check in run-time the state of the SmartCard peripheral.
     (+) HAL_SMARTCARD_GetError() check in run-time errors that could be occurred during communication.
@endverbatim
  * @{
  */

/**
  * @brief Return the SMARTCARD handle state
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval HAL state
  */
HAL_SMARTCARD_StateTypeDef HAL_SMARTCARD_GetState(SMARTCARD_HandleTypeDef *hsc)
{
  uint32_t temp1= 0x00U, temp2 = 0x00U;
  temp1 = hsc->gState;
  temp2 = hsc->RxState;

  return (HAL_SMARTCARD_StateTypeDef)(temp1 | temp2);
}

/**
  * @brief  Return the SMARTCARD error code
  * @param  hsc  Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *              the configuration information for the specified SMARTCARD.
  * @retval SMARTCARD Error Code
  */
uint32_t HAL_SMARTCARD_GetError(SMARTCARD_HandleTypeDef *hsc)
{
  return hsc->ErrorCode;
}

/**
  * @}
  */

/**
  * @}
  */

/** @defgroup SMARTCARD_Private_Functions SMARTCARD Private Functions
  * @{
  */

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
/**
  * @brief  Initialize the callbacks to their default values.
  * @param  hsc SMARTCARD handle.
  * @retval none
  */
void SMARTCARD_InitCallbacksToDefault(SMARTCARD_HandleTypeDef *hsc)
{
  /* Init the SMARTCARD Callback settings */
  hsc->TxCpltCallback            = HAL_SMARTCARD_TxCpltCallback;            /* Legacy weak TxCpltCallback            */
  hsc->RxCpltCallback            = HAL_SMARTCARD_RxCpltCallback;            /* Legacy weak RxCpltCallback            */
  hsc->ErrorCallback             = HAL_SMARTCARD_ErrorCallback;             /* Legacy weak ErrorCallback             */
  hsc->AbortCpltCallback         = HAL_SMARTCARD_AbortCpltCallback;         /* Legacy weak AbortCpltCallback         */
  hsc->AbortTransmitCpltCallback = HAL_SMARTCARD_AbortTransmitCpltCallback; /* Legacy weak AbortTransmitCpltCallback */
  hsc->AbortReceiveCpltCallback  = HAL_SMARTCARD_AbortReceiveCpltCallback;  /* Legacy weak AbortReceiveCpltCallback  */

}
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACKS */

/**
  * @brief DMA SMARTCARD transmit process complete callback
  * @param  hdma   Pointer to a DMA_HandleTypeDef structure that contains
  *                the configuration information for the specified DMA module.
  * @retval None
  */
static void SMARTCARD_DMATransmitCplt(DMA_HandleTypeDef *hdma)
{
  SMARTCARD_HandleTypeDef* hsc = ( SMARTCARD_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;

  hsc->TxXferCount = 0U;

  /* Disable the DMA transfer for transmit request by setting the DMAT bit
     in the USART CR3 register */
  CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAT);

  /* Enable the SMARTCARD Transmit Complete Interrupt */
  SET_BIT(hsc->Instance->CR1, USART_CR1_TCIE);
}

/**
  * @brief DMA SMARTCARD receive process complete callback
  * @param  hdma   Pointer to a DMA_HandleTypeDef structure that contains
  *                the configuration information for the specified DMA module.
  * @retval None
  */
static void SMARTCARD_DMAReceiveCplt(DMA_HandleTypeDef *hdma)
{
  SMARTCARD_HandleTypeDef* hsc = ( SMARTCARD_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;

  hsc->RxXferCount = 0U;

  /* Disable RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_RXNEIE | USART_CR1_PEIE));
  CLEAR_BIT(hsc->Instance->CR3, USART_CR3_EIE);

  /* Disable the DMA transfer for the receiver request by setting the DMAR bit
     in the USART CR3 register */
  CLEAR_BIT(hsc->Instance->CR3, USART_CR3_DMAR);

  /* At end of Rx process, restore hsc->RxState to Ready */
  hsc->RxState = HAL_SMARTCARD_STATE_READY;

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
  /* Call registered Rx complete callback */
  hsc->RxCpltCallback(hsc);
#else
  /* Call legacy weak Rx complete callback */
  HAL_SMARTCARD_RxCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
}

/**
  * @brief DMA SMARTCARD communication error callback
  * @param  hdma   Pointer to a DMA_HandleTypeDef structure that contains
  *                the configuration information for the specified DMA module.
  * @retval None
  */
static void SMARTCARD_DMAError(DMA_HandleTypeDef *hdma)
{
  uint32_t dmarequest = 0x00U;
  SMARTCARD_HandleTypeDef* hsc = ( SMARTCARD_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;
  hsc->RxXferCount = 0U;
  hsc->TxXferCount = 0U;
  hsc->ErrorCode = HAL_SMARTCARD_ERROR_DMA;

  /* Stop SMARTCARD DMA Tx request if ongoing */
  dmarequest = HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAT);
  if((hsc->gState == HAL_SMARTCARD_STATE_BUSY_TX) && dmarequest)
  {
    SMARTCARD_EndTxTransfer(hsc);
  }

  /* Stop SMARTCARD DMA Rx request if ongoing */
  dmarequest = HAL_IS_BIT_SET(hsc->Instance->CR3, USART_CR3_DMAR);
  if((hsc->RxState == HAL_SMARTCARD_STATE_BUSY_RX) && dmarequest)
  {
    SMARTCARD_EndRxTransfer(hsc);
  }

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
  /* Call registered user error callback */
  hsc->ErrorCallback(hsc);
#else
  /* Call legacy weak user error callback */
  HAL_SMARTCARD_ErrorCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
}

/**
  * @brief  This function handles SMARTCARD Communication Timeout.
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @param  Flag   Specifies the SMARTCARD flag to check.
  * @param  Status The new Flag status (SET or RESET).
  * @param  Timeout Timeout duration
  * @param  Tickstart Tick start value
  * @retval HAL status
  */
static HAL_StatusTypeDef SMARTCARD_WaitOnFlagUntilTimeout(SMARTCARD_HandleTypeDef *hsc, uint32_t Flag, FlagStatus Status, uint32_t Tickstart, uint32_t Timeout)
{
  /* Wait until flag is set */
  while((__HAL_SMARTCARD_GET_FLAG(hsc, Flag) ? SET : RESET) == Status)
  {
    /* Check for the Timeout */
    if(Timeout != HAL_MAX_DELAY)
    {
      if((Timeout == 0U)||((HAL_GetTick() - Tickstart ) > Timeout))
      {
        /* Disable TXE and RXNE interrupts for the interrupt process */
        CLEAR_BIT(hsc->Instance->CR1, USART_CR1_TXEIE);
        CLEAR_BIT(hsc->Instance->CR1, USART_CR1_RXNEIE);

        hsc->gState= HAL_SMARTCARD_STATE_READY;
        hsc->RxState= HAL_SMARTCARD_STATE_READY;

        /* Process Unlocked */
        __HAL_UNLOCK(hsc);

        return HAL_TIMEOUT;
      }
    }
  }
  return HAL_OK;
}

/**
  * @brief  End ongoing Tx transfer on SMARTCARD peripheral (following error detection or Transmit completion).
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval None
  */
static void SMARTCARD_EndTxTransfer(SMARTCARD_HandleTypeDef *hsc)
{
  /* At end of Tx process, restore hsc->gState to Ready */
  hsc->gState = HAL_SMARTCARD_STATE_READY;

  /* Disable TXEIE and TCIE interrupts */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_TXEIE | USART_CR1_TCIE));
}


/**
  * @brief  End ongoing Rx transfer on SMARTCARD peripheral (following error detection or Reception completion).
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval None
  */
static void SMARTCARD_EndRxTransfer(SMARTCARD_HandleTypeDef *hsc)
{
  /* At end of Rx process, restore hsc->RxState to Ready */
  hsc->RxState = HAL_SMARTCARD_STATE_READY;

  /* Disable RXNE, PE and ERR (Frame error, noise error, overrun error) interrupts */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_RXNEIE | USART_CR1_PEIE));
  CLEAR_BIT(hsc->Instance->CR3, USART_CR3_EIE);
}

/**
  * @brief Send an amount of data in non blocking mode
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval HAL status
  */
static HAL_StatusTypeDef SMARTCARD_Transmit_IT(SMARTCARD_HandleTypeDef *hsc)
{

  /* Check that a Tx process is ongoing */
  if(hsc->gState == HAL_SMARTCARD_STATE_BUSY_TX)
  {
    hsc->Instance->DR = (uint8_t)(*hsc->pTxBuffPtr & 0xFFU);
    hsc->pTxBuffPtr++;

    if(--hsc->TxXferCount == 0U)
    {
      /* Disable the SMARTCARD Transmit data register empty Interrupt */
      CLEAR_BIT(hsc->Instance->CR1, USART_CR1_TXEIE);

      /* Enable the SMARTCARD Transmit Complete Interrupt */
      SET_BIT(hsc->Instance->CR1, USART_CR1_TCIE);
    }

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Wraps up transmission in non blocking mode.
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for the specified SMARTCARD module.
  * @retval HAL status
  */
static HAL_StatusTypeDef SMARTCARD_EndTransmit_IT(SMARTCARD_HandleTypeDef *hsc)
{
  /* Disable the SMARTCARD Transmit Complete Interrupt */
  CLEAR_BIT(hsc->Instance->CR1, USART_CR1_TCIE);

  /* Disable the SMARTCARD Error Interrupt: (Frame error, noise error, overrun error) */
  CLEAR_BIT(hsc->Instance->CR3, USART_CR3_EIE);

  /* Tx process is ended, restore hsc->gState to Ready */
  hsc->gState = HAL_SMARTCARD_STATE_READY;

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
  /* Call registered Tx complete callback */
  hsc->TxCpltCallback(hsc);
#else
  /* Call legacy weak Tx complete callback */
  HAL_SMARTCARD_TxCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */

  return HAL_OK;
}

/**
  * @brief Receive an amount of data in non blocking mode
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval HAL status
  */
static HAL_StatusTypeDef SMARTCARD_Receive_IT(SMARTCARD_HandleTypeDef *hsc)
{

  /* Check that a Rx process is ongoing */
  if(hsc->RxState == HAL_SMARTCARD_STATE_BUSY_RX)
  {
    *hsc->pRxBuffPtr = (uint8_t)(hsc->Instance->DR & (uint8_t)0xFFU);
    hsc->pRxBuffPtr++;

    if(--hsc->RxXferCount == 0U)
    {
      CLEAR_BIT(hsc->Instance->CR1, USART_CR1_RXNEIE);

      /* Disable the SMARTCARD Parity Error Interrupt */
      CLEAR_BIT(hsc->Instance->CR1, USART_CR1_PEIE);

      /* Disable the SMARTCARD Error Interrupt: (Frame error, noise error, overrun error) */
      CLEAR_BIT(hsc->Instance->CR3, USART_CR3_EIE);

      /* Rx process is completed, restore hsc->RxState to Ready */
      hsc->RxState = HAL_SMARTCARD_STATE_READY;

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
      /* Call registered Rx complete callback */
      hsc->RxCpltCallback(hsc);
#else
      /* Call legacy weak Rx complete callback */
      HAL_SMARTCARD_RxCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */

      return HAL_OK;
    }
    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  DMA SMARTCARD communication abort callback, when initiated by HAL services on Error
  *         (To be called at end of DMA Abort procedure following error occurrence).
  * @param  hdma DMA handle.
  * @retval None
  */
static void SMARTCARD_DMAAbortOnError(DMA_HandleTypeDef *hdma)
{
  SMARTCARD_HandleTypeDef* hsc = (SMARTCARD_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;
  hsc->RxXferCount = 0x00U;
  hsc->TxXferCount = 0x00U;

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
  /* Call registered user error callback */
  hsc->ErrorCallback(hsc);
#else
  /* Call legacy weak user error callback */
  HAL_SMARTCARD_ErrorCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
}

/**
  * @brief  DMA SMARTCARD Tx communication abort callback, when initiated by user
  *         (To be called at end of DMA Tx Abort procedure following user abort request).
  * @note   When this callback is executed, User Abort complete call back is called only if no
  *         Abort still ongoing for Rx DMA Handle.
  * @param  hdma DMA handle.
  * @retval None
  */
static void SMARTCARD_DMATxAbortCallback(DMA_HandleTypeDef *hdma)
{
  SMARTCARD_HandleTypeDef* hsc = ( SMARTCARD_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;

  hsc->hdmatx->XferAbortCallback = NULL;

  /* Check if an Abort process is still ongoing */
  if(hsc->hdmarx != NULL)
  {
    if(hsc->hdmarx->XferAbortCallback != NULL)
    {
      return;
    }
  }

  /* No Abort process still ongoing : All DMA channels are aborted, call user Abort Complete callback */
  hsc->TxXferCount = 0x00U;
  hsc->RxXferCount = 0x00U;

  /* Reset ErrorCode */
  hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;

  /* Restore hsc->gState and hsc->RxState to Ready */
  hsc->gState  = HAL_SMARTCARD_STATE_READY;
  hsc->RxState = HAL_SMARTCARD_STATE_READY;

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
  /* Call registered Abort complete callback */
  hsc->AbortCpltCallback(hsc);
#else
  /* Call legacy weak Abort complete callback */
  HAL_SMARTCARD_AbortCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
}

/**
  * @brief  DMA SMARTCARD Rx communication abort callback, when initiated by user
  *         (To be called at end of DMA Rx Abort procedure following user abort request).
  * @note   When this callback is executed, User Abort complete call back is called only if no
  *         Abort still ongoing for Tx DMA Handle.
  * @param  hdma DMA handle.
  * @retval None
  */
static void SMARTCARD_DMARxAbortCallback(DMA_HandleTypeDef *hdma)
{
  SMARTCARD_HandleTypeDef* hsc = ( SMARTCARD_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;

  hsc->hdmarx->XferAbortCallback = NULL;

  /* Check if an Abort process is still ongoing */
  if(hsc->hdmatx != NULL)
  {
    if(hsc->hdmatx->XferAbortCallback != NULL)
    {
      return;
    }
  }

  /* No Abort process still ongoing : All DMA channels are aborted, call user Abort Complete callback */
  hsc->TxXferCount = 0x00U;
  hsc->RxXferCount = 0x00U;

  /* Reset ErrorCode */
  hsc->ErrorCode = HAL_SMARTCARD_ERROR_NONE;

  /* Restore hsc->gState and hsc->RxState to Ready */
  hsc->gState  = HAL_SMARTCARD_STATE_READY;
  hsc->RxState = HAL_SMARTCARD_STATE_READY;

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
  /* Call registered Abort complete callback */
  hsc->AbortCpltCallback(hsc);
#else
  /* Call legacy weak Abort complete callback */
  HAL_SMARTCARD_AbortCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
}

/**
  * @brief  DMA SMARTCARD Tx communication abort callback, when initiated by user by a call to
  *         HAL_SMARTCARD_AbortTransmit_IT API (Abort only Tx transfer)
  *         (This callback is executed at end of DMA Tx Abort procedure following user abort request,
  *         and leads to user Tx Abort Complete callback execution).
  * @param  hdma DMA handle.
  * @retval None
  */
static void SMARTCARD_DMATxOnlyAbortCallback(DMA_HandleTypeDef *hdma)
{
  SMARTCARD_HandleTypeDef* hsc = ( SMARTCARD_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;

  hsc->TxXferCount = 0x00U;

  /* Restore hsc->gState to Ready */
  hsc->gState = HAL_SMARTCARD_STATE_READY;

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
  /* Call registered Abort Transmit Complete Callback */
  hsc->AbortTransmitCpltCallback(hsc);
#else
  /* Call legacy weak Abort Transmit Complete Callback */
  HAL_SMARTCARD_AbortTransmitCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
}

/**
  * @brief  DMA SMARTCARD Rx communication abort callback, when initiated by user by a call to
  *         HAL_SMARTCARD_AbortReceive_IT API (Abort only Rx transfer)
  *         (This callback is executed at end of DMA Rx Abort procedure following user abort request,
  *         and leads to user Rx Abort Complete callback execution).
  * @param  hdma DMA handle.
  * @retval None
  */
static void SMARTCARD_DMARxOnlyAbortCallback(DMA_HandleTypeDef *hdma)
{
  SMARTCARD_HandleTypeDef* hsc = ( SMARTCARD_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;

  hsc->RxXferCount = 0x00U;

  /* Restore hsc->RxState to Ready */
  hsc->RxState = HAL_SMARTCARD_STATE_READY;

#if (USE_HAL_SMARTCARD_REGISTER_CALLBACKS == 1)
  /* Call registered Abort Receive Complete Callback */
  hsc->AbortReceiveCpltCallback(hsc);
#else
  /* Call legacy weak Abort Receive Complete Callback */
  HAL_SMARTCARD_AbortReceiveCpltCallback(hsc);
#endif /* USE_HAL_SMARTCARD_REGISTER_CALLBACK */
}

/**
  * @brief Configure the SMARTCARD peripheral
  * @param  hsc    Pointer to a SMARTCARD_HandleTypeDef structure that contains
  *                the configuration information for SMARTCARD module.
  * @retval None
  */
static void SMARTCARD_SetConfig(SMARTCARD_HandleTypeDef *hsc)
{
  uint32_t tmpreg = 0x00U;
  uint32_t pclk;

  /* Check the parameters */
  assert_param(IS_SMARTCARD_INSTANCE(hsc->Instance));
  assert_param(IS_SMARTCARD_POLARITY(hsc->Init.CLKPolarity));
  assert_param(IS_SMARTCARD_PHASE(hsc->Init.CLKPhase));
  assert_param(IS_SMARTCARD_LASTBIT(hsc->Init.CLKLastBit));
  assert_param(IS_SMARTCARD_BAUDRATE(hsc->Init.BaudRate));
  assert_param(IS_SMARTCARD_WORD_LENGTH(hsc->Init.WordLength));
  assert_param(IS_SMARTCARD_STOPBITS(hsc->Init.StopBits));
  assert_param(IS_SMARTCARD_PARITY(hsc->Init.Parity));
  assert_param(IS_SMARTCARD_MODE(hsc->Init.Mode));
  assert_param(IS_SMARTCARD_NACK_STATE(hsc->Init.NACKState));

  /* The LBCL, CPOL and CPHA bits have to be selected when both the transmitter and the
     receiver are disabled (TE=RE=0) to ensure that the clock pulses function correctly. */
  CLEAR_BIT(hsc->Instance->CR1, (USART_CR1_TE | USART_CR1_RE));

  /*---------------------------- USART CR2 Configuration ---------------------*/
  tmpreg = hsc->Instance->CR2;
  /* Clear CLKEN, CPOL, CPHA and LBCL bits */
  tmpreg &= (uint32_t)~((uint32_t)(USART_CR2_CPHA | USART_CR2_CPOL | USART_CR2_CLKEN | USART_CR2_LBCL));
  /* Configure the SMARTCARD Clock, CPOL, CPHA and LastBit -----------------------*/
  /* Set CPOL bit according to hsc->Init.CLKPolarity value */
  /* Set CPHA bit according to hsc->Init.CLKPhase value */
  /* Set LBCL bit according to hsc->Init.CLKLastBit value */
  /* Set Stop Bits: Set STOP[13:12] bits according to hsc->Init.StopBits value */
  tmpreg |= (uint32_t)(USART_CR2_CLKEN | hsc->Init.CLKPolarity |
                      hsc->Init.CLKPhase| hsc->Init.CLKLastBit | hsc->Init.StopBits);
  /* Write to USART CR2 */
  WRITE_REG(hsc->Instance->CR2, (uint32_t)tmpreg);

  tmpreg = hsc->Instance->CR2;

  /* Clear STOP[13:12] bits */
  tmpreg &= (uint32_t)~((uint32_t)USART_CR2_STOP);

  /* Set Stop Bits: Set STOP[13:12] bits according to hsc->Init.StopBits value */
  tmpreg |= (uint32_t)(hsc->Init.StopBits);

  /* Write to USART CR2 */
  WRITE_REG(hsc->Instance->CR2, (uint32_t)tmpreg);

  /*-------------------------- USART CR1 Configuration -----------------------*/
  tmpreg = hsc->Instance->CR1;

  /* Clear M, PCE, PS, TE and RE bits */
  tmpreg &= (uint32_t)~((uint32_t)(USART_CR1_M | USART_CR1_PCE | USART_CR1_PS | USART_CR1_TE | \
                                   USART_CR1_RE));

  /* Configure the SMARTCARD Word Length, Parity and mode:
     Set the M bits according to hsc->Init.WordLength value
     Set PCE and PS bits according to hsc->Init.Parity value
     Set TE and RE bits according to hsc->Init.Mode value */
  tmpreg |= (uint32_t)hsc->Init.WordLength | hsc->Init.Parity | hsc->Init.Mode;

  /* Write to USART CR1 */
  WRITE_REG(hsc->Instance->CR1, (uint32_t)tmpreg);

  /*-------------------------- USART CR3 Configuration -----------------------*/
  /* Clear CTSE and RTSE bits */
  CLEAR_BIT(hsc->Instance->CR3, (USART_CR3_RTSE | USART_CR3_CTSE));

  /*-------------------------- USART BRR Configuration -----------------------*/
#if defined(USART6)
  if((hsc->Instance == USART1) || (hsc->Instance == USART6))
  {
    pclk = HAL_RCC_GetPCLK2Freq();
    hsc->Instance->BRR = SMARTCARD_BRR(pclk, hsc->Init.BaudRate);
  }
#else
  if(hsc->Instance == USART1)
  {
    pclk = HAL_RCC_GetPCLK2Freq();
    hsc->Instance->BRR = SMARTCARD_BRR(pclk, hsc->Init.BaudRate);
  }
#endif /* USART6 */
  else
  {
    pclk = HAL_RCC_GetPCLK1Freq();
    hsc->Instance->BRR = SMARTCARD_BRR(pclk, hsc->Init.BaudRate);
  }
}

/**
  * @}
  */

#endif /* HAL_SMARTCARD_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
