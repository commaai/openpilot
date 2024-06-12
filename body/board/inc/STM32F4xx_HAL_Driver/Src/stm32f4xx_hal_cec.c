/**
  ******************************************************************************
  * @file    stm32f4xx_hal_cec.c
  * @author  MCD Application Team
  * @brief   CEC HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the High Definition Multimedia Interface
  *          Consumer Electronics Control Peripheral (CEC).
  *           + Initialization and de-initialization function
  *           + IO operation function
  *           + Peripheral Control function
  *
  *
  @verbatim
 ===============================================================================
                        ##### How to use this driver #####
 ===============================================================================
    [..]
    The CEC HAL driver can be used as follow:

    (#) Declare a CEC_HandleTypeDef handle structure.
    (#) Initialize the CEC low level resources by implementing the HAL_CEC_MspInit ()API:
        (##) Enable the CEC interface clock.
        (##) CEC pins configuration:
            (+++) Enable the clock for the CEC GPIOs.
            (+++) Configure these CEC pins as alternate function pull-up.
        (##) NVIC configuration if you need to use interrupt process (HAL_CEC_Transmit_IT()
             and HAL_CEC_Receive_IT() APIs):
            (+++) Configure the CEC interrupt priority.
            (+++) Enable the NVIC CEC IRQ handle.
            (+++) The specific CEC interrupts (Transmission complete interrupt,
                  RXNE interrupt and Error Interrupts) will be managed using the macros
                  __HAL_CEC_ENABLE_IT() and __HAL_CEC_DISABLE_IT() inside the transmit
                  and receive process.

    (#) Program the Signal Free Time (SFT) and SFT option, Tolerance, reception stop in
        in case of Bit Rising Error, Error-Bit generation conditions, device logical
        address and Listen mode in the hcec Init structure.

    (#) Initialize the CEC registers by calling the HAL_CEC_Init() API.

  [..]
    (@) This API (HAL_CEC_Init()) configures also the low level Hardware (GPIO, CLOCK, CORTEX...etc)
        by calling the customed HAL_CEC_MspInit() API.
  *** Callback registration ***
  =============================================

  The compilation define  USE_HAL_CEC_REGISTER_CALLBACKS when set to 1
  allows the user to configure dynamically the driver callbacks.
  Use Functions HAL_CEC_RegisterCallback() or HAL_CEC_RegisterXXXCallback()
  to register an interrupt callback.

  Function HAL_CEC_RegisterCallback() allows to register following callbacks:
    (+) TxCpltCallback     : Tx Transfer completed callback.
    (+) ErrorCallback      : callback for error detection.
    (+) MspInitCallback    : CEC MspInit.
    (+) MspDeInitCallback  : CEC MspDeInit.
  This function takes as parameters the HAL peripheral handle, the Callback ID
  and a pointer to the user callback function.

  For specific callback HAL_CEC_RxCpltCallback use dedicated register callbacks
  HAL_CEC_RegisterRxCpltCallback().

  Use function HAL_CEC_UnRegisterCallback() to reset a callback to the default
  weak function.
  HAL_CEC_UnRegisterCallback() takes as parameters the HAL peripheral handle,
  and the Callback ID.
  This function allows to reset following callbacks:
    (+) TxCpltCallback     : Tx Transfer completed callback.
    (+) ErrorCallback      : callback for error detection.
    (+) MspInitCallback    : CEC MspInit.
    (+) MspDeInitCallback  : CEC MspDeInit.

  For callback HAL_CEC_RxCpltCallback use dedicated unregister callback :
  HAL_CEC_UnRegisterRxCpltCallback().

  By default, after the HAL_CEC_Init() and when the state is HAL_CEC_STATE_RESET
  all callbacks are set to the corresponding weak functions :
  examples HAL_CEC_TxCpltCallback() , HAL_CEC_RxCpltCallback().
  Exception done for MspInit and MspDeInit functions that are
  reset to the legacy weak function in the HAL_CEC_Init()/ HAL_CEC_DeInit() only when
  these callbacks are null (not registered beforehand).
  if not, MspInit or MspDeInit are not null, the HAL_CEC_Init() / HAL_CEC_DeInit()
  keep and use the user MspInit/MspDeInit functions (registered beforehand)

  Callbacks can be registered/unregistered in HAL_CEC_STATE_READY state only.
  Exception done MspInit/MspDeInit callbacks that can be registered/unregistered
  in HAL_CEC_STATE_READY or HAL_CEC_STATE_RESET state,
  thus registered (user) MspInit/DeInit callbacks can be used during the Init/DeInit.
  In that case first register the MspInit/MspDeInit user callbacks
  using HAL_CEC_RegisterCallback() before calling HAL_CEC_DeInit()
  or HAL_CEC_Init() function.

  When the compilation define USE_HAL_CEC_REGISTER_CALLBACKS is set to 0 or
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

/** @defgroup CEC CEC
  * @brief HAL CEC module driver
  * @{
  */
#ifdef HAL_CEC_MODULE_ENABLED
#if defined (CEC)

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @defgroup CEC_Private_Constants CEC Private Constants
  * @{
  */
/**
  * @}
  */

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/** @defgroup CEC_Private_Functions CEC Private Functions
  * @{
  */
/**
  * @}
  */

/* Exported functions ---------------------------------------------------------*/

/** @defgroup CEC_Exported_Functions CEC Exported Functions
  * @{
  */

/** @defgroup CEC_Exported_Functions_Group1 Initialization and de-initialization functions
  *  @brief    Initialization and Configuration functions
  *
@verbatim
===============================================================================
            ##### Initialization and Configuration functions #####
 ===============================================================================
    [..]
    This subsection provides a set of functions allowing to initialize the CEC
      (+) The following parameters need to be configured:
        (++) SignalFreeTime
        (++) Tolerance
        (++) BRERxStop                 (RX stopped or not upon Bit Rising Error)
        (++) BREErrorBitGen            (Error-Bit generation in case of Bit Rising Error)
        (++) LBPEErrorBitGen           (Error-Bit generation in case of Long Bit Period Error)
        (++) BroadcastMsgNoErrorBitGen (Error-bit generation in case of broadcast message error)
        (++) SignalFreeTimeOption      (SFT Timer start definition)
        (++) OwnAddress                (CEC device address)
        (++) ListenMode

@endverbatim
  * @{
  */

/**
  * @brief Initializes the CEC mode according to the specified
  *         parameters in the CEC_InitTypeDef and creates the associated handle .
  * @param hcec CEC handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CEC_Init(CEC_HandleTypeDef *hcec)
{
  /* Check the CEC handle allocation */
  if ((hcec == NULL) || (hcec->Init.RxBuffer == NULL))
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_CEC_ALL_INSTANCE(hcec->Instance));
  assert_param(IS_CEC_SIGNALFREETIME(hcec->Init.SignalFreeTime));
  assert_param(IS_CEC_TOLERANCE(hcec->Init.Tolerance));
  assert_param(IS_CEC_BRERXSTOP(hcec->Init.BRERxStop));
  assert_param(IS_CEC_BREERRORBITGEN(hcec->Init.BREErrorBitGen));
  assert_param(IS_CEC_LBPEERRORBITGEN(hcec->Init.LBPEErrorBitGen));
  assert_param(IS_CEC_BROADCASTERROR_NO_ERRORBIT_GENERATION(hcec->Init.BroadcastMsgNoErrorBitGen));
  assert_param(IS_CEC_SFTOP(hcec->Init.SignalFreeTimeOption));
  assert_param(IS_CEC_LISTENING_MODE(hcec->Init.ListenMode));
  assert_param(IS_CEC_OWN_ADDRESS(hcec->Init.OwnAddress));

#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1)
  if (hcec->gState == HAL_CEC_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hcec->Lock = HAL_UNLOCKED;

    hcec->TxCpltCallback  = HAL_CEC_TxCpltCallback;  /* Legacy weak TxCpltCallback  */
    hcec->RxCpltCallback = HAL_CEC_RxCpltCallback;   /* Legacy weak RxCpltCallback */
    hcec->ErrorCallback = HAL_CEC_ErrorCallback;     /* Legacy weak ErrorCallback */

    if (hcec->MspInitCallback == NULL)
    {
      hcec->MspInitCallback = HAL_CEC_MspInit; /* Legacy weak MspInit  */
    }

    /* Init the low level hardware */
    hcec->MspInitCallback(hcec);
  }
#else
  if (hcec->gState == HAL_CEC_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hcec->Lock = HAL_UNLOCKED;
    /* Init the low level hardware : GPIO, CLOCK */
    HAL_CEC_MspInit(hcec);
  }
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */

  hcec->gState = HAL_CEC_STATE_BUSY;

  /* Disable the Peripheral */
  __HAL_CEC_DISABLE(hcec);

  /* Write to CEC Control Register */
  hcec->Instance->CFGR = hcec->Init.SignalFreeTime | hcec->Init.Tolerance | hcec->Init.BRERxStop | \
                         hcec->Init.BREErrorBitGen | hcec->Init.LBPEErrorBitGen | hcec->Init.BroadcastMsgNoErrorBitGen | \
                         hcec->Init.SignalFreeTimeOption | ((uint32_t)(hcec->Init.OwnAddress) << 16U) | \
                         hcec->Init.ListenMode;

  /* Enable the following CEC Transmission/Reception interrupts as
    * well as the following CEC Transmission/Reception Errors interrupts
    * Rx Byte Received IT
    * End of Reception IT
    * Rx overrun
    * Rx bit rising error
    * Rx short bit period error
    * Rx long bit period error
    * Rx missing acknowledge
    * Tx Byte Request IT
    * End of Transmission IT
    * Tx Missing Acknowledge IT
    * Tx-Error IT
    * Tx-Buffer Underrun IT
    * Tx arbitration lost   */
  __HAL_CEC_ENABLE_IT(hcec, CEC_IT_RXBR | CEC_IT_RXEND | CEC_IER_RX_ALL_ERR | CEC_IT_TXBR | CEC_IT_TXEND |
                      CEC_IER_TX_ALL_ERR);

  /* Enable the CEC Peripheral */
  __HAL_CEC_ENABLE(hcec);

  hcec->ErrorCode = HAL_CEC_ERROR_NONE;
  hcec->gState = HAL_CEC_STATE_READY;
  hcec->RxState = HAL_CEC_STATE_READY;

  return HAL_OK;
}

/**
  * @brief DeInitializes the CEC peripheral
  * @param hcec CEC handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CEC_DeInit(CEC_HandleTypeDef *hcec)
{
  /* Check the CEC handle allocation */
  if (hcec == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_CEC_ALL_INSTANCE(hcec->Instance));

  hcec->gState = HAL_CEC_STATE_BUSY;

#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1)
  if (hcec->MspDeInitCallback == NULL)
  {
    hcec->MspDeInitCallback = HAL_CEC_MspDeInit; /* Legacy weak MspDeInit  */
  }

  /* DeInit the low level hardware */
  hcec->MspDeInitCallback(hcec);

#else
  /* DeInit the low level hardware */
  HAL_CEC_MspDeInit(hcec);
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */

  /* Disable the Peripheral */
  __HAL_CEC_DISABLE(hcec);

  /* Clear Flags */
  __HAL_CEC_CLEAR_FLAG(hcec, CEC_FLAG_TXEND | CEC_FLAG_TXBR | CEC_FLAG_RXBR | CEC_FLAG_RXEND | CEC_ISR_ALL_ERROR);

  /* Disable the following CEC Transmission/Reception interrupts as
    * well as the following CEC Transmission/Reception Errors interrupts
    * Rx Byte Received IT
    * End of Reception IT
    * Rx overrun
    * Rx bit rising error
    * Rx short bit period error
    * Rx long bit period error
    * Rx missing acknowledge
    * Tx Byte Request IT
    * End of Transmission IT
    * Tx Missing Acknowledge IT
    * Tx-Error IT
    * Tx-Buffer Underrun IT
    * Tx arbitration lost   */
  __HAL_CEC_DISABLE_IT(hcec, CEC_IT_RXBR | CEC_IT_RXEND | CEC_IER_RX_ALL_ERR | CEC_IT_TXBR | CEC_IT_TXEND |
                       CEC_IER_TX_ALL_ERR);

  hcec->ErrorCode = HAL_CEC_ERROR_NONE;
  hcec->gState = HAL_CEC_STATE_RESET;
  hcec->RxState = HAL_CEC_STATE_RESET;

  /* Process Unlock */
  __HAL_UNLOCK(hcec);

  return HAL_OK;
}

/**
  * @brief Initializes the Own Address of the CEC device
  * @param hcec CEC handle
  * @param  CEC_OwnAddress The CEC own address.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CEC_SetDeviceAddress(CEC_HandleTypeDef *hcec, uint16_t CEC_OwnAddress)
{
  /* Check the parameters */
  assert_param(IS_CEC_OWN_ADDRESS(CEC_OwnAddress));

  if ((hcec->gState == HAL_CEC_STATE_READY) && (hcec->RxState == HAL_CEC_STATE_READY))
  {
    /* Process Locked */
    __HAL_LOCK(hcec);

    hcec->gState = HAL_CEC_STATE_BUSY;

    /* Disable the Peripheral */
    __HAL_CEC_DISABLE(hcec);

    if (CEC_OwnAddress != CEC_OWN_ADDRESS_NONE)
    {
      hcec->Instance->CFGR |= ((uint32_t)CEC_OwnAddress << 16);
    }
    else
    {
      hcec->Instance->CFGR &= ~(CEC_CFGR_OAR);
    }

    hcec->gState = HAL_CEC_STATE_READY;
    hcec->ErrorCode = HAL_CEC_ERROR_NONE;

    /* Process Unlocked */
    __HAL_UNLOCK(hcec);

    /* Enable the Peripheral */
    __HAL_CEC_ENABLE(hcec);

    return  HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief CEC MSP Init
  * @param hcec CEC handle
  * @retval None
  */
__weak void HAL_CEC_MspInit(CEC_HandleTypeDef *hcec)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcec);
  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_CEC_MspInit can be implemented in the user file
   */
}

/**
  * @brief CEC MSP DeInit
  * @param hcec CEC handle
  * @retval None
  */
__weak void HAL_CEC_MspDeInit(CEC_HandleTypeDef *hcec)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcec);
  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_CEC_MspDeInit can be implemented in the user file
   */
}
#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User CEC Callback
  *         To be used instead of the weak predefined callback
  * @param  hcec CEC handle
  * @param  CallbackID ID of the callback to be registered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_CEC_TX_CPLT_CB_ID Tx Complete callback ID
  *          @arg @ref HAL_CEC_ERROR_CB_ID Error callback ID
  *          @arg @ref HAL_CEC_MSPINIT_CB_ID MspInit callback ID
  *          @arg @ref HAL_CEC_MSPDEINIT_CB_ID MspDeInit callback ID
  * @param  pCallback pointer to the Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CEC_RegisterCallback(CEC_HandleTypeDef *hcec, HAL_CEC_CallbackIDTypeDef CallbackID,
                                           pCEC_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hcec);

  if (hcec->gState == HAL_CEC_STATE_READY)
  {
    switch (CallbackID)
    {
      case HAL_CEC_TX_CPLT_CB_ID :
        hcec->TxCpltCallback = pCallback;
        break;

      case HAL_CEC_ERROR_CB_ID :
        hcec->ErrorCallback = pCallback;
        break;

      case HAL_CEC_MSPINIT_CB_ID :
        hcec->MspInitCallback = pCallback;
        break;

      case HAL_CEC_MSPDEINIT_CB_ID :
        hcec->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (hcec->gState == HAL_CEC_STATE_RESET)
  {
    switch (CallbackID)
    {
      case HAL_CEC_MSPINIT_CB_ID :
        hcec->MspInitCallback = pCallback;
        break;

      case HAL_CEC_MSPDEINIT_CB_ID :
        hcec->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hcec);

  return status;
}

/**
  * @brief  Unregister an CEC Callback
  *         CEC callabck is redirected to the weak predefined callback
  * @param hcec uart handle
  * @param CallbackID ID of the callback to be unregistered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_CEC_TX_CPLT_CB_ID Tx Complete callback ID
  *          @arg @ref HAL_CEC_ERROR_CB_ID Error callback ID
  *          @arg @ref HAL_CEC_MSPINIT_CB_ID MspInit callback ID
  *          @arg @ref HAL_CEC_MSPDEINIT_CB_ID MspDeInit callback ID
  * @retval status
  */
HAL_StatusTypeDef HAL_CEC_UnRegisterCallback(CEC_HandleTypeDef *hcec, HAL_CEC_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hcec);

  if (hcec->gState == HAL_CEC_STATE_READY)
  {
    switch (CallbackID)
    {
      case HAL_CEC_TX_CPLT_CB_ID :
        hcec->TxCpltCallback = HAL_CEC_TxCpltCallback;  /* Legacy weak  TxCpltCallback */
        break;

      case HAL_CEC_ERROR_CB_ID :
        hcec->ErrorCallback = HAL_CEC_ErrorCallback;  /* Legacy weak ErrorCallback   */
        break;

      case HAL_CEC_MSPINIT_CB_ID :
        hcec->MspInitCallback = HAL_CEC_MspInit;
        break;

      case HAL_CEC_MSPDEINIT_CB_ID :
        hcec->MspDeInitCallback = HAL_CEC_MspDeInit;
        break;

      default :
        /* Update the error code */
        hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (hcec->gState == HAL_CEC_STATE_RESET)
  {
    switch (CallbackID)
    {
      case HAL_CEC_MSPINIT_CB_ID :
        hcec->MspInitCallback = HAL_CEC_MspInit;
        break;

      case HAL_CEC_MSPDEINIT_CB_ID :
        hcec->MspDeInitCallback = HAL_CEC_MspDeInit;
        break;

      default :
        /* Update the error code */
        hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hcec);

  return status;
}

/**
  * @brief  Register CEC RX complete Callback
  *         To be used instead of the weak HAL_CEC_RxCpltCallback() predefined callback
  * @param  hcec CEC handle
  * @param  pCallback pointer to the Rx transfer compelete Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CEC_RegisterRxCpltCallback(CEC_HandleTypeDef *hcec, pCEC_RxCallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hcec);

  if (HAL_CEC_STATE_READY == hcec->RxState)
  {
    hcec->RxCpltCallback = pCallback;
  }
  else
  {
    /* Update the error code */
    hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hcec);
  return status;
}

/**
  * @brief  UnRegister CEC RX complete Callback
  *         CEC RX complete Callback is redirected to the weak HAL_CEC_RxCpltCallback() predefined callback
  * @param  hcec CEC handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CEC_UnRegisterRxCpltCallback(CEC_HandleTypeDef *hcec)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hcec);

  if (HAL_CEC_STATE_READY == hcec->RxState)
  {
    hcec->RxCpltCallback = HAL_CEC_RxCpltCallback; /* Legacy weak  CEC RxCpltCallback  */
  }
  else
  {
    /* Update the error code */
    hcec->ErrorCode |= HAL_CEC_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hcec);
  return status;
}
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup CEC_Exported_Functions_Group2 Input and Output operation functions
  *  @brief CEC Transmit/Receive functions
  *
@verbatim
 ===============================================================================
                      ##### IO operation functions #####
 ===============================================================================
    This subsection provides a set of functions allowing to manage the CEC data transfers.

    (#) The CEC handle must contain the initiator (TX side) and the destination (RX side)
        logical addresses (4-bit long addresses, 0xF for broadcast messages destination)

    (#) The communication is performed using Interrupts.
           These API's return the HAL status.
           The end of the data processing will be indicated through the
           dedicated CEC IRQ when using Interrupt mode.
           The HAL_CEC_TxCpltCallback(), HAL_CEC_RxCpltCallback() user callbacks
           will be executed respectively at the end of the transmit or Receive process
           The HAL_CEC_ErrorCallback() user callback will be executed when a communication
           error is detected

    (#) API's with Interrupt are :
         (+) HAL_CEC_Transmit_IT()
         (+) HAL_CEC_IRQHandler()

    (#) A set of User Callbacks are provided:
         (+) HAL_CEC_TxCpltCallback()
         (+) HAL_CEC_RxCpltCallback()
         (+) HAL_CEC_ErrorCallback()

@endverbatim
  * @{
  */

/**
  * @brief Send data in interrupt mode
  * @param hcec CEC handle
  * @param InitiatorAddress Initiator address
  * @param DestinationAddress destination logical address
  * @param pData pointer to input byte data buffer
  * @param Size amount of data to be sent in bytes (without counting the header).
  *              0 means only the header is sent (ping operation).
  *              Maximum TX size is 15 bytes (1 opcode and up to 14 operands).
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CEC_Transmit_IT(CEC_HandleTypeDef *hcec, uint8_t InitiatorAddress, uint8_t DestinationAddress,
                                      uint8_t *pData, uint32_t Size)
{
  /* if the peripheral isn't already busy and if there is no previous transmission
     already pending due to arbitration lost */
  if (hcec->gState == HAL_CEC_STATE_READY)
  {
    if ((pData == NULL) && (Size > 0U))
    {
      return  HAL_ERROR;
    }

    assert_param(IS_CEC_ADDRESS(DestinationAddress));
    assert_param(IS_CEC_ADDRESS(InitiatorAddress));
    assert_param(IS_CEC_MSGSIZE(Size));

    /* Process Locked */
    __HAL_LOCK(hcec);
    hcec->pTxBuffPtr = pData;
    hcec->gState = HAL_CEC_STATE_BUSY_TX;
    hcec->ErrorCode = HAL_CEC_ERROR_NONE;

    /* initialize the number of bytes to send,
      * 0 means only one header is sent (ping operation) */
    hcec->TxXferCount = (uint16_t)Size;

    /* in case of no payload (Size = 0), sender is only pinging the system;
       Set TX End of Message (TXEOM) bit, must be set before writing data to TXDR */
    if (Size == 0U)
    {
      __HAL_CEC_LAST_BYTE_TX_SET(hcec);
    }

    /* send header block */
    hcec->Instance->TXDR = (uint32_t)(((uint32_t)InitiatorAddress << CEC_INITIATOR_LSB_POS) | DestinationAddress);

    /* Set TX Start of Message  (TXSOM) bit */
    __HAL_CEC_FIRST_BYTE_TX_SET(hcec);

    /* Process Unlocked */
    __HAL_UNLOCK(hcec);

    return HAL_OK;

  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Get size of the received frame.
  * @param hcec CEC handle
  * @retval Frame size
  */
uint32_t HAL_CEC_GetLastReceivedFrameSize(CEC_HandleTypeDef *hcec)
{
  return hcec->RxXferSize;
}

/**
  * @brief Change Rx Buffer.
  * @param hcec CEC handle
  * @param Rxbuffer Rx Buffer
  * @note  This function can be called only inside the HAL_CEC_RxCpltCallback()
  * @retval Frame size
  */
void HAL_CEC_ChangeRxBuffer(CEC_HandleTypeDef *hcec, uint8_t *Rxbuffer)
{
  hcec->Init.RxBuffer = Rxbuffer;
}

/**
  * @brief This function handles CEC interrupt requests.
  * @param hcec CEC handle
  * @retval None
  */
void HAL_CEC_IRQHandler(CEC_HandleTypeDef *hcec)
{

  /* save interrupts register for further error or interrupts handling purposes */
  uint32_t reg;
  reg = hcec->Instance->ISR;


  /* ----------------------------Arbitration Lost Management----------------------------------*/
  /* CEC TX arbitration error interrupt occurred --------------------------------------*/
  if ((reg & CEC_FLAG_ARBLST) != 0U)
  {
    hcec->ErrorCode = HAL_CEC_ERROR_ARBLST;
    __HAL_CEC_CLEAR_FLAG(hcec, CEC_FLAG_ARBLST);
  }

  /* ----------------------------Rx Management----------------------------------*/
  /* CEC RX byte received interrupt  ---------------------------------------------------*/
  if ((reg & CEC_FLAG_RXBR) != 0U)
  {
    /* reception is starting */
    hcec->RxState = HAL_CEC_STATE_BUSY_RX;
    hcec->RxXferSize++;
    /* read received byte */
    *hcec->Init.RxBuffer = (uint8_t) hcec->Instance->RXDR;
    hcec->Init.RxBuffer++;
    __HAL_CEC_CLEAR_FLAG(hcec, CEC_FLAG_RXBR);
  }

  /* CEC RX end received interrupt  ---------------------------------------------------*/
  if ((reg & CEC_FLAG_RXEND) != 0U)
  {
    /* clear IT */
    __HAL_CEC_CLEAR_FLAG(hcec, CEC_FLAG_RXEND);

    /* Rx process is completed, restore hcec->RxState to Ready */
    hcec->RxState = HAL_CEC_STATE_READY;
    hcec->ErrorCode = HAL_CEC_ERROR_NONE;
    hcec->Init.RxBuffer -= hcec->RxXferSize;
#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1U)
    hcec->RxCpltCallback(hcec, hcec->RxXferSize);
#else
    HAL_CEC_RxCpltCallback(hcec, hcec->RxXferSize);
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */
    hcec->RxXferSize = 0U;
  }

  /* ----------------------------Tx Management----------------------------------*/
  /* CEC TX byte request interrupt ------------------------------------------------*/
  if ((reg & CEC_FLAG_TXBR) != 0U)
  {
    --hcec->TxXferCount;
    if (hcec->TxXferCount == 0U)
    {
      /* if this is the last byte transmission, set TX End of Message (TXEOM) bit */
      __HAL_CEC_LAST_BYTE_TX_SET(hcec);
    }
    /* In all cases transmit the byte */
    hcec->Instance->TXDR = *hcec->pTxBuffPtr;
    hcec->pTxBuffPtr++;
    /* clear Tx-Byte request flag */
    __HAL_CEC_CLEAR_FLAG(hcec, CEC_FLAG_TXBR);
  }

  /* CEC TX end interrupt ------------------------------------------------*/
  if ((reg & CEC_FLAG_TXEND) != 0U)
  {
    __HAL_CEC_CLEAR_FLAG(hcec, CEC_FLAG_TXEND);

    /* Tx process is ended, restore hcec->gState to Ready */
    hcec->gState = HAL_CEC_STATE_READY;
    /* Call the Process Unlocked before calling the Tx call back API to give the possibility to
    start again the Transmission under the Tx call back API */
    __HAL_UNLOCK(hcec);
    hcec->ErrorCode = HAL_CEC_ERROR_NONE;
#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1U)
    hcec->TxCpltCallback(hcec);
#else
    HAL_CEC_TxCpltCallback(hcec);
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */
  }

  /* ----------------------------Rx/Tx Error Management----------------------------------*/
  if ((reg & (CEC_ISR_RXOVR | CEC_ISR_BRE | CEC_ISR_SBPE | CEC_ISR_LBPE | CEC_ISR_RXACKE | CEC_ISR_TXUDR | CEC_ISR_TXERR |
              CEC_ISR_TXACKE)) != 0U)
  {
    hcec->ErrorCode = reg;
    __HAL_CEC_CLEAR_FLAG(hcec, HAL_CEC_ERROR_RXOVR | HAL_CEC_ERROR_BRE | CEC_FLAG_LBPE | CEC_FLAG_SBPE |
                         HAL_CEC_ERROR_RXACKE | HAL_CEC_ERROR_TXUDR | HAL_CEC_ERROR_TXERR | HAL_CEC_ERROR_TXACKE);


    if ((reg & (CEC_ISR_RXOVR | CEC_ISR_BRE | CEC_ISR_SBPE | CEC_ISR_LBPE | CEC_ISR_RXACKE)) != 0U)
    {
      hcec->Init.RxBuffer -= hcec->RxXferSize;
      hcec->RxXferSize = 0U;
      hcec->RxState = HAL_CEC_STATE_READY;
    }
    else if (((reg & CEC_ISR_ARBLST) == 0U) && ((reg & (CEC_ISR_TXUDR | CEC_ISR_TXERR | CEC_ISR_TXACKE)) != 0U))
    {
      /* Set the CEC state ready to be able to start again the process */
      hcec->gState = HAL_CEC_STATE_READY;
    }
    else
    {
      /* Nothing todo*/
    }
#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1U)
    hcec->ErrorCallback(hcec);
#else
    /* Error  Call Back */
    HAL_CEC_ErrorCallback(hcec);
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */
  }
  else
  {
    /* Nothing todo*/
  }
}

/**
  * @brief Tx Transfer completed callback
  * @param hcec CEC handle
  * @retval None
  */
__weak void HAL_CEC_TxCpltCallback(CEC_HandleTypeDef *hcec)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcec);
  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_CEC_TxCpltCallback can be implemented in the user file
   */
}

/**
  * @brief Rx Transfer completed callback
  * @param hcec CEC handle
  * @param RxFrameSize Size of frame
  * @retval None
  */
__weak void HAL_CEC_RxCpltCallback(CEC_HandleTypeDef *hcec, uint32_t RxFrameSize)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcec);
  UNUSED(RxFrameSize);
  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_CEC_RxCpltCallback can be implemented in the user file
   */
}

/**
  * @brief CEC error callbacks
  * @param hcec CEC handle
  * @retval None
  */
__weak void HAL_CEC_ErrorCallback(CEC_HandleTypeDef *hcec)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcec);
  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_CEC_ErrorCallback can be implemented in the user file
   */
}
/**
  * @}
  */

/** @defgroup CEC_Exported_Functions_Group3 Peripheral Control function
  *  @brief   CEC control functions
  *
@verbatim
 ===============================================================================
                      ##### Peripheral Control function #####
 ===============================================================================
    [..]
    This subsection provides a set of functions allowing to control the CEC.
     (+) HAL_CEC_GetState() API can be helpful to check in run-time the state of the CEC peripheral.
     (+) HAL_CEC_GetError() API can be helpful to check in run-time the error of the CEC peripheral.
@endverbatim
  * @{
  */
/**
  * @brief return the CEC state
  * @param hcec pointer to a CEC_HandleTypeDef structure that contains
  *              the configuration information for the specified CEC module.
  * @retval HAL state
  */
HAL_CEC_StateTypeDef HAL_CEC_GetState(CEC_HandleTypeDef *hcec)
{
  uint32_t temp1, temp2;
  temp1 = hcec->gState;
  temp2 = hcec->RxState;

  return (HAL_CEC_StateTypeDef)(temp1 | temp2);
}

/**
  * @brief  Return the CEC error code
  * @param  hcec  pointer to a CEC_HandleTypeDef structure that contains
  *              the configuration information for the specified CEC.
  * @retval CEC Error Code
  */
uint32_t HAL_CEC_GetError(CEC_HandleTypeDef *hcec)
{
  return hcec->ErrorCode;
}

/**
  * @}
  */

/**
  * @}
  */
#endif /* CEC */
#endif /* HAL_CEC_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
