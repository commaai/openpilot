/**
  ******************************************************************************
  * @file    stm32f4xx_hal_can.c
  * @author  MCD Application Team
  * @brief   CAN HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the Controller Area Network (CAN) peripheral:
  *           + Initialization and de-initialization functions
  *           + Configuration functions
  *           + Control functions
  *           + Interrupts management
  *           + Callbacks functions
  *           + Peripheral State and Error functions
  *
  @verbatim
  ==============================================================================
                        ##### How to use this driver #####
  ==============================================================================
    [..]
      (#) Initialize the CAN low level resources by implementing the
          HAL_CAN_MspInit():
         (++) Enable the CAN interface clock using __HAL_RCC_CANx_CLK_ENABLE()
         (++) Configure CAN pins
             (+++) Enable the clock for the CAN GPIOs
             (+++) Configure CAN pins as alternate function open-drain
         (++) In case of using interrupts (e.g. HAL_CAN_ActivateNotification())
             (+++) Configure the CAN interrupt priority using
                   HAL_NVIC_SetPriority()
             (+++) Enable the CAN IRQ handler using HAL_NVIC_EnableIRQ()
             (+++) In CAN IRQ handler, call HAL_CAN_IRQHandler()

      (#) Initialize the CAN peripheral using HAL_CAN_Init() function. This
          function resorts to HAL_CAN_MspInit() for low-level initialization.

      (#) Configure the reception filters using the following configuration
          functions:
            (++) HAL_CAN_ConfigFilter()

      (#) Start the CAN module using HAL_CAN_Start() function. At this level
          the node is active on the bus: it receive messages, and can send
          messages.

      (#) To manage messages transmission, the following Tx control functions
          can be used:
            (++) HAL_CAN_AddTxMessage() to request transmission of a new
                 message.
            (++) HAL_CAN_AbortTxRequest() to abort transmission of a pending
                 message.
            (++) HAL_CAN_GetTxMailboxesFreeLevel() to get the number of free Tx
                 mailboxes.
            (++) HAL_CAN_IsTxMessagePending() to check if a message is pending
                 in a Tx mailbox.
            (++) HAL_CAN_GetTxTimestamp() to get the timestamp of Tx message
                 sent, if time triggered communication mode is enabled.

      (#) When a message is received into the CAN Rx FIFOs, it can be retrieved
          using the HAL_CAN_GetRxMessage() function. The function
          HAL_CAN_GetRxFifoFillLevel() allows to know how many Rx message are
          stored in the Rx Fifo.

      (#) Calling the HAL_CAN_Stop() function stops the CAN module.

      (#) The deinitialization is achieved with HAL_CAN_DeInit() function.


      *** Polling mode operation ***
      ==============================
    [..]
      (#) Reception:
            (++) Monitor reception of message using HAL_CAN_GetRxFifoFillLevel()
                 until at least one message is received.
            (++) Then get the message using HAL_CAN_GetRxMessage().

      (#) Transmission:
            (++) Monitor the Tx mailboxes availability until at least one Tx
                 mailbox is free, using HAL_CAN_GetTxMailboxesFreeLevel().
            (++) Then request transmission of a message using
                 HAL_CAN_AddTxMessage().


      *** Interrupt mode operation ***
      ================================
    [..]
      (#) Notifications are activated using HAL_CAN_ActivateNotification()
          function. Then, the process can be controlled through the
          available user callbacks: HAL_CAN_xxxCallback(), using same APIs
          HAL_CAN_GetRxMessage() and HAL_CAN_AddTxMessage().

      (#) Notifications can be deactivated using
          HAL_CAN_DeactivateNotification() function.

      (#) Special care should be taken for CAN_IT_RX_FIFO0_MSG_PENDING and
          CAN_IT_RX_FIFO1_MSG_PENDING notifications. These notifications trig
          the callbacks HAL_CAN_RxFIFO0MsgPendingCallback() and
          HAL_CAN_RxFIFO1MsgPendingCallback(). User has two possible options
          here.
            (++) Directly get the Rx message in the callback, using
                 HAL_CAN_GetRxMessage().
            (++) Or deactivate the notification in the callback without
                 getting the Rx message. The Rx message can then be got later
                 using HAL_CAN_GetRxMessage(). Once the Rx message have been
                 read, the notification can be activated again.


      *** Sleep mode ***
      ==================
    [..]
      (#) The CAN peripheral can be put in sleep mode (low power), using
          HAL_CAN_RequestSleep(). The sleep mode will be entered as soon as the
          current CAN activity (transmission or reception of a CAN frame) will
          be completed.

      (#) A notification can be activated to be informed when the sleep mode
          will be entered.

      (#) It can be checked if the sleep mode is entered using
          HAL_CAN_IsSleepActive().
          Note that the CAN state (accessible from the API HAL_CAN_GetState())
          is HAL_CAN_STATE_SLEEP_PENDING as soon as the sleep mode request is
          submitted (the sleep mode is not yet entered), and become
          HAL_CAN_STATE_SLEEP_ACTIVE when the sleep mode is effective.

      (#) The wake-up from sleep mode can be triggered by two ways:
            (++) Using HAL_CAN_WakeUp(). When returning from this function,
                 the sleep mode is exited (if return status is HAL_OK).
            (++) When a start of Rx CAN frame is detected by the CAN peripheral,
                 if automatic wake up mode is enabled.

  *** Callback registration ***
  =============================================

  The compilation define  USE_HAL_CAN_REGISTER_CALLBACKS when set to 1
  allows the user to configure dynamically the driver callbacks.
  Use Function HAL_CAN_RegisterCallback() to register an interrupt callback.

  Function HAL_CAN_RegisterCallback() allows to register following callbacks:
    (+) TxMailbox0CompleteCallback   : Tx Mailbox 0 Complete Callback.
    (+) TxMailbox1CompleteCallback   : Tx Mailbox 1 Complete Callback.
    (+) TxMailbox2CompleteCallback   : Tx Mailbox 2 Complete Callback.
    (+) TxMailbox0AbortCallback      : Tx Mailbox 0 Abort Callback.
    (+) TxMailbox1AbortCallback      : Tx Mailbox 1 Abort Callback.
    (+) TxMailbox2AbortCallback      : Tx Mailbox 2 Abort Callback.
    (+) RxFifo0MsgPendingCallback    : Rx Fifo 0 Message Pending Callback.
    (+) RxFifo0FullCallback          : Rx Fifo 0 Full Callback.
    (+) RxFifo1MsgPendingCallback    : Rx Fifo 1 Message Pending Callback.
    (+) RxFifo1FullCallback          : Rx Fifo 1 Full Callback.
    (+) SleepCallback                : Sleep Callback.
    (+) WakeUpFromRxMsgCallback      : Wake Up From Rx Message Callback.
    (+) ErrorCallback                : Error Callback.
    (+) MspInitCallback              : CAN MspInit.
    (+) MspDeInitCallback            : CAN MspDeInit.
  This function takes as parameters the HAL peripheral handle, the Callback ID
  and a pointer to the user callback function.

  Use function HAL_CAN_UnRegisterCallback() to reset a callback to the default
  weak function.
  HAL_CAN_UnRegisterCallback takes as parameters the HAL peripheral handle,
  and the Callback ID.
  This function allows to reset following callbacks:
    (+) TxMailbox0CompleteCallback   : Tx Mailbox 0 Complete Callback.
    (+) TxMailbox1CompleteCallback   : Tx Mailbox 1 Complete Callback.
    (+) TxMailbox2CompleteCallback   : Tx Mailbox 2 Complete Callback.
    (+) TxMailbox0AbortCallback      : Tx Mailbox 0 Abort Callback.
    (+) TxMailbox1AbortCallback      : Tx Mailbox 1 Abort Callback.
    (+) TxMailbox2AbortCallback      : Tx Mailbox 2 Abort Callback.
    (+) RxFifo0MsgPendingCallback    : Rx Fifo 0 Message Pending Callback.
    (+) RxFifo0FullCallback          : Rx Fifo 0 Full Callback.
    (+) RxFifo1MsgPendingCallback    : Rx Fifo 1 Message Pending Callback.
    (+) RxFifo1FullCallback          : Rx Fifo 1 Full Callback.
    (+) SleepCallback                : Sleep Callback.
    (+) WakeUpFromRxMsgCallback      : Wake Up From Rx Message Callback.
    (+) ErrorCallback                : Error Callback.
    (+) MspInitCallback              : CAN MspInit.
    (+) MspDeInitCallback            : CAN MspDeInit.

  By default, after the HAL_CAN_Init() and when the state is HAL_CAN_STATE_RESET,
  all callbacks are set to the corresponding weak functions:
  example HAL_CAN_ErrorCallback().
  Exception done for MspInit and MspDeInit functions that are
  reset to the legacy weak function in the HAL_CAN_Init()/ HAL_CAN_DeInit() only when
  these callbacks are null (not registered beforehand).
  if not, MspInit or MspDeInit are not null, the HAL_CAN_Init()/ HAL_CAN_DeInit()
  keep and use the user MspInit/MspDeInit callbacks (registered beforehand)

  Callbacks can be registered/unregistered in HAL_CAN_STATE_READY state only.
  Exception done MspInit/MspDeInit that can be registered/unregistered
  in HAL_CAN_STATE_READY or HAL_CAN_STATE_RESET state,
  thus registered (user) MspInit/DeInit callbacks can be used during the Init/DeInit.
  In that case first register the MspInit/MspDeInit user callbacks
  using HAL_CAN_RegisterCallback() before calling HAL_CAN_DeInit()
  or HAL_CAN_Init() function.

  When The compilation define USE_HAL_CAN_REGISTER_CALLBACKS is set to 0 or
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

#if defined(CAN1)

/** @defgroup CAN CAN
  * @brief CAN driver modules
  * @{
  */

#ifdef HAL_CAN_MODULE_ENABLED

#ifdef HAL_CAN_LEGACY_MODULE_ENABLED
  #error "The CAN driver cannot be used with its legacy, Please enable only one CAN module at once"
#endif

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @defgroup CAN_Private_Constants CAN Private Constants
  * @{
  */
#define CAN_TIMEOUT_VALUE 10U
/**
  * @}
  */
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Exported functions --------------------------------------------------------*/

/** @defgroup CAN_Exported_Functions CAN Exported Functions
  * @{
  */

/** @defgroup CAN_Exported_Functions_Group1 Initialization and de-initialization functions
 *  @brief    Initialization and Configuration functions
 *
@verbatim
  ==============================================================================
              ##### Initialization and de-initialization functions #####
  ==============================================================================
    [..]  This section provides functions allowing to:
      (+) HAL_CAN_Init                       : Initialize and configure the CAN.
      (+) HAL_CAN_DeInit                     : De-initialize the CAN.
      (+) HAL_CAN_MspInit                    : Initialize the CAN MSP.
      (+) HAL_CAN_MspDeInit                  : DeInitialize the CAN MSP.

@endverbatim
  * @{
  */

/**
  * @brief  Initializes the CAN peripheral according to the specified
  *         parameters in the CAN_InitStruct.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_Init(CAN_HandleTypeDef *hcan)
{
  uint32_t tickstart;

  /* Check CAN handle */
  if (hcan == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_CAN_ALL_INSTANCE(hcan->Instance));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.TimeTriggeredMode));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.AutoBusOff));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.AutoWakeUp));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.AutoRetransmission));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.ReceiveFifoLocked));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.TransmitFifoPriority));
  assert_param(IS_CAN_MODE(hcan->Init.Mode));
  assert_param(IS_CAN_SJW(hcan->Init.SyncJumpWidth));
  assert_param(IS_CAN_BS1(hcan->Init.TimeSeg1));
  assert_param(IS_CAN_BS2(hcan->Init.TimeSeg2));
  assert_param(IS_CAN_PRESCALER(hcan->Init.Prescaler));

#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
  if (hcan->State == HAL_CAN_STATE_RESET)
  {
    /* Reset callbacks to legacy functions */
    hcan->RxFifo0MsgPendingCallback  =  HAL_CAN_RxFifo0MsgPendingCallback;  /* Legacy weak RxFifo0MsgPendingCallback  */
    hcan->RxFifo0FullCallback        =  HAL_CAN_RxFifo0FullCallback;        /* Legacy weak RxFifo0FullCallback        */
    hcan->RxFifo1MsgPendingCallback  =  HAL_CAN_RxFifo1MsgPendingCallback;  /* Legacy weak RxFifo1MsgPendingCallback  */
    hcan->RxFifo1FullCallback        =  HAL_CAN_RxFifo1FullCallback;        /* Legacy weak RxFifo1FullCallback        */
    hcan->TxMailbox0CompleteCallback =  HAL_CAN_TxMailbox0CompleteCallback; /* Legacy weak TxMailbox0CompleteCallback */
    hcan->TxMailbox1CompleteCallback =  HAL_CAN_TxMailbox1CompleteCallback; /* Legacy weak TxMailbox1CompleteCallback */
    hcan->TxMailbox2CompleteCallback =  HAL_CAN_TxMailbox2CompleteCallback; /* Legacy weak TxMailbox2CompleteCallback */
    hcan->TxMailbox0AbortCallback    =  HAL_CAN_TxMailbox0AbortCallback;    /* Legacy weak TxMailbox0AbortCallback    */
    hcan->TxMailbox1AbortCallback    =  HAL_CAN_TxMailbox1AbortCallback;    /* Legacy weak TxMailbox1AbortCallback    */
    hcan->TxMailbox2AbortCallback    =  HAL_CAN_TxMailbox2AbortCallback;    /* Legacy weak TxMailbox2AbortCallback    */
    hcan->SleepCallback              =  HAL_CAN_SleepCallback;              /* Legacy weak SleepCallback              */
    hcan->WakeUpFromRxMsgCallback    =  HAL_CAN_WakeUpFromRxMsgCallback;    /* Legacy weak WakeUpFromRxMsgCallback    */
    hcan->ErrorCallback              =  HAL_CAN_ErrorCallback;              /* Legacy weak ErrorCallback              */

    if (hcan->MspInitCallback == NULL)
    {
      hcan->MspInitCallback = HAL_CAN_MspInit; /* Legacy weak MspInit */
    }

    /* Init the low level hardware: CLOCK, NVIC */
    hcan->MspInitCallback(hcan);
  }

#else
  if (hcan->State == HAL_CAN_STATE_RESET)
  {
    /* Init the low level hardware: CLOCK, NVIC */
    HAL_CAN_MspInit(hcan);
  }
#endif /* (USE_HAL_CAN_REGISTER_CALLBACKS) */

  /* Request initialisation */
  SET_BIT(hcan->Instance->MCR, CAN_MCR_INRQ);

  /* Get tick */
  tickstart = HAL_GetTick();

  /* Wait initialisation acknowledge */
  while ((hcan->Instance->MSR & CAN_MSR_INAK) == 0U)
  {
    if ((HAL_GetTick() - tickstart) > CAN_TIMEOUT_VALUE)
    {
      /* Update error code */
      hcan->ErrorCode |= HAL_CAN_ERROR_TIMEOUT;

      /* Change CAN state */
      hcan->State = HAL_CAN_STATE_ERROR;

      return HAL_ERROR;
    }
  }

  /* Exit from sleep mode */
  CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_SLEEP);

  /* Get tick */
  tickstart = HAL_GetTick();

  /* Check Sleep mode leave acknowledge */
  while ((hcan->Instance->MSR & CAN_MSR_SLAK) != 0U)
  {
    if ((HAL_GetTick() - tickstart) > CAN_TIMEOUT_VALUE)
    {
      /* Update error code */
      hcan->ErrorCode |= HAL_CAN_ERROR_TIMEOUT;

      /* Change CAN state */
      hcan->State = HAL_CAN_STATE_ERROR;

      return HAL_ERROR;
    }
  }

  /* Set the time triggered communication mode */
  if (hcan->Init.TimeTriggeredMode == ENABLE)
  {
    SET_BIT(hcan->Instance->MCR, CAN_MCR_TTCM);
  }
  else
  {
    CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_TTCM);
  }

  /* Set the automatic bus-off management */
  if (hcan->Init.AutoBusOff == ENABLE)
  {
    SET_BIT(hcan->Instance->MCR, CAN_MCR_ABOM);
  }
  else
  {
    CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_ABOM);
  }

  /* Set the automatic wake-up mode */
  if (hcan->Init.AutoWakeUp == ENABLE)
  {
    SET_BIT(hcan->Instance->MCR, CAN_MCR_AWUM);
  }
  else
  {
    CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_AWUM);
  }

  /* Set the automatic retransmission */
  if (hcan->Init.AutoRetransmission == ENABLE)
  {
    CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_NART);
  }
  else
  {
    SET_BIT(hcan->Instance->MCR, CAN_MCR_NART);
  }

  /* Set the receive FIFO locked mode */
  if (hcan->Init.ReceiveFifoLocked == ENABLE)
  {
    SET_BIT(hcan->Instance->MCR, CAN_MCR_RFLM);
  }
  else
  {
    CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_RFLM);
  }

  /* Set the transmit FIFO priority */
  if (hcan->Init.TransmitFifoPriority == ENABLE)
  {
    SET_BIT(hcan->Instance->MCR, CAN_MCR_TXFP);
  }
  else
  {
    CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_TXFP);
  }

  /* Set the bit timing register */
  WRITE_REG(hcan->Instance->BTR, (uint32_t)(hcan->Init.Mode           |
                                            hcan->Init.SyncJumpWidth  |
                                            hcan->Init.TimeSeg1       |
                                            hcan->Init.TimeSeg2       |
                                            (hcan->Init.Prescaler - 1U)));

  /* Initialize the error code */
  hcan->ErrorCode = HAL_CAN_ERROR_NONE;

  /* Initialize the CAN state */
  hcan->State = HAL_CAN_STATE_READY;

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Deinitializes the CAN peripheral registers to their default
  *         reset values.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_DeInit(CAN_HandleTypeDef *hcan)
{
  /* Check CAN handle */
  if (hcan == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_CAN_ALL_INSTANCE(hcan->Instance));

  /* Stop the CAN module */
  (void)HAL_CAN_Stop(hcan);

#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
  if (hcan->MspDeInitCallback == NULL)
  {
    hcan->MspDeInitCallback = HAL_CAN_MspDeInit; /* Legacy weak MspDeInit */
  }

  /* DeInit the low level hardware: CLOCK, NVIC */
  hcan->MspDeInitCallback(hcan);

#else
  /* DeInit the low level hardware: CLOCK, NVIC */
  HAL_CAN_MspDeInit(hcan);
#endif /* (USE_HAL_CAN_REGISTER_CALLBACKS) */

  /* Reset the CAN peripheral */
  SET_BIT(hcan->Instance->MCR, CAN_MCR_RESET);

  /* Reset the CAN ErrorCode */
  hcan->ErrorCode = HAL_CAN_ERROR_NONE;

  /* Change CAN state */
  hcan->State = HAL_CAN_STATE_RESET;

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Initializes the CAN MSP.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_MspInit(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitializes the CAN MSP.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_MspDeInit(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_MspDeInit could be implemented in the user file
   */
}

#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
/**
  * @brief  Register a CAN CallBack.
  *         To be used instead of the weak predefined callback
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for CAN module
  * @param  CallbackID ID of the callback to be registered
  *         This parameter can be one of the following values:
  *           @arg @ref HAL_CAN_TX_MAILBOX0_COMPLETE_CB_ID Tx Mailbox 0 Complete callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX1_COMPLETE_CB_ID Tx Mailbox 1 Complete callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX2_COMPLETE_CB_ID Tx Mailbox 2 Complete callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX0_ABORT_CB_ID Tx Mailbox 0 Abort callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX1_ABORT_CB_ID Tx Mailbox 1 Abort callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX2_ABORT_CB_ID Tx Mailbox 2 Abort callback ID
  *           @arg @ref HAL_CAN_RX_FIFO0_MSG_PENDING_CB_ID Rx Fifo 0 message pending callback ID
  *           @arg @ref HAL_CAN_RX_FIFO0_FULL_CB_ID Rx Fifo 0 full callback ID
  *           @arg @ref HAL_CAN_RX_FIFO1_MSG_PENDING_CB_ID Rx Fifo 1 message pending callback ID
  *           @arg @ref HAL_CAN_RX_FIFO1_FULL_CB_ID Rx Fifo 1 full callback ID
  *           @arg @ref HAL_CAN_SLEEP_CB_ID Sleep callback ID
  *           @arg @ref HAL_CAN_WAKEUP_FROM_RX_MSG_CB_ID Wake Up from Rx message callback ID
  *           @arg @ref HAL_CAN_ERROR_CB_ID Error callback ID
  *           @arg @ref HAL_CAN_MSPINIT_CB_ID MspInit callback ID
  *           @arg @ref HAL_CAN_MSPDEINIT_CB_ID MspDeInit callback ID
  * @param  pCallback pointer to the Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_RegisterCallback(CAN_HandleTypeDef *hcan, HAL_CAN_CallbackIDTypeDef CallbackID, void (* pCallback)(CAN_HandleTypeDef *_hcan))
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_INVALID_CALLBACK;

    return HAL_ERROR;
  }

  if (hcan->State == HAL_CAN_STATE_READY)
  {
    switch (CallbackID)
    {
      case HAL_CAN_TX_MAILBOX0_COMPLETE_CB_ID :
        hcan->TxMailbox0CompleteCallback = pCallback;
        break;

      case HAL_CAN_TX_MAILBOX1_COMPLETE_CB_ID :
        hcan->TxMailbox1CompleteCallback = pCallback;
        break;

      case HAL_CAN_TX_MAILBOX2_COMPLETE_CB_ID :
        hcan->TxMailbox2CompleteCallback = pCallback;
        break;

      case HAL_CAN_TX_MAILBOX0_ABORT_CB_ID :
        hcan->TxMailbox0AbortCallback = pCallback;
        break;

      case HAL_CAN_TX_MAILBOX1_ABORT_CB_ID :
        hcan->TxMailbox1AbortCallback = pCallback;
        break;

      case HAL_CAN_TX_MAILBOX2_ABORT_CB_ID :
        hcan->TxMailbox2AbortCallback = pCallback;
        break;

      case HAL_CAN_RX_FIFO0_MSG_PENDING_CB_ID :
        hcan->RxFifo0MsgPendingCallback = pCallback;
        break;

      case HAL_CAN_RX_FIFO0_FULL_CB_ID :
        hcan->RxFifo0FullCallback = pCallback;
        break;

      case HAL_CAN_RX_FIFO1_MSG_PENDING_CB_ID :
        hcan->RxFifo1MsgPendingCallback = pCallback;
        break;

      case HAL_CAN_RX_FIFO1_FULL_CB_ID :
        hcan->RxFifo1FullCallback = pCallback;
        break;

      case HAL_CAN_SLEEP_CB_ID :
        hcan->SleepCallback = pCallback;
        break;

      case HAL_CAN_WAKEUP_FROM_RX_MSG_CB_ID :
        hcan->WakeUpFromRxMsgCallback = pCallback;
        break;

      case HAL_CAN_ERROR_CB_ID :
        hcan->ErrorCallback = pCallback;
        break;

      case HAL_CAN_MSPINIT_CB_ID :
        hcan->MspInitCallback = pCallback;
        break;

      case HAL_CAN_MSPDEINIT_CB_ID :
        hcan->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (hcan->State == HAL_CAN_STATE_RESET)
  {
    switch (CallbackID)
    {
      case HAL_CAN_MSPINIT_CB_ID :
        hcan->MspInitCallback = pCallback;
        break;

      case HAL_CAN_MSPDEINIT_CB_ID :
        hcan->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  return status;
}

/**
  * @brief  Unregister a CAN CallBack.
  *         CAN callabck is redirected to the weak predefined callback
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for CAN module
  * @param  CallbackID ID of the callback to be unregistered
  *         This parameter can be one of the following values:
  *           @arg @ref HAL_CAN_TX_MAILBOX0_COMPLETE_CB_ID Tx Mailbox 0 Complete callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX1_COMPLETE_CB_ID Tx Mailbox 1 Complete callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX2_COMPLETE_CB_ID Tx Mailbox 2 Complete callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX0_ABORT_CB_ID Tx Mailbox 0 Abort callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX1_ABORT_CB_ID Tx Mailbox 1 Abort callback ID
  *           @arg @ref HAL_CAN_TX_MAILBOX2_ABORT_CB_ID Tx Mailbox 2 Abort callback ID
  *           @arg @ref HAL_CAN_RX_FIFO0_MSG_PENDING_CB_ID Rx Fifo 0 message pending callback ID
  *           @arg @ref HAL_CAN_RX_FIFO0_FULL_CB_ID Rx Fifo 0 full callback ID
  *           @arg @ref HAL_CAN_RX_FIFO1_MSG_PENDING_CB_ID Rx Fifo 1 message pending callback ID
  *           @arg @ref HAL_CAN_RX_FIFO1_FULL_CB_ID Rx Fifo 1 full callback ID
  *           @arg @ref HAL_CAN_SLEEP_CB_ID Sleep callback ID
  *           @arg @ref HAL_CAN_WAKEUP_FROM_RX_MSG_CB_ID Wake Up from Rx message callback ID
  *           @arg @ref HAL_CAN_ERROR_CB_ID Error callback ID
  *           @arg @ref HAL_CAN_MSPINIT_CB_ID MspInit callback ID
  *           @arg @ref HAL_CAN_MSPDEINIT_CB_ID MspDeInit callback ID
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_UnRegisterCallback(CAN_HandleTypeDef *hcan, HAL_CAN_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (hcan->State == HAL_CAN_STATE_READY)
  {
    switch (CallbackID)
    {
      case HAL_CAN_TX_MAILBOX0_COMPLETE_CB_ID :
        hcan->TxMailbox0CompleteCallback = HAL_CAN_TxMailbox0CompleteCallback;
        break;

      case HAL_CAN_TX_MAILBOX1_COMPLETE_CB_ID :
        hcan->TxMailbox1CompleteCallback = HAL_CAN_TxMailbox1CompleteCallback;
        break;

      case HAL_CAN_TX_MAILBOX2_COMPLETE_CB_ID :
        hcan->TxMailbox2CompleteCallback = HAL_CAN_TxMailbox2CompleteCallback;
        break;

      case HAL_CAN_TX_MAILBOX0_ABORT_CB_ID :
        hcan->TxMailbox0AbortCallback = HAL_CAN_TxMailbox0AbortCallback;
        break;

      case HAL_CAN_TX_MAILBOX1_ABORT_CB_ID :
        hcan->TxMailbox1AbortCallback = HAL_CAN_TxMailbox1AbortCallback;
        break;

      case HAL_CAN_TX_MAILBOX2_ABORT_CB_ID :
        hcan->TxMailbox2AbortCallback = HAL_CAN_TxMailbox2AbortCallback;
        break;

      case HAL_CAN_RX_FIFO0_MSG_PENDING_CB_ID :
        hcan->RxFifo0MsgPendingCallback = HAL_CAN_RxFifo0MsgPendingCallback;
        break;

      case HAL_CAN_RX_FIFO0_FULL_CB_ID :
        hcan->RxFifo0FullCallback = HAL_CAN_RxFifo0FullCallback;
        break;

      case HAL_CAN_RX_FIFO1_MSG_PENDING_CB_ID :
        hcan->RxFifo1MsgPendingCallback = HAL_CAN_RxFifo1MsgPendingCallback;
        break;

      case HAL_CAN_RX_FIFO1_FULL_CB_ID :
        hcan->RxFifo1FullCallback = HAL_CAN_RxFifo1FullCallback;
        break;

      case HAL_CAN_SLEEP_CB_ID :
        hcan->SleepCallback = HAL_CAN_SleepCallback;
        break;

      case HAL_CAN_WAKEUP_FROM_RX_MSG_CB_ID :
        hcan->WakeUpFromRxMsgCallback = HAL_CAN_WakeUpFromRxMsgCallback;
        break;

      case HAL_CAN_ERROR_CB_ID :
        hcan->ErrorCallback = HAL_CAN_ErrorCallback;
        break;

      case HAL_CAN_MSPINIT_CB_ID :
        hcan->MspInitCallback = HAL_CAN_MspInit;
        break;

      case HAL_CAN_MSPDEINIT_CB_ID :
        hcan->MspDeInitCallback = HAL_CAN_MspDeInit;
        break;

      default :
        /* Update the error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (hcan->State == HAL_CAN_STATE_RESET)
  {
    switch (CallbackID)
    {
      case HAL_CAN_MSPINIT_CB_ID :
        hcan->MspInitCallback = HAL_CAN_MspInit;
        break;

      case HAL_CAN_MSPDEINIT_CB_ID :
        hcan->MspDeInitCallback = HAL_CAN_MspDeInit;
        break;

      default :
        /* Update the error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  return status;
}
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup CAN_Exported_Functions_Group2 Configuration functions
 *  @brief    Configuration functions.
 *
@verbatim
  ==============================================================================
              ##### Configuration functions #####
  ==============================================================================
    [..]  This section provides functions allowing to:
      (+) HAL_CAN_ConfigFilter            : Configure the CAN reception filters

@endverbatim
  * @{
  */

/**
  * @brief  Configures the CAN reception filter according to the specified
  *         parameters in the CAN_FilterInitStruct.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  sFilterConfig pointer to a CAN_FilterTypeDef structure that
  *         contains the filter configuration information.
  * @retval None
  */
HAL_StatusTypeDef HAL_CAN_ConfigFilter(CAN_HandleTypeDef *hcan, CAN_FilterTypeDef *sFilterConfig)
{
  uint32_t filternbrbitpos;
  CAN_TypeDef *can_ip = hcan->Instance;
  HAL_CAN_StateTypeDef state = hcan->State;

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Check the parameters */
    assert_param(IS_CAN_FILTER_ID_HALFWORD(sFilterConfig->FilterIdHigh));
    assert_param(IS_CAN_FILTER_ID_HALFWORD(sFilterConfig->FilterIdLow));
    assert_param(IS_CAN_FILTER_ID_HALFWORD(sFilterConfig->FilterMaskIdHigh));
    assert_param(IS_CAN_FILTER_ID_HALFWORD(sFilterConfig->FilterMaskIdLow));
    assert_param(IS_CAN_FILTER_MODE(sFilterConfig->FilterMode));
    assert_param(IS_CAN_FILTER_SCALE(sFilterConfig->FilterScale));
    assert_param(IS_CAN_FILTER_FIFO(sFilterConfig->FilterFIFOAssignment));
    assert_param(IS_CAN_FILTER_ACTIVATION(sFilterConfig->FilterActivation));

#if defined(CAN3)
    /* Check the CAN instance */
    if (hcan->Instance == CAN3)
    {
      /* CAN3 is single instance with 14 dedicated filters banks */

      /* Check the parameters */
      assert_param(IS_CAN_FILTER_BANK_SINGLE(sFilterConfig->FilterBank));
    }
    else
    {
      /* CAN1 and CAN2 are dual instances with 28 common filters banks */
      /* Select master instance to access the filter banks */
      can_ip = CAN1;

      /* Check the parameters */
      assert_param(IS_CAN_FILTER_BANK_DUAL(sFilterConfig->FilterBank));
      assert_param(IS_CAN_FILTER_BANK_DUAL(sFilterConfig->SlaveStartFilterBank));
    }
#elif defined(CAN2)
    /* CAN1 and CAN2 are dual instances with 28 common filters banks */
    /* Select master instance to access the filter banks */
    can_ip = CAN1;

    /* Check the parameters */
    assert_param(IS_CAN_FILTER_BANK_DUAL(sFilterConfig->FilterBank));
    assert_param(IS_CAN_FILTER_BANK_DUAL(sFilterConfig->SlaveStartFilterBank));
#else
    /* CAN1 is single instance with 14 dedicated filters banks */

    /* Check the parameters */
    assert_param(IS_CAN_FILTER_BANK_SINGLE(sFilterConfig->FilterBank));
#endif

    /* Initialisation mode for the filter */
    SET_BIT(can_ip->FMR, CAN_FMR_FINIT);

#if defined(CAN3)
    /* Check the CAN instance */
    if (can_ip == CAN1)
    {
      /* Select the start filter number of CAN2 slave instance */
      CLEAR_BIT(can_ip->FMR, CAN_FMR_CAN2SB);
      SET_BIT(can_ip->FMR, sFilterConfig->SlaveStartFilterBank << CAN_FMR_CAN2SB_Pos);
    }

#elif defined(CAN2)
    /* Select the start filter number of CAN2 slave instance */
    CLEAR_BIT(can_ip->FMR, CAN_FMR_CAN2SB);
    SET_BIT(can_ip->FMR, sFilterConfig->SlaveStartFilterBank << CAN_FMR_CAN2SB_Pos);

#endif
    /* Convert filter number into bit position */
    filternbrbitpos = (uint32_t)1 << (sFilterConfig->FilterBank & 0x1FU);

    /* Filter Deactivation */
    CLEAR_BIT(can_ip->FA1R, filternbrbitpos);

    /* Filter Scale */
    if (sFilterConfig->FilterScale == CAN_FILTERSCALE_16BIT)
    {
      /* 16-bit scale for the filter */
      CLEAR_BIT(can_ip->FS1R, filternbrbitpos);

      /* First 16-bit identifier and First 16-bit mask */
      /* Or First 16-bit identifier and Second 16-bit identifier */
      can_ip->sFilterRegister[sFilterConfig->FilterBank].FR1 =
        ((0x0000FFFFU & (uint32_t)sFilterConfig->FilterMaskIdLow) << 16U) |
        (0x0000FFFFU & (uint32_t)sFilterConfig->FilterIdLow);

      /* Second 16-bit identifier and Second 16-bit mask */
      /* Or Third 16-bit identifier and Fourth 16-bit identifier */
      can_ip->sFilterRegister[sFilterConfig->FilterBank].FR2 =
        ((0x0000FFFFU & (uint32_t)sFilterConfig->FilterMaskIdHigh) << 16U) |
        (0x0000FFFFU & (uint32_t)sFilterConfig->FilterIdHigh);
    }

    if (sFilterConfig->FilterScale == CAN_FILTERSCALE_32BIT)
    {
      /* 32-bit scale for the filter */
      SET_BIT(can_ip->FS1R, filternbrbitpos);

      /* 32-bit identifier or First 32-bit identifier */
      can_ip->sFilterRegister[sFilterConfig->FilterBank].FR1 =
        ((0x0000FFFFU & (uint32_t)sFilterConfig->FilterIdHigh) << 16U) |
        (0x0000FFFFU & (uint32_t)sFilterConfig->FilterIdLow);

      /* 32-bit mask or Second 32-bit identifier */
      can_ip->sFilterRegister[sFilterConfig->FilterBank].FR2 =
        ((0x0000FFFFU & (uint32_t)sFilterConfig->FilterMaskIdHigh) << 16U) |
        (0x0000FFFFU & (uint32_t)sFilterConfig->FilterMaskIdLow);
    }

    /* Filter Mode */
    if (sFilterConfig->FilterMode == CAN_FILTERMODE_IDMASK)
    {
      /* Id/Mask mode for the filter*/
      CLEAR_BIT(can_ip->FM1R, filternbrbitpos);
    }
    else /* CAN_FilterInitStruct->CAN_FilterMode == CAN_FilterMode_IdList */
    {
      /* Identifier list mode for the filter*/
      SET_BIT(can_ip->FM1R, filternbrbitpos);
    }

    /* Filter FIFO assignment */
    if (sFilterConfig->FilterFIFOAssignment == CAN_FILTER_FIFO0)
    {
      /* FIFO 0 assignation for the filter */
      CLEAR_BIT(can_ip->FFA1R, filternbrbitpos);
    }
    else
    {
      /* FIFO 1 assignation for the filter */
      SET_BIT(can_ip->FFA1R, filternbrbitpos);
    }

    /* Filter activation */
    if (sFilterConfig->FilterActivation == CAN_FILTER_ENABLE)
    {
      SET_BIT(can_ip->FA1R, filternbrbitpos);
    }

    /* Leave the initialisation mode for the filter */
    CLEAR_BIT(can_ip->FMR, CAN_FMR_FINIT);

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_INITIALIZED;

    return HAL_ERROR;
  }
}

/**
  * @}
  */

/** @defgroup CAN_Exported_Functions_Group3 Control functions
 *  @brief    Control functions
 *
@verbatim
  ==============================================================================
                      ##### Control functions #####
  ==============================================================================
    [..]  This section provides functions allowing to:
      (+) HAL_CAN_Start                    : Start the CAN module
      (+) HAL_CAN_Stop                     : Stop the CAN module
      (+) HAL_CAN_RequestSleep             : Request sleep mode entry.
      (+) HAL_CAN_WakeUp                   : Wake up from sleep mode.
      (+) HAL_CAN_IsSleepActive            : Check is sleep mode is active.
      (+) HAL_CAN_AddTxMessage             : Add a message to the Tx mailboxes
                                             and activate the corresponding
                                             transmission request
      (+) HAL_CAN_AbortTxRequest           : Abort transmission request
      (+) HAL_CAN_GetTxMailboxesFreeLevel  : Return Tx mailboxes free level
      (+) HAL_CAN_IsTxMessagePending       : Check if a transmission request is
                                             pending on the selected Tx mailbox
      (+) HAL_CAN_GetRxMessage             : Get a CAN frame from the Rx FIFO
      (+) HAL_CAN_GetRxFifoFillLevel       : Return Rx FIFO fill level

@endverbatim
  * @{
  */

/**
  * @brief  Start the CAN module.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_Start(CAN_HandleTypeDef *hcan)
{
  uint32_t tickstart;

  if (hcan->State == HAL_CAN_STATE_READY)
  {
    /* Change CAN peripheral state */
    hcan->State = HAL_CAN_STATE_LISTENING;

    /* Request leave initialisation */
    CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_INRQ);

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait the acknowledge */
    while ((hcan->Instance->MSR & CAN_MSR_INAK) != 0U)
    {
      /* Check for the Timeout */
      if ((HAL_GetTick() - tickstart) > CAN_TIMEOUT_VALUE)
      {
        /* Update error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_TIMEOUT;

        /* Change CAN state */
        hcan->State = HAL_CAN_STATE_ERROR;

        return HAL_ERROR;
      }
    }

    /* Reset the CAN ErrorCode */
    hcan->ErrorCode = HAL_CAN_ERROR_NONE;

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_READY;

    return HAL_ERROR;
  }
}

/**
  * @brief  Stop the CAN module and enable access to configuration registers.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_Stop(CAN_HandleTypeDef *hcan)
{
  uint32_t tickstart;

  if (hcan->State == HAL_CAN_STATE_LISTENING)
  {
    /* Request initialisation */
    SET_BIT(hcan->Instance->MCR, CAN_MCR_INRQ);

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Wait the acknowledge */
    while ((hcan->Instance->MSR & CAN_MSR_INAK) == 0U)
    {
      /* Check for the Timeout */
      if ((HAL_GetTick() - tickstart) > CAN_TIMEOUT_VALUE)
      {
        /* Update error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_TIMEOUT;

        /* Change CAN state */
        hcan->State = HAL_CAN_STATE_ERROR;

        return HAL_ERROR;
      }
    }

    /* Exit from sleep mode */
    CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_SLEEP);

    /* Change CAN peripheral state */
    hcan->State = HAL_CAN_STATE_READY;

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_STARTED;

    return HAL_ERROR;
  }
}

/**
  * @brief  Request the sleep mode (low power) entry.
  *         When returning from this function, Sleep mode will be entered
  *         as soon as the current CAN activity (transmission or reception
  *         of a CAN frame) has been completed.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL status.
  */
HAL_StatusTypeDef HAL_CAN_RequestSleep(CAN_HandleTypeDef *hcan)
{
  HAL_CAN_StateTypeDef state = hcan->State;

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Request Sleep mode */
    SET_BIT(hcan->Instance->MCR, CAN_MCR_SLEEP);

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_INITIALIZED;

    /* Return function status */
    return HAL_ERROR;
  }
}

/**
  * @brief  Wake up from sleep mode.
  *         When returning with HAL_OK status from this function, Sleep mode
  *         is exited.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL status.
  */
HAL_StatusTypeDef HAL_CAN_WakeUp(CAN_HandleTypeDef *hcan)
{
  __IO uint32_t count = 0;
  uint32_t timeout = 1000000U;
  HAL_CAN_StateTypeDef state = hcan->State;

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Wake up request */
    CLEAR_BIT(hcan->Instance->MCR, CAN_MCR_SLEEP);

    /* Wait sleep mode is exited */
    do
    {
      /* Increment counter */
      count++;

      /* Check if timeout is reached */
      if (count > timeout)
      {
        /* Update error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_TIMEOUT;

        return HAL_ERROR;
      }
    }
    while ((hcan->Instance->MSR & CAN_MSR_SLAK) != 0U);

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_INITIALIZED;

    return HAL_ERROR;
  }
}

/**
  * @brief  Check is sleep mode is active.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval Status
  *          - 0 : Sleep mode is not active.
  *          - 1 : Sleep mode is active.
  */
uint32_t HAL_CAN_IsSleepActive(CAN_HandleTypeDef *hcan)
{
  uint32_t status = 0U;
  HAL_CAN_StateTypeDef state = hcan->State;

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Check Sleep mode */
    if ((hcan->Instance->MSR & CAN_MSR_SLAK) != 0U)
    {
      status = 1U;
    }
  }

  /* Return function status */
  return status;
}

/**
  * @brief  Add a message to the first free Tx mailbox and activate the
  *         corresponding transmission request.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  pHeader pointer to a CAN_TxHeaderTypeDef structure.
  * @param  aData array containing the payload of the Tx frame.
  * @param  pTxMailbox pointer to a variable where the function will return
  *         the TxMailbox used to store the Tx message.
  *         This parameter can be a value of @arg CAN_Tx_Mailboxes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_AddTxMessage(CAN_HandleTypeDef *hcan, CAN_TxHeaderTypeDef *pHeader, uint8_t aData[], uint32_t *pTxMailbox)
{
  uint32_t transmitmailbox;
  HAL_CAN_StateTypeDef state = hcan->State;
  uint32_t tsr = READ_REG(hcan->Instance->TSR);

  /* Check the parameters */
  assert_param(IS_CAN_IDTYPE(pHeader->IDE));
  assert_param(IS_CAN_RTR(pHeader->RTR));
  assert_param(IS_CAN_DLC(pHeader->DLC));
  if (pHeader->IDE == CAN_ID_STD)
  {
    assert_param(IS_CAN_STDID(pHeader->StdId));
  }
  else
  {
    assert_param(IS_CAN_EXTID(pHeader->ExtId));
  }
  assert_param(IS_FUNCTIONAL_STATE(pHeader->TransmitGlobalTime));

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Check that all the Tx mailboxes are not full */
    if (((tsr & CAN_TSR_TME0) != 0U) ||
        ((tsr & CAN_TSR_TME1) != 0U) ||
        ((tsr & CAN_TSR_TME2) != 0U))
    {
      /* Select an empty transmit mailbox */
      transmitmailbox = (tsr & CAN_TSR_CODE) >> CAN_TSR_CODE_Pos;

      /* Check transmit mailbox value */
      if (transmitmailbox > 2U)
      {
        /* Update error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_INTERNAL;

        return HAL_ERROR;
      }

      /* Store the Tx mailbox */
      *pTxMailbox = (uint32_t)1 << transmitmailbox;

      /* Set up the Id */
      if (pHeader->IDE == CAN_ID_STD)
      {
        hcan->Instance->sTxMailBox[transmitmailbox].TIR = ((pHeader->StdId << CAN_TI0R_STID_Pos) |
                                                           pHeader->RTR);
      }
      else
      {
        hcan->Instance->sTxMailBox[transmitmailbox].TIR = ((pHeader->ExtId << CAN_TI0R_EXID_Pos) |
                                                           pHeader->IDE |
                                                           pHeader->RTR);
      }

      /* Set up the DLC */
      hcan->Instance->sTxMailBox[transmitmailbox].TDTR = (pHeader->DLC);

      /* Set up the Transmit Global Time mode */
      if (pHeader->TransmitGlobalTime == ENABLE)
      {
        SET_BIT(hcan->Instance->sTxMailBox[transmitmailbox].TDTR, CAN_TDT0R_TGT);
      }

      /* Set up the data field */
      WRITE_REG(hcan->Instance->sTxMailBox[transmitmailbox].TDHR,
                ((uint32_t)aData[7] << CAN_TDH0R_DATA7_Pos) |
                ((uint32_t)aData[6] << CAN_TDH0R_DATA6_Pos) |
                ((uint32_t)aData[5] << CAN_TDH0R_DATA5_Pos) |
                ((uint32_t)aData[4] << CAN_TDH0R_DATA4_Pos));
      WRITE_REG(hcan->Instance->sTxMailBox[transmitmailbox].TDLR,
                ((uint32_t)aData[3] << CAN_TDL0R_DATA3_Pos) |
                ((uint32_t)aData[2] << CAN_TDL0R_DATA2_Pos) |
                ((uint32_t)aData[1] << CAN_TDL0R_DATA1_Pos) |
                ((uint32_t)aData[0] << CAN_TDL0R_DATA0_Pos));

      /* Request transmission */
      SET_BIT(hcan->Instance->sTxMailBox[transmitmailbox].TIR, CAN_TI0R_TXRQ);

      /* Return function status */
      return HAL_OK;
    }
    else
    {
      /* Update error code */
      hcan->ErrorCode |= HAL_CAN_ERROR_PARAM;

      return HAL_ERROR;
    }
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_INITIALIZED;

    return HAL_ERROR;
  }
}

/**
  * @brief  Abort transmission requests
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  TxMailboxes List of the Tx Mailboxes to abort.
  *         This parameter can be any combination of @arg CAN_Tx_Mailboxes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_AbortTxRequest(CAN_HandleTypeDef *hcan, uint32_t TxMailboxes)
{
  HAL_CAN_StateTypeDef state = hcan->State;

  /* Check function parameters */
  assert_param(IS_CAN_TX_MAILBOX_LIST(TxMailboxes));

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Check Tx Mailbox 0 */
    if ((TxMailboxes & CAN_TX_MAILBOX0) != 0U)
    {
      /* Add cancellation request for Tx Mailbox 0 */
      SET_BIT(hcan->Instance->TSR, CAN_TSR_ABRQ0);
    }

    /* Check Tx Mailbox 1 */
    if ((TxMailboxes & CAN_TX_MAILBOX1) != 0U)
    {
      /* Add cancellation request for Tx Mailbox 1 */
      SET_BIT(hcan->Instance->TSR, CAN_TSR_ABRQ1);
    }

    /* Check Tx Mailbox 2 */
    if ((TxMailboxes & CAN_TX_MAILBOX2) != 0U)
    {
      /* Add cancellation request for Tx Mailbox 2 */
      SET_BIT(hcan->Instance->TSR, CAN_TSR_ABRQ2);
    }

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_INITIALIZED;

    return HAL_ERROR;
  }
}

/**
  * @brief  Return Tx Mailboxes free level: number of free Tx Mailboxes.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval Number of free Tx Mailboxes.
  */
uint32_t HAL_CAN_GetTxMailboxesFreeLevel(CAN_HandleTypeDef *hcan)
{
  uint32_t freelevel = 0U;
  HAL_CAN_StateTypeDef state = hcan->State;

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Check Tx Mailbox 0 status */
    if ((hcan->Instance->TSR & CAN_TSR_TME0) != 0U)
    {
      freelevel++;
    }

    /* Check Tx Mailbox 1 status */
    if ((hcan->Instance->TSR & CAN_TSR_TME1) != 0U)
    {
      freelevel++;
    }

    /* Check Tx Mailbox 2 status */
    if ((hcan->Instance->TSR & CAN_TSR_TME2) != 0U)
    {
      freelevel++;
    }
  }

  /* Return Tx Mailboxes free level */
  return freelevel;
}

/**
  * @brief  Check if a transmission request is pending on the selected Tx
  *         Mailboxes.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  TxMailboxes List of Tx Mailboxes to check.
  *         This parameter can be any combination of @arg CAN_Tx_Mailboxes.
  * @retval Status
  *          - 0 : No pending transmission request on any selected Tx Mailboxes.
  *          - 1 : Pending transmission request on at least one of the selected
  *                Tx Mailbox.
  */
uint32_t HAL_CAN_IsTxMessagePending(CAN_HandleTypeDef *hcan, uint32_t TxMailboxes)
{
  uint32_t status = 0U;
  HAL_CAN_StateTypeDef state = hcan->State;

  /* Check function parameters */
  assert_param(IS_CAN_TX_MAILBOX_LIST(TxMailboxes));

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Check pending transmission request on the selected Tx Mailboxes */
    if ((hcan->Instance->TSR & (TxMailboxes << CAN_TSR_TME0_Pos)) != (TxMailboxes << CAN_TSR_TME0_Pos))
    {
      status = 1U;
    }
  }

  /* Return status */
  return status;
}

/**
  * @brief  Return timestamp of Tx message sent, if time triggered communication
            mode is enabled.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  TxMailbox Tx Mailbox where the timestamp of message sent will be
  *         read.
  *         This parameter can be one value of @arg CAN_Tx_Mailboxes.
  * @retval Timestamp of message sent from Tx Mailbox.
  */
uint32_t HAL_CAN_GetTxTimestamp(CAN_HandleTypeDef *hcan, uint32_t TxMailbox)
{
  uint32_t timestamp = 0U;
  uint32_t transmitmailbox;
  HAL_CAN_StateTypeDef state = hcan->State;

  /* Check function parameters */
  assert_param(IS_CAN_TX_MAILBOX(TxMailbox));

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Select the Tx mailbox */
    transmitmailbox = POSITION_VAL(TxMailbox);

    /* Get timestamp */
    timestamp = (hcan->Instance->sTxMailBox[transmitmailbox].TDTR & CAN_TDT0R_TIME) >> CAN_TDT0R_TIME_Pos;
  }

  /* Return the timestamp */
  return timestamp;
}

/**
  * @brief  Get an CAN frame from the Rx FIFO zone into the message RAM.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  RxFifo Fifo number of the received message to be read.
  *         This parameter can be a value of @arg CAN_receive_FIFO_number.
  * @param  pHeader pointer to a CAN_RxHeaderTypeDef structure where the header
  *         of the Rx frame will be stored.
  * @param  aData array where the payload of the Rx frame will be stored.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_GetRxMessage(CAN_HandleTypeDef *hcan, uint32_t RxFifo, CAN_RxHeaderTypeDef *pHeader, uint8_t aData[])
{
  HAL_CAN_StateTypeDef state = hcan->State;

  assert_param(IS_CAN_RX_FIFO(RxFifo));

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Check the Rx FIFO */
    if (RxFifo == CAN_RX_FIFO0) /* Rx element is assigned to Rx FIFO 0 */
    {
      /* Check that the Rx FIFO 0 is not empty */
      if ((hcan->Instance->RF0R & CAN_RF0R_FMP0) == 0U)
      {
        /* Update error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_PARAM;

        return HAL_ERROR;
      }
    }
    else /* Rx element is assigned to Rx FIFO 1 */
    {
      /* Check that the Rx FIFO 1 is not empty */
      if ((hcan->Instance->RF1R & CAN_RF1R_FMP1) == 0U)
      {
        /* Update error code */
        hcan->ErrorCode |= HAL_CAN_ERROR_PARAM;

        return HAL_ERROR;
      }
    }

    /* Get the header */
    pHeader->IDE = CAN_RI0R_IDE & hcan->Instance->sFIFOMailBox[RxFifo].RIR;
    if (pHeader->IDE == CAN_ID_STD)
    {
      pHeader->StdId = (CAN_RI0R_STID & hcan->Instance->sFIFOMailBox[RxFifo].RIR) >> CAN_TI0R_STID_Pos;
    }
    else
    {
      pHeader->ExtId = ((CAN_RI0R_EXID | CAN_RI0R_STID) & hcan->Instance->sFIFOMailBox[RxFifo].RIR) >> CAN_RI0R_EXID_Pos;
    }
    pHeader->RTR = (CAN_RI0R_RTR & hcan->Instance->sFIFOMailBox[RxFifo].RIR);
    pHeader->DLC = (CAN_RDT0R_DLC & hcan->Instance->sFIFOMailBox[RxFifo].RDTR) >> CAN_RDT0R_DLC_Pos;
    pHeader->FilterMatchIndex = (CAN_RDT0R_FMI & hcan->Instance->sFIFOMailBox[RxFifo].RDTR) >> CAN_RDT0R_FMI_Pos;
    pHeader->Timestamp = (CAN_RDT0R_TIME & hcan->Instance->sFIFOMailBox[RxFifo].RDTR) >> CAN_RDT0R_TIME_Pos;

    /* Get the data */
    aData[0] = (uint8_t)((CAN_RDL0R_DATA0 & hcan->Instance->sFIFOMailBox[RxFifo].RDLR) >> CAN_RDL0R_DATA0_Pos);
    aData[1] = (uint8_t)((CAN_RDL0R_DATA1 & hcan->Instance->sFIFOMailBox[RxFifo].RDLR) >> CAN_RDL0R_DATA1_Pos);
    aData[2] = (uint8_t)((CAN_RDL0R_DATA2 & hcan->Instance->sFIFOMailBox[RxFifo].RDLR) >> CAN_RDL0R_DATA2_Pos);
    aData[3] = (uint8_t)((CAN_RDL0R_DATA3 & hcan->Instance->sFIFOMailBox[RxFifo].RDLR) >> CAN_RDL0R_DATA3_Pos);
    aData[4] = (uint8_t)((CAN_RDH0R_DATA4 & hcan->Instance->sFIFOMailBox[RxFifo].RDHR) >> CAN_RDH0R_DATA4_Pos);
    aData[5] = (uint8_t)((CAN_RDH0R_DATA5 & hcan->Instance->sFIFOMailBox[RxFifo].RDHR) >> CAN_RDH0R_DATA5_Pos);
    aData[6] = (uint8_t)((CAN_RDH0R_DATA6 & hcan->Instance->sFIFOMailBox[RxFifo].RDHR) >> CAN_RDH0R_DATA6_Pos);
    aData[7] = (uint8_t)((CAN_RDH0R_DATA7 & hcan->Instance->sFIFOMailBox[RxFifo].RDHR) >> CAN_RDH0R_DATA7_Pos);

    /* Release the FIFO */
    if (RxFifo == CAN_RX_FIFO0) /* Rx element is assigned to Rx FIFO 0 */
    {
      /* Release RX FIFO 0 */
      SET_BIT(hcan->Instance->RF0R, CAN_RF0R_RFOM0);
    }
    else /* Rx element is assigned to Rx FIFO 1 */
    {
      /* Release RX FIFO 1 */
      SET_BIT(hcan->Instance->RF1R, CAN_RF1R_RFOM1);
    }

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_INITIALIZED;

    return HAL_ERROR;
  }
}

/**
  * @brief  Return Rx FIFO fill level.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  RxFifo Rx FIFO.
  *         This parameter can be a value of @arg CAN_receive_FIFO_number.
  * @retval Number of messages available in Rx FIFO.
  */
uint32_t HAL_CAN_GetRxFifoFillLevel(CAN_HandleTypeDef *hcan, uint32_t RxFifo)
{
  uint32_t filllevel = 0U;
  HAL_CAN_StateTypeDef state = hcan->State;

  /* Check function parameters */
  assert_param(IS_CAN_RX_FIFO(RxFifo));

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    if (RxFifo == CAN_RX_FIFO0)
    {
      filllevel = hcan->Instance->RF0R & CAN_RF0R_FMP0;
    }
    else /* RxFifo == CAN_RX_FIFO1 */
    {
      filllevel = hcan->Instance->RF1R & CAN_RF1R_FMP1;
    }
  }

  /* Return Rx FIFO fill level */
  return filllevel;
}

/**
  * @}
  */

/** @defgroup CAN_Exported_Functions_Group4 Interrupts management
 *  @brief    Interrupts management
 *
@verbatim
  ==============================================================================
                       ##### Interrupts management #####
  ==============================================================================
    [..]  This section provides functions allowing to:
      (+) HAL_CAN_ActivateNotification      : Enable interrupts
      (+) HAL_CAN_DeactivateNotification    : Disable interrupts
      (+) HAL_CAN_IRQHandler                : Handles CAN interrupt request

@endverbatim
  * @{
  */

/**
  * @brief  Enable interrupts.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  ActiveITs indicates which interrupts will be enabled.
  *         This parameter can be any combination of @arg CAN_Interrupts.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_ActivateNotification(CAN_HandleTypeDef *hcan, uint32_t ActiveITs)
{
  HAL_CAN_StateTypeDef state = hcan->State;

  /* Check function parameters */
  assert_param(IS_CAN_IT(ActiveITs));

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Enable the selected interrupts */
    __HAL_CAN_ENABLE_IT(hcan, ActiveITs);

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_INITIALIZED;

    return HAL_ERROR;
  }
}

/**
  * @brief  Disable interrupts.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  InactiveITs indicates which interrupts will be disabled.
  *         This parameter can be any combination of @arg CAN_Interrupts.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_DeactivateNotification(CAN_HandleTypeDef *hcan, uint32_t InactiveITs)
{
  HAL_CAN_StateTypeDef state = hcan->State;

  /* Check function parameters */
  assert_param(IS_CAN_IT(InactiveITs));

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Disable the selected interrupts */
    __HAL_CAN_DISABLE_IT(hcan, InactiveITs);

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_INITIALIZED;

    return HAL_ERROR;
  }
}

/**
  * @brief  Handles CAN interrupt request
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
void HAL_CAN_IRQHandler(CAN_HandleTypeDef *hcan)
{
  uint32_t errorcode = HAL_CAN_ERROR_NONE;
  uint32_t interrupts = READ_REG(hcan->Instance->IER);
  uint32_t msrflags = READ_REG(hcan->Instance->MSR);
  uint32_t tsrflags = READ_REG(hcan->Instance->TSR);
  uint32_t rf0rflags = READ_REG(hcan->Instance->RF0R);
  uint32_t rf1rflags = READ_REG(hcan->Instance->RF1R);
  uint32_t esrflags = READ_REG(hcan->Instance->ESR);

  /* Transmit Mailbox empty interrupt management *****************************/
  if ((interrupts & CAN_IT_TX_MAILBOX_EMPTY) != 0U)
  {
    /* Transmit Mailbox 0 management *****************************************/
    if ((tsrflags & CAN_TSR_RQCP0) != 0U)
    {
      /* Clear the Transmission Complete flag (and TXOK0,ALST0,TERR0 bits) */
      __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_RQCP0);

      if ((tsrflags & CAN_TSR_TXOK0) != 0U)
      {
        /* Transmission Mailbox 0 complete callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
        /* Call registered callback*/
        hcan->TxMailbox0CompleteCallback(hcan);
#else
        /* Call weak (surcharged) callback */
        HAL_CAN_TxMailbox0CompleteCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
      }
      else
      {
        if ((tsrflags & CAN_TSR_ALST0) != 0U)
        {
          /* Update error code */
          errorcode |= HAL_CAN_ERROR_TX_ALST0;
        }
        else if ((tsrflags & CAN_TSR_TERR0) != 0U)
        {
          /* Update error code */
          errorcode |= HAL_CAN_ERROR_TX_TERR0;
        }
        else
        {
          /* Transmission Mailbox 0 abort callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
          /* Call registered callback*/
          hcan->TxMailbox0AbortCallback(hcan);
#else
          /* Call weak (surcharged) callback */
          HAL_CAN_TxMailbox0AbortCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
        }
      }
    }

    /* Transmit Mailbox 1 management *****************************************/
    if ((tsrflags & CAN_TSR_RQCP1) != 0U)
    {
      /* Clear the Transmission Complete flag (and TXOK1,ALST1,TERR1 bits) */
      __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_RQCP1);

      if ((tsrflags & CAN_TSR_TXOK1) != 0U)
      {
        /* Transmission Mailbox 1 complete callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
        /* Call registered callback*/
        hcan->TxMailbox1CompleteCallback(hcan);
#else
        /* Call weak (surcharged) callback */
        HAL_CAN_TxMailbox1CompleteCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
      }
      else
      {
        if ((tsrflags & CAN_TSR_ALST1) != 0U)
        {
          /* Update error code */
          errorcode |= HAL_CAN_ERROR_TX_ALST1;
        }
        else if ((tsrflags & CAN_TSR_TERR1) != 0U)
        {
          /* Update error code */
          errorcode |= HAL_CAN_ERROR_TX_TERR1;
        }
        else
        {
          /* Transmission Mailbox 1 abort callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
          /* Call registered callback*/
          hcan->TxMailbox1AbortCallback(hcan);
#else
          /* Call weak (surcharged) callback */
          HAL_CAN_TxMailbox1AbortCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
        }
      }
    }

    /* Transmit Mailbox 2 management *****************************************/
    if ((tsrflags & CAN_TSR_RQCP2) != 0U)
    {
      /* Clear the Transmission Complete flag (and TXOK2,ALST2,TERR2 bits) */
      __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_RQCP2);

      if ((tsrflags & CAN_TSR_TXOK2) != 0U)
      {
        /* Transmission Mailbox 2 complete callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
        /* Call registered callback*/
        hcan->TxMailbox2CompleteCallback(hcan);
#else
        /* Call weak (surcharged) callback */
        HAL_CAN_TxMailbox2CompleteCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
      }
      else
      {
        if ((tsrflags & CAN_TSR_ALST2) != 0U)
        {
          /* Update error code */
          errorcode |= HAL_CAN_ERROR_TX_ALST2;
        }
        else if ((tsrflags & CAN_TSR_TERR2) != 0U)
        {
          /* Update error code */
          errorcode |= HAL_CAN_ERROR_TX_TERR2;
        }
        else
        {
          /* Transmission Mailbox 2 abort callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
          /* Call registered callback*/
          hcan->TxMailbox2AbortCallback(hcan);
#else
          /* Call weak (surcharged) callback */
          HAL_CAN_TxMailbox2AbortCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
        }
      }
    }
  }

  /* Receive FIFO 0 overrun interrupt management *****************************/
  if ((interrupts & CAN_IT_RX_FIFO0_OVERRUN) != 0U)
  {
    if ((rf0rflags & CAN_RF0R_FOVR0) != 0U)
    {
      /* Set CAN error code to Rx Fifo 0 overrun error */
      errorcode |= HAL_CAN_ERROR_RX_FOV0;

      /* Clear FIFO0 Overrun Flag */
      __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_FOV0);
    }
  }

  /* Receive FIFO 0 full interrupt management ********************************/
  if ((interrupts & CAN_IT_RX_FIFO0_FULL) != 0U)
  {
    if ((rf0rflags & CAN_RF0R_FULL0) != 0U)
    {
      /* Clear FIFO 0 full Flag */
      __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_FF0);

      /* Receive FIFO 0 full Callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
      /* Call registered callback*/
      hcan->RxFifo0FullCallback(hcan);
#else
      /* Call weak (surcharged) callback */
      HAL_CAN_RxFifo0FullCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
    }
  }

  /* Receive FIFO 0 message pending interrupt management *********************/
  if ((interrupts & CAN_IT_RX_FIFO0_MSG_PENDING) != 0U)
  {
    /* Check if message is still pending */
    if ((hcan->Instance->RF0R & CAN_RF0R_FMP0) != 0U)
    {
      /* Receive FIFO 0 message pending Callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
      /* Call registered callback*/
      hcan->RxFifo0MsgPendingCallback(hcan);
#else
      /* Call weak (surcharged) callback */
      HAL_CAN_RxFifo0MsgPendingCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
    }
  }

  /* Receive FIFO 1 overrun interrupt management *****************************/
  if ((interrupts & CAN_IT_RX_FIFO1_OVERRUN) != 0U)
  {
    if ((rf1rflags & CAN_RF1R_FOVR1) != 0U)
    {
      /* Set CAN error code to Rx Fifo 1 overrun error */
      errorcode |= HAL_CAN_ERROR_RX_FOV1;

      /* Clear FIFO1 Overrun Flag */
      __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_FOV1);
    }
  }

  /* Receive FIFO 1 full interrupt management ********************************/
  if ((interrupts & CAN_IT_RX_FIFO1_FULL) != 0U)
  {
    if ((rf1rflags & CAN_RF1R_FULL1) != 0U)
    {
      /* Clear FIFO 1 full Flag */
      __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_FF1);

      /* Receive FIFO 1 full Callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
      /* Call registered callback*/
      hcan->RxFifo1FullCallback(hcan);
#else
      /* Call weak (surcharged) callback */
      HAL_CAN_RxFifo1FullCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
    }
  }

  /* Receive FIFO 1 message pending interrupt management *********************/
  if ((interrupts & CAN_IT_RX_FIFO1_MSG_PENDING) != 0U)
  {
    /* Check if message is still pending */
    if ((hcan->Instance->RF1R & CAN_RF1R_FMP1) != 0U)
    {
      /* Receive FIFO 1 message pending Callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
      /* Call registered callback*/
      hcan->RxFifo1MsgPendingCallback(hcan);
#else
      /* Call weak (surcharged) callback */
      HAL_CAN_RxFifo1MsgPendingCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
    }
  }

  /* Sleep interrupt management *********************************************/
  if ((interrupts & CAN_IT_SLEEP_ACK) != 0U)
  {
    if ((msrflags & CAN_MSR_SLAKI) != 0U)
    {
      /* Clear Sleep interrupt Flag */
      __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_SLAKI);

      /* Sleep Callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
      /* Call registered callback*/
      hcan->SleepCallback(hcan);
#else
      /* Call weak (surcharged) callback */
      HAL_CAN_SleepCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
    }
  }

  /* WakeUp interrupt management *********************************************/
  if ((interrupts & CAN_IT_WAKEUP) != 0U)
  {
    if ((msrflags & CAN_MSR_WKUI) != 0U)
    {
      /* Clear WakeUp Flag */
      __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_WKU);

      /* WakeUp Callback */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
      /* Call registered callback*/
      hcan->WakeUpFromRxMsgCallback(hcan);
#else
      /* Call weak (surcharged) callback */
      HAL_CAN_WakeUpFromRxMsgCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
    }
  }

  /* Error interrupts management *********************************************/
  if ((interrupts & CAN_IT_ERROR) != 0U)
  {
    if ((msrflags & CAN_MSR_ERRI) != 0U)
    {
      /* Check Error Warning Flag */
      if (((interrupts & CAN_IT_ERROR_WARNING) != 0U) &&
          ((esrflags & CAN_ESR_EWGF) != 0U))
      {
        /* Set CAN error code to Error Warning */
        errorcode |= HAL_CAN_ERROR_EWG;

        /* No need for clear of Error Warning Flag as read-only */
      }

      /* Check Error Passive Flag */
      if (((interrupts & CAN_IT_ERROR_PASSIVE) != 0U) &&
          ((esrflags & CAN_ESR_EPVF) != 0U))
      {
        /* Set CAN error code to Error Passive */
        errorcode |= HAL_CAN_ERROR_EPV;

        /* No need for clear of Error Passive Flag as read-only */
      }

      /* Check Bus-off Flag */
      if (((interrupts & CAN_IT_BUSOFF) != 0U) &&
          ((esrflags & CAN_ESR_BOFF) != 0U))
      {
        /* Set CAN error code to Bus-Off */
        errorcode |= HAL_CAN_ERROR_BOF;

        /* No need for clear of Error Bus-Off as read-only */
      }

      /* Check Last Error Code Flag */
      if (((interrupts & CAN_IT_LAST_ERROR_CODE) != 0U) &&
          ((esrflags & CAN_ESR_LEC) != 0U))
      {
        switch (esrflags & CAN_ESR_LEC)
        {
          case (CAN_ESR_LEC_0):
            /* Set CAN error code to Stuff error */
            errorcode |= HAL_CAN_ERROR_STF;
            break;
          case (CAN_ESR_LEC_1):
            /* Set CAN error code to Form error */
            errorcode |= HAL_CAN_ERROR_FOR;
            break;
          case (CAN_ESR_LEC_1 | CAN_ESR_LEC_0):
            /* Set CAN error code to Acknowledgement error */
            errorcode |= HAL_CAN_ERROR_ACK;
            break;
          case (CAN_ESR_LEC_2):
            /* Set CAN error code to Bit recessive error */
            errorcode |= HAL_CAN_ERROR_BR;
            break;
          case (CAN_ESR_LEC_2 | CAN_ESR_LEC_0):
            /* Set CAN error code to Bit Dominant error */
            errorcode |= HAL_CAN_ERROR_BD;
            break;
          case (CAN_ESR_LEC_2 | CAN_ESR_LEC_1):
            /* Set CAN error code to CRC error */
            errorcode |= HAL_CAN_ERROR_CRC;
            break;
          default:
            break;
        }

        /* Clear Last error code Flag */
        CLEAR_BIT(hcan->Instance->ESR, CAN_ESR_LEC);
      }
    }

    /* Clear ERRI Flag */
    __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_ERRI);
  }

  /* Call the Error call Back in case of Errors */
  if (errorcode != HAL_CAN_ERROR_NONE)
  {
    /* Update error code in handle */
    hcan->ErrorCode |= errorcode;

    /* Call Error callback function */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
    /* Call registered callback*/
    hcan->ErrorCallback(hcan);
#else
    /* Call weak (surcharged) callback */
    HAL_CAN_ErrorCallback(hcan);
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
  }
}

/**
  * @}
  */

/** @defgroup CAN_Exported_Functions_Group5 Callback functions
 *  @brief   CAN Callback functions
 *
@verbatim
  ==============================================================================
                          ##### Callback functions #####
  ==============================================================================
    [..]
    This subsection provides the following callback functions:
      (+) HAL_CAN_TxMailbox0CompleteCallback
      (+) HAL_CAN_TxMailbox1CompleteCallback
      (+) HAL_CAN_TxMailbox2CompleteCallback
      (+) HAL_CAN_TxMailbox0AbortCallback
      (+) HAL_CAN_TxMailbox1AbortCallback
      (+) HAL_CAN_TxMailbox2AbortCallback
      (+) HAL_CAN_RxFifo0MsgPendingCallback
      (+) HAL_CAN_RxFifo0FullCallback
      (+) HAL_CAN_RxFifo1MsgPendingCallback
      (+) HAL_CAN_RxFifo1FullCallback
      (+) HAL_CAN_SleepCallback
      (+) HAL_CAN_WakeUpFromRxMsgCallback
      (+) HAL_CAN_ErrorCallback

@endverbatim
  * @{
  */

/**
  * @brief  Transmission Mailbox 0 complete callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_TxMailbox0CompleteCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_TxMailbox0CompleteCallback could be implemented in the
            user file
   */
}

/**
  * @brief  Transmission Mailbox 1 complete callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_TxMailbox1CompleteCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_TxMailbox1CompleteCallback could be implemented in the
            user file
   */
}

/**
  * @brief  Transmission Mailbox 2 complete callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_TxMailbox2CompleteCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_TxMailbox2CompleteCallback could be implemented in the
            user file
   */
}

/**
  * @brief  Transmission Mailbox 0 Cancellation callback.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_TxMailbox0AbortCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_TxMailbox0AbortCallback could be implemented in the
            user file
   */
}

/**
  * @brief  Transmission Mailbox 1 Cancellation callback.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_TxMailbox1AbortCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_TxMailbox1AbortCallback could be implemented in the
            user file
   */
}

/**
  * @brief  Transmission Mailbox 2 Cancellation callback.
  * @param  hcan pointer to an CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_TxMailbox2AbortCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_TxMailbox2AbortCallback could be implemented in the
            user file
   */
}

/**
  * @brief  Rx FIFO 0 message pending callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_RxFifo0MsgPendingCallback could be implemented in the
            user file
   */
}

/**
  * @brief  Rx FIFO 0 full callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_RxFifo0FullCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_RxFifo0FullCallback could be implemented in the user
            file
   */
}

/**
  * @brief  Rx FIFO 1 message pending callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_RxFifo1MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_RxFifo1MsgPendingCallback could be implemented in the
            user file
   */
}

/**
  * @brief  Rx FIFO 1 full callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_RxFifo1FullCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_RxFifo1FullCallback could be implemented in the user
            file
   */
}

/**
  * @brief  Sleep callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_SleepCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_SleepCallback could be implemented in the user file
   */
}

/**
  * @brief  WakeUp from Rx message callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_WakeUpFromRxMsgCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_WakeUpFromRxMsgCallback could be implemented in the
            user file
   */
}

/**
  * @brief  Error CAN callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_ErrorCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_ErrorCallback could be implemented in the user file
   */
}

/**
  * @}
  */

/** @defgroup CAN_Exported_Functions_Group6 Peripheral State and Error functions
 *  @brief   CAN Peripheral State functions
 *
@verbatim
  ==============================================================================
            ##### Peripheral State and Error functions #####
  ==============================================================================
    [..]
    This subsection provides functions allowing to :
      (+) HAL_CAN_GetState()  : Return the CAN state.
      (+) HAL_CAN_GetError()  : Return the CAN error codes if any.
      (+) HAL_CAN_ResetError(): Reset the CAN error codes if any.

@endverbatim
  * @{
  */

/**
  * @brief  Return the CAN state.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL state
  */
HAL_CAN_StateTypeDef HAL_CAN_GetState(CAN_HandleTypeDef *hcan)
{
  HAL_CAN_StateTypeDef state = hcan->State;

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Check sleep mode acknowledge flag */
    if ((hcan->Instance->MSR & CAN_MSR_SLAK) != 0U)
    {
      /* Sleep mode is active */
      state = HAL_CAN_STATE_SLEEP_ACTIVE;
    }
    /* Check sleep mode request flag */
    else if ((hcan->Instance->MCR & CAN_MCR_SLEEP) != 0U)
    {
      /* Sleep mode request is pending */
      state = HAL_CAN_STATE_SLEEP_PENDING;
    }
    else
    {
      /* Neither sleep mode request nor sleep mode acknowledge */
    }
  }

  /* Return CAN state */
  return state;
}

/**
  * @brief  Return the CAN error code.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval CAN Error Code
  */
uint32_t HAL_CAN_GetError(CAN_HandleTypeDef *hcan)
{
  /* Return CAN error code */
  return hcan->ErrorCode;
}

/**
  * @brief  Reset the CAN error code.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_ResetError(CAN_HandleTypeDef *hcan)
{
  HAL_StatusTypeDef status = HAL_OK;
  HAL_CAN_StateTypeDef state = hcan->State;

  if ((state == HAL_CAN_STATE_READY) ||
      (state == HAL_CAN_STATE_LISTENING))
  {
    /* Reset CAN error code */
    hcan->ErrorCode = 0U;
  }
  else
  {
    /* Update error code */
    hcan->ErrorCode |= HAL_CAN_ERROR_NOT_INITIALIZED;

    status = HAL_ERROR;
  }

  /* Return the status */
  return status;
}

/**
  * @}
  */

/**
  * @}
  */

#endif /* HAL_CAN_MODULE_ENABLED */

/**
  * @}
  */

#endif /* CAN1 */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
