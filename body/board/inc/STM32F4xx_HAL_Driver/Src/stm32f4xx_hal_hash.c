/**
  ******************************************************************************
  * @file    stm32f4xx_hal_hash.c
  * @author  MCD Application Team
  * @brief   HASH HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the HASH peripheral:
  *           + Initialization and de-initialization methods
  *           + HASH or HMAC processing in polling mode
  *           + HASH or HMAC processing in interrupt mode
  *           + HASH or HMAC processing in DMA mode
  *           + Peripheral State methods
  *           + HASH or HMAC processing suspension/resumption
  *
  @verbatim
 ===============================================================================
                     ##### How to use this driver #####
 ===============================================================================
    [..]
    The HASH HAL driver can be used as follows:

    (#)Initialize the HASH low level resources by implementing the HAL_HASH_MspInit():
        (##) Enable the HASH interface clock using __HASH_CLK_ENABLE()
        (##) When resorting to interrupt-based APIs (e.g. HAL_HASH_xxx_Start_IT())
            (+++) Configure the HASH interrupt priority using HAL_NVIC_SetPriority()
            (+++) Enable the HASH IRQ handler using HAL_NVIC_EnableIRQ()
            (+++) In HASH IRQ handler, call HAL_HASH_IRQHandler() API
        (##) When resorting to DMA-based APIs  (e.g. HAL_HASH_xxx_Start_DMA())
            (+++) Enable the DMAx interface clock using
                   __DMAx_CLK_ENABLE()
            (+++) Configure and enable one DMA stream to manage data transfer from
                memory to peripheral (input stream). Managing data transfer from
                peripheral to memory can be performed only using CPU.
            (+++) Associate the initialized DMA handle to the HASH DMA handle
                using  __HAL_LINKDMA()
            (+++) Configure the priority and enable the NVIC for the transfer complete
                interrupt on the DMA stream: use
                 HAL_NVIC_SetPriority() and
                 HAL_NVIC_EnableIRQ()

    (#)Initialize the HASH HAL using HAL_HASH_Init(). This function:
        (##) resorts to HAL_HASH_MspInit() for low-level initialization,
        (##) configures the data type: 1-bit, 8-bit, 16-bit or 32-bit.

    (#)Three processing schemes are available:
        (##) Polling mode: processing APIs are blocking functions
             i.e. they process the data and wait till the digest computation is finished,
             e.g. HAL_HASH_xxx_Start() for HASH or HAL_HMAC_xxx_Start() for HMAC
        (##) Interrupt mode: processing APIs are not blocking functions
                i.e. they process the data under interrupt,
                e.g. HAL_HASH_xxx_Start_IT() for HASH or HAL_HMAC_xxx_Start_IT() for HMAC
        (##) DMA mode: processing APIs are not blocking functions and the CPU is
             not used for data transfer i.e. the data transfer is ensured by DMA,
                e.g. HAL_HASH_xxx_Start_DMA() for HASH or HAL_HMAC_xxx_Start_DMA()
                for HMAC. Note that in DMA mode, a call to HAL_HASH_xxx_Finish()
                is then required to retrieve the digest.

    (#)When the processing function is called after HAL_HASH_Init(), the HASH peripheral is
       initialized and processes the buffer fed in input. When the input data have all been
       fed to the Peripheral, the digest computation can start.

    (#)Multi-buffer processing is possible in polling, interrupt and DMA modes.
        (##) In polling mode, only multi-buffer HASH processing is possible.
             API HAL_HASH_xxx_Accumulate() must be called for each input buffer, except for the last one.
             User must resort to HAL_HASH_xxx_Accumulate_End() to enter the last one and retrieve as
             well the computed digest.

        (##) In interrupt mode, API HAL_HASH_xxx_Accumulate_IT() must be called for each input buffer,
             except for the last one.
             User must resort to HAL_HASH_xxx_Accumulate_End_IT() to enter the last one and retrieve as
             well the computed digest.

        (##) In DMA mode, multi-buffer HASH and HMAC processing are possible.
              (+++) HASH processing: once initialization is done, MDMAT bit must be set
               through __HAL_HASH_SET_MDMAT() macro.
             From that point, each buffer can be fed to the Peripheral through HAL_HASH_xxx_Start_DMA() API.
             Before entering the last buffer, reset the MDMAT bit with __HAL_HASH_RESET_MDMAT()
             macro then wrap-up the HASH processing in feeding the last input buffer through the
             same API HAL_HASH_xxx_Start_DMA(). The digest can then be retrieved with a call to
             API HAL_HASH_xxx_Finish().
             (+++) HMAC processing (requires to resort to extended functions):
             after initialization, the key and the first input buffer are entered
             in the Peripheral with the API HAL_HMACEx_xxx_Step1_2_DMA(). This carries out HMAC step 1 and
             starts step 2.
             The following buffers are next entered with the API  HAL_HMACEx_xxx_Step2_DMA(). At this
             point, the HMAC processing is still carrying out step 2.
             Then, step 2 for the last input buffer and step 3 are carried out by a single call
             to HAL_HMACEx_xxx_Step2_3_DMA().

             The digest can finally be retrieved with a call to API HAL_HASH_xxx_Finish().


    (#)Context swapping.
        (##) Two APIs are available to suspend HASH or HMAC processing:
             (+++) HAL_HASH_SwFeed_ProcessSuspend() when data are entered by software (polling or IT mode),
             (+++) HAL_HASH_DMAFeed_ProcessSuspend() when data are entered by DMA.

        (##) When HASH or HMAC processing is suspended, HAL_HASH_ContextSaving() allows
            to save in memory the Peripheral context. This context can be restored afterwards
            to resume the HASH processing thanks to HAL_HASH_ContextRestoring().

        (##) Once the HASH Peripheral has been restored to the same configuration as that at suspension
             time, processing can be restarted with the same API call (same API, same handle,
             same parameters) as done before the suspension. Relevant parameters to restart at
             the proper location are internally saved in the HASH handle.

    (#)Call HAL_HASH_DeInit() to deinitialize the HASH peripheral.

     *** Remarks on message length ***
     ===================================
     [..]
      (#) HAL in interruption mode (interruptions driven)

        (##)Due to HASH peripheral hardware design, the peripheral interruption is triggered every 64 bytes.
        This is why, for driver implementation simplicity’s sake, user is requested to enter a message the
        length of which is a multiple of 4 bytes.

        (##) When the message length (in bytes) is not a multiple of words, a specific field exists in HASH_STR
        to specify which bits to discard at the end of the complete message to process only the message bits
        and not extra bits.

        (##) If user needs to perform a hash computation of a large input buffer that is spread around various places
        in memory and where each piece of this input buffer is not necessarily a multiple of 4 bytes in size, it becomes
        necessary to use a temporary buffer to format the data accordingly before feeding them to the Peripheral.
        It is advised to the user to
       (+++) achieve the first formatting operation by software then enter the data
       (+++) while the Peripheral is processing the first input set, carry out the second formatting
        operation by software, to be ready when DINIS occurs.
       (+++) repeat step 2 until the whole message is processed.

     [..]
      (#) HAL in DMA mode

        (##) Again, due to hardware design, the DMA transfer to feed the data can only be done on a word-basis.
        The same field described above in HASH_STR is used to specify which bits to discard at the end of the
        DMA transfer to process only the message bits and not extra bits. Due to hardware implementation,
        this is possible only at the end of the complete message. When several DMA transfers are needed to
        enter the message, this is not applicable at the end of the intermediary transfers.

        (##) Similarly to the interruption-driven mode, it is suggested to the user to format the consecutive
        chunks of data by software while the DMA transfer and processing is on-going for the first parts of
        the message. Due to the 32-bit alignment required for the DMA transfer, it is underlined that the
        software formatting operation is more complex than in the IT mode.

     *** Callback registration ***
     ===================================
     [..]
      (#) The compilation define  USE_HAL_HASH_REGISTER_CALLBACKS when set to 1
          allows the user to configure dynamically the driver callbacks.
          Use function @ref HAL_HASH_RegisterCallback() to register a user callback.

      (#) Function @ref HAL_HASH_RegisterCallback() allows to register following callbacks:
            (+) InCpltCallback    : callback for input completion.
            (+) DgstCpltCallback  : callback for digest computation completion.
            (+) ErrorCallback     : callback for error.
            (+) MspInitCallback   : HASH MspInit.
            (+) MspDeInitCallback : HASH MspDeInit.
          This function takes as parameters the HAL peripheral handle, the Callback ID
          and a pointer to the user callback function.

      (#) Use function @ref HAL_HASH_UnRegisterCallback() to reset a callback to the default
          weak (surcharged) function.
          @ref HAL_HASH_UnRegisterCallback() takes as parameters the HAL peripheral handle,
          and the Callback ID.
          This function allows to reset following callbacks:
            (+) InCpltCallback    : callback for input completion.
            (+) DgstCpltCallback  : callback for digest computation completion.
            (+) ErrorCallback     : callback for error.
            (+) MspInitCallback   : HASH MspInit.
            (+) MspDeInitCallback : HASH MspDeInit.

      (#) By default, after the @ref HAL_HASH_Init and if the state is HAL_HASH_STATE_RESET
          all callbacks are reset to the corresponding legacy weak (surcharged) functions:
          examples @ref HAL_HASH_InCpltCallback(), @ref HAL_HASH_DgstCpltCallback()
          Exception done for MspInit and MspDeInit callbacks that are respectively
          reset to the legacy weak (surcharged) functions in the @ref HAL_HASH_Init
          and @ref HAL_HASH_DeInit only when these callbacks are null (not registered beforehand)
          If not, MspInit or MspDeInit are not null, the @ref HAL_HASH_Init and @ref HAL_HASH_DeInit
          keep and use the user MspInit/MspDeInit callbacks (registered beforehand).

          Callbacks can be registered/unregistered in READY state only.
          Exception done for MspInit/MspDeInit callbacks that can be registered/unregistered
          in READY or RESET state, thus registered (user) MspInit/DeInit callbacks can be used
          during the Init/DeInit.
          In that case first register the MspInit/MspDeInit user callbacks
          using @ref HAL_HASH_RegisterCallback before calling @ref HAL_HASH_DeInit
          or @ref HAL_HASH_Init function.

          When The compilation define USE_HAL_HASH_REGISTER_CALLBACKS is set to 0 or
          not defined, the callback registering feature is not available
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
#if defined (HASH)

/** @defgroup HASH  HASH
  * @brief HASH HAL module driver.
  * @{
  */

#ifdef HAL_HASH_MODULE_ENABLED

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @defgroup HASH_Private_Constants HASH Private Constants
  * @{
  */

/** @defgroup HASH_Digest_Calculation_Status HASH Digest Calculation Status
  * @{
  */
#define HASH_DIGEST_CALCULATION_NOT_STARTED       ((uint32_t)0x00000000U) /*!< DCAL not set after input data written in DIN register */
#define HASH_DIGEST_CALCULATION_STARTED           ((uint32_t)0x00000001U) /*!< DCAL set after input data written in DIN register     */
/**
  * @}
  */

/** @defgroup HASH_Number_Of_CSR_Registers HASH Number of Context Swap Registers
  * @{
  */
#define HASH_NUMBER_OF_CSR_REGISTERS              54U     /*!< Number of Context Swap Registers */
/**
  * @}
  */

/** @defgroup HASH_TimeOut_Value HASH TimeOut Value
  * @{
  */
#define HASH_TIMEOUTVALUE                         1000U   /*!< Time-out value  */
/**
  * @}
  */

/** @defgroup HASH_DMA_Suspension_Words_Limit HASH DMA suspension words limit
  * @{
  */
#define HASH_DMA_SUSPENSION_WORDS_LIMIT             20U   /*!< Number of words below which DMA suspension is aborted */
/**
  * @}
  */

/**
  * @}
  */

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/** @defgroup HASH_Private_Functions HASH Private Functions
  * @{
  */
static void HASH_DMAXferCplt(DMA_HandleTypeDef *hdma);
static void HASH_DMAError(DMA_HandleTypeDef *hdma);
static void HASH_GetDigest(uint8_t *pMsgDigest, uint8_t Size);
static HAL_StatusTypeDef HASH_WaitOnFlagUntilTimeout(HASH_HandleTypeDef *hhash, uint32_t Flag, FlagStatus Status,
                                                     uint32_t Timeout);
static HAL_StatusTypeDef HASH_WriteData(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size);
static HAL_StatusTypeDef HASH_IT(HASH_HandleTypeDef *hhash);
static uint32_t HASH_Write_Block_Data(HASH_HandleTypeDef *hhash);
static HAL_StatusTypeDef HMAC_Processing(HASH_HandleTypeDef *hhash, uint32_t Timeout);
/**
  * @}
  */

/** @defgroup HASH_Exported_Functions HASH Exported Functions
  * @{
  */

/** @defgroup HASH_Exported_Functions_Group1 Initialization and de-initialization functions
  *  @brief    Initialization, configuration and call-back functions.
  *
@verbatim
 ===============================================================================
              ##### Initialization and de-initialization functions #####
 ===============================================================================
    [..]  This section provides functions allowing to:
      (+) Initialize the HASH according to the specified parameters
          in the HASH_InitTypeDef and create the associated handle
      (+) DeInitialize the HASH peripheral
      (+) Initialize the HASH MCU Specific Package (MSP)
      (+) DeInitialize the HASH MSP

    [..]  This section provides as well call back functions definitions for user
          code to manage:
      (+) Input data transfer to Peripheral completion
      (+) Calculated digest retrieval completion
      (+) Error management



@endverbatim
  * @{
  */

/**
  * @brief  Initialize the HASH according to the specified parameters in the
            HASH_HandleTypeDef and create the associated handle.
  * @note   Only MDMAT and DATATYPE bits of HASH Peripheral are set by HAL_HASH_Init(),
  *         other configuration bits are set by HASH or HMAC processing APIs.
  * @note   MDMAT bit is systematically reset by HAL_HASH_Init(). To set it for
  *         multi-buffer HASH processing, user needs to resort to
  *         __HAL_HASH_SET_MDMAT() macro. For HMAC multi-buffer processing, the
  *         relevant APIs manage themselves the MDMAT bit.
  * @param  hhash HASH handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_Init(HASH_HandleTypeDef *hhash)
{
  /* Check the hash handle allocation */
  if (hhash == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_HASH_DATATYPE(hhash->Init.DataType));

#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
  if (hhash->State == HAL_HASH_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hhash->Lock = HAL_UNLOCKED;

    /* Reset Callback pointers in HAL_HASH_STATE_RESET only */
    hhash->InCpltCallback =  HAL_HASH_InCpltCallback;     /* Legacy weak (surcharged) input completion callback */
    hhash->DgstCpltCallback =  HAL_HASH_DgstCpltCallback; /* Legacy weak (surcharged) digest computation
                                                             completion callback */
    hhash->ErrorCallback =  HAL_HASH_ErrorCallback;       /* Legacy weak (surcharged) error callback */
    if (hhash->MspInitCallback == NULL)
    {
      hhash->MspInitCallback = HAL_HASH_MspInit;
    }

    /* Init the low level hardware */
    hhash->MspInitCallback(hhash);
  }
#else
  if (hhash->State == HAL_HASH_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hhash->Lock = HAL_UNLOCKED;

    /* Init the low level hardware */
    HAL_HASH_MspInit(hhash);
  }
#endif /* (USE_HAL_HASH_REGISTER_CALLBACKS) */

  /* Change the HASH state */
  hhash->State = HAL_HASH_STATE_BUSY;

  /* Reset HashInCount, HashITCounter, HashBuffSize and NbWordsAlreadyPushed */
  hhash->HashInCount = 0;
  hhash->HashBuffSize = 0;
  hhash->HashITCounter = 0;
  hhash->NbWordsAlreadyPushed = 0;
  /* Reset digest calculation bridle (MDMAT bit control) */
  hhash->DigestCalculationDisable = RESET;
  /* Set phase to READY */
  hhash->Phase = HAL_HASH_PHASE_READY;
  /* Reset suspension request flag */
  hhash->SuspendRequest = HAL_HASH_SUSPEND_NONE;

  /* Set the data type bit */
  MODIFY_REG(HASH->CR, HASH_CR_DATATYPE, hhash->Init.DataType);
#if defined(HASH_CR_MDMAT)
  /* Reset MDMAT bit */
  __HAL_HASH_RESET_MDMAT();
#endif /* HASH_CR_MDMAT */
  /* Reset HASH handle status */
  hhash->Status = HAL_OK;

  /* Set the HASH state to Ready */
  hhash->State = HAL_HASH_STATE_READY;

  /* Initialise the error code */
  hhash->ErrorCode = HAL_HASH_ERROR_NONE;

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  DeInitialize the HASH peripheral.
  * @param  hhash HASH handle.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_DeInit(HASH_HandleTypeDef *hhash)
{
  /* Check the HASH handle allocation */
  if (hhash == NULL)
  {
    return HAL_ERROR;
  }

  /* Change the HASH state */
  hhash->State = HAL_HASH_STATE_BUSY;

  /* Set the default HASH phase */
  hhash->Phase = HAL_HASH_PHASE_READY;

  /* Reset HashInCount, HashITCounter and HashBuffSize */
  hhash->HashInCount = 0;
  hhash->HashBuffSize = 0;
  hhash->HashITCounter = 0;
  /* Reset digest calculation bridle (MDMAT bit control) */
  hhash->DigestCalculationDisable = RESET;

#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
  if (hhash->MspDeInitCallback == NULL)
  {
    hhash->MspDeInitCallback = HAL_HASH_MspDeInit;
  }

  /* DeInit the low level hardware */
  hhash->MspDeInitCallback(hhash);
#else
  /* DeInit the low level hardware: CLOCK, NVIC */
  HAL_HASH_MspDeInit(hhash);
#endif /* (USE_HAL_HASH_REGISTER_CALLBACKS) */


  /* Reset HASH handle status */
  hhash->Status = HAL_OK;

  /* Set the HASH state to Ready */
  hhash->State = HAL_HASH_STATE_RESET;

  /* Initialise the error code */
  hhash->ErrorCode = HAL_HASH_ERROR_NONE;

  /* Reset multi buffers accumulation flag */
  hhash->Accumulation = 0U;

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Initialize the HASH MSP.
  * @param  hhash HASH handle.
  * @retval None
  */
__weak void HAL_HASH_MspInit(HASH_HandleTypeDef *hhash)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hhash);

  /* NOTE : This function should not be modified; when the callback is needed,
            HAL_HASH_MspInit() can be implemented in the user file.
   */
}

/**
  * @brief  DeInitialize the HASH MSP.
  * @param  hhash HASH handle.
  * @retval None
  */
__weak void HAL_HASH_MspDeInit(HASH_HandleTypeDef *hhash)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hhash);

  /* NOTE : This function should not be modified; when the callback is needed,
            HAL_HASH_MspDeInit() can be implemented in the user file.
   */
}

/**
  * @brief  Input data transfer complete call back.
  * @note   HAL_HASH_InCpltCallback() is called when the complete input message
  *         has been fed to the Peripheral. This API is invoked only when input data are
  *         entered under interruption or through DMA.
  * @note   In case of HASH or HMAC multi-buffer DMA feeding case (MDMAT bit set),
  *         HAL_HASH_InCpltCallback() is called at the end of each buffer feeding
  *         to the Peripheral.
  * @param  hhash HASH handle.
  * @retval None
  */
__weak void HAL_HASH_InCpltCallback(HASH_HandleTypeDef *hhash)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hhash);

  /* NOTE : This function should not be modified; when the callback is needed,
            HAL_HASH_InCpltCallback() can be implemented in the user file.
   */
}

/**
  * @brief  Digest computation complete call back.
  * @note   HAL_HASH_DgstCpltCallback() is used under interruption, is not
  *         relevant with DMA.
  * @param  hhash HASH handle.
  * @retval None
  */
__weak void HAL_HASH_DgstCpltCallback(HASH_HandleTypeDef *hhash)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hhash);

  /* NOTE : This function should not be modified; when the callback is needed,
            HAL_HASH_DgstCpltCallback() can be implemented in the user file.
   */
}

/**
  * @brief  Error callback.
  * @note   Code user can resort to hhash->Status (HAL_ERROR, HAL_TIMEOUT,...)
  *         to retrieve the error type.
  * @param  hhash HASH handle.
  * @retval None
  */
__weak void HAL_HASH_ErrorCallback(HASH_HandleTypeDef *hhash)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hhash);

  /* NOTE : This function should not be modified; when the callback is needed,
            HAL_HASH_ErrorCallback() can be implemented in the user file.
   */
}

#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User HASH Callback
  *         To be used instead of the weak (surcharged) predefined callback
  * @param hhash HASH handle
  * @param CallbackID ID of the callback to be registered
  *        This parameter can be one of the following values:
  *          @arg @ref HAL_HASH_INPUTCPLT_CB_ID HASH input completion Callback ID
  *          @arg @ref HAL_HASH_DGSTCPLT_CB_ID HASH digest computation completion Callback ID
  *          @arg @ref HAL_HASH_ERROR_CB_ID HASH error Callback ID
  *          @arg @ref HAL_HASH_MSPINIT_CB_ID HASH MspInit callback ID
  *          @arg @ref HAL_HASH_MSPDEINIT_CB_ID HASH MspDeInit callback ID
  * @param pCallback pointer to the Callback function
  * @retval status
  */
HAL_StatusTypeDef HAL_HASH_RegisterCallback(HASH_HandleTypeDef *hhash, HAL_HASH_CallbackIDTypeDef CallbackID,
                                            pHASH_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hhash->ErrorCode |= HAL_HASH_ERROR_INVALID_CALLBACK;
    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hhash);

  if (HAL_HASH_STATE_READY == hhash->State)
  {
    switch (CallbackID)
    {
      case HAL_HASH_INPUTCPLT_CB_ID :
        hhash->InCpltCallback = pCallback;
        break;

      case HAL_HASH_DGSTCPLT_CB_ID :
        hhash->DgstCpltCallback = pCallback;
        break;

      case HAL_HASH_ERROR_CB_ID :
        hhash->ErrorCallback = pCallback;
        break;

      case HAL_HASH_MSPINIT_CB_ID :
        hhash->MspInitCallback = pCallback;
        break;

      case HAL_HASH_MSPDEINIT_CB_ID :
        hhash->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hhash->ErrorCode |= HAL_HASH_ERROR_INVALID_CALLBACK;
        /* update return status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_HASH_STATE_RESET == hhash->State)
  {
    switch (CallbackID)
    {
      case HAL_HASH_MSPINIT_CB_ID :
        hhash->MspInitCallback = pCallback;
        break;

      case HAL_HASH_MSPDEINIT_CB_ID :
        hhash->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hhash->ErrorCode |= HAL_HASH_ERROR_INVALID_CALLBACK;
        /* update return status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hhash->ErrorCode |= HAL_HASH_ERROR_INVALID_CALLBACK;
    /* update return status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hhash);
  return status;
}

/**
  * @brief  Unregister a HASH Callback
  *         HASH Callback is redirected to the weak (surcharged) predefined callback
  * @param hhash HASH handle
  * @param CallbackID ID of the callback to be unregistered
  *        This parameter can be one of the following values:
  *          @arg @ref HAL_HASH_INPUTCPLT_CB_ID HASH input completion Callback ID
  *          @arg @ref HAL_HASH_DGSTCPLT_CB_ID HASH digest computation completion Callback ID
  *          @arg @ref HAL_HASH_ERROR_CB_ID HASH error Callback ID
  *          @arg @ref HAL_HASH_MSPINIT_CB_ID HASH MspInit callback ID
  *          @arg @ref HAL_HASH_MSPDEINIT_CB_ID HASH MspDeInit callback ID
  * @retval status
  */
HAL_StatusTypeDef HAL_HASH_UnRegisterCallback(HASH_HandleTypeDef *hhash, HAL_HASH_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hhash);

  if (HAL_HASH_STATE_READY == hhash->State)
  {
    switch (CallbackID)
    {
      case HAL_HASH_INPUTCPLT_CB_ID :
        hhash->InCpltCallback = HAL_HASH_InCpltCallback;     /* Legacy weak (surcharged) input completion callback */
        break;

      case HAL_HASH_DGSTCPLT_CB_ID :
        hhash->DgstCpltCallback = HAL_HASH_DgstCpltCallback; /* Legacy weak (surcharged) digest computation
                                                                completion callback */
        break;

      case HAL_HASH_ERROR_CB_ID :
        hhash->ErrorCallback = HAL_HASH_ErrorCallback;       /* Legacy weak (surcharged) error callback */
        break;

      case HAL_HASH_MSPINIT_CB_ID :
        hhash->MspInitCallback = HAL_HASH_MspInit;           /* Legacy weak (surcharged) Msp Init */
        break;

      case HAL_HASH_MSPDEINIT_CB_ID :
        hhash->MspDeInitCallback = HAL_HASH_MspDeInit;       /* Legacy weak (surcharged) Msp DeInit */
        break;

      default :
        /* Update the error code */
        hhash->ErrorCode |= HAL_HASH_ERROR_INVALID_CALLBACK;
        /* update return status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_HASH_STATE_RESET == hhash->State)
  {
    switch (CallbackID)
    {
      case HAL_HASH_MSPINIT_CB_ID :
        hhash->MspInitCallback = HAL_HASH_MspInit;           /* Legacy weak (surcharged) Msp Init */
        break;

      case HAL_HASH_MSPDEINIT_CB_ID :
        hhash->MspDeInitCallback = HAL_HASH_MspDeInit;       /* Legacy weak (surcharged) Msp DeInit */
        break;

      default :
        /* Update the error code */
        hhash->ErrorCode |= HAL_HASH_ERROR_INVALID_CALLBACK;
        /* update return status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hhash->ErrorCode |= HAL_HASH_ERROR_INVALID_CALLBACK;
    /* update return status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hhash);
  return status;
}
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup HASH_Exported_Functions_Group2 HASH processing functions in polling mode
  *  @brief   HASH processing functions using polling mode.
  *
@verbatim
 ===============================================================================
                 ##### Polling mode HASH processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in polling mode
          the hash value using one of the following algorithms:
      (+) MD5
         (++) HAL_HASH_MD5_Start()
         (++) HAL_HASH_MD5_Accmlt()
         (++) HAL_HASH_MD5_Accmlt_End()
      (+) SHA1
         (++) HAL_HASH_SHA1_Start()
         (++) HAL_HASH_SHA1_Accmlt()
         (++) HAL_HASH_SHA1_Accmlt_End()

    [..] For a single buffer to be hashed, user can resort to HAL_HASH_xxx_Start().

    [..]  In case of multi-buffer HASH processing (a single digest is computed while
          several buffers are fed to the Peripheral), the user can resort to successive calls
          to HAL_HASH_xxx_Accumulate() and wrap-up the digest computation by a call
          to HAL_HASH_xxx_Accumulate_End().

@endverbatim
  * @{
  */

/**
  * @brief  Initialize the HASH peripheral in MD5 mode, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 16 bytes.
  * @param  Timeout Timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_MD5_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                     uint32_t Timeout)
{
  return HASH_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  If not already done, initialize the HASH peripheral in MD5 mode then
  *         processes pInBuffer.
  * @note   Consecutive calls to HAL_HASH_MD5_Accmlt() can be used to feed
  *         several input buffers back-to-back to the Peripheral that will yield a single
  *         HASH signature once all buffers have been entered. Wrap-up of input
  *         buffers feeding and retrieval of digest is done by a call to
  *         HAL_HASH_MD5_Accmlt_End().
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   Digest is not retrieved by this API, user must resort to HAL_HASH_MD5_Accmlt_End()
  *         to read it, feeding at the same time the last input buffer to the Peripheral.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted. Only HAL_HASH_MD5_Accmlt_End() is able
  *         to manage the ending buffer with a length in bytes not a multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_MD5_Accmlt(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HASH_Accumulate(hhash, pInBuffer, Size, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  End computation of a single HASH signature after several calls to HAL_HASH_MD5_Accmlt() API.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 16 bytes.
  * @param  Timeout Timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_MD5_Accmlt_End(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                          uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  Initialize the HASH peripheral in SHA1 mode, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 20 bytes.
  * @param  Timeout Timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_SHA1_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                      uint32_t Timeout)
{
  return HASH_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_SHA1);
}

/**
  * @brief  If not already done, initialize the HASH peripheral in SHA1 mode then
  *         processes pInBuffer.
  * @note   Consecutive calls to HAL_HASH_SHA1_Accmlt() can be used to feed
  *         several input buffers back-to-back to the Peripheral that will yield a single
  *         HASH signature once all buffers have been entered. Wrap-up of input
  *         buffers feeding and retrieval of digest is done by a call to
  *         HAL_HASH_SHA1_Accmlt_End().
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   Digest is not retrieved by this API, user must resort to HAL_HASH_SHA1_Accmlt_End()
  *         to read it, feeding at the same time the last input buffer to the Peripheral.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted. Only HAL_HASH_SHA1_Accmlt_End() is able
  *         to manage the ending buffer with a length in bytes not a multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_SHA1_Accmlt(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HASH_Accumulate(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA1);
}

/**
  * @brief  End computation of a single HASH signature after several calls to HAL_HASH_SHA1_Accmlt() API.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 20 bytes.
  * @param  Timeout Timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_SHA1_Accmlt_End(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                           uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_SHA1);
}

/**
  * @}
  */

/** @defgroup HASH_Exported_Functions_Group3 HASH processing functions in interrupt mode
  *  @brief   HASH processing functions using interrupt mode.
  *
@verbatim
 ===============================================================================
                 ##### Interruption mode HASH processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in interrupt mode
          the hash value using one of the following algorithms:
      (+) MD5
         (++) HAL_HASH_MD5_Start_IT()
         (++) HAL_HASH_MD5_Accmlt_IT()
         (++) HAL_HASH_MD5_Accmlt_End_IT()
      (+) SHA1
         (++) HAL_HASH_SHA1_Start_IT()
         (++) HAL_HASH_SHA1_Accmlt_IT()
         (++) HAL_HASH_SHA1_Accmlt_End_IT()

    [..]  API HAL_HASH_IRQHandler() manages each HASH interruption.

    [..] Note that HAL_HASH_IRQHandler() manages as well HASH Peripheral interruptions when in
         HMAC processing mode.


@endverbatim
  * @{
  */

/**
  * @brief  Initialize the HASH peripheral in MD5 mode, next process pInBuffer then
  *         read the computed digest in interruption mode.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 16 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_MD5_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                        uint8_t *pOutBuffer)
{
  return HASH_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  If not already done, initialize the HASH peripheral in MD5 mode then
  *         processes pInBuffer in interruption mode.
  * @note   Consecutive calls to HAL_HASH_MD5_Accmlt_IT() can be used to feed
  *         several input buffers back-to-back to the Peripheral that will yield a single
  *         HASH signature once all buffers have been entered. Wrap-up of input
  *         buffers feeding and retrieval of digest is done by a call to
  *         HAL_HASH_MD5_Accmlt_End_IT().
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted. Only HAL_HASH_MD5_Accmlt_End_IT() is able
  *         to manage the ending buffer with a length in bytes not a multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_MD5_Accmlt_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HASH_Accumulate_IT(hhash, pInBuffer, Size, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  End computation of a single HASH signature after several calls to HAL_HASH_MD5_Accmlt_IT() API.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 16 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_MD5_Accmlt_End_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                             uint8_t *pOutBuffer)
{
  return HASH_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  Initialize the HASH peripheral in SHA1 mode, next process pInBuffer then
  *         read the computed digest in interruption mode.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 20 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_SHA1_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                         uint8_t *pOutBuffer)
{
  return HASH_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_SHA1);
}


/**
  * @brief  If not already done, initialize the HASH peripheral in SHA1 mode then
  *         processes pInBuffer in interruption mode.
  * @note   Consecutive calls to HAL_HASH_SHA1_Accmlt_IT() can be used to feed
  *         several input buffers back-to-back to the Peripheral that will yield a single
  *         HASH signature once all buffers have been entered. Wrap-up of input
  *         buffers feeding and retrieval of digest is done by a call to
  *         HAL_HASH_SHA1_Accmlt_End_IT().
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted. Only HAL_HASH_SHA1_Accmlt_End_IT() is able
  *         to manage the ending buffer with a length in bytes not a multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_SHA1_Accmlt_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HASH_Accumulate_IT(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA1);
}

/**
  * @brief  End computation of a single HASH signature after several calls to HAL_HASH_SHA1_Accmlt_IT() API.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 20 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_SHA1_Accmlt_End_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                              uint8_t *pOutBuffer)
{
  return HASH_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_SHA1);
}

/**
  * @brief Handle HASH interrupt request.
  * @param hhash HASH handle.
  * @note  HAL_HASH_IRQHandler() handles interrupts in HMAC processing as well.
  * @note  In case of error reported during the HASH interruption processing,
  *        HAL_HASH_ErrorCallback() API is called so that user code can
  *        manage the error. The error type is available in hhash->Status field.
  * @retval None
  */
void HAL_HASH_IRQHandler(HASH_HandleTypeDef *hhash)
{
  hhash->Status = HASH_IT(hhash);
  if (hhash->Status != HAL_OK)
  {
    hhash->ErrorCode |= HAL_HASH_ERROR_IT;
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
    hhash->ErrorCallback(hhash);
#else
    HAL_HASH_ErrorCallback(hhash);
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */
    /* After error handling by code user, reset HASH handle HAL status */
    hhash->Status = HAL_OK;
  }
}

/**
  * @}
  */

/** @defgroup HASH_Exported_Functions_Group4 HASH processing functions in DMA mode
  *  @brief   HASH processing functions using DMA mode.
  *
@verbatim
 ===============================================================================
                    ##### DMA mode HASH processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in DMA mode
          the hash value using one of the following algorithms:
      (+) MD5
         (++) HAL_HASH_MD5_Start_DMA()
         (++) HAL_HASH_MD5_Finish()
      (+) SHA1
         (++) HAL_HASH_SHA1_Start_DMA()
         (++) HAL_HASH_SHA1_Finish()

    [..]  When resorting to DMA mode to enter the data in the Peripheral, user must resort
          to  HAL_HASH_xxx_Start_DMA() then read the resulting digest with
          HAL_HASH_xxx_Finish().
    [..]  In case of multi-buffer HASH processing, MDMAT bit must first be set before
          the successive calls to HAL_HASH_xxx_Start_DMA(). Then, MDMAT bit needs to be
          reset before the last call to HAL_HASH_xxx_Start_DMA(). Digest is finally
          retrieved thanks to HAL_HASH_xxx_Finish().

@endverbatim
  * @{
  */

/**
  * @brief  Initialize the HASH peripheral in MD5 mode then initiate a DMA transfer
  *         to feed the input buffer to the Peripheral.
  * @note   Once the DMA transfer is finished, HAL_HASH_MD5_Finish() API must
  *         be called to retrieve the computed digest.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_MD5_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return HASH_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  Return the computed digest in MD5 mode.
  * @note   The API waits for DCIS to be set then reads the computed digest.
  * @note   HAL_HASH_MD5_Finish() can be used as well to retrieve the digest in
  *         HMAC MD5 mode.
  * @param  hhash HASH handle.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 16 bytes.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_MD5_Finish(HASH_HandleTypeDef *hhash, uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Finish(hhash, pOutBuffer, Timeout);
}

/**
  * @brief  Initialize the HASH peripheral in SHA1 mode then initiate a DMA transfer
  *         to feed the input buffer to the Peripheral.
  * @note   Once the DMA transfer is finished, HAL_HASH_SHA1_Finish() API must
  *         be called to retrieve the computed digest.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_SHA1_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return HASH_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA1);
}


/**
  * @brief  Return the computed digest in SHA1 mode.
  * @note   The API waits for DCIS to be set then reads the computed digest.
  * @note   HAL_HASH_SHA1_Finish() can be used as well to retrieve the digest in
  *         HMAC SHA1 mode.
  * @param  hhash HASH handle.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 20 bytes.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_SHA1_Finish(HASH_HandleTypeDef *hhash, uint8_t *pOutBuffer, uint32_t Timeout)
{
  return HASH_Finish(hhash, pOutBuffer, Timeout);
}

/**
  * @}
  */

/** @defgroup HASH_Exported_Functions_Group5 HMAC processing functions in polling mode
  *  @brief   HMAC processing functions using polling mode.
  *
@verbatim
 ===============================================================================
                 ##### Polling mode HMAC processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in polling mode
          the HMAC value using one of the following algorithms:
      (+) MD5
         (++) HAL_HMAC_MD5_Start()
      (+) SHA1
         (++) HAL_HMAC_SHA1_Start()


@endverbatim
  * @{
  */

/**
  * @brief  Initialize the HASH peripheral in HMAC MD5 mode, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 16 bytes.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMAC_MD5_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                     uint32_t Timeout)
{
  return HMAC_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  Initialize the HASH peripheral in HMAC SHA1 mode, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 20 bytes.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMAC_SHA1_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                      uint32_t Timeout)
{
  return HMAC_Start(hhash, pInBuffer, Size, pOutBuffer, Timeout, HASH_ALGOSELECTION_SHA1);
}

/**
  * @}
  */


/** @defgroup HASH_Exported_Functions_Group6 HMAC processing functions in interrupt mode
  *  @brief   HMAC processing functions using interrupt mode.
  *
@verbatim
 ===============================================================================
                 ##### Interrupt mode HMAC processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in interrupt mode
          the HMAC value using one of the following algorithms:
      (+) MD5
         (++) HAL_HMAC_MD5_Start_IT()
      (+) SHA1
         (++) HAL_HMAC_SHA1_Start_IT()

@endverbatim
  * @{
  */


/**
  * @brief  Initialize the HASH peripheral in HMAC MD5 mode, next process pInBuffer then
  *         read the computed digest in interrupt mode.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 16 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMAC_MD5_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                        uint8_t *pOutBuffer)
{
  return  HMAC_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_MD5);
}

/**
  * @brief  Initialize the HASH peripheral in HMAC SHA1 mode, next process pInBuffer then
  *         read the computed digest in interrupt mode.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest. Digest size is 20 bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMAC_SHA1_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                         uint8_t *pOutBuffer)
{
  return  HMAC_Start_IT(hhash, pInBuffer, Size, pOutBuffer, HASH_ALGOSELECTION_SHA1);
}

/**
  * @}
  */



/** @defgroup HASH_Exported_Functions_Group7 HMAC processing functions in DMA mode
  *  @brief   HMAC processing functions using DMA modes.
  *
@verbatim
 ===============================================================================
                 ##### DMA mode HMAC processing functions #####
 ===============================================================================
    [..]  This section provides functions allowing to calculate in DMA mode
          the HMAC value using one of the following algorithms:
      (+) MD5
         (++) HAL_HMAC_MD5_Start_DMA()
      (+) SHA1
         (++) HAL_HMAC_SHA1_Start_DMA()

    [..]  When resorting to DMA mode to enter the data in the Peripheral for HMAC processing,
          user must resort to  HAL_HMAC_xxx_Start_DMA() then read the resulting digest
          with HAL_HASH_xxx_Finish().

@endverbatim
  * @{
  */


/**
  * @brief  Initialize the HASH peripheral in HMAC MD5 mode then initiate the required
  *         DMA transfers to feed the key and the input buffer to the Peripheral.
  * @note   Once the DMA transfers are finished (indicated by hhash->State set back
  *         to HAL_HASH_STATE_READY), HAL_HASH_MD5_Finish() API must be called to retrieve
  *         the computed digest.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   If MDMAT bit is set before calling this function (multi-buffer
  *          HASH processing case), the input buffer size (in bytes) must be
  *          a multiple of 4 otherwise, the HASH digest computation is corrupted.
  *          For the processing of the last buffer of the thread, MDMAT bit must
  *          be reset and the buffer length (in bytes) doesn't have to be a
  *          multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMAC_MD5_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_MD5);
}


/**
  * @brief  Initialize the HASH peripheral in HMAC SHA1 mode then initiate the required
  *         DMA transfers to feed the key and the input buffer to the Peripheral.
  * @note   Once the DMA transfers are finished (indicated by hhash->State set back
  *         to HAL_HASH_STATE_READY), HAL_HASH_SHA1_Finish() API must be called to retrieve
  *         the computed digest.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   If MDMAT bit is set before calling this function (multi-buffer
  *          HASH processing case), the input buffer size (in bytes) must be
  *          a multiple of 4 otherwise, the HASH digest computation is corrupted.
  *          For the processing of the last buffer of the thread, MDMAT bit must
  *          be reset and the buffer length (in bytes) doesn't have to be a
  *          multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HMAC_SHA1_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  return  HMAC_Start_DMA(hhash, pInBuffer, Size, HASH_ALGOSELECTION_SHA1);
}

/**
  * @}
  */

/** @defgroup HASH_Exported_Functions_Group8 Peripheral states functions
  *  @brief   Peripheral State functions.
  *
@verbatim
 ===============================================================================
                      ##### Peripheral State methods #####
 ===============================================================================
    [..]
    This section permits to get in run-time the state and the peripheral handle
    status of the peripheral:
      (+) HAL_HASH_GetState()
      (+) HAL_HASH_GetStatus()

    [..]
    Additionally, this subsection provides functions allowing to save and restore
    the HASH or HMAC processing context in case of calculation suspension:
      (+) HAL_HASH_ContextSaving()
      (+) HAL_HASH_ContextRestoring()

    [..]
    This subsection provides functions allowing to suspend the HASH processing
      (+) when input are fed to the Peripheral by software
          (++) HAL_HASH_SwFeed_ProcessSuspend()
      (+) when input are fed to the Peripheral by DMA
          (++) HAL_HASH_DMAFeed_ProcessSuspend()



@endverbatim
  * @{
  */

/**
  * @brief  Return the HASH handle state.
  * @note   The API yields the current state of the handle (BUSY, READY,...).
  * @param  hhash HASH handle.
  * @retval HAL HASH state
  */
HAL_HASH_StateTypeDef HAL_HASH_GetState(HASH_HandleTypeDef *hhash)
{
  return hhash->State;
}


/**
  * @brief Return the HASH HAL status.
  * @note  The API yields the HAL status of the handle: it is the result of the
  *        latest HASH processing and allows to report any issue (e.g. HAL_TIMEOUT).
  * @param  hhash HASH handle.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_GetStatus(HASH_HandleTypeDef *hhash)
{
  return hhash->Status;
}

/**
  * @brief  Save the HASH context in case of processing suspension.
  * @param  hhash HASH handle.
  * @param  pMemBuffer pointer to the memory buffer where the HASH context
  *         is saved.
  * @note   The IMR, STR, CR then all the CSR registers are saved
  *         in that order. Only the r/w bits are read to be restored later on.
  * @note   By default, all the context swap registers (there are
  *         HASH_NUMBER_OF_CSR_REGISTERS of those) are saved.
  * @note   pMemBuffer points to a buffer allocated by the user. The buffer size
  *         must be at least (HASH_NUMBER_OF_CSR_REGISTERS + 3) * 4 uint8 long.
  * @retval None
  */
void HAL_HASH_ContextSaving(HASH_HandleTypeDef *hhash, uint8_t *pMemBuffer)
{
  uint32_t mem_ptr = (uint32_t)pMemBuffer;
  uint32_t csr_ptr = (uint32_t)HASH->CSR;
  uint32_t i;

  /* Prevent unused argument(s) compilation warning */
  UNUSED(hhash);

  /* Save IMR register content */
  *(uint32_t *)(mem_ptr) = READ_BIT(HASH->IMR, HASH_IT_DINI | HASH_IT_DCI);
  mem_ptr += 4U;
  /* Save STR register content */
  *(uint32_t *)(mem_ptr) = READ_BIT(HASH->STR, HASH_STR_NBLW);
  mem_ptr += 4U;
  /* Save CR register content */
#if defined(HASH_CR_MDMAT)
  *(uint32_t *)(mem_ptr) = READ_BIT(HASH->CR, HASH_CR_DMAE | HASH_CR_DATATYPE | HASH_CR_MODE | HASH_CR_ALGO |
                                    HASH_CR_LKEY | HASH_CR_MDMAT);
#else
  *(uint32_t *)(mem_ptr) = READ_BIT(HASH->CR, HASH_CR_DMAE | HASH_CR_DATATYPE | HASH_CR_MODE | HASH_CR_ALGO |
                                    HASH_CR_LKEY);
#endif /* HASH_CR_MDMAT*/
  mem_ptr += 4U;
  /* By default, save all CSRs registers */
  for (i = HASH_NUMBER_OF_CSR_REGISTERS; i > 0U; i--)
  {
    *(uint32_t *)(mem_ptr) = *(uint32_t *)(csr_ptr);
    mem_ptr += 4U;
    csr_ptr += 4U;
  }
}


/**
  * @brief  Restore the HASH context in case of processing resumption.
  * @param  hhash HASH handle.
  * @param  pMemBuffer pointer to the memory buffer where the HASH context
  *         is stored.
  * @note   The IMR, STR, CR then all the CSR registers are restored
  *         in that order. Only the r/w bits are restored.
  * @note   By default, all the context swap registers (HASH_NUMBER_OF_CSR_REGISTERS
  *         of those) are restored (all of them have been saved by default
  *         beforehand).
  * @retval None
  */
void HAL_HASH_ContextRestoring(HASH_HandleTypeDef *hhash, uint8_t *pMemBuffer)
{
  uint32_t mem_ptr = (uint32_t)pMemBuffer;
  uint32_t csr_ptr = (uint32_t)HASH->CSR;
  uint32_t i;

  /* Prevent unused argument(s) compilation warning */
  UNUSED(hhash);

  /* Restore IMR register content */
  WRITE_REG(HASH->IMR, (*(uint32_t *)(mem_ptr)));
  mem_ptr += 4U;
  /* Restore STR register content */
  WRITE_REG(HASH->STR, (*(uint32_t *)(mem_ptr)));
  mem_ptr += 4U;
  /* Restore CR register content */
  WRITE_REG(HASH->CR, (*(uint32_t *)(mem_ptr)));
  mem_ptr += 4U;

  /* Reset the HASH processor before restoring the Context
  Swap Registers (CSR) */
  __HAL_HASH_INIT();

  /* By default, restore all CSR registers */
  for (i = HASH_NUMBER_OF_CSR_REGISTERS; i > 0U; i--)
  {
    WRITE_REG((*(uint32_t *)(csr_ptr)), (*(uint32_t *)(mem_ptr)));
    mem_ptr += 4U;
    csr_ptr += 4U;
  }
}


/**
  * @brief  Initiate HASH processing suspension when in polling or interruption mode.
  * @param  hhash HASH handle.
  * @note   Set the handle field SuspendRequest to the appropriate value so that
  *         the on-going HASH processing is suspended as soon as the required
  *         conditions are met. Note that the actual suspension is carried out
  *         by the functions HASH_WriteData() in polling mode and HASH_IT() in
  *         interruption mode.
  * @retval None
  */
void HAL_HASH_SwFeed_ProcessSuspend(HASH_HandleTypeDef *hhash)
{
  /* Set Handle Suspend Request field */
  hhash->SuspendRequest = HAL_HASH_SUSPEND;
}

/**
  * @brief  Suspend the HASH processing when in DMA mode.
  * @param  hhash HASH handle.
  * @note   When suspension attempt occurs at the very end of a DMA transfer and
  *         all the data have already been entered in the Peripheral, hhash->State is
  *         set to HAL_HASH_STATE_READY and the API returns HAL_ERROR. It is
  *         recommended to wrap-up the processing in reading the digest as usual.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_HASH_DMAFeed_ProcessSuspend(HASH_HandleTypeDef *hhash)
{
  uint32_t tmp_remaining_DMATransferSize_inWords;
  uint32_t tmp_initial_DMATransferSize_inWords;
  uint32_t tmp_words_already_pushed;

  if (hhash->State == HAL_HASH_STATE_READY)
  {
    return HAL_ERROR;
  }
  else
  {

    /* Make sure there is enough time to suspend the processing */
    tmp_remaining_DMATransferSize_inWords = ((DMA_Stream_TypeDef *)hhash->hdmain->Instance)->NDTR;

    if (tmp_remaining_DMATransferSize_inWords <= HASH_DMA_SUSPENSION_WORDS_LIMIT)
    {
      /* No suspension attempted since almost to the end of the transferred data. */
      /* Best option for user code is to wrap up low priority message hashing     */
      return HAL_ERROR;
    }

    /* Wait for BUSY flag to be reset */
    if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_BUSY, SET, HASH_TIMEOUTVALUE) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }

    if (__HAL_HASH_GET_FLAG(HASH_FLAG_DCIS) != RESET)
    {
      return HAL_ERROR;
    }

    /* Wait for BUSY flag to be set */
    if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_BUSY, RESET, HASH_TIMEOUTVALUE) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }
    /* Disable DMA channel */
    /* Note that the Abort function will
      - Clear the transfer error flags
      - Unlock
      - Set the State
    */
    if (HAL_DMA_Abort(hhash->hdmain) != HAL_OK)
    {
      return HAL_ERROR;
    }

    /* Clear DMAE bit */
    CLEAR_BIT(HASH->CR, HASH_CR_DMAE);

    /* Wait for BUSY flag to be reset */
    if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_BUSY, SET, HASH_TIMEOUTVALUE) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }

    if (__HAL_HASH_GET_FLAG(HASH_FLAG_DCIS) != RESET)
    {
      return HAL_ERROR;
    }

    /* At this point, DMA interface is disabled and no transfer is on-going */
    /* Retrieve from the DMA handle how many words remain to be written */
    tmp_remaining_DMATransferSize_inWords = ((DMA_Stream_TypeDef *)hhash->hdmain->Instance)->NDTR;

    if (tmp_remaining_DMATransferSize_inWords == 0U)
    {
      /* All the DMA transfer is actually done. Suspension occurred at the very end
         of the transfer. Either the digest computation is about to start (HASH case)
         or processing is about to move from one step to another (HMAC case).
         In both cases, the processing can't be suspended at this point. It is
         safer to
         - retrieve the low priority block digest before starting the high
           priority block processing (HASH case)
         - re-attempt a new suspension (HMAC case)
         */
      return HAL_ERROR;
    }
    else
    {

      /* Compute how many words were supposed to be transferred by DMA */
      tmp_initial_DMATransferSize_inWords = (((hhash->HashInCount % 4U) != 0U) ? \
                                             ((hhash->HashInCount + 3U) / 4U) : (hhash->HashInCount / 4U));

      /* If discrepancy between the number of words reported by DMA Peripheral and
        the numbers of words entered as reported by HASH Peripheral, correct it */
      /* tmp_words_already_pushed reflects the number of words that were already pushed before
         the start of DMA transfer (multi-buffer processing case) */
      tmp_words_already_pushed = hhash->NbWordsAlreadyPushed;
      if (((tmp_words_already_pushed + tmp_initial_DMATransferSize_inWords - \
            tmp_remaining_DMATransferSize_inWords) % 16U) != HASH_NBW_PUSHED())
      {
        tmp_remaining_DMATransferSize_inWords--; /* one less word to be transferred again */
      }

      /* Accordingly, update the input pointer that points at the next word to be
         transferred to the Peripheral by DMA */
      hhash->pHashInBuffPtr +=  4U * (tmp_initial_DMATransferSize_inWords - tmp_remaining_DMATransferSize_inWords) ;

      /* And store in HashInCount the remaining size to transfer (in bytes) */
      hhash->HashInCount = 4U * tmp_remaining_DMATransferSize_inWords;

    }

    /* Set State as suspended */
    hhash->State = HAL_HASH_STATE_SUSPENDED;

    return HAL_OK;

  }
}

/**
  * @brief  Return the HASH handle error code.
  * @param  hhash pointer to a HASH_HandleTypeDef structure.
  * @retval HASH Error Code
  */
uint32_t HAL_HASH_GetError(HASH_HandleTypeDef *hhash)
{
  /* Return HASH Error Code */
  return hhash->ErrorCode;
}
/**
  * @}
  */


/**
  * @}
  */

/** @defgroup HASH_Private_Functions HASH Private Functions
  * @{
  */

/**
  * @brief DMA HASH Input Data transfer completion callback.
  * @param hdma DMA handle.
  * @note  In case of HMAC processing, HASH_DMAXferCplt() initiates
  *        the next DMA transfer for the following HMAC step.
  * @retval None
  */
static void HASH_DMAXferCplt(DMA_HandleTypeDef *hdma)
{
  HASH_HandleTypeDef *hhash = (HASH_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;
  uint32_t inputaddr;
  uint32_t buffersize;
  HAL_StatusTypeDef status = HAL_OK;

  if (hhash->State != HAL_HASH_STATE_SUSPENDED)
  {

    /* Disable the DMA transfer */
    CLEAR_BIT(HASH->CR, HASH_CR_DMAE);

    if (READ_BIT(HASH->CR, HASH_CR_MODE) == 0U)
    {
      /* If no HMAC processing, input data transfer is now over */

      /* Change the HASH state to ready */
      hhash->State = HAL_HASH_STATE_READY;

      /* Call Input data transfer complete call back */
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
      hhash->InCpltCallback(hhash);
#else
      HAL_HASH_InCpltCallback(hhash);
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */

    }
    else
    {
      /* HMAC processing: depending on the current HMAC step and whether or
      not multi-buffer processing is on-going, the next step is initiated
      and MDMAT bit is set.  */


      if (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_3)
      {
        /* This is the end of HMAC processing */

        /* Change the HASH state to ready */
        hhash->State = HAL_HASH_STATE_READY;

        /* Call Input data transfer complete call back
        (note that the last DMA transfer was that of the key
        for the outer HASH operation). */
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
        hhash->InCpltCallback(hhash);
#else
        HAL_HASH_InCpltCallback(hhash);
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */

        return;
      }
      else if (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_1)
      {
        inputaddr = (uint32_t)hhash->pHashMsgBuffPtr;     /* DMA transfer start address */
        buffersize = hhash->HashBuffSize;                 /* DMA transfer size (in bytes) */
        hhash->Phase = HAL_HASH_PHASE_HMAC_STEP_2;        /* Move phase from Step 1 to Step 2 */

        /* In case of suspension request, save the new starting parameters */
        hhash->HashInCount = hhash->HashBuffSize;         /* Initial DMA transfer size (in bytes) */
        hhash->pHashInBuffPtr  = hhash->pHashMsgBuffPtr ; /* DMA transfer start address           */

        hhash->NbWordsAlreadyPushed = 0U;                  /* Reset number of words already pushed */
#if defined(HASH_CR_MDMAT)
        /* Check whether or not digest calculation must be disabled (in case of multi-buffer HMAC processing) */
        if (hhash->DigestCalculationDisable != RESET)
        {
          /* Digest calculation is disabled: Step 2 must start with MDMAT bit set,
          no digest calculation will be triggered at the end of the input buffer feeding to the Peripheral */
          __HAL_HASH_SET_MDMAT();
        }
#endif /* HASH_CR_MDMAT*/
      }
      else  /*case (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_2)*/
      {
        if (hhash->DigestCalculationDisable != RESET)
        {
          /* No automatic move to Step 3 as a new message buffer will be fed to the Peripheral
          (case of multi-buffer HMAC processing):
          DCAL must not be set.
          Phase remains in Step 2, MDMAT remains set at this point.
          Change the HASH state to ready and call Input data transfer complete call back. */
          hhash->State = HAL_HASH_STATE_READY;
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
          hhash->InCpltCallback(hhash);
#else
          HAL_HASH_InCpltCallback(hhash);
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */
          return ;
        }
        else
        {
          /* Digest calculation is not disabled (case of single buffer input or last buffer
          of multi-buffer HMAC processing) */
          inputaddr = (uint32_t)hhash->Init.pKey;       /* DMA transfer start address */
          buffersize = hhash->Init.KeySize;             /* DMA transfer size (in bytes) */
          hhash->Phase = HAL_HASH_PHASE_HMAC_STEP_3;    /* Move phase from Step 2 to Step 3 */
          /* In case of suspension request, save the new starting parameters */
          hhash->HashInCount = hhash->Init.KeySize;     /* Initial size for second DMA transfer (input data) */
          hhash->pHashInBuffPtr  = hhash->Init.pKey ;   /* address passed to DMA, now entering data message */

          hhash->NbWordsAlreadyPushed = 0U;              /* Reset number of words already pushed */
        }
      }

      /* Configure the Number of valid bits in last word of the message */
      __HAL_HASH_SET_NBVALIDBITS(buffersize);

      /* Set the HASH DMA transfer completion call back */
      hhash->hdmain->XferCpltCallback = HASH_DMAXferCplt;

      /* Enable the DMA In DMA stream */
      status = HAL_DMA_Start_IT(hhash->hdmain, inputaddr, (uint32_t)&HASH->DIN, \
                                (((buffersize % 4U) != 0U) ? ((buffersize + (4U - (buffersize % 4U))) / 4U) : \
                                 (buffersize / 4U)));



      /* Enable DMA requests */
      SET_BIT(HASH->CR, HASH_CR_DMAE);

      /* Return function status */
      if (status != HAL_OK)
      {
        /* Update HASH state machine to error */
        hhash->State = HAL_HASH_STATE_ERROR;
      }
      else
      {
        /* Change HASH state */
        hhash->State = HAL_HASH_STATE_BUSY;
      }
    }
  }

  return;
}

/**
  * @brief DMA HASH communication error callback.
  * @param hdma DMA handle.
  * @note  HASH_DMAError() callback invokes HAL_HASH_ErrorCallback() that
  *        can contain user code to manage the error.
  * @retval None
  */
static void HASH_DMAError(DMA_HandleTypeDef *hdma)
{
  HASH_HandleTypeDef *hhash = (HASH_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (hhash->State != HAL_HASH_STATE_SUSPENDED)
  {
    hhash->ErrorCode |= HAL_HASH_ERROR_DMA;
    /* Set HASH state to ready to prevent any blocking issue in user code
       present in HAL_HASH_ErrorCallback() */
    hhash->State = HAL_HASH_STATE_READY;
    /* Set HASH handle status to error */
    hhash->Status = HAL_ERROR;
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
    hhash->ErrorCallback(hhash);
#else
    HAL_HASH_ErrorCallback(hhash);
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */
    /* After error handling by code user, reset HASH handle HAL status */
    hhash->Status = HAL_OK;

  }
}

/**
  * @brief  Feed the input buffer to the HASH Peripheral.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to input buffer.
  * @param  Size the size of input buffer in bytes.
  * @note   HASH_WriteData() regularly reads hhash->SuspendRequest to check whether
  *         or not the HASH processing must be suspended. If this is the case, the
  *         processing is suspended when possible and the Peripheral feeding point reached at
  *         suspension time is stored in the handle for resumption later on.
  * @retval HAL status
  */
static HAL_StatusTypeDef HASH_WriteData(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size)
{
  uint32_t buffercounter;
  __IO uint32_t inputaddr = (uint32_t) pInBuffer;

  for (buffercounter = 0U; buffercounter < Size; buffercounter += 4U)
  {
    /* Write input data 4 bytes at a time */
    HASH->DIN = *(uint32_t *)inputaddr;
    inputaddr += 4U;

    /* If the suspension flag has been raised and if the processing is not about
    to end, suspend processing */
    if ((hhash->SuspendRequest == HAL_HASH_SUSPEND) && ((buffercounter + 4U) < Size))
    {
      /* Wait for DINIS = 1, which occurs when 16 32-bit locations are free
      in the input buffer */
      if (__HAL_HASH_GET_FLAG(HASH_FLAG_DINIS))
      {
        /* Reset SuspendRequest */
        hhash->SuspendRequest = HAL_HASH_SUSPEND_NONE;

        /* Depending whether the key or the input data were fed to the Peripheral, the feeding point
        reached at suspension time is not saved in the same handle fields */
        if ((hhash->Phase == HAL_HASH_PHASE_PROCESS) || (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_2))
        {
          /* Save current reading and writing locations of Input and Output buffers */
          hhash->pHashInBuffPtr = (uint8_t *)inputaddr;
          /* Save the number of bytes that remain to be processed at this point */
          hhash->HashInCount    =  Size - (buffercounter + 4U);
        }
        else if ((hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_1) || (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_3))
        {
          /* Save current reading and writing locations of Input and Output buffers */
          hhash->pHashKeyBuffPtr  = (uint8_t *)inputaddr;
          /* Save the number of bytes that remain to be processed at this point */
          hhash->HashKeyCount  =  Size - (buffercounter + 4U);
        }
        else
        {
          /* Unexpected phase: unlock process and report error */
          hhash->State = HAL_HASH_STATE_READY;
          __HAL_UNLOCK(hhash);
          return HAL_ERROR;
        }

        /* Set the HASH state to Suspended and exit to stop entering data */
        hhash->State = HAL_HASH_STATE_SUSPENDED;

        return HAL_OK;
      } /* if (__HAL_HASH_GET_FLAG(HASH_FLAG_DINIS))  */
    } /* if ((hhash->SuspendRequest == HAL_HASH_SUSPEND) && ((buffercounter+4) < Size)) */
  }   /* for(buffercounter = 0; buffercounter < Size; buffercounter+=4)                 */

  /* At this point, all the data have been entered to the Peripheral: exit */
  return  HAL_OK;
}

/**
  * @brief  Retrieve the message digest.
  * @param  pMsgDigest pointer to the computed digest.
  * @param  Size message digest size in bytes.
  * @retval None
  */
static void HASH_GetDigest(uint8_t *pMsgDigest, uint8_t Size)
{
  uint32_t msgdigest = (uint32_t)pMsgDigest;

  switch (Size)
  {
    /* Read the message digest */
    case 16:  /* MD5 */
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[0]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[1]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[2]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[3]);
      break;
    case 20:  /* SHA1 */
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[0]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[1]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[2]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[3]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[4]);
      break;
    case 28:  /* SHA224 */
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[0]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[1]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[2]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[3]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[4]);
#if defined(HASH_CR_MDMAT)
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH_DIGEST->HR[5]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH_DIGEST->HR[6]);
#endif /* HASH_CR_MDMAT*/
      break;
    case 32:   /* SHA256 */
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[0]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[1]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[2]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[3]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH->HR[4]);
#if defined(HASH_CR_MDMAT)
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH_DIGEST->HR[5]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH_DIGEST->HR[6]);
      msgdigest += 4U;
      *(uint32_t *)(msgdigest) = __REV(HASH_DIGEST->HR[7]);
#endif /* HASH_CR_MDMAT*/
      break;
    default:
      break;
  }
}



/**
  * @brief  Handle HASH processing Timeout.
  * @param  hhash HASH handle.
  * @param  Flag specifies the HASH flag to check.
  * @param  Status the Flag status (SET or RESET).
  * @param  Timeout Timeout duration.
  * @retval HAL status
  */
static HAL_StatusTypeDef HASH_WaitOnFlagUntilTimeout(HASH_HandleTypeDef *hhash, uint32_t Flag, FlagStatus Status,
                                                     uint32_t Timeout)
{
  uint32_t tickstart = HAL_GetTick();

  /* Wait until flag is set */
  if (Status == RESET)
  {
    while (__HAL_HASH_GET_FLAG(Flag) == RESET)
    {
      /* Check for the Timeout */
      if (Timeout != HAL_MAX_DELAY)
      {
        if (((HAL_GetTick() - tickstart) > Timeout) || (Timeout == 0U))
        {
          /* Set State to Ready to be able to restart later on */
          hhash->State  = HAL_HASH_STATE_READY;
          /* Store time out issue in handle status */
          hhash->Status = HAL_TIMEOUT;

          /* Process Unlocked */
          __HAL_UNLOCK(hhash);

          return HAL_TIMEOUT;
        }
      }
    }
  }
  else
  {
    while (__HAL_HASH_GET_FLAG(Flag) != RESET)
    {
      /* Check for the Timeout */
      if (Timeout != HAL_MAX_DELAY)
      {
        if (((HAL_GetTick() - tickstart) > Timeout) || (Timeout == 0U))
        {
          /* Set State to Ready to be able to restart later on */
          hhash->State  = HAL_HASH_STATE_READY;
          /* Store time out issue in handle status */
          hhash->Status = HAL_TIMEOUT;

          /* Process Unlocked */
          __HAL_UNLOCK(hhash);

          return HAL_TIMEOUT;
        }
      }
    }
  }
  return HAL_OK;
}


/**
  * @brief  HASH processing in interruption mode.
  * @param  hhash HASH handle.
  * @note   HASH_IT() regularly reads hhash->SuspendRequest to check whether
  *         or not the HASH processing must be suspended. If this is the case, the
  *         processing is suspended when possible and the Peripheral feeding point reached at
  *         suspension time is stored in the handle for resumption later on.
  * @retval HAL status
  */
static HAL_StatusTypeDef HASH_IT(HASH_HandleTypeDef *hhash)
{
  if (hhash->State == HAL_HASH_STATE_BUSY)
  {
    /* ITCounter must not be equal to 0 at this point. Report an error if this is the case. */
    if (hhash->HashITCounter == 0U)
    {
      /* Disable Interrupts */
      __HAL_HASH_DISABLE_IT(HASH_IT_DINI | HASH_IT_DCI);
      /* HASH state set back to Ready to prevent any issue in user code
         present in HAL_HASH_ErrorCallback() */
      hhash->State = HAL_HASH_STATE_READY;
      return HAL_ERROR;
    }
    else if (hhash->HashITCounter == 1U)
    {
      /* This is the first call to HASH_IT, the first input data are about to be
         entered in the Peripheral. A specific processing is carried out at this point to
         start-up the processing. */
      hhash->HashITCounter = 2U;
    }
    else
    {
      /* Cruise speed reached, HashITCounter remains equal to 3 until the end of
        the HASH processing or the end of the current step for HMAC processing. */
      hhash->HashITCounter = 3U;
    }

    /* If digest is ready */
    if (__HAL_HASH_GET_FLAG(HASH_FLAG_DCIS))
    {
      /* Read the digest */
      HASH_GetDigest(hhash->pHashOutBuffPtr, HASH_DIGEST_LENGTH());

      /* Disable Interrupts */
      __HAL_HASH_DISABLE_IT(HASH_IT_DINI | HASH_IT_DCI);
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_READY;
      /* Reset HASH state machine */
      hhash->Phase = HAL_HASH_PHASE_READY;
      /* Call digest computation complete call back */
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
      hhash->DgstCpltCallback(hhash);
#else
      HAL_HASH_DgstCpltCallback(hhash);
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */

      return HAL_OK;
    }

    /* If Peripheral ready to accept new data */
    if (__HAL_HASH_GET_FLAG(HASH_FLAG_DINIS))
    {

      /* If the suspension flag has been raised and if the processing is not about
         to end, suspend processing */
      if ((hhash->HashInCount != 0U) && (hhash->SuspendRequest == HAL_HASH_SUSPEND))
      {
        /* Disable Interrupts */
        __HAL_HASH_DISABLE_IT(HASH_IT_DINI | HASH_IT_DCI);

        /* Reset SuspendRequest */
        hhash->SuspendRequest = HAL_HASH_SUSPEND_NONE;

        /* Change the HASH state */
        hhash->State = HAL_HASH_STATE_SUSPENDED;

        return HAL_OK;
      }

      /* Enter input data in the Peripheral through HASH_Write_Block_Data() call and
        check whether the digest calculation has been triggered */
      if (HASH_Write_Block_Data(hhash) == HASH_DIGEST_CALCULATION_STARTED)
      {
        /* Call Input data transfer complete call back
           (called at the end of each step for HMAC) */
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
        hhash->InCpltCallback(hhash);
#else
        HAL_HASH_InCpltCallback(hhash);
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */

        if (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_1)
        {
          /* Wait until Peripheral is not busy anymore */
          if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_BUSY, SET, HASH_TIMEOUTVALUE) != HAL_OK)
          {
            /* Disable Interrupts */
            __HAL_HASH_DISABLE_IT(HASH_IT_DINI | HASH_IT_DCI);
            return HAL_TIMEOUT;
          }
          /* Initialization start for HMAC STEP 2 */
          hhash->Phase = HAL_HASH_PHASE_HMAC_STEP_2;        /* Move phase from Step 1 to Step 2 */
          __HAL_HASH_SET_NBVALIDBITS(hhash->HashBuffSize);  /* Set NBLW for the input message */
          hhash->HashInCount = hhash->HashBuffSize;         /* Set the input data size (in bytes) */
          hhash->pHashInBuffPtr = hhash->pHashMsgBuffPtr;   /* Set the input data address */
          hhash->HashITCounter = 1;                         /* Set ITCounter to 1 to indicate the start
                                                               of a new phase */
          __HAL_HASH_ENABLE_IT(HASH_IT_DINI);               /* Enable IT (was disabled in HASH_Write_Block_Data) */
        }
        else if (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_2)
        {
          /* Wait until Peripheral is not busy anymore */
          if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_BUSY, SET, HASH_TIMEOUTVALUE) != HAL_OK)
          {
            /* Disable Interrupts */
            __HAL_HASH_DISABLE_IT(HASH_IT_DINI | HASH_IT_DCI);
            return HAL_TIMEOUT;
          }
          /* Initialization start for HMAC STEP 3 */
          hhash->Phase = HAL_HASH_PHASE_HMAC_STEP_3;         /* Move phase from Step 2 to Step 3 */
          __HAL_HASH_SET_NBVALIDBITS(hhash->Init.KeySize);   /* Set NBLW for the key */
          hhash->HashInCount = hhash->Init.KeySize;          /* Set the key size (in bytes) */
          hhash->pHashInBuffPtr = hhash->Init.pKey;          /* Set the key address */
          hhash->HashITCounter = 1;                          /* Set ITCounter to 1 to indicate the start
                                                                of a new phase */
          __HAL_HASH_ENABLE_IT(HASH_IT_DINI);                /* Enable IT (was disabled in HASH_Write_Block_Data) */
        }
        else
        {
          /* Nothing to do */
        }
      } /* if (HASH_Write_Block_Data(hhash) == HASH_DIGEST_CALCULATION_STARTED) */
    }  /* if (__HAL_HASH_GET_FLAG(HASH_FLAG_DINIS))*/

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}


/**
  * @brief  Write a block of data in HASH Peripheral in interruption mode.
  * @param  hhash HASH handle.
  * @note   HASH_Write_Block_Data() is called under interruption by HASH_IT().
  * @retval HAL status
  */
static uint32_t HASH_Write_Block_Data(HASH_HandleTypeDef *hhash)
{
  uint32_t inputaddr;
  uint32_t buffercounter;
  uint32_t inputcounter;
  uint32_t ret = HASH_DIGEST_CALCULATION_NOT_STARTED;

  /* If there are more than 64 bytes remaining to be entered */
  if (hhash->HashInCount > 64U)
  {
    inputaddr = (uint32_t)hhash->pHashInBuffPtr;
    /* Write the Input block in the Data IN register
      (16 32-bit words, or 64 bytes are entered) */
    for (buffercounter = 0U; buffercounter < 64U; buffercounter += 4U)
    {
      HASH->DIN = *(uint32_t *)inputaddr;
      inputaddr += 4U;
    }
    /* If this is the start of input data entering, an additional word
      must be entered to start up the HASH processing */
    if (hhash->HashITCounter == 2U)
    {
      HASH->DIN = *(uint32_t *)inputaddr;
      if (hhash->HashInCount >= 68U)
      {
        /* There are still data waiting to be entered in the Peripheral.
           Decrement buffer counter and set pointer to the proper
           memory location for the next data entering round. */
        hhash->HashInCount -= 68U;
        hhash->pHashInBuffPtr += 68U;
      }
      else
      {
        /* All the input buffer has been fed to the HW. */
        hhash->HashInCount = 0U;
      }
    }
    else
    {
      /* 64 bytes have been entered and there are still some remaining:
         Decrement buffer counter and set pointer to the proper
        memory location for the next data entering round.*/
      hhash->HashInCount -= 64U;
      hhash->pHashInBuffPtr += 64U;
    }
  }
  else
  {
    /* 64 or less bytes remain to be entered. This is the last
      data entering round. */

    /* Get the buffer address */
    inputaddr = (uint32_t)hhash->pHashInBuffPtr;
    /* Get the buffer counter */
    inputcounter = hhash->HashInCount;
    /* Disable Interrupts */
    __HAL_HASH_DISABLE_IT(HASH_IT_DINI);

    /* Write the Input block in the Data IN register */
    for (buffercounter = 0U; buffercounter < ((inputcounter + 3U) / 4U); buffercounter++)
    {
      HASH->DIN = *(uint32_t *)inputaddr;
      inputaddr += 4U;
    }

    if (hhash->Accumulation == 1U)
    {
      /* Field accumulation is set, API only feeds data to the Peripheral and under interruption.
         The digest computation will be started when the last buffer data are entered. */

      /* Reset multi buffers accumulation flag */
      hhash->Accumulation = 0U;
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_READY;
      /* Call Input data transfer complete call back */
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
      hhash->InCpltCallback(hhash);
#else
      HAL_HASH_InCpltCallback(hhash);
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */
    }
    else
    {
      /* Start the Digest calculation */
      __HAL_HASH_START_DIGEST();
      /* Return indication that digest calculation has started:
         this return value triggers the call to Input data transfer
         complete call back as well as the proper transition from
         one step to another in HMAC mode. */
      ret = HASH_DIGEST_CALCULATION_STARTED;
    }
    /* Reset buffer counter */
    hhash->HashInCount = 0;
  }

  /* Return whether or digest calculation has started */
  return ret;
}

/**
  * @brief  HMAC processing in polling mode.
  * @param  hhash HASH handle.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
static HAL_StatusTypeDef HMAC_Processing(HASH_HandleTypeDef *hhash, uint32_t Timeout)
{
  /* Ensure first that Phase is correct */
  if ((hhash->Phase != HAL_HASH_PHASE_HMAC_STEP_1) && (hhash->Phase != HAL_HASH_PHASE_HMAC_STEP_2)
      && (hhash->Phase != HAL_HASH_PHASE_HMAC_STEP_3))
  {
    /* Change the HASH state */
    hhash->State = HAL_HASH_STATE_READY;

    /* Process Unlock */
    __HAL_UNLOCK(hhash);

    /* Return function status */
    return HAL_ERROR;
  }

  /* HMAC Step 1 processing */
  if (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_1)
  {
    /************************** STEP 1 ******************************************/
    /* Configure the Number of valid bits in last word of the message */
    __HAL_HASH_SET_NBVALIDBITS(hhash->Init.KeySize);

    /* Write input buffer in Data register */
    hhash->Status = HASH_WriteData(hhash, hhash->pHashKeyBuffPtr, hhash->HashKeyCount);
    if (hhash->Status != HAL_OK)
    {
      return hhash->Status;
    }

    /* Check whether or not key entering process has been suspended */
    if (hhash->State == HAL_HASH_STATE_SUSPENDED)
    {
      /* Process Unlocked */
      __HAL_UNLOCK(hhash);

      /* Stop right there and return function status */
      return HAL_OK;
    }

    /* No processing suspension at this point: set DCAL bit. */
    __HAL_HASH_START_DIGEST();

    /* Wait for BUSY flag to be cleared */
    if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_BUSY, SET, Timeout) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }

    /* Move from Step 1 to Step 2 */
    hhash->Phase = HAL_HASH_PHASE_HMAC_STEP_2;

  }

  /* HMAC Step 2 processing.
     After phase check, HMAC_Processing() may
     - directly start up from this point in resumption case
       if the same Step 2 processing was suspended previously
    - or fall through from the Step 1 processing carried out hereabove */
  if (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_2)
  {
    /************************** STEP 2 ******************************************/
    /* Configure the Number of valid bits in last word of the message */
    __HAL_HASH_SET_NBVALIDBITS(hhash->HashBuffSize);

    /* Write input buffer in Data register */
    hhash->Status = HASH_WriteData(hhash, hhash->pHashInBuffPtr, hhash->HashInCount);
    if (hhash->Status != HAL_OK)
    {
      return hhash->Status;
    }

    /* Check whether or not data entering process has been suspended */
    if (hhash->State == HAL_HASH_STATE_SUSPENDED)
    {
      /* Process Unlocked */
      __HAL_UNLOCK(hhash);

      /* Stop right there and return function status */
      return HAL_OK;
    }

    /* No processing suspension at this point: set DCAL bit. */
    __HAL_HASH_START_DIGEST();

    /* Wait for BUSY flag to be cleared */
    if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_BUSY, SET, Timeout) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }

    /* Move from Step 2 to Step 3 */
    hhash->Phase = HAL_HASH_PHASE_HMAC_STEP_3;
    /* In case Step 1 phase was suspended then resumed,
       set again Key input buffers and size before moving to
       next step */
    hhash->pHashKeyBuffPtr = hhash->Init.pKey;
    hhash->HashKeyCount    = hhash->Init.KeySize;
  }


  /* HMAC Step 3 processing.
      After phase check, HMAC_Processing() may
      - directly start up from this point in resumption case
        if the same Step 3 processing was suspended previously
     - or fall through from the Step 2 processing carried out hereabove */
  if (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_3)
  {
    /************************** STEP 3 ******************************************/
    /* Configure the Number of valid bits in last word of the message */
    __HAL_HASH_SET_NBVALIDBITS(hhash->Init.KeySize);

    /* Write input buffer in Data register */
    hhash->Status = HASH_WriteData(hhash, hhash->pHashKeyBuffPtr, hhash->HashKeyCount);
    if (hhash->Status != HAL_OK)
    {
      return hhash->Status;
    }

    /* Check whether or not key entering process has been suspended */
    if (hhash->State == HAL_HASH_STATE_SUSPENDED)
    {
      /* Process Unlocked */
      __HAL_UNLOCK(hhash);

      /* Stop right there and return function status */
      return HAL_OK;
    }

    /* No processing suspension at this point: start the Digest calculation. */
    __HAL_HASH_START_DIGEST();

    /* Wait for DCIS flag to be set */
    if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_DCIS, RESET, Timeout) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }

    /* Read the message digest */
    HASH_GetDigest(hhash->pHashOutBuffPtr, HASH_DIGEST_LENGTH());

    /* Reset HASH state machine */
    hhash->Phase = HAL_HASH_PHASE_READY;
  }

  /* Change the HASH state */
  hhash->State = HAL_HASH_STATE_READY;

  /* Process Unlock */
  __HAL_UNLOCK(hhash);

  /* Return function status */
  return HAL_OK;
}


/**
  * @brief  Initialize the HASH peripheral, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest.
  * @param  Timeout Timeout value.
  * @param  Algorithm HASH algorithm.
  * @retval HAL status
  */
HAL_StatusTypeDef HASH_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                             uint32_t Timeout, uint32_t Algorithm)
{
  uint8_t *pInBuffer_tmp;  /* input data address, input parameter of HASH_WriteData()         */
  uint32_t Size_tmp; /* input data size (in bytes), input parameter of HASH_WriteData() */
  HAL_HASH_StateTypeDef State_tmp = hhash->State;


  /* Initiate HASH processing in case of start or resumption */
  if ((State_tmp == HAL_HASH_STATE_READY) || (State_tmp == HAL_HASH_STATE_SUSPENDED))
  {
    /* Check input parameters */
    if ((pInBuffer == NULL) || (pOutBuffer == NULL))
    {
      hhash->State = HAL_HASH_STATE_READY;
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hhash);

    /* Check if initialization phase has not been already performed */
    if (hhash->Phase == HAL_HASH_PHASE_READY)
    {
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_BUSY;

      /* Select the HASH algorithm, clear HMAC mode and long key selection bit, reset the HASH processor core */
      MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT, Algorithm | HASH_CR_INIT);

      /* Configure the number of valid bits in last word of the message */
      __HAL_HASH_SET_NBVALIDBITS(Size);

      /* pInBuffer_tmp and Size_tmp are initialized to be used afterwards as
      input parameters of HASH_WriteData() */
      pInBuffer_tmp = pInBuffer;   /* pInBuffer_tmp is set to the input data address */
      Size_tmp = Size;             /* Size_tmp contains the input data size in bytes */

      /* Set the phase */
      hhash->Phase = HAL_HASH_PHASE_PROCESS;
    }
    else if (hhash->Phase == HAL_HASH_PHASE_PROCESS)
    {
      /* if the Peripheral has already been initialized, two cases are possible */

      /* Process resumption time ... */
      if (hhash->State == HAL_HASH_STATE_SUSPENDED)
      {
        /* Since this is resumption, pInBuffer_tmp and Size_tmp are not set
        to the API input parameters but to those saved beforehand by HASH_WriteData()
        when the processing was suspended */
        pInBuffer_tmp = hhash->pHashInBuffPtr;
        Size_tmp = hhash->HashInCount;
      }
      /* ... or multi-buffer HASH processing end */
      else
      {
        /* pInBuffer_tmp and Size_tmp are initialized to be used afterwards as
        input parameters of HASH_WriteData() */
        pInBuffer_tmp = pInBuffer;
        Size_tmp = Size;
        /* Configure the number of valid bits in last word of the message */
        __HAL_HASH_SET_NBVALIDBITS(Size);
      }
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_BUSY;
    }
    else
    {
      /* Phase error */
      hhash->State = HAL_HASH_STATE_READY;

      /* Process Unlocked */
      __HAL_UNLOCK(hhash);

      /* Return function status */
      return HAL_ERROR;
    }


    /* Write input buffer in Data register */
    hhash->Status = HASH_WriteData(hhash, pInBuffer_tmp, Size_tmp);
    if (hhash->Status != HAL_OK)
    {
      return hhash->Status;
    }

    /* If the process has not been suspended, carry on to digest calculation */
    if (hhash->State != HAL_HASH_STATE_SUSPENDED)
    {
      /* Start the Digest calculation */
      __HAL_HASH_START_DIGEST();

      /* Wait for DCIS flag to be set */
      if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_DCIS, RESET, Timeout) != HAL_OK)
      {
        return HAL_TIMEOUT;
      }

      /* Read the message digest */
      HASH_GetDigest(pOutBuffer, HASH_DIGEST_LENGTH());

      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_READY;

      /* Reset HASH state machine */
      hhash->Phase = HAL_HASH_PHASE_READY;

    }

    /* Process Unlocked */
    __HAL_UNLOCK(hhash);

    /* Return function status */
    return HAL_OK;

  }
  else
  {
    return HAL_BUSY;
  }
}


/**
  * @brief  If not already done, initialize the HASH peripheral then
  *         processes pInBuffer.
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @param  Algorithm HASH algorithm.
  * @retval HAL status
  */
HAL_StatusTypeDef HASH_Accumulate(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint32_t Algorithm)
{
  uint8_t *pInBuffer_tmp;   /* input data address, input parameter of HASH_WriteData()         */
  uint32_t Size_tmp;  /* input data size (in bytes), input parameter of HASH_WriteData() */
  HAL_HASH_StateTypeDef State_tmp = hhash->State;

  /* Make sure the input buffer size (in bytes) is a multiple of 4 */
  if ((Size % 4U) != 0U)
  {
    return  HAL_ERROR;
  }

  /* Initiate HASH processing in case of start or resumption */
  if ((State_tmp == HAL_HASH_STATE_READY) || (State_tmp == HAL_HASH_STATE_SUSPENDED))
  {
    /* Check input parameters */
    if ((pInBuffer == NULL) || (Size == 0U))
    {
      hhash->State = HAL_HASH_STATE_READY;
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hhash);

    /* If resuming the HASH processing */
    if (hhash->State == HAL_HASH_STATE_SUSPENDED)
    {
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_BUSY;

      /* Since this is resumption, pInBuffer_tmp and Size_tmp are not set
         to the API input parameters but to those saved beforehand by HASH_WriteData()
         when the processing was suspended */
      pInBuffer_tmp = hhash->pHashInBuffPtr;  /* pInBuffer_tmp is set to the input data address */
      Size_tmp = hhash->HashInCount;          /* Size_tmp contains the input data size in bytes */

    }
    else
    {
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_BUSY;

      /* pInBuffer_tmp and Size_tmp are initialized to be used afterwards as
         input parameters of HASH_WriteData() */
      pInBuffer_tmp = pInBuffer;    /* pInBuffer_tmp is set to the input data address */
      Size_tmp = Size;              /* Size_tmp contains the input data size in bytes */

      /* Check if initialization phase has already be performed */
      if (hhash->Phase == HAL_HASH_PHASE_READY)
      {
        /* Select the HASH algorithm, clear HMAC mode and long key selection bit, reset the HASH processor core */
        MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT, Algorithm | HASH_CR_INIT);
      }

      /* Set the phase */
      hhash->Phase = HAL_HASH_PHASE_PROCESS;

    }

    /* Write input buffer in Data register */
    hhash->Status = HASH_WriteData(hhash, pInBuffer_tmp, Size_tmp);
    if (hhash->Status != HAL_OK)
    {
      return hhash->Status;
    }

    /* If the process has not been suspended, move the state to Ready */
    if (hhash->State != HAL_HASH_STATE_SUSPENDED)
    {
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_READY;
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hhash);

    /* Return function status */
    return HAL_OK;

  }
  else
  {
    return HAL_BUSY;
  }


}


/**
  * @brief  If not already done, initialize the HASH peripheral then
  *         processes pInBuffer in interruption mode.
  * @note   Field hhash->Phase of HASH handle is tested to check whether or not
  *         the Peripheral has already been initialized.
  * @note   The input buffer size (in bytes) must be a multiple of 4 otherwise, the
  *         HASH digest computation is corrupted.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes, must be a multiple of 4.
  * @param  Algorithm HASH algorithm.
  * @retval HAL status
  */
HAL_StatusTypeDef HASH_Accumulate_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint32_t Algorithm)
{
  HAL_HASH_StateTypeDef State_tmp = hhash->State;
  __IO uint32_t inputaddr = (uint32_t) pInBuffer;
  uint32_t SizeVar = Size;

  /* Make sure the input buffer size (in bytes) is a multiple of 4 */
  if ((Size % 4U) != 0U)
  {
    return  HAL_ERROR;
  }

  /* Initiate HASH processing in case of start or resumption */
  if ((State_tmp == HAL_HASH_STATE_READY) || (State_tmp == HAL_HASH_STATE_SUSPENDED))
  {
    /* Check input parameters */
    if ((pInBuffer == NULL) || (Size == 0U))
    {
      hhash->State = HAL_HASH_STATE_READY;
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hhash);

    /* If resuming the HASH processing */
    if (hhash->State == HAL_HASH_STATE_SUSPENDED)
    {
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_BUSY;
    }
    else
    {
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_BUSY;

      /* Check if initialization phase has already be performed */
      if (hhash->Phase == HAL_HASH_PHASE_READY)
      {
        /* Select the HASH algorithm, clear HMAC mode and long key selection bit, reset the HASH processor core */
        MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT, Algorithm | HASH_CR_INIT);
        hhash->HashITCounter = 1;
      }
      else
      {
        hhash->HashITCounter = 3; /* 'cruise-speed' reached during a previous buffer processing */
      }

      /* Set the phase */
      hhash->Phase = HAL_HASH_PHASE_PROCESS;

      /* If DINIS is equal to 0 (for example if an incomplete block has been previously
       fed to the Peripheral), the DINIE interruption won't be triggered when DINIE is set.
       Therefore, first words are manually entered until DINIS raises, or until there
       is not more data to enter. */
      while ((!(__HAL_HASH_GET_FLAG(HASH_FLAG_DINIS))) && (SizeVar > 0U))
      {

        /* Write input data 4 bytes at a time */
        HASH->DIN = *(uint32_t *)inputaddr;
        inputaddr += 4U;
        SizeVar -= 4U;
      }

      /* If DINIS is still not set or if all the data have been fed, stop here */
      if ((!(__HAL_HASH_GET_FLAG(HASH_FLAG_DINIS))) || (SizeVar == 0U))
      {
        /* Change the HASH state */
        hhash->State = HAL_HASH_STATE_READY;

        /* Process Unlock */
        __HAL_UNLOCK(hhash);

        /* Return function status */
        return HAL_OK;
      }

      /* otherwise, carry on in interrupt-mode */
      hhash->HashInCount = SizeVar;               /* Counter used to keep track of number of data
                                                  to be fed to the Peripheral */
      hhash->pHashInBuffPtr = (uint8_t *)inputaddr;       /* Points at data which will be fed to the Peripheral at
                                                  the next interruption */
      /* In case of suspension, hhash->HashInCount and hhash->pHashInBuffPtr contain
         the information describing where the HASH process is stopped.
         These variables are used later on to resume the HASH processing at the
         correct location. */

    }

    /* Set multi buffers accumulation flag */
    hhash->Accumulation = 1U;

    /* Process Unlock */
    __HAL_UNLOCK(hhash);

    /* Enable Data Input interrupt */
    __HAL_HASH_ENABLE_IT(HASH_IT_DINI);

    /* Return function status */
    return HAL_OK;

  }
  else
  {
    return HAL_BUSY;
  }

}



/**
  * @brief  Initialize the HASH peripheral, next process pInBuffer then
  *         read the computed digest in interruption mode.
  * @note   Digest is available in pOutBuffer.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest.
  * @param  Algorithm HASH algorithm.
  * @retval HAL status
  */
HAL_StatusTypeDef HASH_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                uint32_t Algorithm)
{
  HAL_HASH_StateTypeDef State_tmp = hhash->State;
  __IO uint32_t inputaddr = (uint32_t) pInBuffer;
  uint32_t polling_step = 0U;
  uint32_t initialization_skipped = 0U;
  uint32_t SizeVar = Size;

  /* If State is ready or suspended, start or resume IT-based HASH processing */
  if ((State_tmp == HAL_HASH_STATE_READY) || (State_tmp == HAL_HASH_STATE_SUSPENDED))
  {
    /* Check input parameters */
    if ((pInBuffer == NULL) || (Size == 0U) || (pOutBuffer == NULL))
    {
      hhash->State = HAL_HASH_STATE_READY;
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hhash);

    /* Change the HASH state */
    hhash->State = HAL_HASH_STATE_BUSY;

    /* Initialize IT counter */
    hhash->HashITCounter = 1;

    /* Check if initialization phase has already be performed */
    if (hhash->Phase == HAL_HASH_PHASE_READY)
    {
      /* Select the HASH algorithm, clear HMAC mode and long key selection bit, reset the HASH processor core */
      MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT, Algorithm | HASH_CR_INIT);

      /* Configure the number of valid bits in last word of the message */
      __HAL_HASH_SET_NBVALIDBITS(SizeVar);


      hhash->HashInCount = SizeVar;            /* Counter used to keep track of number of data
                                                  to be fed to the Peripheral */
      hhash->pHashInBuffPtr = pInBuffer;       /* Points at data which will be fed to the Peripheral at
                                                  the next interruption */
      /* In case of suspension, hhash->HashInCount and hhash->pHashInBuffPtr contain
         the information describing where the HASH process is stopped.
         These variables are used later on to resume the HASH processing at the
         correct location. */

      hhash->pHashOutBuffPtr = pOutBuffer;     /* Points at the computed digest */
    }
    else
    {
      initialization_skipped = 1; /* info user later on in case of multi-buffer */
    }

    /* Set the phase */
    hhash->Phase = HAL_HASH_PHASE_PROCESS;

    /* If DINIS is equal to 0 (for example if an incomplete block has been previously
      fed to the Peripheral), the DINIE interruption won't be triggered when DINIE is set.
      Therefore, first words are manually entered until DINIS raises. */
    while ((!(__HAL_HASH_GET_FLAG(HASH_FLAG_DINIS))) && (SizeVar > 3U))
    {
      polling_step = 1U; /* note that some words are entered before enabling the interrupt */

      /* Write input data 4 bytes at a time */
      HASH->DIN = *(uint32_t *)inputaddr;
      inputaddr += 4U;
      SizeVar -= 4U;
    }

    if (polling_step == 1U)
    {
      if (SizeVar == 0U)
      {
        /* If all the data have been entered at this point, it only remains to
         read the digest */
        hhash->pHashOutBuffPtr = pOutBuffer;     /* Points at the computed digest */

        /* Start the Digest calculation */
        __HAL_HASH_START_DIGEST();
        /* Process Unlock */
        __HAL_UNLOCK(hhash);

        /* Enable Interrupts */
        __HAL_HASH_ENABLE_IT(HASH_IT_DCI);

        /* Return function status */
        return HAL_OK;
      }
      else if (__HAL_HASH_GET_FLAG(HASH_FLAG_DINIS))
      {
        /* It remains data to enter and the Peripheral is ready to trigger DINIE,
           carry on as usual.
           Update HashInCount and pHashInBuffPtr accordingly. */
        hhash->HashInCount = SizeVar;
        hhash->pHashInBuffPtr = (uint8_t *)inputaddr;
        __HAL_HASH_SET_NBVALIDBITS(
          SizeVar);  /* Update the configuration of the number of valid bits in last word of the message */
        hhash->pHashOutBuffPtr = pOutBuffer;  /* Points at the computed digest */
        if (initialization_skipped == 1U)
        {
          hhash->HashITCounter = 3; /* 'cruise-speed' reached during a previous buffer processing */
        }
      }
      else
      {
        /* DINIS is not set but it remains a few data to enter (not enough for a full word).
           Manually enter the last bytes before enabling DCIE. */
        __HAL_HASH_SET_NBVALIDBITS(SizeVar);
        HASH->DIN = *(uint32_t *)inputaddr;

        /* Start the Digest calculation */
        hhash->pHashOutBuffPtr = pOutBuffer;     /* Points at the computed digest */
        __HAL_HASH_START_DIGEST();
        /* Process Unlock */
        __HAL_UNLOCK(hhash);

        /* Enable Interrupts */
        __HAL_HASH_ENABLE_IT(HASH_IT_DCI);

        /* Return function status */
        return HAL_OK;
      }
    } /*  if (polling_step == 1) */


    /* Process Unlock */
    __HAL_UNLOCK(hhash);

    /* Enable Interrupts */
    __HAL_HASH_ENABLE_IT(HASH_IT_DINI | HASH_IT_DCI);

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }

}


/**
  * @brief  Initialize the HASH peripheral then initiate a DMA transfer
  *         to feed the input buffer to the Peripheral.
  * @note   If MDMAT bit is set before calling this function (multi-buffer
  *          HASH processing case), the input buffer size (in bytes) must be
  *          a multiple of 4 otherwise, the HASH digest computation is corrupted.
  *          For the processing of the last buffer of the thread, MDMAT bit must
  *          be reset and the buffer length (in bytes) doesn't have to be a
  *          multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  Algorithm HASH algorithm.
  * @retval HAL status
  */
HAL_StatusTypeDef HASH_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint32_t Algorithm)
{
  uint32_t inputaddr;
  uint32_t inputSize;
  HAL_StatusTypeDef status ;
  HAL_HASH_StateTypeDef State_tmp = hhash->State;

  #if defined (HASH_CR_MDMAT)
  /* Make sure the input buffer size (in bytes) is a multiple of 4 when MDMAT bit is set
     (case of multi-buffer HASH processing) */
  assert_param(IS_HASH_DMA_MULTIBUFFER_SIZE(Size));
  #endif /* MDMA defined*/
  /* If State is ready or suspended, start or resume polling-based HASH processing */
  if ((State_tmp == HAL_HASH_STATE_READY) || (State_tmp == HAL_HASH_STATE_SUSPENDED))
  {
    /* Check input parameters */
    if ((pInBuffer == NULL) || (Size == 0U) ||
        /* Check phase coherency. Phase must be
           either READY (fresh start)
           or PROCESS (multi-buffer HASH management) */
        ((hhash->Phase != HAL_HASH_PHASE_READY) && (!(IS_HASH_PROCESSING(hhash)))))
    {
      hhash->State = HAL_HASH_STATE_READY;
      return  HAL_ERROR;
    }


    /* Process Locked */
    __HAL_LOCK(hhash);

    /* If not a resumption case */
    if (hhash->State == HAL_HASH_STATE_READY)
    {
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_BUSY;

      /* Check if initialization phase has already been performed.
         If Phase is already set to HAL_HASH_PHASE_PROCESS, this means the
         API is processing a new input data message in case of multi-buffer HASH
         computation. */
      if (hhash->Phase == HAL_HASH_PHASE_READY)
      {
        /* Select the HASH algorithm, clear HMAC mode and long key selection bit, reset the HASH processor core */
        MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT, Algorithm | HASH_CR_INIT);

        /* Set the phase */
        hhash->Phase = HAL_HASH_PHASE_PROCESS;
      }

      /* Configure the Number of valid bits in last word of the message */
      __HAL_HASH_SET_NBVALIDBITS(Size);

      inputaddr = (uint32_t)pInBuffer;     /* DMA transfer start address   */
      inputSize = Size;                    /* DMA transfer size (in bytes) */

      /* In case of suspension request, save the starting parameters */
      hhash->pHashInBuffPtr =  pInBuffer;  /* DMA transfer start address   */
      hhash->HashInCount = Size;           /* DMA transfer size (in bytes) */

    }
    /* If resumption case */
    else
    {
      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_BUSY;

      /* Resumption case, inputaddr and inputSize are not set to the API input parameters
         but to those saved beforehand by HAL_HASH_DMAFeed_ProcessSuspend() when the
         processing was suspended */
      inputaddr = (uint32_t)hhash->pHashInBuffPtr;  /* DMA transfer start address   */
      inputSize = hhash->HashInCount;               /* DMA transfer size (in bytes) */

    }

    /* Set the HASH DMA transfer complete callback */
    hhash->hdmain->XferCpltCallback = HASH_DMAXferCplt;
    /* Set the DMA error callback */
    hhash->hdmain->XferErrorCallback = HASH_DMAError;

    /* Store number of words already pushed to manage proper DMA processing suspension */
    hhash->NbWordsAlreadyPushed = HASH_NBW_PUSHED();

    /* Enable the DMA In DMA stream */
    status = HAL_DMA_Start_IT(hhash->hdmain, inputaddr, (uint32_t)&HASH->DIN, \
                              (((inputSize % 4U) != 0U) ? ((inputSize + (4U - (inputSize % 4U))) / 4U) : \
                               (inputSize / 4U)));

    /* Enable DMA requests */
    SET_BIT(HASH->CR, HASH_CR_DMAE);

    /* Process Unlock */
    __HAL_UNLOCK(hhash);

    /* Return function status */
    if (status != HAL_OK)
    {
      /* Update HASH state machine to error */
      hhash->State = HAL_HASH_STATE_ERROR;
    }

    return status;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Return the computed digest.
  * @note   The API waits for DCIS to be set then reads the computed digest.
  * @param  hhash HASH handle.
  * @param  pOutBuffer pointer to the computed digest.
  * @param  Timeout Timeout value.
  * @retval HAL status
  */
HAL_StatusTypeDef HASH_Finish(HASH_HandleTypeDef *hhash, uint8_t *pOutBuffer, uint32_t Timeout)
{

  if (hhash->State == HAL_HASH_STATE_READY)
  {
    /* Check parameter */
    if (pOutBuffer == NULL)
    {
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hhash);

    /* Change the HASH state to busy */
    hhash->State = HAL_HASH_STATE_BUSY;

    /* Wait for DCIS flag to be set */
    if (HASH_WaitOnFlagUntilTimeout(hhash, HASH_FLAG_DCIS, RESET, Timeout) != HAL_OK)
    {
      return HAL_TIMEOUT;
    }

    /* Read the message digest */
    HASH_GetDigest(pOutBuffer, HASH_DIGEST_LENGTH());

    /* Change the HASH state to ready */
    hhash->State = HAL_HASH_STATE_READY;

    /* Reset HASH state machine */
    hhash->Phase = HAL_HASH_PHASE_READY;

    /* Process UnLock */
    __HAL_UNLOCK(hhash);

    /* Return function status */
    return HAL_OK;

  }
  else
  {
    return HAL_BUSY;
  }

}


/**
  * @brief  Initialize the HASH peripheral in HMAC mode, next process pInBuffer then
  *         read the computed digest.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest.
  * @param  Timeout Timeout value.
  * @param  Algorithm HASH algorithm.
  * @retval HAL status
  */
HAL_StatusTypeDef HMAC_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                             uint32_t Timeout, uint32_t Algorithm)
{
  HAL_HASH_StateTypeDef State_tmp = hhash->State;

  /* If State is ready or suspended, start or resume polling-based HASH processing */
  if ((State_tmp == HAL_HASH_STATE_READY) || (State_tmp == HAL_HASH_STATE_SUSPENDED))
  {
    /* Check input parameters */
    if ((pInBuffer == NULL) || (Size == 0U) || (hhash->Init.pKey == NULL) || (hhash->Init.KeySize == 0U)
        || (pOutBuffer == NULL))
    {
      hhash->State = HAL_HASH_STATE_READY;
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hhash);

    /* Change the HASH state */
    hhash->State = HAL_HASH_STATE_BUSY;

    /* Check if initialization phase has already be performed */
    if (hhash->Phase == HAL_HASH_PHASE_READY)
    {
      /* Check if key size is larger than 64 bytes, accordingly set LKEY and the other setting bits */
      if (hhash->Init.KeySize > 64U)
      {
        MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT,
                   Algorithm | HASH_ALGOMODE_HMAC | HASH_HMAC_KEYTYPE_LONGKEY | HASH_CR_INIT);
      }
      else
      {
        MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT,
                   Algorithm | HASH_ALGOMODE_HMAC | HASH_CR_INIT);
      }
      /* Set the phase to Step 1 */
      hhash->Phase = HAL_HASH_PHASE_HMAC_STEP_1;
      /* Resort to hhash internal fields to feed the Peripheral.
         Parameters will be updated in case of suspension to contain the proper
         information at resumption time. */
      hhash->pHashOutBuffPtr  = pOutBuffer;            /* Output digest address    */
      hhash->pHashInBuffPtr   = pInBuffer;             /* Input data address, HMAC_Processing input
                                                          parameter for Step 2     */
      hhash->HashInCount      = Size;                  /* Input data size, HMAC_Processing input
                                                          parameter for Step 2        */
      hhash->HashBuffSize     = Size;                  /* Store the input buffer size for the whole HMAC process*/
      hhash->pHashKeyBuffPtr  = hhash->Init.pKey;      /* Key address, HMAC_Processing input parameter for Step
                                                          1 and Step 3 */
      hhash->HashKeyCount     = hhash->Init.KeySize;   /* Key size, HMAC_Processing input parameter for Step 1
                                                          and Step 3    */
    }

    /* Carry out HMAC processing */
    return HMAC_Processing(hhash, Timeout);

  }
  else
  {
    return HAL_BUSY;
  }
}



/**
  * @brief  Initialize the HASH peripheral in HMAC mode, next process pInBuffer then
  *         read the computed digest in interruption mode.
  * @note   Digest is available in pOutBuffer.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  pOutBuffer pointer to the computed digest.
  * @param  Algorithm HASH algorithm.
  * @retval HAL status
  */
HAL_StatusTypeDef HMAC_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                uint32_t Algorithm)
{
  HAL_HASH_StateTypeDef State_tmp = hhash->State;

  /* If State is ready or suspended, start or resume IT-based HASH processing */
  if ((State_tmp == HAL_HASH_STATE_READY) || (State_tmp == HAL_HASH_STATE_SUSPENDED))
  {
    /* Check input parameters */
    if ((pInBuffer == NULL) || (Size == 0U) || (hhash->Init.pKey == NULL) || (hhash->Init.KeySize == 0U)
        || (pOutBuffer == NULL))
    {
      hhash->State = HAL_HASH_STATE_READY;
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hhash);

    /* Change the HASH state */
    hhash->State = HAL_HASH_STATE_BUSY;

    /* Initialize IT counter */
    hhash->HashITCounter = 1;

    /* Check if initialization phase has already be performed */
    if (hhash->Phase == HAL_HASH_PHASE_READY)
    {
      /* Check if key size is larger than 64 bytes, accordingly set LKEY and the other setting bits */
      if (hhash->Init.KeySize > 64U)
      {
        MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT,
                   Algorithm | HASH_ALGOMODE_HMAC | HASH_HMAC_KEYTYPE_LONGKEY | HASH_CR_INIT);
      }
      else
      {
        MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT,
                   Algorithm | HASH_ALGOMODE_HMAC | HASH_CR_INIT);
      }

      /* Resort to hhash internal fields hhash->pHashInBuffPtr and hhash->HashInCount
         to feed the Peripheral whatever the HMAC step.
         Lines below are set to start HMAC Step 1 processing where key is entered first. */
      hhash->HashInCount     = hhash->Init.KeySize; /* Key size                      */
      hhash->pHashInBuffPtr  = hhash->Init.pKey ;   /* Key address                   */

      /* Store input and output parameters in handle fields to manage steps transition
         or possible HMAC suspension/resumption */
      hhash->pHashKeyBuffPtr = hhash->Init.pKey;    /* Key address                   */
      hhash->pHashMsgBuffPtr = pInBuffer;           /* Input message address         */
      hhash->HashBuffSize    = Size;                /* Input message size (in bytes) */
      hhash->pHashOutBuffPtr = pOutBuffer;          /* Output digest address         */

      /* Configure the number of valid bits in last word of the key */
      __HAL_HASH_SET_NBVALIDBITS(hhash->Init.KeySize);

      /* Set the phase to Step 1 */
      hhash->Phase = HAL_HASH_PHASE_HMAC_STEP_1;
    }
    else if ((hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_1) || (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_3))
    {
      /* Restart IT-based HASH processing after Step 1 or Step 3 suspension */

    }
    else if (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_2)
    {
      /* Restart IT-based HASH processing after Step 2 suspension */

    }
    else
    {
      /* Error report as phase incorrect */
      /* Process Unlock */
      __HAL_UNLOCK(hhash);
      hhash->State = HAL_HASH_STATE_READY;
      return HAL_ERROR;
    }

    /* Process Unlock */
    __HAL_UNLOCK(hhash);

    /* Enable Interrupts */
    __HAL_HASH_ENABLE_IT(HASH_IT_DINI | HASH_IT_DCI);

    /* Return function status */
    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }

}



/**
  * @brief  Initialize the HASH peripheral in HMAC mode then initiate the required
  *         DMA transfers to feed the key and the input buffer to the Peripheral.
  * @note   Same key is used for the inner and the outer hash functions; pointer to key and
  *         key size are respectively stored in hhash->Init.pKey and hhash->Init.KeySize.
  * @note   In case of multi-buffer HMAC processing, the input buffer size (in bytes) must
  *         be a multiple of 4 otherwise, the HASH digest computation is corrupted.
  *         Only the length of the last buffer of the thread doesn't have to be a
  *         multiple of 4.
  * @param  hhash HASH handle.
  * @param  pInBuffer pointer to the input buffer (buffer to be hashed).
  * @param  Size length of the input buffer in bytes.
  * @param  Algorithm HASH algorithm.
  * @retval HAL status
  */
HAL_StatusTypeDef HMAC_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint32_t Algorithm)
{
  uint32_t inputaddr;
  uint32_t inputSize;
  HAL_StatusTypeDef status ;
  HAL_HASH_StateTypeDef State_tmp = hhash->State;
  /* Make sure the input buffer size (in bytes) is a multiple of 4 when digest calculation
     is disabled (multi-buffer HMAC processing, MDMAT bit to be set) */
  assert_param(IS_HMAC_DMA_MULTIBUFFER_SIZE(hhash, Size));
  /* If State is ready or suspended, start or resume DMA-based HASH processing */
  if ((State_tmp == HAL_HASH_STATE_READY) || (State_tmp == HAL_HASH_STATE_SUSPENDED))
  {
    /* Check input parameters */
    if ((pInBuffer == NULL) || (Size == 0U) || (hhash->Init.pKey == NULL) || (hhash->Init.KeySize == 0U) ||
        /* Check phase coherency. Phase must be
            either READY (fresh start)
            or one of HMAC PROCESS steps (multi-buffer HASH management) */
        ((hhash->Phase != HAL_HASH_PHASE_READY) && (!(IS_HMAC_PROCESSING(hhash)))))
    {
      hhash->State = HAL_HASH_STATE_READY;
      return  HAL_ERROR;
    }


    /* Process Locked */
    __HAL_LOCK(hhash);

    /* If not a case of resumption after suspension */
    if (hhash->State == HAL_HASH_STATE_READY)
    {
      /* Check whether or not initialization phase has already be performed */
      if (hhash->Phase == HAL_HASH_PHASE_READY)
      {
        /* Change the HASH state */
        hhash->State = HAL_HASH_STATE_BUSY;
#if defined(HASH_CR_MDMAT)
        /* Check if key size is larger than 64 bytes, accordingly set LKEY and the other setting bits.
           At the same time, ensure MDMAT bit is cleared. */
        if (hhash->Init.KeySize > 64U)
        {
          MODIFY_REG(HASH->CR, HASH_CR_MDMAT | HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT,
                     Algorithm | HASH_ALGOMODE_HMAC | HASH_HMAC_KEYTYPE_LONGKEY | HASH_CR_INIT);
        }
        else
        {
          MODIFY_REG(HASH->CR, HASH_CR_MDMAT | HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT,
                     Algorithm | HASH_ALGOMODE_HMAC | HASH_CR_INIT);
        }
#else
        /* Check if key size is larger than 64 bytes, accordingly set LKEY and the other setting bits */
        if (hhash->Init.KeySize > 64U)
        {
          MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT,
                     Algorithm | HASH_ALGOMODE_HMAC | HASH_HMAC_KEYTYPE_LONGKEY | HASH_CR_INIT);
        }
        else
        {
          MODIFY_REG(HASH->CR, HASH_CR_LKEY | HASH_CR_ALGO | HASH_CR_MODE | HASH_CR_INIT,
                     Algorithm | HASH_ALGOMODE_HMAC | HASH_CR_INIT);
        }
#endif /* HASH_CR_MDMAT*/
        /* Store input aparameters in handle fields to manage steps transition
           or possible HMAC suspension/resumption */
        hhash->HashInCount = hhash->Init.KeySize;   /* Initial size for first DMA transfer (key size)      */
        hhash->pHashKeyBuffPtr = hhash->Init.pKey;  /* Key address                                         */
        hhash->pHashInBuffPtr  = hhash->Init.pKey ; /* First address passed to DMA (key address at Step 1) */
        hhash->pHashMsgBuffPtr = pInBuffer;         /* Input data address                                  */
        hhash->HashBuffSize = Size;                 /* input data size (in bytes)                          */

        /* Set DMA input parameters */
        inputaddr = (uint32_t)(hhash->Init.pKey);   /* Address passed to DMA (start by entering Key message) */
        inputSize = hhash->Init.KeySize;            /* Size for first DMA transfer (in bytes) */

        /* Configure the number of valid bits in last word of the key */
        __HAL_HASH_SET_NBVALIDBITS(hhash->Init.KeySize);

        /* Set the phase to Step 1 */
        hhash->Phase = HAL_HASH_PHASE_HMAC_STEP_1;

      }
      else if (hhash->Phase == HAL_HASH_PHASE_HMAC_STEP_2)
      {
        /* Process a new input data message in case of multi-buffer HMAC processing
          (this is not a resumption case) */

        /* Change the HASH state */
        hhash->State = HAL_HASH_STATE_BUSY;

        /* Save input parameters to be able to manage possible suspension/resumption */
        hhash->HashInCount = Size;                /* Input message address       */
        hhash->pHashInBuffPtr = pInBuffer;        /* Input message size in bytes */

        /* Set DMA input parameters */
        inputaddr = (uint32_t)pInBuffer;           /* Input message address       */
        inputSize = Size;                          /* Input message size in bytes */

        if (hhash->DigestCalculationDisable == RESET)
        {
          /* This means this is the last buffer of the multi-buffer sequence: DCAL needs to be set. */
#if defined(HASH_CR_MDMAT)
          __HAL_HASH_RESET_MDMAT();
#endif  /* HASH_CR_MDMAT*/
          __HAL_HASH_SET_NBVALIDBITS(inputSize);
        }
      }
      else
      {
        /* Phase not aligned with handle READY state */
        __HAL_UNLOCK(hhash);
        /* Return function status */
        return HAL_ERROR;
      }
    }
    else
    {
      /* Resumption case (phase may be Step 1, 2 or 3) */

      /* Change the HASH state */
      hhash->State = HAL_HASH_STATE_BUSY;

      /* Set DMA input parameters at resumption location;
         inputaddr and inputSize are not set to the API input parameters
         but to those saved beforehand by HAL_HASH_DMAFeed_ProcessSuspend() when the
         processing was suspended. */
      inputaddr = (uint32_t)(hhash->pHashInBuffPtr);  /* Input message address       */
      inputSize = hhash->HashInCount;                 /* Input message size in bytes */
    }


    /* Set the HASH DMA transfer complete callback */
    hhash->hdmain->XferCpltCallback = HASH_DMAXferCplt;
    /* Set the DMA error callback */
    hhash->hdmain->XferErrorCallback = HASH_DMAError;

    /* Store number of words already pushed to manage proper DMA processing suspension */
    hhash->NbWordsAlreadyPushed = HASH_NBW_PUSHED();

    /* Enable the DMA In DMA stream */
    status = HAL_DMA_Start_IT(hhash->hdmain, inputaddr, (uint32_t)&HASH->DIN,  \
                              (((inputSize % 4U) != 0U) ? ((inputSize + (4U - (inputSize % 4U))) / 4U) \
                              : (inputSize / 4U)));
    /* Enable DMA requests */
    SET_BIT(HASH->CR, HASH_CR_DMAE);

    /* Process Unlocked */
    __HAL_UNLOCK(hhash);

    /* Return function status */
    if (status != HAL_OK)
    {
      /* Update HASH state machine to error */
      hhash->State = HAL_HASH_STATE_ERROR;
    }

    /* Return function status */
    return status;
  }
  else
  {
    return HAL_BUSY;
  }
}
/**
  * @}
  */

#endif /* HAL_HASH_MODULE_ENABLED */

/**
  * @}
  */
#endif /*  HASH*/
/**
  * @}
  */



/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
