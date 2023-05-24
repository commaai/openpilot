/**
  ******************************************************************************
  * @file    stm32f4xx_hal_rng.c
  * @author  MCD Application Team
  * @brief   RNG HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the Random Number Generator (RNG) peripheral:
  *           + Initialization and configuration functions
  *           + Peripheral Control functions
  *           + Peripheral State functions
  *
  @verbatim
  ==============================================================================
                     ##### How to use this driver #####
  ==============================================================================
  [..]
      The RNG HAL driver can be used as follows:

      (#) Enable the RNG controller clock using __HAL_RCC_RNG_CLK_ENABLE() macro
          in HAL_RNG_MspInit().
      (#) Activate the RNG peripheral using HAL_RNG_Init() function.
      (#) Wait until the 32 bit Random Number Generator contains a valid
          random data using (polling/interrupt) mode.
      (#) Get the 32 bit random number using HAL_RNG_GenerateRandomNumber() function.

    ##### Callback registration #####
    ==================================

    [..]
    The compilation define USE_HAL_RNG_REGISTER_CALLBACKS when set to 1
    allows the user to configure dynamically the driver callbacks.

    [..]
    Use Function HAL_RNG_RegisterCallback() to register a user callback.
    Function HAL_RNG_RegisterCallback() allows to register following callbacks:
    (+) ErrorCallback             : RNG Error Callback.
    (+) MspInitCallback           : RNG MspInit.
    (+) MspDeInitCallback         : RNG MspDeInit.
    This function takes as parameters the HAL peripheral handle, the Callback ID
    and a pointer to the user callback function.

    [..]
    Use function HAL_RNG_UnRegisterCallback() to reset a callback to the default
    weak (surcharged) function.
    HAL_RNG_UnRegisterCallback() takes as parameters the HAL peripheral handle,
    and the Callback ID.
    This function allows to reset following callbacks:
    (+) ErrorCallback             : RNG Error Callback.
    (+) MspInitCallback           : RNG MspInit.
    (+) MspDeInitCallback         : RNG MspDeInit.

    [..]
    For specific callback ReadyDataCallback, use dedicated register callbacks:
    respectively HAL_RNG_RegisterReadyDataCallback() , HAL_RNG_UnRegisterReadyDataCallback().

    [..]
    By default, after the HAL_RNG_Init() and when the state is HAL_RNG_STATE_RESET
    all callbacks are set to the corresponding weak (surcharged) functions:
    example HAL_RNG_ErrorCallback().
    Exception done for MspInit and MspDeInit functions that are respectively
    reset to the legacy weak (surcharged) functions in the HAL_RNG_Init()
    and HAL_RNG_DeInit() only when these callbacks are null (not registered beforehand).
    If not, MspInit or MspDeInit are not null, the HAL_RNG_Init() and HAL_RNG_DeInit()
    keep and use the user MspInit/MspDeInit callbacks (registered beforehand).

    [..]
    Callbacks can be registered/unregistered in HAL_RNG_STATE_READY state only.
    Exception done MspInit/MspDeInit that can be registered/unregistered
    in HAL_RNG_STATE_READY or HAL_RNG_STATE_RESET state, thus registered (user)
    MspInit/DeInit callbacks can be used during the Init/DeInit.
    In that case first register the MspInit/MspDeInit user callbacks
    using HAL_RNG_RegisterCallback() before calling HAL_RNG_DeInit()
    or HAL_RNG_Init() function.

    [..]
    When The compilation define USE_HAL_RNG_REGISTER_CALLBACKS is set to 0 or
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

#if defined (RNG)

/** @addtogroup RNG
  * @brief RNG HAL module driver.
  * @{
  */

#ifdef HAL_RNG_MODULE_ENABLED

/* Private types -------------------------------------------------------------*/
/* Private defines -----------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup RNG_Private_Constants RNG Private Constants
  * @{
  */
#define RNG_TIMEOUT_VALUE     2U
/**
  * @}
  */
/* Private macros ------------------------------------------------------------*/
/* Private functions prototypes ----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/* Exported functions --------------------------------------------------------*/

/** @addtogroup RNG_Exported_Functions
  * @{
  */

/** @addtogroup RNG_Exported_Functions_Group1
  *  @brief   Initialization and configuration functions
  *
@verbatim
 ===============================================================================
          ##### Initialization and configuration functions #####
 ===============================================================================
    [..]  This section provides functions allowing to:
      (+) Initialize the RNG according to the specified parameters
          in the RNG_InitTypeDef and create the associated handle
      (+) DeInitialize the RNG peripheral
      (+) Initialize the RNG MSP
      (+) DeInitialize RNG MSP

@endverbatim
  * @{
  */

/**
  * @brief  Initializes the RNG peripheral and creates the associated handle.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RNG_Init(RNG_HandleTypeDef *hrng)
{
  /* Check the RNG handle allocation */
  if (hrng == NULL)
  {
    return HAL_ERROR;
  }
  /* Check the parameters */
  assert_param(IS_RNG_ALL_INSTANCE(hrng->Instance));

#if (USE_HAL_RNG_REGISTER_CALLBACKS == 1)
  if (hrng->State == HAL_RNG_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hrng->Lock = HAL_UNLOCKED;

    hrng->ReadyDataCallback  = HAL_RNG_ReadyDataCallback;  /* Legacy weak ReadyDataCallback  */
    hrng->ErrorCallback      = HAL_RNG_ErrorCallback;      /* Legacy weak ErrorCallback      */

    if (hrng->MspInitCallback == NULL)
    {
      hrng->MspInitCallback = HAL_RNG_MspInit; /* Legacy weak MspInit  */
    }

    /* Init the low level hardware */
    hrng->MspInitCallback(hrng);
  }
#else
  if (hrng->State == HAL_RNG_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hrng->Lock = HAL_UNLOCKED;

    /* Init the low level hardware */
    HAL_RNG_MspInit(hrng);
  }
#endif /* USE_HAL_RNG_REGISTER_CALLBACKS */

  /* Change RNG peripheral state */
  hrng->State = HAL_RNG_STATE_BUSY;


  /* Enable the RNG Peripheral */
  __HAL_RNG_ENABLE(hrng);

  /* Initialize the RNG state */
  hrng->State = HAL_RNG_STATE_READY;

  /* Initialise the error code */
  hrng->ErrorCode = HAL_RNG_ERROR_NONE;

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  DeInitializes the RNG peripheral.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RNG_DeInit(RNG_HandleTypeDef *hrng)
{
  /* Check the RNG handle allocation */
  if (hrng == NULL)
  {
    return HAL_ERROR;
  }

  /* Disable the RNG Peripheral */
  CLEAR_BIT(hrng->Instance->CR, RNG_CR_IE | RNG_CR_RNGEN);

  /* Clear RNG interrupt status flags */
  CLEAR_BIT(hrng->Instance->SR, RNG_SR_CEIS | RNG_SR_SEIS);

#if (USE_HAL_RNG_REGISTER_CALLBACKS == 1)
  if (hrng->MspDeInitCallback == NULL)
  {
    hrng->MspDeInitCallback = HAL_RNG_MspDeInit; /* Legacy weak MspDeInit  */
  }

  /* DeInit the low level hardware */
  hrng->MspDeInitCallback(hrng);
#else
  /* DeInit the low level hardware */
  HAL_RNG_MspDeInit(hrng);
#endif /* USE_HAL_RNG_REGISTER_CALLBACKS */

  /* Update the RNG state */
  hrng->State = HAL_RNG_STATE_RESET;

  /* Initialise the error code */
  hrng->ErrorCode = HAL_RNG_ERROR_NONE;

  /* Release Lock */
  __HAL_UNLOCK(hrng);

  /* Return the function status */
  return HAL_OK;
}

/**
  * @brief  Initializes the RNG MSP.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval None
  */
__weak void HAL_RNG_MspInit(RNG_HandleTypeDef *hrng)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hrng);
  /* NOTE : This function should not be modified. When the callback is needed,
            function HAL_RNG_MspInit must be implemented in the user file.
   */
}

/**
  * @brief  DeInitializes the RNG MSP.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval None
  */
__weak void HAL_RNG_MspDeInit(RNG_HandleTypeDef *hrng)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hrng);
  /* NOTE : This function should not be modified. When the callback is needed,
            function HAL_RNG_MspDeInit must be implemented in the user file.
   */
}

#if (USE_HAL_RNG_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User RNG Callback
  *         To be used instead of the weak predefined callback
  * @param  hrng RNG handle
  * @param  CallbackID ID of the callback to be registered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_RNG_ERROR_CB_ID Error callback ID
  *          @arg @ref HAL_RNG_MSPINIT_CB_ID MspInit callback ID
  *          @arg @ref HAL_RNG_MSPDEINIT_CB_ID MspDeInit callback ID
  * @param  pCallback pointer to the Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RNG_RegisterCallback(RNG_HandleTypeDef *hrng, HAL_RNG_CallbackIDTypeDef CallbackID,
                                           pRNG_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hrng);

  if (HAL_RNG_STATE_READY == hrng->State)
  {
    switch (CallbackID)
    {
      case HAL_RNG_ERROR_CB_ID :
        hrng->ErrorCallback = pCallback;
        break;

      case HAL_RNG_MSPINIT_CB_ID :
        hrng->MspInitCallback = pCallback;
        break;

      case HAL_RNG_MSPDEINIT_CB_ID :
        hrng->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_RNG_STATE_RESET == hrng->State)
  {
    switch (CallbackID)
    {
      case HAL_RNG_MSPINIT_CB_ID :
        hrng->MspInitCallback = pCallback;
        break;

      case HAL_RNG_MSPDEINIT_CB_ID :
        hrng->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hrng);
  return status;
}

/**
  * @brief  Unregister an RNG Callback
  *         RNG callabck is redirected to the weak predefined callback
  * @param  hrng RNG handle
  * @param  CallbackID ID of the callback to be unregistered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_RNG_ERROR_CB_ID Error callback ID
  *          @arg @ref HAL_RNG_MSPINIT_CB_ID MspInit callback ID
  *          @arg @ref HAL_RNG_MSPDEINIT_CB_ID MspDeInit callback ID
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RNG_UnRegisterCallback(RNG_HandleTypeDef *hrng, HAL_RNG_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hrng);

  if (HAL_RNG_STATE_READY == hrng->State)
  {
    switch (CallbackID)
    {
      case HAL_RNG_ERROR_CB_ID :
        hrng->ErrorCallback = HAL_RNG_ErrorCallback;          /* Legacy weak ErrorCallback  */
        break;

      case HAL_RNG_MSPINIT_CB_ID :
        hrng->MspInitCallback = HAL_RNG_MspInit;              /* Legacy weak MspInit  */
        break;

      case HAL_RNG_MSPDEINIT_CB_ID :
        hrng->MspDeInitCallback = HAL_RNG_MspDeInit;          /* Legacy weak MspDeInit  */
        break;

      default :
        /* Update the error code */
        hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_RNG_STATE_RESET == hrng->State)
  {
    switch (CallbackID)
    {
      case HAL_RNG_MSPINIT_CB_ID :
        hrng->MspInitCallback = HAL_RNG_MspInit;              /* Legacy weak MspInit  */
        break;

      case HAL_RNG_MSPDEINIT_CB_ID :
        hrng->MspDeInitCallback = HAL_RNG_MspDeInit;          /* Legacy weak MspInit  */
        break;

      default :
        /* Update the error code */
        hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hrng);
  return status;
}

/**
  * @brief  Register Data Ready RNG Callback
  *         To be used instead of the weak HAL_RNG_ReadyDataCallback() predefined callback
  * @param  hrng RNG handle
  * @param  pCallback pointer to the Data Ready Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RNG_RegisterReadyDataCallback(RNG_HandleTypeDef *hrng, pRNG_ReadyDataCallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hrng);

  if (HAL_RNG_STATE_READY == hrng->State)
  {
    hrng->ReadyDataCallback = pCallback;
  }
  else
  {
    /* Update the error code */
    hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hrng);
  return status;
}

/**
  * @brief  UnRegister the Data Ready RNG Callback
  *         Data Ready RNG Callback is redirected to the weak HAL_RNG_ReadyDataCallback() predefined callback
  * @param  hrng RNG handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RNG_UnRegisterReadyDataCallback(RNG_HandleTypeDef *hrng)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hrng);

  if (HAL_RNG_STATE_READY == hrng->State)
  {
    hrng->ReadyDataCallback = HAL_RNG_ReadyDataCallback; /* Legacy weak ReadyDataCallback  */
  }
  else
  {
    /* Update the error code */
    hrng->ErrorCode = HAL_RNG_ERROR_INVALID_CALLBACK;
    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hrng);
  return status;
}

#endif /* USE_HAL_RNG_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @addtogroup RNG_Exported_Functions_Group2
  *  @brief   Peripheral Control functions
  *
@verbatim
 ===============================================================================
                      ##### Peripheral Control functions #####
 ===============================================================================
    [..]  This section provides functions allowing to:
      (+) Get the 32 bit Random number
      (+) Get the 32 bit Random number with interrupt enabled
      (+) Handle RNG interrupt request

@endverbatim
  * @{
  */

/**
  * @brief  Generates a 32-bit random number.
  * @note   Each time the random number data is read the RNG_FLAG_DRDY flag
  *         is automatically cleared.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @param  random32bit pointer to generated random number variable if successful.
  * @retval HAL status
  */

HAL_StatusTypeDef HAL_RNG_GenerateRandomNumber(RNG_HandleTypeDef *hrng, uint32_t *random32bit)
{
  uint32_t tickstart;
  HAL_StatusTypeDef status = HAL_OK;

  /* Process Locked */
  __HAL_LOCK(hrng);

  /* Check RNG peripheral state */
  if (hrng->State == HAL_RNG_STATE_READY)
  {
    /* Change RNG peripheral state */
    hrng->State = HAL_RNG_STATE_BUSY;

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Check if data register contains valid random data */
    while (__HAL_RNG_GET_FLAG(hrng, RNG_FLAG_DRDY) == RESET)
    {
      if ((HAL_GetTick() - tickstart) > RNG_TIMEOUT_VALUE)
      {
        /* New check to avoid false timeout detection in case of preemption */
        if (__HAL_RNG_GET_FLAG(hrng, RNG_FLAG_DRDY) == RESET)
        {
          hrng->State = HAL_RNG_STATE_READY;
          hrng->ErrorCode = HAL_RNG_ERROR_TIMEOUT;
          /* Process Unlocked */
          __HAL_UNLOCK(hrng);
          return HAL_ERROR;
        }
      }
    }

    /* Get a 32bit Random number */
    hrng->RandomNumber = hrng->Instance->DR;
    *random32bit = hrng->RandomNumber;

    hrng->State = HAL_RNG_STATE_READY;
  }
  else
  {
    hrng->ErrorCode = HAL_RNG_ERROR_BUSY;
    status = HAL_ERROR;
  }

  /* Process Unlocked */
  __HAL_UNLOCK(hrng);

  return status;
}

/**
  * @brief  Generates a 32-bit random number in interrupt mode.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_RNG_GenerateRandomNumber_IT(RNG_HandleTypeDef *hrng)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process Locked */
  __HAL_LOCK(hrng);

  /* Check RNG peripheral state */
  if (hrng->State == HAL_RNG_STATE_READY)
  {
    /* Change RNG peripheral state */
    hrng->State = HAL_RNG_STATE_BUSY;

    /* Enable the RNG Interrupts: Data Ready, Clock error, Seed error */
    __HAL_RNG_ENABLE_IT(hrng);
  }
  else
  {
    /* Process Unlocked */
    __HAL_UNLOCK(hrng);

    hrng->ErrorCode = HAL_RNG_ERROR_BUSY;
    status = HAL_ERROR;
  }

  return status;
}

/**
  * @brief  Returns generated random number in polling mode (Obsolete)
  *         Use HAL_RNG_GenerateRandomNumber() API instead.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval Random value
  */
uint32_t HAL_RNG_GetRandomNumber(RNG_HandleTypeDef *hrng)
{
  if (HAL_RNG_GenerateRandomNumber(hrng, &(hrng->RandomNumber)) == HAL_OK)
  {
    return hrng->RandomNumber;
  }
  else
  {
    return 0U;
  }
}

/**
  * @brief  Returns a 32-bit random number with interrupt enabled (Obsolete),
  *         Use HAL_RNG_GenerateRandomNumber_IT() API instead.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval 32-bit random number
  */
uint32_t HAL_RNG_GetRandomNumber_IT(RNG_HandleTypeDef *hrng)
{
  uint32_t random32bit = 0U;

  /* Process locked */
  __HAL_LOCK(hrng);

  /* Change RNG peripheral state */
  hrng->State = HAL_RNG_STATE_BUSY;

  /* Get a 32bit Random number */
  random32bit = hrng->Instance->DR;

  /* Enable the RNG Interrupts: Data Ready, Clock error, Seed error */
  __HAL_RNG_ENABLE_IT(hrng);

  /* Return the 32 bit random number */
  return random32bit;
}

/**
  * @brief  Handles RNG interrupt request.
  * @note   In the case of a clock error, the RNG is no more able to generate
  *         random numbers because the PLL48CLK clock is not correct. User has
  *         to check that the clock controller is correctly configured to provide
  *         the RNG clock and clear the CEIS bit using __HAL_RNG_CLEAR_IT().
  *         The clock error has no impact on the previously generated
  *         random numbers, and the RNG_DR register contents can be used.
  * @note   In the case of a seed error, the generation of random numbers is
  *         interrupted as long as the SECS bit is '1'. If a number is
  *         available in the RNG_DR register, it must not be used because it may
  *         not have enough entropy. In this case, it is recommended to clear the
  *         SEIS bit using __HAL_RNG_CLEAR_IT(), then disable and enable
  *         the RNG peripheral to reinitialize and restart the RNG.
  * @note   User-written HAL_RNG_ErrorCallback() API is called once whether SEIS
  *         or CEIS are set.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval None

  */
void HAL_RNG_IRQHandler(RNG_HandleTypeDef *hrng)
{
  uint32_t rngclockerror = 0U;

  /* RNG clock error interrupt occurred */
  if (__HAL_RNG_GET_IT(hrng, RNG_IT_CEI) != RESET)
  {
    /* Update the error code */
    hrng->ErrorCode = HAL_RNG_ERROR_CLOCK;
    rngclockerror = 1U;
  }
  else if (__HAL_RNG_GET_IT(hrng, RNG_IT_SEI) != RESET)
  {
    /* Update the error code */
    hrng->ErrorCode = HAL_RNG_ERROR_SEED;
    rngclockerror = 1U;
  }
  else
  {
    /* Nothing to do */
  }

  if (rngclockerror == 1U)
  {
    /* Change RNG peripheral state */
    hrng->State = HAL_RNG_STATE_ERROR;

#if (USE_HAL_RNG_REGISTER_CALLBACKS == 1)
    /* Call registered Error callback */
    hrng->ErrorCallback(hrng);
#else
    /* Call legacy weak Error callback */
    HAL_RNG_ErrorCallback(hrng);
#endif /* USE_HAL_RNG_REGISTER_CALLBACKS */

    /* Clear the clock error flag */
    __HAL_RNG_CLEAR_IT(hrng, RNG_IT_CEI | RNG_IT_SEI);

    return;
  }

  /* Check RNG data ready interrupt occurred */
  if (__HAL_RNG_GET_IT(hrng, RNG_IT_DRDY) != RESET)
  {
    /* Generate random number once, so disable the IT */
    __HAL_RNG_DISABLE_IT(hrng);

    /* Get the 32bit Random number (DRDY flag automatically cleared) */
    hrng->RandomNumber = hrng->Instance->DR;

    if (hrng->State != HAL_RNG_STATE_ERROR)
    {
      /* Change RNG peripheral state */
      hrng->State = HAL_RNG_STATE_READY;
      /* Process Unlocked */
      __HAL_UNLOCK(hrng);

#if (USE_HAL_RNG_REGISTER_CALLBACKS == 1)
      /* Call registered Data Ready callback */
      hrng->ReadyDataCallback(hrng, hrng->RandomNumber);
#else
      /* Call legacy weak Data Ready callback */
      HAL_RNG_ReadyDataCallback(hrng, hrng->RandomNumber);
#endif /* USE_HAL_RNG_REGISTER_CALLBACKS */
    }
  }
}

/**
  * @brief  Read latest generated random number.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval random value
  */
uint32_t HAL_RNG_ReadLastRandomNumber(RNG_HandleTypeDef *hrng)
{
  return (hrng->RandomNumber);
}

/**
  * @brief  Data Ready callback in non-blocking mode.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @param  random32bit generated random number.
  * @retval None
  */
__weak void HAL_RNG_ReadyDataCallback(RNG_HandleTypeDef *hrng, uint32_t random32bit)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hrng);
  UNUSED(random32bit);
  /* NOTE : This function should not be modified. When the callback is needed,
            function HAL_RNG_ReadyDataCallback must be implemented in the user file.
   */
}

/**
  * @brief  RNG error callbacks.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval None
  */
__weak void HAL_RNG_ErrorCallback(RNG_HandleTypeDef *hrng)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hrng);
  /* NOTE : This function should not be modified. When the callback is needed,
            function HAL_RNG_ErrorCallback must be implemented in the user file.
   */
}
/**
  * @}
  */


/** @addtogroup RNG_Exported_Functions_Group3
  *  @brief   Peripheral State functions
  *
@verbatim
 ===============================================================================
                      ##### Peripheral State functions #####
 ===============================================================================
    [..]
    This subsection permits to get in run-time the status of the peripheral
    and the data flow.

@endverbatim
  * @{
  */

/**
  * @brief  Returns the RNG state.
  * @param  hrng pointer to a RNG_HandleTypeDef structure that contains
  *                the configuration information for RNG.
  * @retval HAL state
  */
HAL_RNG_StateTypeDef HAL_RNG_GetState(RNG_HandleTypeDef *hrng)
{
  return hrng->State;
}

/**
  * @brief  Return the RNG handle error code.
  * @param  hrng: pointer to a RNG_HandleTypeDef structure.
  * @retval RNG Error Code
  */
uint32_t HAL_RNG_GetError(RNG_HandleTypeDef *hrng)
{
  /* Return RNG Error Code */
  return hrng->ErrorCode;
}
/**
  * @}
  */

/**
  * @}
  */


#endif /* HAL_RNG_MODULE_ENABLED */
/**
  * @}
  */

#endif /* RNG */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
