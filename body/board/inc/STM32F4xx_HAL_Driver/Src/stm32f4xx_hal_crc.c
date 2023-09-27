/**
  ******************************************************************************
  * @file    stm32f4xx_hal_crc.c
  * @author  MCD Application Team
  * @brief   CRC HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the Cyclic Redundancy Check (CRC) peripheral:
  *           + Initialization and de-initialization functions
  *           + Peripheral Control functions
  *           + Peripheral State functions
  *
  @verbatim
 ===============================================================================
                     ##### How to use this driver #####
 ===============================================================================
    [..]
         (+) Enable CRC AHB clock using __HAL_RCC_CRC_CLK_ENABLE();
         (+) Initialize CRC calculator
             (++) specify generating polynomial (peripheral default or non-default one)
             (++) specify initialization value (peripheral default or non-default one)
             (++) specify input data format
             (++) specify input or output data inversion mode if any
         (+) Use HAL_CRC_Accumulate() function to compute the CRC value of the
             input data buffer starting with the previously computed CRC as
             initialization value
         (+) Use HAL_CRC_Calculate() function to compute the CRC value of the
             input data buffer starting with the defined initialization value
             (default or non-default) to initiate CRC calculation

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

/** @defgroup CRC CRC
  * @brief CRC HAL module driver.
  * @{
  */

#ifdef HAL_CRC_MODULE_ENABLED

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/

/** @defgroup CRC_Exported_Functions CRC Exported Functions
  * @{
  */

/** @defgroup CRC_Exported_Functions_Group1 Initialization and de-initialization functions
  *  @brief    Initialization and Configuration functions.
  *
@verbatim
 ===============================================================================
            ##### Initialization and de-initialization functions #####
 ===============================================================================
    [..]  This section provides functions allowing to:
      (+) Initialize the CRC according to the specified parameters
          in the CRC_InitTypeDef and create the associated handle
      (+) DeInitialize the CRC peripheral
      (+) Initialize the CRC MSP (MCU Specific Package)
      (+) DeInitialize the CRC MSP

@endverbatim
  * @{
  */

/**
  * @brief  Initialize the CRC according to the specified
  *         parameters in the CRC_InitTypeDef and create the associated handle.
  * @param  hcrc CRC handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CRC_Init(CRC_HandleTypeDef *hcrc)
{
  /* Check the CRC handle allocation */
  if (hcrc == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_CRC_ALL_INSTANCE(hcrc->Instance));

  if (hcrc->State == HAL_CRC_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hcrc->Lock = HAL_UNLOCKED;
    /* Init the low level hardware */
    HAL_CRC_MspInit(hcrc);
  }

  /* Change CRC peripheral state */
  hcrc->State = HAL_CRC_STATE_READY;

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  DeInitialize the CRC peripheral.
  * @param  hcrc CRC handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CRC_DeInit(CRC_HandleTypeDef *hcrc)
{
  /* Check the CRC handle allocation */
  if (hcrc == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_CRC_ALL_INSTANCE(hcrc->Instance));

  /* Check the CRC peripheral state */
  if (hcrc->State == HAL_CRC_STATE_BUSY)
  {
    return HAL_BUSY;
  }

  /* Change CRC peripheral state */
  hcrc->State = HAL_CRC_STATE_BUSY;

  /* Reset CRC calculation unit */
  __HAL_CRC_DR_RESET(hcrc);

  /* Reset IDR register content */
  CLEAR_BIT(hcrc->Instance->IDR, CRC_IDR_IDR);

  /* DeInit the low level hardware */
  HAL_CRC_MspDeInit(hcrc);

  /* Change CRC peripheral state */
  hcrc->State = HAL_CRC_STATE_RESET;

  /* Process unlocked */
  __HAL_UNLOCK(hcrc);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Initializes the CRC MSP.
  * @param  hcrc CRC handle
  * @retval None
  */
__weak void HAL_CRC_MspInit(CRC_HandleTypeDef *hcrc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcrc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_CRC_MspInit can be implemented in the user file
   */
}

/**
  * @brief  DeInitialize the CRC MSP.
  * @param  hcrc CRC handle
  * @retval None
  */
__weak void HAL_CRC_MspDeInit(CRC_HandleTypeDef *hcrc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcrc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_CRC_MspDeInit can be implemented in the user file
   */
}

/**
  * @}
  */

/** @defgroup CRC_Exported_Functions_Group2 Peripheral Control functions
  *  @brief    management functions.
  *
@verbatim
 ===============================================================================
                      ##### Peripheral Control functions #####
 ===============================================================================
    [..]  This section provides functions allowing to:
      (+) compute the 32-bit CRC value of a 32-bit data buffer
          using combination of the previous CRC value and the new one.

       [..]  or

      (+) compute the 32-bit CRC value of a 32-bit data buffer
          independently of the previous CRC value.

@endverbatim
  * @{
  */

/**
  * @brief  Compute the 32-bit CRC value of a 32-bit data buffer
  *         starting with the previously computed CRC as initialization value.
  * @param  hcrc CRC handle
  * @param  pBuffer pointer to the input data buffer.
  * @param  BufferLength input data buffer length (number of uint32_t words).
  * @retval uint32_t CRC (returned value LSBs for CRC shorter than 32 bits)
  */
uint32_t HAL_CRC_Accumulate(CRC_HandleTypeDef *hcrc, uint32_t pBuffer[], uint32_t BufferLength)
{
  uint32_t index;      /* CRC input data buffer index */
  uint32_t temp = 0U;  /* CRC output (read from hcrc->Instance->DR register) */

  /* Change CRC peripheral state */
  hcrc->State = HAL_CRC_STATE_BUSY;

  /* Enter Data to the CRC calculator */
  for (index = 0U; index < BufferLength; index++)
  {
    hcrc->Instance->DR = pBuffer[index];
  }
  temp = hcrc->Instance->DR;

  /* Change CRC peripheral state */
  hcrc->State = HAL_CRC_STATE_READY;

  /* Return the CRC computed value */
  return temp;
}

/**
  * @brief  Compute the 32-bit CRC value of a 32-bit data buffer
  *         starting with hcrc->Instance->INIT as initialization value.
  * @param  hcrc CRC handle
  * @param  pBuffer pointer to the input data buffer.
  * @param  BufferLength input data buffer length (number of uint32_t words).
  * @retval uint32_t CRC (returned value LSBs for CRC shorter than 32 bits)
  */
uint32_t HAL_CRC_Calculate(CRC_HandleTypeDef *hcrc, uint32_t pBuffer[], uint32_t BufferLength)
{
  uint32_t index;      /* CRC input data buffer index */
  uint32_t temp = 0U;  /* CRC output (read from hcrc->Instance->DR register) */

  /* Change CRC peripheral state */
  hcrc->State = HAL_CRC_STATE_BUSY;

  /* Reset CRC Calculation Unit (hcrc->Instance->INIT is
  *  written in hcrc->Instance->DR) */
  __HAL_CRC_DR_RESET(hcrc);

  /* Enter 32-bit input data to the CRC calculator */
  for (index = 0U; index < BufferLength; index++)
  {
    hcrc->Instance->DR = pBuffer[index];
  }
  temp = hcrc->Instance->DR;

  /* Change CRC peripheral state */
  hcrc->State = HAL_CRC_STATE_READY;

  /* Return the CRC computed value */
  return temp;
}

/**
  * @}
  */

/** @defgroup CRC_Exported_Functions_Group3 Peripheral State functions
  *  @brief    Peripheral State functions.
  *
@verbatim
 ===============================================================================
                      ##### Peripheral State functions #####
 ===============================================================================
    [..]
    This subsection permits to get in run-time the status of the peripheral.

@endverbatim
  * @{
  */

/**
  * @brief  Return the CRC handle state.
  * @param  hcrc CRC handle
  * @retval HAL state
  */
HAL_CRC_StateTypeDef HAL_CRC_GetState(CRC_HandleTypeDef *hcrc)
{
  /* Return CRC handle state */
  return hcrc->State;
}

/**
  * @}
  */

/**
  * @}
  */


#endif /* HAL_CRC_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
