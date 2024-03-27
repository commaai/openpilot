/**
  ******************************************************************************
  * @file    stm32f4xx_hal_fmpi2c_ex.c
  * @author  MCD Application Team
  * @brief   FMPI2C Extended HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of FMPI2C Extended peripheral:
  *           + Filter Mode Functions
  *           + FastModePlus Functions
  *
  @verbatim
  ==============================================================================
               ##### FMPI2C peripheral Extended features  #####
  ==============================================================================

  [..] Comparing to other previous devices, the FMPI2C interface for STM32F4xx
       devices contains the following additional features

       (+) Possibility to disable or enable Analog Noise Filter
       (+) Use of a configured Digital Noise Filter
       (+) Disable or enable Fast Mode Plus

                     ##### How to use this driver #####
  ==============================================================================
  [..] This driver provides functions to:
    (#) Configure FMPI2C Analog noise filter using the function HAL_FMPI2CEx_ConfigAnalogFilter()
    (#) Configure FMPI2C Digital noise filter using the function HAL_FMPI2CEx_ConfigDigitalFilter()
    (#) Configure the enable or disable of fast mode plus driving capability using the functions :
          (++) HAL_FMPI2CEx_EnableFastModePlus()
          (++) HAL_FMPI2CEx_DisableFastModePlus()
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

/** @defgroup FMPI2CEx FMPI2CEx
  * @brief FMPI2C Extended HAL module driver
  * @{
  */

#ifdef HAL_FMPI2C_MODULE_ENABLED
#if defined(FMPI2C_CR1_PE)

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/** @defgroup FMPI2CEx_Exported_Functions FMPI2C Extended Exported Functions
  * @{
  */

/** @defgroup FMPI2CEx_Exported_Functions_Group1 Filter Mode Functions
  * @brief    Filter Mode Functions
  *
@verbatim
 ===============================================================================
                      ##### Filter Mode Functions #####
 ===============================================================================
    [..] This section provides functions allowing to:
      (+) Configure Noise Filters

@endverbatim
  * @{
  */

/**
  * @brief  Configure FMPI2C Analog noise filter.
  * @param  hfmpi2c Pointer to a FMPI2C_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPI2Cx peripheral.
  * @param  AnalogFilter New state of the Analog filter.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPI2CEx_ConfigAnalogFilter(FMPI2C_HandleTypeDef *hfmpi2c, uint32_t AnalogFilter)
{
  /* Check the parameters */
  assert_param(IS_FMPI2C_ALL_INSTANCE(hfmpi2c->Instance));
  assert_param(IS_FMPI2C_ANALOG_FILTER(AnalogFilter));

  if (hfmpi2c->State == HAL_FMPI2C_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hfmpi2c);

    hfmpi2c->State = HAL_FMPI2C_STATE_BUSY;

    /* Disable the selected FMPI2C peripheral */
    __HAL_FMPI2C_DISABLE(hfmpi2c);

    /* Reset FMPI2Cx ANOFF bit */
    hfmpi2c->Instance->CR1 &= ~(FMPI2C_CR1_ANFOFF);

    /* Set analog filter bit*/
    hfmpi2c->Instance->CR1 |= AnalogFilter;

    __HAL_FMPI2C_ENABLE(hfmpi2c);

    hfmpi2c->State = HAL_FMPI2C_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpi2c);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Configure FMPI2C Digital noise filter.
  * @param  hfmpi2c Pointer to a FMPI2C_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPI2Cx peripheral.
  * @param  DigitalFilter Coefficient of digital noise filter between Min_Data=0x00 and Max_Data=0x0F.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPI2CEx_ConfigDigitalFilter(FMPI2C_HandleTypeDef *hfmpi2c, uint32_t DigitalFilter)
{
  uint32_t tmpreg;

  /* Check the parameters */
  assert_param(IS_FMPI2C_ALL_INSTANCE(hfmpi2c->Instance));
  assert_param(IS_FMPI2C_DIGITAL_FILTER(DigitalFilter));

  if (hfmpi2c->State == HAL_FMPI2C_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hfmpi2c);

    hfmpi2c->State = HAL_FMPI2C_STATE_BUSY;

    /* Disable the selected FMPI2C peripheral */
    __HAL_FMPI2C_DISABLE(hfmpi2c);

    /* Get the old register value */
    tmpreg = hfmpi2c->Instance->CR1;

    /* Reset FMPI2Cx DNF bits [11:8] */
    tmpreg &= ~(FMPI2C_CR1_DNF);

    /* Set FMPI2Cx DNF coefficient */
    tmpreg |= DigitalFilter << 8U;

    /* Store the new register value */
    hfmpi2c->Instance->CR1 = tmpreg;

    __HAL_FMPI2C_ENABLE(hfmpi2c);

    hfmpi2c->State = HAL_FMPI2C_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpi2c);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}
/**
  * @}
  */

/** @defgroup FMPI2CEx_Exported_Functions_Group3 Fast Mode Plus Functions
  * @brief    Fast Mode Plus Functions
  *
@verbatim
 ===============================================================================
                      ##### Fast Mode Plus Functions #####
 ===============================================================================
    [..] This section provides functions allowing to:
      (+) Configure Fast Mode Plus

@endverbatim
  * @{
  */

/**
  * @brief Enable the FMPI2C fast mode plus driving capability.
  * @param ConfigFastModePlus Selects the pin.
  *   This parameter can be one of the @ref FMPI2CEx_FastModePlus values
  * @note  For FMPI2C1, fast mode plus driving capability can be enabled on all selected
  *        FMPI2C1 pins using FMPI2C_FASTMODEPLUS_FMPI2C1 parameter or independently
  *        on each one of the following pins PB6, PB7, PB8 and PB9.
  * @note  For remaining FMPI2C1 pins (PA14, PA15...) fast mode plus driving capability
  *        can be enabled only by using FMPI2C_FASTMODEPLUS_FMPI2C1 parameter.
  * @retval None
  */
void HAL_FMPI2CEx_EnableFastModePlus(uint32_t ConfigFastModePlus)
{
  /* Check the parameter */
  assert_param(IS_FMPI2C_FASTMODEPLUS(ConfigFastModePlus));

  /* Enable SYSCFG clock */
  __HAL_RCC_SYSCFG_CLK_ENABLE();

  /* Enable fast mode plus driving capability for selected pin */
  SET_BIT(SYSCFG->CFGR, (uint32_t)ConfigFastModePlus);
}

/**
  * @brief Disable the FMPI2C fast mode plus driving capability.
  * @param ConfigFastModePlus Selects the pin.
  *   This parameter can be one of the @ref FMPI2CEx_FastModePlus values
  * @note  For FMPI2C1, fast mode plus driving capability can be disabled on all selected
  *        FMPI2C1 pins using FMPI2C_FASTMODEPLUS_FMPI2C1 parameter or independently
  *        on each one of the following pins PB6, PB7, PB8 and PB9.
  * @note  For remaining FMPI2C1 pins (PA14, PA15...) fast mode plus driving capability
  *        can be disabled only by using FMPI2C_FASTMODEPLUS_FMPI2C1 parameter.
  * @retval None
  */
void HAL_FMPI2CEx_DisableFastModePlus(uint32_t ConfigFastModePlus)
{
  /* Check the parameter */
  assert_param(IS_FMPI2C_FASTMODEPLUS(ConfigFastModePlus));

  /* Enable SYSCFG clock */
  __HAL_RCC_SYSCFG_CLK_ENABLE();

  /* Disable fast mode plus driving capability for selected pin */
  CLEAR_BIT(SYSCFG->CFGR, (uint32_t)ConfigFastModePlus);
}
/**
  * @}
  */
/**
  * @}
  */

#endif /* FMPI2C_CR1_PE */
#endif /* HAL_FMPI2C_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
