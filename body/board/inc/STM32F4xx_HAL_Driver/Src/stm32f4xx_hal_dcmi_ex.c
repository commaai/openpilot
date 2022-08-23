/**
  ******************************************************************************
  * @file    stm32f4xx_hal_dcmi_ex.c
  * @author  MCD Application Team
  * @brief   DCMI Extension HAL module driver
  *          This file provides firmware functions to manage the following
  *          functionalities of DCMI extension peripheral:
  *           + Extension features functions
  *
  @verbatim
  ==============================================================================
               ##### DCMI peripheral extension features  #####
  ==============================================================================

  [..] Comparing to other previous devices, the DCMI interface for STM32F446xx
       devices contains the following additional features :

       (+) Support of Black and White cameras

                     ##### How to use this driver #####
  ==============================================================================
  [..] This driver provides functions to manage the Black and White feature

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
/** @defgroup DCMIEx DCMIEx
  * @brief DCMI Extended HAL module driver
  * @{
  */

#ifdef HAL_DCMI_MODULE_ENABLED

#if defined(STM32F407xx) || defined(STM32F417xx) || defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) ||\
    defined(STM32F439xx) || defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Exported functions --------------------------------------------------------*/

/** @defgroup DCMIEx_Exported_Functions DCMI Extended Exported Functions
  * @{
  */

/**
  * @}
  */

/** @addtogroup DCMI_Exported_Functions_Group1 Initialization and Configuration functions
  * @{
  */

/**
  * @brief  Initializes the DCMI according to the specified
  *         parameters in the DCMI_InitTypeDef and create the associated handle.
  * @param  hdcmi pointer to a DCMI_HandleTypeDef structure that contains
  *                the configuration information for DCMI.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_DCMI_Init(DCMI_HandleTypeDef *hdcmi)
{
  /* Check the DCMI peripheral state */
  if(hdcmi == NULL)
  {
     return HAL_ERROR;
  }

  /* Check function parameters */
  assert_param(IS_DCMI_ALL_INSTANCE(hdcmi->Instance));
  assert_param(IS_DCMI_PCKPOLARITY(hdcmi->Init.PCKPolarity));
  assert_param(IS_DCMI_VSPOLARITY(hdcmi->Init.VSPolarity));
  assert_param(IS_DCMI_HSPOLARITY(hdcmi->Init.HSPolarity));
  assert_param(IS_DCMI_SYNCHRO(hdcmi->Init.SynchroMode));
  assert_param(IS_DCMI_CAPTURE_RATE(hdcmi->Init.CaptureRate));
  assert_param(IS_DCMI_EXTENDED_DATA(hdcmi->Init.ExtendedDataMode));
  assert_param(IS_DCMI_MODE_JPEG(hdcmi->Init.JPEGMode));
#if defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
  assert_param(IS_DCMI_BYTE_SELECT_MODE(hdcmi->Init.ByteSelectMode));
  assert_param(IS_DCMI_BYTE_SELECT_START(hdcmi->Init.ByteSelectStart));
  assert_param(IS_DCMI_LINE_SELECT_MODE(hdcmi->Init.LineSelectMode));
  assert_param(IS_DCMI_LINE_SELECT_START(hdcmi->Init.LineSelectStart));
#endif /* STM32F446xx || STM32F469xx || STM32F479xx */
  if(hdcmi->State == HAL_DCMI_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hdcmi->Lock = HAL_UNLOCKED;
    /* Init the low level hardware */
  /* Init the DCMI Callback settings */
#if (USE_HAL_DCMI_REGISTER_CALLBACKS == 1)
    hdcmi->FrameEventCallback = HAL_DCMI_FrameEventCallback; /* Legacy weak FrameEventCallback  */
    hdcmi->VsyncEventCallback = HAL_DCMI_VsyncEventCallback; /* Legacy weak VsyncEventCallback  */
    hdcmi->LineEventCallback  = HAL_DCMI_LineEventCallback;  /* Legacy weak LineEventCallback   */
    hdcmi->ErrorCallback      = HAL_DCMI_ErrorCallback;      /* Legacy weak ErrorCallback       */

    if(hdcmi->MspInitCallback == NULL)
    {
      /* Legacy weak MspInit Callback        */
      hdcmi->MspInitCallback = HAL_DCMI_MspInit;
    }
    /* Initialize the low level hardware (MSP) */
    hdcmi->MspInitCallback(hdcmi);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC and DMA */
    HAL_DCMI_MspInit(hdcmi);
#endif /* (USE_HAL_DCMI_REGISTER_CALLBACKS) */
    HAL_DCMI_MspInit(hdcmi);
  }

  /* Change the DCMI state */
  hdcmi->State = HAL_DCMI_STATE_BUSY;
                          /* Configures the HS, VS, DE and PC polarity */
  hdcmi->Instance->CR &= ~(DCMI_CR_PCKPOL | DCMI_CR_HSPOL  | DCMI_CR_VSPOL  | DCMI_CR_EDM_0 |\
                           DCMI_CR_EDM_1  | DCMI_CR_FCRC_0 | DCMI_CR_FCRC_1 | DCMI_CR_JPEG  |\
                           DCMI_CR_ESS
#if defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
                           | DCMI_CR_BSM_0 | DCMI_CR_BSM_1 | DCMI_CR_OEBS |\
                           DCMI_CR_LSM | DCMI_CR_OELS
#endif /* STM32F446xx || STM32F469xx || STM32F479xx */
                           );
  hdcmi->Instance->CR |=  (uint32_t)(hdcmi->Init.SynchroMode | hdcmi->Init.CaptureRate |\
                                     hdcmi->Init.VSPolarity  | hdcmi->Init.HSPolarity  |\
                                     hdcmi->Init.PCKPolarity | hdcmi->Init.ExtendedDataMode |\
                                     hdcmi->Init.JPEGMode
#if defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
                                     | hdcmi->Init.ByteSelectMode |\
                                     hdcmi->Init.ByteSelectStart | hdcmi->Init.LineSelectMode |\
                                     hdcmi->Init.LineSelectStart
#endif /* STM32F446xx || STM32F469xx || STM32F479xx */
                                     );
  if(hdcmi->Init.SynchroMode == DCMI_SYNCHRO_EMBEDDED)
  {
    hdcmi->Instance->ESCR = (((uint32_t)hdcmi->Init.SyncroCode.FrameStartCode)    |
                             ((uint32_t)hdcmi->Init.SyncroCode.LineStartCode << DCMI_POSITION_ESCR_LSC)|
                             ((uint32_t)hdcmi->Init.SyncroCode.LineEndCode << DCMI_POSITION_ESCR_LEC) |
                             ((uint32_t)hdcmi->Init.SyncroCode.FrameEndCode << DCMI_POSITION_ESCR_FEC));

  }

  /* Enable the Line, Vsync, Error and Overrun interrupts */
  __HAL_DCMI_ENABLE_IT(hdcmi, DCMI_IT_LINE | DCMI_IT_VSYNC | DCMI_IT_ERR | DCMI_IT_OVR);

  /* Update error code */
  hdcmi->ErrorCode = HAL_DCMI_ERROR_NONE;

  /* Initialize the DCMI state*/
  hdcmi->State  = HAL_DCMI_STATE_READY;

  return HAL_OK;
}

/**
  * @}
  */
#endif /* STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx || STM32F429xx ||\
          STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */
#endif /* HAL_DCMI_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
