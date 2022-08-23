/**
  ******************************************************************************
  * @file    stm32f4xx_hal_flash_ramfunc.c
  * @author  MCD Application Team
  * @brief   FLASH RAMFUNC module driver.
  *          This file provides a FLASH firmware functions which should be 
  *          executed from internal SRAM
  *            + Stop/Start the flash interface while System Run
  *            + Enable/Disable the flash sleep while System Run
  @verbatim
  ==============================================================================
                    ##### APIs executed from Internal RAM #####
  ==============================================================================
  [..]
    *** ARM Compiler ***
    --------------------
    [..] RAM functions are defined using the toolchain options. 
         Functions that are be executed in RAM should reside in a separate
         source module. Using the 'Options for File' dialog you can simply change
         the 'Code / Const' area of a module to a memory space in physical RAM.
         Available memory areas are declared in the 'Target' tab of the 
         Options for Target' dialog.

    *** ICCARM Compiler ***
    -----------------------
    [..] RAM functions are defined using a specific toolchain keyword "__ramfunc".

    *** GNU Compiler ***
    --------------------
    [..] RAM functions are defined using a specific toolchain attribute
         "__attribute__((section(".RamFunc")))".
  
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

/** @defgroup FLASH_RAMFUNC FLASH RAMFUNC
  * @brief FLASH functions executed from RAM
  * @{
  */
#ifdef HAL_FLASH_MODULE_ENABLED
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || \
    defined(STM32F412Rx) || defined(STM32F412Cx)

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Exported functions --------------------------------------------------------*/
/** @defgroup FLASH_RAMFUNC_Exported_Functions FLASH RAMFUNC Exported Functions
  * @{
  */

/** @defgroup FLASH_RAMFUNC_Exported_Functions_Group1 Peripheral features functions executed from internal RAM 
  *  @brief Peripheral Extended features functions 
  *
@verbatim   

 ===============================================================================
                      ##### ramfunc functions #####
 ===============================================================================  
    [..]
    This subsection provides a set of functions that should be executed from RAM 
    transfers.
    
@endverbatim
  * @{
  */

/**
  * @brief Stop the flash interface while System Run
  * @note  This mode is only available for STM32F41xxx/STM32F446xx devices. 
  * @note  This mode couldn't be set while executing with the flash itself. 
  *        It should be done with specific routine executed from RAM.     
  * @retval HAL status
  */
__RAM_FUNC HAL_StatusTypeDef HAL_FLASHEx_StopFlashInterfaceClk(void)
{
  /* Enable Power ctrl clock */
  __HAL_RCC_PWR_CLK_ENABLE();
  /* Stop the flash interface while System Run */  
  SET_BIT(PWR->CR, PWR_CR_FISSR);
   
  return HAL_OK;
}

/**
  * @brief Start the flash interface while System Run
  * @note  This mode is only available for STM32F411xx/STM32F446xx devices. 
  * @note  This mode couldn't be set while executing with the flash itself. 
  *        It should be done with specific routine executed from RAM.     
  * @retval HAL status
  */
__RAM_FUNC HAL_StatusTypeDef HAL_FLASHEx_StartFlashInterfaceClk(void)
{
  /* Enable Power ctrl clock */
  __HAL_RCC_PWR_CLK_ENABLE();
  /* Start the flash interface while System Run */
  CLEAR_BIT(PWR->CR, PWR_CR_FISSR);

  return HAL_OK;
}

/**
  * @brief Enable the flash sleep while System Run
  * @note  This mode is only available for STM32F41xxx/STM32F446xx devices. 
  * @note  This mode could n't be set while executing with the flash itself. 
  *        It should be done with specific routine executed from RAM.     
  * @retval HAL status
  */
__RAM_FUNC HAL_StatusTypeDef HAL_FLASHEx_EnableFlashSleepMode(void)
{
  /* Enable Power ctrl clock */
  __HAL_RCC_PWR_CLK_ENABLE();
  /* Enable the flash sleep while System Run */
  SET_BIT(PWR->CR, PWR_CR_FMSSR);

  return HAL_OK;
}

/**
  * @brief Disable the flash sleep while System Run
  * @note  This mode is only available for STM32F41xxx/STM32F446xx devices. 
  * @note  This mode couldn't be set while executing with the flash itself. 
  *        It should be done with specific routine executed from RAM.     
  * @retval HAL status
  */
__RAM_FUNC HAL_StatusTypeDef HAL_FLASHEx_DisableFlashSleepMode(void)
{
  /* Enable Power ctrl clock */
  __HAL_RCC_PWR_CLK_ENABLE();
  /* Disable the flash sleep while System Run */
  CLEAR_BIT(PWR->CR, PWR_CR_FMSSR);
  
  return HAL_OK;
}

/**
  * @}
  */

/**
  * @}
  */

#endif /* STM32F410xx || STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */
#endif /* HAL_FLASH_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
