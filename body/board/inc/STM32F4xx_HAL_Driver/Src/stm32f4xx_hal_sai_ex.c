/**
  ******************************************************************************
  * @file    stm32f4xx_hal_sai_ex.c
  * @author  MCD Application Team
  * @brief   SAI Extension HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of SAI extension peripheral:
  *           + Extension features functions
  *
  @verbatim
  ==============================================================================
               ##### SAI peripheral extension features  #####
  ==============================================================================

  [..] Comparing to other previous devices, the SAI interface for STM32F446xx
       devices contains the following additional features :

       (+) Possibility to be clocked from PLLR

                     ##### How to use this driver #####
  ==============================================================================
  [..] This driver provides functions to manage several sources to clock SAI

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

/** @defgroup SAIEx SAIEx
  * @brief SAI Extension HAL module driver
  * @{
  */

#ifdef HAL_SAI_MODULE_ENABLED

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F413xx) || \
    defined(STM32F423xx)

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* SAI registers Masks */
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/** @defgroup SAI_Private_Functions  SAI Private Functions
  * @{
  */
/**
 * @}
 */

/* Exported functions --------------------------------------------------------*/
/** @defgroup SAIEx_Exported_Functions SAI Extended Exported Functions
  * @{
  */

/** @defgroup SAIEx_Exported_Functions_Group1 Extension features functions
  *  @brief   Extension features functions
  *
@verbatim
 ===============================================================================
                       ##### Extension features Functions #####
 ===============================================================================
    [..]
    This subsection provides a set of functions allowing to manage the possible
    SAI clock sources.

@endverbatim
  * @{
  */

/**
  * @brief  Configure SAI Block synchronization mode
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval SAI Clock Input
  */
void SAI_BlockSynchroConfig(SAI_HandleTypeDef *hsai)
{
  uint32_t tmpregisterGCR;

#if defined(STM32F446xx)
  /* This setting must be done with both audio block (A & B) disabled         */
  switch (hsai->Init.SynchroExt)
  {
    case SAI_SYNCEXT_DISABLE :
      tmpregisterGCR = 0U;
      break;
    case SAI_SYNCEXT_OUTBLOCKA_ENABLE :
      tmpregisterGCR = SAI_GCR_SYNCOUT_0;
      break;
    case SAI_SYNCEXT_OUTBLOCKB_ENABLE :
      tmpregisterGCR = SAI_GCR_SYNCOUT_1;
      break;
    default:
      tmpregisterGCR = 0U;
      break;
  }

  if ((hsai->Init.Synchro) == SAI_SYNCHRONOUS_EXT_SAI2)
  {
    tmpregisterGCR |= SAI_GCR_SYNCIN_0;
  }

  if ((hsai->Instance == SAI1_Block_A) || (hsai->Instance == SAI1_Block_B))
  {
    SAI1->GCR = tmpregisterGCR;
  }
  else
  {
    SAI2->GCR = tmpregisterGCR;
  }
#endif /* STM32F446xx */
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
    defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F413xx) || defined(STM32F423xx)
  /* This setting must be done with both audio block (A & B) disabled         */
  switch (hsai->Init.SynchroExt)
  {
    case SAI_SYNCEXT_DISABLE :
      tmpregisterGCR = 0U;
      break;
    case SAI_SYNCEXT_OUTBLOCKA_ENABLE :
      tmpregisterGCR = SAI_GCR_SYNCOUT_0;
      break;
    case SAI_SYNCEXT_OUTBLOCKB_ENABLE :
      tmpregisterGCR = SAI_GCR_SYNCOUT_1;
      break;
    default:
      tmpregisterGCR = 0U;
      break;
  }
  SAI1->GCR = tmpregisterGCR;
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx || STM32F413xx || STM32F423xx */
}
/**
* @brief  Get SAI Input Clock based on SAI source clock selection
* @param  hsai pointer to a SAI_HandleTypeDef structure that contains
*               the configuration information for SAI module.
* @retval SAI Clock Input
*/
uint32_t SAI_GetInputClock(SAI_HandleTypeDef *hsai)
{
  /* This variable used to store the SAI_CK_x (value in Hz) */
  uint32_t saiclocksource = 0U;

#if defined(STM32F446xx)
  if ((hsai->Instance == SAI1_Block_A) || (hsai->Instance == SAI1_Block_B))
  {
    saiclocksource = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_SAI1);
  }
  else /* SAI2_Block_A || SAI2_Block_B*/
  {
    saiclocksource = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_SAI2);
  }
#endif /* STM32F446xx */
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
  defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F413xx) || defined(STM32F423xx)
  uint32_t vcoinput = 0U, tmpreg = 0U;

  /* Check the SAI Block parameters */
  assert_param(IS_SAI_CLK_SOURCE(hsai->Init.ClockSource));

  /* SAI Block clock source selection */
  if (hsai->Instance == SAI1_Block_A)
  {
    __HAL_RCC_SAI_BLOCKACLKSOURCE_CONFIG(hsai->Init.ClockSource);
  }
  else
  {
    __HAL_RCC_SAI_BLOCKBCLKSOURCE_CONFIG((uint32_t)(hsai->Init.ClockSource << 2U));
  }

  /* VCO Input Clock value calculation */
  if ((RCC->PLLCFGR & RCC_PLLCFGR_PLLSRC) == RCC_PLLSOURCE_HSI)
  {
    /* In Case the PLL Source is HSI (Internal Clock) */
    vcoinput = (HSI_VALUE / (uint32_t)(RCC->PLLCFGR & RCC_PLLCFGR_PLLM));
  }
  else
  {
    /* In Case the PLL Source is HSE (External Clock) */
    vcoinput = ((HSE_VALUE / (uint32_t)(RCC->PLLCFGR & RCC_PLLCFGR_PLLM)));
  }
#if defined(STM32F413xx) || defined(STM32F423xx)
  /* SAI_CLK_x : SAI Block Clock configuration for different clock sources selected */
  if (hsai->Init.ClockSource == SAI_CLKSOURCE_PLLR)
  {
    /* Configure the PLLI2S division factor */
    /* PLL_VCO Input  = PLL_SOURCE/PLLM */
    /* PLL_VCO Output = PLL_VCO Input * PLLN */
    /* SAI_CLK(first level) = PLL_VCO Output/PLLR */
    tmpreg = (RCC->PLLCFGR & RCC_PLLCFGR_PLLR) >> 28U;
    saiclocksource = (vcoinput * ((RCC->PLLCFGR & RCC_PLLCFGR_PLLN) >> 6U)) / (tmpreg);

    /* SAI_CLK_x = SAI_CLK(first level)/PLLDIVR */
    tmpreg = (((RCC->DCKCFGR & RCC_DCKCFGR_PLLDIVR) >> 8U) + 1U);

    saiclocksource = saiclocksource / (tmpreg);

  }
  else if (hsai->Init.ClockSource == SAI_CLKSOURCE_PLLI2S)
  {
    /* Configure the PLLI2S division factor */
    /* PLLI2S_VCO Input  = PLL_SOURCE/PLLM */
    /* PLLI2S_VCO Output = PLLI2S_VCO Input * PLLI2SN */
    /* SAI_CLK(first level) = PLLI2S_VCO Output/PLLI2SR */
    tmpreg = (RCC->PLLI2SCFGR & RCC_PLLI2SCFGR_PLLI2SR) >> 28U;
    saiclocksource = (vcoinput * ((RCC->PLLI2SCFGR & RCC_PLLI2SCFGR_PLLI2SN) >> 6U)) / (tmpreg);

    /* SAI_CLK_x = SAI_CLK(first level)/PLLI2SDIVR */
    tmpreg = ((RCC->DCKCFGR & RCC_DCKCFGR_PLLI2SDIVR) + 1U);
    saiclocksource = saiclocksource / (tmpreg);
  }
  else if (hsai->Init.ClockSource == SAI_CLKSOURCE_HS)
  {
    if ((RCC->PLLCFGR & RCC_PLLCFGR_PLLSRC) == RCC_PLLSOURCE_HSE)
    {
      /* Get the I2S source clock value */
      saiclocksource = (uint32_t)(HSE_VALUE);
    }
    else
    {
      /* Get the I2S source clock value */
      saiclocksource = (uint32_t)(HSI_VALUE);
    }
  }
  else /* sConfig->ClockSource == SAI_CLKSource_Ext */
  {
    saiclocksource = EXTERNAL_CLOCK_VALUE;
  }
#else
  /* SAI_CLK_x : SAI Block Clock configuration for different clock sources selected */
  if (hsai->Init.ClockSource == SAI_CLKSOURCE_PLLSAI)
  {
    /* Configure the PLLI2S division factor */
    /* PLLSAI_VCO Input  = PLL_SOURCE/PLLM */
    /* PLLSAI_VCO Output = PLLSAI_VCO Input * PLLSAIN */
    /* SAI_CLK(first level) = PLLSAI_VCO Output/PLLSAIQ */
    tmpreg = (RCC->PLLSAICFGR & RCC_PLLSAICFGR_PLLSAIQ) >> 24U;
    saiclocksource = (vcoinput * ((RCC->PLLSAICFGR & RCC_PLLSAICFGR_PLLSAIN) >> 6U)) / (tmpreg);

    /* SAI_CLK_x = SAI_CLK(first level)/PLLSAIDIVQ */
    tmpreg = (((RCC->DCKCFGR & RCC_DCKCFGR_PLLSAIDIVQ) >> 8U) + 1U);
    saiclocksource = saiclocksource / (tmpreg);

  }
  else if (hsai->Init.ClockSource == SAI_CLKSOURCE_PLLI2S)
  {
    /* Configure the PLLI2S division factor */
    /* PLLI2S_VCO Input  = PLL_SOURCE/PLLM */
    /* PLLI2S_VCO Output = PLLI2S_VCO Input * PLLI2SN */
    /* SAI_CLK(first level) = PLLI2S_VCO Output/PLLI2SQ */
    tmpreg = (RCC->PLLI2SCFGR & RCC_PLLI2SCFGR_PLLI2SQ) >> 24U;
    saiclocksource = (vcoinput * ((RCC->PLLI2SCFGR & RCC_PLLI2SCFGR_PLLI2SN) >> 6U)) / (tmpreg);

    /* SAI_CLK_x = SAI_CLK(first level)/PLLI2SDIVQ */
    tmpreg = ((RCC->DCKCFGR & RCC_DCKCFGR_PLLI2SDIVQ) + 1U);
    saiclocksource = saiclocksource / (tmpreg);
  }
  else /* sConfig->ClockSource == SAI_CLKSource_Ext */
  {
    /* Enable the External Clock selection */
    __HAL_RCC_I2S_CONFIG(RCC_I2SCLKSOURCE_EXT);

    saiclocksource = EXTERNAL_CLOCK_VALUE;
  }
#endif /* STM32F413xx || STM32F423xx */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx || STM32F413xx || STM32F423xx */
  /* the return result is the value of SAI clock */
  return saiclocksource;
}

/**
  * @}
  */

/**
  * @}
  */

#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx  || STM32F446xx || STM32F469xx || STM32F479xx || STM32F413xx || STM32F423xx */
#endif /* HAL_SAI_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
