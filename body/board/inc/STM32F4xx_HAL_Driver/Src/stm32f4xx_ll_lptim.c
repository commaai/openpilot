/**
  ******************************************************************************
  * @file    stm32f4xx_ll_lptim.c
  * @author  MCD Application Team
  * @brief   LPTIM LL module driver.
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
#if defined(USE_FULL_LL_DRIVER)

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_ll_lptim.h"
#include "stm32f4xx_ll_bus.h"
#include "stm32f4xx_ll_rcc.h"


#ifdef  USE_FULL_ASSERT
#include "stm32_assert.h"
#else
#define assert_param(expr) ((void)0U)
#endif /* USE_FULL_ASSERT */

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (LPTIM1)

/** @addtogroup LPTIM_LL
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/** @addtogroup LPTIM_LL_Private_Macros
  * @{
  */
#define IS_LL_LPTIM_CLOCK_SOURCE(__VALUE__) (((__VALUE__) == LL_LPTIM_CLK_SOURCE_INTERNAL) \
                                             || ((__VALUE__) == LL_LPTIM_CLK_SOURCE_EXTERNAL))

#define IS_LL_LPTIM_CLOCK_PRESCALER(__VALUE__) (((__VALUE__) == LL_LPTIM_PRESCALER_DIV1)   \
                                                || ((__VALUE__) == LL_LPTIM_PRESCALER_DIV2)   \
                                                || ((__VALUE__) == LL_LPTIM_PRESCALER_DIV4)   \
                                                || ((__VALUE__) == LL_LPTIM_PRESCALER_DIV8)   \
                                                || ((__VALUE__) == LL_LPTIM_PRESCALER_DIV16)  \
                                                || ((__VALUE__) == LL_LPTIM_PRESCALER_DIV32)  \
                                                || ((__VALUE__) == LL_LPTIM_PRESCALER_DIV64)  \
                                                || ((__VALUE__) == LL_LPTIM_PRESCALER_DIV128))

#define IS_LL_LPTIM_WAVEFORM(__VALUE__) (((__VALUE__) == LL_LPTIM_OUTPUT_WAVEFORM_PWM) \
                                         || ((__VALUE__) == LL_LPTIM_OUTPUT_WAVEFORM_SETONCE))

#define IS_LL_LPTIM_OUTPUT_POLARITY(__VALUE__) (((__VALUE__) == LL_LPTIM_OUTPUT_POLARITY_REGULAR) \
                                                || ((__VALUE__) == LL_LPTIM_OUTPUT_POLARITY_INVERSE))
/**
  * @}
  */


/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/** @defgroup LPTIM_Private_Functions LPTIM Private Functions
  * @{
  */
/**
  * @}
  */
/* Exported functions --------------------------------------------------------*/
/** @addtogroup LPTIM_LL_Exported_Functions
  * @{
  */

/** @addtogroup LPTIM_LL_EF_Init
  * @{
  */

/**
  * @brief  Set LPTIMx registers to their reset values.
  * @param  LPTIMx LP Timer instance
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: LPTIMx registers are de-initialized
  *          - ERROR: invalid LPTIMx instance
  */
ErrorStatus LL_LPTIM_DeInit(LPTIM_TypeDef *LPTIMx)
{
  ErrorStatus result = SUCCESS;

  /* Check the parameters */
  assert_param(IS_LPTIM_INSTANCE(LPTIMx));

  if (LPTIMx == LPTIM1)
  {
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_LPTIM1);
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_LPTIM1);
  }
  else
  {
    result = ERROR;
  }

  return result;
}

/**
  * @brief  Set each fields of the LPTIM_InitStruct structure to its default
  *         value.
  * @param  LPTIM_InitStruct pointer to a @ref LL_LPTIM_InitTypeDef structure
  * @retval None
  */
void LL_LPTIM_StructInit(LL_LPTIM_InitTypeDef *LPTIM_InitStruct)
{
  /* Set the default configuration */
  LPTIM_InitStruct->ClockSource = LL_LPTIM_CLK_SOURCE_INTERNAL;
  LPTIM_InitStruct->Prescaler   = LL_LPTIM_PRESCALER_DIV1;
  LPTIM_InitStruct->Waveform    = LL_LPTIM_OUTPUT_WAVEFORM_PWM;
  LPTIM_InitStruct->Polarity    = LL_LPTIM_OUTPUT_POLARITY_REGULAR;
}

/**
  * @brief  Configure the LPTIMx peripheral according to the specified parameters.
  * @note LL_LPTIM_Init can only be called when the LPTIM instance is disabled.
  * @note LPTIMx can be disabled using unitary function @ref LL_LPTIM_Disable().
  * @param  LPTIMx LP Timer Instance
  * @param  LPTIM_InitStruct pointer to a @ref LL_LPTIM_InitTypeDef structure
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: LPTIMx instance has been initialized
  *          - ERROR: LPTIMx instance hasn't been initialized
  */
ErrorStatus LL_LPTIM_Init(LPTIM_TypeDef *LPTIMx, LL_LPTIM_InitTypeDef *LPTIM_InitStruct)
{
  ErrorStatus result = SUCCESS;
  /* Check the parameters */
  assert_param(IS_LPTIM_INSTANCE(LPTIMx));
  assert_param(IS_LL_LPTIM_CLOCK_SOURCE(LPTIM_InitStruct->ClockSource));
  assert_param(IS_LL_LPTIM_CLOCK_PRESCALER(LPTIM_InitStruct->Prescaler));
  assert_param(IS_LL_LPTIM_WAVEFORM(LPTIM_InitStruct->Waveform));
  assert_param(IS_LL_LPTIM_OUTPUT_POLARITY(LPTIM_InitStruct->Polarity));

  /* The LPTIMx_CFGR register must only be modified when the LPTIM is disabled
     (ENABLE bit is reset to 0).
  */
  if (LL_LPTIM_IsEnabled(LPTIMx) == 1UL)
  {
    result = ERROR;
  }
  else
  {
    /* Set CKSEL bitfield according to ClockSource value */
    /* Set PRESC bitfield according to Prescaler value */
    /* Set WAVE bitfield according to Waveform value */
    /* Set WAVEPOL bitfield according to Polarity value */
    MODIFY_REG(LPTIMx->CFGR,
               (LPTIM_CFGR_CKSEL | LPTIM_CFGR_PRESC | LPTIM_CFGR_WAVE | LPTIM_CFGR_WAVPOL),
               LPTIM_InitStruct->ClockSource | \
               LPTIM_InitStruct->Prescaler | \
               LPTIM_InitStruct->Waveform | \
               LPTIM_InitStruct->Polarity);
  }

  return result;
}

/**
  * @brief  Disable the LPTIM instance
  * @rmtoll CR           ENABLE        LL_LPTIM_Disable
  * @param  LPTIMx Low-Power Timer instance
  * @note   The following sequence is required to solve LPTIM disable HW limitation.
  *         Please check Errata Sheet ES0335 for more details under "MCU may remain
  *         stuck in LPTIM interrupt when entering Stop mode" section.
  * @retval None
  */
void LL_LPTIM_Disable(LPTIM_TypeDef *LPTIMx)
{
  LL_RCC_ClocksTypeDef rcc_clock;
  uint32_t tmpclksource = 0;
  uint32_t tmpIER;
  uint32_t tmpCFGR;
  uint32_t tmpCMP;
  uint32_t tmpARR;
  uint32_t tmpOR;

  /* Check the parameters */
  assert_param(IS_LPTIM_INSTANCE(LPTIMx));

  __disable_irq();

  /********** Save LPTIM Config *********/
  /* Save LPTIM source clock */
  switch ((uint32_t)LPTIMx)
  {
    case LPTIM1_BASE:
      tmpclksource = LL_RCC_GetLPTIMClockSource(LL_RCC_LPTIM1_CLKSOURCE);
      break;
    default:
      break;
  }

  /* Save LPTIM configuration registers */
  tmpIER = LPTIMx->IER;
  tmpCFGR = LPTIMx->CFGR;
  tmpCMP = LPTIMx->CMP;
  tmpARR = LPTIMx->ARR;
  tmpOR = LPTIMx->OR;

  /************* Reset LPTIM ************/
  (void)LL_LPTIM_DeInit(LPTIMx);

  /********* Restore LPTIM Config *******/
  LL_RCC_GetSystemClocksFreq(&rcc_clock);

  if ((tmpCMP != 0UL) || (tmpARR != 0UL))
  {
    /* Force LPTIM source kernel clock from APB */
    switch ((uint32_t)LPTIMx)
    {
      case LPTIM1_BASE:
        LL_RCC_SetLPTIMClockSource(LL_RCC_LPTIM1_CLKSOURCE_PCLK1);
        break;
      default:
        break;
    }

    if (tmpCMP != 0UL)
    {
      /* Restore CMP and ARR registers (LPTIM should be enabled first) */
      LPTIMx->CR |= LPTIM_CR_ENABLE;
      LPTIMx->CMP = tmpCMP;

      /* Polling on CMP write ok status after above restore operation */
      do
      {
        rcc_clock.SYSCLK_Frequency--; /* Used for timeout */
      } while (((LL_LPTIM_IsActiveFlag_CMPOK(LPTIMx) != 1UL)) && ((rcc_clock.SYSCLK_Frequency) > 0UL));

      LL_LPTIM_ClearFlag_CMPOK(LPTIMx);
    }

    if (tmpARR != 0UL)
    {
      LPTIMx->CR |= LPTIM_CR_ENABLE;
      LPTIMx->ARR = tmpARR;

      LL_RCC_GetSystemClocksFreq(&rcc_clock);
      /* Polling on ARR write ok status after above restore operation */
      do
      {
        rcc_clock.SYSCLK_Frequency--; /* Used for timeout */
      }
      while (((LL_LPTIM_IsActiveFlag_ARROK(LPTIMx) != 1UL)) && ((rcc_clock.SYSCLK_Frequency) > 0UL));

      LL_LPTIM_ClearFlag_ARROK(LPTIMx);
    }


    /* Restore LPTIM source kernel clock */
    LL_RCC_SetLPTIMClockSource(tmpclksource);
  }

  /* Restore configuration registers (LPTIM should be disabled first) */
  LPTIMx->CR &= ~(LPTIM_CR_ENABLE);
  LPTIMx->IER = tmpIER;
  LPTIMx->CFGR = tmpCFGR;
  LPTIMx->OR = tmpOR;

  __enable_irq();
}

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#endif /* LPTIM1 */

/**
  * @}
  */

#endif /* USE_FULL_LL_DRIVER */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
