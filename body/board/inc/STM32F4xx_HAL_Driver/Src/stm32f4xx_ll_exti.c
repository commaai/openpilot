/**
  ******************************************************************************
  * @file    stm32f4xx_ll_exti.c
  * @author  MCD Application Team
  * @brief   EXTI LL module driver.
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
#include "stm32f4xx_ll_exti.h"
#ifdef  USE_FULL_ASSERT
#include "stm32_assert.h"
#else
#define assert_param(expr) ((void)0U)
#endif

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (EXTI)

/** @defgroup EXTI_LL EXTI
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/** @addtogroup EXTI_LL_Private_Macros
  * @{
  */

#define IS_LL_EXTI_LINE_0_31(__VALUE__)              (((__VALUE__) & ~LL_EXTI_LINE_ALL_0_31) == 0x00000000U)

#define IS_LL_EXTI_MODE(__VALUE__)                   (((__VALUE__) == LL_EXTI_MODE_IT)            \
                                                   || ((__VALUE__) == LL_EXTI_MODE_EVENT)         \
                                                   || ((__VALUE__) == LL_EXTI_MODE_IT_EVENT))


#define IS_LL_EXTI_TRIGGER(__VALUE__)                (((__VALUE__) == LL_EXTI_TRIGGER_NONE)       \
                                                   || ((__VALUE__) == LL_EXTI_TRIGGER_RISING)     \
                                                   || ((__VALUE__) == LL_EXTI_TRIGGER_FALLING)    \
                                                   || ((__VALUE__) == LL_EXTI_TRIGGER_RISING_FALLING))

/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup EXTI_LL_Exported_Functions
  * @{
  */

/** @addtogroup EXTI_LL_EF_Init
  * @{
  */

/**
  * @brief  De-initialize the EXTI registers to their default reset values.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: EXTI registers are de-initialized
  *          - ERROR: not applicable
  */
uint32_t LL_EXTI_DeInit(void)
{
  /* Interrupt mask register set to default reset values */
  LL_EXTI_WriteReg(IMR,   0x00000000U);
  /* Event mask register set to default reset values */
  LL_EXTI_WriteReg(EMR,   0x00000000U);
  /* Rising Trigger selection register set to default reset values */
  LL_EXTI_WriteReg(RTSR,  0x00000000U);
  /* Falling Trigger selection register set to default reset values */
  LL_EXTI_WriteReg(FTSR,  0x00000000U);
  /* Software interrupt event register set to default reset values */
  LL_EXTI_WriteReg(SWIER, 0x00000000U);
  /* Pending register set to default reset values */
  LL_EXTI_WriteReg(PR,    0x00FFFFFFU);

  return SUCCESS;
}

/**
  * @brief  Initialize the EXTI registers according to the specified parameters in EXTI_InitStruct.
  * @param  EXTI_InitStruct pointer to a @ref LL_EXTI_InitTypeDef structure.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: EXTI registers are initialized
  *          - ERROR: not applicable
  */
uint32_t LL_EXTI_Init(LL_EXTI_InitTypeDef *EXTI_InitStruct)
{
  ErrorStatus status = SUCCESS;
  /* Check the parameters */
  assert_param(IS_LL_EXTI_LINE_0_31(EXTI_InitStruct->Line_0_31));
  assert_param(IS_FUNCTIONAL_STATE(EXTI_InitStruct->LineCommand));
  assert_param(IS_LL_EXTI_MODE(EXTI_InitStruct->Mode));

  /* ENABLE LineCommand */
  if (EXTI_InitStruct->LineCommand != DISABLE)
  {
    assert_param(IS_LL_EXTI_TRIGGER(EXTI_InitStruct->Trigger));

    /* Configure EXTI Lines in range from 0 to 31 */
    if (EXTI_InitStruct->Line_0_31 != LL_EXTI_LINE_NONE)
    {
      switch (EXTI_InitStruct->Mode)
      {
        case LL_EXTI_MODE_IT:
          /* First Disable Event on provided Lines */
          LL_EXTI_DisableEvent_0_31(EXTI_InitStruct->Line_0_31);
          /* Then Enable IT on provided Lines */
          LL_EXTI_EnableIT_0_31(EXTI_InitStruct->Line_0_31);
          break;
        case LL_EXTI_MODE_EVENT:
          /* First Disable IT on provided Lines */
          LL_EXTI_DisableIT_0_31(EXTI_InitStruct->Line_0_31);
          /* Then Enable Event on provided Lines */
          LL_EXTI_EnableEvent_0_31(EXTI_InitStruct->Line_0_31);
          break;
        case LL_EXTI_MODE_IT_EVENT:
          /* Directly Enable IT & Event on provided Lines */
          LL_EXTI_EnableIT_0_31(EXTI_InitStruct->Line_0_31);
          LL_EXTI_EnableEvent_0_31(EXTI_InitStruct->Line_0_31);
          break;
        default:
          status = ERROR;
          break;
      }
      if (EXTI_InitStruct->Trigger != LL_EXTI_TRIGGER_NONE)
      {
        switch (EXTI_InitStruct->Trigger)
        {
          case LL_EXTI_TRIGGER_RISING:
            /* First Disable Falling Trigger on provided Lines */
            LL_EXTI_DisableFallingTrig_0_31(EXTI_InitStruct->Line_0_31);
            /* Then Enable Rising Trigger on provided Lines */
            LL_EXTI_EnableRisingTrig_0_31(EXTI_InitStruct->Line_0_31);
            break;
          case LL_EXTI_TRIGGER_FALLING:
            /* First Disable Rising Trigger on provided Lines */
            LL_EXTI_DisableRisingTrig_0_31(EXTI_InitStruct->Line_0_31);
            /* Then Enable Falling Trigger on provided Lines */
            LL_EXTI_EnableFallingTrig_0_31(EXTI_InitStruct->Line_0_31);
            break;
          case LL_EXTI_TRIGGER_RISING_FALLING:
            LL_EXTI_EnableRisingTrig_0_31(EXTI_InitStruct->Line_0_31);
            LL_EXTI_EnableFallingTrig_0_31(EXTI_InitStruct->Line_0_31);
            break;
          default:
            status = ERROR;
            break;
        }
      }
    }
  }
  /* DISABLE LineCommand */
  else
  {
    /* De-configure EXTI Lines in range from 0 to 31 */
    LL_EXTI_DisableIT_0_31(EXTI_InitStruct->Line_0_31);
    LL_EXTI_DisableEvent_0_31(EXTI_InitStruct->Line_0_31);
  }
  return status;
}

/**
  * @brief  Set each @ref LL_EXTI_InitTypeDef field to default value.
  * @param  EXTI_InitStruct Pointer to a @ref LL_EXTI_InitTypeDef structure.
  * @retval None
  */
void LL_EXTI_StructInit(LL_EXTI_InitTypeDef *EXTI_InitStruct)
{
  EXTI_InitStruct->Line_0_31      = LL_EXTI_LINE_NONE;
  EXTI_InitStruct->LineCommand    = DISABLE;
  EXTI_InitStruct->Mode           = LL_EXTI_MODE_IT;
  EXTI_InitStruct->Trigger        = LL_EXTI_TRIGGER_FALLING;
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

#endif /* defined (EXTI) */

/**
  * @}
  */

#endif /* USE_FULL_LL_DRIVER */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
