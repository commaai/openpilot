/**
  ******************************************************************************
  * @file    stm32f4xx_ll_fmpi2c.c
  * @author  MCD Application Team
  * @brief   FMPI2C LL module driver.
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

#if defined(FMPI2C_CR1_PE)
/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_ll_fmpi2c.h"
#include "stm32f4xx_ll_bus.h"
#ifdef  USE_FULL_ASSERT
#include "stm32_assert.h"
#else
#define assert_param(expr) ((void)0U)
#endif /* USE_FULL_ASSERT */

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (FMPI2C1)

/** @defgroup FMPI2C_LL FMPI2C
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/** @addtogroup FMPI2C_LL_Private_Macros
  * @{
  */

#define IS_LL_FMPI2C_PERIPHERAL_MODE(__VALUE__)    (((__VALUE__) == LL_FMPI2C_MODE_I2C)          || \
                                                 ((__VALUE__) == LL_FMPI2C_MODE_SMBUS_HOST)   || \
                                                 ((__VALUE__) == LL_FMPI2C_MODE_SMBUS_DEVICE) || \
                                                 ((__VALUE__) == LL_FMPI2C_MODE_SMBUS_DEVICE_ARP))

#define IS_LL_FMPI2C_ANALOG_FILTER(__VALUE__)      (((__VALUE__) == LL_FMPI2C_ANALOGFILTER_ENABLE) || \
                                                 ((__VALUE__) == LL_FMPI2C_ANALOGFILTER_DISABLE))

#define IS_LL_FMPI2C_DIGITAL_FILTER(__VALUE__)     ((__VALUE__) <= 0x0000000FU)

#define IS_LL_FMPI2C_OWN_ADDRESS1(__VALUE__)       ((__VALUE__) <= 0x000003FFU)

#define IS_LL_FMPI2C_TYPE_ACKNOWLEDGE(__VALUE__)   (((__VALUE__) == LL_FMPI2C_ACK) || \
                                                 ((__VALUE__) == LL_FMPI2C_NACK))

#define IS_LL_FMPI2C_OWN_ADDRSIZE(__VALUE__)       (((__VALUE__) == LL_FMPI2C_OWNADDRESS1_7BIT) || \
                                                 ((__VALUE__) == LL_FMPI2C_OWNADDRESS1_10BIT))
/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup FMPI2C_LL_Exported_Functions
  * @{
  */

/** @addtogroup FMPI2C_LL_EF_Init
  * @{
  */

/**
  * @brief  De-initialize the FMPI2C registers to their default reset values.
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: FMPI2C registers are de-initialized
  *          - ERROR: FMPI2C registers are not de-initialized
  */
ErrorStatus LL_FMPI2C_DeInit(FMPI2C_TypeDef *FMPI2Cx)
{
  ErrorStatus status = SUCCESS;

  /* Check the FMPI2C Instance FMPI2Cx */
  assert_param(IS_FMPI2C_ALL_INSTANCE(FMPI2Cx));

  if (FMPI2Cx == FMPI2C1)
  {
    /* Force reset of FMPI2C clock */
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_FMPI2C1);

    /* Release reset of FMPI2C clock */
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_FMPI2C1);
  }
  else
  {
    status = ERROR;
  }

  return status;
}

/**
  * @brief  Initialize the FMPI2C registers according to the specified parameters in FMPI2C_InitStruct.
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  FMPI2C_InitStruct pointer to a @ref LL_FMPI2C_InitTypeDef structure.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: FMPI2C registers are initialized
  *          - ERROR: Not applicable
  */
ErrorStatus LL_FMPI2C_Init(FMPI2C_TypeDef *FMPI2Cx, LL_FMPI2C_InitTypeDef *FMPI2C_InitStruct)
{
  /* Check the FMPI2C Instance FMPI2Cx */
  assert_param(IS_FMPI2C_ALL_INSTANCE(FMPI2Cx));

  /* Check the FMPI2C parameters from FMPI2C_InitStruct */
  assert_param(IS_LL_FMPI2C_PERIPHERAL_MODE(FMPI2C_InitStruct->PeripheralMode));
  assert_param(IS_LL_FMPI2C_ANALOG_FILTER(FMPI2C_InitStruct->AnalogFilter));
  assert_param(IS_LL_FMPI2C_DIGITAL_FILTER(FMPI2C_InitStruct->DigitalFilter));
  assert_param(IS_LL_FMPI2C_OWN_ADDRESS1(FMPI2C_InitStruct->OwnAddress1));
  assert_param(IS_LL_FMPI2C_TYPE_ACKNOWLEDGE(FMPI2C_InitStruct->TypeAcknowledge));
  assert_param(IS_LL_FMPI2C_OWN_ADDRSIZE(FMPI2C_InitStruct->OwnAddrSize));

  /* Disable the selected FMPI2Cx Peripheral */
  LL_FMPI2C_Disable(FMPI2Cx);

  /*---------------------------- FMPI2Cx CR1 Configuration ------------------------
   * Configure the analog and digital noise filters with parameters :
   * - AnalogFilter: FMPI2C_CR1_ANFOFF bit
   * - DigitalFilter: FMPI2C_CR1_DNF[3:0] bits
   */
  LL_FMPI2C_ConfigFilters(FMPI2Cx, FMPI2C_InitStruct->AnalogFilter, FMPI2C_InitStruct->DigitalFilter);

  /*---------------------------- FMPI2Cx TIMINGR Configuration --------------------
   * Configure the SDA setup, hold time and the SCL high, low period with parameter :
   * - Timing: FMPI2C_TIMINGR_PRESC[3:0], FMPI2C_TIMINGR_SCLDEL[3:0], FMPI2C_TIMINGR_SDADEL[3:0],
   *           FMPI2C_TIMINGR_SCLH[7:0] and FMPI2C_TIMINGR_SCLL[7:0] bits
   */
  LL_FMPI2C_SetTiming(FMPI2Cx, FMPI2C_InitStruct->Timing);

  /* Enable the selected FMPI2Cx Peripheral */
  LL_FMPI2C_Enable(FMPI2Cx);

  /*---------------------------- FMPI2Cx OAR1 Configuration -----------------------
   * Disable, Configure and Enable FMPI2Cx device own address 1 with parameters :
   * - OwnAddress1:  FMPI2C_OAR1_OA1[9:0] bits
   * - OwnAddrSize:  FMPI2C_OAR1_OA1MODE bit
   */
  LL_FMPI2C_DisableOwnAddress1(FMPI2Cx);
  LL_FMPI2C_SetOwnAddress1(FMPI2Cx, FMPI2C_InitStruct->OwnAddress1, FMPI2C_InitStruct->OwnAddrSize);

  /* OwnAdress1 == 0 is reserved for General Call address */
  if (FMPI2C_InitStruct->OwnAddress1 != 0U)
  {
    LL_FMPI2C_EnableOwnAddress1(FMPI2Cx);
  }

  /*---------------------------- FMPI2Cx MODE Configuration -----------------------
  * Configure FMPI2Cx peripheral mode with parameter :
   * - PeripheralMode: FMPI2C_CR1_SMBDEN and FMPI2C_CR1_SMBHEN bits
   */
  LL_FMPI2C_SetMode(FMPI2Cx, FMPI2C_InitStruct->PeripheralMode);

  /*---------------------------- FMPI2Cx CR2 Configuration ------------------------
   * Configure the ACKnowledge or Non ACKnowledge condition
   * after the address receive match code or next received byte with parameter :
   * - TypeAcknowledge: FMPI2C_CR2_NACK bit
   */
  LL_FMPI2C_AcknowledgeNextData(FMPI2Cx, FMPI2C_InitStruct->TypeAcknowledge);

  return SUCCESS;
}

/**
  * @brief  Set each @ref LL_FMPI2C_InitTypeDef field to default value.
  * @param  FMPI2C_InitStruct Pointer to a @ref LL_FMPI2C_InitTypeDef structure.
  * @retval None
  */
void LL_FMPI2C_StructInit(LL_FMPI2C_InitTypeDef *FMPI2C_InitStruct)
{
  /* Set FMPI2C_InitStruct fields to default values */
  FMPI2C_InitStruct->PeripheralMode  = LL_FMPI2C_MODE_I2C;
  FMPI2C_InitStruct->Timing          = 0U;
  FMPI2C_InitStruct->AnalogFilter    = LL_FMPI2C_ANALOGFILTER_ENABLE;
  FMPI2C_InitStruct->DigitalFilter   = 0U;
  FMPI2C_InitStruct->OwnAddress1     = 0U;
  FMPI2C_InitStruct->TypeAcknowledge = LL_FMPI2C_NACK;
  FMPI2C_InitStruct->OwnAddrSize     = LL_FMPI2C_OWNADDRESS1_7BIT;
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

#endif /* FMPI2C1 */

/**
  * @}
  */

#endif /* FMPI2C_CR1_PE */
#endif /* USE_FULL_LL_DRIVER */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
