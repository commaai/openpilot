/**
  ******************************************************************************
  * @file    stm32f4xx_ll_usart.c
  * @author  MCD Application Team
  * @brief   USART LL module driver.
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
#include "stm32f4xx_ll_usart.h"
#include "stm32f4xx_ll_rcc.h"
#include "stm32f4xx_ll_bus.h"
#ifdef  USE_FULL_ASSERT
#include "stm32_assert.h"
#else
#define assert_param(expr) ((void)0U)
#endif

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (USART1) || defined (USART2) || defined (USART3) || defined (USART6) || defined (UART4) || defined (UART5) || defined (UART7) || defined (UART8) || defined (UART9) || defined (UART10)

/** @addtogroup USART_LL
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @addtogroup USART_LL_Private_Constants
  * @{
  */

/**
  * @}
  */


/* Private macros ------------------------------------------------------------*/
/** @addtogroup USART_LL_Private_Macros
  * @{
  */

/* __BAUDRATE__ The maximum Baud Rate is derived from the maximum clock available
 *              divided by the smallest oversampling used on the USART (i.e. 8)    */
#define IS_LL_USART_BAUDRATE(__BAUDRATE__) ((__BAUDRATE__) <= 12500000U)

/* __VALUE__ In case of oversampling by 16 and 8, BRR content must be greater than or equal to 16d. */
#define IS_LL_USART_BRR_MIN(__VALUE__) ((__VALUE__) >= 16U)

#define IS_LL_USART_DIRECTION(__VALUE__) (((__VALUE__) == LL_USART_DIRECTION_NONE) \
                                          || ((__VALUE__) == LL_USART_DIRECTION_RX) \
                                          || ((__VALUE__) == LL_USART_DIRECTION_TX) \
                                          || ((__VALUE__) == LL_USART_DIRECTION_TX_RX))

#define IS_LL_USART_PARITY(__VALUE__) (((__VALUE__) == LL_USART_PARITY_NONE) \
                                       || ((__VALUE__) == LL_USART_PARITY_EVEN) \
                                       || ((__VALUE__) == LL_USART_PARITY_ODD))

#define IS_LL_USART_DATAWIDTH(__VALUE__) (((__VALUE__) == LL_USART_DATAWIDTH_8B) \
                                          || ((__VALUE__) == LL_USART_DATAWIDTH_9B))

#define IS_LL_USART_OVERSAMPLING(__VALUE__) (((__VALUE__) == LL_USART_OVERSAMPLING_16) \
                                             || ((__VALUE__) == LL_USART_OVERSAMPLING_8))

#define IS_LL_USART_LASTBITCLKOUTPUT(__VALUE__) (((__VALUE__) == LL_USART_LASTCLKPULSE_NO_OUTPUT) \
                                                 || ((__VALUE__) == LL_USART_LASTCLKPULSE_OUTPUT))

#define IS_LL_USART_CLOCKPHASE(__VALUE__) (((__VALUE__) == LL_USART_PHASE_1EDGE) \
                                           || ((__VALUE__) == LL_USART_PHASE_2EDGE))

#define IS_LL_USART_CLOCKPOLARITY(__VALUE__) (((__VALUE__) == LL_USART_POLARITY_LOW) \
                                              || ((__VALUE__) == LL_USART_POLARITY_HIGH))

#define IS_LL_USART_CLOCKOUTPUT(__VALUE__) (((__VALUE__) == LL_USART_CLOCK_DISABLE) \
                                            || ((__VALUE__) == LL_USART_CLOCK_ENABLE))

#define IS_LL_USART_STOPBITS(__VALUE__) (((__VALUE__) == LL_USART_STOPBITS_0_5) \
                                         || ((__VALUE__) == LL_USART_STOPBITS_1) \
                                         || ((__VALUE__) == LL_USART_STOPBITS_1_5) \
                                         || ((__VALUE__) == LL_USART_STOPBITS_2))

#define IS_LL_USART_HWCONTROL(__VALUE__) (((__VALUE__) == LL_USART_HWCONTROL_NONE) \
                                          || ((__VALUE__) == LL_USART_HWCONTROL_RTS) \
                                          || ((__VALUE__) == LL_USART_HWCONTROL_CTS) \
                                          || ((__VALUE__) == LL_USART_HWCONTROL_RTS_CTS))

/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup USART_LL_Exported_Functions
  * @{
  */

/** @addtogroup USART_LL_EF_Init
  * @{
  */

/**
  * @brief  De-initialize USART registers (Registers restored to their default values).
  * @param  USARTx USART Instance
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: USART registers are de-initialized
  *          - ERROR: USART registers are not de-initialized
  */
ErrorStatus LL_USART_DeInit(USART_TypeDef *USARTx)
{
  ErrorStatus status = SUCCESS;

  /* Check the parameters */
  assert_param(IS_UART_INSTANCE(USARTx));

  if (USARTx == USART1)
  {
    /* Force reset of USART clock */
    LL_APB2_GRP1_ForceReset(LL_APB2_GRP1_PERIPH_USART1);

    /* Release reset of USART clock */
    LL_APB2_GRP1_ReleaseReset(LL_APB2_GRP1_PERIPH_USART1);
  }
  else if (USARTx == USART2)
  {
    /* Force reset of USART clock */
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_USART2);

    /* Release reset of USART clock */
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_USART2);
  }
#if defined(USART3)
  else if (USARTx == USART3)
  {
    /* Force reset of USART clock */
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_USART3);

    /* Release reset of USART clock */
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_USART3);
  }
#endif /* USART3 */
#if defined(USART6)
  else if (USARTx == USART6)
  {
    /* Force reset of USART clock */
    LL_APB2_GRP1_ForceReset(LL_APB2_GRP1_PERIPH_USART6);

    /* Release reset of USART clock */
    LL_APB2_GRP1_ReleaseReset(LL_APB2_GRP1_PERIPH_USART6);
  }
#endif /* USART6 */
#if defined(UART4)
  else if (USARTx == UART4)
  {
    /* Force reset of UART clock */
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_UART4);

    /* Release reset of UART clock */
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_UART4);
  }
#endif /* UART4 */
#if defined(UART5)
  else if (USARTx == UART5)
  {
    /* Force reset of UART clock */
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_UART5);

    /* Release reset of UART clock */
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_UART5);
  }
#endif /* UART5 */
#if defined(UART7)
  else if (USARTx == UART7)
  {
    /* Force reset of UART clock */
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_UART7);

    /* Release reset of UART clock */
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_UART7);
  }
#endif /* UART7 */
#if defined(UART8)
  else if (USARTx == UART8)
  {
    /* Force reset of UART clock */
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_UART8);

    /* Release reset of UART clock */
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_UART8);
  }
#endif /* UART8 */
#if defined(UART9)
  else if (USARTx == UART9)
  {
    /* Force reset of UART clock */
    LL_APB2_GRP1_ForceReset(LL_APB2_GRP1_PERIPH_UART9);

    /* Release reset of UART clock */
    LL_APB2_GRP1_ReleaseReset(LL_APB2_GRP1_PERIPH_UART9);
  }
#endif /* UART9 */
#if defined(UART10)
  else if (USARTx == UART10)
  {
    /* Force reset of UART clock */
    LL_APB2_GRP1_ForceReset(LL_APB2_GRP1_PERIPH_UART10);

    /* Release reset of UART clock */
    LL_APB2_GRP1_ReleaseReset(LL_APB2_GRP1_PERIPH_UART10);
  }
#endif /* UART10 */
  else
  {
    status = ERROR;
  }

  return (status);
}

/**
  * @brief  Initialize USART registers according to the specified
  *         parameters in USART_InitStruct.
  * @note   As some bits in USART configuration registers can only be written when the USART is disabled (USART_CR1_UE bit =0),
  *         USART IP should be in disabled state prior calling this function. Otherwise, ERROR result will be returned.
  * @note   Baud rate value stored in USART_InitStruct BaudRate field, should be valid (different from 0).
  * @param  USARTx USART Instance
  * @param  USART_InitStruct pointer to a LL_USART_InitTypeDef structure
  *         that contains the configuration information for the specified USART peripheral.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: USART registers are initialized according to USART_InitStruct content
  *          - ERROR: Problem occurred during USART Registers initialization
  */
ErrorStatus LL_USART_Init(USART_TypeDef *USARTx, LL_USART_InitTypeDef *USART_InitStruct)
{
  ErrorStatus status = ERROR;
  uint32_t periphclk = LL_RCC_PERIPH_FREQUENCY_NO;
  LL_RCC_ClocksTypeDef rcc_clocks;

  /* Check the parameters */
  assert_param(IS_UART_INSTANCE(USARTx));
  assert_param(IS_LL_USART_BAUDRATE(USART_InitStruct->BaudRate));
  assert_param(IS_LL_USART_DATAWIDTH(USART_InitStruct->DataWidth));
  assert_param(IS_LL_USART_STOPBITS(USART_InitStruct->StopBits));
  assert_param(IS_LL_USART_PARITY(USART_InitStruct->Parity));
  assert_param(IS_LL_USART_DIRECTION(USART_InitStruct->TransferDirection));
  assert_param(IS_LL_USART_HWCONTROL(USART_InitStruct->HardwareFlowControl));
  assert_param(IS_LL_USART_OVERSAMPLING(USART_InitStruct->OverSampling));

  /* USART needs to be in disabled state, in order to be able to configure some bits in
     CRx registers */
  if (LL_USART_IsEnabled(USARTx) == 0U)
  {
    /*---------------------------- USART CR1 Configuration -----------------------
     * Configure USARTx CR1 (USART Word Length, Parity, Mode and Oversampling bits) with parameters:
     * - DataWidth:          USART_CR1_M bits according to USART_InitStruct->DataWidth value
     * - Parity:             USART_CR1_PCE, USART_CR1_PS bits according to USART_InitStruct->Parity value
     * - TransferDirection:  USART_CR1_TE, USART_CR1_RE bits according to USART_InitStruct->TransferDirection value
     * - Oversampling:       USART_CR1_OVER8 bit according to USART_InitStruct->OverSampling value.
     */
    MODIFY_REG(USARTx->CR1,
               (USART_CR1_M | USART_CR1_PCE | USART_CR1_PS |
                USART_CR1_TE | USART_CR1_RE | USART_CR1_OVER8),
               (USART_InitStruct->DataWidth | USART_InitStruct->Parity |
                USART_InitStruct->TransferDirection | USART_InitStruct->OverSampling));

    /*---------------------------- USART CR2 Configuration -----------------------
     * Configure USARTx CR2 (Stop bits) with parameters:
     * - Stop Bits:          USART_CR2_STOP bits according to USART_InitStruct->StopBits value.
     * - CLKEN, CPOL, CPHA and LBCL bits are to be configured using LL_USART_ClockInit().
     */
    LL_USART_SetStopBitsLength(USARTx, USART_InitStruct->StopBits);

    /*---------------------------- USART CR3 Configuration -----------------------
     * Configure USARTx CR3 (Hardware Flow Control) with parameters:
     * - HardwareFlowControl: USART_CR3_RTSE, USART_CR3_CTSE bits according to USART_InitStruct->HardwareFlowControl value.
     */
    LL_USART_SetHWFlowCtrl(USARTx, USART_InitStruct->HardwareFlowControl);

    /*---------------------------- USART BRR Configuration -----------------------
     * Retrieve Clock frequency used for USART Peripheral
     */
    LL_RCC_GetSystemClocksFreq(&rcc_clocks);
    if (USARTx == USART1)
    {
      periphclk = rcc_clocks.PCLK2_Frequency;
    }
    else if (USARTx == USART2)
    {
      periphclk = rcc_clocks.PCLK1_Frequency;
    }
#if defined(USART3)
    else if (USARTx == USART3)
    {
      periphclk = rcc_clocks.PCLK1_Frequency;
    }
#endif /* USART3 */
#if defined(USART6)
    else if (USARTx == USART6)
    {
      periphclk = rcc_clocks.PCLK2_Frequency;
    }
#endif /* USART6 */
#if defined(UART4)
    else if (USARTx == UART4)
    {
      periphclk = rcc_clocks.PCLK1_Frequency;
    }
#endif /* UART4 */
#if defined(UART5)
    else if (USARTx == UART5)
    {
      periphclk = rcc_clocks.PCLK1_Frequency;
    }
#endif /* UART5 */
#if defined(UART7)
    else if (USARTx == UART7)
    {
      periphclk = rcc_clocks.PCLK1_Frequency;
    }
#endif /* UART7 */
#if defined(UART8)
    else if (USARTx == UART8)
    {
      periphclk = rcc_clocks.PCLK1_Frequency;
    }
#endif /* UART8 */
#if defined(UART9)
    else if (USARTx == UART9)
    {
      periphclk = rcc_clocks.PCLK2_Frequency;
    }
#endif /* UART9 */
#if defined(UART10)
    else if (USARTx == UART10)
    {
      periphclk = rcc_clocks.PCLK2_Frequency;
    }
#endif /* UART10 */
    else
    {
      /* Nothing to do, as error code is already assigned to ERROR value */
    }

    /* Configure the USART Baud Rate :
       - valid baud rate value (different from 0) is required
       - Peripheral clock as returned by RCC service, should be valid (different from 0).
    */
    if ((periphclk != LL_RCC_PERIPH_FREQUENCY_NO)
        && (USART_InitStruct->BaudRate != 0U))
    {
      status = SUCCESS;
      LL_USART_SetBaudRate(USARTx,
                           periphclk,
                           USART_InitStruct->OverSampling,
                           USART_InitStruct->BaudRate);

      /* Check BRR is greater than or equal to 16d */
      assert_param(IS_LL_USART_BRR_MIN(USARTx->BRR));
    }
  }
  /* Endif (=> USART not in Disabled state => return ERROR) */

  return (status);
}

/**
  * @brief Set each @ref LL_USART_InitTypeDef field to default value.
  * @param USART_InitStruct Pointer to a @ref LL_USART_InitTypeDef structure
  *                         whose fields will be set to default values.
  * @retval None
  */

void LL_USART_StructInit(LL_USART_InitTypeDef *USART_InitStruct)
{
  /* Set USART_InitStruct fields to default values */
  USART_InitStruct->BaudRate            = 9600U;
  USART_InitStruct->DataWidth           = LL_USART_DATAWIDTH_8B;
  USART_InitStruct->StopBits            = LL_USART_STOPBITS_1;
  USART_InitStruct->Parity              = LL_USART_PARITY_NONE ;
  USART_InitStruct->TransferDirection   = LL_USART_DIRECTION_TX_RX;
  USART_InitStruct->HardwareFlowControl = LL_USART_HWCONTROL_NONE;
  USART_InitStruct->OverSampling        = LL_USART_OVERSAMPLING_16;
}

/**
  * @brief  Initialize USART Clock related settings according to the
  *         specified parameters in the USART_ClockInitStruct.
  * @note   As some bits in USART configuration registers can only be written when the USART is disabled (USART_CR1_UE bit =0),
  *         USART IP should be in disabled state prior calling this function. Otherwise, ERROR result will be returned.
  * @param  USARTx USART Instance
  * @param  USART_ClockInitStruct Pointer to a @ref LL_USART_ClockInitTypeDef structure
  *         that contains the Clock configuration information for the specified USART peripheral.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: USART registers related to Clock settings are initialized according to USART_ClockInitStruct content
  *          - ERROR: Problem occurred during USART Registers initialization
  */
ErrorStatus LL_USART_ClockInit(USART_TypeDef *USARTx, LL_USART_ClockInitTypeDef *USART_ClockInitStruct)
{
  ErrorStatus status = SUCCESS;

  /* Check USART Instance and Clock signal output parameters */
  assert_param(IS_UART_INSTANCE(USARTx));
  assert_param(IS_LL_USART_CLOCKOUTPUT(USART_ClockInitStruct->ClockOutput));

  /* USART needs to be in disabled state, in order to be able to configure some bits in
     CRx registers */
  if (LL_USART_IsEnabled(USARTx) == 0U)
  {
    /*---------------------------- USART CR2 Configuration -----------------------*/
    /* If Clock signal has to be output */
    if (USART_ClockInitStruct->ClockOutput == LL_USART_CLOCK_DISABLE)
    {
      /* Deactivate Clock signal delivery :
       * - Disable Clock Output:        USART_CR2_CLKEN cleared
       */
      LL_USART_DisableSCLKOutput(USARTx);
    }
    else
    {
      /* Ensure USART instance is USART capable */
      assert_param(IS_USART_INSTANCE(USARTx));

      /* Check clock related parameters */
      assert_param(IS_LL_USART_CLOCKPOLARITY(USART_ClockInitStruct->ClockPolarity));
      assert_param(IS_LL_USART_CLOCKPHASE(USART_ClockInitStruct->ClockPhase));
      assert_param(IS_LL_USART_LASTBITCLKOUTPUT(USART_ClockInitStruct->LastBitClockPulse));

      /*---------------------------- USART CR2 Configuration -----------------------
       * Configure USARTx CR2 (Clock signal related bits) with parameters:
       * - Enable Clock Output:         USART_CR2_CLKEN set
       * - Clock Polarity:              USART_CR2_CPOL bit according to USART_ClockInitStruct->ClockPolarity value
       * - Clock Phase:                 USART_CR2_CPHA bit according to USART_ClockInitStruct->ClockPhase value
       * - Last Bit Clock Pulse Output: USART_CR2_LBCL bit according to USART_ClockInitStruct->LastBitClockPulse value.
       */
      MODIFY_REG(USARTx->CR2,
                 USART_CR2_CLKEN | USART_CR2_CPHA | USART_CR2_CPOL | USART_CR2_LBCL,
                 USART_CR2_CLKEN | USART_ClockInitStruct->ClockPolarity |
                 USART_ClockInitStruct->ClockPhase | USART_ClockInitStruct->LastBitClockPulse);
    }
  }
  /* Else (USART not in Disabled state => return ERROR */
  else
  {
    status = ERROR;
  }

  return (status);
}

/**
  * @brief Set each field of a @ref LL_USART_ClockInitTypeDef type structure to default value.
  * @param USART_ClockInitStruct Pointer to a @ref LL_USART_ClockInitTypeDef structure
  *                              whose fields will be set to default values.
  * @retval None
  */
void LL_USART_ClockStructInit(LL_USART_ClockInitTypeDef *USART_ClockInitStruct)
{
  /* Set LL_USART_ClockInitStruct fields with default values */
  USART_ClockInitStruct->ClockOutput       = LL_USART_CLOCK_DISABLE;
  USART_ClockInitStruct->ClockPolarity     = LL_USART_POLARITY_LOW;            /* Not relevant when ClockOutput = LL_USART_CLOCK_DISABLE */
  USART_ClockInitStruct->ClockPhase        = LL_USART_PHASE_1EDGE;             /* Not relevant when ClockOutput = LL_USART_CLOCK_DISABLE */
  USART_ClockInitStruct->LastBitClockPulse = LL_USART_LASTCLKPULSE_NO_OUTPUT;  /* Not relevant when ClockOutput = LL_USART_CLOCK_DISABLE */
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

#endif /* USART1 || USART2 || USART3 || USART6 || UART4 || UART5 || UART7 || UART8 || UART9 || UART10 */

/**
  * @}
  */

#endif /* USE_FULL_LL_DRIVER */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/

