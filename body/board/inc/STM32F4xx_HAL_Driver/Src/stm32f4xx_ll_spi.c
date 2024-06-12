/**
  ******************************************************************************
  * @file    stm32f4xx_ll_spi.c
  * @author  MCD Application Team
  * @brief   SPI LL module driver.
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
#include "stm32f4xx_ll_spi.h"
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

#if defined (SPI1) || defined (SPI2) || defined (SPI3) || defined (SPI4) || defined (SPI5) || defined(SPI6)

/** @addtogroup SPI_LL
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/

/* Private constants ---------------------------------------------------------*/
/** @defgroup SPI_LL_Private_Constants SPI Private Constants
  * @{
  */
/* SPI registers Masks */
#define SPI_CR1_CLEAR_MASK                 (SPI_CR1_CPHA    | SPI_CR1_CPOL     | SPI_CR1_MSTR   | \
                                            SPI_CR1_BR      | SPI_CR1_LSBFIRST | SPI_CR1_SSI    | \
                                            SPI_CR1_SSM     | SPI_CR1_RXONLY   | SPI_CR1_DFF    | \
                                            SPI_CR1_CRCNEXT | SPI_CR1_CRCEN    | SPI_CR1_BIDIOE | \
                                            SPI_CR1_BIDIMODE)
/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup SPI_LL_Private_Macros SPI Private Macros
  * @{
  */
#define IS_LL_SPI_TRANSFER_DIRECTION(__VALUE__) (((__VALUE__) == LL_SPI_FULL_DUPLEX)       \
                                                 || ((__VALUE__) == LL_SPI_SIMPLEX_RX)     \
                                                 || ((__VALUE__) == LL_SPI_HALF_DUPLEX_RX) \
                                                 || ((__VALUE__) == LL_SPI_HALF_DUPLEX_TX))

#define IS_LL_SPI_MODE(__VALUE__) (((__VALUE__) == LL_SPI_MODE_MASTER) \
                                   || ((__VALUE__) == LL_SPI_MODE_SLAVE))

#define IS_LL_SPI_DATAWIDTH(__VALUE__) (((__VALUE__) == LL_SPI_DATAWIDTH_8BIT)  \
                                        || ((__VALUE__) == LL_SPI_DATAWIDTH_16BIT))

#define IS_LL_SPI_POLARITY(__VALUE__) (((__VALUE__) == LL_SPI_POLARITY_LOW) \
                                       || ((__VALUE__) == LL_SPI_POLARITY_HIGH))

#define IS_LL_SPI_PHASE(__VALUE__) (((__VALUE__) == LL_SPI_PHASE_1EDGE) \
                                    || ((__VALUE__) == LL_SPI_PHASE_2EDGE))

#define IS_LL_SPI_NSS(__VALUE__) (((__VALUE__) == LL_SPI_NSS_SOFT)          \
                                  || ((__VALUE__) == LL_SPI_NSS_HARD_INPUT) \
                                  || ((__VALUE__) == LL_SPI_NSS_HARD_OUTPUT))

#define IS_LL_SPI_BAUDRATE(__VALUE__) (((__VALUE__) == LL_SPI_BAUDRATEPRESCALER_DIV2)      \
                                       || ((__VALUE__) == LL_SPI_BAUDRATEPRESCALER_DIV4)   \
                                       || ((__VALUE__) == LL_SPI_BAUDRATEPRESCALER_DIV8)   \
                                       || ((__VALUE__) == LL_SPI_BAUDRATEPRESCALER_DIV16)  \
                                       || ((__VALUE__) == LL_SPI_BAUDRATEPRESCALER_DIV32)  \
                                       || ((__VALUE__) == LL_SPI_BAUDRATEPRESCALER_DIV64)  \
                                       || ((__VALUE__) == LL_SPI_BAUDRATEPRESCALER_DIV128) \
                                       || ((__VALUE__) == LL_SPI_BAUDRATEPRESCALER_DIV256))

#define IS_LL_SPI_BITORDER(__VALUE__) (((__VALUE__) == LL_SPI_LSB_FIRST) \
                                       || ((__VALUE__) == LL_SPI_MSB_FIRST))

#define IS_LL_SPI_CRCCALCULATION(__VALUE__) (((__VALUE__) == LL_SPI_CRCCALCULATION_ENABLE) \
                                             || ((__VALUE__) == LL_SPI_CRCCALCULATION_DISABLE))

#define IS_LL_SPI_CRC_POLYNOMIAL(__VALUE__) ((__VALUE__) >= 0x1U)

/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup SPI_LL_Exported_Functions
  * @{
  */

/** @addtogroup SPI_LL_EF_Init
  * @{
  */

/**
  * @brief  De-initialize the SPI registers to their default reset values.
  * @param  SPIx SPI Instance
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: SPI registers are de-initialized
  *          - ERROR: SPI registers are not de-initialized
  */
ErrorStatus LL_SPI_DeInit(SPI_TypeDef *SPIx)
{
  ErrorStatus status = ERROR;

  /* Check the parameters */
  assert_param(IS_SPI_ALL_INSTANCE(SPIx));

#if defined(SPI1)
  if (SPIx == SPI1)
  {
    /* Force reset of SPI clock */
    LL_APB2_GRP1_ForceReset(LL_APB2_GRP1_PERIPH_SPI1);

    /* Release reset of SPI clock */
    LL_APB2_GRP1_ReleaseReset(LL_APB2_GRP1_PERIPH_SPI1);

    status = SUCCESS;
  }
#endif /* SPI1 */
#if defined(SPI2)
  if (SPIx == SPI2)
  {
    /* Force reset of SPI clock */
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_SPI2);

    /* Release reset of SPI clock */
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_SPI2);

    status = SUCCESS;
  }
#endif /* SPI2 */
#if defined(SPI3)
  if (SPIx == SPI3)
  {
    /* Force reset of SPI clock */
    LL_APB1_GRP1_ForceReset(LL_APB1_GRP1_PERIPH_SPI3);

    /* Release reset of SPI clock */
    LL_APB1_GRP1_ReleaseReset(LL_APB1_GRP1_PERIPH_SPI3);

    status = SUCCESS;
  }
#endif /* SPI3 */
#if defined(SPI4)
  if (SPIx == SPI4)
  {
    /* Force reset of SPI clock */
    LL_APB2_GRP1_ForceReset(LL_APB2_GRP1_PERIPH_SPI4);

    /* Release reset of SPI clock */
    LL_APB2_GRP1_ReleaseReset(LL_APB2_GRP1_PERIPH_SPI4);

    status = SUCCESS;
  }
#endif /* SPI4 */
#if defined(SPI5)
  if (SPIx == SPI5)
  {
    /* Force reset of SPI clock */
    LL_APB2_GRP1_ForceReset(LL_APB2_GRP1_PERIPH_SPI5);

    /* Release reset of SPI clock */
    LL_APB2_GRP1_ReleaseReset(LL_APB2_GRP1_PERIPH_SPI5);

    status = SUCCESS;
  }
#endif /* SPI5 */
#if defined(SPI6)
  if (SPIx == SPI6)
  {
    /* Force reset of SPI clock */
    LL_APB2_GRP1_ForceReset(LL_APB2_GRP1_PERIPH_SPI6);

    /* Release reset of SPI clock */
    LL_APB2_GRP1_ReleaseReset(LL_APB2_GRP1_PERIPH_SPI6);

    status = SUCCESS;
  }
#endif /* SPI6 */

  return status;
}

/**
  * @brief  Initialize the SPI registers according to the specified parameters in SPI_InitStruct.
  * @note   As some bits in SPI configuration registers can only be written when the SPI is disabled (SPI_CR1_SPE bit =0),
  *         SPI peripheral should be in disabled state prior calling this function. Otherwise, ERROR result will be returned.
  * @param  SPIx SPI Instance
  * @param  SPI_InitStruct pointer to a @ref LL_SPI_InitTypeDef structure
  * @retval An ErrorStatus enumeration value. (Return always SUCCESS)
  */
ErrorStatus LL_SPI_Init(SPI_TypeDef *SPIx, LL_SPI_InitTypeDef *SPI_InitStruct)
{
  ErrorStatus status = ERROR;

  /* Check the SPI Instance SPIx*/
  assert_param(IS_SPI_ALL_INSTANCE(SPIx));

  /* Check the SPI parameters from SPI_InitStruct*/
  assert_param(IS_LL_SPI_TRANSFER_DIRECTION(SPI_InitStruct->TransferDirection));
  assert_param(IS_LL_SPI_MODE(SPI_InitStruct->Mode));
  assert_param(IS_LL_SPI_DATAWIDTH(SPI_InitStruct->DataWidth));
  assert_param(IS_LL_SPI_POLARITY(SPI_InitStruct->ClockPolarity));
  assert_param(IS_LL_SPI_PHASE(SPI_InitStruct->ClockPhase));
  assert_param(IS_LL_SPI_NSS(SPI_InitStruct->NSS));
  assert_param(IS_LL_SPI_BAUDRATE(SPI_InitStruct->BaudRate));
  assert_param(IS_LL_SPI_BITORDER(SPI_InitStruct->BitOrder));
  assert_param(IS_LL_SPI_CRCCALCULATION(SPI_InitStruct->CRCCalculation));

  if (LL_SPI_IsEnabled(SPIx) == 0x00000000U)
  {
    /*---------------------------- SPIx CR1 Configuration ------------------------
     * Configure SPIx CR1 with parameters:
     * - TransferDirection:  SPI_CR1_BIDIMODE, SPI_CR1_BIDIOE and SPI_CR1_RXONLY bits
     * - Master/Slave Mode:  SPI_CR1_MSTR bit
     * - DataWidth:          SPI_CR1_DFF bit
     * - ClockPolarity:      SPI_CR1_CPOL bit
     * - ClockPhase:         SPI_CR1_CPHA bit
     * - NSS management:     SPI_CR1_SSM bit
     * - BaudRate prescaler: SPI_CR1_BR[2:0] bits
     * - BitOrder:           SPI_CR1_LSBFIRST bit
     * - CRCCalculation:     SPI_CR1_CRCEN bit
     */
    MODIFY_REG(SPIx->CR1,
               SPI_CR1_CLEAR_MASK,
               SPI_InitStruct->TransferDirection | SPI_InitStruct->Mode | SPI_InitStruct->DataWidth |
               SPI_InitStruct->ClockPolarity | SPI_InitStruct->ClockPhase |
               SPI_InitStruct->NSS | SPI_InitStruct->BaudRate |
               SPI_InitStruct->BitOrder | SPI_InitStruct->CRCCalculation);

    /*---------------------------- SPIx CR2 Configuration ------------------------
     * Configure SPIx CR2 with parameters:
     * - NSS management:     SSOE bit
     */
    MODIFY_REG(SPIx->CR2, SPI_CR2_SSOE, (SPI_InitStruct->NSS >> 16U));

    /*---------------------------- SPIx CRCPR Configuration ----------------------
     * Configure SPIx CRCPR with parameters:
     * - CRCPoly:            CRCPOLY[15:0] bits
     */
    if (SPI_InitStruct->CRCCalculation == LL_SPI_CRCCALCULATION_ENABLE)
    {
      assert_param(IS_LL_SPI_CRC_POLYNOMIAL(SPI_InitStruct->CRCPoly));
      LL_SPI_SetCRCPolynomial(SPIx, SPI_InitStruct->CRCPoly);
    }
    status = SUCCESS;
  }

  /* Activate the SPI mode (Reset I2SMOD bit in I2SCFGR register) */
  CLEAR_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_I2SMOD);
  return status;
}

/**
  * @brief  Set each @ref LL_SPI_InitTypeDef field to default value.
  * @param  SPI_InitStruct pointer to a @ref LL_SPI_InitTypeDef structure
  * whose fields will be set to default values.
  * @retval None
  */
void LL_SPI_StructInit(LL_SPI_InitTypeDef *SPI_InitStruct)
{
  /* Set SPI_InitStruct fields to default values */
  SPI_InitStruct->TransferDirection = LL_SPI_FULL_DUPLEX;
  SPI_InitStruct->Mode              = LL_SPI_MODE_SLAVE;
  SPI_InitStruct->DataWidth         = LL_SPI_DATAWIDTH_8BIT;
  SPI_InitStruct->ClockPolarity     = LL_SPI_POLARITY_LOW;
  SPI_InitStruct->ClockPhase        = LL_SPI_PHASE_1EDGE;
  SPI_InitStruct->NSS               = LL_SPI_NSS_HARD_INPUT;
  SPI_InitStruct->BaudRate          = LL_SPI_BAUDRATEPRESCALER_DIV2;
  SPI_InitStruct->BitOrder          = LL_SPI_MSB_FIRST;
  SPI_InitStruct->CRCCalculation    = LL_SPI_CRCCALCULATION_DISABLE;
  SPI_InitStruct->CRCPoly           = 7U;
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

/** @addtogroup I2S_LL
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup I2S_LL_Private_Constants I2S Private Constants
  * @{
  */
/* I2S registers Masks */
#define I2S_I2SCFGR_CLEAR_MASK             (SPI_I2SCFGR_CHLEN   | SPI_I2SCFGR_DATLEN | \
                                            SPI_I2SCFGR_CKPOL   | SPI_I2SCFGR_I2SSTD | \
                                            SPI_I2SCFGR_I2SCFG  | SPI_I2SCFGR_I2SMOD )

#define I2S_I2SPR_CLEAR_MASK               0x0002U
/**
  * @}
  */
/* Private macros ------------------------------------------------------------*/
/** @defgroup I2S_LL_Private_Macros I2S Private Macros
  * @{
  */

#define IS_LL_I2S_DATAFORMAT(__VALUE__)  (((__VALUE__) == LL_I2S_DATAFORMAT_16B)             \
                                          || ((__VALUE__) == LL_I2S_DATAFORMAT_16B_EXTENDED) \
                                          || ((__VALUE__) == LL_I2S_DATAFORMAT_24B)          \
                                          || ((__VALUE__) == LL_I2S_DATAFORMAT_32B))

#define IS_LL_I2S_CPOL(__VALUE__)        (((__VALUE__) == LL_I2S_POLARITY_LOW)  \
                                          || ((__VALUE__) == LL_I2S_POLARITY_HIGH))

#define IS_LL_I2S_STANDARD(__VALUE__)    (((__VALUE__) == LL_I2S_STANDARD_PHILIPS)      \
                                          || ((__VALUE__) == LL_I2S_STANDARD_MSB)       \
                                          || ((__VALUE__) == LL_I2S_STANDARD_LSB)       \
                                          || ((__VALUE__) == LL_I2S_STANDARD_PCM_SHORT) \
                                          || ((__VALUE__) == LL_I2S_STANDARD_PCM_LONG))

#define IS_LL_I2S_MODE(__VALUE__)        (((__VALUE__) == LL_I2S_MODE_SLAVE_TX)     \
                                          || ((__VALUE__) == LL_I2S_MODE_SLAVE_RX)  \
                                          || ((__VALUE__) == LL_I2S_MODE_MASTER_TX) \
                                          || ((__VALUE__) == LL_I2S_MODE_MASTER_RX))

#define IS_LL_I2S_MCLK_OUTPUT(__VALUE__) (((__VALUE__) == LL_I2S_MCLK_OUTPUT_ENABLE) \
                                          || ((__VALUE__) == LL_I2S_MCLK_OUTPUT_DISABLE))

#define IS_LL_I2S_AUDIO_FREQ(__VALUE__) ((((__VALUE__) >= LL_I2S_AUDIOFREQ_8K)       \
                                          && ((__VALUE__) <= LL_I2S_AUDIOFREQ_192K)) \
                                         || ((__VALUE__) == LL_I2S_AUDIOFREQ_DEFAULT))

#define IS_LL_I2S_PRESCALER_LINEAR(__VALUE__)  ((__VALUE__) >= 0x2U)

#define IS_LL_I2S_PRESCALER_PARITY(__VALUE__) (((__VALUE__) == LL_I2S_PRESCALER_PARITY_EVEN) \
                                               || ((__VALUE__) == LL_I2S_PRESCALER_PARITY_ODD))
/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup I2S_LL_Exported_Functions
  * @{
  */

/** @addtogroup I2S_LL_EF_Init
  * @{
  */

/**
  * @brief  De-initialize the SPI/I2S registers to their default reset values.
  * @param  SPIx SPI Instance
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: SPI registers are de-initialized
  *          - ERROR: SPI registers are not de-initialized
  */
ErrorStatus LL_I2S_DeInit(SPI_TypeDef *SPIx)
{
  return LL_SPI_DeInit(SPIx);
}

/**
  * @brief  Initializes the SPI/I2S registers according to the specified parameters in I2S_InitStruct.
  * @note   As some bits in SPI configuration registers can only be written when the SPI is disabled (SPI_CR1_SPE bit =0),
  *         SPI peripheral should be in disabled state prior calling this function. Otherwise, ERROR result will be returned.
  * @param  SPIx SPI Instance
  * @param  I2S_InitStruct pointer to a @ref LL_I2S_InitTypeDef structure
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: SPI registers are Initialized
  *          - ERROR: SPI registers are not Initialized
  */
ErrorStatus LL_I2S_Init(SPI_TypeDef *SPIx, LL_I2S_InitTypeDef *I2S_InitStruct)
{
  uint32_t i2sdiv = 2U;
  uint32_t i2sodd = 0U;
  uint32_t packetlength = 1U;
  uint32_t tmp;
  uint32_t sourceclock;
  ErrorStatus status = ERROR;

  /* Check the I2S parameters */
  assert_param(IS_I2S_ALL_INSTANCE(SPIx));
  assert_param(IS_LL_I2S_MODE(I2S_InitStruct->Mode));
  assert_param(IS_LL_I2S_STANDARD(I2S_InitStruct->Standard));
  assert_param(IS_LL_I2S_DATAFORMAT(I2S_InitStruct->DataFormat));
  assert_param(IS_LL_I2S_MCLK_OUTPUT(I2S_InitStruct->MCLKOutput));
  assert_param(IS_LL_I2S_AUDIO_FREQ(I2S_InitStruct->AudioFreq));
  assert_param(IS_LL_I2S_CPOL(I2S_InitStruct->ClockPolarity));

  if (LL_I2S_IsEnabled(SPIx) == 0x00000000U)
  {
    /*---------------------------- SPIx I2SCFGR Configuration --------------------
     * Configure SPIx I2SCFGR with parameters:
     * - Mode:          SPI_I2SCFGR_I2SCFG[1:0] bit
     * - Standard:      SPI_I2SCFGR_I2SSTD[1:0] and SPI_I2SCFGR_PCMSYNC bits
     * - DataFormat:    SPI_I2SCFGR_CHLEN and SPI_I2SCFGR_DATLEN bits
     * - ClockPolarity: SPI_I2SCFGR_CKPOL bit
     */

    /* Write to SPIx I2SCFGR */
    MODIFY_REG(SPIx->I2SCFGR,
               I2S_I2SCFGR_CLEAR_MASK,
               I2S_InitStruct->Mode | I2S_InitStruct->Standard |
               I2S_InitStruct->DataFormat | I2S_InitStruct->ClockPolarity |
               SPI_I2SCFGR_I2SMOD);

    /*---------------------------- SPIx I2SPR Configuration ----------------------
     * Configure SPIx I2SPR with parameters:
     * - MCLKOutput:    SPI_I2SPR_MCKOE bit
     * - AudioFreq:     SPI_I2SPR_I2SDIV[7:0] and SPI_I2SPR_ODD bits
     */

    /* If the requested audio frequency is not the default, compute the prescaler (i2sodd, i2sdiv)
     * else, default values are used:  i2sodd = 0U, i2sdiv = 2U.
     */
    if (I2S_InitStruct->AudioFreq != LL_I2S_AUDIOFREQ_DEFAULT)
    {
      /* Check the frame length (For the Prescaler computing)
       * Default value: LL_I2S_DATAFORMAT_16B (packetlength = 1U).
       */
      if (I2S_InitStruct->DataFormat != LL_I2S_DATAFORMAT_16B)
      {
        /* Packet length is 32 bits */
        packetlength = 2U;
      }

      /* If an external I2S clock has to be used, the specific define should be set
      in the project configuration or in the stm32f4xx_ll_rcc.h file */
      /* Get the I2S source clock value */
      sourceclock = LL_RCC_GetI2SClockFreq(LL_RCC_I2S1_CLKSOURCE);

      /* Compute the Real divider depending on the MCLK output state with a floating point */
      if (I2S_InitStruct->MCLKOutput == LL_I2S_MCLK_OUTPUT_ENABLE)
      {
        /* MCLK output is enabled */
        tmp = (((((sourceclock / 256U) * 10U) / I2S_InitStruct->AudioFreq)) + 5U);
      }
      else
      {
        /* MCLK output is disabled */
        tmp = (((((sourceclock / (32U * packetlength)) * 10U) / I2S_InitStruct->AudioFreq)) + 5U);
      }

      /* Remove the floating point */
      tmp = tmp / 10U;

      /* Check the parity of the divider */
      i2sodd = (tmp & (uint16_t)0x0001U);

      /* Compute the i2sdiv prescaler */
      i2sdiv = ((tmp - i2sodd) / 2U);

      /* Get the Mask for the Odd bit (SPI_I2SPR[8]) register */
      i2sodd = (i2sodd << 8U);
    }

    /* Test if the divider is 1 or 0 or greater than 0xFF */
    if ((i2sdiv < 2U) || (i2sdiv > 0xFFU))
    {
      /* Set the default values */
      i2sdiv = 2U;
      i2sodd = 0U;
    }

    /* Write to SPIx I2SPR register the computed value */
    WRITE_REG(SPIx->I2SPR, i2sdiv | i2sodd | I2S_InitStruct->MCLKOutput);

    status = SUCCESS;
  }
  return status;
}

/**
  * @brief  Set each @ref LL_I2S_InitTypeDef field to default value.
  * @param  I2S_InitStruct pointer to a @ref LL_I2S_InitTypeDef structure
  *         whose fields will be set to default values.
  * @retval None
  */
void LL_I2S_StructInit(LL_I2S_InitTypeDef *I2S_InitStruct)
{
  /*--------------- Reset I2S init structure parameters values -----------------*/
  I2S_InitStruct->Mode              = LL_I2S_MODE_SLAVE_TX;
  I2S_InitStruct->Standard          = LL_I2S_STANDARD_PHILIPS;
  I2S_InitStruct->DataFormat        = LL_I2S_DATAFORMAT_16B;
  I2S_InitStruct->MCLKOutput        = LL_I2S_MCLK_OUTPUT_DISABLE;
  I2S_InitStruct->AudioFreq         = LL_I2S_AUDIOFREQ_DEFAULT;
  I2S_InitStruct->ClockPolarity     = LL_I2S_POLARITY_LOW;
}

/**
  * @brief  Set linear and parity prescaler.
  * @note   To calculate value of PrescalerLinear(I2SDIV[7:0] bits) and PrescalerParity(ODD bit)\n
  *         Check Audio frequency table and formulas inside Reference Manual (SPI/I2S).
  * @param  SPIx SPI Instance
  * @param  PrescalerLinear value Min_Data=0x02 and Max_Data=0xFF.
  * @param  PrescalerParity This parameter can be one of the following values:
  *         @arg @ref LL_I2S_PRESCALER_PARITY_EVEN
  *         @arg @ref LL_I2S_PRESCALER_PARITY_ODD
  * @retval None
  */
void LL_I2S_ConfigPrescaler(SPI_TypeDef *SPIx, uint32_t PrescalerLinear, uint32_t PrescalerParity)
{
  /* Check the I2S parameters */
  assert_param(IS_I2S_ALL_INSTANCE(SPIx));
  assert_param(IS_LL_I2S_PRESCALER_LINEAR(PrescalerLinear));
  assert_param(IS_LL_I2S_PRESCALER_PARITY(PrescalerParity));

  /* Write to SPIx I2SPR */
  MODIFY_REG(SPIx->I2SPR, SPI_I2SPR_I2SDIV | SPI_I2SPR_ODD, PrescalerLinear | (PrescalerParity << 8U));
}

#if defined (SPI_I2S_FULLDUPLEX_SUPPORT)
/**
  * @brief  Configures the full duplex mode for the I2Sx peripheral using its extension
  *         I2Sxext according to the specified parameters in the I2S_InitStruct.
  * @note   The structure pointed by I2S_InitStruct parameter should be the same
  *         used for the master I2S peripheral. In this case, if the master is
  *         configured as transmitter, the slave will be receiver and vice versa.
  *         Or you can force a different mode by modifying the field I2S_Mode to the
  *         value I2S_SlaveRx or I2S_SlaveTx independently of the master configuration.
  * @param  I2Sxext SPI Instance
  * @param  I2S_InitStruct pointer to a @ref LL_I2S_InitTypeDef structure
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: I2Sxext registers are Initialized
  *          - ERROR: I2Sxext registers are not Initialized
  */
ErrorStatus  LL_I2S_InitFullDuplex(SPI_TypeDef *I2Sxext, LL_I2S_InitTypeDef *I2S_InitStruct)
{
  uint32_t mode = 0U;
  ErrorStatus status = ERROR;

  /* Check the I2S parameters */
  assert_param(IS_I2S_EXT_ALL_INSTANCE(I2Sxext));
  assert_param(IS_LL_I2S_MODE(I2S_InitStruct->Mode));
  assert_param(IS_LL_I2S_STANDARD(I2S_InitStruct->Standard));
  assert_param(IS_LL_I2S_DATAFORMAT(I2S_InitStruct->DataFormat));
  assert_param(IS_LL_I2S_CPOL(I2S_InitStruct->ClockPolarity));

  if (LL_I2S_IsEnabled(I2Sxext) == 0x00000000U)
  {
    /*---------------------------- SPIx I2SCFGR Configuration --------------------
     * Configure SPIx I2SCFGR with parameters:
     * - Mode:          SPI_I2SCFGR_I2SCFG[1:0] bit
     * - Standard:      SPI_I2SCFGR_I2SSTD[1:0] and SPI_I2SCFGR_PCMSYNC bits
     * - DataFormat:    SPI_I2SCFGR_CHLEN and SPI_I2SCFGR_DATLEN bits
     * - ClockPolarity: SPI_I2SCFGR_CKPOL bit
     */

    /* Reset I2SPR registers */
    WRITE_REG(I2Sxext->I2SPR, I2S_I2SPR_CLEAR_MASK);

    /* Get the mode to be configured for the extended I2S */
    if ((I2S_InitStruct->Mode == LL_I2S_MODE_MASTER_TX) || (I2S_InitStruct->Mode == LL_I2S_MODE_SLAVE_TX))
    {
      mode = LL_I2S_MODE_SLAVE_RX;
    }
    else
    {
      if ((I2S_InitStruct->Mode == LL_I2S_MODE_MASTER_RX) || (I2S_InitStruct->Mode == LL_I2S_MODE_SLAVE_RX))
      {
        mode = LL_I2S_MODE_SLAVE_TX;
      }
    }

    /* Write to SPIx I2SCFGR */
    MODIFY_REG(I2Sxext->I2SCFGR,
               I2S_I2SCFGR_CLEAR_MASK,
               I2S_InitStruct->Standard |
               I2S_InitStruct->DataFormat | I2S_InitStruct->ClockPolarity |
               SPI_I2SCFGR_I2SMOD | mode);

    status = SUCCESS;
  }
  return status;
}
#endif /* SPI_I2S_FULLDUPLEX_SUPPORT */

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#endif /* defined (SPI1) || defined (SPI2) || defined (SPI3) || defined (SPI4) || defined (SPI5) || defined(SPI6) */

/**
  * @}
  */

#endif /* USE_FULL_LL_DRIVER */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
