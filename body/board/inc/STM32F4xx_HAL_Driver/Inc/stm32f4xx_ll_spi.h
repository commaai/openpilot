/**
  ******************************************************************************
  * @file    stm32f4xx_ll_spi.h
  * @author  MCD Application Team
  * @brief   Header file of SPI LL module.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef STM32F4xx_LL_SPI_H
#define STM32F4xx_LL_SPI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (SPI1) || defined (SPI2) || defined (SPI3) || defined (SPI4) || defined (SPI5) || defined(SPI6)

/** @defgroup SPI_LL SPI
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/

/* Exported types ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup SPI_LL_ES_INIT SPI Exported Init structure
  * @{
  */

/**
  * @brief  SPI Init structures definition
  */
typedef struct
{
  uint32_t TransferDirection;       /*!< Specifies the SPI unidirectional or bidirectional data mode.
                                         This parameter can be a value of @ref SPI_LL_EC_TRANSFER_MODE.

                                         This feature can be modified afterwards using unitary function @ref LL_SPI_SetTransferDirection().*/

  uint32_t Mode;                    /*!< Specifies the SPI mode (Master/Slave).
                                         This parameter can be a value of @ref SPI_LL_EC_MODE.

                                         This feature can be modified afterwards using unitary function @ref LL_SPI_SetMode().*/

  uint32_t DataWidth;               /*!< Specifies the SPI data width.
                                         This parameter can be a value of @ref SPI_LL_EC_DATAWIDTH.

                                         This feature can be modified afterwards using unitary function @ref LL_SPI_SetDataWidth().*/

  uint32_t ClockPolarity;           /*!< Specifies the serial clock steady state.
                                         This parameter can be a value of @ref SPI_LL_EC_POLARITY.

                                         This feature can be modified afterwards using unitary function @ref LL_SPI_SetClockPolarity().*/

  uint32_t ClockPhase;              /*!< Specifies the clock active edge for the bit capture.
                                         This parameter can be a value of @ref SPI_LL_EC_PHASE.

                                         This feature can be modified afterwards using unitary function @ref LL_SPI_SetClockPhase().*/

  uint32_t NSS;                     /*!< Specifies whether the NSS signal is managed by hardware (NSS pin) or by software using the SSI bit.
                                         This parameter can be a value of @ref SPI_LL_EC_NSS_MODE.

                                         This feature can be modified afterwards using unitary function @ref LL_SPI_SetNSSMode().*/

  uint32_t BaudRate;                /*!< Specifies the BaudRate prescaler value which will be used to configure the transmit and receive SCK clock.
                                         This parameter can be a value of @ref SPI_LL_EC_BAUDRATEPRESCALER.
                                         @note The communication clock is derived from the master clock. The slave clock does not need to be set.

                                         This feature can be modified afterwards using unitary function @ref LL_SPI_SetBaudRatePrescaler().*/

  uint32_t BitOrder;                /*!< Specifies whether data transfers start from MSB or LSB bit.
                                         This parameter can be a value of @ref SPI_LL_EC_BIT_ORDER.

                                         This feature can be modified afterwards using unitary function @ref LL_SPI_SetTransferBitOrder().*/

  uint32_t CRCCalculation;          /*!< Specifies if the CRC calculation is enabled or not.
                                         This parameter can be a value of @ref SPI_LL_EC_CRC_CALCULATION.

                                         This feature can be modified afterwards using unitary functions @ref LL_SPI_EnableCRC() and @ref LL_SPI_DisableCRC().*/

  uint32_t CRCPoly;                 /*!< Specifies the polynomial used for the CRC calculation.
                                         This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFFFF.

                                         This feature can be modified afterwards using unitary function @ref LL_SPI_SetCRCPolynomial().*/

} LL_SPI_InitTypeDef;

/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/* Exported constants --------------------------------------------------------*/
/** @defgroup SPI_LL_Exported_Constants SPI Exported Constants
  * @{
  */

/** @defgroup SPI_LL_EC_GET_FLAG Get Flags Defines
  * @brief    Flags defines which can be used with LL_SPI_ReadReg function
  * @{
  */
#define LL_SPI_SR_RXNE                     SPI_SR_RXNE               /*!< Rx buffer not empty flag         */
#define LL_SPI_SR_TXE                      SPI_SR_TXE                /*!< Tx buffer empty flag             */
#define LL_SPI_SR_BSY                      SPI_SR_BSY                /*!< Busy flag                        */
#define LL_SPI_SR_CRCERR                   SPI_SR_CRCERR             /*!< CRC error flag                   */
#define LL_SPI_SR_MODF                     SPI_SR_MODF               /*!< Mode fault flag                  */
#define LL_SPI_SR_OVR                      SPI_SR_OVR                /*!< Overrun flag                     */
#define LL_SPI_SR_FRE                      SPI_SR_FRE                /*!< TI mode frame format error flag  */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_IT IT Defines
  * @brief    IT defines which can be used with LL_SPI_ReadReg and  LL_SPI_WriteReg functions
  * @{
  */
#define LL_SPI_CR2_RXNEIE                  SPI_CR2_RXNEIE            /*!< Rx buffer not empty interrupt enable */
#define LL_SPI_CR2_TXEIE                   SPI_CR2_TXEIE             /*!< Tx buffer empty interrupt enable     */
#define LL_SPI_CR2_ERRIE                   SPI_CR2_ERRIE             /*!< Error interrupt enable               */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_MODE Operation Mode
  * @{
  */
#define LL_SPI_MODE_MASTER                 (SPI_CR1_MSTR | SPI_CR1_SSI)    /*!< Master configuration  */
#define LL_SPI_MODE_SLAVE                  0x00000000U                     /*!< Slave configuration   */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_PROTOCOL Serial Protocol
  * @{
  */
#define LL_SPI_PROTOCOL_MOTOROLA           0x00000000U               /*!< Motorola mode. Used as default value */
#define LL_SPI_PROTOCOL_TI                 (SPI_CR2_FRF)             /*!< TI mode                              */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_PHASE Clock Phase
  * @{
  */
#define LL_SPI_PHASE_1EDGE                 0x00000000U               /*!< First clock transition is the first data capture edge  */
#define LL_SPI_PHASE_2EDGE                 (SPI_CR1_CPHA)            /*!< Second clock transition is the first data capture edge */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_POLARITY Clock Polarity
  * @{
  */
#define LL_SPI_POLARITY_LOW                0x00000000U               /*!< Clock to 0 when idle */
#define LL_SPI_POLARITY_HIGH               (SPI_CR1_CPOL)            /*!< Clock to 1 when idle */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_BAUDRATEPRESCALER Baud Rate Prescaler
  * @{
  */
#define LL_SPI_BAUDRATEPRESCALER_DIV2      0x00000000U                                    /*!< BaudRate control equal to fPCLK/2   */
#define LL_SPI_BAUDRATEPRESCALER_DIV4      (SPI_CR1_BR_0)                                 /*!< BaudRate control equal to fPCLK/4   */
#define LL_SPI_BAUDRATEPRESCALER_DIV8      (SPI_CR1_BR_1)                                 /*!< BaudRate control equal to fPCLK/8   */
#define LL_SPI_BAUDRATEPRESCALER_DIV16     (SPI_CR1_BR_1 | SPI_CR1_BR_0)                  /*!< BaudRate control equal to fPCLK/16  */
#define LL_SPI_BAUDRATEPRESCALER_DIV32     (SPI_CR1_BR_2)                                 /*!< BaudRate control equal to fPCLK/32  */
#define LL_SPI_BAUDRATEPRESCALER_DIV64     (SPI_CR1_BR_2 | SPI_CR1_BR_0)                  /*!< BaudRate control equal to fPCLK/64  */
#define LL_SPI_BAUDRATEPRESCALER_DIV128    (SPI_CR1_BR_2 | SPI_CR1_BR_1)                  /*!< BaudRate control equal to fPCLK/128 */
#define LL_SPI_BAUDRATEPRESCALER_DIV256    (SPI_CR1_BR_2 | SPI_CR1_BR_1 | SPI_CR1_BR_0)   /*!< BaudRate control equal to fPCLK/256 */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_BIT_ORDER Transmission Bit Order
  * @{
  */
#define LL_SPI_LSB_FIRST                   (SPI_CR1_LSBFIRST)        /*!< Data is transmitted/received with the LSB first */
#define LL_SPI_MSB_FIRST                   0x00000000U               /*!< Data is transmitted/received with the MSB first */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_TRANSFER_MODE Transfer Mode
  * @{
  */
#define LL_SPI_FULL_DUPLEX                 0x00000000U                          /*!< Full-Duplex mode. Rx and Tx transfer on 2 lines */
#define LL_SPI_SIMPLEX_RX                  (SPI_CR1_RXONLY)                     /*!< Simplex Rx mode.  Rx transfer only on 1 line    */
#define LL_SPI_HALF_DUPLEX_RX              (SPI_CR1_BIDIMODE)                   /*!< Half-Duplex Rx mode. Rx transfer on 1 line      */
#define LL_SPI_HALF_DUPLEX_TX              (SPI_CR1_BIDIMODE | SPI_CR1_BIDIOE)  /*!< Half-Duplex Tx mode. Tx transfer on 1 line      */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_NSS_MODE Slave Select Pin Mode
  * @{
  */
#define LL_SPI_NSS_SOFT                    (SPI_CR1_SSM)                     /*!< NSS managed internally. NSS pin not used and free              */
#define LL_SPI_NSS_HARD_INPUT              0x00000000U                       /*!< NSS pin used in Input. Only used in Master mode                */
#define LL_SPI_NSS_HARD_OUTPUT             (((uint32_t)SPI_CR2_SSOE << 16U)) /*!< NSS pin used in Output. Only used in Slave mode as chip select */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_DATAWIDTH Datawidth
  * @{
  */
#define LL_SPI_DATAWIDTH_8BIT              0x00000000U                       /*!< Data length for SPI transfer:  8 bits */
#define LL_SPI_DATAWIDTH_16BIT             (SPI_CR1_DFF)                     /*!< Data length for SPI transfer:  16 bits */
/**
  * @}
  */
#if defined(USE_FULL_LL_DRIVER)

/** @defgroup SPI_LL_EC_CRC_CALCULATION CRC Calculation
  * @{
  */
#define LL_SPI_CRCCALCULATION_DISABLE      0x00000000U               /*!< CRC calculation disabled */
#define LL_SPI_CRCCALCULATION_ENABLE       (SPI_CR1_CRCEN)           /*!< CRC calculation enabled  */
/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup SPI_LL_Exported_Macros SPI Exported Macros
  * @{
  */

/** @defgroup SPI_LL_EM_WRITE_READ Common Write and read registers Macros
  * @{
  */

/**
  * @brief  Write a value in SPI register
  * @param  __INSTANCE__ SPI Instance
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_SPI_WriteReg(__INSTANCE__, __REG__, __VALUE__) WRITE_REG(__INSTANCE__->__REG__, (__VALUE__))

/**
  * @brief  Read a value in SPI register
  * @param  __INSTANCE__ SPI Instance
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_SPI_ReadReg(__INSTANCE__, __REG__) READ_REG(__INSTANCE__->__REG__)
/**
  * @}
  */

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup SPI_LL_Exported_Functions SPI Exported Functions
  * @{
  */

/** @defgroup SPI_LL_EF_Configuration Configuration
  * @{
  */

/**
  * @brief  Enable SPI peripheral
  * @rmtoll CR1          SPE           LL_SPI_Enable
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_Enable(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->CR1, SPI_CR1_SPE);
}

/**
  * @brief  Disable SPI peripheral
  * @note   When disabling the SPI, follow the procedure described in the Reference Manual.
  * @rmtoll CR1          SPE           LL_SPI_Disable
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_Disable(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->CR1, SPI_CR1_SPE);
}

/**
  * @brief  Check if SPI peripheral is enabled
  * @rmtoll CR1          SPE           LL_SPI_IsEnabled
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsEnabled(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->CR1, SPI_CR1_SPE) == (SPI_CR1_SPE)) ? 1UL : 0UL);
}

/**
  * @brief  Set SPI operation mode to Master or Slave
  * @note   This bit should not be changed when communication is ongoing.
  * @rmtoll CR1          MSTR          LL_SPI_SetMode\n
  *         CR1          SSI           LL_SPI_SetMode
  * @param  SPIx SPI Instance
  * @param  Mode This parameter can be one of the following values:
  *         @arg @ref LL_SPI_MODE_MASTER
  *         @arg @ref LL_SPI_MODE_SLAVE
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetMode(SPI_TypeDef *SPIx, uint32_t Mode)
{
  MODIFY_REG(SPIx->CR1, SPI_CR1_MSTR | SPI_CR1_SSI, Mode);
}

/**
  * @brief  Get SPI operation mode (Master or Slave)
  * @rmtoll CR1          MSTR          LL_SPI_GetMode\n
  *         CR1          SSI           LL_SPI_GetMode
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SPI_MODE_MASTER
  *         @arg @ref LL_SPI_MODE_SLAVE
  */
__STATIC_INLINE uint32_t LL_SPI_GetMode(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->CR1, SPI_CR1_MSTR | SPI_CR1_SSI));
}

/**
  * @brief  Set serial protocol used
  * @note   This bit should be written only when SPI is disabled (SPE = 0) for correct operation.
  * @rmtoll CR2          FRF           LL_SPI_SetStandard
  * @param  SPIx SPI Instance
  * @param  Standard This parameter can be one of the following values:
  *         @arg @ref LL_SPI_PROTOCOL_MOTOROLA
  *         @arg @ref LL_SPI_PROTOCOL_TI
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetStandard(SPI_TypeDef *SPIx, uint32_t Standard)
{
  MODIFY_REG(SPIx->CR2, SPI_CR2_FRF, Standard);
}

/**
  * @brief  Get serial protocol used
  * @rmtoll CR2          FRF           LL_SPI_GetStandard
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SPI_PROTOCOL_MOTOROLA
  *         @arg @ref LL_SPI_PROTOCOL_TI
  */
__STATIC_INLINE uint32_t LL_SPI_GetStandard(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->CR2, SPI_CR2_FRF));
}

/**
  * @brief  Set clock phase
  * @note   This bit should not be changed when communication is ongoing.
  *         This bit is not used in SPI TI mode.
  * @rmtoll CR1          CPHA          LL_SPI_SetClockPhase
  * @param  SPIx SPI Instance
  * @param  ClockPhase This parameter can be one of the following values:
  *         @arg @ref LL_SPI_PHASE_1EDGE
  *         @arg @ref LL_SPI_PHASE_2EDGE
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetClockPhase(SPI_TypeDef *SPIx, uint32_t ClockPhase)
{
  MODIFY_REG(SPIx->CR1, SPI_CR1_CPHA, ClockPhase);
}

/**
  * @brief  Get clock phase
  * @rmtoll CR1          CPHA          LL_SPI_GetClockPhase
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SPI_PHASE_1EDGE
  *         @arg @ref LL_SPI_PHASE_2EDGE
  */
__STATIC_INLINE uint32_t LL_SPI_GetClockPhase(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->CR1, SPI_CR1_CPHA));
}

/**
  * @brief  Set clock polarity
  * @note   This bit should not be changed when communication is ongoing.
  *         This bit is not used in SPI TI mode.
  * @rmtoll CR1          CPOL          LL_SPI_SetClockPolarity
  * @param  SPIx SPI Instance
  * @param  ClockPolarity This parameter can be one of the following values:
  *         @arg @ref LL_SPI_POLARITY_LOW
  *         @arg @ref LL_SPI_POLARITY_HIGH
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetClockPolarity(SPI_TypeDef *SPIx, uint32_t ClockPolarity)
{
  MODIFY_REG(SPIx->CR1, SPI_CR1_CPOL, ClockPolarity);
}

/**
  * @brief  Get clock polarity
  * @rmtoll CR1          CPOL          LL_SPI_GetClockPolarity
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SPI_POLARITY_LOW
  *         @arg @ref LL_SPI_POLARITY_HIGH
  */
__STATIC_INLINE uint32_t LL_SPI_GetClockPolarity(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->CR1, SPI_CR1_CPOL));
}

/**
  * @brief  Set baud rate prescaler
  * @note   These bits should not be changed when communication is ongoing. SPI BaudRate = fPCLK/Prescaler.
  * @rmtoll CR1          BR            LL_SPI_SetBaudRatePrescaler
  * @param  SPIx SPI Instance
  * @param  BaudRate This parameter can be one of the following values:
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV2
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV4
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV8
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV16
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV32
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV64
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV128
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV256
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetBaudRatePrescaler(SPI_TypeDef *SPIx, uint32_t BaudRate)
{
  MODIFY_REG(SPIx->CR1, SPI_CR1_BR, BaudRate);
}

/**
  * @brief  Get baud rate prescaler
  * @rmtoll CR1          BR            LL_SPI_GetBaudRatePrescaler
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV2
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV4
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV8
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV16
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV32
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV64
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV128
  *         @arg @ref LL_SPI_BAUDRATEPRESCALER_DIV256
  */
__STATIC_INLINE uint32_t LL_SPI_GetBaudRatePrescaler(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->CR1, SPI_CR1_BR));
}

/**
  * @brief  Set transfer bit order
  * @note   This bit should not be changed when communication is ongoing. This bit is not used in SPI TI mode.
  * @rmtoll CR1          LSBFIRST      LL_SPI_SetTransferBitOrder
  * @param  SPIx SPI Instance
  * @param  BitOrder This parameter can be one of the following values:
  *         @arg @ref LL_SPI_LSB_FIRST
  *         @arg @ref LL_SPI_MSB_FIRST
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetTransferBitOrder(SPI_TypeDef *SPIx, uint32_t BitOrder)
{
  MODIFY_REG(SPIx->CR1, SPI_CR1_LSBFIRST, BitOrder);
}

/**
  * @brief  Get transfer bit order
  * @rmtoll CR1          LSBFIRST      LL_SPI_GetTransferBitOrder
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SPI_LSB_FIRST
  *         @arg @ref LL_SPI_MSB_FIRST
  */
__STATIC_INLINE uint32_t LL_SPI_GetTransferBitOrder(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->CR1, SPI_CR1_LSBFIRST));
}

/**
  * @brief  Set transfer direction mode
  * @note   For Half-Duplex mode, Rx Direction is set by default.
  *         In master mode, the MOSI pin is used and in slave mode, the MISO pin is used for Half-Duplex.
  * @rmtoll CR1          RXONLY        LL_SPI_SetTransferDirection\n
  *         CR1          BIDIMODE      LL_SPI_SetTransferDirection\n
  *         CR1          BIDIOE        LL_SPI_SetTransferDirection
  * @param  SPIx SPI Instance
  * @param  TransferDirection This parameter can be one of the following values:
  *         @arg @ref LL_SPI_FULL_DUPLEX
  *         @arg @ref LL_SPI_SIMPLEX_RX
  *         @arg @ref LL_SPI_HALF_DUPLEX_RX
  *         @arg @ref LL_SPI_HALF_DUPLEX_TX
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetTransferDirection(SPI_TypeDef *SPIx, uint32_t TransferDirection)
{
  MODIFY_REG(SPIx->CR1, SPI_CR1_RXONLY | SPI_CR1_BIDIMODE | SPI_CR1_BIDIOE, TransferDirection);
}

/**
  * @brief  Get transfer direction mode
  * @rmtoll CR1          RXONLY        LL_SPI_GetTransferDirection\n
  *         CR1          BIDIMODE      LL_SPI_GetTransferDirection\n
  *         CR1          BIDIOE        LL_SPI_GetTransferDirection
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SPI_FULL_DUPLEX
  *         @arg @ref LL_SPI_SIMPLEX_RX
  *         @arg @ref LL_SPI_HALF_DUPLEX_RX
  *         @arg @ref LL_SPI_HALF_DUPLEX_TX
  */
__STATIC_INLINE uint32_t LL_SPI_GetTransferDirection(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->CR1, SPI_CR1_RXONLY | SPI_CR1_BIDIMODE | SPI_CR1_BIDIOE));
}

/**
  * @brief  Set frame data width
  * @rmtoll CR1          DFF           LL_SPI_SetDataWidth
  * @param  SPIx SPI Instance
  * @param  DataWidth This parameter can be one of the following values:
  *         @arg @ref LL_SPI_DATAWIDTH_8BIT
  *         @arg @ref LL_SPI_DATAWIDTH_16BIT
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetDataWidth(SPI_TypeDef *SPIx, uint32_t DataWidth)
{
  MODIFY_REG(SPIx->CR1, SPI_CR1_DFF, DataWidth);
}

/**
  * @brief  Get frame data width
  * @rmtoll CR1          DFF           LL_SPI_GetDataWidth
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SPI_DATAWIDTH_8BIT
  *         @arg @ref LL_SPI_DATAWIDTH_16BIT
  */
__STATIC_INLINE uint32_t LL_SPI_GetDataWidth(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->CR1, SPI_CR1_DFF));
}

/**
  * @}
  */

/** @defgroup SPI_LL_EF_CRC_Management CRC Management
  * @{
  */

/**
  * @brief  Enable CRC
  * @note   This bit should be written only when SPI is disabled (SPE = 0) for correct operation.
  * @rmtoll CR1          CRCEN         LL_SPI_EnableCRC
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_EnableCRC(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->CR1, SPI_CR1_CRCEN);
}

/**
  * @brief  Disable CRC
  * @note   This bit should be written only when SPI is disabled (SPE = 0) for correct operation.
  * @rmtoll CR1          CRCEN         LL_SPI_DisableCRC
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_DisableCRC(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->CR1, SPI_CR1_CRCEN);
}

/**
  * @brief  Check if CRC is enabled
  * @note   This bit should be written only when SPI is disabled (SPE = 0) for correct operation.
  * @rmtoll CR1          CRCEN         LL_SPI_IsEnabledCRC
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsEnabledCRC(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->CR1, SPI_CR1_CRCEN) == (SPI_CR1_CRCEN)) ? 1UL : 0UL);
}

/**
  * @brief  Set CRCNext to transfer CRC on the line
  * @note   This bit has to be written as soon as the last data is written in the SPIx_DR register.
  * @rmtoll CR1          CRCNEXT       LL_SPI_SetCRCNext
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetCRCNext(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->CR1, SPI_CR1_CRCNEXT);
}

/**
  * @brief  Set polynomial for CRC calculation
  * @rmtoll CRCPR        CRCPOLY       LL_SPI_SetCRCPolynomial
  * @param  SPIx SPI Instance
  * @param  CRCPoly This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFFFF
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetCRCPolynomial(SPI_TypeDef *SPIx, uint32_t CRCPoly)
{
  WRITE_REG(SPIx->CRCPR, (uint16_t)CRCPoly);
}

/**
  * @brief  Get polynomial for CRC calculation
  * @rmtoll CRCPR        CRCPOLY       LL_SPI_GetCRCPolynomial
  * @param  SPIx SPI Instance
  * @retval Returned value is a number between Min_Data = 0x00 and Max_Data = 0xFFFF
  */
__STATIC_INLINE uint32_t LL_SPI_GetCRCPolynomial(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_REG(SPIx->CRCPR));
}

/**
  * @brief  Get Rx CRC
  * @rmtoll RXCRCR       RXCRC         LL_SPI_GetRxCRC
  * @param  SPIx SPI Instance
  * @retval Returned value is a number between Min_Data = 0x00 and Max_Data = 0xFFFF
  */
__STATIC_INLINE uint32_t LL_SPI_GetRxCRC(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_REG(SPIx->RXCRCR));
}

/**
  * @brief  Get Tx CRC
  * @rmtoll TXCRCR       TXCRC         LL_SPI_GetTxCRC
  * @param  SPIx SPI Instance
  * @retval Returned value is a number between Min_Data = 0x00 and Max_Data = 0xFFFF
  */
__STATIC_INLINE uint32_t LL_SPI_GetTxCRC(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_REG(SPIx->TXCRCR));
}

/**
  * @}
  */

/** @defgroup SPI_LL_EF_NSS_Management Slave Select Pin Management
  * @{
  */

/**
  * @brief  Set NSS mode
  * @note   LL_SPI_NSS_SOFT Mode is not used in SPI TI mode.
  * @rmtoll CR1          SSM           LL_SPI_SetNSSMode\n
  * @rmtoll CR2          SSOE          LL_SPI_SetNSSMode
  * @param  SPIx SPI Instance
  * @param  NSS This parameter can be one of the following values:
  *         @arg @ref LL_SPI_NSS_SOFT
  *         @arg @ref LL_SPI_NSS_HARD_INPUT
  *         @arg @ref LL_SPI_NSS_HARD_OUTPUT
  * @retval None
  */
__STATIC_INLINE void LL_SPI_SetNSSMode(SPI_TypeDef *SPIx, uint32_t NSS)
{
  MODIFY_REG(SPIx->CR1, SPI_CR1_SSM,  NSS);
  MODIFY_REG(SPIx->CR2, SPI_CR2_SSOE, ((uint32_t)(NSS >> 16U)));
}

/**
  * @brief  Get NSS mode
  * @rmtoll CR1          SSM           LL_SPI_GetNSSMode\n
  * @rmtoll CR2          SSOE          LL_SPI_GetNSSMode
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_SPI_NSS_SOFT
  *         @arg @ref LL_SPI_NSS_HARD_INPUT
  *         @arg @ref LL_SPI_NSS_HARD_OUTPUT
  */
__STATIC_INLINE uint32_t LL_SPI_GetNSSMode(SPI_TypeDef *SPIx)
{
  uint32_t Ssm  = (READ_BIT(SPIx->CR1, SPI_CR1_SSM));
  uint32_t Ssoe = (READ_BIT(SPIx->CR2,  SPI_CR2_SSOE) << 16U);
  return (Ssm | Ssoe);
}

/**
  * @}
  */

/** @defgroup SPI_LL_EF_FLAG_Management FLAG Management
  * @{
  */

/**
  * @brief  Check if Rx buffer is not empty
  * @rmtoll SR           RXNE          LL_SPI_IsActiveFlag_RXNE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsActiveFlag_RXNE(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->SR, SPI_SR_RXNE) == (SPI_SR_RXNE)) ? 1UL : 0UL);
}

/**
  * @brief  Check if Tx buffer is empty
  * @rmtoll SR           TXE           LL_SPI_IsActiveFlag_TXE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsActiveFlag_TXE(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->SR, SPI_SR_TXE) == (SPI_SR_TXE)) ? 1UL : 0UL);
}

/**
  * @brief  Get CRC error flag
  * @rmtoll SR           CRCERR        LL_SPI_IsActiveFlag_CRCERR
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsActiveFlag_CRCERR(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->SR, SPI_SR_CRCERR) == (SPI_SR_CRCERR)) ? 1UL : 0UL);
}

/**
  * @brief  Get mode fault error flag
  * @rmtoll SR           MODF          LL_SPI_IsActiveFlag_MODF
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsActiveFlag_MODF(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->SR, SPI_SR_MODF) == (SPI_SR_MODF)) ? 1UL : 0UL);
}

/**
  * @brief  Get overrun error flag
  * @rmtoll SR           OVR           LL_SPI_IsActiveFlag_OVR
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsActiveFlag_OVR(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->SR, SPI_SR_OVR) == (SPI_SR_OVR)) ? 1UL : 0UL);
}

/**
  * @brief  Get busy flag
  * @note   The BSY flag is cleared under any one of the following conditions:
  * -When the SPI is correctly disabled
  * -When a fault is detected in Master mode (MODF bit set to 1)
  * -In Master mode, when it finishes a data transmission and no new data is ready to be
  * sent
  * -In Slave mode, when the BSY flag is set to '0' for at least one SPI clock cycle between
  * each data transfer.
  * @rmtoll SR           BSY           LL_SPI_IsActiveFlag_BSY
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsActiveFlag_BSY(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->SR, SPI_SR_BSY) == (SPI_SR_BSY)) ? 1UL : 0UL);
}

/**
  * @brief  Get frame format error flag
  * @rmtoll SR           FRE           LL_SPI_IsActiveFlag_FRE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsActiveFlag_FRE(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->SR, SPI_SR_FRE) == (SPI_SR_FRE)) ? 1UL : 0UL);
}

/**
  * @brief  Clear CRC error flag
  * @rmtoll SR           CRCERR        LL_SPI_ClearFlag_CRCERR
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_ClearFlag_CRCERR(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->SR, SPI_SR_CRCERR);
}

/**
  * @brief  Clear mode fault error flag
  * @note   Clearing this flag is done by a read access to the SPIx_SR
  *         register followed by a write access to the SPIx_CR1 register
  * @rmtoll SR           MODF          LL_SPI_ClearFlag_MODF
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_ClearFlag_MODF(SPI_TypeDef *SPIx)
{
  __IO uint32_t tmpreg_sr;
  tmpreg_sr = SPIx->SR;
  (void) tmpreg_sr;
  CLEAR_BIT(SPIx->CR1, SPI_CR1_SPE);
}

/**
  * @brief  Clear overrun error flag
  * @note   Clearing this flag is done by a read access to the SPIx_DR
  *         register followed by a read access to the SPIx_SR register
  * @rmtoll SR           OVR           LL_SPI_ClearFlag_OVR
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_ClearFlag_OVR(SPI_TypeDef *SPIx)
{
  __IO uint32_t tmpreg;
  tmpreg = SPIx->DR;
  (void) tmpreg;
  tmpreg = SPIx->SR;
  (void) tmpreg;
}

/**
  * @brief  Clear frame format error flag
  * @note   Clearing this flag is done by reading SPIx_SR register
  * @rmtoll SR           FRE           LL_SPI_ClearFlag_FRE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_ClearFlag_FRE(SPI_TypeDef *SPIx)
{
  __IO uint32_t tmpreg;
  tmpreg = SPIx->SR;
  (void) tmpreg;
}

/**
  * @}
  */

/** @defgroup SPI_LL_EF_IT_Management Interrupt Management
  * @{
  */

/**
  * @brief  Enable error interrupt
  * @note   This bit controls the generation of an interrupt when an error condition occurs (CRCERR, OVR, MODF in SPI mode, FRE at TI mode).
  * @rmtoll CR2          ERRIE         LL_SPI_EnableIT_ERR
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_EnableIT_ERR(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->CR2, SPI_CR2_ERRIE);
}

/**
  * @brief  Enable Rx buffer not empty interrupt
  * @rmtoll CR2          RXNEIE        LL_SPI_EnableIT_RXNE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_EnableIT_RXNE(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->CR2, SPI_CR2_RXNEIE);
}

/**
  * @brief  Enable Tx buffer empty interrupt
  * @rmtoll CR2          TXEIE         LL_SPI_EnableIT_TXE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_EnableIT_TXE(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->CR2, SPI_CR2_TXEIE);
}

/**
  * @brief  Disable error interrupt
  * @note   This bit controls the generation of an interrupt when an error condition occurs (CRCERR, OVR, MODF in SPI mode, FRE at TI mode).
  * @rmtoll CR2          ERRIE         LL_SPI_DisableIT_ERR
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_DisableIT_ERR(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->CR2, SPI_CR2_ERRIE);
}

/**
  * @brief  Disable Rx buffer not empty interrupt
  * @rmtoll CR2          RXNEIE        LL_SPI_DisableIT_RXNE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_DisableIT_RXNE(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->CR2, SPI_CR2_RXNEIE);
}

/**
  * @brief  Disable Tx buffer empty interrupt
  * @rmtoll CR2          TXEIE         LL_SPI_DisableIT_TXE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_DisableIT_TXE(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->CR2, SPI_CR2_TXEIE);
}

/**
  * @brief  Check if error interrupt is enabled
  * @rmtoll CR2          ERRIE         LL_SPI_IsEnabledIT_ERR
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsEnabledIT_ERR(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->CR2, SPI_CR2_ERRIE) == (SPI_CR2_ERRIE)) ? 1UL : 0UL);
}

/**
  * @brief  Check if Rx buffer not empty interrupt is enabled
  * @rmtoll CR2          RXNEIE        LL_SPI_IsEnabledIT_RXNE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsEnabledIT_RXNE(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->CR2, SPI_CR2_RXNEIE) == (SPI_CR2_RXNEIE)) ? 1UL : 0UL);
}

/**
  * @brief  Check if Tx buffer empty interrupt
  * @rmtoll CR2          TXEIE         LL_SPI_IsEnabledIT_TXE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsEnabledIT_TXE(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->CR2, SPI_CR2_TXEIE) == (SPI_CR2_TXEIE)) ? 1UL : 0UL);
}

/**
  * @}
  */

/** @defgroup SPI_LL_EF_DMA_Management DMA Management
  * @{
  */

/**
  * @brief  Enable DMA Rx
  * @rmtoll CR2          RXDMAEN       LL_SPI_EnableDMAReq_RX
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_EnableDMAReq_RX(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->CR2, SPI_CR2_RXDMAEN);
}

/**
  * @brief  Disable DMA Rx
  * @rmtoll CR2          RXDMAEN       LL_SPI_DisableDMAReq_RX
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_DisableDMAReq_RX(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->CR2, SPI_CR2_RXDMAEN);
}

/**
  * @brief  Check if DMA Rx is enabled
  * @rmtoll CR2          RXDMAEN       LL_SPI_IsEnabledDMAReq_RX
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsEnabledDMAReq_RX(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->CR2, SPI_CR2_RXDMAEN) == (SPI_CR2_RXDMAEN)) ? 1UL : 0UL);
}

/**
  * @brief  Enable DMA Tx
  * @rmtoll CR2          TXDMAEN       LL_SPI_EnableDMAReq_TX
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_EnableDMAReq_TX(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->CR2, SPI_CR2_TXDMAEN);
}

/**
  * @brief  Disable DMA Tx
  * @rmtoll CR2          TXDMAEN       LL_SPI_DisableDMAReq_TX
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_SPI_DisableDMAReq_TX(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->CR2, SPI_CR2_TXDMAEN);
}

/**
  * @brief  Check if DMA Tx is enabled
  * @rmtoll CR2          TXDMAEN       LL_SPI_IsEnabledDMAReq_TX
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_SPI_IsEnabledDMAReq_TX(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->CR2, SPI_CR2_TXDMAEN) == (SPI_CR2_TXDMAEN)) ? 1UL : 0UL);
}

/**
  * @brief  Get the data register address used for DMA transfer
  * @rmtoll DR           DR            LL_SPI_DMA_GetRegAddr
  * @param  SPIx SPI Instance
  * @retval Address of data register
  */
__STATIC_INLINE uint32_t LL_SPI_DMA_GetRegAddr(SPI_TypeDef *SPIx)
{
  return (uint32_t) &(SPIx->DR);
}

/**
  * @}
  */

/** @defgroup SPI_LL_EF_DATA_Management DATA Management
  * @{
  */

/**
  * @brief  Read 8-Bits in the data register
  * @rmtoll DR           DR            LL_SPI_ReceiveData8
  * @param  SPIx SPI Instance
  * @retval RxData Value between Min_Data=0x00 and Max_Data=0xFF
  */
__STATIC_INLINE uint8_t LL_SPI_ReceiveData8(SPI_TypeDef *SPIx)
{
  return (*((__IO uint8_t *)&SPIx->DR));
}

/**
  * @brief  Read 16-Bits in the data register
  * @rmtoll DR           DR            LL_SPI_ReceiveData16
  * @param  SPIx SPI Instance
  * @retval RxData Value between Min_Data=0x00 and Max_Data=0xFFFF
  */
__STATIC_INLINE uint16_t LL_SPI_ReceiveData16(SPI_TypeDef *SPIx)
{
  return (uint16_t)(READ_REG(SPIx->DR));
}

/**
  * @brief  Write 8-Bits in the data register
  * @rmtoll DR           DR            LL_SPI_TransmitData8
  * @param  SPIx SPI Instance
  * @param  TxData Value between Min_Data=0x00 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_SPI_TransmitData8(SPI_TypeDef *SPIx, uint8_t TxData)
{
#if defined (__GNUC__)
  __IO uint8_t *spidr = ((__IO uint8_t *)&SPIx->DR);
  *spidr = TxData;
#else
  *((__IO uint8_t *)&SPIx->DR) = TxData;
#endif /* __GNUC__ */
}

/**
  * @brief  Write 16-Bits in the data register
  * @rmtoll DR           DR            LL_SPI_TransmitData16
  * @param  SPIx SPI Instance
  * @param  TxData Value between Min_Data=0x00 and Max_Data=0xFFFF
  * @retval None
  */
__STATIC_INLINE void LL_SPI_TransmitData16(SPI_TypeDef *SPIx, uint16_t TxData)
{
#if defined (__GNUC__)
  __IO uint16_t *spidr = ((__IO uint16_t *)&SPIx->DR);
  *spidr = TxData;
#else
  SPIx->DR = TxData;
#endif /* __GNUC__ */
}

/**
  * @}
  */
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup SPI_LL_EF_Init Initialization and de-initialization functions
  * @{
  */

ErrorStatus LL_SPI_DeInit(SPI_TypeDef *SPIx);
ErrorStatus LL_SPI_Init(SPI_TypeDef *SPIx, LL_SPI_InitTypeDef *SPI_InitStruct);
void        LL_SPI_StructInit(LL_SPI_InitTypeDef *SPI_InitStruct);

/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */
/**
  * @}
  */

/**
  * @}
  */

/** @defgroup I2S_LL I2S
  * @{
  */

/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/

/* Exported types ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup I2S_LL_ES_INIT I2S Exported Init structure
  * @{
  */

/**
  * @brief  I2S Init structure definition
  */

typedef struct
{
  uint32_t Mode;                    /*!< Specifies the I2S operating mode.
                                         This parameter can be a value of @ref I2S_LL_EC_MODE

                                         This feature can be modified afterwards using unitary function @ref LL_I2S_SetTransferMode().*/

  uint32_t Standard;                /*!< Specifies the standard used for the I2S communication.
                                         This parameter can be a value of @ref I2S_LL_EC_STANDARD

                                         This feature can be modified afterwards using unitary function @ref LL_I2S_SetStandard().*/


  uint32_t DataFormat;              /*!< Specifies the data format for the I2S communication.
                                         This parameter can be a value of @ref I2S_LL_EC_DATA_FORMAT

                                         This feature can be modified afterwards using unitary function @ref LL_I2S_SetDataFormat().*/


  uint32_t MCLKOutput;              /*!< Specifies whether the I2S MCLK output is enabled or not.
                                         This parameter can be a value of @ref I2S_LL_EC_MCLK_OUTPUT

                                         This feature can be modified afterwards using unitary functions @ref LL_I2S_EnableMasterClock() or @ref LL_I2S_DisableMasterClock.*/


  uint32_t AudioFreq;               /*!< Specifies the frequency selected for the I2S communication.
                                         This parameter can be a value of @ref I2S_LL_EC_AUDIO_FREQ

                                         Audio Frequency can be modified afterwards using Reference manual formulas to calculate Prescaler Linear, Parity
                                         and unitary functions @ref LL_I2S_SetPrescalerLinear() and @ref LL_I2S_SetPrescalerParity() to set it.*/


  uint32_t ClockPolarity;           /*!< Specifies the idle state of the I2S clock.
                                         This parameter can be a value of @ref I2S_LL_EC_POLARITY

                                         This feature can be modified afterwards using unitary function @ref LL_I2S_SetClockPolarity().*/

} LL_I2S_InitTypeDef;

/**
  * @}
  */
#endif /*USE_FULL_LL_DRIVER*/

/* Exported constants --------------------------------------------------------*/
/** @defgroup I2S_LL_Exported_Constants I2S Exported Constants
  * @{
  */

/** @defgroup I2S_LL_EC_GET_FLAG Get Flags Defines
  * @brief    Flags defines which can be used with LL_I2S_ReadReg function
  * @{
  */
#define LL_I2S_SR_RXNE                     LL_SPI_SR_RXNE            /*!< Rx buffer not empty flag         */
#define LL_I2S_SR_TXE                      LL_SPI_SR_TXE             /*!< Tx buffer empty flag             */
#define LL_I2S_SR_BSY                      LL_SPI_SR_BSY             /*!< Busy flag                        */
#define LL_I2S_SR_UDR                      SPI_SR_UDR                /*!< Underrun flag                    */
#define LL_I2S_SR_OVR                      LL_SPI_SR_OVR             /*!< Overrun flag                     */
#define LL_I2S_SR_FRE                      LL_SPI_SR_FRE             /*!< TI mode frame format error flag  */
/**
  * @}
  */

/** @defgroup SPI_LL_EC_IT IT Defines
  * @brief    IT defines which can be used with LL_SPI_ReadReg and  LL_SPI_WriteReg functions
  * @{
  */
#define LL_I2S_CR2_RXNEIE                  LL_SPI_CR2_RXNEIE         /*!< Rx buffer not empty interrupt enable */
#define LL_I2S_CR2_TXEIE                   LL_SPI_CR2_TXEIE          /*!< Tx buffer empty interrupt enable     */
#define LL_I2S_CR2_ERRIE                   LL_SPI_CR2_ERRIE          /*!< Error interrupt enable               */
/**
  * @}
  */

/** @defgroup I2S_LL_EC_DATA_FORMAT Data format
  * @{
  */
#define LL_I2S_DATAFORMAT_16B              0x00000000U                                   /*!< Data length 16 bits, Channel length 16bit */
#define LL_I2S_DATAFORMAT_16B_EXTENDED     (SPI_I2SCFGR_CHLEN)                           /*!< Data length 16 bits, Channel length 32bit */
#define LL_I2S_DATAFORMAT_24B              (SPI_I2SCFGR_CHLEN | SPI_I2SCFGR_DATLEN_0)    /*!< Data length 24 bits, Channel length 32bit */
#define LL_I2S_DATAFORMAT_32B              (SPI_I2SCFGR_CHLEN | SPI_I2SCFGR_DATLEN_1)    /*!< Data length 16 bits, Channel length 32bit */
/**
  * @}
  */

/** @defgroup I2S_LL_EC_POLARITY Clock Polarity
  * @{
  */
#define LL_I2S_POLARITY_LOW                0x00000000U               /*!< Clock steady state is low level  */
#define LL_I2S_POLARITY_HIGH               (SPI_I2SCFGR_CKPOL)       /*!< Clock steady state is high level */
/**
  * @}
  */

/** @defgroup I2S_LL_EC_STANDARD I2s Standard
  * @{
  */
#define LL_I2S_STANDARD_PHILIPS            0x00000000U                                                         /*!< I2S standard philips                      */
#define LL_I2S_STANDARD_MSB                (SPI_I2SCFGR_I2SSTD_0)                                              /*!< MSB justified standard (left justified)   */
#define LL_I2S_STANDARD_LSB                (SPI_I2SCFGR_I2SSTD_1)                                              /*!< LSB justified standard (right justified)  */
#define LL_I2S_STANDARD_PCM_SHORT          (SPI_I2SCFGR_I2SSTD_0 | SPI_I2SCFGR_I2SSTD_1)                       /*!< PCM standard, short frame synchronization */
#define LL_I2S_STANDARD_PCM_LONG           (SPI_I2SCFGR_I2SSTD_0 | SPI_I2SCFGR_I2SSTD_1 | SPI_I2SCFGR_PCMSYNC) /*!< PCM standard, long frame synchronization  */
/**
  * @}
  */

/** @defgroup I2S_LL_EC_MODE Operation Mode
  * @{
  */
#define LL_I2S_MODE_SLAVE_TX               0x00000000U                                   /*!< Slave Tx configuration  */
#define LL_I2S_MODE_SLAVE_RX               (SPI_I2SCFGR_I2SCFG_0)                        /*!< Slave Rx configuration  */
#define LL_I2S_MODE_MASTER_TX              (SPI_I2SCFGR_I2SCFG_1)                        /*!< Master Tx configuration */
#define LL_I2S_MODE_MASTER_RX              (SPI_I2SCFGR_I2SCFG_0 | SPI_I2SCFGR_I2SCFG_1) /*!< Master Rx configuration */
/**
  * @}
  */

/** @defgroup I2S_LL_EC_PRESCALER_FACTOR Prescaler Factor
  * @{
  */
#define LL_I2S_PRESCALER_PARITY_EVEN       0x00000000U               /*!< Odd factor: Real divider value is =  I2SDIV * 2    */
#define LL_I2S_PRESCALER_PARITY_ODD        (SPI_I2SPR_ODD >> 8U)     /*!< Odd factor: Real divider value is = (I2SDIV * 2)+1 */
/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)

/** @defgroup I2S_LL_EC_MCLK_OUTPUT MCLK Output
  * @{
  */
#define LL_I2S_MCLK_OUTPUT_DISABLE         0x00000000U               /*!< Master clock output is disabled */
#define LL_I2S_MCLK_OUTPUT_ENABLE          (SPI_I2SPR_MCKOE)         /*!< Master clock output is enabled  */
/**
  * @}
  */

/** @defgroup I2S_LL_EC_AUDIO_FREQ Audio Frequency
  * @{
  */

#define LL_I2S_AUDIOFREQ_192K              192000U       /*!< Audio Frequency configuration 192000 Hz       */
#define LL_I2S_AUDIOFREQ_96K               96000U        /*!< Audio Frequency configuration  96000 Hz       */
#define LL_I2S_AUDIOFREQ_48K               48000U        /*!< Audio Frequency configuration  48000 Hz       */
#define LL_I2S_AUDIOFREQ_44K               44100U        /*!< Audio Frequency configuration  44100 Hz       */
#define LL_I2S_AUDIOFREQ_32K               32000U        /*!< Audio Frequency configuration  32000 Hz       */
#define LL_I2S_AUDIOFREQ_22K               22050U        /*!< Audio Frequency configuration  22050 Hz       */
#define LL_I2S_AUDIOFREQ_16K               16000U        /*!< Audio Frequency configuration  16000 Hz       */
#define LL_I2S_AUDIOFREQ_11K               11025U        /*!< Audio Frequency configuration  11025 Hz       */
#define LL_I2S_AUDIOFREQ_8K                8000U         /*!< Audio Frequency configuration   8000 Hz       */
#define LL_I2S_AUDIOFREQ_DEFAULT           2U            /*!< Audio Freq not specified. Register I2SDIV = 2 */
/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup I2S_LL_Exported_Macros I2S Exported Macros
  * @{
  */

/** @defgroup I2S_LL_EM_WRITE_READ Common Write and read registers Macros
  * @{
  */

/**
  * @brief  Write a value in I2S register
  * @param  __INSTANCE__ I2S Instance
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_I2S_WriteReg(__INSTANCE__, __REG__, __VALUE__) WRITE_REG(__INSTANCE__->__REG__, (__VALUE__))

/**
  * @brief  Read a value in I2S register
  * @param  __INSTANCE__ I2S Instance
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_I2S_ReadReg(__INSTANCE__, __REG__) READ_REG(__INSTANCE__->__REG__)
/**
  * @}
  */

/**
  * @}
  */


/* Exported functions --------------------------------------------------------*/

/** @defgroup I2S_LL_Exported_Functions I2S Exported Functions
  * @{
  */

/** @defgroup I2S_LL_EF_Configuration Configuration
  * @{
  */

/**
  * @brief  Select I2S mode and Enable I2S peripheral
  * @rmtoll I2SCFGR      I2SMOD        LL_I2S_Enable\n
  *         I2SCFGR      I2SE          LL_I2S_Enable
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_Enable(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_I2SMOD | SPI_I2SCFGR_I2SE);
}

/**
  * @brief  Disable I2S peripheral
  * @rmtoll I2SCFGR      I2SE          LL_I2S_Disable
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_Disable(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_I2SMOD | SPI_I2SCFGR_I2SE);
}

/**
  * @brief  Check if I2S peripheral is enabled
  * @rmtoll I2SCFGR      I2SE          LL_I2S_IsEnabled
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsEnabled(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_I2SE) == (SPI_I2SCFGR_I2SE)) ? 1UL : 0UL);
}

/**
  * @brief  Set I2S data frame length
  * @rmtoll I2SCFGR      DATLEN        LL_I2S_SetDataFormat\n
  *         I2SCFGR      CHLEN         LL_I2S_SetDataFormat
  * @param  SPIx SPI Instance
  * @param  DataFormat This parameter can be one of the following values:
  *         @arg @ref LL_I2S_DATAFORMAT_16B
  *         @arg @ref LL_I2S_DATAFORMAT_16B_EXTENDED
  *         @arg @ref LL_I2S_DATAFORMAT_24B
  *         @arg @ref LL_I2S_DATAFORMAT_32B
  * @retval None
  */
__STATIC_INLINE void LL_I2S_SetDataFormat(SPI_TypeDef *SPIx, uint32_t DataFormat)
{
  MODIFY_REG(SPIx->I2SCFGR, SPI_I2SCFGR_DATLEN | SPI_I2SCFGR_CHLEN, DataFormat);
}

/**
  * @brief  Get I2S data frame length
  * @rmtoll I2SCFGR      DATLEN        LL_I2S_GetDataFormat\n
  *         I2SCFGR      CHLEN         LL_I2S_GetDataFormat
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_I2S_DATAFORMAT_16B
  *         @arg @ref LL_I2S_DATAFORMAT_16B_EXTENDED
  *         @arg @ref LL_I2S_DATAFORMAT_24B
  *         @arg @ref LL_I2S_DATAFORMAT_32B
  */
__STATIC_INLINE uint32_t LL_I2S_GetDataFormat(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_DATLEN | SPI_I2SCFGR_CHLEN));
}

/**
  * @brief  Set I2S clock polarity
  * @rmtoll I2SCFGR      CKPOL         LL_I2S_SetClockPolarity
  * @param  SPIx SPI Instance
  * @param  ClockPolarity This parameter can be one of the following values:
  *         @arg @ref LL_I2S_POLARITY_LOW
  *         @arg @ref LL_I2S_POLARITY_HIGH
  * @retval None
  */
__STATIC_INLINE void LL_I2S_SetClockPolarity(SPI_TypeDef *SPIx, uint32_t ClockPolarity)
{
  SET_BIT(SPIx->I2SCFGR, ClockPolarity);
}

/**
  * @brief  Get I2S clock polarity
  * @rmtoll I2SCFGR      CKPOL         LL_I2S_GetClockPolarity
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_I2S_POLARITY_LOW
  *         @arg @ref LL_I2S_POLARITY_HIGH
  */
__STATIC_INLINE uint32_t LL_I2S_GetClockPolarity(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_CKPOL));
}

/**
  * @brief  Set I2S standard protocol
  * @rmtoll I2SCFGR      I2SSTD        LL_I2S_SetStandard\n
  *         I2SCFGR      PCMSYNC       LL_I2S_SetStandard
  * @param  SPIx SPI Instance
  * @param  Standard This parameter can be one of the following values:
  *         @arg @ref LL_I2S_STANDARD_PHILIPS
  *         @arg @ref LL_I2S_STANDARD_MSB
  *         @arg @ref LL_I2S_STANDARD_LSB
  *         @arg @ref LL_I2S_STANDARD_PCM_SHORT
  *         @arg @ref LL_I2S_STANDARD_PCM_LONG
  * @retval None
  */
__STATIC_INLINE void LL_I2S_SetStandard(SPI_TypeDef *SPIx, uint32_t Standard)
{
  MODIFY_REG(SPIx->I2SCFGR, SPI_I2SCFGR_I2SSTD | SPI_I2SCFGR_PCMSYNC, Standard);
}

/**
  * @brief  Get I2S standard protocol
  * @rmtoll I2SCFGR      I2SSTD        LL_I2S_GetStandard\n
  *         I2SCFGR      PCMSYNC       LL_I2S_GetStandard
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_I2S_STANDARD_PHILIPS
  *         @arg @ref LL_I2S_STANDARD_MSB
  *         @arg @ref LL_I2S_STANDARD_LSB
  *         @arg @ref LL_I2S_STANDARD_PCM_SHORT
  *         @arg @ref LL_I2S_STANDARD_PCM_LONG
  */
__STATIC_INLINE uint32_t LL_I2S_GetStandard(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_I2SSTD | SPI_I2SCFGR_PCMSYNC));
}

/**
  * @brief  Set I2S transfer mode
  * @rmtoll I2SCFGR      I2SCFG        LL_I2S_SetTransferMode
  * @param  SPIx SPI Instance
  * @param  Mode This parameter can be one of the following values:
  *         @arg @ref LL_I2S_MODE_SLAVE_TX
  *         @arg @ref LL_I2S_MODE_SLAVE_RX
  *         @arg @ref LL_I2S_MODE_MASTER_TX
  *         @arg @ref LL_I2S_MODE_MASTER_RX
  * @retval None
  */
__STATIC_INLINE void LL_I2S_SetTransferMode(SPI_TypeDef *SPIx, uint32_t Mode)
{
  MODIFY_REG(SPIx->I2SCFGR, SPI_I2SCFGR_I2SCFG, Mode);
}

/**
  * @brief  Get I2S transfer mode
  * @rmtoll I2SCFGR      I2SCFG        LL_I2S_GetTransferMode
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_I2S_MODE_SLAVE_TX
  *         @arg @ref LL_I2S_MODE_SLAVE_RX
  *         @arg @ref LL_I2S_MODE_MASTER_TX
  *         @arg @ref LL_I2S_MODE_MASTER_RX
  */
__STATIC_INLINE uint32_t LL_I2S_GetTransferMode(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_I2SCFG));
}

/**
  * @brief  Set I2S linear prescaler
  * @rmtoll I2SPR        I2SDIV        LL_I2S_SetPrescalerLinear
  * @param  SPIx SPI Instance
  * @param  PrescalerLinear Value between Min_Data=0x02 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_I2S_SetPrescalerLinear(SPI_TypeDef *SPIx, uint8_t PrescalerLinear)
{
  MODIFY_REG(SPIx->I2SPR, SPI_I2SPR_I2SDIV, PrescalerLinear);
}

/**
  * @brief  Get I2S linear prescaler
  * @rmtoll I2SPR        I2SDIV        LL_I2S_GetPrescalerLinear
  * @param  SPIx SPI Instance
  * @retval PrescalerLinear Value between Min_Data=0x02 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_I2S_GetPrescalerLinear(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->I2SPR, SPI_I2SPR_I2SDIV));
}

/**
  * @brief  Set I2S parity prescaler
  * @rmtoll I2SPR        ODD           LL_I2S_SetPrescalerParity
  * @param  SPIx SPI Instance
  * @param  PrescalerParity This parameter can be one of the following values:
  *         @arg @ref LL_I2S_PRESCALER_PARITY_EVEN
  *         @arg @ref LL_I2S_PRESCALER_PARITY_ODD
  * @retval None
  */
__STATIC_INLINE void LL_I2S_SetPrescalerParity(SPI_TypeDef *SPIx, uint32_t PrescalerParity)
{
  MODIFY_REG(SPIx->I2SPR, SPI_I2SPR_ODD, PrescalerParity << 8U);
}

/**
  * @brief  Get I2S parity prescaler
  * @rmtoll I2SPR        ODD           LL_I2S_GetPrescalerParity
  * @param  SPIx SPI Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_I2S_PRESCALER_PARITY_EVEN
  *         @arg @ref LL_I2S_PRESCALER_PARITY_ODD
  */
__STATIC_INLINE uint32_t LL_I2S_GetPrescalerParity(SPI_TypeDef *SPIx)
{
  return (uint32_t)(READ_BIT(SPIx->I2SPR, SPI_I2SPR_ODD) >> 8U);
}

/**
  * @brief  Enable the master clock output (Pin MCK)
  * @rmtoll I2SPR        MCKOE         LL_I2S_EnableMasterClock
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_EnableMasterClock(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->I2SPR, SPI_I2SPR_MCKOE);
}

/**
  * @brief  Disable the master clock output (Pin MCK)
  * @rmtoll I2SPR        MCKOE         LL_I2S_DisableMasterClock
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_DisableMasterClock(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->I2SPR, SPI_I2SPR_MCKOE);
}

/**
  * @brief  Check if the master clock output (Pin MCK) is enabled
  * @rmtoll I2SPR        MCKOE         LL_I2S_IsEnabledMasterClock
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsEnabledMasterClock(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->I2SPR, SPI_I2SPR_MCKOE) == (SPI_I2SPR_MCKOE)) ? 1UL : 0UL);
}

#if defined(SPI_I2SCFGR_ASTRTEN)
/**
  * @brief  Enable asynchronous start
  * @rmtoll I2SCFGR      ASTRTEN       LL_I2S_EnableAsyncStart
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_EnableAsyncStart(SPI_TypeDef *SPIx)
{
  SET_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_ASTRTEN);
}

/**
  * @brief  Disable  asynchronous start
  * @rmtoll I2SCFGR      ASTRTEN       LL_I2S_DisableAsyncStart
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_DisableAsyncStart(SPI_TypeDef *SPIx)
{
  CLEAR_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_ASTRTEN);
}

/**
  * @brief  Check if asynchronous start is enabled
  * @rmtoll I2SCFGR      ASTRTEN       LL_I2S_IsEnabledAsyncStart
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsEnabledAsyncStart(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->I2SCFGR, SPI_I2SCFGR_ASTRTEN) == (SPI_I2SCFGR_ASTRTEN)) ? 1UL : 0UL);
}
#endif /* SPI_I2SCFGR_ASTRTEN */

/**
  * @}
  */

/** @defgroup I2S_LL_EF_FLAG FLAG Management
  * @{
  */

/**
  * @brief  Check if Rx buffer is not empty
  * @rmtoll SR           RXNE          LL_I2S_IsActiveFlag_RXNE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsActiveFlag_RXNE(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsActiveFlag_RXNE(SPIx);
}

/**
  * @brief  Check if Tx buffer is empty
  * @rmtoll SR           TXE           LL_I2S_IsActiveFlag_TXE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsActiveFlag_TXE(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsActiveFlag_TXE(SPIx);
}

/**
  * @brief  Get busy flag
  * @rmtoll SR           BSY           LL_I2S_IsActiveFlag_BSY
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsActiveFlag_BSY(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsActiveFlag_BSY(SPIx);
}

/**
  * @brief  Get overrun error flag
  * @rmtoll SR           OVR           LL_I2S_IsActiveFlag_OVR
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsActiveFlag_OVR(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsActiveFlag_OVR(SPIx);
}

/**
  * @brief  Get underrun error flag
  * @rmtoll SR           UDR           LL_I2S_IsActiveFlag_UDR
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsActiveFlag_UDR(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->SR, SPI_SR_UDR) == (SPI_SR_UDR)) ? 1UL : 0UL);
}

/**
  * @brief  Get frame format error flag
  * @rmtoll SR           FRE           LL_I2S_IsActiveFlag_FRE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsActiveFlag_FRE(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsActiveFlag_FRE(SPIx);
}

/**
  * @brief  Get channel side flag.
  * @note   0: Channel Left has to be transmitted or has been received\n
  *         1: Channel Right has to be transmitted or has been received\n
  *         It has no significance in PCM mode.
  * @rmtoll SR           CHSIDE        LL_I2S_IsActiveFlag_CHSIDE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsActiveFlag_CHSIDE(SPI_TypeDef *SPIx)
{
  return ((READ_BIT(SPIx->SR, SPI_SR_CHSIDE) == (SPI_SR_CHSIDE)) ? 1UL : 0UL);
}

/**
  * @brief  Clear overrun error flag
  * @rmtoll SR           OVR           LL_I2S_ClearFlag_OVR
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_ClearFlag_OVR(SPI_TypeDef *SPIx)
{
  LL_SPI_ClearFlag_OVR(SPIx);
}

/**
  * @brief  Clear underrun error flag
  * @rmtoll SR           UDR           LL_I2S_ClearFlag_UDR
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_ClearFlag_UDR(SPI_TypeDef *SPIx)
{
  __IO uint32_t tmpreg;
  tmpreg = SPIx->SR;
  (void)tmpreg;
}

/**
  * @brief  Clear frame format error flag
  * @rmtoll SR           FRE           LL_I2S_ClearFlag_FRE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_ClearFlag_FRE(SPI_TypeDef *SPIx)
{
  LL_SPI_ClearFlag_FRE(SPIx);
}

/**
  * @}
  */

/** @defgroup I2S_LL_EF_IT Interrupt Management
  * @{
  */

/**
  * @brief  Enable error IT
  * @note   This bit controls the generation of an interrupt when an error condition occurs (OVR, UDR and FRE in I2S mode).
  * @rmtoll CR2          ERRIE         LL_I2S_EnableIT_ERR
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_EnableIT_ERR(SPI_TypeDef *SPIx)
{
  LL_SPI_EnableIT_ERR(SPIx);
}

/**
  * @brief  Enable Rx buffer not empty IT
  * @rmtoll CR2          RXNEIE        LL_I2S_EnableIT_RXNE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_EnableIT_RXNE(SPI_TypeDef *SPIx)
{
  LL_SPI_EnableIT_RXNE(SPIx);
}

/**
  * @brief  Enable Tx buffer empty IT
  * @rmtoll CR2          TXEIE         LL_I2S_EnableIT_TXE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_EnableIT_TXE(SPI_TypeDef *SPIx)
{
  LL_SPI_EnableIT_TXE(SPIx);
}

/**
  * @brief  Disable error IT
  * @note   This bit controls the generation of an interrupt when an error condition occurs (OVR, UDR and FRE in I2S mode).
  * @rmtoll CR2          ERRIE         LL_I2S_DisableIT_ERR
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_DisableIT_ERR(SPI_TypeDef *SPIx)
{
  LL_SPI_DisableIT_ERR(SPIx);
}

/**
  * @brief  Disable Rx buffer not empty IT
  * @rmtoll CR2          RXNEIE        LL_I2S_DisableIT_RXNE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_DisableIT_RXNE(SPI_TypeDef *SPIx)
{
  LL_SPI_DisableIT_RXNE(SPIx);
}

/**
  * @brief  Disable Tx buffer empty IT
  * @rmtoll CR2          TXEIE         LL_I2S_DisableIT_TXE
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_DisableIT_TXE(SPI_TypeDef *SPIx)
{
  LL_SPI_DisableIT_TXE(SPIx);
}

/**
  * @brief  Check if ERR IT is enabled
  * @rmtoll CR2          ERRIE         LL_I2S_IsEnabledIT_ERR
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsEnabledIT_ERR(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsEnabledIT_ERR(SPIx);
}

/**
  * @brief  Check if RXNE IT is enabled
  * @rmtoll CR2          RXNEIE        LL_I2S_IsEnabledIT_RXNE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsEnabledIT_RXNE(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsEnabledIT_RXNE(SPIx);
}

/**
  * @brief  Check if TXE IT is enabled
  * @rmtoll CR2          TXEIE         LL_I2S_IsEnabledIT_TXE
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsEnabledIT_TXE(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsEnabledIT_TXE(SPIx);
}

/**
  * @}
  */

/** @defgroup I2S_LL_EF_DMA DMA Management
  * @{
  */

/**
  * @brief  Enable DMA Rx
  * @rmtoll CR2          RXDMAEN       LL_I2S_EnableDMAReq_RX
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_EnableDMAReq_RX(SPI_TypeDef *SPIx)
{
  LL_SPI_EnableDMAReq_RX(SPIx);
}

/**
  * @brief  Disable DMA Rx
  * @rmtoll CR2          RXDMAEN       LL_I2S_DisableDMAReq_RX
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_DisableDMAReq_RX(SPI_TypeDef *SPIx)
{
  LL_SPI_DisableDMAReq_RX(SPIx);
}

/**
  * @brief  Check if DMA Rx is enabled
  * @rmtoll CR2          RXDMAEN       LL_I2S_IsEnabledDMAReq_RX
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsEnabledDMAReq_RX(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsEnabledDMAReq_RX(SPIx);
}

/**
  * @brief  Enable DMA Tx
  * @rmtoll CR2          TXDMAEN       LL_I2S_EnableDMAReq_TX
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_EnableDMAReq_TX(SPI_TypeDef *SPIx)
{
  LL_SPI_EnableDMAReq_TX(SPIx);
}

/**
  * @brief  Disable DMA Tx
  * @rmtoll CR2          TXDMAEN       LL_I2S_DisableDMAReq_TX
  * @param  SPIx SPI Instance
  * @retval None
  */
__STATIC_INLINE void LL_I2S_DisableDMAReq_TX(SPI_TypeDef *SPIx)
{
  LL_SPI_DisableDMAReq_TX(SPIx);
}

/**
  * @brief  Check if DMA Tx is enabled
  * @rmtoll CR2          TXDMAEN       LL_I2S_IsEnabledDMAReq_TX
  * @param  SPIx SPI Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2S_IsEnabledDMAReq_TX(SPI_TypeDef *SPIx)
{
  return LL_SPI_IsEnabledDMAReq_TX(SPIx);
}

/**
  * @}
  */

/** @defgroup I2S_LL_EF_DATA DATA Management
  * @{
  */

/**
  * @brief  Read 16-Bits in data register
  * @rmtoll DR           DR            LL_I2S_ReceiveData16
  * @param  SPIx SPI Instance
  * @retval RxData Value between Min_Data=0x0000 and Max_Data=0xFFFF
  */
__STATIC_INLINE uint16_t LL_I2S_ReceiveData16(SPI_TypeDef *SPIx)
{
  return LL_SPI_ReceiveData16(SPIx);
}

/**
  * @brief  Write 16-Bits in data register
  * @rmtoll DR           DR            LL_I2S_TransmitData16
  * @param  SPIx SPI Instance
  * @param  TxData Value between Min_Data=0x0000 and Max_Data=0xFFFF
  * @retval None
  */
__STATIC_INLINE void LL_I2S_TransmitData16(SPI_TypeDef *SPIx, uint16_t TxData)
{
  LL_SPI_TransmitData16(SPIx, TxData);
}

/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup I2S_LL_EF_Init Initialization and de-initialization functions
  * @{
  */

ErrorStatus LL_I2S_DeInit(SPI_TypeDef *SPIx);
ErrorStatus LL_I2S_Init(SPI_TypeDef *SPIx, LL_I2S_InitTypeDef *I2S_InitStruct);
void        LL_I2S_StructInit(LL_I2S_InitTypeDef *I2S_InitStruct);
void        LL_I2S_ConfigPrescaler(SPI_TypeDef *SPIx, uint32_t PrescalerLinear, uint32_t PrescalerParity);
#if defined (SPI_I2S_FULLDUPLEX_SUPPORT)
ErrorStatus LL_I2S_InitFullDuplex(SPI_TypeDef *I2Sxext, LL_I2S_InitTypeDef *I2S_InitStruct);
#endif /* SPI_I2S_FULLDUPLEX_SUPPORT */

/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

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

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_LL_SPI_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
