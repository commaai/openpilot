/**
  ******************************************************************************
  * @file    stm32f4xx_ll_fmpi2c.h
  * @author  MCD Application Team
  * @brief   Header file of FMPI2C LL module.
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
#ifndef STM32F4xx_LL_FMPI2C_H
#define STM32F4xx_LL_FMPI2C_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(FMPI2C_CR1_PE)
/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

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
/** @defgroup FMPI2C_LL_Private_Constants FMPI2C Private Constants
  * @{
  */
/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup FMPI2C_LL_Private_Macros FMPI2C Private Macros
  * @{
  */
/**
  * @}
  */
#endif /*USE_FULL_LL_DRIVER*/

/* Exported types ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup FMPI2C_LL_ES_INIT FMPI2C Exported Init structure
  * @{
  */
typedef struct
{
  uint32_t PeripheralMode;      /*!< Specifies the peripheral mode.
                                     This parameter can be a value of @ref FMPI2C_LL_EC_PERIPHERAL_MODE.

                                     This feature can be modified afterwards using unitary function
                                     @ref LL_FMPI2C_SetMode(). */

  uint32_t Timing;              /*!< Specifies the SDA setup, hold time and the SCL high, low period values.
                                     This parameter must be set by referring to the STM32CubeMX Tool and
                                     the helper macro @ref __LL_FMPI2C_CONVERT_TIMINGS().

                                     This feature can be modified afterwards using unitary function
                                     @ref LL_FMPI2C_SetTiming(). */

  uint32_t AnalogFilter;        /*!< Enables or disables analog noise filter.
                                     This parameter can be a value of @ref FMPI2C_LL_EC_ANALOGFILTER_SELECTION.

                                     This feature can be modified afterwards using unitary functions
                                     @ref LL_FMPI2C_EnableAnalogFilter() or LL_FMPI2C_DisableAnalogFilter(). */

  uint32_t DigitalFilter;       /*!< Configures the digital noise filter.
                                     This parameter can be a number between Min_Data = 0x00 and Max_Data = 0x0F.

                                     This feature can be modified afterwards using unitary function
                                     @ref LL_FMPI2C_SetDigitalFilter(). */

  uint32_t OwnAddress1;         /*!< Specifies the device own address 1.
                                     This parameter must be a value between Min_Data = 0x00 and Max_Data = 0x3FF.

                                     This feature can be modified afterwards using unitary function
                                     @ref LL_FMPI2C_SetOwnAddress1(). */

  uint32_t TypeAcknowledge;     /*!< Specifies the ACKnowledge or Non ACKnowledge condition after the address receive
                                     match code or next received byte.
                                     This parameter can be a value of @ref FMPI2C_LL_EC_I2C_ACKNOWLEDGE.

                                     This feature can be modified afterwards using unitary function
                                     @ref LL_FMPI2C_AcknowledgeNextData(). */

  uint32_t OwnAddrSize;         /*!< Specifies the device own address 1 size (7-bit or 10-bit).
                                     This parameter can be a value of @ref FMPI2C_LL_EC_OWNADDRESS1.

                                     This feature can be modified afterwards using unitary function
                                     @ref LL_FMPI2C_SetOwnAddress1(). */
} LL_FMPI2C_InitTypeDef;
/**
  * @}
  */
#endif /*USE_FULL_LL_DRIVER*/

/* Exported constants --------------------------------------------------------*/
/** @defgroup FMPI2C_LL_Exported_Constants FMPI2C Exported Constants
  * @{
  */

/** @defgroup FMPI2C_LL_EC_CLEAR_FLAG Clear Flags Defines
  * @brief    Flags defines which can be used with LL_FMPI2C_WriteReg function
  * @{
  */
#define LL_FMPI2C_ICR_ADDRCF                   FMPI2C_ICR_ADDRCF          /*!< Address Matched flag   */
#define LL_FMPI2C_ICR_NACKCF                   FMPI2C_ICR_NACKCF          /*!< Not Acknowledge flag   */
#define LL_FMPI2C_ICR_STOPCF                   FMPI2C_ICR_STOPCF          /*!< Stop detection flag    */
#define LL_FMPI2C_ICR_BERRCF                   FMPI2C_ICR_BERRCF          /*!< Bus error flag         */
#define LL_FMPI2C_ICR_ARLOCF                   FMPI2C_ICR_ARLOCF          /*!< Arbitration Lost flag  */
#define LL_FMPI2C_ICR_OVRCF                    FMPI2C_ICR_OVRCF           /*!< Overrun/Underrun flag  */
#define LL_FMPI2C_ICR_PECCF                    FMPI2C_ICR_PECCF           /*!< PEC error flag         */
#define LL_FMPI2C_ICR_TIMOUTCF                 FMPI2C_ICR_TIMOUTCF        /*!< Timeout detection flag */
#define LL_FMPI2C_ICR_ALERTCF                  FMPI2C_ICR_ALERTCF         /*!< Alert flag             */
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_GET_FLAG Get Flags Defines
  * @brief    Flags defines which can be used with LL_FMPI2C_ReadReg function
  * @{
  */
#define LL_FMPI2C_ISR_TXE                      FMPI2C_ISR_TXE             /*!< Transmit data register empty        */
#define LL_FMPI2C_ISR_TXIS                     FMPI2C_ISR_TXIS            /*!< Transmit interrupt status           */
#define LL_FMPI2C_ISR_RXNE                     FMPI2C_ISR_RXNE            /*!< Receive data register not empty     */
#define LL_FMPI2C_ISR_ADDR                     FMPI2C_ISR_ADDR            /*!< Address matched (slave mode)        */
#define LL_FMPI2C_ISR_NACKF                    FMPI2C_ISR_NACKF           /*!< Not Acknowledge received flag       */
#define LL_FMPI2C_ISR_STOPF                    FMPI2C_ISR_STOPF           /*!< Stop detection flag                 */
#define LL_FMPI2C_ISR_TC                       FMPI2C_ISR_TC              /*!< Transfer Complete (master mode)     */
#define LL_FMPI2C_ISR_TCR                      FMPI2C_ISR_TCR             /*!< Transfer Complete Reload            */
#define LL_FMPI2C_ISR_BERR                     FMPI2C_ISR_BERR            /*!< Bus error                           */
#define LL_FMPI2C_ISR_ARLO                     FMPI2C_ISR_ARLO            /*!< Arbitration lost                    */
#define LL_FMPI2C_ISR_OVR                      FMPI2C_ISR_OVR             /*!< Overrun/Underrun (slave mode)       */
#define LL_FMPI2C_ISR_PECERR                   FMPI2C_ISR_PECERR          /*!< PEC Error in reception (SMBus mode) */
#define LL_FMPI2C_ISR_TIMEOUT                  FMPI2C_ISR_TIMEOUT         /*!< Timeout detection flag (SMBus mode) */
#define LL_FMPI2C_ISR_ALERT                    FMPI2C_ISR_ALERT           /*!< SMBus alert (SMBus mode)            */
#define LL_FMPI2C_ISR_BUSY                     FMPI2C_ISR_BUSY            /*!< Bus busy                            */
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_IT IT Defines
  * @brief    IT defines which can be used with LL_FMPI2C_ReadReg and  LL_FMPI2C_WriteReg functions
  * @{
  */
#define LL_FMPI2C_CR1_TXIE                     FMPI2C_CR1_TXIE            /*!< TX Interrupt enable                         */
#define LL_FMPI2C_CR1_RXIE                     FMPI2C_CR1_RXIE            /*!< RX Interrupt enable                         */
#define LL_FMPI2C_CR1_ADDRIE                   FMPI2C_CR1_ADDRIE          /*!< Address match Interrupt enable (slave only) */
#define LL_FMPI2C_CR1_NACKIE                   FMPI2C_CR1_NACKIE          /*!< Not acknowledge received Interrupt enable   */
#define LL_FMPI2C_CR1_STOPIE                   FMPI2C_CR1_STOPIE          /*!< STOP detection Interrupt enable             */
#define LL_FMPI2C_CR1_TCIE                     FMPI2C_CR1_TCIE            /*!< Transfer Complete interrupt enable          */
#define LL_FMPI2C_CR1_ERRIE                    FMPI2C_CR1_ERRIE           /*!< Error interrupts enable                     */
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_PERIPHERAL_MODE Peripheral Mode
  * @{
  */
#define LL_FMPI2C_MODE_I2C                    0x00000000U              /*!< FMPI2C Master or Slave mode                 */
#define LL_FMPI2C_MODE_SMBUS_HOST             FMPI2C_CR1_SMBHEN           /*!< SMBus Host address acknowledge           */
#define LL_FMPI2C_MODE_SMBUS_DEVICE           0x00000000U              /*!< SMBus Device default mode
                                                                         (Default address not acknowledge)        */
#define LL_FMPI2C_MODE_SMBUS_DEVICE_ARP       FMPI2C_CR1_SMBDEN           /*!< SMBus Device Default address acknowledge */
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_ANALOGFILTER_SELECTION Analog Filter Selection
  * @{
  */
#define LL_FMPI2C_ANALOGFILTER_ENABLE          0x00000000U             /*!< Analog filter is enabled.  */
#define LL_FMPI2C_ANALOGFILTER_DISABLE         FMPI2C_CR1_ANFOFF          /*!< Analog filter is disabled. */
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_ADDRESSING_MODE Master Addressing Mode
  * @{
  */
#define LL_FMPI2C_ADDRESSING_MODE_7BIT         0x00000000U              /*!< Master operates in 7-bit addressing mode. */
#define LL_FMPI2C_ADDRESSING_MODE_10BIT        FMPI2C_CR2_ADD10            /*!< Master operates in 10-bit addressing mode.*/
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_OWNADDRESS1 Own Address 1 Length
  * @{
  */
#define LL_FMPI2C_OWNADDRESS1_7BIT             0x00000000U             /*!< Own address 1 is a 7-bit address. */
#define LL_FMPI2C_OWNADDRESS1_10BIT            FMPI2C_OAR1_OA1MODE        /*!< Own address 1 is a 10-bit address.*/
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_OWNADDRESS2 Own Address 2 Masks
  * @{
  */
#define LL_FMPI2C_OWNADDRESS2_NOMASK           FMPI2C_OAR2_OA2NOMASK      /*!< Own Address2 No mask.                 */
#define LL_FMPI2C_OWNADDRESS2_MASK01           FMPI2C_OAR2_OA2MASK01      /*!< Only Address2 bits[7:2] are compared. */
#define LL_FMPI2C_OWNADDRESS2_MASK02           FMPI2C_OAR2_OA2MASK02      /*!< Only Address2 bits[7:3] are compared. */
#define LL_FMPI2C_OWNADDRESS2_MASK03           FMPI2C_OAR2_OA2MASK03      /*!< Only Address2 bits[7:4] are compared. */
#define LL_FMPI2C_OWNADDRESS2_MASK04           FMPI2C_OAR2_OA2MASK04      /*!< Only Address2 bits[7:5] are compared. */
#define LL_FMPI2C_OWNADDRESS2_MASK05           FMPI2C_OAR2_OA2MASK05      /*!< Only Address2 bits[7:6] are compared. */
#define LL_FMPI2C_OWNADDRESS2_MASK06           FMPI2C_OAR2_OA2MASK06      /*!< Only Address2 bits[7] are compared.   */
#define LL_FMPI2C_OWNADDRESS2_MASK07           FMPI2C_OAR2_OA2MASK07      /*!< No comparison is done.
                                                                         All Address2 are acknowledged.        */
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_I2C_ACKNOWLEDGE Acknowledge Generation
  * @{
  */
#define LL_FMPI2C_ACK                          0x00000000U              /*!< ACK is sent after current received byte. */
#define LL_FMPI2C_NACK                         FMPI2C_CR2_NACK             /*!< NACK is sent after current received byte.*/
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_ADDRSLAVE Slave Address Length
  * @{
  */
#define LL_FMPI2C_ADDRSLAVE_7BIT               0x00000000U              /*!< Slave Address in 7-bit. */
#define LL_FMPI2C_ADDRSLAVE_10BIT              FMPI2C_CR2_ADD10            /*!< Slave Address in 10-bit.*/
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_REQUEST Transfer Request Direction
  * @{
  */
#define LL_FMPI2C_REQUEST_WRITE                0x00000000U              /*!< Master request a write transfer. */
#define LL_FMPI2C_REQUEST_READ                 FMPI2C_CR2_RD_WRN           /*!< Master request a read transfer.  */
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_MODE Transfer End Mode
  * @{
  */
#define LL_FMPI2C_MODE_RELOAD                  FMPI2C_CR2_RELOAD           /*!< Enable FMPI2C Reload mode.     */
#define LL_FMPI2C_MODE_AUTOEND                 FMPI2C_CR2_AUTOEND          /*!< Enable FMPI2C Automatic end mode
                                                                          with no HW PEC comparison.  */
#define LL_FMPI2C_MODE_SOFTEND                 0x00000000U              /*!< Enable FMPI2C Software end mode
                                                                          with no HW PEC comparison.  */
#define LL_FMPI2C_MODE_SMBUS_RELOAD            LL_FMPI2C_MODE_RELOAD       /*!< Enable FMPSMBUS Automatic end mode
                                                                          with HW PEC comparison.     */
#define LL_FMPI2C_MODE_SMBUS_AUTOEND_NO_PEC    LL_FMPI2C_MODE_AUTOEND      /*!< Enable FMPSMBUS Automatic end mode
                                                                          with HW PEC comparison.     */
#define LL_FMPI2C_MODE_SMBUS_SOFTEND_NO_PEC    LL_FMPI2C_MODE_SOFTEND      /*!< Enable FMPSMBUS Software end mode
                                                                          with HW PEC comparison.     */
#define LL_FMPI2C_MODE_SMBUS_AUTOEND_WITH_PEC  (uint32_t)(LL_FMPI2C_MODE_AUTOEND | FMPI2C_CR2_PECBYTE)
/*!< Enable FMPSMBUS Automatic end mode with HW PEC comparison.   */
#define LL_FMPI2C_MODE_SMBUS_SOFTEND_WITH_PEC  (uint32_t)(LL_FMPI2C_MODE_SOFTEND | FMPI2C_CR2_PECBYTE)
/*!< Enable FMPSMBUS Software end mode with HW PEC comparison.    */
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_GENERATE Start And Stop Generation
  * @{
  */
#define LL_FMPI2C_GENERATE_NOSTARTSTOP         0x00000000U
/*!< Don't Generate Stop and Start condition. */
#define LL_FMPI2C_GENERATE_STOP                (uint32_t)(0x80000000U | FMPI2C_CR2_STOP)
/*!< Generate Stop condition (Size should be set to 0).      */
#define LL_FMPI2C_GENERATE_START_READ          (uint32_t)(0x80000000U | FMPI2C_CR2_START | FMPI2C_CR2_RD_WRN)
/*!< Generate Start for read request. */
#define LL_FMPI2C_GENERATE_START_WRITE         (uint32_t)(0x80000000U | FMPI2C_CR2_START)
/*!< Generate Start for write request. */
#define LL_FMPI2C_GENERATE_RESTART_7BIT_READ   (uint32_t)(0x80000000U | FMPI2C_CR2_START | FMPI2C_CR2_RD_WRN)
/*!< Generate Restart for read request, slave 7Bit address.  */
#define LL_FMPI2C_GENERATE_RESTART_7BIT_WRITE  (uint32_t)(0x80000000U | FMPI2C_CR2_START)
/*!< Generate Restart for write request, slave 7Bit address. */
#define LL_FMPI2C_GENERATE_RESTART_10BIT_READ  (uint32_t)(0x80000000U | FMPI2C_CR2_START | \
                                                       FMPI2C_CR2_RD_WRN | FMPI2C_CR2_HEAD10R)
/*!< Generate Restart for read request, slave 10Bit address. */
#define LL_FMPI2C_GENERATE_RESTART_10BIT_WRITE (uint32_t)(0x80000000U | FMPI2C_CR2_START)
/*!< Generate Restart for write request, slave 10Bit address.*/
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_DIRECTION Read Write Direction
  * @{
  */
#define LL_FMPI2C_DIRECTION_WRITE              0x00000000U              /*!< Write transfer request by master,
                                                                          slave enters receiver mode.  */
#define LL_FMPI2C_DIRECTION_READ               FMPI2C_ISR_DIR              /*!< Read transfer request by master,
                                                                          slave enters transmitter mode.*/
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_DMA_REG_DATA DMA Register Data
  * @{
  */
#define LL_FMPI2C_DMA_REG_DATA_TRANSMIT        0x00000000U              /*!< Get address of data register used for
                                                                          transmission */
#define LL_FMPI2C_DMA_REG_DATA_RECEIVE         0x00000001U              /*!< Get address of data register used for
                                                                          reception */
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_SMBUS_TIMEOUTA_MODE SMBus TimeoutA Mode SCL SDA Timeout
  * @{
  */
#define LL_FMPI2C_FMPSMBUS_TIMEOUTA_MODE_SCL_LOW      0x00000000U          /*!< TimeoutA is used to detect
                                                                          SCL low level timeout.              */
#define LL_FMPI2C_FMPSMBUS_TIMEOUTA_MODE_SDA_SCL_HIGH FMPI2C_TIMEOUTR_TIDLE   /*!< TimeoutA is used to detect
                                                                          both SCL and SDA high level timeout.*/
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EC_SMBUS_TIMEOUT_SELECTION SMBus Timeout Selection
  * @{
  */
#define LL_FMPI2C_FMPSMBUS_TIMEOUTA               FMPI2C_TIMEOUTR_TIMOUTEN                 /*!< TimeoutA enable bit          */
#define LL_FMPI2C_FMPSMBUS_TIMEOUTB               FMPI2C_TIMEOUTR_TEXTEN                   /*!< TimeoutB (extended clock)
                                                                                       enable bit                   */
#define LL_FMPI2C_FMPSMBUS_ALL_TIMEOUT            (uint32_t)(FMPI2C_TIMEOUTR_TIMOUTEN | \
                                                       FMPI2C_TIMEOUTR_TEXTEN)       /*!< TimeoutA and TimeoutB
(extended clock) enable bits */
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup FMPI2C_LL_Exported_Macros FMPI2C Exported Macros
  * @{
  */

/** @defgroup FMPI2C_LL_EM_WRITE_READ Common Write and read registers Macros
  * @{
  */

/**
  * @brief  Write a value in FMPI2C register
  * @param  __INSTANCE__ FMPI2C Instance
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_FMPI2C_WriteReg(__INSTANCE__, __REG__, __VALUE__) WRITE_REG(__INSTANCE__->__REG__, (__VALUE__))

/**
  * @brief  Read a value in FMPI2C register
  * @param  __INSTANCE__ FMPI2C Instance
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_FMPI2C_ReadReg(__INSTANCE__, __REG__) READ_REG(__INSTANCE__->__REG__)
/**
  * @}
  */

/** @defgroup FMPI2C_LL_EM_CONVERT_TIMINGS Convert SDA SCL timings
  * @{
  */
/**
  * @brief  Configure the SDA setup, hold time and the SCL high, low period.
  * @param  __PRESCALER__ This parameter must be a value between  Min_Data=0 and Max_Data=0xF.
  * @param  __SETUP_TIME__ This parameter must be a value between Min_Data=0 and Max_Data=0xF.
                           (tscldel = (SCLDEL+1)xtpresc)
  * @param  __HOLD_TIME__  This parameter must be a value between Min_Data=0 and Max_Data=0xF.
                           (tsdadel = SDADELxtpresc)
  * @param  __SCLH_PERIOD__ This parameter must be a value between Min_Data=0 and Max_Data=0xFF.
                            (tsclh = (SCLH+1)xtpresc)
  * @param  __SCLL_PERIOD__ This parameter must be a value between  Min_Data=0 and Max_Data=0xFF.
                            (tscll = (SCLL+1)xtpresc)
  * @retval Value between Min_Data=0 and Max_Data=0xFFFFFFFF
  */
#define __LL_FMPI2C_CONVERT_TIMINGS(__PRESCALER__, __SETUP_TIME__, __HOLD_TIME__, __SCLH_PERIOD__, __SCLL_PERIOD__) \
  ((((uint32_t)(__PRESCALER__)    << FMPI2C_TIMINGR_PRESC_Pos)  & FMPI2C_TIMINGR_PRESC)   | \
   (((uint32_t)(__SETUP_TIME__)   << FMPI2C_TIMINGR_SCLDEL_Pos) & FMPI2C_TIMINGR_SCLDEL)  | \
   (((uint32_t)(__HOLD_TIME__)    << FMPI2C_TIMINGR_SDADEL_Pos) & FMPI2C_TIMINGR_SDADEL)  | \
   (((uint32_t)(__SCLH_PERIOD__)  << FMPI2C_TIMINGR_SCLH_Pos)   & FMPI2C_TIMINGR_SCLH)    | \
   (((uint32_t)(__SCLL_PERIOD__)  << FMPI2C_TIMINGR_SCLL_Pos)   & FMPI2C_TIMINGR_SCLL))
/**
  * @}
  */

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup FMPI2C_LL_Exported_Functions FMPI2C Exported Functions
  * @{
  */

/** @defgroup FMPI2C_LL_EF_Configuration Configuration
  * @{
  */

/**
  * @brief  Enable FMPI2C peripheral (PE = 1).
  * @rmtoll CR1          PE            LL_FMPI2C_Enable
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_Enable(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_PE);
}

/**
  * @brief  Disable FMPI2C peripheral (PE = 0).
  * @note   When PE = 0, the FMPI2C SCL and SDA lines are released.
  *         Internal state machines and status bits are put back to their reset value.
  *         When cleared, PE must be kept low for at least 3 APB clock cycles.
  * @rmtoll CR1          PE            LL_FMPI2C_Disable
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_Disable(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_PE);
}

/**
  * @brief  Check if the FMPI2C peripheral is enabled or disabled.
  * @rmtoll CR1          PE            LL_FMPI2C_IsEnabled
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabled(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_PE) == (FMPI2C_CR1_PE)) ? 1UL : 0UL);
}

/**
  * @brief  Configure Noise Filters (Analog and Digital).
  * @note   If the analog filter is also enabled, the digital filter is added to analog filter.
  *         The filters can only be programmed when the FMPI2C is disabled (PE = 0).
  * @rmtoll CR1          ANFOFF        LL_FMPI2C_ConfigFilters\n
  *         CR1          DNF           LL_FMPI2C_ConfigFilters
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  AnalogFilter This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_ANALOGFILTER_ENABLE
  *         @arg @ref LL_FMPI2C_ANALOGFILTER_DISABLE
  * @param  DigitalFilter This parameter must be a value between Min_Data=0x00 (Digital filter disabled)
                          and Max_Data=0x0F (Digital filter enabled and filtering capability up to 15*tfmpi2cclk).
  *         This parameter is used to configure the digital noise filter on SDA and SCL input.
  *         The digital filter will filter spikes with a length of up to DNF[3:0]*tfmpi2cclk.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ConfigFilters(FMPI2C_TypeDef *FMPI2Cx, uint32_t AnalogFilter, uint32_t DigitalFilter)
{
  MODIFY_REG(FMPI2Cx->CR1, FMPI2C_CR1_ANFOFF | FMPI2C_CR1_DNF, AnalogFilter | (DigitalFilter << FMPI2C_CR1_DNF_Pos));
}

/**
  * @brief  Configure Digital Noise Filter.
  * @note   If the analog filter is also enabled, the digital filter is added to analog filter.
  *         This filter can only be programmed when the FMPI2C is disabled (PE = 0).
  * @rmtoll CR1          DNF           LL_FMPI2C_SetDigitalFilter
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  DigitalFilter This parameter must be a value between Min_Data=0x00 (Digital filter disabled)
                          and Max_Data=0x0F (Digital filter enabled and filtering capability up to 15*tfmpi2cclk).
  *         This parameter is used to configure the digital noise filter on SDA and SCL input.
  *         The digital filter will filter spikes with a length of up to DNF[3:0]*tfmpi2cclk.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetDigitalFilter(FMPI2C_TypeDef *FMPI2Cx, uint32_t DigitalFilter)
{
  MODIFY_REG(FMPI2Cx->CR1, FMPI2C_CR1_DNF, DigitalFilter << FMPI2C_CR1_DNF_Pos);
}

/**
  * @brief  Get the current Digital Noise Filter configuration.
  * @rmtoll CR1          DNF           LL_FMPI2C_GetDigitalFilter
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x0 and Max_Data=0xF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetDigitalFilter(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_DNF) >> FMPI2C_CR1_DNF_Pos);
}

/**
  * @brief  Enable Analog Noise Filter.
  * @note   This filter can only be programmed when the FMPI2C is disabled (PE = 0).
  * @rmtoll CR1          ANFOFF        LL_FMPI2C_EnableAnalogFilter
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableAnalogFilter(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ANFOFF);
}

/**
  * @brief  Disable Analog Noise Filter.
  * @note   This filter can only be programmed when the FMPI2C is disabled (PE = 0).
  * @rmtoll CR1          ANFOFF        LL_FMPI2C_DisableAnalogFilter
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableAnalogFilter(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ANFOFF);
}

/**
  * @brief  Check if Analog Noise Filter is enabled or disabled.
  * @rmtoll CR1          ANFOFF        LL_FMPI2C_IsEnabledAnalogFilter
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledAnalogFilter(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ANFOFF) != (FMPI2C_CR1_ANFOFF)) ? 1UL : 0UL);
}

/**
  * @brief  Enable DMA transmission requests.
  * @rmtoll CR1          TXDMAEN       LL_FMPI2C_EnableDMAReq_TX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableDMAReq_TX(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_TXDMAEN);
}

/**
  * @brief  Disable DMA transmission requests.
  * @rmtoll CR1          TXDMAEN       LL_FMPI2C_DisableDMAReq_TX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableDMAReq_TX(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_TXDMAEN);
}

/**
  * @brief  Check if DMA transmission requests are enabled or disabled.
  * @rmtoll CR1          TXDMAEN       LL_FMPI2C_IsEnabledDMAReq_TX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledDMAReq_TX(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_TXDMAEN) == (FMPI2C_CR1_TXDMAEN)) ? 1UL : 0UL);
}

/**
  * @brief  Enable DMA reception requests.
  * @rmtoll CR1          RXDMAEN       LL_FMPI2C_EnableDMAReq_RX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableDMAReq_RX(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_RXDMAEN);
}

/**
  * @brief  Disable DMA reception requests.
  * @rmtoll CR1          RXDMAEN       LL_FMPI2C_DisableDMAReq_RX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableDMAReq_RX(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_RXDMAEN);
}

/**
  * @brief  Check if DMA reception requests are enabled or disabled.
  * @rmtoll CR1          RXDMAEN       LL_FMPI2C_IsEnabledDMAReq_RX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledDMAReq_RX(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_RXDMAEN) == (FMPI2C_CR1_RXDMAEN)) ? 1UL : 0UL);
}

/**
  * @brief  Get the data register address used for DMA transfer
  * @rmtoll TXDR         TXDATA        LL_FMPI2C_DMA_GetRegAddr\n
  *         RXDR         RXDATA        LL_FMPI2C_DMA_GetRegAddr
  * @param  FMPI2Cx FMPI2C Instance
  * @param  Direction This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_DMA_REG_DATA_TRANSMIT
  *         @arg @ref LL_FMPI2C_DMA_REG_DATA_RECEIVE
  * @retval Address of data register
  */
__STATIC_INLINE uint32_t LL_FMPI2C_DMA_GetRegAddr(FMPI2C_TypeDef *FMPI2Cx, uint32_t Direction)
{
  uint32_t data_reg_addr;

  if (Direction == LL_FMPI2C_DMA_REG_DATA_TRANSMIT)
  {
    /* return address of TXDR register */
    data_reg_addr = (uint32_t) &(FMPI2Cx->TXDR);
  }
  else
  {
    /* return address of RXDR register */
    data_reg_addr = (uint32_t) &(FMPI2Cx->RXDR);
  }

  return data_reg_addr;
}

/**
  * @brief  Enable Clock stretching.
  * @note   This bit can only be programmed when the FMPI2C is disabled (PE = 0).
  * @rmtoll CR1          NOSTRETCH     LL_FMPI2C_EnableClockStretching
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableClockStretching(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_NOSTRETCH);
}

/**
  * @brief  Disable Clock stretching.
  * @note   This bit can only be programmed when the FMPI2C is disabled (PE = 0).
  * @rmtoll CR1          NOSTRETCH     LL_FMPI2C_DisableClockStretching
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableClockStretching(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_NOSTRETCH);
}

/**
  * @brief  Check if Clock stretching is enabled or disabled.
  * @rmtoll CR1          NOSTRETCH     LL_FMPI2C_IsEnabledClockStretching
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledClockStretching(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_NOSTRETCH) != (FMPI2C_CR1_NOSTRETCH)) ? 1UL : 0UL);
}

/**
  * @brief  Enable hardware byte control in slave mode.
  * @rmtoll CR1          SBC           LL_FMPI2C_EnableSlaveByteControl
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableSlaveByteControl(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_SBC);
}

/**
  * @brief  Disable hardware byte control in slave mode.
  * @rmtoll CR1          SBC           LL_FMPI2C_DisableSlaveByteControl
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableSlaveByteControl(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_SBC);
}

/**
  * @brief  Check if hardware byte control in slave mode is enabled or disabled.
  * @rmtoll CR1          SBC           LL_FMPI2C_IsEnabledSlaveByteControl
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledSlaveByteControl(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_SBC) == (FMPI2C_CR1_SBC)) ? 1UL : 0UL);
}

/**
  * @brief  Enable General Call.
  * @note   When enabled the Address 0x00 is ACKed.
  * @rmtoll CR1          GCEN          LL_FMPI2C_EnableGeneralCall
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableGeneralCall(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_GCEN);
}

/**
  * @brief  Disable General Call.
  * @note   When disabled the Address 0x00 is NACKed.
  * @rmtoll CR1          GCEN          LL_FMPI2C_DisableGeneralCall
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableGeneralCall(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_GCEN);
}

/**
  * @brief  Check if General Call is enabled or disabled.
  * @rmtoll CR1          GCEN          LL_FMPI2C_IsEnabledGeneralCall
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledGeneralCall(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_GCEN) == (FMPI2C_CR1_GCEN)) ? 1UL : 0UL);
}

/**
  * @brief  Configure the Master to operate in 7-bit or 10-bit addressing mode.
  * @note   Changing this bit is not allowed, when the START bit is set.
  * @rmtoll CR2          ADD10         LL_FMPI2C_SetMasterAddressingMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  AddressingMode This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_ADDRESSING_MODE_7BIT
  *         @arg @ref LL_FMPI2C_ADDRESSING_MODE_10BIT
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetMasterAddressingMode(FMPI2C_TypeDef *FMPI2Cx, uint32_t AddressingMode)
{
  MODIFY_REG(FMPI2Cx->CR2, FMPI2C_CR2_ADD10, AddressingMode);
}

/**
  * @brief  Get the Master addressing mode.
  * @rmtoll CR2          ADD10         LL_FMPI2C_GetMasterAddressingMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_FMPI2C_ADDRESSING_MODE_7BIT
  *         @arg @ref LL_FMPI2C_ADDRESSING_MODE_10BIT
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetMasterAddressingMode(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->CR2, FMPI2C_CR2_ADD10));
}

/**
  * @brief  Set the Own Address1.
  * @rmtoll OAR1         OA1           LL_FMPI2C_SetOwnAddress1\n
  *         OAR1         OA1MODE       LL_FMPI2C_SetOwnAddress1
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  OwnAddress1 This parameter must be a value between Min_Data=0 and Max_Data=0x3FF.
  * @param  OwnAddrSize This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_OWNADDRESS1_7BIT
  *         @arg @ref LL_FMPI2C_OWNADDRESS1_10BIT
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetOwnAddress1(FMPI2C_TypeDef *FMPI2Cx, uint32_t OwnAddress1, uint32_t OwnAddrSize)
{
  MODIFY_REG(FMPI2Cx->OAR1, FMPI2C_OAR1_OA1 | FMPI2C_OAR1_OA1MODE, OwnAddress1 | OwnAddrSize);
}

/**
  * @brief  Enable acknowledge on Own Address1 match address.
  * @rmtoll OAR1         OA1EN         LL_FMPI2C_EnableOwnAddress1
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableOwnAddress1(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->OAR1, FMPI2C_OAR1_OA1EN);
}

/**
  * @brief  Disable acknowledge on Own Address1 match address.
  * @rmtoll OAR1         OA1EN         LL_FMPI2C_DisableOwnAddress1
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableOwnAddress1(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->OAR1, FMPI2C_OAR1_OA1EN);
}

/**
  * @brief  Check if Own Address1 acknowledge is enabled or disabled.
  * @rmtoll OAR1         OA1EN         LL_FMPI2C_IsEnabledOwnAddress1
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledOwnAddress1(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->OAR1, FMPI2C_OAR1_OA1EN) == (FMPI2C_OAR1_OA1EN)) ? 1UL : 0UL);
}

/**
  * @brief  Set the 7bits Own Address2.
  * @note   This action has no effect if own address2 is enabled.
  * @rmtoll OAR2         OA2           LL_FMPI2C_SetOwnAddress2\n
  *         OAR2         OA2MSK        LL_FMPI2C_SetOwnAddress2
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  OwnAddress2 Value between Min_Data=0 and Max_Data=0x7F.
  * @param  OwnAddrMask This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_OWNADDRESS2_NOMASK
  *         @arg @ref LL_FMPI2C_OWNADDRESS2_MASK01
  *         @arg @ref LL_FMPI2C_OWNADDRESS2_MASK02
  *         @arg @ref LL_FMPI2C_OWNADDRESS2_MASK03
  *         @arg @ref LL_FMPI2C_OWNADDRESS2_MASK04
  *         @arg @ref LL_FMPI2C_OWNADDRESS2_MASK05
  *         @arg @ref LL_FMPI2C_OWNADDRESS2_MASK06
  *         @arg @ref LL_FMPI2C_OWNADDRESS2_MASK07
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetOwnAddress2(FMPI2C_TypeDef *FMPI2Cx, uint32_t OwnAddress2, uint32_t OwnAddrMask)
{
  MODIFY_REG(FMPI2Cx->OAR2, FMPI2C_OAR2_OA2 | FMPI2C_OAR2_OA2MSK, OwnAddress2 | OwnAddrMask);
}

/**
  * @brief  Enable acknowledge on Own Address2 match address.
  * @rmtoll OAR2         OA2EN         LL_FMPI2C_EnableOwnAddress2
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableOwnAddress2(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->OAR2, FMPI2C_OAR2_OA2EN);
}

/**
  * @brief  Disable  acknowledge on Own Address2 match address.
  * @rmtoll OAR2         OA2EN         LL_FMPI2C_DisableOwnAddress2
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableOwnAddress2(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->OAR2, FMPI2C_OAR2_OA2EN);
}

/**
  * @brief  Check if Own Address1 acknowledge is enabled or disabled.
  * @rmtoll OAR2         OA2EN         LL_FMPI2C_IsEnabledOwnAddress2
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledOwnAddress2(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->OAR2, FMPI2C_OAR2_OA2EN) == (FMPI2C_OAR2_OA2EN)) ? 1UL : 0UL);
}

/**
  * @brief  Configure the SDA setup, hold time and the SCL high, low period.
  * @note   This bit can only be programmed when the FMPI2C is disabled (PE = 0).
  * @rmtoll TIMINGR      TIMINGR       LL_FMPI2C_SetTiming
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  Timing This parameter must be a value between Min_Data=0 and Max_Data=0xFFFFFFFF.
  * @note   This parameter is computed with the STM32CubeMX Tool.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetTiming(FMPI2C_TypeDef *FMPI2Cx, uint32_t Timing)
{
  WRITE_REG(FMPI2Cx->TIMINGR, Timing);
}

/**
  * @brief  Get the Timing Prescaler setting.
  * @rmtoll TIMINGR      PRESC         LL_FMPI2C_GetTimingPrescaler
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x0 and Max_Data=0xF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetTimingPrescaler(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->TIMINGR, FMPI2C_TIMINGR_PRESC) >> FMPI2C_TIMINGR_PRESC_Pos);
}

/**
  * @brief  Get the SCL low period setting.
  * @rmtoll TIMINGR      SCLL          LL_FMPI2C_GetClockLowPeriod
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x00 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetClockLowPeriod(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->TIMINGR, FMPI2C_TIMINGR_SCLL) >> FMPI2C_TIMINGR_SCLL_Pos);
}

/**
  * @brief  Get the SCL high period setting.
  * @rmtoll TIMINGR      SCLH          LL_FMPI2C_GetClockHighPeriod
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x00 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetClockHighPeriod(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->TIMINGR, FMPI2C_TIMINGR_SCLH) >> FMPI2C_TIMINGR_SCLH_Pos);
}

/**
  * @brief  Get the SDA hold time.
  * @rmtoll TIMINGR      SDADEL        LL_FMPI2C_GetDataHoldTime
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x0 and Max_Data=0xF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetDataHoldTime(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->TIMINGR, FMPI2C_TIMINGR_SDADEL) >> FMPI2C_TIMINGR_SDADEL_Pos);
}

/**
  * @brief  Get the SDA setup time.
  * @rmtoll TIMINGR      SCLDEL        LL_FMPI2C_GetDataSetupTime
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x0 and Max_Data=0xF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetDataSetupTime(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->TIMINGR, FMPI2C_TIMINGR_SCLDEL) >> FMPI2C_TIMINGR_SCLDEL_Pos);
}

/**
  * @brief  Configure peripheral mode.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll CR1          SMBHEN        LL_FMPI2C_SetMode\n
  *         CR1          SMBDEN        LL_FMPI2C_SetMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  PeripheralMode This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_MODE_I2C
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_HOST
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_DEVICE
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_DEVICE_ARP
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetMode(FMPI2C_TypeDef *FMPI2Cx, uint32_t PeripheralMode)
{
  MODIFY_REG(FMPI2Cx->CR1, FMPI2C_CR1_SMBHEN | FMPI2C_CR1_SMBDEN, PeripheralMode);
}

/**
  * @brief  Get peripheral mode.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll CR1          SMBHEN        LL_FMPI2C_GetMode\n
  *         CR1          SMBDEN        LL_FMPI2C_GetMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_FMPI2C_MODE_I2C
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_HOST
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_DEVICE
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_DEVICE_ARP
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetMode(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_SMBHEN | FMPI2C_CR1_SMBDEN));
}

/**
  * @brief  Enable SMBus alert (Host or Device mode)
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   SMBus Device mode:
  *         - SMBus Alert pin is drived low and
  *           Alert Response Address Header acknowledge is enabled.
  *         SMBus Host mode:
  *         - SMBus Alert pin management is supported.
  * @rmtoll CR1          ALERTEN       LL_FMPI2C_EnableSMBusAlert
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableSMBusAlert(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ALERTEN);
}

/**
  * @brief  Disable SMBus alert (Host or Device mode)
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   SMBus Device mode:
  *         - SMBus Alert pin is not drived (can be used as a standard GPIO) and
  *           Alert Response Address Header acknowledge is disabled.
  *         SMBus Host mode:
  *         - SMBus Alert pin management is not supported.
  * @rmtoll CR1          ALERTEN       LL_FMPI2C_DisableSMBusAlert
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableSMBusAlert(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ALERTEN);
}

/**
  * @brief  Check if SMBus alert (Host or Device mode) is enabled or disabled.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll CR1          ALERTEN       LL_FMPI2C_IsEnabledSMBusAlert
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledSMBusAlert(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ALERTEN) == (FMPI2C_CR1_ALERTEN)) ? 1UL : 0UL);
}

/**
  * @brief  Enable SMBus Packet Error Calculation (PEC).
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll CR1          PECEN         LL_FMPI2C_EnableSMBusPEC
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableSMBusPEC(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_PECEN);
}

/**
  * @brief  Disable SMBus Packet Error Calculation (PEC).
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll CR1          PECEN         LL_FMPI2C_DisableSMBusPEC
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableSMBusPEC(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_PECEN);
}

/**
  * @brief  Check if SMBus Packet Error Calculation (PEC) is enabled or disabled.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll CR1          PECEN         LL_FMPI2C_IsEnabledSMBusPEC
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledSMBusPEC(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_PECEN) == (FMPI2C_CR1_PECEN)) ? 1UL : 0UL);
}

/**
  * @brief  Configure the SMBus Clock Timeout.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   This configuration can only be programmed when associated Timeout is disabled (TimeoutA and/orTimeoutB).
  * @rmtoll TIMEOUTR     TIMEOUTA      LL_FMPI2C_ConfigSMBusTimeout\n
  *         TIMEOUTR     TIDLE         LL_FMPI2C_ConfigSMBusTimeout\n
  *         TIMEOUTR     TIMEOUTB      LL_FMPI2C_ConfigSMBusTimeout
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  TimeoutA This parameter must be a value between  Min_Data=0 and Max_Data=0xFFF.
  * @param  TimeoutAMode This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTA_MODE_SCL_LOW
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTA_MODE_SDA_SCL_HIGH
  * @param  TimeoutB
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ConfigSMBusTimeout(FMPI2C_TypeDef *FMPI2Cx, uint32_t TimeoutA, uint32_t TimeoutAMode,
                                               uint32_t TimeoutB)
{
  MODIFY_REG(FMPI2Cx->TIMEOUTR, FMPI2C_TIMEOUTR_TIMEOUTA | FMPI2C_TIMEOUTR_TIDLE | FMPI2C_TIMEOUTR_TIMEOUTB,
             TimeoutA | TimeoutAMode | (TimeoutB << FMPI2C_TIMEOUTR_TIMEOUTB_Pos));
}

/**
  * @brief  Configure the SMBus Clock TimeoutA (SCL low timeout or SCL and SDA high timeout depends on TimeoutA mode).
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   These bits can only be programmed when TimeoutA is disabled.
  * @rmtoll TIMEOUTR     TIMEOUTA      LL_FMPI2C_SetSMBusTimeoutA
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  TimeoutA This parameter must be a value between  Min_Data=0 and Max_Data=0xFFF.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetSMBusTimeoutA(FMPI2C_TypeDef *FMPI2Cx, uint32_t TimeoutA)
{
  WRITE_REG(FMPI2Cx->TIMEOUTR, TimeoutA);
}

/**
  * @brief  Get the SMBus Clock TimeoutA setting.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll TIMEOUTR     TIMEOUTA      LL_FMPI2C_GetSMBusTimeoutA
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0 and Max_Data=0xFFF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetSMBusTimeoutA(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->TIMEOUTR, FMPI2C_TIMEOUTR_TIMEOUTA));
}

/**
  * @brief  Set the SMBus Clock TimeoutA mode.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   This bit can only be programmed when TimeoutA is disabled.
  * @rmtoll TIMEOUTR     TIDLE         LL_FMPI2C_SetSMBusTimeoutAMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  TimeoutAMode This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTA_MODE_SCL_LOW
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTA_MODE_SDA_SCL_HIGH
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetSMBusTimeoutAMode(FMPI2C_TypeDef *FMPI2Cx, uint32_t TimeoutAMode)
{
  WRITE_REG(FMPI2Cx->TIMEOUTR, TimeoutAMode);
}

/**
  * @brief  Get the SMBus Clock TimeoutA mode.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll TIMEOUTR     TIDLE         LL_FMPI2C_GetSMBusTimeoutAMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTA_MODE_SCL_LOW
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTA_MODE_SDA_SCL_HIGH
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetSMBusTimeoutAMode(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->TIMEOUTR, FMPI2C_TIMEOUTR_TIDLE));
}

/**
  * @brief  Configure the SMBus Extended Cumulative Clock TimeoutB (Master or Slave mode).
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   These bits can only be programmed when TimeoutB is disabled.
  * @rmtoll TIMEOUTR     TIMEOUTB      LL_FMPI2C_SetSMBusTimeoutB
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  TimeoutB This parameter must be a value between  Min_Data=0 and Max_Data=0xFFF.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetSMBusTimeoutB(FMPI2C_TypeDef *FMPI2Cx, uint32_t TimeoutB)
{
  WRITE_REG(FMPI2Cx->TIMEOUTR, TimeoutB << FMPI2C_TIMEOUTR_TIMEOUTB_Pos);
}

/**
  * @brief  Get the SMBus Extended Cumulative Clock TimeoutB setting.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll TIMEOUTR     TIMEOUTB      LL_FMPI2C_GetSMBusTimeoutB
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0 and Max_Data=0xFFF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetSMBusTimeoutB(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->TIMEOUTR, FMPI2C_TIMEOUTR_TIMEOUTB) >> FMPI2C_TIMEOUTR_TIMEOUTB_Pos);
}

/**
  * @brief  Enable the SMBus Clock Timeout.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll TIMEOUTR     TIMOUTEN      LL_FMPI2C_EnableSMBusTimeout\n
  *         TIMEOUTR     TEXTEN        LL_FMPI2C_EnableSMBusTimeout
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  ClockTimeout This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTA
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTB
  *         @arg @ref LL_FMPI2C_FMPSMBUS_ALL_TIMEOUT
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableSMBusTimeout(FMPI2C_TypeDef *FMPI2Cx, uint32_t ClockTimeout)
{
  SET_BIT(FMPI2Cx->TIMEOUTR, ClockTimeout);
}

/**
  * @brief  Disable the SMBus Clock Timeout.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll TIMEOUTR     TIMOUTEN      LL_FMPI2C_DisableSMBusTimeout\n
  *         TIMEOUTR     TEXTEN        LL_FMPI2C_DisableSMBusTimeout
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  ClockTimeout This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTA
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTB
  *         @arg @ref LL_FMPI2C_FMPSMBUS_ALL_TIMEOUT
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableSMBusTimeout(FMPI2C_TypeDef *FMPI2Cx, uint32_t ClockTimeout)
{
  CLEAR_BIT(FMPI2Cx->TIMEOUTR, ClockTimeout);
}

/**
  * @brief  Check if the SMBus Clock Timeout is enabled or disabled.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll TIMEOUTR     TIMOUTEN      LL_FMPI2C_IsEnabledSMBusTimeout\n
  *         TIMEOUTR     TEXTEN        LL_FMPI2C_IsEnabledSMBusTimeout
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  ClockTimeout This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTA
  *         @arg @ref LL_FMPI2C_FMPSMBUS_TIMEOUTB
  *         @arg @ref LL_FMPI2C_FMPSMBUS_ALL_TIMEOUT
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledSMBusTimeout(FMPI2C_TypeDef *FMPI2Cx, uint32_t ClockTimeout)
{
  return ((READ_BIT(FMPI2Cx->TIMEOUTR, (FMPI2C_TIMEOUTR_TIMOUTEN | FMPI2C_TIMEOUTR_TEXTEN)) == \
                    (ClockTimeout)) ? 1UL : 0UL);
}

/**
  * @}
  */

/** @defgroup FMPI2C_LL_EF_IT_Management IT_Management
  * @{
  */

/**
  * @brief  Enable TXIS interrupt.
  * @rmtoll CR1          TXIE          LL_FMPI2C_EnableIT_TX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableIT_TX(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_TXIE);
}

/**
  * @brief  Disable TXIS interrupt.
  * @rmtoll CR1          TXIE          LL_FMPI2C_DisableIT_TX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableIT_TX(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_TXIE);
}

/**
  * @brief  Check if the TXIS Interrupt is enabled or disabled.
  * @rmtoll CR1          TXIE          LL_FMPI2C_IsEnabledIT_TX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledIT_TX(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_TXIE) == (FMPI2C_CR1_TXIE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable RXNE interrupt.
  * @rmtoll CR1          RXIE          LL_FMPI2C_EnableIT_RX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableIT_RX(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_RXIE);
}

/**
  * @brief  Disable RXNE interrupt.
  * @rmtoll CR1          RXIE          LL_FMPI2C_DisableIT_RX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableIT_RX(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_RXIE);
}

/**
  * @brief  Check if the RXNE Interrupt is enabled or disabled.
  * @rmtoll CR1          RXIE          LL_FMPI2C_IsEnabledIT_RX
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledIT_RX(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_RXIE) == (FMPI2C_CR1_RXIE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable Address match interrupt (slave mode only).
  * @rmtoll CR1          ADDRIE        LL_FMPI2C_EnableIT_ADDR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableIT_ADDR(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ADDRIE);
}

/**
  * @brief  Disable Address match interrupt (slave mode only).
  * @rmtoll CR1          ADDRIE        LL_FMPI2C_DisableIT_ADDR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableIT_ADDR(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ADDRIE);
}

/**
  * @brief  Check if Address match interrupt is enabled or disabled.
  * @rmtoll CR1          ADDRIE        LL_FMPI2C_IsEnabledIT_ADDR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledIT_ADDR(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ADDRIE) == (FMPI2C_CR1_ADDRIE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable Not acknowledge received interrupt.
  * @rmtoll CR1          NACKIE        LL_FMPI2C_EnableIT_NACK
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableIT_NACK(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_NACKIE);
}

/**
  * @brief  Disable Not acknowledge received interrupt.
  * @rmtoll CR1          NACKIE        LL_FMPI2C_DisableIT_NACK
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableIT_NACK(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_NACKIE);
}

/**
  * @brief  Check if Not acknowledge received interrupt is enabled or disabled.
  * @rmtoll CR1          NACKIE        LL_FMPI2C_IsEnabledIT_NACK
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledIT_NACK(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_NACKIE) == (FMPI2C_CR1_NACKIE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable STOP detection interrupt.
  * @rmtoll CR1          STOPIE        LL_FMPI2C_EnableIT_STOP
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableIT_STOP(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_STOPIE);
}

/**
  * @brief  Disable STOP detection interrupt.
  * @rmtoll CR1          STOPIE        LL_FMPI2C_DisableIT_STOP
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableIT_STOP(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_STOPIE);
}

/**
  * @brief  Check if STOP detection interrupt is enabled or disabled.
  * @rmtoll CR1          STOPIE        LL_FMPI2C_IsEnabledIT_STOP
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledIT_STOP(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_STOPIE) == (FMPI2C_CR1_STOPIE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable Transfer Complete interrupt.
  * @note   Any of these events will generate interrupt :
  *         Transfer Complete (TC)
  *         Transfer Complete Reload (TCR)
  * @rmtoll CR1          TCIE          LL_FMPI2C_EnableIT_TC
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableIT_TC(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_TCIE);
}

/**
  * @brief  Disable Transfer Complete interrupt.
  * @note   Any of these events will generate interrupt :
  *         Transfer Complete (TC)
  *         Transfer Complete Reload (TCR)
  * @rmtoll CR1          TCIE          LL_FMPI2C_DisableIT_TC
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableIT_TC(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_TCIE);
}

/**
  * @brief  Check if Transfer Complete interrupt is enabled or disabled.
  * @rmtoll CR1          TCIE          LL_FMPI2C_IsEnabledIT_TC
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledIT_TC(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_TCIE) == (FMPI2C_CR1_TCIE)) ? 1UL : 0UL);
}

/**
  * @brief  Enable Error interrupts.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   Any of these errors will generate interrupt :
  *         Arbitration Loss (ARLO)
  *         Bus Error detection (BERR)
  *         Overrun/Underrun (OVR)
  *         SMBus Timeout detection (TIMEOUT)
  *         SMBus PEC error detection (PECERR)
  *         SMBus Alert pin event detection (ALERT)
  * @rmtoll CR1          ERRIE         LL_FMPI2C_EnableIT_ERR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableIT_ERR(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ERRIE);
}

/**
  * @brief  Disable Error interrupts.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   Any of these errors will generate interrupt :
  *         Arbitration Loss (ARLO)
  *         Bus Error detection (BERR)
  *         Overrun/Underrun (OVR)
  *         SMBus Timeout detection (TIMEOUT)
  *         SMBus PEC error detection (PECERR)
  *         SMBus Alert pin event detection (ALERT)
  * @rmtoll CR1          ERRIE         LL_FMPI2C_DisableIT_ERR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableIT_ERR(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ERRIE);
}

/**
  * @brief  Check if Error interrupts are enabled or disabled.
  * @rmtoll CR1          ERRIE         LL_FMPI2C_IsEnabledIT_ERR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledIT_ERR(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR1, FMPI2C_CR1_ERRIE) == (FMPI2C_CR1_ERRIE)) ? 1UL : 0UL);
}

/**
  * @}
  */

/** @defgroup FMPI2C_LL_EF_FLAG_management FLAG_management
  * @{
  */

/**
  * @brief  Indicate the status of Transmit data register empty flag.
  * @note   RESET: When next data is written in Transmit data register.
  *         SET: When Transmit data register is empty.
  * @rmtoll ISR          TXE           LL_FMPI2C_IsActiveFlag_TXE
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_TXE(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_TXE) == (FMPI2C_ISR_TXE)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Transmit interrupt flag.
  * @note   RESET: When next data is written in Transmit data register.
  *         SET: When Transmit data register is empty.
  * @rmtoll ISR          TXIS          LL_FMPI2C_IsActiveFlag_TXIS
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_TXIS(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_TXIS) == (FMPI2C_ISR_TXIS)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Receive data register not empty flag.
  * @note   RESET: When Receive data register is read.
  *         SET: When the received data is copied in Receive data register.
  * @rmtoll ISR          RXNE          LL_FMPI2C_IsActiveFlag_RXNE
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_RXNE(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_RXNE) == (FMPI2C_ISR_RXNE)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Address matched flag (slave mode).
  * @note   RESET: Clear default value.
  *         SET: When the received slave address matched with one of the enabled slave address.
  * @rmtoll ISR          ADDR          LL_FMPI2C_IsActiveFlag_ADDR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_ADDR(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_ADDR) == (FMPI2C_ISR_ADDR)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Not Acknowledge received flag.
  * @note   RESET: Clear default value.
  *         SET: When a NACK is received after a byte transmission.
  * @rmtoll ISR          NACKF         LL_FMPI2C_IsActiveFlag_NACK
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_NACK(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_NACKF) == (FMPI2C_ISR_NACKF)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Stop detection flag.
  * @note   RESET: Clear default value.
  *         SET: When a Stop condition is detected.
  * @rmtoll ISR          STOPF         LL_FMPI2C_IsActiveFlag_STOP
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_STOP(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_STOPF) == (FMPI2C_ISR_STOPF)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Transfer complete flag (master mode).
  * @note   RESET: Clear default value.
  *         SET: When RELOAD=0, AUTOEND=0 and NBYTES date have been transferred.
  * @rmtoll ISR          TC            LL_FMPI2C_IsActiveFlag_TC
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_TC(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_TC) == (FMPI2C_ISR_TC)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Transfer complete flag (master mode).
  * @note   RESET: Clear default value.
  *         SET: When RELOAD=1 and NBYTES date have been transferred.
  * @rmtoll ISR          TCR           LL_FMPI2C_IsActiveFlag_TCR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_TCR(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_TCR) == (FMPI2C_ISR_TCR)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Bus error flag.
  * @note   RESET: Clear default value.
  *         SET: When a misplaced Start or Stop condition is detected.
  * @rmtoll ISR          BERR          LL_FMPI2C_IsActiveFlag_BERR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_BERR(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_BERR) == (FMPI2C_ISR_BERR)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Arbitration lost flag.
  * @note   RESET: Clear default value.
  *         SET: When arbitration lost.
  * @rmtoll ISR          ARLO          LL_FMPI2C_IsActiveFlag_ARLO
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_ARLO(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_ARLO) == (FMPI2C_ISR_ARLO)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Overrun/Underrun flag (slave mode).
  * @note   RESET: Clear default value.
  *         SET: When an overrun/underrun error occurs (Clock Stretching Disabled).
  * @rmtoll ISR          OVR           LL_FMPI2C_IsActiveFlag_OVR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_OVR(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_OVR) == (FMPI2C_ISR_OVR)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of SMBus PEC error flag in reception.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   RESET: Clear default value.
  *         SET: When the received PEC does not match with the PEC register content.
  * @rmtoll ISR          PECERR        LL_FMPI2C_IsActiveSMBusFlag_PECERR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveSMBusFlag_PECERR(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_PECERR) == (FMPI2C_ISR_PECERR)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of SMBus Timeout detection flag.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   RESET: Clear default value.
  *         SET: When a timeout or extended clock timeout occurs.
  * @rmtoll ISR          TIMEOUT       LL_FMPI2C_IsActiveSMBusFlag_TIMEOUT
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveSMBusFlag_TIMEOUT(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_TIMEOUT) == (FMPI2C_ISR_TIMEOUT)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of SMBus alert flag.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   RESET: Clear default value.
  *         SET: When SMBus host configuration, SMBus alert enabled and
  *              a falling edge event occurs on SMBA pin.
  * @rmtoll ISR          ALERT         LL_FMPI2C_IsActiveSMBusFlag_ALERT
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveSMBusFlag_ALERT(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_ALERT) == (FMPI2C_ISR_ALERT)) ? 1UL : 0UL);
}

/**
  * @brief  Indicate the status of Bus Busy flag.
  * @note   RESET: Clear default value.
  *         SET: When a Start condition is detected.
  * @rmtoll ISR          BUSY          LL_FMPI2C_IsActiveFlag_BUSY
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsActiveFlag_BUSY(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_BUSY) == (FMPI2C_ISR_BUSY)) ? 1UL : 0UL);
}

/**
  * @brief  Clear Address Matched flag.
  * @rmtoll ICR          ADDRCF        LL_FMPI2C_ClearFlag_ADDR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearFlag_ADDR(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->ICR, FMPI2C_ICR_ADDRCF);
}

/**
  * @brief  Clear Not Acknowledge flag.
  * @rmtoll ICR          NACKCF        LL_FMPI2C_ClearFlag_NACK
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearFlag_NACK(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->ICR, FMPI2C_ICR_NACKCF);
}

/**
  * @brief  Clear Stop detection flag.
  * @rmtoll ICR          STOPCF        LL_FMPI2C_ClearFlag_STOP
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearFlag_STOP(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->ICR, FMPI2C_ICR_STOPCF);
}

/**
  * @brief  Clear Transmit data register empty flag (TXE).
  * @note   This bit can be clear by software in order to flush the transmit data register (TXDR).
  * @rmtoll ISR          TXE           LL_FMPI2C_ClearFlag_TXE
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearFlag_TXE(FMPI2C_TypeDef *FMPI2Cx)
{
  WRITE_REG(FMPI2Cx->ISR, FMPI2C_ISR_TXE);
}

/**
  * @brief  Clear Bus error flag.
  * @rmtoll ICR          BERRCF        LL_FMPI2C_ClearFlag_BERR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearFlag_BERR(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->ICR, FMPI2C_ICR_BERRCF);
}

/**
  * @brief  Clear Arbitration lost flag.
  * @rmtoll ICR          ARLOCF        LL_FMPI2C_ClearFlag_ARLO
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearFlag_ARLO(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->ICR, FMPI2C_ICR_ARLOCF);
}

/**
  * @brief  Clear Overrun/Underrun flag.
  * @rmtoll ICR          OVRCF         LL_FMPI2C_ClearFlag_OVR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearFlag_OVR(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->ICR, FMPI2C_ICR_OVRCF);
}

/**
  * @brief  Clear SMBus PEC error flag.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll ICR          PECCF         LL_FMPI2C_ClearSMBusFlag_PECERR
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearSMBusFlag_PECERR(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->ICR, FMPI2C_ICR_PECCF);
}

/**
  * @brief  Clear SMBus Timeout detection flag.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll ICR          TIMOUTCF      LL_FMPI2C_ClearSMBusFlag_TIMEOUT
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearSMBusFlag_TIMEOUT(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->ICR, FMPI2C_ICR_TIMOUTCF);
}

/**
  * @brief  Clear SMBus Alert flag.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll ICR          ALERTCF       LL_FMPI2C_ClearSMBusFlag_ALERT
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_ClearSMBusFlag_ALERT(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->ICR, FMPI2C_ICR_ALERTCF);
}

/**
  * @}
  */

/** @defgroup FMPI2C_LL_EF_Data_Management Data_Management
  * @{
  */

/**
  * @brief  Enable automatic STOP condition generation (master mode).
  * @note   Automatic end mode : a STOP condition is automatically sent when NBYTES data are transferred.
  *         This bit has no effect in slave mode or when RELOAD bit is set.
  * @rmtoll CR2          AUTOEND       LL_FMPI2C_EnableAutoEndMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableAutoEndMode(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR2, FMPI2C_CR2_AUTOEND);
}

/**
  * @brief  Disable automatic STOP condition generation (master mode).
  * @note   Software end mode : TC flag is set when NBYTES data are transferre, stretching SCL low.
  * @rmtoll CR2          AUTOEND       LL_FMPI2C_DisableAutoEndMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableAutoEndMode(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR2, FMPI2C_CR2_AUTOEND);
}

/**
  * @brief  Check if automatic STOP condition is enabled or disabled.
  * @rmtoll CR2          AUTOEND       LL_FMPI2C_IsEnabledAutoEndMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledAutoEndMode(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR2, FMPI2C_CR2_AUTOEND) == (FMPI2C_CR2_AUTOEND)) ? 1UL : 0UL);
}

/**
  * @brief  Enable reload mode (master mode).
  * @note   The transfer is not completed after the NBYTES data transfer, NBYTES will be reloaded when TCR flag is set.
  * @rmtoll CR2          RELOAD       LL_FMPI2C_EnableReloadMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableReloadMode(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR2, FMPI2C_CR2_RELOAD);
}

/**
  * @brief  Disable reload mode (master mode).
  * @note   The transfer is completed after the NBYTES data transfer(STOP or RESTART will follow).
  * @rmtoll CR2          RELOAD       LL_FMPI2C_DisableReloadMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableReloadMode(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR2, FMPI2C_CR2_RELOAD);
}

/**
  * @brief  Check if reload mode is enabled or disabled.
  * @rmtoll CR2          RELOAD       LL_FMPI2C_IsEnabledReloadMode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledReloadMode(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR2, FMPI2C_CR2_RELOAD) == (FMPI2C_CR2_RELOAD)) ? 1UL : 0UL);
}

/**
  * @brief  Configure the number of bytes for transfer.
  * @note   Changing these bits when START bit is set is not allowed.
  * @rmtoll CR2          NBYTES           LL_FMPI2C_SetTransferSize
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  TransferSize This parameter must be a value between Min_Data=0x00 and Max_Data=0xFF.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetTransferSize(FMPI2C_TypeDef *FMPI2Cx, uint32_t TransferSize)
{
  MODIFY_REG(FMPI2Cx->CR2, FMPI2C_CR2_NBYTES, TransferSize << FMPI2C_CR2_NBYTES_Pos);
}

/**
  * @brief  Get the number of bytes configured for transfer.
  * @rmtoll CR2          NBYTES           LL_FMPI2C_GetTransferSize
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetTransferSize(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->CR2, FMPI2C_CR2_NBYTES) >> FMPI2C_CR2_NBYTES_Pos);
}

/**
  * @brief  Prepare the generation of a ACKnowledge or Non ACKnowledge condition after the address receive match code
            or next received byte.
  * @note   Usage in Slave mode only.
  * @rmtoll CR2          NACK          LL_FMPI2C_AcknowledgeNextData
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  TypeAcknowledge This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_ACK
  *         @arg @ref LL_FMPI2C_NACK
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_AcknowledgeNextData(FMPI2C_TypeDef *FMPI2Cx, uint32_t TypeAcknowledge)
{
  MODIFY_REG(FMPI2Cx->CR2, FMPI2C_CR2_NACK, TypeAcknowledge);
}

/**
  * @brief  Generate a START or RESTART condition
  * @note   The START bit can be set even if bus is BUSY or FMPI2C is in slave mode.
  *         This action has no effect when RELOAD is set.
  * @rmtoll CR2          START           LL_FMPI2C_GenerateStartCondition
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_GenerateStartCondition(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR2, FMPI2C_CR2_START);
}

/**
  * @brief  Generate a STOP condition after the current byte transfer (master mode).
  * @rmtoll CR2          STOP          LL_FMPI2C_GenerateStopCondition
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_GenerateStopCondition(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR2, FMPI2C_CR2_STOP);
}

/**
  * @brief  Enable automatic RESTART Read request condition for 10bit address header (master mode).
  * @note   The master sends the complete 10bit slave address read sequence :
  *         Start + 2 bytes 10bit address in Write direction + Restart + first 7 bits of 10bit address
            in Read direction.
  * @rmtoll CR2          HEAD10R       LL_FMPI2C_EnableAuto10BitRead
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableAuto10BitRead(FMPI2C_TypeDef *FMPI2Cx)
{
  CLEAR_BIT(FMPI2Cx->CR2, FMPI2C_CR2_HEAD10R);
}

/**
  * @brief  Disable automatic RESTART Read request condition for 10bit address header (master mode).
  * @note   The master only sends the first 7 bits of 10bit address in Read direction.
  * @rmtoll CR2          HEAD10R       LL_FMPI2C_DisableAuto10BitRead
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_DisableAuto10BitRead(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR2, FMPI2C_CR2_HEAD10R);
}

/**
  * @brief  Check if automatic RESTART Read request condition for 10bit address header is enabled or disabled.
  * @rmtoll CR2          HEAD10R       LL_FMPI2C_IsEnabledAuto10BitRead
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledAuto10BitRead(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR2, FMPI2C_CR2_HEAD10R) != (FMPI2C_CR2_HEAD10R)) ? 1UL : 0UL);
}

/**
  * @brief  Configure the transfer direction (master mode).
  * @note   Changing these bits when START bit is set is not allowed.
  * @rmtoll CR2          RD_WRN           LL_FMPI2C_SetTransferRequest
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  TransferRequest This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_REQUEST_WRITE
  *         @arg @ref LL_FMPI2C_REQUEST_READ
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetTransferRequest(FMPI2C_TypeDef *FMPI2Cx, uint32_t TransferRequest)
{
  MODIFY_REG(FMPI2Cx->CR2, FMPI2C_CR2_RD_WRN, TransferRequest);
}

/**
  * @brief  Get the transfer direction requested (master mode).
  * @rmtoll CR2          RD_WRN           LL_FMPI2C_GetTransferRequest
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_FMPI2C_REQUEST_WRITE
  *         @arg @ref LL_FMPI2C_REQUEST_READ
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetTransferRequest(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->CR2, FMPI2C_CR2_RD_WRN));
}

/**
  * @brief  Configure the slave address for transfer (master mode).
  * @note   Changing these bits when START bit is set is not allowed.
  * @rmtoll CR2          SADD           LL_FMPI2C_SetSlaveAddr
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  SlaveAddr This parameter must be a value between Min_Data=0x00 and Max_Data=0x3F.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_SetSlaveAddr(FMPI2C_TypeDef *FMPI2Cx, uint32_t SlaveAddr)
{
  MODIFY_REG(FMPI2Cx->CR2, FMPI2C_CR2_SADD, SlaveAddr);
}

/**
  * @brief  Get the slave address programmed for transfer.
  * @rmtoll CR2          SADD           LL_FMPI2C_GetSlaveAddr
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x0 and Max_Data=0x3F
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetSlaveAddr(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->CR2, FMPI2C_CR2_SADD));
}

/**
  * @brief  Handles FMPI2Cx communication when starting transfer or during transfer (TC or TCR flag are set).
  * @rmtoll CR2          SADD          LL_FMPI2C_HandleTransfer\n
  *         CR2          ADD10         LL_FMPI2C_HandleTransfer\n
  *         CR2          RD_WRN        LL_FMPI2C_HandleTransfer\n
  *         CR2          START         LL_FMPI2C_HandleTransfer\n
  *         CR2          STOP          LL_FMPI2C_HandleTransfer\n
  *         CR2          RELOAD        LL_FMPI2C_HandleTransfer\n
  *         CR2          NBYTES        LL_FMPI2C_HandleTransfer\n
  *         CR2          AUTOEND       LL_FMPI2C_HandleTransfer\n
  *         CR2          HEAD10R       LL_FMPI2C_HandleTransfer
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  SlaveAddr Specifies the slave address to be programmed.
  * @param  SlaveAddrSize This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_ADDRSLAVE_7BIT
  *         @arg @ref LL_FMPI2C_ADDRSLAVE_10BIT
  * @param  TransferSize Specifies the number of bytes to be programmed.
  *                       This parameter must be a value between Min_Data=0 and Max_Data=255.
  * @param  EndMode This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_MODE_RELOAD
  *         @arg @ref LL_FMPI2C_MODE_AUTOEND
  *         @arg @ref LL_FMPI2C_MODE_SOFTEND
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_RELOAD
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_AUTOEND_NO_PEC
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_SOFTEND_NO_PEC
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_AUTOEND_WITH_PEC
  *         @arg @ref LL_FMPI2C_MODE_SMBUS_SOFTEND_WITH_PEC
  * @param  Request This parameter can be one of the following values:
  *         @arg @ref LL_FMPI2C_GENERATE_NOSTARTSTOP
  *         @arg @ref LL_FMPI2C_GENERATE_STOP
  *         @arg @ref LL_FMPI2C_GENERATE_START_READ
  *         @arg @ref LL_FMPI2C_GENERATE_START_WRITE
  *         @arg @ref LL_FMPI2C_GENERATE_RESTART_7BIT_READ
  *         @arg @ref LL_FMPI2C_GENERATE_RESTART_7BIT_WRITE
  *         @arg @ref LL_FMPI2C_GENERATE_RESTART_10BIT_READ
  *         @arg @ref LL_FMPI2C_GENERATE_RESTART_10BIT_WRITE
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_HandleTransfer(FMPI2C_TypeDef *FMPI2Cx, uint32_t SlaveAddr, uint32_t SlaveAddrSize,
                                           uint32_t TransferSize, uint32_t EndMode, uint32_t Request)
{
  MODIFY_REG(FMPI2Cx->CR2, FMPI2C_CR2_SADD | FMPI2C_CR2_ADD10 |
             (FMPI2C_CR2_RD_WRN & (uint32_t)(Request >> (31U - FMPI2C_CR2_RD_WRN_Pos))) |
             FMPI2C_CR2_START | FMPI2C_CR2_STOP | FMPI2C_CR2_RELOAD |
             FMPI2C_CR2_NBYTES | FMPI2C_CR2_AUTOEND | FMPI2C_CR2_HEAD10R,
             SlaveAddr | SlaveAddrSize | (TransferSize << FMPI2C_CR2_NBYTES_Pos) | EndMode | Request);
}

/**
  * @brief  Indicate the value of transfer direction (slave mode).
  * @note   RESET: Write transfer, Slave enters in receiver mode.
  *         SET: Read transfer, Slave enters in transmitter mode.
  * @rmtoll ISR          DIR           LL_FMPI2C_GetTransferDirection
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_FMPI2C_DIRECTION_WRITE
  *         @arg @ref LL_FMPI2C_DIRECTION_READ
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetTransferDirection(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_DIR));
}

/**
  * @brief  Return the slave matched address.
  * @rmtoll ISR          ADDCODE       LL_FMPI2C_GetAddressMatchCode
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x00 and Max_Data=0x3F
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetAddressMatchCode(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->ISR, FMPI2C_ISR_ADDCODE) >> FMPI2C_ISR_ADDCODE_Pos << 1);
}

/**
  * @brief  Enable internal comparison of the SMBus Packet Error byte (transmission or reception mode).
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @note   This feature is cleared by hardware when the PEC byte is transferred, or when a STOP condition
            or an Address Matched is received.
  *         This bit has no effect when RELOAD bit is set.
  *         This bit has no effect in device mode when SBC bit is not set.
  * @rmtoll CR2          PECBYTE       LL_FMPI2C_EnableSMBusPECCompare
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_EnableSMBusPECCompare(FMPI2C_TypeDef *FMPI2Cx)
{
  SET_BIT(FMPI2Cx->CR2, FMPI2C_CR2_PECBYTE);
}

/**
  * @brief  Check if the SMBus Packet Error byte internal comparison is requested or not.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll CR2          PECBYTE       LL_FMPI2C_IsEnabledSMBusPECCompare
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_FMPI2C_IsEnabledSMBusPECCompare(FMPI2C_TypeDef *FMPI2Cx)
{
  return ((READ_BIT(FMPI2Cx->CR2, FMPI2C_CR2_PECBYTE) == (FMPI2C_CR2_PECBYTE)) ? 1UL : 0UL);
}

/**
  * @brief  Get the SMBus Packet Error byte calculated.
  * @note   The macro IS_FMPSMBUS_ALL_INSTANCE(FMPI2Cx) can be used to check whether or not
  *         SMBus feature is supported by the FMPI2Cx Instance.
  * @rmtoll PECR         PEC           LL_FMPI2C_GetSMBusPEC
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x00 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_FMPI2C_GetSMBusPEC(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint32_t)(READ_BIT(FMPI2Cx->PECR, FMPI2C_PECR_PEC));
}

/**
  * @brief  Read Receive Data register.
  * @rmtoll RXDR         RXDATA        LL_FMPI2C_ReceiveData8
  * @param  FMPI2Cx FMPI2C Instance.
  * @retval Value between Min_Data=0x00 and Max_Data=0xFF
  */
__STATIC_INLINE uint8_t LL_FMPI2C_ReceiveData8(FMPI2C_TypeDef *FMPI2Cx)
{
  return (uint8_t)(READ_BIT(FMPI2Cx->RXDR, FMPI2C_RXDR_RXDATA));
}

/**
  * @brief  Write in Transmit Data Register .
  * @rmtoll TXDR         TXDATA        LL_FMPI2C_TransmitData8
  * @param  FMPI2Cx FMPI2C Instance.
  * @param  Data Value between Min_Data=0x00 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_FMPI2C_TransmitData8(FMPI2C_TypeDef *FMPI2Cx, uint8_t Data)
{
  WRITE_REG(FMPI2Cx->TXDR, Data);
}

/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup FMPI2C_LL_EF_Init Initialization and de-initialization functions
  * @{
  */

ErrorStatus LL_FMPI2C_Init(FMPI2C_TypeDef *FMPI2Cx, LL_FMPI2C_InitTypeDef *FMPI2C_InitStruct);
ErrorStatus LL_FMPI2C_DeInit(FMPI2C_TypeDef *FMPI2Cx);
void LL_FMPI2C_StructInit(LL_FMPI2C_InitTypeDef *FMPI2C_InitStruct);


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

#endif /* FMPI2C1 */

/**
  * @}
  */

#endif /* FMPI2C_CR1_PE */
#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_LL_FMPI2C_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
