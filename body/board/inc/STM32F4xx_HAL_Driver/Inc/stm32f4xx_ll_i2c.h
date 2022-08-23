/**
  ******************************************************************************
  * @file    stm32f4xx_ll_i2c.h
  * @author  MCD Application Team
  * @brief   Header file of I2C LL module.
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
#ifndef __STM32F4xx_LL_I2C_H
#define __STM32F4xx_LL_I2C_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (I2C1) || defined (I2C2) || defined (I2C3)

/** @defgroup I2C_LL I2C
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/

/* Private constants ---------------------------------------------------------*/
/** @defgroup I2C_LL_Private_Constants I2C Private Constants
  * @{
  */

/* Defines used to perform compute and check in the macros */
#define LL_I2C_MAX_SPEED_STANDARD           100000U
#define LL_I2C_MAX_SPEED_FAST               400000U
/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup I2C_LL_Private_Macros I2C Private Macros
  * @{
  */
/**
  * @}
  */
#endif /*USE_FULL_LL_DRIVER*/

/* Exported types ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup I2C_LL_ES_INIT I2C Exported Init structure
  * @{
  */
typedef struct
{
  uint32_t PeripheralMode;      /*!< Specifies the peripheral mode.
                                     This parameter can be a value of @ref I2C_LL_EC_PERIPHERAL_MODE

                                     This feature can be modified afterwards using unitary function @ref LL_I2C_SetMode(). */

  uint32_t ClockSpeed;          /*!< Specifies the clock frequency.
                                     This parameter must be set to a value lower than 400kHz (in Hz)

                                     This feature can be modified afterwards using unitary function @ref LL_I2C_SetClockPeriod()
                                     or @ref LL_I2C_SetDutyCycle() or @ref LL_I2C_SetClockSpeedMode() or @ref LL_I2C_ConfigSpeed(). */

  uint32_t DutyCycle;           /*!< Specifies the I2C fast mode duty cycle.
                                     This parameter can be a value of @ref I2C_LL_EC_DUTYCYCLE

                                     This feature can be modified afterwards using unitary function @ref LL_I2C_SetDutyCycle(). */

#if  defined(I2C_FLTR_ANOFF)&&defined(I2C_FLTR_DNF)
  uint32_t AnalogFilter;        /*!< Enables or disables analog noise filter.
                                     This parameter can be a value of @ref I2C_LL_EC_ANALOGFILTER_SELECTION

                                     This feature can be modified afterwards using unitary functions @ref LL_I2C_EnableAnalogFilter() or LL_I2C_DisableAnalogFilter(). */

  uint32_t DigitalFilter;       /*!< Configures the digital noise filter.
                                     This parameter can be a number between Min_Data = 0x00 and Max_Data = 0x0F

                                     This feature can be modified afterwards using unitary function @ref LL_I2C_SetDigitalFilter(). */

#endif
  uint32_t OwnAddress1;         /*!< Specifies the device own address 1.
                                     This parameter must be a value between Min_Data = 0x00 and Max_Data = 0x3FF

                                     This feature can be modified afterwards using unitary function @ref LL_I2C_SetOwnAddress1(). */

  uint32_t TypeAcknowledge;     /*!< Specifies the ACKnowledge or Non ACKnowledge condition after the address receive match code or next received byte.
                                     This parameter can be a value of @ref I2C_LL_EC_I2C_ACKNOWLEDGE

                                     This feature can be modified afterwards using unitary function @ref LL_I2C_AcknowledgeNextData(). */

  uint32_t OwnAddrSize;         /*!< Specifies the device own address 1 size (7-bit or 10-bit).
                                     This parameter can be a value of @ref I2C_LL_EC_OWNADDRESS1

                                     This feature can be modified afterwards using unitary function @ref LL_I2C_SetOwnAddress1(). */
} LL_I2C_InitTypeDef;
/**
  * @}
  */
#endif /*USE_FULL_LL_DRIVER*/

/* Exported constants --------------------------------------------------------*/
/** @defgroup I2C_LL_Exported_Constants I2C Exported Constants
  * @{
  */

/** @defgroup I2C_LL_EC_GET_FLAG Get Flags Defines
  * @brief    Flags defines which can be used with LL_I2C_ReadReg function
  * @{
  */
#define LL_I2C_SR1_SB                       I2C_SR1_SB              /*!< Start Bit (master mode)                   */
#define LL_I2C_SR1_ADDR                     I2C_SR1_ADDR            /*!< Address sent (master mode) or
                                                                         Address matched flag (slave mode)         */
#define LL_I2C_SR1_BTF                      I2C_SR1_BTF             /*!< Byte Transfer Finished flag               */
#define LL_I2C_SR1_ADD10                    I2C_SR1_ADD10           /*!< 10-bit header sent (master mode)          */
#define LL_I2C_SR1_STOPF                    I2C_SR1_STOPF           /*!< Stop detection flag (slave mode)          */
#define LL_I2C_SR1_RXNE                     I2C_SR1_RXNE            /*!< Data register not empty (receivers)       */
#define LL_I2C_SR1_TXE                      I2C_SR1_TXE             /*!< Data register empty (transmitters)        */
#define LL_I2C_SR1_BERR                     I2C_SR1_BERR            /*!< Bus error                                 */
#define LL_I2C_SR1_ARLO                     I2C_SR1_ARLO            /*!< Arbitration lost                          */
#define LL_I2C_SR1_AF                       I2C_SR1_AF              /*!< Acknowledge failure flag                  */
#define LL_I2C_SR1_OVR                      I2C_SR1_OVR             /*!< Overrun/Underrun                          */
#define LL_I2C_SR1_PECERR                   I2C_ISR_PECERR          /*!< PEC Error in reception (SMBus mode)       */
#define LL_I2C_SR1_TIMEOUT                  I2C_ISR_TIMEOUT         /*!< Timeout detection flag (SMBus mode)       */
#define LL_I2C_SR1_SMALERT                  I2C_ISR_SMALERT         /*!< SMBus alert (SMBus mode)                  */
#define LL_I2C_SR2_MSL                      I2C_SR2_MSL             /*!< Master/Slave flag                         */
#define LL_I2C_SR2_BUSY                     I2C_SR2_BUSY            /*!< Bus busy flag                             */
#define LL_I2C_SR2_TRA                      I2C_SR2_TRA             /*!< Transmitter/receiver direction            */
#define LL_I2C_SR2_GENCALL                  I2C_SR2_GENCALL         /*!< General call address (Slave mode)         */
#define LL_I2C_SR2_SMBDEFAULT               I2C_SR2_SMBDEFAULT      /*!< SMBus Device default address (Slave mode) */
#define LL_I2C_SR2_SMBHOST                  I2C_SR2_SMBHOST         /*!< SMBus Host address (Slave mode)           */
#define LL_I2C_SR2_DUALF                    I2C_SR2_DUALF           /*!< Dual flag  (Slave mode)                   */
/**
  * @}
  */

/** @defgroup I2C_LL_EC_IT IT Defines
  * @brief    IT defines which can be used with LL_I2C_ReadReg and  LL_I2C_WriteReg functions
  * @{
  */
#define LL_I2C_CR2_ITEVTEN                  I2C_CR2_ITEVTEN         /*!< Events interrupts enable */
#define LL_I2C_CR2_ITBUFEN                  I2C_CR2_ITBUFEN         /*!< Buffer interrupts enable */
#define LL_I2C_CR2_ITERREN                  I2C_CR2_ITERREN         /*!< Error interrupts enable  */
/**
  * @}
  */

#if defined(I2C_FLTR_ANOFF)
/** @defgroup I2C_LL_EC_ANALOGFILTER_SELECTION  Analog Filter Selection
  * @{
  */
#define LL_I2C_ANALOGFILTER_ENABLE          0x00000000U             /*!< Analog filter is enabled. */
#define LL_I2C_ANALOGFILTER_DISABLE         I2C_FLTR_ANOFF          /*!< Analog filter is disabled.*/
/**
  * @}
  */

#endif
/** @defgroup I2C_LL_EC_OWNADDRESS1 Own Address 1 Length
  * @{
  */
#define LL_I2C_OWNADDRESS1_7BIT             0x00004000U                                /*!< Own address 1 is a 7-bit address.   */
#define LL_I2C_OWNADDRESS1_10BIT            (uint32_t)(I2C_OAR1_ADDMODE | 0x00004000U) /*!< Own address 1 is a 10-bit address.  */
/**
  * @}
  */

/** @defgroup I2C_LL_EC_DUTYCYCLE Fast Mode Duty Cycle
  * @{
  */
#define LL_I2C_DUTYCYCLE_2                  0x00000000U             /*!< I2C fast mode Tlow/Thigh = 2        */
#define LL_I2C_DUTYCYCLE_16_9               I2C_CCR_DUTY            /*!< I2C fast mode Tlow/Thigh = 16/9     */
/**
  * @}
  */

/** @defgroup I2C_LL_EC_CLOCK_SPEED_MODE Master Clock Speed Mode
  * @{
  */
#define LL_I2C_CLOCK_SPEED_STANDARD_MODE    0x00000000U             /*!< Master clock speed range is standard mode */
#define LL_I2C_CLOCK_SPEED_FAST_MODE        I2C_CCR_FS              /*!< Master clock speed range is fast mode     */
/**
  * @}
  */

/** @defgroup I2C_LL_EC_PERIPHERAL_MODE Peripheral Mode
  * @{
  */
#define LL_I2C_MODE_I2C                     0x00000000U                                                 /*!< I2C Master or Slave mode                                    */
#define LL_I2C_MODE_SMBUS_HOST              (uint32_t)(I2C_CR1_SMBUS | I2C_CR1_SMBTYPE | I2C_CR1_ENARP) /*!< SMBus Host address acknowledge                              */
#define LL_I2C_MODE_SMBUS_DEVICE            I2C_CR1_SMBUS                                               /*!< SMBus Device default mode (Default address not acknowledge) */
#define LL_I2C_MODE_SMBUS_DEVICE_ARP        (uint32_t)(I2C_CR1_SMBUS | I2C_CR1_ENARP)                   /*!< SMBus Device Default address acknowledge                    */
/**
  * @}
  */

/** @defgroup I2C_LL_EC_I2C_ACKNOWLEDGE Acknowledge Generation
  * @{
  */
#define LL_I2C_ACK                          I2C_CR1_ACK             /*!< ACK is sent after current received byte. */
#define LL_I2C_NACK                         0x00000000U             /*!< NACK is sent after current received byte.*/
/**
  * @}
  */

/** @defgroup I2C_LL_EC_DIRECTION Read Write Direction
  * @{
  */
#define LL_I2C_DIRECTION_WRITE              I2C_SR2_TRA             /*!< Bus is in write transfer */
#define LL_I2C_DIRECTION_READ               0x00000000U             /*!< Bus is in read transfer  */
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup I2C_LL_Exported_Macros I2C Exported Macros
  * @{
  */

/** @defgroup I2C_LL_EM_WRITE_READ Common Write and read registers Macros
  * @{
  */

/**
  * @brief  Write a value in I2C register
  * @param  __INSTANCE__ I2C Instance
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_I2C_WriteReg(__INSTANCE__, __REG__, __VALUE__) WRITE_REG(__INSTANCE__->__REG__, (__VALUE__))

/**
  * @brief  Read a value in I2C register
  * @param  __INSTANCE__ I2C Instance
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_I2C_ReadReg(__INSTANCE__, __REG__) READ_REG(__INSTANCE__->__REG__)
/**
  * @}
  */

/** @defgroup I2C_LL_EM_Exported_Macros_Helper Exported_Macros_Helper
  * @{
  */

/**
  * @brief  Convert Peripheral Clock Frequency in Mhz.
  * @param  __PCLK__ This parameter must be a value of peripheral clock (in Hz).
  * @retval Value of peripheral clock (in Mhz)
  */
#define __LL_I2C_FREQ_HZ_TO_MHZ(__PCLK__)                               (uint32_t)((__PCLK__)/1000000U)

/**
  * @brief  Convert Peripheral Clock Frequency in Hz.
  * @param  __PCLK__ This parameter must be a value of peripheral clock (in Mhz).
  * @retval Value of peripheral clock (in Hz)
  */
#define __LL_I2C_FREQ_MHZ_TO_HZ(__PCLK__)                               (uint32_t)((__PCLK__)*1000000U)

/**
  * @brief  Compute I2C Clock rising time.
  * @param  __FREQRANGE__ This parameter must be a value of peripheral clock (in Mhz).
  * @param  __SPEED__ This parameter must be a value lower than 400kHz (in Hz).
  * @retval Value between Min_Data=0x02 and Max_Data=0x3F
  */
#define __LL_I2C_RISE_TIME(__FREQRANGE__, __SPEED__)                    (uint32_t)(((__SPEED__) <= LL_I2C_MAX_SPEED_STANDARD) ? ((__FREQRANGE__) + 1U) : ((((__FREQRANGE__) * 300U) / 1000U) + 1U))

/**
  * @brief  Compute Speed clock range to a Clock Control Register (I2C_CCR_CCR) value.
  * @param  __PCLK__ This parameter must be a value of peripheral clock (in Hz).
  * @param  __SPEED__ This parameter must be a value lower than 400kHz (in Hz).
  * @param  __DUTYCYCLE__ This parameter can be one of the following values:
  *         @arg @ref LL_I2C_DUTYCYCLE_2
  *         @arg @ref LL_I2C_DUTYCYCLE_16_9
  * @retval Value between Min_Data=0x004 and Max_Data=0xFFF, except in FAST DUTY mode where Min_Data=0x001.
  */
#define __LL_I2C_SPEED_TO_CCR(__PCLK__, __SPEED__, __DUTYCYCLE__)       (uint32_t)(((__SPEED__) <= LL_I2C_MAX_SPEED_STANDARD)? \
                                                                                  (__LL_I2C_SPEED_STANDARD_TO_CCR((__PCLK__), (__SPEED__))) : \
                                                                                  (__LL_I2C_SPEED_FAST_TO_CCR((__PCLK__), (__SPEED__), (__DUTYCYCLE__))))

/**
  * @brief  Compute Speed Standard clock range to a Clock Control Register (I2C_CCR_CCR) value.
  * @param  __PCLK__ This parameter must be a value of peripheral clock (in Hz).
  * @param  __SPEED__ This parameter must be a value lower than 100kHz (in Hz).
  * @retval Value between Min_Data=0x004 and Max_Data=0xFFF.
  */
#define __LL_I2C_SPEED_STANDARD_TO_CCR(__PCLK__, __SPEED__)             (uint32_t)(((((__PCLK__)/((__SPEED__) << 1U)) & I2C_CCR_CCR) < 4U)? 4U:((__PCLK__) / ((__SPEED__) << 1U)))

/**
  * @brief  Compute Speed Fast clock range to a Clock Control Register (I2C_CCR_CCR) value.
  * @param  __PCLK__ This parameter must be a value of peripheral clock (in Hz).
  * @param  __SPEED__ This parameter must be a value between Min_Data=100Khz and Max_Data=400Khz (in Hz).
  * @param  __DUTYCYCLE__ This parameter can be one of the following values:
  *         @arg @ref LL_I2C_DUTYCYCLE_2
  *         @arg @ref LL_I2C_DUTYCYCLE_16_9
  * @retval Value between Min_Data=0x001 and Max_Data=0xFFF
  */
#define __LL_I2C_SPEED_FAST_TO_CCR(__PCLK__, __SPEED__, __DUTYCYCLE__)  (uint32_t)(((__DUTYCYCLE__) == LL_I2C_DUTYCYCLE_2)? \
                                                                            (((((__PCLK__) / ((__SPEED__) * 3U)) & I2C_CCR_CCR) == 0U)? 1U:((__PCLK__) / ((__SPEED__) * 3U))) : \
                                                                            (((((__PCLK__) / ((__SPEED__) * 25U)) & I2C_CCR_CCR) == 0U)? 1U:((__PCLK__) / ((__SPEED__) * 25U))))

/**
  * @brief  Get the Least significant bits of a 10-Bits address.
  * @param  __ADDRESS__ This parameter must be a value of a 10-Bits slave address.
  * @retval Value between Min_Data=0x00 and Max_Data=0xFF
  */
#define __LL_I2C_10BIT_ADDRESS(__ADDRESS__)                             ((uint8_t)((uint16_t)((__ADDRESS__) & (uint16_t)(0x00FF))))

/**
  * @brief  Convert a 10-Bits address to a 10-Bits header with Write direction.
  * @param  __ADDRESS__ This parameter must be a value of a 10-Bits slave address.
  * @retval Value between Min_Data=0xF0 and Max_Data=0xF6
  */
#define __LL_I2C_10BIT_HEADER_WRITE(__ADDRESS__)                        ((uint8_t)((uint16_t)((uint16_t)(((uint16_t)((__ADDRESS__) & (uint16_t)(0x0300))) >> 7) | (uint16_t)(0xF0))))

/**
  * @brief  Convert a 10-Bits address to a 10-Bits header with Read direction.
  * @param  __ADDRESS__ This parameter must be a value of a 10-Bits slave address.
  * @retval Value between Min_Data=0xF1 and Max_Data=0xF7
  */
#define __LL_I2C_10BIT_HEADER_READ(__ADDRESS__)                         ((uint8_t)((uint16_t)((uint16_t)(((uint16_t)((__ADDRESS__) & (uint16_t)(0x0300))) >> 7) | (uint16_t)(0xF1))))

/**
  * @}
  */

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/

/** @defgroup I2C_LL_Exported_Functions I2C Exported Functions
  * @{
  */

/** @defgroup I2C_LL_EF_Configuration Configuration
  * @{
  */

/**
  * @brief  Enable I2C peripheral (PE = 1).
  * @rmtoll CR1          PE            LL_I2C_Enable
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_Enable(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_PE);
}

/**
  * @brief  Disable I2C peripheral (PE = 0).
  * @rmtoll CR1          PE            LL_I2C_Disable
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_Disable(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR1, I2C_CR1_PE);
}

/**
  * @brief  Check if the I2C peripheral is enabled or disabled.
  * @rmtoll CR1          PE            LL_I2C_IsEnabled
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabled(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR1, I2C_CR1_PE) == (I2C_CR1_PE));
}

#if  defined(I2C_FLTR_ANOFF)&&defined(I2C_FLTR_DNF)
/**
  * @brief  Configure Noise Filters (Analog and Digital).
  * @note   If the analog filter is also enabled, the digital filter is added to analog filter.
  *         The filters can only be programmed when the I2C is disabled (PE = 0).
  * @rmtoll FLTR         ANOFF         LL_I2C_ConfigFilters\n
  *         FLTR         DNF           LL_I2C_ConfigFilters
  * @param  I2Cx I2C Instance.
  * @param  AnalogFilter This parameter can be one of the following values:
  *         @arg @ref LL_I2C_ANALOGFILTER_ENABLE
  *         @arg @ref LL_I2C_ANALOGFILTER_DISABLE
  * @param  DigitalFilter This parameter must be a value between Min_Data=0x00 (Digital filter disabled) and Max_Data=0x0F (Digital filter enabled and filtering capability up to 15*TPCLK1)
  *               This parameter is used to configure the digital noise filter on SDA and SCL input. The digital filter will suppress the spikes with a length of up to DNF[3:0]*TPCLK1.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ConfigFilters(I2C_TypeDef *I2Cx, uint32_t AnalogFilter, uint32_t DigitalFilter)
{
  MODIFY_REG(I2Cx->FLTR, I2C_FLTR_ANOFF | I2C_FLTR_DNF, AnalogFilter | DigitalFilter);
}
#endif
#if defined(I2C_FLTR_DNF)

/**
  * @brief  Configure Digital Noise Filter.
  * @note   If the analog filter is also enabled, the digital filter is added to analog filter.
  *         This filter can only be programmed when the I2C is disabled (PE = 0).
  * @rmtoll FLTR         DNF           LL_I2C_SetDigitalFilter
  * @param  I2Cx I2C Instance.
  * @param  DigitalFilter This parameter must be a value between Min_Data=0x00 (Digital filter disabled) and Max_Data=0x0F (Digital filter enabled and filtering capability up to 15*TPCLK1)
  *               This parameter is used to configure the digital noise filter on SDA and SCL input. The digital filter will suppress the spikes with a length of up to DNF[3:0]*TPCLK1.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_SetDigitalFilter(I2C_TypeDef *I2Cx, uint32_t DigitalFilter)
{
  MODIFY_REG(I2Cx->FLTR, I2C_FLTR_DNF, DigitalFilter);
}

/**
  * @brief  Get the current Digital Noise Filter configuration.
  * @rmtoll FLTR         DNF           LL_I2C_GetDigitalFilter
  * @param  I2Cx I2C Instance.
  * @retval Value between Min_Data=0x0 and Max_Data=0xF
  */
__STATIC_INLINE uint32_t LL_I2C_GetDigitalFilter(I2C_TypeDef *I2Cx)
{
  return (uint32_t)(READ_BIT(I2Cx->FLTR, I2C_FLTR_DNF));
}
#endif
#if defined(I2C_FLTR_ANOFF)

/**
  * @brief  Enable Analog Noise Filter.
  * @note   This filter can only be programmed when the I2C is disabled (PE = 0).
  * @rmtoll FLTR         ANOFF         LL_I2C_EnableAnalogFilter
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableAnalogFilter(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->FLTR, I2C_FLTR_ANOFF);
}

/**
  * @brief  Disable Analog Noise Filter.
  * @note   This filter can only be programmed when the I2C is disabled (PE = 0).
  * @rmtoll FLTR         ANOFF         LL_I2C_DisableAnalogFilter
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableAnalogFilter(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->FLTR, I2C_FLTR_ANOFF);
}

/**
  * @brief  Check if Analog Noise Filter is enabled or disabled.
  * @rmtoll FLTR         ANOFF         LL_I2C_IsEnabledAnalogFilter
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledAnalogFilter(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->FLTR, I2C_FLTR_ANOFF) == (I2C_FLTR_ANOFF));
}
#endif

/**
  * @brief  Enable DMA transmission requests.
  * @rmtoll CR2          DMAEN         LL_I2C_EnableDMAReq_TX
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableDMAReq_TX(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR2, I2C_CR2_DMAEN);
}

/**
  * @brief  Disable DMA transmission requests.
  * @rmtoll CR2          DMAEN         LL_I2C_DisableDMAReq_TX
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableDMAReq_TX(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR2, I2C_CR2_DMAEN);
}

/**
  * @brief  Check if DMA transmission requests are enabled or disabled.
  * @rmtoll CR2          DMAEN         LL_I2C_IsEnabledDMAReq_TX
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledDMAReq_TX(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR2, I2C_CR2_DMAEN) == (I2C_CR2_DMAEN));
}

/**
  * @brief  Enable DMA reception requests.
  * @rmtoll CR2          DMAEN         LL_I2C_EnableDMAReq_RX
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableDMAReq_RX(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR2, I2C_CR2_DMAEN);
}

/**
  * @brief  Disable DMA reception requests.
  * @rmtoll CR2          DMAEN         LL_I2C_DisableDMAReq_RX
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableDMAReq_RX(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR2, I2C_CR2_DMAEN);
}

/**
  * @brief  Check if DMA reception requests are enabled or disabled.
  * @rmtoll CR2          DMAEN         LL_I2C_IsEnabledDMAReq_RX
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledDMAReq_RX(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR2, I2C_CR2_DMAEN) == (I2C_CR2_DMAEN));
}

/**
  * @brief  Get the data register address used for DMA transfer.
  * @rmtoll DR           DR            LL_I2C_DMA_GetRegAddr
  * @param  I2Cx I2C Instance.
  * @retval Address of data register
  */
__STATIC_INLINE uint32_t LL_I2C_DMA_GetRegAddr(I2C_TypeDef *I2Cx)
{
  return (uint32_t) & (I2Cx->DR);
}

/**
  * @brief  Enable Clock stretching.
  * @note   This bit can only be programmed when the I2C is disabled (PE = 0).
  * @rmtoll CR1          NOSTRETCH     LL_I2C_EnableClockStretching
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableClockStretching(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR1, I2C_CR1_NOSTRETCH);
}

/**
  * @brief  Disable Clock stretching.
  * @note   This bit can only be programmed when the I2C is disabled (PE = 0).
  * @rmtoll CR1          NOSTRETCH     LL_I2C_DisableClockStretching
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableClockStretching(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_NOSTRETCH);
}

/**
  * @brief  Check if Clock stretching is enabled or disabled.
  * @rmtoll CR1          NOSTRETCH     LL_I2C_IsEnabledClockStretching
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledClockStretching(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR1, I2C_CR1_NOSTRETCH) != (I2C_CR1_NOSTRETCH));
}

/**
  * @brief  Enable General Call.
  * @note   When enabled the Address 0x00 is ACKed.
  * @rmtoll CR1          ENGC          LL_I2C_EnableGeneralCall
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableGeneralCall(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_ENGC);
}

/**
  * @brief  Disable General Call.
  * @note   When disabled the Address 0x00 is NACKed.
  * @rmtoll CR1          ENGC          LL_I2C_DisableGeneralCall
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableGeneralCall(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR1, I2C_CR1_ENGC);
}

/**
  * @brief  Check if General Call is enabled or disabled.
  * @rmtoll CR1          ENGC          LL_I2C_IsEnabledGeneralCall
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledGeneralCall(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR1, I2C_CR1_ENGC) == (I2C_CR1_ENGC));
}

/**
  * @brief  Set the Own Address1.
  * @rmtoll OAR1         ADD0          LL_I2C_SetOwnAddress1\n
  *         OAR1         ADD1_7        LL_I2C_SetOwnAddress1\n
  *         OAR1         ADD8_9        LL_I2C_SetOwnAddress1\n
  *         OAR1         ADDMODE       LL_I2C_SetOwnAddress1
  * @param  I2Cx I2C Instance.
  * @param  OwnAddress1 This parameter must be a value between Min_Data=0 and Max_Data=0x3FF.
  * @param  OwnAddrSize This parameter can be one of the following values:
  *         @arg @ref LL_I2C_OWNADDRESS1_7BIT
  *         @arg @ref LL_I2C_OWNADDRESS1_10BIT
  * @retval None
  */
__STATIC_INLINE void LL_I2C_SetOwnAddress1(I2C_TypeDef *I2Cx, uint32_t OwnAddress1, uint32_t OwnAddrSize)
{
  MODIFY_REG(I2Cx->OAR1, I2C_OAR1_ADD0 | I2C_OAR1_ADD1_7 | I2C_OAR1_ADD8_9 | I2C_OAR1_ADDMODE, OwnAddress1 | OwnAddrSize);
}

/**
  * @brief  Set the 7bits Own Address2.
  * @note   This action has no effect if own address2 is enabled.
  * @rmtoll OAR2         ADD2          LL_I2C_SetOwnAddress2
  * @param  I2Cx I2C Instance.
  * @param  OwnAddress2 This parameter must be a value between Min_Data=0 and Max_Data=0x7F.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_SetOwnAddress2(I2C_TypeDef *I2Cx, uint32_t OwnAddress2)
{
  MODIFY_REG(I2Cx->OAR2, I2C_OAR2_ADD2, OwnAddress2);
}

/**
  * @brief  Enable acknowledge on Own Address2 match address.
  * @rmtoll OAR2         ENDUAL        LL_I2C_EnableOwnAddress2
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableOwnAddress2(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->OAR2, I2C_OAR2_ENDUAL);
}

/**
  * @brief  Disable  acknowledge on Own Address2 match address.
  * @rmtoll OAR2         ENDUAL        LL_I2C_DisableOwnAddress2
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableOwnAddress2(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->OAR2, I2C_OAR2_ENDUAL);
}

/**
  * @brief  Check if Own Address1 acknowledge is enabled or disabled.
  * @rmtoll OAR2         ENDUAL        LL_I2C_IsEnabledOwnAddress2
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledOwnAddress2(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->OAR2, I2C_OAR2_ENDUAL) == (I2C_OAR2_ENDUAL));
}

/**
  * @brief  Configure the Peripheral clock frequency.
  * @rmtoll CR2          FREQ          LL_I2C_SetPeriphClock
  * @param  I2Cx I2C Instance.
  * @param  PeriphClock Peripheral Clock (in Hz)
  * @retval None
  */
__STATIC_INLINE void LL_I2C_SetPeriphClock(I2C_TypeDef *I2Cx, uint32_t PeriphClock)
{
  MODIFY_REG(I2Cx->CR2, I2C_CR2_FREQ, __LL_I2C_FREQ_HZ_TO_MHZ(PeriphClock));
}

/**
  * @brief  Get the Peripheral clock frequency.
  * @rmtoll CR2          FREQ          LL_I2C_GetPeriphClock
  * @param  I2Cx I2C Instance.
  * @retval Value of Peripheral Clock (in Hz)
  */
__STATIC_INLINE uint32_t LL_I2C_GetPeriphClock(I2C_TypeDef *I2Cx)
{
  return (uint32_t)(__LL_I2C_FREQ_MHZ_TO_HZ(READ_BIT(I2Cx->CR2, I2C_CR2_FREQ)));
}

/**
  * @brief  Configure the Duty cycle (Fast mode only).
  * @rmtoll CCR          DUTY          LL_I2C_SetDutyCycle
  * @param  I2Cx I2C Instance.
  * @param  DutyCycle This parameter can be one of the following values:
  *         @arg @ref LL_I2C_DUTYCYCLE_2
  *         @arg @ref LL_I2C_DUTYCYCLE_16_9
  * @retval None
  */
__STATIC_INLINE void LL_I2C_SetDutyCycle(I2C_TypeDef *I2Cx, uint32_t DutyCycle)
{
  MODIFY_REG(I2Cx->CCR, I2C_CCR_DUTY, DutyCycle);
}

/**
  * @brief  Get the Duty cycle (Fast mode only).
  * @rmtoll CCR          DUTY          LL_I2C_GetDutyCycle
  * @param  I2Cx I2C Instance.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_I2C_DUTYCYCLE_2
  *         @arg @ref LL_I2C_DUTYCYCLE_16_9
  */
__STATIC_INLINE uint32_t LL_I2C_GetDutyCycle(I2C_TypeDef *I2Cx)
{
  return (uint32_t)(READ_BIT(I2Cx->CCR, I2C_CCR_DUTY));
}

/**
  * @brief  Configure the I2C master clock speed mode.
  * @rmtoll CCR          FS            LL_I2C_SetClockSpeedMode
  * @param  I2Cx I2C Instance.
  * @param  ClockSpeedMode This parameter can be one of the following values:
  *         @arg @ref LL_I2C_CLOCK_SPEED_STANDARD_MODE
  *         @arg @ref LL_I2C_CLOCK_SPEED_FAST_MODE
  * @retval None
  */
__STATIC_INLINE void LL_I2C_SetClockSpeedMode(I2C_TypeDef *I2Cx, uint32_t ClockSpeedMode)
{
  MODIFY_REG(I2Cx->CCR, I2C_CCR_FS, ClockSpeedMode);
}

/**
  * @brief  Get the the I2C master speed mode.
  * @rmtoll CCR          FS            LL_I2C_GetClockSpeedMode
  * @param  I2Cx I2C Instance.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_I2C_CLOCK_SPEED_STANDARD_MODE
  *         @arg @ref LL_I2C_CLOCK_SPEED_FAST_MODE
  */
__STATIC_INLINE uint32_t LL_I2C_GetClockSpeedMode(I2C_TypeDef *I2Cx)
{
  return (uint32_t)(READ_BIT(I2Cx->CCR, I2C_CCR_FS));
}

/**
  * @brief  Configure the SCL, SDA rising time.
  * @note   This bit can only be programmed when the I2C is disabled (PE = 0).
  * @rmtoll TRISE        TRISE         LL_I2C_SetRiseTime
  * @param  I2Cx I2C Instance.
  * @param  RiseTime This parameter must be a value between Min_Data=0x02 and Max_Data=0x3F.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_SetRiseTime(I2C_TypeDef *I2Cx, uint32_t RiseTime)
{
  MODIFY_REG(I2Cx->TRISE, I2C_TRISE_TRISE, RiseTime);
}

/**
  * @brief  Get the SCL, SDA rising time.
  * @rmtoll TRISE        TRISE         LL_I2C_GetRiseTime
  * @param  I2Cx I2C Instance.
  * @retval Value between Min_Data=0x02 and Max_Data=0x3F
  */
__STATIC_INLINE uint32_t LL_I2C_GetRiseTime(I2C_TypeDef *I2Cx)
{
  return (uint32_t)(READ_BIT(I2Cx->TRISE, I2C_TRISE_TRISE));
}

/**
  * @brief  Configure the SCL high and low period.
  * @note   This bit can only be programmed when the I2C is disabled (PE = 0).
  * @rmtoll CCR          CCR           LL_I2C_SetClockPeriod
  * @param  I2Cx I2C Instance.
  * @param  ClockPeriod This parameter must be a value between Min_Data=0x004 and Max_Data=0xFFF, except in FAST DUTY mode where Min_Data=0x001.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_SetClockPeriod(I2C_TypeDef *I2Cx, uint32_t ClockPeriod)
{
  MODIFY_REG(I2Cx->CCR, I2C_CCR_CCR, ClockPeriod);
}

/**
  * @brief  Get the SCL high and low period.
  * @rmtoll CCR          CCR           LL_I2C_GetClockPeriod
  * @param  I2Cx I2C Instance.
  * @retval Value between Min_Data=0x004 and Max_Data=0xFFF, except in FAST DUTY mode where Min_Data=0x001.
  */
__STATIC_INLINE uint32_t LL_I2C_GetClockPeriod(I2C_TypeDef *I2Cx)
{
  return (uint32_t)(READ_BIT(I2Cx->CCR, I2C_CCR_CCR));
}

/**
  * @brief  Configure the SCL speed.
  * @note   This bit can only be programmed when the I2C is disabled (PE = 0).
  * @rmtoll CR2          FREQ          LL_I2C_ConfigSpeed\n
  *         TRISE        TRISE         LL_I2C_ConfigSpeed\n
  *         CCR          FS            LL_I2C_ConfigSpeed\n
  *         CCR          DUTY          LL_I2C_ConfigSpeed\n
  *         CCR          CCR           LL_I2C_ConfigSpeed
  * @param  I2Cx I2C Instance.
  * @param  PeriphClock Peripheral Clock (in Hz)
  * @param  ClockSpeed This parameter must be a value lower than 400kHz (in Hz).
  * @param  DutyCycle This parameter can be one of the following values:
  *         @arg @ref LL_I2C_DUTYCYCLE_2
  *         @arg @ref LL_I2C_DUTYCYCLE_16_9
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ConfigSpeed(I2C_TypeDef *I2Cx, uint32_t PeriphClock, uint32_t ClockSpeed,
                                        uint32_t DutyCycle)
{
  uint32_t freqrange = 0x0U;
  uint32_t clockconfig = 0x0U;

  /* Compute frequency range */
  freqrange = __LL_I2C_FREQ_HZ_TO_MHZ(PeriphClock);

  /* Configure I2Cx: Frequency range register */
  MODIFY_REG(I2Cx->CR2, I2C_CR2_FREQ, freqrange);

  /* Configure I2Cx: Rise Time register */
  MODIFY_REG(I2Cx->TRISE, I2C_TRISE_TRISE, __LL_I2C_RISE_TIME(freqrange, ClockSpeed));

  /* Configure Speed mode, Duty Cycle and Clock control register value */
  if (ClockSpeed > LL_I2C_MAX_SPEED_STANDARD)
  {
    /* Set Speed mode at fast and duty cycle for Clock Speed request in fast clock range */
    clockconfig = LL_I2C_CLOCK_SPEED_FAST_MODE                                          | \
                  __LL_I2C_SPEED_FAST_TO_CCR(PeriphClock, ClockSpeed, DutyCycle)        | \
                  DutyCycle;
  }
  else
  {
    /* Set Speed mode at standard for Clock Speed request in standard clock range */
    clockconfig = LL_I2C_CLOCK_SPEED_STANDARD_MODE                                      | \
                  __LL_I2C_SPEED_STANDARD_TO_CCR(PeriphClock, ClockSpeed);
  }

  /* Configure I2Cx: Clock control register */
  MODIFY_REG(I2Cx->CCR, (I2C_CCR_FS | I2C_CCR_DUTY | I2C_CCR_CCR), clockconfig);
}

/**
  * @brief  Configure peripheral mode.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll CR1          SMBUS         LL_I2C_SetMode\n
  *         CR1          SMBTYPE       LL_I2C_SetMode\n
  *         CR1          ENARP         LL_I2C_SetMode
  * @param  I2Cx I2C Instance.
  * @param  PeripheralMode This parameter can be one of the following values:
  *         @arg @ref LL_I2C_MODE_I2C
  *         @arg @ref LL_I2C_MODE_SMBUS_HOST
  *         @arg @ref LL_I2C_MODE_SMBUS_DEVICE
  *         @arg @ref LL_I2C_MODE_SMBUS_DEVICE_ARP
  * @retval None
  */
__STATIC_INLINE void LL_I2C_SetMode(I2C_TypeDef *I2Cx, uint32_t PeripheralMode)
{
  MODIFY_REG(I2Cx->CR1, I2C_CR1_SMBUS | I2C_CR1_SMBTYPE | I2C_CR1_ENARP, PeripheralMode);
}

/**
  * @brief  Get peripheral mode.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll CR1          SMBUS         LL_I2C_GetMode\n
  *         CR1          SMBTYPE       LL_I2C_GetMode\n
  *         CR1          ENARP         LL_I2C_GetMode
  * @param  I2Cx I2C Instance.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_I2C_MODE_I2C
  *         @arg @ref LL_I2C_MODE_SMBUS_HOST
  *         @arg @ref LL_I2C_MODE_SMBUS_DEVICE
  *         @arg @ref LL_I2C_MODE_SMBUS_DEVICE_ARP
  */
__STATIC_INLINE uint32_t LL_I2C_GetMode(I2C_TypeDef *I2Cx)
{
  return (uint32_t)(READ_BIT(I2Cx->CR1, I2C_CR1_SMBUS | I2C_CR1_SMBTYPE | I2C_CR1_ENARP));
}

/**
  * @brief  Enable SMBus alert (Host or Device mode)
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @note   SMBus Device mode:
  *         - SMBus Alert pin is drived low and
  *           Alert Response Address Header acknowledge is enabled.
  *         SMBus Host mode:
  *         - SMBus Alert pin management is supported.
  * @rmtoll CR1          ALERT         LL_I2C_EnableSMBusAlert
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableSMBusAlert(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_ALERT);
}

/**
  * @brief  Disable SMBus alert (Host or Device mode)
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @note   SMBus Device mode:
  *         - SMBus Alert pin is not drived (can be used as a standard GPIO) and
  *           Alert Response Address Header acknowledge is disabled.
  *         SMBus Host mode:
  *         - SMBus Alert pin management is not supported.
  * @rmtoll CR1          ALERT         LL_I2C_DisableSMBusAlert
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableSMBusAlert(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR1, I2C_CR1_ALERT);
}

/**
  * @brief  Check if SMBus alert (Host or Device mode) is enabled or disabled.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll CR1          ALERT         LL_I2C_IsEnabledSMBusAlert
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledSMBusAlert(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR1, I2C_CR1_ALERT) == (I2C_CR1_ALERT));
}

/**
  * @brief  Enable SMBus Packet Error Calculation (PEC).
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll CR1          ENPEC         LL_I2C_EnableSMBusPEC
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableSMBusPEC(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_ENPEC);
}

/**
  * @brief  Disable SMBus Packet Error Calculation (PEC).
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll CR1          ENPEC         LL_I2C_DisableSMBusPEC
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableSMBusPEC(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR1, I2C_CR1_ENPEC);
}

/**
  * @brief  Check if SMBus Packet Error Calculation (PEC) is enabled or disabled.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll CR1          ENPEC         LL_I2C_IsEnabledSMBusPEC
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledSMBusPEC(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR1, I2C_CR1_ENPEC) == (I2C_CR1_ENPEC));
}

/**
  * @}
  */

/** @defgroup I2C_LL_EF_IT_Management IT_Management
  * @{
  */

/**
  * @brief  Enable TXE interrupt.
  * @rmtoll CR2          ITEVTEN       LL_I2C_EnableIT_TX\n
  *         CR2          ITBUFEN       LL_I2C_EnableIT_TX
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableIT_TX(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR2, I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN);
}

/**
  * @brief  Disable TXE interrupt.
  * @rmtoll CR2          ITEVTEN       LL_I2C_DisableIT_TX\n
  *         CR2          ITBUFEN       LL_I2C_DisableIT_TX
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableIT_TX(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR2, I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN);
}

/**
  * @brief  Check if the TXE Interrupt is enabled or disabled.
  * @rmtoll CR2          ITEVTEN       LL_I2C_IsEnabledIT_TX\n
  *         CR2          ITBUFEN       LL_I2C_IsEnabledIT_TX
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledIT_TX(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR2, I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN) == (I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN));
}

/**
  * @brief  Enable RXNE interrupt.
  * @rmtoll CR2          ITEVTEN       LL_I2C_EnableIT_RX\n
  *         CR2          ITBUFEN       LL_I2C_EnableIT_RX
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableIT_RX(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR2, I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN);
}

/**
  * @brief  Disable RXNE interrupt.
  * @rmtoll CR2          ITEVTEN       LL_I2C_DisableIT_RX\n
  *         CR2          ITBUFEN       LL_I2C_DisableIT_RX
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableIT_RX(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR2, I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN);
}

/**
  * @brief  Check if the RXNE Interrupt is enabled or disabled.
  * @rmtoll CR2          ITEVTEN       LL_I2C_IsEnabledIT_RX\n
  *         CR2          ITBUFEN       LL_I2C_IsEnabledIT_RX
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledIT_RX(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR2, I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN) == (I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN));
}

/**
  * @brief  Enable Events interrupts.
  * @note   Any of these events will generate interrupt :
  *         Start Bit (SB)
  *         Address sent, Address matched (ADDR)
  *         10-bit header sent (ADD10)
  *         Stop detection  (STOPF)
  *         Byte transfer finished (BTF)
  *
  * @note   Any of these events will generate interrupt if Buffer interrupts are enabled too(using unitary function @ref LL_I2C_EnableIT_BUF()) :
  *         Receive buffer not empty (RXNE)
  *         Transmit buffer empty (TXE)
  * @rmtoll CR2          ITEVTEN       LL_I2C_EnableIT_EVT
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableIT_EVT(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR2, I2C_CR2_ITEVTEN);
}

/**
  * @brief  Disable Events interrupts.
  * @note   Any of these events will generate interrupt :
  *         Start Bit (SB)
  *         Address sent, Address matched (ADDR)
  *         10-bit header sent (ADD10)
  *         Stop detection  (STOPF)
  *         Byte transfer finished (BTF)
  *         Receive buffer not empty (RXNE)
  *         Transmit buffer empty (TXE)
  * @rmtoll CR2          ITEVTEN       LL_I2C_DisableIT_EVT
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableIT_EVT(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR2, I2C_CR2_ITEVTEN);
}

/**
  * @brief  Check if Events interrupts are enabled or disabled.
  * @rmtoll CR2          ITEVTEN       LL_I2C_IsEnabledIT_EVT
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledIT_EVT(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR2, I2C_CR2_ITEVTEN) == (I2C_CR2_ITEVTEN));
}

/**
  * @brief  Enable Buffer interrupts.
  * @note   Any of these Buffer events will generate interrupt if Events interrupts are enabled too(using unitary function @ref LL_I2C_EnableIT_EVT()) :
  *         Receive buffer not empty (RXNE)
  *         Transmit buffer empty (TXE)
  * @rmtoll CR2          ITBUFEN       LL_I2C_EnableIT_BUF
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableIT_BUF(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR2, I2C_CR2_ITBUFEN);
}

/**
  * @brief  Disable Buffer interrupts.
  * @note   Any of these Buffer events will generate interrupt :
  *         Receive buffer not empty (RXNE)
  *         Transmit buffer empty (TXE)
  * @rmtoll CR2          ITBUFEN       LL_I2C_DisableIT_BUF
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableIT_BUF(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR2, I2C_CR2_ITBUFEN);
}

/**
  * @brief  Check if Buffer interrupts are enabled or disabled.
  * @rmtoll CR2          ITBUFEN       LL_I2C_IsEnabledIT_BUF
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledIT_BUF(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR2, I2C_CR2_ITBUFEN) == (I2C_CR2_ITBUFEN));
}

/**
  * @brief  Enable Error interrupts.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @note   Any of these errors will generate interrupt :
  *         Bus Error detection (BERR)
  *         Arbitration Loss (ARLO)
  *         Acknowledge Failure(AF)
  *         Overrun/Underrun (OVR)
  *         SMBus Timeout detection (TIMEOUT)
  *         SMBus PEC error detection (PECERR)
  *         SMBus Alert pin event detection (SMBALERT)
  * @rmtoll CR2          ITERREN       LL_I2C_EnableIT_ERR
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableIT_ERR(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR2, I2C_CR2_ITERREN);
}

/**
  * @brief  Disable Error interrupts.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @note   Any of these errors will generate interrupt :
  *         Bus Error detection (BERR)
  *         Arbitration Loss (ARLO)
  *         Acknowledge Failure(AF)
  *         Overrun/Underrun (OVR)
  *         SMBus Timeout detection (TIMEOUT)
  *         SMBus PEC error detection (PECERR)
  *         SMBus Alert pin event detection (SMBALERT)
  * @rmtoll CR2          ITERREN       LL_I2C_DisableIT_ERR
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableIT_ERR(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR2, I2C_CR2_ITERREN);
}

/**
  * @brief  Check if Error interrupts are enabled or disabled.
  * @rmtoll CR2          ITERREN       LL_I2C_IsEnabledIT_ERR
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledIT_ERR(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR2, I2C_CR2_ITERREN) == (I2C_CR2_ITERREN));
}

/**
  * @}
  */

/** @defgroup I2C_LL_EF_FLAG_management FLAG_management
  * @{
  */

/**
  * @brief  Indicate the status of Transmit data register empty flag.
  * @note   RESET: When next data is written in Transmit data register.
  *         SET: When Transmit data register is empty.
  * @rmtoll SR1          TXE           LL_I2C_IsActiveFlag_TXE
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_TXE(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_TXE) == (I2C_SR1_TXE));
}

/**
  * @brief  Indicate the status of Byte Transfer Finished flag.
  *         RESET: When Data byte transfer not done.
  *         SET: When Data byte transfer succeeded.
  * @rmtoll SR1          BTF           LL_I2C_IsActiveFlag_BTF
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_BTF(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_BTF) == (I2C_SR1_BTF));
}

/**
  * @brief  Indicate the status of Receive data register not empty flag.
  * @note   RESET: When Receive data register is read.
  *         SET: When the received data is copied in Receive data register.
  * @rmtoll SR1          RXNE          LL_I2C_IsActiveFlag_RXNE
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_RXNE(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_RXNE) == (I2C_SR1_RXNE));
}

/**
  * @brief  Indicate the status of Start Bit (master mode).
  * @note   RESET: When No Start condition.
  *         SET: When Start condition is generated.
  * @rmtoll SR1          SB            LL_I2C_IsActiveFlag_SB
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_SB(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_SB) == (I2C_SR1_SB));
}

/**
  * @brief  Indicate the status of Address sent (master mode) or Address matched flag (slave mode).
  * @note   RESET: Clear default value.
  *         SET: When the address is fully sent (master mode) or when the received slave address matched with one of the enabled slave address (slave mode).
  * @rmtoll SR1          ADDR          LL_I2C_IsActiveFlag_ADDR
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_ADDR(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_ADDR) == (I2C_SR1_ADDR));
}

/**
  * @brief  Indicate the status of 10-bit header sent (master mode).
  * @note   RESET: When no ADD10 event occurred.
  *         SET: When the master has sent the first address byte (header).
  * @rmtoll SR1          ADD10         LL_I2C_IsActiveFlag_ADD10
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_ADD10(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_ADD10) == (I2C_SR1_ADD10));
}

/**
  * @brief  Indicate the status of Acknowledge failure flag.
  * @note   RESET: No acknowledge failure.
  *         SET: When an acknowledge failure is received after a byte transmission.
  * @rmtoll SR1          AF            LL_I2C_IsActiveFlag_AF
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_AF(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_AF) == (I2C_SR1_AF));
}

/**
  * @brief  Indicate the status of Stop detection flag (slave mode).
  * @note   RESET: Clear default value.
  *         SET: When a Stop condition is detected.
  * @rmtoll SR1          STOPF         LL_I2C_IsActiveFlag_STOP
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_STOP(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_STOPF) == (I2C_SR1_STOPF));
}

/**
  * @brief  Indicate the status of Bus error flag.
  * @note   RESET: Clear default value.
  *         SET: When a misplaced Start or Stop condition is detected.
  * @rmtoll SR1          BERR          LL_I2C_IsActiveFlag_BERR
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_BERR(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_BERR) == (I2C_SR1_BERR));
}

/**
  * @brief  Indicate the status of Arbitration lost flag.
  * @note   RESET: Clear default value.
  *         SET: When arbitration lost.
  * @rmtoll SR1          ARLO          LL_I2C_IsActiveFlag_ARLO
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_ARLO(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_ARLO) == (I2C_SR1_ARLO));
}

/**
  * @brief  Indicate the status of Overrun/Underrun flag.
  * @note   RESET: Clear default value.
  *         SET: When an overrun/underrun error occurs (Clock Stretching Disabled).
  * @rmtoll SR1          OVR           LL_I2C_IsActiveFlag_OVR
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_OVR(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_OVR) == (I2C_SR1_OVR));
}

/**
  * @brief  Indicate the status of SMBus PEC error flag in reception.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll SR1          PECERR        LL_I2C_IsActiveSMBusFlag_PECERR
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveSMBusFlag_PECERR(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_PECERR) == (I2C_SR1_PECERR));
}

/**
  * @brief  Indicate the status of SMBus Timeout detection flag.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll SR1          TIMEOUT       LL_I2C_IsActiveSMBusFlag_TIMEOUT
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveSMBusFlag_TIMEOUT(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_TIMEOUT) == (I2C_SR1_TIMEOUT));
}

/**
  * @brief  Indicate the status of SMBus alert flag.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll SR1          SMBALERT      LL_I2C_IsActiveSMBusFlag_ALERT
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveSMBusFlag_ALERT(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR1, I2C_SR1_SMBALERT) == (I2C_SR1_SMBALERT));
}

/**
  * @brief  Indicate the status of Bus Busy flag.
  * @note   RESET: Clear default value.
  *         SET: When a Start condition is detected.
  * @rmtoll SR2          BUSY          LL_I2C_IsActiveFlag_BUSY
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_BUSY(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR2, I2C_SR2_BUSY) == (I2C_SR2_BUSY));
}

/**
  * @brief  Indicate the status of Dual flag.
  * @note   RESET: Received address matched with OAR1.
  *         SET: Received address matched with OAR2.
  * @rmtoll SR2          DUALF         LL_I2C_IsActiveFlag_DUAL
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_DUAL(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR2, I2C_SR2_DUALF) == (I2C_SR2_DUALF));
}

/**
  * @brief  Indicate the status of SMBus Host address reception (Slave mode).
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @note   RESET: No SMBus Host address
  *         SET: SMBus Host address received.
  * @note   This status is cleared by hardware after a STOP condition or repeated START condition.
  * @rmtoll SR2          SMBHOST       LL_I2C_IsActiveSMBusFlag_SMBHOST
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveSMBusFlag_SMBHOST(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR2, I2C_SR2_SMBHOST) == (I2C_SR2_SMBHOST));
}

/**
  * @brief  Indicate the status of SMBus Device default address reception (Slave mode).
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @note   RESET: No SMBus Device default address
  *         SET: SMBus Device default address received.
  * @note   This status is cleared by hardware after a STOP condition or repeated START condition.
  * @rmtoll SR2          SMBDEFAULT    LL_I2C_IsActiveSMBusFlag_SMBDEFAULT
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveSMBusFlag_SMBDEFAULT(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR2, I2C_SR2_SMBDEFAULT) == (I2C_SR2_SMBDEFAULT));
}

/**
  * @brief  Indicate the status of General call address reception (Slave mode).
  * @note   RESET: No General call address
  *         SET: General call address received.
  * @note   This status is cleared by hardware after a STOP condition or repeated START condition.
  * @rmtoll SR2          GENCALL       LL_I2C_IsActiveFlag_GENCALL
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_GENCALL(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR2, I2C_SR2_GENCALL) == (I2C_SR2_GENCALL));
}

/**
  * @brief  Indicate the status of Master/Slave flag.
  * @note   RESET: Slave Mode.
  *         SET: Master Mode.
  * @rmtoll SR2          MSL           LL_I2C_IsActiveFlag_MSL
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsActiveFlag_MSL(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->SR2, I2C_SR2_MSL) == (I2C_SR2_MSL));
}

/**
  * @brief  Clear Address Matched flag.
  * @note   Clearing this flag is done by a read access to the I2Cx_SR1
  *         register followed by a read access to the I2Cx_SR2 register.
  * @rmtoll SR1          ADDR          LL_I2C_ClearFlag_ADDR
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ClearFlag_ADDR(I2C_TypeDef *I2Cx)
{
  __IO uint32_t tmpreg;
  tmpreg = I2Cx->SR1;
  (void) tmpreg;
  tmpreg = I2Cx->SR2;
  (void) tmpreg;
}

/**
  * @brief  Clear Acknowledge failure flag.
  * @rmtoll SR1          AF            LL_I2C_ClearFlag_AF
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ClearFlag_AF(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->SR1, I2C_SR1_AF);
}

/**
  * @brief  Clear Stop detection flag.
  * @note   Clearing this flag is done by a read access to the I2Cx_SR1
  *         register followed by a write access to I2Cx_CR1 register.
  * @rmtoll SR1          STOPF         LL_I2C_ClearFlag_STOP\n
  *         CR1          PE            LL_I2C_ClearFlag_STOP
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ClearFlag_STOP(I2C_TypeDef *I2Cx)
{
  __IO uint32_t tmpreg;
  tmpreg = I2Cx->SR1;
  (void) tmpreg;
  SET_BIT(I2Cx->CR1, I2C_CR1_PE);
}

/**
  * @brief  Clear Bus error flag.
  * @rmtoll SR1          BERR          LL_I2C_ClearFlag_BERR
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ClearFlag_BERR(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->SR1, I2C_SR1_BERR);
}

/**
  * @brief  Clear Arbitration lost flag.
  * @rmtoll SR1          ARLO          LL_I2C_ClearFlag_ARLO
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ClearFlag_ARLO(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->SR1, I2C_SR1_ARLO);
}

/**
  * @brief  Clear Overrun/Underrun flag.
  * @rmtoll SR1          OVR           LL_I2C_ClearFlag_OVR
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ClearFlag_OVR(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->SR1, I2C_SR1_OVR);
}

/**
  * @brief  Clear SMBus PEC error flag.
  * @rmtoll SR1          PECERR        LL_I2C_ClearSMBusFlag_PECERR
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ClearSMBusFlag_PECERR(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->SR1, I2C_SR1_PECERR);
}

/**
  * @brief  Clear SMBus Timeout detection flag.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll SR1          TIMEOUT       LL_I2C_ClearSMBusFlag_TIMEOUT
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ClearSMBusFlag_TIMEOUT(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->SR1, I2C_SR1_TIMEOUT);
}

/**
  * @brief  Clear SMBus Alert flag.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll SR1          SMBALERT      LL_I2C_ClearSMBusFlag_ALERT
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_ClearSMBusFlag_ALERT(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->SR1, I2C_SR1_SMBALERT);
}

/**
  * @}
  */

/** @defgroup I2C_LL_EF_Data_Management Data_Management
  * @{
  */

/**
  * @brief  Enable Reset of I2C peripheral.
  * @rmtoll CR1          SWRST         LL_I2C_EnableReset
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableReset(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_SWRST);
}

/**
  * @brief  Disable Reset of I2C peripheral.
  * @rmtoll CR1          SWRST         LL_I2C_DisableReset
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableReset(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR1, I2C_CR1_SWRST);
}

/**
  * @brief  Check if the I2C peripheral is under reset state or not.
  * @rmtoll CR1          SWRST         LL_I2C_IsResetEnabled
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsResetEnabled(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR1, I2C_CR1_SWRST) == (I2C_CR1_SWRST));
}

/**
  * @brief  Prepare the generation of a ACKnowledge or Non ACKnowledge condition after the address receive match code or next received byte.
  * @note   Usage in Slave or Master mode.
  * @rmtoll CR1          ACK           LL_I2C_AcknowledgeNextData
  * @param  I2Cx I2C Instance.
  * @param  TypeAcknowledge This parameter can be one of the following values:
  *         @arg @ref LL_I2C_ACK
  *         @arg @ref LL_I2C_NACK
  * @retval None
  */
__STATIC_INLINE void LL_I2C_AcknowledgeNextData(I2C_TypeDef *I2Cx, uint32_t TypeAcknowledge)
{
  MODIFY_REG(I2Cx->CR1, I2C_CR1_ACK, TypeAcknowledge);
}

/**
  * @brief  Generate a START or RESTART condition
  * @note   The START bit can be set even if bus is BUSY or I2C is in slave mode.
  *         This action has no effect when RELOAD is set.
  * @rmtoll CR1          START         LL_I2C_GenerateStartCondition
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_GenerateStartCondition(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_START);
}

/**
  * @brief  Generate a STOP condition after the current byte transfer (master mode).
  * @rmtoll CR1          STOP          LL_I2C_GenerateStopCondition
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_GenerateStopCondition(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_STOP);
}

/**
  * @brief  Enable bit POS (master/host mode).
  * @note   In that case, the ACK bit controls the (N)ACK of the next byte received or the PEC bit indicates that the next byte in shift register is a PEC.
  * @rmtoll CR1          POS           LL_I2C_EnableBitPOS
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableBitPOS(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_POS);
}

/**
  * @brief  Disable bit POS (master/host mode).
  * @note   In that case, the ACK bit controls the (N)ACK of the current byte received or the PEC bit indicates that the current byte in shift register is a PEC.
  * @rmtoll CR1          POS           LL_I2C_DisableBitPOS
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableBitPOS(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR1, I2C_CR1_POS);
}

/**
  * @brief  Check if bit POS  is enabled or disabled.
  * @rmtoll CR1          POS           LL_I2C_IsEnabledBitPOS
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledBitPOS(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR1, I2C_CR1_POS) == (I2C_CR1_POS));
}

/**
  * @brief  Indicate the value of transfer direction.
  * @note   RESET: Bus is in read transfer (peripheral point of view).
  *         SET: Bus is in write transfer (peripheral point of view).
  * @rmtoll SR2          TRA           LL_I2C_GetTransferDirection
  * @param  I2Cx I2C Instance.
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_I2C_DIRECTION_WRITE
  *         @arg @ref LL_I2C_DIRECTION_READ
  */
__STATIC_INLINE uint32_t LL_I2C_GetTransferDirection(I2C_TypeDef *I2Cx)
{
  return (uint32_t)(READ_BIT(I2Cx->SR2, I2C_SR2_TRA));
}

/**
  * @brief  Enable DMA last transfer.
  * @note   This action mean that next DMA EOT is the last transfer.
  * @rmtoll CR2          LAST          LL_I2C_EnableLastDMA
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableLastDMA(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR2, I2C_CR2_LAST);
}

/**
  * @brief  Disable DMA last transfer.
  * @note   This action mean that next DMA EOT is not the last transfer.
  * @rmtoll CR2          LAST          LL_I2C_DisableLastDMA
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableLastDMA(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR2, I2C_CR2_LAST);
}

/**
  * @brief  Check if DMA last transfer is enabled or disabled.
  * @rmtoll CR2          LAST          LL_I2C_IsEnabledLastDMA
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledLastDMA(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR2, I2C_CR2_LAST) == (I2C_CR2_LAST));
}

/**
  * @brief  Enable transfer or internal comparison of the SMBus Packet Error byte (transmission or reception mode).
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @note   This feature is cleared by hardware when the PEC byte is transferred or compared,
  *         or by a START or STOP condition, it is also cleared by software.
  * @rmtoll CR1          PEC           LL_I2C_EnableSMBusPECCompare
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_EnableSMBusPECCompare(I2C_TypeDef *I2Cx)
{
  SET_BIT(I2Cx->CR1, I2C_CR1_PEC);
}

/**
  * @brief  Disable transfer or internal comparison of the SMBus Packet Error byte (transmission or reception mode).
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll CR1          PEC           LL_I2C_DisableSMBusPECCompare
  * @param  I2Cx I2C Instance.
  * @retval None
  */
__STATIC_INLINE void LL_I2C_DisableSMBusPECCompare(I2C_TypeDef *I2Cx)
{
  CLEAR_BIT(I2Cx->CR1, I2C_CR1_PEC);
}

/**
  * @brief  Check if the SMBus Packet Error byte transfer or internal comparison is requested or not.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll CR1          PEC           LL_I2C_IsEnabledSMBusPECCompare
  * @param  I2Cx I2C Instance.
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_I2C_IsEnabledSMBusPECCompare(I2C_TypeDef *I2Cx)
{
  return (READ_BIT(I2Cx->CR1, I2C_CR1_PEC) == (I2C_CR1_PEC));
}

/**
  * @brief  Get the SMBus Packet Error byte calculated.
  * @note   Macro @ref IS_SMBUS_ALL_INSTANCE(I2Cx) can be used to check whether or not
  *         SMBus feature is supported by the I2Cx Instance.
  * @rmtoll SR2          PEC           LL_I2C_GetSMBusPEC
  * @param  I2Cx I2C Instance.
  * @retval Value between Min_Data=0x00 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_I2C_GetSMBusPEC(I2C_TypeDef *I2Cx)
{
  return (uint32_t)(READ_BIT(I2Cx->SR2, I2C_SR2_PEC) >> I2C_SR2_PEC_Pos);
}

/**
  * @brief  Read Receive Data register.
  * @rmtoll DR           DR            LL_I2C_ReceiveData8
  * @param  I2Cx I2C Instance.
  * @retval Value between Min_Data=0x0 and Max_Data=0xFF
  */
__STATIC_INLINE uint8_t LL_I2C_ReceiveData8(I2C_TypeDef *I2Cx)
{
  return (uint8_t)(READ_BIT(I2Cx->DR, I2C_DR_DR));
}

/**
  * @brief  Write in Transmit Data Register .
  * @rmtoll DR           DR            LL_I2C_TransmitData8
  * @param  I2Cx I2C Instance.
  * @param  Data Value between Min_Data=0x0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_I2C_TransmitData8(I2C_TypeDef *I2Cx, uint8_t Data)
{
  MODIFY_REG(I2Cx->DR, I2C_DR_DR, Data);
}

/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup I2C_LL_EF_Init Initialization and de-initialization functions
  * @{
  */

uint32_t LL_I2C_Init(I2C_TypeDef *I2Cx, LL_I2C_InitTypeDef *I2C_InitStruct);
uint32_t LL_I2C_DeInit(I2C_TypeDef *I2Cx);
void LL_I2C_StructInit(LL_I2C_InitTypeDef *I2C_InitStruct);


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

#endif /* I2C1 || I2C2 || I2C3 */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_LL_I2C_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
