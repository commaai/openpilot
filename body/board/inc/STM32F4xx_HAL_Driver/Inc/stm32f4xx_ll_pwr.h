/**
  ******************************************************************************
  * @file    stm32f4xx_ll_pwr.h
  * @author  MCD Application Team
  * @brief   Header file of PWR LL module.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM32F4xx_LL_PWR_H
#define __STM32F4xx_LL_PWR_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined(PWR)

/** @defgroup PWR_LL PWR
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
/** @defgroup PWR_LL_Exported_Constants PWR Exported Constants
  * @{
  */

/** @defgroup PWR_LL_EC_CLEAR_FLAG Clear Flags Defines
  * @brief    Flags defines which can be used with LL_PWR_WriteReg function
  * @{
  */
#define LL_PWR_CR_CSBF                     PWR_CR_CSBF            /*!< Clear standby flag */
#define LL_PWR_CR_CWUF                     PWR_CR_CWUF            /*!< Clear wakeup flag */
/**
  * @}
  */

/** @defgroup PWR_LL_EC_GET_FLAG Get Flags Defines
  * @brief    Flags defines which can be used with LL_PWR_ReadReg function
  * @{
  */
#define LL_PWR_CSR_WUF                     PWR_CSR_WUF            /*!< Wakeup flag */
#define LL_PWR_CSR_SBF                     PWR_CSR_SBF            /*!< Standby flag */
#define LL_PWR_CSR_PVDO                    PWR_CSR_PVDO           /*!< Power voltage detector output flag */
#define LL_PWR_CSR_VOS                     PWR_CSR_VOSRDY            /*!< Voltage scaling select flag */
#if defined(PWR_CSR_EWUP)
#define LL_PWR_CSR_EWUP1                   PWR_CSR_EWUP           /*!< Enable WKUP pin */
#elif defined(PWR_CSR_EWUP1)
#define LL_PWR_CSR_EWUP1                   PWR_CSR_EWUP1          /*!< Enable WKUP pin 1 */
#endif /* PWR_CSR_EWUP */
#if defined(PWR_CSR_EWUP2)
#define LL_PWR_CSR_EWUP2                   PWR_CSR_EWUP2          /*!< Enable WKUP pin 2 */
#endif /* PWR_CSR_EWUP2 */
#if defined(PWR_CSR_EWUP3)
#define LL_PWR_CSR_EWUP3                   PWR_CSR_EWUP3          /*!< Enable WKUP pin 3 */
#endif /* PWR_CSR_EWUP3 */
/**
  * @}
  */

/** @defgroup PWR_LL_EC_REGU_VOLTAGE Regulator Voltage
  * @{
  */
#if defined(PWR_CR_VOS_0)
#define LL_PWR_REGU_VOLTAGE_SCALE3         (PWR_CR_VOS_0)
#define LL_PWR_REGU_VOLTAGE_SCALE2         (PWR_CR_VOS_1)
#define LL_PWR_REGU_VOLTAGE_SCALE1         (PWR_CR_VOS_0 | PWR_CR_VOS_1) /* The SCALE1 is not available for STM32F401xx devices */
#else
#define LL_PWR_REGU_VOLTAGE_SCALE1         (PWR_CR_VOS)
#define LL_PWR_REGU_VOLTAGE_SCALE2         0x00000000U
#endif /* PWR_CR_VOS_0 */
/**
  * @}
  */

/** @defgroup PWR_LL_EC_MODE_PWR Mode Power
  * @{
  */
#define LL_PWR_MODE_STOP_MAINREGU             0x00000000U                    /*!< Enter Stop mode when the CPU enters deepsleep */
#define LL_PWR_MODE_STOP_LPREGU               (PWR_CR_LPDS)                  /*!< Enter Stop mode (with low power Regulator ON) when the CPU enters deepsleep */
#if defined(PWR_CR_MRUDS) && defined(PWR_CR_LPUDS) && defined(PWR_CR_FPDS)
#define LL_PWR_MODE_STOP_MAINREGU_UNDERDRIVE  (PWR_CR_MRUDS | PWR_CR_FPDS)                 /*!< Enter Stop mode (with main Regulator in under-drive mode) when the CPU enters deepsleep */
#define LL_PWR_MODE_STOP_LPREGU_UNDERDRIVE    (PWR_CR_LPDS | PWR_CR_LPUDS | PWR_CR_FPDS)   /*!< Enter Stop mode (with low power Regulator in under-drive mode) when the CPU enters deepsleep */
#endif /* PWR_CR_MRUDS && PWR_CR_LPUDS && PWR_CR_FPDS */
#if defined(PWR_CR_MRLVDS) && defined(PWR_CR_LPLVDS) && defined(PWR_CR_FPDS)
#define LL_PWR_MODE_STOP_MAINREGU_DEEPSLEEP  (PWR_CR_MRLVDS | PWR_CR_FPDS)                 /*!< Enter Stop mode (with main Regulator in Deep Sleep mode) when the CPU enters deepsleep */
#define LL_PWR_MODE_STOP_LPREGU_DEEPSLEEP    (PWR_CR_LPDS | PWR_CR_LPLVDS | PWR_CR_FPDS)   /*!< Enter Stop mode (with low power Regulator in Deep Sleep mode) when the CPU enters deepsleep */
#endif /* PWR_CR_MRLVDS && PWR_CR_LPLVDS && PWR_CR_FPDS */
#define LL_PWR_MODE_STANDBY                   (PWR_CR_PDDS)                  /*!< Enter Standby mode when the CPU enters deepsleep */
/**
  * @}
  */

/** @defgroup PWR_LL_EC_REGU_MODE_DS_MODE  Regulator Mode In Deep Sleep Mode
 * @{
 */
#define LL_PWR_REGU_DSMODE_MAIN        0x00000000U           /*!< Voltage Regulator in main mode during deepsleep mode */
#define LL_PWR_REGU_DSMODE_LOW_POWER   (PWR_CR_LPDS)         /*!< Voltage Regulator in low-power mode during deepsleep mode */
/**
  * @}
  */

/** @defgroup PWR_LL_EC_PVDLEVEL Power Voltage Detector Level
  * @{
  */
#define LL_PWR_PVDLEVEL_0                  (PWR_CR_PLS_LEV0)      /*!< Voltage threshold detected by PVD 2.2 V */
#define LL_PWR_PVDLEVEL_1                  (PWR_CR_PLS_LEV1)      /*!< Voltage threshold detected by PVD 2.3 V */
#define LL_PWR_PVDLEVEL_2                  (PWR_CR_PLS_LEV2)      /*!< Voltage threshold detected by PVD 2.4 V */
#define LL_PWR_PVDLEVEL_3                  (PWR_CR_PLS_LEV3)      /*!< Voltage threshold detected by PVD 2.5 V */
#define LL_PWR_PVDLEVEL_4                  (PWR_CR_PLS_LEV4)      /*!< Voltage threshold detected by PVD 2.6 V */
#define LL_PWR_PVDLEVEL_5                  (PWR_CR_PLS_LEV5)      /*!< Voltage threshold detected by PVD 2.7 V */
#define LL_PWR_PVDLEVEL_6                  (PWR_CR_PLS_LEV6)      /*!< Voltage threshold detected by PVD 2.8 V */
#define LL_PWR_PVDLEVEL_7                  (PWR_CR_PLS_LEV7)      /*!< Voltage threshold detected by PVD 2.9 V */
/**
  * @}
  */
/** @defgroup PWR_LL_EC_WAKEUP_PIN  Wakeup Pins
  * @{
  */
#if defined(PWR_CSR_EWUP)
#define LL_PWR_WAKEUP_PIN1                 (PWR_CSR_EWUP)         /*!< WKUP pin : PA0 */
#endif /* PWR_CSR_EWUP */
#if defined(PWR_CSR_EWUP1)
#define LL_PWR_WAKEUP_PIN1                 (PWR_CSR_EWUP1)        /*!< WKUP pin 1 : PA0 */
#endif /* PWR_CSR_EWUP1 */
#if defined(PWR_CSR_EWUP2)
#define LL_PWR_WAKEUP_PIN2                 (PWR_CSR_EWUP2)        /*!< WKUP pin 2 : PC0 or PC13 according to device */
#endif /* PWR_CSR_EWUP2 */
#if defined(PWR_CSR_EWUP3)
#define LL_PWR_WAKEUP_PIN3                 (PWR_CSR_EWUP3)        /*!< WKUP pin 3 : PC1 */
#endif /* PWR_CSR_EWUP3 */
/**
  * @}
  */

/**
  * @}
  */


/* Exported macro ------------------------------------------------------------*/
/** @defgroup PWR_LL_Exported_Macros PWR Exported Macros
  * @{
  */

/** @defgroup PWR_LL_EM_WRITE_READ Common write and read registers Macros
  * @{
  */

/**
  * @brief  Write a value in PWR register
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_PWR_WriteReg(__REG__, __VALUE__) WRITE_REG(PWR->__REG__, (__VALUE__))

/**
  * @brief  Read a value in PWR register
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_PWR_ReadReg(__REG__) READ_REG(PWR->__REG__)
/**
  * @}
  */

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup PWR_LL_Exported_Functions PWR Exported Functions
  * @{
  */

/** @defgroup PWR_LL_EF_Configuration Configuration
  * @{
  */
#if defined(PWR_CR_FISSR)
/**
  * @brief  Enable FLASH interface STOP while system Run is ON
  * @rmtoll CR    FISSR       LL_PWR_EnableFLASHInterfaceSTOP
  * @note  This mode is enabled only with STOP low power mode.
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableFLASHInterfaceSTOP(void)
{
  SET_BIT(PWR->CR, PWR_CR_FISSR);
}

/**
  * @brief  Disable FLASH Interface STOP while system Run is ON
  * @rmtoll CR    FISSR       LL_PWR_DisableFLASHInterfaceSTOP
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableFLASHInterfaceSTOP(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_FISSR);
}

/**
  * @brief  Check if FLASH Interface STOP while system Run feature is enabled
  * @rmtoll CR    FISSR       LL_PWR_IsEnabledFLASHInterfaceSTOP
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledFLASHInterfaceSTOP(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_FISSR) == (PWR_CR_FISSR));
}
#endif /* PWR_CR_FISSR */

#if defined(PWR_CR_FMSSR)
/**
  * @brief  Enable FLASH Memory STOP while system Run is ON
  * @rmtoll CR    FMSSR       LL_PWR_EnableFLASHMemorySTOP
  * @note  This mode is enabled only with STOP low power mode.
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableFLASHMemorySTOP(void)
{
  SET_BIT(PWR->CR, PWR_CR_FMSSR);
}

/**
  * @brief  Disable FLASH Memory STOP while system Run is ON
  * @rmtoll CR    FMSSR       LL_PWR_DisableFLASHMemorySTOP
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableFLASHMemorySTOP(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_FMSSR);
}

/**
  * @brief  Check if FLASH Memory STOP while system Run feature is enabled
  * @rmtoll CR    FMSSR       LL_PWR_IsEnabledFLASHMemorySTOP
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledFLASHMemorySTOP(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_FMSSR) == (PWR_CR_FMSSR));
}
#endif /* PWR_CR_FMSSR */
#if defined(PWR_CR_UDEN)
/**
  * @brief  Enable Under Drive Mode
  * @rmtoll CR    UDEN       LL_PWR_EnableUnderDriveMode
  * @note  This mode is enabled only with STOP low power mode.
  *        In this mode, the 1.2V domain is preserved in reduced leakage mode. This 
  *        mode is only available when the main Regulator or the low power Regulator 
  *        is in low voltage mode.      
  * @note  If the Under-drive mode was enabled, it is automatically disabled after 
  *        exiting Stop mode. 
  *        When the voltage Regulator operates in Under-drive mode, an additional  
  *        startup delay is induced when waking up from Stop mode.
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableUnderDriveMode(void)
{
  SET_BIT(PWR->CR, PWR_CR_UDEN);
}

/**
  * @brief  Disable Under Drive Mode
  * @rmtoll CR    UDEN       LL_PWR_DisableUnderDriveMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableUnderDriveMode(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_UDEN);
}

/**
  * @brief  Check if Under Drive Mode is enabled
  * @rmtoll CR    UDEN       LL_PWR_IsEnabledUnderDriveMode
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledUnderDriveMode(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_UDEN) == (PWR_CR_UDEN));
}
#endif /* PWR_CR_UDEN */

#if defined(PWR_CR_ODSWEN)
/**
  * @brief  Enable Over drive switching
  * @rmtoll CR    ODSWEN       LL_PWR_EnableOverDriveSwitching
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableOverDriveSwitching(void)
{
  SET_BIT(PWR->CR, PWR_CR_ODSWEN);
}

/**
  * @brief  Disable Over drive switching
  * @rmtoll CR    ODSWEN       LL_PWR_DisableOverDriveSwitching
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableOverDriveSwitching(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_ODSWEN);
}

/**
  * @brief  Check if Over drive switching is enabled
  * @rmtoll CR    ODSWEN       LL_PWR_IsEnabledOverDriveSwitching
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledOverDriveSwitching(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_ODSWEN) == (PWR_CR_ODSWEN));
}
#endif /* PWR_CR_ODSWEN */
#if defined(PWR_CR_ODEN)
/**
  * @brief  Enable Over drive Mode
  * @rmtoll CR    ODEN       LL_PWR_EnableOverDriveMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableOverDriveMode(void)
{
  SET_BIT(PWR->CR, PWR_CR_ODEN);
}

/**
  * @brief  Disable Over drive Mode
  * @rmtoll CR    ODEN       LL_PWR_DisableOverDriveMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableOverDriveMode(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_ODEN);
}

/**
  * @brief  Check if Over drive switching is enabled
  * @rmtoll CR    ODEN       LL_PWR_IsEnabledOverDriveMode
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledOverDriveMode(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_ODEN) == (PWR_CR_ODEN));
}
#endif /* PWR_CR_ODEN */
#if defined(PWR_CR_MRUDS)
/**
  * @brief  Enable Main Regulator in deepsleep under-drive Mode
  * @rmtoll CR    MRUDS       LL_PWR_EnableMainRegulatorDeepSleepUDMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableMainRegulatorDeepSleepUDMode(void)
{
  SET_BIT(PWR->CR, PWR_CR_MRUDS);
}

/**
  * @brief  Disable Main Regulator in deepsleep under-drive Mode
  * @rmtoll CR    MRUDS       LL_PWR_DisableMainRegulatorDeepSleepUDMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableMainRegulatorDeepSleepUDMode(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_MRUDS);
}

/**
  * @brief  Check if Main Regulator in deepsleep under-drive Mode is enabled
  * @rmtoll CR    MRUDS       LL_PWR_IsEnabledMainRegulatorDeepSleepUDMode
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledMainRegulatorDeepSleepUDMode(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_MRUDS) == (PWR_CR_MRUDS));
}
#endif /* PWR_CR_MRUDS */

#if defined(PWR_CR_LPUDS)
/**
  * @brief  Enable Low Power Regulator in deepsleep under-drive Mode
  * @rmtoll CR    LPUDS       LL_PWR_EnableLowPowerRegulatorDeepSleepUDMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableLowPowerRegulatorDeepSleepUDMode(void)
{
  SET_BIT(PWR->CR, PWR_CR_LPUDS);
}

/**
  * @brief  Disable Low Power Regulator in deepsleep under-drive Mode
  * @rmtoll CR    LPUDS       LL_PWR_DisableLowPowerRegulatorDeepSleepUDMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableLowPowerRegulatorDeepSleepUDMode(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_LPUDS);
}

/**
  * @brief  Check if Low Power Regulator in deepsleep under-drive Mode is enabled
  * @rmtoll CR    LPUDS       LL_PWR_IsEnabledLowPowerRegulatorDeepSleepUDMode
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledLowPowerRegulatorDeepSleepUDMode(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_LPUDS) == (PWR_CR_LPUDS));
}
#endif /* PWR_CR_LPUDS */

#if defined(PWR_CR_MRLVDS)
/**
  * @brief  Enable Main Regulator low voltage Mode
  * @rmtoll CR    MRLVDS       LL_PWR_EnableMainRegulatorLowVoltageMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableMainRegulatorLowVoltageMode(void)
{
  SET_BIT(PWR->CR, PWR_CR_MRLVDS);
}

/**
  * @brief  Disable Main Regulator low voltage Mode
  * @rmtoll CR    MRLVDS       LL_PWR_DisableMainRegulatorLowVoltageMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableMainRegulatorLowVoltageMode(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_MRLVDS);
}

/**
  * @brief  Check if Main Regulator low voltage Mode is enabled
  * @rmtoll CR    MRLVDS       LL_PWR_IsEnabledMainRegulatorLowVoltageMode
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledMainRegulatorLowVoltageMode(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_MRLVDS) == (PWR_CR_MRLVDS));
}
#endif /* PWR_CR_MRLVDS */

#if defined(PWR_CR_LPLVDS)
/**
  * @brief  Enable Low Power Regulator low voltage Mode
  * @rmtoll CR    LPLVDS       LL_PWR_EnableLowPowerRegulatorLowVoltageMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableLowPowerRegulatorLowVoltageMode(void)
{
  SET_BIT(PWR->CR, PWR_CR_LPLVDS);
}

/**
  * @brief  Disable Low Power Regulator low voltage Mode
  * @rmtoll CR    LPLVDS       LL_PWR_DisableLowPowerRegulatorLowVoltageMode
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableLowPowerRegulatorLowVoltageMode(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_LPLVDS);
}

/**
  * @brief  Check if Low Power Regulator low voltage Mode is enabled
  * @rmtoll CR    LPLVDS       LL_PWR_IsEnabledLowPowerRegulatorLowVoltageMode
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledLowPowerRegulatorLowVoltageMode(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_LPLVDS) == (PWR_CR_LPLVDS));
}
#endif /* PWR_CR_LPLVDS */
/**
  * @brief  Set the main internal Regulator output voltage
  * @rmtoll CR    VOS       LL_PWR_SetRegulVoltageScaling
  * @param  VoltageScaling This parameter can be one of the following values:
  *         @arg @ref LL_PWR_REGU_VOLTAGE_SCALE1 (*)
  *         @arg @ref LL_PWR_REGU_VOLTAGE_SCALE2
  *         @arg @ref LL_PWR_REGU_VOLTAGE_SCALE3
  *         (*) LL_PWR_REGU_VOLTAGE_SCALE1 is not available for STM32F401xx devices
  * @retval None
  */
__STATIC_INLINE void LL_PWR_SetRegulVoltageScaling(uint32_t VoltageScaling)
{
  MODIFY_REG(PWR->CR, PWR_CR_VOS, VoltageScaling);
}

/**
  * @brief  Get the main internal Regulator output voltage
  * @rmtoll CR    VOS       LL_PWR_GetRegulVoltageScaling
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_PWR_REGU_VOLTAGE_SCALE1 (*)
  *         @arg @ref LL_PWR_REGU_VOLTAGE_SCALE2
  *         @arg @ref LL_PWR_REGU_VOLTAGE_SCALE3
  *         (*) LL_PWR_REGU_VOLTAGE_SCALE1 is not available for STM32F401xx devices
  */
__STATIC_INLINE uint32_t LL_PWR_GetRegulVoltageScaling(void)
{
  return (uint32_t)(READ_BIT(PWR->CR, PWR_CR_VOS));
}
/**
  * @brief  Enable the Flash Power Down in Stop Mode
  * @rmtoll CR    FPDS       LL_PWR_EnableFlashPowerDown
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableFlashPowerDown(void)
{
  SET_BIT(PWR->CR, PWR_CR_FPDS);
}

/**
  * @brief  Disable the Flash Power Down in Stop Mode
  * @rmtoll CR    FPDS       LL_PWR_DisableFlashPowerDown
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableFlashPowerDown(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_FPDS);
}

/**
  * @brief  Check if the Flash Power Down in Stop Mode is enabled
  * @rmtoll CR    FPDS       LL_PWR_IsEnabledFlashPowerDown
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledFlashPowerDown(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_FPDS) == (PWR_CR_FPDS));
}

/**
  * @brief  Enable access to the backup domain
  * @rmtoll CR    DBP       LL_PWR_EnableBkUpAccess
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableBkUpAccess(void)
{
  SET_BIT(PWR->CR, PWR_CR_DBP);
}

/**
  * @brief  Disable access to the backup domain
  * @rmtoll CR    DBP       LL_PWR_DisableBkUpAccess
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableBkUpAccess(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_DBP);
}

/**
  * @brief  Check if the backup domain is enabled
  * @rmtoll CR    DBP       LL_PWR_IsEnabledBkUpAccess
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledBkUpAccess(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_DBP) == (PWR_CR_DBP));
}
/**
  * @brief  Enable the backup Regulator
  * @rmtoll CSR    BRE       LL_PWR_EnableBkUpRegulator
  * @note The BRE bit of the PWR_CSR register is protected against parasitic write access.
  * The LL_PWR_EnableBkUpAccess() must be called before using this API.
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableBkUpRegulator(void)
{
  SET_BIT(PWR->CSR, PWR_CSR_BRE);
}

/**
  * @brief  Disable the backup Regulator
  * @rmtoll CSR    BRE       LL_PWR_DisableBkUpRegulator
  * @note The BRE bit of the PWR_CSR register is protected against parasitic write access.
  * The LL_PWR_EnableBkUpAccess() must be called before using this API.
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableBkUpRegulator(void)
{
  CLEAR_BIT(PWR->CSR, PWR_CSR_BRE);
}

/**
  * @brief  Check if the backup Regulator is enabled
  * @rmtoll CSR    BRE       LL_PWR_IsEnabledBkUpRegulator
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledBkUpRegulator(void)
{
  return (READ_BIT(PWR->CSR, PWR_CSR_BRE) == (PWR_CSR_BRE));
}

/**
  * @brief  Set voltage Regulator mode during deep sleep mode
  * @rmtoll CR    LPDS         LL_PWR_SetRegulModeDS
  * @param  RegulMode This parameter can be one of the following values:
  *         @arg @ref LL_PWR_REGU_DSMODE_MAIN
  *         @arg @ref LL_PWR_REGU_DSMODE_LOW_POWER
  * @retval None
  */
__STATIC_INLINE void LL_PWR_SetRegulModeDS(uint32_t RegulMode)
{
  MODIFY_REG(PWR->CR, PWR_CR_LPDS, RegulMode);
}

/**
  * @brief  Get voltage Regulator mode during deep sleep mode
  * @rmtoll CR    LPDS         LL_PWR_GetRegulModeDS
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_PWR_REGU_DSMODE_MAIN
  *         @arg @ref LL_PWR_REGU_DSMODE_LOW_POWER
  */
__STATIC_INLINE uint32_t LL_PWR_GetRegulModeDS(void)
{
  return (uint32_t)(READ_BIT(PWR->CR, PWR_CR_LPDS));
}

/**
  * @brief  Set Power Down mode when CPU enters deepsleep
  * @rmtoll CR    PDDS         LL_PWR_SetPowerMode\n
  * @rmtoll CR    MRUDS        LL_PWR_SetPowerMode\n
  * @rmtoll CR    LPUDS        LL_PWR_SetPowerMode\n
  * @rmtoll CR    FPDS         LL_PWR_SetPowerMode\n
  * @rmtoll CR    MRLVDS       LL_PWR_SetPowerMode\n
  * @rmtoll CR    LPlVDS       LL_PWR_SetPowerMode\n
  * @rmtoll CR    FPDS         LL_PWR_SetPowerMode\n
  * @rmtoll CR    LPDS         LL_PWR_SetPowerMode
  * @param  PDMode This parameter can be one of the following values:
  *         @arg @ref LL_PWR_MODE_STOP_MAINREGU
  *         @arg @ref LL_PWR_MODE_STOP_LPREGU
  *         @arg @ref LL_PWR_MODE_STOP_MAINREGU_UNDERDRIVE (*)
  *         @arg @ref LL_PWR_MODE_STOP_LPREGU_UNDERDRIVE (*)
  *         @arg @ref LL_PWR_MODE_STOP_MAINREGU_DEEPSLEEP (*)
  *         @arg @ref LL_PWR_MODE_STOP_LPREGU_DEEPSLEEP (*)
  *
  *         (*) not available on all devices
  *         @arg @ref LL_PWR_MODE_STANDBY
  * @retval None
  */
__STATIC_INLINE void LL_PWR_SetPowerMode(uint32_t PDMode)
{
#if defined(PWR_CR_MRUDS) && defined(PWR_CR_LPUDS) && defined(PWR_CR_FPDS)
  MODIFY_REG(PWR->CR, (PWR_CR_PDDS | PWR_CR_LPDS | PWR_CR_FPDS | PWR_CR_LPUDS | PWR_CR_MRUDS), PDMode);
#elif defined(PWR_CR_MRLVDS) && defined(PWR_CR_LPLVDS) && defined(PWR_CR_FPDS)
  MODIFY_REG(PWR->CR, (PWR_CR_PDDS | PWR_CR_LPDS | PWR_CR_FPDS | PWR_CR_LPLVDS | PWR_CR_MRLVDS), PDMode);
#else
  MODIFY_REG(PWR->CR, (PWR_CR_PDDS| PWR_CR_LPDS), PDMode);
#endif /* PWR_CR_MRUDS && PWR_CR_LPUDS && PWR_CR_FPDS */
}

/**
  * @brief  Get Power Down mode when CPU enters deepsleep
  * @rmtoll CR    PDDS         LL_PWR_GetPowerMode\n
  * @rmtoll CR    MRUDS        LL_PWR_GetPowerMode\n
  * @rmtoll CR    LPUDS        LL_PWR_GetPowerMode\n
  * @rmtoll CR    FPDS         LL_PWR_GetPowerMode\n
  * @rmtoll CR    MRLVDS       LL_PWR_GetPowerMode\n
  * @rmtoll CR    LPLVDS       LL_PWR_GetPowerMode\n
  * @rmtoll CR    FPDS         LL_PWR_GetPowerMode\n
  * @rmtoll CR    LPDS         LL_PWR_GetPowerMode
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_PWR_MODE_STOP_MAINREGU
  *         @arg @ref LL_PWR_MODE_STOP_LPREGU
  *         @arg @ref LL_PWR_MODE_STOP_MAINREGU_UNDERDRIVE (*)
  *         @arg @ref LL_PWR_MODE_STOP_LPREGU_UNDERDRIVE (*)
  *         @arg @ref LL_PWR_MODE_STOP_MAINREGU_DEEPSLEEP (*)
  *         @arg @ref LL_PWR_MODE_STOP_LPREGU_DEEPSLEEP (*)
  *
  *         (*) not available on all devices
  *         @arg @ref LL_PWR_MODE_STANDBY
  */
__STATIC_INLINE uint32_t LL_PWR_GetPowerMode(void)
{
#if defined(PWR_CR_MRUDS) && defined(PWR_CR_LPUDS) && defined(PWR_CR_FPDS)
  return (uint32_t)(READ_BIT(PWR->CR, (PWR_CR_PDDS | PWR_CR_LPDS | PWR_CR_FPDS | PWR_CR_LPUDS | PWR_CR_MRUDS)));
#elif defined(PWR_CR_MRLVDS) && defined(PWR_CR_LPLVDS) && defined(PWR_CR_FPDS)
  return (uint32_t)(READ_BIT(PWR->CR, (PWR_CR_PDDS | PWR_CR_LPDS | PWR_CR_FPDS | PWR_CR_LPLVDS | PWR_CR_MRLVDS)));
#else
  return (uint32_t)(READ_BIT(PWR->CR, (PWR_CR_PDDS| PWR_CR_LPDS)));
#endif /* PWR_CR_MRUDS && PWR_CR_LPUDS && PWR_CR_FPDS */
}

/**
  * @brief  Configure the voltage threshold detected by the Power Voltage Detector
  * @rmtoll CR    PLS       LL_PWR_SetPVDLevel
  * @param  PVDLevel This parameter can be one of the following values:
  *         @arg @ref LL_PWR_PVDLEVEL_0
  *         @arg @ref LL_PWR_PVDLEVEL_1
  *         @arg @ref LL_PWR_PVDLEVEL_2
  *         @arg @ref LL_PWR_PVDLEVEL_3
  *         @arg @ref LL_PWR_PVDLEVEL_4
  *         @arg @ref LL_PWR_PVDLEVEL_5
  *         @arg @ref LL_PWR_PVDLEVEL_6
  *         @arg @ref LL_PWR_PVDLEVEL_7
  * @retval None
  */
__STATIC_INLINE void LL_PWR_SetPVDLevel(uint32_t PVDLevel)
{
  MODIFY_REG(PWR->CR, PWR_CR_PLS, PVDLevel);
}

/**
  * @brief  Get the voltage threshold detection
  * @rmtoll CR    PLS       LL_PWR_GetPVDLevel
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_PWR_PVDLEVEL_0
  *         @arg @ref LL_PWR_PVDLEVEL_1
  *         @arg @ref LL_PWR_PVDLEVEL_2
  *         @arg @ref LL_PWR_PVDLEVEL_3
  *         @arg @ref LL_PWR_PVDLEVEL_4
  *         @arg @ref LL_PWR_PVDLEVEL_5
  *         @arg @ref LL_PWR_PVDLEVEL_6
  *         @arg @ref LL_PWR_PVDLEVEL_7
  */
__STATIC_INLINE uint32_t LL_PWR_GetPVDLevel(void)
{
  return (uint32_t)(READ_BIT(PWR->CR, PWR_CR_PLS));
}

/**
  * @brief  Enable Power Voltage Detector
  * @rmtoll CR    PVDE       LL_PWR_EnablePVD
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnablePVD(void)
{
  SET_BIT(PWR->CR, PWR_CR_PVDE);
}

/**
  * @brief  Disable Power Voltage Detector
  * @rmtoll CR    PVDE       LL_PWR_DisablePVD
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisablePVD(void)
{
  CLEAR_BIT(PWR->CR, PWR_CR_PVDE);
}

/**
  * @brief  Check if Power Voltage Detector is enabled
  * @rmtoll CR    PVDE       LL_PWR_IsEnabledPVD
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledPVD(void)
{
  return (READ_BIT(PWR->CR, PWR_CR_PVDE) == (PWR_CR_PVDE));
}

/**
  * @brief  Enable the WakeUp PINx functionality
  * @rmtoll CSR   EWUP        LL_PWR_EnableWakeUpPin\n
  * @rmtoll CSR   EWUP1       LL_PWR_EnableWakeUpPin\n
  * @rmtoll CSR   EWUP2       LL_PWR_EnableWakeUpPin\n
  * @rmtoll CSR   EWUP3       LL_PWR_EnableWakeUpPin
  * @param  WakeUpPin This parameter can be one of the following values:
  *         @arg @ref LL_PWR_WAKEUP_PIN1
  *         @arg @ref LL_PWR_WAKEUP_PIN2 (*)
  *         @arg @ref LL_PWR_WAKEUP_PIN3 (*)
  *
  *         (*) not available on all devices
  * @retval None
  */
__STATIC_INLINE void LL_PWR_EnableWakeUpPin(uint32_t WakeUpPin)
{
  SET_BIT(PWR->CSR, WakeUpPin);
}

/**
  * @brief  Disable the WakeUp PINx functionality
  * @rmtoll CSR   EWUP        LL_PWR_DisableWakeUpPin\n
  * @rmtoll CSR   EWUP1       LL_PWR_DisableWakeUpPin\n
  * @rmtoll CSR   EWUP2       LL_PWR_DisableWakeUpPin\n
  * @rmtoll CSR   EWUP3       LL_PWR_DisableWakeUpPin
  * @param  WakeUpPin This parameter can be one of the following values:
  *         @arg @ref LL_PWR_WAKEUP_PIN1
  *         @arg @ref LL_PWR_WAKEUP_PIN2 (*)
  *         @arg @ref LL_PWR_WAKEUP_PIN3 (*)
  *
  *         (*) not available on all devices
  * @retval None
  */
__STATIC_INLINE void LL_PWR_DisableWakeUpPin(uint32_t WakeUpPin)
{
  CLEAR_BIT(PWR->CSR, WakeUpPin);
}

/**
  * @brief  Check if the WakeUp PINx functionality is enabled
  * @rmtoll CSR   EWUP        LL_PWR_IsEnabledWakeUpPin\n
  * @rmtoll CSR   EWUP1       LL_PWR_IsEnabledWakeUpPin\n
  * @rmtoll CSR   EWUP2       LL_PWR_IsEnabledWakeUpPin\n
  * @rmtoll CSR   EWUP3       LL_PWR_IsEnabledWakeUpPin
  * @param  WakeUpPin This parameter can be one of the following values:
  *         @arg @ref LL_PWR_WAKEUP_PIN1
  *         @arg @ref LL_PWR_WAKEUP_PIN2 (*)
  *         @arg @ref LL_PWR_WAKEUP_PIN3 (*)
  *
  *         (*) not available on all devices
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsEnabledWakeUpPin(uint32_t WakeUpPin)
{
  return (READ_BIT(PWR->CSR, WakeUpPin) == (WakeUpPin));
}


/**
  * @}
  */

/** @defgroup PWR_LL_EF_FLAG_Management FLAG_Management
  * @{
  */

/**
  * @brief  Get Wake-up Flag
  * @rmtoll CSR   WUF       LL_PWR_IsActiveFlag_WU
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsActiveFlag_WU(void)
{
  return (READ_BIT(PWR->CSR, PWR_CSR_WUF) == (PWR_CSR_WUF));
}

/**
  * @brief  Get Standby Flag
  * @rmtoll CSR   SBF       LL_PWR_IsActiveFlag_SB
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsActiveFlag_SB(void)
{
  return (READ_BIT(PWR->CSR, PWR_CSR_SBF) == (PWR_CSR_SBF));
}

/**
  * @brief  Get Backup Regulator ready Flag
  * @rmtoll CSR   BRR       LL_PWR_IsActiveFlag_BRR
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsActiveFlag_BRR(void)
{
  return (READ_BIT(PWR->CSR, PWR_CSR_BRR) == (PWR_CSR_BRR));
}
/**
  * @brief  Indicate whether VDD voltage is below the selected PVD threshold
  * @rmtoll CSR   PVDO       LL_PWR_IsActiveFlag_PVDO
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsActiveFlag_PVDO(void)
{
  return (READ_BIT(PWR->CSR, PWR_CSR_PVDO) == (PWR_CSR_PVDO));
}

/**
  * @brief  Indicate whether the Regulator is ready in the selected voltage range or if its output voltage is still changing to the required voltage level
  * @rmtoll CSR   VOS       LL_PWR_IsActiveFlag_VOS
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsActiveFlag_VOS(void)
{
  return (READ_BIT(PWR->CSR, LL_PWR_CSR_VOS) == (LL_PWR_CSR_VOS));
}
#if defined(PWR_CR_ODEN)
/**
  * @brief  Indicate whether the Over-Drive mode is ready or not
  * @rmtoll CSR   ODRDY       LL_PWR_IsActiveFlag_OD
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsActiveFlag_OD(void)
{
  return (READ_BIT(PWR->CSR, PWR_CSR_ODRDY) == (PWR_CSR_ODRDY));
}
#endif /* PWR_CR_ODEN */

#if defined(PWR_CR_ODSWEN)
/**
  * @brief  Indicate whether the Over-Drive mode switching is ready or not
  * @rmtoll CSR   ODSWRDY       LL_PWR_IsActiveFlag_ODSW
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsActiveFlag_ODSW(void)
{
  return (READ_BIT(PWR->CSR, PWR_CSR_ODSWRDY) == (PWR_CSR_ODSWRDY));
}
#endif /* PWR_CR_ODSWEN */

#if defined(PWR_CR_UDEN)
/**
  * @brief  Indicate whether the Under-Drive mode is ready or not
  * @rmtoll CSR   UDRDY       LL_PWR_IsActiveFlag_UD
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_PWR_IsActiveFlag_UD(void)
{
  return (READ_BIT(PWR->CSR, PWR_CSR_UDRDY) == (PWR_CSR_UDRDY));
}
#endif /* PWR_CR_UDEN */
/**
  * @brief  Clear Standby Flag
  * @rmtoll CR   CSBF       LL_PWR_ClearFlag_SB
  * @retval None
  */
__STATIC_INLINE void LL_PWR_ClearFlag_SB(void)
{
  SET_BIT(PWR->CR, PWR_CR_CSBF);
}

/**
  * @brief  Clear Wake-up Flags
  * @rmtoll CR   CWUF       LL_PWR_ClearFlag_WU
  * @retval None
  */
__STATIC_INLINE void LL_PWR_ClearFlag_WU(void)
{
  SET_BIT(PWR->CR, PWR_CR_CWUF);
}
#if defined(PWR_CSR_UDRDY)
/**
  * @brief  Clear Under-Drive ready Flag
  * @rmtoll CSR          UDRDY         LL_PWR_ClearFlag_UD
  * @retval None
  */
__STATIC_INLINE void LL_PWR_ClearFlag_UD(void)
{
  WRITE_REG(PWR->CSR, PWR_CSR_UDRDY);
}
#endif /* PWR_CSR_UDRDY */

/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup PWR_LL_EF_Init De-initialization function
  * @{
  */
ErrorStatus LL_PWR_DeInit(void);
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

#endif /* defined(PWR) */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_LL_PWR_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
