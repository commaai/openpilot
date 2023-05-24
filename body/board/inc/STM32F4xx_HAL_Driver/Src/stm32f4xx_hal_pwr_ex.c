/**
  ******************************************************************************
  * @file    stm32f4xx_hal_pwr_ex.c
  * @author  MCD Application Team
  * @brief   Extended PWR HAL module driver.
  *          This file provides firmware functions to manage the following 
  *          functionalities of PWR extension peripheral:           
  *           + Peripheral Extended features functions
  *         
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

/** @defgroup PWREx PWREx
  * @brief PWR HAL module driver
  * @{
  */

#ifdef HAL_PWR_MODULE_ENABLED

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @addtogroup PWREx_Private_Constants
  * @{
  */    
#define PWR_OVERDRIVE_TIMEOUT_VALUE  1000U
#define PWR_UDERDRIVE_TIMEOUT_VALUE  1000U
#define PWR_BKPREG_TIMEOUT_VALUE     1000U
#define PWR_VOSRDY_TIMEOUT_VALUE     1000U
/**
  * @}
  */

   
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/** @defgroup PWREx_Exported_Functions PWREx Exported Functions
  *  @{
  */

/** @defgroup PWREx_Exported_Functions_Group1 Peripheral Extended features functions 
  *  @brief Peripheral Extended features functions 
  *
@verbatim   

 ===============================================================================
                 ##### Peripheral extended features functions #####
 ===============================================================================

    *** Main and Backup Regulators configuration ***
    ================================================
    [..] 
      (+) The backup domain includes 4 Kbytes of backup SRAM accessible only from 
          the CPU, and address in 32-bit, 16-bit or 8-bit mode. Its content is 
          retained even in Standby or VBAT mode when the low power backup regulator
          is enabled. It can be considered as an internal EEPROM when VBAT is 
          always present. You can use the HAL_PWREx_EnableBkUpReg() function to 
          enable the low power backup regulator. 

      (+) When the backup domain is supplied by VDD (analog switch connected to VDD) 
          the backup SRAM is powered from VDD which replaces the VBAT power supply to 
          save battery life.

      (+) The backup SRAM is not mass erased by a tamper event. It is read 
          protected to prevent confidential data, such as cryptographic private 
          key, from being accessed. The backup SRAM can be erased only through 
          the Flash interface when a protection level change from level 1 to 
          level 0 is requested. 
      -@- Refer to the description of Read protection (RDP) in the Flash 
          programming manual.

      (+) The main internal regulator can be configured to have a tradeoff between 
          performance and power consumption when the device does not operate at 
          the maximum frequency. This is done through __HAL_PWR_MAINREGULATORMODE_CONFIG() 
          macro which configure VOS bit in PWR_CR register
          
        Refer to the product datasheets for more details.

    *** FLASH Power Down configuration ****
    =======================================
    [..] 
      (+) By setting the FPDS bit in the PWR_CR register by using the 
          HAL_PWREx_EnableFlashPowerDown() function, the Flash memory also enters power 
          down mode when the device enters Stop mode. When the Flash memory 
          is in power down mode, an additional startup delay is incurred when 
          waking up from Stop mode.
          
           (+) For STM32F42xxx/43xxx/446xx/469xx/479xx Devices, the scale can be modified only when the PLL 
           is OFF and the HSI or HSE clock source is selected as system clock. 
           The new value programmed is active only when the PLL is ON.
           When the PLL is OFF, the voltage scale 3 is automatically selected. 
        Refer to the datasheets for more details.

    *** Over-Drive and Under-Drive configuration ****
    =================================================
    [..]         
       (+) For STM32F42xxx/43xxx/446xx/469xx/479xx Devices, in Run mode: the main regulator has
           2 operating modes available:
        (++) Normal mode: The CPU and core logic operate at maximum frequency at a given 
             voltage scaling (scale 1, scale 2 or scale 3)
        (++) Over-drive mode: This mode allows the CPU and the core logic to operate at a 
            higher frequency than the normal mode for a given voltage scaling (scale 1,  
            scale 2 or scale 3). This mode is enabled through HAL_PWREx_EnableOverDrive() function and
            disabled by HAL_PWREx_DisableOverDrive() function, to enter or exit from Over-drive mode please follow 
            the sequence described in Reference manual.
             
       (+) For STM32F42xxx/43xxx/446xx/469xx/479xx Devices, in Stop mode: the main regulator or low power regulator 
           supplies a low power voltage to the 1.2V domain, thus preserving the content of registers 
           and internal SRAM. 2 operating modes are available:
         (++) Normal mode: the 1.2V domain is preserved in nominal leakage mode. This mode is only 
              available when the main regulator or the low power regulator is used in Scale 3 or 
              low voltage mode.
         (++) Under-drive mode: the 1.2V domain is preserved in reduced leakage mode. This mode is only
              available when the main regulator or the low power regulator is in low voltage mode.

@endverbatim
  * @{
  */

/**
  * @brief Enables the Backup Regulator.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_PWREx_EnableBkUpReg(void)
{
  uint32_t tickstart = 0U;

  *(__IO uint32_t *) CSR_BRE_BB = (uint32_t)ENABLE;

  /* Get tick */
  tickstart = HAL_GetTick();

  /* Wait till Backup regulator ready flag is set */  
  while(__HAL_PWR_GET_FLAG(PWR_FLAG_BRR) == RESET)
  {
    if((HAL_GetTick() - tickstart ) > PWR_BKPREG_TIMEOUT_VALUE)
    {
      return HAL_TIMEOUT;
    } 
  }
  return HAL_OK;
}

/**
  * @brief Disables the Backup Regulator.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_PWREx_DisableBkUpReg(void)
{
  uint32_t tickstart = 0U;

  *(__IO uint32_t *) CSR_BRE_BB = (uint32_t)DISABLE;

  /* Get tick */
  tickstart = HAL_GetTick();

  /* Wait till Backup regulator ready flag is set */  
  while(__HAL_PWR_GET_FLAG(PWR_FLAG_BRR) != RESET)
  {
    if((HAL_GetTick() - tickstart ) > PWR_BKPREG_TIMEOUT_VALUE)
    {
      return HAL_TIMEOUT;
    } 
  }
  return HAL_OK;
}

/**
  * @brief Enables the Flash Power Down in Stop mode.
  * @retval None
  */
void HAL_PWREx_EnableFlashPowerDown(void)
{
  *(__IO uint32_t *) CR_FPDS_BB = (uint32_t)ENABLE;
}

/**
  * @brief Disables the Flash Power Down in Stop mode.
  * @retval None
  */
void HAL_PWREx_DisableFlashPowerDown(void)
{
  *(__IO uint32_t *) CR_FPDS_BB = (uint32_t)DISABLE;
}

/**
  * @brief Return Voltage Scaling Range.
  * @retval The configured scale for the regulator voltage(VOS bit field).
  *         The returned value can be one of the following:
  *            - @arg PWR_REGULATOR_VOLTAGE_SCALE1: Regulator voltage output Scale 1 mode
  *            - @arg PWR_REGULATOR_VOLTAGE_SCALE2: Regulator voltage output Scale 2 mode
  *            - @arg PWR_REGULATOR_VOLTAGE_SCALE3: Regulator voltage output Scale 3 mode
  */  
uint32_t HAL_PWREx_GetVoltageRange(void)
{
  return (PWR->CR & PWR_CR_VOS);
}

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx)
/**
  * @brief Configures the main internal regulator output voltage.
  * @param  VoltageScaling specifies the regulator output voltage to achieve
  *         a tradeoff between performance and power consumption.
  *          This parameter can be one of the following values:
  *            @arg PWR_REGULATOR_VOLTAGE_SCALE1: Regulator voltage output range 1 mode,
  *                                               the maximum value of fHCLK = 168 MHz.
  *            @arg PWR_REGULATOR_VOLTAGE_SCALE2: Regulator voltage output range 2 mode,
  *                                               the maximum value of fHCLK = 144 MHz.
  * @note  When moving from Range 1 to Range 2, the system frequency must be decreased to
  *        a value below 144 MHz before calling HAL_PWREx_ConfigVoltageScaling() API.
  *        When moving from Range 2 to Range 1, the system frequency can be increased to
  *        a value up to 168 MHz after calling HAL_PWREx_ConfigVoltageScaling() API.
  * @retval HAL Status
  */
HAL_StatusTypeDef HAL_PWREx_ControlVoltageScaling(uint32_t VoltageScaling)
{
  uint32_t tickstart = 0U;
  
  assert_param(IS_PWR_VOLTAGE_SCALING_RANGE(VoltageScaling));
  
  /* Enable PWR RCC Clock Peripheral */
  __HAL_RCC_PWR_CLK_ENABLE();
  
  /* Set Range */
  __HAL_PWR_VOLTAGESCALING_CONFIG(VoltageScaling);
  
  /* Get Start Tick*/
  tickstart = HAL_GetTick();
  while((__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY) == RESET))
  {
    if((HAL_GetTick() - tickstart ) > PWR_VOSRDY_TIMEOUT_VALUE)
    {
      return HAL_TIMEOUT;
    } 
  }

  return HAL_OK;
}

#elif defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
      defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) || \
      defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) || \
      defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || \
      defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/**
  * @brief Configures the main internal regulator output voltage.
  * @param  VoltageScaling specifies the regulator output voltage to achieve
  *         a tradeoff between performance and power consumption.
  *          This parameter can be one of the following values:
  *            @arg PWR_REGULATOR_VOLTAGE_SCALE1: Regulator voltage output range 1 mode,
  *                                               the maximum value of fHCLK is 168 MHz. It can be extended to
  *                                               180 MHz by activating the over-drive mode.
  *            @arg PWR_REGULATOR_VOLTAGE_SCALE2: Regulator voltage output range 2 mode,
  *                                               the maximum value of fHCLK is 144 MHz. It can be extended to,                
  *                                               168 MHz by activating the over-drive mode.
  *            @arg PWR_REGULATOR_VOLTAGE_SCALE3: Regulator voltage output range 3 mode,
  *                                               the maximum value of fHCLK is 120 MHz.
  * @note To update the system clock frequency(SYSCLK):
  *        - Set the HSI or HSE as system clock frequency using the HAL_RCC_ClockConfig().
  *        - Call the HAL_RCC_OscConfig() to configure the PLL.
  *        - Call HAL_PWREx_ConfigVoltageScaling() API to adjust the voltage scale.
  *        - Set the new system clock frequency using the HAL_RCC_ClockConfig().
  * @note The scale can be modified only when the HSI or HSE clock source is selected 
  *        as system clock source, otherwise the API returns HAL_ERROR.  
  * @note When the PLL is OFF, the voltage scale 3 is automatically selected and the VOS bits
  *       value in the PWR_CR1 register are not taken in account.
  * @note This API forces the PLL state ON to allow the possibility to configure the voltage scale 1 or 2.
  * @note The new voltage scale is active only when the PLL is ON.  
  * @retval HAL Status
  */
HAL_StatusTypeDef HAL_PWREx_ControlVoltageScaling(uint32_t VoltageScaling)
{
  uint32_t tickstart = 0U;
  
  assert_param(IS_PWR_VOLTAGE_SCALING_RANGE(VoltageScaling));
  
  /* Enable PWR RCC Clock Peripheral */
  __HAL_RCC_PWR_CLK_ENABLE();
  
  /* Check if the PLL is used as system clock or not */
  if(__HAL_RCC_GET_SYSCLK_SOURCE() != RCC_CFGR_SWS_PLL)
  {
    /* Disable the main PLL */
    __HAL_RCC_PLL_DISABLE();
    
    /* Get Start Tick */
    tickstart = HAL_GetTick();    
    /* Wait till PLL is disabled */  
    while(__HAL_RCC_GET_FLAG(RCC_FLAG_PLLRDY) != RESET)
    {
      if((HAL_GetTick() - tickstart ) > PLL_TIMEOUT_VALUE)
      {
        return HAL_TIMEOUT;
      }
    }
    
    /* Set Range */
    __HAL_PWR_VOLTAGESCALING_CONFIG(VoltageScaling);
    
    /* Enable the main PLL */
    __HAL_RCC_PLL_ENABLE();
    
    /* Get Start Tick */
    tickstart = HAL_GetTick();
    /* Wait till PLL is ready */  
    while(__HAL_RCC_GET_FLAG(RCC_FLAG_PLLRDY) == RESET)
    {
      if((HAL_GetTick() - tickstart ) > PLL_TIMEOUT_VALUE)
      {
        return HAL_TIMEOUT;
      } 
    }
    
    /* Get Start Tick */
    tickstart = HAL_GetTick();
    while((__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY) == RESET))
    {
      if((HAL_GetTick() - tickstart ) > PWR_VOSRDY_TIMEOUT_VALUE)
      {
        return HAL_TIMEOUT;
      } 
    }
  }
  else
  {
    return HAL_ERROR;
  }

  return HAL_OK;
}
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx */

#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) ||\
    defined(STM32F411xE) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) ||\
    defined(STM32F413xx) || defined(STM32F423xx)
/**
  * @brief Enables Main Regulator low voltage mode.
  * @note  This mode is only available for STM32F401xx/STM32F410xx/STM32F411xx/STM32F412Zx/STM32F412Rx/STM32F412Vx/STM32F412Cx/
  *        STM32F413xx/STM32F423xx devices.   
  * @retval None
  */
void HAL_PWREx_EnableMainRegulatorLowVoltage(void)
{
  *(__IO uint32_t *) CR_MRLVDS_BB = (uint32_t)ENABLE;
}

/**
  * @brief Disables Main Regulator low voltage mode.
  * @note  This mode is only available for STM32F401xx/STM32F410xx/STM32F411xx/STM32F412Zx/STM32F412Rx/STM32F412Vx/STM32F412Cx/
  *        STM32F413xx/STM32F423xxdevices. 
  * @retval None
  */
void HAL_PWREx_DisableMainRegulatorLowVoltage(void)
{
  *(__IO uint32_t *) CR_MRLVDS_BB = (uint32_t)DISABLE;
}

/**
  * @brief Enables Low Power Regulator low voltage mode.
  * @note  This mode is only available for STM32F401xx/STM32F410xx/STM32F411xx/STM32F412Zx/STM32F412Rx/STM32F412Vx/STM32F412Cx/
  *        STM32F413xx/STM32F423xx devices.   
  * @retval None
  */
void HAL_PWREx_EnableLowRegulatorLowVoltage(void)
{
  *(__IO uint32_t *) CR_LPLVDS_BB = (uint32_t)ENABLE;
}

/**
  * @brief Disables Low Power Regulator low voltage mode.
  * @note  This mode is only available for STM32F401xx/STM32F410xx/STM32F411xx/STM32F412Zx/STM32F412Rx/STM32F412Vx/STM32F412Cx/
  *        STM32F413xx/STM32F423xx  devices.   
  * @retval None
  */
void HAL_PWREx_DisableLowRegulatorLowVoltage(void)
{
  *(__IO uint32_t *) CR_LPLVDS_BB = (uint32_t)DISABLE;
}

#endif /* STM32F401xC || STM32F401xE || STM32F410xx || STM32F411xE || STM32F412Zx || STM32F412Rx || STM32F412Vx || STM32F412Cx ||
          STM32F413xx || STM32F423xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
/**
  * @brief  Activates the Over-Drive mode.
  * @note   This function can be used only for STM32F42xx/STM32F43xx/STM32F446xx/STM32F469xx/STM32F479xx devices.
  *         This mode allows the CPU and the core logic to operate at a higher frequency
  *         than the normal mode for a given voltage scaling (scale 1, scale 2 or scale 3).   
  * @note   It is recommended to enter or exit Over-drive mode when the application is not running 
  *         critical tasks and when the system clock source is either HSI or HSE. 
  *         During the Over-drive switch activation, no peripheral clocks should be enabled.   
  *         The peripheral clocks must be enabled once the Over-drive mode is activated.   
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_PWREx_EnableOverDrive(void)
{
  uint32_t tickstart = 0U;

  __HAL_RCC_PWR_CLK_ENABLE();
  
  /* Enable the Over-drive to extend the clock frequency to 180 Mhz */
  __HAL_PWR_OVERDRIVE_ENABLE();

  /* Get tick */
  tickstart = HAL_GetTick();

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_ODRDY))
  {
    if((HAL_GetTick() - tickstart) > PWR_OVERDRIVE_TIMEOUT_VALUE)
    {
      return HAL_TIMEOUT;
    }
  }
  
  /* Enable the Over-drive switch */
  __HAL_PWR_OVERDRIVESWITCHING_ENABLE();

  /* Get tick */
  tickstart = HAL_GetTick();

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_ODSWRDY))
  {
    if((HAL_GetTick() - tickstart ) > PWR_OVERDRIVE_TIMEOUT_VALUE)
    {
      return HAL_TIMEOUT;
    }
  } 
  return HAL_OK;
}

/**
  * @brief  Deactivates the Over-Drive mode.
  * @note   This function can be used only for STM32F42xx/STM32F43xx/STM32F446xx/STM32F469xx/STM32F479xx devices.
  *         This mode allows the CPU and the core logic to operate at a higher frequency
  *         than the normal mode for a given voltage scaling (scale 1, scale 2 or scale 3).    
  * @note   It is recommended to enter or exit Over-drive mode when the application is not running 
  *         critical tasks and when the system clock source is either HSI or HSE. 
  *         During the Over-drive switch activation, no peripheral clocks should be enabled.   
  *         The peripheral clocks must be enabled once the Over-drive mode is activated.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_PWREx_DisableOverDrive(void)
{
  uint32_t tickstart = 0U;
  
  __HAL_RCC_PWR_CLK_ENABLE();
    
  /* Disable the Over-drive switch */
  __HAL_PWR_OVERDRIVESWITCHING_DISABLE();
  
  /* Get tick */
  tickstart = HAL_GetTick();
 
  while(__HAL_PWR_GET_FLAG(PWR_FLAG_ODSWRDY))
  {
    if((HAL_GetTick() - tickstart) > PWR_OVERDRIVE_TIMEOUT_VALUE)
    {
      return HAL_TIMEOUT;
    }
  } 
  
  /* Disable the Over-drive */
  __HAL_PWR_OVERDRIVE_DISABLE();

  /* Get tick */
  tickstart = HAL_GetTick();

  while(__HAL_PWR_GET_FLAG(PWR_FLAG_ODRDY))
  {
    if((HAL_GetTick() - tickstart) > PWR_OVERDRIVE_TIMEOUT_VALUE)
    {
      return HAL_TIMEOUT;
    }
  }
  
  return HAL_OK;
}

/**
  * @brief  Enters in Under-Drive STOP mode.
  *  
  * @note   This mode is only available for STM32F42xxx/STM32F43xxx/STM32F446xx/STM32F469xx/STM32F479xx devices.
  * 
  * @note    This mode can be selected only when the Under-Drive is already active 
  *   
  * @note    This mode is enabled only with STOP low power mode.
  *          In this mode, the 1.2V domain is preserved in reduced leakage mode. This 
  *          mode is only available when the main regulator or the low power regulator 
  *          is in low voltage mode
  *        
  * @note   If the Under-drive mode was enabled, it is automatically disabled after 
  *         exiting Stop mode. 
  *         When the voltage regulator operates in Under-drive mode, an additional  
  *         startup delay is induced when waking up from Stop mode.
  *                    
  * @note   In Stop mode, all I/O pins keep the same state as in Run mode.
  *   
  * @note   When exiting Stop mode by issuing an interrupt or a wake-up event, 
  *         the HSI RC oscillator is selected as system clock.
  *           
  * @note   When the voltage regulator operates in low power mode, an additional 
  *         startup delay is incurred when waking up from Stop mode. 
  *         By keeping the internal regulator ON during Stop mode, the consumption 
  *         is higher although the startup time is reduced.
  *     
  * @param  Regulator specifies the regulator state in STOP mode.
  *          This parameter can be one of the following values:
  *            @arg PWR_MAINREGULATOR_UNDERDRIVE_ON:  Main Regulator in under-drive mode 
  *                 and Flash memory in power-down when the device is in Stop under-drive mode
  *            @arg PWR_LOWPOWERREGULATOR_UNDERDRIVE_ON:  Low Power Regulator in under-drive mode 
  *                and Flash memory in power-down when the device is in Stop under-drive mode
  * @param  STOPEntry specifies if STOP mode in entered with WFI or WFE instruction.
  *          This parameter can be one of the following values:
  *            @arg PWR_SLEEPENTRY_WFI: enter STOP mode with WFI instruction
  *            @arg PWR_SLEEPENTRY_WFE: enter STOP mode with WFE instruction
  * @retval None
  */
HAL_StatusTypeDef HAL_PWREx_EnterUnderDriveSTOPMode(uint32_t Regulator, uint8_t STOPEntry)
{
  uint32_t tmpreg1 = 0U;

  /* Check the parameters */
  assert_param(IS_PWR_REGULATOR_UNDERDRIVE(Regulator));
  assert_param(IS_PWR_STOP_ENTRY(STOPEntry));
  
  /* Enable Power ctrl clock */
  __HAL_RCC_PWR_CLK_ENABLE();
  /* Enable the Under-drive Mode ---------------------------------------------*/
  /* Clear Under-drive flag */
  __HAL_PWR_CLEAR_ODRUDR_FLAG();
  
  /* Enable the Under-drive */ 
  __HAL_PWR_UNDERDRIVE_ENABLE();

  /* Select the regulator state in STOP mode ---------------------------------*/
  tmpreg1 = PWR->CR;
  /* Clear PDDS, LPDS, MRLUDS and LPLUDS bits */
  tmpreg1 &= (uint32_t)~(PWR_CR_PDDS | PWR_CR_LPDS | PWR_CR_LPUDS | PWR_CR_MRUDS);
  
  /* Set LPDS, MRLUDS and LPLUDS bits according to PWR_Regulator value */
  tmpreg1 |= Regulator;
  
  /* Store the new value */
  PWR->CR = tmpreg1;
  
  /* Set SLEEPDEEP bit of Cortex System Control Register */
  SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
  
  /* Select STOP mode entry --------------------------------------------------*/
  if(STOPEntry == PWR_SLEEPENTRY_WFI)
  {   
    /* Request Wait For Interrupt */
    __WFI();
  }
  else
  {
    /* Request Wait For Event */
    __WFE();
  }
  /* Reset SLEEPDEEP bit of Cortex System Control Register */
  SCB->SCR &= (uint32_t)~((uint32_t)SCB_SCR_SLEEPDEEP_Msk);

  return HAL_OK;  
}

#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */
/**
  * @}
  */

/**
  * @}
  */

#endif /* HAL_PWR_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
