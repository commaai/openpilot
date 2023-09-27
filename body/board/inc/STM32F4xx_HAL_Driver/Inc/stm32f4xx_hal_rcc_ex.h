/**
  ******************************************************************************
  * @file    stm32f4xx_hal_rcc_ex.h
  * @author  MCD Application Team
  * @brief   Header file of RCC HAL Extension module.
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
#ifndef __STM32F4xx_HAL_RCC_EX_H
#define __STM32F4xx_HAL_RCC_EX_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup RCCEx
  * @{
  */ 

/* Exported types ------------------------------------------------------------*/
/** @defgroup RCCEx_Exported_Types RCCEx Exported Types
  * @{
  */

/**
  * @brief  RCC PLL configuration structure definition
  */
typedef struct
{
  uint32_t PLLState;   /*!< The new state of the PLL.
                            This parameter can be a value of @ref RCC_PLL_Config                      */

  uint32_t PLLSource;  /*!< RCC_PLLSource: PLL entry clock source.
                            This parameter must be a value of @ref RCC_PLL_Clock_Source               */

  uint32_t PLLM;       /*!< PLLM: Division factor for PLL VCO input clock.
                            This parameter must be a number between Min_Data = 0 and Max_Data = 63    */

  uint32_t PLLN;       /*!< PLLN: Multiplication factor for PLL VCO output clock.
                            This parameter must be a number between Min_Data = 50 and Max_Data = 432 
                            except for STM32F411xE devices where the Min_Data = 192 */

  uint32_t PLLP;       /*!< PLLP: Division factor for main system clock (SYSCLK).
                            This parameter must be a value of @ref RCC_PLLP_Clock_Divider             */

  uint32_t PLLQ;       /*!< PLLQ: Division factor for OTG FS, SDIO and RNG clocks.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 15    */
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) ||\
    defined(STM32F413xx) || defined(STM32F423xx)
  uint32_t PLLR;       /*!< PLLR: PLL division factor for I2S, SAI, SYSTEM, SPDIFRX clocks.
                            This parameter is only available in STM32F410xx/STM32F446xx/STM32F469xx/STM32F479xx
                            and STM32F412Zx/STM32F412Vx/STM32F412Rx/STM32F412Cx/STM32F413xx/STM32F423xx devices. 
                            This parameter must be a number between Min_Data = 2 and Max_Data = 7     */
#endif /* STM32F410xx || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */ 
}RCC_PLLInitTypeDef;

#if defined(STM32F446xx)
/** 
  * @brief  PLLI2S Clock structure definition  
  */
typedef struct
{
  uint32_t PLLI2SM;    /*!< Specifies division factor for PLL VCO input clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 63       */

  uint32_t PLLI2SN;    /*!< Specifies the multiplication factor for PLLI2S VCO output clock.
                            This parameter must be a number between Min_Data = 50 and Max_Data = 432    */

  uint32_t PLLI2SP;    /*!< Specifies division factor for SPDIFRX Clock.
                            This parameter must be a value of @ref RCCEx_PLLI2SP_Clock_Divider           */

  uint32_t PLLI2SQ;    /*!< Specifies the division factor for SAI clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 15. 
                            This parameter will be used only when PLLI2S is selected as Clock Source SAI */
                           
  uint32_t PLLI2SR;    /*!< Specifies the division factor for I2S clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 7. 
                            This parameter will be used only when PLLI2S is selected as Clock Source I2S */
}RCC_PLLI2SInitTypeDef;

/** 
  * @brief  PLLSAI Clock structure definition  
  */
typedef struct
{
  uint32_t PLLSAIM;    /*!< Specifies division factor for PLL VCO input clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 63       */

  uint32_t PLLSAIN;    /*!< Specifies the multiplication factor for PLLI2S VCO output clock.
                            This parameter must be a number between Min_Data = 50 and Max_Data = 432    */

  uint32_t PLLSAIP;    /*!< Specifies division factor for OTG FS, SDIO and RNG clocks.
                            This parameter must be a value of @ref RCCEx_PLLSAIP_Clock_Divider           */
                                                             
  uint32_t PLLSAIQ;    /*!< Specifies the division factor for SAI clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 15.
                            This parameter will be used only when PLLSAI is selected as Clock Source SAI */
}RCC_PLLSAIInitTypeDef;

/** 
  * @brief  RCC extended clocks structure definition  
  */
typedef struct
{
  uint32_t PeriphClockSelection; /*!< The Extended Clock to be configured.
                                      This parameter can be a value of @ref RCCEx_Periph_Clock_Selection */

  RCC_PLLI2SInitTypeDef PLLI2S;  /*!< PLL I2S structure parameters. 
                                      This parameter will be used only when PLLI2S is selected as Clock Source I2S or SAI */

  RCC_PLLSAIInitTypeDef PLLSAI;  /*!< PLL SAI structure parameters. 
                                      This parameter will be used only when PLLI2S is selected as Clock Source SAI or LTDC */

  uint32_t PLLI2SDivQ;           /*!< Specifies the PLLI2S division factor for SAI1 clock.
                                      This parameter must be a number between Min_Data = 1 and Max_Data = 32
                                      This parameter will be used only when PLLI2S is selected as Clock Source SAI */

  uint32_t PLLSAIDivQ;           /*!< Specifies the PLLI2S division factor for SAI1 clock.
                                      This parameter must be a number between Min_Data = 1 and Max_Data = 32
                                      This parameter will be used only when PLLSAI is selected as Clock Source SAI */

  uint32_t Sai1ClockSelection;    /*!< Specifies SAI1 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_SAI1_Clock_Source */

  uint32_t Sai2ClockSelection;    /*!< Specifies SAI2 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_SAI2_Clock_Source */
                                      
  uint32_t I2sApb1ClockSelection;    /*!< Specifies I2S APB1 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_I2SAPB1_Clock_Source */

  uint32_t I2sApb2ClockSelection;    /*!< Specifies I2S APB2 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_I2SAPB2_Clock_Source */

  uint32_t RTCClockSelection;      /*!< Specifies RTC Clock Source Selection. 
                                      This parameter can be a value of @ref RCC_RTC_Clock_Source */

  uint32_t SdioClockSelection;    /*!< Specifies SDIO Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_SDIO_Clock_Source */

  uint32_t CecClockSelection;      /*!< Specifies CEC Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_CEC_Clock_Source */

  uint32_t Fmpi2c1ClockSelection;  /*!< Specifies FMPI2C1 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_FMPI2C1_Clock_Source */

  uint32_t SpdifClockSelection;    /*!< Specifies SPDIFRX Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_SPDIFRX_Clock_Source */

  uint32_t Clk48ClockSelection;     /*!< Specifies CLK48 Clock Selection this clock used OTG FS, SDIO and RNG clocks. 
                                      This parameter can be a value of @ref RCCEx_CLK48_Clock_Source */
  
  uint8_t TIMPresSelection;      /*!< Specifies TIM Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_TIM_PRescaler_Selection */
}RCC_PeriphCLKInitTypeDef;
#endif /* STM32F446xx */   

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
/** 
  * @brief  RCC extended clocks structure definition
  */
typedef struct
{
  uint32_t PeriphClockSelection;   /*!< The Extended Clock to be configured.
                                      This parameter can be a value of @ref RCCEx_Periph_Clock_Selection */

  uint32_t I2SClockSelection;      /*!< Specifies RTC Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_I2S_APB_Clock_Source */
                                      
  uint32_t RTCClockSelection;      /*!< Specifies RTC Clock Source Selection. 
                                      This parameter can be a value of @ref RCC_RTC_Clock_Source */

  uint32_t Lptim1ClockSelection;   /*!< Specifies LPTIM1 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_LPTIM1_Clock_Source */
  
  uint32_t Fmpi2c1ClockSelection;  /*!< Specifies FMPI2C1 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_FMPI2C1_Clock_Source */

  uint8_t TIMPresSelection;        /*!< Specifies TIM Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_TIM_PRescaler_Selection */
}RCC_PeriphCLKInitTypeDef;
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/** 
  * @brief  PLLI2S Clock structure definition  
  */
typedef struct
{
  uint32_t PLLI2SM;    /*!< Specifies division factor for PLL VCO input clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 63       */

  uint32_t PLLI2SN;    /*!< Specifies the multiplication factor for PLLI2S VCO output clock.
                            This parameter must be a number between Min_Data = 50 and Max_Data = 432    */

  uint32_t PLLI2SQ;    /*!< Specifies the division factor for SAI clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 15. 
                            This parameter will be used only when PLLI2S is selected as Clock Source SAI */
                           
  uint32_t PLLI2SR;    /*!< Specifies the division factor for I2S clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 7. 
                            This parameter will be used only when PLLI2S is selected as Clock Source I2S */
}RCC_PLLI2SInitTypeDef;

/** 
  * @brief  RCC extended clocks structure definition
  */
typedef struct
{
  uint32_t PeriphClockSelection; /*!< The Extended Clock to be configured.
                                      This parameter can be a value of @ref RCCEx_Periph_Clock_Selection */

  RCC_PLLI2SInitTypeDef PLLI2S;  /*!< PLL I2S structure parameters. 
                                      This parameter will be used only when PLLI2S is selected as Clock Source I2S */
  
#if defined(STM32F413xx) || defined(STM32F423xx)
  uint32_t PLLDivR;              /*!< Specifies the PLL division factor for SAI1 clock.
                                      This parameter must be a number between Min_Data = 1 and Max_Data = 32
                                      This parameter will be used only when PLL is selected as Clock Source SAI */

  uint32_t PLLI2SDivR;           /*!< Specifies the PLLI2S division factor for SAI1 clock.
                                      This parameter must be a number between Min_Data = 1 and Max_Data = 32
                                      This parameter will be used only when PLLI2S is selected as Clock Source SAI */
#endif /* STM32F413xx || STM32F423xx */  
                                      
  uint32_t I2sApb1ClockSelection;    /*!< Specifies I2S APB1 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_I2SAPB1_Clock_Source */

  uint32_t I2sApb2ClockSelection;    /*!< Specifies I2S APB2 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_I2SAPB2_Clock_Source */

  uint32_t RTCClockSelection;      /*!< Specifies RTC Clock Source Selection. 
                                      This parameter can be a value of @ref RCC_RTC_Clock_Source */

  uint32_t SdioClockSelection;    /*!< Specifies SDIO Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_SDIO_Clock_Source */

  uint32_t Fmpi2c1ClockSelection;  /*!< Specifies FMPI2C1 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_FMPI2C1_Clock_Source */

  uint32_t Clk48ClockSelection;     /*!< Specifies CLK48 Clock Selection this clock used OTG FS, SDIO and RNG clocks.
                                      This parameter can be a value of @ref RCCEx_CLK48_Clock_Source */
  
  uint32_t Dfsdm1ClockSelection;    /*!< Specifies DFSDM1 Clock Selection.
                                      This parameter can be a value of @ref RCCEx_DFSDM1_Kernel_Clock_Source */

  uint32_t Dfsdm1AudioClockSelection;/*!< Specifies DFSDM1 Audio Clock Selection.
                                      This parameter can be a value of @ref RCCEx_DFSDM1_Audio_Clock_Source */
  
#if defined(STM32F413xx) || defined(STM32F423xx)
  uint32_t Dfsdm2ClockSelection;    /*!< Specifies DFSDM2 Clock Selection.
                                      This parameter can be a value of @ref RCCEx_DFSDM2_Kernel_Clock_Source */

  uint32_t Dfsdm2AudioClockSelection;/*!< Specifies DFSDM2 Audio Clock Selection.
                                      This parameter can be a value of @ref RCCEx_DFSDM2_Audio_Clock_Source */
  
  uint32_t Lptim1ClockSelection;   /*!< Specifies LPTIM1 Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_LPTIM1_Clock_Source */
  
  uint32_t SaiAClockSelection;     /*!< Specifies SAI1_A Clock Prescalers Selection
                                        This parameter can be a value of @ref RCCEx_SAI1_BlockA_Clock_Source */

  uint32_t SaiBClockSelection;     /*!< Specifies SAI1_B Clock Prescalers Selection
                                        This parameter can be a value of @ref RCCEx_SAI1_BlockB_Clock_Source */
#endif /* STM32F413xx || STM32F423xx */

  uint32_t PLLI2SSelection;      /*!< Specifies PLL I2S Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_PLL_I2S_Clock_Source */

  uint8_t TIMPresSelection;      /*!< Specifies TIM Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_TIM_PRescaler_Selection */
}RCC_PeriphCLKInitTypeDef;
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)

/** 
  * @brief  PLLI2S Clock structure definition  
  */
typedef struct
{
  uint32_t PLLI2SN;    /*!< Specifies the multiplication factor for PLLI2S VCO output clock.
                            This parameter must be a number between Min_Data = 50 and Max_Data = 432.
                            This parameter will be used only when PLLI2S is selected as Clock Source I2S or SAI */

  uint32_t PLLI2SR;    /*!< Specifies the division factor for I2S clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 7. 
                            This parameter will be used only when PLLI2S is selected as Clock Source I2S or SAI */

  uint32_t PLLI2SQ;    /*!< Specifies the division factor for SAI1 clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 15. 
                            This parameter will be used only when PLLI2S is selected as Clock Source SAI */
}RCC_PLLI2SInitTypeDef;

/** 
  * @brief  PLLSAI Clock structure definition  
  */
typedef struct
{
  uint32_t PLLSAIN;    /*!< Specifies the multiplication factor for PLLI2S VCO output clock.
                            This parameter must be a number between Min_Data = 50 and Max_Data = 432.
                            This parameter will be used only when PLLSAI is selected as Clock Source SAI or LTDC */ 
#if defined(STM32F469xx) || defined(STM32F479xx)
  uint32_t PLLSAIP;    /*!< Specifies division factor for OTG FS and SDIO clocks.
                            This parameter is only available in STM32F469xx/STM32F479xx devices.
                            This parameter must be a value of @ref RCCEx_PLLSAIP_Clock_Divider  */  
#endif /* STM32F469xx || STM32F479xx */
                                 
  uint32_t PLLSAIQ;    /*!< Specifies the division factor for SAI1 clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 15.
                            This parameter will be used only when PLLSAI is selected as Clock Source SAI or LTDC */
                              
  uint32_t PLLSAIR;    /*!< specifies the division factor for LTDC clock
                            This parameter must be a number between Min_Data = 2 and Max_Data = 7.
                            This parameter will be used only when PLLSAI is selected as Clock Source LTDC */

}RCC_PLLSAIInitTypeDef;

/** 
  * @brief  RCC extended clocks structure definition  
  */
typedef struct
{
  uint32_t PeriphClockSelection; /*!< The Extended Clock to be configured.
                                      This parameter can be a value of @ref RCCEx_Periph_Clock_Selection */

  RCC_PLLI2SInitTypeDef PLLI2S;  /*!< PLL I2S structure parameters. 
                                      This parameter will be used only when PLLI2S is selected as Clock Source I2S or SAI */

  RCC_PLLSAIInitTypeDef PLLSAI;  /*!< PLL SAI structure parameters. 
                                      This parameter will be used only when PLLI2S is selected as Clock Source SAI or LTDC */

  uint32_t PLLI2SDivQ;           /*!< Specifies the PLLI2S division factor for SAI1 clock.
                                      This parameter must be a number between Min_Data = 1 and Max_Data = 32
                                      This parameter will be used only when PLLI2S is selected as Clock Source SAI */

  uint32_t PLLSAIDivQ;           /*!< Specifies the PLLI2S division factor for SAI1 clock.
                                      This parameter must be a number between Min_Data = 1 and Max_Data = 32
                                      This parameter will be used only when PLLSAI is selected as Clock Source SAI */

  uint32_t PLLSAIDivR;           /*!< Specifies the PLLSAI division factor for LTDC clock.
                                      This parameter must be one value of @ref RCCEx_PLLSAI_DIVR */

  uint32_t RTCClockSelection;      /*!< Specifies RTC Clock Prescalers Selection. 
                                      This parameter can be a value of @ref RCC_RTC_Clock_Source */

  uint8_t TIMPresSelection;      /*!< Specifies TIM Clock Prescalers Selection. 
                                      This parameter can be a value of @ref RCCEx_TIM_PRescaler_Selection */
#if defined(STM32F469xx) || defined(STM32F479xx)
  uint32_t Clk48ClockSelection;  /*!< Specifies CLK48 Clock Selection this clock used OTG FS, SDIO and RNG clocks. 
                                      This parameter can be a value of @ref RCCEx_CLK48_Clock_Source */

  uint32_t SdioClockSelection;   /*!< Specifies SDIO Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_SDIO_Clock_Source */  
#endif /* STM32F469xx || STM32F479xx */  
}RCC_PeriphCLKInitTypeDef;

#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE)
/** 
  * @brief  PLLI2S Clock structure definition  
  */
typedef struct
{
#if defined(STM32F411xE)
  uint32_t PLLI2SM;    /*!< PLLM: Division factor for PLLI2S VCO input clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 62  */
#endif /* STM32F411xE */
                                
  uint32_t PLLI2SN;    /*!< Specifies the multiplication factor for PLLI2S VCO output clock.
                            This parameter must be a number between Min_Data = 50 and Max_Data = 432
                            Except for STM32F411xE devices where the Min_Data = 192. 
                            This parameter will be used only when PLLI2S is selected as Clock Source I2S or SAI */

  uint32_t PLLI2SR;    /*!< Specifies the division factor for I2S clock.
                            This parameter must be a number between Min_Data = 2 and Max_Data = 7. 
                            This parameter will be used only when PLLI2S is selected as Clock Source I2S or SAI */

}RCC_PLLI2SInitTypeDef;
 
/** 
  * @brief  RCC extended clocks structure definition  
  */
typedef struct
{
  uint32_t PeriphClockSelection; /*!< The Extended Clock to be configured.
                                      This parameter can be a value of @ref RCCEx_Periph_Clock_Selection */

  RCC_PLLI2SInitTypeDef PLLI2S;  /*!< PLL I2S structure parameters.
                                      This parameter will be used only when PLLI2S is selected as Clock Source I2S or SAI */

  uint32_t RTCClockSelection;      /*!< Specifies RTC Clock Prescalers Selection.
                                       This parameter can be a value of @ref RCC_RTC_Clock_Source */
#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) 
  uint8_t TIMPresSelection;        /*!< Specifies TIM Clock Source Selection. 
                                      This parameter can be a value of @ref RCCEx_TIM_PRescaler_Selection */
#endif /* STM32F401xC || STM32F401xE || STM32F411xE */
}RCC_PeriphCLKInitTypeDef;
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F401xC || STM32F401xE || STM32F411xE */
/**
  * @}
  */ 

/* Exported constants --------------------------------------------------------*/
/** @defgroup RCCEx_Exported_Constants RCCEx Exported Constants
  * @{
  */

/** @defgroup RCCEx_Periph_Clock_Selection RCC Periph Clock Selection
  * @{
  */
/* Peripheral Clock source for STM32F412Zx/STM32F412Vx/STM32F412Rx/STM32F412Cx */
#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) ||\
    defined(STM32F413xx) || defined(STM32F423xx)
#define RCC_PERIPHCLK_I2S_APB1        0x00000001U
#define RCC_PERIPHCLK_I2S_APB2        0x00000002U
#define RCC_PERIPHCLK_TIM             0x00000004U
#define RCC_PERIPHCLK_RTC             0x00000008U
#define RCC_PERIPHCLK_FMPI2C1         0x00000010U
#define RCC_PERIPHCLK_CLK48           0x00000020U
#define RCC_PERIPHCLK_SDIO            0x00000040U
#define RCC_PERIPHCLK_PLLI2S          0x00000080U
#define RCC_PERIPHCLK_DFSDM1          0x00000100U
#define RCC_PERIPHCLK_DFSDM1_AUDIO    0x00000200U
#endif /* STM32F412Zx || STM32F412Vx) || STM32F412Rx || STM32F412Cx */
#if defined(STM32F413xx) || defined(STM32F423xx)
#define RCC_PERIPHCLK_DFSDM2          0x00000400U
#define RCC_PERIPHCLK_DFSDM2_AUDIO    0x00000800U
#define RCC_PERIPHCLK_LPTIM1          0x00001000U
#define RCC_PERIPHCLK_SAIA            0x00002000U
#define RCC_PERIPHCLK_SAIB            0x00004000U
#endif /* STM32F413xx || STM32F423xx */
/*----------------------------------------------------------------------------*/

/*------------------- Peripheral Clock source for STM32F410xx ----------------*/
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define RCC_PERIPHCLK_I2S             0x00000001U
#define RCC_PERIPHCLK_TIM             0x00000002U
#define RCC_PERIPHCLK_RTC             0x00000004U
#define RCC_PERIPHCLK_FMPI2C1         0x00000008U
#define RCC_PERIPHCLK_LPTIM1          0x00000010U
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */
/*----------------------------------------------------------------------------*/

/*------------------- Peripheral Clock source for STM32F446xx ----------------*/
#if defined(STM32F446xx)
#define RCC_PERIPHCLK_I2S_APB1        0x00000001U
#define RCC_PERIPHCLK_I2S_APB2        0x00000002U
#define RCC_PERIPHCLK_SAI1            0x00000004U
#define RCC_PERIPHCLK_SAI2            0x00000008U
#define RCC_PERIPHCLK_TIM             0x00000010U
#define RCC_PERIPHCLK_RTC             0x00000020U
#define RCC_PERIPHCLK_CEC             0x00000040U
#define RCC_PERIPHCLK_FMPI2C1         0x00000080U
#define RCC_PERIPHCLK_CLK48           0x00000100U
#define RCC_PERIPHCLK_SDIO            0x00000200U
#define RCC_PERIPHCLK_SPDIFRX         0x00000400U
#define RCC_PERIPHCLK_PLLI2S          0x00000800U
#endif /* STM32F446xx */
/*-----------------------------------------------------------------------------*/
    
/*----------- Peripheral Clock source for STM32F469xx/STM32F479xx -------------*/
#if defined(STM32F469xx) || defined(STM32F479xx)
#define RCC_PERIPHCLK_I2S             0x00000001U
#define RCC_PERIPHCLK_SAI_PLLI2S      0x00000002U
#define RCC_PERIPHCLK_SAI_PLLSAI      0x00000004U
#define RCC_PERIPHCLK_LTDC            0x00000008U
#define RCC_PERIPHCLK_TIM             0x00000010U
#define RCC_PERIPHCLK_RTC             0x00000020U
#define RCC_PERIPHCLK_PLLI2S          0x00000040U
#define RCC_PERIPHCLK_CLK48           0x00000080U
#define RCC_PERIPHCLK_SDIO            0x00000100U
#endif /* STM32F469xx || STM32F479xx */
/*----------------------------------------------------------------------------*/

/*-------- Peripheral Clock source for STM32F42xxx/STM32F43xxx ---------------*/
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx)
#define RCC_PERIPHCLK_I2S             0x00000001U
#define RCC_PERIPHCLK_SAI_PLLI2S      0x00000002U
#define RCC_PERIPHCLK_SAI_PLLSAI      0x00000004U
#define RCC_PERIPHCLK_LTDC            0x00000008U
#define RCC_PERIPHCLK_TIM             0x00000010U
#define RCC_PERIPHCLK_RTC             0x00000020U
#define RCC_PERIPHCLK_PLLI2S          0x00000040U
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx */
/*----------------------------------------------------------------------------*/

/*-------- Peripheral Clock source for STM32F40xxx/STM32F41xxx ---------------*/
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx)|| defined(STM32F417xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) 
#define RCC_PERIPHCLK_I2S             0x00000001U
#define RCC_PERIPHCLK_RTC             0x00000002U
#define RCC_PERIPHCLK_PLLI2S          0x00000004U
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F401xC || STM32F401xE || STM32F411xE */
#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE)
#define RCC_PERIPHCLK_TIM             0x00000008U
#endif /* STM32F401xC || STM32F401xE || STM32F411xE */      
/*----------------------------------------------------------------------------*/
/**
  * @}
  */
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || \
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F469xx) || \
    defined(STM32F479xx) 
/** @defgroup RCCEx_I2S_Clock_Source I2S Clock Source
  * @{
  */
#define RCC_I2SCLKSOURCE_PLLI2S         0x00000000U
#define RCC_I2SCLKSOURCE_EXT            0x00000001U
/**
  * @}
  */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx ||
          STM32F401xC || STM32F401xE || STM32F411xE || STM32F469xx || STM32F479xx */

/** @defgroup RCCEx_PLLSAI_DIVR RCC PLLSAI DIVR
  * @{
  */
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F446xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx) 
#define RCC_PLLSAIDIVR_2                0x00000000U
#define RCC_PLLSAIDIVR_4                0x00010000U
#define RCC_PLLSAIDIVR_8                0x00020000U
#define RCC_PLLSAIDIVR_16               0x00030000U
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup RCCEx_PLLI2SP_Clock_Divider RCC PLLI2SP Clock Divider
  * @{
  */
#if defined(STM32F446xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || \
    defined(STM32F412Rx) || defined(STM32F412Cx)
#define RCC_PLLI2SP_DIV2                  0x00000002U
#define RCC_PLLI2SP_DIV4                  0x00000004U
#define RCC_PLLI2SP_DIV6                  0x00000006U
#define RCC_PLLI2SP_DIV8                  0x00000008U
#endif /* STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */
/**
  * @}
  */

/** @defgroup RCCEx_PLLSAIP_Clock_Divider RCC PLLSAIP Clock Divider
  * @{
  */
#if defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx) 
#define RCC_PLLSAIP_DIV2                  0x00000002U
#define RCC_PLLSAIP_DIV4                  0x00000004U
#define RCC_PLLSAIP_DIV6                  0x00000006U
#define RCC_PLLSAIP_DIV8                  0x00000008U
#endif /* STM32F446xx || STM32F469xx || STM32F479xx */
/**
  * @}
  */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
/** @defgroup RCCEx_SAI_BlockA_Clock_Source  RCC SAI BlockA Clock Source
  * @{
  */
#define RCC_SAIACLKSOURCE_PLLSAI             0x00000000U
#define RCC_SAIACLKSOURCE_PLLI2S             0x00100000U
#define RCC_SAIACLKSOURCE_EXT                0x00200000U
/**
  * @}
  */ 

/** @defgroup RCCEx_SAI_BlockB_Clock_Source  RCC SAI BlockB Clock Source
  * @{
  */
#define RCC_SAIBCLKSOURCE_PLLSAI             0x00000000U
#define RCC_SAIBCLKSOURCE_PLLI2S             0x00400000U
#define RCC_SAIBCLKSOURCE_EXT                0x00800000U
/**
  * @}
  */ 
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */
      
#if defined(STM32F469xx) || defined(STM32F479xx)
/** @defgroup RCCEx_CLK48_Clock_Source  RCC CLK48 Clock Source
  * @{
  */
#define RCC_CLK48CLKSOURCE_PLLQ              0x00000000U
#define RCC_CLK48CLKSOURCE_PLLSAIP           ((uint32_t)RCC_DCKCFGR_CK48MSEL)
/**
  * @}
  */

/** @defgroup RCCEx_SDIO_Clock_Source  RCC SDIO Clock Source
  * @{
  */
#define RCC_SDIOCLKSOURCE_CLK48             0x00000000U
#define RCC_SDIOCLKSOURCE_SYSCLK            ((uint32_t)RCC_DCKCFGR_SDIOSEL)
/**
  * @}
  */    
  
/** @defgroup RCCEx_DSI_Clock_Source  RCC DSI Clock Source
  * @{
  */
#define RCC_DSICLKSOURCE_DSIPHY             0x00000000U
#define RCC_DSICLKSOURCE_PLLR               ((uint32_t)RCC_DCKCFGR_DSISEL)
/**
  * @}
  */
#endif /* STM32F469xx || STM32F479xx */

#if defined(STM32F446xx)
/** @defgroup RCCEx_SAI1_Clock_Source RCC SAI1 Clock Source 
  * @{
  */
#define RCC_SAI1CLKSOURCE_PLLSAI             0x00000000U
#define RCC_SAI1CLKSOURCE_PLLI2S             ((uint32_t)RCC_DCKCFGR_SAI1SRC_0)
#define RCC_SAI1CLKSOURCE_PLLR               ((uint32_t)RCC_DCKCFGR_SAI1SRC_1)
#define RCC_SAI1CLKSOURCE_EXT                ((uint32_t)RCC_DCKCFGR_SAI1SRC)
/**
  * @}
  */ 

/** @defgroup RCCEx_SAI2_Clock_Source  RCC SAI2 Clock Source
  * @{
  */
#define RCC_SAI2CLKSOURCE_PLLSAI             0x00000000U
#define RCC_SAI2CLKSOURCE_PLLI2S             ((uint32_t)RCC_DCKCFGR_SAI2SRC_0)
#define RCC_SAI2CLKSOURCE_PLLR               ((uint32_t)RCC_DCKCFGR_SAI2SRC_1)
#define RCC_SAI2CLKSOURCE_PLLSRC             ((uint32_t)RCC_DCKCFGR_SAI2SRC)
/**
  * @}
  */

/** @defgroup RCCEx_I2SAPB1_Clock_Source  RCC I2S APB1 Clock Source
  * @{
  */
#define RCC_I2SAPB1CLKSOURCE_PLLI2S          0x00000000U
#define RCC_I2SAPB1CLKSOURCE_EXT             ((uint32_t)RCC_DCKCFGR_I2S1SRC_0)
#define RCC_I2SAPB1CLKSOURCE_PLLR            ((uint32_t)RCC_DCKCFGR_I2S1SRC_1)
#define RCC_I2SAPB1CLKSOURCE_PLLSRC          ((uint32_t)RCC_DCKCFGR_I2S1SRC)
/**
  * @}
  */ 

/** @defgroup RCCEx_I2SAPB2_Clock_Source  RCC I2S APB2 Clock Source
  * @{
  */
#define RCC_I2SAPB2CLKSOURCE_PLLI2S          0x00000000U
#define RCC_I2SAPB2CLKSOURCE_EXT             ((uint32_t)RCC_DCKCFGR_I2S2SRC_0)
#define RCC_I2SAPB2CLKSOURCE_PLLR            ((uint32_t)RCC_DCKCFGR_I2S2SRC_1)
#define RCC_I2SAPB2CLKSOURCE_PLLSRC          ((uint32_t)RCC_DCKCFGR_I2S2SRC)
/**
  * @}
  */

/** @defgroup RCCEx_FMPI2C1_Clock_Source  RCC FMPI2C1 Clock Source
  * @{
  */
#define RCC_FMPI2C1CLKSOURCE_PCLK1            0x00000000U
#define RCC_FMPI2C1CLKSOURCE_SYSCLK           ((uint32_t)RCC_DCKCFGR2_FMPI2C1SEL_0)
#define RCC_FMPI2C1CLKSOURCE_HSI              ((uint32_t)RCC_DCKCFGR2_FMPI2C1SEL_1)
/**
  * @}
  */

/** @defgroup RCCEx_CEC_Clock_Source  RCC CEC Clock Source
  * @{
  */
#define RCC_CECCLKSOURCE_HSI                0x00000000U
#define RCC_CECCLKSOURCE_LSE                ((uint32_t)RCC_DCKCFGR2_CECSEL)
/**
  * @}
  */

/** @defgroup RCCEx_CLK48_Clock_Source  RCC CLK48 Clock Source
  * @{
  */
#define RCC_CLK48CLKSOURCE_PLLQ              0x00000000U
#define RCC_CLK48CLKSOURCE_PLLSAIP           ((uint32_t)RCC_DCKCFGR2_CK48MSEL)
/**
  * @}
  */

/** @defgroup RCCEx_SDIO_Clock_Source  RCC SDIO Clock Source
  * @{
  */
#define RCC_SDIOCLKSOURCE_CLK48             0x00000000U
#define RCC_SDIOCLKSOURCE_SYSCLK            ((uint32_t)RCC_DCKCFGR2_SDIOSEL)
/**
  * @}
  */

/** @defgroup RCCEx_SPDIFRX_Clock_Source   RCC SPDIFRX Clock Source
  * @{
  */
#define RCC_SPDIFRXCLKSOURCE_PLLR           0x00000000U
#define RCC_SPDIFRXCLKSOURCE_PLLI2SP        ((uint32_t)RCC_DCKCFGR2_SPDIFRXSEL)
/**
  * @}
  */

#endif /* STM32F446xx */

#if defined(STM32F413xx) || defined(STM32F423xx)
/** @defgroup RCCEx_SAI1_BlockA_Clock_Source  RCC SAI BlockA Clock Source
  * @{
  */
#define RCC_SAIACLKSOURCE_PLLI2SR            0x00000000U
#define RCC_SAIACLKSOURCE_EXT                ((uint32_t)RCC_DCKCFGR_SAI1ASRC_0)
#define RCC_SAIACLKSOURCE_PLLR               ((uint32_t)RCC_DCKCFGR_SAI1ASRC_1)
#define RCC_SAIACLKSOURCE_PLLSRC             ((uint32_t)RCC_DCKCFGR_SAI1ASRC_0 | RCC_DCKCFGR_SAI1ASRC_1)
/**
  * @}
  */ 

/** @defgroup RCCEx_SAI1_BlockB_Clock_Source  RCC SAI BlockB Clock Source
  * @{
  */
#define RCC_SAIBCLKSOURCE_PLLI2SR            0x00000000U
#define RCC_SAIBCLKSOURCE_EXT                ((uint32_t)RCC_DCKCFGR_SAI1BSRC_0)
#define RCC_SAIBCLKSOURCE_PLLR               ((uint32_t)RCC_DCKCFGR_SAI1BSRC_1)
#define RCC_SAIBCLKSOURCE_PLLSRC             ((uint32_t)RCC_DCKCFGR_SAI1BSRC_0 | RCC_DCKCFGR_SAI1BSRC_1)
/**
  * @}
  */ 
      
/** @defgroup RCCEx_LPTIM1_Clock_Source  RCC LPTIM1 Clock Source
  * @{
  */
#define RCC_LPTIM1CLKSOURCE_PCLK1           0x00000000U
#define RCC_LPTIM1CLKSOURCE_HSI             ((uint32_t)RCC_DCKCFGR2_LPTIM1SEL_0)
#define RCC_LPTIM1CLKSOURCE_LSI             ((uint32_t)RCC_DCKCFGR2_LPTIM1SEL_1)
#define RCC_LPTIM1CLKSOURCE_LSE             ((uint32_t)RCC_DCKCFGR2_LPTIM1SEL_0 | RCC_DCKCFGR2_LPTIM1SEL_1)
/**
  * @}
  */
      

/** @defgroup RCCEx_DFSDM2_Audio_Clock_Source  RCC DFSDM2 Audio Clock Source
  * @{
  */
#define RCC_DFSDM2AUDIOCLKSOURCE_I2S1       0x00000000U
#define RCC_DFSDM2AUDIOCLKSOURCE_I2S2       ((uint32_t)RCC_DCKCFGR_CKDFSDM2ASEL)
/**
  * @}
  */

/** @defgroup RCCEx_DFSDM2_Kernel_Clock_Source  RCC DFSDM2 Kernel Clock Source
  * @{
  */
#define RCC_DFSDM2CLKSOURCE_PCLK2           0x00000000U
#define RCC_DFSDM2CLKSOURCE_SYSCLK          ((uint32_t)RCC_DCKCFGR_CKDFSDM1SEL)
/**
  * @}
  */

#endif /* STM32F413xx || STM32F423xx */

#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/** @defgroup RCCEx_PLL_I2S_Clock_Source PLL I2S Clock Source
  * @{
  */
#define RCC_PLLI2SCLKSOURCE_PLLSRC          0x00000000U 
#define RCC_PLLI2SCLKSOURCE_EXT             ((uint32_t)RCC_PLLI2SCFGR_PLLI2SSRC)
/**
  * @}
  */

/** @defgroup RCCEx_DFSDM1_Audio_Clock_Source  RCC DFSDM1 Audio Clock Source
  * @{
  */
#define RCC_DFSDM1AUDIOCLKSOURCE_I2S1       0x00000000U
#define RCC_DFSDM1AUDIOCLKSOURCE_I2S2       ((uint32_t)RCC_DCKCFGR_CKDFSDM1ASEL)
/**
  * @}
  */

/** @defgroup RCCEx_DFSDM1_Kernel_Clock_Source  RCC DFSDM1 Kernel Clock Source
  * @{
  */
#define RCC_DFSDM1CLKSOURCE_PCLK2           0x00000000U
#define RCC_DFSDM1CLKSOURCE_SYSCLK          ((uint32_t)RCC_DCKCFGR_CKDFSDM1SEL)
/**
  * @}
  */

/** @defgroup RCCEx_I2SAPB1_Clock_Source  RCC I2S APB1 Clock Source
  * @{
  */
#define RCC_I2SAPB1CLKSOURCE_PLLI2S         0x00000000U
#define RCC_I2SAPB1CLKSOURCE_EXT            ((uint32_t)RCC_DCKCFGR_I2S1SRC_0)
#define RCC_I2SAPB1CLKSOURCE_PLLR           ((uint32_t)RCC_DCKCFGR_I2S1SRC_1)
#define RCC_I2SAPB1CLKSOURCE_PLLSRC         ((uint32_t)RCC_DCKCFGR_I2S1SRC)
/**
  * @}
  */

/** @defgroup RCCEx_I2SAPB2_Clock_Source  RCC I2S APB2 Clock Source
  * @{
  */
#define RCC_I2SAPB2CLKSOURCE_PLLI2S         0x00000000U
#define RCC_I2SAPB2CLKSOURCE_EXT            ((uint32_t)RCC_DCKCFGR_I2S2SRC_0)
#define RCC_I2SAPB2CLKSOURCE_PLLR           ((uint32_t)RCC_DCKCFGR_I2S2SRC_1)
#define RCC_I2SAPB2CLKSOURCE_PLLSRC         ((uint32_t)RCC_DCKCFGR_I2S2SRC)
/**
  * @}
  */

/** @defgroup RCCEx_FMPI2C1_Clock_Source  RCC FMPI2C1 Clock Source
  * @{
  */
#define RCC_FMPI2C1CLKSOURCE_PCLK1          0x00000000U
#define RCC_FMPI2C1CLKSOURCE_SYSCLK         ((uint32_t)RCC_DCKCFGR2_FMPI2C1SEL_0)
#define RCC_FMPI2C1CLKSOURCE_HSI            ((uint32_t)RCC_DCKCFGR2_FMPI2C1SEL_1)
/**
  * @}
  */

/** @defgroup RCCEx_CLK48_Clock_Source  RCC CLK48 Clock Source
  * @{
  */
#define RCC_CLK48CLKSOURCE_PLLQ             0x00000000U
#define RCC_CLK48CLKSOURCE_PLLI2SQ          ((uint32_t)RCC_DCKCFGR2_CK48MSEL)
/**
  * @}
  */

/** @defgroup RCCEx_SDIO_Clock_Source  RCC SDIO Clock Source
  * @{
  */
#define RCC_SDIOCLKSOURCE_CLK48             0x00000000U
#define RCC_SDIOCLKSOURCE_SYSCLK            ((uint32_t)RCC_DCKCFGR2_SDIOSEL)
/**
  * @}
  */
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)

/** @defgroup RCCEx_I2S_APB_Clock_Source  RCC I2S APB Clock Source
  * @{
  */
#define RCC_I2SAPBCLKSOURCE_PLLR            0x00000000U
#define RCC_I2SAPBCLKSOURCE_EXT             ((uint32_t)RCC_DCKCFGR_I2SSRC_0)
#define RCC_I2SAPBCLKSOURCE_PLLSRC          ((uint32_t)RCC_DCKCFGR_I2SSRC_1)
/**
  * @}
  */

/** @defgroup RCCEx_FMPI2C1_Clock_Source  RCC FMPI2C1 Clock Source
  * @{
  */
#define RCC_FMPI2C1CLKSOURCE_PCLK1              0x00000000U
#define RCC_FMPI2C1CLKSOURCE_SYSCLK             ((uint32_t)RCC_DCKCFGR2_FMPI2C1SEL_0)
#define RCC_FMPI2C1CLKSOURCE_HSI                ((uint32_t)RCC_DCKCFGR2_FMPI2C1SEL_1)
/**
  * @}
  */

/** @defgroup RCCEx_LPTIM1_Clock_Source  RCC LPTIM1 Clock Source
  * @{
  */
#define RCC_LPTIM1CLKSOURCE_PCLK1          0x00000000U
#define RCC_LPTIM1CLKSOURCE_HSI            ((uint32_t)RCC_DCKCFGR2_LPTIM1SEL_0)
#define RCC_LPTIM1CLKSOURCE_LSI            ((uint32_t)RCC_DCKCFGR2_LPTIM1SEL_1)
#define RCC_LPTIM1CLKSOURCE_LSE            ((uint32_t)RCC_DCKCFGR2_LPTIM1SEL_0 | RCC_DCKCFGR2_LPTIM1SEL_1)
/**
  * @}
  */
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) ||\
    defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/** @defgroup RCCEx_TIM_PRescaler_Selection  RCC TIM PRescaler Selection
  * @{
  */
#define RCC_TIMPRES_DESACTIVATED        ((uint8_t)0x00)
#define RCC_TIMPRES_ACTIVATED           ((uint8_t)0x01)
/**
  * @}
  */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F401xC || STM32F401xE ||\
          STM32F410xx || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F411xE) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)
/** @defgroup RCCEx_LSE_Dual_Mode_Selection  RCC LSE Dual Mode Selection
  * @{
  */
#define RCC_LSE_LOWPOWER_MODE           ((uint8_t)0x00)
#define RCC_LSE_HIGHDRIVE_MODE          ((uint8_t)0x01)
/**
  * @}
  */
#endif /* STM32F410xx || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx || STM32F412Vx ||\
          STM32F412Rx || STM32F412Cx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || \
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx) || \
    defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || \
    defined(STM32F412Rx) || defined(STM32F413xx) || defined(STM32F423xx)
/** @defgroup RCC_MCO2_Clock_Source MCO2 Clock Source
  * @{
  */
#define RCC_MCO2SOURCE_SYSCLK            0x00000000U
#define RCC_MCO2SOURCE_PLLI2SCLK         RCC_CFGR_MCO2_0
#define RCC_MCO2SOURCE_HSE               RCC_CFGR_MCO2_1
#define RCC_MCO2SOURCE_PLLCLK            RCC_CFGR_MCO2
/**
  * @}
  */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx ||
          STM32F401xC || STM32F401xE || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx || STM32F412Vx ||
          STM32F412Rx || STM32F413xx | STM32F423xx */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
/** @defgroup RCC_MCO2_Clock_Source MCO2 Clock Source
  * @{
  */
#define RCC_MCO2SOURCE_SYSCLK            0x00000000U
#define RCC_MCO2SOURCE_I2SCLK            RCC_CFGR_MCO2_0
#define RCC_MCO2SOURCE_HSE               RCC_CFGR_MCO2_1
#define RCC_MCO2SOURCE_PLLCLK            RCC_CFGR_MCO2
/**
  * @}
  */
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

/**
  * @}
  */
     
/* Exported macro ------------------------------------------------------------*/
/** @defgroup RCCEx_Exported_Macros RCCEx Exported Macros
  * @{
  */
/*------------------- STM32F42xxx/STM32F43xxx/STM32F469xx/STM32F479xx --------*/
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
/** @defgroup RCCEx_AHB1_Clock_Enable_Disable AHB1 Peripheral Clock Enable Disable
  * @brief  Enables or disables the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_BKPSRAM_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_BKPSRAMEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_BKPSRAMEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_CCMDATARAMEN_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_CRC_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOD_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOE_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOI_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOIEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOIEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOF_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOFEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOFEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOG_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOGEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOGEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOJ_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOJEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOJEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOK_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOKEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOKEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_DMA2D_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_DMA2DEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_DMA2DEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_ETHMAC_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_ETHMACTX_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACTXEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACTXEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_ETHMACRX_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACRXEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACRXEN);\
                                         UNUSED(tmpreg); \
                                         } while(0U)
#define __HAL_RCC_ETHMACPTP_CLK_ENABLE() do { \
                                         __IO uint32_t tmpreg = 0x00U; \
                                         SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACPTPEN);\
                                         /* Delay after an RCC peripheral clock enabling */ \
                                         tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACPTPEN);\
                                         UNUSED(tmpreg); \
                                         } while(0U)
#define __HAL_RCC_USB_OTG_HS_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSULPIEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSULPIEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOD_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIODEN))
#define __HAL_RCC_GPIOE_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOEEN))
#define __HAL_RCC_GPIOF_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOFEN))
#define __HAL_RCC_GPIOG_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOGEN))
#define __HAL_RCC_GPIOI_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOIEN))
#define __HAL_RCC_GPIOJ_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOJEN))
#define __HAL_RCC_GPIOK_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOKEN))
#define __HAL_RCC_DMA2D_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_DMA2DEN))
#define __HAL_RCC_ETHMAC_CLK_DISABLE()          (RCC->AHB1ENR &= ~(RCC_AHB1ENR_ETHMACEN))
#define __HAL_RCC_ETHMACTX_CLK_DISABLE()        (RCC->AHB1ENR &= ~(RCC_AHB1ENR_ETHMACTXEN))
#define __HAL_RCC_ETHMACRX_CLK_DISABLE()        (RCC->AHB1ENR &= ~(RCC_AHB1ENR_ETHMACRXEN))
#define __HAL_RCC_ETHMACPTP_CLK_DISABLE()       (RCC->AHB1ENR &= ~(RCC_AHB1ENR_ETHMACPTPEN))
#define __HAL_RCC_USB_OTG_HS_CLK_DISABLE()      (RCC->AHB1ENR &= ~(RCC_AHB1ENR_OTGHSEN))
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_DISABLE() (RCC->AHB1ENR &= ~(RCC_AHB1ENR_OTGHSULPIEN))
#define __HAL_RCC_BKPSRAM_CLK_DISABLE()         (RCC->AHB1ENR &= ~(RCC_AHB1ENR_BKPSRAMEN))
#define __HAL_RCC_CCMDATARAMEN_CLK_DISABLE()    (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CCMDATARAMEN))
#define __HAL_RCC_CRC_CLK_DISABLE()             (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CRCEN))

/**
  * @brief  Enable ETHERNET clock.
  */
#define __HAL_RCC_ETH_CLK_ENABLE() do {                                     \
                                        __HAL_RCC_ETHMAC_CLK_ENABLE();      \
                                        __HAL_RCC_ETHMACTX_CLK_ENABLE();    \
                                        __HAL_RCC_ETHMACRX_CLK_ENABLE();    \
                                      } while(0U)
/**
  * @brief  Disable ETHERNET clock.
  */
#define __HAL_RCC_ETH_CLK_DISABLE()  do {                                      \
                                          __HAL_RCC_ETHMACTX_CLK_DISABLE();    \
                                          __HAL_RCC_ETHMACRX_CLK_DISABLE();    \
                                          __HAL_RCC_ETHMAC_CLK_DISABLE();      \
                                        } while(0U)
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB1_Peripheral_Clock_Enable_Disable_Status AHB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_GPIOD_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) != RESET) 
#define __HAL_RCC_GPIOE_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) != RESET) 
#define __HAL_RCC_GPIOF_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOFEN)) != RESET) 
#define __HAL_RCC_GPIOG_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOGEN)) != RESET)
#define __HAL_RCC_GPIOI_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOIEN)) != RESET) 
#define __HAL_RCC_GPIOJ_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOJEN)) != RESET) 
#define __HAL_RCC_GPIOK_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOKEN)) != RESET)
#define __HAL_RCC_DMA2D_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_DMA2DEN)) != RESET) 
#define __HAL_RCC_ETHMAC_IS_CLK_ENABLED()          ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACEN)) != RESET) 
#define __HAL_RCC_ETHMACTX_IS_CLK_ENABLED()        ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACTXEN)) != RESET)
#define __HAL_RCC_ETHMACRX_IS_CLK_ENABLED()        ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACRXEN)) != RESET)
#define __HAL_RCC_ETHMACPTP_IS_CLK_ENABLED()       ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACPTPEN)) != RESET)
#define __HAL_RCC_USB_OTG_HS_IS_CLK_ENABLED()      ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSEN)) != RESET)
#define __HAL_RCC_USB_OTG_HS_ULPI_IS_CLK_ENABLED() ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSULPIEN)) != RESET)
#define __HAL_RCC_BKPSRAM_IS_CLK_ENABLED()         ((RCC->AHB1ENR & (RCC_AHB1ENR_BKPSRAMEN)) != RESET)
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_ENABLED()    ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN)) != RESET) 
#define __HAL_RCC_CRC_IS_CLK_ENABLED()             ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) != RESET)
#define __HAL_RCC_ETH_IS_CLK_ENABLED()             (__HAL_RCC_ETHMAC_IS_CLK_ENABLED()   && \
                                                    __HAL_RCC_ETHMACTX_IS_CLK_ENABLED() && \
                                                    __HAL_RCC_ETHMACRX_IS_CLK_ENABLED()) 

#define __HAL_RCC_GPIOD_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) == RESET) 
#define __HAL_RCC_GPIOE_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) == RESET) 
#define __HAL_RCC_GPIOF_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOFEN)) == RESET) 
#define __HAL_RCC_GPIOG_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOGEN)) == RESET)
#define __HAL_RCC_GPIOI_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOIEN)) == RESET) 
#define __HAL_RCC_GPIOJ_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOJEN)) == RESET) 
#define __HAL_RCC_GPIOK_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOKEN)) == RESET)
#define __HAL_RCC_DMA2D_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_DMA2DEN)) == RESET) 
#define __HAL_RCC_ETHMAC_IS_CLK_DISABLED()          ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACEN)) == RESET) 
#define __HAL_RCC_ETHMACTX_IS_CLK_DISABLED()        ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACTXEN)) == RESET)
#define __HAL_RCC_ETHMACRX_IS_CLK_DISABLED()        ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACRXEN)) == RESET)
#define __HAL_RCC_ETHMACPTP_IS_CLK_DISABLED()       ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACPTPEN)) == RESET)
#define __HAL_RCC_USB_OTG_HS_IS_CLK_DISABLED()      ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSEN)) == RESET)
#define __HAL_RCC_USB_OTG_HS_ULPI_IS_CLK_DISABLED() ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSULPIEN)) == RESET)
#define __HAL_RCC_BKPSRAM_IS_CLK_DISABLED()         ((RCC->AHB1ENR & (RCC_AHB1ENR_BKPSRAMEN)) == RESET)
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_DISABLED()    ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN)) == RESET) 
#define __HAL_RCC_CRC_IS_CLK_DISABLED()             ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) == RESET)
#define __HAL_RCC_ETH_IS_CLK_DISABLED()             (__HAL_RCC_ETHMAC_IS_CLK_DISABLED()   && \
                                                     __HAL_RCC_ETHMACTX_IS_CLK_DISABLED() && \
                                                     __HAL_RCC_ETHMACRX_IS_CLK_DISABLED())
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB2_Clock_Enable_Disable AHB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
 #define __HAL_RCC_DCMI_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_DCMIEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_DCMIEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_DCMI_CLK_DISABLE()  (RCC->AHB2ENR &= ~(RCC_AHB2ENR_DCMIEN))

#if defined(STM32F437xx)|| defined(STM32F439xx) || defined(STM32F479xx)
#define __HAL_RCC_CRYP_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_CRYPEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_CRYPEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_HASH_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_HASHEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_HASHEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_CRYP_CLK_DISABLE()  (RCC->AHB2ENR &= ~(RCC_AHB2ENR_CRYPEN))
#define __HAL_RCC_HASH_CLK_DISABLE()  (RCC->AHB2ENR &= ~(RCC_AHB2ENR_HASHEN))
#endif /* STM32F437xx || STM32F439xx || STM32F479xx */

#define __HAL_RCC_USB_OTG_FS_CLK_ENABLE()  do {(RCC->AHB2ENR |= (RCC_AHB2ENR_OTGFSEN));\
                                               __HAL_RCC_SYSCFG_CLK_ENABLE();\
                                              }while(0U)
                                        
#define __HAL_RCC_USB_OTG_FS_CLK_DISABLE() (RCC->AHB2ENR &= ~(RCC_AHB2ENR_OTGFSEN))

#define __HAL_RCC_RNG_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_RNGEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_RNGEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_RNG_CLK_DISABLE()   (RCC->AHB2ENR &= ~(RCC_AHB2ENR_RNGEN))
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB2_Peripheral_Clock_Enable_Disable_Status AHB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */ 
#define __HAL_RCC_DCMI_IS_CLK_ENABLED()        ((RCC->AHB2ENR & (RCC_AHB2ENR_DCMIEN)) != RESET)
#define __HAL_RCC_DCMI_IS_CLK_DISABLED()       ((RCC->AHB2ENR & (RCC_AHB2ENR_DCMIEN)) == RESET)

#if defined(STM32F437xx)|| defined(STM32F439xx) || defined(STM32F479xx)
#define __HAL_RCC_CRYP_IS_CLK_ENABLED()        ((RCC->AHB2ENR & (RCC_AHB2ENR_CRYPEN)) != RESET)
#define __HAL_RCC_CRYP_IS_CLK_DISABLED()       ((RCC->AHB2ENR & (RCC_AHB2ENR_CRYPEN)) == RESET)

#define __HAL_RCC_HASH_IS_CLK_ENABLED()        ((RCC->AHB2ENR & (RCC_AHB2ENR_HASHEN)) != RESET)
#define __HAL_RCC_HASH_IS_CLK_DISABLED()       ((RCC->AHB2ENR & (RCC_AHB2ENR_HASHEN)) == RESET)
#endif /* STM32F437xx || STM32F439xx || STM32F479xx */

#define __HAL_RCC_USB_OTG_FS_IS_CLK_ENABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) != RESET)
#define __HAL_RCC_USB_OTG_FS_IS_CLK_DISABLED() ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) == RESET)

#define __HAL_RCC_RNG_IS_CLK_ENABLED()         ((RCC->AHB2ENR & (RCC_AHB2ENR_RNGEN)) != RESET) 
#define __HAL_RCC_RNG_IS_CLK_DISABLED()        ((RCC->AHB2ENR & (RCC_AHB2ENR_RNGEN)) == RESET)     
/**
  * @}
  */   

/** @defgroup RCCEx_AHB3_Clock_Enable_Disable AHB3 Peripheral Clock Enable Disable
  * @brief  Enables or disables the AHB3 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{  
  */
#define __HAL_RCC_FMC_CLK_ENABLE()    do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB3ENR, RCC_AHB3ENR_FMCEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB3ENR, RCC_AHB3ENR_FMCEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_FMC_CLK_DISABLE()  (RCC->AHB3ENR &= ~(RCC_AHB3ENR_FMCEN))
#if defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_QSPI_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB3ENR, RCC_AHB3ENR_QSPIEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB3ENR, RCC_AHB3ENR_QSPIEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_QSPI_CLK_DISABLE()  (RCC->AHB3ENR &= ~(RCC_AHB3ENR_QSPIEN))
#endif /* STM32F469xx || STM32F479xx */
/**
  * @}
  */


/** @defgroup RCCEx_AHB3_Peripheral_Clock_Enable_Disable_Status AHB3 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB3 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_FMC_IS_CLK_ENABLED()   ((RCC->AHB3ENR & (RCC_AHB3ENR_FMCEN)) != RESET)
#define __HAL_RCC_FMC_IS_CLK_DISABLED()  ((RCC->AHB3ENR & (RCC_AHB3ENR_FMCEN)) == RESET)
#if defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_QSPI_IS_CLK_ENABLED()  ((RCC->AHB3ENR & (RCC_AHB3ENR_QSPIEN)) != RESET)
#define __HAL_RCC_QSPI_IS_CLK_DISABLED() ((RCC->AHB3ENR & (RCC_AHB3ENR_QSPIEN)) == RESET)
#endif /* STM32F469xx || STM32F479xx */  
/**
  * @}
  */
    
/** @defgroup RCCEx_APB1_Clock_Enable_Disable APB1 Peripheral Clock Enable Disable
  * @brief  Enable or disable the Low Speed APB (APB1) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM6_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM7_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM7EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM7EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM12_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM12EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM12EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM13_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM13EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM13EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM14_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM14_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_USART3_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_USART3EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_USART3EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART4_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART4EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART4EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART5_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART5EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART5EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CAN1_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CAN2_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN2EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN2EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_DAC_CLK_ENABLE()    do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART7_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART7EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART7EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART8_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART8EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART8EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM2_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_I2C3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM2_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM2EN))
#define __HAL_RCC_TIM3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM3EN))
#define __HAL_RCC_TIM4_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM4EN))
#define __HAL_RCC_SPI3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_SPI3EN))
#define __HAL_RCC_I2C3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_I2C3EN))
#define __HAL_RCC_TIM6_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM6EN))
#define __HAL_RCC_TIM7_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM7EN))
#define __HAL_RCC_TIM12_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM12EN))
#define __HAL_RCC_TIM13_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM13EN))
#define __HAL_RCC_TIM14_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM14EN))
#define __HAL_RCC_USART3_CLK_DISABLE() (RCC->APB1ENR &= ~(RCC_APB1ENR_USART3EN))
#define __HAL_RCC_UART4_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_UART4EN))
#define __HAL_RCC_UART5_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_UART5EN))
#define __HAL_RCC_CAN1_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_CAN1EN))
#define __HAL_RCC_CAN2_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_CAN2EN))
#define __HAL_RCC_DAC_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_DACEN))
#define __HAL_RCC_UART7_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_UART7EN))
#define __HAL_RCC_UART8_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_UART8EN))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Peripheral_Clock_Enable_Disable_Status APB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM2_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) != RESET)  
#define __HAL_RCC_TIM3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) != RESET) 
#define __HAL_RCC_TIM4_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) != RESET)
#define __HAL_RCC_SPI3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) != RESET) 
#define __HAL_RCC_I2C3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) != RESET)
#define __HAL_RCC_TIM6_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) != RESET) 
#define __HAL_RCC_TIM7_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM7EN)) != RESET) 
#define __HAL_RCC_TIM12_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM12EN)) != RESET) 
#define __HAL_RCC_TIM13_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM13EN)) != RESET)  
#define __HAL_RCC_TIM14_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM14EN)) != RESET) 
#define __HAL_RCC_USART3_IS_CLK_ENABLED() ((RCC->APB1ENR & (RCC_APB1ENR_USART3EN)) != RESET) 
#define __HAL_RCC_UART4_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART4EN)) != RESET) 
#define __HAL_RCC_UART5_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART5EN)) != RESET) 
#define __HAL_RCC_CAN1_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN1EN)) != RESET)
#define __HAL_RCC_CAN2_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN2EN)) != RESET)
#define __HAL_RCC_DAC_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) != RESET) 
#define __HAL_RCC_UART7_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART7EN)) != RESET)
#define __HAL_RCC_UART8_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART8EN)) != RESET) 

#define __HAL_RCC_TIM2_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) == RESET)  
#define __HAL_RCC_TIM3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) == RESET) 
#define __HAL_RCC_TIM4_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) == RESET)
#define __HAL_RCC_SPI3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) == RESET) 
#define __HAL_RCC_I2C3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) == RESET)
#define __HAL_RCC_TIM6_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) == RESET) 
#define __HAL_RCC_TIM7_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM7EN)) == RESET) 
#define __HAL_RCC_TIM12_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM12EN)) == RESET) 
#define __HAL_RCC_TIM13_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM13EN)) == RESET)  
#define __HAL_RCC_TIM14_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM14EN)) == RESET) 
#define __HAL_RCC_USART3_IS_CLK_DISABLED() ((RCC->APB1ENR & (RCC_APB1ENR_USART3EN)) == RESET) 
#define __HAL_RCC_UART4_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART4EN)) == RESET) 
#define __HAL_RCC_UART5_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART5EN)) == RESET) 
#define __HAL_RCC_CAN1_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN1EN)) == RESET)
#define __HAL_RCC_CAN2_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN2EN)) == RESET)
#define __HAL_RCC_DAC_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) == RESET) 
#define __HAL_RCC_UART7_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART7EN)) == RESET)
#define __HAL_RCC_UART8_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART8EN)) == RESET) 
/**
  * @}
  */
    
/** @defgroup RCCEx_APB2_Clock_Enable_Disable APB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the High Speed APB (APB2) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM8_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM8EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM8EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_ADC2_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC2EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC2EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_ADC3_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC3EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC3EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI5_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI5EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI5EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI6_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI6EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI6EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SAI1_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SAI1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SAI1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SDIO_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM10_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SDIO_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SDIOEN))
#define __HAL_RCC_SPI4_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI4EN))
#define __HAL_RCC_TIM10_CLK_DISABLE()  (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM10EN))
#define __HAL_RCC_TIM8_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM8EN))
#define __HAL_RCC_ADC2_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_ADC2EN))
#define __HAL_RCC_ADC3_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_ADC3EN))
#define __HAL_RCC_SPI5_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI5EN))
#define __HAL_RCC_SPI6_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI6EN))
#define __HAL_RCC_SAI1_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SAI1EN))

#if defined(STM32F429xx)|| defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_LTDC_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_LTDCEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_LTDCEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_LTDC_CLK_DISABLE() (RCC->APB2ENR &= ~(RCC_APB2ENR_LTDCEN))
#endif /* STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_DSI_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_DSIEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_DSIEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_DSI_CLK_DISABLE() (RCC->APB2ENR &= ~(RCC_APB2ENR_DSIEN))
#endif /* STM32F469xx || STM32F479xx */
/**
  * @}
  */
  
/** @defgroup RCCEx_APB2_Peripheral_Clock_Enable_Disable_Status APB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */  
#define __HAL_RCC_TIM8_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_TIM8EN)) != RESET)
#define __HAL_RCC_ADC2_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_ADC2EN)) != RESET)
#define __HAL_RCC_ADC3_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_ADC3EN)) != RESET) 
#define __HAL_RCC_SPI5_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_SPI5EN)) != RESET) 
#define __HAL_RCC_SPI6_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_SPI6EN)) != RESET) 
#define __HAL_RCC_SAI1_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_SAI1EN)) != RESET) 
#define __HAL_RCC_SDIO_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) != RESET)
#define __HAL_RCC_SPI4_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) != RESET)
#define __HAL_RCC_TIM10_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN))!= RESET)  

#define __HAL_RCC_SDIO_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) == RESET)
#define __HAL_RCC_SPI4_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) == RESET)
#define __HAL_RCC_TIM10_IS_CLK_DISABLED()  ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN))== RESET)
#define __HAL_RCC_TIM8_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_TIM8EN)) == RESET)
#define __HAL_RCC_ADC2_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_ADC2EN)) == RESET)
#define __HAL_RCC_ADC3_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_ADC3EN)) == RESET)
#define __HAL_RCC_SPI5_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI5EN)) == RESET)
#define __HAL_RCC_SPI6_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI6EN)) == RESET)
#define __HAL_RCC_SAI1_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SAI1EN)) == RESET)

#if defined(STM32F429xx)|| defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_LTDC_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_LTDCEN)) != RESET)
#define __HAL_RCC_LTDC_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_LTDCEN)) == RESET)
#endif /* STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_DSI_IS_CLK_ENABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_DSIEN)) != RESET)
#define __HAL_RCC_DSI_IS_CLK_DISABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_DSIEN)) == RESET)
#endif /* STM32F469xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_Force_Release_Reset AHB1 Force Release Reset 
  * @brief  Force or release AHB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_GPIOD_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_GPIOF_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOFRST))
#define __HAL_RCC_GPIOG_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOGRST))
#define __HAL_RCC_GPIOI_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOIRST))
#define __HAL_RCC_ETHMAC_FORCE_RESET()   (RCC->AHB1RSTR |= (RCC_AHB1RSTR_ETHMACRST))
#define __HAL_RCC_USB_OTG_HS_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_OTGHRST))
#define __HAL_RCC_GPIOJ_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOJRST))
#define __HAL_RCC_GPIOK_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOKRST))
#define __HAL_RCC_DMA2D_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_DMA2DRST))
#define __HAL_RCC_CRC_FORCE_RESET()      (RCC->AHB1RSTR |= (RCC_AHB1RSTR_CRCRST))

#define __HAL_RCC_GPIOD_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_GPIOF_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOFRST))
#define __HAL_RCC_GPIOG_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOGRST))
#define __HAL_RCC_GPIOI_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOIRST))
#define __HAL_RCC_ETHMAC_RELEASE_RESET() (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_ETHMACRST))
#define __HAL_RCC_USB_OTG_HS_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_OTGHRST))
#define __HAL_RCC_GPIOJ_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOJRST))
#define __HAL_RCC_GPIOK_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOKRST))
#define __HAL_RCC_DMA2D_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_DMA2DRST))
#define __HAL_RCC_CRC_RELEASE_RESET()    (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_CRCRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Force_Release_Reset AHB2 Force Release Reset 
  * @brief  Force or release AHB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_AHB2_FORCE_RESET()    (RCC->AHB2RSTR = 0xFFFFFFFFU) 
#define __HAL_RCC_USB_OTG_FS_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_OTGFSRST))
#define __HAL_RCC_RNG_FORCE_RESET()    (RCC->AHB2RSTR |= (RCC_AHB2RSTR_RNGRST))
#define __HAL_RCC_DCMI_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_DCMIRST))

#define __HAL_RCC_AHB2_RELEASE_RESET()  (RCC->AHB2RSTR = 0x00U)
#define __HAL_RCC_USB_OTG_FS_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_OTGFSRST))
#define __HAL_RCC_RNG_RELEASE_RESET()  (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_RNGRST))
#define __HAL_RCC_DCMI_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_DCMIRST))

#if defined(STM32F437xx)|| defined(STM32F439xx) || defined(STM32F479xx) 
#define __HAL_RCC_CRYP_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_CRYPRST))
#define __HAL_RCC_HASH_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_HASHRST))

#define __HAL_RCC_CRYP_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_CRYPRST))
#define __HAL_RCC_HASH_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_HASHRST))
#endif /* STM32F437xx || STM32F439xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Force_Release_Reset AHB3 Force Release Reset 
  * @brief  Force or release AHB3 peripheral reset.
  * @{
  */ 
#define __HAL_RCC_AHB3_FORCE_RESET() (RCC->AHB3RSTR = 0xFFFFFFFFU)
#define __HAL_RCC_AHB3_RELEASE_RESET() (RCC->AHB3RSTR = 0x00U) 
#define __HAL_RCC_FMC_FORCE_RESET()    (RCC->AHB3RSTR |= (RCC_AHB3RSTR_FMCRST))
#define __HAL_RCC_FMC_RELEASE_RESET()  (RCC->AHB3RSTR &= ~(RCC_AHB3RSTR_FMCRST))

#if defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_QSPI_FORCE_RESET()   (RCC->AHB3RSTR |= (RCC_AHB3RSTR_QSPIRST))
#define __HAL_RCC_QSPI_RELEASE_RESET()   (RCC->AHB3RSTR &= ~(RCC_AHB3RSTR_QSPIRST))  
#endif /* STM32F469xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Force_Release_Reset APB1 Force Release Reset 
  * @brief  Force or release APB1 peripheral reset.
  * @{
  */ 
#define __HAL_RCC_TIM6_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_TIM7_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM7RST))
#define __HAL_RCC_TIM12_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM12RST))
#define __HAL_RCC_TIM13_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM13RST))
#define __HAL_RCC_TIM14_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM14RST))
#define __HAL_RCC_USART3_FORCE_RESET()   (RCC->APB1RSTR |= (RCC_APB1RSTR_USART3RST))
#define __HAL_RCC_UART4_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART4RST))
#define __HAL_RCC_UART5_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART5RST))
#define __HAL_RCC_CAN1_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_CAN1RST))
#define __HAL_RCC_CAN2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_CAN2RST))
#define __HAL_RCC_DAC_FORCE_RESET()      (RCC->APB1RSTR |= (RCC_APB1RSTR_DACRST))
#define __HAL_RCC_UART7_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART7RST))
#define __HAL_RCC_UART8_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART8RST))
#define __HAL_RCC_TIM2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_I2C3RST))

#define __HAL_RCC_TIM2_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_I2C3RST))
#define __HAL_RCC_TIM6_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_TIM7_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM7RST))
#define __HAL_RCC_TIM12_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM12RST))
#define __HAL_RCC_TIM13_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM13RST))
#define __HAL_RCC_TIM14_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM14RST))
#define __HAL_RCC_USART3_RELEASE_RESET() (RCC->APB1RSTR &= ~(RCC_APB1RSTR_USART3RST))
#define __HAL_RCC_UART4_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART4RST))
#define __HAL_RCC_UART5_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART5RST))
#define __HAL_RCC_CAN1_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CAN1RST))
#define __HAL_RCC_CAN2_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CAN2RST))
#define __HAL_RCC_DAC_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_DACRST))
#define __HAL_RCC_UART7_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART7RST))
#define __HAL_RCC_UART8_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART8RST))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Force_Release_Reset APB2 Force Release Reset 
  * @brief  Force or release APB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_TIM8_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM8RST))
#define __HAL_RCC_SPI5_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI5RST))
#define __HAL_RCC_SPI6_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI6RST))
#define __HAL_RCC_SAI1_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_SAI1RST))
#define __HAL_RCC_SDIO_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_FORCE_RESET()  (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM10RST))

#define __HAL_RCC_SDIO_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_RELEASE_RESET()(RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM10RST))
#define __HAL_RCC_TIM8_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM8RST))
#define __HAL_RCC_SPI5_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI5RST))
#define __HAL_RCC_SPI6_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI6RST))
#define __HAL_RCC_SAI1_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SAI1RST))

#if defined(STM32F429xx)|| defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_LTDC_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_LTDCRST))
#define __HAL_RCC_LTDC_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_LTDCRST))
#endif /* STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_DSI_FORCE_RESET()   (RCC->APB2RSTR |=  (RCC_APB2RSTR_DSIRST))
#define __HAL_RCC_DSI_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_DSIRST))
#endif /* STM32F469xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_LowPower_Enable_Disable AHB1 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_GPIOD_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_GPIOF_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOFLPEN))
#define __HAL_RCC_GPIOG_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOGLPEN))
#define __HAL_RCC_GPIOI_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOILPEN))
#define __HAL_RCC_SRAM2_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM2LPEN))
#define __HAL_RCC_ETHMAC_CLK_SLEEP_ENABLE()     (RCC->AHB1LPENR |= (RCC_AHB1LPENR_ETHMACLPEN))
#define __HAL_RCC_ETHMACTX_CLK_SLEEP_ENABLE()   (RCC->AHB1LPENR |= (RCC_AHB1LPENR_ETHMACTXLPEN))
#define __HAL_RCC_ETHMACRX_CLK_SLEEP_ENABLE()   (RCC->AHB1LPENR |= (RCC_AHB1LPENR_ETHMACRXLPEN))
#define __HAL_RCC_ETHMACPTP_CLK_SLEEP_ENABLE()  (RCC->AHB1LPENR |= (RCC_AHB1LPENR_ETHMACPTPLPEN))
#define __HAL_RCC_USB_OTG_HS_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_OTGHSLPEN))
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_SLEEP_ENABLE()  (RCC->AHB1LPENR |= (RCC_AHB1LPENR_OTGHSULPILPEN))
#define __HAL_RCC_GPIOJ_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOJLPEN))
#define __HAL_RCC_GPIOK_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOKLPEN))
#define __HAL_RCC_SRAM3_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM3LPEN))
#define __HAL_RCC_DMA2D_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_DMA2DLPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_ENABLE()        (RCC->AHB1LPENR |= (RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM1LPEN))
#define __HAL_RCC_BKPSRAM_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_BKPSRAMLPEN))

#define __HAL_RCC_GPIOD_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_GPIOF_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOFLPEN))
#define __HAL_RCC_GPIOG_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOGLPEN))
#define __HAL_RCC_GPIOI_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOILPEN))
#define __HAL_RCC_SRAM2_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM2LPEN))
#define __HAL_RCC_ETHMAC_CLK_SLEEP_DISABLE()    (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_ETHMACLPEN))
#define __HAL_RCC_ETHMACTX_CLK_SLEEP_DISABLE()  (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_ETHMACTXLPEN))
#define __HAL_RCC_ETHMACRX_CLK_SLEEP_DISABLE()  (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_ETHMACRXLPEN))
#define __HAL_RCC_ETHMACPTP_CLK_SLEEP_DISABLE() (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_ETHMACPTPLPEN))
#define __HAL_RCC_USB_OTG_HS_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_OTGHSLPEN))
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_SLEEP_DISABLE() (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_OTGHSULPILPEN))
#define __HAL_RCC_GPIOJ_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOJLPEN))
#define __HAL_RCC_GPIOK_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOKLPEN))
#define __HAL_RCC_DMA2D_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_DMA2DLPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_DISABLE()       (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM1LPEN))
#define __HAL_RCC_BKPSRAM_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_BKPSRAMLPEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_LowPower_Enable_Disable AHB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wake-up from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_OTGFSLPEN))
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_OTGFSLPEN))

#define __HAL_RCC_RNG_CLK_SLEEP_ENABLE()   (RCC->AHB2LPENR |= (RCC_AHB2LPENR_RNGLPEN))
#define __HAL_RCC_RNG_CLK_SLEEP_DISABLE()  (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_RNGLPEN))

#define __HAL_RCC_DCMI_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_DCMILPEN))
#define __HAL_RCC_DCMI_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_DCMILPEN))

#if defined(STM32F437xx)|| defined(STM32F439xx) || defined(STM32F479xx) 
#define __HAL_RCC_CRYP_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_CRYPLPEN))
#define __HAL_RCC_HASH_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_HASHLPEN))

#define __HAL_RCC_CRYP_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_CRYPLPEN))
#define __HAL_RCC_HASH_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_HASHLPEN))
#endif /* STM32F437xx || STM32F439xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_LowPower_Enable_Disable AHB3 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB3 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_FMC_CLK_SLEEP_ENABLE()  (RCC->AHB3LPENR |= (RCC_AHB3LPENR_FMCLPEN))
#define __HAL_RCC_FMC_CLK_SLEEP_DISABLE() (RCC->AHB3LPENR &= ~(RCC_AHB3LPENR_FMCLPEN))

#if defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_QSPI_CLK_SLEEP_ENABLE()  (RCC->AHB3LPENR |= (RCC_AHB3LPENR_QSPILPEN))
#define __HAL_RCC_QSPI_CLK_SLEEP_DISABLE()  (RCC->AHB3LPENR &= ~(RCC_AHB3LPENR_QSPILPEN))
#endif /* STM32F469xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup RCCEx_APB1_LowPower_Enable_Disable APB1 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */  
#define __HAL_RCC_TIM6_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_TIM7_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM7LPEN))
#define __HAL_RCC_TIM12_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM12LPEN))
#define __HAL_RCC_TIM13_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM13LPEN))
#define __HAL_RCC_TIM14_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM14LPEN))
#define __HAL_RCC_USART3_CLK_SLEEP_ENABLE()  (RCC->APB1LPENR |= (RCC_APB1LPENR_USART3LPEN))
#define __HAL_RCC_UART4_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART4LPEN))
#define __HAL_RCC_UART5_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART5LPEN))
#define __HAL_RCC_CAN1_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_CAN1LPEN))
#define __HAL_RCC_CAN2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_CAN2LPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_ENABLE()     (RCC->APB1LPENR |= (RCC_APB1LPENR_DACLPEN))
#define __HAL_RCC_UART7_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART7LPEN))
#define __HAL_RCC_UART8_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART8LPEN))
#define __HAL_RCC_TIM2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_I2C3LPEN))

#define __HAL_RCC_TIM2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_I2C3LPEN))
#define __HAL_RCC_TIM6_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_TIM7_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM7LPEN))
#define __HAL_RCC_TIM12_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM12LPEN))
#define __HAL_RCC_TIM13_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM13LPEN))
#define __HAL_RCC_TIM14_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM14LPEN))
#define __HAL_RCC_USART3_CLK_SLEEP_DISABLE() (RCC->APB1LPENR &= ~(RCC_APB1LPENR_USART3LPEN))
#define __HAL_RCC_UART4_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART4LPEN))
#define __HAL_RCC_UART5_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART5LPEN))
#define __HAL_RCC_CAN1_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CAN1LPEN))
#define __HAL_RCC_CAN2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CAN2LPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_DISABLE()    (RCC->APB1LPENR &= ~(RCC_APB1LPENR_DACLPEN))
#define __HAL_RCC_UART7_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART7LPEN))
#define __HAL_RCC_UART8_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART8LPEN))
/**
  * @}
  */
                                        
/** @defgroup RCCEx_APB2_LowPower_Enable_Disable APB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */ 
#define __HAL_RCC_TIM8_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_TIM8LPEN))
#define __HAL_RCC_ADC2_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_ADC2LPEN))
#define __HAL_RCC_ADC3_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_ADC3LPEN))
#define __HAL_RCC_SPI5_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI5LPEN))
#define __HAL_RCC_SPI6_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI6LPEN))
#define __HAL_RCC_SAI1_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SAI1LPEN))
#define __HAL_RCC_SDIO_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_ENABLE()(RCC->APB2LPENR |= (RCC_APB2LPENR_TIM10LPEN))

#define __HAL_RCC_SDIO_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_DISABLE()(RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM10LPEN))
#define __HAL_RCC_TIM8_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM8LPEN))
#define __HAL_RCC_ADC2_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_ADC2LPEN))
#define __HAL_RCC_ADC3_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_ADC3LPEN))
#define __HAL_RCC_SPI5_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI5LPEN))
#define __HAL_RCC_SPI6_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI6LPEN))
#define __HAL_RCC_SAI1_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SAI1LPEN))

#if defined(STM32F429xx)|| defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_LTDC_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_LTDCLPEN))

#define __HAL_RCC_LTDC_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_LTDCLPEN))
#endif /* STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F469xx) || defined(STM32F479xx)
#define __HAL_RCC_DSI_CLK_SLEEP_ENABLE()  (RCC->APB2LPENR |=  (RCC_APB2LPENR_DSILPEN))
#define __HAL_RCC_DSI_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_DSILPEN))
#endif /* STM32F469xx || STM32F479xx */
/**
  * @}
  */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */
/*----------------------------------------------------------------------------*/

/*----------------------------------- STM32F40xxx/STM32F41xxx-----------------*/
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx)|| defined(STM32F417xx)
/** @defgroup RCCEx_AHB1_Clock_Enable_Disable AHB1 Peripheral Clock Enable Disable
  * @brief  Enables or disables the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_BKPSRAM_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_BKPSRAMEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_BKPSRAMEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_CCMDATARAMEN_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_CRC_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOD_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_GPIOE_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_GPIOI_CLK_ENABLE()   do { \
                                       __IO uint32_t tmpreg = 0x00U; \
                                       SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOIEN);\
                                       /* Delay after an RCC peripheral clock enabling */ \
                                       tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOIEN);\
                                       UNUSED(tmpreg); \
                                       } while(0U)
#define __HAL_RCC_GPIOF_CLK_ENABLE()   do { \
                                       __IO uint32_t tmpreg = 0x00U; \
                                       SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOFEN);\
                                       /* Delay after an RCC peripheral clock enabling */ \
                                       tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOFEN);\
                                       UNUSED(tmpreg); \
                                       } while(0U)
#define __HAL_RCC_GPIOG_CLK_ENABLE()   do { \
                                       __IO uint32_t tmpreg = 0x00U; \
                                       SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOGEN);\
                                       /* Delay after an RCC peripheral clock enabling */ \
                                       tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOGEN);\
                                       UNUSED(tmpreg); \
                                       } while(0U)
#define __HAL_RCC_USB_OTG_HS_CLK_ENABLE()   do { \
                                       __IO uint32_t tmpreg = 0x00U; \
                                       SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSEN);\
                                       /* Delay after an RCC peripheral clock enabling */ \
                                       tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSEN);\
                                       UNUSED(tmpreg); \
                                       } while(0U)
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_ENABLE()   do { \
                                       __IO uint32_t tmpreg = 0x00U; \
                                       SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSULPIEN);\
                                       /* Delay after an RCC peripheral clock enabling */ \
                                       tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSULPIEN);\
                                       UNUSED(tmpreg); \
                                       } while(0U)
#define __HAL_RCC_GPIOD_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIODEN))
#define __HAL_RCC_GPIOE_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOEEN))
#define __HAL_RCC_GPIOF_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOFEN))
#define __HAL_RCC_GPIOG_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOGEN))
#define __HAL_RCC_GPIOI_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOIEN))
#define __HAL_RCC_USB_OTG_HS_CLK_DISABLE()      (RCC->AHB1ENR &= ~(RCC_AHB1ENR_OTGHSEN))
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_DISABLE() (RCC->AHB1ENR &= ~(RCC_AHB1ENR_OTGHSULPIEN))
#define __HAL_RCC_BKPSRAM_CLK_DISABLE()         (RCC->AHB1ENR &= ~(RCC_AHB1ENR_BKPSRAMEN))
#define __HAL_RCC_CCMDATARAMEN_CLK_DISABLE()    (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CCMDATARAMEN))
#define __HAL_RCC_CRC_CLK_DISABLE()             (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CRCEN))
#if defined(STM32F407xx)|| defined(STM32F417xx)
/**
  * @brief  Enable ETHERNET clock.
  */
#define __HAL_RCC_ETHMAC_CLK_ENABLE()  do { \
                                       __IO uint32_t tmpreg = 0x00U; \
                                       SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACEN);\
                                       /* Delay after an RCC peripheral clock enabling */ \
                                       tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACEN);\
                                       UNUSED(tmpreg); \
                                       } while(0U)
#define __HAL_RCC_ETHMACTX_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACTXEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACTXEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_ETHMACRX_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACRXEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACRXEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_ETHMACPTP_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACPTPEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_ETHMACPTPEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_ETH_CLK_ENABLE()      do {                            \
                                        __HAL_RCC_ETHMAC_CLK_ENABLE();      \
                                        __HAL_RCC_ETHMACTX_CLK_ENABLE();    \
                                        __HAL_RCC_ETHMACRX_CLK_ENABLE();    \
                                        } while(0U)

/**
  * @brief  Disable ETHERNET clock.
  */
#define __HAL_RCC_ETHMAC_CLK_DISABLE()    (RCC->AHB1ENR &= ~(RCC_AHB1ENR_ETHMACEN))
#define __HAL_RCC_ETHMACTX_CLK_DISABLE()  (RCC->AHB1ENR &= ~(RCC_AHB1ENR_ETHMACTXEN))
#define __HAL_RCC_ETHMACRX_CLK_DISABLE()  (RCC->AHB1ENR &= ~(RCC_AHB1ENR_ETHMACRXEN))
#define __HAL_RCC_ETHMACPTP_CLK_DISABLE() (RCC->AHB1ENR &= ~(RCC_AHB1ENR_ETHMACPTPEN))  
#define __HAL_RCC_ETH_CLK_DISABLE()       do {                             \
                                           __HAL_RCC_ETHMACTX_CLK_DISABLE();    \
                                           __HAL_RCC_ETHMACRX_CLK_DISABLE();    \
                                           __HAL_RCC_ETHMAC_CLK_DISABLE();      \
                                          } while(0U)
#endif /* STM32F407xx || STM32F417xx */
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB1_Peripheral_Clock_Enable_Disable_Status AHB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */  
#define __HAL_RCC_BKPSRAM_IS_CLK_ENABLED()          ((RCC->AHB1ENR & (RCC_AHB1ENR_BKPSRAMEN)) != RESET)
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_ENABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN)) != RESET)
#define __HAL_RCC_CRC_IS_CLK_ENABLED()              ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) != RESET)
#define __HAL_RCC_GPIOD_IS_CLK_ENABLED()            ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) != RESET)
#define __HAL_RCC_GPIOE_IS_CLK_ENABLED()            ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) != RESET)
#define __HAL_RCC_GPIOI_IS_CLK_ENABLED()            ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOIEN)) != RESET)
#define __HAL_RCC_GPIOF_IS_CLK_ENABLED()            ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOFEN)) != RESET)
#define __HAL_RCC_GPIOG_IS_CLK_ENABLED()            ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOGEN)) != RESET)
#define __HAL_RCC_USB_OTG_HS_IS_CLK_ENABLED()       ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSEN)) != RESET)
#define __HAL_RCC_USB_OTG_HS_ULPI_IS_CLK_ENABLED()  ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSULPIEN)) != RESET)

#define __HAL_RCC_GPIOD_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) == RESET)
#define __HAL_RCC_GPIOE_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) == RESET)
#define __HAL_RCC_GPIOF_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOFEN)) == RESET)
#define __HAL_RCC_GPIOG_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOGEN)) == RESET)
#define __HAL_RCC_GPIOI_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOIEN)) == RESET)
#define __HAL_RCC_USB_OTG_HS_IS_CLK_DISABLED()      ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSEN)) == RESET)
#define __HAL_RCC_USB_OTG_HS_ULPI_IS_CLK_DISABLED() ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSULPIEN))== RESET)
#define __HAL_RCC_BKPSRAM_IS_CLK_DISABLED()         ((RCC->AHB1ENR & (RCC_AHB1ENR_BKPSRAMEN)) == RESET)
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_DISABLED()    ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN)) == RESET)
#define __HAL_RCC_CRC_IS_CLK_DISABLED()             ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) == RESET)
#if defined(STM32F407xx)|| defined(STM32F417xx)
/**
  * @brief  Enable ETHERNET clock.
  */
#define __HAL_RCC_ETHMAC_IS_CLK_ENABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACEN)) != RESET)
#define __HAL_RCC_ETHMACTX_IS_CLK_ENABLED()   ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACTXEN)) != RESET)
#define __HAL_RCC_ETHMACRX_IS_CLK_ENABLED()   ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACRXEN)) != RESET)
#define __HAL_RCC_ETHMACPTP_IS_CLK_ENABLED()  ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACPTPEN)) != RESET)
#define __HAL_RCC_ETH_IS_CLK_ENABLED()        (__HAL_RCC_ETHMAC_IS_CLK_ENABLED()   && \
                                               __HAL_RCC_ETHMACTX_IS_CLK_ENABLED() && \
                                               __HAL_RCC_ETHMACRX_IS_CLK_ENABLED())
/**
  * @brief  Disable ETHERNET clock.
  */
#define __HAL_RCC_ETHMAC_IS_CLK_DISABLED()    ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACEN)) == RESET)
#define __HAL_RCC_ETHMACTX_IS_CLK_DISABLED()  ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACTXEN)) == RESET)
#define __HAL_RCC_ETHMACRX_IS_CLK_DISABLED()  ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACRXEN)) == RESET)
#define __HAL_RCC_ETHMACPTP_IS_CLK_DISABLED() ((RCC->AHB1ENR & (RCC_AHB1ENR_ETHMACPTPEN)) == RESET)
#define __HAL_RCC_ETH_IS_CLK_DISABLED()        (__HAL_RCC_ETHMAC_IS_CLK_DISABLED()   && \
                                                __HAL_RCC_ETHMACTX_IS_CLK_DISABLED() && \
                                                __HAL_RCC_ETHMACRX_IS_CLK_DISABLED())
#endif /* STM32F407xx || STM32F417xx */
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB2_Clock_Enable_Disable AHB2 Peripheral Clock Enable Disable 
  * @brief  Enable or disable the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_CLK_ENABLE()  do {(RCC->AHB2ENR |= (RCC_AHB2ENR_OTGFSEN));\
                                               __HAL_RCC_SYSCFG_CLK_ENABLE();\
                                              }while(0U)
                                        
#define __HAL_RCC_USB_OTG_FS_CLK_DISABLE() (RCC->AHB2ENR &= ~(RCC_AHB2ENR_OTGFSEN))

#define __HAL_RCC_RNG_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_RNGEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_RNGEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_RNG_CLK_DISABLE()   (RCC->AHB2ENR &= ~(RCC_AHB2ENR_RNGEN))

#if defined(STM32F407xx)|| defined(STM32F417xx) 
#define __HAL_RCC_DCMI_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_DCMIEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_DCMIEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_DCMI_CLK_DISABLE()  (RCC->AHB2ENR &= ~(RCC_AHB2ENR_DCMIEN))
#endif /* STM32F407xx || STM32F417xx */

#if defined(STM32F415xx) || defined(STM32F417xx)
#define __HAL_RCC_CRYP_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_CRYPEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_CRYPEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_HASH_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_HASHEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_HASHEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CRYP_CLK_DISABLE()  (RCC->AHB2ENR &= ~(RCC_AHB2ENR_CRYPEN))
#define __HAL_RCC_HASH_CLK_DISABLE()  (RCC->AHB2ENR &= ~(RCC_AHB2ENR_HASHEN))
#endif /* STM32F415xx || STM32F417xx */
/**
  * @}
  */


/** @defgroup RCCEx_AHB2_Peripheral_Clock_Enable_Disable_Status AHB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_IS_CLK_ENABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) != RESET)
#define __HAL_RCC_USB_OTG_FS_IS_CLK_DISABLED() ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) == RESET) 

#define __HAL_RCC_RNG_IS_CLK_ENABLED()   ((RCC->AHB2ENR & (RCC_AHB2ENR_RNGEN)) != RESET)   
#define __HAL_RCC_RNG_IS_CLK_DISABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_RNGEN)) == RESET) 

#if defined(STM32F407xx)|| defined(STM32F417xx) 
#define __HAL_RCC_DCMI_IS_CLK_ENABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_DCMIEN)) != RESET) 
#define __HAL_RCC_DCMI_IS_CLK_DISABLED() ((RCC->AHB2ENR & (RCC_AHB2ENR_DCMIEN)) == RESET) 
#endif /* STM32F407xx || STM32F417xx */

#if defined(STM32F415xx) || defined(STM32F417xx)
#define __HAL_RCC_CRYP_IS_CLK_ENABLED()   ((RCC->AHB2ENR & (RCC_AHB2ENR_CRYPEN)) != RESET) 
#define __HAL_RCC_HASH_IS_CLK_ENABLED()   ((RCC->AHB2ENR & (RCC_AHB2ENR_HASHEN)) != RESET) 

#define __HAL_RCC_CRYP_IS_CLK_DISABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_CRYPEN)) == RESET) 
#define __HAL_RCC_HASH_IS_CLK_DISABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_HASHEN)) == RESET) 
#endif /* STM32F415xx || STM32F417xx */  
/**
  * @}
  */  
  
/** @defgroup RCCEx_AHB3_Clock_Enable_Disable AHB3 Peripheral Clock Enable Disable
  * @brief  Enables or disables the AHB3 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{  
  */
#define __HAL_RCC_FSMC_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB3ENR, RCC_AHB3ENR_FSMCEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB3ENR, RCC_AHB3ENR_FSMCEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_FSMC_CLK_DISABLE() (RCC->AHB3ENR &= ~(RCC_AHB3ENR_FSMCEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Peripheral_Clock_Enable_Disable_Status AHB3 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB3 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_FSMC_IS_CLK_ENABLED()   ((RCC->AHB3ENR & (RCC_AHB3ENR_FSMCEN)) != RESET) 
#define __HAL_RCC_FSMC_IS_CLK_DISABLED()  ((RCC->AHB3ENR & (RCC_AHB3ENR_FSMCEN)) == RESET) 
/**
  * @}
  */   
   
/** @defgroup RCCEx_APB1_Clock_Enable_Disable APB1 Peripheral Clock Enable Disable
  * @brief  Enable or disable the Low Speed APB (APB1) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{  
  */
#define __HAL_RCC_TIM6_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM7_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM7EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM7EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM12_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM12EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM12EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM13_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM13EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM13EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM14_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_USART3_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_USART3EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_USART3EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART4_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART4EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART4EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART5_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART5EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART5EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CAN1_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CAN2_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN2EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN2EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_DAC_CLK_ENABLE()    do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM2_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_I2C3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM2_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM2EN))
#define __HAL_RCC_TIM3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM3EN))
#define __HAL_RCC_TIM4_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM4EN))
#define __HAL_RCC_SPI3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_SPI3EN))
#define __HAL_RCC_I2C3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_I2C3EN))
#define __HAL_RCC_TIM6_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM6EN))
#define __HAL_RCC_TIM7_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM7EN))
#define __HAL_RCC_TIM12_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM12EN))
#define __HAL_RCC_TIM13_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM13EN))
#define __HAL_RCC_TIM14_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM14EN))
#define __HAL_RCC_USART3_CLK_DISABLE() (RCC->APB1ENR &= ~(RCC_APB1ENR_USART3EN))
#define __HAL_RCC_UART4_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_UART4EN))
#define __HAL_RCC_UART5_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_UART5EN))
#define __HAL_RCC_CAN1_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_CAN1EN))
#define __HAL_RCC_CAN2_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_CAN2EN))
#define __HAL_RCC_DAC_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_DACEN))
/**
  * @}
  */
 
/** @defgroup RCCEx_APB1_Peripheral_Clock_Enable_Disable_Status APB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */ 
#define __HAL_RCC_TIM2_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) != RESET)  
#define __HAL_RCC_TIM3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) != RESET) 
#define __HAL_RCC_TIM4_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) != RESET)
#define __HAL_RCC_SPI3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) != RESET) 
#define __HAL_RCC_I2C3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) != RESET) 
#define __HAL_RCC_TIM6_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) != RESET) 
#define __HAL_RCC_TIM7_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM7EN)) != RESET) 
#define __HAL_RCC_TIM12_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM12EN)) != RESET) 
#define __HAL_RCC_TIM13_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM13EN)) != RESET) 
#define __HAL_RCC_TIM14_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM14EN)) != RESET) 
#define __HAL_RCC_USART3_IS_CLK_ENABLED() ((RCC->APB1ENR & (RCC_APB1ENR_USART3EN)) != RESET) 
#define __HAL_RCC_UART4_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART4EN)) != RESET) 
#define __HAL_RCC_UART5_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART5EN)) != RESET) 
#define __HAL_RCC_CAN1_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN1EN)) != RESET) 
#define __HAL_RCC_CAN2_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN2EN)) != RESET) 
#define __HAL_RCC_DAC_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) != RESET) 

#define __HAL_RCC_TIM2_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) == RESET)  
#define __HAL_RCC_TIM3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) == RESET) 
#define __HAL_RCC_TIM4_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) == RESET)
#define __HAL_RCC_SPI3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) == RESET) 
#define __HAL_RCC_I2C3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) == RESET) 
#define __HAL_RCC_TIM6_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) == RESET) 
#define __HAL_RCC_TIM7_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM7EN)) == RESET) 
#define __HAL_RCC_TIM12_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM12EN)) == RESET) 
#define __HAL_RCC_TIM13_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM13EN)) == RESET) 
#define __HAL_RCC_TIM14_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_TIM14EN)) == RESET) 
#define __HAL_RCC_USART3_IS_CLK_DISABLED() ((RCC->APB1ENR & (RCC_APB1ENR_USART3EN)) == RESET) 
#define __HAL_RCC_UART4_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART4EN)) == RESET) 
#define __HAL_RCC_UART5_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART5EN)) == RESET) 
#define __HAL_RCC_CAN1_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN1EN)) == RESET) 
#define __HAL_RCC_CAN2_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN2EN)) == RESET) 
#define __HAL_RCC_DAC_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) == RESET) 
  /**
  * @}
  */
  
/** @defgroup RCCEx_APB2_Clock_Enable_Disable APB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the High Speed APB (APB2) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */ 
#define __HAL_RCC_TIM8_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM8EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM8EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_ADC2_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC2EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC2EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_ADC3_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC3EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC3EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SDIO_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM10_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_SDIO_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SDIOEN))
#define __HAL_RCC_SPI4_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI4EN))
#define __HAL_RCC_TIM10_CLK_DISABLE()  (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM10EN))
#define __HAL_RCC_TIM8_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM8EN))
#define __HAL_RCC_ADC2_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_ADC2EN))
#define __HAL_RCC_ADC3_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_ADC3EN))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Peripheral_Clock_Enable_Disable_Status APB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_SDIO_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) != RESET)  
#define __HAL_RCC_SPI4_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) != RESET)  
#define __HAL_RCC_TIM10_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) != RESET) 
#define __HAL_RCC_TIM8_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_TIM8EN)) != RESET) 
#define __HAL_RCC_ADC2_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_ADC2EN)) != RESET) 
#define __HAL_RCC_ADC3_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_ADC3EN)) != RESET)
  
#define __HAL_RCC_SDIO_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) == RESET)  
#define __HAL_RCC_SPI4_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) == RESET)  
#define __HAL_RCC_TIM10_IS_CLK_DISABLED()  ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) == RESET) 
#define __HAL_RCC_TIM8_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_TIM8EN)) == RESET) 
#define __HAL_RCC_ADC2_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_ADC2EN)) == RESET) 
#define __HAL_RCC_ADC3_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_ADC3EN)) == RESET)
/**
  * @}
  */
    
/** @defgroup RCCEx_AHB1_Force_Release_Reset AHB1 Force Release Reset 
  * @brief  Force or release AHB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_GPIOD_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_GPIOF_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOFRST))
#define __HAL_RCC_GPIOG_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOGRST))
#define __HAL_RCC_GPIOI_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOIRST))
#define __HAL_RCC_ETHMAC_FORCE_RESET()   (RCC->AHB1RSTR |= (RCC_AHB1RSTR_ETHMACRST))
#define __HAL_RCC_USB_OTG_HS_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_OTGHRST))
#define __HAL_RCC_CRC_FORCE_RESET()     (RCC->AHB1RSTR |= (RCC_AHB1RSTR_CRCRST))

#define __HAL_RCC_GPIOD_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_GPIOF_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOFRST))
#define __HAL_RCC_GPIOG_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOGRST))
#define __HAL_RCC_GPIOI_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOIRST))
#define __HAL_RCC_ETHMAC_RELEASE_RESET() (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_ETHMACRST))
#define __HAL_RCC_USB_OTG_HS_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_OTGHRST))
#define __HAL_RCC_CRC_RELEASE_RESET()    (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_CRCRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Force_Release_Reset AHB2 Force Release Reset 
  * @brief  Force or release AHB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_AHB2_FORCE_RESET()         (RCC->AHB2RSTR = 0xFFFFFFFFU) 
#define __HAL_RCC_AHB2_RELEASE_RESET()       (RCC->AHB2RSTR = 0x00U)

#if defined(STM32F407xx)|| defined(STM32F417xx)  
#define __HAL_RCC_DCMI_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_DCMIRST))
#define __HAL_RCC_DCMI_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_DCMIRST))
#endif /* STM32F407xx || STM32F417xx */

#if defined(STM32F415xx) || defined(STM32F417xx) 
#define __HAL_RCC_CRYP_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_CRYPRST))
#define __HAL_RCC_HASH_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_HASHRST))

#define __HAL_RCC_CRYP_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_CRYPRST))
#define __HAL_RCC_HASH_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_HASHRST))
#endif /* STM32F415xx || STM32F417xx */
   
#define __HAL_RCC_USB_OTG_FS_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_OTGFSRST))
#define __HAL_RCC_USB_OTG_FS_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_OTGFSRST))

#define __HAL_RCC_RNG_FORCE_RESET()    (RCC->AHB2RSTR |= (RCC_AHB2RSTR_RNGRST))
#define __HAL_RCC_RNG_RELEASE_RESET()  (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_RNGRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Force_Release_Reset AHB3 Force Release Reset 
  * @brief  Force or release AHB3 peripheral reset.
  * @{
  */ 
#define __HAL_RCC_AHB3_FORCE_RESET() (RCC->AHB3RSTR = 0xFFFFFFFFU)
#define __HAL_RCC_AHB3_RELEASE_RESET() (RCC->AHB3RSTR = 0x00U) 

#define __HAL_RCC_FSMC_FORCE_RESET()   (RCC->AHB3RSTR |= (RCC_AHB3RSTR_FSMCRST))
#define __HAL_RCC_FSMC_RELEASE_RESET() (RCC->AHB3RSTR &= ~(RCC_AHB3RSTR_FSMCRST))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Force_Release_Reset APB1 Force Release Reset 
  * @brief  Force or release APB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_TIM6_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_TIM7_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM7RST))
#define __HAL_RCC_TIM12_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM12RST))
#define __HAL_RCC_TIM13_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM13RST))
#define __HAL_RCC_TIM14_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM14RST))
#define __HAL_RCC_USART3_FORCE_RESET()   (RCC->APB1RSTR |= (RCC_APB1RSTR_USART3RST))
#define __HAL_RCC_UART4_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART4RST))
#define __HAL_RCC_UART5_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART5RST))
#define __HAL_RCC_CAN1_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_CAN1RST))
#define __HAL_RCC_CAN2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_CAN2RST))
#define __HAL_RCC_DAC_FORCE_RESET()      (RCC->APB1RSTR |= (RCC_APB1RSTR_DACRST))
#define __HAL_RCC_TIM2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_I2C3RST))

#define __HAL_RCC_TIM2_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_I2C3RST))
#define __HAL_RCC_TIM6_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_TIM7_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM7RST))
#define __HAL_RCC_TIM12_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM12RST))
#define __HAL_RCC_TIM13_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM13RST))
#define __HAL_RCC_TIM14_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM14RST))
#define __HAL_RCC_USART3_RELEASE_RESET() (RCC->APB1RSTR &= ~(RCC_APB1RSTR_USART3RST))
#define __HAL_RCC_UART4_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART4RST))
#define __HAL_RCC_UART5_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART5RST))
#define __HAL_RCC_CAN1_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CAN1RST))
#define __HAL_RCC_CAN2_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CAN2RST))
#define __HAL_RCC_DAC_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_DACRST))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Force_Release_Reset APB2 Force Release Reset 
  * @brief  Force or release APB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_TIM8_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM8RST))
#define __HAL_RCC_SDIO_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_FORCE_RESET()  (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM10RST))
                                          
#define __HAL_RCC_SDIO_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_RELEASE_RESET()(RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM10RST))
#define __HAL_RCC_TIM8_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM8RST))
/**
  * @}
  */
                                        
/** @defgroup RCCEx_AHB1_LowPower_Enable_Disable AHB1 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_GPIOD_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_GPIOF_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOFLPEN))
#define __HAL_RCC_GPIOG_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOGLPEN))
#define __HAL_RCC_GPIOI_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOILPEN))
#define __HAL_RCC_SRAM2_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM2LPEN))
#define __HAL_RCC_ETHMAC_CLK_SLEEP_ENABLE()     (RCC->AHB1LPENR |= (RCC_AHB1LPENR_ETHMACLPEN))
#define __HAL_RCC_ETHMACTX_CLK_SLEEP_ENABLE()   (RCC->AHB1LPENR |= (RCC_AHB1LPENR_ETHMACTXLPEN))
#define __HAL_RCC_ETHMACRX_CLK_SLEEP_ENABLE()   (RCC->AHB1LPENR |= (RCC_AHB1LPENR_ETHMACRXLPEN))
#define __HAL_RCC_ETHMACPTP_CLK_SLEEP_ENABLE()  (RCC->AHB1LPENR |= (RCC_AHB1LPENR_ETHMACPTPLPEN))
#define __HAL_RCC_USB_OTG_HS_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_OTGHSLPEN))
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_SLEEP_ENABLE()  (RCC->AHB1LPENR |= (RCC_AHB1LPENR_OTGHSULPILPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM1LPEN))
#define __HAL_RCC_BKPSRAM_CLK_SLEEP_ENABLE()  (RCC->AHB1LPENR |= (RCC_AHB1LPENR_BKPSRAMLPEN))

#define __HAL_RCC_GPIOD_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_GPIOF_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOFLPEN))
#define __HAL_RCC_GPIOG_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOGLPEN))
#define __HAL_RCC_GPIOI_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOILPEN))
#define __HAL_RCC_SRAM2_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM2LPEN))
#define __HAL_RCC_ETHMAC_CLK_SLEEP_DISABLE()    (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_ETHMACLPEN))
#define __HAL_RCC_ETHMACTX_CLK_SLEEP_DISABLE()  (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_ETHMACTXLPEN))
#define __HAL_RCC_ETHMACRX_CLK_SLEEP_DISABLE()  (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_ETHMACRXLPEN))
#define __HAL_RCC_ETHMACPTP_CLK_SLEEP_DISABLE() (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_ETHMACPTPLPEN))
#define __HAL_RCC_USB_OTG_HS_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_OTGHSLPEN))
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_SLEEP_DISABLE() (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_OTGHSULPILPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_DISABLE()       (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM1LPEN))
#define __HAL_RCC_BKPSRAM_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_BKPSRAMLPEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_LowPower_Enable_Disable AHB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wake-up from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_OTGFSLPEN))
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_OTGFSLPEN))

#define __HAL_RCC_RNG_CLK_SLEEP_ENABLE()   (RCC->AHB2LPENR |= (RCC_AHB2LPENR_RNGLPEN))
#define __HAL_RCC_RNG_CLK_SLEEP_DISABLE()  (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_RNGLPEN))

#if defined(STM32F407xx)|| defined(STM32F417xx) 
#define __HAL_RCC_DCMI_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_DCMILPEN))
#define __HAL_RCC_DCMI_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_DCMILPEN))
#endif /* STM32F407xx || STM32F417xx */

#if defined(STM32F415xx) || defined(STM32F417xx) 
#define __HAL_RCC_CRYP_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_CRYPLPEN))
#define __HAL_RCC_HASH_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_HASHLPEN))

#define __HAL_RCC_CRYP_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_CRYPLPEN))
#define __HAL_RCC_HASH_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_HASHLPEN))
#endif /* STM32F415xx || STM32F417xx */
/**
  * @}
  */
                                        
/** @defgroup RCCEx_AHB3_LowPower_Enable_Disable AHB3 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB3 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_FSMC_CLK_SLEEP_ENABLE()  (RCC->AHB3LPENR |= (RCC_AHB3LPENR_FSMCLPEN))
#define __HAL_RCC_FSMC_CLK_SLEEP_DISABLE() (RCC->AHB3LPENR &= ~(RCC_AHB3LPENR_FSMCLPEN))
/**
  * @}
  */
                                        
/** @defgroup RCCEx_APB1_LowPower_Enable_Disable APB1 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_TIM6_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_TIM7_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM7LPEN))
#define __HAL_RCC_TIM12_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM12LPEN))
#define __HAL_RCC_TIM13_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM13LPEN))
#define __HAL_RCC_TIM14_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM14LPEN))
#define __HAL_RCC_USART3_CLK_SLEEP_ENABLE()  (RCC->APB1LPENR |= (RCC_APB1LPENR_USART3LPEN))
#define __HAL_RCC_UART4_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART4LPEN))
#define __HAL_RCC_UART5_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART5LPEN))
#define __HAL_RCC_CAN1_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_CAN1LPEN))
#define __HAL_RCC_CAN2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_CAN2LPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_ENABLE()     (RCC->APB1LPENR |= (RCC_APB1LPENR_DACLPEN))
#define __HAL_RCC_TIM2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_I2C3LPEN))

#define __HAL_RCC_TIM2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_I2C3LPEN))
#define __HAL_RCC_TIM6_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_TIM7_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM7LPEN))
#define __HAL_RCC_TIM12_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM12LPEN))
#define __HAL_RCC_TIM13_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM13LPEN))
#define __HAL_RCC_TIM14_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM14LPEN))
#define __HAL_RCC_USART3_CLK_SLEEP_DISABLE() (RCC->APB1LPENR &= ~(RCC_APB1LPENR_USART3LPEN))
#define __HAL_RCC_UART4_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART4LPEN))
#define __HAL_RCC_UART5_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART5LPEN))
#define __HAL_RCC_CAN1_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CAN1LPEN))
#define __HAL_RCC_CAN2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CAN2LPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_DISABLE()    (RCC->APB1LPENR &= ~(RCC_APB1LPENR_DACLPEN))
/**
  * @}
  */
                                        
/** @defgroup RCCEx_APB2_LowPower_Enable_Disable APB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_TIM8_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_TIM8LPEN))
#define __HAL_RCC_ADC2_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_ADC2LPEN))
#define __HAL_RCC_ADC3_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_ADC3LPEN))
#define __HAL_RCC_SDIO_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_ENABLE()(RCC->APB2LPENR |= (RCC_APB2LPENR_TIM10LPEN))

#define __HAL_RCC_SDIO_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_DISABLE()(RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM10LPEN))
#define __HAL_RCC_TIM8_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM8LPEN))
#define __HAL_RCC_ADC2_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_ADC2LPEN))
#define __HAL_RCC_ADC3_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_ADC3LPEN))
/**
  * @}
  */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx */
/*----------------------------------------------------------------------------*/

/*------------------------- STM32F401xE/STM32F401xC --------------------------*/
#if defined(STM32F401xC) || defined(STM32F401xE)
/** @defgroup RCCEx_AHB1_Clock_Enable_Disable AHB1 Peripheral Clock Enable Disable
  * @brief  Enable or disable the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.   
  * @{
  */
#define __HAL_RCC_GPIOD_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_GPIOE_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CRC_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CCMDATARAMEN_CLK_ENABLE()  do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_GPIOD_CLK_DISABLE()        (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIODEN))
#define __HAL_RCC_GPIOE_CLK_DISABLE()        (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOEEN))
#define __HAL_RCC_CRC_CLK_DISABLE()          (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CRCEN))
#define __HAL_RCC_CCMDATARAMEN_CLK_DISABLE() (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CCMDATARAMEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_Peripheral_Clock_Enable_Disable_Status AHB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_GPIOD_IS_CLK_ENABLED()        ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) != RESET) 
#define __HAL_RCC_GPIOE_IS_CLK_ENABLED()        ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) != RESET)  
#define __HAL_RCC_CRC_IS_CLK_ENABLED()          ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) != RESET)  
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_ENABLED() ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN)) != RESET)  

#define __HAL_RCC_GPIOD_IS_CLK_DISABLED()        ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) == RESET) 
#define __HAL_RCC_GPIOE_IS_CLK_DISABLED()        ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) == RESET)  
#define __HAL_RCC_CRC_IS_CLK_DISABLED()          ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) == RESET)  
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_DISABLED() ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN)) == RESET)  
/**
  * @}
  */ 
  
/** @defgroup RCCEx_AHB2_Clock_Enable_Disable AHB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_CLK_ENABLE()  do {(RCC->AHB2ENR |= (RCC_AHB2ENR_OTGFSEN));\
                                               __HAL_RCC_SYSCFG_CLK_ENABLE();\
                                              }while(0U)
                                        
#define __HAL_RCC_USB_OTG_FS_CLK_DISABLE() (RCC->AHB2ENR &= ~(RCC_AHB2ENR_OTGFSEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Peripheral_Clock_Enable_Disable_Status AHB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_IS_CLK_ENABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) != RESET)
#define __HAL_RCC_USB_OTG_FS_IS_CLK_DISABLED() ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) == RESET)
/**
  * @}
  */  
  
/** @defgroup RCC_APB1_Clock_Enable_Disable APB1 Peripheral Clock Enable Disable
  * @brief  Enable or disable the Low Speed APB (APB1) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM2_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_I2C3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM2_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM2EN))
#define __HAL_RCC_TIM3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM3EN))
#define __HAL_RCC_TIM4_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM4EN))
#define __HAL_RCC_SPI3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_SPI3EN))
#define __HAL_RCC_I2C3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_I2C3EN))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Peripheral_Clock_Enable_Disable_Status APB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM2_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) != RESET) 
#define __HAL_RCC_TIM3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) != RESET) 
#define __HAL_RCC_TIM4_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) != RESET) 
#define __HAL_RCC_SPI3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) != RESET) 
#define __HAL_RCC_I2C3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) != RESET)

#define __HAL_RCC_TIM2_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) == RESET) 
#define __HAL_RCC_TIM3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) == RESET) 
#define __HAL_RCC_TIM4_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) == RESET) 
#define __HAL_RCC_SPI3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) == RESET) 
#define __HAL_RCC_I2C3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) == RESET)
/**
  * @}
  */ 
  
/** @defgroup RCCEx_APB2_Clock_Enable_Disable APB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the High Speed APB (APB2) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_SDIO_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM10_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_SDIO_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SDIOEN))
#define __HAL_RCC_SPI4_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI4EN))
#define __HAL_RCC_TIM10_CLK_DISABLE()  (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM10EN))
/**
  * @}
  */
  
/** @defgroup RCCEx_APB2_Peripheral_Clock_Enable_Disable_Status APB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_SDIO_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) != RESET)  
#define __HAL_RCC_SPI4_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) != RESET)   
#define __HAL_RCC_TIM10_IS_CLK_ENABLED()  ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) != RESET)  

#define __HAL_RCC_SDIO_IS_CLK_DISABLED()  ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) == RESET)
#define __HAL_RCC_SPI4_IS_CLK_DISABLED()  ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) == RESET) 
#define __HAL_RCC_TIM10_IS_CLK_DISABLED() ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) == RESET) 
/**
  * @}
  */
/** @defgroup RCCEx_AHB1_Force_Release_Reset AHB1 Force Release Reset 
  * @brief  Force or release AHB1 peripheral reset.
  * @{
  */  
#define __HAL_RCC_AHB1_FORCE_RESET()    (RCC->AHB1RSTR = 0xFFFFFFFFU)
#define __HAL_RCC_GPIOD_FORCE_RESET()   (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_FORCE_RESET()   (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_CRC_FORCE_RESET()     (RCC->AHB1RSTR |= (RCC_AHB1RSTR_CRCRST))

#define __HAL_RCC_AHB1_RELEASE_RESET()  (RCC->AHB1RSTR = 0x00U)
#define __HAL_RCC_GPIOD_RELEASE_RESET() (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_RELEASE_RESET() (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_CRC_RELEASE_RESET()   (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_CRCRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Force_Release_Reset AHB2 Force Release Reset 
  * @brief  Force or release AHB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_AHB2_FORCE_RESET()    (RCC->AHB2RSTR = 0xFFFFFFFFU) 
#define __HAL_RCC_USB_OTG_FS_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_OTGFSRST))

#define __HAL_RCC_AHB2_RELEASE_RESET()  (RCC->AHB2RSTR = 0x00U)
#define __HAL_RCC_USB_OTG_FS_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_OTGFSRST))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Force_Release_Reset APB1 Force Release Reset 
  * @brief  Force or release APB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_APB1_FORCE_RESET()     (RCC->APB1RSTR = 0xFFFFFFFFU)  
#define __HAL_RCC_TIM2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_I2C3RST))

#define __HAL_RCC_APB1_RELEASE_RESET()   (RCC->APB1RSTR = 0x00U) 
#define __HAL_RCC_TIM2_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_I2C3RST))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Force_Release_Reset APB2 Force Release Reset 
  * @brief  Force or release APB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_APB2_FORCE_RESET()     (RCC->APB2RSTR = 0xFFFFFFFFU)  
#define __HAL_RCC_SDIO_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_FORCE_RESET()    (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM10RST))

#define __HAL_RCC_APB2_RELEASE_RESET()   (RCC->APB2RSTR = 0x00U)
#define __HAL_RCC_SDIO_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_RELEASE_RESET()  (RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM10RST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Force_Release_Reset AHB3 Force Release Reset 
  * @brief  Force or release AHB3 peripheral reset.
  * @{
  */ 
#define __HAL_RCC_AHB3_FORCE_RESET() (RCC->AHB3RSTR = 0xFFFFFFFFU)
#define __HAL_RCC_AHB3_RELEASE_RESET() (RCC->AHB3RSTR = 0x00U) 
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_LowPower_Enable_Disable AHB1 Peripheral Low Power Enable Disable 
  * @brief  Enable or disable the AHB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wake-up from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_GPIOD_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM1LPEN))

#define __HAL_RCC_GPIOD_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM1LPEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_LowPower_Enable_Disable AHB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wake-up from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_OTGFSLPEN))

#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_DISABLE()   (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_OTGFSLPEN))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_LowPower_Enable_Disable APB1 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wake-up from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_TIM2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_I2C3LPEN))

#define __HAL_RCC_TIM2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_I2C3LPEN))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_LowPower_Enable_Disable APB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wake-up from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_SDIO_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_ENABLE()   (RCC->APB2LPENR |= (RCC_APB2LPENR_TIM10LPEN))

#define __HAL_RCC_SDIO_CLK_SLEEP_DISABLE()   (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_DISABLE()   (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_DISABLE()  (RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM10LPEN))
/**
  * @}
  */
#endif /* STM32F401xC || STM32F401xE*/
/*----------------------------------------------------------------------------*/

/*-------------------------------- STM32F410xx -------------------------------*/
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
/** @defgroup RCCEx_AHB1_Clock_Enable_Disable AHB1 Peripheral Clock Enable Disable     
  * @brief  Enables or disables the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_CRC_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_RNG_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_RNGEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_RNGEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_CRC_CLK_DISABLE()     (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CRCEN))
#define __HAL_RCC_RNG_CLK_DISABLE()     (RCC->AHB1ENR &= ~(RCC_AHB1ENR_RNGEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_Peripheral_Clock_Enable_Disable_Status AHB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_CRC_IS_CLK_ENABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) != RESET)
#define __HAL_RCC_RNG_IS_CLK_ENABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_RNGEN)) != RESET)
      
#define __HAL_RCC_CRC_IS_CLK_DISABLED()    ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) == RESET)
#define __HAL_RCC_RNG_IS_CLK_DISABLED()    ((RCC->AHB1ENR & (RCC_AHB1ENR_RNGEN)) == RESET)
/**
  * @}
  */
  
/** @defgroup RCCEx_APB1_Clock_Enable_Disable APB1 Peripheral Clock Enable Disable  
  * @brief  Enable or disable the High Speed APB (APB1) peripheral clock.
  * @{
  */
#define __HAL_RCC_TIM6_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_LPTIM1_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_LPTIM1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_LPTIM1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_RTCAPB_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_RTCAPBEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_RTCAPBEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_FMPI2C1_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_FMPI2C1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_FMPI2C1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U) 
#define __HAL_RCC_DAC_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
                                        
#define __HAL_RCC_TIM6_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM6EN))
#define __HAL_RCC_RTCAPB_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_RTCAPBEN))
#define __HAL_RCC_LPTIM1_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_LPTIM1EN))
#define __HAL_RCC_FMPI2C1_CLK_DISABLE() (RCC->APB1ENR &= ~(RCC_APB1ENR_FMPI2C1EN))
#define __HAL_RCC_DAC_CLK_DISABLE()     (RCC->APB1ENR &= ~(RCC_APB1ENR_DACEN))
/**
  * @}
  */
  
/** @defgroup RCCEx_APB1_Peripheral_Clock_Enable_Disable_Status APB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */  
#define __HAL_RCC_TIM6_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) != RESET) 
#define __HAL_RCC_RTCAPB_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_RTCAPBEN)) != RESET)
#define __HAL_RCC_LPTIM1_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_LPTIM1EN)) != RESET) 
#define __HAL_RCC_FMPI2C1_IS_CLK_ENABLED() ((RCC->APB1ENR & (RCC_APB1ENR_FMPI2C1EN)) != RESET)  
#define __HAL_RCC_DAC_IS_CLK_ENABLED()     ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) != RESET) 

#define __HAL_RCC_TIM6_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) == RESET) 
#define __HAL_RCC_RTCAPB_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_RTCAPBEN)) == RESET)
#define __HAL_RCC_LPTIM1_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_LPTIM1EN)) == RESET) 
#define __HAL_RCC_FMPI2C1_IS_CLK_DISABLED() ((RCC->APB1ENR & (RCC_APB1ENR_FMPI2C1EN)) == RESET)  
#define __HAL_RCC_DAC_IS_CLK_DISABLED()     ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) == RESET)  
/**
  * @}
  */
  
/** @defgroup RCCEx_APB2_Clock_Enable_Disable APB2 Peripheral Clock Enable Disable  
  * @brief  Enable or disable the High Speed APB (APB2) peripheral clock.
  * @{
  */  
#define __HAL_RCC_SPI5_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI5EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI5EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_EXTIT_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_EXTITEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_EXTITEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI5_CLK_DISABLE()    (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI5EN))
#define __HAL_RCC_EXTIT_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_EXTITEN))
/**
  * @}
  */
  
/** @defgroup RCCEx_APB2_Peripheral_Clock_Enable_Disable_Status APB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_SPI5_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_SPI5EN)) != RESET)  
#define __HAL_RCC_EXTIT_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_EXTITEN)) != RESET)  
  
#define __HAL_RCC_SPI5_IS_CLK_DISABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_SPI5EN)) == RESET)  
#define __HAL_RCC_EXTIT_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_EXTITEN)) == RESET)  
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB1_Force_Release_Reset AHB1 Force Release Reset 
  * @brief  Force or release AHB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_CRC_FORCE_RESET()     (RCC->AHB1RSTR |= (RCC_AHB1RSTR_CRCRST))
#define __HAL_RCC_RNG_FORCE_RESET()     (RCC->AHB1RSTR |= (RCC_AHB1RSTR_RNGRST))
#define __HAL_RCC_CRC_RELEASE_RESET()   (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_CRCRST))
#define __HAL_RCC_RNG_RELEASE_RESET()   (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_RNGRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Force_Release_Reset AHB2 Force Release Reset 
  * @brief  Force or release AHB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_AHB2_FORCE_RESET()
#define __HAL_RCC_AHB2_RELEASE_RESET()
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Force_Release_Reset AHB3 Force Release Reset 
  * @brief  Force or release AHB3 peripheral reset.
  * @{
  */ 
#define __HAL_RCC_AHB3_FORCE_RESET()
#define __HAL_RCC_AHB3_RELEASE_RESET()
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Force_Release_Reset APB1 Force Release Reset 
  * @brief  Force or release APB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_TIM6_FORCE_RESET()      (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_LPTIM1_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_LPTIM1RST))
#define __HAL_RCC_FMPI2C1_FORCE_RESET()   (RCC->APB1RSTR |= (RCC_APB1RSTR_FMPI2C1RST))
#define __HAL_RCC_DAC_FORCE_RESET()       (RCC->APB1RSTR |= (RCC_APB1RSTR_DACRST))

#define __HAL_RCC_TIM6_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_LPTIM1_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_LPTIM1RST))
#define __HAL_RCC_FMPI2C1_RELEASE_RESET() (RCC->APB1RSTR &= ~(RCC_APB1RSTR_FMPI2C1RST))
#define __HAL_RCC_DAC_RELEASE_RESET()     (RCC->APB1RSTR &= ~(RCC_APB1RSTR_DACRST))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Force_Release_Reset APB2 Force Release Reset 
  * @brief  Force or release APB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_SPI5_FORCE_RESET()      (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI5RST))
#define __HAL_RCC_SPI5_RELEASE_RESET()    (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI5RST))                                        
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_LowPower_Enable_Disable AHB1 Peripheral Low Power Enable Disable  
  * @brief  Enable or disable the AHB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_RNG_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_RNGLPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM1LPEN))

#define __HAL_RCC_RNG_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_RNGLPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM1LPEN))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_LowPower_Enable_Disable APB1 Peripheral Low Power Enable Disable                                         
  * @brief  Enable or disable the APB1 peripheral clock during Low Power (Sleep) mode.
  * @{
  */
#define __HAL_RCC_TIM6_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_LPTIM1_CLK_SLEEP_ENABLE()  (RCC->APB1LPENR |= (RCC_APB1LPENR_LPTIM1LPEN))
#define __HAL_RCC_RTCAPB_CLK_SLEEP_ENABLE()  (RCC->APB1LPENR |= (RCC_APB1LPENR_RTCAPBLPEN))
#define __HAL_RCC_FMPI2C1_CLK_SLEEP_ENABLE() (RCC->APB1LPENR |= (RCC_APB1LPENR_FMPI2C1LPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_ENABLE()     (RCC->APB1LPENR |= (RCC_APB1LPENR_DACLPEN))

#define __HAL_RCC_TIM6_CLK_SLEEP_DISABLE()    (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_LPTIM1_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_LPTIM1LPEN))
#define __HAL_RCC_RTCAPB_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_RTCAPBLPEN))
#define __HAL_RCC_FMPI2C1_CLK_SLEEP_DISABLE() (RCC->APB1LPENR &= ~(RCC_APB1LPENR_FMPI2C1LPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_DISABLE()     (RCC->APB1LPENR &= ~(RCC_APB1LPENR_DACLPEN))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_LowPower_Enable_Disable APB2 Peripheral Low Power Enable Disable                                         
  * @brief  Enable or disable the APB2 peripheral clock during Low Power (Sleep) mode.
  * @{
  */
#define __HAL_RCC_SPI5_CLK_SLEEP_ENABLE()     (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI5LPEN))
#define __HAL_RCC_EXTIT_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_EXTITLPEN))                                
#define __HAL_RCC_SPI5_CLK_SLEEP_DISABLE()    (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI5LPEN))                                        
#define __HAL_RCC_EXTIT_CLK_SLEEP_DISABLE()   (RCC->APB2LPENR &= ~(RCC_APB2LPENR_EXTITLPEN))
/**
  * @}
  */

#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */
/*----------------------------------------------------------------------------*/

/*-------------------------------- STM32F411xx -------------------------------*/
#if defined(STM32F411xE)
/** @defgroup RCCEx_AHB1_Clock_Enable_Disable AHB1 Peripheral Clock Enable Disable
  * @brief  Enables or disables the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_CCMDATARAMEN_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOD_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOE_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_CRC_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOD_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIODEN))
#define __HAL_RCC_GPIOE_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOEEN))
#define __HAL_RCC_CCMDATARAMEN_CLK_DISABLE()    (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CCMDATARAMEN))
#define __HAL_RCC_CRC_CLK_DISABLE()             (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CRCEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_Peripheral_Clock_Enable_Disable_Status AHB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */  
#define __HAL_RCC_GPIOD_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) != RESET) 
#define __HAL_RCC_GPIOE_IS_CLK_ENABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) != RESET) 
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_ENABLED()    ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN)) != RESET) 
#define __HAL_RCC_CRC_IS_CLK_ENABLED()             ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) != RESET) 

#define __HAL_RCC_GPIOD_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) == RESET) 
#define __HAL_RCC_GPIOE_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) == RESET) 
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_DISABLED()    ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN)) == RESET) 
#define __HAL_RCC_CRC_IS_CLK_DISABLED()             ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) == RESET) 
/**
  * @}
  */
  
/** @defgroup RCCEX_AHB2_Clock_Enable_Disable AHB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_CLK_ENABLE()  do {(RCC->AHB2ENR |= (RCC_AHB2ENR_OTGFSEN));\
                                               __HAL_RCC_SYSCFG_CLK_ENABLE();\
                                              }while(0U)

#define __HAL_RCC_USB_OTG_FS_CLK_DISABLE() (RCC->AHB2ENR &= ~(RCC_AHB2ENR_OTGFSEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Peripheral_Clock_Enable_Disable_Status AHB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_IS_CLK_ENABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) != RESET)
#define __HAL_RCC_USB_OTG_FS_IS_CLK_DISABLED() ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) == RESET)
/**
  * @}
  */  

/** @defgroup RCCEx_APB1_Clock_Enable_Disable APB1 Peripheral Clock Enable Disable
  * @brief  Enable or disable the Low Speed APB (APB1) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it. 
  * @{
  */
#define __HAL_RCC_TIM2_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_TIM3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_TIM4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_SPI3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_I2C3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_TIM2_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM2EN))
#define __HAL_RCC_TIM3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM3EN))
#define __HAL_RCC_TIM4_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM4EN))
#define __HAL_RCC_SPI3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_SPI3EN))
#define __HAL_RCC_I2C3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_I2C3EN))
/**
  * @}
  */ 
  
/** @defgroup RCCEx_APB1_Peripheral_Clock_Enable_Disable_Status APB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM2_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) != RESET) 
#define __HAL_RCC_TIM3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) != RESET) 
#define __HAL_RCC_TIM4_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) != RESET) 
#define __HAL_RCC_SPI3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) != RESET) 
#define __HAL_RCC_I2C3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) != RESET)

#define __HAL_RCC_TIM2_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) == RESET) 
#define __HAL_RCC_TIM3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) == RESET) 
#define __HAL_RCC_TIM4_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) == RESET) 
#define __HAL_RCC_SPI3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) == RESET) 
#define __HAL_RCC_I2C3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) == RESET) 
/**
  * @}
  */ 
  
/** @defgroup RCCEx_APB2_Clock_Enable_Disable APB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the High Speed APB (APB2) peripheral clock.
  * @{
  */
#define __HAL_RCC_SPI5_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI5EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI5EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SDIO_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM10_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SDIO_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SDIOEN))
#define __HAL_RCC_SPI4_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI4EN))
#define __HAL_RCC_TIM10_CLK_DISABLE()  (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM10EN))
#define __HAL_RCC_SPI5_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI5EN))
/**
  * @}
  */
  
/** @defgroup RCCEx_APB2_Peripheral_Clock_Enable_Disable_Status APB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_SDIO_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) != RESET)  
#define __HAL_RCC_SPI4_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) != RESET)   
#define __HAL_RCC_TIM10_IS_CLK_ENABLED()  ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) != RESET)  
#define __HAL_RCC_SPI5_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI5EN)) != RESET) 

#define __HAL_RCC_SDIO_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) == RESET)  
#define __HAL_RCC_SPI4_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) == RESET)   
#define __HAL_RCC_TIM10_IS_CLK_DISABLED()  ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) == RESET)  
#define __HAL_RCC_SPI5_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI5EN)) == RESET)   
/**
  * @}
  */  
  
/** @defgroup RCCEx_AHB1_Force_Release_Reset AHB1 Force Release Reset 
  * @brief  Force or release AHB1 peripheral reset.
  * @{
  */ 
#define __HAL_RCC_GPIOD_FORCE_RESET()   (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_FORCE_RESET()   (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_CRC_FORCE_RESET()     (RCC->AHB1RSTR |= (RCC_AHB1RSTR_CRCRST))

#define __HAL_RCC_GPIOD_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_CRC_RELEASE_RESET()    (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_CRCRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Force_Release_Reset AHB2 Force Release Reset 
  * @brief  Force or release AHB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_AHB2_FORCE_RESET()    (RCC->AHB2RSTR = 0xFFFFFFFFU) 
#define __HAL_RCC_USB_OTG_FS_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_OTGFSRST))

#define __HAL_RCC_AHB2_RELEASE_RESET()  (RCC->AHB2RSTR = 0x00U)
#define __HAL_RCC_USB_OTG_FS_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_OTGFSRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Force_Release_Reset AHB3 Force Release Reset 
  * @brief  Force or release AHB3 peripheral reset.
  * @{
  */ 
#define __HAL_RCC_AHB3_FORCE_RESET() (RCC->AHB3RSTR = 0xFFFFFFFFU)
#define __HAL_RCC_AHB3_RELEASE_RESET() (RCC->AHB3RSTR = 0x00U) 
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Force_Release_Reset APB1 Force Release Reset 
  * @brief  Force or release APB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_TIM2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_I2C3RST))

#define __HAL_RCC_TIM2_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_I2C3RST))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Force_Release_Reset APB2 Force Release Reset 
  * @brief  Force or release APB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_SPI5_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI5RST))
#define __HAL_RCC_SDIO_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_FORCE_RESET()    (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM10RST))

#define __HAL_RCC_SDIO_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_RELEASE_RESET()  (RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM10RST))
#define __HAL_RCC_SPI5_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI5RST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_LowPower_Enable_Disable AHB1 Peripheral Low Power Enable Disable 
  * @brief  Enable or disable the AHB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_GPIOD_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM1LPEN))
                                        
#define __HAL_RCC_GPIOD_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIODLPEN))                                        
#define __HAL_RCC_GPIOE_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM1LPEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_LowPower_Enable_Disable AHB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wake-up from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_OTGFSLPEN))
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_DISABLE()   (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_OTGFSLPEN))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_LowPower_Enable_Disable APB1 Peripheral Low Power Enable Disable 
  * @brief  Enable or disable the APB1 peripheral clock during Low Power (Sleep) mode.
  * @{
  */
#define __HAL_RCC_TIM2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_I2C3LPEN))

#define __HAL_RCC_TIM2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_I2C3LPEN))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_LowPower_Enable_Disable APB2 Peripheral Low Power Enable Disable 
  * @brief  Enable or disable the APB2 peripheral clock during Low Power (Sleep) mode.
  * @{
  */
#define __HAL_RCC_SPI5_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI5LPEN))
#define __HAL_RCC_SDIO_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_ENABLE()   (RCC->APB2LPENR |= (RCC_APB2LPENR_TIM10LPEN))

#define __HAL_RCC_SDIO_CLK_SLEEP_DISABLE()   (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_DISABLE()   (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_DISABLE()  (RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM10LPEN))
#define __HAL_RCC_SPI5_CLK_SLEEP_DISABLE()   (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI5LPEN))
/**
  * @}
  */
#endif /* STM32F411xE */
/*----------------------------------------------------------------------------*/

/*---------------------------------- STM32F446xx -----------------------------*/
#if defined(STM32F446xx)
/** @defgroup RCCEx_AHB1_Clock_Enable_Disable AHB1 Peripheral Clock Enable Disable
  * @brief  Enables or disables the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_BKPSRAM_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_BKPSRAMEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_BKPSRAMEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_CCMDATARAMEN_CLK_ENABLE() do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CCMDATARAMEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_CRC_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                        UNUSED(tmpreg); \
                                        } while(0U)
#define __HAL_RCC_GPIOD_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_GPIOE_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_GPIOF_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOFEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOFEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_GPIOG_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOGEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOGEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_USB_OTG_HS_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSULPIEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_OTGHSULPIEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_GPIOD_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIODEN))
#define __HAL_RCC_GPIOE_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOEEN))
#define __HAL_RCC_GPIOF_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOFEN))
#define __HAL_RCC_GPIOG_CLK_DISABLE()           (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOGEN))
#define __HAL_RCC_USB_OTG_HS_CLK_DISABLE()      (RCC->AHB1ENR &= ~(RCC_AHB1ENR_OTGHSEN))
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_DISABLE() (RCC->AHB1ENR &= ~(RCC_AHB1ENR_OTGHSULPIEN))
#define __HAL_RCC_BKPSRAM_CLK_DISABLE()         (RCC->AHB1ENR &= ~(RCC_AHB1ENR_BKPSRAMEN))
#define __HAL_RCC_CCMDATARAMEN_CLK_DISABLE()    (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CCMDATARAMEN))
#define __HAL_RCC_CRC_CLK_DISABLE()             (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CRCEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_Peripheral_Clock_Enable_Disable_Status AHB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_GPIOD_IS_CLK_ENABLED()            ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) != RESET) 
#define __HAL_RCC_GPIOE_IS_CLK_ENABLED()            ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) != RESET) 
#define __HAL_RCC_GPIOF_IS_CLK_ENABLED()            ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOFEN)) != RESET) 
#define __HAL_RCC_GPIOG_IS_CLK_ENABLED()            ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOGEN)) != RESET) 
#define __HAL_RCC_USB_OTG_HS_IS_CLK_ENABLED()       ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSEN)) != RESET)  
#define __HAL_RCC_USB_OTG_HS_ULPI_IS_CLK_ENABLED()  ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSULPIEN)) != RESET) 
#define __HAL_RCC_BKPSRAM_IS_CLK_ENABLED()          ((RCC->AHB1ENR & (RCC_AHB1ENR_BKPSRAMEN)) != RESET)  
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_ENABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN))!= RESET)  
#define __HAL_RCC_CRC_IS_CLK_ENABLED()              ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) != RESET) 

#define __HAL_RCC_GPIOD_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) == RESET) 
#define __HAL_RCC_GPIOE_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) == RESET) 
#define __HAL_RCC_GPIOF_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOFEN)) == RESET) 
#define __HAL_RCC_GPIOG_IS_CLK_DISABLED()           ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOGEN)) == RESET) 
#define __HAL_RCC_USB_OTG_HS_IS_CLK_DISABLED()      ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSEN)) == RESET)  
#define __HAL_RCC_USB_OTG_HS_ULPI_IS_CLK_DISABLED() ((RCC->AHB1ENR & (RCC_AHB1ENR_OTGHSULPIEN)) == RESET) 
#define __HAL_RCC_BKPSRAM_IS_CLK_DISABLED()         ((RCC->AHB1ENR & (RCC_AHB1ENR_BKPSRAMEN)) == RESET)  
#define __HAL_RCC_CCMDATARAMEN_IS_CLK_DISABLED()    ((RCC->AHB1ENR & (RCC_AHB1ENR_CCMDATARAMEN)) == RESET)  
#define __HAL_RCC_CRC_IS_CLK_DISABLED()             ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) == RESET) 
/**
  * @}
  */  
  
/** @defgroup RCCEx_AHB2_Clock_Enable_Disable AHB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_DCMI_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_DCMIEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_DCMIEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_DCMI_CLK_DISABLE()  (RCC->AHB2ENR &= ~(RCC_AHB2ENR_DCMIEN))
#define __HAL_RCC_USB_OTG_FS_CLK_ENABLE()  do {(RCC->AHB2ENR |= (RCC_AHB2ENR_OTGFSEN));\
                                               __HAL_RCC_SYSCFG_CLK_ENABLE();\
                                              }while(0U)
                                        
#define __HAL_RCC_USB_OTG_FS_CLK_DISABLE() (RCC->AHB2ENR &= ~(RCC_AHB2ENR_OTGFSEN))

#define __HAL_RCC_RNG_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_RNGEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_RNGEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_RNG_CLK_DISABLE()   (RCC->AHB2ENR &= ~(RCC_AHB2ENR_RNGEN))
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB2_Peripheral_Clock_Enable_Disable_Status AHB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_DCMI_IS_CLK_ENABLED()        ((RCC->AHB2ENR & (RCC_AHB2ENR_DCMIEN)) != RESET)
#define __HAL_RCC_DCMI_IS_CLK_DISABLED()       ((RCC->AHB2ENR & (RCC_AHB2ENR_DCMIEN)) == RESET)

#define __HAL_RCC_USB_OTG_FS_IS_CLK_ENABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) != RESET)
#define __HAL_RCC_USB_OTG_FS_IS_CLK_DISABLED() ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) == RESET)

#define __HAL_RCC_RNG_IS_CLK_ENABLED()    ((RCC->AHB2ENR & (RCC_AHB2ENR_RNGEN)) != RESET) 
#define __HAL_RCC_RNG_IS_CLK_DISABLED()   ((RCC->AHB2ENR & (RCC_AHB2ENR_RNGEN)) == RESET) 
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB3_Clock_Enable_Disable AHB3 Peripheral Clock Enable Disable
  * @brief  Enables or disables the AHB3 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it. 
  * @{
  */
#define __HAL_RCC_FMC_CLK_ENABLE()    do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB3ENR, RCC_AHB3ENR_FMCEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB3ENR, RCC_AHB3ENR_FMCEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_QSPI_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB3ENR, RCC_AHB3ENR_QSPIEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB3ENR, RCC_AHB3ENR_QSPIEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_FMC_CLK_DISABLE()    (RCC->AHB3ENR &= ~(RCC_AHB3ENR_FMCEN))
#define __HAL_RCC_QSPI_CLK_DISABLE()   (RCC->AHB3ENR &= ~(RCC_AHB3ENR_QSPIEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Peripheral_Clock_Enable_Disable_Status AHB3 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB3 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_FMC_IS_CLK_ENABLED()   ((RCC->AHB3ENR & (RCC_AHB3ENR_FMCEN)) != RESET)
#define __HAL_RCC_QSPI_IS_CLK_ENABLED()  ((RCC->AHB3ENR & (RCC_AHB3ENR_QSPIEN)) != RESET)

#define __HAL_RCC_FMC_IS_CLK_DISABLED()  ((RCC->AHB3ENR & (RCC_AHB3ENR_FMCEN)) == RESET)
#define __HAL_RCC_QSPI_IS_CLK_DISABLED() ((RCC->AHB3ENR & (RCC_AHB3ENR_QSPIEN)) == RESET)
/**
  * @}
  */
  
/** @defgroup RCCEx_APB1_Clock_Enable_Disable APB1 Peripheral Clock Enable Disable
  * @brief  Enable or disable the Low Speed APB (APB1) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it. 
  * @{
  */
#define __HAL_RCC_TIM6_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM7_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM7EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM7EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM12_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM12EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM12EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM13_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM13EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM13EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM14_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPDIFRX_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_SPDIFRXEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_SPDIFRXEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_USART3_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_USART3EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_USART3EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART4_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART4EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART4EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART5_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART5EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART5EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_FMPI2C1_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_FMPI2C1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_FMPI2C1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CAN1_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CAN2_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN2EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN2EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CEC_CLK_ENABLE()    do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CECEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CECEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_DAC_CLK_ENABLE()    do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM2_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_I2C3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM2_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM2EN))
#define __HAL_RCC_TIM3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM3EN))
#define __HAL_RCC_TIM4_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM4EN))
#define __HAL_RCC_SPI3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_SPI3EN))
#define __HAL_RCC_I2C3_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_I2C3EN))
#define __HAL_RCC_TIM6_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM6EN))
#define __HAL_RCC_TIM7_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM7EN))
#define __HAL_RCC_TIM12_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM12EN))
#define __HAL_RCC_TIM13_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM13EN))
#define __HAL_RCC_TIM14_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM14EN))
#define __HAL_RCC_SPDIFRX_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_SPDIFRXEN))
#define __HAL_RCC_USART3_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_USART3EN))
#define __HAL_RCC_UART4_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_UART4EN))
#define __HAL_RCC_UART5_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_UART5EN))
#define __HAL_RCC_FMPI2C1_CLK_DISABLE() (RCC->APB1ENR &= ~(RCC_APB1ENR_FMPI2C1EN))
#define __HAL_RCC_CAN1_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_CAN1EN))
#define __HAL_RCC_CAN2_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_CAN2EN))
#define __HAL_RCC_CEC_CLK_DISABLE()     (RCC->APB1ENR &= ~(RCC_APB1ENR_CECEN))
#define __HAL_RCC_DAC_CLK_DISABLE()     (RCC->APB1ENR &= ~(RCC_APB1ENR_DACEN))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Peripheral_Clock_Enable_Disable_Status APB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM2_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) != RESET)  
#define __HAL_RCC_TIM3_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) != RESET)
#define __HAL_RCC_TIM4_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) != RESET) 
#define __HAL_RCC_SPI3_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) != RESET)
#define __HAL_RCC_I2C3_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) != RESET)
#define __HAL_RCC_TIM6_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) != RESET)
#define __HAL_RCC_TIM7_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM7EN)) != RESET)
#define __HAL_RCC_TIM12_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM12EN)) != RESET)
#define __HAL_RCC_TIM13_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM13EN)) != RESET)
#define __HAL_RCC_TIM14_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM14EN)) != RESET)
#define __HAL_RCC_SPDIFRX_IS_CLK_ENABLED() ((RCC->APB1ENR & (RCC_APB1ENR_SPDIFRXEN)) != RESET)
#define __HAL_RCC_USART3_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_USART3EN)) != RESET)
#define __HAL_RCC_UART4_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_UART4EN)) != RESET)
#define __HAL_RCC_UART5_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_UART5EN)) != RESET)
#define __HAL_RCC_FMPI2C1_IS_CLK_ENABLED() ((RCC->APB1ENR & (RCC_APB1ENR_FMPI2C1EN)) != RESET)
#define __HAL_RCC_CAN1_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_CAN1EN)) != RESET)
#define __HAL_RCC_CAN2_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_CAN2EN)) != RESET)
#define __HAL_RCC_CEC_IS_CLK_ENABLED()     ((RCC->APB1ENR & (RCC_APB1ENR_CECEN)) != RESET)
#define __HAL_RCC_DAC_IS_CLK_ENABLED()     ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) != RESET)

#define __HAL_RCC_TIM2_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) == RESET)  
#define __HAL_RCC_TIM3_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) == RESET)
#define __HAL_RCC_TIM4_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) == RESET) 
#define __HAL_RCC_SPI3_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) == RESET)
#define __HAL_RCC_I2C3_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) == RESET)
#define __HAL_RCC_TIM6_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) == RESET)
#define __HAL_RCC_TIM7_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM7EN)) == RESET)
#define __HAL_RCC_TIM12_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM12EN)) == RESET)
#define __HAL_RCC_TIM13_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM13EN)) == RESET)
#define __HAL_RCC_TIM14_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM14EN)) == RESET)
#define __HAL_RCC_SPDIFRX_IS_CLK_DISABLED() ((RCC->APB1ENR & (RCC_APB1ENR_SPDIFRXEN)) == RESET)
#define __HAL_RCC_USART3_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_USART3EN)) == RESET)
#define __HAL_RCC_UART4_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_UART4EN)) == RESET)
#define __HAL_RCC_UART5_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_UART5EN)) == RESET)
#define __HAL_RCC_FMPI2C1_IS_CLK_DISABLED() ((RCC->APB1ENR & (RCC_APB1ENR_FMPI2C1EN)) == RESET)
#define __HAL_RCC_CAN1_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_CAN1EN)) == RESET)
#define __HAL_RCC_CAN2_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_CAN2EN)) == RESET)
#define __HAL_RCC_CEC_IS_CLK_DISABLED()     ((RCC->APB1ENR & (RCC_APB1ENR_CECEN)) == RESET)
#define __HAL_RCC_DAC_IS_CLK_DISABLED()     ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) == RESET)
/**
  * @}
  */
  
/** @defgroup RCCEx_APB2_Clock_Enable_Disable APB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the High Speed APB (APB2) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM8_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM8EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM8EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_ADC2_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC2EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC2EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_ADC3_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC3EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_ADC3EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SAI1_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SAI1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SAI1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SAI2_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SAI2EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SAI2EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SDIO_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM10_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SDIO_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SDIOEN))
#define __HAL_RCC_SPI4_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI4EN))
#define __HAL_RCC_TIM10_CLK_DISABLE()  (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM10EN))
#define __HAL_RCC_TIM8_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM8EN))
#define __HAL_RCC_ADC2_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_ADC2EN))
#define __HAL_RCC_ADC3_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_ADC3EN))
#define __HAL_RCC_SAI1_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SAI1EN))
#define __HAL_RCC_SAI2_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_SAI2EN))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Peripheral_Clock_Enable_Disable_Status APB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_SDIO_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) != RESET) 
#define __HAL_RCC_SPI4_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) != RESET) 
#define __HAL_RCC_TIM10_IS_CLK_ENABLED()  ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) != RESET)  
#define __HAL_RCC_TIM8_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_TIM8EN)) != RESET)
#define __HAL_RCC_ADC2_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_ADC2EN)) != RESET) 
#define __HAL_RCC_ADC3_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_ADC3EN)) != RESET) 
#define __HAL_RCC_SAI1_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SAI1EN)) != RESET)
#define __HAL_RCC_SAI2_IS_CLK_ENABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SAI2EN)) != RESET)

#define __HAL_RCC_SDIO_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) == RESET) 
#define __HAL_RCC_SPI4_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) == RESET) 
#define __HAL_RCC_TIM10_IS_CLK_DISABLED()  ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) == RESET)  
#define __HAL_RCC_TIM8_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_TIM8EN)) == RESET)
#define __HAL_RCC_ADC2_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_ADC2EN)) == RESET) 
#define __HAL_RCC_ADC3_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_ADC3EN)) == RESET) 
#define __HAL_RCC_SAI1_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SAI1EN)) == RESET)
#define __HAL_RCC_SAI2_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_SAI2EN)) == RESET) 
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB1_Force_Release_Reset AHB1 Force Release Reset 
  * @brief  Force or release AHB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_GPIOD_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_GPIOF_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOFRST))
#define __HAL_RCC_GPIOG_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOGRST))
#define __HAL_RCC_USB_OTG_HS_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_OTGHRST))
#define __HAL_RCC_CRC_FORCE_RESET()      (RCC->AHB1RSTR |= (RCC_AHB1RSTR_CRCRST))

#define __HAL_RCC_GPIOD_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIODRST))
#define __HAL_RCC_GPIOE_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOERST))
#define __HAL_RCC_GPIOF_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOFRST))
#define __HAL_RCC_GPIOG_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOGRST))
#define __HAL_RCC_USB_OTG_HS_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_OTGHRST))
#define __HAL_RCC_CRC_RELEASE_RESET()    (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_CRCRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Force_Release_Reset AHB2 Force Release Reset 
  * @brief  Force or release AHB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_AHB2_FORCE_RESET()    (RCC->AHB2RSTR = 0xFFFFFFFFU) 
#define __HAL_RCC_USB_OTG_FS_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_OTGFSRST))
#define __HAL_RCC_RNG_FORCE_RESET()    (RCC->AHB2RSTR |= (RCC_AHB2RSTR_RNGRST))
#define __HAL_RCC_DCMI_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_DCMIRST))

#define __HAL_RCC_AHB2_RELEASE_RESET()  (RCC->AHB2RSTR = 0x00U)
#define __HAL_RCC_USB_OTG_FS_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_OTGFSRST))
#define __HAL_RCC_RNG_RELEASE_RESET()  (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_RNGRST))
#define __HAL_RCC_DCMI_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_DCMIRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Force_Release_Reset AHB3 Force Release Reset 
  * @brief  Force or release AHB3 peripheral reset.
  * @{
  */ 
#define __HAL_RCC_AHB3_FORCE_RESET() (RCC->AHB3RSTR = 0xFFFFFFFFU)
#define __HAL_RCC_AHB3_RELEASE_RESET() (RCC->AHB3RSTR = 0x00U) 

#define __HAL_RCC_FMC_FORCE_RESET()    (RCC->AHB3RSTR |= (RCC_AHB3RSTR_FMCRST))
#define __HAL_RCC_QSPI_FORCE_RESET()   (RCC->AHB3RSTR |= (RCC_AHB3RSTR_QSPIRST))

#define __HAL_RCC_FMC_RELEASE_RESET()    (RCC->AHB3RSTR &= ~(RCC_AHB3RSTR_FMCRST))
#define __HAL_RCC_QSPI_RELEASE_RESET()   (RCC->AHB3RSTR &= ~(RCC_AHB3RSTR_QSPIRST))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Force_Release_Reset APB1 Force Release Reset 
  * @brief  Force or release APB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_TIM6_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_TIM7_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM7RST))
#define __HAL_RCC_TIM12_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM12RST))
#define __HAL_RCC_TIM13_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM13RST))
#define __HAL_RCC_TIM14_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM14RST))
#define __HAL_RCC_SPDIFRX_FORCE_RESET()  (RCC->APB1RSTR |= (RCC_APB1RSTR_SPDIFRXRST))
#define __HAL_RCC_USART3_FORCE_RESET()   (RCC->APB1RSTR |= (RCC_APB1RSTR_USART3RST))
#define __HAL_RCC_UART4_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART4RST))
#define __HAL_RCC_UART5_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART5RST))
#define __HAL_RCC_FMPI2C1_FORCE_RESET()  (RCC->APB1RSTR |= (RCC_APB1RSTR_FMPI2C1RST))
#define __HAL_RCC_CAN1_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_CAN1RST))
#define __HAL_RCC_CAN2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_CAN2RST))
#define __HAL_RCC_CEC_FORCE_RESET()      (RCC->APB1RSTR |= (RCC_APB1RSTR_CECRST))
#define __HAL_RCC_DAC_FORCE_RESET()      (RCC->APB1RSTR |= (RCC_APB1RSTR_DACRST))
#define __HAL_RCC_TIM2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_I2C3RST))
                                          
#define __HAL_RCC_TIM2_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_SPI3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_I2C3_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_I2C3RST))
#define __HAL_RCC_TIM6_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_TIM7_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM7RST))
#define __HAL_RCC_TIM12_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM12RST))
#define __HAL_RCC_TIM13_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM13RST))
#define __HAL_RCC_TIM14_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM14RST))
#define __HAL_RCC_SPDIFRX_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_SPDIFRXRST))
#define __HAL_RCC_USART3_RELEASE_RESET() (RCC->APB1RSTR &= ~(RCC_APB1RSTR_USART3RST))
#define __HAL_RCC_UART4_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART4RST))
#define __HAL_RCC_UART5_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART5RST))
#define __HAL_RCC_FMPI2C1_RELEASE_RESET() (RCC->APB1RSTR &= ~(RCC_APB1RSTR_FMPI2C1RST))
#define __HAL_RCC_CAN1_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CAN1RST))
#define __HAL_RCC_CAN2_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CAN2RST))
#define __HAL_RCC_CEC_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CECRST))
#define __HAL_RCC_DAC_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_DACRST))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Force_Release_Reset APB2 Force Release Reset 
  * @brief  Force or release APB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_TIM8_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM8RST))
#define __HAL_RCC_SAI1_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SAI1RST)) 
#define __HAL_RCC_SAI2_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SAI2RST))
#define __HAL_RCC_SDIO_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_FORCE_RESET()    (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM10RST))

#define __HAL_RCC_SDIO_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_RELEASE_RESET()  (RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM10RST))
#define __HAL_RCC_TIM8_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM8RST))
#define __HAL_RCC_SAI1_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SAI1RST))
#define __HAL_RCC_SAI2_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SAI2RST)) 
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_LowPower_Enable_Disable AHB1 Peripheral Low Power Enable Disable 
  * @brief  Enable or disable the AHB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_GPIOD_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_GPIOF_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOFLPEN))
#define __HAL_RCC_GPIOG_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOGLPEN))
#define __HAL_RCC_SRAM2_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM2LPEN))
#define __HAL_RCC_USB_OTG_HS_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_OTGHSLPEN))
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_SLEEP_ENABLE()  (RCC->AHB1LPENR |= (RCC_AHB1LPENR_OTGHSULPILPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_ENABLE()        (RCC->AHB1LPENR |= (RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM1LPEN))
#define __HAL_RCC_BKPSRAM_CLK_SLEEP_ENABLE()    (RCC->AHB1LPENR |= (RCC_AHB1LPENR_BKPSRAMLPEN))

#define __HAL_RCC_GPIOD_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_GPIOF_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOFLPEN))
#define __HAL_RCC_GPIOG_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOGLPEN))
#define __HAL_RCC_SRAM2_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM2LPEN))
#define __HAL_RCC_USB_OTG_HS_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_OTGHSLPEN))
#define __HAL_RCC_USB_OTG_HS_ULPI_CLK_SLEEP_DISABLE() (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_OTGHSULPILPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_DISABLE()       (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM1LPEN))
#define __HAL_RCC_BKPSRAM_CLK_SLEEP_DISABLE()   (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_BKPSRAMLPEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_LowPower_Enable_Disable AHB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wake-up from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_OTGFSLPEN))
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_OTGFSLPEN))

#define __HAL_RCC_RNG_CLK_SLEEP_ENABLE()   (RCC->AHB2LPENR |= (RCC_AHB2LPENR_RNGLPEN))
#define __HAL_RCC_RNG_CLK_SLEEP_DISABLE()  (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_RNGLPEN))

#define __HAL_RCC_DCMI_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_DCMILPEN))
#define __HAL_RCC_DCMI_CLK_SLEEP_DISABLE() (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_DCMILPEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_LowPower_Enable_Disable AHB3 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB3 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_FMC_CLK_SLEEP_ENABLE()   (RCC->AHB3LPENR |= (RCC_AHB3LPENR_FMCLPEN))
#define __HAL_RCC_QSPI_CLK_SLEEP_ENABLE()  (RCC->AHB3LPENR |= (RCC_AHB3LPENR_QSPILPEN))

#define __HAL_RCC_FMC_CLK_SLEEP_DISABLE()   (RCC->AHB3LPENR &= ~(RCC_AHB3LPENR_FMCLPEN))
#define __HAL_RCC_QSPI_CLK_SLEEP_DISABLE()  (RCC->AHB3LPENR &= ~(RCC_AHB3LPENR_QSPILPEN))
/**
  * @}
  */

/** @defgroup RCCEx_APB1_LowPower_Enable_Disable APB1 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */ 
#define __HAL_RCC_TIM6_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_TIM7_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM7LPEN))
#define __HAL_RCC_TIM12_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM12LPEN))
#define __HAL_RCC_TIM13_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM13LPEN))
#define __HAL_RCC_TIM14_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM14LPEN))
#define __HAL_RCC_SPDIFRX_CLK_SLEEP_ENABLE() (RCC->APB1LPENR |= (RCC_APB1LPENR_SPDIFRXLPEN))
#define __HAL_RCC_USART3_CLK_SLEEP_ENABLE()  (RCC->APB1LPENR |= (RCC_APB1LPENR_USART3LPEN))
#define __HAL_RCC_UART4_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART4LPEN))
#define __HAL_RCC_UART5_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART5LPEN))
#define __HAL_RCC_FMPI2C1_CLK_SLEEP_ENABLE() (RCC->APB1LPENR |= (RCC_APB1LPENR_FMPI2C1LPEN))
#define __HAL_RCC_CAN1_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_CAN1LPEN))
#define __HAL_RCC_CAN2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_CAN2LPEN))
#define __HAL_RCC_CEC_CLK_SLEEP_ENABLE()     (RCC->APB1LPENR |= (RCC_APB1LPENR_CECLPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_ENABLE()     (RCC->APB1LPENR |= (RCC_APB1LPENR_DACLPEN))
#define __HAL_RCC_TIM2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_I2C3LPEN))

#define __HAL_RCC_TIM2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_SPI3LPEN))
#define __HAL_RCC_I2C3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_I2C3LPEN))
#define __HAL_RCC_TIM6_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_TIM7_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM7LPEN))
#define __HAL_RCC_TIM12_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM12LPEN))
#define __HAL_RCC_TIM13_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM13LPEN))
#define __HAL_RCC_TIM14_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM14LPEN))
#define __HAL_RCC_SPDIFRX_CLK_SLEEP_DISABLE()(RCC->APB1LPENR &= ~(RCC_APB1LPENR_SPDIFRXLPEN))
#define __HAL_RCC_USART3_CLK_SLEEP_DISABLE() (RCC->APB1LPENR &= ~(RCC_APB1LPENR_USART3LPEN))
#define __HAL_RCC_UART4_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART4LPEN))
#define __HAL_RCC_UART5_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART5LPEN))
#define __HAL_RCC_FMPI2C1_CLK_SLEEP_DISABLE()(RCC->APB1LPENR &= ~(RCC_APB1LPENR_FMPI2C1LPEN))
#define __HAL_RCC_CAN1_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CAN1LPEN))
#define __HAL_RCC_CAN2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CAN2LPEN))
#define __HAL_RCC_CEC_CLK_SLEEP_DISABLE()    (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CECLPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_DISABLE()    (RCC->APB1LPENR &= ~(RCC_APB1LPENR_DACLPEN))
/**
  * @}
  */

/** @defgroup RCCEx_APB2_LowPower_Enable_Disable APB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_TIM8_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_TIM8LPEN))
#define __HAL_RCC_ADC2_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_ADC2LPEN))
#define __HAL_RCC_ADC3_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_ADC3LPEN))
#define __HAL_RCC_SAI1_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SAI1LPEN))
#define __HAL_RCC_SAI2_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SAI2LPEN))
#define __HAL_RCC_SDIO_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_ENABLE() (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_ENABLE()(RCC->APB2LPENR |= (RCC_APB2LPENR_TIM10LPEN))

#define __HAL_RCC_SDIO_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_DISABLE()(RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM10LPEN))
#define __HAL_RCC_TIM8_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM8LPEN))
#define __HAL_RCC_ADC2_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_ADC2LPEN))
#define __HAL_RCC_ADC3_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_ADC3LPEN))
#define __HAL_RCC_SAI1_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SAI1LPEN))
#define __HAL_RCC_SAI2_CLK_SLEEP_DISABLE() (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SAI2LPEN))
/**
  * @}
  */

#endif /* STM32F446xx */
/*----------------------------------------------------------------------------*/

/*-------STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx-------*/
#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx) 
/** @defgroup RCCEx_AHB1_Clock_Enable_Disable AHB1 Peripheral Clock Enable Disable
  * @brief  Enables or disables the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#if defined(STM32F412Rx) || defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx) 
#define __HAL_RCC_GPIOD_CLK_ENABLE()   do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIODEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#endif /* STM32F412Rx || STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx) 
#define __HAL_RCC_GPIOE_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOEEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#endif /* STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)                                        
#define __HAL_RCC_GPIOF_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOFEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOFEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_GPIOG_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOGEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOGEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#endif /*  STM32F412Zx || STM32F413xx || STM32F423xx */                                       
#define __HAL_RCC_CRC_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_CRCEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)                                        
#if defined(STM32F412Rx) || defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx) 
#define __HAL_RCC_GPIOD_CLK_DISABLE()        (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIODEN))
#endif /* STM32F412Rx || STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx) 
#define __HAL_RCC_GPIOE_CLK_DISABLE()        (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOEEN))
#endif /* STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOF_CLK_DISABLE()        (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOFEN))
#define __HAL_RCC_GPIOG_CLK_DISABLE()        (RCC->AHB1ENR &= ~(RCC_AHB1ENR_GPIOGEN))
#endif /*  STM32F412Zx || STM32F413xx || STM32F423xx */
#define __HAL_RCC_CRC_CLK_DISABLE()          (RCC->AHB1ENR &= ~(RCC_AHB1ENR_CRCEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_Peripheral_Clock_Enable_Disable_Status AHB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#if defined(STM32F412Rx) || defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOD_IS_CLK_ENABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) != RESET)
#endif /* STM32F412Rx || STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOE_IS_CLK_ENABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) != RESET)
#endif /* STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOF_IS_CLK_ENABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOFEN)) != RESET)
#define __HAL_RCC_GPIOG_IS_CLK_ENABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOGEN)) != RESET)
#endif /*  STM32F412Zx || STM32F413xx || STM32F423xx */
#define __HAL_RCC_CRC_IS_CLK_ENABLED()       ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) != RESET)

#if defined(STM32F412Rx) || defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOD_IS_CLK_DISABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIODEN)) == RESET)
#endif /* STM32F412Rx || STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOE_IS_CLK_DISABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOEEN)) == RESET)
#endif /* STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOF_IS_CLK_DISABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOFEN)) == RESET)
#define __HAL_RCC_GPIOG_IS_CLK_DISABLED()     ((RCC->AHB1ENR & (RCC_AHB1ENR_GPIOGEN)) == RESET)
#endif /*  STM32F412Zx || STM32F413xx || STM32F423xx */
#define __HAL_RCC_CRC_IS_CLK_DISABLED()       ((RCC->AHB1ENR & (RCC_AHB1ENR_CRCEN)) == RESET)
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Clock_Enable_Disable AHB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#if defined(STM32F423xx)
#define __HAL_RCC_AES_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_AESEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_AESEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_AES_CLK_DISABLE()  (RCC->AHB2ENR &= ~(RCC_AHB2ENR_AESEN))
#endif /* STM32F423xx */
                                        
#define __HAL_RCC_RNG_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->AHB2ENR, RCC_AHB2ENR_RNGEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->AHB2ENR, RCC_AHB2ENR_RNGEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_RNG_CLK_DISABLE()   (RCC->AHB2ENR &= ~(RCC_AHB2ENR_RNGEN))
                                     
#define __HAL_RCC_USB_OTG_FS_CLK_ENABLE()  do {(RCC->AHB2ENR |= (RCC_AHB2ENR_OTGFSEN));\
                                               __HAL_RCC_SYSCFG_CLK_ENABLE();\
                                              }while(0U)
                                        
#define __HAL_RCC_USB_OTG_FS_CLK_DISABLE() (RCC->AHB2ENR &= ~(RCC_AHB2ENR_OTGFSEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Peripheral_Clock_Enable_Disable_Status AHB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#if defined(STM32F423xx)
#define __HAL_RCC_AES_IS_CLK_ENABLED()        ((RCC->AHB2ENR & (RCC_AHB2ENR_AESEN)) != RESET)
#define __HAL_RCC_AES_IS_CLK_DISABLED()       ((RCC->AHB2ENR & (RCC_AHB2ENR_AESEN)) == RESET)
#endif /* STM32F423xx */
                                        
#define __HAL_RCC_USB_OTG_FS_IS_CLK_ENABLED()  ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) != RESET)
#define __HAL_RCC_USB_OTG_FS_IS_CLK_DISABLED() ((RCC->AHB2ENR & (RCC_AHB2ENR_OTGFSEN)) == RESET)
          
#define __HAL_RCC_RNG_IS_CLK_ENABLED()         ((RCC->AHB2ENR & (RCC_AHB2ENR_RNGEN)) != RESET)   
#define __HAL_RCC_RNG_IS_CLK_DISABLED()        ((RCC->AHB2ENR & (RCC_AHB2ENR_RNGEN)) == RESET)   
/**
  * @}
  */  

/** @defgroup RCCEx_AHB3_Clock_Enable_Disable AHB3 Peripheral Clock Enable Disable
  * @brief  Enables or disables the AHB3 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it. 
  * @{
  */
#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_FSMC_CLK_ENABLE()    do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB3ENR, RCC_AHB3ENR_FSMCEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB3ENR, RCC_AHB3ENR_FSMCEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_QSPI_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->AHB3ENR, RCC_AHB3ENR_QSPIEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->AHB3ENR, RCC_AHB3ENR_QSPIEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)

#define __HAL_RCC_FSMC_CLK_DISABLE()    (RCC->AHB3ENR &= ~(RCC_AHB3ENR_FSMCEN))
#define __HAL_RCC_QSPI_CLK_DISABLE()    (RCC->AHB3ENR &= ~(RCC_AHB3ENR_QSPIEN))
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F413xx || STM32F423xx */ 
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Peripheral_Clock_Enable_Disable_Status AHB3 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the AHB3 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_FSMC_IS_CLK_ENABLED()  ((RCC->AHB3ENR & (RCC_AHB3ENR_FSMCEN)) != RESET) 
#define __HAL_RCC_QSPI_IS_CLK_ENABLED()  ((RCC->AHB3ENR & (RCC_AHB3ENR_QSPIEN)) != RESET) 

#define __HAL_RCC_FSMC_IS_CLK_DISABLED() ((RCC->AHB3ENR & (RCC_AHB3ENR_FSMCEN)) == RESET)
#define __HAL_RCC_QSPI_IS_CLK_DISABLED() ((RCC->AHB3ENR & (RCC_AHB3ENR_QSPIEN)) == RESET)
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F413xx || STM32F423xx */

/**
  * @}
  */
  
/** @defgroup RCCEx_APB1_Clock_Enable_Disable APB1 Peripheral Clock Enable Disable
  * @brief  Enable or disable the Low Speed APB (APB1) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it. 
  * @{
  */
#define __HAL_RCC_TIM6_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM6EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM7_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM7EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM7EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM12_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM12EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM12EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM13_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM13EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM13EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM14_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM14EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#if defined(STM32F413xx) || defined(STM32F423xx)                                        
#define __HAL_RCC_LPTIM1_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_LPTIM1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_LPTIM1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#endif /* STM32F413xx || STM32F423xx */  
#define __HAL_RCC_RTCAPB_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_RTCAPBEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_RTCAPBEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_USART3_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_USART3EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_USART3EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)

#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART4_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART4EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART4EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART5_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART5EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART5EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#endif /* STM32F413xx || STM32F423xx */
                                        
#define __HAL_RCC_FMPI2C1_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_FMPI2C1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_FMPI2C1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CAN1_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_CAN2_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN2EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN2EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_CAN3_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN3EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_CAN3EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#endif /* STM32F413xx || STM32F423xx */
#define __HAL_RCC_TIM2_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM2EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_TIM4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_TIM4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_SPI3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_SPI3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_I2C3_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_I2C3EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_DAC_CLK_ENABLE()    do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_DACEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART7_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART7EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART7EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART8_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB1ENR, RCC_APB1ENR_UART8EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_UART8EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#endif /* STM32F413xx || STM32F423xx */
                                        
#define __HAL_RCC_TIM2_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM2EN))
#define __HAL_RCC_TIM3_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM3EN))
#define __HAL_RCC_TIM4_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM4EN))
#define __HAL_RCC_TIM6_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM6EN))
#define __HAL_RCC_TIM7_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM7EN))
#define __HAL_RCC_TIM12_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM12EN))
#define __HAL_RCC_TIM13_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM13EN))
#define __HAL_RCC_TIM14_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_TIM14EN)) 
#if defined(STM32F413xx) || defined(STM32F423xx)                 
#define __HAL_RCC_LPTIM1_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_LPTIM1EN))
#endif /* STM32F413xx || STM32F423xx */
#define __HAL_RCC_RTCAPB_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_RTCAPBEN))                                       
#define __HAL_RCC_SPI3_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_SPI3EN))
#define __HAL_RCC_USART3_CLK_DISABLE()  (RCC->APB1ENR &= ~(RCC_APB1ENR_USART3EN))
#if defined(STM32F413xx) || defined(STM32F423xx)   
#define __HAL_RCC_UART4_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_UART4EN))
#define __HAL_RCC_UART5_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_UART5EN))
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_I2C3_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_I2C3EN))
#define __HAL_RCC_FMPI2C1_CLK_DISABLE() (RCC->APB1ENR &= ~(RCC_APB1ENR_FMPI2C1EN))
#define __HAL_RCC_CAN1_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_CAN1EN))
#define __HAL_RCC_CAN2_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_CAN2EN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_CAN3_CLK_DISABLE()    (RCC->APB1ENR &= ~(RCC_APB1ENR_CAN3EN))                                        
#define __HAL_RCC_DAC_CLK_DISABLE()     (RCC->APB1ENR &= ~(RCC_APB1ENR_DACEN))
#define __HAL_RCC_UART7_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_UART7EN))
#define __HAL_RCC_UART8_CLK_DISABLE()   (RCC->APB1ENR &= ~(RCC_APB1ENR_UART8EN))
#endif /* STM32F413xx || STM32F423xx */                                        
                                        
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Peripheral_Clock_Enable_Disable_Status APB1 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB1 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM2_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) != RESET) 
#define __HAL_RCC_TIM3_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) != RESET)
#define __HAL_RCC_TIM4_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) != RESET)
#define __HAL_RCC_TIM6_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) != RESET)
#define __HAL_RCC_TIM7_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM7EN)) != RESET)
#define __HAL_RCC_TIM12_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM12EN)) != RESET)
#define __HAL_RCC_TIM13_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM13EN)) != RESET)
#define __HAL_RCC_TIM14_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM14EN)) != RESET) 
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_LPTIM1_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_LPTIM1EN)) != RESET) 
#endif /* STM32F413xx || STM32F423xx */                                            
#define __HAL_RCC_RTCAPB_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_RTCAPBEN)) != RESET)                                    
#define __HAL_RCC_SPI3_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) != RESET)
#define __HAL_RCC_USART3_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_USART3EN)) != RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART4_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART4EN)) != RESET) 
#define __HAL_RCC_UART5_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART5EN)) != RESET) 
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_I2C3_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) != RESET)
#define __HAL_RCC_FMPI2C1_IS_CLK_ENABLED() ((RCC->APB1ENR & (RCC_APB1ENR_FMPI2C1EN)) != RESET)
#define __HAL_RCC_CAN1_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_CAN1EN))!= RESET)
#define __HAL_RCC_CAN2_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_CAN2EN)) != RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_CAN3_IS_CLK_ENABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN3EN)) != RESET)
#define __HAL_RCC_DAC_IS_CLK_ENABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) != RESET) 
#define __HAL_RCC_UART7_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART7EN)) != RESET)
#define __HAL_RCC_UART8_IS_CLK_ENABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART8EN)) != RESET) 
#endif /* STM32F413xx || STM32F423xx */                                         

#define __HAL_RCC_TIM2_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM2EN)) == RESET) 
#define __HAL_RCC_TIM3_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM3EN)) == RESET)
#define __HAL_RCC_TIM4_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM4EN)) == RESET)
#define __HAL_RCC_TIM6_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM6EN)) == RESET)
#define __HAL_RCC_TIM7_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_TIM7EN)) == RESET)
#define __HAL_RCC_TIM12_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM12EN)) == RESET)
#define __HAL_RCC_TIM13_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM13EN)) == RESET)
#define __HAL_RCC_TIM14_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_TIM14EN)) == RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)                                          
#define __HAL_RCC_LPTIM1_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_LPTIM1EN)) == RESET) 
#endif /* STM32F413xx || STM32F423xx */                                         
#define __HAL_RCC_RTCAPB_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_RTCAPBEN)) == RESET)                                        
#define __HAL_RCC_SPI3_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_SPI3EN)) == RESET)
#define __HAL_RCC_USART3_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_USART3EN)) == RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)                                           
#define __HAL_RCC_UART4_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART4EN)) == RESET) 
#define __HAL_RCC_UART5_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART5EN)) == RESET) 
#endif /* STM32F413xx || STM32F423xx */                                          
#define __HAL_RCC_I2C3_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_I2C3EN)) == RESET)
#define __HAL_RCC_FMPI2C1_IS_CLK_DISABLED() ((RCC->APB1ENR & (RCC_APB1ENR_FMPI2C1EN)) == RESET)
#define __HAL_RCC_CAN1_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_CAN1EN)) == RESET)
#define __HAL_RCC_CAN2_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_CAN2EN)) == RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)                                        
#define __HAL_RCC_CAN3_IS_CLK_DISABLED()   ((RCC->APB1ENR & (RCC_APB1ENR_CAN3EN)) == RESET)
#define __HAL_RCC_DAC_IS_CLK_DISABLED()    ((RCC->APB1ENR & (RCC_APB1ENR_DACEN)) == RESET) 
#define __HAL_RCC_UART7_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART7EN)) == RESET)
#define __HAL_RCC_UART8_IS_CLK_DISABLED()  ((RCC->APB1ENR & (RCC_APB1ENR_UART8EN)) == RESET)
#endif /* STM32F413xx || STM32F423xx */                                         
/**
  * @}
  */  
/** @defgroup RCCEx_APB2_Clock_Enable_Disable APB2 Peripheral Clock Enable Disable
  * @brief  Enable or disable the High Speed APB (APB2) peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before 
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM8_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM8EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM8EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART9_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_UART9EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_UART9EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_UART10_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_UART10EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_UART10EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)                                          
#endif /* STM32F413xx || STM32F423xx */                                   
#define __HAL_RCC_SDIO_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SDIOEN);\
                                        UNUSED(tmpreg); \
                                      } while(0U) 
#define __HAL_RCC_SPI4_CLK_ENABLE()     do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI4EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U)
#define __HAL_RCC_EXTIT_CLK_ENABLE()  do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_EXTITEN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_EXTITEN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)                                        
#define __HAL_RCC_TIM10_CLK_ENABLE()    do { \
                                        __IO uint32_t tmpreg = 0x00U; \
                                        SET_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        /* Delay after an RCC peripheral clock enabling */ \
                                        tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_TIM10EN);\
                                        UNUSED(tmpreg); \
                                      } while(0U) 
#define __HAL_RCC_SPI5_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI5EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SPI5EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SAI1_CLK_ENABLE()   do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_SAI1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_SAI1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)                                         
#endif /* STM32F413xx || STM32F423xx */                                          
#define __HAL_RCC_DFSDM1_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_DFSDM1EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_DFSDM1EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_DFSDM2_CLK_ENABLE() do { \
                                      __IO uint32_t tmpreg = 0x00U; \
                                      SET_BIT(RCC->APB2ENR, RCC_APB2ENR_DFSDM2EN);\
                                      /* Delay after an RCC peripheral clock enabling */ \
                                      tmpreg = READ_BIT(RCC->APB2ENR, RCC_APB2ENR_DFSDM2EN);\
                                      UNUSED(tmpreg); \
                                      } while(0U)                                        
#endif /* STM32F413xx || STM32F423xx */                                        
 
#define __HAL_RCC_TIM8_CLK_DISABLE()    (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM8EN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART9_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_UART9EN))
#define __HAL_RCC_UART10_CLK_DISABLE()  (RCC->APB2ENR &= ~(RCC_APB2ENR_UART10EN))                                        
#endif /* STM32F413xx || STM32F423xx */                                         
#define __HAL_RCC_SDIO_CLK_DISABLE()    (RCC->APB2ENR &= ~(RCC_APB2ENR_SDIOEN))
#define __HAL_RCC_SPI4_CLK_DISABLE()    (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI4EN))
#define __HAL_RCC_EXTIT_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_EXTITEN))
#define __HAL_RCC_TIM10_CLK_DISABLE()   (RCC->APB2ENR &= ~(RCC_APB2ENR_TIM10EN))
#define __HAL_RCC_SPI5_CLK_DISABLE()    (RCC->APB2ENR &= ~(RCC_APB2ENR_SPI5EN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SAI1_CLK_DISABLE()    (RCC->APB2ENR &= ~(RCC_APB2ENR_SAI1EN))                                        
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_DFSDM1_CLK_DISABLE()  (RCC->APB2ENR &= ~(RCC_APB2ENR_DFSDM1EN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_DFSDM2_CLK_DISABLE()  (RCC->APB2ENR &= ~(RCC_APB2ENR_DFSDM2EN))                                      
#endif /* STM32F413xx || STM32F423xx */                                         
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Peripheral_Clock_Enable_Disable_Status APB2 Peripheral Clock Enable Disable Status
  * @brief  Get the enable or disable status of the APB2 peripheral clock.
  * @note   After reset, the peripheral clock (used for registers read/write access)
  *         is disabled and the application software has to enable this clock before
  *         using it.
  * @{
  */
#define __HAL_RCC_TIM8_IS_CLK_ENABLED()      ((RCC->APB2ENR & (RCC_APB2ENR_TIM8EN)) != RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART9_IS_CLK_ENABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_UART9EN)) != RESET)
#define __HAL_RCC_UART10_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_UART10EN)) != RESET)                                        
#endif /* STM32F413xx || STM32F423xx */                                          
#define __HAL_RCC_SDIO_IS_CLK_ENABLED()      ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) != RESET) 
#define __HAL_RCC_SPI4_IS_CLK_ENABLED()      ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) != RESET)
#define __HAL_RCC_EXTIT_IS_CLK_ENABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_EXTITEN)) != RESET)
#define __HAL_RCC_TIM10_IS_CLK_ENABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) != RESET)
#define __HAL_RCC_SPI5_IS_CLK_ENABLED()      ((RCC->APB2ENR & (RCC_APB2ENR_SPI5EN)) != RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SAI1_IS_CLK_ENABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_SAI1EN)) != RESET)                                    
#endif /* STM32F413xx || STM32F423xx */                                         
#define __HAL_RCC_DFSDM1_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_DFSDM1EN)) != RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_DFSDM2_IS_CLK_ENABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_DFSDM2EN)) != RESET)                                  
#endif /* STM32F413xx || STM32F423xx */                                         

#define __HAL_RCC_TIM8_IS_CLK_DISABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_TIM8EN)) == RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART9_IS_CLK_DISABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_UART9EN)) == RESET) 
#define __HAL_RCC_UART10_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_UART10EN)) == RESET)                                         
#endif /* STM32F413xx || STM32F423xx */                                         
#define __HAL_RCC_SDIO_IS_CLK_DISABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_SDIOEN)) == RESET) 
#define __HAL_RCC_SPI4_IS_CLK_DISABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_SPI4EN)) == RESET)
#define __HAL_RCC_EXTIT_IS_CLK_DISABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_EXTITEN)) == RESET)
#define __HAL_RCC_TIM10_IS_CLK_DISABLED()    ((RCC->APB2ENR & (RCC_APB2ENR_TIM10EN)) == RESET)
#define __HAL_RCC_SPI5_IS_CLK_DISABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_SPI5EN)) == RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SAI1_IS_CLK_DISABLED()     ((RCC->APB2ENR & (RCC_APB2ENR_SAI1EN)) == RESET)                                       
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_DFSDM1_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_DFSDM1EN)) == RESET)
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_DFSDM2_IS_CLK_DISABLED()   ((RCC->APB2ENR & (RCC_APB2ENR_DFSDM2EN)) == RESET)                                       
#endif /* STM32F413xx || STM32F423xx */                                         
/**
  * @}
  */
  
/** @defgroup RCCEx_AHB1_Force_Release_Reset AHB1 Force Release Reset 
  * @brief  Force or release AHB1 peripheral reset.
  * @{
  */
#if defined(STM32F412Rx) || defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOD_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIODRST))
#endif /* STM32F412Rx || STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOE_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOERST))
#endif /* STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOF_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOFRST))
#define __HAL_RCC_GPIOG_FORCE_RESET()    (RCC->AHB1RSTR |= (RCC_AHB1RSTR_GPIOGRST))
#endif /*  STM32F412Zx || STM32F413xx || STM32F423xx */
#define __HAL_RCC_CRC_FORCE_RESET()      (RCC->AHB1RSTR |= (RCC_AHB1RSTR_CRCRST))

#if defined(STM32F412Rx) || defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOD_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIODRST))
#endif /* STM32F412Rx || STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Vx) || defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOE_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOERST))
#endif /* STM32F412Vx || STM32F412Zx ||  STM32F413xx || STM32F423xx */
#if defined(STM32F412Zx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_GPIOF_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOFRST))
#define __HAL_RCC_GPIOG_RELEASE_RESET()  (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_GPIOGRST))
#endif /*  STM32F412Zx || STM32F413xx || STM32F423xx */
#define __HAL_RCC_CRC_RELEASE_RESET()    (RCC->AHB1RSTR &= ~(RCC_AHB1RSTR_CRCRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_Force_Release_Reset AHB2 Force Release Reset 
  * @brief  Force or release AHB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_AHB2_FORCE_RESET()    (RCC->AHB2RSTR = 0xFFFFFFFFU)
#define __HAL_RCC_AHB2_RELEASE_RESET()  (RCC->AHB2RSTR = 0x00U)

#if defined(STM32F423xx)
#define __HAL_RCC_AES_FORCE_RESET()    (RCC->AHB2RSTR |= (RCC_AHB2RSTR_AESRST))
#define __HAL_RCC_AES_RELEASE_RESET()  (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_AESRST))                                        
#endif /* STM32F423xx */ 
                                        
#define __HAL_RCC_USB_OTG_FS_FORCE_RESET()   (RCC->AHB2RSTR |= (RCC_AHB2RSTR_OTGFSRST))
#define __HAL_RCC_USB_OTG_FS_RELEASE_RESET() (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_OTGFSRST))

#define __HAL_RCC_RNG_FORCE_RESET()    (RCC->AHB2RSTR |= (RCC_AHB2RSTR_RNGRST))
#define __HAL_RCC_RNG_RELEASE_RESET()  (RCC->AHB2RSTR &= ~(RCC_AHB2RSTR_RNGRST))
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_Force_Release_Reset AHB3 Force Release Reset 
  * @brief  Force or release AHB3 peripheral reset.
  * @{
  */ 
#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_AHB3_FORCE_RESET() (RCC->AHB3RSTR = 0xFFFFFFFFU)
#define __HAL_RCC_AHB3_RELEASE_RESET() (RCC->AHB3RSTR = 0x00U) 

#define __HAL_RCC_FSMC_FORCE_RESET()    (RCC->AHB3RSTR |= (RCC_AHB3RSTR_FSMCRST))
#define __HAL_RCC_QSPI_FORCE_RESET()   (RCC->AHB3RSTR |= (RCC_AHB3RSTR_QSPIRST))

#define __HAL_RCC_FSMC_RELEASE_RESET()    (RCC->AHB3RSTR &= ~(RCC_AHB3RSTR_FSMCRST))
#define __HAL_RCC_QSPI_RELEASE_RESET()   (RCC->AHB3RSTR &= ~(RCC_AHB3RSTR_QSPIRST))
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F413xx || STM32F423xx */ 
#if defined(STM32F412Cx)
#define __HAL_RCC_AHB3_FORCE_RESET()
#define __HAL_RCC_AHB3_RELEASE_RESET()

#define __HAL_RCC_FSMC_FORCE_RESET()
#define __HAL_RCC_QSPI_FORCE_RESET()

#define __HAL_RCC_FSMC_RELEASE_RESET()
#define __HAL_RCC_QSPI_RELEASE_RESET()
#endif /* STM32F412Cx */
/**
  * @}
  */

/** @defgroup RCCEx_APB1_Force_Release_Reset APB1 Force Release Reset 
  * @brief  Force or release APB1 peripheral reset.
  * @{
  */
#define __HAL_RCC_TIM2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM3RST)) 
#define __HAL_RCC_TIM4_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM4RST))                                        
#define __HAL_RCC_TIM6_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_TIM7_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM7RST))
#define __HAL_RCC_TIM12_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM12RST))
#define __HAL_RCC_TIM13_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM13RST))
#define __HAL_RCC_TIM14_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_TIM14RST)) 
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_LPTIM1_FORCE_RESET()   (RCC->APB1RSTR |= (RCC_APB1RSTR_LPTIM1RST)) 
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_SPI3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_SPI3RST))                                        
#define __HAL_RCC_USART3_FORCE_RESET()   (RCC->APB1RSTR |= (RCC_APB1RSTR_USART3RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART4_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART4RST))
#define __HAL_RCC_UART5_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART5RST))                                        
#endif /* STM32F413xx || STM32F423xx */                                          
#define __HAL_RCC_I2C3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_I2C3RST))                                        
#define __HAL_RCC_FMPI2C1_FORCE_RESET()  (RCC->APB1RSTR |= (RCC_APB1RSTR_FMPI2C1RST))
#define __HAL_RCC_CAN1_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_CAN1RST))
#define __HAL_RCC_CAN2_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_CAN2RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_CAN3_FORCE_RESET()     (RCC->APB1RSTR |= (RCC_APB1RSTR_CAN3RST))
#define __HAL_RCC_DAC_FORCE_RESET()      (RCC->APB1RSTR |= (RCC_APB1RSTR_DACRST))
#define __HAL_RCC_UART7_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART7RST))
#define __HAL_RCC_UART8_FORCE_RESET()    (RCC->APB1RSTR |= (RCC_APB1RSTR_UART8RST))                                        
#endif /* STM32F413xx || STM32F423xx */                                        

#define __HAL_RCC_TIM2_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM2RST))
#define __HAL_RCC_TIM3_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM3RST))
#define __HAL_RCC_TIM4_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM4RST))
#define __HAL_RCC_TIM6_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM6RST))
#define __HAL_RCC_TIM7_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM7RST))
#define __HAL_RCC_TIM12_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM12RST))
#define __HAL_RCC_TIM13_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM13RST))
#define __HAL_RCC_TIM14_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_TIM14RST)) 
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_LPTIM1_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_LPTIM1RST))
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_SPI3_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_SPI3RST))
#define __HAL_RCC_USART3_RELEASE_RESET()  (RCC->APB1RSTR &= ~(RCC_APB1RSTR_USART3RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART4_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART4RST))
#define __HAL_RCC_UART5_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART5RST))
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_I2C3_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_I2C3RST))                                        
#define __HAL_RCC_FMPI2C1_RELEASE_RESET() (RCC->APB1RSTR &= ~(RCC_APB1RSTR_FMPI2C1RST))
#define __HAL_RCC_CAN1_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CAN1RST))
#define __HAL_RCC_CAN2_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CAN2RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_CAN3_RELEASE_RESET()    (RCC->APB1RSTR &= ~(RCC_APB1RSTR_CAN3RST))
#define __HAL_RCC_DAC_RELEASE_RESET()     (RCC->APB1RSTR &= ~(RCC_APB1RSTR_DACRST))
#define __HAL_RCC_UART7_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART7RST))
#define __HAL_RCC_UART8_RELEASE_RESET()   (RCC->APB1RSTR &= ~(RCC_APB1RSTR_UART8RST))                                      
#endif /* STM32F413xx || STM32F423xx */                                         
/**
  * @}
  */

/** @defgroup RCCEx_APB2_Force_Release_Reset APB2 Force Release Reset
  * @brief  Force or release APB2 peripheral reset.
  * @{
  */
#define __HAL_RCC_TIM8_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM8RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART9_FORCE_RESET()    (RCC->APB2RSTR |= (RCC_APB2RSTR_UART9RST))
#define __HAL_RCC_UART10_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_UART10RST))                                        
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_SDIO_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_FORCE_RESET()    (RCC->APB2RSTR |= (RCC_APB2RSTR_TIM10RST))                                        
#define __HAL_RCC_SPI5_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SPI5RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SAI1_FORCE_RESET()     (RCC->APB2RSTR |= (RCC_APB2RSTR_SAI1RST))
#endif /* STM32F413xx || STM32F423xx */                                         
#define __HAL_RCC_DFSDM1_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_DFSDM1RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_DFSDM2_FORCE_RESET()   (RCC->APB2RSTR |= (RCC_APB2RSTR_DFSDM2RST))
#endif /* STM32F413xx || STM32F423xx */                                        

#define __HAL_RCC_TIM8_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM8RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART9_RELEASE_RESET()  (RCC->APB2RSTR &= ~(RCC_APB2RSTR_UART9RST))
#define __HAL_RCC_UART10_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_UART10RST))                                        
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_SDIO_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SDIORST))
#define __HAL_RCC_SPI4_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI4RST))
#define __HAL_RCC_TIM10_RELEASE_RESET()  (RCC->APB2RSTR &= ~(RCC_APB2RSTR_TIM10RST))
#define __HAL_RCC_SPI5_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SPI5RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SAI1_RELEASE_RESET()   (RCC->APB2RSTR &= ~(RCC_APB2RSTR_SAI1RST))
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_DFSDM1_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_DFSDM1RST))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_DFSDM2_RELEASE_RESET() (RCC->APB2RSTR &= ~(RCC_APB2RSTR_DFSDM2RST))
#endif /* STM32F413xx || STM32F423xx */                                        
/**
  * @}
  */

/** @defgroup RCCEx_AHB1_LowPower_Enable_Disable AHB1 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_GPIOD_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_GPIOF_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOFLPEN))
#define __HAL_RCC_GPIOG_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_GPIOGLPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_ENABLE()        (RCC->AHB1LPENR |= (RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM1LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SRAM2_CLK_SLEEP_ENABLE()      (RCC->AHB1LPENR |= (RCC_AHB1LPENR_SRAM2LPEN))
#endif /* STM32F413xx || STM32F423xx */                                        

#define __HAL_RCC_GPIOD_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIODLPEN))
#define __HAL_RCC_GPIOE_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOELPEN))
#define __HAL_RCC_GPIOF_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOFLPEN))
#define __HAL_RCC_GPIOG_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_GPIOGLPEN))
#define __HAL_RCC_CRC_CLK_SLEEP_DISABLE()       (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_CRCLPEN))
#define __HAL_RCC_FLITF_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_FLITFLPEN))
#define __HAL_RCC_SRAM1_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM1LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SRAM2_CLK_SLEEP_DISABLE()     (RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_SRAM2LPEN))
#endif /* STM32F413xx || STM32F423xx */                                        
/**
  * @}
  */

/** @defgroup RCCEx_AHB2_LowPower_Enable_Disable AHB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wake-up from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#if defined(STM32F423xx)
#define __HAL_RCC_AES_CLK_SLEEP_ENABLE()      (RCC->AHB2LPENR |= (RCC_AHB2LPENR_AESLPEN))                                        
#define __HAL_RCC_AES_CLK_SLEEP_DISABLE()     (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_AESLPEN))
#endif /* STM32F423xx */
                                        
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_ENABLE()  (RCC->AHB2LPENR |= (RCC_AHB2LPENR_OTGFSLPEN))
#define __HAL_RCC_USB_OTG_FS_CLK_SLEEP_DISABLE()   (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_OTGFSLPEN))

#define __HAL_RCC_RNG_CLK_SLEEP_ENABLE()   (RCC->AHB2LPENR |= (RCC_AHB2LPENR_RNGLPEN))
#define __HAL_RCC_RNG_CLK_SLEEP_DISABLE()  (RCC->AHB2LPENR &= ~(RCC_AHB2LPENR_RNGLPEN))
/**
  * @}
  */

/** @defgroup RCCEx_AHB3_LowPower_Enable_Disable AHB3 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the AHB3 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_FSMC_CLK_SLEEP_ENABLE()   (RCC->AHB3LPENR |= (RCC_AHB3LPENR_FSMCLPEN))
#define __HAL_RCC_QSPI_CLK_SLEEP_ENABLE()  (RCC->AHB3LPENR |= (RCC_AHB3LPENR_QSPILPEN))

#define __HAL_RCC_FSMC_CLK_SLEEP_DISABLE()   (RCC->AHB3LPENR &= ~(RCC_AHB3LPENR_FSMCLPEN))
#define __HAL_RCC_QSPI_CLK_SLEEP_DISABLE()  (RCC->AHB3LPENR &= ~(RCC_AHB3LPENR_QSPILPEN))
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F413xx || STM32F423xx */

/**
  * @}
  */

/** @defgroup RCCEx_APB1_LowPower_Enable_Disable APB1 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB1 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_TIM2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM4LPEN))                                        
#define __HAL_RCC_TIM6_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_TIM7_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM7LPEN))
#define __HAL_RCC_TIM12_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM12LPEN))
#define __HAL_RCC_TIM13_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM13LPEN))
#define __HAL_RCC_TIM14_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_TIM14LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_LPTIM1_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_LPTIM1LPEN))
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_RTCAPB_CLK_SLEEP_ENABLE()  (RCC->APB1LPENR |= (RCC_APB1LPENR_RTCAPBLPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_SPI3LPEN))                                       
#define __HAL_RCC_USART3_CLK_SLEEP_ENABLE()  (RCC->APB1LPENR |= (RCC_APB1LPENR_USART3LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART4_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART4LPEN))
#define __HAL_RCC_UART5_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART5LPEN))
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_I2C3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_I2C3LPEN))                                        
#define __HAL_RCC_FMPI2C1_CLK_SLEEP_ENABLE() (RCC->APB1LPENR |= (RCC_APB1LPENR_FMPI2C1LPEN))
#define __HAL_RCC_CAN1_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_CAN1LPEN))
#define __HAL_RCC_CAN2_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_CAN2LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_CAN3_CLK_SLEEP_ENABLE()    (RCC->APB1LPENR |= (RCC_APB1LPENR_CAN3LPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_ENABLE()     (RCC->APB1LPENR |= (RCC_APB1LPENR_DACLPEN))
#define __HAL_RCC_UART7_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART7LPEN))
#define __HAL_RCC_UART8_CLK_SLEEP_ENABLE()   (RCC->APB1LPENR |= (RCC_APB1LPENR_UART8LPEN))                                        
#endif /* STM32F413xx || STM32F423xx */                                        

#define __HAL_RCC_TIM2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM2LPEN))
#define __HAL_RCC_TIM3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM3LPEN))
#define __HAL_RCC_TIM4_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM4LPEN))
#define __HAL_RCC_TIM6_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM6LPEN))
#define __HAL_RCC_TIM7_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM7LPEN))
#define __HAL_RCC_TIM12_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM12LPEN))
#define __HAL_RCC_TIM13_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM13LPEN))
#define __HAL_RCC_TIM14_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_TIM14LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_LPTIM1_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_LPTIM1LPEN))
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_RTCAPB_CLK_SLEEP_DISABLE() (RCC->APB1LPENR &= ~(RCC_APB1LPENR_RTCAPBLPEN))
#define __HAL_RCC_SPI3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_SPI3LPEN))                                        
#define __HAL_RCC_USART3_CLK_SLEEP_DISABLE() (RCC->APB1LPENR &= ~(RCC_APB1LPENR_USART3LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART4_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART4LPEN))
#define __HAL_RCC_UART5_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART5LPEN))
#endif /* STM32F413xx || STM32F423xx */                                
#define __HAL_RCC_I2C3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_I2C3LPEN))                                        
#define __HAL_RCC_FMPI2C1_CLK_SLEEP_DISABLE()(RCC->APB1LPENR &= ~(RCC_APB1LPENR_FMPI2C1LPEN))
#define __HAL_RCC_CAN1_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CAN1LPEN))
#define __HAL_RCC_CAN2_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CAN2LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_CAN3_CLK_SLEEP_DISABLE()   (RCC->APB1LPENR &= ~(RCC_APB1LPENR_CAN3LPEN))
#define __HAL_RCC_DAC_CLK_SLEEP_DISABLE()    (RCC->APB1LPENR &= ~(RCC_APB1LPENR_DACLPEN))
#define __HAL_RCC_UART7_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART7LPEN))
#define __HAL_RCC_UART8_CLK_SLEEP_DISABLE()  (RCC->APB1LPENR &= ~(RCC_APB1LPENR_UART8LPEN))                                        
#endif /* STM32F413xx || STM32F423xx */                                     
/**
  * @}
  */

/** @defgroup RCCEx_APB2_LowPower_Enable_Disable APB2 Peripheral Low Power Enable Disable
  * @brief  Enable or disable the APB2 peripheral clock during Low Power (Sleep) mode.
  * @note   Peripheral clock gating in SLEEP mode can be used to further reduce
  *         power consumption.
  * @note   After wakeup from SLEEP mode, the peripheral clock is enabled again.
  * @note   By default, all peripheral clocks are enabled during SLEEP mode.
  * @{
  */
#define __HAL_RCC_TIM8_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_TIM8LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART9_CLK_SLEEP_ENABLE()   (RCC->APB2LPENR |= (RCC_APB2LPENR_UART9LPEN))
#define __HAL_RCC_UART10_CLK_SLEEP_ENABLE()  (RCC->APB2LPENR |= (RCC_APB2LPENR_UART10LPEN))                                        
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_SDIO_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI4LPEN))  
#define __HAL_RCC_EXTIT_CLK_SLEEP_ENABLE()   (RCC->APB2LPENR |= (RCC_APB2LPENR_EXTITLPEN)) 
#define __HAL_RCC_TIM10_CLK_SLEEP_ENABLE()   (RCC->APB2LPENR |= (RCC_APB2LPENR_TIM10LPEN))                                        
#define __HAL_RCC_SPI5_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_SPI5LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SAI1_CLK_SLEEP_ENABLE()    (RCC->APB2LPENR |= (RCC_APB2LPENR_SAI1LPEN))
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_DFSDM1_CLK_SLEEP_ENABLE()  (RCC->APB2LPENR |= (RCC_APB2LPENR_DFSDM1LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_DFSDM2_CLK_SLEEP_ENABLE()  (RCC->APB2LPENR |= (RCC_APB2LPENR_DFSDM2LPEN))
#endif /* STM32F413xx || STM32F423xx */
                                        
#define __HAL_RCC_TIM8_CLK_SLEEP_DISABLE()    (RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM8LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_UART9_CLK_SLEEP_DISABLE()   (RCC->APB2LPENR &= ~(RCC_APB2LPENR_UART9LPEN))
#define __HAL_RCC_UART10_CLK_SLEEP_DISABLE()  (RCC->APB2LPENR &= ~(RCC_APB2LPENR_UART10LPEN))                                        
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_SDIO_CLK_SLEEP_DISABLE()    (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SDIOLPEN))
#define __HAL_RCC_SPI4_CLK_SLEEP_DISABLE()    (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI4LPEN))
#define __HAL_RCC_EXTIT_CLK_SLEEP_DISABLE()   (RCC->APB2LPENR &= ~(RCC_APB2LPENR_EXTITLPEN))
#define __HAL_RCC_TIM10_CLK_SLEEP_DISABLE()   (RCC->APB2LPENR &= ~(RCC_APB2LPENR_TIM10LPEN))    
#define __HAL_RCC_SPI5_CLK_SLEEP_DISABLE()    (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SPI5LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_SAI1_CLK_SLEEP_DISABLE()    (RCC->APB2LPENR &= ~(RCC_APB2LPENR_SAI1LPEN))
#endif /* STM32F413xx || STM32F423xx */                                        
#define __HAL_RCC_DFSDM1_CLK_SLEEP_DISABLE()  (RCC->APB2LPENR &= ~(RCC_APB2LPENR_DFSDM1LPEN))
#if defined(STM32F413xx) || defined(STM32F423xx)
#define __HAL_RCC_DFSDM2_CLK_SLEEP_DISABLE()  (RCC->APB2LPENR &= ~(RCC_APB2LPENR_DFSDM2LPEN))
#endif /* STM32F413xx || STM32F423xx */                                        
/**
  * @}
  */
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */
/*----------------------------------------------------------------------------*/

/*------------------------------- PLL Configuration --------------------------*/
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F446xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || \
    defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/** @brief  Macro to configure the main PLL clock source, multiplication and division factors.
  * @note   This function must be used only when the main PLL is disabled.
  * @param  __RCC_PLLSource__ specifies the PLL entry clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_PLLSOURCE_HSI: HSI oscillator clock selected as PLL clock entry
  *            @arg RCC_PLLSOURCE_HSE: HSE oscillator clock selected as PLL clock entry
  * @note   This clock source (RCC_PLLSource) is common for the main PLL and PLLI2S.  
  * @param  __PLLM__ specifies the division factor for PLL VCO input clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 63.
  * @note   You have to set the PLLM parameter correctly to ensure that the VCO input
  *         frequency ranges from 1 to 2 MHz. It is recommended to select a frequency
  *         of 2 MHz to limit PLL jitter.
  * @param  __PLLN__ specifies the multiplication factor for PLL VCO output clock
  *         This parameter must be a number between Min_Data = 50 and Max_Data = 432.
  * @note   You have to set the PLLN parameter correctly to ensure that the VCO
  *         output frequency is between 100 and 432 MHz.
  *   
  * @param  __PLLP__ specifies the division factor for main system clock (SYSCLK)
  *         This parameter must be a number in the range {2, 4, 6, or 8}.
  *           
  * @param  __PLLQ__ specifies the division factor for OTG FS, SDIO and RNG clocks
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 15.
  * @note   If the USB OTG FS is used in your application, you have to set the
  *         PLLQ parameter correctly to have 48 MHz clock for the USB. However,
  *         the SDIO and RNG need a frequency lower than or equal to 48 MHz to work
  *         correctly.
  *     
  * @param  __PLLR__ PLL division factor for I2S, SAI, SYSTEM, SPDIFRX clocks.
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 7.
  * @note   This parameter is only available in STM32F446xx/STM32F469xx/STM32F479xx/
            STM32F412Zx/STM32F412Vx/STM32F412Rx/STM32F412Cx/STM32F413xx/STM32F423xx devices.
  *      
  */
#define __HAL_RCC_PLL_CONFIG(__RCC_PLLSource__, __PLLM__, __PLLN__, __PLLP__, __PLLQ__,__PLLR__)  \
                            (RCC->PLLCFGR = ((__RCC_PLLSource__) | (__PLLM__)                   | \
                            ((__PLLN__) << RCC_PLLCFGR_PLLN_Pos)                      | \
                            ((((__PLLP__) >> 1U) -1U) << RCC_PLLCFGR_PLLP_Pos)          | \
                            ((__PLLQ__) << RCC_PLLCFGR_PLLQ_Pos)                      | \
                            ((__PLLR__) << RCC_PLLCFGR_PLLR_Pos)))
#else
/** @brief  Macro to configure the main PLL clock source, multiplication and division factors.
  * @note   This function must be used only when the main PLL is disabled.
  * @param  __RCC_PLLSource__ specifies the PLL entry clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_PLLSOURCE_HSI: HSI oscillator clock selected as PLL clock entry
  *            @arg RCC_PLLSOURCE_HSE: HSE oscillator clock selected as PLL clock entry
  * @note   This clock source (RCC_PLLSource) is common for the main PLL and PLLI2S.  
  * @param  __PLLM__ specifies the division factor for PLL VCO input clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 63.
  * @note   You have to set the PLLM parameter correctly to ensure that the VCO input
  *         frequency ranges from 1 to 2 MHz. It is recommended to select a frequency
  *         of 2 MHz to limit PLL jitter.
  * @param  __PLLN__ specifies the multiplication factor for PLL VCO output clock
  *         This parameter must be a number between Min_Data = 50 and Max_Data = 432
  *         Except for STM32F411xE devices where Min_Data = 192.
  * @note   You have to set the PLLN parameter correctly to ensure that the VCO
  *         output frequency is between 100 and 432 MHz, Except for STM32F411xE devices
  *         where frequency is between 192 and 432 MHz.
  * @param  __PLLP__ specifies the division factor for main system clock (SYSCLK)
  *         This parameter must be a number in the range {2, 4, 6, or 8}.
  *           
  * @param  __PLLQ__ specifies the division factor for OTG FS, SDIO and RNG clocks
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 15.
  * @note   If the USB OTG FS is used in your application, you have to set the
  *         PLLQ parameter correctly to have 48 MHz clock for the USB. However,
  *         the SDIO and RNG need a frequency lower than or equal to 48 MHz to work
  *         correctly.
  *      
  */
#define __HAL_RCC_PLL_CONFIG(__RCC_PLLSource__, __PLLM__, __PLLN__, __PLLP__, __PLLQ__)     \
                            (RCC->PLLCFGR = (0x20000000U | (__RCC_PLLSource__) | (__PLLM__)| \
                            ((__PLLN__) << RCC_PLLCFGR_PLLN_Pos)                | \
                            ((((__PLLP__) >> 1U) -1U) << RCC_PLLCFGR_PLLP_Pos)    | \
                            ((__PLLQ__) << RCC_PLLCFGR_PLLQ_Pos)))
 #endif /* STM32F410xx || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */
/*----------------------------------------------------------------------------*/
                             
/*----------------------------PLLI2S Configuration ---------------------------*/
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || \
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx) || \
    defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || \
    defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)

/** @brief Macros to enable or disable the PLLI2S. 
  * @note  The PLLI2S is disabled by hardware when entering STOP and STANDBY modes.
  */
#define __HAL_RCC_PLLI2S_ENABLE() (*(__IO uint32_t *) RCC_CR_PLLI2SON_BB = ENABLE)
#define __HAL_RCC_PLLI2S_DISABLE() (*(__IO uint32_t *) RCC_CR_PLLI2SON_BB = DISABLE)

#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx ||
          STM32F401xC || STM32F401xE || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx || STM32F412Vx || 
          STM32F412Rx || STM32F412Cx */
#if defined(STM32F446xx)
/** @brief  Macro to configure the PLLI2S clock multiplication and division factors .
  * @note   This macro must be used only when the PLLI2S is disabled.
  * @note   PLLI2S clock source is common with the main PLL (configured in 
  *         HAL_RCC_ClockConfig() API).
  * @param  __PLLI2SM__ specifies the division factor for PLLI2S VCO input clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 63.
  * @note   You have to set the PLLI2SM parameter correctly to ensure that the VCO input
  *         frequency ranges from 1 to 2 MHz. It is recommended to select a frequency
  *         of 1 MHz to limit PLLI2S jitter.
  *
  * @param  __PLLI2SN__ specifies the multiplication factor for PLLI2S VCO output clock
  *         This parameter must be a number between Min_Data = 50 and Max_Data = 432.
  * @note   You have to set the PLLI2SN parameter correctly to ensure that the VCO 
  *         output frequency is between Min_Data = 100 and Max_Data = 432 MHz.
  *
  * @param  __PLLI2SP__ specifies division factor for SPDIFRX Clock.
  *         This parameter must be a number in the range {2, 4, 6, or 8}.
  * @note   the PLLI2SP parameter is only available with STM32F446xx Devices
  *                 
  * @param  __PLLI2SR__ specifies the division factor for I2S clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 7.
  * @note   You have to set the PLLI2SR parameter correctly to not exceed 192 MHz
  *         on the I2S clock frequency.
  *   
  * @param  __PLLI2SQ__ specifies the division factor for SAI clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 15.
  */
#define __HAL_RCC_PLLI2S_CONFIG(__PLLI2SM__, __PLLI2SN__, __PLLI2SP__, __PLLI2SQ__, __PLLI2SR__)    \
                               (RCC->PLLI2SCFGR = ((__PLLI2SM__)                                   |\
                               ((__PLLI2SN__) << RCC_PLLI2SCFGR_PLLI2SN_Pos)             |\
                               ((((__PLLI2SP__) >> 1U) -1U) << RCC_PLLI2SCFGR_PLLI2SP_Pos) |\
                               ((__PLLI2SQ__) << RCC_PLLI2SCFGR_PLLI2SQ_Pos)             |\
                               ((__PLLI2SR__) << RCC_PLLI2SCFGR_PLLI2SR_Pos)))
#elif defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) ||\
      defined(STM32F413xx) || defined(STM32F423xx)
/** @brief  Macro to configure the PLLI2S clock multiplication and division factors .
  * @note   This macro must be used only when the PLLI2S is disabled.
  * @note   PLLI2S clock source is common with the main PLL (configured in 
  *         HAL_RCC_ClockConfig() API).
  * @param  __PLLI2SM__ specifies the division factor for PLLI2S VCO input clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 63.
  * @note   You have to set the PLLI2SM parameter correctly to ensure that the VCO input
  *         frequency ranges from 1 to 2 MHz. It is recommended to select a frequency
  *         of 1 MHz to limit PLLI2S jitter.
  *
  * @param  __PLLI2SN__ specifies the multiplication factor for PLLI2S VCO output clock
  *         This parameter must be a number between Min_Data = 50 and Max_Data = 432.
  * @note   You have to set the PLLI2SN parameter correctly to ensure that the VCO 
  *         output frequency is between Min_Data = 100 and Max_Data = 432 MHz.
  *
  * @param  __PLLI2SR__ specifies the division factor for I2S clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 7.
  * @note   You have to set the PLLI2SR parameter correctly to not exceed 192 MHz
  *         on the I2S clock frequency.
  *
  * @param  __PLLI2SQ__ specifies the division factor for SAI clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 15.
  */
#define __HAL_RCC_PLLI2S_CONFIG(__PLLI2SM__, __PLLI2SN__, __PLLI2SQ__, __PLLI2SR__)    \
                               (RCC->PLLI2SCFGR = ((__PLLI2SM__)                                   |\
                               ((__PLLI2SN__) << RCC_PLLI2SCFGR_PLLI2SN_Pos)             |\
                               ((__PLLI2SQ__) << RCC_PLLI2SCFGR_PLLI2SQ_Pos)             |\
                               ((__PLLI2SR__) << RCC_PLLI2SCFGR_PLLI2SR_Pos)))
#else
/** @brief  Macro to configure the PLLI2S clock multiplication and division factors .
  * @note   This macro must be used only when the PLLI2S is disabled.
  * @note   PLLI2S clock source is common with the main PLL (configured in 
  *         HAL_RCC_ClockConfig() API).
  * @param  __PLLI2SN__ specifies the multiplication factor for PLLI2S VCO output clock
  *         This parameter must be a number between Min_Data = 50 and Max_Data = 432.
  * @note   You have to set the PLLI2SN parameter correctly to ensure that the VCO 
  *         output frequency is between Min_Data = 100 and Max_Data = 432 MHz.
  *
  * @param  __PLLI2SR__ specifies the division factor for I2S clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 7.
  * @note   You have to set the PLLI2SR parameter correctly to not exceed 192 MHz
  *         on the I2S clock frequency.
  *
  */
#define __HAL_RCC_PLLI2S_CONFIG(__PLLI2SN__, __PLLI2SR__)                                                    \
                               (RCC->PLLI2SCFGR = (((__PLLI2SN__) << RCC_PLLI2SCFGR_PLLI2SN_Pos)  |\
                               ((__PLLI2SR__) << RCC_PLLI2SCFGR_PLLI2SR_Pos)))
#endif /* STM32F446xx */

#if defined(STM32F411xE)
/** @brief  Macro to configure the PLLI2S clock multiplication and division factors .
  * @note   This macro must be used only when the PLLI2S is disabled.
  * @note   This macro must be used only when the PLLI2S is disabled.
  * @note   PLLI2S clock source is common with the main PLL (configured in 
  *         HAL_RCC_ClockConfig() API).
  * @param  __PLLI2SM__ specifies the division factor for PLLI2S VCO input clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 63.
  * @note   The PLLI2SM parameter is only used with STM32F411xE/STM32F410xx Devices
  * @note   You have to set the PLLI2SM parameter correctly to ensure that the VCO input
  *         frequency ranges from 1 to 2 MHz. It is recommended to select a frequency
  *         of 2 MHz to limit PLLI2S jitter.    
  * @param  __PLLI2SN__ specifies the multiplication factor for PLLI2S VCO output clock
  *         This parameter must be a number between Min_Data = 192 and Max_Data = 432.
  * @note   You have to set the PLLI2SN parameter correctly to ensure that the VCO 
  *         output frequency is between Min_Data = 192 and Max_Data = 432 MHz.
  * @param  __PLLI2SR__ specifies the division factor for I2S clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 7.
  * @note   You have to set the PLLI2SR parameter correctly to not exceed 192 MHz
  *         on the I2S clock frequency.
  */
#define __HAL_RCC_PLLI2S_I2SCLK_CONFIG(__PLLI2SM__, __PLLI2SN__, __PLLI2SR__) (RCC->PLLI2SCFGR = ((__PLLI2SM__)                                                       |\
                                                                                                  ((__PLLI2SN__) << RCC_PLLI2SCFGR_PLLI2SN_Pos)             |\
                                                                                                  ((__PLLI2SR__) << RCC_PLLI2SCFGR_PLLI2SR_Pos)))
#endif /* STM32F411xE */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
/** @brief  Macro used by the SAI HAL driver to configure the PLLI2S clock multiplication and division factors.
  * @note   This macro must be used only when the PLLI2S is disabled.
  * @note   PLLI2S clock source is common with the main PLL (configured in 
  *         HAL_RCC_ClockConfig() API)             
  * @param  __PLLI2SN__ specifies the multiplication factor for PLLI2S VCO output clock.
  *         This parameter must be a number between Min_Data = 50 and Max_Data = 432.
  * @note   You have to set the PLLI2SN parameter correctly to ensure that the VCO 
  *         output frequency is between Min_Data = 100 and Max_Data = 432 MHz.
  * @param  __PLLI2SQ__ specifies the division factor for SAI1 clock.
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 15. 
  * @note   the PLLI2SQ parameter is only available with STM32F427xx/437xx/429xx/439xx/469xx/479xx 
  *         Devices and can be configured using the __HAL_RCC_PLLI2S_PLLSAICLK_CONFIG() macro
  * @param  __PLLI2SR__ specifies the division factor for I2S clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 7.
  * @note   You have to set the PLLI2SR parameter correctly to not exceed 192 MHz
  *         on the I2S clock frequency.
  */
#define __HAL_RCC_PLLI2S_SAICLK_CONFIG(__PLLI2SN__, __PLLI2SQ__, __PLLI2SR__) (RCC->PLLI2SCFGR = ((__PLLI2SN__) << 6U)  |\
                                                                                                 ((__PLLI2SQ__) << 24U) |\
                                                                                                 ((__PLLI2SR__) << 28U))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */   
/*----------------------------------------------------------------------------*/

/*------------------------------ PLLSAI Configuration ------------------------*/
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
/** @brief Macros to Enable or Disable the PLLISAI. 
  * @note  The PLLSAI is only available with STM32F429x/439x Devices.
  * @note  The PLLSAI is disabled by hardware when entering STOP and STANDBY modes. 
  */
#define __HAL_RCC_PLLSAI_ENABLE() (*(__IO uint32_t *) RCC_CR_PLLSAION_BB = ENABLE)
#define __HAL_RCC_PLLSAI_DISABLE() (*(__IO uint32_t *) RCC_CR_PLLSAION_BB = DISABLE)

#if defined(STM32F446xx)
/** @brief  Macro to configure the PLLSAI clock multiplication and division factors.
  *
  * @param  __PLLSAIM__ specifies the division factor for PLLSAI VCO input clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 63.
  * @note   You have to set the PLLSAIM parameter correctly to ensure that the VCO input
  *         frequency ranges from 1 to 2 MHz. It is recommended to select a frequency
  *         of 1 MHz to limit PLLI2S jitter.
  * @note   The PLLSAIM parameter is only used with STM32F446xx Devices
  *             
  * @param  __PLLSAIN__ specifies the multiplication factor for PLLSAI VCO output clock.
  *         This parameter must be a number between Min_Data = 50 and Max_Data = 432.
  * @note   You have to set the PLLSAIN parameter correctly to ensure that the VCO 
  *         output frequency is between Min_Data = 100 and Max_Data = 432 MHz.
  *
  * @param  __PLLSAIP__ specifies division factor for OTG FS, SDIO and RNG clocks.
  *         This parameter must be a number in the range {2, 4, 6, or 8}.
  * @note   the PLLSAIP parameter is only available with STM32F446xx Devices
  *                 
  * @param  __PLLSAIQ__ specifies the division factor for SAI clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 15.
  *           
  * @param  __PLLSAIR__ specifies the division factor for LTDC clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 7.
  * @note   the PLLI2SR parameter is only available with STM32F427/437/429/439xx Devices  
  */
#define __HAL_RCC_PLLSAI_CONFIG(__PLLSAIM__, __PLLSAIN__, __PLLSAIP__, __PLLSAIQ__, __PLLSAIR__)     \
                               (RCC->PLLSAICFGR = ((__PLLSAIM__)                                   | \
                               ((__PLLSAIN__) << RCC_PLLSAICFGR_PLLSAIN_Pos)             | \
                               ((((__PLLSAIP__) >> 1U) -1U) << RCC_PLLSAICFGR_PLLSAIP_Pos) | \
                               ((__PLLSAIQ__) << RCC_PLLSAICFGR_PLLSAIQ_Pos))) 
#endif /* STM32F446xx */
                                 
#if defined(STM32F469xx) || defined(STM32F479xx)
/** @brief  Macro to configure the PLLSAI clock multiplication and division factors.
  *             
  * @param  __PLLSAIN__ specifies the multiplication factor for PLLSAI VCO output clock.
  *         This parameter must be a number between Min_Data = 50 and Max_Data = 432.
  * @note   You have to set the PLLSAIN parameter correctly to ensure that the VCO 
  *         output frequency is between Min_Data = 100 and Max_Data = 432 MHz.
  *
  * @param  __PLLSAIP__ specifies division factor for SDIO and CLK48 clocks.
  *         This parameter must be a number in the range {2, 4, 6, or 8}.
  *                 
  * @param  __PLLSAIQ__ specifies the division factor for SAI clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 15.
  *           
  * @param  __PLLSAIR__ specifies the division factor for LTDC clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 7.  
  */
#define __HAL_RCC_PLLSAI_CONFIG(__PLLSAIN__, __PLLSAIP__, __PLLSAIQ__, __PLLSAIR__) \
                               (RCC->PLLSAICFGR = (((__PLLSAIN__) << RCC_PLLSAICFGR_PLLSAIN_Pos)             |\
                                                   ((((__PLLSAIP__) >> 1U) -1U) << RCC_PLLSAICFGR_PLLSAIP_Pos) |\
                                                   ((__PLLSAIQ__) << RCC_PLLSAICFGR_PLLSAIQ_Pos)             |\
                                                   ((__PLLSAIR__) << RCC_PLLSAICFGR_PLLSAIR_Pos)))
#endif /* STM32F469xx || STM32F479xx */                                 

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx)
/** @brief  Macro to configure the PLLSAI clock multiplication and division factors.
  *             
  * @param  __PLLSAIN__ specifies the multiplication factor for PLLSAI VCO output clock.
  *         This parameter must be a number between Min_Data = 50 and Max_Data = 432.
  * @note   You have to set the PLLSAIN parameter correctly to ensure that the VCO 
  *         output frequency is between Min_Data = 100 and Max_Data = 432 MHz.
  *
  * @param  __PLLSAIQ__ specifies the division factor for SAI clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 15.
  *           
  * @param  __PLLSAIR__ specifies the division factor for LTDC clock
  *         This parameter must be a number between Min_Data = 2 and Max_Data = 7.
  * @note   the PLLI2SR parameter is only available with STM32F427/437/429/439xx Devices  
  */
#define __HAL_RCC_PLLSAI_CONFIG(__PLLSAIN__, __PLLSAIQ__, __PLLSAIR__)                                        \
                               (RCC->PLLSAICFGR = (((__PLLSAIN__) << RCC_PLLSAICFGR_PLLSAIN_Pos)  | \
                               ((__PLLSAIQ__) << RCC_PLLSAICFGR_PLLSAIQ_Pos)                      | \
                               ((__PLLSAIR__) << RCC_PLLSAICFGR_PLLSAIR_Pos)))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx */

#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */
/*----------------------------------------------------------------------------*/

/*------------------- PLLSAI/PLLI2S Dividers Configuration -------------------*/
#if defined(STM32F413xx) || defined(STM32F423xx)
/** @brief  Macro to configure the SAI clock Divider coming from PLLI2S.
  * @note   This function must be called before enabling the PLLI2S.
  * @param  __PLLI2SDivR__ specifies the PLLI2S division factor for SAI1 clock.
  *          This parameter must be a number between 1 and 32.
  *          SAI1 clock frequency = f(PLLI2SR) / __PLLI2SDivR__ 
  */
#define __HAL_RCC_PLLI2S_PLLSAICLKDIVR_CONFIG(__PLLI2SDivR__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_PLLI2SDIVR, (__PLLI2SDivR__)-1U))

/** @brief  Macro to configure the SAI clock Divider coming from PLL.
  * @param  __PLLDivR__ specifies the PLL division factor for SAI1 clock.
  *          This parameter must be a number between 1 and 32.
  *          SAI1 clock frequency = f(PLLR) / __PLLDivR__ 
  */
#define __HAL_RCC_PLL_PLLSAICLKDIVR_CONFIG(__PLLDivR__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_PLLDIVR, ((__PLLDivR__)-1U)<<8U))                                 
#endif /* STM32F413xx || STM32F423xx */  
                                 
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx)  || defined(STM32F446xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
/** @brief  Macro to configure the SAI clock Divider coming from PLLI2S.
  * @note   This function must be called before enabling the PLLI2S.
  * @param  __PLLI2SDivQ__ specifies the PLLI2S division factor for SAI1 clock.
  *          This parameter must be a number between 1 and 32.
  *          SAI1 clock frequency = f(PLLI2SQ) / __PLLI2SDivQ__ 
  */
#define __HAL_RCC_PLLI2S_PLLSAICLKDIVQ_CONFIG(__PLLI2SDivQ__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_PLLI2SDIVQ, (__PLLI2SDivQ__)-1U))

/** @brief  Macro to configure the SAI clock Divider coming from PLLSAI.
  * @note   This function must be called before enabling the PLLSAI.
  * @param  __PLLSAIDivQ__ specifies the PLLSAI division factor for SAI1 clock .
  *         This parameter must be a number between Min_Data = 1 and Max_Data = 32.
  *         SAI1 clock frequency = f(PLLSAIQ) / __PLLSAIDivQ__  
  */
#define __HAL_RCC_PLLSAI_PLLSAICLKDIVQ_CONFIG(__PLLSAIDivQ__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_PLLSAIDIVQ, ((__PLLSAIDivQ__)-1U)<<8U))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
/** @brief  Macro to configure the LTDC clock Divider coming from PLLSAI.
  * 
  * @note   The LTDC peripheral is only available with STM32F427/437/429/439/469/479xx Devices.
  * @note   This function must be called before enabling the PLLSAI. 
  * @param  __PLLSAIDivR__ specifies the PLLSAI division factor for LTDC clock .
  *          This parameter must be a number between Min_Data = 2 and Max_Data = 16.
  *          LTDC clock frequency = f(PLLSAIR) / __PLLSAIDivR__ 
  */
#define __HAL_RCC_PLLSAI_PLLSAICLKDIVR_CONFIG(__PLLSAIDivR__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_PLLSAIDIVR, (__PLLSAIDivR__)))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */
/*----------------------------------------------------------------------------*/

/*------------------------- Peripheral Clock selection -----------------------*/
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F469xx) ||\
    defined(STM32F479xx)
/** @brief  Macro to configure the I2S clock source (I2SCLK).
  * @note   This function must be called before enabling the I2S APB clock.
  * @param  __SOURCE__ specifies the I2S clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_I2SCLKSOURCE_PLLI2S: PLLI2S clock used as I2S clock source.
  *            @arg RCC_I2SCLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin
  *                                       used as I2S clock source.
  */
#define __HAL_RCC_I2S_CONFIG(__SOURCE__) (*(__IO uint32_t *) RCC_CFGR_I2SSRC_BB = (__SOURCE__))


/** @brief  Macro to get the I2S clock source (I2SCLK).
  * @retval The clock source can be one of the following values:
  *            @arg @ref RCC_I2SCLKSOURCE_PLLI2S: PLLI2S clock used as I2S clock source.
  *            @arg @ref RCC_I2SCLKSOURCE_EXT External clock mapped on the I2S_CKIN pin
  *                                        used as I2S clock source
  */
#define __HAL_RCC_GET_I2S_SOURCE() ((uint32_t)(READ_BIT(RCC->CFGR, RCC_CFGR_I2SSRC)))
#endif /* STM32F40xxx || STM32F41xxx || STM32F42xxx || STM32F43xxx || STM32F469xx || STM32F479xx */
                                 
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
                                 
/** @brief  Macro to configure SAI1BlockA clock source selection.
  * @note   The SAI peripheral is only available with STM32F427/437/429/439/469/479xx Devices.      
  * @note   This function must be called before enabling PLLSAI, PLLI2S and  
  *         the SAI clock.
  * @param  __SOURCE__ specifies the SAI Block A clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SAIACLKSOURCE_PLLI2S: PLLI2S_Q clock divided by PLLI2SDIVQ used 
  *                                           as SAI1 Block A clock. 
  *            @arg RCC_SAIACLKSOURCE_PLLSAI: PLLISAI_Q clock divided by PLLSAIDIVQ used 
  *                                           as SAI1 Block A clock.
  *            @arg RCC_SAIACLKSOURCE_Ext: External clock mapped on the I2S_CKIN pin
  *                                        used as SAI1 Block A clock.
  */
#define __HAL_RCC_SAI_BLOCKACLKSOURCE_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_SAI1ASRC, (__SOURCE__)))

/** @brief  Macro to configure SAI1BlockB clock source selection.
  * @note   The SAI peripheral is only available with STM32F427/437/429/439/469/479xx Devices.
  * @note   This function must be called before enabling PLLSAI, PLLI2S and  
  *         the SAI clock.
  * @param  __SOURCE__ specifies the SAI Block B clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SAIBCLKSOURCE_PLLI2S: PLLI2S_Q clock divided by PLLI2SDIVQ used 
  *                                           as SAI1 Block B clock. 
  *            @arg RCC_SAIBCLKSOURCE_PLLSAI: PLLISAI_Q clock divided by PLLSAIDIVQ used 
  *                                           as SAI1 Block B clock. 
  *            @arg RCC_SAIBCLKSOURCE_Ext: External clock mapped on the I2S_CKIN pin
  *                                        used as SAI1 Block B clock.
  */
#define __HAL_RCC_SAI_BLOCKBCLKSOURCE_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_SAI1BSRC, (__SOURCE__)))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F446xx)
/** @brief  Macro to configure SAI1 clock source selection.
  * @note   This configuration is only available with STM32F446xx Devices.
  * @note   This function must be called before enabling PLL, PLLSAI, PLLI2S and  
  *         the SAI clock.
  * @param  __SOURCE__ specifies the SAI1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SAI1CLKSOURCE_PLLI2S: PLLI2S_Q clock divided by PLLI2SDIVQ used as SAI1 clock. 
  *            @arg RCC_SAI1CLKSOURCE_PLLSAI: PLLISAI_Q clock divided by PLLSAIDIVQ used as SAI1 clock.
  *            @arg RCC_SAI1CLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as SAI1 clock.  
  *            @arg RCC_SAI1CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin used as SAI1 clock.
  */
#define __HAL_RCC_SAI1_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_SAI1SRC, (__SOURCE__)))

/** @brief  Macro to Get SAI1 clock source selection.
  * @note   This configuration is only available with STM32F446xx Devices.      
  * @retval The clock source can be one of the following values:
  *            @arg RCC_SAI1CLKSOURCE_PLLI2S: PLLI2S_Q clock divided by PLLI2SDIVQ used as SAI1 clock. 
  *            @arg RCC_SAI1CLKSOURCE_PLLSAI: PLLISAI_Q clock divided by PLLSAIDIVQ used as SAI1 clock.
  *            @arg RCC_SAI1CLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as SAI1 clock.  
  *            @arg RCC_SAI1CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin used as SAI1 clock.
  */
#define __HAL_RCC_GET_SAI1_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_SAI1SRC))

/** @brief  Macro to configure SAI2 clock source selection.
  * @note   This configuration is only available with STM32F446xx Devices.      
  * @note   This function must be called before enabling PLL, PLLSAI, PLLI2S and  
  *         the SAI clock.
  * @param  __SOURCE__ specifies the SAI2 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SAI2CLKSOURCE_PLLI2S: PLLI2S_Q clock divided by PLLI2SDIVQ used as SAI2 clock. 
  *            @arg RCC_SAI2CLKSOURCE_PLLSAI: PLLISAI_Q clock divided by PLLSAIDIVQ used as SAI2 clock.
  *            @arg RCC_SAI2CLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as SAI2 clock.  
  *            @arg RCC_SAI2CLKSOURCE_PLLSRC: HSI or HSE depending from PLL Source clock used as SAI2 clock.
  */
#define __HAL_RCC_SAI2_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_SAI2SRC, (__SOURCE__)))

/** @brief  Macro to Get SAI2 clock source selection.
  * @note   This configuration is only available with STM32F446xx Devices.      
  * @retval The clock source can be one of the following values:
  *            @arg RCC_SAI2CLKSOURCE_PLLI2S: PLLI2S_Q clock divided by PLLI2SDIVQ used as SAI2 clock. 
  *            @arg RCC_SAI2CLKSOURCE_PLLSAI: PLLISAI_Q clock divided by PLLSAIDIVQ used as SAI2 clock.
  *            @arg RCC_SAI2CLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as SAI2 clock.  
  *            @arg RCC_SAI2CLKSOURCE_PLLSRC: HSI or HSE depending from PLL Source clock used as SAI2 clock.
  */
#define __HAL_RCC_GET_SAI2_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_SAI2SRC))

/** @brief  Macro to configure I2S APB1 clock source selection.
  * @note   This function must be called before enabling PLL, PLLI2S and the I2S clock.
  * @param  __SOURCE__ specifies the I2S APB1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLI2S: PLLI2S VCO output clock divided by PLLI2SR used as I2S clock. 
  *            @arg RCC_I2SAPB1CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin used as I2S APB1 clock.
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as I2S APB1 clock.  
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_I2S_APB1_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_I2S1SRC, (__SOURCE__)))

/** @brief  Macro to Get I2S APB1 clock source selection.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLI2S: PLLI2S VCO output clock divided by PLLI2SR used as I2S clock. 
  *            @arg RCC_I2SAPB1CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin used as I2S APB1 clock.
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as I2S APB1 clock.  
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_GET_I2S_APB1_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_I2S1SRC))

/** @brief  Macro to configure I2S APB2 clock source selection.
  * @note   This function must be called before enabling PLL, PLLI2S and the I2S clock.
  * @param  __SOURCE__ specifies the SAI Block A clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLI2S: PLLI2S VCO output clock divided by PLLI2SR used as I2S clock. 
  *            @arg RCC_I2SAPB2CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin used as I2S APB2 clock.
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as I2S APB2 clock.  
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_I2S_APB2_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_I2S2SRC, (__SOURCE__)))

/** @brief  Macro to Get I2S APB2 clock source selection.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLI2S: PLLI2S VCO output clock divided by PLLI2SR used as I2S clock. 
  *            @arg RCC_I2SAPB2CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin used as I2S APB2 clock.
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as I2S APB2 clock.  
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_GET_I2S_APB2_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_I2S2SRC))

/** @brief  Macro to configure the CEC clock.
  * @param  __SOURCE__ specifies the CEC clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_CECCLKSOURCE_HSI: HSI selected as CEC clock
  *            @arg RCC_CECCLKSOURCE_LSE: LSE selected as CEC clock
  */
#define __HAL_RCC_CEC_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_CECSEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the CEC clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_CECCLKSOURCE_HSI488: HSI selected as CEC clock
  *            @arg RCC_CECCLKSOURCE_LSE: LSE selected as CEC clock
  */
#define __HAL_RCC_GET_CEC_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_CECSEL))

/** @brief  Macro to configure the FMPI2C1 clock.
  * @param  __SOURCE__ specifies the FMPI2C1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_FMPI2C1CLKSOURCE_PCLK1: PCLK1 selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_SYSCLK: SYS clock selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_HSI: HSI selected as FMPI2C1 clock
  */
#define __HAL_RCC_FMPI2C1_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_FMPI2C1SEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the FMPI2C1 clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_FMPI2C1CLKSOURCE_PCLK1: PCLK1 selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_SYSCLK: SYS clock selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_HSI: HSI selected as FMPI2C1 clock
  */
#define __HAL_RCC_GET_FMPI2C1_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_FMPI2C1SEL))

/** @brief  Macro to configure the CLK48 clock.
  * @param  __SOURCE__ specifies the CLK48 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_CLK48CLKSOURCE_PLLQ: PLL VCO Output divided by PLLQ used as CLK48 clock. 
  *            @arg RCC_CLK48CLKSOURCE_PLLSAIP: PLLSAI VCO Output divided by PLLSAIP used as CLK48 clock. 
  */
#define __HAL_RCC_CLK48_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_CK48MSEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the CLK48 clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_CLK48CLKSOURCE_PLLQ: PLL VCO Output divided by PLLQ used as CLK48 clock. 
  *            @arg RCC_CLK48CLKSOURCE_PLLSAIP: PLLSAI VCO Output divided by PLLSAIP used as CLK48 clock. 
  */
#define __HAL_RCC_GET_CLK48_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_CK48MSEL))

/** @brief  Macro to configure the SDIO clock.
  * @param  __SOURCE__ specifies the SDIO clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SDIOCLKSOURCE_CLK48: CLK48 output used as SDIO clock. 
  *            @arg RCC_SDIOCLKSOURCE_SYSCLK: System clock output used as SDIO clock. 
  */
#define __HAL_RCC_SDIO_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_SDIOSEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the SDIO clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_SDIOCLKSOURCE_CLK48: CLK48 output used as SDIO clock. 
  *            @arg RCC_SDIOCLKSOURCE_SYSCLK: System clock output used as SDIO clock. 
  */
#define __HAL_RCC_GET_SDIO_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_SDIOSEL))

/** @brief  Macro to configure the SPDIFRX clock.
  * @param  __SOURCE__ specifies the SPDIFRX clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SPDIFRXCLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as SPDIFRX clock.  
  *            @arg RCC_SPDIFRXCLKSOURCE_PLLI2SP: PLLI2S VCO Output divided by PLLI2SP used as SPDIFRX clock. 
  */
#define __HAL_RCC_SPDIFRX_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_SPDIFRXSEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the SPDIFRX clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_SPDIFRXCLKSOURCE_PLLR: PLL VCO Output divided by PLLR used as SPDIFRX clock.  
  *            @arg RCC_SPDIFRXCLKSOURCE_PLLI2SP: PLLI2S VCO Output divided by PLLI2SP used as SPDIFRX clock. 
  */
#define __HAL_RCC_GET_SPDIFRX_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_SPDIFRXSEL))
#endif /* STM32F446xx */
      
#if defined(STM32F469xx) || defined(STM32F479xx)
      
/** @brief  Macro to configure the CLK48 clock.
  * @param  __SOURCE__ specifies the CLK48 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_CLK48CLKSOURCE_PLLQ: PLL VCO Output divided by PLLQ used as CLK48 clock. 
  *            @arg RCC_CLK48CLKSOURCE_PLLSAIP: PLLSAI VCO Output divided by PLLSAIP used as CLK48 clock. 
  */
#define __HAL_RCC_CLK48_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_CK48MSEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the CLK48 clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_CLK48CLKSOURCE_PLLQ: PLL VCO Output divided by PLLQ used as CLK48 clock. 
  *            @arg RCC_CLK48CLKSOURCE_PLLSAIP: PLLSAI VCO Output divided by PLLSAIP used as CLK48 clock. 
  */
#define __HAL_RCC_GET_CLK48_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_CK48MSEL))

/** @brief  Macro to configure the SDIO clock.
  * @param  __SOURCE__ specifies the SDIO clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SDIOCLKSOURCE_CLK48: CLK48 output used as SDIO clock. 
  *            @arg RCC_SDIOCLKSOURCE_SYSCLK: System clock output used as SDIO clock. 
  */
#define __HAL_RCC_SDIO_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_SDIOSEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the SDIO clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_SDIOCLKSOURCE_CLK48: CLK48 output used as SDIO clock. 
  *            @arg RCC_SDIOCLKSOURCE_SYSCLK: System clock output used as SDIO clock. 
  */
#define __HAL_RCC_GET_SDIO_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_SDIOSEL))  
      
/** @brief  Macro to configure the DSI clock.
  * @param  __SOURCE__ specifies the DSI clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_DSICLKSOURCE_PLLR: PLLR output used as DSI clock. 
  *            @arg RCC_DSICLKSOURCE_DSIPHY: DSI-PHY output used as DSI clock. 
  */
#define __HAL_RCC_DSI_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_DSISEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the DSI clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_DSICLKSOURCE_PLLR: PLLR output used as DSI clock. 
  *            @arg RCC_DSICLKSOURCE_DSIPHY: DSI-PHY output used as DSI clock. 
  */
#define __HAL_RCC_GET_DSI_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_DSISEL))       
      
#endif /* STM32F469xx || STM32F479xx */

#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) ||\
    defined(STM32F413xx) || defined(STM32F423xx)
 /** @brief  Macro to configure the DFSDM1 clock.
  * @param  __DFSDM1_CLKSOURCE__ specifies the DFSDM1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_DFSDM1CLKSOURCE_PCLK2: PCLK2 clock used as kernel clock. 
  *            @arg RCC_DFSDM1CLKSOURCE_SYSCLK: System clock used as kernel clock.
  * @retval None
  */
#define __HAL_RCC_DFSDM1_CONFIG(__DFSDM1_CLKSOURCE__)  MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_CKDFSDM1SEL, (__DFSDM1_CLKSOURCE__))

/** @brief  Macro to get the DFSDM1 clock source.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_DFSDM1CLKSOURCE_PCLK2: PCLK2 clock used as kernel clock. 
  *            @arg RCC_DFSDM1CLKSOURCE_SYSCLK: System clock used as kernel clock.
  */
#define __HAL_RCC_GET_DFSDM1_SOURCE() ((uint32_t)(READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_CKDFSDM1SEL)))

/** @brief  Macro to configure DFSDM1 Audio clock source selection.
  * @note   This configuration is only available with STM32F412Zx/STM32F412Vx/STM32F412Rx/STM32F412Cx/
            STM32F413xx/STM32F423xx Devices.
  * @param  __SOURCE__ specifies the DFSDM1 Audio clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_DFSDM1AUDIOCLKSOURCE_I2S1: CK_I2S_PCLK1 selected as audio clock
  *            @arg RCC_DFSDM1AUDIOCLKSOURCE_I2S2: CK_I2S_PCLK2 selected as audio clock
  */
#define __HAL_RCC_DFSDM1AUDIO_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_CKDFSDM1ASEL, (__SOURCE__)))

/** @brief  Macro to Get DFSDM1 Audio clock source selection.
  * @note   This configuration is only available with STM32F412Zx/STM32F412Vx/STM32F412Rx/STM32F412Cx/
            STM32F413xx/STM32F423xx Devices.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_DFSDM1AUDIOCLKSOURCE_I2S1: CK_I2S_PCLK1 selected as audio clock
  *            @arg RCC_DFSDM1AUDIOCLKSOURCE_I2S2: CK_I2S_PCLK2 selected as audio clock
  */
#define __HAL_RCC_GET_DFSDM1AUDIO_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_CKDFSDM1ASEL))

#if defined(STM32F413xx) || defined(STM32F423xx)
 /** @brief  Macro to configure the DFSDM2 clock.
  * @param  __DFSDM2_CLKSOURCE__ specifies the DFSDM1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_DFSDM2CLKSOURCE_PCLK2: PCLK2 clock used as kernel clock. 
  *            @arg RCC_DFSDM2CLKSOURCE_SYSCLK: System clock used as kernel clock.
  * @retval None
  */
#define __HAL_RCC_DFSDM2_CONFIG(__DFSDM2_CLKSOURCE__)  MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_CKDFSDM1SEL, (__DFSDM2_CLKSOURCE__))

/** @brief  Macro to get the DFSDM2 clock source.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_DFSDM2CLKSOURCE_PCLK2: PCLK2 clock used as kernel clock. 
  *            @arg RCC_DFSDM2CLKSOURCE_SYSCLK: System clock used as kernel clock.
  */
#define __HAL_RCC_GET_DFSDM2_SOURCE() ((uint32_t)(READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_CKDFSDM1SEL)))

/** @brief  Macro to configure DFSDM1 Audio clock source selection.
  * @note   This configuration is only available with STM32F413xx/STM32F423xx Devices.
  * @param  __SOURCE__ specifies the DFSDM2 Audio clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_DFSDM2AUDIOCLKSOURCE_I2S1: CK_I2S_PCLK1 selected as audio clock
  *            @arg RCC_DFSDM2AUDIOCLKSOURCE_I2S2: CK_I2S_PCLK2 selected as audio clock
  */
#define __HAL_RCC_DFSDM2AUDIO_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_CKDFSDM2ASEL, (__SOURCE__)))

/** @brief  Macro to Get DFSDM2 Audio clock source selection.
  * @note   This configuration is only available with STM32F413xx/STM32F423xx Devices.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_DFSDM2AUDIOCLKSOURCE_I2S1: CK_I2S_PCLK1 selected as audio clock
  *            @arg RCC_DFSDM2AUDIOCLKSOURCE_I2S2: CK_I2S_PCLK2 selected as audio clock
  */
#define __HAL_RCC_GET_DFSDM2AUDIO_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_CKDFSDM2ASEL))
      
/** @brief  Macro to configure SAI1BlockA clock source selection.
  * @note   The SAI peripheral is only available with STM32F413xx/STM32F423xx Devices.      
  * @note   This function must be called before enabling PLLSAI, PLLI2S and  
  *         the SAI clock.
  * @param  __SOURCE__ specifies the SAI Block A clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SAIACLKSOURCE_PLLI2SR: PLLI2S_R clock divided (R2) used as SAI1 Block A clock.
  *            @arg RCC_SAIACLKSOURCE_EXT: External clock mapped on the I2S_CKIN pinused as SAI1 Block A clock.
  *            @arg RCC_SAIACLKSOURCE_PLLR: PLL_R clock divided (R1) used as SAI1 Block A clock.
  *            @arg RCC_SAIACLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_SAI_BLOCKACLKSOURCE_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_SAI1ASRC, (__SOURCE__)))
      
/** @brief  Macro to Get SAI1 BlockA clock source selection.
  * @note   This configuration is only available with STM32F413xx/STM32F423xx Devices.      
  * @retval The clock source can be one of the following values:
  *            @arg RCC_SAIACLKSOURCE_PLLI2SR: PLLI2S_R clock divided (R2) used as SAI1 Block A clock.
  *            @arg RCC_SAIACLKSOURCE_EXT: External clock mapped on the I2S_CKIN pinused as SAI1 Block A clock.
  *            @arg RCC_SAIACLKSOURCE_PLLR: PLL_R clock divided (R1) used as SAI1 Block A clock.
  *            @arg RCC_SAIACLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_GET_SAI_BLOCKA_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_SAI1ASRC))

/** @brief  Macro to configure SAI1 BlockB clock source selection.
  * @note   The SAI peripheral is only available with STM32F413xx/STM32F423xx Devices.
  * @note   This function must be called before enabling PLLSAI, PLLI2S and  
  *         the SAI clock.
  * @param  __SOURCE__ specifies the SAI Block B clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SAIBCLKSOURCE_PLLI2SR: PLLI2S_R clock divided (R2) used as SAI1 Block A clock.
  *            @arg RCC_SAIBCLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin used as SAI1 Block A clock.
  *            @arg RCC_SAIBCLKSOURCE_PLLR: PLL_R clock divided (R1) used as SAI1 Block A clock.
  *            @arg RCC_SAIBCLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_SAI_BLOCKBCLKSOURCE_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_SAI1BSRC, (__SOURCE__)))
      
/** @brief  Macro to Get SAI1 BlockB clock source selection.
  * @note   This configuration is only available with STM32F413xx/STM32F423xx Devices.      
  * @retval The clock source can be one of the following values:
  *            @arg RCC_SAIBCLKSOURCE_PLLI2SR: PLLI2S_R clock divided (R2) used as SAI1 Block A clock.
  *            @arg RCC_SAIBCLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin used as SAI1 Block A clock.
  *            @arg RCC_SAIBCLKSOURCE_PLLR: PLL_R clock divided (R1) used as SAI1 Block A clock.
  *            @arg RCC_SAIBCLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_GET_SAI_BLOCKB_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_SAI1BSRC))

/** @brief  Macro to configure the LPTIM1 clock.
  * @param  __SOURCE__ specifies the LPTIM1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_LPTIM1CLKSOURCE_PCLK1: PCLK selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_HSI: HSI clock selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_LSI: LSI selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_LSE: LSE selected as LPTIM1 clock
  */
#define __HAL_RCC_LPTIM1_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_LPTIM1SEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the LPTIM1 clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_LPTIM1CLKSOURCE_PCLK1: PCLK selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_HSI: HSI clock selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_LSI: LSI selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_LSE: LSE selected as LPTIM1 clock
  */
#define __HAL_RCC_GET_LPTIM1_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_LPTIM1SEL))      
#endif /* STM32F413xx || STM32F423xx */
      
/** @brief  Macro to configure I2S APB1 clock source selection.
  * @param  __SOURCE__ specifies the I2S APB1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLI2S: PLLI2S VCO output clock divided by PLLI2SR.
  *            @arg RCC_I2SAPB1CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin.
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLR: PLL VCO Output divided by PLLR.
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_I2S_APB1_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_I2S1SRC, (__SOURCE__)))

/** @brief  Macro to Get I2S APB1 clock source selection.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLI2S: PLLI2S VCO output clock divided by PLLI2SR.
  *            @arg RCC_I2SAPB1CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin.
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLR: PLL VCO Output divided by PLLR.
  *            @arg RCC_I2SAPB1CLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_GET_I2S_APB1_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_I2S1SRC))

/** @brief  Macro to configure I2S APB2 clock source selection.
  * @param  __SOURCE__ specifies the I2S APB2 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLI2S: PLLI2S VCO output clock divided by PLLI2SR.
  *            @arg RCC_I2SAPB2CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin.
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLR: PLL VCO Output divided by PLLR.
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_I2S_APB2_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_I2S2SRC, (__SOURCE__)))

/** @brief  Macro to Get I2S APB2 clock source selection.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLI2S: PLLI2S VCO output clock divided by PLLI2SR.
  *            @arg RCC_I2SAPB2CLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin.
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLR: PLL VCO Output divided by PLLR.
  *            @arg RCC_I2SAPB2CLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  */
#define __HAL_RCC_GET_I2S_APB2_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_I2S2SRC))

/** @brief  Macro to configure the PLL I2S clock source (PLLI2SCLK).
  * @note   This macro must be called before enabling the I2S APB clock.
  * @param  __SOURCE__ specifies the I2S clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_PLLI2SCLKSOURCE_PLLSRC: HSI or HSE depending from PLL source Clock.
  *            @arg RCC_PLLI2SCLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin
  *                                       used as I2S clock source.
  */
#define __HAL_RCC_PLL_I2S_CONFIG(__SOURCE__) (*(__IO uint32_t *) RCC_PLLI2SCFGR_PLLI2SSRC_BB = (__SOURCE__))
      
/** @brief  Macro to configure the FMPI2C1 clock.
  * @param  __SOURCE__ specifies the FMPI2C1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_FMPI2C1CLKSOURCE_PCLK1: PCLK1 selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_SYSCLK: SYS clock selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_HSI: HSI selected as FMPI2C1 clock
  */
#define __HAL_RCC_FMPI2C1_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_FMPI2C1SEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the FMPI2C1 clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_FMPI2C1CLKSOURCE_PCLK1: PCLK1 selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_SYSCLK: SYS clock selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_HSI: HSI selected as FMPI2C1 clock
  */
#define __HAL_RCC_GET_FMPI2C1_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_FMPI2C1SEL))

/** @brief  Macro to configure the CLK48 clock.
  * @param  __SOURCE__ specifies the CLK48 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_CLK48CLKSOURCE_PLLQ: PLL VCO Output divided by PLLQ used as CLK48 clock. 
  *            @arg RCC_CLK48CLKSOURCE_PLLI2SQ: PLLI2S VCO Output divided by PLLI2SQ used as CLK48 clock.
  */
#define __HAL_RCC_CLK48_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_CK48MSEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the CLK48 clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_CLK48CLKSOURCE_PLLQ: PLL VCO Output divided by PLLQ used as CLK48 clock. 
  *            @arg RCC_CLK48CLKSOURCE_PLLI2SQ: PLLI2S VCO Output divided by PLLI2SQ used as CLK48 clock
  */
#define __HAL_RCC_GET_CLK48_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_CK48MSEL))

/** @brief  Macro to configure the SDIO clock.
  * @param  __SOURCE__ specifies the SDIO clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_SDIOCLKSOURCE_CLK48: CLK48 output used as SDIO clock. 
  *            @arg RCC_SDIOCLKSOURCE_SYSCLK: System clock output used as SDIO clock. 
  */
#define __HAL_RCC_SDIO_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_SDIOSEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the SDIO clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_SDIOCLKSOURCE_CLK48: CLK48 output used as SDIO clock. 
  *            @arg RCC_SDIOCLKSOURCE_SYSCLK: System clock output used as SDIO clock. 
  */
#define __HAL_RCC_GET_SDIO_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_SDIOSEL))

#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
/** @brief  Macro to configure I2S clock source selection.
  * @param  __SOURCE__ specifies the I2S clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_I2SAPBCLKSOURCE_PLLR: PLL VCO output clock divided by PLLR.
  *            @arg RCC_I2SAPBCLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin.
  *            @arg RCC_I2SAPBCLKSOURCE_PLLSRC: HSI/HSE depends on PLLSRC.
  */
#define __HAL_RCC_I2S_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR, RCC_DCKCFGR_I2SSRC, (__SOURCE__)))

/** @brief  Macro to Get I2S clock source selection.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_I2SAPBCLKSOURCE_PLLR: PLL VCO output clock divided by PLLR.
  *            @arg RCC_I2SAPBCLKSOURCE_EXT: External clock mapped on the I2S_CKIN pin.
  *            @arg RCC_I2SAPBCLKSOURCE_PLLSRC: HSI/HSE depends on PLLSRC.
  */
#define __HAL_RCC_GET_I2S_SOURCE() (READ_BIT(RCC->DCKCFGR, RCC_DCKCFGR_I2SSRC))

/** @brief  Macro to configure the FMPI2C1 clock.
  * @param  __SOURCE__ specifies the FMPI2C1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_FMPI2C1CLKSOURCE_PCLK1: PCLK1 selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_SYSCLK: SYS clock selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_HSI: HSI selected as FMPI2C1 clock
  */
#define __HAL_RCC_FMPI2C1_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_FMPI2C1SEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the FMPI2C1 clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_FMPI2C1CLKSOURCE_PCLK1: PCLK1 selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_SYSCLK: SYS clock selected as FMPI2C1 clock
  *            @arg RCC_FMPI2C1CLKSOURCE_HSI: HSI selected as FMPI2C1 clock
  */
#define __HAL_RCC_GET_FMPI2C1_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_FMPI2C1SEL))

/** @brief  Macro to configure the LPTIM1 clock.
  * @param  __SOURCE__ specifies the LPTIM1 clock source.
  *         This parameter can be one of the following values:
  *            @arg RCC_LPTIM1CLKSOURCE_PCLK1: PCLK1 selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_HSI: HSI clock selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_LSI: LSI selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_LSE: LSE selected as LPTIM1 clock
  */
#define __HAL_RCC_LPTIM1_CONFIG(__SOURCE__) (MODIFY_REG(RCC->DCKCFGR2, RCC_DCKCFGR2_LPTIM1SEL, (uint32_t)(__SOURCE__)))

/** @brief  Macro to Get the LPTIM1 clock.
  * @retval The clock source can be one of the following values:
  *            @arg RCC_LPTIM1CLKSOURCE_PCLK1: PCLK1 selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_HSI: HSI clock selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_LSI: LSI selected as LPTIM1 clock
  *            @arg RCC_LPTIM1CLKSOURCE_LSE: LSE selected as LPTIM1 clock
  */
#define __HAL_RCC_GET_LPTIM1_SOURCE() (READ_BIT(RCC->DCKCFGR2, RCC_DCKCFGR2_LPTIM1SEL))
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */
      
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) ||\
    defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/** @brief  Macro to configure the Timers clocks prescalers 
  * @note   This feature is only available with STM32F429x/439x Devices.  
  * @param  __PRESC__  specifies the Timers clocks prescalers selection
  *         This parameter can be one of the following values:
  *            @arg RCC_TIMPRES_DESACTIVATED: The Timers kernels clocks prescaler is 
  *                 equal to HPRE if PPREx is corresponding to division by 1 or 2, 
  *                 else it is equal to [(HPRE * PPREx) / 2] if PPREx is corresponding to 
  *                 division by 4 or more.       
  *            @arg RCC_TIMPRES_ACTIVATED: The Timers kernels clocks prescaler is 
  *                 equal to HPRE if PPREx is corresponding to division by 1, 2 or 4, 
  *                 else it is equal to [(HPRE * PPREx) / 4] if PPREx is corresponding 
  *                 to division by 8 or more.
  */     
#define __HAL_RCC_TIMCLKPRESCALER(__PRESC__) (*(__IO uint32_t *) RCC_DCKCFGR_TIMPRE_BB = (__PRESC__))

#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx) || STM32F401xC || STM32F401xE || STM32F410xx || STM32F411xE ||\
          STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx  || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx ||\
          STM32F423xx */

/*----------------------------------------------------------------------------*/

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
/** @brief Enable PLLSAI_RDY interrupt.
  */
#define __HAL_RCC_PLLSAI_ENABLE_IT() (RCC->CIR |= (RCC_CIR_PLLSAIRDYIE))

/** @brief Disable PLLSAI_RDY interrupt.
  */
#define __HAL_RCC_PLLSAI_DISABLE_IT() (RCC->CIR &= ~(RCC_CIR_PLLSAIRDYIE))

/** @brief Clear the PLLSAI RDY interrupt pending bits.
  */
#define __HAL_RCC_PLLSAI_CLEAR_IT() (RCC->CIR |= (RCC_CIR_PLLSAIRDYF))

/** @brief Check the PLLSAI RDY interrupt has occurred or not.
  * @retval The new state (TRUE or FALSE).
  */
#define __HAL_RCC_PLLSAI_GET_IT() ((RCC->CIR & (RCC_CIR_PLLSAIRDYIE)) == (RCC_CIR_PLLSAIRDYIE))

/** @brief  Check PLLSAI RDY flag is set or not.
  * @retval The new state (TRUE or FALSE).
  */
#define __HAL_RCC_PLLSAI_GET_FLAG() ((RCC->CR & (RCC_CR_PLLSAIRDY)) == (RCC_CR_PLLSAIRDY))

#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
/** @brief  Macros to enable or disable the RCC MCO1 feature.
  */
#define __HAL_RCC_MCO1_ENABLE() (*(__IO uint32_t *) RCC_CFGR_MCO1EN_BB = ENABLE)
#define __HAL_RCC_MCO1_DISABLE() (*(__IO uint32_t *) RCC_CFGR_MCO1EN_BB = DISABLE)

/** @brief  Macros to enable or disable the RCC MCO2 feature.
  */
#define __HAL_RCC_MCO2_ENABLE() (*(__IO uint32_t *) RCC_CFGR_MCO2EN_BB = ENABLE)
#define __HAL_RCC_MCO2_DISABLE() (*(__IO uint32_t *) RCC_CFGR_MCO2EN_BB = DISABLE)

#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup RCCEx_Exported_Functions
  *  @{
  */

/** @addtogroup RCCEx_Exported_Functions_Group1
  *  @{
  */
HAL_StatusTypeDef HAL_RCCEx_PeriphCLKConfig(RCC_PeriphCLKInitTypeDef  *PeriphClkInit);
void HAL_RCCEx_GetPeriphCLKConfig(RCC_PeriphCLKInitTypeDef  *PeriphClkInit);

uint32_t HAL_RCCEx_GetPeriphCLKFreq(uint32_t PeriphClk);

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F411xE) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)
void HAL_RCCEx_SelectLSEMode(uint8_t Mode);
#endif /* STM32F410xx || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */
#if defined(RCC_PLLI2S_SUPPORT)
HAL_StatusTypeDef HAL_RCCEx_EnablePLLI2S(RCC_PLLI2SInitTypeDef  *PLLI2SInit);
HAL_StatusTypeDef HAL_RCCEx_DisablePLLI2S(void);
#endif /* RCC_PLLI2S_SUPPORT */
#if defined(RCC_PLLSAI_SUPPORT)
HAL_StatusTypeDef HAL_RCCEx_EnablePLLSAI(RCC_PLLSAIInitTypeDef  *PLLSAIInit);
HAL_StatusTypeDef HAL_RCCEx_DisablePLLSAI(void);
#endif /* RCC_PLLSAI_SUPPORT */
/**
  * @}
  */ 

/**
  * @}
  */
/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup RCCEx_Private_Constants RCCEx Private Constants
  * @{
  */

/** @defgroup RCCEx_BitAddress_AliasRegion RCC BitAddress AliasRegion
  * @brief RCC registers bit address in the alias region
  * @{
  */
/* --- CR Register ---*/  
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
/* Alias word address of PLLSAION bit */
#define RCC_PLLSAION_BIT_NUMBER       0x1CU
#define RCC_CR_PLLSAION_BB            (PERIPH_BB_BASE + (RCC_CR_OFFSET * 32U) + (RCC_PLLSAION_BIT_NUMBER * 4U))

#define PLLSAI_TIMEOUT_VALUE          2U  /* Timeout value fixed to 2 ms  */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || \
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx) || \
    defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || \
    defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/* Alias word address of PLLI2SON bit */
#define RCC_PLLI2SON_BIT_NUMBER    0x1AU
#define RCC_CR_PLLI2SON_BB         (PERIPH_BB_BASE + (RCC_CR_OFFSET * 32U) + (RCC_PLLI2SON_BIT_NUMBER * 4U))
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx ||
          STM32F401xC || STM32F401xE || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx || STM32F412Vx ||
          STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

/* --- DCKCFGR Register ---*/
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F401xC) ||\
    defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) ||\
    defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/* Alias word address of TIMPRE bit */
#define RCC_DCKCFGR_OFFSET            (RCC_OFFSET + 0x8CU)
#define RCC_TIMPRE_BIT_NUMBER          0x18U
#define RCC_DCKCFGR_TIMPRE_BB         (PERIPH_BB_BASE + (RCC_DCKCFGR_OFFSET * 32U) + (RCC_TIMPRE_BIT_NUMBER * 4U))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F410xx || STM32F401xC ||\
          STM32F401xE || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

/* --- CFGR Register ---*/
#define RCC_CFGR_OFFSET            (RCC_OFFSET + 0x08U)
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || \
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx) || \
    defined(STM32F469xx) || defined(STM32F479xx)
/* Alias word address of I2SSRC bit */
#define RCC_I2SSRC_BIT_NUMBER      0x17U
#define RCC_CFGR_I2SSRC_BB         (PERIPH_BB_BASE + (RCC_CFGR_OFFSET * 32U) + (RCC_I2SSRC_BIT_NUMBER * 4U))
      
#define PLLI2S_TIMEOUT_VALUE       2U  /* Timeout value fixed to 2 ms  */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx ||
          STM32F401xC || STM32F401xE || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx */
      
#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) ||\
    defined(STM32F413xx) || defined(STM32F423xx)
/* --- PLLI2SCFGR Register ---*/
#define RCC_PLLI2SCFGR_OFFSET         (RCC_OFFSET + 0x84U)
/* Alias word address of PLLI2SSRC bit */
#define RCC_PLLI2SSRC_BIT_NUMBER      0x16U
#define RCC_PLLI2SCFGR_PLLI2SSRC_BB         (PERIPH_BB_BASE + (RCC_PLLI2SCFGR_OFFSET * 32U) + (RCC_PLLI2SSRC_BIT_NUMBER * 4U))
      
#define PLLI2S_TIMEOUT_VALUE          2U  /* Timeout value fixed to 2 ms */
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx | STM32F423xx */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
/* Alias word address of MCO1EN bit */
#define RCC_MCO1EN_BIT_NUMBER      0x8U
#define RCC_CFGR_MCO1EN_BB         (PERIPH_BB_BASE + (RCC_CFGR_OFFSET * 32U) + (RCC_MCO1EN_BIT_NUMBER * 4U))

/* Alias word address of MCO2EN bit */
#define RCC_MCO2EN_BIT_NUMBER      0x9U
#define RCC_CFGR_MCO2EN_BB         (PERIPH_BB_BASE + (RCC_CFGR_OFFSET * 32U) + (RCC_MCO2EN_BIT_NUMBER * 4U))
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

#define PLL_TIMEOUT_VALUE          2U  /* 2 ms */
/**
  * @}
  */

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup RCCEx_Private_Macros RCCEx Private Macros
  * @{
  */
/** @defgroup RCCEx_IS_RCC_Definitions RCC Private macros to check input parameters
  * @{
  */
#define IS_RCC_PLLN_VALUE(VALUE) ((50U <= (VALUE)) && ((VALUE) <= 432U))
#define IS_RCC_PLLI2SN_VALUE(VALUE) ((50U <= (VALUE)) && ((VALUE) <= 432U))
      
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx)
#define IS_RCC_PERIPHCLOCK(SELECTION) ((1U <= (SELECTION)) && ((SELECTION) <= 0x0000007FU))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx)|| defined(STM32F417xx) 
#define IS_RCC_PERIPHCLOCK(SELECTION) ((1U <= (SELECTION)) && ((SELECTION) <= 0x00000007U))
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx */

#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) 
#define IS_RCC_PERIPHCLOCK(SELECTION) ((1U <= (SELECTION)) && ((SELECTION) <= 0x0000000FU))
#endif /* STM32F401xC || STM32F401xE || STM32F411xE */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define IS_RCC_PERIPHCLOCK(SELECTION) ((1U <= (SELECTION)) && ((SELECTION) <= 0x0000001FU))
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

#if defined(STM32F446xx)
#define IS_RCC_PERIPHCLOCK(SELECTION) ((1U <= (SELECTION)) && ((SELECTION) <= 0x00000FFFU))
#endif /* STM32F446xx */

#if defined(STM32F469xx) || defined(STM32F479xx)
#define IS_RCC_PERIPHCLOCK(SELECTION) ((1U <= (SELECTION)) && ((SELECTION) <= 0x000001FFU))
#endif /* STM32F469xx || STM32F479xx */

#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx)
#define IS_RCC_PERIPHCLOCK(SELECTION) ((1U <= (SELECTION)) && ((SELECTION) <= 0x000003FFU))
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */
      
#if defined(STM32F413xx) || defined(STM32F423xx)
#define IS_RCC_PERIPHCLOCK(SELECTION) ((1U <= (SELECTION)) && ((SELECTION) <= 0x00007FFFU))
#endif /* STM32F413xx || STM32F423xx */
      
#define IS_RCC_PLLI2SR_VALUE(VALUE) ((2U <= (VALUE)) && ((VALUE) <= 7U))

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define IS_RCC_PLLI2SQ_VALUE(VALUE)     ((2U <= (VALUE)) && ((VALUE) <= 15U))

#define IS_RCC_PLLSAIN_VALUE(VALUE)     ((50U <= (VALUE)) && ((VALUE) <= 432U))

#define IS_RCC_PLLSAIQ_VALUE(VALUE)     ((2U <= (VALUE)) && ((VALUE) <= 15U))

#define IS_RCC_PLLSAIR_VALUE(VALUE)     ((2U <= (VALUE)) && ((VALUE) <= 7U))

#define IS_RCC_PLLSAI_DIVQ_VALUE(VALUE) ((1U <= (VALUE)) && ((VALUE) <= 32U))

#define IS_RCC_PLLI2S_DIVQ_VALUE(VALUE) ((1U <= (VALUE)) && ((VALUE) <= 32U))

#define IS_RCC_PLLSAI_DIVR_VALUE(VALUE) (((VALUE) == RCC_PLLSAIDIVR_2)  ||\
                                         ((VALUE) == RCC_PLLSAIDIVR_4)  ||\
                                         ((VALUE) == RCC_PLLSAIDIVR_8)  ||\
                                         ((VALUE) == RCC_PLLSAIDIVR_16))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */

#if defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || \
    defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
#define IS_RCC_PLLI2SM_VALUE(VALUE)   ((2U <= (VALUE)) && ((VALUE) <= 63U))
 
#define IS_RCC_LSE_MODE(MODE)           (((MODE) == RCC_LSE_LOWPOWER_MODE) ||\
                                         ((MODE) == RCC_LSE_HIGHDRIVE_MODE))
#endif /* STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx  */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define IS_RCC_PLLR_VALUE(VALUE) ((2U <= (VALUE)) && ((VALUE) <= 7U))

#define IS_RCC_LSE_MODE(MODE)           (((MODE) == RCC_LSE_LOWPOWER_MODE) ||\
                                         ((MODE) == RCC_LSE_HIGHDRIVE_MODE))

#define IS_RCC_FMPI2C1CLKSOURCE(SOURCE)   (((SOURCE) == RCC_FMPI2C1CLKSOURCE_PCLK1)    ||\
                                           ((SOURCE) == RCC_FMPI2C1CLKSOURCE_SYSCLK) ||\
                                           ((SOURCE) == RCC_FMPI2C1CLKSOURCE_HSI))

#define IS_RCC_LPTIM1CLKSOURCE(SOURCE)   (((SOURCE) == RCC_LPTIM1CLKSOURCE_PCLK1) ||\
                                          ((SOURCE) == RCC_LPTIM1CLKSOURCE_HSI) ||\
                                          ((SOURCE) == RCC_LPTIM1CLKSOURCE_LSI) ||\
                                          ((SOURCE) == RCC_LPTIM1CLKSOURCE_LSE))

#define IS_RCC_I2SAPBCLKSOURCE(SOURCE)      (((SOURCE) == RCC_I2SAPBCLKSOURCE_PLLR)    ||\
                                             ((SOURCE) == RCC_I2SAPBCLKSOURCE_EXT)    ||\
                                             ((SOURCE) == RCC_I2SAPBCLKSOURCE_PLLSRC))
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

#if defined(STM32F446xx)
#define IS_RCC_PLLR_VALUE(VALUE) ((2U <= (VALUE)) && ((VALUE) <= 7U))
  
#define IS_RCC_PLLI2SP_VALUE(VALUE)       (((VALUE) == RCC_PLLI2SP_DIV2) ||\
                                           ((VALUE) == RCC_PLLI2SP_DIV4) ||\
                                           ((VALUE) == RCC_PLLI2SP_DIV6) ||\
                                           ((VALUE) == RCC_PLLI2SP_DIV8))

#define IS_RCC_PLLSAIM_VALUE(VALUE)       ((VALUE) <= 63U)
  
#define IS_RCC_PLLSAIP_VALUE(VALUE)       (((VALUE) == RCC_PLLSAIP_DIV2) ||\
                                           ((VALUE) == RCC_PLLSAIP_DIV4) ||\
                                           ((VALUE) == RCC_PLLSAIP_DIV6) ||\
                                           ((VALUE) == RCC_PLLSAIP_DIV8))

#define IS_RCC_SAI1CLKSOURCE(SOURCE)      (((SOURCE) == RCC_SAI1CLKSOURCE_PLLSAI) ||\
                                           ((SOURCE) == RCC_SAI1CLKSOURCE_PLLI2S) ||\
                                           ((SOURCE) == RCC_SAI1CLKSOURCE_PLLR)   ||\
                                           ((SOURCE) == RCC_SAI1CLKSOURCE_EXT))

#define IS_RCC_SAI2CLKSOURCE(SOURCE)      (((SOURCE) == RCC_SAI2CLKSOURCE_PLLSAI) ||\
                                           ((SOURCE) == RCC_SAI2CLKSOURCE_PLLI2S) ||\
                                           ((SOURCE) == RCC_SAI2CLKSOURCE_PLLR)   ||\
                                           ((SOURCE) == RCC_SAI2CLKSOURCE_PLLSRC))
 
#define IS_RCC_I2SAPB1CLKSOURCE(SOURCE)   (((SOURCE) == RCC_I2SAPB1CLKSOURCE_PLLI2S) ||\
                                           ((SOURCE) == RCC_I2SAPB1CLKSOURCE_EXT)    ||\
                                           ((SOURCE) == RCC_I2SAPB1CLKSOURCE_PLLR)   ||\
                                           ((SOURCE) == RCC_I2SAPB1CLKSOURCE_PLLSRC))
                                              
 #define IS_RCC_I2SAPB2CLKSOURCE(SOURCE)  (((SOURCE) == RCC_I2SAPB2CLKSOURCE_PLLI2S) ||\
                                           ((SOURCE) == RCC_I2SAPB2CLKSOURCE_EXT)    ||\
                                           ((SOURCE) == RCC_I2SAPB2CLKSOURCE_PLLR)   ||\
                                           ((SOURCE) == RCC_I2SAPB2CLKSOURCE_PLLSRC))

#define IS_RCC_FMPI2C1CLKSOURCE(SOURCE)   (((SOURCE) == RCC_FMPI2C1CLKSOURCE_PCLK1)    ||\
                                           ((SOURCE) == RCC_FMPI2C1CLKSOURCE_SYSCLK) ||\
                                           ((SOURCE) == RCC_FMPI2C1CLKSOURCE_HSI))

#define IS_RCC_CECCLKSOURCE(SOURCE)       (((SOURCE) == RCC_CECCLKSOURCE_HSI)   ||\
                                           ((SOURCE) == RCC_CECCLKSOURCE_LSE))

#define IS_RCC_CLK48CLKSOURCE(SOURCE)      (((SOURCE) == RCC_CLK48CLKSOURCE_PLLQ) ||\
                                            ((SOURCE) == RCC_CLK48CLKSOURCE_PLLSAIP))

#define IS_RCC_SDIOCLKSOURCE(SOURCE)      (((SOURCE) == RCC_SDIOCLKSOURCE_CLK48) ||\
                                           ((SOURCE) == RCC_SDIOCLKSOURCE_SYSCLK))

#define IS_RCC_SPDIFRXCLKSOURCE(SOURCE)   (((SOURCE) == RCC_SPDIFRXCLKSOURCE_PLLR) ||\
                                           ((SOURCE) == RCC_SPDIFRXCLKSOURCE_PLLI2SP))  
#endif /* STM32F446xx */

#if defined(STM32F469xx) || defined(STM32F479xx)
#define IS_RCC_PLLR_VALUE(VALUE)            ((2U <= (VALUE)) && ((VALUE) <= 7U))

#define IS_RCC_PLLSAIP_VALUE(VALUE)         (((VALUE) == RCC_PLLSAIP_DIV2) ||\
                                             ((VALUE) == RCC_PLLSAIP_DIV4) ||\
                                             ((VALUE) == RCC_PLLSAIP_DIV6) ||\
                                             ((VALUE) == RCC_PLLSAIP_DIV8))
 
#define IS_RCC_CLK48CLKSOURCE(SOURCE)        (((SOURCE) == RCC_CLK48CLKSOURCE_PLLQ) ||\
                                              ((SOURCE) == RCC_CLK48CLKSOURCE_PLLSAIP))

#define IS_RCC_SDIOCLKSOURCE(SOURCE)        (((SOURCE) == RCC_SDIOCLKSOURCE_CLK48) ||\
                                             ((SOURCE) == RCC_SDIOCLKSOURCE_SYSCLK))

#define IS_RCC_DSIBYTELANECLKSOURCE(SOURCE) (((SOURCE) == RCC_DSICLKSOURCE_PLLR)  ||\
                                             ((SOURCE) == RCC_DSICLKSOURCE_DSIPHY))

#define IS_RCC_LSE_MODE(MODE)               (((MODE) == RCC_LSE_LOWPOWER_MODE) ||\
                                             ((MODE) == RCC_LSE_HIGHDRIVE_MODE))
#endif /* STM32F469xx || STM32F479xx */

#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) ||\
    defined(STM32F413xx) || defined(STM32F423xx)
#define IS_RCC_PLLI2SQ_VALUE(VALUE) ((2U <= (VALUE)) && ((VALUE) <= 15U))
    
#define IS_RCC_PLLR_VALUE(VALUE) ((2U <= (VALUE)) && ((VALUE) <= 7U))

#define IS_RCC_PLLI2SCLKSOURCE(__SOURCE__) (((__SOURCE__) == RCC_PLLI2SCLKSOURCE_PLLSRC) || \
                                            ((__SOURCE__) == RCC_PLLI2SCLKSOURCE_EXT))
 
#define IS_RCC_I2SAPB1CLKSOURCE(SOURCE)   (((SOURCE) == RCC_I2SAPB1CLKSOURCE_PLLI2S) ||\
                                           ((SOURCE) == RCC_I2SAPB1CLKSOURCE_EXT)    ||\
                                           ((SOURCE) == RCC_I2SAPB1CLKSOURCE_PLLR)   ||\
                                           ((SOURCE) == RCC_I2SAPB1CLKSOURCE_PLLSRC))
                                              
 #define IS_RCC_I2SAPB2CLKSOURCE(SOURCE)  (((SOURCE) == RCC_I2SAPB2CLKSOURCE_PLLI2S) ||\
                                           ((SOURCE) == RCC_I2SAPB2CLKSOURCE_EXT)    ||\
                                           ((SOURCE) == RCC_I2SAPB2CLKSOURCE_PLLR)   ||\
                                           ((SOURCE) == RCC_I2SAPB2CLKSOURCE_PLLSRC))

#define IS_RCC_FMPI2C1CLKSOURCE(SOURCE)   (((SOURCE) == RCC_FMPI2C1CLKSOURCE_PCLK1)    ||\
                                           ((SOURCE) == RCC_FMPI2C1CLKSOURCE_SYSCLK) ||\
                                           ((SOURCE) == RCC_FMPI2C1CLKSOURCE_HSI))

#define IS_RCC_CLK48CLKSOURCE(SOURCE)      (((SOURCE) == RCC_CLK48CLKSOURCE_PLLQ) ||\
                                            ((SOURCE) == RCC_CLK48CLKSOURCE_PLLI2SQ))

#define IS_RCC_SDIOCLKSOURCE(SOURCE)      (((SOURCE) == RCC_SDIOCLKSOURCE_CLK48) ||\
                                           ((SOURCE) == RCC_SDIOCLKSOURCE_SYSCLK))

#define IS_RCC_DFSDM1CLKSOURCE(__SOURCE__) (((__SOURCE__) == RCC_DFSDM1CLKSOURCE_PCLK2) || \
                                            ((__SOURCE__) == RCC_DFSDM1CLKSOURCE_SYSCLK))

#define IS_RCC_DFSDM1AUDIOCLKSOURCE(__SOURCE__) (((__SOURCE__) == RCC_DFSDM1AUDIOCLKSOURCE_I2S1) || \
                                                 ((__SOURCE__) == RCC_DFSDM1AUDIOCLKSOURCE_I2S2))

#if defined(STM32F413xx) || defined(STM32F423xx)
#define IS_RCC_DFSDM2CLKSOURCE(__SOURCE__) (((__SOURCE__) == RCC_DFSDM2CLKSOURCE_PCLK2) || \
                                            ((__SOURCE__) == RCC_DFSDM2CLKSOURCE_SYSCLK))

#define IS_RCC_DFSDM2AUDIOCLKSOURCE(__SOURCE__) (((__SOURCE__) == RCC_DFSDM2AUDIOCLKSOURCE_I2S1) || \
                                                 ((__SOURCE__) == RCC_DFSDM2AUDIOCLKSOURCE_I2S2))

#define IS_RCC_LPTIM1CLKSOURCE(SOURCE)   (((SOURCE) == RCC_LPTIM1CLKSOURCE_PCLK1) ||\
                                          ((SOURCE) == RCC_LPTIM1CLKSOURCE_HSI)  ||\
                                          ((SOURCE) == RCC_LPTIM1CLKSOURCE_LSI)  ||\
                                          ((SOURCE) == RCC_LPTIM1CLKSOURCE_LSE))

#define IS_RCC_SAIACLKSOURCE(SOURCE)     (((SOURCE) == RCC_SAIACLKSOURCE_PLLI2SR) ||\
                                          ((SOURCE) == RCC_SAIACLKSOURCE_EXT)     ||\
                                          ((SOURCE) == RCC_SAIACLKSOURCE_PLLR)    ||\
                                          ((SOURCE) == RCC_SAIACLKSOURCE_PLLSRC))

#define IS_RCC_SAIBCLKSOURCE(SOURCE)     (((SOURCE) == RCC_SAIBCLKSOURCE_PLLI2SR) ||\
                                          ((SOURCE) == RCC_SAIBCLKSOURCE_EXT)     ||\
                                          ((SOURCE) == RCC_SAIBCLKSOURCE_PLLR)    ||\
                                          ((SOURCE) == RCC_SAIBCLKSOURCE_PLLSRC))

#define IS_RCC_PLL_DIVR_VALUE(VALUE) ((1U <= (VALUE)) && ((VALUE) <= 32U))

#define IS_RCC_PLLI2S_DIVR_VALUE(VALUE) ((1U <= (VALUE)) && ((VALUE) <= 32U))

#endif /* STM32F413xx || STM32F423xx */
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || \
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx) || \
    defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || \
    defined(STM32F412Rx) || defined(STM32F413xx) || defined(STM32F423xx)
      
#define IS_RCC_MCO2SOURCE(SOURCE) (((SOURCE) == RCC_MCO2SOURCE_SYSCLK) || ((SOURCE) == RCC_MCO2SOURCE_PLLI2SCLK)|| \
                                   ((SOURCE) == RCC_MCO2SOURCE_HSE)    || ((SOURCE) == RCC_MCO2SOURCE_PLLCLK))

#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx ||
          STM32F401xC || STM32F401xE || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx || STM32F412Vx || \
          STM32F412Rx */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)      
#define IS_RCC_MCO2SOURCE(SOURCE) (((SOURCE) == RCC_MCO2SOURCE_SYSCLK) || ((SOURCE) == RCC_MCO2SOURCE_I2SCLK)|| \
                                   ((SOURCE) == RCC_MCO2SOURCE_HSE)    || ((SOURCE) == RCC_MCO2SOURCE_PLLCLK))
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */
/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */ 

/**
  * @}
  */  
#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_RCC_EX_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
