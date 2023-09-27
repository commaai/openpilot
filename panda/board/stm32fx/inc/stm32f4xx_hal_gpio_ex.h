/**
  ******************************************************************************
  * @file    stm32f4xx_hal_gpio_ex.h
  * @author  MCD Application Team
  * @version V1.6.0
  * @date    04-November-2016
  * @brief   Header file of GPIO HAL Extension module.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT(c) 2016 STMicroelectronics</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *   1. Redistributions of source code must retain the above copyright notice,
  *      this list of conditions and the following disclaimer.
  *   2. Redistributions in binary form must reproduce the above copyright notice,
  *      this list of conditions and the following disclaimer in the documentation
  *      and/or other materials provided with the distribution.
  *   3. Neither the name of STMicroelectronics nor the names of its contributors
  *      may be used to endorse or promote products derived from this software
  *      without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM32F4xx_HAL_GPIO_EX_H
#define __STM32F4xx_HAL_GPIO_EX_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup GPIOEx GPIOEx
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
/** @defgroup GPIOEx_Exported_Constants GPIO Exported Constants
  * @{
  */

/** @defgroup GPIO_Alternate_function_selection GPIO Alternate Function Selection
  * @{
  */

/*------------------------------------------ STM32F429xx/STM32F439xx ---------*/
#if defined(STM32F429xx) || defined(STM32F439xx)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM8          ((uint8_t)0x03U)  /* TIM8 Alternate Function mapping  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1 Alternate Function mapping        */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_SPI3          ((uint8_t)0x05U)  /* SPI3/I2S3 Alternate Function mapping   */
#define GPIO_AF5_SPI4          ((uint8_t)0x05U)  /* SPI4 Alternate Function mapping        */
#define GPIO_AF5_SPI5          ((uint8_t)0x05U)  /* SPI5 Alternate Function mapping        */
#define GPIO_AF5_SPI6          ((uint8_t)0x05U)  /* SPI6 Alternate Function mapping        */
#define GPIO_AF5_I2S3ext       ((uint8_t)0x05U)  /* I2S3ext_SD Alternate Function mapping  */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_I2S2ext       ((uint8_t)0x06U)  /* I2S2ext_SD Alternate Function mapping */
#define GPIO_AF6_SAI1          ((uint8_t)0x06U)  /* SAI1 Alternate Function mapping       */

/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_USART3        ((uint8_t)0x07U)  /* USART3 Alternate Function mapping     */
#define GPIO_AF7_I2S3ext       ((uint8_t)0x07U)  /* I2S3ext_SD Alternate Function mapping */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_UART4         ((uint8_t)0x08U)  /* UART4 Alternate Function mapping  */
#define GPIO_AF8_UART5         ((uint8_t)0x08U)  /* UART5 Alternate Function mapping  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */
#define GPIO_AF8_UART7         ((uint8_t)0x08U)  /* UART7 Alternate Function mapping  */
#define GPIO_AF8_UART8         ((uint8_t)0x08U)  /* UART8 Alternate Function mapping  */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_CAN1          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping    */
#define GPIO_AF9_CAN2          ((uint8_t)0x09U)  /* CAN2 Alternate Function mapping    */
#define GPIO_AF9_TIM12         ((uint8_t)0x09U)  /* TIM12 Alternate Function mapping   */
#define GPIO_AF9_TIM13         ((uint8_t)0x09U)  /* TIM13 Alternate Function mapping   */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping   */
#define GPIO_AF9_LTDC          ((uint8_t)0x09U)  /* LCD-TFT Alternate Function mapping */

/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */
#define GPIO_AF10_OTG_HS        ((uint8_t)0x0AU)  /* OTG_HS Alternate Function mapping */

/**
  * @brief   AF 11 selection
  */
#define GPIO_AF11_ETH           ((uint8_t)0x0BU)  /* ETHERNET Alternate Function mapping */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_FMC           ((uint8_t)0x0CU)  /* FMC Alternate Function mapping                      */
#define GPIO_AF12_OTG_HS_FS     ((uint8_t)0x0CU)  /* OTG HS configured in FS, Alternate Function mapping */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping                     */

/**
  * @brief   AF 13 selection
  */
#define GPIO_AF13_DCMI          ((uint8_t)0x0DU)  /* DCMI Alternate Function mapping */

/**
  * @brief   AF 14 selection
  */
#define GPIO_AF14_LTDC          ((uint8_t)0x0EU)  /* LCD-TFT Alternate Function mapping */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */
#endif /* STM32F429xx || STM32F439xx */
/*----------------------------------------------------------------------------*/

/*---------------------------------- STM32F427xx/STM32F437xx------------------*/
#if defined(STM32F427xx) || defined(STM32F437xx)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM8          ((uint8_t)0x03U)  /* TIM8 Alternate Function mapping  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1 Alternate Function mapping        */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_SPI3          ((uint8_t)0x05U)  /* SPI3/I2S3 Alternate Function mapping   */
#define GPIO_AF5_SPI4          ((uint8_t)0x05U)  /* SPI4 Alternate Function mapping        */
#define GPIO_AF5_SPI5          ((uint8_t)0x05U)  /* SPI5 Alternate Function mapping        */
#define GPIO_AF5_SPI6          ((uint8_t)0x05U)  /* SPI6 Alternate Function mapping        */
/** @brief  GPIO_Legacy
  */
#define GPIO_AF5_I2S3ext       GPIO_AF5_SPI3    /* I2S3ext_SD Alternate Function mapping  */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_I2S2ext       ((uint8_t)0x06U)  /* I2S2ext_SD Alternate Function mapping */
#define GPIO_AF6_SAI1          ((uint8_t)0x06U)  /* SAI1 Alternate Function mapping       */

/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_USART3        ((uint8_t)0x07U)  /* USART3 Alternate Function mapping     */
#define GPIO_AF7_I2S3ext       ((uint8_t)0x07U)  /* I2S3ext_SD Alternate Function mapping */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_UART4         ((uint8_t)0x08U)  /* UART4 Alternate Function mapping  */
#define GPIO_AF8_UART5         ((uint8_t)0x08U)  /* UART5 Alternate Function mapping  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */
#define GPIO_AF8_UART7         ((uint8_t)0x08U)  /* UART7 Alternate Function mapping  */
#define GPIO_AF8_UART8         ((uint8_t)0x08U)  /* UART8 Alternate Function mapping  */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_CAN1          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping  */
#define GPIO_AF9_CAN2          ((uint8_t)0x09U)  /* CAN2 Alternate Function mapping  */
#define GPIO_AF9_TIM12         ((uint8_t)0x09U)  /* TIM12 Alternate Function mapping */
#define GPIO_AF9_TIM13         ((uint8_t)0x09U)  /* TIM13 Alternate Function mapping */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping */

/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */
#define GPIO_AF10_OTG_HS        ((uint8_t)0x0AU)  /* OTG_HS Alternate Function mapping */

/**
  * @brief   AF 11 selection
  */
#define GPIO_AF11_ETH           ((uint8_t)0x0BU)  /* ETHERNET Alternate Function mapping */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_FMC           ((uint8_t)0x0CU)  /* FMC Alternate Function mapping                      */
#define GPIO_AF12_OTG_HS_FS     ((uint8_t)0x0CU)  /* OTG HS configured in FS, Alternate Function mapping */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping                     */

/**
  * @brief   AF 13 selection
  */
#define GPIO_AF13_DCMI          ((uint8_t)0x0DU)  /* DCMI Alternate Function mapping */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */
#endif /* STM32F427xx || STM32F437xx */
/*----------------------------------------------------------------------------*/

/*---------------------------------- STM32F407xx/STM32F417xx------------------*/
#if defined(STM32F407xx) || defined(STM32F417xx)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM8          ((uint8_t)0x03U)  /* TIM8 Alternate Function mapping  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1 Alternate Function mapping        */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_I2S3ext       ((uint8_t)0x05U)  /* I2S3ext_SD Alternate Function mapping  */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_I2S2ext       ((uint8_t)0x06U)  /* I2S2ext_SD Alternate Function mapping */

/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_USART3        ((uint8_t)0x07U)  /* USART3 Alternate Function mapping     */
#define GPIO_AF7_I2S3ext       ((uint8_t)0x07U)  /* I2S3ext_SD Alternate Function mapping */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_UART4         ((uint8_t)0x08U)  /* UART4 Alternate Function mapping  */
#define GPIO_AF8_UART5         ((uint8_t)0x08U)  /* UART5 Alternate Function mapping  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_CAN1          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping  */
#define GPIO_AF9_CAN2          ((uint8_t)0x09U)  /* CAN2 Alternate Function mapping  */
#define GPIO_AF9_TIM12         ((uint8_t)0x09U)  /* TIM12 Alternate Function mapping */
#define GPIO_AF9_TIM13         ((uint8_t)0x09U)  /* TIM13 Alternate Function mapping */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping */

/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */
#define GPIO_AF10_OTG_HS        ((uint8_t)0x0AU)  /* OTG_HS Alternate Function mapping */

/**
  * @brief   AF 11 selection
  */
#define GPIO_AF11_ETH           ((uint8_t)0x0BU)  /* ETHERNET Alternate Function mapping */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_FSMC          ((uint8_t)0x0CU)  /* FSMC Alternate Function mapping                     */
#define GPIO_AF12_OTG_HS_FS     ((uint8_t)0x0CU)  /* OTG HS configured in FS, Alternate Function mapping */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping                     */

/**
  * @brief   AF 13 selection
  */
#define GPIO_AF13_DCMI          ((uint8_t)0x0DU)  /* DCMI Alternate Function mapping */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */
#endif /* STM32F407xx || STM32F417xx */
/*----------------------------------------------------------------------------*/

/*---------------------------------- STM32F405xx/STM32F415xx------------------*/
#if defined(STM32F405xx) || defined(STM32F415xx)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM8          ((uint8_t)0x03U)  /* TIM8 Alternate Function mapping  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1 Alternate Function mapping        */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_I2S3ext       ((uint8_t)0x05U)  /* I2S3ext_SD Alternate Function mapping  */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_I2S2ext       ((uint8_t)0x06U)  /* I2S2ext_SD Alternate Function mapping */

/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_USART3        ((uint8_t)0x07U)  /* USART3 Alternate Function mapping     */
#define GPIO_AF7_I2S3ext       ((uint8_t)0x07U)  /* I2S3ext_SD Alternate Function mapping */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_UART4         ((uint8_t)0x08U)  /* UART4 Alternate Function mapping  */
#define GPIO_AF8_UART5         ((uint8_t)0x08U)  /* UART5 Alternate Function mapping  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_CAN1          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping  */
#define GPIO_AF9_CAN2          ((uint8_t)0x09U)  /* CAN2 Alternate Function mapping  */
#define GPIO_AF9_TIM12         ((uint8_t)0x09U)  /* TIM12 Alternate Function mapping */
#define GPIO_AF9_TIM13         ((uint8_t)0x09U)  /* TIM13 Alternate Function mapping */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping */

/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */
#define GPIO_AF10_OTG_HS        ((uint8_t)0x0AU)  /* OTG_HS Alternate Function mapping */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_FSMC          ((uint8_t)0x0CU)  /* FSMC Alternate Function mapping                     */
#define GPIO_AF12_OTG_HS_FS     ((uint8_t)0x0CU)  /* OTG HS configured in FS, Alternate Function mapping */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping                     */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */
#endif /* STM32F405xx || STM32F415xx */

/*----------------------------------------------------------------------------*/

/*---------------------------------------- STM32F401xx------------------------*/
#if defined(STM32F401xC) || defined(STM32F401xE)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1 Alternate Function mapping        */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_SPI4          ((uint8_t)0x05U)  /* SPI4 Alternate Function mapping        */
#define GPIO_AF5_I2S3ext       ((uint8_t)0x05U)  /* I2S3ext_SD Alternate Function mapping   */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_I2S2ext       ((uint8_t)0x06U)  /* I2S2ext_SD Alternate Function mapping */

/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_I2S3ext       ((uint8_t)0x07U)  /* I2S3ext_SD Alternate Function mapping */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping */
#define GPIO_AF9_I2C2          ((uint8_t)0x09U)  /* I2C2 Alternate Function mapping  */
#define GPIO_AF9_I2C3          ((uint8_t)0x09U)  /* I2C3 Alternate Function mapping  */


/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping  */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */
#endif /* STM32F401xC || STM32F401xE */
/*----------------------------------------------------------------------------*/

/*--------------- STM32F412Zx/STM32F412Vx/STM32F412Rx/STM32F412Cx-------------*/
#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM8          ((uint8_t)0x03U)  /* TIM8 Alternate Function mapping  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping    */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping    */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping    */
#define GPIO_AF4_FMPI2C1       ((uint8_t)0x04U)  /* FMPI2C1 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1/I2S1 Alternate Function mapping   */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_SPI3          ((uint8_t)0x05U)  /* SPI3/I2S3 Alternate Function mapping   */
#define GPIO_AF5_SPI4          ((uint8_t)0x05U)  /* SPI4/I2S4 Alternate Function mapping   */
#define GPIO_AF5_I2S3ext       ((uint8_t)0x05U)  /* I2S3ext_SD Alternate Function mapping  */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI2          ((uint8_t)0x06U)  /* I2S2 Alternate Function mapping       */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_SPI4          ((uint8_t)0x06U)  /* SPI4/I2S4 Alternate Function mapping  */
#define GPIO_AF6_SPI5          ((uint8_t)0x06U)  /* SPI5/I2S5 Alternate Function mapping  */
#define GPIO_AF6_I2S2ext       ((uint8_t)0x06U)  /* I2S2ext_SD Alternate Function mapping */
#define GPIO_AF6_DFSDM1        ((uint8_t)0x06U)  /* DFSDM1 Alternate Function mapping     */
/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_SPI3          ((uint8_t)0x07U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_USART3        ((uint8_t)0x07U)  /* USART3 Alternate Function mapping     */
#define GPIO_AF7_I2S3ext       ((uint8_t)0x07U)  /* I2S3ext_SD Alternate Function mapping */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */
#define GPIO_AF8_USART3        ((uint8_t)0x08U)  /* USART3 Alternate Function mapping */
#define GPIO_AF8_DFSDM1        ((uint8_t)0x08U)  /* DFSDM1 Alternate Function mapping */
#define GPIO_AF8_CAN1          ((uint8_t)0x08U)  /* CAN1 Alternate Function mapping   */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_TIM13         ((uint8_t)0x09U)  /* TIM13 Alternate Function mapping   */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping   */
#define GPIO_AF9_I2C2          ((uint8_t)0x09U)  /* I2C2 Alternate Function mapping    */
#define GPIO_AF9_I2C3          ((uint8_t)0x09U)  /* I2C3 Alternate Function mapping    */
#define GPIO_AF9_FMPI2C1       ((uint8_t)0x09U)  /* FMPI2C1 Alternate Function mapping */
#define GPIO_AF9_CAN1          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping    */
#define GPIO_AF9_CAN2          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping    */
#define GPIO_AF9_QSPI          ((uint8_t)0x09U)  /* QSPI Alternate Function mapping    */

/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */
#define GPIO_AF10_DFSDM1        ((uint8_t)0x0AU)  /* DFSDM1 Alternate Function mapping */
#define GPIO_AF10_QSPI          ((uint8_t)0x0AU)  /* QSPI Alternate Function mapping   */
#define GPIO_AF10_FMC           ((uint8_t)0x0AU)  /* FMC Alternate Function mapping    */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping  */
#define GPIO_AF12_FSMC          ((uint8_t)0x0CU)  /* FMC Alternate Function mapping   */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */

/*----------------------------------------------------------------------------*/

/*--------------- STM32F413xx/STM32F423xx-------------------------------------*/
#if defined(STM32F413xx) || defined(STM32F423xx)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */
#define GPIO_AF1_LPTIM1        ((uint8_t)0x01U)  /* LPTIM1 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM8          ((uint8_t)0x03U)  /* TIM8 Alternate Function mapping  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */
#define GPIO_AF3_DFSDM2        ((uint8_t)0x03U)  /* DFSDM2 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping    */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping    */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping    */
#define GPIO_AF4_FMPI2C1       ((uint8_t)0x04U)  /* FMPI2C1 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1/I2S1 Alternate Function mapping   */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_SPI3          ((uint8_t)0x05U)  /* SPI3/I2S3 Alternate Function mapping   */
#define GPIO_AF5_SPI4          ((uint8_t)0x05U)  /* SPI4/I2S4 Alternate Function mapping   */
#define GPIO_AF5_I2S3ext       ((uint8_t)0x05U)  /* I2S3ext_SD Alternate Function mapping  */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI2          ((uint8_t)0x06U)  /* I2S2 Alternate Function mapping       */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_SPI4          ((uint8_t)0x06U)  /* SPI4/I2S4 Alternate Function mapping  */
#define GPIO_AF6_SPI5          ((uint8_t)0x06U)  /* SPI5/I2S5 Alternate Function mapping  */
#define GPIO_AF6_I2S2ext       ((uint8_t)0x06U)  /* I2S2ext_SD Alternate Function mapping */
#define GPIO_AF6_DFSDM1        ((uint8_t)0x06U)  /* DFSDM1 Alternate Function mapping     */
#define GPIO_AF6_DFSDM2        ((uint8_t)0x06U)  /* DFSDM2 Alternate Function mapping     */
/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_SPI3          ((uint8_t)0x07U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF7_SAI1          ((uint8_t)0x07U)  /* SAI1 Alternate Function mapping       */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_USART3        ((uint8_t)0x07U)  /* USART3 Alternate Function mapping     */
#define GPIO_AF7_I2S3ext       ((uint8_t)0x07U)  /* I2S3ext_SD Alternate Function mapping */
#define GPIO_AF7_DFSDM2        ((uint8_t)0x07U)  /* DFSDM2 Alternate Function mapping     */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */
#define GPIO_AF8_USART3        ((uint8_t)0x08U)  /* USART3 Alternate Function mapping */
#define GPIO_AF8_UART4         ((uint8_t)0x08U)  /* UART4 Alternate Function mapping  */
#define GPIO_AF8_UART5         ((uint8_t)0x08U)  /* UART5 Alternate Function mapping  */
#define GPIO_AF8_UART7         ((uint8_t)0x08U)  /* UART8 Alternate Function mapping  */
#define GPIO_AF8_UART8         ((uint8_t)0x08U)  /* UART8 Alternate Function mapping  */
#define GPIO_AF8_DFSDM1        ((uint8_t)0x08U)  /* DFSDM1 Alternate Function mapping */
#define GPIO_AF8_CAN1          ((uint8_t)0x08U)  /* CAN1 Alternate Function mapping   */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_TIM12         ((uint8_t)0x09U)  /* TIM12 Alternate Function mapping   */
#define GPIO_AF9_TIM13         ((uint8_t)0x09U)  /* TIM13 Alternate Function mapping   */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping   */
#define GPIO_AF9_I2C2          ((uint8_t)0x09U)  /* I2C2 Alternate Function mapping    */
#define GPIO_AF9_I2C3          ((uint8_t)0x09U)  /* I2C3 Alternate Function mapping    */
#define GPIO_AF9_FMPI2C1       ((uint8_t)0x09U)  /* FMPI2C1 Alternate Function mapping */
#define GPIO_AF9_CAN1          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping    */
#define GPIO_AF9_CAN2          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping    */
#define GPIO_AF9_QSPI          ((uint8_t)0x09U)  /* QSPI Alternate Function mapping    */

/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_SAI1          ((uint8_t)0x0AU)  /* SAI1 Alternate Function mapping   */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */
#define GPIO_AF10_DFSDM1        ((uint8_t)0x0AU)  /* DFSDM1 Alternate Function mapping */
#define GPIO_AF10_DFSDM2        ((uint8_t)0x0AU)  /* DFSDM2 Alternate Function mapping */
#define GPIO_AF10_QSPI          ((uint8_t)0x0AU)  /* QSPI Alternate Function mapping   */
#define GPIO_AF10_FSMC          ((uint8_t)0x0AU)  /* FSMC Alternate Function mapping   */

/**
  * @brief   AF 11 selection
  */
#define GPIO_AF11_UART4         ((uint8_t)0x0BU)  /* UART4 Alternate Function mapping  */
#define GPIO_AF11_UART5         ((uint8_t)0x0BU)  /* UART5 Alternate Function mapping  */
#define GPIO_AF11_UART9         ((uint8_t)0x0BU)  /* UART9 Alternate Function mapping  */
#define GPIO_AF11_UART10        ((uint8_t)0x0BU)  /* UART10 Alternate Function mapping */
#define GPIO_AF11_CAN3          ((uint8_t)0x0BU)  /* CAN3 Alternate Function mapping   */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping  */
#define GPIO_AF12_FSMC          ((uint8_t)0x0CU)  /* FMC Alternate Function mapping   */

/**
  * @brief   AF 14 selection
  */
#define GPIO_AF14_RNG           ((uint8_t)0x0EU)  /* RNG Alternate Function mapping  */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */
#endif /* STM32F413xx || STM32F423xx */

/*---------------------------------------- STM32F411xx------------------------*/
#if defined(STM32F411xE)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1/I2S1 Alternate Function mapping   */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_SPI3          ((uint8_t)0x05U)  /* SPI3/I2S3 Alternate Function mapping   */
#define GPIO_AF5_SPI4          ((uint8_t)0x05U)  /* SPI4 Alternate Function mapping        */
#define GPIO_AF5_I2S3ext       ((uint8_t)0x05U)  /* I2S3ext_SD Alternate Function mapping  */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI2          ((uint8_t)0x06U)  /* I2S2 Alternate Function mapping       */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_SPI4          ((uint8_t)0x06U)  /* SPI4/I2S4 Alternate Function mapping  */
#define GPIO_AF6_SPI5          ((uint8_t)0x06U)  /* SPI5/I2S5 Alternate Function mapping  */
#define GPIO_AF6_I2S2ext       ((uint8_t)0x06U)  /* I2S2ext_SD Alternate Function mapping */

/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_SPI3          ((uint8_t)0x07U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_I2S3ext       ((uint8_t)0x07U)  /* I2S3ext_SD Alternate Function mapping */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping */
#define GPIO_AF9_I2C2          ((uint8_t)0x09U)  /* I2C2 Alternate Function mapping  */
#define GPIO_AF9_I2C3          ((uint8_t)0x09U)  /* I2C3 Alternate Function mapping  */

/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping  */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */
#endif /* STM32F411xE */

/*---------------------------------------- STM32F410xx------------------------*/
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_LPTIM1        ((uint8_t)0x01U)  /* LPTIM1 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping */
#define GPIO_AF4_FMPI2C1       ((uint8_t)0x04U)  /* FMPI2C1 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1/I2S1 Alternate Function mapping   */
#if defined(STM32F410Cx) || defined(STM32F410Rx)
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#endif /* STM32F410Cx || STM32F410Rx */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI1          ((uint8_t)0x06U)  /* SPI1 Alternate Function mapping  */
#if defined(STM32F410Cx) || defined(STM32F410Rx)
#define GPIO_AF6_SPI2          ((uint8_t)0x06U)  /* I2S2 Alternate Function mapping       */
#endif /* STM32F410Cx || STM32F410Rx */
#define GPIO_AF6_SPI5          ((uint8_t)0x06U)  /* SPI5/I2S5 Alternate Function mapping  */
/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_I2C2          ((uint8_t)0x09U)  /* I2C2 Alternate Function mapping  */
#define GPIO_AF9_FMPI2C1       ((uint8_t)0x09U)  /* FMPI2C1 Alternate Function mapping */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

/*---------------------------------------- STM32F446xx -----------------------*/
#if defined(STM32F446xx)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM8          ((uint8_t)0x03U)  /* TIM8 Alternate Function mapping  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */
#define GPIO_AF3_CEC           ((uint8_t)0x03U)  /* CEC Alternate Function mapping   */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping */
#define GPIO_AF4_FMPI2C1       ((uint8_t)0x04U)  /* FMPI2C1 Alternate Function mapping */
#define GPIO_AF4_CEC           ((uint8_t)0x04U)  /* CEC Alternate Function mapping  */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1/I2S1 Alternate Function mapping   */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_SPI3          ((uint8_t)0x05U)  /* SPI3/I2S3 Alternate Function mapping   */
#define GPIO_AF5_SPI4          ((uint8_t)0x05U)  /* SPI4 Alternate Function mapping        */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI2          ((uint8_t)0x06U)  /* SPI2/I2S2 Alternate Function mapping  */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_SPI4          ((uint8_t)0x06U)  /* SPI4 Alternate Function mapping       */
#define GPIO_AF6_SAI1          ((uint8_t)0x06U)  /* SAI1 Alternate Function mapping       */

/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_USART3        ((uint8_t)0x07U)  /* USART3 Alternate Function mapping     */
#define GPIO_AF7_UART5         ((uint8_t)0x07U)  /* UART5 Alternate Function mapping      */
#define GPIO_AF7_SPI2          ((uint8_t)0x07U)  /* SPI2/I2S2 Alternate Function mapping  */
#define GPIO_AF7_SPI3          ((uint8_t)0x07U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF7_SPDIFRX       ((uint8_t)0x07U)  /* SPDIFRX Alternate Function mapping      */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_UART4         ((uint8_t)0x08U)  /* UART4 Alternate Function mapping  */
#define GPIO_AF8_UART5         ((uint8_t)0x08U)  /* UART5 Alternate Function mapping  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */
#define GPIO_AF8_SPDIFRX       ((uint8_t)0x08U)  /* SPDIFRX Alternate Function mapping  */
#define GPIO_AF8_SAI2          ((uint8_t)0x08U)  /* SAI2 Alternate Function mapping   */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_CAN1          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping  */
#define GPIO_AF9_CAN2          ((uint8_t)0x09U)  /* CAN2 Alternate Function mapping  */
#define GPIO_AF9_TIM12         ((uint8_t)0x09U)  /* TIM12 Alternate Function mapping */
#define GPIO_AF9_TIM13         ((uint8_t)0x09U)  /* TIM13 Alternate Function mapping */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping */
#define GPIO_AF9_QSPI          ((uint8_t)0x09U)  /* QSPI Alternate Function mapping  */

/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */
#define GPIO_AF10_OTG_HS        ((uint8_t)0x0AU)  /* OTG_HS Alternate Function mapping */
#define GPIO_AF10_SAI2          ((uint8_t)0x0AU)  /* SAI2 Alternate Function mapping   */
#define GPIO_AF10_QSPI          ((uint8_t)0x0AU)  /* QSPI Alternate Function mapping  */

/**
  * @brief   AF 11 selection
  */
#define GPIO_AF11_ETH           ((uint8_t)0x0BU)  /* ETHERNET Alternate Function mapping */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_FMC           ((uint8_t)0x0CU)  /* FMC Alternate Function mapping                      */
#define GPIO_AF12_OTG_HS_FS     ((uint8_t)0x0CU)  /* OTG HS configured in FS, Alternate Function mapping */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping                     */

/**
  * @brief   AF 13 selection
  */
#define GPIO_AF13_DCMI          ((uint8_t)0x0DU)  /* DCMI Alternate Function mapping */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */

#endif /* STM32F446xx */
/*----------------------------------------------------------------------------*/

/*-------------------------------- STM32F469xx/STM32F479xx--------------------*/
#if defined(STM32F469xx) || defined(STM32F479xx)
/**
  * @brief   AF 0 selection
  */
#define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00U)  /* RTC_50Hz Alternate Function mapping                       */
#define GPIO_AF0_MCO           ((uint8_t)0x00U)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
#define GPIO_AF0_TAMPER        ((uint8_t)0x00U)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
#define GPIO_AF0_SWJ           ((uint8_t)0x00U)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
#define GPIO_AF0_TRACE         ((uint8_t)0x00U)  /* TRACE Alternate Function mapping                          */

/**
  * @brief   AF 1 selection
  */
#define GPIO_AF1_TIM1          ((uint8_t)0x01U)  /* TIM1 Alternate Function mapping */
#define GPIO_AF1_TIM2          ((uint8_t)0x01U)  /* TIM2 Alternate Function mapping */

/**
  * @brief   AF 2 selection
  */
#define GPIO_AF2_TIM3          ((uint8_t)0x02U)  /* TIM3 Alternate Function mapping */
#define GPIO_AF2_TIM4          ((uint8_t)0x02U)  /* TIM4 Alternate Function mapping */
#define GPIO_AF2_TIM5          ((uint8_t)0x02U)  /* TIM5 Alternate Function mapping */

/**
  * @brief   AF 3 selection
  */
#define GPIO_AF3_TIM8          ((uint8_t)0x03U)  /* TIM8 Alternate Function mapping  */
#define GPIO_AF3_TIM9          ((uint8_t)0x03U)  /* TIM9 Alternate Function mapping  */
#define GPIO_AF3_TIM10         ((uint8_t)0x03U)  /* TIM10 Alternate Function mapping */
#define GPIO_AF3_TIM11         ((uint8_t)0x03U)  /* TIM11 Alternate Function mapping */

/**
  * @brief   AF 4 selection
  */
#define GPIO_AF4_I2C1          ((uint8_t)0x04U)  /* I2C1 Alternate Function mapping */
#define GPIO_AF4_I2C2          ((uint8_t)0x04U)  /* I2C2 Alternate Function mapping */
#define GPIO_AF4_I2C3          ((uint8_t)0x04U)  /* I2C3 Alternate Function mapping */

/**
  * @brief   AF 5 selection
  */
#define GPIO_AF5_SPI1          ((uint8_t)0x05U)  /* SPI1 Alternate Function mapping        */
#define GPIO_AF5_SPI2          ((uint8_t)0x05U)  /* SPI2/I2S2 Alternate Function mapping   */
#define GPIO_AF5_SPI3          ((uint8_t)0x05U)  /* SPI3/I2S3 Alternate Function mapping   */
#define GPIO_AF5_SPI4          ((uint8_t)0x05U)  /* SPI4 Alternate Function mapping        */
#define GPIO_AF5_SPI5          ((uint8_t)0x05U)  /* SPI5 Alternate Function mapping        */
#define GPIO_AF5_SPI6          ((uint8_t)0x05U)  /* SPI6 Alternate Function mapping        */
#define GPIO_AF5_I2S3ext       ((uint8_t)0x05U)  /* I2S3ext_SD Alternate Function mapping  */

/**
  * @brief   AF 6 selection
  */
#define GPIO_AF6_SPI3          ((uint8_t)0x06U)  /* SPI3/I2S3 Alternate Function mapping  */
#define GPIO_AF6_I2S2ext       ((uint8_t)0x06U)  /* I2S2ext_SD Alternate Function mapping */
#define GPIO_AF6_SAI1          ((uint8_t)0x06U)  /* SAI1 Alternate Function mapping       */

/**
  * @brief   AF 7 selection
  */
#define GPIO_AF7_USART1        ((uint8_t)0x07U)  /* USART1 Alternate Function mapping     */
#define GPIO_AF7_USART2        ((uint8_t)0x07U)  /* USART2 Alternate Function mapping     */
#define GPIO_AF7_USART3        ((uint8_t)0x07U)  /* USART3 Alternate Function mapping     */
#define GPIO_AF7_I2S3ext       ((uint8_t)0x07U)  /* I2S3ext_SD Alternate Function mapping */

/**
  * @brief   AF 8 selection
  */
#define GPIO_AF8_UART4         ((uint8_t)0x08U)  /* UART4 Alternate Function mapping  */
#define GPIO_AF8_UART5         ((uint8_t)0x08U)  /* UART5 Alternate Function mapping  */
#define GPIO_AF8_USART6        ((uint8_t)0x08U)  /* USART6 Alternate Function mapping */
#define GPIO_AF8_UART7         ((uint8_t)0x08U)  /* UART7 Alternate Function mapping  */
#define GPIO_AF8_UART8         ((uint8_t)0x08U)  /* UART8 Alternate Function mapping  */

/**
  * @brief   AF 9 selection
  */
#define GPIO_AF9_CAN1          ((uint8_t)0x09U)  /* CAN1 Alternate Function mapping    */
#define GPIO_AF9_CAN2          ((uint8_t)0x09U)  /* CAN2 Alternate Function mapping    */
#define GPIO_AF9_TIM12         ((uint8_t)0x09U)  /* TIM12 Alternate Function mapping   */
#define GPIO_AF9_TIM13         ((uint8_t)0x09U)  /* TIM13 Alternate Function mapping   */
#define GPIO_AF9_TIM14         ((uint8_t)0x09U)  /* TIM14 Alternate Function mapping   */
#define GPIO_AF9_LTDC          ((uint8_t)0x09U)  /* LCD-TFT Alternate Function mapping */
#define GPIO_AF9_QSPI          ((uint8_t)0x09U)  /* QSPI Alternate Function mapping    */

/**
  * @brief   AF 10 selection
  */
#define GPIO_AF10_OTG_FS        ((uint8_t)0x0AU)  /* OTG_FS Alternate Function mapping */
#define GPIO_AF10_OTG_HS        ((uint8_t)0x0AU)  /* OTG_HS Alternate Function mapping */
#define GPIO_AF10_QSPI          ((uint8_t)0x0AU)  /* QSPI Alternate Function mapping   */

/**
  * @brief   AF 11 selection
  */
#define GPIO_AF11_ETH           ((uint8_t)0x0BU)  /* ETHERNET Alternate Function mapping */

/**
  * @brief   AF 12 selection
  */
#define GPIO_AF12_FMC           ((uint8_t)0x0CU)  /* FMC Alternate Function mapping                      */
#define GPIO_AF12_OTG_HS_FS     ((uint8_t)0x0CU)  /* OTG HS configured in FS, Alternate Function mapping */
#define GPIO_AF12_SDIO          ((uint8_t)0x0CU)  /* SDIO Alternate Function mapping                     */

/**
  * @brief   AF 13 selection
  */
#define GPIO_AF13_DCMI          ((uint8_t)0x0DU)  /* DCMI Alternate Function mapping */
#define GPIO_AF13_DSI           ((uint8_t)0x0DU)  /* DSI Alternate Function mapping  */

/**
  * @brief   AF 14 selection
  */
#define GPIO_AF14_LTDC          ((uint8_t)0x0EU)  /* LCD-TFT Alternate Function mapping */

/**
  * @brief   AF 15 selection
  */
#define GPIO_AF15_EVENTOUT      ((uint8_t)0x0FU)  /* EVENTOUT Alternate Function mapping */

#endif /* STM32F469xx || STM32F479xx */
/*----------------------------------------------------------------------------*/
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup GPIOEx_Exported_Macros GPIO Exported Macros
  * @{
  */
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup GPIOEx_Exported_Functions GPIO Exported Functions
  * @{
  */
/**
  * @}
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup GPIOEx_Private_Constants GPIO Private Constants
  * @{
  */
/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup GPIOEx_Private_Macros GPIO Private Macros
  * @{
  */
/** @defgroup GPIOEx_Get_Port_Index GPIO Get Port Index
  * @{
  */
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx)
#define GPIO_GET_INDEX(__GPIOx__)    (uint8_t)(((__GPIOx__) == (GPIOA))? 0U :\
                                               ((__GPIOx__) == (GPIOB))? 1U :\
                                               ((__GPIOx__) == (GPIOC))? 2U :\
                                               ((__GPIOx__) == (GPIOD))? 3U :\
                                               ((__GPIOx__) == (GPIOE))? 4U :\
                                               ((__GPIOx__) == (GPIOF))? 5U :\
                                               ((__GPIOx__) == (GPIOG))? 6U :\
                                               ((__GPIOx__) == (GPIOH))? 7U : 8U)
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
#define GPIO_GET_INDEX(__GPIOx__)    (uint8_t)(((__GPIOx__) == (GPIOA))? 0U :\
                                               ((__GPIOx__) == (GPIOB))? 1U :\
                                               ((__GPIOx__) == (GPIOC))? 2U :\
                                               ((__GPIOx__) == (GPIOD))? 3U :\
                                               ((__GPIOx__) == (GPIOE))? 4U :\
                                               ((__GPIOx__) == (GPIOF))? 5U :\
                                               ((__GPIOx__) == (GPIOG))? 6U :\
                                               ((__GPIOx__) == (GPIOH))? 7U :\
                                               ((__GPIOx__) == (GPIOI))? 8U :\
                                               ((__GPIOx__) == (GPIOJ))? 9U : 10U)
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define GPIO_GET_INDEX(__GPIOx__)    (uint8_t)(((__GPIOx__) == (GPIOA))? 0U :\
                                               ((__GPIOx__) == (GPIOB))? 1U :\
                                               ((__GPIOx__) == (GPIOC))? 2U : 7U)
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F411xE)
#define GPIO_GET_INDEX(__GPIOx__)    (uint8_t)(((__GPIOx__) == (GPIOA))? 0U :\
                                               ((__GPIOx__) == (GPIOB))? 1U :\
                                               ((__GPIOx__) == (GPIOC))? 2U :\
                                               ((__GPIOx__) == (GPIOD))? 3U :\
                                               ((__GPIOx__) == (GPIOE))? 4U : 7U)
#endif /* STM32F401xC || STM32F401xE || STM32F411xE */

#if defined(STM32F446xx) || defined(STM32F412Zx) ||defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx)  || defined(STM32F413xx) || defined(STM32F423xx)
#define GPIO_GET_INDEX(__GPIOx__)    (uint8_t)(((__GPIOx__) == (GPIOA))? 0U :\
                                               ((__GPIOx__) == (GPIOB))? 1U :\
                                               ((__GPIOx__) == (GPIOC))? 2U :\
                                               ((__GPIOx__) == (GPIOD))? 3U :\
                                               ((__GPIOx__) == (GPIOE))? 4U :\
                                               ((__GPIOx__) == (GPIOF))? 5U :\
                                               ((__GPIOx__) == (GPIOG))? 6U : 7U)
#endif /* STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

/**
  * @}
  */

/** @defgroup GPIOEx_IS_Alternat_function_selection GPIO Check Alternate Function
  * @{
  */
/*------------------------- STM32F429xx/STM32F439xx---------------------------*/
#if defined(STM32F429xx) || defined(STM32F439xx)
#define IS_GPIO_AF(AF)   (((AF) == GPIO_AF0_RTC_50Hz)   || ((AF) == GPIO_AF9_TIM14)      || \
                          ((AF) == GPIO_AF0_MCO)        || ((AF) == GPIO_AF0_TAMPER)     || \
                          ((AF) == GPIO_AF0_SWJ)        || ((AF) == GPIO_AF0_TRACE)      || \
                          ((AF) == GPIO_AF1_TIM1)       || ((AF) == GPIO_AF1_TIM2)       || \
                          ((AF) == GPIO_AF2_TIM3)       || ((AF) == GPIO_AF2_TIM4)       || \
                          ((AF) == GPIO_AF2_TIM5)       || ((AF) == GPIO_AF3_TIM8)       || \
                          ((AF) == GPIO_AF4_I2C1)       || ((AF) == GPIO_AF4_I2C2)       || \
                          ((AF) == GPIO_AF4_I2C3)       || ((AF) == GPIO_AF5_SPI1)       || \
                          ((AF) == GPIO_AF5_SPI2)       || ((AF) == GPIO_AF9_TIM13)      || \
                          ((AF) == GPIO_AF6_SPI3)       || ((AF) == GPIO_AF9_TIM12)      || \
                          ((AF) == GPIO_AF7_USART1)     || ((AF) == GPIO_AF7_USART2)     || \
                          ((AF) == GPIO_AF7_USART3)     || ((AF) == GPIO_AF8_UART4)      || \
                          ((AF) == GPIO_AF8_UART5)      || ((AF) == GPIO_AF8_USART6)     || \
                          ((AF) == GPIO_AF9_CAN1)       || ((AF) == GPIO_AF9_CAN2)       || \
                          ((AF) == GPIO_AF10_OTG_FS)    || ((AF) == GPIO_AF10_OTG_HS)    || \
                          ((AF) == GPIO_AF11_ETH)       || ((AF) == GPIO_AF12_OTG_HS_FS) || \
                          ((AF) == GPIO_AF12_SDIO)      || ((AF) == GPIO_AF13_DCMI)      || \
                          ((AF) == GPIO_AF15_EVENTOUT)  || ((AF) == GPIO_AF5_SPI4)       || \
                          ((AF) == GPIO_AF5_SPI5)       || ((AF) == GPIO_AF5_SPI6)       || \
                          ((AF) == GPIO_AF8_UART7)      || ((AF) == GPIO_AF8_UART8)      || \
                          ((AF) == GPIO_AF12_FMC)       ||  ((AF) == GPIO_AF6_SAI1)      || \
                          ((AF) == GPIO_AF14_LTDC))

#endif /* STM32F429xx || STM32F439xx */
/*----------------------------------------------------------------------------*/

/*---------------------------------- STM32F427xx/STM32F437xx------------------*/
#if defined(STM32F427xx) || defined(STM32F437xx)
#define IS_GPIO_AF(AF)   (((AF) == GPIO_AF0_RTC_50Hz)   || ((AF) == GPIO_AF9_TIM14)      || \
                          ((AF) == GPIO_AF0_MCO)        || ((AF) == GPIO_AF0_TAMPER)     || \
                          ((AF) == GPIO_AF0_SWJ)        || ((AF) == GPIO_AF0_TRACE)      || \
                          ((AF) == GPIO_AF1_TIM1)       || ((AF) == GPIO_AF1_TIM2)       || \
                          ((AF) == GPIO_AF2_TIM3)       || ((AF) == GPIO_AF2_TIM4)       || \
                          ((AF) == GPIO_AF2_TIM5)       || ((AF) == GPIO_AF3_TIM8)       || \
                          ((AF) == GPIO_AF4_I2C1)       || ((AF) == GPIO_AF4_I2C2)       || \
                          ((AF) == GPIO_AF4_I2C3)       || ((AF) == GPIO_AF5_SPI1)       || \
                          ((AF) == GPIO_AF5_SPI2)       || ((AF) == GPIO_AF9_TIM13)      || \
                          ((AF) == GPIO_AF6_SPI3)       || ((AF) == GPIO_AF9_TIM12)      || \
                          ((AF) == GPIO_AF7_USART1)     || ((AF) == GPIO_AF7_USART2)     || \
                          ((AF) == GPIO_AF7_USART3)     || ((AF) == GPIO_AF8_UART4)      || \
                          ((AF) == GPIO_AF8_UART5)      || ((AF) == GPIO_AF8_USART6)     || \
                          ((AF) == GPIO_AF9_CAN1)       || ((AF) == GPIO_AF9_CAN2)       || \
                          ((AF) == GPIO_AF10_OTG_FS)    || ((AF) == GPIO_AF10_OTG_HS)    || \
                          ((AF) == GPIO_AF11_ETH)       || ((AF) == GPIO_AF12_OTG_HS_FS) || \
                          ((AF) == GPIO_AF12_SDIO)      || ((AF) == GPIO_AF13_DCMI)      || \
                          ((AF) == GPIO_AF15_EVENTOUT)  || ((AF) == GPIO_AF5_SPI4)       || \
                          ((AF) == GPIO_AF5_SPI5)       || ((AF) == GPIO_AF5_SPI6)       || \
                          ((AF) == GPIO_AF8_UART7)      || ((AF) == GPIO_AF8_UART8)      || \
                          ((AF) == GPIO_AF12_FMC)       ||  ((AF) == GPIO_AF6_SAI1))

#endif /* STM32F427xx || STM32F437xx */
/*----------------------------------------------------------------------------*/

/*---------------------------------- STM32F407xx/STM32F417xx------------------*/
#if defined(STM32F407xx) || defined(STM32F417xx)
#define IS_GPIO_AF(AF)   (((AF) == GPIO_AF0_RTC_50Hz)   || ((AF) == GPIO_AF9_TIM14)      || \
                          ((AF) == GPIO_AF0_MCO)        || ((AF) == GPIO_AF0_TAMPER)     || \
                          ((AF) == GPIO_AF0_SWJ)        || ((AF) == GPIO_AF0_TRACE)      || \
                          ((AF) == GPIO_AF1_TIM1)       || ((AF) == GPIO_AF1_TIM2)       || \
                          ((AF) == GPIO_AF2_TIM3)       || ((AF) == GPIO_AF2_TIM4)       || \
                          ((AF) == GPIO_AF2_TIM5)       || ((AF) == GPIO_AF3_TIM8)       || \
                          ((AF) == GPIO_AF4_I2C1)       || ((AF) == GPIO_AF4_I2C2)       || \
                          ((AF) == GPIO_AF4_I2C3)       || ((AF) == GPIO_AF5_SPI1)       || \
                          ((AF) == GPIO_AF5_SPI2)       || ((AF) == GPIO_AF9_TIM13)      || \
                          ((AF) == GPIO_AF6_SPI3)       || ((AF) == GPIO_AF9_TIM12)      || \
                          ((AF) == GPIO_AF7_USART1)     || ((AF) == GPIO_AF7_USART2)     || \
                          ((AF) == GPIO_AF7_USART3)     || ((AF) == GPIO_AF8_UART4)      || \
                          ((AF) == GPIO_AF8_UART5)      || ((AF) == GPIO_AF8_USART6)     || \
                          ((AF) == GPIO_AF9_CAN1)       || ((AF) == GPIO_AF9_CAN2)       || \
                          ((AF) == GPIO_AF10_OTG_FS)    || ((AF) == GPIO_AF10_OTG_HS)    || \
                          ((AF) == GPIO_AF11_ETH)       || ((AF) == GPIO_AF12_OTG_HS_FS) || \
                          ((AF) == GPIO_AF12_SDIO)      || ((AF) == GPIO_AF13_DCMI)      || \
                          ((AF) == GPIO_AF12_FSMC)      || ((AF) == GPIO_AF15_EVENTOUT))

#endif /* STM32F407xx || STM32F417xx */
/*----------------------------------------------------------------------------*/

/*---------------------------------- STM32F405xx/STM32F415xx------------------*/
#if defined(STM32F405xx) || defined(STM32F415xx)
#define IS_GPIO_AF(AF)   (((AF) == GPIO_AF0_RTC_50Hz)   || ((AF) == GPIO_AF9_TIM14)      || \
                          ((AF) == GPIO_AF0_MCO)        || ((AF) == GPIO_AF0_TAMPER)     || \
                          ((AF) == GPIO_AF0_SWJ)        || ((AF) == GPIO_AF0_TRACE)      || \
                          ((AF) == GPIO_AF1_TIM1)       || ((AF) == GPIO_AF1_TIM2)       || \
                          ((AF) == GPIO_AF2_TIM3)       || ((AF) == GPIO_AF2_TIM4)       || \
                          ((AF) == GPIO_AF2_TIM5)       || ((AF) == GPIO_AF3_TIM8)       || \
                          ((AF) == GPIO_AF4_I2C1)       || ((AF) == GPIO_AF4_I2C2)       || \
                          ((AF) == GPIO_AF4_I2C3)       || ((AF) == GPIO_AF5_SPI1)       || \
                          ((AF) == GPIO_AF5_SPI2)       || ((AF) == GPIO_AF9_TIM13)      || \
                          ((AF) == GPIO_AF6_SPI3)       || ((AF) == GPIO_AF9_TIM12)      || \
                          ((AF) == GPIO_AF7_USART1)     || ((AF) == GPIO_AF7_USART2)     || \
                          ((AF) == GPIO_AF7_USART3)     || ((AF) == GPIO_AF8_UART4)      || \
                          ((AF) == GPIO_AF8_UART5)      || ((AF) == GPIO_AF8_USART6)     || \
                          ((AF) == GPIO_AF9_CAN1)       || ((AF) == GPIO_AF9_CAN2)       || \
                          ((AF) == GPIO_AF10_OTG_FS)    || ((AF) == GPIO_AF10_OTG_HS)    || \
                          ((AF) == GPIO_AF12_OTG_HS_FS) || ((AF) == GPIO_AF12_SDIO)      || \
                          ((AF) == GPIO_AF12_FSMC)      || ((AF) == GPIO_AF15_EVENTOUT))

#endif /* STM32F405xx || STM32F415xx */

/*----------------------------------------------------------------------------*/

/*---------------------------------------- STM32F401xx------------------------*/
#if defined(STM32F401xC) || defined(STM32F401xE)
#define IS_GPIO_AF(AF)   (((AF) == GPIO_AF0_RTC_50Hz)   || ((AF) == GPIO_AF9_TIM14)      || \
                          ((AF) == GPIO_AF0_MCO)        || ((AF) == GPIO_AF0_TAMPER)     || \
                          ((AF) == GPIO_AF0_SWJ)        || ((AF) == GPIO_AF0_TRACE)      || \
                          ((AF) == GPIO_AF1_TIM1)       || ((AF) == GPIO_AF1_TIM2)       || \
                          ((AF) == GPIO_AF2_TIM3)       || ((AF) == GPIO_AF2_TIM4)       || \
                          ((AF) == GPIO_AF2_TIM5)       || ((AF) == GPIO_AF4_I2C1)       || \
                          ((AF) == GPIO_AF4_I2C2)       || ((AF) == GPIO_AF4_I2C3)       || \
                          ((AF) == GPIO_AF5_SPI1)       || ((AF) == GPIO_AF5_SPI2)       || \
                          ((AF) == GPIO_AF6_SPI3)       || ((AF) == GPIO_AF5_SPI4)       || \
                          ((AF) == GPIO_AF7_USART1)     || ((AF) == GPIO_AF7_USART2)     || \
                          ((AF) == GPIO_AF8_USART6)     || ((AF) == GPIO_AF10_OTG_FS)    || \
                          ((AF) == GPIO_AF9_I2C2)       || ((AF) == GPIO_AF9_I2C3)       || \
                          ((AF) == GPIO_AF12_SDIO)      || ((AF) == GPIO_AF15_EVENTOUT))

#endif /* STM32F401xC || STM32F401xE */
/*----------------------------------------------------------------------------*/
/*---------------------------------------- STM32F410xx------------------------*/
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define IS_GPIO_AF(AF)   (((AF) < 10U) || ((AF) == 15U))
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

/*---------------------------------------- STM32F411xx------------------------*/
#if defined(STM32F411xE)
#define IS_GPIO_AF(AF)   (((AF) == GPIO_AF0_RTC_50Hz)   || ((AF) == GPIO_AF9_TIM14)      || \
                          ((AF) == GPIO_AF0_MCO)        || ((AF) == GPIO_AF0_TAMPER)     || \
                          ((AF) == GPIO_AF0_SWJ)        || ((AF) == GPIO_AF0_TRACE)      || \
                          ((AF) == GPIO_AF1_TIM1)       || ((AF) == GPIO_AF1_TIM2)       || \
                          ((AF) == GPIO_AF2_TIM3)       || ((AF) == GPIO_AF2_TIM4)       || \
                          ((AF) == GPIO_AF2_TIM5)       || ((AF) == GPIO_AF4_I2C1)       || \
                          ((AF) == GPIO_AF4_I2C2)       || ((AF) == GPIO_AF4_I2C3)       || \
                          ((AF) == GPIO_AF5_SPI1)       || ((AF) == GPIO_AF5_SPI2)       || \
                          ((AF) == GPIO_AF5_SPI3)       || ((AF) == GPIO_AF6_SPI4)       || \
                          ((AF) == GPIO_AF6_SPI3)       || ((AF) == GPIO_AF5_SPI4)       || \
                          ((AF) == GPIO_AF6_SPI5)       || ((AF) == GPIO_AF7_SPI3)       || \
                          ((AF) == GPIO_AF7_USART1)     || ((AF) == GPIO_AF7_USART2)     || \
                          ((AF) == GPIO_AF8_USART6)     || ((AF) == GPIO_AF10_OTG_FS)    || \
                          ((AF) == GPIO_AF9_I2C2)       || ((AF) == GPIO_AF9_I2C3)       || \
                          ((AF) == GPIO_AF12_SDIO)      || ((AF) == GPIO_AF15_EVENTOUT))

#endif /* STM32F411xE */
/*----------------------------------------------------------------------------*/

/*----------------------------------------------- STM32F446xx ----------------*/
#if defined(STM32F446xx)
#define IS_GPIO_AF(AF)   (((AF) == GPIO_AF0_RTC_50Hz)   || ((AF) == GPIO_AF9_TIM14)      || \
                          ((AF) == GPIO_AF0_MCO)        || ((AF) == GPIO_AF0_TAMPER)     || \
                          ((AF) == GPIO_AF0_SWJ)        || ((AF) == GPIO_AF0_TRACE)      || \
                          ((AF) == GPIO_AF1_TIM1)       || ((AF) == GPIO_AF1_TIM2)       || \
                          ((AF) == GPIO_AF2_TIM3)       || ((AF) == GPIO_AF2_TIM4)       || \
                          ((AF) == GPIO_AF2_TIM5)       || ((AF) == GPIO_AF3_TIM8)       || \
                          ((AF) == GPIO_AF4_I2C1)       || ((AF) == GPIO_AF4_I2C2)       || \
                          ((AF) == GPIO_AF4_I2C3)       || ((AF) == GPIO_AF5_SPI1)       || \
                          ((AF) == GPIO_AF5_SPI2)       || ((AF) == GPIO_AF9_TIM13)      || \
                          ((AF) == GPIO_AF6_SPI3)       || ((AF) == GPIO_AF9_TIM12)      || \
                          ((AF) == GPIO_AF7_USART1)     || ((AF) == GPIO_AF7_USART2)     || \
                          ((AF) == GPIO_AF7_USART3)     || ((AF) == GPIO_AF8_UART4)      || \
                          ((AF) == GPIO_AF8_UART5)      || ((AF) == GPIO_AF8_USART6)     || \
                          ((AF) == GPIO_AF9_CAN1)       || ((AF) == GPIO_AF9_CAN2)       || \
                          ((AF) == GPIO_AF10_OTG_FS)    || ((AF) == GPIO_AF10_OTG_HS)    || \
                          ((AF) == GPIO_AF11_ETH)       || ((AF) == GPIO_AF12_OTG_HS_FS) || \
                          ((AF) == GPIO_AF12_SDIO)      || ((AF) == GPIO_AF13_DCMI)      || \
                          ((AF) == GPIO_AF15_EVENTOUT)  || ((AF) == GPIO_AF5_SPI4)       || \
                          ((AF) == GPIO_AF12_FMC)       ||  ((AF) == GPIO_AF6_SAI1)      || \
                          ((AF) == GPIO_AF3_CEC)        ||  ((AF) == GPIO_AF4_CEC)       || \
                          ((AF) == GPIO_AF5_SPI3)       ||  ((AF) == GPIO_AF6_SPI2)      || \
                          ((AF) == GPIO_AF6_SPI4)       ||  ((AF) == GPIO_AF7_UART5)     || \
                          ((AF) == GPIO_AF7_SPI2)       ||  ((AF) == GPIO_AF7_SPI3)      || \
                          ((AF) == GPIO_AF7_SPDIFRX)    ||  ((AF) == GPIO_AF8_SPDIFRX)   || \
                          ((AF) == GPIO_AF8_SAI2)       ||  ((AF) == GPIO_AF9_QSPI)      || \
                          ((AF) == GPIO_AF10_SAI2)      ||  ((AF) == GPIO_AF10_QSPI))

#endif /* STM32F446xx */
/*----------------------------------------------------------------------------*/

/*------------------------------------------- STM32F469xx/STM32F479xx --------*/
#if defined(STM32F469xx) || defined(STM32F479xx)
#define IS_GPIO_AF(AF)   (((AF) == GPIO_AF0_RTC_50Hz)   || ((AF) == GPIO_AF9_TIM14)      || \
                          ((AF) == GPIO_AF0_MCO)        || ((AF) == GPIO_AF0_TAMPER)     || \
                          ((AF) == GPIO_AF0_SWJ)        || ((AF) == GPIO_AF0_TRACE)      || \
                          ((AF) == GPIO_AF1_TIM1)       || ((AF) == GPIO_AF1_TIM2)       || \
                          ((AF) == GPIO_AF2_TIM3)       || ((AF) == GPIO_AF2_TIM4)       || \
                          ((AF) == GPIO_AF2_TIM5)       || ((AF) == GPIO_AF3_TIM8)       || \
                          ((AF) == GPIO_AF4_I2C1)       || ((AF) == GPIO_AF4_I2C2)       || \
                          ((AF) == GPIO_AF4_I2C3)       || ((AF) == GPIO_AF5_SPI1)       || \
                          ((AF) == GPIO_AF5_SPI2)       || ((AF) == GPIO_AF9_TIM13)      || \
                          ((AF) == GPIO_AF6_SPI3)       || ((AF) == GPIO_AF9_TIM12)      || \
                          ((AF) == GPIO_AF7_USART1)     || ((AF) == GPIO_AF7_USART2)     || \
                          ((AF) == GPIO_AF7_USART3)     || ((AF) == GPIO_AF8_UART4)      || \
                          ((AF) == GPIO_AF8_UART5)      || ((AF) == GPIO_AF8_USART6)     || \
                          ((AF) == GPIO_AF9_CAN1)       || ((AF) == GPIO_AF9_CAN2)       || \
                          ((AF) == GPIO_AF10_OTG_FS)    || ((AF) == GPIO_AF10_OTG_HS)    || \
                          ((AF) == GPIO_AF11_ETH)       || ((AF) == GPIO_AF12_OTG_HS_FS) || \
                          ((AF) == GPIO_AF12_SDIO)      || ((AF) == GPIO_AF13_DCMI)      || \
                          ((AF) == GPIO_AF15_EVENTOUT)  || ((AF) == GPIO_AF5_SPI4)       || \
                          ((AF) == GPIO_AF5_SPI5)       || ((AF) == GPIO_AF5_SPI6)       || \
                          ((AF) == GPIO_AF8_UART7)      || ((AF) == GPIO_AF8_UART8)      || \
                          ((AF) == GPIO_AF12_FMC)       || ((AF) == GPIO_AF6_SAI1)       || \
                          ((AF) == GPIO_AF14_LTDC)      || ((AF) == GPIO_AF13_DSI)      || \
                          ((AF) == GPIO_AF9_QSPI)       || ((AF) == GPIO_AF10_QSPI))

#endif /* STM32F469xx || STM32F479xx */
/*----------------------------------------------------------------------------*/

/*------------------STM32F412Zx/STM32F412Vx/STM32F412Rx/STM32F412Cx-----------*/
#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx)
#define IS_GPIO_AF(AF)   (((AF) < 16U) && ((AF) != 11U) && ((AF) != 14U) && ((AF) != 13U))
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */
/*----------------------------------------------------------------------------*/

/*------------------STM32F413xx/STM32F423xx-----------------------------------*/
#if defined(STM32F413xx) || defined(STM32F423xx)
#define IS_GPIO_AF(AF)   (((AF) < 16U) && ((AF) != 13U))
#endif /* STM32F413xx || STM32F423xx */
/*----------------------------------------------------------------------------*/

/**
  * @}
  */

/**
  * @}
  */

/* Private functions ---------------------------------------------------------*/
/** @defgroup GPIOEx_Private_Functions GPIO Private Functions
  * @{
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

#endif /* __STM32F4xx_HAL_GPIO_EX_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
