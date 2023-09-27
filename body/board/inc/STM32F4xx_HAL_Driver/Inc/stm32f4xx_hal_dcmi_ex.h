/**
  ******************************************************************************
  * @file    stm32f4xx_hal_dcmi_ex.h
  * @author  MCD Application Team
  * @brief   Header file of DCMI Extension HAL module.
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
#ifndef __STM32F4xx_HAL_DCMI_EX_H
#define __STM32F4xx_HAL_DCMI_EX_H

#ifdef __cplusplus
 extern "C" {
#endif

#if defined(STM32F407xx) || defined(STM32F417xx) || defined(STM32F427xx) || defined(STM32F437xx) ||\
    defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx)

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"


/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup DCMIEx
  * @brief DCMI HAL module driver
  * @{
  */  

/* Exported types ------------------------------------------------------------*/
/** @defgroup DCMIEx_Exported_Types DCMI Extended Exported Types
  * @{
  */
/** 
  * @brief   DCMIEx Embedded Synchronisation CODE Init structure definition
  */ 
typedef struct
{
  uint8_t FrameStartCode; /*!< Specifies the code of the frame start delimiter. */
  uint8_t LineStartCode;  /*!< Specifies the code of the line start delimiter.  */
  uint8_t LineEndCode;    /*!< Specifies the code of the line end delimiter.    */
  uint8_t FrameEndCode;   /*!< Specifies the code of the frame end delimiter.   */
}DCMI_CodesInitTypeDef;

/** 
  * @brief   DCMI Init structure definition
  */  
typedef struct
{
  uint32_t  SynchroMode;                /*!< Specifies the Synchronization Mode: Hardware or Embedded.
                                             This parameter can be a value of @ref DCMI_Synchronization_Mode   */

  uint32_t  PCKPolarity;                /*!< Specifies the Pixel clock polarity: Falling or Rising.
                                             This parameter can be a value of @ref DCMI_PIXCK_Polarity         */

  uint32_t  VSPolarity;                 /*!< Specifies the Vertical synchronization polarity: High or Low.
                                             This parameter can be a value of @ref DCMI_VSYNC_Polarity         */

  uint32_t  HSPolarity;                 /*!< Specifies the Horizontal synchronization polarity: High or Low.
                                             This parameter can be a value of @ref DCMI_HSYNC_Polarity         */

  uint32_t  CaptureRate;                /*!< Specifies the frequency of frame capture: All, 1/2 or 1/4.
                                             This parameter can be a value of @ref DCMI_Capture_Rate           */

  uint32_t  ExtendedDataMode;           /*!< Specifies the data width: 8-bit, 10-bit, 12-bit or 14-bit.
                                             This parameter can be a value of @ref DCMI_Extended_Data_Mode     */

  DCMI_CodesInitTypeDef SyncroCode;     /*!< Specifies the code of the frame start delimiter.                  */

  uint32_t JPEGMode;                    /*!< Enable or Disable the JPEG mode
                                             This parameter can be a value of @ref DCMI_MODE_JPEG              */
#if defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
  uint32_t ByteSelectMode;              /*!< Specifies the data to be captured by the interface 
                                            This parameter can be a value of @ref DCMIEx_Byte_Select_Mode      */

  uint32_t ByteSelectStart;             /*!< Specifies if the data to be captured by the interface is even or odd
                                            This parameter can be a value of @ref DCMIEx_Byte_Select_Start     */

  uint32_t LineSelectMode;              /*!< Specifies the line of data to be captured by the interface 
                                            This parameter can be a value of @ref DCMIEx_Line_Select_Mode      */

  uint32_t LineSelectStart;             /*!< Specifies if the line of data to be captured by the interface is even or odd
                                            This parameter can be a value of @ref DCMIEx_Line_Select_Start     */

#endif /* STM32F446xx || STM32F469xx || STM32F479xx */
}DCMI_InitTypeDef;

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
#if defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
/** @defgroup DCMIEx_Exported_Constants DCMI Exported Constants
  * @{
  */

/** @defgroup DCMIEx_Byte_Select_Mode DCMI Byte Select Mode
  * @{
  */
#define DCMI_BSM_ALL                 0x00000000U                                 /*!< Interface captures all received data                       */
#define DCMI_BSM_OTHER               ((uint32_t)DCMI_CR_BSM_0)                   /*!< Interface captures every other byte from the received data */
#define DCMI_BSM_ALTERNATE_4         ((uint32_t)DCMI_CR_BSM_1)                   /*!< Interface captures one byte out of four                    */
#define DCMI_BSM_ALTERNATE_2         ((uint32_t)(DCMI_CR_BSM_0 | DCMI_CR_BSM_1)) /*!< Interface captures two bytes out of four                   */

/**
  * @}
  */

/** @defgroup DCMIEx_Byte_Select_Start DCMI Byte Select Start
  * @{
  */ 
#define DCMI_OEBS_ODD               0x00000000U              /*!< Interface captures first data from the frame/line start, second one being dropped  */
#define DCMI_OEBS_EVEN              ((uint32_t)DCMI_CR_OEBS) /*!< Interface captures second data from the frame/line start, first one being dropped */

/**
  * @}
  */

/** @defgroup DCMIEx_Line_Select_Mode DCMI Line Select Mode
  * @{
  */
#define DCMI_LSM_ALL                 0x00000000U             /*!< Interface captures all received lines  */
#define DCMI_LSM_ALTERNATE_2         ((uint32_t)DCMI_CR_LSM) /*!< Interface captures one line out of two */

/**
  * @}
  */

/** @defgroup DCMIEx_Line_Select_Start DCMI Line Select Start
  * @{
  */ 
#define DCMI_OELS_ODD               0x00000000U              /*!< Interface captures first line from the frame start, second one being dropped  */
#define DCMI_OELS_EVEN              ((uint32_t)DCMI_CR_OELS) /*!< Interface captures second line from the frame start, first one being dropped */

/**
  * @}
  */
  
/**
  * @}
  */
#endif /* STM32F446xx || STM32F469xx || STM32F479xx */

/* Exported macro ------------------------------------------------------------*/
/* Exported functions --------------------------------------------------------*/
/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
#define DCMI_POSITION_ESCR_LSC     (uint32_t)DCMI_ESCR_LSC_Pos     /*!< Required left shift to set line start delimiter */
#define DCMI_POSITION_ESCR_LEC     (uint32_t)DCMI_ESCR_LEC_Pos     /*!< Required left shift to set line end delimiter   */
#define DCMI_POSITION_ESCR_FEC     (uint32_t)DCMI_ESCR_FEC_Pos     /*!< Required left shift to set frame end delimiter  */

/* Private macro -------------------------------------------------------------*/
#if defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
/** @defgroup DCMIEx_Private_Macros DCMI Extended Private Macros
  * @{
  */
#define IS_DCMI_BYTE_SELECT_MODE(MODE)(((MODE) == DCMI_BSM_ALL) || \
                                       ((MODE) == DCMI_BSM_OTHER) || \
                                       ((MODE) == DCMI_BSM_ALTERNATE_4) || \
                                       ((MODE) == DCMI_BSM_ALTERNATE_2))

#define IS_DCMI_BYTE_SELECT_START(POLARITY)(((POLARITY) == DCMI_OEBS_ODD) || \
                                            ((POLARITY) == DCMI_OEBS_EVEN))

#define IS_DCMI_LINE_SELECT_MODE(MODE)(((MODE) == DCMI_LSM_ALL) || \
                                       ((MODE) == DCMI_LSM_ALTERNATE_2))

#define IS_DCMI_LINE_SELECT_START(POLARITY)(((POLARITY) == DCMI_OELS_ODD) || \
                                            ((POLARITY) == DCMI_OELS_EVEN))
#endif /* STM32F446xx || STM32F469xx || STM32F479xx */
/**
  * @}
  */

/* Private functions ---------------------------------------------------------*/
#endif /* STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx ||\
          STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx ||\
          STM32F479xx */


/**
  * @}
  */

/**
  * @}
  */ 

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_DCMI_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
