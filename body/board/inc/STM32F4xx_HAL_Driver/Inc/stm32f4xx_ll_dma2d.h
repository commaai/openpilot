/**
  ******************************************************************************
  * @file    stm32f4xx_ll_dma2d.h
  * @author  MCD Application Team
  * @brief   Header file of DMA2D LL module.
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
#ifndef STM32F4xx_LL_DMA2D_H
#define STM32F4xx_LL_DMA2D_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx.h"

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (DMA2D)

/** @defgroup DMA2D_LL DMA2D
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup DMA2D_LL_Private_Macros DMA2D Private Macros
  * @{
  */

/**
  * @}
  */
#endif /*USE_FULL_LL_DRIVER*/

/* Exported types ------------------------------------------------------------*/
#if defined(USE_FULL_LL_DRIVER)
/** @defgroup DMA2D_LL_ES_Init_Struct DMA2D Exported Init structures
  * @{
  */

/**
  * @brief LL DMA2D Init Structure Definition
  */
typedef struct
{
  uint32_t Mode;                 /*!< Specifies the DMA2D transfer mode.
                                      - This parameter can be one value of @ref DMA2D_LL_EC_MODE.

                                      This parameter can be modified afterwards,
                                      using unitary function @ref LL_DMA2D_SetMode(). */

  uint32_t ColorMode;            /*!< Specifies the color format of the output image.
                                      - This parameter can be one value of @ref DMA2D_LL_EC_OUTPUT_COLOR_MODE.

                                      This parameter can be modified afterwards using,
                                      unitary function @ref LL_DMA2D_SetOutputColorMode(). */

  uint32_t OutputBlue;           /*!< Specifies the Blue value of the output image.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if ARGB8888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if RGB888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if RGB565 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if ARGB1555 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x0F if ARGB4444 color mode is selected.

                                      This parameter can be modified afterwards,
                                      using unitary function @ref LL_DMA2D_SetOutputColor() or configuration
                                      function @ref LL_DMA2D_ConfigOutputColor(). */

  uint32_t OutputGreen;          /*!< Specifies the Green value of the output image.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if ARGB8888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if RGB888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x3F if RGB565 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if ARGB1555 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x0F if ARGB4444 color mode is selected.

                                      This parameter can be modified afterwards
                                      using unitary function @ref LL_DMA2D_SetOutputColor() or configuration
                                      function @ref LL_DMA2D_ConfigOutputColor(). */

  uint32_t OutputRed;            /*!< Specifies the Red value of the output image.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if ARGB8888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if RGB888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if RGB565 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if ARGB1555 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x0F if ARGB4444 color mode is selected.

                                      This parameter can be modified afterwards
                                      using unitary function @ref LL_DMA2D_SetOutputColor() or configuration
                                      function @ref LL_DMA2D_ConfigOutputColor(). */

  uint32_t OutputAlpha;          /*!< Specifies the Alpha channel of the output image.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if ARGB8888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x01 if ARGB1555 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x0F if ARGB4444 color mode is selected.
                                      - This parameter is not considered if RGB888 or RGB565 color mode is selected.

                                      This parameter can be modified afterwards using,
                                      unitary function @ref LL_DMA2D_SetOutputColor() or configuration
                                      function @ref LL_DMA2D_ConfigOutputColor(). */

  uint32_t OutputMemoryAddress;  /*!< Specifies the memory address.
                                      - This parameter must be a number between:
                                        Min_Data = 0x0000 and Max_Data = 0xFFFFFFFF.

                                      This parameter can be modified afterwards,
                                      using unitary function @ref LL_DMA2D_SetOutputMemAddr(). */



  uint32_t LineOffset;           /*!< Specifies the output line offset value.
                                      - This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0x3FFF.

                                      This parameter can be modified afterwards,
                                      using unitary function @ref LL_DMA2D_SetLineOffset(). */

  uint32_t NbrOfLines;           /*!< Specifies the number of lines of the area to be transferred.
                                      - This parameter must be a number between:
                                        Min_Data = 0x0000 and Max_Data = 0xFFFF.

                                      This parameter can be modified afterwards,
                                      using unitary function @ref LL_DMA2D_SetNbrOfLines(). */

  uint32_t NbrOfPixelsPerLines;  /*!< Specifies the number of pixels per lines of the area to be transferred.
                                      - This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0x3FFF.

                                      This parameter can be modified afterwards using,
                                      unitary function @ref LL_DMA2D_SetNbrOfPixelsPerLines(). */


} LL_DMA2D_InitTypeDef;

/**
  * @brief LL DMA2D Layer Configuration Structure Definition
  */
typedef struct
{
  uint32_t MemoryAddress;        /*!< Specifies the foreground or background memory address.
                                      - This parameter must be a number between:
                                        Min_Data = 0x0000 and Max_Data = 0xFFFFFFFF.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetMemAddr() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetMemAddr() for background layer. */

  uint32_t LineOffset;           /*!< Specifies the foreground or background line offset value.
                                      - This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0x3FFF.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetLineOffset() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetLineOffset() for background layer. */

  uint32_t ColorMode;            /*!< Specifies the foreground or background color mode.
                                      - This parameter can be one value of @ref DMA2D_LL_EC_INPUT_COLOR_MODE.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetColorMode() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetColorMode() for background layer. */

  uint32_t CLUTColorMode;        /*!< Specifies the foreground or background CLUT color mode.
                                       - This parameter can be one value of @ref DMA2D_LL_EC_CLUT_COLOR_MODE.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetCLUTColorMode() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetCLUTColorMode() for background layer. */

  uint32_t CLUTSize;             /*!< Specifies the foreground or background CLUT size.
                                      - This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetCLUTSize() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetCLUTSize() for background layer. */

  uint32_t AlphaMode;            /*!< Specifies the foreground or background alpha mode.
                                       - This parameter can be one value of @ref DMA2D_LL_EC_ALPHA_MODE.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetAlphaMode() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetAlphaMode() for background layer. */

  uint32_t Alpha;                /*!< Specifies the foreground or background Alpha value.
                                      - This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetAlpha() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetAlpha() for background layer. */

  uint32_t Blue;                 /*!< Specifies the foreground or background Blue color value.
                                      - This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetBlueColor() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetBlueColor() for background layer. */

  uint32_t Green;                /*!< Specifies the foreground or background Green color value.
                                      - This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetGreenColor() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetGreenColor() for background layer. */

  uint32_t Red;                  /*!< Specifies the foreground or background Red color value.
                                      - This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetRedColor() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetRedColor() for background layer. */

  uint32_t CLUTMemoryAddress;    /*!< Specifies the foreground or background CLUT memory address.
                                      - This parameter must be a number between:
                                        Min_Data = 0x0000 and Max_Data = 0xFFFFFFFF.

                                      This parameter can be modified afterwards using unitary functions
                                      - @ref LL_DMA2D_FGND_SetCLUTMemAddr() for foreground layer,
                                      - @ref LL_DMA2D_BGND_SetCLUTMemAddr() for background layer. */



} LL_DMA2D_LayerCfgTypeDef;

/**
  * @brief LL DMA2D Output Color Structure Definition
  */
typedef struct
{
  uint32_t ColorMode;            /*!< Specifies the color format of the output image.
                                      - This parameter can be one value of @ref DMA2D_LL_EC_OUTPUT_COLOR_MODE.

                                      This parameter can be modified afterwards using
                    unitary function @ref LL_DMA2D_SetOutputColorMode(). */

  uint32_t OutputBlue;           /*!< Specifies the Blue value of the output image.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if ARGB8888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if RGB888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if RGB565 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if ARGB1555 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x0F if ARGB4444 color mode is selected.

                                      This parameter can be modified afterwards using,
                                      unitary function @ref LL_DMA2D_SetOutputColor() or configuration
                                      function @ref LL_DMA2D_ConfigOutputColor(). */

  uint32_t OutputGreen;          /*!< Specifies the Green value of the output image.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if ARGB8888 color mode is selected.
                                      - This parameter must be a number between
                                        Min_Data = 0x00 and Max_Data = 0xFF if RGB888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x3F if RGB565 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if ARGB1555 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x0F if ARGB4444 color mode is selected.

                                      This parameter can be modified afterwards,
                                      using unitary function @ref LL_DMA2D_SetOutputColor() or configuration
                                      function @ref LL_DMA2D_ConfigOutputColor(). */

  uint32_t OutputRed;            /*!< Specifies the Red value of the output image.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if ARGB8888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if RGB888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if RGB565 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x1F if ARGB1555 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x0F if ARGB4444 color mode is selected.

                                      This parameter can be modified afterwards,
                                      using unitary function @ref LL_DMA2D_SetOutputColor() or configuration
                                      function @ref LL_DMA2D_ConfigOutputColor(). */

  uint32_t OutputAlpha;          /*!< Specifies the Alpha channel of the output image.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0xFF if ARGB8888 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x01 if ARGB1555 color mode is selected.
                                      - This parameter must be a number between:
                                        Min_Data = 0x00 and Max_Data = 0x0F if ARGB4444 color mode is selected.
                                      - This parameter is not considered if RGB888 or RGB565 color mode is selected.

                                      This parameter can be modified afterwards,
                                      using unitary function @ref LL_DMA2D_SetOutputColor() or configuration
                                      function @ref LL_DMA2D_ConfigOutputColor(). */

} LL_DMA2D_ColorTypeDef;

/**
  * @}
  */
#endif /* USE_FULL_LL_DRIVER */

/* Exported constants --------------------------------------------------------*/
/** @defgroup DMA2D_LL_Exported_Constants DMA2D Exported Constants
  * @{
  */

/** @defgroup DMA2D_LL_EC_GET_FLAG Get Flags Defines
  * @brief    Flags defines which can be used with LL_DMA2D_ReadReg function
  * @{
  */
#define LL_DMA2D_FLAG_CEIF          DMA2D_ISR_CEIF     /*!< Configuration Error Interrupt Flag */
#define LL_DMA2D_FLAG_CTCIF         DMA2D_ISR_CTCIF    /*!< CLUT Transfer Complete Interrupt Flag */
#define LL_DMA2D_FLAG_CAEIF         DMA2D_ISR_CAEIF    /*!< CLUT Access Error Interrupt Flag */
#define LL_DMA2D_FLAG_TWIF          DMA2D_ISR_TWIF     /*!< Transfer Watermark Interrupt Flag */
#define LL_DMA2D_FLAG_TCIF          DMA2D_ISR_TCIF     /*!< Transfer Complete Interrupt Flag */
#define LL_DMA2D_FLAG_TEIF          DMA2D_ISR_TEIF     /*!< Transfer Error Interrupt Flag */
/**
  * @}
  */

/** @defgroup DMA2D_LL_EC_IT IT Defines
  * @brief    IT defines which can be used with LL_DMA2D_ReadReg and  LL_DMA2D_WriteReg functions
  * @{
  */
#define LL_DMA2D_IT_CEIE             DMA2D_CR_CEIE    /*!< Configuration Error Interrupt */
#define LL_DMA2D_IT_CTCIE            DMA2D_CR_CTCIE   /*!< CLUT Transfer Complete Interrupt */
#define LL_DMA2D_IT_CAEIE            DMA2D_CR_CAEIE   /*!< CLUT Access Error Interrupt */
#define LL_DMA2D_IT_TWIE             DMA2D_CR_TWIE    /*!< Transfer Watermark Interrupt */
#define LL_DMA2D_IT_TCIE             DMA2D_CR_TCIE    /*!< Transfer Complete Interrupt */
#define LL_DMA2D_IT_TEIE             DMA2D_CR_TEIE    /*!< Transfer Error Interrupt */
/**
  * @}
  */

/** @defgroup DMA2D_LL_EC_MODE Mode
  * @{
  */
#define LL_DMA2D_MODE_M2M                       0x00000000U                       /*!< DMA2D memory to memory transfer mode */
#define LL_DMA2D_MODE_M2M_PFC                   DMA2D_CR_MODE_0                   /*!< DMA2D memory to memory with pixel format conversion transfer mode */
#define LL_DMA2D_MODE_M2M_BLEND                 DMA2D_CR_MODE_1                   /*!< DMA2D memory to memory with blending transfer mode */
#define LL_DMA2D_MODE_R2M                       DMA2D_CR_MODE                     /*!< DMA2D register to memory transfer mode */
/**
  * @}
  */

/** @defgroup DMA2D_LL_EC_OUTPUT_COLOR_MODE Output Color Mode
  * @{
  */
#define LL_DMA2D_OUTPUT_MODE_ARGB8888     0x00000000U                           /*!< ARGB8888 */
#define LL_DMA2D_OUTPUT_MODE_RGB888       DMA2D_OPFCCR_CM_0                     /*!< RGB888   */
#define LL_DMA2D_OUTPUT_MODE_RGB565       DMA2D_OPFCCR_CM_1                     /*!< RGB565   */
#define LL_DMA2D_OUTPUT_MODE_ARGB1555     (DMA2D_OPFCCR_CM_0|DMA2D_OPFCCR_CM_1) /*!< ARGB1555 */
#define LL_DMA2D_OUTPUT_MODE_ARGB4444     DMA2D_OPFCCR_CM_2                     /*!< ARGB4444 */
/**
  * @}
  */

/** @defgroup DMA2D_LL_EC_INPUT_COLOR_MODE Input Color Mode
  * @{
  */
#define LL_DMA2D_INPUT_MODE_ARGB8888      0x00000000U                                                /*!< ARGB8888 */
#define LL_DMA2D_INPUT_MODE_RGB888        DMA2D_FGPFCCR_CM_0                                         /*!< RGB888   */
#define LL_DMA2D_INPUT_MODE_RGB565        DMA2D_FGPFCCR_CM_1                                         /*!< RGB565   */
#define LL_DMA2D_INPUT_MODE_ARGB1555      (DMA2D_FGPFCCR_CM_0|DMA2D_FGPFCCR_CM_1)                    /*!< ARGB1555 */
#define LL_DMA2D_INPUT_MODE_ARGB4444      DMA2D_FGPFCCR_CM_2                                         /*!< ARGB4444 */
#define LL_DMA2D_INPUT_MODE_L8            (DMA2D_FGPFCCR_CM_0|DMA2D_FGPFCCR_CM_2)                    /*!< L8       */
#define LL_DMA2D_INPUT_MODE_AL44          (DMA2D_FGPFCCR_CM_1|DMA2D_FGPFCCR_CM_2)                    /*!< AL44     */
#define LL_DMA2D_INPUT_MODE_AL88          (DMA2D_FGPFCCR_CM_0|DMA2D_FGPFCCR_CM_1|DMA2D_FGPFCCR_CM_2) /*!< AL88     */
#define LL_DMA2D_INPUT_MODE_L4            DMA2D_FGPFCCR_CM_3                                         /*!< L4       */
#define LL_DMA2D_INPUT_MODE_A8            (DMA2D_FGPFCCR_CM_0|DMA2D_FGPFCCR_CM_3)                    /*!< A8       */
#define LL_DMA2D_INPUT_MODE_A4            (DMA2D_FGPFCCR_CM_1|DMA2D_FGPFCCR_CM_3)                    /*!< A4       */
/**
  * @}
  */

/** @defgroup DMA2D_LL_EC_ALPHA_MODE Alpha Mode
  * @{
  */
#define LL_DMA2D_ALPHA_MODE_NO_MODIF       0x00000000U             /*!< No modification of the alpha channel value */
#define LL_DMA2D_ALPHA_MODE_REPLACE        DMA2D_FGPFCCR_AM_0      /*!< Replace original alpha channel value by
                                                                        programmed alpha value                     */
#define LL_DMA2D_ALPHA_MODE_COMBINE        DMA2D_FGPFCCR_AM_1      /*!< Replace original alpha channel value by
                                                                        programmed alpha value with,
                                                                        original alpha channel value               */
/**
  * @}
  */




/** @defgroup DMA2D_LL_EC_CLUT_COLOR_MODE CLUT Color Mode
  * @{
  */
#define LL_DMA2D_CLUT_COLOR_MODE_ARGB8888          0x00000000U                     /*!< ARGB8888 */
#define LL_DMA2D_CLUT_COLOR_MODE_RGB888            DMA2D_FGPFCCR_CCM               /*!< RGB888   */
/**
  * @}
  */


/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup DMA2D_LL_Exported_Macros DMA2D Exported Macros
  * @{
  */

/** @defgroup DMA2D_LL_EM_WRITE_READ Common Write and read registers Macros
  * @{
  */

/**
  * @brief  Write a value in DMA2D register.
  * @param  __INSTANCE__ DMA2D Instance
  * @param  __REG__ Register to be written
  * @param  __VALUE__ Value to be written in the register
  * @retval None
  */
#define LL_DMA2D_WriteReg(__INSTANCE__, __REG__, __VALUE__) WRITE_REG((__INSTANCE__)->__REG__, (__VALUE__))

/**
  * @brief  Read a value in DMA2D register.
  * @param  __INSTANCE__ DMA2D Instance
  * @param  __REG__ Register to be read
  * @retval Register value
  */
#define LL_DMA2D_ReadReg(__INSTANCE__, __REG__) READ_REG((__INSTANCE__)->__REG__)
/**
  * @}
  */

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup DMA2D_LL_Exported_Functions DMA2D Exported Functions
  * @{
  */

/** @defgroup DMA2D_LL_EF_Configuration Configuration Functions
  * @{
  */

/**
  * @brief  Start a DMA2D transfer.
  * @rmtoll CR          START            LL_DMA2D_Start
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_Start(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->CR, DMA2D_CR_START);
}

/**
  * @brief  Indicate if a DMA2D transfer is ongoing.
  * @rmtoll CR          START            LL_DMA2D_IsTransferOngoing
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsTransferOngoing(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->CR, DMA2D_CR_START) == (DMA2D_CR_START)) ? 1UL : 0UL);
}

/**
  * @brief  Suspend DMA2D transfer.
  * @note   This API can be used to suspend automatic foreground or background CLUT loading.
  * @rmtoll CR          SUSP            LL_DMA2D_Suspend
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_Suspend(DMA2D_TypeDef *DMA2Dx)
{
  MODIFY_REG(DMA2Dx->CR, DMA2D_CR_SUSP | DMA2D_CR_START, DMA2D_CR_SUSP);
}

/**
  * @brief  Resume DMA2D transfer.
  * @note   This API can be used to resume automatic foreground or background CLUT loading.
  * @rmtoll CR          SUSP            LL_DMA2D_Resume
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_Resume(DMA2D_TypeDef *DMA2Dx)
{
  CLEAR_BIT(DMA2Dx->CR, DMA2D_CR_SUSP | DMA2D_CR_START);
}

/**
  * @brief  Indicate if DMA2D transfer is suspended.
  * @note   This API can be used to indicate whether or not automatic foreground or
  *         background CLUT loading is suspended.
  * @rmtoll CR          SUSP            LL_DMA2D_IsSuspended
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsSuspended(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->CR, DMA2D_CR_SUSP) == (DMA2D_CR_SUSP)) ? 1UL : 0UL);
}

/**
  * @brief  Abort DMA2D transfer.
  * @note   This API can be used to abort automatic foreground or background CLUT loading.
  * @rmtoll CR          ABORT            LL_DMA2D_Abort
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_Abort(DMA2D_TypeDef *DMA2Dx)
{
  MODIFY_REG(DMA2Dx->CR, DMA2D_CR_ABORT | DMA2D_CR_START, DMA2D_CR_ABORT);
}

/**
  * @brief  Indicate if DMA2D transfer is aborted.
  * @note   This API can be used to indicate whether or not automatic foreground or
  *         background CLUT loading is aborted.
  * @rmtoll CR          ABORT            LL_DMA2D_IsAborted
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsAborted(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->CR, DMA2D_CR_ABORT) == (DMA2D_CR_ABORT)) ? 1UL : 0UL);
}

/**
  * @brief  Set DMA2D mode.
  * @rmtoll CR          MODE          LL_DMA2D_SetMode
  * @param  DMA2Dx DMA2D Instance
  * @param  Mode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_MODE_M2M
  *         @arg @ref LL_DMA2D_MODE_M2M_PFC
  *         @arg @ref LL_DMA2D_MODE_M2M_BLEND
  *         @arg @ref LL_DMA2D_MODE_R2M
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_SetMode(DMA2D_TypeDef *DMA2Dx, uint32_t Mode)
{
  MODIFY_REG(DMA2Dx->CR, DMA2D_CR_MODE, Mode);
}

/**
  * @brief  Return DMA2D mode
  * @rmtoll CR          MODE         LL_DMA2D_GetMode
  * @param  DMA2Dx DMA2D Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_DMA2D_MODE_M2M
  *         @arg @ref LL_DMA2D_MODE_M2M_PFC
  *         @arg @ref LL_DMA2D_MODE_M2M_BLEND
  *         @arg @ref LL_DMA2D_MODE_R2M
  */
__STATIC_INLINE uint32_t LL_DMA2D_GetMode(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->CR, DMA2D_CR_MODE));
}

/**
  * @brief  Set DMA2D output color mode.
  * @rmtoll OPFCCR          CM          LL_DMA2D_SetOutputColorMode
  * @param  DMA2Dx DMA2D Instance
  * @param  ColorMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB4444
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_SetOutputColorMode(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode)
{
  MODIFY_REG(DMA2Dx->OPFCCR, DMA2D_OPFCCR_CM, ColorMode);
}

/**
  * @brief  Return DMA2D output color mode.
  * @rmtoll OPFCCR          CM         LL_DMA2D_GetOutputColorMode
  * @param  DMA2Dx DMA2D Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB4444
  */
__STATIC_INLINE uint32_t LL_DMA2D_GetOutputColorMode(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->OPFCCR, DMA2D_OPFCCR_CM));
}




/**
  * @brief  Set DMA2D line offset, expressed on 14 bits ([13:0] bits).
  * @rmtoll OOR          LO          LL_DMA2D_SetLineOffset
  * @param  DMA2Dx DMA2D Instance
  * @param  LineOffset Value between Min_Data=0 and Max_Data=0x3FFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_SetLineOffset(DMA2D_TypeDef *DMA2Dx, uint32_t LineOffset)
{
  MODIFY_REG(DMA2Dx->OOR, DMA2D_OOR_LO, LineOffset);
}

/**
  * @brief  Return DMA2D line offset, expressed on 14 bits ([13:0] bits).
  * @rmtoll OOR          LO         LL_DMA2D_GetLineOffset
  * @param  DMA2Dx DMA2D Instance
  * @retval Line offset value between Min_Data=0 and Max_Data=0x3FFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_GetLineOffset(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->OOR, DMA2D_OOR_LO));
}

/**
  * @brief  Set DMA2D number of pixels per lines, expressed on 14 bits ([13:0] bits).
  * @rmtoll NLR          PL          LL_DMA2D_SetNbrOfPixelsPerLines
  * @param  DMA2Dx DMA2D Instance
  * @param  NbrOfPixelsPerLines Value between Min_Data=0 and Max_Data=0x3FFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_SetNbrOfPixelsPerLines(DMA2D_TypeDef *DMA2Dx, uint32_t NbrOfPixelsPerLines)
{
  MODIFY_REG(DMA2Dx->NLR, DMA2D_NLR_PL, (NbrOfPixelsPerLines << DMA2D_NLR_PL_Pos));
}

/**
  * @brief  Return DMA2D number of pixels per lines, expressed on 14 bits ([13:0] bits)
  * @rmtoll NLR          PL          LL_DMA2D_GetNbrOfPixelsPerLines
  * @param  DMA2Dx DMA2D Instance
  * @retval Number of pixels per lines value between Min_Data=0 and Max_Data=0x3FFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_GetNbrOfPixelsPerLines(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->NLR, DMA2D_NLR_PL) >> DMA2D_NLR_PL_Pos);
}

/**
  * @brief  Set DMA2D number of lines, expressed on 16 bits ([15:0] bits).
  * @rmtoll NLR          NL          LL_DMA2D_SetNbrOfLines
  * @param  DMA2Dx DMA2D Instance
  * @param  NbrOfLines Value between Min_Data=0 and Max_Data=0xFFFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_SetNbrOfLines(DMA2D_TypeDef *DMA2Dx, uint32_t NbrOfLines)
{
  MODIFY_REG(DMA2Dx->NLR, DMA2D_NLR_NL, NbrOfLines);
}

/**
  * @brief  Return DMA2D number of lines, expressed on 16 bits ([15:0] bits).
  * @rmtoll NLR          NL          LL_DMA2D_GetNbrOfLines
  * @param  DMA2Dx DMA2D Instance
  * @retval Number of lines value between Min_Data=0 and Max_Data=0xFFFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_GetNbrOfLines(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->NLR, DMA2D_NLR_NL));
}

/**
  * @brief  Set DMA2D output memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll OMAR          MA          LL_DMA2D_SetOutputMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @param  OutputMemoryAddress Value between Min_Data=0 and Max_Data=0xFFFFFFFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_SetOutputMemAddr(DMA2D_TypeDef *DMA2Dx, uint32_t OutputMemoryAddress)
{
  LL_DMA2D_WriteReg(DMA2Dx, OMAR, OutputMemoryAddress);
}

/**
  * @brief  Get DMA2D output memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll OMAR          MA          LL_DMA2D_GetOutputMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @retval Output memory address value between Min_Data=0 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_GetOutputMemAddr(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(LL_DMA2D_ReadReg(DMA2Dx, OMAR));
}

/**
  * @brief  Set DMA2D output color, expressed on 32 bits ([31:0] bits).
  * @note   Output color format depends on output color mode, ARGB8888, RGB888,
  *         RGB565, ARGB1555 or ARGB4444.
  * @note LL_DMA2D_ConfigOutputColor() API may be used instead if colors values formatting
  *       with respect to color mode is not done by the user code.
  * @rmtoll OCOLR        BLUE        LL_DMA2D_SetOutputColor\n
  *         OCOLR        GREEN       LL_DMA2D_SetOutputColor\n
  *         OCOLR        RED         LL_DMA2D_SetOutputColor\n
  *         OCOLR        ALPHA       LL_DMA2D_SetOutputColor
  * @param  DMA2Dx DMA2D Instance
  * @param  OutputColor Value between Min_Data=0 and Max_Data=0xFFFFFFFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_SetOutputColor(DMA2D_TypeDef *DMA2Dx, uint32_t OutputColor)
{
  MODIFY_REG(DMA2Dx->OCOLR, (DMA2D_OCOLR_BLUE_1 | DMA2D_OCOLR_GREEN_1 | DMA2D_OCOLR_RED_1 | DMA2D_OCOLR_ALPHA_1), \
             OutputColor);
}

/**
  * @brief  Get DMA2D output color, expressed on 32 bits ([31:0] bits).
  * @note   Alpha channel and red, green, blue color values must be retrieved from the returned
  *         value based on the output color mode (ARGB8888, RGB888,  RGB565, ARGB1555 or ARGB4444)
  *         as set by @ref LL_DMA2D_SetOutputColorMode.
  * @rmtoll OCOLR        BLUE        LL_DMA2D_GetOutputColor\n
  *         OCOLR        GREEN       LL_DMA2D_GetOutputColor\n
  *         OCOLR        RED         LL_DMA2D_GetOutputColor\n
  *         OCOLR        ALPHA       LL_DMA2D_GetOutputColor
  * @param  DMA2Dx DMA2D Instance
  * @retval Output color value between Min_Data=0 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_GetOutputColor(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->OCOLR, \
                             (DMA2D_OCOLR_BLUE_1 | DMA2D_OCOLR_GREEN_1 | DMA2D_OCOLR_RED_1 | DMA2D_OCOLR_ALPHA_1)));
}

/**
  * @brief  Set DMA2D line watermark, expressed on 16 bits ([15:0] bits).
  * @rmtoll LWR          LW          LL_DMA2D_SetLineWatermark
  * @param  DMA2Dx DMA2D Instance
  * @param  LineWatermark Value between Min_Data=0 and Max_Data=0xFFFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_SetLineWatermark(DMA2D_TypeDef *DMA2Dx, uint32_t LineWatermark)
{
  MODIFY_REG(DMA2Dx->LWR, DMA2D_LWR_LW, LineWatermark);
}

/**
  * @brief  Return DMA2D line watermark, expressed on 16 bits ([15:0] bits).
  * @rmtoll LWR          LW          LL_DMA2D_GetLineWatermark
  * @param  DMA2Dx DMA2D Instance
  * @retval Line watermark value between Min_Data=0 and Max_Data=0xFFFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_GetLineWatermark(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->LWR, DMA2D_LWR_LW));
}

/**
  * @brief  Set DMA2D dead time, expressed on 8 bits ([7:0] bits).
  * @rmtoll AMTCR          DT          LL_DMA2D_SetDeadTime
  * @param  DMA2Dx DMA2D Instance
  * @param  DeadTime Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_SetDeadTime(DMA2D_TypeDef *DMA2Dx, uint32_t DeadTime)
{
  MODIFY_REG(DMA2Dx->AMTCR, DMA2D_AMTCR_DT, (DeadTime << DMA2D_AMTCR_DT_Pos));
}

/**
  * @brief  Return DMA2D dead time, expressed on 8 bits ([7:0] bits).
  * @rmtoll AMTCR          DT          LL_DMA2D_GetDeadTime
  * @param  DMA2Dx DMA2D Instance
  * @retval Dead time value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_GetDeadTime(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->AMTCR, DMA2D_AMTCR_DT) >> DMA2D_AMTCR_DT_Pos);
}

/**
  * @brief  Enable DMA2D dead time functionality.
  * @rmtoll AMTCR          EN            LL_DMA2D_EnableDeadTime
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_EnableDeadTime(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->AMTCR, DMA2D_AMTCR_EN);
}

/**
  * @brief  Disable DMA2D dead time functionality.
  * @rmtoll AMTCR          EN            LL_DMA2D_DisableDeadTime
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_DisableDeadTime(DMA2D_TypeDef *DMA2Dx)
{
  CLEAR_BIT(DMA2Dx->AMTCR, DMA2D_AMTCR_EN);
}

/**
  * @brief  Indicate if DMA2D dead time functionality is enabled.
  * @rmtoll AMTCR          EN            LL_DMA2D_IsEnabledDeadTime
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsEnabledDeadTime(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->AMTCR, DMA2D_AMTCR_EN) == (DMA2D_AMTCR_EN)) ? 1UL : 0UL);
}

/** @defgroup DMA2D_LL_EF_FGND_Configuration Foreground Configuration Functions
  * @{
  */

/**
  * @brief  Set DMA2D foreground memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll FGMAR          MA          LL_DMA2D_FGND_SetMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @param  MemoryAddress Value between Min_Data=0 and Max_Data=0xFFFFFFFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetMemAddr(DMA2D_TypeDef *DMA2Dx, uint32_t MemoryAddress)
{
  LL_DMA2D_WriteReg(DMA2Dx, FGMAR, MemoryAddress);
}

/**
  * @brief  Get DMA2D foreground memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll FGMAR          MA          LL_DMA2D_FGND_GetMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @retval Foreground memory address value between Min_Data=0 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetMemAddr(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(LL_DMA2D_ReadReg(DMA2Dx, FGMAR));
}

/**
  * @brief  Enable DMA2D foreground CLUT loading.
  * @rmtoll FGPFCCR          START            LL_DMA2D_FGND_EnableCLUTLoad
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_EnableCLUTLoad(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_START);
}

/**
  * @brief  Indicate if DMA2D foreground CLUT loading is enabled.
  * @rmtoll FGPFCCR          START            LL_DMA2D_FGND_IsEnabledCLUTLoad
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_IsEnabledCLUTLoad(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_START) == (DMA2D_FGPFCCR_START)) ? 1UL : 0UL);
}

/**
  * @brief  Set DMA2D foreground color mode.
  * @rmtoll FGPFCCR          CM          LL_DMA2D_FGND_SetColorMode
  * @param  DMA2Dx DMA2D Instance
  * @param  ColorMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_INPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_INPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB4444
  *         @arg @ref LL_DMA2D_INPUT_MODE_L8
  *         @arg @ref LL_DMA2D_INPUT_MODE_AL44
  *         @arg @ref LL_DMA2D_INPUT_MODE_AL88
  *         @arg @ref LL_DMA2D_INPUT_MODE_L4
  *         @arg @ref LL_DMA2D_INPUT_MODE_A8
  *         @arg @ref LL_DMA2D_INPUT_MODE_A4
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetColorMode(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode)
{
  MODIFY_REG(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_CM, ColorMode);
}

/**
  * @brief  Return DMA2D foreground color mode.
  * @rmtoll FGPFCCR          CM         LL_DMA2D_FGND_GetColorMode
  * @param  DMA2Dx DMA2D Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_INPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_INPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB4444
  *         @arg @ref LL_DMA2D_INPUT_MODE_L8
  *         @arg @ref LL_DMA2D_INPUT_MODE_AL44
  *         @arg @ref LL_DMA2D_INPUT_MODE_AL88
  *         @arg @ref LL_DMA2D_INPUT_MODE_L4
  *         @arg @ref LL_DMA2D_INPUT_MODE_A8
  *         @arg @ref LL_DMA2D_INPUT_MODE_A4
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetColorMode(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_CM));
}

/**
  * @brief  Set DMA2D foreground alpha mode.
  * @rmtoll FGPFCCR          AM          LL_DMA2D_FGND_SetAlphaMode
  * @param  DMA2Dx DMA2D Instance
  * @param  AphaMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_ALPHA_MODE_NO_MODIF
  *         @arg @ref LL_DMA2D_ALPHA_MODE_REPLACE
  *         @arg @ref LL_DMA2D_ALPHA_MODE_COMBINE
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetAlphaMode(DMA2D_TypeDef *DMA2Dx, uint32_t AphaMode)
{
  MODIFY_REG(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_AM, AphaMode);
}

/**
  * @brief  Return DMA2D foreground alpha mode.
  * @rmtoll FGPFCCR          AM         LL_DMA2D_FGND_GetAlphaMode
  * @param  DMA2Dx DMA2D Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_DMA2D_ALPHA_MODE_NO_MODIF
  *         @arg @ref LL_DMA2D_ALPHA_MODE_REPLACE
  *         @arg @ref LL_DMA2D_ALPHA_MODE_COMBINE
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetAlphaMode(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_AM));
}

/**
  * @brief  Set DMA2D foreground alpha value, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGPFCCR          ALPHA          LL_DMA2D_FGND_SetAlpha
  * @param  DMA2Dx DMA2D Instance
  * @param  Alpha Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetAlpha(DMA2D_TypeDef *DMA2Dx, uint32_t Alpha)
{
  MODIFY_REG(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_ALPHA, (Alpha << DMA2D_FGPFCCR_ALPHA_Pos));
}

/**
  * @brief  Return DMA2D foreground alpha value, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGPFCCR          ALPHA         LL_DMA2D_FGND_GetAlpha
  * @param  DMA2Dx DMA2D Instance
  * @retval Alpha value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetAlpha(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_ALPHA) >> DMA2D_FGPFCCR_ALPHA_Pos);
}


/**
  * @brief  Set DMA2D foreground line offset, expressed on 14 bits ([13:0] bits).
  * @rmtoll FGOR          LO          LL_DMA2D_FGND_SetLineOffset
  * @param  DMA2Dx DMA2D Instance
  * @param  LineOffset Value between Min_Data=0 and Max_Data=0x3FF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetLineOffset(DMA2D_TypeDef *DMA2Dx, uint32_t LineOffset)
{
  MODIFY_REG(DMA2Dx->FGOR, DMA2D_FGOR_LO, LineOffset);
}

/**
  * @brief  Return DMA2D foreground line offset, expressed on 14 bits ([13:0] bits).
  * @rmtoll FGOR          LO         LL_DMA2D_FGND_GetLineOffset
  * @param  DMA2Dx DMA2D Instance
  * @retval Foreground line offset value between Min_Data=0 and Max_Data=0x3FF
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetLineOffset(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->FGOR, DMA2D_FGOR_LO));
}

/**
  * @brief  Set DMA2D foreground color values, expressed on 24 bits ([23:0] bits).
  * @rmtoll FGCOLR          RED          LL_DMA2D_FGND_SetColor
  * @rmtoll FGCOLR          GREEN        LL_DMA2D_FGND_SetColor
  * @rmtoll FGCOLR          BLUE         LL_DMA2D_FGND_SetColor
  * @param  DMA2Dx DMA2D Instance
  * @param  Red   Value between Min_Data=0 and Max_Data=0xFF
  * @param  Green Value between Min_Data=0 and Max_Data=0xFF
  * @param  Blue  Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetColor(DMA2D_TypeDef *DMA2Dx, uint32_t Red, uint32_t Green, uint32_t Blue)
{
  MODIFY_REG(DMA2Dx->FGCOLR, (DMA2D_FGCOLR_RED | DMA2D_FGCOLR_GREEN | DMA2D_FGCOLR_BLUE), \
             ((Red << DMA2D_FGCOLR_RED_Pos) | (Green << DMA2D_FGCOLR_GREEN_Pos) | Blue));
}

/**
  * @brief  Set DMA2D foreground red color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGCOLR          RED          LL_DMA2D_FGND_SetRedColor
  * @param  DMA2Dx DMA2D Instance
  * @param  Red Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetRedColor(DMA2D_TypeDef *DMA2Dx, uint32_t Red)
{
  MODIFY_REG(DMA2Dx->FGCOLR, DMA2D_FGCOLR_RED, (Red << DMA2D_FGCOLR_RED_Pos));
}

/**
  * @brief  Return DMA2D foreground red color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGCOLR          RED         LL_DMA2D_FGND_GetRedColor
  * @param  DMA2Dx DMA2D Instance
  * @retval Red color value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetRedColor(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->FGCOLR, DMA2D_FGCOLR_RED) >> DMA2D_FGCOLR_RED_Pos);
}

/**
  * @brief  Set DMA2D foreground green color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGCOLR          GREEN          LL_DMA2D_FGND_SetGreenColor
  * @param  DMA2Dx DMA2D Instance
  * @param  Green Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetGreenColor(DMA2D_TypeDef *DMA2Dx, uint32_t Green)
{
  MODIFY_REG(DMA2Dx->FGCOLR, DMA2D_FGCOLR_GREEN, (Green << DMA2D_FGCOLR_GREEN_Pos));
}

/**
  * @brief  Return DMA2D foreground green color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGCOLR          GREEN         LL_DMA2D_FGND_GetGreenColor
  * @param  DMA2Dx DMA2D Instance
  * @retval Green color value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetGreenColor(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->FGCOLR, DMA2D_FGCOLR_GREEN) >> DMA2D_FGCOLR_GREEN_Pos);
}

/**
  * @brief  Set DMA2D foreground blue color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGCOLR          BLUE          LL_DMA2D_FGND_SetBlueColor
  * @param  DMA2Dx DMA2D Instance
  * @param  Blue Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetBlueColor(DMA2D_TypeDef *DMA2Dx, uint32_t Blue)
{
  MODIFY_REG(DMA2Dx->FGCOLR, DMA2D_FGCOLR_BLUE, Blue);
}

/**
  * @brief  Return DMA2D foreground blue color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGCOLR          BLUE         LL_DMA2D_FGND_GetBlueColor
  * @param  DMA2Dx DMA2D Instance
  * @retval Blue color value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetBlueColor(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->FGCOLR, DMA2D_FGCOLR_BLUE));
}

/**
  * @brief  Set DMA2D foreground CLUT memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll FGCMAR          MA          LL_DMA2D_FGND_SetCLUTMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @param  CLUTMemoryAddress Value between Min_Data=0 and Max_Data=0xFFFFFFFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetCLUTMemAddr(DMA2D_TypeDef *DMA2Dx, uint32_t CLUTMemoryAddress)
{
  LL_DMA2D_WriteReg(DMA2Dx, FGCMAR, CLUTMemoryAddress);
}

/**
  * @brief  Get DMA2D foreground CLUT memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll FGCMAR          MA          LL_DMA2D_FGND_GetCLUTMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @retval Foreground CLUT memory address value between Min_Data=0 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetCLUTMemAddr(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(LL_DMA2D_ReadReg(DMA2Dx, FGCMAR));
}

/**
  * @brief  Set DMA2D foreground CLUT size, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGPFCCR          CS          LL_DMA2D_FGND_SetCLUTSize
  * @param  DMA2Dx DMA2D Instance
  * @param  CLUTSize Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetCLUTSize(DMA2D_TypeDef *DMA2Dx, uint32_t CLUTSize)
{
  MODIFY_REG(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_CS, (CLUTSize << DMA2D_FGPFCCR_CS_Pos));
}

/**
  * @brief  Get DMA2D foreground CLUT size, expressed on 8 bits ([7:0] bits).
  * @rmtoll FGPFCCR          CS          LL_DMA2D_FGND_GetCLUTSize
  * @param  DMA2Dx DMA2D Instance
  * @retval Foreground CLUT size value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetCLUTSize(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_CS) >> DMA2D_FGPFCCR_CS_Pos);
}

/**
  * @brief  Set DMA2D foreground CLUT color mode.
  * @rmtoll FGPFCCR          CCM          LL_DMA2D_FGND_SetCLUTColorMode
  * @param  DMA2Dx DMA2D Instance
  * @param  CLUTColorMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_CLUT_COLOR_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_CLUT_COLOR_MODE_RGB888
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_FGND_SetCLUTColorMode(DMA2D_TypeDef *DMA2Dx, uint32_t CLUTColorMode)
{
  MODIFY_REG(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_CCM, CLUTColorMode);
}

/**
  * @brief  Return DMA2D foreground CLUT color mode.
  * @rmtoll FGPFCCR          CCM         LL_DMA2D_FGND_GetCLUTColorMode
  * @param  DMA2Dx DMA2D Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_DMA2D_CLUT_COLOR_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_CLUT_COLOR_MODE_RGB888
  */
__STATIC_INLINE uint32_t LL_DMA2D_FGND_GetCLUTColorMode(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->FGPFCCR, DMA2D_FGPFCCR_CCM));
}

/**
  * @}
  */

/** @defgroup DMA2D_LL_EF_BGND_Configuration Background Configuration Functions
  * @{
  */

/**
  * @brief  Set DMA2D background memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll BGMAR          MA          LL_DMA2D_BGND_SetMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @param  MemoryAddress Value between Min_Data=0 and Max_Data=0xFFFFFFFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetMemAddr(DMA2D_TypeDef *DMA2Dx, uint32_t MemoryAddress)
{
  LL_DMA2D_WriteReg(DMA2Dx, BGMAR, MemoryAddress);
}

/**
  * @brief  Get DMA2D background memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll BGMAR          MA          LL_DMA2D_BGND_GetMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @retval Background memory address value between Min_Data=0 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t LL_DMA2D_BGND_GetMemAddr(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(LL_DMA2D_ReadReg(DMA2Dx, BGMAR));
}

/**
  * @brief  Enable DMA2D background CLUT loading.
  * @rmtoll BGPFCCR          START            LL_DMA2D_BGND_EnableCLUTLoad
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_EnableCLUTLoad(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_START);
}

/**
  * @brief  Indicate if DMA2D background CLUT loading is enabled.
  * @rmtoll BGPFCCR          START            LL_DMA2D_BGND_IsEnabledCLUTLoad
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_BGND_IsEnabledCLUTLoad(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_START) == (DMA2D_BGPFCCR_START)) ? 1UL : 0UL);
}

/**
  * @brief  Set DMA2D background color mode.
  * @rmtoll BGPFCCR          CM          LL_DMA2D_BGND_SetColorMode
  * @param  DMA2Dx DMA2D Instance
  * @param  ColorMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_INPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_INPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB4444
  *         @arg @ref LL_DMA2D_INPUT_MODE_L8
  *         @arg @ref LL_DMA2D_INPUT_MODE_AL44
  *         @arg @ref LL_DMA2D_INPUT_MODE_AL88
  *         @arg @ref LL_DMA2D_INPUT_MODE_L4
  *         @arg @ref LL_DMA2D_INPUT_MODE_A8
  *         @arg @ref LL_DMA2D_INPUT_MODE_A4
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetColorMode(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode)
{
  MODIFY_REG(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_CM, ColorMode);
}

/**
  * @brief  Return DMA2D background color mode.
  * @rmtoll BGPFCCR          CM          LL_DMA2D_BGND_GetColorMode
  * @param  DMA2Dx DMA2D Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_INPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_INPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_INPUT_MODE_ARGB4444
  *         @arg @ref LL_DMA2D_INPUT_MODE_L8
  *         @arg @ref LL_DMA2D_INPUT_MODE_AL44
  *         @arg @ref LL_DMA2D_INPUT_MODE_AL88
  *         @arg @ref LL_DMA2D_INPUT_MODE_L4
  *         @arg @ref LL_DMA2D_INPUT_MODE_A8
  *         @arg @ref LL_DMA2D_INPUT_MODE_A4
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetColorMode(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_CM));
}

/**
  * @brief  Set DMA2D background alpha mode.
  * @rmtoll BGPFCCR          AM         LL_DMA2D_BGND_SetAlphaMode
  * @param  DMA2Dx DMA2D Instance
  * @param  AphaMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_ALPHA_MODE_NO_MODIF
  *         @arg @ref LL_DMA2D_ALPHA_MODE_REPLACE
  *         @arg @ref LL_DMA2D_ALPHA_MODE_COMBINE
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetAlphaMode(DMA2D_TypeDef *DMA2Dx, uint32_t AphaMode)
{
  MODIFY_REG(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_AM, AphaMode);
}

/**
  * @brief  Return DMA2D background alpha mode.
  * @rmtoll BGPFCCR          AM          LL_DMA2D_BGND_GetAlphaMode
  * @param  DMA2Dx DMA2D Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_DMA2D_ALPHA_MODE_NO_MODIF
  *         @arg @ref LL_DMA2D_ALPHA_MODE_REPLACE
  *         @arg @ref LL_DMA2D_ALPHA_MODE_COMBINE
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetAlphaMode(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_AM));
}

/**
  * @brief  Set DMA2D background alpha value, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGPFCCR          ALPHA         LL_DMA2D_BGND_SetAlpha
  * @param  DMA2Dx DMA2D Instance
  * @param  Alpha Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetAlpha(DMA2D_TypeDef *DMA2Dx, uint32_t Alpha)
{
  MODIFY_REG(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_ALPHA, (Alpha << DMA2D_BGPFCCR_ALPHA_Pos));
}

/**
  * @brief  Return DMA2D background alpha value, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGPFCCR          ALPHA          LL_DMA2D_BGND_GetAlpha
  * @param  DMA2Dx DMA2D Instance
  * @retval Alpha value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetAlpha(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_ALPHA) >> DMA2D_BGPFCCR_ALPHA_Pos);
}


/**
  * @brief  Set DMA2D background line offset, expressed on 14 bits ([13:0] bits).
  * @rmtoll BGOR          LO         LL_DMA2D_BGND_SetLineOffset
  * @param  DMA2Dx DMA2D Instance
  * @param  LineOffset Value between Min_Data=0 and Max_Data=0x3FF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetLineOffset(DMA2D_TypeDef *DMA2Dx, uint32_t LineOffset)
{
  MODIFY_REG(DMA2Dx->BGOR, DMA2D_BGOR_LO, LineOffset);
}

/**
  * @brief  Return DMA2D background line offset, expressed on 14 bits ([13:0] bits).
  * @rmtoll BGOR          LO          LL_DMA2D_BGND_GetLineOffset
  * @param  DMA2Dx DMA2D Instance
  * @retval Background line offset value between Min_Data=0 and Max_Data=0x3FF
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetLineOffset(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->BGOR, DMA2D_BGOR_LO));
}

/**
  * @brief  Set DMA2D background color values, expressed on 24 bits ([23:0] bits).
  * @rmtoll BGCOLR          RED          LL_DMA2D_BGND_SetColor
  * @rmtoll BGCOLR          GREEN        LL_DMA2D_BGND_SetColor
  * @rmtoll BGCOLR          BLUE         LL_DMA2D_BGND_SetColor
  * @param  DMA2Dx DMA2D Instance
  * @param  Red   Value between Min_Data=0 and Max_Data=0xFF
  * @param  Green Value between Min_Data=0 and Max_Data=0xFF
  * @param  Blue  Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetColor(DMA2D_TypeDef *DMA2Dx, uint32_t Red, uint32_t Green, uint32_t Blue)
{
  MODIFY_REG(DMA2Dx->BGCOLR, (DMA2D_BGCOLR_RED | DMA2D_BGCOLR_GREEN | DMA2D_BGCOLR_BLUE), \
             ((Red << DMA2D_BGCOLR_RED_Pos) | (Green << DMA2D_BGCOLR_GREEN_Pos) | Blue));
}

/**
  * @brief  Set DMA2D background red color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGCOLR          RED         LL_DMA2D_BGND_SetRedColor
  * @param  DMA2Dx DMA2D Instance
  * @param  Red Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetRedColor(DMA2D_TypeDef *DMA2Dx, uint32_t Red)
{
  MODIFY_REG(DMA2Dx->BGCOLR, DMA2D_BGCOLR_RED, (Red << DMA2D_BGCOLR_RED_Pos));
}

/**
  * @brief  Return DMA2D background red color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGCOLR          RED          LL_DMA2D_BGND_GetRedColor
  * @param  DMA2Dx DMA2D Instance
  * @retval Red color value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetRedColor(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->BGCOLR, DMA2D_BGCOLR_RED) >> DMA2D_BGCOLR_RED_Pos);
}

/**
  * @brief  Set DMA2D background green color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGCOLR          GREEN         LL_DMA2D_BGND_SetGreenColor
  * @param  DMA2Dx DMA2D Instance
  * @param  Green Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetGreenColor(DMA2D_TypeDef *DMA2Dx, uint32_t Green)
{
  MODIFY_REG(DMA2Dx->BGCOLR, DMA2D_BGCOLR_GREEN, (Green << DMA2D_BGCOLR_GREEN_Pos));
}

/**
  * @brief  Return DMA2D background green color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGCOLR          GREEN          LL_DMA2D_BGND_GetGreenColor
  * @param  DMA2Dx DMA2D Instance
  * @retval Green color value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetGreenColor(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->BGCOLR, DMA2D_BGCOLR_GREEN) >> DMA2D_BGCOLR_GREEN_Pos);
}

/**
  * @brief  Set DMA2D background blue color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGCOLR          BLUE         LL_DMA2D_BGND_SetBlueColor
  * @param  DMA2Dx DMA2D Instance
  * @param  Blue Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetBlueColor(DMA2D_TypeDef *DMA2Dx, uint32_t Blue)
{
  MODIFY_REG(DMA2Dx->BGCOLR, DMA2D_BGCOLR_BLUE, Blue);
}

/**
  * @brief  Return DMA2D background blue color value, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGCOLR          BLUE          LL_DMA2D_BGND_GetBlueColor
  * @param  DMA2Dx DMA2D Instance
  * @retval Blue color value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetBlueColor(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->BGCOLR, DMA2D_BGCOLR_BLUE));
}

/**
  * @brief  Set DMA2D background CLUT memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll BGCMAR          MA         LL_DMA2D_BGND_SetCLUTMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @param  CLUTMemoryAddress Value between Min_Data=0 and Max_Data=0xFFFFFFFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetCLUTMemAddr(DMA2D_TypeDef *DMA2Dx, uint32_t CLUTMemoryAddress)
{
  LL_DMA2D_WriteReg(DMA2Dx, BGCMAR, CLUTMemoryAddress);
}

/**
  * @brief  Get DMA2D background CLUT memory address, expressed on 32 bits ([31:0] bits).
  * @rmtoll BGCMAR          MA           LL_DMA2D_BGND_GetCLUTMemAddr
  * @param  DMA2Dx DMA2D Instance
  * @retval Background CLUT memory address value between Min_Data=0 and Max_Data=0xFFFFFFFF
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetCLUTMemAddr(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(LL_DMA2D_ReadReg(DMA2Dx, BGCMAR));
}

/**
  * @brief  Set DMA2D background CLUT size, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGPFCCR          CS         LL_DMA2D_BGND_SetCLUTSize
  * @param  DMA2Dx DMA2D Instance
  * @param  CLUTSize Value between Min_Data=0 and Max_Data=0xFF
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetCLUTSize(DMA2D_TypeDef *DMA2Dx, uint32_t CLUTSize)
{
  MODIFY_REG(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_CS, (CLUTSize << DMA2D_BGPFCCR_CS_Pos));
}

/**
  * @brief  Get DMA2D background CLUT size, expressed on 8 bits ([7:0] bits).
  * @rmtoll BGPFCCR          CS           LL_DMA2D_BGND_GetCLUTSize
  * @param  DMA2Dx DMA2D Instance
  * @retval Background CLUT size value between Min_Data=0 and Max_Data=0xFF
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetCLUTSize(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_CS) >> DMA2D_BGPFCCR_CS_Pos);
}

/**
  * @brief  Set DMA2D background CLUT color mode.
  * @rmtoll BGPFCCR          CCM         LL_DMA2D_BGND_SetCLUTColorMode
  * @param  DMA2Dx DMA2D Instance
  * @param  CLUTColorMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_CLUT_COLOR_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_CLUT_COLOR_MODE_RGB888
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_BGND_SetCLUTColorMode(DMA2D_TypeDef *DMA2Dx, uint32_t CLUTColorMode)
{
  MODIFY_REG(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_CCM, CLUTColorMode);
}

/**
  * @brief  Return DMA2D background CLUT color mode.
  * @rmtoll BGPFCCR          CCM          LL_DMA2D_BGND_GetCLUTColorMode
  * @param  DMA2Dx DMA2D Instance
  * @retval Returned value can be one of the following values:
  *         @arg @ref LL_DMA2D_CLUT_COLOR_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_CLUT_COLOR_MODE_RGB888
  */
__STATIC_INLINE uint32_t  LL_DMA2D_BGND_GetCLUTColorMode(DMA2D_TypeDef *DMA2Dx)
{
  return (uint32_t)(READ_BIT(DMA2Dx->BGPFCCR, DMA2D_BGPFCCR_CCM));
}

/**
  * @}
  */

/**
  * @}
  */


/** @defgroup DMA2D_LL_EF_FLAG_MANAGEMENT Flag Management
  * @{
  */

/**
  * @brief  Check if the DMA2D Configuration Error Interrupt Flag is set or not
  * @rmtoll ISR          CEIF            LL_DMA2D_IsActiveFlag_CE
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsActiveFlag_CE(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->ISR, DMA2D_ISR_CEIF) == (DMA2D_ISR_CEIF)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D CLUT Transfer Complete Interrupt Flag is set or not
  * @rmtoll ISR          CTCIF            LL_DMA2D_IsActiveFlag_CTC
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsActiveFlag_CTC(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->ISR, DMA2D_ISR_CTCIF) == (DMA2D_ISR_CTCIF)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D CLUT Access Error Interrupt Flag is set or not
  * @rmtoll ISR          CAEIF            LL_DMA2D_IsActiveFlag_CAE
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsActiveFlag_CAE(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->ISR, DMA2D_ISR_CAEIF) == (DMA2D_ISR_CAEIF)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D Transfer Watermark Interrupt Flag is set or not
  * @rmtoll ISR          TWIF            LL_DMA2D_IsActiveFlag_TW
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsActiveFlag_TW(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->ISR, DMA2D_ISR_TWIF) == (DMA2D_ISR_TWIF)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D Transfer Complete Interrupt Flag is set or not
  * @rmtoll ISR          TCIF            LL_DMA2D_IsActiveFlag_TC
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsActiveFlag_TC(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->ISR, DMA2D_ISR_TCIF) == (DMA2D_ISR_TCIF)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D Transfer Error Interrupt Flag is set or not
  * @rmtoll ISR          TEIF            LL_DMA2D_IsActiveFlag_TE
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsActiveFlag_TE(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->ISR, DMA2D_ISR_TEIF) == (DMA2D_ISR_TEIF)) ? 1UL : 0UL);
}

/**
  * @brief  Clear DMA2D Configuration Error Interrupt Flag
  * @rmtoll IFCR          CCEIF          LL_DMA2D_ClearFlag_CE
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_ClearFlag_CE(DMA2D_TypeDef *DMA2Dx)
{
  WRITE_REG(DMA2Dx->IFCR, DMA2D_IFCR_CCEIF);
}

/**
  * @brief  Clear DMA2D CLUT Transfer Complete Interrupt Flag
  * @rmtoll IFCR          CCTCIF          LL_DMA2D_ClearFlag_CTC
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_ClearFlag_CTC(DMA2D_TypeDef *DMA2Dx)
{
  WRITE_REG(DMA2Dx->IFCR, DMA2D_IFCR_CCTCIF);
}

/**
  * @brief  Clear DMA2D CLUT Access Error Interrupt Flag
  * @rmtoll IFCR          CAECIF          LL_DMA2D_ClearFlag_CAE
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_ClearFlag_CAE(DMA2D_TypeDef *DMA2Dx)
{
  WRITE_REG(DMA2Dx->IFCR, DMA2D_IFCR_CAECIF);
}

/**
  * @brief  Clear DMA2D Transfer Watermark Interrupt Flag
  * @rmtoll IFCR          CTWIF          LL_DMA2D_ClearFlag_TW
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_ClearFlag_TW(DMA2D_TypeDef *DMA2Dx)
{
  WRITE_REG(DMA2Dx->IFCR, DMA2D_IFCR_CTWIF);
}

/**
  * @brief  Clear DMA2D Transfer Complete Interrupt Flag
  * @rmtoll IFCR          CTCIF          LL_DMA2D_ClearFlag_TC
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_ClearFlag_TC(DMA2D_TypeDef *DMA2Dx)
{
  WRITE_REG(DMA2Dx->IFCR, DMA2D_IFCR_CTCIF);
}

/**
  * @brief  Clear DMA2D Transfer Error Interrupt Flag
  * @rmtoll IFCR          CTEIF          LL_DMA2D_ClearFlag_TE
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_ClearFlag_TE(DMA2D_TypeDef *DMA2Dx)
{
  WRITE_REG(DMA2Dx->IFCR, DMA2D_IFCR_CTEIF);
}

/**
  * @}
  */

/** @defgroup DMA2D_LL_EF_IT_MANAGEMENT Interruption Management
  * @{
  */

/**
  * @brief  Enable Configuration Error Interrupt
  * @rmtoll CR          CEIE        LL_DMA2D_EnableIT_CE
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_EnableIT_CE(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->CR, DMA2D_CR_CEIE);
}

/**
  * @brief  Enable CLUT Transfer Complete Interrupt
  * @rmtoll CR          CTCIE        LL_DMA2D_EnableIT_CTC
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_EnableIT_CTC(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->CR, DMA2D_CR_CTCIE);
}

/**
  * @brief  Enable CLUT Access Error Interrupt
  * @rmtoll CR          CAEIE        LL_DMA2D_EnableIT_CAE
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_EnableIT_CAE(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->CR, DMA2D_CR_CAEIE);
}

/**
  * @brief  Enable Transfer Watermark Interrupt
  * @rmtoll CR          TWIE        LL_DMA2D_EnableIT_TW
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_EnableIT_TW(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->CR, DMA2D_CR_TWIE);
}

/**
  * @brief  Enable Transfer Complete Interrupt
  * @rmtoll CR          TCIE        LL_DMA2D_EnableIT_TC
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_EnableIT_TC(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->CR, DMA2D_CR_TCIE);
}

/**
  * @brief  Enable Transfer Error Interrupt
  * @rmtoll CR          TEIE        LL_DMA2D_EnableIT_TE
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_EnableIT_TE(DMA2D_TypeDef *DMA2Dx)
{
  SET_BIT(DMA2Dx->CR, DMA2D_CR_TEIE);
}

/**
  * @brief  Disable Configuration Error Interrupt
  * @rmtoll CR          CEIE        LL_DMA2D_DisableIT_CE
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_DisableIT_CE(DMA2D_TypeDef *DMA2Dx)
{
  CLEAR_BIT(DMA2Dx->CR, DMA2D_CR_CEIE);
}

/**
  * @brief  Disable CLUT Transfer Complete Interrupt
  * @rmtoll CR          CTCIE        LL_DMA2D_DisableIT_CTC
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_DisableIT_CTC(DMA2D_TypeDef *DMA2Dx)
{
  CLEAR_BIT(DMA2Dx->CR, DMA2D_CR_CTCIE);
}

/**
  * @brief  Disable CLUT Access Error Interrupt
  * @rmtoll CR          CAEIE        LL_DMA2D_DisableIT_CAE
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_DisableIT_CAE(DMA2D_TypeDef *DMA2Dx)
{
  CLEAR_BIT(DMA2Dx->CR, DMA2D_CR_CAEIE);
}

/**
  * @brief  Disable Transfer Watermark Interrupt
  * @rmtoll CR          TWIE        LL_DMA2D_DisableIT_TW
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_DisableIT_TW(DMA2D_TypeDef *DMA2Dx)
{
  CLEAR_BIT(DMA2Dx->CR, DMA2D_CR_TWIE);
}

/**
  * @brief  Disable Transfer Complete Interrupt
  * @rmtoll CR          TCIE        LL_DMA2D_DisableIT_TC
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_DisableIT_TC(DMA2D_TypeDef *DMA2Dx)
{
  CLEAR_BIT(DMA2Dx->CR, DMA2D_CR_TCIE);
}

/**
  * @brief  Disable Transfer Error Interrupt
  * @rmtoll CR          TEIE        LL_DMA2D_DisableIT_TE
  * @param  DMA2Dx DMA2D Instance
  * @retval None
  */
__STATIC_INLINE void LL_DMA2D_DisableIT_TE(DMA2D_TypeDef *DMA2Dx)
{
  CLEAR_BIT(DMA2Dx->CR, DMA2D_CR_TEIE);
}

/**
  * @brief  Check if the DMA2D Configuration Error interrupt source is enabled or disabled.
  * @rmtoll CR          CEIE        LL_DMA2D_IsEnabledIT_CE
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsEnabledIT_CE(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->CR, DMA2D_CR_CEIE) == (DMA2D_CR_CEIE)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D CLUT Transfer Complete interrupt source is enabled or disabled.
  * @rmtoll CR          CTCIE        LL_DMA2D_IsEnabledIT_CTC
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsEnabledIT_CTC(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->CR, DMA2D_CR_CTCIE) == (DMA2D_CR_CTCIE)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D CLUT Access Error interrupt source is enabled or disabled.
  * @rmtoll CR          CAEIE        LL_DMA2D_IsEnabledIT_CAE
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsEnabledIT_CAE(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->CR, DMA2D_CR_CAEIE) == (DMA2D_CR_CAEIE)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D Transfer Watermark interrupt source is enabled or disabled.
  * @rmtoll CR          TWIE        LL_DMA2D_IsEnabledIT_TW
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsEnabledIT_TW(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->CR, DMA2D_CR_TWIE) == (DMA2D_CR_TWIE)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D Transfer Complete interrupt source is enabled or disabled.
  * @rmtoll CR          TCIE        LL_DMA2D_IsEnabledIT_TC
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsEnabledIT_TC(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->CR, DMA2D_CR_TCIE) == (DMA2D_CR_TCIE)) ? 1UL : 0UL);
}

/**
  * @brief  Check if the DMA2D Transfer Error interrupt source is enabled or disabled.
  * @rmtoll CR          TEIE        LL_DMA2D_IsEnabledIT_TE
  * @param  DMA2Dx DMA2D Instance
  * @retval State of bit (1 or 0).
  */
__STATIC_INLINE uint32_t LL_DMA2D_IsEnabledIT_TE(DMA2D_TypeDef *DMA2Dx)
{
  return ((READ_BIT(DMA2Dx->CR, DMA2D_CR_TEIE) == (DMA2D_CR_TEIE)) ? 1UL : 0UL);
}



/**
  * @}
  */

#if defined(USE_FULL_LL_DRIVER)
/** @defgroup DMA2D_LL_EF_Init_Functions Initialization and De-initialization Functions
  * @{
  */

ErrorStatus LL_DMA2D_DeInit(DMA2D_TypeDef *DMA2Dx);
ErrorStatus LL_DMA2D_Init(DMA2D_TypeDef *DMA2Dx, LL_DMA2D_InitTypeDef *DMA2D_InitStruct);
void LL_DMA2D_StructInit(LL_DMA2D_InitTypeDef *DMA2D_InitStruct);
void LL_DMA2D_ConfigLayer(DMA2D_TypeDef *DMA2Dx, LL_DMA2D_LayerCfgTypeDef *DMA2D_LayerCfg, uint32_t LayerIdx);
void LL_DMA2D_LayerCfgStructInit(LL_DMA2D_LayerCfgTypeDef *DMA2D_LayerCfg);
void LL_DMA2D_ConfigOutputColor(DMA2D_TypeDef *DMA2Dx, LL_DMA2D_ColorTypeDef *DMA2D_ColorStruct);
uint32_t LL_DMA2D_GetOutputBlueColor(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode);
uint32_t LL_DMA2D_GetOutputGreenColor(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode);
uint32_t LL_DMA2D_GetOutputRedColor(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode);
uint32_t LL_DMA2D_GetOutputAlphaColor(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode);
void LL_DMA2D_ConfigSize(DMA2D_TypeDef *DMA2Dx, uint32_t NbrOfLines, uint32_t NbrOfPixelsPerLines);

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

#endif /* defined (DMA2D) */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_LL_DMA2D_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
