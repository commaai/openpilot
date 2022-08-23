/**
  ******************************************************************************
  * @file    stm32f4xx_ll_dma2d.c
  * @author  MCD Application Team
  * @brief   DMA2D LL module driver.
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
#include "stm32f4xx_ll_dma2d.h"
#include "stm32f4xx_ll_bus.h"
#ifdef  USE_FULL_ASSERT
#include "stm32_assert.h"
#else
#define assert_param(expr) ((void)0U)
#endif  /* USE_FULL_ASSERT */

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (DMA2D)

/** @addtogroup DMA2D_LL
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @addtogroup DMA2D_LL_Private_Constants DMA2D Private Constants
  * @{
  */
#define LL_DMA2D_COLOR            0xFFU                                      /*!< Maximum output color setting                   */
#define LL_DMA2D_NUMBEROFLINES    DMA2D_NLR_NL                               /*!< Maximum number of lines                        */
#define LL_DMA2D_NUMBEROFPIXELS   (DMA2D_NLR_PL >> DMA2D_NLR_PL_Pos)         /*!< Maximum number of pixels per lines             */
#define LL_DMA2D_OFFSET_MAX       0x3FFFU                                    /*!< Maximum output line offset expressed in pixels */
#define LL_DMA2D_CLUTSIZE_MAX     0xFFU                                      /*!< Maximum CLUT size                              */
/**
  * @}
  */
/* Private macros ------------------------------------------------------------*/
/** @addtogroup DMA2D_LL_Private_Macros
  * @{
  */
#define IS_LL_DMA2D_MODE(MODE)          (((MODE) == LL_DMA2D_MODE_M2M)       || \
                                         ((MODE) == LL_DMA2D_MODE_M2M_PFC)   || \
                                         ((MODE) == LL_DMA2D_MODE_M2M_BLEND) || \
                                         ((MODE) == LL_DMA2D_MODE_R2M))

#define IS_LL_DMA2D_OCMODE(MODE_ARGB)   (((MODE_ARGB) == LL_DMA2D_OUTPUT_MODE_ARGB8888) || \
                                         ((MODE_ARGB) == LL_DMA2D_OUTPUT_MODE_RGB888)   || \
                                         ((MODE_ARGB) == LL_DMA2D_OUTPUT_MODE_RGB565)   || \
                                         ((MODE_ARGB) == LL_DMA2D_OUTPUT_MODE_ARGB1555) || \
                                         ((MODE_ARGB) == LL_DMA2D_OUTPUT_MODE_ARGB4444))

#define IS_LL_DMA2D_GREEN(GREEN)        ((GREEN) <= LL_DMA2D_COLOR)
#define IS_LL_DMA2D_RED(RED)            ((RED)   <= LL_DMA2D_COLOR)
#define IS_LL_DMA2D_BLUE(BLUE)          ((BLUE)  <= LL_DMA2D_COLOR)
#define IS_LL_DMA2D_ALPHA(ALPHA)        ((ALPHA) <= LL_DMA2D_COLOR)


#define IS_LL_DMA2D_OFFSET(OFFSET)      ((OFFSET) <= LL_DMA2D_OFFSET_MAX)

#define IS_LL_DMA2D_LINE(LINES)         ((LINES)  <= LL_DMA2D_NUMBEROFLINES)
#define IS_LL_DMA2D_PIXEL(PIXELS)       ((PIXELS) <= LL_DMA2D_NUMBEROFPIXELS)



#define IS_LL_DMA2D_LCMODE(MODE_ARGB)   (((MODE_ARGB) == LL_DMA2D_INPUT_MODE_ARGB8888) || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_RGB888)   || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_RGB565)   || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_ARGB1555) || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_ARGB4444) || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_L8)       || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_AL44)     || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_AL88)     || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_L4)       || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_A8)       || \
                                         ((MODE_ARGB) == LL_DMA2D_INPUT_MODE_A4))

#define IS_LL_DMA2D_CLUTCMODE(CLUTCMODE) (((CLUTCMODE) == LL_DMA2D_CLUT_COLOR_MODE_ARGB8888) || \
                                          ((CLUTCMODE) == LL_DMA2D_CLUT_COLOR_MODE_RGB888))

#define IS_LL_DMA2D_CLUTSIZE(SIZE)      ((SIZE) <= LL_DMA2D_CLUTSIZE_MAX)

#define IS_LL_DMA2D_ALPHAMODE(MODE)     (((MODE) == LL_DMA2D_ALPHA_MODE_NO_MODIF) || \
                                         ((MODE) == LL_DMA2D_ALPHA_MODE_REPLACE)  || \
                                         ((MODE) == LL_DMA2D_ALPHA_MODE_COMBINE))


/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup DMA2D_LL_Exported_Functions
  * @{
  */

/** @addtogroup DMA2D_LL_EF_Init_Functions Initialization and De-initialization Functions
  * @{
  */

/**
  * @brief  De-initialize DMA2D registers (registers restored to their default values).
  * @param  DMA2Dx DMA2D Instance
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: DMA2D registers are de-initialized
  *          - ERROR: DMA2D registers are not de-initialized
  */
ErrorStatus LL_DMA2D_DeInit(DMA2D_TypeDef *DMA2Dx)
{
  ErrorStatus status = SUCCESS;

  /* Check the parameters */
  assert_param(IS_DMA2D_ALL_INSTANCE(DMA2Dx));

  if (DMA2Dx == DMA2D)
  {
    /* Force reset of DMA2D clock */
    LL_AHB1_GRP1_ForceReset(LL_AHB1_GRP1_PERIPH_DMA2D);

    /* Release reset of DMA2D clock */
    LL_AHB1_GRP1_ReleaseReset(LL_AHB1_GRP1_PERIPH_DMA2D);
  }
  else
  {
    status = ERROR;
  }

  return (status);
}

/**
  * @brief  Initialize DMA2D registers according to the specified parameters in DMA2D_InitStruct.
  * @note   DMA2D transfers must be disabled to set initialization bits in configuration registers,
  *         otherwise ERROR result is returned.
  * @param  DMA2Dx DMA2D Instance
  * @param  DMA2D_InitStruct  pointer to a LL_DMA2D_InitTypeDef structure
  *         that contains the configuration information for the specified DMA2D peripheral.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: DMA2D registers are initialized according to DMA2D_InitStruct content
  *          - ERROR: Issue occurred during DMA2D registers initialization
  */
ErrorStatus LL_DMA2D_Init(DMA2D_TypeDef *DMA2Dx, LL_DMA2D_InitTypeDef *DMA2D_InitStruct)
{
  ErrorStatus status = ERROR;
  LL_DMA2D_ColorTypeDef dma2d_colorstruct;
  uint32_t tmp;
  uint32_t tmp1;
  uint32_t tmp2;
  uint32_t regMask;
  uint32_t regValue;

  /* Check the parameters */
  assert_param(IS_DMA2D_ALL_INSTANCE(DMA2Dx));
  assert_param(IS_LL_DMA2D_MODE(DMA2D_InitStruct->Mode));
  assert_param(IS_LL_DMA2D_OCMODE(DMA2D_InitStruct->ColorMode));
  assert_param(IS_LL_DMA2D_LINE(DMA2D_InitStruct->NbrOfLines));
  assert_param(IS_LL_DMA2D_PIXEL(DMA2D_InitStruct->NbrOfPixelsPerLines));
  assert_param(IS_LL_DMA2D_GREEN(DMA2D_InitStruct->OutputGreen));
  assert_param(IS_LL_DMA2D_RED(DMA2D_InitStruct->OutputRed));
  assert_param(IS_LL_DMA2D_BLUE(DMA2D_InitStruct->OutputBlue));
  assert_param(IS_LL_DMA2D_ALPHA(DMA2D_InitStruct->OutputAlpha));
  assert_param(IS_LL_DMA2D_OFFSET(DMA2D_InitStruct->LineOffset));

  /* DMA2D transfers must be disabled to configure bits in initialization registers */
  tmp = LL_DMA2D_IsTransferOngoing(DMA2Dx);
  tmp1 = LL_DMA2D_FGND_IsEnabledCLUTLoad(DMA2Dx);
  tmp2 = LL_DMA2D_BGND_IsEnabledCLUTLoad(DMA2Dx);
  if ((tmp == 0U) && (tmp1 == 0U) && (tmp2 == 0U))
  {
    /* DMA2D CR register configuration -------------------------------------------*/
    LL_DMA2D_SetMode(DMA2Dx, DMA2D_InitStruct->Mode);

    /* DMA2D OPFCCR register configuration ---------------------------------------*/
    regMask = DMA2D_OPFCCR_CM;
    regValue = DMA2D_InitStruct->ColorMode;




    MODIFY_REG(DMA2Dx->OPFCCR, regMask, regValue);

    /* DMA2D OOR register configuration ------------------------------------------*/
    LL_DMA2D_SetLineOffset(DMA2Dx, DMA2D_InitStruct->LineOffset);

    /* DMA2D NLR register configuration ------------------------------------------*/
    LL_DMA2D_ConfigSize(DMA2Dx, DMA2D_InitStruct->NbrOfLines, DMA2D_InitStruct->NbrOfPixelsPerLines);

    /* DMA2D OMAR register configuration ------------------------------------------*/
    LL_DMA2D_SetOutputMemAddr(DMA2Dx, DMA2D_InitStruct->OutputMemoryAddress);

    /* DMA2D OCOLR register configuration ------------------------------------------*/
    dma2d_colorstruct.ColorMode   = DMA2D_InitStruct->ColorMode;
    dma2d_colorstruct.OutputBlue  = DMA2D_InitStruct->OutputBlue;
    dma2d_colorstruct.OutputGreen = DMA2D_InitStruct->OutputGreen;
    dma2d_colorstruct.OutputRed   = DMA2D_InitStruct->OutputRed;
    dma2d_colorstruct.OutputAlpha = DMA2D_InitStruct->OutputAlpha;
    LL_DMA2D_ConfigOutputColor(DMA2Dx, &dma2d_colorstruct);

    status = SUCCESS;
  }
  /* If DMA2D transfers are not disabled, return ERROR */

  return (status);
}

/**
  * @brief Set each @ref LL_DMA2D_InitTypeDef field to default value.
  * @param DMA2D_InitStruct  pointer to a @ref LL_DMA2D_InitTypeDef structure
  *                          whose fields will be set to default values.
  * @retval None
  */
void LL_DMA2D_StructInit(LL_DMA2D_InitTypeDef *DMA2D_InitStruct)
{
  /* Set DMA2D_InitStruct fields to default values */
  DMA2D_InitStruct->Mode                = LL_DMA2D_MODE_M2M;
  DMA2D_InitStruct->ColorMode           = LL_DMA2D_OUTPUT_MODE_ARGB8888;
  DMA2D_InitStruct->NbrOfLines          = 0x0U;
  DMA2D_InitStruct->NbrOfPixelsPerLines = 0x0U;
  DMA2D_InitStruct->LineOffset          = 0x0U;
  DMA2D_InitStruct->OutputBlue          = 0x0U;
  DMA2D_InitStruct->OutputGreen         = 0x0U;
  DMA2D_InitStruct->OutputRed           = 0x0U;
  DMA2D_InitStruct->OutputAlpha         = 0x0U;
  DMA2D_InitStruct->OutputMemoryAddress = 0x0U;
}

/**
  * @brief  Configure the foreground or background according to the specified parameters
  *         in the LL_DMA2D_LayerCfgTypeDef structure.
  * @param  DMA2Dx DMA2D Instance
  * @param  DMA2D_LayerCfg  pointer to a LL_DMA2D_LayerCfgTypeDef structure that contains
  *         the configuration information for the specified layer.
  * @param  LayerIdx  DMA2D Layer index.
  *                   This parameter can be one of the following values:
  *                   0(background) / 1(foreground)
  * @retval None
  */
void LL_DMA2D_ConfigLayer(DMA2D_TypeDef *DMA2Dx, LL_DMA2D_LayerCfgTypeDef *DMA2D_LayerCfg, uint32_t LayerIdx)
{
  /* Check the parameters */
  assert_param(IS_LL_DMA2D_OFFSET(DMA2D_LayerCfg->LineOffset));
  assert_param(IS_LL_DMA2D_LCMODE(DMA2D_LayerCfg->ColorMode));
  assert_param(IS_LL_DMA2D_CLUTCMODE(DMA2D_LayerCfg->CLUTColorMode));
  assert_param(IS_LL_DMA2D_CLUTSIZE(DMA2D_LayerCfg->CLUTSize));
  assert_param(IS_LL_DMA2D_ALPHAMODE(DMA2D_LayerCfg->AlphaMode));
  assert_param(IS_LL_DMA2D_GREEN(DMA2D_LayerCfg->Green));
  assert_param(IS_LL_DMA2D_RED(DMA2D_LayerCfg->Red));
  assert_param(IS_LL_DMA2D_BLUE(DMA2D_LayerCfg->Blue));
  assert_param(IS_LL_DMA2D_ALPHA(DMA2D_LayerCfg->Alpha));


  if (LayerIdx == 0U)
  {
    /* Configure the background memory address */
    LL_DMA2D_BGND_SetMemAddr(DMA2Dx, DMA2D_LayerCfg->MemoryAddress);

    /* Configure the background line offset */
    LL_DMA2D_BGND_SetLineOffset(DMA2Dx, DMA2D_LayerCfg->LineOffset);

    /* Configure the background Alpha value, Alpha mode, CLUT size, CLUT Color mode and Color mode */
    MODIFY_REG(DMA2Dx->BGPFCCR, \
               (DMA2D_BGPFCCR_ALPHA | DMA2D_BGPFCCR_AM | DMA2D_BGPFCCR_CS | DMA2D_BGPFCCR_CCM | DMA2D_BGPFCCR_CM), \
               ((DMA2D_LayerCfg->Alpha << DMA2D_BGPFCCR_ALPHA_Pos) | DMA2D_LayerCfg->AlphaMode | \
                (DMA2D_LayerCfg->CLUTSize << DMA2D_BGPFCCR_CS_Pos) | DMA2D_LayerCfg->CLUTColorMode | \
                DMA2D_LayerCfg->ColorMode));

    /* Configure the background color */
    LL_DMA2D_BGND_SetColor(DMA2Dx, DMA2D_LayerCfg->Red, DMA2D_LayerCfg->Green, DMA2D_LayerCfg->Blue);

    /* Configure the background CLUT memory address */
    LL_DMA2D_BGND_SetCLUTMemAddr(DMA2Dx, DMA2D_LayerCfg->CLUTMemoryAddress);
  }
  else
  {
    /* Configure the foreground memory address */
    LL_DMA2D_FGND_SetMemAddr(DMA2Dx, DMA2D_LayerCfg->MemoryAddress);

    /* Configure the foreground line offset */
    LL_DMA2D_FGND_SetLineOffset(DMA2Dx, DMA2D_LayerCfg->LineOffset);

    /* Configure the foreground Alpha value, Alpha mode, CLUT size, CLUT Color mode and Color mode */
    MODIFY_REG(DMA2Dx->FGPFCCR, \
               (DMA2D_FGPFCCR_ALPHA | DMA2D_FGPFCCR_AM | DMA2D_FGPFCCR_CS | DMA2D_FGPFCCR_CCM | DMA2D_FGPFCCR_CM), \
               ((DMA2D_LayerCfg->Alpha << DMA2D_FGPFCCR_ALPHA_Pos) | DMA2D_LayerCfg->AlphaMode | \
                (DMA2D_LayerCfg->CLUTSize << DMA2D_FGPFCCR_CS_Pos) | DMA2D_LayerCfg->CLUTColorMode | \
                DMA2D_LayerCfg->ColorMode));

    /* Configure the foreground color */
    LL_DMA2D_FGND_SetColor(DMA2Dx, DMA2D_LayerCfg->Red, DMA2D_LayerCfg->Green, DMA2D_LayerCfg->Blue);

    /* Configure the foreground CLUT memory address */
    LL_DMA2D_FGND_SetCLUTMemAddr(DMA2Dx, DMA2D_LayerCfg->CLUTMemoryAddress);
  }
}

/**
  * @brief Set each @ref LL_DMA2D_LayerCfgTypeDef field to default value.
  * @param DMA2D_LayerCfg  pointer to a @ref LL_DMA2D_LayerCfgTypeDef structure
  *                        whose fields will be set to default values.
  * @retval None
  */
void LL_DMA2D_LayerCfgStructInit(LL_DMA2D_LayerCfgTypeDef *DMA2D_LayerCfg)
{
  /* Set DMA2D_LayerCfg fields to default values */
  DMA2D_LayerCfg->MemoryAddress      = 0x0U;
  DMA2D_LayerCfg->ColorMode          = LL_DMA2D_INPUT_MODE_ARGB8888;
  DMA2D_LayerCfg->LineOffset         = 0x0U;
  DMA2D_LayerCfg->CLUTColorMode      = LL_DMA2D_CLUT_COLOR_MODE_ARGB8888;
  DMA2D_LayerCfg->CLUTSize           = 0x0U;
  DMA2D_LayerCfg->AlphaMode          = LL_DMA2D_ALPHA_MODE_NO_MODIF;
  DMA2D_LayerCfg->Alpha              = 0x0U;
  DMA2D_LayerCfg->Blue               = 0x0U;
  DMA2D_LayerCfg->Green              = 0x0U;
  DMA2D_LayerCfg->Red                = 0x0U;
  DMA2D_LayerCfg->CLUTMemoryAddress  = 0x0U;
}

/**
  * @brief  Initialize DMA2D output color register according to the specified parameters
  *         in DMA2D_ColorStruct.
  * @param  DMA2Dx DMA2D Instance
  * @param  DMA2D_ColorStruct  pointer to a LL_DMA2D_ColorTypeDef structure that contains
  *         the color configuration information for the specified DMA2D peripheral.
  * @retval None
  */
void LL_DMA2D_ConfigOutputColor(DMA2D_TypeDef *DMA2Dx, LL_DMA2D_ColorTypeDef *DMA2D_ColorStruct)
{
  uint32_t outgreen;
  uint32_t outred;
  uint32_t outalpha;

  /* Check the parameters */
  assert_param(IS_DMA2D_ALL_INSTANCE(DMA2Dx));
  assert_param(IS_LL_DMA2D_OCMODE(DMA2D_ColorStruct->ColorMode));
  assert_param(IS_LL_DMA2D_GREEN(DMA2D_ColorStruct->OutputGreen));
  assert_param(IS_LL_DMA2D_RED(DMA2D_ColorStruct->OutputRed));
  assert_param(IS_LL_DMA2D_BLUE(DMA2D_ColorStruct->OutputBlue));
  assert_param(IS_LL_DMA2D_ALPHA(DMA2D_ColorStruct->OutputAlpha));

  /* DMA2D OCOLR register configuration ------------------------------------------*/
  if (DMA2D_ColorStruct->ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB8888)
  {
    outgreen = DMA2D_ColorStruct->OutputGreen << 8U;
    outred = DMA2D_ColorStruct->OutputRed << 16U;
    outalpha = DMA2D_ColorStruct->OutputAlpha << 24U;
  }
  else if (DMA2D_ColorStruct->ColorMode == LL_DMA2D_OUTPUT_MODE_RGB888)
  {
    outgreen = DMA2D_ColorStruct->OutputGreen << 8U;
    outred = DMA2D_ColorStruct->OutputRed << 16U;
    outalpha = 0x00000000U;
  }
  else if (DMA2D_ColorStruct->ColorMode == LL_DMA2D_OUTPUT_MODE_RGB565)
  {
    outgreen = DMA2D_ColorStruct->OutputGreen << 5U;
    outred = DMA2D_ColorStruct->OutputRed << 11U;
    outalpha = 0x00000000U;
  }
  else if (DMA2D_ColorStruct->ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB1555)
  {
    outgreen = DMA2D_ColorStruct->OutputGreen << 5U;
    outred = DMA2D_ColorStruct->OutputRed << 10U;
    outalpha = DMA2D_ColorStruct->OutputAlpha << 15U;
  }
  else /* ColorMode = LL_DMA2D_OUTPUT_MODE_ARGB4444 */
  {
    outgreen = DMA2D_ColorStruct->OutputGreen << 4U;
    outred = DMA2D_ColorStruct->OutputRed << 8U;
    outalpha = DMA2D_ColorStruct->OutputAlpha << 12U;
  }
  LL_DMA2D_SetOutputColor(DMA2Dx, (outgreen | outred | DMA2D_ColorStruct->OutputBlue | outalpha));
}

/**
  * @brief  Return DMA2D output Blue color.
  * @param  DMA2Dx DMA2D Instance.
  * @param  ColorMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB4444
  * @retval Output Blue color value between Min_Data=0 and Max_Data=0xFF
  */
uint32_t LL_DMA2D_GetOutputBlueColor(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode)
{
  uint32_t color;

  /* Check the parameters */
  assert_param(IS_DMA2D_ALL_INSTANCE(DMA2Dx));
  assert_param(IS_LL_DMA2D_OCMODE(ColorMode));

  /* DMA2D OCOLR register reading ------------------------------------------*/
  if (ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB8888)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xFFU));
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_RGB888)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xFFU));
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_RGB565)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0x1FU));
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB1555)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0x1FU));
  }
  else /* ColorMode = LL_DMA2D_OUTPUT_MODE_ARGB4444 */
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xFU));
  }

  return color;
}

/**
  * @brief  Return DMA2D output Green color.
  * @param  DMA2Dx DMA2D Instance.
  * @param  ColorMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB4444
  * @retval Output Green color value between Min_Data=0 and Max_Data=0xFF
  */
uint32_t LL_DMA2D_GetOutputGreenColor(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode)
{
  uint32_t color;

  /* Check the parameters */
  assert_param(IS_DMA2D_ALL_INSTANCE(DMA2Dx));
  assert_param(IS_LL_DMA2D_OCMODE(ColorMode));

  /* DMA2D OCOLR register reading ------------------------------------------*/
  if (ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB8888)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xFF00U) >> 8U);
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_RGB888)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xFF00U) >> 8U);
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_RGB565)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0x7E0U) >> 5U);
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB1555)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0x3E0U) >> 5U);
  }
  else /* ColorMode = LL_DMA2D_OUTPUT_MODE_ARGB4444 */
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xF0U) >> 4U);
  }

  return color;
}

/**
  * @brief  Return DMA2D output Red color.
  * @param  DMA2Dx DMA2D Instance.
  * @param  ColorMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB4444
  * @retval Output Red color value between Min_Data=0 and Max_Data=0xFF
  */
uint32_t LL_DMA2D_GetOutputRedColor(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode)
{
  uint32_t color;

  /* Check the parameters */
  assert_param(IS_DMA2D_ALL_INSTANCE(DMA2Dx));
  assert_param(IS_LL_DMA2D_OCMODE(ColorMode));

  /* DMA2D OCOLR register reading ------------------------------------------*/
  if (ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB8888)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xFF0000U) >> 16U);
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_RGB888)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xFF0000U) >> 16U);
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_RGB565)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xF800U) >> 11U);
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB1555)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0x7C00U) >> 10U);
  }
  else /* ColorMode = LL_DMA2D_OUTPUT_MODE_ARGB4444 */
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xF00U) >> 8U);
  }

  return color;
}

/**
  * @brief  Return DMA2D output Alpha color.
  * @param  DMA2Dx DMA2D Instance.
  * @param  ColorMode This parameter can be one of the following values:
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB8888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB888
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_RGB565
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB1555
  *         @arg @ref LL_DMA2D_OUTPUT_MODE_ARGB4444
  * @retval Output Alpha color value between Min_Data=0 and Max_Data=0xFF
  */
uint32_t LL_DMA2D_GetOutputAlphaColor(DMA2D_TypeDef *DMA2Dx, uint32_t ColorMode)
{
  uint32_t color;

  /* Check the parameters */
  assert_param(IS_DMA2D_ALL_INSTANCE(DMA2Dx));
  assert_param(IS_LL_DMA2D_OCMODE(ColorMode));

  /* DMA2D OCOLR register reading ------------------------------------------*/
  if (ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB8888)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xFF000000U) >> 24U);
  }
  else if ((ColorMode == LL_DMA2D_OUTPUT_MODE_RGB888) || (ColorMode == LL_DMA2D_OUTPUT_MODE_RGB565))
  {
    color = 0x0U;
  }
  else if (ColorMode == LL_DMA2D_OUTPUT_MODE_ARGB1555)
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0x8000U) >> 15U);
  }
  else /* ColorMode = LL_DMA2D_OUTPUT_MODE_ARGB4444 */
  {
    color = (uint32_t)(READ_BIT(DMA2Dx->OCOLR, 0xF000U) >> 12U);
  }

  return color;
}

/**
  * @brief  Configure DMA2D transfer size.
  * @param  DMA2Dx DMA2D Instance
  * @param  NbrOfLines Value between Min_Data=0 and Max_Data=0xFFFF
  * @param  NbrOfPixelsPerLines Value between Min_Data=0 and Max_Data=0x3FFF
  * @retval None
  */
void LL_DMA2D_ConfigSize(DMA2D_TypeDef *DMA2Dx, uint32_t NbrOfLines, uint32_t NbrOfPixelsPerLines)
{
  MODIFY_REG(DMA2Dx->NLR, (DMA2D_NLR_PL | DMA2D_NLR_NL), \
             ((NbrOfPixelsPerLines << DMA2D_NLR_PL_Pos) | NbrOfLines));
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

#endif /* defined (DMA2D) */

/**
  * @}
  */

#endif /* USE_FULL_LL_DRIVER */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/

