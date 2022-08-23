/**
  ******************************************************************************
  * @file    stm32f4xx_ll_dma.c
  * @author  MCD Application Team
  * @brief   DMA LL module driver.
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
#if defined(USE_FULL_LL_DRIVER)

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_ll_dma.h"
#include "stm32f4xx_ll_bus.h"
#ifdef  USE_FULL_ASSERT
#include "stm32_assert.h"
#else
#define assert_param(expr) ((void)0U)
#endif

/** @addtogroup STM32F4xx_LL_Driver
  * @{
  */

#if defined (DMA1) || defined (DMA2)

/** @defgroup DMA_LL DMA
  * @{
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/** @addtogroup DMA_LL_Private_Macros
  * @{
  */
#define IS_LL_DMA_DIRECTION(__VALUE__)          (((__VALUE__) == LL_DMA_DIRECTION_PERIPH_TO_MEMORY) || \
                                                 ((__VALUE__) == LL_DMA_DIRECTION_MEMORY_TO_PERIPH) || \
                                                 ((__VALUE__) == LL_DMA_DIRECTION_MEMORY_TO_MEMORY))

#define IS_LL_DMA_MODE(__VALUE__)               (((__VALUE__) == LL_DMA_MODE_NORMAL)    || \
                                                 ((__VALUE__) == LL_DMA_MODE_CIRCULAR)  || \
                                                 ((__VALUE__) == LL_DMA_MODE_PFCTRL))

#define IS_LL_DMA_PERIPHINCMODE(__VALUE__)      (((__VALUE__) == LL_DMA_PERIPH_INCREMENT) || \
                                                 ((__VALUE__) == LL_DMA_PERIPH_NOINCREMENT))

#define IS_LL_DMA_MEMORYINCMODE(__VALUE__)      (((__VALUE__) == LL_DMA_MEMORY_INCREMENT) || \
                                                 ((__VALUE__) == LL_DMA_MEMORY_NOINCREMENT))

#define IS_LL_DMA_PERIPHDATASIZE(__VALUE__)     (((__VALUE__) == LL_DMA_PDATAALIGN_BYTE)      || \
                                                 ((__VALUE__) == LL_DMA_PDATAALIGN_HALFWORD)  || \
                                                 ((__VALUE__) == LL_DMA_PDATAALIGN_WORD))

#define IS_LL_DMA_MEMORYDATASIZE(__VALUE__)     (((__VALUE__) == LL_DMA_MDATAALIGN_BYTE)      || \
                                                 ((__VALUE__) == LL_DMA_MDATAALIGN_HALFWORD)  || \
                                                 ((__VALUE__) == LL_DMA_MDATAALIGN_WORD))

#define IS_LL_DMA_NBDATA(__VALUE__)             ((__VALUE__)  <= 0x0000FFFFU)

#define IS_LL_DMA_CHANNEL(__VALUE__)            (((__VALUE__) == LL_DMA_CHANNEL_0)  || \
                                                 ((__VALUE__) == LL_DMA_CHANNEL_1)  || \
                                                 ((__VALUE__) == LL_DMA_CHANNEL_2)  || \
                                                 ((__VALUE__) == LL_DMA_CHANNEL_3)  || \
                                                 ((__VALUE__) == LL_DMA_CHANNEL_4)  || \
                                                 ((__VALUE__) == LL_DMA_CHANNEL_5)  || \
                                                 ((__VALUE__) == LL_DMA_CHANNEL_6)  || \
                                                 ((__VALUE__) == LL_DMA_CHANNEL_7))

#define IS_LL_DMA_PRIORITY(__VALUE__)           (((__VALUE__) == LL_DMA_PRIORITY_LOW)    || \
                                                 ((__VALUE__) == LL_DMA_PRIORITY_MEDIUM) || \
                                                 ((__VALUE__) == LL_DMA_PRIORITY_HIGH)   || \
                                                 ((__VALUE__) == LL_DMA_PRIORITY_VERYHIGH))

#define IS_LL_DMA_ALL_STREAM_INSTANCE(INSTANCE, STREAM)   ((((INSTANCE) == DMA1) && \
                                                           (((STREAM) == LL_DMA_STREAM_0) || \
                                                            ((STREAM) == LL_DMA_STREAM_1) || \
                                                            ((STREAM) == LL_DMA_STREAM_2) || \
                                                            ((STREAM) == LL_DMA_STREAM_3) || \
                                                            ((STREAM) == LL_DMA_STREAM_4) || \
                                                            ((STREAM) == LL_DMA_STREAM_5) || \
                                                            ((STREAM) == LL_DMA_STREAM_6) || \
                                                            ((STREAM) == LL_DMA_STREAM_7) || \
                                                            ((STREAM) == LL_DMA_STREAM_ALL))) ||\
                                                            (((INSTANCE) == DMA2) && \
                                                          (((STREAM) == LL_DMA_STREAM_0) || \
                                                           ((STREAM) == LL_DMA_STREAM_1) || \
                                                           ((STREAM) == LL_DMA_STREAM_2) || \
                                                           ((STREAM) == LL_DMA_STREAM_3) || \
                                                           ((STREAM) == LL_DMA_STREAM_4) || \
                                                           ((STREAM) == LL_DMA_STREAM_5) || \
                                                           ((STREAM) == LL_DMA_STREAM_6) || \
                                                           ((STREAM) == LL_DMA_STREAM_7) || \
                                                           ((STREAM) == LL_DMA_STREAM_ALL))))

#define IS_LL_DMA_FIFO_MODE_STATE(STATE) (((STATE) == LL_DMA_FIFOMODE_DISABLE ) || \
                                          ((STATE) == LL_DMA_FIFOMODE_ENABLE))

#define IS_LL_DMA_FIFO_THRESHOLD(THRESHOLD) (((THRESHOLD) == LL_DMA_FIFOTHRESHOLD_1_4) || \
                                             ((THRESHOLD) == LL_DMA_FIFOTHRESHOLD_1_2)  || \
                                             ((THRESHOLD) == LL_DMA_FIFOTHRESHOLD_3_4)  || \
                                             ((THRESHOLD) == LL_DMA_FIFOTHRESHOLD_FULL))

#define IS_LL_DMA_MEMORY_BURST(BURST) (((BURST) == LL_DMA_MBURST_SINGLE) || \
                                       ((BURST) == LL_DMA_MBURST_INC4)   || \
                                       ((BURST) == LL_DMA_MBURST_INC8)   || \
                                       ((BURST) == LL_DMA_MBURST_INC16))

#define IS_LL_DMA_PERIPHERAL_BURST(BURST) (((BURST) == LL_DMA_PBURST_SINGLE) || \
                                           ((BURST) == LL_DMA_PBURST_INC4)   || \
                                           ((BURST) == LL_DMA_PBURST_INC8)   || \
                                           ((BURST) == LL_DMA_PBURST_INC16))

/**
  * @}
  */

/* Private function prototypes -----------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup DMA_LL_Exported_Functions
  * @{
  */

/** @addtogroup DMA_LL_EF_Init
  * @{
  */

/**
  * @brief  De-initialize the DMA registers to their default reset values.
  * @param  DMAx DMAx Instance
  * @param  Stream This parameter can be one of the following values:
  *         @arg @ref LL_DMA_STREAM_0
  *         @arg @ref LL_DMA_STREAM_1
  *         @arg @ref LL_DMA_STREAM_2
  *         @arg @ref LL_DMA_STREAM_3
  *         @arg @ref LL_DMA_STREAM_4
  *         @arg @ref LL_DMA_STREAM_5
  *         @arg @ref LL_DMA_STREAM_6
  *         @arg @ref LL_DMA_STREAM_7
  *         @arg @ref LL_DMA_STREAM_ALL
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: DMA registers are de-initialized
  *          - ERROR: DMA registers are not de-initialized
  */
uint32_t LL_DMA_DeInit(DMA_TypeDef *DMAx, uint32_t Stream)
{
  DMA_Stream_TypeDef *tmp = (DMA_Stream_TypeDef *)DMA1_Stream0;
  ErrorStatus status = SUCCESS;

  /* Check the DMA Instance DMAx and Stream parameters*/
  assert_param(IS_LL_DMA_ALL_STREAM_INSTANCE(DMAx, Stream));

  if (Stream == LL_DMA_STREAM_ALL)
  {
    if (DMAx == DMA1)
    {
      /* Force reset of DMA clock */
      LL_AHB1_GRP1_ForceReset(LL_AHB1_GRP1_PERIPH_DMA1);

      /* Release reset of DMA clock */
      LL_AHB1_GRP1_ReleaseReset(LL_AHB1_GRP1_PERIPH_DMA1);
    }
    else if (DMAx == DMA2)
    {
      /* Force reset of DMA clock */
      LL_AHB1_GRP1_ForceReset(LL_AHB1_GRP1_PERIPH_DMA2);

      /* Release reset of DMA clock */
      LL_AHB1_GRP1_ReleaseReset(LL_AHB1_GRP1_PERIPH_DMA2);
    }
    else
    {
      status = ERROR;
    }
  }
  else
  {
    /* Disable the selected Stream */
    LL_DMA_DisableStream(DMAx,Stream);

    /* Get the DMA Stream Instance */
    tmp = (DMA_Stream_TypeDef *)(__LL_DMA_GET_STREAM_INSTANCE(DMAx, Stream));

    /* Reset DMAx_Streamy configuration register */
    LL_DMA_WriteReg(tmp, CR, 0U);

    /* Reset DMAx_Streamy remaining bytes register */
    LL_DMA_WriteReg(tmp, NDTR, 0U);

    /* Reset DMAx_Streamy peripheral address register */
    LL_DMA_WriteReg(tmp, PAR, 0U);

    /* Reset DMAx_Streamy memory address register */
    LL_DMA_WriteReg(tmp, M0AR, 0U);

    /* Reset DMAx_Streamy memory address register */
    LL_DMA_WriteReg(tmp, M1AR, 0U);

    /* Reset DMAx_Streamy FIFO control register */
    LL_DMA_WriteReg(tmp, FCR, 0x00000021U);

    /* Reset Channel register field for DMAx Stream*/
    LL_DMA_SetChannelSelection(DMAx, Stream, LL_DMA_CHANNEL_0);

    if(Stream == LL_DMA_STREAM_0)
    {
       /* Reset the Stream0 pending flags */
       DMAx->LIFCR = 0x0000003FU;
    }
    else if(Stream == LL_DMA_STREAM_1)
    {
       /* Reset the Stream1 pending flags */
       DMAx->LIFCR = 0x00000F40U;
    }
    else if(Stream == LL_DMA_STREAM_2)
    {
       /* Reset the Stream2 pending flags */
       DMAx->LIFCR = 0x003F0000U;
    }
    else if(Stream == LL_DMA_STREAM_3)
    {
       /* Reset the Stream3 pending flags */
       DMAx->LIFCR = 0x0F400000U;
    }
    else if(Stream == LL_DMA_STREAM_4)
    {
       /* Reset the Stream4 pending flags */
       DMAx->HIFCR = 0x0000003FU;
    }
    else if(Stream == LL_DMA_STREAM_5)
    {
       /* Reset the Stream5 pending flags */
       DMAx->HIFCR = 0x00000F40U;
    }
    else if(Stream == LL_DMA_STREAM_6)
    {
       /* Reset the Stream6 pending flags */
       DMAx->HIFCR = 0x003F0000U;
    }
    else if(Stream == LL_DMA_STREAM_7)
    {
       /* Reset the Stream7 pending flags */
       DMAx->HIFCR = 0x0F400000U;
    }
    else
    {
      status = ERROR;
    }
  }

  return status;
}

/**
  * @brief  Initialize the DMA registers according to the specified parameters in DMA_InitStruct.
  * @note   To convert DMAx_Streamy Instance to DMAx Instance and Streamy, use helper macros :
  *         @arg @ref __LL_DMA_GET_INSTANCE
  *         @arg @ref __LL_DMA_GET_STREAM
  * @param  DMAx DMAx Instance
  * @param  Stream This parameter can be one of the following values:
  *         @arg @ref LL_DMA_STREAM_0
  *         @arg @ref LL_DMA_STREAM_1
  *         @arg @ref LL_DMA_STREAM_2
  *         @arg @ref LL_DMA_STREAM_3
  *         @arg @ref LL_DMA_STREAM_4
  *         @arg @ref LL_DMA_STREAM_5
  *         @arg @ref LL_DMA_STREAM_6
  *         @arg @ref LL_DMA_STREAM_7
  * @param  DMA_InitStruct pointer to a @ref LL_DMA_InitTypeDef structure.
  * @retval An ErrorStatus enumeration value:
  *          - SUCCESS: DMA registers are initialized
  *          - ERROR: Not applicable
  */
uint32_t LL_DMA_Init(DMA_TypeDef *DMAx, uint32_t Stream, LL_DMA_InitTypeDef *DMA_InitStruct)
{
  /* Check the DMA Instance DMAx and Stream parameters*/
  assert_param(IS_LL_DMA_ALL_STREAM_INSTANCE(DMAx, Stream));

  /* Check the DMA parameters from DMA_InitStruct */
  assert_param(IS_LL_DMA_DIRECTION(DMA_InitStruct->Direction));
  assert_param(IS_LL_DMA_MODE(DMA_InitStruct->Mode));
  assert_param(IS_LL_DMA_PERIPHINCMODE(DMA_InitStruct->PeriphOrM2MSrcIncMode));
  assert_param(IS_LL_DMA_MEMORYINCMODE(DMA_InitStruct->MemoryOrM2MDstIncMode));
  assert_param(IS_LL_DMA_PERIPHDATASIZE(DMA_InitStruct->PeriphOrM2MSrcDataSize));
  assert_param(IS_LL_DMA_MEMORYDATASIZE(DMA_InitStruct->MemoryOrM2MDstDataSize));
  assert_param(IS_LL_DMA_NBDATA(DMA_InitStruct->NbData));
  assert_param(IS_LL_DMA_CHANNEL(DMA_InitStruct->Channel));
  assert_param(IS_LL_DMA_PRIORITY(DMA_InitStruct->Priority));
  assert_param(IS_LL_DMA_FIFO_MODE_STATE(DMA_InitStruct->FIFOMode));
  /* Check the memory burst, peripheral burst and FIFO threshold parameters only
     when FIFO mode is enabled */
  if(DMA_InitStruct->FIFOMode != LL_DMA_FIFOMODE_DISABLE)
  {
    assert_param(IS_LL_DMA_FIFO_THRESHOLD(DMA_InitStruct->FIFOThreshold));
    assert_param(IS_LL_DMA_MEMORY_BURST(DMA_InitStruct->MemBurst));
    assert_param(IS_LL_DMA_PERIPHERAL_BURST(DMA_InitStruct->PeriphBurst));
  }

  /*---------------------------- DMAx SxCR Configuration ------------------------
   * Configure DMAx_Streamy: data transfer direction, data transfer mode,
   *                          peripheral and memory increment mode,
   *                          data size alignment and  priority level with parameters :
   * - Direction:      DMA_SxCR_DIR[1:0] bits
   * - Mode:           DMA_SxCR_CIRC bit
   * - PeriphOrM2MSrcIncMode:  DMA_SxCR_PINC bit
   * - MemoryOrM2MDstIncMode:  DMA_SxCR_MINC bit
   * - PeriphOrM2MSrcDataSize: DMA_SxCR_PSIZE[1:0] bits
   * - MemoryOrM2MDstDataSize: DMA_SxCR_MSIZE[1:0] bits
   * - Priority:               DMA_SxCR_PL[1:0] bits
   */
  LL_DMA_ConfigTransfer(DMAx, Stream, DMA_InitStruct->Direction | \
                        DMA_InitStruct->Mode                    | \
                        DMA_InitStruct->PeriphOrM2MSrcIncMode   | \
                        DMA_InitStruct->MemoryOrM2MDstIncMode   | \
                        DMA_InitStruct->PeriphOrM2MSrcDataSize  | \
                        DMA_InitStruct->MemoryOrM2MDstDataSize  | \
                        DMA_InitStruct->Priority
                        );

  if(DMA_InitStruct->FIFOMode != LL_DMA_FIFOMODE_DISABLE)
  {
    /*---------------------------- DMAx SxFCR Configuration ------------------------
     * Configure DMAx_Streamy:  fifo mode and fifo threshold with parameters :
     * - FIFOMode:                DMA_SxFCR_DMDIS bit
     * - FIFOThreshold:           DMA_SxFCR_FTH[1:0] bits
     */
    LL_DMA_ConfigFifo(DMAx, Stream, DMA_InitStruct->FIFOMode, DMA_InitStruct->FIFOThreshold);   

    /*---------------------------- DMAx SxCR Configuration --------------------------
     * Configure DMAx_Streamy:  memory burst transfer with parameters :
     * - MemBurst:                DMA_SxCR_MBURST[1:0] bits
     */
    LL_DMA_SetMemoryBurstxfer(DMAx,Stream,DMA_InitStruct->MemBurst); 

    /*---------------------------- DMAx SxCR Configuration --------------------------
     * Configure DMAx_Streamy:  peripheral burst transfer with parameters :
     * - PeriphBurst:             DMA_SxCR_PBURST[1:0] bits
     */
    LL_DMA_SetPeriphBurstxfer(DMAx,Stream,DMA_InitStruct->PeriphBurst);
  }

  /*-------------------------- DMAx SxM0AR Configuration --------------------------
   * Configure the memory or destination base address with parameter :
   * - MemoryOrM2MDstAddress:     DMA_SxM0AR_M0A[31:0] bits
   */
  LL_DMA_SetMemoryAddress(DMAx, Stream, DMA_InitStruct->MemoryOrM2MDstAddress);

  /*-------------------------- DMAx SxPAR Configuration ---------------------------
   * Configure the peripheral or source base address with parameter :
   * - PeriphOrM2MSrcAddress:     DMA_SxPAR_PA[31:0] bits
   */
  LL_DMA_SetPeriphAddress(DMAx, Stream, DMA_InitStruct->PeriphOrM2MSrcAddress);

  /*--------------------------- DMAx SxNDTR Configuration -------------------------
   * Configure the peripheral base address with parameter :
   * - NbData:                    DMA_SxNDT[15:0] bits
   */
  LL_DMA_SetDataLength(DMAx, Stream, DMA_InitStruct->NbData);

  /*--------------------------- DMA SxCR_CHSEL Configuration ----------------------
   * Configure the peripheral base address with parameter :
   * - PeriphRequest:             DMA_SxCR_CHSEL[2:0] bits
   */
  LL_DMA_SetChannelSelection(DMAx, Stream, DMA_InitStruct->Channel);

  return SUCCESS;
}

/**
  * @brief  Set each @ref LL_DMA_InitTypeDef field to default value.
  * @param  DMA_InitStruct Pointer to a @ref LL_DMA_InitTypeDef structure.
  * @retval None
  */
void LL_DMA_StructInit(LL_DMA_InitTypeDef *DMA_InitStruct)
{
  /* Set DMA_InitStruct fields to default values */
  DMA_InitStruct->PeriphOrM2MSrcAddress  = 0x00000000U;
  DMA_InitStruct->MemoryOrM2MDstAddress  = 0x00000000U;
  DMA_InitStruct->Direction              = LL_DMA_DIRECTION_PERIPH_TO_MEMORY;
  DMA_InitStruct->Mode                   = LL_DMA_MODE_NORMAL;
  DMA_InitStruct->PeriphOrM2MSrcIncMode  = LL_DMA_PERIPH_NOINCREMENT;
  DMA_InitStruct->MemoryOrM2MDstIncMode  = LL_DMA_MEMORY_NOINCREMENT;
  DMA_InitStruct->PeriphOrM2MSrcDataSize = LL_DMA_PDATAALIGN_BYTE;
  DMA_InitStruct->MemoryOrM2MDstDataSize = LL_DMA_MDATAALIGN_BYTE;
  DMA_InitStruct->NbData                 = 0x00000000U;
  DMA_InitStruct->Channel                = LL_DMA_CHANNEL_0;
  DMA_InitStruct->Priority               = LL_DMA_PRIORITY_LOW;
  DMA_InitStruct->FIFOMode               = LL_DMA_FIFOMODE_DISABLE;
  DMA_InitStruct->FIFOThreshold          = LL_DMA_FIFOTHRESHOLD_1_4;
  DMA_InitStruct->MemBurst               = LL_DMA_MBURST_SINGLE;
  DMA_InitStruct->PeriphBurst            = LL_DMA_PBURST_SINGLE;
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

#endif /* DMA1 || DMA2 */

/**
  * @}
  */

#endif /* USE_FULL_LL_DRIVER */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
