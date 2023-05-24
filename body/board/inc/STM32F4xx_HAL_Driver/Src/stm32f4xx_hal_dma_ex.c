/**
  ******************************************************************************
  * @file    stm32f4xx_hal_dma_ex.c
  * @author  MCD Application Team
  * @brief   DMA Extension HAL module driver
  *         This file provides firmware functions to manage the following 
  *         functionalities of the DMA Extension peripheral:
  *           + Extended features functions
  *
  @verbatim
  ==============================================================================
                        ##### How to use this driver #####
  ==============================================================================
  [..]
  The DMA Extension HAL driver can be used as follows:
   (#) Start a multi buffer transfer using the HAL_DMA_MultiBufferStart() function
       for polling mode or HAL_DMA_MultiBufferStart_IT() for interrupt mode.
                   
     -@-  In Memory-to-Memory transfer mode, Multi (Double) Buffer mode is not allowed.
     -@-  When Multi (Double) Buffer mode is enabled the, transfer is circular by default.
     -@-  In Multi (Double) buffer mode, it is possible to update the base address for 
          the AHB memory port on the fly (DMA_SxM0AR or DMA_SxM1AR) when the stream is enabled. 
  
  @endverbatim
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

/** @defgroup DMAEx DMAEx
  * @brief DMA Extended HAL module driver
  * @{
  */

#ifdef HAL_DMA_MODULE_ENABLED

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private Constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/** @addtogroup DMAEx_Private_Functions
  * @{
  */
static void DMA_MultiBufferSetConfig(DMA_HandleTypeDef *hdma, uint32_t SrcAddress, uint32_t DstAddress, uint32_t DataLength);
/**
  * @}
  */

/* Exported functions ---------------------------------------------------------*/

/** @addtogroup DMAEx_Exported_Functions
  * @{
  */


/** @addtogroup DMAEx_Exported_Functions_Group1
  *
@verbatim   
 ===============================================================================
                #####  Extended features functions  #####
 ===============================================================================  
    [..]  This section provides functions allowing to:
      (+) Configure the source, destination address and data length and 
          Start MultiBuffer DMA transfer
      (+) Configure the source, destination address and data length and 
          Start MultiBuffer DMA transfer with interrupt
      (+) Change on the fly the memory0 or memory1 address.
      
@endverbatim
  * @{
  */


/**
  * @brief  Starts the multi_buffer DMA Transfer.
  * @param  hdma       pointer to a DMA_HandleTypeDef structure that contains
  *                     the configuration information for the specified DMA Stream.  
  * @param  SrcAddress The source memory Buffer address
  * @param  DstAddress The destination memory Buffer address
  * @param  SecondMemAddress The second memory Buffer address in case of multi buffer Transfer  
  * @param  DataLength The length of data to be transferred from source to destination
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_DMAEx_MultiBufferStart(DMA_HandleTypeDef *hdma, uint32_t SrcAddress, uint32_t DstAddress, uint32_t SecondMemAddress, uint32_t DataLength)
{
  HAL_StatusTypeDef status = HAL_OK;
  
  /* Check the parameters */
  assert_param(IS_DMA_BUFFER_SIZE(DataLength));
  
  /* Memory-to-memory transfer not supported in double buffering mode */
  if (hdma->Init.Direction == DMA_MEMORY_TO_MEMORY)
  {
    hdma->ErrorCode = HAL_DMA_ERROR_NOT_SUPPORTED;
    status = HAL_ERROR;
  }
  else
  {
    /* Process Locked */
    __HAL_LOCK(hdma);
    
    if(HAL_DMA_STATE_READY == hdma->State)
    {
      /* Change DMA peripheral state */
      hdma->State = HAL_DMA_STATE_BUSY; 
      
      /* Enable the double buffer mode */
      hdma->Instance->CR |= (uint32_t)DMA_SxCR_DBM;
      
      /* Configure DMA Stream destination address */
      hdma->Instance->M1AR = SecondMemAddress;
      
      /* Configure the source, destination address and the data length */
      DMA_MultiBufferSetConfig(hdma, SrcAddress, DstAddress, DataLength);
      
      /* Enable the peripheral */
      __HAL_DMA_ENABLE(hdma);
    }
    else
    {
      /* Return error status */
      status = HAL_BUSY;
    }
  }
  return status;
}

/**
  * @brief  Starts the multi_buffer DMA Transfer with interrupt enabled.
  * @param  hdma       pointer to a DMA_HandleTypeDef structure that contains
  *                     the configuration information for the specified DMA Stream.  
  * @param  SrcAddress The source memory Buffer address
  * @param  DstAddress The destination memory Buffer address
  * @param  SecondMemAddress The second memory Buffer address in case of multi buffer Transfer  
  * @param  DataLength The length of data to be transferred from source to destination
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_DMAEx_MultiBufferStart_IT(DMA_HandleTypeDef *hdma, uint32_t SrcAddress, uint32_t DstAddress, uint32_t SecondMemAddress, uint32_t DataLength)
{
  HAL_StatusTypeDef status = HAL_OK;
  
  /* Check the parameters */
  assert_param(IS_DMA_BUFFER_SIZE(DataLength));
  
  /* Memory-to-memory transfer not supported in double buffering mode */
  if (hdma->Init.Direction == DMA_MEMORY_TO_MEMORY)
  {
    hdma->ErrorCode = HAL_DMA_ERROR_NOT_SUPPORTED;
    return HAL_ERROR;
  }
  
  /* Check callback functions */
  if ((NULL == hdma->XferCpltCallback) || (NULL == hdma->XferM1CpltCallback) || (NULL == hdma->XferErrorCallback))
  {
    hdma->ErrorCode = HAL_DMA_ERROR_PARAM;
    return HAL_ERROR;
  }
  
  /* Process locked */
  __HAL_LOCK(hdma);
  
  if(HAL_DMA_STATE_READY == hdma->State)
  {
    /* Change DMA peripheral state */
    hdma->State = HAL_DMA_STATE_BUSY;
    
    /* Initialize the error code */
    hdma->ErrorCode = HAL_DMA_ERROR_NONE;
    
    /* Enable the Double buffer mode */
    hdma->Instance->CR |= (uint32_t)DMA_SxCR_DBM;
    
    /* Configure DMA Stream destination address */
    hdma->Instance->M1AR = SecondMemAddress;
    
    /* Configure the source, destination address and the data length */
    DMA_MultiBufferSetConfig(hdma, SrcAddress, DstAddress, DataLength); 
    
    /* Clear all flags */
    __HAL_DMA_CLEAR_FLAG (hdma, __HAL_DMA_GET_TC_FLAG_INDEX(hdma));
    __HAL_DMA_CLEAR_FLAG (hdma, __HAL_DMA_GET_HT_FLAG_INDEX(hdma));
    __HAL_DMA_CLEAR_FLAG (hdma, __HAL_DMA_GET_TE_FLAG_INDEX(hdma));
    __HAL_DMA_CLEAR_FLAG (hdma, __HAL_DMA_GET_DME_FLAG_INDEX(hdma));
    __HAL_DMA_CLEAR_FLAG (hdma, __HAL_DMA_GET_FE_FLAG_INDEX(hdma));

    /* Enable Common interrupts*/
    hdma->Instance->CR  |= DMA_IT_TC | DMA_IT_TE | DMA_IT_DME;
    hdma->Instance->FCR |= DMA_IT_FE;
    
    if((hdma->XferHalfCpltCallback != NULL) || (hdma->XferM1HalfCpltCallback != NULL))
    {
      hdma->Instance->CR  |= DMA_IT_HT;
    }
    
    /* Enable the peripheral */
    __HAL_DMA_ENABLE(hdma); 
  }
  else
  {     
    /* Process unlocked */
    __HAL_UNLOCK(hdma);	  
    
    /* Return error status */
    status = HAL_BUSY;
  }  
  return status; 
}

/**
  * @brief  Change the memory0 or memory1 address on the fly.
  * @param  hdma       pointer to a DMA_HandleTypeDef structure that contains
  *                     the configuration information for the specified DMA Stream.  
  * @param  Address    The new address
  * @param  memory     the memory to be changed, This parameter can be one of 
  *                     the following values:
  *                      MEMORY0 /
  *                      MEMORY1
  * @note   The MEMORY0 address can be changed only when the current transfer use
  *         MEMORY1 and the MEMORY1 address can be changed only when the current 
  *         transfer use MEMORY0.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_DMAEx_ChangeMemory(DMA_HandleTypeDef *hdma, uint32_t Address, HAL_DMA_MemoryTypeDef memory)
{
  if(memory == MEMORY0)
  {
    /* change the memory0 address */
    hdma->Instance->M0AR = Address;
  }
  else
  {
    /* change the memory1 address */
    hdma->Instance->M1AR = Address;
  }

  return HAL_OK;
}

/**
  * @}
  */

/**
  * @}
  */

/** @addtogroup DMAEx_Private_Functions
  * @{
  */

/**
  * @brief  Set the DMA Transfer parameter.
  * @param  hdma       pointer to a DMA_HandleTypeDef structure that contains
  *                     the configuration information for the specified DMA Stream.  
  * @param  SrcAddress The source memory Buffer address
  * @param  DstAddress The destination memory Buffer address
  * @param  DataLength The length of data to be transferred from source to destination
  * @retval HAL status
  */
static void DMA_MultiBufferSetConfig(DMA_HandleTypeDef *hdma, uint32_t SrcAddress, uint32_t DstAddress, uint32_t DataLength)
{  
  /* Configure DMA Stream data length */
  hdma->Instance->NDTR = DataLength;
  
  /* Peripheral to Memory */
  if((hdma->Init.Direction) == DMA_MEMORY_TO_PERIPH)
  {   
    /* Configure DMA Stream destination address */
    hdma->Instance->PAR = DstAddress;
    
    /* Configure DMA Stream source address */
    hdma->Instance->M0AR = SrcAddress;
  }
  /* Memory to Peripheral */
  else
  {
    /* Configure DMA Stream source address */
    hdma->Instance->PAR = SrcAddress;
    
    /* Configure DMA Stream destination address */
    hdma->Instance->M0AR = DstAddress;
  }
}

/**
  * @}
  */

#endif /* HAL_DMA_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
