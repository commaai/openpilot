/**
  ******************************************************************************
  * @file    stm32f4xx_hal_sdram.h
  * @author  MCD Application Team
  * @brief   Header file of SDRAM HAL module.
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
#ifndef __STM32F4xx_HAL_SDRAM_H
#define __STM32F4xx_HAL_SDRAM_H

#ifdef __cplusplus
 extern "C" {
#endif

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_ll_fmc.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup SDRAM
  * @{
  */ 

/* Exported typedef ----------------------------------------------------------*/   
/** @defgroup SDRAM_Exported_Types SDRAM Exported Types
  * @{
  */

/** 
  * @brief  HAL SDRAM State structure definition  
  */ 
typedef enum
{
  HAL_SDRAM_STATE_RESET             = 0x00U,  /*!< SDRAM not yet initialized or disabled */
  HAL_SDRAM_STATE_READY             = 0x01U,  /*!< SDRAM initialized and ready for use   */
  HAL_SDRAM_STATE_BUSY              = 0x02U,  /*!< SDRAM internal process is ongoing     */
  HAL_SDRAM_STATE_ERROR             = 0x03U,  /*!< SDRAM error state                     */
  HAL_SDRAM_STATE_WRITE_PROTECTED   = 0x04U,  /*!< SDRAM device write protected          */
  HAL_SDRAM_STATE_PRECHARGED        = 0x05U   /*!< SDRAM device precharged               */

} HAL_SDRAM_StateTypeDef;

/**
  * @brief  SDRAM handle Structure definition
  */
#if (USE_HAL_SDRAM_REGISTER_CALLBACKS == 1)
typedef struct __SDRAM_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_SDRAM_REGISTER_CALLBACKS  */
{
  FMC_SDRAM_TypeDef             *Instance;  /*!< Register base address                 */
  
  FMC_SDRAM_InitTypeDef         Init;       /*!< SDRAM device configuration parameters */
  
  __IO HAL_SDRAM_StateTypeDef   State;      /*!< SDRAM access state                    */
  
  HAL_LockTypeDef               Lock;       /*!< SDRAM locking object                  */ 

  DMA_HandleTypeDef             *hdma;      /*!< Pointer DMA handler                   */

#if (USE_HAL_SDRAM_REGISTER_CALLBACKS == 1)
  void  (* MspInitCallback)        ( struct __SDRAM_HandleTypeDef * hsdram);    /*!< SDRAM Msp Init callback              */
  void  (* MspDeInitCallback)      ( struct __SDRAM_HandleTypeDef * hsdram);    /*!< SDRAM Msp DeInit callback            */
  void  (* RefreshErrorCallback)   ( struct __SDRAM_HandleTypeDef * hsdram);    /*!< SDRAM Refresh Error callback         */
  void  (* DmaXferCpltCallback)    ( DMA_HandleTypeDef * hdma);                 /*!< SDRAM DMA Xfer Complete callback     */
  void  (* DmaXferErrorCallback)   ( DMA_HandleTypeDef * hdma);                 /*!< SDRAM DMA Xfer Error callback        */
#endif
} SDRAM_HandleTypeDef;

#if (USE_HAL_SDRAM_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL SDRAM Callback ID enumeration definition
  */
typedef enum
{
  HAL_SDRAM_MSP_INIT_CB_ID       = 0x00U,  /*!< SDRAM MspInit Callback ID           */
  HAL_SDRAM_MSP_DEINIT_CB_ID     = 0x01U,  /*!< SDRAM MspDeInit Callback ID         */
  HAL_SDRAM_REFRESH_ERR_CB_ID    = 0x02U,  /*!< SDRAM Refresh Error Callback ID     */
  HAL_SDRAM_DMA_XFER_CPLT_CB_ID  = 0x03U,  /*!< SDRAM DMA Xfer Complete Callback ID */
  HAL_SDRAM_DMA_XFER_ERR_CB_ID   = 0x04U   /*!< SDRAM DMA Xfer Error Callback ID    */
}HAL_SDRAM_CallbackIDTypeDef;

/**
  * @brief  HAL SDRAM Callback pointer definition
  */
typedef void (*pSDRAM_CallbackTypeDef)(SDRAM_HandleTypeDef *hsdram);
typedef void (*pSDRAM_DmaCallbackTypeDef)(DMA_HandleTypeDef *hdma);
#endif
/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/* Exported macro ------------------------------------------------------------*/
/** @defgroup SDRAM_Exported_Macros SDRAM Exported Macros
  * @{
  */

/** @brief Reset SDRAM handle state
  * @param  __HANDLE__ specifies the SDRAM handle.
  * @retval None
  */
#if (USE_HAL_SDRAM_REGISTER_CALLBACKS == 1)
#define __HAL_SDRAM_RESET_HANDLE_STATE(__HANDLE__)        do {                                               \
                                                               (__HANDLE__)->State = HAL_SDRAM_STATE_RESET;  \
                                                               (__HANDLE__)->MspInitCallback = NULL;         \
                                                               (__HANDLE__)->MspDeInitCallback = NULL;       \
                                                             } while(0)
#else
#define __HAL_SDRAM_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_SDRAM_STATE_RESET)
#endif
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup SDRAM_Exported_Functions SDRAM Exported Functions
  * @{
  */

/** @addtogroup SDRAM_Exported_Functions_Group1 
  * @{
  */

/* Initialization/de-initialization functions *********************************/
HAL_StatusTypeDef HAL_SDRAM_Init(SDRAM_HandleTypeDef *hsdram, FMC_SDRAM_TimingTypeDef *Timing);
HAL_StatusTypeDef HAL_SDRAM_DeInit(SDRAM_HandleTypeDef *hsdram);
void HAL_SDRAM_MspInit(SDRAM_HandleTypeDef *hsdram);
void HAL_SDRAM_MspDeInit(SDRAM_HandleTypeDef *hsdram);

void HAL_SDRAM_IRQHandler(SDRAM_HandleTypeDef *hsdram);
void HAL_SDRAM_RefreshErrorCallback(SDRAM_HandleTypeDef *hsdram);
void HAL_SDRAM_DMA_XferCpltCallback(DMA_HandleTypeDef *hdma);
void HAL_SDRAM_DMA_XferErrorCallback(DMA_HandleTypeDef *hdma);
/**
  * @}
  */

/** @addtogroup SDRAM_Exported_Functions_Group2 
  * @{
  */
/* I/O operation functions ****************************************************/
HAL_StatusTypeDef HAL_SDRAM_Read_8b(SDRAM_HandleTypeDef *hsdram, uint32_t *pAddress, uint8_t *pDstBuffer, uint32_t BufferSize);
HAL_StatusTypeDef HAL_SDRAM_Write_8b(SDRAM_HandleTypeDef *hsdram, uint32_t *pAddress, uint8_t *pSrcBuffer, uint32_t BufferSize);
HAL_StatusTypeDef HAL_SDRAM_Read_16b(SDRAM_HandleTypeDef *hsdram, uint32_t *pAddress, uint16_t *pDstBuffer, uint32_t BufferSize);
HAL_StatusTypeDef HAL_SDRAM_Write_16b(SDRAM_HandleTypeDef *hsdram, uint32_t *pAddress, uint16_t *pSrcBuffer, uint32_t BufferSize);
HAL_StatusTypeDef HAL_SDRAM_Read_32b(SDRAM_HandleTypeDef *hsdram, uint32_t *pAddress, uint32_t *pDstBuffer, uint32_t BufferSize);
HAL_StatusTypeDef HAL_SDRAM_Write_32b(SDRAM_HandleTypeDef *hsdram, uint32_t *pAddress, uint32_t *pSrcBuffer, uint32_t BufferSize);

HAL_StatusTypeDef HAL_SDRAM_Read_DMA(SDRAM_HandleTypeDef *hsdram, uint32_t * pAddress, uint32_t *pDstBuffer, uint32_t BufferSize);
HAL_StatusTypeDef HAL_SDRAM_Write_DMA(SDRAM_HandleTypeDef *hsdram, uint32_t *pAddress, uint32_t *pSrcBuffer, uint32_t BufferSize);

#if (USE_HAL_SDRAM_REGISTER_CALLBACKS == 1)
/* SDRAM callback registering/unregistering */
HAL_StatusTypeDef HAL_SDRAM_RegisterCallback(SDRAM_HandleTypeDef *hsdram, HAL_SDRAM_CallbackIDTypeDef CallbackId, pSDRAM_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_SDRAM_UnRegisterCallback(SDRAM_HandleTypeDef *hsdram, HAL_SDRAM_CallbackIDTypeDef CallbackId);
HAL_StatusTypeDef HAL_SDRAM_RegisterDmaCallback(SDRAM_HandleTypeDef *hsdram, HAL_SDRAM_CallbackIDTypeDef CallbackId, pSDRAM_DmaCallbackTypeDef pCallback);
#endif

/**
  * @}
  */
  
/** @addtogroup SDRAM_Exported_Functions_Group3 
  * @{
  */
/* SDRAM Control functions  *****************************************************/
HAL_StatusTypeDef HAL_SDRAM_WriteProtection_Enable(SDRAM_HandleTypeDef *hsdram);
HAL_StatusTypeDef HAL_SDRAM_WriteProtection_Disable(SDRAM_HandleTypeDef *hsdram);
HAL_StatusTypeDef HAL_SDRAM_SendCommand(SDRAM_HandleTypeDef *hsdram, FMC_SDRAM_CommandTypeDef *Command, uint32_t Timeout);
HAL_StatusTypeDef HAL_SDRAM_ProgramRefreshRate(SDRAM_HandleTypeDef *hsdram, uint32_t RefreshRate);
HAL_StatusTypeDef HAL_SDRAM_SetAutoRefreshNumber(SDRAM_HandleTypeDef *hsdram, uint32_t AutoRefreshNumber);
uint32_t          HAL_SDRAM_GetModeStatus(SDRAM_HandleTypeDef *hsdram);
/**
  * @}
  */

/** @addtogroup SDRAM_Exported_Functions_Group4 
  * @{
  */
/* SDRAM State functions ********************************************************/
HAL_SDRAM_StateTypeDef  HAL_SDRAM_GetState(SDRAM_HandleTypeDef *hsdram);
/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_SDRAM_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
