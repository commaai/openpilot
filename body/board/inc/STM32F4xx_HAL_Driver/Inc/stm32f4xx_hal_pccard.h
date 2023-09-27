/**
  ******************************************************************************
  * @file    stm32f4xx_hal_pccard.h
  * @author  MCD Application Team
  * @brief   Header file of PCCARD HAL module.
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
#ifndef __STM32F4xx_HAL_PCCARD_H
#define __STM32F4xx_HAL_PCCARD_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx)
  #include "stm32f4xx_ll_fsmc.h"
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx)
  #include "stm32f4xx_ll_fmc.h"
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx */

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx)

/** @addtogroup PCCARD
  * @{
  */ 

/* Exported typedef ----------------------------------------------------------*/
/** @defgroup PCCARD_Exported_Types PCCARD Exported Types
  * @{
  */

/** 
  * @brief  HAL PCCARD State structures definition  
  */ 
typedef enum
{
  HAL_PCCARD_STATE_RESET     = 0x00U,    /*!< PCCARD peripheral not yet initialized or disabled */
  HAL_PCCARD_STATE_READY     = 0x01U,    /*!< PCCARD peripheral ready                           */
  HAL_PCCARD_STATE_BUSY      = 0x02U,    /*!< PCCARD peripheral busy                            */
  HAL_PCCARD_STATE_ERROR     = 0x04U     /*!< PCCARD peripheral error                           */
}HAL_PCCARD_StateTypeDef;

typedef enum
{
  HAL_PCCARD_STATUS_SUCCESS = 0U,
  HAL_PCCARD_STATUS_ONGOING,
  HAL_PCCARD_STATUS_ERROR,
  HAL_PCCARD_STATUS_TIMEOUT
}HAL_PCCARD_StatusTypeDef;

/**
  * @brief  FMC_PCCARD handle Structure definition
  */
#if (USE_HAL_PCCARD_REGISTER_CALLBACKS == 1)  
typedef struct __PCCARD_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_PCCARD_REGISTER_CALLBACKS  */
{
  FMC_PCCARD_TypeDef           *Instance;              /*!< Register base address for PCCARD device          */
  
  FMC_PCCARD_InitTypeDef       Init;                   /*!< PCCARD device control configuration parameters   */

  __IO HAL_PCCARD_StateTypeDef State;                  /*!< PCCARD device access state                       */

  HAL_LockTypeDef              Lock;                   /*!< PCCARD Lock                                      */

#if (USE_HAL_PCCARD_REGISTER_CALLBACKS == 1)
  void  (* MspInitCallback)        ( struct __PCCARD_HandleTypeDef * hpccard);    /*!< PCCARD Msp Init callback              */
  void  (* MspDeInitCallback)      ( struct __PCCARD_HandleTypeDef * hpccard);    /*!< PCCARD Msp DeInit callback            */
  void  (* ItCallback)             ( struct __PCCARD_HandleTypeDef * hpccard);    /*!< PCCARD IT callback                    */
#endif
} PCCARD_HandleTypeDef;

#if (USE_HAL_PCCARD_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL PCCARD Callback ID enumeration definition
  */
typedef enum
{
  HAL_PCCARD_MSP_INIT_CB_ID       = 0x00U,  /*!< PCCARD MspInit Callback ID          */
  HAL_PCCARD_MSP_DEINIT_CB_ID     = 0x01U,  /*!< PCCARD MspDeInit Callback ID        */
  HAL_PCCARD_IT_CB_ID             = 0x02U   /*!< PCCARD IT Callback ID               */
}HAL_PCCARD_CallbackIDTypeDef;

/**
  * @brief  HAL PCCARD Callback pointer definition
  */
typedef void (*pPCCARD_CallbackTypeDef)(PCCARD_HandleTypeDef *hpccard);
#endif
/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/* Exported macro ------------------------------------------------------------*/
/** @defgroup PCCARD_Exported_Macros PCCARD Exported Macros
  * @{
  */
/** @brief Reset PCCARD handle state
  * @param  __HANDLE__ specifies the PCCARD handle.
  * @retval None
  */
#if (USE_HAL_PCCARD_REGISTER_CALLBACKS == 1)
#define __HAL_PCCARD_RESET_HANDLE_STATE(__HANDLE__)       do {                                               \
                                                               (__HANDLE__)->State = HAL_PCCARD_STATE_RESET; \
                                                               (__HANDLE__)->MspInitCallback = NULL;         \
                                                               (__HANDLE__)->MspDeInitCallback = NULL;       \
                                                             } while(0)
#else
#define __HAL_PCCARD_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_PCCARD_STATE_RESET)
#endif
/**
  * @}
  */ 

/* Exported functions --------------------------------------------------------*/
/** @addtogroup PCCARD_Exported_Functions 
  * @{
  */

/** @addtogroup PCCARD_Exported_Functions_Group1 
  * @{
  */
/* Initialization/de-initialization functions  **********************************/
HAL_StatusTypeDef  HAL_PCCARD_Init(PCCARD_HandleTypeDef *hpccard, FMC_NAND_PCC_TimingTypeDef *ComSpaceTiming, FMC_NAND_PCC_TimingTypeDef *AttSpaceTiming, FMC_NAND_PCC_TimingTypeDef *IOSpaceTiming);
HAL_StatusTypeDef  HAL_PCCARD_DeInit(PCCARD_HandleTypeDef *hpccard);
void HAL_PCCARD_MspInit(PCCARD_HandleTypeDef *hpccard);
void HAL_PCCARD_MspDeInit(PCCARD_HandleTypeDef *hpccard);
/**
  * @}
  */

/** @addtogroup PCCARD_Exported_Functions_Group2
  * @{
  */
/* IO operation functions  *****************************************************/
HAL_StatusTypeDef  HAL_PCCARD_Read_ID(PCCARD_HandleTypeDef *hpccard, uint8_t CompactFlash_ID[], uint8_t *pStatus);
HAL_StatusTypeDef  HAL_PCCARD_Write_Sector(PCCARD_HandleTypeDef *hpccard, uint16_t *pBuffer, uint16_t SectorAddress,  uint8_t *pStatus);
HAL_StatusTypeDef  HAL_PCCARD_Read_Sector(PCCARD_HandleTypeDef *hpccard, uint16_t *pBuffer, uint16_t SectorAddress, uint8_t *pStatus);
HAL_StatusTypeDef  HAL_PCCARD_Erase_Sector(PCCARD_HandleTypeDef *hpccard, uint16_t SectorAddress, uint8_t *pStatus);
HAL_StatusTypeDef  HAL_PCCARD_Reset(PCCARD_HandleTypeDef *hpccard);
void               HAL_PCCARD_IRQHandler(PCCARD_HandleTypeDef *hpccard);
void               HAL_PCCARD_ITCallback(PCCARD_HandleTypeDef *hpccard);

#if (USE_HAL_PCCARD_REGISTER_CALLBACKS == 1)
/* PCCARD callback registering/unregistering */
HAL_StatusTypeDef  HAL_PCCARD_RegisterCallback(PCCARD_HandleTypeDef *hpccard, HAL_PCCARD_CallbackIDTypeDef CallbackId, pPCCARD_CallbackTypeDef pCallback);
HAL_StatusTypeDef  HAL_PCCARD_UnRegisterCallback(PCCARD_HandleTypeDef *hpccard, HAL_PCCARD_CallbackIDTypeDef CallbackId);
#endif
/**
  * @}
  */

/** @addtogroup PCCARD_Exported_Functions_Group3
  * @{
  */
/* PCCARD State functions *******************************************************/
HAL_PCCARD_StateTypeDef  HAL_PCCARD_GetState(PCCARD_HandleTypeDef *hpccard);
HAL_PCCARD_StatusTypeDef HAL_PCCARD_GetStatus(PCCARD_HandleTypeDef *hpccard);
HAL_PCCARD_StatusTypeDef HAL_PCCARD_ReadStatus(PCCARD_HandleTypeDef *hpccard);
/**
  * @}
  */

/**
  * @}
  */
/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup PCCARD_Private_Constants PCCARD Private Constants
  * @{
  */
#define PCCARD_DEVICE_ADDRESS             0x90000000U
#define PCCARD_ATTRIBUTE_SPACE_ADDRESS    0x98000000U              /* Attribute space size to @0x9BFF FFFF */
#define PCCARD_COMMON_SPACE_ADDRESS       PCCARD_DEVICE_ADDRESS    /* Common space size to @0x93FF FFFF    */
#define PCCARD_IO_SPACE_ADDRESS           0x9C000000U              /* IO space size to @0x9FFF FFFF        */
#define PCCARD_IO_SPACE_PRIMARY_ADDR      0x9C0001F0U              /* IO space size to @0x9FFF FFFF        */

/* Flash-ATA registers description */
#define ATA_DATA                       ((uint8_t)0x00)    /* Data register */
#define ATA_SECTOR_COUNT               ((uint8_t)0x02)    /* Sector Count register */
#define ATA_SECTOR_NUMBER              ((uint8_t)0x03)    /* Sector Number register */
#define ATA_CYLINDER_LOW               ((uint8_t)0x04)    /* Cylinder low register */
#define ATA_CYLINDER_HIGH              ((uint8_t)0x05)    /* Cylinder high register */
#define ATA_CARD_HEAD                  ((uint8_t)0x06)    /* Card/Head register */
#define ATA_STATUS_CMD                 ((uint8_t)0x07)    /* Status(read)/Command(write) register */
#define ATA_STATUS_CMD_ALTERNATE       ((uint8_t)0x0E)    /* Alternate Status(read)/Command(write) register */
#define ATA_COMMON_DATA_AREA           ((uint16_t)0x0400) /* Start of data area (for Common access only!) */
#define ATA_CARD_CONFIGURATION         ((uint16_t)0x0202) /* Card Configuration and Status Register */

/* Flash-ATA commands */
#define ATA_READ_SECTOR_CMD            ((uint8_t)0x20)
#define ATA_WRITE_SECTOR_CMD           ((uint8_t)0x30)
#define ATA_ERASE_SECTOR_CMD           ((uint8_t)0xC0)
#define ATA_IDENTIFY_CMD               ((uint8_t)0xEC)

/* PC Card/Compact Flash status */
#define PCCARD_TIMEOUT_ERROR           ((uint8_t)0x60)
#define PCCARD_BUSY                    ((uint8_t)0x80)
#define PCCARD_PROGR                   ((uint8_t)0x01)
#define PCCARD_READY                   ((uint8_t)0x40)

#define PCCARD_SECTOR_SIZE             255U               /* In half words */ 

/**
  * @}
  */
/* Compact Flash redefinition */
#define HAL_CF_Init                 HAL_PCCARD_Init
#define HAL_CF_DeInit               HAL_PCCARD_DeInit
#define HAL_CF_MspInit              HAL_PCCARD_MspInit
#define HAL_CF_MspDeInit            HAL_PCCARD_MspDeInit

#define HAL_CF_Read_ID              HAL_PCCARD_Read_ID
#define HAL_CF_Write_Sector         HAL_PCCARD_Write_Sector
#define HAL_CF_Read_Sector          HAL_PCCARD_Read_Sector
#define HAL_CF_Erase_Sector         HAL_PCCARD_Erase_Sector
#define HAL_CF_Reset                HAL_PCCARD_Reset
#define HAL_CF_IRQHandler           HAL_PCCARD_IRQHandler
#define HAL_CF_ITCallback           HAL_PCCARD_ITCallback

#define HAL_CF_GetState             HAL_PCCARD_GetState
#define HAL_CF_GetStatus            HAL_PCCARD_GetStatus
#define HAL_CF_ReadStatus           HAL_PCCARD_ReadStatus

#define HAL_CF_STATUS_SUCCESS       HAL_PCCARD_STATUS_SUCCESS
#define HAL_CF_STATUS_ONGOING       HAL_PCCARD_STATUS_ONGOING
#define HAL_CF_STATUS_ERROR         HAL_PCCARD_STATUS_ERROR
#define HAL_CF_STATUS_TIMEOUT       HAL_PCCARD_STATUS_TIMEOUT
#define HAL_CF_StatusTypeDef        HAL_PCCARD_StatusTypeDef

#define CF_DEVICE_ADDRESS           PCCARD_DEVICE_ADDRESS
#define CF_ATTRIBUTE_SPACE_ADDRESS  PCCARD_ATTRIBUTE_SPACE_ADDRESS
#define CF_COMMON_SPACE_ADDRESS     PCCARD_COMMON_SPACE_ADDRESS   
#define CF_IO_SPACE_ADDRESS         PCCARD_IO_SPACE_ADDRESS       
#define CF_IO_SPACE_PRIMARY_ADDR    PCCARD_IO_SPACE_PRIMARY_ADDR  

#define CF_TIMEOUT_ERROR            PCCARD_TIMEOUT_ERROR
#define CF_BUSY                     PCCARD_BUSY         
#define CF_PROGR                    PCCARD_PROGR        
#define CF_READY                    PCCARD_READY        

#define CF_SECTOR_SIZE              PCCARD_SECTOR_SIZE

/* Private macros ------------------------------------------------------------*/
/**
  * @}
  */

#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx ||\
          STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx */


/**
  * @}
  */
  
#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_PCCARD_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
