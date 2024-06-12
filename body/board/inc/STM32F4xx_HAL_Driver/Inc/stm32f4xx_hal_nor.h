/**
  ******************************************************************************
  * @file    stm32f4xx_hal_nor.h
  * @author  MCD Application Team
  * @brief   Header file of NOR HAL module.
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
#ifndef __STM32F4xx_HAL_NOR_H
#define __STM32F4xx_HAL_NOR_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F413xx) || defined(STM32F423xx)
  #include "stm32f4xx_ll_fsmc.h"
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F413xx || STM32F423xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
  #include "stm32f4xx_ll_fmc.h"
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup NOR
  * @{
  */ 

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F413xx) || defined(STM32F423xx)

/* Exported typedef ----------------------------------------------------------*/
/** @defgroup NOR_Exported_Types NOR Exported Types
  * @{
  */

/** 
  * @brief  HAL SRAM State structures definition  
  */ 
typedef enum
{  
  HAL_NOR_STATE_RESET             = 0x00U,  /*!< NOR not yet initialized or disabled  */
  HAL_NOR_STATE_READY             = 0x01U,  /*!< NOR initialized and ready for use    */
  HAL_NOR_STATE_BUSY              = 0x02U,  /*!< NOR internal processing is ongoing   */
  HAL_NOR_STATE_ERROR             = 0x03U,  /*!< NOR error state                      */
  HAL_NOR_STATE_PROTECTED         = 0x04U   /*!< NOR NORSRAM device write protected   */
}HAL_NOR_StateTypeDef;

/**
  * @brief  FMC NOR Status typedef
  */
typedef enum
{
  HAL_NOR_STATUS_SUCCESS  = 0U,
  HAL_NOR_STATUS_ONGOING,
  HAL_NOR_STATUS_ERROR,
  HAL_NOR_STATUS_TIMEOUT
}HAL_NOR_StatusTypeDef;

/**
  * @brief  FMC NOR ID typedef
  */
typedef struct
{
  uint16_t Manufacturer_Code;  /*!< Defines the device's manufacturer code used to identify the memory       */

  uint16_t Device_Code1;

  uint16_t Device_Code2;

  uint16_t Device_Code3;       /*!< Defines the device's codes used to identify the memory. 
                                    These codes can be accessed by performing read operations with specific 
                                    control signals and addresses set.They can also be accessed by issuing 
                                    an Auto Select command                                                   */
}NOR_IDTypeDef;

/**
  * @brief  FMC NOR CFI typedef
  */
typedef struct
{
  /*!< Defines the information stored in the memory's Common flash interface
       which contains a description of various electrical and timing parameters, 
       density information and functions supported by the memory                   */

  uint16_t CFI_1;

  uint16_t CFI_2;

  uint16_t CFI_3;

  uint16_t CFI_4;
}NOR_CFITypeDef;

/** 
  * @brief  NOR handle Structure definition
  */
#if (USE_HAL_NOR_REGISTER_CALLBACKS == 1)
typedef struct __NOR_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_NOR_REGISTER_CALLBACKS  */

{
  FMC_NORSRAM_TypeDef           *Instance;    /*!< Register base address                        */

  FMC_NORSRAM_EXTENDED_TypeDef  *Extended;    /*!< Extended mode register base address          */

  FMC_NORSRAM_InitTypeDef       Init;         /*!< NOR device control configuration parameters  */

  HAL_LockTypeDef               Lock;         /*!< NOR locking object                           */

  __IO HAL_NOR_StateTypeDef     State;        /*!< NOR device access state                      */

#if (USE_HAL_NOR_REGISTER_CALLBACKS == 1)
  void  (* MspInitCallback)        ( struct __NOR_HandleTypeDef * hnor);    /*!< NOR Msp Init callback              */
  void  (* MspDeInitCallback)      ( struct __NOR_HandleTypeDef * hnor);    /*!< NOR Msp DeInit callback            */
#endif
} NOR_HandleTypeDef;

#if (USE_HAL_NOR_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL NOR Callback ID enumeration definition
  */
typedef enum
{
  HAL_NOR_MSP_INIT_CB_ID       = 0x00U,  /*!< NOR MspInit Callback ID          */
  HAL_NOR_MSP_DEINIT_CB_ID     = 0x01U   /*!< NOR MspDeInit Callback ID        */
}HAL_NOR_CallbackIDTypeDef;

/**
  * @brief  HAL NOR Callback pointer definition
  */
typedef void (*pNOR_CallbackTypeDef)(NOR_HandleTypeDef *hnor);
#endif
/**
  * @}
  */
  
/* Exported constants --------------------------------------------------------*/
/* Exported macros ------------------------------------------------------------*/
/** @defgroup NOR_Exported_Macros NOR Exported Macros
  * @{
  */
/** @brief Reset NOR handle state
  * @param  __HANDLE__ specifies the NOR handle.
  * @retval None
  */
#if (USE_HAL_NOR_REGISTER_CALLBACKS == 1)
#define __HAL_NOR_RESET_HANDLE_STATE(__HANDLE__)          do {                                             \
                                                               (__HANDLE__)->State = HAL_NOR_STATE_RESET;  \
                                                               (__HANDLE__)->MspInitCallback = NULL;       \
                                                               (__HANDLE__)->MspDeInitCallback = NULL;     \
                                                             } while(0)
#else
#define __HAL_NOR_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_NOR_STATE_RESET)
#endif
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup NOR_Exported_Functions 
 *  @{
 */

/** @addtogroup NOR_Exported_Functions_Group1
 *  @{
 */
/* Initialization/de-initialization functions  ********************************/
HAL_StatusTypeDef HAL_NOR_Init(NOR_HandleTypeDef *hnor, FMC_NORSRAM_TimingTypeDef *Timing, FMC_NORSRAM_TimingTypeDef *ExtTiming);
HAL_StatusTypeDef HAL_NOR_DeInit(NOR_HandleTypeDef *hnor);
void HAL_NOR_MspInit(NOR_HandleTypeDef *hnor);
void HAL_NOR_MspDeInit(NOR_HandleTypeDef *hnor);
void HAL_NOR_MspWait(NOR_HandleTypeDef *hnor, uint32_t Timeout);
/**
  * @}
  */
  
/** @addtogroup NOR_Exported_Functions_Group2
 *  @{
 */
/* I/O operation functions  ***************************************************/
HAL_StatusTypeDef HAL_NOR_Read_ID(NOR_HandleTypeDef *hnor, NOR_IDTypeDef *pNOR_ID);
HAL_StatusTypeDef HAL_NOR_ReturnToReadMode(NOR_HandleTypeDef *hnor);
HAL_StatusTypeDef HAL_NOR_Read(NOR_HandleTypeDef *hnor, uint32_t *pAddress, uint16_t *pData);
HAL_StatusTypeDef HAL_NOR_Program(NOR_HandleTypeDef *hnor, uint32_t *pAddress, uint16_t *pData);

HAL_StatusTypeDef HAL_NOR_ReadBuffer(NOR_HandleTypeDef *hnor, uint32_t uwAddress, uint16_t *pData, uint32_t uwBufferSize);
HAL_StatusTypeDef HAL_NOR_ProgramBuffer(NOR_HandleTypeDef *hnor, uint32_t uwAddress, uint16_t *pData, uint32_t uwBufferSize);

HAL_StatusTypeDef HAL_NOR_Erase_Block(NOR_HandleTypeDef *hnor, uint32_t BlockAddress, uint32_t Address);
HAL_StatusTypeDef HAL_NOR_Erase_Chip(NOR_HandleTypeDef *hnor, uint32_t Address);
HAL_StatusTypeDef HAL_NOR_Read_CFI(NOR_HandleTypeDef *hnor, NOR_CFITypeDef *pNOR_CFI);

#if (USE_HAL_NOR_REGISTER_CALLBACKS == 1)
/* NOR callback registering/unregistering */
HAL_StatusTypeDef HAL_NOR_RegisterCallback(NOR_HandleTypeDef *hnor, HAL_NOR_CallbackIDTypeDef CallbackId, pNOR_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_NOR_UnRegisterCallback(NOR_HandleTypeDef *hnor, HAL_NOR_CallbackIDTypeDef CallbackId);
#endif
/**
  * @}
  */
  
/** @addtogroup NOR_Exported_Functions_Group3
 *  @{
 */
/* NOR Control functions  *****************************************************/
HAL_StatusTypeDef HAL_NOR_WriteOperation_Enable(NOR_HandleTypeDef *hnor);
HAL_StatusTypeDef HAL_NOR_WriteOperation_Disable(NOR_HandleTypeDef *hnor);
/**
  * @}
  */
  
/** @addtogroup NOR_Exported_Functions_Group4
 *  @{
 */
/* NOR State functions ********************************************************/
HAL_NOR_StateTypeDef  HAL_NOR_GetState(NOR_HandleTypeDef *hnor);
HAL_NOR_StatusTypeDef HAL_NOR_GetStatus(NOR_HandleTypeDef *hnor, uint32_t Address, uint32_t Timeout);
/**
  * @}
  */
    
/**
  * @}
  */
  
/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup NOR_Private_Constants NOR Private Constants
  * @{
  */
/* NOR device IDs addresses */
#define MC_ADDRESS               ((uint16_t)0x0000)
#define DEVICE_CODE1_ADDR        ((uint16_t)0x0001)
#define DEVICE_CODE2_ADDR        ((uint16_t)0x000E)
#define DEVICE_CODE3_ADDR        ((uint16_t)0x000F)

/* NOR CFI IDs addresses */
#define CFI1_ADDRESS             ((uint16_t)0x0061)
#define CFI2_ADDRESS             ((uint16_t)0x0062)
#define CFI3_ADDRESS             ((uint16_t)0x0063)
#define CFI4_ADDRESS             ((uint16_t)0x0064)

/* NOR operation wait timeout */
#define NOR_TMEOUT               ((uint16_t)0xFFFF)
   
/* NOR memory data width */
#define NOR_MEMORY_8B            ((uint8_t)0x00)
#define NOR_MEMORY_16B           ((uint8_t)0x01)

/* NOR memory device read/write start address */
#define NOR_MEMORY_ADRESS1       0x60000000U
#define NOR_MEMORY_ADRESS2       0x64000000U
#define NOR_MEMORY_ADRESS3       0x68000000U
#define NOR_MEMORY_ADRESS4       0x6C000000U
/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup NOR_Private_Macros NOR Private Macros
  * @{
  */
/**
  * @brief  NOR memory address shifting.
  * @param  __NOR_ADDRESS__ NOR base address 
  * @param  NOR_MEMORY_WIDTH NOR memory width
  * @param  ADDRESS NOR memory address 
  * @retval NOR shifted address value
  */
#define NOR_ADDR_SHIFT(__NOR_ADDRESS__, NOR_MEMORY_WIDTH, ADDRESS)    (uint32_t)(((NOR_MEMORY_WIDTH) == NOR_MEMORY_16B)? ((uint32_t)((__NOR_ADDRESS__) + (2U * (ADDRESS)))):\
                                                                                 ((uint32_t)((__NOR_ADDRESS__) + (ADDRESS))))
 
/**
  * @brief  NOR memory write data to specified address.
  * @param  ADDRESS NOR memory address 
  * @param  DATA Data to write
  * @retval None
  */
#define NOR_WRITE(ADDRESS, DATA)  (*(__IO uint16_t *)((uint32_t)(ADDRESS)) = (DATA))

/**
  * @}
  */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx ||\
          STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx ||\
          STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F413xx || STM32F423xx */
/**
  * @}
  */ 

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_NOR_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
