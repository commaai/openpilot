/**
  ******************************************************************************
  * @file    stm32f4xx_hal_flash_ex.h
  * @author  MCD Application Team
  * @brief   Header file of FLASH HAL Extension module.
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
#ifndef __STM32F4xx_HAL_FLASH_EX_H
#define __STM32F4xx_HAL_FLASH_EX_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup FLASHEx
  * @{
  */ 

/* Exported types ------------------------------------------------------------*/
/** @defgroup FLASHEx_Exported_Types FLASH Exported Types
  * @{
  */

/**
  * @brief  FLASH Erase structure definition
  */
typedef struct
{
  uint32_t TypeErase;   /*!< Mass erase or sector Erase.
                             This parameter can be a value of @ref FLASHEx_Type_Erase */

  uint32_t Banks;       /*!< Select banks to erase when Mass erase is enabled.
                             This parameter must be a value of @ref FLASHEx_Banks */

  uint32_t Sector;      /*!< Initial FLASH sector to erase when Mass erase is disabled
                             This parameter must be a value of @ref FLASHEx_Sectors */

  uint32_t NbSectors;   /*!< Number of sectors to be erased.
                             This parameter must be a value between 1 and (max number of sectors - value of Initial sector)*/

  uint32_t VoltageRange;/*!< The device voltage range which defines the erase parallelism
                             This parameter must be a value of @ref FLASHEx_Voltage_Range */

} FLASH_EraseInitTypeDef;

/**
  * @brief  FLASH Option Bytes Program structure definition
  */
typedef struct
{
  uint32_t OptionType;   /*!< Option byte to be configured.
                              This parameter can be a value of @ref FLASHEx_Option_Type */

  uint32_t WRPState;     /*!< Write protection activation or deactivation.
                              This parameter can be a value of @ref FLASHEx_WRP_State */

  uint32_t WRPSector;         /*!< Specifies the sector(s) to be write protected.
                              The value of this parameter depend on device used within the same series */

  uint32_t Banks;        /*!< Select banks for WRP activation/deactivation of all sectors.
                              This parameter must be a value of @ref FLASHEx_Banks */        

  uint32_t RDPLevel;     /*!< Set the read protection level.
                              This parameter can be a value of @ref FLASHEx_Option_Bytes_Read_Protection */

  uint32_t BORLevel;     /*!< Set the BOR Level.
                              This parameter can be a value of @ref FLASHEx_BOR_Reset_Level */

  uint8_t  USERConfig;   /*!< Program the FLASH User Option Byte: IWDG_SW / RST_STOP / RST_STDBY. */

} FLASH_OBProgramInitTypeDef;

/**
  * @brief  FLASH Advanced Option Bytes Program structure definition
  */
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) ||\
    defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
typedef struct
{
  uint32_t OptionType;     /*!< Option byte to be configured for extension.
                                This parameter can be a value of @ref FLASHEx_Advanced_Option_Type */

  uint32_t PCROPState;     /*!< PCROP activation or deactivation.
                                This parameter can be a value of @ref FLASHEx_PCROP_State */

#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
  uint16_t Sectors;        /*!< specifies the sector(s) set for PCROP.
                                This parameter can be a value of @ref FLASHEx_Option_Bytes_PC_ReadWrite_Protection */
#endif /* STM32F401xC || STM32F401xE || STM32F410xx || STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx ||\
          STM32F412Cx || STM32F413xx || STM32F423xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
  uint32_t Banks;          /*!< Select banks for PCROP activation/deactivation of all sectors.
                                This parameter must be a value of @ref FLASHEx_Banks */

  uint16_t SectorsBank1;   /*!< Specifies the sector(s) set for PCROP for Bank1.
                                This parameter can be a value of @ref FLASHEx_Option_Bytes_PC_ReadWrite_Protection */

  uint16_t SectorsBank2;   /*!< Specifies the sector(s) set for PCROP for Bank2.
                                This parameter can be a value of @ref FLASHEx_Option_Bytes_PC_ReadWrite_Protection */

  uint8_t BootConfig;      /*!< Specifies Option bytes for boot config.
                                This parameter can be a value of @ref FLASHEx_Dual_Boot */

#endif /*STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */
}FLASH_AdvOBProgramInitTypeDef;
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F401xC || STM32F401xE || STM32F410xx || STM32F411xE || STM32F446xx ||
          STM32F469xx || STM32F479xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */
/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/

/** @defgroup FLASHEx_Exported_Constants FLASH Exported Constants
  * @{
  */

/** @defgroup FLASHEx_Type_Erase FLASH Type Erase
  * @{
  */ 
#define FLASH_TYPEERASE_SECTORS         0x00000000U  /*!< Sectors erase only          */
#define FLASH_TYPEERASE_MASSERASE       0x00000001U  /*!< Flash Mass erase activation */
/**
  * @}
  */
  
/** @defgroup FLASHEx_Voltage_Range FLASH Voltage Range
  * @{
  */ 
#define FLASH_VOLTAGE_RANGE_1        0x00000000U  /*!< Device operating range: 1.8V to 2.1V                */
#define FLASH_VOLTAGE_RANGE_2        0x00000001U  /*!< Device operating range: 2.1V to 2.7V                */
#define FLASH_VOLTAGE_RANGE_3        0x00000002U  /*!< Device operating range: 2.7V to 3.6V                */
#define FLASH_VOLTAGE_RANGE_4        0x00000003U  /*!< Device operating range: 2.7V to 3.6V + External Vpp */
/**
  * @}
  */
  
/** @defgroup FLASHEx_WRP_State FLASH WRP State
  * @{
  */ 
#define OB_WRPSTATE_DISABLE       0x00000000U  /*!< Disable the write protection of the desired bank 1 sectors */
#define OB_WRPSTATE_ENABLE        0x00000001U  /*!< Enable the write protection of the desired bank 1 sectors  */
/**
  * @}
  */
  
/** @defgroup FLASHEx_Option_Type FLASH Option Type
  * @{
  */ 
#define OPTIONBYTE_WRP        0x00000001U  /*!< WRP option byte configuration  */
#define OPTIONBYTE_RDP        0x00000002U  /*!< RDP option byte configuration  */
#define OPTIONBYTE_USER       0x00000004U  /*!< USER option byte configuration */
#define OPTIONBYTE_BOR        0x00000008U  /*!< BOR option byte configuration  */
/**
  * @}
  */
  
/** @defgroup FLASHEx_Option_Bytes_Read_Protection FLASH Option Bytes Read Protection
  * @{
  */
#define OB_RDP_LEVEL_0   ((uint8_t)0xAA)
#define OB_RDP_LEVEL_1   ((uint8_t)0x55)
#define OB_RDP_LEVEL_2   ((uint8_t)0xCC) /*!< Warning: When enabling read protection level 2 
                                              it s no more possible to go back to level 1 or 0 */
/**
  * @}
  */ 
  
/** @defgroup FLASHEx_Option_Bytes_IWatchdog FLASH Option Bytes IWatchdog
  * @{
  */ 
#define OB_IWDG_SW                     ((uint8_t)0x20)  /*!< Software IWDG selected */
#define OB_IWDG_HW                     ((uint8_t)0x00)  /*!< Hardware IWDG selected */
/**
  * @}
  */ 
  
/** @defgroup FLASHEx_Option_Bytes_nRST_STOP FLASH Option Bytes nRST_STOP
  * @{
  */ 
#define OB_STOP_NO_RST                 ((uint8_t)0x40) /*!< No reset generated when entering in STOP */
#define OB_STOP_RST                    ((uint8_t)0x00) /*!< Reset generated when entering in STOP    */
/**
  * @}
  */ 


/** @defgroup FLASHEx_Option_Bytes_nRST_STDBY FLASH Option Bytes nRST_STDBY
  * @{
  */ 
#define OB_STDBY_NO_RST                ((uint8_t)0x80) /*!< No reset generated when entering in STANDBY */
#define OB_STDBY_RST                   ((uint8_t)0x00) /*!< Reset generated when entering in STANDBY    */
/**
  * @}
  */    

/** @defgroup FLASHEx_BOR_Reset_Level FLASH BOR Reset Level
  * @{
  */  
#define OB_BOR_LEVEL3          ((uint8_t)0x00)  /*!< Supply voltage ranges from 2.70 to 3.60 V */
#define OB_BOR_LEVEL2          ((uint8_t)0x04)  /*!< Supply voltage ranges from 2.40 to 2.70 V */
#define OB_BOR_LEVEL1          ((uint8_t)0x08)  /*!< Supply voltage ranges from 2.10 to 2.40 V */
#define OB_BOR_OFF             ((uint8_t)0x0C)  /*!< Supply voltage ranges from 1.62 to 2.10 V */
/**
  * @}
  */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) ||\
    defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/** @defgroup FLASHEx_PCROP_State FLASH PCROP State
  * @{
  */ 
#define OB_PCROP_STATE_DISABLE       0x00000000U  /*!< Disable PCROP */
#define OB_PCROP_STATE_ENABLE        0x00000001U  /*!< Enable PCROP  */
/**
  * @}
  */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F401xC || STM32F401xE ||\
          STM32F410xx || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

/** @defgroup FLASHEx_Advanced_Option_Type FLASH Advanced Option Type
  * @{
  */ 
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
#define OPTIONBYTE_PCROP        0x00000001U  /*!< PCROP option byte configuration      */
#define OPTIONBYTE_BOOTCONFIG   0x00000002U  /*!< BOOTConfig option byte configuration */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)
#define OPTIONBYTE_PCROP        0x00000001U  /*!<PCROP option byte configuration */
#endif /* STM32F401xC || STM32F401xE || STM32F410xx || STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx ||
          STM32F413xx || STM32F423xx */
/**
  * @}
  */

/** @defgroup FLASH_Latency FLASH Latency
  * @{
  */
/*------------------------- STM32F42xxx/STM32F43xxx/STM32F446xx/STM32F469xx/STM32F479xx ----------------------*/  
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define FLASH_LATENCY_0                FLASH_ACR_LATENCY_0WS   /*!< FLASH Zero Latency cycle      */
#define FLASH_LATENCY_1                FLASH_ACR_LATENCY_1WS   /*!< FLASH One Latency cycle       */
#define FLASH_LATENCY_2                FLASH_ACR_LATENCY_2WS   /*!< FLASH Two Latency cycles      */
#define FLASH_LATENCY_3                FLASH_ACR_LATENCY_3WS   /*!< FLASH Three Latency cycles    */
#define FLASH_LATENCY_4                FLASH_ACR_LATENCY_4WS   /*!< FLASH Four Latency cycles     */
#define FLASH_LATENCY_5                FLASH_ACR_LATENCY_5WS   /*!< FLASH Five Latency cycles     */
#define FLASH_LATENCY_6                FLASH_ACR_LATENCY_6WS   /*!< FLASH Six Latency cycles      */
#define FLASH_LATENCY_7                FLASH_ACR_LATENCY_7WS   /*!< FLASH Seven Latency cycles    */
#define FLASH_LATENCY_8                FLASH_ACR_LATENCY_8WS   /*!< FLASH Eight Latency cycles    */
#define FLASH_LATENCY_9                FLASH_ACR_LATENCY_9WS   /*!< FLASH Nine Latency cycles     */
#define FLASH_LATENCY_10               FLASH_ACR_LATENCY_10WS  /*!< FLASH Ten Latency cycles      */
#define FLASH_LATENCY_11               FLASH_ACR_LATENCY_11WS  /*!< FLASH Eleven Latency cycles   */
#define FLASH_LATENCY_12               FLASH_ACR_LATENCY_12WS  /*!< FLASH Twelve Latency cycles   */
#define FLASH_LATENCY_13               FLASH_ACR_LATENCY_13WS  /*!< FLASH Thirteen Latency cycles */
#define FLASH_LATENCY_14               FLASH_ACR_LATENCY_14WS  /*!< FLASH Fourteen Latency cycles */
#define FLASH_LATENCY_15               FLASH_ACR_LATENCY_15WS  /*!< FLASH Fifteen Latency cycles  */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */
/*--------------------------------------------------------------------------------------------------------------*/

/*-------------------------- STM32F40xxx/STM32F41xxx/STM32F401xx/STM32F411xx/STM32F423xx -----------------------*/ 
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F412Zx) || defined(STM32F412Vx) ||\
    defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
     
#define FLASH_LATENCY_0                FLASH_ACR_LATENCY_0WS   /*!< FLASH Zero Latency cycle      */
#define FLASH_LATENCY_1                FLASH_ACR_LATENCY_1WS   /*!< FLASH One Latency cycle       */
#define FLASH_LATENCY_2                FLASH_ACR_LATENCY_2WS   /*!< FLASH Two Latency cycles      */
#define FLASH_LATENCY_3                FLASH_ACR_LATENCY_3WS   /*!< FLASH Three Latency cycles    */
#define FLASH_LATENCY_4                FLASH_ACR_LATENCY_4WS   /*!< FLASH Four Latency cycles     */
#define FLASH_LATENCY_5                FLASH_ACR_LATENCY_5WS   /*!< FLASH Five Latency cycles     */
#define FLASH_LATENCY_6                FLASH_ACR_LATENCY_6WS   /*!< FLASH Six Latency cycles      */
#define FLASH_LATENCY_7                FLASH_ACR_LATENCY_7WS   /*!< FLASH Seven Latency cycles    */
#endif /* STM32F40xxx || STM32F41xxx || STM32F401xx || STM32F410xx || STM32F411xE || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx ||
          STM32F413xx || STM32F423xx */
/*--------------------------------------------------------------------------------------------------------------*/

/**
  * @}
  */ 
  

/** @defgroup FLASHEx_Banks FLASH Banks
  * @{
  */
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
#define FLASH_BANK_1     1U /*!< Bank 1   */
#define FLASH_BANK_2     2U /*!< Bank 2   */
#define FLASH_BANK_BOTH  ((uint32_t)FLASH_BANK_1 | FLASH_BANK_2) /*!< Bank1 and Bank2  */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)
#define FLASH_BANK_1     1U /*!< Bank 1   */
#endif /* STM32F40xxx || STM32F41xxx || STM32F401xx || STM32F410xx || STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx
          STM32F413xx || STM32F423xx */
/**
  * @}
  */ 
    
/** @defgroup FLASHEx_MassErase_bit FLASH Mass Erase bit
  * @{
  */
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
#define FLASH_MER_BIT     (FLASH_CR_MER1 | FLASH_CR_MER2) /*!< 2 MER bits here to clear */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)
#define FLASH_MER_BIT     (FLASH_CR_MER) /*!< only 1 MER Bit */
#endif /* STM32F40xxx || STM32F41xxx || STM32F401xx || STM32F410xx || STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx
          STM32F413xx || STM32F423xx */
/**
  * @}
  */ 

/** @defgroup FLASHEx_Sectors FLASH Sectors
  * @{
  */
/*-------------------------------------- STM32F42xxx/STM32F43xxx/STM32F469xx ------------------------------------*/   
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
#define FLASH_SECTOR_0     0U  /*!< Sector Number 0   */
#define FLASH_SECTOR_1     1U  /*!< Sector Number 1   */
#define FLASH_SECTOR_2     2U  /*!< Sector Number 2   */
#define FLASH_SECTOR_3     3U  /*!< Sector Number 3   */
#define FLASH_SECTOR_4     4U  /*!< Sector Number 4   */
#define FLASH_SECTOR_5     5U  /*!< Sector Number 5   */
#define FLASH_SECTOR_6     6U  /*!< Sector Number 6   */
#define FLASH_SECTOR_7     7U  /*!< Sector Number 7   */
#define FLASH_SECTOR_8     8U  /*!< Sector Number 8   */
#define FLASH_SECTOR_9     9U  /*!< Sector Number 9   */
#define FLASH_SECTOR_10    10U /*!< Sector Number 10  */
#define FLASH_SECTOR_11    11U /*!< Sector Number 11  */
#define FLASH_SECTOR_12    12U /*!< Sector Number 12  */
#define FLASH_SECTOR_13    13U /*!< Sector Number 13  */
#define FLASH_SECTOR_14    14U /*!< Sector Number 14  */
#define FLASH_SECTOR_15    15U /*!< Sector Number 15  */
#define FLASH_SECTOR_16    16U /*!< Sector Number 16  */
#define FLASH_SECTOR_17    17U /*!< Sector Number 17  */
#define FLASH_SECTOR_18    18U /*!< Sector Number 18  */
#define FLASH_SECTOR_19    19U /*!< Sector Number 19  */
#define FLASH_SECTOR_20    20U /*!< Sector Number 20  */
#define FLASH_SECTOR_21    21U /*!< Sector Number 21  */
#define FLASH_SECTOR_22    22U /*!< Sector Number 22  */
#define FLASH_SECTOR_23    23U /*!< Sector Number 23  */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */
/*-----------------------------------------------------------------------------------------------------*/

/*-------------------------------------- STM32F413xx/STM32F423xx --------------------------------------*/   
#if defined(STM32F413xx) || defined(STM32F423xx)
#define FLASH_SECTOR_0     0U  /*!< Sector Number 0   */
#define FLASH_SECTOR_1     1U  /*!< Sector Number 1   */
#define FLASH_SECTOR_2     2U  /*!< Sector Number 2   */
#define FLASH_SECTOR_3     3U  /*!< Sector Number 3   */
#define FLASH_SECTOR_4     4U  /*!< Sector Number 4   */
#define FLASH_SECTOR_5     5U  /*!< Sector Number 5   */
#define FLASH_SECTOR_6     6U  /*!< Sector Number 6   */
#define FLASH_SECTOR_7     7U  /*!< Sector Number 7   */
#define FLASH_SECTOR_8     8U  /*!< Sector Number 8   */
#define FLASH_SECTOR_9     9U  /*!< Sector Number 9   */
#define FLASH_SECTOR_10    10U /*!< Sector Number 10  */
#define FLASH_SECTOR_11    11U /*!< Sector Number 11  */
#define FLASH_SECTOR_12    12U /*!< Sector Number 12  */
#define FLASH_SECTOR_13    13U /*!< Sector Number 13  */
#define FLASH_SECTOR_14    14U /*!< Sector Number 14  */
#define FLASH_SECTOR_15    15U /*!< Sector Number 15  */
#endif /* STM32F413xx || STM32F423xx */
/*-----------------------------------------------------------------------------------------------------*/      

/*--------------------------------------- STM32F40xxx/STM32F41xxx -------------------------------------*/ 
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx)  
#define FLASH_SECTOR_0     0U  /*!< Sector Number 0   */
#define FLASH_SECTOR_1     1U  /*!< Sector Number 1   */
#define FLASH_SECTOR_2     2U  /*!< Sector Number 2   */
#define FLASH_SECTOR_3     3U  /*!< Sector Number 3   */
#define FLASH_SECTOR_4     4U  /*!< Sector Number 4   */
#define FLASH_SECTOR_5     5U  /*!< Sector Number 5   */
#define FLASH_SECTOR_6     6U  /*!< Sector Number 6   */
#define FLASH_SECTOR_7     7U  /*!< Sector Number 7   */
#define FLASH_SECTOR_8     8U  /*!< Sector Number 8   */
#define FLASH_SECTOR_9     9U  /*!< Sector Number 9   */
#define FLASH_SECTOR_10    10U /*!< Sector Number 10  */
#define FLASH_SECTOR_11    11U /*!< Sector Number 11  */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */
/*-----------------------------------------------------------------------------------------------------*/

/*--------------------------------------------- STM32F401xC -------------------------------------------*/ 
#if defined(STM32F401xC)
#define FLASH_SECTOR_0     0U /*!< Sector Number 0   */
#define FLASH_SECTOR_1     1U /*!< Sector Number 1   */
#define FLASH_SECTOR_2     2U /*!< Sector Number 2   */
#define FLASH_SECTOR_3     3U /*!< Sector Number 3   */
#define FLASH_SECTOR_4     4U /*!< Sector Number 4   */
#define FLASH_SECTOR_5     5U /*!< Sector Number 5   */
#endif /* STM32F401xC */
/*-----------------------------------------------------------------------------------------------------*/

/*--------------------------------------------- STM32F410xx -------------------------------------------*/ 
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define FLASH_SECTOR_0     0U /*!< Sector Number 0   */
#define FLASH_SECTOR_1     1U /*!< Sector Number 1   */
#define FLASH_SECTOR_2     2U /*!< Sector Number 2   */
#define FLASH_SECTOR_3     3U /*!< Sector Number 3   */
#define FLASH_SECTOR_4     4U /*!< Sector Number 4   */
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */
/*-----------------------------------------------------------------------------------------------------*/

/*---------------------------------- STM32F401xE/STM32F411xE/STM32F446xx ------------------------------*/
#if defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx)
#define FLASH_SECTOR_0     0U /*!< Sector Number 0   */
#define FLASH_SECTOR_1     1U /*!< Sector Number 1   */
#define FLASH_SECTOR_2     2U /*!< Sector Number 2   */
#define FLASH_SECTOR_3     3U /*!< Sector Number 3   */
#define FLASH_SECTOR_4     4U /*!< Sector Number 4   */
#define FLASH_SECTOR_5     5U /*!< Sector Number 5   */
#define FLASH_SECTOR_6     6U /*!< Sector Number 6   */
#define FLASH_SECTOR_7     7U /*!< Sector Number 7   */
#endif /* STM32F401xE || STM32F411xE || STM32F446xx */
/*-----------------------------------------------------------------------------------------------------*/

/**
  * @}
  */ 

/** @defgroup FLASHEx_Option_Bytes_Write_Protection FLASH Option Bytes Write Protection
  * @{
  */
/*--------------------------- STM32F42xxx/STM32F43xxx/STM32F469xx/STM32F479xx -------------------------*/  
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx) 
#define OB_WRP_SECTOR_0       0x00000001U /*!< Write protection of Sector0     */
#define OB_WRP_SECTOR_1       0x00000002U /*!< Write protection of Sector1     */
#define OB_WRP_SECTOR_2       0x00000004U /*!< Write protection of Sector2     */
#define OB_WRP_SECTOR_3       0x00000008U /*!< Write protection of Sector3     */
#define OB_WRP_SECTOR_4       0x00000010U /*!< Write protection of Sector4     */
#define OB_WRP_SECTOR_5       0x00000020U /*!< Write protection of Sector5     */
#define OB_WRP_SECTOR_6       0x00000040U /*!< Write protection of Sector6     */
#define OB_WRP_SECTOR_7       0x00000080U /*!< Write protection of Sector7     */
#define OB_WRP_SECTOR_8       0x00000100U /*!< Write protection of Sector8     */
#define OB_WRP_SECTOR_9       0x00000200U /*!< Write protection of Sector9     */
#define OB_WRP_SECTOR_10      0x00000400U /*!< Write protection of Sector10    */
#define OB_WRP_SECTOR_11      0x00000800U /*!< Write protection of Sector11    */
#define OB_WRP_SECTOR_12      0x00000001U << 12U /*!< Write protection of Sector12    */
#define OB_WRP_SECTOR_13      0x00000002U << 12U /*!< Write protection of Sector13    */
#define OB_WRP_SECTOR_14      0x00000004U << 12U /*!< Write protection of Sector14    */
#define OB_WRP_SECTOR_15      0x00000008U << 12U /*!< Write protection of Sector15    */
#define OB_WRP_SECTOR_16      0x00000010U << 12U /*!< Write protection of Sector16    */
#define OB_WRP_SECTOR_17      0x00000020U << 12U /*!< Write protection of Sector17    */
#define OB_WRP_SECTOR_18      0x00000040U << 12U /*!< Write protection of Sector18    */
#define OB_WRP_SECTOR_19      0x00000080U << 12U /*!< Write protection of Sector19    */
#define OB_WRP_SECTOR_20      0x00000100U << 12U /*!< Write protection of Sector20    */
#define OB_WRP_SECTOR_21      0x00000200U << 12U /*!< Write protection of Sector21    */
#define OB_WRP_SECTOR_22      0x00000400U << 12U /*!< Write protection of Sector22    */
#define OB_WRP_SECTOR_23      0x00000800U << 12U /*!< Write protection of Sector23    */
#define OB_WRP_SECTOR_All     0x00000FFFU << 12U /*!< Write protection of all Sectors */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */
/*-----------------------------------------------------------------------------------------------------*/

/*--------------------------------------- STM32F413xx/STM32F423xx -------------------------------------*/ 
#if defined(STM32F413xx) || defined(STM32F423xx)  
#define OB_WRP_SECTOR_0       0x00000001U /*!< Write protection of Sector0     */
#define OB_WRP_SECTOR_1       0x00000002U /*!< Write protection of Sector1     */
#define OB_WRP_SECTOR_2       0x00000004U /*!< Write protection of Sector2     */
#define OB_WRP_SECTOR_3       0x00000008U /*!< Write protection of Sector3     */
#define OB_WRP_SECTOR_4       0x00000010U /*!< Write protection of Sector4     */
#define OB_WRP_SECTOR_5       0x00000020U /*!< Write protection of Sector5     */
#define OB_WRP_SECTOR_6       0x00000040U /*!< Write protection of Sector6     */
#define OB_WRP_SECTOR_7       0x00000080U /*!< Write protection of Sector7     */
#define OB_WRP_SECTOR_8       0x00000100U /*!< Write protection of Sector8     */
#define OB_WRP_SECTOR_9       0x00000200U /*!< Write protection of Sector9     */
#define OB_WRP_SECTOR_10      0x00000400U /*!< Write protection of Sector10    */
#define OB_WRP_SECTOR_11      0x00000800U /*!< Write protection of Sector11    */
#define OB_WRP_SECTOR_12      0x00001000U /*!< Write protection of Sector12    */
#define OB_WRP_SECTOR_13      0x00002000U /*!< Write protection of Sector13    */
#define OB_WRP_SECTOR_14      0x00004000U /*!< Write protection of Sector14    */
#define OB_WRP_SECTOR_15      0x00004000U /*!< Write protection of Sector15    */      
#define OB_WRP_SECTOR_All     0x00007FFFU /*!< Write protection of all Sectors */
#endif /* STM32F413xx || STM32F423xx */
/*-----------------------------------------------------------------------------------------------------*/    
      
/*--------------------------------------- STM32F40xxx/STM32F41xxx -------------------------------------*/ 
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx)  
#define OB_WRP_SECTOR_0       0x00000001U /*!< Write protection of Sector0     */
#define OB_WRP_SECTOR_1       0x00000002U /*!< Write protection of Sector1     */
#define OB_WRP_SECTOR_2       0x00000004U /*!< Write protection of Sector2     */
#define OB_WRP_SECTOR_3       0x00000008U /*!< Write protection of Sector3     */
#define OB_WRP_SECTOR_4       0x00000010U /*!< Write protection of Sector4     */
#define OB_WRP_SECTOR_5       0x00000020U /*!< Write protection of Sector5     */
#define OB_WRP_SECTOR_6       0x00000040U /*!< Write protection of Sector6     */
#define OB_WRP_SECTOR_7       0x00000080U /*!< Write protection of Sector7     */
#define OB_WRP_SECTOR_8       0x00000100U /*!< Write protection of Sector8     */
#define OB_WRP_SECTOR_9       0x00000200U /*!< Write protection of Sector9     */
#define OB_WRP_SECTOR_10      0x00000400U /*!< Write protection of Sector10    */
#define OB_WRP_SECTOR_11      0x00000800U /*!< Write protection of Sector11    */
#define OB_WRP_SECTOR_All     0x00000FFFU /*!< Write protection of all Sectors */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */
/*-----------------------------------------------------------------------------------------------------*/

/*--------------------------------------------- STM32F401xC -------------------------------------------*/
#if defined(STM32F401xC)
#define OB_WRP_SECTOR_0       0x00000001U /*!< Write protection of Sector0     */
#define OB_WRP_SECTOR_1       0x00000002U /*!< Write protection of Sector1     */
#define OB_WRP_SECTOR_2       0x00000004U /*!< Write protection of Sector2     */
#define OB_WRP_SECTOR_3       0x00000008U /*!< Write protection of Sector3     */
#define OB_WRP_SECTOR_4       0x00000010U /*!< Write protection of Sector4     */
#define OB_WRP_SECTOR_5       0x00000020U /*!< Write protection of Sector5     */
#define OB_WRP_SECTOR_All     0x00000FFFU /*!< Write protection of all Sectors */
#endif /* STM32F401xC */
/*-----------------------------------------------------------------------------------------------------*/
 
/*--------------------------------------------- STM32F410xx -------------------------------------------*/
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define OB_WRP_SECTOR_0       0x00000001U /*!< Write protection of Sector0     */
#define OB_WRP_SECTOR_1       0x00000002U /*!< Write protection of Sector1     */
#define OB_WRP_SECTOR_2       0x00000004U /*!< Write protection of Sector2     */
#define OB_WRP_SECTOR_3       0x00000008U /*!< Write protection of Sector3     */
#define OB_WRP_SECTOR_4       0x00000010U /*!< Write protection of Sector4     */
#define OB_WRP_SECTOR_All     0x00000FFFU /*!< Write protection of all Sectors */
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */
/*-----------------------------------------------------------------------------------------------------*/

/*---------------------------------- STM32F401xE/STM32F411xE/STM32F446xx ------------------------------*/
#if defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx)
#define OB_WRP_SECTOR_0       0x00000001U /*!< Write protection of Sector0     */
#define OB_WRP_SECTOR_1       0x00000002U /*!< Write protection of Sector1     */
#define OB_WRP_SECTOR_2       0x00000004U /*!< Write protection of Sector2     */
#define OB_WRP_SECTOR_3       0x00000008U /*!< Write protection of Sector3     */
#define OB_WRP_SECTOR_4       0x00000010U /*!< Write protection of Sector4     */
#define OB_WRP_SECTOR_5       0x00000020U /*!< Write protection of Sector5     */
#define OB_WRP_SECTOR_6       0x00000040U /*!< Write protection of Sector6     */
#define OB_WRP_SECTOR_7       0x00000080U /*!< Write protection of Sector7     */
#define OB_WRP_SECTOR_All     0x00000FFFU /*!< Write protection of all Sectors */
#endif /* STM32F401xE || STM32F411xE || STM32F446xx */
/*-----------------------------------------------------------------------------------------------------*/
/**
  * @}
  */
  
/** @defgroup FLASHEx_Option_Bytes_PC_ReadWrite_Protection FLASH Option Bytes PC ReadWrite Protection
  * @{
  */
/*-------------------------------- STM32F42xxx/STM32F43xxx/STM32F469xx/STM32F479xx ---------------------------*/   
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx) 
#define OB_PCROP_SECTOR_0        0x00000001U /*!< PC Read/Write protection of Sector0      */
#define OB_PCROP_SECTOR_1        0x00000002U /*!< PC Read/Write protection of Sector1      */
#define OB_PCROP_SECTOR_2        0x00000004U /*!< PC Read/Write protection of Sector2      */
#define OB_PCROP_SECTOR_3        0x00000008U /*!< PC Read/Write protection of Sector3      */
#define OB_PCROP_SECTOR_4        0x00000010U /*!< PC Read/Write protection of Sector4      */
#define OB_PCROP_SECTOR_5        0x00000020U /*!< PC Read/Write protection of Sector5      */
#define OB_PCROP_SECTOR_6        0x00000040U /*!< PC Read/Write protection of Sector6      */
#define OB_PCROP_SECTOR_7        0x00000080U /*!< PC Read/Write protection of Sector7      */
#define OB_PCROP_SECTOR_8        0x00000100U /*!< PC Read/Write protection of Sector8      */
#define OB_PCROP_SECTOR_9        0x00000200U /*!< PC Read/Write protection of Sector9      */
#define OB_PCROP_SECTOR_10       0x00000400U /*!< PC Read/Write protection of Sector10     */
#define OB_PCROP_SECTOR_11       0x00000800U /*!< PC Read/Write protection of Sector11     */
#define OB_PCROP_SECTOR_12       0x00000001U /*!< PC Read/Write protection of Sector12     */
#define OB_PCROP_SECTOR_13       0x00000002U /*!< PC Read/Write protection of Sector13     */
#define OB_PCROP_SECTOR_14       0x00000004U /*!< PC Read/Write protection of Sector14     */
#define OB_PCROP_SECTOR_15       0x00000008U /*!< PC Read/Write protection of Sector15     */
#define OB_PCROP_SECTOR_16       0x00000010U /*!< PC Read/Write protection of Sector16     */
#define OB_PCROP_SECTOR_17       0x00000020U /*!< PC Read/Write protection of Sector17     */
#define OB_PCROP_SECTOR_18       0x00000040U /*!< PC Read/Write protection of Sector18     */
#define OB_PCROP_SECTOR_19       0x00000080U /*!< PC Read/Write protection of Sector19     */
#define OB_PCROP_SECTOR_20       0x00000100U /*!< PC Read/Write protection of Sector20     */
#define OB_PCROP_SECTOR_21       0x00000200U /*!< PC Read/Write protection of Sector21     */
#define OB_PCROP_SECTOR_22       0x00000400U /*!< PC Read/Write protection of Sector22     */
#define OB_PCROP_SECTOR_23       0x00000800U /*!< PC Read/Write protection of Sector23     */
#define OB_PCROP_SECTOR_All      0x00000FFFU /*!< PC Read/Write protection of all Sectors  */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */
/*-----------------------------------------------------------------------------------------------------*/
      
/*------------------------------------- STM32F413xx/STM32F423xx ---------------------------------------*/
#if defined(STM32F413xx) || defined(STM32F423xx)  
#define OB_PCROP_SECTOR_0        0x00000001U /*!< PC Read/Write protection of Sector0      */
#define OB_PCROP_SECTOR_1        0x00000002U /*!< PC Read/Write protection of Sector1      */
#define OB_PCROP_SECTOR_2        0x00000004U /*!< PC Read/Write protection of Sector2      */
#define OB_PCROP_SECTOR_3        0x00000008U /*!< PC Read/Write protection of Sector3      */
#define OB_PCROP_SECTOR_4        0x00000010U /*!< PC Read/Write protection of Sector4      */
#define OB_PCROP_SECTOR_5        0x00000020U /*!< PC Read/Write protection of Sector5      */
#define OB_PCROP_SECTOR_6        0x00000040U /*!< PC Read/Write protection of Sector6      */
#define OB_PCROP_SECTOR_7        0x00000080U /*!< PC Read/Write protection of Sector7      */
#define OB_PCROP_SECTOR_8        0x00000100U /*!< PC Read/Write protection of Sector8      */
#define OB_PCROP_SECTOR_9        0x00000200U /*!< PC Read/Write protection of Sector9      */
#define OB_PCROP_SECTOR_10       0x00000400U /*!< PC Read/Write protection of Sector10     */
#define OB_PCROP_SECTOR_11       0x00000800U /*!< PC Read/Write protection of Sector11     */
#define OB_PCROP_SECTOR_12       0x00001000U /*!< PC Read/Write protection of Sector12     */
#define OB_PCROP_SECTOR_13       0x00002000U /*!< PC Read/Write protection of Sector13     */
#define OB_PCROP_SECTOR_14       0x00004000U /*!< PC Read/Write protection of Sector14     */
#define OB_PCROP_SECTOR_15       0x00004000U /*!< PC Read/Write protection of Sector15     */      
#define OB_PCROP_SECTOR_All      0x00007FFFU /*!< PC Read/Write protection of all Sectors  */
#endif /* STM32F413xx || STM32F423xx */
/*-----------------------------------------------------------------------------------------------------*/      

/*--------------------------------------------- STM32F401xC -------------------------------------------*/
#if defined(STM32F401xC)
#define OB_PCROP_SECTOR_0        0x00000001U /*!< PC Read/Write protection of Sector0      */
#define OB_PCROP_SECTOR_1        0x00000002U /*!< PC Read/Write protection of Sector1      */
#define OB_PCROP_SECTOR_2        0x00000004U /*!< PC Read/Write protection of Sector2      */
#define OB_PCROP_SECTOR_3        0x00000008U /*!< PC Read/Write protection of Sector3      */
#define OB_PCROP_SECTOR_4        0x00000010U /*!< PC Read/Write protection of Sector4      */
#define OB_PCROP_SECTOR_5        0x00000020U /*!< PC Read/Write protection of Sector5      */
#define OB_PCROP_SECTOR_All      0x00000FFFU /*!< PC Read/Write protection of all Sectors  */
#endif /* STM32F401xC */
/*-----------------------------------------------------------------------------------------------------*/

/*--------------------------------------------- STM32F410xx -------------------------------------------*/
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define OB_PCROP_SECTOR_0        0x00000001U /*!< PC Read/Write protection of Sector0      */
#define OB_PCROP_SECTOR_1        0x00000002U /*!< PC Read/Write protection of Sector1      */
#define OB_PCROP_SECTOR_2        0x00000004U /*!< PC Read/Write protection of Sector2      */
#define OB_PCROP_SECTOR_3        0x00000008U /*!< PC Read/Write protection of Sector3      */
#define OB_PCROP_SECTOR_4        0x00000010U /*!< PC Read/Write protection of Sector4      */
#define OB_PCROP_SECTOR_All      0x00000FFFU /*!< PC Read/Write protection of all Sectors  */
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */
/*-----------------------------------------------------------------------------------------------------*/

/*-------------- STM32F401xE/STM32F411xE/STM32F412Zx/STM32F412Vx/STM32F412Rx/STM32F412Cx/STM32F446xx --*/
#if defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx)  
#define OB_PCROP_SECTOR_0        0x00000001U /*!< PC Read/Write protection of Sector0      */
#define OB_PCROP_SECTOR_1        0x00000002U /*!< PC Read/Write protection of Sector1      */
#define OB_PCROP_SECTOR_2        0x00000004U /*!< PC Read/Write protection of Sector2      */
#define OB_PCROP_SECTOR_3        0x00000008U /*!< PC Read/Write protection of Sector3      */
#define OB_PCROP_SECTOR_4        0x00000010U /*!< PC Read/Write protection of Sector4      */
#define OB_PCROP_SECTOR_5        0x00000020U /*!< PC Read/Write protection of Sector5      */
#define OB_PCROP_SECTOR_6        0x00000040U /*!< PC Read/Write protection of Sector6      */
#define OB_PCROP_SECTOR_7        0x00000080U /*!< PC Read/Write protection of Sector7      */
#define OB_PCROP_SECTOR_All      0x00000FFFU /*!< PC Read/Write protection of all Sectors  */
#endif /* STM32F401xE || STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */
/*-----------------------------------------------------------------------------------------------------*/

/**
  * @}
  */
  
/** @defgroup FLASHEx_Dual_Boot FLASH Dual Boot
  * @{
  */
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx) 
#define OB_DUAL_BOOT_ENABLE   ((uint8_t)0x10) /*!< Dual Bank Boot Enable                             */
#define OB_DUAL_BOOT_DISABLE  ((uint8_t)0x00) /*!< Dual Bank Boot Disable, always boot on User Flash */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */
/**
  * @}
  */

/** @defgroup  FLASHEx_Selection_Protection_Mode FLASH Selection Protection Mode
  * @{
  */
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) ||\
    defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
#define OB_PCROP_DESELECTED     ((uint8_t)0x00) /*!< Disabled PcROP, nWPRi bits used for Write Protection on sector i */
#define OB_PCROP_SELECTED       ((uint8_t)0x80) /*!< Enable PcROP, nWPRi bits used for PCRoP Protection on sector i   */
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F401xC || STM32F401xE ||\
          STM32F410xx || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */
/**
  * @}
  */

/**
  * @}
  */ 
  
/* Exported macro ------------------------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup FLASHEx_Exported_Functions
  * @{
  */

/** @addtogroup FLASHEx_Exported_Functions_Group1
  * @{
  */
/* Extension Program operation functions  *************************************/
HAL_StatusTypeDef HAL_FLASHEx_Erase(FLASH_EraseInitTypeDef *pEraseInit, uint32_t *SectorError);
HAL_StatusTypeDef HAL_FLASHEx_Erase_IT(FLASH_EraseInitTypeDef *pEraseInit);
HAL_StatusTypeDef HAL_FLASHEx_OBProgram(FLASH_OBProgramInitTypeDef *pOBInit);
void              HAL_FLASHEx_OBGetConfig(FLASH_OBProgramInitTypeDef *pOBInit);

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) ||\
    defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
HAL_StatusTypeDef HAL_FLASHEx_AdvOBProgram (FLASH_AdvOBProgramInitTypeDef *pAdvOBInit);
void              HAL_FLASHEx_AdvOBGetConfig(FLASH_AdvOBProgramInitTypeDef *pAdvOBInit);
HAL_StatusTypeDef HAL_FLASHEx_OB_SelectPCROP(void);
HAL_StatusTypeDef HAL_FLASHEx_OB_DeSelectPCROP(void);
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F401xC || STM32F401xE ||\
          STM32F410xx || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
uint16_t          HAL_FLASHEx_OB_GetBank2WRP(void);
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */
/**
  * @}
  */

/**
  * @}
  */
/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup FLASHEx_Private_Constants FLASH Private Constants
  * @{
  */
/*--------------------------------- STM32F42xxx/STM32F43xxx/STM32F469xx/STM32F479xx---------------------*/ 
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define FLASH_SECTOR_TOTAL  24U
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

/*-------------------------------------- STM32F413xx/STM32F423xx ---------------------------------------*/
#if defined(STM32F413xx) || defined(STM32F423xx)
#define FLASH_SECTOR_TOTAL  16U
#endif /* STM32F413xx || STM32F423xx */

/*--------------------------------------- STM32F40xxx/STM32F41xxx -------------------------------------*/ 
#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx)  
#define FLASH_SECTOR_TOTAL  12U
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */

/*--------------------------------------------- STM32F401xC -------------------------------------------*/ 
#if defined(STM32F401xC)
#define FLASH_SECTOR_TOTAL  6U
#endif /* STM32F401xC */

/*--------------------------------------------- STM32F410xx -------------------------------------------*/ 
#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define FLASH_SECTOR_TOTAL  5U
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

/*--------------------------------- STM32F401xE/STM32F411xE/STM32F412xG/STM32F446xx -------------------*/
#if defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx)
#define FLASH_SECTOR_TOTAL  8U
#endif /* STM32F401xE || STM32F411xE || STM32F446xx */

/** 
  * @brief OPTCR1 register byte 2 (Bits[23:16]) base address  
  */ 
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)  
#define OPTCR1_BYTE2_ADDRESS         0x40023C1AU
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup FLASHEx_Private_Macros FLASH Private Macros
  * @{
  */

/** @defgroup FLASHEx_IS_FLASH_Definitions FLASH Private macros to check input parameters
  * @{
  */

#define IS_FLASH_TYPEERASE(VALUE)(((VALUE) == FLASH_TYPEERASE_SECTORS) || \
                                  ((VALUE) == FLASH_TYPEERASE_MASSERASE))  

#define IS_VOLTAGERANGE(RANGE)(((RANGE) == FLASH_VOLTAGE_RANGE_1) || \
                               ((RANGE) == FLASH_VOLTAGE_RANGE_2) || \
                               ((RANGE) == FLASH_VOLTAGE_RANGE_3) || \
                               ((RANGE) == FLASH_VOLTAGE_RANGE_4))  

#define IS_WRPSTATE(VALUE)(((VALUE) == OB_WRPSTATE_DISABLE) || \
                           ((VALUE) == OB_WRPSTATE_ENABLE))  

#define IS_OPTIONBYTE(VALUE)(((VALUE) <= (OPTIONBYTE_WRP|OPTIONBYTE_RDP|OPTIONBYTE_USER|OPTIONBYTE_BOR)))

#define IS_OB_RDP_LEVEL(LEVEL) (((LEVEL) == OB_RDP_LEVEL_0) ||\
                                ((LEVEL) == OB_RDP_LEVEL_1) ||\
                                ((LEVEL) == OB_RDP_LEVEL_2))

#define IS_OB_IWDG_SOURCE(SOURCE) (((SOURCE) == OB_IWDG_SW) || ((SOURCE) == OB_IWDG_HW))

#define IS_OB_STOP_SOURCE(SOURCE) (((SOURCE) == OB_STOP_NO_RST) || ((SOURCE) == OB_STOP_RST))

#define IS_OB_STDBY_SOURCE(SOURCE) (((SOURCE) == OB_STDBY_NO_RST) || ((SOURCE) == OB_STDBY_RST))

#define IS_OB_BOR_LEVEL(LEVEL) (((LEVEL) == OB_BOR_LEVEL1) || ((LEVEL) == OB_BOR_LEVEL2) ||\
                                ((LEVEL) == OB_BOR_LEVEL3) || ((LEVEL) == OB_BOR_OFF))

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) ||\
    defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
#define IS_PCROPSTATE(VALUE)(((VALUE) == OB_PCROP_STATE_DISABLE) || \
                             ((VALUE) == OB_PCROP_STATE_ENABLE))  
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F401xC || STM32F401xE ||\
          STM32F410xx || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx)
#define IS_OBEX(VALUE)(((VALUE) == OPTIONBYTE_PCROP) || \
                       ((VALUE) == OPTIONBYTE_BOOTCONFIG))  
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)
#define IS_OBEX(VALUE)(((VALUE) == OPTIONBYTE_PCROP))  
#endif /* STM32F401xC || STM32F401xE || STM32F410xx || STM32F411xE || STM32F446xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */
  
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx)|| defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define IS_FLASH_LATENCY(LATENCY) (((LATENCY) == FLASH_LATENCY_0)  || \
                                   ((LATENCY) == FLASH_LATENCY_1)  || \
                                   ((LATENCY) == FLASH_LATENCY_2)  || \
                                   ((LATENCY) == FLASH_LATENCY_3)  || \
                                   ((LATENCY) == FLASH_LATENCY_4)  || \
                                   ((LATENCY) == FLASH_LATENCY_5)  || \
                                   ((LATENCY) == FLASH_LATENCY_6)  || \
                                   ((LATENCY) == FLASH_LATENCY_7)  || \
                                   ((LATENCY) == FLASH_LATENCY_8)  || \
                                   ((LATENCY) == FLASH_LATENCY_9)  || \
                                   ((LATENCY) == FLASH_LATENCY_10) || \
                                   ((LATENCY) == FLASH_LATENCY_11) || \
                                   ((LATENCY) == FLASH_LATENCY_12) || \
                                   ((LATENCY) == FLASH_LATENCY_13) || \
                                   ((LATENCY) == FLASH_LATENCY_14) || \
                                   ((LATENCY) == FLASH_LATENCY_15))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F412Zx) || defined(STM32F412Vx) ||\
    defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
#define IS_FLASH_LATENCY(LATENCY) (((LATENCY) == FLASH_LATENCY_0)  || \
                                   ((LATENCY) == FLASH_LATENCY_1)  || \
                                   ((LATENCY) == FLASH_LATENCY_2)  || \
                                   ((LATENCY) == FLASH_LATENCY_3)  || \
                                   ((LATENCY) == FLASH_LATENCY_4)  || \
                                   ((LATENCY) == FLASH_LATENCY_5)  || \
                                   ((LATENCY) == FLASH_LATENCY_6)  || \
                                   ((LATENCY) == FLASH_LATENCY_7))
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F401xC || STM32F401xE || STM32F410xx || STM32F411xE || STM32F412Zx || STM32F412Vx ||\
          STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define IS_FLASH_BANK(BANK) (((BANK) == FLASH_BANK_1)  || \
                             ((BANK) == FLASH_BANK_2)  || \
                             ((BANK) == FLASH_BANK_BOTH))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx|| STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)
#define IS_FLASH_BANK(BANK) (((BANK) == FLASH_BANK_1))
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F401xC || STM32F401xE || STM32F410xx || STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx ||\
          STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */
 
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define IS_FLASH_SECTOR(SECTOR) ( ((SECTOR) == FLASH_SECTOR_0)   || ((SECTOR) == FLASH_SECTOR_1)   ||\
                                  ((SECTOR) == FLASH_SECTOR_2)   || ((SECTOR) == FLASH_SECTOR_3)   ||\
                                  ((SECTOR) == FLASH_SECTOR_4)   || ((SECTOR) == FLASH_SECTOR_5)   ||\
                                  ((SECTOR) == FLASH_SECTOR_6)   || ((SECTOR) == FLASH_SECTOR_7)   ||\
                                  ((SECTOR) == FLASH_SECTOR_8)   || ((SECTOR) == FLASH_SECTOR_9)   ||\
                                  ((SECTOR) == FLASH_SECTOR_10)  || ((SECTOR) == FLASH_SECTOR_11)  ||\
                                  ((SECTOR) == FLASH_SECTOR_12)  || ((SECTOR) == FLASH_SECTOR_13)  ||\
                                  ((SECTOR) == FLASH_SECTOR_14)  || ((SECTOR) == FLASH_SECTOR_15)  ||\
                                  ((SECTOR) == FLASH_SECTOR_16)  || ((SECTOR) == FLASH_SECTOR_17)  ||\
                                  ((SECTOR) == FLASH_SECTOR_18)  || ((SECTOR) == FLASH_SECTOR_19)  ||\
                                  ((SECTOR) == FLASH_SECTOR_20)  || ((SECTOR) == FLASH_SECTOR_21)  ||\
                                  ((SECTOR) == FLASH_SECTOR_22)  || ((SECTOR) == FLASH_SECTOR_23))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F413xx) || defined(STM32F423xx)
#define IS_FLASH_SECTOR(SECTOR) ( ((SECTOR) == FLASH_SECTOR_0)   || ((SECTOR) == FLASH_SECTOR_1)   ||\
                                  ((SECTOR) == FLASH_SECTOR_2)   || ((SECTOR) == FLASH_SECTOR_3)   ||\
                                  ((SECTOR) == FLASH_SECTOR_4)   || ((SECTOR) == FLASH_SECTOR_5)   ||\
                                  ((SECTOR) == FLASH_SECTOR_6)   || ((SECTOR) == FLASH_SECTOR_7)   ||\
                                  ((SECTOR) == FLASH_SECTOR_8)   || ((SECTOR) == FLASH_SECTOR_9)   ||\
                                  ((SECTOR) == FLASH_SECTOR_10)  || ((SECTOR) == FLASH_SECTOR_11)  ||\
                                  ((SECTOR) == FLASH_SECTOR_12)  || ((SECTOR) == FLASH_SECTOR_13)  ||\
                                  ((SECTOR) == FLASH_SECTOR_14)  || ((SECTOR) == FLASH_SECTOR_15))
#endif /* STM32F413xx || STM32F423xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx)  
#define IS_FLASH_SECTOR(SECTOR) (((SECTOR) == FLASH_SECTOR_0)   || ((SECTOR) == FLASH_SECTOR_1)   ||\
                                 ((SECTOR) == FLASH_SECTOR_2)   || ((SECTOR) == FLASH_SECTOR_3)   ||\
                                 ((SECTOR) == FLASH_SECTOR_4)   || ((SECTOR) == FLASH_SECTOR_5)   ||\
                                 ((SECTOR) == FLASH_SECTOR_6)   || ((SECTOR) == FLASH_SECTOR_7)   ||\
                                 ((SECTOR) == FLASH_SECTOR_8)   || ((SECTOR) == FLASH_SECTOR_9)   ||\
                                 ((SECTOR) == FLASH_SECTOR_10)  || ((SECTOR) == FLASH_SECTOR_11))
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */

#if defined(STM32F401xC)
#define IS_FLASH_SECTOR(SECTOR) (((SECTOR) == FLASH_SECTOR_0)   || ((SECTOR) == FLASH_SECTOR_1)   ||\
                                 ((SECTOR) == FLASH_SECTOR_2)   || ((SECTOR) == FLASH_SECTOR_3)   ||\
                                 ((SECTOR) == FLASH_SECTOR_4)   || ((SECTOR) == FLASH_SECTOR_5))
#endif /* STM32F401xC */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define IS_FLASH_SECTOR(SECTOR) (((SECTOR) == FLASH_SECTOR_0)   || ((SECTOR) == FLASH_SECTOR_1)   ||\
                                 ((SECTOR) == FLASH_SECTOR_2)   || ((SECTOR) == FLASH_SECTOR_3)   ||\
                                 ((SECTOR) == FLASH_SECTOR_4))
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

#if defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx)
#define IS_FLASH_SECTOR(SECTOR) (((SECTOR) == FLASH_SECTOR_0)   || ((SECTOR) == FLASH_SECTOR_1)   ||\
                                 ((SECTOR) == FLASH_SECTOR_2)   || ((SECTOR) == FLASH_SECTOR_3)   ||\
                                 ((SECTOR) == FLASH_SECTOR_4)   || ((SECTOR) == FLASH_SECTOR_5)   ||\
                                 ((SECTOR) == FLASH_SECTOR_6)   || ((SECTOR) == FLASH_SECTOR_7))
#endif /* STM32F401xE || STM32F411xE || STM32F446xx */

#define IS_FLASH_ADDRESS(ADDRESS) ((((ADDRESS) >= FLASH_BASE) && ((ADDRESS) <= FLASH_END)) || \
                                   (((ADDRESS) >= FLASH_OTP_BASE) && ((ADDRESS) <= FLASH_OTP_END)))

#define IS_FLASH_NBSECTORS(NBSECTORS) (((NBSECTORS) != 0) && ((NBSECTORS) <= FLASH_SECTOR_TOTAL))
  
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx) 
#define IS_OB_WRP_SECTOR(SECTOR)((((SECTOR) & 0xFF000000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F413xx) || defined(STM32F423xx) 
#define IS_OB_WRP_SECTOR(SECTOR)((((SECTOR) & 0xFFFF8000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F413xx || STM32F423xx */

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx)
#define IS_OB_WRP_SECTOR(SECTOR)((((SECTOR) & 0xFFFFF000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx */

#if defined(STM32F401xC)
#define IS_OB_WRP_SECTOR(SECTOR)((((SECTOR) & 0xFFFFF000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F401xC */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define IS_OB_WRP_SECTOR(SECTOR)((((SECTOR) & 0xFFFFF000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

#if defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) || defined(STM32F412Vx) ||\
    defined(STM32F412Rx) || defined(STM32F412Cx)  
#define IS_OB_WRP_SECTOR(SECTOR)((((SECTOR) & 0xFFFFF000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F401xE || STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */
   
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F469xx) || defined(STM32F479xx)
#define IS_OB_PCROP(SECTOR)((((SECTOR) & 0xFFFFF000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F413xx) || defined(STM32F423xx)
#define IS_OB_PCROP(SECTOR)((((SECTOR) & 0xFFFF8000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))      
#endif /* STM32F413xx || STM32F423xx */

#if defined(STM32F401xC)
#define IS_OB_PCROP(SECTOR)((((SECTOR) & 0xFFFFF000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F401xC */

#if defined(STM32F410Tx) || defined(STM32F410Cx) || defined(STM32F410Rx)
#define IS_OB_PCROP(SECTOR)((((SECTOR) & 0xFFFFF000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F410Tx || STM32F410Cx || STM32F410Rx */

#if defined(STM32F401xE) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F412Zx) || defined(STM32F412Vx) ||\
    defined(STM32F412Rx) || defined(STM32F412Cx)  
#define IS_OB_PCROP(SECTOR)((((SECTOR) & 0xFFFFF000U) == 0x00000000U) && ((SECTOR) != 0x00000000U))
#endif /* STM32F401xE || STM32F411xE || STM32F446xx || STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F469xx) || defined(STM32F479xx) 
#define IS_OB_BOOT(BOOT) (((BOOT) == OB_DUAL_BOOT_ENABLE) || ((BOOT) == OB_DUAL_BOOT_DISABLE))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F469xx || STM32F479xx */

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F401xC) || defined(STM32F401xE) || defined(STM32F410Tx) || defined(STM32F410Cx) ||\
    defined(STM32F410Rx) || defined(STM32F411xE) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx) || defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) ||\
    defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
#define IS_OB_PCROP_SELECT(PCROP) (((PCROP) == OB_PCROP_SELECTED) || ((PCROP) == OB_PCROP_DESELECTED))
#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F401xC || STM32F401xE ||\
          STM32F410xx || STM32F411xE || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */
/**
  * @}
  */

/**
  * @}
  */

/* Private functions ---------------------------------------------------------*/
/** @defgroup FLASHEx_Private_Functions FLASH Private Functions
  * @{
  */
void FLASH_Erase_Sector(uint32_t Sector, uint8_t VoltageRange);
void FLASH_FlushCaches(void);
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

#endif /* __STM32F4xx_HAL_FLASH_EX_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
