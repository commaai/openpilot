/**
  ******************************************************************************
  * @file    stm32f4xx_hal_qspi.h
  * @author  MCD Application Team
  * @brief   Header file of QSPI HAL module.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2016 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                       opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef STM32F4xx_HAL_QSPI_H
#define STM32F4xx_HAL_QSPI_H

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

#if defined(QUADSPI)

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup QSPI
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup QSPI_Exported_Types QSPI Exported Types
  * @{
  */

/**
  * @brief  QSPI Init structure definition
  */
typedef struct
{
  uint32_t ClockPrescaler;     /* Specifies the prescaler factor for generating clock based on the AHB clock.
                                  This parameter can be a number between 0 and 255 */
  uint32_t FifoThreshold;      /* Specifies the threshold number of bytes in the FIFO (used only in indirect mode)
                                  This parameter can be a value between 1 and 32 */
  uint32_t SampleShifting;     /* Specifies the Sample Shift. The data is sampled 1/2 clock cycle delay later to
                                  take in account external signal delays. (It should be QSPI_SAMPLE_SHIFTING_NONE in DDR mode)
                                  This parameter can be a value of @ref QSPI_SampleShifting */
  uint32_t FlashSize;          /* Specifies the Flash Size. FlashSize+1 is effectively the number of address bits
                                  required to address the flash memory. The flash capacity can be up to 4GB
                                  (addressed using 32 bits) in indirect mode, but the addressable space in
                                  memory-mapped mode is limited to 256MB
                                  This parameter can be a number between 0 and 31 */
  uint32_t ChipSelectHighTime; /* Specifies the Chip Select High Time. ChipSelectHighTime+1 defines the minimum number
                                  of clock cycles which the chip select must remain high between commands.
                                  This parameter can be a value of @ref QSPI_ChipSelectHighTime */
  uint32_t ClockMode;          /* Specifies the Clock Mode. It indicates the level that clock takes between commands.
                                  This parameter can be a value of @ref QSPI_ClockMode */
  uint32_t FlashID;            /* Specifies the Flash which will be used,
                                  This parameter can be a value of @ref QSPI_Flash_Select */
  uint32_t DualFlash;          /* Specifies the Dual Flash Mode State
                                  This parameter can be a value of @ref QSPI_DualFlash_Mode */
}QSPI_InitTypeDef;

/**
  * @brief HAL QSPI State structures definition
  */
typedef enum
{
  HAL_QSPI_STATE_RESET             = 0x00U,    /*!< Peripheral not initialized                            */
  HAL_QSPI_STATE_READY             = 0x01U,    /*!< Peripheral initialized and ready for use              */
  HAL_QSPI_STATE_BUSY              = 0x02U,    /*!< Peripheral in indirect mode and busy                  */
  HAL_QSPI_STATE_BUSY_INDIRECT_TX  = 0x12U,    /*!< Peripheral in indirect mode with transmission ongoing */
  HAL_QSPI_STATE_BUSY_INDIRECT_RX  = 0x22U,    /*!< Peripheral in indirect mode with reception ongoing    */
  HAL_QSPI_STATE_BUSY_AUTO_POLLING = 0x42U,    /*!< Peripheral in auto polling mode ongoing               */
  HAL_QSPI_STATE_BUSY_MEM_MAPPED   = 0x82U,    /*!< Peripheral in memory mapped mode ongoing              */
  HAL_QSPI_STATE_ABORT             = 0x08U,    /*!< Peripheral with abort request ongoing                 */
  HAL_QSPI_STATE_ERROR             = 0x04U     /*!< Peripheral in error                                   */
}HAL_QSPI_StateTypeDef;

/**
  * @brief  QSPI Handle Structure definition
  */
#if (USE_HAL_QSPI_REGISTER_CALLBACKS == 1)
typedef struct __QSPI_HandleTypeDef
#else
typedef struct
#endif
{
  QUADSPI_TypeDef            *Instance;        /* QSPI registers base address        */
  QSPI_InitTypeDef           Init;             /* QSPI communication parameters      */
  uint8_t                    *pTxBuffPtr;      /* Pointer to QSPI Tx transfer Buffer */
  __IO uint32_t              TxXferSize;       /* QSPI Tx Transfer size              */
  __IO uint32_t              TxXferCount;      /* QSPI Tx Transfer Counter           */
  uint8_t                    *pRxBuffPtr;      /* Pointer to QSPI Rx transfer Buffer */
  __IO uint32_t              RxXferSize;       /* QSPI Rx Transfer size              */
  __IO uint32_t              RxXferCount;      /* QSPI Rx Transfer Counter           */
  DMA_HandleTypeDef          *hdma;            /* QSPI Rx/Tx DMA Handle parameters   */
  __IO HAL_LockTypeDef       Lock;             /* Locking object                     */
  __IO HAL_QSPI_StateTypeDef State;            /* QSPI communication state           */
  __IO uint32_t              ErrorCode;        /* QSPI Error code                    */
  uint32_t                   Timeout;          /* Timeout for the QSPI memory access */
#if (USE_HAL_QSPI_REGISTER_CALLBACKS == 1)
  void (* ErrorCallback)        (struct __QSPI_HandleTypeDef *hqspi);
  void (* AbortCpltCallback)    (struct __QSPI_HandleTypeDef *hqspi);
  void (* FifoThresholdCallback)(struct __QSPI_HandleTypeDef *hqspi);
  void (* CmdCpltCallback)      (struct __QSPI_HandleTypeDef *hqspi);
  void (* RxCpltCallback)       (struct __QSPI_HandleTypeDef *hqspi);
  void (* TxCpltCallback)       (struct __QSPI_HandleTypeDef *hqspi);
  void (* RxHalfCpltCallback)   (struct __QSPI_HandleTypeDef *hqspi);
  void (* TxHalfCpltCallback)   (struct __QSPI_HandleTypeDef *hqspi);
  void (* StatusMatchCallback)  (struct __QSPI_HandleTypeDef *hqspi);
  void (* TimeOutCallback)      (struct __QSPI_HandleTypeDef *hqspi);

  void (* MspInitCallback)      (struct __QSPI_HandleTypeDef *hqspi);
  void (* MspDeInitCallback)    (struct __QSPI_HandleTypeDef *hqspi);
#endif
}QSPI_HandleTypeDef;

/**
  * @brief  QSPI Command structure definition
  */
typedef struct
{
  uint32_t Instruction;        /* Specifies the Instruction to be sent
                                  This parameter can be a value (8-bit) between 0x00 and 0xFF */
  uint32_t Address;            /* Specifies the Address to be sent (Size from 1 to 4 bytes according AddressSize)
                                  This parameter can be a value (32-bits) between 0x0 and 0xFFFFFFFF */
  uint32_t AlternateBytes;     /* Specifies the Alternate Bytes to be sent (Size from 1 to 4 bytes according AlternateBytesSize)
                                  This parameter can be a value (32-bits) between 0x0 and 0xFFFFFFFF */
  uint32_t AddressSize;        /* Specifies the Address Size
                                  This parameter can be a value of @ref QSPI_AddressSize */
  uint32_t AlternateBytesSize; /* Specifies the Alternate Bytes Size
                                  This parameter can be a value of @ref QSPI_AlternateBytesSize */
  uint32_t DummyCycles;        /* Specifies the Number of Dummy Cycles.
                                  This parameter can be a number between 0 and 31 */
  uint32_t InstructionMode;    /* Specifies the Instruction Mode
                                  This parameter can be a value of @ref QSPI_InstructionMode */
  uint32_t AddressMode;        /* Specifies the Address Mode
                                  This parameter can be a value of @ref QSPI_AddressMode */
  uint32_t AlternateByteMode;  /* Specifies the Alternate Bytes Mode
                                  This parameter can be a value of @ref QSPI_AlternateBytesMode */
  uint32_t DataMode;           /* Specifies the Data Mode (used for dummy cycles and data phases)
                                  This parameter can be a value of @ref QSPI_DataMode */
  uint32_t NbData;             /* Specifies the number of data to transfer. (This is the number of bytes)
                                  This parameter can be any value between 0 and 0xFFFFFFFF (0 means undefined length
                                  until end of memory)*/
  uint32_t DdrMode;            /* Specifies the double data rate mode for address, alternate byte and data phase
                                  This parameter can be a value of @ref QSPI_DdrMode */
  uint32_t DdrHoldHalfCycle;   /* Specifies if the DDR hold is enabled. When enabled it delays the data
                                  output by one half of system clock in DDR mode.
                                  This parameter can be a value of @ref QSPI_DdrHoldHalfCycle */
  uint32_t SIOOMode;           /* Specifies the send instruction only once mode
                                  This parameter can be a value of @ref QSPI_SIOOMode */
}QSPI_CommandTypeDef;

/**
  * @brief  QSPI Auto Polling mode configuration structure definition
  */
typedef struct
{
  uint32_t Match;              /* Specifies the value to be compared with the masked status register to get a match.
                                  This parameter can be any value between 0 and 0xFFFFFFFF */
  uint32_t Mask;               /* Specifies the mask to be applied to the status bytes received.
                                  This parameter can be any value between 0 and 0xFFFFFFFF */
  uint32_t Interval;           /* Specifies the number of clock cycles between two read during automatic polling phases.
                                  This parameter can be any value between 0 and 0xFFFF */
  uint32_t StatusBytesSize;    /* Specifies the size of the status bytes received.
                                  This parameter can be any value between 1 and 4 */
  uint32_t MatchMode;          /* Specifies the method used for determining a match.
                                  This parameter can be a value of @ref QSPI_MatchMode */
  uint32_t AutomaticStop;      /* Specifies if automatic polling is stopped after a match.
                                  This parameter can be a value of @ref QSPI_AutomaticStop */
}QSPI_AutoPollingTypeDef;

/**
  * @brief  QSPI Memory Mapped mode configuration structure definition
  */
typedef struct
{
  uint32_t TimeOutPeriod;      /* Specifies the number of clock to wait when the FIFO is full before to release the chip select.
                                  This parameter can be any value between 0 and 0xFFFF */
  uint32_t TimeOutActivation;  /* Specifies if the timeout counter is enabled to release the chip select.
                                  This parameter can be a value of @ref QSPI_TimeOutActivation */
}QSPI_MemoryMappedTypeDef;

#if (USE_HAL_QSPI_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL QSPI Callback ID enumeration definition
  */
typedef enum
{
  HAL_QSPI_ERROR_CB_ID          = 0x00U,  /*!< QSPI Error Callback ID            */
  HAL_QSPI_ABORT_CB_ID          = 0x01U,  /*!< QSPI Abort Callback ID            */
  HAL_QSPI_FIFO_THRESHOLD_CB_ID = 0x02U,  /*!< QSPI FIFO Threshold Callback ID   */
  HAL_QSPI_CMD_CPLT_CB_ID       = 0x03U,  /*!< QSPI Command Complete Callback ID */
  HAL_QSPI_RX_CPLT_CB_ID        = 0x04U,  /*!< QSPI Rx Complete Callback ID      */
  HAL_QSPI_TX_CPLT_CB_ID        = 0x05U,  /*!< QSPI Tx Complete Callback ID      */
  HAL_QSPI_RX_HALF_CPLT_CB_ID   = 0x06U,  /*!< QSPI Rx Half Complete Callback ID */
  HAL_QSPI_TX_HALF_CPLT_CB_ID   = 0x07U,  /*!< QSPI Tx Half Complete Callback ID */
  HAL_QSPI_STATUS_MATCH_CB_ID   = 0x08U,  /*!< QSPI Status Match Callback ID     */
  HAL_QSPI_TIMEOUT_CB_ID        = 0x09U,  /*!< QSPI Timeout Callback ID          */

  HAL_QSPI_MSP_INIT_CB_ID       = 0x0AU,  /*!< QSPI MspInit Callback ID          */
  HAL_QSPI_MSP_DEINIT_CB_ID     = 0x0B0   /*!< QSPI MspDeInit Callback ID        */
}HAL_QSPI_CallbackIDTypeDef;

/**
  * @brief  HAL QSPI Callback pointer definition
  */
typedef void (*pQSPI_CallbackTypeDef)(QSPI_HandleTypeDef *hqspi);
#endif
/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup QSPI_Exported_Constants QSPI Exported Constants
  * @{
  */

/** @defgroup QSPI_ErrorCode QSPI Error Code
  * @{
  */
#define HAL_QSPI_ERROR_NONE             0x00000000U /*!< No error                 */
#define HAL_QSPI_ERROR_TIMEOUT          0x00000001U /*!< Timeout error            */
#define HAL_QSPI_ERROR_TRANSFER         0x00000002U /*!< Transfer error           */
#define HAL_QSPI_ERROR_DMA              0x00000004U /*!< DMA transfer error       */
#define HAL_QSPI_ERROR_INVALID_PARAM    0x00000008U /*!< Invalid parameters error */
#if (USE_HAL_QSPI_REGISTER_CALLBACKS == 1)
#define HAL_QSPI_ERROR_INVALID_CALLBACK 0x00000010U /*!< Invalid callback error   */
#endif
/**
  * @}
  */

/** @defgroup QSPI_SampleShifting QSPI Sample Shifting
  * @{
  */
#define QSPI_SAMPLE_SHIFTING_NONE      0x00000000U                   /*!<No clock cycle shift to sample data*/
#define QSPI_SAMPLE_SHIFTING_HALFCYCLE ((uint32_t)QUADSPI_CR_SSHIFT) /*!<1/2 clock cycle shift to sample data*/
/**
  * @}
  */

/** @defgroup QSPI_ChipSelectHighTime QSPI ChipSelect High Time
  * @{
  */
#define QSPI_CS_HIGH_TIME_1_CYCLE      0x00000000U                                         /*!<nCS stay high for at least 1 clock cycle between commands*/
#define QSPI_CS_HIGH_TIME_2_CYCLE      ((uint32_t)QUADSPI_DCR_CSHT_0)                      /*!<nCS stay high for at least 2 clock cycles between commands*/
#define QSPI_CS_HIGH_TIME_3_CYCLE      ((uint32_t)QUADSPI_DCR_CSHT_1)                      /*!<nCS stay high for at least 3 clock cycles between commands*/
#define QSPI_CS_HIGH_TIME_4_CYCLE      ((uint32_t)QUADSPI_DCR_CSHT_0 | QUADSPI_DCR_CSHT_1) /*!<nCS stay high for at least 4 clock cycles between commands*/
#define QSPI_CS_HIGH_TIME_5_CYCLE      ((uint32_t)QUADSPI_DCR_CSHT_2)                      /*!<nCS stay high for at least 5 clock cycles between commands*/
#define QSPI_CS_HIGH_TIME_6_CYCLE      ((uint32_t)QUADSPI_DCR_CSHT_2 | QUADSPI_DCR_CSHT_0) /*!<nCS stay high for at least 6 clock cycles between commands*/
#define QSPI_CS_HIGH_TIME_7_CYCLE      ((uint32_t)QUADSPI_DCR_CSHT_2 | QUADSPI_DCR_CSHT_1) /*!<nCS stay high for at least 7 clock cycles between commands*/
#define QSPI_CS_HIGH_TIME_8_CYCLE      ((uint32_t)QUADSPI_DCR_CSHT)                        /*!<nCS stay high for at least 8 clock cycles between commands*/
/**
  * @}
  */

/** @defgroup QSPI_ClockMode QSPI Clock Mode
  * @{
  */
#define QSPI_CLOCK_MODE_0              0x00000000U                    /*!<Clk stays low while nCS is released*/
#define QSPI_CLOCK_MODE_3              ((uint32_t)QUADSPI_DCR_CKMODE) /*!<Clk goes high while nCS is released*/
/**
  * @}
  */

/** @defgroup QSPI_Flash_Select QSPI Flash Select
  * @{
  */
#define QSPI_FLASH_ID_1                0x00000000U                 /*!<FLASH 1 selected*/
#define QSPI_FLASH_ID_2                ((uint32_t)QUADSPI_CR_FSEL) /*!<FLASH 2 selected*/
/**
  * @}
  */

  /** @defgroup QSPI_DualFlash_Mode QSPI Dual Flash Mode
  * @{
  */
#define QSPI_DUALFLASH_ENABLE          ((uint32_t)QUADSPI_CR_DFM) /*!<Dual-flash mode enabled*/
#define QSPI_DUALFLASH_DISABLE         0x00000000U                /*!<Dual-flash mode disabled*/
/**
  * @}
  */

/** @defgroup QSPI_AddressSize QSPI Address Size
  * @{
  */
#define QSPI_ADDRESS_8_BITS            0x00000000U                      /*!<8-bit address*/
#define QSPI_ADDRESS_16_BITS           ((uint32_t)QUADSPI_CCR_ADSIZE_0) /*!<16-bit address*/
#define QSPI_ADDRESS_24_BITS           ((uint32_t)QUADSPI_CCR_ADSIZE_1) /*!<24-bit address*/
#define QSPI_ADDRESS_32_BITS           ((uint32_t)QUADSPI_CCR_ADSIZE)   /*!<32-bit address*/
/**
  * @}
  */

/** @defgroup QSPI_AlternateBytesSize QSPI Alternate Bytes Size
  * @{
  */
#define QSPI_ALTERNATE_BYTES_8_BITS    0x00000000U                      /*!<8-bit alternate bytes*/
#define QSPI_ALTERNATE_BYTES_16_BITS   ((uint32_t)QUADSPI_CCR_ABSIZE_0) /*!<16-bit alternate bytes*/
#define QSPI_ALTERNATE_BYTES_24_BITS   ((uint32_t)QUADSPI_CCR_ABSIZE_1) /*!<24-bit alternate bytes*/
#define QSPI_ALTERNATE_BYTES_32_BITS   ((uint32_t)QUADSPI_CCR_ABSIZE)   /*!<32-bit alternate bytes*/
/**
  * @}
  */

/** @defgroup QSPI_InstructionMode QSPI Instruction Mode
* @{
*/
#define QSPI_INSTRUCTION_NONE          0x00000000U                     /*!<No instruction*/
#define QSPI_INSTRUCTION_1_LINE        ((uint32_t)QUADSPI_CCR_IMODE_0) /*!<Instruction on a single line*/
#define QSPI_INSTRUCTION_2_LINES       ((uint32_t)QUADSPI_CCR_IMODE_1) /*!<Instruction on two lines*/
#define QSPI_INSTRUCTION_4_LINES       ((uint32_t)QUADSPI_CCR_IMODE)   /*!<Instruction on four lines*/
/**
  * @}
  */

/** @defgroup QSPI_AddressMode QSPI Address Mode
* @{
*/
#define QSPI_ADDRESS_NONE              0x00000000U                      /*!<No address*/
#define QSPI_ADDRESS_1_LINE            ((uint32_t)QUADSPI_CCR_ADMODE_0) /*!<Address on a single line*/
#define QSPI_ADDRESS_2_LINES           ((uint32_t)QUADSPI_CCR_ADMODE_1) /*!<Address on two lines*/
#define QSPI_ADDRESS_4_LINES           ((uint32_t)QUADSPI_CCR_ADMODE)   /*!<Address on four lines*/
/**
  * @}
  */

/** @defgroup QSPI_AlternateBytesMode QSPI Alternate Bytes Mode
* @{
*/
#define QSPI_ALTERNATE_BYTES_NONE      0x00000000U                      /*!<No alternate bytes*/
#define QSPI_ALTERNATE_BYTES_1_LINE    ((uint32_t)QUADSPI_CCR_ABMODE_0) /*!<Alternate bytes on a single line*/
#define QSPI_ALTERNATE_BYTES_2_LINES   ((uint32_t)QUADSPI_CCR_ABMODE_1) /*!<Alternate bytes on two lines*/
#define QSPI_ALTERNATE_BYTES_4_LINES   ((uint32_t)QUADSPI_CCR_ABMODE)   /*!<Alternate bytes on four lines*/
/**
  * @}
  */

/** @defgroup QSPI_DataMode QSPI Data Mode
  * @{
  */
#define QSPI_DATA_NONE                 0x00000000U                     /*!<No data*/
#define QSPI_DATA_1_LINE               ((uint32_t)QUADSPI_CCR_DMODE_0) /*!<Data on a single line*/
#define QSPI_DATA_2_LINES              ((uint32_t)QUADSPI_CCR_DMODE_1) /*!<Data on two lines*/
#define QSPI_DATA_4_LINES              ((uint32_t)QUADSPI_CCR_DMODE)   /*!<Data on four lines*/
/**
  * @}
  */

/** @defgroup QSPI_DdrMode QSPI DDR Mode
  * @{
  */
#define QSPI_DDR_MODE_DISABLE          0x00000000U                  /*!<Double data rate mode disabled*/
#define QSPI_DDR_MODE_ENABLE           ((uint32_t)QUADSPI_CCR_DDRM) /*!<Double data rate mode enabled*/
/**
  * @}
  */

/** @defgroup QSPI_DdrHoldHalfCycle QSPI DDR Data Output Delay
  * @{
  */
#define QSPI_DDR_HHC_ANALOG_DELAY      0x00000000U                  /*!<Delay the data output using analog delay in DDR mode*/
#define QSPI_DDR_HHC_HALF_CLK_DELAY    ((uint32_t)QUADSPI_CCR_DHHC) /*!<Delay the data output by one half of system clock in DDR mode*/
/**
  * @}
  */

/** @defgroup QSPI_SIOOMode QSPI Send Instruction Mode
  * @{
  */
#define QSPI_SIOO_INST_EVERY_CMD       0x00000000U                  /*!<Send instruction on every transaction*/
#define QSPI_SIOO_INST_ONLY_FIRST_CMD  ((uint32_t)QUADSPI_CCR_SIOO) /*!<Send instruction only for the first command*/
/**
  * @}
  */

/** @defgroup QSPI_MatchMode QSPI Match Mode
  * @{
  */
#define QSPI_MATCH_MODE_AND            0x00000000U                /*!<AND match mode between unmasked bits*/
#define QSPI_MATCH_MODE_OR             ((uint32_t)QUADSPI_CR_PMM) /*!<OR match mode between unmasked bits*/
/**
  * @}
  */

/** @defgroup QSPI_AutomaticStop QSPI Automatic Stop
  * @{
  */
#define QSPI_AUTOMATIC_STOP_DISABLE    0x00000000U                 /*!<AutoPolling stops only with abort or QSPI disabling*/
#define QSPI_AUTOMATIC_STOP_ENABLE     ((uint32_t)QUADSPI_CR_APMS) /*!<AutoPolling stops as soon as there is a match*/
/**
  * @}
  */

/** @defgroup QSPI_TimeOutActivation QSPI Timeout Activation
  * @{
  */
#define QSPI_TIMEOUT_COUNTER_DISABLE   0x00000000U                 /*!<Timeout counter disabled, nCS remains active*/
#define QSPI_TIMEOUT_COUNTER_ENABLE    ((uint32_t)QUADSPI_CR_TCEN) /*!<Timeout counter enabled, nCS released when timeout expires*/
/**
  * @}
  */

/** @defgroup QSPI_Flags QSPI Flags
  * @{
  */
#define QSPI_FLAG_BUSY                 QUADSPI_SR_BUSY /*!<Busy flag: operation is ongoing*/
#define QSPI_FLAG_TO                   QUADSPI_SR_TOF  /*!<Timeout flag: timeout occurs in memory-mapped mode*/
#define QSPI_FLAG_SM                   QUADSPI_SR_SMF  /*!<Status match flag: received data matches in autopolling mode*/
#define QSPI_FLAG_FT                   QUADSPI_SR_FTF  /*!<Fifo threshold flag: Fifo threshold reached or data left after read from memory is complete*/
#define QSPI_FLAG_TC                   QUADSPI_SR_TCF  /*!<Transfer complete flag: programmed number of data have been transferred or the transfer has been aborted*/
#define QSPI_FLAG_TE                   QUADSPI_SR_TEF  /*!<Transfer error flag: invalid address is being accessed*/
/**
  * @}
  */

/** @defgroup QSPI_Interrupts QSPI Interrupts
  * @{
  */
#define QSPI_IT_TO                     QUADSPI_CR_TOIE /*!<Interrupt on the timeout flag*/
#define QSPI_IT_SM                     QUADSPI_CR_SMIE /*!<Interrupt on the status match flag*/
#define QSPI_IT_FT                     QUADSPI_CR_FTIE /*!<Interrupt on the fifo threshold flag*/
#define QSPI_IT_TC                     QUADSPI_CR_TCIE /*!<Interrupt on the transfer complete flag*/
#define QSPI_IT_TE                     QUADSPI_CR_TEIE /*!<Interrupt on the transfer error flag*/
/**
  * @}
  */

/** @defgroup QSPI_Timeout_definition QSPI Timeout definition
  * @brief QSPI Timeout definition
  * @{
  */
#define HAL_QSPI_TIMEOUT_DEFAULT_VALUE 5000U /* 5 s */
/**
  * @}
  */

/**
  * @}
  */

/* Exported macros -----------------------------------------------------------*/
/** @defgroup QSPI_Exported_Macros QSPI Exported Macros
  * @{
  */
/** @brief Reset QSPI handle state.
  * @param  __HANDLE__ : QSPI handle.
  * @retval None
  */
#if (USE_HAL_QSPI_REGISTER_CALLBACKS == 1)
#define __HAL_QSPI_RESET_HANDLE_STATE(__HANDLE__)           do {                                              \
                                                                  (__HANDLE__)->State = HAL_QSPI_STATE_RESET; \
                                                                  (__HANDLE__)->MspInitCallback = NULL;       \
                                                                  (__HANDLE__)->MspDeInitCallback = NULL;     \
                                                               } while(0)
#else
#define __HAL_QSPI_RESET_HANDLE_STATE(__HANDLE__)           ((__HANDLE__)->State = HAL_QSPI_STATE_RESET)
#endif

/** @brief  Enable the QSPI peripheral.
  * @param  __HANDLE__ : specifies the QSPI Handle.
  * @retval None
  */
#define __HAL_QSPI_ENABLE(__HANDLE__)                       SET_BIT((__HANDLE__)->Instance->CR, QUADSPI_CR_EN)

/** @brief  Disable the QSPI peripheral.
  * @param  __HANDLE__ : specifies the QSPI Handle.
  * @retval None
  */
#define __HAL_QSPI_DISABLE(__HANDLE__)                      CLEAR_BIT((__HANDLE__)->Instance->CR, QUADSPI_CR_EN)

/** @brief  Enable the specified QSPI interrupt.
  * @param  __HANDLE__ : specifies the QSPI Handle.
  * @param  __INTERRUPT__ : specifies the QSPI interrupt source to enable.
  *          This parameter can be one of the following values:
  *            @arg QSPI_IT_TO: QSPI Timeout interrupt
  *            @arg QSPI_IT_SM: QSPI Status match interrupt
  *            @arg QSPI_IT_FT: QSPI FIFO threshold interrupt
  *            @arg QSPI_IT_TC: QSPI Transfer complete interrupt
  *            @arg QSPI_IT_TE: QSPI Transfer error interrupt
  * @retval None
  */
#define __HAL_QSPI_ENABLE_IT(__HANDLE__, __INTERRUPT__)     SET_BIT((__HANDLE__)->Instance->CR, (__INTERRUPT__))


/** @brief  Disable the specified QSPI interrupt.
  * @param  __HANDLE__ : specifies the QSPI Handle.
  * @param  __INTERRUPT__ : specifies the QSPI interrupt source to disable.
  *          This parameter can be one of the following values:
  *            @arg QSPI_IT_TO: QSPI Timeout interrupt
  *            @arg QSPI_IT_SM: QSPI Status match interrupt
  *            @arg QSPI_IT_FT: QSPI FIFO threshold interrupt
  *            @arg QSPI_IT_TC: QSPI Transfer complete interrupt
  *            @arg QSPI_IT_TE: QSPI Transfer error interrupt
  * @retval None
  */
#define __HAL_QSPI_DISABLE_IT(__HANDLE__, __INTERRUPT__)    CLEAR_BIT((__HANDLE__)->Instance->CR, (__INTERRUPT__))

/** @brief  Check whether the specified QSPI interrupt source is enabled or not.
  * @param  __HANDLE__ : specifies the QSPI Handle.
  * @param  __INTERRUPT__ : specifies the QSPI interrupt source to check.
  *          This parameter can be one of the following values:
  *            @arg QSPI_IT_TO: QSPI Timeout interrupt
  *            @arg QSPI_IT_SM: QSPI Status match interrupt
  *            @arg QSPI_IT_FT: QSPI FIFO threshold interrupt
  *            @arg QSPI_IT_TC: QSPI Transfer complete interrupt
  *            @arg QSPI_IT_TE: QSPI Transfer error interrupt
  * @retval The new state of __INTERRUPT__ (TRUE or FALSE).
  */
#define __HAL_QSPI_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) (READ_BIT((__HANDLE__)->Instance->CR, (__INTERRUPT__)) == (__INTERRUPT__))

/**
  * @brief  Check whether the selected QSPI flag is set or not.
  * @param  __HANDLE__ : specifies the QSPI Handle.
  * @param  __FLAG__ : specifies the QSPI flag to check.
  *          This parameter can be one of the following values:
  *            @arg QSPI_FLAG_BUSY: QSPI Busy flag
  *            @arg QSPI_FLAG_TO:   QSPI Timeout flag
  *            @arg QSPI_FLAG_SM:   QSPI Status match flag
  *            @arg QSPI_FLAG_FT:   QSPI FIFO threshold flag
  *            @arg QSPI_FLAG_TC:   QSPI Transfer complete flag
  *            @arg QSPI_FLAG_TE:   QSPI Transfer error flag
  * @retval None
  */
#define __HAL_QSPI_GET_FLAG(__HANDLE__, __FLAG__)           ((READ_BIT((__HANDLE__)->Instance->SR, (__FLAG__)) != 0U) ? SET : RESET)

/** @brief  Clears the specified QSPI's flag status.
  * @param  __HANDLE__ : specifies the QSPI Handle.
  * @param  __FLAG__ : specifies the QSPI clear register flag that needs to be set
  *          This parameter can be one of the following values:
  *            @arg QSPI_FLAG_TO: QSPI Timeout flag
  *            @arg QSPI_FLAG_SM: QSPI Status match flag
  *            @arg QSPI_FLAG_TC: QSPI Transfer complete flag
  *            @arg QSPI_FLAG_TE: QSPI Transfer error flag
  * @retval None
  */
#define __HAL_QSPI_CLEAR_FLAG(__HANDLE__, __FLAG__)         WRITE_REG((__HANDLE__)->Instance->FCR, (__FLAG__))
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup QSPI_Exported_Functions
  * @{
  */

/** @addtogroup QSPI_Exported_Functions_Group1
  * @{
  */
/* Initialization/de-initialization functions  ********************************/
HAL_StatusTypeDef     HAL_QSPI_Init     (QSPI_HandleTypeDef *hqspi);
HAL_StatusTypeDef     HAL_QSPI_DeInit   (QSPI_HandleTypeDef *hqspi);
void                  HAL_QSPI_MspInit  (QSPI_HandleTypeDef *hqspi);
void                  HAL_QSPI_MspDeInit(QSPI_HandleTypeDef *hqspi);
/**
  * @}
  */

/** @addtogroup QSPI_Exported_Functions_Group2
  * @{
  */
/* IO operation functions *****************************************************/
/* QSPI IRQ handler method */
void                  HAL_QSPI_IRQHandler(QSPI_HandleTypeDef *hqspi);

/* QSPI indirect mode */
HAL_StatusTypeDef     HAL_QSPI_Command      (QSPI_HandleTypeDef *hqspi, QSPI_CommandTypeDef *cmd, uint32_t Timeout);
HAL_StatusTypeDef     HAL_QSPI_Transmit     (QSPI_HandleTypeDef *hqspi, uint8_t *pData, uint32_t Timeout);
HAL_StatusTypeDef     HAL_QSPI_Receive      (QSPI_HandleTypeDef *hqspi, uint8_t *pData, uint32_t Timeout);
HAL_StatusTypeDef     HAL_QSPI_Command_IT   (QSPI_HandleTypeDef *hqspi, QSPI_CommandTypeDef *cmd);
HAL_StatusTypeDef     HAL_QSPI_Transmit_IT  (QSPI_HandleTypeDef *hqspi, uint8_t *pData);
HAL_StatusTypeDef     HAL_QSPI_Receive_IT   (QSPI_HandleTypeDef *hqspi, uint8_t *pData);
HAL_StatusTypeDef     HAL_QSPI_Transmit_DMA (QSPI_HandleTypeDef *hqspi, uint8_t *pData);
HAL_StatusTypeDef     HAL_QSPI_Receive_DMA  (QSPI_HandleTypeDef *hqspi, uint8_t *pData);

/* QSPI status flag polling mode */
HAL_StatusTypeDef     HAL_QSPI_AutoPolling   (QSPI_HandleTypeDef *hqspi, QSPI_CommandTypeDef *cmd, QSPI_AutoPollingTypeDef *cfg, uint32_t Timeout);
HAL_StatusTypeDef     HAL_QSPI_AutoPolling_IT(QSPI_HandleTypeDef *hqspi, QSPI_CommandTypeDef *cmd, QSPI_AutoPollingTypeDef *cfg);

/* QSPI memory-mapped mode */
HAL_StatusTypeDef     HAL_QSPI_MemoryMapped(QSPI_HandleTypeDef *hqspi, QSPI_CommandTypeDef *cmd, QSPI_MemoryMappedTypeDef *cfg);

/* Callback functions in non-blocking modes ***********************************/
void                  HAL_QSPI_ErrorCallback        (QSPI_HandleTypeDef *hqspi);
void                  HAL_QSPI_AbortCpltCallback    (QSPI_HandleTypeDef *hqspi);
void                  HAL_QSPI_FifoThresholdCallback(QSPI_HandleTypeDef *hqspi);

/* QSPI indirect mode */
void                  HAL_QSPI_CmdCpltCallback      (QSPI_HandleTypeDef *hqspi);
void                  HAL_QSPI_RxCpltCallback       (QSPI_HandleTypeDef *hqspi);
void                  HAL_QSPI_TxCpltCallback       (QSPI_HandleTypeDef *hqspi);
void                  HAL_QSPI_RxHalfCpltCallback   (QSPI_HandleTypeDef *hqspi);
void                  HAL_QSPI_TxHalfCpltCallback   (QSPI_HandleTypeDef *hqspi);

/* QSPI status flag polling mode */
void                  HAL_QSPI_StatusMatchCallback  (QSPI_HandleTypeDef *hqspi);

/* QSPI memory-mapped mode */
void                  HAL_QSPI_TimeOutCallback      (QSPI_HandleTypeDef *hqspi);

#if (USE_HAL_QSPI_REGISTER_CALLBACKS == 1)
/* QSPI callback registering/unregistering */
HAL_StatusTypeDef     HAL_QSPI_RegisterCallback     (QSPI_HandleTypeDef *hqspi, HAL_QSPI_CallbackIDTypeDef CallbackId, pQSPI_CallbackTypeDef pCallback);
HAL_StatusTypeDef     HAL_QSPI_UnRegisterCallback   (QSPI_HandleTypeDef *hqspi, HAL_QSPI_CallbackIDTypeDef CallbackId);
#endif
/**
  * @}
  */

/** @addtogroup QSPI_Exported_Functions_Group3
  * @{
  */
/* Peripheral Control and State functions  ************************************/
HAL_QSPI_StateTypeDef HAL_QSPI_GetState        (QSPI_HandleTypeDef *hqspi);
uint32_t              HAL_QSPI_GetError        (QSPI_HandleTypeDef *hqspi);
HAL_StatusTypeDef     HAL_QSPI_Abort           (QSPI_HandleTypeDef *hqspi);
HAL_StatusTypeDef     HAL_QSPI_Abort_IT        (QSPI_HandleTypeDef *hqspi);
void                  HAL_QSPI_SetTimeout      (QSPI_HandleTypeDef *hqspi, uint32_t Timeout);
HAL_StatusTypeDef     HAL_QSPI_SetFifoThreshold(QSPI_HandleTypeDef *hqspi, uint32_t Threshold);
uint32_t              HAL_QSPI_GetFifoThreshold(QSPI_HandleTypeDef *hqspi);
HAL_StatusTypeDef     HAL_QSPI_SetFlashID      (QSPI_HandleTypeDef *hqspi, uint32_t FlashID);
/**
  * @}
  */

/**
  * @}
  */
/* End of exported functions -------------------------------------------------*/

/* Private macros ------------------------------------------------------------*/
/** @defgroup QSPI_Private_Macros QSPI Private Macros
  * @{
  */
#define IS_QSPI_CLOCK_PRESCALER(PRESCALER) ((PRESCALER) <= 0xFFU)

#define IS_QSPI_FIFO_THRESHOLD(THR)        (((THR) > 0U) && ((THR) <= 32U))

#define IS_QSPI_SSHIFT(SSHIFT)             (((SSHIFT) == QSPI_SAMPLE_SHIFTING_NONE) || \
                                            ((SSHIFT) == QSPI_SAMPLE_SHIFTING_HALFCYCLE))

#define IS_QSPI_FLASH_SIZE(FSIZE)          (((FSIZE) <= 31U))

#define IS_QSPI_CS_HIGH_TIME(CSHTIME)      (((CSHTIME) == QSPI_CS_HIGH_TIME_1_CYCLE) || \
                                            ((CSHTIME) == QSPI_CS_HIGH_TIME_2_CYCLE) || \
                                            ((CSHTIME) == QSPI_CS_HIGH_TIME_3_CYCLE) || \
                                            ((CSHTIME) == QSPI_CS_HIGH_TIME_4_CYCLE) || \
                                            ((CSHTIME) == QSPI_CS_HIGH_TIME_5_CYCLE) || \
                                            ((CSHTIME) == QSPI_CS_HIGH_TIME_6_CYCLE) || \
                                            ((CSHTIME) == QSPI_CS_HIGH_TIME_7_CYCLE) || \
                                            ((CSHTIME) == QSPI_CS_HIGH_TIME_8_CYCLE))

#define IS_QSPI_CLOCK_MODE(CLKMODE)        (((CLKMODE) == QSPI_CLOCK_MODE_0) || \
                                            ((CLKMODE) == QSPI_CLOCK_MODE_3))

#define IS_QSPI_FLASH_ID(FLASH_ID)         (((FLASH_ID) == QSPI_FLASH_ID_1) || \
                                            ((FLASH_ID) == QSPI_FLASH_ID_2))

#define IS_QSPI_DUAL_FLASH_MODE(MODE)      (((MODE) == QSPI_DUALFLASH_ENABLE) || \
                                            ((MODE) == QSPI_DUALFLASH_DISABLE))

#define IS_QSPI_INSTRUCTION(INSTRUCTION)   ((INSTRUCTION) <= 0xFFU)

#define IS_QSPI_ADDRESS_SIZE(ADDR_SIZE)    (((ADDR_SIZE) == QSPI_ADDRESS_8_BITS)  || \
                                            ((ADDR_SIZE) == QSPI_ADDRESS_16_BITS) || \
                                            ((ADDR_SIZE) == QSPI_ADDRESS_24_BITS) || \
                                            ((ADDR_SIZE) == QSPI_ADDRESS_32_BITS))

#define IS_QSPI_ALTERNATE_BYTES_SIZE(SIZE) (((SIZE) == QSPI_ALTERNATE_BYTES_8_BITS)  || \
                                            ((SIZE) == QSPI_ALTERNATE_BYTES_16_BITS) || \
                                            ((SIZE) == QSPI_ALTERNATE_BYTES_24_BITS) || \
                                            ((SIZE) == QSPI_ALTERNATE_BYTES_32_BITS))

#define IS_QSPI_DUMMY_CYCLES(DCY)          ((DCY) <= 31U)

#define IS_QSPI_INSTRUCTION_MODE(MODE)     (((MODE) == QSPI_INSTRUCTION_NONE)    || \
                                            ((MODE) == QSPI_INSTRUCTION_1_LINE)  || \
                                            ((MODE) == QSPI_INSTRUCTION_2_LINES) || \
                                            ((MODE) == QSPI_INSTRUCTION_4_LINES))

#define IS_QSPI_ADDRESS_MODE(MODE)         (((MODE) == QSPI_ADDRESS_NONE)    || \
                                            ((MODE) == QSPI_ADDRESS_1_LINE)  || \
                                            ((MODE) == QSPI_ADDRESS_2_LINES) || \
                                            ((MODE) == QSPI_ADDRESS_4_LINES))

#define IS_QSPI_ALTERNATE_BYTES_MODE(MODE) (((MODE) == QSPI_ALTERNATE_BYTES_NONE)    || \
                                            ((MODE) == QSPI_ALTERNATE_BYTES_1_LINE)  || \
                                            ((MODE) == QSPI_ALTERNATE_BYTES_2_LINES) || \
                                            ((MODE) == QSPI_ALTERNATE_BYTES_4_LINES))

#define IS_QSPI_DATA_MODE(MODE)            (((MODE) == QSPI_DATA_NONE)    || \
                                            ((MODE) == QSPI_DATA_1_LINE)  || \
                                            ((MODE) == QSPI_DATA_2_LINES) || \
                                            ((MODE) == QSPI_DATA_4_LINES))

#define IS_QSPI_DDR_MODE(DDR_MODE)         (((DDR_MODE) == QSPI_DDR_MODE_DISABLE) || \
                                            ((DDR_MODE) == QSPI_DDR_MODE_ENABLE))

#define IS_QSPI_DDR_HHC(DDR_HHC)           (((DDR_HHC) == QSPI_DDR_HHC_ANALOG_DELAY) || \
                                            ((DDR_HHC) == QSPI_DDR_HHC_HALF_CLK_DELAY))

#define IS_QSPI_SIOO_MODE(SIOO_MODE)       (((SIOO_MODE) == QSPI_SIOO_INST_EVERY_CMD) || \
                                            ((SIOO_MODE) == QSPI_SIOO_INST_ONLY_FIRST_CMD))

#define IS_QSPI_INTERVAL(INTERVAL)         ((INTERVAL) <= QUADSPI_PIR_INTERVAL)

#define IS_QSPI_STATUS_BYTES_SIZE(SIZE)    (((SIZE) >= 1U) && ((SIZE) <= 4U))

#define IS_QSPI_MATCH_MODE(MODE)           (((MODE) == QSPI_MATCH_MODE_AND) || \
                                            ((MODE) == QSPI_MATCH_MODE_OR))

#define IS_QSPI_AUTOMATIC_STOP(APMS)       (((APMS) == QSPI_AUTOMATIC_STOP_DISABLE) || \
                                            ((APMS) == QSPI_AUTOMATIC_STOP_ENABLE))

#define IS_QSPI_TIMEOUT_ACTIVATION(TCEN)   (((TCEN) == QSPI_TIMEOUT_COUNTER_DISABLE) || \
                                            ((TCEN) == QSPI_TIMEOUT_COUNTER_ENABLE))

#define IS_QSPI_TIMEOUT_PERIOD(PERIOD)     ((PERIOD) <= 0xFFFFU)
/**
* @}
*/
/* End of private macros -----------------------------------------------------*/

/**
  * @}
  */

/**
  * @}
  */

#endif /* defined(QUADSPI) || defined(QUADSPI1) || defined(QUADSPI2) */

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_HAL_QSPI_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
