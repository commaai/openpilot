/**
  ******************************************************************************
  * @file    stm32f4xx_ll_sdmmc.h
  * @author  MCD Application Team
  * @brief   Header file of SDMMC HAL module.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2017 STMicroelectronics.
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
#ifndef STM32F4xx_LL_SDMMC_H
#define STM32F4xx_LL_SDMMC_H

#ifdef __cplusplus
 extern "C" {
#endif

#if defined(SDIO)

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_Driver
  * @{
  */

/** @addtogroup SDMMC_LL
  * @{
  */ 

/* Exported types ------------------------------------------------------------*/ 
/** @defgroup SDMMC_LL_Exported_Types SDMMC_LL Exported Types
  * @{
  */
  
/** 
  * @brief  SDMMC Configuration Structure definition  
  */
typedef struct
{
  uint32_t ClockEdge;            /*!< Specifies the clock transition on which the bit capture is made.
                                      This parameter can be a value of @ref SDMMC_LL_Clock_Edge                 */

  uint32_t ClockBypass;          /*!< Specifies whether the SDMMC Clock divider bypass is
                                      enabled or disabled.
                                      This parameter can be a value of @ref SDMMC_LL_Clock_Bypass               */

  uint32_t ClockPowerSave;       /*!< Specifies whether SDMMC Clock output is enabled or
                                      disabled when the bus is idle.
                                      This parameter can be a value of @ref SDMMC_LL_Clock_Power_Save           */

  uint32_t BusWide;              /*!< Specifies the SDMMC bus width.
                                      This parameter can be a value of @ref SDMMC_LL_Bus_Wide                   */

  uint32_t HardwareFlowControl;  /*!< Specifies whether the SDMMC hardware flow control is enabled or disabled.
                                      This parameter can be a value of @ref SDMMC_LL_Hardware_Flow_Control      */

  uint32_t ClockDiv;             /*!< Specifies the clock frequency of the SDMMC controller.
                                      This parameter can be a value between Min_Data = 0 and Max_Data = 255 */  
  
}SDIO_InitTypeDef;
  

/** 
  * @brief  SDMMC Command Control structure 
  */
typedef struct                                                                                            
{
  uint32_t Argument;            /*!< Specifies the SDMMC command argument which is sent
                                     to a card as part of a command message. If a command
                                     contains an argument, it must be loaded into this register
                                     before writing the command to the command register.              */

  uint32_t CmdIndex;            /*!< Specifies the SDMMC command index. It must be Min_Data = 0 and 
                                     Max_Data = 64                                                    */

  uint32_t Response;            /*!< Specifies the SDMMC response type.
                                     This parameter can be a value of @ref SDMMC_LL_Response_Type         */

  uint32_t WaitForInterrupt;    /*!< Specifies whether SDMMC wait for interrupt request is 
                                     enabled or disabled.
                                     This parameter can be a value of @ref SDMMC_LL_Wait_Interrupt_State  */

  uint32_t CPSM;                /*!< Specifies whether SDMMC Command path state machine (CPSM)
                                     is enabled or disabled.
                                     This parameter can be a value of @ref SDMMC_LL_CPSM_State            */
}SDIO_CmdInitTypeDef;


/** 
  * @brief  SDMMC Data Control structure 
  */
typedef struct
{
  uint32_t DataTimeOut;         /*!< Specifies the data timeout period in card bus clock periods.  */

  uint32_t DataLength;          /*!< Specifies the number of data bytes to be transferred.         */
 
  uint32_t DataBlockSize;       /*!< Specifies the data block size for block transfer.
                                     This parameter can be a value of @ref SDMMC_LL_Data_Block_Size    */
 
  uint32_t TransferDir;         /*!< Specifies the data transfer direction, whether the transfer
                                     is a read or write.
                                     This parameter can be a value of @ref SDMMC_LL_Transfer_Direction */
 
  uint32_t TransferMode;        /*!< Specifies whether data transfer is in stream or block mode.
                                     This parameter can be a value of @ref SDMMC_LL_Transfer_Type      */
 
  uint32_t DPSM;                /*!< Specifies whether SDMMC Data path state machine (DPSM)
                                     is enabled or disabled.
                                     This parameter can be a value of @ref SDMMC_LL_DPSM_State         */
}SDIO_DataInitTypeDef;

/**
  * @}
  */
  
/* Exported constants --------------------------------------------------------*/
/** @defgroup SDMMC_LL_Exported_Constants SDMMC_LL Exported Constants
  * @{
  */
#define SDMMC_ERROR_NONE                                0x00000000U    /*!< No error                                                      */
#define SDMMC_ERROR_CMD_CRC_FAIL                        0x00000001U    /*!< Command response received (but CRC check failed)              */
#define SDMMC_ERROR_DATA_CRC_FAIL                       0x00000002U    /*!< Data block sent/received (CRC check failed)                   */
#define SDMMC_ERROR_CMD_RSP_TIMEOUT                     0x00000004U    /*!< Command response timeout                                      */
#define SDMMC_ERROR_DATA_TIMEOUT                        0x00000008U    /*!< Data timeout                                                  */
#define SDMMC_ERROR_TX_UNDERRUN                         0x00000010U    /*!< Transmit FIFO underrun                                        */
#define SDMMC_ERROR_RX_OVERRUN                          0x00000020U    /*!< Receive FIFO overrun                                          */
#define SDMMC_ERROR_ADDR_MISALIGNED                     0x00000040U    /*!< Misaligned address                                            */
#define SDMMC_ERROR_BLOCK_LEN_ERR                       0x00000080U    /*!< Transferred block length is not allowed for the card or the
                                                                            number of transferred bytes does not match the block length   */
#define SDMMC_ERROR_ERASE_SEQ_ERR                       0x00000100U    /*!< An error in the sequence of erase command occurs              */
#define SDMMC_ERROR_BAD_ERASE_PARAM                     0x00000200U    /*!< An invalid selection for erase groups                         */
#define SDMMC_ERROR_WRITE_PROT_VIOLATION                0x00000400U    /*!< Attempt to program a write protect block                      */
#define SDMMC_ERROR_LOCK_UNLOCK_FAILED                  0x00000800U    /*!< Sequence or password error has been detected in unlock
                                                                            command or if there was an attempt to access a locked card    */
#define SDMMC_ERROR_COM_CRC_FAILED                      0x00001000U    /*!< CRC check of the previous command failed                      */
#define SDMMC_ERROR_ILLEGAL_CMD                         0x00002000U    /*!< Command is not legal for the card state                       */
#define SDMMC_ERROR_CARD_ECC_FAILED                     0x00004000U    /*!< Card internal ECC was applied but failed to correct the data  */
#define SDMMC_ERROR_CC_ERR                              0x00008000U    /*!< Internal card controller error                                */
#define SDMMC_ERROR_GENERAL_UNKNOWN_ERR                 0x00010000U    /*!< General or unknown error                                      */
#define SDMMC_ERROR_STREAM_READ_UNDERRUN                0x00020000U    /*!< The card could not sustain data reading in stream rmode       */
#define SDMMC_ERROR_STREAM_WRITE_OVERRUN                0x00040000U    /*!< The card could not sustain data programming in stream mode    */
#define SDMMC_ERROR_CID_CSD_OVERWRITE                   0x00080000U    /*!< CID/CSD overwrite error                                       */
#define SDMMC_ERROR_WP_ERASE_SKIP                       0x00100000U    /*!< Only partial address space was erased                         */
#define SDMMC_ERROR_CARD_ECC_DISABLED                   0x00200000U    /*!< Command has been executed without using internal ECC          */
#define SDMMC_ERROR_ERASE_RESET                         0x00400000U    /*!< Erase sequence was cleared before executing because an out
                                                                            of erase sequence command was received                        */
#define SDMMC_ERROR_AKE_SEQ_ERR                         0x00800000U    /*!< Error in sequence of authentication                           */
#define SDMMC_ERROR_INVALID_VOLTRANGE                   0x01000000U    /*!< Error in case of invalid voltage range                        */
#define SDMMC_ERROR_ADDR_OUT_OF_RANGE                   0x02000000U    /*!< Error when addressed block is out of range                    */
#define SDMMC_ERROR_REQUEST_NOT_APPLICABLE              0x04000000U    /*!< Error when command request is not applicable                  */
#define SDMMC_ERROR_INVALID_PARAMETER                   0x08000000U    /*!< the used parameter is not valid                               */
#define SDMMC_ERROR_UNSUPPORTED_FEATURE                 0x10000000U    /*!< Error when feature is not insupported                         */
#define SDMMC_ERROR_BUSY                                0x20000000U    /*!< Error when transfer process is busy                           */
#define SDMMC_ERROR_DMA                                 0x40000000U    /*!< Error while DMA transfer                                      */
#define SDMMC_ERROR_TIMEOUT                             0x80000000U    /*!< Timeout error                                                 */

/** 
  * @brief SDMMC Commands Index 
  */
#define SDMMC_CMD_GO_IDLE_STATE                                 0U    /*!< Resets the SD memory card.                                                               */
#define SDMMC_CMD_SEND_OP_COND                                  1U    /*!< Sends host capacity support information and activates the card's initialization process. */
#define SDMMC_CMD_ALL_SEND_CID                                  2U    /*!< Asks any card connected to the host to send the CID numbers on the CMD line.             */
#define SDMMC_CMD_SET_REL_ADDR                                  3U    /*!< Asks the card to publish a new relative address (RCA).                                   */
#define SDMMC_CMD_SET_DSR                                       4U    /*!< Programs the DSR of all cards.                                                           */
#define SDMMC_CMD_SDMMC_SEN_OP_COND                             5U    /*!< Sends host capacity support information (HCS) and asks the accessed card to send its
                                                                           operating condition register (OCR) content in the response on the CMD line.              */
#define SDMMC_CMD_HS_SWITCH                                     6U    /*!< Checks switchable function (mode 0) and switch card function (mode 1).                   */
#define SDMMC_CMD_SEL_DESEL_CARD                                7U    /*!< Selects the card by its own relative address and gets deselected by any other address    */
#define SDMMC_CMD_HS_SEND_EXT_CSD                               8U    /*!< Sends SD Memory Card interface condition, which includes host supply voltage information
                                                                           and asks the card whether card supports voltage.                                         */
#define SDMMC_CMD_SEND_CSD                                      9U    /*!< Addressed card sends its card specific data (CSD) on the CMD line.                       */
#define SDMMC_CMD_SEND_CID                                      10U   /*!< Addressed card sends its card identification (CID) on the CMD line.                      */
#define SDMMC_CMD_READ_DAT_UNTIL_STOP                           11U   /*!< SD card doesn't support it.                                                              */
#define SDMMC_CMD_STOP_TRANSMISSION                             12U   /*!< Forces the card to stop transmission.                                                    */
#define SDMMC_CMD_SEND_STATUS                                   13U   /*!< Addressed card sends its status register.                                                */
#define SDMMC_CMD_HS_BUSTEST_READ                               14U   /*!< Reserved                                                                                 */
#define SDMMC_CMD_GO_INACTIVE_STATE                             15U   /*!< Sends an addressed card into the inactive state.                                         */
#define SDMMC_CMD_SET_BLOCKLEN                                  16U   /*!< Sets the block length (in bytes for SDSC) for all following block commands
                                                                           (read, write, lock). Default block length is fixed to 512 Bytes. Not effective 
                                                                           for SDHS and SDXC.                                                                       */
#define SDMMC_CMD_READ_SINGLE_BLOCK                             17U   /*!< Reads single block of size selected by SET_BLOCKLEN in case of SDSC, and a block of
                                                                           fixed 512 bytes in case of SDHC and SDXC.                                                */
#define SDMMC_CMD_READ_MULT_BLOCK                               18U   /*!< Continuously transfers data blocks from card to host until interrupted by
                                                                           STOP_TRANSMISSION command.                                                               */
#define SDMMC_CMD_HS_BUSTEST_WRITE                              19U   /*!< 64 bytes tuning pattern is sent for SDR50 and SDR104.                                    */
#define SDMMC_CMD_WRITE_DAT_UNTIL_STOP                          20U   /*!< Speed class control command.                                                             */
#define SDMMC_CMD_SET_BLOCK_COUNT                               23U   /*!< Specify block count for CMD18 and CMD25.                                                 */
#define SDMMC_CMD_WRITE_SINGLE_BLOCK                            24U   /*!< Writes single block of size selected by SET_BLOCKLEN in case of SDSC, and a block of
                                                                           fixed 512 bytes in case of SDHC and SDXC.                                                */
#define SDMMC_CMD_WRITE_MULT_BLOCK                              25U   /*!< Continuously writes blocks of data until a STOP_TRANSMISSION follows.                    */
#define SDMMC_CMD_PROG_CID                                      26U   /*!< Reserved for manufacturers.                                                              */
#define SDMMC_CMD_PROG_CSD                                      27U   /*!< Programming of the programmable bits of the CSD.                                         */
#define SDMMC_CMD_SET_WRITE_PROT                                28U   /*!< Sets the write protection bit of the addressed group.                                    */
#define SDMMC_CMD_CLR_WRITE_PROT                                29U   /*!< Clears the write protection bit of the addressed group.                                  */
#define SDMMC_CMD_SEND_WRITE_PROT                               30U   /*!< Asks the card to send the status of the write protection bits.                           */
#define SDMMC_CMD_SD_ERASE_GRP_START                            32U   /*!< Sets the address of the first write block to be erased. (For SD card only).              */
#define SDMMC_CMD_SD_ERASE_GRP_END                              33U   /*!< Sets the address of the last write block of the continuous range to be erased.           */
#define SDMMC_CMD_ERASE_GRP_START                               35U   /*!< Sets the address of the first write block to be erased. Reserved for each command
                                                                           system set by switch function command (CMD6).                                            */
#define SDMMC_CMD_ERASE_GRP_END                                 36U   /*!< Sets the address of the last write block of the continuous range to be erased.
                                                                           Reserved for each command system set by switch function command (CMD6).                  */
#define SDMMC_CMD_ERASE                                         38U   /*!< Reserved for SD security applications.                                                   */
#define SDMMC_CMD_FAST_IO                                       39U   /*!< SD card doesn't support it (Reserved).                                                   */
#define SDMMC_CMD_GO_IRQ_STATE                                  40U   /*!< SD card doesn't support it (Reserved).                                                   */
#define SDMMC_CMD_LOCK_UNLOCK                                   42U   /*!< Sets/resets the password or lock/unlock the card. The size of the data block is set by
                                                                           the SET_BLOCK_LEN command.                                                               */
#define SDMMC_CMD_APP_CMD                                       55U   /*!< Indicates to the card that the next command is an application specific command rather
                                                                           than a standard command.                                                                 */
#define SDMMC_CMD_GEN_CMD                                       56U   /*!< Used either to transfer a data block to the card or to get a data block from the card
                                                                           for general purpose/application specific commands.                                       */
#define SDMMC_CMD_NO_CMD                                        64U   /*!< No command                                                                               */

/** 
  * @brief Following commands are SD Card Specific commands.
  *        SDMMC_APP_CMD should be sent before sending these commands. 
  */
#define SDMMC_CMD_APP_SD_SET_BUSWIDTH                           6U    /*!< (ACMD6) Defines the data bus width to be used for data transfer. The allowed data bus
                                                                            widths are given in SCR register.                                                       */
#define SDMMC_CMD_SD_APP_STATUS                                 13U   /*!< (ACMD13) Sends the SD status.                                                            */
#define SDMMC_CMD_SD_APP_SEND_NUM_WRITE_BLOCKS                  22U   /*!< (ACMD22) Sends the number of the written (without errors) write blocks. Responds with
                                                                           32bit+CRC data block.                                                                    */
#define SDMMC_CMD_SD_APP_OP_COND                                41U   /*!< (ACMD41) Sends host capacity support information (HCS) and asks the accessed card to
                                                                           send its operating condition register (OCR) content in the response on the CMD line.     */
#define SDMMC_CMD_SD_APP_SET_CLR_CARD_DETECT                    42U   /*!< (ACMD42) Connect/Disconnect the 50 KOhm pull-up resistor on CD/DAT3 (pin 1) of the card  */
#define SDMMC_CMD_SD_APP_SEND_SCR                               51U   /*!< Reads the SD Configuration Register (SCR).                                               */
#define SDMMC_CMD_SDMMC_RW_DIRECT                               52U   /*!< For SD I/O card only, reserved for security specification.                               */
#define SDMMC_CMD_SDMMC_RW_EXTENDED                             53U   /*!< For SD I/O card only, reserved for security specification.                               */

/** 
  * @brief Following commands are SD Card Specific security commands.
  *        SDMMC_CMD_APP_CMD should be sent before sending these commands. 
  */
#define SDMMC_CMD_SD_APP_GET_MKB                                43U
#define SDMMC_CMD_SD_APP_GET_MID                                44U
#define SDMMC_CMD_SD_APP_SET_CER_RN1                            45U
#define SDMMC_CMD_SD_APP_GET_CER_RN2                            46U
#define SDMMC_CMD_SD_APP_SET_CER_RES2                           47U
#define SDMMC_CMD_SD_APP_GET_CER_RES1                           48U
#define SDMMC_CMD_SD_APP_SECURE_READ_MULTIPLE_BLOCK             18U
#define SDMMC_CMD_SD_APP_SECURE_WRITE_MULTIPLE_BLOCK            25U
#define SDMMC_CMD_SD_APP_SECURE_ERASE                           38U
#define SDMMC_CMD_SD_APP_CHANGE_SECURE_AREA                     49U
#define SDMMC_CMD_SD_APP_SECURE_WRITE_MKB                       48U

/** 
  * @brief  Masks for errors Card Status R1 (OCR Register) 
  */
#define SDMMC_OCR_ADDR_OUT_OF_RANGE                   0x80000000U
#define SDMMC_OCR_ADDR_MISALIGNED                     0x40000000U
#define SDMMC_OCR_BLOCK_LEN_ERR                       0x20000000U
#define SDMMC_OCR_ERASE_SEQ_ERR                       0x10000000U
#define SDMMC_OCR_BAD_ERASE_PARAM                     0x08000000U
#define SDMMC_OCR_WRITE_PROT_VIOLATION                0x04000000U
#define SDMMC_OCR_LOCK_UNLOCK_FAILED                  0x01000000U
#define SDMMC_OCR_COM_CRC_FAILED                      0x00800000U
#define SDMMC_OCR_ILLEGAL_CMD                         0x00400000U
#define SDMMC_OCR_CARD_ECC_FAILED                     0x00200000U
#define SDMMC_OCR_CC_ERROR                            0x00100000U
#define SDMMC_OCR_GENERAL_UNKNOWN_ERROR               0x00080000U
#define SDMMC_OCR_STREAM_READ_UNDERRUN                0x00040000U
#define SDMMC_OCR_STREAM_WRITE_OVERRUN                0x00020000U
#define SDMMC_OCR_CID_CSD_OVERWRITE                   0x00010000U
#define SDMMC_OCR_WP_ERASE_SKIP                       0x00008000U
#define SDMMC_OCR_CARD_ECC_DISABLED                   0x00004000U
#define SDMMC_OCR_ERASE_RESET                         0x00002000U
#define SDMMC_OCR_AKE_SEQ_ERROR                       0x00000008U
#define SDMMC_OCR_ERRORBITS                           0xFDFFE008U

/** 
  * @brief  Masks for R6 Response 
  */
#define SDMMC_R6_GENERAL_UNKNOWN_ERROR                0x00002000U
#define SDMMC_R6_ILLEGAL_CMD                          0x00004000U
#define SDMMC_R6_COM_CRC_FAILED                       0x00008000U

#define SDMMC_VOLTAGE_WINDOW_SD                       0x80100000U
#define SDMMC_HIGH_CAPACITY                           0x40000000U
#define SDMMC_STD_CAPACITY                            0x00000000U
#define SDMMC_CHECK_PATTERN                           0x000001AAU
#define SD_SWITCH_1_8V_CAPACITY                       0x01000000U

#define SDMMC_MAX_VOLT_TRIAL                          0x0000FFFFU

#define SDMMC_MAX_TRIAL                               0x0000FFFFU

#define SDMMC_ALLZERO                                 0x00000000U

#define SDMMC_WIDE_BUS_SUPPORT                        0x00040000U
#define SDMMC_SINGLE_BUS_SUPPORT                      0x00010000U
#define SDMMC_CARD_LOCKED                             0x02000000U

#ifndef SDMMC_DATATIMEOUT
#define SDMMC_DATATIMEOUT                             0xFFFFFFFFU
#endif /* SDMMC_DATATIMEOUT */

#define SDMMC_0TO7BITS                                0x000000FFU
#define SDMMC_8TO15BITS                               0x0000FF00U
#define SDMMC_16TO23BITS                              0x00FF0000U
#define SDMMC_24TO31BITS                              0xFF000000U
#define SDMMC_MAX_DATA_LENGTH                         0x01FFFFFFU

#define SDMMC_HALFFIFO                                0x00000008U
#define SDMMC_HALFFIFOBYTES                           0x00000020U

/** 
  * @brief  Command Class supported
  */
#define SDIO_CCCC_ERASE                       0x00000020U

#define SDIO_CMDTIMEOUT                       5000U         /* Command send and response timeout */
#define SDIO_MAXERASETIMEOUT                  63000U        /* Max erase Timeout 63 s            */
#define SDIO_STOPTRANSFERTIMEOUT              100000000U    /* Timeout for STOP TRANSMISSION command */

/** @defgroup SDIO_LL_Clock_Edge Clock Edge
  * @{
  */
#define SDIO_CLOCK_EDGE_RISING               0x00000000U
#define SDIO_CLOCK_EDGE_FALLING              SDIO_CLKCR_NEGEDGE

#define IS_SDIO_CLOCK_EDGE(EDGE) (((EDGE) == SDIO_CLOCK_EDGE_RISING) || \
                                          ((EDGE) == SDIO_CLOCK_EDGE_FALLING))
/**
  * @}
  */

/** @defgroup SDIO_LL_Clock_Bypass Clock Bypass
  * @{
  */
#define SDIO_CLOCK_BYPASS_DISABLE             0x00000000U
#define SDIO_CLOCK_BYPASS_ENABLE              SDIO_CLKCR_BYPASS   

#define IS_SDIO_CLOCK_BYPASS(BYPASS) (((BYPASS) == SDIO_CLOCK_BYPASS_DISABLE) || \
                                              ((BYPASS) == SDIO_CLOCK_BYPASS_ENABLE))
/**
  * @}
  */ 

/** @defgroup SDIO_LL_Clock_Power_Save Clock Power Saving
  * @{
  */
#define SDIO_CLOCK_POWER_SAVE_DISABLE         0x00000000U
#define SDIO_CLOCK_POWER_SAVE_ENABLE          SDIO_CLKCR_PWRSAV

#define IS_SDIO_CLOCK_POWER_SAVE(SAVE) (((SAVE) == SDIO_CLOCK_POWER_SAVE_DISABLE) || \
                                                ((SAVE) == SDIO_CLOCK_POWER_SAVE_ENABLE))
/**
  * @}
  */

/** @defgroup SDIO_LL_Bus_Wide Bus Width
  * @{
  */
#define SDIO_BUS_WIDE_1B                      0x00000000U
#define SDIO_BUS_WIDE_4B                      SDIO_CLKCR_WIDBUS_0
#define SDIO_BUS_WIDE_8B                      SDIO_CLKCR_WIDBUS_1

#define IS_SDIO_BUS_WIDE(WIDE) (((WIDE) == SDIO_BUS_WIDE_1B) || \
                                        ((WIDE) == SDIO_BUS_WIDE_4B) || \
                                        ((WIDE) == SDIO_BUS_WIDE_8B))
/**
  * @}
  */

/** @defgroup SDIO_LL_Hardware_Flow_Control Hardware Flow Control
  * @{
  */
#define SDIO_HARDWARE_FLOW_CONTROL_DISABLE    0x00000000U
#define SDIO_HARDWARE_FLOW_CONTROL_ENABLE     SDIO_CLKCR_HWFC_EN

#define IS_SDIO_HARDWARE_FLOW_CONTROL(CONTROL) (((CONTROL) == SDIO_HARDWARE_FLOW_CONTROL_DISABLE) || \
                                                        ((CONTROL) == SDIO_HARDWARE_FLOW_CONTROL_ENABLE))
/**
  * @}
  */
  
/** @defgroup SDIO_LL_Clock_Division Clock Division
  * @{
  */
#define IS_SDIO_CLKDIV(DIV)   ((DIV) <= 0xFFU)
/**
  * @}
  */  
    
/** @defgroup SDIO_LL_Command_Index Command Index
  * @{
  */
#define IS_SDIO_CMD_INDEX(INDEX)            ((INDEX) < 0x40U)
/**
  * @}
  */

/** @defgroup SDIO_LL_Response_Type Response Type
  * @{
  */
#define SDIO_RESPONSE_NO                    0x00000000U
#define SDIO_RESPONSE_SHORT                 SDIO_CMD_WAITRESP_0
#define SDIO_RESPONSE_LONG                  SDIO_CMD_WAITRESP

#define IS_SDIO_RESPONSE(RESPONSE) (((RESPONSE) == SDIO_RESPONSE_NO)    || \
                                            ((RESPONSE) == SDIO_RESPONSE_SHORT) || \
                                            ((RESPONSE) == SDIO_RESPONSE_LONG))
/**
  * @}
  */

/** @defgroup SDIO_LL_Wait_Interrupt_State Wait Interrupt
  * @{
  */
#define SDIO_WAIT_NO                        0x00000000U
#define SDIO_WAIT_IT                        SDIO_CMD_WAITINT 
#define SDIO_WAIT_PEND                      SDIO_CMD_WAITPEND

#define IS_SDIO_WAIT(WAIT) (((WAIT) == SDIO_WAIT_NO) || \
                                    ((WAIT) == SDIO_WAIT_IT) || \
                                    ((WAIT) == SDIO_WAIT_PEND))
/**
  * @}
  */

/** @defgroup SDIO_LL_CPSM_State CPSM State
  * @{
  */
#define SDIO_CPSM_DISABLE                   0x00000000U
#define SDIO_CPSM_ENABLE                    SDIO_CMD_CPSMEN

#define IS_SDIO_CPSM(CPSM) (((CPSM) == SDIO_CPSM_DISABLE) || \
                                    ((CPSM) == SDIO_CPSM_ENABLE))
/**
  * @}
  */  

/** @defgroup SDIO_LL_Response_Registers Response Register
  * @{
  */
#define SDIO_RESP1                          0x00000000U
#define SDIO_RESP2                          0x00000004U
#define SDIO_RESP3                          0x00000008U
#define SDIO_RESP4                          0x0000000CU

#define IS_SDIO_RESP(RESP) (((RESP) == SDIO_RESP1) || \
                                    ((RESP) == SDIO_RESP2) || \
                                    ((RESP) == SDIO_RESP3) || \
                                    ((RESP) == SDIO_RESP4))
/**
  * @}
  */

/** @defgroup SDIO_LL_Data_Length Data Length
  * @{
  */
#define IS_SDIO_DATA_LENGTH(LENGTH) ((LENGTH) <= 0x01FFFFFFU)
/**
  * @}
  */

/** @defgroup SDIO_LL_Data_Block_Size  Data Block Size
  * @{
  */
#define SDIO_DATABLOCK_SIZE_1B               0x00000000U
#define SDIO_DATABLOCK_SIZE_2B               SDIO_DCTRL_DBLOCKSIZE_0
#define SDIO_DATABLOCK_SIZE_4B               SDIO_DCTRL_DBLOCKSIZE_1
#define SDIO_DATABLOCK_SIZE_8B               (SDIO_DCTRL_DBLOCKSIZE_0|SDIO_DCTRL_DBLOCKSIZE_1)
#define SDIO_DATABLOCK_SIZE_16B              SDIO_DCTRL_DBLOCKSIZE_2
#define SDIO_DATABLOCK_SIZE_32B              (SDIO_DCTRL_DBLOCKSIZE_0|SDIO_DCTRL_DBLOCKSIZE_2)
#define SDIO_DATABLOCK_SIZE_64B              (SDIO_DCTRL_DBLOCKSIZE_1|SDIO_DCTRL_DBLOCKSIZE_2)
#define SDIO_DATABLOCK_SIZE_128B             (SDIO_DCTRL_DBLOCKSIZE_0|SDIO_DCTRL_DBLOCKSIZE_1|SDIO_DCTRL_DBLOCKSIZE_2)
#define SDIO_DATABLOCK_SIZE_256B             SDIO_DCTRL_DBLOCKSIZE_3
#define SDIO_DATABLOCK_SIZE_512B             (SDIO_DCTRL_DBLOCKSIZE_0|SDIO_DCTRL_DBLOCKSIZE_3)
#define SDIO_DATABLOCK_SIZE_1024B            (SDIO_DCTRL_DBLOCKSIZE_1|SDIO_DCTRL_DBLOCKSIZE_3)
#define SDIO_DATABLOCK_SIZE_2048B            (SDIO_DCTRL_DBLOCKSIZE_0|SDIO_DCTRL_DBLOCKSIZE_1|SDIO_DCTRL_DBLOCKSIZE_3) 
#define SDIO_DATABLOCK_SIZE_4096B            (SDIO_DCTRL_DBLOCKSIZE_2|SDIO_DCTRL_DBLOCKSIZE_3)
#define SDIO_DATABLOCK_SIZE_8192B            (SDIO_DCTRL_DBLOCKSIZE_0|SDIO_DCTRL_DBLOCKSIZE_2|SDIO_DCTRL_DBLOCKSIZE_3)
#define SDIO_DATABLOCK_SIZE_16384B           (SDIO_DCTRL_DBLOCKSIZE_1|SDIO_DCTRL_DBLOCKSIZE_2|SDIO_DCTRL_DBLOCKSIZE_3)

#define IS_SDIO_BLOCK_SIZE(SIZE) (((SIZE) == SDIO_DATABLOCK_SIZE_1B)    || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_2B)    || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_4B)    || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_8B)    || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_16B)   || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_32B)   || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_64B)   || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_128B)  || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_256B)  || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_512B)  || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_1024B) || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_2048B) || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_4096B) || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_8192B) || \
                                          ((SIZE) == SDIO_DATABLOCK_SIZE_16384B)) 
/**
  * @}
  */

/** @defgroup SDIO_LL_Transfer_Direction Transfer Direction
  * @{
  */
#define SDIO_TRANSFER_DIR_TO_CARD            0x00000000U
#define SDIO_TRANSFER_DIR_TO_SDIO    SDIO_DCTRL_DTDIR

#define IS_SDIO_TRANSFER_DIR(DIR) (((DIR) == SDIO_TRANSFER_DIR_TO_CARD) || \
                                           ((DIR) == SDIO_TRANSFER_DIR_TO_SDIO))
/**
  * @}
  */

/** @defgroup SDIO_LL_Transfer_Type Transfer Type
  * @{
  */
#define SDIO_TRANSFER_MODE_BLOCK             0x00000000U
#define SDIO_TRANSFER_MODE_STREAM            SDIO_DCTRL_DTMODE

#define IS_SDIO_TRANSFER_MODE(MODE) (((MODE) == SDIO_TRANSFER_MODE_BLOCK) || \
                                             ((MODE) == SDIO_TRANSFER_MODE_STREAM))
/**
  * @}
  */

/** @defgroup SDIO_LL_DPSM_State DPSM State
  * @{
  */
#define SDIO_DPSM_DISABLE                    0x00000000U
#define SDIO_DPSM_ENABLE                     SDIO_DCTRL_DTEN

#define IS_SDIO_DPSM(DPSM) (((DPSM) == SDIO_DPSM_DISABLE) ||\
                                    ((DPSM) == SDIO_DPSM_ENABLE))
/**
  * @}
  */
  
/** @defgroup SDIO_LL_Read_Wait_Mode Read Wait Mode
  * @{
  */
#define SDIO_READ_WAIT_MODE_DATA2                0x00000000U
#define SDIO_READ_WAIT_MODE_CLK                  (SDIO_DCTRL_RWMOD)

#define IS_SDIO_READWAIT_MODE(MODE) (((MODE) == SDIO_READ_WAIT_MODE_CLK) || \
                                             ((MODE) == SDIO_READ_WAIT_MODE_DATA2))
/**
  * @}
  */  

/** @defgroup SDIO_LL_Interrupt_sources Interrupt Sources
  * @{
  */
#define SDIO_IT_CCRCFAIL                    SDIO_MASK_CCRCFAILIE
#define SDIO_IT_DCRCFAIL                    SDIO_MASK_DCRCFAILIE
#define SDIO_IT_CTIMEOUT                    SDIO_MASK_CTIMEOUTIE
#define SDIO_IT_DTIMEOUT                    SDIO_MASK_DTIMEOUTIE
#define SDIO_IT_TXUNDERR                    SDIO_MASK_TXUNDERRIE
#define SDIO_IT_RXOVERR                     SDIO_MASK_RXOVERRIE
#define SDIO_IT_CMDREND                     SDIO_MASK_CMDRENDIE
#define SDIO_IT_CMDSENT                     SDIO_MASK_CMDSENTIE
#define SDIO_IT_DATAEND                     SDIO_MASK_DATAENDIE
#if defined(SDIO_STA_STBITERR)
#define SDIO_IT_STBITERR                    SDIO_MASK_STBITERRIE
#endif
#define SDIO_IT_DBCKEND                     SDIO_MASK_DBCKENDIE
#define SDIO_IT_CMDACT                      SDIO_MASK_CMDACTIE
#define SDIO_IT_TXACT                       SDIO_MASK_TXACTIE
#define SDIO_IT_RXACT                       SDIO_MASK_RXACTIE
#define SDIO_IT_TXFIFOHE                    SDIO_MASK_TXFIFOHEIE
#define SDIO_IT_RXFIFOHF                    SDIO_MASK_RXFIFOHFIE
#define SDIO_IT_TXFIFOF                     SDIO_MASK_TXFIFOFIE
#define SDIO_IT_RXFIFOF                     SDIO_MASK_RXFIFOFIE
#define SDIO_IT_TXFIFOE                     SDIO_MASK_TXFIFOEIE
#define SDIO_IT_RXFIFOE                     SDIO_MASK_RXFIFOEIE
#define SDIO_IT_TXDAVL                      SDIO_MASK_TXDAVLIE
#define SDIO_IT_RXDAVL                      SDIO_MASK_RXDAVLIE
#define SDIO_IT_SDIOIT                      SDIO_MASK_SDIOITIE
#if defined(SDIO_CMD_CEATACMD)
#define SDIO_IT_CEATAEND                    SDIO_MASK_CEATAENDIE
#endif
/**
  * @}
  */ 

/** @defgroup SDIO_LL_Flags Flags
  * @{
  */
#define SDIO_FLAG_CCRCFAIL                  SDIO_STA_CCRCFAIL
#define SDIO_FLAG_DCRCFAIL                  SDIO_STA_DCRCFAIL
#define SDIO_FLAG_CTIMEOUT                  SDIO_STA_CTIMEOUT
#define SDIO_FLAG_DTIMEOUT                  SDIO_STA_DTIMEOUT
#define SDIO_FLAG_TXUNDERR                  SDIO_STA_TXUNDERR
#define SDIO_FLAG_RXOVERR                   SDIO_STA_RXOVERR
#define SDIO_FLAG_CMDREND                   SDIO_STA_CMDREND
#define SDIO_FLAG_CMDSENT                   SDIO_STA_CMDSENT
#define SDIO_FLAG_DATAEND                   SDIO_STA_DATAEND
#if defined(SDIO_STA_STBITERR)
#define SDIO_FLAG_STBITERR                  SDIO_STA_STBITERR
#endif
#define SDIO_FLAG_DBCKEND                   SDIO_STA_DBCKEND
#define SDIO_FLAG_CMDACT                    SDIO_STA_CMDACT
#define SDIO_FLAG_TXACT                     SDIO_STA_TXACT
#define SDIO_FLAG_RXACT                     SDIO_STA_RXACT
#define SDIO_FLAG_TXFIFOHE                  SDIO_STA_TXFIFOHE
#define SDIO_FLAG_RXFIFOHF                  SDIO_STA_RXFIFOHF
#define SDIO_FLAG_TXFIFOF                   SDIO_STA_TXFIFOF
#define SDIO_FLAG_RXFIFOF                   SDIO_STA_RXFIFOF
#define SDIO_FLAG_TXFIFOE                   SDIO_STA_TXFIFOE
#define SDIO_FLAG_RXFIFOE                   SDIO_STA_RXFIFOE
#define SDIO_FLAG_TXDAVL                    SDIO_STA_TXDAVL
#define SDIO_FLAG_RXDAVL                    SDIO_STA_RXDAVL
#define SDIO_FLAG_SDIOIT                    SDIO_STA_SDIOIT
#if defined(SDIO_CMD_CEATACMD)
#define SDIO_FLAG_CEATAEND                  SDIO_STA_CEATAEND
#endif
#define SDIO_STATIC_FLAGS                   ((uint32_t)(SDIO_FLAG_CCRCFAIL | SDIO_FLAG_DCRCFAIL | SDIO_FLAG_CTIMEOUT |\
                                                         SDIO_FLAG_DTIMEOUT | SDIO_FLAG_TXUNDERR | SDIO_FLAG_RXOVERR  |\
                                                         SDIO_FLAG_CMDREND  | SDIO_FLAG_CMDSENT  | SDIO_FLAG_DATAEND  |\
                                                         SDIO_FLAG_DBCKEND | SDIO_FLAG_SDIOIT))  

#define SDIO_STATIC_CMD_FLAGS               ((uint32_t)(SDIO_FLAG_CCRCFAIL | SDIO_FLAG_CTIMEOUT | SDIO_FLAG_CMDREND |\
                                                         SDIO_FLAG_CMDSENT))

#define SDIO_STATIC_DATA_FLAGS              ((uint32_t)(SDIO_FLAG_DCRCFAIL | SDIO_FLAG_DTIMEOUT | SDIO_FLAG_TXUNDERR |\
                                                         SDIO_FLAG_RXOVERR  | SDIO_FLAG_DATAEND  | SDIO_FLAG_DBCKEND))
/**
  * @}
  */

/**
  * @}
  */
  
/* Exported macro ------------------------------------------------------------*/
/** @defgroup SDIO_LL_Exported_macros SDIO_LL Exported Macros
  * @{
  */

/** @defgroup SDMMC_LL_Alias_Region Bit Address in the alias region
  * @{
  */
/* ------------ SDIO registers bit address in the alias region -------------- */
#define SDIO_OFFSET               (SDIO_BASE - PERIPH_BASE)

/* --- CLKCR Register ---*/
/* Alias word address of CLKEN bit */
#define CLKCR_OFFSET              (SDIO_OFFSET + 0x04U)
#define CLKEN_BITNUMBER           0x08U
#define CLKCR_CLKEN_BB            (PERIPH_BB_BASE + (CLKCR_OFFSET * 32U) + (CLKEN_BITNUMBER * 4U))

/* --- CMD Register ---*/
/* Alias word address of SDIOSUSPEND bit */
#define CMD_OFFSET                (SDIO_OFFSET + 0x0CU)
#define SDIOSUSPEND_BITNUMBER     0x0BU
#define CMD_SDIOSUSPEND_BB        (PERIPH_BB_BASE + (CMD_OFFSET * 32U) + (SDIOSUSPEND_BITNUMBER * 4U))

/* Alias word address of ENCMDCOMPL bit */
#define ENCMDCOMPL_BITNUMBER      0x0CU
#define CMD_ENCMDCOMPL_BB         (PERIPH_BB_BASE + (CMD_OFFSET * 32U) + (ENCMDCOMPL_BITNUMBER * 4U))

/* Alias word address of NIEN bit */
#define NIEN_BITNUMBER            0x0DU
#define CMD_NIEN_BB               (PERIPH_BB_BASE + (CMD_OFFSET * 32U) + (NIEN_BITNUMBER * 4U))

/* Alias word address of ATACMD bit */
#define ATACMD_BITNUMBER          0x0EU
#define CMD_ATACMD_BB             (PERIPH_BB_BASE + (CMD_OFFSET * 32U) + (ATACMD_BITNUMBER * 4U))

/* --- DCTRL Register ---*/
/* Alias word address of DMAEN bit */
#define DCTRL_OFFSET              (SDIO_OFFSET + 0x2CU)
#define DMAEN_BITNUMBER           0x03U
#define DCTRL_DMAEN_BB            (PERIPH_BB_BASE + (DCTRL_OFFSET * 32U) + (DMAEN_BITNUMBER * 4U))

/* Alias word address of RWSTART bit */
#define RWSTART_BITNUMBER         0x08U
#define DCTRL_RWSTART_BB          (PERIPH_BB_BASE + (DCTRL_OFFSET * 32U) + (RWSTART_BITNUMBER * 4U))

/* Alias word address of RWSTOP bit */
#define RWSTOP_BITNUMBER          0x09U
#define DCTRL_RWSTOP_BB           (PERIPH_BB_BASE + (DCTRL_OFFSET * 32U) + (RWSTOP_BITNUMBER * 4U))

/* Alias word address of RWMOD bit */
#define RWMOD_BITNUMBER           0x0AU
#define DCTRL_RWMOD_BB            (PERIPH_BB_BASE + (DCTRL_OFFSET * 32U) + (RWMOD_BITNUMBER * 4U))

/* Alias word address of SDIOEN bit */
#define SDIOEN_BITNUMBER          0x0BU
#define DCTRL_SDIOEN_BB           (PERIPH_BB_BASE + (DCTRL_OFFSET * 32U) + (SDIOEN_BITNUMBER * 4U))
/**
  * @}
  */

/** @defgroup SDIO_LL_Register Bits And Addresses Definitions
  * @brief SDIO_LL registers bit address in the alias region
  * @{
  */
/* ---------------------- SDIO registers bit mask --------------------------- */
/* --- CLKCR Register ---*/
/* CLKCR register clear mask */ 
#define CLKCR_CLEAR_MASK         ((uint32_t)(SDIO_CLKCR_CLKDIV  | SDIO_CLKCR_PWRSAV |\
                                             SDIO_CLKCR_BYPASS  | SDIO_CLKCR_WIDBUS |\
                                             SDIO_CLKCR_NEGEDGE | SDIO_CLKCR_HWFC_EN))

/* --- DCTRL Register ---*/
/* SDIO DCTRL Clear Mask */
#define DCTRL_CLEAR_MASK         ((uint32_t)(SDIO_DCTRL_DTEN    | SDIO_DCTRL_DTDIR |\
                                             SDIO_DCTRL_DTMODE  | SDIO_DCTRL_DBLOCKSIZE))

/* --- CMD Register ---*/
/* CMD Register clear mask */
#define CMD_CLEAR_MASK           ((uint32_t)(SDIO_CMD_CMDINDEX | SDIO_CMD_WAITRESP |\
                                             SDIO_CMD_WAITINT  | SDIO_CMD_WAITPEND |\
                                             SDIO_CMD_CPSMEN   | SDIO_CMD_SDIOSUSPEND))

/* SDIO Initialization Frequency (400KHz max) */
#define SDIO_INIT_CLK_DIV     ((uint8_t)0x76)    /* 48MHz / (SDMMC_INIT_CLK_DIV + 2) < 400KHz */

/* SDIO Data Transfer Frequency (25MHz max) */
#define SDIO_TRANSFER_CLK_DIV ((uint8_t)0x0)     /* 48MHz / (SDMMC_TRANSFER_CLK_DIV + 2) < 25MHz */
/**
  * @}
  */

/** @defgroup SDIO_LL_Interrupt_Clock Interrupt And Clock Configuration
 *  @brief macros to handle interrupts and specific clock configurations
 * @{
 */
 
/**
  * @brief  Enable the SDIO device.
  * @param  __INSTANCE__: SDIO Instance  
  * @retval None
  */
#define __SDIO_ENABLE(__INSTANCE__)  (*(__IO uint32_t *)CLKCR_CLKEN_BB = ENABLE)

/**
  * @brief  Disable the SDIO device.
  * @param  __INSTANCE__: SDIO Instance  
  * @retval None
  */
#define __SDIO_DISABLE(__INSTANCE__)  (*(__IO uint32_t *)CLKCR_CLKEN_BB = DISABLE)

/**
  * @brief  Enable the SDIO DMA transfer.
  * @param  __INSTANCE__: SDIO Instance  
  * @retval None
  */
#define __SDIO_DMA_ENABLE(__INSTANCE__)  (*(__IO uint32_t *)DCTRL_DMAEN_BB = ENABLE)

/**
  * @brief  Disable the SDIO DMA transfer.
  * @param  __INSTANCE__: SDIO Instance   
  * @retval None
  */
#define __SDIO_DMA_DISABLE(__INSTANCE__)  (*(__IO uint32_t *)DCTRL_DMAEN_BB = DISABLE)
 
/**
  * @brief  Enable the SDIO device interrupt.
  * @param  __INSTANCE__ : Pointer to SDIO register base  
  * @param  __INTERRUPT__ : specifies the SDIO interrupt sources to be enabled.
  *         This parameter can be one or a combination of the following values:
  *            @arg SDIO_IT_CCRCFAIL: Command response received (CRC check failed) interrupt
  *            @arg SDIO_IT_DCRCFAIL: Data block sent/received (CRC check failed) interrupt
  *            @arg SDIO_IT_CTIMEOUT: Command response timeout interrupt
  *            @arg SDIO_IT_DTIMEOUT: Data timeout interrupt
  *            @arg SDIO_IT_TXUNDERR: Transmit FIFO underrun error interrupt
  *            @arg SDIO_IT_RXOVERR:  Received FIFO overrun error interrupt
  *            @arg SDIO_IT_CMDREND:  Command response received (CRC check passed) interrupt
  *            @arg SDIO_IT_CMDSENT:  Command sent (no response required) interrupt
  *            @arg SDIO_IT_DATAEND:  Data end (data counter, DATACOUNT, is zero) interrupt
  *            @arg SDIO_IT_DBCKEND:  Data block sent/received (CRC check passed) interrupt
  *            @arg SDIO_IT_CMDACT:   Command transfer in progress interrupt
  *            @arg SDIO_IT_TXACT:    Data transmit in progress interrupt
  *            @arg SDIO_IT_RXACT:    Data receive in progress interrupt
  *            @arg SDIO_IT_TXFIFOHE: Transmit FIFO Half Empty interrupt
  *            @arg SDIO_IT_RXFIFOHF: Receive FIFO Half Full interrupt
  *            @arg SDIO_IT_TXFIFOF:  Transmit FIFO full interrupt
  *            @arg SDIO_IT_RXFIFOF:  Receive FIFO full interrupt
  *            @arg SDIO_IT_TXFIFOE:  Transmit FIFO empty interrupt
  *            @arg SDIO_IT_RXFIFOE:  Receive FIFO empty interrupt
  *            @arg SDIO_IT_TXDAVL:   Data available in transmit FIFO interrupt
  *            @arg SDIO_IT_RXDAVL:   Data available in receive FIFO interrupt
  *            @arg SDIO_IT_SDIOIT:   SDIO interrupt received interrupt 
  * @retval None
  */
#define __SDIO_ENABLE_IT(__INSTANCE__, __INTERRUPT__)  ((__INSTANCE__)->MASK |= (__INTERRUPT__))

/**
  * @brief  Disable the SDIO device interrupt.
  * @param  __INSTANCE__ : Pointer to SDIO register base   
  * @param  __INTERRUPT__ : specifies the SDIO interrupt sources to be disabled.
  *          This parameter can be one or a combination of the following values:
  *            @arg SDIO_IT_CCRCFAIL: Command response received (CRC check failed) interrupt
  *            @arg SDIO_IT_DCRCFAIL: Data block sent/received (CRC check failed) interrupt
  *            @arg SDIO_IT_CTIMEOUT: Command response timeout interrupt
  *            @arg SDIO_IT_DTIMEOUT: Data timeout interrupt
  *            @arg SDIO_IT_TXUNDERR: Transmit FIFO underrun error interrupt
  *            @arg SDIO_IT_RXOVERR:  Received FIFO overrun error interrupt
  *            @arg SDIO_IT_CMDREND:  Command response received (CRC check passed) interrupt
  *            @arg SDIO_IT_CMDSENT:  Command sent (no response required) interrupt
  *            @arg SDIO_IT_DATAEND:  Data end (data counter, DATACOUNT, is zero) interrupt
  *            @arg SDIO_IT_DBCKEND:  Data block sent/received (CRC check passed) interrupt
  *            @arg SDIO_IT_CMDACT:   Command transfer in progress interrupt
  *            @arg SDIO_IT_TXACT:    Data transmit in progress interrupt
  *            @arg SDIO_IT_RXACT:    Data receive in progress interrupt
  *            @arg SDIO_IT_TXFIFOHE: Transmit FIFO Half Empty interrupt
  *            @arg SDIO_IT_RXFIFOHF: Receive FIFO Half Full interrupt
  *            @arg SDIO_IT_TXFIFOF:  Transmit FIFO full interrupt
  *            @arg SDIO_IT_RXFIFOF:  Receive FIFO full interrupt
  *            @arg SDIO_IT_TXFIFOE:  Transmit FIFO empty interrupt
  *            @arg SDIO_IT_RXFIFOE:  Receive FIFO empty interrupt
  *            @arg SDIO_IT_TXDAVL:   Data available in transmit FIFO interrupt
  *            @arg SDIO_IT_RXDAVL:   Data available in receive FIFO interrupt
  *            @arg SDIO_IT_SDIOIT:   SDIO interrupt received interrupt   
  * @retval None
  */
#define __SDIO_DISABLE_IT(__INSTANCE__, __INTERRUPT__)  ((__INSTANCE__)->MASK &= ~(__INTERRUPT__))

/**
  * @brief  Checks whether the specified SDIO flag is set or not. 
  * @param  __INSTANCE__ : Pointer to SDIO register base   
  * @param  __FLAG__: specifies the flag to check. 
  *          This parameter can be one of the following values:
  *            @arg SDIO_FLAG_CCRCFAIL: Command response received (CRC check failed)
  *            @arg SDIO_FLAG_DCRCFAIL: Data block sent/received (CRC check failed)
  *            @arg SDIO_FLAG_CTIMEOUT: Command response timeout
  *            @arg SDIO_FLAG_DTIMEOUT: Data timeout
  *            @arg SDIO_FLAG_TXUNDERR: Transmit FIFO underrun error
  *            @arg SDIO_FLAG_RXOVERR:  Received FIFO overrun error
  *            @arg SDIO_FLAG_CMDREND:  Command response received (CRC check passed)
  *            @arg SDIO_FLAG_CMDSENT:  Command sent (no response required)
  *            @arg SDIO_FLAG_DATAEND:  Data end (data counter, DATACOUNT, is zero)
  *            @arg SDIO_FLAG_DBCKEND:  Data block sent/received (CRC check passed)
  *            @arg SDIO_FLAG_CMDACT:   Command transfer in progress
  *            @arg SDIO_FLAG_TXACT:    Data transmit in progress
  *            @arg SDIO_FLAG_RXACT:    Data receive in progress
  *            @arg SDIO_FLAG_TXFIFOHE: Transmit FIFO Half Empty
  *            @arg SDIO_FLAG_RXFIFOHF: Receive FIFO Half Full
  *            @arg SDIO_FLAG_TXFIFOF:  Transmit FIFO full
  *            @arg SDIO_FLAG_RXFIFOF:  Receive FIFO full
  *            @arg SDIO_FLAG_TXFIFOE:  Transmit FIFO empty
  *            @arg SDIO_FLAG_RXFIFOE:  Receive FIFO empty
  *            @arg SDIO_FLAG_TXDAVL:   Data available in transmit FIFO
  *            @arg SDIO_FLAG_RXDAVL:   Data available in receive FIFO
  *            @arg SDIO_FLAG_SDIOIT:   SDIO interrupt received
  * @retval The new state of SDIO_FLAG (SET or RESET).
  */
#define __SDIO_GET_FLAG(__INSTANCE__, __FLAG__)  (((__INSTANCE__)->STA &(__FLAG__)) != 0U)


/**
  * @brief  Clears the SDIO pending flags.
  * @param  __INSTANCE__ : Pointer to SDIO register base  
  * @param  __FLAG__: specifies the flag to clear.  
  *          This parameter can be one or a combination of the following values:
  *            @arg SDIO_FLAG_CCRCFAIL: Command response received (CRC check failed)
  *            @arg SDIO_FLAG_DCRCFAIL: Data block sent/received (CRC check failed)
  *            @arg SDIO_FLAG_CTIMEOUT: Command response timeout
  *            @arg SDIO_FLAG_DTIMEOUT: Data timeout
  *            @arg SDIO_FLAG_TXUNDERR: Transmit FIFO underrun error
  *            @arg SDIO_FLAG_RXOVERR:  Received FIFO overrun error
  *            @arg SDIO_FLAG_CMDREND:  Command response received (CRC check passed)
  *            @arg SDIO_FLAG_CMDSENT:  Command sent (no response required)
  *            @arg SDIO_FLAG_DATAEND:  Data end (data counter, DATACOUNT, is zero)
  *            @arg SDIO_FLAG_DBCKEND:  Data block sent/received (CRC check passed)
  *            @arg SDIO_FLAG_SDIOIT:   SDIO interrupt received
  * @retval None
  */
#define __SDIO_CLEAR_FLAG(__INSTANCE__, __FLAG__)  ((__INSTANCE__)->ICR = (__FLAG__))

/**
  * @brief  Checks whether the specified SDIO interrupt has occurred or not.
  * @param  __INSTANCE__ : Pointer to SDIO register base   
  * @param  __INTERRUPT__: specifies the SDIO interrupt source to check. 
  *          This parameter can be one of the following values:
  *            @arg SDIO_IT_CCRCFAIL: Command response received (CRC check failed) interrupt
  *            @arg SDIO_IT_DCRCFAIL: Data block sent/received (CRC check failed) interrupt
  *            @arg SDIO_IT_CTIMEOUT: Command response timeout interrupt
  *            @arg SDIO_IT_DTIMEOUT: Data timeout interrupt
  *            @arg SDIO_IT_TXUNDERR: Transmit FIFO underrun error interrupt
  *            @arg SDIO_IT_RXOVERR:  Received FIFO overrun error interrupt
  *            @arg SDIO_IT_CMDREND:  Command response received (CRC check passed) interrupt
  *            @arg SDIO_IT_CMDSENT:  Command sent (no response required) interrupt
  *            @arg SDIO_IT_DATAEND:  Data end (data counter, DATACOUNT, is zero) interrupt
  *            @arg SDIO_IT_DBCKEND:  Data block sent/received (CRC check passed) interrupt
  *            @arg SDIO_IT_CMDACT:   Command transfer in progress interrupt
  *            @arg SDIO_IT_TXACT:    Data transmit in progress interrupt
  *            @arg SDIO_IT_RXACT:    Data receive in progress interrupt
  *            @arg SDIO_IT_TXFIFOHE: Transmit FIFO Half Empty interrupt
  *            @arg SDIO_IT_RXFIFOHF: Receive FIFO Half Full interrupt
  *            @arg SDIO_IT_TXFIFOF:  Transmit FIFO full interrupt
  *            @arg SDIO_IT_RXFIFOF:  Receive FIFO full interrupt
  *            @arg SDIO_IT_TXFIFOE:  Transmit FIFO empty interrupt
  *            @arg SDIO_IT_RXFIFOE:  Receive FIFO empty interrupt
  *            @arg SDIO_IT_TXDAVL:   Data available in transmit FIFO interrupt
  *            @arg SDIO_IT_RXDAVL:   Data available in receive FIFO interrupt
  *            @arg SDIO_IT_SDIOIT:   SDIO interrupt received interrupt
  * @retval The new state of SDIO_IT (SET or RESET).
  */
#define __SDIO_GET_IT  (__INSTANCE__, __INTERRUPT__)  (((__INSTANCE__)->STA &(__INTERRUPT__)) == (__INTERRUPT__))

/**
  * @brief  Clears the SDIO's interrupt pending bits.
  * @param  __INSTANCE__ : Pointer to SDIO register base 
  * @param  __INTERRUPT__: specifies the interrupt pending bit to clear. 
  *          This parameter can be one or a combination of the following values:
  *            @arg SDIO_IT_CCRCFAIL: Command response received (CRC check failed) interrupt
  *            @arg SDIO_IT_DCRCFAIL: Data block sent/received (CRC check failed) interrupt
  *            @arg SDIO_IT_CTIMEOUT: Command response timeout interrupt
  *            @arg SDIO_IT_DTIMEOUT: Data timeout interrupt
  *            @arg SDIO_IT_TXUNDERR: Transmit FIFO underrun error interrupt
  *            @arg SDIO_IT_RXOVERR:  Received FIFO overrun error interrupt
  *            @arg SDIO_IT_CMDREND:  Command response received (CRC check passed) interrupt
  *            @arg SDIO_IT_CMDSENT:  Command sent (no response required) interrupt
  *            @arg SDIO_IT_DATAEND:  Data end (data counter, DATACOUNT, is zero) interrupt
  *            @arg SDIO_IT_SDIOIT:   SDIO interrupt received interrupt
  * @retval None
  */
#define __SDIO_CLEAR_IT(__INSTANCE__, __INTERRUPT__)  ((__INSTANCE__)->ICR = (__INTERRUPT__))

/**
  * @brief  Enable Start the SD I/O Read Wait operation.
  * @param  __INSTANCE__ : Pointer to SDIO register base  
  * @retval None
  */
#define __SDIO_START_READWAIT_ENABLE(__INSTANCE__)  (*(__IO uint32_t *) DCTRL_RWSTART_BB = ENABLE)

/**
  * @brief  Disable Start the SD I/O Read Wait operations.
  * @param  __INSTANCE__ : Pointer to SDIO register base   
  * @retval None
  */
#define __SDIO_START_READWAIT_DISABLE(__INSTANCE__)  (*(__IO uint32_t *) DCTRL_RWSTART_BB = DISABLE)

/**
  * @brief  Enable Start the SD I/O Read Wait operation.
  * @param  __INSTANCE__ : Pointer to SDIO register base   
  * @retval None
  */
#define __SDIO_STOP_READWAIT_ENABLE(__INSTANCE__)  (*(__IO uint32_t *) DCTRL_RWSTOP_BB = ENABLE)

/**
  * @brief  Disable Stop the SD I/O Read Wait operations.
  * @param  __INSTANCE__ : Pointer to SDIO register base  
  * @retval None
  */
#define __SDIO_STOP_READWAIT_DISABLE(__INSTANCE__)  (*(__IO uint32_t *) DCTRL_RWSTOP_BB = DISABLE)

/**
  * @brief  Enable the SD I/O Mode Operation.
  * @param  __INSTANCE__ : Pointer to SDIO register base   
  * @retval None
  */
#define __SDIO_OPERATION_ENABLE(__INSTANCE__)  (*(__IO uint32_t *) DCTRL_SDIOEN_BB = ENABLE)

/**
  * @brief  Disable the SD I/O Mode Operation.
  * @param  __INSTANCE__ : Pointer to SDIO register base 
  * @retval None
  */
#define __SDIO_OPERATION_DISABLE(__INSTANCE__)  (*(__IO uint32_t *) DCTRL_SDIOEN_BB = DISABLE)

/**
  * @brief  Enable the SD I/O Suspend command sending.
  * @param  __INSTANCE__ : Pointer to SDIO register base  
  * @retval None
  */
#define __SDIO_SUSPEND_CMD_ENABLE(__INSTANCE__)  (*(__IO uint32_t *) CMD_SDIOSUSPEND_BB = ENABLE)

/**
  * @brief  Disable the SD I/O Suspend command sending.
  * @param  __INSTANCE__ : Pointer to SDIO register base  
  * @retval None
  */
#define __SDIO_SUSPEND_CMD_DISABLE(__INSTANCE__)  (*(__IO uint32_t *) CMD_SDIOSUSPEND_BB = DISABLE)

#if defined(SDIO_CMD_CEATACMD)
/**
  * @brief  Enable the command completion signal.
  * @retval None
  */    
#define __SDIO_CEATA_CMD_COMPLETION_ENABLE()   (*(__IO uint32_t *) CMD_ENCMDCOMPL_BB = ENABLE)

/**
  * @brief  Disable the command completion signal.
  * @retval None
  */  
#define __SDIO_CEATA_CMD_COMPLETION_DISABLE()   (*(__IO uint32_t *) CMD_ENCMDCOMPL_BB = DISABLE)

/**
  * @brief  Enable the CE-ATA interrupt.
  * @retval None
  */    
#define __SDIO_CEATA_ENABLE_IT()   (*(__IO uint32_t *) CMD_NIEN_BB = (uint32_t)0U)

/**
  * @brief  Disable the CE-ATA interrupt.
  * @retval None
  */  
#define __SDIO_CEATA_DISABLE_IT()   (*(__IO uint32_t *) CMD_NIEN_BB = (uint32_t)1U)

/**
  * @brief  Enable send CE-ATA command (CMD61).
  * @retval None
  */  
#define __SDIO_CEATA_SENDCMD_ENABLE()   (*(__IO uint32_t *) CMD_ATACMD_BB = ENABLE)

/**
  * @brief  Disable send CE-ATA command (CMD61).
  * @retval None
  */  
#define __SDIO_CEATA_SENDCMD_DISABLE()   (*(__IO uint32_t *) CMD_ATACMD_BB = DISABLE)
   
#endif
/**
  * @}
  */

/**
  * @}
  */  

/* Exported functions --------------------------------------------------------*/
/** @addtogroup SDMMC_LL_Exported_Functions
  * @{
  */
  
/* Initialization/de-initialization functions  **********************************/
/** @addtogroup HAL_SDMMC_LL_Group1
  * @{
  */
HAL_StatusTypeDef SDIO_Init(SDIO_TypeDef *SDIOx, SDIO_InitTypeDef Init);
/**
  * @}
  */
  
/* I/O operation functions  *****************************************************/
/** @addtogroup HAL_SDMMC_LL_Group2
  * @{
  */
uint32_t          SDIO_ReadFIFO(SDIO_TypeDef *SDIOx);
HAL_StatusTypeDef SDIO_WriteFIFO(SDIO_TypeDef *SDIOx, uint32_t *pWriteData);
/**
  * @}
  */
  
/* Peripheral Control functions  ************************************************/
/** @addtogroup HAL_SDMMC_LL_Group3
  * @{
  */
HAL_StatusTypeDef SDIO_PowerState_ON(SDIO_TypeDef *SDIOx);
HAL_StatusTypeDef SDIO_PowerState_OFF(SDIO_TypeDef *SDIOx);
uint32_t          SDIO_GetPowerState(SDIO_TypeDef *SDIOx);

/* Command path state machine (CPSM) management functions */
HAL_StatusTypeDef SDIO_SendCommand(SDIO_TypeDef *SDIOx, SDIO_CmdInitTypeDef *Command);
uint8_t           SDIO_GetCommandResponse(SDIO_TypeDef *SDIOx);
uint32_t          SDIO_GetResponse(SDIO_TypeDef *SDIOx, uint32_t Response);

/* Data path state machine (DPSM) management functions */
HAL_StatusTypeDef SDIO_ConfigData(SDIO_TypeDef *SDIOx, SDIO_DataInitTypeDef* Data);
uint32_t          SDIO_GetDataCounter(SDIO_TypeDef *SDIOx);
uint32_t          SDIO_GetFIFOCount(SDIO_TypeDef *SDIOx);

/* SDMMC Cards mode management functions */
HAL_StatusTypeDef SDIO_SetSDMMCReadWaitMode(SDIO_TypeDef *SDIOx, uint32_t SDIO_ReadWaitMode);

/* SDMMC Commands management functions */
uint32_t SDMMC_CmdBlockLength(SDIO_TypeDef *SDIOx, uint32_t BlockSize);
uint32_t SDMMC_CmdReadSingleBlock(SDIO_TypeDef *SDIOx, uint32_t ReadAdd);
uint32_t SDMMC_CmdReadMultiBlock(SDIO_TypeDef *SDIOx, uint32_t ReadAdd);
uint32_t SDMMC_CmdWriteSingleBlock(SDIO_TypeDef *SDIOx, uint32_t WriteAdd);
uint32_t SDMMC_CmdWriteMultiBlock(SDIO_TypeDef *SDIOx, uint32_t WriteAdd);
uint32_t SDMMC_CmdEraseStartAdd(SDIO_TypeDef *SDIOx, uint32_t StartAdd);
uint32_t SDMMC_CmdSDEraseStartAdd(SDIO_TypeDef *SDIOx, uint32_t StartAdd);
uint32_t SDMMC_CmdEraseEndAdd(SDIO_TypeDef *SDIOx, uint32_t EndAdd);
uint32_t SDMMC_CmdSDEraseEndAdd(SDIO_TypeDef *SDIOx, uint32_t EndAdd);
uint32_t SDMMC_CmdErase(SDIO_TypeDef *SDIOx);
uint32_t SDMMC_CmdStopTransfer(SDIO_TypeDef *SDIOx);
uint32_t SDMMC_CmdSelDesel(SDIO_TypeDef *SDIOx, uint64_t Addr);
uint32_t SDMMC_CmdGoIdleState(SDIO_TypeDef *SDIOx);
uint32_t SDMMC_CmdOperCond(SDIO_TypeDef *SDIOx);
uint32_t SDMMC_CmdAppCommand(SDIO_TypeDef *SDIOx, uint32_t Argument);
uint32_t SDMMC_CmdAppOperCommand(SDIO_TypeDef *SDIOx, uint32_t Argument);
uint32_t SDMMC_CmdBusWidth(SDIO_TypeDef *SDIOx, uint32_t BusWidth);
uint32_t SDMMC_CmdSendSCR(SDIO_TypeDef *SDIOx);
uint32_t SDMMC_CmdSendCID(SDIO_TypeDef *SDIOx);
uint32_t SDMMC_CmdSendCSD(SDIO_TypeDef *SDIOx, uint32_t Argument);
uint32_t SDMMC_CmdSendEXTCSD(SDIO_TypeDef *SDIOx, uint32_t Argument);
uint32_t SDMMC_CmdSetRelAdd(SDIO_TypeDef *SDIOx, uint16_t *pRCA);
uint32_t SDMMC_CmdSendStatus(SDIO_TypeDef *SDIOx, uint32_t Argument);
uint32_t SDMMC_CmdStatusRegister(SDIO_TypeDef *SDIOx);
uint32_t SDMMC_CmdOpCondition(SDIO_TypeDef *SDIOx, uint32_t Argument);
uint32_t SDMMC_CmdSwitch(SDIO_TypeDef *SDIOx, uint32_t Argument);

/**
  * @}
  */
  
/**
  * @}
  */
  
/**
  * @}
  */ 

/**
  * @}
  */

#endif /* SDIO */

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_LL_SDMMC_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
