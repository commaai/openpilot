/**
  ******************************************************************************
  * @file    stm32f4xx_hal_can_legacy.h
  * @author  MCD Application Team
  * @brief   Header file of CAN HAL module.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT(c) 2017 STMicroelectronics</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *   1. Redistributions of source code must retain the above copyright notice,
  *      this list of conditions and the following disclaimer.
  *   2. Redistributions in binary form must reproduce the above copyright notice,
  *      this list of conditions and the following disclaimer in the documentation
  *      and/or other materials provided with the distribution.
  *   3. Neither the name of STMicroelectronics nor the names of its contributors
  *      may be used to endorse or promote products derived from this software
  *      without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __STM32F4xx_HAL_CAN_LEGACY_H
#define __STM32F4xx_HAL_CAN_LEGACY_H

#ifdef __cplusplus
 extern "C" {
#endif

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)
/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup CAN
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup CAN_Exported_Types CAN Exported Types
  * @{
  */

/**
  * @brief  HAL State structures definition
  */
typedef enum
{
  HAL_CAN_STATE_RESET             = 0x00U,  /*!< CAN not yet initialized or disabled */
  HAL_CAN_STATE_READY             = 0x01U,  /*!< CAN initialized and ready for use   */
  HAL_CAN_STATE_BUSY              = 0x02U,  /*!< CAN process is ongoing              */
  HAL_CAN_STATE_BUSY_TX           = 0x12U,  /*!< CAN process is ongoing              */
  HAL_CAN_STATE_BUSY_RX0          = 0x22U,  /*!< CAN process is ongoing              */
  HAL_CAN_STATE_BUSY_RX1          = 0x32U,  /*!< CAN process is ongoing              */
  HAL_CAN_STATE_BUSY_TX_RX0       = 0x42U,  /*!< CAN process is ongoing              */
  HAL_CAN_STATE_BUSY_TX_RX1       = 0x52U,  /*!< CAN process is ongoing              */
  HAL_CAN_STATE_BUSY_RX0_RX1      = 0x62U,  /*!< CAN process is ongoing              */
  HAL_CAN_STATE_BUSY_TX_RX0_RX1   = 0x72U,  /*!< CAN process is ongoing              */
  HAL_CAN_STATE_TIMEOUT           = 0x03U,  /*!< CAN in Timeout state                */
  HAL_CAN_STATE_ERROR             = 0x04U   /*!< CAN error state                     */

}HAL_CAN_StateTypeDef;

/**
  * @brief  CAN init structure definition
  */
typedef struct
{
  uint32_t Prescaler;  /*!< Specifies the length of a time quantum.
                            This parameter must be a number between Min_Data = 1 and Max_Data = 1024 */

  uint32_t Mode;       /*!< Specifies the CAN operating mode.
                            This parameter can be a value of @ref CAN_operating_mode */

  uint32_t SJW;        /*!< Specifies the maximum number of time quanta
                            the CAN hardware is allowed to lengthen or
                            shorten a bit to perform resynchronization.
                            This parameter can be a value of @ref CAN_synchronisation_jump_width */

  uint32_t BS1;        /*!< Specifies the number of time quanta in Bit Segment 1.
                            This parameter can be a value of @ref CAN_time_quantum_in_bit_segment_1 */

  uint32_t BS2;        /*!< Specifies the number of time quanta in Bit Segment 2.
                            This parameter can be a value of @ref CAN_time_quantum_in_bit_segment_2 */

  uint32_t TTCM;       /*!< Enable or disable the time triggered communication mode.
                            This parameter can be set to ENABLE or DISABLE. */

  uint32_t ABOM;       /*!< Enable or disable the automatic bus-off management.
                            This parameter can be set to ENABLE or DISABLE */

  uint32_t AWUM;       /*!< Enable or disable the automatic wake-up mode.
                            This parameter can be set to ENABLE or DISABLE */

  uint32_t NART;       /*!< Enable or disable the non-automatic retransmission mode.
                            This parameter can be set to ENABLE or DISABLE */

  uint32_t RFLM;       /*!< Enable or disable the receive FIFO Locked mode.
                            This parameter can be set to ENABLE or DISABLE */

  uint32_t TXFP;       /*!< Enable or disable the transmit FIFO priority.
                            This parameter can be set to ENABLE or DISABLE */
}CAN_InitTypeDef;

/**
  * @brief  CAN filter configuration structure definition
  */
typedef struct
{
  uint32_t FilterIdHigh;          /*!< Specifies the filter identification number (MSBs for a 32-bit
                                       configuration, first one for a 16-bit configuration).
                                       This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF */

  uint32_t FilterIdLow;           /*!< Specifies the filter identification number (LSBs for a 32-bit
                                       configuration, second one for a 16-bit configuration).
                                       This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF */

  uint32_t FilterMaskIdHigh;      /*!< Specifies the filter mask number or identification number,
                                       according to the mode (MSBs for a 32-bit configuration,
                                       first one for a 16-bit configuration).
                                       This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF */

  uint32_t FilterMaskIdLow;       /*!< Specifies the filter mask number or identification number,
                                       according to the mode (LSBs for a 32-bit configuration,
                                       second one for a 16-bit configuration).
                                       This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF */

  uint32_t FilterFIFOAssignment;  /*!< Specifies the FIFO (0 or 1) which will be assigned to the filter.
                                       This parameter can be a value of @ref CAN_filter_FIFO */

  uint32_t FilterNumber;          /*!< Specifies the filter which will be initialized.
                                       This parameter must be a number between Min_Data = 0 and Max_Data = 27 */

  uint32_t FilterMode;            /*!< Specifies the filter mode to be initialized.
                                       This parameter can be a value of @ref CAN_filter_mode */

  uint32_t FilterScale;           /*!< Specifies the filter scale.
                                       This parameter can be a value of @ref CAN_filter_scale */

  uint32_t FilterActivation;      /*!< Enable or disable the filter.
                                       This parameter can be set to ENABLE or DISABLE. */

  uint32_t BankNumber;            /*!< Select the start slave bank filter.
                                       This parameter must be a number between Min_Data = 0 and Max_Data = 28 */

}CAN_FilterConfTypeDef;

/**
  * @brief  CAN Tx message structure definition
  */
typedef struct
{
  uint32_t StdId;    /*!< Specifies the standard identifier.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 0x7FF */

  uint32_t ExtId;    /*!< Specifies the extended identifier.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 0x1FFFFFFF */

  uint32_t IDE;      /*!< Specifies the type of identifier for the message that will be transmitted.
                          This parameter can be a value of @ref CAN_Identifier_Type */

  uint32_t RTR;      /*!< Specifies the type of frame for the message that will be transmitted.
                          This parameter can be a value of @ref CAN_remote_transmission_request */

  uint32_t DLC;      /*!< Specifies the length of the frame that will be transmitted.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 8 */

  uint8_t Data[8];   /*!< Contains the data to be transmitted.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 0xFF */

}CanTxMsgTypeDef;

/**
  * @brief  CAN Rx message structure definition
  */
typedef struct
{
  uint32_t StdId;       /*!< Specifies the standard identifier.
                             This parameter must be a number between Min_Data = 0 and Max_Data = 0x7FF */

  uint32_t ExtId;       /*!< Specifies the extended identifier.
                             This parameter must be a number between Min_Data = 0 and Max_Data = 0x1FFFFFFF */

  uint32_t IDE;         /*!< Specifies the type of identifier for the message that will be received.
                             This parameter can be a value of @ref CAN_Identifier_Type */

  uint32_t RTR;         /*!< Specifies the type of frame for the received message.
                             This parameter can be a value of @ref CAN_remote_transmission_request */

  uint32_t DLC;         /*!< Specifies the length of the frame that will be received.
                             This parameter must be a number between Min_Data = 0 and Max_Data = 8 */

  uint8_t Data[8];      /*!< Contains the data to be received.
                             This parameter must be a number between Min_Data = 0 and Max_Data = 0xFF */

  uint32_t FMI;         /*!< Specifies the index of the filter the message stored in the mailbox passes through.
                             This parameter must be a number between Min_Data = 0 and Max_Data = 0xFF */

  uint32_t FIFONumber;  /*!< Specifies the receive FIFO number.
                             This parameter can be CAN_FIFO0 or CAN_FIFO1 */

}CanRxMsgTypeDef;

/**
  * @brief  CAN handle Structure definition
  */
typedef struct
{
  CAN_TypeDef                 *Instance;  /*!< Register base address          */

  CAN_InitTypeDef             Init;       /*!< CAN required parameters        */

  CanTxMsgTypeDef*            pTxMsg;     /*!< Pointer to transmit structure  */

  CanRxMsgTypeDef*            pRxMsg;     /*!< Pointer to reception structure for RX FIFO0 msg */

  CanRxMsgTypeDef*            pRx1Msg;    /*!< Pointer to reception structure for RX FIFO1 msg */

  __IO HAL_CAN_StateTypeDef   State;      /*!< CAN communication state        */

  HAL_LockTypeDef             Lock;       /*!< CAN locking object             */

  __IO uint32_t               ErrorCode;  /*!< CAN Error code                 */

}CAN_HandleTypeDef;

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup CAN_Exported_Constants CAN Exported Constants
  * @{
  */

/** @defgroup CAN_Error_Code CAN Error Code
  * @{
  */
#define   HAL_CAN_ERROR_NONE      0x00000000U    /*!< No error             */
#define   HAL_CAN_ERROR_EWG       0x00000001U    /*!< EWG error            */
#define   HAL_CAN_ERROR_EPV       0x00000002U    /*!< EPV error            */
#define   HAL_CAN_ERROR_BOF       0x00000004U    /*!< BOF error            */
#define   HAL_CAN_ERROR_STF       0x00000008U    /*!< Stuff error          */
#define   HAL_CAN_ERROR_FOR       0x00000010U    /*!< Form error           */
#define   HAL_CAN_ERROR_ACK       0x00000020U    /*!< Acknowledgment error */
#define   HAL_CAN_ERROR_BR        0x00000040U    /*!< Bit recessive        */
#define   HAL_CAN_ERROR_BD        0x00000080U    /*!< LEC dominant         */
#define   HAL_CAN_ERROR_CRC       0x00000100U    /*!< LEC transfer error   */
#define   HAL_CAN_ERROR_FOV0      0x00000200U    /*!< FIFO0 overrun error  */
#define   HAL_CAN_ERROR_FOV1      0x00000400U    /*!< FIFO1 overrun error  */
#define   HAL_CAN_ERROR_TXFAIL    0x00000800U    /*!< Transmit failure     */
/**
  * @}
  */

/** @defgroup CAN_InitStatus CAN InitStatus
  * @{
  */
#define CAN_INITSTATUS_FAILED       ((uint8_t)0x00)  /*!< CAN initialization failed */
#define CAN_INITSTATUS_SUCCESS      ((uint8_t)0x01)  /*!< CAN initialization OK */
/**
  * @}
  */

/** @defgroup CAN_operating_mode CAN Operating Mode
  * @{
  */
#define CAN_MODE_NORMAL             0x00000000U                                /*!< Normal mode   */
#define CAN_MODE_LOOPBACK           ((uint32_t)CAN_BTR_LBKM)                   /*!< Loopback mode */
#define CAN_MODE_SILENT             ((uint32_t)CAN_BTR_SILM)                   /*!< Silent mode   */
#define CAN_MODE_SILENT_LOOPBACK    ((uint32_t)(CAN_BTR_LBKM | CAN_BTR_SILM))  /*!< Loopback combined with silent mode */
/**
  * @}
  */

/** @defgroup CAN_synchronisation_jump_width CAN Synchronisation Jump Width
  * @{
  */
#define CAN_SJW_1TQ                 0x00000000U                /*!< 1 time quantum */
#define CAN_SJW_2TQ                 ((uint32_t)CAN_BTR_SJW_0)  /*!< 2 time quantum */
#define CAN_SJW_3TQ                 ((uint32_t)CAN_BTR_SJW_1)  /*!< 3 time quantum */
#define CAN_SJW_4TQ                 ((uint32_t)CAN_BTR_SJW)    /*!< 4 time quantum */
/**
  * @}
  */

/** @defgroup CAN_time_quantum_in_bit_segment_1 CAN Time Quantum in bit segment 1
  * @{
  */
#define CAN_BS1_1TQ                 0x00000000U                                                  /*!< 1 time quantum  */
#define CAN_BS1_2TQ                 ((uint32_t)CAN_BTR_TS1_0)                                    /*!< 2 time quantum  */
#define CAN_BS1_3TQ                 ((uint32_t)CAN_BTR_TS1_1)                                    /*!< 3 time quantum  */
#define CAN_BS1_4TQ                 ((uint32_t)(CAN_BTR_TS1_1 | CAN_BTR_TS1_0))                  /*!< 4 time quantum  */
#define CAN_BS1_5TQ                 ((uint32_t)CAN_BTR_TS1_2)                                    /*!< 5 time quantum  */
#define CAN_BS1_6TQ                 ((uint32_t)(CAN_BTR_TS1_2 | CAN_BTR_TS1_0))                  /*!< 6 time quantum  */
#define CAN_BS1_7TQ                 ((uint32_t)(CAN_BTR_TS1_2 | CAN_BTR_TS1_1))                  /*!< 7 time quantum  */
#define CAN_BS1_8TQ                 ((uint32_t)(CAN_BTR_TS1_2 | CAN_BTR_TS1_1 | CAN_BTR_TS1_0))  /*!< 8 time quantum  */
#define CAN_BS1_9TQ                 ((uint32_t)CAN_BTR_TS1_3)                                    /*!< 9 time quantum  */
#define CAN_BS1_10TQ                ((uint32_t)(CAN_BTR_TS1_3 | CAN_BTR_TS1_0))                  /*!< 10 time quantum */
#define CAN_BS1_11TQ                ((uint32_t)(CAN_BTR_TS1_3 | CAN_BTR_TS1_1))                  /*!< 11 time quantum */
#define CAN_BS1_12TQ                ((uint32_t)(CAN_BTR_TS1_3 | CAN_BTR_TS1_1 | CAN_BTR_TS1_0))  /*!< 12 time quantum */
#define CAN_BS1_13TQ                ((uint32_t)(CAN_BTR_TS1_3 | CAN_BTR_TS1_2))                  /*!< 13 time quantum */
#define CAN_BS1_14TQ                ((uint32_t)(CAN_BTR_TS1_3 | CAN_BTR_TS1_2 | CAN_BTR_TS1_0))  /*!< 14 time quantum */
#define CAN_BS1_15TQ                ((uint32_t)(CAN_BTR_TS1_3 | CAN_BTR_TS1_2 | CAN_BTR_TS1_1))  /*!< 15 time quantum */
#define CAN_BS1_16TQ                ((uint32_t)CAN_BTR_TS1) /*!< 16 time quantum */
/**
  * @}
  */

/** @defgroup CAN_time_quantum_in_bit_segment_2 CAN Time Quantum in bit segment 2
  * @{
  */
#define CAN_BS2_1TQ                 0x00000000U                                  /*!< 1 time quantum */
#define CAN_BS2_2TQ                 ((uint32_t)CAN_BTR_TS2_0)                    /*!< 2 time quantum */
#define CAN_BS2_3TQ                 ((uint32_t)CAN_BTR_TS2_1)                    /*!< 3 time quantum */
#define CAN_BS2_4TQ                 ((uint32_t)(CAN_BTR_TS2_1 | CAN_BTR_TS2_0))  /*!< 4 time quantum */
#define CAN_BS2_5TQ                 ((uint32_t)CAN_BTR_TS2_2)                    /*!< 5 time quantum */
#define CAN_BS2_6TQ                 ((uint32_t)(CAN_BTR_TS2_2 | CAN_BTR_TS2_0))  /*!< 6 time quantum */
#define CAN_BS2_7TQ                 ((uint32_t)(CAN_BTR_TS2_2 | CAN_BTR_TS2_1))  /*!< 7 time quantum */
#define CAN_BS2_8TQ                 ((uint32_t)CAN_BTR_TS2)                      /*!< 8 time quantum */
/**
  * @}
  */

/** @defgroup CAN_filter_mode  CAN Filter Mode
  * @{
  */
#define CAN_FILTERMODE_IDMASK       ((uint8_t)0x00)  /*!< Identifier mask mode */
#define CAN_FILTERMODE_IDLIST       ((uint8_t)0x01)  /*!< Identifier list mode */
/**
  * @}
  */

/** @defgroup CAN_filter_scale CAN Filter Scale
  * @{
  */
#define CAN_FILTERSCALE_16BIT       ((uint8_t)0x00)  /*!< Two 16-bit filters */
#define CAN_FILTERSCALE_32BIT       ((uint8_t)0x01)  /*!< One 32-bit filter  */
/**
  * @}
  */

/** @defgroup CAN_filter_FIFO CAN Filter FIFO
  * @{
  */
#define CAN_FILTER_FIFO0             ((uint8_t)0x00)  /*!< Filter FIFO 0 assignment for filter x */
#define CAN_FILTER_FIFO1             ((uint8_t)0x01)  /*!< Filter FIFO 1 assignment for filter x */
/**
  * @}
  */

/** @defgroup CAN_Identifier_Type CAN Identifier Type
  * @{
  */
#define CAN_ID_STD                  0x00000000U  /*!< Standard Id */
#define CAN_ID_EXT                  0x00000004U  /*!< Extended Id */
/**
  * @}
  */

/** @defgroup CAN_remote_transmission_request CAN Remote Transmission Request
  * @{
  */
#define CAN_RTR_DATA                0x00000000U  /*!< Data frame */
#define CAN_RTR_REMOTE              0x00000002U  /*!< Remote frame */
/**
  * @}
  */

/** @defgroup CAN_receive_FIFO_number_constants CAN Receive FIFO Number Constants
  * @{
  */
#define CAN_FIFO0                   ((uint8_t)0x00)  /*!< CAN FIFO 0 used to receive */
#define CAN_FIFO1                   ((uint8_t)0x01)  /*!< CAN FIFO 1 used to receive */
/**
  * @}
  */

/** @defgroup CAN_flags CAN Flags
  * @{
  */
/* If the flag is 0x3XXXXXXX, it means that it can be used with CAN_GetFlagStatus()
   and CAN_ClearFlag() functions. */
/* If the flag is 0x1XXXXXXX, it means that it can only be used with
   CAN_GetFlagStatus() function.  */

/* Transmit Flags */
#define CAN_FLAG_RQCP0             0x00000500U  /*!< Request MailBox0 flag         */
#define CAN_FLAG_RQCP1             0x00000508U  /*!< Request MailBox1 flag         */
#define CAN_FLAG_RQCP2             0x00000510U  /*!< Request MailBox2 flag         */
#define CAN_FLAG_TXOK0             0x00000501U  /*!< Transmission OK MailBox0 flag */
#define CAN_FLAG_TXOK1             0x00000509U  /*!< Transmission OK MailBox1 flag */
#define CAN_FLAG_TXOK2             0x00000511U  /*!< Transmission OK MailBox2 flag */
#define CAN_FLAG_TME0              0x0000051AU  /*!< Transmit mailbox 0 empty flag */
#define CAN_FLAG_TME1              0x0000051BU  /*!< Transmit mailbox 0 empty flag */
#define CAN_FLAG_TME2              0x0000051CU  /*!< Transmit mailbox 0 empty flag */

/* Receive Flags */
#define CAN_FLAG_FF0               0x00000203U  /*!< FIFO 0 Full flag    */
#define CAN_FLAG_FOV0              0x00000204U  /*!< FIFO 0 Overrun flag */

#define CAN_FLAG_FF1               0x00000403U  /*!< FIFO 1 Full flag    */
#define CAN_FLAG_FOV1              0x00000404U  /*!< FIFO 1 Overrun flag */

/* Operating Mode Flags */
#define CAN_FLAG_INAK              0x00000100U  /*!<  Initialization acknowledge flag */
#define CAN_FLAG_SLAK              0x00000101U  /*!< Sleep acknowledge flag */
#define CAN_FLAG_ERRI              0x00000102U  /*!<  Error flag */
#define CAN_FLAG_WKU               0x00000103U  /*!< Wake up flag           */
#define CAN_FLAG_SLAKI             0x00000104U  /*!< Sleep acknowledge flag */

/* @note When SLAK interrupt is disabled (SLKIE=0), no polling on SLAKI is possible.
         In this case the SLAK bit can be polled.*/

/* Error Flags */
#define CAN_FLAG_EWG               0x00000300U  /*!< Error warning flag   */
#define CAN_FLAG_EPV               0x00000301U  /*!< Error passive flag   */
#define CAN_FLAG_BOF               0x00000302U  /*!< Bus-Off flag         */
/**
  * @}
  */

/** @defgroup CAN_Interrupts CAN Interrupts
  * @{
  */
#define CAN_IT_TME                  ((uint32_t)CAN_IER_TMEIE)   /*!< Transmit mailbox empty interrupt */

/* Receive Interrupts */
#define CAN_IT_FMP0                 ((uint32_t)CAN_IER_FMPIE0)  /*!< FIFO 0 message pending interrupt */
#define CAN_IT_FF0                  ((uint32_t)CAN_IER_FFIE0)   /*!< FIFO 0 full interrupt            */
#define CAN_IT_FOV0                 ((uint32_t)CAN_IER_FOVIE0)  /*!< FIFO 0 overrun interrupt         */
#define CAN_IT_FMP1                 ((uint32_t)CAN_IER_FMPIE1)  /*!< FIFO 1 message pending interrupt */
#define CAN_IT_FF1                  ((uint32_t)CAN_IER_FFIE1)   /*!< FIFO 1 full interrupt            */
#define CAN_IT_FOV1                 ((uint32_t)CAN_IER_FOVIE1)  /*!< FIFO 1 overrun interrupt         */

/* Operating Mode Interrupts */
#define CAN_IT_WKU                  ((uint32_t)CAN_IER_WKUIE)  /*!< Wake-up interrupt           */
#define CAN_IT_SLK                  ((uint32_t)CAN_IER_SLKIE)  /*!< Sleep acknowledge interrupt */

/* Error Interrupts */
#define CAN_IT_EWG                  ((uint32_t)CAN_IER_EWGIE) /*!< Error warning interrupt   */
#define CAN_IT_EPV                  ((uint32_t)CAN_IER_EPVIE) /*!< Error passive interrupt   */
#define CAN_IT_BOF                  ((uint32_t)CAN_IER_BOFIE) /*!< Bus-off interrupt         */
#define CAN_IT_LEC                  ((uint32_t)CAN_IER_LECIE) /*!< Last error code interrupt */
#define CAN_IT_ERR                  ((uint32_t)CAN_IER_ERRIE) /*!< Error Interrupt           */
/**
  * @}
  */

/** @defgroup CAN_Mailboxes_Definition CAN Mailboxes Definition
  * @{
  */
#define CAN_TXMAILBOX_0   ((uint8_t)0x00)
#define CAN_TXMAILBOX_1   ((uint8_t)0x01)
#define CAN_TXMAILBOX_2   ((uint8_t)0x02)
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup CAN_Exported_Macros CAN Exported Macros
  * @{
  */

/** @brief Reset CAN handle state
  * @param  __HANDLE__ specifies the CAN Handle.
  * @retval None
  */
#define __HAL_CAN_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_CAN_STATE_RESET)

/**
  * @brief  Enable the specified CAN interrupts.
  * @param  __HANDLE__ CAN handle
  * @param  __INTERRUPT__ CAN Interrupt
  * @retval None
  */
#define __HAL_CAN_ENABLE_IT(__HANDLE__, __INTERRUPT__) (((__HANDLE__)->Instance->IER) |= (__INTERRUPT__))

/**
  * @brief  Disable the specified CAN interrupts.
  * @param  __HANDLE__ CAN handle
  * @param  __INTERRUPT__ CAN Interrupt
  * @retval None
  */
#define __HAL_CAN_DISABLE_IT(__HANDLE__, __INTERRUPT__) (((__HANDLE__)->Instance->IER) &= ~(__INTERRUPT__))

/**
  * @brief  Return the number of pending received messages.
  * @param  __HANDLE__ CAN handle
  * @param  __FIFONUMBER__ Receive FIFO number, CAN_FIFO0 or CAN_FIFO1.
  * @retval The number of pending message.
  */
#define __HAL_CAN_MSG_PENDING(__HANDLE__, __FIFONUMBER__) (((__FIFONUMBER__) == CAN_FIFO0)? \
((uint8_t)((__HANDLE__)->Instance->RF0R&0x03U)) : ((uint8_t)((__HANDLE__)->Instance->RF1R & 0x03U)))

/** @brief  Check whether the specified CAN flag is set or not.
  * @param  __HANDLE__ CAN Handle
  * @param  __FLAG__ specifies the flag to check.
  *         This parameter can be one of the following values:
  *            @arg CAN_TSR_RQCP0: Request MailBox0 Flag
  *            @arg CAN_TSR_RQCP1: Request MailBox1 Flag
  *            @arg CAN_TSR_RQCP2: Request MailBox2 Flag
  *            @arg CAN_FLAG_TXOK0: Transmission OK MailBox0 Flag
  *            @arg CAN_FLAG_TXOK1: Transmission OK MailBox1 Flag
  *            @arg CAN_FLAG_TXOK2: Transmission OK MailBox2 Flag
  *            @arg CAN_FLAG_TME0: Transmit mailbox 0 empty Flag
  *            @arg CAN_FLAG_TME1: Transmit mailbox 1 empty Flag
  *            @arg CAN_FLAG_TME2: Transmit mailbox 2 empty Flag
  *            @arg CAN_FLAG_FMP0: FIFO 0 Message Pending Flag
  *            @arg CAN_FLAG_FF0: FIFO 0 Full Flag
  *            @arg CAN_FLAG_FOV0: FIFO 0 Overrun Flag
  *            @arg CAN_FLAG_FMP1: FIFO 1 Message Pending Flag
  *            @arg CAN_FLAG_FF1: FIFO 1 Full Flag
  *            @arg CAN_FLAG_FOV1: FIFO 1 Overrun Flag
  *            @arg CAN_FLAG_WKU: Wake up Flag
  *            @arg CAN_FLAG_SLAK: Sleep acknowledge Flag
  *            @arg CAN_FLAG_SLAKI: Sleep acknowledge Flag
  *            @arg CAN_FLAG_EWG: Error Warning Flag
  *            @arg CAN_FLAG_EPV: Error Passive Flag
  *            @arg CAN_FLAG_BOF: Bus-Off Flag
  * @retval The new state of __FLAG__ (TRUE or FALSE).
  */
#define __HAL_CAN_GET_FLAG(__HANDLE__, __FLAG__) \
((((__FLAG__) >> 8U) == 5U)? ((((__HANDLE__)->Instance->TSR) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
 (((__FLAG__) >> 8U) == 2U)? ((((__HANDLE__)->Instance->RF0R) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
 (((__FLAG__) >> 8U) == 4U)? ((((__HANDLE__)->Instance->RF1R) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
 (((__FLAG__) >> 8U) == 1U)? ((((__HANDLE__)->Instance->MSR) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
 ((((__HANDLE__)->Instance->ESR) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))))

/** @brief  Clear the specified CAN pending flag.
  * @param  __HANDLE__ CAN Handle.
  * @param  __FLAG__ specifies the flag to check.
  *         This parameter can be one of the following values:
  *            @arg CAN_TSR_RQCP0: Request MailBox0 Flag
  *            @arg CAN_TSR_RQCP1: Request MailBox1 Flag
  *            @arg CAN_TSR_RQCP2: Request MailBox2 Flag
  *            @arg CAN_FLAG_TXOK0: Transmission OK MailBox0 Flag
  *            @arg CAN_FLAG_TXOK1: Transmission OK MailBox1 Flag
  *            @arg CAN_FLAG_TXOK2: Transmission OK MailBox2 Flag
  *            @arg CAN_FLAG_TME0: Transmit mailbox 0 empty Flag
  *            @arg CAN_FLAG_TME1: Transmit mailbox 1 empty Flag
  *            @arg CAN_FLAG_TME2: Transmit mailbox 2 empty Flag
  *            @arg CAN_FLAG_FMP0: FIFO 0 Message Pending Flag
  *            @arg CAN_FLAG_FF0: FIFO 0 Full Flag
  *            @arg CAN_FLAG_FOV0: FIFO 0 Overrun Flag
  *            @arg CAN_FLAG_FMP1: FIFO 1 Message Pending Flag
  *            @arg CAN_FLAG_FF1: FIFO 1 Full Flag
  *            @arg CAN_FLAG_FOV1: FIFO 1 Overrun Flag
  *            @arg CAN_FLAG_WKU: Wake up Flag
  *            @arg CAN_FLAG_SLAK: Sleep acknowledge Flag
  *            @arg CAN_FLAG_SLAKI: Sleep acknowledge Flag
  * @retval The new state of __FLAG__ (TRUE or FALSE).
  */
#define __HAL_CAN_CLEAR_FLAG(__HANDLE__, __FLAG__) \
((((__FLAG__) >> 8U) == 5U)? (((__HANDLE__)->Instance->TSR) = (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
 (((__FLAG__) >> 8U) == 2U)? (((__HANDLE__)->Instance->RF0R) = (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
 (((__FLAG__) >> 8U) == 4U)? (((__HANDLE__)->Instance->RF1R) = (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
 (((__HANDLE__)->Instance->MSR) = ((uint32_t)1U << ((__FLAG__) & CAN_FLAG_MASK))))

/** @brief  Check if the specified CAN interrupt source is enabled or disabled.
  * @param  __HANDLE__ CAN Handle
  * @param  __INTERRUPT__ specifies the CAN interrupt source to check.
  *          This parameter can be one of the following values:
  *             @arg CAN_IT_TME: Transmit mailbox empty interrupt enable
  *             @arg CAN_IT_FMP0: FIFO0 message pending interrupt enable
  *             @arg CAN_IT_FMP1: FIFO1 message pending interrupt enable
  * @retval The new state of __IT__ (TRUE or FALSE).
  */
#define __HAL_CAN_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) ((((__HANDLE__)->Instance->IER & (__INTERRUPT__)) == (__INTERRUPT__)) ? SET : RESET)

/**
  * @brief  Check the transmission status of a CAN Frame.
  * @param  __HANDLE__ CAN Handle
  * @param  __TRANSMITMAILBOX__ the number of the mailbox that is used for transmission.
  * @retval The new status of transmission  (TRUE or FALSE).
  */
#define __HAL_CAN_TRANSMIT_STATUS(__HANDLE__, __TRANSMITMAILBOX__)\
(((__TRANSMITMAILBOX__) == CAN_TXMAILBOX_0)? ((((__HANDLE__)->Instance->TSR) & (CAN_TSR_RQCP0 | CAN_TSR_TXOK0 | CAN_TSR_TME0)) == (CAN_TSR_RQCP0 | CAN_TSR_TXOK0 | CAN_TSR_TME0)) :\
 ((__TRANSMITMAILBOX__) == CAN_TXMAILBOX_1)? ((((__HANDLE__)->Instance->TSR) & (CAN_TSR_RQCP1 | CAN_TSR_TXOK1 | CAN_TSR_TME1)) == (CAN_TSR_RQCP1 | CAN_TSR_TXOK1 | CAN_TSR_TME1)) :\
 ((((__HANDLE__)->Instance->TSR) & (CAN_TSR_RQCP2 | CAN_TSR_TXOK2 | CAN_TSR_TME2)) == (CAN_TSR_RQCP2 | CAN_TSR_TXOK2 | CAN_TSR_TME2)))

/**
  * @brief  Release the specified receive FIFO.
  * @param  __HANDLE__ CAN handle
  * @param  __FIFONUMBER__ Receive FIFO number, CAN_FIFO0 or CAN_FIFO1.
  * @retval None
  */
#define __HAL_CAN_FIFO_RELEASE(__HANDLE__, __FIFONUMBER__) (((__FIFONUMBER__) == CAN_FIFO0)? \
((__HANDLE__)->Instance->RF0R = CAN_RF0R_RFOM0) : ((__HANDLE__)->Instance->RF1R = CAN_RF1R_RFOM1))

/**
  * @brief  Cancel a transmit request.
  * @param  __HANDLE__ CAN Handle
  * @param  __TRANSMITMAILBOX__ the number of the mailbox that is used for transmission.
  * @retval None
  */
#define __HAL_CAN_CANCEL_TRANSMIT(__HANDLE__, __TRANSMITMAILBOX__)\
(((__TRANSMITMAILBOX__) == CAN_TXMAILBOX_0)? ((__HANDLE__)->Instance->TSR = CAN_TSR_ABRQ0) :\
 ((__TRANSMITMAILBOX__) == CAN_TXMAILBOX_1)? ((__HANDLE__)->Instance->TSR = CAN_TSR_ABRQ1) :\
 ((__HANDLE__)->Instance->TSR = CAN_TSR_ABRQ2))

/**
  * @brief  Enable or disable the DBG Freeze for CAN.
  * @param  __HANDLE__ CAN Handle
  * @param  __NEWSTATE__ new state of the CAN peripheral.
  *          This parameter can be: ENABLE (CAN reception/transmission is frozen
  *          during debug. Reception FIFOs can still be accessed/controlled normally)
  *          or DISABLE (CAN is working during debug).
  * @retval None
  */
#define __HAL_CAN_DBG_FREEZE(__HANDLE__, __NEWSTATE__) (((__NEWSTATE__) == ENABLE)? \
((__HANDLE__)->Instance->MCR |= CAN_MCR_DBF) : ((__HANDLE__)->Instance->MCR &= ~CAN_MCR_DBF))

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup CAN_Exported_Functions
  * @{
  */

/** @addtogroup CAN_Exported_Functions_Group1
  * @{
  */
/* Initialization/de-initialization functions ***********************************/
HAL_StatusTypeDef HAL_CAN_Init(CAN_HandleTypeDef* hcan);
HAL_StatusTypeDef HAL_CAN_ConfigFilter(CAN_HandleTypeDef* hcan, CAN_FilterConfTypeDef* sFilterConfig);
HAL_StatusTypeDef HAL_CAN_DeInit(CAN_HandleTypeDef* hcan);
void HAL_CAN_MspInit(CAN_HandleTypeDef* hcan);
void HAL_CAN_MspDeInit(CAN_HandleTypeDef* hcan);
/**
  * @}
  */

/** @addtogroup CAN_Exported_Functions_Group2
  * @{
  */
/* I/O operation functions ******************************************************/
HAL_StatusTypeDef HAL_CAN_Transmit(CAN_HandleTypeDef *hcan, uint32_t Timeout);
HAL_StatusTypeDef HAL_CAN_Transmit_IT(CAN_HandleTypeDef *hcan);
HAL_StatusTypeDef HAL_CAN_Receive(CAN_HandleTypeDef *hcan, uint8_t FIFONumber, uint32_t Timeout);
HAL_StatusTypeDef HAL_CAN_Receive_IT(CAN_HandleTypeDef *hcan, uint8_t FIFONumber);
HAL_StatusTypeDef HAL_CAN_Sleep(CAN_HandleTypeDef *hcan);
HAL_StatusTypeDef HAL_CAN_WakeUp(CAN_HandleTypeDef *hcan);
void HAL_CAN_IRQHandler(CAN_HandleTypeDef* hcan);
void HAL_CAN_TxCpltCallback(CAN_HandleTypeDef* hcan);
void HAL_CAN_RxCpltCallback(CAN_HandleTypeDef* hcan);
void HAL_CAN_ErrorCallback(CAN_HandleTypeDef *hcan);
/**
  * @}
  */

/** @addtogroup CAN_Exported_Functions_Group3
  * @{
  */
/* Peripheral State functions ***************************************************/
uint32_t HAL_CAN_GetError(CAN_HandleTypeDef *hcan);
HAL_CAN_StateTypeDef HAL_CAN_GetState(CAN_HandleTypeDef* hcan);
/**
  * @}
  */

/**
  * @}
  */

/* Private types -------------------------------------------------------------*/
/** @defgroup CAN_Private_Types CAN Private Types
  * @{
  */

/**
  * @}
  */

/* Private variables ---------------------------------------------------------*/
/** @defgroup CAN_Private_Variables CAN Private Variables
  * @{
  */

/**
  * @}
  */ 

/* Private constants ---------------------------------------------------------*/
/** @defgroup CAN_Private_Constants CAN Private Constants
  * @{
  */
#define CAN_TXSTATUS_NOMAILBOX      ((uint8_t)0x04)  /*!< CAN cell did not provide CAN_TxStatus_NoMailBox */
#define CAN_FLAG_MASK               0x000000FFU
/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup CAN_Private_Macros CAN Private Macros
  * @{
  */
#define IS_CAN_MODE(MODE) (((MODE) == CAN_MODE_NORMAL) || \
                           ((MODE) == CAN_MODE_LOOPBACK)|| \
                           ((MODE) == CAN_MODE_SILENT) || \
                           ((MODE) == CAN_MODE_SILENT_LOOPBACK))
#define IS_CAN_SJW(SJW) (((SJW) == CAN_SJW_1TQ) || ((SJW) == CAN_SJW_2TQ)|| \
                         ((SJW) == CAN_SJW_3TQ) || ((SJW) == CAN_SJW_4TQ))
#define IS_CAN_BS1(BS1) ((BS1) <= CAN_BS1_16TQ)
#define IS_CAN_BS2(BS2) ((BS2) <= CAN_BS2_8TQ)
#define IS_CAN_PRESCALER(PRESCALER) (((PRESCALER) >= 1U) && ((PRESCALER) <= 1024U))
#define IS_CAN_FILTER_NUMBER(NUMBER) ((NUMBER) <= 27U)
#define IS_CAN_FILTER_MODE(MODE) (((MODE) == CAN_FILTERMODE_IDMASK) || \
                                  ((MODE) == CAN_FILTERMODE_IDLIST))
#define IS_CAN_FILTER_SCALE(SCALE) (((SCALE) == CAN_FILTERSCALE_16BIT) || \
                                    ((SCALE) == CAN_FILTERSCALE_32BIT))
#define IS_CAN_FILTER_FIFO(FIFO) (((FIFO) == CAN_FILTER_FIFO0) || \
                                  ((FIFO) == CAN_FILTER_FIFO1))
#define IS_CAN_BANKNUMBER(BANKNUMBER) ((BANKNUMBER) <= 28U)

#define IS_CAN_TRANSMITMAILBOX(TRANSMITMAILBOX) ((TRANSMITMAILBOX) <= ((uint8_t)0x02))
#define IS_CAN_STDID(STDID)   ((STDID) <= ((uint32_t)0x7FFU))
#define IS_CAN_EXTID(EXTID)   ((EXTID) <= 0x1FFFFFFFU)
#define IS_CAN_DLC(DLC)       ((DLC) <= ((uint8_t)0x08))

#define IS_CAN_IDTYPE(IDTYPE)  (((IDTYPE) == CAN_ID_STD) || \
                                ((IDTYPE) == CAN_ID_EXT))
#define IS_CAN_RTR(RTR) (((RTR) == CAN_RTR_DATA) || ((RTR) == CAN_RTR_REMOTE))
#define IS_CAN_FIFO(FIFO) (((FIFO) == CAN_FIFO0) || ((FIFO) == CAN_FIFO1))

/**
  * @}
  */

/* Private functions ---------------------------------------------------------*/
/** @defgroup CAN_Private_Functions CAN Private Functions
  * @{
  */

/**
  * @}
  */

#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx ||\
          STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

/**
  * @}
  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_CAN_LEGACY_H */


/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
