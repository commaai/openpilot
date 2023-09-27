/**
  ******************************************************************************
  * @file    stm32f4xx_hal_can.h
  * @author  MCD Application Team
  * @brief   Header file of CAN HAL module.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef STM32F4xx_HAL_CAN_H
#define STM32F4xx_HAL_CAN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

#if defined (CAN1)
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
  HAL_CAN_STATE_LISTENING         = 0x02U,  /*!< CAN receive process is ongoing      */
  HAL_CAN_STATE_SLEEP_PENDING     = 0x03U,  /*!< CAN sleep request is pending        */
  HAL_CAN_STATE_SLEEP_ACTIVE      = 0x04U,  /*!< CAN sleep mode is active            */
  HAL_CAN_STATE_ERROR             = 0x05U   /*!< CAN error state                     */

} HAL_CAN_StateTypeDef;

/**
  * @brief  CAN init structure definition
  */
typedef struct
{
  uint32_t Prescaler;                  /*!< Specifies the length of a time quantum.
                                            This parameter must be a number between Min_Data = 1 and Max_Data = 1024. */

  uint32_t Mode;                       /*!< Specifies the CAN operating mode.
                                            This parameter can be a value of @ref CAN_operating_mode */

  uint32_t SyncJumpWidth;              /*!< Specifies the maximum number of time quanta the CAN hardware
                                            is allowed to lengthen or shorten a bit to perform resynchronization.
                                            This parameter can be a value of @ref CAN_synchronisation_jump_width */

  uint32_t TimeSeg1;                   /*!< Specifies the number of time quanta in Bit Segment 1.
                                            This parameter can be a value of @ref CAN_time_quantum_in_bit_segment_1 */

  uint32_t TimeSeg2;                   /*!< Specifies the number of time quanta in Bit Segment 2.
                                            This parameter can be a value of @ref CAN_time_quantum_in_bit_segment_2 */

  FunctionalState TimeTriggeredMode;   /*!< Enable or disable the time triggered communication mode.
                                            This parameter can be set to ENABLE or DISABLE. */

  FunctionalState AutoBusOff;          /*!< Enable or disable the automatic bus-off management.
                                            This parameter can be set to ENABLE or DISABLE. */

  FunctionalState AutoWakeUp;          /*!< Enable or disable the automatic wake-up mode.
                                            This parameter can be set to ENABLE or DISABLE. */

  FunctionalState AutoRetransmission;  /*!< Enable or disable the non-automatic retransmission mode.
                                            This parameter can be set to ENABLE or DISABLE. */

  FunctionalState ReceiveFifoLocked;   /*!< Enable or disable the Receive FIFO Locked mode.
                                            This parameter can be set to ENABLE or DISABLE. */

  FunctionalState TransmitFifoPriority;/*!< Enable or disable the transmit FIFO priority.
                                            This parameter can be set to ENABLE or DISABLE. */

} CAN_InitTypeDef;

/**
  * @brief  CAN filter configuration structure definition
  */
typedef struct
{
  uint32_t FilterIdHigh;          /*!< Specifies the filter identification number (MSBs for a 32-bit
                                       configuration, first one for a 16-bit configuration).
                                       This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF. */

  uint32_t FilterIdLow;           /*!< Specifies the filter identification number (LSBs for a 32-bit
                                       configuration, second one for a 16-bit configuration).
                                       This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF. */

  uint32_t FilterMaskIdHigh;      /*!< Specifies the filter mask number or identification number,
                                       according to the mode (MSBs for a 32-bit configuration,
                                       first one for a 16-bit configuration).
                                       This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF. */

  uint32_t FilterMaskIdLow;       /*!< Specifies the filter mask number or identification number,
                                       according to the mode (LSBs for a 32-bit configuration,
                                       second one for a 16-bit configuration).
                                       This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0xFFFF. */

  uint32_t FilterFIFOAssignment;  /*!< Specifies the FIFO (0 or 1U) which will be assigned to the filter.
                                       This parameter can be a value of @ref CAN_filter_FIFO */

  uint32_t FilterBank;            /*!< Specifies the filter bank which will be initialized.
                                       For single CAN instance(14 dedicated filter banks),
                                       this parameter must be a number between Min_Data = 0 and Max_Data = 13.
                                       For dual CAN instances(28 filter banks shared),
                                       this parameter must be a number between Min_Data = 0 and Max_Data = 27. */

  uint32_t FilterMode;            /*!< Specifies the filter mode to be initialized.
                                       This parameter can be a value of @ref CAN_filter_mode */

  uint32_t FilterScale;           /*!< Specifies the filter scale.
                                       This parameter can be a value of @ref CAN_filter_scale */

  uint32_t FilterActivation;      /*!< Enable or disable the filter.
                                       This parameter can be a value of @ref CAN_filter_activation */

  uint32_t SlaveStartFilterBank;  /*!< Select the start filter bank for the slave CAN instance.
                                       For single CAN instances, this parameter is meaningless.
                                       For dual CAN instances, all filter banks with lower index are assigned to master
                                       CAN instance, whereas all filter banks with greater index are assigned to slave
                                       CAN instance.
                                       This parameter must be a number between Min_Data = 0 and Max_Data = 27. */

} CAN_FilterTypeDef;

/**
  * @brief  CAN Tx message header structure definition
  */
typedef struct
{
  uint32_t StdId;    /*!< Specifies the standard identifier.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 0x7FF. */

  uint32_t ExtId;    /*!< Specifies the extended identifier.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 0x1FFFFFFF. */

  uint32_t IDE;      /*!< Specifies the type of identifier for the message that will be transmitted.
                          This parameter can be a value of @ref CAN_identifier_type */

  uint32_t RTR;      /*!< Specifies the type of frame for the message that will be transmitted.
                          This parameter can be a value of @ref CAN_remote_transmission_request */

  uint32_t DLC;      /*!< Specifies the length of the frame that will be transmitted.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 8. */

  FunctionalState TransmitGlobalTime; /*!< Specifies whether the timestamp counter value captured on start
                          of frame transmission, is sent in DATA6 and DATA7 replacing pData[6] and pData[7].
                          @note: Time Triggered Communication Mode must be enabled.
                          @note: DLC must be programmed as 8 bytes, in order these 2 bytes are sent.
                          This parameter can be set to ENABLE or DISABLE. */

} CAN_TxHeaderTypeDef;

/**
  * @brief  CAN Rx message header structure definition
  */
typedef struct
{
  uint32_t StdId;    /*!< Specifies the standard identifier.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 0x7FF. */

  uint32_t ExtId;    /*!< Specifies the extended identifier.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 0x1FFFFFFF. */

  uint32_t IDE;      /*!< Specifies the type of identifier for the message that will be transmitted.
                          This parameter can be a value of @ref CAN_identifier_type */

  uint32_t RTR;      /*!< Specifies the type of frame for the message that will be transmitted.
                          This parameter can be a value of @ref CAN_remote_transmission_request */

  uint32_t DLC;      /*!< Specifies the length of the frame that will be transmitted.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 8. */

  uint32_t Timestamp; /*!< Specifies the timestamp counter value captured on start of frame reception.
                          @note: Time Triggered Communication Mode must be enabled.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 0xFFFF. */

  uint32_t FilterMatchIndex; /*!< Specifies the index of matching acceptance filter element.
                          This parameter must be a number between Min_Data = 0 and Max_Data = 0xFF. */

} CAN_RxHeaderTypeDef;

/**
  * @brief  CAN handle Structure definition
  */
typedef struct __CAN_HandleTypeDef
{
  CAN_TypeDef                 *Instance;                 /*!< Register base address */

  CAN_InitTypeDef             Init;                      /*!< CAN required parameters */

  __IO HAL_CAN_StateTypeDef   State;                     /*!< CAN communication state */

  __IO uint32_t               ErrorCode;                 /*!< CAN Error code.
                                                              This parameter can be a value of @ref CAN_Error_Code */

#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
  void (* TxMailbox0CompleteCallback)(struct __CAN_HandleTypeDef *hcan);/*!< CAN Tx Mailbox 0 complete callback    */
  void (* TxMailbox1CompleteCallback)(struct __CAN_HandleTypeDef *hcan);/*!< CAN Tx Mailbox 1 complete callback    */
  void (* TxMailbox2CompleteCallback)(struct __CAN_HandleTypeDef *hcan);/*!< CAN Tx Mailbox 2 complete callback    */
  void (* TxMailbox0AbortCallback)(struct __CAN_HandleTypeDef *hcan);   /*!< CAN Tx Mailbox 0 abort callback       */
  void (* TxMailbox1AbortCallback)(struct __CAN_HandleTypeDef *hcan);   /*!< CAN Tx Mailbox 1 abort callback       */
  void (* TxMailbox2AbortCallback)(struct __CAN_HandleTypeDef *hcan);   /*!< CAN Tx Mailbox 2 abort callback       */
  void (* RxFifo0MsgPendingCallback)(struct __CAN_HandleTypeDef *hcan); /*!< CAN Rx FIFO 0 msg pending callback    */
  void (* RxFifo0FullCallback)(struct __CAN_HandleTypeDef *hcan);       /*!< CAN Rx FIFO 0 full callback           */
  void (* RxFifo1MsgPendingCallback)(struct __CAN_HandleTypeDef *hcan); /*!< CAN Rx FIFO 1 msg pending callback    */
  void (* RxFifo1FullCallback)(struct __CAN_HandleTypeDef *hcan);       /*!< CAN Rx FIFO 1 full callback           */
  void (* SleepCallback)(struct __CAN_HandleTypeDef *hcan);             /*!< CAN Sleep callback                    */
  void (* WakeUpFromRxMsgCallback)(struct __CAN_HandleTypeDef *hcan);   /*!< CAN Wake Up from Rx msg callback      */
  void (* ErrorCallback)(struct __CAN_HandleTypeDef *hcan);             /*!< CAN Error callback                    */

  void (* MspInitCallback)(struct __CAN_HandleTypeDef *hcan);           /*!< CAN Msp Init callback                 */
  void (* MspDeInitCallback)(struct __CAN_HandleTypeDef *hcan);         /*!< CAN Msp DeInit callback               */

#endif /* (USE_HAL_CAN_REGISTER_CALLBACKS) */
} CAN_HandleTypeDef;

#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
/**
  * @brief  HAL CAN common Callback ID enumeration definition
  */
typedef enum
{
  HAL_CAN_TX_MAILBOX0_COMPLETE_CB_ID       = 0x00U,    /*!< CAN Tx Mailbox 0 complete callback ID         */
  HAL_CAN_TX_MAILBOX1_COMPLETE_CB_ID       = 0x01U,    /*!< CAN Tx Mailbox 1 complete callback ID         */
  HAL_CAN_TX_MAILBOX2_COMPLETE_CB_ID       = 0x02U,    /*!< CAN Tx Mailbox 2 complete callback ID         */
  HAL_CAN_TX_MAILBOX0_ABORT_CB_ID          = 0x03U,    /*!< CAN Tx Mailbox 0 abort callback ID            */
  HAL_CAN_TX_MAILBOX1_ABORT_CB_ID          = 0x04U,    /*!< CAN Tx Mailbox 1 abort callback ID            */
  HAL_CAN_TX_MAILBOX2_ABORT_CB_ID          = 0x05U,    /*!< CAN Tx Mailbox 2 abort callback ID            */
  HAL_CAN_RX_FIFO0_MSG_PENDING_CB_ID       = 0x06U,    /*!< CAN Rx FIFO 0 message pending callback ID     */
  HAL_CAN_RX_FIFO0_FULL_CB_ID              = 0x07U,    /*!< CAN Rx FIFO 0 full callback ID                */
  HAL_CAN_RX_FIFO1_MSG_PENDING_CB_ID       = 0x08U,    /*!< CAN Rx FIFO 1 message pending callback ID     */
  HAL_CAN_RX_FIFO1_FULL_CB_ID              = 0x09U,    /*!< CAN Rx FIFO 1 full callback ID                */
  HAL_CAN_SLEEP_CB_ID                      = 0x0AU,    /*!< CAN Sleep callback ID                         */
  HAL_CAN_WAKEUP_FROM_RX_MSG_CB_ID         = 0x0BU,    /*!< CAN Wake Up from Rx msg callback ID          */
  HAL_CAN_ERROR_CB_ID                      = 0x0CU,    /*!< CAN Error callback ID                         */

  HAL_CAN_MSPINIT_CB_ID                    = 0x0DU,    /*!< CAN MspInit callback ID                       */
  HAL_CAN_MSPDEINIT_CB_ID                  = 0x0EU,    /*!< CAN MspDeInit callback ID                     */

} HAL_CAN_CallbackIDTypeDef;

/**
  * @brief  HAL CAN Callback pointer definition
  */
typedef  void (*pCAN_CallbackTypeDef)(CAN_HandleTypeDef *hcan); /*!< pointer to a CAN callback function   */

#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
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
#define HAL_CAN_ERROR_NONE            (0x00000000U)  /*!< No error                                             */
#define HAL_CAN_ERROR_EWG             (0x00000001U)  /*!< Protocol Error Warning                               */
#define HAL_CAN_ERROR_EPV             (0x00000002U)  /*!< Error Passive                                        */
#define HAL_CAN_ERROR_BOF             (0x00000004U)  /*!< Bus-off error                                        */
#define HAL_CAN_ERROR_STF             (0x00000008U)  /*!< Stuff error                                          */
#define HAL_CAN_ERROR_FOR             (0x00000010U)  /*!< Form error                                           */
#define HAL_CAN_ERROR_ACK             (0x00000020U)  /*!< Acknowledgment error                                 */
#define HAL_CAN_ERROR_BR              (0x00000040U)  /*!< Bit recessive error                                  */
#define HAL_CAN_ERROR_BD              (0x00000080U)  /*!< Bit dominant error                                   */
#define HAL_CAN_ERROR_CRC             (0x00000100U)  /*!< CRC error                                            */
#define HAL_CAN_ERROR_RX_FOV0         (0x00000200U)  /*!< Rx FIFO0 overrun error                               */
#define HAL_CAN_ERROR_RX_FOV1         (0x00000400U)  /*!< Rx FIFO1 overrun error                               */
#define HAL_CAN_ERROR_TX_ALST0        (0x00000800U)  /*!< TxMailbox 0 transmit failure due to arbitration lost */
#define HAL_CAN_ERROR_TX_TERR0        (0x00001000U)  /*!< TxMailbox 0 transmit failure due to transmit error    */
#define HAL_CAN_ERROR_TX_ALST1        (0x00002000U)  /*!< TxMailbox 1 transmit failure due to arbitration lost */
#define HAL_CAN_ERROR_TX_TERR1        (0x00004000U)  /*!< TxMailbox 1 transmit failure due to transmit error    */
#define HAL_CAN_ERROR_TX_ALST2        (0x00008000U)  /*!< TxMailbox 2 transmit failure due to arbitration lost */
#define HAL_CAN_ERROR_TX_TERR2        (0x00010000U)  /*!< TxMailbox 2 transmit failure due to transmit error    */
#define HAL_CAN_ERROR_TIMEOUT         (0x00020000U)  /*!< Timeout error                                        */
#define HAL_CAN_ERROR_NOT_INITIALIZED (0x00040000U)  /*!< Peripheral not initialized                           */
#define HAL_CAN_ERROR_NOT_READY       (0x00080000U)  /*!< Peripheral not ready                                 */
#define HAL_CAN_ERROR_NOT_STARTED     (0x00100000U)  /*!< Peripheral not started                               */
#define HAL_CAN_ERROR_PARAM           (0x00200000U)  /*!< Parameter error                                      */

#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
#define HAL_CAN_ERROR_INVALID_CALLBACK (0x00400000U) /*!< Invalid Callback error                               */
#endif /* USE_HAL_CAN_REGISTER_CALLBACKS */
#define HAL_CAN_ERROR_INTERNAL        (0x00800000U)  /*!< Internal error                                       */

/**
  * @}
  */

/** @defgroup CAN_InitStatus CAN InitStatus
  * @{
  */
#define CAN_INITSTATUS_FAILED       (0x00000000U)  /*!< CAN initialization failed */
#define CAN_INITSTATUS_SUCCESS      (0x00000001U)  /*!< CAN initialization OK     */
/**
  * @}
  */

/** @defgroup CAN_operating_mode CAN Operating Mode
  * @{
  */
#define CAN_MODE_NORMAL             (0x00000000U)                              /*!< Normal mode   */
#define CAN_MODE_LOOPBACK           ((uint32_t)CAN_BTR_LBKM)                   /*!< Loopback mode */
#define CAN_MODE_SILENT             ((uint32_t)CAN_BTR_SILM)                   /*!< Silent mode   */
#define CAN_MODE_SILENT_LOOPBACK    ((uint32_t)(CAN_BTR_LBKM | CAN_BTR_SILM))  /*!< Loopback combined with silent mode */
/**
  * @}
  */


/** @defgroup CAN_synchronisation_jump_width CAN Synchronization Jump Width
  * @{
  */
#define CAN_SJW_1TQ                 (0x00000000U)              /*!< 1 time quantum */
#define CAN_SJW_2TQ                 ((uint32_t)CAN_BTR_SJW_0)  /*!< 2 time quantum */
#define CAN_SJW_3TQ                 ((uint32_t)CAN_BTR_SJW_1)  /*!< 3 time quantum */
#define CAN_SJW_4TQ                 ((uint32_t)CAN_BTR_SJW)    /*!< 4 time quantum */
/**
  * @}
  */

/** @defgroup CAN_time_quantum_in_bit_segment_1 CAN Time Quantum in Bit Segment 1
  * @{
  */
#define CAN_BS1_1TQ                 (0x00000000U)                                                /*!< 1 time quantum  */
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

/** @defgroup CAN_time_quantum_in_bit_segment_2 CAN Time Quantum in Bit Segment 2
  * @{
  */
#define CAN_BS2_1TQ                 (0x00000000U)                                /*!< 1 time quantum */
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

/** @defgroup CAN_filter_mode CAN Filter Mode
  * @{
  */
#define CAN_FILTERMODE_IDMASK       (0x00000000U)  /*!< Identifier mask mode */
#define CAN_FILTERMODE_IDLIST       (0x00000001U)  /*!< Identifier list mode */
/**
  * @}
  */

/** @defgroup CAN_filter_scale CAN Filter Scale
  * @{
  */
#define CAN_FILTERSCALE_16BIT       (0x00000000U)  /*!< Two 16-bit filters */
#define CAN_FILTERSCALE_32BIT       (0x00000001U)  /*!< One 32-bit filter  */
/**
  * @}
  */

/** @defgroup CAN_filter_activation CAN Filter Activation
  * @{
  */
#define CAN_FILTER_DISABLE          (0x00000000U)  /*!< Disable filter */
#define CAN_FILTER_ENABLE           (0x00000001U)  /*!< Enable filter  */
/**
  * @}
  */

/** @defgroup CAN_filter_FIFO CAN Filter FIFO
  * @{
  */
#define CAN_FILTER_FIFO0            (0x00000000U)  /*!< Filter FIFO 0 assignment for filter x */
#define CAN_FILTER_FIFO1            (0x00000001U)  /*!< Filter FIFO 1 assignment for filter x */
/**
  * @}
  */

/** @defgroup CAN_identifier_type CAN Identifier Type
  * @{
  */
#define CAN_ID_STD                  (0x00000000U)  /*!< Standard Id */
#define CAN_ID_EXT                  (0x00000004U)  /*!< Extended Id */
/**
  * @}
  */

/** @defgroup CAN_remote_transmission_request CAN Remote Transmission Request
  * @{
  */
#define CAN_RTR_DATA                (0x00000000U)  /*!< Data frame   */
#define CAN_RTR_REMOTE              (0x00000002U)  /*!< Remote frame */
/**
  * @}
  */

/** @defgroup CAN_receive_FIFO_number CAN Receive FIFO Number
  * @{
  */
#define CAN_RX_FIFO0                (0x00000000U)  /*!< CAN receive FIFO 0 */
#define CAN_RX_FIFO1                (0x00000001U)  /*!< CAN receive FIFO 1 */
/**
  * @}
  */

/** @defgroup CAN_Tx_Mailboxes CAN Tx Mailboxes
  * @{
  */
#define CAN_TX_MAILBOX0             (0x00000001U)  /*!< Tx Mailbox 0  */
#define CAN_TX_MAILBOX1             (0x00000002U)  /*!< Tx Mailbox 1  */
#define CAN_TX_MAILBOX2             (0x00000004U)  /*!< Tx Mailbox 2  */
/**
  * @}
  */

/** @defgroup CAN_flags CAN Flags
  * @{
  */
/* Transmit Flags */
#define CAN_FLAG_RQCP0              (0x00000500U)  /*!< Request complete MailBox 0 flag   */
#define CAN_FLAG_TXOK0              (0x00000501U)  /*!< Transmission OK MailBox 0 flag    */
#define CAN_FLAG_ALST0              (0x00000502U)  /*!< Arbitration Lost MailBox 0 flag   */
#define CAN_FLAG_TERR0              (0x00000503U)  /*!< Transmission error MailBox 0 flag */
#define CAN_FLAG_RQCP1              (0x00000508U)  /*!< Request complete MailBox1 flag    */
#define CAN_FLAG_TXOK1              (0x00000509U)  /*!< Transmission OK MailBox 1 flag    */
#define CAN_FLAG_ALST1              (0x0000050AU)  /*!< Arbitration Lost MailBox 1 flag   */
#define CAN_FLAG_TERR1              (0x0000050BU)  /*!< Transmission error MailBox 1 flag */
#define CAN_FLAG_RQCP2              (0x00000510U)  /*!< Request complete MailBox2 flag    */
#define CAN_FLAG_TXOK2              (0x00000511U)  /*!< Transmission OK MailBox 2 flag    */
#define CAN_FLAG_ALST2              (0x00000512U)  /*!< Arbitration Lost MailBox 2 flag   */
#define CAN_FLAG_TERR2              (0x00000513U)  /*!< Transmission error MailBox 2 flag */
#define CAN_FLAG_TME0               (0x0000051AU)  /*!< Transmit mailbox 0 empty flag     */
#define CAN_FLAG_TME1               (0x0000051BU)  /*!< Transmit mailbox 1 empty flag     */
#define CAN_FLAG_TME2               (0x0000051CU)  /*!< Transmit mailbox 2 empty flag     */
#define CAN_FLAG_LOW0               (0x0000051DU)  /*!< Lowest priority mailbox 0 flag    */
#define CAN_FLAG_LOW1               (0x0000051EU)  /*!< Lowest priority mailbox 1 flag    */
#define CAN_FLAG_LOW2               (0x0000051FU)  /*!< Lowest priority mailbox 2 flag    */

/* Receive Flags */
#define CAN_FLAG_FF0                (0x00000203U)  /*!< RX FIFO 0 Full flag               */
#define CAN_FLAG_FOV0               (0x00000204U)  /*!< RX FIFO 0 Overrun flag            */
#define CAN_FLAG_FF1                (0x00000403U)  /*!< RX FIFO 1 Full flag               */
#define CAN_FLAG_FOV1               (0x00000404U)  /*!< RX FIFO 1 Overrun flag            */

/* Operating Mode Flags */
#define CAN_FLAG_INAK               (0x00000100U)  /*!< Initialization acknowledge flag   */
#define CAN_FLAG_SLAK               (0x00000101U)  /*!< Sleep acknowledge flag            */
#define CAN_FLAG_ERRI               (0x00000102U)  /*!< Error flag                        */
#define CAN_FLAG_WKU                (0x00000103U)  /*!< Wake up interrupt flag            */
#define CAN_FLAG_SLAKI              (0x00000104U)  /*!< Sleep acknowledge interrupt flag  */

/* Error Flags */
#define CAN_FLAG_EWG                (0x00000300U)  /*!< Error warning flag                */
#define CAN_FLAG_EPV                (0x00000301U)  /*!< Error passive flag                */
#define CAN_FLAG_BOF                (0x00000302U)  /*!< Bus-Off flag                      */
/**
  * @}
  */


/** @defgroup CAN_Interrupts CAN Interrupts
  * @{
  */
/* Transmit Interrupt */
#define CAN_IT_TX_MAILBOX_EMPTY     ((uint32_t)CAN_IER_TMEIE)   /*!< Transmit mailbox empty interrupt */

/* Receive Interrupts */
#define CAN_IT_RX_FIFO0_MSG_PENDING ((uint32_t)CAN_IER_FMPIE0)  /*!< FIFO 0 message pending interrupt */
#define CAN_IT_RX_FIFO0_FULL        ((uint32_t)CAN_IER_FFIE0)   /*!< FIFO 0 full interrupt            */
#define CAN_IT_RX_FIFO0_OVERRUN     ((uint32_t)CAN_IER_FOVIE0)  /*!< FIFO 0 overrun interrupt         */
#define CAN_IT_RX_FIFO1_MSG_PENDING ((uint32_t)CAN_IER_FMPIE1)  /*!< FIFO 1 message pending interrupt */
#define CAN_IT_RX_FIFO1_FULL        ((uint32_t)CAN_IER_FFIE1)   /*!< FIFO 1 full interrupt            */
#define CAN_IT_RX_FIFO1_OVERRUN     ((uint32_t)CAN_IER_FOVIE1)  /*!< FIFO 1 overrun interrupt         */

/* Operating Mode Interrupts */
#define CAN_IT_WAKEUP               ((uint32_t)CAN_IER_WKUIE)   /*!< Wake-up interrupt                */
#define CAN_IT_SLEEP_ACK            ((uint32_t)CAN_IER_SLKIE)   /*!< Sleep acknowledge interrupt      */

/* Error Interrupts */
#define CAN_IT_ERROR_WARNING        ((uint32_t)CAN_IER_EWGIE)   /*!< Error warning interrupt          */
#define CAN_IT_ERROR_PASSIVE        ((uint32_t)CAN_IER_EPVIE)   /*!< Error passive interrupt          */
#define CAN_IT_BUSOFF               ((uint32_t)CAN_IER_BOFIE)   /*!< Bus-off interrupt                */
#define CAN_IT_LAST_ERROR_CODE      ((uint32_t)CAN_IER_LECIE)   /*!< Last error code interrupt        */
#define CAN_IT_ERROR                ((uint32_t)CAN_IER_ERRIE)   /*!< Error Interrupt                  */
/**
  * @}
  */

/**
  * @}
  */

/* Exported macros -----------------------------------------------------------*/
/** @defgroup CAN_Exported_Macros CAN Exported Macros
  * @{
  */

/** @brief  Reset CAN handle state
  * @param  __HANDLE__ CAN handle.
  * @retval None
  */
#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
#define __HAL_CAN_RESET_HANDLE_STATE(__HANDLE__) do{                                              \
                                                     (__HANDLE__)->State = HAL_CAN_STATE_RESET;   \
                                                     (__HANDLE__)->MspInitCallback = NULL;        \
                                                     (__HANDLE__)->MspDeInitCallback = NULL;      \
                                                   } while(0)
#else
#define __HAL_CAN_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_CAN_STATE_RESET)
#endif /*USE_HAL_CAN_REGISTER_CALLBACKS */

/**
  * @brief  Enable the specified CAN interrupts.
  * @param  __HANDLE__ CAN handle.
  * @param  __INTERRUPT__ CAN Interrupt sources to enable.
  *           This parameter can be any combination of @arg CAN_Interrupts
  * @retval None
  */
#define __HAL_CAN_ENABLE_IT(__HANDLE__, __INTERRUPT__) (((__HANDLE__)->Instance->IER) |= (__INTERRUPT__))

/**
  * @brief  Disable the specified CAN interrupts.
  * @param  __HANDLE__ CAN handle.
  * @param  __INTERRUPT__ CAN Interrupt sources to disable.
  *           This parameter can be any combination of @arg CAN_Interrupts
  * @retval None
  */
#define __HAL_CAN_DISABLE_IT(__HANDLE__, __INTERRUPT__) (((__HANDLE__)->Instance->IER) &= ~(__INTERRUPT__))

/** @brief  Check if the specified CAN interrupt source is enabled or disabled.
  * @param  __HANDLE__ specifies the CAN Handle.
  * @param  __INTERRUPT__ specifies the CAN interrupt source to check.
  *           This parameter can be a value of @arg CAN_Interrupts
  * @retval The state of __IT__ (TRUE or FALSE).
  */
#define __HAL_CAN_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) (((__HANDLE__)->Instance->IER) & (__INTERRUPT__))

/** @brief  Check whether the specified CAN flag is set or not.
  * @param  __HANDLE__ specifies the CAN Handle.
  * @param  __FLAG__ specifies the flag to check.
  *         This parameter can be one of @arg CAN_flags
  * @retval The state of __FLAG__ (TRUE or FALSE).
  */
#define __HAL_CAN_GET_FLAG(__HANDLE__, __FLAG__) \
  ((((__FLAG__) >> 8U) == 5U)? ((((__HANDLE__)->Instance->TSR) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
   (((__FLAG__) >> 8U) == 2U)? ((((__HANDLE__)->Instance->RF0R) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
   (((__FLAG__) >> 8U) == 4U)? ((((__HANDLE__)->Instance->RF1R) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
   (((__FLAG__) >> 8U) == 1U)? ((((__HANDLE__)->Instance->MSR) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
   (((__FLAG__) >> 8U) == 3U)? ((((__HANDLE__)->Instance->ESR) & (1U << ((__FLAG__) & CAN_FLAG_MASK))) == (1U << ((__FLAG__) & CAN_FLAG_MASK))): 0U)

/** @brief  Clear the specified CAN pending flag.
  * @param  __HANDLE__ specifies the CAN Handle.
  * @param  __FLAG__ specifies the flag to check.
  *         This parameter can be one of the following values:
  *            @arg CAN_FLAG_RQCP0: Request complete MailBox 0 Flag
  *            @arg CAN_FLAG_TXOK0: Transmission OK MailBox 0 Flag
  *            @arg CAN_FLAG_ALST0: Arbitration Lost MailBox 0 Flag
  *            @arg CAN_FLAG_TERR0: Transmission error MailBox 0 Flag
  *            @arg CAN_FLAG_RQCP1: Request complete MailBox 1 Flag
  *            @arg CAN_FLAG_TXOK1: Transmission OK MailBox 1 Flag
  *            @arg CAN_FLAG_ALST1: Arbitration Lost MailBox 1 Flag
  *            @arg CAN_FLAG_TERR1: Transmission error MailBox 1 Flag
  *            @arg CAN_FLAG_RQCP2: Request complete MailBox 2 Flag
  *            @arg CAN_FLAG_TXOK2: Transmission OK MailBox 2 Flag
  *            @arg CAN_FLAG_ALST2: Arbitration Lost MailBox 2 Flag
  *            @arg CAN_FLAG_TERR2: Transmission error MailBox 2 Flag
  *            @arg CAN_FLAG_FF0:   RX FIFO 0 Full Flag
  *            @arg CAN_FLAG_FOV0:  RX FIFO 0 Overrun Flag
  *            @arg CAN_FLAG_FF1:   RX FIFO 1 Full Flag
  *            @arg CAN_FLAG_FOV1:  RX FIFO 1 Overrun Flag
  *            @arg CAN_FLAG_WKUI:  Wake up Interrupt Flag
  *            @arg CAN_FLAG_SLAKI: Sleep acknowledge Interrupt Flag
  * @retval None
  */
#define __HAL_CAN_CLEAR_FLAG(__HANDLE__, __FLAG__) \
  ((((__FLAG__) >> 8U) == 5U)? (((__HANDLE__)->Instance->TSR) = (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
   (((__FLAG__) >> 8U) == 2U)? (((__HANDLE__)->Instance->RF0R) = (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
   (((__FLAG__) >> 8U) == 4U)? (((__HANDLE__)->Instance->RF1R) = (1U << ((__FLAG__) & CAN_FLAG_MASK))): \
   (((__FLAG__) >> 8U) == 1U)? (((__HANDLE__)->Instance->MSR) = (1U << ((__FLAG__) & CAN_FLAG_MASK))): 0U)

/**
 * @}
 */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup CAN_Exported_Functions CAN Exported Functions
  * @{
  */

/** @addtogroup CAN_Exported_Functions_Group1 Initialization and de-initialization functions
 *  @brief    Initialization and Configuration functions
 * @{
 */

/* Initialization and de-initialization functions *****************************/
HAL_StatusTypeDef HAL_CAN_Init(CAN_HandleTypeDef *hcan);
HAL_StatusTypeDef HAL_CAN_DeInit(CAN_HandleTypeDef *hcan);
void HAL_CAN_MspInit(CAN_HandleTypeDef *hcan);
void HAL_CAN_MspDeInit(CAN_HandleTypeDef *hcan);

#if USE_HAL_CAN_REGISTER_CALLBACKS == 1
/* Callbacks Register/UnRegister functions  ***********************************/
HAL_StatusTypeDef HAL_CAN_RegisterCallback(CAN_HandleTypeDef *hcan, HAL_CAN_CallbackIDTypeDef CallbackID, void (* pCallback)(CAN_HandleTypeDef *_hcan));
HAL_StatusTypeDef HAL_CAN_UnRegisterCallback(CAN_HandleTypeDef *hcan, HAL_CAN_CallbackIDTypeDef CallbackID);

#endif /* (USE_HAL_CAN_REGISTER_CALLBACKS) */
/**
 * @}
 */

/** @addtogroup CAN_Exported_Functions_Group2 Configuration functions
 *  @brief    Configuration functions
 * @{
 */

/* Configuration functions ****************************************************/
HAL_StatusTypeDef HAL_CAN_ConfigFilter(CAN_HandleTypeDef *hcan, CAN_FilterTypeDef *sFilterConfig);

/**
 * @}
 */

/** @addtogroup CAN_Exported_Functions_Group3 Control functions
 *  @brief    Control functions
 * @{
 */

/* Control functions **********************************************************/
HAL_StatusTypeDef HAL_CAN_Start(CAN_HandleTypeDef *hcan);
HAL_StatusTypeDef HAL_CAN_Stop(CAN_HandleTypeDef *hcan);
HAL_StatusTypeDef HAL_CAN_RequestSleep(CAN_HandleTypeDef *hcan);
HAL_StatusTypeDef HAL_CAN_WakeUp(CAN_HandleTypeDef *hcan);
uint32_t HAL_CAN_IsSleepActive(CAN_HandleTypeDef *hcan);
HAL_StatusTypeDef HAL_CAN_AddTxMessage(CAN_HandleTypeDef *hcan, CAN_TxHeaderTypeDef *pHeader, uint8_t aData[], uint32_t *pTxMailbox);
HAL_StatusTypeDef HAL_CAN_AbortTxRequest(CAN_HandleTypeDef *hcan, uint32_t TxMailboxes);
uint32_t HAL_CAN_GetTxMailboxesFreeLevel(CAN_HandleTypeDef *hcan);
uint32_t HAL_CAN_IsTxMessagePending(CAN_HandleTypeDef *hcan, uint32_t TxMailboxes);
uint32_t HAL_CAN_GetTxTimestamp(CAN_HandleTypeDef *hcan, uint32_t TxMailbox);
HAL_StatusTypeDef HAL_CAN_GetRxMessage(CAN_HandleTypeDef *hcan, uint32_t RxFifo, CAN_RxHeaderTypeDef *pHeader, uint8_t aData[]);
uint32_t HAL_CAN_GetRxFifoFillLevel(CAN_HandleTypeDef *hcan, uint32_t RxFifo);

/**
 * @}
 */

/** @addtogroup CAN_Exported_Functions_Group4 Interrupts management
 *  @brief    Interrupts management
 * @{
 */
/* Interrupts management ******************************************************/
HAL_StatusTypeDef HAL_CAN_ActivateNotification(CAN_HandleTypeDef *hcan, uint32_t ActiveITs);
HAL_StatusTypeDef HAL_CAN_DeactivateNotification(CAN_HandleTypeDef *hcan, uint32_t InactiveITs);
void HAL_CAN_IRQHandler(CAN_HandleTypeDef *hcan);

/**
 * @}
 */

/** @addtogroup CAN_Exported_Functions_Group5 Callback functions
 *  @brief    Callback functions
 * @{
 */
/* Callbacks functions ********************************************************/

void HAL_CAN_TxMailbox0CompleteCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_TxMailbox1CompleteCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_TxMailbox2CompleteCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_TxMailbox0AbortCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_TxMailbox1AbortCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_TxMailbox2AbortCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_RxFifo0FullCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_RxFifo1MsgPendingCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_RxFifo1FullCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_SleepCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_WakeUpFromRxMsgCallback(CAN_HandleTypeDef *hcan);
void HAL_CAN_ErrorCallback(CAN_HandleTypeDef *hcan);

/**
 * @}
 */

/** @addtogroup CAN_Exported_Functions_Group6 Peripheral State and Error functions
 *  @brief   CAN Peripheral State functions
 * @{
 */
/* Peripheral State and Error functions ***************************************/
HAL_CAN_StateTypeDef HAL_CAN_GetState(CAN_HandleTypeDef *hcan);
uint32_t HAL_CAN_GetError(CAN_HandleTypeDef *hcan);
HAL_StatusTypeDef HAL_CAN_ResetError(CAN_HandleTypeDef *hcan);

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
#define CAN_FLAG_MASK  (0x000000FFU)
/**
  * @}
  */

/* Private Macros -----------------------------------------------------------*/
/** @defgroup CAN_Private_Macros CAN Private Macros
  * @{
  */

#define IS_CAN_MODE(MODE) (((MODE) == CAN_MODE_NORMAL) || \
                           ((MODE) == CAN_MODE_LOOPBACK)|| \
                           ((MODE) == CAN_MODE_SILENT) || \
                           ((MODE) == CAN_MODE_SILENT_LOOPBACK))
#define IS_CAN_SJW(SJW) (((SJW) == CAN_SJW_1TQ) || ((SJW) == CAN_SJW_2TQ) || \
                         ((SJW) == CAN_SJW_3TQ) || ((SJW) == CAN_SJW_4TQ))
#define IS_CAN_BS1(BS1) (((BS1) == CAN_BS1_1TQ) || ((BS1) == CAN_BS1_2TQ) || \
                         ((BS1) == CAN_BS1_3TQ) || ((BS1) == CAN_BS1_4TQ) || \
                         ((BS1) == CAN_BS1_5TQ) || ((BS1) == CAN_BS1_6TQ) || \
                         ((BS1) == CAN_BS1_7TQ) || ((BS1) == CAN_BS1_8TQ) || \
                         ((BS1) == CAN_BS1_9TQ) || ((BS1) == CAN_BS1_10TQ)|| \
                         ((BS1) == CAN_BS1_11TQ)|| ((BS1) == CAN_BS1_12TQ)|| \
                         ((BS1) == CAN_BS1_13TQ)|| ((BS1) == CAN_BS1_14TQ)|| \
                         ((BS1) == CAN_BS1_15TQ)|| ((BS1) == CAN_BS1_16TQ))
#define IS_CAN_BS2(BS2) (((BS2) == CAN_BS2_1TQ) || ((BS2) == CAN_BS2_2TQ) || \
                         ((BS2) == CAN_BS2_3TQ) || ((BS2) == CAN_BS2_4TQ) || \
                         ((BS2) == CAN_BS2_5TQ) || ((BS2) == CAN_BS2_6TQ) || \
                         ((BS2) == CAN_BS2_7TQ) || ((BS2) == CAN_BS2_8TQ))
#define IS_CAN_PRESCALER(PRESCALER) (((PRESCALER) >= 1U) && ((PRESCALER) <= 1024U))
#define IS_CAN_FILTER_ID_HALFWORD(HALFWORD) ((HALFWORD) <= 0xFFFFU)
#define IS_CAN_FILTER_BANK_DUAL(BANK) ((BANK) <= 27U)
#define IS_CAN_FILTER_BANK_SINGLE(BANK) ((BANK) <= 13U)
#define IS_CAN_FILTER_MODE(MODE) (((MODE) == CAN_FILTERMODE_IDMASK) || \
                                  ((MODE) == CAN_FILTERMODE_IDLIST))
#define IS_CAN_FILTER_SCALE(SCALE) (((SCALE) == CAN_FILTERSCALE_16BIT) || \
                                    ((SCALE) == CAN_FILTERSCALE_32BIT))
#define IS_CAN_FILTER_ACTIVATION(ACTIVATION) (((ACTIVATION) == CAN_FILTER_DISABLE) || \
                                              ((ACTIVATION) == CAN_FILTER_ENABLE))
#define IS_CAN_FILTER_FIFO(FIFO) (((FIFO) == CAN_FILTER_FIFO0) || \
                                  ((FIFO) == CAN_FILTER_FIFO1))
#define IS_CAN_TX_MAILBOX(TRANSMITMAILBOX) (((TRANSMITMAILBOX) == CAN_TX_MAILBOX0 ) || \
                                            ((TRANSMITMAILBOX) == CAN_TX_MAILBOX1 ) || \
                                            ((TRANSMITMAILBOX) == CAN_TX_MAILBOX2 ))
#define IS_CAN_TX_MAILBOX_LIST(TRANSMITMAILBOX) ((TRANSMITMAILBOX) <= (CAN_TX_MAILBOX0 | CAN_TX_MAILBOX1 | CAN_TX_MAILBOX2))
#define IS_CAN_STDID(STDID)   ((STDID) <= 0x7FFU)
#define IS_CAN_EXTID(EXTID)   ((EXTID) <= 0x1FFFFFFFU)
#define IS_CAN_DLC(DLC)       ((DLC) <= 8U)
#define IS_CAN_IDTYPE(IDTYPE)  (((IDTYPE) == CAN_ID_STD) || \
                                ((IDTYPE) == CAN_ID_EXT))
#define IS_CAN_RTR(RTR) (((RTR) == CAN_RTR_DATA) || ((RTR) == CAN_RTR_REMOTE))
#define IS_CAN_RX_FIFO(FIFO) (((FIFO) == CAN_RX_FIFO0) || ((FIFO) == CAN_RX_FIFO1))
#define IS_CAN_IT(IT) ((IT) <= (CAN_IT_TX_MAILBOX_EMPTY     | CAN_IT_RX_FIFO0_MSG_PENDING      | \
                                CAN_IT_RX_FIFO0_FULL        | CAN_IT_RX_FIFO0_OVERRUN          | \
                                CAN_IT_RX_FIFO1_MSG_PENDING | CAN_IT_RX_FIFO1_FULL             | \
                                CAN_IT_RX_FIFO1_OVERRUN     | CAN_IT_WAKEUP                    | \
                                CAN_IT_SLEEP_ACK            | CAN_IT_ERROR_WARNING             | \
                                CAN_IT_ERROR_PASSIVE        | CAN_IT_BUSOFF                    | \
                                CAN_IT_LAST_ERROR_CODE      | CAN_IT_ERROR))

/**
  * @}
  */
/* End of private macros -----------------------------------------------------*/

/**
  * @}
  */


#endif /* CAN1 */
/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_HAL_CAN_H */


/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
