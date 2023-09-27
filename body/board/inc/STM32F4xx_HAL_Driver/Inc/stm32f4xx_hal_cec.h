/**
  ******************************************************************************
  * @file    stm32f4xx_hal_cec.h
  * @author  MCD Application Team
  * @brief   Header file of CEC HAL module.
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
#ifndef STM32F4xx_HAL_CEC_H
#define STM32F4xx_HAL_CEC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

#if defined (CEC)

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup CEC
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup CEC_Exported_Types CEC Exported Types
  * @{
  */

/**
  * @brief CEC Init Structure definition
  */
typedef struct
{
  uint32_t SignalFreeTime;               /*!< Set SFT field, specifies the Signal Free Time.
                                              It can be one of @ref CEC_Signal_Free_Time
                                              and belongs to the set {0,...,7} where
                                              0x0 is the default configuration
                                              else means 0.5 + (SignalFreeTime - 1) nominal data bit periods */

  uint32_t Tolerance;                    /*!< Set RXTOL bit, specifies the tolerance accepted on the received waveforms,
                                              it can be a value of @ref CEC_Tolerance : it is either CEC_STANDARD_TOLERANCE
                                              or CEC_EXTENDED_TOLERANCE */

  uint32_t BRERxStop;                    /*!< Set BRESTP bit @ref CEC_BRERxStop : specifies whether or not a Bit Rising Error stops the reception.
                                              CEC_NO_RX_STOP_ON_BRE: reception is not stopped.
                                              CEC_RX_STOP_ON_BRE:    reception is stopped. */

  uint32_t BREErrorBitGen;               /*!< Set BREGEN bit @ref CEC_BREErrorBitGen : specifies whether or not an Error-Bit is generated on the
                                              CEC line upon Bit Rising Error detection.
                                              CEC_BRE_ERRORBIT_NO_GENERATION: no error-bit generation.
                                              CEC_BRE_ERRORBIT_GENERATION:    error-bit generation if BRESTP is set. */

  uint32_t LBPEErrorBitGen;              /*!< Set LBPEGEN bit @ref CEC_LBPEErrorBitGen : specifies whether or not an Error-Bit is generated on the
                                              CEC line upon Long Bit Period Error detection.
                                              CEC_LBPE_ERRORBIT_NO_GENERATION:  no error-bit generation.
                                              CEC_LBPE_ERRORBIT_GENERATION:     error-bit generation. */

  uint32_t BroadcastMsgNoErrorBitGen;    /*!< Set BRDNOGEN bit @ref CEC_BroadCastMsgErrorBitGen : allows to avoid an Error-Bit generation on the CEC line
                                              upon an error detected on a broadcast message.

                                              It supersedes BREGEN and LBPEGEN bits for a broadcast message error handling. It can take two values:

                                              1) CEC_BROADCASTERROR_ERRORBIT_GENERATION.
                                                 a) BRE detection: error-bit generation on the CEC line if BRESTP=CEC_RX_STOP_ON_BRE
                                                    and BREGEN=CEC_BRE_ERRORBIT_NO_GENERATION.
                                                 b) LBPE detection: error-bit generation on the CEC line
                                                    if LBPGEN=CEC_LBPE_ERRORBIT_NO_GENERATION.

                                              2) CEC_BROADCASTERROR_NO_ERRORBIT_GENERATION.
                                                 no error-bit generation in case neither a) nor b) are satisfied. Additionally,
                                                 there is no error-bit generation in case of Short Bit Period Error detection in
                                                 a broadcast message while LSTN bit is set. */

  uint32_t SignalFreeTimeOption;         /*!< Set SFTOP bit @ref CEC_SFT_Option : specifies when SFT timer starts.
                                              CEC_SFT_START_ON_TXSOM SFT:    timer starts when TXSOM is set by software.
                                              CEC_SFT_START_ON_TX_RX_END:  SFT timer starts automatically at the end of message transmission/reception. */

  uint32_t ListenMode;                   /*!< Set LSTN bit @ref CEC_Listening_Mode : specifies device listening mode. It can take two values:

                                              CEC_REDUCED_LISTENING_MODE: CEC peripheral receives only message addressed to its
                                                own address (OAR). Messages addressed to different destination are ignored.
                                                Broadcast messages are always received.

                                              CEC_FULL_LISTENING_MODE: CEC peripheral receives messages addressed to its own
                                                address (OAR) with positive acknowledge. Messages addressed to different destination
                                                are received, but without interfering with the CEC bus: no acknowledge sent.  */

  uint16_t  OwnAddress;                 /*!< Own addresses configuration
                                             This parameter can be a value of @ref CEC_OWN_ADDRESS */

  uint8_t  *RxBuffer;                    /*!< CEC Rx buffer pointeur */


} CEC_InitTypeDef;

/**
  * @brief HAL CEC State definition
  * @note  HAL CEC State value is a combination of 2 different substates: gState and RxState (see @ref CEC_State_Definition).
  *        - gState contains CEC state information related to global Handle management
  *          and also information related to Tx operations.
  *          gState value coding follow below described bitmap :
  *          b7 (not used)
  *             x  : Should be set to 0
  *          b6  Error information
  *             0  : No Error
  *             1  : Error
  *          b5     CEC peripheral initialization status
  *             0  : Reset (peripheral not initialized)
  *             1  : Init done (peripheral initialized. HAL CEC Init function already called)
  *          b4-b3  (not used)
  *             xx : Should be set to 00
  *          b2     Intrinsic process state
  *             0  : Ready
  *             1  : Busy (peripheral busy with some configuration or internal operations)
  *          b1     (not used)
  *             x  : Should be set to 0
  *          b0     Tx state
  *             0  : Ready (no Tx operation ongoing)
  *             1  : Busy (Tx operation ongoing)
  *        - RxState contains information related to Rx operations.
  *          RxState value coding follow below described bitmap :
  *          b7-b6  (not used)
  *             xx : Should be set to 00
  *          b5     CEC peripheral initialization status
  *             0  : Reset (peripheral not initialized)
  *             1  : Init done (peripheral initialized)
  *          b4-b2  (not used)
  *            xxx : Should be set to 000
  *          b1     Rx state
  *             0  : Ready (no Rx operation ongoing)
  *             1  : Busy (Rx operation ongoing)
  *          b0     (not used)
  *             x  : Should be set to 0.
  */
typedef uint32_t HAL_CEC_StateTypeDef;

/**
  * @brief  CEC handle Structure definition
  */
#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1)
typedef struct __CEC_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */
{
  CEC_TypeDef             *Instance;      /*!< CEC registers base address */

  CEC_InitTypeDef         Init;           /*!< CEC communication parameters */

  uint8_t                 *pTxBuffPtr;    /*!< Pointer to CEC Tx transfer Buffer */

  uint16_t                TxXferCount;    /*!< CEC Tx Transfer Counter */

  uint16_t                RxXferSize;     /*!< CEC Rx Transfer size, 0: header received only */

  HAL_LockTypeDef         Lock;           /*!< Locking object */

  HAL_CEC_StateTypeDef    gState;         /*!< CEC state information related to global Handle management
                                               and also related to Tx operations.
                                               This parameter can be a value of @ref HAL_CEC_StateTypeDef */

  HAL_CEC_StateTypeDef    RxState;        /*!< CEC state information related to Rx operations.
                                               This parameter can be a value of @ref HAL_CEC_StateTypeDef */

  uint32_t                ErrorCode;      /*!< For errors handling purposes, copy of ISR register
                                               in case error is reported */

#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1)
  void (* TxCpltCallback)(struct __CEC_HandleTypeDef
                          *hcec);                                /*!< CEC Tx Transfer completed callback */
  void (* RxCpltCallback)(struct __CEC_HandleTypeDef *hcec,
                          uint32_t RxFrameSize);          /*!< CEC Rx Transfer completed callback */
  void (* ErrorCallback)(struct __CEC_HandleTypeDef *hcec);                                 /*!< CEC error callback */

  void (* MspInitCallback)(struct __CEC_HandleTypeDef *hcec);               /*!< CEC Msp Init callback              */
  void (* MspDeInitCallback)(struct __CEC_HandleTypeDef *hcec);             /*!< CEC Msp DeInit callback            */

#endif /* (USE_HAL_CEC_REGISTER_CALLBACKS) */
} CEC_HandleTypeDef;

#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL CEC Callback ID enumeration definition
  */
typedef enum
{
  HAL_CEC_TX_CPLT_CB_ID      = 0x00U,    /*!< CEC Tx Transfer completed callback ID  */
  HAL_CEC_RX_CPLT_CB_ID      = 0x01U,    /*!< CEC Rx Transfer completed callback ID  */
  HAL_CEC_ERROR_CB_ID        = 0x02U,    /*!< CEC error callback ID                 */
  HAL_CEC_MSPINIT_CB_ID      = 0x03U,    /*!< CEC Msp Init callback ID               */
  HAL_CEC_MSPDEINIT_CB_ID    = 0x04U     /*!< CEC Msp DeInit callback ID             */
} HAL_CEC_CallbackIDTypeDef;

/**
  * @brief  HAL CEC Callback pointer definition
  */
typedef  void (*pCEC_CallbackTypeDef)(CEC_HandleTypeDef *hcec);  /*!< pointer to an CEC callback function */
typedef  void (*pCEC_RxCallbackTypeDef)(CEC_HandleTypeDef *hcec,
                                        uint32_t RxFrameSize);  /*!< pointer to an Rx Transfer completed callback function */
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */
/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup CEC_Exported_Constants CEC Exported Constants
  * @{
  */
/** @defgroup CEC_State_Definition CEC State Code Definition
  * @{
  */
#define   HAL_CEC_STATE_RESET      ((uint32_t)0x00000000)   /*!< Peripheral is not yet Initialized
                                                                 Value is allowed for gState and RxState  */
#define   HAL_CEC_STATE_READY      ((uint32_t)0x00000020)   /*!< Peripheral Initialized and ready for use
                                                                 Value is allowed for gState and RxState  */
#define   HAL_CEC_STATE_BUSY       ((uint32_t)0x00000024)   /*!< an internal process is ongoing
                                                                 Value is allowed for gState only         */
#define   HAL_CEC_STATE_BUSY_RX    ((uint32_t)0x00000022)   /*!< Data Reception process is ongoing
                                                                 Value is allowed for RxState only        */
#define   HAL_CEC_STATE_BUSY_TX    ((uint32_t)0x00000021)   /*!< Data Transmission process is ongoing
                                                                 Value is allowed for gState only         */
#define   HAL_CEC_STATE_BUSY_RX_TX ((uint32_t)0x00000023)   /*!< an internal process is ongoing
                                                                 Value is allowed for gState only         */
#define   HAL_CEC_STATE_ERROR      ((uint32_t)0x00000050)   /*!< Error Value is allowed for gState only   */
/**
  * @}
  */
/** @defgroup CEC_Error_Code CEC Error Code
  * @{
  */
#define HAL_CEC_ERROR_NONE    (uint32_t) 0x0000U     /*!< no error                      */
#define HAL_CEC_ERROR_RXOVR   CEC_ISR_RXOVR          /*!< CEC Rx-Overrun                */
#define HAL_CEC_ERROR_BRE     CEC_ISR_BRE            /*!< CEC Rx Bit Rising Error       */
#define HAL_CEC_ERROR_SBPE    CEC_ISR_SBPE           /*!< CEC Rx Short Bit period Error */
#define HAL_CEC_ERROR_LBPE    CEC_ISR_LBPE           /*!< CEC Rx Long Bit period Error  */
#define HAL_CEC_ERROR_RXACKE  CEC_ISR_RXACKE         /*!< CEC Rx Missing Acknowledge    */
#define HAL_CEC_ERROR_ARBLST  CEC_ISR_ARBLST         /*!< CEC Arbitration Lost          */
#define HAL_CEC_ERROR_TXUDR   CEC_ISR_TXUDR          /*!< CEC Tx-Buffer Underrun        */
#define HAL_CEC_ERROR_TXERR   CEC_ISR_TXERR          /*!< CEC Tx-Error                  */
#define HAL_CEC_ERROR_TXACKE  CEC_ISR_TXACKE         /*!< CEC Tx Missing Acknowledge    */
#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1)
#define  HAL_CEC_ERROR_INVALID_CALLBACK ((uint32_t)0x00002000U) /*!< Invalid Callback Error  */
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */
/**
  * @}
  */

/** @defgroup CEC_Signal_Free_Time  CEC Signal Free Time setting parameter
  * @{
  */
#define CEC_DEFAULT_SFT                    ((uint32_t)0x00000000U)
#define CEC_0_5_BITPERIOD_SFT              ((uint32_t)0x00000001U)
#define CEC_1_5_BITPERIOD_SFT              ((uint32_t)0x00000002U)
#define CEC_2_5_BITPERIOD_SFT              ((uint32_t)0x00000003U)
#define CEC_3_5_BITPERIOD_SFT              ((uint32_t)0x00000004U)
#define CEC_4_5_BITPERIOD_SFT              ((uint32_t)0x00000005U)
#define CEC_5_5_BITPERIOD_SFT              ((uint32_t)0x00000006U)
#define CEC_6_5_BITPERIOD_SFT              ((uint32_t)0x00000007U)
/**
  * @}
  */

/** @defgroup CEC_Tolerance CEC Receiver Tolerance
  * @{
  */
#define CEC_STANDARD_TOLERANCE             ((uint32_t)0x00000000U)
#define CEC_EXTENDED_TOLERANCE             ((uint32_t)CEC_CFGR_RXTOL)
/**
  * @}
  */

/** @defgroup CEC_BRERxStop CEC Reception Stop on Error
  * @{
  */
#define CEC_NO_RX_STOP_ON_BRE             ((uint32_t)0x00000000U)
#define CEC_RX_STOP_ON_BRE                ((uint32_t)CEC_CFGR_BRESTP)
/**
  * @}
  */

/** @defgroup CEC_BREErrorBitGen  CEC Error Bit Generation if Bit Rise Error reported
  * @{
  */
#define CEC_BRE_ERRORBIT_NO_GENERATION     ((uint32_t)0x00000000U)
#define CEC_BRE_ERRORBIT_GENERATION        ((uint32_t)CEC_CFGR_BREGEN)
/**
  * @}
  */

/** @defgroup CEC_LBPEErrorBitGen  CEC Error Bit Generation if Long Bit Period Error reported
  * @{
  */
#define CEC_LBPE_ERRORBIT_NO_GENERATION     ((uint32_t)0x00000000U)
#define CEC_LBPE_ERRORBIT_GENERATION        ((uint32_t)CEC_CFGR_LBPEGEN)
/**
  * @}
  */

/** @defgroup CEC_BroadCastMsgErrorBitGen  CEC Error Bit Generation on Broadcast message
  * @{
  */
#define CEC_BROADCASTERROR_ERRORBIT_GENERATION     ((uint32_t)0x00000000U)
#define CEC_BROADCASTERROR_NO_ERRORBIT_GENERATION  ((uint32_t)CEC_CFGR_BRDNOGEN)
/**
  * @}
  */

/** @defgroup CEC_SFT_Option     CEC Signal Free Time start option
  * @{
  */
#define CEC_SFT_START_ON_TXSOM           ((uint32_t)0x00000000U)
#define CEC_SFT_START_ON_TX_RX_END       ((uint32_t)CEC_CFGR_SFTOPT)
/**
  * @}
  */

/** @defgroup CEC_Listening_Mode    CEC Listening mode option
  * @{
  */
#define CEC_REDUCED_LISTENING_MODE          ((uint32_t)0x00000000U)
#define CEC_FULL_LISTENING_MODE             ((uint32_t)CEC_CFGR_LSTN)
/**
  * @}
  */

/** @defgroup CEC_OAR_Position   CEC Device Own Address position in CEC CFGR register
  * @{
  */
#define CEC_CFGR_OAR_LSB_POS            ((uint32_t) 16U)
/**
  * @}
  */

/** @defgroup CEC_Initiator_Position   CEC Initiator logical address position in message header
  * @{
  */
#define CEC_INITIATOR_LSB_POS           ((uint32_t) 4U)
/**
  * @}
  */

/** @defgroup CEC_OWN_ADDRESS   CEC Own Address
  * @{
  */
#define CEC_OWN_ADDRESS_NONE           ((uint16_t) 0x0000U)   /* Reset value */
#define CEC_OWN_ADDRESS_0              ((uint16_t) 0x0001U)   /* Logical Address 0 */
#define CEC_OWN_ADDRESS_1              ((uint16_t) 0x0002U)   /* Logical Address 1 */
#define CEC_OWN_ADDRESS_2              ((uint16_t) 0x0004U)   /* Logical Address 2 */
#define CEC_OWN_ADDRESS_3              ((uint16_t) 0x0008U)   /* Logical Address 3 */
#define CEC_OWN_ADDRESS_4              ((uint16_t) 0x0010U)   /* Logical Address 4 */
#define CEC_OWN_ADDRESS_5              ((uint16_t) 0x0020U)   /* Logical Address 5 */
#define CEC_OWN_ADDRESS_6              ((uint16_t) 0x0040U)   /* Logical Address 6 */
#define CEC_OWN_ADDRESS_7              ((uint16_t) 0x0080U)   /* Logical Address 7 */
#define CEC_OWN_ADDRESS_8              ((uint16_t) 0x0100U)   /* Logical Address 9 */
#define CEC_OWN_ADDRESS_9              ((uint16_t) 0x0200U)   /* Logical Address 10 */
#define CEC_OWN_ADDRESS_10             ((uint16_t) 0x0400U)   /* Logical Address 11 */
#define CEC_OWN_ADDRESS_11             ((uint16_t) 0x0800U)   /* Logical Address 12 */
#define CEC_OWN_ADDRESS_12             ((uint16_t) 0x1000U)   /* Logical Address 13 */
#define CEC_OWN_ADDRESS_13             ((uint16_t) 0x2000U)   /* Logical Address 14 */
#define CEC_OWN_ADDRESS_14             ((uint16_t) 0x4000U)   /* Logical Address 15 */
/**
  * @}
  */

/** @defgroup CEC_Interrupts_Definitions  CEC Interrupts definition
  * @{
  */
#define CEC_IT_TXACKE                   CEC_IER_TXACKEIE
#define CEC_IT_TXERR                    CEC_IER_TXERRIE
#define CEC_IT_TXUDR                    CEC_IER_TXUDRIE
#define CEC_IT_TXEND                    CEC_IER_TXENDIE
#define CEC_IT_TXBR                     CEC_IER_TXBRIE
#define CEC_IT_ARBLST                   CEC_IER_ARBLSTIE
#define CEC_IT_RXACKE                   CEC_IER_RXACKEIE
#define CEC_IT_LBPE                     CEC_IER_LBPEIE
#define CEC_IT_SBPE                     CEC_IER_SBPEIE
#define CEC_IT_BRE                      CEC_IER_BREIE
#define CEC_IT_RXOVR                    CEC_IER_RXOVRIE
#define CEC_IT_RXEND                    CEC_IER_RXENDIE
#define CEC_IT_RXBR                     CEC_IER_RXBRIE
/**
  * @}
  */

/** @defgroup CEC_Flags_Definitions  CEC Flags definition
  * @{
  */
#define CEC_FLAG_TXACKE                 CEC_ISR_TXACKE
#define CEC_FLAG_TXERR                  CEC_ISR_TXERR
#define CEC_FLAG_TXUDR                  CEC_ISR_TXUDR
#define CEC_FLAG_TXEND                  CEC_ISR_TXEND
#define CEC_FLAG_TXBR                   CEC_ISR_TXBR
#define CEC_FLAG_ARBLST                 CEC_ISR_ARBLST
#define CEC_FLAG_RXACKE                 CEC_ISR_RXACKE
#define CEC_FLAG_LBPE                   CEC_ISR_LBPE
#define CEC_FLAG_SBPE                   CEC_ISR_SBPE
#define CEC_FLAG_BRE                    CEC_ISR_BRE
#define CEC_FLAG_RXOVR                  CEC_ISR_RXOVR
#define CEC_FLAG_RXEND                  CEC_ISR_RXEND
#define CEC_FLAG_RXBR                   CEC_ISR_RXBR
/**
  * @}
  */

/** @defgroup CEC_ALL_ERROR CEC all RX or TX errors flags
  * @{
  */
#define CEC_ISR_ALL_ERROR              ((uint32_t)CEC_ISR_RXOVR|CEC_ISR_BRE|CEC_ISR_SBPE|CEC_ISR_LBPE|CEC_ISR_RXACKE|\
                                        CEC_ISR_ARBLST|CEC_ISR_TXUDR|CEC_ISR_TXERR|CEC_ISR_TXACKE)
/**
  * @}
  */

/** @defgroup CEC_IER_ALL_RX CEC all RX errors interrupts enabling flag
  * @{
  */
#define CEC_IER_RX_ALL_ERR              ((uint32_t)CEC_IER_RXACKEIE|CEC_IER_LBPEIE|CEC_IER_SBPEIE|CEC_IER_BREIE|CEC_IER_RXOVRIE)
/**
  * @}
  */

/** @defgroup CEC_IER_ALL_TX CEC all TX errors interrupts enabling flag
  * @{
  */
#define CEC_IER_TX_ALL_ERR              ((uint32_t)CEC_IER_TXACKEIE|CEC_IER_TXERRIE|CEC_IER_TXUDRIE|CEC_IER_ARBLSTIE)
/**
  * @}
  */

/**
  * @}
  */

/* Exported macros -----------------------------------------------------------*/
/** @defgroup CEC_Exported_Macros CEC Exported Macros
  * @{
  */

/** @brief  Reset CEC handle gstate & RxState
  * @param  __HANDLE__ CEC handle.
  * @retval None
  */
#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1)
#define __HAL_CEC_RESET_HANDLE_STATE(__HANDLE__) do{                                                   \
                                                       (__HANDLE__)->gState = HAL_CEC_STATE_RESET;     \
                                                       (__HANDLE__)->RxState = HAL_CEC_STATE_RESET;    \
                                                       (__HANDLE__)->MspInitCallback = NULL;           \
                                                       (__HANDLE__)->MspDeInitCallback = NULL;         \
                                                     } while(0)
#else
#define __HAL_CEC_RESET_HANDLE_STATE(__HANDLE__) do{                                                   \
                                                       (__HANDLE__)->gState = HAL_CEC_STATE_RESET;     \
                                                       (__HANDLE__)->RxState = HAL_CEC_STATE_RESET;    \
                                                     } while(0)
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */
/** @brief  Checks whether or not the specified CEC interrupt flag is set.
  * @param  __HANDLE__ specifies the CEC Handle.
  * @param  __FLAG__ specifies the flag to check.
  *            @arg CEC_FLAG_TXACKE: Tx Missing acknowledge Error
  *            @arg CEC_FLAG_TXERR: Tx Error.
  *            @arg CEC_FLAG_TXUDR: Tx-Buffer Underrun.
  *            @arg CEC_FLAG_TXEND: End of transmission (successful transmission of the last byte).
  *            @arg CEC_FLAG_TXBR: Tx-Byte Request.
  *            @arg CEC_FLAG_ARBLST: Arbitration Lost
  *            @arg CEC_FLAG_RXACKE: Rx-Missing Acknowledge
  *            @arg CEC_FLAG_LBPE: Rx Long period Error
  *            @arg CEC_FLAG_SBPE: Rx Short period Error
  *            @arg CEC_FLAG_BRE: Rx Bit Rising Error
  *            @arg CEC_FLAG_RXOVR: Rx Overrun.
  *            @arg CEC_FLAG_RXEND: End Of Reception.
  *            @arg CEC_FLAG_RXBR: Rx-Byte Received.
  * @retval ITStatus
  */
#define __HAL_CEC_GET_FLAG(__HANDLE__, __FLAG__)        ((__HANDLE__)->Instance->ISR & (__FLAG__))

/** @brief  Clears the interrupt or status flag when raised (write at 1)
  * @param  __HANDLE__ specifies the CEC Handle.
  * @param  __FLAG__ specifies the interrupt/status flag to clear.
  *        This parameter can be one of the following values:
  *            @arg CEC_FLAG_TXACKE: Tx Missing acknowledge Error
  *            @arg CEC_FLAG_TXERR: Tx Error.
  *            @arg CEC_FLAG_TXUDR: Tx-Buffer Underrun.
  *            @arg CEC_FLAG_TXEND: End of transmission (successful transmission of the last byte).
  *            @arg CEC_FLAG_TXBR: Tx-Byte Request.
  *            @arg CEC_FLAG_ARBLST: Arbitration Lost
  *            @arg CEC_FLAG_RXACKE: Rx-Missing Acknowledge
  *            @arg CEC_FLAG_LBPE: Rx Long period Error
  *            @arg CEC_FLAG_SBPE: Rx Short period Error
  *            @arg CEC_FLAG_BRE: Rx Bit Rising Error
  *            @arg CEC_FLAG_RXOVR: Rx Overrun.
  *            @arg CEC_FLAG_RXEND: End Of Reception.
  *            @arg CEC_FLAG_RXBR: Rx-Byte Received.
  * @retval none
  */
#define __HAL_CEC_CLEAR_FLAG(__HANDLE__, __FLAG__)         ((__HANDLE__)->Instance->ISR |= (__FLAG__))

/** @brief  Enables the specified CEC interrupt.
  * @param  __HANDLE__ specifies the CEC Handle.
  * @param  __INTERRUPT__ specifies the CEC interrupt to enable.
  *          This parameter can be one of the following values:
  *            @arg CEC_IT_TXACKE: Tx Missing acknowledge Error IT Enable
  *            @arg CEC_IT_TXERR: Tx Error IT Enable
  *            @arg CEC_IT_TXUDR: Tx-Buffer Underrun IT Enable
  *            @arg CEC_IT_TXEND: End of transmission IT Enable
  *            @arg CEC_IT_TXBR: Tx-Byte Request IT Enable
  *            @arg CEC_IT_ARBLST: Arbitration Lost IT Enable
  *            @arg CEC_IT_RXACKE: Rx-Missing Acknowledge IT Enable
  *            @arg CEC_IT_LBPE: Rx Long period Error IT Enable
  *            @arg CEC_IT_SBPE: Rx Short period Error IT Enable
  *            @arg CEC_IT_BRE: Rx Bit Rising Error IT Enable
  *            @arg CEC_IT_RXOVR: Rx Overrun IT Enable
  *            @arg CEC_IT_RXEND: End Of Reception IT Enable
  *            @arg CEC_IT_RXBR: Rx-Byte Received IT Enable
  * @retval none
  */
#define __HAL_CEC_ENABLE_IT(__HANDLE__, __INTERRUPT__)     ((__HANDLE__)->Instance->IER |= (__INTERRUPT__))

/** @brief  Disables the specified CEC interrupt.
  * @param  __HANDLE__ specifies the CEC Handle.
  * @param  __INTERRUPT__ specifies the CEC interrupt to disable.
  *          This parameter can be one of the following values:
  *            @arg CEC_IT_TXACKE: Tx Missing acknowledge Error IT Enable
  *            @arg CEC_IT_TXERR: Tx Error IT Enable
  *            @arg CEC_IT_TXUDR: Tx-Buffer Underrun IT Enable
  *            @arg CEC_IT_TXEND: End of transmission IT Enable
  *            @arg CEC_IT_TXBR: Tx-Byte Request IT Enable
  *            @arg CEC_IT_ARBLST: Arbitration Lost IT Enable
  *            @arg CEC_IT_RXACKE: Rx-Missing Acknowledge IT Enable
  *            @arg CEC_IT_LBPE: Rx Long period Error IT Enable
  *            @arg CEC_IT_SBPE: Rx Short period Error IT Enable
  *            @arg CEC_IT_BRE: Rx Bit Rising Error IT Enable
  *            @arg CEC_IT_RXOVR: Rx Overrun IT Enable
  *            @arg CEC_IT_RXEND: End Of Reception IT Enable
  *            @arg CEC_IT_RXBR: Rx-Byte Received IT Enable
  * @retval none
  */
#define __HAL_CEC_DISABLE_IT(__HANDLE__, __INTERRUPT__)    ((__HANDLE__)->Instance->IER &= (~(__INTERRUPT__)))

/** @brief  Checks whether or not the specified CEC interrupt is enabled.
  * @param  __HANDLE__ specifies the CEC Handle.
  * @param  __INTERRUPT__ specifies the CEC interrupt to check.
  *          This parameter can be one of the following values:
  *            @arg CEC_IT_TXACKE: Tx Missing acknowledge Error IT Enable
  *            @arg CEC_IT_TXERR: Tx Error IT Enable
  *            @arg CEC_IT_TXUDR: Tx-Buffer Underrun IT Enable
  *            @arg CEC_IT_TXEND: End of transmission IT Enable
  *            @arg CEC_IT_TXBR: Tx-Byte Request IT Enable
  *            @arg CEC_IT_ARBLST: Arbitration Lost IT Enable
  *            @arg CEC_IT_RXACKE: Rx-Missing Acknowledge IT Enable
  *            @arg CEC_IT_LBPE: Rx Long period Error IT Enable
  *            @arg CEC_IT_SBPE: Rx Short period Error IT Enable
  *            @arg CEC_IT_BRE: Rx Bit Rising Error IT Enable
  *            @arg CEC_IT_RXOVR: Rx Overrun IT Enable
  *            @arg CEC_IT_RXEND: End Of Reception IT Enable
  *            @arg CEC_IT_RXBR: Rx-Byte Received IT Enable
  * @retval FlagStatus
  */
#define __HAL_CEC_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->IER & (__INTERRUPT__))

/** @brief  Enables the CEC device
  * @param  __HANDLE__ specifies the CEC Handle.
  * @retval none
  */
#define __HAL_CEC_ENABLE(__HANDLE__)                   ((__HANDLE__)->Instance->CR |=  CEC_CR_CECEN)

/** @brief  Disables the CEC device
  * @param  __HANDLE__ specifies the CEC Handle.
  * @retval none
  */
#define __HAL_CEC_DISABLE(__HANDLE__)                  ((__HANDLE__)->Instance->CR &=  ~CEC_CR_CECEN)

/** @brief  Set Transmission Start flag
  * @param  __HANDLE__ specifies the CEC Handle.
  * @retval none
  */
#define __HAL_CEC_FIRST_BYTE_TX_SET(__HANDLE__)        ((__HANDLE__)->Instance->CR |=  CEC_CR_TXSOM)

/** @brief  Set Transmission End flag
  * @param  __HANDLE__ specifies the CEC Handle.
  * @retval none
  * If the CEC message consists of only one byte, TXEOM must be set before of TXSOM.
  */
#define __HAL_CEC_LAST_BYTE_TX_SET(__HANDLE__)         ((__HANDLE__)->Instance->CR |=  CEC_CR_TXEOM)

/** @brief  Get Transmission Start flag
  * @param  __HANDLE__ specifies the CEC Handle.
  * @retval FlagStatus
  */
#define __HAL_CEC_GET_TRANSMISSION_START_FLAG(__HANDLE__) ((__HANDLE__)->Instance->CR & CEC_CR_TXSOM)

/** @brief  Get Transmission End flag
  * @param  __HANDLE__ specifies the CEC Handle.
  * @retval FlagStatus
  */
#define __HAL_CEC_GET_TRANSMISSION_END_FLAG(__HANDLE__)   ((__HANDLE__)->Instance->CR & CEC_CR_TXEOM)

/** @brief  Clear OAR register
  * @param  __HANDLE__ specifies the CEC Handle.
  * @retval none
  */
#define __HAL_CEC_CLEAR_OAR(__HANDLE__)   CLEAR_BIT((__HANDLE__)->Instance->CFGR, CEC_CFGR_OAR)

/** @brief  Set OAR register (without resetting previously set address in case of multi-address mode)
  *          To reset OAR, __HAL_CEC_CLEAR_OAR() needs to be called beforehand
  * @param  __HANDLE__ specifies the CEC Handle.
  * @param  __ADDRESS__ Own Address value (CEC logical address is identified by bit position)
  * @retval none
  */
#define __HAL_CEC_SET_OAR(__HANDLE__,__ADDRESS__)   SET_BIT((__HANDLE__)->Instance->CFGR, (__ADDRESS__)<< CEC_CFGR_OAR_LSB_POS)

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup CEC_Exported_Functions
  * @{
  */

/** @addtogroup CEC_Exported_Functions_Group1
  * @{
  */
/* Initialization and de-initialization functions  ****************************/
HAL_StatusTypeDef HAL_CEC_Init(CEC_HandleTypeDef *hcec);
HAL_StatusTypeDef HAL_CEC_DeInit(CEC_HandleTypeDef *hcec);
HAL_StatusTypeDef HAL_CEC_SetDeviceAddress(CEC_HandleTypeDef *hcec, uint16_t CEC_OwnAddress);
void HAL_CEC_MspInit(CEC_HandleTypeDef *hcec);
void HAL_CEC_MspDeInit(CEC_HandleTypeDef *hcec);

#if (USE_HAL_CEC_REGISTER_CALLBACKS == 1)
HAL_StatusTypeDef HAL_CEC_RegisterCallback(CEC_HandleTypeDef *hcec, HAL_CEC_CallbackIDTypeDef CallbackID,
                                           pCEC_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_CEC_UnRegisterCallback(CEC_HandleTypeDef *hcec, HAL_CEC_CallbackIDTypeDef CallbackID);

HAL_StatusTypeDef HAL_CEC_RegisterRxCpltCallback(CEC_HandleTypeDef *hcec, pCEC_RxCallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_CEC_UnRegisterRxCpltCallback(CEC_HandleTypeDef *hcec);
#endif /* USE_HAL_CEC_REGISTER_CALLBACKS */
/**
  * @}
  */

/** @addtogroup CEC_Exported_Functions_Group2
  * @{
  */
/* I/O operation functions  ***************************************************/
HAL_StatusTypeDef HAL_CEC_Transmit_IT(CEC_HandleTypeDef *hcec, uint8_t InitiatorAddress, uint8_t DestinationAddress,
                                      uint8_t *pData, uint32_t Size);
uint32_t HAL_CEC_GetLastReceivedFrameSize(CEC_HandleTypeDef *hcec);
void HAL_CEC_ChangeRxBuffer(CEC_HandleTypeDef *hcec, uint8_t *Rxbuffer);
void HAL_CEC_IRQHandler(CEC_HandleTypeDef *hcec);
void HAL_CEC_TxCpltCallback(CEC_HandleTypeDef *hcec);
void HAL_CEC_RxCpltCallback(CEC_HandleTypeDef *hcec, uint32_t RxFrameSize);
void HAL_CEC_ErrorCallback(CEC_HandleTypeDef *hcec);
/**
  * @}
  */

/** @addtogroup CEC_Exported_Functions_Group3
  * @{
  */
/* Peripheral State functions  ************************************************/
HAL_CEC_StateTypeDef HAL_CEC_GetState(CEC_HandleTypeDef *hcec);
uint32_t HAL_CEC_GetError(CEC_HandleTypeDef *hcec);
/**
  * @}
  */

/**
  * @}
  */

/* Private types -------------------------------------------------------------*/
/** @defgroup CEC_Private_Types CEC Private Types
  * @{
  */

/**
  * @}
  */

/* Private variables ---------------------------------------------------------*/
/** @defgroup CEC_Private_Variables CEC Private Variables
  * @{
  */

/**
  * @}
  */

/* Private constants ---------------------------------------------------------*/
/** @defgroup CEC_Private_Constants CEC Private Constants
  * @{
  */

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup CEC_Private_Macros CEC Private Macros
  * @{
  */

#define IS_CEC_SIGNALFREETIME(__SFT__)     ((__SFT__) <= CEC_CFGR_SFT)

#define IS_CEC_TOLERANCE(__RXTOL__)        (((__RXTOL__) == CEC_STANDARD_TOLERANCE) || \
                                            ((__RXTOL__) == CEC_EXTENDED_TOLERANCE))

#define IS_CEC_BRERXSTOP(__BRERXSTOP__)   (((__BRERXSTOP__) == CEC_NO_RX_STOP_ON_BRE) || \
                                           ((__BRERXSTOP__) == CEC_RX_STOP_ON_BRE))

#define IS_CEC_BREERRORBITGEN(__ERRORBITGEN__) (((__ERRORBITGEN__) == CEC_BRE_ERRORBIT_NO_GENERATION) || \
                                                ((__ERRORBITGEN__) == CEC_BRE_ERRORBIT_GENERATION))

#define IS_CEC_LBPEERRORBITGEN(__ERRORBITGEN__) (((__ERRORBITGEN__) == CEC_LBPE_ERRORBIT_NO_GENERATION) || \
                                                 ((__ERRORBITGEN__) == CEC_LBPE_ERRORBIT_GENERATION))

#define IS_CEC_BROADCASTERROR_NO_ERRORBIT_GENERATION(__ERRORBITGEN__) (((__ERRORBITGEN__) == CEC_BROADCASTERROR_ERRORBIT_GENERATION) || \
                                                                       ((__ERRORBITGEN__) == CEC_BROADCASTERROR_NO_ERRORBIT_GENERATION))

#define IS_CEC_SFTOP(__SFTOP__)          (((__SFTOP__) == CEC_SFT_START_ON_TXSOM) || \
                                          ((__SFTOP__) == CEC_SFT_START_ON_TX_RX_END))

#define IS_CEC_LISTENING_MODE(__MODE__)     (((__MODE__) == CEC_REDUCED_LISTENING_MODE) || \
                                             ((__MODE__) == CEC_FULL_LISTENING_MODE))

/** @brief Check CEC message size.
  *       The message size is the payload size: without counting the header,
  *       it varies from 0 byte (ping operation, one header only, no payload) to
  *       15 bytes (1 opcode and up to 14 operands following the header).
  * @param  __SIZE__ CEC message size.
  * @retval Test result (TRUE or FALSE).
  */
#define IS_CEC_MSGSIZE(__SIZE__) ((__SIZE__) <= 0x10U)

/** @brief Check CEC device Own Address Register (OAR) setting.
  *        OAR address is written in a 15-bit field within CEC_CFGR register.
  * @param  __ADDRESS__ CEC own address.
  * @retval Test result (TRUE or FALSE).
  */
#define IS_CEC_OWN_ADDRESS(__ADDRESS__) ((__ADDRESS__) <= 0x7FFFU)

/** @brief Check CEC initiator or destination logical address setting.
  *        Initiator and destination addresses are coded over 4 bits.
  * @param  __ADDRESS__ CEC initiator or logical address.
  * @retval Test result (TRUE or FALSE).
  */
#define IS_CEC_ADDRESS(__ADDRESS__) ((__ADDRESS__) <= 0xFU)
/**
  * @}
  */
/* Private functions ---------------------------------------------------------*/
/** @defgroup CEC_Private_Functions CEC Private Functions
  * @{
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

#endif /* CEC */

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xxHAL_CEC_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
