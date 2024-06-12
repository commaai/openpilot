/**
  ******************************************************************************
  * @file    stm32f4xx_hal_fmpsmbus.h
  * @author  MCD Application Team
  * @brief   Header file of FMPSMBUS HAL module.
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
#ifndef STM32F4xx_HAL_FMPSMBUS_H
#define STM32F4xx_HAL_FMPSMBUS_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(FMPI2C_CR1_PE)
/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup FMPSMBUS
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup FMPSMBUS_Exported_Types FMPSMBUS Exported Types
  * @{
  */

/** @defgroup FMPSMBUS_Configuration_Structure_definition FMPSMBUS Configuration Structure definition
  * @brief  FMPSMBUS Configuration Structure definition
  * @{
  */
typedef struct
{
  uint32_t Timing;                 /*!< Specifies the FMPSMBUS_TIMINGR_register value.
                                        This parameter calculated by referring to FMPSMBUS initialization section
                                        in Reference manual */
  uint32_t AnalogFilter;           /*!< Specifies if Analog Filter is enable or not.
                                        This parameter can be a value of @ref FMPSMBUS_Analog_Filter */

  uint32_t OwnAddress1;            /*!< Specifies the first device own address.
                                        This parameter can be a 7-bit or 10-bit address. */

  uint32_t AddressingMode;         /*!< Specifies if 7-bit or 10-bit addressing mode for master is selected.
                                        This parameter can be a value of @ref FMPSMBUS_addressing_mode */

  uint32_t DualAddressMode;        /*!< Specifies if dual addressing mode is selected.
                                        This parameter can be a value of @ref FMPSMBUS_dual_addressing_mode */

  uint32_t OwnAddress2;            /*!< Specifies the second device own address if dual addressing mode is selected
                                        This parameter can be a 7-bit address. */

  uint32_t OwnAddress2Masks;       /*!< Specifies the acknowledge mask address second device own address
                                        if dual addressing mode is selected
                                        This parameter can be a value of @ref FMPSMBUS_own_address2_masks. */

  uint32_t GeneralCallMode;        /*!< Specifies if general call mode is selected.
                                        This parameter can be a value of @ref FMPSMBUS_general_call_addressing_mode. */

  uint32_t NoStretchMode;          /*!< Specifies if nostretch mode is selected.
                                        This parameter can be a value of @ref FMPSMBUS_nostretch_mode */

  uint32_t PacketErrorCheckMode;   /*!< Specifies if Packet Error Check mode is selected.
                                        This parameter can be a value of @ref FMPSMBUS_packet_error_check_mode */

  uint32_t PeripheralMode;         /*!< Specifies which mode of Periphal is selected.
                                        This parameter can be a value of @ref FMPSMBUS_peripheral_mode */

  uint32_t SMBusTimeout;           /*!< Specifies the content of the 32 Bits FMPSMBUS_TIMEOUT_register value.
                                        (Enable bits and different timeout values)
                                        This parameter calculated by referring to FMPSMBUS initialization section
                                        in Reference manual */
} FMPSMBUS_InitTypeDef;
/**
  * @}
  */

/** @defgroup HAL_state_definition HAL state definition
  * @brief  HAL State definition
  * @{
  */
#define HAL_FMPSMBUS_STATE_RESET           (0x00000000U)  /*!< FMPSMBUS not yet initialized or disabled         */
#define HAL_FMPSMBUS_STATE_READY           (0x00000001U)  /*!< FMPSMBUS initialized and ready for use           */
#define HAL_FMPSMBUS_STATE_BUSY            (0x00000002U)  /*!< FMPSMBUS internal process is ongoing             */
#define HAL_FMPSMBUS_STATE_MASTER_BUSY_TX  (0x00000012U)  /*!< Master Data Transmission process is ongoing   */
#define HAL_FMPSMBUS_STATE_MASTER_BUSY_RX  (0x00000022U)  /*!< Master Data Reception process is ongoing      */
#define HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX   (0x00000032U)  /*!< Slave Data Transmission process is ongoing    */
#define HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX   (0x00000042U)  /*!< Slave Data Reception process is ongoing       */
#define HAL_FMPSMBUS_STATE_TIMEOUT         (0x00000003U)  /*!< Timeout state                                 */
#define HAL_FMPSMBUS_STATE_ERROR           (0x00000004U)  /*!< Reception process is ongoing                  */
#define HAL_FMPSMBUS_STATE_LISTEN          (0x00000008U)  /*!< Address Listen Mode is ongoing                */
/**
  * @}
  */

/** @defgroup FMPSMBUS_Error_Code_definition FMPSMBUS Error Code definition
  * @brief  FMPSMBUS Error Code definition
  * @{
  */
#define HAL_FMPSMBUS_ERROR_NONE            (0x00000000U)    /*!< No error             */
#define HAL_FMPSMBUS_ERROR_BERR            (0x00000001U)    /*!< BERR error           */
#define HAL_FMPSMBUS_ERROR_ARLO            (0x00000002U)    /*!< ARLO error           */
#define HAL_FMPSMBUS_ERROR_ACKF            (0x00000004U)    /*!< ACKF error           */
#define HAL_FMPSMBUS_ERROR_OVR             (0x00000008U)    /*!< OVR error            */
#define HAL_FMPSMBUS_ERROR_HALTIMEOUT      (0x00000010U)    /*!< Timeout error        */
#define HAL_FMPSMBUS_ERROR_BUSTIMEOUT      (0x00000020U)    /*!< Bus Timeout error    */
#define HAL_FMPSMBUS_ERROR_ALERT           (0x00000040U)    /*!< Alert error          */
#define HAL_FMPSMBUS_ERROR_PECERR          (0x00000080U)    /*!< PEC error            */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
#define HAL_FMPSMBUS_ERROR_INVALID_CALLBACK  (0x00000100U)  /*!< Invalid Callback error   */
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
#define HAL_FMPSMBUS_ERROR_INVALID_PARAM    (0x00000200U)   /*!< Invalid Parameters error */
/**
  * @}
  */

/** @defgroup FMPSMBUS_handle_Structure_definition FMPSMBUS handle Structure definition
  * @brief  FMPSMBUS handle Structure definition
  * @{
  */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
typedef struct __FMPSMBUS_HandleTypeDef
#else
typedef struct
#endif  /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
{
  FMPI2C_TypeDef                  *Instance;       /*!< FMPSMBUS registers base address       */

  FMPSMBUS_InitTypeDef            Init;            /*!< FMPSMBUS communication parameters     */

  uint8_t                      *pBuffPtr;       /*!< Pointer to FMPSMBUS transfer buffer   */

  uint16_t                     XferSize;        /*!< FMPSMBUS transfer size                */

  __IO uint16_t                XferCount;       /*!< FMPSMBUS transfer counter             */

  __IO uint32_t                XferOptions;     /*!< FMPSMBUS transfer options             */

  __IO uint32_t                PreviousState;   /*!< FMPSMBUS communication Previous state */

  HAL_LockTypeDef              Lock;            /*!< FMPSMBUS locking object               */

  __IO uint32_t                State;           /*!< FMPSMBUS communication state          */

  __IO uint32_t                ErrorCode;       /*!< FMPSMBUS Error code                   */

#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
  void (* MasterTxCpltCallback)(struct __FMPSMBUS_HandleTypeDef *hfmpsmbus);
  /*!< FMPSMBUS Master Tx Transfer completed callback */
  void (* MasterRxCpltCallback)(struct __FMPSMBUS_HandleTypeDef *hfmpsmbus);
  /*!< FMPSMBUS Master Rx Transfer completed callback */
  void (* SlaveTxCpltCallback)(struct __FMPSMBUS_HandleTypeDef *hfmpsmbus);
  /*!< FMPSMBUS Slave Tx Transfer completed callback  */
  void (* SlaveRxCpltCallback)(struct __FMPSMBUS_HandleTypeDef *hfmpsmbus);
  /*!< FMPSMBUS Slave Rx Transfer completed callback  */
  void (* ListenCpltCallback)(struct __FMPSMBUS_HandleTypeDef *hfmpsmbus);
  /*!< FMPSMBUS Listen Complete callback              */
  void (* ErrorCallback)(struct __FMPSMBUS_HandleTypeDef *hfmpsmbus);
  /*!< FMPSMBUS Error callback                        */

  void (* AddrCallback)(struct __FMPSMBUS_HandleTypeDef *hfmpsmbus, uint8_t TransferDirection, uint16_t AddrMatchCode);
  /*!< FMPSMBUS Slave Address Match callback */

  void (* MspInitCallback)(struct __FMPSMBUS_HandleTypeDef *hfmpsmbus);
  /*!< FMPSMBUS Msp Init callback                     */
  void (* MspDeInitCallback)(struct __FMPSMBUS_HandleTypeDef *hfmpsmbus);
  /*!< FMPSMBUS Msp DeInit callback                   */

#endif  /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
} FMPSMBUS_HandleTypeDef;

#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL FMPSMBUS Callback ID enumeration definition
  */
typedef enum
{
  HAL_FMPSMBUS_MASTER_TX_COMPLETE_CB_ID      = 0x00U,    /*!< FMPSMBUS Master Tx Transfer completed callback ID  */
  HAL_FMPSMBUS_MASTER_RX_COMPLETE_CB_ID      = 0x01U,    /*!< FMPSMBUS Master Rx Transfer completed callback ID  */
  HAL_FMPSMBUS_SLAVE_TX_COMPLETE_CB_ID       = 0x02U,    /*!< FMPSMBUS Slave Tx Transfer completed callback ID   */
  HAL_FMPSMBUS_SLAVE_RX_COMPLETE_CB_ID       = 0x03U,    /*!< FMPSMBUS Slave Rx Transfer completed callback ID   */
  HAL_FMPSMBUS_LISTEN_COMPLETE_CB_ID         = 0x04U,    /*!< FMPSMBUS Listen Complete callback ID               */
  HAL_FMPSMBUS_ERROR_CB_ID                   = 0x05U,    /*!< FMPSMBUS Error callback ID                         */

  HAL_FMPSMBUS_MSPINIT_CB_ID                 = 0x06U,    /*!< FMPSMBUS Msp Init callback ID                      */
  HAL_FMPSMBUS_MSPDEINIT_CB_ID               = 0x07U     /*!< FMPSMBUS Msp DeInit callback ID                    */

} HAL_FMPSMBUS_CallbackIDTypeDef;

/**
  * @brief  HAL FMPSMBUS Callback pointer definition
  */
typedef  void (*pFMPSMBUS_CallbackTypeDef)(FMPSMBUS_HandleTypeDef *hfmpsmbus);
/*!< pointer to an FMPSMBUS callback function */
typedef  void (*pFMPSMBUS_AddrCallbackTypeDef)(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint8_t TransferDirection,
                                            uint16_t AddrMatchCode);
/*!< pointer to an FMPSMBUS Address Match callback function */

#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
/**
  * @}
  */

/**
  * @}
  */
/* Exported constants --------------------------------------------------------*/

/** @defgroup FMPSMBUS_Exported_Constants FMPSMBUS Exported Constants
  * @{
  */

/** @defgroup FMPSMBUS_Analog_Filter FMPSMBUS Analog Filter
  * @{
  */
#define FMPSMBUS_ANALOGFILTER_ENABLE               (0x00000000U)
#define FMPSMBUS_ANALOGFILTER_DISABLE              FMPI2C_CR1_ANFOFF
/**
  * @}
  */

/** @defgroup FMPSMBUS_addressing_mode FMPSMBUS addressing mode
  * @{
  */
#define FMPSMBUS_ADDRESSINGMODE_7BIT               (0x00000001U)
#define FMPSMBUS_ADDRESSINGMODE_10BIT              (0x00000002U)
/**
  * @}
  */

/** @defgroup FMPSMBUS_dual_addressing_mode FMPSMBUS dual addressing mode
  * @{
  */

#define FMPSMBUS_DUALADDRESS_DISABLE               (0x00000000U)
#define FMPSMBUS_DUALADDRESS_ENABLE                FMPI2C_OAR2_OA2EN
/**
  * @}
  */

/** @defgroup FMPSMBUS_own_address2_masks FMPSMBUS ownaddress2 masks
  * @{
  */

#define FMPSMBUS_OA2_NOMASK                        ((uint8_t)0x00U)
#define FMPSMBUS_OA2_MASK01                        ((uint8_t)0x01U)
#define FMPSMBUS_OA2_MASK02                        ((uint8_t)0x02U)
#define FMPSMBUS_OA2_MASK03                        ((uint8_t)0x03U)
#define FMPSMBUS_OA2_MASK04                        ((uint8_t)0x04U)
#define FMPSMBUS_OA2_MASK05                        ((uint8_t)0x05U)
#define FMPSMBUS_OA2_MASK06                        ((uint8_t)0x06U)
#define FMPSMBUS_OA2_MASK07                        ((uint8_t)0x07U)
/**
  * @}
  */


/** @defgroup FMPSMBUS_general_call_addressing_mode FMPSMBUS general call addressing mode
  * @{
  */
#define FMPSMBUS_GENERALCALL_DISABLE               (0x00000000U)
#define FMPSMBUS_GENERALCALL_ENABLE                FMPI2C_CR1_GCEN
/**
  * @}
  */

/** @defgroup FMPSMBUS_nostretch_mode FMPSMBUS nostretch mode
  * @{
  */
#define FMPSMBUS_NOSTRETCH_DISABLE                 (0x00000000U)
#define FMPSMBUS_NOSTRETCH_ENABLE                  FMPI2C_CR1_NOSTRETCH
/**
  * @}
  */

/** @defgroup FMPSMBUS_packet_error_check_mode FMPSMBUS packet error check mode
  * @{
  */
#define FMPSMBUS_PEC_DISABLE                       (0x00000000U)
#define FMPSMBUS_PEC_ENABLE                        FMPI2C_CR1_PECEN
/**
  * @}
  */

/** @defgroup FMPSMBUS_peripheral_mode FMPSMBUS peripheral mode
  * @{
  */
#define FMPSMBUS_PERIPHERAL_MODE_FMPSMBUS_HOST        FMPI2C_CR1_SMBHEN
#define FMPSMBUS_PERIPHERAL_MODE_FMPSMBUS_SLAVE       (0x00000000U)
#define FMPSMBUS_PERIPHERAL_MODE_FMPSMBUS_SLAVE_ARP   FMPI2C_CR1_SMBDEN
/**
  * @}
  */

/** @defgroup FMPSMBUS_ReloadEndMode_definition FMPSMBUS ReloadEndMode definition
  * @{
  */

#define  FMPSMBUS_SOFTEND_MODE                     (0x00000000U)
#define  FMPSMBUS_RELOAD_MODE                      FMPI2C_CR2_RELOAD
#define  FMPSMBUS_AUTOEND_MODE                     FMPI2C_CR2_AUTOEND
#define  FMPSMBUS_SENDPEC_MODE                     FMPI2C_CR2_PECBYTE
/**
  * @}
  */

/** @defgroup FMPSMBUS_StartStopMode_definition FMPSMBUS StartStopMode definition
  * @{
  */

#define  FMPSMBUS_NO_STARTSTOP                     (0x00000000U)
#define  FMPSMBUS_GENERATE_STOP                    (uint32_t)(0x80000000U | FMPI2C_CR2_STOP)
#define  FMPSMBUS_GENERATE_START_READ              (uint32_t)(0x80000000U | FMPI2C_CR2_START | FMPI2C_CR2_RD_WRN)
#define  FMPSMBUS_GENERATE_START_WRITE             (uint32_t)(0x80000000U | FMPI2C_CR2_START)
/**
  * @}
  */

/** @defgroup FMPSMBUS_XferOptions_definition FMPSMBUS XferOptions definition
  * @{
  */

/* List of XferOptions in usage of :
 * 1- Restart condition when direction change
 * 2- No Restart condition in other use cases
 */
#define  FMPSMBUS_FIRST_FRAME                      FMPSMBUS_SOFTEND_MODE
#define  FMPSMBUS_NEXT_FRAME                       ((uint32_t)(FMPSMBUS_RELOAD_MODE | FMPSMBUS_SOFTEND_MODE))
#define  FMPSMBUS_FIRST_AND_LAST_FRAME_NO_PEC      FMPSMBUS_AUTOEND_MODE
#define  FMPSMBUS_LAST_FRAME_NO_PEC                FMPSMBUS_AUTOEND_MODE
#define  FMPSMBUS_FIRST_FRAME_WITH_PEC             ((uint32_t)(FMPSMBUS_SOFTEND_MODE | FMPSMBUS_SENDPEC_MODE))
#define  FMPSMBUS_FIRST_AND_LAST_FRAME_WITH_PEC    ((uint32_t)(FMPSMBUS_AUTOEND_MODE | FMPSMBUS_SENDPEC_MODE))
#define  FMPSMBUS_LAST_FRAME_WITH_PEC              ((uint32_t)(FMPSMBUS_AUTOEND_MODE | FMPSMBUS_SENDPEC_MODE))

/* List of XferOptions in usage of :
 * 1- Restart condition in all use cases (direction change or not)
 */
#define  FMPSMBUS_OTHER_FRAME_NO_PEC               (0x000000AAU)
#define  FMPSMBUS_OTHER_FRAME_WITH_PEC             (0x0000AA00U)
#define  FMPSMBUS_OTHER_AND_LAST_FRAME_NO_PEC      (0x00AA0000U)
#define  FMPSMBUS_OTHER_AND_LAST_FRAME_WITH_PEC    (0xAA000000U)
/**
  * @}
  */

/** @defgroup FMPSMBUS_Interrupt_configuration_definition FMPSMBUS Interrupt configuration definition
  * @brief FMPSMBUS Interrupt definition
  *        Elements values convention: 0xXXXXXXXX
  *           - XXXXXXXX  : Interrupt control mask
  * @{
  */
#define FMPSMBUS_IT_ERRI                           FMPI2C_CR1_ERRIE
#define FMPSMBUS_IT_TCI                            FMPI2C_CR1_TCIE
#define FMPSMBUS_IT_STOPI                          FMPI2C_CR1_STOPIE
#define FMPSMBUS_IT_NACKI                          FMPI2C_CR1_NACKIE
#define FMPSMBUS_IT_ADDRI                          FMPI2C_CR1_ADDRIE
#define FMPSMBUS_IT_RXI                            FMPI2C_CR1_RXIE
#define FMPSMBUS_IT_TXI                            FMPI2C_CR1_TXIE
#define FMPSMBUS_IT_TX                             (FMPSMBUS_IT_ERRI | FMPSMBUS_IT_TCI | FMPSMBUS_IT_STOPI | \
                                                   FMPSMBUS_IT_NACKI | FMPSMBUS_IT_TXI)
#define FMPSMBUS_IT_RX                             (FMPSMBUS_IT_ERRI | FMPSMBUS_IT_TCI | FMPSMBUS_IT_NACKI | \
                                                   FMPSMBUS_IT_RXI)
#define FMPSMBUS_IT_ALERT                          (FMPSMBUS_IT_ERRI)
#define FMPSMBUS_IT_ADDR                           (FMPSMBUS_IT_ADDRI | FMPSMBUS_IT_STOPI | FMPSMBUS_IT_NACKI)
/**
  * @}
  */

/** @defgroup FMPSMBUS_Flag_definition FMPSMBUS Flag definition
  * @brief Flag definition
  *        Elements values convention: 0xXXXXYYYY
  *           - XXXXXXXX  : Flag mask
  * @{
  */

#define  FMPSMBUS_FLAG_TXE                         FMPI2C_ISR_TXE
#define  FMPSMBUS_FLAG_TXIS                        FMPI2C_ISR_TXIS
#define  FMPSMBUS_FLAG_RXNE                        FMPI2C_ISR_RXNE
#define  FMPSMBUS_FLAG_ADDR                        FMPI2C_ISR_ADDR
#define  FMPSMBUS_FLAG_AF                          FMPI2C_ISR_NACKF
#define  FMPSMBUS_FLAG_STOPF                       FMPI2C_ISR_STOPF
#define  FMPSMBUS_FLAG_TC                          FMPI2C_ISR_TC
#define  FMPSMBUS_FLAG_TCR                         FMPI2C_ISR_TCR
#define  FMPSMBUS_FLAG_BERR                        FMPI2C_ISR_BERR
#define  FMPSMBUS_FLAG_ARLO                        FMPI2C_ISR_ARLO
#define  FMPSMBUS_FLAG_OVR                         FMPI2C_ISR_OVR
#define  FMPSMBUS_FLAG_PECERR                      FMPI2C_ISR_PECERR
#define  FMPSMBUS_FLAG_TIMEOUT                     FMPI2C_ISR_TIMEOUT
#define  FMPSMBUS_FLAG_ALERT                       FMPI2C_ISR_ALERT
#define  FMPSMBUS_FLAG_BUSY                        FMPI2C_ISR_BUSY
#define  FMPSMBUS_FLAG_DIR                         FMPI2C_ISR_DIR
/**
  * @}
  */

/**
  * @}
  */

/* Exported macros ------------------------------------------------------------*/
/** @defgroup FMPSMBUS_Exported_Macros FMPSMBUS Exported Macros
  * @{
  */

/** @brief  Reset FMPSMBUS handle state.
  * @param  __HANDLE__ specifies the FMPSMBUS Handle.
  * @retval None
  */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
#define __HAL_FMPSMBUS_RESET_HANDLE_STATE(__HANDLE__)           do{                                               \
                                                                 (__HANDLE__)->State = HAL_FMPSMBUS_STATE_RESET;  \
                                                                 (__HANDLE__)->MspInitCallback = NULL;            \
                                                                 (__HANDLE__)->MspDeInitCallback = NULL;          \
                                                               } while(0)
#else
#define __HAL_FMPSMBUS_RESET_HANDLE_STATE(__HANDLE__)         ((__HANDLE__)->State = HAL_FMPSMBUS_STATE_RESET)
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */

/** @brief  Enable the specified FMPSMBUS interrupts.
  * @param  __HANDLE__ specifies the FMPSMBUS Handle.
  * @param  __INTERRUPT__ specifies the interrupt source to enable.
  *        This parameter can be one of the following values:
  *            @arg @ref FMPSMBUS_IT_ERRI  Errors interrupt enable
  *            @arg @ref FMPSMBUS_IT_TCI   Transfer complete interrupt enable
  *            @arg @ref FMPSMBUS_IT_STOPI STOP detection interrupt enable
  *            @arg @ref FMPSMBUS_IT_NACKI NACK received interrupt enable
  *            @arg @ref FMPSMBUS_IT_ADDRI Address match interrupt enable
  *            @arg @ref FMPSMBUS_IT_RXI   RX interrupt enable
  *            @arg @ref FMPSMBUS_IT_TXI   TX interrupt enable
  *
  * @retval None
  */
#define __HAL_FMPSMBUS_ENABLE_IT(__HANDLE__, __INTERRUPT__)   ((__HANDLE__)->Instance->CR1 |= (__INTERRUPT__))

/** @brief  Disable the specified FMPSMBUS interrupts.
  * @param  __HANDLE__ specifies the FMPSMBUS Handle.
  * @param  __INTERRUPT__ specifies the interrupt source to disable.
  *        This parameter can be one of the following values:
  *            @arg @ref FMPSMBUS_IT_ERRI  Errors interrupt enable
  *            @arg @ref FMPSMBUS_IT_TCI   Transfer complete interrupt enable
  *            @arg @ref FMPSMBUS_IT_STOPI STOP detection interrupt enable
  *            @arg @ref FMPSMBUS_IT_NACKI NACK received interrupt enable
  *            @arg @ref FMPSMBUS_IT_ADDRI Address match interrupt enable
  *            @arg @ref FMPSMBUS_IT_RXI   RX interrupt enable
  *            @arg @ref FMPSMBUS_IT_TXI   TX interrupt enable
  *
  * @retval None
  */
#define __HAL_FMPSMBUS_DISABLE_IT(__HANDLE__, __INTERRUPT__)  ((__HANDLE__)->Instance->CR1 &= (~(__INTERRUPT__)))

/** @brief  Check whether the specified FMPSMBUS interrupt source is enabled or not.
  * @param  __HANDLE__ specifies the FMPSMBUS Handle.
  * @param  __INTERRUPT__ specifies the FMPSMBUS interrupt source to check.
  *          This parameter can be one of the following values:
  *            @arg @ref FMPSMBUS_IT_ERRI  Errors interrupt enable
  *            @arg @ref FMPSMBUS_IT_TCI   Transfer complete interrupt enable
  *            @arg @ref FMPSMBUS_IT_STOPI STOP detection interrupt enable
  *            @arg @ref FMPSMBUS_IT_NACKI NACK received interrupt enable
  *            @arg @ref FMPSMBUS_IT_ADDRI Address match interrupt enable
  *            @arg @ref FMPSMBUS_IT_RXI   RX interrupt enable
  *            @arg @ref FMPSMBUS_IT_TXI   TX interrupt enable
  *
  * @retval The new state of __IT__ (SET or RESET).
  */
#define __HAL_FMPSMBUS_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) \
  ((((__HANDLE__)->Instance->CR1 & (__INTERRUPT__)) == (__INTERRUPT__)) ? SET : RESET)

/** @brief  Check whether the specified FMPSMBUS flag is set or not.
  * @param  __HANDLE__ specifies the FMPSMBUS Handle.
  * @param  __FLAG__ specifies the flag to check.
  *        This parameter can be one of the following values:
  *            @arg @ref FMPSMBUS_FLAG_TXE     Transmit data register empty
  *            @arg @ref FMPSMBUS_FLAG_TXIS    Transmit interrupt status
  *            @arg @ref FMPSMBUS_FLAG_RXNE    Receive data register not empty
  *            @arg @ref FMPSMBUS_FLAG_ADDR    Address matched (slave mode)
  *            @arg @ref FMPSMBUS_FLAG_AF      NACK received flag
  *            @arg @ref FMPSMBUS_FLAG_STOPF   STOP detection flag
  *            @arg @ref FMPSMBUS_FLAG_TC      Transfer complete (master mode)
  *            @arg @ref FMPSMBUS_FLAG_TCR     Transfer complete reload
  *            @arg @ref FMPSMBUS_FLAG_BERR    Bus error
  *            @arg @ref FMPSMBUS_FLAG_ARLO    Arbitration lost
  *            @arg @ref FMPSMBUS_FLAG_OVR     Overrun/Underrun
  *            @arg @ref FMPSMBUS_FLAG_PECERR  PEC error in reception
  *            @arg @ref FMPSMBUS_FLAG_TIMEOUT Timeout or Tlow detection flag
  *            @arg @ref FMPSMBUS_FLAG_ALERT   SMBus alert
  *            @arg @ref FMPSMBUS_FLAG_BUSY    Bus busy
  *            @arg @ref FMPSMBUS_FLAG_DIR     Transfer direction (slave mode)
  *
  * @retval The new state of __FLAG__ (SET or RESET).
  */
#define FMPSMBUS_FLAG_MASK  (0x0001FFFFU)
#define __HAL_FMPSMBUS_GET_FLAG(__HANDLE__, __FLAG__) \
  (((((__HANDLE__)->Instance->ISR) & ((__FLAG__) & FMPSMBUS_FLAG_MASK)) == \
    ((__FLAG__) & FMPSMBUS_FLAG_MASK)) ? SET : RESET)

/** @brief  Clear the FMPSMBUS pending flags which are cleared by writing 1 in a specific bit.
  * @param  __HANDLE__ specifies the FMPSMBUS Handle.
  * @param  __FLAG__ specifies the flag to clear.
  *          This parameter can be any combination of the following values:
  *            @arg @ref FMPSMBUS_FLAG_ADDR    Address matched (slave mode)
  *            @arg @ref FMPSMBUS_FLAG_AF      NACK received flag
  *            @arg @ref FMPSMBUS_FLAG_STOPF   STOP detection flag
  *            @arg @ref FMPSMBUS_FLAG_BERR    Bus error
  *            @arg @ref FMPSMBUS_FLAG_ARLO    Arbitration lost
  *            @arg @ref FMPSMBUS_FLAG_OVR     Overrun/Underrun
  *            @arg @ref FMPSMBUS_FLAG_PECERR  PEC error in reception
  *            @arg @ref FMPSMBUS_FLAG_TIMEOUT Timeout or Tlow detection flag
  *            @arg @ref FMPSMBUS_FLAG_ALERT   SMBus alert
  *
  * @retval None
  */
#define __HAL_FMPSMBUS_CLEAR_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->ICR = (__FLAG__))

/** @brief  Enable the specified FMPSMBUS peripheral.
  * @param  __HANDLE__ specifies the FMPSMBUS Handle.
  * @retval None
  */
#define __HAL_FMPSMBUS_ENABLE(__HANDLE__)                  (SET_BIT((__HANDLE__)->Instance->CR1, FMPI2C_CR1_PE))

/** @brief  Disable the specified FMPSMBUS peripheral.
  * @param  __HANDLE__ specifies the FMPSMBUS Handle.
  * @retval None
  */
#define __HAL_FMPSMBUS_DISABLE(__HANDLE__)                 (CLEAR_BIT((__HANDLE__)->Instance->CR1, FMPI2C_CR1_PE))

/** @brief  Generate a Non-Acknowledge FMPSMBUS peripheral in Slave mode.
  * @param  __HANDLE__ specifies the FMPSMBUS Handle.
  * @retval None
  */
#define __HAL_FMPSMBUS_GENERATE_NACK(__HANDLE__)           (SET_BIT((__HANDLE__)->Instance->CR2, FMPI2C_CR2_NACK))

/**
  * @}
  */


/* Private constants ---------------------------------------------------------*/

/* Private macros ------------------------------------------------------------*/
/** @defgroup FMPSMBUS_Private_Macro FMPSMBUS Private Macros
  * @{
  */

#define IS_FMPSMBUS_ANALOG_FILTER(FILTER)                  (((FILTER) == FMPSMBUS_ANALOGFILTER_ENABLE) || \
                                                         ((FILTER) == FMPSMBUS_ANALOGFILTER_DISABLE))

#define IS_FMPSMBUS_DIGITAL_FILTER(FILTER)                 ((FILTER) <= 0x0000000FU)

#define IS_FMPSMBUS_ADDRESSING_MODE(MODE)                  (((MODE) == FMPSMBUS_ADDRESSINGMODE_7BIT)  || \
                                                         ((MODE) == FMPSMBUS_ADDRESSINGMODE_10BIT))

#define IS_FMPSMBUS_DUAL_ADDRESS(ADDRESS)                  (((ADDRESS) == FMPSMBUS_DUALADDRESS_DISABLE) || \
                                                         ((ADDRESS) == FMPSMBUS_DUALADDRESS_ENABLE))

#define IS_FMPSMBUS_OWN_ADDRESS2_MASK(MASK)                (((MASK) == FMPSMBUS_OA2_NOMASK)    || \
                                                         ((MASK) == FMPSMBUS_OA2_MASK01)    || \
                                                         ((MASK) == FMPSMBUS_OA2_MASK02)    || \
                                                         ((MASK) == FMPSMBUS_OA2_MASK03)    || \
                                                         ((MASK) == FMPSMBUS_OA2_MASK04)    || \
                                                         ((MASK) == FMPSMBUS_OA2_MASK05)    || \
                                                         ((MASK) == FMPSMBUS_OA2_MASK06)    || \
                                                         ((MASK) == FMPSMBUS_OA2_MASK07))

#define IS_FMPSMBUS_GENERAL_CALL(CALL)                     (((CALL) == FMPSMBUS_GENERALCALL_DISABLE) || \
                                                         ((CALL) == FMPSMBUS_GENERALCALL_ENABLE))

#define IS_FMPSMBUS_NO_STRETCH(STRETCH)                    (((STRETCH) == FMPSMBUS_NOSTRETCH_DISABLE) || \
                                                         ((STRETCH) == FMPSMBUS_NOSTRETCH_ENABLE))

#define IS_FMPSMBUS_PEC(PEC)                               (((PEC) == FMPSMBUS_PEC_DISABLE) || \
                                                         ((PEC) == FMPSMBUS_PEC_ENABLE))

#define IS_FMPSMBUS_PERIPHERAL_MODE(MODE)                  (((MODE) == FMPSMBUS_PERIPHERAL_MODE_FMPSMBUS_HOST)   || \
                                                         ((MODE) == FMPSMBUS_PERIPHERAL_MODE_FMPSMBUS_SLAVE)  || \
                                                         ((MODE) == FMPSMBUS_PERIPHERAL_MODE_FMPSMBUS_SLAVE_ARP))

#define IS_FMPSMBUS_TRANSFER_MODE(MODE)                 (((MODE) == FMPSMBUS_RELOAD_MODE)                          || \
                                                      ((MODE) == FMPSMBUS_AUTOEND_MODE)                         || \
                                                      ((MODE) == FMPSMBUS_SOFTEND_MODE)                         || \
                                                      ((MODE) == FMPSMBUS_SENDPEC_MODE)                         || \
                                                      ((MODE) == (FMPSMBUS_RELOAD_MODE | FMPSMBUS_SENDPEC_MODE))   || \
                                                      ((MODE) == (FMPSMBUS_AUTOEND_MODE | FMPSMBUS_SENDPEC_MODE))  || \
                                                      ((MODE) == (FMPSMBUS_AUTOEND_MODE | FMPSMBUS_RELOAD_MODE))   || \
                                                      ((MODE) == (FMPSMBUS_AUTOEND_MODE | FMPSMBUS_SENDPEC_MODE | \
                                                                  FMPSMBUS_RELOAD_MODE )))


#define IS_FMPSMBUS_TRANSFER_REQUEST(REQUEST)              (((REQUEST) == FMPSMBUS_GENERATE_STOP)              || \
                                                         ((REQUEST) == FMPSMBUS_GENERATE_START_READ)        || \
                                                         ((REQUEST) == FMPSMBUS_GENERATE_START_WRITE)       || \
                                                         ((REQUEST) == FMPSMBUS_NO_STARTSTOP))


#define IS_FMPSMBUS_TRANSFER_OPTIONS_REQUEST(REQUEST)   (IS_FMPSMBUS_TRANSFER_OTHER_OPTIONS_REQUEST(REQUEST)       || \
                                                      ((REQUEST) == FMPSMBUS_FIRST_FRAME)                       || \
                                                      ((REQUEST) == FMPSMBUS_NEXT_FRAME)                        || \
                                                      ((REQUEST) == FMPSMBUS_FIRST_AND_LAST_FRAME_NO_PEC)       || \
                                                      ((REQUEST) == FMPSMBUS_LAST_FRAME_NO_PEC)                 || \
                                                      ((REQUEST) == FMPSMBUS_FIRST_FRAME_WITH_PEC)              || \
                                                      ((REQUEST) == FMPSMBUS_FIRST_AND_LAST_FRAME_WITH_PEC)     || \
                                                      ((REQUEST) == FMPSMBUS_LAST_FRAME_WITH_PEC))

#define IS_FMPSMBUS_TRANSFER_OTHER_OPTIONS_REQUEST(REQUEST) (((REQUEST) == FMPSMBUS_OTHER_FRAME_NO_PEC)             || \
                                                          ((REQUEST) == FMPSMBUS_OTHER_AND_LAST_FRAME_NO_PEC)    || \
                                                          ((REQUEST) == FMPSMBUS_OTHER_FRAME_WITH_PEC)           || \
                                                          ((REQUEST) == FMPSMBUS_OTHER_AND_LAST_FRAME_WITH_PEC))

#define FMPSMBUS_RESET_CR1(__HANDLE__)                    ((__HANDLE__)->Instance->CR1 &= \
                                                        (uint32_t)~((uint32_t)(FMPI2C_CR1_SMBHEN | FMPI2C_CR1_SMBDEN | \
                                                                    FMPI2C_CR1_PECEN)))
#define FMPSMBUS_RESET_CR2(__HANDLE__)                    ((__HANDLE__)->Instance->CR2 &= \
                                                        (uint32_t)~((uint32_t)(FMPI2C_CR2_SADD | FMPI2C_CR2_HEAD10R | \
                                                                    FMPI2C_CR2_NBYTES | FMPI2C_CR2_RELOAD | \
                                                                    FMPI2C_CR2_RD_WRN)))

#define FMPSMBUS_GENERATE_START(__ADDMODE__,__ADDRESS__)     (((__ADDMODE__) == FMPSMBUS_ADDRESSINGMODE_7BIT) ? \
                                                           (uint32_t)((((uint32_t)(__ADDRESS__) & (FMPI2C_CR2_SADD)) | \
                                                                       (FMPI2C_CR2_START) | (FMPI2C_CR2_AUTOEND)) & \
                                                                      (~FMPI2C_CR2_RD_WRN)) : \
                                                           (uint32_t)((((uint32_t)(__ADDRESS__) & \
                                                                        (FMPI2C_CR2_SADD)) | (FMPI2C_CR2_ADD10) | \
                                                                       (FMPI2C_CR2_START)) & (~FMPI2C_CR2_RD_WRN)))

#define FMPSMBUS_GET_ADDR_MATCH(__HANDLE__)                  (((__HANDLE__)->Instance->ISR & FMPI2C_ISR_ADDCODE) >> 17U)
#define FMPSMBUS_GET_DIR(__HANDLE__)                         (((__HANDLE__)->Instance->ISR & FMPI2C_ISR_DIR) >> 16U)
#define FMPSMBUS_GET_STOP_MODE(__HANDLE__)                   ((__HANDLE__)->Instance->CR2 & FMPI2C_CR2_AUTOEND)
#define FMPSMBUS_GET_PEC_MODE(__HANDLE__)                    ((__HANDLE__)->Instance->CR2 & FMPI2C_CR2_PECBYTE)
#define FMPSMBUS_GET_ALERT_ENABLED(__HANDLE__)                ((__HANDLE__)->Instance->CR1 & FMPI2C_CR1_ALERTEN)

#define FMPSMBUS_CHECK_FLAG(__ISR__, __FLAG__)             ((((__ISR__) & ((__FLAG__) & FMPSMBUS_FLAG_MASK)) == \
                                                          ((__FLAG__) & FMPSMBUS_FLAG_MASK)) ? SET : RESET)
#define FMPSMBUS_CHECK_IT_SOURCE(__CR1__, __IT__)          ((((__CR1__) & (__IT__)) == (__IT__)) ? SET : RESET)

#define IS_FMPSMBUS_OWN_ADDRESS1(ADDRESS1)                         ((ADDRESS1) <= 0x000003FFU)
#define IS_FMPSMBUS_OWN_ADDRESS2(ADDRESS2)                         ((ADDRESS2) <= (uint16_t)0x00FFU)

/**
  * @}
  */

/* Include FMPSMBUS HAL Extended module */
#include "stm32f4xx_hal_fmpsmbus_ex.h"

/* Exported functions --------------------------------------------------------*/
/** @addtogroup FMPSMBUS_Exported_Functions FMPSMBUS Exported Functions
  * @{
  */

/** @addtogroup FMPSMBUS_Exported_Functions_Group1 Initialization and de-initialization functions
  * @{
  */

/* Initialization and de-initialization functions  ****************************/
HAL_StatusTypeDef HAL_FMPSMBUS_Init(FMPSMBUS_HandleTypeDef *hfmpsmbus);
HAL_StatusTypeDef HAL_FMPSMBUS_DeInit(FMPSMBUS_HandleTypeDef *hfmpsmbus);
void HAL_FMPSMBUS_MspInit(FMPSMBUS_HandleTypeDef *hfmpsmbus);
void HAL_FMPSMBUS_MspDeInit(FMPSMBUS_HandleTypeDef *hfmpsmbus);
HAL_StatusTypeDef HAL_FMPSMBUS_ConfigAnalogFilter(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t AnalogFilter);
HAL_StatusTypeDef HAL_FMPSMBUS_ConfigDigitalFilter(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t DigitalFilter);

/* Callbacks Register/UnRegister functions  ***********************************/
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
HAL_StatusTypeDef HAL_FMPSMBUS_RegisterCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus,
                                                HAL_FMPSMBUS_CallbackIDTypeDef CallbackID,
                                                pFMPSMBUS_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_FMPSMBUS_UnRegisterCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus,
                                                  HAL_FMPSMBUS_CallbackIDTypeDef CallbackID);

HAL_StatusTypeDef HAL_FMPSMBUS_RegisterAddrCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus,
                                                    pFMPSMBUS_AddrCallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_FMPSMBUS_UnRegisterAddrCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
/**
  * @}
  */

/** @addtogroup FMPSMBUS_Exported_Functions_Group2 Input and Output operation functions
  * @{
  */

/* IO operation functions  *****************************************************/
/** @addtogroup Blocking_mode_Polling Blocking mode Polling
  * @{
  */
/******* Blocking mode: Polling */
HAL_StatusTypeDef HAL_FMPSMBUS_IsDeviceReady(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint16_t DevAddress, uint32_t Trials,
                                          uint32_t Timeout);
/**
  * @}
  */

/** @addtogroup Non-Blocking_mode_Interrupt Non-Blocking mode Interrupt
  * @{
  */
/******* Non-Blocking mode: Interrupt */
HAL_StatusTypeDef HAL_FMPSMBUS_Master_Transmit_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint16_t DevAddress,
                                                  uint8_t *pData, uint16_t Size, uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPSMBUS_Master_Receive_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint16_t DevAddress,
                                                 uint8_t *pData, uint16_t Size, uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPSMBUS_Master_Abort_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint16_t DevAddress);
HAL_StatusTypeDef HAL_FMPSMBUS_Slave_Transmit_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint8_t *pData, uint16_t Size,
                                              uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPSMBUS_Slave_Receive_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint8_t *pData, uint16_t Size,
                                             uint32_t XferOptions);

HAL_StatusTypeDef HAL_FMPSMBUS_EnableAlert_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus);
HAL_StatusTypeDef HAL_FMPSMBUS_DisableAlert_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus);
HAL_StatusTypeDef HAL_FMPSMBUS_EnableListen_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus);
HAL_StatusTypeDef HAL_FMPSMBUS_DisableListen_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus);
/**
  * @}
  */

/** @addtogroup FMPSMBUS_IRQ_Handler_and_Callbacks IRQ Handler and Callbacks
  * @{
  */
/******* FMPSMBUS IRQHandler and Callbacks used in non blocking modes (Interrupt) */
void HAL_FMPSMBUS_EV_IRQHandler(FMPSMBUS_HandleTypeDef *hfmpsmbus);
void HAL_FMPSMBUS_ER_IRQHandler(FMPSMBUS_HandleTypeDef *hfmpsmbus);
void HAL_FMPSMBUS_MasterTxCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus);
void HAL_FMPSMBUS_MasterRxCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus);
void HAL_FMPSMBUS_SlaveTxCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus);
void HAL_FMPSMBUS_SlaveRxCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus);
void HAL_FMPSMBUS_AddrCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint8_t TransferDirection, uint16_t AddrMatchCode);
void HAL_FMPSMBUS_ListenCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus);
void HAL_FMPSMBUS_ErrorCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus);

/**
  * @}
  */

/** @addtogroup FMPSMBUS_Exported_Functions_Group3 Peripheral State and Errors functions
  *  @{
  */

/* Peripheral State and Errors functions  **************************************************/
uint32_t HAL_FMPSMBUS_GetState(FMPSMBUS_HandleTypeDef *hfmpsmbus);
uint32_t HAL_FMPSMBUS_GetError(FMPSMBUS_HandleTypeDef *hfmpsmbus);

/**
  * @}
  */

/**
  * @}
  */

/* Private Functions ---------------------------------------------------------*/
/** @defgroup FMPSMBUS_Private_Functions FMPSMBUS Private Functions
  * @{
  */
/* Private functions are defined in stm32f4xx_hal_fmpsmbus.c file */
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

#endif /* FMPI2C_CR1_PE */
#ifdef __cplusplus
}
#endif


#endif /* STM32F4xx_HAL_FMPSMBUS_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
