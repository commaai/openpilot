/**
  ******************************************************************************
  * @file    stm32f4xx_hal_fmpi2c.h
  * @author  MCD Application Team
  * @brief   Header file of FMPI2C HAL module.
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
#ifndef STM32F4xx_HAL_FMPI2C_H
#define STM32F4xx_HAL_FMPI2C_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(FMPI2C_CR1_PE)
/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup FMPI2C
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup FMPI2C_Exported_Types FMPI2C Exported Types
  * @{
  */

/** @defgroup FMPI2C_Configuration_Structure_definition FMPI2C Configuration Structure definition
  * @brief  FMPI2C Configuration Structure definition
  * @{
  */
typedef struct
{
  uint32_t Timing;              /*!< Specifies the FMPI2C_TIMINGR_register value.
                                     This parameter calculated by referring to FMPI2C initialization section
                                     in Reference manual */

  uint32_t OwnAddress1;         /*!< Specifies the first device own address.
                                     This parameter can be a 7-bit or 10-bit address. */

  uint32_t AddressingMode;      /*!< Specifies if 7-bit or 10-bit addressing mode is selected.
                                     This parameter can be a value of @ref FMPI2C_ADDRESSING_MODE */

  uint32_t DualAddressMode;     /*!< Specifies if dual addressing mode is selected.
                                     This parameter can be a value of @ref FMPI2C_DUAL_ADDRESSING_MODE */

  uint32_t OwnAddress2;         /*!< Specifies the second device own address if dual addressing mode is selected
                                     This parameter can be a 7-bit address. */

  uint32_t OwnAddress2Masks;    /*!< Specifies the acknowledge mask address second device own address if dual addressing
                                     mode is selected.
                                     This parameter can be a value of @ref FMPI2C_OWN_ADDRESS2_MASKS */

  uint32_t GeneralCallMode;     /*!< Specifies if general call mode is selected.
                                     This parameter can be a value of @ref FMPI2C_GENERAL_CALL_ADDRESSING_MODE */

  uint32_t NoStretchMode;       /*!< Specifies if nostretch mode is selected.
                                     This parameter can be a value of @ref FMPI2C_NOSTRETCH_MODE */

} FMPI2C_InitTypeDef;

/**
  * @}
  */

/** @defgroup HAL_state_structure_definition HAL state structure definition
  * @brief  HAL State structure definition
  * @note  HAL FMPI2C State value coding follow below described bitmap :\n
  *          b7-b6  Error information\n
  *             00 : No Error\n
  *             01 : Abort (Abort user request on going)\n
  *             10 : Timeout\n
  *             11 : Error\n
  *          b5     Peripheral initialization status\n
  *             0  : Reset (peripheral not initialized)\n
  *             1  : Init done (peripheral initialized and ready to use. HAL FMPI2C Init function called)\n
  *          b4     (not used)\n
  *             x  : Should be set to 0\n
  *          b3\n
  *             0  : Ready or Busy (No Listen mode ongoing)\n
  *             1  : Listen (peripheral in Address Listen Mode)\n
  *          b2     Intrinsic process state\n
  *             0  : Ready\n
  *             1  : Busy (peripheral busy with some configuration or internal operations)\n
  *          b1     Rx state\n
  *             0  : Ready (no Rx operation ongoing)\n
  *             1  : Busy (Rx operation ongoing)\n
  *          b0     Tx state\n
  *             0  : Ready (no Tx operation ongoing)\n
  *             1  : Busy (Tx operation ongoing)
  * @{
  */
typedef enum
{
  HAL_FMPI2C_STATE_RESET             = 0x00U,   /*!< Peripheral is not yet Initialized         */
  HAL_FMPI2C_STATE_READY             = 0x20U,   /*!< Peripheral Initialized and ready for use  */
  HAL_FMPI2C_STATE_BUSY              = 0x24U,   /*!< An internal process is ongoing            */
  HAL_FMPI2C_STATE_BUSY_TX           = 0x21U,   /*!< Data Transmission process is ongoing      */
  HAL_FMPI2C_STATE_BUSY_RX           = 0x22U,   /*!< Data Reception process is ongoing         */
  HAL_FMPI2C_STATE_LISTEN            = 0x28U,   /*!< Address Listen Mode is ongoing            */
  HAL_FMPI2C_STATE_BUSY_TX_LISTEN    = 0x29U,   /*!< Address Listen Mode and Data Transmission
                                                 process is ongoing                         */
  HAL_FMPI2C_STATE_BUSY_RX_LISTEN    = 0x2AU,   /*!< Address Listen Mode and Data Reception
                                                 process is ongoing                         */
  HAL_FMPI2C_STATE_ABORT             = 0x60U,   /*!< Abort user request ongoing                */
  HAL_FMPI2C_STATE_TIMEOUT           = 0xA0U,   /*!< Timeout state                             */
  HAL_FMPI2C_STATE_ERROR             = 0xE0U    /*!< Error                                     */

} HAL_FMPI2C_StateTypeDef;

/**
  * @}
  */

/** @defgroup HAL_mode_structure_definition HAL mode structure definition
  * @brief  HAL Mode structure definition
  * @note  HAL FMPI2C Mode value coding follow below described bitmap :\n
  *          b7     (not used)\n
  *             x  : Should be set to 0\n
  *          b6\n
  *             0  : None\n
  *             1  : Memory (HAL FMPI2C communication is in Memory Mode)\n
  *          b5\n
  *             0  : None\n
  *             1  : Slave (HAL FMPI2C communication is in Slave Mode)\n
  *          b4\n
  *             0  : None\n
  *             1  : Master (HAL FMPI2C communication is in Master Mode)\n
  *          b3-b2-b1-b0  (not used)\n
  *             xxxx : Should be set to 0000
  * @{
  */
typedef enum
{
  HAL_FMPI2C_MODE_NONE               = 0x00U,   /*!< No FMPI2C communication on going             */
  HAL_FMPI2C_MODE_MASTER             = 0x10U,   /*!< FMPI2C communication is in Master Mode       */
  HAL_FMPI2C_MODE_SLAVE              = 0x20U,   /*!< FMPI2C communication is in Slave Mode        */
  HAL_FMPI2C_MODE_MEM                = 0x40U    /*!< FMPI2C communication is in Memory Mode       */

} HAL_FMPI2C_ModeTypeDef;

/**
  * @}
  */

/** @defgroup FMPI2C_Error_Code_definition FMPI2C Error Code definition
  * @brief  FMPI2C Error Code definition
  * @{
  */
#define HAL_FMPI2C_ERROR_NONE      (0x00000000U)    /*!< No error              */
#define HAL_FMPI2C_ERROR_BERR      (0x00000001U)    /*!< BERR error            */
#define HAL_FMPI2C_ERROR_ARLO      (0x00000002U)    /*!< ARLO error            */
#define HAL_FMPI2C_ERROR_AF        (0x00000004U)    /*!< ACKF error            */
#define HAL_FMPI2C_ERROR_OVR       (0x00000008U)    /*!< OVR error             */
#define HAL_FMPI2C_ERROR_DMA       (0x00000010U)    /*!< DMA transfer error    */
#define HAL_FMPI2C_ERROR_TIMEOUT   (0x00000020U)    /*!< Timeout error         */
#define HAL_FMPI2C_ERROR_SIZE      (0x00000040U)    /*!< Size Management error */
#define HAL_FMPI2C_ERROR_DMA_PARAM (0x00000080U)    /*!< DMA Parameter Error   */
#if (USE_HAL_FMPI2C_REGISTER_CALLBACKS == 1)
#define HAL_FMPI2C_ERROR_INVALID_CALLBACK  (0x00000100U)    /*!< Invalid Callback error */
#endif /* USE_HAL_FMPI2C_REGISTER_CALLBACKS */
#define HAL_FMPI2C_ERROR_INVALID_PARAM     (0x00000200U)    /*!< Invalid Parameters error  */
/**
  * @}
  */

/** @defgroup FMPI2C_handle_Structure_definition FMPI2C handle Structure definition
  * @brief  FMPI2C handle Structure definition
  * @{
  */
typedef struct __FMPI2C_HandleTypeDef
{
  FMPI2C_TypeDef                *Instance;      /*!< FMPI2C registers base address                */

  FMPI2C_InitTypeDef            Init;           /*!< FMPI2C communication parameters              */

  uint8_t                    *pBuffPtr;      /*!< Pointer to FMPI2C transfer buffer            */

  uint16_t                   XferSize;       /*!< FMPI2C transfer size                         */

  __IO uint16_t              XferCount;      /*!< FMPI2C transfer counter                      */

  __IO uint32_t              XferOptions;    /*!< FMPI2C sequantial transfer options, this parameter can
                                                  be a value of @ref FMPI2C_XFEROPTIONS */

  __IO uint32_t              PreviousState;  /*!< FMPI2C communication Previous state          */

  HAL_StatusTypeDef(*XferISR)(struct __FMPI2C_HandleTypeDef *hfmpi2c, uint32_t ITFlags, uint32_t ITSources);
  /*!< FMPI2C transfer IRQ handler function pointer */

  DMA_HandleTypeDef          *hdmatx;        /*!< FMPI2C Tx DMA handle parameters              */

  DMA_HandleTypeDef          *hdmarx;        /*!< FMPI2C Rx DMA handle parameters              */

  HAL_LockTypeDef            Lock;           /*!< FMPI2C locking object                        */

  __IO HAL_FMPI2C_StateTypeDef  State;          /*!< FMPI2C communication state                   */

  __IO HAL_FMPI2C_ModeTypeDef   Mode;           /*!< FMPI2C communication mode                    */

  __IO uint32_t              ErrorCode;      /*!< FMPI2C Error code                            */

  __IO uint32_t              AddrEventCount; /*!< FMPI2C Address Event counter                 */

#if (USE_HAL_FMPI2C_REGISTER_CALLBACKS == 1)
  void (* MasterTxCpltCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Master Tx Transfer completed callback */
  void (* MasterRxCpltCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Master Rx Transfer completed callback */
  void (* SlaveTxCpltCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Slave Tx Transfer completed callback  */
  void (* SlaveRxCpltCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Slave Rx Transfer completed callback  */
  void (* ListenCpltCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Listen Complete callback              */
  void (* MemTxCpltCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Memory Tx Transfer completed callback */
  void (* MemRxCpltCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Memory Rx Transfer completed callback */
  void (* ErrorCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Error callback                        */
  void (* AbortCpltCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Abort callback                        */

  void (* AddrCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c, uint8_t TransferDirection, uint16_t AddrMatchCode);
  /*!< FMPI2C Slave Address Match callback */

  void (* MspInitCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Msp Init callback                     */
  void (* MspDeInitCallback)(struct __FMPI2C_HandleTypeDef *hfmpi2c);
  /*!< FMPI2C Msp DeInit callback                   */

#endif  /* USE_HAL_FMPI2C_REGISTER_CALLBACKS */
} FMPI2C_HandleTypeDef;

#if (USE_HAL_FMPI2C_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL FMPI2C Callback ID enumeration definition
  */
typedef enum
{
  HAL_FMPI2C_MASTER_TX_COMPLETE_CB_ID      = 0x00U,    /*!< FMPI2C Master Tx Transfer completed callback ID  */
  HAL_FMPI2C_MASTER_RX_COMPLETE_CB_ID      = 0x01U,    /*!< FMPI2C Master Rx Transfer completed callback ID  */
  HAL_FMPI2C_SLAVE_TX_COMPLETE_CB_ID       = 0x02U,    /*!< FMPI2C Slave Tx Transfer completed callback ID   */
  HAL_FMPI2C_SLAVE_RX_COMPLETE_CB_ID       = 0x03U,    /*!< FMPI2C Slave Rx Transfer completed callback ID   */
  HAL_FMPI2C_LISTEN_COMPLETE_CB_ID         = 0x04U,    /*!< FMPI2C Listen Complete callback ID               */
  HAL_FMPI2C_MEM_TX_COMPLETE_CB_ID         = 0x05U,    /*!< FMPI2C Memory Tx Transfer callback ID            */
  HAL_FMPI2C_MEM_RX_COMPLETE_CB_ID         = 0x06U,    /*!< FMPI2C Memory Rx Transfer completed callback ID  */
  HAL_FMPI2C_ERROR_CB_ID                   = 0x07U,    /*!< FMPI2C Error callback ID                         */
  HAL_FMPI2C_ABORT_CB_ID                   = 0x08U,    /*!< FMPI2C Abort callback ID                         */

  HAL_FMPI2C_MSPINIT_CB_ID                 = 0x09U,    /*!< FMPI2C Msp Init callback ID                      */
  HAL_FMPI2C_MSPDEINIT_CB_ID               = 0x0AU     /*!< FMPI2C Msp DeInit callback ID                    */

} HAL_FMPI2C_CallbackIDTypeDef;

/**
  * @brief  HAL FMPI2C Callback pointer definition
  */
typedef  void (*pFMPI2C_CallbackTypeDef)(FMPI2C_HandleTypeDef *hfmpi2c);
/*!< pointer to an FMPI2C callback function */
typedef  void (*pFMPI2C_AddrCallbackTypeDef)(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t TransferDirection,
                                             uint16_t AddrMatchCode);
/*!< pointer to an FMPI2C Address Match callback function */

#endif /* USE_HAL_FMPI2C_REGISTER_CALLBACKS */
/**
  * @}
  */

/**
  * @}
  */
/* Exported constants --------------------------------------------------------*/

/** @defgroup FMPI2C_Exported_Constants FMPI2C Exported Constants
  * @{
  */

/** @defgroup FMPI2C_XFEROPTIONS  FMPI2C Sequential Transfer Options
  * @{
  */
#define FMPI2C_FIRST_FRAME                 ((uint32_t)FMPI2C_SOFTEND_MODE)
#define FMPI2C_FIRST_AND_NEXT_FRAME        ((uint32_t)(FMPI2C_RELOAD_MODE | FMPI2C_SOFTEND_MODE))
#define FMPI2C_NEXT_FRAME                  ((uint32_t)(FMPI2C_RELOAD_MODE | FMPI2C_SOFTEND_MODE))
#define FMPI2C_FIRST_AND_LAST_FRAME        ((uint32_t)FMPI2C_AUTOEND_MODE)
#define FMPI2C_LAST_FRAME                  ((uint32_t)FMPI2C_AUTOEND_MODE)
#define FMPI2C_LAST_FRAME_NO_STOP          ((uint32_t)FMPI2C_SOFTEND_MODE)

/* List of XferOptions in usage of :
 * 1- Restart condition in all use cases (direction change or not)
 */
#define  FMPI2C_OTHER_FRAME                (0x000000AAU)
#define  FMPI2C_OTHER_AND_LAST_FRAME       (0x0000AA00U)
/**
  * @}
  */

/** @defgroup FMPI2C_ADDRESSING_MODE FMPI2C Addressing Mode
  * @{
  */
#define FMPI2C_ADDRESSINGMODE_7BIT         (0x00000001U)
#define FMPI2C_ADDRESSINGMODE_10BIT        (0x00000002U)
/**
  * @}
  */

/** @defgroup FMPI2C_DUAL_ADDRESSING_MODE FMPI2C Dual Addressing Mode
  * @{
  */
#define FMPI2C_DUALADDRESS_DISABLE         (0x00000000U)
#define FMPI2C_DUALADDRESS_ENABLE          FMPI2C_OAR2_OA2EN
/**
  * @}
  */

/** @defgroup FMPI2C_OWN_ADDRESS2_MASKS FMPI2C Own Address2 Masks
  * @{
  */
#define FMPI2C_OA2_NOMASK                  ((uint8_t)0x00U)
#define FMPI2C_OA2_MASK01                  ((uint8_t)0x01U)
#define FMPI2C_OA2_MASK02                  ((uint8_t)0x02U)
#define FMPI2C_OA2_MASK03                  ((uint8_t)0x03U)
#define FMPI2C_OA2_MASK04                  ((uint8_t)0x04U)
#define FMPI2C_OA2_MASK05                  ((uint8_t)0x05U)
#define FMPI2C_OA2_MASK06                  ((uint8_t)0x06U)
#define FMPI2C_OA2_MASK07                  ((uint8_t)0x07U)
/**
  * @}
  */

/** @defgroup FMPI2C_GENERAL_CALL_ADDRESSING_MODE FMPI2C General Call Addressing Mode
  * @{
  */
#define FMPI2C_GENERALCALL_DISABLE         (0x00000000U)
#define FMPI2C_GENERALCALL_ENABLE          FMPI2C_CR1_GCEN
/**
  * @}
  */

/** @defgroup FMPI2C_NOSTRETCH_MODE FMPI2C No-Stretch Mode
  * @{
  */
#define FMPI2C_NOSTRETCH_DISABLE           (0x00000000U)
#define FMPI2C_NOSTRETCH_ENABLE            FMPI2C_CR1_NOSTRETCH
/**
  * @}
  */

/** @defgroup FMPI2C_MEMORY_ADDRESS_SIZE FMPI2C Memory Address Size
  * @{
  */
#define FMPI2C_MEMADD_SIZE_8BIT            (0x00000001U)
#define FMPI2C_MEMADD_SIZE_16BIT           (0x00000002U)
/**
  * @}
  */

/** @defgroup FMPI2C_XFERDIRECTION FMPI2C Transfer Direction Master Point of View
  * @{
  */
#define FMPI2C_DIRECTION_TRANSMIT          (0x00000000U)
#define FMPI2C_DIRECTION_RECEIVE           (0x00000001U)
/**
  * @}
  */

/** @defgroup FMPI2C_RELOAD_END_MODE FMPI2C Reload End Mode
  * @{
  */
#define  FMPI2C_RELOAD_MODE                FMPI2C_CR2_RELOAD
#define  FMPI2C_AUTOEND_MODE               FMPI2C_CR2_AUTOEND
#define  FMPI2C_SOFTEND_MODE               (0x00000000U)
/**
  * @}
  */

/** @defgroup FMPI2C_START_STOP_MODE FMPI2C Start or Stop Mode
  * @{
  */
#define  FMPI2C_NO_STARTSTOP               (0x00000000U)
#define  FMPI2C_GENERATE_STOP              (uint32_t)(0x80000000U | FMPI2C_CR2_STOP)
#define  FMPI2C_GENERATE_START_READ        (uint32_t)(0x80000000U | FMPI2C_CR2_START | FMPI2C_CR2_RD_WRN)
#define  FMPI2C_GENERATE_START_WRITE       (uint32_t)(0x80000000U | FMPI2C_CR2_START)
/**
  * @}
  */

/** @defgroup FMPI2C_Interrupt_configuration_definition FMPI2C Interrupt configuration definition
  * @brief FMPI2C Interrupt definition
  *        Elements values convention: 0xXXXXXXXX
  *           - XXXXXXXX  : Interrupt control mask
  * @{
  */
#define FMPI2C_IT_ERRI                     FMPI2C_CR1_ERRIE
#define FMPI2C_IT_TCI                      FMPI2C_CR1_TCIE
#define FMPI2C_IT_STOPI                    FMPI2C_CR1_STOPIE
#define FMPI2C_IT_NACKI                    FMPI2C_CR1_NACKIE
#define FMPI2C_IT_ADDRI                    FMPI2C_CR1_ADDRIE
#define FMPI2C_IT_RXI                      FMPI2C_CR1_RXIE
#define FMPI2C_IT_TXI                      FMPI2C_CR1_TXIE
/**
  * @}
  */

/** @defgroup FMPI2C_Flag_definition FMPI2C Flag definition
  * @{
  */
#define FMPI2C_FLAG_TXE                    FMPI2C_ISR_TXE
#define FMPI2C_FLAG_TXIS                   FMPI2C_ISR_TXIS
#define FMPI2C_FLAG_RXNE                   FMPI2C_ISR_RXNE
#define FMPI2C_FLAG_ADDR                   FMPI2C_ISR_ADDR
#define FMPI2C_FLAG_AF                     FMPI2C_ISR_NACKF
#define FMPI2C_FLAG_STOPF                  FMPI2C_ISR_STOPF
#define FMPI2C_FLAG_TC                     FMPI2C_ISR_TC
#define FMPI2C_FLAG_TCR                    FMPI2C_ISR_TCR
#define FMPI2C_FLAG_BERR                   FMPI2C_ISR_BERR
#define FMPI2C_FLAG_ARLO                   FMPI2C_ISR_ARLO
#define FMPI2C_FLAG_OVR                    FMPI2C_ISR_OVR
#define FMPI2C_FLAG_PECERR                 FMPI2C_ISR_PECERR
#define FMPI2C_FLAG_TIMEOUT                FMPI2C_ISR_TIMEOUT
#define FMPI2C_FLAG_ALERT                  FMPI2C_ISR_ALERT
#define FMPI2C_FLAG_BUSY                   FMPI2C_ISR_BUSY
#define FMPI2C_FLAG_DIR                    FMPI2C_ISR_DIR
/**
  * @}
  */

/**
  * @}
  */

/* Exported macros -----------------------------------------------------------*/

/** @defgroup FMPI2C_Exported_Macros FMPI2C Exported Macros
  * @{
  */

/** @brief Reset FMPI2C handle state.
  * @param  __HANDLE__ specifies the FMPI2C Handle.
  * @retval None
  */
#if (USE_HAL_FMPI2C_REGISTER_CALLBACKS == 1)
#define __HAL_FMPI2C_RESET_HANDLE_STATE(__HANDLE__)                do{                                             \
                                                                    (__HANDLE__)->State = HAL_FMPI2C_STATE_RESET;  \
                                                                    (__HANDLE__)->MspInitCallback = NULL;       \
                                                                    (__HANDLE__)->MspDeInitCallback = NULL;     \
                                                                  } while(0)
#else
#define __HAL_FMPI2C_RESET_HANDLE_STATE(__HANDLE__)                ((__HANDLE__)->State = HAL_FMPI2C_STATE_RESET)
#endif /* USE_HAL_FMPI2C_REGISTER_CALLBACKS */

/** @brief  Enable the specified FMPI2C interrupt.
  * @param  __HANDLE__ specifies the FMPI2C Handle.
  * @param  __INTERRUPT__ specifies the interrupt source to enable.
  *        This parameter can be one of the following values:
  *            @arg @ref FMPI2C_IT_ERRI  Errors interrupt enable
  *            @arg @ref FMPI2C_IT_TCI   Transfer complete interrupt enable
  *            @arg @ref FMPI2C_IT_STOPI STOP detection interrupt enable
  *            @arg @ref FMPI2C_IT_NACKI NACK received interrupt enable
  *            @arg @ref FMPI2C_IT_ADDRI Address match interrupt enable
  *            @arg @ref FMPI2C_IT_RXI   RX interrupt enable
  *            @arg @ref FMPI2C_IT_TXI   TX interrupt enable
  *
  * @retval None
  */
#define __HAL_FMPI2C_ENABLE_IT(__HANDLE__, __INTERRUPT__)          ((__HANDLE__)->Instance->CR1 |= (__INTERRUPT__))

/** @brief  Disable the specified FMPI2C interrupt.
  * @param  __HANDLE__ specifies the FMPI2C Handle.
  * @param  __INTERRUPT__ specifies the interrupt source to disable.
  *        This parameter can be one of the following values:
  *            @arg @ref FMPI2C_IT_ERRI  Errors interrupt enable
  *            @arg @ref FMPI2C_IT_TCI   Transfer complete interrupt enable
  *            @arg @ref FMPI2C_IT_STOPI STOP detection interrupt enable
  *            @arg @ref FMPI2C_IT_NACKI NACK received interrupt enable
  *            @arg @ref FMPI2C_IT_ADDRI Address match interrupt enable
  *            @arg @ref FMPI2C_IT_RXI   RX interrupt enable
  *            @arg @ref FMPI2C_IT_TXI   TX interrupt enable
  *
  * @retval None
  */
#define __HAL_FMPI2C_DISABLE_IT(__HANDLE__, __INTERRUPT__)         ((__HANDLE__)->Instance->CR1 &= (~(__INTERRUPT__)))

/** @brief  Check whether the specified FMPI2C interrupt source is enabled or not.
  * @param  __HANDLE__ specifies the FMPI2C Handle.
  * @param  __INTERRUPT__ specifies the FMPI2C interrupt source to check.
  *          This parameter can be one of the following values:
  *            @arg @ref FMPI2C_IT_ERRI  Errors interrupt enable
  *            @arg @ref FMPI2C_IT_TCI   Transfer complete interrupt enable
  *            @arg @ref FMPI2C_IT_STOPI STOP detection interrupt enable
  *            @arg @ref FMPI2C_IT_NACKI NACK received interrupt enable
  *            @arg @ref FMPI2C_IT_ADDRI Address match interrupt enable
  *            @arg @ref FMPI2C_IT_RXI   RX interrupt enable
  *            @arg @ref FMPI2C_IT_TXI   TX interrupt enable
  *
  * @retval The new state of __INTERRUPT__ (SET or RESET).
  */
#define __HAL_FMPI2C_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__)      ((((__HANDLE__)->Instance->CR1 & \
                                                                   (__INTERRUPT__)) == (__INTERRUPT__)) ? SET : RESET)

/** @brief  Check whether the specified FMPI2C flag is set or not.
  * @param  __HANDLE__ specifies the FMPI2C Handle.
  * @param  __FLAG__ specifies the flag to check.
  *        This parameter can be one of the following values:
  *            @arg @ref FMPI2C_FLAG_TXE     Transmit data register empty
  *            @arg @ref FMPI2C_FLAG_TXIS    Transmit interrupt status
  *            @arg @ref FMPI2C_FLAG_RXNE    Receive data register not empty
  *            @arg @ref FMPI2C_FLAG_ADDR    Address matched (slave mode)
  *            @arg @ref FMPI2C_FLAG_AF      Acknowledge failure received flag
  *            @arg @ref FMPI2C_FLAG_STOPF   STOP detection flag
  *            @arg @ref FMPI2C_FLAG_TC      Transfer complete (master mode)
  *            @arg @ref FMPI2C_FLAG_TCR     Transfer complete reload
  *            @arg @ref FMPI2C_FLAG_BERR    Bus error
  *            @arg @ref FMPI2C_FLAG_ARLO    Arbitration lost
  *            @arg @ref FMPI2C_FLAG_OVR     Overrun/Underrun
  *            @arg @ref FMPI2C_FLAG_PECERR  PEC error in reception
  *            @arg @ref FMPI2C_FLAG_TIMEOUT Timeout or Tlow detection flag
  *            @arg @ref FMPI2C_FLAG_ALERT   SMBus alert
  *            @arg @ref FMPI2C_FLAG_BUSY    Bus busy
  *            @arg @ref FMPI2C_FLAG_DIR     Transfer direction (slave mode)
  *
  * @retval The new state of __FLAG__ (SET or RESET).
  */
#define FMPI2C_FLAG_MASK  (0x0001FFFFU)
#define __HAL_FMPI2C_GET_FLAG(__HANDLE__, __FLAG__) (((((__HANDLE__)->Instance->ISR) & \
                                                    (__FLAG__)) == (__FLAG__)) ? SET : RESET)

/** @brief  Clear the FMPI2C pending flags which are cleared by writing 1 in a specific bit.
  * @param  __HANDLE__ specifies the FMPI2C Handle.
  * @param  __FLAG__ specifies the flag to clear.
  *          This parameter can be any combination of the following values:
  *            @arg @ref FMPI2C_FLAG_TXE     Transmit data register empty
  *            @arg @ref FMPI2C_FLAG_ADDR    Address matched (slave mode)
  *            @arg @ref FMPI2C_FLAG_AF      Acknowledge failure received flag
  *            @arg @ref FMPI2C_FLAG_STOPF   STOP detection flag
  *            @arg @ref FMPI2C_FLAG_BERR    Bus error
  *            @arg @ref FMPI2C_FLAG_ARLO    Arbitration lost
  *            @arg @ref FMPI2C_FLAG_OVR     Overrun/Underrun
  *            @arg @ref FMPI2C_FLAG_PECERR  PEC error in reception
  *            @arg @ref FMPI2C_FLAG_TIMEOUT Timeout or Tlow detection flag
  *            @arg @ref FMPI2C_FLAG_ALERT   SMBus alert
  *
  * @retval None
  */
#define __HAL_FMPI2C_CLEAR_FLAG(__HANDLE__, __FLAG__) (((__FLAG__) == FMPI2C_FLAG_TXE) ? \
                                                    ((__HANDLE__)->Instance->ISR |= (__FLAG__)) : \
                                                    ((__HANDLE__)->Instance->ICR = (__FLAG__)))

/** @brief  Enable the specified FMPI2C peripheral.
  * @param  __HANDLE__ specifies the FMPI2C Handle.
  * @retval None
  */
#define __HAL_FMPI2C_ENABLE(__HANDLE__)                         (SET_BIT((__HANDLE__)->Instance->CR1, FMPI2C_CR1_PE))

/** @brief  Disable the specified FMPI2C peripheral.
  * @param  __HANDLE__ specifies the FMPI2C Handle.
  * @retval None
  */
#define __HAL_FMPI2C_DISABLE(__HANDLE__)                        (CLEAR_BIT((__HANDLE__)->Instance->CR1, FMPI2C_CR1_PE))

/** @brief  Generate a Non-Acknowledge FMPI2C peripheral in Slave mode.
  * @param  __HANDLE__ specifies the FMPI2C Handle.
  * @retval None
  */
#define __HAL_FMPI2C_GENERATE_NACK(__HANDLE__)                  (SET_BIT((__HANDLE__)->Instance->CR2, FMPI2C_CR2_NACK))
/**
  * @}
  */

/* Include FMPI2C HAL Extended module */
#include "stm32f4xx_hal_fmpi2c_ex.h"

/* Exported functions --------------------------------------------------------*/
/** @addtogroup FMPI2C_Exported_Functions
  * @{
  */

/** @addtogroup FMPI2C_Exported_Functions_Group1 Initialization and de-initialization functions
  * @{
  */
/* Initialization and de-initialization functions******************************/
HAL_StatusTypeDef HAL_FMPI2C_Init(FMPI2C_HandleTypeDef *hfmpi2c);
HAL_StatusTypeDef HAL_FMPI2C_DeInit(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_MspInit(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_MspDeInit(FMPI2C_HandleTypeDef *hfmpi2c);

/* Callbacks Register/UnRegister functions  ***********************************/
#if (USE_HAL_FMPI2C_REGISTER_CALLBACKS == 1)
HAL_StatusTypeDef HAL_FMPI2C_RegisterCallback(FMPI2C_HandleTypeDef *hfmpi2c, HAL_FMPI2C_CallbackIDTypeDef CallbackID,
                                           pFMPI2C_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_FMPI2C_UnRegisterCallback(FMPI2C_HandleTypeDef *hfmpi2c, HAL_FMPI2C_CallbackIDTypeDef CallbackID);

HAL_StatusTypeDef HAL_FMPI2C_RegisterAddrCallback(FMPI2C_HandleTypeDef *hfmpi2c, pFMPI2C_AddrCallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_FMPI2C_UnRegisterAddrCallback(FMPI2C_HandleTypeDef *hfmpi2c);
#endif /* USE_HAL_FMPI2C_REGISTER_CALLBACKS */
/**
  * @}
  */

/** @addtogroup FMPI2C_Exported_Functions_Group2 Input and Output operation functions
  * @{
  */
/* IO operation functions  ****************************************************/
/******* Blocking mode: Polling */
HAL_StatusTypeDef HAL_FMPI2C_Master_Transmit(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                             uint16_t Size, uint32_t Timeout);
HAL_StatusTypeDef HAL_FMPI2C_Master_Receive(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                            uint16_t Size, uint32_t Timeout);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Transmit(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size,
                                            uint32_t Timeout);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Receive(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size,
                                           uint32_t Timeout);
HAL_StatusTypeDef HAL_FMPI2C_Mem_Write(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint16_t MemAddress,
                                    uint16_t MemAddSize, uint8_t *pData, uint16_t Size, uint32_t Timeout);
HAL_StatusTypeDef HAL_FMPI2C_Mem_Read(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint16_t MemAddress,
                                   uint16_t MemAddSize, uint8_t *pData, uint16_t Size, uint32_t Timeout);
HAL_StatusTypeDef HAL_FMPI2C_IsDeviceReady(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint32_t Trials,
                                        uint32_t Timeout);

/******* Non-Blocking mode: Interrupt */
HAL_StatusTypeDef HAL_FMPI2C_Master_Transmit_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                             uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Master_Receive_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                            uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Transmit_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Receive_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Mem_Write_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint16_t MemAddress,
                                       uint16_t MemAddSize, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Mem_Read_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint16_t MemAddress,
                                      uint16_t MemAddSize, uint8_t *pData, uint16_t Size);

HAL_StatusTypeDef HAL_FMPI2C_Master_Seq_Transmit_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                                 uint16_t Size, uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPI2C_Master_Seq_Receive_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                                uint16_t Size, uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Seq_Transmit_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size,
                                                uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Seq_Receive_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size,
                                               uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPI2C_EnableListen_IT(FMPI2C_HandleTypeDef *hfmpi2c);
HAL_StatusTypeDef HAL_FMPI2C_DisableListen_IT(FMPI2C_HandleTypeDef *hfmpi2c);
HAL_StatusTypeDef HAL_FMPI2C_Master_Abort_IT(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress);

/******* Non-Blocking mode: DMA */
HAL_StatusTypeDef HAL_FMPI2C_Master_Transmit_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                              uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Master_Receive_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                             uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Transmit_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Receive_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Mem_Write_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint16_t MemAddress,
                                        uint16_t MemAddSize, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_FMPI2C_Mem_Read_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint16_t MemAddress,
                                       uint16_t MemAddSize, uint8_t *pData, uint16_t Size);

HAL_StatusTypeDef HAL_FMPI2C_Master_Seq_Transmit_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                                  uint16_t Size, uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPI2C_Master_Seq_Receive_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint16_t DevAddress, uint8_t *pData,
                                                 uint16_t Size, uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Seq_Transmit_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size,
                                                 uint32_t XferOptions);
HAL_StatusTypeDef HAL_FMPI2C_Slave_Seq_Receive_DMA(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t *pData, uint16_t Size,
                                                uint32_t XferOptions);
/**
  * @}
  */

/** @addtogroup FMPI2C_IRQ_Handler_and_Callbacks IRQ Handler and Callbacks
  * @{
  */
/******* FMPI2C IRQHandler and Callbacks used in non blocking modes (Interrupt and DMA) */
void HAL_FMPI2C_EV_IRQHandler(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_ER_IRQHandler(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_MasterTxCpltCallback(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_MasterRxCpltCallback(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_SlaveTxCpltCallback(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_SlaveRxCpltCallback(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_AddrCallback(FMPI2C_HandleTypeDef *hfmpi2c, uint8_t TransferDirection, uint16_t AddrMatchCode);
void HAL_FMPI2C_ListenCpltCallback(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_MemTxCpltCallback(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_MemRxCpltCallback(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_ErrorCallback(FMPI2C_HandleTypeDef *hfmpi2c);
void HAL_FMPI2C_AbortCpltCallback(FMPI2C_HandleTypeDef *hfmpi2c);
/**
  * @}
  */

/** @addtogroup FMPI2C_Exported_Functions_Group3 Peripheral State, Mode and Error functions
  * @{
  */
/* Peripheral State, Mode and Error functions  *********************************/
HAL_FMPI2C_StateTypeDef HAL_FMPI2C_GetState(FMPI2C_HandleTypeDef *hfmpi2c);
HAL_FMPI2C_ModeTypeDef  HAL_FMPI2C_GetMode(FMPI2C_HandleTypeDef *hfmpi2c);
uint32_t             HAL_FMPI2C_GetError(FMPI2C_HandleTypeDef *hfmpi2c);

/**
  * @}
  */

/**
  * @}
  */

/* Private constants ---------------------------------------------------------*/
/** @defgroup FMPI2C_Private_Constants FMPI2C Private Constants
  * @{
  */

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup FMPI2C_Private_Macro FMPI2C Private Macros
  * @{
  */

#define IS_FMPI2C_ADDRESSING_MODE(MODE)    (((MODE) == FMPI2C_ADDRESSINGMODE_7BIT) || \
                                         ((MODE) == FMPI2C_ADDRESSINGMODE_10BIT))

#define IS_FMPI2C_DUAL_ADDRESS(ADDRESS)    (((ADDRESS) == FMPI2C_DUALADDRESS_DISABLE) || \
                                         ((ADDRESS) == FMPI2C_DUALADDRESS_ENABLE))

#define IS_FMPI2C_OWN_ADDRESS2_MASK(MASK)  (((MASK) == FMPI2C_OA2_NOMASK)  || \
                                         ((MASK) == FMPI2C_OA2_MASK01) || \
                                         ((MASK) == FMPI2C_OA2_MASK02) || \
                                         ((MASK) == FMPI2C_OA2_MASK03) || \
                                         ((MASK) == FMPI2C_OA2_MASK04) || \
                                         ((MASK) == FMPI2C_OA2_MASK05) || \
                                         ((MASK) == FMPI2C_OA2_MASK06) || \
                                         ((MASK) == FMPI2C_OA2_MASK07))

#define IS_FMPI2C_GENERAL_CALL(CALL)       (((CALL) == FMPI2C_GENERALCALL_DISABLE) || \
                                         ((CALL) == FMPI2C_GENERALCALL_ENABLE))

#define IS_FMPI2C_NO_STRETCH(STRETCH)      (((STRETCH) == FMPI2C_NOSTRETCH_DISABLE) || \
                                         ((STRETCH) == FMPI2C_NOSTRETCH_ENABLE))

#define IS_FMPI2C_MEMADD_SIZE(SIZE)        (((SIZE) == FMPI2C_MEMADD_SIZE_8BIT) || \
                                         ((SIZE) == FMPI2C_MEMADD_SIZE_16BIT))

#define IS_TRANSFER_MODE(MODE)          (((MODE) == FMPI2C_RELOAD_MODE)   || \
                                         ((MODE) == FMPI2C_AUTOEND_MODE) || \
                                         ((MODE) == FMPI2C_SOFTEND_MODE))

#define IS_TRANSFER_REQUEST(REQUEST)    (((REQUEST) == FMPI2C_GENERATE_STOP)        || \
                                         ((REQUEST) == FMPI2C_GENERATE_START_READ)  || \
                                         ((REQUEST) == FMPI2C_GENERATE_START_WRITE) || \
                                         ((REQUEST) == FMPI2C_NO_STARTSTOP))

#define IS_FMPI2C_TRANSFER_OPTIONS_REQUEST(REQUEST)  (((REQUEST) == FMPI2C_FIRST_FRAME)          || \
                                                   ((REQUEST) == FMPI2C_FIRST_AND_NEXT_FRAME) || \
                                                   ((REQUEST) == FMPI2C_NEXT_FRAME)           || \
                                                   ((REQUEST) == FMPI2C_FIRST_AND_LAST_FRAME) || \
                                                   ((REQUEST) == FMPI2C_LAST_FRAME)           || \
                                                   ((REQUEST) == FMPI2C_LAST_FRAME_NO_STOP)   || \
                                                   IS_FMPI2C_TRANSFER_OTHER_OPTIONS_REQUEST(REQUEST))

#define IS_FMPI2C_TRANSFER_OTHER_OPTIONS_REQUEST(REQUEST) (((REQUEST) == FMPI2C_OTHER_FRAME)     || \
                                                        ((REQUEST) == FMPI2C_OTHER_AND_LAST_FRAME))

#define FMPI2C_RESET_CR2(__HANDLE__)                 ((__HANDLE__)->Instance->CR2 &= \
                                                   (uint32_t)~((uint32_t)(FMPI2C_CR2_SADD   | FMPI2C_CR2_HEAD10R | \
                                                                          FMPI2C_CR2_NBYTES | FMPI2C_CR2_RELOAD  | \
                                                                          FMPI2C_CR2_RD_WRN)))

#define FMPI2C_GET_ADDR_MATCH(__HANDLE__)            ((uint16_t)(((__HANDLE__)->Instance->ISR & FMPI2C_ISR_ADDCODE) \
                                                                  >> 16U))
#define FMPI2C_GET_DIR(__HANDLE__)                   ((uint8_t)(((__HANDLE__)->Instance->ISR & FMPI2C_ISR_DIR) \
                                                                  >> 16U))
#define FMPI2C_GET_STOP_MODE(__HANDLE__)             ((__HANDLE__)->Instance->CR2 & FMPI2C_CR2_AUTOEND)
#define FMPI2C_GET_OWN_ADDRESS1(__HANDLE__)          ((uint16_t)((__HANDLE__)->Instance->OAR1 & FMPI2C_OAR1_OA1))
#define FMPI2C_GET_OWN_ADDRESS2(__HANDLE__)          ((uint16_t)((__HANDLE__)->Instance->OAR2 & FMPI2C_OAR2_OA2))

#define IS_FMPI2C_OWN_ADDRESS1(ADDRESS1)             ((ADDRESS1) <= 0x000003FFU)
#define IS_FMPI2C_OWN_ADDRESS2(ADDRESS2)             ((ADDRESS2) <= (uint16_t)0x00FFU)

#define FMPI2C_MEM_ADD_MSB(__ADDRESS__)              ((uint8_t)((uint16_t)(((uint16_t)((__ADDRESS__) & \
                                                                         (uint16_t)(0xFF00U))) >> 8U)))
#define FMPI2C_MEM_ADD_LSB(__ADDRESS__)              ((uint8_t)((uint16_t)((__ADDRESS__) & (uint16_t)(0x00FFU))))

#define FMPI2C_GENERATE_START(__ADDMODE__,__ADDRESS__) (((__ADDMODE__) == FMPI2C_ADDRESSINGMODE_7BIT) ? \
                                                     (uint32_t)((((uint32_t)(__ADDRESS__) & (FMPI2C_CR2_SADD)) | \
                                                                 (FMPI2C_CR2_START) | (FMPI2C_CR2_AUTOEND)) & \
                                                                (~FMPI2C_CR2_RD_WRN)) : \
                                                     (uint32_t)((((uint32_t)(__ADDRESS__) & (FMPI2C_CR2_SADD)) | \
                                                                 (FMPI2C_CR2_ADD10) | (FMPI2C_CR2_START)) & \
                                                                (~FMPI2C_CR2_RD_WRN)))

#define FMPI2C_CHECK_FLAG(__ISR__, __FLAG__)         ((((__ISR__) & ((__FLAG__) & FMPI2C_FLAG_MASK)) == \
                                                    ((__FLAG__) & FMPI2C_FLAG_MASK)) ? SET : RESET)
#define FMPI2C_CHECK_IT_SOURCE(__CR1__, __IT__)      ((((__CR1__) & (__IT__)) == (__IT__)) ? SET : RESET)
/**
  * @}
  */

/* Private Functions ---------------------------------------------------------*/
/** @defgroup FMPI2C_Private_Functions FMPI2C Private Functions
  * @{
  */
/* Private functions are defined in stm32f4xx_hal_fmpi2c.c file */
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


#endif /* STM32F4xx_HAL_FMPI2C_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
