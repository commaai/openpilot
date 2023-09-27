/**
  ******************************************************************************
  * @file    stm32f4xx_hal_smbus.h
  * @author  MCD Application Team
  * @brief   Header file of SMBUS HAL module.
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
#ifndef __STM32F4xx_HAL_SMBUS_H
#define __STM32F4xx_HAL_SMBUS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup SMBUS
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup SMBUS_Exported_Types SMBUS Exported Types
  * @{
  */

/**
  * @brief  SMBUS Configuration Structure definition
  */
typedef struct
{
  uint32_t ClockSpeed;           /*!< Specifies the clock frequency.
                                    This parameter must be set to a value lower than 100kHz                  */

  uint32_t AnalogFilter;         /*!< Specifies if Analog Filter is enable or not.
                                  This parameter can be a value of @ref SMBUS_Analog_Filter                  */

  uint32_t OwnAddress1;          /*!< Specifies the first device own address.
                                    This parameter can be a 7-bit or 10-bit address.                         */

  uint32_t AddressingMode;       /*!< Specifies if 7-bit or 10-bit addressing mode is selected.
                                    This parameter can be a value of @ref SMBUS_addressing_mode              */

  uint32_t DualAddressMode;      /*!< Specifies if dual addressing mode is selected.
                                    This parameter can be a value of @ref SMBUS_dual_addressing_mode         */

  uint32_t OwnAddress2;          /*!< Specifies the second device own address if dual addressing mode is
                                     selected. This parameter can be a 7-bit address.                        */

  uint32_t GeneralCallMode;      /*!< Specifies if general call mode is selected.
                                    This parameter can be a value of @ref SMBUS_general_call_addressing_mode */

  uint32_t NoStretchMode;        /*!< Specifies if nostretch mode is selected.
                                    This parameter can be a value of @ref SMBUS_nostretch_mode               */

  uint32_t PacketErrorCheckMode; /*!< Specifies if Packet Error Check mode is selected.
                                     This parameter can be a value of @ref SMBUS_packet_error_check_mode     */

  uint32_t PeripheralMode;       /*!< Specifies which mode of Periphal is selected.
                                     This parameter can be a value of @ref SMBUS_peripheral_mode             */

} SMBUS_InitTypeDef;

/**
  * @brief  HAL State structure definition
  * @note  HAL SMBUS State value coding follow below described bitmap :
  *          b7-b6  Error information
  *             00 : No Error
  *             01 : Abort (Abort user request on going)
  *             10 : Timeout
  *             11 : Error
  *          b5     IP initialisation status
  *             0  : Reset (IP not initialized)
  *             1  : Init done (IP initialized and ready to use. HAL SMBUS Init function called)
  *          b4     (not used)
  *             x  : Should be set to 0
  *          b3
  *             0  : Ready or Busy (No Listen mode ongoing)
  *             1  : Listen (IP in Address Listen Mode)
  *          b2     Intrinsic process state
  *             0  : Ready
  *             1  : Busy (IP busy with some configuration or internal operations)
  *          b1     Rx state
  *             0  : Ready (no Rx operation ongoing)
  *             1  : Busy (Rx operation ongoing)
  *          b0     Tx state
  *             0  : Ready (no Tx operation ongoing)
  *             1  : Busy (Tx operation ongoing)
  */
typedef enum
{

  HAL_SMBUS_STATE_RESET             = 0x00U,   /*!< Peripheral is not yet Initialized         */
  HAL_SMBUS_STATE_READY             = 0x20U,   /*!< Peripheral Initialized and ready for use  */
  HAL_SMBUS_STATE_BUSY              = 0x24U,   /*!< An internal process is ongoing            */
  HAL_SMBUS_STATE_BUSY_TX           = 0x21U,   /*!< Data Transmission process is ongoing      */
  HAL_SMBUS_STATE_BUSY_RX           = 0x22U,   /*!< Data Reception process is ongoing         */
  HAL_SMBUS_STATE_LISTEN            = 0x28U,   /*!< Address Listen Mode is ongoing            */
  HAL_SMBUS_STATE_BUSY_TX_LISTEN    = 0x29U,   /*!< Address Listen Mode and Data Transmission
                                                 process is ongoing                           */
  HAL_SMBUS_STATE_BUSY_RX_LISTEN    = 0x2AU,   /*!< Address Listen Mode and Data Reception
                                                 process is ongoing                           */
  HAL_SMBUS_STATE_ABORT             = 0x60U,   /*!< Abort user request ongoing                */
  HAL_SMBUS_STATE_TIMEOUT           = 0xA0U,   /*!< Timeout state                             */
  HAL_SMBUS_STATE_ERROR             = 0xE0U    /*!< Error                                     */
} HAL_SMBUS_StateTypeDef;

/**
  * @brief  HAL Mode structure definition
  * @note   HAL SMBUS Mode value coding follow below described bitmap :
  *          b7     (not used)
  *             x  : Should be set to 0
  *          b6     (not used)
  *             x  : Should be set to 0
  *          b5
  *             0  : None
  *             1  : Slave (HAL SMBUS communication is in Slave/Device Mode)
  *          b4
  *             0  : None
  *             1  : Master (HAL SMBUS communication is in Master/Host Mode)
  *          b3-b2-b1-b0  (not used)
  *             xxxx : Should be set to 0000
  */
typedef enum
{
  HAL_SMBUS_MODE_NONE               = 0x00U,   /*!< No SMBUS communication on going              */
  HAL_SMBUS_MODE_MASTER             = 0x10U,   /*!< SMBUS communication is in Master Mode        */
  HAL_SMBUS_MODE_SLAVE              = 0x20U,   /*!< SMBUS communication is in Slave Mode         */

} HAL_SMBUS_ModeTypeDef;

/**
  * @brief  SMBUS handle Structure definition
  */
typedef struct __SMBUS_HandleTypeDef
{
  I2C_TypeDef                 *Instance;        /*!< SMBUS registers base address                  */

  SMBUS_InitTypeDef             Init;           /*!< SMBUS communication parameters              */

  uint8_t                       *pBuffPtr;      /*!< Pointer to SMBUS transfer buffer            */

  uint16_t                      XferSize;       /*!< SMBUS transfer size                         */

  __IO uint16_t                 XferCount;      /*!< SMBUS transfer counter                      */

  __IO uint32_t                 XferOptions;    /*!< SMBUS transfer options this parameter can
                                                     be a value of @ref SMBUS_OPTIONS            */

  __IO uint32_t                 PreviousState;  /*!< SMBUS communication Previous state and mode
                                                     context for internal usage                  */

  HAL_LockTypeDef               Lock;           /*!< SMBUS locking object                        */

  __IO HAL_SMBUS_StateTypeDef   State;          /*!< SMBUS communication state                   */

  __IO HAL_SMBUS_ModeTypeDef    Mode;           /*!< SMBUS communication mode                    */

  __IO uint32_t                 ErrorCode;      /*!< SMBUS Error code                            */

  __IO uint32_t                 Devaddress;     /*!< SMBUS Target device address                 */

  __IO uint32_t                 EventCount;     /*!< SMBUS Event counter                         */

  uint8_t                       XferPEC;        /*!< SMBUS PEC data in reception mode            */

#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
  void (* MasterTxCpltCallback)(struct __SMBUS_HandleTypeDef *hsmbus);           /*!< SMBUS Master Tx Transfer completed callback */
  void (* MasterRxCpltCallback)(struct __SMBUS_HandleTypeDef *hsmbus);           /*!< SMBUS Master Rx Transfer completed callback */
  void (* SlaveTxCpltCallback)(struct __SMBUS_HandleTypeDef *hsmbus);            /*!< SMBUS Slave Tx Transfer completed callback  */
  void (* SlaveRxCpltCallback)(struct __SMBUS_HandleTypeDef *hsmbus);            /*!< SMBUS Slave Rx Transfer completed callback  */
  void (* ListenCpltCallback)(struct __SMBUS_HandleTypeDef *hsmbus);             /*!< SMBUS Listen Complete callback              */
  void (* MemTxCpltCallback)(struct __SMBUS_HandleTypeDef *hsmbus);              /*!< SMBUS Memory Tx Transfer completed callback */
  void (* MemRxCpltCallback)(struct __SMBUS_HandleTypeDef *hsmbus);              /*!< SMBUS Memory Rx Transfer completed callback */
  void (* ErrorCallback)(struct __SMBUS_HandleTypeDef *hsmbus);                  /*!< SMBUS Error callback                        */
  void (* AbortCpltCallback)(struct __SMBUS_HandleTypeDef *hsmbus);              /*!< SMBUS Abort callback                        */
  void (* AddrCallback)(struct __SMBUS_HandleTypeDef *hsmbus, uint8_t TransferDirection, uint16_t AddrMatchCode);  /*!< SMBUS Slave Address Match callback */
  void (* MspInitCallback)(struct __SMBUS_HandleTypeDef *hsmbus);                /*!< SMBUS Msp Init callback                     */
  void (* MspDeInitCallback)(struct __SMBUS_HandleTypeDef *hsmbus);              /*!< SMBUS Msp DeInit callback                   */

#endif  /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
} SMBUS_HandleTypeDef;

#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL SMBUS Callback ID enumeration definition
  */
typedef enum
{
  HAL_SMBUS_MASTER_TX_COMPLETE_CB_ID      = 0x00U,    /*!< SMBUS Master Tx Transfer completed callback ID  */
  HAL_SMBUS_MASTER_RX_COMPLETE_CB_ID      = 0x01U,    /*!< SMBUS Master Rx Transfer completed callback ID  */
  HAL_SMBUS_SLAVE_TX_COMPLETE_CB_ID       = 0x02U,    /*!< SMBUS Slave Tx Transfer completed callback ID   */
  HAL_SMBUS_SLAVE_RX_COMPLETE_CB_ID       = 0x03U,    /*!< SMBUS Slave Rx Transfer completed callback ID   */
  HAL_SMBUS_LISTEN_COMPLETE_CB_ID         = 0x04U,    /*!< SMBUS Listen Complete callback ID               */
  HAL_SMBUS_ERROR_CB_ID                   = 0x07U,    /*!< SMBUS Error callback ID                         */
  HAL_SMBUS_ABORT_CB_ID                   = 0x08U,    /*!< SMBUS Abort callback ID                         */
  HAL_SMBUS_MSPINIT_CB_ID                 = 0x09U,    /*!< SMBUS Msp Init callback ID                      */
  HAL_SMBUS_MSPDEINIT_CB_ID               = 0x0AU     /*!< SMBUS Msp DeInit callback ID                    */

} HAL_SMBUS_CallbackIDTypeDef;

/**
  * @brief  HAL SMBUS Callback pointer definition
  */
typedef  void (*pSMBUS_CallbackTypeDef)(SMBUS_HandleTypeDef *hsmbus); /*!< pointer to an I2C callback function */
typedef  void (*pSMBUS_AddrCallbackTypeDef)(SMBUS_HandleTypeDef *hsmbus, uint8_t TransferDirection, uint16_t AddrMatchCode); /*!< pointer to an I2C Address Match callback function */

#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup SMBUS_Exported_Constants SMBUS Exported Constants
  * @{
  */

/** @defgroup SMBUS_Error_Code_definition SMBUS Error Code
  * @brief    SMBUS Error Code
  * @{
  */
#define HAL_SMBUS_ERROR_NONE              0x00000000U    /*!< No error               */
#define HAL_SMBUS_ERROR_BERR              0x00000001U    /*!< BERR error             */
#define HAL_SMBUS_ERROR_ARLO              0x00000002U    /*!< ARLO error             */
#define HAL_SMBUS_ERROR_AF                0x00000004U    /*!< AF error               */
#define HAL_SMBUS_ERROR_OVR               0x00000008U    /*!< OVR error              */
#define HAL_SMBUS_ERROR_TIMEOUT           0x00000010U    /*!< Timeout Error          */
#define HAL_SMBUS_ERROR_ALERT             0x00000020U    /*!< Alert error            */
#define HAL_SMBUS_ERROR_PECERR            0x00000040U    /*!< PEC error              */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
#define HAL_SMBUS_ERROR_INVALID_CALLBACK  0x00000080U    /*!< Invalid Callback error */
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup SMBUS_Analog_Filter SMBUS Analog Filter
  * @{
  */
#define SMBUS_ANALOGFILTER_ENABLE        0x00000000U
#define SMBUS_ANALOGFILTER_DISABLE       I2C_FLTR_ANOFF
/**
  * @}
  */

/** @defgroup SMBUS_addressing_mode SMBUS addressing mode
 * @{
 */
#define SMBUS_ADDRESSINGMODE_7BIT        0x00004000U
#define SMBUS_ADDRESSINGMODE_10BIT       (I2C_OAR1_ADDMODE | 0x00004000U)
/**
  * @}
  */

/** @defgroup SMBUS_dual_addressing_mode  SMBUS dual addressing mode
  * @{
  */
#define SMBUS_DUALADDRESS_DISABLE        0x00000000U
#define SMBUS_DUALADDRESS_ENABLE         I2C_OAR2_ENDUAL
/**
  * @}
  */

/** @defgroup SMBUS_general_call_addressing_mode SMBUS general call addressing mode
  * @{
  */
#define SMBUS_GENERALCALL_DISABLE        0x00000000U
#define SMBUS_GENERALCALL_ENABLE         I2C_CR1_ENGC
/**
  * @}
  */

/** @defgroup SMBUS_nostretch_mode SMBUS nostretch mode
  * @{
  */
#define SMBUS_NOSTRETCH_DISABLE          0x00000000U
#define SMBUS_NOSTRETCH_ENABLE           I2C_CR1_NOSTRETCH
/**
  * @}
  */

/** @defgroup SMBUS_packet_error_check_mode SMBUS packet error check mode
  * @{
  */
#define SMBUS_PEC_DISABLE                0x00000000U
#define SMBUS_PEC_ENABLE                 I2C_CR1_ENPEC
/**
  * @}
  */

/** @defgroup SMBUS_peripheral_mode SMBUS peripheral mode
* @{
*/
#define SMBUS_PERIPHERAL_MODE_SMBUS_HOST        (uint32_t)(I2C_CR1_SMBUS | I2C_CR1_SMBTYPE | I2C_CR1_ENARP)
#define SMBUS_PERIPHERAL_MODE_SMBUS_SLAVE       I2C_CR1_SMBUS
#define SMBUS_PERIPHERAL_MODE_SMBUS_SLAVE_ARP   (uint32_t)(I2C_CR1_SMBUS | I2C_CR1_ENARP)
/**
* @}
*/

/** @defgroup SMBUS_XferDirection_definition SMBUS XferDirection definition
  * @{
  */
#define SMBUS_DIRECTION_RECEIVE           0x00000000U
#define SMBUS_DIRECTION_TRANSMIT          0x00000001U
/**
  * @}
  */

/** @defgroup SMBUS_XferOptions_definition SMBUS XferOptions definition
  * @{
  */
#define  SMBUS_FIRST_FRAME                       0x00000001U
#define  SMBUS_NEXT_FRAME                        0x00000002U
#define  SMBUS_FIRST_AND_LAST_FRAME_NO_PEC       0x00000003U
#define  SMBUS_LAST_FRAME_NO_PEC                 0x00000004U
#define  SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC     0x00000005U
#define  SMBUS_LAST_FRAME_WITH_PEC               0x00000006U
/**
  * @}
  */

/** @defgroup SMBUS_Interrupt_configuration_definition SMBUS Interrupt configuration definition
  * @{
  */
#define SMBUS_IT_BUF                      I2C_CR2_ITBUFEN
#define SMBUS_IT_EVT                      I2C_CR2_ITEVTEN
#define SMBUS_IT_ERR                      I2C_CR2_ITERREN
/**
  * @}
  */

/** @defgroup SMBUS_Flag_definition SMBUS Flag definition
  * @{
  */
#define SMBUS_FLAG_SMBALERT               0x00018000U
#define SMBUS_FLAG_TIMEOUT                0x00014000U
#define SMBUS_FLAG_PECERR                 0x00011000U
#define SMBUS_FLAG_OVR                    0x00010800U
#define SMBUS_FLAG_AF                     0x00010400U
#define SMBUS_FLAG_ARLO                   0x00010200U
#define SMBUS_FLAG_BERR                   0x00010100U
#define SMBUS_FLAG_TXE                    0x00010080U
#define SMBUS_FLAG_RXNE                   0x00010040U
#define SMBUS_FLAG_STOPF                  0x00010010U
#define SMBUS_FLAG_ADD10                  0x00010008U
#define SMBUS_FLAG_BTF                    0x00010004U
#define SMBUS_FLAG_ADDR                   0x00010002U
#define SMBUS_FLAG_SB                     0x00010001U
#define SMBUS_FLAG_DUALF                  0x00100080U
#define SMBUS_FLAG_SMBHOST                0x00100040U
#define SMBUS_FLAG_SMBDEFAULT             0x00100020U
#define SMBUS_FLAG_GENCALL                0x00100010U
#define SMBUS_FLAG_TRA                    0x00100004U
#define SMBUS_FLAG_BUSY                   0x00100002U
#define SMBUS_FLAG_MSL                    0x00100001U
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup SMBUS_Exported_Macros SMBUS Exported Macros
  * @{
  */

/** @brief Reset SMBUS handle state
  * @param  __HANDLE__ specifies the SMBUS Handle.
  *         This parameter can be SMBUS where x: 1, 2, or 3 to select the SMBUS peripheral.
  * @retval None
  */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
#define __HAL_SMBUS_RESET_HANDLE_STATE(__HANDLE__)          do{                                                  \
                                                                (__HANDLE__)->State = HAL_SMBUS_STATE_RESET;     \
                                                                (__HANDLE__)->MspInitCallback = NULL;            \
                                                                (__HANDLE__)->MspDeInitCallback = NULL;          \
                                                              } while(0)
#else
#define __HAL_SMBUS_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_SMBUS_STATE_RESET)
#endif

/** @brief  Enable or disable the specified SMBUS interrupts.
  * @param  __HANDLE__ specifies the SMBUS Handle.
  *         This parameter can be SMBUS where x: 1, 2, or 3 to select the SMBUS peripheral.
  * @param  __INTERRUPT__ specifies the interrupt source to enable or disable.
  *         This parameter can be one of the following values:
  *            @arg SMBUS_IT_BUF: Buffer interrupt enable
  *            @arg SMBUS_IT_EVT: Event interrupt enable
  *            @arg SMBUS_IT_ERR: Error interrupt enable
  * @retval None
  */
#define __HAL_SMBUS_ENABLE_IT(__HANDLE__, __INTERRUPT__)   ((__HANDLE__)->Instance->CR2 |= (__INTERRUPT__))
#define __HAL_SMBUS_DISABLE_IT(__HANDLE__, __INTERRUPT__)  ((__HANDLE__)->Instance->CR2 &= (~(__INTERRUPT__)))

/** @brief  Checks if the specified SMBUS interrupt source is enabled or disabled.
  * @param  __HANDLE__ specifies the SMBUS Handle.
  *         This parameter can be SMBUS where x: 1, 2, or 3 to select the SMBUS peripheral.
  * @param  __INTERRUPT__ specifies the SMBUS interrupt source to check.
  *          This parameter can be one of the following values:
  *            @arg SMBUS_IT_BUF: Buffer interrupt enable
  *            @arg SMBUS_IT_EVT: Event interrupt enable
  *            @arg SMBUS_IT_ERR: Error interrupt enable
  * @retval The new state of __INTERRUPT__ (TRUE or FALSE).
  */
#define __HAL_SMBUS_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) ((((__HANDLE__)->Instance->CR2 & (__INTERRUPT__)) == (__INTERRUPT__)) ? SET : RESET)

/** @brief  Checks whether the specified SMBUS flag is set or not.
  * @param  __HANDLE__ specifies the SMBUS Handle.
  *         This parameter can be SMBUS where x: 1, 2, or 3 to select the SMBUS peripheral.
  * @param  __FLAG__ specifies the flag to check.
  *         This parameter can be one of the following values:
  *            @arg SMBUS_FLAG_SMBALERT: SMBus Alert flag
  *            @arg SMBUS_FLAG_TIMEOUT: Timeout or Tlow error flag
  *            @arg SMBUS_FLAG_PECERR: PEC error in reception flag
  *            @arg SMBUS_FLAG_OVR: Overrun/Underrun flag
  *            @arg SMBUS_FLAG_AF: Acknowledge failure flag
  *            @arg SMBUS_FLAG_ARLO: Arbitration lost flag
  *            @arg SMBUS_FLAG_BERR: Bus error flag
  *            @arg SMBUS_FLAG_TXE: Data register empty flag
  *            @arg SMBUS_FLAG_RXNE: Data register not empty flag
  *            @arg SMBUS_FLAG_STOPF: Stop detection flag
  *            @arg SMBUS_FLAG_ADD10: 10-bit header sent flag
  *            @arg SMBUS_FLAG_BTF: Byte transfer finished flag
  *            @arg SMBUS_FLAG_ADDR: Address sent flag
  *                                Address matched flag
  *            @arg SMBUS_FLAG_SB: Start bit flag
  *            @arg SMBUS_FLAG_DUALF: Dual flag
  *            @arg SMBUS_FLAG_SMBHOST: SMBus host header
  *            @arg SMBUS_FLAG_SMBDEFAULT: SMBus default header
  *            @arg SMBUS_FLAG_GENCALL: General call header flag
  *            @arg SMBUS_FLAG_TRA: Transmitter/Receiver flag
  *            @arg SMBUS_FLAG_BUSY: Bus busy flag
  *            @arg SMBUS_FLAG_MSL: Master/Slave flag
  * @retval The new state of __FLAG__ (TRUE or FALSE).
  */
#define __HAL_SMBUS_GET_FLAG(__HANDLE__, __FLAG__) ((((uint8_t)((__FLAG__) >> 16U)) == 0x01U)?((((__HANDLE__)->Instance->SR1) & ((__FLAG__) & SMBUS_FLAG_MASK)) == ((__FLAG__) & SMBUS_FLAG_MASK)): \
                                                 ((((__HANDLE__)->Instance->SR2) & ((__FLAG__) & SMBUS_FLAG_MASK)) == ((__FLAG__) & SMBUS_FLAG_MASK)))

/** @brief  Clears the SMBUS pending flags which are cleared by writing 0 in a specific bit.
  * @param  __HANDLE__ specifies the SMBUS Handle.
  *         This parameter can be SMBUS where x: 1, 2, or 3 to select the SMBUS peripheral.
  * @param  __FLAG__ specifies the flag to clear.
  *         This parameter can be any combination of the following values:
  *            @arg SMBUS_FLAG_SMBALERT: SMBus Alert flag
  *            @arg SMBUS_FLAG_TIMEOUT: Timeout or Tlow error flag
  *            @arg SMBUS_FLAG_PECERR: PEC error in reception flag
  *            @arg SMBUS_FLAG_OVR: Overrun/Underrun flag (Slave mode)
  *            @arg SMBUS_FLAG_AF: Acknowledge failure flag
  *            @arg SMBUS_FLAG_ARLO: Arbitration lost flag (Master mode)
  *            @arg SMBUS_FLAG_BERR: Bus error flag
  * @retval None
  */
#define __HAL_SMBUS_CLEAR_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->SR1 = ~((__FLAG__) & SMBUS_FLAG_MASK))

/** @brief  Clears the SMBUS ADDR pending flag.
  * @param  __HANDLE__ specifies the SMBUS Handle.
  *         This parameter can be SMBUS where x: 1, 2, or 3 to select the SMBUS peripheral.
  * @retval None
  */
#define __HAL_SMBUS_CLEAR_ADDRFLAG(__HANDLE__)    \
  do{                                           \
    __IO uint32_t tmpreg = 0x00U;               \
    tmpreg = (__HANDLE__)->Instance->SR1;       \
    tmpreg = (__HANDLE__)->Instance->SR2;       \
    UNUSED(tmpreg);                             \
  } while(0)

/** @brief  Clears the SMBUS STOPF pending flag.
  * @param  __HANDLE__ specifies the SMBUS Handle.
  *         This parameter can be SMBUS where x: 1, 2, or 3 to select the SMBUS peripheral.
  * @retval None
  */
#define __HAL_SMBUS_CLEAR_STOPFLAG(__HANDLE__)    \
  do{                                           \
    __IO uint32_t tmpreg = 0x00U;               \
    tmpreg = (__HANDLE__)->Instance->SR1;       \
    (__HANDLE__)->Instance->CR1 |= I2C_CR1_PE;  \
    UNUSED(tmpreg);                             \
  } while(0)

/** @brief  Enable the SMBUS peripheral.
  * @param  __HANDLE__ specifies the SMBUS Handle.
  *         This parameter can be SMBUSx where x: 1 or 2  to select the SMBUS peripheral.
  * @retval None
  */
#define __HAL_SMBUS_ENABLE(__HANDLE__)           ((__HANDLE__)->Instance->CR1 |=  I2C_CR1_PE)

/** @brief  Disable the SMBUS peripheral.
  * @param  __HANDLE__ specifies the SMBUS Handle.
  *         This parameter can be SMBUSx where x: 1 or 2  to select the SMBUS peripheral.
  * @retval None
  */
#define __HAL_SMBUS_DISABLE(__HANDLE__)          ((__HANDLE__)->Instance->CR1 &=  ~I2C_CR1_PE)

/** @brief  Generate a Non-Acknowledge SMBUS peripheral in Slave mode.
  * @param  __HANDLE__ specifies the SMBUS Handle.
  * @retval None
  */
#define __HAL_SMBUS_GENERATE_NACK(__HANDLE__)    (CLEAR_BIT((__HANDLE__)->Instance->CR1, I2C_CR1_ACK))

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup SMBUS_Exported_Functions
  * @{
  */

/** @addtogroup SMBUS_Exported_Functions_Group1 Initialization and de-initialization functions
 * @{
 */

/* Initialization/de-initialization functions  **********************************/
HAL_StatusTypeDef HAL_SMBUS_Init(SMBUS_HandleTypeDef *hsmbus);
HAL_StatusTypeDef HAL_SMBUS_DeInit(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_MspInit(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_MspDeInit(SMBUS_HandleTypeDef *hsmbus);

/* Callbacks Register/UnRegister functions  ************************************/
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
HAL_StatusTypeDef HAL_SMBUS_RegisterCallback(SMBUS_HandleTypeDef *hsmbus, HAL_SMBUS_CallbackIDTypeDef CallbackID, pSMBUS_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_SMBUS_UnRegisterCallback(SMBUS_HandleTypeDef *hsmbus, HAL_SMBUS_CallbackIDTypeDef CallbackID);

HAL_StatusTypeDef HAL_SMBUS_RegisterAddrCallback(SMBUS_HandleTypeDef *hsmbus, pSMBUS_AddrCallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_SMBUS_UnRegisterAddrCallback(SMBUS_HandleTypeDef *hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @addtogroup SMBUS_Exported_Functions_Group2 Input and Output operation functions
  * @{
  */

/* IO operation functions  *****************************************************/
/** @addtogroup Blocking_mode_Polling Blocking mode Polling
 * @{
 */
/******* Blocking mode: Polling */
HAL_StatusTypeDef HAL_SMBUS_IsDeviceReady(SMBUS_HandleTypeDef *hsmbus, uint16_t DevAddress, uint32_t Trials, uint32_t Timeout);
/**
  * @}
  */

/** @addtogroup Non-Blocking_mode_Interrupt Non-Blocking mode Interrupt
 * @{
 */
/******* Non-Blocking mode: Interrupt */
HAL_StatusTypeDef HAL_SMBUS_Master_Transmit_IT(SMBUS_HandleTypeDef *hsmbus, uint16_t DevAddress, uint8_t *pData, uint16_t Size, uint32_t XferOptions);
HAL_StatusTypeDef HAL_SMBUS_Master_Receive_IT(SMBUS_HandleTypeDef *hsmbus, uint16_t DevAddress, uint8_t *pData, uint16_t Size, uint32_t XferOptions);
HAL_StatusTypeDef HAL_SMBUS_Master_Abort_IT(SMBUS_HandleTypeDef *hsmbus, uint16_t DevAddress);
HAL_StatusTypeDef HAL_SMBUS_Slave_Transmit_IT(SMBUS_HandleTypeDef *hsmbus, uint8_t *pData, uint16_t Size, uint32_t XferOptions);
HAL_StatusTypeDef HAL_SMBUS_Slave_Receive_IT(SMBUS_HandleTypeDef *hsmbus, uint8_t *pData, uint16_t Size, uint32_t XferOptions);

HAL_StatusTypeDef HAL_SMBUS_EnableAlert_IT(SMBUS_HandleTypeDef *hsmbus);
HAL_StatusTypeDef HAL_SMBUS_DisableAlert_IT(SMBUS_HandleTypeDef *hsmbus);
HAL_StatusTypeDef HAL_SMBUS_EnableListen_IT(SMBUS_HandleTypeDef *hsmbus);
HAL_StatusTypeDef HAL_SMBUS_DisableListen_IT(SMBUS_HandleTypeDef *hsmbus);

/****** Filter Configuration functions  */
#if  defined(I2C_FLTR_ANOFF)&&defined(I2C_FLTR_DNF)
HAL_StatusTypeDef HAL_SMBUS_ConfigAnalogFilter(SMBUS_HandleTypeDef *hsmbus, uint32_t AnalogFilter);
HAL_StatusTypeDef HAL_SMBUS_ConfigDigitalFilter(SMBUS_HandleTypeDef *hsmbus, uint32_t DigitalFilter);
#endif
/**
  * @}
  */

/** @addtogroup SMBUS_IRQ_Handler_and_Callbacks IRQ Handler and Callbacks
 * @{
 */
/******* SMBUS IRQHandler and Callbacks used in non blocking modes (Interrupt) */
void HAL_SMBUS_EV_IRQHandler(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_ER_IRQHandler(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_MasterTxCpltCallback(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_MasterRxCpltCallback(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_SlaveTxCpltCallback(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_SlaveRxCpltCallback(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_AddrCallback(SMBUS_HandleTypeDef *hsmbus, uint8_t TransferDirection, uint16_t AddrMatchCode);
void HAL_SMBUS_ListenCpltCallback(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_ErrorCallback(SMBUS_HandleTypeDef *hsmbus);
void HAL_SMBUS_AbortCpltCallback(SMBUS_HandleTypeDef *hsmbus);

/**
  * @}
  */

/** @addtogroup SMBUS_Exported_Functions_Group3 Peripheral State, Mode and Error functions
  * @{
  */

/* Peripheral State, mode and Errors functions  **************************************************/
HAL_SMBUS_StateTypeDef HAL_SMBUS_GetState(SMBUS_HandleTypeDef *hsmbus);
HAL_SMBUS_ModeTypeDef HAL_SMBUS_GetMode(SMBUS_HandleTypeDef *hsmbus);
uint32_t HAL_SMBUS_GetError(SMBUS_HandleTypeDef *hsmbus);

/**
  * @}
  */

/**
  * @}
  */
/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup SMBUS_Private_Constants SMBUS Private Constants
  * @{
  */
#define SMBUS_FLAG_MASK  0x0000FFFFU
/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup SMBUS_Private_Macros SMBUS Private Macros
  * @{
  */

#define SMBUS_FREQRANGE(__PCLK__)                  ((__PCLK__)/1000000U)

#define SMBUS_RISE_TIME(__FREQRANGE__)             ( ((__FREQRANGE__) + 1U))

#define SMBUS_SPEED_STANDARD(__PCLK__, __SPEED__)  (((((__PCLK__)/((__SPEED__) << 1U)) & I2C_CCR_CCR) < 4U)? 4U:((__PCLK__) / ((__SPEED__) << 1U)))

#define SMBUS_7BIT_ADD_WRITE(__ADDRESS__)          ((uint8_t)((__ADDRESS__) & (~I2C_OAR1_ADD0)))

#define SMBUS_7BIT_ADD_READ(__ADDRESS__)           ((uint8_t)((__ADDRESS__) | I2C_OAR1_ADD0))

#define SMBUS_10BIT_ADDRESS(__ADDRESS__)           ((uint8_t)((uint16_t)((__ADDRESS__) & (uint16_t)0x00FF)))

#define SMBUS_10BIT_HEADER_WRITE(__ADDRESS__)      ((uint8_t)((uint16_t)((uint16_t)(((uint16_t)((__ADDRESS__) & (uint16_t)0x0300)) >> 7) | (uint16_t)0x00F0)))

#define SMBUS_10BIT_HEADER_READ(__ADDRESS__)       ((uint8_t)((uint16_t)((uint16_t)(((uint16_t)((__ADDRESS__) & (uint16_t)0x0300)) >> 7) | (uint16_t)(0x00F1))))

#define SMBUS_GET_PEC_MODE(__HANDLE__)             ((__HANDLE__)->Instance->CR1 & I2C_CR1_ENPEC)

#define SMBUS_GET_PEC_VALUE(__HANDLE__)             ((__HANDLE__)->XferPEC)

#if  defined(I2C_FLTR_ANOFF)&&defined(I2C_FLTR_DNF)
#define IS_SMBUS_ANALOG_FILTER(FILTER)             (((FILTER) == SMBUS_ANALOGFILTER_ENABLE) || \
                                                    ((FILTER) == SMBUS_ANALOGFILTER_DISABLE))
#define IS_SMBUS_DIGITAL_FILTER(FILTER)            ((FILTER) <= 0x0000000FU)
#endif
#define IS_SMBUS_ADDRESSING_MODE(ADDRESS)          (((ADDRESS) == SMBUS_ADDRESSINGMODE_7BIT) || \
                                                    ((ADDRESS) == SMBUS_ADDRESSINGMODE_10BIT))

#define IS_SMBUS_DUAL_ADDRESS(ADDRESS)             (((ADDRESS) == SMBUS_DUALADDRESS_DISABLE) || \
                                                    ((ADDRESS) == SMBUS_DUALADDRESS_ENABLE))

#define IS_SMBUS_GENERAL_CALL(CALL)                (((CALL) == SMBUS_GENERALCALL_DISABLE)    || \
                                                    ((CALL) == SMBUS_GENERALCALL_ENABLE))

#define IS_SMBUS_NO_STRETCH(STRETCH)               (((STRETCH) == SMBUS_NOSTRETCH_DISABLE)   || \
                                                    ((STRETCH) == SMBUS_NOSTRETCH_ENABLE))

#define IS_SMBUS_PEC(PEC)                          (((PEC) == SMBUS_PEC_DISABLE) || \
                                                     ((PEC) == SMBUS_PEC_ENABLE))

#define IS_SMBUS_PERIPHERAL_MODE(MODE)             (((MODE) == SMBUS_PERIPHERAL_MODE_SMBUS_HOST)      || \
                                                    ((MODE) == SMBUS_PERIPHERAL_MODE_SMBUS_SLAVE)     || \
                                                    ((MODE) == SMBUS_PERIPHERAL_MODE_SMBUS_SLAVE_ARP))

#define IS_SMBUS_CLOCK_SPEED(SPEED)                (((SPEED) > 0U) && ((SPEED) <= 100000U))

#define IS_SMBUS_OWN_ADDRESS1(ADDRESS1)            (((ADDRESS1) & 0xFFFFFC00U) == 0U)

#define IS_SMBUS_OWN_ADDRESS2(ADDRESS2)            (((ADDRESS2) & 0xFFFFFF01U) == 0U)

#define IS_SMBUS_TRANSFER_OPTIONS_REQUEST(REQUEST) (((REQUEST) == SMBUS_FIRST_FRAME)                   || \
                                                    ((REQUEST) == SMBUS_NEXT_FRAME)                    || \
                                                    ((REQUEST) == SMBUS_FIRST_AND_LAST_FRAME_NO_PEC)   || \
                                                    ((REQUEST) == SMBUS_LAST_FRAME_NO_PEC)             || \
                                                    ((REQUEST) == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || \
                                                    ((REQUEST) == SMBUS_LAST_FRAME_WITH_PEC))

/**
  * @}
  */

/* Private Functions ---------------------------------------------------------*/
/** @defgroup SMBUS_Private_Functions SMBUS Private Functions
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

/**
* @}
*/

#ifdef __cplusplus
}
#endif


#endif /* __STM32F4xx_HAL_SMBUS_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
