/**
  ******************************************************************************
  * @file    stm32f4xx_hal_irda.h
  * @author  MCD Application Team
  * @brief   Header file of IRDA HAL module.
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
#ifndef __STM32F4xx_HAL_IRDA_H
#define __STM32F4xx_HAL_IRDA_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup IRDA
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup IRDA_Exported_Types IRDA Exported Types
  * @{
  */
/**
  * @brief IRDA Init Structure definition
  */
typedef struct
{
  uint32_t BaudRate;                  /*!< This member configures the IRDA communication baud rate.
                                           The baud rate is computed using the following formula:
                                           - IntegerDivider = ((PCLKx) / (8 * (hirda->Init.BaudRate)))
                                           - FractionalDivider = ((IntegerDivider - ((uint32_t) IntegerDivider)) * 8) + 0.5 */

  uint32_t WordLength;                /*!< Specifies the number of data bits transmitted or received in a frame.
                                           This parameter can be a value of @ref IRDA_Word_Length */

  uint32_t Parity;                    /*!< Specifies the parity mode.
                                           This parameter can be a value of @ref IRDA_Parity
                                           @note When parity is enabled, the computed parity is inserted
                                                 at the MSB position of the transmitted data (9th bit when
                                                 the word length is set to 9 data bits; 8th bit when the
                                                 word length is set to 8 data bits). */

  uint32_t Mode;                      /*!< Specifies whether the Receive or Transmit mode is enabled or disabled.
                                           This parameter can be a value of @ref IRDA_Mode */

  uint8_t  Prescaler;                 /*!< Specifies the Prescaler value to be programmed
                                           in the IrDA low-power Baud Register, for defining pulse width on which
                                           burst acceptance/rejection will be decided. This value is used as divisor
                                           of system clock to achieve required pulse width. */

  uint32_t IrDAMode;                  /*!< Specifies the IrDA mode
                                           This parameter can be a value of @ref IRDA_Low_Power */
} IRDA_InitTypeDef;

/**
  * @brief HAL IRDA State structures definition
  * @note  HAL IRDA State value is a combination of 2 different substates: gState and RxState.
  *        - gState contains IRDA state information related to global Handle management
  *          and also information related to Tx operations.
  *          gState value coding follow below described bitmap :
  *          b7-b6  Error information
  *             00 : No Error
  *             01 : (Not Used)
  *             10 : Timeout
  *             11 : Error
  *          b5     IP initialisation status
  *             0  : Reset (IP not initialized)
  *             1  : Init done (IP initialized. HAL IRDA Init function already called)
  *          b4-b3  (not used)
  *             xx : Should be set to 00
  *          b2     Intrinsic process state
  *             0  : Ready
  *             1  : Busy (IP busy with some configuration or internal operations)
  *          b1     (not used)
  *             x  : Should be set to 0
  *          b0     Tx state
  *             0  : Ready (no Tx operation ongoing)
  *             1  : Busy (Tx operation ongoing)
  *        - RxState contains information related to Rx operations.
  *          RxState value coding follow below described bitmap :
  *          b7-b6  (not used)
  *             xx : Should be set to 00
  *          b5     IP initialisation status
  *             0  : Reset (IP not initialized)
  *             1  : Init done (IP initialized)
  *          b4-b2  (not used)
  *            xxx : Should be set to 000
  *          b1     Rx state
  *             0  : Ready (no Rx operation ongoing)
  *             1  : Busy (Rx operation ongoing)
  *          b0     (not used)
  *             x  : Should be set to 0.
  */
typedef enum
{
  HAL_IRDA_STATE_RESET             = 0x00U,    /*!< Peripheral is not yet Initialized
                                                   Value is allowed for gState and RxState */
  HAL_IRDA_STATE_READY             = 0x20U,    /*!< Peripheral Initialized and ready for use
                                                   Value is allowed for gState and RxState */
  HAL_IRDA_STATE_BUSY              = 0x24U,    /*!< An internal process is ongoing
                                                   Value is allowed for gState only */
  HAL_IRDA_STATE_BUSY_TX           = 0x21U,    /*!< Data Transmission process is ongoing
                                                   Value is allowed for gState only */
  HAL_IRDA_STATE_BUSY_RX           = 0x22U,    /*!< Data Reception process is ongoing
                                                   Value is allowed for RxState only */
  HAL_IRDA_STATE_BUSY_TX_RX        = 0x23U,    /*!< Data Transmission and Reception process is ongoing
                                                   Not to be used for neither gState nor RxState.
                                                   Value is result of combination (Or) between gState and RxState values */
  HAL_IRDA_STATE_TIMEOUT           = 0xA0U,    /*!< Timeout state
                                                   Value is allowed for gState only */
  HAL_IRDA_STATE_ERROR             = 0xE0U     /*!< Error
                                                   Value is allowed for gState only */
} HAL_IRDA_StateTypeDef;

/**
  * @brief IRDA handle Structure definition
  */
#if (USE_HAL_IRDA_REGISTER_CALLBACKS == 1)
typedef struct __IRDA_HandleTypeDef
#else
typedef struct
#endif  /* USE_HAL_IRDA_REGISTER_CALLBACKS */
{
  USART_TypeDef               *Instance;        /*!<  USART registers base address       */

  IRDA_InitTypeDef            Init;             /*!<  IRDA communication parameters      */

  uint8_t                     *pTxBuffPtr;      /*!<  Pointer to IRDA Tx transfer Buffer */

  uint16_t                    TxXferSize;       /*!<  IRDA Tx Transfer size              */

  __IO uint16_t               TxXferCount;      /*!<  IRDA Tx Transfer Counter           */

  uint8_t                     *pRxBuffPtr;      /*!<  Pointer to IRDA Rx transfer Buffer */

  uint16_t                    RxXferSize;       /*!<  IRDA Rx Transfer size              */

  __IO uint16_t               RxXferCount;      /*!<  IRDA Rx Transfer Counter           */

  DMA_HandleTypeDef           *hdmatx;          /*!<  IRDA Tx DMA Handle parameters      */

  DMA_HandleTypeDef           *hdmarx;          /*!<  IRDA Rx DMA Handle parameters      */

  HAL_LockTypeDef             Lock;             /*!<  Locking object                     */

  __IO HAL_IRDA_StateTypeDef  gState;           /*!<  IRDA state information related to global Handle management
                                                   and also related to Tx operations.
                                                   This parameter can be a value of @ref HAL_IRDA_StateTypeDef */

  __IO HAL_IRDA_StateTypeDef  RxState;          /*!<  IRDA state information related to Rx operations.
                                                   This parameter can be a value of @ref HAL_IRDA_StateTypeDef */

  __IO uint32_t               ErrorCode;        /*!< IRDA Error code                    */

#if (USE_HAL_IRDA_REGISTER_CALLBACKS == 1)
  void (* TxHalfCpltCallback)(struct __IRDA_HandleTypeDef *hirda);        /*!< IRDA Tx Half Complete Callback        */

  void (* TxCpltCallback)(struct __IRDA_HandleTypeDef *hirda);            /*!< IRDA Tx Complete Callback             */

  void (* RxHalfCpltCallback)(struct __IRDA_HandleTypeDef *hirda);        /*!< IRDA Rx Half Complete Callback        */

  void (* RxCpltCallback)(struct __IRDA_HandleTypeDef *hirda);            /*!< IRDA Rx Complete Callback             */

  void (* ErrorCallback)(struct __IRDA_HandleTypeDef *hirda);             /*!< IRDA Error Callback                   */

  void (* AbortCpltCallback)(struct __IRDA_HandleTypeDef *hirda);         /*!< IRDA Abort Complete Callback          */

  void (* AbortTransmitCpltCallback)(struct __IRDA_HandleTypeDef *hirda); /*!< IRDA Abort Transmit Complete Callback */

  void (* AbortReceiveCpltCallback)(struct __IRDA_HandleTypeDef *hirda);  /*!< IRDA Abort Receive Complete Callback  */


  void (* MspInitCallback)(struct __IRDA_HandleTypeDef *hirda);           /*!< IRDA Msp Init callback                */

  void (* MspDeInitCallback)(struct __IRDA_HandleTypeDef *hirda);         /*!< IRDA Msp DeInit callback              */
#endif  /* USE_HAL_IRDA_REGISTER_CALLBACKS */

} IRDA_HandleTypeDef;

#if (USE_HAL_IRDA_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL IRDA Callback ID enumeration definition
  */
typedef enum
{
  HAL_IRDA_TX_HALFCOMPLETE_CB_ID         = 0x00U,    /*!< IRDA Tx Half Complete Callback ID        */
  HAL_IRDA_TX_COMPLETE_CB_ID             = 0x01U,    /*!< IRDA Tx Complete Callback ID             */
  HAL_IRDA_RX_HALFCOMPLETE_CB_ID         = 0x02U,    /*!< IRDA Rx Half Complete Callback ID        */
  HAL_IRDA_RX_COMPLETE_CB_ID             = 0x03U,    /*!< IRDA Rx Complete Callback ID             */
  HAL_IRDA_ERROR_CB_ID                   = 0x04U,    /*!< IRDA Error Callback ID                   */
  HAL_IRDA_ABORT_COMPLETE_CB_ID          = 0x05U,    /*!< IRDA Abort Complete Callback ID          */
  HAL_IRDA_ABORT_TRANSMIT_COMPLETE_CB_ID = 0x06U,    /*!< IRDA Abort Transmit Complete Callback ID */
  HAL_IRDA_ABORT_RECEIVE_COMPLETE_CB_ID  = 0x07U,    /*!< IRDA Abort Receive Complete Callback ID  */

  HAL_IRDA_MSPINIT_CB_ID                 = 0x08U,    /*!< IRDA MspInit callback ID                 */
  HAL_IRDA_MSPDEINIT_CB_ID               = 0x09U     /*!< IRDA MspDeInit callback ID               */

} HAL_IRDA_CallbackIDTypeDef;

/**
  * @brief  HAL IRDA Callback pointer definition
  */
typedef  void (*pIRDA_CallbackTypeDef)(IRDA_HandleTypeDef *hirda);  /*!< pointer to an IRDA callback function */

#endif /* USE_HAL_IRDA_REGISTER_CALLBACKS */

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup IRDA_Exported_Constants IRDA Exported constants
  * @{
  */
/** @defgroup IRDA_Error_Code IRDA Error Code
  * @{
  */
#define HAL_IRDA_ERROR_NONE        0x00000000U   /*!< No error            */
#define HAL_IRDA_ERROR_PE          0x00000001U   /*!< Parity error        */
#define HAL_IRDA_ERROR_NE          0x00000002U   /*!< Noise error         */
#define HAL_IRDA_ERROR_FE          0x00000004U   /*!< Frame error         */
#define HAL_IRDA_ERROR_ORE         0x00000008U   /*!< Overrun error       */
#define HAL_IRDA_ERROR_DMA         0x00000010U   /*!< DMA transfer error  */
#if (USE_HAL_IRDA_REGISTER_CALLBACKS == 1)
#define HAL_IRDA_ERROR_INVALID_CALLBACK   ((uint32_t)0x00000020U)   /*!< Invalid Callback error  */
#endif /* USE_HAL_IRDA_REGISTER_CALLBACKS */
/**
  * @}
  */

/** @defgroup IRDA_Word_Length IRDA Word Length
  * @{
  */
#define IRDA_WORDLENGTH_8B         0x00000000U
#define IRDA_WORDLENGTH_9B         ((uint32_t)USART_CR1_M)
/**
  * @}
  */

/** @defgroup IRDA_Parity  IRDA Parity
  * @{
  */
#define IRDA_PARITY_NONE           0x00000000U
#define IRDA_PARITY_EVEN           ((uint32_t)USART_CR1_PCE)
#define IRDA_PARITY_ODD            ((uint32_t)(USART_CR1_PCE | USART_CR1_PS))
/**
  * @}
  */

/** @defgroup IRDA_Mode IRDA Transfer Mode
  * @{
  */
#define IRDA_MODE_RX               ((uint32_t)USART_CR1_RE)
#define IRDA_MODE_TX               ((uint32_t)USART_CR1_TE)
#define IRDA_MODE_TX_RX            ((uint32_t)(USART_CR1_TE |USART_CR1_RE))
/**
  * @}
  */

/** @defgroup IRDA_Low_Power IRDA Low Power
  * @{
  */
#define IRDA_POWERMODE_LOWPOWER    ((uint32_t)USART_CR3_IRLP)
#define IRDA_POWERMODE_NORMAL      0x00000000U
/**
  * @}
  */

/** @defgroup IRDA_Flags IRDA Flags
  *        Elements values convention: 0xXXXX
  *           - 0xXXXX  : Flag mask in the SR register
  * @{
  */
#define IRDA_FLAG_TXE              ((uint32_t)USART_SR_TXE)
#define IRDA_FLAG_TC               ((uint32_t)USART_SR_TC)
#define IRDA_FLAG_RXNE             ((uint32_t)USART_SR_RXNE)
#define IRDA_FLAG_IDLE             ((uint32_t)USART_SR_IDLE)
#define IRDA_FLAG_ORE              ((uint32_t)USART_SR_ORE)
#define IRDA_FLAG_NE               ((uint32_t)USART_SR_NE)
#define IRDA_FLAG_FE               ((uint32_t)USART_SR_FE)
#define IRDA_FLAG_PE               ((uint32_t)USART_SR_PE)
/**
  * @}
  */

/** @defgroup IRDA_Interrupt_definition IRDA Interrupt Definitions
  *        Elements values convention: 0xY000XXXX
  *           - XXXX  : Interrupt mask in the XX register
  *           - Y  : Interrupt source register (2bits)
  *                 - 01: CR1 register
  *                 - 10: CR2 register
  *                 - 11: CR3 register
  * @{
  */
#define IRDA_IT_PE                 ((uint32_t)(IRDA_CR1_REG_INDEX << 28U | USART_CR1_PEIE))
#define IRDA_IT_TXE                ((uint32_t)(IRDA_CR1_REG_INDEX << 28U | USART_CR1_TXEIE))
#define IRDA_IT_TC                 ((uint32_t)(IRDA_CR1_REG_INDEX << 28U | USART_CR1_TCIE))
#define IRDA_IT_RXNE               ((uint32_t)(IRDA_CR1_REG_INDEX << 28U | USART_CR1_RXNEIE))
#define IRDA_IT_IDLE               ((uint32_t)(IRDA_CR1_REG_INDEX << 28U | USART_CR1_IDLEIE))

#define IRDA_IT_LBD                ((uint32_t)(IRDA_CR2_REG_INDEX << 28U | USART_CR2_LBDIE))

#define IRDA_IT_CTS                ((uint32_t)(IRDA_CR3_REG_INDEX << 28U | USART_CR3_CTSIE))
#define IRDA_IT_ERR                ((uint32_t)(IRDA_CR3_REG_INDEX << 28U | USART_CR3_EIE))
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup IRDA_Exported_Macros IRDA Exported Macros
  * @{
  */

/** @brief Reset IRDA handle gstate & RxState
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @retval None
  */
#if USE_HAL_IRDA_REGISTER_CALLBACKS == 1
#define __HAL_IRDA_RESET_HANDLE_STATE(__HANDLE__)  do{                                                   \
                                                       (__HANDLE__)->gState = HAL_IRDA_STATE_RESET;      \
                                                       (__HANDLE__)->RxState = HAL_IRDA_STATE_RESET;     \
                                                       (__HANDLE__)->MspInitCallback = NULL;             \
                                                       (__HANDLE__)->MspDeInitCallback = NULL;           \
                                                     } while(0U)
#else
#define __HAL_IRDA_RESET_HANDLE_STATE(__HANDLE__)  do{                                                   \
                                                       (__HANDLE__)->gState = HAL_IRDA_STATE_RESET;      \
                                                       (__HANDLE__)->RxState = HAL_IRDA_STATE_RESET;     \
                                                     } while(0U)
#endif /*USE_HAL_IRDA_REGISTER_CALLBACKS  */

/** @brief  Flush the IRDA DR register
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @retval None
  */
#define __HAL_IRDA_FLUSH_DRREGISTER(__HANDLE__) ((__HANDLE__)->Instance->DR)

/** @brief  Check whether the specified IRDA flag is set or not.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @param  __FLAG__ specifies the flag to check.
  *        This parameter can be one of the following values:
  *            @arg IRDA_FLAG_TXE:  Transmit data register empty flag
  *            @arg IRDA_FLAG_TC:   Transmission Complete flag
  *            @arg IRDA_FLAG_RXNE: Receive data register not empty flag
  *            @arg IRDA_FLAG_IDLE: Idle Line detection flag
  *            @arg IRDA_FLAG_ORE:  OverRun Error flag
  *            @arg IRDA_FLAG_NE:   Noise Error flag
  *            @arg IRDA_FLAG_FE:   Framing Error flag
  *            @arg IRDA_FLAG_PE:   Parity Error flag
  * @retval The new state of __FLAG__ (TRUE or FALSE).
  */
#define __HAL_IRDA_GET_FLAG(__HANDLE__, __FLAG__) (((__HANDLE__)->Instance->SR & (__FLAG__)) == (__FLAG__))

/** @brief  Clear the specified IRDA pending flag.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @param  __FLAG__ specifies the flag to check.
  *          This parameter can be any combination of the following values:
  *            @arg IRDA_FLAG_TC:   Transmission Complete flag.
  *            @arg IRDA_FLAG_RXNE: Receive data register not empty flag.
  *
  * @note   PE (Parity error), FE (Framing error), NE (Noise error), ORE (OverRun
  *          error) and IDLE (Idle line detected) flags are cleared by software
  *          sequence: a read operation to USART_SR register followed by a read
  *          operation to USART_DR register.
  * @note   RXNE flag can be also cleared by a read to the USART_DR register.
  * @note   TC flag can be also cleared by software sequence: a read operation to
  *          USART_SR register followed by a write operation to USART_DR register.
  * @note   TXE flag is cleared only by a write to the USART_DR register.
  * @retval None
  */
#define __HAL_IRDA_CLEAR_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->SR = ~(__FLAG__))

/** @brief  Clear the IRDA PE pending flag.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @retval None
  */
#define __HAL_IRDA_CLEAR_PEFLAG(__HANDLE__)     \
  do{                                           \
    __IO uint32_t tmpreg = 0x00U;               \
    tmpreg = (__HANDLE__)->Instance->SR;        \
    tmpreg = (__HANDLE__)->Instance->DR;        \
    UNUSED(tmpreg);                             \
  } while(0U)

/** @brief  Clear the IRDA FE pending flag.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @retval None
  */
#define __HAL_IRDA_CLEAR_FEFLAG(__HANDLE__) __HAL_IRDA_CLEAR_PEFLAG(__HANDLE__)

/** @brief  Clear the IRDA NE pending flag.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @retval None
  */
#define __HAL_IRDA_CLEAR_NEFLAG(__HANDLE__) __HAL_IRDA_CLEAR_PEFLAG(__HANDLE__)

/** @brief  Clear the IRDA ORE pending flag.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @retval None
  */
#define __HAL_IRDA_CLEAR_OREFLAG(__HANDLE__) __HAL_IRDA_CLEAR_PEFLAG(__HANDLE__)

/** @brief  Clear the IRDA IDLE pending flag.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @retval None
  */
#define __HAL_IRDA_CLEAR_IDLEFLAG(__HANDLE__) __HAL_IRDA_CLEAR_PEFLAG(__HANDLE__)

/** @brief  Enable the specified IRDA interrupt.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @param  __INTERRUPT__ specifies the IRDA interrupt source to enable.
  *          This parameter can be one of the following values:
  *            @arg IRDA_IT_TXE:  Transmit Data Register empty interrupt
  *            @arg IRDA_IT_TC:   Transmission complete interrupt
  *            @arg IRDA_IT_RXNE: Receive Data register not empty interrupt
  *            @arg IRDA_IT_IDLE: Idle line detection interrupt
  *            @arg IRDA_IT_PE:   Parity Error interrupt
  *            @arg IRDA_IT_ERR:  Error interrupt(Frame error, noise error, overrun error)
  * @retval None
  */
#define __HAL_IRDA_ENABLE_IT(__HANDLE__, __INTERRUPT__)   ((((__INTERRUPT__) >> 28U) == IRDA_CR1_REG_INDEX)? ((__HANDLE__)->Instance->CR1 |= ((__INTERRUPT__) & IRDA_IT_MASK)): \
                                                           (((__INTERRUPT__) >> 28U) == IRDA_CR2_REG_INDEX)? ((__HANDLE__)->Instance->CR2 |=  ((__INTERRUPT__) & IRDA_IT_MASK)): \
                                                            ((__HANDLE__)->Instance->CR3 |= ((__INTERRUPT__) & IRDA_IT_MASK)))
/** @brief  Disable the specified IRDA interrupt.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @param  __INTERRUPT__ specifies the IRDA interrupt source to disable.
  *          This parameter can be one of the following values:
  *            @arg IRDA_IT_TXE:  Transmit Data Register empty interrupt
  *            @arg IRDA_IT_TC:   Transmission complete interrupt
  *            @arg IRDA_IT_RXNE: Receive Data register not empty interrupt
  *            @arg IRDA_IT_IDLE: Idle line detection interrupt
  *            @arg IRDA_IT_PE:   Parity Error interrupt
  *            @arg IRDA_IT_ERR:  Error interrupt(Frame error, noise error, overrun error)
  * @retval None
  */
#define __HAL_IRDA_DISABLE_IT(__HANDLE__, __INTERRUPT__)  ((((__INTERRUPT__) >> 28U) == IRDA_CR1_REG_INDEX)? ((__HANDLE__)->Instance->CR1 &= ~((__INTERRUPT__) & IRDA_IT_MASK)): \
                                                           (((__INTERRUPT__) >> 28U) == IRDA_CR2_REG_INDEX)? ((__HANDLE__)->Instance->CR2 &= ~((__INTERRUPT__) & IRDA_IT_MASK)): \
                                                           ((__HANDLE__)->Instance->CR3 &= ~ ((__INTERRUPT__) & IRDA_IT_MASK)))

/** @brief  Check whether the specified IRDA interrupt has occurred or not.
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @param  __IT__ specifies the IRDA interrupt source to check.
  *          This parameter can be one of the following values:
  *            @arg IRDA_IT_TXE: Transmit Data Register empty interrupt
  *            @arg IRDA_IT_TC:  Transmission complete interrupt
  *            @arg IRDA_IT_RXNE: Receive Data register not empty interrupt
  *            @arg IRDA_IT_IDLE: Idle line detection interrupt
  *            @arg IRDA_IT_ERR: Error interrupt
  *            @arg IRDA_IT_PE: Parity Error interrupt
  * @retval The new state of __IT__ (TRUE or FALSE).
  */
#define __HAL_IRDA_GET_IT_SOURCE(__HANDLE__, __IT__) (((((__IT__) >> 28U) == IRDA_CR1_REG_INDEX)? (__HANDLE__)->Instance->CR1:(((((uint32_t)(__IT__)) >> 28U) == IRDA_CR2_REG_INDEX)? \
                                                      (__HANDLE__)->Instance->CR2 : (__HANDLE__)->Instance->CR3)) & (((uint32_t)(__IT__)) & IRDA_IT_MASK))

/** @brief  Macro to enable the IRDA's one bit sample method
  * @param  __HANDLE__ specifies the IRDA Handle.
  * @retval None
  */
#define __HAL_IRDA_ONE_BIT_SAMPLE_ENABLE(__HANDLE__) ((__HANDLE__)->Instance->CR3 |= USART_CR3_ONEBIT)

/** @brief  Macro to disable the IRDA's one bit sample method
  * @param  __HANDLE__ specifies the IRDA Handle.
  * @retval None
  */
#define __HAL_IRDA_ONE_BIT_SAMPLE_DISABLE(__HANDLE__) ((__HANDLE__)->Instance->CR3 &= (uint16_t)~((uint16_t)USART_CR3_ONEBIT))

/** @brief  Enable UART/USART associated to IRDA Handle
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @retval None
  */
#define __HAL_IRDA_ENABLE(__HANDLE__)                   (SET_BIT((__HANDLE__)->Instance->CR1, USART_CR1_UE))

/** @brief  Disable UART/USART associated to IRDA Handle
  * @param  __HANDLE__ specifies the IRDA Handle.
  *         IRDA Handle selects the USARTx or UARTy peripheral
  *         (USART,UART availability and x,y values depending on device).
  * @retval None
  */
#define __HAL_IRDA_DISABLE(__HANDLE__)                  (CLEAR_BIT((__HANDLE__)->Instance->CR1, USART_CR1_UE))

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup IRDA_Exported_Functions
  * @{
  */

/** @addtogroup IRDA_Exported_Functions_Group1
  * @{
  */
/* Initialization/de-initialization functions  **********************************/
HAL_StatusTypeDef HAL_IRDA_Init(IRDA_HandleTypeDef *hirda);
HAL_StatusTypeDef HAL_IRDA_DeInit(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_MspInit(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_MspDeInit(IRDA_HandleTypeDef *hirda);

#if (USE_HAL_IRDA_REGISTER_CALLBACKS == 1)
/* Callbacks Register/UnRegister functions  ***********************************/
HAL_StatusTypeDef HAL_IRDA_RegisterCallback(IRDA_HandleTypeDef *hirda, HAL_IRDA_CallbackIDTypeDef CallbackID, pIRDA_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_IRDA_UnRegisterCallback(IRDA_HandleTypeDef *hirda, HAL_IRDA_CallbackIDTypeDef CallbackID);
#endif /* USE_HAL_IRDA_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @addtogroup IRDA_Exported_Functions_Group2
  * @{
  */
/* IO operation functions *******************************************************/
HAL_StatusTypeDef HAL_IRDA_Transmit(IRDA_HandleTypeDef *hirda, uint8_t *pData, uint16_t Size, uint32_t Timeout);
HAL_StatusTypeDef HAL_IRDA_Receive(IRDA_HandleTypeDef *hirda, uint8_t *pData, uint16_t Size, uint32_t Timeout);
HAL_StatusTypeDef HAL_IRDA_Transmit_IT(IRDA_HandleTypeDef *hirda, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_IRDA_Receive_IT(IRDA_HandleTypeDef *hirda, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_IRDA_Transmit_DMA(IRDA_HandleTypeDef *hirda, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_IRDA_Receive_DMA(IRDA_HandleTypeDef *hirda, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_IRDA_DMAPause(IRDA_HandleTypeDef *hirda);
HAL_StatusTypeDef HAL_IRDA_DMAResume(IRDA_HandleTypeDef *hirda);
HAL_StatusTypeDef HAL_IRDA_DMAStop(IRDA_HandleTypeDef *hirda);
/* Transfer Abort functions */
HAL_StatusTypeDef HAL_IRDA_Abort(IRDA_HandleTypeDef *hirda);
HAL_StatusTypeDef HAL_IRDA_AbortTransmit(IRDA_HandleTypeDef *hirda);
HAL_StatusTypeDef HAL_IRDA_AbortReceive(IRDA_HandleTypeDef *hirda);
HAL_StatusTypeDef HAL_IRDA_Abort_IT(IRDA_HandleTypeDef *hirda);
HAL_StatusTypeDef HAL_IRDA_AbortTransmit_IT(IRDA_HandleTypeDef *hirda);
HAL_StatusTypeDef HAL_IRDA_AbortReceive_IT(IRDA_HandleTypeDef *hirda);

void HAL_IRDA_IRQHandler(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_TxCpltCallback(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_RxCpltCallback(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_TxHalfCpltCallback(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_RxHalfCpltCallback(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_ErrorCallback(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_AbortCpltCallback(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_AbortTransmitCpltCallback(IRDA_HandleTypeDef *hirda);
void HAL_IRDA_AbortReceiveCpltCallback(IRDA_HandleTypeDef *hirda);
/**
  * @}
  */

/** @addtogroup IRDA_Exported_Functions_Group3
  * @{
  */
/* Peripheral State functions  **************************************************/
HAL_IRDA_StateTypeDef HAL_IRDA_GetState(IRDA_HandleTypeDef *hirda);
uint32_t HAL_IRDA_GetError(IRDA_HandleTypeDef *hirda);
/**
  * @}
  */

/**
  * @}
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup IRDA_Private_Constants IRDA Private Constants
  * @{
  */

/** @brief IRDA interruptions flag mask
  *
  */
#define IRDA_IT_MASK  ((uint32_t) USART_CR1_PEIE | USART_CR1_TXEIE | USART_CR1_TCIE | USART_CR1_RXNEIE | \
                                  USART_CR1_IDLEIE | USART_CR2_LBDIE | USART_CR3_CTSIE | USART_CR3_EIE )

#define IRDA_CR1_REG_INDEX         1U
#define IRDA_CR2_REG_INDEX         2U
#define IRDA_CR3_REG_INDEX         3U
/**
  * @}
  */

/* Private macros --------------------------------------------------------*/
/** @defgroup IRDA_Private_Macros   IRDA Private Macros
  * @{
  */
#define IS_IRDA_WORD_LENGTH(LENGTH)   (((LENGTH) == IRDA_WORDLENGTH_8B) || \
                                       ((LENGTH) == IRDA_WORDLENGTH_9B))

#define IS_IRDA_PARITY(PARITY)        (((PARITY) == IRDA_PARITY_NONE) || \
                                       ((PARITY) == IRDA_PARITY_EVEN) || \
                                       ((PARITY) == IRDA_PARITY_ODD))

#define IS_IRDA_MODE(MODE)            ((((MODE) & 0x0000FFF3U) == 0x00U) && ((MODE) != 0x00000000U))

#define IS_IRDA_POWERMODE(MODE)       (((MODE) == IRDA_POWERMODE_LOWPOWER) || \
                                       ((MODE) == IRDA_POWERMODE_NORMAL))

#define IS_IRDA_BAUDRATE(BAUDRATE)    ((BAUDRATE) < 115201U)

#define IRDA_DIV(_PCLK_, _BAUD_)      ((uint32_t)((((uint64_t)(_PCLK_))*25U)/(4U*(((uint64_t)(_BAUD_))))))

#define IRDA_DIVMANT(_PCLK_, _BAUD_)  (IRDA_DIV((_PCLK_), (_BAUD_))/100U)

#define IRDA_DIVFRAQ(_PCLK_, _BAUD_)  ((((IRDA_DIV((_PCLK_), (_BAUD_)) - (IRDA_DIVMANT((_PCLK_), (_BAUD_)) * 100U)) * 16U) + 50U) / 100U)

/* UART BRR = mantissa + overflow + fraction
            = (UART DIVMANT << 4) + (UART DIVFRAQ & 0xF0) + (UART DIVFRAQ & 0x0FU) */
#define IRDA_BRR(_PCLK_, _BAUD_)      (((IRDA_DIVMANT((_PCLK_), (_BAUD_)) << 4U) + \
                                        (IRDA_DIVFRAQ((_PCLK_), (_BAUD_)) & 0xF0U)) + \
                                        (IRDA_DIVFRAQ((_PCLK_), (_BAUD_)) & 0x0FU))

/**
  * @}
  */

/* Private functions ---------------------------------------------------------*/
/** @defgroup IRDA_Private_Functions IRDA Private Functions
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

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_IRDA_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
