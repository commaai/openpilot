/**
  ******************************************************************************
  * @file    stm32f4xx_hal_hash.h
  * @author  MCD Application Team
  * @brief   Header file of HASH HAL module.
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
#ifndef STM32F4xx_HAL_HASH_H
#define STM32F4xx_HAL_HASH_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */
#if defined (HASH)
/** @addtogroup HASH
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup HASH_Exported_Types HASH Exported Types
  * @{
  */

/**
  * @brief  HASH Configuration Structure definition
  */
typedef struct
{
  uint32_t DataType;    /*!< 32-bit data, 16-bit data, 8-bit data or 1-bit data.
                              This parameter can be a value of @ref HASH_Data_Type. */

  uint32_t KeySize;     /*!< The key size is used only in HMAC operation. */

  uint8_t *pKey;        /*!< The key is used only in HMAC operation. */

} HASH_InitTypeDef;

/**
  * @brief HAL State structures definition
  */
typedef enum
{
  HAL_HASH_STATE_RESET             = 0x00U,    /*!< Peripheral is not initialized            */
  HAL_HASH_STATE_READY             = 0x01U,    /*!< Peripheral Initialized and ready for use */
  HAL_HASH_STATE_BUSY              = 0x02U,    /*!< Processing (hashing) is ongoing          */
  HAL_HASH_STATE_TIMEOUT           = 0x06U,    /*!< Timeout state                            */
  HAL_HASH_STATE_ERROR             = 0x07U,    /*!< Error state                              */
  HAL_HASH_STATE_SUSPENDED         = 0x08U     /*!< Suspended state                          */
} HAL_HASH_StateTypeDef;

/**
  * @brief HAL phase structures definition
  */
typedef enum
{
  HAL_HASH_PHASE_READY             = 0x01U,    /*!< HASH peripheral is ready to start                    */
  HAL_HASH_PHASE_PROCESS           = 0x02U,    /*!< HASH peripheral is in HASH processing phase          */
  HAL_HASH_PHASE_HMAC_STEP_1       = 0x03U,    /*!< HASH peripheral is in HMAC step 1 processing phase
                                              (step 1 consists in entering the inner hash function key) */
  HAL_HASH_PHASE_HMAC_STEP_2       = 0x04U,    /*!< HASH peripheral is in HMAC step 2 processing phase
                                              (step 2 consists in entering the message text) */
  HAL_HASH_PHASE_HMAC_STEP_3       = 0x05U     /*!< HASH peripheral is in HMAC step 3 processing phase
                                              (step 3 consists in entering the outer hash function key) */
} HAL_HASH_PhaseTypeDef;

/**
  * @brief HAL HASH mode suspend definitions
  */
typedef enum
{
  HAL_HASH_SUSPEND_NONE            = 0x00U,    /*!< HASH peripheral suspension not requested */
  HAL_HASH_SUSPEND                 = 0x01U     /*!< HASH peripheral suspension is requested  */
} HAL_HASH_SuspendTypeDef;

#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1U)
/**
  * @brief  HAL HASH common Callback ID enumeration definition
  */
typedef enum
{
  HAL_HASH_MSPINIT_CB_ID           = 0x00U,    /*!< HASH MspInit callback ID     */
  HAL_HASH_MSPDEINIT_CB_ID         = 0x01U,    /*!< HASH MspDeInit callback ID   */
  HAL_HASH_INPUTCPLT_CB_ID         = 0x02U,    /*!< HASH input completion callback ID */
  HAL_HASH_DGSTCPLT_CB_ID          = 0x03U,    /*!< HASH digest computation completion callback ID */
  HAL_HASH_ERROR_CB_ID             = 0x04U,    /*!< HASH error callback ID     */
} HAL_HASH_CallbackIDTypeDef;
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */


/**
  * @brief  HASH Handle Structure definition
  */
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
typedef struct __HASH_HandleTypeDef
#else
typedef struct
#endif /* (USE_HAL_HASH_REGISTER_CALLBACKS) */
{
  HASH_InitTypeDef           Init;             /*!< HASH required parameters */

  uint8_t                    *pHashInBuffPtr;  /*!< Pointer to input buffer */

  uint8_t                    *pHashOutBuffPtr; /*!< Pointer to output buffer (digest) */

  uint8_t                    *pHashKeyBuffPtr; /*!< Pointer to key buffer (HMAC only) */

  uint8_t                    *pHashMsgBuffPtr; /*!< Pointer to message buffer (HMAC only) */

  uint32_t                   HashBuffSize;     /*!< Size of buffer to be processed */

  __IO uint32_t              HashInCount;      /*!< Counter of inputted data */

  __IO uint32_t              HashITCounter;    /*!< Counter of issued interrupts */

  __IO uint32_t              HashKeyCount;     /*!< Counter for Key inputted data (HMAC only) */

  HAL_StatusTypeDef          Status;           /*!< HASH peripheral status   */

  HAL_HASH_PhaseTypeDef      Phase;            /*!< HASH peripheral phase   */

  DMA_HandleTypeDef          *hdmain;          /*!< HASH In DMA Handle parameters */

  HAL_LockTypeDef            Lock;             /*!< Locking object */

  __IO HAL_HASH_StateTypeDef State;            /*!< HASH peripheral state */

  HAL_HASH_SuspendTypeDef    SuspendRequest;   /*!< HASH peripheral suspension request flag */

  FlagStatus                 DigestCalculationDisable;  /*!< Digest calculation phase skip (MDMAT bit control) for multi-buffers DMA-based HMAC computation */

  __IO uint32_t              NbWordsAlreadyPushed;      /*!< Numbers of words already pushed in FIFO before inputting new block */

  __IO  uint32_t             ErrorCode;        /*!< HASH Error code */

  __IO  uint32_t             Accumulation;     /*!< HASH multi buffers accumulation flag */

#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
  void (* InCpltCallback)(struct __HASH_HandleTypeDef *hhash);         /*!< HASH input completion callback */

  void (* DgstCpltCallback)(struct __HASH_HandleTypeDef *hhash);       /*!< HASH digest computation completion callback */

  void (* ErrorCallback)(struct __HASH_HandleTypeDef *hhash);          /*!< HASH error callback */

  void (* MspInitCallback)(struct __HASH_HandleTypeDef *hhash);        /*!< HASH Msp Init callback */

  void (* MspDeInitCallback)(struct __HASH_HandleTypeDef *hhash);      /*!< HASH Msp DeInit callback */

#endif /* (USE_HAL_HASH_REGISTER_CALLBACKS) */
} HASH_HandleTypeDef;

#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1U)
/**
  * @brief  HAL HASH Callback pointer definition
  */
typedef  void (*pHASH_CallbackTypeDef)(HASH_HandleTypeDef *hhash);  /*!< pointer to a HASH common callback functions */
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/

/** @defgroup HASH_Exported_Constants  HASH Exported Constants
  * @{
  */

/** @defgroup HASH_Algo_Selection   HASH algorithm selection
  * @{
  */
#define HASH_ALGOSELECTION_SHA1      0x00000000U /*!< HASH function is SHA1   */
#define HASH_ALGOSELECTION_MD5       HASH_CR_ALGO_0     /*!< HASH function is MD5    */
#define HASH_ALGOSELECTION_SHA224    HASH_CR_ALGO_1     /*!< HASH function is SHA224 */
#define HASH_ALGOSELECTION_SHA256    HASH_CR_ALGO       /*!< HASH function is SHA256 */
/**
  * @}
  */

/** @defgroup HASH_Algorithm_Mode   HASH algorithm mode
  * @{
  */
#define HASH_ALGOMODE_HASH         0x00000000U /*!< Algorithm is HASH */
#define HASH_ALGOMODE_HMAC         HASH_CR_MODE           /*!< Algorithm is HMAC */
/**
  * @}
  */

/** @defgroup HASH_Data_Type      HASH input data type
  * @{
  */
#define HASH_DATATYPE_32B          0x00000000U /*!< 32-bit data. No swapping                     */
#define HASH_DATATYPE_16B          HASH_CR_DATATYPE_0 /*!< 16-bit data. Each half word is swapped       */
#define HASH_DATATYPE_8B           HASH_CR_DATATYPE_1 /*!< 8-bit data. All bytes are swapped            */
#define HASH_DATATYPE_1B           HASH_CR_DATATYPE   /*!< 1-bit data. In the word all bits are swapped */
/**
  * @}
  */

/** @defgroup HASH_HMAC_Long_key_only_for_HMAC_mode   HMAC key length type
  * @{
  */
#define HASH_HMAC_KEYTYPE_SHORTKEY      0x00000000U /*!< HMAC Key size is <= 64 bytes */
#define HASH_HMAC_KEYTYPE_LONGKEY       HASH_CR_LKEY           /*!< HMAC Key size is > 64 bytes  */
/**
  * @}
  */

/** @defgroup HASH_flags_definition  HASH flags definitions
  * @{
  */
#define HASH_FLAG_DINIS            HASH_SR_DINIS  /*!< 16 locations are free in the DIN : a new block can be entered in the Peripheral */
#define HASH_FLAG_DCIS             HASH_SR_DCIS   /*!< Digest calculation complete                                                     */
#define HASH_FLAG_DMAS             HASH_SR_DMAS   /*!< DMA interface is enabled (DMAE=1) or a transfer is ongoing                      */
#define HASH_FLAG_BUSY             HASH_SR_BUSY   /*!< The hash core is Busy, processing a block of data                               */
#define HASH_FLAG_DINNE            HASH_CR_DINNE  /*!< DIN not empty : the input buffer contains at least one word of data             */

/**
  * @}
  */

/** @defgroup HASH_interrupts_definition   HASH interrupts definitions
  * @{
  */
#define HASH_IT_DINI               HASH_IMR_DINIE  /*!< A new block can be entered into the input buffer (DIN) */
#define HASH_IT_DCI                HASH_IMR_DCIE   /*!< Digest calculation complete                            */

/**
  * @}
  */

/** @defgroup HASH_Error_Definition   HASH Error Definition
  * @{
  */
#define  HAL_HASH_ERROR_NONE             0x00000000U   /*!< No error                */
#define  HAL_HASH_ERROR_IT               0x00000001U   /*!< IT-based process error  */
#define  HAL_HASH_ERROR_DMA              0x00000002U   /*!< DMA-based process error */
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1U)
#define  HAL_HASH_ERROR_INVALID_CALLBACK 0x00000004U   /*!< Invalid Callback error  */
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */
/**
  * @}
  */

/**
  * @}
  */

/* Exported macros -----------------------------------------------------------*/
/** @defgroup HASH_Exported_Macros HASH Exported Macros
  * @{
  */

/** @brief  Check whether or not the specified HASH flag is set.
  * @param  __FLAG__ specifies the flag to check.
  *        This parameter can be one of the following values:
  *            @arg @ref HASH_FLAG_DINIS A new block can be entered into the input buffer.
  *            @arg @ref HASH_FLAG_DCIS Digest calculation complete.
  *            @arg @ref HASH_FLAG_DMAS DMA interface is enabled (DMAE=1) or a transfer is ongoing.
  *            @arg @ref HASH_FLAG_BUSY The hash core is Busy : processing a block of data.
  *            @arg @ref HASH_FLAG_DINNE DIN not empty : the input buffer contains at least one word of data.
  * @retval The new state of __FLAG__ (TRUE or FALSE).
  */
#define __HAL_HASH_GET_FLAG(__FLAG__)  (((__FLAG__) > 8U)  ?                    \
                                        ((HASH->CR & (__FLAG__)) == (__FLAG__)) :\
                                        ((HASH->SR & (__FLAG__)) == (__FLAG__)) )


/** @brief  Clear the specified HASH flag.
  * @param  __FLAG__ specifies the flag to clear.
  *        This parameter can be one of the following values:
  *            @arg @ref HASH_FLAG_DINIS A new block can be entered into the input buffer.
  *            @arg @ref HASH_FLAG_DCIS Digest calculation complete
  * @retval None
  */
#define __HAL_HASH_CLEAR_FLAG(__FLAG__) CLEAR_BIT(HASH->SR, (__FLAG__))


/** @brief  Enable the specified HASH interrupt.
  * @param  __INTERRUPT__ specifies the HASH interrupt source to enable.
  *          This parameter can be one of the following values:
  *            @arg @ref HASH_IT_DINI  A new block can be entered into the input buffer (DIN)
  *            @arg @ref HASH_IT_DCI   Digest calculation complete
  * @retval None
  */
#define __HAL_HASH_ENABLE_IT(__INTERRUPT__)   SET_BIT(HASH->IMR, (__INTERRUPT__))

/** @brief  Disable the specified HASH interrupt.
  * @param  __INTERRUPT__ specifies the HASH interrupt source to disable.
  *          This parameter can be one of the following values:
  *            @arg @ref HASH_IT_DINI  A new block can be entered into the input buffer (DIN)
  *            @arg @ref HASH_IT_DCI   Digest calculation complete
  * @retval None
  */
#define __HAL_HASH_DISABLE_IT(__INTERRUPT__)   CLEAR_BIT(HASH->IMR, (__INTERRUPT__))

/** @brief Reset HASH handle state.
  * @param  __HANDLE__ HASH handle.
  * @retval None
  */

#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
#define __HAL_HASH_RESET_HANDLE_STATE(__HANDLE__) do{\
                                                      (__HANDLE__)->State = HAL_HASH_STATE_RESET;\
                                                      (__HANDLE__)->MspInitCallback = NULL;      \
                                                      (__HANDLE__)->MspDeInitCallback = NULL;    \
                                                     }while(0)
#else
#define __HAL_HASH_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_HASH_STATE_RESET)
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */


/** @brief Reset HASH handle status.
  * @param  __HANDLE__ HASH handle.
  * @retval None
  */
#define __HAL_HASH_RESET_HANDLE_STATUS(__HANDLE__) ((__HANDLE__)->Status = HAL_OK)

/**
  * @brief  Enable the multi-buffer DMA transfer mode.
  * @note   This bit is set when hashing large files when multiple DMA transfers are needed.
  * @retval None
  */
#define __HAL_HASH_SET_MDMAT()          SET_BIT(HASH->CR, HASH_CR_MDMAT)

/**
  * @brief  Disable the multi-buffer DMA transfer mode.
  * @retval None
  */
#define __HAL_HASH_RESET_MDMAT()        CLEAR_BIT(HASH->CR, HASH_CR_MDMAT)


/**
  * @brief Start the digest computation.
  * @retval None
  */
#define __HAL_HASH_START_DIGEST()       SET_BIT(HASH->STR, HASH_STR_DCAL)

/**
  * @brief Set the number of valid bits in the last word written in data register DIN.
  * @param  __SIZE__ size in bytes of last data written in Data register.
  * @retval None
  */
#define  __HAL_HASH_SET_NBVALIDBITS(__SIZE__)    MODIFY_REG(HASH->STR, HASH_STR_NBLW, 8U * ((__SIZE__) % 4U))

/**
  * @brief Reset the HASH core.
  * @retval None
  */
#define __HAL_HASH_INIT()       SET_BIT(HASH->CR, HASH_CR_INIT)

/**
  * @}
  */


/* Private macros --------------------------------------------------------*/
/** @defgroup HASH_Private_Macros   HASH Private Macros
  * @{
  */
/**
  * @brief  Return digest length in bytes.
  * @retval Digest length
  */
#if defined(HASH_CR_MDMAT)
#define HASH_DIGEST_LENGTH() ((READ_BIT(HASH->CR, HASH_CR_ALGO) == HASH_ALGOSELECTION_SHA1)   ?  20U : \
                              ((READ_BIT(HASH->CR, HASH_CR_ALGO) == HASH_ALGOSELECTION_SHA224) ?  28U : \
                               ((READ_BIT(HASH->CR, HASH_CR_ALGO) == HASH_ALGOSELECTION_SHA256) ?  32U : 16U ) ) )
#else
#define HASH_DIGEST_LENGTH() ((READ_BIT(HASH->CR, HASH_CR_ALGO) == HASH_ALGOSELECTION_SHA1)   ?  20U : 16)
#endif  /* HASH_CR_MDMAT*/
/**
  * @brief  Return number of words already pushed in the FIFO.
  * @retval Number of words already pushed in the FIFO
  */
#define HASH_NBW_PUSHED() ((READ_BIT(HASH->CR, HASH_CR_NBW)) >> 8U)

/**
  * @brief Ensure that HASH input data type is valid.
  * @param __DATATYPE__ HASH input data type.
  * @retval SET (__DATATYPE__ is valid) or RESET (__DATATYPE__ is invalid)
  */
#define IS_HASH_DATATYPE(__DATATYPE__) (((__DATATYPE__) == HASH_DATATYPE_32B)|| \
                                        ((__DATATYPE__) == HASH_DATATYPE_16B)|| \
                                        ((__DATATYPE__) == HASH_DATATYPE_8B) || \
                                        ((__DATATYPE__) == HASH_DATATYPE_1B))

/**
  * @brief Ensure that input data buffer size is valid for multi-buffer HASH
  *        processing in DMA mode.
  * @note  This check is valid only for multi-buffer HASH processing in DMA mode.
  * @param __SIZE__ input data buffer size.
  * @retval SET (__SIZE__ is valid) or RESET (__SIZE__ is invalid)
  */
#define IS_HASH_DMA_MULTIBUFFER_SIZE(__SIZE__)  ((READ_BIT(HASH->CR, HASH_CR_MDMAT) == 0U) || (((__SIZE__) % 4U) == 0U))

/**
  * @brief Ensure that input data buffer size is valid for multi-buffer HMAC
  *        processing in DMA mode.
  * @note  This check is valid only for multi-buffer HMAC processing in DMA mode.
  * @param __HANDLE__ HASH handle.
  * @param __SIZE__ input data buffer size.
  * @retval SET (__SIZE__ is valid) or RESET (__SIZE__ is invalid)
  */
#define IS_HMAC_DMA_MULTIBUFFER_SIZE(__HANDLE__,__SIZE__)  ((((__HANDLE__)->DigestCalculationDisable) == RESET)\
                                                            || (((__SIZE__) % 4U) == 0U))
/**
  * @brief Ensure that handle phase is set to HASH processing.
  * @param __HANDLE__ HASH handle.
  * @retval SET (handle phase is set to HASH processing) or RESET (handle phase is not set to HASH processing)
  */
#define IS_HASH_PROCESSING(__HANDLE__)  ((__HANDLE__)->Phase == HAL_HASH_PHASE_PROCESS)

/**
  * @brief Ensure that handle phase is set to HMAC processing.
  * @param __HANDLE__ HASH handle.
  * @retval SET (handle phase is set to HMAC processing) or RESET (handle phase is not set to HMAC processing)
  */
#define IS_HMAC_PROCESSING(__HANDLE__)  (((__HANDLE__)->Phase == HAL_HASH_PHASE_HMAC_STEP_1) || \
                                         ((__HANDLE__)->Phase == HAL_HASH_PHASE_HMAC_STEP_2) || \
                                         ((__HANDLE__)->Phase == HAL_HASH_PHASE_HMAC_STEP_3))

/**
  * @}
  */

/* Include HASH HAL Extended module */
#include "stm32f4xx_hal_hash_ex.h"
/* Exported functions --------------------------------------------------------*/

/** @addtogroup HASH_Exported_Functions HASH Exported Functions
  * @{
  */

/** @addtogroup HASH_Exported_Functions_Group1 Initialization and de-initialization functions
  * @{
  */

/* Initialization/de-initialization methods  **********************************/
HAL_StatusTypeDef HAL_HASH_Init(HASH_HandleTypeDef *hhash);
HAL_StatusTypeDef HAL_HASH_DeInit(HASH_HandleTypeDef *hhash);
void HAL_HASH_MspInit(HASH_HandleTypeDef *hhash);
void HAL_HASH_MspDeInit(HASH_HandleTypeDef *hhash);
void HAL_HASH_InCpltCallback(HASH_HandleTypeDef *hhash);
void HAL_HASH_DgstCpltCallback(HASH_HandleTypeDef *hhash);
void HAL_HASH_ErrorCallback(HASH_HandleTypeDef *hhash);
/* Callbacks Register/UnRegister functions  ***********************************/
#if (USE_HAL_HASH_REGISTER_CALLBACKS == 1)
HAL_StatusTypeDef HAL_HASH_RegisterCallback(HASH_HandleTypeDef *hhash, HAL_HASH_CallbackIDTypeDef CallbackID,
                                            pHASH_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_HASH_UnRegisterCallback(HASH_HandleTypeDef *hhash, HAL_HASH_CallbackIDTypeDef CallbackID);
#endif /* USE_HAL_HASH_REGISTER_CALLBACKS */


/**
  * @}
  */

/** @addtogroup HASH_Exported_Functions_Group2 HASH processing functions in polling mode
  * @{
  */


/* HASH processing using polling  *********************************************/
HAL_StatusTypeDef HAL_HASH_SHA1_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                      uint32_t Timeout);
HAL_StatusTypeDef HAL_HASH_MD5_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                     uint32_t Timeout);
HAL_StatusTypeDef HAL_HASH_MD5_Accmlt(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size);
HAL_StatusTypeDef HAL_HASH_SHA1_Accmlt(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size);
HAL_StatusTypeDef HAL_HASH_MD5_Accmlt_End(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                          uint8_t *pOutBuffer, uint32_t Timeout);
HAL_StatusTypeDef HAL_HASH_SHA1_Accmlt_End(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                           uint8_t *pOutBuffer, uint32_t Timeout);


/**
  * @}
  */

/** @addtogroup HASH_Exported_Functions_Group3 HASH processing functions in interrupt mode
  * @{
  */

/* HASH processing using IT  **************************************************/
HAL_StatusTypeDef HAL_HASH_SHA1_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                         uint8_t *pOutBuffer);
HAL_StatusTypeDef HAL_HASH_SHA1_Accmlt_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size);
HAL_StatusTypeDef HAL_HASH_SHA1_Accmlt_End_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                              uint8_t *pOutBuffer);
HAL_StatusTypeDef HAL_HASH_MD5_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                        uint8_t *pOutBuffer);
HAL_StatusTypeDef HAL_HASH_MD5_Accmlt_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size);
HAL_StatusTypeDef HAL_HASH_MD5_Accmlt_End_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                             uint8_t *pOutBuffer);
void HAL_HASH_IRQHandler(HASH_HandleTypeDef *hhash);
/**
  * @}
  */

/** @addtogroup HASH_Exported_Functions_Group4 HASH processing functions in DMA mode
  * @{
  */

/* HASH processing using DMA  *************************************************/
HAL_StatusTypeDef HAL_HASH_SHA1_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size);
HAL_StatusTypeDef HAL_HASH_SHA1_Finish(HASH_HandleTypeDef *hhash, uint8_t *pOutBuffer, uint32_t Timeout);
HAL_StatusTypeDef HAL_HASH_MD5_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size);
HAL_StatusTypeDef HAL_HASH_MD5_Finish(HASH_HandleTypeDef *hhash, uint8_t *pOutBuffer, uint32_t Timeout);

/**
  * @}
  */

/** @addtogroup HASH_Exported_Functions_Group5 HMAC processing functions in polling mode
  * @{
  */

/* HASH-MAC processing using polling  *****************************************/
HAL_StatusTypeDef HAL_HMAC_SHA1_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                      uint32_t Timeout);
HAL_StatusTypeDef HAL_HMAC_MD5_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                     uint32_t Timeout);

/**
  * @}
  */

/** @addtogroup HASH_Exported_Functions_Group6 HMAC processing functions in interrupt mode
  * @{
  */

HAL_StatusTypeDef HAL_HMAC_MD5_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                        uint8_t *pOutBuffer);
HAL_StatusTypeDef HAL_HMAC_SHA1_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size,
                                         uint8_t *pOutBuffer);

/**
  * @}
  */

/** @addtogroup HASH_Exported_Functions_Group7 HMAC processing functions in DMA mode
  * @{
  */

/* HASH-HMAC processing using DMA  ********************************************/
HAL_StatusTypeDef HAL_HMAC_SHA1_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size);
HAL_StatusTypeDef HAL_HMAC_MD5_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size);

/**
  * @}
  */

/** @addtogroup HASH_Exported_Functions_Group8 Peripheral states functions
  * @{
  */


/* Peripheral State methods  **************************************************/
HAL_HASH_StateTypeDef HAL_HASH_GetState(HASH_HandleTypeDef *hhash);
HAL_StatusTypeDef HAL_HASH_GetStatus(HASH_HandleTypeDef *hhash);
void HAL_HASH_ContextSaving(HASH_HandleTypeDef *hhash, uint8_t *pMemBuffer);
void HAL_HASH_ContextRestoring(HASH_HandleTypeDef *hhash, uint8_t *pMemBuffer);
void HAL_HASH_SwFeed_ProcessSuspend(HASH_HandleTypeDef *hhash);
HAL_StatusTypeDef HAL_HASH_DMAFeed_ProcessSuspend(HASH_HandleTypeDef *hhash);
uint32_t HAL_HASH_GetError(HASH_HandleTypeDef *hhash);

/**
  * @}
  */

/**
  * @}
  */

/* Private functions -----------------------------------------------------------*/

/** @addtogroup HASH_Private_Functions HASH Private Functions
  * @{
  */

/* Private functions */
HAL_StatusTypeDef HASH_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                             uint32_t Timeout, uint32_t Algorithm);
HAL_StatusTypeDef HASH_Accumulate(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint32_t Algorithm);
HAL_StatusTypeDef HASH_Accumulate_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint32_t Algorithm);
HAL_StatusTypeDef HASH_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                uint32_t Algorithm);
HAL_StatusTypeDef HASH_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint32_t Algorithm);
HAL_StatusTypeDef HASH_Finish(HASH_HandleTypeDef *hhash, uint8_t *pOutBuffer, uint32_t Timeout);
HAL_StatusTypeDef HMAC_Start(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                             uint32_t Timeout, uint32_t Algorithm);
HAL_StatusTypeDef HMAC_Start_IT(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint8_t *pOutBuffer,
                                uint32_t Algorithm);
HAL_StatusTypeDef HMAC_Start_DMA(HASH_HandleTypeDef *hhash, uint8_t *pInBuffer, uint32_t Size, uint32_t Algorithm);

/**
  * @}
  */

/**
  * @}
  */
#endif /*  HASH*/
/**
  * @}
  */


#ifdef __cplusplus
}
#endif


#endif /* STM32F4xx_HAL_HASH_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
