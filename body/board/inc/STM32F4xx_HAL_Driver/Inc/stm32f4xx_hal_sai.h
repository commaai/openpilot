/**
  ******************************************************************************
  * @file    stm32f4xx_hal_sai.h
  * @author  MCD Application Team
  * @brief   Header file of SAI HAL module.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2017 STMicroelectronics.
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
#ifndef __STM32F4xx_HAL_SAI_H
#define __STM32F4xx_HAL_SAI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */
#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) || \
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F413xx) || \
    defined(STM32F423xx)

/** @addtogroup SAI
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup SAI_Exported_Types SAI Exported Types
  * @{
  */

/**
  * @brief  HAL State structures definition
  */
typedef enum
{
  HAL_SAI_STATE_RESET      = 0x00U,  /*!< SAI not yet initialized or disabled                */
  HAL_SAI_STATE_READY      = 0x01U,  /*!< SAI initialized and ready for use                  */
  HAL_SAI_STATE_BUSY       = 0x02U,  /*!< SAI internal process is ongoing                    */
  HAL_SAI_STATE_BUSY_TX    = 0x12U,  /*!< Data transmission process is ongoing               */
  HAL_SAI_STATE_BUSY_RX    = 0x22U,  /*!< Data reception process is ongoing                  */
  HAL_SAI_STATE_TIMEOUT    = 0x03U,  /*!< SAI timeout state                                  */
  HAL_SAI_STATE_ERROR      = 0x04U   /*!< SAI error state                                    */
} HAL_SAI_StateTypeDef;

/**
  * @brief  SAI Callback prototype
  */
typedef void (*SAIcallback)(void);

/** @defgroup SAI_Init_Structure_definition SAI Init Structure definition
  * @brief  SAI Init Structure definition
  * @{
  */
typedef struct
{
  uint32_t AudioMode;           /*!< Specifies the SAI Block audio Mode.
                                     This parameter can be a value of @ref SAI_Block_Mode                      */

  uint32_t Synchro;             /*!< Specifies SAI Block synchronization
                                     This parameter can be a value of @ref SAI_Block_Synchronization           */

  uint32_t SynchroExt;          /*!< Specifies SAI external output synchronization, this setup is common
                                     for BlockA and BlockB
                                     This parameter can be a value of @ref SAI_Block_SyncExt
                                     @note: If both audio blocks of same SAI are used, this parameter has
                                            to be set to the same value for each audio block                   */

  uint32_t OutputDrive;         /*!< Specifies when SAI Block outputs are driven.
                                     This parameter can be a value of @ref SAI_Block_Output_Drive
                                     @note this value has to be set before enabling the audio block
                                     but after the audio block configuration.                                  */

  uint32_t NoDivider;           /*!< Specifies whether master clock will be divided or not.
                                     This parameter can be a value of @ref SAI_Block_NoDivider
                                     @note  If bit NODIV in the SAI_xCR1 register is cleared, the frame length
                                            should be aligned to a number equal to a power of 2, from 8 to 256.
                                            If bit NODIV in the SAI_xCR1 register is set, the frame length can
                                            take any of the values without constraint since the input clock of
                                            the audio block should be equal to the bit clock.
                                             There is no MCLK_x clock which can be output.                     */

  uint32_t FIFOThreshold;       /*!< Specifies SAI Block FIFO threshold.
                                     This parameter can be a value of @ref SAI_Block_Fifo_Threshold            */

  uint32_t ClockSource;         /*!< Specifies the SAI Block x Clock source.
                                     This parameter is not used for STM32F446xx devices.                       */

  uint32_t AudioFrequency;      /*!< Specifies the audio frequency sampling.
                                     This parameter can be a value of @ref SAI_Audio_Frequency                 */

  uint32_t Mckdiv;              /*!< Specifies the master clock divider.
                                     This parameter must be a number between Min_Data = 0 and Max_Data = 15.
                                     @note This parameter is used only if AudioFrequency is set to
                                           SAI_AUDIO_FREQUENCY_MCKDIV otherwise it is internally computed. */

  uint32_t MonoStereoMode;      /*!< Specifies if the mono or stereo mode is selected.
                                     This parameter can be a value of @ref SAI_Mono_Stereo_Mode                */

  uint32_t CompandingMode;      /*!< Specifies the companding mode type.
                                     This parameter can be a value of @ref SAI_Block_Companding_Mode           */

  uint32_t TriState;            /*!< Specifies the companding mode type.
                                     This parameter can be a value of @ref SAI_TRIState_Management             */

  /* This part of the structure is automatically filled if your are using the high level intialisation
     function HAL_SAI_InitProtocol                                                                             */

  uint32_t Protocol;        /*!< Specifies the SAI Block protocol.
                                 This parameter can be a value of @ref SAI_Block_Protocol                      */

  uint32_t DataSize;        /*!< Specifies the SAI Block data size.
                                 This parameter can be a value of @ref SAI_Block_Data_Size                     */

  uint32_t FirstBit;        /*!< Specifies whether data transfers start from MSB or LSB bit.
                                 This parameter can be a value of @ref SAI_Block_MSB_LSB_transmission          */

  uint32_t ClockStrobing;   /*!< Specifies the SAI Block clock strobing edge sensitivity.
                                 This parameter can be a value of @ref SAI_Block_Clock_Strobing                */
} SAI_InitTypeDef;
/**
  * @}
  */

/** @defgroup SAI_Frame_Structure_definition SAI Frame Structure definition
  * @brief  SAI Frame Init structure definition
  * @note   For SPDIF and AC97 protocol, these parameters are not used (set by hardware).
  * @{
  */
typedef struct
{
  uint32_t FrameLength;         /*!< Specifies the Frame length, the number of SCK clocks for each audio frame.
                                     This parameter must be a number between Min_Data = 8 and Max_Data = 256.
                                     @note  If master clock MCLK_x pin is declared as an output, the frame length
                                            should be aligned to a number equal to power of 2 in order to keep
                                            in an audio frame, an integer number of MCLK pulses by bit Clock. */

  uint32_t ActiveFrameLength;  /*!< Specifies the Frame synchronization active level length.
                                    This Parameter specifies the length in number of bit clock (SCK + 1)
                                    of the active level of FS signal in audio frame.
                                    This parameter must be a number between Min_Data = 1 and Max_Data = 128   */

  uint32_t FSDefinition;       /*!< Specifies the Frame synchronization definition.
                                    This parameter can be a value of @ref SAI_Block_FS_Definition             */

  uint32_t FSPolarity;         /*!< Specifies the Frame synchronization Polarity.
                                    This parameter can be a value of @ref SAI_Block_FS_Polarity               */

  uint32_t FSOffset;           /*!< Specifies the Frame synchronization Offset.
                                    This parameter can be a value of @ref SAI_Block_FS_Offset                 */
} SAI_FrameInitTypeDef;
/**
  * @}
  */

/** @defgroup SAI_Slot_Structure_definition SAI Slot Structure definition
  * @brief   SAI Block Slot Init Structure definition
  * @note    For SPDIF protocol, these parameters are not used (set by hardware).
  * @note    For AC97 protocol, only SlotActive parameter is used (the others are set by hardware).
  * @{
  */
typedef struct
{
  uint32_t FirstBitOffset;  /*!< Specifies the position of first data transfer bit in the slot.
                                 This parameter must be a number between Min_Data = 0 and Max_Data = 24 */

  uint32_t SlotSize;        /*!< Specifies the Slot Size.
                                 This parameter can be a value of @ref SAI_Block_Slot_Size              */

  uint32_t SlotNumber;      /*!< Specifies the number of slot in the audio frame.
                                 This parameter must be a number between Min_Data = 1 and Max_Data = 16 */

  uint32_t SlotActive;      /*!< Specifies the slots in audio frame that will be activated.
                                 This parameter can be a value of @ref SAI_Block_Slot_Active            */
} SAI_SlotInitTypeDef;
/**
  * @}
  */

/** @defgroup SAI_Handle_Structure_definition SAI Handle Structure definition
  * @brief  SAI handle Structure definition
  * @{
  */
typedef struct __SAI_HandleTypeDef
{
  SAI_Block_TypeDef         *Instance;  /*!< SAI Blockx registers base address        */

  SAI_InitTypeDef           Init;       /*!< SAI communication parameters             */

  SAI_FrameInitTypeDef      FrameInit;  /*!< SAI Frame configuration parameters       */

  SAI_SlotInitTypeDef       SlotInit;   /*!< SAI Slot configuration parameters        */

  uint8_t                  *pBuffPtr;  /*!< Pointer to SAI transfer Buffer            */

  uint16_t                  XferSize;  /*!< SAI transfer size                         */

  uint16_t                  XferCount; /*!< SAI transfer counter                      */

  DMA_HandleTypeDef         *hdmatx;     /*!< SAI Tx DMA handle parameters            */

  DMA_HandleTypeDef         *hdmarx;     /*!< SAI Rx DMA handle parameters            */

  SAIcallback               mutecallback;/*!< SAI mute callback                       */

  void (*InterruptServiceRoutine)(struct __SAI_HandleTypeDef *hsai); /* function pointer for IRQ handler   */

  HAL_LockTypeDef           Lock;        /*!< SAI locking object                      */

  __IO HAL_SAI_StateTypeDef State;       /*!< SAI communication state                 */

  __IO uint32_t             ErrorCode;    /*!< SAI Error code */

#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
  void (*RxCpltCallback)(struct __SAI_HandleTypeDef *hsai);      /*!< SAI receive complete callback */
  void (*RxHalfCpltCallback)(struct __SAI_HandleTypeDef *hsai);  /*!< SAI receive half complete callback */
  void (*TxCpltCallback)(struct __SAI_HandleTypeDef *hsai);      /*!< SAI transmit complete callback */
  void (*TxHalfCpltCallback)(struct __SAI_HandleTypeDef *hsai);  /*!< SAI transmit half complete callback */
  void (*ErrorCallback)(struct __SAI_HandleTypeDef *hsai);       /*!< SAI error callback */
  void (*MspInitCallback)(struct __SAI_HandleTypeDef *hsai);     /*!< SAI MSP init callback */
  void (*MspDeInitCallback)(struct __SAI_HandleTypeDef *hsai);   /*!< SAI MSP de-init callback */
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
} SAI_HandleTypeDef;
/**
  * @}
  */

#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
/**
  * @brief  SAI callback ID enumeration definition
  */
typedef enum
{
  HAL_SAI_RX_COMPLETE_CB_ID       = 0x00U, /*!< SAI receive complete callback ID */
  HAL_SAI_RX_HALFCOMPLETE_CB_ID   = 0x01U, /*!< SAI receive half complete callback ID */
  HAL_SAI_TX_COMPLETE_CB_ID       = 0x02U, /*!< SAI transmit complete callback ID */
  HAL_SAI_TX_HALFCOMPLETE_CB_ID   = 0x03U, /*!< SAI transmit half complete callback ID */
  HAL_SAI_ERROR_CB_ID             = 0x04U, /*!< SAI error callback ID */
  HAL_SAI_MSPINIT_CB_ID           = 0x05U, /*!< SAI MSP init callback ID */
  HAL_SAI_MSPDEINIT_CB_ID         = 0x06U  /*!< SAI MSP de-init callback ID */
} HAL_SAI_CallbackIDTypeDef;

/**
  * @brief  SAI callback pointer definition
  */
typedef void (*pSAI_CallbackTypeDef)(SAI_HandleTypeDef *hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup SAI_Exported_Constants SAI Exported Constants
  * @{
  */

/** @defgroup SAI_Error_Code SAI Error Code
  * @{
  */
#define HAL_SAI_ERROR_NONE    0x00000000U  /*!< No error                                    */
#define HAL_SAI_ERROR_OVR     0x00000001U  /*!< Overrun Error                               */
#define HAL_SAI_ERROR_UDR     0x00000002U  /*!< Underrun error                              */
#define HAL_SAI_ERROR_AFSDET  0x00000004U  /*!< Anticipated Frame synchronisation detection */
#define HAL_SAI_ERROR_LFSDET  0x00000008U  /*!< Late Frame synchronisation detection        */
#define HAL_SAI_ERROR_CNREADY 0x00000010U  /*!< codec not ready                             */
#define HAL_SAI_ERROR_WCKCFG  0x00000020U  /*!< Wrong clock configuration                   */
#define HAL_SAI_ERROR_TIMEOUT 0x00000040U  /*!< Timeout error                               */
#define HAL_SAI_ERROR_DMA     0x00000080U  /*!< DMA error                                   */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
#define HAL_SAI_ERROR_INVALID_CALLBACK   0x00000100U  /*!< Invalid callback error */
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
/**
  * @}
  */

/** @defgroup SAI_Block_SyncExt SAI External synchronisation
  * @{
  */
#define SAI_SYNCEXT_DISABLE          0U
#define SAI_SYNCEXT_OUTBLOCKA_ENABLE 1U
#define SAI_SYNCEXT_OUTBLOCKB_ENABLE 2U
/**
  * @}
  */

/** @defgroup SAI_Protocol SAI Supported protocol
  * @{
  */
#define SAI_I2S_STANDARD      0U
#define SAI_I2S_MSBJUSTIFIED  1U
#define SAI_I2S_LSBJUSTIFIED  2U
#define SAI_PCM_LONG          3U
#define SAI_PCM_SHORT         4U
/**
  * @}
  */

/** @defgroup SAI_Protocol_DataSize SAI protocol data size
  * @{
  */
#define SAI_PROTOCOL_DATASIZE_16BIT         0U
#define SAI_PROTOCOL_DATASIZE_16BITEXTENDED 1U
#define SAI_PROTOCOL_DATASIZE_24BIT         2U
#define SAI_PROTOCOL_DATASIZE_32BIT         3U
/**
  * @}
  */

/** @defgroup SAI_Audio_Frequency SAI Audio Frequency
  * @{
  */
#define SAI_AUDIO_FREQUENCY_192K          192000U
#define SAI_AUDIO_FREQUENCY_96K           96000U
#define SAI_AUDIO_FREQUENCY_48K           48000U
#define SAI_AUDIO_FREQUENCY_44K           44100U
#define SAI_AUDIO_FREQUENCY_32K           32000U
#define SAI_AUDIO_FREQUENCY_22K           22050U
#define SAI_AUDIO_FREQUENCY_16K           16000U
#define SAI_AUDIO_FREQUENCY_11K           11025U
#define SAI_AUDIO_FREQUENCY_8K            8000U
#define SAI_AUDIO_FREQUENCY_MCKDIV        0U
/**
  * @}
  */

/** @defgroup SAI_Block_Mode SAI Block Mode
  * @{
  */
#define SAI_MODEMASTER_TX         0x00000000U
#define SAI_MODEMASTER_RX         ((uint32_t)SAI_xCR1_MODE_0)
#define SAI_MODESLAVE_TX          ((uint32_t)SAI_xCR1_MODE_1)
#define SAI_MODESLAVE_RX          ((uint32_t)(SAI_xCR1_MODE_1 | SAI_xCR1_MODE_0))
/**
  * @}
  */

/** @defgroup SAI_Block_Protocol SAI Block Protocol
  * @{
  */
#define SAI_FREE_PROTOCOL                 0x00000000U
#define SAI_SPDIF_PROTOCOL                ((uint32_t)SAI_xCR1_PRTCFG_0)
#define SAI_AC97_PROTOCOL                 ((uint32_t)SAI_xCR1_PRTCFG_1)
/**
  * @}
  */

/** @defgroup SAI_Block_Data_Size SAI Block Data Size
  * @{
  */
#define SAI_DATASIZE_8                   ((uint32_t)SAI_xCR1_DS_1)
#define SAI_DATASIZE_10                  ((uint32_t)(SAI_xCR1_DS_1 | SAI_xCR1_DS_0))
#define SAI_DATASIZE_16                  ((uint32_t)SAI_xCR1_DS_2)
#define SAI_DATASIZE_20                  ((uint32_t)(SAI_xCR1_DS_2 | SAI_xCR1_DS_0))
#define SAI_DATASIZE_24                  ((uint32_t)(SAI_xCR1_DS_2 | SAI_xCR1_DS_1))
#define SAI_DATASIZE_32                  ((uint32_t)(SAI_xCR1_DS_2 | SAI_xCR1_DS_1 | SAI_xCR1_DS_0))
/**
  * @}
  */

/** @defgroup SAI_Block_MSB_LSB_transmission SAI Block MSB LSB transmission
  * @{
  */
#define SAI_FIRSTBIT_MSB                  0x00000000U
#define SAI_FIRSTBIT_LSB                  ((uint32_t)SAI_xCR1_LSBFIRST)
/**
  * @}
  */

/** @defgroup SAI_Block_Clock_Strobing SAI Block Clock Strobing
  * @{
  */
#define SAI_CLOCKSTROBING_FALLINGEDGE     0U
#define SAI_CLOCKSTROBING_RISINGEDGE      1U
/**
  * @}
  */

/** @defgroup SAI_Block_Synchronization SAI Block Synchronization
  * @{
  */
#define SAI_ASYNCHRONOUS                  0U /*!< Asynchronous                             */
#define SAI_SYNCHRONOUS                   1U /*!< Synchronous with other block of same SAI */
#define SAI_SYNCHRONOUS_EXT_SAI1          2U /*!< Synchronous with other SAI, SAI1         */
#define SAI_SYNCHRONOUS_EXT_SAI2          3U /*!< Synchronous with other SAI, SAI2         */
/**
  * @}
  */

/** @defgroup SAI_Block_Output_Drive SAI Block Output Drive
  * @{
  */
#define SAI_OUTPUTDRIVE_DISABLE          0x00000000U
#define SAI_OUTPUTDRIVE_ENABLE           ((uint32_t)SAI_xCR1_OUTDRIV)
/**
  * @}
  */

/** @defgroup SAI_Block_NoDivider SAI Block NoDivider
  * @{
  */
#define SAI_MASTERDIVIDER_ENABLE         0x00000000U
#define SAI_MASTERDIVIDER_DISABLE        ((uint32_t)SAI_xCR1_NODIV)
/**
  * @}
  */

/** @defgroup SAI_Block_FS_Definition SAI Block FS Definition
  * @{
  */
#define SAI_FS_STARTFRAME                 0x00000000U
#define SAI_FS_CHANNEL_IDENTIFICATION     ((uint32_t)SAI_xFRCR_FSDEF)
/**
  * @}
  */

/** @defgroup SAI_Block_FS_Polarity SAI Block FS Polarity
  * @{
  */
#define SAI_FS_ACTIVE_LOW                  0x00000000U
#define SAI_FS_ACTIVE_HIGH                 ((uint32_t)SAI_xFRCR_FSPOL)
/**
  * @}
  */

/** @defgroup SAI_Block_FS_Offset SAI Block FS Offset
  * @{
  */
#define SAI_FS_FIRSTBIT                   0x00000000U
#define SAI_FS_BEFOREFIRSTBIT             ((uint32_t)SAI_xFRCR_FSOFF)
/**
  * @}
  */

/** @defgroup SAI_Block_Slot_Size SAI Block Slot Size
  * @{
  */
#define SAI_SLOTSIZE_DATASIZE             0x00000000U
#define SAI_SLOTSIZE_16B                  ((uint32_t)SAI_xSLOTR_SLOTSZ_0)
#define SAI_SLOTSIZE_32B                  ((uint32_t)SAI_xSLOTR_SLOTSZ_1)
/**
  * @}
  */

/** @defgroup SAI_Block_Slot_Active SAI Block Slot Active
  * @{
  */
#define SAI_SLOT_NOTACTIVE           0x00000000U
#define SAI_SLOTACTIVE_0             0x00000001U
#define SAI_SLOTACTIVE_1             0x00000002U
#define SAI_SLOTACTIVE_2             0x00000004U
#define SAI_SLOTACTIVE_3             0x00000008U
#define SAI_SLOTACTIVE_4             0x00000010U
#define SAI_SLOTACTIVE_5             0x00000020U
#define SAI_SLOTACTIVE_6             0x00000040U
#define SAI_SLOTACTIVE_7             0x00000080U
#define SAI_SLOTACTIVE_8             0x00000100U
#define SAI_SLOTACTIVE_9             0x00000200U
#define SAI_SLOTACTIVE_10            0x00000400U
#define SAI_SLOTACTIVE_11            0x00000800U
#define SAI_SLOTACTIVE_12            0x00001000U
#define SAI_SLOTACTIVE_13            0x00002000U
#define SAI_SLOTACTIVE_14            0x00004000U
#define SAI_SLOTACTIVE_15            0x00008000U
#define SAI_SLOTACTIVE_ALL           0x0000FFFFU
/**
  * @}
  */

/** @defgroup SAI_Mono_Stereo_Mode SAI Mono Stereo Mode
  * @{
  */
#define SAI_STEREOMODE                    0x00000000U
#define SAI_MONOMODE                      ((uint32_t)SAI_xCR1_MONO)
/**
  * @}
  */

/** @defgroup SAI_TRIState_Management SAI TRIState Management
  * @{
  */
#define SAI_OUTPUT_NOTRELEASED              0x00000000U
#define SAI_OUTPUT_RELEASED                 ((uint32_t)SAI_xCR2_TRIS)
/**
  * @}
  */

/** @defgroup SAI_Block_Fifo_Threshold SAI Block Fifo Threshold
  * @{
  */
#define SAI_FIFOTHRESHOLD_EMPTY  0x00000000U
#define SAI_FIFOTHRESHOLD_1QF    ((uint32_t)(SAI_xCR2_FTH_0))
#define SAI_FIFOTHRESHOLD_HF     ((uint32_t)(SAI_xCR2_FTH_1))
#define SAI_FIFOTHRESHOLD_3QF    ((uint32_t)(SAI_xCR2_FTH_1 | SAI_xCR2_FTH_0))
#define SAI_FIFOTHRESHOLD_FULL   ((uint32_t)(SAI_xCR2_FTH_2))
/**
  * @}
  */

/** @defgroup SAI_Block_Companding_Mode SAI Block Companding Mode
  * @{
  */
#define SAI_NOCOMPANDING                 0x00000000U
#define SAI_ULAW_1CPL_COMPANDING         ((uint32_t)(SAI_xCR2_COMP_1))
#define SAI_ALAW_1CPL_COMPANDING         ((uint32_t)(SAI_xCR2_COMP_1 | SAI_xCR2_COMP_0))
#define SAI_ULAW_2CPL_COMPANDING         ((uint32_t)(SAI_xCR2_COMP_1 | SAI_xCR2_CPL))
#define SAI_ALAW_2CPL_COMPANDING         ((uint32_t)(SAI_xCR2_COMP_1 | SAI_xCR2_COMP_0 | SAI_xCR2_CPL))
/**
  * @}
  */

/** @defgroup SAI_Block_Mute_Value SAI Block Mute Value
  * @{
  */
#define SAI_ZERO_VALUE                      0x00000000U
#define SAI_LAST_SENT_VALUE                 ((uint32_t)SAI_xCR2_MUTEVAL)
/**
  * @}
  */

/** @defgroup SAI_Block_Interrupts_Definition SAI Block Interrupts Definition
  * @{
  */
#define SAI_IT_OVRUDR                     ((uint32_t)SAI_xIMR_OVRUDRIE)
#define SAI_IT_MUTEDET                    ((uint32_t)SAI_xIMR_MUTEDETIE)
#define SAI_IT_WCKCFG                     ((uint32_t)SAI_xIMR_WCKCFGIE)
#define SAI_IT_FREQ                       ((uint32_t)SAI_xIMR_FREQIE)
#define SAI_IT_CNRDY                      ((uint32_t)SAI_xIMR_CNRDYIE)
#define SAI_IT_AFSDET                     ((uint32_t)SAI_xIMR_AFSDETIE)
#define SAI_IT_LFSDET                     ((uint32_t)SAI_xIMR_LFSDETIE)
/**
  * @}
  */

/** @defgroup SAI_Block_Flags_Definition  SAI Block Flags Definition
  * @{
  */
#define SAI_FLAG_OVRUDR                   ((uint32_t)SAI_xSR_OVRUDR)
#define SAI_FLAG_MUTEDET                  ((uint32_t)SAI_xSR_MUTEDET)
#define SAI_FLAG_WCKCFG                   ((uint32_t)SAI_xSR_WCKCFG)
#define SAI_FLAG_FREQ                     ((uint32_t)SAI_xSR_FREQ)
#define SAI_FLAG_CNRDY                    ((uint32_t)SAI_xSR_CNRDY)
#define SAI_FLAG_AFSDET                   ((uint32_t)SAI_xSR_AFSDET)
#define SAI_FLAG_LFSDET                   ((uint32_t)SAI_xSR_LFSDET)
/**
  * @}
  */

/** @defgroup SAI_Block_Fifo_Status_Level   SAI Block Fifo Status Level
  * @{
  */
#define SAI_FIFOSTATUS_EMPTY              0x00000000U
#define SAI_FIFOSTATUS_LESS1QUARTERFULL   0x00010000U
#define SAI_FIFOSTATUS_1QUARTERFULL       0x00020000U
#define SAI_FIFOSTATUS_HALFFULL           0x00030000U
#define SAI_FIFOSTATUS_3QUARTERFULL       0x00040000U
#define SAI_FIFOSTATUS_FULL               0x00050000U
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup SAI_Exported_Macros SAI Exported Macros
  * @brief macros to handle interrupts and specific configurations
  * @{
  */

/** @brief Reset SAI handle state
  * @param  __HANDLE__ specifies the SAI Handle.
  * @retval None
  */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
#define __HAL_SAI_RESET_HANDLE_STATE(__HANDLE__) do{                                            \
                                                     (__HANDLE__)->State = HAL_SAI_STATE_RESET; \
                                                     (__HANDLE__)->MspInitCallback = NULL;      \
                                                     (__HANDLE__)->MspDeInitCallback = NULL;    \
                                                   } while(0U)
#else
#define __HAL_SAI_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_SAI_STATE_RESET)
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */

/** @brief  Enable or disable the specified SAI interrupts.
  * @param  __HANDLE__ specifies the SAI Handle.
  * @param  __INTERRUPT__ specifies the interrupt source to enable or disable.
  *         This parameter can be one of the following values:
  *            @arg SAI_IT_OVRUDR: Overrun underrun interrupt enable
  *            @arg SAI_IT_MUTEDET: Mute detection interrupt enable
  *            @arg SAI_IT_WCKCFG: Wrong Clock Configuration interrupt enable
  *            @arg SAI_IT_FREQ: FIFO request interrupt enable
  *            @arg SAI_IT_CNRDY: Codec not ready interrupt enable
  *            @arg SAI_IT_AFSDET: Anticipated frame synchronization detection interrupt enable
  *            @arg SAI_IT_LFSDET: Late frame synchronization detection interrupt enable
  * @retval None
  */
#define __HAL_SAI_ENABLE_IT(__HANDLE__, __INTERRUPT__)   ((__HANDLE__)->Instance->IMR |= (__INTERRUPT__))
#define __HAL_SAI_DISABLE_IT(__HANDLE__, __INTERRUPT__)  ((__HANDLE__)->Instance->IMR &= (~(__INTERRUPT__)))

/** @brief  Check if the specified SAI interrupt source is enabled or disabled.
  * @param  __HANDLE__ specifies the SAI Handle.
  *         This parameter can be SAI where x: 1, 2, or 3 to select the SAI peripheral.
  * @param  __INTERRUPT__ specifies the SAI interrupt source to check.
  *         This parameter can be one of the following values:
  *            @arg SAI_IT_OVRUDR: Overrun underrun interrupt enable
  *            @arg SAI_IT_MUTEDET: Mute detection interrupt enable
  *            @arg SAI_IT_WCKCFG: Wrong Clock Configuration interrupt enable
  *            @arg SAI_IT_FREQ: FIFO request interrupt enable
  *            @arg SAI_IT_CNRDY: Codec not ready interrupt enable
  *            @arg SAI_IT_AFSDET: Anticipated frame synchronization detection interrupt enable
  *            @arg SAI_IT_LFSDET: Late frame synchronization detection interrupt enable
  * @retval The new state of __INTERRUPT__ (TRUE or FALSE).
  */
#define __HAL_SAI_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) ((((__HANDLE__)->Instance->IMR & (__INTERRUPT__)) == (__INTERRUPT__)) ? SET : RESET)

/** @brief  Check whether the specified SAI flag is set or not.
  * @param  __HANDLE__ specifies the SAI Handle.
  * @param  __FLAG__ specifies the flag to check.
  *         This parameter can be one of the following values:
  *            @arg SAI_FLAG_OVRUDR: Overrun underrun flag.
  *            @arg SAI_FLAG_MUTEDET: Mute detection flag.
  *            @arg SAI_FLAG_WCKCFG: Wrong Clock Configuration flag.
  *            @arg SAI_FLAG_FREQ: FIFO request flag.
  *            @arg SAI_FLAG_CNRDY: Codec not ready flag.
  *            @arg SAI_FLAG_AFSDET: Anticipated frame synchronization detection flag.
  *            @arg SAI_FLAG_LFSDET: Late frame synchronization detection flag.
  * @retval The new state of __FLAG__ (TRUE or FALSE).
  */
#define __HAL_SAI_GET_FLAG(__HANDLE__, __FLAG__) ((((__HANDLE__)->Instance->SR) & (__FLAG__)) == (__FLAG__))

/** @brief  Clear the specified SAI pending flag.
  * @param  __HANDLE__ specifies the SAI Handle.
  * @param  __FLAG__ specifies the flag to check.
  *          This parameter can be any combination of the following values:
  *            @arg SAI_FLAG_OVRUDR: Clear Overrun underrun
  *            @arg SAI_FLAG_MUTEDET: Clear Mute detection
  *            @arg SAI_FLAG_WCKCFG: Clear Wrong Clock Configuration
  *            @arg SAI_FLAG_FREQ: Clear FIFO request
  *            @arg SAI_FLAG_CNRDY: Clear Codec not ready
  *            @arg SAI_FLAG_AFSDET: Clear Anticipated frame synchronization detection
  *            @arg SAI_FLAG_LFSDET: Clear Late frame synchronization detection
  * @retval None
  */
#define __HAL_SAI_CLEAR_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->CLRFR = (__FLAG__))

/** @brief  Enable SAI
  * @param  __HANDLE__ specifies the SAI Handle.
  * @retval None
  */
#define __HAL_SAI_ENABLE(__HANDLE__) ((__HANDLE__)->Instance->CR1 |=  SAI_xCR1_SAIEN)

/** @brief  Disable SAI
  * @param  __HANDLE__ specifies the SAI Handle.
  * @retval None
  */
#define __HAL_SAI_DISABLE(__HANDLE__) ((__HANDLE__)->Instance->CR1 &=  ~SAI_xCR1_SAIEN)

/**
  * @}
  */

/* Include HAL SAI Extension module */
#include "stm32f4xx_hal_sai_ex.h"

/* Exported functions --------------------------------------------------------*/
/** @addtogroup SAI_Exported_Functions
  * @{
  */

/* Initialization/de-initialization functions  **********************************/
/** @addtogroup SAI_Exported_Functions_Group1
  * @{
  */
HAL_StatusTypeDef HAL_SAI_InitProtocol(SAI_HandleTypeDef *hsai, uint32_t protocol, uint32_t datasize, uint32_t nbslot);
HAL_StatusTypeDef HAL_SAI_Init(SAI_HandleTypeDef *hsai);
HAL_StatusTypeDef HAL_SAI_DeInit(SAI_HandleTypeDef *hsai);
void HAL_SAI_MspInit(SAI_HandleTypeDef *hsai);
void HAL_SAI_MspDeInit(SAI_HandleTypeDef *hsai);

#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
/* SAI callbacks register/unregister functions ********************************/
HAL_StatusTypeDef HAL_SAI_RegisterCallback(SAI_HandleTypeDef        *hsai,
                                           HAL_SAI_CallbackIDTypeDef CallbackID,
                                           pSAI_CallbackTypeDef      pCallback);
HAL_StatusTypeDef HAL_SAI_UnRegisterCallback(SAI_HandleTypeDef        *hsai,
                                             HAL_SAI_CallbackIDTypeDef CallbackID);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
/**
  * @}
  */

/* I/O operation functions  *****************************************************/
/** @addtogroup SAI_Exported_Functions_Group2
  * @{
  */
/* Blocking mode: Polling */
HAL_StatusTypeDef HAL_SAI_Transmit(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size, uint32_t Timeout);
HAL_StatusTypeDef HAL_SAI_Receive(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size, uint32_t Timeout);

/* Non-Blocking mode: Interrupt */
HAL_StatusTypeDef HAL_SAI_Transmit_IT(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_SAI_Receive_IT(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size);

/* Non-Blocking mode: DMA */
HAL_StatusTypeDef HAL_SAI_Transmit_DMA(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_SAI_Receive_DMA(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size);
HAL_StatusTypeDef HAL_SAI_DMAPause(SAI_HandleTypeDef *hsai);
HAL_StatusTypeDef HAL_SAI_DMAResume(SAI_HandleTypeDef *hsai);
HAL_StatusTypeDef HAL_SAI_DMAStop(SAI_HandleTypeDef *hsai);

/* Abort function */
HAL_StatusTypeDef HAL_SAI_Abort(SAI_HandleTypeDef *hsai);

/* Mute management */
HAL_StatusTypeDef HAL_SAI_EnableTxMuteMode(SAI_HandleTypeDef *hsai, uint16_t val);
HAL_StatusTypeDef HAL_SAI_DisableTxMuteMode(SAI_HandleTypeDef *hsai);
HAL_StatusTypeDef HAL_SAI_EnableRxMuteMode(SAI_HandleTypeDef *hsai, SAIcallback callback, uint16_t counter);
HAL_StatusTypeDef HAL_SAI_DisableRxMuteMode(SAI_HandleTypeDef *hsai);

/* SAI IRQHandler and Callbacks used in non blocking modes (Interrupt and DMA) */
void HAL_SAI_IRQHandler(SAI_HandleTypeDef *hsai);
void HAL_SAI_TxHalfCpltCallback(SAI_HandleTypeDef *hsai);
void HAL_SAI_TxCpltCallback(SAI_HandleTypeDef *hsai);
void HAL_SAI_RxHalfCpltCallback(SAI_HandleTypeDef *hsai);
void HAL_SAI_RxCpltCallback(SAI_HandleTypeDef *hsai);
void HAL_SAI_ErrorCallback(SAI_HandleTypeDef *hsai);
/**
  * @}
  */

/** @addtogroup SAI_Exported_Functions_Group3
  * @{
  */
/* Peripheral State functions  ************************************************/
HAL_SAI_StateTypeDef HAL_SAI_GetState(SAI_HandleTypeDef *hsai);
uint32_t HAL_SAI_GetError(SAI_HandleTypeDef *hsai);
/**
  * @}
  */

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @addtogroup SAI_Private_Macros
  * @{
  */
#define IS_SAI_BLOCK_SYNCEXT(STATE)   (((STATE) == SAI_SYNCEXT_DISABLE)           ||\
                                       ((STATE) == SAI_SYNCEXT_OUTBLOCKA_ENABLE)  ||\
                                       ((STATE) == SAI_SYNCEXT_OUTBLOCKB_ENABLE))

#define IS_SAI_SUPPORTED_PROTOCOL(PROTOCOL)   (((PROTOCOL) == SAI_I2S_STANDARD)     ||\
                                               ((PROTOCOL) == SAI_I2S_MSBJUSTIFIED) ||\
                                               ((PROTOCOL) == SAI_I2S_LSBJUSTIFIED) ||\
                                               ((PROTOCOL) == SAI_PCM_LONG)         ||\
                                               ((PROTOCOL) == SAI_PCM_SHORT))

#define IS_SAI_PROTOCOL_DATASIZE(DATASIZE)   (((DATASIZE) == SAI_PROTOCOL_DATASIZE_16BIT)         ||\
                                              ((DATASIZE) == SAI_PROTOCOL_DATASIZE_16BITEXTENDED) ||\
                                              ((DATASIZE) == SAI_PROTOCOL_DATASIZE_24BIT)         ||\
                                              ((DATASIZE) == SAI_PROTOCOL_DATASIZE_32BIT))

#define IS_SAI_AUDIO_FREQUENCY(AUDIO) (((AUDIO) == SAI_AUDIO_FREQUENCY_192K) || ((AUDIO) == SAI_AUDIO_FREQUENCY_96K) || \
                                       ((AUDIO) == SAI_AUDIO_FREQUENCY_48K)  || ((AUDIO) == SAI_AUDIO_FREQUENCY_44K) || \
                                       ((AUDIO) == SAI_AUDIO_FREQUENCY_32K)  || ((AUDIO) == SAI_AUDIO_FREQUENCY_22K) || \
                                       ((AUDIO) == SAI_AUDIO_FREQUENCY_16K)  || ((AUDIO) == SAI_AUDIO_FREQUENCY_11K) || \
                                       ((AUDIO) == SAI_AUDIO_FREQUENCY_8K)   || ((AUDIO) == SAI_AUDIO_FREQUENCY_MCKDIV))

#define IS_SAI_BLOCK_MODE(MODE)    (((MODE) == SAI_MODEMASTER_TX) || \
                                    ((MODE) == SAI_MODEMASTER_RX) || \
                                    ((MODE) == SAI_MODESLAVE_TX)  || \
                                    ((MODE) == SAI_MODESLAVE_RX))

#define IS_SAI_BLOCK_PROTOCOL(PROTOCOL) (((PROTOCOL) == SAI_FREE_PROTOCOL)  || \
                                         ((PROTOCOL) == SAI_AC97_PROTOCOL)  || \
                                         ((PROTOCOL) == SAI_SPDIF_PROTOCOL))

#define IS_SAI_BLOCK_DATASIZE(DATASIZE) (((DATASIZE) == SAI_DATASIZE_8)  || \
                                         ((DATASIZE) == SAI_DATASIZE_10) || \
                                         ((DATASIZE) == SAI_DATASIZE_16) || \
                                         ((DATASIZE) == SAI_DATASIZE_20) || \
                                         ((DATASIZE) == SAI_DATASIZE_24) || \
                                         ((DATASIZE) == SAI_DATASIZE_32))

#define IS_SAI_BLOCK_FIRST_BIT(BIT) (((BIT) == SAI_FIRSTBIT_MSB) || \
                                     ((BIT) == SAI_FIRSTBIT_LSB))

#define IS_SAI_BLOCK_CLOCK_STROBING(CLOCK) (((CLOCK) == SAI_CLOCKSTROBING_FALLINGEDGE) || \
                                            ((CLOCK) == SAI_CLOCKSTROBING_RISINGEDGE))

#define IS_SAI_BLOCK_SYNCHRO(SYNCHRO) (((SYNCHRO) == SAI_ASYNCHRONOUS)         || \
                                       ((SYNCHRO) == SAI_SYNCHRONOUS)          || \
                                       ((SYNCHRO) == SAI_SYNCHRONOUS_EXT_SAI1) ||\
                                       ((SYNCHRO) == SAI_SYNCHRONOUS_EXT_SAI2))

#define IS_SAI_BLOCK_OUTPUT_DRIVE(DRIVE) (((DRIVE) == SAI_OUTPUTDRIVE_DISABLE) || \
                                          ((DRIVE) == SAI_OUTPUTDRIVE_ENABLE))

#define IS_SAI_BLOCK_NODIVIDER(NODIVIDER) (((NODIVIDER) == SAI_MASTERDIVIDER_ENABLE) || \
                                           ((NODIVIDER) == SAI_MASTERDIVIDER_DISABLE))

#define IS_SAI_BLOCK_MUTE_COUNTER(COUNTER) ((COUNTER) <= 63U)

#define IS_SAI_BLOCK_MUTE_VALUE(VALUE)    (((VALUE) == SAI_ZERO_VALUE)     || \
                                           ((VALUE) == SAI_LAST_SENT_VALUE))

#define IS_SAI_BLOCK_COMPANDING_MODE(MODE)    (((MODE) == SAI_NOCOMPANDING)         || \
                                               ((MODE) == SAI_ULAW_1CPL_COMPANDING) || \
                                               ((MODE) == SAI_ALAW_1CPL_COMPANDING) || \
                                               ((MODE) == SAI_ULAW_2CPL_COMPANDING) || \
                                               ((MODE) == SAI_ALAW_2CPL_COMPANDING))

#define IS_SAI_BLOCK_FIFO_THRESHOLD(THRESHOLD) (((THRESHOLD) == SAI_FIFOTHRESHOLD_EMPTY)   || \
                                                ((THRESHOLD) == SAI_FIFOTHRESHOLD_1QF)     || \
                                                ((THRESHOLD) == SAI_FIFOTHRESHOLD_HF)      || \
                                                ((THRESHOLD) == SAI_FIFOTHRESHOLD_3QF)     || \
                                                ((THRESHOLD) == SAI_FIFOTHRESHOLD_FULL))

#define IS_SAI_BLOCK_TRISTATE_MANAGEMENT(STATE) (((STATE) == SAI_OUTPUT_NOTRELEASED) ||\
                                                 ((STATE) == SAI_OUTPUT_RELEASED))

#define IS_SAI_MONO_STEREO_MODE(MODE) (((MODE) == SAI_MONOMODE) ||\
                                       ((MODE) == SAI_STEREOMODE))

#define IS_SAI_SLOT_ACTIVE(ACTIVE)  ((ACTIVE) <= SAI_SLOTACTIVE_ALL)

#define IS_SAI_BLOCK_SLOT_NUMBER(NUMBER) ((1U <= (NUMBER)) && ((NUMBER) <= 16U))

#define IS_SAI_BLOCK_SLOT_SIZE(SIZE) (((SIZE) == SAI_SLOTSIZE_DATASIZE) || \
                                      ((SIZE) == SAI_SLOTSIZE_16B)      || \
                                      ((SIZE) == SAI_SLOTSIZE_32B))

#define IS_SAI_BLOCK_FIRSTBIT_OFFSET(OFFSET) ((OFFSET) <= 24U)

#define IS_SAI_BLOCK_FS_OFFSET(OFFSET) (((OFFSET) == SAI_FS_FIRSTBIT) || \
                                        ((OFFSET) == SAI_FS_BEFOREFIRSTBIT))

#define IS_SAI_BLOCK_FS_POLARITY(POLARITY) (((POLARITY) == SAI_FS_ACTIVE_LOW) || \
                                            ((POLARITY) == SAI_FS_ACTIVE_HIGH))

#define IS_SAI_BLOCK_FS_DEFINITION(DEFINITION) (((DEFINITION) == SAI_FS_STARTFRAME) || \
                                                ((DEFINITION) == SAI_FS_CHANNEL_IDENTIFICATION))

#define IS_SAI_BLOCK_MASTER_DIVIDER(DIVIDER) ((DIVIDER) <= 15U)

#define IS_SAI_BLOCK_FRAME_LENGTH(LENGTH) ((8U <= (LENGTH)) && ((LENGTH) <= 256U))

#define IS_SAI_BLOCK_ACTIVE_FRAME(LENGTH) ((1U <= (LENGTH)) && ((LENGTH) <= 128U))

/**
  * @}
  */

/* Private functions ---------------------------------------------------------*/
/** @defgroup SAI_Private_Functions SAI Private Functions
  * @{
  */

/**
  * @}
  */

/**
  * @}
  */

#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx || STM32F413xx || STM32F423xx  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_SAI_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
