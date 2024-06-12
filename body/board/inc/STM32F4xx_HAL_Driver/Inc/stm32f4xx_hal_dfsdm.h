/**
  ******************************************************************************
  * @file    stm32f4xx_hal_dfsdm.h
  * @author  MCD Application Team
  * @brief   Header file of DFSDM HAL module.
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
#ifndef __STM32F4xx_HAL_DFSDM_H
#define __STM32F4xx_HAL_DFSDM_H

#ifdef __cplusplus
 extern "C" {
#endif

#if defined(STM32F412Zx) || defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) || defined(STM32F423xx)
/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup DFSDM
  * @{
  */ 

/* Exported types ------------------------------------------------------------*/
/** @defgroup DFSDM_Exported_Types DFSDM Exported Types
  * @{
  */

/** 
  * @brief  HAL DFSDM Channel states definition  
  */ 
typedef enum
{
  HAL_DFSDM_CHANNEL_STATE_RESET = 0x00U, /*!< DFSDM channel not initialized */
  HAL_DFSDM_CHANNEL_STATE_READY = 0x01U, /*!< DFSDM channel initialized and ready for use */
  HAL_DFSDM_CHANNEL_STATE_ERROR = 0xFFU  /*!< DFSDM channel state error */
}HAL_DFSDM_Channel_StateTypeDef;

/** 
  * @brief  DFSDM channel output clock structure definition  
  */  
typedef struct
{
  FunctionalState Activation; /*!< Output clock enable/disable */
  uint32_t        Selection;  /*!< Output clock is system clock or audio clock.
                                   This parameter can be a value of @ref DFSDM_Channel_OuputClock */
  uint32_t        Divider;    /*!< Output clock divider.
                                   This parameter must be a number between Min_Data = 2 and Max_Data = 256 */
}DFSDM_Channel_OutputClockTypeDef;

/** 
  * @brief  DFSDM channel input structure definition  
  */  
typedef struct
{
  uint32_t Multiplexer; /*!< Input is external serial inputs or internal register.
                             This parameter can be a value of @ref DFSDM_Channel_InputMultiplexer */
  uint32_t DataPacking; /*!< Standard, interleaved or dual mode for internal register.
                             This parameter can be a value of @ref DFSDM_Channel_DataPacking */
  uint32_t Pins;        /*!< Input pins are taken from same or following channel.
                             This parameter can be a value of @ref DFSDM_Channel_InputPins */
}DFSDM_Channel_InputTypeDef;

/** 
  * @brief  DFSDM channel serial interface structure definition  
  */  
typedef struct
{
  uint32_t Type;     /*!< SPI or Manchester modes.
                          This parameter can be a value of @ref DFSDM_Channel_SerialInterfaceType */
  uint32_t SpiClock; /*!< SPI clock select (external or internal with different sampling point).
                          This parameter can be a value of @ref DFSDM_Channel_SpiClock */
}DFSDM_Channel_SerialInterfaceTypeDef;

/** 
  * @brief  DFSDM channel analog watchdog structure definition  
  */  
typedef struct
{
  uint32_t FilterOrder;  /*!< Analog watchdog Sinc filter order.
                              This parameter can be a value of @ref DFSDM_Channel_AwdFilterOrder */
  uint32_t Oversampling; /*!< Analog watchdog filter oversampling ratio.
                              This parameter must be a number between Min_Data = 1 and Max_Data = 32 */
}DFSDM_Channel_AwdTypeDef;

/** 
  * @brief  DFSDM channel init structure definition  
  */  
typedef struct
{
  DFSDM_Channel_OutputClockTypeDef     OutputClock;     /*!< DFSDM channel output clock parameters */
  DFSDM_Channel_InputTypeDef           Input;           /*!< DFSDM channel input parameters */
  DFSDM_Channel_SerialInterfaceTypeDef SerialInterface; /*!< DFSDM channel serial interface parameters */
  DFSDM_Channel_AwdTypeDef             Awd;             /*!< DFSDM channel analog watchdog parameters */
  int32_t                              Offset;          /*!< DFSDM channel offset.
                                                             This parameter must be a number between Min_Data = -8388608 and Max_Data = 8388607 */
  uint32_t                             RightBitShift;   /*!< DFSDM channel right bit shift.
                                                             This parameter must be a number between Min_Data = 0x00 and Max_Data = 0x1F */
}DFSDM_Channel_InitTypeDef;

/** 
  * @brief  DFSDM channel handle structure definition  
  */  
#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
typedef struct __DFSDM_Channel_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_DFSDM_REGISTER_CALLBACKS */ 
{
  DFSDM_Channel_TypeDef          *Instance; /*!< DFSDM channel instance */
  DFSDM_Channel_InitTypeDef      Init;      /*!< DFSDM channel init parameters */
  HAL_DFSDM_Channel_StateTypeDef State;     /*!< DFSDM channel state */
#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
  void (*CkabCallback)      (struct __DFSDM_Channel_HandleTypeDef *hdfsdm_channel); /*!< DFSDM channel clock absence detection callback */
  void (*ScdCallback)       (struct __DFSDM_Channel_HandleTypeDef *hdfsdm_channel); /*!< DFSDM channel short circuit detection callback */
  void (*MspInitCallback)   (struct __DFSDM_Channel_HandleTypeDef *hdfsdm_channel); /*!< DFSDM channel MSP init callback */
  void (*MspDeInitCallback) (struct __DFSDM_Channel_HandleTypeDef *hdfsdm_channel); /*!< DFSDM channel MSP de-init callback */
#endif
}DFSDM_Channel_HandleTypeDef;

#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
/**
  * @brief  DFSDM channel callback ID enumeration definition
  */
typedef enum
{
  HAL_DFSDM_CHANNEL_CKAB_CB_ID      = 0x00U, /*!< DFSDM channel clock absence detection callback ID */
  HAL_DFSDM_CHANNEL_SCD_CB_ID       = 0x01U, /*!< DFSDM channel short circuit detection callback ID */
  HAL_DFSDM_CHANNEL_MSPINIT_CB_ID   = 0x02U, /*!< DFSDM channel MSP init callback ID */
  HAL_DFSDM_CHANNEL_MSPDEINIT_CB_ID = 0x03U  /*!< DFSDM channel MSP de-init callback ID */
}HAL_DFSDM_Channel_CallbackIDTypeDef;

/**
  * @brief  DFSDM channel callback pointer definition
  */
typedef void (*pDFSDM_Channel_CallbackTypeDef)(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
#endif
/** 
  * @brief  HAL DFSDM Filter states definition  
  */ 
typedef enum
{
  HAL_DFSDM_FILTER_STATE_RESET   = 0x00U, /*!< DFSDM filter not initialized */
  HAL_DFSDM_FILTER_STATE_READY   = 0x01U, /*!< DFSDM filter initialized and ready for use */
  HAL_DFSDM_FILTER_STATE_REG     = 0x02U, /*!< DFSDM filter regular conversion in progress */
  HAL_DFSDM_FILTER_STATE_INJ     = 0x03U, /*!< DFSDM filter injected conversion in progress */
  HAL_DFSDM_FILTER_STATE_REG_INJ = 0x04U, /*!< DFSDM filter regular and injected conversions in progress */
  HAL_DFSDM_FILTER_STATE_ERROR   = 0xFFU  /*!< DFSDM filter state error */
}HAL_DFSDM_Filter_StateTypeDef;

/** 
  * @brief  DFSDM filter regular conversion parameters structure definition  
  */  
typedef struct
{
  uint32_t        Trigger;  /*!< Trigger used to start regular conversion: software or synchronous.
                                 This parameter can be a value of @ref DFSDM_Filter_Trigger */
  FunctionalState FastMode; /*!< Enable/disable fast mode for regular conversion */
  FunctionalState DmaMode;  /*!< Enable/disable DMA for regular conversion */
}DFSDM_Filter_RegularParamTypeDef;

/** 
  * @brief  DFSDM filter injected conversion parameters structure definition  
  */  
typedef struct
{
  uint32_t        Trigger;        /*!< Trigger used to start injected conversion: software, external or synchronous.
                                       This parameter can be a value of @ref DFSDM_Filter_Trigger */
  FunctionalState ScanMode;       /*!< Enable/disable scanning mode for injected conversion */
  FunctionalState DmaMode;        /*!< Enable/disable DMA for injected conversion */
  uint32_t        ExtTrigger;     /*!< External trigger.
                                       This parameter can be a value of @ref DFSDM_Filter_ExtTrigger */
  uint32_t        ExtTriggerEdge; /*!< External trigger edge: rising, falling or both.
                                       This parameter can be a value of @ref DFSDM_Filter_ExtTriggerEdge */
}DFSDM_Filter_InjectedParamTypeDef;

/** 
  * @brief  DFSDM filter parameters structure definition  
  */  
typedef struct
{
  uint32_t SincOrder;       /*!< Sinc filter order.
                                 This parameter can be a value of @ref DFSDM_Filter_SincOrder */
  uint32_t Oversampling;    /*!< Filter oversampling ratio.
                                 This parameter must be a number between Min_Data = 1 and Max_Data = 1024 */
  uint32_t IntOversampling; /*!< Integrator oversampling ratio.
                                 This parameter must be a number between Min_Data = 1 and Max_Data = 256 */
}DFSDM_Filter_FilterParamTypeDef;

/** 
  * @brief  DFSDM filter init structure definition  
  */  
typedef struct
{
  DFSDM_Filter_RegularParamTypeDef  RegularParam;  /*!< DFSDM regular conversion parameters */
  DFSDM_Filter_InjectedParamTypeDef InjectedParam; /*!< DFSDM injected conversion parameters */
  DFSDM_Filter_FilterParamTypeDef   FilterParam;   /*!< DFSDM filter parameters */
}DFSDM_Filter_InitTypeDef;

/** 
  * @brief  DFSDM filter handle structure definition  
  */  
#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
typedef struct __DFSDM_Filter_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_DFSDM_REGISTER_CALLBACKS */
{
  DFSDM_Filter_TypeDef          *Instance;           /*!< DFSDM filter instance */
  DFSDM_Filter_InitTypeDef      Init;                /*!< DFSDM filter init parameters */
  DMA_HandleTypeDef             *hdmaReg;            /*!< Pointer on DMA handler for regular conversions */
  DMA_HandleTypeDef             *hdmaInj;            /*!< Pointer on DMA handler for injected conversions */
  uint32_t                      RegularContMode;     /*!< Regular conversion continuous mode */
  uint32_t                      RegularTrigger;      /*!< Trigger used for regular conversion */
  uint32_t                      InjectedTrigger;     /*!< Trigger used for injected conversion */
  uint32_t                      ExtTriggerEdge;      /*!< Rising, falling or both edges selected */
  FunctionalState               InjectedScanMode;    /*!< Injected scanning mode */
  uint32_t                      InjectedChannelsNbr; /*!< Number of channels in injected sequence */
  uint32_t                      InjConvRemaining;    /*!< Injected conversions remaining */
  HAL_DFSDM_Filter_StateTypeDef State;               /*!< DFSDM filter state */
  uint32_t                      ErrorCode;           /*!< DFSDM filter error code */  
#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
  void (*AwdCallback)             (struct __DFSDM_Filter_HandleTypeDef *hdfsdm_filter,
                                   uint32_t Channel, uint32_t Threshold);               /*!< DFSDM filter analog watchdog callback */
  void (*RegConvCpltCallback)     (struct __DFSDM_Filter_HandleTypeDef *hdfsdm_filter); /*!< DFSDM filter regular conversion complete callback */
  void (*RegConvHalfCpltCallback) (struct __DFSDM_Filter_HandleTypeDef *hdfsdm_filter); /*!< DFSDM filter half regular conversion complete callback */
  void (*InjConvCpltCallback)     (struct __DFSDM_Filter_HandleTypeDef *hdfsdm_filter); /*!< DFSDM filter injected conversion complete callback */
  void (*InjConvHalfCpltCallback) (struct __DFSDM_Filter_HandleTypeDef *hdfsdm_filter); /*!< DFSDM filter half injected conversion complete callback */
  void (*ErrorCallback)           (struct __DFSDM_Filter_HandleTypeDef *hdfsdm_filter); /*!< DFSDM filter error callback */
  void (*MspInitCallback)         (struct __DFSDM_Filter_HandleTypeDef *hdfsdm_filter); /*!< DFSDM filter MSP init callback */
  void (*MspDeInitCallback)       (struct __DFSDM_Filter_HandleTypeDef *hdfsdm_filter); /*!< DFSDM filter MSP de-init callback */
#endif
}DFSDM_Filter_HandleTypeDef;

/** 
  * @brief  DFSDM filter analog watchdog parameters structure definition  
  */  
typedef struct
{
  uint32_t DataSource;      /*!< Values from digital filter or from channel watchdog filter.
                                 This parameter can be a value of @ref DFSDM_Filter_AwdDataSource */
  uint32_t Channel;         /*!< Analog watchdog channel selection.
                                 This parameter can be a values combination of @ref DFSDM_Channel_Selection */
  int32_t  HighThreshold;   /*!< High threshold for the analog watchdog.
                                 This parameter must be a number between Min_Data = -8388608 and Max_Data = 8388607 */
  int32_t  LowThreshold;    /*!< Low threshold for the analog watchdog.
                                 This parameter must be a number between Min_Data = -8388608 and Max_Data = 8388607 */
  uint32_t HighBreakSignal; /*!< Break signal assigned to analog watchdog high threshold event.
                                 This parameter can be a values combination of @ref DFSDM_BreakSignals */
  uint32_t LowBreakSignal;  /*!< Break signal assigned to analog watchdog low threshold event.
                                 This parameter can be a values combination of @ref DFSDM_BreakSignals */
}DFSDM_Filter_AwdParamTypeDef;

#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
/**
  * @brief  DFSDM filter callback ID enumeration definition
  */
typedef enum
{
  HAL_DFSDM_FILTER_REGCONV_COMPLETE_CB_ID     = 0x00U, /*!< DFSDM filter regular conversion complete callback ID */
  HAL_DFSDM_FILTER_REGCONV_HALFCOMPLETE_CB_ID = 0x01U, /*!< DFSDM filter half regular conversion complete callback ID */
  HAL_DFSDM_FILTER_INJCONV_COMPLETE_CB_ID     = 0x02U, /*!< DFSDM filter injected conversion complete callback ID */
  HAL_DFSDM_FILTER_INJCONV_HALFCOMPLETE_CB_ID = 0x03U, /*!< DFSDM filter half injected conversion complete callback ID */
  HAL_DFSDM_FILTER_ERROR_CB_ID                = 0x04U, /*!< DFSDM filter error callback ID */
  HAL_DFSDM_FILTER_MSPINIT_CB_ID              = 0x05U, /*!< DFSDM filter MSP init callback ID */
  HAL_DFSDM_FILTER_MSPDEINIT_CB_ID            = 0x06U  /*!< DFSDM filter MSP de-init callback ID */
}HAL_DFSDM_Filter_CallbackIDTypeDef;

/**
  * @brief  DFSDM filter callback pointer definition
  */
typedef void (*pDFSDM_Filter_CallbackTypeDef)(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
typedef void (*pDFSDM_Filter_AwdCallbackTypeDef)(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, uint32_t Channel, uint32_t Threshold);
#endif

/**
  * @}
  */ 
#if defined(SYSCFG_MCHDLYCR_BSCKSEL)  
/** 
  * @brief  Synchronization parameters structure definition for STM32F413xx/STM32F423xx devices  
  */  
typedef struct
{
  uint32_t  DFSDM1ClockIn;            /*!< Source selection for DFSDM1_Ckin.
                                          This parameter can be a value of @ref DFSDM_1_CLOCKIN_SELECTION*/
  uint32_t  DFSDM2ClockIn;            /*!< Source selection for DFSDM2_Ckin.
                                          This parameter can be a value of @ref DFSDM_2_CLOCKIN_SELECTION*/
  uint32_t  DFSDM1ClockOut;            /*!< Source selection for DFSDM1_Ckout.
                                          This parameter can be a value of @ref DFSDM_1_CLOCKOUT_SELECTION*/
  uint32_t  DFSDM2ClockOut;            /*!< Source selection for DFSDM2_Ckout.
                                          This parameter can be a value of @ref DFSDM_2_CLOCKOUT_SELECTION*/
  uint32_t  DFSDM1BitClkDistribution; /*!< Distribution of the DFSDM1 bitstream clock gated by TIM4 OC1 or TIM4 OC2.
                                          This parameter can be a value of @ref DFSDM_1_BIT_STREAM_DISTRIBUTION 
                                         @note The DFSDM2 audio gated by TIM4 OC2 can be injected on CKIN0 or CKIN2 
                                         @note The DFSDM2 audio gated by TIM4 OC1 can be injected on CKIN1 or CKIN3 */
  uint32_t  DFSDM2BitClkDistribution; /*!< Distribution of the DFSDM2 bitstream clock gated by TIM3 OC1 or TIM3 OC2 or TIM3 OC3 or TIM3 OC4.
                                          This parameter can be a value of @ref DFSDM_2_BIT_STREAM_DISTRIBUTION 
                                         @note The DFSDM2 audio gated by TIM3 OC4 can be injected on CKIN0 or CKIN4 
                                         @note The DFSDM2 audio gated by TIM3 OC3 can be injected on CKIN1 or CKIN5 
                                         @note The DFSDM2 audio gated by TIM3 OC2 can be injected on CKIN2 or CKIN6 
                                         @note The DFSDM2 audio gated by TIM3 OC1 can be injected on CKIN3 or CKIN7 */							 
  uint32_t  DFSDM1DataDistribution;   /*!< Source selection for DatIn0 and DatIn2 of DFSDM1.
                                         This parameter can be a value of @ref DFSDM_1_DATA_DISTRIBUTION */
  uint32_t  DFSDM2DataDistribution;  /*!< Source selection for DatIn0, DatIn2, DatIn4 and DatIn6 of DFSDM2.
                                         This parameter can be a value of @ref DFSDM_2_DATA_DISTRIBUTION */
}DFSDM_MultiChannelConfigTypeDef;
#endif  /* SYSCFG_MCHDLYCR_BSCKSEL */
/**
  * @}
  */ 
  
/* End of exported types -----------------------------------------------------*/

/* Exported constants --------------------------------------------------------*/
/** @defgroup DFSDM_Exported_Constants DFSDM Exported Constants
  * @{
  */

/** @defgroup DFSDM_Channel_OuputClock DFSDM channel output clock selection
  * @{
  */
#define DFSDM_CHANNEL_OUTPUT_CLOCK_SYSTEM    0x00000000U             /*!< Source for ouput clock is system clock */
#define DFSDM_CHANNEL_OUTPUT_CLOCK_AUDIO     DFSDM_CHCFGR1_CKOUTSRC  /*!< Source for ouput clock is audio clock */
/**
  * @}
  */

/** @defgroup DFSDM_Channel_InputMultiplexer DFSDM channel input multiplexer
  * @{
  */
#define DFSDM_CHANNEL_EXTERNAL_INPUTS        0x00000000U             /*!< Data are taken from external inputs */
#define DFSDM_CHANNEL_INTERNAL_REGISTER      DFSDM_CHCFGR1_DATMPX_1  /*!< Data are taken from internal register */
/**
  * @}
  */

/** @defgroup DFSDM_Channel_DataPacking DFSDM channel input data packing
  * @{
  */
#define DFSDM_CHANNEL_STANDARD_MODE          0x00000000U             /*!< Standard data packing mode */
#define DFSDM_CHANNEL_INTERLEAVED_MODE       DFSDM_CHCFGR1_DATPACK_0 /*!< Interleaved data packing mode */
#define DFSDM_CHANNEL_DUAL_MODE              DFSDM_CHCFGR1_DATPACK_1 /*!< Dual data packing mode */
/**
  * @}
  */

/** @defgroup DFSDM_Channel_InputPins DFSDM channel input pins
  * @{
  */
#define DFSDM_CHANNEL_SAME_CHANNEL_PINS      0x00000000U             /*!< Input from pins on same channel */
#define DFSDM_CHANNEL_FOLLOWING_CHANNEL_PINS DFSDM_CHCFGR1_CHINSEL   /*!< Input from pins on following channel */
/**
  * @}
  */

/** @defgroup DFSDM_Channel_SerialInterfaceType DFSDM channel serial interface type
  * @{
  */
#define DFSDM_CHANNEL_SPI_RISING             0x00000000U             /*!< SPI with rising edge */
#define DFSDM_CHANNEL_SPI_FALLING            DFSDM_CHCFGR1_SITP_0    /*!< SPI with falling edge */
#define DFSDM_CHANNEL_MANCHESTER_RISING      DFSDM_CHCFGR1_SITP_1    /*!< Manchester with rising edge */
#define DFSDM_CHANNEL_MANCHESTER_FALLING     DFSDM_CHCFGR1_SITP      /*!< Manchester with falling edge */
/**
  * @}
  */

/** @defgroup DFSDM_Channel_SpiClock DFSDM channel SPI clock selection
  * @{
  */
#define DFSDM_CHANNEL_SPI_CLOCK_EXTERNAL              0x00000000U              /*!< External SPI clock */
#define DFSDM_CHANNEL_SPI_CLOCK_INTERNAL              DFSDM_CHCFGR1_SPICKSEL_0 /*!< Internal SPI clock */
#define DFSDM_CHANNEL_SPI_CLOCK_INTERNAL_DIV2_FALLING DFSDM_CHCFGR1_SPICKSEL_1 /*!< Internal SPI clock divided by 2, falling edge */
#define DFSDM_CHANNEL_SPI_CLOCK_INTERNAL_DIV2_RISING  DFSDM_CHCFGR1_SPICKSEL   /*!< Internal SPI clock divided by 2, rising edge */
/**
  * @}
  */

/** @defgroup DFSDM_Channel_AwdFilterOrder DFSDM channel analog watchdog filter order
  * @{
  */
#define DFSDM_CHANNEL_FASTSINC_ORDER         0x00000000U              /*!< FastSinc filter type */
#define DFSDM_CHANNEL_SINC1_ORDER            DFSDM_CHAWSCDR_AWFORD_0  /*!< Sinc 1 filter type */
#define DFSDM_CHANNEL_SINC2_ORDER            DFSDM_CHAWSCDR_AWFORD_1  /*!< Sinc 2 filter type */
#define DFSDM_CHANNEL_SINC3_ORDER            DFSDM_CHAWSCDR_AWFORD    /*!< Sinc 3 filter type */
/**
  * @}
  */

/** @defgroup DFSDM_Filter_Trigger DFSDM filter conversion trigger
  * @{
  */
#define DFSDM_FILTER_SW_TRIGGER              0x00000000U              /*!< Software trigger */
#define DFSDM_FILTER_SYNC_TRIGGER            0x00000001U              /*!< Synchronous with DFSDM_FLT0 */
#define DFSDM_FILTER_EXT_TRIGGER             0x00000002U              /*!< External trigger (only for injected conversion) */
/**
  * @}
  */

/** @defgroup DFSDM_Filter_ExtTrigger DFSDM filter external trigger
  * @{
  */
#if defined(STM32F413xx) || defined(STM32F423xx) 
/* Trigger for stm32f413xx and STM32f423xx devices */
#define DFSDM_FILTER_EXT_TRIG_TIM1_TRGO  0x00000000U                                       /*!< For All DFSDM1/2 filters */
#define DFSDM_FILTER_EXT_TRIG_TIM3_TRGO  DFSDM_FLTCR1_JEXTSEL_0                            /*!< For All DFSDM1/2 filters */
#define DFSDM_FILTER_EXT_TRIG_TIM8_TRGO  DFSDM_FLTCR1_JEXTSEL_1                            /*!< For All DFSDM1/2 filters */
#define DFSDM_FILTER_EXT_TRIG_TIM10_OC1  (DFSDM_FLTCR1_JEXTSEL_0 | DFSDM_FLTCR1_JEXTSEL_1) /*!< For DFSDM1 filter 0 and 1 and DFSDM2 filter 0, 1 and 2 */
#define DFSDM_FILTER_EXT_TRIG_TIM2_TRGO  (DFSDM_FLTCR1_JEXTSEL_0 | DFSDM_FLTCR1_JEXTSEL_1) /*!< For DFSDM2 filter 3 */
#define DFSDM_FILTER_EXT_TRIG_TIM4_TRGO  DFSDM_FLTCR1_JEXTSEL_2                            /*!< For DFSDM1 filter 0 and 1 and DFSDM2 filter 0, 1 and 2 */
#define DFSDM_FILTER_EXT_TRIG_TIM11_OC1  DFSDM_FLTCR1_JEXTSEL_2                            /*!< For DFSDM2 filter 3 */
#define DFSDM_FILTER_EXT_TRIG_TIM6_TRGO  (DFSDM_FLTCR1_JEXTSEL_0 | DFSDM_FLTCR1_JEXTSEL_2) /*!< For DFSDM1 filter 0 and 1 and DFSDM2 filter 0 and 1 */
#define DFSDM_FILTER_EXT_TRIG_TIM7_TRGO  (DFSDM_FLTCR1_JEXTSEL_0 | DFSDM_FLTCR1_JEXTSEL_2) /*!< For DFSDM2 filter 2 and 3*/
#define DFSDM_FILTER_EXT_TRIG_EXTI11     (DFSDM_FLTCR1_JEXTSEL_1 | DFSDM_FLTCR1_JEXTSEL_2) /*!< For All DFSDM1/2 filters */
#define DFSDM_FILTER_EXT_TRIG_EXTI15     DFSDM_FLTCR1_JEXTSEL                              /*!< For All DFSDM1/2 filters */
#else
/* Trigger for stm32f412xx devices */
#define DFSDM_FILTER_EXT_TRIG_TIM1_TRGO  0x00000000U                                       /*!< For DFSDM1 filter 0 and 1*/
#define DFSDM_FILTER_EXT_TRIG_TIM3_TRGO  DFSDM_FLTCR1_JEXTSEL_0                            /*!< For DFSDM1 filter 0 and 1*/
#define DFSDM_FILTER_EXT_TRIG_TIM8_TRGO  DFSDM_FLTCR1_JEXTSEL_1                            /*!< For DFSDM1 filter 0 and 1*/
#define DFSDM_FILTER_EXT_TRIG_TIM10_OC1  (DFSDM_FLTCR1_JEXTSEL_0 | DFSDM_FLTCR1_JEXTSEL_1) /*!< For DFSDM1 filter 0 and 1*/
#define DFSDM_FILTER_EXT_TRIG_TIM4_TRGO  DFSDM_FLTCR1_JEXTSEL_2                            /*!< For DFSDM1 filter 0 and 1*/
#define DFSDM_FILTER_EXT_TRIG_TIM6_TRGO  (DFSDM_FLTCR1_JEXTSEL_0 | DFSDM_FLTCR1_JEXTSEL_2) /*!< For DFSDM1 filter 0 and 1*/
#define DFSDM_FILTER_EXT_TRIG_EXTI11     (DFSDM_FLTCR1_JEXTSEL_1 | DFSDM_FLTCR1_JEXTSEL_2) /*!< For DFSDM1 filter 0 and 1*/
#define DFSDM_FILTER_EXT_TRIG_EXTI15     DFSDM_FLTCR1_JEXTSEL                              /*!< For DFSDM1 filter 0 and 1*/
#endif
/**
  * @}
  */
  
/** @defgroup DFSDM_Filter_ExtTriggerEdge DFSDM filter external trigger edge
  * @{
  */
#define DFSDM_FILTER_EXT_TRIG_RISING_EDGE  DFSDM_FLTCR1_JEXTEN_0    /*!< External rising edge */
#define DFSDM_FILTER_EXT_TRIG_FALLING_EDGE DFSDM_FLTCR1_JEXTEN_1    /*!< External falling edge */
#define DFSDM_FILTER_EXT_TRIG_BOTH_EDGES   DFSDM_FLTCR1_JEXTEN      /*!< External rising and falling edges */
/**
  * @}
  */

/** @defgroup DFSDM_Filter_SincOrder DFSDM filter sinc order
  * @{
  */
#define DFSDM_FILTER_FASTSINC_ORDER        0x00000000U                                 /*!< FastSinc filter type */
#define DFSDM_FILTER_SINC1_ORDER           DFSDM_FLTFCR_FORD_0                         /*!< Sinc 1 filter type */
#define DFSDM_FILTER_SINC2_ORDER           DFSDM_FLTFCR_FORD_1                         /*!< Sinc 2 filter type */
#define DFSDM_FILTER_SINC3_ORDER           (DFSDM_FLTFCR_FORD_0 | DFSDM_FLTFCR_FORD_1) /*!< Sinc 3 filter type */
#define DFSDM_FILTER_SINC4_ORDER           DFSDM_FLTFCR_FORD_2                         /*!< Sinc 4 filter type */
#define DFSDM_FILTER_SINC5_ORDER           (DFSDM_FLTFCR_FORD_0 | DFSDM_FLTFCR_FORD_2) /*!< Sinc 5 filter type */
/**
  * @}
  */

/** @defgroup DFSDM_Filter_AwdDataSource DFSDM filter analog watchdog data source
  * @{
  */
#define DFSDM_FILTER_AWD_FILTER_DATA       0x00000000U             /*!< From digital filter */
#define DFSDM_FILTER_AWD_CHANNEL_DATA      DFSDM_FLTCR1_AWFSEL     /*!< From analog watchdog channel */
/**
  * @}
  */

/** @defgroup DFSDM_Filter_ErrorCode DFSDM filter error code
  * @{
  */ 
#define DFSDM_FILTER_ERROR_NONE             0x00000000U            /*!< No error */
#define DFSDM_FILTER_ERROR_REGULAR_OVERRUN  0x00000001U            /*!< Overrun occurs during regular conversion */
#define DFSDM_FILTER_ERROR_INJECTED_OVERRUN 0x00000002U            /*!< Overrun occurs during injected conversion */
#define DFSDM_FILTER_ERROR_DMA              0x00000003U            /*!< DMA error occurs */
#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
#define DFSDM_FILTER_ERROR_INVALID_CALLBACK 0x00000004U        /*!< Invalid callback error occurs */
#endif
/**
  * @}
  */

/** @defgroup DFSDM_BreakSignals DFSDM break signals
  * @{
  */
#define DFSDM_NO_BREAK_SIGNAL               0x00000000U            /*!< No break signal */
#define DFSDM_BREAK_SIGNAL_0                0x00000001U            /*!< Break signal 0 */
#define DFSDM_BREAK_SIGNAL_1                0x00000002U            /*!< Break signal 1 */
#define DFSDM_BREAK_SIGNAL_2                0x00000004U            /*!< Break signal 2 */
#define DFSDM_BREAK_SIGNAL_3                0x00000008U            /*!< Break signal 3 */
/**
  * @}
  */

/** @defgroup DFSDM_Channel_Selection DFSDM Channel Selection
  * @{
  */
/* DFSDM Channels ------------------------------------------------------------*/
/* The DFSDM channels are defined as follows:
   - in 16-bit LSB the channel mask is set
   - in 16-bit MSB the channel number is set
   e.g. for channel 3 definition:
        - the channel mask is 0x00000008 (bit 3 is set)
        - the channel number 3 is 0x00030000 
        --> Consequently, channel 3 definition is 0x00000008 | 0x00030000 = 0x00030008 */
#define DFSDM_CHANNEL_0                     0x00000001U
#define DFSDM_CHANNEL_1                     0x00010002U
#define DFSDM_CHANNEL_2                     0x00020004U
#define DFSDM_CHANNEL_3                     0x00030008U
#define DFSDM_CHANNEL_4                     0x00040010U            /* only for stmm32f413xx and stm32f423xx devices */
#define DFSDM_CHANNEL_5                     0x00050020U            /* only for stmm32f413xx and stm32f423xx devices */
#define DFSDM_CHANNEL_6                     0x00060040U            /* only for stmm32f413xx and stm32f423xx devices */
#define DFSDM_CHANNEL_7                     0x00070080U            /* only for stmm32f413xx and stm32f423xx devices */
/**
  * @}
  */

/** @defgroup DFSDM_ContinuousMode DFSDM Continuous Mode
  * @{
  */
#define DFSDM_CONTINUOUS_CONV_OFF            0x00000000U           /*!< Conversion are not continuous */
#define DFSDM_CONTINUOUS_CONV_ON             0x00000001U           /*!< Conversion are continuous */
/**
  * @}
  */

/** @defgroup DFSDM_AwdThreshold DFSDM analog watchdog threshold
  * @{
  */
#define DFSDM_AWD_HIGH_THRESHOLD            0x00000000U            /*!< Analog watchdog high threshold */
#define DFSDM_AWD_LOW_THRESHOLD             0x00000001U            /*!< Analog watchdog low threshold */
/**
  * @}
  */

#if defined(SYSCFG_MCHDLYCR_BSCKSEL)
/** @defgroup DFSDM_1_CLOCKOUT_SELECTION  DFSDM1 ClockOut Selection
  * @{
  */  
#define DFSDM1_CKOUT_DFSDM2_CKOUT           0x00000080U
#define DFSDM1_CKOUT_DFSDM1                 0x00000000U
/**
  * @}
  */  
  
/** @defgroup DFSDM_2_CLOCKOUT_SELECTION  DFSDM2 ClockOut Selection
  * @{
  */ 
#define DFSDM2_CKOUT_DFSDM2_CKOUT           0x00040000U
#define DFSDM2_CKOUT_DFSDM2                 0x00000000U  
/**
  * @}
  */ 

/** @defgroup DFSDM_1_CLOCKIN_SELECTION  DFSDM1 ClockIn Selection
  * @{
  */ 
#define DFSDM1_CKIN_DFSDM2_CKOUT            0x00000040U
#define DFSDM1_CKIN_PAD                     0x00000000U
/**
  * @}
  */

/** @defgroup DFSDM_2_CLOCKIN_SELECTION  DFSDM2 ClockIn Selection
  * @{
  */ 
#define DFSDM2_CKIN_DFSDM2_CKOUT            0x00020000U
#define DFSDM2_CKIN_PAD                     0x00000000U
/**
  * @}
  */

/** @defgroup DFSDM_1_BIT_STREAM_DISTRIBUTION  DFSDM1 Bit Stream Distribution 
  * @{
  */ 
#define DFSDM1_T4_OC2_BITSTREAM_CKIN0       0x00000000U                   /* TIM4_OC2 to CLKIN0 */
#define DFSDM1_T4_OC2_BITSTREAM_CKIN2       SYSCFG_MCHDLYCR_DFSDM1CK02SEL /* TIM4_OC2 to CLKIN2 */
#define DFSDM1_T4_OC1_BITSTREAM_CKIN3       SYSCFG_MCHDLYCR_DFSDM1CK13SEL /* TIM4_OC1 to CLKIN3 */
#define DFSDM1_T4_OC1_BITSTREAM_CKIN1       0x00000000U                   /* TIM4_OC1 to CLKIN1 */
/**
  * @}
  */
  
/** @defgroup DFSDM_2_BIT_STREAM_DISTRIBUTION  DFSDM12 Bit Stream Distribution 
  * @{
  */
#define DFSDM2_T3_OC4_BITSTREAM_CKIN0       0x00000000U                   /* TIM3_OC4 to CKIN0 */ 
#define DFSDM2_T3_OC4_BITSTREAM_CKIN4       SYSCFG_MCHDLYCR_DFSDM2CK04SEL /* TIM3_OC4 to CKIN4 */ 
#define DFSDM2_T3_OC3_BITSTREAM_CKIN5       SYSCFG_MCHDLYCR_DFSDM2CK15SEL /* TIM3_OC3 to CKIN5 */ 
#define DFSDM2_T3_OC3_BITSTREAM_CKIN1       0x00000000U                   /* TIM3_OC3 to CKIN1 */ 
#define DFSDM2_T3_OC2_BITSTREAM_CKIN6       SYSCFG_MCHDLYCR_DFSDM2CK26SEL /* TIM3_OC2to CKIN6 */
#define DFSDM2_T3_OC2_BITSTREAM_CKIN2       0x00000000U                   /* TIM3_OC2 to CKIN2 */
#define DFSDM2_T3_OC1_BITSTREAM_CKIN3       0x00000000U                   /* TIM3_OC1 to CKIN3 */
#define DFSDM2_T3_OC1_BITSTREAM_CKIN7       SYSCFG_MCHDLYCR_DFSDM2CK37SEL /* TIM3_OC1 to CKIN7 */
/**
  * @}
  */

/** @defgroup DFSDM_1_DATA_DISTRIBUTION DFSDM1 Data Distribution 
  * @{
  */  
#define DFSDM1_DATIN0_TO_DATIN0_PAD       0x00000000U
#define DFSDM1_DATIN0_TO_DATIN1_PAD       SYSCFG_MCHDLYCR_DFSDM1D0SEL
#define DFSDM1_DATIN2_TO_DATIN2_PAD       0x00000000U
#define DFSDM1_DATIN2_TO_DATIN3_PAD       SYSCFG_MCHDLYCR_DFSDM1D2SEL
/**
  * @}
  */
  
/** @defgroup DFSDM_2_DATA_DISTRIBUTION DFSDM2 Data Distribution 
  * @{
  */
#define DFSDM2_DATIN0_TO_DATIN0_PAD       0x00000000U
#define DFSDM2_DATIN0_TO_DATIN1_PAD       SYSCFG_MCHDLYCR_DFSDM2D0SEL
#define DFSDM2_DATIN2_TO_DATIN2_PAD       0x00000000U
#define DFSDM2_DATIN2_TO_DATIN3_PAD       SYSCFG_MCHDLYCR_DFSDM2D2SEL
#define DFSDM2_DATIN4_TO_DATIN4_PAD       0x00000000U
#define DFSDM2_DATIN4_TO_DATIN5_PAD       SYSCFG_MCHDLYCR_DFSDM2D4SEL 
#define DFSDM2_DATIN6_TO_DATIN6_PAD       0x00000000U 
#define DFSDM2_DATIN6_TO_DATIN7_PAD       SYSCFG_MCHDLYCR_DFSDM2D6SEL 
/**
  * @}
  */
  
/** @defgroup HAL_MCHDLY_CLOCK HAL MCHDLY Clock enable
  * @{
  */
#define HAL_MCHDLY_CLOCK_DFSDM2       SYSCFG_MCHDLYCR_MCHDLY2EN
#define HAL_MCHDLY_CLOCK_DFSDM1       SYSCFG_MCHDLYCR_MCHDLY1EN
/**
  * @}
  */

/** @defgroup DFSDM_CLOCKIN_SOURCE  DFSDM Clock In Source Selection
  * @{
  */
#define HAL_DFSDM2_CKIN_PAD           0x00040000U
#define HAL_DFSDM2_CKIN_DM            SYSCFG_MCHDLYCR_DFSDM2CFG
#define HAL_DFSDM1_CKIN_PAD           0x00000000U
#define HAL_DFSDM1_CKIN_DM            SYSCFG_MCHDLYCR_DFSDM1CFG
/**
  * @}
  */
  
/** @defgroup DFSDM_CLOCKOUT_SOURCE   DFSDM Clock Source Selection
  * @{
  */
#define HAL_DFSDM2_CKOUT_DFSDM2       0x10000000U
#define HAL_DFSDM2_CKOUT_M27          SYSCFG_MCHDLYCR_DFSDM2CKOSEL
#define HAL_DFSDM1_CKOUT_DFSDM1       0x00000000U
#define HAL_DFSDM1_CKOUT_M27          SYSCFG_MCHDLYCR_DFSDM1CKOSEL
/**
  * @}
  */

/** @defgroup DFSDM_DATAIN0_SOURCE   DFSDM Source Selection For DATAIN0
  * @{
  */
#define HAL_DATAIN0_DFSDM2_PAD        0x10000000U
#define HAL_DATAIN0_DFSDM2_DATAIN1    SYSCFG_MCHDLYCR_DFSDM2D0SEL
#define HAL_DATAIN0_DFSDM1_PAD        0x00000000U
#define HAL_DATAIN0_DFSDM1_DATAIN1    SYSCFG_MCHDLYCR_DFSDM1D0SEL
/**
  * @}
  */

/** @defgroup DFSDM_DATAIN2_SOURCE   DFSDM Source Selection For DATAIN2
  * @{
  */
#define HAL_DATAIN2_DFSDM2_PAD        0x10000000U
#define HAL_DATAIN2_DFSDM2_DATAIN3    SYSCFG_MCHDLYCR_DFSDM2D2SEL
#define HAL_DATAIN2_DFSDM1_PAD        0x00000000U
#define HAL_DATAIN2_DFSDM1_DATAIN3    SYSCFG_MCHDLYCR_DFSDM1D2SEL
/**
  * @}
  */ 
 
/** @defgroup DFSDM_DATAIN4_SOURCE   DFSDM Source Selection For DATAIN4
  * @{
  */
#define HAL_DATAIN4_DFSDM2_PAD        0x00000000U
#define HAL_DATAIN4_DFSDM2_DATAIN5    SYSCFG_MCHDLYCR_DFSDM2D4SEL
/**
  * @}
  */ 
  
/** @defgroup DFSDM_DATAIN6_SOURCE   DFSDM Source Selection For DATAIN6
  * @{
  */
#define HAL_DATAIN6_DFSDM2_PAD        0x00000000U
#define HAL_DATAIN6_DFSDM2_DATAIN7    SYSCFG_MCHDLYCR_DFSDM2D6SEL
/**
  * @}
  */ 

/** @defgroup DFSDM1_CLKIN_SOURCE   DFSDM1 Source Selection For CLKIN
  * @{
  */
#define HAL_DFSDM1_CLKIN0_TIM4OC2      0x01000000U
#define HAL_DFSDM1_CLKIN2_TIM4OC2      SYSCFG_MCHDLYCR_DFSDM1CK02SEL
#define HAL_DFSDM1_CLKIN1_TIM4OC1      0x02000000U
#define HAL_DFSDM1_CLKIN3_TIM4OC1      SYSCFG_MCHDLYCR_DFSDM1CK13SEL
/**
  * @}
  */ 
  
/** @defgroup DFSDM2_CLKIN_SOURCE  DFSDM2 Source Selection For CLKIN
  * @{
  */
#define HAL_DFSDM2_CLKIN0_TIM3OC4      0x04000000U
#define HAL_DFSDM2_CLKIN4_TIM3OC4      SYSCFG_MCHDLYCR_DFSDM2CK04SEL
#define HAL_DFSDM2_CLKIN1_TIM3OC3      0x08000000U
#define HAL_DFSDM2_CLKIN5_TIM3OC3      SYSCFG_MCHDLYCR_DFSDM2CK15SEL
#define HAL_DFSDM2_CLKIN2_TIM3OC2      0x10000000U
#define HAL_DFSDM2_CLKIN6_TIM3OC2      SYSCFG_MCHDLYCR_DFSDM2CK26SEL
#define HAL_DFSDM2_CLKIN3_TIM3OC1      0x00000000U
#define HAL_DFSDM2_CLKIN7_TIM3OC1      SYSCFG_MCHDLYCR_DFSDM2CK37SEL
/**
  * @}
  */    
       
#endif /* SYSCFG_MCHDLYCR_BSCKSEL*/
/**
  * @}
  */ 
/* End of exported constants -------------------------------------------------*/

/* Exported macros -----------------------------------------------------------*/
/** @defgroup DFSDM_Exported_Macros DFSDM Exported Macros
 * @{
 */

/** @brief  Reset DFSDM channel handle state.
  * @param  __HANDLE__ DFSDM channel handle.
  * @retval None
  */
#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
#define __HAL_DFSDM_CHANNEL_RESET_HANDLE_STATE(__HANDLE__) do{                                                      \
                                                               (__HANDLE__)->State = HAL_DFSDM_CHANNEL_STATE_RESET; \
                                                               (__HANDLE__)->MspInitCallback = NULL;                \
                                                               (__HANDLE__)->MspDeInitCallback = NULL;              \
                                                             } while(0)
#else
#define __HAL_DFSDM_CHANNEL_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_DFSDM_CHANNEL_STATE_RESET)
#endif

/** @brief  Reset DFSDM filter handle state.
  * @param  __HANDLE__ DFSDM filter handle.
  * @retval None
  */
#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
#define __HAL_DFSDM_FILTER_RESET_HANDLE_STATE(__HANDLE__) do{                                                     \
                                                              (__HANDLE__)->State = HAL_DFSDM_FILTER_STATE_RESET; \
                                                              (__HANDLE__)->MspInitCallback = NULL;               \
                                                              (__HANDLE__)->MspDeInitCallback = NULL;             \
                                                            } while(0)
#else
#define __HAL_DFSDM_FILTER_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_DFSDM_FILTER_STATE_RESET)
#endif

/**
  * @}
  */
/* End of exported macros ----------------------------------------------------*/

/* Exported functions --------------------------------------------------------*/
/** @addtogroup DFSDM_Exported_Functions DFSDM Exported Functions
  * @{
  */

/** @addtogroup DFSDM_Exported_Functions_Group1_Channel Channel initialization and de-initialization functions
  * @{
  */
/* Channel initialization and de-initialization functions *********************/
HAL_StatusTypeDef HAL_DFSDM_ChannelInit(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
HAL_StatusTypeDef HAL_DFSDM_ChannelDeInit(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
void HAL_DFSDM_ChannelMspInit(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
void HAL_DFSDM_ChannelMspDeInit(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
/* Channel callbacks register/unregister functions ****************************/
HAL_StatusTypeDef HAL_DFSDM_Channel_RegisterCallback(DFSDM_Channel_HandleTypeDef        *hdfsdm_channel,
                                                     HAL_DFSDM_Channel_CallbackIDTypeDef CallbackID,
                                                     pDFSDM_Channel_CallbackTypeDef      pCallback);
HAL_StatusTypeDef HAL_DFSDM_Channel_UnRegisterCallback(DFSDM_Channel_HandleTypeDef        *hdfsdm_channel,
                                                       HAL_DFSDM_Channel_CallbackIDTypeDef CallbackID);
#endif
/**
  * @}
  */

/** @addtogroup DFSDM_Exported_Functions_Group2_Channel Channel operation functions
  * @{
  */
/* Channel operation functions ************************************************/
HAL_StatusTypeDef HAL_DFSDM_ChannelCkabStart(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
HAL_StatusTypeDef HAL_DFSDM_ChannelCkabStart_IT(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
HAL_StatusTypeDef HAL_DFSDM_ChannelCkabStop(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
HAL_StatusTypeDef HAL_DFSDM_ChannelCkabStop_IT(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);

HAL_StatusTypeDef HAL_DFSDM_ChannelScdStart(DFSDM_Channel_HandleTypeDef *hdfsdm_channel, uint32_t Threshold, uint32_t BreakSignal);
HAL_StatusTypeDef HAL_DFSDM_ChannelScdStart_IT(DFSDM_Channel_HandleTypeDef *hdfsdm_channel, uint32_t Threshold, uint32_t BreakSignal);
HAL_StatusTypeDef HAL_DFSDM_ChannelScdStop(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
HAL_StatusTypeDef HAL_DFSDM_ChannelScdStop_IT(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);

int16_t           HAL_DFSDM_ChannelGetAwdValue(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
HAL_StatusTypeDef HAL_DFSDM_ChannelModifyOffset(DFSDM_Channel_HandleTypeDef *hdfsdm_channel, int32_t Offset);

HAL_StatusTypeDef HAL_DFSDM_ChannelPollForCkab(DFSDM_Channel_HandleTypeDef *hdfsdm_channel, uint32_t Timeout);
HAL_StatusTypeDef HAL_DFSDM_ChannelPollForScd(DFSDM_Channel_HandleTypeDef *hdfsdm_channel, uint32_t Timeout);

void HAL_DFSDM_ChannelCkabCallback(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
void HAL_DFSDM_ChannelScdCallback(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
/**
  * @}
  */

/** @defgroup DFSDM_Exported_Functions_Group3_Channel Channel state function
  * @{
  */
/* Channel state function *****************************************************/
HAL_DFSDM_Channel_StateTypeDef HAL_DFSDM_ChannelGetState(DFSDM_Channel_HandleTypeDef *hdfsdm_channel);
/**
  * @}
  */

/** @addtogroup DFSDM_Exported_Functions_Group1_Filter Filter initialization and de-initialization functions
  * @{
  */
/* Filter initialization and de-initialization functions *********************/
HAL_StatusTypeDef HAL_DFSDM_FilterInit(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterDeInit(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
void HAL_DFSDM_FilterMspInit(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
void HAL_DFSDM_FilterMspDeInit(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
#if (USE_HAL_DFSDM_REGISTER_CALLBACKS == 1)
/* Filter callbacks register/unregister functions ****************************/
HAL_StatusTypeDef HAL_DFSDM_Filter_RegisterCallback(DFSDM_Filter_HandleTypeDef        *hdfsdm_filter,
                                                    HAL_DFSDM_Filter_CallbackIDTypeDef CallbackID,
                                                    pDFSDM_Filter_CallbackTypeDef      pCallback);
HAL_StatusTypeDef HAL_DFSDM_Filter_UnRegisterCallback(DFSDM_Filter_HandleTypeDef        *hdfsdm_filter,
                                                      HAL_DFSDM_Filter_CallbackIDTypeDef CallbackID);
HAL_StatusTypeDef HAL_DFSDM_Filter_RegisterAwdCallback(DFSDM_Filter_HandleTypeDef      *hdfsdm_filter,
                                                       pDFSDM_Filter_AwdCallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_DFSDM_Filter_UnRegisterAwdCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
#endif
/**
  * @}
  */

/** @addtogroup DFSDM_Exported_Functions_Group2_Filter Filter control functions
  * @{
  */
/* Filter control functions *********************/
HAL_StatusTypeDef HAL_DFSDM_FilterConfigRegChannel(DFSDM_Filter_HandleTypeDef *hdfsdm_filter,
                                                   uint32_t                    Channel,
                                                   uint32_t                    ContinuousMode);
HAL_StatusTypeDef HAL_DFSDM_FilterConfigInjChannel(DFSDM_Filter_HandleTypeDef *hdfsdm_filter,
                                                   uint32_t                    Channel);
/**
  * @}
  */

/** @addtogroup DFSDM_Exported_Functions_Group3_Filter Filter operation functions
  * @{
  */
/* Filter operation functions *********************/
HAL_StatusTypeDef HAL_DFSDM_FilterRegularStart(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterRegularStart_IT(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterRegularStart_DMA(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, int32_t *pData, uint32_t Length);
HAL_StatusTypeDef HAL_DFSDM_FilterRegularMsbStart_DMA(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, int16_t *pData, uint32_t Length);
HAL_StatusTypeDef HAL_DFSDM_FilterRegularStop(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterRegularStop_IT(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterRegularStop_DMA(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterInjectedStart(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterInjectedStart_IT(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterInjectedStart_DMA(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, int32_t *pData, uint32_t Length);
HAL_StatusTypeDef HAL_DFSDM_FilterInjectedMsbStart_DMA(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, int16_t *pData, uint32_t Length);
HAL_StatusTypeDef HAL_DFSDM_FilterInjectedStop(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterInjectedStop_IT(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterInjectedStop_DMA(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterAwdStart_IT(DFSDM_Filter_HandleTypeDef *hdfsdm_filter,
                                              DFSDM_Filter_AwdParamTypeDef* awdParam);
HAL_StatusTypeDef HAL_DFSDM_FilterAwdStop_IT(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
HAL_StatusTypeDef HAL_DFSDM_FilterExdStart(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, uint32_t Channel);
HAL_StatusTypeDef HAL_DFSDM_FilterExdStop(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);

int32_t  HAL_DFSDM_FilterGetRegularValue(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, uint32_t* Channel);
int32_t  HAL_DFSDM_FilterGetInjectedValue(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, uint32_t* Channel);
int32_t  HAL_DFSDM_FilterGetExdMaxValue(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, uint32_t* Channel);
int32_t  HAL_DFSDM_FilterGetExdMinValue(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, uint32_t* Channel);
uint32_t HAL_DFSDM_FilterGetConvTimeValue(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);

void HAL_DFSDM_IRQHandler(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);

HAL_StatusTypeDef HAL_DFSDM_FilterPollForRegConversion(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, uint32_t Timeout);
HAL_StatusTypeDef HAL_DFSDM_FilterPollForInjConversion(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, uint32_t Timeout);

void HAL_DFSDM_FilterRegConvCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
void HAL_DFSDM_FilterRegConvHalfCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
void HAL_DFSDM_FilterInjConvCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
void HAL_DFSDM_FilterInjConvHalfCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
void HAL_DFSDM_FilterAwdCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter, uint32_t Channel, uint32_t Threshold);
void HAL_DFSDM_FilterErrorCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
/**
  * @}
  */

/** @addtogroup DFSDM_Exported_Functions_Group4_Filter Filter state functions
  * @{
  */
/* Filter state functions *****************************************************/
HAL_DFSDM_Filter_StateTypeDef HAL_DFSDM_FilterGetState(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
uint32_t                      HAL_DFSDM_FilterGetError(DFSDM_Filter_HandleTypeDef *hdfsdm_filter);
/**
  * @}
  */
/** @addtogroup DFSDM_Exported_Functions_Group5_Filter MultiChannel operation functions
  * @{
  */
#if defined(SYSCFG_MCHDLYCR_BSCKSEL)
void HAL_DFSDM_ConfigMultiChannelDelay(DFSDM_MultiChannelConfigTypeDef* mchdlystruct);
void HAL_DFSDM_BitstreamClock_Start(void);
void HAL_DFSDM_BitstreamClock_Stop(void);
void HAL_DFSDM_DisableDelayClock(uint32_t MCHDLY);
void HAL_DFSDM_EnableDelayClock(uint32_t MCHDLY);
void HAL_DFSDM_ClockIn_SourceSelection(uint32_t source);
void HAL_DFSDM_ClockOut_SourceSelection(uint32_t source);
void HAL_DFSDM_DataIn0_SourceSelection(uint32_t source);
void HAL_DFSDM_DataIn2_SourceSelection(uint32_t source);
void HAL_DFSDM_DataIn4_SourceSelection(uint32_t source);
void HAL_DFSDM_DataIn6_SourceSelection(uint32_t source);
void HAL_DFSDM_BitStreamClkDistribution_Config(uint32_t source);
#endif /* SYSCFG_MCHDLYCR_BSCKSEL */
/**
  * @}
  */
/**
  * @}
  */
/* End of exported functions -------------------------------------------------*/

/* Private macros ------------------------------------------------------------*/
/** @defgroup DFSDM_Private_Macros DFSDM Private Macros
* @{
*/
#define IS_DFSDM_CHANNEL_OUTPUT_CLOCK(CLOCK)          (((CLOCK) == DFSDM_CHANNEL_OUTPUT_CLOCK_SYSTEM) || \
                                                       ((CLOCK) == DFSDM_CHANNEL_OUTPUT_CLOCK_AUDIO))
#define IS_DFSDM_CHANNEL_OUTPUT_CLOCK_DIVIDER(DIVIDER) ((2U <= (DIVIDER)) && ((DIVIDER) <= 256U))
#define IS_DFSDM_CHANNEL_INPUT(INPUT)                 (((INPUT) == DFSDM_CHANNEL_EXTERNAL_INPUTS) || \
                                                       ((INPUT) == DFSDM_CHANNEL_INTERNAL_REGISTER))
#define IS_DFSDM_CHANNEL_DATA_PACKING(MODE)           (((MODE) == DFSDM_CHANNEL_STANDARD_MODE) || \
                                                       ((MODE) == DFSDM_CHANNEL_INTERLEAVED_MODE) || \
                                                       ((MODE) == DFSDM_CHANNEL_DUAL_MODE))
#define IS_DFSDM_CHANNEL_INPUT_PINS(PINS)             (((PINS) == DFSDM_CHANNEL_SAME_CHANNEL_PINS) || \
                                                       ((PINS) == DFSDM_CHANNEL_FOLLOWING_CHANNEL_PINS))
#define IS_DFSDM_CHANNEL_SERIAL_INTERFACE_TYPE(MODE)  (((MODE) == DFSDM_CHANNEL_SPI_RISING) || \
                                                       ((MODE) == DFSDM_CHANNEL_SPI_FALLING) || \
                                                       ((MODE) == DFSDM_CHANNEL_MANCHESTER_RISING) || \
                                                       ((MODE) == DFSDM_CHANNEL_MANCHESTER_FALLING))
#define IS_DFSDM_CHANNEL_SPI_CLOCK(TYPE)              (((TYPE) == DFSDM_CHANNEL_SPI_CLOCK_EXTERNAL) || \
                                                       ((TYPE) == DFSDM_CHANNEL_SPI_CLOCK_INTERNAL) || \
                                                       ((TYPE) == DFSDM_CHANNEL_SPI_CLOCK_INTERNAL_DIV2_FALLING) || \
                                                       ((TYPE) == DFSDM_CHANNEL_SPI_CLOCK_INTERNAL_DIV2_RISING))
#define IS_DFSDM_CHANNEL_FILTER_ORDER(ORDER)          (((ORDER) == DFSDM_CHANNEL_FASTSINC_ORDER) || \
                                                       ((ORDER) == DFSDM_CHANNEL_SINC1_ORDER) || \
                                                       ((ORDER) == DFSDM_CHANNEL_SINC2_ORDER) || \
                                                       ((ORDER) == DFSDM_CHANNEL_SINC3_ORDER))
#define IS_DFSDM_CHANNEL_FILTER_OVS_RATIO(RATIO)       ((1U <= (RATIO)) && ((RATIO) <= 32U))
#define IS_DFSDM_CHANNEL_OFFSET(VALUE)                 ((-8388608 <= (VALUE)) && ((VALUE) <= 8388607))
#define IS_DFSDM_CHANNEL_RIGHT_BIT_SHIFT(VALUE)        ((VALUE) <= 0x1FU)
#define IS_DFSDM_CHANNEL_SCD_THRESHOLD(VALUE)          ((VALUE) <= 0xFFU)
#define IS_DFSDM_FILTER_REG_TRIGGER(TRIG)             (((TRIG) == DFSDM_FILTER_SW_TRIGGER) || \
                                                       ((TRIG) == DFSDM_FILTER_SYNC_TRIGGER))
#define IS_DFSDM_FILTER_INJ_TRIGGER(TRIG)             (((TRIG) == DFSDM_FILTER_SW_TRIGGER) || \
                                                       ((TRIG) == DFSDM_FILTER_SYNC_TRIGGER) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIGGER))
#if defined (STM32F413xx) || defined (STM32F423xx)
#define IS_DFSDM_FILTER_EXT_TRIG(TRIG)                (((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM1_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM3_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM8_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM10_OC1) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM2_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM4_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM11_OC1) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM6_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_EXTI11)    || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_EXTI15))
#define IS_DFSDM_DELAY_CLOCK(CLOCK)                     (((CLOCK) == HAL_MCHDLY_CLOCK_DFSDM2) || \
                                                         ((CLOCK) == HAL_MCHDLY_CLOCK_DFSDM1)) 
#else
#define IS_DFSDM_FILTER_EXT_TRIG(TRIG)                (((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM1_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM3_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM8_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM10_OC1) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM4_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_TIM6_TRGO) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_EXTI11) || \
                                                       ((TRIG) == DFSDM_FILTER_EXT_TRIG_EXTI15))
#endif
#define IS_DFSDM_FILTER_EXT_TRIG_EDGE(EDGE)           (((EDGE) == DFSDM_FILTER_EXT_TRIG_RISING_EDGE)  || \
                                                       ((EDGE) == DFSDM_FILTER_EXT_TRIG_FALLING_EDGE)  || \
                                                       ((EDGE) == DFSDM_FILTER_EXT_TRIG_BOTH_EDGES))
#define IS_DFSDM_FILTER_SINC_ORDER(ORDER)             (((ORDER) == DFSDM_FILTER_FASTSINC_ORDER) || \
                                                       ((ORDER) == DFSDM_FILTER_SINC1_ORDER) || \
                                                       ((ORDER) == DFSDM_FILTER_SINC2_ORDER) || \
                                                       ((ORDER) == DFSDM_FILTER_SINC3_ORDER) || \
                                                       ((ORDER) == DFSDM_FILTER_SINC4_ORDER) || \
                                                       ((ORDER) == DFSDM_FILTER_SINC5_ORDER))
#define IS_DFSDM_FILTER_OVS_RATIO(RATIO)               ((1U <= (RATIO)) && ((RATIO) <= 1024U))
#define IS_DFSDM_FILTER_INTEGRATOR_OVS_RATIO(RATIO)    ((1U <= (RATIO)) && ((RATIO) <= 256U))
#define IS_DFSDM_FILTER_AWD_DATA_SOURCE(DATA)         (((DATA) == DFSDM_FILTER_AWD_FILTER_DATA)  || \
                                                       ((DATA) == DFSDM_FILTER_AWD_CHANNEL_DATA))
#define IS_DFSDM_FILTER_AWD_THRESHOLD(VALUE)           ((-8388608 <= (VALUE)) && ((VALUE) <= 8388607))
#define IS_DFSDM_BREAK_SIGNALS(VALUE)                  ((VALUE) <= 0x0FU)
#if defined(DFSDM2_Channel0)
#define IS_DFSDM_REGULAR_CHANNEL(CHANNEL)             (((CHANNEL) == DFSDM_CHANNEL_0)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_1)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_2)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_3)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_4)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_5)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_6)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_7))
#define IS_DFSDM_INJECTED_CHANNEL(CHANNEL)            (((CHANNEL) != 0U) && ((CHANNEL) <= 0x000F00FFU))
#else
#define IS_DFSDM_REGULAR_CHANNEL(CHANNEL)             (((CHANNEL) == DFSDM_CHANNEL_0)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_1)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_2)  || \
                                                       ((CHANNEL) == DFSDM_CHANNEL_3))
#define IS_DFSDM_INJECTED_CHANNEL(CHANNEL)            (((CHANNEL) != 0U) && ((CHANNEL) <= 0x0003000FU))
#endif
#define IS_DFSDM_CONTINUOUS_MODE(MODE)                (((MODE) == DFSDM_CONTINUOUS_CONV_OFF)  || \
                                                       ((MODE) == DFSDM_CONTINUOUS_CONV_ON))
#if defined(DFSDM2_Channel0)
#define IS_DFSDM1_CHANNEL_INSTANCE(INSTANCE)          (((INSTANCE) == DFSDM1_Channel0) || \
                                                       ((INSTANCE) == DFSDM1_Channel1) || \
                                                       ((INSTANCE) == DFSDM1_Channel2) || \
                                                       ((INSTANCE) == DFSDM1_Channel3))
#define IS_DFSDM1_FILTER_INSTANCE(INSTANCE)          (((INSTANCE) == DFSDM1_Filter0) || \
                                                      ((INSTANCE) == DFSDM1_Filter1)) 
#endif /* DFSDM2_Channel0 */

#if defined(SYSCFG_MCHDLYCR_BSCKSEL)
#define IS_DFSDM_CLOCKIN_SELECTION(SELECTION)           (((SELECTION) == HAL_DFSDM2_CKIN_PAD) || \
                                                          ((SELECTION) == HAL_DFSDM2_CKIN_DM)  || \
                                                          ((SELECTION) == HAL_DFSDM1_CKIN_PAD) || \
                                                          ((SELECTION) == HAL_DFSDM1_CKIN_DM)) 
#define IS_DFSDM_CLOCKOUT_SELECTION(SELECTION)         (((SELECTION) == HAL_DFSDM2_CKOUT_DFSDM2) || \
                                                          ((SELECTION) == HAL_DFSDM2_CKOUT_M27)    || \
                                                          ((SELECTION) == HAL_DFSDM1_CKOUT_DFSDM1) || \
                                                          ((SELECTION) == HAL_DFSDM1_CKOUT_M27)) 
#define IS_DFSDM_DATAIN0_SRC_SELECTION(SELECTION)        (((SELECTION) == HAL_DATAIN0_DFSDM2_PAD)    || \
                                                          ((SELECTION) == HAL_DATAIN0_DFSDM2_DATAIN1) || \
                                                          ((SELECTION) == HAL_DATAIN0_DFSDM1_PAD)     || \
                                                          ((SELECTION) == HAL_DATAIN0_DFSDM1_DATAIN1))
#define IS_DFSDM_DATAIN2_SRC_SELECTION(SELECTION)        (((SELECTION) == HAL_DATAIN2_DFSDM2_PAD)    || \
                                                          ((SELECTION) == HAL_DATAIN2_DFSDM2_DATAIN3) || \
                                                          ((SELECTION) == HAL_DATAIN2_DFSDM1_PAD)     || \
                                                          ((SELECTION) == HAL_DATAIN2_DFSDM1_DATAIN3))
#define IS_DFSDM_DATAIN4_SRC_SELECTION(SELECTION)        (((SELECTION) == HAL_DATAIN4_DFSDM2_PAD)    || \
                                                          ((SELECTION) == HAL_DATAIN4_DFSDM2_DATAIN5))
#define IS_DFSDM_DATAIN6_SRC_SELECTION(SELECTION)         (((SELECTION) == HAL_DATAIN6_DFSDM2_PAD)    || \
                                                          ((SELECTION) == HAL_DATAIN6_DFSDM2_DATAIN7))
#define IS_DFSDM_BITSTREM_CLK_DISTRIBUTION(DISTRIBUTION) (((DISTRIBUTION) == HAL_DFSDM1_CLKIN0_TIM4OC2) || \
                                                          ((DISTRIBUTION) == HAL_DFSDM1_CLKIN2_TIM4OC2) || \
                                                          ((DISTRIBUTION) == HAL_DFSDM1_CLKIN1_TIM4OC1) || \
                                                          ((DISTRIBUTION) == HAL_DFSDM1_CLKIN3_TIM4OC1) || \
                                                          ((DISTRIBUTION) == HAL_DFSDM2_CLKIN0_TIM3OC4) || \
                                                          ((DISTRIBUTION) == HAL_DFSDM2_CLKIN4_TIM3OC4) || \
                                                          ((DISTRIBUTION) == HAL_DFSDM2_CLKIN1_TIM3OC3)|| \
                                                          ((DISTRIBUTION) == HAL_DFSDM2_CLKIN5_TIM3OC3) || \
                                                          ((DISTRIBUTION) == HAL_DFSDM2_CLKIN2_TIM3OC2) || \
                                                          ((DISTRIBUTION) == HAL_DFSDM2_CLKIN6_TIM3OC2) || \
                                                          ((DISTRIBUTION) == HAL_DFSDM2_CLKIN3_TIM3OC1)|| \
                                                          ((DISTRIBUTION) == HAL_DFSDM2_CLKIN7_TIM3OC1))
#define IS_DFSDM_DFSDM1_CLKOUT(CLKOUT)                 (((CLKOUT) == DFSDM1_CKOUT_DFSDM2_CKOUT)    || \
                                                        ((CLKOUT) == DFSDM1_CKOUT_DFSDM1))
#define IS_DFSDM_DFSDM2_CLKOUT(CLKOUT)                 (((CLKOUT) == DFSDM2_CKOUT_DFSDM2_CKOUT)    || \
                                                        ((CLKOUT) == DFSDM2_CKOUT_DFSDM2))   
#define IS_DFSDM_DFSDM1_CLKIN(CLKIN)                   (((CLKIN) == DFSDM1_CKIN_DFSDM2_CKOUT)    || \
                                                        ((CLKIN) == DFSDM1_CKIN_PAD))                                   
#define IS_DFSDM_DFSDM2_CLKIN(CLKIN)                   (((CLKIN) == DFSDM2_CKIN_DFSDM2_CKOUT)    || \
                                                        ((CLKIN) == DFSDM2_CKIN_PAD))
#define IS_DFSDM_DFSDM1_BIT_CLK(CLK)                   (((CLK) == DFSDM1_T4_OC2_BITSTREAM_CKIN0)  || \
                                                        ((CLK) == DFSDM1_T4_OC2_BITSTREAM_CKIN2)  || \
                                                        ((CLK) == DFSDM1_T4_OC1_BITSTREAM_CKIN3)  || \
                                                        ((CLK) == DFSDM1_T4_OC1_BITSTREAM_CKIN1)  || \
                                                        ((CLK) <= 0x30U)) 

#define IS_DFSDM_DFSDM2_BIT_CLK(CLK)                   (((CLK) == DFSDM2_T3_OC4_BITSTREAM_CKIN0)  || \
                                                        ((CLK) == DFSDM2_T3_OC4_BITSTREAM_CKIN4)  || \
                                                        ((CLK) == DFSDM2_T3_OC3_BITSTREAM_CKIN5)  || \
                                                        ((CLK) == DFSDM2_T3_OC3_BITSTREAM_CKIN1)  || \
                                                        ((CLK) == DFSDM2_T3_OC2_BITSTREAM_CKIN6)  || \
                                                        ((CLK) == DFSDM2_T3_OC2_BITSTREAM_CKIN2)  || \
                                                        ((CLK) == DFSDM2_T3_OC1_BITSTREAM_CKIN3)  || \
                                                        ((CLK) == DFSDM2_T3_OC1_BITSTREAM_CKIN7)|| \
                                                        ((CLK) <= 0x1E000U)) 

#define IS_DFSDM_DFSDM1_DATA_DISTRIBUTION(DISTRIBUTION)(((DISTRIBUTION) == DFSDM1_DATIN0_TO_DATIN0_PAD )|| \
                                                        ((DISTRIBUTION) == DFSDM1_DATIN0_TO_DATIN1_PAD) || \
                                                        ((DISTRIBUTION) == DFSDM1_DATIN2_TO_DATIN2_PAD) || \
                                                        ((DISTRIBUTION) == DFSDM1_DATIN2_TO_DATIN3_PAD)|| \
                                                        ((DISTRIBUTION) <= 0xCU)) 
                                                      
#define IS_DFSDM_DFSDM2_DATA_DISTRIBUTION(DISTRIBUTION)(((DISTRIBUTION) == DFSDM2_DATIN0_TO_DATIN0_PAD)|| \
                                                        ((DISTRIBUTION) == DFSDM2_DATIN0_TO_DATIN1_PAD)|| \
                                                        ((DISTRIBUTION) == DFSDM2_DATIN2_TO_DATIN2_PAD)|| \
                                                        ((DISTRIBUTION) == DFSDM2_DATIN2_TO_DATIN3_PAD)|| \
                                                        ((DISTRIBUTION) == DFSDM2_DATIN4_TO_DATIN4_PAD)|| \
                                                        ((DISTRIBUTION) == DFSDM2_DATIN4_TO_DATIN5_PAD)|| \
                                                        ((DISTRIBUTION) == DFSDM2_DATIN6_TO_DATIN6_PAD)|| \
                                                        ((DISTRIBUTION) == DFSDM2_DATIN6_TO_DATIN7_PAD)|| \
                                                        ((DISTRIBUTION) <= 0x1D00U))                                                                                       
#endif /* (SYSCFG_MCHDLYCR_BSCKSEL) */
/**
  * @}
  */ 
/* End of private macros -----------------------------------------------------*/

/**
  * @}
  */ 

/**
  * @}
  */
#endif /* STM32F412Zx || STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */
#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_DFSDM_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
