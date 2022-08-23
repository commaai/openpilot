/**
  ******************************************************************************
  * @file    stm32f4xx_hal_dma2d.h
  * @author  MCD Application Team
  * @brief   Header file of DMA2D HAL module.
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
#ifndef STM32F4xx_HAL_DMA2D_H
#define STM32F4xx_HAL_DMA2D_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

#if defined (DMA2D)

/** @addtogroup DMA2D DMA2D
  * @brief DMA2D HAL module driver
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup DMA2D_Exported_Types DMA2D Exported Types
  * @{
  */
#define MAX_DMA2D_LAYER  2U  /*!< DMA2D maximum number of layers */

/**
  * @brief DMA2D CLUT Structure definition
  */
typedef struct
{
  uint32_t *pCLUT;                  /*!< Configures the DMA2D CLUT memory address.*/

  uint32_t CLUTColorMode;           /*!< Configures the DMA2D CLUT color mode.
                                         This parameter can be one value of @ref DMA2D_CLUT_CM. */

  uint32_t Size;                    /*!< Configures the DMA2D CLUT size.
                                         This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF.*/
} DMA2D_CLUTCfgTypeDef;

/**
  * @brief DMA2D Init structure definition
  */
typedef struct
{
  uint32_t             Mode;               /*!< Configures the DMA2D transfer mode.
                                                This parameter can be one value of @ref DMA2D_Mode. */

  uint32_t             ColorMode;          /*!< Configures the color format of the output image.
                                                This parameter can be one value of @ref DMA2D_Output_Color_Mode. */

  uint32_t             OutputOffset;       /*!< Specifies the Offset value.
                                                This parameter must be a number between
                                                Min_Data = 0x0000 and Max_Data = 0x3FFF. */




} DMA2D_InitTypeDef;


/**
  * @brief DMA2D Layer structure definition
  */
typedef struct
{
  uint32_t             InputOffset;       /*!< Configures the DMA2D foreground or background offset.
                                               This parameter must be a number between
                                               Min_Data = 0x0000 and Max_Data = 0x3FFF. */

  uint32_t             InputColorMode;    /*!< Configures the DMA2D foreground or background color mode.
                                               This parameter can be one value of @ref DMA2D_Input_Color_Mode. */

  uint32_t             AlphaMode;         /*!< Configures the DMA2D foreground or background alpha mode.
                                               This parameter can be one value of @ref DMA2D_Alpha_Mode. */

  uint32_t             InputAlpha;        /*!< Specifies the DMA2D foreground or background alpha value and color value
                                               in case of A8 or A4 color mode.
                                               This parameter must be a number between Min_Data = 0x00
                                               and Max_Data = 0xFF except for the color modes detailed below.
                                               @note In case of A8 or A4 color mode (ARGB),
                                               this parameter must be a number between
                                               Min_Data = 0x00000000 and Max_Data = 0xFFFFFFFF where
                                               - InputAlpha[24:31] is the alpha value ALPHA[0:7]
                                               - InputAlpha[16:23] is the red value RED[0:7]
                                               - InputAlpha[8:15] is the green value GREEN[0:7]
                                               - InputAlpha[0:7] is the blue value BLUE[0:7]. */


} DMA2D_LayerCfgTypeDef;

/**
  * @brief  HAL DMA2D State structures definition
  */
typedef enum
{
  HAL_DMA2D_STATE_RESET             = 0x00U,    /*!< DMA2D not yet initialized or disabled       */
  HAL_DMA2D_STATE_READY             = 0x01U,    /*!< Peripheral Initialized and ready for use    */
  HAL_DMA2D_STATE_BUSY              = 0x02U,    /*!< An internal process is ongoing              */
  HAL_DMA2D_STATE_TIMEOUT           = 0x03U,    /*!< Timeout state                               */
  HAL_DMA2D_STATE_ERROR             = 0x04U,    /*!< DMA2D state error                           */
  HAL_DMA2D_STATE_SUSPEND           = 0x05U     /*!< DMA2D process is suspended                  */
} HAL_DMA2D_StateTypeDef;

/**
  * @brief  DMA2D handle Structure definition
  */
typedef struct __DMA2D_HandleTypeDef
{
  DMA2D_TypeDef               *Instance;                                  /*!< DMA2D register base address.           */

  DMA2D_InitTypeDef           Init;                                       /*!< DMA2D communication parameters.        */

  void (* XferCpltCallback)(struct __DMA2D_HandleTypeDef *hdma2d);        /*!< DMA2D transfer complete callback.      */

  void (* XferErrorCallback)(struct __DMA2D_HandleTypeDef *hdma2d);       /*!< DMA2D transfer error callback.         */

#if (USE_HAL_DMA2D_REGISTER_CALLBACKS == 1)
  void (* LineEventCallback)(struct __DMA2D_HandleTypeDef *hdma2d);       /*!< DMA2D line event callback.             */

  void (* CLUTLoadingCpltCallback)(struct __DMA2D_HandleTypeDef *hdma2d); /*!< DMA2D CLUT loading completion callback */

  void (* MspInitCallback)(struct __DMA2D_HandleTypeDef *hdma2d);         /*!< DMA2D Msp Init callback.               */

  void (* MspDeInitCallback)(struct __DMA2D_HandleTypeDef *hdma2d);       /*!< DMA2D Msp DeInit callback.             */

#endif /* (USE_HAL_DMA2D_REGISTER_CALLBACKS) */

  DMA2D_LayerCfgTypeDef       LayerCfg[MAX_DMA2D_LAYER];                  /*!< DMA2D Layers parameters                */

  HAL_LockTypeDef             Lock;                                       /*!< DMA2D lock.                            */

  __IO HAL_DMA2D_StateTypeDef State;                                      /*!< DMA2D transfer state.                  */

  __IO uint32_t               ErrorCode;                                  /*!< DMA2D error code.                      */
} DMA2D_HandleTypeDef;

#if (USE_HAL_DMA2D_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL DMA2D Callback pointer definition
  */
typedef void (*pDMA2D_CallbackTypeDef)(DMA2D_HandleTypeDef *hdma2d); /*!< Pointer to a DMA2D common callback function */
#endif /* USE_HAL_DMA2D_REGISTER_CALLBACKS */
/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup DMA2D_Exported_Constants DMA2D Exported Constants
  * @{
  */

/** @defgroup DMA2D_Error_Code DMA2D Error Code
  * @{
  */
#define HAL_DMA2D_ERROR_NONE        0x00000000U  /*!< No error             */
#define HAL_DMA2D_ERROR_TE          0x00000001U  /*!< Transfer error       */
#define HAL_DMA2D_ERROR_CE          0x00000002U  /*!< Configuration error  */
#define HAL_DMA2D_ERROR_CAE         0x00000004U  /*!< CLUT access error    */
#define HAL_DMA2D_ERROR_TIMEOUT     0x00000020U  /*!< Timeout error        */
#if (USE_HAL_DMA2D_REGISTER_CALLBACKS == 1)
#define HAL_DMA2D_ERROR_INVALID_CALLBACK 0x00000040U  /*!< Invalid callback error  */
#endif /* USE_HAL_UART_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup DMA2D_Mode DMA2D Mode
  * @{
  */
#define DMA2D_M2M                   0x00000000U                         /*!< DMA2D memory to memory transfer mode */
#define DMA2D_M2M_PFC               DMA2D_CR_MODE_0                     /*!< DMA2D memory to memory with pixel format conversion transfer mode */
#define DMA2D_M2M_BLEND             DMA2D_CR_MODE_1                     /*!< DMA2D memory to memory with blending transfer mode */
#define DMA2D_R2M                   DMA2D_CR_MODE            /*!< DMA2D register to memory transfer mode */
/**
  * @}
  */

/** @defgroup DMA2D_Output_Color_Mode DMA2D Output Color Mode
  * @{
  */
#define DMA2D_OUTPUT_ARGB8888       0x00000000U                           /*!< ARGB8888 DMA2D color mode */
#define DMA2D_OUTPUT_RGB888         DMA2D_OPFCCR_CM_0                     /*!< RGB888 DMA2D color mode   */
#define DMA2D_OUTPUT_RGB565         DMA2D_OPFCCR_CM_1                     /*!< RGB565 DMA2D color mode   */
#define DMA2D_OUTPUT_ARGB1555       (DMA2D_OPFCCR_CM_0|DMA2D_OPFCCR_CM_1) /*!< ARGB1555 DMA2D color mode */
#define DMA2D_OUTPUT_ARGB4444       DMA2D_OPFCCR_CM_2                     /*!< ARGB4444 DMA2D color mode */
/**
  * @}
  */

/** @defgroup DMA2D_Input_Color_Mode DMA2D Input Color Mode
  * @{
  */
#define DMA2D_INPUT_ARGB8888        0x00000000U  /*!< ARGB8888 color mode */
#define DMA2D_INPUT_RGB888          0x00000001U  /*!< RGB888 color mode   */
#define DMA2D_INPUT_RGB565          0x00000002U  /*!< RGB565 color mode   */
#define DMA2D_INPUT_ARGB1555        0x00000003U  /*!< ARGB1555 color mode */
#define DMA2D_INPUT_ARGB4444        0x00000004U  /*!< ARGB4444 color mode */
#define DMA2D_INPUT_L8              0x00000005U  /*!< L8 color mode       */
#define DMA2D_INPUT_AL44            0x00000006U  /*!< AL44 color mode     */
#define DMA2D_INPUT_AL88            0x00000007U  /*!< AL88 color mode     */
#define DMA2D_INPUT_L4              0x00000008U  /*!< L4 color mode       */
#define DMA2D_INPUT_A8              0x00000009U  /*!< A8 color mode       */
#define DMA2D_INPUT_A4              0x0000000AU  /*!< A4 color mode       */
/**
  * @}
  */

/** @defgroup DMA2D_Alpha_Mode DMA2D Alpha Mode
  * @{
  */
#define DMA2D_NO_MODIF_ALPHA        0x00000000U  /*!< No modification of the alpha channel value                     */
#define DMA2D_REPLACE_ALPHA         0x00000001U  /*!< Replace original alpha channel value by programmed alpha value */
#define DMA2D_COMBINE_ALPHA         0x00000002U  /*!< Replace original alpha channel value by programmed alpha value
                                                      with original alpha channel value                              */
/**
  * @}
  */






/** @defgroup DMA2D_CLUT_CM DMA2D CLUT Color Mode
  * @{
  */
#define DMA2D_CCM_ARGB8888          0x00000000U  /*!< ARGB8888 DMA2D CLUT color mode */
#define DMA2D_CCM_RGB888            0x00000001U  /*!< RGB888 DMA2D CLUT color mode   */
/**
  * @}
  */

/** @defgroup DMA2D_Interrupts DMA2D Interrupts
  * @{
  */
#define DMA2D_IT_CE                 DMA2D_CR_CEIE            /*!< Configuration Error Interrupt */
#define DMA2D_IT_CTC                DMA2D_CR_CTCIE           /*!< CLUT Transfer Complete Interrupt */
#define DMA2D_IT_CAE                DMA2D_CR_CAEIE           /*!< CLUT Access Error Interrupt */
#define DMA2D_IT_TW                 DMA2D_CR_TWIE            /*!< Transfer Watermark Interrupt */
#define DMA2D_IT_TC                 DMA2D_CR_TCIE            /*!< Transfer Complete Interrupt */
#define DMA2D_IT_TE                 DMA2D_CR_TEIE            /*!< Transfer Error Interrupt */
/**
  * @}
  */

/** @defgroup DMA2D_Flags DMA2D Flags
  * @{
  */
#define DMA2D_FLAG_CE               DMA2D_ISR_CEIF           /*!< Configuration Error Interrupt Flag */
#define DMA2D_FLAG_CTC              DMA2D_ISR_CTCIF          /*!< CLUT Transfer Complete Interrupt Flag */
#define DMA2D_FLAG_CAE              DMA2D_ISR_CAEIF          /*!< CLUT Access Error Interrupt Flag */
#define DMA2D_FLAG_TW               DMA2D_ISR_TWIF           /*!< Transfer Watermark Interrupt Flag */
#define DMA2D_FLAG_TC               DMA2D_ISR_TCIF           /*!< Transfer Complete Interrupt Flag */
#define DMA2D_FLAG_TE               DMA2D_ISR_TEIF           /*!< Transfer Error Interrupt Flag */
/**
  * @}
  */

/** @defgroup DMA2D_Aliases DMA2D API Aliases
  * @{
  */
#define HAL_DMA2D_DisableCLUT       HAL_DMA2D_CLUTLoading_Abort    /*!< Aliased to HAL_DMA2D_CLUTLoading_Abort 
                                                                        for compatibility with legacy code */
/**
  * @}
  */

#if (USE_HAL_DMA2D_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL DMA2D common Callback ID enumeration definition
  */
typedef enum
{
  HAL_DMA2D_MSPINIT_CB_ID           = 0x00U,    /*!< DMA2D MspInit callback ID                 */
  HAL_DMA2D_MSPDEINIT_CB_ID         = 0x01U,    /*!< DMA2D MspDeInit callback ID               */
  HAL_DMA2D_TRANSFERCOMPLETE_CB_ID  = 0x02U,    /*!< DMA2D transfer complete callback ID       */
  HAL_DMA2D_TRANSFERERROR_CB_ID     = 0x03U,    /*!< DMA2D transfer error callback ID          */
  HAL_DMA2D_LINEEVENT_CB_ID         = 0x04U,    /*!< DMA2D line event callback ID              */
  HAL_DMA2D_CLUTLOADINGCPLT_CB_ID   = 0x05U,    /*!< DMA2D CLUT loading completion callback ID */
} HAL_DMA2D_CallbackIDTypeDef;
#endif /* USE_HAL_DMA2D_REGISTER_CALLBACKS */


/**
  * @}
  */
/* Exported macros ------------------------------------------------------------*/
/** @defgroup DMA2D_Exported_Macros DMA2D Exported Macros
  * @{
  */

/** @brief Reset DMA2D handle state
  * @param  __HANDLE__ specifies the DMA2D handle.
  * @retval None
  */
#if (USE_HAL_DMA2D_REGISTER_CALLBACKS == 1)
#define __HAL_DMA2D_RESET_HANDLE_STATE(__HANDLE__) do{                                             \
                                                       (__HANDLE__)->State = HAL_DMA2D_STATE_RESET;\
                                                       (__HANDLE__)->MspInitCallback = NULL;       \
                                                       (__HANDLE__)->MspDeInitCallback = NULL;     \
                                                     }while(0)
#else
#define __HAL_DMA2D_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_DMA2D_STATE_RESET)
#endif /* USE_HAL_DMA2D_REGISTER_CALLBACKS */


/**
  * @brief  Enable the DMA2D.
  * @param  __HANDLE__ DMA2D handle
  * @retval None.
  */
#define __HAL_DMA2D_ENABLE(__HANDLE__)        ((__HANDLE__)->Instance->CR |= DMA2D_CR_START)


/* Interrupt & Flag management */
/**
  * @brief  Get the DMA2D pending flags.
  * @param  __HANDLE__ DMA2D handle
  * @param  __FLAG__ flag to check.
  *          This parameter can be any combination of the following values:
  *            @arg DMA2D_FLAG_CE:  Configuration error flag
  *            @arg DMA2D_FLAG_CTC: CLUT transfer complete flag
  *            @arg DMA2D_FLAG_CAE: CLUT access error flag
  *            @arg DMA2D_FLAG_TW:  Transfer Watermark flag
  *            @arg DMA2D_FLAG_TC:  Transfer complete flag
  *            @arg DMA2D_FLAG_TE:  Transfer error flag
  * @retval The state of FLAG.
  */
#define __HAL_DMA2D_GET_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->ISR & (__FLAG__))

/**
  * @brief  Clear the DMA2D pending flags.
  * @param  __HANDLE__ DMA2D handle
  * @param  __FLAG__ specifies the flag to clear.
  *          This parameter can be any combination of the following values:
  *            @arg DMA2D_FLAG_CE:  Configuration error flag
  *            @arg DMA2D_FLAG_CTC: CLUT transfer complete flag
  *            @arg DMA2D_FLAG_CAE: CLUT access error flag
  *            @arg DMA2D_FLAG_TW:  Transfer Watermark flag
  *            @arg DMA2D_FLAG_TC:  Transfer complete flag
  *            @arg DMA2D_FLAG_TE:  Transfer error flag
  * @retval None
  */
#define __HAL_DMA2D_CLEAR_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->IFCR = (__FLAG__))

/**
  * @brief  Enable the specified DMA2D interrupts.
  * @param  __HANDLE__ DMA2D handle
  * @param __INTERRUPT__ specifies the DMA2D interrupt sources to be enabled.
  *          This parameter can be any combination of the following values:
  *            @arg DMA2D_IT_CE:  Configuration error interrupt mask
  *            @arg DMA2D_IT_CTC: CLUT transfer complete interrupt mask
  *            @arg DMA2D_IT_CAE: CLUT access error interrupt mask
  *            @arg DMA2D_IT_TW:  Transfer Watermark interrupt mask
  *            @arg DMA2D_IT_TC:  Transfer complete interrupt mask
  *            @arg DMA2D_IT_TE:  Transfer error interrupt mask
  * @retval None
  */
#define __HAL_DMA2D_ENABLE_IT(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->CR |= (__INTERRUPT__))

/**
  * @brief  Disable the specified DMA2D interrupts.
  * @param  __HANDLE__ DMA2D handle
  * @param __INTERRUPT__ specifies the DMA2D interrupt sources to be disabled.
  *          This parameter can be any combination of the following values:
  *            @arg DMA2D_IT_CE:  Configuration error interrupt mask
  *            @arg DMA2D_IT_CTC: CLUT transfer complete interrupt mask
  *            @arg DMA2D_IT_CAE: CLUT access error interrupt mask
  *            @arg DMA2D_IT_TW:  Transfer Watermark interrupt mask
  *            @arg DMA2D_IT_TC:  Transfer complete interrupt mask
  *            @arg DMA2D_IT_TE:  Transfer error interrupt mask
  * @retval None
  */
#define __HAL_DMA2D_DISABLE_IT(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->CR &= ~(__INTERRUPT__))

/**
  * @brief  Check whether the specified DMA2D interrupt source is enabled or not.
  * @param  __HANDLE__ DMA2D handle
  * @param  __INTERRUPT__ specifies the DMA2D interrupt source to check.
  *          This parameter can be one of the following values:
  *            @arg DMA2D_IT_CE:  Configuration error interrupt mask
  *            @arg DMA2D_IT_CTC: CLUT transfer complete interrupt mask
  *            @arg DMA2D_IT_CAE: CLUT access error interrupt mask
  *            @arg DMA2D_IT_TW:  Transfer Watermark interrupt mask
  *            @arg DMA2D_IT_TC:  Transfer complete interrupt mask
  *            @arg DMA2D_IT_TE:  Transfer error interrupt mask
  * @retval The state of INTERRUPT source.
  */
#define __HAL_DMA2D_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->CR & (__INTERRUPT__))

/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup DMA2D_Exported_Functions DMA2D Exported Functions
  * @{
  */

/** @addtogroup DMA2D_Exported_Functions_Group1 Initialization and de-initialization functions
  * @{
  */

/* Initialization and de-initialization functions *******************************/
HAL_StatusTypeDef HAL_DMA2D_Init(DMA2D_HandleTypeDef *hdma2d);
HAL_StatusTypeDef HAL_DMA2D_DeInit(DMA2D_HandleTypeDef *hdma2d);
void              HAL_DMA2D_MspInit(DMA2D_HandleTypeDef *hdma2d);
void              HAL_DMA2D_MspDeInit(DMA2D_HandleTypeDef *hdma2d);
/* Callbacks Register/UnRegister functions  ***********************************/
#if (USE_HAL_DMA2D_REGISTER_CALLBACKS == 1)
HAL_StatusTypeDef HAL_DMA2D_RegisterCallback(DMA2D_HandleTypeDef *hdma2d, HAL_DMA2D_CallbackIDTypeDef CallbackID,
                                             pDMA2D_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_DMA2D_UnRegisterCallback(DMA2D_HandleTypeDef *hdma2d, HAL_DMA2D_CallbackIDTypeDef CallbackID);
#endif /* USE_HAL_DMA2D_REGISTER_CALLBACKS */

/**
  * @}
  */


/** @addtogroup DMA2D_Exported_Functions_Group2 IO operation functions
  * @{
  */

/* IO operation functions *******************************************************/
HAL_StatusTypeDef HAL_DMA2D_Start(DMA2D_HandleTypeDef *hdma2d, uint32_t pdata, uint32_t DstAddress, uint32_t Width,
                                  uint32_t Height);
HAL_StatusTypeDef HAL_DMA2D_BlendingStart(DMA2D_HandleTypeDef *hdma2d, uint32_t SrcAddress1, uint32_t SrcAddress2,
                                          uint32_t DstAddress, uint32_t Width,  uint32_t Height);
HAL_StatusTypeDef HAL_DMA2D_Start_IT(DMA2D_HandleTypeDef *hdma2d, uint32_t pdata, uint32_t DstAddress, uint32_t Width,
                                     uint32_t Height);
HAL_StatusTypeDef HAL_DMA2D_BlendingStart_IT(DMA2D_HandleTypeDef *hdma2d, uint32_t SrcAddress1, uint32_t SrcAddress2,
                                             uint32_t DstAddress, uint32_t Width, uint32_t Height);
HAL_StatusTypeDef HAL_DMA2D_Suspend(DMA2D_HandleTypeDef *hdma2d);
HAL_StatusTypeDef HAL_DMA2D_Resume(DMA2D_HandleTypeDef *hdma2d);
HAL_StatusTypeDef HAL_DMA2D_Abort(DMA2D_HandleTypeDef *hdma2d);
HAL_StatusTypeDef HAL_DMA2D_EnableCLUT(DMA2D_HandleTypeDef *hdma2d, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_CLUTStartLoad(DMA2D_HandleTypeDef *hdma2d, DMA2D_CLUTCfgTypeDef *CLUTCfg,
                                          uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_CLUTStartLoad_IT(DMA2D_HandleTypeDef *hdma2d, DMA2D_CLUTCfgTypeDef *CLUTCfg,
                                             uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_CLUTLoad(DMA2D_HandleTypeDef *hdma2d, DMA2D_CLUTCfgTypeDef CLUTCfg, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_CLUTLoad_IT(DMA2D_HandleTypeDef *hdma2d, DMA2D_CLUTCfgTypeDef CLUTCfg, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_CLUTLoading_Abort(DMA2D_HandleTypeDef *hdma2d, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_CLUTLoading_Suspend(DMA2D_HandleTypeDef *hdma2d, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_CLUTLoading_Resume(DMA2D_HandleTypeDef *hdma2d, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_PollForTransfer(DMA2D_HandleTypeDef *hdma2d, uint32_t Timeout);
void              HAL_DMA2D_IRQHandler(DMA2D_HandleTypeDef *hdma2d);
void              HAL_DMA2D_LineEventCallback(DMA2D_HandleTypeDef *hdma2d);
void              HAL_DMA2D_CLUTLoadingCpltCallback(DMA2D_HandleTypeDef *hdma2d);

/**
  * @}
  */

/** @addtogroup DMA2D_Exported_Functions_Group3 Peripheral Control functions
  * @{
  */

/* Peripheral Control functions *************************************************/
HAL_StatusTypeDef HAL_DMA2D_ConfigLayer(DMA2D_HandleTypeDef *hdma2d, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_ConfigCLUT(DMA2D_HandleTypeDef *hdma2d, DMA2D_CLUTCfgTypeDef CLUTCfg, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_DMA2D_ProgramLineEvent(DMA2D_HandleTypeDef *hdma2d, uint32_t Line);
HAL_StatusTypeDef HAL_DMA2D_EnableDeadTime(DMA2D_HandleTypeDef *hdma2d);
HAL_StatusTypeDef HAL_DMA2D_DisableDeadTime(DMA2D_HandleTypeDef *hdma2d);
HAL_StatusTypeDef HAL_DMA2D_ConfigDeadTime(DMA2D_HandleTypeDef *hdma2d, uint8_t DeadTime);

/**
  * @}
  */

/** @addtogroup DMA2D_Exported_Functions_Group4 Peripheral State and Error functions
  * @{
  */

/* Peripheral State functions ***************************************************/
HAL_DMA2D_StateTypeDef HAL_DMA2D_GetState(DMA2D_HandleTypeDef *hdma2d);
uint32_t               HAL_DMA2D_GetError(DMA2D_HandleTypeDef *hdma2d);

/**
  * @}
  */

/**
  * @}
  */

/* Private constants ---------------------------------------------------------*/

/** @addtogroup DMA2D_Private_Constants DMA2D Private Constants
  * @{
  */

/** @defgroup DMA2D_Maximum_Line_WaterMark DMA2D Maximum Line Watermark
  * @{
  */
#define DMA2D_LINE_WATERMARK_MAX            DMA2D_LWR_LW       /*!< DMA2D maximum line watermark */
/**
  * @}
  */

/** @defgroup DMA2D_Color_Value DMA2D Color Value
  * @{
  */
#define DMA2D_COLOR_VALUE                 0x000000FFU  /*!< Color value mask */
/**
  * @}
  */

/** @defgroup DMA2D_Max_Layer DMA2D Maximum Number of Layers
  * @{
  */
#define DMA2D_MAX_LAYER         2U         /*!< DMA2D maximum number of layers */
/**
  * @}
  */

/** @defgroup DMA2D_Layers DMA2D Layers
  * @{
  */
#define DMA2D_BACKGROUND_LAYER             0x00000000U   /*!< DMA2D Background Layer (layer 0) */
#define DMA2D_FOREGROUND_LAYER             0x00000001U   /*!< DMA2D Foreground Layer (layer 1) */
/**
  * @}
  */

/** @defgroup DMA2D_Offset DMA2D Offset
  * @{
  */
#define DMA2D_OFFSET                DMA2D_FGOR_LO            /*!< maximum Line Offset */
/**
  * @}
  */

/** @defgroup DMA2D_Size DMA2D Size
  * @{
  */
#define DMA2D_PIXEL                 (DMA2D_NLR_PL >> 16U)    /*!< DMA2D maximum number of pixels per line */
#define DMA2D_LINE                  DMA2D_NLR_NL             /*!< DMA2D maximum number of lines           */
/**
  * @}
  */

/** @defgroup DMA2D_CLUT_Size DMA2D CLUT Size
  * @{
  */
#define DMA2D_CLUT_SIZE             (DMA2D_FGPFCCR_CS >> 8U)  /*!< DMA2D maximum CLUT size */
/**
  * @}
  */

/**
  * @}
  */


/* Private macros ------------------------------------------------------------*/
/** @defgroup DMA2D_Private_Macros DMA2D Private Macros
  * @{
  */
#define IS_DMA2D_LAYER(LAYER)                 (((LAYER) == DMA2D_BACKGROUND_LAYER)\
                                               || ((LAYER) == DMA2D_FOREGROUND_LAYER))

#define IS_DMA2D_MODE(MODE)                   (((MODE) == DMA2D_M2M)       || ((MODE) == DMA2D_M2M_PFC) || \
                                               ((MODE) == DMA2D_M2M_BLEND) || ((MODE) == DMA2D_R2M))

#define IS_DMA2D_CMODE(MODE_ARGB)             (((MODE_ARGB) == DMA2D_OUTPUT_ARGB8888) || \
                                               ((MODE_ARGB) == DMA2D_OUTPUT_RGB888)   || \
                                               ((MODE_ARGB) == DMA2D_OUTPUT_RGB565)   || \
                                               ((MODE_ARGB) == DMA2D_OUTPUT_ARGB1555) || \
                                               ((MODE_ARGB) == DMA2D_OUTPUT_ARGB4444))

#define IS_DMA2D_COLOR(COLOR)                 ((COLOR) <= DMA2D_COLOR_VALUE)
#define IS_DMA2D_LINE(LINE)                   ((LINE) <= DMA2D_LINE)
#define IS_DMA2D_PIXEL(PIXEL)                 ((PIXEL) <= DMA2D_PIXEL)
#define IS_DMA2D_OFFSET(OOFFSET)              ((OOFFSET) <= DMA2D_OFFSET)

#define IS_DMA2D_INPUT_COLOR_MODE(INPUT_CM)   (((INPUT_CM) == DMA2D_INPUT_ARGB8888) || \
                                               ((INPUT_CM) == DMA2D_INPUT_RGB888)   || \
                                               ((INPUT_CM) == DMA2D_INPUT_RGB565)   || \
                                               ((INPUT_CM) == DMA2D_INPUT_ARGB1555) || \
                                               ((INPUT_CM) == DMA2D_INPUT_ARGB4444) || \
                                               ((INPUT_CM) == DMA2D_INPUT_L8)       || \
                                               ((INPUT_CM) == DMA2D_INPUT_AL44)     || \
                                               ((INPUT_CM) == DMA2D_INPUT_AL88)     || \
                                               ((INPUT_CM) == DMA2D_INPUT_L4)       || \
                                               ((INPUT_CM) == DMA2D_INPUT_A8)       || \
                                               ((INPUT_CM) == DMA2D_INPUT_A4))

#define IS_DMA2D_ALPHA_MODE(AlphaMode)        (((AlphaMode) == DMA2D_NO_MODIF_ALPHA) || \
                                               ((AlphaMode) == DMA2D_REPLACE_ALPHA)  || \
                                               ((AlphaMode) == DMA2D_COMBINE_ALPHA))





#define IS_DMA2D_CLUT_CM(CLUT_CM)             (((CLUT_CM) == DMA2D_CCM_ARGB8888) || ((CLUT_CM) == DMA2D_CCM_RGB888))
#define IS_DMA2D_CLUT_SIZE(CLUT_SIZE)         ((CLUT_SIZE) <= DMA2D_CLUT_SIZE)
#define IS_DMA2D_LINEWATERMARK(LineWatermark) ((LineWatermark) <= DMA2D_LINE_WATERMARK_MAX)
#define IS_DMA2D_IT(IT)                       (((IT) == DMA2D_IT_CTC) || ((IT) == DMA2D_IT_CAE) || \
                                               ((IT) == DMA2D_IT_TW)  || ((IT) == DMA2D_IT_TC)  || \
                                               ((IT) == DMA2D_IT_TE)  || ((IT) == DMA2D_IT_CE))
#define IS_DMA2D_GET_FLAG(FLAG)               (((FLAG) == DMA2D_FLAG_CTC) || ((FLAG) == DMA2D_FLAG_CAE) || \
                                               ((FLAG) == DMA2D_FLAG_TW)  || ((FLAG) == DMA2D_FLAG_TC)  || \
                                               ((FLAG) == DMA2D_FLAG_TE)  || ((FLAG) == DMA2D_FLAG_CE))
/**
  * @}
  */

/**
  * @}
  */

#endif /* defined (DMA2D) */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_HAL_DMA2D_H */


/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
