/**
  ******************************************************************************
  * @file    stm32f4xx_hal_ltdc.h
  * @author  MCD Application Team
  * @brief   Header file of LTDC HAL module.
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
#ifndef STM32F4xx_HAL_LTDC_H
#define STM32F4xx_HAL_LTDC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

#if defined (LTDC)

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup LTDC LTDC
  * @brief LTDC HAL module driver
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup LTDC_Exported_Types LTDC Exported Types
  * @{
  */
#define MAX_LAYER  2U

/**
  * @brief  LTDC color structure definition
  */
typedef struct
{
  uint8_t Blue;                    /*!< Configures the blue value.
                                        This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF. */

  uint8_t Green;                   /*!< Configures the green value.
                                        This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF. */

  uint8_t Red;                     /*!< Configures the red value.
                                        This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF. */

  uint8_t Reserved;                /*!< Reserved 0xFF */
} LTDC_ColorTypeDef;

/**
  * @brief  LTDC Init structure definition
  */
typedef struct
{
  uint32_t            HSPolarity;                /*!< configures the horizontal synchronization polarity.
                                                      This parameter can be one value of @ref LTDC_HS_POLARITY */

  uint32_t            VSPolarity;                /*!< configures the vertical synchronization polarity.
                                                      This parameter can be one value of @ref LTDC_VS_POLARITY */

  uint32_t            DEPolarity;                /*!< configures the data enable polarity.
                                                      This parameter can be one of value of @ref LTDC_DE_POLARITY */

  uint32_t            PCPolarity;                /*!< configures the pixel clock polarity.
                                                      This parameter can be one of value of @ref LTDC_PC_POLARITY */

  uint32_t            HorizontalSync;            /*!< configures the number of Horizontal synchronization width.
                                                      This parameter must be a number between Min_Data = 0x000 and Max_Data = 0xFFF. */

  uint32_t            VerticalSync;              /*!< configures the number of Vertical synchronization height.
                                                      This parameter must be a number between Min_Data = 0x000 and Max_Data = 0x7FF. */

  uint32_t            AccumulatedHBP;            /*!< configures the accumulated horizontal back porch width.
                                                      This parameter must be a number between Min_Data = LTDC_HorizontalSync and Max_Data = 0xFFF. */

  uint32_t            AccumulatedVBP;            /*!< configures the accumulated vertical back porch height.
                                                      This parameter must be a number between Min_Data = LTDC_VerticalSync and Max_Data = 0x7FF. */

  uint32_t            AccumulatedActiveW;        /*!< configures the accumulated active width.
                                                      This parameter must be a number between Min_Data = LTDC_AccumulatedHBP and Max_Data = 0xFFF. */

  uint32_t            AccumulatedActiveH;        /*!< configures the accumulated active height.
                                                      This parameter must be a number between Min_Data = LTDC_AccumulatedVBP and Max_Data = 0x7FF. */

  uint32_t            TotalWidth;                /*!< configures the total width.
                                                      This parameter must be a number between Min_Data = LTDC_AccumulatedActiveW and Max_Data = 0xFFF. */

  uint32_t            TotalHeigh;                /*!< configures the total height.
                                                      This parameter must be a number between Min_Data = LTDC_AccumulatedActiveH and Max_Data = 0x7FF. */

  LTDC_ColorTypeDef   Backcolor;                 /*!< Configures the background color. */
} LTDC_InitTypeDef;

/**
  * @brief  LTDC Layer structure definition
  */
typedef struct
{
  uint32_t WindowX0;                   /*!< Configures the Window Horizontal Start Position.
                                            This parameter must be a number between Min_Data = 0x000 and Max_Data = 0xFFF. */

  uint32_t WindowX1;                   /*!< Configures the Window Horizontal Stop Position.
                                            This parameter must be a number between Min_Data = 0x000 and Max_Data = 0xFFF. */

  uint32_t WindowY0;                   /*!< Configures the Window vertical Start Position.
                                            This parameter must be a number between Min_Data = 0x000 and Max_Data = 0x7FF. */

  uint32_t WindowY1;                   /*!< Configures the Window vertical Stop Position.
                                            This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0x7FF. */

  uint32_t PixelFormat;                /*!< Specifies the pixel format.
                                            This parameter can be one of value of @ref LTDC_Pixelformat */

  uint32_t Alpha;                      /*!< Specifies the constant alpha used for blending.
                                            This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF. */

  uint32_t Alpha0;                     /*!< Configures the default alpha value.
                                            This parameter must be a number between Min_Data = 0x00 and Max_Data = 0xFF. */

  uint32_t BlendingFactor1;            /*!< Select the blending factor 1.
                                            This parameter can be one of value of @ref LTDC_BlendingFactor1 */

  uint32_t BlendingFactor2;            /*!< Select the blending factor 2.
                                            This parameter can be one of value of @ref LTDC_BlendingFactor2 */

  uint32_t FBStartAdress;              /*!< Configures the color frame buffer address */

  uint32_t ImageWidth;                 /*!< Configures the color frame buffer line length.
                                            This parameter must be a number between Min_Data = 0x0000 and Max_Data = 0x1FFF. */

  uint32_t ImageHeight;                /*!< Specifies the number of line in frame buffer.
                                            This parameter must be a number between Min_Data = 0x000 and Max_Data = 0x7FF. */

  LTDC_ColorTypeDef   Backcolor;       /*!< Configures the layer background color. */
} LTDC_LayerCfgTypeDef;

/**
  * @brief  HAL LTDC State structures definition
  */
typedef enum
{
  HAL_LTDC_STATE_RESET             = 0x00U,    /*!< LTDC not yet initialized or disabled */
  HAL_LTDC_STATE_READY             = 0x01U,    /*!< LTDC initialized and ready for use   */
  HAL_LTDC_STATE_BUSY              = 0x02U,    /*!< LTDC internal process is ongoing     */
  HAL_LTDC_STATE_TIMEOUT           = 0x03U,    /*!< LTDC Timeout state                   */
  HAL_LTDC_STATE_ERROR             = 0x04U     /*!< LTDC state error                     */
} HAL_LTDC_StateTypeDef;

/**
  * @brief  LTDC handle Structure definition
  */
#if (USE_HAL_LTDC_REGISTER_CALLBACKS == 1)
typedef struct __LTDC_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_LTDC_REGISTER_CALLBACKS */
{
  LTDC_TypeDef                *Instance;                /*!< LTDC Register base address                */

  LTDC_InitTypeDef            Init;                     /*!< LTDC parameters                           */

  LTDC_LayerCfgTypeDef        LayerCfg[MAX_LAYER];      /*!< LTDC Layers parameters                    */

  HAL_LockTypeDef             Lock;                     /*!< LTDC Lock                                 */

  __IO HAL_LTDC_StateTypeDef  State;                    /*!< LTDC state                                */

  __IO uint32_t               ErrorCode;                /*!< LTDC Error code                           */

#if (USE_HAL_LTDC_REGISTER_CALLBACKS == 1)
  void (* LineEventCallback)(struct __LTDC_HandleTypeDef *hltdc);     /*!< LTDC Line Event Callback    */
  void (* ReloadEventCallback)(struct __LTDC_HandleTypeDef *hltdc);   /*!< LTDC Reload Event Callback  */
  void (* ErrorCallback)(struct __LTDC_HandleTypeDef *hltdc);         /*!< LTDC Error Callback         */

  void (* MspInitCallback)(struct __LTDC_HandleTypeDef *hltdc);       /*!< LTDC Msp Init callback      */
  void (* MspDeInitCallback)(struct __LTDC_HandleTypeDef *hltdc);     /*!< LTDC Msp DeInit callback    */

#endif /* USE_HAL_LTDC_REGISTER_CALLBACKS */


} LTDC_HandleTypeDef;

#if (USE_HAL_LTDC_REGISTER_CALLBACKS == 1)
/**
  * @brief  HAL LTDC Callback ID enumeration definition
  */
typedef enum
{
  HAL_LTDC_MSPINIT_CB_ID            = 0x00U,    /*!< LTDC MspInit callback ID       */
  HAL_LTDC_MSPDEINIT_CB_ID          = 0x01U,    /*!< LTDC MspDeInit callback ID     */

  HAL_LTDC_LINE_EVENT_CB_ID         = 0x02U,    /*!< LTDC Line Event Callback ID    */
  HAL_LTDC_RELOAD_EVENT_CB_ID       = 0x03U,    /*!< LTDC Reload Callback ID        */
  HAL_LTDC_ERROR_CB_ID              = 0x04U     /*!< LTDC Error Callback ID         */

} HAL_LTDC_CallbackIDTypeDef;

/**
  * @brief  HAL LTDC Callback pointer definition
  */
typedef  void (*pLTDC_CallbackTypeDef)(LTDC_HandleTypeDef *hltdc);  /*!< pointer to an LTDC callback function */

#endif /* USE_HAL_LTDC_REGISTER_CALLBACKS */

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup LTDC_Exported_Constants LTDC Exported Constants
  * @{
  */

/** @defgroup LTDC_Error_Code LTDC Error Code
  * @{
  */
#define HAL_LTDC_ERROR_NONE               0x00000000U   /*!< LTDC No error             */
#define HAL_LTDC_ERROR_TE                 0x00000001U   /*!< LTDC Transfer error       */
#define HAL_LTDC_ERROR_FU                 0x00000002U   /*!< LTDC FIFO Underrun        */
#define HAL_LTDC_ERROR_TIMEOUT            0x00000020U   /*!< LTDC Timeout error        */
#if (USE_HAL_LTDC_REGISTER_CALLBACKS == 1)
#define  HAL_LTDC_ERROR_INVALID_CALLBACK  0x00000040U   /*!< LTDC Invalid Callback error  */
#endif /* USE_HAL_LTDC_REGISTER_CALLBACKS */
/**
  * @}
  */

/** @defgroup LTDC_Layer LTDC Layer
  * @{
  */
#define LTDC_LAYER_1                      0x00000000U   /*!< LTDC Layer 1 */
#define LTDC_LAYER_2                      0x00000001U   /*!< LTDC Layer 2 */
/**
  * @}
  */

/** @defgroup LTDC_HS_POLARITY LTDC HS POLARITY
  * @{
  */
#define LTDC_HSPOLARITY_AL                0x00000000U   /*!< Horizontal Synchronization is active low. */
#define LTDC_HSPOLARITY_AH                LTDC_GCR_HSPOL            /*!< Horizontal Synchronization is active high. */
/**
  * @}
  */

/** @defgroup LTDC_VS_POLARITY LTDC VS POLARITY
  * @{
  */
#define LTDC_VSPOLARITY_AL                0x00000000U   /*!< Vertical Synchronization is active low. */
#define LTDC_VSPOLARITY_AH                LTDC_GCR_VSPOL            /*!< Vertical Synchronization is active high. */
/**
  * @}
  */

/** @defgroup LTDC_DE_POLARITY LTDC DE POLARITY
  * @{
  */
#define LTDC_DEPOLARITY_AL                0x00000000U   /*!< Data Enable, is active low. */
#define LTDC_DEPOLARITY_AH                LTDC_GCR_DEPOL            /*!< Data Enable, is active high. */
/**
  * @}
  */

/** @defgroup LTDC_PC_POLARITY LTDC PC POLARITY
  * @{
  */
#define LTDC_PCPOLARITY_IPC               0x00000000U   /*!< input pixel clock. */
#define LTDC_PCPOLARITY_IIPC              LTDC_GCR_PCPOL            /*!< inverted input pixel clock. */
/**
  * @}
  */

/** @defgroup LTDC_SYNC LTDC SYNC
  * @{
  */
#define LTDC_HORIZONTALSYNC               (LTDC_SSCR_HSW >> 16U)    /*!< Horizontal synchronization width. */
#define LTDC_VERTICALSYNC                 LTDC_SSCR_VSH             /*!< Vertical synchronization height. */
/**
  * @}
  */

/** @defgroup LTDC_BACK_COLOR LTDC BACK COLOR
  * @{
  */
#define LTDC_COLOR                        0x000000FFU   /*!< Color mask */
/**
  * @}
  */

/** @defgroup LTDC_BlendingFactor1 LTDC Blending Factor1
  * @{
  */
#define LTDC_BLENDING_FACTOR1_CA          0x00000400U   /*!< Blending factor : Cte Alpha */
#define LTDC_BLENDING_FACTOR1_PAxCA       0x00000600U   /*!< Blending factor : Cte Alpha x Pixel Alpha*/
/**
  * @}
  */

/** @defgroup LTDC_BlendingFactor2 LTDC Blending Factor2
  * @{
  */
#define LTDC_BLENDING_FACTOR2_CA          0x00000005U   /*!< Blending factor : Cte Alpha */
#define LTDC_BLENDING_FACTOR2_PAxCA       0x00000007U   /*!< Blending factor : Cte Alpha x Pixel Alpha*/
/**
  * @}
  */

/** @defgroup LTDC_Pixelformat LTDC Pixel format
  * @{
  */
#define LTDC_PIXEL_FORMAT_ARGB8888        0x00000000U   /*!< ARGB8888 LTDC pixel format */
#define LTDC_PIXEL_FORMAT_RGB888          0x00000001U   /*!< RGB888 LTDC pixel format   */
#define LTDC_PIXEL_FORMAT_RGB565          0x00000002U   /*!< RGB565 LTDC pixel format   */
#define LTDC_PIXEL_FORMAT_ARGB1555        0x00000003U   /*!< ARGB1555 LTDC pixel format */
#define LTDC_PIXEL_FORMAT_ARGB4444        0x00000004U   /*!< ARGB4444 LTDC pixel format */
#define LTDC_PIXEL_FORMAT_L8              0x00000005U   /*!< L8 LTDC pixel format       */
#define LTDC_PIXEL_FORMAT_AL44            0x00000006U   /*!< AL44 LTDC pixel format     */
#define LTDC_PIXEL_FORMAT_AL88            0x00000007U   /*!< AL88 LTDC pixel format     */
/**
  * @}
  */

/** @defgroup LTDC_Alpha LTDC Alpha
  * @{
  */
#define LTDC_ALPHA                        LTDC_LxCACR_CONSTA        /*!< LTDC Constant Alpha mask */
/**
  * @}
  */

/** @defgroup LTDC_LAYER_Config LTDC LAYER Config
  * @{
  */
#define LTDC_STOPPOSITION                 (LTDC_LxWHPCR_WHSPPOS >> 16U) /*!< LTDC Layer stop position  */
#define LTDC_STARTPOSITION                LTDC_LxWHPCR_WHSTPOS          /*!< LTDC Layer start position */

#define LTDC_COLOR_FRAME_BUFFER           LTDC_LxCFBLR_CFBLL            /*!< LTDC Layer Line length    */
#define LTDC_LINE_NUMBER                  LTDC_LxCFBLNR_CFBLNBR         /*!< LTDC Layer Line number    */
/**
  * @}
  */

/** @defgroup LTDC_Interrupts LTDC Interrupts
  * @{
  */
#define LTDC_IT_LI                        LTDC_IER_LIE              /*!< LTDC Line Interrupt            */
#define LTDC_IT_FU                        LTDC_IER_FUIE             /*!< LTDC FIFO Underrun Interrupt   */
#define LTDC_IT_TE                        LTDC_IER_TERRIE           /*!< LTDC Transfer Error Interrupt  */
#define LTDC_IT_RR                        LTDC_IER_RRIE             /*!< LTDC Register Reload Interrupt */
/**
  * @}
  */

/** @defgroup LTDC_Flags LTDC Flags
  * @{
  */
#define LTDC_FLAG_LI                      LTDC_ISR_LIF              /*!< LTDC Line Interrupt Flag            */
#define LTDC_FLAG_FU                      LTDC_ISR_FUIF             /*!< LTDC FIFO Underrun interrupt Flag   */
#define LTDC_FLAG_TE                      LTDC_ISR_TERRIF           /*!< LTDC Transfer Error interrupt Flag  */
#define LTDC_FLAG_RR                      LTDC_ISR_RRIF             /*!< LTDC Register Reload interrupt Flag */
/**
  * @}
  */

/** @defgroup LTDC_Reload_Type LTDC Reload Type
  * @{
  */
#define LTDC_RELOAD_IMMEDIATE             LTDC_SRCR_IMR             /*!< Immediate Reload */
#define LTDC_RELOAD_VERTICAL_BLANKING     LTDC_SRCR_VBR             /*!< Vertical Blanking Reload */
/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup LTDC_Exported_Macros LTDC Exported Macros
  * @{
  */

/** @brief Reset LTDC handle state.
  * @param  __HANDLE__  LTDC handle
  * @retval None
  */
#if (USE_HAL_LTDC_REGISTER_CALLBACKS == 1)
#define __HAL_LTDC_RESET_HANDLE_STATE(__HANDLE__) do{                                                  \
                                                      (__HANDLE__)->State = HAL_LTDC_STATE_RESET;     \
                                                      (__HANDLE__)->MspInitCallback = NULL;           \
                                                      (__HANDLE__)->MspDeInitCallback = NULL;         \
                                                    } while(0)
#else
#define __HAL_LTDC_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = HAL_LTDC_STATE_RESET)
#endif /*USE_HAL_LTDC_REGISTER_CALLBACKS */

/**
  * @brief  Enable the LTDC.
  * @param  __HANDLE__  LTDC handle
  * @retval None.
  */
#define __HAL_LTDC_ENABLE(__HANDLE__)    ((__HANDLE__)->Instance->GCR |= LTDC_GCR_LTDCEN)

/**
  * @brief  Disable the LTDC.
  * @param  __HANDLE__  LTDC handle
  * @retval None.
  */
#define __HAL_LTDC_DISABLE(__HANDLE__)   ((__HANDLE__)->Instance->GCR &= ~(LTDC_GCR_LTDCEN))

/**
  * @brief  Enable the LTDC Layer.
  * @param  __HANDLE__  LTDC handle
  * @param  __LAYER__   Specify the layer to be enabled.
  *                     This parameter can be LTDC_LAYER_1 (0) or LTDC_LAYER_2 (1).
  * @retval None.
  */
#define __HAL_LTDC_LAYER_ENABLE(__HANDLE__, __LAYER__)  ((LTDC_LAYER((__HANDLE__), (__LAYER__)))->CR |= (uint32_t)LTDC_LxCR_LEN)

/**
  * @brief  Disable the LTDC Layer.
  * @param  __HANDLE__  LTDC handle
  * @param  __LAYER__   Specify the layer to be disabled.
  *                     This parameter can be LTDC_LAYER_1 (0) or LTDC_LAYER_2 (1).
  * @retval None.
  */
#define __HAL_LTDC_LAYER_DISABLE(__HANDLE__, __LAYER__) ((LTDC_LAYER((__HANDLE__), (__LAYER__)))->CR &= ~(uint32_t)LTDC_LxCR_LEN)

/**
  * @brief  Reload immediately all LTDC Layers.
  * @param  __HANDLE__  LTDC handle
  * @retval None.
  */
#define __HAL_LTDC_RELOAD_IMMEDIATE_CONFIG(__HANDLE__)  ((__HANDLE__)->Instance->SRCR |= LTDC_SRCR_IMR)

/**
  * @brief  Reload during vertical blanking period all LTDC Layers.
  * @param  __HANDLE__  LTDC handle
  * @retval None.
  */
#define __HAL_LTDC_VERTICAL_BLANKING_RELOAD_CONFIG(__HANDLE__)  ((__HANDLE__)->Instance->SRCR |= LTDC_SRCR_VBR)

/* Interrupt & Flag management */
/**
  * @brief  Get the LTDC pending flags.
  * @param  __HANDLE__  LTDC handle
  * @param  __FLAG__    Get the specified flag.
  *          This parameter can be any combination of the following values:
  *            @arg LTDC_FLAG_LI: Line Interrupt flag
  *            @arg LTDC_FLAG_FU: FIFO Underrun Interrupt flag
  *            @arg LTDC_FLAG_TE: Transfer Error interrupt flag
  *            @arg LTDC_FLAG_RR: Register Reload Interrupt Flag
  * @retval The state of FLAG (SET or RESET).
  */
#define __HAL_LTDC_GET_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->ISR & (__FLAG__))

/**
  * @brief  Clears the LTDC pending flags.
  * @param  __HANDLE__  LTDC handle
  * @param  __FLAG__    Specify the flag to clear.
  *          This parameter can be any combination of the following values:
  *            @arg LTDC_FLAG_LI: Line Interrupt flag
  *            @arg LTDC_FLAG_FU: FIFO Underrun Interrupt flag
  *            @arg LTDC_FLAG_TE: Transfer Error interrupt flag
  *            @arg LTDC_FLAG_RR: Register Reload Interrupt Flag
  * @retval None
  */
#define __HAL_LTDC_CLEAR_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->ICR = (__FLAG__))

/**
  * @brief  Enables the specified LTDC interrupts.
  * @param  __HANDLE__  LTDC handle
  * @param __INTERRUPT__ Specify the LTDC interrupt sources to be enabled.
  *          This parameter can be any combination of the following values:
  *            @arg LTDC_IT_LI: Line Interrupt flag
  *            @arg LTDC_IT_FU: FIFO Underrun Interrupt flag
  *            @arg LTDC_IT_TE: Transfer Error interrupt flag
  *            @arg LTDC_IT_RR: Register Reload Interrupt Flag
  * @retval None
  */
#define __HAL_LTDC_ENABLE_IT(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->IER |= (__INTERRUPT__))

/**
  * @brief  Disables the specified LTDC interrupts.
  * @param  __HANDLE__  LTDC handle
  * @param __INTERRUPT__ Specify the LTDC interrupt sources to be disabled.
  *          This parameter can be any combination of the following values:
  *            @arg LTDC_IT_LI: Line Interrupt flag
  *            @arg LTDC_IT_FU: FIFO Underrun Interrupt flag
  *            @arg LTDC_IT_TE: Transfer Error interrupt flag
  *            @arg LTDC_IT_RR: Register Reload Interrupt Flag
  * @retval None
  */
#define __HAL_LTDC_DISABLE_IT(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->IER &= ~(__INTERRUPT__))

/**
  * @brief  Check whether the specified LTDC interrupt has occurred or not.
  * @param  __HANDLE__  LTDC handle
  * @param __INTERRUPT__ Specify the LTDC interrupt source to check.
  *          This parameter can be one of the following values:
  *            @arg LTDC_IT_LI: Line Interrupt flag
  *            @arg LTDC_IT_FU: FIFO Underrun Interrupt flag
  *            @arg LTDC_IT_TE: Transfer Error interrupt flag
  *            @arg LTDC_IT_RR: Register Reload Interrupt Flag
  * @retval The state of INTERRUPT (SET or RESET).
  */
#define __HAL_LTDC_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->IER & (__INTERRUPT__))
/**
  * @}
  */

/* Include LTDC HAL Extension module */
#include "stm32f4xx_hal_ltdc_ex.h"

/* Exported functions --------------------------------------------------------*/
/** @addtogroup LTDC_Exported_Functions
  * @{
  */
/** @addtogroup LTDC_Exported_Functions_Group1
  * @{
  */
/* Initialization and de-initialization functions *****************************/
HAL_StatusTypeDef HAL_LTDC_Init(LTDC_HandleTypeDef *hltdc);
HAL_StatusTypeDef HAL_LTDC_DeInit(LTDC_HandleTypeDef *hltdc);
void HAL_LTDC_MspInit(LTDC_HandleTypeDef *hltdc);
void HAL_LTDC_MspDeInit(LTDC_HandleTypeDef *hltdc);
void HAL_LTDC_ErrorCallback(LTDC_HandleTypeDef *hltdc);
void HAL_LTDC_LineEventCallback(LTDC_HandleTypeDef *hltdc);
void HAL_LTDC_ReloadEventCallback(LTDC_HandleTypeDef *hltdc);

/* Callbacks Register/UnRegister functions  ***********************************/
#if (USE_HAL_LTDC_REGISTER_CALLBACKS == 1)
HAL_StatusTypeDef HAL_LTDC_RegisterCallback(LTDC_HandleTypeDef *hltdc, HAL_LTDC_CallbackIDTypeDef CallbackID, pLTDC_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_LTDC_UnRegisterCallback(LTDC_HandleTypeDef *hltdc, HAL_LTDC_CallbackIDTypeDef CallbackID);
#endif /* USE_HAL_LTDC_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @addtogroup LTDC_Exported_Functions_Group2
  * @{
  */
/* IO operation functions *****************************************************/
void  HAL_LTDC_IRQHandler(LTDC_HandleTypeDef *hltdc);
/**
  * @}
  */

/** @addtogroup LTDC_Exported_Functions_Group3
  * @{
  */
/* Peripheral Control functions ***********************************************/
HAL_StatusTypeDef HAL_LTDC_ConfigLayer(LTDC_HandleTypeDef *hltdc, LTDC_LayerCfgTypeDef *pLayerCfg, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetWindowSize(LTDC_HandleTypeDef *hltdc, uint32_t XSize, uint32_t YSize, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetWindowPosition(LTDC_HandleTypeDef *hltdc, uint32_t X0, uint32_t Y0, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetPixelFormat(LTDC_HandleTypeDef *hltdc, uint32_t Pixelformat, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetAlpha(LTDC_HandleTypeDef *hltdc, uint32_t Alpha, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetAddress(LTDC_HandleTypeDef *hltdc, uint32_t Address, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetPitch(LTDC_HandleTypeDef *hltdc, uint32_t LinePitchInPixels, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_ConfigColorKeying(LTDC_HandleTypeDef *hltdc, uint32_t RGBValue, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_ConfigCLUT(LTDC_HandleTypeDef *hltdc, uint32_t *pCLUT, uint32_t CLUTSize, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_EnableColorKeying(LTDC_HandleTypeDef *hltdc, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_DisableColorKeying(LTDC_HandleTypeDef *hltdc, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_EnableCLUT(LTDC_HandleTypeDef *hltdc, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_DisableCLUT(LTDC_HandleTypeDef *hltdc, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_ProgramLineEvent(LTDC_HandleTypeDef *hltdc, uint32_t Line);
HAL_StatusTypeDef HAL_LTDC_EnableDither(LTDC_HandleTypeDef *hltdc);
HAL_StatusTypeDef HAL_LTDC_DisableDither(LTDC_HandleTypeDef *hltdc);
HAL_StatusTypeDef HAL_LTDC_Reload(LTDC_HandleTypeDef *hltdc, uint32_t ReloadType);
HAL_StatusTypeDef HAL_LTDC_ConfigLayer_NoReload(LTDC_HandleTypeDef *hltdc, LTDC_LayerCfgTypeDef *pLayerCfg, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetWindowSize_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t XSize, uint32_t YSize, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetWindowPosition_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t X0, uint32_t Y0, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetPixelFormat_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t Pixelformat, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetAlpha_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t Alpha, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetAddress_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t Address, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_SetPitch_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t LinePitchInPixels, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_ConfigColorKeying_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t RGBValue, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_EnableColorKeying_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_DisableColorKeying_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_EnableCLUT_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t LayerIdx);
HAL_StatusTypeDef HAL_LTDC_DisableCLUT_NoReload(LTDC_HandleTypeDef *hltdc, uint32_t LayerIdx);

/**
  * @}
  */

/** @addtogroup LTDC_Exported_Functions_Group4
  * @{
  */
/* Peripheral State functions *************************************************/
HAL_LTDC_StateTypeDef HAL_LTDC_GetState(LTDC_HandleTypeDef *hltdc);
uint32_t              HAL_LTDC_GetError(LTDC_HandleTypeDef *hltdc);
/**
  * @}
  */

/**
  * @}
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/* Private macros ------------------------------------------------------------*/
/** @defgroup LTDC_Private_Macros LTDC Private Macros
  * @{
  */
#define LTDC_LAYER(__HANDLE__, __LAYER__)              ((LTDC_Layer_TypeDef *)((uint32_t)(((uint32_t)((__HANDLE__)->Instance)) + 0x84U + (0x80U*(__LAYER__)))))
#define IS_LTDC_LAYER(__LAYER__)                       ((__LAYER__) < MAX_LAYER)
#define IS_LTDC_HSPOL(__HSPOL__)                       (((__HSPOL__) == LTDC_HSPOLARITY_AL) || ((__HSPOL__) == LTDC_HSPOLARITY_AH))
#define IS_LTDC_VSPOL(__VSPOL__)                       (((__VSPOL__) == LTDC_VSPOLARITY_AL) || ((__VSPOL__) == LTDC_VSPOLARITY_AH))
#define IS_LTDC_DEPOL(__DEPOL__)                       (((__DEPOL__) == LTDC_DEPOLARITY_AL) || ((__DEPOL__) ==  LTDC_DEPOLARITY_AH))
#define IS_LTDC_PCPOL(__PCPOL__)                       (((__PCPOL__) == LTDC_PCPOLARITY_IPC) || ((__PCPOL__) ==  LTDC_PCPOLARITY_IIPC))
#define IS_LTDC_HSYNC(__HSYNC__)                       ((__HSYNC__)  <= LTDC_HORIZONTALSYNC)
#define IS_LTDC_VSYNC(__VSYNC__)                       ((__VSYNC__)  <= LTDC_VERTICALSYNC)
#define IS_LTDC_AHBP(__AHBP__)                         ((__AHBP__)   <= LTDC_HORIZONTALSYNC)
#define IS_LTDC_AVBP(__AVBP__)                         ((__AVBP__)   <= LTDC_VERTICALSYNC)
#define IS_LTDC_AAW(__AAW__)                           ((__AAW__)    <= LTDC_HORIZONTALSYNC)
#define IS_LTDC_AAH(__AAH__)                           ((__AAH__)    <= LTDC_VERTICALSYNC)
#define IS_LTDC_TOTALW(__TOTALW__)                     ((__TOTALW__) <= LTDC_HORIZONTALSYNC)
#define IS_LTDC_TOTALH(__TOTALH__)                     ((__TOTALH__) <= LTDC_VERTICALSYNC)
#define IS_LTDC_BLUEVALUE(__BBLUE__)                   ((__BBLUE__)  <= LTDC_COLOR)
#define IS_LTDC_GREENVALUE(__BGREEN__)                 ((__BGREEN__) <= LTDC_COLOR)
#define IS_LTDC_REDVALUE(__BRED__)                     ((__BRED__)   <= LTDC_COLOR)
#define IS_LTDC_BLENDING_FACTOR1(__BLENDING_FACTOR1__) (((__BLENDING_FACTOR1__) == LTDC_BLENDING_FACTOR1_CA) || \
                                                        ((__BLENDING_FACTOR1__) == LTDC_BLENDING_FACTOR1_PAxCA))
#define IS_LTDC_BLENDING_FACTOR2(__BLENDING_FACTOR1__) (((__BLENDING_FACTOR1__) == LTDC_BLENDING_FACTOR2_CA) || \
                                                        ((__BLENDING_FACTOR1__) == LTDC_BLENDING_FACTOR2_PAxCA))
#define IS_LTDC_PIXEL_FORMAT(__PIXEL_FORMAT__)         (((__PIXEL_FORMAT__) == LTDC_PIXEL_FORMAT_ARGB8888) || ((__PIXEL_FORMAT__) == LTDC_PIXEL_FORMAT_RGB888)   || \
                                                        ((__PIXEL_FORMAT__) == LTDC_PIXEL_FORMAT_RGB565)   || ((__PIXEL_FORMAT__) == LTDC_PIXEL_FORMAT_ARGB1555) || \
                                                        ((__PIXEL_FORMAT__) == LTDC_PIXEL_FORMAT_ARGB4444) || ((__PIXEL_FORMAT__) == LTDC_PIXEL_FORMAT_L8)       || \
                                                        ((__PIXEL_FORMAT__) == LTDC_PIXEL_FORMAT_AL44)     || ((__PIXEL_FORMAT__) == LTDC_PIXEL_FORMAT_AL88))
#define IS_LTDC_ALPHA(__ALPHA__)                       ((__ALPHA__) <= LTDC_ALPHA)
#define IS_LTDC_HCONFIGST(__HCONFIGST__)               ((__HCONFIGST__) <= LTDC_STARTPOSITION)
#define IS_LTDC_HCONFIGSP(__HCONFIGSP__)               ((__HCONFIGSP__) <= LTDC_STOPPOSITION)
#define IS_LTDC_VCONFIGST(__VCONFIGST__)               ((__VCONFIGST__) <= LTDC_STARTPOSITION)
#define IS_LTDC_VCONFIGSP(__VCONFIGSP__)               ((__VCONFIGSP__) <= LTDC_STOPPOSITION)
#define IS_LTDC_CFBP(__CFBP__)                         ((__CFBP__) <= LTDC_COLOR_FRAME_BUFFER)
#define IS_LTDC_CFBLL(__CFBLL__)                       ((__CFBLL__) <= LTDC_COLOR_FRAME_BUFFER)
#define IS_LTDC_CFBLNBR(__CFBLNBR__)                   ((__CFBLNBR__) <= LTDC_LINE_NUMBER)
#define IS_LTDC_LIPOS(__LIPOS__)                       ((__LIPOS__) <= 0x7FFU)
#define IS_LTDC_RELOAD(__RELOADTYPE__)                 (((__RELOADTYPE__) == LTDC_RELOAD_IMMEDIATE) || ((__RELOADTYPE__) == LTDC_RELOAD_VERTICAL_BLANKING))
/**
  * @}
  */

/* Private functions ---------------------------------------------------------*/
/** @defgroup LTDC_Private_Functions LTDC Private Functions
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

#endif /* LTDC */

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_HAL_LTDC_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
