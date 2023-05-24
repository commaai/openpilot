/**
  ******************************************************************************
  * @file    stm32f4xx_hal_dcmi.h
  * @author  MCD Application Team
  * @brief   Header file of DCMI HAL module.
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
#ifndef __STM32F4xx_HAL_DCMI_H
#define __STM32F4xx_HAL_DCMI_H

#ifdef __cplusplus
 extern "C" {
#endif

#if defined(STM32F407xx) || defined(STM32F417xx) || defined(STM32F427xx) || defined(STM32F437xx) ||\
    defined(STM32F429xx) || defined(STM32F439xx) || defined(STM32F446xx) || defined(STM32F469xx) ||\
    defined(STM32F479xx)
/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/* Include DCMI HAL Extended module */
/* (include on top of file since DCMI structures are defined in extended file) */
#include "stm32f4xx_hal_dcmi_ex.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup DCMI DCMI
  * @brief DCMI HAL module driver
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup DCMI_Exported_Types DCMI Exported Types
  * @{
  */
/**
  * @brief   DCMI Embedded Synchronisation CODE Init structure definition
  */
typedef struct
{
  uint8_t FrameStartUnmask; /*!< Specifies the frame start delimiter unmask. */
  uint8_t LineStartUnmask;  /*!< Specifies the line start delimiter unmask.  */
  uint8_t LineEndUnmask;    /*!< Specifies the line end delimiter unmask.    */
  uint8_t FrameEndUnmask;   /*!< Specifies the frame end delimiter unmask.   */
}DCMI_SyncUnmaskTypeDef;
/**
  * @brief  HAL DCMI State structures definition
  */ 
typedef enum
{
  HAL_DCMI_STATE_RESET             = 0x00U,  /*!< DCMI not yet initialized or disabled  */
  HAL_DCMI_STATE_READY             = 0x01U,  /*!< DCMI initialized and ready for use    */
  HAL_DCMI_STATE_BUSY              = 0x02U,  /*!< DCMI internal processing is ongoing   */
  HAL_DCMI_STATE_TIMEOUT           = 0x03U,  /*!< DCMI timeout state                    */
  HAL_DCMI_STATE_ERROR             = 0x04U,  /*!< DCMI error state                      */
  HAL_DCMI_STATE_SUSPENDED         = 0x05U   /*!< DCMI suspend state                    */
}HAL_DCMI_StateTypeDef;

/** 
  * @brief  DCMI handle Structure definition
  */
typedef struct __DCMI_HandleTypeDef
{
  DCMI_TypeDef                  *Instance;           /*!< DCMI Register base address   */

  DCMI_InitTypeDef              Init;                /*!< DCMI parameters              */

  HAL_LockTypeDef               Lock;                /*!< DCMI locking object          */

  __IO HAL_DCMI_StateTypeDef    State;               /*!< DCMI state                   */

  __IO uint32_t                 XferCount;           /*!< DMA transfer counter         */

  __IO uint32_t                 XferSize;            /*!< DMA transfer size            */

  uint32_t                      XferTransferNumber;  /*!< DMA transfer number          */

  uint32_t                      pBuffPtr;            /*!< Pointer to DMA output buffer */

  DMA_HandleTypeDef             *DMA_Handle;         /*!< Pointer to the DMA handler   */

  __IO uint32_t                 ErrorCode;           /*!< DCMI Error code              */
#if (USE_HAL_DCMI_REGISTER_CALLBACKS == 1)
  void    (* FrameEventCallback) ( struct __DCMI_HandleTypeDef *hdcmi);  /*!< DCMI Frame Event Callback */
  void    (* VsyncEventCallback) ( struct __DCMI_HandleTypeDef *hdcmi);  /*!< DCMI Vsync Event Callback */
  void    (* LineEventCallback ) ( struct __DCMI_HandleTypeDef *hdcmi);  /*!< DCMI Line Event Callback  */ 
  void    (* ErrorCallback)      ( struct __DCMI_HandleTypeDef *hdcmi);  /*!< DCMI Error Callback       */
  void    (* MspInitCallback)    ( struct __DCMI_HandleTypeDef *hdcmi);  /*!< DCMI Msp Init callback    */
  void    (* MspDeInitCallback)  ( struct __DCMI_HandleTypeDef *hdcmi);  /*!< DCMI Msp DeInit callback  */  
#endif  /* USE_HAL_DCMI_REGISTER_CALLBACKS */ 
}DCMI_HandleTypeDef;

#if (USE_HAL_DCMI_REGISTER_CALLBACKS == 1)
typedef enum
{
  HAL_DCMI_FRAME_EVENT_CB_ID    = 0x00U,    /*!< DCMI Frame Event Callback ID */
  HAL_DCMI_VSYNC_EVENT_CB_ID    = 0x01U,    /*!< DCMI Vsync Event Callback ID */
  HAL_DCMI_LINE_EVENT_CB_ID     = 0x02U,    /*!< DCMI Line Event Callback ID  */ 
  HAL_DCMI_ERROR_CB_ID          = 0x03U,    /*!< DCMI Error Callback ID       */
  HAL_DCMI_MSPINIT_CB_ID        = 0x04U,    /*!< DCMI MspInit callback ID     */
  HAL_DCMI_MSPDEINIT_CB_ID      = 0x05U     /*!< DCMI MspDeInit callback ID   */
  
}HAL_DCMI_CallbackIDTypeDef;
    
typedef void (*pDCMI_CallbackTypeDef)(DCMI_HandleTypeDef *hdcmi);
#endif /* USE_HAL_DCMI_REGISTER_CALLBACKS */


/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup DCMI_Exported_Constants DCMI Exported Constants
  * @{
  */

/** @defgroup DCMI_Error_Code DCMI Error Code
  * @{
  */
#define HAL_DCMI_ERROR_NONE      0x00000000U    /*!< No error              */
#define HAL_DCMI_ERROR_OVR       0x00000001U    /*!< Overrun error         */
#define HAL_DCMI_ERROR_SYNC      0x00000002U    /*!< Synchronization error */
#define HAL_DCMI_ERROR_TIMEOUT   0x00000020U    /*!< Timeout error         */
#define HAL_DCMI_ERROR_DMA       0x00000040U    /*!< DMA error             */
#if (USE_HAL_DCMI_REGISTER_CALLBACKS == 1)
#define HAL_DCMI_ERROR_INVALID_CALLBACK ((uint32_t)0x00000080U)  /*!< Invalid callback error */
#endif
/**
  * @}
  */

/** @defgroup DCMI_Capture_Mode DCMI Capture Mode
  * @{
  */ 
#define DCMI_MODE_CONTINUOUS         0x00000000U               /*!< The received data are transferred continuously
                                                                    into the destination memory through the DMA             */
#define DCMI_MODE_SNAPSHOT             ((uint32_t)DCMI_CR_CM)  /*!< Once activated, the interface waits for the start of
                                                                    frame and then transfers a single frame through the DMA */
/**
  * @}
  */

/** @defgroup DCMI_Synchronization_Mode DCMI Synchronization Mode
  * @{
  */
#define DCMI_SYNCHRO_HARDWARE        0x00000000U               /*!< Hardware synchronization data capture (frame/line start/stop)
                                                                    is synchronized with the HSYNC/VSYNC signals                  */
#define DCMI_SYNCHRO_EMBEDDED        ((uint32_t)DCMI_CR_ESS)   /*!< Embedded synchronization data capture is synchronized with
                                                                    synchronization codes embedded in the data flow               */

/**
  * @}
  */

/** @defgroup DCMI_PIXCK_Polarity DCMI PIXCK Polarity
  * @{
  */
#define DCMI_PCKPOLARITY_FALLING     0x00000000U                 /*!< Pixel clock active on Falling edge */
#define DCMI_PCKPOLARITY_RISING      ((uint32_t)DCMI_CR_PCKPOL)  /*!< Pixel clock active on Rising edge  */

/**
  * @}
  */

/** @defgroup DCMI_VSYNC_Polarity DCMI VSYNC Polarity
  * @{
  */
#define DCMI_VSPOLARITY_LOW          0x00000000U                /*!< Vertical synchronization active Low  */
#define DCMI_VSPOLARITY_HIGH         ((uint32_t)DCMI_CR_VSPOL)  /*!< Vertical synchronization active High */

/**
  * @}
  */

/** @defgroup DCMI_HSYNC_Polarity DCMI HSYNC Polarity
  * @{
  */ 
#define DCMI_HSPOLARITY_LOW          0x00000000U                /*!< Horizontal synchronization active Low  */
#define DCMI_HSPOLARITY_HIGH         ((uint32_t)DCMI_CR_HSPOL)  /*!< Horizontal synchronization active High */

/**
  * @}
  */

/** @defgroup DCMI_MODE_JPEG DCMI MODE JPEG
  * @{
  */
#define DCMI_JPEG_DISABLE            0x00000000U               /*!< Mode JPEG Disabled  */
#define DCMI_JPEG_ENABLE             ((uint32_t)DCMI_CR_JPEG)  /*!< Mode JPEG Enabled   */

/**
  * @}
  */

/** @defgroup DCMI_Capture_Rate DCMI Capture Rate
  * @{
  */
#define DCMI_CR_ALL_FRAME            0x00000000U                 /*!< All frames are captured        */
#define DCMI_CR_ALTERNATE_2_FRAME    ((uint32_t)DCMI_CR_FCRC_0)  /*!< Every alternate frame captured */
#define DCMI_CR_ALTERNATE_4_FRAME    ((uint32_t)DCMI_CR_FCRC_1)  /*!< One frame in 4 frames captured */

/**
  * @}
  */

/** @defgroup DCMI_Extended_Data_Mode DCMI Extended Data Mode
  * @{
  */
#define DCMI_EXTEND_DATA_8B     0x00000000U                                  /*!< Interface captures 8-bit data on every pixel clock  */
#define DCMI_EXTEND_DATA_10B    ((uint32_t)DCMI_CR_EDM_0)                    /*!< Interface captures 10-bit data on every pixel clock */
#define DCMI_EXTEND_DATA_12B    ((uint32_t)DCMI_CR_EDM_1)                    /*!< Interface captures 12-bit data on every pixel clock */
#define DCMI_EXTEND_DATA_14B    ((uint32_t)(DCMI_CR_EDM_0 | DCMI_CR_EDM_1))  /*!< Interface captures 14-bit data on every pixel clock */

/**
  * @}
  */

/** @defgroup DCMI_Window_Coordinate DCMI Window Coordinate
  * @{
  */
#define DCMI_WINDOW_COORDINATE    0x3FFFU   /*!< Window coordinate */

/**
  * @}
  */

/** @defgroup DCMI_Window_Height DCMI Window Height
  * @{
  */ 
#define DCMI_WINDOW_HEIGHT        0x1FFFU   /*!< Window Height */

/**
  * @}
  */

/** @defgroup DCMI_Window_Vertical_Line DCMI Window Vertical Line
  * @{
  */
#define DCMI_POSITION_CWSIZE_VLINE         (uint32_t)DCMI_CWSIZE_VLINE_Pos /*!< Required left shift to set crop window vertical line count       */
#define DCMI_POSITION_CWSTRT_VST           (uint32_t)DCMI_CWSTRT_VST_Pos   /*!< Required left shift to set crop window vertical start line count */

/**
  * @}
  */

/** @defgroup DCMI_interrupt_sources  DCMI interrupt sources
  * @{
  */
#define DCMI_IT_FRAME    ((uint32_t)DCMI_IER_FRAME_IE)    /*!< Capture complete interrupt      */
#define DCMI_IT_OVR      ((uint32_t)DCMI_IER_OVR_IE)      /*!< Overrun interrupt               */
#define DCMI_IT_ERR      ((uint32_t)DCMI_IER_ERR_IE)      /*!< Synchronization error interrupt */
#define DCMI_IT_VSYNC    ((uint32_t)DCMI_IER_VSYNC_IE)    /*!< VSYNC interrupt                 */
#define DCMI_IT_LINE     ((uint32_t)DCMI_IER_LINE_IE)     /*!< Line interrupt                  */
/**
  * @}
  */

/** @defgroup DCMI_Flags DCMI Flags
  * @{
  */

/** 
  * @brief   DCMI SR register
  */
#define DCMI_FLAG_HSYNC     ((uint32_t)DCMI_SR_INDEX|DCMI_SR_HSYNC) /*!< HSYNC pin state (active line / synchronization between lines)   */
#define DCMI_FLAG_VSYNC     ((uint32_t)DCMI_SR_INDEX|DCMI_SR_VSYNC) /*!< VSYNC pin state (active frame / synchronization between frames) */
#define DCMI_FLAG_FNE       ((uint32_t)DCMI_SR_INDEX|DCMI_SR_FNE)   /*!< FIFO not empty flag                                             */
/** 
  * @brief   DCMI RIS register
  */ 
#define DCMI_FLAG_FRAMERI    ((uint32_t)DCMI_RISR_FRAME_RIS)  /*!< Frame capture complete interrupt flag */
#define DCMI_FLAG_OVRRI      ((uint32_t)DCMI_RISR_OVR_RIS)    /*!< Overrun interrupt flag                */
#define DCMI_FLAG_ERRRI      ((uint32_t)DCMI_RISR_ERR_RIS)    /*!< Synchronization error interrupt flag  */
#define DCMI_FLAG_VSYNCRI    ((uint32_t)DCMI_RISR_VSYNC_RIS)  /*!< VSYNC interrupt flag                  */
#define DCMI_FLAG_LINERI     ((uint32_t)DCMI_RISR_LINE_RIS)   /*!< Line interrupt flag                   */
/** 
  * @brief   DCMI MIS register
  */ 
#define DCMI_FLAG_FRAMEMI    ((uint32_t)DCMI_MIS_INDEX|DCMI_MIS_FRAME_MIS)  /*!< DCMI Frame capture complete masked interrupt status */
#define DCMI_FLAG_OVRMI      ((uint32_t)DCMI_MIS_INDEX|DCMI_MIS_OVR_MIS  )  /*!< DCMI Overrun masked interrupt status                */
#define DCMI_FLAG_ERRMI      ((uint32_t)DCMI_MIS_INDEX|DCMI_MIS_ERR_MIS  )  /*!< DCMI Synchronization error masked interrupt status  */
#define DCMI_FLAG_VSYNCMI    ((uint32_t)DCMI_MIS_INDEX|DCMI_MIS_VSYNC_MIS)  /*!< DCMI VSYNC masked interrupt status                  */
#define DCMI_FLAG_LINEMI     ((uint32_t)DCMI_MIS_INDEX|DCMI_MIS_LINE_MIS )  /*!< DCMI Line masked interrupt status                   */
/**
  * @}
  */

/**
  * @}
  */
 
/* Exported macro ------------------------------------------------------------*/
/** @defgroup DCMI_Exported_Macros DCMI Exported Macros
  * @{
  */
  
/** @brief Reset DCMI handle state
  * @param  __HANDLE__ specifies the DCMI handle.
  * @retval None
  */
#define __HAL_DCMI_RESET_HANDLE_STATE(__HANDLE__) do{                                            \
                                                     (__HANDLE__)->State = HAL_DCMI_STATE_RESET; \
                                                     (__HANDLE__)->MspInitCallback = NULL;      \
                                                     (__HANDLE__)->MspDeInitCallback = NULL;    \
                                                   } while(0)

/**
  * @brief  Enable the DCMI.
  * @param  __HANDLE__ DCMI handle
  * @retval None
  */
#define __HAL_DCMI_ENABLE(__HANDLE__)    ((__HANDLE__)->Instance->CR |= DCMI_CR_ENABLE)

/**
  * @brief  Disable the DCMI.
  * @param  __HANDLE__ DCMI handle
  * @retval None
  */
#define __HAL_DCMI_DISABLE(__HANDLE__)   ((__HANDLE__)->Instance->CR &= ~(DCMI_CR_ENABLE))

/* Interrupt & Flag management */
/**
  * @brief  Get the DCMI pending flag.
  * @param  __HANDLE__ DCMI handle
  * @param  __FLAG__ Get the specified flag.
  *         This parameter can be one of the following values (no combination allowed)
  *            @arg DCMI_FLAG_HSYNC: HSYNC pin state (active line / synchronization between lines)
  *            @arg DCMI_FLAG_VSYNC: VSYNC pin state (active frame / synchronization between frames)
  *            @arg DCMI_FLAG_FNE: FIFO empty flag
  *            @arg DCMI_FLAG_FRAMERI: Frame capture complete flag mask
  *            @arg DCMI_FLAG_OVRRI: Overrun flag mask
  *            @arg DCMI_FLAG_ERRRI: Synchronization error flag mask
  *            @arg DCMI_FLAG_VSYNCRI: VSYNC flag mask
  *            @arg DCMI_FLAG_LINERI: Line flag mask
  *            @arg DCMI_FLAG_FRAMEMI: DCMI Capture complete masked interrupt status
  *            @arg DCMI_FLAG_OVRMI: DCMI Overrun masked interrupt status
  *            @arg DCMI_FLAG_ERRMI: DCMI Synchronization error masked interrupt status
  *            @arg DCMI_FLAG_VSYNCMI: DCMI VSYNC masked interrupt status
  *            @arg DCMI_FLAG_LINEMI: DCMI Line masked interrupt status
  * @retval The state of FLAG.
  */
#define __HAL_DCMI_GET_FLAG(__HANDLE__, __FLAG__)\
((((__FLAG__) & (DCMI_SR_INDEX|DCMI_MIS_INDEX)) == 0x0U)? ((__HANDLE__)->Instance->RISR & (__FLAG__)) :\
 (((__FLAG__) & DCMI_SR_INDEX) == 0x0U)? ((__HANDLE__)->Instance->MISR & (__FLAG__)) : ((__HANDLE__)->Instance->SR & (__FLAG__)))

/**
  * @brief  Clear the DCMI pending flags.
  * @param  __HANDLE__ DCMI handle
  * @param  __FLAG__ specifies the flag to clear.
  *         This parameter can be any combination of the following values:
  *            @arg DCMI_FLAG_FRAMERI: Frame capture complete flag mask
  *            @arg DCMI_FLAG_OVRRI: Overrun flag mask
  *            @arg DCMI_FLAG_ERRRI: Synchronization error flag mask
  *            @arg DCMI_FLAG_VSYNCRI: VSYNC flag mask
  *            @arg DCMI_FLAG_LINERI: Line flag mask
  * @retval None
  */
#define __HAL_DCMI_CLEAR_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->ICR = (__FLAG__))

/**
  * @brief  Enable the specified DCMI interrupts.
  * @param  __HANDLE__    DCMI handle
  * @param  __INTERRUPT__ specifies the DCMI interrupt sources to be enabled.
  *         This parameter can be any combination of the following values:
  *            @arg DCMI_IT_FRAME: Frame capture complete interrupt mask
  *            @arg DCMI_IT_OVR: Overrun interrupt mask
  *            @arg DCMI_IT_ERR: Synchronization error interrupt mask
  *            @arg DCMI_IT_VSYNC: VSYNC interrupt mask
  *            @arg DCMI_IT_LINE: Line interrupt mask
  * @retval None
  */
#define __HAL_DCMI_ENABLE_IT(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->IER |= (__INTERRUPT__))

/**
  * @brief  Disable the specified DCMI interrupts.
  * @param  __HANDLE__ DCMI handle
  * @param  __INTERRUPT__ specifies the DCMI interrupt sources to be enabled.
  *         This parameter can be any combination of the following values:
  *            @arg DCMI_IT_FRAME: Frame capture complete interrupt mask
  *            @arg DCMI_IT_OVR: Overrun interrupt mask
  *            @arg DCMI_IT_ERR: Synchronization error interrupt mask
  *            @arg DCMI_IT_VSYNC: VSYNC interrupt mask
  *            @arg DCMI_IT_LINE: Line interrupt mask
  * @retval None
  */
#define __HAL_DCMI_DISABLE_IT(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->IER &= ~(__INTERRUPT__))

/**
  * @brief  Check whether the specified DCMI interrupt has occurred or not.
  * @param  __HANDLE__ DCMI handle
  * @param  __INTERRUPT__ specifies the DCMI interrupt source to check.
  *         This parameter can be one of the following values:
  *            @arg DCMI_IT_FRAME: Frame capture complete interrupt mask
  *            @arg DCMI_IT_OVR: Overrun interrupt mask
  *            @arg DCMI_IT_ERR: Synchronization error interrupt mask
  *            @arg DCMI_IT_VSYNC: VSYNC interrupt mask
  *            @arg DCMI_IT_LINE: Line interrupt mask
  * @retval The state of INTERRUPT.
  */
#define __HAL_DCMI_GET_IT_SOURCE(__HANDLE__, __INTERRUPT__) ((__HANDLE__)->Instance->MISR & (__INTERRUPT__))

/**
  * @}
  */
  
/* Exported functions --------------------------------------------------------*/
/** @addtogroup DCMI_Exported_Functions DCMI Exported Functions
  * @{
  */

/** @addtogroup DCMI_Exported_Functions_Group1 Initialization and Configuration functions
 * @{
 */
/* Initialization and de-initialization functions *****************************/
HAL_StatusTypeDef HAL_DCMI_Init(DCMI_HandleTypeDef *hdcmi);
HAL_StatusTypeDef HAL_DCMI_DeInit(DCMI_HandleTypeDef *hdcmi);
void              HAL_DCMI_MspInit(DCMI_HandleTypeDef* hdcmi);
void              HAL_DCMI_MspDeInit(DCMI_HandleTypeDef* hdcmi);

/* Callbacks Register/UnRegister functions  ***********************************/
#if (USE_HAL_DCMI_REGISTER_CALLBACKS == 1)   
HAL_StatusTypeDef HAL_DCMI_RegisterCallback(DCMI_HandleTypeDef *hdcmi, HAL_DCMI_CallbackIDTypeDef CallbackID, pDCMI_CallbackTypeDef pCallback);
HAL_StatusTypeDef HAL_DCMI_UnRegisterCallback(DCMI_HandleTypeDef *hdcmi, HAL_DCMI_CallbackIDTypeDef CallbackID);
#endif /* USE_HAL_DCMI_REGISTER_CALLBACKS */
/**
  * @}
  */

/** @addtogroup DCMI_Exported_Functions_Group2 IO operation functions
 * @{
 */
/* IO operation functions *****************************************************/
HAL_StatusTypeDef HAL_DCMI_Start_DMA(DCMI_HandleTypeDef* hdcmi, uint32_t DCMI_Mode, uint32_t pData, uint32_t Length);
HAL_StatusTypeDef HAL_DCMI_Stop(DCMI_HandleTypeDef* hdcmi);
HAL_StatusTypeDef HAL_DCMI_Suspend(DCMI_HandleTypeDef* hdcmi);
HAL_StatusTypeDef HAL_DCMI_Resume(DCMI_HandleTypeDef* hdcmi);
void              HAL_DCMI_ErrorCallback(DCMI_HandleTypeDef *hdcmi);
void              HAL_DCMI_LineEventCallback(DCMI_HandleTypeDef *hdcmi);
void              HAL_DCMI_FrameEventCallback(DCMI_HandleTypeDef *hdcmi);
void              HAL_DCMI_VsyncEventCallback(DCMI_HandleTypeDef *hdcmi);
void              HAL_DCMI_VsyncCallback(DCMI_HandleTypeDef *hdcmi);
void              HAL_DCMI_HsyncCallback(DCMI_HandleTypeDef *hdcmi);
void              HAL_DCMI_IRQHandler(DCMI_HandleTypeDef *hdcmi);
/**
  * @}
  */
  
/** @addtogroup DCMI_Exported_Functions_Group3 Peripheral Control functions
 * @{
 */
/* Peripheral Control functions ***********************************************/
HAL_StatusTypeDef HAL_DCMI_ConfigCrop(DCMI_HandleTypeDef *hdcmi, uint32_t X0, uint32_t Y0, uint32_t XSize, uint32_t YSize);
HAL_StatusTypeDef HAL_DCMI_EnableCrop(DCMI_HandleTypeDef *hdcmi);
HAL_StatusTypeDef HAL_DCMI_DisableCrop(DCMI_HandleTypeDef *hdcmi);
HAL_StatusTypeDef HAL_DCMI_ConfigSyncUnmask(DCMI_HandleTypeDef *hdcmi, DCMI_SyncUnmaskTypeDef *SyncUnmask);
/**
  * @}
  */
  
/** @addtogroup DCMI_Exported_Functions_Group4 Peripheral State functions
 * @{
 */
/* Peripheral State functions *************************************************/
HAL_DCMI_StateTypeDef HAL_DCMI_GetState(DCMI_HandleTypeDef *hdcmi);
uint32_t              HAL_DCMI_GetError(DCMI_HandleTypeDef *hdcmi);
/**
  * @}
  */

/**
  * @}
  */

/* Private types -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup DCMI_Private_Constants DCMI Private Constants
  * @{
  */
#define DCMI_MIS_INDEX        0x1000U /*!< DCMI MIS register index */
#define DCMI_SR_INDEX         0x2000U /*!< DCMI SR register index  */
/**
  * @}
  */   
/* Private macro -------------------------------------------------------------*/
/** @defgroup DCMI_Private_Macros DCMI Private Macros
  * @{
  */
#define IS_DCMI_CAPTURE_MODE(MODE)(((MODE) == DCMI_MODE_CONTINUOUS) || \
                                   ((MODE) == DCMI_MODE_SNAPSHOT))

#define IS_DCMI_SYNCHRO(MODE)(((MODE) == DCMI_SYNCHRO_HARDWARE) || \
                              ((MODE) == DCMI_SYNCHRO_EMBEDDED))
                              
#define IS_DCMI_PCKPOLARITY(POLARITY)(((POLARITY) == DCMI_PCKPOLARITY_FALLING) || \
                                      ((POLARITY) == DCMI_PCKPOLARITY_RISING))
                                      
#define IS_DCMI_VSPOLARITY(POLARITY)(((POLARITY) == DCMI_VSPOLARITY_LOW) || \
                                     ((POLARITY) == DCMI_VSPOLARITY_HIGH))
                                     
#define IS_DCMI_HSPOLARITY(POLARITY)(((POLARITY) == DCMI_HSPOLARITY_LOW) || \
                                     ((POLARITY) == DCMI_HSPOLARITY_HIGH))
                                     
#define IS_DCMI_MODE_JPEG(JPEG_MODE)(((JPEG_MODE) == DCMI_JPEG_DISABLE) || \
                                     ((JPEG_MODE) == DCMI_JPEG_ENABLE))
                                     
#define IS_DCMI_CAPTURE_RATE(RATE) (((RATE) == DCMI_CR_ALL_FRAME)         || \
                                    ((RATE) == DCMI_CR_ALTERNATE_2_FRAME) || \
                                    ((RATE) == DCMI_CR_ALTERNATE_4_FRAME))
                                    
#define IS_DCMI_EXTENDED_DATA(DATA)(((DATA) == DCMI_EXTEND_DATA_8B)  || \
                                    ((DATA) == DCMI_EXTEND_DATA_10B) || \
                                    ((DATA) == DCMI_EXTEND_DATA_12B) || \
                                    ((DATA) == DCMI_EXTEND_DATA_14B))
                                    
#define IS_DCMI_WINDOW_COORDINATE(COORDINATE) ((COORDINATE) <= DCMI_WINDOW_COORDINATE)

#define IS_DCMI_WINDOW_HEIGHT(HEIGHT) ((HEIGHT) <= DCMI_WINDOW_HEIGHT)

/**
  * @}
  */

/* Private functions ---------------------------------------------------------*/
/** @addtogroup DCMI_Private_Functions DCMI Private Functions
  * @{
  */
  
/**
  * @}
  */
      
#endif /* STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx ||\
          STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx ||\
          STM32F479xx */

/**
  * @}
  */
    
/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __STM32F4xx_HAL_DCMI_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
