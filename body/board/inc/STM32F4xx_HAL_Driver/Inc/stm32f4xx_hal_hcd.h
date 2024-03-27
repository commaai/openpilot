/**
  ******************************************************************************
  * @file    stm32f4xx_hal_hcd.h
  * @author  MCD Application Team
  * @brief   Header file of HCD HAL module.
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
#ifndef STM32F4xx_HAL_HCD_H
#define STM32F4xx_HAL_HCD_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_ll_usb.h"

#if defined (USB_OTG_FS) || defined (USB_OTG_HS)
/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @addtogroup HCD HCD
  * @{
  */

/* Exported types ------------------------------------------------------------*/
/** @defgroup HCD_Exported_Types HCD Exported Types
  * @{
  */

/** @defgroup HCD_Exported_Types_Group1 HCD State Structure definition
  * @{
  */
typedef enum
{
  HAL_HCD_STATE_RESET    = 0x00,
  HAL_HCD_STATE_READY    = 0x01,
  HAL_HCD_STATE_ERROR    = 0x02,
  HAL_HCD_STATE_BUSY     = 0x03,
  HAL_HCD_STATE_TIMEOUT  = 0x04
} HCD_StateTypeDef;

typedef USB_OTG_GlobalTypeDef   HCD_TypeDef;
typedef USB_OTG_CfgTypeDef      HCD_InitTypeDef;
typedef USB_OTG_HCTypeDef       HCD_HCTypeDef;
typedef USB_OTG_URBStateTypeDef HCD_URBStateTypeDef;
typedef USB_OTG_HCStateTypeDef  HCD_HCStateTypeDef;
/**
  * @}
  */

/** @defgroup HCD_Exported_Types_Group2 HCD Handle Structure definition
  * @{
  */
#if (USE_HAL_HCD_REGISTER_CALLBACKS == 1U)
typedef struct __HCD_HandleTypeDef
#else
typedef struct
#endif /* USE_HAL_HCD_REGISTER_CALLBACKS */
{
  HCD_TypeDef               *Instance;  /*!< Register base address    */
  HCD_InitTypeDef           Init;       /*!< HCD required parameters  */
  HCD_HCTypeDef             hc[16];     /*!< Host channels parameters */
  HAL_LockTypeDef           Lock;       /*!< HCD peripheral status    */
  __IO HCD_StateTypeDef     State;      /*!< HCD communication state  */
  __IO  uint32_t            ErrorCode;  /*!< HCD Error code           */
  void                      *pData;     /*!< Pointer Stack Handler    */
#if (USE_HAL_HCD_REGISTER_CALLBACKS == 1U)
  void (* SOFCallback)(struct __HCD_HandleTypeDef *hhcd);                               /*!< USB OTG HCD SOF callback                */
  void (* ConnectCallback)(struct __HCD_HandleTypeDef *hhcd);                           /*!< USB OTG HCD Connect callback            */
  void (* DisconnectCallback)(struct __HCD_HandleTypeDef *hhcd);                        /*!< USB OTG HCD Disconnect callback         */
  void (* PortEnabledCallback)(struct __HCD_HandleTypeDef *hhcd);                       /*!< USB OTG HCD Port Enable callback        */
  void (* PortDisabledCallback)(struct __HCD_HandleTypeDef *hhcd);                      /*!< USB OTG HCD Port Disable callback       */
  void (* HC_NotifyURBChangeCallback)(struct __HCD_HandleTypeDef *hhcd, uint8_t chnum,
                                      HCD_URBStateTypeDef urb_state);                   /*!< USB OTG HCD Host Channel Notify URB Change callback  */

  void (* MspInitCallback)(struct __HCD_HandleTypeDef *hhcd);                           /*!< USB OTG HCD Msp Init callback           */
  void (* MspDeInitCallback)(struct __HCD_HandleTypeDef *hhcd);                         /*!< USB OTG HCD Msp DeInit callback         */
#endif /* USE_HAL_HCD_REGISTER_CALLBACKS */
} HCD_HandleTypeDef;
/**
  * @}
  */

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup HCD_Exported_Constants HCD Exported Constants
  * @{
  */

/** @defgroup HCD_Speed HCD Speed
  * @{
  */
#define HCD_SPEED_HIGH               USBH_HS_SPEED
#define HCD_SPEED_FULL               USBH_FSLS_SPEED
#define HCD_SPEED_LOW                USBH_FSLS_SPEED
/**
  * @}
  */

/** @defgroup HCD_Device_Speed HCD Device Speed
  * @{
  */
#define HCD_DEVICE_SPEED_HIGH               0U
#define HCD_DEVICE_SPEED_FULL               1U
#define HCD_DEVICE_SPEED_LOW                2U
/**
  * @}
  */

/** @defgroup HCD_PHY_Module HCD PHY Module
  * @{
  */
#define HCD_PHY_ULPI                 1U
#define HCD_PHY_EMBEDDED             2U
/**
  * @}
  */

/** @defgroup HCD_Error_Code_definition HCD Error Code definition
  * @brief  HCD Error Code definition
  * @{
  */
#if (USE_HAL_HCD_REGISTER_CALLBACKS == 1U)
#define  HAL_HCD_ERROR_INVALID_CALLBACK                        (0x00000010U)    /*!< Invalid Callback error  */
#endif /* USE_HAL_HCD_REGISTER_CALLBACKS */

/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup HCD_Exported_Macros HCD Exported Macros
  *  @brief macros to handle interrupts and specific clock configurations
  * @{
  */
#define __HAL_HCD_ENABLE(__HANDLE__)                   (void)USB_EnableGlobalInt ((__HANDLE__)->Instance)
#define __HAL_HCD_DISABLE(__HANDLE__)                  (void)USB_DisableGlobalInt ((__HANDLE__)->Instance)

#define __HAL_HCD_GET_FLAG(__HANDLE__, __INTERRUPT__)      ((USB_ReadInterrupts((__HANDLE__)->Instance)\
                                                             & (__INTERRUPT__)) == (__INTERRUPT__))
#define __HAL_HCD_CLEAR_FLAG(__HANDLE__, __INTERRUPT__)    (((__HANDLE__)->Instance->GINTSTS) = (__INTERRUPT__))
#define __HAL_HCD_IS_INVALID_INTERRUPT(__HANDLE__)         (USB_ReadInterrupts((__HANDLE__)->Instance) == 0U)

#define __HAL_HCD_CLEAR_HC_INT(chnum, __INTERRUPT__)  (USBx_HC(chnum)->HCINT = (__INTERRUPT__))
#define __HAL_HCD_MASK_HALT_HC_INT(chnum)             (USBx_HC(chnum)->HCINTMSK &= ~USB_OTG_HCINTMSK_CHHM)
#define __HAL_HCD_UNMASK_HALT_HC_INT(chnum)           (USBx_HC(chnum)->HCINTMSK |= USB_OTG_HCINTMSK_CHHM)
#define __HAL_HCD_MASK_ACK_HC_INT(chnum)              (USBx_HC(chnum)->HCINTMSK &= ~USB_OTG_HCINTMSK_ACKM)
#define __HAL_HCD_UNMASK_ACK_HC_INT(chnum)            (USBx_HC(chnum)->HCINTMSK |= USB_OTG_HCINTMSK_ACKM)
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @addtogroup HCD_Exported_Functions HCD Exported Functions
  * @{
  */

/** @defgroup HCD_Exported_Functions_Group1 Initialization and de-initialization functions
  * @{
  */
HAL_StatusTypeDef HAL_HCD_Init(HCD_HandleTypeDef *hhcd);
HAL_StatusTypeDef HAL_HCD_DeInit(HCD_HandleTypeDef *hhcd);
HAL_StatusTypeDef HAL_HCD_HC_Init(HCD_HandleTypeDef *hhcd, uint8_t ch_num,
                                  uint8_t epnum, uint8_t dev_address,
                                  uint8_t speed, uint8_t ep_type, uint16_t mps);

HAL_StatusTypeDef HAL_HCD_HC_Halt(HCD_HandleTypeDef *hhcd, uint8_t ch_num);
void              HAL_HCD_MspInit(HCD_HandleTypeDef *hhcd);
void              HAL_HCD_MspDeInit(HCD_HandleTypeDef *hhcd);

#if (USE_HAL_HCD_REGISTER_CALLBACKS == 1U)
/** @defgroup HAL_HCD_Callback_ID_enumeration_definition HAL USB OTG HCD Callback ID enumeration definition
  * @brief  HAL USB OTG HCD Callback ID enumeration definition
  * @{
  */
typedef enum
{
  HAL_HCD_SOF_CB_ID            = 0x01,       /*!< USB HCD SOF callback ID           */
  HAL_HCD_CONNECT_CB_ID        = 0x02,       /*!< USB HCD Connect callback ID       */
  HAL_HCD_DISCONNECT_CB_ID     = 0x03,       /*!< USB HCD Disconnect callback ID    */
  HAL_HCD_PORT_ENABLED_CB_ID   = 0x04,       /*!< USB HCD Port Enable callback ID   */
  HAL_HCD_PORT_DISABLED_CB_ID  = 0x05,       /*!< USB HCD Port Disable callback ID  */

  HAL_HCD_MSPINIT_CB_ID        = 0x06,       /*!< USB HCD MspInit callback ID       */
  HAL_HCD_MSPDEINIT_CB_ID      = 0x07        /*!< USB HCD MspDeInit callback ID     */

} HAL_HCD_CallbackIDTypeDef;
/**
  * @}
  */

/** @defgroup HAL_HCD_Callback_pointer_definition HAL USB OTG HCD Callback pointer definition
  * @brief  HAL USB OTG HCD Callback pointer definition
  * @{
  */

typedef void (*pHCD_CallbackTypeDef)(HCD_HandleTypeDef *hhcd);                   /*!< pointer to a common USB OTG HCD callback function  */
typedef void (*pHCD_HC_NotifyURBChangeCallbackTypeDef)(HCD_HandleTypeDef *hhcd,
                                                       uint8_t epnum,
                                                       HCD_URBStateTypeDef urb_state);   /*!< pointer to USB OTG HCD host channel  callback */
/**
  * @}
  */

HAL_StatusTypeDef HAL_HCD_RegisterCallback(HCD_HandleTypeDef *hhcd,
                                           HAL_HCD_CallbackIDTypeDef CallbackID,
                                           pHCD_CallbackTypeDef pCallback);

HAL_StatusTypeDef HAL_HCD_UnRegisterCallback(HCD_HandleTypeDef *hhcd,
                                             HAL_HCD_CallbackIDTypeDef CallbackID);

HAL_StatusTypeDef HAL_HCD_RegisterHC_NotifyURBChangeCallback(HCD_HandleTypeDef *hhcd,
                                                             pHCD_HC_NotifyURBChangeCallbackTypeDef pCallback);

HAL_StatusTypeDef HAL_HCD_UnRegisterHC_NotifyURBChangeCallback(HCD_HandleTypeDef *hhcd);
#endif /* USE_HAL_HCD_REGISTER_CALLBACKS */
/**
  * @}
  */

/* I/O operation functions  ***************************************************/
/** @addtogroup HCD_Exported_Functions_Group2 Input and Output operation functions
  * @{
  */
HAL_StatusTypeDef HAL_HCD_HC_SubmitRequest(HCD_HandleTypeDef *hhcd, uint8_t ch_num,
                                           uint8_t direction, uint8_t ep_type,
                                           uint8_t token, uint8_t *pbuff,
                                           uint16_t length, uint8_t do_ping);

/* Non-Blocking mode: Interrupt */
void HAL_HCD_IRQHandler(HCD_HandleTypeDef *hhcd);
void HAL_HCD_WKUP_IRQHandler(HCD_HandleTypeDef *hhcd);
void HAL_HCD_SOF_Callback(HCD_HandleTypeDef *hhcd);
void HAL_HCD_Connect_Callback(HCD_HandleTypeDef *hhcd);
void HAL_HCD_Disconnect_Callback(HCD_HandleTypeDef *hhcd);
void HAL_HCD_PortEnabled_Callback(HCD_HandleTypeDef *hhcd);
void HAL_HCD_PortDisabled_Callback(HCD_HandleTypeDef *hhcd);
void HAL_HCD_HC_NotifyURBChange_Callback(HCD_HandleTypeDef *hhcd, uint8_t chnum,
                                         HCD_URBStateTypeDef urb_state);
/**
  * @}
  */

/* Peripheral Control functions  **********************************************/
/** @addtogroup HCD_Exported_Functions_Group3 Peripheral Control functions
  * @{
  */
HAL_StatusTypeDef HAL_HCD_ResetPort(HCD_HandleTypeDef *hhcd);
HAL_StatusTypeDef HAL_HCD_Start(HCD_HandleTypeDef *hhcd);
HAL_StatusTypeDef HAL_HCD_Stop(HCD_HandleTypeDef *hhcd);
/**
  * @}
  */

/* Peripheral State functions  ************************************************/
/** @addtogroup HCD_Exported_Functions_Group4 Peripheral State functions
  * @{
  */
HCD_StateTypeDef        HAL_HCD_GetState(HCD_HandleTypeDef *hhcd);
HCD_URBStateTypeDef     HAL_HCD_HC_GetURBState(HCD_HandleTypeDef *hhcd, uint8_t chnum);
HCD_HCStateTypeDef      HAL_HCD_HC_GetState(HCD_HandleTypeDef *hhcd, uint8_t chnum);
uint32_t                HAL_HCD_HC_GetXferCount(HCD_HandleTypeDef *hhcd, uint8_t chnum);
uint32_t                HAL_HCD_GetCurrentFrame(HCD_HandleTypeDef *hhcd);
uint32_t                HAL_HCD_GetCurrentSpeed(HCD_HandleTypeDef *hhcd);

/**
  * @}
  */

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup HCD_Private_Macros HCD Private Macros
  * @{
  */
/**
  * @}
  */
/* Private functions prototypes ----------------------------------------------*/

/**
  * @}
  */
/**
  * @}
  */
#endif /* defined (USB_OTG_FS) || defined (USB_OTG_HS) */

#ifdef __cplusplus
}
#endif

#endif /* STM32F4xx_HAL_HCD_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
