/**
  ******************************************************************************
  * @file    stm32f4xx_hal_exti.h
  * @author  MCD Application Team
  * @brief   Header file of EXTI HAL module.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2018 STMicroelectronics.
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
#ifndef STM32f4xx_HAL_EXTI_H
#define STM32f4xx_HAL_EXTI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal_def.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup EXTI EXTI
  * @brief EXTI HAL module driver
  * @{
  */

/* Exported types ------------------------------------------------------------*/

/** @defgroup EXTI_Exported_Types EXTI Exported Types
  * @{
  */
typedef enum
{
  HAL_EXTI_COMMON_CB_ID          = 0x00U
} EXTI_CallbackIDTypeDef;

/**
  * @brief  EXTI Handle structure definition
  */
typedef struct
{
  uint32_t Line;                    /*!<  Exti line number */
  void (* PendingCallback)(void);   /*!<  Exti pending callback */
} EXTI_HandleTypeDef;

/**
  * @brief  EXTI Configuration structure definition
  */
typedef struct
{
  uint32_t Line;      /*!< The Exti line to be configured. This parameter
                           can be a value of @ref EXTI_Line */
  uint32_t Mode;      /*!< The Exit Mode to be configured for a core.
                           This parameter can be a combination of @ref EXTI_Mode */
  uint32_t Trigger;   /*!< The Exti Trigger to be configured. This parameter
                           can be a value of @ref EXTI_Trigger */
  uint32_t GPIOSel;   /*!< The Exti GPIO multiplexer selection to be configured.
                           This parameter is only possible for line 0 to 15. It
                           can be a value of @ref EXTI_GPIOSel */
} EXTI_ConfigTypeDef;

/**
  * @}
  */

/* Exported constants --------------------------------------------------------*/
/** @defgroup EXTI_Exported_Constants EXTI Exported Constants
  * @{
  */

/** @defgroup EXTI_Line  EXTI Line
  * @{
  */
#define EXTI_LINE_0                        (EXTI_GPIO       | 0x00u)    /*!< External interrupt line 0 */
#define EXTI_LINE_1                        (EXTI_GPIO       | 0x01u)    /*!< External interrupt line 1 */
#define EXTI_LINE_2                        (EXTI_GPIO       | 0x02u)    /*!< External interrupt line 2 */
#define EXTI_LINE_3                        (EXTI_GPIO       | 0x03u)    /*!< External interrupt line 3 */
#define EXTI_LINE_4                        (EXTI_GPIO       | 0x04u)    /*!< External interrupt line 4 */
#define EXTI_LINE_5                        (EXTI_GPIO       | 0x05u)    /*!< External interrupt line 5 */
#define EXTI_LINE_6                        (EXTI_GPIO       | 0x06u)    /*!< External interrupt line 6 */
#define EXTI_LINE_7                        (EXTI_GPIO       | 0x07u)    /*!< External interrupt line 7 */
#define EXTI_LINE_8                        (EXTI_GPIO       | 0x08u)    /*!< External interrupt line 8 */
#define EXTI_LINE_9                        (EXTI_GPIO       | 0x09u)    /*!< External interrupt line 9 */
#define EXTI_LINE_10                       (EXTI_GPIO       | 0x0Au)    /*!< External interrupt line 10 */
#define EXTI_LINE_11                       (EXTI_GPIO       | 0x0Bu)    /*!< External interrupt line 11 */
#define EXTI_LINE_12                       (EXTI_GPIO       | 0x0Cu)    /*!< External interrupt line 12 */
#define EXTI_LINE_13                       (EXTI_GPIO       | 0x0Du)    /*!< External interrupt line 13 */
#define EXTI_LINE_14                       (EXTI_GPIO       | 0x0Eu)    /*!< External interrupt line 14 */
#define EXTI_LINE_15                       (EXTI_GPIO       | 0x0Fu)    /*!< External interrupt line 15 */
#define EXTI_LINE_16                       (EXTI_CONFIG     | 0x10u)    /*!< External interrupt line 16 Connected to the PVD Output */
#define EXTI_LINE_17                       (EXTI_CONFIG     | 0x11u)    /*!< External interrupt line 17 Connected to the RTC Alarm event */
#if defined(EXTI_IMR_IM18)
#define EXTI_LINE_18                       (EXTI_CONFIG     | 0x12u)    /*!< External interrupt line 18 Connected to the USB OTG FS Wakeup from suspend event */
#else
#define EXTI_LINE_18                       (EXTI_RESERVED   | 0x12u)    /*!< No interrupt supported in this line */
#endif /* EXTI_IMR_IM18 */
#if defined(EXTI_IMR_IM19)
#define EXTI_LINE_19                       (EXTI_CONFIG     | 0x13u)    /*!< External interrupt line 19 Connected to the Ethernet Wakeup event */
#else
#define EXTI_LINE_19                       (EXTI_RESERVED   | 0x13u)    /*!< No interrupt supported in this line */
#endif /* EXTI_IMR_IM19 */
#if defined(EXTI_IMR_IM20)
#define EXTI_LINE_20                       (EXTI_CONFIG     | 0x14u)    /*!< External interrupt line 20 Connected to the USB OTG HS (configured in FS) Wakeup event  */
#else
#define EXTI_LINE_20                       (EXTI_RESERVED   | 0x14u)    /*!< No interrupt supported in this line */
#endif /* EXTI_IMR_IM20 */
#define EXTI_LINE_21                       (EXTI_CONFIG     | 0x15u)    /*!< External interrupt line 21 Connected to the RTC Tamper and Time Stamp events */
#define EXTI_LINE_22                       (EXTI_CONFIG     | 0x16u)    /*!< External interrupt line 22 Connected to the RTC Wakeup event */
#if defined(EXTI_IMR_IM23)
#define EXTI_LINE_23                       (EXTI_CONFIG     | 0x17u)    /*!< External interrupt line 23 Connected to the LPTIM1 asynchronous event */
#endif /* EXTI_IMR_IM23 */

/**
  * @}
  */

/** @defgroup EXTI_Mode  EXTI Mode
  * @{
  */
#define EXTI_MODE_NONE                      0x00000000u
#define EXTI_MODE_INTERRUPT                 0x00000001u
#define EXTI_MODE_EVENT                     0x00000002u
/**
  * @}
  */

/** @defgroup EXTI_Trigger  EXTI Trigger
  * @{
  */

#define EXTI_TRIGGER_NONE                   0x00000000u
#define EXTI_TRIGGER_RISING                 0x00000001u
#define EXTI_TRIGGER_FALLING                0x00000002u
#define EXTI_TRIGGER_RISING_FALLING         (EXTI_TRIGGER_RISING | EXTI_TRIGGER_FALLING)
/**
  * @}
  */

/** @defgroup EXTI_GPIOSel  EXTI GPIOSel
  * @brief
  * @{
  */
#define EXTI_GPIOA                          0x00000000u
#define EXTI_GPIOB                          0x00000001u
#define EXTI_GPIOC                          0x00000002u
#if defined (GPIOD)
#define EXTI_GPIOD                          0x00000003u
#endif /* GPIOD */
#if defined (GPIOE)
#define EXTI_GPIOE                          0x00000004u
#endif /* GPIOE */
#if defined (GPIOF)
#define EXTI_GPIOF                          0x00000005u
#endif /* GPIOF */
#if defined (GPIOG)
#define EXTI_GPIOG                          0x00000006u
#endif /* GPIOG */
#if defined (GPIOH)
#define EXTI_GPIOH                          0x00000007u
#endif /* GPIOH */
#if defined (GPIOI)
#define EXTI_GPIOI                          0x00000008u
#endif /* GPIOI */
#if defined (GPIOJ)
#define EXTI_GPIOJ                          0x00000009u
#endif /* GPIOJ */
#if defined (GPIOK)
#define EXTI_GPIOK                          0x0000000Au
#endif /* GPIOK */

/**
  * @}
  */

/**
  * @}
  */

/* Exported macro ------------------------------------------------------------*/
/** @defgroup EXTI_Exported_Macros EXTI Exported Macros
  * @{
  */

/**
  * @}
  */

/* Private constants --------------------------------------------------------*/
/** @defgroup EXTI_Private_Constants EXTI Private Constants
  * @{
  */
/**
  * @brief  EXTI Line property definition
  */
#define EXTI_PROPERTY_SHIFT                  24u
#define EXTI_CONFIG                         (0x02uL << EXTI_PROPERTY_SHIFT)
#define EXTI_GPIO                           ((0x04uL << EXTI_PROPERTY_SHIFT) | EXTI_CONFIG)
#define EXTI_RESERVED                       (0x08uL << EXTI_PROPERTY_SHIFT)
#define EXTI_PROPERTY_MASK                  (EXTI_CONFIG | EXTI_GPIO)

/**
  * @brief  EXTI bit usage
  */
#define EXTI_PIN_MASK                       0x0000001Fu

/**
  * @brief  EXTI Mask for interrupt & event mode
  */
#define EXTI_MODE_MASK                      (EXTI_MODE_EVENT | EXTI_MODE_INTERRUPT)

/**
  * @brief  EXTI Mask for trigger possibilities
  */
#define EXTI_TRIGGER_MASK                   (EXTI_TRIGGER_RISING | EXTI_TRIGGER_FALLING)

/**
  * @brief  EXTI Line number
  */
#if defined(EXTI_IMR_IM23)
#define EXTI_LINE_NB                        24UL
#else
#define EXTI_LINE_NB                        23UL
#endif /* EXTI_IMR_IM23 */

/**
  * @}
  */

/* Private macros ------------------------------------------------------------*/
/** @defgroup EXTI_Private_Macros EXTI Private Macros
  * @{
  */
#define IS_EXTI_LINE(__EXTI_LINE__)          ((((__EXTI_LINE__) & ~(EXTI_PROPERTY_MASK | EXTI_PIN_MASK)) == 0x00u) && \
                                             ((((__EXTI_LINE__) & EXTI_PROPERTY_MASK) == EXTI_CONFIG)              || \
                                              (((__EXTI_LINE__) & EXTI_PROPERTY_MASK) == EXTI_GPIO))               && \
                                              (((__EXTI_LINE__) & EXTI_PIN_MASK) < EXTI_LINE_NB))

#define IS_EXTI_MODE(__EXTI_LINE__)          ((((__EXTI_LINE__) & EXTI_MODE_MASK) != 0x00u) && \
                                              (((__EXTI_LINE__) & ~EXTI_MODE_MASK) == 0x00u))

#define IS_EXTI_TRIGGER(__EXTI_LINE__)       (((__EXTI_LINE__)  & ~EXTI_TRIGGER_MASK) == 0x00u)

#define IS_EXTI_PENDING_EDGE(__EXTI_LINE__)  ((__EXTI_LINE__) == EXTI_TRIGGER_RISING_FALLING)

#define IS_EXTI_CONFIG_LINE(__EXTI_LINE__)   (((__EXTI_LINE__) & EXTI_CONFIG) != 0x00u)

#if !defined (GPIOD)
#define IS_EXTI_GPIO_PORT(__PORT__)     (((__PORT__) == EXTI_GPIOA) || \
                                         ((__PORT__) == EXTI_GPIOB) || \
                                         ((__PORT__) == EXTI_GPIOC) || \
                                         ((__PORT__) == EXTI_GPIOH))
#elif !defined (GPIOE)
#define IS_EXTI_GPIO_PORT(__PORT__)     (((__PORT__) == EXTI_GPIOA) || \
                                         ((__PORT__) == EXTI_GPIOB) || \
                                         ((__PORT__) == EXTI_GPIOC) || \
                                         ((__PORT__) == EXTI_GPIOD) || \
                                         ((__PORT__) == EXTI_GPIOH))
#elif !defined (GPIOF)
#define IS_EXTI_GPIO_PORT(__PORT__)     (((__PORT__) == EXTI_GPIOA) || \
                                         ((__PORT__) == EXTI_GPIOB) || \
                                         ((__PORT__) == EXTI_GPIOC) || \
                                         ((__PORT__) == EXTI_GPIOD) || \
                                         ((__PORT__) == EXTI_GPIOE) || \
                                         ((__PORT__) == EXTI_GPIOH))
#elif !defined (GPIOI)
#define IS_EXTI_GPIO_PORT(__PORT__)     (((__PORT__) == EXTI_GPIOA) || \
                                         ((__PORT__) == EXTI_GPIOB) || \
                                         ((__PORT__) == EXTI_GPIOC) || \
                                         ((__PORT__) == EXTI_GPIOD) || \
                                         ((__PORT__) == EXTI_GPIOE) || \
                                         ((__PORT__) == EXTI_GPIOF) || \
                                         ((__PORT__) == EXTI_GPIOG) || \
                                         ((__PORT__) == EXTI_GPIOH))
#elif !defined (GPIOJ)
#define IS_EXTI_GPIO_PORT(__PORT__)     (((__PORT__) == EXTI_GPIOA) || \
                                         ((__PORT__) == EXTI_GPIOB) || \
                                         ((__PORT__) == EXTI_GPIOC) || \
                                         ((__PORT__) == EXTI_GPIOD) || \
                                         ((__PORT__) == EXTI_GPIOE) || \
                                         ((__PORT__) == EXTI_GPIOF) || \
                                         ((__PORT__) == EXTI_GPIOG) || \
                                         ((__PORT__) == EXTI_GPIOH) || \
                                         ((__PORT__) == EXTI_GPIOI))
#else
#define IS_EXTI_GPIO_PORT(__PORT__)     (((__PORT__) == EXTI_GPIOA) || \
                                         ((__PORT__) == EXTI_GPIOB) || \
                                         ((__PORT__) == EXTI_GPIOC) || \
                                         ((__PORT__) == EXTI_GPIOD) || \
                                         ((__PORT__) == EXTI_GPIOE) || \
                                         ((__PORT__) == EXTI_GPIOF) || \
                                         ((__PORT__) == EXTI_GPIOG) || \
                                         ((__PORT__) == EXTI_GPIOH) || \
                                         ((__PORT__) == EXTI_GPIOI) || \
                                         ((__PORT__) == EXTI_GPIOJ) || \
                                         ((__PORT__) == EXTI_GPIOK))
#endif /* GPIOD */

#define IS_EXTI_GPIO_PIN(__PIN__)       ((__PIN__) < 16U)
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup EXTI_Exported_Functions EXTI Exported Functions
  * @brief    EXTI Exported Functions
  * @{
  */

/** @defgroup EXTI_Exported_Functions_Group1 Configuration functions
  * @brief    Configuration functions
  * @{
  */
/* Configuration functions ****************************************************/
HAL_StatusTypeDef HAL_EXTI_SetConfigLine(EXTI_HandleTypeDef *hexti, EXTI_ConfigTypeDef *pExtiConfig);
HAL_StatusTypeDef HAL_EXTI_GetConfigLine(EXTI_HandleTypeDef *hexti, EXTI_ConfigTypeDef *pExtiConfig);
HAL_StatusTypeDef HAL_EXTI_ClearConfigLine(EXTI_HandleTypeDef *hexti);
HAL_StatusTypeDef HAL_EXTI_RegisterCallback(EXTI_HandleTypeDef *hexti, EXTI_CallbackIDTypeDef CallbackID, void (*pPendingCbfn)(void));
HAL_StatusTypeDef HAL_EXTI_GetHandle(EXTI_HandleTypeDef *hexti, uint32_t ExtiLine);
/**
  * @}
  */

/** @defgroup EXTI_Exported_Functions_Group2 IO operation functions
  * @brief    IO operation functions
  * @{
  */
/* IO operation functions *****************************************************/
void HAL_EXTI_IRQHandler(EXTI_HandleTypeDef *hexti);
uint32_t HAL_EXTI_GetPending(EXTI_HandleTypeDef *hexti, uint32_t Edge);
void HAL_EXTI_ClearPending(EXTI_HandleTypeDef *hexti, uint32_t Edge);
void HAL_EXTI_GenerateSWI(EXTI_HandleTypeDef *hexti);

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

#endif /* STM32f4xx_HAL_EXTI_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
