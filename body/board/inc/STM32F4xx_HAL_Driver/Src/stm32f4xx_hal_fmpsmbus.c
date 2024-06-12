/**
  ******************************************************************************
  * @file    stm32f4xx_hal_fmpsmbus.c
  * @author  MCD Application Team
  * @brief   FMPSMBUS HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the System Management Bus (SMBus) peripheral,
  *          based on I2C principles of operation :
  *           + Initialization and de-initialization functions
  *           + IO operation functions
  *           + Peripheral State and Errors functions
  *
  @verbatim
  ==============================================================================
                        ##### How to use this driver #####
  ==============================================================================
    [..]
    The FMPSMBUS HAL driver can be used as follows:

    (#) Declare a FMPSMBUS_HandleTypeDef handle structure, for example:
        FMPSMBUS_HandleTypeDef  hfmpsmbus;

    (#)Initialize the FMPSMBUS low level resources by implementing the HAL_FMPSMBUS_MspInit() API:
        (##) Enable the FMPSMBUSx interface clock
        (##) FMPSMBUS pins configuration
            (+++) Enable the clock for the FMPSMBUS GPIOs
            (+++) Configure FMPSMBUS pins as alternate function open-drain
        (##) NVIC configuration if you need to use interrupt process
            (+++) Configure the FMPSMBUSx interrupt priority
            (+++) Enable the NVIC FMPSMBUS IRQ Channel

    (#) Configure the Communication Clock Timing, Bus Timeout, Own Address1, Master Addressing mode,
        Dual Addressing mode, Own Address2, Own Address2 Mask, General call, Nostretch mode,
        Peripheral mode and Packet Error Check mode in the hfmpsmbus Init structure.

    (#) Initialize the FMPSMBUS registers by calling the HAL_FMPSMBUS_Init() API:
        (++) These API's configures also the low level Hardware GPIO, CLOCK, CORTEX...etc)
             by calling the customized HAL_FMPSMBUS_MspInit(&hfmpsmbus) API.

    (#) To check if target device is ready for communication, use the function HAL_FMPSMBUS_IsDeviceReady()

    (#) For FMPSMBUS IO operations, only one mode of operations is available within this driver

    *** Interrupt mode IO operation ***
    ===================================
    [..]
      (+) Transmit in master/host FMPSMBUS mode an amount of data in non-blocking mode using HAL_FMPSMBUS_Master_Transmit_IT()
      (++) At transmission end of transfer HAL_FMPSMBUS_MasterTxCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_FMPSMBUS_MasterTxCpltCallback()
      (+) Receive in master/host FMPSMBUS mode an amount of data in non-blocking mode using HAL_FMPSMBUS_Master_Receive_IT()
      (++) At reception end of transfer HAL_FMPSMBUS_MasterRxCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_FMPSMBUS_MasterRxCpltCallback()
      (+) Abort a master/host FMPSMBUS process communication with Interrupt using HAL_FMPSMBUS_Master_Abort_IT()
      (++) The associated previous transfer callback is called at the end of abort process
      (++) mean HAL_FMPSMBUS_MasterTxCpltCallback() in case of previous state was master transmit
      (++) mean HAL_FMPSMBUS_MasterRxCpltCallback() in case of previous state was master receive
      (+) Enable/disable the Address listen mode in slave/device or host/slave FMPSMBUS mode
           using HAL_FMPSMBUS_EnableListen_IT() HAL_FMPSMBUS_DisableListen_IT()
      (++) When address slave/device FMPSMBUS match, HAL_FMPSMBUS_AddrCallback() is executed and user can
           add his own code to check the Address Match Code and the transmission direction request by master/host (Write/Read).
      (++) At Listen mode end HAL_FMPSMBUS_ListenCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_FMPSMBUS_ListenCpltCallback()
      (+) Transmit in slave/device FMPSMBUS mode an amount of data in non-blocking mode using HAL_FMPSMBUS_Slave_Transmit_IT()
      (++) At transmission end of transfer HAL_FMPSMBUS_SlaveTxCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_FMPSMBUS_SlaveTxCpltCallback()
      (+) Receive in slave/device FMPSMBUS mode an amount of data in non-blocking mode using HAL_FMPSMBUS_Slave_Receive_IT()
      (++) At reception end of transfer HAL_FMPSMBUS_SlaveRxCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_FMPSMBUS_SlaveRxCpltCallback()
      (+) Enable/Disable the FMPSMBUS alert mode using HAL_FMPSMBUS_EnableAlert_IT() HAL_FMPSMBUS_DisableAlert_IT()
      (++) When FMPSMBUS Alert is generated HAL_FMPSMBUS_ErrorCallback() is executed and user can
           add his own code by customization of function pointer HAL_FMPSMBUS_ErrorCallback()
           to check the Alert Error Code using function HAL_FMPSMBUS_GetError()
      (+) Get HAL state machine or error values using HAL_FMPSMBUS_GetState() or HAL_FMPSMBUS_GetError()
      (+) In case of transfer Error, HAL_FMPSMBUS_ErrorCallback() function is executed and user can
           add his own code by customization of function pointer HAL_FMPSMBUS_ErrorCallback()
           to check the Error Code using function HAL_FMPSMBUS_GetError()

     *** FMPSMBUS HAL driver macros list ***
     ==================================
     [..]
       Below the list of most used macros in FMPSMBUS HAL driver.

      (+) __HAL_FMPSMBUS_ENABLE:      Enable the FMPSMBUS peripheral
      (+) __HAL_FMPSMBUS_DISABLE:     Disable the FMPSMBUS peripheral
      (+) __HAL_FMPSMBUS_GET_FLAG:    Check whether the specified FMPSMBUS flag is set or not
      (+) __HAL_FMPSMBUS_CLEAR_FLAG:  Clear the specified FMPSMBUS pending flag
      (+) __HAL_FMPSMBUS_ENABLE_IT:   Enable the specified FMPSMBUS interrupt
      (+) __HAL_FMPSMBUS_DISABLE_IT:  Disable the specified FMPSMBUS interrupt

     *** Callback registration ***
     =============================================
    [..]
     The compilation flag USE_HAL_FMPSMBUS_REGISTER_CALLBACKS when set to 1
     allows the user to configure dynamically the driver callbacks.
     Use Functions HAL_FMPSMBUS_RegisterCallback() or HAL_FMPSMBUS_RegisterAddrCallback()
     to register an interrupt callback.
    [..]
     Function HAL_FMPSMBUS_RegisterCallback() allows to register following callbacks:
       (+) MasterTxCpltCallback : callback for Master transmission end of transfer.
       (+) MasterRxCpltCallback : callback for Master reception end of transfer.
       (+) SlaveTxCpltCallback  : callback for Slave transmission end of transfer.
       (+) SlaveRxCpltCallback  : callback for Slave reception end of transfer.
       (+) ListenCpltCallback   : callback for end of listen mode.
       (+) ErrorCallback        : callback for error detection.
       (+) MspInitCallback      : callback for Msp Init.
       (+) MspDeInitCallback    : callback for Msp DeInit.
     This function takes as parameters the HAL peripheral handle, the Callback ID
     and a pointer to the user callback function.
    [..]
     For specific callback AddrCallback use dedicated register callbacks : HAL_FMPSMBUS_RegisterAddrCallback.
    [..]
     Use function HAL_FMPSMBUS_UnRegisterCallback to reset a callback to the default
     weak function.
     HAL_FMPSMBUS_UnRegisterCallback takes as parameters the HAL peripheral handle,
     and the Callback ID.
     This function allows to reset following callbacks:
       (+) MasterTxCpltCallback : callback for Master transmission end of transfer.
       (+) MasterRxCpltCallback : callback for Master reception end of transfer.
       (+) SlaveTxCpltCallback  : callback for Slave transmission end of transfer.
       (+) SlaveRxCpltCallback  : callback for Slave reception end of transfer.
       (+) ListenCpltCallback   : callback for end of listen mode.
       (+) ErrorCallback        : callback for error detection.
       (+) MspInitCallback      : callback for Msp Init.
       (+) MspDeInitCallback    : callback for Msp DeInit.
    [..]
     For callback AddrCallback use dedicated register callbacks : HAL_FMPSMBUS_UnRegisterAddrCallback.
    [..]
     By default, after the HAL_FMPSMBUS_Init() and when the state is HAL_FMPI2C_STATE_RESET
     all callbacks are set to the corresponding weak functions:
     examples HAL_FMPSMBUS_MasterTxCpltCallback(), HAL_FMPSMBUS_MasterRxCpltCallback().
     Exception done for MspInit and MspDeInit functions that are
     reset to the legacy weak functions in the HAL_FMPSMBUS_Init()/ HAL_FMPSMBUS_DeInit() only when
     these callbacks are null (not registered beforehand).
     If MspInit or MspDeInit are not null, the HAL_FMPSMBUS_Init()/ HAL_FMPSMBUS_DeInit()
     keep and use the user MspInit/MspDeInit callbacks (registered beforehand) whatever the state.
    [..]
     Callbacks can be registered/unregistered in HAL_FMPI2C_STATE_READY state only.
     Exception done MspInit/MspDeInit functions that can be registered/unregistered
     in HAL_FMPI2C_STATE_READY or HAL_FMPI2C_STATE_RESET state,
     thus registered (user) MspInit/DeInit callbacks can be used during the Init/DeInit.
     Then, the user first registers the MspInit/MspDeInit user callbacks
     using HAL_FMPSMBUS_RegisterCallback() before calling HAL_FMPSMBUS_DeInit()
     or HAL_FMPSMBUS_Init() function.
    [..]
     When the compilation flag USE_HAL_FMPSMBUS_REGISTER_CALLBACKS is set to 0 or
     not defined, the callback registration feature is not available and all callbacks
     are set to the corresponding weak functions.

     [..]
       (@) You can refer to the FMPSMBUS HAL driver header file for more useful macros

  @endverbatim
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

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup FMPSMBUS FMPSMBUS
  * @brief FMPSMBUS HAL module driver
  * @{
  */

#ifdef HAL_FMPSMBUS_MODULE_ENABLED

#if defined(FMPI2C_CR1_PE)
/* Private typedef -----------------------------------------------------------*/
/* Private constants ---------------------------------------------------------*/
/** @defgroup FMPSMBUS_Private_Define FMPSMBUS Private Constants
  * @{
  */
#define TIMING_CLEAR_MASK   (0xF0FFFFFFUL)     /*!< FMPSMBUS TIMING clear register Mask */
#define HAL_TIMEOUT_ADDR    (10000U)           /*!< 10 s  */
#define HAL_TIMEOUT_BUSY    (25U)              /*!< 25 ms */
#define HAL_TIMEOUT_DIR     (25U)              /*!< 25 ms */
#define HAL_TIMEOUT_RXNE    (25U)              /*!< 25 ms */
#define HAL_TIMEOUT_STOPF   (25U)              /*!< 25 ms */
#define HAL_TIMEOUT_TC      (25U)              /*!< 25 ms */
#define HAL_TIMEOUT_TCR     (25U)              /*!< 25 ms */
#define HAL_TIMEOUT_TXIS    (25U)              /*!< 25 ms */
#define MAX_NBYTE_SIZE      255U
/**
  * @}
  */

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/** @addtogroup FMPSMBUS_Private_Functions FMPSMBUS Private Functions
  * @{
  */
static HAL_StatusTypeDef FMPSMBUS_WaitOnFlagUntilTimeout(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t Flag,
                                                         FlagStatus Status, uint32_t Timeout);

static void FMPSMBUS_Enable_IRQ(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t InterruptRequest);
static void FMPSMBUS_Disable_IRQ(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t InterruptRequest);
static HAL_StatusTypeDef FMPSMBUS_Master_ISR(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t StatusFlags);
static HAL_StatusTypeDef FMPSMBUS_Slave_ISR(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t StatusFlags);

static void FMPSMBUS_ConvertOtherXferOptions(FMPSMBUS_HandleTypeDef *hfmpsmbus);

static void FMPSMBUS_ITErrorHandler(FMPSMBUS_HandleTypeDef *hfmpsmbus);

static void FMPSMBUS_TransferConfig(FMPSMBUS_HandleTypeDef *hfmpsmbus,  uint16_t DevAddress, uint8_t Size,
                                    uint32_t Mode, uint32_t Request);
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/

/** @defgroup FMPSMBUS_Exported_Functions FMPSMBUS Exported Functions
  * @{
  */

/** @defgroup FMPSMBUS_Exported_Functions_Group1 Initialization and de-initialization functions
  *  @brief    Initialization and Configuration functions
  *
@verbatim
 ===============================================================================
              ##### Initialization and de-initialization functions #####
 ===============================================================================
    [..]  This subsection provides a set of functions allowing to initialize and
          deinitialize the FMPSMBUSx peripheral:

      (+) User must Implement HAL_FMPSMBUS_MspInit() function in which he configures
          all related peripherals resources (CLOCK, GPIO, IT and NVIC ).

      (+) Call the function HAL_FMPSMBUS_Init() to configure the selected device with
          the selected configuration:
        (++) Clock Timing
        (++) Bus Timeout
        (++) Analog Filer mode
        (++) Own Address 1
        (++) Addressing mode (Master, Slave)
        (++) Dual Addressing mode
        (++) Own Address 2
        (++) Own Address 2 Mask
        (++) General call mode
        (++) Nostretch mode
        (++) Packet Error Check mode
        (++) Peripheral mode


      (+) Call the function HAL_FMPSMBUS_DeInit() to restore the default configuration
          of the selected FMPSMBUSx peripheral.

      (+) Enable/Disable Analog/Digital filters with HAL_FMPSMBUS_ConfigAnalogFilter() and
          HAL_FMPSMBUS_ConfigDigitalFilter().

@endverbatim
  * @{
  */

/**
  * @brief  Initialize the FMPSMBUS according to the specified parameters
  *         in the FMPSMBUS_InitTypeDef and initialize the associated handle.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_Init(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Check the FMPSMBUS handle allocation */
  if (hfmpsmbus == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_FMPSMBUS_ALL_INSTANCE(hfmpsmbus->Instance));
  assert_param(IS_FMPSMBUS_ANALOG_FILTER(hfmpsmbus->Init.AnalogFilter));
  assert_param(IS_FMPSMBUS_OWN_ADDRESS1(hfmpsmbus->Init.OwnAddress1));
  assert_param(IS_FMPSMBUS_ADDRESSING_MODE(hfmpsmbus->Init.AddressingMode));
  assert_param(IS_FMPSMBUS_DUAL_ADDRESS(hfmpsmbus->Init.DualAddressMode));
  assert_param(IS_FMPSMBUS_OWN_ADDRESS2(hfmpsmbus->Init.OwnAddress2));
  assert_param(IS_FMPSMBUS_OWN_ADDRESS2_MASK(hfmpsmbus->Init.OwnAddress2Masks));
  assert_param(IS_FMPSMBUS_GENERAL_CALL(hfmpsmbus->Init.GeneralCallMode));
  assert_param(IS_FMPSMBUS_NO_STRETCH(hfmpsmbus->Init.NoStretchMode));
  assert_param(IS_FMPSMBUS_PEC(hfmpsmbus->Init.PacketErrorCheckMode));
  assert_param(IS_FMPSMBUS_PERIPHERAL_MODE(hfmpsmbus->Init.PeripheralMode));

  if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hfmpsmbus->Lock = HAL_UNLOCKED;

#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
    hfmpsmbus->MasterTxCpltCallback = HAL_FMPSMBUS_MasterTxCpltCallback; /* Legacy weak MasterTxCpltCallback */
    hfmpsmbus->MasterRxCpltCallback = HAL_FMPSMBUS_MasterRxCpltCallback; /* Legacy weak MasterRxCpltCallback */
    hfmpsmbus->SlaveTxCpltCallback  = HAL_FMPSMBUS_SlaveTxCpltCallback;  /* Legacy weak SlaveTxCpltCallback  */
    hfmpsmbus->SlaveRxCpltCallback  = HAL_FMPSMBUS_SlaveRxCpltCallback;  /* Legacy weak SlaveRxCpltCallback  */
    hfmpsmbus->ListenCpltCallback   = HAL_FMPSMBUS_ListenCpltCallback;   /* Legacy weak ListenCpltCallback   */
    hfmpsmbus->ErrorCallback        = HAL_FMPSMBUS_ErrorCallback;        /* Legacy weak ErrorCallback        */
    hfmpsmbus->AddrCallback         = HAL_FMPSMBUS_AddrCallback;         /* Legacy weak AddrCallback         */

    if (hfmpsmbus->MspInitCallback == NULL)
    {
      hfmpsmbus->MspInitCallback = HAL_FMPSMBUS_MspInit; /* Legacy weak MspInit  */
    }

    /* Init the low level hardware : GPIO, CLOCK, CORTEX...etc */
    hfmpsmbus->MspInitCallback(hfmpsmbus);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    HAL_FMPSMBUS_MspInit(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
  }

  hfmpsmbus->State = HAL_FMPSMBUS_STATE_BUSY;

  /* Disable the selected FMPSMBUS peripheral */
  __HAL_FMPSMBUS_DISABLE(hfmpsmbus);

  /*---------------------------- FMPSMBUSx TIMINGR Configuration ------------------------*/
  /* Configure FMPSMBUSx: Frequency range */
  hfmpsmbus->Instance->TIMINGR = hfmpsmbus->Init.Timing & TIMING_CLEAR_MASK;

  /*---------------------------- FMPSMBUSx TIMEOUTR Configuration ------------------------*/
  /* Configure FMPSMBUSx: Bus Timeout  */
  hfmpsmbus->Instance->TIMEOUTR &= ~FMPI2C_TIMEOUTR_TIMOUTEN;
  hfmpsmbus->Instance->TIMEOUTR &= ~FMPI2C_TIMEOUTR_TEXTEN;
  hfmpsmbus->Instance->TIMEOUTR = hfmpsmbus->Init.SMBusTimeout;

  /*---------------------------- FMPSMBUSx OAR1 Configuration -----------------------*/
  /* Configure FMPSMBUSx: Own Address1 and ack own address1 mode */
  hfmpsmbus->Instance->OAR1 &= ~FMPI2C_OAR1_OA1EN;

  if (hfmpsmbus->Init.OwnAddress1 != 0UL)
  {
    if (hfmpsmbus->Init.AddressingMode == FMPSMBUS_ADDRESSINGMODE_7BIT)
    {
      hfmpsmbus->Instance->OAR1 = (FMPI2C_OAR1_OA1EN | hfmpsmbus->Init.OwnAddress1);
    }
    else /* FMPSMBUS_ADDRESSINGMODE_10BIT */
    {
      hfmpsmbus->Instance->OAR1 = (FMPI2C_OAR1_OA1EN | FMPI2C_OAR1_OA1MODE | hfmpsmbus->Init.OwnAddress1);
    }
  }

  /*---------------------------- FMPSMBUSx CR2 Configuration ------------------------*/
  /* Configure FMPSMBUSx: Addressing Master mode */
  if (hfmpsmbus->Init.AddressingMode == FMPSMBUS_ADDRESSINGMODE_10BIT)
  {
    hfmpsmbus->Instance->CR2 = (FMPI2C_CR2_ADD10);
  }
  /* Enable the AUTOEND by default, and enable NACK (should be disable only during Slave process) */
  /* AUTOEND and NACK bit will be manage during Transfer process */
  hfmpsmbus->Instance->CR2 |= (FMPI2C_CR2_AUTOEND | FMPI2C_CR2_NACK);

  /*---------------------------- FMPSMBUSx OAR2 Configuration -----------------------*/
  /* Configure FMPSMBUSx: Dual mode and Own Address2 */
  hfmpsmbus->Instance->OAR2 = (hfmpsmbus->Init.DualAddressMode | hfmpsmbus->Init.OwnAddress2 | \
                            (hfmpsmbus->Init.OwnAddress2Masks << 8U));

  /*---------------------------- FMPSMBUSx CR1 Configuration ------------------------*/
  /* Configure FMPSMBUSx: Generalcall and NoStretch mode */
  hfmpsmbus->Instance->CR1 = (hfmpsmbus->Init.GeneralCallMode | hfmpsmbus->Init.NoStretchMode | \
                           hfmpsmbus->Init.PacketErrorCheckMode | hfmpsmbus->Init.PeripheralMode | \
                           hfmpsmbus->Init.AnalogFilter);

  /* Enable Slave Byte Control only in case of Packet Error Check is enabled
     and FMPSMBUS Peripheral is set in Slave mode */
  if ((hfmpsmbus->Init.PacketErrorCheckMode == FMPSMBUS_PEC_ENABLE) && \
      ((hfmpsmbus->Init.PeripheralMode == FMPSMBUS_PERIPHERAL_MODE_FMPSMBUS_SLAVE) || \
       (hfmpsmbus->Init.PeripheralMode == FMPSMBUS_PERIPHERAL_MODE_FMPSMBUS_SLAVE_ARP)))
  {
    hfmpsmbus->Instance->CR1 |= FMPI2C_CR1_SBC;
  }

  /* Enable the selected FMPSMBUS peripheral */
  __HAL_FMPSMBUS_ENABLE(hfmpsmbus);

  hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_NONE;
  hfmpsmbus->PreviousState = HAL_FMPSMBUS_STATE_READY;
  hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  DeInitialize the FMPSMBUS peripheral.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_DeInit(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Check the FMPSMBUS handle allocation */
  if (hfmpsmbus == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_FMPSMBUS_ALL_INSTANCE(hfmpsmbus->Instance));

  hfmpsmbus->State = HAL_FMPSMBUS_STATE_BUSY;

  /* Disable the FMPSMBUS Peripheral Clock */
  __HAL_FMPSMBUS_DISABLE(hfmpsmbus);

#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
  if (hfmpsmbus->MspDeInitCallback == NULL)
  {
    hfmpsmbus->MspDeInitCallback = HAL_FMPSMBUS_MspDeInit; /* Legacy weak MspDeInit  */
  }

  /* DeInit the low level hardware: GPIO, CLOCK, NVIC */
  hfmpsmbus->MspDeInitCallback(hfmpsmbus);
#else
  /* DeInit the low level hardware: GPIO, CLOCK, NVIC */
  HAL_FMPSMBUS_MspDeInit(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */

  hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_NONE;
  hfmpsmbus->PreviousState =  HAL_FMPSMBUS_STATE_RESET;
  hfmpsmbus->State = HAL_FMPSMBUS_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(hfmpsmbus);

  return HAL_OK;
}

/**
  * @brief Initialize the FMPSMBUS MSP.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
__weak void HAL_FMPSMBUS_MspInit(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hfmpsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_FMPSMBUS_MspInit could be implemented in the user file
   */
}

/**
  * @brief DeInitialize the FMPSMBUS MSP.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
__weak void HAL_FMPSMBUS_MspDeInit(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hfmpsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_FMPSMBUS_MspDeInit could be implemented in the user file
   */
}

/**
  * @brief  Configure Analog noise filter.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  AnalogFilter This parameter can be one of the following values:
  *         @arg @ref FMPSMBUS_ANALOGFILTER_ENABLE
  *         @arg @ref FMPSMBUS_ANALOGFILTER_DISABLE
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_ConfigAnalogFilter(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t AnalogFilter)
{
  /* Check the parameters */
  assert_param(IS_FMPSMBUS_ALL_INSTANCE(hfmpsmbus->Instance));
  assert_param(IS_FMPSMBUS_ANALOG_FILTER(AnalogFilter));

  if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hfmpsmbus);

    hfmpsmbus->State = HAL_FMPSMBUS_STATE_BUSY;

    /* Disable the selected FMPSMBUS peripheral */
    __HAL_FMPSMBUS_DISABLE(hfmpsmbus);

    /* Reset ANOFF bit */
    hfmpsmbus->Instance->CR1 &= ~(FMPI2C_CR1_ANFOFF);

    /* Set analog filter bit*/
    hfmpsmbus->Instance->CR1 |= AnalogFilter;

    __HAL_FMPSMBUS_ENABLE(hfmpsmbus);

    hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Configure Digital noise filter.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  DigitalFilter Coefficient of digital noise filter between Min_Data=0x00 and Max_Data=0x0F.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_ConfigDigitalFilter(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t DigitalFilter)
{
  uint32_t tmpreg;

  /* Check the parameters */
  assert_param(IS_FMPSMBUS_ALL_INSTANCE(hfmpsmbus->Instance));
  assert_param(IS_FMPSMBUS_DIGITAL_FILTER(DigitalFilter));

  if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hfmpsmbus);

    hfmpsmbus->State = HAL_FMPSMBUS_STATE_BUSY;

    /* Disable the selected FMPSMBUS peripheral */
    __HAL_FMPSMBUS_DISABLE(hfmpsmbus);

    /* Get the old register value */
    tmpreg = hfmpsmbus->Instance->CR1;

    /* Reset FMPI2C DNF bits [11:8] */
    tmpreg &= ~(FMPI2C_CR1_DNF);

    /* Set FMPI2Cx DNF coefficient */
    tmpreg |= DigitalFilter << FMPI2C_CR1_DNF_Pos;

    /* Store the new register value */
    hfmpsmbus->Instance->CR1 = tmpreg;

    __HAL_FMPSMBUS_ENABLE(hfmpsmbus);

    hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User FMPSMBUS Callback
  *         To be used instead of the weak predefined callback
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  CallbackID ID of the callback to be registered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_FMPSMBUS_MASTER_TX_COMPLETE_CB_ID Master Tx Transfer completed callback ID
  *          @arg @ref HAL_FMPSMBUS_MASTER_RX_COMPLETE_CB_ID Master Rx Transfer completed callback ID
  *          @arg @ref HAL_FMPSMBUS_SLAVE_TX_COMPLETE_CB_ID Slave Tx Transfer completed callback ID
  *          @arg @ref HAL_FMPSMBUS_SLAVE_RX_COMPLETE_CB_ID Slave Rx Transfer completed callback ID
  *          @arg @ref HAL_FMPSMBUS_LISTEN_COMPLETE_CB_ID Listen Complete callback ID
  *          @arg @ref HAL_FMPSMBUS_ERROR_CB_ID Error callback ID
  *          @arg @ref HAL_FMPSMBUS_MSPINIT_CB_ID MspInit callback ID
  *          @arg @ref HAL_FMPSMBUS_MSPDEINIT_CB_ID MspDeInit callback ID
  * @param  pCallback pointer to the Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_RegisterCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus,
                                                HAL_FMPSMBUS_CallbackIDTypeDef CallbackID,
                                                pFMPSMBUS_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

    return HAL_ERROR;
  }

  /* Process locked */
  __HAL_LOCK(hfmpsmbus);

  if (HAL_FMPSMBUS_STATE_READY == hfmpsmbus->State)
  {
    switch (CallbackID)
    {
      case HAL_FMPSMBUS_MASTER_TX_COMPLETE_CB_ID :
        hfmpsmbus->MasterTxCpltCallback = pCallback;
        break;

      case HAL_FMPSMBUS_MASTER_RX_COMPLETE_CB_ID :
        hfmpsmbus->MasterRxCpltCallback = pCallback;
        break;

      case HAL_FMPSMBUS_SLAVE_TX_COMPLETE_CB_ID :
        hfmpsmbus->SlaveTxCpltCallback = pCallback;
        break;

      case HAL_FMPSMBUS_SLAVE_RX_COMPLETE_CB_ID :
        hfmpsmbus->SlaveRxCpltCallback = pCallback;
        break;

      case HAL_FMPSMBUS_LISTEN_COMPLETE_CB_ID :
        hfmpsmbus->ListenCpltCallback = pCallback;
        break;

      case HAL_FMPSMBUS_ERROR_CB_ID :
        hfmpsmbus->ErrorCallback = pCallback;
        break;

      case HAL_FMPSMBUS_MSPINIT_CB_ID :
        hfmpsmbus->MspInitCallback = pCallback;
        break;

      case HAL_FMPSMBUS_MSPDEINIT_CB_ID :
        hfmpsmbus->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_FMPSMBUS_STATE_RESET == hfmpsmbus->State)
  {
    switch (CallbackID)
    {
      case HAL_FMPSMBUS_MSPINIT_CB_ID :
        hfmpsmbus->MspInitCallback = pCallback;
        break;

      case HAL_FMPSMBUS_MSPDEINIT_CB_ID :
        hfmpsmbus->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hfmpsmbus);
  return status;
}

/**
  * @brief  Unregister an FMPSMBUS Callback
  *         FMPSMBUS callback is redirected to the weak predefined callback
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  CallbackID ID of the callback to be unregistered
  *         This parameter can be one of the following values:
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_FMPSMBUS_MASTER_TX_COMPLETE_CB_ID Master Tx Transfer completed callback ID
  *          @arg @ref HAL_FMPSMBUS_MASTER_RX_COMPLETE_CB_ID Master Rx Transfer completed callback ID
  *          @arg @ref HAL_FMPSMBUS_SLAVE_TX_COMPLETE_CB_ID Slave Tx Transfer completed callback ID
  *          @arg @ref HAL_FMPSMBUS_SLAVE_RX_COMPLETE_CB_ID Slave Rx Transfer completed callback ID
  *          @arg @ref HAL_FMPSMBUS_LISTEN_COMPLETE_CB_ID Listen Complete callback ID
  *          @arg @ref HAL_FMPSMBUS_ERROR_CB_ID Error callback ID
  *          @arg @ref HAL_FMPSMBUS_MSPINIT_CB_ID MspInit callback ID
  *          @arg @ref HAL_FMPSMBUS_MSPDEINIT_CB_ID MspDeInit callback ID
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_UnRegisterCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus,
                                                  HAL_FMPSMBUS_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hfmpsmbus);

  if (HAL_FMPSMBUS_STATE_READY == hfmpsmbus->State)
  {
    switch (CallbackID)
    {
      case HAL_FMPSMBUS_MASTER_TX_COMPLETE_CB_ID :
        hfmpsmbus->MasterTxCpltCallback = HAL_FMPSMBUS_MasterTxCpltCallback; /* Legacy weak MasterTxCpltCallback */
        break;

      case HAL_FMPSMBUS_MASTER_RX_COMPLETE_CB_ID :
        hfmpsmbus->MasterRxCpltCallback = HAL_FMPSMBUS_MasterRxCpltCallback; /* Legacy weak MasterRxCpltCallback */
        break;

      case HAL_FMPSMBUS_SLAVE_TX_COMPLETE_CB_ID :
        hfmpsmbus->SlaveTxCpltCallback = HAL_FMPSMBUS_SlaveTxCpltCallback;   /* Legacy weak SlaveTxCpltCallback  */
        break;

      case HAL_FMPSMBUS_SLAVE_RX_COMPLETE_CB_ID :
        hfmpsmbus->SlaveRxCpltCallback = HAL_FMPSMBUS_SlaveRxCpltCallback;   /* Legacy weak SlaveRxCpltCallback  */
        break;

      case HAL_FMPSMBUS_LISTEN_COMPLETE_CB_ID :
        hfmpsmbus->ListenCpltCallback = HAL_FMPSMBUS_ListenCpltCallback;     /* Legacy weak ListenCpltCallback   */
        break;

      case HAL_FMPSMBUS_ERROR_CB_ID :
        hfmpsmbus->ErrorCallback = HAL_FMPSMBUS_ErrorCallback;               /* Legacy weak ErrorCallback        */
        break;

      case HAL_FMPSMBUS_MSPINIT_CB_ID :
        hfmpsmbus->MspInitCallback = HAL_FMPSMBUS_MspInit;                   /* Legacy weak MspInit              */
        break;

      case HAL_FMPSMBUS_MSPDEINIT_CB_ID :
        hfmpsmbus->MspDeInitCallback = HAL_FMPSMBUS_MspDeInit;               /* Legacy weak MspDeInit            */
        break;

      default :
        /* Update the error code */
        hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_FMPSMBUS_STATE_RESET == hfmpsmbus->State)
  {
    switch (CallbackID)
    {
      case HAL_FMPSMBUS_MSPINIT_CB_ID :
        hfmpsmbus->MspInitCallback = HAL_FMPSMBUS_MspInit;                   /* Legacy weak MspInit              */
        break;

      case HAL_FMPSMBUS_MSPDEINIT_CB_ID :
        hfmpsmbus->MspDeInitCallback = HAL_FMPSMBUS_MspDeInit;               /* Legacy weak MspDeInit            */
        break;

      default :
        /* Update the error code */
        hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hfmpsmbus);
  return status;
}

/**
  * @brief  Register the Slave Address Match FMPSMBUS Callback
  *         To be used instead of the weak HAL_FMPSMBUS_AddrCallback() predefined callback
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  pCallback pointer to the Address Match Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_RegisterAddrCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus,
                                                    pFMPSMBUS_AddrCallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hfmpsmbus);

  if (HAL_FMPSMBUS_STATE_READY == hfmpsmbus->State)
  {
    hfmpsmbus->AddrCallback = pCallback;
  }
  else
  {
    /* Update the error code */
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hfmpsmbus);
  return status;
}

/**
  * @brief  UnRegister the Slave Address Match FMPSMBUS Callback
  *         Info Ready FMPSMBUS Callback is redirected to the weak HAL_FMPSMBUS_AddrCallback() predefined callback
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_UnRegisterAddrCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hfmpsmbus);

  if (HAL_FMPSMBUS_STATE_READY == hfmpsmbus->State)
  {
    hfmpsmbus->AddrCallback = HAL_FMPSMBUS_AddrCallback; /* Legacy weak AddrCallback  */
  }
  else
  {
    /* Update the error code */
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hfmpsmbus);
  return status;
}

#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup FMPSMBUS_Exported_Functions_Group2 Input and Output operation functions
  *  @brief   Data transfers functions
  *
@verbatim
 ===============================================================================
                      ##### IO operation functions #####
 ===============================================================================
    [..]
    This subsection provides a set of functions allowing to manage the FMPSMBUS data
    transfers.

    (#) Blocking mode function to check if device is ready for usage is :
        (++) HAL_FMPSMBUS_IsDeviceReady()

    (#) There is only one mode of transfer:
       (++) Non-Blocking mode : The communication is performed using Interrupts.
            These functions return the status of the transfer startup.
            The end of the data processing will be indicated through the
            dedicated FMPSMBUS IRQ when using Interrupt mode.

    (#) Non-Blocking mode functions with Interrupt are :
        (++) HAL_FMPSMBUS_Master_Transmit_IT()
        (++) HAL_FMPSMBUS_Master_Receive_IT()
        (++) HAL_FMPSMBUS_Slave_Transmit_IT()
        (++) HAL_FMPSMBUS_Slave_Receive_IT()
        (++) HAL_FMPSMBUS_EnableListen_IT() or alias HAL_FMPSMBUS_EnableListen_IT()
        (++) HAL_FMPSMBUS_DisableListen_IT()
        (++) HAL_FMPSMBUS_EnableAlert_IT()
        (++) HAL_FMPSMBUS_DisableAlert_IT()

    (#) A set of Transfer Complete Callbacks are provided in non-Blocking mode:
        (++) HAL_FMPSMBUS_MasterTxCpltCallback()
        (++) HAL_FMPSMBUS_MasterRxCpltCallback()
        (++) HAL_FMPSMBUS_SlaveTxCpltCallback()
        (++) HAL_FMPSMBUS_SlaveRxCpltCallback()
        (++) HAL_FMPSMBUS_AddrCallback()
        (++) HAL_FMPSMBUS_ListenCpltCallback()
        (++) HAL_FMPSMBUS_ErrorCallback()

@endverbatim
  * @{
  */

/**
  * @brief  Transmit in master/host FMPSMBUS mode an amount of data in non-blocking mode with Interrupt.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  DevAddress Target device address: The device 7 bits address value
  *         in datasheet must be shifted to the left before calling the interface
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @param  XferOptions Options of Transfer, value of @ref FMPSMBUS_XferOptions_definition
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_Master_Transmit_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint16_t DevAddress,
                                                  uint8_t *pData, uint16_t Size, uint32_t XferOptions)
{
  uint32_t tmp;

  /* Check the parameters */
  assert_param(IS_FMPSMBUS_TRANSFER_OPTIONS_REQUEST(XferOptions));

  if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hfmpsmbus);

    hfmpsmbus->State = HAL_FMPSMBUS_STATE_MASTER_BUSY_TX;
    hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_NONE;
    /* Prepare transfer parameters */
    hfmpsmbus->pBuffPtr = pData;
    hfmpsmbus->XferCount = Size;
    hfmpsmbus->XferOptions = XferOptions;

    /* In case of Quick command, remove autoend mode */
    /* Manage the stop generation by software */
    if (hfmpsmbus->pBuffPtr == NULL)
    {
      hfmpsmbus->XferOptions &= ~FMPSMBUS_AUTOEND_MODE;
    }

    if (Size > MAX_NBYTE_SIZE)
    {
      hfmpsmbus->XferSize = MAX_NBYTE_SIZE;
    }
    else
    {
      hfmpsmbus->XferSize = Size;
    }

    /* Send Slave Address */
    /* Set NBYTES to write and reload if size > MAX_NBYTE_SIZE and generate RESTART */
    if ((hfmpsmbus->XferSize < hfmpsmbus->XferCount) && (hfmpsmbus->XferSize == MAX_NBYTE_SIZE))
    {
      FMPSMBUS_TransferConfig(hfmpsmbus, DevAddress, (uint8_t)hfmpsmbus->XferSize,
                           FMPSMBUS_RELOAD_MODE | (hfmpsmbus->XferOptions & FMPSMBUS_SENDPEC_MODE),
                           FMPSMBUS_GENERATE_START_WRITE);
    }
    else
    {
      /* If transfer direction not change, do not generate Restart Condition */
      /* Mean Previous state is same as current state */

      /* Store current volatile XferOptions, misra rule */
      tmp = hfmpsmbus->XferOptions;

      if ((hfmpsmbus->PreviousState == HAL_FMPSMBUS_STATE_MASTER_BUSY_TX) && \
          (IS_FMPSMBUS_TRANSFER_OTHER_OPTIONS_REQUEST(tmp) == 0))
      {
        FMPSMBUS_TransferConfig(hfmpsmbus, DevAddress, (uint8_t)hfmpsmbus->XferSize, hfmpsmbus->XferOptions,
                                FMPSMBUS_NO_STARTSTOP);
      }
      /* Else transfer direction change, so generate Restart with new transfer direction */
      else
      {
        /* Convert OTHER_xxx XferOptions if any */
        FMPSMBUS_ConvertOtherXferOptions(hfmpsmbus);

        /* Handle Transfer */
        FMPSMBUS_TransferConfig(hfmpsmbus, DevAddress, (uint8_t)hfmpsmbus->XferSize,
                             hfmpsmbus->XferOptions,
                             FMPSMBUS_GENERATE_START_WRITE);
      }

      /* If PEC mode is enable, size to transmit manage by SW part should be Size-1 byte, corresponding to PEC byte */
      /* PEC byte is automatically sent by HW block, no need to manage it in Transmit process */
      if (FMPSMBUS_GET_PEC_MODE(hfmpsmbus) != 0UL)
      {
        hfmpsmbus->XferSize--;
        hfmpsmbus->XferCount--;
      }
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    /* Note : The FMPSMBUS interrupts must be enabled after unlocking current process
              to avoid the risk of FMPSMBUS interrupt handle execution before current
              process unlock */
    FMPSMBUS_Enable_IRQ(hfmpsmbus, FMPSMBUS_IT_TX);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Receive in master/host FMPSMBUS mode an amount of data in non-blocking mode with Interrupt.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  DevAddress Target device address: The device 7 bits address value
  *         in datasheet must be shifted to the left before calling the interface
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @param  XferOptions Options of Transfer, value of @ref FMPSMBUS_XferOptions_definition
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_Master_Receive_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint16_t DevAddress, uint8_t *pData,
                                              uint16_t Size, uint32_t XferOptions)
{
  uint32_t tmp;

  /* Check the parameters */
  assert_param(IS_FMPSMBUS_TRANSFER_OPTIONS_REQUEST(XferOptions));

  if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hfmpsmbus);

    hfmpsmbus->State = HAL_FMPSMBUS_STATE_MASTER_BUSY_RX;
    hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_NONE;

    /* Prepare transfer parameters */
    hfmpsmbus->pBuffPtr = pData;
    hfmpsmbus->XferCount = Size;
    hfmpsmbus->XferOptions = XferOptions;

    /* In case of Quick command, remove autoend mode */
    /* Manage the stop generation by software */
    if (hfmpsmbus->pBuffPtr == NULL)
    {
      hfmpsmbus->XferOptions &= ~FMPSMBUS_AUTOEND_MODE;
    }

    if (Size > MAX_NBYTE_SIZE)
    {
      hfmpsmbus->XferSize = MAX_NBYTE_SIZE;
    }
    else
    {
      hfmpsmbus->XferSize = Size;
    }

    /* Send Slave Address */
    /* Set NBYTES to write and reload if size > MAX_NBYTE_SIZE and generate RESTART */
    if ((hfmpsmbus->XferSize < hfmpsmbus->XferCount) && (hfmpsmbus->XferSize == MAX_NBYTE_SIZE))
    {
      FMPSMBUS_TransferConfig(hfmpsmbus, DevAddress, (uint8_t)hfmpsmbus->XferSize,
                           FMPSMBUS_RELOAD_MODE  | (hfmpsmbus->XferOptions & FMPSMBUS_SENDPEC_MODE),
                           FMPSMBUS_GENERATE_START_READ);
    }
    else
    {
      /* If transfer direction not change, do not generate Restart Condition */
      /* Mean Previous state is same as current state */

      /* Store current volatile XferOptions, Misra rule */
      tmp = hfmpsmbus->XferOptions;

      if ((hfmpsmbus->PreviousState == HAL_FMPSMBUS_STATE_MASTER_BUSY_RX) && \
          (IS_FMPSMBUS_TRANSFER_OTHER_OPTIONS_REQUEST(tmp) == 0))
      {
        FMPSMBUS_TransferConfig(hfmpsmbus, DevAddress, (uint8_t)hfmpsmbus->XferSize, hfmpsmbus->XferOptions,
                                FMPSMBUS_NO_STARTSTOP);
      }
      /* Else transfer direction change, so generate Restart with new transfer direction */
      else
      {
        /* Convert OTHER_xxx XferOptions if any */
        FMPSMBUS_ConvertOtherXferOptions(hfmpsmbus);

        /* Handle Transfer */
        FMPSMBUS_TransferConfig(hfmpsmbus, DevAddress, (uint8_t)hfmpsmbus->XferSize,
                             hfmpsmbus->XferOptions,
                             FMPSMBUS_GENERATE_START_READ);
      }
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    /* Note : The FMPSMBUS interrupts must be enabled after unlocking current process
              to avoid the risk of FMPSMBUS interrupt handle execution before current
              process unlock */
    FMPSMBUS_Enable_IRQ(hfmpsmbus, FMPSMBUS_IT_RX);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Abort a master/host FMPSMBUS process communication with Interrupt.
  * @note   This abort can be called only if state is ready
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  DevAddress Target device address: The device 7 bits address value
  *         in datasheet must be shifted to the left before calling the interface
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_Master_Abort_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint16_t DevAddress)
{
  if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hfmpsmbus);

    /* Keep the same state as previous */
    /* to perform as well the call of the corresponding end of transfer callback */
    if (hfmpsmbus->PreviousState == HAL_FMPSMBUS_STATE_MASTER_BUSY_TX)
    {
      hfmpsmbus->State = HAL_FMPSMBUS_STATE_MASTER_BUSY_TX;
    }
    else if (hfmpsmbus->PreviousState == HAL_FMPSMBUS_STATE_MASTER_BUSY_RX)
    {
      hfmpsmbus->State = HAL_FMPSMBUS_STATE_MASTER_BUSY_RX;
    }
    else
    {
      /* Wrong usage of abort function */
      /* This function should be used only in case of abort monitored by master device */
      return HAL_ERROR;
    }
    hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_NONE;

    /* Set NBYTES to 1 to generate a dummy read on FMPSMBUS peripheral */
    /* Set AUTOEND mode, this will generate a NACK then STOP condition to abort the current transfer */
    FMPSMBUS_TransferConfig(hfmpsmbus, DevAddress, 1, FMPSMBUS_AUTOEND_MODE, FMPSMBUS_NO_STARTSTOP);

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    /* Note : The FMPSMBUS interrupts must be enabled after unlocking current process
              to avoid the risk of FMPSMBUS interrupt handle execution before current
              process unlock */
    if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_MASTER_BUSY_TX)
    {
      FMPSMBUS_Enable_IRQ(hfmpsmbus, FMPSMBUS_IT_TX);
    }
    else if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_MASTER_BUSY_RX)
    {
      FMPSMBUS_Enable_IRQ(hfmpsmbus, FMPSMBUS_IT_RX);
    }
    else
    {
      /* Nothing to do */
    }

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Transmit in slave/device FMPSMBUS mode an amount of data in non-blocking mode with Interrupt.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @param  XferOptions Options of Transfer, value of @ref FMPSMBUS_XferOptions_definition
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_Slave_Transmit_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint8_t *pData, uint16_t Size,
                                              uint32_t XferOptions)
{
  /* Check the parameters */
  assert_param(IS_FMPSMBUS_TRANSFER_OPTIONS_REQUEST(XferOptions));

  if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_LISTEN) == HAL_FMPSMBUS_STATE_LISTEN)
  {
    if ((pData == NULL) || (Size == 0UL))
    {
      hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_INVALID_PARAM;
      return HAL_ERROR;
    }

    /* Disable Interrupts, to prevent preemption during treatment in case of multicall */
    FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_ADDR | FMPSMBUS_IT_TX);

    /* Process Locked */
    __HAL_LOCK(hfmpsmbus);

    hfmpsmbus->State = (HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX | HAL_FMPSMBUS_STATE_LISTEN);
    hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_NONE;

    /* Set SBC bit to manage Acknowledge at each bit */
    hfmpsmbus->Instance->CR1 |= FMPI2C_CR1_SBC;

    /* Enable Address Acknowledge */
    hfmpsmbus->Instance->CR2 &= ~FMPI2C_CR2_NACK;

    /* Prepare transfer parameters */
    hfmpsmbus->pBuffPtr = pData;
    hfmpsmbus->XferCount = Size;
    hfmpsmbus->XferOptions = XferOptions;

    /* Convert OTHER_xxx XferOptions if any */
    FMPSMBUS_ConvertOtherXferOptions(hfmpsmbus);

    if (Size > MAX_NBYTE_SIZE)
    {
      hfmpsmbus->XferSize = MAX_NBYTE_SIZE;
    }
    else
    {
      hfmpsmbus->XferSize = Size;
    }

    /* Set NBYTES to write and reload if size > MAX_NBYTE_SIZE and generate RESTART */
    if ((hfmpsmbus->XferSize < hfmpsmbus->XferCount) && (hfmpsmbus->XferSize == MAX_NBYTE_SIZE))
    {
      FMPSMBUS_TransferConfig(hfmpsmbus, 0, (uint8_t)hfmpsmbus->XferSize,
                           FMPSMBUS_RELOAD_MODE | (hfmpsmbus->XferOptions & FMPSMBUS_SENDPEC_MODE),
                           FMPSMBUS_NO_STARTSTOP);
    }
    else
    {
      /* Set NBYTE to transmit */
      FMPSMBUS_TransferConfig(hfmpsmbus, 0, (uint8_t)hfmpsmbus->XferSize, hfmpsmbus->XferOptions,
                              FMPSMBUS_NO_STARTSTOP);

      /* If PEC mode is enable, size to transmit should be Size-1 byte, corresponding to PEC byte */
      /* PEC byte is automatically sent by HW block, no need to manage it in Transmit process */
      if (FMPSMBUS_GET_PEC_MODE(hfmpsmbus) != 0UL)
      {
        hfmpsmbus->XferSize--;
        hfmpsmbus->XferCount--;
      }
    }

    /* Clear ADDR flag after prepare the transfer parameters */
    /* This action will generate an acknowledge to the HOST */
    __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_ADDR);

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    /* Note : The FMPSMBUS interrupts must be enabled after unlocking current process
              to avoid the risk of FMPSMBUS interrupt handle execution before current
              process unlock */
    /* REnable ADDR interrupt */
    FMPSMBUS_Enable_IRQ(hfmpsmbus, FMPSMBUS_IT_TX | FMPSMBUS_IT_ADDR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Receive in slave/device FMPSMBUS mode an amount of data in non-blocking mode with Interrupt.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @param  XferOptions Options of Transfer, value of @ref FMPSMBUS_XferOptions_definition
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_Slave_Receive_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint8_t *pData, uint16_t Size,
                                             uint32_t XferOptions)
{
  /* Check the parameters */
  assert_param(IS_FMPSMBUS_TRANSFER_OPTIONS_REQUEST(XferOptions));

  if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_LISTEN) == HAL_FMPSMBUS_STATE_LISTEN)
  {
    if ((pData == NULL) || (Size == 0UL))
    {
      hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_INVALID_PARAM;
      return HAL_ERROR;
    }

    /* Disable Interrupts, to prevent preemption during treatment in case of multicall */
    FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_ADDR | FMPSMBUS_IT_RX);

    /* Process Locked */
    __HAL_LOCK(hfmpsmbus);

    hfmpsmbus->State = (HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX | HAL_FMPSMBUS_STATE_LISTEN);
    hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_NONE;

    /* Set SBC bit to manage Acknowledge at each bit */
    hfmpsmbus->Instance->CR1 |= FMPI2C_CR1_SBC;

    /* Enable Address Acknowledge */
    hfmpsmbus->Instance->CR2 &= ~FMPI2C_CR2_NACK;

    /* Prepare transfer parameters */
    hfmpsmbus->pBuffPtr = pData;
    hfmpsmbus->XferSize = Size;
    hfmpsmbus->XferCount = Size;
    hfmpsmbus->XferOptions = XferOptions;

    /* Convert OTHER_xxx XferOptions if any */
    FMPSMBUS_ConvertOtherXferOptions(hfmpsmbus);

    /* Set NBYTE to receive */
    /* If XferSize equal "1", or XferSize equal "2" with PEC requested (mean 1 data byte + 1 PEC byte */
    /* no need to set RELOAD bit mode, a ACK will be automatically generated in that case */
    /* else need to set RELOAD bit mode to generate an automatic ACK at each byte Received */
    /* This RELOAD bit will be reset for last BYTE to be receive in FMPSMBUS_Slave_ISR */
    if (((FMPSMBUS_GET_PEC_MODE(hfmpsmbus) != 0UL) && (hfmpsmbus->XferSize == 2U)) || (hfmpsmbus->XferSize == 1U))
    {
      FMPSMBUS_TransferConfig(hfmpsmbus, 0, (uint8_t)hfmpsmbus->XferSize, hfmpsmbus->XferOptions,
                              FMPSMBUS_NO_STARTSTOP);
    }
    else
    {
      FMPSMBUS_TransferConfig(hfmpsmbus, 0, 1, hfmpsmbus->XferOptions | FMPSMBUS_RELOAD_MODE, FMPSMBUS_NO_STARTSTOP);
    }

    /* Clear ADDR flag after prepare the transfer parameters */
    /* This action will generate an acknowledge to the HOST */
    __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_ADDR);

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    /* Note : The FMPSMBUS interrupts must be enabled after unlocking current process
              to avoid the risk of FMPSMBUS interrupt handle execution before current
              process unlock */
    /* REnable ADDR interrupt */
    FMPSMBUS_Enable_IRQ(hfmpsmbus, FMPSMBUS_IT_RX | FMPSMBUS_IT_ADDR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Enable the Address listen mode with Interrupt.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_EnableListen_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  hfmpsmbus->State = HAL_FMPSMBUS_STATE_LISTEN;

  /* Enable the Address Match interrupt */
  FMPSMBUS_Enable_IRQ(hfmpsmbus, FMPSMBUS_IT_ADDR);

  return HAL_OK;
}

/**
  * @brief  Disable the Address listen mode with Interrupt.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_DisableListen_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Disable Address listen mode only if a transfer is not ongoing */
  if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_LISTEN)
  {
    hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

    /* Disable the Address Match interrupt */
    FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_ADDR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Enable the FMPSMBUS alert mode with Interrupt.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUSx peripheral.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_EnableAlert_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Enable SMBus alert */
  hfmpsmbus->Instance->CR1 |= FMPI2C_CR1_ALERTEN;

  /* Clear ALERT flag */
  __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_ALERT);

  /* Enable Alert Interrupt */
  FMPSMBUS_Enable_IRQ(hfmpsmbus, FMPSMBUS_IT_ALERT);

  return HAL_OK;
}
/**
  * @brief  Disable the FMPSMBUS alert mode with Interrupt.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUSx peripheral.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_DisableAlert_IT(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Enable SMBus alert */
  hfmpsmbus->Instance->CR1 &= ~FMPI2C_CR1_ALERTEN;

  /* Disable Alert Interrupt */
  FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_ALERT);

  return HAL_OK;
}

/**
  * @brief  Check if target device is ready for communication.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  DevAddress Target device address: The device 7 bits address value
  *         in datasheet must be shifted to the left before calling the interface
  * @param  Trials Number of trials
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_FMPSMBUS_IsDeviceReady(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint16_t DevAddress, uint32_t Trials,
                                          uint32_t Timeout)
{
  uint32_t tickstart;

  __IO uint32_t FMPSMBUS_Trials = 0UL;

  FlagStatus tmp1;
  FlagStatus tmp2;

  if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_READY)
  {
    if (__HAL_FMPSMBUS_GET_FLAG(hfmpsmbus, FMPSMBUS_FLAG_BUSY) != RESET)
    {
      return HAL_BUSY;
    }

    /* Process Locked */
    __HAL_LOCK(hfmpsmbus);

    hfmpsmbus->State = HAL_FMPSMBUS_STATE_BUSY;
    hfmpsmbus->ErrorCode = HAL_FMPSMBUS_ERROR_NONE;

    do
    {
      /* Generate Start */
      hfmpsmbus->Instance->CR2 = FMPSMBUS_GENERATE_START(hfmpsmbus->Init.AddressingMode, DevAddress);

      /* No need to Check TC flag, with AUTOEND mode the stop is automatically generated */
      /* Wait until STOPF flag is set or a NACK flag is set*/
      tickstart = HAL_GetTick();

      tmp1 = __HAL_FMPSMBUS_GET_FLAG(hfmpsmbus, FMPSMBUS_FLAG_STOPF);
      tmp2 = __HAL_FMPSMBUS_GET_FLAG(hfmpsmbus, FMPSMBUS_FLAG_AF);

      while ((tmp1 == RESET) && (tmp2 == RESET))
      {
        if (Timeout != HAL_MAX_DELAY)
        {
          if (((HAL_GetTick() - tickstart) > Timeout) || (Timeout == 0UL))
          {
            /* Device is ready */
            hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

            /* Update FMPSMBUS error code */
            hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_HALTIMEOUT;

            /* Process Unlocked */
            __HAL_UNLOCK(hfmpsmbus);
            return HAL_ERROR;
          }
        }

        tmp1 = __HAL_FMPSMBUS_GET_FLAG(hfmpsmbus, FMPSMBUS_FLAG_STOPF);
        tmp2 = __HAL_FMPSMBUS_GET_FLAG(hfmpsmbus, FMPSMBUS_FLAG_AF);
      }

      /* Check if the NACKF flag has not been set */
      if (__HAL_FMPSMBUS_GET_FLAG(hfmpsmbus, FMPSMBUS_FLAG_AF) == RESET)
      {
        /* Wait until STOPF flag is reset */
        if (FMPSMBUS_WaitOnFlagUntilTimeout(hfmpsmbus, FMPSMBUS_FLAG_STOPF, RESET, Timeout) != HAL_OK)
        {
          return HAL_ERROR;
        }

        /* Clear STOP Flag */
        __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_STOPF);

        /* Device is ready */
        hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

        /* Process Unlocked */
        __HAL_UNLOCK(hfmpsmbus);

        return HAL_OK;
      }
      else
      {
        /* Wait until STOPF flag is reset */
        if (FMPSMBUS_WaitOnFlagUntilTimeout(hfmpsmbus, FMPSMBUS_FLAG_STOPF, RESET, Timeout) != HAL_OK)
        {
          return HAL_ERROR;
        }

        /* Clear NACK Flag */
        __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_AF);

        /* Clear STOP Flag, auto generated with autoend*/
        __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_STOPF);
      }

      /* Check if the maximum allowed number of trials has been reached */
      if (FMPSMBUS_Trials == Trials)
      {
        /* Generate Stop */
        hfmpsmbus->Instance->CR2 |= FMPI2C_CR2_STOP;

        /* Wait until STOPF flag is reset */
        if (FMPSMBUS_WaitOnFlagUntilTimeout(hfmpsmbus, FMPSMBUS_FLAG_STOPF, RESET, Timeout) != HAL_OK)
        {
          return HAL_ERROR;
        }

        /* Clear STOP Flag */
        __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_STOPF);
      }

      /* Increment Trials */
      FMPSMBUS_Trials++;
    } while (FMPSMBUS_Trials < Trials);

    hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

    /* Update FMPSMBUS error code */
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_HALTIMEOUT;

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    return HAL_ERROR;
  }
  else
  {
    return HAL_BUSY;
  }
}
/**
  * @}
  */

/** @defgroup FMPSMBUS_IRQ_Handler_and_Callbacks IRQ Handler and Callbacks
  * @{
  */

/**
  * @brief  Handle FMPSMBUS event interrupt request.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
void HAL_FMPSMBUS_EV_IRQHandler(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Use a local variable to store the current ISR flags */
  /* This action will avoid a wrong treatment due to ISR flags change during interrupt handler */
  uint32_t tmpisrvalue = READ_REG(hfmpsmbus->Instance->ISR);
  uint32_t tmpcr1value = READ_REG(hfmpsmbus->Instance->CR1);

  /* FMPSMBUS in mode Transmitter ---------------------------------------------------*/
  if ((FMPSMBUS_CHECK_IT_SOURCE(tmpcr1value, (FMPSMBUS_IT_TCI | FMPSMBUS_IT_STOPI |
                                            FMPSMBUS_IT_NACKI | FMPSMBUS_IT_TXI)) != RESET) &&
      ((FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_TXIS) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_TCR) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_TC) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_STOPF) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_AF) != RESET)))
  {
    /* Slave mode selected */
    if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX) == HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX)
    {
      (void)FMPSMBUS_Slave_ISR(hfmpsmbus, tmpisrvalue);
    }
    /* Master mode selected */
    else if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_MASTER_BUSY_TX) == HAL_FMPSMBUS_STATE_MASTER_BUSY_TX)
    {
      (void)FMPSMBUS_Master_ISR(hfmpsmbus, tmpisrvalue);
    }
    else
    {
      /* Nothing to do */
    }
  }

  /* FMPSMBUS in mode Receiver ----------------------------------------------------*/
  if ((FMPSMBUS_CHECK_IT_SOURCE(tmpcr1value, (FMPSMBUS_IT_TCI | FMPSMBUS_IT_STOPI |
                                            FMPSMBUS_IT_NACKI | FMPSMBUS_IT_RXI)) != RESET) &&
      ((FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_RXNE) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_TCR) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_TC) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_STOPF) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_AF) != RESET)))
  {
    /* Slave mode selected */
    if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX) == HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX)
    {
      (void)FMPSMBUS_Slave_ISR(hfmpsmbus, tmpisrvalue);
    }
    /* Master mode selected */
    else if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_MASTER_BUSY_RX) == HAL_FMPSMBUS_STATE_MASTER_BUSY_RX)
    {
      (void)FMPSMBUS_Master_ISR(hfmpsmbus, tmpisrvalue);
    }
    else
    {
      /* Nothing to do */
    }
  }

  /* FMPSMBUS in mode Listener Only --------------------------------------------------*/
  if (((FMPSMBUS_CHECK_IT_SOURCE(tmpcr1value, FMPSMBUS_IT_ADDRI) != RESET) ||
       (FMPSMBUS_CHECK_IT_SOURCE(tmpcr1value, FMPSMBUS_IT_STOPI) != RESET) ||
       (FMPSMBUS_CHECK_IT_SOURCE(tmpcr1value, FMPSMBUS_IT_NACKI) != RESET)) &&
      ((FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_ADDR) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_STOPF) != RESET) ||
       (FMPSMBUS_CHECK_FLAG(tmpisrvalue, FMPSMBUS_FLAG_AF) != RESET)))
  {
    if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_LISTEN) == HAL_FMPSMBUS_STATE_LISTEN)
    {
      (void)FMPSMBUS_Slave_ISR(hfmpsmbus, tmpisrvalue);
    }
  }
}

/**
  * @brief  Handle FMPSMBUS error interrupt request.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
void HAL_FMPSMBUS_ER_IRQHandler(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  FMPSMBUS_ITErrorHandler(hfmpsmbus);
}

/**
  * @brief  Master Tx Transfer completed callback.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
__weak void HAL_FMPSMBUS_MasterTxCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hfmpsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_FMPSMBUS_MasterTxCpltCallback() could be implemented in the user file
   */
}

/**
  * @brief  Master Rx Transfer completed callback.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
__weak void HAL_FMPSMBUS_MasterRxCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hfmpsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_FMPSMBUS_MasterRxCpltCallback() could be implemented in the user file
   */
}

/** @brief  Slave Tx Transfer completed callback.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
__weak void HAL_FMPSMBUS_SlaveTxCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hfmpsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_FMPSMBUS_SlaveTxCpltCallback() could be implemented in the user file
   */
}

/**
  * @brief  Slave Rx Transfer completed callback.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
__weak void HAL_FMPSMBUS_SlaveRxCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hfmpsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_FMPSMBUS_SlaveRxCpltCallback() could be implemented in the user file
   */
}

/**
  * @brief  Slave Address Match callback.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  TransferDirection Master request Transfer Direction (Write/Read)
  * @param  AddrMatchCode Address Match Code
  * @retval None
  */
__weak void HAL_FMPSMBUS_AddrCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint8_t TransferDirection,
                                      uint16_t AddrMatchCode)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hfmpsmbus);
  UNUSED(TransferDirection);
  UNUSED(AddrMatchCode);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_FMPSMBUS_AddrCallback() could be implemented in the user file
   */
}

/**
  * @brief  Listen Complete callback.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
__weak void HAL_FMPSMBUS_ListenCpltCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hfmpsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_FMPSMBUS_ListenCpltCallback() could be implemented in the user file
   */
}

/**
  * @brief  FMPSMBUS error callback.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval None
  */
__weak void HAL_FMPSMBUS_ErrorCallback(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hfmpsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_FMPSMBUS_ErrorCallback() could be implemented in the user file
   */
}

/**
  * @}
  */

/** @defgroup FMPSMBUS_Exported_Functions_Group3 Peripheral State and Errors functions
  *  @brief   Peripheral State and Errors functions
  *
@verbatim
 ===============================================================================
            ##### Peripheral State and Errors functions #####
 ===============================================================================
    [..]
    This subsection permits to get in run-time the status of the peripheral
    and the data flow.

@endverbatim
  * @{
  */

/**
  * @brief  Return the FMPSMBUS handle state.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @retval HAL state
  */
uint32_t HAL_FMPSMBUS_GetState(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* Return FMPSMBUS handle state */
  return hfmpsmbus->State;
}

/**
  * @brief  Return the FMPSMBUS error code.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *              the configuration information for the specified FMPSMBUS.
  * @retval FMPSMBUS Error Code
  */
uint32_t HAL_FMPSMBUS_GetError(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  return hfmpsmbus->ErrorCode;
}

/**
  * @}
  */

/**
  * @}
  */

/** @addtogroup FMPSMBUS_Private_Functions FMPSMBUS Private Functions
  *  @brief   Data transfers Private functions
  * @{
  */

/**
  * @brief  Interrupt Sub-Routine which handle the Interrupt Flags Master Mode.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  StatusFlags Value of Interrupt Flags.
  * @retval HAL status
  */
static HAL_StatusTypeDef FMPSMBUS_Master_ISR(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t StatusFlags)
{
  uint16_t DevAddress;

  /* Process Locked */
  __HAL_LOCK(hfmpsmbus);

  if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_AF) != RESET)
  {
    /* Clear NACK Flag */
    __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_AF);

    /* Set corresponding Error Code */
    /* No need to generate STOP, it is automatically done */
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_ACKF;

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    /* Call the Error callback to inform upper layer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
    hfmpsmbus->ErrorCallback(hfmpsmbus);
#else
    HAL_FMPSMBUS_ErrorCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
  }
  else if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_STOPF) != RESET)
  {
    /* Check and treat errors if errors occurs during STOP process */
    FMPSMBUS_ITErrorHandler(hfmpsmbus);

    /* Call the corresponding callback to inform upper layer of End of Transfer */
    if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_MASTER_BUSY_TX)
    {
      /* Disable Interrupt */
      FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_TX);

      /* Clear STOP Flag */
      __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_STOPF);

      /* Clear Configuration Register 2 */
      FMPSMBUS_RESET_CR2(hfmpsmbus);

      /* Flush remaining data in Fifo register in case of error occurs before TXEmpty */
      /* Disable the selected FMPSMBUS peripheral */
      __HAL_FMPSMBUS_DISABLE(hfmpsmbus);

      hfmpsmbus->PreviousState = HAL_FMPSMBUS_STATE_READY;
      hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

      /* Process Unlocked */
      __HAL_UNLOCK(hfmpsmbus);

      /* Re-enable the selected FMPSMBUS peripheral */
      __HAL_FMPSMBUS_ENABLE(hfmpsmbus);

      /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
      hfmpsmbus->MasterTxCpltCallback(hfmpsmbus);
#else
      HAL_FMPSMBUS_MasterTxCpltCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
    }
    else if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_MASTER_BUSY_RX)
    {
      /* Store Last receive data if any */
      if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_RXNE) != RESET)
      {
        /* Read data from RXDR */
        *hfmpsmbus->pBuffPtr = (uint8_t)(hfmpsmbus->Instance->RXDR);

        /* Increment Buffer pointer */
        hfmpsmbus->pBuffPtr++;

        if ((hfmpsmbus->XferSize > 0U))
        {
          hfmpsmbus->XferSize--;
          hfmpsmbus->XferCount--;
        }
      }

      /* Disable Interrupt */
      FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_RX);

      /* Clear STOP Flag */
      __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_STOPF);

      /* Clear Configuration Register 2 */
      FMPSMBUS_RESET_CR2(hfmpsmbus);

      hfmpsmbus->PreviousState = HAL_FMPSMBUS_STATE_READY;
      hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

      /* Process Unlocked */
      __HAL_UNLOCK(hfmpsmbus);

      /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
      hfmpsmbus->MasterRxCpltCallback(hfmpsmbus);
#else
      HAL_FMPSMBUS_MasterRxCpltCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
    }
    else
    {
      /* Nothing to do */
    }
  }
  else if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_RXNE) != RESET)
  {
    /* Read data from RXDR */
    *hfmpsmbus->pBuffPtr = (uint8_t)(hfmpsmbus->Instance->RXDR);

    /* Increment Buffer pointer */
    hfmpsmbus->pBuffPtr++;

    /* Increment Size counter */
    hfmpsmbus->XferSize--;
    hfmpsmbus->XferCount--;
  }
  else if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_TXIS) != RESET)
  {
    /* Write data to TXDR */
    hfmpsmbus->Instance->TXDR = *hfmpsmbus->pBuffPtr;

    /* Increment Buffer pointer */
    hfmpsmbus->pBuffPtr++;

    /* Increment Size counter */
    hfmpsmbus->XferSize--;
    hfmpsmbus->XferCount--;
  }
  else if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_TCR) != RESET)
  {
    if ((hfmpsmbus->XferCount != 0U) && (hfmpsmbus->XferSize == 0U))
    {
      DevAddress = (uint16_t)(hfmpsmbus->Instance->CR2 & FMPI2C_CR2_SADD);

      if (hfmpsmbus->XferCount > MAX_NBYTE_SIZE)
      {
        FMPSMBUS_TransferConfig(hfmpsmbus, DevAddress, MAX_NBYTE_SIZE,
                             (FMPSMBUS_RELOAD_MODE | (hfmpsmbus->XferOptions & FMPSMBUS_SENDPEC_MODE)),
                             FMPSMBUS_NO_STARTSTOP);
        hfmpsmbus->XferSize = MAX_NBYTE_SIZE;
      }
      else
      {
        hfmpsmbus->XferSize = hfmpsmbus->XferCount;
        FMPSMBUS_TransferConfig(hfmpsmbus, DevAddress, (uint8_t)hfmpsmbus->XferSize, hfmpsmbus->XferOptions,
                                FMPSMBUS_NO_STARTSTOP);
        /* If PEC mode is enable, size to transmit should be Size-1 byte, corresponding to PEC byte */
        /* PEC byte is automatically sent by HW block, no need to manage it in Transmit process */
        if (FMPSMBUS_GET_PEC_MODE(hfmpsmbus) != 0UL)
        {
          hfmpsmbus->XferSize--;
          hfmpsmbus->XferCount--;
        }
      }
    }
    else if ((hfmpsmbus->XferCount == 0U) && (hfmpsmbus->XferSize == 0U))
    {
      /* Call TxCpltCallback() if no stop mode is set */
      if (FMPSMBUS_GET_STOP_MODE(hfmpsmbus) != FMPSMBUS_AUTOEND_MODE)
      {
        /* Call the corresponding callback to inform upper layer of End of Transfer */
        if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_MASTER_BUSY_TX)
        {
          /* Disable Interrupt */
          FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_TX);
          hfmpsmbus->PreviousState = hfmpsmbus->State;
          hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hfmpsmbus);

          /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
          hfmpsmbus->MasterTxCpltCallback(hfmpsmbus);
#else
          HAL_FMPSMBUS_MasterTxCpltCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
        }
        else if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_MASTER_BUSY_RX)
        {
          FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_RX);
          hfmpsmbus->PreviousState = hfmpsmbus->State;
          hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hfmpsmbus);

          /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
          hfmpsmbus->MasterRxCpltCallback(hfmpsmbus);
#else
          HAL_FMPSMBUS_MasterRxCpltCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
        }
        else
        {
          /* Nothing to do */
        }
      }
    }
    else
    {
      /* Nothing to do */
    }
  }
  else if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_TC) != RESET)
  {
    if (hfmpsmbus->XferCount == 0U)
    {
      /* Specific use case for Quick command */
      if (hfmpsmbus->pBuffPtr == NULL)
      {
        /* Generate a Stop command */
        hfmpsmbus->Instance->CR2 |= FMPI2C_CR2_STOP;
      }
      /* Call TxCpltCallback() if no stop mode is set */
      else if (FMPSMBUS_GET_STOP_MODE(hfmpsmbus) != FMPSMBUS_AUTOEND_MODE)
      {
        /* No Generate Stop, to permit restart mode */
        /* The stop will be done at the end of transfer, when FMPSMBUS_AUTOEND_MODE enable */

        /* Call the corresponding callback to inform upper layer of End of Transfer */
        if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_MASTER_BUSY_TX)
        {
          /* Disable Interrupt */
          FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_TX);
          hfmpsmbus->PreviousState = hfmpsmbus->State;
          hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hfmpsmbus);

          /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
          hfmpsmbus->MasterTxCpltCallback(hfmpsmbus);
#else
          HAL_FMPSMBUS_MasterTxCpltCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
        }
        else if (hfmpsmbus->State == HAL_FMPSMBUS_STATE_MASTER_BUSY_RX)
        {
          FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_RX);
          hfmpsmbus->PreviousState = hfmpsmbus->State;
          hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hfmpsmbus);

          /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
          hfmpsmbus->MasterRxCpltCallback(hfmpsmbus);
#else
          HAL_FMPSMBUS_MasterRxCpltCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
        }
        else
        {
          /* Nothing to do */
        }
      }
      else
      {
        /* Nothing to do */
      }
    }
  }
  else
  {
    /* Nothing to do */
  }

  /* Process Unlocked */
  __HAL_UNLOCK(hfmpsmbus);

  return HAL_OK;
}
/**
  * @brief  Interrupt Sub-Routine which handle the Interrupt Flags Slave Mode.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  StatusFlags Value of Interrupt Flags.
  * @retval HAL status
  */
static HAL_StatusTypeDef FMPSMBUS_Slave_ISR(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t StatusFlags)
{
  uint8_t TransferDirection;
  uint16_t SlaveAddrCode;

  /* Process Locked */
  __HAL_LOCK(hfmpsmbus);

  if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_AF) != RESET)
  {
    /* Check that FMPSMBUS transfer finished */
    /* if yes, normal usecase, a NACK is sent by the HOST when Transfer is finished */
    /* Mean XferCount == 0*/
    /* So clear Flag NACKF only */
    if (hfmpsmbus->XferCount == 0U)
    {
      /* Clear NACK Flag */
      __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_AF);

      /* Process Unlocked */
      __HAL_UNLOCK(hfmpsmbus);
    }
    else
    {
      /* if no, error usecase, a Non-Acknowledge of last Data is generated by the HOST*/
      /* Clear NACK Flag */
      __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_AF);

      /* Set HAL State to "Idle" State, mean to LISTEN state */
      /* So reset Slave Busy state */
      hfmpsmbus->PreviousState = hfmpsmbus->State;
      hfmpsmbus->State &= ~((uint32_t)HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX);
      hfmpsmbus->State &= ~((uint32_t)HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX);

      /* Disable RX/TX Interrupts, keep only ADDR Interrupt */
      FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_RX | FMPSMBUS_IT_TX);

      /* Set ErrorCode corresponding to a Non-Acknowledge */
      hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_ACKF;

      /* Process Unlocked */
      __HAL_UNLOCK(hfmpsmbus);

      /* Call the Error callback to inform upper layer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
      hfmpsmbus->ErrorCallback(hfmpsmbus);
#else
      HAL_FMPSMBUS_ErrorCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
    }
  }
  else if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_ADDR) != RESET)
  {
    TransferDirection = (uint8_t)(FMPSMBUS_GET_DIR(hfmpsmbus));
    SlaveAddrCode = (uint16_t)(FMPSMBUS_GET_ADDR_MATCH(hfmpsmbus));

    /* Disable ADDR interrupt to prevent multiple ADDRInterrupt*/
    /* Other ADDRInterrupt will be treat in next Listen usecase */
    __HAL_FMPSMBUS_DISABLE_IT(hfmpsmbus, FMPSMBUS_IT_ADDRI);

    /* Process Unlocked */
    __HAL_UNLOCK(hfmpsmbus);

    /* Call Slave Addr callback */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
    hfmpsmbus->AddrCallback(hfmpsmbus, TransferDirection, SlaveAddrCode);
#else
    HAL_FMPSMBUS_AddrCallback(hfmpsmbus, TransferDirection, SlaveAddrCode);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
  }
  else if ((FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_RXNE) != RESET) ||
           (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_TCR) != RESET))
  {
    if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX) == HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX)
    {
      /* Read data from RXDR */
      *hfmpsmbus->pBuffPtr = (uint8_t)(hfmpsmbus->Instance->RXDR);

      /* Increment Buffer pointer */
      hfmpsmbus->pBuffPtr++;

      hfmpsmbus->XferSize--;
      hfmpsmbus->XferCount--;

      if (hfmpsmbus->XferCount == 1U)
      {
        /* Receive last Byte, can be PEC byte in case of PEC BYTE enabled */
        /* or only the last Byte of Transfer */
        /* So reset the RELOAD bit mode */
        hfmpsmbus->XferOptions &= ~FMPSMBUS_RELOAD_MODE;
        FMPSMBUS_TransferConfig(hfmpsmbus, 0, 1, hfmpsmbus->XferOptions, FMPSMBUS_NO_STARTSTOP);
      }
      else if (hfmpsmbus->XferCount == 0U)
      {
        /* Last Byte is received, disable Interrupt */
        FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_RX);

        /* Remove HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX, keep only HAL_FMPSMBUS_STATE_LISTEN */
        hfmpsmbus->PreviousState = hfmpsmbus->State;
        hfmpsmbus->State &= ~((uint32_t)HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX);

        /* Process Unlocked */
        __HAL_UNLOCK(hfmpsmbus);

        /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
        hfmpsmbus->SlaveRxCpltCallback(hfmpsmbus);
#else
        HAL_FMPSMBUS_SlaveRxCpltCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
      }
      else
      {
        /* Set Reload for next Bytes */
        FMPSMBUS_TransferConfig(hfmpsmbus, 0, 1,
                             FMPSMBUS_RELOAD_MODE  | (hfmpsmbus->XferOptions & FMPSMBUS_SENDPEC_MODE),
                             FMPSMBUS_NO_STARTSTOP);

        /* Ack last Byte Read */
        hfmpsmbus->Instance->CR2 &= ~FMPI2C_CR2_NACK;
      }
    }
    else if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX) == HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX)
    {
      if ((hfmpsmbus->XferCount != 0U) && (hfmpsmbus->XferSize == 0U))
      {
        if (hfmpsmbus->XferCount > MAX_NBYTE_SIZE)
        {
          FMPSMBUS_TransferConfig(hfmpsmbus, 0, MAX_NBYTE_SIZE,
                               (FMPSMBUS_RELOAD_MODE | (hfmpsmbus->XferOptions & FMPSMBUS_SENDPEC_MODE)),
                               FMPSMBUS_NO_STARTSTOP);
          hfmpsmbus->XferSize = MAX_NBYTE_SIZE;
        }
        else
        {
          hfmpsmbus->XferSize = hfmpsmbus->XferCount;
          FMPSMBUS_TransferConfig(hfmpsmbus, 0, (uint8_t)hfmpsmbus->XferSize, hfmpsmbus->XferOptions,
                                  FMPSMBUS_NO_STARTSTOP);
          /* If PEC mode is enable, size to transmit should be Size-1 byte, corresponding to PEC byte */
          /* PEC byte is automatically sent by HW block, no need to manage it in Transmit process */
          if (FMPSMBUS_GET_PEC_MODE(hfmpsmbus) != 0UL)
          {
            hfmpsmbus->XferSize--;
            hfmpsmbus->XferCount--;
          }
        }
      }
    }
    else
    {
      /* Nothing to do */
    }
  }
  else if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_TXIS) != RESET)
  {
    /* Write data to TXDR only if XferCount not reach "0" */
    /* A TXIS flag can be set, during STOP treatment      */
    /* Check if all Data have already been sent */
    /* If it is the case, this last write in TXDR is not sent, correspond to a dummy TXIS event */
    if (hfmpsmbus->XferCount > 0U)
    {
      /* Write data to TXDR */
      hfmpsmbus->Instance->TXDR = *hfmpsmbus->pBuffPtr;

      /* Increment Buffer pointer */
      hfmpsmbus->pBuffPtr++;

      hfmpsmbus->XferCount--;
      hfmpsmbus->XferSize--;
    }

    if (hfmpsmbus->XferCount == 0U)
    {
      /* Last Byte is Transmitted */
      /* Remove HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX, keep only HAL_FMPSMBUS_STATE_LISTEN */
      FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_TX);
      hfmpsmbus->PreviousState = hfmpsmbus->State;
      hfmpsmbus->State &= ~((uint32_t)HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX);

      /* Process Unlocked */
      __HAL_UNLOCK(hfmpsmbus);

      /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
      hfmpsmbus->SlaveTxCpltCallback(hfmpsmbus);
#else
      HAL_FMPSMBUS_SlaveTxCpltCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
    }
  }
  else
  {
    /* Nothing to do */
  }

  /* Check if STOPF is set */
  if (FMPSMBUS_CHECK_FLAG(StatusFlags, FMPSMBUS_FLAG_STOPF) != RESET)
  {
    if ((hfmpsmbus->State & HAL_FMPSMBUS_STATE_LISTEN) == HAL_FMPSMBUS_STATE_LISTEN)
    {
      /* Store Last receive data if any */
      if (__HAL_FMPSMBUS_GET_FLAG(hfmpsmbus, FMPSMBUS_FLAG_RXNE) != RESET)
      {
        /* Read data from RXDR */
        *hfmpsmbus->pBuffPtr = (uint8_t)(hfmpsmbus->Instance->RXDR);

        /* Increment Buffer pointer */
        hfmpsmbus->pBuffPtr++;

        if ((hfmpsmbus->XferSize > 0U))
        {
          hfmpsmbus->XferSize--;
          hfmpsmbus->XferCount--;
        }
      }

      /* Disable RX and TX Interrupts */
      FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_RX | FMPSMBUS_IT_TX);

      /* Disable ADDR Interrupt */
      FMPSMBUS_Disable_IRQ(hfmpsmbus, FMPSMBUS_IT_ADDR);

      /* Disable Address Acknowledge */
      hfmpsmbus->Instance->CR2 |= FMPI2C_CR2_NACK;

      /* Clear Configuration Register 2 */
      FMPSMBUS_RESET_CR2(hfmpsmbus);

      /* Clear STOP Flag */
      __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_STOPF);

      /* Clear ADDR flag */
      __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_ADDR);

      hfmpsmbus->XferOptions = 0;
      hfmpsmbus->PreviousState = hfmpsmbus->State;
      hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

      /* Process Unlocked */
      __HAL_UNLOCK(hfmpsmbus);

      /* Call the Listen Complete callback, to inform upper layer of the end of Listen usecase */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
      hfmpsmbus->ListenCpltCallback(hfmpsmbus);
#else
      HAL_FMPSMBUS_ListenCpltCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
    }
  }

  /* Process Unlocked */
  __HAL_UNLOCK(hfmpsmbus);

  return HAL_OK;
}
/**
  * @brief  Manage the enabling of Interrupts.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  InterruptRequest Value of @ref FMPSMBUS_Interrupt_configuration_definition.
  * @retval HAL status
  */
static void FMPSMBUS_Enable_IRQ(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t InterruptRequest)
{
  uint32_t tmpisr = 0UL;

  if ((InterruptRequest & FMPSMBUS_IT_ALERT) == FMPSMBUS_IT_ALERT)
  {
    /* Enable ERR interrupt */
    tmpisr |= FMPSMBUS_IT_ERRI;
  }

  if ((InterruptRequest & FMPSMBUS_IT_ADDR) == FMPSMBUS_IT_ADDR)
  {
    /* Enable ADDR, STOP interrupt */
    tmpisr |= FMPSMBUS_IT_ADDRI | FMPSMBUS_IT_STOPI | FMPSMBUS_IT_NACKI | FMPSMBUS_IT_ERRI;
  }

  if ((InterruptRequest & FMPSMBUS_IT_TX) == FMPSMBUS_IT_TX)
  {
    /* Enable ERR, TC, STOP, NACK, RXI interrupt */
    tmpisr |= FMPSMBUS_IT_ERRI | FMPSMBUS_IT_TCI | FMPSMBUS_IT_STOPI | FMPSMBUS_IT_NACKI | FMPSMBUS_IT_TXI;
  }

  if ((InterruptRequest & FMPSMBUS_IT_RX) == FMPSMBUS_IT_RX)
  {
    /* Enable ERR, TC, STOP, NACK, TXI interrupt */
    tmpisr |= FMPSMBUS_IT_ERRI | FMPSMBUS_IT_TCI | FMPSMBUS_IT_STOPI | FMPSMBUS_IT_NACKI | FMPSMBUS_IT_RXI;
  }

  /* Enable interrupts only at the end */
  /* to avoid the risk of FMPSMBUS interrupt handle execution before */
  /* all interrupts requested done */
  __HAL_FMPSMBUS_ENABLE_IT(hfmpsmbus, tmpisr);
}
/**
  * @brief  Manage the disabling of Interrupts.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  InterruptRequest Value of @ref FMPSMBUS_Interrupt_configuration_definition.
  * @retval HAL status
  */
static void FMPSMBUS_Disable_IRQ(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t InterruptRequest)
{
  uint32_t tmpisr = 0UL;
  uint32_t tmpstate = hfmpsmbus->State;

  if ((tmpstate == HAL_FMPSMBUS_STATE_READY) && ((InterruptRequest & FMPSMBUS_IT_ALERT) == FMPSMBUS_IT_ALERT))
  {
    /* Disable ERR interrupt */
    tmpisr |= FMPSMBUS_IT_ERRI;
  }

  if ((InterruptRequest & FMPSMBUS_IT_TX) == FMPSMBUS_IT_TX)
  {
    /* Disable TC, STOP, NACK and TXI interrupt */
    tmpisr |= FMPSMBUS_IT_TCI | FMPSMBUS_IT_TXI;

    if ((FMPSMBUS_GET_ALERT_ENABLED(hfmpsmbus) == 0UL)
        && ((tmpstate & HAL_FMPSMBUS_STATE_LISTEN) != HAL_FMPSMBUS_STATE_LISTEN))
    {
      /* Disable ERR interrupt */
      tmpisr |= FMPSMBUS_IT_ERRI;
    }

    if ((tmpstate & HAL_FMPSMBUS_STATE_LISTEN) != HAL_FMPSMBUS_STATE_LISTEN)
    {
      /* Disable STOP and NACK interrupt */
      tmpisr |= FMPSMBUS_IT_STOPI | FMPSMBUS_IT_NACKI;
    }
  }

  if ((InterruptRequest & FMPSMBUS_IT_RX) == FMPSMBUS_IT_RX)
  {
    /* Disable TC, STOP, NACK and RXI interrupt */
    tmpisr |= FMPSMBUS_IT_TCI | FMPSMBUS_IT_RXI;

    if ((FMPSMBUS_GET_ALERT_ENABLED(hfmpsmbus) == 0UL)
        && ((tmpstate & HAL_FMPSMBUS_STATE_LISTEN) != HAL_FMPSMBUS_STATE_LISTEN))
    {
      /* Disable ERR interrupt */
      tmpisr |= FMPSMBUS_IT_ERRI;
    }

    if ((tmpstate & HAL_FMPSMBUS_STATE_LISTEN) != HAL_FMPSMBUS_STATE_LISTEN)
    {
      /* Disable STOP and NACK interrupt */
      tmpisr |= FMPSMBUS_IT_STOPI | FMPSMBUS_IT_NACKI;
    }
  }

  if ((InterruptRequest & FMPSMBUS_IT_ADDR) == FMPSMBUS_IT_ADDR)
  {
    /* Disable ADDR, STOP and NACK interrupt */
    tmpisr |= FMPSMBUS_IT_ADDRI | FMPSMBUS_IT_STOPI | FMPSMBUS_IT_NACKI;

    if (FMPSMBUS_GET_ALERT_ENABLED(hfmpsmbus) == 0UL)
    {
      /* Disable ERR interrupt */
      tmpisr |= FMPSMBUS_IT_ERRI;
    }
  }

  /* Disable interrupts only at the end */
  /* to avoid a breaking situation like at "t" time */
  /* all disable interrupts request are not done */
  __HAL_FMPSMBUS_DISABLE_IT(hfmpsmbus, tmpisr);
}

/**
  * @brief  FMPSMBUS interrupts error handler.
  * @param  hfmpsmbus FMPSMBUS handle.
  * @retval None
  */
static void FMPSMBUS_ITErrorHandler(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  uint32_t itflags   = READ_REG(hfmpsmbus->Instance->ISR);
  uint32_t itsources = READ_REG(hfmpsmbus->Instance->CR1);
  uint32_t tmpstate;
  uint32_t tmperror;

  /* FMPSMBUS Bus error interrupt occurred ------------------------------------*/
  if (((itflags & FMPSMBUS_FLAG_BERR) == FMPSMBUS_FLAG_BERR) && \
      ((itsources & FMPSMBUS_IT_ERRI) == FMPSMBUS_IT_ERRI))
  {
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_BERR;

    /* Clear BERR flag */
    __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_BERR);
  }

  /* FMPSMBUS Over-Run/Under-Run interrupt occurred ----------------------------------------*/
  if (((itflags & FMPSMBUS_FLAG_OVR) == FMPSMBUS_FLAG_OVR) && \
      ((itsources & FMPSMBUS_IT_ERRI) == FMPSMBUS_IT_ERRI))
  {
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_OVR;

    /* Clear OVR flag */
    __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_OVR);
  }

  /* FMPSMBUS Arbitration Loss error interrupt occurred ------------------------------------*/
  if (((itflags & FMPSMBUS_FLAG_ARLO) == FMPSMBUS_FLAG_ARLO) && \
      ((itsources & FMPSMBUS_IT_ERRI) == FMPSMBUS_IT_ERRI))
  {
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_ARLO;

    /* Clear ARLO flag */
    __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_ARLO);
  }

  /* FMPSMBUS Timeout error interrupt occurred ---------------------------------------------*/
  if (((itflags & FMPSMBUS_FLAG_TIMEOUT) == FMPSMBUS_FLAG_TIMEOUT) && \
      ((itsources & FMPSMBUS_IT_ERRI) == FMPSMBUS_IT_ERRI))
  {
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_BUSTIMEOUT;

    /* Clear TIMEOUT flag */
    __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_TIMEOUT);
  }

  /* FMPSMBUS Alert error interrupt occurred -----------------------------------------------*/
  if (((itflags & FMPSMBUS_FLAG_ALERT) == FMPSMBUS_FLAG_ALERT) && \
      ((itsources & FMPSMBUS_IT_ERRI) == FMPSMBUS_IT_ERRI))
  {
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_ALERT;

    /* Clear ALERT flag */
    __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_ALERT);
  }

  /* FMPSMBUS Packet Error Check error interrupt occurred ----------------------------------*/
  if (((itflags & FMPSMBUS_FLAG_PECERR) == FMPSMBUS_FLAG_PECERR) && \
      ((itsources & FMPSMBUS_IT_ERRI) == FMPSMBUS_IT_ERRI))
  {
    hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_PECERR;

    /* Clear PEC error flag */
    __HAL_FMPSMBUS_CLEAR_FLAG(hfmpsmbus, FMPSMBUS_FLAG_PECERR);
  }

  /* Store current volatile hfmpsmbus->State, misra rule */
  tmperror = hfmpsmbus->ErrorCode;

  /* Call the Error Callback in case of Error detected */
  if ((tmperror != HAL_FMPSMBUS_ERROR_NONE) && (tmperror != HAL_FMPSMBUS_ERROR_ACKF))
  {
    /* Do not Reset the HAL state in case of ALERT error */
    if ((tmperror & HAL_FMPSMBUS_ERROR_ALERT) != HAL_FMPSMBUS_ERROR_ALERT)
    {
      /* Store current volatile hfmpsmbus->State, misra rule */
      tmpstate = hfmpsmbus->State;

      if (((tmpstate & HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX) == HAL_FMPSMBUS_STATE_SLAVE_BUSY_TX)
          || ((tmpstate & HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX) == HAL_FMPSMBUS_STATE_SLAVE_BUSY_RX))
      {
        /* Reset only HAL_FMPSMBUS_STATE_SLAVE_BUSY_XX */
        /* keep HAL_FMPSMBUS_STATE_LISTEN if set */
        hfmpsmbus->PreviousState = HAL_FMPSMBUS_STATE_READY;
        hfmpsmbus->State = HAL_FMPSMBUS_STATE_LISTEN;
      }
    }

    /* Call the Error callback to inform upper layer */
#if (USE_HAL_FMPSMBUS_REGISTER_CALLBACKS == 1)
    hfmpsmbus->ErrorCallback(hfmpsmbus);
#else
    HAL_FMPSMBUS_ErrorCallback(hfmpsmbus);
#endif /* USE_HAL_FMPSMBUS_REGISTER_CALLBACKS */
  }
}

/**
  * @brief  Handle FMPSMBUS Communication Timeout.
  * @param  hfmpsmbus Pointer to a FMPSMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified FMPSMBUS.
  * @param  Flag Specifies the FMPSMBUS flag to check.
  * @param  Status The new Flag status (SET or RESET).
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
static HAL_StatusTypeDef FMPSMBUS_WaitOnFlagUntilTimeout(FMPSMBUS_HandleTypeDef *hfmpsmbus, uint32_t Flag,
                                                         FlagStatus Status, uint32_t Timeout)
{
  uint32_t tickstart = HAL_GetTick();

  /* Wait until flag is set */
  while ((FlagStatus)(__HAL_FMPSMBUS_GET_FLAG(hfmpsmbus, Flag)) == Status)
  {
    /* Check for the Timeout */
    if (Timeout != HAL_MAX_DELAY)
    {
      if (((HAL_GetTick() - tickstart) > Timeout) || (Timeout == 0UL))
      {
        hfmpsmbus->PreviousState = hfmpsmbus->State;
        hfmpsmbus->State = HAL_FMPSMBUS_STATE_READY;

        /* Update FMPSMBUS error code */
        hfmpsmbus->ErrorCode |= HAL_FMPSMBUS_ERROR_HALTIMEOUT;

        /* Process Unlocked */
        __HAL_UNLOCK(hfmpsmbus);

        return HAL_ERROR;
      }
    }
  }

  return HAL_OK;
}

/**
  * @brief  Handle FMPSMBUSx communication when starting transfer or during transfer (TC or TCR flag are set).
  * @param  hfmpsmbus FMPSMBUS handle.
  * @param  DevAddress specifies the slave address to be programmed.
  * @param  Size specifies the number of bytes to be programmed.
  *   This parameter must be a value between 0 and 255.
  * @param  Mode New state of the FMPSMBUS START condition generation.
  *   This parameter can be one or a combination  of the following values:
  *     @arg @ref FMPSMBUS_RELOAD_MODE Enable Reload mode.
  *     @arg @ref FMPSMBUS_AUTOEND_MODE Enable Automatic end mode.
  *     @arg @ref FMPSMBUS_SOFTEND_MODE Enable Software end mode and Reload mode.
  *     @arg @ref FMPSMBUS_SENDPEC_MODE Enable Packet Error Calculation mode.
  * @param  Request New state of the FMPSMBUS START condition generation.
  *   This parameter can be one of the following values:
  *     @arg @ref FMPSMBUS_NO_STARTSTOP Don't Generate stop and start condition.
  *     @arg @ref FMPSMBUS_GENERATE_STOP Generate stop condition (Size should be set to 0).
  *     @arg @ref FMPSMBUS_GENERATE_START_READ Generate Restart for read request.
  *     @arg @ref FMPSMBUS_GENERATE_START_WRITE Generate Restart for write request.
  * @retval None
  */
static void FMPSMBUS_TransferConfig(FMPSMBUS_HandleTypeDef *hfmpsmbus,  uint16_t DevAddress, uint8_t Size,
                                    uint32_t Mode, uint32_t Request)
{
  /* Check the parameters */
  assert_param(IS_FMPSMBUS_ALL_INSTANCE(hfmpsmbus->Instance));
  assert_param(IS_FMPSMBUS_TRANSFER_MODE(Mode));
  assert_param(IS_FMPSMBUS_TRANSFER_REQUEST(Request));

  /* update CR2 register */
  MODIFY_REG(hfmpsmbus->Instance->CR2,
             ((FMPI2C_CR2_SADD | FMPI2C_CR2_NBYTES | FMPI2C_CR2_RELOAD | FMPI2C_CR2_AUTOEND | \
               (FMPI2C_CR2_RD_WRN & (uint32_t)(Request >> (31UL - FMPI2C_CR2_RD_WRN_Pos))) | \
               FMPI2C_CR2_START | FMPI2C_CR2_STOP | FMPI2C_CR2_PECBYTE)), \
             (uint32_t)(((uint32_t)DevAddress & FMPI2C_CR2_SADD) | \
                        (((uint32_t)Size << FMPI2C_CR2_NBYTES_Pos) & FMPI2C_CR2_NBYTES) | \
                        (uint32_t)Mode | (uint32_t)Request));
}

/**
  * @brief  Convert FMPSMBUSx OTHER_xxx XferOptions to functional XferOptions.
  * @param  hfmpsmbus FMPSMBUS handle.
  * @retval None
  */
static void FMPSMBUS_ConvertOtherXferOptions(FMPSMBUS_HandleTypeDef *hfmpsmbus)
{
  /* if user set XferOptions to FMPSMBUS_OTHER_FRAME_NO_PEC   */
  /* it request implicitly to generate a restart condition */
  /* set XferOptions to FMPSMBUS_FIRST_FRAME                  */
  if (hfmpsmbus->XferOptions == FMPSMBUS_OTHER_FRAME_NO_PEC)
  {
    hfmpsmbus->XferOptions = FMPSMBUS_FIRST_FRAME;
  }
  /* else if user set XferOptions to FMPSMBUS_OTHER_FRAME_WITH_PEC */
  /* it request implicitly to generate a restart condition      */
  /* set XferOptions to FMPSMBUS_FIRST_FRAME | FMPSMBUS_SENDPEC_MODE  */
  else if (hfmpsmbus->XferOptions == FMPSMBUS_OTHER_FRAME_WITH_PEC)
  {
    hfmpsmbus->XferOptions = FMPSMBUS_FIRST_FRAME | FMPSMBUS_SENDPEC_MODE;
  }
  /* else if user set XferOptions to FMPSMBUS_OTHER_AND_LAST_FRAME_NO_PEC */
  /* it request implicitly to generate a restart condition             */
  /* then generate a stop condition at the end of transfer             */
  /* set XferOptions to FMPSMBUS_FIRST_AND_LAST_FRAME_NO_PEC              */
  else if (hfmpsmbus->XferOptions == FMPSMBUS_OTHER_AND_LAST_FRAME_NO_PEC)
  {
    hfmpsmbus->XferOptions = FMPSMBUS_FIRST_AND_LAST_FRAME_NO_PEC;
  }
  /* else if user set XferOptions to FMPSMBUS_OTHER_AND_LAST_FRAME_WITH_PEC */
  /* it request implicitly to generate a restart condition               */
  /* then generate a stop condition at the end of transfer               */
  /* set XferOptions to FMPSMBUS_FIRST_AND_LAST_FRAME_WITH_PEC              */
  else if (hfmpsmbus->XferOptions == FMPSMBUS_OTHER_AND_LAST_FRAME_WITH_PEC)
  {
    hfmpsmbus->XferOptions = FMPSMBUS_FIRST_AND_LAST_FRAME_WITH_PEC;
  }
  else
  {
    /* Nothing to do */
  }
}
/**
  * @}
  */

#endif /* FMPI2C_CR1_PE */
#endif /* HAL_FMPSMBUS_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
