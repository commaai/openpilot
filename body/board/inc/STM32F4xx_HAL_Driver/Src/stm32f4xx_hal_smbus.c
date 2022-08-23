/**
  ******************************************************************************
  * @file    stm32f4xx_hal_smbus.c
  * @author  MCD Application Team
  * @brief   SMBUS HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the System Management Bus (SMBus) peripheral,
  *          based on SMBUS principals of operation :
  *           + Initialization and de-initialization functions
  *           + IO operation functions
  *           + Peripheral State, Mode and Error functions
  *
  @verbatim
  ==============================================================================
                        ##### How to use this driver #####
  ==============================================================================
  [..]
    The SMBUS HAL driver can be used as follows:

    (#) Declare a SMBUS_HandleTypeDef handle structure, for example:
        SMBUS_HandleTypeDef  hsmbus;

    (#)Initialize the SMBUS low level resources by implementing the HAL_SMBUS_MspInit() API:
        (##) Enable the SMBUSx interface clock
        (##) SMBUS pins configuration
            (+++) Enable the clock for the SMBUS GPIOs
            (+++) Configure SMBUS pins as alternate function open-drain
        (##) NVIC configuration if you need to use interrupt process
            (+++) Configure the SMBUSx interrupt priority
            (+++) Enable the NVIC SMBUS IRQ Channel

    (#) Configure the Communication Speed, Duty cycle, Addressing mode, Own Address1,
        Dual Addressing mode, Own Address2, General call and Nostretch mode in the hsmbus Init structure.

    (#) Initialize the SMBUS registers by calling the HAL_SMBUS_Init(), configures also the low level Hardware
        (GPIO, CLOCK, NVIC...etc) by calling the customized HAL_SMBUS_MspInit(&hsmbus) API.

    (#) To check if target device is ready for communication, use the function HAL_SMBUS_IsDeviceReady()

    (#) For SMBUS IO operations, only one mode of operations is available within this driver :


    *** Interrupt mode IO operation ***
    ===================================

  [..]
      (+) Transmit in master/host SMBUS mode an amount of data in non blocking mode using HAL_SMBUS_Master_Transmit_IT()
      (++) At transmission end of transfer HAL_SMBUS_MasterTxCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_SMBUS_MasterTxCpltCallback()
      (+) Receive in master/host SMBUS mode an amount of data in non blocking mode using HAL_SMBUS_Master_Receive_IT()
      (++) At reception end of transfer HAL_SMBUS_MasterRxCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_SMBUS_MasterRxCpltCallback()
      (+) Abort a master/Host SMBUS process communication with Interrupt using HAL_SMBUS_Master_Abort_IT()
      (++) End of abort process, HAL_SMBUS_AbortCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_SMBUS_AbortCpltCallback()
      (+) Enable/disable the Address listen mode in slave/device or host/slave SMBUS mode
           using HAL_SMBUS_EnableListen_IT() HAL_SMBUS_DisableListen_IT()
      (++) When address slave/device SMBUS match, HAL_SMBUS_AddrCallback() is executed and user can
           add his own code to check the Address Match Code and the transmission direction request by master/host (Write/Read).
      (++) At Listen mode end HAL_SMBUS_ListenCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_SMBUS_ListenCpltCallback()
      (+) Transmit in slave/device SMBUS mode an amount of data in non blocking mode using HAL_SMBUS_Slave_Transmit_IT()
      (++) At transmission end of transfer HAL_SMBUS_SlaveTxCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_SMBUS_SlaveTxCpltCallback()
      (+) Receive in slave/device SMBUS mode an amount of data in non blocking mode using HAL_SMBUS_Slave_Receive_IT()
      (++) At reception end of transfer HAL_SMBUS_SlaveRxCpltCallback() is executed and user can
           add his own code by customization of function pointer HAL_SMBUS_SlaveRxCpltCallback()
      (+) Enable/Disable the SMBUS alert mode using HAL_SMBUS_EnableAlert_IT() and HAL_SMBUS_DisableAlert_IT()
      (++) When SMBUS Alert is generated HAL_SMBUS_ErrorCallback() is executed and user can
           add his own code by customization of function pointer HAL_SMBUS_ErrorCallback()
           to check the Alert Error Code using function HAL_SMBUS_GetError()
      (+) Get HAL state machine or error values using HAL_SMBUS_GetState() or HAL_SMBUS_GetError()
      (+) In case of transfer Error, HAL_SMBUS_ErrorCallback() function is executed and user can
           add his own code by customization of function pointer HAL_SMBUS_ErrorCallback()
           to check the Error Code using function HAL_SMBUS_GetError()


     *** SMBUS HAL driver macros list ***
     ==================================
     [..]
       Below the list of most used macros in SMBUS HAL driver.

      (+) __HAL_SMBUS_ENABLE    : Enable the SMBUS peripheral
      (+) __HAL_SMBUS_DISABLE   : Disable the SMBUS peripheral
      (+) __HAL_SMBUS_GET_FLAG  : Checks whether the specified SMBUS flag is set or not
      (+) __HAL_SMBUS_CLEAR_FLAG: Clear the specified SMBUS pending flag
      (+) __HAL_SMBUS_ENABLE_IT : Enable the specified SMBUS interrupt
      (+) __HAL_SMBUS_DISABLE_IT: Disable the specified SMBUS interrupt

     [..]
       (@) You can refer to the SMBUS HAL driver header file for more useful macros

     *** Callback registration ***
     =============================================
    [..]
     The compilation flag USE_HAL_SMBUS_REGISTER_CALLBACKS when set to 1
     allows the user to configure dynamically the driver callbacks.
     Use Functions HAL_SMBUS_RegisterCallback() or HAL_SMBUS_RegisterXXXCallback()
     to register an interrupt callback.

     Function HAL_SMBUS_RegisterCallback() allows to register following callbacks:
       (+) MasterTxCpltCallback : callback for Master transmission end of transfer.
       (+) MasterRxCpltCallback : callback for Master reception end of transfer.
       (+) SlaveTxCpltCallback  : callback for Slave transmission end of transfer.
       (+) SlaveRxCpltCallback  : callback for Slave reception end of transfer.
       (+) ListenCpltCallback   : callback for end of listen mode.
       (+) ErrorCallback        : callback for error detection.
       (+) AbortCpltCallback    : callback for abort completion process.
       (+) MspInitCallback      : callback for Msp Init.
       (+) MspDeInitCallback    : callback for Msp DeInit.
     This function takes as parameters the HAL peripheral handle, the Callback ID
     and a pointer to the user callback function.
    [..]
     For specific callback AddrCallback use dedicated register callbacks : HAL_SMBUS_RegisterAddrCallback().
    [..]
     Use function HAL_SMBUS_UnRegisterCallback to reset a callback to the default
     weak function.
     HAL_SMBUS_UnRegisterCallback takes as parameters the HAL peripheral handle,
     and the Callback ID.
     This function allows to reset following callbacks:
       (+) MasterTxCpltCallback : callback for Master transmission end of transfer.
       (+) MasterRxCpltCallback : callback for Master reception end of transfer.
       (+) SlaveTxCpltCallback  : callback for Slave transmission end of transfer.
       (+) SlaveRxCpltCallback  : callback for Slave reception end of transfer.
       (+) ListenCpltCallback   : callback for end of listen mode.
       (+) ErrorCallback        : callback for error detection.
       (+) AbortCpltCallback    : callback for abort completion process.
       (+) MspInitCallback      : callback for Msp Init.
       (+) MspDeInitCallback    : callback for Msp DeInit.
    [..]
     For callback AddrCallback use dedicated register callbacks : HAL_SMBUS_UnRegisterAddrCallback().
    [..]
     By default, after the HAL_SMBUS_Init() and when the state is HAL_SMBUS_STATE_RESET
     all callbacks are set to the corresponding weak functions:
     examples HAL_SMBUS_MasterTxCpltCallback(), HAL_SMBUS_MasterRxCpltCallback().
     Exception done for MspInit and MspDeInit functions that are
     reset to the legacy weak functions in the HAL_SMBUS_Init()/ HAL_SMBUS_DeInit() only when
     these callbacks are null (not registered beforehand).
     If MspInit or MspDeInit are not null, the HAL_SMBUS_Init()/ HAL_SMBUS_DeInit()
     keep and use the user MspInit/MspDeInit callbacks (registered beforehand) whatever the state.
    [..]
     Callbacks can be registered/unregistered in HAL_SMBUS_STATE_READY state only.
     Exception done MspInit/MspDeInit functions that can be registered/unregistered
     in HAL_SMBUS_STATE_READY or HAL_SMBUS_STATE_RESET state,
     thus registered (user) MspInit/DeInit callbacks can be used during the Init/DeInit.
     Then, the user first registers the MspInit/MspDeInit user callbacks
     using HAL_SMBUS_RegisterCallback() before calling HAL_SMBUS_DeInit()
     or HAL_SMBUS_Init() function.
    [..]
     When the compilation flag USE_HAL_SMBUS_REGISTER_CALLBACKS is set to 0 or
     not defined, the callback registration feature is not available and all callbacks
     are set to the corresponding weak functions.

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

/** @defgroup SMBUS SMBUS
  * @brief SMBUS HAL module driver
  * @{
  */

#ifdef HAL_SMBUS_MODULE_ENABLED

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @addtogroup SMBUS_Private_Define
  * @{
  */
#define SMBUS_TIMEOUT_FLAG          35U         /*!< Timeout 35 ms             */
#define SMBUS_TIMEOUT_BUSY_FLAG     25U         /*!< Timeout 25 ms             */
#define SMBUS_NO_OPTION_FRAME       0xFFFF0000U /*!< XferOptions default value */

#define SMBUS_SENDPEC_MODE          I2C_CR1_PEC
#define SMBUS_GET_PEC(__HANDLE__)             (((__HANDLE__)->Instance->SR2 & I2C_SR2_PEC) >> 8)

/* Private define for @ref PreviousState usage */
#define SMBUS_STATE_MSK             ((uint32_t)((HAL_SMBUS_STATE_BUSY_TX | HAL_SMBUS_STATE_BUSY_RX) & (~(uint32_t)HAL_SMBUS_STATE_READY))) /*!< Mask State define, keep only RX and TX bits            */
#define SMBUS_STATE_NONE            ((uint32_t)(HAL_SMBUS_MODE_NONE))                                                                      /*!< Default Value                                          */
#define SMBUS_STATE_MASTER_BUSY_TX  ((uint32_t)((HAL_SMBUS_STATE_BUSY_TX & SMBUS_STATE_MSK) | HAL_SMBUS_MODE_MASTER))                      /*!< Master Busy TX, combinaison of State LSB and Mode enum */
#define SMBUS_STATE_MASTER_BUSY_RX  ((uint32_t)((HAL_SMBUS_STATE_BUSY_RX & SMBUS_STATE_MSK) | HAL_SMBUS_MODE_MASTER))                      /*!< Master Busy RX, combinaison of State LSB and Mode enum */
#define SMBUS_STATE_SLAVE_BUSY_TX   ((uint32_t)((HAL_SMBUS_STATE_BUSY_TX & SMBUS_STATE_MSK) | HAL_SMBUS_MODE_SLAVE))                       /*!< Slave Busy TX, combinaison of State LSB and Mode enum  */
#define SMBUS_STATE_SLAVE_BUSY_RX   ((uint32_t)((HAL_SMBUS_STATE_BUSY_RX & SMBUS_STATE_MSK) | HAL_SMBUS_MODE_SLAVE))                       /*!< Slave Busy RX, combinaison of State LSB and Mode enum  */

/**
  * @}
  */

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/

/** @addtogroup SMBUS_Private_Functions
  * @{
  */

static HAL_StatusTypeDef SMBUS_WaitOnFlagUntilTimeout(SMBUS_HandleTypeDef *hsmbus, uint32_t Flag, FlagStatus Status, uint32_t Timeout, uint32_t Tickstart);
static void SMBUS_ITError(SMBUS_HandleTypeDef *hsmbus);

/* Private functions for SMBUS transfer IRQ handler */
static HAL_StatusTypeDef SMBUS_MasterTransmit_TXE(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_MasterTransmit_BTF(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_MasterReceive_RXNE(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_MasterReceive_BTF(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_Master_SB(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_Master_ADD10(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_Master_ADDR(SMBUS_HandleTypeDef *hsmbus);

static HAL_StatusTypeDef SMBUS_SlaveTransmit_TXE(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_SlaveTransmit_BTF(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_SlaveReceive_RXNE(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_SlaveReceive_BTF(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_Slave_ADDR(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_Slave_STOPF(SMBUS_HandleTypeDef *hsmbus);
static HAL_StatusTypeDef SMBUS_Slave_AF(SMBUS_HandleTypeDef *hsmbus);
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup SMBUS_Exported_Functions SMBUS Exported Functions
  * @{
  */

/** @defgroup SMBUS_Exported_Functions_Group1 Initialization and de-initialization functions
 *  @brief    Initialization and Configuration functions
 *
@verbatim
 ===============================================================================
              ##### Initialization and de-initialization functions #####
 ===============================================================================
    [..]  This subsection provides a set of functions allowing to initialize and
          deinitialize the SMBUSx peripheral:

      (+) User must Implement HAL_SMBUS_MspInit() function in which he configures
          all related peripherals resources (CLOCK, GPIO, IT and NVIC).

      (+) Call the function HAL_SMBUS_Init() to configure the selected device with
          the selected configuration:
        (++) Communication Speed
        (++) Addressing mode
        (++) Own Address 1
        (++) Dual Addressing mode
        (++) Own Address 2
        (++) General call mode
        (++) Nostretch mode
        (++) Packet Error Check mode
        (++) Peripheral mode

      (+) Call the function HAL_SMBUS_DeInit() to restore the default configuration
          of the selected SMBUSx peripheral.

@endverbatim
  * @{
  */

/**
  * @brief  Initializes the SMBUS according to the specified parameters
  *         in the SMBUS_InitTypeDef and initialize the associated handle.
  * @param  hsmbus pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_Init(SMBUS_HandleTypeDef *hsmbus)
{
  uint32_t freqrange = 0U;
  uint32_t pclk1 = 0U;

  /* Check the SMBUS handle allocation */
  if (hsmbus == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_SMBUS_ALL_INSTANCE(hsmbus->Instance));
#if  defined(I2C_FLTR_ANOFF)
  assert_param(IS_SMBUS_ANALOG_FILTER(hsmbus->Init.AnalogFilter));
#endif
  assert_param(IS_SMBUS_CLOCK_SPEED(hsmbus->Init.ClockSpeed));
  assert_param(IS_SMBUS_OWN_ADDRESS1(hsmbus->Init.OwnAddress1));
  assert_param(IS_SMBUS_ADDRESSING_MODE(hsmbus->Init.AddressingMode));
  assert_param(IS_SMBUS_DUAL_ADDRESS(hsmbus->Init.DualAddressMode));
  assert_param(IS_SMBUS_OWN_ADDRESS2(hsmbus->Init.OwnAddress2));
  assert_param(IS_SMBUS_GENERAL_CALL(hsmbus->Init.GeneralCallMode));
  assert_param(IS_SMBUS_NO_STRETCH(hsmbus->Init.NoStretchMode));
  assert_param(IS_SMBUS_PEC(hsmbus->Init.PacketErrorCheckMode));
  assert_param(IS_SMBUS_PERIPHERAL_MODE(hsmbus->Init.PeripheralMode));

  if (hsmbus->State == HAL_SMBUS_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hsmbus->Lock = HAL_UNLOCKED;
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
    /* Init the SMBUS Callback settings */
    hsmbus->MasterTxCpltCallback = HAL_SMBUS_MasterTxCpltCallback; /* Legacy weak MasterTxCpltCallback */
    hsmbus->MasterRxCpltCallback = HAL_SMBUS_MasterRxCpltCallback; /* Legacy weak MasterRxCpltCallback */
    hsmbus->SlaveTxCpltCallback  = HAL_SMBUS_SlaveTxCpltCallback;  /* Legacy weak SlaveTxCpltCallback  */
    hsmbus->SlaveRxCpltCallback  = HAL_SMBUS_SlaveRxCpltCallback;  /* Legacy weak SlaveRxCpltCallback  */
    hsmbus->ListenCpltCallback   = HAL_SMBUS_ListenCpltCallback;   /* Legacy weak ListenCpltCallback   */
    hsmbus->ErrorCallback        = HAL_SMBUS_ErrorCallback;        /* Legacy weak ErrorCallback        */
    hsmbus->AbortCpltCallback    = HAL_SMBUS_AbortCpltCallback;    /* Legacy weak AbortCpltCallback    */
    hsmbus->AddrCallback         = HAL_SMBUS_AddrCallback;         /* Legacy weak AddrCallback         */

    if (hsmbus->MspInitCallback == NULL)
    {
      hsmbus->MspInitCallback = HAL_SMBUS_MspInit; /* Legacy weak MspInit  */
    }

    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    hsmbus->MspInitCallback(hsmbus);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC */
    HAL_SMBUS_MspInit(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
  }

  hsmbus->State = HAL_SMBUS_STATE_BUSY;

  /* Disable the selected SMBUS peripheral */
  __HAL_SMBUS_DISABLE(hsmbus);

  /* Get PCLK1 frequency */
  pclk1 = HAL_RCC_GetPCLK1Freq();

  /* Calculate frequency range */
  freqrange = SMBUS_FREQRANGE(pclk1);

  /*---------------------------- SMBUSx CR2 Configuration ----------------------*/
  /* Configure SMBUSx: Frequency range */
  MODIFY_REG(hsmbus->Instance->CR2, I2C_CR2_FREQ, freqrange);

  /*---------------------------- SMBUSx TRISE Configuration --------------------*/
  /* Configure SMBUSx: Rise Time */
  MODIFY_REG(hsmbus->Instance->TRISE, I2C_TRISE_TRISE, SMBUS_RISE_TIME(freqrange));

  /*---------------------------- SMBUSx CCR Configuration ----------------------*/
  /* Configure SMBUSx: Speed */
  MODIFY_REG(hsmbus->Instance->CCR, (I2C_CCR_FS | I2C_CCR_DUTY | I2C_CCR_CCR), SMBUS_SPEED_STANDARD(pclk1, hsmbus->Init.ClockSpeed));

  /*---------------------------- SMBUSx CR1 Configuration ----------------------*/
  /* Configure SMBUSx: Generalcall , PEC , Peripheral mode and  NoStretch mode */
  MODIFY_REG(hsmbus->Instance->CR1, (I2C_CR1_NOSTRETCH | I2C_CR1_ENGC | I2C_CR1_ENPEC | I2C_CR1_ENARP | I2C_CR1_SMBTYPE | I2C_CR1_SMBUS), (hsmbus->Init.NoStretchMode | hsmbus->Init.GeneralCallMode |  hsmbus->Init.PacketErrorCheckMode | hsmbus->Init.PeripheralMode));

  /*---------------------------- SMBUSx OAR1 Configuration ---------------------*/
  /* Configure SMBUSx: Own Address1 and addressing mode */
  MODIFY_REG(hsmbus->Instance->OAR1, (I2C_OAR1_ADDMODE | I2C_OAR1_ADD8_9 | I2C_OAR1_ADD1_7 | I2C_OAR1_ADD0), (hsmbus->Init.AddressingMode | hsmbus->Init.OwnAddress1));

  /*---------------------------- SMBUSx OAR2 Configuration ---------------------*/
  /* Configure SMBUSx: Dual mode and Own Address2 */
  MODIFY_REG(hsmbus->Instance->OAR2, (I2C_OAR2_ENDUAL | I2C_OAR2_ADD2), (hsmbus->Init.DualAddressMode | hsmbus->Init.OwnAddress2));
#if  defined(I2C_FLTR_ANOFF)
  /*---------------------------- SMBUSx FLTR Configuration ------------------------*/
  /* Configure SMBUSx: Analog noise filter */
  SET_BIT(hsmbus->Instance->FLTR, hsmbus->Init.AnalogFilter);
#endif

  /* Enable the selected SMBUS peripheral */
  __HAL_SMBUS_ENABLE(hsmbus);

  hsmbus->ErrorCode = HAL_SMBUS_ERROR_NONE;
  hsmbus->State = HAL_SMBUS_STATE_READY;
  hsmbus->PreviousState = SMBUS_STATE_NONE;
  hsmbus->Mode = HAL_SMBUS_MODE_NONE;
  hsmbus->XferPEC = 0x00;

  return HAL_OK;
}

/**
  * @brief  DeInitializes the SMBUS peripheral.
  * @param  hsmbus pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_DeInit(SMBUS_HandleTypeDef *hsmbus)
{
  /* Check the SMBUS handle allocation */
  if (hsmbus == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_SMBUS_ALL_INSTANCE(hsmbus->Instance));

  hsmbus->State = HAL_SMBUS_STATE_BUSY;

  /* Disable the SMBUS Peripheral Clock */
  __HAL_SMBUS_DISABLE(hsmbus);

#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
  if (hsmbus->MspDeInitCallback == NULL)
  {
    hsmbus->MspDeInitCallback = HAL_SMBUS_MspDeInit; /* Legacy weak MspDeInit  */
  }

  /* DeInit the low level hardware: GPIO, CLOCK, NVIC */
  hsmbus->MspDeInitCallback(hsmbus);
#else
  /* DeInit the low level hardware: GPIO, CLOCK, NVIC */
  HAL_SMBUS_MspDeInit(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */

  hsmbus->ErrorCode     = HAL_SMBUS_ERROR_NONE;
  hsmbus->State         = HAL_SMBUS_STATE_RESET;
  hsmbus->PreviousState = SMBUS_STATE_NONE;
  hsmbus->Mode          = HAL_SMBUS_MODE_NONE;

  /* Release Lock */
  __HAL_UNLOCK(hsmbus);

  return HAL_OK;
}

/**
  * @brief  Initialize the SMBUS MSP.
  * @param  hsmbus pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS
  * @retval None
  */
__weak void HAL_SMBUS_MspInit(SMBUS_HandleTypeDef *hsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_SMBUS_MspInit could be implemented in the user file
   */
}

/**
  * @brief  DeInitialize the SMBUS MSP.
  * @param  hsmbus pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS
  * @retval None
  */
__weak void HAL_SMBUS_MspDeInit(SMBUS_HandleTypeDef *hsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_SMBUS_MspDeInit could be implemented in the user file
   */
}

#if  defined(I2C_FLTR_ANOFF)&&defined(I2C_FLTR_DNF)
/**
  * @brief  Configures SMBUS Analog noise filter.
  * @param  hsmbus pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUSx peripheral.
  * @param  AnalogFilter new state of the Analog filter.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_ConfigAnalogFilter(SMBUS_HandleTypeDef *hsmbus, uint32_t AnalogFilter)
{
  /* Check the parameters */
  assert_param(IS_SMBUS_ALL_INSTANCE(hsmbus->Instance));
  assert_param(IS_SMBUS_ANALOG_FILTER(AnalogFilter));

  if (hsmbus->State == HAL_SMBUS_STATE_READY)
  {
    hsmbus->State = HAL_SMBUS_STATE_BUSY;

    /* Disable the selected SMBUS peripheral */
    __HAL_SMBUS_DISABLE(hsmbus);

    /* Reset SMBUSx ANOFF bit */
    hsmbus->Instance->FLTR &= ~(I2C_FLTR_ANOFF);

    /* Disable the analog filter */
    hsmbus->Instance->FLTR |= AnalogFilter;

    __HAL_SMBUS_ENABLE(hsmbus);

    hsmbus->State = HAL_SMBUS_STATE_READY;

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Configures SMBUS Digital noise filter.
  * @param  hsmbus pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUSx peripheral.
  * @param  DigitalFilter Coefficient of digital noise filter between 0x00 and 0x0F.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_ConfigDigitalFilter(SMBUS_HandleTypeDef *hsmbus, uint32_t DigitalFilter)
{
  uint16_t tmpreg = 0;

  /* Check the parameters */
  assert_param(IS_SMBUS_ALL_INSTANCE(hsmbus->Instance));
  assert_param(IS_SMBUS_DIGITAL_FILTER(DigitalFilter));

  if (hsmbus->State == HAL_SMBUS_STATE_READY)
  {
    hsmbus->State = HAL_SMBUS_STATE_BUSY;

    /* Disable the selected SMBUS peripheral */
    __HAL_SMBUS_DISABLE(hsmbus);

    /* Get the old register value */
    tmpreg = hsmbus->Instance->FLTR;

    /* Reset SMBUSx DNF bit [3:0] */
    tmpreg &= ~(I2C_FLTR_DNF);

    /* Set SMBUSx DNF coefficient */
    tmpreg |= DigitalFilter;

    /* Store the new register value */
    hsmbus->Instance->FLTR = tmpreg;

    __HAL_SMBUS_ENABLE(hsmbus);

    hsmbus->State = HAL_SMBUS_STATE_READY;

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}
#endif
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User SMBUS Callback
  *         To be used instead of the weak predefined callback
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @param  CallbackID ID of the callback to be registered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_SMBUS_MASTER_TX_COMPLETE_CB_ID Master Tx Transfer completed callback ID
  *          @arg @ref HAL_SMBUS_MASTER_RX_COMPLETE_CB_ID Master Rx Transfer completed callback ID
  *          @arg @ref HAL_SMBUS_SLAVE_TX_COMPLETE_CB_ID Slave Tx Transfer completed callback ID
  *          @arg @ref HAL_SMBUS_SLAVE_RX_COMPLETE_CB_ID Slave Rx Transfer completed callback ID
  *          @arg @ref HAL_SMBUS_LISTEN_COMPLETE_CB_ID Listen Complete callback ID
  *          @arg @ref HAL_SMBUS_ERROR_CB_ID Error callback ID
  *          @arg @ref HAL_SMBUS_ABORT_CB_ID Abort callback ID
  *          @arg @ref HAL_SMBUS_MSPINIT_CB_ID MspInit callback ID
  *          @arg @ref HAL_SMBUS_MSPDEINIT_CB_ID MspDeInit callback ID
  * @param  pCallback pointer to the Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_RegisterCallback(SMBUS_HandleTypeDef *hsmbus, HAL_SMBUS_CallbackIDTypeDef CallbackID, pSMBUS_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_INVALID_CALLBACK;

    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hsmbus);

  if (HAL_SMBUS_STATE_READY == hsmbus->State)
  {
    switch (CallbackID)
    {
      case HAL_SMBUS_MASTER_TX_COMPLETE_CB_ID :
        hsmbus->MasterTxCpltCallback = pCallback;
        break;

      case HAL_SMBUS_MASTER_RX_COMPLETE_CB_ID :
        hsmbus->MasterRxCpltCallback = pCallback;
        break;

      case HAL_SMBUS_SLAVE_TX_COMPLETE_CB_ID :
        hsmbus->SlaveTxCpltCallback = pCallback;
        break;

      case HAL_SMBUS_SLAVE_RX_COMPLETE_CB_ID :
        hsmbus->SlaveRxCpltCallback = pCallback;
        break;

      case HAL_SMBUS_LISTEN_COMPLETE_CB_ID :
        hsmbus->ListenCpltCallback = pCallback;
        break;

      case HAL_SMBUS_ERROR_CB_ID :
        hsmbus->ErrorCallback = pCallback;
        break;

      case HAL_SMBUS_ABORT_CB_ID :
        hsmbus->AbortCpltCallback = pCallback;
        break;

      case HAL_SMBUS_MSPINIT_CB_ID :
        hsmbus->MspInitCallback = pCallback;
        break;

      case HAL_SMBUS_MSPDEINIT_CB_ID :
        hsmbus->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hsmbus->ErrorCode |= HAL_SMBUS_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_SMBUS_STATE_RESET == hsmbus->State)
  {
    switch (CallbackID)
    {
      case HAL_SMBUS_MSPINIT_CB_ID :
        hsmbus->MspInitCallback = pCallback;
        break;

      case HAL_SMBUS_MSPDEINIT_CB_ID :
        hsmbus->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hsmbus->ErrorCode |= HAL_SMBUS_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hsmbus);
  return status;
}

/**
  * @brief  Unregister an SMBUS Callback
  *         SMBUS callback is redirected to the weak predefined callback
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @param  CallbackID ID of the callback to be unregistered
  *         This parameter can be one of the following values:
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_SMBUS_MASTER_TX_COMPLETE_CB_ID Master Tx Transfer completed callback ID
  *          @arg @ref HAL_SMBUS_MASTER_RX_COMPLETE_CB_ID Master Rx Transfer completed callback ID
  *          @arg @ref HAL_SMBUS_SLAVE_TX_COMPLETE_CB_ID Slave Tx Transfer completed callback ID
  *          @arg @ref HAL_SMBUS_SLAVE_RX_COMPLETE_CB_ID Slave Rx Transfer completed callback ID
  *          @arg @ref HAL_SMBUS_LISTEN_COMPLETE_CB_ID Listen Complete callback ID
  *          @arg @ref HAL_SMBUS_ERROR_CB_ID Error callback ID
  *          @arg @ref HAL_SMBUS_ABORT_CB_ID Abort callback ID
  *          @arg @ref HAL_SMBUS_MSPINIT_CB_ID MspInit callback ID
  *          @arg @ref HAL_SMBUS_MSPDEINIT_CB_ID MspDeInit callback ID
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_UnRegisterCallback(SMBUS_HandleTypeDef *hsmbus, HAL_SMBUS_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hsmbus);

  if (HAL_SMBUS_STATE_READY == hsmbus->State)
  {
    switch (CallbackID)
    {
      case HAL_SMBUS_MASTER_TX_COMPLETE_CB_ID :
        hsmbus->MasterTxCpltCallback = HAL_SMBUS_MasterTxCpltCallback; /* Legacy weak MasterTxCpltCallback */
        break;

      case HAL_SMBUS_MASTER_RX_COMPLETE_CB_ID :
        hsmbus->MasterRxCpltCallback = HAL_SMBUS_MasterRxCpltCallback; /* Legacy weak MasterRxCpltCallback */
        break;

      case HAL_SMBUS_SLAVE_TX_COMPLETE_CB_ID :
        hsmbus->SlaveTxCpltCallback = HAL_SMBUS_SlaveTxCpltCallback;   /* Legacy weak SlaveTxCpltCallback  */
        break;

      case HAL_SMBUS_SLAVE_RX_COMPLETE_CB_ID :
        hsmbus->SlaveRxCpltCallback = HAL_SMBUS_SlaveRxCpltCallback;   /* Legacy weak SlaveRxCpltCallback  */
        break;

      case HAL_SMBUS_LISTEN_COMPLETE_CB_ID :
        hsmbus->ListenCpltCallback = HAL_SMBUS_ListenCpltCallback;     /* Legacy weak ListenCpltCallback   */
        break;

      case HAL_SMBUS_ERROR_CB_ID :
        hsmbus->ErrorCallback = HAL_SMBUS_ErrorCallback;               /* Legacy weak ErrorCallback        */
        break;

      case HAL_SMBUS_ABORT_CB_ID :
        hsmbus->AbortCpltCallback = HAL_SMBUS_AbortCpltCallback;       /* Legacy weak AbortCpltCallback    */
        break;

      case HAL_SMBUS_MSPINIT_CB_ID :
        hsmbus->MspInitCallback = HAL_SMBUS_MspInit;                   /* Legacy weak MspInit              */
        break;

      case HAL_SMBUS_MSPDEINIT_CB_ID :
        hsmbus->MspDeInitCallback = HAL_SMBUS_MspDeInit;               /* Legacy weak MspDeInit            */
        break;

      default :
        /* Update the error code */
        hsmbus->ErrorCode |= HAL_SMBUS_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_SMBUS_STATE_RESET == hsmbus->State)
  {
    switch (CallbackID)
    {
      case HAL_SMBUS_MSPINIT_CB_ID :
        hsmbus->MspInitCallback = HAL_SMBUS_MspInit;                   /* Legacy weak MspInit              */
        break;

      case HAL_SMBUS_MSPDEINIT_CB_ID :
        hsmbus->MspDeInitCallback = HAL_SMBUS_MspDeInit;               /* Legacy weak MspDeInit            */
        break;

      default :
        /* Update the error code */
        hsmbus->ErrorCode |= HAL_SMBUS_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hsmbus);
  return status;
}

/**
  * @brief  Register the Slave Address Match SMBUS Callback
  *         To be used instead of the weak HAL_SMBUS_AddrCallback() predefined callback
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @param  pCallback pointer to the Address Match Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_RegisterAddrCallback(SMBUS_HandleTypeDef *hsmbus, pSMBUS_AddrCallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    return HAL_ERROR;
  }
  /* Process locked */
  __HAL_LOCK(hsmbus);

  if (HAL_SMBUS_STATE_READY == hsmbus->State)
  {
    hsmbus->AddrCallback = pCallback;
  }
  else
  {
    /* Update the error code */
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hsmbus);
  return status;
}

/**
  * @brief  UnRegister the Slave Address Match SMBUS Callback
  *         Info Ready SMBUS Callback is redirected to the weak HAL_SMBUS_AddrCallback() predefined callback
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_UnRegisterAddrCallback(SMBUS_HandleTypeDef *hsmbus)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hsmbus);

  if (HAL_SMBUS_STATE_READY == hsmbus->State)
  {
    hsmbus->AddrCallback = HAL_SMBUS_AddrCallback; /* Legacy weak AddrCallback  */
  }
  else
  {
    /* Update the error code */
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hsmbus);
  return status;
}

#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup SMBUS_Exported_Functions_Group2 Input and Output operation functions
 *  @brief    Data transfers functions
 *
@verbatim
 ===============================================================================
                      ##### IO operation functions #####
 ===============================================================================
    [..]
    This subsection provides a set of functions allowing to manage the SMBUS data
    transfers.

    (#) Blocking mode function to check if device is ready for usage is :
        (++) HAL_SMBUS_IsDeviceReady()

    (#) There is only one mode of transfer:
       (++) Non Blocking mode : The communication is performed using Interrupts.
            These functions return the status of the transfer startup.
            The end of the data processing will be indicated through the
            dedicated SMBUS IRQ when using Interrupt mode.

    (#) Non Blocking mode functions with Interrupt are :
        (++) HAL_SMBUS_Master_Transmit_IT()
        (++) HAL_SMBUS_Master_Receive_IT()
        (++) HAL_SMBUS_Master_Abort_IT()
        (++) HAL_SMBUS_Slave_Transmit_IT()
        (++) HAL_SMBUS_Slave_Receive_IT()
        (++) HAL_SMBUS_EnableAlert_IT()
        (++) HAL_SMBUS_DisableAlert_IT()

    (#) A set of Transfer Complete Callbacks are provided in No_Blocking mode:
        (++) HAL_SMBUS_MasterTxCpltCallback()
        (++) HAL_SMBUS_MasterRxCpltCallback()
        (++) HAL_SMBUS_SlaveTxCpltCallback()
        (++) HAL_SMBUS_SlaveRxCpltCallback()
        (++) HAL_SMBUS_AddrCallback()
        (++) HAL_SMBUS_ListenCpltCallback()
        (++) HAL_SMBUS_ErrorCallback()
        (++) HAL_SMBUS_AbortCpltCallback()

@endverbatim
  * @{
  */

/**
  * @brief  Transmits in master mode an amount of data in blocking mode.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS.
  * @param  DevAddress Target device address The device 7 bits address value
  *         in datasheet must be shifted to the left before calling the interface
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @param  XferOptions Options of Transfer
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_Master_Transmit_IT(SMBUS_HandleTypeDef *hsmbus, uint16_t DevAddress, uint8_t *pData, uint16_t Size, uint32_t XferOptions)
{
  uint32_t count      = 0x00U;

  /* Check the parameters */
  assert_param(IS_SMBUS_TRANSFER_OPTIONS_REQUEST(XferOptions));

  if (hsmbus->State == HAL_SMBUS_STATE_READY)
  {
    /* Check Busy Flag only if FIRST call of Master interface */
    if ((XferOptions == SMBUS_FIRST_AND_LAST_FRAME_NO_PEC) || (XferOptions == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || (XferOptions == SMBUS_FIRST_FRAME))
    {
      /* Wait until BUSY flag is reset */
      count = SMBUS_TIMEOUT_BUSY_FLAG * (SystemCoreClock / 25U / 1000U);
      do
      {
        if (count-- == 0U)
        {
          hsmbus->PreviousState = SMBUS_STATE_NONE;
          hsmbus->State = HAL_SMBUS_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hsmbus);

          return HAL_TIMEOUT;
        }
      }
      while (__HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_BUSY) != RESET);
    }

    /* Process Locked */
    __HAL_LOCK(hsmbus);

    /* Check if the SMBUS is already enabled */
    if ((hsmbus->Instance->CR1 & I2C_CR1_PE) != I2C_CR1_PE)
    {
      /* Enable SMBUS peripheral */
      __HAL_SMBUS_ENABLE(hsmbus);
    }

    /* Disable Pos */
    CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_POS);

    hsmbus->State     = HAL_SMBUS_STATE_BUSY_TX;
    hsmbus->ErrorCode = HAL_SMBUS_ERROR_NONE;
    hsmbus->Mode      = HAL_SMBUS_MODE_MASTER;

    /* Prepare transfer parameters */
    hsmbus->pBuffPtr    = pData;
    hsmbus->XferCount   = Size;
    hsmbus->XferOptions = XferOptions;
    hsmbus->XferSize    = hsmbus->XferCount;
    hsmbus->Devaddress  = DevAddress;

    /* Generate Start */
    SET_BIT(hsmbus->Instance->CR1, I2C_CR1_START);

    /* Process Unlocked */
    __HAL_UNLOCK(hsmbus);

    /* Note : The SMBUS interrupts must be enabled after unlocking current process
    to avoid the risk of hsmbus interrupt handle execution before current
    process unlock */

    /* Enable EVT, BUF and ERR interrupt */
    __HAL_SMBUS_ENABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}
/**
  * @brief  Receive in master/host SMBUS mode an amount of data in non blocking mode with Interrupt.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS.
  * @param  DevAddress Target device address The device 7 bits address value
  *         in datasheet must be shifted to the left before calling the interface
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @param  XferOptions Options of Transfer, value of @ref SMBUS_XferOptions_definition
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_Master_Receive_IT(SMBUS_HandleTypeDef *hsmbus, uint16_t DevAddress, uint8_t *pData, uint16_t Size, uint32_t XferOptions)
{
  __IO uint32_t count = 0U;

  /* Check the parameters */
  assert_param(IS_SMBUS_TRANSFER_OPTIONS_REQUEST(XferOptions));

  if (hsmbus->State == HAL_SMBUS_STATE_READY)
  {
    /* Check Busy Flag only if FIRST call of Master interface */
    if ((XferOptions == SMBUS_FIRST_AND_LAST_FRAME_NO_PEC) || (XferOptions == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || (XferOptions == SMBUS_FIRST_FRAME))
    {
      /* Wait until BUSY flag is reset */
      count = SMBUS_TIMEOUT_BUSY_FLAG * (SystemCoreClock / 25U / 1000U);
      do
      {
        if (count-- == 0U)
        {
          hsmbus->PreviousState = SMBUS_STATE_NONE;
          hsmbus->State = HAL_SMBUS_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hsmbus);

          return HAL_TIMEOUT;
        }
      }
      while (__HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_BUSY) != RESET);
    }

    /* Process Locked */
    __HAL_LOCK(hsmbus);

    /* Check if the SMBUS is already enabled */
    if ((hsmbus->Instance->CR1 & I2C_CR1_PE) != I2C_CR1_PE)
    {
      /* Enable SMBUS peripheral */
      __HAL_SMBUS_ENABLE(hsmbus);
    }

    /* Disable Pos */
    CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_POS);

    hsmbus->State     = HAL_SMBUS_STATE_BUSY_RX;
    hsmbus->ErrorCode = HAL_SMBUS_ERROR_NONE;
    hsmbus->Mode      = HAL_SMBUS_MODE_MASTER;

    /* Prepare transfer parameters */
    hsmbus->pBuffPtr    = pData;
    hsmbus->XferCount   = Size;
    hsmbus->XferOptions = XferOptions;
    hsmbus->XferSize    = hsmbus->XferCount;
    hsmbus->Devaddress  = DevAddress;

    if ((hsmbus->PreviousState == SMBUS_STATE_MASTER_BUSY_TX) || (hsmbus->PreviousState == SMBUS_STATE_NONE))
    {
      /* Generate Start condition if first transfer */
      if ((XferOptions == SMBUS_NEXT_FRAME)  || (XferOptions == SMBUS_FIRST_AND_LAST_FRAME_NO_PEC) || (XferOptions == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || (XferOptions == SMBUS_FIRST_FRAME)  || (XferOptions == SMBUS_NO_OPTION_FRAME))
      {
        /* Enable Acknowledge */
        SET_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

        /* Generate Start */
        SET_BIT(hsmbus->Instance->CR1, I2C_CR1_START);
      }

      if ((XferOptions == SMBUS_LAST_FRAME_NO_PEC) || (XferOptions == SMBUS_LAST_FRAME_WITH_PEC))
      {
        if (hsmbus->PreviousState == SMBUS_STATE_NONE)
        {
          /* Enable Acknowledge */
          SET_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);
        }

        if (hsmbus->PreviousState == SMBUS_STATE_MASTER_BUSY_TX)
        {
          /* Enable Acknowledge */
          SET_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

          /* Generate Start */
          SET_BIT(hsmbus->Instance->CR1, I2C_CR1_START);
        }
      }
    }



    /* Process Unlocked */
    __HAL_UNLOCK(hsmbus);

    /* Note : The SMBUS interrupts must be enabled after unlocking current process
    to avoid the risk of SMBUS interrupt handle execution before current
    process unlock */

    /* Enable EVT, BUF and ERR interrupt */
    __HAL_SMBUS_ENABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Abort a master/host SMBUS process communication with Interrupt.
  * @note   This abort can be called only if state is ready
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS.
  * @param  DevAddress Target device address The device 7 bits address value
  *         in datasheet must be shifted to the left before calling the interface
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_Master_Abort_IT(SMBUS_HandleTypeDef *hsmbus, uint16_t DevAddress)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(DevAddress);
  if (hsmbus->Init.PeripheralMode == SMBUS_PERIPHERAL_MODE_SMBUS_HOST)
  {
    /* Process Locked */
    __HAL_LOCK(hsmbus);

    hsmbus->ErrorCode = HAL_SMBUS_ERROR_NONE;

    hsmbus->PreviousState = SMBUS_STATE_NONE;
    hsmbus->State = HAL_SMBUS_STATE_ABORT;


    /* Disable Acknowledge */
    CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

    /* Generate Stop */
    SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);

    hsmbus->XferCount = 0U;

    /* Disable EVT, BUF and ERR interrupt */
    __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

    /* Process Unlocked */
    __HAL_UNLOCK(hsmbus);

    /* Call the corresponding callback to inform upper layer of End of Transfer */
    SMBUS_ITError(hsmbus);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}


/**
  * @brief  Transmit in slave/device SMBUS mode an amount of data in non blocking mode with Interrupt.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @param  XferOptions Options of Transfer, value of @ref SMBUS_XferOptions_definition
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_Slave_Transmit_IT(SMBUS_HandleTypeDef *hsmbus, uint8_t *pData, uint16_t Size, uint32_t XferOptions)
{
  /* Check the parameters */
  assert_param(IS_SMBUS_TRANSFER_OPTIONS_REQUEST(XferOptions));

  if (hsmbus->State == HAL_SMBUS_STATE_LISTEN)
  {
    if ((pData == NULL) || (Size == 0U))
    {
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hsmbus);

    /* Check if the SMBUS is already enabled */
    if ((hsmbus->Instance->CR1 & I2C_CR1_PE) != I2C_CR1_PE)
    {
      /* Enable SMBUS peripheral */
      __HAL_SMBUS_ENABLE(hsmbus);
    }

    /* Disable Pos */
    CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_POS);

    hsmbus->State     = HAL_SMBUS_STATE_BUSY_TX_LISTEN;
    hsmbus->ErrorCode = HAL_SMBUS_ERROR_NONE;
    hsmbus->Mode      = HAL_SMBUS_MODE_SLAVE;

    /* Prepare transfer parameters */
    hsmbus->pBuffPtr    = pData;
    hsmbus->XferCount   = Size;
    hsmbus->XferOptions = XferOptions;
    hsmbus->XferSize    = hsmbus->XferCount;

    /* Clear ADDR flag after prepare the transfer parameters */
    /* This action will generate an acknowledge to the HOST */
    __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);

    /* Process Unlocked */
    __HAL_UNLOCK(hsmbus);

    /* Note : The SMBUS interrupts must be enabled after unlocking current process
              to avoid the risk of SMBUS interrupt handle execution before current
              process unlock */

    /* Enable EVT, BUF and ERR interrupt */
    __HAL_SMBUS_ENABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Enable the Address listen mode with Interrupt.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @param  XferOptions Options of Transfer, value of @ref SMBUS_XferOptions_definition
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_Slave_Receive_IT(SMBUS_HandleTypeDef *hsmbus, uint8_t *pData, uint16_t Size, uint32_t XferOptions)
{
  /* Check the parameters */
  assert_param(IS_SMBUS_TRANSFER_OPTIONS_REQUEST(XferOptions));

  if (hsmbus->State == HAL_SMBUS_STATE_LISTEN)
  {
    if ((pData == NULL) || (Size == 0U))
    {
      return  HAL_ERROR;
    }

    /* Process Locked */
    __HAL_LOCK(hsmbus);

    /* Check if the SMBUS is already enabled */
    if ((hsmbus->Instance->CR1 & I2C_CR1_PE) != I2C_CR1_PE)
    {
      /* Enable SMBUS peripheral */
      __HAL_SMBUS_ENABLE(hsmbus);
    }

    /* Disable Pos */
    CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_POS);

    hsmbus->State     = HAL_SMBUS_STATE_BUSY_RX_LISTEN;
    hsmbus->ErrorCode = HAL_SMBUS_ERROR_NONE;
    hsmbus->Mode      = HAL_SMBUS_MODE_SLAVE;



    /* Prepare transfer parameters */
    hsmbus->pBuffPtr = pData;
    hsmbus->XferCount = Size;
    hsmbus->XferOptions = XferOptions;
    hsmbus->XferSize    = hsmbus->XferCount;

    __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);

    /* Process Unlocked */
    __HAL_UNLOCK(hsmbus);

    /* Note : The SMBUS interrupts must be enabled after unlocking current process
              to avoid the risk of SMBUS interrupt handle execution before current
              process unlock */

    /* Enable EVT, BUF and ERR interrupt */
    __HAL_SMBUS_ENABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}


/**
  * @brief  Enable the Address listen mode with Interrupt.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_EnableListen_IT(SMBUS_HandleTypeDef *hsmbus)
{
  if (hsmbus->State == HAL_SMBUS_STATE_READY)
  {
    hsmbus->State = HAL_SMBUS_STATE_LISTEN;

    /* Check if the SMBUS is already enabled */
    if ((hsmbus->Instance->CR1 & I2C_CR1_PE) != I2C_CR1_PE)
    {
      /* Enable SMBUS peripheral */
      __HAL_SMBUS_ENABLE(hsmbus);
    }

    /* Enable Address Acknowledge */
    SET_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

    /* Enable EVT and ERR interrupt */
    __HAL_SMBUS_ENABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_ERR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Disable the Address listen mode with Interrupt.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_DisableListen_IT(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of tmp to prevent undefined behavior of volatile usage */
  uint32_t tmp;

  /* Disable Address listen mode only if a transfer is not ongoing */
  if (hsmbus->State == HAL_SMBUS_STATE_LISTEN)
  {
    tmp = (uint32_t)(hsmbus->State) & SMBUS_STATE_MSK;
    hsmbus->PreviousState = tmp | (uint32_t)(hsmbus->Mode);
    hsmbus->State = HAL_SMBUS_STATE_READY;
    hsmbus->Mode = HAL_SMBUS_MODE_NONE;

    /* Disable Address Acknowledge */
    CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

    /* Disable EVT and ERR interrupt */
    __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_ERR);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Enable the SMBUS alert mode with Interrupt.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUSx peripheral.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_EnableAlert_IT(SMBUS_HandleTypeDef *hsmbus)
{
  /* Enable SMBus alert */
  SET_BIT(hsmbus->Instance->CR1, I2C_CR1_ALERT);

  /* Clear ALERT flag */
  __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_SMBALERT);

  /* Enable Alert Interrupt */
  __HAL_SMBUS_ENABLE_IT(hsmbus, SMBUS_IT_ERR);

  return HAL_OK;
}
/**
  * @brief  Disable the SMBUS alert mode with Interrupt.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUSx peripheral.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_DisableAlert_IT(SMBUS_HandleTypeDef *hsmbus)
{
  /* Disable SMBus alert */
  CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ALERT);

  /* Disable Alert Interrupt */
  __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_ERR);

  return HAL_OK;
}


/**
  * @brief  Check if target device is ready for communication.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for the specified SMBUS.
  * @param  DevAddress Target device address The device 7 bits address value
  *         in datasheet must be shifted to the left before calling the interface
  * @param  Trials Number of trials
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SMBUS_IsDeviceReady(SMBUS_HandleTypeDef *hsmbus, uint16_t DevAddress, uint32_t Trials, uint32_t Timeout)
{
  uint32_t tickstart = 0U, tmp1 = 0U, tmp2 = 0U, tmp3 = 0U, SMBUS_Trials = 1U;

  /* Get tick */
  tickstart = HAL_GetTick();

  if (hsmbus->State == HAL_SMBUS_STATE_READY)
  {
    /* Wait until BUSY flag is reset */
    if (SMBUS_WaitOnFlagUntilTimeout(hsmbus, SMBUS_FLAG_BUSY, SET, SMBUS_TIMEOUT_BUSY_FLAG, tickstart) != HAL_OK)
    {
      return HAL_BUSY;
    }

    /* Process Locked */
    __HAL_LOCK(hsmbus);

    /* Check if the SMBUS is already enabled */
    if ((hsmbus->Instance->CR1 & I2C_CR1_PE) != I2C_CR1_PE)
    {
      /* Enable SMBUS peripheral */
      __HAL_SMBUS_ENABLE(hsmbus);
    }

    /* Disable Pos */
    CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_POS);

    hsmbus->State = HAL_SMBUS_STATE_BUSY;
    hsmbus->ErrorCode = HAL_SMBUS_ERROR_NONE;
    hsmbus->XferOptions = SMBUS_NO_OPTION_FRAME;

    do
    {
      /* Generate Start */
      SET_BIT(hsmbus->Instance->CR1, I2C_CR1_START);

      /* Wait until SB flag is set */
      if (SMBUS_WaitOnFlagUntilTimeout(hsmbus, SMBUS_FLAG_SB, RESET, Timeout, tickstart) != HAL_OK)
      {
        return HAL_TIMEOUT;
      }

      /* Send slave address */
      hsmbus->Instance->DR = SMBUS_7BIT_ADD_WRITE(DevAddress);

      /* Wait until ADDR or AF flag are set */
      /* Get tick */
      tickstart = HAL_GetTick();

      tmp1 = __HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_ADDR);
      tmp2 = __HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_AF);
      tmp3 = hsmbus->State;
      while ((tmp1 == RESET) && (tmp2 == RESET) && (tmp3 != HAL_SMBUS_STATE_TIMEOUT))
      {
        if ((Timeout == 0U) || ((HAL_GetTick() - tickstart) > Timeout))
        {
          hsmbus->State = HAL_SMBUS_STATE_TIMEOUT;
        }
        tmp1 = __HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_ADDR);
        tmp2 = __HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_AF);
        tmp3 = hsmbus->State;
      }

      hsmbus->State = HAL_SMBUS_STATE_READY;

      /* Check if the ADDR flag has been set */
      if (__HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_ADDR) == SET)
      {
        /* Generate Stop */
        SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);

        /* Clear ADDR Flag */
        __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);

        /* Wait until BUSY flag is reset */
        if (SMBUS_WaitOnFlagUntilTimeout(hsmbus, SMBUS_FLAG_BUSY, SET, SMBUS_TIMEOUT_BUSY_FLAG, tickstart) != HAL_OK)
        {
          return HAL_TIMEOUT;
        }

        hsmbus->State = HAL_SMBUS_STATE_READY;

        /* Process Unlocked */
        __HAL_UNLOCK(hsmbus);

        return HAL_OK;
      }
      else
      {
        /* Generate Stop */
        SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);

        /* Clear AF Flag */
        __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_AF);

        /* Wait until BUSY flag is reset */
        if (SMBUS_WaitOnFlagUntilTimeout(hsmbus, SMBUS_FLAG_BUSY, SET, SMBUS_TIMEOUT_BUSY_FLAG, tickstart) != HAL_OK)
        {
          return HAL_TIMEOUT;
        }
      }
    }
    while (SMBUS_Trials++ < Trials);

    hsmbus->State = HAL_SMBUS_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hsmbus);

    return HAL_ERROR;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  This function handles SMBUS event interrupt request.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval None
  */
void HAL_SMBUS_EV_IRQHandler(SMBUS_HandleTypeDef *hsmbus)
{
  uint32_t sr2itflags   = READ_REG(hsmbus->Instance->SR2);
  uint32_t sr1itflags   = READ_REG(hsmbus->Instance->SR1);
  uint32_t itsources    = READ_REG(hsmbus->Instance->CR2);

  uint32_t CurrentMode  = hsmbus->Mode;

  /* Master mode selected */
  if (CurrentMode == HAL_SMBUS_MODE_MASTER)
  {
    /* SB Set ----------------------------------------------------------------*/
    if (((sr1itflags & SMBUS_FLAG_SB) != RESET) && ((itsources & SMBUS_IT_EVT) != RESET))
    {
      SMBUS_Master_SB(hsmbus);
    }
    /* ADD10 Set -------------------------------------------------------------*/
    else if (((sr1itflags & SMBUS_FLAG_ADD10) != RESET) && ((itsources & SMBUS_IT_EVT) != RESET))
    {
      SMBUS_Master_ADD10(hsmbus);
    }
    /* ADDR Set --------------------------------------------------------------*/
    else if (((sr1itflags & SMBUS_FLAG_ADDR) != RESET) && ((itsources & SMBUS_IT_EVT) != RESET))
    {
      SMBUS_Master_ADDR(hsmbus);
    }
    /* SMBUS in mode Transmitter -----------------------------------------------*/
    if ((sr2itflags & SMBUS_FLAG_TRA) != RESET)
    {
      /* TXE set and BTF reset -----------------------------------------------*/
      if (((sr1itflags & SMBUS_FLAG_TXE) != RESET) && ((itsources & SMBUS_IT_BUF) != RESET) && ((sr1itflags & SMBUS_FLAG_BTF) == RESET))
      {
        SMBUS_MasterTransmit_TXE(hsmbus);
      }
      /* BTF set -------------------------------------------------------------*/
      else if (((sr1itflags & SMBUS_FLAG_BTF) != RESET) && ((itsources & SMBUS_IT_EVT) != RESET))
      {
        SMBUS_MasterTransmit_BTF(hsmbus);
      }
    }
    /* SMBUS in mode Receiver --------------------------------------------------*/
    else
    {
      /* RXNE set and BTF reset -----------------------------------------------*/
      if (((sr1itflags & SMBUS_FLAG_RXNE) != RESET) && ((itsources & SMBUS_IT_BUF) != RESET) && ((sr1itflags & SMBUS_FLAG_BTF) == RESET))
      {
        SMBUS_MasterReceive_RXNE(hsmbus);
      }
      /* BTF set -------------------------------------------------------------*/
      else if (((sr1itflags & SMBUS_FLAG_BTF) != RESET) && ((itsources & SMBUS_IT_EVT) != RESET))
      {
        SMBUS_MasterReceive_BTF(hsmbus);
      }
    }
  }
  /* Slave mode selected */
  else
  {
    /* ADDR set --------------------------------------------------------------*/
    if (((sr1itflags & SMBUS_FLAG_ADDR) != RESET) && ((itsources & SMBUS_IT_EVT) != RESET))
    {
      SMBUS_Slave_ADDR(hsmbus);
    }
    /* STOPF set --------------------------------------------------------------*/
    else if (((sr1itflags & SMBUS_FLAG_STOPF) != RESET) && ((itsources & SMBUS_IT_EVT) != RESET))
    {
      SMBUS_Slave_STOPF(hsmbus);
    }
    /* SMBUS in mode Transmitter -----------------------------------------------*/
    else if ((sr2itflags & SMBUS_FLAG_TRA) != RESET)
    {
      /* TXE set and BTF reset -----------------------------------------------*/
      if (((sr1itflags & SMBUS_FLAG_TXE) != RESET) && ((itsources & SMBUS_IT_BUF) != RESET) && ((sr1itflags & SMBUS_FLAG_BTF) == RESET))
      {
        SMBUS_SlaveTransmit_TXE(hsmbus);
      }
      /* BTF set -------------------------------------------------------------*/
      else if (((sr1itflags & SMBUS_FLAG_BTF) != RESET) && ((itsources & SMBUS_IT_EVT) != RESET))
      {
        SMBUS_SlaveTransmit_BTF(hsmbus);
      }
    }
    /* SMBUS in mode Receiver --------------------------------------------------*/
    else
    {
      /* RXNE set and BTF reset ----------------------------------------------*/
      if (((sr1itflags & SMBUS_FLAG_RXNE) != RESET) && ((itsources & SMBUS_IT_BUF) != RESET) && ((sr1itflags & SMBUS_FLAG_BTF) == RESET))
      {
        SMBUS_SlaveReceive_RXNE(hsmbus);
      }
      /* BTF set -------------------------------------------------------------*/
      else if (((sr1itflags & SMBUS_FLAG_BTF) != RESET) && ((itsources & SMBUS_IT_EVT) != RESET))
      {
        SMBUS_SlaveReceive_BTF(hsmbus);
      }
    }
  }
}

/**
  * @brief  This function handles SMBUS error interrupt request.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval None
  */
void HAL_SMBUS_ER_IRQHandler(SMBUS_HandleTypeDef *hsmbus)
{
  uint32_t tmp1 = 0U, tmp2 = 0U, tmp3 = 0U, tmp4 = 0U;
  uint32_t sr1itflags = READ_REG(hsmbus->Instance->SR1);
  uint32_t itsources  = READ_REG(hsmbus->Instance->CR2);

  /* SMBUS Bus error interrupt occurred ------------------------------------*/
  if (((sr1itflags & SMBUS_FLAG_BERR) != RESET) && ((itsources & SMBUS_IT_ERR) != RESET))
  {
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_BERR;

    /* Clear BERR flag */
    __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_BERR);

  }

  /* SMBUS Over-Run/Under-Run interrupt occurred ----------------------------------------*/
  if (((sr1itflags & SMBUS_FLAG_OVR) != RESET) && ((itsources & SMBUS_IT_ERR) != RESET))
  {
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_OVR;

    /* Clear OVR flag */
    __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_OVR);
  }

  /* SMBUS Arbitration Loss error interrupt occurred ------------------------------------*/
  if (((sr1itflags & SMBUS_FLAG_ARLO) != RESET) && ((itsources & SMBUS_IT_ERR) != RESET))
  {
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_ARLO;

    /* Clear ARLO flag */
    __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_ARLO);
  }

  /* SMBUS Acknowledge failure error interrupt occurred ------------------------------------*/
  if (((sr1itflags & SMBUS_FLAG_AF) != RESET) && ((itsources & SMBUS_IT_ERR) != RESET))
  {
    tmp1 = hsmbus->Mode;
    tmp2 = hsmbus->XferCount;
    tmp3 = hsmbus->State;
    tmp4 = hsmbus->PreviousState;

    if ((tmp1 == HAL_SMBUS_MODE_SLAVE) && (tmp2 == 0U) && \
        ((tmp3 == HAL_SMBUS_STATE_BUSY_TX) || (tmp3 == HAL_SMBUS_STATE_BUSY_TX_LISTEN) || \
         ((tmp3 == HAL_SMBUS_STATE_LISTEN) && (tmp4 == SMBUS_STATE_SLAVE_BUSY_TX))))
    {
      SMBUS_Slave_AF(hsmbus);
    }
    else
    {
      hsmbus->ErrorCode |= HAL_SMBUS_ERROR_AF;

      /* Do not generate a STOP in case of Slave receive non acknowledge during transfer (mean not at the end of transfer) */
      if (hsmbus->Mode == HAL_SMBUS_MODE_MASTER)
      {
        /* Generate Stop */
        SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);

      }

      /* Clear AF flag */
      __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_AF);
    }
  }

  /* SMBUS Timeout error interrupt occurred ---------------------------------------------*/
  if (((sr1itflags & SMBUS_FLAG_TIMEOUT) != RESET) && ((itsources & SMBUS_IT_ERR) != RESET))
  {
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_TIMEOUT;

    /* Clear TIMEOUT flag */
    __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_TIMEOUT);

  }

  /* SMBUS Alert error interrupt occurred -----------------------------------------------*/
  if (((sr1itflags & SMBUS_FLAG_SMBALERT) != RESET) && ((itsources & SMBUS_IT_ERR) != RESET))
  {
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_ALERT;

    /* Clear ALERT flag */
    __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_SMBALERT);
  }

  /* SMBUS Packet Error Check error interrupt occurred ----------------------------------*/
  if (((sr1itflags & SMBUS_FLAG_PECERR) != RESET) && ((itsources & SMBUS_IT_ERR) != RESET))
  {
    hsmbus->ErrorCode |= HAL_SMBUS_ERROR_PECERR;

    /* Clear PEC error flag */
    __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_PECERR);
  }

  /* Call the Error Callback in case of Error detected -----------------------*/
  if (hsmbus->ErrorCode != HAL_SMBUS_ERROR_NONE)
  {
    SMBUS_ITError(hsmbus);
  }
}

/**
  * @brief  Master Tx Transfer completed callback.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval None
  */
__weak void HAL_SMBUS_MasterTxCpltCallback(SMBUS_HandleTypeDef *hsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMBUS_MasterTxCpltCallback can be implemented in the user file
   */
}

/**
  * @brief  Master Rx Transfer completed callback.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval None
  */
__weak void HAL_SMBUS_MasterRxCpltCallback(SMBUS_HandleTypeDef *hsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMBUS_MasterRxCpltCallback can be implemented in the user file
   */
}

/** @brief  Slave Tx Transfer completed callback.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval None
  */
__weak void HAL_SMBUS_SlaveTxCpltCallback(SMBUS_HandleTypeDef *hsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMBUS_SlaveTxCpltCallback can be implemented in the user file
   */
}

/**
  * @brief  Slave Rx Transfer completed callback.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval None
  */
__weak void HAL_SMBUS_SlaveRxCpltCallback(SMBUS_HandleTypeDef *hsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMBUS_SlaveRxCpltCallback can be implemented in the user file
   */
}

/**
  * @brief  Slave Address Match callback.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @param  TransferDirection Master request Transfer Direction (Write/Read), value of @ref SMBUS_XferOptions_definition
  * @param  AddrMatchCode Address Match Code
  * @retval None
  */
__weak void HAL_SMBUS_AddrCallback(SMBUS_HandleTypeDef *hsmbus, uint8_t TransferDirection, uint16_t AddrMatchCode)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);
  UNUSED(TransferDirection);
  UNUSED(AddrMatchCode);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMBUS_AddrCallback can be implemented in the user file
   */
}

/**
  * @brief  Listen Complete callback.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval None
  */
__weak void HAL_SMBUS_ListenCpltCallback(SMBUS_HandleTypeDef *hsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
          the HAL_SMBUS_ListenCpltCallback can be implemented in the user file
  */
}

/**
  * @brief  SMBUS error callback.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval None
  */
__weak void HAL_SMBUS_ErrorCallback(SMBUS_HandleTypeDef *hsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMBUS_ErrorCallback can be implemented in the user file
   */
}

/**
  * @brief  SMBUS abort callback.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval None
  */
__weak void HAL_SMBUS_AbortCpltCallback(SMBUS_HandleTypeDef *hsmbus)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsmbus);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SMBUS_AbortCpltCallback could be implemented in the user file
   */
}

/**
  * @}
  */

/** @defgroup   SMBUS_Exported_Functions_Group3 Peripheral State, Mode and Error functions
  *  @brief   Peripheral State and Errors functions
  *
@verbatim
 ===============================================================================
            ##### Peripheral State, Mode and Error functions #####
 ===============================================================================
    [..]
    This subsection permits to get in run-time the status of the peripheral
    and the data flow.

@endverbatim
  * @{
  */

/**
  * @brief  Return the SMBUS handle state.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for the specified SMBUS.
  * @retval HAL state
  */
HAL_SMBUS_StateTypeDef HAL_SMBUS_GetState(SMBUS_HandleTypeDef *hsmbus)
{
  /* Return SMBUS handle state */
  return hsmbus->State;
}

/**
  * @brief  Return the SMBUS Master, Slave or no mode.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *                the configuration information for SMBUS module
  * @retval HAL mode
  */
HAL_SMBUS_ModeTypeDef HAL_SMBUS_GetMode(SMBUS_HandleTypeDef *hsmbus)
{
  return hsmbus->Mode;
}

/**
  * @brief  Return the SMBUS error code
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *              the configuration information for the specified SMBUS.
  * @retval SMBUS Error Code
  */
uint32_t HAL_SMBUS_GetError(SMBUS_HandleTypeDef *hsmbus)
{
  return hsmbus->ErrorCode;
}

/**
  * @}
  */

/**
  * @}
  */

/** @addtogroup SMBUS_Private_Functions
  * @{
  */

/**
  * @brief  Handle TXE flag for Master
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_MasterTransmit_TXE(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variables to prevent undefined behavior of volatile usage */
  uint32_t CurrentState       = hsmbus->State;
  uint32_t CurrentXferOptions = hsmbus->XferOptions;

  if ((hsmbus->XferSize == 0U) && (CurrentState == HAL_SMBUS_STATE_BUSY_TX))
  {
    /* Call TxCpltCallback() directly if no stop mode is set */
    if (((CurrentXferOptions == SMBUS_FIRST_FRAME) || (CurrentXferOptions == SMBUS_NEXT_FRAME)) && (CurrentXferOptions != SMBUS_NO_OPTION_FRAME))
    {
      __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

      hsmbus->PreviousState = SMBUS_STATE_MASTER_BUSY_TX;
      hsmbus->Mode = HAL_SMBUS_MODE_NONE;
      hsmbus->State = HAL_SMBUS_STATE_READY;

#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
      hsmbus->MasterTxCpltCallback(hsmbus);
#else
      HAL_SMBUS_MasterTxCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
    }
    else /* Generate Stop condition then Call TxCpltCallback() */
    {
      /* Disable EVT, BUF and ERR interrupt */
      __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

      /* Generate Stop */
      SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);

      hsmbus->PreviousState = HAL_SMBUS_STATE_READY;
      hsmbus->State = HAL_SMBUS_STATE_READY;

      hsmbus->Mode = HAL_SMBUS_MODE_NONE;
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
      hsmbus->MasterTxCpltCallback(hsmbus);
#else
      HAL_SMBUS_MasterTxCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
    }
  }
  else if (CurrentState == HAL_SMBUS_STATE_BUSY_TX)
  {

    if ((hsmbus->XferCount == 2U) && (SMBUS_GET_PEC_MODE(hsmbus) == SMBUS_PEC_ENABLE) && ((hsmbus->XferOptions == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || (hsmbus->XferOptions == SMBUS_LAST_FRAME_WITH_PEC)))
    {
      hsmbus->XferCount--;
    }

    if (hsmbus->XferCount == 0U)
    {

      /* Disable BUF interrupt */
      __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_BUF);

      if ((SMBUS_GET_PEC_MODE(hsmbus) == SMBUS_PEC_ENABLE) && ((hsmbus->XferOptions == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || (hsmbus->XferOptions == SMBUS_LAST_FRAME_WITH_PEC)))
      {
        SET_BIT(hsmbus->Instance->CR1, I2C_CR1_PEC);
      }

    }
    else
    {
      /* Write data to DR */
      hsmbus->Instance->DR = (*hsmbus->pBuffPtr++);
      hsmbus->XferCount--;
    }
  }
  return HAL_OK;
}

/**
  * @brief  Handle BTF flag for Master transmitter
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_MasterTransmit_BTF(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variables to prevent undefined behavior of volatile usage */
  uint32_t CurrentXferOptions = hsmbus->XferOptions;

  if (hsmbus->State == HAL_SMBUS_STATE_BUSY_TX)
  {
    if (hsmbus->XferCount != 0U)
    {
      /* Write data to DR */
      hsmbus->Instance->DR = (*hsmbus->pBuffPtr++);
      hsmbus->XferCount--;
    }
    else
    {
      /* Call TxCpltCallback() directly if no stop mode is set */
      if (((CurrentXferOptions == SMBUS_FIRST_FRAME) || (CurrentXferOptions == SMBUS_NEXT_FRAME)) && (CurrentXferOptions != SMBUS_NO_OPTION_FRAME))
      {
        __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

        hsmbus->PreviousState = SMBUS_STATE_MASTER_BUSY_TX;
        hsmbus->Mode = HAL_SMBUS_MODE_NONE;
        hsmbus->State = HAL_SMBUS_STATE_READY;

#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
        hsmbus->MasterTxCpltCallback(hsmbus);
#else
        HAL_SMBUS_MasterTxCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
      }
      else /* Generate Stop condition then Call TxCpltCallback() */
      {
        /* Disable EVT, BUF and ERR interrupt */
        __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

        /* Generate Stop */
        SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);

        hsmbus->PreviousState = HAL_SMBUS_STATE_READY;
        hsmbus->State = HAL_SMBUS_STATE_READY;
        hsmbus->Mode = HAL_SMBUS_MODE_NONE;
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
        hsmbus->MasterTxCpltCallback(hsmbus);
#else
        HAL_SMBUS_MasterTxCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
      }
    }
  }
  return HAL_OK;
}

/**
  * @brief  Handle RXNE flag for Master
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_MasterReceive_RXNE(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variables to prevent undefined behavior of volatile usage */
  uint32_t CurrentXferOptions = hsmbus->XferOptions;

  if (hsmbus->State == HAL_SMBUS_STATE_BUSY_RX)
  {
    uint32_t tmp = hsmbus->XferCount;

    if (tmp > 3U)
    {
      /* Read data from DR */
      (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
      hsmbus->XferCount--;

      if (hsmbus->XferCount == 3)
      {
        /* Disable BUF interrupt, this help to treat correctly the last 4 bytes
        on BTF subroutine */
        /* Disable BUF interrupt */
        __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_BUF);
      }
    }

    else if (tmp == 2U)
    {

      /* Read data from DR */
      (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
      hsmbus->XferCount--;

      if ((CurrentXferOptions == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || (CurrentXferOptions == SMBUS_LAST_FRAME_WITH_PEC))
      {
        /* PEC of slave */
        hsmbus->XferPEC = SMBUS_GET_PEC(hsmbus);

      }
      /* Generate Stop */
      SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);
    }

    else if ((tmp == 1U) || (tmp == 0U))
    {
      /* Disable Acknowledge */
      CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

      /* Disable EVT, BUF and ERR interrupt */
      __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

      /* Read data from DR */
      (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
      hsmbus->XferCount--;

      hsmbus->State = HAL_SMBUS_STATE_READY;
      hsmbus->PreviousState = SMBUS_STATE_NONE;
      hsmbus->Mode = HAL_SMBUS_MODE_NONE;

#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
      hsmbus->MasterRxCpltCallback(hsmbus);
#else
      HAL_SMBUS_MasterRxCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
    }
  }

  return HAL_OK;
}

/**
  * @brief  Handle BTF flag for Master receiver
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_MasterReceive_BTF(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variables to prevent undefined behavior of volatile usage */
  uint32_t CurrentXferOptions = hsmbus->XferOptions;

  if (hsmbus->XferCount == 4U)
  {
    /* Disable BUF interrupt, this help to treat correctly the last 2 bytes
       on BTF subroutine if there is a reception delay between N-1 and N byte */
    __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_BUF);

    /* Read data from DR */
    (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
    hsmbus->XferCount--;
    hsmbus->XferPEC = SMBUS_GET_PEC(hsmbus);
  }
  else if (hsmbus->XferCount == 3U)
  {
    /* Disable BUF interrupt, this help to treat correctly the last 2 bytes
       on BTF subroutine if there is a reception delay between N-1 and N byte */
    __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_BUF);

    /* Disable Acknowledge */
    CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

    /* Read data from DR */
    (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
    hsmbus->XferCount--;
    hsmbus->XferPEC = SMBUS_GET_PEC(hsmbus);
  }
  else if (hsmbus->XferCount == 2U)
  {
    /* Prepare next transfer or stop current transfer */
    if ((CurrentXferOptions == SMBUS_NEXT_FRAME) || (CurrentXferOptions == SMBUS_FIRST_FRAME))
    {
      /* Disable Acknowledge */
      CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

      /* Generate ReStart */
      SET_BIT(hsmbus->Instance->CR1, I2C_CR1_START);
    }
    else
    {
      /* Generate Stop */
      SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);
    }

    /* Read data from DR */
    (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
    hsmbus->XferCount--;

    /* Read data from DR */
    (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
    hsmbus->XferCount--;

    /* Disable EVT and ERR interrupt */
    __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_ERR);

    hsmbus->State = HAL_SMBUS_STATE_READY;
    hsmbus->PreviousState = SMBUS_STATE_NONE;
    hsmbus->Mode = HAL_SMBUS_MODE_NONE;
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
    hsmbus->MasterRxCpltCallback(hsmbus);
#else
    HAL_SMBUS_MasterRxCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
  }
  else
  {
    /* Read data from DR */
    (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
    hsmbus->XferCount--;
  }
  return HAL_OK;
}

/**
  * @brief  Handle SB flag for Master
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_Master_SB(SMBUS_HandleTypeDef *hsmbus)
{
  if (hsmbus->Init.AddressingMode == SMBUS_ADDRESSINGMODE_7BIT)
  {
    /* Send slave 7 Bits address */
    if (hsmbus->State == HAL_SMBUS_STATE_BUSY_TX)
    {
      hsmbus->Instance->DR = SMBUS_7BIT_ADD_WRITE(hsmbus->Devaddress);
    }
    else
    {
      hsmbus->Instance->DR = SMBUS_7BIT_ADD_READ(hsmbus->Devaddress);
    }
  }
  else
  {
    if (hsmbus->EventCount == 0U)
    {
      /* Send header of slave address */
      hsmbus->Instance->DR = SMBUS_10BIT_HEADER_WRITE(hsmbus->Devaddress);
    }
    else if (hsmbus->EventCount == 1U)
    {
      /* Send header of slave address */
      hsmbus->Instance->DR = SMBUS_10BIT_HEADER_READ(hsmbus->Devaddress);
    }
  }
  return HAL_OK;
}

/**
  * @brief  Handle ADD10 flag for Master
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_Master_ADD10(SMBUS_HandleTypeDef *hsmbus)
{
  /* Send slave address */
  hsmbus->Instance->DR = SMBUS_10BIT_ADDRESS(hsmbus->Devaddress);

  return HAL_OK;
}

/**
  * @brief  Handle ADDR flag for Master
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_Master_ADDR(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variable to prevent undefined behavior of volatile usage */
  uint32_t Prev_State         = hsmbus->PreviousState;

  if (hsmbus->State == HAL_SMBUS_STATE_BUSY_RX)
  {
    if ((hsmbus->EventCount == 0U) && (hsmbus->Init.AddressingMode == SMBUS_ADDRESSINGMODE_10BIT))
    {
      /* Clear ADDR flag */
      __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);

      /* Generate Restart */
      SET_BIT(hsmbus->Instance->CR1, I2C_CR1_START);

      hsmbus->EventCount++;
    }
    else
    {
      /*  In the case of the Quick Command, the ADDR flag is cleared and a stop is generated */
      if (hsmbus->XferCount == 0U)
      {
        /* Clear ADDR flag */
        __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);

        /* Generate Stop */
        SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);
      }
      else if (hsmbus->XferCount == 1U)
      {
        /* Prepare next transfer or stop current transfer */
        if ((hsmbus->XferOptions == SMBUS_FIRST_FRAME) && (Prev_State != SMBUS_STATE_MASTER_BUSY_RX))
        {
          /* Disable Acknowledge */
          CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

          /* Clear ADDR flag */
          __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);
        }
        else if ((hsmbus->XferOptions == SMBUS_NEXT_FRAME) && (Prev_State != SMBUS_STATE_MASTER_BUSY_RX))
        {
          /* Enable Acknowledge */
          SET_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

          /* Clear ADDR flag */
          __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);
        }
        else
        {
          /* Disable Acknowledge */
          CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

          /* Clear ADDR flag */
          __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);

          /* Generate Stop */
          SET_BIT(hsmbus->Instance->CR1, I2C_CR1_STOP);
        }
      }
      else if (hsmbus->XferCount == 2U)
      {
        if (hsmbus->XferOptions != SMBUS_NEXT_FRAME)
        {
          /* Disable Acknowledge */
          CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

          /* Enable Pos */
          SET_BIT(hsmbus->Instance->CR1, I2C_CR1_POS);


        }
        else
        {
          /* Enable Acknowledge */
          SET_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);
        }

        /* Clear ADDR flag */
        __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);
      }
      else
      {
        /* Enable Acknowledge */
        SET_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

        /* Clear ADDR flag */
        __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);
      }

      /* Reset Event counter  */
      hsmbus->EventCount = 0U;
    }
  }
  else
  {
    /* Clear ADDR flag */
    __HAL_SMBUS_CLEAR_ADDRFLAG(hsmbus);
  }

  return HAL_OK;
}

/**
  * @brief  Handle TXE flag for Slave
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_SlaveTransmit_TXE(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variables to prevent undefined behavior of volatile usage */
  uint32_t CurrentState = hsmbus->State;

  if (hsmbus->XferCount != 0U)
  {
    /* Write data to DR */
    hsmbus->Instance->DR = (*hsmbus->pBuffPtr++);
    hsmbus->XferCount--;

    if ((hsmbus->XferCount == 2U) && (SMBUS_GET_PEC_MODE(hsmbus) == SMBUS_PEC_ENABLE) && ((hsmbus->XferOptions == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || (hsmbus->XferOptions == SMBUS_LAST_FRAME_WITH_PEC)))
    {
      hsmbus->XferCount--;
    }

    if ((hsmbus->XferCount == 0U) && (CurrentState == (HAL_SMBUS_STATE_BUSY_TX_LISTEN)))
    {
      /* Last Byte is received, disable Interrupt */
      __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_BUF);

      /* Set state at HAL_SMBUS_STATE_LISTEN */
      hsmbus->PreviousState = SMBUS_STATE_SLAVE_BUSY_TX;
      hsmbus->State = HAL_SMBUS_STATE_LISTEN;

      /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
      hsmbus->SlaveTxCpltCallback(hsmbus);
#else
      HAL_SMBUS_SlaveTxCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
    }
  }
  return HAL_OK;
}

/**
  * @brief  Handle BTF flag for Slave transmitter
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_SlaveTransmit_BTF(SMBUS_HandleTypeDef *hsmbus)
{
  if (hsmbus->XferCount != 0U)
  {
    /* Write data to DR */
    hsmbus->Instance->DR = (*hsmbus->pBuffPtr++);
    hsmbus->XferCount--;
  }



  else if ((hsmbus->XferCount == 0U) && (SMBUS_GET_PEC_MODE(hsmbus) == SMBUS_PEC_ENABLE) && ((hsmbus->XferOptions == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || (hsmbus->XferOptions == SMBUS_LAST_FRAME_WITH_PEC)))
  {
    SET_BIT(hsmbus->Instance->CR1, I2C_CR1_PEC);
  }
  return HAL_OK;
}

/**
  * @brief  Handle RXNE flag for Slave
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_SlaveReceive_RXNE(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variables to prevent undefined behavior of volatile usage */
  uint32_t CurrentState = hsmbus->State;

  if (hsmbus->XferCount != 0U)
  {
    /* Read data from DR */
    (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
    hsmbus->XferCount--;

    if ((hsmbus->XferCount == 1U) && (SMBUS_GET_PEC_MODE(hsmbus) == SMBUS_PEC_ENABLE) && ((hsmbus->XferOptions == SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || (hsmbus->XferOptions == SMBUS_LAST_FRAME_WITH_PEC)))
    {
      SET_BIT(hsmbus->Instance->CR1, I2C_CR1_PEC);
      hsmbus->XferPEC = SMBUS_GET_PEC(hsmbus);
    }
    if ((hsmbus->XferCount == 0U) && (CurrentState == HAL_SMBUS_STATE_BUSY_RX_LISTEN))
    {
      /* Last Byte is received, disable Interrupt */
      __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_BUF);

      /* Set state at HAL_SMBUS_STATE_LISTEN */
      hsmbus->PreviousState = SMBUS_STATE_SLAVE_BUSY_RX;
      hsmbus->State = HAL_SMBUS_STATE_LISTEN;

      /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
      hsmbus->SlaveRxCpltCallback(hsmbus);
#else
      HAL_SMBUS_SlaveRxCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
    }
  }
  return HAL_OK;
}

/**
  * @brief  Handle BTF flag for Slave receiver
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_SlaveReceive_BTF(SMBUS_HandleTypeDef *hsmbus)
{
  if (hsmbus->XferCount != 0U)
  {
    /* Read data from DR */
    (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
    hsmbus->XferCount--;
  }

  return HAL_OK;
}

/**
  * @brief  Handle ADD flag for Slave
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_Slave_ADDR(SMBUS_HandleTypeDef *hsmbus)
{
  uint8_t TransferDirection = SMBUS_DIRECTION_RECEIVE ;
  uint16_t SlaveAddrCode = 0U;

  /* Transfer Direction requested by Master */
  if (__HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_TRA) == RESET)
  {
    TransferDirection = SMBUS_DIRECTION_TRANSMIT;
  }

  if (__HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_DUALF) == RESET)
  {
    SlaveAddrCode = hsmbus->Init.OwnAddress1;
  }
  else
  {
    SlaveAddrCode = hsmbus->Init.OwnAddress2;
  }

  /* Call Slave Addr callback */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
  hsmbus->AddrCallback(hsmbus, TransferDirection, SlaveAddrCode);
#else
  HAL_SMBUS_AddrCallback(hsmbus, TransferDirection, SlaveAddrCode);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */

  return HAL_OK;
}

/**
  * @brief  Handle STOPF flag for Slave
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_Slave_STOPF(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variable to prevent undefined behavior of volatile usage */
  uint32_t CurrentState = hsmbus->State;

  /* Disable EVT, BUF and ERR interrupt */
  __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

  /* Clear STOPF flag */
  __HAL_SMBUS_CLEAR_STOPFLAG(hsmbus);

  /* Disable Acknowledge */
  CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

  /* All data are not transferred, so set error code accordingly */
  if (hsmbus->XferCount != 0U)
  {
    /* Store Last receive data if any */
    if (__HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_BTF) == SET)
    {
      /* Read data from DR */
      (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;

      if (hsmbus->XferCount > 0)
      {
        hsmbus->XferSize--;
        hsmbus->XferCount--;
      }
    }

    /* Store Last receive data if any */
    if (__HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_RXNE) == SET)
    {
      /* Read data from DR */
      (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;

      if (hsmbus->XferCount > 0)
      {
        hsmbus->XferSize--;
        hsmbus->XferCount--;
      }
    }
  }

  if (hsmbus->ErrorCode != HAL_SMBUS_ERROR_NONE)
  {
    /* Call the corresponding callback to inform upper layer of End of Transfer */
    SMBUS_ITError(hsmbus);
  }
  else
  {
    if ((CurrentState == HAL_SMBUS_STATE_LISTEN) || (CurrentState == HAL_SMBUS_STATE_BUSY_RX_LISTEN)  || \
        (CurrentState == HAL_SMBUS_STATE_BUSY_TX_LISTEN))
    {
      hsmbus->XferOptions = SMBUS_NO_OPTION_FRAME;
      hsmbus->PreviousState = HAL_SMBUS_STATE_READY;
      hsmbus->State = HAL_SMBUS_STATE_READY;
      hsmbus->Mode = HAL_SMBUS_MODE_NONE;

#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
      hsmbus->ListenCpltCallback(hsmbus);
#else
      HAL_SMBUS_ListenCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
    }
  }
  return HAL_OK;
}

/**
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_Slave_AF(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variables to prevent undefined behavior of volatile usage */
  uint32_t CurrentState       = hsmbus->State;
  uint32_t CurrentXferOptions = hsmbus->XferOptions;

  if (((CurrentXferOptions ==  SMBUS_FIRST_AND_LAST_FRAME_NO_PEC) || (CurrentXferOptions ==  SMBUS_FIRST_AND_LAST_FRAME_WITH_PEC) || \
       (CurrentXferOptions == SMBUS_LAST_FRAME_NO_PEC) || (CurrentXferOptions ==  SMBUS_LAST_FRAME_WITH_PEC)) && \
      (CurrentState == HAL_SMBUS_STATE_LISTEN))
  {
    hsmbus->XferOptions = SMBUS_NO_OPTION_FRAME;

    /* Disable EVT, BUF and ERR interrupt */
    __HAL_SMBUS_DISABLE_IT(hsmbus, SMBUS_IT_EVT | SMBUS_IT_BUF | SMBUS_IT_ERR);

    /* Clear AF flag */
    __HAL_SMBUS_CLEAR_FLAG(hsmbus, SMBUS_FLAG_AF);

    /* Disable Acknowledge */
    CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_ACK);

    hsmbus->PreviousState = HAL_SMBUS_STATE_READY;
    hsmbus->State = HAL_SMBUS_STATE_READY;
    hsmbus->Mode = HAL_SMBUS_MODE_NONE;

    /* Call the Listen Complete callback, to inform upper layer of the end of Listen usecase */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
    hsmbus->ListenCpltCallback(hsmbus);
#else
    HAL_SMBUS_ListenCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
  }
  return HAL_OK;
}



/**
  * @brief SMBUS interrupts error process
  * @param  hsmbus SMBUS handle.
  * @retval None
  */
static void SMBUS_ITError(SMBUS_HandleTypeDef *hsmbus)
{
  /* Declaration of temporary variable to prevent undefined behavior of volatile usage */
  uint32_t CurrentState = hsmbus->State;

  if ((CurrentState == HAL_SMBUS_STATE_BUSY_TX_LISTEN) || (CurrentState == HAL_SMBUS_STATE_BUSY_RX_LISTEN))
  {
    /* keep HAL_SMBUS_STATE_LISTEN */
    hsmbus->PreviousState = SMBUS_STATE_NONE;
    hsmbus->State = HAL_SMBUS_STATE_LISTEN;
  }
  else
  {
    /* If state is an abort treatment on going, don't change state */
    /* This change will be done later */
    if (hsmbus->State != HAL_SMBUS_STATE_ABORT)
    {
      hsmbus->State = HAL_SMBUS_STATE_READY;
    }
    hsmbus->PreviousState = SMBUS_STATE_NONE;
    hsmbus->Mode = HAL_SMBUS_MODE_NONE;
  }

  /* Disable Pos bit in SMBUS CR1 when error occurred in Master/Mem Receive IT Process */
  CLEAR_BIT(hsmbus->Instance->CR1, I2C_CR1_POS);

  if (hsmbus->State == HAL_SMBUS_STATE_ABORT)
  {
    hsmbus->State = HAL_SMBUS_STATE_READY;
    hsmbus->ErrorCode = HAL_SMBUS_ERROR_NONE;

    /* Store Last receive data if any */
    if (__HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_RXNE) == SET)
    {
      /* Read data from DR */
      (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
    }

    /* Disable SMBUS peripheral to prevent dummy data in buffer */
    __HAL_SMBUS_DISABLE(hsmbus);

    /* Call the corresponding callback to inform upper layer of End of Transfer */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
    hsmbus->AbortCpltCallback(hsmbus);
#else
    HAL_SMBUS_AbortCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
  }
  else
  {
    /* Store Last receive data if any */
    if (__HAL_SMBUS_GET_FLAG(hsmbus, SMBUS_FLAG_RXNE) == SET)
    {
      /* Read data from DR */
      (*hsmbus->pBuffPtr++) = hsmbus->Instance->DR;
    }

    /* Call user error callback */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
    hsmbus->ErrorCallback(hsmbus);
#else
    HAL_SMBUS_ErrorCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
  }
  /* STOP Flag is not set after a NACK reception */
  /* So may inform upper layer that listen phase is stopped */
  /* during NACK error treatment */
  if ((hsmbus->State == HAL_SMBUS_STATE_LISTEN) && ((hsmbus->ErrorCode & HAL_SMBUS_ERROR_AF) == HAL_SMBUS_ERROR_AF))
  {
    hsmbus->XferOptions = SMBUS_NO_OPTION_FRAME;
    hsmbus->PreviousState = SMBUS_STATE_NONE;
    hsmbus->State = HAL_SMBUS_STATE_READY;
    hsmbus->Mode = HAL_SMBUS_MODE_NONE;

    /* Call the Listen Complete callback, to inform upper layer of the end of Listen usecase */
#if (USE_HAL_SMBUS_REGISTER_CALLBACKS == 1)
    hsmbus->ListenCpltCallback(hsmbus);
#else
    HAL_SMBUS_ListenCpltCallback(hsmbus);
#endif /* USE_HAL_SMBUS_REGISTER_CALLBACKS */
  }
}

/**
  * @brief  This function handles SMBUS Communication Timeout.
  * @param  hsmbus Pointer to a SMBUS_HandleTypeDef structure that contains
  *         the configuration information for SMBUS module
  * @param  Flag specifies the SMBUS flag to check.
  * @param  Status The new Flag status (SET or RESET).
  * @param  Timeout Timeout duration
  * @param  Tickstart Tick start value
  * @retval HAL status
  */
static HAL_StatusTypeDef SMBUS_WaitOnFlagUntilTimeout(SMBUS_HandleTypeDef *hsmbus, uint32_t Flag, FlagStatus Status, uint32_t Timeout, uint32_t Tickstart)
{
  /* Wait until flag is set */
  if (Status == RESET)
  {
    while (__HAL_SMBUS_GET_FLAG(hsmbus, Flag) == RESET)
    {
      /* Check for the Timeout */
      if (Timeout != HAL_MAX_DELAY)
      {
        if ((Timeout == 0U) || ((HAL_GetTick() - Tickstart) > Timeout))
        {
          hsmbus->PreviousState = SMBUS_STATE_NONE;
          hsmbus->State = HAL_SMBUS_STATE_READY;
          hsmbus->Mode = HAL_SMBUS_MODE_NONE;

          /* Process Unlocked */
          __HAL_UNLOCK(hsmbus);
          return HAL_TIMEOUT;
        }
      }
    }
  }
  return HAL_OK;
}

/**
  * @}
  */


#endif /* HAL_SMBUS_MODULE_ENABLED */

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
