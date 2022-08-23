/**
  ******************************************************************************
  * @file    stm32f4xx_hal_sai.c
  * @author  MCD Application Team
  * @brief   SAI HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the Serial Audio Interface (SAI) peripheral:
  *           + Initialization/de-initialization functions
  *           + I/O operation functions
  *           + Peripheral Control functions
  *           + Peripheral State functions
  *
  @verbatim
  ==============================================================================
                  ##### How to use this driver #####
  ==============================================================================

  [..]
    The SAI HAL driver can be used as follows:

    (#) Declare a SAI_HandleTypeDef handle structure (eg. SAI_HandleTypeDef hsai).
    (#) Initialize the SAI low level resources by implementing the HAL_SAI_MspInit() API:
        (##) Enable the SAI interface clock.
        (##) SAI pins configuration:
            (+++) Enable the clock for the SAI GPIOs.
            (+++) Configure these SAI pins as alternate function pull-up.
        (##) NVIC configuration if you need to use interrupt process (HAL_SAI_Transmit_IT()
             and HAL_SAI_Receive_IT() APIs):
            (+++) Configure the SAI interrupt priority.
            (+++) Enable the NVIC SAI IRQ handle.

        (##) DMA Configuration if you need to use DMA process (HAL_SAI_Transmit_DMA()
             and HAL_SAI_Receive_DMA() APIs):
            (+++) Declare a DMA handle structure for the Tx/Rx stream.
            (+++) Enable the DMAx interface clock.
            (+++) Configure the declared DMA handle structure with the required Tx/Rx parameters.
            (+++) Configure the DMA Tx/Rx Stream.
            (+++) Associate the initialized DMA handle to the SAI DMA Tx/Rx handle.
            (+++) Configure the priority and enable the NVIC for the transfer complete interrupt on the
                DMA Tx/Rx Stream.

    (#) The initialization can be done by two ways
        (##) Expert mode : Initialize the structures Init, FrameInit and SlotInit and call HAL_SAI_Init().
        (##) Simplified mode : Initialize the high part of Init Structure and call HAL_SAI_InitProtocol().

  [..]
    (@) The specific SAI interrupts (FIFO request and Overrun underrun interrupt)
        will be managed using the macros __HAL_SAI_ENABLE_IT() and __HAL_SAI_DISABLE_IT()
        inside the transmit and receive process.

  [..]
   (@) SAI Clock Source configuration is managed differently depending on the selected
       STM32F4 devices :
       (+@) For STM32F446xx devices, the configuration is managed through RCCEx_PeriphCLKConfig()
            function in the HAL RCC drivers
       (+@) For STM32F439xx/STM32F437xx/STM32F429xx/STM32F427xx devices, the configuration
            is managed within HAL SAI drivers through HAL_SAI_Init() function using
            ClockSource field of SAI_InitTypeDef structure.
  [..]
   (@) Make sure that either:
       (+@) I2S PLL is configured or
       (+@) SAI PLL is configured or
       (+@) External clock source is configured after setting correctly
            the define constant EXTERNAL_CLOCK_VALUE in the stm32f4xx_hal_conf.h file.
  [..]
    (@) In master Tx mode: enabling the audio block immediately generates the bit clock
        for the external slaves even if there is no data in the FIFO, However FS signal
        generation is conditioned by the presence of data in the FIFO.

  [..]
    (@) In master Rx mode: enabling the audio block immediately generates the bit clock
        and FS signal for the external slaves.

  [..]
    (@) It is mandatory to respect the following conditions in order to avoid bad SAI behavior:
        (+@) First bit Offset <= (SLOT size - Data size)
        (+@) Data size <= SLOT size
        (+@) Number of SLOT x SLOT size = Frame length
        (+@) The number of slots should be even when SAI_FS_CHANNEL_IDENTIFICATION is selected.

  [..]
    Three operation modes are available within this driver :

    *** Polling mode IO operation ***
    =================================
    [..]
      (+) Send an amount of data in blocking mode using HAL_SAI_Transmit()
      (+) Receive an amount of data in blocking mode using HAL_SAI_Receive()

    *** Interrupt mode IO operation ***
    ===================================
    [..]
      (+) Send an amount of data in non-blocking mode using HAL_SAI_Transmit_IT()
      (+) At transmission end of transfer HAL_SAI_TxCpltCallback() is executed and user can
          add his own code by customization of function pointer HAL_SAI_TxCpltCallback()
      (+) Receive an amount of data in non-blocking mode using HAL_SAI_Receive_IT()
      (+) At reception end of transfer HAL_SAI_RxCpltCallback() is executed and user can
          add his own code by customization of function pointer HAL_SAI_RxCpltCallback()
      (+) In case of flag error, HAL_SAI_ErrorCallback() function is executed and user can
          add his own code by customization of function pointer HAL_SAI_ErrorCallback()

    *** DMA mode IO operation ***
    =============================
    [..]
      (+) Send an amount of data in non-blocking mode (DMA) using HAL_SAI_Transmit_DMA()
      (+) At transmission end of transfer HAL_SAI_TxCpltCallback() is executed and user can
          add his own code by customization of function pointer HAL_SAI_TxCpltCallback()
      (+) Receive an amount of data in non-blocking mode (DMA) using HAL_SAI_Receive_DMA()
      (+) At reception end of transfer HAL_SAI_RxCpltCallback() is executed and user can
          add his own code by customization of function pointer HAL_SAI_RxCpltCallback()
      (+) In case of flag error, HAL_SAI_ErrorCallback() function is executed and user can
          add his own code by customization of function pointer HAL_SAI_ErrorCallback()
      (+) Pause the DMA Transfer using HAL_SAI_DMAPause()
      (+) Resume the DMA Transfer using HAL_SAI_DMAResume()
      (+) Stop the DMA Transfer using HAL_SAI_DMAStop()

    *** SAI HAL driver additional function list ***
    ===============================================
    [..]
      Below the list the others API available SAI HAL driver :

      (+) HAL_SAI_EnableTxMuteMode(): Enable the mute in tx mode
      (+) HAL_SAI_DisableTxMuteMode(): Disable the mute in tx mode
      (+) HAL_SAI_EnableRxMuteMode(): Enable the mute in Rx mode
      (+) HAL_SAI_DisableRxMuteMode(): Disable the mute in Rx mode
      (+) HAL_SAI_FlushRxFifo(): Flush the rx fifo.
      (+) HAL_SAI_Abort(): Abort the current transfer

    *** SAI HAL driver macros list ***
    ==================================
    [..]
      Below the list of most used macros in SAI HAL driver :

      (+) __HAL_SAI_ENABLE(): Enable the SAI peripheral
      (+) __HAL_SAI_DISABLE(): Disable the SAI peripheral
      (+) __HAL_SAI_ENABLE_IT(): Enable the specified SAI interrupts
      (+) __HAL_SAI_DISABLE_IT(): Disable the specified SAI interrupts
      (+) __HAL_SAI_GET_IT_SOURCE(): Check if the specified SAI interrupt source is
          enabled or disabled
      (+) __HAL_SAI_GET_FLAG(): Check whether the specified SAI flag is set or not

    *** Callback registration ***
    =============================
    [..]
    The compilation define USE_HAL_SAI_REGISTER_CALLBACKS when set to 1
    allows the user to configure dynamically the driver callbacks.
    Use functions HAL_SAI_RegisterCallback() to register a user callback.

    [..]
    Function HAL_SAI_RegisterCallback() allows to register following callbacks:
      (+) RxCpltCallback     : SAI receive complete.
      (+) RxHalfCpltCallback : SAI receive half complete.
      (+) TxCpltCallback     : SAI transmit complete.
      (+) TxHalfCpltCallback : SAI transmit half complete.
      (+) ErrorCallback      : SAI error.
      (+) MspInitCallback    : SAI MspInit.
      (+) MspDeInitCallback  : SAI MspDeInit.
    [..]
    This function takes as parameters the HAL peripheral handle, the callback ID
    and a pointer to the user callback function.

    [..]
    Use function HAL_SAI_UnRegisterCallback() to reset a callback to the default
    weak (surcharged) function.
    HAL_SAI_UnRegisterCallback() takes as parameters the HAL peripheral handle,
    and the callback ID.
    [..]
    This function allows to reset following callbacks:
      (+) RxCpltCallback     : SAI receive complete.
      (+) RxHalfCpltCallback : SAI receive half complete.
      (+) TxCpltCallback     : SAI transmit complete.
      (+) TxHalfCpltCallback : SAI transmit half complete.
      (+) ErrorCallback      : SAI error.
      (+) MspInitCallback    : SAI MspInit.
      (+) MspDeInitCallback  : SAI MspDeInit.

    [..]
    By default, after the HAL_SAI_Init and if the state is HAL_SAI_STATE_RESET
    all callbacks are reset to the corresponding legacy weak (surcharged) functions:
    examples HAL_SAI_RxCpltCallback(), HAL_SAI_ErrorCallback().
    Exception done for MspInit and MspDeInit callbacks that are respectively
    reset to the legacy weak (surcharged) functions in the HAL_SAI_Init
    and HAL_SAI_DeInit only when these callbacks are null (not registered beforehand).
    If not, MspInit or MspDeInit are not null, the HAL_SAI_Init and HAL_SAI_DeInit
    keep and use the user MspInit/MspDeInit callbacks (registered beforehand).

    [..]
    Callbacks can be registered/unregistered in READY state only.
    Exception done for MspInit/MspDeInit callbacks that can be registered/unregistered
    in READY or RESET state, thus registered (user) MspInit/DeInit callbacks can be used
    during the Init/DeInit.
    In that case first register the MspInit/MspDeInit user callbacks
    using HAL_SAI_RegisterCallback before calling HAL_SAI_DeInit
    or HAL_SAI_Init function.

    [..]
    When the compilation define USE_HAL_SAI_REGISTER_CALLBACKS is set to 0 or
    not defined, the callback registering feature is not available
    and weak (surcharged) callbacks are used.

  @endverbatim
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

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup SAI SAI
  * @brief SAI HAL module driver
  * @{
  */

#ifdef HAL_SAI_MODULE_ENABLED

#if defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)

/** @defgroup SAI_Private_Typedefs  SAI Private Typedefs
  * @{
  */
typedef enum
{
  SAI_MODE_DMA,
  SAI_MODE_IT
} SAI_ModeTypedef;
/**
  * @}
  */

/* Private define ------------------------------------------------------------*/

/** @defgroup SAI_Private_Constants  SAI Private Constants
  * @{
  */
#define SAI_DEFAULT_TIMEOUT   4U /* 4ms */
#define SAI_LONG_TIMEOUT      1000U /* 1s */
/**
  * @}
  */

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/** @defgroup SAI_Private_Functions  SAI Private Functions
  * @{
  */
static void SAI_FillFifo(SAI_HandleTypeDef *hsai);
static uint32_t SAI_InterruptFlag(SAI_HandleTypeDef *hsai, uint32_t mode);
static HAL_StatusTypeDef SAI_InitI2S(SAI_HandleTypeDef *hsai, uint32_t protocol, uint32_t datasize, uint32_t nbslot);
static HAL_StatusTypeDef SAI_InitPCM(SAI_HandleTypeDef *hsai, uint32_t protocol, uint32_t datasize, uint32_t nbslot);

static HAL_StatusTypeDef SAI_Disable(SAI_HandleTypeDef *hsai);
static void SAI_Transmit_IT8Bit(SAI_HandleTypeDef *hsai);
static void SAI_Transmit_IT16Bit(SAI_HandleTypeDef *hsai);
static void SAI_Transmit_IT32Bit(SAI_HandleTypeDef *hsai);
static void SAI_Receive_IT8Bit(SAI_HandleTypeDef *hsai);
static void SAI_Receive_IT16Bit(SAI_HandleTypeDef *hsai);
static void SAI_Receive_IT32Bit(SAI_HandleTypeDef *hsai);

static void SAI_DMATxCplt(DMA_HandleTypeDef *hdma);
static void SAI_DMATxHalfCplt(DMA_HandleTypeDef *hdma);
static void SAI_DMARxCplt(DMA_HandleTypeDef *hdma);
static void SAI_DMARxHalfCplt(DMA_HandleTypeDef *hdma);
static void SAI_DMAError(DMA_HandleTypeDef *hdma);
static void SAI_DMAAbort(DMA_HandleTypeDef *hdma);
/**
  * @}
  */

/* Exported functions ---------------------------------------------------------*/
/** @defgroup SAI_Exported_Functions SAI Exported Functions
  * @{
  */

/** @defgroup SAI_Exported_Functions_Group1 Initialization and de-initialization functions
  * @brief    Initialization and Configuration functions
  *
@verbatim
 ===============================================================================
            ##### Initialization and de-initialization functions #####
 ===============================================================================
  [..]  This subsection provides a set of functions allowing to initialize and
        de-initialize the SAIx peripheral:

      (+) User must implement HAL_SAI_MspInit() function in which he configures
          all related peripherals resources (CLOCK, GPIO, DMA, IT and NVIC ).

      (+) Call the function HAL_SAI_Init() to configure the selected device with
          the selected configuration:
        (++) Mode (Master/slave TX/RX)
        (++) Protocol
        (++) Data Size
        (++) MCLK Output
        (++) Audio frequency
        (++) FIFO Threshold
        (++) Frame Config
        (++) Slot Config

      (+) Call the function HAL_SAI_DeInit() to restore the default configuration
          of the selected SAI peripheral.

@endverbatim
  * @{
  */

/**
  * @brief  Initialize the structure FrameInit, SlotInit and the low part of
  *         Init according to the specified parameters and call the function
  *         HAL_SAI_Init to initialize the SAI block.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  protocol one of the supported protocol @ref SAI_Protocol
  * @param  datasize one of the supported datasize @ref SAI_Protocol_DataSize
  *                   the configuration information for SAI module.
  * @param  nbslot Number of slot.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_InitProtocol(SAI_HandleTypeDef *hsai, uint32_t protocol, uint32_t datasize, uint32_t nbslot)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Check the parameters */
  assert_param(IS_SAI_SUPPORTED_PROTOCOL(protocol));
  assert_param(IS_SAI_PROTOCOL_DATASIZE(datasize));

  switch (protocol)
  {
    case SAI_I2S_STANDARD :
    case SAI_I2S_MSBJUSTIFIED :
    case SAI_I2S_LSBJUSTIFIED :
      status = SAI_InitI2S(hsai, protocol, datasize, nbslot);
      break;
    case SAI_PCM_LONG :
    case SAI_PCM_SHORT :
      status = SAI_InitPCM(hsai, protocol, datasize, nbslot);
      break;
    default :
      status = HAL_ERROR;
      break;
  }

  if (status == HAL_OK)
  {
    status = HAL_SAI_Init(hsai);
  }

  return status;
}

/**
  * @brief  Initialize the SAI according to the specified parameters.
  *         in the SAI_InitTypeDef structure and initialize the associated handle.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_Init(SAI_HandleTypeDef *hsai)
{
  uint32_t tmpregisterGCR = 0U;

  /* This variable used to store the SAI_CK_x (value in Hz) */
  uint32_t freq = 0U;

  /* This variable is used to compute CKSTR bits of SAI CR1 according to
     ClockStrobing and AudioMode fields */
  uint32_t ckstr_bits = 0U;
  uint32_t syncen_bits = 0U;

  /* Check the SAI handle allocation */
  if (hsai == NULL)
  {
    return HAL_ERROR;
  }

  /* check the instance */
  assert_param(IS_SAI_ALL_INSTANCE(hsai->Instance));

  /* Check the SAI Block parameters */
  assert_param(IS_SAI_AUDIO_FREQUENCY(hsai->Init.AudioFrequency));
  assert_param(IS_SAI_BLOCK_PROTOCOL(hsai->Init.Protocol));
  assert_param(IS_SAI_BLOCK_MODE(hsai->Init.AudioMode));
  assert_param(IS_SAI_BLOCK_SYNCEXT(hsai->Init.SynchroExt));
  assert_param(IS_SAI_BLOCK_DATASIZE(hsai->Init.DataSize));
  assert_param(IS_SAI_BLOCK_FIRST_BIT(hsai->Init.FirstBit));
  assert_param(IS_SAI_BLOCK_CLOCK_STROBING(hsai->Init.ClockStrobing));
  assert_param(IS_SAI_BLOCK_SYNCHRO(hsai->Init.Synchro));
  assert_param(IS_SAI_BLOCK_OUTPUT_DRIVE(hsai->Init.OutputDrive));
  assert_param(IS_SAI_BLOCK_NODIVIDER(hsai->Init.NoDivider));
  assert_param(IS_SAI_BLOCK_FIFO_THRESHOLD(hsai->Init.FIFOThreshold));
  assert_param(IS_SAI_MONO_STEREO_MODE(hsai->Init.MonoStereoMode));
  assert_param(IS_SAI_BLOCK_COMPANDING_MODE(hsai->Init.CompandingMode));
  assert_param(IS_SAI_BLOCK_TRISTATE_MANAGEMENT(hsai->Init.TriState));

  /* Check the SAI Block Frame parameters */
  assert_param(IS_SAI_BLOCK_FRAME_LENGTH(hsai->FrameInit.FrameLength));
  assert_param(IS_SAI_BLOCK_ACTIVE_FRAME(hsai->FrameInit.ActiveFrameLength));
  assert_param(IS_SAI_BLOCK_FS_DEFINITION(hsai->FrameInit.FSDefinition));
  assert_param(IS_SAI_BLOCK_FS_POLARITY(hsai->FrameInit.FSPolarity));
  assert_param(IS_SAI_BLOCK_FS_OFFSET(hsai->FrameInit.FSOffset));

  /* Check the SAI Block Slot parameters */
  assert_param(IS_SAI_BLOCK_FIRSTBIT_OFFSET(hsai->SlotInit.FirstBitOffset));
  assert_param(IS_SAI_BLOCK_SLOT_SIZE(hsai->SlotInit.SlotSize));
  assert_param(IS_SAI_BLOCK_SLOT_NUMBER(hsai->SlotInit.SlotNumber));
  assert_param(IS_SAI_SLOT_ACTIVE(hsai->SlotInit.SlotActive));

  if (hsai->State == HAL_SAI_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hsai->Lock = HAL_UNLOCKED;

#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
    /* Reset callback pointers to the weak predefined callbacks */
    hsai->RxCpltCallback     = HAL_SAI_RxCpltCallback;
    hsai->RxHalfCpltCallback = HAL_SAI_RxHalfCpltCallback;
    hsai->TxCpltCallback     = HAL_SAI_TxCpltCallback;
    hsai->TxHalfCpltCallback = HAL_SAI_TxHalfCpltCallback;
    hsai->ErrorCallback      = HAL_SAI_ErrorCallback;

    /* Init the low level hardware : GPIO, CLOCK, NVIC and DMA */
    if (hsai->MspInitCallback == NULL)
    {
      hsai->MspInitCallback = HAL_SAI_MspInit;
    }
    hsai->MspInitCallback(hsai);
#else
    /* Init the low level hardware : GPIO, CLOCK, NVIC and DMA */
    HAL_SAI_MspInit(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
  }

  hsai->State = HAL_SAI_STATE_BUSY;

  /* Disable the selected SAI peripheral */
  SAI_Disable(hsai);

  /* SAI Block Synchro Configuration -----------------------------------------*/
  SAI_BlockSynchroConfig(hsai);

  /* Configure Master Clock using the following formula :
     MCLK_x = SAI_CK_x / (MCKDIV[3:0] * 2) with MCLK_x = 256 * FS
     FS = SAI_CK_x / (MCKDIV[3:0] * 2) * 256
     MCKDIV[3:0] = SAI_CK_x / FS * 512 */
  if (hsai->Init.AudioFrequency != SAI_AUDIO_FREQUENCY_MCKDIV)
  {
    /* Get SAI clock source based on Source clock selection from RCC */
    freq = SAI_GetInputClock(hsai);

    /* (saiclocksource x 10) to keep Significant digits */
    tmpregisterGCR = (((freq * 10U) / ((hsai->Init.AudioFrequency) * 512U)));

    hsai->Init.Mckdiv = tmpregisterGCR / 10U;

    /* Round result to the nearest integer */
    if ((tmpregisterGCR % 10U) > 8U)
    {
      hsai->Init.Mckdiv += 1U;
    }

    /* For SPDIF protocol, SAI shall provide a bit clock twice faster the symbol-rate */
    if (hsai->Init.Protocol == SAI_SPDIF_PROTOCOL)
    {
      hsai->Init.Mckdiv = hsai->Init.Mckdiv >> 1;
    }
  }

  /* Check the SAI Block master clock divider parameter */
  assert_param(IS_SAI_BLOCK_MASTER_DIVIDER(hsai->Init.Mckdiv));

  /* Compute CKSTR bits of SAI CR1 according to ClockStrobing and AudioMode */
  if ((hsai->Init.AudioMode == SAI_MODEMASTER_TX) || (hsai->Init.AudioMode == SAI_MODESLAVE_TX))
  {
    ckstr_bits = (hsai->Init.ClockStrobing == SAI_CLOCKSTROBING_RISINGEDGE) ? 0U : SAI_xCR1_CKSTR;
  }
  else
  {
    ckstr_bits = (hsai->Init.ClockStrobing == SAI_CLOCKSTROBING_RISINGEDGE) ? SAI_xCR1_CKSTR : 0U;
  }

  /* SAI Block Configuration -------------------------------------------------*/
  switch (hsai->Init.Synchro)
  {
    case SAI_ASYNCHRONOUS :
    {
      syncen_bits = 0U;
    }
    break;
    case SAI_SYNCHRONOUS :
    {
      syncen_bits = SAI_xCR1_SYNCEN_0;
    }
    break;
    case SAI_SYNCHRONOUS_EXT_SAI1 :
    case SAI_SYNCHRONOUS_EXT_SAI2 :
    {
      syncen_bits = SAI_xCR1_SYNCEN_1;
    }
    break;
    default:
      break;
  }

  /* SAI CR1 Configuration */
  hsai->Instance->CR1 &= ~(SAI_xCR1_MODE | SAI_xCR1_PRTCFG |  SAI_xCR1_DS |      \
                           SAI_xCR1_LSBFIRST | SAI_xCR1_CKSTR | SAI_xCR1_SYNCEN | \
                           SAI_xCR1_MONO | SAI_xCR1_OUTDRIV  | SAI_xCR1_DMAEN |  \
                           SAI_xCR1_NODIV | SAI_xCR1_MCKDIV);

  hsai->Instance->CR1 |= (hsai->Init.AudioMode | hsai->Init.Protocol |           \
                          hsai->Init.DataSize | hsai->Init.FirstBit  |           \
                          ckstr_bits | syncen_bits |                             \
                          hsai->Init.MonoStereoMode | hsai->Init.OutputDrive |   \
                          hsai->Init.NoDivider | (hsai->Init.Mckdiv << 20U));

  /* SAI CR2 Configuration */
  hsai->Instance->CR2 &= ~(SAI_xCR2_FTH | SAI_xCR2_FFLUSH | SAI_xCR2_COMP | SAI_xCR2_CPL);
  hsai->Instance->CR2 |= (hsai->Init.FIFOThreshold | hsai->Init.CompandingMode | hsai->Init.TriState);

  /* SAI Frame Configuration -----------------------------------------*/
  hsai->Instance->FRCR &= (~(SAI_xFRCR_FRL | SAI_xFRCR_FSALL | SAI_xFRCR_FSDEF | \
                             SAI_xFRCR_FSPOL | SAI_xFRCR_FSOFF));
  hsai->Instance->FRCR |= ((hsai->FrameInit.FrameLength - 1U) |
                           hsai->FrameInit.FSOffset     |
                           hsai->FrameInit.FSDefinition |
                           hsai->FrameInit.FSPolarity   |
                           ((hsai->FrameInit.ActiveFrameLength - 1U) << 8U));

  /* SAI Block_x SLOT Configuration ------------------------------------------*/
  /* This register has no meaning in AC 97 and SPDIF audio protocol */
  hsai->Instance->SLOTR &= ~(SAI_xSLOTR_FBOFF | SAI_xSLOTR_SLOTSZ | \
                             SAI_xSLOTR_NBSLOT | SAI_xSLOTR_SLOTEN);

  hsai->Instance->SLOTR |=  hsai->SlotInit.FirstBitOffset |  hsai->SlotInit.SlotSize  | \
                            (hsai->SlotInit.SlotActive << 16U) | ((hsai->SlotInit.SlotNumber - 1U) <<  8U);

  /* Initialize the error code */
  hsai->ErrorCode = HAL_SAI_ERROR_NONE;

  /* Initialize the SAI state */
  hsai->State = HAL_SAI_STATE_READY;

  /* Release Lock */
  __HAL_UNLOCK(hsai);

  return HAL_OK;
}

/**
  * @brief  DeInitialize the SAI peripheral.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_DeInit(SAI_HandleTypeDef *hsai)
{
  /* Check the SAI handle allocation */
  if (hsai == NULL)
  {
    return HAL_ERROR;
  }

  hsai->State = HAL_SAI_STATE_BUSY;

  /* Disabled All interrupt and clear all the flag */
  hsai->Instance->IMR = 0U;
  hsai->Instance->CLRFR = 0xFFFFFFFFU;

  /* Disable the SAI */
  SAI_Disable(hsai);

  /* Flush the fifo */
  SET_BIT(hsai->Instance->CR2, SAI_xCR2_FFLUSH);

  /* DeInit the low level hardware: GPIO, CLOCK, NVIC and DMA */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
  if (hsai->MspDeInitCallback == NULL)
  {
    hsai->MspDeInitCallback = HAL_SAI_MspDeInit;
  }
  hsai->MspDeInitCallback(hsai);
#else
  HAL_SAI_MspDeInit(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */

  /* Initialize the error code */
  hsai->ErrorCode = HAL_SAI_ERROR_NONE;

  /* Initialize the SAI state */
  hsai->State = HAL_SAI_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(hsai);

  return HAL_OK;
}

/**
  * @brief Initialize the SAI MSP.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
__weak void HAL_SAI_MspInit(SAI_HandleTypeDef *hsai)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsai);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SAI_MspInit could be implemented in the user file
   */
}

/**
  * @brief DeInitialize the SAI MSP.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
__weak void HAL_SAI_MspDeInit(SAI_HandleTypeDef *hsai)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsai);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SAI_MspDeInit could be implemented in the user file
   */
}

#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a user SAI callback
  *         to be used instead of the weak predefined callback.
  * @param  hsai SAI handle.
  * @param  CallbackID ID of the callback to be registered.
  *         This parameter can be one of the following values:
  *           @arg @ref HAL_SAI_RX_COMPLETE_CB_ID receive complete callback ID.
  *           @arg @ref HAL_SAI_RX_HALFCOMPLETE_CB_ID receive half complete callback ID.
  *           @arg @ref HAL_SAI_TX_COMPLETE_CB_ID transmit complete callback ID.
  *           @arg @ref HAL_SAI_TX_HALFCOMPLETE_CB_ID transmit half complete callback ID.
  *           @arg @ref HAL_SAI_ERROR_CB_ID error callback ID.
  *           @arg @ref HAL_SAI_MSPINIT_CB_ID MSP init callback ID.
  *           @arg @ref HAL_SAI_MSPDEINIT_CB_ID MSP de-init callback ID.
  * @param  pCallback pointer to the callback function.
  * @retval HAL status.
  */
HAL_StatusTypeDef HAL_SAI_RegisterCallback(SAI_HandleTypeDef        *hsai,
                                           HAL_SAI_CallbackIDTypeDef CallbackID,
                                           pSAI_CallbackTypeDef      pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* update the error code */
    hsai->ErrorCode |= HAL_SAI_ERROR_INVALID_CALLBACK;
    /* update return status */
    status = HAL_ERROR;
  }
  else
  {
    if (HAL_SAI_STATE_READY == hsai->State)
    {
      switch (CallbackID)
      {
        case HAL_SAI_RX_COMPLETE_CB_ID :
          hsai->RxCpltCallback = pCallback;
          break;
        case HAL_SAI_RX_HALFCOMPLETE_CB_ID :
          hsai->RxHalfCpltCallback = pCallback;
          break;
        case HAL_SAI_TX_COMPLETE_CB_ID :
          hsai->TxCpltCallback = pCallback;
          break;
        case HAL_SAI_TX_HALFCOMPLETE_CB_ID :
          hsai->TxHalfCpltCallback = pCallback;
          break;
        case HAL_SAI_ERROR_CB_ID :
          hsai->ErrorCallback = pCallback;
          break;
        case HAL_SAI_MSPINIT_CB_ID :
          hsai->MspInitCallback = pCallback;
          break;
        case HAL_SAI_MSPDEINIT_CB_ID :
          hsai->MspDeInitCallback = pCallback;
          break;
        default :
          /* update the error code */
          hsai->ErrorCode |= HAL_SAI_ERROR_INVALID_CALLBACK;
          /* update return status */
          status = HAL_ERROR;
          break;
      }
    }
    else if (HAL_SAI_STATE_RESET == hsai->State)
    {
      switch (CallbackID)
      {
        case HAL_SAI_MSPINIT_CB_ID :
          hsai->MspInitCallback = pCallback;
          break;
        case HAL_SAI_MSPDEINIT_CB_ID :
          hsai->MspDeInitCallback = pCallback;
          break;
        default :
          /* update the error code */
          hsai->ErrorCode |= HAL_SAI_ERROR_INVALID_CALLBACK;
          /* update return status */
          status = HAL_ERROR;
          break;
      }
    }
    else
    {
      /* update the error code */
      hsai->ErrorCode |= HAL_SAI_ERROR_INVALID_CALLBACK;
      /* update return status */
      status = HAL_ERROR;
    }
  }
  return status;
}

/**
  * @brief  Unregister a user SAI callback.
  *         SAI callback is redirected to the weak predefined callback.
  * @param  hsai SAI handle.
  * @param  CallbackID ID of the callback to be unregistered.
  *         This parameter can be one of the following values:
  *           @arg @ref HAL_SAI_RX_COMPLETE_CB_ID receive complete callback ID.
  *           @arg @ref HAL_SAI_RX_HALFCOMPLETE_CB_ID receive half complete callback ID.
  *           @arg @ref HAL_SAI_TX_COMPLETE_CB_ID transmit complete callback ID.
  *           @arg @ref HAL_SAI_TX_HALFCOMPLETE_CB_ID transmit half complete callback ID.
  *           @arg @ref HAL_SAI_ERROR_CB_ID error callback ID.
  *           @arg @ref HAL_SAI_MSPINIT_CB_ID MSP init callback ID.
  *           @arg @ref HAL_SAI_MSPDEINIT_CB_ID MSP de-init callback ID.
  * @retval HAL status.
  */
HAL_StatusTypeDef HAL_SAI_UnRegisterCallback(SAI_HandleTypeDef        *hsai,
                                             HAL_SAI_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (HAL_SAI_STATE_READY == hsai->State)
  {
    switch (CallbackID)
    {
      case HAL_SAI_RX_COMPLETE_CB_ID :
        hsai->RxCpltCallback = HAL_SAI_RxCpltCallback;
        break;
      case HAL_SAI_RX_HALFCOMPLETE_CB_ID :
        hsai->RxHalfCpltCallback = HAL_SAI_RxHalfCpltCallback;
        break;
      case HAL_SAI_TX_COMPLETE_CB_ID :
        hsai->TxCpltCallback = HAL_SAI_TxCpltCallback;
        break;
      case HAL_SAI_TX_HALFCOMPLETE_CB_ID :
        hsai->TxHalfCpltCallback = HAL_SAI_TxHalfCpltCallback;
        break;
      case HAL_SAI_ERROR_CB_ID :
        hsai->ErrorCallback = HAL_SAI_ErrorCallback;
        break;
      case HAL_SAI_MSPINIT_CB_ID :
        hsai->MspInitCallback = HAL_SAI_MspInit;
        break;
      case HAL_SAI_MSPDEINIT_CB_ID :
        hsai->MspDeInitCallback = HAL_SAI_MspDeInit;
        break;
      default :
        /* update the error code */
        hsai->ErrorCode |= HAL_SAI_ERROR_INVALID_CALLBACK;
        /* update return status */
        status = HAL_ERROR;
        break;
    }
  }
  else if (HAL_SAI_STATE_RESET == hsai->State)
  {
    switch (CallbackID)
    {
      case HAL_SAI_MSPINIT_CB_ID :
        hsai->MspInitCallback = HAL_SAI_MspInit;
        break;
      case HAL_SAI_MSPDEINIT_CB_ID :
        hsai->MspDeInitCallback = HAL_SAI_MspDeInit;
        break;
      default :
        /* update the error code */
        hsai->ErrorCode |= HAL_SAI_ERROR_INVALID_CALLBACK;
        /* update return status */
        status = HAL_ERROR;
        break;
    }
  }
  else
  {
    /* update the error code */
    hsai->ErrorCode |= HAL_SAI_ERROR_INVALID_CALLBACK;
    /* update return status */
    status = HAL_ERROR;
  }
  return status;
}
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */

/**
  * @}
  */

/** @defgroup SAI_Exported_Functions_Group2 IO operation functions
  * @brief    Data transfers functions
  *
@verbatim
  ==============================================================================
                      ##### IO operation functions #####
  ==============================================================================
  [..]
    This subsection provides a set of functions allowing to manage the SAI data
    transfers.

    (+) There are two modes of transfer:
      (++) Blocking mode : The communication is performed in the polling mode.
           The status of all data processing is returned by the same function
           after finishing transfer.
      (++) No-Blocking mode : The communication is performed using Interrupts
           or DMA. These functions return the status of the transfer startup.
           The end of the data processing will be indicated through the
           dedicated SAI IRQ when using Interrupt mode or the DMA IRQ when
           using DMA mode.

    (+) Blocking mode functions are :
      (++) HAL_SAI_Transmit()
      (++) HAL_SAI_Receive()

    (+) Non Blocking mode functions with Interrupt are :
      (++) HAL_SAI_Transmit_IT()
      (++) HAL_SAI_Receive_IT()

    (+) Non Blocking mode functions with DMA are :
      (++) HAL_SAI_Transmit_DMA()
      (++) HAL_SAI_Receive_DMA()

    (+) A set of Transfer Complete Callbacks are provided in non Blocking mode:
      (++) HAL_SAI_TxCpltCallback()
      (++) HAL_SAI_RxCpltCallback()
      (++) HAL_SAI_ErrorCallback()

@endverbatim
  * @{
  */

/**
  * @brief  Transmit an amount of data in blocking mode.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_Transmit(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size, uint32_t Timeout)
{
  uint32_t tickstart = HAL_GetTick();

  if ((pData == NULL) || (Size == 0))
  {
    return  HAL_ERROR;
  }

  if (hsai->State == HAL_SAI_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hsai);

    hsai->XferSize = Size;
    hsai->XferCount = Size;
    hsai->pBuffPtr = pData;
    hsai->State = HAL_SAI_STATE_BUSY_TX;
    hsai->ErrorCode = HAL_SAI_ERROR_NONE;

    /* Check if the SAI is already enabled */
    if ((hsai->Instance->CR1 & SAI_xCR1_SAIEN) == RESET)
    {
      /* fill the fifo with data before to enabled the SAI */
      SAI_FillFifo(hsai);
      /* Enable SAI peripheral */
      __HAL_SAI_ENABLE(hsai);
    }

    while (hsai->XferCount > 0U)
    {
      /* Write data if the FIFO is not full */
      if ((hsai->Instance->SR & SAI_xSR_FLVL) != SAI_FIFOSTATUS_FULL)
      {
        if ((hsai->Init.DataSize == SAI_DATASIZE_8) && (hsai->Init.CompandingMode == SAI_NOCOMPANDING))
        {
          hsai->Instance->DR = (*hsai->pBuffPtr++);
        }
        else if (hsai->Init.DataSize <= SAI_DATASIZE_16)
        {
          hsai->Instance->DR = *((uint16_t *)hsai->pBuffPtr);
          hsai->pBuffPtr += 2U;
        }
        else
        {
          hsai->Instance->DR = *((uint32_t *)hsai->pBuffPtr);
          hsai->pBuffPtr += 4U;
        }
        hsai->XferCount--;
      }
      else
      {
        /* Check for the Timeout */
        if ((Timeout != HAL_MAX_DELAY) && ((Timeout == 0U) || ((HAL_GetTick() - tickstart) > Timeout)))
        {
          /* Update error code */
          hsai->ErrorCode |= HAL_SAI_ERROR_TIMEOUT;

          /* Clear all the flags */
          hsai->Instance->CLRFR = 0xFFFFFFFFU;

          /* Disable SAI peripheral */
          SAI_Disable(hsai);

          /* Flush the fifo */
          SET_BIT(hsai->Instance->CR2, SAI_xCR2_FFLUSH);

          /* Change the SAI state */
          hsai->State = HAL_SAI_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hsai);

          return HAL_ERROR;
        }
      }
    }

    hsai->State = HAL_SAI_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hsai);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Receive an amount of data in blocking mode.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be received
  * @param  Timeout Timeout duration
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_Receive(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size, uint32_t Timeout)
{
  uint32_t tickstart = HAL_GetTick();

  if ((pData == NULL) || (Size == 0))
  {
    return  HAL_ERROR;
  }

  if (hsai->State == HAL_SAI_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hsai);

    hsai->pBuffPtr = pData;
    hsai->XferSize = Size;
    hsai->XferCount = Size;
    hsai->State = HAL_SAI_STATE_BUSY_RX;
    hsai->ErrorCode = HAL_SAI_ERROR_NONE;

    /* Check if the SAI is already enabled */
    if ((hsai->Instance->CR1 & SAI_xCR1_SAIEN) == RESET)
    {
      /* Enable SAI peripheral */
      __HAL_SAI_ENABLE(hsai);
    }

    /* Receive data */
    while (hsai->XferCount > 0U)
    {
      if ((hsai->Instance->SR & SAI_xSR_FLVL) != SAI_FIFOSTATUS_EMPTY)
      {
        if ((hsai->Init.DataSize == SAI_DATASIZE_8) && (hsai->Init.CompandingMode == SAI_NOCOMPANDING))
        {
          (*hsai->pBuffPtr++) = hsai->Instance->DR;
        }
        else if (hsai->Init.DataSize <= SAI_DATASIZE_16)
        {
          *((uint16_t *)hsai->pBuffPtr) = hsai->Instance->DR;
          hsai->pBuffPtr += 2U;
        }
        else
        {
          *((uint32_t *)hsai->pBuffPtr) = hsai->Instance->DR;
          hsai->pBuffPtr += 4U;
        }
        hsai->XferCount--;
      }
      else
      {
        /* Check for the Timeout */
        if ((Timeout != HAL_MAX_DELAY) && ((Timeout == 0U) || ((HAL_GetTick() - tickstart) > Timeout)))
        {
          /* Update error code */
          hsai->ErrorCode |= HAL_SAI_ERROR_TIMEOUT;

          /* Clear all the flags */
          hsai->Instance->CLRFR = 0xFFFFFFFFU;

          /* Disable SAI peripheral */
          SAI_Disable(hsai);

          /* Flush the fifo */
          SET_BIT(hsai->Instance->CR2, SAI_xCR2_FFLUSH);

          /* Change the SAI state */
          hsai->State = HAL_SAI_STATE_READY;

          /* Process Unlocked */
          __HAL_UNLOCK(hsai);

          return HAL_ERROR;
        }
      }
    }

    hsai->State = HAL_SAI_STATE_READY;

    /* Process Unlocked */
    __HAL_UNLOCK(hsai);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Transmit an amount of data in non-blocking mode with Interrupt.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_Transmit_IT(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size)
{
  if ((pData == NULL) || (Size == 0))
  {
    return  HAL_ERROR;
  }

  if (hsai->State == HAL_SAI_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hsai);

    hsai->pBuffPtr = pData;
    hsai->XferSize = Size;
    hsai->XferCount = Size;
    hsai->ErrorCode = HAL_SAI_ERROR_NONE;
    hsai->State = HAL_SAI_STATE_BUSY_TX;

    if ((hsai->Init.DataSize == SAI_DATASIZE_8) && (hsai->Init.CompandingMode == SAI_NOCOMPANDING))
    {
      hsai->InterruptServiceRoutine = SAI_Transmit_IT8Bit;
    }
    else if (hsai->Init.DataSize <= SAI_DATASIZE_16)
    {
      hsai->InterruptServiceRoutine = SAI_Transmit_IT16Bit;
    }
    else
    {
      hsai->InterruptServiceRoutine = SAI_Transmit_IT32Bit;
    }

    /* Fill the fifo before starting the communication */
    SAI_FillFifo(hsai);

    /* Enable FRQ and OVRUDR interrupts */
    __HAL_SAI_ENABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_IT));

    /* Check if the SAI is already enabled */
    if ((hsai->Instance->CR1 & SAI_xCR1_SAIEN) == RESET)
    {
      /* Enable SAI peripheral */
      __HAL_SAI_ENABLE(hsai);
    }
    /* Process Unlocked */
    __HAL_UNLOCK(hsai);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Receive an amount of data in non-blocking mode with Interrupt.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be received
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_Receive_IT(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size)
{
  if ((pData == NULL) || (Size == 0))
  {
    return  HAL_ERROR;
  }

  if (hsai->State == HAL_SAI_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hsai);

    hsai->pBuffPtr = pData;
    hsai->XferSize = Size;
    hsai->XferCount = Size;
    hsai->ErrorCode = HAL_SAI_ERROR_NONE;
    hsai->State = HAL_SAI_STATE_BUSY_RX;

    if ((hsai->Init.DataSize == SAI_DATASIZE_8) && (hsai->Init.CompandingMode == SAI_NOCOMPANDING))
    {
      hsai->InterruptServiceRoutine = SAI_Receive_IT8Bit;
    }
    else if (hsai->Init.DataSize <= SAI_DATASIZE_16)
    {
      hsai->InterruptServiceRoutine = SAI_Receive_IT16Bit;
    }
    else
    {
      hsai->InterruptServiceRoutine = SAI_Receive_IT32Bit;
    }

    /* Enable TXE and OVRUDR interrupts */
    __HAL_SAI_ENABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_IT));

    /* Check if the SAI is already enabled */
    if ((hsai->Instance->CR1 & SAI_xCR1_SAIEN) == RESET)
    {
      /* Enable SAI peripheral */
      __HAL_SAI_ENABLE(hsai);
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hsai);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief Pause the audio stream playing from the Media.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_DMAPause(SAI_HandleTypeDef *hsai)
{
  /* Process Locked */
  __HAL_LOCK(hsai);

  /* Pause the audio file playing by disabling the SAI DMA requests */
  hsai->Instance->CR1 &= ~SAI_xCR1_DMAEN;

  /* Process Unlocked */
  __HAL_UNLOCK(hsai);

  return HAL_OK;
}

/**
  * @brief Resume the audio stream playing from the Media.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_DMAResume(SAI_HandleTypeDef *hsai)
{
  /* Process Locked */
  __HAL_LOCK(hsai);

  /* Enable the SAI DMA requests */
  hsai->Instance->CR1 |= SAI_xCR1_DMAEN;

  /* If the SAI peripheral is still not enabled, enable it */
  if ((hsai->Instance->CR1 & SAI_xCR1_SAIEN) == RESET)
  {
    /* Enable SAI peripheral */
    __HAL_SAI_ENABLE(hsai);
  }

  /* Process Unlocked */
  __HAL_UNLOCK(hsai);

  return HAL_OK;
}

/**
  * @brief Stop the audio stream playing from the Media.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_DMAStop(SAI_HandleTypeDef *hsai)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process Locked */
  __HAL_LOCK(hsai);

  /* Disable the SAI DMA request */
  hsai->Instance->CR1 &= ~SAI_xCR1_DMAEN;

  /* Abort the SAI Tx DMA Stream */
  if ((hsai->hdmatx != NULL) && (hsai->State == HAL_SAI_STATE_BUSY_TX))
  {
    if (HAL_DMA_Abort(hsai->hdmatx) != HAL_OK)
    {
      /* If the DMA Tx errorCode is different from DMA No Transfer then return Error */
      if (hsai->hdmatx->ErrorCode != HAL_DMA_ERROR_NO_XFER)
      {
        status = HAL_ERROR;
        hsai->ErrorCode |= HAL_SAI_ERROR_DMA;
      }
    }
  }

  /* Abort the SAI Rx DMA Stream */
  if ((hsai->hdmarx != NULL) && (hsai->State == HAL_SAI_STATE_BUSY_RX))
  {
    if (HAL_DMA_Abort(hsai->hdmarx) != HAL_OK)
    {
      /* If the DMA Rx errorCode is different from DMA No Transfer then return Error */
      if (hsai->hdmarx->ErrorCode != HAL_DMA_ERROR_NO_XFER)
      {
        status = HAL_ERROR;
        hsai->ErrorCode |= HAL_SAI_ERROR_DMA;
      }
    }
  }

  /* Disable SAI peripheral */
  SAI_Disable(hsai);

  /* Flush the fifo */
  SET_BIT(hsai->Instance->CR2, SAI_xCR2_FFLUSH);

  /* Set hsai state to ready */
  hsai->State = HAL_SAI_STATE_READY;

  /* Process Unlocked */
  __HAL_UNLOCK(hsai);

  return status;
}

/**
  * @brief Abort the current transfer and disable the SAI.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_Abort(SAI_HandleTypeDef *hsai)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process Locked */
  __HAL_LOCK(hsai);

  /* Check SAI DMA is enabled or not */
  if ((hsai->Instance->CR1 & SAI_xCR1_DMAEN) == SAI_xCR1_DMAEN)
  {
    /* Disable the SAI DMA request */
    hsai->Instance->CR1 &= ~SAI_xCR1_DMAEN;

    /* Abort the SAI Tx DMA Stream */
    if ((hsai->hdmatx != NULL) && (hsai->State == HAL_SAI_STATE_BUSY_TX))
    {
      if (HAL_DMA_Abort(hsai->hdmatx) != HAL_OK)
      {
        /* If the DMA Tx errorCode is different from DMA No Transfer then return Error */
        if (hsai->hdmatx->ErrorCode != HAL_DMA_ERROR_NO_XFER)
        {
          status = HAL_ERROR;
          hsai->ErrorCode |= HAL_SAI_ERROR_DMA;
        }
      }
    }

    /* Abort the SAI Rx DMA Stream */
    if ((hsai->hdmarx != NULL) && (hsai->State == HAL_SAI_STATE_BUSY_RX))
    {
      if (HAL_DMA_Abort(hsai->hdmarx) != HAL_OK)
      {
        /* If the DMA Rx errorCode is different from DMA No Transfer then return Error */
        if (hsai->hdmarx->ErrorCode != HAL_DMA_ERROR_NO_XFER)
        {
          status = HAL_ERROR;
          hsai->ErrorCode |= HAL_SAI_ERROR_DMA;
        }
      }
    }
  }

  /* Disabled All interrupt and clear all the flag */
  hsai->Instance->IMR = 0U;
  hsai->Instance->CLRFR = 0xFFFFFFFFU;

  /* Disable SAI peripheral */
  SAI_Disable(hsai);

  /* Flush the fifo */
  SET_BIT(hsai->Instance->CR2, SAI_xCR2_FFLUSH);

  /* Set hsai state to ready */
  hsai->State = HAL_SAI_STATE_READY;

  /* Process Unlocked */
  __HAL_UNLOCK(hsai);

  return status;
}

/**
  * @brief  Transmit an amount of data in non-blocking mode with DMA.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be sent
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_Transmit_DMA(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size)
{
  uint32_t tickstart = HAL_GetTick();

  if ((pData == NULL) || (Size == 0))
  {
    return  HAL_ERROR;
  }

  if (hsai->State == HAL_SAI_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hsai);

    hsai->pBuffPtr = pData;
    hsai->XferSize = Size;
    hsai->XferCount = Size;
    hsai->ErrorCode = HAL_SAI_ERROR_NONE;
    hsai->State = HAL_SAI_STATE_BUSY_TX;

    /* Set the SAI Tx DMA Half transfer complete callback */
    hsai->hdmatx->XferHalfCpltCallback = SAI_DMATxHalfCplt;

    /* Set the SAI TxDMA transfer complete callback */
    hsai->hdmatx->XferCpltCallback = SAI_DMATxCplt;

    /* Set the DMA error callback */
    hsai->hdmatx->XferErrorCallback = SAI_DMAError;

    /* Set the DMA Tx abort callback */
    hsai->hdmatx->XferAbortCallback = NULL;

    /* Enable the Tx DMA Stream */
    if (HAL_DMA_Start_IT(hsai->hdmatx, (uint32_t)hsai->pBuffPtr, (uint32_t)&hsai->Instance->DR, hsai->XferSize) != HAL_OK)
    {
      __HAL_UNLOCK(hsai);
      return  HAL_ERROR;
    }

    /* Enable the interrupts for error handling */
    __HAL_SAI_ENABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_DMA));

    /* Enable SAI Tx DMA Request */
    hsai->Instance->CR1 |= SAI_xCR1_DMAEN;

    /* Wait until FIFO is not empty */
    while ((hsai->Instance->SR & SAI_xSR_FLVL) == SAI_FIFOSTATUS_EMPTY)
    {
      /* Check for the Timeout */
      if ((HAL_GetTick() - tickstart) > SAI_LONG_TIMEOUT)
      {
        /* Update error code */
        hsai->ErrorCode |= HAL_SAI_ERROR_TIMEOUT;

        /* Process Unlocked */
        __HAL_UNLOCK(hsai);

        return HAL_TIMEOUT;
      }
    }

    /* Check if the SAI is already enabled */
    if ((hsai->Instance->CR1 & SAI_xCR1_SAIEN) == 0U)
    {
      /* Enable SAI peripheral */
      __HAL_SAI_ENABLE(hsai);
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hsai);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Receive an amount of data in non-blocking mode with DMA.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  pData Pointer to data buffer
  * @param  Size Amount of data to be received
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_Receive_DMA(SAI_HandleTypeDef *hsai, uint8_t *pData, uint16_t Size)
{
  if ((pData == NULL) || (Size == 0))
  {
    return  HAL_ERROR;
  }

  if (hsai->State == HAL_SAI_STATE_READY)
  {
    /* Process Locked */
    __HAL_LOCK(hsai);

    hsai->pBuffPtr = pData;
    hsai->XferSize = Size;
    hsai->XferCount = Size;
    hsai->ErrorCode = HAL_SAI_ERROR_NONE;
    hsai->State = HAL_SAI_STATE_BUSY_RX;

    /* Set the SAI Rx DMA Half transfer complete callback */
    hsai->hdmarx->XferHalfCpltCallback = SAI_DMARxHalfCplt;

    /* Set the SAI Rx DMA transfer complete callback */
    hsai->hdmarx->XferCpltCallback = SAI_DMARxCplt;

    /* Set the DMA error callback */
    hsai->hdmarx->XferErrorCallback = SAI_DMAError;

    /* Set the DMA Rx abort callback */
    hsai->hdmarx->XferAbortCallback = NULL;

    /* Enable the Rx DMA Stream */
    if (HAL_DMA_Start_IT(hsai->hdmarx, (uint32_t)&hsai->Instance->DR, (uint32_t)hsai->pBuffPtr, hsai->XferSize) != HAL_OK)
    {
      __HAL_UNLOCK(hsai);
      return  HAL_ERROR;
    }

    /* Enable the interrupts for error handling */
    __HAL_SAI_ENABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_DMA));

    /* Enable SAI Rx DMA Request */
    hsai->Instance->CR1 |= SAI_xCR1_DMAEN;

    /* Check if the SAI is already enabled */
    if ((hsai->Instance->CR1 & SAI_xCR1_SAIEN) == RESET)
    {
      /* Enable SAI peripheral */
      __HAL_SAI_ENABLE(hsai);
    }

    /* Process Unlocked */
    __HAL_UNLOCK(hsai);

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Enable the Tx mute mode.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  val  value sent during the mute @ref SAI_Block_Mute_Value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_EnableTxMuteMode(SAI_HandleTypeDef *hsai, uint16_t val)
{
  assert_param(IS_SAI_BLOCK_MUTE_VALUE(val));

  if (hsai->State != HAL_SAI_STATE_RESET)
  {
    CLEAR_BIT(hsai->Instance->CR2, SAI_xCR2_MUTEVAL | SAI_xCR2_MUTE);
    SET_BIT(hsai->Instance->CR2, SAI_xCR2_MUTE | val);
    return HAL_OK;
  }
  return HAL_ERROR;
}

/**
  * @brief  Disable the Tx mute mode.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_DisableTxMuteMode(SAI_HandleTypeDef *hsai)
{
  if (hsai->State != HAL_SAI_STATE_RESET)
  {
    CLEAR_BIT(hsai->Instance->CR2, SAI_xCR2_MUTEVAL | SAI_xCR2_MUTE);
    return HAL_OK;
  }
  return HAL_ERROR;
}

/**
  * @brief  Enable the Rx mute detection.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  callback function called when the mute is detected.
  * @param  counter number a data before mute detection max 63.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_EnableRxMuteMode(SAI_HandleTypeDef *hsai, SAIcallback callback, uint16_t counter)
{
  assert_param(IS_SAI_BLOCK_MUTE_COUNTER(counter));

  if (hsai->State != HAL_SAI_STATE_RESET)
  {
    /* set the mute counter */
    CLEAR_BIT(hsai->Instance->CR2, SAI_xCR2_MUTECNT);
    SET_BIT(hsai->Instance->CR2, (uint32_t)((uint32_t)counter << SAI_xCR2_MUTECNT_Pos));
    hsai->mutecallback = callback;
    /* enable the IT interrupt */
    __HAL_SAI_ENABLE_IT(hsai, SAI_IT_MUTEDET);
    return HAL_OK;
  }
  return HAL_ERROR;
}

/**
  * @brief  Disable the Rx mute detection.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_SAI_DisableRxMuteMode(SAI_HandleTypeDef *hsai)
{
  if (hsai->State != HAL_SAI_STATE_RESET)
  {
    /* set the mutecallback to NULL */
    hsai->mutecallback = (SAIcallback)NULL;
    /* enable the IT interrupt */
    __HAL_SAI_DISABLE_IT(hsai, SAI_IT_MUTEDET);
    return HAL_OK;
  }
  return HAL_ERROR;
}

/**
  * @brief  Handle SAI interrupt request.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
void HAL_SAI_IRQHandler(SAI_HandleTypeDef *hsai)
{
  if (hsai->State != HAL_SAI_STATE_RESET)
  {
    uint32_t itflags = hsai->Instance->SR;
    uint32_t itsources = hsai->Instance->IMR;
    uint32_t cr1config = hsai->Instance->CR1;
    uint32_t tmperror;

    /* SAI Fifo request interrupt occurred ------------------------------------*/
    if (((itflags & SAI_xSR_FREQ) == SAI_xSR_FREQ) && ((itsources & SAI_IT_FREQ) == SAI_IT_FREQ))
    {
      hsai->InterruptServiceRoutine(hsai);
    }
    /* SAI Overrun error interrupt occurred ----------------------------------*/
    else if (((itflags & SAI_FLAG_OVRUDR) == SAI_FLAG_OVRUDR) && ((itsources & SAI_IT_OVRUDR) == SAI_IT_OVRUDR))
    {
      /* Clear the SAI Overrun flag */
      __HAL_SAI_CLEAR_FLAG(hsai, SAI_FLAG_OVRUDR);

      /* Get the SAI error code */
      tmperror = ((hsai->State == HAL_SAI_STATE_BUSY_RX) ? HAL_SAI_ERROR_OVR : HAL_SAI_ERROR_UDR);

      /* Change the SAI error code */
      hsai->ErrorCode |= tmperror;

      /* the transfer is not stopped, we will forward the information to the user and we let the user decide what needs to be done */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
      hsai->ErrorCallback(hsai);
#else
      HAL_SAI_ErrorCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
    }
    /* SAI mutedet interrupt occurred ----------------------------------*/
    else if (((itflags & SAI_FLAG_MUTEDET) == SAI_FLAG_MUTEDET) && ((itsources & SAI_IT_MUTEDET) == SAI_IT_MUTEDET))
    {
      /* Clear the SAI mutedet flag */
      __HAL_SAI_CLEAR_FLAG(hsai, SAI_FLAG_MUTEDET);

      /* call the call back function */
      if (hsai->mutecallback != (SAIcallback)NULL)
      {
        /* inform the user that an RX mute event has been detected */
        hsai->mutecallback();
      }
    }
    /* SAI AFSDET interrupt occurred ----------------------------------*/
    else if (((itflags & SAI_FLAG_AFSDET) == SAI_FLAG_AFSDET) && ((itsources & SAI_IT_AFSDET) == SAI_IT_AFSDET))
    {
      /* Clear the SAI AFSDET flag */
      __HAL_SAI_CLEAR_FLAG(hsai, SAI_FLAG_AFSDET);

      /* Change the SAI error code */
      hsai->ErrorCode |= HAL_SAI_ERROR_AFSDET;

      /* Check SAI DMA is enabled or not */
      if ((cr1config & SAI_xCR1_DMAEN) == SAI_xCR1_DMAEN)
      {
        /* Abort the SAI DMA Streams */
        if (hsai->hdmatx != NULL)
        {
          /* Set the DMA Tx abort callback */
          hsai->hdmatx->XferAbortCallback = SAI_DMAAbort;

          /* Abort DMA in IT mode */
          HAL_DMA_Abort_IT(hsai->hdmatx);
        }
        else if (hsai->hdmarx != NULL)
        {
          /* Set the DMA Rx abort callback */
          hsai->hdmarx->XferAbortCallback = SAI_DMAAbort;

          /* Abort DMA in IT mode */
          HAL_DMA_Abort_IT(hsai->hdmarx);
        }
      }
      else
      {
        /* Abort SAI */
        HAL_SAI_Abort(hsai);

        /* Set error callback */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
        hsai->ErrorCallback(hsai);
#else
        HAL_SAI_ErrorCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
      }
    }
    /* SAI LFSDET interrupt occurred ----------------------------------*/
    else if (((itflags & SAI_FLAG_LFSDET) == SAI_FLAG_LFSDET) && ((itsources & SAI_IT_LFSDET) == SAI_IT_LFSDET))
    {
      /* Clear the SAI LFSDET flag */
      __HAL_SAI_CLEAR_FLAG(hsai, SAI_FLAG_LFSDET);

      /* Change the SAI error code */
      hsai->ErrorCode |= HAL_SAI_ERROR_LFSDET;

      /* Check SAI DMA is enabled or not */
      if ((cr1config & SAI_xCR1_DMAEN) == SAI_xCR1_DMAEN)
      {
        /* Abort the SAI DMA Streams */
        if (hsai->hdmatx != NULL)
        {
          /* Set the DMA Tx abort callback */
          hsai->hdmatx->XferAbortCallback = SAI_DMAAbort;

          /* Abort DMA in IT mode */
          HAL_DMA_Abort_IT(hsai->hdmatx);
        }
        else if (hsai->hdmarx != NULL)
        {
          /* Set the DMA Rx abort callback */
          hsai->hdmarx->XferAbortCallback = SAI_DMAAbort;

          /* Abort DMA in IT mode */
          HAL_DMA_Abort_IT(hsai->hdmarx);
        }
      }
      else
      {
        /* Abort SAI */
        HAL_SAI_Abort(hsai);

        /* Set error callback */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
        hsai->ErrorCallback(hsai);
#else
        HAL_SAI_ErrorCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
      }
    }
    /* SAI WCKCFG interrupt occurred ----------------------------------*/
    else if (((itflags & SAI_FLAG_WCKCFG) == SAI_FLAG_WCKCFG) && ((itsources & SAI_IT_WCKCFG) == SAI_IT_WCKCFG))
    {
      /* Clear the SAI WCKCFG flag */
      __HAL_SAI_CLEAR_FLAG(hsai, SAI_FLAG_WCKCFG);

      /* Change the SAI error code */
      hsai->ErrorCode |= HAL_SAI_ERROR_WCKCFG;

      /* Check SAI DMA is enabled or not */
      if ((cr1config & SAI_xCR1_DMAEN) == SAI_xCR1_DMAEN)
      {
        /* Abort the SAI DMA Streams */
        if (hsai->hdmatx != NULL)
        {
          /* Set the DMA Tx abort callback */
          hsai->hdmatx->XferAbortCallback = SAI_DMAAbort;

          /* Abort DMA in IT mode */
          HAL_DMA_Abort_IT(hsai->hdmatx);
        }
        else if (hsai->hdmarx != NULL)
        {
          /* Set the DMA Rx abort callback */
          hsai->hdmarx->XferAbortCallback = SAI_DMAAbort;

          /* Abort DMA in IT mode */
          HAL_DMA_Abort_IT(hsai->hdmarx);
        }
      }
      else
      {
        /* If WCKCFG occurs, SAI audio block is automatically disabled */
        /* Disable all interrupts and clear all flags */
        hsai->Instance->IMR = 0U;
        hsai->Instance->CLRFR = 0xFFFFFFFFU;

        /* Set the SAI state to ready to be able to start again the process */
        hsai->State = HAL_SAI_STATE_READY;

        /* Initialize XferCount */
        hsai->XferCount = 0U;

        /* SAI error Callback */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
        hsai->ErrorCallback(hsai);
#else
        HAL_SAI_ErrorCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
      }
    }
    /* SAI CNRDY interrupt occurred ----------------------------------*/
    else if (((itflags & SAI_FLAG_CNRDY) == SAI_FLAG_CNRDY) && ((itsources & SAI_IT_CNRDY) == SAI_IT_CNRDY))
    {
      /* Clear the SAI CNRDY flag */
      __HAL_SAI_CLEAR_FLAG(hsai, SAI_FLAG_CNRDY);

      /* Change the SAI error code */
      hsai->ErrorCode |= HAL_SAI_ERROR_CNREADY;

      /* the transfer is not stopped, we will forward the information to the user and we let the user decide what needs to be done */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
      hsai->ErrorCallback(hsai);
#else
      HAL_SAI_ErrorCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
    }
    else
    {
      /* Nothing to do */
    }
  }
}

/**
  * @brief Tx Transfer completed callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *                the configuration information for SAI module.
  * @retval None
  */
__weak void HAL_SAI_TxCpltCallback(SAI_HandleTypeDef *hsai)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsai);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SAI_TxCpltCallback could be implemented in the user file
   */
}

/**
  * @brief Tx Transfer Half completed callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
__weak void HAL_SAI_TxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsai);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SAI_TxHalfCpltCallback could be implemented in the user file
   */
}

/**
  * @brief Rx Transfer completed callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
__weak void HAL_SAI_RxCpltCallback(SAI_HandleTypeDef *hsai)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsai);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SAI_RxCpltCallback could be implemented in the user file
   */
}

/**
  * @brief Rx Transfer half completed callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
__weak void HAL_SAI_RxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsai);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SAI_RxHalfCpltCallback could be implemented in the user file
   */
}

/**
  * @brief SAI error callback.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
__weak void HAL_SAI_ErrorCallback(SAI_HandleTypeDef *hsai)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hsai);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_SAI_ErrorCallback could be implemented in the user file
   */
}

/**
  * @}
  */

/** @defgroup SAI_Exported_Functions_Group3 Peripheral State functions
  * @brief    Peripheral State functions
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
  * @brief  Return the SAI handle state.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval HAL state
  */
HAL_SAI_StateTypeDef HAL_SAI_GetState(SAI_HandleTypeDef *hsai)
{
  return hsai->State;
}

/**
  * @brief  Return the SAI error code.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *              the configuration information for the specified SAI Block.
  * @retval SAI Error Code
  */
uint32_t HAL_SAI_GetError(SAI_HandleTypeDef *hsai)
{
  return hsai->ErrorCode;
}
/**
  * @}
  */

/**
  * @}
  */

/** @addtogroup SAI_Private_Functions
  * @brief      Private functions
  * @{
  */

/**
  * @brief  Initialize the SAI I2S protocol according to the specified parameters
  *         in the SAI_InitTypeDef and create the associated handle.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  protocol one of the supported protocol.
  * @param  datasize one of the supported datasize @ref SAI_Protocol_DataSize
  *                the configuration information for SAI module.
  * @param  nbslot number of slot minimum value is 2 and max is 16.
  *                    the value must be a multiple of 2.
  * @retval HAL status
  */
static HAL_StatusTypeDef SAI_InitI2S(SAI_HandleTypeDef *hsai, uint32_t protocol, uint32_t datasize, uint32_t nbslot)
{
  hsai->Init.Protocol            = SAI_FREE_PROTOCOL;
  hsai->Init.FirstBit            = SAI_FIRSTBIT_MSB;
  /* Compute ClockStrobing according AudioMode */
  if ((hsai->Init.AudioMode == SAI_MODEMASTER_TX) || (hsai->Init.AudioMode == SAI_MODESLAVE_TX))
  {
    /* Transmit */
    hsai->Init.ClockStrobing     = SAI_CLOCKSTROBING_FALLINGEDGE;
  }
  else
  {
    /* Receive */
    hsai->Init.ClockStrobing     = SAI_CLOCKSTROBING_RISINGEDGE;
  }
  hsai->FrameInit.FSDefinition   = SAI_FS_CHANNEL_IDENTIFICATION;
  hsai->SlotInit.SlotActive      = SAI_SLOTACTIVE_ALL;
  hsai->SlotInit.FirstBitOffset  = 0U;
  hsai->SlotInit.SlotNumber      = nbslot;

  /* in IS2 the number of slot must be even */
  if ((nbslot & 0x1U) != 0U)
  {
    return HAL_ERROR;
  }

  if (protocol == SAI_I2S_STANDARD)
  {
    hsai->FrameInit.FSPolarity = SAI_FS_ACTIVE_LOW;
    hsai->FrameInit.FSOffset   = SAI_FS_BEFOREFIRSTBIT;
  }
  else
  {
    /* SAI_I2S_MSBJUSTIFIED or SAI_I2S_LSBJUSTIFIED */
    hsai->FrameInit.FSPolarity = SAI_FS_ACTIVE_HIGH;
    hsai->FrameInit.FSOffset   = SAI_FS_FIRSTBIT;
  }

  /* Frame definition */
  switch (datasize)
  {
    case SAI_PROTOCOL_DATASIZE_16BIT:
      hsai->Init.DataSize = SAI_DATASIZE_16;
      hsai->FrameInit.FrameLength = 32U * (nbslot / 2U);
      hsai->FrameInit.ActiveFrameLength = 16U * (nbslot / 2U);
      hsai->SlotInit.SlotSize = SAI_SLOTSIZE_16B;
      break;
    case SAI_PROTOCOL_DATASIZE_16BITEXTENDED :
      hsai->Init.DataSize = SAI_DATASIZE_16;
      hsai->FrameInit.FrameLength = 64U * (nbslot / 2U);
      hsai->FrameInit.ActiveFrameLength = 32U * (nbslot / 2U);
      hsai->SlotInit.SlotSize = SAI_SLOTSIZE_32B;
      break;
    case SAI_PROTOCOL_DATASIZE_24BIT:
      hsai->Init.DataSize = SAI_DATASIZE_24;
      hsai->FrameInit.FrameLength = 64U * (nbslot / 2U);
      hsai->FrameInit.ActiveFrameLength = 32U * (nbslot / 2U);
      hsai->SlotInit.SlotSize = SAI_SLOTSIZE_32B;
      break;
    case SAI_PROTOCOL_DATASIZE_32BIT:
      hsai->Init.DataSize = SAI_DATASIZE_32;
      hsai->FrameInit.FrameLength = 64U * (nbslot / 2U);
      hsai->FrameInit.ActiveFrameLength = 32U * (nbslot / 2U);
      hsai->SlotInit.SlotSize = SAI_SLOTSIZE_32B;
      break;
    default :
      return HAL_ERROR;
  }
  if (protocol == SAI_I2S_LSBJUSTIFIED)
  {
    if (datasize == SAI_PROTOCOL_DATASIZE_16BITEXTENDED)
    {
      hsai->SlotInit.FirstBitOffset = 16U;
    }
    if (datasize == SAI_PROTOCOL_DATASIZE_24BIT)
    {
      hsai->SlotInit.FirstBitOffset = 8U;
    }
  }
  return HAL_OK;
}

/**
  * @brief  Initialize the SAI PCM protocol according to the specified parameters
  *         in the SAI_InitTypeDef and create the associated handle.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  protocol one of the supported protocol
  * @param  datasize one of the supported datasize @ref SAI_Protocol_DataSize
  * @param  nbslot number of slot minimum value is 1 and the max is 16.
  * @retval HAL status
  */
static HAL_StatusTypeDef SAI_InitPCM(SAI_HandleTypeDef *hsai, uint32_t protocol, uint32_t datasize, uint32_t nbslot)
{
  hsai->Init.Protocol            = SAI_FREE_PROTOCOL;
  hsai->Init.FirstBit            = SAI_FIRSTBIT_MSB;
  /* Compute ClockStrobing according AudioMode */
  if ((hsai->Init.AudioMode == SAI_MODEMASTER_TX) || (hsai->Init.AudioMode == SAI_MODESLAVE_TX))
  {
    /* Transmit */
    hsai->Init.ClockStrobing     = SAI_CLOCKSTROBING_RISINGEDGE;
  }
  else
  {
    /* Receive */
    hsai->Init.ClockStrobing     = SAI_CLOCKSTROBING_FALLINGEDGE;
  }
  hsai->FrameInit.FSDefinition   = SAI_FS_STARTFRAME;
  hsai->FrameInit.FSPolarity     = SAI_FS_ACTIVE_HIGH;
  hsai->FrameInit.FSOffset       = SAI_FS_BEFOREFIRSTBIT;
  hsai->SlotInit.FirstBitOffset  = 0U;
  hsai->SlotInit.SlotNumber      = nbslot;
  hsai->SlotInit.SlotActive      = SAI_SLOTACTIVE_ALL;

  if (protocol == SAI_PCM_SHORT)
  {
    hsai->FrameInit.ActiveFrameLength = 1;
  }
  else
  {
    /* SAI_PCM_LONG */
    hsai->FrameInit.ActiveFrameLength = 13;
  }

  switch (datasize)
  {
    case SAI_PROTOCOL_DATASIZE_16BIT:
      hsai->Init.DataSize = SAI_DATASIZE_16;
      hsai->FrameInit.FrameLength = 16U * nbslot;
      hsai->SlotInit.SlotSize = SAI_SLOTSIZE_16B;
      break;
    case SAI_PROTOCOL_DATASIZE_16BITEXTENDED :
      hsai->Init.DataSize = SAI_DATASIZE_16;
      hsai->FrameInit.FrameLength = 32U * nbslot;
      hsai->SlotInit.SlotSize = SAI_SLOTSIZE_32B;
      break;
    case SAI_PROTOCOL_DATASIZE_24BIT :
      hsai->Init.DataSize = SAI_DATASIZE_24;
      hsai->FrameInit.FrameLength = 32U * nbslot;
      hsai->SlotInit.SlotSize = SAI_SLOTSIZE_32B;
      break;
    case SAI_PROTOCOL_DATASIZE_32BIT:
      hsai->Init.DataSize = SAI_DATASIZE_32;
      hsai->FrameInit.FrameLength = 32U * nbslot;
      hsai->SlotInit.SlotSize = SAI_SLOTSIZE_32B;
      break;
    default :
      return HAL_ERROR;
  }

  return HAL_OK;
}

/**
  * @brief  Fill the fifo.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
static void SAI_FillFifo(SAI_HandleTypeDef *hsai)
{
  /* fill the fifo with data before to enabled the SAI */
  while (((hsai->Instance->SR & SAI_xSR_FLVL) != SAI_FIFOSTATUS_FULL) && (hsai->XferCount > 0U))
  {
    if ((hsai->Init.DataSize == SAI_DATASIZE_8) && (hsai->Init.CompandingMode == SAI_NOCOMPANDING))
    {
      hsai->Instance->DR = (*hsai->pBuffPtr++);
    }
    else if (hsai->Init.DataSize <= SAI_DATASIZE_16)
    {
      hsai->Instance->DR = *((uint32_t *)hsai->pBuffPtr);
      hsai->pBuffPtr += 2U;
    }
    else
    {
      hsai->Instance->DR = *((uint32_t *)hsai->pBuffPtr);
      hsai->pBuffPtr += 4U;
    }
    hsai->XferCount--;
  }
}

/**
  * @brief  Return the interrupt flag to set according the SAI setup.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @param  mode SAI_MODE_DMA or SAI_MODE_IT
  * @retval the list of the IT flag to enable
 */
static uint32_t SAI_InterruptFlag(SAI_HandleTypeDef *hsai, uint32_t mode)
{
  uint32_t tmpIT = SAI_IT_OVRUDR;

  if (mode == SAI_MODE_IT)
  {
    tmpIT |= SAI_IT_FREQ;
  }

  if ((hsai->Init.Protocol == SAI_AC97_PROTOCOL) &&
      ((hsai->Init.AudioMode == SAI_MODESLAVE_RX) || (hsai->Init.AudioMode == SAI_MODEMASTER_RX)))
  {
    tmpIT |= SAI_IT_CNRDY;
  }

  if ((hsai->Init.AudioMode == SAI_MODESLAVE_RX) || (hsai->Init.AudioMode == SAI_MODESLAVE_TX))
  {
    tmpIT |= SAI_IT_AFSDET | SAI_IT_LFSDET;
  }
  else
  {
    /* hsai has been configured in master mode */
    tmpIT |= SAI_IT_WCKCFG;
  }
  return tmpIT;
}

/**
  * @brief  Disable the SAI and wait for the disabling.
  * @param  hsai  pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
static HAL_StatusTypeDef SAI_Disable(SAI_HandleTypeDef *hsai)
{
  uint32_t count = SAI_DEFAULT_TIMEOUT * (SystemCoreClock / 7U / 1000U);
  HAL_StatusTypeDef status = HAL_OK;

  /* Disable the SAI instance */
  __HAL_SAI_DISABLE(hsai);

  do
  {
    /* Check for the Timeout */
    if (count-- == 0U)
    {
      /* Update error code */
      hsai->ErrorCode |= HAL_SAI_ERROR_TIMEOUT;
      status = HAL_TIMEOUT;
      break;
    }
  }
  while ((hsai->Instance->CR1 & SAI_xCR1_SAIEN) != RESET);

  return status;
}

/**
  * @brief  Tx Handler for Transmit in Interrupt mode 8-Bit transfer.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
static void SAI_Transmit_IT8Bit(SAI_HandleTypeDef *hsai)
{
  if (hsai->XferCount == 0U)
  {
    /* Handle the end of the transmission */
    /* Disable FREQ and OVRUDR interrupts */
    __HAL_SAI_DISABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_IT));
    hsai->State = HAL_SAI_STATE_READY;
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
    hsai->TxCpltCallback(hsai);
#else
    HAL_SAI_TxCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
  }
  else
  {
    /* Write data on DR register */
    hsai->Instance->DR = (*hsai->pBuffPtr++);
    hsai->XferCount--;
  }
}

/**
  * @brief  Tx Handler for Transmit in Interrupt mode for 16-Bit transfer.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
static void SAI_Transmit_IT16Bit(SAI_HandleTypeDef *hsai)
{
  if (hsai->XferCount == 0U)
  {
    /* Handle the end of the transmission */
    /* Disable FREQ and OVRUDR interrupts */
    __HAL_SAI_DISABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_IT));
    hsai->State = HAL_SAI_STATE_READY;
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
    hsai->TxCpltCallback(hsai);
#else
    HAL_SAI_TxCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
  }
  else
  {
    /* Write data on DR register */
    hsai->Instance->DR = *(uint16_t *)hsai->pBuffPtr;
    hsai->pBuffPtr += 2U;
    hsai->XferCount--;
  }
}

/**
  * @brief  Tx Handler for Transmit in Interrupt mode for 32-Bit transfer.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
static void SAI_Transmit_IT32Bit(SAI_HandleTypeDef *hsai)
{
  if (hsai->XferCount == 0U)
  {
    /* Handle the end of the transmission */
    /* Disable FREQ and OVRUDR interrupts */
    __HAL_SAI_DISABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_IT));
    hsai->State = HAL_SAI_STATE_READY;
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
    hsai->TxCpltCallback(hsai);
#else
    HAL_SAI_TxCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
  }
  else
  {
    /* Write data on DR register */
    hsai->Instance->DR = *(uint32_t *)hsai->pBuffPtr;
    hsai->pBuffPtr += 4U;
    hsai->XferCount--;
  }
}

/**
  * @brief  Rx Handler for Receive in Interrupt mode 8-Bit transfer.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
static void SAI_Receive_IT8Bit(SAI_HandleTypeDef *hsai)
{
  /* Receive data */
  (*hsai->pBuffPtr++) = hsai->Instance->DR;
  hsai->XferCount--;

  /* Check end of the transfer */
  if (hsai->XferCount == 0U)
  {
    /* Disable TXE and OVRUDR interrupts */
    __HAL_SAI_DISABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_IT));

    /* Clear the SAI Overrun flag */
    __HAL_SAI_CLEAR_FLAG(hsai, SAI_FLAG_OVRUDR);

    hsai->State = HAL_SAI_STATE_READY;
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
    hsai->RxCpltCallback(hsai);
#else
    HAL_SAI_RxCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
  }
}

/**
  * @brief  Rx Handler for Receive in Interrupt mode for 16-Bit transfer.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
static void SAI_Receive_IT16Bit(SAI_HandleTypeDef *hsai)
{
  /* Receive data */
  *(uint16_t *)hsai->pBuffPtr = hsai->Instance->DR;
  hsai->pBuffPtr += 2U;
  hsai->XferCount--;

  /* Check end of the transfer */
  if (hsai->XferCount == 0U)
  {
    /* Disable TXE and OVRUDR interrupts */
    __HAL_SAI_DISABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_IT));

    /* Clear the SAI Overrun flag */
    __HAL_SAI_CLEAR_FLAG(hsai, SAI_FLAG_OVRUDR);

    hsai->State = HAL_SAI_STATE_READY;
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
    hsai->RxCpltCallback(hsai);
#else
    HAL_SAI_RxCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
  }
}

/**
  * @brief  Rx Handler for Receive in Interrupt mode for 32-Bit transfer.
  * @param  hsai pointer to a SAI_HandleTypeDef structure that contains
  *               the configuration information for SAI module.
  * @retval None
  */
static void SAI_Receive_IT32Bit(SAI_HandleTypeDef *hsai)
{
  /* Receive data */
  *(uint32_t *)hsai->pBuffPtr = hsai->Instance->DR;
  hsai->pBuffPtr += 4U;
  hsai->XferCount--;

  /* Check end of the transfer */
  if (hsai->XferCount == 0U)
  {
    /* Disable TXE and OVRUDR interrupts */
    __HAL_SAI_DISABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_IT));

    /* Clear the SAI Overrun flag */
    __HAL_SAI_CLEAR_FLAG(hsai, SAI_FLAG_OVRUDR);

    hsai->State = HAL_SAI_STATE_READY;
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
    hsai->RxCpltCallback(hsai);
#else
    HAL_SAI_RxCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
  }
}

/**
  * @brief DMA SAI transmit process complete callback.
  * @param  hdma pointer to a DMA_HandleTypeDef structure that contains
  *               the configuration information for the specified DMA module.
  * @retval None
  */
static void SAI_DMATxCplt(DMA_HandleTypeDef *hdma)
{
  SAI_HandleTypeDef *hsai = (SAI_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (hdma->Init.Mode != DMA_CIRCULAR)
  {
    hsai->XferCount = 0U;

    /* Disable SAI Tx DMA Request */
    hsai->Instance->CR1 &= (uint32_t)(~SAI_xCR1_DMAEN);

    /* Stop the interrupts error handling */
    __HAL_SAI_DISABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_DMA));

    hsai->State = HAL_SAI_STATE_READY;
  }
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
  hsai->TxCpltCallback(hsai);
#else
  HAL_SAI_TxCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
}

/**
  * @brief DMA SAI transmit process half complete callback.
  * @param  hdma pointer to a DMA_HandleTypeDef structure that contains
  *               the configuration information for the specified DMA module.
  * @retval None
  */
static void SAI_DMATxHalfCplt(DMA_HandleTypeDef *hdma)
{
  SAI_HandleTypeDef *hsai = (SAI_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
  hsai->TxHalfCpltCallback(hsai);
#else
  HAL_SAI_TxHalfCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
}

/**
  * @brief DMA SAI receive process complete callback.
  * @param  hdma pointer to a DMA_HandleTypeDef structure that contains
  *               the configuration information for the specified DMA module.
  * @retval None
  */
static void SAI_DMARxCplt(DMA_HandleTypeDef *hdma)
{
  SAI_HandleTypeDef *hsai = (SAI_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  if (hdma->Init.Mode != DMA_CIRCULAR)
  {
    /* Disable Rx DMA Request */
    hsai->Instance->CR1 &= (uint32_t)(~SAI_xCR1_DMAEN);
    hsai->XferCount = 0U;

    /* Stop the interrupts error handling */
    __HAL_SAI_DISABLE_IT(hsai, SAI_InterruptFlag(hsai, SAI_MODE_DMA));

    hsai->State = HAL_SAI_STATE_READY;
  }
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
  hsai->RxCpltCallback(hsai);
#else
  HAL_SAI_RxCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
}

/**
  * @brief DMA SAI receive process half complete callback
  * @param  hdma pointer to a DMA_HandleTypeDef structure that contains
  *               the configuration information for the specified DMA module.
  * @retval None
  */
static void SAI_DMARxHalfCplt(DMA_HandleTypeDef *hdma)
{
  SAI_HandleTypeDef *hsai = (SAI_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
  hsai->RxHalfCpltCallback(hsai);
#else
  HAL_SAI_RxHalfCpltCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
}

/**
  * @brief DMA SAI communication error callback.
  * @param  hdma pointer to a DMA_HandleTypeDef structure that contains
  *               the configuration information for the specified DMA module.
  * @retval None
  */
static void SAI_DMAError(DMA_HandleTypeDef *hdma)
{
  SAI_HandleTypeDef *hsai = (SAI_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  /* Set SAI error code */
  hsai->ErrorCode |= HAL_SAI_ERROR_DMA;

  if ((hsai->hdmatx->ErrorCode == HAL_DMA_ERROR_TE) || (hsai->hdmarx->ErrorCode == HAL_DMA_ERROR_TE))
  {
    /* Disable the SAI DMA request */
    hsai->Instance->CR1 &= ~SAI_xCR1_DMAEN;

    /* Disable SAI peripheral */
    SAI_Disable(hsai);

    /* Set the SAI state ready to be able to start again the process */
    hsai->State = HAL_SAI_STATE_READY;

    /* Initialize XferCount */
    hsai->XferCount = 0U;
  }
  /* SAI error Callback */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
  hsai->ErrorCallback(hsai);
#else
  HAL_SAI_ErrorCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
}

/**
  * @brief  DMA SAI Abort callback.
  * @param  hdma pointer to a DMA_HandleTypeDef structure that contains
  *               the configuration information for the specified DMA module.
  * @retval None
  */
static void SAI_DMAAbort(DMA_HandleTypeDef *hdma)
{
  SAI_HandleTypeDef *hsai = (SAI_HandleTypeDef *)((DMA_HandleTypeDef *)hdma)->Parent;

  /* Disable DMA request */
  hsai->Instance->CR1 &= ~SAI_xCR1_DMAEN;

  /* Disable all interrupts and clear all flags */
  hsai->Instance->IMR = 0U;
  hsai->Instance->CLRFR = 0xFFFFFFFFU;

  if (hsai->ErrorCode != HAL_SAI_ERROR_WCKCFG)
  {
    /* Disable SAI peripheral */
    SAI_Disable(hsai);

    /* Flush the fifo */
    SET_BIT(hsai->Instance->CR2, SAI_xCR2_FFLUSH);
  }
  /* Set the SAI state to ready to be able to start again the process */
  hsai->State = HAL_SAI_STATE_READY;

  /* Initialize XferCount */
  hsai->XferCount = 0U;

  /* SAI error Callback */
#if (USE_HAL_SAI_REGISTER_CALLBACKS == 1)
  hsai->ErrorCallback(hsai);
#else
  HAL_SAI_ErrorCallback(hsai);
#endif /* USE_HAL_SAI_REGISTER_CALLBACKS */
}

/**
  * @}
  */

#endif /* STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx || STM32F413xx || STM32F423xx */
#endif /* HAL_SAI_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
