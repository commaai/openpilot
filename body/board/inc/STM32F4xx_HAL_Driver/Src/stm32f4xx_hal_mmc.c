/**
  ******************************************************************************
  * @file    stm32f4xx_hal_mmc.c
  * @author  MCD Application Team
  * @brief   MMC card HAL module driver.
  *          This file provides firmware functions to manage the following
  *          functionalities of the Secure Digital (MMC) peripheral:
  *           + Initialization and de-initialization functions
  *           + IO operation functions
  *           + Peripheral Control functions
  *           + MMC card Control functions
  *
  @verbatim
  ==============================================================================
                        ##### How to use this driver #####
  ==============================================================================
  [..]
    This driver implements a high level communication layer for read and write from/to
    this memory. The needed STM32 hardware resources (SDMMC and GPIO) are performed by
    the user in HAL_MMC_MspInit() function (MSP layer).
    Basically, the MSP layer configuration should be the same as we provide in the
    examples.
    You can easily tailor this configuration according to hardware resources.

  [..]
    This driver is a generic layered driver for SDMMC memories which uses the HAL
    SDMMC driver functions to interface with MMC and eMMC cards devices.
    It is used as follows:

    (#)Initialize the SDMMC low level resources by implement the HAL_MMC_MspInit() API:
        (##) Enable the SDMMC interface clock using __HAL_RCC_SDMMC_CLK_ENABLE();
        (##) SDMMC pins configuration for MMC card
            (+++) Enable the clock for the SDMMC GPIOs using the functions __HAL_RCC_GPIOx_CLK_ENABLE();
            (+++) Configure these SDMMC pins as alternate function pull-up using HAL_GPIO_Init()
                  and according to your pin assignment;
        (##) DMA Configuration if you need to use DMA process (HAL_MMC_ReadBlocks_DMA()
             and HAL_MMC_WriteBlocks_DMA() APIs).
            (+++) Enable the DMAx interface clock using __HAL_RCC_DMAx_CLK_ENABLE(); 
            (+++) Configure the DMA using the function HAL_DMA_Init() with predeclared and filled. 
        (##) NVIC configuration if you need to use interrupt process when using DMA transfer.
            (+++) Configure the SDMMC and DMA interrupt priorities using function HAL_NVIC_SetPriority();
                  DMA priority is superior to SDMMC's priority
            (+++) Enable the NVIC DMA and SDMMC IRQs using function HAL_NVIC_EnableIRQ()
            (+++) SDMMC interrupts are managed using the macros __HAL_MMC_ENABLE_IT() 
                  and __HAL_MMC_DISABLE_IT() inside the communication process.
            (+++) SDMMC interrupts pending bits are managed using the macros __HAL_MMC_GET_IT()
                  and __HAL_MMC_CLEAR_IT()
        (##) NVIC configuration if you need to use interrupt process (HAL_MMC_ReadBlocks_IT()
             and HAL_MMC_WriteBlocks_IT() APIs).
            (+++) Configure the SDMMC interrupt priorities using function HAL_NVIC_SetPriority();
            (+++) Enable the NVIC SDMMC IRQs using function HAL_NVIC_EnableIRQ()
            (+++) SDMMC interrupts are managed using the macros __HAL_MMC_ENABLE_IT()
                  and __HAL_MMC_DISABLE_IT() inside the communication process.
            (+++) SDMMC interrupts pending bits are managed using the macros __HAL_MMC_GET_IT()
                  and __HAL_MMC_CLEAR_IT()
    (#) At this stage, you can perform MMC read/write/erase operations after MMC card initialization


  *** MMC Card Initialization and configuration ***
  ================================================
  [..]
    To initialize the MMC Card, use the HAL_MMC_Init() function. It Initializes
    SDMMC Peripheral (STM32 side) and the MMC Card, and put it into StandBy State (Ready for data transfer).
    This function provide the following operations:

    (#) Initialize the SDMMC peripheral interface with defaullt configuration.
        The initialization process is done at 400KHz. You can change or adapt
        this frequency by adjusting the "ClockDiv" field.
        The MMC Card frequency (SDMMC_CK) is computed as follows:

           SDMMC_CK = SDMMCCLK / (ClockDiv + 2)

        In initialization mode and according to the MMC Card standard,
        make sure that the SDMMC_CK frequency doesn't exceed 400KHz.

        This phase of initialization is done through SDMMC_Init() and
        SDMMC_PowerState_ON() SDMMC low level APIs.

    (#) Initialize the MMC card. The API used is HAL_MMC_InitCard().
        This phase allows the card initialization and identification
        and check the MMC Card type (Standard Capacity or High Capacity)
        The initialization flow is compatible with MMC standard.

        This API (HAL_MMC_InitCard()) could be used also to reinitialize the card in case
        of plug-off plug-in.
  
    (#) Configure the MMC Card Data transfer frequency. By Default, the card transfer
        frequency is set to 24MHz. You can change or adapt this frequency by adjusting 
        the "ClockDiv" field.
        In transfer mode and according to the MMC Card standard, make sure that the
        SDMMC_CK frequency doesn't exceed 25MHz and 50MHz in High-speed mode switch.
        To be able to use a frequency higher than 24MHz, you should use the SDMMC
        peripheral in bypass mode. Refer to the corresponding reference manual
        for more details.

    (#) Select the corresponding MMC Card according to the address read with the step 2.

    (#) Configure the MMC Card in wide bus mode: 4-bits data.

  *** MMC Card Read operation ***
  ==============================
  [..]
    (+) You can read from MMC card in polling mode by using function HAL_MMC_ReadBlocks().
        This function support only 512-bytes block length (the block size should be
        chosen as 512 bytes).
        You can choose either one block read operation or multiple block read operation
        by adjusting the "NumberOfBlocks" parameter.
        After this, you have to ensure that the transfer is done correctly. The check is done
        through HAL_MMC_GetCardState() function for MMC card state.

    (+) You can read from MMC card in DMA mode by using function HAL_MMC_ReadBlocks_DMA().
        This function support only 512-bytes block length (the block size should be
        chosen as 512 bytes).
        You can choose either one block read operation or multiple block read operation
        by adjusting the "NumberOfBlocks" parameter.
        After this, you have to ensure that the transfer is done correctly. The check is done
        through HAL_MMC_GetCardState() function for MMC card state.
        You could also check the DMA transfer process through the MMC Rx interrupt event.

    (+) You can read from MMC card in Interrupt mode by using function HAL_MMC_ReadBlocks_IT().
        This function allows the read of 512 bytes blocks.
        You can choose either one block read operation or multiple block read operation 
        by adjusting the "NumberOfBlocks" parameter.
        After this, you have to ensure that the transfer is done correctly. The check is done
        through HAL_MMC_GetCardState() function for MMC card state.
        You could also check the IT transfer process through the MMC Rx interrupt event.

  *** MMC Card Write operation ***
  ===============================
  [..]
    (+) You can write to MMC card in polling mode by using function HAL_MMC_WriteBlocks().
        This function support only 512-bytes block length (the block size should be
        chosen as 512 bytes).
        You can choose either one block read operation or multiple block read operation
        by adjusting the "NumberOfBlocks" parameter.
        After this, you have to ensure that the transfer is done correctly. The check is done
        through HAL_MMC_GetCardState() function for MMC card state.

    (+) You can write to MMC card in DMA mode by using function HAL_MMC_WriteBlocks_DMA().
        This function support only 512-bytes block length (the block size should be
        chosen as 512 byte).
        You can choose either one block read operation or multiple block read operation
        by adjusting the "NumberOfBlocks" parameter.
        After this, you have to ensure that the transfer is done correctly. The check is done
        through HAL_MMC_GetCardState() function for MMC card state.
        You could also check the DMA transfer process through the MMC Tx interrupt event.  

    (+) You can write to MMC card in Interrupt mode by using function HAL_MMC_WriteBlocks_IT().
        This function allows the read of 512 bytes blocks.
        You can choose either one block read operation or multiple block read operation 
        by adjusting the "NumberOfBlocks" parameter.
        After this, you have to ensure that the transfer is done correctly. The check is done
        through HAL_MMC_GetCardState() function for MMC card state.
        You could also check the IT transfer process through the MMC Tx interrupt event.

  *** MMC card information ***
  =========================== 
  [..]
    (+) To get MMC card information, you can use the function HAL_MMC_GetCardInfo().
        It returns useful information about the MMC card such as block size, card type,
        block number ...

  *** MMC card CSD register ***
  ============================
  [..]
    (+) The HAL_MMC_GetCardCSD() API allows to get the parameters of the CSD register.
        Some of the CSD parameters are useful for card initialization and identification.

  *** MMC card CID register ***
  ============================
  [..]
    (+) The HAL_MMC_GetCardCID() API allows to get the parameters of the CID register.
        Some of the CID parameters are useful for card initialization and identification.

  *** MMC HAL driver macros list ***
  ==================================
  [..]
    Below the list of most used macros in MMC HAL driver.

    (+) __HAL_MMC_ENABLE : Enable the MMC device
    (+) __HAL_MMC_DISABLE : Disable the MMC device
    (+) __HAL_MMC_DMA_ENABLE: Enable the SDMMC DMA transfer
    (+) __HAL_MMC_DMA_DISABLE: Disable the SDMMC DMA transfer
    (+) __HAL_MMC_ENABLE_IT: Enable the MMC device interrupt
    (+) __HAL_MMC_DISABLE_IT: Disable the MMC device interrupt
    (+) __HAL_MMC_GET_FLAG:Check whether the specified MMC flag is set or not
    (+) __HAL_MMC_CLEAR_FLAG: Clear the MMC's pending flags

  [..]
    (@) You can refer to the MMC HAL driver header file for more useful macros

  *** Callback registration ***
  =============================================
  [..]
    The compilation define USE_HAL_MMC_REGISTER_CALLBACKS when set to 1
    allows the user to configure dynamically the driver callbacks.

    Use Functions HAL_MMC_RegisterCallback() to register a user callback,
    it allows to register following callbacks:
      (+) TxCpltCallback : callback when a transmission transfer is completed.
      (+) RxCpltCallback : callback when a reception transfer is completed.
      (+) ErrorCallback : callback when error occurs.
      (+) AbortCpltCallback : callback when abort is completed.
      (+) MspInitCallback    : MMC MspInit.
      (+) MspDeInitCallback  : MMC MspDeInit.
    This function takes as parameters the HAL peripheral handle, the Callback ID
    and a pointer to the user callback function.

    Use function HAL_MMC_UnRegisterCallback() to reset a callback to the default
    weak (surcharged) function. It allows to reset following callbacks:
      (+) TxCpltCallback : callback when a transmission transfer is completed.
      (+) RxCpltCallback : callback when a reception transfer is completed.
      (+) ErrorCallback : callback when error occurs.
      (+) AbortCpltCallback : callback when abort is completed.
      (+) MspInitCallback    : MMC MspInit.
      (+) MspDeInitCallback  : MMC MspDeInit.
    This function) takes as parameters the HAL peripheral handle and the Callback ID.

    By default, after the HAL_MMC_Init and if the state is HAL_MMC_STATE_RESET
    all callbacks are reset to the corresponding legacy weak (surcharged) functions.
    Exception done for MspInit and MspDeInit callbacks that are respectively
    reset to the legacy weak (surcharged) functions in the HAL_MMC_Init
    and  HAL_MMC_DeInit only when these callbacks are null (not registered beforehand).
    If not, MspInit or MspDeInit are not null, the HAL_MMC_Init and HAL_MMC_DeInit
    keep and use the user MspInit/MspDeInit callbacks (registered beforehand)

    Callbacks can be registered/unregistered in READY state only.
    Exception done for MspInit/MspDeInit callbacks that can be registered/unregistered
    in READY or RESET state, thus registered (user) MspInit/DeInit callbacks can be used
    during the Init/DeInit.
    In that case first register the MspInit/MspDeInit user callbacks
    using HAL_MMC_RegisterCallback before calling HAL_MMC_DeInit
    or HAL_MMC_Init function.

    When The compilation define USE_HAL_MMC_REGISTER_CALLBACKS is set to 0 or
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
  *                       opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup MMC MMC
  * @brief MMC HAL module driver
  * @{
  */

#ifdef HAL_MMC_MODULE_ENABLED

#if defined(SDIO)

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @addtogroup MMC_Private_Defines
  * @{
  */

/**
  * @}
  */

/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/
/** @defgroup MMC_Private_Functions MMC Private Functions
  * @{
  */
static uint32_t MMC_InitCard(MMC_HandleTypeDef *hmmc);
static uint32_t MMC_PowerON(MMC_HandleTypeDef *hmmc);
static uint32_t MMC_SendStatus(MMC_HandleTypeDef *hmmc, uint32_t *pCardStatus);
static uint32_t MMC_ReadExtCSD(MMC_HandleTypeDef *hmmc, uint32_t *pFieldData, uint16_t FieldIndex, uint32_t Timeout);
static void     MMC_PowerOFF(MMC_HandleTypeDef *hmmc);
static void     MMC_Write_IT(MMC_HandleTypeDef *hmmc);
static void     MMC_Read_IT(MMC_HandleTypeDef *hmmc);
static void     MMC_DMATransmitCplt(DMA_HandleTypeDef *hdma);
static void     MMC_DMAReceiveCplt(DMA_HandleTypeDef *hdma);
static void     MMC_DMAError(DMA_HandleTypeDef *hdma);
static void     MMC_DMATxAbort(DMA_HandleTypeDef *hdma);
static void     MMC_DMARxAbort(DMA_HandleTypeDef *hdma);
/**
  * @}
  */
/* Exported functions --------------------------------------------------------*/
/** @addtogroup MMC_Exported_Functions
  * @{
  */

/** @addtogroup MMC_Exported_Functions_Group1
 *  @brief   Initialization and de-initialization functions
 *
@verbatim
  ==============================================================================
          ##### Initialization and de-initialization functions #####
  ==============================================================================
  [..]
    This section provides functions allowing to initialize/de-initialize the MMC
    card device to be ready for use.

@endverbatim
  * @{
  */

/**
  * @brief  Initializes the MMC according to the specified parameters in the
            MMC_HandleTypeDef and create the associated handle.
  * @param  hmmc: Pointer to the MMC handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_Init(MMC_HandleTypeDef *hmmc)
{
  /* Check the MMC handle allocation */
  if(hmmc == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_SDIO_ALL_INSTANCE(hmmc->Instance));
  assert_param(IS_SDIO_CLOCK_EDGE(hmmc->Init.ClockEdge));
  assert_param(IS_SDIO_CLOCK_BYPASS(hmmc->Init.ClockBypass));
  assert_param(IS_SDIO_CLOCK_POWER_SAVE(hmmc->Init.ClockPowerSave));
  assert_param(IS_SDIO_BUS_WIDE(hmmc->Init.BusWide));
  assert_param(IS_SDIO_HARDWARE_FLOW_CONTROL(hmmc->Init.HardwareFlowControl));
  assert_param(IS_SDIO_CLKDIV(hmmc->Init.ClockDiv));

  if(hmmc->State == HAL_MMC_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hmmc->Lock = HAL_UNLOCKED;
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
    /* Reset Callback pointers in HAL_MMC_STATE_RESET only */
    hmmc->TxCpltCallback    = HAL_MMC_TxCpltCallback;
    hmmc->RxCpltCallback    = HAL_MMC_RxCpltCallback;
    hmmc->ErrorCallback     = HAL_MMC_ErrorCallback;
    hmmc->AbortCpltCallback = HAL_MMC_AbortCallback;

    if(hmmc->MspInitCallback == NULL)
    {
      hmmc->MspInitCallback = HAL_MMC_MspInit;
    }

    /* Init the low level hardware */
    hmmc->MspInitCallback(hmmc);
#else
    /* Init the low level hardware : GPIO, CLOCK, CORTEX...etc */
    HAL_MMC_MspInit(hmmc);
#endif
  }

  hmmc->State = HAL_MMC_STATE_BUSY;

  /* Initialize the Card parameters */
  if(HAL_MMC_InitCard(hmmc) == HAL_ERROR)
  {
    return HAL_ERROR;
  }

  /* Initialize the error code */
  hmmc->ErrorCode = HAL_DMA_ERROR_NONE;

  /* Initialize the MMC operation */
  hmmc->Context = MMC_CONTEXT_NONE;

  /* Initialize the MMC state */
  hmmc->State = HAL_MMC_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  Initializes the MMC Card.
  * @param  hmmc: Pointer to MMC handle
  * @note   This function initializes the MMC card. It could be used when a card
            re-initialization is needed.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_InitCard(MMC_HandleTypeDef *hmmc)
{
  uint32_t errorstate;
  MMC_InitTypeDef Init;
  HAL_StatusTypeDef status;
  
  /* Default SDIO peripheral configuration for MMC card initialization */
  Init.ClockEdge           = SDIO_CLOCK_EDGE_RISING;
  Init.ClockBypass         = SDIO_CLOCK_BYPASS_DISABLE;
  Init.ClockPowerSave      = SDIO_CLOCK_POWER_SAVE_DISABLE;
  Init.BusWide             = SDIO_BUS_WIDE_1B;
  Init.HardwareFlowControl = SDIO_HARDWARE_FLOW_CONTROL_DISABLE;
  Init.ClockDiv            = SDIO_INIT_CLK_DIV;

  /* Initialize SDIO peripheral interface with default configuration */
  status = SDIO_Init(hmmc->Instance, Init);
  if(status == HAL_ERROR)
  {
    return HAL_ERROR;
  }

  /* Disable SDIO Clock */
  __HAL_MMC_DISABLE(hmmc); 
  
  /* Set Power State to ON */
  status = SDIO_PowerState_ON(hmmc->Instance);
  if(status == HAL_ERROR)
  {
    return HAL_ERROR;
  }

  /* Enable MMC Clock */
  __HAL_MMC_ENABLE(hmmc);

  /* Identify card operating voltage */
  errorstate = MMC_PowerON(hmmc);
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    hmmc->State = HAL_MMC_STATE_READY;
    hmmc->ErrorCode |= errorstate;
    return HAL_ERROR;
  }

  /* Card initialization */
  errorstate = MMC_InitCard(hmmc);
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    hmmc->State = HAL_MMC_STATE_READY;
    hmmc->ErrorCode |= errorstate;
    return HAL_ERROR;
  }

  /* Set Block Size for Card */
  errorstate = SDMMC_CmdBlockLength(hmmc->Instance, MMC_BLOCKSIZE);
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    /* Clear all the static flags */
    __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
    hmmc->ErrorCode |= errorstate;
    hmmc->State = HAL_MMC_STATE_READY;
    return HAL_ERROR;
  }

  return HAL_OK;
}

/**
  * @brief  De-Initializes the MMC card.
  * @param  hmmc: Pointer to MMC handle
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_DeInit(MMC_HandleTypeDef *hmmc)
{
  /* Check the MMC handle allocation */
  if(hmmc == NULL)
  {
    return HAL_ERROR;
  }

  /* Check the parameters */
  assert_param(IS_SDIO_ALL_INSTANCE(hmmc->Instance));

  hmmc->State = HAL_MMC_STATE_BUSY;

  /* Set MMC power state to off */
  MMC_PowerOFF(hmmc);

#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
  if(hmmc->MspDeInitCallback == NULL)
  {
    hmmc->MspDeInitCallback = HAL_MMC_MspDeInit;
  }

  /* DeInit the low level hardware */
  hmmc->MspDeInitCallback(hmmc);
#else
  /* De-Initialize the MSP layer */
  HAL_MMC_MspDeInit(hmmc);
#endif

  hmmc->ErrorCode = HAL_MMC_ERROR_NONE;
  hmmc->State = HAL_MMC_STATE_RESET;

  return HAL_OK;
}


/**
  * @brief  Initializes the MMC MSP.
  * @param  hmmc: Pointer to MMC handle
  * @retval None
  */
__weak void HAL_MMC_MspInit(MMC_HandleTypeDef *hmmc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hmmc);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_MMC_MspInit could be implemented in the user file
   */
}

/**
  * @brief  De-Initialize MMC MSP.
  * @param  hmmc: Pointer to MMC handle
  * @retval None
  */
__weak void HAL_MMC_MspDeInit(MMC_HandleTypeDef *hmmc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hmmc);

  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_MMC_MspDeInit could be implemented in the user file
   */
}

/**
  * @}
  */

/** @addtogroup MMC_Exported_Functions_Group2
 *  @brief   Data transfer functions
 *
@verbatim
  ==============================================================================
                        ##### IO operation functions #####
  ==============================================================================
  [..]
    This subsection provides a set of functions allowing to manage the data
    transfer from/to MMC card.

@endverbatim
  * @{
  */

/**
  * @brief  Reads block(s) from a specified address in a card. The Data transfer
  *         is managed by polling mode.
  * @note   This API should be followed by a check on the card state through
  *         HAL_MMC_GetCardState().
  * @param  hmmc: Pointer to MMC handle
  * @param  pData: pointer to the buffer that will contain the received data
  * @param  BlockAdd: Block Address from where data is to be read
  * @param  NumberOfBlocks: Number of MMC blocks to read
  * @param  Timeout: Specify timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_ReadBlocks(MMC_HandleTypeDef *hmmc, uint8_t *pData, uint32_t BlockAdd, uint32_t NumberOfBlocks, uint32_t Timeout)
{
  SDIO_DataInitTypeDef config;
  uint32_t errorstate;
  uint32_t tickstart = HAL_GetTick();
  uint32_t count, data, dataremaining;
  uint32_t add = BlockAdd;
  uint8_t *tempbuff = pData;

  if(NULL == pData)
  {
    hmmc->ErrorCode |= HAL_MMC_ERROR_PARAM;
    return HAL_ERROR;
  }

  if(hmmc->State == HAL_MMC_STATE_READY)
  {
    hmmc->ErrorCode = HAL_MMC_ERROR_NONE;

    if((BlockAdd + NumberOfBlocks) > (hmmc->MmcCard.LogBlockNbr))
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_ADDR_OUT_OF_RANGE;
      return HAL_ERROR;
    }

    hmmc->State = HAL_MMC_STATE_BUSY;

    /* Initialize data control register */
    hmmc->Instance->DCTRL = 0U;

    if ((hmmc->MmcCard.CardType) != MMC_HIGH_CAPACITY_CARD)
    {
      add *= 512U;
    }

    /* Configure the MMC DPSM (Data Path State Machine) */
    config.DataTimeOut   = SDMMC_DATATIMEOUT;
    config.DataLength    = NumberOfBlocks * MMC_BLOCKSIZE;
    config.DataBlockSize = SDIO_DATABLOCK_SIZE_512B;
    config.TransferDir   = SDIO_TRANSFER_DIR_TO_SDIO;
    config.TransferMode  = SDIO_TRANSFER_MODE_BLOCK;
    config.DPSM          = SDIO_DPSM_ENABLE;
    (void)SDIO_ConfigData(hmmc->Instance, &config);

    /* Read block(s) in polling mode */
    if(NumberOfBlocks > 1U)
    {
      hmmc->Context = MMC_CONTEXT_READ_MULTIPLE_BLOCK;

      /* Read Multi Block command */
      errorstate = SDMMC_CmdReadMultiBlock(hmmc->Instance, add);
    }
    else
    {
      hmmc->Context = MMC_CONTEXT_READ_SINGLE_BLOCK;

      /* Read Single Block command */
      errorstate = SDMMC_CmdReadSingleBlock(hmmc->Instance, add);
    }
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= errorstate;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    /* Poll on SDIO flags */
    dataremaining = config.DataLength;
#if defined(SDIO_STA_STBITERR)
    while(!__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_RXOVERR | SDIO_FLAG_DCRCFAIL | SDIO_FLAG_DTIMEOUT | SDIO_FLAG_DATAEND | SDIO_FLAG_STBITERR))
#else /* SDIO_STA_STBITERR not defined */
    while(!__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_RXOVERR | SDIO_FLAG_DCRCFAIL | SDIO_FLAG_DTIMEOUT | SDIO_FLAG_DATAEND))
#endif /* SDIO_STA_STBITERR */
    {
      if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_RXFIFOHF) && (dataremaining > 0U))
      {
        /* Read data from SDIO Rx FIFO */
        for(count = 0U; count < 8U; count++)
        {
          data = SDIO_ReadFIFO(hmmc->Instance);
          *tempbuff = (uint8_t)(data & 0xFFU);
          tempbuff++;
          dataremaining--;
          *tempbuff = (uint8_t)((data >> 8U) & 0xFFU);
          tempbuff++;
          dataremaining--;
          *tempbuff = (uint8_t)((data >> 16U) & 0xFFU);
          tempbuff++;
          dataremaining--;
          *tempbuff = (uint8_t)((data >> 24U) & 0xFFU);
          tempbuff++;
          dataremaining--;
        }
      }

      if(((HAL_GetTick()-tickstart) >=  Timeout) || (Timeout == 0U))
      {
        /* Clear all the static flags */
        __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
        hmmc->ErrorCode |= HAL_MMC_ERROR_TIMEOUT;
        hmmc->State= HAL_MMC_STATE_READY;
        return HAL_TIMEOUT;
      }
    }

    /* Send stop transmission command in case of multiblock read */
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DATAEND) && (NumberOfBlocks > 1U))
    {
      /* Send stop transmission command */
      errorstate = SDMMC_CmdStopTransfer(hmmc->Instance);
      if(errorstate != HAL_MMC_ERROR_NONE)
      {
        /* Clear all the static flags */
        __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
        hmmc->ErrorCode |= errorstate;
        hmmc->State = HAL_MMC_STATE_READY;
        return HAL_ERROR;
      }
    }

    /* Get error state */
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DTIMEOUT))
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_DATA_TIMEOUT;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }
    else if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DCRCFAIL))
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_DATA_CRC_FAIL;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }
    else if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_RXOVERR))
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_RX_OVERRUN;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }
    else
    {
      /* Nothing to do */
    }

    /* Empty FIFO if there is still any data */
    while ((__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_RXDAVL)) && (dataremaining > 0U))
    {
      data = SDIO_ReadFIFO(hmmc->Instance);
      *tempbuff = (uint8_t)(data & 0xFFU);
      tempbuff++;
      dataremaining--;
      *tempbuff = (uint8_t)((data >> 8U) & 0xFFU);
      tempbuff++;
      dataremaining--;
      *tempbuff = (uint8_t)((data >> 16U) & 0xFFU);
      tempbuff++;
      dataremaining--;
      *tempbuff = (uint8_t)((data >> 24U) & 0xFFU);
      tempbuff++;
      dataremaining--;

      if(((HAL_GetTick()-tickstart) >=  Timeout) || (Timeout == 0U))
      {
        /* Clear all the static flags */
        __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);        
        hmmc->ErrorCode |= HAL_MMC_ERROR_TIMEOUT;
        hmmc->State= HAL_MMC_STATE_READY;
        return HAL_ERROR;
      }
    }

    /* Clear all the static flags */
    __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_DATA_FLAGS);

    hmmc->State = HAL_MMC_STATE_READY;

    return HAL_OK;
  }
  else
  {
    hmmc->ErrorCode |= HAL_MMC_ERROR_BUSY;
    return HAL_ERROR;
  }
}

/**
  * @brief  Allows to write block(s) to a specified address in a card. The Data
  *         transfer is managed by polling mode.
  * @note   This API should be followed by a check on the card state through
  *         HAL_MMC_GetCardState().
  * @param  hmmc: Pointer to MMC handle
  * @param  pData: pointer to the buffer that will contain the data to transmit
  * @param  BlockAdd: Block Address where data will be written
  * @param  NumberOfBlocks: Number of MMC blocks to write
  * @param  Timeout: Specify timeout value
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_WriteBlocks(MMC_HandleTypeDef *hmmc, uint8_t *pData, uint32_t BlockAdd, uint32_t NumberOfBlocks, uint32_t Timeout)
{
  SDIO_DataInitTypeDef config;
  uint32_t errorstate;
  uint32_t tickstart = HAL_GetTick();
  uint32_t count, data, dataremaining;
  uint32_t add = BlockAdd;
  uint8_t *tempbuff = pData;

  if(NULL == pData)
  {
    hmmc->ErrorCode |= HAL_MMC_ERROR_PARAM;
    return HAL_ERROR;
  }

  if(hmmc->State == HAL_MMC_STATE_READY)
  {
    hmmc->ErrorCode = HAL_MMC_ERROR_NONE;

    if((BlockAdd + NumberOfBlocks) > (hmmc->MmcCard.LogBlockNbr))
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_ADDR_OUT_OF_RANGE;
      return HAL_ERROR;
    }

    hmmc->State = HAL_MMC_STATE_BUSY;

    /* Initialize data control register */
    hmmc->Instance->DCTRL = 0U;

    if ((hmmc->MmcCard.CardType) != MMC_HIGH_CAPACITY_CARD)
    {
      add *= 512U;
    }

    /* Write Blocks in Polling mode */
    if(NumberOfBlocks > 1U)
    {
      hmmc->Context = MMC_CONTEXT_WRITE_MULTIPLE_BLOCK;

      /* Write Multi Block command */
      errorstate = SDMMC_CmdWriteMultiBlock(hmmc->Instance, add);
    }
    else
    {
      hmmc->Context = MMC_CONTEXT_WRITE_SINGLE_BLOCK;

      /* Write Single Block command */
      errorstate = SDMMC_CmdWriteSingleBlock(hmmc->Instance, add);
    }
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= errorstate;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    /* Configure the MMC DPSM (Data Path State Machine) */
    config.DataTimeOut   = SDMMC_DATATIMEOUT;
    config.DataLength    = NumberOfBlocks * MMC_BLOCKSIZE;
    config.DataBlockSize = SDIO_DATABLOCK_SIZE_512B;
    config.TransferDir   = SDIO_TRANSFER_DIR_TO_CARD;
    config.TransferMode  = SDIO_TRANSFER_MODE_BLOCK;
    config.DPSM          = SDIO_DPSM_ENABLE;
    (void)SDIO_ConfigData(hmmc->Instance, &config);

    /* Write block(s) in polling mode */
    dataremaining = config.DataLength;
#if defined(SDIO_STA_STBITERR)
    while(!__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_TXUNDERR | SDIO_FLAG_DCRCFAIL | SDIO_FLAG_DTIMEOUT | SDIO_FLAG_DATAEND | SDIO_FLAG_STBITERR))
#else /* SDIO_STA_STBITERR not defined */
    while(!__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_TXUNDERR | SDIO_FLAG_DCRCFAIL | SDIO_FLAG_DTIMEOUT | SDIO_FLAG_DATAEND))
#endif /* SDIO_STA_STBITERR */
    {
      if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_TXFIFOHE) && (dataremaining > 0U))
      {
        /* Write data to SDIO Tx FIFO */
        for(count = 0U; count < 8U; count++)
        {
          data = (uint32_t)(*tempbuff);
          tempbuff++;
          dataremaining--;
          data |= ((uint32_t)(*tempbuff) << 8U);
          tempbuff++;
          dataremaining--;
          data |= ((uint32_t)(*tempbuff) << 16U);
          tempbuff++;
          dataremaining--;
          data |= ((uint32_t)(*tempbuff) << 24U);
          tempbuff++;
          dataremaining--;
          (void)SDIO_WriteFIFO(hmmc->Instance, &data);
        }
      }

      if(((HAL_GetTick()-tickstart) >=  Timeout) || (Timeout == 0U))
      {
        /* Clear all the static flags */
        __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
        hmmc->ErrorCode |= errorstate;
        hmmc->State = HAL_MMC_STATE_READY;
        return HAL_TIMEOUT;
      }
    }

    /* Send stop transmission command in case of multiblock write */
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DATAEND) && (NumberOfBlocks > 1U))
    {
      /* Send stop transmission command */
      errorstate = SDMMC_CmdStopTransfer(hmmc->Instance);
      if(errorstate != HAL_MMC_ERROR_NONE)
      {
        /* Clear all the static flags */
        __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
        hmmc->ErrorCode |= errorstate;
        hmmc->State = HAL_MMC_STATE_READY;
        return HAL_ERROR;
      }
    }

    /* Get error state */
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DTIMEOUT))
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_DATA_TIMEOUT;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }
    else if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DCRCFAIL))
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_DATA_CRC_FAIL;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }
    else if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_TXUNDERR))
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_TX_UNDERRUN;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }
    else
    {
      /* Nothing to do */
    }

    /* Clear all the static flags */
    __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_DATA_FLAGS);

    hmmc->State = HAL_MMC_STATE_READY;

    return HAL_OK;
  }
  else
  {
    hmmc->ErrorCode |= HAL_MMC_ERROR_BUSY;
    return HAL_ERROR;
  }
}

/**
  * @brief  Reads block(s) from a specified address in a card. The Data transfer
  *         is managed in interrupt mode.
  * @note   This API should be followed by a check on the card state through
  *         HAL_MMC_GetCardState().
  * @note   You could also check the IT transfer process through the MMC Rx
  *         interrupt event.
  * @param  hmmc: Pointer to MMC handle
  * @param  pData: Pointer to the buffer that will contain the received data
  * @param  BlockAdd: Block Address from where data is to be read
  * @param  NumberOfBlocks: Number of blocks to read.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_ReadBlocks_IT(MMC_HandleTypeDef *hmmc, uint8_t *pData, uint32_t BlockAdd, uint32_t NumberOfBlocks)
{
  SDIO_DataInitTypeDef config;
  uint32_t errorstate;
  uint32_t add = BlockAdd;

  if(NULL == pData)
  {
    hmmc->ErrorCode |= HAL_MMC_ERROR_PARAM;
    return HAL_ERROR;
  }

  if(hmmc->State == HAL_MMC_STATE_READY)
  {
    hmmc->ErrorCode = HAL_MMC_ERROR_NONE;

    if((BlockAdd + NumberOfBlocks) > (hmmc->MmcCard.LogBlockNbr))
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_ADDR_OUT_OF_RANGE;
      return HAL_ERROR;
    }

    hmmc->State = HAL_MMC_STATE_BUSY;

    /* Initialize data control register */
    hmmc->Instance->DCTRL = 0U;

    hmmc->pRxBuffPtr = pData;
    hmmc->RxXferSize = MMC_BLOCKSIZE * NumberOfBlocks;

#if defined(SDIO_STA_STBITERR)
    __HAL_MMC_ENABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_RXOVERR | SDIO_IT_DATAEND | SDIO_FLAG_RXFIFOHF | SDIO_IT_STBITERR));
#else /* SDIO_STA_STBITERR not defined */
    __HAL_MMC_ENABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_RXOVERR | SDIO_IT_DATAEND | SDIO_FLAG_RXFIFOHF));
#endif /* SDIO_STA_STBITERR */

    if ((hmmc->MmcCard.CardType) != MMC_HIGH_CAPACITY_CARD)
    {
      add *= 512U;
    }

    /* Configure the MMC DPSM (Data Path State Machine) */
    config.DataTimeOut   = SDMMC_DATATIMEOUT;
    config.DataLength    = MMC_BLOCKSIZE * NumberOfBlocks;
    config.DataBlockSize = SDIO_DATABLOCK_SIZE_512B;
    config.TransferDir   = SDIO_TRANSFER_DIR_TO_SDIO;
    config.TransferMode  = SDIO_TRANSFER_MODE_BLOCK;
    config.DPSM          = SDIO_DPSM_ENABLE;
    (void)SDIO_ConfigData(hmmc->Instance, &config);

    /* Read Blocks in IT mode */
    if(NumberOfBlocks > 1U)
    {
      hmmc->Context = (MMC_CONTEXT_READ_MULTIPLE_BLOCK | MMC_CONTEXT_IT);

      /* Read Multi Block command */
      errorstate = SDMMC_CmdReadMultiBlock(hmmc->Instance, add);
    }
    else
    {
      hmmc->Context = (MMC_CONTEXT_READ_SINGLE_BLOCK | MMC_CONTEXT_IT);

      /* Read Single Block command */
      errorstate = SDMMC_CmdReadSingleBlock(hmmc->Instance, add);
    }

    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= errorstate;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Writes block(s) to a specified address in a card. The Data transfer
  *         is managed in interrupt mode.
  * @note   This API should be followed by a check on the card state through
  *         HAL_MMC_GetCardState().
  * @note   You could also check the IT transfer process through the MMC Tx
  *         interrupt event.
  * @param  hmmc: Pointer to MMC handle
  * @param  pData: Pointer to the buffer that will contain the data to transmit
  * @param  BlockAdd: Block Address where data will be written
  * @param  NumberOfBlocks: Number of blocks to write
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_WriteBlocks_IT(MMC_HandleTypeDef *hmmc, uint8_t *pData, uint32_t BlockAdd, uint32_t NumberOfBlocks)
{
  SDIO_DataInitTypeDef config;
  uint32_t errorstate;
  uint32_t add = BlockAdd;

  if(NULL == pData)
  {
    hmmc->ErrorCode |= HAL_MMC_ERROR_PARAM;
    return HAL_ERROR;
  }

  if(hmmc->State == HAL_MMC_STATE_READY)
  {
    hmmc->ErrorCode = HAL_MMC_ERROR_NONE;

    if((BlockAdd + NumberOfBlocks) > (hmmc->MmcCard.LogBlockNbr))
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_ADDR_OUT_OF_RANGE;
      return HAL_ERROR;
    }

    hmmc->State = HAL_MMC_STATE_BUSY;

    /* Initialize data control register */
    hmmc->Instance->DCTRL = 0U;

    hmmc->pTxBuffPtr = pData;
    hmmc->TxXferSize = MMC_BLOCKSIZE * NumberOfBlocks;

    /* Enable transfer interrupts */
#if defined(SDIO_STA_STBITERR)
    __HAL_MMC_ENABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_TXUNDERR | SDIO_IT_DATAEND | SDIO_FLAG_TXFIFOHE | SDIO_IT_STBITERR));
#else /* SDIO_STA_STBITERR not defined */
    __HAL_MMC_ENABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_TXUNDERR | SDIO_IT_DATAEND | SDIO_FLAG_TXFIFOHE));
#endif /* SDIO_STA_STBITERR */

    if ((hmmc->MmcCard.CardType) != MMC_HIGH_CAPACITY_CARD)
    {
      add *= 512U;
    }

    /* Write Blocks in Polling mode */
    if(NumberOfBlocks > 1U)
    {
      hmmc->Context = (MMC_CONTEXT_WRITE_MULTIPLE_BLOCK| MMC_CONTEXT_IT);

      /* Write Multi Block command */
      errorstate = SDMMC_CmdWriteMultiBlock(hmmc->Instance, add);
    }
    else
    {
      hmmc->Context = (MMC_CONTEXT_WRITE_SINGLE_BLOCK | MMC_CONTEXT_IT);

      /* Write Single Block command */
      errorstate = SDMMC_CmdWriteSingleBlock(hmmc->Instance, add);
    }
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= errorstate;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    /* Configure the MMC DPSM (Data Path State Machine) */ 
    config.DataTimeOut   = SDMMC_DATATIMEOUT;
    config.DataLength    = MMC_BLOCKSIZE * NumberOfBlocks;
    config.DataBlockSize = SDIO_DATABLOCK_SIZE_512B;
    config.TransferDir   = SDIO_TRANSFER_DIR_TO_CARD;
    config.TransferMode  = SDIO_TRANSFER_MODE_BLOCK;
    config.DPSM          = SDIO_DPSM_ENABLE;
    (void)SDIO_ConfigData(hmmc->Instance, &config);
    
    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Reads block(s) from a specified address in a card. The Data transfer
  *         is managed by DMA mode.
  * @note   This API should be followed by a check on the card state through
  *         HAL_MMC_GetCardState().
  * @note   You could also check the DMA transfer process through the MMC Rx
  *         interrupt event.
  * @param  hmmc: Pointer MMC handle
  * @param  pData: Pointer to the buffer that will contain the received data
  * @param  BlockAdd: Block Address from where data is to be read
  * @param  NumberOfBlocks: Number of blocks to read.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_ReadBlocks_DMA(MMC_HandleTypeDef *hmmc, uint8_t *pData, uint32_t BlockAdd, uint32_t NumberOfBlocks)
{
  SDIO_DataInitTypeDef config;
  uint32_t errorstate;
  uint32_t add = BlockAdd;

  if(NULL == pData)
  {
    hmmc->ErrorCode |= HAL_MMC_ERROR_PARAM;
    return HAL_ERROR;
  }

  if(hmmc->State == HAL_MMC_STATE_READY)
  {
    hmmc->ErrorCode = HAL_DMA_ERROR_NONE;

    if((BlockAdd + NumberOfBlocks) > (hmmc->MmcCard.LogBlockNbr))
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_ADDR_OUT_OF_RANGE;
      return HAL_ERROR;
    }

    hmmc->State = HAL_MMC_STATE_BUSY;

    /* Initialize data control register */
    hmmc->Instance->DCTRL = 0U;

#if defined(SDIO_STA_STBITERR)
    __HAL_MMC_ENABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_RXOVERR | SDIO_IT_DATAEND | SDIO_IT_STBITERR));
#else /* SDIO_STA_STBITERR not defined */
    __HAL_MMC_ENABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_RXOVERR | SDIO_IT_DATAEND));
#endif /* SDIO_STA_STBITERR */

    /* Set the DMA transfer complete callback */
    hmmc->hdmarx->XferCpltCallback = MMC_DMAReceiveCplt;

    /* Set the DMA error callback */
    hmmc->hdmarx->XferErrorCallback = MMC_DMAError;

    /* Set the DMA Abort callback */
    hmmc->hdmarx->XferAbortCallback = NULL;

    if ((hmmc->MmcCard.CardType) != MMC_HIGH_CAPACITY_CARD)
    {
      add *= 512U;
    }

    /* Force DMA Direction */
    hmmc->hdmarx->Init.Direction = DMA_PERIPH_TO_MEMORY;
    MODIFY_REG(hmmc->hdmarx->Instance->CR, DMA_SxCR_DIR, hmmc->hdmarx->Init.Direction);

    /* Enable the DMA Channel */
    if(HAL_DMA_Start_IT(hmmc->hdmarx, (uint32_t)&hmmc->Instance->FIFO, (uint32_t)pData, (uint32_t)(MMC_BLOCKSIZE * NumberOfBlocks)/4) != HAL_OK)
    {
      __HAL_MMC_DISABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_RXOVERR | SDIO_IT_DATAEND));
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode = HAL_MMC_ERROR_DMA;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }
    else
    {
      /* Enable MMC DMA transfer */
      __HAL_MMC_DMA_ENABLE(hmmc);

      /* Configure the MMC DPSM (Data Path State Machine) */
      config.DataTimeOut   = SDMMC_DATATIMEOUT;
      config.DataLength    = MMC_BLOCKSIZE * NumberOfBlocks;
      config.DataBlockSize = SDIO_DATABLOCK_SIZE_512B;
      config.TransferDir   = SDIO_TRANSFER_DIR_TO_SDIO;
      config.TransferMode  = SDIO_TRANSFER_MODE_BLOCK;
      config.DPSM          = SDIO_DPSM_ENABLE;
      (void)SDIO_ConfigData(hmmc->Instance, &config);

      /* Read Blocks in DMA mode */
      if(NumberOfBlocks > 1U)
      {
        hmmc->Context = (MMC_CONTEXT_READ_MULTIPLE_BLOCK | MMC_CONTEXT_DMA);

        /* Read Multi Block command */
        errorstate = SDMMC_CmdReadMultiBlock(hmmc->Instance, add);
      }
      else
      {
        hmmc->Context = (MMC_CONTEXT_READ_SINGLE_BLOCK | MMC_CONTEXT_DMA);

        /* Read Single Block command */
        errorstate = SDMMC_CmdReadSingleBlock(hmmc->Instance, add);
      }
      if(errorstate != HAL_MMC_ERROR_NONE)
      {
        /* Clear all the static flags */
        __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS); 
        __HAL_MMC_DISABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_RXOVERR | SDIO_IT_DATAEND));
        hmmc->ErrorCode = errorstate;
        hmmc->State = HAL_MMC_STATE_READY;
        return HAL_ERROR;
      }

      return HAL_OK;
    }
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Writes block(s) to a specified address in a card. The Data transfer
  *         is managed by DMA mode.
  * @note   This API should be followed by a check on the card state through
  *         HAL_MMC_GetCardState().
  * @note   You could also check the DMA transfer process through the MMC Tx
  *         interrupt event.
  * @param  hmmc: Pointer to MMC handle
  * @param  pData: Pointer to the buffer that will contain the data to transmit
  * @param  BlockAdd: Block Address where data will be written
  * @param  NumberOfBlocks: Number of blocks to write
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_WriteBlocks_DMA(MMC_HandleTypeDef *hmmc, uint8_t *pData, uint32_t BlockAdd, uint32_t NumberOfBlocks)
{
  SDIO_DataInitTypeDef config;
  uint32_t errorstate;
  uint32_t add = BlockAdd;

  if(NULL == pData)
  {
    hmmc->ErrorCode |= HAL_MMC_ERROR_PARAM;
    return HAL_ERROR;
  }

  if(hmmc->State == HAL_MMC_STATE_READY)
  {
    hmmc->ErrorCode = HAL_MMC_ERROR_NONE;

    if((BlockAdd + NumberOfBlocks) > (hmmc->MmcCard.LogBlockNbr))
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_ADDR_OUT_OF_RANGE;
      return HAL_ERROR;
    }

    hmmc->State = HAL_MMC_STATE_BUSY;

    /* Initialize data control register */
    hmmc->Instance->DCTRL = 0U;

    /* Enable MMC Error interrupts */
#if defined(SDIO_STA_STBITERR)
	__HAL_MMC_ENABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_TXUNDERR | SDIO_IT_STBITERR));
#else /* SDIO_STA_STBITERR not defined */
	__HAL_MMC_ENABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_TXUNDERR));
#endif /* SDIO_STA_STBITERR */	

    /* Set the DMA transfer complete callback */
    hmmc->hdmatx->XferCpltCallback = MMC_DMATransmitCplt;

    /* Set the DMA error callback */
    hmmc->hdmatx->XferErrorCallback = MMC_DMAError;

    /* Set the DMA Abort callback */
    hmmc->hdmatx->XferAbortCallback = NULL;

    if ((hmmc->MmcCard.CardType) != MMC_HIGH_CAPACITY_CARD)
    {
      add *= 512U;
    }


    /* Write Blocks in Polling mode */
    if(NumberOfBlocks > 1U)
    {
      hmmc->Context = (MMC_CONTEXT_WRITE_MULTIPLE_BLOCK | MMC_CONTEXT_DMA);

      /* Write Multi Block command */
      errorstate = SDMMC_CmdWriteMultiBlock(hmmc->Instance, add);
    }
    else
    {
      hmmc->Context = (MMC_CONTEXT_WRITE_SINGLE_BLOCK | MMC_CONTEXT_DMA);

      /* Write Single Block command */
      errorstate = SDMMC_CmdWriteSingleBlock(hmmc->Instance, add);
    }
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      __HAL_MMC_DISABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_TXUNDERR | SDIO_IT_DATAEND));
      hmmc->ErrorCode |= errorstate;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    /* Enable SDIO DMA transfer */
    __HAL_MMC_DMA_ENABLE(hmmc);

    /* Force DMA Direction */
    hmmc->hdmatx->Init.Direction = DMA_MEMORY_TO_PERIPH;
    MODIFY_REG(hmmc->hdmatx->Instance->CR, DMA_SxCR_DIR, hmmc->hdmatx->Init.Direction);

    /* Enable the DMA Channel */
    if(HAL_DMA_Start_IT(hmmc->hdmatx, (uint32_t)pData, (uint32_t)&hmmc->Instance->FIFO, (uint32_t)(MMC_BLOCKSIZE * NumberOfBlocks)/4) != HAL_OK)
    {
      __HAL_MMC_DISABLE_IT(hmmc, (SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT | SDIO_IT_TXUNDERR | SDIO_IT_DATAEND));
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_DMA;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }
    else
    {    
      /* Configure the MMC DPSM (Data Path State Machine) */ 
      config.DataTimeOut   = SDMMC_DATATIMEOUT;
      config.DataLength    = MMC_BLOCKSIZE * NumberOfBlocks;
      config.DataBlockSize = SDIO_DATABLOCK_SIZE_512B;
      config.TransferDir   = SDIO_TRANSFER_DIR_TO_CARD;
      config.TransferMode  = SDIO_TRANSFER_MODE_BLOCK;
      config.DPSM          = SDIO_DPSM_ENABLE;
      (void)SDIO_ConfigData(hmmc->Instance, &config);

      return HAL_OK;
    }
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  Erases the specified memory area of the given MMC card.
  * @note   This API should be followed by a check on the card state through
  *         HAL_MMC_GetCardState().
  * @param  hmmc: Pointer to MMC handle
  * @param  BlockStartAdd: Start Block address
  * @param  BlockEndAdd: End Block address
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_Erase(MMC_HandleTypeDef *hmmc, uint32_t BlockStartAdd, uint32_t BlockEndAdd)
{
  uint32_t errorstate;
  uint32_t start_add = BlockStartAdd;
  uint32_t end_add = BlockEndAdd;

  if(hmmc->State == HAL_MMC_STATE_READY)
  {
    hmmc->ErrorCode = HAL_MMC_ERROR_NONE;

    if(end_add < start_add)
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_PARAM;
      return HAL_ERROR;
    }

    if(end_add > (hmmc->MmcCard.LogBlockNbr))
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_ADDR_OUT_OF_RANGE;
      return HAL_ERROR;
    }

    hmmc->State = HAL_MMC_STATE_BUSY;

    /* Check if the card command class supports erase command */
    if(((hmmc->MmcCard.Class) & SDIO_CCCC_ERASE) == 0U)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_REQUEST_NOT_APPLICABLE;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    if((SDIO_GetResponse(hmmc->Instance, SDIO_RESP1) & SDMMC_CARD_LOCKED) == SDMMC_CARD_LOCKED)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_LOCK_UNLOCK_FAILED;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    if ((hmmc->MmcCard.CardType) != MMC_HIGH_CAPACITY_CARD)
    {
      start_add *= 512U;
      end_add   *= 512U;
    }

    /* Send CMD35 MMC_ERASE_GRP_START with argument as addr  */
    errorstate = SDMMC_CmdEraseStartAdd(hmmc->Instance, start_add);
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= errorstate;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    /* Send CMD36 MMC_ERASE_GRP_END with argument as addr  */
    errorstate = SDMMC_CmdEraseEndAdd(hmmc->Instance, end_add);
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= errorstate;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    /* Send CMD38 ERASE */
    errorstate = SDMMC_CmdErase(hmmc->Instance);
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= errorstate;
      hmmc->State = HAL_MMC_STATE_READY;
      return HAL_ERROR;
    }

    hmmc->State = HAL_MMC_STATE_READY;

    return HAL_OK;
  }
  else
  {
    return HAL_BUSY;
  }
}

/**
  * @brief  This function handles MMC card interrupt request.
  * @param  hmmc: Pointer to MMC handle
  * @retval None
  */
void HAL_MMC_IRQHandler(MMC_HandleTypeDef *hmmc)
{
  uint32_t errorstate;
  uint32_t context = hmmc->Context;

  /* Check for SDIO interrupt flags */
  if((__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_RXFIFOHF) != RESET) && ((context & MMC_CONTEXT_IT) != 0U))
  {
    MMC_Read_IT(hmmc);
  }

  else if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DATAEND) != RESET)
  {
    __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_FLAG_DATAEND);

#if defined(SDIO_STA_STBITERR)
    __HAL_MMC_DISABLE_IT(hmmc, SDIO_IT_DATAEND | SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT|\
                             SDIO_IT_TXUNDERR| SDIO_IT_RXOVERR | SDIO_IT_STBITERR);
#else /* SDIO_STA_STBITERR not defined */
    __HAL_MMC_DISABLE_IT(hmmc, SDIO_IT_DATAEND  | SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT |\
                               SDIO_IT_TXUNDERR | SDIO_IT_RXOVERR  | SDIO_IT_TXFIFOHE |\
                               SDIO_IT_RXFIFOHF);
#endif /* SDIO_STA_STBITERR */
    
    hmmc->Instance->DCTRL &= ~(SDIO_DCTRL_DTEN);

    if((context & MMC_CONTEXT_DMA) != 0U)
    {
      if((context & MMC_CONTEXT_WRITE_MULTIPLE_BLOCK) != 0U)
      {
        errorstate = SDMMC_CmdStopTransfer(hmmc->Instance);
        if(errorstate != HAL_MMC_ERROR_NONE)
        {
          hmmc->ErrorCode |= errorstate;
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
          hmmc->ErrorCallback(hmmc);
#else
          HAL_MMC_ErrorCallback(hmmc);
#endif
        }
      }
      if(((context & MMC_CONTEXT_READ_SINGLE_BLOCK) == 0U) && ((context & MMC_CONTEXT_READ_MULTIPLE_BLOCK) == 0U))
      {
        /* Disable the DMA transfer for transmit request by setting the DMAEN bit
        in the MMC DCTRL register */
        hmmc->Instance->DCTRL &= (uint32_t)~((uint32_t)SDIO_DCTRL_DMAEN);
        
        hmmc->State = HAL_MMC_STATE_READY;
        
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
        hmmc->TxCpltCallback(hmmc);
#else
        HAL_MMC_TxCpltCallback(hmmc);
#endif
      }
    }
    else if((context & MMC_CONTEXT_IT) != 0U)
    {
      /* Stop Transfer for Write Multi blocks or Read Multi blocks */
      if(((context & MMC_CONTEXT_READ_MULTIPLE_BLOCK) != 0U) || ((context & MMC_CONTEXT_WRITE_MULTIPLE_BLOCK) != 0U))
      {
        errorstate = SDMMC_CmdStopTransfer(hmmc->Instance);
        if(errorstate != HAL_MMC_ERROR_NONE)
        {
          hmmc->ErrorCode |= errorstate;
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
          hmmc->ErrorCallback(hmmc);
#else
          HAL_MMC_ErrorCallback(hmmc);
#endif
        }
      }

      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_DATA_FLAGS);

      hmmc->State = HAL_MMC_STATE_READY;
      if(((context & MMC_CONTEXT_READ_SINGLE_BLOCK) != 0U) || ((context & MMC_CONTEXT_READ_MULTIPLE_BLOCK) != 0U))
      {
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
        hmmc->RxCpltCallback(hmmc);
#else
        HAL_MMC_RxCpltCallback(hmmc);
#endif
      }
      else
      {
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
        hmmc->TxCpltCallback(hmmc);
#else
        HAL_MMC_TxCpltCallback(hmmc);
#endif
      }
    }
    else
    {
      /* Nothing to do */
    }
  }

  else if((__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_TXFIFOHE) != RESET) && ((context & MMC_CONTEXT_IT) != 0U))
  {
    MMC_Write_IT(hmmc);
  }

#if defined(SDIO_STA_STBITERR)
  else if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DCRCFAIL | SDIO_FLAG_DTIMEOUT | SDIO_FLAG_RXOVERR | SDIO_FLAG_TXUNDERR | SDIO_FLAG_STBITERR) != RESET)
#else /* SDIO_STA_STBITERR not defined */
  else if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DCRCFAIL | SDIO_FLAG_DTIMEOUT | SDIO_FLAG_RXOVERR | SDIO_FLAG_TXUNDERR) != RESET)
#endif /* SDIO_STA_STBITERR */
  {
    /* Set Error code */
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DCRCFAIL) != RESET)
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_DATA_CRC_FAIL;
    }
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_DTIMEOUT) != RESET)
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_DATA_TIMEOUT;
    }
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_RXOVERR) != RESET)
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_RX_OVERRUN;
    }
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_TXUNDERR) != RESET)
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_TX_UNDERRUN;
    }
#if defined(SDIO_STA_STBITERR)
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_STBITERR) != RESET)
    {
      hmmc->ErrorCode |= HAL_MMC_ERROR_DATA_TIMEOUT;
    }
#endif /* SDIO_STA_STBITERR */

#if defined(SDIO_STA_STBITERR)
    /* Clear All flags */
    __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_DATA_FLAGS | SDIO_FLAG_STBITERR);

    /* Disable all interrupts */
    __HAL_MMC_DISABLE_IT(hmmc, SDIO_IT_DATAEND | SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT|\
                               SDIO_IT_TXUNDERR| SDIO_IT_RXOVERR | SDIO_IT_STBITERR);
#else /* SDIO_STA_STBITERR */
    /* Clear All flags */
    __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_DATA_FLAGS);
    
    /* Disable all interrupts */
    __HAL_MMC_DISABLE_IT(hmmc, SDIO_IT_DATAEND | SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT|\
                             SDIO_IT_TXUNDERR| SDIO_IT_RXOVERR);
#endif /* SDIO_STA_STBITERR */

    hmmc->ErrorCode |= SDMMC_CmdStopTransfer(hmmc->Instance);

    if((context & MMC_CONTEXT_IT) != 0U)
    {
      /* Set the MMC state to ready to be able to start again the process */
      hmmc->State = HAL_MMC_STATE_READY;
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
      hmmc->ErrorCallback(hmmc);
#else
      HAL_MMC_ErrorCallback(hmmc);
#endif /* USE_HAL_MMC_REGISTER_CALLBACKS */
    }
    else if((context & MMC_CONTEXT_DMA) != 0U)
    {
      /* Abort the MMC DMA Streams */
      if(hmmc->hdmatx != NULL)
      {
        /* Set the DMA Tx abort callback */
        hmmc->hdmatx->XferAbortCallback = MMC_DMATxAbort;
        /* Abort DMA in IT mode */
        if(HAL_DMA_Abort_IT(hmmc->hdmatx) != HAL_OK)
        {
          MMC_DMATxAbort(hmmc->hdmatx);
        }
      }
      else if(hmmc->hdmarx != NULL)
      {
        /* Set the DMA Rx abort callback */
        hmmc->hdmarx->XferAbortCallback = MMC_DMARxAbort;
        /* Abort DMA in IT mode */
        if(HAL_DMA_Abort_IT(hmmc->hdmarx) != HAL_OK)
        {
          MMC_DMARxAbort(hmmc->hdmarx);
        }
      }
      else
      {
        hmmc->ErrorCode = HAL_MMC_ERROR_NONE;
        hmmc->State = HAL_MMC_STATE_READY;
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
        hmmc->AbortCpltCallback(hmmc);
#else
        HAL_MMC_AbortCallback(hmmc);
#endif
      }
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

/**
  * @brief return the MMC state
  * @param hmmc: Pointer to mmc handle
  * @retval HAL state
  */
HAL_MMC_StateTypeDef HAL_MMC_GetState(MMC_HandleTypeDef *hmmc)
{
  return hmmc->State;
}

/**
* @brief  Return the MMC error code
* @param  hmmc : Pointer to a MMC_HandleTypeDef structure that contains
  *              the configuration information.
* @retval MMC Error Code
*/
uint32_t HAL_MMC_GetError(MMC_HandleTypeDef *hmmc)
{
  return hmmc->ErrorCode;
}

/**
  * @brief Tx Transfer completed callbacks
  * @param hmmc: Pointer to MMC handle
  * @retval None
  */
__weak void HAL_MMC_TxCpltCallback(MMC_HandleTypeDef *hmmc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hmmc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_MMC_TxCpltCallback can be implemented in the user file
   */
}

/**
  * @brief Rx Transfer completed callbacks
  * @param hmmc: Pointer MMC handle
  * @retval None
  */
__weak void HAL_MMC_RxCpltCallback(MMC_HandleTypeDef *hmmc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hmmc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_MMC_RxCpltCallback can be implemented in the user file
   */
}

/**
  * @brief MMC error callbacks
  * @param hmmc: Pointer MMC handle
  * @retval None
  */
__weak void HAL_MMC_ErrorCallback(MMC_HandleTypeDef *hmmc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hmmc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_MMC_ErrorCallback can be implemented in the user file
   */
}

/**
  * @brief MMC Abort callbacks
  * @param hmmc: Pointer MMC handle
  * @retval None
  */
__weak void HAL_MMC_AbortCallback(MMC_HandleTypeDef *hmmc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hmmc);

  /* NOTE : This function should not be modified, when the callback is needed,
            the HAL_MMC_AbortCallback can be implemented in the user file
   */
}

#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
/**
  * @brief  Register a User MMC Callback
  *         To be used instead of the weak (surcharged) predefined callback
  * @param hmmc : MMC handle
  * @param CallbackId : ID of the callback to be registered
  *        This parameter can be one of the following values:
  *          @arg @ref HAL_MMC_TX_CPLT_CB_ID    MMC Tx Complete Callback ID
  *          @arg @ref HAL_MMC_RX_CPLT_CB_ID    MMC Rx Complete Callback ID
  *          @arg @ref HAL_MMC_ERROR_CB_ID      MMC Error Callback ID
  *          @arg @ref HAL_MMC_ABORT_CB_ID      MMC Abort Callback ID
  *          @arg @ref HAL_MMC_MSP_INIT_CB_ID   MMC MspInit Callback ID
  *          @arg @ref HAL_MMC_MSP_DEINIT_CB_ID MMC MspDeInit Callback ID
  * @param pCallback : pointer to the Callback function
  * @retval status
  */
HAL_StatusTypeDef HAL_MMC_RegisterCallback(MMC_HandleTypeDef *hmmc, HAL_MMC_CallbackIDTypeDef CallbackId, pMMC_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if(pCallback == NULL)
  {
    /* Update the error code */
    hmmc->ErrorCode |= HAL_MMC_ERROR_INVALID_CALLBACK;
    return HAL_ERROR;
  }

  /* Process locked */
  __HAL_LOCK(hmmc);

  if(hmmc->State == HAL_MMC_STATE_READY)
  {
    switch (CallbackId)
    {
    case HAL_MMC_TX_CPLT_CB_ID :
      hmmc->TxCpltCallback = pCallback;
      break;
    case HAL_MMC_RX_CPLT_CB_ID :
      hmmc->RxCpltCallback = pCallback;
      break;
    case HAL_MMC_ERROR_CB_ID :
      hmmc->ErrorCallback = pCallback;
      break;
    case HAL_MMC_ABORT_CB_ID :
      hmmc->AbortCpltCallback = pCallback;
      break;
    case HAL_MMC_MSP_INIT_CB_ID :
      hmmc->MspInitCallback = pCallback;
      break;
    case HAL_MMC_MSP_DEINIT_CB_ID :
      hmmc->MspDeInitCallback = pCallback;
      break;
    default :
      /* Update the error code */
      hmmc->ErrorCode |= HAL_MMC_ERROR_INVALID_CALLBACK;
      /* update return status */
      status =  HAL_ERROR;
      break;
    }
  }
  else if (hmmc->State == HAL_MMC_STATE_RESET)
  {
    switch (CallbackId)
    {
    case HAL_MMC_MSP_INIT_CB_ID :
      hmmc->MspInitCallback = pCallback;
      break;
    case HAL_MMC_MSP_DEINIT_CB_ID :
      hmmc->MspDeInitCallback = pCallback;
      break;
    default :
      /* Update the error code */
      hmmc->ErrorCode |= HAL_MMC_ERROR_INVALID_CALLBACK;
      /* update return status */
      status =  HAL_ERROR;
      break;
    }
  }
  else
  {
    /* Update the error code */
    hmmc->ErrorCode |= HAL_MMC_ERROR_INVALID_CALLBACK;
    /* update return status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hmmc);
  return status;
}

/**
  * @brief  Unregister a User MMC Callback
  *         MMC Callback is redirected to the weak (surcharged) predefined callback
  * @param hmmc : MMC handle
  * @param CallbackId : ID of the callback to be unregistered
  *        This parameter can be one of the following values:
  *          @arg @ref HAL_MMC_TX_CPLT_CB_ID    MMC Tx Complete Callback ID
  *          @arg @ref HAL_MMC_RX_CPLT_CB_ID    MMC Rx Complete Callback ID
  *          @arg @ref HAL_MMC_ERROR_CB_ID      MMC Error Callback ID
  *          @arg @ref HAL_MMC_ABORT_CB_ID      MMC Abort Callback ID
  *          @arg @ref HAL_MMC_MSP_INIT_CB_ID   MMC MspInit Callback ID
  *          @arg @ref HAL_MMC_MSP_DEINIT_CB_ID MMC MspDeInit Callback ID
  * @retval status
  */
HAL_StatusTypeDef HAL_MMC_UnRegisterCallback(MMC_HandleTypeDef *hmmc, HAL_MMC_CallbackIDTypeDef CallbackId)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hmmc);

  if(hmmc->State == HAL_MMC_STATE_READY)
  {
    switch (CallbackId)
    {
    case HAL_MMC_TX_CPLT_CB_ID :
      hmmc->TxCpltCallback = HAL_MMC_TxCpltCallback;
      break;
    case HAL_MMC_RX_CPLT_CB_ID :
      hmmc->RxCpltCallback = HAL_MMC_RxCpltCallback;
      break;
    case HAL_MMC_ERROR_CB_ID :
      hmmc->ErrorCallback = HAL_MMC_ErrorCallback;
      break;
    case HAL_MMC_ABORT_CB_ID :
      hmmc->AbortCpltCallback = HAL_MMC_AbortCallback;
      break;
    case HAL_MMC_MSP_INIT_CB_ID :
      hmmc->MspInitCallback = HAL_MMC_MspInit;
      break;
    case HAL_MMC_MSP_DEINIT_CB_ID :
      hmmc->MspDeInitCallback = HAL_MMC_MspDeInit;
      break;
    default :
      /* Update the error code */
      hmmc->ErrorCode |= HAL_MMC_ERROR_INVALID_CALLBACK;
      /* update return status */
      status =  HAL_ERROR;
      break;
    }
  }
  else if (hmmc->State == HAL_MMC_STATE_RESET)
  {
    switch (CallbackId)
    {
    case HAL_MMC_MSP_INIT_CB_ID :
      hmmc->MspInitCallback = HAL_MMC_MspInit;
      break;
    case HAL_MMC_MSP_DEINIT_CB_ID :
      hmmc->MspDeInitCallback = HAL_MMC_MspDeInit;
      break;
    default :
      /* Update the error code */
      hmmc->ErrorCode |= HAL_MMC_ERROR_INVALID_CALLBACK;
      /* update return status */
      status =  HAL_ERROR;
      break;
    }
  }
  else
  {
    /* Update the error code */
    hmmc->ErrorCode |= HAL_MMC_ERROR_INVALID_CALLBACK;
    /* update return status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hmmc);
  return status;
}
#endif

/**
  * @}
  */

/** @addtogroup MMC_Exported_Functions_Group3
 *  @brief   management functions
 *
@verbatim
  ==============================================================================
                      ##### Peripheral Control functions #####
  ==============================================================================
  [..]
    This subsection provides a set of functions allowing to control the MMC card
    operations and get the related information

@endverbatim
  * @{
  */

/**
  * @brief  Returns information the information of the card which are stored on
  *         the CID register.
  * @param  hmmc: Pointer to MMC handle
  * @param  pCID: Pointer to a HAL_MMC_CIDTypedef structure that
  *         contains all CID register parameters
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_GetCardCID(MMC_HandleTypeDef *hmmc, HAL_MMC_CardCIDTypeDef *pCID)
{
  pCID->ManufacturerID = (uint8_t)((hmmc->CID[0] & 0xFF000000U) >> 24U);

  pCID->OEM_AppliID = (uint16_t)((hmmc->CID[0] & 0x00FFFF00U) >> 8U);

  pCID->ProdName1 = (((hmmc->CID[0] & 0x000000FFU) << 24U) | ((hmmc->CID[1] & 0xFFFFFF00U) >> 8U));

  pCID->ProdName2 = (uint8_t)(hmmc->CID[1] & 0x000000FFU);

  pCID->ProdRev = (uint8_t)((hmmc->CID[2] & 0xFF000000U) >> 24U);

  pCID->ProdSN = (((hmmc->CID[2] & 0x00FFFFFFU) << 8U) | ((hmmc->CID[3] & 0xFF000000U) >> 24U));

  pCID->Reserved1 = (uint8_t)((hmmc->CID[3] & 0x00F00000U) >> 20U);

  pCID->ManufactDate = (uint16_t)((hmmc->CID[3] & 0x000FFF00U) >> 8U);

  pCID->CID_CRC = (uint8_t)((hmmc->CID[3] & 0x000000FEU) >> 1U);

  pCID->Reserved2 = 1U;

  return HAL_OK;
}

/**
  * @brief  Returns information the information of the card which are stored on
  *         the CSD register.
  * @param  hmmc: Pointer to MMC handle
  * @param  pCSD: Pointer to a HAL_MMC_CardCSDTypeDef structure that
  *         contains all CSD register parameters
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_GetCardCSD(MMC_HandleTypeDef *hmmc, HAL_MMC_CardCSDTypeDef *pCSD)
{
  uint32_t block_nbr = 0;

  pCSD->CSDStruct = (uint8_t)((hmmc->CSD[0] & 0xC0000000U) >> 30U);

  pCSD->SysSpecVersion = (uint8_t)((hmmc->CSD[0] & 0x3C000000U) >> 26U);

  pCSD->Reserved1 = (uint8_t)((hmmc->CSD[0] & 0x03000000U) >> 24U);

  pCSD->TAAC = (uint8_t)((hmmc->CSD[0] & 0x00FF0000U) >> 16U);

  pCSD->NSAC = (uint8_t)((hmmc->CSD[0] & 0x0000FF00U) >> 8U);

  pCSD->MaxBusClkFrec = (uint8_t)(hmmc->CSD[0] & 0x000000FFU);

  pCSD->CardComdClasses = (uint16_t)((hmmc->CSD[1] & 0xFFF00000U) >> 20U);

  pCSD->RdBlockLen = (uint8_t)((hmmc->CSD[1] & 0x000F0000U) >> 16U);

  pCSD->PartBlockRead   = (uint8_t)((hmmc->CSD[1] & 0x00008000U) >> 15U);

  pCSD->WrBlockMisalign = (uint8_t)((hmmc->CSD[1] & 0x00004000U) >> 14U);

  pCSD->RdBlockMisalign = (uint8_t)((hmmc->CSD[1] & 0x00002000U) >> 13U);

  pCSD->DSRImpl = (uint8_t)((hmmc->CSD[1] & 0x00001000U) >> 12U);

  pCSD->Reserved2 = 0U; /*!< Reserved */

  pCSD->DeviceSize = (((hmmc->CSD[1] & 0x000003FFU) << 2U) | ((hmmc->CSD[2] & 0xC0000000U) >> 30U));

  pCSD->MaxRdCurrentVDDMin = (uint8_t)((hmmc->CSD[2] & 0x38000000U) >> 27U);

  pCSD->MaxRdCurrentVDDMax = (uint8_t)((hmmc->CSD[2] & 0x07000000U) >> 24U);

  pCSD->MaxWrCurrentVDDMin = (uint8_t)((hmmc->CSD[2] & 0x00E00000U) >> 21U);

  pCSD->MaxWrCurrentVDDMax = (uint8_t)((hmmc->CSD[2] & 0x001C0000U) >> 18U);

  pCSD->DeviceSizeMul = (uint8_t)((hmmc->CSD[2] & 0x00038000U) >> 15U);

  if(MMC_ReadExtCSD(hmmc, &block_nbr, 212, 0x0FFFFFFFU) != HAL_OK) /* Field SEC_COUNT [215:212] */
  {
    return HAL_ERROR;
  }

  if(hmmc->MmcCard.CardType == MMC_LOW_CAPACITY_CARD)
  {
    hmmc->MmcCard.BlockNbr  = (pCSD->DeviceSize + 1U) ;
    hmmc->MmcCard.BlockNbr *= (1UL << ((pCSD->DeviceSizeMul & 0x07U) + 2U));
    hmmc->MmcCard.BlockSize = (1UL << (pCSD->RdBlockLen & 0x0FU));
    hmmc->MmcCard.LogBlockNbr =  (hmmc->MmcCard.BlockNbr) * ((hmmc->MmcCard.BlockSize) / 512U);
    hmmc->MmcCard.LogBlockSize = 512U;
  }
  else if(hmmc->MmcCard.CardType == MMC_HIGH_CAPACITY_CARD)
  {
    hmmc->MmcCard.BlockNbr = block_nbr;
    hmmc->MmcCard.LogBlockNbr = hmmc->MmcCard.BlockNbr;
    hmmc->MmcCard.BlockSize = 512U;
    hmmc->MmcCard.LogBlockSize = hmmc->MmcCard.BlockSize;
  }
  else
  {
    /* Clear all the static flags */
    __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
    hmmc->ErrorCode |= HAL_MMC_ERROR_UNSUPPORTED_FEATURE;
    hmmc->State = HAL_MMC_STATE_READY;
    return HAL_ERROR;
  }

  pCSD->EraseGrSize = (uint8_t)((hmmc->CSD[2] & 0x00004000U) >> 14U);

  pCSD->EraseGrMul = (uint8_t)((hmmc->CSD[2] & 0x00003F80U) >> 7U);

  pCSD->WrProtectGrSize = (uint8_t)(hmmc->CSD[2] & 0x0000007FU);

  pCSD->WrProtectGrEnable = (uint8_t)((hmmc->CSD[3] & 0x80000000U) >> 31U);

  pCSD->ManDeflECC = (uint8_t)((hmmc->CSD[3] & 0x60000000U) >> 29U);

  pCSD->WrSpeedFact = (uint8_t)((hmmc->CSD[3] & 0x1C000000U) >> 26U);

  pCSD->MaxWrBlockLen= (uint8_t)((hmmc->CSD[3] & 0x03C00000U) >> 22U);

  pCSD->WriteBlockPaPartial = (uint8_t)((hmmc->CSD[3] & 0x00200000U) >> 21U);

  pCSD->Reserved3 = 0;

  pCSD->ContentProtectAppli = (uint8_t)((hmmc->CSD[3] & 0x00010000U) >> 16U);

  pCSD->FileFormatGroup = (uint8_t)((hmmc->CSD[3] & 0x00008000U) >> 15U);

  pCSD->CopyFlag = (uint8_t)((hmmc->CSD[3] & 0x00004000U) >> 14U);

  pCSD->PermWrProtect = (uint8_t)((hmmc->CSD[3] & 0x00002000U) >> 13U);

  pCSD->TempWrProtect = (uint8_t)((hmmc->CSD[3] & 0x00001000U) >> 12U);

  pCSD->FileFormat = (uint8_t)((hmmc->CSD[3] & 0x00000C00U) >> 10U);

  pCSD->ECC= (uint8_t)((hmmc->CSD[3] & 0x00000300U) >> 8U);

  pCSD->CSD_CRC = (uint8_t)((hmmc->CSD[3] & 0x000000FEU) >> 1U);

  pCSD->Reserved4 = 1;

  return HAL_OK;
}

/**
  * @brief  Gets the MMC card info.
  * @param  hmmc: Pointer to MMC handle
  * @param  pCardInfo: Pointer to the HAL_MMC_CardInfoTypeDef structure that
  *         will contain the MMC card status information
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_GetCardInfo(MMC_HandleTypeDef *hmmc, HAL_MMC_CardInfoTypeDef *pCardInfo)
{
  pCardInfo->CardType     = (uint32_t)(hmmc->MmcCard.CardType);
  pCardInfo->Class        = (uint32_t)(hmmc->MmcCard.Class);
  pCardInfo->RelCardAdd   = (uint32_t)(hmmc->MmcCard.RelCardAdd);
  pCardInfo->BlockNbr     = (uint32_t)(hmmc->MmcCard.BlockNbr);
  pCardInfo->BlockSize    = (uint32_t)(hmmc->MmcCard.BlockSize);
  pCardInfo->LogBlockNbr  = (uint32_t)(hmmc->MmcCard.LogBlockNbr);
  pCardInfo->LogBlockSize = (uint32_t)(hmmc->MmcCard.LogBlockSize);

  return HAL_OK;
}

/**
  * @brief  Enables wide bus operation for the requested card if supported by
  *         card.
  * @param  hmmc: Pointer to MMC handle
  * @param  WideMode: Specifies the MMC card wide bus mode
  *          This parameter can be one of the following values:
  *            @arg SDIO_BUS_WIDE_8B: 8-bit data transfer
  *            @arg SDIO_BUS_WIDE_4B: 4-bit data transfer
  *            @arg SDIO_BUS_WIDE_1B: 1-bit data transfer
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_ConfigWideBusOperation(MMC_HandleTypeDef *hmmc, uint32_t WideMode)
{
  __IO uint32_t count = 0U;
  SDIO_InitTypeDef Init;
  uint32_t errorstate;
  uint32_t response = 0U, busy = 0U;

  /* Check the parameters */
  assert_param(IS_SDIO_BUS_WIDE(WideMode));

  /* Change State */
  hmmc->State = HAL_MMC_STATE_BUSY;

  /* Update Clock for Bus mode update */
  Init.ClockEdge           = SDIO_CLOCK_EDGE_RISING;
  Init.ClockBypass         = SDIO_CLOCK_BYPASS_DISABLE;
  Init.ClockPowerSave      = SDIO_CLOCK_POWER_SAVE_DISABLE;
  Init.BusWide             = WideMode;
  Init.HardwareFlowControl = SDIO_HARDWARE_FLOW_CONTROL_DISABLE;
  Init.ClockDiv            = SDIO_INIT_CLK_DIV;
  /* Initialize SDIO*/
  (void)SDIO_Init(hmmc->Instance, Init); 

  if(WideMode == SDIO_BUS_WIDE_8B)
  {
    errorstate = SDMMC_CmdSwitch(hmmc->Instance, 0x03B70200U);
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      hmmc->ErrorCode |= errorstate;
    }
  }
  else if(WideMode == SDIO_BUS_WIDE_4B)
  {
    errorstate = SDMMC_CmdSwitch(hmmc->Instance, 0x03B70100U);
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      hmmc->ErrorCode |= errorstate;
    }
  }
  else if(WideMode == SDIO_BUS_WIDE_1B)
  {
    errorstate = SDMMC_CmdSwitch(hmmc->Instance, 0x03B70000U);
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      hmmc->ErrorCode |= errorstate;
    }
  }
  else
  {
    /* WideMode is not a valid argument*/
    hmmc->ErrorCode |= HAL_MMC_ERROR_PARAM;
  }

  /* Check for switch error and violation of the trial number of sending CMD 13 */
  while(busy == 0U)
  {
    if(count == SDMMC_MAX_TRIAL)
    {
      hmmc->State = HAL_MMC_STATE_READY;
      hmmc->ErrorCode |= HAL_MMC_ERROR_REQUEST_NOT_APPLICABLE;
      return HAL_ERROR;
    }
    count++;

    /* While card is not ready for data and trial number for sending CMD13 is not exceeded */
    errorstate = SDMMC_CmdSendStatus(hmmc->Instance, (uint32_t)(((uint32_t)hmmc->MmcCard.RelCardAdd) << 16U));
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      hmmc->ErrorCode |= errorstate;
    }

    /* Get command response */
    response = SDIO_GetResponse(hmmc->Instance, SDIO_RESP1);

    /* Get operating voltage*/
    busy = (((response >> 7U) == 1U) ? 0U : 1U);
  }

  /* While card is not ready for data and trial number for sending CMD13 is not exceeded */
  count = SDMMC_DATATIMEOUT;
  while((response & 0x00000100U) == 0U)
  {
    if(count == 0U)
    {
      hmmc->State = HAL_MMC_STATE_READY;
      hmmc->ErrorCode |= HAL_MMC_ERROR_REQUEST_NOT_APPLICABLE;
      return HAL_ERROR;
    }
    count--;

    /* While card is not ready for data and trial number for sending CMD13 is not exceeded */
    errorstate = SDMMC_CmdSendStatus(hmmc->Instance, (uint32_t)(((uint32_t)hmmc->MmcCard.RelCardAdd) << 16U));
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      hmmc->ErrorCode |= errorstate;
    }

    /* Get command response */
    response = SDIO_GetResponse(hmmc->Instance, SDIO_RESP1);
  }

  if(hmmc->ErrorCode != HAL_MMC_ERROR_NONE)
  {
    /* Clear all the static flags */
    __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
    hmmc->State = HAL_MMC_STATE_READY;
    return HAL_ERROR;
  }
  else
  {
    /* Configure the SDIO peripheral */
    Init.ClockEdge           = hmmc->Init.ClockEdge;
    Init.ClockBypass         = hmmc->Init.ClockBypass;
    Init.ClockPowerSave      = hmmc->Init.ClockPowerSave;
    Init.BusWide             = WideMode;
    Init.HardwareFlowControl = hmmc->Init.HardwareFlowControl;
    Init.ClockDiv            = hmmc->Init.ClockDiv;
    (void)SDIO_Init(hmmc->Instance, Init);
  }

  /* Change State */
  hmmc->State = HAL_MMC_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  Gets the current mmc card data state.
  * @param  hmmc: pointer to MMC handle
  * @retval Card state
  */
HAL_MMC_CardStateTypeDef HAL_MMC_GetCardState(MMC_HandleTypeDef *hmmc)
{
  uint32_t cardstate;
  uint32_t errorstate;
  uint32_t resp1 = 0U;

  errorstate = MMC_SendStatus(hmmc, &resp1);
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    hmmc->ErrorCode |= errorstate;
  }

  cardstate = ((resp1 >> 9U) & 0x0FU);

  return (HAL_MMC_CardStateTypeDef)cardstate;
}

/**
  * @brief  Abort the current transfer and disable the MMC.
  * @param  hmmc: pointer to a MMC_HandleTypeDef structure that contains
  *                the configuration information for MMC module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_Abort(MMC_HandleTypeDef *hmmc)
{
  HAL_MMC_CardStateTypeDef CardState;

  /* DIsable All interrupts */
  __HAL_MMC_DISABLE_IT(hmmc, SDIO_IT_DATAEND | SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT|\
                             SDIO_IT_TXUNDERR| SDIO_IT_RXOVERR);

  /* Clear All flags */
  __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_DATA_FLAGS);

  if((hmmc->hdmatx != NULL) || (hmmc->hdmarx != NULL))
  {
    /* Disable the MMC DMA request */
    hmmc->Instance->DCTRL &= (uint32_t)~((uint32_t)SDIO_DCTRL_DMAEN);
    
    /* Abort the MMC DMA Tx Stream */
    if(hmmc->hdmatx != NULL)
    {
      if(HAL_DMA_Abort(hmmc->hdmatx) != HAL_OK)
      {
        hmmc->ErrorCode |= HAL_MMC_ERROR_DMA;
      }
    }
    /* Abort the MMC DMA Rx Stream */
    if(hmmc->hdmarx != NULL)
    {
      if(HAL_DMA_Abort(hmmc->hdmarx) != HAL_OK)
      {
        hmmc->ErrorCode |= HAL_MMC_ERROR_DMA;
      }
    }
  }

  hmmc->State = HAL_MMC_STATE_READY;

  /* Initialize the MMC operation */
  hmmc->Context = MMC_CONTEXT_NONE;

  CardState = HAL_MMC_GetCardState(hmmc);
  if((CardState == HAL_MMC_CARD_RECEIVING) || (CardState == HAL_MMC_CARD_SENDING))
  {
    hmmc->ErrorCode = SDMMC_CmdStopTransfer(hmmc->Instance);
  }
  if(hmmc->ErrorCode != HAL_MMC_ERROR_NONE)
  {
    return HAL_ERROR;
  }
  return HAL_OK;
}

/**
  * @brief  Abort the current transfer and disable the MMC (IT mode).
  * @param  hmmc: pointer to a MMC_HandleTypeDef structure that contains
  *                the configuration information for MMC module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_MMC_Abort_IT(MMC_HandleTypeDef *hmmc)
{
  HAL_MMC_CardStateTypeDef CardState;

  /* DIsable All interrupts */
  __HAL_MMC_DISABLE_IT(hmmc, SDIO_IT_DATAEND | SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT|\
                           SDIO_IT_TXUNDERR| SDIO_IT_RXOVERR);

  /* Clear All flags */
  __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_DATA_FLAGS);

  if((hmmc->hdmatx != NULL) || (hmmc->hdmarx != NULL))
  {
    /* Disable the MMC DMA request */
    hmmc->Instance->DCTRL &= (uint32_t)~((uint32_t)SDIO_DCTRL_DMAEN);
    
    /* Abort the MMC DMA Tx Stream */
    if(hmmc->hdmatx != NULL)
    {
      hmmc->hdmatx->XferAbortCallback =  MMC_DMATxAbort;
      if(HAL_DMA_Abort_IT(hmmc->hdmatx) != HAL_OK)
      {
        hmmc->hdmatx = NULL;
      }
    }
    /* Abort the MMC DMA Rx Stream */
    if(hmmc->hdmarx != NULL)
    {
      hmmc->hdmarx->XferAbortCallback =  MMC_DMARxAbort;
      if(HAL_DMA_Abort_IT(hmmc->hdmarx) != HAL_OK)
      {
        hmmc->hdmarx = NULL;
      }
    }
  }
  
  /* No transfer ongoing on both DMA channels*/
  if((hmmc->hdmatx == NULL) && (hmmc->hdmarx == NULL))
  {
    CardState = HAL_MMC_GetCardState(hmmc);
    hmmc->State = HAL_MMC_STATE_READY;

    if((CardState == HAL_MMC_CARD_RECEIVING) || (CardState == HAL_MMC_CARD_SENDING))
    {
      hmmc->ErrorCode = SDMMC_CmdStopTransfer(hmmc->Instance);
    }
    if(hmmc->ErrorCode != HAL_MMC_ERROR_NONE)
    {
      return HAL_ERROR;
    }
    else
    {
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
      hmmc->AbortCpltCallback(hmmc);
#else
      HAL_MMC_AbortCallback(hmmc);
#endif
    }
  }

  return HAL_OK;
}

/**
  * @}
  */

/**
  * @}
  */

/* Private function ----------------------------------------------------------*/
/** @addtogroup MMC_Private_Functions
  * @{
  */

/**
  * @brief  DMA MMC transmit process complete callback 
  * @param  hdma: DMA handle
  * @retval None
  */
static void MMC_DMATransmitCplt(DMA_HandleTypeDef *hdma)     
{
  MMC_HandleTypeDef* hmmc = (MMC_HandleTypeDef* )(hdma->Parent);
  
  /* Enable DATAEND Interrupt */
  __HAL_MMC_ENABLE_IT(hmmc, (SDIO_IT_DATAEND));
}

/**
  * @brief  DMA MMC receive process complete callback 
  * @param  hdma: DMA handle
  * @retval None
  */
static void MMC_DMAReceiveCplt(DMA_HandleTypeDef *hdma)  
{
  MMC_HandleTypeDef* hmmc = (MMC_HandleTypeDef* )(hdma->Parent);
  uint32_t errorstate;
  
  /* Send stop command in multiblock write */
  if(hmmc->Context == (MMC_CONTEXT_READ_MULTIPLE_BLOCK | MMC_CONTEXT_DMA))
  {
    errorstate = SDMMC_CmdStopTransfer(hmmc->Instance);
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      hmmc->ErrorCode |= errorstate;
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
      hmmc->ErrorCallback(hmmc);
#else
      HAL_MMC_ErrorCallback(hmmc);
#endif
    }
  }
  
  /* Disable the DMA transfer for transmit request by setting the DMAEN bit
  in the MMC DCTRL register */
  hmmc->Instance->DCTRL &= (uint32_t)~((uint32_t)SDIO_DCTRL_DMAEN);
  
  /* Clear all the static flags */
  __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_DATA_FLAGS);
  
  hmmc->State = HAL_MMC_STATE_READY;

#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
  hmmc->RxCpltCallback(hmmc);
#else
  HAL_MMC_RxCpltCallback(hmmc);
#endif
}

/**
  * @brief  DMA MMC communication error callback 
  * @param  hdma: DMA handle
  * @retval None
  */
static void MMC_DMAError(DMA_HandleTypeDef *hdma)   
{
  MMC_HandleTypeDef* hmmc = (MMC_HandleTypeDef* )(hdma->Parent);
  HAL_MMC_CardStateTypeDef CardState;
  uint32_t RxErrorCode, TxErrorCode;
  
  /* if DMA error is FIFO error ignore it */
  if(HAL_DMA_GetError(hdma) != HAL_DMA_ERROR_FE)
  {
    RxErrorCode = hmmc->hdmarx->ErrorCode;
    TxErrorCode = hmmc->hdmatx->ErrorCode;  
    if((RxErrorCode == HAL_DMA_ERROR_TE) || (TxErrorCode == HAL_DMA_ERROR_TE))
    {
      /* Clear All flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      
      /* Disable All interrupts */
      __HAL_MMC_DISABLE_IT(hmmc, SDIO_IT_DATAEND | SDIO_IT_DCRCFAIL | SDIO_IT_DTIMEOUT|\
        SDIO_IT_TXUNDERR| SDIO_IT_RXOVERR);
      
      hmmc->ErrorCode |= HAL_MMC_ERROR_DMA;
      CardState = HAL_MMC_GetCardState(hmmc);
      if((CardState == HAL_MMC_CARD_RECEIVING) || (CardState == HAL_MMC_CARD_SENDING))
      {
        hmmc->ErrorCode |= SDMMC_CmdStopTransfer(hmmc->Instance);
      }
      
      hmmc->State= HAL_MMC_STATE_READY;
    }
    
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
    hmmc->ErrorCallback(hmmc);
#else
    HAL_MMC_ErrorCallback(hmmc);
#endif
  }
}

/**
  * @brief  DMA MMC Tx Abort callback 
  * @param  hdma: DMA handle
  * @retval None
  */
static void MMC_DMATxAbort(DMA_HandleTypeDef *hdma)   
{
  MMC_HandleTypeDef* hmmc = (MMC_HandleTypeDef* )(hdma->Parent);
  HAL_MMC_CardStateTypeDef CardState;
  
  if(hmmc->hdmatx != NULL)
  {
    hmmc->hdmatx = NULL;
  }
  
  /* All DMA channels are aborted */
  if(hmmc->hdmarx == NULL)
  {
    CardState = HAL_MMC_GetCardState(hmmc);
    hmmc->ErrorCode = HAL_MMC_ERROR_NONE;
    hmmc->State = HAL_MMC_STATE_READY;
    if((CardState == HAL_MMC_CARD_RECEIVING) || (CardState == HAL_MMC_CARD_SENDING))
    {
      hmmc->ErrorCode |= SDMMC_CmdStopTransfer(hmmc->Instance);
      
      if(hmmc->ErrorCode != HAL_MMC_ERROR_NONE)
      {
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
        hmmc->AbortCpltCallback(hmmc);
#else
        HAL_MMC_AbortCallback(hmmc);
#endif
      }
      else
      {
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
        hmmc->ErrorCallback(hmmc);
#else
        HAL_MMC_ErrorCallback(hmmc);
#endif
      }
    }
  }
}

/**
  * @brief  DMA MMC Rx Abort callback 
  * @param  hdma: DMA handle
  * @retval None
  */
static void MMC_DMARxAbort(DMA_HandleTypeDef *hdma)   
{
  MMC_HandleTypeDef* hmmc = (MMC_HandleTypeDef* )(hdma->Parent);
  HAL_MMC_CardStateTypeDef CardState;
  
  if(hmmc->hdmarx != NULL)
  {
    hmmc->hdmarx = NULL;
  }
  
  /* All DMA channels are aborted */
  if(hmmc->hdmatx == NULL)
  {
    CardState = HAL_MMC_GetCardState(hmmc);
    hmmc->ErrorCode = HAL_MMC_ERROR_NONE;
    hmmc->State = HAL_MMC_STATE_READY;
    if((CardState == HAL_MMC_CARD_RECEIVING) || (CardState == HAL_MMC_CARD_SENDING))
    {
      hmmc->ErrorCode |= SDMMC_CmdStopTransfer(hmmc->Instance);
      
      if(hmmc->ErrorCode != HAL_MMC_ERROR_NONE)
      {
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
        hmmc->AbortCpltCallback(hmmc);
#else
        HAL_MMC_AbortCallback(hmmc);
#endif
      }
      else
      {
#if defined (USE_HAL_MMC_REGISTER_CALLBACKS) && (USE_HAL_MMC_REGISTER_CALLBACKS == 1U)
        hmmc->ErrorCallback(hmmc);
#else
        HAL_MMC_ErrorCallback(hmmc);
#endif
      }
    }
  }
}

/**
  * @brief  Initializes the mmc card.
  * @param  hmmc: Pointer to MMC handle
  * @retval MMC Card error state
  */
static uint32_t MMC_InitCard(MMC_HandleTypeDef *hmmc)
{
  HAL_MMC_CardCSDTypeDef CSD;
  uint32_t errorstate;
  uint16_t mmc_rca = 1U;

  /* Check the power State */
  if(SDIO_GetPowerState(hmmc->Instance) == 0U)
  {
    /* Power off */
    return HAL_MMC_ERROR_REQUEST_NOT_APPLICABLE;
  }

  /* Send CMD2 ALL_SEND_CID */
  errorstate = SDMMC_CmdSendCID(hmmc->Instance);
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    return errorstate;
  }
  else
  {
    /* Get Card identification number data */
    hmmc->CID[0U] = SDIO_GetResponse(hmmc->Instance, SDIO_RESP1);
    hmmc->CID[1U] = SDIO_GetResponse(hmmc->Instance, SDIO_RESP2);
    hmmc->CID[2U] = SDIO_GetResponse(hmmc->Instance, SDIO_RESP3);
    hmmc->CID[3U] = SDIO_GetResponse(hmmc->Instance, SDIO_RESP4);
  }

  /* Send CMD3 SET_REL_ADDR with argument 0 */
  /* MMC Card publishes its RCA. */
  errorstate = SDMMC_CmdSetRelAdd(hmmc->Instance, &mmc_rca);
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    return errorstate;
  }

  /* Get the MMC card RCA */
  hmmc->MmcCard.RelCardAdd = mmc_rca;

  /* Send CMD9 SEND_CSD with argument as card's RCA */
  errorstate = SDMMC_CmdSendCSD(hmmc->Instance, (uint32_t)(hmmc->MmcCard.RelCardAdd << 16U));
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    return errorstate;
  }
  else
  {
    /* Get Card Specific Data */
    hmmc->CSD[0U] = SDIO_GetResponse(hmmc->Instance, SDIO_RESP1);
    hmmc->CSD[1U] = SDIO_GetResponse(hmmc->Instance, SDIO_RESP2);
    hmmc->CSD[2U] = SDIO_GetResponse(hmmc->Instance, SDIO_RESP3);
    hmmc->CSD[3U] = SDIO_GetResponse(hmmc->Instance, SDIO_RESP4);
  }

  /* Get the Card Class */
  hmmc->MmcCard.Class = (SDIO_GetResponse(hmmc->Instance, SDIO_RESP2) >> 20U);

  /* Get CSD parameters */
  if (HAL_MMC_GetCardCSD(hmmc, &CSD) != HAL_OK)
  {
    return hmmc->ErrorCode;
  }

  /* Select the Card */
  errorstate = SDMMC_CmdSelDesel(hmmc->Instance, (uint32_t)(((uint32_t)hmmc->MmcCard.RelCardAdd) << 16U));
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    return errorstate;
  }

  /* Configure SDIO peripheral interface */
  (void)SDIO_Init(hmmc->Instance, hmmc->Init);

  /* All cards are initialized */
  return HAL_MMC_ERROR_NONE;
}

/**
  * @brief  Enquires cards about their operating voltage and configures clock
  *         controls and stores MMC information that will be needed in future
  *         in the MMC handle.
  * @param  hmmc: Pointer to MMC handle
  * @retval error state
  */
static uint32_t MMC_PowerON(MMC_HandleTypeDef *hmmc)
{
  __IO uint32_t count = 0U;
  uint32_t response = 0U, validvoltage = 0U;
  uint32_t errorstate;

  /* CMD0: GO_IDLE_STATE */
  errorstate = SDMMC_CmdGoIdleState(hmmc->Instance);
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    return errorstate;
  }

  while(validvoltage == 0U)
  {
    if(count++ == SDMMC_MAX_VOLT_TRIAL)
    {
      return HAL_MMC_ERROR_INVALID_VOLTRANGE;
    }

    /* SEND CMD1 APP_CMD with MMC_HIGH_VOLTAGE_RANGE(0xC0FF8000) as argument */
    errorstate = SDMMC_CmdOpCondition(hmmc->Instance, eMMC_HIGH_VOLTAGE_RANGE);
    if(errorstate != HAL_MMC_ERROR_NONE)
    {
      return HAL_MMC_ERROR_UNSUPPORTED_FEATURE;
    }

    /* Get command response */
    response = SDIO_GetResponse(hmmc->Instance, SDIO_RESP1);

    /* Get operating voltage*/
    validvoltage = (((response >> 31U) == 1U) ? 1U : 0U);
  }

  /* When power routine is finished and command returns valid voltage */
  if (((response & (0xFF000000U)) >> 24U) == 0xC0U)
  {
    hmmc->MmcCard.CardType = MMC_HIGH_CAPACITY_CARD;
  }
  else
  {
    hmmc->MmcCard.CardType = MMC_LOW_CAPACITY_CARD;
  }

  return HAL_MMC_ERROR_NONE;
}

/**
  * @brief  Turns the SDIO output signals off.
  * @param  hmmc: Pointer to MMC handle
  * @retval None
  */
static void MMC_PowerOFF(MMC_HandleTypeDef *hmmc)
{
  /* Set Power State to OFF */
  (void)SDIO_PowerState_OFF(hmmc->Instance);
}

/**
  * @brief  Returns the current card's status.
  * @param  hmmc: Pointer to MMC handle
  * @param  pCardStatus: pointer to the buffer that will contain the MMC card
  *         status (Card Status register)
  * @retval error state
  */
static uint32_t MMC_SendStatus(MMC_HandleTypeDef *hmmc, uint32_t *pCardStatus)
{
  uint32_t errorstate;

  if(pCardStatus == NULL)
  {
    return HAL_MMC_ERROR_PARAM;
  }

  /* Send Status command */
  errorstate = SDMMC_CmdSendStatus(hmmc->Instance, (uint32_t)(hmmc->MmcCard.RelCardAdd << 16U));
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    return errorstate;
  }

  /* Get MMC card status */
  *pCardStatus = SDIO_GetResponse(hmmc->Instance, SDIO_RESP1);

  return HAL_MMC_ERROR_NONE;
}

/**
  * @brief  Reads extended CSD register to get the sectors number of the device
  * @param  hmmc: Pointer to MMC handle
  * @param  pFieldData: Pointer to the read buffer
  * @param  FieldIndex: Index of the field to be read
  * @param  Timeout: Specify timeout value
  * @retval HAL status
  */
static uint32_t MMC_ReadExtCSD(MMC_HandleTypeDef *hmmc, uint32_t *pFieldData, uint16_t FieldIndex, uint32_t Timeout)
{
  SDIO_DataInitTypeDef config;
  uint32_t errorstate;
  uint32_t tickstart = HAL_GetTick();
  uint32_t count;
  uint32_t i = 0;
  uint32_t tmp_data;

  hmmc->ErrorCode = HAL_MMC_ERROR_NONE;

  /* Initialize data control register */
  hmmc->Instance->DCTRL = 0;

  /* Configure the MMC DPSM (Data Path State Machine) */
  config.DataTimeOut   = SDMMC_DATATIMEOUT;
  config.DataLength    = 512;
  config.DataBlockSize = SDIO_DATABLOCK_SIZE_512B;
  config.TransferDir   = SDIO_TRANSFER_DIR_TO_SDIO;
  config.TransferMode  = SDIO_TRANSFER_MODE_BLOCK;
  config.DPSM          = SDIO_DPSM_ENABLE;
  (void)SDIO_ConfigData(hmmc->Instance, &config);

  /* Set Block Size for Card */
  errorstate = SDMMC_CmdSendEXTCSD(hmmc->Instance, 0);
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    /* Clear all the static flags */
    __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
    hmmc->ErrorCode |= errorstate;
    hmmc->State = HAL_MMC_STATE_READY;
    return HAL_ERROR;
  }

  /* Poll on SDMMC flags */
  while(!__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_RXOVERR | SDIO_FLAG_DCRCFAIL | SDIO_FLAG_DTIMEOUT | SDIO_FLAG_DATAEND))
  {
    if(__HAL_MMC_GET_FLAG(hmmc, SDIO_FLAG_RXFIFOHF))
    {
      /* Read data from SDMMC Rx FIFO */
      for(count = 0U; count < 8U; count++)
      {
        tmp_data = SDIO_ReadFIFO(hmmc->Instance);
        /* eg : SEC_COUNT   : FieldIndex = 212 => i+count = 53 */
        /*      DEVICE_TYPE : FieldIndex = 196 => i+count = 49 */
        if ((i + count) == ((uint32_t)FieldIndex/4U))
        {
          *pFieldData = tmp_data;
        }
      }
      i += 8U;
    }

    if(((HAL_GetTick()-tickstart) >=  Timeout) || (Timeout == 0U))
    {
      /* Clear all the static flags */
      __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_FLAGS);
      hmmc->ErrorCode |= HAL_MMC_ERROR_TIMEOUT;
      hmmc->State= HAL_MMC_STATE_READY;
      return HAL_TIMEOUT;
    }
  }

  /* While card is not ready for data and trial number for sending CMD13 is not exceeded */
  errorstate = SDMMC_CmdSendStatus(hmmc->Instance, (uint32_t)(((uint32_t)hmmc->MmcCard.RelCardAdd) << 16));
  if(errorstate != HAL_MMC_ERROR_NONE)
  {
    hmmc->ErrorCode |= errorstate;
  }

  /* Clear all the static flags */
  __HAL_MMC_CLEAR_FLAG(hmmc, SDIO_STATIC_DATA_FLAGS);

  hmmc->State = HAL_MMC_STATE_READY;

  return HAL_OK;
}


/**
  * @brief  Wrap up reading in non-blocking mode.
  * @param  hmmc: pointer to a MMC_HandleTypeDef structure that contains
  *              the configuration information.
  * @retval None
  */
static void MMC_Read_IT(MMC_HandleTypeDef *hmmc)
{
  uint32_t count, data, dataremaining;
  uint8_t* tmp;

  tmp = hmmc->pRxBuffPtr;
  dataremaining = hmmc->RxXferSize;

  if (dataremaining > 0U)
  {
    /* Read data from SDIO Rx FIFO */
    for(count = 0U; count < 8U; count++)
    {
      data = SDIO_ReadFIFO(hmmc->Instance);
      *tmp = (uint8_t)(data & 0xFFU);
      tmp++;
      dataremaining--;
      *tmp = (uint8_t)((data >> 8U) & 0xFFU);
      tmp++;
      dataremaining--;
      *tmp = (uint8_t)((data >> 16U) & 0xFFU);
      tmp++;
      dataremaining--;
      *tmp = (uint8_t)((data >> 24U) & 0xFFU);
      tmp++;
      dataremaining--;
    }

    hmmc->pRxBuffPtr = tmp;
    hmmc->RxXferSize = dataremaining;
  }
}

/**
  * @brief  Wrap up writing in non-blocking mode.
  * @param  hmmc: pointer to a MMC_HandleTypeDef structure that contains
  *              the configuration information.
  * @retval None
  */
static void MMC_Write_IT(MMC_HandleTypeDef *hmmc)
{
  uint32_t count, data, dataremaining;
  uint8_t* tmp;

  tmp = hmmc->pTxBuffPtr;
  dataremaining = hmmc->TxXferSize;

  if (dataremaining > 0U)
  {
    /* Write data to SDIO Tx FIFO */
    for(count = 0U; count < 8U; count++)
    {
      data = (uint32_t)(*tmp);
      tmp++;
      dataremaining--;
      data |= ((uint32_t)(*tmp) << 8U);
      tmp++;
      dataremaining--;
      data |= ((uint32_t)(*tmp) << 16U);
      tmp++;
      dataremaining--;
      data |= ((uint32_t)(*tmp) << 24U);
      tmp++;
      dataremaining--;
      (void)SDIO_WriteFIFO(hmmc->Instance, &data);
    }

    hmmc->pTxBuffPtr = tmp;
    hmmc->TxXferSize = dataremaining;
  }
}

/**
  * @}
  */

#endif /* SDIO */

#endif /* HAL_MMC_MODULE_ENABLED */

/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
