/**
  ******************************************************************************
  * @file    stm32f4xx_hal_adc.c
  * @author  MCD Application Team
  * @brief   This file provides firmware functions to manage the following 
  *          functionalities of the Analog to Digital Converter (ADC) peripheral:
  *           + Initialization and de-initialization functions
  *           + IO operation functions
  *           + State and errors functions
  *         
  @verbatim
  ==============================================================================
                    ##### ADC Peripheral features #####
  ==============================================================================
  [..] 
  (#) 12-bit, 10-bit, 8-bit or 6-bit configurable resolution.
  (#) Interrupt generation at the end of conversion, end of injected conversion,  
      and in case of analog watchdog or overrun events
  (#) Single and continuous conversion modes.
  (#) Scan mode for automatic conversion of channel 0 to channel x.
  (#) Data alignment with in-built data coherency.
  (#) Channel-wise programmable sampling time.
  (#) External trigger option with configurable polarity for both regular and 
      injected conversion.
  (#) Dual/Triple mode (on devices with 2 ADCs or more).
  (#) Configurable DMA data storage in Dual/Triple ADC mode. 
  (#) Configurable delay between conversions in Dual/Triple interleaved mode.
  (#) ADC conversion type (refer to the datasheets).
  (#) ADC supply requirements: 2.4 V to 3.6 V at full speed and down to 1.8 V at 
      slower speed.
  (#) ADC input range: VREF(minus) = VIN = VREF(plus).
  (#) DMA request generation during regular channel conversion.


                     ##### How to use this driver #####
  ==============================================================================
  [..]
  (#)Initialize the ADC low level resources by implementing the HAL_ADC_MspInit():
       (##) Enable the ADC interface clock using __HAL_RCC_ADC_CLK_ENABLE()
       (##) ADC pins configuration
             (+++) Enable the clock for the ADC GPIOs using the following function:
                   __HAL_RCC_GPIOx_CLK_ENABLE()  
             (+++) Configure these ADC pins in analog mode using HAL_GPIO_Init() 
       (##) In case of using interrupts (e.g. HAL_ADC_Start_IT())
             (+++) Configure the ADC interrupt priority using HAL_NVIC_SetPriority()
             (+++) Enable the ADC IRQ handler using HAL_NVIC_EnableIRQ()
             (+++) In ADC IRQ handler, call HAL_ADC_IRQHandler()
       (##) In case of using DMA to control data transfer (e.g. HAL_ADC_Start_DMA())
             (+++) Enable the DMAx interface clock using __HAL_RCC_DMAx_CLK_ENABLE()
             (+++) Configure and enable two DMA streams stream for managing data
                 transfer from peripheral to memory (output stream)
             (+++) Associate the initialized DMA handle to the CRYP DMA handle
                 using  __HAL_LINKDMA()
             (+++) Configure the priority and enable the NVIC for the transfer complete
                 interrupt on the two DMA Streams. The output stream should have higher
                 priority than the input stream.
                       
    *** Configuration of ADC, groups regular/injected, channels parameters ***
  ==============================================================================
  [..]
  (#) Configure the ADC parameters (resolution, data alignment, ...)
      and regular group parameters (conversion trigger, sequencer, ...)
      using function HAL_ADC_Init().

  (#) Configure the channels for regular group parameters (channel number, 
      channel rank into sequencer, ..., into regular group)
      using function HAL_ADC_ConfigChannel().

  (#) Optionally, configure the injected group parameters (conversion trigger, 
      sequencer, ..., of injected group)
      and the channels for injected group parameters (channel number, 
      channel rank into sequencer, ..., into injected group)
      using function HAL_ADCEx_InjectedConfigChannel().

  (#) Optionally, configure the analog watchdog parameters (channels
      monitored, thresholds, ...) using function HAL_ADC_AnalogWDGConfig().

  (#) Optionally, for devices with several ADC instances: configure the 
      multimode parameters using function HAL_ADCEx_MultiModeConfigChannel().

                       *** Execution of ADC conversions ***
  ==============================================================================
  [..]  
  (#) ADC driver can be used among three modes: polling, interruption,
      transfer by DMA.    

     *** Polling mode IO operation ***
     =================================
     [..]    
       (+) Start the ADC peripheral using HAL_ADC_Start() 
       (+) Wait for end of conversion using HAL_ADC_PollForConversion(), at this stage
           user can specify the value of timeout according to his end application      
       (+) To read the ADC converted values, use the HAL_ADC_GetValue() function.
       (+) Stop the ADC peripheral using HAL_ADC_Stop()
       
     *** Interrupt mode IO operation ***    
     ===================================
     [..]    
       (+) Start the ADC peripheral using HAL_ADC_Start_IT() 
       (+) Use HAL_ADC_IRQHandler() called under ADC_IRQHandler() Interrupt subroutine
       (+) At ADC end of conversion HAL_ADC_ConvCpltCallback() function is executed and user can 
           add his own code by customization of function pointer HAL_ADC_ConvCpltCallback 
       (+) In case of ADC Error, HAL_ADC_ErrorCallback() function is executed and user can 
           add his own code by customization of function pointer HAL_ADC_ErrorCallback
       (+) Stop the ADC peripheral using HAL_ADC_Stop_IT()     

     *** DMA mode IO operation ***    
     ==============================
     [..]    
       (+) Start the ADC peripheral using HAL_ADC_Start_DMA(), at this stage the user specify the length 
           of data to be transferred at each end of conversion 
       (+) At The end of data transfer by HAL_ADC_ConvCpltCallback() function is executed and user can 
           add his own code by customization of function pointer HAL_ADC_ConvCpltCallback 
       (+) In case of transfer Error, HAL_ADC_ErrorCallback() function is executed and user can 
           add his own code by customization of function pointer HAL_ADC_ErrorCallback
       (+) Stop the ADC peripheral using HAL_ADC_Stop_DMA()
                    
     *** ADC HAL driver macros list ***
     ============================================= 
     [..]
       Below the list of most used macros in ADC HAL driver.
       
      (+) __HAL_ADC_ENABLE : Enable the ADC peripheral
      (+) __HAL_ADC_DISABLE : Disable the ADC peripheral
      (+) __HAL_ADC_ENABLE_IT: Enable the ADC end of conversion interrupt
      (+) __HAL_ADC_DISABLE_IT: Disable the ADC end of conversion interrupt
      (+) __HAL_ADC_GET_IT_SOURCE: Check if the specified ADC interrupt source is enabled or disabled
      (+) __HAL_ADC_CLEAR_FLAG: Clear the ADC's pending flags
      (+) __HAL_ADC_GET_FLAG: Get the selected ADC's flag status
      (+) ADC_GET_RESOLUTION: Return resolution bits in CR1 register 
      
     [..] 
       (@) You can refer to the ADC HAL driver header file for more useful macros 

                      *** Deinitialization of ADC ***
  ==============================================================================
  [..]
  (#) Disable the ADC interface
     (++) ADC clock can be hard reset and disabled at RCC top level.
     (++) Hard reset of ADC peripherals
          using macro __HAL_RCC_ADC_FORCE_RESET(), __HAL_RCC_ADC_RELEASE_RESET().
     (++) ADC clock disable using the equivalent macro/functions as configuration step.
               (+++) Example:
                   Into HAL_ADC_MspDeInit() (recommended code location) or with
                   other device clock parameters configuration:
               (+++) HAL_RCC_GetOscConfig(&RCC_OscInitStructure);
               (+++) RCC_OscInitStructure.OscillatorType = RCC_OSCILLATORTYPE_HSI;
               (+++) RCC_OscInitStructure.HSIState = RCC_HSI_OFF; (if not used for system clock)
               (+++) HAL_RCC_OscConfig(&RCC_OscInitStructure);

  (#) ADC pins configuration
     (++) Disable the clock for the ADC GPIOs using macro __HAL_RCC_GPIOx_CLK_DISABLE()

  (#) Optionally, in case of usage of ADC with interruptions:
     (++) Disable the NVIC for ADC using function HAL_NVIC_DisableIRQ(ADCx_IRQn)

  (#) Optionally, in case of usage of DMA:
        (++) Deinitialize the DMA using function HAL_DMA_DeInit().
        (++) Disable the NVIC for DMA using function HAL_NVIC_DisableIRQ(DMAx_Channelx_IRQn)   
                      *** Callback registration ***
  ==============================================================================
    [..]

     The compilation flag USE_HAL_ADC_REGISTER_CALLBACKS, when set to 1,
     allows the user to configure dynamically the driver callbacks.
     Use Functions HAL_ADC_RegisterCallback()
     to register an interrupt callback.
    [..]

     Function HAL_ADC_RegisterCallback() allows to register following callbacks:
       (+) ConvCpltCallback               : ADC conversion complete callback
       (+) ConvHalfCpltCallback           : ADC conversion DMA half-transfer callback
       (+) LevelOutOfWindowCallback       : ADC analog watchdog 1 callback
       (+) ErrorCallback                  : ADC error callback
       (+) InjectedConvCpltCallback       : ADC group injected conversion complete callback
       (+) InjectedQueueOverflowCallback  : ADC group injected context queue overflow callback
       (+) LevelOutOfWindow2Callback      : ADC analog watchdog 2 callback
       (+) LevelOutOfWindow3Callback      : ADC analog watchdog 3 callback
       (+) EndOfSamplingCallback          : ADC end of sampling callback
       (+) MspInitCallback                : ADC Msp Init callback
       (+) MspDeInitCallback              : ADC Msp DeInit callback
     This function takes as parameters the HAL peripheral handle, the Callback ID
     and a pointer to the user callback function.
    [..]

     Use function HAL_ADC_UnRegisterCallback to reset a callback to the default
     weak function.
    [..]

     HAL_ADC_UnRegisterCallback takes as parameters the HAL peripheral handle,
     and the Callback ID.
     This function allows to reset following callbacks:
       (+) ConvCpltCallback               : ADC conversion complete callback
       (+) ConvHalfCpltCallback           : ADC conversion DMA half-transfer callback
       (+) LevelOutOfWindowCallback       : ADC analog watchdog 1 callback
       (+) ErrorCallback                  : ADC error callback
       (+) InjectedConvCpltCallback       : ADC group injected conversion complete callback
       (+) InjectedQueueOverflowCallback  : ADC group injected context queue overflow callback
       (+) LevelOutOfWindow2Callback      : ADC analog watchdog 2 callback
       (+) LevelOutOfWindow3Callback      : ADC analog watchdog 3 callback
       (+) EndOfSamplingCallback          : ADC end of sampling callback
       (+) MspInitCallback                : ADC Msp Init callback
       (+) MspDeInitCallback              : ADC Msp DeInit callback
     [..]

     By default, after the HAL_ADC_Init() and when the state is HAL_ADC_STATE_RESET
     all callbacks are set to the corresponding weak functions:
     examples HAL_ADC_ConvCpltCallback(), HAL_ADC_ErrorCallback().
     Exception done for MspInit and MspDeInit functions that are
     reset to the legacy weak functions in the HAL_ADC_Init()/ HAL_ADC_DeInit() only when
     these callbacks are null (not registered beforehand).
    [..]

     If MspInit or MspDeInit are not null, the HAL_ADC_Init()/ HAL_ADC_DeInit()
     keep and use the user MspInit/MspDeInit callbacks (registered beforehand) whatever the state.
     [..]

     Callbacks can be registered/unregistered in HAL_ADC_STATE_READY state only.
     Exception done MspInit/MspDeInit functions that can be registered/unregistered
     in HAL_ADC_STATE_READY or HAL_ADC_STATE_RESET state,
     thus registered (user) MspInit/DeInit callbacks can be used during the Init/DeInit.
    [..]

     Then, the user first registers the MspInit/MspDeInit user callbacks
     using HAL_ADC_RegisterCallback() before calling HAL_ADC_DeInit()
     or HAL_ADC_Init() function.
     [..]

     When the compilation flag USE_HAL_ADC_REGISTER_CALLBACKS is set to 0 or
     not defined, the callback registration feature is not available and all callbacks
     are set to the corresponding weak functions.

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

/** @defgroup ADC ADC
  * @brief ADC driver modules
  * @{
  */ 

#ifdef HAL_ADC_MODULE_ENABLED
    
/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/** @addtogroup ADC_Private_Functions
  * @{
  */
/* Private function prototypes -----------------------------------------------*/
static void ADC_Init(ADC_HandleTypeDef* hadc);
static void ADC_DMAConvCplt(DMA_HandleTypeDef *hdma);
static void ADC_DMAError(DMA_HandleTypeDef *hdma);
static void ADC_DMAHalfConvCplt(DMA_HandleTypeDef *hdma);
/**
  * @}
  */
/* Exported functions --------------------------------------------------------*/
/** @defgroup ADC_Exported_Functions ADC Exported Functions
  * @{
  */

/** @defgroup ADC_Exported_Functions_Group1 Initialization and de-initialization functions 
 *  @brief    Initialization and Configuration functions 
 *
@verbatim    
 ===============================================================================
              ##### Initialization and de-initialization functions #####
 ===============================================================================
    [..]  This section provides functions allowing to:
      (+) Initialize and configure the ADC. 
      (+) De-initialize the ADC. 
         
@endverbatim
  * @{
  */

/**
  * @brief  Initializes the ADCx peripheral according to the specified parameters 
  *         in the ADC_InitStruct and initializes the ADC MSP.
  *           
  * @note   This function is used to configure the global features of the ADC ( 
  *         ClockPrescaler, Resolution, Data Alignment and number of conversion), however,
  *         the rest of the configuration parameters are specific to the regular
  *         channels group (scan mode activation, continuous mode activation,
  *         External trigger source and edge, DMA continuous request after the  
  *         last transfer and End of conversion selection).
  *             
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.  
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_Init(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;
  
  /* Check ADC handle */
  if(hadc == NULL)
  {
    return HAL_ERROR;
  }
  
  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  assert_param(IS_ADC_CLOCKPRESCALER(hadc->Init.ClockPrescaler));
  assert_param(IS_ADC_RESOLUTION(hadc->Init.Resolution));
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ScanConvMode));
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ContinuousConvMode));
  assert_param(IS_ADC_EXT_TRIG(hadc->Init.ExternalTrigConv));
  assert_param(IS_ADC_DATA_ALIGN(hadc->Init.DataAlign));
  assert_param(IS_ADC_REGULAR_LENGTH(hadc->Init.NbrOfConversion));
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.DMAContinuousRequests));
  assert_param(IS_ADC_EOCSelection(hadc->Init.EOCSelection));
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.DiscontinuousConvMode));
  
  if(hadc->Init.ExternalTrigConv != ADC_SOFTWARE_START)
  {
    assert_param(IS_ADC_EXT_TRIG_EDGE(hadc->Init.ExternalTrigConvEdge));
  }
  
  if(hadc->State == HAL_ADC_STATE_RESET)
  {
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
    /* Init the ADC Callback settings */
    hadc->ConvCpltCallback              = HAL_ADC_ConvCpltCallback;                 /* Legacy weak callback */
    hadc->ConvHalfCpltCallback          = HAL_ADC_ConvHalfCpltCallback;             /* Legacy weak callback */
    hadc->LevelOutOfWindowCallback      = HAL_ADC_LevelOutOfWindowCallback;         /* Legacy weak callback */
    hadc->ErrorCallback                 = HAL_ADC_ErrorCallback;                    /* Legacy weak callback */
    hadc->InjectedConvCpltCallback      = HAL_ADCEx_InjectedConvCpltCallback;       /* Legacy weak callback */
    if (hadc->MspInitCallback == NULL)
    {
      hadc->MspInitCallback = HAL_ADC_MspInit; /* Legacy weak MspInit  */
    }

    /* Init the low level hardware */
    hadc->MspInitCallback(hadc);
#else
    /* Init the low level hardware */
    HAL_ADC_MspInit(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */

    /* Initialize ADC error code */
    ADC_CLEAR_ERRORCODE(hadc);
    
    /* Allocate lock resource and initialize it */
    hadc->Lock = HAL_UNLOCKED;
  }
  
  /* Configuration of ADC parameters if previous preliminary actions are      */ 
  /* correctly completed.                                                     */
  if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL))
  {
    /* Set ADC state */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                      HAL_ADC_STATE_BUSY_INTERNAL);
    
    /* Set ADC parameters */
    ADC_Init(hadc);
    
    /* Set ADC error code to none */
    ADC_CLEAR_ERRORCODE(hadc);
    
    /* Set the ADC state */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_BUSY_INTERNAL,
                      HAL_ADC_STATE_READY);
  }
  else
  {
    tmp_hal_status = HAL_ERROR;
  }
  
  /* Release Lock */
  __HAL_UNLOCK(hadc);

  /* Return function status */
  return tmp_hal_status;
}

/**
  * @brief  Deinitializes the ADCx peripheral registers to their default reset values. 
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.  
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_DeInit(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;
  
  /* Check ADC handle */
  if(hadc == NULL)
  {
    return HAL_ERROR;
  }
  
  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  
  /* Set ADC state */
  SET_BIT(hadc->State, HAL_ADC_STATE_BUSY_INTERNAL);
  
  /* Stop potential conversion on going, on regular and injected groups */
  /* Disable ADC peripheral */
  __HAL_ADC_DISABLE(hadc);
  
  /* Configuration of ADC parameters if previous preliminary actions are      */ 
  /* correctly completed.                                                     */
  if(HAL_IS_BIT_CLR(hadc->Instance->CR2, ADC_CR2_ADON))
  {
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
  if (hadc->MspDeInitCallback == NULL)
  {
    hadc->MspDeInitCallback = HAL_ADC_MspDeInit; /* Legacy weak MspDeInit  */
  }

  /* DeInit the low level hardware: RCC clock, NVIC */
  hadc->MspDeInitCallback(hadc);
#else
  /* DeInit the low level hardware: RCC clock, NVIC */
  HAL_ADC_MspDeInit(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
    
    /* Set ADC error code to none */
    ADC_CLEAR_ERRORCODE(hadc);
    
    /* Set ADC state */
    hadc->State = HAL_ADC_STATE_RESET;
  }
  
  /* Process unlocked */
  __HAL_UNLOCK(hadc);
  
  /* Return function status */
  return tmp_hal_status;
}

#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User ADC Callback
  *         To be used instead of the weak predefined callback
  * @param  hadc Pointer to a ADC_HandleTypeDef structure that contains
  *                the configuration information for the specified ADC.
  * @param  CallbackID ID of the callback to be registered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_ADC_CONVERSION_COMPLETE_CB_ID      ADC conversion complete callback ID
  *          @arg @ref HAL_ADC_CONVERSION_HALF_CB_ID          ADC conversion DMA half-transfer callback ID
  *          @arg @ref HAL_ADC_LEVEL_OUT_OF_WINDOW_1_CB_ID    ADC analog watchdog 1 callback ID
  *          @arg @ref HAL_ADC_ERROR_CB_ID                    ADC error callback ID
  *          @arg @ref HAL_ADC_INJ_CONVERSION_COMPLETE_CB_ID  ADC group injected conversion complete callback ID
  *          @arg @ref HAL_ADC_MSPINIT_CB_ID                  ADC Msp Init callback ID
  *          @arg @ref HAL_ADC_MSPDEINIT_CB_ID                ADC Msp DeInit callback ID
  * @param  pCallback pointer to the Callback function
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_RegisterCallback(ADC_HandleTypeDef *hadc, HAL_ADC_CallbackIDTypeDef CallbackID, pADC_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if (pCallback == NULL)
  {
    /* Update the error code */
    hadc->ErrorCode |= HAL_ADC_ERROR_INVALID_CALLBACK;

    return HAL_ERROR;
  }

  if ((hadc->State & HAL_ADC_STATE_READY) != 0UL)
  {
    switch (CallbackID)
    {
      case HAL_ADC_CONVERSION_COMPLETE_CB_ID :
        hadc->ConvCpltCallback = pCallback;
        break;

      case HAL_ADC_CONVERSION_HALF_CB_ID :
        hadc->ConvHalfCpltCallback = pCallback;
        break;

      case HAL_ADC_LEVEL_OUT_OF_WINDOW_1_CB_ID :
        hadc->LevelOutOfWindowCallback = pCallback;
        break;

      case HAL_ADC_ERROR_CB_ID :
        hadc->ErrorCallback = pCallback;
        break;

      case HAL_ADC_INJ_CONVERSION_COMPLETE_CB_ID :
        hadc->InjectedConvCpltCallback = pCallback;
        break;

      case HAL_ADC_MSPINIT_CB_ID :
        hadc->MspInitCallback = pCallback;
        break;

      case HAL_ADC_MSPDEINIT_CB_ID :
        hadc->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hadc->ErrorCode |= HAL_ADC_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status = HAL_ERROR;
        break;
    }
  }
  else if (HAL_ADC_STATE_RESET == hadc->State)
  {
    switch (CallbackID)
    {
      case HAL_ADC_MSPINIT_CB_ID :
        hadc->MspInitCallback = pCallback;
        break;

      case HAL_ADC_MSPDEINIT_CB_ID :
        hadc->MspDeInitCallback = pCallback;
        break;

      default :
        /* Update the error code */
        hadc->ErrorCode |= HAL_ADC_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status = HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hadc->ErrorCode |= HAL_ADC_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  return status;
}

/**
  * @brief  Unregister a ADC Callback
  *         ADC callback is redirected to the weak predefined callback
  * @param  hadc Pointer to a ADC_HandleTypeDef structure that contains
  *                the configuration information for the specified ADC.
  * @param  CallbackID ID of the callback to be unregistered
  *         This parameter can be one of the following values:
  *          @arg @ref HAL_ADC_CONVERSION_COMPLETE_CB_ID      ADC conversion complete callback ID
  *          @arg @ref HAL_ADC_CONVERSION_HALF_CB_ID          ADC conversion DMA half-transfer callback ID
  *          @arg @ref HAL_ADC_LEVEL_OUT_OF_WINDOW_1_CB_ID    ADC analog watchdog 1 callback ID
  *          @arg @ref HAL_ADC_ERROR_CB_ID                    ADC error callback ID
  *          @arg @ref HAL_ADC_INJ_CONVERSION_COMPLETE_CB_ID  ADC group injected conversion complete callback ID
  *          @arg @ref HAL_ADC_MSPINIT_CB_ID                  ADC Msp Init callback ID
  *          @arg @ref HAL_ADC_MSPDEINIT_CB_ID                ADC Msp DeInit callback ID
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_UnRegisterCallback(ADC_HandleTypeDef *hadc, HAL_ADC_CallbackIDTypeDef CallbackID)
{
  HAL_StatusTypeDef status = HAL_OK;

  if ((hadc->State & HAL_ADC_STATE_READY) != 0UL)
  {
    switch (CallbackID)
    {
      case HAL_ADC_CONVERSION_COMPLETE_CB_ID :
        hadc->ConvCpltCallback = HAL_ADC_ConvCpltCallback;
        break;

      case HAL_ADC_CONVERSION_HALF_CB_ID :
        hadc->ConvHalfCpltCallback = HAL_ADC_ConvHalfCpltCallback;
        break;

      case HAL_ADC_LEVEL_OUT_OF_WINDOW_1_CB_ID :
        hadc->LevelOutOfWindowCallback = HAL_ADC_LevelOutOfWindowCallback;
        break;

      case HAL_ADC_ERROR_CB_ID :
        hadc->ErrorCallback = HAL_ADC_ErrorCallback;
        break;

      case HAL_ADC_INJ_CONVERSION_COMPLETE_CB_ID :
        hadc->InjectedConvCpltCallback = HAL_ADCEx_InjectedConvCpltCallback;
        break;

      case HAL_ADC_MSPINIT_CB_ID :
        hadc->MspInitCallback = HAL_ADC_MspInit; /* Legacy weak MspInit              */
        break;

      case HAL_ADC_MSPDEINIT_CB_ID :
        hadc->MspDeInitCallback = HAL_ADC_MspDeInit; /* Legacy weak MspDeInit            */
        break;

      default :
        /* Update the error code */
        hadc->ErrorCode |= HAL_ADC_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else if (HAL_ADC_STATE_RESET == hadc->State)
  {
    switch (CallbackID)
    {
      case HAL_ADC_MSPINIT_CB_ID :
        hadc->MspInitCallback = HAL_ADC_MspInit;                   /* Legacy weak MspInit              */
        break;

      case HAL_ADC_MSPDEINIT_CB_ID :
        hadc->MspDeInitCallback = HAL_ADC_MspDeInit;               /* Legacy weak MspDeInit            */
        break;

      default :
        /* Update the error code */
        hadc->ErrorCode |= HAL_ADC_ERROR_INVALID_CALLBACK;

        /* Return error status */
        status =  HAL_ERROR;
        break;
    }
  }
  else
  {
    /* Update the error code */
    hadc->ErrorCode |= HAL_ADC_ERROR_INVALID_CALLBACK;

    /* Return error status */
    status =  HAL_ERROR;
  }

  return status;
}

#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */

/**
  * @brief  Initializes the ADC MSP.
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.  
  * @retval None
  */
__weak void HAL_ADC_MspInit(ADC_HandleTypeDef* hadc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hadc);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_ADC_MspInit could be implemented in the user file
   */ 
}

/**
  * @brief  DeInitializes the ADC MSP.
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.  
  * @retval None
  */
__weak void HAL_ADC_MspDeInit(ADC_HandleTypeDef* hadc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hadc);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_ADC_MspDeInit could be implemented in the user file
   */ 
}

/**
  * @}
  */

/** @defgroup ADC_Exported_Functions_Group2 IO operation functions
 *  @brief    IO operation functions 
 *
@verbatim   
 ===============================================================================
             ##### IO operation functions #####
 ===============================================================================  
    [..]  This section provides functions allowing to:
      (+) Start conversion of regular channel.
      (+) Stop conversion of regular channel.
      (+) Start conversion of regular channel and enable interrupt.
      (+) Stop conversion of regular channel and disable interrupt.
      (+) Start conversion of regular channel and enable DMA transfer.
      (+) Stop conversion of regular channel and disable DMA transfer.
      (+) Handle ADC interrupt request. 
               
@endverbatim
  * @{
  */

/**
  * @brief  Enables ADC and starts conversion of the regular channels.
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_Start(ADC_HandleTypeDef* hadc)
{
  __IO uint32_t counter = 0U;
  ADC_Common_TypeDef *tmpADC_Common;
  
  /* Check the parameters */
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ContinuousConvMode));
  assert_param(IS_ADC_EXT_TRIG_EDGE(hadc->Init.ExternalTrigConvEdge)); 
  
  /* Process locked */
  __HAL_LOCK(hadc);
  
  /* Enable the ADC peripheral */
  /* Check if ADC peripheral is disabled in order to enable it and wait during 
  Tstab time the ADC's stabilization */
  if((hadc->Instance->CR2 & ADC_CR2_ADON) != ADC_CR2_ADON)
  {  
    /* Enable the Peripheral */
    __HAL_ADC_ENABLE(hadc);
    
    /* Delay for ADC stabilization time */
    /* Compute number of CPU cycles to wait for */
    counter = (ADC_STAB_DELAY_US * (SystemCoreClock / 1000000U));
    while(counter != 0U)
    {
      counter--;
    }
  }
  
  /* Start conversion if ADC is effectively enabled */
  if(HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_ADON))
  {
    /* Set ADC state                                                          */
    /* - Clear state bitfield related to regular group conversion results     */
    /* - Set state bitfield related to regular group operation                */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_READY | HAL_ADC_STATE_REG_EOC | HAL_ADC_STATE_REG_OVR,
                      HAL_ADC_STATE_REG_BUSY);
    
    /* If conversions on group regular are also triggering group injected,    */
    /* update ADC state.                                                      */
    if (READ_BIT(hadc->Instance->CR1, ADC_CR1_JAUTO) != RESET)
    {
      ADC_STATE_CLR_SET(hadc->State, HAL_ADC_STATE_INJ_EOC, HAL_ADC_STATE_INJ_BUSY);  
    }
    
    /* State machine update: Check if an injected conversion is ongoing */
    if (HAL_IS_BIT_SET(hadc->State, HAL_ADC_STATE_INJ_BUSY))
    {
      /* Reset ADC error code fields related to conversions on group regular */
      CLEAR_BIT(hadc->ErrorCode, (HAL_ADC_ERROR_OVR | HAL_ADC_ERROR_DMA));         
    }
    else
    {
      /* Reset ADC all error code fields */
      ADC_CLEAR_ERRORCODE(hadc);
    } 

    /* Process unlocked */
    /* Unlock before starting ADC conversions: in case of potential           */
    /* interruption, to let the process to ADC IRQ Handler.                   */
    __HAL_UNLOCK(hadc);

    /* Pointer to the common control register to which is belonging hadc    */
    /* (Depending on STM32F4 product, there may be up to 3 ADCs and 1 common */
    /* control register)                                                    */
    tmpADC_Common = ADC_COMMON_REGISTER(hadc);

    /* Clear regular group conversion flag and overrun flag */
    /* (To ensure of no unknown state from potential previous ADC operations) */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_EOC | ADC_FLAG_OVR);
    
    /* Check if Multimode enabled */
    if(HAL_IS_BIT_CLR(tmpADC_Common->CCR, ADC_CCR_MULTI))
    {
#if defined(ADC2) && defined(ADC3)
      if((hadc->Instance == ADC1) || ((hadc->Instance == ADC2) && ((ADC->CCR & ADC_CCR_MULTI_Msk) < ADC_CCR_MULTI_0)) \
                                  || ((hadc->Instance == ADC3) && ((ADC->CCR & ADC_CCR_MULTI_Msk) < ADC_CCR_MULTI_4)))
      {
#endif /* ADC2 || ADC3 */
        /* if no external trigger present enable software conversion of regular channels */
        if((hadc->Instance->CR2 & ADC_CR2_EXTEN) == RESET) 
        {
          /* Enable the selected ADC software conversion for regular group */
          hadc->Instance->CR2 |= (uint32_t)ADC_CR2_SWSTART;
        }
#if defined(ADC2) && defined(ADC3)
      }
#endif /* ADC2 || ADC3 */
    }
    else
    {
      /* if instance of handle correspond to ADC1 and  no external trigger present enable software conversion of regular channels */
      if((hadc->Instance == ADC1) && ((hadc->Instance->CR2 & ADC_CR2_EXTEN) == RESET))
      {
        /* Enable the selected ADC software conversion for regular group */
          hadc->Instance->CR2 |= (uint32_t)ADC_CR2_SWSTART;
      }
    }
  }
  else
  {
    /* Update ADC state machine to error */
    SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL);

    /* Set ADC error code to ADC IP internal error */
    SET_BIT(hadc->ErrorCode, HAL_ADC_ERROR_INTERNAL);
  }
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Disables ADC and stop conversion of regular channels.
  * 
  * @note   Caution: This function will stop also injected channels.  
  *
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  *
  * @retval HAL status.
  */
HAL_StatusTypeDef HAL_ADC_Stop(ADC_HandleTypeDef* hadc)
{
  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  
  /* Process locked */
  __HAL_LOCK(hadc);
  
  /* Stop potential conversion on going, on regular and injected groups */
  /* Disable ADC peripheral */
  __HAL_ADC_DISABLE(hadc);
  
  /* Check if ADC is effectively disabled */
  if(HAL_IS_BIT_CLR(hadc->Instance->CR2, ADC_CR2_ADON))
  {
    /* Set ADC state */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                      HAL_ADC_STATE_READY);
  }
  
  /* Process unlocked */
  __HAL_UNLOCK(hadc);
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Poll for regular conversion complete
  * @note   ADC conversion flags EOS (end of sequence) and EOC (end of
  *         conversion) are cleared by this function.
  * @note   This function cannot be used in a particular setup: ADC configured 
  *         in DMA mode and polling for end of each conversion (ADC init
  *         parameter "EOCSelection" set to ADC_EOC_SINGLE_CONV).
  *         In this case, DMA resets the flag EOC and polling cannot be
  *         performed on each conversion. Nevertheless, polling can still 
  *         be performed on the complete sequence.
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @param  Timeout Timeout value in millisecond.  
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_PollForConversion(ADC_HandleTypeDef* hadc, uint32_t Timeout)
{
  uint32_t tickstart = 0U;
 
  /* Verification that ADC configuration is compliant with polling for      */
  /* each conversion:                                                       */
  /* Particular case is ADC configured in DMA mode and ADC sequencer with   */
  /* several ranks and polling for end of each conversion.                  */
  /* For code simplicity sake, this particular case is generalized to       */
  /* ADC configured in DMA mode and polling for end of each conversion.     */
  if (HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_EOCS) &&
      HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_DMA)    )
  {
    /* Update ADC state machine to error */
    SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_CONFIG);
    
    /* Process unlocked */
    __HAL_UNLOCK(hadc);
    
    return HAL_ERROR;
  }

  /* Get tick */ 
  tickstart = HAL_GetTick();

  /* Check End of conversion flag */
  while(!(__HAL_ADC_GET_FLAG(hadc, ADC_FLAG_EOC)))
  {
    /* Check if timeout is disabled (set to infinite wait) */
    if(Timeout != HAL_MAX_DELAY)
    {
      if((Timeout == 0U) || ((HAL_GetTick() - tickstart ) > Timeout))
      {
        /* New check to avoid false timeout detection in case of preemption */
        if(!(__HAL_ADC_GET_FLAG(hadc, ADC_FLAG_EOC)))
        {
          /* Update ADC state machine to timeout */
          SET_BIT(hadc->State, HAL_ADC_STATE_TIMEOUT);
          
          /* Process unlocked */
          __HAL_UNLOCK(hadc);
          
          return HAL_TIMEOUT;
        }
      }
    }
  }
  
  /* Clear regular group conversion flag */
  __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_STRT | ADC_FLAG_EOC);
  
  /* Update ADC state machine */
  SET_BIT(hadc->State, HAL_ADC_STATE_REG_EOC);
  
  /* Determine whether any further conversion upcoming on group regular       */
  /* by external trigger, continuous mode or scan sequence on going.          */
  /* Note: On STM32F4, there is no independent flag of end of sequence.       */
  /*       The test of scan sequence on going is done either with scan        */
  /*       sequence disabled or with end of conversion flag set to            */
  /*       of end of sequence.                                                */
  if(ADC_IS_SOFTWARE_START_REGULAR(hadc)                   &&
     (hadc->Init.ContinuousConvMode == DISABLE)            &&
     (HAL_IS_BIT_CLR(hadc->Instance->SQR1, ADC_SQR1_L) ||
      HAL_IS_BIT_CLR(hadc->Instance->CR2, ADC_CR2_EOCS)  )   )
  {
    /* Set ADC state */
    CLEAR_BIT(hadc->State, HAL_ADC_STATE_REG_BUSY);   
    
    if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_INJ_BUSY))
    { 
      SET_BIT(hadc->State, HAL_ADC_STATE_READY);
    }
  }
  
  /* Return ADC state */
  return HAL_OK;
}

/**
  * @brief  Poll for conversion event
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @param  EventType the ADC event type.
  *          This parameter can be one of the following values:
  *            @arg ADC_AWD_EVENT: ADC Analog watch Dog event.
  *            @arg ADC_OVR_EVENT: ADC Overrun event.
  * @param  Timeout Timeout value in millisecond.   
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_PollForEvent(ADC_HandleTypeDef* hadc, uint32_t EventType, uint32_t Timeout)
{
  uint32_t tickstart = 0U;
  
  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  assert_param(IS_ADC_EVENT_TYPE(EventType));

  /* Get tick */
  tickstart = HAL_GetTick();

  /* Check selected event flag */
  while(!(__HAL_ADC_GET_FLAG(hadc,EventType)))
  {
    /* Check for the Timeout */
    if(Timeout != HAL_MAX_DELAY)
    {
      if((Timeout == 0U) || ((HAL_GetTick() - tickstart ) > Timeout))
      {
        /* New check to avoid false timeout detection in case of preemption */
        if(!(__HAL_ADC_GET_FLAG(hadc,EventType)))
        {
          /* Update ADC state machine to timeout */
          SET_BIT(hadc->State, HAL_ADC_STATE_TIMEOUT);
          
          /* Process unlocked */
          __HAL_UNLOCK(hadc);
          
          return HAL_TIMEOUT;
        }
      }
    }
  }
  
  /* Analog watchdog (level out of window) event */
  if(EventType == ADC_AWD_EVENT)
  {
    /* Set ADC state */
    SET_BIT(hadc->State, HAL_ADC_STATE_AWD1);
      
    /* Clear ADC analog watchdog flag */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_AWD);
  }
  /* Overrun event */
  else
  {
    /* Set ADC state */
    SET_BIT(hadc->State, HAL_ADC_STATE_REG_OVR);
    /* Set ADC error code to overrun */
    SET_BIT(hadc->ErrorCode, HAL_ADC_ERROR_OVR);
    
    /* Clear ADC overrun flag */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_OVR);
  }
  
  /* Return ADC state */
  return HAL_OK;
}


/**
  * @brief  Enables the interrupt and starts ADC conversion of regular channels.
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval HAL status.
  */
HAL_StatusTypeDef HAL_ADC_Start_IT(ADC_HandleTypeDef* hadc)
{
  __IO uint32_t counter = 0U;
  ADC_Common_TypeDef *tmpADC_Common;
  
  /* Check the parameters */
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ContinuousConvMode));
  assert_param(IS_ADC_EXT_TRIG_EDGE(hadc->Init.ExternalTrigConvEdge)); 
  
  /* Process locked */
  __HAL_LOCK(hadc);
  
  /* Enable the ADC peripheral */
  /* Check if ADC peripheral is disabled in order to enable it and wait during 
  Tstab time the ADC's stabilization */
  if((hadc->Instance->CR2 & ADC_CR2_ADON) != ADC_CR2_ADON)
  {  
    /* Enable the Peripheral */
    __HAL_ADC_ENABLE(hadc);
    
    /* Delay for ADC stabilization time */
    /* Compute number of CPU cycles to wait for */
    counter = (ADC_STAB_DELAY_US * (SystemCoreClock / 1000000U));
    while(counter != 0U)
    {
      counter--;
    }
  }
  
  /* Start conversion if ADC is effectively enabled */
  if(HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_ADON))
  {
    /* Set ADC state                                                          */
    /* - Clear state bitfield related to regular group conversion results     */
    /* - Set state bitfield related to regular group operation                */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_READY | HAL_ADC_STATE_REG_EOC | HAL_ADC_STATE_REG_OVR,
                      HAL_ADC_STATE_REG_BUSY);
    
    /* If conversions on group regular are also triggering group injected,    */
    /* update ADC state.                                                      */
    if (READ_BIT(hadc->Instance->CR1, ADC_CR1_JAUTO) != RESET)
    {
      ADC_STATE_CLR_SET(hadc->State, HAL_ADC_STATE_INJ_EOC, HAL_ADC_STATE_INJ_BUSY);  
    }
    
    /* State machine update: Check if an injected conversion is ongoing */
    if (HAL_IS_BIT_SET(hadc->State, HAL_ADC_STATE_INJ_BUSY))
    {
      /* Reset ADC error code fields related to conversions on group regular */
      CLEAR_BIT(hadc->ErrorCode, (HAL_ADC_ERROR_OVR | HAL_ADC_ERROR_DMA));         
    }
    else
    {
      /* Reset ADC all error code fields */
      ADC_CLEAR_ERRORCODE(hadc);
    }

    /* Process unlocked */
    /* Unlock before starting ADC conversions: in case of potential           */
    /* interruption, to let the process to ADC IRQ Handler.                   */
    __HAL_UNLOCK(hadc);

    /* Pointer to the common control register to which is belonging hadc    */
    /* (Depending on STM32F4 product, there may be up to 3 ADCs and 1 common */
    /* control register)                                                    */
    tmpADC_Common = ADC_COMMON_REGISTER(hadc);

    /* Clear regular group conversion flag and overrun flag */
    /* (To ensure of no unknown state from potential previous ADC operations) */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_EOC | ADC_FLAG_OVR);
    
    /* Enable end of conversion interrupt for regular group */
    __HAL_ADC_ENABLE_IT(hadc, (ADC_IT_EOC | ADC_IT_OVR));
    
    /* Check if Multimode enabled */
    if(HAL_IS_BIT_CLR(tmpADC_Common->CCR, ADC_CCR_MULTI))
    {
#if defined(ADC2) && defined(ADC3)
      if((hadc->Instance == ADC1) || ((hadc->Instance == ADC2) && ((ADC->CCR & ADC_CCR_MULTI_Msk) < ADC_CCR_MULTI_0)) \
                                  || ((hadc->Instance == ADC3) && ((ADC->CCR & ADC_CCR_MULTI_Msk) < ADC_CCR_MULTI_4)))
      {
#endif /* ADC2 || ADC3 */
        /* if no external trigger present enable software conversion of regular channels */
        if((hadc->Instance->CR2 & ADC_CR2_EXTEN) == RESET) 
        {
          /* Enable the selected ADC software conversion for regular group */
          hadc->Instance->CR2 |= (uint32_t)ADC_CR2_SWSTART;
        }
#if defined(ADC2) && defined(ADC3)
      }
#endif /* ADC2 || ADC3 */
    }
    else
    {
      /* if instance of handle correspond to ADC1 and  no external trigger present enable software conversion of regular channels */
      if((hadc->Instance == ADC1) && ((hadc->Instance->CR2 & ADC_CR2_EXTEN) == RESET))
      {
        /* Enable the selected ADC software conversion for regular group */
          hadc->Instance->CR2 |= (uint32_t)ADC_CR2_SWSTART;
      }
    }
  }
  else
  {
    /* Update ADC state machine to error */
    SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL);

    /* Set ADC error code to ADC IP internal error */
    SET_BIT(hadc->ErrorCode, HAL_ADC_ERROR_INTERNAL);
  }
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Disables the interrupt and stop ADC conversion of regular channels.
  * 
  * @note   Caution: This function will stop also injected channels.  
  *
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval HAL status.
  */
HAL_StatusTypeDef HAL_ADC_Stop_IT(ADC_HandleTypeDef* hadc)
{
  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  
  /* Process locked */
  __HAL_LOCK(hadc);
  
  /* Stop potential conversion on going, on regular and injected groups */
  /* Disable ADC peripheral */
  __HAL_ADC_DISABLE(hadc);
  
  /* Check if ADC is effectively disabled */
  if(HAL_IS_BIT_CLR(hadc->Instance->CR2, ADC_CR2_ADON))
  {
  	/* Disable ADC end of conversion interrupt for regular group */
    __HAL_ADC_DISABLE_IT(hadc, (ADC_IT_EOC | ADC_IT_OVR));

    /* Set ADC state */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                      HAL_ADC_STATE_READY);
  }
  
  /* Process unlocked */
  __HAL_UNLOCK(hadc);
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Handles ADC interrupt request  
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval None
  */
void HAL_ADC_IRQHandler(ADC_HandleTypeDef* hadc)
{
  uint32_t tmp1 = 0U, tmp2 = 0U;
  
  /* Check the parameters */
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ContinuousConvMode));
  assert_param(IS_ADC_REGULAR_LENGTH(hadc->Init.NbrOfConversion));
  assert_param(IS_ADC_EOCSelection(hadc->Init.EOCSelection));
  
  tmp1 = __HAL_ADC_GET_FLAG(hadc, ADC_FLAG_EOC);
  tmp2 = __HAL_ADC_GET_IT_SOURCE(hadc, ADC_IT_EOC);
  /* Check End of conversion flag for regular channels */
  if(tmp1 && tmp2)
  {
    /* Update state machine on conversion status if not in error state */
    if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL))
    {
      /* Set ADC state */
      SET_BIT(hadc->State, HAL_ADC_STATE_REG_EOC); 
    }
    
    /* Determine whether any further conversion upcoming on group regular   */
    /* by external trigger, continuous mode or scan sequence on going.      */
    /* Note: On STM32F4, there is no independent flag of end of sequence.   */
    /*       The test of scan sequence on going is done either with scan    */
    /*       sequence disabled or with end of conversion flag set to        */
    /*       of end of sequence.                                            */
    if(ADC_IS_SOFTWARE_START_REGULAR(hadc)                   &&
       (hadc->Init.ContinuousConvMode == DISABLE)            &&
       (HAL_IS_BIT_CLR(hadc->Instance->SQR1, ADC_SQR1_L) || 
        HAL_IS_BIT_CLR(hadc->Instance->CR2, ADC_CR2_EOCS)  )   )
    {
      /* Disable ADC end of single conversion interrupt on group regular */
      /* Note: Overrun interrupt was enabled with EOC interrupt in          */
      /* HAL_ADC_Start_IT(), but is not disabled here because can be used   */
      /* by overrun IRQ process below.                                      */
      __HAL_ADC_DISABLE_IT(hadc, ADC_IT_EOC);
      
      /* Set ADC state */
      CLEAR_BIT(hadc->State, HAL_ADC_STATE_REG_BUSY);
      
      if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_INJ_BUSY))
      {
        SET_BIT(hadc->State, HAL_ADC_STATE_READY);
      }
    }
    
    /* Conversion complete callback */
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
    hadc->ConvCpltCallback(hadc);
#else
    HAL_ADC_ConvCpltCallback(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
    
    /* Clear regular group conversion flag */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_STRT | ADC_FLAG_EOC);
  }
  
  tmp1 = __HAL_ADC_GET_FLAG(hadc, ADC_FLAG_JEOC);
  tmp2 = __HAL_ADC_GET_IT_SOURCE(hadc, ADC_IT_JEOC);                               
  /* Check End of conversion flag for injected channels */
  if(tmp1 && tmp2)
  {
    /* Update state machine on conversion status if not in error state */
    if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL))
    {
      /* Set ADC state */
      SET_BIT(hadc->State, HAL_ADC_STATE_INJ_EOC);
    }

    /* Determine whether any further conversion upcoming on group injected  */
    /* by external trigger, scan sequence on going or by automatic injected */
    /* conversion from group regular (same conditions as group regular      */
    /* interruption disabling above).                                       */
    if(ADC_IS_SOFTWARE_START_INJECTED(hadc)                    &&
       (HAL_IS_BIT_CLR(hadc->Instance->JSQR, ADC_JSQR_JL)  ||
        HAL_IS_BIT_CLR(hadc->Instance->CR2, ADC_CR2_EOCS)    ) &&
       (HAL_IS_BIT_CLR(hadc->Instance->CR1, ADC_CR1_JAUTO) &&
        (ADC_IS_SOFTWARE_START_REGULAR(hadc)       &&
        (hadc->Init.ContinuousConvMode == DISABLE)   )       )   )
    {
      /* Disable ADC end of single conversion interrupt on group injected */
      __HAL_ADC_DISABLE_IT(hadc, ADC_IT_JEOC);
      
      /* Set ADC state */
      CLEAR_BIT(hadc->State, HAL_ADC_STATE_INJ_BUSY);   

      if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_REG_BUSY))
      { 
        SET_BIT(hadc->State, HAL_ADC_STATE_READY);
      }
    }

    /* Conversion complete callback */ 
    /* Conversion complete callback */ 
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
      hadc->InjectedConvCpltCallback(hadc);
#else
      HAL_ADCEx_InjectedConvCpltCallback(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
    
    /* Clear injected group conversion flag */
    __HAL_ADC_CLEAR_FLAG(hadc, (ADC_FLAG_JSTRT | ADC_FLAG_JEOC));
  }
  
  tmp1 = __HAL_ADC_GET_FLAG(hadc, ADC_FLAG_AWD);
  tmp2 = __HAL_ADC_GET_IT_SOURCE(hadc, ADC_IT_AWD);                          
  /* Check Analog watchdog flag */
  if(tmp1 && tmp2)
  {
    if(__HAL_ADC_GET_FLAG(hadc, ADC_FLAG_AWD))
    {
      /* Set ADC state */
      SET_BIT(hadc->State, HAL_ADC_STATE_AWD1);
      
      /* Level out of window callback */
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
      hadc->LevelOutOfWindowCallback(hadc);
#else
      HAL_ADC_LevelOutOfWindowCallback(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
      
      /* Clear the ADC analog watchdog flag */
      __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_AWD);
    }
  }
  
  tmp1 = __HAL_ADC_GET_FLAG(hadc, ADC_FLAG_OVR);
  tmp2 = __HAL_ADC_GET_IT_SOURCE(hadc, ADC_IT_OVR);
  /* Check Overrun flag */
  if(tmp1 && tmp2)
  {
    /* Note: On STM32F4, ADC overrun can be set through other parameters    */
    /*       refer to description of parameter "EOCSelection" for more      */
    /*       details.                                                       */
    
    /* Set ADC error code to overrun */
    SET_BIT(hadc->ErrorCode, HAL_ADC_ERROR_OVR);
    
    /* Clear ADC overrun flag */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_OVR);
    
    /* Error callback */ 
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
      hadc->ErrorCallback(hadc);
#else
      HAL_ADC_ErrorCallback(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
    
    /* Clear the Overrun flag */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_OVR);
  }
}

/**
  * @brief  Enables ADC DMA request after last transfer (Single-ADC mode) and enables ADC peripheral  
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @param  pData The destination Buffer address.
  * @param  Length The length of data to be transferred from ADC peripheral to memory.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_Start_DMA(ADC_HandleTypeDef* hadc, uint32_t* pData, uint32_t Length)
{
  __IO uint32_t counter = 0U;
  ADC_Common_TypeDef *tmpADC_Common;
  
  /* Check the parameters */
  assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ContinuousConvMode));
  assert_param(IS_ADC_EXT_TRIG_EDGE(hadc->Init.ExternalTrigConvEdge)); 
  
  /* Process locked */
  __HAL_LOCK(hadc);
  
  /* Enable the ADC peripheral */
  /* Check if ADC peripheral is disabled in order to enable it and wait during 
  Tstab time the ADC's stabilization */
  if((hadc->Instance->CR2 & ADC_CR2_ADON) != ADC_CR2_ADON)
  {  
    /* Enable the Peripheral */
    __HAL_ADC_ENABLE(hadc);
    
    /* Delay for ADC stabilization time */
    /* Compute number of CPU cycles to wait for */
    counter = (ADC_STAB_DELAY_US * (SystemCoreClock / 1000000U));
    while(counter != 0U)
    {
      counter--;
    }
  }
  
  /* Check ADC DMA Mode                                                     */
  /* - disable the DMA Mode if it is already enabled                        */
  if((hadc->Instance->CR2 & ADC_CR2_DMA) == ADC_CR2_DMA)
  {
    CLEAR_BIT(hadc->Instance->CR2, ADC_CR2_DMA);
  }
  
  /* Start conversion if ADC is effectively enabled */
  if(HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_ADON))
  {
    /* Set ADC state                                                          */
    /* - Clear state bitfield related to regular group conversion results     */
    /* - Set state bitfield related to regular group operation                */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_READY | HAL_ADC_STATE_REG_EOC | HAL_ADC_STATE_REG_OVR,
                      HAL_ADC_STATE_REG_BUSY);
    
    /* If conversions on group regular are also triggering group injected,    */
    /* update ADC state.                                                      */
    if (READ_BIT(hadc->Instance->CR1, ADC_CR1_JAUTO) != RESET)
    {
      ADC_STATE_CLR_SET(hadc->State, HAL_ADC_STATE_INJ_EOC, HAL_ADC_STATE_INJ_BUSY);  
    }
    
    /* State machine update: Check if an injected conversion is ongoing */
    if (HAL_IS_BIT_SET(hadc->State, HAL_ADC_STATE_INJ_BUSY))
    {
      /* Reset ADC error code fields related to conversions on group regular */
      CLEAR_BIT(hadc->ErrorCode, (HAL_ADC_ERROR_OVR | HAL_ADC_ERROR_DMA));         
    }
    else
    {
      /* Reset ADC all error code fields */
      ADC_CLEAR_ERRORCODE(hadc);
    }

    /* Process unlocked */
    /* Unlock before starting ADC conversions: in case of potential           */
    /* interruption, to let the process to ADC IRQ Handler.                   */
    __HAL_UNLOCK(hadc);   

    /* Pointer to the common control register to which is belonging hadc    */
    /* (Depending on STM32F4 product, there may be up to 3 ADCs and 1 common */
    /* control register)                                                    */
    tmpADC_Common = ADC_COMMON_REGISTER(hadc);

    /* Set the DMA transfer complete callback */
    hadc->DMA_Handle->XferCpltCallback = ADC_DMAConvCplt;

    /* Set the DMA half transfer complete callback */
    hadc->DMA_Handle->XferHalfCpltCallback = ADC_DMAHalfConvCplt;
    
    /* Set the DMA error callback */
    hadc->DMA_Handle->XferErrorCallback = ADC_DMAError;

    
    /* Manage ADC and DMA start: ADC overrun interruption, DMA start, ADC     */
    /* start (in case of SW start):                                           */
    
    /* Clear regular group conversion flag and overrun flag */
    /* (To ensure of no unknown state from potential previous ADC operations) */
    __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_EOC | ADC_FLAG_OVR);

    /* Enable ADC overrun interrupt */
    __HAL_ADC_ENABLE_IT(hadc, ADC_IT_OVR);
    
    /* Enable ADC DMA mode */
    hadc->Instance->CR2 |= ADC_CR2_DMA;
    
    /* Start the DMA channel */
    HAL_DMA_Start_IT(hadc->DMA_Handle, (uint32_t)&hadc->Instance->DR, (uint32_t)pData, Length);
    
    /* Check if Multimode enabled */
    if(HAL_IS_BIT_CLR(tmpADC_Common->CCR, ADC_CCR_MULTI))
    {
#if defined(ADC2) && defined(ADC3)
      if((hadc->Instance == ADC1) || ((hadc->Instance == ADC2) && ((ADC->CCR & ADC_CCR_MULTI_Msk) < ADC_CCR_MULTI_0)) \
                                  || ((hadc->Instance == ADC3) && ((ADC->CCR & ADC_CCR_MULTI_Msk) < ADC_CCR_MULTI_4)))
      {
#endif /* ADC2 || ADC3 */
        /* if no external trigger present enable software conversion of regular channels */
        if((hadc->Instance->CR2 & ADC_CR2_EXTEN) == RESET) 
        {
          /* Enable the selected ADC software conversion for regular group */
          hadc->Instance->CR2 |= (uint32_t)ADC_CR2_SWSTART;
        }
#if defined(ADC2) && defined(ADC3)
      }
#endif /* ADC2 || ADC3 */
    }
    else
    {
      /* if instance of handle correspond to ADC1 and  no external trigger present enable software conversion of regular channels */
      if((hadc->Instance == ADC1) && ((hadc->Instance->CR2 & ADC_CR2_EXTEN) == RESET))
      {
        /* Enable the selected ADC software conversion for regular group */
          hadc->Instance->CR2 |= (uint32_t)ADC_CR2_SWSTART;
      }
    }
  }
  else
  {
    /* Update ADC state machine to error */
    SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL);

    /* Set ADC error code to ADC IP internal error */
    SET_BIT(hadc->ErrorCode, HAL_ADC_ERROR_INTERNAL);
  }
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Disables ADC DMA (Single-ADC mode) and disables ADC peripheral    
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_Stop_DMA(ADC_HandleTypeDef* hadc)
{
  HAL_StatusTypeDef tmp_hal_status = HAL_OK;
  
  /* Check the parameters */
  assert_param(IS_ADC_ALL_INSTANCE(hadc->Instance));
  
  /* Process locked */
  __HAL_LOCK(hadc);
  
  /* Stop potential conversion on going, on regular and injected groups */
  /* Disable ADC peripheral */
  __HAL_ADC_DISABLE(hadc);
  
  /* Check if ADC is effectively disabled */
  if(HAL_IS_BIT_CLR(hadc->Instance->CR2, ADC_CR2_ADON))
  {
    /* Disable the selected ADC DMA mode */
    hadc->Instance->CR2 &= ~ADC_CR2_DMA;
    
    /* Disable the DMA channel (in case of DMA in circular mode or stop while */
    /* DMA transfer is on going)                                              */
    if (hadc->DMA_Handle->State == HAL_DMA_STATE_BUSY)
    {
      tmp_hal_status = HAL_DMA_Abort(hadc->DMA_Handle);
      
      /* Check if DMA channel effectively disabled */
      if (tmp_hal_status != HAL_OK)
      {
        /* Update ADC state machine to error */
        SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_DMA);
      }
    }
    
    /* Disable ADC overrun interrupt */
    __HAL_ADC_DISABLE_IT(hadc, ADC_IT_OVR);
    
    /* Set ADC state */
    ADC_STATE_CLR_SET(hadc->State,
                      HAL_ADC_STATE_REG_BUSY | HAL_ADC_STATE_INJ_BUSY,
                      HAL_ADC_STATE_READY);
  }
  
  /* Process unlocked */
  __HAL_UNLOCK(hadc);
  
  /* Return function status */
  return tmp_hal_status;
}

/**
  * @brief  Gets the converted value from data register of regular channel.
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval Converted value
  */
uint32_t HAL_ADC_GetValue(ADC_HandleTypeDef* hadc)
{       
  /* Return the selected ADC converted value */ 
  return hadc->Instance->DR;
}

/**
  * @brief  Regular conversion complete callback in non blocking mode 
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval None
  */
__weak void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hadc);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_ADC_ConvCpltCallback could be implemented in the user file
   */
}

/**
  * @brief  Regular conversion half DMA transfer callback in non blocking mode 
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval None
  */
__weak void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef* hadc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hadc);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_ADC_ConvHalfCpltCallback could be implemented in the user file
   */
}

/**
  * @brief  Analog watchdog callback in non blocking mode 
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval None
  */
__weak void HAL_ADC_LevelOutOfWindowCallback(ADC_HandleTypeDef* hadc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hadc);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_ADC_LevelOoutOfWindowCallback could be implemented in the user file
   */
}

/**
  * @brief  Error ADC callback.
  * @note   In case of error due to overrun when using ADC with DMA transfer 
  *         (HAL ADC handle paramater "ErrorCode" to state "HAL_ADC_ERROR_OVR"):
  *         - Reinitialize the DMA using function "HAL_ADC_Stop_DMA()".
  *         - If needed, restart a new ADC conversion using function
  *           "HAL_ADC_Start_DMA()"
  *           (this function is also clearing overrun flag)
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval None
  */
__weak void HAL_ADC_ErrorCallback(ADC_HandleTypeDef *hadc)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hadc);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_ADC_ErrorCallback could be implemented in the user file
   */
}

/**
  * @}
  */
  
/** @defgroup ADC_Exported_Functions_Group3 Peripheral Control functions
 *  @brief   	Peripheral Control functions 
 *
@verbatim   
 ===============================================================================
             ##### Peripheral Control functions #####
 ===============================================================================  
    [..]  This section provides functions allowing to:
      (+) Configure regular channels. 
      (+) Configure injected channels.
      (+) Configure multimode.
      (+) Configure the analog watch dog.
      
@endverbatim
  * @{
  */

  /**
  * @brief  Configures for the selected ADC regular channel its corresponding
  *         rank in the sequencer and its sample time.
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @param  sConfig ADC configuration structure. 
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_ADC_ConfigChannel(ADC_HandleTypeDef* hadc, ADC_ChannelConfTypeDef* sConfig)
{
  __IO uint32_t counter = 0U;
  ADC_Common_TypeDef *tmpADC_Common;
  
  /* Check the parameters */
  assert_param(IS_ADC_CHANNEL(sConfig->Channel));
  assert_param(IS_ADC_REGULAR_RANK(sConfig->Rank));
  assert_param(IS_ADC_SAMPLE_TIME(sConfig->SamplingTime));
  
  /* Process locked */
  __HAL_LOCK(hadc);
    
  /* if ADC_Channel_10 ... ADC_Channel_18 is selected */
  if (sConfig->Channel > ADC_CHANNEL_9)
  {
    /* Clear the old sample time */
    hadc->Instance->SMPR1 &= ~ADC_SMPR1(ADC_SMPR1_SMP10, sConfig->Channel);
    
    /* Set the new sample time */
    hadc->Instance->SMPR1 |= ADC_SMPR1(sConfig->SamplingTime, sConfig->Channel);
  }
  else /* ADC_Channel include in ADC_Channel_[0..9] */
  {
    /* Clear the old sample time */
    hadc->Instance->SMPR2 &= ~ADC_SMPR2(ADC_SMPR2_SMP0, sConfig->Channel);
    
    /* Set the new sample time */
    hadc->Instance->SMPR2 |= ADC_SMPR2(sConfig->SamplingTime, sConfig->Channel);
  }
  
  /* For Rank 1 to 6 */
  if (sConfig->Rank < 7U)
  {
    /* Clear the old SQx bits for the selected rank */
    hadc->Instance->SQR3 &= ~ADC_SQR3_RK(ADC_SQR3_SQ1, sConfig->Rank);
    
    /* Set the SQx bits for the selected rank */
    hadc->Instance->SQR3 |= ADC_SQR3_RK(sConfig->Channel, sConfig->Rank);
  }
  /* For Rank 7 to 12 */
  else if (sConfig->Rank < 13U)
  {
    /* Clear the old SQx bits for the selected rank */
    hadc->Instance->SQR2 &= ~ADC_SQR2_RK(ADC_SQR2_SQ7, sConfig->Rank);
    
    /* Set the SQx bits for the selected rank */
    hadc->Instance->SQR2 |= ADC_SQR2_RK(sConfig->Channel, sConfig->Rank);
  }
  /* For Rank 13 to 16 */
  else
  {
    /* Clear the old SQx bits for the selected rank */
    hadc->Instance->SQR1 &= ~ADC_SQR1_RK(ADC_SQR1_SQ13, sConfig->Rank);
    
    /* Set the SQx bits for the selected rank */
    hadc->Instance->SQR1 |= ADC_SQR1_RK(sConfig->Channel, sConfig->Rank);
  }

    /* Pointer to the common control register to which is belonging hadc    */
    /* (Depending on STM32F4 product, there may be up to 3 ADCs and 1 common */
    /* control register)                                                    */
    tmpADC_Common = ADC_COMMON_REGISTER(hadc);

  /* if ADC1 Channel_18 is selected for VBAT Channel ennable VBATE */
  if ((hadc->Instance == ADC1) && (sConfig->Channel == ADC_CHANNEL_VBAT))
  {
    /* Disable the TEMPSENSOR channel in case of using board with multiplixed ADC_CHANNEL_VBAT & ADC_CHANNEL_TEMPSENSOR*/    
    if ((uint16_t)ADC_CHANNEL_TEMPSENSOR == (uint16_t)ADC_CHANNEL_VBAT)
    {
      tmpADC_Common->CCR &= ~ADC_CCR_TSVREFE;
    }
    /* Enable the VBAT channel*/
    tmpADC_Common->CCR |= ADC_CCR_VBATE;
  }
  
  /* if ADC1 Channel_16 or Channel_18 is selected for Temperature sensor or 
     Channel_17 is selected for VREFINT enable TSVREFE */
  if ((hadc->Instance == ADC1) && ((sConfig->Channel == ADC_CHANNEL_TEMPSENSOR) || (sConfig->Channel == ADC_CHANNEL_VREFINT)))
  {
    /* Disable the VBAT channel in case of using board with multiplixed ADC_CHANNEL_VBAT & ADC_CHANNEL_TEMPSENSOR*/
    if ((uint16_t)ADC_CHANNEL_TEMPSENSOR == (uint16_t)ADC_CHANNEL_VBAT)
    {
      tmpADC_Common->CCR &= ~ADC_CCR_VBATE;
    }
    /* Enable the Temperature sensor and VREFINT channel*/
    tmpADC_Common->CCR |= ADC_CCR_TSVREFE;
    
    if(sConfig->Channel == ADC_CHANNEL_TEMPSENSOR)
    {
      /* Delay for temperature sensor stabilization time */
      /* Compute number of CPU cycles to wait for */
      counter = (ADC_TEMPSENSOR_DELAY_US * (SystemCoreClock / 1000000U));
      while(counter != 0U)
      {
        counter--;
      }
    }
  }
  
  /* Process unlocked */
  __HAL_UNLOCK(hadc);
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Configures the analog watchdog.
  * @note Analog watchdog thresholds can be modified while ADC conversion
  * is on going.
  * In this case, some constraints must be taken into account:
  * The programmed threshold values are effective from the next
  * ADC EOC (end of unitary conversion).
  * Considering that registers write delay may happen due to
  * bus activity, this might cause an uncertainty on the
  * effective timing of the new programmed threshold values.
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @param  AnalogWDGConfig  pointer to an ADC_AnalogWDGConfTypeDef structure 
  *         that contains the configuration information of ADC analog watchdog.
  * @retval HAL status	  
  */
HAL_StatusTypeDef HAL_ADC_AnalogWDGConfig(ADC_HandleTypeDef* hadc, ADC_AnalogWDGConfTypeDef* AnalogWDGConfig)
{
#ifdef USE_FULL_ASSERT  
  uint32_t tmp = 0U;
#endif /* USE_FULL_ASSERT  */  
  
  /* Check the parameters */
  assert_param(IS_ADC_ANALOG_WATCHDOG(AnalogWDGConfig->WatchdogMode));
  assert_param(IS_ADC_CHANNEL(AnalogWDGConfig->Channel));
  assert_param(IS_FUNCTIONAL_STATE(AnalogWDGConfig->ITMode));

#ifdef USE_FULL_ASSERT  
  tmp = ADC_GET_RESOLUTION(hadc);
  assert_param(IS_ADC_RANGE(tmp, AnalogWDGConfig->HighThreshold));
  assert_param(IS_ADC_RANGE(tmp, AnalogWDGConfig->LowThreshold));
#endif /* USE_FULL_ASSERT  */
  
  /* Process locked */
  __HAL_LOCK(hadc);
  
  if(AnalogWDGConfig->ITMode == ENABLE)
  {
    /* Enable the ADC Analog watchdog interrupt */
    __HAL_ADC_ENABLE_IT(hadc, ADC_IT_AWD);
  }
  else
  {
    /* Disable the ADC Analog watchdog interrupt */
    __HAL_ADC_DISABLE_IT(hadc, ADC_IT_AWD);
  }
  
  /* Clear AWDEN, JAWDEN and AWDSGL bits */
  hadc->Instance->CR1 &=  ~(ADC_CR1_AWDSGL | ADC_CR1_JAWDEN | ADC_CR1_AWDEN);
  
  /* Set the analog watchdog enable mode */
  hadc->Instance->CR1 |= AnalogWDGConfig->WatchdogMode;
  
  /* Set the high threshold */
  hadc->Instance->HTR = AnalogWDGConfig->HighThreshold;
  
  /* Set the low threshold */
  hadc->Instance->LTR = AnalogWDGConfig->LowThreshold;
  
  /* Clear the Analog watchdog channel select bits */
  hadc->Instance->CR1 &= ~ADC_CR1_AWDCH;
  
  /* Set the Analog watchdog channel */
  hadc->Instance->CR1 |= (uint32_t)((uint16_t)(AnalogWDGConfig->Channel));
  
  /* Process unlocked */
  __HAL_UNLOCK(hadc);
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @}
  */

/** @defgroup ADC_Exported_Functions_Group4 ADC Peripheral State functions
 *  @brief   ADC Peripheral State functions 
 *
@verbatim   
 ===============================================================================
            ##### Peripheral State and errors functions #####
 ===============================================================================  
    [..]
    This subsection provides functions allowing to
      (+) Check the ADC state
      (+) Check the ADC Error
         
@endverbatim
  * @{
  */
  
/**
  * @brief  return the ADC state
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval HAL state
  */
uint32_t HAL_ADC_GetState(ADC_HandleTypeDef* hadc)
{
  /* Return ADC state */
  return hadc->State;
}

/**
  * @brief  Return the ADC error code
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.
  * @retval ADC Error Code
  */
uint32_t HAL_ADC_GetError(ADC_HandleTypeDef *hadc)
{
  return hadc->ErrorCode;
}

/**
  * @}
  */

/** @addtogroup ADC_Private_Functions
  * @{
  */

/**
  * @brief  Initializes the ADCx peripheral according to the specified parameters 
  *         in the ADC_InitStruct without initializing the ADC MSP.       
  * @param  hadc pointer to a ADC_HandleTypeDef structure that contains
  *         the configuration information for the specified ADC.  
  * @retval None
  */
static void ADC_Init(ADC_HandleTypeDef* hadc)
{
  ADC_Common_TypeDef *tmpADC_Common;
  
  /* Set ADC parameters */
  /* Pointer to the common control register to which is belonging hadc    */
  /* (Depending on STM32F4 product, there may be up to 3 ADCs and 1 common */
  /* control register)                                                    */
  tmpADC_Common = ADC_COMMON_REGISTER(hadc);
  
  /* Set the ADC clock prescaler */
  tmpADC_Common->CCR &= ~(ADC_CCR_ADCPRE);
  tmpADC_Common->CCR |=  hadc->Init.ClockPrescaler;
  
  /* Set ADC scan mode */
  hadc->Instance->CR1 &= ~(ADC_CR1_SCAN);
  hadc->Instance->CR1 |=  ADC_CR1_SCANCONV(hadc->Init.ScanConvMode);
  
  /* Set ADC resolution */
  hadc->Instance->CR1 &= ~(ADC_CR1_RES);
  hadc->Instance->CR1 |=  hadc->Init.Resolution;
  
  /* Set ADC data alignment */
  hadc->Instance->CR2 &= ~(ADC_CR2_ALIGN);
  hadc->Instance->CR2 |= hadc->Init.DataAlign;
  
  /* Enable external trigger if trigger selection is different of software  */
  /* start.                                                                 */
  /* Note: This configuration keeps the hardware feature of parameter       */
  /*       ExternalTrigConvEdge "trigger edge none" equivalent to           */
  /*       software start.                                                  */
  if(hadc->Init.ExternalTrigConv != ADC_SOFTWARE_START)
  {
    /* Select external trigger to start conversion */
    hadc->Instance->CR2 &= ~(ADC_CR2_EXTSEL);
    hadc->Instance->CR2 |= hadc->Init.ExternalTrigConv;
    
    /* Select external trigger polarity */
    hadc->Instance->CR2 &= ~(ADC_CR2_EXTEN);
    hadc->Instance->CR2 |= hadc->Init.ExternalTrigConvEdge;
  }
  else
  {
    /* Reset the external trigger */
    hadc->Instance->CR2 &= ~(ADC_CR2_EXTSEL);
    hadc->Instance->CR2 &= ~(ADC_CR2_EXTEN);
  }
  
  /* Enable or disable ADC continuous conversion mode */
  hadc->Instance->CR2 &= ~(ADC_CR2_CONT);
  hadc->Instance->CR2 |= ADC_CR2_CONTINUOUS((uint32_t)hadc->Init.ContinuousConvMode);
  
  if(hadc->Init.DiscontinuousConvMode != DISABLE)
  {
    assert_param(IS_ADC_REGULAR_DISC_NUMBER(hadc->Init.NbrOfDiscConversion));
  
    /* Enable the selected ADC regular discontinuous mode */
    hadc->Instance->CR1 |= (uint32_t)ADC_CR1_DISCEN;
    
    /* Set the number of channels to be converted in discontinuous mode */
    hadc->Instance->CR1 &= ~(ADC_CR1_DISCNUM);
    hadc->Instance->CR1 |=  ADC_CR1_DISCONTINUOUS(hadc->Init.NbrOfDiscConversion);
  }
  else
  {
    /* Disable the selected ADC regular discontinuous mode */
    hadc->Instance->CR1 &= ~(ADC_CR1_DISCEN);
  }
  
  /* Set ADC number of conversion */
  hadc->Instance->SQR1 &= ~(ADC_SQR1_L);
  hadc->Instance->SQR1 |=  ADC_SQR1(hadc->Init.NbrOfConversion);
  
  /* Enable or disable ADC DMA continuous request */
  hadc->Instance->CR2 &= ~(ADC_CR2_DDS);
  hadc->Instance->CR2 |= ADC_CR2_DMAContReq((uint32_t)hadc->Init.DMAContinuousRequests);
  
  /* Enable or disable ADC end of conversion selection */
  hadc->Instance->CR2 &= ~(ADC_CR2_EOCS);
  hadc->Instance->CR2 |= ADC_CR2_EOCSelection(hadc->Init.EOCSelection);
}

/**
  * @brief  DMA transfer complete callback. 
  * @param  hdma pointer to a DMA_HandleTypeDef structure that contains
  *                the configuration information for the specified DMA module.
  * @retval None
  */
static void ADC_DMAConvCplt(DMA_HandleTypeDef *hdma)   
{
  /* Retrieve ADC handle corresponding to current DMA handle */
  ADC_HandleTypeDef* hadc = ( ADC_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;
  
  /* Update state machine on conversion status if not in error state */
  if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL | HAL_ADC_STATE_ERROR_DMA))
  {
    /* Update ADC state machine */
    SET_BIT(hadc->State, HAL_ADC_STATE_REG_EOC);
    
    /* Determine whether any further conversion upcoming on group regular   */
    /* by external trigger, continuous mode or scan sequence on going.      */
    /* Note: On STM32F4, there is no independent flag of end of sequence.   */
    /*       The test of scan sequence on going is done either with scan    */
    /*       sequence disabled or with end of conversion flag set to        */
    /*       of end of sequence.                                            */
    if(ADC_IS_SOFTWARE_START_REGULAR(hadc)                   &&
       (hadc->Init.ContinuousConvMode == DISABLE)            &&
       (HAL_IS_BIT_CLR(hadc->Instance->SQR1, ADC_SQR1_L) || 
        HAL_IS_BIT_CLR(hadc->Instance->CR2, ADC_CR2_EOCS)  )   )
    {
      /* Disable ADC end of single conversion interrupt on group regular */
      /* Note: Overrun interrupt was enabled with EOC interrupt in          */
      /* HAL_ADC_Start_IT(), but is not disabled here because can be used   */
      /* by overrun IRQ process below.                                      */
      __HAL_ADC_DISABLE_IT(hadc, ADC_IT_EOC);
      
      /* Set ADC state */
      CLEAR_BIT(hadc->State, HAL_ADC_STATE_REG_BUSY);   
      
      if (HAL_IS_BIT_CLR(hadc->State, HAL_ADC_STATE_INJ_BUSY))
      {
        SET_BIT(hadc->State, HAL_ADC_STATE_READY);
      }
    }
    
    /* Conversion complete callback */
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
    hadc->ConvCpltCallback(hadc);
#else
    HAL_ADC_ConvCpltCallback(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
  }
  else /* DMA and-or internal error occurred */
  {
    if ((hadc->State & HAL_ADC_STATE_ERROR_INTERNAL) != 0UL)
    {
      /* Call HAL ADC Error Callback function */
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
      hadc->ErrorCallback(hadc);
#else
      HAL_ADC_ErrorCallback(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
    }
	else
	{
      /* Call DMA error callback */
      hadc->DMA_Handle->XferErrorCallback(hdma);
    }
  }
}

/**
  * @brief  DMA half transfer complete callback. 
  * @param  hdma pointer to a DMA_HandleTypeDef structure that contains
  *                the configuration information for the specified DMA module.
  * @retval None
  */
static void ADC_DMAHalfConvCplt(DMA_HandleTypeDef *hdma)   
{
  ADC_HandleTypeDef* hadc = ( ADC_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;
   /* Half conversion callback */
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
  hadc->ConvHalfCpltCallback(hadc);
#else
  HAL_ADC_ConvHalfCpltCallback(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
}

/**
  * @brief  DMA error callback 
  * @param  hdma pointer to a DMA_HandleTypeDef structure that contains
  *                the configuration information for the specified DMA module.
  * @retval None
  */
static void ADC_DMAError(DMA_HandleTypeDef *hdma)   
{
  ADC_HandleTypeDef* hadc = ( ADC_HandleTypeDef* )((DMA_HandleTypeDef* )hdma)->Parent;
  hadc->State= HAL_ADC_STATE_ERROR_DMA;
  /* Set ADC error code to DMA error */
  hadc->ErrorCode |= HAL_ADC_ERROR_DMA;
   /* Error callback */
#if (USE_HAL_ADC_REGISTER_CALLBACKS == 1)
  hadc->ErrorCallback(hadc);
#else
  HAL_ADC_ErrorCallback(hadc);
#endif /* USE_HAL_ADC_REGISTER_CALLBACKS */
}

/**
  * @}
  */

/**
  * @}
  */

#endif /* HAL_ADC_MODULE_ENABLED */
/**
  * @}
  */ 

/**
  * @}
  */ 

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
