/**
  ******************************************************************************
  * @file    stm32f4xx_hal_can.c
  * @author  MCD Application Team
  * @brief   This file provides firmware functions to manage the following 
  *          functionalities of the Controller Area Network (CAN) peripheral:
  *           + Initialization and de-initialization functions 
  *           + IO operation functions
  *           + Peripheral Control functions
  *           + Peripheral State and Error functions
  *
  @verbatim
  ==============================================================================
                                 ##### User NOTE #####
  ==============================================================================
    [..]
      (#) This HAL CAN driver is deprecated, it contains some CAN Tx/Rx FIFO management limitations.
          Another HAL CAN driver version has been designed with new API's, to fix these limitations.

  ==============================================================================
                        ##### How to use this driver #####
  ==============================================================================
    [..]            
      (#) Enable the CAN controller interface clock using 
          __HAL_RCC_CAN1_CLK_ENABLE() for CAN1, __HAL_RCC_CAN2_CLK_ENABLE() for CAN2
         and __HAL_RCC_CAN3_CLK_ENABLE() for CAN3
      -@- In case you are using CAN2 only, you have to enable the CAN1 clock.
       
      (#) CAN pins configuration
        (++) Enable the clock for the CAN GPIOs using the following function:
             __GPIOx_CLK_ENABLE()   
        (++) Connect and configure the involved CAN pins to AF9 using the 
              following function HAL_GPIO_Init() 
              
      (#) Initialize and configure the CAN using CAN_Init() function.   
                 
      (#) Transmit the desired CAN frame using HAL_CAN_Transmit() function.

      (#) Or transmit the desired CAN frame using HAL_CAN_Transmit_IT() function.
           
      (#) Receive a CAN frame using HAL_CAN_Receive() function.

      (#) Or receive a CAN frame using HAL_CAN_Receive_IT() function.

     *** Polling mode IO operation ***
     =================================
     [..]    
       (+) Start the CAN peripheral transmission and wait the end of this operation 
           using HAL_CAN_Transmit(), at this stage user can specify the value of timeout
           according to his end application
       (+) Start the CAN peripheral reception and wait the end of this operation 
           using HAL_CAN_Receive(), at this stage user can specify the value of timeout
           according to his end application 
       
     *** Interrupt mode IO operation ***    
     ===================================
     [..]    
       (+) Start the CAN peripheral transmission using HAL_CAN_Transmit_IT()
       (+) Start the CAN peripheral reception using HAL_CAN_Receive_IT()         
       (+) Use HAL_CAN_IRQHandler() called under the used CAN Interrupt subroutine
       (+) At CAN end of transmission HAL_CAN_TxCpltCallback() function is executed and user can 
            add his own code by customization of function pointer HAL_CAN_TxCpltCallback 
       (+) In case of CAN Error, HAL_CAN_ErrorCallback() function is executed and user can 
            add his own code by customization of function pointer HAL_CAN_ErrorCallback
 
     *** CAN HAL driver macros list ***
     ============================================= 
     [..]
       Below the list of most used macros in CAN HAL driver.
       
      (+) __HAL_CAN_ENABLE_IT: Enable the specified CAN interrupts
      (+) __HAL_CAN_DISABLE_IT: Disable the specified CAN interrupts
      (+) __HAL_CAN_GET_IT_SOURCE: Check if the specified CAN interrupt source is enabled or disabled
      (+) __HAL_CAN_CLEAR_FLAG: Clear the CAN's pending flags
      (+) __HAL_CAN_GET_FLAG: Get the selected CAN's flag status

     [..]
      (@) You can refer to the CAN Legacy HAL driver header file for more useful macros

  @endverbatim
           
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT(c) 2017 STMicroelectronics</center></h2>
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *   1. Redistributions of source code must retain the above copyright notice,
  *      this list of conditions and the following disclaimer.
  *   2. Redistributions in binary form must reproduce the above copyright notice,
  *      this list of conditions and the following disclaimer in the documentation
  *      and/or other materials provided with the distribution.
  *   3. Neither the name of STMicroelectronics nor the names of its contributors
  *      may be used to endorse or promote products derived from this software
  *      without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_hal.h"

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup CAN CAN
  * @brief CAN driver modules
  * @{
  */ 
  
#ifdef HAL_CAN_LEGACY_MODULE_ENABLED  

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx) || defined(STM32F412Zx) ||\
    defined(STM32F412Vx) || defined(STM32F412Rx) || defined(STM32F412Cx) || defined(STM32F413xx) ||\
    defined(STM32F423xx)

#ifdef HAL_CAN_MODULE_ENABLED
/* Select HAL CAN module in stm32f4xx_hal_conf.h file:
   (#) HAL_CAN_MODULE_ENABLED for new HAL CAN driver fixing FIFO limitations
   (#) HAL_CAN_LEGACY_MODULE_ENABLED for legacy HAL CAN driver */
#error 'The HAL CAN driver cannot be used with its legacy, Please ensure to enable only one HAL CAN module at once in stm32f4xx_hal_conf.h file'
#endif /* HAL_CAN_MODULE_ENABLED */

#warning 'Legacy HAL CAN driver is enabled! It can be used with known limitations, refer to the release notes. However it is recommended to use rather the new HAL CAN driver'

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @addtogroup CAN_Private_Constants
  * @{
  */
#define CAN_TIMEOUT_VALUE  10U
/**
  * @}
  */
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/** @addtogroup CAN_Private_Functions
  * @{
  */
static HAL_StatusTypeDef CAN_Receive_IT(CAN_HandleTypeDef* hcan, uint8_t FIFONumber);
static HAL_StatusTypeDef CAN_Transmit_IT(CAN_HandleTypeDef* hcan);
/**
  * @}
  */

/* Exported functions --------------------------------------------------------*/
/** @defgroup CAN_Exported_Functions CAN Exported Functions
  * @{
  */

/** @defgroup CAN_Exported_Functions_Group1 Initialization and de-initialization functions 
 *  @brief    Initialization and Configuration functions 
 *
@verbatim    
  ==============================================================================
              ##### Initialization and de-initialization functions #####
  ==============================================================================
    [..]  This section provides functions allowing to:
      (+) Initialize and configure the CAN. 
      (+) De-initialize the CAN. 
         
@endverbatim
  * @{
  */
  
/**
  * @brief  Initializes the CAN peripheral according to the specified
  *         parameters in the CAN_InitStruct.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_Init(CAN_HandleTypeDef* hcan)
{
  uint32_t InitStatus = CAN_INITSTATUS_FAILED;
  uint32_t tickstart = 0U;
  
  /* Check CAN handle */
  if(hcan == NULL)
  {
     return HAL_ERROR;
  }
  
  /* Check the parameters */
  assert_param(IS_CAN_ALL_INSTANCE(hcan->Instance));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.TTCM));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.ABOM));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.AWUM));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.NART));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.RFLM));
  assert_param(IS_FUNCTIONAL_STATE(hcan->Init.TXFP));
  assert_param(IS_CAN_MODE(hcan->Init.Mode));
  assert_param(IS_CAN_SJW(hcan->Init.SJW));
  assert_param(IS_CAN_BS1(hcan->Init.BS1));
  assert_param(IS_CAN_BS2(hcan->Init.BS2));
  assert_param(IS_CAN_PRESCALER(hcan->Init.Prescaler));
  

  if(hcan->State == HAL_CAN_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hcan->Lock = HAL_UNLOCKED;
    /* Init the low level hardware */
    HAL_CAN_MspInit(hcan);
  }
  
  /* Initialize the CAN state*/
  hcan->State = HAL_CAN_STATE_BUSY;
  
  /* Exit from sleep mode */
  hcan->Instance->MCR &= (~(uint32_t)CAN_MCR_SLEEP);

  /* Request initialisation */
  hcan->Instance->MCR |= CAN_MCR_INRQ ;
  
  /* Get tick */
  tickstart = HAL_GetTick();

  /* Wait the acknowledge */
  while((hcan->Instance->MSR & CAN_MSR_INAK) != CAN_MSR_INAK)
  {
    if((HAL_GetTick() - tickstart ) > CAN_TIMEOUT_VALUE)
    {
      hcan->State= HAL_CAN_STATE_TIMEOUT;
      /* Process unlocked */
      __HAL_UNLOCK(hcan);
      return HAL_TIMEOUT;
    }
  }

  /* Check acknowledge */
  if ((hcan->Instance->MSR & CAN_MSR_INAK) == CAN_MSR_INAK)
  {
    /* Set the time triggered communication mode */
    if (hcan->Init.TTCM == ENABLE)
    {
      hcan->Instance->MCR |= CAN_MCR_TTCM;
    }
    else
    {
      hcan->Instance->MCR &= ~(uint32_t)CAN_MCR_TTCM;
    }

    /* Set the automatic bus-off management */
    if (hcan->Init.ABOM == ENABLE)
    {
      hcan->Instance->MCR |= CAN_MCR_ABOM;
    }
    else
    {
      hcan->Instance->MCR &= ~(uint32_t)CAN_MCR_ABOM;
    }

    /* Set the automatic wake-up mode */
    if (hcan->Init.AWUM == ENABLE)
    {
      hcan->Instance->MCR |= CAN_MCR_AWUM;
    }
    else
    {
      hcan->Instance->MCR &= ~(uint32_t)CAN_MCR_AWUM;
    }

    /* Set the no automatic retransmission */
    if (hcan->Init.NART == ENABLE)
    {
      hcan->Instance->MCR |= CAN_MCR_NART;
    }
    else
    {
      hcan->Instance->MCR &= ~(uint32_t)CAN_MCR_NART;
    }

    /* Set the receive FIFO locked mode */
    if (hcan->Init.RFLM == ENABLE)
    {
      hcan->Instance->MCR |= CAN_MCR_RFLM;
    }
    else
    {
      hcan->Instance->MCR &= ~(uint32_t)CAN_MCR_RFLM;
    }

    /* Set the transmit FIFO priority */
    if (hcan->Init.TXFP == ENABLE)
    {
      hcan->Instance->MCR |= CAN_MCR_TXFP;
    }
    else
    {
      hcan->Instance->MCR &= ~(uint32_t)CAN_MCR_TXFP;
    }

    /* Set the bit timing register */
    hcan->Instance->BTR = (uint32_t)((uint32_t)hcan->Init.Mode) | \
                ((uint32_t)hcan->Init.SJW) | \
                ((uint32_t)hcan->Init.BS1) | \
                ((uint32_t)hcan->Init.BS2) | \
               ((uint32_t)hcan->Init.Prescaler - 1U);

    /* Request leave initialisation */
    hcan->Instance->MCR &= ~(uint32_t)CAN_MCR_INRQ;

  /* Get tick */
  tickstart = HAL_GetTick();

   /* Wait the acknowledge */
   while((hcan->Instance->MSR & CAN_MSR_INAK) == CAN_MSR_INAK)
   {
    if((HAL_GetTick() - tickstart ) > CAN_TIMEOUT_VALUE)
     {
       hcan->State= HAL_CAN_STATE_TIMEOUT;
       /* Process unlocked */
       __HAL_UNLOCK(hcan);
       return HAL_TIMEOUT;
     }
   }

    /* Check acknowledged */
    if ((hcan->Instance->MSR & CAN_MSR_INAK) != CAN_MSR_INAK)
    {
      InitStatus = CAN_INITSTATUS_SUCCESS;
    }
  }
 
  if(InitStatus == CAN_INITSTATUS_SUCCESS)
  {
    /* Set CAN error code to none */
    hcan->ErrorCode = HAL_CAN_ERROR_NONE;
    
    /* Initialize the CAN state */
    hcan->State = HAL_CAN_STATE_READY;
  
    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Initialize the CAN state */
    hcan->State = HAL_CAN_STATE_ERROR;
    
    /* Return function status */
    return HAL_ERROR;
  }
}

/**
  * @brief  Configures the CAN reception filter according to the specified
  *         parameters in the CAN_FilterInitStruct.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @param  sFilterConfig pointer to a CAN_FilterConfTypeDef structure that
  *         contains the filter configuration information.
  * @retval None
  */
HAL_StatusTypeDef HAL_CAN_ConfigFilter(CAN_HandleTypeDef* hcan, CAN_FilterConfTypeDef* sFilterConfig)
{
  uint32_t filternbrbitpos = 0U;
  CAN_TypeDef *can_ip;
  
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);

  /* Check the parameters */
  assert_param(IS_CAN_FILTER_NUMBER(sFilterConfig->FilterNumber));
  assert_param(IS_CAN_FILTER_MODE(sFilterConfig->FilterMode));
  assert_param(IS_CAN_FILTER_SCALE(sFilterConfig->FilterScale));
  assert_param(IS_CAN_FILTER_FIFO(sFilterConfig->FilterFIFOAssignment));
  assert_param(IS_FUNCTIONAL_STATE(sFilterConfig->FilterActivation));
  assert_param(IS_CAN_BANKNUMBER(sFilterConfig->BankNumber));
  
  filternbrbitpos = 1U << sFilterConfig->FilterNumber;
#if defined (CAN3)
  /* Check the CAN instance */
  if(hcan->Instance == CAN3)
  {	
    can_ip = CAN3;
  }
  else
  {
    can_ip = CAN1;    
  }
#else
  can_ip = CAN1;
#endif
  
  /* Initialisation mode for the filter */
  can_ip->FMR |= (uint32_t)CAN_FMR_FINIT;

#if defined (CAN2)  
  /* Select the start slave bank */
  can_ip->FMR &= ~((uint32_t)CAN_FMR_CAN2SB);
  can_ip->FMR |= (uint32_t)(sFilterConfig->BankNumber << 8U);
#endif
     
  /* Filter Deactivation */
  can_ip->FA1R &= ~(uint32_t)filternbrbitpos;

  /* Filter Scale */
  if (sFilterConfig->FilterScale == CAN_FILTERSCALE_16BIT)
  {
    /* 16-bit scale for the filter */
    can_ip->FS1R &= ~(uint32_t)filternbrbitpos;

    /* First 16-bit identifier and First 16-bit mask */
    /* Or First 16-bit identifier and Second 16-bit identifier */
    can_ip->sFilterRegister[sFilterConfig->FilterNumber].FR1 = 
       ((0x0000FFFFU & (uint32_t)sFilterConfig->FilterMaskIdLow) << 16U) |
        (0x0000FFFFU & (uint32_t)sFilterConfig->FilterIdLow);

    /* Second 16-bit identifier and Second 16-bit mask */
    /* Or Third 16-bit identifier and Fourth 16-bit identifier */
    can_ip->sFilterRegister[sFilterConfig->FilterNumber].FR2 = 
       ((0x0000FFFFU & (uint32_t)sFilterConfig->FilterMaskIdHigh) << 16U) |
        (0x0000FFFFU & (uint32_t)sFilterConfig->FilterIdHigh);
  }

  if (sFilterConfig->FilterScale == CAN_FILTERSCALE_32BIT)
  {
    /* 32-bit scale for the filter */
    can_ip->FS1R |= filternbrbitpos;
    
    /* 32-bit identifier or First 32-bit identifier */
    can_ip->sFilterRegister[sFilterConfig->FilterNumber].FR1 = 
       ((0x0000FFFFU & (uint32_t)sFilterConfig->FilterIdHigh) << 16U) |
        (0x0000FFFFU & (uint32_t)sFilterConfig->FilterIdLow);
    /* 32-bit mask or Second 32-bit identifier */
    can_ip->sFilterRegister[sFilterConfig->FilterNumber].FR2 = 
       ((0x0000FFFFU & (uint32_t)sFilterConfig->FilterMaskIdHigh) << 16U) |
        (0x0000FFFFU & (uint32_t)sFilterConfig->FilterMaskIdLow);
  }

  /* Filter Mode */
  if (sFilterConfig->FilterMode == CAN_FILTERMODE_IDMASK)
  {
    /*Id/Mask mode for the filter*/
    can_ip->FM1R &= ~(uint32_t)filternbrbitpos;
  }
  else /* CAN_FilterInitStruct->CAN_FilterMode == CAN_FilterMode_IdList */
  {
    /*Identifier list mode for the filter*/
    can_ip->FM1R |= (uint32_t)filternbrbitpos;
  }

  /* Filter FIFO assignment */
  if (sFilterConfig->FilterFIFOAssignment == CAN_FILTER_FIFO0)
  {
    /* FIFO 0 assignation for the filter */
    can_ip->FFA1R &= ~(uint32_t)filternbrbitpos;
  }

  if (sFilterConfig->FilterFIFOAssignment == CAN_FILTER_FIFO1)
  {
    /* FIFO 1 assignation for the filter */
    can_ip->FFA1R |= (uint32_t)filternbrbitpos;
  }
  
  /* Filter activation */
  if (sFilterConfig->FilterActivation == ENABLE)
  {
    can_ip->FA1R |= filternbrbitpos;
  }

  /* Leave the initialisation mode for the filter */
  can_ip->FMR &= ~((uint32_t)CAN_FMR_FINIT);
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Deinitializes the CANx peripheral registers to their default reset values. 
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_DeInit(CAN_HandleTypeDef* hcan)
{
  /* Check CAN handle */
  if(hcan == NULL)
  {
     return HAL_ERROR;
  }
  
  /* Check the parameters */
  assert_param(IS_CAN_ALL_INSTANCE(hcan->Instance));
  
  /* Change CAN state */
  hcan->State = HAL_CAN_STATE_BUSY;
  
  /* DeInit the low level hardware */
  HAL_CAN_MspDeInit(hcan);
  
  /* Change CAN state */
  hcan->State = HAL_CAN_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(hcan);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Initializes the CAN MSP.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @retval None
  */
__weak void HAL_CAN_MspInit(CAN_HandleTypeDef* hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_MspInit could be implemented in the user file
   */ 
}

/**
  * @brief  DeInitializes the CAN MSP.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @retval None
  */
__weak void HAL_CAN_MspDeInit(CAN_HandleTypeDef* hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_MspDeInit could be implemented in the user file
   */ 
}

/**
  * @}
  */

/** @defgroup CAN_Exported_Functions_Group2 IO operation functions
 *  @brief    IO operation functions 
 *
@verbatim   
  ==============================================================================
                      ##### IO operation functions #####
  ==============================================================================
    [..]  This section provides functions allowing to:
      (+) Transmit a CAN frame message.
      (+) Receive a CAN frame message.
      (+) Enter CAN peripheral in sleep mode. 
      (+) Wake up the CAN peripheral from sleep mode.
               
@endverbatim
  * @{
  */

/**
  * @brief  Initiates and transmits a CAN frame message.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @param  Timeout Specify Timeout value   
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_Transmit(CAN_HandleTypeDef* hcan, uint32_t Timeout)
{
  uint32_t transmitmailbox = CAN_TXSTATUS_NOMAILBOX;
  uint32_t tickstart = 0U;

  /* Check the parameters */
  assert_param(IS_CAN_IDTYPE(hcan->pTxMsg->IDE));
  assert_param(IS_CAN_RTR(hcan->pTxMsg->RTR));
  assert_param(IS_CAN_DLC(hcan->pTxMsg->DLC));

  if(((hcan->Instance->TSR&CAN_TSR_TME0) == CAN_TSR_TME0) || \
     ((hcan->Instance->TSR&CAN_TSR_TME1) == CAN_TSR_TME1) || \
     ((hcan->Instance->TSR&CAN_TSR_TME2) == CAN_TSR_TME2))
  {  
    /* Process locked */
    __HAL_LOCK(hcan);
  
    /* Change CAN state */
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_RX0):
          hcan->State = HAL_CAN_STATE_BUSY_TX_RX0;
          break;
      case(HAL_CAN_STATE_BUSY_RX1):
          hcan->State = HAL_CAN_STATE_BUSY_TX_RX1;
          break;
      case(HAL_CAN_STATE_BUSY_RX0_RX1):
          hcan->State = HAL_CAN_STATE_BUSY_TX_RX0_RX1;
          break;
      default: /* HAL_CAN_STATE_READY */
          hcan->State = HAL_CAN_STATE_BUSY_TX;
          break;
    }
  
    /* Select one empty transmit mailbox */
    if ((hcan->Instance->TSR&CAN_TSR_TME0) == CAN_TSR_TME0)
    {
      transmitmailbox = CAN_TXMAILBOX_0;
    }
    else if ((hcan->Instance->TSR&CAN_TSR_TME1) == CAN_TSR_TME1)
    {
      transmitmailbox = CAN_TXMAILBOX_1;
    }
    else
    {
      transmitmailbox = CAN_TXMAILBOX_2;
    }

    /* Set up the Id */
    hcan->Instance->sTxMailBox[transmitmailbox].TIR &= CAN_TI0R_TXRQ;
    if (hcan->pTxMsg->IDE == CAN_ID_STD)
    {
      assert_param(IS_CAN_STDID(hcan->pTxMsg->StdId));  
      hcan->Instance->sTxMailBox[transmitmailbox].TIR |= ((hcan->pTxMsg->StdId << 21U) | \
                                                  hcan->pTxMsg->RTR);
    }
    else
    {
      assert_param(IS_CAN_EXTID(hcan->pTxMsg->ExtId));
      hcan->Instance->sTxMailBox[transmitmailbox].TIR |= ((hcan->pTxMsg->ExtId << 3U) | \
                                                  hcan->pTxMsg->IDE | \
                                                  hcan->pTxMsg->RTR);
    }
    
    /* Set up the DLC */
    hcan->pTxMsg->DLC &= (uint8_t)0x0000000F;
    hcan->Instance->sTxMailBox[transmitmailbox].TDTR &= (uint32_t)0xFFFFFFF0U;
    hcan->Instance->sTxMailBox[transmitmailbox].TDTR |= hcan->pTxMsg->DLC;

    /* Set up the data field */
    hcan->Instance->sTxMailBox[transmitmailbox].TDLR = (((uint32_t)hcan->pTxMsg->Data[3U] << 24U) | 
                                             ((uint32_t)hcan->pTxMsg->Data[2U] << 16U) |
                                             ((uint32_t)hcan->pTxMsg->Data[1U] << 8U) | 
                                             ((uint32_t)hcan->pTxMsg->Data[0U]));
    hcan->Instance->sTxMailBox[transmitmailbox].TDHR = (((uint32_t)hcan->pTxMsg->Data[7U] << 24U) | 
                                             ((uint32_t)hcan->pTxMsg->Data[6U] << 16U) |
                                             ((uint32_t)hcan->pTxMsg->Data[5U] << 8U) |
                                             ((uint32_t)hcan->pTxMsg->Data[4U]));
    /* Request transmission */
    hcan->Instance->sTxMailBox[transmitmailbox].TIR |= CAN_TI0R_TXRQ;
  
    /* Get tick */
    tickstart = HAL_GetTick();
  
    /* Check End of transmission flag */
    while(!(__HAL_CAN_TRANSMIT_STATUS(hcan, transmitmailbox)))
    {
      /* Check for the Timeout */
      if(Timeout != HAL_MAX_DELAY)
      {
       if((Timeout == 0U)||((HAL_GetTick() - tickstart ) > Timeout))
       {
         hcan->State = HAL_CAN_STATE_TIMEOUT;

         __HAL_CAN_CANCEL_TRANSMIT(hcan, transmitmailbox);

         /* Process unlocked */
         __HAL_UNLOCK(hcan);
         return HAL_TIMEOUT;
        }
      }
    }

    /* Change CAN state */
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_TX_RX0):
          hcan->State = HAL_CAN_STATE_BUSY_RX0;
          break;
      case(HAL_CAN_STATE_BUSY_TX_RX1):
          hcan->State = HAL_CAN_STATE_BUSY_RX1;
          break;
      case(HAL_CAN_STATE_BUSY_TX_RX0_RX1):
          hcan->State = HAL_CAN_STATE_BUSY_RX0_RX1;
          break;
      default: /* HAL_CAN_STATE_BUSY_TX */
          hcan->State = HAL_CAN_STATE_READY;
          break;
    }

    /* Process unlocked */
    __HAL_UNLOCK(hcan);
    
    /* Return function status */
    return HAL_OK;
  }
  else
  {
    /* Change CAN state */
    hcan->State = HAL_CAN_STATE_ERROR; 

    /* Return function status */
    return HAL_ERROR;
  }
}

/**
  * @brief  Initiates and transmits a CAN frame message.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_Transmit_IT(CAN_HandleTypeDef* hcan)
{
  uint32_t  transmitmailbox = CAN_TXSTATUS_NOMAILBOX;
  
  /* Check the parameters */
  assert_param(IS_CAN_IDTYPE(hcan->pTxMsg->IDE));
  assert_param(IS_CAN_RTR(hcan->pTxMsg->RTR));
  assert_param(IS_CAN_DLC(hcan->pTxMsg->DLC));
  
  if(((hcan->Instance->TSR&CAN_TSR_TME0) == CAN_TSR_TME0) || \
     ((hcan->Instance->TSR&CAN_TSR_TME1) == CAN_TSR_TME1) || \
     ((hcan->Instance->TSR&CAN_TSR_TME2) == CAN_TSR_TME2))
  {
    /* Process Locked */
    __HAL_LOCK(hcan);

    /* Select one empty transmit mailbox */
    if((hcan->Instance->TSR&CAN_TSR_TME0) == CAN_TSR_TME0)
    {
      transmitmailbox = CAN_TXMAILBOX_0;
    }
    else if((hcan->Instance->TSR&CAN_TSR_TME1) == CAN_TSR_TME1)
    {
      transmitmailbox = CAN_TXMAILBOX_1;
    }
    else
    {
      transmitmailbox = CAN_TXMAILBOX_2;
    }

    /* Set up the Id */
    hcan->Instance->sTxMailBox[transmitmailbox].TIR &= CAN_TI0R_TXRQ;
    if(hcan->pTxMsg->IDE == CAN_ID_STD)
    {
      assert_param(IS_CAN_STDID(hcan->pTxMsg->StdId));  
      hcan->Instance->sTxMailBox[transmitmailbox].TIR |= ((hcan->pTxMsg->StdId << 21U) | \
                                                hcan->pTxMsg->RTR);
    }
    else
    {
      assert_param(IS_CAN_EXTID(hcan->pTxMsg->ExtId));
      hcan->Instance->sTxMailBox[transmitmailbox].TIR |= ((hcan->pTxMsg->ExtId << 3U) | \
                                                hcan->pTxMsg->IDE | \
                                                hcan->pTxMsg->RTR);
    }
    
    /* Set up the DLC */
    hcan->pTxMsg->DLC &= (uint8_t)0x0000000F;
    hcan->Instance->sTxMailBox[transmitmailbox].TDTR &= (uint32_t)0xFFFFFFF0U;
    hcan->Instance->sTxMailBox[transmitmailbox].TDTR |= hcan->pTxMsg->DLC;

    /* Set up the data field */
    hcan->Instance->sTxMailBox[transmitmailbox].TDLR = (((uint32_t)hcan->pTxMsg->Data[3U] << 24U) |
                                           ((uint32_t)hcan->pTxMsg->Data[2U] << 16U) |
                                           ((uint32_t)hcan->pTxMsg->Data[1U] << 8U) | 
                                           ((uint32_t)hcan->pTxMsg->Data[0U]));
    hcan->Instance->sTxMailBox[transmitmailbox].TDHR = (((uint32_t)hcan->pTxMsg->Data[7U] << 24U) |
                                           ((uint32_t)hcan->pTxMsg->Data[6U] << 16U) |
                                           ((uint32_t)hcan->pTxMsg->Data[5U] << 8U) |
                                           ((uint32_t)hcan->pTxMsg->Data[4U]));

    /* Change CAN state */
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_RX0):
          hcan->State = HAL_CAN_STATE_BUSY_TX_RX0;
          break;
      case(HAL_CAN_STATE_BUSY_RX1):
          hcan->State = HAL_CAN_STATE_BUSY_TX_RX1;
          break;
      case(HAL_CAN_STATE_BUSY_RX0_RX1):
          hcan->State = HAL_CAN_STATE_BUSY_TX_RX0_RX1;
          break;
      default: /* HAL_CAN_STATE_READY */
          hcan->State = HAL_CAN_STATE_BUSY_TX;
          break;
    }

    /* Set CAN error code to none */
    hcan->ErrorCode = HAL_CAN_ERROR_NONE;

    /* Process Unlocked */
    __HAL_UNLOCK(hcan);

    /* Request transmission */
    hcan->Instance->sTxMailBox[transmitmailbox].TIR |= CAN_TI0R_TXRQ;

    /* Enable Error warning, Error passive, Bus-off,
       Last error and Error Interrupts */
    __HAL_CAN_ENABLE_IT(hcan, CAN_IT_EWG |
                              CAN_IT_EPV |
                              CAN_IT_BOF |
                              CAN_IT_LEC |
                              CAN_IT_ERR |
                              CAN_IT_TME);
  }
  else
  {
    /* Change CAN state */
    hcan->State = HAL_CAN_STATE_ERROR; 

    /* Return function status */
    return HAL_ERROR;
  }
  
  return HAL_OK;
}

/**
  * @brief  Receives a correct CAN frame.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @param  FIFONumber FIFO Number value
  * @param  Timeout Specify Timeout value 
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_Receive(CAN_HandleTypeDef* hcan, uint8_t FIFONumber, uint32_t Timeout)
{
  uint32_t tickstart = 0U;
  CanRxMsgTypeDef* pRxMsg = NULL;
 
  /* Check the parameters */
  assert_param(IS_CAN_FIFO(FIFONumber));

  /* Check if CAN state is not busy for RX FIFO0 */
  if ((FIFONumber == CAN_FIFO0) && ((hcan->State == HAL_CAN_STATE_BUSY_RX0) ||         \
                                    (hcan->State == HAL_CAN_STATE_BUSY_TX_RX0) ||      \
                                    (hcan->State == HAL_CAN_STATE_BUSY_RX0_RX1) ||     \
                                    (hcan->State == HAL_CAN_STATE_BUSY_TX_RX0_RX1)))
  {
    return HAL_BUSY;
  }

  /* Check if CAN state is not busy for RX FIFO1 */
  if ((FIFONumber == CAN_FIFO1) && ((hcan->State == HAL_CAN_STATE_BUSY_RX1) ||         \
                                    (hcan->State == HAL_CAN_STATE_BUSY_TX_RX1) ||      \
                                    (hcan->State == HAL_CAN_STATE_BUSY_RX0_RX1) ||     \
                                    (hcan->State == HAL_CAN_STATE_BUSY_TX_RX0_RX1)))
  {
    return HAL_BUSY;
  }

  /* Process locked */
  __HAL_LOCK(hcan);

  /* Change CAN state */
  if (FIFONumber == CAN_FIFO0)
  {
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_TX):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX0;
        break;
      case(HAL_CAN_STATE_BUSY_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_RX0_RX1;
        break;
      case(HAL_CAN_STATE_BUSY_TX_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX0_RX1;
        break;
      default: /* HAL_CAN_STATE_READY */
        hcan->State = HAL_CAN_STATE_BUSY_RX0;
        break;
    }
  }
  else /* FIFONumber == CAN_FIFO1 */
  {
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_TX):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX1;
        break;
      case(HAL_CAN_STATE_BUSY_RX0):
        hcan->State = HAL_CAN_STATE_BUSY_RX0_RX1;
        break;
      case(HAL_CAN_STATE_BUSY_TX_RX0):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX0_RX1;
        break;
      default: /* HAL_CAN_STATE_READY */
        hcan->State = HAL_CAN_STATE_BUSY_RX1;
        break;
    }
  }

  /* Get tick */ 
  tickstart = HAL_GetTick();
  
  /* Check pending message */
  while(__HAL_CAN_MSG_PENDING(hcan, FIFONumber) == 0U)
  {
    /* Check for the Timeout */
    if(Timeout != HAL_MAX_DELAY)
    {
      if((Timeout == 0U)||((HAL_GetTick() - tickstart ) > Timeout))
      {
        hcan->State = HAL_CAN_STATE_TIMEOUT;
        /* Process unlocked */
        __HAL_UNLOCK(hcan);
        return HAL_TIMEOUT;
      }
    }
  }

  /* Set RxMsg pointer */
  if(FIFONumber == CAN_FIFO0)
  {
    pRxMsg = hcan->pRxMsg;
  }
  else /* FIFONumber == CAN_FIFO1 */
  {
    pRxMsg = hcan->pRx1Msg;
  }

  /* Get the Id */
  pRxMsg->IDE = (uint8_t)0x04 & hcan->Instance->sFIFOMailBox[FIFONumber].RIR;
  if (pRxMsg->IDE == CAN_ID_STD)
  {
    pRxMsg->StdId = 0x000007FFU & (hcan->Instance->sFIFOMailBox[FIFONumber].RIR >> 21U);
  }
  else
  {
    pRxMsg->ExtId = 0x1FFFFFFFU & (hcan->Instance->sFIFOMailBox[FIFONumber].RIR >> 3U);
  }
  
  pRxMsg->RTR = (uint8_t)0x02 & hcan->Instance->sFIFOMailBox[FIFONumber].RIR;
  /* Get the DLC */
  pRxMsg->DLC = (uint8_t)0x0F & hcan->Instance->sFIFOMailBox[FIFONumber].RDTR;
  /* Get the FMI */
  pRxMsg->FMI = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDTR >> 8U);
  /* Get the FIFONumber */
  pRxMsg->FIFONumber = FIFONumber;
  /* Get the data field */
  pRxMsg->Data[0] = (uint8_t)0xFF & hcan->Instance->sFIFOMailBox[FIFONumber].RDLR;
  pRxMsg->Data[1] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDLR >> 8U);
  pRxMsg->Data[2] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDLR >> 16U);
  pRxMsg->Data[3] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDLR >> 24U);
  pRxMsg->Data[4] = (uint8_t)0xFF & hcan->Instance->sFIFOMailBox[FIFONumber].RDHR;
  pRxMsg->Data[5] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDHR >> 8U);
  pRxMsg->Data[6] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDHR >> 16U);
  pRxMsg->Data[7] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDHR >> 24U);

  /* Release the FIFO */
  if(FIFONumber == CAN_FIFO0)
  {
    /* Release FIFO0 */
    __HAL_CAN_FIFO_RELEASE(hcan, CAN_FIFO0);
  }
  else /* FIFONumber == CAN_FIFO1 */
  {
    /* Release FIFO1 */
    __HAL_CAN_FIFO_RELEASE(hcan, CAN_FIFO1);
  }

  /* Change CAN state */
  if (FIFONumber == CAN_FIFO0)
  {
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_TX_RX0):
        hcan->State = HAL_CAN_STATE_BUSY_TX;
        break;
      case(HAL_CAN_STATE_BUSY_RX0_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_RX1;
        break;
      case(HAL_CAN_STATE_BUSY_TX_RX0_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX1;
        break;
      default: /* HAL_CAN_STATE_BUSY_RX0 */
        hcan->State = HAL_CAN_STATE_READY;
        break;
    }
  }
  else /* FIFONumber == CAN_FIFO1 */
  {
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_TX_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_TX;
        break;
      case(HAL_CAN_STATE_BUSY_RX0_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_RX0;
        break;
      case(HAL_CAN_STATE_BUSY_TX_RX0_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX0;
        break;
      default: /* HAL_CAN_STATE_BUSY_RX1 */
        hcan->State = HAL_CAN_STATE_READY;
        break;
    }
  }

  /* Process unlocked */
  __HAL_UNLOCK(hcan);
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Receives a correct CAN frame.
  * @param  hcan       Pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @param  FIFONumber Specify the FIFO number    
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_CAN_Receive_IT(CAN_HandleTypeDef* hcan, uint8_t FIFONumber)
{
  /* Check the parameters */
  assert_param(IS_CAN_FIFO(FIFONumber));
  
  /* Check if CAN state is not busy for RX FIFO0 */
  if((FIFONumber == CAN_FIFO0) && ((hcan->State == HAL_CAN_STATE_BUSY_RX0) ||         \
                                   (hcan->State == HAL_CAN_STATE_BUSY_TX_RX0) ||      \
                                   (hcan->State == HAL_CAN_STATE_BUSY_RX0_RX1) ||     \
                                   (hcan->State == HAL_CAN_STATE_BUSY_TX_RX0_RX1)))
  {
    return HAL_BUSY;
  }

  /* Check if CAN state is not busy for RX FIFO1 */
  if((FIFONumber == CAN_FIFO1) && ((hcan->State == HAL_CAN_STATE_BUSY_RX1) ||         \
                                   (hcan->State == HAL_CAN_STATE_BUSY_TX_RX1) ||      \
                                   (hcan->State == HAL_CAN_STATE_BUSY_RX0_RX1) ||     \
                                   (hcan->State == HAL_CAN_STATE_BUSY_TX_RX0_RX1)))
  {
    return HAL_BUSY;
  }

  /* Process locked */
  __HAL_LOCK(hcan);

  /* Change CAN state */
  if(FIFONumber == CAN_FIFO0)
  {
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_TX):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX0;
        break;
      case(HAL_CAN_STATE_BUSY_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_RX0_RX1;
        break;
      case(HAL_CAN_STATE_BUSY_TX_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX0_RX1;
        break;
      default: /* HAL_CAN_STATE_READY */
        hcan->State = HAL_CAN_STATE_BUSY_RX0;
        break;
    }
  }
  else /* FIFONumber == CAN_FIFO1 */
  {
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_TX):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX1;
        break;
      case(HAL_CAN_STATE_BUSY_RX0):
        hcan->State = HAL_CAN_STATE_BUSY_RX0_RX1;
        break;
      case(HAL_CAN_STATE_BUSY_TX_RX0):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX0_RX1;
        break;
      default: /* HAL_CAN_STATE_READY */
        hcan->State = HAL_CAN_STATE_BUSY_RX1;
        break;
    }
  }
  /* Set CAN error code to none */
  hcan->ErrorCode = HAL_CAN_ERROR_NONE;

  /* Enable interrupts: */
  /*  - Enable Error warning Interrupt */
  /*  - Enable Error passive Interrupt */
  /*  - Enable Bus-off Interrupt */
  /*  - Enable Last error code Interrupt */
  /*  - Enable Error Interrupt */
  /*  - Enable Transmit mailbox empty Interrupt */
  __HAL_CAN_ENABLE_IT(hcan, CAN_IT_EWG |
                            CAN_IT_EPV |
                            CAN_IT_BOF |
                            CAN_IT_LEC |
                            CAN_IT_ERR |
                            CAN_IT_TME);

  /* Process unlocked */
   __HAL_UNLOCK(hcan);

  if(FIFONumber == CAN_FIFO0)
  {
    /* Enable FIFO 0 overrun and message pending Interrupt */
    __HAL_CAN_ENABLE_IT(hcan, CAN_IT_FOV0 | CAN_IT_FMP0);
  }
  else
  {
    /* Enable FIFO 1 overrun and message pending Interrupt */
    __HAL_CAN_ENABLE_IT(hcan, CAN_IT_FOV1 | CAN_IT_FMP1);
  }

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Enters the Sleep (low power) mode.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL status.
  */
HAL_StatusTypeDef HAL_CAN_Sleep(CAN_HandleTypeDef* hcan)
{
  uint32_t tickstart = 0U;
   
  /* Process locked */
  __HAL_LOCK(hcan);
  
  /* Change CAN state */
  hcan->State = HAL_CAN_STATE_BUSY; 
    
  /* Request Sleep mode */
   hcan->Instance->MCR = (((hcan->Instance->MCR) & (uint32_t)(~(uint32_t)CAN_MCR_INRQ)) | CAN_MCR_SLEEP);

  /* Sleep mode status */
  if ((hcan->Instance->MSR & (CAN_MSR_SLAK|CAN_MSR_INAK)) != CAN_MSR_SLAK)
  {
    /* Process unlocked */
    __HAL_UNLOCK(hcan);

    /* Return function status */
    return HAL_ERROR;
  }

  /* Get tick */
  tickstart = HAL_GetTick();
  
  /* Wait the acknowledge */
  while((hcan->Instance->MSR & (CAN_MSR_SLAK|CAN_MSR_INAK)) != CAN_MSR_SLAK)
  {
    if((HAL_GetTick()  - tickstart) > CAN_TIMEOUT_VALUE)
    {
      hcan->State = HAL_CAN_STATE_TIMEOUT;
      /* Process unlocked */
      __HAL_UNLOCK(hcan);
      return HAL_TIMEOUT;
    }
  }

  /* Change CAN state */
  hcan->State = HAL_CAN_STATE_READY;

  /* Process unlocked */
  __HAL_UNLOCK(hcan);

  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Wakes up the CAN peripheral from sleep mode, after that the CAN peripheral
  *         is in the normal mode.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL status.
  */
HAL_StatusTypeDef HAL_CAN_WakeUp(CAN_HandleTypeDef* hcan)
{
  uint32_t tickstart = 0U;
    
  /* Process locked */
  __HAL_LOCK(hcan);
  
  /* Change CAN state */
  hcan->State = HAL_CAN_STATE_BUSY;  
 
  /* Wake up request */
  hcan->Instance->MCR &= ~(uint32_t)CAN_MCR_SLEEP;

  /* Get tick */ 
  tickstart = HAL_GetTick();

  /* Sleep mode status */
  while((hcan->Instance->MSR & CAN_MSR_SLAK) == CAN_MSR_SLAK)
  {
    if((HAL_GetTick()  - tickstart) > CAN_TIMEOUT_VALUE)
    {
      hcan->State= HAL_CAN_STATE_TIMEOUT;
      /* Process unlocked */
      __HAL_UNLOCK(hcan);
      return HAL_TIMEOUT;
    }
  }
  if((hcan->Instance->MSR & CAN_MSR_SLAK) == CAN_MSR_SLAK)
  {
    /* Process unlocked */
    __HAL_UNLOCK(hcan);
 
    /* Return function status */
    return HAL_ERROR;
  }
  
  /* Change CAN state */
  hcan->State = HAL_CAN_STATE_READY; 
  
  /* Process unlocked */
  __HAL_UNLOCK(hcan);
  
  /* Return function status */
  return HAL_OK;
}

/**
  * @brief  Handles CAN interrupt request  
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
void HAL_CAN_IRQHandler(CAN_HandleTypeDef* hcan)
{
  uint32_t tmp1 = 0U, tmp2 = 0U, tmp3 = 0U;
  uint32_t errorcode = HAL_CAN_ERROR_NONE;

  /* Check Overrun flag for FIFO0 */
  tmp1 = __HAL_CAN_GET_FLAG(hcan, CAN_FLAG_FOV0);
  tmp2 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_FOV0);
  if(tmp1 && tmp2)
  {
    /* Set CAN error code to FOV0 error */
    errorcode |= HAL_CAN_ERROR_FOV0;

    /* Clear FIFO0 Overrun Flag */
    __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_FOV0);
  }
  /* Check Overrun flag for FIFO1 */
  tmp1 = __HAL_CAN_GET_FLAG(hcan, CAN_FLAG_FOV1);
  tmp2 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_FOV1);

  if(tmp1 && tmp2)
  {
    /* Set CAN error code to FOV1 error */
    errorcode |= HAL_CAN_ERROR_FOV1;

    /* Clear FIFO1 Overrun Flag */
    __HAL_CAN_CLEAR_FLAG(hcan, CAN_FLAG_FOV1);
  }

  /* Check End of transmission flag */
  if(__HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_TME))
  {
    tmp1 = __HAL_CAN_TRANSMIT_STATUS(hcan, CAN_TXMAILBOX_0);
    tmp2 = __HAL_CAN_TRANSMIT_STATUS(hcan, CAN_TXMAILBOX_1);
    tmp3 = __HAL_CAN_TRANSMIT_STATUS(hcan, CAN_TXMAILBOX_2);
    if(tmp1 || tmp2 || tmp3)  
    {
      tmp1 = __HAL_CAN_GET_FLAG(hcan, CAN_FLAG_TXOK0);
      tmp2 = __HAL_CAN_GET_FLAG(hcan, CAN_FLAG_TXOK1);
      tmp3 = __HAL_CAN_GET_FLAG(hcan, CAN_FLAG_TXOK2);
      /* Check Transmit success */
      if(tmp1 || tmp2 || tmp3)
      {
        /* Call transmit function */
        CAN_Transmit_IT(hcan);
      }
      else /* Transmit failure */
      {
        /* Set CAN error code to TXFAIL error */
        errorcode |= HAL_CAN_ERROR_TXFAIL;
      }

      /* Clear transmission status flags (RQCPx and TXOKx) */
      SET_BIT(hcan->Instance->TSR, CAN_TSR_RQCP0  | CAN_TSR_RQCP1  | CAN_TSR_RQCP2 | \
                                   CAN_FLAG_TXOK0 | CAN_FLAG_TXOK1 | CAN_FLAG_TXOK2);
    }
  }

  tmp1 = __HAL_CAN_MSG_PENDING(hcan, CAN_FIFO0);
  tmp2 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_FMP0);
  /* Check End of reception flag for FIFO0 */
  if((tmp1 != 0U) && tmp2)
  {
    /* Call receive function */
    CAN_Receive_IT(hcan, CAN_FIFO0);
  }

  tmp1 = __HAL_CAN_MSG_PENDING(hcan, CAN_FIFO1);
  tmp2 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_FMP1);
  /* Check End of reception flag for FIFO1 */
  if((tmp1 != 0U) && tmp2)
  {
    /* Call receive function */
    CAN_Receive_IT(hcan, CAN_FIFO1);
  }

  /* Set error code in handle */
  hcan->ErrorCode |= errorcode;

  tmp1 = __HAL_CAN_GET_FLAG(hcan, CAN_FLAG_EWG);
  tmp2 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_EWG);
  tmp3 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_ERR);
  /* Check Error Warning Flag */
  if(tmp1 && tmp2 && tmp3)
  {
    /* Set CAN error code to EWG error */
    hcan->ErrorCode |= HAL_CAN_ERROR_EWG;
  }
  
  tmp1 = __HAL_CAN_GET_FLAG(hcan, CAN_FLAG_EPV);
  tmp2 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_EPV);
  tmp3 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_ERR); 
  /* Check Error Passive Flag */
  if(tmp1 && tmp2 && tmp3)
  {
    /* Set CAN error code to EPV error */
    hcan->ErrorCode |= HAL_CAN_ERROR_EPV;
  }
  
  tmp1 = __HAL_CAN_GET_FLAG(hcan, CAN_FLAG_BOF);
  tmp2 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_BOF);
  tmp3 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_ERR);  
  /* Check Bus-Off Flag */
  if(tmp1 && tmp2 && tmp3)
  {
    /* Set CAN error code to BOF error */
    hcan->ErrorCode |= HAL_CAN_ERROR_BOF;
  }
  
  tmp1 = HAL_IS_BIT_CLR(hcan->Instance->ESR, CAN_ESR_LEC);
  tmp2 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_LEC);
  tmp3 = __HAL_CAN_GET_IT_SOURCE(hcan, CAN_IT_ERR);
  /* Check Last error code Flag */
  if((!tmp1) && tmp2 && tmp3)
  {
    tmp1 = (hcan->Instance->ESR) & CAN_ESR_LEC;
    switch(tmp1)
    {
      case(CAN_ESR_LEC_0):
          /* Set CAN error code to STF error */
          hcan->ErrorCode |= HAL_CAN_ERROR_STF;
          break;
      case(CAN_ESR_LEC_1):
          /* Set CAN error code to FOR error */
          hcan->ErrorCode |= HAL_CAN_ERROR_FOR;
          break;
      case(CAN_ESR_LEC_1 | CAN_ESR_LEC_0):
          /* Set CAN error code to ACK error */
          hcan->ErrorCode |= HAL_CAN_ERROR_ACK;
          break;
      case(CAN_ESR_LEC_2):
          /* Set CAN error code to BR error */
          hcan->ErrorCode |= HAL_CAN_ERROR_BR;
          break;
      case(CAN_ESR_LEC_2 | CAN_ESR_LEC_0):
          /* Set CAN error code to BD error */
          hcan->ErrorCode |= HAL_CAN_ERROR_BD;
          break;
      case(CAN_ESR_LEC_2 | CAN_ESR_LEC_1):
          /* Set CAN error code to CRC error */
          hcan->ErrorCode |= HAL_CAN_ERROR_CRC;
          break;
      default:
          break;
    }

    /* Clear Last error code Flag */ 
    hcan->Instance->ESR &= ~(CAN_ESR_LEC);
  }
  
  /* Call the Error call Back in case of Errors */
  if(hcan->ErrorCode != HAL_CAN_ERROR_NONE)
  {
    /* Clear ERRI Flag */ 
    hcan->Instance->MSR = CAN_MSR_ERRI; 
    /* Set the CAN state ready to be able to start again the process */
    hcan->State = HAL_CAN_STATE_READY;

    /* Disable interrupts: */
    /*  - Disable Error warning Interrupt */
    /*  - Disable Error passive Interrupt */
    /*  - Disable Bus-off Interrupt */
    /*  - Disable Last error code Interrupt */
    /*  - Disable Error Interrupt */
    /*  - Disable FIFO 0 message pending Interrupt */
    /*  - Disable FIFO 0 Overrun Interrupt */
    /*  - Disable FIFO 1 message pending Interrupt */
    /*  - Disable FIFO 1 Overrun Interrupt */
    /*  - Disable Transmit mailbox empty Interrupt */
    __HAL_CAN_DISABLE_IT(hcan, CAN_IT_EWG |
                               CAN_IT_EPV |
                               CAN_IT_BOF |
                               CAN_IT_LEC |
                               CAN_IT_ERR |
                               CAN_IT_FMP0|
                               CAN_IT_FOV0|
                               CAN_IT_FMP1|
                               CAN_IT_FOV1|
                               CAN_IT_TME);

    /* Call Error callback function */
    HAL_CAN_ErrorCallback(hcan);
  }  
}

/**
  * @brief  Transmission  complete callback in non blocking mode 
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_TxCpltCallback(CAN_HandleTypeDef* hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_TxCpltCallback could be implemented in the user file
   */
}

/**
  * @brief  Transmission  complete callback in non blocking mode 
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_RxCpltCallback(CAN_HandleTypeDef* hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_RxCpltCallback could be implemented in the user file
   */
}

/**
  * @brief  Error CAN callback.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval None
  */
__weak void HAL_CAN_ErrorCallback(CAN_HandleTypeDef *hcan)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hcan);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_CAN_ErrorCallback could be implemented in the user file
   */
}

/**
  * @}
  */

/** @defgroup CAN_Exported_Functions_Group3 Peripheral State and Error functions
  *  @brief   CAN Peripheral State functions 
  *
@verbatim   
  ==============================================================================
            ##### Peripheral State and Error functions #####
  ==============================================================================
    [..]
    This subsection provides functions allowing to :
      (+) Check the CAN state.
      (+) Check CAN Errors detected during interrupt process
         
@endverbatim
  * @{
  */

/**
  * @brief  return the CAN state
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval HAL state
  */
HAL_CAN_StateTypeDef HAL_CAN_GetState(CAN_HandleTypeDef* hcan)
{
  /* Return CAN state */
  return hcan->State;
}

/**
  * @brief  Return the CAN error code
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.
  * @retval CAN Error Code
  */
uint32_t HAL_CAN_GetError(CAN_HandleTypeDef *hcan)
{
  return hcan->ErrorCode;
}

/**
  * @}
  */
/**
  * @brief  Initiates and transmits a CAN frame message.
  * @param  hcan pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @retval HAL status
  */
static HAL_StatusTypeDef CAN_Transmit_IT(CAN_HandleTypeDef* hcan)
{
  /* Disable Transmit mailbox empty Interrupt */
  __HAL_CAN_DISABLE_IT(hcan, CAN_IT_TME);
  
  if(hcan->State == HAL_CAN_STATE_BUSY_TX)
  {   
    /* Disable Error warning, Error passive, Bus-off, Last error code
       and Error Interrupts */
    __HAL_CAN_DISABLE_IT(hcan, CAN_IT_EWG |
                               CAN_IT_EPV |
                               CAN_IT_BOF |
                               CAN_IT_LEC |
                               CAN_IT_ERR );
  }

  /* Change CAN state */
  switch(hcan->State)
  {
    case(HAL_CAN_STATE_BUSY_TX_RX0):
      hcan->State = HAL_CAN_STATE_BUSY_RX0;
      break;
    case(HAL_CAN_STATE_BUSY_TX_RX1):
      hcan->State = HAL_CAN_STATE_BUSY_RX1;
      break;
    case(HAL_CAN_STATE_BUSY_TX_RX0_RX1):
      hcan->State = HAL_CAN_STATE_BUSY_RX0_RX1;
      break;
    default: /* HAL_CAN_STATE_BUSY_TX */
      hcan->State = HAL_CAN_STATE_READY;
      break;
  }

  /* Transmission complete callback */ 
  HAL_CAN_TxCpltCallback(hcan);
  
  return HAL_OK;
}

/**
  * @brief  Receives a correct CAN frame.
  * @param  hcan       Pointer to a CAN_HandleTypeDef structure that contains
  *         the configuration information for the specified CAN.  
  * @param  FIFONumber Specify the FIFO number    
  * @retval HAL status
  * @retval None
  */
static HAL_StatusTypeDef CAN_Receive_IT(CAN_HandleTypeDef* hcan, uint8_t FIFONumber)
{
  uint32_t tmp1 = 0U;
  CanRxMsgTypeDef* pRxMsg = NULL;

  /* Set RxMsg pointer */
  if(FIFONumber == CAN_FIFO0)
  {
    pRxMsg = hcan->pRxMsg;
  }
  else /* FIFONumber == CAN_FIFO1 */
  {
    pRxMsg = hcan->pRx1Msg;
  }

  /* Get the Id */
  pRxMsg->IDE = (uint8_t)0x04 & hcan->Instance->sFIFOMailBox[FIFONumber].RIR;
  if (pRxMsg->IDE == CAN_ID_STD)
  {
    pRxMsg->StdId = 0x000007FFU & (hcan->Instance->sFIFOMailBox[FIFONumber].RIR >> 21U);
  }
  else
  {
    pRxMsg->ExtId = 0x1FFFFFFFU & (hcan->Instance->sFIFOMailBox[FIFONumber].RIR >> 3U);
  }
  
  pRxMsg->RTR = (uint8_t)0x02 & hcan->Instance->sFIFOMailBox[FIFONumber].RIR;
  /* Get the DLC */
  pRxMsg->DLC = (uint8_t)0x0F & hcan->Instance->sFIFOMailBox[FIFONumber].RDTR;
  /* Get the FIFONumber */
  pRxMsg->FIFONumber = FIFONumber;
  /* Get the FMI */
  pRxMsg->FMI = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDTR >> 8U);
  /* Get the data field */
  pRxMsg->Data[0] = (uint8_t)0xFF & hcan->Instance->sFIFOMailBox[FIFONumber].RDLR;
  pRxMsg->Data[1] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDLR >> 8U);
  pRxMsg->Data[2] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDLR >> 16U);
  pRxMsg->Data[3] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDLR >> 24U);
  pRxMsg->Data[4] = (uint8_t)0xFF & hcan->Instance->sFIFOMailBox[FIFONumber].RDHR;
  pRxMsg->Data[5] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDHR >> 8U);
  pRxMsg->Data[6] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDHR >> 16U);
  pRxMsg->Data[7] = (uint8_t)0xFF & (hcan->Instance->sFIFOMailBox[FIFONumber].RDHR >> 24U);
  /* Release the FIFO */
  /* Release FIFO0 */
  if (FIFONumber == CAN_FIFO0)
  {
    __HAL_CAN_FIFO_RELEASE(hcan, CAN_FIFO0);

    /* Disable FIFO 0 overrun and message pending Interrupt */
    __HAL_CAN_DISABLE_IT(hcan, CAN_IT_FOV0 | CAN_IT_FMP0);
  }
  /* Release FIFO1 */
  else /* FIFONumber == CAN_FIFO1 */
  {
    __HAL_CAN_FIFO_RELEASE(hcan, CAN_FIFO1);

    /* Disable FIFO 1 overrun and message pending Interrupt */
    __HAL_CAN_DISABLE_IT(hcan, CAN_IT_FOV1 | CAN_IT_FMP1);
  }

  tmp1 = hcan->State;
  if((tmp1 == HAL_CAN_STATE_BUSY_RX0) || (tmp1 == HAL_CAN_STATE_BUSY_RX1))
  {   
    /* Disable Error warning, Error passive, Bus-off, Last error code
       and Error Interrupts */
    __HAL_CAN_DISABLE_IT(hcan, CAN_IT_EWG |
                               CAN_IT_EPV |
                               CAN_IT_BOF |
                               CAN_IT_LEC |
                               CAN_IT_ERR);
  }

  /* Change CAN state */
  if (FIFONumber == CAN_FIFO0)
  {
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_TX_RX0):
        hcan->State = HAL_CAN_STATE_BUSY_TX;
        break;
      case(HAL_CAN_STATE_BUSY_RX0_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_RX1;
        break;
      case(HAL_CAN_STATE_BUSY_TX_RX0_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX1;
        break;
      default: /* HAL_CAN_STATE_BUSY_RX0 */
        hcan->State = HAL_CAN_STATE_READY;
        break;
    }
  }
  else /* FIFONumber == CAN_FIFO1 */
  {
    switch(hcan->State)
    {
      case(HAL_CAN_STATE_BUSY_TX_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_TX;
        break;
      case(HAL_CAN_STATE_BUSY_RX0_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_RX0;
        break;
      case(HAL_CAN_STATE_BUSY_TX_RX0_RX1):
        hcan->State = HAL_CAN_STATE_BUSY_TX_RX0;
        break;
      default: /* HAL_CAN_STATE_BUSY_RX1 */
        hcan->State = HAL_CAN_STATE_READY;
        break;
    }
  }

  /* Receive complete callback */ 
  HAL_CAN_RxCpltCallback(hcan);

  /* Return function status */
  return HAL_OK;
}

/**
  * @}
  */
#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx || STM32F427xx || STM32F437xx ||\
          STM32F429xx || STM32F439xx || STM32F446xx || STM32F469xx || STM32F479xx || STM32F412Zx ||\
          STM32F412Vx || STM32F412Rx || STM32F412Cx || STM32F413xx || STM32F423xx */

#endif /* HAL_CAN_LEGACY_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
