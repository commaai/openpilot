/**
  ******************************************************************************
  * @file    stm32f4xx_ll_sdmmc.c
  * @author  MCD Application Team
  * @brief   SDMMC Low Layer HAL module driver.
  *    
  *          This file provides firmware functions to manage the following 
  *          functionalities of the SDMMC peripheral:
  *           + Initialization/de-initialization functions
  *           + I/O operation functions
  *           + Peripheral Control functions 
  *           + Peripheral State functions
  *         
  @verbatim
  ==============================================================================
                       ##### SDMMC peripheral features #####
  ==============================================================================        
    [..] The SD/SDMMC MMC card host interface (SDMMC) provides an interface between the AHB
         peripheral bus and MultiMedia cards (MMCs), SD memory cards, SDMMC cards and CE-ATA
         devices.
    
    [..] The SDMMC features include the following:
         (+) Full compliance with MultiMedia Card System Specification Version 4.2. Card support
             for three different databus modes: 1-bit (default), 4-bit and 8-bit
         (+) Full compatibility with previous versions of MultiMedia Cards (forward compatibility)
         (+) Full compliance with SD Memory Card Specifications Version 2.0
         (+) Full compliance with SD I/O Card Specification Version 2.0: card support for two
             different data bus modes: 1-bit (default) and 4-bit
         (+) Full support of the CE-ATA features (full compliance with CE-ATA digital protocol
             Rev1.1)
         (+) Data transfer up to 48 MHz for the 8 bit mode
         (+) Data and command output enable signals to control external bidirectional drivers
   
                           ##### How to use this driver #####
  ==============================================================================
    [..]
      This driver is a considered as a driver of service for external devices drivers 
      that interfaces with the SDMMC peripheral.
      According to the device used (SD card/ MMC card / SDMMC card ...), a set of APIs 
      is used in the device's driver to perform SDMMC operations and functionalities.
   
      This driver is almost transparent for the final user, it is only used to implement other
      functionalities of the external device.
   
    [..]
      (+) The SDMMC clock (SDMMCCLK = 48 MHz) is coming from a specific output (MSI, PLLUSB1CLK,
          PLLUSB2CLK). Before start working with SDMMC peripheral make sure that the
          PLL is well configured.
          The SDMMC peripheral uses two clock signals:
          (++) SDMMC adapter clock (SDMMCCLK = 48 MHz)
          (++) APB2 bus clock (PCLK2)
       
          -@@- PCLK2 and SDMMC_CK clock frequencies must respect the following condition:
               Frequency(PCLK2) >= (3 / 8 x Frequency(SDMMC_CK))
  
      (+) Enable/Disable peripheral clock using RCC peripheral macros related to SDMMC
          peripheral.

      (+) Enable the Power ON State using the SDIO_PowerState_ON() 
          function and disable it using the function SDIO_PowerState_OFF().
                
      (+) Enable/Disable the clock using the __SDIO_ENABLE()/__SDIO_DISABLE() macros.
  
      (+) Enable/Disable the peripheral interrupts using the macros __SDIO_ENABLE_IT() 
          and __SDIO_DISABLE_IT() if you need to use interrupt mode. 
  
      (+) When using the DMA mode 
          (++) Configure the DMA in the MSP layer of the external device
          (++) Active the needed channel Request 
          (++) Enable the DMA using __SDIO_DMA_ENABLE() macro or Disable it using the macro
               __SDIO_DMA_DISABLE().
  
      (+) To control the CPSM (Command Path State Machine) and send 
          commands to the card use the SDIO_SendCommand(), 
          SDIO_GetCommandResponse() and SDIO_GetResponse() functions. First, user has
          to fill the command structure (pointer to SDIO_CmdInitTypeDef) according 
          to the selected command to be sent.
          The parameters that should be filled are:
           (++) Command Argument
           (++) Command Index
           (++) Command Response type
           (++) Command Wait
           (++) CPSM Status (Enable or Disable).
  
          -@@- To check if the command is well received, read the SDIO_CMDRESP
              register using the SDIO_GetCommandResponse().
              The SDMMC responses registers (SDIO_RESP1 to SDIO_RESP2), use the
              SDIO_GetResponse() function.
  
      (+) To control the DPSM (Data Path State Machine) and send/receive 
           data to/from the card use the SDIO_DataConfig(), SDIO_GetDataCounter(), 
          SDIO_ReadFIFO(), SDIO_WriteFIFO() and SDIO_GetFIFOCount() functions.
  
    *** Read Operations ***
    =======================
    [..]
      (#) First, user has to fill the data structure (pointer to
          SDIO_DataInitTypeDef) according to the selected data type to be received.
          The parameters that should be filled are:
           (++) Data TimeOut
           (++) Data Length
           (++) Data Block size
           (++) Data Transfer direction: should be from card (To SDMMC)
           (++) Data Transfer mode
           (++) DPSM Status (Enable or Disable)
                                     
      (#) Configure the SDMMC resources to receive the data from the card
          according to selected transfer mode (Refer to Step 8, 9 and 10).
  
      (#) Send the selected Read command (refer to step 11).
                    
      (#) Use the SDIO flags/interrupts to check the transfer status.
  
    *** Write Operations ***
    ========================
    [..]
     (#) First, user has to fill the data structure (pointer to
         SDIO_DataInitTypeDef) according to the selected data type to be received.
         The parameters that should be filled are:
          (++) Data TimeOut
          (++) Data Length
          (++) Data Block size
          (++) Data Transfer direction:  should be to card (To CARD)
          (++) Data Transfer mode
          (++) DPSM Status (Enable or Disable)
  
     (#) Configure the SDMMC resources to send the data to the card according to 
         selected transfer mode.
                     
     (#) Send the selected Write command.
                    
     (#) Use the SDIO flags/interrupts to check the transfer status.
       
    *** Command management operations ***
    =====================================
    [..]
     (#) The commands used for Read/Write/Erase operations are managed in 
         separate functions. 
         Each function allows to send the needed command with the related argument,
         then check the response.
         By the same approach, you could implement a command and check the response.
  
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

#if defined(SDIO)

/** @addtogroup STM32F4xx_HAL_Driver
  * @{
  */

/** @defgroup SDMMC_LL SDMMC Low Layer
  * @brief Low layer module for SD
  * @{
  */

#if defined(HAL_SD_MODULE_ENABLED) || defined(HAL_MMC_MODULE_ENABLED)

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
static uint32_t SDMMC_GetCmdError(SDIO_TypeDef *SDIOx);
static uint32_t SDMMC_GetCmdResp1(SDIO_TypeDef *SDIOx, uint8_t SD_CMD, uint32_t Timeout);
static uint32_t SDMMC_GetCmdResp2(SDIO_TypeDef *SDIOx);
static uint32_t SDMMC_GetCmdResp3(SDIO_TypeDef *SDIOx);
static uint32_t SDMMC_GetCmdResp7(SDIO_TypeDef *SDIOx);
static uint32_t SDMMC_GetCmdResp6(SDIO_TypeDef *SDIOx, uint8_t SD_CMD, uint16_t *pRCA);

/* Exported functions --------------------------------------------------------*/

/** @defgroup SDMMC_LL_Exported_Functions SDMMC Low Layer Exported Functions
  * @{
  */

/** @defgroup HAL_SDMMC_LL_Group1 Initialization de-initialization functions 
 *  @brief    Initialization and Configuration functions 
 *
@verbatim    
 ===============================================================================
              ##### Initialization/de-initialization functions #####
 ===============================================================================
    [..]  This section provides functions allowing to:
 
@endverbatim
  * @{
  */

/**
  * @brief  Initializes the SDMMC according to the specified
  *         parameters in the SDMMC_InitTypeDef and create the associated handle.
  * @param  SDIOx: Pointer to SDMMC register base
  * @param  Init: SDMMC initialization structure   
  * @retval HAL status
  */
HAL_StatusTypeDef SDIO_Init(SDIO_TypeDef *SDIOx, SDIO_InitTypeDef Init)
{
  uint32_t tmpreg = 0;

  /* Check the parameters */
  assert_param(IS_SDIO_ALL_INSTANCE(SDIOx));
  assert_param(IS_SDIO_CLOCK_EDGE(Init.ClockEdge)); 
  assert_param(IS_SDIO_CLOCK_BYPASS(Init.ClockBypass));
  assert_param(IS_SDIO_CLOCK_POWER_SAVE(Init.ClockPowerSave));
  assert_param(IS_SDIO_BUS_WIDE(Init.BusWide));
  assert_param(IS_SDIO_HARDWARE_FLOW_CONTROL(Init.HardwareFlowControl));
  assert_param(IS_SDIO_CLKDIV(Init.ClockDiv));
  
  /* Set SDMMC configuration parameters */
  tmpreg |= (Init.ClockEdge           |\
             Init.ClockBypass         |\
             Init.ClockPowerSave      |\
             Init.BusWide             |\
             Init.HardwareFlowControl |\
             Init.ClockDiv
             ); 
  
  /* Write to SDMMC CLKCR */
  MODIFY_REG(SDIOx->CLKCR, CLKCR_CLEAR_MASK, tmpreg);  

  return HAL_OK;
}


/**
  * @}
  */

/** @defgroup HAL_SDMMC_LL_Group2 IO operation functions 
 *  @brief   Data transfers functions 
 *
@verbatim   
 ===============================================================================
                      ##### I/O operation functions #####
 ===============================================================================  
    [..]
    This subsection provides a set of functions allowing to manage the SDMMC data 
    transfers.

@endverbatim
  * @{
  */

/**
  * @brief  Read data (word) from Rx FIFO in blocking mode (polling) 
  * @param  SDIOx: Pointer to SDMMC register base
  * @retval HAL status
  */
uint32_t SDIO_ReadFIFO(SDIO_TypeDef *SDIOx)
{
  /* Read data from Rx FIFO */ 
  return (SDIOx->FIFO);
}

/**
  * @brief  Write data (word) to Tx FIFO in blocking mode (polling) 
  * @param  SDIOx: Pointer to SDMMC register base
  * @param  pWriteData: pointer to data to write
  * @retval HAL status
  */
HAL_StatusTypeDef SDIO_WriteFIFO(SDIO_TypeDef *SDIOx, uint32_t *pWriteData)
{ 
  /* Write data to FIFO */ 
  SDIOx->FIFO = *pWriteData;

  return HAL_OK;
}

/**
  * @}
  */

/** @defgroup HAL_SDMMC_LL_Group3 Peripheral Control functions 
 *  @brief   management functions 
 *
@verbatim   
 ===============================================================================
                      ##### Peripheral Control functions #####
 ===============================================================================  
    [..]
    This subsection provides a set of functions allowing to control the SDMMC data 
    transfers.

@endverbatim
  * @{
  */

/**
  * @brief  Set SDMMC Power state to ON. 
  * @param  SDIOx: Pointer to SDMMC register base
  * @retval HAL status
  */
HAL_StatusTypeDef SDIO_PowerState_ON(SDIO_TypeDef *SDIOx)
{  
  /* Set power state to ON */ 
  SDIOx->POWER = SDIO_POWER_PWRCTRL;

  /* 1ms: required power up waiting time before starting the SD initialization
  sequence */
  HAL_Delay(2);
  
  return HAL_OK;
}

/**
  * @brief  Set SDMMC Power state to OFF. 
  * @param  SDIOx: Pointer to SDMMC register base
  * @retval HAL status
  */
HAL_StatusTypeDef SDIO_PowerState_OFF(SDIO_TypeDef *SDIOx)
{
  /* Set power state to OFF */
  SDIOx->POWER = (uint32_t)0x00000000;
  
  return HAL_OK;
}

/**
  * @brief  Get SDMMC Power state. 
  * @param  SDIOx: Pointer to SDMMC register base
  * @retval Power status of the controller. The returned value can be one of the 
  *         following values:
  *            - 0x00: Power OFF
  *            - 0x02: Power UP
  *            - 0x03: Power ON 
  */
uint32_t SDIO_GetPowerState(SDIO_TypeDef *SDIOx)  
{
  return (SDIOx->POWER & SDIO_POWER_PWRCTRL);
}

/**
  * @brief  Configure the SDMMC command path according to the specified parameters in
  *         SDIO_CmdInitTypeDef structure and send the command 
  * @param  SDIOx: Pointer to SDMMC register base
  * @param  Command: pointer to a SDIO_CmdInitTypeDef structure that contains 
  *         the configuration information for the SDMMC command
  * @retval HAL status
  */
HAL_StatusTypeDef SDIO_SendCommand(SDIO_TypeDef *SDIOx, SDIO_CmdInitTypeDef *Command)
{
  uint32_t tmpreg = 0;
  
  /* Check the parameters */
  assert_param(IS_SDIO_CMD_INDEX(Command->CmdIndex));
  assert_param(IS_SDIO_RESPONSE(Command->Response));
  assert_param(IS_SDIO_WAIT(Command->WaitForInterrupt));
  assert_param(IS_SDIO_CPSM(Command->CPSM));

  /* Set the SDMMC Argument value */
  SDIOx->ARG = Command->Argument;

  /* Set SDMMC command parameters */
  tmpreg |= (uint32_t)(Command->CmdIndex         |\
                       Command->Response         |\
                       Command->WaitForInterrupt |\
                       Command->CPSM);
  
  /* Write to SDMMC CMD register */
  MODIFY_REG(SDIOx->CMD, CMD_CLEAR_MASK, tmpreg); 
  
  return HAL_OK;  
}

/**
  * @brief  Return the command index of last command for which response received
  * @param  SDIOx: Pointer to SDMMC register base
  * @retval Command index of the last command response received
  */
uint8_t SDIO_GetCommandResponse(SDIO_TypeDef *SDIOx)
{
  return (uint8_t)(SDIOx->RESPCMD);
}


/**
  * @brief  Return the response received from the card for the last command
  * @param  SDIOx: Pointer to SDMMC register base    
  * @param  Response: Specifies the SDMMC response register. 
  *          This parameter can be one of the following values:
  *            @arg SDIO_RESP1: Response Register 1
  *            @arg SDIO_RESP2: Response Register 2
  *            @arg SDIO_RESP3: Response Register 3
  *            @arg SDIO_RESP4: Response Register 4  
  * @retval The Corresponding response register value
  */
uint32_t SDIO_GetResponse(SDIO_TypeDef *SDIOx, uint32_t Response)
{
  uint32_t tmp;

  /* Check the parameters */
  assert_param(IS_SDIO_RESP(Response));
  
  /* Get the response */
  tmp = (uint32_t)(&(SDIOx->RESP1)) + Response;
  
  return (*(__IO uint32_t *) tmp);
}  

/**
  * @brief  Configure the SDMMC data path according to the specified 
  *         parameters in the SDIO_DataInitTypeDef.
  * @param  SDIOx: Pointer to SDIO register base  
  * @param  Data : pointer to a SDIO_DataInitTypeDef structure 
  *         that contains the configuration information for the SDMMC data.
  * @retval HAL status
  */
HAL_StatusTypeDef SDIO_ConfigData(SDIO_TypeDef *SDIOx, SDIO_DataInitTypeDef* Data)
{
  uint32_t tmpreg = 0;
  
  /* Check the parameters */
  assert_param(IS_SDIO_DATA_LENGTH(Data->DataLength));
  assert_param(IS_SDIO_BLOCK_SIZE(Data->DataBlockSize));
  assert_param(IS_SDIO_TRANSFER_DIR(Data->TransferDir));
  assert_param(IS_SDIO_TRANSFER_MODE(Data->TransferMode));
  assert_param(IS_SDIO_DPSM(Data->DPSM));

  /* Set the SDMMC Data TimeOut value */
  SDIOx->DTIMER = Data->DataTimeOut;

  /* Set the SDMMC DataLength value */
  SDIOx->DLEN = Data->DataLength;

  /* Set the SDMMC data configuration parameters */
  tmpreg |= (uint32_t)(Data->DataBlockSize |\
                       Data->TransferDir   |\
                       Data->TransferMode  |\
                       Data->DPSM);
  
  /* Write to SDMMC DCTRL */
  MODIFY_REG(SDIOx->DCTRL, DCTRL_CLEAR_MASK, tmpreg);

  return HAL_OK;

}

/**
  * @brief  Returns number of remaining data bytes to be transferred.
  * @param  SDIOx: Pointer to SDIO register base
  * @retval Number of remaining data bytes to be transferred
  */
uint32_t SDIO_GetDataCounter(SDIO_TypeDef *SDIOx)
{
  return (SDIOx->DCOUNT);
}

/**
  * @brief  Get the FIFO data
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval Data received
  */
uint32_t SDIO_GetFIFOCount(SDIO_TypeDef *SDIOx)
{
  return (SDIOx->FIFO);
}

/**
  * @brief  Sets one of the two options of inserting read wait interval.
  * @param  SDIOx: Pointer to SDIO register base   
  * @param  SDIO_ReadWaitMode: SDMMC Read Wait operation mode.
  *          This parameter can be:
  *            @arg SDIO_READ_WAIT_MODE_CLK: Read Wait control by stopping SDMMCCLK
  *            @arg SDIO_READ_WAIT_MODE_DATA2: Read Wait control using SDMMC_DATA2
  * @retval None
  */
HAL_StatusTypeDef SDIO_SetSDMMCReadWaitMode(SDIO_TypeDef *SDIOx, uint32_t SDIO_ReadWaitMode)
{
  /* Check the parameters */
  assert_param(IS_SDIO_READWAIT_MODE(SDIO_ReadWaitMode));

  /* Set SDMMC read wait mode */
  MODIFY_REG(SDIOx->DCTRL, SDIO_DCTRL_RWMOD, SDIO_ReadWaitMode);
  
  return HAL_OK;  
}

/**
  * @}
  */


/** @defgroup HAL_SDMMC_LL_Group4 Command management functions 
 *  @brief   Data transfers functions 
 *
@verbatim   
 ===============================================================================
                   ##### Commands management functions #####
 ===============================================================================  
    [..]
    This subsection provides a set of functions allowing to manage the needed commands.

@endverbatim
  * @{
  */

/**
  * @brief  Send the Data Block Length command and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdBlockLength(SDIO_TypeDef *SDIOx, uint32_t BlockSize)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = (uint32_t)BlockSize;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SET_BLOCKLEN;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_SET_BLOCKLEN, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Read Single Block command and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdReadSingleBlock(SDIO_TypeDef *SDIOx, uint32_t ReadAdd)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = (uint32_t)ReadAdd;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_READ_SINGLE_BLOCK;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_READ_SINGLE_BLOCK, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Read Multi Block command and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdReadMultiBlock(SDIO_TypeDef *SDIOx, uint32_t ReadAdd)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = (uint32_t)ReadAdd;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_READ_MULT_BLOCK;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_READ_MULT_BLOCK, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Write Single Block command and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdWriteSingleBlock(SDIO_TypeDef *SDIOx, uint32_t WriteAdd)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = (uint32_t)WriteAdd;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_WRITE_SINGLE_BLOCK;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_WRITE_SINGLE_BLOCK, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Write Multi Block command and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdWriteMultiBlock(SDIO_TypeDef *SDIOx, uint32_t WriteAdd)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = (uint32_t)WriteAdd;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_WRITE_MULT_BLOCK;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_WRITE_MULT_BLOCK, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Start Address Erase command for SD and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdSDEraseStartAdd(SDIO_TypeDef *SDIOx, uint32_t StartAdd)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = (uint32_t)StartAdd;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SD_ERASE_GRP_START;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_SD_ERASE_GRP_START, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the End Address Erase command for SD and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdSDEraseEndAdd(SDIO_TypeDef *SDIOx, uint32_t EndAdd)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = (uint32_t)EndAdd;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SD_ERASE_GRP_END;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_SD_ERASE_GRP_END, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Start Address Erase command and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdEraseStartAdd(SDIO_TypeDef *SDIOx, uint32_t StartAdd)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = (uint32_t)StartAdd;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_ERASE_GRP_START;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_ERASE_GRP_START, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the End Address Erase command and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdEraseEndAdd(SDIO_TypeDef *SDIOx, uint32_t EndAdd)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = (uint32_t)EndAdd;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_ERASE_GRP_END;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_ERASE_GRP_END, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Erase command and check the response
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdErase(SDIO_TypeDef *SDIOx)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Set Block Size for Card */ 
  sdmmc_cmdinit.Argument         = 0U;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_ERASE;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_ERASE, SDIO_MAXERASETIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Stop Transfer command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdStopTransfer(SDIO_TypeDef *SDIOx)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Send CMD12 STOP_TRANSMISSION  */
  sdmmc_cmdinit.Argument         = 0U;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_STOP_TRANSMISSION;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_STOP_TRANSMISSION, SDIO_STOPTRANSFERTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Select Deselect command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @param  addr: Address of the card to be selected  
  * @retval HAL status
  */
uint32_t SDMMC_CmdSelDesel(SDIO_TypeDef *SDIOx, uint64_t Addr)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Send CMD7 SDMMC_SEL_DESEL_CARD */
  sdmmc_cmdinit.Argument         = (uint32_t)Addr;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SEL_DESEL_CARD;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_SEL_DESEL_CARD, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Go Idle State command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdGoIdleState(SDIO_TypeDef *SDIOx)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  sdmmc_cmdinit.Argument         = 0U;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_GO_IDLE_STATE;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_NO;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdError(SDIOx);

  return errorstate;
}

/**
  * @brief  Send the Operating Condition command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdOperCond(SDIO_TypeDef *SDIOx)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Send CMD8 to verify SD card interface operating condition */
  /* Argument: - [31:12]: Reserved (shall be set to '0')
  - [11:8]: Supply Voltage (VHS) 0x1 (Range: 2.7-3.6 V)
  - [7:0]: Check Pattern (recommended 0xAA) */
  /* CMD Response: R7 */
  sdmmc_cmdinit.Argument         = SDMMC_CHECK_PATTERN;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_HS_SEND_EXT_CSD;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp7(SDIOx);

  return errorstate;
}

/**
  * @brief  Send the Application command to verify that that the next command 
  *         is an application specific com-mand rather than a standard command
  *         and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @param  Argument: Command Argument 
  * @retval HAL status
  */
uint32_t SDMMC_CmdAppCommand(SDIO_TypeDef *SDIOx, uint32_t Argument)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  sdmmc_cmdinit.Argument         = (uint32_t)Argument;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_APP_CMD;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  /* If there is a HAL_ERROR, it is a MMC card, else
  it is a SD card: SD card 2.0 (voltage range mismatch)
     or SD card 1.x */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_APP_CMD, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the command asking the accessed card to send its operating 
  *         condition register (OCR)
  * @param  SDIOx: Pointer to SDIO register base 
  * @param  Argument: Command Argument
  * @retval HAL status
  */
uint32_t SDMMC_CmdAppOperCommand(SDIO_TypeDef *SDIOx, uint32_t Argument)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  sdmmc_cmdinit.Argument         = SDMMC_VOLTAGE_WINDOW_SD | Argument;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SD_APP_OP_COND;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp3(SDIOx);

  return errorstate;
}

/**
  * @brief  Send the Bus Width command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @param  BusWidth: BusWidth
  * @retval HAL status
  */
uint32_t SDMMC_CmdBusWidth(SDIO_TypeDef *SDIOx, uint32_t BusWidth)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  sdmmc_cmdinit.Argument         = (uint32_t)BusWidth;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_APP_SD_SET_BUSWIDTH;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_APP_SD_SET_BUSWIDTH, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Send SCR command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdSendSCR(SDIO_TypeDef *SDIOx)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Send CMD51 SD_APP_SEND_SCR */
  sdmmc_cmdinit.Argument         = 0U;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SD_APP_SEND_SCR;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_SD_APP_SEND_SCR, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Send CID command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdSendCID(SDIO_TypeDef *SDIOx)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Send CMD2 ALL_SEND_CID */
  sdmmc_cmdinit.Argument         = 0U;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_ALL_SEND_CID;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_LONG;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp2(SDIOx);

  return errorstate;
}

/**
  * @brief  Send the Send CSD command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @param  Argument: Command Argument
  * @retval HAL status
  */
uint32_t SDMMC_CmdSendCSD(SDIO_TypeDef *SDIOx, uint32_t Argument)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Send CMD9 SEND_CSD */
  sdmmc_cmdinit.Argument         = Argument;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SEND_CSD;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_LONG;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp2(SDIOx);

  return errorstate;
}

/**
  * @brief  Send the Send CSD command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @param  pRCA: Card RCA  
  * @retval HAL status
  */
uint32_t SDMMC_CmdSetRelAdd(SDIO_TypeDef *SDIOx, uint16_t *pRCA)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Send CMD3 SD_CMD_SET_REL_ADDR */
  sdmmc_cmdinit.Argument         = 0U;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SET_REL_ADDR;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp6(SDIOx, SDMMC_CMD_SET_REL_ADDR, pRCA);

  return errorstate;
}

/**
  * @brief  Send the Status command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @param  Argument: Command Argument
  * @retval HAL status
  */
uint32_t SDMMC_CmdSendStatus(SDIO_TypeDef *SDIOx, uint32_t Argument)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  sdmmc_cmdinit.Argument         = Argument;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SEND_STATUS;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_SEND_STATUS, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Send the Status register command and check the response.
  * @param  SDIOx: Pointer to SDIO register base 
  * @retval HAL status
  */
uint32_t SDMMC_CmdStatusRegister(SDIO_TypeDef *SDIOx)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  sdmmc_cmdinit.Argument         = 0U;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SD_APP_STATUS;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_SD_APP_STATUS, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @brief  Sends host capacity support information and activates the card's 
  *         initialization process. Send SDMMC_CMD_SEND_OP_COND command
  * @param  SDIOx: Pointer to SDIO register base 
  * @parame Argument: Argument used for the command
  * @retval HAL status
  */
uint32_t SDMMC_CmdOpCondition(SDIO_TypeDef *SDIOx, uint32_t Argument)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  sdmmc_cmdinit.Argument         = Argument;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_SEND_OP_COND;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp3(SDIOx);

  return errorstate;
}

/**
  * @brief  Checks switchable function and switch card function. SDMMC_CMD_HS_SWITCH command
  * @param  SDIOx: Pointer to SDIO register base 
  * @parame Argument: Argument used for the command
  * @retval HAL status
  */
uint32_t SDMMC_CmdSwitch(SDIO_TypeDef *SDIOx, uint32_t Argument)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;
  
  /* Send CMD6 to activate SDR50 Mode and Power Limit 1.44W */
  /* CMD Response: R1 */
  sdmmc_cmdinit.Argument         = Argument; /* SDMMC_SDR25_SWITCH_PATTERN */
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_HS_SWITCH;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);
  
  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_HS_SWITCH, SDIO_CMDTIMEOUT);

  return errorstate;
}

/**
  * @}
  */

/* Private function ----------------------------------------------------------*/  
/** @addtogroup SD_Private_Functions
  * @{
  */
    
/**
  * @brief  Checks for error conditions for CMD0.
  * @param  hsd: SD handle
  * @retval SD Card error state
  */
static uint32_t SDMMC_GetCmdError(SDIO_TypeDef *SDIOx)
{
  /* 8 is the number of required instructions cycles for the below loop statement.
  The SDIO_CMDTIMEOUT is expressed in ms */
  uint32_t count = SDIO_CMDTIMEOUT * (SystemCoreClock / 8U /1000U);
  
  do
  {
    if (count-- == 0U)
    {
      return SDMMC_ERROR_TIMEOUT;
    }
    
  }while(!__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CMDSENT));
  
  /* Clear all the static flags */
  __SDIO_CLEAR_FLAG(SDIOx, SDIO_STATIC_CMD_FLAGS);
  
  return SDMMC_ERROR_NONE;
}

/**
  * @brief  Checks for error conditions for R1 response.
  * @param  hsd: SD handle
  * @param  SD_CMD: The sent command index  
  * @retval SD Card error state
  */
static uint32_t SDMMC_GetCmdResp1(SDIO_TypeDef *SDIOx, uint8_t SD_CMD, uint32_t Timeout)
{
  uint32_t response_r1;
  uint32_t sta_reg;
  
  /* 8 is the number of required instructions cycles for the below loop statement.
  The Timeout is expressed in ms */
  uint32_t count = Timeout * (SystemCoreClock / 8U /1000U);
  
  do
  {
    if (count-- == 0U)
    {
      return SDMMC_ERROR_TIMEOUT;
    }
    sta_reg = SDIOx->STA;
  }while(((sta_reg & (SDIO_FLAG_CCRCFAIL | SDIO_FLAG_CMDREND | SDIO_FLAG_CTIMEOUT)) == 0U) ||
         ((sta_reg & SDIO_FLAG_CMDACT) != 0U ));
    
  if(__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT))
  {
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT);
    
    return SDMMC_ERROR_CMD_RSP_TIMEOUT;
  }
  else if(__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CCRCFAIL))
  {
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CCRCFAIL);
    
    return SDMMC_ERROR_CMD_CRC_FAIL;
  }
  else
  {
    /* Nothing to do */
  }
  
  /* Clear all the static flags */
  __SDIO_CLEAR_FLAG(SDIOx, SDIO_STATIC_CMD_FLAGS);
  
  /* Check response received is of desired command */
  if(SDIO_GetCommandResponse(SDIOx) != SD_CMD)
  {
    return SDMMC_ERROR_CMD_CRC_FAIL;
  }
  
  /* We have received response, retrieve it for analysis  */
  response_r1 = SDIO_GetResponse(SDIOx, SDIO_RESP1);
  
  if((response_r1 & SDMMC_OCR_ERRORBITS) == SDMMC_ALLZERO)
  {
    return SDMMC_ERROR_NONE;
  }
  else if((response_r1 & SDMMC_OCR_ADDR_OUT_OF_RANGE) == SDMMC_OCR_ADDR_OUT_OF_RANGE)
  {
    return SDMMC_ERROR_ADDR_OUT_OF_RANGE;
  }
  else if((response_r1 & SDMMC_OCR_ADDR_MISALIGNED) == SDMMC_OCR_ADDR_MISALIGNED)
  {
    return SDMMC_ERROR_ADDR_MISALIGNED;
  }
  else if((response_r1 & SDMMC_OCR_BLOCK_LEN_ERR) == SDMMC_OCR_BLOCK_LEN_ERR)
  {
    return SDMMC_ERROR_BLOCK_LEN_ERR;
  }
  else if((response_r1 & SDMMC_OCR_ERASE_SEQ_ERR) == SDMMC_OCR_ERASE_SEQ_ERR)
  {
    return SDMMC_ERROR_ERASE_SEQ_ERR;
  }
  else if((response_r1 & SDMMC_OCR_BAD_ERASE_PARAM) == SDMMC_OCR_BAD_ERASE_PARAM)
  {
    return SDMMC_ERROR_BAD_ERASE_PARAM;
  }
  else if((response_r1 & SDMMC_OCR_WRITE_PROT_VIOLATION) == SDMMC_OCR_WRITE_PROT_VIOLATION)
  {
    return SDMMC_ERROR_WRITE_PROT_VIOLATION;
  }
  else if((response_r1 & SDMMC_OCR_LOCK_UNLOCK_FAILED) == SDMMC_OCR_LOCK_UNLOCK_FAILED)
  {
    return SDMMC_ERROR_LOCK_UNLOCK_FAILED;
  }
  else if((response_r1 & SDMMC_OCR_COM_CRC_FAILED) == SDMMC_OCR_COM_CRC_FAILED)
  {
    return SDMMC_ERROR_COM_CRC_FAILED;
  }
  else if((response_r1 & SDMMC_OCR_ILLEGAL_CMD) == SDMMC_OCR_ILLEGAL_CMD)
  {
    return SDMMC_ERROR_ILLEGAL_CMD;
  }
  else if((response_r1 & SDMMC_OCR_CARD_ECC_FAILED) == SDMMC_OCR_CARD_ECC_FAILED)
  {
    return SDMMC_ERROR_CARD_ECC_FAILED;
  }
  else if((response_r1 & SDMMC_OCR_CC_ERROR) == SDMMC_OCR_CC_ERROR)
  {
    return SDMMC_ERROR_CC_ERR;
  }
  else if((response_r1 & SDMMC_OCR_STREAM_READ_UNDERRUN) == SDMMC_OCR_STREAM_READ_UNDERRUN)
  {
    return SDMMC_ERROR_STREAM_READ_UNDERRUN;
  }
  else if((response_r1 & SDMMC_OCR_STREAM_WRITE_OVERRUN) == SDMMC_OCR_STREAM_WRITE_OVERRUN)
  {
    return SDMMC_ERROR_STREAM_WRITE_OVERRUN;
  }
  else if((response_r1 & SDMMC_OCR_CID_CSD_OVERWRITE) == SDMMC_OCR_CID_CSD_OVERWRITE)
  {
    return SDMMC_ERROR_CID_CSD_OVERWRITE;
  }
  else if((response_r1 & SDMMC_OCR_WP_ERASE_SKIP) == SDMMC_OCR_WP_ERASE_SKIP)
  {
    return SDMMC_ERROR_WP_ERASE_SKIP;
  }
  else if((response_r1 & SDMMC_OCR_CARD_ECC_DISABLED) == SDMMC_OCR_CARD_ECC_DISABLED)
  {
    return SDMMC_ERROR_CARD_ECC_DISABLED;
  }
  else if((response_r1 & SDMMC_OCR_ERASE_RESET) == SDMMC_OCR_ERASE_RESET)
  {
    return SDMMC_ERROR_ERASE_RESET;
  }
  else if((response_r1 & SDMMC_OCR_AKE_SEQ_ERROR) == SDMMC_OCR_AKE_SEQ_ERROR)
  {
    return SDMMC_ERROR_AKE_SEQ_ERR;
  }
  else
  {
    return SDMMC_ERROR_GENERAL_UNKNOWN_ERR;
  }
}

/**
  * @brief  Checks for error conditions for R2 (CID or CSD) response.
  * @param  hsd: SD handle
  * @retval SD Card error state
  */
static uint32_t SDMMC_GetCmdResp2(SDIO_TypeDef *SDIOx)
{
  uint32_t sta_reg;
  /* 8 is the number of required instructions cycles for the below loop statement.
  The SDIO_CMDTIMEOUT is expressed in ms */
  uint32_t count = SDIO_CMDTIMEOUT * (SystemCoreClock / 8U /1000U);
  
  do
  {
    if (count-- == 0U)
    {
      return SDMMC_ERROR_TIMEOUT;
    }
    sta_reg = SDIOx->STA;
  }while(((sta_reg & (SDIO_FLAG_CCRCFAIL | SDIO_FLAG_CMDREND | SDIO_FLAG_CTIMEOUT)) == 0U) ||
         ((sta_reg & SDIO_FLAG_CMDACT) != 0U ));
    
  if (__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT))
  {
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT);
    
    return SDMMC_ERROR_CMD_RSP_TIMEOUT;
  }
  else if (__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CCRCFAIL))
  {
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CCRCFAIL);
    
    return SDMMC_ERROR_CMD_CRC_FAIL;
  }
  else
  {
    /* No error flag set */
    /* Clear all the static flags */
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_STATIC_CMD_FLAGS);
  }

  return SDMMC_ERROR_NONE;
}

/**
  * @brief  Checks for error conditions for R3 (OCR) response.
  * @param  hsd: SD handle
  * @retval SD Card error state
  */
static uint32_t SDMMC_GetCmdResp3(SDIO_TypeDef *SDIOx)
{
  uint32_t sta_reg;
  /* 8 is the number of required instructions cycles for the below loop statement.
  The SDIO_CMDTIMEOUT is expressed in ms */
  uint32_t count = SDIO_CMDTIMEOUT * (SystemCoreClock / 8U /1000U);
  
  do
  {
    if (count-- == 0U)
    {
      return SDMMC_ERROR_TIMEOUT;
    }
    sta_reg = SDIOx->STA;
  }while(((sta_reg & (SDIO_FLAG_CCRCFAIL | SDIO_FLAG_CMDREND | SDIO_FLAG_CTIMEOUT)) == 0U) ||
         ((sta_reg & SDIO_FLAG_CMDACT) != 0U ));
    
  if(__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT))
  {
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT);
    
    return SDMMC_ERROR_CMD_RSP_TIMEOUT;
  }
  else
  {  
    /* Clear all the static flags */
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_STATIC_CMD_FLAGS);
  }
  
  return SDMMC_ERROR_NONE;
}

/**
  * @brief  Checks for error conditions for R6 (RCA) response.
  * @param  hsd: SD handle
  * @param  SD_CMD: The sent command index
  * @param  pRCA: Pointer to the variable that will contain the SD card relative 
  *         address RCA   
  * @retval SD Card error state
  */
static uint32_t SDMMC_GetCmdResp6(SDIO_TypeDef *SDIOx, uint8_t SD_CMD, uint16_t *pRCA)
{
  uint32_t response_r1;
  uint32_t sta_reg;

  /* 8 is the number of required instructions cycles for the below loop statement.
  The SDIO_CMDTIMEOUT is expressed in ms */
  uint32_t count = SDIO_CMDTIMEOUT * (SystemCoreClock / 8U /1000U);
  
  do
  {
    if (count-- == 0U)
    {
      return SDMMC_ERROR_TIMEOUT;
    }
    sta_reg = SDIOx->STA;
  }while(((sta_reg & (SDIO_FLAG_CCRCFAIL | SDIO_FLAG_CMDREND | SDIO_FLAG_CTIMEOUT)) == 0U) ||
         ((sta_reg & SDIO_FLAG_CMDACT) != 0U ));
    
  if(__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT))
  {
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT);
    
    return SDMMC_ERROR_CMD_RSP_TIMEOUT;
  }
  else if(__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CCRCFAIL))
  {
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CCRCFAIL);
    
    return SDMMC_ERROR_CMD_CRC_FAIL;
  }
  else
  {
    /* Nothing to do */
  }
  
  /* Check response received is of desired command */
  if(SDIO_GetCommandResponse(SDIOx) != SD_CMD)
  {
    return SDMMC_ERROR_CMD_CRC_FAIL;
  }
  
  /* Clear all the static flags */
  __SDIO_CLEAR_FLAG(SDIOx, SDIO_STATIC_CMD_FLAGS);
  
  /* We have received response, retrieve it.  */
  response_r1 = SDIO_GetResponse(SDIOx, SDIO_RESP1);
  
  if((response_r1 & (SDMMC_R6_GENERAL_UNKNOWN_ERROR | SDMMC_R6_ILLEGAL_CMD | SDMMC_R6_COM_CRC_FAILED)) == SDMMC_ALLZERO)
  {
    *pRCA = (uint16_t) (response_r1 >> 16);
    
    return SDMMC_ERROR_NONE;
  }
  else if((response_r1 & SDMMC_R6_ILLEGAL_CMD) == SDMMC_R6_ILLEGAL_CMD)
  {
    return SDMMC_ERROR_ILLEGAL_CMD;
  }
  else if((response_r1 & SDMMC_R6_COM_CRC_FAILED) == SDMMC_R6_COM_CRC_FAILED)
  {
    return SDMMC_ERROR_COM_CRC_FAILED;
  }
  else
  {
    return SDMMC_ERROR_GENERAL_UNKNOWN_ERR;
  }
}

/**
  * @brief  Checks for error conditions for R7 response.
  * @param  hsd: SD handle
  * @retval SD Card error state
  */
static uint32_t SDMMC_GetCmdResp7(SDIO_TypeDef *SDIOx)
{
  uint32_t sta_reg;
  /* 8 is the number of required instructions cycles for the below loop statement.
  The SDIO_CMDTIMEOUT is expressed in ms */
  uint32_t count = SDIO_CMDTIMEOUT * (SystemCoreClock / 8U /1000U);
  
  do
  {
    if (count-- == 0U)
    {
      return SDMMC_ERROR_TIMEOUT;
    }
    sta_reg = SDIOx->STA;
  }while(((sta_reg & (SDIO_FLAG_CCRCFAIL | SDIO_FLAG_CMDREND | SDIO_FLAG_CTIMEOUT)) == 0U) ||
         ((sta_reg & SDIO_FLAG_CMDACT) != 0U ));
    
  if(__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT))
  {
    /* Card is SD V2.0 compliant */
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CTIMEOUT);
    
    return SDMMC_ERROR_CMD_RSP_TIMEOUT;
  }
  else if(__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CCRCFAIL))
  {
    /* Card is SD V2.0 compliant */
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CCRCFAIL);
    
    return SDMMC_ERROR_CMD_CRC_FAIL;
  }
  else
  {
    /* Nothing to do */
  }
  
  if(__SDIO_GET_FLAG(SDIOx, SDIO_FLAG_CMDREND))
  {
    /* Card is SD V2.0 compliant */
    __SDIO_CLEAR_FLAG(SDIOx, SDIO_FLAG_CMDREND);
  }
  
  return SDMMC_ERROR_NONE;
  
}

/**
  * @brief  Send the Send EXT_CSD command and check the response.
  * @param  SDIOx: Pointer to SDMMC register base
  * @param  Argument: Command Argument
  * @retval HAL status
  */
uint32_t SDMMC_CmdSendEXTCSD(SDIO_TypeDef *SDIOx, uint32_t Argument)
{
  SDIO_CmdInitTypeDef  sdmmc_cmdinit;
  uint32_t errorstate;

  /* Send CMD9 SEND_CSD */
  sdmmc_cmdinit.Argument         = Argument;
  sdmmc_cmdinit.CmdIndex         = SDMMC_CMD_HS_SEND_EXT_CSD;
  sdmmc_cmdinit.Response         = SDIO_RESPONSE_SHORT;
  sdmmc_cmdinit.WaitForInterrupt = SDIO_WAIT_NO;
  sdmmc_cmdinit.CPSM             = SDIO_CPSM_ENABLE;
  (void)SDIO_SendCommand(SDIOx, &sdmmc_cmdinit);

  /* Check for error conditions */
  errorstate = SDMMC_GetCmdResp1(SDIOx, SDMMC_CMD_HS_SEND_EXT_CSD,SDIO_CMDTIMEOUT);

  return errorstate;
}


/**
  * @}
  */

#endif /* HAL_SD_MODULE_ENABLED || HAL_MMC_MODULE_ENABLED */
/**
  * @}
  */

/**
  * @}
  */

#endif /* SDIO */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
