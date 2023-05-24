/**
  ******************************************************************************
  * @file    stm32f4xx_hal_nand.c
  * @author  MCD Application Team
  * @brief   NAND HAL module driver.
  *          This file provides a generic firmware to drive NAND memories mounted 
  *          as external device.
  *         
  @verbatim
  ==============================================================================
                         ##### How to use this driver #####
  ==============================================================================    
    [..]
      This driver is a generic layered driver which contains a set of APIs used to 
      control NAND flash memories. It uses the FMC/FSMC layer functions to interface 
      with NAND devices. This driver is used as follows:
    
      (+) NAND flash memory configuration sequence using the function HAL_NAND_Init() 
          with control and timing parameters for both common and attribute spaces.
            
      (+) Read NAND flash memory maker and device IDs using the function
          HAL_NAND_Read_ID(). The read information is stored in the NAND_ID_TypeDef 
          structure declared by the function caller. 
        
      (+) Access NAND flash memory by read/write operations using the functions
          HAL_NAND_Read_Page_8b()/HAL_NAND_Read_SpareArea_8b(), 
          HAL_NAND_Write_Page_8b()/HAL_NAND_Write_SpareArea_8b(),
          HAL_NAND_Read_Page_16b()/HAL_NAND_Read_SpareArea_16b(), 
          HAL_NAND_Write_Page_16b()/HAL_NAND_Write_SpareArea_16b()
          to read/write page(s)/spare area(s). These functions use specific device 
          information (Block, page size..) predefined by the user in the HAL_NAND_Info_TypeDef 
          structure. The read/write address information is contained by the Nand_Address_Typedef
          structure passed as parameter.
        
      (+) Perform NAND flash Reset chip operation using the function HAL_NAND_Reset().
        
      (+) Perform NAND flash erase block operation using the function HAL_NAND_Erase_Block().
          The erase block address information is contained in the Nand_Address_Typedef 
          structure passed as parameter.
    
      (+) Read the NAND flash status operation using the function HAL_NAND_Read_Status().
        
      (+) You can also control the NAND device by calling the control APIs HAL_NAND_ECC_Enable()/
          HAL_NAND_ECC_Disable() to respectively enable/disable the ECC code correction
          feature or the function HAL_NAND_GetECC() to get the ECC correction code. 
       
      (+) You can monitor the NAND device HAL state by calling the function
          HAL_NAND_GetState()  

    [..]
      (@) This driver is a set of generic APIs which handle standard NAND flash operations.
          If a NAND flash device contains different operations and/or implementations, 
          it should be implemented separately.

    *** Callback registration ***
    =============================================
    [..]
      The compilation define  USE_HAL_NAND_REGISTER_CALLBACKS when set to 1
      allows the user to configure dynamically the driver callbacks.

      Use Functions @ref HAL_NAND_RegisterCallback() to register a user callback,
      it allows to register following callbacks:
        (+) MspInitCallback    : NAND MspInit.
        (+) MspDeInitCallback  : NAND MspDeInit.
      This function takes as parameters the HAL peripheral handle, the Callback ID
      and a pointer to the user callback function.

      Use function @ref HAL_NAND_UnRegisterCallback() to reset a callback to the default
      weak (surcharged) function. It allows to reset following callbacks:
        (+) MspInitCallback    : NAND MspInit.
        (+) MspDeInitCallback  : NAND MspDeInit.
      This function) takes as parameters the HAL peripheral handle and the Callback ID.

      By default, after the @ref HAL_NAND_Init and if the state is HAL_NAND_STATE_RESET
      all callbacks are reset to the corresponding legacy weak (surcharged) functions.
      Exception done for MspInit and MspDeInit callbacks that are respectively
      reset to the legacy weak (surcharged) functions in the @ref HAL_NAND_Init
      and @ref  HAL_NAND_DeInit only when these callbacks are null (not registered beforehand).
      If not, MspInit or MspDeInit are not null, the @ref HAL_NAND_Init and @ref HAL_NAND_DeInit
      keep and use the user MspInit/MspDeInit callbacks (registered beforehand)

      Callbacks can be registered/unregistered in READY state only.
      Exception done for MspInit/MspDeInit callbacks that can be registered/unregistered
      in READY or RESET state, thus registered (user) MspInit/DeInit callbacks can be used
      during the Init/DeInit.
      In that case first register the MspInit/MspDeInit user callbacks
      using @ref HAL_NAND_RegisterCallback before calling @ref HAL_NAND_DeInit
      or @ref HAL_NAND_Init function.

      When The compilation define USE_HAL_NAND_REGISTER_CALLBACKS is set to 0 or
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


#ifdef HAL_NAND_MODULE_ENABLED

#if defined(STM32F405xx) || defined(STM32F415xx) || defined(STM32F407xx) || defined(STM32F417xx) ||\
    defined(STM32F427xx) || defined(STM32F437xx) || defined(STM32F429xx) || defined(STM32F439xx) ||\
    defined(STM32F446xx) || defined(STM32F469xx) || defined(STM32F479xx)

/** @defgroup NAND NAND 
  * @brief NAND HAL module driver
  * @{
  */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/** @defgroup NAND_Private_Constants NAND Private Constants
  * @{
  */

/**
  * @}
  */

/* Private macro -------------------------------------------------------------*/    
/** @defgroup NAND_Private_Macros NAND Private Macros
  * @{
  */

/**
  * @}
  */
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Exported functions --------------------------------------------------------*/
/** @defgroup NAND_Exported_Functions NAND Exported Functions
  * @{
  */
    
/** @defgroup NAND_Exported_Functions_Group1 Initialization and de-initialization functions 
  * @brief    Initialization and Configuration functions 
  *
  @verbatim    
  ==============================================================================
            ##### NAND Initialization and de-initialization functions #####
  ==============================================================================
  [..]  
    This section provides functions allowing to initialize/de-initialize
    the NAND memory
  
@endverbatim
  * @{
  */
    
/**
  * @brief  Perform NAND memory Initialization sequence
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  ComSpace_Timing pointer to Common space timing structure
  * @param  AttSpace_Timing pointer to Attribute space timing structure
  * @retval HAL status
  */
HAL_StatusTypeDef  HAL_NAND_Init(NAND_HandleTypeDef *hnand, FMC_NAND_PCC_TimingTypeDef *ComSpace_Timing, FMC_NAND_PCC_TimingTypeDef *AttSpace_Timing)
{
  /* Check the NAND handle state */
  if(hnand == NULL)
  {
     return HAL_ERROR;
  }

  if(hnand->State == HAL_NAND_STATE_RESET)
  {
    /* Allocate lock resource and initialize it */
    hnand->Lock = HAL_UNLOCKED;

#if (USE_HAL_NAND_REGISTER_CALLBACKS == 1)
    if(hnand->MspInitCallback == NULL)
    {
      hnand->MspInitCallback = HAL_NAND_MspInit;
    }
    hnand->ItCallback = HAL_NAND_ITCallback;

    /* Init the low level hardware */
    hnand->MspInitCallback(hnand);
#else
    /* Initialize the low level hardware (MSP) */
    HAL_NAND_MspInit(hnand);
#endif
  }

  /* Initialize NAND control Interface */
  FMC_NAND_Init(hnand->Instance, &(hnand->Init));
  
  /* Initialize NAND common space timing Interface */  
  FMC_NAND_CommonSpace_Timing_Init(hnand->Instance, ComSpace_Timing, hnand->Init.NandBank);
  
  /* Initialize NAND attribute space timing Interface */  
  FMC_NAND_AttributeSpace_Timing_Init(hnand->Instance, AttSpace_Timing, hnand->Init.NandBank);
  
  /* Enable the NAND device */
  __FMC_NAND_ENABLE(hnand->Instance, hnand->Init.NandBank);
  
  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_READY;

  return HAL_OK;
}

/**
  * @brief  Perform NAND memory De-Initialization sequence
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_DeInit(NAND_HandleTypeDef *hnand)  
{
#if (USE_HAL_NAND_REGISTER_CALLBACKS == 1)
  if(hnand->MspDeInitCallback == NULL)
  {
    hnand->MspDeInitCallback = HAL_NAND_MspDeInit;
  }

  /* DeInit the low level hardware */
  hnand->MspDeInitCallback(hnand);
#else
  /* Initialize the low level hardware (MSP) */
  HAL_NAND_MspDeInit(hnand);
#endif

  /* Configure the NAND registers with their reset values */
  FMC_NAND_DeInit(hnand->Instance, hnand->Init.NandBank);

  /* Reset the NAND controller state */
  hnand->State = HAL_NAND_STATE_RESET;

  /* Release Lock */
  __HAL_UNLOCK(hnand);

  return HAL_OK;
}

/**
  * @brief  NAND MSP Init
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval None
  */
__weak void HAL_NAND_MspInit(NAND_HandleTypeDef *hnand)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hnand);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_NAND_MspInit could be implemented in the user file
   */ 
}

/**
  * @brief  NAND MSP DeInit
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval None
  */
__weak void HAL_NAND_MspDeInit(NAND_HandleTypeDef *hnand)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hnand);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_NAND_MspDeInit could be implemented in the user file
   */ 
}


/**
  * @brief  This function handles NAND device interrupt request.
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval HAL status
*/
void HAL_NAND_IRQHandler(NAND_HandleTypeDef *hnand)
{
  /* Check NAND interrupt Rising edge flag */
  if(__FMC_NAND_GET_FLAG(hnand->Instance, hnand->Init.NandBank, FMC_FLAG_RISING_EDGE))
  {
    /* NAND interrupt callback*/
#if (USE_HAL_NAND_REGISTER_CALLBACKS == 1)
    hnand->ItCallback(hnand);
#else
    HAL_NAND_ITCallback(hnand);
#endif

    /* Clear NAND interrupt Rising edge pending bit */
    __FMC_NAND_CLEAR_FLAG(hnand->Instance, hnand->Init.NandBank, FMC_FLAG_RISING_EDGE);
  }
  
  /* Check NAND interrupt Level flag */
  if(__FMC_NAND_GET_FLAG(hnand->Instance, hnand->Init.NandBank, FMC_FLAG_LEVEL))
  {
    /* NAND interrupt callback*/
#if (USE_HAL_NAND_REGISTER_CALLBACKS == 1)
    hnand->ItCallback(hnand);
#else
    HAL_NAND_ITCallback(hnand);
#endif

    /* Clear NAND interrupt Level pending bit */
    __FMC_NAND_CLEAR_FLAG(hnand->Instance, hnand->Init.NandBank, FMC_FLAG_LEVEL);
  }

  /* Check NAND interrupt Falling edge flag */
  if(__FMC_NAND_GET_FLAG(hnand->Instance, hnand->Init.NandBank, FMC_FLAG_FALLING_EDGE))
  {
    /* NAND interrupt callback*/
#if (USE_HAL_NAND_REGISTER_CALLBACKS == 1)
    hnand->ItCallback(hnand);
#else
    HAL_NAND_ITCallback(hnand);
#endif

    /* Clear NAND interrupt Falling edge pending bit */
    __FMC_NAND_CLEAR_FLAG(hnand->Instance, hnand->Init.NandBank, FMC_FLAG_FALLING_EDGE);
  }
  
  /* Check NAND interrupt FIFO empty flag */
  if(__FMC_NAND_GET_FLAG(hnand->Instance, hnand->Init.NandBank, FMC_FLAG_FEMPT))
  {
    /* NAND interrupt callback*/
#if (USE_HAL_NAND_REGISTER_CALLBACKS == 1)
    hnand->ItCallback(hnand);
#else
    HAL_NAND_ITCallback(hnand);
#endif

    /* Clear NAND interrupt FIFO empty pending bit */
    __FMC_NAND_CLEAR_FLAG(hnand->Instance, hnand->Init.NandBank, FMC_FLAG_FEMPT);
  }
}

/**
  * @brief  NAND interrupt feature callback
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval None
  */
__weak void HAL_NAND_ITCallback(NAND_HandleTypeDef *hnand)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(hnand);
  /* NOTE : This function Should not be modified, when the callback is needed,
            the HAL_NAND_ITCallback could be implemented in the user file
   */
}
 
/**
  * @}
  */
  
/** @defgroup NAND_Exported_Functions_Group2 Input and Output functions 
  * @brief    Input Output and memory control functions 
  *
  @verbatim    
  ==============================================================================
                    ##### NAND Input and Output functions #####
  ==============================================================================
  [..]  
    This section provides functions allowing to use and control the NAND 
    memory
  
@endverbatim
  * @{
  */

/**
  * @brief  Read the NAND memory electronic signature
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pNAND_ID NAND ID structure
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_Read_ID(NAND_HandleTypeDef *hnand, NAND_IDTypeDef *pNAND_ID)
{
  __IO uint32_t data = 0U;
  __IO uint32_t data1 = 0U;
  uint32_t deviceaddress = 0U;

  /* Process Locked */
  __HAL_LOCK(hnand);  
  
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }
  
  /* Update the NAND controller state */ 
  hnand->State = HAL_NAND_STATE_BUSY;
  
  /* Send Read ID command sequence */   
  *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA))  = NAND_CMD_READID;
  *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;

  /* Read the electronic signature from NAND flash */
#ifdef FSMC_PCR2_PWID
  if (hnand->Init.MemoryDataWidth == FSMC_NAND_PCC_MEM_BUS_WIDTH_8)
#else /* FMC_PCR2_PWID is defined */
  if (hnand->Init.MemoryDataWidth == FMC_NAND_PCC_MEM_BUS_WIDTH_8)
#endif
  {
    data = *(__IO uint32_t *)deviceaddress;

    /* Return the data read */
    pNAND_ID->Maker_Id   = ADDR_1ST_CYCLE(data);
    pNAND_ID->Device_Id  = ADDR_2ND_CYCLE(data);
    pNAND_ID->Third_Id   = ADDR_3RD_CYCLE(data);
    pNAND_ID->Fourth_Id  = ADDR_4TH_CYCLE(data);
  }
  else
  {
    data = *(__IO uint32_t *)deviceaddress;
    data1 = *((__IO uint32_t *)deviceaddress + 4U);
    
    /* Return the data read */
    pNAND_ID->Maker_Id   = ADDR_1ST_CYCLE(data);
    pNAND_ID->Device_Id  = ADDR_3RD_CYCLE(data);
    pNAND_ID->Third_Id   = ADDR_1ST_CYCLE(data1);
    pNAND_ID->Fourth_Id  = ADDR_3RD_CYCLE(data1);
  }
  
  /* Update the NAND controller state */ 
  hnand->State = HAL_NAND_STATE_READY;
  
  /* Process unlocked */
  __HAL_UNLOCK(hnand);
   
  return HAL_OK;
}

/**
  * @brief  NAND memory reset
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_Reset(NAND_HandleTypeDef *hnand)
{
  uint32_t deviceaddress = 0U;

  /* Process Locked */
  __HAL_LOCK(hnand);

  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }

  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }  
  
  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_BUSY; 
  
  /* Send NAND reset command */  
  *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = 0xFF;


  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_READY;
  
  /* Process unlocked */
  __HAL_UNLOCK(hnand);

  return HAL_OK;

}

/**
  * @brief  Configure the device: Enter the physical parameters of the device
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pDeviceConfig  pointer to NAND_DeviceConfigTypeDef structure
  * @retval HAL status
  */
HAL_StatusTypeDef  HAL_NAND_ConfigDevice(NAND_HandleTypeDef *hnand, NAND_DeviceConfigTypeDef *pDeviceConfig)
{
  hnand->Config.PageSize           = pDeviceConfig->PageSize;
  hnand->Config.SpareAreaSize      = pDeviceConfig->SpareAreaSize;
  hnand->Config.BlockSize          = pDeviceConfig->BlockSize;
  hnand->Config.BlockNbr           = pDeviceConfig->BlockNbr;
  hnand->Config.PlaneSize          = pDeviceConfig->PlaneSize;
  hnand->Config.PlaneNbr           = pDeviceConfig->PlaneNbr;
  hnand->Config.ExtraCommandEnable = pDeviceConfig->ExtraCommandEnable;
  
  return HAL_OK;
}
  
/**
  * @brief  Read Page(s) from NAND memory block (8-bits addressing)
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pAddress  pointer to NAND address structure
  * @param  pBuffer  pointer to destination read buffer
  * @param  NumPageToRead  number of pages to read from block 
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_Read_Page_8b(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress, uint8_t *pBuffer, uint32_t NumPageToRead)
{   
  __IO uint32_t index  = 0U;
  uint32_t tickstart = 0U;
  uint32_t deviceaddress = 0U, size = 0U, numPagesRead = 0U, nandaddress = 0U;
  
  /* Process Locked */
  __HAL_LOCK(hnand); 
  
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }

  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_BUSY;
  
  /* NAND raw address calculation */
  nandaddress = ARRAY_ADDRESS(pAddress, hnand);

  /* Page(s) read loop */  
  while((NumPageToRead != 0U) && (nandaddress < ((hnand->Config.BlockSize) * (hnand->Config.BlockNbr))))
  {
    /* update the buffer size */
    size = (hnand->Config.PageSize) + ((hnand->Config.PageSize) * numPagesRead);
    
    /* Send read page command sequence */
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_A;
   
    /* Cards with page size <= 512 bytes */
    if((hnand->Config.PageSize) <= 512U)
    {
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
    else /* (hnand->Config.PageSize) > 512 */
    {
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
  
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA))  = NAND_CMD_AREA_TRUE1;
      
    /* Check if an extra command is needed for reading pages  */
    if(hnand->Config.ExtraCommandEnable == ENABLE)
    {
      /* Get tick */
      tickstart = HAL_GetTick();
      
      /* Read status until NAND is ready */
      while(HAL_NAND_Read_Status(hnand) != NAND_READY)
      {
        if((HAL_GetTick() - tickstart ) > NAND_WRITE_TIMEOUT)
        {
          return HAL_TIMEOUT; 
        }
      }
      
      /* Go back to read mode */
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = ((uint8_t)0x00);
      __DSB();
    }
    
    /* Get Data into Buffer */    
    for(; index < size; index++)
    {
      *(uint8_t *)pBuffer++ = *(uint8_t *)deviceaddress;
    }
    
    /* Increment read pages number */
    numPagesRead++;
    
    /* Decrement pages to read */
    NumPageToRead--;
    
    /* Increment the NAND address */
    nandaddress = (uint32_t)(nandaddress + 1U);
  }
  
  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_READY;
  
  /* Process unlocked */
  __HAL_UNLOCK(hnand);

  return HAL_OK;
}

/**
  * @brief  Read Page(s) from NAND memory block (16-bits addressing)
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pAddress  pointer to NAND address structure
  * @param  pBuffer  pointer to destination read buffer. pBuffer should be 16bits aligned
  * @param  NumPageToRead  number of pages to read from block 
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_Read_Page_16b(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress, uint16_t *pBuffer, uint32_t NumPageToRead)
{   
  __IO uint32_t index  = 0U;
  uint32_t tickstart = 0U;
  uint32_t deviceaddress = 0U, size = 0U, numPagesRead = 0U, nandaddress = 0U;
  
  /* Process Locked */
  __HAL_LOCK(hnand); 
  
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }

  /* Update the NAND controller state */ 
  hnand->State = HAL_NAND_STATE_BUSY;
  
  /* NAND raw address calculation */
  nandaddress = ARRAY_ADDRESS(pAddress, hnand);
  
  /* Page(s) read loop */  
  while((NumPageToRead != 0U) && (nandaddress < ((hnand->Config.BlockSize) * (hnand->Config.BlockNbr))))
  {
    /* update the buffer size */
    size = (hnand->Config.PageSize) + ((hnand->Config.PageSize) * numPagesRead);
    
    /* Send read page command sequence */
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_A;  
    __DSB();
    
    /* Cards with page size <= 512 bytes */
    if((hnand->Config.PageSize) <= 512U)
    {
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
    else /* (hnand->Config.PageSize) > 512 */
    {
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
  
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA))  = NAND_CMD_AREA_TRUE1;
    
    if(hnand->Config.ExtraCommandEnable == ENABLE)
    {
      /* Get tick */
      tickstart = HAL_GetTick();
      
      /* Read status until NAND is ready */
      while(HAL_NAND_Read_Status(hnand) != NAND_READY)
      {
        if((HAL_GetTick() - tickstart ) > NAND_WRITE_TIMEOUT)
        {
          return HAL_TIMEOUT; 
        }
      }
      
      /* Go back to read mode */
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = ((uint8_t)0x00);
    }

    /* Calculate PageSize */
#ifdef FSMC_PCR2_PWID
    if (hnand->Init.MemoryDataWidth == FSMC_NAND_PCC_MEM_BUS_WIDTH_8)
#else /* FMC_PCR2_PWID is defined */
    if (hnand->Init.MemoryDataWidth == FMC_NAND_PCC_MEM_BUS_WIDTH_8)
#endif
    {
      size = size / 2U;
    }
    else
    {
      /* Do nothing */
      /* Keep the same PageSize for FMC_NAND_MEM_BUS_WIDTH_16*/
    }
    
    /* Get Data into Buffer */    
    for(; index < size; index++)
    {
      *(uint16_t *)pBuffer++ = *(uint16_t *)deviceaddress;
    }
    
    /* Increment read pages number */
    numPagesRead++;
    
    /* Decrement pages to read */
    NumPageToRead--;
    
    /* Increment the NAND address */
    nandaddress = (uint32_t)(nandaddress + 1U);
  }
  
  /* Update the NAND controller state */ 
  hnand->State = HAL_NAND_STATE_READY;
  
  /* Process unlocked */
  __HAL_UNLOCK(hnand);  
    
  return HAL_OK;
}

/**
  * @brief  Write Page(s) to NAND memory block (8-bits addressing)
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pAddress  pointer to NAND address structure
  * @param  pBuffer  pointer to source buffer to write  
  * @param  NumPageToWrite   number of pages to write to block 
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_Write_Page_8b(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress, uint8_t *pBuffer, uint32_t NumPageToWrite)
{
  __IO uint32_t index = 0U;
  uint32_t tickstart = 0U;
  uint32_t deviceaddress = 0U, size = 0U, numPagesWritten = 0U, nandaddress = 0U;
  
  /* Process Locked */
  __HAL_LOCK(hnand);  

  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }
  
  /* Update the NAND controller state */ 
  hnand->State = HAL_NAND_STATE_BUSY;
  
  /* NAND raw address calculation */
  nandaddress = ARRAY_ADDRESS(pAddress, hnand);
    
  /* Page(s) write loop */
  while((NumPageToWrite != 0U) && (nandaddress < ((hnand->Config.BlockSize) * (hnand->Config.BlockNbr))))
  {
    /* update the buffer size */
    size = hnand->Config.PageSize + ((hnand->Config.PageSize) * numPagesWritten);
    
    /* Send write page command sequence */
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_A;
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE0;

    /* Cards with page size <= 512 bytes */
    if((hnand->Config.PageSize) <= 512U)
    {
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
    else /* (hnand->Config.PageSize) > 512 */
    {
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        __DSB();
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
        __DSB();
      }
    }
  

    /* Write data to memory */
    for(; index < size; index++)
    {
      *(__IO uint8_t *)deviceaddress = *(uint8_t *)pBuffer++;
    }
   
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE_TRUE1;

    /* Get tick */
    tickstart = HAL_GetTick();

    /* Read status until NAND is ready */
    while(HAL_NAND_Read_Status(hnand) != NAND_READY)
    {
      
      if((HAL_GetTick() - tickstart ) > NAND_WRITE_TIMEOUT)
      {
        return HAL_TIMEOUT; 
      }
    }
 
    /* Increment written pages number */
    numPagesWritten++;
    
    /* Decrement pages to write */
    NumPageToWrite--;
    
    /* Increment the NAND address */
    nandaddress = (uint32_t)(nandaddress + 1U);
  }
  
  /* Update the NAND controller state */ 
  hnand->State = HAL_NAND_STATE_READY;
  
  /* Process unlocked */
  __HAL_UNLOCK(hnand);
  
  return HAL_OK;
}

/**
  * @brief  Write Page(s) to NAND memory block (16-bits addressing)
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pAddress  pointer to NAND address structure
  * @param  pBuffer  pointer to source buffer to write. pBuffer should be 16bits aligned
  * @param  NumPageToWrite   number of pages to write to block 
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_Write_Page_16b(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress, uint16_t *pBuffer, uint32_t NumPageToWrite)
{
  __IO uint32_t index = 0U;
  uint32_t tickstart = 0U;
  uint32_t deviceaddress = 0U, size = 0U, numPagesWritten = 0U, nandaddress = 0U;
  
  /* Process Locked */
  __HAL_LOCK(hnand);  

  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }
  
  /* Update the NAND controller state */ 
  hnand->State = HAL_NAND_STATE_BUSY;
  
  /* NAND raw address calculation */
  nandaddress = ARRAY_ADDRESS(pAddress, hnand);
  
  /* Page(s) write loop */
  while((NumPageToWrite != 0U) && (nandaddress < ((hnand->Config.BlockSize) * (hnand->Config.BlockNbr))))
  {
    /* update the buffer size */
    size = (hnand->Config.PageSize) + ((hnand->Config.PageSize) * numPagesWritten);
 
    /* Send write page command sequence */
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_A;
    __DSB();
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE0;
    __DSB();

    /* Cards with page size <= 512 bytes */
    if((hnand->Config.PageSize) <= 512U)
    {
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
    else /* (hnand->Config.PageSize) > 512 */
    {
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }

    /* Calculate PageSize */
#ifdef FSMC_PCR2_PWID
    if (hnand->Init.MemoryDataWidth == FSMC_NAND_PCC_MEM_BUS_WIDTH_8)
#else /* FMC_PCR2_PWID is defined */
    if (hnand->Init.MemoryDataWidth == FMC_NAND_PCC_MEM_BUS_WIDTH_8)
#endif
    {
      size = size / 2U;
    }
    else
    {
      /* Do nothing */
      /* Keep the same PageSize for FMC_NAND_MEM_BUS_WIDTH_16*/
    }
  
    /* Write data to memory */
    for(; index < size; index++)
    {
      *(__IO uint16_t *)deviceaddress = *(uint16_t *)pBuffer++;
    }
   
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE_TRUE1;
    
    /* Get tick */
    tickstart = HAL_GetTick();

    /* Read status until NAND is ready */
    while(HAL_NAND_Read_Status(hnand) != NAND_READY)
    { 
      if((HAL_GetTick() - tickstart ) > NAND_WRITE_TIMEOUT)
      {
        return HAL_TIMEOUT; 
      } 
    }   
 
    /* Increment written pages number */
    numPagesWritten++;
    
    /* Decrement pages to write */
    NumPageToWrite--;
    
    /* Increment the NAND address */
    nandaddress = (uint32_t)(nandaddress + 1U);
  }
  
  /* Update the NAND controller state */ 
  hnand->State = HAL_NAND_STATE_READY;
  
  /* Process unlocked */
  __HAL_UNLOCK(hnand);      
  
  return HAL_OK;
}

/**
  * @brief  Read Spare area(s) from NAND memory 
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pAddress  pointer to NAND address structure
  * @param  pBuffer pointer to source buffer to write  
  * @param  NumSpareAreaToRead Number of spare area to read  
  * @retval HAL status
*/
HAL_StatusTypeDef HAL_NAND_Read_SpareArea_8b(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress, uint8_t *pBuffer, uint32_t NumSpareAreaToRead)
{
  __IO uint32_t index = 0U;
  uint32_t tickstart = 0U;
  uint32_t deviceaddress = 0U, size = 0U, numSpareAreaRead = 0U, nandaddress = 0U, columnaddress = 0U;
  
  /* Process Locked */
  __HAL_LOCK(hnand);  
  
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }
  
  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_BUSY;
  
  /* NAND raw address calculation */
  nandaddress = ARRAY_ADDRESS(pAddress, hnand);
  
  /* Column in page address */
  columnaddress = COLUMN_ADDRESS(hnand);
  
  /* Spare area(s) read loop */ 
  while((NumSpareAreaToRead != 0U) && (nandaddress < ((hnand->Config.BlockSize) * (hnand->Config.BlockNbr))))
  {     
    /* update the buffer size */
    size = (hnand->Config.SpareAreaSize) + ((hnand->Config.SpareAreaSize) * numSpareAreaRead);

    /* Cards with page size <= 512 bytes */
    if((hnand->Config.PageSize) <= 512U)
    {
      /* Send read spare area command sequence */     
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_C;
      
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
    else /* (hnand->Config.PageSize) > 512 */
    {
      /* Send read spare area command sequence */ 
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_A;
      
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_1ST_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_2ND_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_1ST_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_2ND_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }

    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_TRUE1;
    
    if(hnand->Config.ExtraCommandEnable == ENABLE)
    {
      /* Get tick */
      tickstart = HAL_GetTick();
      
      /* Read status until NAND is ready */
      while(HAL_NAND_Read_Status(hnand) != NAND_READY)
      {
        if((HAL_GetTick() - tickstart ) > NAND_WRITE_TIMEOUT)
        {
          return HAL_TIMEOUT; 
        }
      }
      
      /* Go back to read mode */
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = ((uint8_t)0x00);
    }
    
    /* Get Data into Buffer */
    for(; index < size; index++)
    {
      *(uint8_t *)pBuffer++ = *(uint8_t *)deviceaddress;
    }
    
    /* Increment read spare areas number */
    numSpareAreaRead++;
    
    /* Decrement spare areas to read */
    NumSpareAreaToRead--;
    
    /* Increment the NAND address */
    nandaddress = (uint32_t)(nandaddress + 1U);
  }
  
  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_READY;
  
  /* Process unlocked */
  __HAL_UNLOCK(hnand);

  return HAL_OK;  
}

/**
  * @brief  Read Spare area(s) from NAND memory (16-bits addressing)
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pAddress  pointer to NAND address structure
  * @param  pBuffer pointer to source buffer to write. pBuffer should be 16bits aligned.
  * @param  NumSpareAreaToRead Number of spare area to read  
  * @retval HAL status
*/
HAL_StatusTypeDef HAL_NAND_Read_SpareArea_16b(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress, uint16_t *pBuffer, uint32_t NumSpareAreaToRead)
{
  __IO uint32_t index = 0U; 
  uint32_t tickstart = 0U;
  uint32_t deviceaddress = 0U, size = 0U, numSpareAreaRead = 0U, nandaddress = 0U, columnaddress = 0U;
  
  /* Process Locked */
  __HAL_LOCK(hnand);
  
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }
  
  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_BUSY;
  
  /* NAND raw address calculation */
  nandaddress = ARRAY_ADDRESS(pAddress, hnand);
  
  /* Column in page address */
  columnaddress = (uint32_t)(COLUMN_ADDRESS(hnand));
  
  /* Spare area(s) read loop */ 
  while((NumSpareAreaToRead != 0U) && (nandaddress < ((hnand->Config.BlockSize) * (hnand->Config.BlockNbr))))
  {
    /* update the buffer size */
    size = (hnand->Config.SpareAreaSize) + ((hnand->Config.SpareAreaSize) * numSpareAreaRead);

    /* Cards with page size <= 512 bytes */
    if((hnand->Config.PageSize) <= 512U)
    {
      /* Send read spare area command sequence */     
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_C;
      
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
    else /* (hnand->Config.PageSize) > 512 */
    {
      /* Send read spare area command sequence */     
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_A;
      
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_1ST_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_2ND_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_1ST_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_2ND_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }

    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_TRUE1;

    if(hnand->Config.ExtraCommandEnable == ENABLE)
    {
      /* Get tick */
      tickstart = HAL_GetTick();
      
      /* Read status until NAND is ready */
      while(HAL_NAND_Read_Status(hnand) != NAND_READY)
      {
        if((HAL_GetTick() - tickstart ) > NAND_WRITE_TIMEOUT)
        {
          return HAL_TIMEOUT; 
        }
      }
      
      /* Go back to read mode */
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = ((uint8_t)0x00);
    }
    
    /* Get Data into Buffer */
    for(; index < size; index++)
    {
      *(uint16_t *)pBuffer++ = *(uint16_t *)deviceaddress;
    }
    
    /* Increment read spare areas number */
    numSpareAreaRead++;
    
    /* Decrement spare areas to read */
    NumSpareAreaToRead--;
    
    /* Increment the NAND address */
    nandaddress = (uint32_t)(nandaddress + 1U);
  }
  
  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_READY;
  
  /* Process unlocked */
  __HAL_UNLOCK(hnand);     

  return HAL_OK;  
}

/**
  * @brief  Write Spare area(s) to NAND memory 
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pAddress  pointer to NAND address structure
  * @param  pBuffer  pointer to source buffer to write  
  * @param  NumSpareAreaTowrite   number of spare areas to write to block
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_Write_SpareArea_8b(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress, uint8_t *pBuffer, uint32_t NumSpareAreaTowrite)
{
  __IO uint32_t index = 0U;
  uint32_t tickstart = 0U;
  uint32_t deviceaddress = 0U, size = 0U, numSpareAreaWritten = 0U, nandaddress = 0U, columnaddress = 0U;

  /* Process Locked */
  __HAL_LOCK(hnand); 
  
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }
  
  /* Update the FMC_NAND controller state */
  hnand->State = HAL_NAND_STATE_BUSY;  
  
  /* Page address calculation */
  nandaddress = ARRAY_ADDRESS(pAddress, hnand); 
  
  /* Column in page address */
  columnaddress = COLUMN_ADDRESS(hnand);
  
  /* Spare area(s) write loop */
  while((NumSpareAreaTowrite != 0U) && (nandaddress < ((hnand->Config.BlockSize) * (hnand->Config.BlockNbr))))
  {
    /* update the buffer size */
    size = (hnand->Config.SpareAreaSize) + ((hnand->Config.SpareAreaSize) * numSpareAreaWritten);

    /* Cards with page size <= 512 bytes */
    if((hnand->Config.PageSize) <= 512U)
    {
      /* Send write Spare area command sequence */
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_C;
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE0;
      
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
    else /* (hnand->Config.PageSize) > 512 */
    {
      /* Send write Spare area command sequence */
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_A;
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE0;
    
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_1ST_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_2ND_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_1ST_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_2ND_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
  
    /* Write data to memory */
    for(; index < size; index++)
    {
      *(__IO uint8_t *)deviceaddress = *(uint8_t *)pBuffer++;
    }
   
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE_TRUE1;
    
    /* Get tick */
    tickstart = HAL_GetTick();
    
    /* Read status until NAND is ready */
    while(HAL_NAND_Read_Status(hnand) != NAND_READY)
    {
      if((HAL_GetTick() - tickstart ) > NAND_WRITE_TIMEOUT)
      {
        return HAL_TIMEOUT; 
      }
    }

    /* Increment written spare areas number */
    numSpareAreaWritten++;
    
    /* Decrement spare areas to write */
    NumSpareAreaTowrite--;
    
    /* Increment the NAND address */
    nandaddress = (uint32_t)(nandaddress + 1U);
  }

  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_READY;

  /* Process unlocked */
  __HAL_UNLOCK(hnand);
    
  return HAL_OK;
}

/**
  * @brief  Write Spare area(s) to NAND memory (16-bits addressing)
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pAddress  pointer to NAND address structure
  * @param  pBuffer  pointer to source buffer to write. pBuffer should be 16bits aligned.  
  * @param  NumSpareAreaTowrite   number of spare areas to write to block
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_Write_SpareArea_16b(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress, uint16_t *pBuffer, uint32_t NumSpareAreaTowrite)
{
  __IO uint32_t index = 0U;
  uint32_t tickstart = 0U;
  uint32_t deviceaddress = 0U, size = 0U, numSpareAreaWritten = 0U, nandaddress = 0U, columnaddress = 0U;

  /* Process Locked */
  __HAL_LOCK(hnand); 
  
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }
  
  /* Update the FMC_NAND controller state */
  hnand->State = HAL_NAND_STATE_BUSY;  
  
  /* NAND raw address calculation */
  nandaddress = ARRAY_ADDRESS(pAddress, hnand);
  
  /* Column in page address */
  columnaddress = (uint32_t)(COLUMN_ADDRESS(hnand));
  
  /* Spare area(s) write loop */
  while((NumSpareAreaTowrite != 0U) && (nandaddress < ((hnand->Config.BlockSize) * (hnand->Config.BlockNbr))))
  {
    /* update the buffer size */
    size = (hnand->Config.SpareAreaSize) + ((hnand->Config.SpareAreaSize) * numSpareAreaWritten);

    /* Cards with page size <= 512 bytes */
    if((hnand->Config.PageSize) <= 512U)
    {
      /* Send write Spare area command sequence */
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_C;
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE0;
    
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = 0x00;
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
    else /* (hnand->Config.PageSize) > 512 */
    {
      /* Send write Spare area command sequence */
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_AREA_A;
      *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE0;
    
      if (((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) <= 65535U)
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_1ST_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_2ND_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
      }
      else /* ((hnand->Config.BlockSize)*(hnand->Config.BlockNbr)) > 65535 */
      {
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_1ST_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = COLUMN_2ND_CYCLE(columnaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(nandaddress);
        *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(nandaddress);
      }
    }
  
    /* Write data to memory */
    for(; index < size; index++)
    {
      *(__IO uint16_t *)deviceaddress = *(uint16_t *)pBuffer++;
    }
   
    *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_WRITE_TRUE1;
   
    /* Get tick */
    tickstart = HAL_GetTick();

    /* Read status until NAND is ready */
    while(HAL_NAND_Read_Status(hnand) != NAND_READY)
    {    
      if((HAL_GetTick() - tickstart ) > NAND_WRITE_TIMEOUT)
      {
        return HAL_TIMEOUT; 
      }
    }

    /* Increment written spare areas number */
    numSpareAreaWritten++;
    
    /* Decrement spare areas to write */
    NumSpareAreaTowrite--;
    
    /* Increment the NAND address */
    nandaddress = (uint32_t)(nandaddress + 1U);
  }

  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_READY;

  /* Process unlocked */
  __HAL_UNLOCK(hnand);

  return HAL_OK;  
}

/**
  * @brief  NAND memory Block erase 
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  pAddress  pointer to NAND address structure
  * @retval HAL status
  */
HAL_StatusTypeDef HAL_NAND_Erase_Block(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress)
{
  uint32_t deviceaddress = 0U;
  uint32_t tickstart = 0U;
  
  /* Process Locked */
  __HAL_LOCK(hnand);
  
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  }
  
  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_BUSY;  
  
  /* Send Erase block command sequence */
  *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_ERASE0;

  *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_1ST_CYCLE(ARRAY_ADDRESS(pAddress, hnand));
  *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_2ND_CYCLE(ARRAY_ADDRESS(pAddress, hnand));
  *(__IO uint8_t *)((uint32_t)(deviceaddress | ADDR_AREA)) = ADDR_3RD_CYCLE(ARRAY_ADDRESS(pAddress, hnand));
    
  *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_ERASE1; 
  
  /* Update the NAND controller state */
  hnand->State = HAL_NAND_STATE_READY;
  
  /* Get tick */
  tickstart = HAL_GetTick();
  
  /* Read status until NAND is ready */
  while(HAL_NAND_Read_Status(hnand) != NAND_READY)
  {
    if((HAL_GetTick() - tickstart ) > NAND_WRITE_TIMEOUT)
    {
      /* Process unlocked */
      __HAL_UNLOCK(hnand);    
  
      return HAL_TIMEOUT; 
    } 
  }    
 
  /* Process unlocked */
  __HAL_UNLOCK(hnand);    
  
  return HAL_OK;  
}

/**
  * @brief  NAND memory read status 
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval NAND status
  */
uint32_t HAL_NAND_Read_Status(NAND_HandleTypeDef *hnand)
{
  uint32_t data = 0U;
  uint32_t deviceaddress = 0U;
  
  /* Identify the device address */
  if(hnand->Init.NandBank == FMC_NAND_BANK2)
  {
    deviceaddress = NAND_DEVICE1;
  }
  else
  {
    deviceaddress = NAND_DEVICE2;
  } 

  /* Send Read status operation command */
  *(__IO uint8_t *)((uint32_t)(deviceaddress | CMD_AREA)) = NAND_CMD_STATUS;
  
  /* Read status register data */
  data = *(__IO uint8_t *)deviceaddress;

  /* Return the status */
  if((data & NAND_ERROR) == NAND_ERROR)
  {
    return NAND_ERROR;
  } 
  else if((data & NAND_READY) == NAND_READY)
  {
    return NAND_READY;
  }

  return NAND_BUSY; 
}

/**
  * @brief  Increment the NAND memory address
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param pAddress pointer to NAND address structure
  * @retval The new status of the increment address operation. It can be:
  *           - NAND_VALID_ADDRESS: When the new address is valid address
  *           - NAND_INVALID_ADDRESS: When the new address is invalid address
  */
uint32_t HAL_NAND_Address_Inc(NAND_HandleTypeDef *hnand, NAND_AddressTypeDef *pAddress)
{
  uint32_t status = NAND_VALID_ADDRESS;
 
  /* Increment page address */
  pAddress->Page++;

  /* Check NAND address is valid */
  if(pAddress->Page == hnand->Config.BlockSize)
  {
    pAddress->Page = 0U;
    pAddress->Block++;
    
    if(pAddress->Block == hnand->Config.PlaneSize)
    {
      pAddress->Block = 0U;
      pAddress->Plane++;

      if(pAddress->Plane == (hnand->Config.PlaneNbr))
      {
        status = NAND_INVALID_ADDRESS;
      }
    }
  } 
  
  return (status);
}

#if (USE_HAL_NAND_REGISTER_CALLBACKS == 1)
/**
  * @brief  Register a User NAND Callback
  *         To be used instead of the weak (surcharged) predefined callback
  * @param hnand : NAND handle
  * @param CallbackId : ID of the callback to be registered
  *        This parameter can be one of the following values:
  *          @arg @ref HAL_NAND_MSP_INIT_CB_ID       NAND MspInit callback ID
  *          @arg @ref HAL_NAND_MSP_DEINIT_CB_ID     NAND MspDeInit callback ID
  *          @arg @ref HAL_NAND_IT_CB_ID             NAND IT callback ID
  * @param pCallback : pointer to the Callback function
  * @retval status
  */
HAL_StatusTypeDef HAL_NAND_RegisterCallback (NAND_HandleTypeDef *hnand, HAL_NAND_CallbackIDTypeDef CallbackId, pNAND_CallbackTypeDef pCallback)
{
  HAL_StatusTypeDef status = HAL_OK;

  if(pCallback == NULL)
  {
    return HAL_ERROR;
  }

  /* Process locked */
  __HAL_LOCK(hnand);

  if(hnand->State == HAL_NAND_STATE_READY)
  {
    switch (CallbackId)
    {
    case HAL_NAND_MSP_INIT_CB_ID :
      hnand->MspInitCallback = pCallback;
      break;
    case HAL_NAND_MSP_DEINIT_CB_ID :
      hnand->MspDeInitCallback = pCallback;
      break;
    case HAL_NAND_IT_CB_ID :
      hnand->ItCallback = pCallback;
      break;
    default :
      /* update return status */
      status =  HAL_ERROR;
      break;
    }
  }
  else if(hnand->State == HAL_NAND_STATE_RESET)
  {
    switch (CallbackId)
    {
    case HAL_NAND_MSP_INIT_CB_ID :
      hnand->MspInitCallback = pCallback;
      break;
    case HAL_NAND_MSP_DEINIT_CB_ID :
      hnand->MspDeInitCallback = pCallback;
      break;
    default :
      /* update return status */
      status =  HAL_ERROR;
      break;
    }
  }
  else
  {
    /* update return status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hnand);
  return status;
}

/**
  * @brief  Unregister a User NAND Callback
  *         NAND Callback is redirected to the weak (surcharged) predefined callback
  * @param hnand : NAND handle
  * @param CallbackId : ID of the callback to be unregistered
  *        This parameter can be one of the following values:
  *          @arg @ref HAL_NAND_MSP_INIT_CB_ID       NAND MspInit callback ID
  *          @arg @ref HAL_NAND_MSP_DEINIT_CB_ID     NAND MspDeInit callback ID
  *          @arg @ref HAL_NAND_IT_CB_ID             NAND IT callback ID
  * @retval status
  */
HAL_StatusTypeDef HAL_NAND_UnRegisterCallback (NAND_HandleTypeDef *hnand, HAL_NAND_CallbackIDTypeDef CallbackId)
{
  HAL_StatusTypeDef status = HAL_OK;

  /* Process locked */
  __HAL_LOCK(hnand);

  if(hnand->State == HAL_NAND_STATE_READY)
  {
    switch (CallbackId)
    {
    case HAL_NAND_MSP_INIT_CB_ID :
      hnand->MspInitCallback = HAL_NAND_MspInit;
      break;
    case HAL_NAND_MSP_DEINIT_CB_ID :
      hnand->MspDeInitCallback = HAL_NAND_MspDeInit;
      break;
    case HAL_NAND_IT_CB_ID :
      hnand->ItCallback = HAL_NAND_ITCallback;
      break;
    default :
      /* update return status */
      status =  HAL_ERROR;
      break;
    }
  }
  else if(hnand->State == HAL_NAND_STATE_RESET)
  {
    switch (CallbackId)
    {
    case HAL_NAND_MSP_INIT_CB_ID :
      hnand->MspInitCallback = HAL_NAND_MspInit;
      break;
    case HAL_NAND_MSP_DEINIT_CB_ID :
      hnand->MspDeInitCallback = HAL_NAND_MspDeInit;
      break;
    default :
      /* update return status */
      status =  HAL_ERROR;
      break;
    }
  }
  else
  {
    /* update return status */
    status =  HAL_ERROR;
  }

  /* Release Lock */
  __HAL_UNLOCK(hnand);
  return status;
}
#endif

/**
  * @}
  */

/** @defgroup NAND_Exported_Functions_Group3 Peripheral Control functions 
 *  @brief   management functions 
 *
@verbatim   
  ==============================================================================
                         ##### NAND Control functions #####
  ==============================================================================  
  [..]
    This subsection provides a set of functions allowing to control dynamically
    the NAND interface.

@endverbatim
  * @{
  */ 

    
/**
  * @brief  Enables dynamically NAND ECC feature.
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval HAL status
  */    
HAL_StatusTypeDef  HAL_NAND_ECC_Enable(NAND_HandleTypeDef *hnand)
{
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }

  /* Update the NAND state */
  hnand->State = HAL_NAND_STATE_BUSY;
   
  /* Enable ECC feature */
  FMC_NAND_ECC_Enable(hnand->Instance, hnand->Init.NandBank);
  
  /* Update the NAND state */
  hnand->State = HAL_NAND_STATE_READY;
  
  return HAL_OK;
}

/**
  * @brief  Disables dynamically FMC_NAND ECC feature.
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval HAL status
  */  
HAL_StatusTypeDef  HAL_NAND_ECC_Disable(NAND_HandleTypeDef *hnand)
{
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }

  /* Update the NAND state */
  hnand->State = HAL_NAND_STATE_BUSY;
    
  /* Disable ECC feature */
  FMC_NAND_ECC_Disable(hnand->Instance, hnand->Init.NandBank);
  
  /* Update the NAND state */
  hnand->State = HAL_NAND_STATE_READY;
  
  return HAL_OK;  
}

/**
  * @brief  Disables dynamically NAND ECC feature.
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @param  ECCval pointer to ECC value 
  * @param  Timeout maximum timeout to wait    
  * @retval HAL status
  */
HAL_StatusTypeDef  HAL_NAND_GetECC(NAND_HandleTypeDef *hnand, uint32_t *ECCval, uint32_t Timeout)
{
  HAL_StatusTypeDef status = HAL_OK;
  
  /* Check the NAND controller state */
  if(hnand->State == HAL_NAND_STATE_BUSY)
  {
     return HAL_BUSY;
  }
  
  /* Update the NAND state */
  hnand->State = HAL_NAND_STATE_BUSY;  
   
  /* Get NAND ECC value */
  status = FMC_NAND_GetECC(hnand->Instance, ECCval, hnand->Init.NandBank, Timeout);
  
  /* Update the NAND state */
  hnand->State = HAL_NAND_STATE_READY;

  return status;  
}

/**
  * @}
  */
  
    
/** @defgroup NAND_Exported_Functions_Group4 Peripheral State functions 
 *  @brief   Peripheral State functions 
 *
@verbatim   
  ==============================================================================
                         ##### NAND State functions #####
  ==============================================================================  
  [..]
    This subsection permits to get in run-time the status of the NAND controller 
    and the data flow.

@endverbatim
  * @{
  */
  
/**
  * @brief  return the NAND state
  * @param  hnand pointer to a NAND_HandleTypeDef structure that contains
  *                the configuration information for NAND module.
  * @retval HAL state
  */
HAL_NAND_StateTypeDef HAL_NAND_GetState(NAND_HandleTypeDef *hnand)
{
  return hnand->State;
}

/**
  * @}
  */  

/**
  * @}
  */

/**
  * @}
  */

#endif /* STM32F405xx || STM32F415xx || STM32F407xx || STM32F417xx ||\
          STM32F427xx || STM32F437xx || STM32F429xx || STM32F439xx ||\
          STM32F446xx || STM32F469xx || STM32F479xx */
#endif /* HAL_NAND_MODULE_ENABLED  */

/**
  * @}
  */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
