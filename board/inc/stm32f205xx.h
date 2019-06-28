/**
  ******************************************************************************
  * @file    stm32f205xx.h
  * @author  MCD Application Team
  * @version V2.1.2
  * @date    29-June-2016
  * @brief   CMSIS STM32F205xx Device Peripheral Access Layer Header File. 
  *          This file contains :  
  *           - Data structures and the address mapping for all peripherals
  *           - Peripherals registers declarations and bits definition
  *           - Macros to access peripheral’s registers hardware
  *
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT(c) 2016 STMicroelectronics</center></h2>
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

/** @addtogroup CMSIS
  * @{
  */

/** @addtogroup stm32f205xx
  * @{
  */
    
#ifndef __STM32F205xx_H
#define __STM32F205xx_H

#ifdef __cplusplus
 extern "C" {
#endif /* __cplusplus */
  

/** @addtogroup Configuration_section_for_CMSIS
  * @{
  */

/**
  * @brief Configuration of the Cortex-M3 Processor and Core Peripherals 
  */
#define __CM3_REV                 0x0200U  /*!< Core revision r0p1                            */
#define __MPU_PRESENT             1U       /*!< STM32F2XX provides an MPU                     */
#define __NVIC_PRIO_BITS          4U       /*!< STM32F2XX uses 4 Bits for the Priority Levels */
#define __Vendor_SysTickConfig    0U       /*!< Set to 1 if different SysTick Config is used  */

/**
  * @}
  */
   
/** @addtogroup Peripheral_interrupt_number_definition
  * @{
  */

/**
 * @brief STM32F2XX Interrupt Number Definition, according to the selected device 
 *        in @ref Library_configuration_section 
 */
typedef enum
{
/******  Cortex-M3 Processor Exceptions Numbers ****************************************************************/
  NonMaskableInt_IRQn         = -14,    /*!< 2 Non Maskable Interrupt                                          */
  MemoryManagement_IRQn       = -12,    /*!< 4 Cortex-M3 Memory Management Interrupt                           */
  BusFault_IRQn               = -11,    /*!< 5 Cortex-M3 Bus Fault Interrupt                                   */
  UsageFault_IRQn             = -10,    /*!< 6 Cortex-M3 Usage Fault Interrupt                                 */
  SVCall_IRQn                 = -5,     /*!< 11 Cortex-M3 SV Call Interrupt                                    */
  DebugMonitor_IRQn           = -4,     /*!< 12 Cortex-M3 Debug Monitor Interrupt                              */
  PendSV_IRQn                 = -2,     /*!< 14 Cortex-M3 Pend SV Interrupt                                    */
  SysTick_IRQn                = -1,     /*!< 15 Cortex-M3 System Tick Interrupt                                */
/******  STM32 specific Interrupt Numbers **********************************************************************/
  WWDG_IRQn                   = 0,      /*!< Window WatchDog Interrupt                                         */
  PVD_IRQn                    = 1,      /*!< PVD through EXTI Line detection Interrupt                         */
  TAMP_STAMP_IRQn             = 2,      /*!< Tamper and TimeStamp interrupts through the EXTI line             */
  RTC_WKUP_IRQn               = 3,      /*!< RTC Wakeup interrupt through the EXTI line                        */
  FLASH_IRQn                  = 4,      /*!< FLASH global Interrupt                                            */
  RCC_IRQn                    = 5,      /*!< RCC global Interrupt                                              */
  EXTI0_IRQn                  = 6,      /*!< EXTI Line0 Interrupt                                              */
  EXTI1_IRQn                  = 7,      /*!< EXTI Line1 Interrupt                                              */
  EXTI2_IRQn                  = 8,      /*!< EXTI Line2 Interrupt                                              */
  EXTI3_IRQn                  = 9,      /*!< EXTI Line3 Interrupt                                              */
  EXTI4_IRQn                  = 10,     /*!< EXTI Line4 Interrupt                                              */
  DMA1_Stream0_IRQn           = 11,     /*!< DMA1 Stream 0 global Interrupt                                    */
  DMA1_Stream1_IRQn           = 12,     /*!< DMA1 Stream 1 global Interrupt                                    */
  DMA1_Stream2_IRQn           = 13,     /*!< DMA1 Stream 2 global Interrupt                                    */
  DMA1_Stream3_IRQn           = 14,     /*!< DMA1 Stream 3 global Interrupt                                    */
  DMA1_Stream4_IRQn           = 15,     /*!< DMA1 Stream 4 global Interrupt                                    */
  DMA1_Stream5_IRQn           = 16,     /*!< DMA1 Stream 5 global Interrupt                                    */
  DMA1_Stream6_IRQn           = 17,     /*!< DMA1 Stream 6 global Interrupt                                    */
  ADC_IRQn                    = 18,     /*!< ADC1, ADC2 and ADC3 global Interrupts                             */
  CAN1_TX_IRQn                = 19,     /*!< CAN1 TX Interrupt                                                 */
  CAN1_RX0_IRQn               = 20,     /*!< CAN1 RX0 Interrupt                                                */
  CAN1_RX1_IRQn               = 21,     /*!< CAN1 RX1 Interrupt                                                */
  CAN1_SCE_IRQn               = 22,     /*!< CAN1 SCE Interrupt                                                */
  EXTI9_5_IRQn                = 23,     /*!< External Line[9:5] Interrupts                                     */
  TIM1_BRK_TIM9_IRQn          = 24,     /*!< TIM1 Break interrupt and TIM9 global interrupt                    */
  TIM1_UP_TIM10_IRQn          = 25,     /*!< TIM1 Update Interrupt and TIM10 global interrupt                  */
  TIM1_TRG_COM_TIM11_IRQn     = 26,     /*!< TIM1 Trigger and Commutation Interrupt and TIM11 global interrupt */
  TIM1_CC_IRQn                = 27,     /*!< TIM1 Capture Compare Interrupt                                    */
  TIM2_IRQn                   = 28,     /*!< TIM2 global Interrupt                                             */
  TIM3_IRQn                   = 29,     /*!< TIM3 global Interrupt                                             */
  TIM4_IRQn                   = 30,     /*!< TIM4 global Interrupt                                             */
  I2C1_EV_IRQn                = 31,     /*!< I2C1 Event Interrupt                                              */
  I2C1_ER_IRQn                = 32,     /*!< I2C1 Error Interrupt                                              */
  I2C2_EV_IRQn                = 33,     /*!< I2C2 Event Interrupt                                              */
  I2C2_ER_IRQn                = 34,     /*!< I2C2 Error Interrupt                                              */  
  SPI1_IRQn                   = 35,     /*!< SPI1 global Interrupt                                             */
  SPI2_IRQn                   = 36,     /*!< SPI2 global Interrupt                                             */
  USART1_IRQn                 = 37,     /*!< USART1 global Interrupt                                           */
  USART2_IRQn                 = 38,     /*!< USART2 global Interrupt                                           */
  USART3_IRQn                 = 39,     /*!< USART3 global Interrupt                                           */
  EXTI15_10_IRQn              = 40,     /*!< External Line[15:10] Interrupts                                   */
  RTC_Alarm_IRQn              = 41,     /*!< RTC Alarm (A and B) through EXTI Line Interrupt                   */
  OTG_FS_WKUP_IRQn            = 42,     /*!< USB OTG FS Wakeup through EXTI line interrupt                     */    
  TIM8_BRK_TIM12_IRQn         = 43,     /*!< TIM8 Break Interrupt and TIM12 global interrupt                   */
  TIM8_UP_TIM13_IRQn          = 44,     /*!< TIM8 Update Interrupt and TIM13 global interrupt                  */
  TIM8_TRG_COM_TIM14_IRQn     = 45,     /*!< TIM8 Trigger and Commutation Interrupt and TIM14 global interrupt */
  TIM8_CC_IRQn                = 46,     /*!< TIM8 Capture Compare Interrupt                                    */
  DMA1_Stream7_IRQn           = 47,     /*!< DMA1 Stream7 Interrupt                                            */
  FSMC_IRQn                   = 48,     /*!< FSMC global Interrupt                                             */
  SDIO_IRQn                   = 49,     /*!< SDIO global Interrupt                                             */
  TIM5_IRQn                   = 50,     /*!< TIM5 global Interrupt                                             */
  SPI3_IRQn                   = 51,     /*!< SPI3 global Interrupt                                             */
  UART4_IRQn                  = 52,     /*!< UART4 global Interrupt                                            */
  UART5_IRQn                  = 53,     /*!< UART5 global Interrupt                                            */
  TIM6_DAC_IRQn               = 54,     /*!< TIM6 global and DAC1&2 underrun error  interrupts                 */
  TIM7_IRQn                   = 55,     /*!< TIM7 global interrupt                                             */
  DMA2_Stream0_IRQn           = 56,     /*!< DMA2 Stream 0 global Interrupt                                    */
  DMA2_Stream1_IRQn           = 57,     /*!< DMA2 Stream 1 global Interrupt                                    */
  DMA2_Stream2_IRQn           = 58,     /*!< DMA2 Stream 2 global Interrupt                                    */
  DMA2_Stream3_IRQn           = 59,     /*!< DMA2 Stream 3 global Interrupt                                    */
  DMA2_Stream4_IRQn           = 60,     /*!< DMA2 Stream 4 global Interrupt                                    */
  CAN2_TX_IRQn                = 63,     /*!< CAN2 TX Interrupt                                                 */
  CAN2_RX0_IRQn               = 64,     /*!< CAN2 RX0 Interrupt                                                */
  CAN2_RX1_IRQn               = 65,     /*!< CAN2 RX1 Interrupt                                                */
  CAN2_SCE_IRQn               = 66,     /*!< CAN2 SCE Interrupt                                                */
  OTG_FS_IRQn                 = 67,     /*!< USB OTG FS global Interrupt                                       */
  DMA2_Stream5_IRQn           = 68,     /*!< DMA2 Stream 5 global interrupt                                    */
  DMA2_Stream6_IRQn           = 69,     /*!< DMA2 Stream 6 global interrupt                                    */
  DMA2_Stream7_IRQn           = 70,     /*!< DMA2 Stream 7 global interrupt                                    */
  USART6_IRQn                 = 71,     /*!< USART6 global interrupt                                           */
  I2C3_EV_IRQn                = 72,     /*!< I2C3 event interrupt                                              */
  I2C3_ER_IRQn                = 73,     /*!< I2C3 error interrupt                                              */
  OTG_HS_EP1_OUT_IRQn         = 74,     /*!< USB OTG HS End Point 1 Out global interrupt                       */
  OTG_HS_EP1_IN_IRQn          = 75,     /*!< USB OTG HS End Point 1 In global interrupt                        */
  OTG_HS_WKUP_IRQn            = 76,     /*!< USB OTG HS Wakeup through EXTI interrupt                          */
  OTG_HS_IRQn                 = 77,     /*!< USB OTG HS global interrupt                                       */
  HASH_RNG_IRQn               = 80      /*!< Hash and RNG global interrupt                                     */
} IRQn_Type;

/**
  * @}
  */

#include "core_cm3.h"
#include "system_stm32f2xx.h"
#include <stdint.h>

/** @addtogroup Peripheral_registers_structures
  * @{
  */   

/** 
  * @brief Analog to Digital Converter  
  */

typedef struct
{
  __IO uint32_t SR;     /*!< ADC status register,                         Address offset: 0x00 */
  __IO uint32_t CR1;    /*!< ADC control register 1,                      Address offset: 0x04 */      
  __IO uint32_t CR2;    /*!< ADC control register 2,                      Address offset: 0x08 */
  __IO uint32_t SMPR1;  /*!< ADC sample time register 1,                  Address offset: 0x0C */
  __IO uint32_t SMPR2;  /*!< ADC sample time register 2,                  Address offset: 0x10 */
  __IO uint32_t JOFR1;  /*!< ADC injected channel data offset register 1, Address offset: 0x14 */
  __IO uint32_t JOFR2;  /*!< ADC injected channel data offset register 2, Address offset: 0x18 */
  __IO uint32_t JOFR3;  /*!< ADC injected channel data offset register 3, Address offset: 0x1C */
  __IO uint32_t JOFR4;  /*!< ADC injected channel data offset register 4, Address offset: 0x20 */
  __IO uint32_t HTR;    /*!< ADC watchdog higher threshold register,      Address offset: 0x24 */
  __IO uint32_t LTR;    /*!< ADC watchdog lower threshold register,       Address offset: 0x28 */
  __IO uint32_t SQR1;   /*!< ADC regular sequence register 1,             Address offset: 0x2C */
  __IO uint32_t SQR2;   /*!< ADC regular sequence register 2,             Address offset: 0x30 */
  __IO uint32_t SQR3;   /*!< ADC regular sequence register 3,             Address offset: 0x34 */
  __IO uint32_t JSQR;   /*!< ADC injected sequence register,              Address offset: 0x38*/
  __IO uint32_t JDR1;   /*!< ADC injected data register 1,                Address offset: 0x3C */
  __IO uint32_t JDR2;   /*!< ADC injected data register 2,                Address offset: 0x40 */
  __IO uint32_t JDR3;   /*!< ADC injected data register 3,                Address offset: 0x44 */
  __IO uint32_t JDR4;   /*!< ADC injected data register 4,                Address offset: 0x48 */
  __IO uint32_t DR;     /*!< ADC regular data register,                   Address offset: 0x4C */
} ADC_TypeDef;

typedef struct
{
  __IO uint32_t CSR;    /*!< ADC Common status register,                  Address offset: ADC1 base address + 0x300 */
  __IO uint32_t CCR;    /*!< ADC common control register,                 Address offset: ADC1 base address + 0x304 */
  __IO uint32_t CDR;    /*!< ADC common regular data register for dual
                             AND triple modes,                            Address offset: ADC1 base address + 0x308 */
} ADC_Common_TypeDef;


/** 
  * @brief Controller Area Network TxMailBox 
  */

typedef struct
{
  __IO uint32_t TIR;  /*!< CAN TX mailbox identifier register */
  __IO uint32_t TDTR; /*!< CAN mailbox data length control and time stamp register */
  __IO uint32_t TDLR; /*!< CAN mailbox data low register */
  __IO uint32_t TDHR; /*!< CAN mailbox data high register */
} CAN_TxMailBox_TypeDef;

/** 
  * @brief Controller Area Network FIFOMailBox 
  */
  
typedef struct
{
  __IO uint32_t RIR;  /*!< CAN receive FIFO mailbox identifier register */
  __IO uint32_t RDTR; /*!< CAN receive FIFO mailbox data length control and time stamp register */
  __IO uint32_t RDLR; /*!< CAN receive FIFO mailbox data low register */
  __IO uint32_t RDHR; /*!< CAN receive FIFO mailbox data high register */
} CAN_FIFOMailBox_TypeDef;

/** 
  * @brief Controller Area Network FilterRegister 
  */
  
typedef struct
{
  __IO uint32_t FR1; /*!< CAN Filter bank register 1 */
  __IO uint32_t FR2; /*!< CAN Filter bank register 1 */
} CAN_FilterRegister_TypeDef;

/** 
  * @brief Controller Area Network 
  */
  
typedef struct
{
  __IO uint32_t              MCR;                 /*!< CAN master control register,         Address offset: 0x00          */
  __IO uint32_t              MSR;                 /*!< CAN master status register,          Address offset: 0x04          */
  __IO uint32_t              TSR;                 /*!< CAN transmit status register,        Address offset: 0x08          */
  __IO uint32_t              RF0R;                /*!< CAN receive FIFO 0 register,         Address offset: 0x0C          */
  __IO uint32_t              RF1R;                /*!< CAN receive FIFO 1 register,         Address offset: 0x10          */
  __IO uint32_t              IER;                 /*!< CAN interrupt enable register,       Address offset: 0x14          */
  __IO uint32_t              ESR;                 /*!< CAN error status register,           Address offset: 0x18          */
  __IO uint32_t              BTR;                 /*!< CAN bit timing register,             Address offset: 0x1C          */
  uint32_t                   RESERVED0[88];       /*!< Reserved, 0x020 - 0x17F                                            */
  CAN_TxMailBox_TypeDef      sTxMailBox[3];       /*!< CAN Tx MailBox,                      Address offset: 0x180 - 0x1AC */
  CAN_FIFOMailBox_TypeDef    sFIFOMailBox[2];     /*!< CAN FIFO MailBox,                    Address offset: 0x1B0 - 0x1CC */
  uint32_t                   RESERVED1[12];       /*!< Reserved, 0x1D0 - 0x1FF                                            */
  __IO uint32_t              FMR;                 /*!< CAN filter master register,          Address offset: 0x200         */
  __IO uint32_t              FM1R;                /*!< CAN filter mode register,            Address offset: 0x204         */
  uint32_t                   RESERVED2;           /*!< Reserved, 0x208                                                    */
  __IO uint32_t              FS1R;                /*!< CAN filter scale register,           Address offset: 0x20C         */
  uint32_t                   RESERVED3;           /*!< Reserved, 0x210                                                    */
  __IO uint32_t              FFA1R;               /*!< CAN filter FIFO assignment register, Address offset: 0x214         */
  uint32_t                   RESERVED4;           /*!< Reserved, 0x218                                                    */
  __IO uint32_t              FA1R;                /*!< CAN filter activation register,      Address offset: 0x21C         */
  uint32_t                   RESERVED5[8];        /*!< Reserved, 0x220-0x23F                                              */ 
  CAN_FilterRegister_TypeDef sFilterRegister[28]; /*!< CAN Filter Register,                 Address offset: 0x240-0x31C   */
} CAN_TypeDef;

/** 
  * @brief CRC calculation unit 
  */

typedef struct
{
  __IO uint32_t DR;         /*!< CRC Data register,             Address offset: 0x00 */
  __IO uint8_t  IDR;        /*!< CRC Independent data register, Address offset: 0x04 */
  uint8_t       RESERVED0;  /*!< Reserved, 0x05                                      */
  uint16_t      RESERVED1;  /*!< Reserved, 0x06                                      */
  __IO uint32_t CR;         /*!< CRC Control register,          Address offset: 0x08 */
} CRC_TypeDef;

/** 
  * @brief Digital to Analog Converter
  */

typedef struct
{
  __IO uint32_t CR;       /*!< DAC control register,                                    Address offset: 0x00 */
  __IO uint32_t SWTRIGR;  /*!< DAC software trigger register,                           Address offset: 0x04 */
  __IO uint32_t DHR12R1;  /*!< DAC channel1 12-bit right-aligned data holding register, Address offset: 0x08 */
  __IO uint32_t DHR12L1;  /*!< DAC channel1 12-bit left aligned data holding register,  Address offset: 0x0C */
  __IO uint32_t DHR8R1;   /*!< DAC channel1 8-bit right aligned data holding register,  Address offset: 0x10 */
  __IO uint32_t DHR12R2;  /*!< DAC channel2 12-bit right aligned data holding register, Address offset: 0x14 */
  __IO uint32_t DHR12L2;  /*!< DAC channel2 12-bit left aligned data holding register,  Address offset: 0x18 */
  __IO uint32_t DHR8R2;   /*!< DAC channel2 8-bit right-aligned data holding register,  Address offset: 0x1C */
  __IO uint32_t DHR12RD;  /*!< Dual DAC 12-bit right-aligned data holding register,     Address offset: 0x20 */
  __IO uint32_t DHR12LD;  /*!< DUAL DAC 12-bit left aligned data holding register,      Address offset: 0x24 */
  __IO uint32_t DHR8RD;   /*!< DUAL DAC 8-bit right aligned data holding register,      Address offset: 0x28 */
  __IO uint32_t DOR1;     /*!< DAC channel1 data output register,                       Address offset: 0x2C */
  __IO uint32_t DOR2;     /*!< DAC channel2 data output register,                       Address offset: 0x30 */
  __IO uint32_t SR;       /*!< DAC status register,                                     Address offset: 0x34 */
} DAC_TypeDef;

/** 
  * @brief Debug MCU
  */

typedef struct
{
  __IO uint32_t IDCODE;  /*!< MCU device ID code,               Address offset: 0x00 */
  __IO uint32_t CR;      /*!< Debug MCU configuration register, Address offset: 0x04 */
  __IO uint32_t APB1FZ;  /*!< Debug MCU APB1 freeze register,   Address offset: 0x08 */
  __IO uint32_t APB2FZ;  /*!< Debug MCU APB2 freeze register,   Address offset: 0x0C */
}DBGMCU_TypeDef;


/** 
  * @brief DMA Controller
  */

typedef struct
{
  __IO uint32_t CR;     /*!< DMA stream x configuration register      */
  __IO uint32_t NDTR;   /*!< DMA stream x number of data register     */
  __IO uint32_t PAR;    /*!< DMA stream x peripheral address register */
  __IO uint32_t M0AR;   /*!< DMA stream x memory 0 address register   */
  __IO uint32_t M1AR;   /*!< DMA stream x memory 1 address register   */
  __IO uint32_t FCR;    /*!< DMA stream x FIFO control register       */
} DMA_Stream_TypeDef;

typedef struct
{
  __IO uint32_t LISR;   /*!< DMA low interrupt status register,      Address offset: 0x00 */
  __IO uint32_t HISR;   /*!< DMA high interrupt status register,     Address offset: 0x04 */
  __IO uint32_t LIFCR;  /*!< DMA low interrupt flag clear register,  Address offset: 0x08 */
  __IO uint32_t HIFCR;  /*!< DMA high interrupt flag clear register, Address offset: 0x0C */
} DMA_TypeDef;


/** 
  * @brief External Interrupt/Event Controller
  */

typedef struct
{
  __IO uint32_t IMR;    /*!< EXTI Interrupt mask register,            Address offset: 0x00 */
  __IO uint32_t EMR;    /*!< EXTI Event mask register,                Address offset: 0x04 */
  __IO uint32_t RTSR;   /*!< EXTI Rising trigger selection register,  Address offset: 0x08 */
  __IO uint32_t FTSR;   /*!< EXTI Falling trigger selection register, Address offset: 0x0C */
  __IO uint32_t SWIER;  /*!< EXTI Software interrupt event register,  Address offset: 0x10 */
  __IO uint32_t PR;     /*!< EXTI Pending register,                   Address offset: 0x14 */
} EXTI_TypeDef;

/** 
  * @brief FLASH Registers
  */

typedef struct
{
  __IO uint32_t ACR;      /*!< FLASH access control register, Address offset: 0x00 */
  __IO uint32_t KEYR;     /*!< FLASH key register,            Address offset: 0x04 */
  __IO uint32_t OPTKEYR;  /*!< FLASH option key register,     Address offset: 0x08 */
  __IO uint32_t SR;       /*!< FLASH status register,         Address offset: 0x0C */
  __IO uint32_t CR;       /*!< FLASH control register,        Address offset: 0x10 */
  __IO uint32_t OPTCR;    /*!< FLASH option control register, Address offset: 0x14 */
} FLASH_TypeDef;


/** 
  * @brief Flexible Static Memory Controller
  */

typedef struct
{
  __IO uint32_t BTCR[8];    /*!< NOR/PSRAM chip-select control register(BCR) and chip-select timing register(BTR), Address offset: 0x00-1C */   
} FSMC_Bank1_TypeDef; 

/** 
  * @brief Flexible Static Memory Controller Bank1E
  */
  
typedef struct
{
  __IO uint32_t BWTR[7];    /*!< NOR/PSRAM write timing registers, Address offset: 0x104-0x11C */
} FSMC_Bank1E_TypeDef;

/** 
  * @brief Flexible Static Memory Controller Bank2
  */
  
typedef struct
{
  __IO uint32_t PCR2;       /*!< NAND Flash control register 2,                       Address offset: 0x60 */
  __IO uint32_t SR2;        /*!< NAND Flash FIFO status and interrupt register 2,     Address offset: 0x64 */
  __IO uint32_t PMEM2;      /*!< NAND Flash Common memory space timing register 2,    Address offset: 0x68 */
  __IO uint32_t PATT2;      /*!< NAND Flash Attribute memory space timing register 2, Address offset: 0x6C */
  uint32_t      RESERVED0;  /*!< Reserved, 0x70                                                            */
  __IO uint32_t ECCR2;      /*!< NAND Flash ECC result registers 2,                   Address offset: 0x74 */
  uint32_t      RESERVED1;  /*!< Reserved, 0x78                                                            */
  uint32_t      RESERVED2;  /*!< Reserved, 0x7C                                                            */
  __IO uint32_t PCR3;       /*!< NAND Flash control register 3,                       Address offset: 0x80 */
  __IO uint32_t SR3;        /*!< NAND Flash FIFO status and interrupt register 3,     Address offset: 0x84 */
  __IO uint32_t PMEM3;      /*!< NAND Flash Common memory space timing register 3,    Address offset: 0x88 */
  __IO uint32_t PATT3;      /*!< NAND Flash Attribute memory space timing register 3, Address offset: 0x8C */
  uint32_t      RESERVED3;  /*!< Reserved, 0x90                                                            */
  __IO uint32_t ECCR3;      /*!< NAND Flash ECC result registers 3,                   Address offset: 0x94 */
} FSMC_Bank2_3_TypeDef;

/** 
  * @brief Flexible Static Memory Controller Bank4
  */
  
typedef struct
{
  __IO uint32_t PCR4;       /*!< PC Card  control register 4,                       Address offset: 0xA0 */
  __IO uint32_t SR4;        /*!< PC Card  FIFO status and interrupt register 4,     Address offset: 0xA4 */
  __IO uint32_t PMEM4;      /*!< PC Card  Common memory space timing register 4,    Address offset: 0xA8 */
  __IO uint32_t PATT4;      /*!< PC Card  Attribute memory space timing register 4, Address offset: 0xAC */
  __IO uint32_t PIO4;       /*!< PC Card  I/O space timing register 4,              Address offset: 0xB0 */
} FSMC_Bank4_TypeDef; 


/** 
  * @brief General Purpose I/O
  */

typedef struct
{
  __IO uint32_t MODER;    /*!< GPIO port mode register,               Address offset: 0x00      */
  __IO uint32_t OTYPER;   /*!< GPIO port output type register,        Address offset: 0x04      */
  __IO uint32_t OSPEEDR;  /*!< GPIO port output speed register,       Address offset: 0x08      */
  __IO uint32_t PUPDR;    /*!< GPIO port pull-up/pull-down register,  Address offset: 0x0C      */
  __IO uint32_t IDR;      /*!< GPIO port input data register,         Address offset: 0x10      */
  __IO uint32_t ODR;      /*!< GPIO port output data register,        Address offset: 0x14      */
  __IO uint32_t BSRR;     /*!< GPIO port bit set/reset register,      Address offset: 0x18      */
  __IO uint32_t LCKR;     /*!< GPIO port configuration lock register, Address offset: 0x1C      */
  __IO uint32_t AFR[2];   /*!< GPIO alternate function registers,     Address offset: 0x20-0x24 */
} GPIO_TypeDef;

/** 
  * @brief System configuration controller
  */
  
typedef struct
{
  __IO uint32_t MEMRMP;       /*!< SYSCFG memory remap register,                      Address offset: 0x00      */
  __IO uint32_t PMC;          /*!< SYSCFG peripheral mode configuration register,     Address offset: 0x04      */
  __IO uint32_t EXTICR[4];    /*!< SYSCFG external interrupt configuration registers, Address offset: 0x08-0x14 */
  uint32_t      RESERVED[2];  /*!< Reserved, 0x18-0x1C                                                          */ 
  __IO uint32_t CMPCR;        /*!< SYSCFG Compensation cell control register,         Address offset: 0x20      */
} SYSCFG_TypeDef;

/** 
  * @brief Inter-integrated Circuit Interface
  */

typedef struct
{
  __IO uint32_t CR1;        /*!< I2C Control register 1,     Address offset: 0x00 */
  __IO uint32_t CR2;        /*!< I2C Control register 2,     Address offset: 0x04 */
  __IO uint32_t OAR1;       /*!< I2C Own address register 1, Address offset: 0x08 */
  __IO uint32_t OAR2;       /*!< I2C Own address register 2, Address offset: 0x0C */
  __IO uint32_t DR;         /*!< I2C Data register,          Address offset: 0x10 */
  __IO uint32_t SR1;        /*!< I2C Status register 1,      Address offset: 0x14 */
  __IO uint32_t SR2;        /*!< I2C Status register 2,      Address offset: 0x18 */
  __IO uint32_t CCR;        /*!< I2C Clock control register, Address offset: 0x1C */
  __IO uint32_t TRISE;      /*!< I2C TRISE register,         Address offset: 0x20 */
} I2C_TypeDef;

/** 
  * @brief Independent WATCHDOG
  */

typedef struct
{
  __IO uint32_t KR;   /*!< IWDG Key register,       Address offset: 0x00 */
  __IO uint32_t PR;   /*!< IWDG Prescaler register, Address offset: 0x04 */
  __IO uint32_t RLR;  /*!< IWDG Reload register,    Address offset: 0x08 */
  __IO uint32_t SR;   /*!< IWDG Status register,    Address offset: 0x0C */
} IWDG_TypeDef;

/** 
  * @brief Power Control
  */

typedef struct
{
  __IO uint32_t CR;   /*!< PWR power control register,        Address offset: 0x00 */
  __IO uint32_t CSR;  /*!< PWR power control/status register, Address offset: 0x04 */
} PWR_TypeDef;

/** 
  * @brief Reset and Clock Control
  */

typedef struct
{
  __IO uint32_t CR;            /*!< RCC clock control register,                                  Address offset: 0x00 */
  __IO uint32_t PLLCFGR;       /*!< RCC PLL configuration register,                              Address offset: 0x04 */
  __IO uint32_t CFGR;          /*!< RCC clock configuration register,                            Address offset: 0x08 */
  __IO uint32_t CIR;           /*!< RCC clock interrupt register,                                Address offset: 0x0C */
  __IO uint32_t AHB1RSTR;      /*!< RCC AHB1 peripheral reset register,                          Address offset: 0x10 */
  __IO uint32_t AHB2RSTR;      /*!< RCC AHB2 peripheral reset register,                          Address offset: 0x14 */
  __IO uint32_t AHB3RSTR;      /*!< RCC AHB3 peripheral reset register,                          Address offset: 0x18 */
  uint32_t      RESERVED0;     /*!< Reserved, 0x1C                                                                    */
  __IO uint32_t APB1RSTR;      /*!< RCC APB1 peripheral reset register,                          Address offset: 0x20 */
  __IO uint32_t APB2RSTR;      /*!< RCC APB2 peripheral reset register,                          Address offset: 0x24 */
  uint32_t      RESERVED1[2];  /*!< Reserved, 0x28-0x2C                                                               */
  __IO uint32_t AHB1ENR;       /*!< RCC AHB1 peripheral clock register,                          Address offset: 0x30 */
  __IO uint32_t AHB2ENR;       /*!< RCC AHB2 peripheral clock register,                          Address offset: 0x34 */
  __IO uint32_t AHB3ENR;       /*!< RCC AHB3 peripheral clock register,                          Address offset: 0x38 */
  uint32_t      RESERVED2;     /*!< Reserved, 0x3C                                                                    */
  __IO uint32_t APB1ENR;       /*!< RCC APB1 peripheral clock enable register,                   Address offset: 0x40 */
  __IO uint32_t APB2ENR;       /*!< RCC APB2 peripheral clock enable register,                   Address offset: 0x44 */
  uint32_t      RESERVED3[2];  /*!< Reserved, 0x48-0x4C                                                               */
  __IO uint32_t AHB1LPENR;     /*!< RCC AHB1 peripheral clock enable in low power mode register, Address offset: 0x50 */
  __IO uint32_t AHB2LPENR;     /*!< RCC AHB2 peripheral clock enable in low power mode register, Address offset: 0x54 */
  __IO uint32_t AHB3LPENR;     /*!< RCC AHB3 peripheral clock enable in low power mode register, Address offset: 0x58 */
  uint32_t      RESERVED4;     /*!< Reserved, 0x5C                                                                    */
  __IO uint32_t APB1LPENR;     /*!< RCC APB1 peripheral clock enable in low power mode register, Address offset: 0x60 */
  __IO uint32_t APB2LPENR;     /*!< RCC APB2 peripheral clock enable in low power mode register, Address offset: 0x64 */
  uint32_t      RESERVED5[2];  /*!< Reserved, 0x68-0x6C                                                               */
  __IO uint32_t BDCR;          /*!< RCC Backup domain control register,                          Address offset: 0x70 */
  __IO uint32_t CSR;           /*!< RCC clock control & status register,                         Address offset: 0x74 */
  uint32_t      RESERVED6[2];  /*!< Reserved, 0x78-0x7C                                                               */
  __IO uint32_t SSCGR;         /*!< RCC spread spectrum clock generation register,               Address offset: 0x80 */
  __IO uint32_t PLLI2SCFGR;    /*!< RCC PLLI2S configuration register,                           Address offset: 0x84 */

} RCC_TypeDef;

/** 
  * @brief Real-Time Clock
  */

typedef struct
{
  __IO uint32_t TR;      /*!< RTC time register,                                        Address offset: 0x00 */
  __IO uint32_t DR;      /*!< RTC date register,                                        Address offset: 0x04 */
  __IO uint32_t CR;      /*!< RTC control register,                                     Address offset: 0x08 */
  __IO uint32_t ISR;     /*!< RTC initialization and status register,                   Address offset: 0x0C */
  __IO uint32_t PRER;    /*!< RTC prescaler register,                                   Address offset: 0x10 */
  __IO uint32_t WUTR;    /*!< RTC wakeup timer register,                                Address offset: 0x14 */
  __IO uint32_t CALIBR;  /*!< RTC calibration register,                                 Address offset: 0x18 */
  __IO uint32_t ALRMAR;  /*!< RTC alarm A register,                                     Address offset: 0x1C */
  __IO uint32_t ALRMBR;  /*!< RTC alarm B register,                                     Address offset: 0x20 */
  __IO uint32_t WPR;     /*!< RTC write protection register,                            Address offset: 0x24 */
  uint32_t RESERVED1;    /*!< Reserved, 0x28                                                                 */
  uint32_t RESERVED2;    /*!< Reserved, 0x2C                                                                 */
  __IO uint32_t TSTR;    /*!< RTC time stamp time register,                             Address offset: 0x30 */
  __IO uint32_t TSDR;    /*!< RTC time stamp date register,                             Address offset: 0x34 */
  uint32_t RESERVED3;    /*!< Reserved, 0x38                                                                 */
  uint32_t RESERVED4;    /*!< Reserved, 0x3C                                                                 */
  __IO uint32_t TAFCR;   /*!< RTC tamper and alternate function configuration register, Address offset: 0x40 */
  uint32_t RESERVED5;    /*!< Reserved, 0x44                                                                 */
  uint32_t RESERVED6;    /*!< Reserved, 0x48                                                                 */
  uint32_t RESERVED7;    /*!< Reserved, 0x4C                                                                 */
  __IO uint32_t BKP0R;   /*!< RTC backup register 1,                                    Address offset: 0x50 */
  __IO uint32_t BKP1R;   /*!< RTC backup register 1,                                    Address offset: 0x54 */
  __IO uint32_t BKP2R;   /*!< RTC backup register 2,                                    Address offset: 0x58 */
  __IO uint32_t BKP3R;   /*!< RTC backup register 3,                                    Address offset: 0x5C */
  __IO uint32_t BKP4R;   /*!< RTC backup register 4,                                    Address offset: 0x60 */
  __IO uint32_t BKP5R;   /*!< RTC backup register 5,                                    Address offset: 0x64 */
  __IO uint32_t BKP6R;   /*!< RTC backup register 6,                                    Address offset: 0x68 */
  __IO uint32_t BKP7R;   /*!< RTC backup register 7,                                    Address offset: 0x6C */
  __IO uint32_t BKP8R;   /*!< RTC backup register 8,                                    Address offset: 0x70 */
  __IO uint32_t BKP9R;   /*!< RTC backup register 9,                                    Address offset: 0x74 */
  __IO uint32_t BKP10R;  /*!< RTC backup register 10,                                   Address offset: 0x78 */
  __IO uint32_t BKP11R;  /*!< RTC backup register 11,                                   Address offset: 0x7C */
  __IO uint32_t BKP12R;  /*!< RTC backup register 12,                                   Address offset: 0x80 */
  __IO uint32_t BKP13R;  /*!< RTC backup register 13,                                   Address offset: 0x84 */
  __IO uint32_t BKP14R;  /*!< RTC backup register 14,                                   Address offset: 0x88 */
  __IO uint32_t BKP15R;  /*!< RTC backup register 15,                                   Address offset: 0x8C */
  __IO uint32_t BKP16R;  /*!< RTC backup register 16,                                   Address offset: 0x90 */
  __IO uint32_t BKP17R;  /*!< RTC backup register 17,                                   Address offset: 0x94 */
  __IO uint32_t BKP18R;  /*!< RTC backup register 18,                                   Address offset: 0x98 */
  __IO uint32_t BKP19R;  /*!< RTC backup register 19,                                   Address offset: 0x9C */
} RTC_TypeDef;


/** 
  * @brief SD host Interface
  */

typedef struct
{
  __IO uint32_t POWER;          /*!< SDIO power control register,    Address offset: 0x00 */
  __IO uint32_t CLKCR;          /*!< SDI clock control register,     Address offset: 0x04 */
  __IO uint32_t ARG;            /*!< SDIO argument register,         Address offset: 0x08 */
  __IO uint32_t CMD;            /*!< SDIO command register,          Address offset: 0x0C */
  __IO const uint32_t  RESPCMD; /*!< SDIO command response register, Address offset: 0x10 */
  __IO const uint32_t  RESP1;   /*!< SDIO response 1 register,       Address offset: 0x14 */
  __IO const uint32_t  RESP2;   /*!< SDIO response 2 register,       Address offset: 0x18 */
  __IO const uint32_t  RESP3;   /*!< SDIO response 3 register,       Address offset: 0x1C */
  __IO const uint32_t  RESP4;   /*!< SDIO response 4 register,       Address offset: 0x20 */
  __IO uint32_t DTIMER;         /*!< SDIO data timer register,       Address offset: 0x24 */
  __IO uint32_t DLEN;           /*!< SDIO data length register,      Address offset: 0x28 */
  __IO uint32_t DCTRL;          /*!< SDIO data control register,     Address offset: 0x2C */
  __IO const uint32_t  DCOUNT;  /*!< SDIO data counter register,     Address offset: 0x30 */
  __IO const uint32_t  STA;     /*!< SDIO status register,           Address offset: 0x34 */
  __IO uint32_t ICR;            /*!< SDIO interrupt clear register,  Address offset: 0x38 */
  __IO uint32_t MASK;           /*!< SDIO mask register,             Address offset: 0x3C */
  uint32_t      RESERVED0[2];   /*!< Reserved, 0x40-0x44                                  */
  __IO const uint32_t  FIFOCNT; /*!< SDIO FIFO counter register,     Address offset: 0x48 */
  uint32_t      RESERVED1[13];  /*!< Reserved, 0x4C-0x7C                                  */
  __IO uint32_t FIFO;           /*!< SDIO data FIFO register,        Address offset: 0x80 */
} SDIO_TypeDef;

/** 
  * @brief Serial Peripheral Interface
  */

typedef struct
{
  __IO uint32_t CR1;        /*!< SPI control register 1 (not used in I2S mode),      Address offset: 0x00 */
  __IO uint32_t CR2;        /*!< SPI control register 2,                             Address offset: 0x04 */
  __IO uint32_t SR;         /*!< SPI status register,                                Address offset: 0x08 */
  __IO uint32_t DR;         /*!< SPI data register,                                  Address offset: 0x0C */
  __IO uint32_t CRCPR;      /*!< SPI CRC polynomial register (not used in I2S mode), Address offset: 0x10 */
  __IO uint32_t RXCRCR;     /*!< SPI RX CRC register (not used in I2S mode),         Address offset: 0x14 */
  __IO uint32_t TXCRCR;     /*!< SPI TX CRC register (not used in I2S mode),         Address offset: 0x18 */
  __IO uint32_t I2SCFGR;    /*!< SPI_I2S configuration register,                     Address offset: 0x1C */
  __IO uint32_t I2SPR;      /*!< SPI_I2S prescaler register,                         Address offset: 0x20 */
} SPI_TypeDef;

/** 
  * @brief TIM
  */

typedef struct
{
  __IO uint32_t CR1;         /*!< TIM control register 1,              Address offset: 0x00 */
  __IO uint32_t CR2;         /*!< TIM control register 2,              Address offset: 0x04 */
  __IO uint32_t SMCR;        /*!< TIM slave mode control register,     Address offset: 0x08 */
  __IO uint32_t DIER;        /*!< TIM DMA/interrupt enable register,   Address offset: 0x0C */
  __IO uint32_t SR;          /*!< TIM status register,                 Address offset: 0x10 */
  __IO uint32_t EGR;         /*!< TIM event generation register,       Address offset: 0x14 */
  __IO uint32_t CCMR1;       /*!< TIM capture/compare mode register 1, Address offset: 0x18 */
  __IO uint32_t CCMR2;       /*!< TIM capture/compare mode register 2, Address offset: 0x1C */
  __IO uint32_t CCER;        /*!< TIM capture/compare enable register, Address offset: 0x20 */
  __IO uint32_t CNT;         /*!< TIM counter register,                Address offset: 0x24 */
  __IO uint32_t PSC;         /*!< TIM prescaler,                       Address offset: 0x28 */
  __IO uint32_t ARR;         /*!< TIM auto-reload register,            Address offset: 0x2C */
  __IO uint32_t RCR;         /*!< TIM repetition counter register,     Address offset: 0x30 */
  __IO uint32_t CCR1;        /*!< TIM capture/compare register 1,      Address offset: 0x34 */
  __IO uint32_t CCR2;        /*!< TIM capture/compare register 2,      Address offset: 0x38 */
  __IO uint32_t CCR3;        /*!< TIM capture/compare register 3,      Address offset: 0x3C */
  __IO uint32_t CCR4;        /*!< TIM capture/compare register 4,      Address offset: 0x40 */
  __IO uint32_t BDTR;        /*!< TIM break and dead-time register,    Address offset: 0x44 */
  __IO uint32_t DCR;         /*!< TIM DMA control register,            Address offset: 0x48 */
  __IO uint32_t DMAR;        /*!< TIM DMA address for full transfer,   Address offset: 0x4C */
  __IO uint32_t OR;          /*!< TIM option register,                 Address offset: 0x50 */
} TIM_TypeDef;

/** 
  * @brief Universal Synchronous Asynchronous Receiver Transmitter
  */
 
typedef struct
{
  __IO uint32_t SR;         /*!< USART Status register,                   Address offset: 0x00 */
  __IO uint32_t DR;         /*!< USART Data register,                     Address offset: 0x04 */
  __IO uint32_t BRR;        /*!< USART Baud rate register,                Address offset: 0x08 */
  __IO uint32_t CR1;        /*!< USART Control register 1,                Address offset: 0x0C */
  __IO uint32_t CR2;        /*!< USART Control register 2,                Address offset: 0x10 */
  __IO uint32_t CR3;        /*!< USART Control register 3,                Address offset: 0x14 */
  __IO uint32_t GTPR;       /*!< USART Guard time and prescaler register, Address offset: 0x18 */
} USART_TypeDef;

/** 
  * @brief Window WATCHDOG
  */

typedef struct
{
  __IO uint32_t CR;   /*!< WWDG Control register,       Address offset: 0x00 */
  __IO uint32_t CFR;  /*!< WWDG Configuration register, Address offset: 0x04 */
  __IO uint32_t SR;   /*!< WWDG Status register,        Address offset: 0x08 */
} WWDG_TypeDef;


/** 
  * @brief RNG
  */
  
typedef struct 
{
  __IO uint32_t CR;  /*!< RNG control register, Address offset: 0x00 */
  __IO uint32_t SR;  /*!< RNG status register,  Address offset: 0x04 */
  __IO uint32_t DR;  /*!< RNG data register,    Address offset: 0x08 */
} RNG_TypeDef;


 
/** 
  * @brief __USB_OTG_Core_register
  */
typedef struct
{
  __IO uint32_t GOTGCTL;              /*!<  USB_OTG Control and Status Register    Address offset : 0x00      */
  __IO uint32_t GOTGINT;              /*!<  USB_OTG Interrupt Register             Address offset : 0x04      */
  __IO uint32_t GAHBCFG;              /*!<  Core AHB Configuration Register        Address offset : 0x08      */
  __IO uint32_t GUSBCFG;              /*!<  Core USB Configuration Register        Address offset : 0x0C      */
  __IO uint32_t GRSTCTL;              /*!<  Core Reset Register                    Address offset : 0x10      */
  __IO uint32_t GINTSTS;              /*!<  Core Interrupt Register                Address offset : 0x14      */
  __IO uint32_t GINTMSK;              /*!<  Core Interrupt Mask Register           Address offset : 0x18      */
  __IO uint32_t GRXSTSR;              /*!<  Receive Sts Q Read Register            Address offset : 0x1C      */
  __IO uint32_t GRXSTSP;              /*!<  Receive Sts Q Read & POP Register      Address offset : 0x20      */
  __IO uint32_t GRXFSIZ;              /* Receive FIFO Size Register                Address offset : 0x24      */
  __IO uint32_t DIEPTXF0_HNPTXFSIZ;   /*!<  EP0 / Non Periodic Tx FIFO Size Register Address offset : 0x28    */
  __IO uint32_t HNPTXSTS;             /*!<  Non Periodic Tx FIFO/Queue Sts reg     Address offset : 0x2C      */
  uint32_t Reserved30[2];             /* Reserved                                  Address offset : 0x30      */
  __IO uint32_t GCCFG;                /*!<  General Purpose IO Register            Address offset : 0x38      */
  __IO uint32_t CID;                  /*!< User ID Register                          Address offset : 0x3C      */
  uint32_t  Reserved40[48];           /*!< Reserved                                  Address offset : 0x40-0xFF */
  __IO uint32_t HPTXFSIZ;             /*!< Host Periodic Tx FIFO Size Reg            Address offset : 0x100 */
  __IO uint32_t DIEPTXF[0x0F];        /*!< dev Periodic Transmit FIFO */
}
USB_OTG_GlobalTypeDef;



/** 
  * @brief __device_Registers
  */
typedef struct 
{
  __IO uint32_t DCFG;         /*!< dev Configuration Register   Address offset : 0x800 */
  __IO uint32_t DCTL;         /*!< dev Control Register         Address offset : 0x804 */
  __IO uint32_t DSTS;         /*!< dev Status Register (RO)     Address offset : 0x808 */
  uint32_t Reserved0C;        /*!< Reserved                     Address offset : 0x80C */
  __IO uint32_t DIEPMSK;      /* !< dev IN Endpoint Mask        Address offset : 0x810 */
  __IO uint32_t DOEPMSK;      /*!< dev OUT Endpoint Mask        Address offset : 0x814 */
  __IO uint32_t DAINT;        /*!< dev All Endpoints Itr Reg    Address offset : 0x818 */
  __IO uint32_t DAINTMSK;     /*!< dev All Endpoints Itr Mask   Address offset : 0x81C */
  uint32_t  Reserved20;       /*!< Reserved                     Address offset : 0x820 */
  uint32_t Reserved9;         /*!< Reserved                     Address offset : 0x824 */
  __IO uint32_t DVBUSDIS;     /*!< dev VBUS discharge Register  Address offset : 0x828 */
  __IO uint32_t DVBUSPULSE;   /*!< dev VBUS Pulse Register      Address offset : 0x82C */
  __IO uint32_t DTHRCTL;      /*!< dev thr                      Address offset : 0x830 */
  __IO uint32_t DIEPEMPMSK;   /*!< dev empty msk                Address offset : 0x834 */
  __IO uint32_t DEACHINT;     /*!< dedicated EP interrupt       Address offset : 0x838 */
  __IO uint32_t DEACHMSK;     /*!< dedicated EP msk             Address offset : 0x83C */  
  uint32_t Reserved40;        /*!< dedicated EP mask            Address offset : 0x840 */
  __IO uint32_t DINEP1MSK;    /*!< dedicated EP mask            Address offset : 0x844 */
  uint32_t  Reserved44[15];   /*!< Reserved                     Address offset : 0x844-0x87C */
  __IO uint32_t DOUTEP1MSK;   /*!< dedicated EP msk             Address offset : 0x884 */   
}
USB_OTG_DeviceTypeDef;


/** 
  * @brief __IN_Endpoint-Specific_Register
  */
typedef struct 
{
  __IO uint32_t DIEPCTL;        /* dev IN Endpoint Control Reg 900h + (ep_num * 20h) + 00h     */
  uint32_t Reserved04;          /* Reserved                       900h + (ep_num * 20h) + 04h  */
  __IO uint32_t DIEPINT;        /* dev IN Endpoint Itr Reg     900h + (ep_num * 20h) + 08h     */
  uint32_t Reserved0C;          /* Reserved                       900h + (ep_num * 20h) + 0Ch  */
  __IO uint32_t DIEPTSIZ;       /* IN Endpoint Txfer Size   900h + (ep_num * 20h) + 10h        */
  __IO uint32_t DIEPDMA;        /* IN Endpoint DMA Address Reg    900h + (ep_num * 20h) + 14h  */
  __IO uint32_t DTXFSTS;        /*IN Endpoint Tx FIFO Status Reg 900h + (ep_num * 20h) + 18h   */
  uint32_t Reserved18;           /* Reserved  900h+(ep_num*20h)+1Ch-900h+ (ep_num * 20h) + 1Ch */
}
USB_OTG_INEndpointTypeDef;


/** 
  * @brief __OUT_Endpoint-Specific_Registers
  */
typedef struct 
{
  __IO uint32_t DOEPCTL;       /* dev OUT Endpoint Control Reg  B00h + (ep_num * 20h) + 00h*/
  uint32_t Reserved04;         /* Reserved                      B00h + (ep_num * 20h) + 04h*/
  __IO uint32_t DOEPINT;       /* dev OUT Endpoint Itr Reg      B00h + (ep_num * 20h) + 08h*/
  uint32_t Reserved0C;         /* Reserved                      B00h + (ep_num * 20h) + 0Ch*/
  __IO uint32_t DOEPTSIZ;      /* dev OUT Endpoint Txfer Size   B00h + (ep_num * 20h) + 10h*/
  __IO uint32_t DOEPDMA;       /* dev OUT Endpoint DMA Address  B00h + (ep_num * 20h) + 14h*/
  uint32_t Reserved18[2];      /* Reserved B00h + (ep_num * 20h) + 18h - B00h + (ep_num * 20h) + 1Ch*/
}
USB_OTG_OUTEndpointTypeDef;


/** 
  * @brief __Host_Mode_Register_Structures
  */
typedef struct 
{
  __IO uint32_t HCFG;             /* Host Configuration Register    400h*/
  __IO uint32_t HFIR;             /* Host Frame Interval Register   404h*/
  __IO uint32_t HFNUM;            /* Host Frame Nbr/Frame Remaining 408h*/
  uint32_t Reserved40C;           /* Reserved                       40Ch*/
  __IO uint32_t HPTXSTS;          /* Host Periodic Tx FIFO/ Queue Status 410h*/
  __IO uint32_t HAINT;            /* Host All Channels Interrupt Register 414h*/
  __IO uint32_t HAINTMSK;         /* Host All Channels Interrupt Mask 418h*/
}
USB_OTG_HostTypeDef;


/** 
  * @brief __Host_Channel_Specific_Registers
  */
typedef struct
{
  __IO uint32_t HCCHAR;
  __IO uint32_t HCSPLT;
  __IO uint32_t HCINT;
  __IO uint32_t HCINTMSK;
  __IO uint32_t HCTSIZ;
  __IO uint32_t HCDMA;
  uint32_t Reserved[2];
}
USB_OTG_HostChannelTypeDef;

    
/** 
  * @brief Peripheral_memory_map
  */
#define FLASH_BASE            0x08000000U /*!< FLASH(up to 1 MB) base address in the alias region                         */
#define SRAM1_BASE            0x20000000U /*!< SRAM1(112 KB) base address in the alias region                             */
#define SRAM2_BASE            0x2001C000U /*!< SRAM2(16 KB) base address in the alias region                              */
#define PERIPH_BASE           0x40000000U /*!< Peripheral base address in the alias region                                */
#define BKPSRAM_BASE          0x40024000U /*!< Backup SRAM(4 KB) base address in the alias region                         */
#define FSMC_R_BASE           0xA0000000U /*!< FSMC registers base address                                                */
#define SRAM1_BB_BASE         0x22000000U /*!< SRAM1(112 KB) base address in the bit-band region                          */
#define SRAM2_BB_BASE         0x22380000U /*!< SRAM2(16 KB) base address in the bit-band region                           */
#define PERIPH_BB_BASE        0x42000000U /*!< Peripheral base address in the bit-band region                             */
#define BKPSRAM_BB_BASE       0x42480000U /*!< Backup SRAM(4 KB) base address in the bit-band region                      */
#define FLASH_END             0x080FFFFFU /*!< FLASH end address                                                          */

/* Legacy defines */
#define SRAM_BASE             SRAM1_BASE
#define SRAM_BB_BASE          SRAM1_BB_BASE


/*!< Peripheral memory map */
#define APB1PERIPH_BASE       PERIPH_BASE
#define APB2PERIPH_BASE       (PERIPH_BASE + 0x00010000U)
#define AHB1PERIPH_BASE       (PERIPH_BASE + 0x00020000U)
#define AHB2PERIPH_BASE       (PERIPH_BASE + 0x10000000U)

/*!< APB1 peripherals */
#define TIM2_BASE             (APB1PERIPH_BASE + 0x0000U)
#define TIM3_BASE             (APB1PERIPH_BASE + 0x0400U)
#define TIM4_BASE             (APB1PERIPH_BASE + 0x0800U)
#define TIM5_BASE             (APB1PERIPH_BASE + 0x0C00U)
#define TIM6_BASE             (APB1PERIPH_BASE + 0x1000U)
#define TIM7_BASE             (APB1PERIPH_BASE + 0x1400U)
#define TIM12_BASE            (APB1PERIPH_BASE + 0x1800U)
#define TIM13_BASE            (APB1PERIPH_BASE + 0x1C00U)
#define TIM14_BASE            (APB1PERIPH_BASE + 0x2000U)
#define RTC_BASE              (APB1PERIPH_BASE + 0x2800U)
#define WWDG_BASE             (APB1PERIPH_BASE + 0x2C00U)
#define IWDG_BASE             (APB1PERIPH_BASE + 0x3000U)
#define SPI2_BASE             (APB1PERIPH_BASE + 0x3800U)
#define SPI3_BASE             (APB1PERIPH_BASE + 0x3C00U)
#define USART2_BASE           (APB1PERIPH_BASE + 0x4400U)
#define USART3_BASE           (APB1PERIPH_BASE + 0x4800U)
#define UART4_BASE            (APB1PERIPH_BASE + 0x4C00U)
#define UART5_BASE            (APB1PERIPH_BASE + 0x5000U)
#define I2C1_BASE             (APB1PERIPH_BASE + 0x5400U)
#define I2C2_BASE             (APB1PERIPH_BASE + 0x5800U)
#define I2C3_BASE             (APB1PERIPH_BASE + 0x5C00U)
#define CAN1_BASE             (APB1PERIPH_BASE + 0x6400U)
#define CAN2_BASE             (APB1PERIPH_BASE + 0x6800U)
#define PWR_BASE              (APB1PERIPH_BASE + 0x7000U)
#define DAC_BASE              (APB1PERIPH_BASE + 0x7400U)

/*!< APB2 peripherals */
#define TIM1_BASE             (APB2PERIPH_BASE + 0x0000U)
#define TIM8_BASE             (APB2PERIPH_BASE + 0x0400U)
#define USART1_BASE           (APB2PERIPH_BASE + 0x1000U)
#define USART6_BASE           (APB2PERIPH_BASE + 0x1400U)
#define ADC1_BASE             (APB2PERIPH_BASE + 0x2000U)
#define ADC2_BASE             (APB2PERIPH_BASE + 0x2100U)
#define ADC3_BASE             (APB2PERIPH_BASE + 0x2200U)
#define ADC_BASE              (APB2PERIPH_BASE + 0x2300U)
#define SDIO_BASE             (APB2PERIPH_BASE + 0x2C00U)
#define SPI1_BASE             (APB2PERIPH_BASE + 0x3000U)
#define SYSCFG_BASE           (APB2PERIPH_BASE + 0x3800U)
#define EXTI_BASE             (APB2PERIPH_BASE + 0x3C00U)
#define TIM9_BASE             (APB2PERIPH_BASE + 0x4000U)
#define TIM10_BASE            (APB2PERIPH_BASE + 0x4400U)
#define TIM11_BASE            (APB2PERIPH_BASE + 0x4800U)

/*!< AHB1 peripherals */
#define GPIOA_BASE            (AHB1PERIPH_BASE + 0x0000U)
#define GPIOB_BASE            (AHB1PERIPH_BASE + 0x0400U)
#define GPIOC_BASE            (AHB1PERIPH_BASE + 0x0800U)
#define GPIOD_BASE            (AHB1PERIPH_BASE + 0x0C00U)
#define GPIOE_BASE            (AHB1PERIPH_BASE + 0x1000U)
#define GPIOF_BASE            (AHB1PERIPH_BASE + 0x1400U)
#define GPIOG_BASE            (AHB1PERIPH_BASE + 0x1800U)
#define GPIOH_BASE            (AHB1PERIPH_BASE + 0x1C00U)
#define GPIOI_BASE            (AHB1PERIPH_BASE + 0x2000U)
#define CRC_BASE              (AHB1PERIPH_BASE + 0x3000U)
#define RCC_BASE              (AHB1PERIPH_BASE + 0x3800U)
#define FLASH_R_BASE          (AHB1PERIPH_BASE + 0x3C00U)
#define DMA1_BASE             (AHB1PERIPH_BASE + 0x6000U)
#define DMA1_Stream0_BASE     (DMA1_BASE + 0x010U)
#define DMA1_Stream1_BASE     (DMA1_BASE + 0x028U)
#define DMA1_Stream2_BASE     (DMA1_BASE + 0x040U)
#define DMA1_Stream3_BASE     (DMA1_BASE + 0x058U)
#define DMA1_Stream4_BASE     (DMA1_BASE + 0x070U)
#define DMA1_Stream5_BASE     (DMA1_BASE + 0x088U)
#define DMA1_Stream6_BASE     (DMA1_BASE + 0x0A0U)
#define DMA1_Stream7_BASE     (DMA1_BASE + 0x0B8U)
#define DMA2_BASE             (AHB1PERIPH_BASE + 0x6400U)
#define DMA2_Stream0_BASE     (DMA2_BASE + 0x010U)
#define DMA2_Stream1_BASE     (DMA2_BASE + 0x028U)
#define DMA2_Stream2_BASE     (DMA2_BASE + 0x040U)
#define DMA2_Stream3_BASE     (DMA2_BASE + 0x058U)
#define DMA2_Stream4_BASE     (DMA2_BASE + 0x070U)
#define DMA2_Stream5_BASE     (DMA2_BASE + 0x088U)
#define DMA2_Stream6_BASE     (DMA2_BASE + 0x0A0U)
#define DMA2_Stream7_BASE     (DMA2_BASE + 0x0B8U)

/*!< AHB2 peripherals */
#define RNG_BASE              (AHB2PERIPH_BASE + 0x60800U)

/*!< FSMC Bankx registers base address */
#define FSMC_Bank1_R_BASE     (FSMC_R_BASE + 0x0000U)
#define FSMC_Bank1E_R_BASE    (FSMC_R_BASE + 0x0104U)
#define FSMC_Bank2_3_R_BASE   (FSMC_R_BASE + 0x0060U)
#define FSMC_Bank4_R_BASE     (FSMC_R_BASE + 0x00A0U)

/* Debug MCU registers base address */
#define DBGMCU_BASE           0xE0042000U

/*!< USB registers base address */
#define USB_OTG_HS_PERIPH_BASE               0x40040000U
#define USB_OTG_FS_PERIPH_BASE               0x50000000U

#define USB_OTG_GLOBAL_BASE                  0x000U
#define USB_OTG_DEVICE_BASE                  0x800U
#define USB_OTG_IN_ENDPOINT_BASE             0x900U
#define USB_OTG_OUT_ENDPOINT_BASE            0xB00U
#define USB_OTG_EP_REG_SIZE                  0x20U
#define USB_OTG_HOST_BASE                    0x400U
#define USB_OTG_HOST_PORT_BASE               0x440U
#define USB_OTG_HOST_CHANNEL_BASE            0x500U
#define USB_OTG_HOST_CHANNEL_SIZE            0x20U
#define USB_OTG_PCGCCTL_BASE                 0xE00U
#define USB_OTG_FIFO_BASE                    0x1000U
#define USB_OTG_FIFO_SIZE                    0x1000U

/**
  * @}
  */
  
/** @addtogroup Peripheral_declaration
  * @{
  */  
#define TIM2                ((TIM_TypeDef *) TIM2_BASE)
#define TIM3                ((TIM_TypeDef *) TIM3_BASE)
#define TIM4                ((TIM_TypeDef *) TIM4_BASE)
#define TIM5                ((TIM_TypeDef *) TIM5_BASE)
#define TIM6                ((TIM_TypeDef *) TIM6_BASE)
#define TIM7                ((TIM_TypeDef *) TIM7_BASE)
#define TIM12               ((TIM_TypeDef *) TIM12_BASE)
#define TIM13               ((TIM_TypeDef *) TIM13_BASE)
#define TIM14               ((TIM_TypeDef *) TIM14_BASE)
#define RTC                 ((RTC_TypeDef *) RTC_BASE)
#define WWDG                ((WWDG_TypeDef *) WWDG_BASE)
#define IWDG                ((IWDG_TypeDef *) IWDG_BASE)
#define SPI2                ((SPI_TypeDef *) SPI2_BASE)
#define SPI3                ((SPI_TypeDef *) SPI3_BASE)
#define USART2              ((USART_TypeDef *) USART2_BASE)
#define USART3              ((USART_TypeDef *) USART3_BASE)
#define UART4               ((USART_TypeDef *) UART4_BASE)
#define UART5               ((USART_TypeDef *) UART5_BASE)
#define I2C1                ((I2C_TypeDef *) I2C1_BASE)
#define I2C2                ((I2C_TypeDef *) I2C2_BASE)
#define I2C3                ((I2C_TypeDef *) I2C3_BASE)
#define CAN1                ((CAN_TypeDef *) CAN1_BASE)
#define CAN2                ((CAN_TypeDef *) CAN2_BASE)
#define PWR                 ((PWR_TypeDef *) PWR_BASE)
#define DAC                 ((DAC_TypeDef *) DAC_BASE)
#define TIM1                ((TIM_TypeDef *) TIM1_BASE)
#define TIM8                ((TIM_TypeDef *) TIM8_BASE)
#define USART1              ((USART_TypeDef *) USART1_BASE)
#define USART6              ((USART_TypeDef *) USART6_BASE)
#define ADC                 ((ADC_Common_TypeDef *) ADC_BASE)
#define ADC1                ((ADC_TypeDef *) ADC1_BASE)
#define ADC2                ((ADC_TypeDef *) ADC2_BASE)
#define ADC3                ((ADC_TypeDef *) ADC3_BASE)
#define SDIO                ((SDIO_TypeDef *) SDIO_BASE)
#define SPI1                ((SPI_TypeDef *) SPI1_BASE) 
#define SYSCFG              ((SYSCFG_TypeDef *) SYSCFG_BASE)
#define EXTI                ((EXTI_TypeDef *) EXTI_BASE)
#define TIM9                ((TIM_TypeDef *) TIM9_BASE)
#define TIM10               ((TIM_TypeDef *) TIM10_BASE)
#define TIM11               ((TIM_TypeDef *) TIM11_BASE)
#define GPIOA               ((GPIO_TypeDef *) GPIOA_BASE)
#define GPIOB               ((GPIO_TypeDef *) GPIOB_BASE)
#define GPIOC               ((GPIO_TypeDef *) GPIOC_BASE)
#define GPIOD               ((GPIO_TypeDef *) GPIOD_BASE)
#define GPIOE               ((GPIO_TypeDef *) GPIOE_BASE)
#define GPIOF               ((GPIO_TypeDef *) GPIOF_BASE)
#define GPIOG               ((GPIO_TypeDef *) GPIOG_BASE)
#define GPIOH               ((GPIO_TypeDef *) GPIOH_BASE)
#define GPIOI               ((GPIO_TypeDef *) GPIOI_BASE)
#define CRC                 ((CRC_TypeDef *) CRC_BASE)
#define RCC                 ((RCC_TypeDef *) RCC_BASE)
#define FLASH               ((FLASH_TypeDef *) FLASH_R_BASE)
#define DMA1                ((DMA_TypeDef *) DMA1_BASE)
#define DMA1_Stream0        ((DMA_Stream_TypeDef *) DMA1_Stream0_BASE)
#define DMA1_Stream1        ((DMA_Stream_TypeDef *) DMA1_Stream1_BASE)
#define DMA1_Stream2        ((DMA_Stream_TypeDef *) DMA1_Stream2_BASE)
#define DMA1_Stream3        ((DMA_Stream_TypeDef *) DMA1_Stream3_BASE)
#define DMA1_Stream4        ((DMA_Stream_TypeDef *) DMA1_Stream4_BASE)
#define DMA1_Stream5        ((DMA_Stream_TypeDef *) DMA1_Stream5_BASE)
#define DMA1_Stream6        ((DMA_Stream_TypeDef *) DMA1_Stream6_BASE)
#define DMA1_Stream7        ((DMA_Stream_TypeDef *) DMA1_Stream7_BASE)
#define DMA2                ((DMA_TypeDef *) DMA2_BASE)
#define DMA2_Stream0        ((DMA_Stream_TypeDef *) DMA2_Stream0_BASE)
#define DMA2_Stream1        ((DMA_Stream_TypeDef *) DMA2_Stream1_BASE)
#define DMA2_Stream2        ((DMA_Stream_TypeDef *) DMA2_Stream2_BASE)
#define DMA2_Stream3        ((DMA_Stream_TypeDef *) DMA2_Stream3_BASE)
#define DMA2_Stream4        ((DMA_Stream_TypeDef *) DMA2_Stream4_BASE)
#define DMA2_Stream5        ((DMA_Stream_TypeDef *) DMA2_Stream5_BASE)
#define DMA2_Stream6        ((DMA_Stream_TypeDef *) DMA2_Stream6_BASE)
#define DMA2_Stream7        ((DMA_Stream_TypeDef *) DMA2_Stream7_BASE)  
#define RNG                 ((RNG_TypeDef *) RNG_BASE)
#define FSMC_Bank1          ((FSMC_Bank1_TypeDef *) FSMC_Bank1_R_BASE)
#define FSMC_Bank1E         ((FSMC_Bank1E_TypeDef *) FSMC_Bank1E_R_BASE)
#define FSMC_Bank2_3        ((FSMC_Bank2_3_TypeDef *) FSMC_Bank2_3_R_BASE)
#define FSMC_Bank4          ((FSMC_Bank4_TypeDef *) FSMC_Bank4_R_BASE)

#define DBGMCU              ((DBGMCU_TypeDef *) DBGMCU_BASE)

#define USB_OTG_FS          ((USB_OTG_GlobalTypeDef *) USB_OTG_FS_PERIPH_BASE)
#define USB_OTG_HS          ((USB_OTG_GlobalTypeDef *) USB_OTG_HS_PERIPH_BASE)

/**
  * @}
  */

/** @addtogroup Exported_constants
  * @{
  */
  
  /** @addtogroup Peripheral_Registers_Bits_Definition
  * @{
  */
    
/******************************************************************************/
/*                         Peripheral Registers_Bits_Definition               */
/******************************************************************************/

/******************************************************************************/
/*                                                                            */
/*                        Analog to Digital Converter                         */
/*                                                                            */
/******************************************************************************/
/********************  Bit definition for ADC_SR register  ********************/
#define  ADC_SR_AWD                          0x00000001U       /*!<Analog watchdog flag */
#define  ADC_SR_EOC                          0x00000002U       /*!<End of conversion */
#define  ADC_SR_JEOC                         0x00000004U       /*!<Injected channel end of conversion */
#define  ADC_SR_JSTRT                        0x00000008U       /*!<Injected channel Start flag */
#define  ADC_SR_STRT                         0x00000010U       /*!<Regular channel Start flag */
#define  ADC_SR_OVR                          0x00000020U       /*!<Overrun flag */

/*******************  Bit definition for ADC_CR1 register  ********************/
#define  ADC_CR1_AWDCH                       0x0000001FU        /*!<AWDCH[4:0] bits (Analog watchdog channel select bits) */
#define  ADC_CR1_AWDCH_0                     0x00000001U        /*!<Bit 0 */
#define  ADC_CR1_AWDCH_1                     0x00000002U        /*!<Bit 1 */
#define  ADC_CR1_AWDCH_2                     0x00000004U        /*!<Bit 2 */
#define  ADC_CR1_AWDCH_3                     0x00000008U        /*!<Bit 3 */
#define  ADC_CR1_AWDCH_4                     0x00000010U        /*!<Bit 4 */
#define  ADC_CR1_EOCIE                       0x00000020U        /*!<Interrupt enable for EOC */
#define  ADC_CR1_AWDIE                       0x00000040U        /*!<AAnalog Watchdog interrupt enable */
#define  ADC_CR1_JEOCIE                      0x00000080U        /*!<Interrupt enable for injected channels */
#define  ADC_CR1_SCAN                        0x00000100U        /*!<Scan mode */
#define  ADC_CR1_AWDSGL                      0x00000200U        /*!<Enable the watchdog on a single channel in scan mode */
#define  ADC_CR1_JAUTO                       0x00000400U        /*!<Automatic injected group conversion */
#define  ADC_CR1_DISCEN                      0x00000800U        /*!<Discontinuous mode on regular channels */
#define  ADC_CR1_JDISCEN                     0x00001000U        /*!<Discontinuous mode on injected channels */
#define  ADC_CR1_DISCNUM                     0x0000E000U        /*!<DISCNUM[2:0] bits (Discontinuous mode channel count) */
#define  ADC_CR1_DISCNUM_0                   0x00002000U        /*!<Bit 0 */
#define  ADC_CR1_DISCNUM_1                   0x00004000U        /*!<Bit 1 */
#define  ADC_CR1_DISCNUM_2                   0x00008000U        /*!<Bit 2 */
#define  ADC_CR1_JAWDEN                      0x00400000U        /*!<Analog watchdog enable on injected channels */
#define  ADC_CR1_AWDEN                       0x00800000U        /*!<Analog watchdog enable on regular channels */
#define  ADC_CR1_RES                         0x03000000U        /*!<RES[2:0] bits (Resolution) */
#define  ADC_CR1_RES_0                       0x01000000U        /*!<Bit 0 */
#define  ADC_CR1_RES_1                       0x02000000U        /*!<Bit 1 */
#define  ADC_CR1_OVRIE                       0x04000000U         /*!<overrun interrupt enable */
  
/*******************  Bit definition for ADC_CR2 register  ********************/
#define  ADC_CR2_ADON                        0x00000001U        /*!<A/D Converter ON / OFF */
#define  ADC_CR2_CONT                        0x00000002U        /*!<Continuous Conversion */
#define  ADC_CR2_DMA                         0x00000100U        /*!<Direct Memory access mode */
#define  ADC_CR2_DDS                         0x00000200U        /*!<DMA disable selection (Single ADC) */
#define  ADC_CR2_EOCS                        0x00000400U        /*!<End of conversion selection */
#define  ADC_CR2_ALIGN                       0x00000800U        /*!<Data Alignment */
#define  ADC_CR2_JEXTSEL                     0x000F0000U        /*!<JEXTSEL[3:0] bits (External event select for injected group) */
#define  ADC_CR2_JEXTSEL_0                   0x00010000U        /*!<Bit 0 */
#define  ADC_CR2_JEXTSEL_1                   0x00020000U        /*!<Bit 1 */
#define  ADC_CR2_JEXTSEL_2                   0x00040000U        /*!<Bit 2 */
#define  ADC_CR2_JEXTSEL_3                   0x00080000U        /*!<Bit 3 */
#define  ADC_CR2_JEXTEN                      0x00300000U        /*!<JEXTEN[1:0] bits (External Trigger Conversion mode for injected channelsp) */
#define  ADC_CR2_JEXTEN_0                    0x00100000U        /*!<Bit 0 */
#define  ADC_CR2_JEXTEN_1                    0x00200000U        /*!<Bit 1 */
#define  ADC_CR2_JSWSTART                    0x00400000U        /*!<Start Conversion of injected channels */
#define  ADC_CR2_EXTSEL                      0x0F000000U        /*!<EXTSEL[3:0] bits (External Event Select for regular group) */
#define  ADC_CR2_EXTSEL_0                    0x01000000U        /*!<Bit 0 */
#define  ADC_CR2_EXTSEL_1                    0x02000000U        /*!<Bit 1 */
#define  ADC_CR2_EXTSEL_2                    0x04000000U        /*!<Bit 2 */
#define  ADC_CR2_EXTSEL_3                    0x08000000U        /*!<Bit 3 */
#define  ADC_CR2_EXTEN                       0x30000000U        /*!<EXTEN[1:0] bits (External Trigger Conversion mode for regular channelsp) */
#define  ADC_CR2_EXTEN_0                     0x10000000U        /*!<Bit 0 */
#define  ADC_CR2_EXTEN_1                     0x20000000U        /*!<Bit 1 */
#define  ADC_CR2_SWSTART                     0x40000000U        /*!<Start Conversion of regular channels */

/******************  Bit definition for ADC_SMPR1 register  *******************/
#define  ADC_SMPR1_SMP10                     0x00000007U        /*!<SMP10[2:0] bits (Channel 10 Sample time selection) */
#define  ADC_SMPR1_SMP10_0                   0x00000001U        /*!<Bit 0 */
#define  ADC_SMPR1_SMP10_1                   0x00000002U        /*!<Bit 1 */
#define  ADC_SMPR1_SMP10_2                   0x00000004U        /*!<Bit 2 */
#define  ADC_SMPR1_SMP11                     0x00000038U        /*!<SMP11[2:0] bits (Channel 11 Sample time selection) */
#define  ADC_SMPR1_SMP11_0                   0x00000008U        /*!<Bit 0 */
#define  ADC_SMPR1_SMP11_1                   0x00000010U        /*!<Bit 1 */
#define  ADC_SMPR1_SMP11_2                   0x00000020U        /*!<Bit 2 */
#define  ADC_SMPR1_SMP12                     0x000001C0U        /*!<SMP12[2:0] bits (Channel 12 Sample time selection) */
#define  ADC_SMPR1_SMP12_0                   0x00000040U        /*!<Bit 0 */
#define  ADC_SMPR1_SMP12_1                   0x00000080U        /*!<Bit 1 */
#define  ADC_SMPR1_SMP12_2                   0x00000100U        /*!<Bit 2 */
#define  ADC_SMPR1_SMP13                     0x00000E00U        /*!<SMP13[2:0] bits (Channel 13 Sample time selection) */
#define  ADC_SMPR1_SMP13_0                   0x00000200U        /*!<Bit 0 */
#define  ADC_SMPR1_SMP13_1                   0x00000400U        /*!<Bit 1 */
#define  ADC_SMPR1_SMP13_2                   0x00000800U        /*!<Bit 2 */
#define  ADC_SMPR1_SMP14                     0x00007000U        /*!<SMP14[2:0] bits (Channel 14 Sample time selection) */
#define  ADC_SMPR1_SMP14_0                   0x00001000U        /*!<Bit 0 */
#define  ADC_SMPR1_SMP14_1                   0x00002000U        /*!<Bit 1 */
#define  ADC_SMPR1_SMP14_2                   0x00004000U        /*!<Bit 2 */
#define  ADC_SMPR1_SMP15                     0x00038000U        /*!<SMP15[2:0] bits (Channel 15 Sample time selection) */
#define  ADC_SMPR1_SMP15_0                   0x00008000U        /*!<Bit 0 */
#define  ADC_SMPR1_SMP15_1                   0x00010000U        /*!<Bit 1 */
#define  ADC_SMPR1_SMP15_2                   0x00020000U        /*!<Bit 2 */
#define  ADC_SMPR1_SMP16                     0x001C0000U        /*!<SMP16[2:0] bits (Channel 16 Sample time selection) */
#define  ADC_SMPR1_SMP16_0                   0x00040000U        /*!<Bit 0 */
#define  ADC_SMPR1_SMP16_1                   0x00080000U        /*!<Bit 1 */
#define  ADC_SMPR1_SMP16_2                   0x00100000U        /*!<Bit 2 */
#define  ADC_SMPR1_SMP17                     0x00E00000U        /*!<SMP17[2:0] bits (Channel 17 Sample time selection) */
#define  ADC_SMPR1_SMP17_0                   0x00200000U        /*!<Bit 0 */
#define  ADC_SMPR1_SMP17_1                   0x00400000U        /*!<Bit 1 */
#define  ADC_SMPR1_SMP17_2                   0x00800000U        /*!<Bit 2 */
#define  ADC_SMPR1_SMP18                     0x07000000U        /*!<SMP18[2:0] bits (Channel 18 Sample time selection) */
#define  ADC_SMPR1_SMP18_0                   0x01000000U        /*!<Bit 0 */
#define  ADC_SMPR1_SMP18_1                   0x02000000U        /*!<Bit 1 */
#define  ADC_SMPR1_SMP18_2                   0x04000000U        /*!<Bit 2 */

/******************  Bit definition for ADC_SMPR2 register  *******************/
#define  ADC_SMPR2_SMP0                      0x00000007U        /*!<SMP0[2:0] bits (Channel 0 Sample time selection) */
#define  ADC_SMPR2_SMP0_0                    0x00000001U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP0_1                    0x00000002U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP0_2                    0x00000004U        /*!<Bit 2 */
#define  ADC_SMPR2_SMP1                      0x00000038U        /*!<SMP1[2:0] bits (Channel 1 Sample time selection) */
#define  ADC_SMPR2_SMP1_0                    0x00000008U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP1_1                    0x00000010U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP1_2                    0x00000020U        /*!<Bit 2 */
#define  ADC_SMPR2_SMP2                      0x000001C0U        /*!<SMP2[2:0] bits (Channel 2 Sample time selection) */
#define  ADC_SMPR2_SMP2_0                    0x00000040U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP2_1                    0x00000080U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP2_2                    0x00000100U        /*!<Bit 2 */
#define  ADC_SMPR2_SMP3                      0x00000E00U        /*!<SMP3[2:0] bits (Channel 3 Sample time selection) */
#define  ADC_SMPR2_SMP3_0                    0x00000200U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP3_1                    0x00000400U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP3_2                    0x00000800U        /*!<Bit 2 */
#define  ADC_SMPR2_SMP4                      0x00007000U        /*!<SMP4[2:0] bits (Channel 4 Sample time selection) */
#define  ADC_SMPR2_SMP4_0                    0x00001000U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP4_1                    0x00002000U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP4_2                    0x00004000U        /*!<Bit 2 */
#define  ADC_SMPR2_SMP5                      0x00038000U        /*!<SMP5[2:0] bits (Channel 5 Sample time selection) */
#define  ADC_SMPR2_SMP5_0                    0x00008000U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP5_1                    0x00010000U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP5_2                    0x00020000U        /*!<Bit 2 */
#define  ADC_SMPR2_SMP6                      0x001C0000U        /*!<SMP6[2:0] bits (Channel 6 Sample time selection) */
#define  ADC_SMPR2_SMP6_0                    0x00040000U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP6_1                    0x00080000U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP6_2                    0x00100000U        /*!<Bit 2 */
#define  ADC_SMPR2_SMP7                      0x00E00000U        /*!<SMP7[2:0] bits (Channel 7 Sample time selection) */
#define  ADC_SMPR2_SMP7_0                    0x00200000U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP7_1                    0x00400000U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP7_2                    0x00800000U        /*!<Bit 2 */
#define  ADC_SMPR2_SMP8                      0x07000000U        /*!<SMP8[2:0] bits (Channel 8 Sample time selection) */
#define  ADC_SMPR2_SMP8_0                    0x01000000U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP8_1                    0x02000000U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP8_2                    0x04000000U        /*!<Bit 2 */
#define  ADC_SMPR2_SMP9                      0x38000000U        /*!<SMP9[2:0] bits (Channel 9 Sample time selection) */
#define  ADC_SMPR2_SMP9_0                    0x08000000U        /*!<Bit 0 */
#define  ADC_SMPR2_SMP9_1                    0x10000000U        /*!<Bit 1 */
#define  ADC_SMPR2_SMP9_2                    0x20000000U        /*!<Bit 2 */

/******************  Bit definition for ADC_JOFR1 register  *******************/
#define  ADC_JOFR1_JOFFSET1                  0x00000FFFU        /*!<Data offset for injected channel 1 */

/******************  Bit definition for ADC_JOFR2 register  *******************/
#define  ADC_JOFR2_JOFFSET2                  0x00000FFFU        /*!<Data offset for injected channel 2 */

/******************  Bit definition for ADC_JOFR3 register  *******************/
#define  ADC_JOFR3_JOFFSET3                  0x00000FFFU        /*!<Data offset for injected channel 3 */

/******************  Bit definition for ADC_JOFR4 register  *******************/
#define  ADC_JOFR4_JOFFSET4                  0x00000FFFU        /*!<Data offset for injected channel 4 */

/*******************  Bit definition for ADC_HTR register  ********************/
#define  ADC_HTR_HT                          0x00000FFFU        /*!<Analog watchdog high threshold */

/*******************  Bit definition for ADC_LTR register  ********************/
#define  ADC_LTR_LT                          0x00000FFFU         /*!<Analog watchdog low threshold */

/*******************  Bit definition for ADC_SQR1 register  *******************/
#define  ADC_SQR1_SQ13                       0x0000001FU        /*!<SQ13[4:0] bits (13th conversion in regular sequence) */
#define  ADC_SQR1_SQ13_0                     0x00000001U        /*!<Bit 0 */
#define  ADC_SQR1_SQ13_1                     0x00000002U        /*!<Bit 1 */
#define  ADC_SQR1_SQ13_2                     0x00000004U        /*!<Bit 2 */
#define  ADC_SQR1_SQ13_3                     0x00000008U        /*!<Bit 3 */
#define  ADC_SQR1_SQ13_4                     0x00000010U        /*!<Bit 4 */
#define  ADC_SQR1_SQ14                       0x000003E0U        /*!<SQ14[4:0] bits (14th conversion in regular sequence) */
#define  ADC_SQR1_SQ14_0                     0x00000020U        /*!<Bit 0 */
#define  ADC_SQR1_SQ14_1                     0x00000040U        /*!<Bit 1 */
#define  ADC_SQR1_SQ14_2                     0x00000080U        /*!<Bit 2 */
#define  ADC_SQR1_SQ14_3                     0x00000100U        /*!<Bit 3 */
#define  ADC_SQR1_SQ14_4                     0x00000200U        /*!<Bit 4 */
#define  ADC_SQR1_SQ15                       0x00007C00U        /*!<SQ15[4:0] bits (15th conversion in regular sequence) */
#define  ADC_SQR1_SQ15_0                     0x00000400U        /*!<Bit 0 */
#define  ADC_SQR1_SQ15_1                     0x00000800U        /*!<Bit 1 */
#define  ADC_SQR1_SQ15_2                     0x00001000U        /*!<Bit 2 */
#define  ADC_SQR1_SQ15_3                     0x00002000U        /*!<Bit 3 */
#define  ADC_SQR1_SQ15_4                     0x00004000U        /*!<Bit 4 */
#define  ADC_SQR1_SQ16                       0x000F8000U        /*!<SQ16[4:0] bits (16th conversion in regular sequence) */
#define  ADC_SQR1_SQ16_0                     0x00008000U        /*!<Bit 0 */
#define  ADC_SQR1_SQ16_1                     0x00010000U        /*!<Bit 1 */
#define  ADC_SQR1_SQ16_2                     0x00020000U        /*!<Bit 2 */
#define  ADC_SQR1_SQ16_3                     0x00040000U        /*!<Bit 3 */
#define  ADC_SQR1_SQ16_4                     0x00080000U        /*!<Bit 4 */
#define  ADC_SQR1_L                          0x00F00000U        /*!<L[3:0] bits (Regular channel sequence length) */
#define  ADC_SQR1_L_0                        0x00100000U        /*!<Bit 0 */
#define  ADC_SQR1_L_1                        0x00200000U        /*!<Bit 1 */
#define  ADC_SQR1_L_2                        0x00400000U        /*!<Bit 2 */
#define  ADC_SQR1_L_3                        0x00800000U        /*!<Bit 3 */

/*******************  Bit definition for ADC_SQR2 register  *******************/
#define  ADC_SQR2_SQ7                        0x0000001FU        /*!<SQ7[4:0] bits (7th conversion in regular sequence) */
#define  ADC_SQR2_SQ7_0                      0x00000001U        /*!<Bit 0 */
#define  ADC_SQR2_SQ7_1                      0x00000002U        /*!<Bit 1 */
#define  ADC_SQR2_SQ7_2                      0x00000004U        /*!<Bit 2 */
#define  ADC_SQR2_SQ7_3                      0x00000008U        /*!<Bit 3 */
#define  ADC_SQR2_SQ7_4                      0x00000010U        /*!<Bit 4 */
#define  ADC_SQR2_SQ8                        0x000003E0U        /*!<SQ8[4:0] bits (8th conversion in regular sequence) */
#define  ADC_SQR2_SQ8_0                      0x00000020U        /*!<Bit 0 */
#define  ADC_SQR2_SQ8_1                      0x00000040U        /*!<Bit 1 */
#define  ADC_SQR2_SQ8_2                      0x00000080U        /*!<Bit 2 */
#define  ADC_SQR2_SQ8_3                      0x00000100U        /*!<Bit 3 */
#define  ADC_SQR2_SQ8_4                      0x00000200U        /*!<Bit 4 */
#define  ADC_SQR2_SQ9                        0x00007C00U        /*!<SQ9[4:0] bits (9th conversion in regular sequence) */
#define  ADC_SQR2_SQ9_0                      0x00000400U        /*!<Bit 0 */
#define  ADC_SQR2_SQ9_1                      0x00000800U        /*!<Bit 1 */
#define  ADC_SQR2_SQ9_2                      0x00001000U        /*!<Bit 2 */
#define  ADC_SQR2_SQ9_3                      0x00002000U        /*!<Bit 3 */
#define  ADC_SQR2_SQ9_4                      0x00004000U        /*!<Bit 4 */
#define  ADC_SQR2_SQ10                       0x000F8000U        /*!<SQ10[4:0] bits (10th conversion in regular sequence) */
#define  ADC_SQR2_SQ10_0                     0x00008000U        /*!<Bit 0 */
#define  ADC_SQR2_SQ10_1                     0x00010000U        /*!<Bit 1 */
#define  ADC_SQR2_SQ10_2                     0x00020000U        /*!<Bit 2 */
#define  ADC_SQR2_SQ10_3                     0x00040000U        /*!<Bit 3 */
#define  ADC_SQR2_SQ10_4                     0x00080000U        /*!<Bit 4 */
#define  ADC_SQR2_SQ11                       0x01F00000U        /*!<SQ11[4:0] bits (11th conversion in regular sequence) */
#define  ADC_SQR2_SQ11_0                     0x00100000U        /*!<Bit 0 */
#define  ADC_SQR2_SQ11_1                     0x00200000U        /*!<Bit 1 */
#define  ADC_SQR2_SQ11_2                     0x00400000U        /*!<Bit 2 */
#define  ADC_SQR2_SQ11_3                     0x00800000U        /*!<Bit 3 */
#define  ADC_SQR2_SQ11_4                     0x01000000U        /*!<Bit 4 */
#define  ADC_SQR2_SQ12                       0x3E000000U        /*!<SQ12[4:0] bits (12th conversion in regular sequence) */
#define  ADC_SQR2_SQ12_0                     0x02000000U        /*!<Bit 0 */
#define  ADC_SQR2_SQ12_1                     0x04000000U        /*!<Bit 1 */
#define  ADC_SQR2_SQ12_2                     0x08000000U        /*!<Bit 2 */
#define  ADC_SQR2_SQ12_3                     0x10000000U        /*!<Bit 3 */
#define  ADC_SQR2_SQ12_4                     0x20000000U        /*!<Bit 4 */

/*******************  Bit definition for ADC_SQR3 register  *******************/
#define  ADC_SQR3_SQ1                        0x0000001FU        /*!<SQ1[4:0] bits (1st conversion in regular sequence) */
#define  ADC_SQR3_SQ1_0                      0x00000001U        /*!<Bit 0 */
#define  ADC_SQR3_SQ1_1                      0x00000002U        /*!<Bit 1 */
#define  ADC_SQR3_SQ1_2                      0x00000004U        /*!<Bit 2 */
#define  ADC_SQR3_SQ1_3                      0x00000008U        /*!<Bit 3 */
#define  ADC_SQR3_SQ1_4                      0x00000010U        /*!<Bit 4 */
#define  ADC_SQR3_SQ2                        0x000003E0U        /*!<SQ2[4:0] bits (2nd conversion in regular sequence) */
#define  ADC_SQR3_SQ2_0                      0x00000020U        /*!<Bit 0 */
#define  ADC_SQR3_SQ2_1                      0x00000040U        /*!<Bit 1 */
#define  ADC_SQR3_SQ2_2                      0x00000080U        /*!<Bit 2 */
#define  ADC_SQR3_SQ2_3                      0x00000100U        /*!<Bit 3 */
#define  ADC_SQR3_SQ2_4                      0x00000200U        /*!<Bit 4 */
#define  ADC_SQR3_SQ3                        0x00007C00U        /*!<SQ3[4:0] bits (3rd conversion in regular sequence) */
#define  ADC_SQR3_SQ3_0                      0x00000400U        /*!<Bit 0 */
#define  ADC_SQR3_SQ3_1                      0x00000800U        /*!<Bit 1 */
#define  ADC_SQR3_SQ3_2                      0x00001000U        /*!<Bit 2 */
#define  ADC_SQR3_SQ3_3                      0x00002000U        /*!<Bit 3 */
#define  ADC_SQR3_SQ3_4                      0x00004000U        /*!<Bit 4 */
#define  ADC_SQR3_SQ4                        0x000F8000U        /*!<SQ4[4:0] bits (4th conversion in regular sequence) */
#define  ADC_SQR3_SQ4_0                      0x00008000U        /*!<Bit 0 */
#define  ADC_SQR3_SQ4_1                      0x00010000U        /*!<Bit 1 */
#define  ADC_SQR3_SQ4_2                      0x00020000U        /*!<Bit 2 */
#define  ADC_SQR3_SQ4_3                      0x00040000U        /*!<Bit 3 */
#define  ADC_SQR3_SQ4_4                      0x00080000U        /*!<Bit 4 */
#define  ADC_SQR3_SQ5                        0x01F00000U        /*!<SQ5[4:0] bits (5th conversion in regular sequence) */
#define  ADC_SQR3_SQ5_0                      0x00100000U        /*!<Bit 0 */
#define  ADC_SQR3_SQ5_1                      0x00200000U        /*!<Bit 1 */
#define  ADC_SQR3_SQ5_2                      0x00400000U        /*!<Bit 2 */
#define  ADC_SQR3_SQ5_3                      0x00800000U        /*!<Bit 3 */
#define  ADC_SQR3_SQ5_4                      0x01000000U        /*!<Bit 4 */
#define  ADC_SQR3_SQ6                        0x3E000000U        /*!<SQ6[4:0] bits (6th conversion in regular sequence) */
#define  ADC_SQR3_SQ6_0                      0x02000000U        /*!<Bit 0 */
#define  ADC_SQR3_SQ6_1                      0x04000000U        /*!<Bit 1 */
#define  ADC_SQR3_SQ6_2                      0x08000000U        /*!<Bit 2 */
#define  ADC_SQR3_SQ6_3                      0x10000000U        /*!<Bit 3 */
#define  ADC_SQR3_SQ6_4                      0x20000000U        /*!<Bit 4 */

/*******************  Bit definition for ADC_JSQR register  *******************/
#define  ADC_JSQR_JSQ1                       0x0000001FU        /*!<JSQ1[4:0] bits (1st conversion in injected sequence) */  
#define  ADC_JSQR_JSQ1_0                     0x00000001U        /*!<Bit 0 */
#define  ADC_JSQR_JSQ1_1                     0x00000002U        /*!<Bit 1 */
#define  ADC_JSQR_JSQ1_2                     0x00000004U        /*!<Bit 2 */
#define  ADC_JSQR_JSQ1_3                     0x00000008U        /*!<Bit 3 */
#define  ADC_JSQR_JSQ1_4                     0x00000010U        /*!<Bit 4 */
#define  ADC_JSQR_JSQ2                       0x000003E0U        /*!<JSQ2[4:0] bits (2nd conversion in injected sequence) */
#define  ADC_JSQR_JSQ2_0                     0x00000020U        /*!<Bit 0 */
#define  ADC_JSQR_JSQ2_1                     0x00000040U        /*!<Bit 1 */
#define  ADC_JSQR_JSQ2_2                     0x00000080U        /*!<Bit 2 */
#define  ADC_JSQR_JSQ2_3                     0x00000100U        /*!<Bit 3 */
#define  ADC_JSQR_JSQ2_4                     0x00000200U        /*!<Bit 4 */
#define  ADC_JSQR_JSQ3                       0x00007C00U        /*!<JSQ3[4:0] bits (3rd conversion in injected sequence) */
#define  ADC_JSQR_JSQ3_0                     0x00000400U        /*!<Bit 0 */
#define  ADC_JSQR_JSQ3_1                     0x00000800U        /*!<Bit 1 */
#define  ADC_JSQR_JSQ3_2                     0x00001000U        /*!<Bit 2 */
#define  ADC_JSQR_JSQ3_3                     0x00002000U        /*!<Bit 3 */
#define  ADC_JSQR_JSQ3_4                     0x00004000U        /*!<Bit 4 */
#define  ADC_JSQR_JSQ4                       0x000F8000U        /*!<JSQ4[4:0] bits (4th conversion in injected sequence) */
#define  ADC_JSQR_JSQ4_0                     0x00008000U        /*!<Bit 0 */
#define  ADC_JSQR_JSQ4_1                     0x00010000U        /*!<Bit 1 */
#define  ADC_JSQR_JSQ4_2                     0x00020000U        /*!<Bit 2 */
#define  ADC_JSQR_JSQ4_3                     0x00040000U        /*!<Bit 3 */
#define  ADC_JSQR_JSQ4_4                     0x00080000U        /*!<Bit 4 */
#define  ADC_JSQR_JL                         0x00300000U        /*!<JL[1:0] bits (Injected Sequence length) */
#define  ADC_JSQR_JL_0                       0x00100000U        /*!<Bit 0 */
#define  ADC_JSQR_JL_1                       0x00200000U        /*!<Bit 1 */

/*******************  Bit definition for ADC_JDR1 register  *******************/
#define  ADC_JDR1_JDATA                      0x0000FFFFU        /*!<Injected data */

/*******************  Bit definition for ADC_JDR2 register  *******************/
#define  ADC_JDR2_JDATA                      0x0000FFFFU        /*!<Injected data */

/*******************  Bit definition for ADC_JDR3 register  *******************/
#define  ADC_JDR3_JDATA                      0x0000FFFFU        /*!<Injected data */

/*******************  Bit definition for ADC_JDR4 register  *******************/
#define  ADC_JDR4_JDATA                      0x0000FFFFU        /*!<Injected data */

/********************  Bit definition for ADC_DR register  ********************/
#define  ADC_DR_DATA                         0x0000FFFFU        /*!<Regular data */
#define  ADC_DR_ADC2DATA                     0xFFFF0000U        /*!<ADC2 data */

/*******************  Bit definition for ADC_CSR register  ********************/
#define  ADC_CSR_AWD1                        0x00000001U        /*!<ADC1 Analog watchdog flag */
#define  ADC_CSR_EOC1                        0x00000002U        /*!<ADC1 End of conversion */
#define  ADC_CSR_JEOC1                       0x00000004U        /*!<ADC1 Injected channel end of conversion */
#define  ADC_CSR_JSTRT1                      0x00000008U        /*!<ADC1 Injected channel Start flag */
#define  ADC_CSR_STRT1                       0x00000010U        /*!<ADC1 Regular channel Start flag */
#define  ADC_CSR_OVR1                        0x00000020U        /*!<ADC1 DMA overrun  flag */
#define  ADC_CSR_AWD2                        0x00000100U        /*!<ADC2 Analog watchdog flag */
#define  ADC_CSR_EOC2                        0x00000200U        /*!<ADC2 End of conversion */
#define  ADC_CSR_JEOC2                       0x00000400U        /*!<ADC2 Injected channel end of conversion */
#define  ADC_CSR_JSTRT2                      0x00000800U        /*!<ADC2 Injected channel Start flag */
#define  ADC_CSR_STRT2                       0x00001000U        /*!<ADC2 Regular channel Start flag */
#define  ADC_CSR_OVR2                        0x00002000U        /*!<ADC2 DMA overrun  flag */
#define  ADC_CSR_AWD3                        0x00010000U        /*!<ADC3 Analog watchdog flag */
#define  ADC_CSR_EOC3                        0x00020000U        /*!<ADC3 End of conversion */
#define  ADC_CSR_JEOC3                       0x00040000U        /*!<ADC3 Injected channel end of conversion */
#define  ADC_CSR_JSTRT3                      0x00080000U        /*!<ADC3 Injected channel Start flag */
#define  ADC_CSR_STRT3                       0x00100000U        /*!<ADC3 Regular channel Start flag */
#define  ADC_CSR_OVR3                        0x00200000U        /*!<ADC3 DMA overrun  flag */

/* Legacy defines */
#define  ADC_CSR_DOVR1                        ADC_CSR_OVR1
#define  ADC_CSR_DOVR2                        ADC_CSR_OVR2
#define  ADC_CSR_DOVR3                        ADC_CSR_OVR3

/*******************  Bit definition for ADC_CCR register  ********************/
#define  ADC_CCR_MULTI                       0x0000001FU        /*!<MULTI[4:0] bits (Multi-ADC mode selection) */  
#define  ADC_CCR_MULTI_0                     0x00000001U        /*!<Bit 0 */
#define  ADC_CCR_MULTI_1                     0x00000002U        /*!<Bit 1 */
#define  ADC_CCR_MULTI_2                     0x00000004U        /*!<Bit 2 */
#define  ADC_CCR_MULTI_3                     0x00000008U        /*!<Bit 3 */
#define  ADC_CCR_MULTI_4                     0x00000010U        /*!<Bit 4 */
#define  ADC_CCR_DELAY                       0x00000F00U        /*!<DELAY[3:0] bits (Delay between 2 sampling phases) */  
#define  ADC_CCR_DELAY_0                     0x00000100U        /*!<Bit 0 */
#define  ADC_CCR_DELAY_1                     0x00000200U        /*!<Bit 1 */
#define  ADC_CCR_DELAY_2                     0x00000400U        /*!<Bit 2 */
#define  ADC_CCR_DELAY_3                     0x00000800U        /*!<Bit 3 */
#define  ADC_CCR_DDS                         0x00002000U        /*!<DMA disable selection (Multi-ADC mode) */
#define  ADC_CCR_DMA                         0x0000C000U        /*!<DMA[1:0] bits (Direct Memory Access mode for multimode) */  
#define  ADC_CCR_DMA_0                       0x00004000U        /*!<Bit 0 */
#define  ADC_CCR_DMA_1                       0x00008000U        /*!<Bit 1 */
#define  ADC_CCR_ADCPRE                      0x00030000U        /*!<ADCPRE[1:0] bits (ADC prescaler) */  
#define  ADC_CCR_ADCPRE_0                    0x00010000U        /*!<Bit 0 */
#define  ADC_CCR_ADCPRE_1                    0x00020000U        /*!<Bit 1 */
#define  ADC_CCR_VBATE                       0x00400000U        /*!<VBAT Enable */
#define  ADC_CCR_TSVREFE                     0x00800000U        /*!<Temperature Sensor and VREFINT Enable */

/*******************  Bit definition for ADC_CDR register  ********************/
#define  ADC_CDR_DATA1                      0x0000FFFFU         /*!<1st data of a pair of regular conversions */
#define  ADC_CDR_DATA2                      0xFFFF0000U         /*!<2nd data of a pair of regular conversions */

/******************************************************************************/
/*                                                                            */
/*                         Controller Area Network                            */
/*                                                                            */
/******************************************************************************/
/*!<CAN control and status registers */
/*******************  Bit definition for CAN_MCR register  ********************/
#define  CAN_MCR_INRQ                        0x00000001U        /*!<Initialization Request */
#define  CAN_MCR_SLEEP                       0x00000002U        /*!<Sleep Mode Request */
#define  CAN_MCR_TXFP                        0x00000004U        /*!<Transmit FIFO Priority */
#define  CAN_MCR_RFLM                        0x00000008U        /*!<Receive FIFO Locked Mode */
#define  CAN_MCR_NART                        0x00000010U        /*!<No Automatic Retransmission */
#define  CAN_MCR_AWUM                        0x00000020U        /*!<Automatic Wakeup Mode */
#define  CAN_MCR_ABOM                        0x00000040U        /*!<Automatic Bus-Off Management */
#define  CAN_MCR_TTCM                        0x00000080U        /*!<Time Triggered Communication Mode */
#define  CAN_MCR_RESET                       0x00008000U        /*!<bxCAN software master reset */
#define  CAN_MCR_DBF                         0x00010000U        /*!<bxCAN Debug freeze */
/*******************  Bit definition for CAN_MSR register  ********************/
#define  CAN_MSR_INAK                        0x00000001U        /*!<Initialization Acknowledge */
#define  CAN_MSR_SLAK                        0x00000002U        /*!<Sleep Acknowledge */
#define  CAN_MSR_ERRI                        0x00000004U        /*!<Error Interrupt */
#define  CAN_MSR_WKUI                        0x00000008U        /*!<Wakeup Interrupt */
#define  CAN_MSR_SLAKI                       0x00000010U        /*!<Sleep Acknowledge Interrupt */
#define  CAN_MSR_TXM                         0x00000100U        /*!<Transmit Mode */
#define  CAN_MSR_RXM                         0x00000200U        /*!<Receive Mode */
#define  CAN_MSR_SAMP                        0x00000400U        /*!<Last Sample Point */
#define  CAN_MSR_RX                          0x00000800U        /*!<CAN Rx Signal */

/*******************  Bit definition for CAN_TSR register  ********************/
#define  CAN_TSR_RQCP0                       0x00000001U        /*!<Request Completed Mailbox0 */
#define  CAN_TSR_TXOK0                       0x00000002U        /*!<Transmission OK of Mailbox0 */
#define  CAN_TSR_ALST0                       0x00000004U        /*!<Arbitration Lost for Mailbox0 */
#define  CAN_TSR_TERR0                       0x00000008U        /*!<Transmission Error of Mailbox0 */
#define  CAN_TSR_ABRQ0                       0x00000080U        /*!<Abort Request for Mailbox0 */
#define  CAN_TSR_RQCP1                       0x00000100U        /*!<Request Completed Mailbox1 */
#define  CAN_TSR_TXOK1                       0x00000200U        /*!<Transmission OK of Mailbox1 */
#define  CAN_TSR_ALST1                       0x00000400U        /*!<Arbitration Lost for Mailbox1 */
#define  CAN_TSR_TERR1                       0x00000800U        /*!<Transmission Error of Mailbox1 */
#define  CAN_TSR_ABRQ1                       0x00008000U        /*!<Abort Request for Mailbox 1 */
#define  CAN_TSR_RQCP2                       0x00010000U        /*!<Request Completed Mailbox2 */
#define  CAN_TSR_TXOK2                       0x00020000U        /*!<Transmission OK of Mailbox 2 */
#define  CAN_TSR_ALST2                       0x00040000U        /*!<Arbitration Lost for mailbox 2 */
#define  CAN_TSR_TERR2                       0x00080000U        /*!<Transmission Error of Mailbox 2 */
#define  CAN_TSR_ABRQ2                       0x00800000U        /*!<Abort Request for Mailbox 2 */
#define  CAN_TSR_CODE                        0x03000000U        /*!<Mailbox Code */

#define  CAN_TSR_TME                         0x1C000000U        /*!<TME[2:0] bits */
#define  CAN_TSR_TME0                        0x04000000U        /*!<Transmit Mailbox 0 Empty */
#define  CAN_TSR_TME1                        0x08000000U        /*!<Transmit Mailbox 1 Empty */
#define  CAN_TSR_TME2                        0x10000000U        /*!<Transmit Mailbox 2 Empty */

#define  CAN_TSR_LOW                         0xE0000000U        /*!<LOW[2:0] bits */
#define  CAN_TSR_LOW0                        0x20000000U        /*!<Lowest Priority Flag for Mailbox 0 */
#define  CAN_TSR_LOW1                        0x40000000U        /*!<Lowest Priority Flag for Mailbox 1 */
#define  CAN_TSR_LOW2                        0x80000000U        /*!<Lowest Priority Flag for Mailbox 2 */

/*******************  Bit definition for CAN_RF0R register  *******************/
#define  CAN_RF0R_FMP0                       0x00000003U        /*!<FIFO 0 Message Pending */
#define  CAN_RF0R_FULL0                      0x00000008U        /*!<FIFO 0 Full */
#define  CAN_RF0R_FOVR0                      0x00000010U        /*!<FIFO 0 Overrun */
#define  CAN_RF0R_RFOM0                      0x00000020U        /*!<Release FIFO 0 Output Mailbox */

/*******************  Bit definition for CAN_RF1R register  *******************/
#define  CAN_RF1R_FMP1                       0x00000003U         /*!<FIFO 1 Message Pending */
#define  CAN_RF1R_FULL1                      0x00000008U         /*!<FIFO 1 Full */
#define  CAN_RF1R_FOVR1                      0x00000010U         /*!<FIFO 1 Overrun */
#define  CAN_RF1R_RFOM1                      0x00000020U         /*!<Release FIFO 1 Output Mailbox */

/********************  Bit definition for CAN_IER register  *******************/
#define  CAN_IER_TMEIE                       0x00000001U        /*!<Transmit Mailbox Empty Interrupt Enable */
#define  CAN_IER_FMPIE0                      0x00000002U        /*!<FIFO Message Pending Interrupt Enable */
#define  CAN_IER_FFIE0                       0x00000004U        /*!<FIFO Full Interrupt Enable */
#define  CAN_IER_FOVIE0                      0x00000008U        /*!<FIFO Overrun Interrupt Enable */
#define  CAN_IER_FMPIE1                      0x00000010U        /*!<FIFO Message Pending Interrupt Enable */
#define  CAN_IER_FFIE1                       0x00000020U        /*!<FIFO Full Interrupt Enable */
#define  CAN_IER_FOVIE1                      0x00000040U        /*!<FIFO Overrun Interrupt Enable */
#define  CAN_IER_EWGIE                       0x00000100U        /*!<Error Warning Interrupt Enable */
#define  CAN_IER_EPVIE                       0x00000200U        /*!<Error Passive Interrupt Enable */
#define  CAN_IER_BOFIE                       0x00000400U        /*!<Bus-Off Interrupt Enable */
#define  CAN_IER_LECIE                       0x00000800U        /*!<Last Error Code Interrupt Enable */
#define  CAN_IER_ERRIE                       0x00008000U        /*!<Error Interrupt Enable */
#define  CAN_IER_WKUIE                       0x00010000U        /*!<Wakeup Interrupt Enable */
#define  CAN_IER_SLKIE                       0x00020000U        /*!<Sleep Interrupt Enable */

/********************  Bit definition for CAN_ESR register  *******************/
#define  CAN_ESR_EWGF                        0x00000001U        /*!<Error Warning Flag */
#define  CAN_ESR_EPVF                        0x00000002U        /*!<Error Passive Flag */
#define  CAN_ESR_BOFF                        0x00000004U        /*!<Bus-Off Flag */

#define  CAN_ESR_LEC                         0x00000070U        /*!<LEC[2:0] bits (Last Error Code) */
#define  CAN_ESR_LEC_0                       0x00000010U        /*!<Bit 0 */
#define  CAN_ESR_LEC_1                       0x00000020U        /*!<Bit 1 */
#define  CAN_ESR_LEC_2                       0x00000040U        /*!<Bit 2 */

#define  CAN_ESR_TEC                         0x00FF0000U        /*!<Least significant byte of the 9-bit Transmit Error Counter */
#define  CAN_ESR_REC                         0xFF000000U        /*!<Receive Error Counter */

/*******************  Bit definition for CAN_BTR register  ********************/
#define  CAN_BTR_BRP                         0x000003FFU        /*!<Baud Rate Prescaler */
#define  CAN_BTR_TS1                         0x000F0000U        /*!<Time Segment 1 */
#define  CAN_BTR_TS1_0                       0x00010000U        /*!<Bit 0 */
#define  CAN_BTR_TS1_1                       0x00020000U        /*!<Bit 1 */
#define  CAN_BTR_TS1_2                       0x00040000U        /*!<Bit 2 */
#define  CAN_BTR_TS1_3                       0x00080000U        /*!<Bit 3 */
#define  CAN_BTR_TS2                         0x00700000U        /*!<Time Segment 2 */
#define  CAN_BTR_TS2_0                       0x00100000U        /*!<Bit 0 */
#define  CAN_BTR_TS2_1                       0x00200000U        /*!<Bit 1 */
#define  CAN_BTR_TS2_2                       0x00400000U        /*!<Bit 2 */
#define  CAN_BTR_SJW                         0x03000000U        /*!<Resynchronization Jump Width */
#define  CAN_BTR_SJW_0                       0x01000000U        /*!<Bit 0 */
#define  CAN_BTR_SJW_1                       0x02000000U        /*!<Bit 1 */
#define  CAN_BTR_LBKM                        0x40000000U        /*!<Loop Back Mode (Debug) */
#define  CAN_BTR_SILM                        0x80000000U        /*!<Silent Mode */


/*!<Mailbox registers */
/******************  Bit definition for CAN_TI0R register  ********************/
#define  CAN_TI0R_TXRQ                       0x00000001U        /*!<Transmit Mailbox Request */
#define  CAN_TI0R_RTR                        0x00000002U        /*!<Remote Transmission Request */
#define  CAN_TI0R_IDE                        0x00000004U        /*!<Identifier Extension */
#define  CAN_TI0R_EXID                       0x001FFFF8U        /*!<Extended Identifier */
#define  CAN_TI0R_STID                       0xFFE00000U        /*!<Standard Identifier or Extended Identifier */

/******************  Bit definition for CAN_TDT0R register  *******************/
#define  CAN_TDT0R_DLC                       0x0000000FU        /*!<Data Length Code */
#define  CAN_TDT0R_TGT                       0x00000100U        /*!<Transmit Global Time */
#define  CAN_TDT0R_TIME                      0xFFFF0000U        /*!<Message Time Stamp */

/******************  Bit definition for CAN_TDL0R register  *******************/
#define  CAN_TDL0R_DATA0                     0x000000FFU        /*!<Data byte 0 */
#define  CAN_TDL0R_DATA1                     0x0000FF00U        /*!<Data byte 1 */
#define  CAN_TDL0R_DATA2                     0x00FF0000U        /*!<Data byte 2 */
#define  CAN_TDL0R_DATA3                     0xFF000000U        /*!<Data byte 3 */

/******************  Bit definition for CAN_TDH0R register  *******************/
#define  CAN_TDH0R_DATA4                     0x000000FFU        /*!<Data byte 4 */
#define  CAN_TDH0R_DATA5                     0x0000FF00U        /*!<Data byte 5 */
#define  CAN_TDH0R_DATA6                     0x00FF0000U        /*!<Data byte 6 */
#define  CAN_TDH0R_DATA7                     0xFF000000U        /*!<Data byte 7 */

/*******************  Bit definition for CAN_TI1R register  *******************/
#define  CAN_TI1R_TXRQ                       0x00000001U        /*!<Transmit Mailbox Request */
#define  CAN_TI1R_RTR                        0x00000002U        /*!<Remote Transmission Request */
#define  CAN_TI1R_IDE                        0x00000004U        /*!<Identifier Extension */
#define  CAN_TI1R_EXID                       0x001FFFF8U        /*!<Extended Identifier */
#define  CAN_TI1R_STID                       0xFFE00000U        /*!<Standard Identifier or Extended Identifier */

/*******************  Bit definition for CAN_TDT1R register  ******************/
#define  CAN_TDT1R_DLC                       0x0000000FU        /*!<Data Length Code */
#define  CAN_TDT1R_TGT                       0x00000100U        /*!<Transmit Global Time */
#define  CAN_TDT1R_TIME                      0xFFFF0000U        /*!<Message Time Stamp */

/*******************  Bit definition for CAN_TDL1R register  ******************/
#define  CAN_TDL1R_DATA0                     0x000000FFU        /*!<Data byte 0 */
#define  CAN_TDL1R_DATA1                     0x0000FF00U        /*!<Data byte 1 */
#define  CAN_TDL1R_DATA2                     0x00FF0000U        /*!<Data byte 2 */
#define  CAN_TDL1R_DATA3                     0xFF000000U        /*!<Data byte 3 */

/*******************  Bit definition for CAN_TDH1R register  ******************/
#define  CAN_TDH1R_DATA4                     0x000000FFU        /*!<Data byte 4 */
#define  CAN_TDH1R_DATA5                     0x0000FF00U        /*!<Data byte 5 */
#define  CAN_TDH1R_DATA6                     0x00FF0000U        /*!<Data byte 6 */
#define  CAN_TDH1R_DATA7                     0xFF000000U        /*!<Data byte 7 */

/*******************  Bit definition for CAN_TI2R register  *******************/
#define  CAN_TI2R_TXRQ                       0x00000001U        /*!<Transmit Mailbox Request */
#define  CAN_TI2R_RTR                        0x00000002U        /*!<Remote Transmission Request */
#define  CAN_TI2R_IDE                        0x00000004U        /*!<Identifier Extension */
#define  CAN_TI2R_EXID                       0x001FFFF8U        /*!<Extended identifier */
#define  CAN_TI2R_STID                       0xFFE00000U        /*!<Standard Identifier or Extended Identifier */

/*******************  Bit definition for CAN_TDT2R register  ******************/  
#define  CAN_TDT2R_DLC                       0x0000000FU        /*!<Data Length Code */
#define  CAN_TDT2R_TGT                       0x00000100U        /*!<Transmit Global Time */
#define  CAN_TDT2R_TIME                      0xFFFF0000U        /*!<Message Time Stamp */

/*******************  Bit definition for CAN_TDL2R register  ******************/
#define  CAN_TDL2R_DATA0                     0x000000FFU        /*!<Data byte 0 */
#define  CAN_TDL2R_DATA1                     0x0000FF00U        /*!<Data byte 1 */
#define  CAN_TDL2R_DATA2                     0x00FF0000U        /*!<Data byte 2 */
#define  CAN_TDL2R_DATA3                     0xFF000000U        /*!<Data byte 3 */

/*******************  Bit definition for CAN_TDH2R register  ******************/
#define  CAN_TDH2R_DATA4                     0x000000FFU        /*!<Data byte 4 */
#define  CAN_TDH2R_DATA5                     0x0000FF00U        /*!<Data byte 5 */
#define  CAN_TDH2R_DATA6                     0x00FF0000U        /*!<Data byte 6 */
#define  CAN_TDH2R_DATA7                     0xFF000000U        /*!<Data byte 7 */

/*******************  Bit definition for CAN_RI0R register  *******************/
#define  CAN_RI0R_RTR                        0x00000002U        /*!<Remote Transmission Request */
#define  CAN_RI0R_IDE                        0x00000004U        /*!<Identifier Extension */
#define  CAN_RI0R_EXID                       0x001FFFF8U        /*!<Extended Identifier */
#define  CAN_RI0R_STID                       0xFFE00000U        /*!<Standard Identifier or Extended Identifier */

/*******************  Bit definition for CAN_RDT0R register  ******************/
#define  CAN_RDT0R_DLC                       0x0000000FU        /*!<Data Length Code */
#define  CAN_RDT0R_FMI                       0x0000FF00U        /*!<Filter Match Index */
#define  CAN_RDT0R_TIME                      0xFFFF0000U        /*!<Message Time Stamp */

/*******************  Bit definition for CAN_RDL0R register  ******************/
#define  CAN_RDL0R_DATA0                     0x000000FFU        /*!<Data byte 0 */
#define  CAN_RDL0R_DATA1                     0x0000FF00U        /*!<Data byte 1 */
#define  CAN_RDL0R_DATA2                     0x00FF0000U        /*!<Data byte 2 */
#define  CAN_RDL0R_DATA3                     0xFF000000U        /*!<Data byte 3 */

/*******************  Bit definition for CAN_RDH0R register  ******************/
#define  CAN_RDH0R_DATA4                     0x000000FFU        /*!<Data byte 4 */
#define  CAN_RDH0R_DATA5                     0x0000FF00U        /*!<Data byte 5 */
#define  CAN_RDH0R_DATA6                     0x00FF0000U        /*!<Data byte 6 */
#define  CAN_RDH0R_DATA7                     0xFF000000U        /*!<Data byte 7 */

/*******************  Bit definition for CAN_RI1R register  *******************/
#define  CAN_RI1R_RTR                        0x00000002U        /*!<Remote Transmission Request */
#define  CAN_RI1R_IDE                        0x00000004U        /*!<Identifier Extension */
#define  CAN_RI1R_EXID                       0x001FFFF8U        /*!<Extended identifier */
#define  CAN_RI1R_STID                       0xFFE00000U        /*!<Standard Identifier or Extended Identifier */

/*******************  Bit definition for CAN_RDT1R register  ******************/
#define  CAN_RDT1R_DLC                       0x0000000FU        /*!<Data Length Code */
#define  CAN_RDT1R_FMI                       0x0000FF00U        /*!<Filter Match Index */
#define  CAN_RDT1R_TIME                      0xFFFF0000U        /*!<Message Time Stamp */

/*******************  Bit definition for CAN_RDL1R register  ******************/
#define  CAN_RDL1R_DATA0                     0x000000FFU        /*!<Data byte 0 */
#define  CAN_RDL1R_DATA1                     0x0000FF00U        /*!<Data byte 1 */
#define  CAN_RDL1R_DATA2                     0x00FF0000U        /*!<Data byte 2 */
#define  CAN_RDL1R_DATA3                     0xFF000000U        /*!<Data byte 3 */

/*******************  Bit definition for CAN_RDH1R register  ******************/
#define  CAN_RDH1R_DATA4                     0x000000FFU        /*!<Data byte 4 */
#define  CAN_RDH1R_DATA5                     0x0000FF00U        /*!<Data byte 5 */
#define  CAN_RDH1R_DATA6                     0x00FF0000U        /*!<Data byte 6 */
#define  CAN_RDH1R_DATA7                     0xFF000000U        /*!<Data byte 7 */

/*!<CAN filter registers */
/*******************  Bit definition for CAN_FMR register  ********************/
#define  CAN_FMR_FINIT                       0x00000001U        /*!<Filter Init Mode */
#define  CAN_FMR_CAN2SB                      0x00003F00U        /*!<CAN2 start bank */

/*************  Bit definition for CAN_FM1R register  *******************/
#define  CAN_FM1R_FBM                        0x0FFFFFFFU        /*!<Filter Mode */
#define  CAN_FM1R_FBM0                       0x00000001U        /*!<Filter Init Mode bit 0 */
#define  CAN_FM1R_FBM1                       0x00000002U        /*!<Filter Init Mode bit 1 */
#define  CAN_FM1R_FBM2                       0x00000004U        /*!<Filter Init Mode bit 2 */
#define  CAN_FM1R_FBM3                       0x00000008U        /*!<Filter Init Mode bit 3 */
#define  CAN_FM1R_FBM4                       0x00000010U        /*!<Filter Init Mode bit 4 */
#define  CAN_FM1R_FBM5                       0x00000020U        /*!<Filter Init Mode bit 5 */
#define  CAN_FM1R_FBM6                       0x00000040U        /*!<Filter Init Mode bit 6 */
#define  CAN_FM1R_FBM7                       0x00000080U        /*!<Filter Init Mode bit 7 */
#define  CAN_FM1R_FBM8                       0x00000100U        /*!<Filter Init Mode bit 8 */
#define  CAN_FM1R_FBM9                       0x00000200U        /*!<Filter Init Mode bit 9 */
#define  CAN_FM1R_FBM10                      0x00000400U        /*!<Filter Init Mode bit 10 */
#define  CAN_FM1R_FBM11                      0x00000800U        /*!<Filter Init Mode bit 11 */
#define  CAN_FM1R_FBM12                      0x00001000U        /*!<Filter Init Mode bit 12 */
#define  CAN_FM1R_FBM13                      0x00002000U        /*!<Filter Init Mode bit 13 */
#define  CAN_FM1R_FBM14                      0x00004000U        /*!<Filter Init Mode bit 14 */
#define  CAN_FM1R_FBM15                      0x00008000U        /*!<Filter Init Mode bit 15 */
#define  CAN_FM1R_FBM16                      0x00010000U        /*!<Filter Init Mode bit 16 */
#define  CAN_FM1R_FBM17                      0x00020000U        /*!<Filter Init Mode bit 17 */
#define  CAN_FM1R_FBM18                      0x00040000U        /*!<Filter Init Mode bit 18 */
#define  CAN_FM1R_FBM19                      0x00080000U        /*!<Filter Init Mode bit 19 */
#define  CAN_FM1R_FBM20                      0x00100000U        /*!<Filter Init Mode bit 20 */
#define  CAN_FM1R_FBM21                      0x00200000U        /*!<Filter Init Mode bit 21 */
#define  CAN_FM1R_FBM22                      0x00400000U        /*!<Filter Init Mode bit 22 */
#define  CAN_FM1R_FBM23                      0x00800000U        /*!<Filter Init Mode bit 23 */
#define  CAN_FM1R_FBM24                      0x01000000U        /*!<Filter Init Mode bit 24 */
#define  CAN_FM1R_FBM25                      0x02000000U        /*!<Filter Init Mode bit 25 */
#define  CAN_FM1R_FBM26                      0x04000000U        /*!<Filter Init Mode bit 26 */
#define  CAN_FM1R_FBM27                      0x08000000U        /*!<Filter Init Mode bit 27 */

/*******************  Bit definition for CAN_FS1R register  *******************/
#define  CAN_FS1R_FSC                        0x0FFFFFFFU        /*!<Filter Scale Configuration */
#define  CAN_FS1R_FSC0                       0x00000001U        /*!<Filter Scale Configuration bit 0 */
#define  CAN_FS1R_FSC1                       0x00000002U        /*!<Filter Scale Configuration bit 1 */
#define  CAN_FS1R_FSC2                       0x00000004U        /*!<Filter Scale Configuration bit 2 */
#define  CAN_FS1R_FSC3                       0x00000008U        /*!<Filter Scale Configuration bit 3 */
#define  CAN_FS1R_FSC4                       0x00000010U        /*!<Filter Scale Configuration bit 4 */
#define  CAN_FS1R_FSC5                       0x00000020U        /*!<Filter Scale Configuration bit 5 */
#define  CAN_FS1R_FSC6                       0x00000040U        /*!<Filter Scale Configuration bit 6 */
#define  CAN_FS1R_FSC7                       0x00000080U        /*!<Filter Scale Configuration bit 7 */
#define  CAN_FS1R_FSC8                       0x00000100U        /*!<Filter Scale Configuration bit 8 */
#define  CAN_FS1R_FSC9                       0x00000200U        /*!<Filter Scale Configuration bit 9 */
#define  CAN_FS1R_FSC10                      0x00000400U        /*!<Filter Scale Configuration bit 10 */
#define  CAN_FS1R_FSC11                      0x00000800U        /*!<Filter Scale Configuration bit 11 */
#define  CAN_FS1R_FSC12                      0x00001000U        /*!<Filter Scale Configuration bit 12 */
#define  CAN_FS1R_FSC13                      0x00002000U        /*!<Filter Scale Configuration bit 13 */
#define  CAN_FS1R_FSC14                      0x00004000U        /*!<Filter Scale Configuration bit 14 */
#define  CAN_FS1R_FSC15                      0x00008000U        /*!<Filter Scale Configuration bit 15 */
#define  CAN_FS1R_FSC16                      0x00010000U        /*!<Filter Scale Configuration bit 16 */
#define  CAN_FS1R_FSC17                      0x00020000U        /*!<Filter Scale Configuration bit 17 */
#define  CAN_FS1R_FSC18                      0x00040000U        /*!<Filter Scale Configuration bit 18 */
#define  CAN_FS1R_FSC19                      0x00080000U        /*!<Filter Scale Configuration bit 19 */
#define  CAN_FS1R_FSC20                      0x00100000U        /*!<Filter Scale Configuration bit 20 */
#define  CAN_FS1R_FSC21                      0x00200000U        /*!<Filter Scale Configuration bit 21 */
#define  CAN_FS1R_FSC22                      0x00400000U        /*!<Filter Scale Configuration bit 22 */
#define  CAN_FS1R_FSC23                      0x00800000U        /*!<Filter Scale Configuration bit 23 */
#define  CAN_FS1R_FSC24                      0x01000000U        /*!<Filter Scale Configuration bit 24 */
#define  CAN_FS1R_FSC25                      0x02000000U        /*!<Filter Scale Configuration bit 25 */
#define  CAN_FS1R_FSC26                      0x04000000U        /*!<Filter Scale Configuration bit 26 */
#define  CAN_FS1R_FSC27                      0x08000000U        /*!<Filter Scale Configuration bit 27 */

/******************  Bit definition for CAN_FFA1R register  *******************/
#define  CAN_FFA1R_FFA                        0x0FFFFFFFU        /*!<Filter FIFO Assignment */
#define  CAN_FFA1R_FFA0                       0x00000001U        /*!<Filter FIFO Assignment bit 0 */
#define  CAN_FFA1R_FFA1                       0x00000002U        /*!<Filter FIFO Assignment bit 1 */
#define  CAN_FFA1R_FFA2                       0x00000004U        /*!<Filter FIFO Assignment bit 2 */
#define  CAN_FFA1R_FFA3                       0x00000008U        /*!<Filter FIFO Assignment bit 3 */
#define  CAN_FFA1R_FFA4                       0x00000010U        /*!<Filter FIFO Assignment bit 4 */
#define  CAN_FFA1R_FFA5                       0x00000020U        /*!<Filter FIFO Assignment bit 5 */
#define  CAN_FFA1R_FFA6                       0x00000040U        /*!<Filter FIFO Assignment bit 6 */
#define  CAN_FFA1R_FFA7                       0x00000080U        /*!<Filter FIFO Assignment bit 7 */
#define  CAN_FFA1R_FFA8                       0x00000100U        /*!<Filter FIFO Assignment bit 8 */
#define  CAN_FFA1R_FFA9                       0x00000200U        /*!<Filter FIFO Assignment bit 9 */
#define  CAN_FFA1R_FFA10                      0x00000400U        /*!<Filter FIFO Assignment bit 10 */
#define  CAN_FFA1R_FFA11                      0x00000800U        /*!<Filter FIFO Assignment bit 11 */
#define  CAN_FFA1R_FFA12                      0x00001000U        /*!<Filter FIFO Assignment bit 12 */
#define  CAN_FFA1R_FFA13                      0x00002000U        /*!<Filter FIFO Assignment bit 13 */
#define  CAN_FFA1R_FFA14                      0x00004000U        /*!<Filter FIFO Assignment bit 14 */
#define  CAN_FFA1R_FFA15                      0x00008000U        /*!<Filter FIFO Assignment bit 15 */
#define  CAN_FFA1R_FFA16                      0x00010000U        /*!<Filter FIFO Assignment bit 16 */
#define  CAN_FFA1R_FFA17                      0x00020000U        /*!<Filter FIFO Assignment bit 17 */
#define  CAN_FFA1R_FFA18                      0x00040000U        /*!<Filter FIFO Assignment bit 18 */
#define  CAN_FFA1R_FFA19                      0x00080000U        /*!<Filter FIFO Assignment bit 19 */
#define  CAN_FFA1R_FFA20                      0x00100000U        /*!<Filter FIFO Assignment bit 20 */
#define  CAN_FFA1R_FFA21                      0x00200000U        /*!<Filter FIFO Assignment bit 21 */
#define  CAN_FFA1R_FFA22                      0x00400000U        /*!<Filter FIFO Assignment bit 22 */
#define  CAN_FFA1R_FFA23                      0x00800000U        /*!<Filter FIFO Assignment bit 23 */
#define  CAN_FFA1R_FFA24                      0x01000000U        /*!<Filter FIFO Assignment bit 24 */
#define  CAN_FFA1R_FFA25                      0x02000000U        /*!<Filter FIFO Assignment bit 25 */
#define  CAN_FFA1R_FFA26                      0x04000000U        /*!<Filter FIFO Assignment bit 26 */
#define  CAN_FFA1R_FFA27                      0x08000000U        /*!<Filter FIFO Assignment bit 27 */

/*******************  Bit definition for CAN_FA1R register  *******************/
#define  CAN_FA1R_FACT                        0x0FFFFFFFU        /*!<Filter Active */
#define  CAN_FA1R_FACT0                       0x00000001U        /*!<Filter Active bit 0 */
#define  CAN_FA1R_FACT1                       0x00000002U        /*!<Filter Active bit 1 */
#define  CAN_FA1R_FACT2                       0x00000004U        /*!<Filter Active bit 2 */
#define  CAN_FA1R_FACT3                       0x00000008U        /*!<Filter Active bit 3 */
#define  CAN_FA1R_FACT4                       0x00000010U        /*!<Filter Active bit 4 */
#define  CAN_FA1R_FACT5                       0x00000020U        /*!<Filter Active bit 5 */
#define  CAN_FA1R_FACT6                       0x00000040U        /*!<Filter Active bit 6 */
#define  CAN_FA1R_FACT7                       0x00000080U        /*!<Filter Active bit 7 */
#define  CAN_FA1R_FACT8                       0x00000100U        /*!<Filter Active bit 8 */
#define  CAN_FA1R_FACT9                       0x00000200U        /*!<Filter Active bit 9 */
#define  CAN_FA1R_FACT10                      0x00000400U        /*!<Filter Active bit 10 */
#define  CAN_FA1R_FACT11                      0x00000800U        /*!<Filter Active bit 11 */
#define  CAN_FA1R_FACT12                      0x00001000U        /*!<Filter Active bit 12 */
#define  CAN_FA1R_FACT13                      0x00002000U        /*!<Filter Active bit 13 */
#define  CAN_FA1R_FACT14                      0x00004000U        /*!<Filter Active bit 14 */
#define  CAN_FA1R_FACT15                      0x00008000U        /*!<Filter Active bit 15 */
#define  CAN_FA1R_FACT16                      0x00010000U        /*!<Filter Active bit 16 */
#define  CAN_FA1R_FACT17                      0x00020000U        /*!<Filter Active bit 17 */
#define  CAN_FA1R_FACT18                      0x00040000U        /*!<Filter Active bit 18 */
#define  CAN_FA1R_FACT19                      0x00080000U        /*!<Filter Active bit 19 */
#define  CAN_FA1R_FACT20                      0x00100000U        /*!<Filter Active bit 20 */
#define  CAN_FA1R_FACT21                      0x00200000U        /*!<Filter Active bit 21 */
#define  CAN_FA1R_FACT22                      0x00400000U        /*!<Filter Active bit 22 */
#define  CAN_FA1R_FACT23                      0x00800000U        /*!<Filter Active bit 23 */
#define  CAN_FA1R_FACT24                      0x01000000U        /*!<Filter Active bit 24 */
#define  CAN_FA1R_FACT25                      0x02000000U        /*!<Filter Active bit 25 */
#define  CAN_FA1R_FACT26                      0x04000000U        /*!<Filter Active bit 26 */
#define  CAN_FA1R_FACT27                      0x08000000U        /*!<Filter Active bit 27 */

/*******************  Bit definition for CAN_F0R1 register  *******************/
#define  CAN_F0R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F0R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F0R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F0R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F0R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F0R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F0R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F0R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F0R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F0R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F0R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F0R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F0R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F0R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F0R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F0R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F0R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F0R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F0R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F0R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F0R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F0R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F0R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F0R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F0R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F0R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F0R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F0R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F0R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F0R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F0R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F0R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F1R1 register  *******************/
#define  CAN_F1R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F1R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F1R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F1R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F1R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F1R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F1R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F1R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F1R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F1R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F1R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F1R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F1R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F1R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F1R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F1R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F1R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F1R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F1R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F1R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F1R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F1R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F1R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F1R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F1R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F1R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F1R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F1R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F1R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F1R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F1R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F1R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F2R1 register  *******************/
#define  CAN_F2R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F2R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F2R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F2R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F2R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F2R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F2R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F2R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F2R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F2R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F2R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F2R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F2R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F2R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F2R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F2R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F2R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F2R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F2R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F2R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F2R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F2R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F2R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F2R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F2R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F2R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F2R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F2R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F2R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F2R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F2R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F2R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F3R1 register  *******************/
#define  CAN_F3R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F3R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F3R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F3R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F3R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F3R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F3R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F3R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F3R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F3R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F3R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F3R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F3R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F3R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F3R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F3R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F3R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F3R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F3R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F3R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F3R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F3R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F3R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F3R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F3R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F3R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F3R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F3R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F3R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F3R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F3R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F3R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F4R1 register  *******************/
#define  CAN_F4R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F4R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F4R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F4R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F4R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F4R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F4R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F4R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F4R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F4R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F4R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F4R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F4R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F4R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F4R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F4R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F4R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F4R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F4R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F4R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F4R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F4R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F4R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F4R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F4R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F4R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F4R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F4R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F4R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F4R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F4R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F4R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F5R1 register  *******************/
#define  CAN_F5R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F5R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F5R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F5R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F5R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F5R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F5R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F5R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F5R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F5R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F5R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F5R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F5R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F5R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F5R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F5R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F5R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F5R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F5R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F5R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F5R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F5R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F5R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F5R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F5R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F5R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F5R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F5R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F5R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F5R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F5R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F5R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F6R1 register  *******************/
#define  CAN_F6R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F6R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F6R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F6R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F6R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F6R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F6R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F6R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F6R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F6R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F6R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F6R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F6R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F6R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F6R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F6R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F6R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F6R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F6R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F6R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F6R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F6R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F6R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F6R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F6R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F6R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F6R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F6R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F6R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F6R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F6R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F6R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F7R1 register  *******************/
#define  CAN_F7R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F7R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F7R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F7R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F7R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F7R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F7R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F7R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F7R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F7R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F7R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F7R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F7R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F7R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F7R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F7R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F7R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F7R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F7R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F7R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F7R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F7R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F7R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F7R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F7R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F7R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F7R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F7R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F7R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F7R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F7R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F7R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F8R1 register  *******************/
#define  CAN_F8R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F8R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F8R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F8R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F8R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F8R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F8R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F8R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F8R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F8R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F8R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F8R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F8R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F8R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F8R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F8R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F8R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F8R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F8R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F8R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F8R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F8R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F8R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F8R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F8R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F8R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F8R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F8R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F8R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F8R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F8R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F8R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F9R1 register  *******************/
#define  CAN_F9R1_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F9R1_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F9R1_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F9R1_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F9R1_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F9R1_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F9R1_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F9R1_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F9R1_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F9R1_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F9R1_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F9R1_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F9R1_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F9R1_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F9R1_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F9R1_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F9R1_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F9R1_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F9R1_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F9R1_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F9R1_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F9R1_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F9R1_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F9R1_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F9R1_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F9R1_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F9R1_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F9R1_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F9R1_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F9R1_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F9R1_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F9R1_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F10R1 register  ******************/
#define  CAN_F10R1_FB0                       0x00000001U        /*!<Filter bit 0 */
#define  CAN_F10R1_FB1                       0x00000002U        /*!<Filter bit 1 */
#define  CAN_F10R1_FB2                       0x00000004U        /*!<Filter bit 2 */
#define  CAN_F10R1_FB3                       0x00000008U        /*!<Filter bit 3 */
#define  CAN_F10R1_FB4                       0x00000010U        /*!<Filter bit 4 */
#define  CAN_F10R1_FB5                       0x00000020U        /*!<Filter bit 5 */
#define  CAN_F10R1_FB6                       0x00000040U        /*!<Filter bit 6 */
#define  CAN_F10R1_FB7                       0x00000080U        /*!<Filter bit 7 */
#define  CAN_F10R1_FB8                       0x00000100U        /*!<Filter bit 8 */
#define  CAN_F10R1_FB9                       0x00000200U        /*!<Filter bit 9 */
#define  CAN_F10R1_FB10                      0x00000400U        /*!<Filter bit 10 */
#define  CAN_F10R1_FB11                      0x00000800U        /*!<Filter bit 11 */
#define  CAN_F10R1_FB12                      0x00001000U        /*!<Filter bit 12 */
#define  CAN_F10R1_FB13                      0x00002000U        /*!<Filter bit 13 */
#define  CAN_F10R1_FB14                      0x00004000U        /*!<Filter bit 14 */
#define  CAN_F10R1_FB15                      0x00008000U        /*!<Filter bit 15 */
#define  CAN_F10R1_FB16                      0x00010000U        /*!<Filter bit 16 */
#define  CAN_F10R1_FB17                      0x00020000U        /*!<Filter bit 17 */
#define  CAN_F10R1_FB18                      0x00040000U        /*!<Filter bit 18 */
#define  CAN_F10R1_FB19                      0x00080000U        /*!<Filter bit 19 */
#define  CAN_F10R1_FB20                      0x00100000U        /*!<Filter bit 20 */
#define  CAN_F10R1_FB21                      0x00200000U        /*!<Filter bit 21 */
#define  CAN_F10R1_FB22                      0x00400000U        /*!<Filter bit 22 */
#define  CAN_F10R1_FB23                      0x00800000U        /*!<Filter bit 23 */
#define  CAN_F10R1_FB24                      0x01000000U        /*!<Filter bit 24 */
#define  CAN_F10R1_FB25                      0x02000000U        /*!<Filter bit 25 */
#define  CAN_F10R1_FB26                      0x04000000U        /*!<Filter bit 26 */
#define  CAN_F10R1_FB27                      0x08000000U        /*!<Filter bit 27 */
#define  CAN_F10R1_FB28                      0x10000000U        /*!<Filter bit 28 */
#define  CAN_F10R1_FB29                      0x20000000U        /*!<Filter bit 29 */
#define  CAN_F10R1_FB30                      0x40000000U        /*!<Filter bit 30 */
#define  CAN_F10R1_FB31                      0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F11R1 register  ******************/
#define  CAN_F11R1_FB0                       0x00000001U        /*!<Filter bit 0 */
#define  CAN_F11R1_FB1                       0x00000002U        /*!<Filter bit 1 */
#define  CAN_F11R1_FB2                       0x00000004U        /*!<Filter bit 2 */
#define  CAN_F11R1_FB3                       0x00000008U        /*!<Filter bit 3 */
#define  CAN_F11R1_FB4                       0x00000010U        /*!<Filter bit 4 */
#define  CAN_F11R1_FB5                       0x00000020U        /*!<Filter bit 5 */
#define  CAN_F11R1_FB6                       0x00000040U        /*!<Filter bit 6 */
#define  CAN_F11R1_FB7                       0x00000080U        /*!<Filter bit 7 */
#define  CAN_F11R1_FB8                       0x00000100U        /*!<Filter bit 8 */
#define  CAN_F11R1_FB9                       0x00000200U        /*!<Filter bit 9 */
#define  CAN_F11R1_FB10                      0x00000400U        /*!<Filter bit 10 */
#define  CAN_F11R1_FB11                      0x00000800U        /*!<Filter bit 11 */
#define  CAN_F11R1_FB12                      0x00001000U        /*!<Filter bit 12 */
#define  CAN_F11R1_FB13                      0x00002000U        /*!<Filter bit 13 */
#define  CAN_F11R1_FB14                      0x00004000U        /*!<Filter bit 14 */
#define  CAN_F11R1_FB15                      0x00008000U        /*!<Filter bit 15 */
#define  CAN_F11R1_FB16                      0x00010000U        /*!<Filter bit 16 */
#define  CAN_F11R1_FB17                      0x00020000U        /*!<Filter bit 17 */
#define  CAN_F11R1_FB18                      0x00040000U        /*!<Filter bit 18 */
#define  CAN_F11R1_FB19                      0x00080000U        /*!<Filter bit 19 */
#define  CAN_F11R1_FB20                      0x00100000U        /*!<Filter bit 20 */
#define  CAN_F11R1_FB21                      0x00200000U        /*!<Filter bit 21 */
#define  CAN_F11R1_FB22                      0x00400000U        /*!<Filter bit 22 */
#define  CAN_F11R1_FB23                      0x00800000U        /*!<Filter bit 23 */
#define  CAN_F11R1_FB24                      0x01000000U        /*!<Filter bit 24 */
#define  CAN_F11R1_FB25                      0x02000000U        /*!<Filter bit 25 */
#define  CAN_F11R1_FB26                      0x04000000U        /*!<Filter bit 26 */
#define  CAN_F11R1_FB27                      0x08000000U        /*!<Filter bit 27 */
#define  CAN_F11R1_FB28                      0x10000000U        /*!<Filter bit 28 */
#define  CAN_F11R1_FB29                      0x20000000U        /*!<Filter bit 29 */
#define  CAN_F11R1_FB30                      0x40000000U        /*!<Filter bit 30 */
#define  CAN_F11R1_FB31                      0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F12R1 register  ******************/
#define  CAN_F12R1_FB0                       0x00000001U        /*!<Filter bit 0 */
#define  CAN_F12R1_FB1                       0x00000002U        /*!<Filter bit 1 */
#define  CAN_F12R1_FB2                       0x00000004U        /*!<Filter bit 2 */
#define  CAN_F12R1_FB3                       0x00000008U        /*!<Filter bit 3 */
#define  CAN_F12R1_FB4                       0x00000010U        /*!<Filter bit 4 */
#define  CAN_F12R1_FB5                       0x00000020U        /*!<Filter bit 5 */
#define  CAN_F12R1_FB6                       0x00000040U        /*!<Filter bit 6 */
#define  CAN_F12R1_FB7                       0x00000080U        /*!<Filter bit 7 */
#define  CAN_F12R1_FB8                       0x00000100U        /*!<Filter bit 8 */
#define  CAN_F12R1_FB9                       0x00000200U        /*!<Filter bit 9 */
#define  CAN_F12R1_FB10                      0x00000400U        /*!<Filter bit 10 */
#define  CAN_F12R1_FB11                      0x00000800U        /*!<Filter bit 11 */
#define  CAN_F12R1_FB12                      0x00001000U        /*!<Filter bit 12 */
#define  CAN_F12R1_FB13                      0x00002000U        /*!<Filter bit 13 */
#define  CAN_F12R1_FB14                      0x00004000U        /*!<Filter bit 14 */
#define  CAN_F12R1_FB15                      0x00008000U        /*!<Filter bit 15 */
#define  CAN_F12R1_FB16                      0x00010000U        /*!<Filter bit 16 */
#define  CAN_F12R1_FB17                      0x00020000U        /*!<Filter bit 17 */
#define  CAN_F12R1_FB18                      0x00040000U        /*!<Filter bit 18 */
#define  CAN_F12R1_FB19                      0x00080000U        /*!<Filter bit 19 */
#define  CAN_F12R1_FB20                      0x00100000U        /*!<Filter bit 20 */
#define  CAN_F12R1_FB21                      0x00200000U        /*!<Filter bit 21 */
#define  CAN_F12R1_FB22                      0x00400000U        /*!<Filter bit 22 */
#define  CAN_F12R1_FB23                      0x00800000U        /*!<Filter bit 23 */
#define  CAN_F12R1_FB24                      0x01000000U        /*!<Filter bit 24 */
#define  CAN_F12R1_FB25                      0x02000000U        /*!<Filter bit 25 */
#define  CAN_F12R1_FB26                      0x04000000U        /*!<Filter bit 26 */
#define  CAN_F12R1_FB27                      0x08000000U        /*!<Filter bit 27 */
#define  CAN_F12R1_FB28                      0x10000000U        /*!<Filter bit 28 */
#define  CAN_F12R1_FB29                      0x20000000U        /*!<Filter bit 29 */
#define  CAN_F12R1_FB30                      0x40000000U        /*!<Filter bit 30 */
#define  CAN_F12R1_FB31                      0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F13R1 register  ******************/
#define  CAN_F13R1_FB0                       0x00000001U        /*!<Filter bit 0 */
#define  CAN_F13R1_FB1                       0x00000002U        /*!<Filter bit 1 */
#define  CAN_F13R1_FB2                       0x00000004U        /*!<Filter bit 2 */
#define  CAN_F13R1_FB3                       0x00000008U        /*!<Filter bit 3 */
#define  CAN_F13R1_FB4                       0x00000010U        /*!<Filter bit 4 */
#define  CAN_F13R1_FB5                       0x00000020U        /*!<Filter bit 5 */
#define  CAN_F13R1_FB6                       0x00000040U        /*!<Filter bit 6 */
#define  CAN_F13R1_FB7                       0x00000080U        /*!<Filter bit 7 */
#define  CAN_F13R1_FB8                       0x00000100U        /*!<Filter bit 8 */
#define  CAN_F13R1_FB9                       0x00000200U        /*!<Filter bit 9 */
#define  CAN_F13R1_FB10                      0x00000400U        /*!<Filter bit 10 */
#define  CAN_F13R1_FB11                      0x00000800U        /*!<Filter bit 11 */
#define  CAN_F13R1_FB12                      0x00001000U        /*!<Filter bit 12 */
#define  CAN_F13R1_FB13                      0x00002000U        /*!<Filter bit 13 */
#define  CAN_F13R1_FB14                      0x00004000U        /*!<Filter bit 14 */
#define  CAN_F13R1_FB15                      0x00008000U        /*!<Filter bit 15 */
#define  CAN_F13R1_FB16                      0x00010000U        /*!<Filter bit 16 */
#define  CAN_F13R1_FB17                      0x00020000U        /*!<Filter bit 17 */
#define  CAN_F13R1_FB18                      0x00040000U        /*!<Filter bit 18 */
#define  CAN_F13R1_FB19                      0x00080000U        /*!<Filter bit 19 */
#define  CAN_F13R1_FB20                      0x00100000U        /*!<Filter bit 20 */
#define  CAN_F13R1_FB21                      0x00200000U        /*!<Filter bit 21 */
#define  CAN_F13R1_FB22                      0x00400000U        /*!<Filter bit 22 */
#define  CAN_F13R1_FB23                      0x00800000U        /*!<Filter bit 23 */
#define  CAN_F13R1_FB24                      0x01000000U        /*!<Filter bit 24 */
#define  CAN_F13R1_FB25                      0x02000000U        /*!<Filter bit 25 */
#define  CAN_F13R1_FB26                      0x04000000U        /*!<Filter bit 26 */
#define  CAN_F13R1_FB27                      0x08000000U        /*!<Filter bit 27 */
#define  CAN_F13R1_FB28                      0x10000000U        /*!<Filter bit 28 */
#define  CAN_F13R1_FB29                      0x20000000U        /*!<Filter bit 29 */
#define  CAN_F13R1_FB30                      0x40000000U        /*!<Filter bit 30 */
#define  CAN_F13R1_FB31                      0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F0R2 register  *******************/
#define  CAN_F0R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F0R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F0R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F0R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F0R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F0R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F0R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F0R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F0R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F0R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F0R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F0R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F0R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F0R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F0R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F0R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F0R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F0R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F0R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F0R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F0R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F0R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F0R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F0R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F0R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F0R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F0R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F0R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F0R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F0R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F0R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F0R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F1R2 register  *******************/
#define  CAN_F1R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F1R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F1R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F1R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F1R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F1R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F1R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F1R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F1R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F1R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F1R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F1R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F1R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F1R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F1R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F1R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F1R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F1R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F1R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F1R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F1R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F1R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F1R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F1R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F1R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F1R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F1R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F1R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F1R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F1R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F1R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F1R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F2R2 register  *******************/
#define  CAN_F2R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F2R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F2R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F2R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F2R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F2R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F2R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F2R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F2R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F2R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F2R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F2R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F2R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F2R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F2R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F2R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F2R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F2R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F2R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F2R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F2R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F2R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F2R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F2R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F2R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F2R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F2R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F2R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F2R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F2R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F2R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F2R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F3R2 register  *******************/
#define  CAN_F3R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F3R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F3R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F3R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F3R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F3R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F3R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F3R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F3R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F3R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F3R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F3R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F3R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F3R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F3R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F3R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F3R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F3R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F3R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F3R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F3R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F3R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F3R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F3R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F3R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F3R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F3R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F3R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F3R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F3R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F3R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F3R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F4R2 register  *******************/
#define  CAN_F4R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F4R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F4R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F4R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F4R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F4R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F4R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F4R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F4R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F4R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F4R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F4R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F4R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F4R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F4R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F4R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F4R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F4R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F4R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F4R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F4R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F4R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F4R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F4R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F4R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F4R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F4R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F4R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F4R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F4R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F4R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F4R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F5R2 register  *******************/
#define  CAN_F5R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F5R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F5R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F5R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F5R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F5R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F5R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F5R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F5R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F5R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F5R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F5R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F5R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F5R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F5R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F5R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F5R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F5R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F5R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F5R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F5R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F5R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F5R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F5R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F5R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F5R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F5R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F5R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F5R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F5R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F5R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F5R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F6R2 register  *******************/
#define  CAN_F6R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F6R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F6R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F6R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F6R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F6R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F6R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F6R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F6R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F6R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F6R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F6R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F6R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F6R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F6R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F6R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F6R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F6R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F6R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F6R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F6R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F6R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F6R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F6R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F6R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F6R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F6R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F6R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F6R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F6R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F6R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F6R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F7R2 register  *******************/
#define  CAN_F7R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F7R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F7R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F7R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F7R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F7R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F7R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F7R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F7R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F7R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F7R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F7R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F7R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F7R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F7R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F7R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F7R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F7R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F7R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F7R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F7R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F7R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F7R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F7R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F7R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F7R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F7R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F7R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F7R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F7R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F7R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F7R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F8R2 register  *******************/
#define  CAN_F8R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F8R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F8R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F8R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F8R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F8R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F8R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F8R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F8R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F8R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F8R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F8R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F8R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F8R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F8R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F8R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F8R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F8R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F8R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F8R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F8R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F8R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F8R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F8R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F8R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F8R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F8R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F8R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F8R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F8R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F8R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F8R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F9R2 register  *******************/
#define  CAN_F9R2_FB0                        0x00000001U        /*!<Filter bit 0 */
#define  CAN_F9R2_FB1                        0x00000002U        /*!<Filter bit 1 */
#define  CAN_F9R2_FB2                        0x00000004U        /*!<Filter bit 2 */
#define  CAN_F9R2_FB3                        0x00000008U        /*!<Filter bit 3 */
#define  CAN_F9R2_FB4                        0x00000010U        /*!<Filter bit 4 */
#define  CAN_F9R2_FB5                        0x00000020U        /*!<Filter bit 5 */
#define  CAN_F9R2_FB6                        0x00000040U        /*!<Filter bit 6 */
#define  CAN_F9R2_FB7                        0x00000080U        /*!<Filter bit 7 */
#define  CAN_F9R2_FB8                        0x00000100U        /*!<Filter bit 8 */
#define  CAN_F9R2_FB9                        0x00000200U        /*!<Filter bit 9 */
#define  CAN_F9R2_FB10                       0x00000400U        /*!<Filter bit 10 */
#define  CAN_F9R2_FB11                       0x00000800U        /*!<Filter bit 11 */
#define  CAN_F9R2_FB12                       0x00001000U        /*!<Filter bit 12 */
#define  CAN_F9R2_FB13                       0x00002000U        /*!<Filter bit 13 */
#define  CAN_F9R2_FB14                       0x00004000U        /*!<Filter bit 14 */
#define  CAN_F9R2_FB15                       0x00008000U        /*!<Filter bit 15 */
#define  CAN_F9R2_FB16                       0x00010000U        /*!<Filter bit 16 */
#define  CAN_F9R2_FB17                       0x00020000U        /*!<Filter bit 17 */
#define  CAN_F9R2_FB18                       0x00040000U        /*!<Filter bit 18 */
#define  CAN_F9R2_FB19                       0x00080000U        /*!<Filter bit 19 */
#define  CAN_F9R2_FB20                       0x00100000U        /*!<Filter bit 20 */
#define  CAN_F9R2_FB21                       0x00200000U        /*!<Filter bit 21 */
#define  CAN_F9R2_FB22                       0x00400000U        /*!<Filter bit 22 */
#define  CAN_F9R2_FB23                       0x00800000U        /*!<Filter bit 23 */
#define  CAN_F9R2_FB24                       0x01000000U        /*!<Filter bit 24 */
#define  CAN_F9R2_FB25                       0x02000000U        /*!<Filter bit 25 */
#define  CAN_F9R2_FB26                       0x04000000U        /*!<Filter bit 26 */
#define  CAN_F9R2_FB27                       0x08000000U        /*!<Filter bit 27 */
#define  CAN_F9R2_FB28                       0x10000000U        /*!<Filter bit 28 */
#define  CAN_F9R2_FB29                       0x20000000U        /*!<Filter bit 29 */
#define  CAN_F9R2_FB30                       0x40000000U        /*!<Filter bit 30 */
#define  CAN_F9R2_FB31                       0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F10R2 register  ******************/
#define  CAN_F10R2_FB0                       0x00000001U        /*!<Filter bit 0 */
#define  CAN_F10R2_FB1                       0x00000002U        /*!<Filter bit 1 */
#define  CAN_F10R2_FB2                       0x00000004U        /*!<Filter bit 2 */
#define  CAN_F10R2_FB3                       0x00000008U        /*!<Filter bit 3 */
#define  CAN_F10R2_FB4                       0x00000010U        /*!<Filter bit 4 */
#define  CAN_F10R2_FB5                       0x00000020U        /*!<Filter bit 5 */
#define  CAN_F10R2_FB6                       0x00000040U        /*!<Filter bit 6 */
#define  CAN_F10R2_FB7                       0x00000080U        /*!<Filter bit 7 */
#define  CAN_F10R2_FB8                       0x00000100U        /*!<Filter bit 8 */
#define  CAN_F10R2_FB9                       0x00000200U        /*!<Filter bit 9 */
#define  CAN_F10R2_FB10                      0x00000400U        /*!<Filter bit 10 */
#define  CAN_F10R2_FB11                      0x00000800U        /*!<Filter bit 11 */
#define  CAN_F10R2_FB12                      0x00001000U        /*!<Filter bit 12 */
#define  CAN_F10R2_FB13                      0x00002000U        /*!<Filter bit 13 */
#define  CAN_F10R2_FB14                      0x00004000U        /*!<Filter bit 14 */
#define  CAN_F10R2_FB15                      0x00008000U        /*!<Filter bit 15 */
#define  CAN_F10R2_FB16                      0x00010000U        /*!<Filter bit 16 */
#define  CAN_F10R2_FB17                      0x00020000U        /*!<Filter bit 17 */
#define  CAN_F10R2_FB18                      0x00040000U        /*!<Filter bit 18 */
#define  CAN_F10R2_FB19                      0x00080000U        /*!<Filter bit 19 */
#define  CAN_F10R2_FB20                      0x00100000U        /*!<Filter bit 20 */
#define  CAN_F10R2_FB21                      0x00200000U        /*!<Filter bit 21 */
#define  CAN_F10R2_FB22                      0x00400000U        /*!<Filter bit 22 */
#define  CAN_F10R2_FB23                      0x00800000U        /*!<Filter bit 23 */
#define  CAN_F10R2_FB24                      0x01000000U        /*!<Filter bit 24 */
#define  CAN_F10R2_FB25                      0x02000000U        /*!<Filter bit 25 */
#define  CAN_F10R2_FB26                      0x04000000U        /*!<Filter bit 26 */
#define  CAN_F10R2_FB27                      0x08000000U        /*!<Filter bit 27 */
#define  CAN_F10R2_FB28                      0x10000000U        /*!<Filter bit 28 */
#define  CAN_F10R2_FB29                      0x20000000U        /*!<Filter bit 29 */
#define  CAN_F10R2_FB30                      0x40000000U        /*!<Filter bit 30 */
#define  CAN_F10R2_FB31                      0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F11R2 register  ******************/
#define  CAN_F11R2_FB0                       0x00000001U        /*!<Filter bit 0 */
#define  CAN_F11R2_FB1                       0x00000002U        /*!<Filter bit 1 */
#define  CAN_F11R2_FB2                       0x00000004U        /*!<Filter bit 2 */
#define  CAN_F11R2_FB3                       0x00000008U        /*!<Filter bit 3 */
#define  CAN_F11R2_FB4                       0x00000010U        /*!<Filter bit 4 */
#define  CAN_F11R2_FB5                       0x00000020U        /*!<Filter bit 5 */
#define  CAN_F11R2_FB6                       0x00000040U        /*!<Filter bit 6 */
#define  CAN_F11R2_FB7                       0x00000080U        /*!<Filter bit 7 */
#define  CAN_F11R2_FB8                       0x00000100U        /*!<Filter bit 8 */
#define  CAN_F11R2_FB9                       0x00000200U        /*!<Filter bit 9 */
#define  CAN_F11R2_FB10                      0x00000400U        /*!<Filter bit 10 */
#define  CAN_F11R2_FB11                      0x00000800U        /*!<Filter bit 11 */
#define  CAN_F11R2_FB12                      0x00001000U        /*!<Filter bit 12 */
#define  CAN_F11R2_FB13                      0x00002000U        /*!<Filter bit 13 */
#define  CAN_F11R2_FB14                      0x00004000U        /*!<Filter bit 14 */
#define  CAN_F11R2_FB15                      0x00008000U        /*!<Filter bit 15 */
#define  CAN_F11R2_FB16                      0x00010000U        /*!<Filter bit 16 */
#define  CAN_F11R2_FB17                      0x00020000U        /*!<Filter bit 17 */
#define  CAN_F11R2_FB18                      0x00040000U        /*!<Filter bit 18 */
#define  CAN_F11R2_FB19                      0x00080000U        /*!<Filter bit 19 */
#define  CAN_F11R2_FB20                      0x00100000U        /*!<Filter bit 20 */
#define  CAN_F11R2_FB21                      0x00200000U        /*!<Filter bit 21 */
#define  CAN_F11R2_FB22                      0x00400000U        /*!<Filter bit 22 */
#define  CAN_F11R2_FB23                      0x00800000U        /*!<Filter bit 23 */
#define  CAN_F11R2_FB24                      0x01000000U        /*!<Filter bit 24 */
#define  CAN_F11R2_FB25                      0x02000000U        /*!<Filter bit 25 */
#define  CAN_F11R2_FB26                      0x04000000U        /*!<Filter bit 26 */
#define  CAN_F11R2_FB27                      0x08000000U        /*!<Filter bit 27 */
#define  CAN_F11R2_FB28                      0x10000000U        /*!<Filter bit 28 */
#define  CAN_F11R2_FB29                      0x20000000U        /*!<Filter bit 29 */
#define  CAN_F11R2_FB30                      0x40000000U        /*!<Filter bit 30 */
#define  CAN_F11R2_FB31                      0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F12R2 register  ******************/
#define  CAN_F12R2_FB0                       0x00000001U        /*!<Filter bit 0 */
#define  CAN_F12R2_FB1                       0x00000002U        /*!<Filter bit 1 */
#define  CAN_F12R2_FB2                       0x00000004U        /*!<Filter bit 2 */
#define  CAN_F12R2_FB3                       0x00000008U        /*!<Filter bit 3 */
#define  CAN_F12R2_FB4                       0x00000010U        /*!<Filter bit 4 */
#define  CAN_F12R2_FB5                       0x00000020U        /*!<Filter bit 5 */
#define  CAN_F12R2_FB6                       0x00000040U        /*!<Filter bit 6 */
#define  CAN_F12R2_FB7                       0x00000080U        /*!<Filter bit 7 */
#define  CAN_F12R2_FB8                       0x00000100U        /*!<Filter bit 8 */
#define  CAN_F12R2_FB9                       0x00000200U        /*!<Filter bit 9 */
#define  CAN_F12R2_FB10                      0x00000400U        /*!<Filter bit 10 */
#define  CAN_F12R2_FB11                      0x00000800U        /*!<Filter bit 11 */
#define  CAN_F12R2_FB12                      0x00001000U        /*!<Filter bit 12 */
#define  CAN_F12R2_FB13                      0x00002000U        /*!<Filter bit 13 */
#define  CAN_F12R2_FB14                      0x00004000U        /*!<Filter bit 14 */
#define  CAN_F12R2_FB15                      0x00008000U        /*!<Filter bit 15 */
#define  CAN_F12R2_FB16                      0x00010000U        /*!<Filter bit 16 */
#define  CAN_F12R2_FB17                      0x00020000U        /*!<Filter bit 17 */
#define  CAN_F12R2_FB18                      0x00040000U        /*!<Filter bit 18 */
#define  CAN_F12R2_FB19                      0x00080000U        /*!<Filter bit 19 */
#define  CAN_F12R2_FB20                      0x00100000U        /*!<Filter bit 20 */
#define  CAN_F12R2_FB21                      0x00200000U        /*!<Filter bit 21 */
#define  CAN_F12R2_FB22                      0x00400000U        /*!<Filter bit 22 */
#define  CAN_F12R2_FB23                      0x00800000U        /*!<Filter bit 23 */
#define  CAN_F12R2_FB24                      0x01000000U        /*!<Filter bit 24 */
#define  CAN_F12R2_FB25                      0x02000000U        /*!<Filter bit 25 */
#define  CAN_F12R2_FB26                      0x04000000U        /*!<Filter bit 26 */
#define  CAN_F12R2_FB27                      0x08000000U        /*!<Filter bit 27 */
#define  CAN_F12R2_FB28                      0x10000000U        /*!<Filter bit 28 */
#define  CAN_F12R2_FB29                      0x20000000U        /*!<Filter bit 29 */
#define  CAN_F12R2_FB30                      0x40000000U        /*!<Filter bit 30 */
#define  CAN_F12R2_FB31                      0x80000000U        /*!<Filter bit 31 */

/*******************  Bit definition for CAN_F13R2 register  ******************/
#define  CAN_F13R2_FB0                       0x00000001U        /*!<Filter bit 0 */
#define  CAN_F13R2_FB1                       0x00000002U        /*!<Filter bit 1 */
#define  CAN_F13R2_FB2                       0x00000004U        /*!<Filter bit 2 */
#define  CAN_F13R2_FB3                       0x00000008U        /*!<Filter bit 3 */
#define  CAN_F13R2_FB4                       0x00000010U        /*!<Filter bit 4 */
#define  CAN_F13R2_FB5                       0x00000020U        /*!<Filter bit 5 */
#define  CAN_F13R2_FB6                       0x00000040U        /*!<Filter bit 6 */
#define  CAN_F13R2_FB7                       0x00000080U        /*!<Filter bit 7 */
#define  CAN_F13R2_FB8                       0x00000100U        /*!<Filter bit 8 */
#define  CAN_F13R2_FB9                       0x00000200U        /*!<Filter bit 9 */
#define  CAN_F13R2_FB10                      0x00000400U        /*!<Filter bit 10 */
#define  CAN_F13R2_FB11                      0x00000800U        /*!<Filter bit 11 */
#define  CAN_F13R2_FB12                      0x00001000U        /*!<Filter bit 12 */
#define  CAN_F13R2_FB13                      0x00002000U        /*!<Filter bit 13 */
#define  CAN_F13R2_FB14                      0x00004000U        /*!<Filter bit 14 */
#define  CAN_F13R2_FB15                      0x00008000U        /*!<Filter bit 15 */
#define  CAN_F13R2_FB16                      0x00010000U        /*!<Filter bit 16 */
#define  CAN_F13R2_FB17                      0x00020000U        /*!<Filter bit 17 */
#define  CAN_F13R2_FB18                      0x00040000U        /*!<Filter bit 18 */
#define  CAN_F13R2_FB19                      0x00080000U        /*!<Filter bit 19 */
#define  CAN_F13R2_FB20                      0x00100000U        /*!<Filter bit 20 */
#define  CAN_F13R2_FB21                      0x00200000U        /*!<Filter bit 21 */
#define  CAN_F13R2_FB22                      0x00400000U        /*!<Filter bit 22 */
#define  CAN_F13R2_FB23                      0x00800000U        /*!<Filter bit 23 */
#define  CAN_F13R2_FB24                      0x01000000U        /*!<Filter bit 24 */
#define  CAN_F13R2_FB25                      0x02000000U        /*!<Filter bit 25 */
#define  CAN_F13R2_FB26                      0x04000000U        /*!<Filter bit 26 */
#define  CAN_F13R2_FB27                      0x08000000U        /*!<Filter bit 27 */
#define  CAN_F13R2_FB28                      0x10000000U        /*!<Filter bit 28 */
#define  CAN_F13R2_FB29                      0x20000000U        /*!<Filter bit 29 */
#define  CAN_F13R2_FB30                      0x40000000U        /*!<Filter bit 30 */
#define  CAN_F13R2_FB31                      0x80000000U        /*!<Filter bit 31 */

/******************************************************************************/
/*                                                                            */
/*                          CRC calculation unit                              */
/*                                                                            */
/******************************************************************************/
/*******************  Bit definition for CRC_DR register  *********************/
#define  CRC_DR_DR                           0xFFFFFFFFU /*!< Data register bits */


/*******************  Bit definition for CRC_IDR register  ********************/
#define  CRC_IDR_IDR                         0x000000FFU /*!< General-purpose 8-bit data register bits */


/********************  Bit definition for CRC_CR register  ********************/
#define  CRC_CR_RESET                        0x00000001U /*!< RESET bit */

/******************************************************************************/
/*                                                                            */
/*                      Digital to Analog Converter                           */
/*                                                                            */
/******************************************************************************/
/********************  Bit definition for DAC_CR register  ********************/
#define  DAC_CR_EN1                          0x00000001U        /*!<DAC channel1 enable */
#define  DAC_CR_BOFF1                        0x00000002U        /*!<DAC channel1 output buffer disable */
#define  DAC_CR_TEN1                         0x00000004U        /*!<DAC channel1 Trigger enable */

#define  DAC_CR_TSEL1                        0x00000038U        /*!<TSEL1[2:0] (DAC channel1 Trigger selection) */
#define  DAC_CR_TSEL1_0                      0x00000008U        /*!<Bit 0 */
#define  DAC_CR_TSEL1_1                      0x00000010U        /*!<Bit 1 */
#define  DAC_CR_TSEL1_2                      0x00000020U        /*!<Bit 2 */

#define  DAC_CR_WAVE1                        0x000000C0U        /*!<WAVE1[1:0] (DAC channel1 noise/triangle wave generation enable) */
#define  DAC_CR_WAVE1_0                      0x00000040U        /*!<Bit 0 */
#define  DAC_CR_WAVE1_1                      0x00000080U        /*!<Bit 1 */

#define  DAC_CR_MAMP1                        0x00000F00U        /*!<MAMP1[3:0] (DAC channel1 Mask/Amplitude selector) */
#define  DAC_CR_MAMP1_0                      0x00000100U        /*!<Bit 0 */
#define  DAC_CR_MAMP1_1                      0x00000200U        /*!<Bit 1 */
#define  DAC_CR_MAMP1_2                      0x00000400U        /*!<Bit 2 */
#define  DAC_CR_MAMP1_3                      0x00000800U        /*!<Bit 3 */

#define  DAC_CR_DMAEN1                       0x00001000U        /*!<DAC channel1 DMA enable */
#define  DAC_CR_DMAUDRIE1                    0x00002000U        /*!<DAC channel1 DMA underrun interrupt enable*/
#define  DAC_CR_EN2                          0x00010000U        /*!<DAC channel2 enable */
#define  DAC_CR_BOFF2                        0x00020000U        /*!<DAC channel2 output buffer disable */
#define  DAC_CR_TEN2                         0x00040000U        /*!<DAC channel2 Trigger enable */

#define  DAC_CR_TSEL2                        0x00380000U        /*!<TSEL2[2:0] (DAC channel2 Trigger selection) */
#define  DAC_CR_TSEL2_0                      0x00080000U        /*!<Bit 0 */
#define  DAC_CR_TSEL2_1                      0x00100000U        /*!<Bit 1 */
#define  DAC_CR_TSEL2_2                      0x00200000U        /*!<Bit 2 */

#define  DAC_CR_WAVE2                        0x00C00000U        /*!<WAVE2[1:0] (DAC channel2 noise/triangle wave generation enable) */
#define  DAC_CR_WAVE2_0                      0x00400000U        /*!<Bit 0 */
#define  DAC_CR_WAVE2_1                      0x00800000U        /*!<Bit 1 */

#define  DAC_CR_MAMP2                        0x0F000000U        /*!<MAMP2[3:0] (DAC channel2 Mask/Amplitude selector) */
#define  DAC_CR_MAMP2_0                      0x01000000U        /*!<Bit 0 */
#define  DAC_CR_MAMP2_1                      0x02000000U        /*!<Bit 1 */
#define  DAC_CR_MAMP2_2                      0x04000000U        /*!<Bit 2 */
#define  DAC_CR_MAMP2_3                      0x08000000U        /*!<Bit 3 */

#define  DAC_CR_DMAEN2                       0x10000000U        /*!<DAC channel2 DMA enabled */
#define  DAC_CR_DMAUDRIE2                    0x20000000U        /*!<DAC channel2 DMA underrun interrupt enable*/

/*****************  Bit definition for DAC_SWTRIGR register  ******************/
#define  DAC_SWTRIGR_SWTRIG1                 0x00000001U        /*!<DAC channel1 software trigger */
#define  DAC_SWTRIGR_SWTRIG2                 0x00000002U        /*!<DAC channel2 software trigger */

/*****************  Bit definition for DAC_DHR12R1 register  ******************/
#define  DAC_DHR12R1_DACC1DHR                0x00000FFFU        /*!<DAC channel1 12-bit Right aligned data */

/*****************  Bit definition for DAC_DHR12L1 register  ******************/
#define  DAC_DHR12L1_DACC1DHR                0x0000FFF0U        /*!<DAC channel1 12-bit Left aligned data */

/******************  Bit definition for DAC_DHR8R1 register  ******************/
#define  DAC_DHR8R1_DACC1DHR                 0x000000FFU        /*!<DAC channel1 8-bit Right aligned data */

/*****************  Bit definition for DAC_DHR12R2 register  ******************/
#define  DAC_DHR12R2_DACC2DHR                0x00000FFFU        /*!<DAC channel2 12-bit Right aligned data */

/*****************  Bit definition for DAC_DHR12L2 register  ******************/
#define  DAC_DHR12L2_DACC2DHR                0x0000FFF0U        /*!<DAC channel2 12-bit Left aligned data */

/******************  Bit definition for DAC_DHR8R2 register  ******************/
#define  DAC_DHR8R2_DACC2DHR                 0x000000FFU        /*!<DAC channel2 8-bit Right aligned data */

/*****************  Bit definition for DAC_DHR12RD register  ******************/
#define  DAC_DHR12RD_DACC1DHR                0x00000FFFU        /*!<DAC channel1 12-bit Right aligned data */
#define  DAC_DHR12RD_DACC2DHR                0x0FFF0000U        /*!<DAC channel2 12-bit Right aligned data */

/*****************  Bit definition for DAC_DHR12LD register  ******************/
#define  DAC_DHR12LD_DACC1DHR                0x0000FFF0U        /*!<DAC channel1 12-bit Left aligned data */
#define  DAC_DHR12LD_DACC2DHR                0xFFF00000U        /*!<DAC channel2 12-bit Left aligned data */

/******************  Bit definition for DAC_DHR8RD register  ******************/
#define  DAC_DHR8RD_DACC1DHR                 0x000000FFU        /*!<DAC channel1 8-bit Right aligned data */
#define  DAC_DHR8RD_DACC2DHR                 0x0000FF00U        /*!<DAC channel2 8-bit Right aligned data */

/*******************  Bit definition for DAC_DOR1 register  *******************/
#define  DAC_DOR1_DACC1DOR                   0x00000FFFU        /*!<DAC channel1 data output */

/*******************  Bit definition for DAC_DOR2 register  *******************/
#define  DAC_DOR2_DACC2DOR                   0x00000FFFU        /*!<DAC channel2 data output */

/********************  Bit definition for DAC_SR register  ********************/
#define  DAC_SR_DMAUDR1                      0x00002000U        /*!<DAC channel1 DMA underrun flag */
#define  DAC_SR_DMAUDR2                      0x20000000U        /*!<DAC channel2 DMA underrun flag */

/******************************************************************************/
/*                                                                            */
/*                                 Debug MCU                                  */
/*                                                                            */
/******************************************************************************/

/******************************************************************************/
/*                                                                            */
/*                             DMA Controller                                 */
/*                                                                            */
/******************************************************************************/
/********************  Bits definition for DMA_SxCR register  *****************/
#define DMA_SxCR_CHSEL                       0x0E000000U
#define DMA_SxCR_CHSEL_0                     0x02000000U
#define DMA_SxCR_CHSEL_1                     0x04000000U
#define DMA_SxCR_CHSEL_2                     0x08000000U
#define DMA_SxCR_MBURST                      0x01800000U
#define DMA_SxCR_MBURST_0                    0x00800000U
#define DMA_SxCR_MBURST_1                    0x01000000U
#define DMA_SxCR_PBURST                      0x00600000U
#define DMA_SxCR_PBURST_0                    0x00200000U
#define DMA_SxCR_PBURST_1                    0x00400000U
#define DMA_SxCR_CT                          0x00080000U
#define DMA_SxCR_DBM                         0x00040000U
#define DMA_SxCR_PL                          0x00030000U
#define DMA_SxCR_PL_0                        0x00010000U
#define DMA_SxCR_PL_1                        0x00020000U
#define DMA_SxCR_PINCOS                      0x00008000U
#define DMA_SxCR_MSIZE                       0x00006000U
#define DMA_SxCR_MSIZE_0                     0x00002000U
#define DMA_SxCR_MSIZE_1                     0x00004000U
#define DMA_SxCR_PSIZE                       0x00001800U
#define DMA_SxCR_PSIZE_0                     0x00000800U
#define DMA_SxCR_PSIZE_1                     0x00001000U
#define DMA_SxCR_MINC                        0x00000400U
#define DMA_SxCR_PINC                        0x00000200U
#define DMA_SxCR_CIRC                        0x00000100U
#define DMA_SxCR_DIR                         0x000000C0U
#define DMA_SxCR_DIR_0                       0x00000040U
#define DMA_SxCR_DIR_1                       0x00000080U
#define DMA_SxCR_PFCTRL                      0x00000020U
#define DMA_SxCR_TCIE                        0x00000010U
#define DMA_SxCR_HTIE                        0x00000008U
#define DMA_SxCR_TEIE                        0x00000004U
#define DMA_SxCR_DMEIE                       0x00000002U
#define DMA_SxCR_EN                          0x00000001U

/* Legacy defines */
#define DMA_SxCR_ACK                         0x00100000U

/********************  Bits definition for DMA_SxCNDTR register  **************/
#define DMA_SxNDT                            0x0000FFFFU
#define DMA_SxNDT_0                          0x00000001U
#define DMA_SxNDT_1                          0x00000002U
#define DMA_SxNDT_2                          0x00000004U
#define DMA_SxNDT_3                          0x00000008U
#define DMA_SxNDT_4                          0x00000010U
#define DMA_SxNDT_5                          0x00000020U
#define DMA_SxNDT_6                          0x00000040U
#define DMA_SxNDT_7                          0x00000080U
#define DMA_SxNDT_8                          0x00000100U
#define DMA_SxNDT_9                          0x00000200U
#define DMA_SxNDT_10                         0x00000400U
#define DMA_SxNDT_11                         0x00000800U
#define DMA_SxNDT_12                         0x00001000U
#define DMA_SxNDT_13                         0x00002000U
#define DMA_SxNDT_14                         0x00004000U
#define DMA_SxNDT_15                         0x00008000U

/********************  Bits definition for DMA_SxFCR register  ****************/ 
#define DMA_SxFCR_FEIE                       0x00000080U
#define DMA_SxFCR_FS                         0x00000038U
#define DMA_SxFCR_FS_0                       0x00000008U
#define DMA_SxFCR_FS_1                       0x00000010U
#define DMA_SxFCR_FS_2                       0x00000020U
#define DMA_SxFCR_DMDIS                      0x00000004U
#define DMA_SxFCR_FTH                        0x00000003U
#define DMA_SxFCR_FTH_0                      0x00000001U
#define DMA_SxFCR_FTH_1                      0x00000002U

/********************  Bits definition for DMA_LISR register  *****************/ 
#define DMA_LISR_TCIF3                       0x08000000U
#define DMA_LISR_HTIF3                       0x04000000U
#define DMA_LISR_TEIF3                       0x02000000U
#define DMA_LISR_DMEIF3                      0x01000000U
#define DMA_LISR_FEIF3                       0x00400000U
#define DMA_LISR_TCIF2                       0x00200000U
#define DMA_LISR_HTIF2                       0x00100000U
#define DMA_LISR_TEIF2                       0x00080000U
#define DMA_LISR_DMEIF2                      0x00040000U
#define DMA_LISR_FEIF2                       0x00010000U
#define DMA_LISR_TCIF1                       0x00000800U
#define DMA_LISR_HTIF1                       0x00000400U
#define DMA_LISR_TEIF1                       0x00000200U
#define DMA_LISR_DMEIF1                      0x00000100U
#define DMA_LISR_FEIF1                       0x00000040U
#define DMA_LISR_TCIF0                       0x00000020U
#define DMA_LISR_HTIF0                       0x00000010U
#define DMA_LISR_TEIF0                       0x00000008U
#define DMA_LISR_DMEIF0                      0x00000004U
#define DMA_LISR_FEIF0                       0x00000001U

/********************  Bits definition for DMA_HISR register  *****************/ 
#define DMA_HISR_TCIF7                       0x08000000U
#define DMA_HISR_HTIF7                       0x04000000U
#define DMA_HISR_TEIF7                       0x02000000U
#define DMA_HISR_DMEIF7                      0x01000000U
#define DMA_HISR_FEIF7                       0x00400000U
#define DMA_HISR_TCIF6                       0x00200000U
#define DMA_HISR_HTIF6                       0x00100000U
#define DMA_HISR_TEIF6                       0x00080000U
#define DMA_HISR_DMEIF6                      0x00040000U
#define DMA_HISR_FEIF6                       0x00010000U
#define DMA_HISR_TCIF5                       0x00000800U
#define DMA_HISR_HTIF5                       0x00000400U
#define DMA_HISR_TEIF5                       0x00000200U
#define DMA_HISR_DMEIF5                      0x00000100U
#define DMA_HISR_FEIF5                       0x00000040U
#define DMA_HISR_TCIF4                       0x00000020U
#define DMA_HISR_HTIF4                       0x00000010U
#define DMA_HISR_TEIF4                       0x00000008U
#define DMA_HISR_DMEIF4                      0x00000004U
#define DMA_HISR_FEIF4                       0x00000001U

/********************  Bits definition for DMA_LIFCR register  ****************/ 
#define DMA_LIFCR_CTCIF3                     0x08000000U
#define DMA_LIFCR_CHTIF3                     0x04000000U
#define DMA_LIFCR_CTEIF3                     0x02000000U
#define DMA_LIFCR_CDMEIF3                    0x01000000U
#define DMA_LIFCR_CFEIF3                     0x00400000U
#define DMA_LIFCR_CTCIF2                     0x00200000U
#define DMA_LIFCR_CHTIF2                     0x00100000U
#define DMA_LIFCR_CTEIF2                     0x00080000U
#define DMA_LIFCR_CDMEIF2                    0x00040000U
#define DMA_LIFCR_CFEIF2                     0x00010000U
#define DMA_LIFCR_CTCIF1                     0x00000800U
#define DMA_LIFCR_CHTIF1                     0x00000400U
#define DMA_LIFCR_CTEIF1                     0x00000200U
#define DMA_LIFCR_CDMEIF1                    0x00000100U
#define DMA_LIFCR_CFEIF1                     0x00000040U
#define DMA_LIFCR_CTCIF0                     0x00000020U
#define DMA_LIFCR_CHTIF0                     0x00000010U
#define DMA_LIFCR_CTEIF0                     0x00000008U
#define DMA_LIFCR_CDMEIF0                    0x00000004U
#define DMA_LIFCR_CFEIF0                     0x00000001U

/********************  Bits definition for DMA_HIFCR  register  ****************/ 
#define DMA_HIFCR_CTCIF7                     0x08000000U
#define DMA_HIFCR_CHTIF7                     0x04000000U
#define DMA_HIFCR_CTEIF7                     0x02000000U
#define DMA_HIFCR_CDMEIF7                    0x01000000U
#define DMA_HIFCR_CFEIF7                     0x00400000U
#define DMA_HIFCR_CTCIF6                     0x00200000U
#define DMA_HIFCR_CHTIF6                     0x00100000U
#define DMA_HIFCR_CTEIF6                     0x00080000U
#define DMA_HIFCR_CDMEIF6                    0x00040000U
#define DMA_HIFCR_CFEIF6                     0x00010000U
#define DMA_HIFCR_CTCIF5                     0x00000800U
#define DMA_HIFCR_CHTIF5                     0x00000400U
#define DMA_HIFCR_CTEIF5                     0x00000200U
#define DMA_HIFCR_CDMEIF5                    0x00000100U
#define DMA_HIFCR_CFEIF5                     0x00000040U
#define DMA_HIFCR_CTCIF4                     0x00000020U
#define DMA_HIFCR_CHTIF4                     0x00000010U
#define DMA_HIFCR_CTEIF4                     0x00000008U
#define DMA_HIFCR_CDMEIF4                    0x00000004U
#define DMA_HIFCR_CFEIF4                     0x00000001U


/******************************************************************************/
/*                                                                            */
/*                    External Interrupt/Event Controller                     */
/*                                                                            */
/******************************************************************************/
/*******************  Bit definition for EXTI_IMR register  *******************/
#define  EXTI_IMR_MR0                        0x00000001U        /*!< Interrupt Mask on line 0 */
#define  EXTI_IMR_MR1                        0x00000002U        /*!< Interrupt Mask on line 1 */
#define  EXTI_IMR_MR2                        0x00000004U        /*!< Interrupt Mask on line 2 */
#define  EXTI_IMR_MR3                        0x00000008U        /*!< Interrupt Mask on line 3 */
#define  EXTI_IMR_MR4                        0x00000010U        /*!< Interrupt Mask on line 4 */
#define  EXTI_IMR_MR5                        0x00000020U        /*!< Interrupt Mask on line 5 */
#define  EXTI_IMR_MR6                        0x00000040U        /*!< Interrupt Mask on line 6 */
#define  EXTI_IMR_MR7                        0x00000080U        /*!< Interrupt Mask on line 7 */
#define  EXTI_IMR_MR8                        0x00000100U        /*!< Interrupt Mask on line 8 */
#define  EXTI_IMR_MR9                        0x00000200U        /*!< Interrupt Mask on line 9 */
#define  EXTI_IMR_MR10                       0x00000400U        /*!< Interrupt Mask on line 10 */
#define  EXTI_IMR_MR11                       0x00000800U        /*!< Interrupt Mask on line 11 */
#define  EXTI_IMR_MR12                       0x00001000U        /*!< Interrupt Mask on line 12 */
#define  EXTI_IMR_MR13                       0x00002000U        /*!< Interrupt Mask on line 13 */
#define  EXTI_IMR_MR14                       0x00004000U        /*!< Interrupt Mask on line 14 */
#define  EXTI_IMR_MR15                       0x00008000U        /*!< Interrupt Mask on line 15 */
#define  EXTI_IMR_MR16                       0x00010000U        /*!< Interrupt Mask on line 16 */
#define  EXTI_IMR_MR17                       0x00020000U        /*!< Interrupt Mask on line 17 */
#define  EXTI_IMR_MR18                       0x00040000U        /*!< Interrupt Mask on line 18 */
#define  EXTI_IMR_MR19                       0x00080000U        /*!< Interrupt Mask on line 19 */
#define  EXTI_IMR_MR20                       0x00100000U        /*!< Interrupt Mask on line 20 */
#define  EXTI_IMR_MR21                       0x00200000U        /*!< Interrupt Mask on line 21 */
#define  EXTI_IMR_MR22                       0x00400000U        /*!< Interrupt Mask on line 22 */

/*******************  Bit definition for EXTI_EMR register  *******************/
#define  EXTI_EMR_MR0                        0x00000001U        /*!< Event Mask on line 0 */
#define  EXTI_EMR_MR1                        0x00000002U        /*!< Event Mask on line 1 */
#define  EXTI_EMR_MR2                        0x00000004U        /*!< Event Mask on line 2 */
#define  EXTI_EMR_MR3                        0x00000008U        /*!< Event Mask on line 3 */
#define  EXTI_EMR_MR4                        0x00000010U        /*!< Event Mask on line 4 */
#define  EXTI_EMR_MR5                        0x00000020U        /*!< Event Mask on line 5 */
#define  EXTI_EMR_MR6                        0x00000040U        /*!< Event Mask on line 6 */
#define  EXTI_EMR_MR7                        0x00000080U        /*!< Event Mask on line 7 */
#define  EXTI_EMR_MR8                        0x00000100U        /*!< Event Mask on line 8 */
#define  EXTI_EMR_MR9                        0x00000200U        /*!< Event Mask on line 9 */
#define  EXTI_EMR_MR10                       0x00000400U        /*!< Event Mask on line 10 */
#define  EXTI_EMR_MR11                       0x00000800U        /*!< Event Mask on line 11 */
#define  EXTI_EMR_MR12                       0x00001000U        /*!< Event Mask on line 12 */
#define  EXTI_EMR_MR13                       0x00002000U        /*!< Event Mask on line 13 */
#define  EXTI_EMR_MR14                       0x00004000U        /*!< Event Mask on line 14 */
#define  EXTI_EMR_MR15                       0x00008000U        /*!< Event Mask on line 15 */
#define  EXTI_EMR_MR16                       0x00010000U        /*!< Event Mask on line 16 */
#define  EXTI_EMR_MR17                       0x00020000U        /*!< Event Mask on line 17 */
#define  EXTI_EMR_MR18                       0x00040000U        /*!< Event Mask on line 18 */
#define  EXTI_EMR_MR19                       0x00080000U        /*!< Event Mask on line 19 */
#define  EXTI_EMR_MR20                       0x00100000U        /*!< Event Mask on line 20 */
#define  EXTI_EMR_MR21                       0x00200000U        /*!< Event Mask on line 21 */
#define  EXTI_EMR_MR22                       0x00400000U        /*!< Event Mask on line 22 */

/******************  Bit definition for EXTI_RTSR register  *******************/
#define  EXTI_RTSR_TR0                       0x00000001U        /*!< Rising trigger event configuration bit of line 0 */
#define  EXTI_RTSR_TR1                       0x00000002U        /*!< Rising trigger event configuration bit of line 1 */
#define  EXTI_RTSR_TR2                       0x00000004U        /*!< Rising trigger event configuration bit of line 2 */
#define  EXTI_RTSR_TR3                       0x00000008U        /*!< Rising trigger event configuration bit of line 3 */
#define  EXTI_RTSR_TR4                       0x00000010U        /*!< Rising trigger event configuration bit of line 4 */
#define  EXTI_RTSR_TR5                       0x00000020U        /*!< Rising trigger event configuration bit of line 5 */
#define  EXTI_RTSR_TR6                       0x00000040U        /*!< Rising trigger event configuration bit of line 6 */
#define  EXTI_RTSR_TR7                       0x00000080U        /*!< Rising trigger event configuration bit of line 7 */
#define  EXTI_RTSR_TR8                       0x00000100U        /*!< Rising trigger event configuration bit of line 8 */
#define  EXTI_RTSR_TR9                       0x00000200U        /*!< Rising trigger event configuration bit of line 9 */
#define  EXTI_RTSR_TR10                      0x00000400U        /*!< Rising trigger event configuration bit of line 10 */
#define  EXTI_RTSR_TR11                      0x00000800U        /*!< Rising trigger event configuration bit of line 11 */
#define  EXTI_RTSR_TR12                      0x00001000U        /*!< Rising trigger event configuration bit of line 12 */
#define  EXTI_RTSR_TR13                      0x00002000U        /*!< Rising trigger event configuration bit of line 13 */
#define  EXTI_RTSR_TR14                      0x00004000U        /*!< Rising trigger event configuration bit of line 14 */
#define  EXTI_RTSR_TR15                      0x00008000U        /*!< Rising trigger event configuration bit of line 15 */
#define  EXTI_RTSR_TR16                      0x00010000U        /*!< Rising trigger event configuration bit of line 16 */
#define  EXTI_RTSR_TR17                      0x00020000U        /*!< Rising trigger event configuration bit of line 17 */
#define  EXTI_RTSR_TR18                      0x00040000U        /*!< Rising trigger event configuration bit of line 18 */
#define  EXTI_RTSR_TR19                      0x00080000U        /*!< Rising trigger event configuration bit of line 19 */
#define  EXTI_RTSR_TR20                      0x00100000U        /*!< Rising trigger event configuration bit of line 20 */
#define  EXTI_RTSR_TR21                      0x00200000U        /*!< Rising trigger event configuration bit of line 21 */
#define  EXTI_RTSR_TR22                      0x00400000U        /*!< Rising trigger event configuration bit of line 22 */

/******************  Bit definition for EXTI_FTSR register  *******************/
#define  EXTI_FTSR_TR0                       0x00000001U        /*!< Falling trigger event configuration bit of line 0 */
#define  EXTI_FTSR_TR1                       0x00000002U        /*!< Falling trigger event configuration bit of line 1 */
#define  EXTI_FTSR_TR2                       0x00000004U        /*!< Falling trigger event configuration bit of line 2 */
#define  EXTI_FTSR_TR3                       0x00000008U        /*!< Falling trigger event configuration bit of line 3 */
#define  EXTI_FTSR_TR4                       0x00000010U        /*!< Falling trigger event configuration bit of line 4 */
#define  EXTI_FTSR_TR5                       0x00000020U        /*!< Falling trigger event configuration bit of line 5 */
#define  EXTI_FTSR_TR6                       0x00000040U        /*!< Falling trigger event configuration bit of line 6 */
#define  EXTI_FTSR_TR7                       0x00000080U        /*!< Falling trigger event configuration bit of line 7 */
#define  EXTI_FTSR_TR8                       0x00000100U        /*!< Falling trigger event configuration bit of line 8 */
#define  EXTI_FTSR_TR9                       0x00000200U        /*!< Falling trigger event configuration bit of line 9 */
#define  EXTI_FTSR_TR10                      0x00000400U        /*!< Falling trigger event configuration bit of line 10 */
#define  EXTI_FTSR_TR11                      0x00000800U        /*!< Falling trigger event configuration bit of line 11 */
#define  EXTI_FTSR_TR12                      0x00001000U        /*!< Falling trigger event configuration bit of line 12 */
#define  EXTI_FTSR_TR13                      0x00002000U        /*!< Falling trigger event configuration bit of line 13 */
#define  EXTI_FTSR_TR14                      0x00004000U        /*!< Falling trigger event configuration bit of line 14 */
#define  EXTI_FTSR_TR15                      0x00008000U        /*!< Falling trigger event configuration bit of line 15 */
#define  EXTI_FTSR_TR16                      0x00010000U        /*!< Falling trigger event configuration bit of line 16 */
#define  EXTI_FTSR_TR17                      0x00020000U        /*!< Falling trigger event configuration bit of line 17 */
#define  EXTI_FTSR_TR18                      0x00040000U        /*!< Falling trigger event configuration bit of line 18 */
#define  EXTI_FTSR_TR19                      0x00080000U        /*!< Falling trigger event configuration bit of line 19 */
#define  EXTI_FTSR_TR20                      0x00100000U        /*!< Falling trigger event configuration bit of line 20 */
#define  EXTI_FTSR_TR21                      0x00200000U        /*!< Falling trigger event configuration bit of line 21 */
#define  EXTI_FTSR_TR22                      0x00400000U        /*!< Falling trigger event configuration bit of line 22 */

/******************  Bit definition for EXTI_SWIER register  ******************/
#define  EXTI_SWIER_SWIER0                   0x00000001U        /*!< Software Interrupt on line 0 */
#define  EXTI_SWIER_SWIER1                   0x00000002U        /*!< Software Interrupt on line 1 */
#define  EXTI_SWIER_SWIER2                   0x00000004U        /*!< Software Interrupt on line 2 */
#define  EXTI_SWIER_SWIER3                   0x00000008U        /*!< Software Interrupt on line 3 */
#define  EXTI_SWIER_SWIER4                   0x00000010U        /*!< Software Interrupt on line 4 */
#define  EXTI_SWIER_SWIER5                   0x00000020U        /*!< Software Interrupt on line 5 */
#define  EXTI_SWIER_SWIER6                   0x00000040U        /*!< Software Interrupt on line 6 */
#define  EXTI_SWIER_SWIER7                   0x00000080U        /*!< Software Interrupt on line 7 */
#define  EXTI_SWIER_SWIER8                   0x00000100U        /*!< Software Interrupt on line 8 */
#define  EXTI_SWIER_SWIER9                   0x00000200U        /*!< Software Interrupt on line 9 */
#define  EXTI_SWIER_SWIER10                  0x00000400U        /*!< Software Interrupt on line 10 */
#define  EXTI_SWIER_SWIER11                  0x00000800U        /*!< Software Interrupt on line 11 */
#define  EXTI_SWIER_SWIER12                  0x00001000U        /*!< Software Interrupt on line 12 */
#define  EXTI_SWIER_SWIER13                  0x00002000U        /*!< Software Interrupt on line 13 */
#define  EXTI_SWIER_SWIER14                  0x00004000U        /*!< Software Interrupt on line 14 */
#define  EXTI_SWIER_SWIER15                  0x00008000U        /*!< Software Interrupt on line 15 */
#define  EXTI_SWIER_SWIER16                  0x00010000U        /*!< Software Interrupt on line 16 */
#define  EXTI_SWIER_SWIER17                  0x00020000U        /*!< Software Interrupt on line 17 */
#define  EXTI_SWIER_SWIER18                  0x00040000U        /*!< Software Interrupt on line 18 */
#define  EXTI_SWIER_SWIER19                  0x00080000U        /*!< Software Interrupt on line 19 */
#define  EXTI_SWIER_SWIER20                  0x00100000U        /*!< Software Interrupt on line 20 */
#define  EXTI_SWIER_SWIER21                  0x00200000U        /*!< Software Interrupt on line 21 */
#define  EXTI_SWIER_SWIER22                  0x00400000U        /*!< Software Interrupt on line 22 */

/*******************  Bit definition for EXTI_PR register  ********************/
#define  EXTI_PR_PR0                         0x00000001U        /*!< Pending bit for line 0 */
#define  EXTI_PR_PR1                         0x00000002U        /*!< Pending bit for line 1 */
#define  EXTI_PR_PR2                         0x00000004U        /*!< Pending bit for line 2 */
#define  EXTI_PR_PR3                         0x00000008U        /*!< Pending bit for line 3 */
#define  EXTI_PR_PR4                         0x00000010U        /*!< Pending bit for line 4 */
#define  EXTI_PR_PR5                         0x00000020U        /*!< Pending bit for line 5 */
#define  EXTI_PR_PR6                         0x00000040U        /*!< Pending bit for line 6 */
#define  EXTI_PR_PR7                         0x00000080U        /*!< Pending bit for line 7 */
#define  EXTI_PR_PR8                         0x00000100U        /*!< Pending bit for line 8 */
#define  EXTI_PR_PR9                         0x00000200U        /*!< Pending bit for line 9 */
#define  EXTI_PR_PR10                        0x00000400U        /*!< Pending bit for line 10 */
#define  EXTI_PR_PR11                        0x00000800U        /*!< Pending bit for line 11 */
#define  EXTI_PR_PR12                        0x00001000U        /*!< Pending bit for line 12 */
#define  EXTI_PR_PR13                        0x00002000U        /*!< Pending bit for line 13 */
#define  EXTI_PR_PR14                        0x00004000U        /*!< Pending bit for line 14 */
#define  EXTI_PR_PR15                        0x00008000U        /*!< Pending bit for line 15 */
#define  EXTI_PR_PR16                        0x00010000U        /*!< Pending bit for line 16 */
#define  EXTI_PR_PR17                        0x00020000U        /*!< Pending bit for line 17 */
#define  EXTI_PR_PR18                        0x00040000U        /*!< Pending bit for line 18 */
#define  EXTI_PR_PR19                        0x00080000U        /*!< Pending bit for line 19 */
#define  EXTI_PR_PR20                        0x00100000U        /*!< Pending bit for line 20 */
#define  EXTI_PR_PR21                        0x00200000U        /*!< Pending bit for line 21 */
#define  EXTI_PR_PR22                        0x00400000U        /*!< Pending bit for line 22 */

/******************************************************************************/
/*                                                                            */
/*                                    FLASH                                   */
/*                                                                            */
/******************************************************************************/
/*******************  Bits definition for FLASH_ACR register  *****************/
#define FLASH_ACR_LATENCY                    0x0000000FU
#define FLASH_ACR_LATENCY_0WS                0x00000000U
#define FLASH_ACR_LATENCY_1WS                0x00000001U
#define FLASH_ACR_LATENCY_2WS                0x00000002U
#define FLASH_ACR_LATENCY_3WS                0x00000003U
#define FLASH_ACR_LATENCY_4WS                0x00000004U
#define FLASH_ACR_LATENCY_5WS                0x00000005U
#define FLASH_ACR_LATENCY_6WS                0x00000006U
#define FLASH_ACR_LATENCY_7WS                0x00000007U

#define FLASH_ACR_PRFTEN                     0x00000100U
#define FLASH_ACR_ICEN                       0x00000200U
#define FLASH_ACR_DCEN                       0x00000400U
#define FLASH_ACR_ICRST                      0x00000800U
#define FLASH_ACR_DCRST                      0x00001000U
#define FLASH_ACR_BYTE0_ADDRESS              0x40023C00U
#define FLASH_ACR_BYTE2_ADDRESS              0x40023C03U

/*******************  Bits definition for FLASH_SR register  ******************/
#define FLASH_SR_EOP                         0x00000001U
#define FLASH_SR_SOP                         0x00000002U
#define FLASH_SR_WRPERR                      0x00000010U
#define FLASH_SR_PGAERR                      0x00000020U
#define FLASH_SR_PGPERR                      0x00000040U
#define FLASH_SR_PGSERR                      0x00000080U
#define FLASH_SR_BSY                         0x00010000U

/*******************  Bits definition for FLASH_CR register  ******************/
#define FLASH_CR_PG                          0x00000001U
#define FLASH_CR_SER                         0x00000002U
#define FLASH_CR_MER                         0x00000004U
#define FLASH_CR_SNB                         0x000000F8U
#define FLASH_CR_SNB_0                       0x00000008U
#define FLASH_CR_SNB_1                       0x00000010U
#define FLASH_CR_SNB_2                       0x00000020U
#define FLASH_CR_SNB_3                       0x00000040U
#define FLASH_CR_SNB_4                       0x00000080U
#define FLASH_CR_PSIZE                       0x00000300U
#define FLASH_CR_PSIZE_0                     0x00000100U
#define FLASH_CR_PSIZE_1                     0x00000200U
#define FLASH_CR_STRT                        0x00010000U
#define FLASH_CR_EOPIE                       0x01000000U
#define FLASH_CR_LOCK                        0x80000000U

/*******************  Bits definition for FLASH_OPTCR register  ***************/
#define FLASH_OPTCR_OPTLOCK                 0x00000001U
#define FLASH_OPTCR_OPTSTRT                 0x00000002U
#define FLASH_OPTCR_BOR_LEV_0               0x00000004U
#define FLASH_OPTCR_BOR_LEV_1               0x00000008U
#define FLASH_OPTCR_BOR_LEV                 0x0000000CU

#define FLASH_OPTCR_WDG_SW                  0x00000020U
#define FLASH_OPTCR_nRST_STOP               0x00000040U
#define FLASH_OPTCR_nRST_STDBY              0x00000080U
#define FLASH_OPTCR_RDP                     0x0000FF00U
#define FLASH_OPTCR_RDP_0                   0x00000100U
#define FLASH_OPTCR_RDP_1                   0x00000200U
#define FLASH_OPTCR_RDP_2                   0x00000400U
#define FLASH_OPTCR_RDP_3                   0x00000800U
#define FLASH_OPTCR_RDP_4                   0x00001000U
#define FLASH_OPTCR_RDP_5                   0x00002000U
#define FLASH_OPTCR_RDP_6                   0x00004000U
#define FLASH_OPTCR_RDP_7                   0x00008000U
#define FLASH_OPTCR_nWRP                    0x0FFF0000U
#define FLASH_OPTCR_nWRP_0                  0x00010000U
#define FLASH_OPTCR_nWRP_1                  0x00020000U
#define FLASH_OPTCR_nWRP_2                  0x00040000U
#define FLASH_OPTCR_nWRP_3                  0x00080000U
#define FLASH_OPTCR_nWRP_4                  0x00100000U
#define FLASH_OPTCR_nWRP_5                  0x00200000U
#define FLASH_OPTCR_nWRP_6                  0x00400000U
#define FLASH_OPTCR_nWRP_7                  0x00800000U
#define FLASH_OPTCR_nWRP_8                  0x01000000U
#define FLASH_OPTCR_nWRP_9                  0x02000000U
#define FLASH_OPTCR_nWRP_10                 0x04000000U
#define FLASH_OPTCR_nWRP_11                 0x08000000U

/******************************************************************************/
/*                                                                            */
/*                       Flexible Static Memory Controller                    */
/*                                                                            */
/******************************************************************************/
/******************  Bit definition for FSMC_BCR1 register  *******************/
#define  FSMC_BCR1_MBKEN                     0x00000001U        /*!<Memory bank enable bit                 */
#define  FSMC_BCR1_MUXEN                     0x00000002U        /*!<Address/data multiplexing enable bit   */

#define  FSMC_BCR1_MTYP                      0x0000000CU        /*!<MTYP[1:0] bits (Memory type)           */
#define  FSMC_BCR1_MTYP_0                    0x00000004U        /*!<Bit 0 */
#define  FSMC_BCR1_MTYP_1                    0x00000008U        /*!<Bit 1 */

#define  FSMC_BCR1_MWID                      0x00000030U        /*!<MWID[1:0] bits (Memory data bus width) */
#define  FSMC_BCR1_MWID_0                    0x00000010U        /*!<Bit 0 */
#define  FSMC_BCR1_MWID_1                    0x00000020U        /*!<Bit 1 */

#define  FSMC_BCR1_FACCEN                    0x00000040U        /*!<Flash access enable                    */
#define  FSMC_BCR1_BURSTEN                   0x00000100U        /*!<Burst enable bit                       */
#define  FSMC_BCR1_WAITPOL                   0x00000200U        /*!<Wait signal polarity bit               */
#define  FSMC_BCR1_WRAPMOD                   0x00000400U        /*!<Wrapped burst mode support             */
#define  FSMC_BCR1_WAITCFG                   0x00000800U        /*!<Wait timing configuration              */
#define  FSMC_BCR1_WREN                      0x00001000U        /*!<Write enable bit                       */
#define  FSMC_BCR1_WAITEN                    0x00002000U        /*!<Wait enable bit                        */
#define  FSMC_BCR1_EXTMOD                    0x00004000U        /*!<Extended mode enable                   */
#define  FSMC_BCR1_ASYNCWAIT                 0x00008000U        /*!<Asynchronous wait                      */
#define  FSMC_BCR1_CBURSTRW                  0x00080000U        /*!<Write burst enable                     */

/******************  Bit definition for FSMC_BCR2 register  *******************/
#define  FSMC_BCR2_MBKEN                     0x00000001U        /*!<Memory bank enable bit                */
#define  FSMC_BCR2_MUXEN                     0x00000002U        /*!<Address/data multiplexing enable bit   */

#define  FSMC_BCR2_MTYP                      0x0000000CU        /*!<MTYP[1:0] bits (Memory type)           */
#define  FSMC_BCR2_MTYP_0                    0x00000004U        /*!<Bit 0 */
#define  FSMC_BCR2_MTYP_1                    0x00000008U        /*!<Bit 1 */

#define  FSMC_BCR2_MWID                      0x00000030U        /*!<MWID[1:0] bits (Memory data bus width) */
#define  FSMC_BCR2_MWID_0                    0x00000010U        /*!<Bit 0 */
#define  FSMC_BCR2_MWID_1                    0x00000020U        /*!<Bit 1 */

#define  FSMC_BCR2_FACCEN                    0x00000040U        /*!<Flash access enable                    */
#define  FSMC_BCR2_BURSTEN                   0x00000100U        /*!<Burst enable bit                       */
#define  FSMC_BCR2_WAITPOL                   0x00000200U        /*!<Wait signal polarity bit               */
#define  FSMC_BCR2_WRAPMOD                   0x00000400U        /*!<Wrapped burst mode support             */
#define  FSMC_BCR2_WAITCFG                   0x00000800U        /*!<Wait timing configuration              */
#define  FSMC_BCR2_WREN                      0x00001000U        /*!<Write enable bit                       */
#define  FSMC_BCR2_WAITEN                    0x00002000U        /*!<Wait enable bit                        */
#define  FSMC_BCR2_EXTMOD                    0x00004000U        /*!<Extended mode enable                   */
#define  FSMC_BCR2_ASYNCWAIT                 0x00008000U        /*!<Asynchronous wait                      */
#define  FSMC_BCR2_CBURSTRW                  0x00080000U        /*!<Write burst enable                     */

/******************  Bit definition for FSMC_BCR3 register  *******************/
#define  FSMC_BCR3_MBKEN                     0x00000001U        /*!<Memory bank enable bit                 */
#define  FSMC_BCR3_MUXEN                     0x00000002U        /*!<Address/data multiplexing enable bit   */

#define  FSMC_BCR3_MTYP                      0x0000000CU        /*!<MTYP[1:0] bits (Memory type)           */
#define  FSMC_BCR3_MTYP_0                    0x00000004U        /*!<Bit 0 */
#define  FSMC_BCR3_MTYP_1                    0x00000008U        /*!<Bit 1 */

#define  FSMC_BCR3_MWID                      0x00000030U        /*!<MWID[1:0] bits (Memory data bus width) */
#define  FSMC_BCR3_MWID_0                    0x00000010U        /*!<Bit 0 */
#define  FSMC_BCR3_MWID_1                    0x00000020U        /*!<Bit 1 */

#define  FSMC_BCR3_FACCEN                    0x00000040U        /*!<Flash access enable                    */
#define  FSMC_BCR3_BURSTEN                   0x00000100U        /*!<Burst enable bit                       */
#define  FSMC_BCR3_WAITPOL                   0x00000200U        /*!<Wait signal polarity bit               */
#define  FSMC_BCR3_WRAPMOD                   0x00000400U        /*!<Wrapped burst mode support             */
#define  FSMC_BCR3_WAITCFG                   0x00000800U        /*!<Wait timing configuration              */
#define  FSMC_BCR3_WREN                      0x00001000U        /*!<Write enable bit                       */
#define  FSMC_BCR3_WAITEN                    0x00002000U        /*!<Wait enable bit                        */
#define  FSMC_BCR3_EXTMOD                    0x00004000U        /*!<Extended mode enable                   */
#define  FSMC_BCR3_ASYNCWAIT                 0x00008000U        /*!<Asynchronous wait                      */
#define  FSMC_BCR3_CBURSTRW                  0x00080000U        /*!<Write burst enable                     */

/******************  Bit definition for FSMC_BCR4 register  *******************/
#define  FSMC_BCR4_MBKEN                     0x00000001U        /*!<Memory bank enable bit */
#define  FSMC_BCR4_MUXEN                     0x00000002U        /*!<Address/data multiplexing enable bit   */

#define  FSMC_BCR4_MTYP                      0x0000000CU        /*!<MTYP[1:0] bits (Memory type)           */
#define  FSMC_BCR4_MTYP_0                    0x00000004U        /*!<Bit 0 */
#define  FSMC_BCR4_MTYP_1                    0x00000008U        /*!<Bit 1 */

#define  FSMC_BCR4_MWID                      0x00000030U        /*!<MWID[1:0] bits (Memory data bus width) */
#define  FSMC_BCR4_MWID_0                    0x00000010U        /*!<Bit 0 */
#define  FSMC_BCR4_MWID_1                    0x00000020U        /*!<Bit 1 */

#define  FSMC_BCR4_FACCEN                    0x00000040U        /*!<Flash access enable                    */
#define  FSMC_BCR4_BURSTEN                   0x00000100U        /*!<Burst enable bit                       */
#define  FSMC_BCR4_WAITPOL                   0x00000200U        /*!<Wait signal polarity bit               */
#define  FSMC_BCR4_WRAPMOD                   0x00000400U        /*!<Wrapped burst mode support             */
#define  FSMC_BCR4_WAITCFG                   0x00000800U        /*!<Wait timing configuration              */
#define  FSMC_BCR4_WREN                      0x00001000U        /*!<Write enable bit                       */
#define  FSMC_BCR4_WAITEN                    0x00002000U        /*!<Wait enable bit                        */
#define  FSMC_BCR4_EXTMOD                    0x00004000U        /*!<Extended mode enable                   */
#define  FSMC_BCR4_ASYNCWAIT                 0x00008000U        /*!<Asynchronous wait                      */
#define  FSMC_BCR4_CBURSTRW                  0x00080000U        /*!<Write burst enable                     */

/******************  Bit definition for FSMC_BTR1 register  ******************/
#define  FSMC_BTR1_ADDSET                    0x0000000FU        /*!<ADDSET[3:0] bits (Address setup phase duration) */
#define  FSMC_BTR1_ADDSET_0                  0x00000001U        /*!<Bit 0 */
#define  FSMC_BTR1_ADDSET_1                  0x00000002U        /*!<Bit 1 */
#define  FSMC_BTR1_ADDSET_2                  0x00000004U        /*!<Bit 2 */
#define  FSMC_BTR1_ADDSET_3                  0x00000008U        /*!<Bit 3 */

#define  FSMC_BTR1_ADDHLD                    0x000000F0U        /*!<ADDHLD[3:0] bits (Address-hold phase duration) */
#define  FSMC_BTR1_ADDHLD_0                  0x00000010U        /*!<Bit 0 */
#define  FSMC_BTR1_ADDHLD_1                  0x00000020U        /*!<Bit 1 */
#define  FSMC_BTR1_ADDHLD_2                  0x00000040U        /*!<Bit 2 */
#define  FSMC_BTR1_ADDHLD_3                  0x00000080U        /*!<Bit 3 */

#define  FSMC_BTR1_DATAST                    0x0000FF00U        /*!<DATAST [7:0] bits (Data-phase duration) */
#define  FSMC_BTR1_DATAST_0                  0x00000100U        /*!<Bit 0 */
#define  FSMC_BTR1_DATAST_1                  0x00000200U        /*!<Bit 1 */
#define  FSMC_BTR1_DATAST_2                  0x00000400U        /*!<Bit 2 */
#define  FSMC_BTR1_DATAST_3                  0x00000800U        /*!<Bit 3 */
#define  FSMC_BTR1_DATAST_4                  0x00001000U        /*!<Bit 4 */
#define  FSMC_BTR1_DATAST_5                  0x00002000U        /*!<Bit 5 */
#define  FSMC_BTR1_DATAST_6                  0x00004000U        /*!<Bit 6 */
#define  FSMC_BTR1_DATAST_7                  0x00008000U        /*!<Bit 7 */

#define  FSMC_BTR1_BUSTURN                   0x000F0000U        /*!<BUSTURN[3:0] bits (Bus turnaround phase duration) */
#define  FSMC_BTR1_BUSTURN_0                 0x00010000U        /*!<Bit 0 */
#define  FSMC_BTR1_BUSTURN_1                 0x00020000U        /*!<Bit 1 */
#define  FSMC_BTR1_BUSTURN_2                 0x00040000U        /*!<Bit 2 */
#define  FSMC_BTR1_BUSTURN_3                 0x00080000U        /*!<Bit 3 */

#define  FSMC_BTR1_CLKDIV                    0x00F00000U        /*!<CLKDIV[3:0] bits (Clock divide ratio) */
#define  FSMC_BTR1_CLKDIV_0                  0x00100000U        /*!<Bit 0 */
#define  FSMC_BTR1_CLKDIV_1                  0x00200000U        /*!<Bit 1 */
#define  FSMC_BTR1_CLKDIV_2                  0x00400000U        /*!<Bit 2 */
#define  FSMC_BTR1_CLKDIV_3                  0x00800000U        /*!<Bit 3 */

#define  FSMC_BTR1_DATLAT                    0x0F000000U        /*!<DATLA[3:0] bits (Data latency) */
#define  FSMC_BTR1_DATLAT_0                  0x01000000U        /*!<Bit 0 */
#define  FSMC_BTR1_DATLAT_1                  0x02000000U        /*!<Bit 1 */
#define  FSMC_BTR1_DATLAT_2                  0x04000000U        /*!<Bit 2 */
#define  FSMC_BTR1_DATLAT_3                  0x08000000U        /*!<Bit 3 */

#define  FSMC_BTR1_ACCMOD                    0x30000000U        /*!<ACCMOD[1:0] bits (Access mode) */
#define  FSMC_BTR1_ACCMOD_0                  0x10000000U        /*!<Bit 0 */
#define  FSMC_BTR1_ACCMOD_1                  0x20000000U        /*!<Bit 1 */

/******************  Bit definition for FSMC_BTR2 register  *******************/
#define  FSMC_BTR2_ADDSET                    0x0000000FU        /*!<ADDSET[3:0] bits (Address setup phase duration) */
#define  FSMC_BTR2_ADDSET_0                  0x00000001U        /*!<Bit 0 */
#define  FSMC_BTR2_ADDSET_1                  0x00000002U        /*!<Bit 1 */
#define  FSMC_BTR2_ADDSET_2                  0x00000004U        /*!<Bit 2 */
#define  FSMC_BTR2_ADDSET_3                  0x00000008U        /*!<Bit 3 */

#define  FSMC_BTR2_ADDHLD                    0x000000F0U        /*!<ADDHLD[3:0] bits (Address-hold phase duration) */
#define  FSMC_BTR2_ADDHLD_0                  0x00000010U        /*!<Bit 0 */
#define  FSMC_BTR2_ADDHLD_1                  0x00000020U        /*!<Bit 1 */
#define  FSMC_BTR2_ADDHLD_2                  0x00000040U        /*!<Bit 2 */
#define  FSMC_BTR2_ADDHLD_3                  0x00000080U        /*!<Bit 3 */

#define  FSMC_BTR2_DATAST                    0x0000FF00U        /*!<DATAST [7:0] bits (Data-phase duration) */
#define  FSMC_BTR2_DATAST_0                  0x00000100U        /*!<Bit 0 */
#define  FSMC_BTR2_DATAST_1                  0x00000200U        /*!<Bit 1 */
#define  FSMC_BTR2_DATAST_2                  0x00000400U        /*!<Bit 2 */
#define  FSMC_BTR2_DATAST_3                  0x00000800U        /*!<Bit 3 */
#define  FSMC_BTR2_DATAST_4                  0x00001000U        /*!<Bit 4 */
#define  FSMC_BTR2_DATAST_5                  0x00002000U        /*!<Bit 5 */
#define  FSMC_BTR2_DATAST_6                  0x00004000U        /*!<Bit 6 */
#define  FSMC_BTR2_DATAST_7                  0x00008000U        /*!<Bit 7 */

#define  FSMC_BTR2_BUSTURN                   0x000F0000U        /*!<BUSTURN[3:0] bits (Bus turnaround phase duration) */
#define  FSMC_BTR2_BUSTURN_0                 0x00010000U        /*!<Bit 0 */
#define  FSMC_BTR2_BUSTURN_1                 0x00020000U        /*!<Bit 1 */
#define  FSMC_BTR2_BUSTURN_2                 0x00040000U        /*!<Bit 2 */
#define  FSMC_BTR2_BUSTURN_3                 0x00080000U        /*!<Bit 3 */

#define  FSMC_BTR2_CLKDIV                    0x00F00000U        /*!<CLKDIV[3:0] bits (Clock divide ratio) */
#define  FSMC_BTR2_CLKDIV_0                  0x00100000U        /*!<Bit 0 */
#define  FSMC_BTR2_CLKDIV_1                  0x00200000U        /*!<Bit 1 */
#define  FSMC_BTR2_CLKDIV_2                  0x00400000U        /*!<Bit 2 */
#define  FSMC_BTR2_CLKDIV_3                  0x00800000U        /*!<Bit 3 */

#define  FSMC_BTR2_DATLAT                    0x0F000000U        /*!<DATLA[3:0] bits (Data latency) */
#define  FSMC_BTR2_DATLAT_0                  0x01000000U        /*!<Bit 0 */
#define  FSMC_BTR2_DATLAT_1                  0x02000000U        /*!<Bit 1 */
#define  FSMC_BTR2_DATLAT_2                  0x04000000U        /*!<Bit 2 */
#define  FSMC_BTR2_DATLAT_3                  0x08000000U        /*!<Bit 3 */

#define  FSMC_BTR2_ACCMOD                    0x30000000U        /*!<ACCMOD[1:0] bits (Access mode) */
#define  FSMC_BTR2_ACCMOD_0                  0x10000000U        /*!<Bit 0 */
#define  FSMC_BTR2_ACCMOD_1                  0x20000000U        /*!<Bit 1 */

/*******************  Bit definition for FSMC_BTR3 register  *******************/
#define  FSMC_BTR3_ADDSET                    0x0000000FU        /*!<ADDSET[3:0] bits (Address setup phase duration) */
#define  FSMC_BTR3_ADDSET_0                  0x00000001U        /*!<Bit 0 */
#define  FSMC_BTR3_ADDSET_1                  0x00000002U        /*!<Bit 1 */
#define  FSMC_BTR3_ADDSET_2                  0x00000004U        /*!<Bit 2 */
#define  FSMC_BTR3_ADDSET_3                  0x00000008U        /*!<Bit 3 */

#define  FSMC_BTR3_ADDHLD                    0x000000F0U        /*!<ADDHLD[3:0] bits (Address-hold phase duration) */
#define  FSMC_BTR3_ADDHLD_0                  0x00000010U        /*!<Bit 0 */
#define  FSMC_BTR3_ADDHLD_1                  0x00000020U        /*!<Bit 1 */
#define  FSMC_BTR3_ADDHLD_2                  0x00000040U        /*!<Bit 2 */
#define  FSMC_BTR3_ADDHLD_3                  0x00000080U        /*!<Bit 3 */

#define  FSMC_BTR3_DATAST                    0x0000FF00U        /*!<DATAST [7:0] bits (Data-phase duration) */
#define  FSMC_BTR3_DATAST_0                  0x00000100U        /*!<Bit 0 */
#define  FSMC_BTR3_DATAST_1                  0x00000200U        /*!<Bit 1 */
#define  FSMC_BTR3_DATAST_2                  0x00000400U        /*!<Bit 2 */
#define  FSMC_BTR3_DATAST_3                  0x00000800U        /*!<Bit 3 */
#define  FSMC_BTR3_DATAST_4                  0x00001000U        /*!<Bit 4 */
#define  FSMC_BTR3_DATAST_5                  0x00002000U        /*!<Bit 5 */
#define  FSMC_BTR3_DATAST_6                  0x00004000U        /*!<Bit 6 */
#define  FSMC_BTR3_DATAST_7                  0x00008000U        /*!<Bit 7 */

#define  FSMC_BTR3_BUSTURN                   0x000F0000U        /*!<BUSTURN[3:0] bits (Bus turnaround phase duration) */
#define  FSMC_BTR3_BUSTURN_0                 0x00010000U        /*!<Bit 0 */
#define  FSMC_BTR3_BUSTURN_1                 0x00020000U        /*!<Bit 1 */
#define  FSMC_BTR3_BUSTURN_2                 0x00040000U        /*!<Bit 2 */
#define  FSMC_BTR3_BUSTURN_3                 0x00080000U        /*!<Bit 3 */

#define  FSMC_BTR3_CLKDIV                    0x00F00000U        /*!<CLKDIV[3:0] bits (Clock divide ratio) */
#define  FSMC_BTR3_CLKDIV_0                  0x00100000U        /*!<Bit 0 */
#define  FSMC_BTR3_CLKDIV_1                  0x00200000U        /*!<Bit 1 */
#define  FSMC_BTR3_CLKDIV_2                  0x00400000U        /*!<Bit 2 */
#define  FSMC_BTR3_CLKDIV_3                  0x00800000U        /*!<Bit 3 */

#define  FSMC_BTR3_DATLAT                    0x0F000000U        /*!<DATLA[3:0] bits (Data latency) */
#define  FSMC_BTR3_DATLAT_0                  0x01000000U        /*!<Bit 0 */
#define  FSMC_BTR3_DATLAT_1                  0x02000000U        /*!<Bit 1 */
#define  FSMC_BTR3_DATLAT_2                  0x04000000U        /*!<Bit 2 */
#define  FSMC_BTR3_DATLAT_3                  0x08000000U        /*!<Bit 3 */

#define  FSMC_BTR3_ACCMOD                    0x30000000U        /*!<ACCMOD[1:0] bits (Access mode) */
#define  FSMC_BTR3_ACCMOD_0                  0x10000000U        /*!<Bit 0 */
#define  FSMC_BTR3_ACCMOD_1                  0x20000000U        /*!<Bit 1 */

/******************  Bit definition for FSMC_BTR4 register  *******************/
#define  FSMC_BTR4_ADDSET                    0x0000000FU        /*!<ADDSET[3:0] bits (Address setup phase duration) */
#define  FSMC_BTR4_ADDSET_0                  0x00000001U        /*!<Bit 0 */
#define  FSMC_BTR4_ADDSET_1                  0x00000002U        /*!<Bit 1 */
#define  FSMC_BTR4_ADDSET_2                  0x00000004U        /*!<Bit 2 */
#define  FSMC_BTR4_ADDSET_3                  0x00000008U        /*!<Bit 3 */

#define  FSMC_BTR4_ADDHLD                    0x000000F0U        /*!<ADDHLD[3:0] bits (Address-hold phase duration) */
#define  FSMC_BTR4_ADDHLD_0                  0x00000010U        /*!<Bit 0 */
#define  FSMC_BTR4_ADDHLD_1                  0x00000020U        /*!<Bit 1 */
#define  FSMC_BTR4_ADDHLD_2                  0x00000040U        /*!<Bit 2 */
#define  FSMC_BTR4_ADDHLD_3                  0x00000080U        /*!<Bit 3 */

#define  FSMC_BTR4_DATAST                    0x0000FF00U        /*!<DATAST [7:0] bits (Data-phase duration) */
#define  FSMC_BTR4_DATAST_0                  0x00000100U        /*!<Bit 0 */
#define  FSMC_BTR4_DATAST_1                  0x00000200U        /*!<Bit 1 */
#define  FSMC_BTR4_DATAST_2                  0x00000400U        /*!<Bit 2 */
#define  FSMC_BTR4_DATAST_3                  0x00000800U        /*!<Bit 3 */
#define  FSMC_BTR4_DATAST_4                  0x00001000U        /*!<Bit 4 */
#define  FSMC_BTR4_DATAST_5                  0x00002000U        /*!<Bit 5 */
#define  FSMC_BTR4_DATAST_6                  0x00004000U        /*!<Bit 6 */
#define  FSMC_BTR4_DATAST_7                  0x00008000U        /*!<Bit 7 */

#define  FSMC_BTR4_BUSTURN                   0x000F0000U        /*!<BUSTURN[3:0] bits (Bus turnaround phase duration) */
#define  FSMC_BTR4_BUSTURN_0                 0x00010000U        /*!<Bit 0 */
#define  FSMC_BTR4_BUSTURN_1                 0x00020000U        /*!<Bit 1 */
#define  FSMC_BTR4_BUSTURN_2                 0x00040000U        /*!<Bit 2 */
#define  FSMC_BTR4_BUSTURN_3                 0x00080000U        /*!<Bit 3 */

#define  FSMC_BTR4_CLKDIV                    0x00F00000U        /*!<CLKDIV[3:0] bits (Clock divide ratio) */
#define  FSMC_BTR4_CLKDIV_0                  0x00100000U        /*!<Bit 0 */
#define  FSMC_BTR4_CLKDIV_1                  0x00200000U        /*!<Bit 1 */
#define  FSMC_BTR4_CLKDIV_2                  0x00400000U        /*!<Bit 2 */
#define  FSMC_BTR4_CLKDIV_3                  0x00800000U        /*!<Bit 3 */

#define  FSMC_BTR4_DATLAT                    0x0F000000U        /*!<DATLA[3:0] bits (Data latency) */
#define  FSMC_BTR4_DATLAT_0                  0x01000000U        /*!<Bit 0 */
#define  FSMC_BTR4_DATLAT_1                  0x02000000U        /*!<Bit 1 */
#define  FSMC_BTR4_DATLAT_2                  0x04000000U        /*!<Bit 2 */
#define  FSMC_BTR4_DATLAT_3                  0x08000000U        /*!<Bit 3 */

#define  FSMC_BTR4_ACCMOD                    0x30000000U        /*!<ACCMOD[1:0] bits (Access mode) */
#define  FSMC_BTR4_ACCMOD_0                  0x10000000U        /*!<Bit 0 */
#define  FSMC_BTR4_ACCMOD_1                  0x20000000U        /*!<Bit 1 */

/******************  Bit definition for FSMC_BWTR1 register  ******************/
#define  FSMC_BWTR1_ADDSET                   0x0000000FU        /*!<ADDSET[3:0] bits (Address setup phase duration) */
#define  FSMC_BWTR1_ADDSET_0                 0x00000001U        /*!<Bit 0 */
#define  FSMC_BWTR1_ADDSET_1                 0x00000002U        /*!<Bit 1 */
#define  FSMC_BWTR1_ADDSET_2                 0x00000004U        /*!<Bit 2 */
#define  FSMC_BWTR1_ADDSET_3                 0x00000008U        /*!<Bit 3 */

#define  FSMC_BWTR1_ADDHLD                   0x000000F0U        /*!<ADDHLD[3:0] bits (Address-hold phase duration) */
#define  FSMC_BWTR1_ADDHLD_0                 0x00000010U        /*!<Bit 0 */
#define  FSMC_BWTR1_ADDHLD_1                 0x00000020U        /*!<Bit 1 */
#define  FSMC_BWTR1_ADDHLD_2                 0x00000040U        /*!<Bit 2 */
#define  FSMC_BWTR1_ADDHLD_3                 0x00000080U        /*!<Bit 3 */

#define  FSMC_BWTR1_DATAST                   0x0000FF00U        /*!<DATAST [7:0] bits (Data-phase duration) */
#define  FSMC_BWTR1_DATAST_0                 0x00000100U        /*!<Bit 0 */
#define  FSMC_BWTR1_DATAST_1                 0x00000200U        /*!<Bit 1 */
#define  FSMC_BWTR1_DATAST_2                 0x00000400U        /*!<Bit 2 */
#define  FSMC_BWTR1_DATAST_3                 0x00000800U        /*!<Bit 3 */
#define  FSMC_BWTR1_DATAST_4                 0x00001000U        /*!<Bit 4 */
#define  FSMC_BWTR1_DATAST_5                 0x00002000U        /*!<Bit 5 */
#define  FSMC_BWTR1_DATAST_6                 0x00004000U        /*!<Bit 6 */
#define  FSMC_BWTR1_DATAST_7                 0x00008000U        /*!<Bit 7 */

#define  FSMC_BWTR1_BUSTURN                  0x000F0000U        /*!<BUSTURN[3:0] bits (Bus turnaround duration) */
#define  FSMC_BWTR1_BUSTURN_0                0x00010000U        /*!<Bit 0 */
#define  FSMC_BWTR1_BUSTURN_1                0x00020000U        /*!<Bit 1 */
#define  FSMC_BWTR1_BUSTURN_2                0x00040000U        /*!<Bit 2 */
#define  FSMC_BWTR1_BUSTURN_3                0x00080000U        /*!<Bit 3 */

#define  FSMC_BWTR1_ACCMOD                   0x30000000U        /*!<ACCMOD[1:0] bits (Access mode) */
#define  FSMC_BWTR1_ACCMOD_0                 0x10000000U        /*!<Bit 0 */
#define  FSMC_BWTR1_ACCMOD_1                 0x20000000U        /*!<Bit 1 */

/******************  Bit definition for FSMC_BWTR2 register  ******************/
#define  FSMC_BWTR2_ADDSET                   0x0000000FU        /*!<ADDSET[3:0] bits (Address setup phase duration) */
#define  FSMC_BWTR2_ADDSET_0                 0x00000001U        /*!<Bit 0 */
#define  FSMC_BWTR2_ADDSET_1                 0x00000002U        /*!<Bit 1 */
#define  FSMC_BWTR2_ADDSET_2                 0x00000004U        /*!<Bit 2 */
#define  FSMC_BWTR2_ADDSET_3                 0x00000008U        /*!<Bit 3 */

#define  FSMC_BWTR2_ADDHLD                   0x000000F0U        /*!<ADDHLD[3:0] bits (Address-hold phase duration) */
#define  FSMC_BWTR2_ADDHLD_0                 0x00000010U        /*!<Bit 0 */
#define  FSMC_BWTR2_ADDHLD_1                 0x00000020U        /*!<Bit 1 */
#define  FSMC_BWTR2_ADDHLD_2                 0x00000040U        /*!<Bit 2 */
#define  FSMC_BWTR2_ADDHLD_3                 0x00000080U        /*!<Bit 3 */

#define  FSMC_BWTR2_DATAST                   0x0000FF00U        /*!<DATAST [7:0] bits (Data-phase duration) */
#define  FSMC_BWTR2_DATAST_0                 0x00000100U        /*!<Bit 0 */
#define  FSMC_BWTR2_DATAST_1                 0x00000200U        /*!<Bit 1 */
#define  FSMC_BWTR2_DATAST_2                 0x00000400U        /*!<Bit 2 */
#define  FSMC_BWTR2_DATAST_3                 0x00000800U        /*!<Bit 3 */
#define  FSMC_BWTR2_DATAST_4                 0x00001000U        /*!<Bit 4 */
#define  FSMC_BWTR2_DATAST_5                 0x00002000U        /*!<Bit 5 */
#define  FSMC_BWTR2_DATAST_6                 0x00004000U        /*!<Bit 6 */
#define  FSMC_BWTR2_DATAST_7                 0x00008000U        /*!<Bit 7 */

#define  FSMC_BWTR2_BUSTURN                  0x000F0000U        /*!<BUSTURN[3:0] bits (Bus turnaround duration) */
#define  FSMC_BWTR2_BUSTURN_0                0x00010000U        /*!<Bit 0 */
#define  FSMC_BWTR2_BUSTURN_1                0x00020000U        /*!<Bit 1 */
#define  FSMC_BWTR2_BUSTURN_2                0x00040000U        /*!<Bit 2 */
#define  FSMC_BWTR2_BUSTURN_3                0x00080000U        /*!<Bit 3 */

#define  FSMC_BWTR2_ACCMOD                   0x30000000U        /*!<ACCMOD[1:0] bits (Access mode) */
#define  FSMC_BWTR2_ACCMOD_0                 0x10000000U        /*!<Bit 0 */
#define  FSMC_BWTR2_ACCMOD_1                 0x20000000U        /*!<Bit 1 */

/******************  Bit definition for FSMC_BWTR3 register  ******************/
#define  FSMC_BWTR3_ADDSET                   0x0000000FU        /*!<ADDSET[3:0] bits (Address setup phase duration) */
#define  FSMC_BWTR3_ADDSET_0                 0x00000001U        /*!<Bit 0 */
#define  FSMC_BWTR3_ADDSET_1                 0x00000002U        /*!<Bit 1 */
#define  FSMC_BWTR3_ADDSET_2                 0x00000004U        /*!<Bit 2 */
#define  FSMC_BWTR3_ADDSET_3                 0x00000008U        /*!<Bit 3 */

#define  FSMC_BWTR3_ADDHLD                   0x000000F0U        /*!<ADDHLD[3:0] bits (Address-hold phase duration) */
#define  FSMC_BWTR3_ADDHLD_0                 0x00000010U        /*!<Bit 0 */
#define  FSMC_BWTR3_ADDHLD_1                 0x00000020U        /*!<Bit 1 */
#define  FSMC_BWTR3_ADDHLD_2                 0x00000040U        /*!<Bit 2 */
#define  FSMC_BWTR3_ADDHLD_3                 0x00000080U        /*!<Bit 3 */

#define  FSMC_BWTR3_DATAST                   0x0000FF00U        /*!<DATAST [7:0] bits (Data-phase duration) */
#define  FSMC_BWTR3_DATAST_0                 0x00000100U        /*!<Bit 0 */
#define  FSMC_BWTR3_DATAST_1                 0x00000200U        /*!<Bit 1 */
#define  FSMC_BWTR3_DATAST_2                 0x00000400U        /*!<Bit 2 */
#define  FSMC_BWTR3_DATAST_3                 0x00000800U        /*!<Bit 3 */
#define  FSMC_BWTR3_DATAST_4                 0x00001000U        /*!<Bit 4 */
#define  FSMC_BWTR3_DATAST_5                 0x00002000U        /*!<Bit 5 */
#define  FSMC_BWTR3_DATAST_6                 0x00004000U        /*!<Bit 6 */
#define  FSMC_BWTR3_DATAST_7                 0x00008000U        /*!<Bit 7 */

#define  FSMC_BWTR3_BUSTURN                  0x000F0000U        /*!<BUSTURN[3:0] bits (Bus turnaround duration) */
#define  FSMC_BWTR3_BUSTURN_0                0x00010000U        /*!<Bit 0 */
#define  FSMC_BWTR3_BUSTURN_1                0x00020000U        /*!<Bit 1 */
#define  FSMC_BWTR3_BUSTURN_2                0x00040000U        /*!<Bit 2 */
#define  FSMC_BWTR3_BUSTURN_3                0x00080000U        /*!<Bit 3 */

#define  FSMC_BWTR3_ACCMOD                   0x30000000U        /*!<ACCMOD[1:0] bits (Access mode) */
#define  FSMC_BWTR3_ACCMOD_0                 0x10000000U        /*!<Bit 0 */
#define  FSMC_BWTR3_ACCMOD_1                 0x20000000U        /*!<Bit 1 */

/******************  Bit definition for FSMC_BWTR4 register  ******************/
#define  FSMC_BWTR4_ADDSET                   0x0000000FU        /*!<ADDSET[3:0] bits (Address setup phase duration) */
#define  FSMC_BWTR4_ADDSET_0                 0x00000001U        /*!<Bit 0 */
#define  FSMC_BWTR4_ADDSET_1                 0x00000002U        /*!<Bit 1 */
#define  FSMC_BWTR4_ADDSET_2                 0x00000004U        /*!<Bit 2 */
#define  FSMC_BWTR4_ADDSET_3                 0x00000008U        /*!<Bit 3 */

#define  FSMC_BWTR4_ADDHLD                   0x000000F0U        /*!<ADDHLD[3:0] bits (Address-hold phase duration) */
#define  FSMC_BWTR4_ADDHLD_0                 0x00000010U        /*!<Bit 0 */
#define  FSMC_BWTR4_ADDHLD_1                 0x00000020U        /*!<Bit 1 */
#define  FSMC_BWTR4_ADDHLD_2                 0x00000040U        /*!<Bit 2 */
#define  FSMC_BWTR4_ADDHLD_3                 0x00000080U        /*!<Bit 3 */

#define  FSMC_BWTR4_DATAST                   0x0000FF00U        /*!<DATAST [3:0] bits (Data-phase duration) */
#define  FSMC_BWTR4_DATAST_0                 0x00000100U        /*!<Bit 0 */
#define  FSMC_BWTR4_DATAST_1                 0x00000200U        /*!<Bit 1 */
#define  FSMC_BWTR4_DATAST_2                 0x00000400U        /*!<Bit 2 */
#define  FSMC_BWTR4_DATAST_3                 0x00000800U        /*!<Bit 3 */
#define  FSMC_BWTR4_DATAST_4                 0x00001000U        /*!<Bit 4 */
#define  FSMC_BWTR4_DATAST_5                 0x00002000U        /*!<Bit 5 */
#define  FSMC_BWTR4_DATAST_6                 0x00004000U        /*!<Bit 6 */
#define  FSMC_BWTR4_DATAST_7                 0x00008000U        /*!<Bit 7 */

#define  FSMC_BWTR4_BUSTURN                  0x000F0000U        /*!<BUSTURN[3:0] bits (Bus turnaround duration) */
#define  FSMC_BWTR4_BUSTURN_0                0x00010000U        /*!<Bit 0 */
#define  FSMC_BWTR4_BUSTURN_1                0x00020000U        /*!<Bit 1 */
#define  FSMC_BWTR4_BUSTURN_2                0x00040000U        /*!<Bit 2 */
#define  FSMC_BWTR4_BUSTURN_3                0x00080000U        /*!<Bit 3 */

#define  FSMC_BWTR4_ACCMOD                   0x30000000U        /*!<ACCMOD[1:0] bits (Access mode) */
#define  FSMC_BWTR4_ACCMOD_0                 0x10000000U        /*!<Bit 0 */
#define  FSMC_BWTR4_ACCMOD_1                 0x20000000U        /*!<Bit 1 */

/******************  Bit definition for FSMC_PCR2 register  *******************/
#define  FSMC_PCR2_PWAITEN                   0x00000002U        /*!<Wait feature enable bit */
#define  FSMC_PCR2_PBKEN                     0x00000004U        /*!<PC Card/NAND Flash memory bank enable bit */
#define  FSMC_PCR2_PTYP                      0x00000008U        /*!<Memory type */

#define  FSMC_PCR2_PWID                      0x00000030U        /*!<PWID[1:0] bits (NAND Flash databus width) */
#define  FSMC_PCR2_PWID_0                    0x00000010U        /*!<Bit 0 */
#define  FSMC_PCR2_PWID_1                    0x00000020U        /*!<Bit 1 */

#define  FSMC_PCR2_ECCEN                     0x00000040U        /*!<ECC computation logic enable bit */

#define  FSMC_PCR2_TCLR                      0x00001E00U        /*!<TCLR[3:0] bits (CLE to RE delay) */
#define  FSMC_PCR2_TCLR_0                    0x00000200U        /*!<Bit 0 */
#define  FSMC_PCR2_TCLR_1                    0x00000400U        /*!<Bit 1 */
#define  FSMC_PCR2_TCLR_2                    0x00000800U        /*!<Bit 2 */
#define  FSMC_PCR2_TCLR_3                    0x00001000U        /*!<Bit 3 */

#define  FSMC_PCR2_TAR                       0x0001E000U        /*!<TAR[3:0] bits (ALE to RE delay) */
#define  FSMC_PCR2_TAR_0                     0x00002000U        /*!<Bit 0 */
#define  FSMC_PCR2_TAR_1                     0x00004000U        /*!<Bit 1 */
#define  FSMC_PCR2_TAR_2                     0x00008000U        /*!<Bit 2 */
#define  FSMC_PCR2_TAR_3                     0x00010000U        /*!<Bit 3 */

#define  FSMC_PCR2_ECCPS                     0x000E0000U        /*!<ECCPS[1:0] bits (ECC page size) */
#define  FSMC_PCR2_ECCPS_0                   0x00020000U        /*!<Bit 0 */
#define  FSMC_PCR2_ECCPS_1                   0x00040000U        /*!<Bit 1 */
#define  FSMC_PCR2_ECCPS_2                   0x00080000U        /*!<Bit 2 */

/******************  Bit definition for FSMC_PCR3 register  *******************/
#define  FSMC_PCR3_PWAITEN                   0x00000002U        /*!<Wait feature enable bit */
#define  FSMC_PCR3_PBKEN                     0x00000004U        /*!<PC Card/NAND Flash memory bank enable bit */
#define  FSMC_PCR3_PTYP                      0x00000008U        /*!<Memory type */

#define  FSMC_PCR3_PWID                      0x00000030U        /*!<PWID[1:0] bits (NAND Flash databus width) */
#define  FSMC_PCR3_PWID_0                    0x00000010U        /*!<Bit 0 */
#define  FSMC_PCR3_PWID_1                    0x00000020U        /*!<Bit 1 */

#define  FSMC_PCR3_ECCEN                     0x00000040U        /*!<ECC computation logic enable bit */

#define  FSMC_PCR3_TCLR                      0x00001E00U        /*!<TCLR[3:0] bits (CLE to RE delay) */
#define  FSMC_PCR3_TCLR_0                    0x00000200U        /*!<Bit 0 */
#define  FSMC_PCR3_TCLR_1                    0x00000400U        /*!<Bit 1 */
#define  FSMC_PCR3_TCLR_2                    0x00000800U        /*!<Bit 2 */
#define  FSMC_PCR3_TCLR_3                    0x00001000U        /*!<Bit 3 */

#define  FSMC_PCR3_TAR                       0x0001E000U        /*!<TAR[3:0] bits (ALE to RE delay) */
#define  FSMC_PCR3_TAR_0                     0x00002000U        /*!<Bit 0 */
#define  FSMC_PCR3_TAR_1                     0x00004000U        /*!<Bit 1 */
#define  FSMC_PCR3_TAR_2                     0x00008000U        /*!<Bit 2 */
#define  FSMC_PCR3_TAR_3                     0x00010000U        /*!<Bit 3 */

#define  FSMC_PCR3_ECCPS                     0x000E0000U        /*!<ECCPS[2:0] bits (ECC page size) */
#define  FSMC_PCR3_ECCPS_0                   0x00020000U        /*!<Bit 0 */
#define  FSMC_PCR3_ECCPS_1                   0x00040000U        /*!<Bit 1 */
#define  FSMC_PCR3_ECCPS_2                   0x00080000U        /*!<Bit 2 */

/******************  Bit definition for FSMC_PCR4 register  *******************/
#define  FSMC_PCR4_PWAITEN                   0x00000002U        /*!<Wait feature enable bit */
#define  FSMC_PCR4_PBKEN                     0x00000004U        /*!<PC Card/NAND Flash memory bank enable bit */
#define  FSMC_PCR4_PTYP                      0x00000008U        /*!<Memory type */

#define  FSMC_PCR4_PWID                      0x00000030U        /*!<PWID[1:0] bits (NAND Flash databus width) */
#define  FSMC_PCR4_PWID_0                    0x00000010U        /*!<Bit 0 */
#define  FSMC_PCR4_PWID_1                    0x00000020U        /*!<Bit 1 */

#define  FSMC_PCR4_ECCEN                     0x00000040U        /*!<ECC computation logic enable bit */

#define  FSMC_PCR4_TCLR                      0x00001E00U        /*!<TCLR[3:0] bits (CLE to RE delay) */
#define  FSMC_PCR4_TCLR_0                    0x00000200U        /*!<Bit 0 */
#define  FSMC_PCR4_TCLR_1                    0x00000400U        /*!<Bit 1 */
#define  FSMC_PCR4_TCLR_2                    0x00000800U        /*!<Bit 2 */
#define  FSMC_PCR4_TCLR_3                    0x00001000U        /*!<Bit 3 */

#define  FSMC_PCR4_TAR                       0x0001E000U        /*!<TAR[3:0] bits (ALE to RE delay) */
#define  FSMC_PCR4_TAR_0                     0x00002000U        /*!<Bit 0 */
#define  FSMC_PCR4_TAR_1                     0x00004000U        /*!<Bit 1 */
#define  FSMC_PCR4_TAR_2                     0x00008000U        /*!<Bit 2 */
#define  FSMC_PCR4_TAR_3                     0x00010000U        /*!<Bit 3 */

#define  FSMC_PCR4_ECCPS                     0x000E0000U        /*!<ECCPS[2:0] bits (ECC page size) */
#define  FSMC_PCR4_ECCPS_0                   0x00020000U        /*!<Bit 0 */
#define  FSMC_PCR4_ECCPS_1                   0x00040000U        /*!<Bit 1 */
#define  FSMC_PCR4_ECCPS_2                   0x00080000U        /*!<Bit 2 */

/*******************  Bit definition for FSMC_SR2 register  *******************/
#define  FSMC_SR2_IRS                        0x00000001U        /*!<Interrupt Rising Edge status */
#define  FSMC_SR2_ILS                        0x00000002U        /*!<Interrupt Level status */
#define  FSMC_SR2_IFS                        0x00000004U        /*!<Interrupt Falling Edge status */
#define  FSMC_SR2_IREN                       0x00000008U        /*!<Interrupt Rising Edge detection Enable bit */
#define  FSMC_SR2_ILEN                       0x00000010U        /*!<Interrupt Level detection Enable bit */
#define  FSMC_SR2_IFEN                       0x00000020U        /*!<Interrupt Falling Edge detection Enable bit */
#define  FSMC_SR2_FEMPT                      0x00000040U        /*!<FIFO empty */

/*******************  Bit definition for FSMC_SR3 register  *******************/
#define  FSMC_SR3_IRS                        0x00000001U        /*!<Interrupt Rising Edge status */
#define  FSMC_SR3_ILS                        0x00000002U        /*!<Interrupt Level status */
#define  FSMC_SR3_IFS                        0x00000004U        /*!<Interrupt Falling Edge status */
#define  FSMC_SR3_IREN                       0x00000008U        /*!<Interrupt Rising Edge detection Enable bit */
#define  FSMC_SR3_ILEN                       0x00000010U        /*!<Interrupt Level detection Enable bit */
#define  FSMC_SR3_IFEN                       0x00000020U        /*!<Interrupt Falling Edge detection Enable bit */
#define  FSMC_SR3_FEMPT                      0x00000040U        /*!<FIFO empty */

/*******************  Bit definition for FSMC_SR4 register  *******************/
#define  FSMC_SR4_IRS                        0x00000001U        /*!<Interrupt Rising Edge status */
#define  FSMC_SR4_ILS                        0x00000002U        /*!<Interrupt Level status */
#define  FSMC_SR4_IFS                        0x00000004U        /*!<Interrupt Falling Edge status */
#define  FSMC_SR4_IREN                       0x00000008U        /*!<Interrupt Rising Edge detection Enable bit */
#define  FSMC_SR4_ILEN                       0x00000010U        /*!<Interrupt Level detection Enable bit */
#define  FSMC_SR4_IFEN                       0x00000020U        /*!<Interrupt Falling Edge detection Enable bit */
#define  FSMC_SR4_FEMPT                      0x00000040U        /*!<FIFO empty */

/******************  Bit definition for FSMC_PMEM2 register  ******************/
#define  FSMC_PMEM2_MEMSET2                  0x000000FFU        /*!<MEMSET2[7:0] bits (Common memory 2 setup time) */
#define  FSMC_PMEM2_MEMSET2_0                0x00000001U        /*!<Bit 0 */
#define  FSMC_PMEM2_MEMSET2_1                0x00000002U        /*!<Bit 1 */
#define  FSMC_PMEM2_MEMSET2_2                0x00000004U        /*!<Bit 2 */
#define  FSMC_PMEM2_MEMSET2_3                0x00000008U        /*!<Bit 3 */
#define  FSMC_PMEM2_MEMSET2_4                0x00000010U        /*!<Bit 4 */
#define  FSMC_PMEM2_MEMSET2_5                0x00000020U        /*!<Bit 5 */
#define  FSMC_PMEM2_MEMSET2_6                0x00000040U        /*!<Bit 6 */
#define  FSMC_PMEM2_MEMSET2_7                0x00000080U        /*!<Bit 7 */

#define  FSMC_PMEM2_MEMWAIT2                 0x0000FF00U        /*!<MEMWAIT2[7:0] bits (Common memory 2 wait time) */
#define  FSMC_PMEM2_MEMWAIT2_0               0x00000100U        /*!<Bit 0 */
#define  FSMC_PMEM2_MEMWAIT2_1               0x00000200U        /*!<Bit 1 */
#define  FSMC_PMEM2_MEMWAIT2_2               0x00000400U        /*!<Bit 2 */
#define  FSMC_PMEM2_MEMWAIT2_3               0x00000800U        /*!<Bit 3 */
#define  FSMC_PMEM2_MEMWAIT2_4               0x00001000U        /*!<Bit 4 */
#define  FSMC_PMEM2_MEMWAIT2_5               0x00002000U        /*!<Bit 5 */
#define  FSMC_PMEM2_MEMWAIT2_6               0x00004000U        /*!<Bit 6 */
#define  FSMC_PMEM2_MEMWAIT2_7               0x00008000U        /*!<Bit 7 */

#define  FSMC_PMEM2_MEMHOLD2                 0x00FF0000U        /*!<MEMHOLD2[7:0] bits (Common memory 2 hold time) */
#define  FSMC_PMEM2_MEMHOLD2_0               0x00010000U        /*!<Bit 0 */
#define  FSMC_PMEM2_MEMHOLD2_1               0x00020000U        /*!<Bit 1 */
#define  FSMC_PMEM2_MEMHOLD2_2               0x00040000U        /*!<Bit 2 */
#define  FSMC_PMEM2_MEMHOLD2_3               0x00080000U        /*!<Bit 3 */
#define  FSMC_PMEM2_MEMHOLD2_4               0x00100000U        /*!<Bit 4 */
#define  FSMC_PMEM2_MEMHOLD2_5               0x00200000U        /*!<Bit 5 */
#define  FSMC_PMEM2_MEMHOLD2_6               0x00400000U        /*!<Bit 6 */
#define  FSMC_PMEM2_MEMHOLD2_7               0x00800000U        /*!<Bit 7 */

#define  FSMC_PMEM2_MEMHIZ2                  0xFF000000U        /*!<MEMHIZ2[7:0] bits (Common memory 2 databus HiZ time) */
#define  FSMC_PMEM2_MEMHIZ2_0                0x01000000U        /*!<Bit 0 */
#define  FSMC_PMEM2_MEMHIZ2_1                0x02000000U        /*!<Bit 1 */
#define  FSMC_PMEM2_MEMHIZ2_2                0x04000000U        /*!<Bit 2 */
#define  FSMC_PMEM2_MEMHIZ2_3                0x08000000U        /*!<Bit 3 */
#define  FSMC_PMEM2_MEMHIZ2_4                0x10000000U        /*!<Bit 4 */
#define  FSMC_PMEM2_MEMHIZ2_5                0x20000000U        /*!<Bit 5 */
#define  FSMC_PMEM2_MEMHIZ2_6                0x40000000U        /*!<Bit 6 */
#define  FSMC_PMEM2_MEMHIZ2_7                0x80000000U        /*!<Bit 7 */

/******************  Bit definition for FSMC_PMEM3 register  ******************/
#define  FSMC_PMEM3_MEMSET3                  0x000000FFU        /*!<MEMSET3[7:0] bits (Common memory 3 setup time) */
#define  FSMC_PMEM3_MEMSET3_0                0x00000001U        /*!<Bit 0 */
#define  FSMC_PMEM3_MEMSET3_1                0x00000002U        /*!<Bit 1 */
#define  FSMC_PMEM3_MEMSET3_2                0x00000004U        /*!<Bit 2 */
#define  FSMC_PMEM3_MEMSET3_3                0x00000008U        /*!<Bit 3 */
#define  FSMC_PMEM3_MEMSET3_4                0x00000010U        /*!<Bit 4 */
#define  FSMC_PMEM3_MEMSET3_5                0x00000020U        /*!<Bit 5 */
#define  FSMC_PMEM3_MEMSET3_6                0x00000040U        /*!<Bit 6 */
#define  FSMC_PMEM3_MEMSET3_7                0x00000080U        /*!<Bit 7 */

#define  FSMC_PMEM3_MEMWAIT3                 0x0000FF00U        /*!<MEMWAIT3[7:0] bits (Common memory 3 wait time) */
#define  FSMC_PMEM3_MEMWAIT3_0               0x00000100U        /*!<Bit 0 */
#define  FSMC_PMEM3_MEMWAIT3_1               0x00000200U        /*!<Bit 1 */
#define  FSMC_PMEM3_MEMWAIT3_2               0x00000400U        /*!<Bit 2 */
#define  FSMC_PMEM3_MEMWAIT3_3               0x00000800U        /*!<Bit 3 */
#define  FSMC_PMEM3_MEMWAIT3_4               0x00001000U        /*!<Bit 4 */
#define  FSMC_PMEM3_MEMWAIT3_5               0x00002000U        /*!<Bit 5 */
#define  FSMC_PMEM3_MEMWAIT3_6               0x00004000U        /*!<Bit 6 */
#define  FSMC_PMEM3_MEMWAIT3_7               0x00008000U        /*!<Bit 7 */

#define  FSMC_PMEM3_MEMHOLD3                 0x00FF0000U        /*!<MEMHOLD3[7:0] bits (Common memory 3 hold time) */
#define  FSMC_PMEM3_MEMHOLD3_0               0x00010000U        /*!<Bit 0 */
#define  FSMC_PMEM3_MEMHOLD3_1               0x00020000U        /*!<Bit 1 */
#define  FSMC_PMEM3_MEMHOLD3_2               0x00040000U        /*!<Bit 2 */
#define  FSMC_PMEM3_MEMHOLD3_3               0x00080000U        /*!<Bit 3 */
#define  FSMC_PMEM3_MEMHOLD3_4               0x00100000U        /*!<Bit 4 */
#define  FSMC_PMEM3_MEMHOLD3_5               0x00200000U        /*!<Bit 5 */
#define  FSMC_PMEM3_MEMHOLD3_6               0x00400000U        /*!<Bit 6 */
#define  FSMC_PMEM3_MEMHOLD3_7               0x00800000U        /*!<Bit 7 */

#define  FSMC_PMEM3_MEMHIZ3                  0xFF000000U        /*!<MEMHIZ3[7:0] bits (Common memory 3 databus HiZ time) */
#define  FSMC_PMEM3_MEMHIZ3_0                0x01000000U        /*!<Bit 0 */
#define  FSMC_PMEM3_MEMHIZ3_1                0x02000000U        /*!<Bit 1 */
#define  FSMC_PMEM3_MEMHIZ3_2                0x04000000U        /*!<Bit 2 */
#define  FSMC_PMEM3_MEMHIZ3_3                0x08000000U        /*!<Bit 3 */
#define  FSMC_PMEM3_MEMHIZ3_4                0x10000000U        /*!<Bit 4 */
#define  FSMC_PMEM3_MEMHIZ3_5                0x20000000U        /*!<Bit 5 */
#define  FSMC_PMEM3_MEMHIZ3_6                0x40000000U        /*!<Bit 6 */
#define  FSMC_PMEM3_MEMHIZ3_7                0x80000000U        /*!<Bit 7 */

/******************  Bit definition for FSMC_PMEM4 register  ******************/
#define  FSMC_PMEM4_MEMSET4                  0x000000FFU        /*!<MEMSET4[7:0] bits (Common memory 4 setup time) */
#define  FSMC_PMEM4_MEMSET4_0                0x00000001U        /*!<Bit 0 */
#define  FSMC_PMEM4_MEMSET4_1                0x00000002U        /*!<Bit 1 */
#define  FSMC_PMEM4_MEMSET4_2                0x00000004U        /*!<Bit 2 */
#define  FSMC_PMEM4_MEMSET4_3                0x00000008U        /*!<Bit 3 */
#define  FSMC_PMEM4_MEMSET4_4                0x00000010U        /*!<Bit 4 */
#define  FSMC_PMEM4_MEMSET4_5                0x00000020U        /*!<Bit 5 */
#define  FSMC_PMEM4_MEMSET4_6                0x00000040U        /*!<Bit 6 */
#define  FSMC_PMEM4_MEMSET4_7                0x00000080U        /*!<Bit 7 */

#define  FSMC_PMEM4_MEMWAIT4                 0x0000FF00U        /*!<MEMWAIT4[7:0] bits (Common memory 4 wait time) */
#define  FSMC_PMEM4_MEMWAIT4_0               0x00000100U        /*!<Bit 0 */
#define  FSMC_PMEM4_MEMWAIT4_1               0x00000200U        /*!<Bit 1 */
#define  FSMC_PMEM4_MEMWAIT4_2               0x00000400U        /*!<Bit 2 */
#define  FSMC_PMEM4_MEMWAIT4_3               0x00000800U        /*!<Bit 3 */
#define  FSMC_PMEM4_MEMWAIT4_4               0x00001000U        /*!<Bit 4 */
#define  FSMC_PMEM4_MEMWAIT4_5               0x00002000U        /*!<Bit 5 */
#define  FSMC_PMEM4_MEMWAIT4_6               0x00004000U        /*!<Bit 6 */
#define  FSMC_PMEM4_MEMWAIT4_7               0x00008000U        /*!<Bit 7 */

#define  FSMC_PMEM4_MEMHOLD4                 0x00FF0000U        /*!<MEMHOLD4[7:0] bits (Common memory 4 hold time) */
#define  FSMC_PMEM4_MEMHOLD4_0               0x00010000U        /*!<Bit 0 */
#define  FSMC_PMEM4_MEMHOLD4_1               0x00020000U        /*!<Bit 1 */
#define  FSMC_PMEM4_MEMHOLD4_2               0x00040000U        /*!<Bit 2 */
#define  FSMC_PMEM4_MEMHOLD4_3               0x00080000U        /*!<Bit 3 */
#define  FSMC_PMEM4_MEMHOLD4_4               0x00100000U        /*!<Bit 4 */
#define  FSMC_PMEM4_MEMHOLD4_5               0x00200000U        /*!<Bit 5 */
#define  FSMC_PMEM4_MEMHOLD4_6               0x00400000U        /*!<Bit 6 */
#define  FSMC_PMEM4_MEMHOLD4_7               0x00800000U        /*!<Bit 7 */

#define  FSMC_PMEM4_MEMHIZ4                  0xFF000000U        /*!<MEMHIZ4[7:0] bits (Common memory 4 databus HiZ time) */
#define  FSMC_PMEM4_MEMHIZ4_0                0x01000000U        /*!<Bit 0 */
#define  FSMC_PMEM4_MEMHIZ4_1                0x02000000U        /*!<Bit 1 */
#define  FSMC_PMEM4_MEMHIZ4_2                0x04000000U        /*!<Bit 2 */
#define  FSMC_PMEM4_MEMHIZ4_3                0x08000000U        /*!<Bit 3 */
#define  FSMC_PMEM4_MEMHIZ4_4                0x10000000U        /*!<Bit 4 */
#define  FSMC_PMEM4_MEMHIZ4_5                0x20000000U        /*!<Bit 5 */
#define  FSMC_PMEM4_MEMHIZ4_6                0x40000000U        /*!<Bit 6 */
#define  FSMC_PMEM4_MEMHIZ4_7                0x80000000U        /*!<Bit 7 */

/******************  Bit definition for FSMC_PATT2 register  ******************/
#define  FSMC_PATT2_ATTSET2                  0x000000FFU        /*!<ATTSET2[7:0] bits (Attribute memory 2 setup time) */
#define  FSMC_PATT2_ATTSET2_0                0x00000001U        /*!<Bit 0 */
#define  FSMC_PATT2_ATTSET2_1                0x00000002U        /*!<Bit 1 */
#define  FSMC_PATT2_ATTSET2_2                0x00000004U        /*!<Bit 2 */
#define  FSMC_PATT2_ATTSET2_3                0x00000008U        /*!<Bit 3 */
#define  FSMC_PATT2_ATTSET2_4                0x00000010U        /*!<Bit 4 */
#define  FSMC_PATT2_ATTSET2_5                0x00000020U        /*!<Bit 5 */
#define  FSMC_PATT2_ATTSET2_6                0x00000040U        /*!<Bit 6 */
#define  FSMC_PATT2_ATTSET2_7                0x00000080U        /*!<Bit 7 */

#define  FSMC_PATT2_ATTWAIT2                 0x0000FF00U        /*!<ATTWAIT2[7:0] bits (Attribute memory 2 wait time) */
#define  FSMC_PATT2_ATTWAIT2_0               0x00000100U        /*!<Bit 0 */
#define  FSMC_PATT2_ATTWAIT2_1               0x00000200U        /*!<Bit 1 */
#define  FSMC_PATT2_ATTWAIT2_2               0x00000400U        /*!<Bit 2 */
#define  FSMC_PATT2_ATTWAIT2_3               0x00000800U        /*!<Bit 3 */
#define  FSMC_PATT2_ATTWAIT2_4               0x00001000U        /*!<Bit 4 */
#define  FSMC_PATT2_ATTWAIT2_5               0x00002000U        /*!<Bit 5 */
#define  FSMC_PATT2_ATTWAIT2_6               0x00004000U        /*!<Bit 6 */
#define  FSMC_PATT2_ATTWAIT2_7               0x00008000U        /*!<Bit 7 */

#define  FSMC_PATT2_ATTHOLD2                 0x00FF0000U        /*!<ATTHOLD2[7:0] bits (Attribute memory 2 hold time) */
#define  FSMC_PATT2_ATTHOLD2_0               0x00010000U        /*!<Bit 0 */
#define  FSMC_PATT2_ATTHOLD2_1               0x00020000U        /*!<Bit 1 */
#define  FSMC_PATT2_ATTHOLD2_2               0x00040000U        /*!<Bit 2 */
#define  FSMC_PATT2_ATTHOLD2_3               0x00080000U        /*!<Bit 3 */
#define  FSMC_PATT2_ATTHOLD2_4               0x00100000U        /*!<Bit 4 */
#define  FSMC_PATT2_ATTHOLD2_5               0x00200000U        /*!<Bit 5 */
#define  FSMC_PATT2_ATTHOLD2_6               0x00400000U        /*!<Bit 6 */
#define  FSMC_PATT2_ATTHOLD2_7               0x00800000U        /*!<Bit 7 */

#define  FSMC_PATT2_ATTHIZ2                  0xFF000000U        /*!<ATTHIZ2[7:0] bits (Attribute memory 2 databus HiZ time) */
#define  FSMC_PATT2_ATTHIZ2_0                0x01000000U        /*!<Bit 0 */
#define  FSMC_PATT2_ATTHIZ2_1                0x02000000U        /*!<Bit 1 */
#define  FSMC_PATT2_ATTHIZ2_2                0x04000000U        /*!<Bit 2 */
#define  FSMC_PATT2_ATTHIZ2_3                0x08000000U        /*!<Bit 3 */
#define  FSMC_PATT2_ATTHIZ2_4                0x10000000U        /*!<Bit 4 */
#define  FSMC_PATT2_ATTHIZ2_5                0x20000000U        /*!<Bit 5 */
#define  FSMC_PATT2_ATTHIZ2_6                0x40000000U        /*!<Bit 6 */
#define  FSMC_PATT2_ATTHIZ2_7                0x80000000U        /*!<Bit 7 */

/******************  Bit definition for FSMC_PATT3 register  ******************/
#define  FSMC_PATT3_ATTSET3                  0x000000FFU        /*!<ATTSET3[7:0] bits (Attribute memory 3 setup time) */
#define  FSMC_PATT3_ATTSET3_0                0x00000001U        /*!<Bit 0 */
#define  FSMC_PATT3_ATTSET3_1                0x00000002U        /*!<Bit 1 */
#define  FSMC_PATT3_ATTSET3_2                0x00000004U        /*!<Bit 2 */
#define  FSMC_PATT3_ATTSET3_3                0x00000008U        /*!<Bit 3 */
#define  FSMC_PATT3_ATTSET3_4                0x00000010U        /*!<Bit 4 */
#define  FSMC_PATT3_ATTSET3_5                0x00000020U        /*!<Bit 5 */
#define  FSMC_PATT3_ATTSET3_6                0x00000040U        /*!<Bit 6 */
#define  FSMC_PATT3_ATTSET3_7                0x00000080U        /*!<Bit 7 */

#define  FSMC_PATT3_ATTWAIT3                 0x0000FF00U        /*!<ATTWAIT3[7:0] bits (Attribute memory 3 wait time) */
#define  FSMC_PATT3_ATTWAIT3_0               0x00000100U        /*!<Bit 0 */
#define  FSMC_PATT3_ATTWAIT3_1               0x00000200U        /*!<Bit 1 */
#define  FSMC_PATT3_ATTWAIT3_2               0x00000400U        /*!<Bit 2 */
#define  FSMC_PATT3_ATTWAIT3_3               0x00000800U        /*!<Bit 3 */
#define  FSMC_PATT3_ATTWAIT3_4               0x00001000U        /*!<Bit 4 */
#define  FSMC_PATT3_ATTWAIT3_5               0x00002000U        /*!<Bit 5 */
#define  FSMC_PATT3_ATTWAIT3_6               0x00004000U        /*!<Bit 6 */
#define  FSMC_PATT3_ATTWAIT3_7               0x00008000U        /*!<Bit 7 */

#define  FSMC_PATT3_ATTHOLD3                 0x00FF0000U        /*!<ATTHOLD3[7:0] bits (Attribute memory 3 hold time) */
#define  FSMC_PATT3_ATTHOLD3_0               0x00010000U        /*!<Bit 0 */
#define  FSMC_PATT3_ATTHOLD3_1               0x00020000U        /*!<Bit 1 */
#define  FSMC_PATT3_ATTHOLD3_2               0x00040000U        /*!<Bit 2 */
#define  FSMC_PATT3_ATTHOLD3_3               0x00080000U        /*!<Bit 3 */
#define  FSMC_PATT3_ATTHOLD3_4               0x00100000U        /*!<Bit 4 */
#define  FSMC_PATT3_ATTHOLD3_5               0x00200000U        /*!<Bit 5 */
#define  FSMC_PATT3_ATTHOLD3_6               0x00400000U        /*!<Bit 6 */
#define  FSMC_PATT3_ATTHOLD3_7               0x00800000U        /*!<Bit 7 */

#define  FSMC_PATT3_ATTHIZ3                  0xFF000000U        /*!<ATTHIZ3[7:0] bits (Attribute memory 3 databus HiZ time) */
#define  FSMC_PATT3_ATTHIZ3_0                0x01000000U        /*!<Bit 0 */
#define  FSMC_PATT3_ATTHIZ3_1                0x02000000U        /*!<Bit 1 */
#define  FSMC_PATT3_ATTHIZ3_2                0x04000000U        /*!<Bit 2 */
#define  FSMC_PATT3_ATTHIZ3_3                0x08000000U        /*!<Bit 3 */
#define  FSMC_PATT3_ATTHIZ3_4                0x10000000U        /*!<Bit 4 */
#define  FSMC_PATT3_ATTHIZ3_5                0x20000000U        /*!<Bit 5 */
#define  FSMC_PATT3_ATTHIZ3_6                0x40000000U        /*!<Bit 6 */
#define  FSMC_PATT3_ATTHIZ3_7                0x80000000U        /*!<Bit 7 */

/******************  Bit definition for FSMC_PATT4 register  ******************/
#define  FSMC_PATT4_ATTSET4                  0x000000FFU        /*!<ATTSET4[7:0] bits (Attribute memory 4 setup time) */
#define  FSMC_PATT4_ATTSET4_0                0x00000001U        /*!<Bit 0 */
#define  FSMC_PATT4_ATTSET4_1                0x00000002U        /*!<Bit 1 */
#define  FSMC_PATT4_ATTSET4_2                0x00000004U        /*!<Bit 2 */
#define  FSMC_PATT4_ATTSET4_3                0x00000008U        /*!<Bit 3 */
#define  FSMC_PATT4_ATTSET4_4                0x00000010U        /*!<Bit 4 */
#define  FSMC_PATT4_ATTSET4_5                0x00000020U        /*!<Bit 5 */
#define  FSMC_PATT4_ATTSET4_6                0x00000040U        /*!<Bit 6 */
#define  FSMC_PATT4_ATTSET4_7                0x00000080U        /*!<Bit 7 */

#define  FSMC_PATT4_ATTWAIT4                 0x0000FF00U        /*!<ATTWAIT4[7:0] bits (Attribute memory 4 wait time) */
#define  FSMC_PATT4_ATTWAIT4_0               0x00000100U        /*!<Bit 0 */
#define  FSMC_PATT4_ATTWAIT4_1               0x00000200U        /*!<Bit 1 */
#define  FSMC_PATT4_ATTWAIT4_2               0x00000400U        /*!<Bit 2 */
#define  FSMC_PATT4_ATTWAIT4_3               0x00000800U        /*!<Bit 3 */
#define  FSMC_PATT4_ATTWAIT4_4               0x00001000U        /*!<Bit 4 */
#define  FSMC_PATT4_ATTWAIT4_5               0x00002000U        /*!<Bit 5 */
#define  FSMC_PATT4_ATTWAIT4_6               0x00004000U        /*!<Bit 6 */
#define  FSMC_PATT4_ATTWAIT4_7               0x00008000U        /*!<Bit 7 */

#define  FSMC_PATT4_ATTHOLD4                 0x00FF0000U        /*!<ATTHOLD4[7:0] bits (Attribute memory 4 hold time) */
#define  FSMC_PATT4_ATTHOLD4_0               0x00010000U        /*!<Bit 0 */
#define  FSMC_PATT4_ATTHOLD4_1               0x00020000U        /*!<Bit 1 */
#define  FSMC_PATT4_ATTHOLD4_2               0x00040000U        /*!<Bit 2 */
#define  FSMC_PATT4_ATTHOLD4_3               0x00080000U        /*!<Bit 3 */
#define  FSMC_PATT4_ATTHOLD4_4               0x00100000U        /*!<Bit 4 */
#define  FSMC_PATT4_ATTHOLD4_5               0x00200000U        /*!<Bit 5 */
#define  FSMC_PATT4_ATTHOLD4_6               0x00400000U        /*!<Bit 6 */
#define  FSMC_PATT4_ATTHOLD4_7               0x00800000U        /*!<Bit 7 */

#define  FSMC_PATT4_ATTHIZ4                  0xFF000000U        /*!<ATTHIZ4[7:0] bits (Attribute memory 4 databus HiZ time) */
#define  FSMC_PATT4_ATTHIZ4_0                0x01000000U        /*!<Bit 0 */
#define  FSMC_PATT4_ATTHIZ4_1                0x02000000U        /*!<Bit 1 */
#define  FSMC_PATT4_ATTHIZ4_2                0x04000000U        /*!<Bit 2 */
#define  FSMC_PATT4_ATTHIZ4_3                0x08000000U        /*!<Bit 3 */
#define  FSMC_PATT4_ATTHIZ4_4                0x10000000U        /*!<Bit 4 */
#define  FSMC_PATT4_ATTHIZ4_5                0x20000000U        /*!<Bit 5 */
#define  FSMC_PATT4_ATTHIZ4_6                0x40000000U        /*!<Bit 6 */
#define  FSMC_PATT4_ATTHIZ4_7                0x80000000U        /*!<Bit 7 */

/******************  Bit definition for FSMC_PIO4 register  *******************/
#define  FSMC_PIO4_IOSET4                    0x000000FFU        /*!<IOSET4[7:0] bits (I/O 4 setup time) */
#define  FSMC_PIO4_IOSET4_0                  0x00000001U        /*!<Bit 0 */
#define  FSMC_PIO4_IOSET4_1                  0x00000002U        /*!<Bit 1 */
#define  FSMC_PIO4_IOSET4_2                  0x00000004U        /*!<Bit 2 */
#define  FSMC_PIO4_IOSET4_3                  0x00000008U        /*!<Bit 3 */
#define  FSMC_PIO4_IOSET4_4                  0x00000010U        /*!<Bit 4 */
#define  FSMC_PIO4_IOSET4_5                  0x00000020U        /*!<Bit 5 */
#define  FSMC_PIO4_IOSET4_6                  0x00000040U        /*!<Bit 6 */
#define  FSMC_PIO4_IOSET4_7                  0x00000080U        /*!<Bit 7 */

#define  FSMC_PIO4_IOWAIT4                   0x0000FF00U        /*!<IOWAIT4[7:0] bits (I/O 4 wait time) */
#define  FSMC_PIO4_IOWAIT4_0                 0x00000100U        /*!<Bit 0 */
#define  FSMC_PIO4_IOWAIT4_1                 0x00000200U        /*!<Bit 1 */
#define  FSMC_PIO4_IOWAIT4_2                 0x00000400U        /*!<Bit 2 */
#define  FSMC_PIO4_IOWAIT4_3                 0x00000800U        /*!<Bit 3 */
#define  FSMC_PIO4_IOWAIT4_4                 0x00001000U        /*!<Bit 4 */
#define  FSMC_PIO4_IOWAIT4_5                 0x00002000U        /*!<Bit 5 */
#define  FSMC_PIO4_IOWAIT4_6                 0x00004000U        /*!<Bit 6 */
#define  FSMC_PIO4_IOWAIT4_7                 0x00008000U        /*!<Bit 7 */

#define  FSMC_PIO4_IOHOLD4                   0x00FF0000U        /*!<IOHOLD4[7:0] bits (I/O 4 hold time) */
#define  FSMC_PIO4_IOHOLD4_0                 0x00010000U        /*!<Bit 0 */
#define  FSMC_PIO4_IOHOLD4_1                 0x00020000U        /*!<Bit 1 */
#define  FSMC_PIO4_IOHOLD4_2                 0x00040000U        /*!<Bit 2 */
#define  FSMC_PIO4_IOHOLD4_3                 0x00080000U        /*!<Bit 3 */
#define  FSMC_PIO4_IOHOLD4_4                 0x00100000U        /*!<Bit 4 */
#define  FSMC_PIO4_IOHOLD4_5                 0x00200000U        /*!<Bit 5 */
#define  FSMC_PIO4_IOHOLD4_6                 0x00400000U        /*!<Bit 6 */
#define  FSMC_PIO4_IOHOLD4_7                 0x00800000U        /*!<Bit 7 */

#define  FSMC_PIO4_IOHIZ4                    0xFF000000U        /*!<IOHIZ4[7:0] bits (I/O 4 databus HiZ time) */
#define  FSMC_PIO4_IOHIZ4_0                  0x01000000U        /*!<Bit 0 */
#define  FSMC_PIO4_IOHIZ4_1                  0x02000000U        /*!<Bit 1 */
#define  FSMC_PIO4_IOHIZ4_2                  0x04000000U        /*!<Bit 2 */
#define  FSMC_PIO4_IOHIZ4_3                  0x08000000U        /*!<Bit 3 */
#define  FSMC_PIO4_IOHIZ4_4                  0x10000000U        /*!<Bit 4 */
#define  FSMC_PIO4_IOHIZ4_5                  0x20000000U        /*!<Bit 5 */
#define  FSMC_PIO4_IOHIZ4_6                  0x40000000U        /*!<Bit 6 */
#define  FSMC_PIO4_IOHIZ4_7                  0x80000000U        /*!<Bit 7 */

/******************  Bit definition for FSMC_ECCR2 register  ******************/
#define  FSMC_ECCR2_ECC2                     0xFFFFFFFFU        /*!<ECC result */

/******************  Bit definition for FSMC_ECCR3 register  ******************/
#define  FSMC_ECCR3_ECC3                     0xFFFFFFFFU        /*!<ECC result */

/******************************************************************************/
/*                                                                            */
/*                            General Purpose I/O                             */
/*                                                                            */
/******************************************************************************/
/******************  Bits definition for GPIO_MODER register  *****************/
#define GPIO_MODER_MODE0                    ((uint32_t)0x00000003U)
#define GPIO_MODER_MODE0_0                  ((uint32_t)0x00000001U)
#define GPIO_MODER_MODE0_1                  ((uint32_t)0x00000002U)
#define GPIO_MODER_MODE1                    ((uint32_t)0x0000000CU)
#define GPIO_MODER_MODE1_0                  ((uint32_t)0x00000004U)
#define GPIO_MODER_MODE1_1                  ((uint32_t)0x00000008U)
#define GPIO_MODER_MODE2                    ((uint32_t)0x00000030U)
#define GPIO_MODER_MODE2_0                  ((uint32_t)0x00000010U)
#define GPIO_MODER_MODE2_1                  ((uint32_t)0x00000020U)
#define GPIO_MODER_MODE3                    ((uint32_t)0x000000C0U)
#define GPIO_MODER_MODE3_0                  ((uint32_t)0x00000040U)
#define GPIO_MODER_MODE3_1                  ((uint32_t)0x00000080U)
#define GPIO_MODER_MODE4                    ((uint32_t)0x00000300U)
#define GPIO_MODER_MODE4_0                  ((uint32_t)0x00000100U)
#define GPIO_MODER_MODE4_1                  ((uint32_t)0x00000200U)
#define GPIO_MODER_MODE5                    ((uint32_t)0x00000C00U)
#define GPIO_MODER_MODE5_0                  ((uint32_t)0x00000400U)
#define GPIO_MODER_MODE5_1                  ((uint32_t)0x00000800U)
#define GPIO_MODER_MODE6                    ((uint32_t)0x00003000U)
#define GPIO_MODER_MODE6_0                  ((uint32_t)0x00001000U)
#define GPIO_MODER_MODE6_1                  ((uint32_t)0x00002000U)
#define GPIO_MODER_MODE7                    ((uint32_t)0x0000C000U)
#define GPIO_MODER_MODE7_0                  ((uint32_t)0x00004000U)
#define GPIO_MODER_MODE7_1                  ((uint32_t)0x00008000U)
#define GPIO_MODER_MODE8                    ((uint32_t)0x00030000U)
#define GPIO_MODER_MODE8_0                  ((uint32_t)0x00010000U)
#define GPIO_MODER_MODE8_1                  ((uint32_t)0x00020000U)
#define GPIO_MODER_MODE9                    ((uint32_t)0x000C0000U)
#define GPIO_MODER_MODE9_0                  ((uint32_t)0x00040000U)
#define GPIO_MODER_MODE9_1                  ((uint32_t)0x00080000U)
#define GPIO_MODER_MODE10                   ((uint32_t)0x00300000U)
#define GPIO_MODER_MODE10_0                 ((uint32_t)0x00100000U)
#define GPIO_MODER_MODE10_1                 ((uint32_t)0x00200000U)
#define GPIO_MODER_MODE11                   ((uint32_t)0x00C00000U)
#define GPIO_MODER_MODE11_0                 ((uint32_t)0x00400000U)
#define GPIO_MODER_MODE11_1                 ((uint32_t)0x00800000U)
#define GPIO_MODER_MODE12                   ((uint32_t)0x03000000U)
#define GPIO_MODER_MODE12_0                 ((uint32_t)0x01000000U)
#define GPIO_MODER_MODE12_1                 ((uint32_t)0x02000000U)
#define GPIO_MODER_MODE13                   ((uint32_t)0x0C000000U)
#define GPIO_MODER_MODE13_0                 ((uint32_t)0x04000000U)
#define GPIO_MODER_MODE13_1                 ((uint32_t)0x08000000U)
#define GPIO_MODER_MODE14                   ((uint32_t)0x30000000U)
#define GPIO_MODER_MODE14_0                 ((uint32_t)0x10000000U)
#define GPIO_MODER_MODE14_1                 ((uint32_t)0x20000000U)
#define GPIO_MODER_MODE15                   ((uint32_t)0xC0000000U)
#define GPIO_MODER_MODE15_0                 ((uint32_t)0x40000000U)
#define GPIO_MODER_MODE15_1                 ((uint32_t)0x80000000U)
/* Legacy defines */
#define GPIO_MODER_MODER0                    0x00000003U
#define GPIO_MODER_MODER0_0                  0x00000001U
#define GPIO_MODER_MODER0_1                  0x00000002U
#define GPIO_MODER_MODER1                    0x0000000CU
#define GPIO_MODER_MODER1_0                  0x00000004U
#define GPIO_MODER_MODER1_1                  0x00000008U
#define GPIO_MODER_MODER2                    0x00000030U
#define GPIO_MODER_MODER2_0                  0x00000010U
#define GPIO_MODER_MODER2_1                  0x00000020U
#define GPIO_MODER_MODER3                    0x000000C0U
#define GPIO_MODER_MODER3_0                  0x00000040U
#define GPIO_MODER_MODER3_1                  0x00000080U
#define GPIO_MODER_MODER4                    0x00000300U
#define GPIO_MODER_MODER4_0                  0x00000100U
#define GPIO_MODER_MODER4_1                  0x00000200U
#define GPIO_MODER_MODER5                    0x00000C00U
#define GPIO_MODER_MODER5_0                  0x00000400U
#define GPIO_MODER_MODER5_1                  0x00000800U
#define GPIO_MODER_MODER6                    0x00003000U
#define GPIO_MODER_MODER6_0                  0x00001000U
#define GPIO_MODER_MODER6_1                  0x00002000U
#define GPIO_MODER_MODER7                    0x0000C000U
#define GPIO_MODER_MODER7_0                  0x00004000U
#define GPIO_MODER_MODER7_1                  0x00008000U
#define GPIO_MODER_MODER8                    0x00030000U
#define GPIO_MODER_MODER8_0                  0x00010000U
#define GPIO_MODER_MODER8_1                  0x00020000U
#define GPIO_MODER_MODER9                    0x000C0000U
#define GPIO_MODER_MODER9_0                  0x00040000U
#define GPIO_MODER_MODER9_1                  0x00080000U
#define GPIO_MODER_MODER10                   0x00300000U
#define GPIO_MODER_MODER10_0                 0x00100000U
#define GPIO_MODER_MODER10_1                 0x00200000U
#define GPIO_MODER_MODER11                   0x00C00000U
#define GPIO_MODER_MODER11_0                 0x00400000U
#define GPIO_MODER_MODER11_1                 0x00800000U
#define GPIO_MODER_MODER12                   0x03000000U
#define GPIO_MODER_MODER12_0                 0x01000000U
#define GPIO_MODER_MODER12_1                 0x02000000U
#define GPIO_MODER_MODER13                   0x0C000000U
#define GPIO_MODER_MODER13_0                 0x04000000U
#define GPIO_MODER_MODER13_1                 0x08000000U
#define GPIO_MODER_MODER14                   0x30000000U
#define GPIO_MODER_MODER14_0                 0x10000000U
#define GPIO_MODER_MODER14_1                 0x20000000U
#define GPIO_MODER_MODER15                   0xC0000000U
#define GPIO_MODER_MODER15_0                 0x40000000U
#define GPIO_MODER_MODER15_1                 0x80000000U

/******************  Bits definition for GPIO_OTYPER register  ****************/
#define GPIO_OTYPER_OT0                     ((uint32_t)0x00000001U)
#define GPIO_OTYPER_OT1                     ((uint32_t)0x00000002U)
#define GPIO_OTYPER_OT2                     ((uint32_t)0x00000004U)
#define GPIO_OTYPER_OT3                     ((uint32_t)0x00000008U)
#define GPIO_OTYPER_OT4                     ((uint32_t)0x00000010U)
#define GPIO_OTYPER_OT5                     ((uint32_t)0x00000020U)
#define GPIO_OTYPER_OT6                     ((uint32_t)0x00000040U)
#define GPIO_OTYPER_OT7                     ((uint32_t)0x00000080U)
#define GPIO_OTYPER_OT8                     ((uint32_t)0x00000100U)
#define GPIO_OTYPER_OT9                     ((uint32_t)0x00000200U)
#define GPIO_OTYPER_OT10                    ((uint32_t)0x00000400U)
#define GPIO_OTYPER_OT11                    ((uint32_t)0x00000800U)
#define GPIO_OTYPER_OT12                    ((uint32_t)0x00001000U)
#define GPIO_OTYPER_OT13                    ((uint32_t)0x00002000U)
#define GPIO_OTYPER_OT14                    ((uint32_t)0x00004000U)
#define GPIO_OTYPER_OT15                    ((uint32_t)0x00008000U)

/* Legacy defines */
#define GPIO_OTYPER_OT_0                    GPIO_OTYPER_OT0
#define GPIO_OTYPER_OT_1                    GPIO_OTYPER_OT1
#define GPIO_OTYPER_OT_2                    GPIO_OTYPER_OT2
#define GPIO_OTYPER_OT_3                    GPIO_OTYPER_OT3
#define GPIO_OTYPER_OT_4                    GPIO_OTYPER_OT4
#define GPIO_OTYPER_OT_5                    GPIO_OTYPER_OT5
#define GPIO_OTYPER_OT_6                    GPIO_OTYPER_OT6
#define GPIO_OTYPER_OT_7                    GPIO_OTYPER_OT7
#define GPIO_OTYPER_OT_8                    GPIO_OTYPER_OT8
#define GPIO_OTYPER_OT_9                    GPIO_OTYPER_OT9
#define GPIO_OTYPER_OT_10                   GPIO_OTYPER_OT10
#define GPIO_OTYPER_OT_11                   GPIO_OTYPER_OT11
#define GPIO_OTYPER_OT_12                   GPIO_OTYPER_OT12
#define GPIO_OTYPER_OT_13                   GPIO_OTYPER_OT13
#define GPIO_OTYPER_OT_14                   GPIO_OTYPER_OT14
#define GPIO_OTYPER_OT_15                   GPIO_OTYPER_OT15

/******************  Bits definition for GPIO_OSPEEDR register  ***************/
#define GPIO_OSPEEDR_OSPEED0                ((uint32_t)0x00000003U)
#define GPIO_OSPEEDR_OSPEED0_0              ((uint32_t)0x00000001U)
#define GPIO_OSPEEDR_OSPEED0_1              ((uint32_t)0x00000002U)
#define GPIO_OSPEEDR_OSPEED1                ((uint32_t)0x0000000CU)
#define GPIO_OSPEEDR_OSPEED1_0              ((uint32_t)0x00000004U)
#define GPIO_OSPEEDR_OSPEED1_1              ((uint32_t)0x00000008U)
#define GPIO_OSPEEDR_OSPEED2                ((uint32_t)0x00000030U)
#define GPIO_OSPEEDR_OSPEED2_0              ((uint32_t)0x00000010U)
#define GPIO_OSPEEDR_OSPEED2_1              ((uint32_t)0x00000020U)
#define GPIO_OSPEEDR_OSPEED3                ((uint32_t)0x000000C0U)
#define GPIO_OSPEEDR_OSPEED3_0              ((uint32_t)0x00000040U)
#define GPIO_OSPEEDR_OSPEED3_1              ((uint32_t)0x00000080U)
#define GPIO_OSPEEDR_OSPEED4                ((uint32_t)0x00000300U)
#define GPIO_OSPEEDR_OSPEED4_0              ((uint32_t)0x00000100U)
#define GPIO_OSPEEDR_OSPEED4_1              ((uint32_t)0x00000200U)
#define GPIO_OSPEEDR_OSPEED5                ((uint32_t)0x00000C00U)
#define GPIO_OSPEEDR_OSPEED5_0              ((uint32_t)0x00000400U)
#define GPIO_OSPEEDR_OSPEED5_1              ((uint32_t)0x00000800U)
#define GPIO_OSPEEDR_OSPEED6                ((uint32_t)0x00003000U)
#define GPIO_OSPEEDR_OSPEED6_0              ((uint32_t)0x00001000U)
#define GPIO_OSPEEDR_OSPEED6_1              ((uint32_t)0x00002000U)
#define GPIO_OSPEEDR_OSPEED7                ((uint32_t)0x0000C000U)
#define GPIO_OSPEEDR_OSPEED7_0              ((uint32_t)0x00004000U)
#define GPIO_OSPEEDR_OSPEED7_1              ((uint32_t)0x00008000U)
#define GPIO_OSPEEDR_OSPEED8                ((uint32_t)0x00030000U)
#define GPIO_OSPEEDR_OSPEED8_0              ((uint32_t)0x00010000U)
#define GPIO_OSPEEDR_OSPEED8_1              ((uint32_t)0x00020000U)
#define GPIO_OSPEEDR_OSPEED9                ((uint32_t)0x000C0000U)
#define GPIO_OSPEEDR_OSPEED9_0              ((uint32_t)0x00040000U)
#define GPIO_OSPEEDR_OSPEED9_1              ((uint32_t)0x00080000U)
#define GPIO_OSPEEDR_OSPEED10               ((uint32_t)0x00300000U)
#define GPIO_OSPEEDR_OSPEED10_0             ((uint32_t)0x00100000U)
#define GPIO_OSPEEDR_OSPEED10_1             ((uint32_t)0x00200000U)
#define GPIO_OSPEEDR_OSPEED11               ((uint32_t)0x00C00000U)
#define GPIO_OSPEEDR_OSPEED11_0             ((uint32_t)0x00400000U)
#define GPIO_OSPEEDR_OSPEED11_1             ((uint32_t)0x00800000U)
#define GPIO_OSPEEDR_OSPEED12               ((uint32_t)0x03000000U)
#define GPIO_OSPEEDR_OSPEED12_0             ((uint32_t)0x01000000U)
#define GPIO_OSPEEDR_OSPEED12_1             ((uint32_t)0x02000000U)
#define GPIO_OSPEEDR_OSPEED13               ((uint32_t)0x0C000000U)
#define GPIO_OSPEEDR_OSPEED13_0             ((uint32_t)0x04000000U)
#define GPIO_OSPEEDR_OSPEED13_1             ((uint32_t)0x08000000U)
#define GPIO_OSPEEDR_OSPEED14               ((uint32_t)0x30000000U)
#define GPIO_OSPEEDR_OSPEED14_0             ((uint32_t)0x10000000U)
#define GPIO_OSPEEDR_OSPEED14_1             ((uint32_t)0x20000000U)
#define GPIO_OSPEEDR_OSPEED15               ((uint32_t)0xC0000000U)
#define GPIO_OSPEEDR_OSPEED15_0             ((uint32_t)0x40000000U)
#define GPIO_OSPEEDR_OSPEED15_1             ((uint32_t)0x80000000U)

/* Legacy defines */
#define GPIO_OSPEEDER_OSPEEDR0              GPIO_OSPEEDR_OSPEED0
#define GPIO_OSPEEDER_OSPEEDR0_0            GPIO_OSPEEDR_OSPEED0_0
#define GPIO_OSPEEDER_OSPEEDR0_1            GPIO_OSPEEDR_OSPEED0_1
#define GPIO_OSPEEDER_OSPEEDR1              GPIO_OSPEEDR_OSPEED1
#define GPIO_OSPEEDER_OSPEEDR1_0            GPIO_OSPEEDR_OSPEED1_0
#define GPIO_OSPEEDER_OSPEEDR1_1            GPIO_OSPEEDR_OSPEED1_1
#define GPIO_OSPEEDER_OSPEEDR2              GPIO_OSPEEDR_OSPEED2
#define GPIO_OSPEEDER_OSPEEDR2_0            GPIO_OSPEEDR_OSPEED2_0
#define GPIO_OSPEEDER_OSPEEDR2_1            GPIO_OSPEEDR_OSPEED2_1
#define GPIO_OSPEEDER_OSPEEDR3              GPIO_OSPEEDR_OSPEED3
#define GPIO_OSPEEDER_OSPEEDR3_0            GPIO_OSPEEDR_OSPEED3_0
#define GPIO_OSPEEDER_OSPEEDR3_1            GPIO_OSPEEDR_OSPEED3_1
#define GPIO_OSPEEDER_OSPEEDR4              GPIO_OSPEEDR_OSPEED4
#define GPIO_OSPEEDER_OSPEEDR4_0            GPIO_OSPEEDR_OSPEED4_0
#define GPIO_OSPEEDER_OSPEEDR4_1            GPIO_OSPEEDR_OSPEED4_1
#define GPIO_OSPEEDER_OSPEEDR5              GPIO_OSPEEDR_OSPEED5
#define GPIO_OSPEEDER_OSPEEDR5_0            GPIO_OSPEEDR_OSPEED5_0
#define GPIO_OSPEEDER_OSPEEDR5_1            GPIO_OSPEEDR_OSPEED5_1
#define GPIO_OSPEEDER_OSPEEDR6              GPIO_OSPEEDR_OSPEED6
#define GPIO_OSPEEDER_OSPEEDR6_0            GPIO_OSPEEDR_OSPEED6_0
#define GPIO_OSPEEDER_OSPEEDR6_1            GPIO_OSPEEDR_OSPEED6_1
#define GPIO_OSPEEDER_OSPEEDR7              GPIO_OSPEEDR_OSPEED7
#define GPIO_OSPEEDER_OSPEEDR7_0            GPIO_OSPEEDR_OSPEED7_0
#define GPIO_OSPEEDER_OSPEEDR7_1            GPIO_OSPEEDR_OSPEED7_1
#define GPIO_OSPEEDER_OSPEEDR8              GPIO_OSPEEDR_OSPEED8
#define GPIO_OSPEEDER_OSPEEDR8_0            GPIO_OSPEEDR_OSPEED8_0
#define GPIO_OSPEEDER_OSPEEDR8_1            GPIO_OSPEEDR_OSPEED8_1
#define GPIO_OSPEEDER_OSPEEDR9              GPIO_OSPEEDR_OSPEED9
#define GPIO_OSPEEDER_OSPEEDR9_0            GPIO_OSPEEDR_OSPEED9_0
#define GPIO_OSPEEDER_OSPEEDR9_1            GPIO_OSPEEDR_OSPEED9_1
#define GPIO_OSPEEDER_OSPEEDR10             GPIO_OSPEEDR_OSPEED10
#define GPIO_OSPEEDER_OSPEEDR10_0           GPIO_OSPEEDR_OSPEED10_0
#define GPIO_OSPEEDER_OSPEEDR10_1           GPIO_OSPEEDR_OSPEED10_1
#define GPIO_OSPEEDER_OSPEEDR11             GPIO_OSPEEDR_OSPEED11
#define GPIO_OSPEEDER_OSPEEDR11_0           GPIO_OSPEEDR_OSPEED11_0
#define GPIO_OSPEEDER_OSPEEDR11_1           GPIO_OSPEEDR_OSPEED11_1
#define GPIO_OSPEEDER_OSPEEDR12             GPIO_OSPEEDR_OSPEED12
#define GPIO_OSPEEDER_OSPEEDR12_0           GPIO_OSPEEDR_OSPEED12_0
#define GPIO_OSPEEDER_OSPEEDR12_1           GPIO_OSPEEDR_OSPEED12_1
#define GPIO_OSPEEDER_OSPEEDR13             GPIO_OSPEEDR_OSPEED13
#define GPIO_OSPEEDER_OSPEEDR13_0           GPIO_OSPEEDR_OSPEED13_0
#define GPIO_OSPEEDER_OSPEEDR13_1           GPIO_OSPEEDR_OSPEED13_1
#define GPIO_OSPEEDER_OSPEEDR14             GPIO_OSPEEDR_OSPEED14
#define GPIO_OSPEEDER_OSPEEDR14_0           GPIO_OSPEEDR_OSPEED14_0
#define GPIO_OSPEEDER_OSPEEDR14_1           GPIO_OSPEEDR_OSPEED14_1
#define GPIO_OSPEEDER_OSPEEDR15             GPIO_OSPEEDR_OSPEED15
#define GPIO_OSPEEDER_OSPEEDR15_0           GPIO_OSPEEDR_OSPEED15_0
#define GPIO_OSPEEDER_OSPEEDR15_1           GPIO_OSPEEDR_OSPEED15_1

/******************  Bits definition for GPIO_PUPDR register  *****************/
#define GPIO_PUPDR_PUPD0                    ((uint32_t)0x00000003U)
#define GPIO_PUPDR_PUPD0_0                  ((uint32_t)0x00000001U)
#define GPIO_PUPDR_PUPD0_1                  ((uint32_t)0x00000002U)
#define GPIO_PUPDR_PUPD1                    ((uint32_t)0x0000000CU)
#define GPIO_PUPDR_PUPD1_0                  ((uint32_t)0x00000004U)
#define GPIO_PUPDR_PUPD1_1                  ((uint32_t)0x00000008U)
#define GPIO_PUPDR_PUPD2                    ((uint32_t)0x00000030U)
#define GPIO_PUPDR_PUPD2_0                  ((uint32_t)0x00000010U)
#define GPIO_PUPDR_PUPD2_1                  ((uint32_t)0x00000020U)
#define GPIO_PUPDR_PUPD3                    ((uint32_t)0x000000C0U)
#define GPIO_PUPDR_PUPD3_0                  ((uint32_t)0x00000040U)
#define GPIO_PUPDR_PUPD3_1                  ((uint32_t)0x00000080U)
#define GPIO_PUPDR_PUPD4                    ((uint32_t)0x00000300U)
#define GPIO_PUPDR_PUPD4_0                  ((uint32_t)0x00000100U)
#define GPIO_PUPDR_PUPD4_1                  ((uint32_t)0x00000200U)
#define GPIO_PUPDR_PUPD5                    ((uint32_t)0x00000C00U)
#define GPIO_PUPDR_PUPD5_0                  ((uint32_t)0x00000400U)
#define GPIO_PUPDR_PUPD5_1                  ((uint32_t)0x00000800U)
#define GPIO_PUPDR_PUPD6                    ((uint32_t)0x00003000U)
#define GPIO_PUPDR_PUPD6_0                  ((uint32_t)0x00001000U)
#define GPIO_PUPDR_PUPD6_1                  ((uint32_t)0x00002000U)
#define GPIO_PUPDR_PUPD7                    ((uint32_t)0x0000C000U)
#define GPIO_PUPDR_PUPD7_0                  ((uint32_t)0x00004000U)
#define GPIO_PUPDR_PUPD7_1                  ((uint32_t)0x00008000U)
#define GPIO_PUPDR_PUPD8                    ((uint32_t)0x00030000U)
#define GPIO_PUPDR_PUPD8_0                  ((uint32_t)0x00010000U)
#define GPIO_PUPDR_PUPD8_1                  ((uint32_t)0x00020000U)
#define GPIO_PUPDR_PUPD9                    ((uint32_t)0x000C0000U)
#define GPIO_PUPDR_PUPD9_0                  ((uint32_t)0x00040000U)
#define GPIO_PUPDR_PUPD9_1                  ((uint32_t)0x00080000U)
#define GPIO_PUPDR_PUPD10                   ((uint32_t)0x00300000U)
#define GPIO_PUPDR_PUPD10_0                 ((uint32_t)0x00100000U)
#define GPIO_PUPDR_PUPD10_1                 ((uint32_t)0x00200000U)
#define GPIO_PUPDR_PUPD11                   ((uint32_t)0x00C00000U)
#define GPIO_PUPDR_PUPD11_0                 ((uint32_t)0x00400000U)
#define GPIO_PUPDR_PUPD11_1                 ((uint32_t)0x00800000U)
#define GPIO_PUPDR_PUPD12                   ((uint32_t)0x03000000U)
#define GPIO_PUPDR_PUPD12_0                 ((uint32_t)0x01000000U)
#define GPIO_PUPDR_PUPD12_1                 ((uint32_t)0x02000000U)
#define GPIO_PUPDR_PUPD13                   ((uint32_t)0x0C000000U)
#define GPIO_PUPDR_PUPD13_0                 ((uint32_t)0x04000000U)
#define GPIO_PUPDR_PUPD13_1                 ((uint32_t)0x08000000U)
#define GPIO_PUPDR_PUPD14                   ((uint32_t)0x30000000U)
#define GPIO_PUPDR_PUPD14_0                 ((uint32_t)0x10000000U)
#define GPIO_PUPDR_PUPD14_1                 ((uint32_t)0x20000000U)
#define GPIO_PUPDR_PUPD15                   ((uint32_t)0xC0000000U)
#define GPIO_PUPDR_PUPD15_0                 ((uint32_t)0x40000000U)
#define GPIO_PUPDR_PUPD15_1                 ((uint32_t)0x80000000U)

/* Legacy defines */
#define GPIO_PUPDR_PUPDR0                   GPIO_PUPDR_PUPD0
#define GPIO_PUPDR_PUPDR0_0                 GPIO_PUPDR_PUPD0_0
#define GPIO_PUPDR_PUPDR0_1                 GPIO_PUPDR_PUPD0_1
#define GPIO_PUPDR_PUPDR1                   GPIO_PUPDR_PUPD1
#define GPIO_PUPDR_PUPDR1_0                 GPIO_PUPDR_PUPD1_0
#define GPIO_PUPDR_PUPDR1_1                 GPIO_PUPDR_PUPD1_1
#define GPIO_PUPDR_PUPDR2                   GPIO_PUPDR_PUPD2
#define GPIO_PUPDR_PUPDR2_0                 GPIO_PUPDR_PUPD2_0
#define GPIO_PUPDR_PUPDR2_1                 GPIO_PUPDR_PUPD2_1
#define GPIO_PUPDR_PUPDR3                   GPIO_PUPDR_PUPD3
#define GPIO_PUPDR_PUPDR3_0                 GPIO_PUPDR_PUPD3_0
#define GPIO_PUPDR_PUPDR3_1                 GPIO_PUPDR_PUPD3_1
#define GPIO_PUPDR_PUPDR4                   GPIO_PUPDR_PUPD4
#define GPIO_PUPDR_PUPDR4_0                 GPIO_PUPDR_PUPD4_0
#define GPIO_PUPDR_PUPDR4_1                 GPIO_PUPDR_PUPD4_1
#define GPIO_PUPDR_PUPDR5                   GPIO_PUPDR_PUPD5
#define GPIO_PUPDR_PUPDR5_0                 GPIO_PUPDR_PUPD5_0
#define GPIO_PUPDR_PUPDR5_1                 GPIO_PUPDR_PUPD5_1
#define GPIO_PUPDR_PUPDR6                   GPIO_PUPDR_PUPD6
#define GPIO_PUPDR_PUPDR6_0                 GPIO_PUPDR_PUPD6_0
#define GPIO_PUPDR_PUPDR6_1                 GPIO_PUPDR_PUPD6_1
#define GPIO_PUPDR_PUPDR7                   GPIO_PUPDR_PUPD7
#define GPIO_PUPDR_PUPDR7_0                 GPIO_PUPDR_PUPD7_0
#define GPIO_PUPDR_PUPDR7_1                 GPIO_PUPDR_PUPD7_1
#define GPIO_PUPDR_PUPDR8                   GPIO_PUPDR_PUPD8
#define GPIO_PUPDR_PUPDR8_0                 GPIO_PUPDR_PUPD8_0
#define GPIO_PUPDR_PUPDR8_1                 GPIO_PUPDR_PUPD8_1
#define GPIO_PUPDR_PUPDR9                   GPIO_PUPDR_PUPD9
#define GPIO_PUPDR_PUPDR9_0                 GPIO_PUPDR_PUPD9_0
#define GPIO_PUPDR_PUPDR9_1                 GPIO_PUPDR_PUPD9_1
#define GPIO_PUPDR_PUPDR10                  GPIO_PUPDR_PUPD10
#define GPIO_PUPDR_PUPDR10_0                GPIO_PUPDR_PUPD10_0
#define GPIO_PUPDR_PUPDR10_1                GPIO_PUPDR_PUPD10_1
#define GPIO_PUPDR_PUPDR11                  GPIO_PUPDR_PUPD11
#define GPIO_PUPDR_PUPDR11_0                GPIO_PUPDR_PUPD11_0
#define GPIO_PUPDR_PUPDR11_1                GPIO_PUPDR_PUPD11_1
#define GPIO_PUPDR_PUPDR12                  GPIO_PUPDR_PUPD12
#define GPIO_PUPDR_PUPDR12_0                GPIO_PUPDR_PUPD12_0
#define GPIO_PUPDR_PUPDR12_1                GPIO_PUPDR_PUPD12_1
#define GPIO_PUPDR_PUPDR13                  GPIO_PUPDR_PUPD13
#define GPIO_PUPDR_PUPDR13_0                GPIO_PUPDR_PUPD13_0
#define GPIO_PUPDR_PUPDR13_1                GPIO_PUPDR_PUPD13_1
#define GPIO_PUPDR_PUPDR14                  GPIO_PUPDR_PUPD14
#define GPIO_PUPDR_PUPDR14_0                GPIO_PUPDR_PUPD14_0
#define GPIO_PUPDR_PUPDR14_1                GPIO_PUPDR_PUPD14_1
#define GPIO_PUPDR_PUPDR15                  GPIO_PUPDR_PUPD15
#define GPIO_PUPDR_PUPDR15_0                GPIO_PUPDR_PUPD15_0
#define GPIO_PUPDR_PUPDR15_1                GPIO_PUPDR_PUPD15_1

/******************  Bits definition for GPIO_IDR register  *******************/
#define GPIO_IDR_ID0                        ((uint32_t)0x00000001U)
#define GPIO_IDR_ID1                        ((uint32_t)0x00000002U)
#define GPIO_IDR_ID2                        ((uint32_t)0x00000004U)
#define GPIO_IDR_ID3                        ((uint32_t)0x00000008U)
#define GPIO_IDR_ID4                        ((uint32_t)0x00000010U)
#define GPIO_IDR_ID5                        ((uint32_t)0x00000020U)
#define GPIO_IDR_ID6                        ((uint32_t)0x00000040U)
#define GPIO_IDR_ID7                        ((uint32_t)0x00000080U)
#define GPIO_IDR_ID8                        ((uint32_t)0x00000100U)
#define GPIO_IDR_ID9                        ((uint32_t)0x00000200U)
#define GPIO_IDR_ID10                       ((uint32_t)0x00000400U)
#define GPIO_IDR_ID11                       ((uint32_t)0x00000800U)
#define GPIO_IDR_ID12                       ((uint32_t)0x00001000U)
#define GPIO_IDR_ID13                       ((uint32_t)0x00002000U)
#define GPIO_IDR_ID14                       ((uint32_t)0x00004000U)
#define GPIO_IDR_ID15                       ((uint32_t)0x00008000U)

/* Legacy defines */
#define GPIO_IDR_IDR_0                      GPIO_IDR_ID0
#define GPIO_IDR_IDR_1                      GPIO_IDR_ID1
#define GPIO_IDR_IDR_2                      GPIO_IDR_ID2
#define GPIO_IDR_IDR_3                      GPIO_IDR_ID3
#define GPIO_IDR_IDR_4                      GPIO_IDR_ID4
#define GPIO_IDR_IDR_5                      GPIO_IDR_ID5
#define GPIO_IDR_IDR_6                      GPIO_IDR_ID6
#define GPIO_IDR_IDR_7                      GPIO_IDR_ID7
#define GPIO_IDR_IDR_8                      GPIO_IDR_ID8
#define GPIO_IDR_IDR_9                      GPIO_IDR_ID9
#define GPIO_IDR_IDR_10                     GPIO_IDR_ID10
#define GPIO_IDR_IDR_11                     GPIO_IDR_ID11
#define GPIO_IDR_IDR_12                     GPIO_IDR_ID12
#define GPIO_IDR_IDR_13                     GPIO_IDR_ID13
#define GPIO_IDR_IDR_14                     GPIO_IDR_ID14
#define GPIO_IDR_IDR_15                     GPIO_IDR_ID15

/******************  Bits definition for GPIO_ODR register  *******************/
#define GPIO_ODR_OD0                        ((uint32_t)0x00000001U)
#define GPIO_ODR_OD1                        ((uint32_t)0x00000002U)
#define GPIO_ODR_OD2                        ((uint32_t)0x00000004U)
#define GPIO_ODR_OD3                        ((uint32_t)0x00000008U)
#define GPIO_ODR_OD4                        ((uint32_t)0x00000010U)
#define GPIO_ODR_OD5                        ((uint32_t)0x00000020U)
#define GPIO_ODR_OD6                        ((uint32_t)0x00000040U)
#define GPIO_ODR_OD7                        ((uint32_t)0x00000080U)
#define GPIO_ODR_OD8                        ((uint32_t)0x00000100U)
#define GPIO_ODR_OD9                        ((uint32_t)0x00000200U)
#define GPIO_ODR_OD10                       ((uint32_t)0x00000400U)
#define GPIO_ODR_OD11                       ((uint32_t)0x00000800U)
#define GPIO_ODR_OD12                       ((uint32_t)0x00001000U)
#define GPIO_ODR_OD13                       ((uint32_t)0x00002000U)
#define GPIO_ODR_OD14                       ((uint32_t)0x00004000U)
#define GPIO_ODR_OD15                       ((uint32_t)0x00008000U)

/* Legacy defines */
#define GPIO_ODR_ODR_0                      GPIO_ODR_OD0
#define GPIO_ODR_ODR_1                      GPIO_ODR_OD1
#define GPIO_ODR_ODR_2                      GPIO_ODR_OD2
#define GPIO_ODR_ODR_3                      GPIO_ODR_OD3
#define GPIO_ODR_ODR_4                      GPIO_ODR_OD4
#define GPIO_ODR_ODR_5                      GPIO_ODR_OD5
#define GPIO_ODR_ODR_6                      GPIO_ODR_OD6
#define GPIO_ODR_ODR_7                      GPIO_ODR_OD7
#define GPIO_ODR_ODR_8                      GPIO_ODR_OD8
#define GPIO_ODR_ODR_9                      GPIO_ODR_OD9
#define GPIO_ODR_ODR_10                     GPIO_ODR_OD10
#define GPIO_ODR_ODR_11                     GPIO_ODR_OD11
#define GPIO_ODR_ODR_12                     GPIO_ODR_OD12
#define GPIO_ODR_ODR_13                     GPIO_ODR_OD13
#define GPIO_ODR_ODR_14                     GPIO_ODR_OD14
#define GPIO_ODR_ODR_15                     GPIO_ODR_OD15

/******************  Bits definition for GPIO_BSRR register  ******************/
#define GPIO_BSRR_BS0                       ((uint32_t)0x00000001U)
#define GPIO_BSRR_BS1                       ((uint32_t)0x00000002U)
#define GPIO_BSRR_BS2                       ((uint32_t)0x00000004U)
#define GPIO_BSRR_BS3                       ((uint32_t)0x00000008U)
#define GPIO_BSRR_BS4                       ((uint32_t)0x00000010U)
#define GPIO_BSRR_BS5                       ((uint32_t)0x00000020U)
#define GPIO_BSRR_BS6                       ((uint32_t)0x00000040U)
#define GPIO_BSRR_BS7                       ((uint32_t)0x00000080U)
#define GPIO_BSRR_BS8                       ((uint32_t)0x00000100U)
#define GPIO_BSRR_BS9                       ((uint32_t)0x00000200U)
#define GPIO_BSRR_BS10                      ((uint32_t)0x00000400U)
#define GPIO_BSRR_BS11                      ((uint32_t)0x00000800U)
#define GPIO_BSRR_BS12                      ((uint32_t)0x00001000U)
#define GPIO_BSRR_BS13                      ((uint32_t)0x00002000U)
#define GPIO_BSRR_BS14                      ((uint32_t)0x00004000U)
#define GPIO_BSRR_BS15                      ((uint32_t)0x00008000U)
#define GPIO_BSRR_BR0                       ((uint32_t)0x00010000U)
#define GPIO_BSRR_BR1                       ((uint32_t)0x00020000U)
#define GPIO_BSRR_BR2                       ((uint32_t)0x00040000U)
#define GPIO_BSRR_BR3                       ((uint32_t)0x00080000U)
#define GPIO_BSRR_BR4                       ((uint32_t)0x00100000U)
#define GPIO_BSRR_BR5                       ((uint32_t)0x00200000U)
#define GPIO_BSRR_BR6                       ((uint32_t)0x00400000U)
#define GPIO_BSRR_BR7                       ((uint32_t)0x00800000U)
#define GPIO_BSRR_BR8                       ((uint32_t)0x01000000U)
#define GPIO_BSRR_BR9                       ((uint32_t)0x02000000U)
#define GPIO_BSRR_BR10                      ((uint32_t)0x04000000U)
#define GPIO_BSRR_BR11                      ((uint32_t)0x08000000U)
#define GPIO_BSRR_BR12                      ((uint32_t)0x10000000U)
#define GPIO_BSRR_BR13                      ((uint32_t)0x20000000U)
#define GPIO_BSRR_BR14                      ((uint32_t)0x40000000U)
#define GPIO_BSRR_BR15                      ((uint32_t)0x80000000U)

/* Legacy defines */
#define GPIO_BSRR_BS_0                      GPIO_BSRR_BS0
#define GPIO_BSRR_BS_1                      GPIO_BSRR_BS1
#define GPIO_BSRR_BS_2                      GPIO_BSRR_BS2
#define GPIO_BSRR_BS_3                      GPIO_BSRR_BS3
#define GPIO_BSRR_BS_4                      GPIO_BSRR_BS4
#define GPIO_BSRR_BS_5                      GPIO_BSRR_BS5
#define GPIO_BSRR_BS_6                      GPIO_BSRR_BS6
#define GPIO_BSRR_BS_7                      GPIO_BSRR_BS7
#define GPIO_BSRR_BS_8                      GPIO_BSRR_BS8
#define GPIO_BSRR_BS_9                      GPIO_BSRR_BS9
#define GPIO_BSRR_BS_10                     GPIO_BSRR_BS10
#define GPIO_BSRR_BS_11                     GPIO_BSRR_BS11
#define GPIO_BSRR_BS_12                     GPIO_BSRR_BS12
#define GPIO_BSRR_BS_13                     GPIO_BSRR_BS13
#define GPIO_BSRR_BS_14                     GPIO_BSRR_BS14
#define GPIO_BSRR_BS_15                     GPIO_BSRR_BS15
#define GPIO_BSRR_BR_0                      GPIO_BSRR_BR0
#define GPIO_BSRR_BR_1                      GPIO_BSRR_BR1
#define GPIO_BSRR_BR_2                      GPIO_BSRR_BR2
#define GPIO_BSRR_BR_3                      GPIO_BSRR_BR3
#define GPIO_BSRR_BR_4                      GPIO_BSRR_BR4
#define GPIO_BSRR_BR_5                      GPIO_BSRR_BR5
#define GPIO_BSRR_BR_6                      GPIO_BSRR_BR6
#define GPIO_BSRR_BR_7                      GPIO_BSRR_BR7
#define GPIO_BSRR_BR_8                      GPIO_BSRR_BR8
#define GPIO_BSRR_BR_9                      GPIO_BSRR_BR9
#define GPIO_BSRR_BR_10                     GPIO_BSRR_BR10
#define GPIO_BSRR_BR_11                     GPIO_BSRR_BR11
#define GPIO_BSRR_BR_12                     GPIO_BSRR_BR12
#define GPIO_BSRR_BR_13                     GPIO_BSRR_BR13
#define GPIO_BSRR_BR_14                     GPIO_BSRR_BR14
#define GPIO_BSRR_BR_15                     GPIO_BSRR_BR15

/****************** Bit definition for GPIO_LCKR register *********************/
#define GPIO_LCKR_LCK0                       0x00000001U
#define GPIO_LCKR_LCK1                       0x00000002U
#define GPIO_LCKR_LCK2                       0x00000004U
#define GPIO_LCKR_LCK3                       0x00000008U
#define GPIO_LCKR_LCK4                       0x00000010U
#define GPIO_LCKR_LCK5                       0x00000020U
#define GPIO_LCKR_LCK6                       0x00000040U
#define GPIO_LCKR_LCK7                       0x00000080U
#define GPIO_LCKR_LCK8                       0x00000100U
#define GPIO_LCKR_LCK9                       0x00000200U
#define GPIO_LCKR_LCK10                      0x00000400U
#define GPIO_LCKR_LCK11                      0x00000800U
#define GPIO_LCKR_LCK12                      0x00001000U
#define GPIO_LCKR_LCK13                      0x00002000U
#define GPIO_LCKR_LCK14                      0x00004000U
#define GPIO_LCKR_LCK15                      0x00008000U
#define GPIO_LCKR_LCKK                       0x00010000U

/****************** Bit definition for GPIO_AFRL register *********************/
#define GPIO_AFRL_AFSEL0                    ((uint32_t)0x0000000FU)
#define GPIO_AFRL_AFSEL0_0                  ((uint32_t)0x00000001U)
#define GPIO_AFRL_AFSEL0_1                  ((uint32_t)0x00000002U)
#define GPIO_AFRL_AFSEL0_2                  ((uint32_t)0x00000004U)
#define GPIO_AFRL_AFSEL0_3                  ((uint32_t)0x00000008U)
#define GPIO_AFRL_AFSEL1                    ((uint32_t)0x000000F0U)
#define GPIO_AFRL_AFSEL1_0                  ((uint32_t)0x00000010U)
#define GPIO_AFRL_AFSEL1_1                  ((uint32_t)0x00000020U)
#define GPIO_AFRL_AFSEL1_2                  ((uint32_t)0x00000040U)
#define GPIO_AFRL_AFSEL1_3                  ((uint32_t)0x00000080U)
#define GPIO_AFRL_AFSEL2                    ((uint32_t)0x00000F00U)
#define GPIO_AFRL_AFSEL2_0                  ((uint32_t)0x00000100U)
#define GPIO_AFRL_AFSEL2_1                  ((uint32_t)0x00000200U)
#define GPIO_AFRL_AFSEL2_2                  ((uint32_t)0x00000400U)
#define GPIO_AFRL_AFSEL2_3                  ((uint32_t)0x00000800U)
#define GPIO_AFRL_AFSEL3                    ((uint32_t)0x0000F000U)
#define GPIO_AFRL_AFSEL3_0                  ((uint32_t)0x00001000U)
#define GPIO_AFRL_AFSEL3_1                  ((uint32_t)0x00002000U)
#define GPIO_AFRL_AFSEL3_2                  ((uint32_t)0x00004000U)
#define GPIO_AFRL_AFSEL3_3                  ((uint32_t)0x00008000U)
#define GPIO_AFRL_AFSEL4                    ((uint32_t)0x000F0000U)
#define GPIO_AFRL_AFSEL4_0                  ((uint32_t)0x00010000U)
#define GPIO_AFRL_AFSEL4_1                  ((uint32_t)0x00020000U)
#define GPIO_AFRL_AFSEL4_2                  ((uint32_t)0x00040000U)
#define GPIO_AFRL_AFSEL4_3                  ((uint32_t)0x00080000U)
#define GPIO_AFRL_AFSEL5                    ((uint32_t)0x00F00000U)
#define GPIO_AFRL_AFSEL5_0                  ((uint32_t)0x00100000U)
#define GPIO_AFRL_AFSEL5_1                  ((uint32_t)0x00200000U)
#define GPIO_AFRL_AFSEL5_2                  ((uint32_t)0x00400000U)
#define GPIO_AFRL_AFSEL5_3                  ((uint32_t)0x00800000U)
#define GPIO_AFRL_AFSEL6                    ((uint32_t)0x0F000000U)
#define GPIO_AFRL_AFSEL6_0                  ((uint32_t)0x01000000U)
#define GPIO_AFRL_AFSEL6_1                  ((uint32_t)0x02000000U)
#define GPIO_AFRL_AFSEL6_2                  ((uint32_t)0x04000000U)
#define GPIO_AFRL_AFSEL6_3                  ((uint32_t)0x08000000U)
#define GPIO_AFRL_AFSEL7                    ((uint32_t)0xF0000000U)
#define GPIO_AFRL_AFSEL7_0                  ((uint32_t)0x10000000U)
#define GPIO_AFRL_AFSEL7_1                  ((uint32_t)0x20000000U)
#define GPIO_AFRL_AFSEL7_2                  ((uint32_t)0x40000000U)
#define GPIO_AFRL_AFSEL7_3                  ((uint32_t)0x80000000U)

/* Legacy defines */
#define GPIO_AFRL_AFRL0                      GPIO_AFRL_AFSEL0
#define GPIO_AFRL_AFRL0_0                    GPIO_AFRL_AFSEL0_0
#define GPIO_AFRL_AFRL0_1                    GPIO_AFRL_AFSEL0_1
#define GPIO_AFRL_AFRL0_2                    GPIO_AFRL_AFSEL0_2
#define GPIO_AFRL_AFRL0_3                    GPIO_AFRL_AFSEL0_3
#define GPIO_AFRL_AFRL1                      GPIO_AFRL_AFSEL1
#define GPIO_AFRL_AFRL1_0                    GPIO_AFRL_AFSEL1_0
#define GPIO_AFRL_AFRL1_1                    GPIO_AFRL_AFSEL1_1
#define GPIO_AFRL_AFRL1_2                    GPIO_AFRL_AFSEL1_2
#define GPIO_AFRL_AFRL1_3                    GPIO_AFRL_AFSEL1_3
#define GPIO_AFRL_AFRL2                      GPIO_AFRL_AFSEL2
#define GPIO_AFRL_AFRL2_0                    GPIO_AFRL_AFSEL2_0
#define GPIO_AFRL_AFRL2_1                    GPIO_AFRL_AFSEL2_1
#define GPIO_AFRL_AFRL2_2                    GPIO_AFRL_AFSEL2_2
#define GPIO_AFRL_AFRL2_3                    GPIO_AFRL_AFSEL2_3
#define GPIO_AFRL_AFRL3                      GPIO_AFRL_AFSEL3
#define GPIO_AFRL_AFRL3_0                    GPIO_AFRL_AFSEL3_0
#define GPIO_AFRL_AFRL3_1                    GPIO_AFRL_AFSEL3_1
#define GPIO_AFRL_AFRL3_2                    GPIO_AFRL_AFSEL3_2
#define GPIO_AFRL_AFRL3_3                    GPIO_AFRL_AFSEL3_3
#define GPIO_AFRL_AFRL4                      GPIO_AFRL_AFSEL4
#define GPIO_AFRL_AFRL4_0                    GPIO_AFRL_AFSEL4_0
#define GPIO_AFRL_AFRL4_1                    GPIO_AFRL_AFSEL4_1
#define GPIO_AFRL_AFRL4_2                    GPIO_AFRL_AFSEL4_2
#define GPIO_AFRL_AFRL4_3                    GPIO_AFRL_AFSEL4_3
#define GPIO_AFRL_AFRL5                      GPIO_AFRL_AFSEL5
#define GPIO_AFRL_AFRL5_0                    GPIO_AFRL_AFSEL5_0
#define GPIO_AFRL_AFRL5_1                    GPIO_AFRL_AFSEL5_1
#define GPIO_AFRL_AFRL5_2                    GPIO_AFRL_AFSEL5_2
#define GPIO_AFRL_AFRL5_3                    GPIO_AFRL_AFSEL5_3
#define GPIO_AFRL_AFRL6                      GPIO_AFRL_AFSEL6
#define GPIO_AFRL_AFRL6_0                    GPIO_AFRL_AFSEL6_0
#define GPIO_AFRL_AFRL6_1                    GPIO_AFRL_AFSEL6_1
#define GPIO_AFRL_AFRL6_2                    GPIO_AFRL_AFSEL6_2
#define GPIO_AFRL_AFRL6_3                    GPIO_AFRL_AFSEL6_3
#define GPIO_AFRL_AFRL7                      GPIO_AFRL_AFSEL7
#define GPIO_AFRL_AFRL7_0                    GPIO_AFRL_AFSEL7_0
#define GPIO_AFRL_AFRL7_1                    GPIO_AFRL_AFSEL7_1
#define GPIO_AFRL_AFRL7_2                    GPIO_AFRL_AFSEL7_2
#define GPIO_AFRL_AFRL7_3                    GPIO_AFRL_AFSEL7_3

/****************** Bit definition for GPIO_AFRH register *********************/
#define GPIO_AFRH_AFSEL8                    ((uint32_t)0x0000000FU)
#define GPIO_AFRH_AFSEL8_0                  ((uint32_t)0x00000001U)
#define GPIO_AFRH_AFSEL8_1                  ((uint32_t)0x00000002U)
#define GPIO_AFRH_AFSEL8_2                  ((uint32_t)0x00000004U)
#define GPIO_AFRH_AFSEL8_3                  ((uint32_t)0x00000008U)
#define GPIO_AFRH_AFSEL9                    ((uint32_t)0x000000F0U)
#define GPIO_AFRH_AFSEL9_0                  ((uint32_t)0x00000010U)
#define GPIO_AFRH_AFSEL9_1                  ((uint32_t)0x00000020U)
#define GPIO_AFRH_AFSEL9_2                  ((uint32_t)0x00000040U)
#define GPIO_AFRH_AFSEL9_3                  ((uint32_t)0x00000080U)
#define GPIO_AFRH_AFSEL10                   ((uint32_t)0x00000F00U)
#define GPIO_AFRH_AFSEL10_0                 ((uint32_t)0x00000100U)
#define GPIO_AFRH_AFSEL10_1                 ((uint32_t)0x00000200U)
#define GPIO_AFRH_AFSEL10_2                 ((uint32_t)0x00000400U)
#define GPIO_AFRH_AFSEL10_3                 ((uint32_t)0x00000800U)
#define GPIO_AFRH_AFSEL11                   ((uint32_t)0x0000F000U)
#define GPIO_AFRH_AFSEL11_0                 ((uint32_t)0x00001000U)
#define GPIO_AFRH_AFSEL11_1                 ((uint32_t)0x00002000U)
#define GPIO_AFRH_AFSEL11_2                 ((uint32_t)0x00004000U)
#define GPIO_AFRH_AFSEL11_3                 ((uint32_t)0x00008000U)
#define GPIO_AFRH_AFSEL12                   ((uint32_t)0x000F0000U)
#define GPIO_AFRH_AFSEL12_0                 ((uint32_t)0x00010000U)
#define GPIO_AFRH_AFSEL12_1                 ((uint32_t)0x00020000U)
#define GPIO_AFRH_AFSEL12_2                 ((uint32_t)0x00040000U)
#define GPIO_AFRH_AFSEL12_3                 ((uint32_t)0x00080000U)
#define GPIO_AFRH_AFSEL13                   ((uint32_t)0x00F00000U)
#define GPIO_AFRH_AFSEL13_0                 ((uint32_t)0x00100000U)
#define GPIO_AFRH_AFSEL13_1                 ((uint32_t)0x00200000U)
#define GPIO_AFRH_AFSEL13_2                 ((uint32_t)0x00400000U)
#define GPIO_AFRH_AFSEL13_3                 ((uint32_t)0x00800000U)
#define GPIO_AFRH_AFSEL14                   ((uint32_t)0x0F000000U)
#define GPIO_AFRH_AFSEL14_0                 ((uint32_t)0x01000000U)
#define GPIO_AFRH_AFSEL14_1                 ((uint32_t)0x02000000U)
#define GPIO_AFRH_AFSEL14_2                 ((uint32_t)0x04000000U)
#define GPIO_AFRH_AFSEL14_3                 ((uint32_t)0x08000000U)
#define GPIO_AFRH_AFSEL15                   ((uint32_t)0xF0000000U)
#define GPIO_AFRH_AFSEL15_0                 ((uint32_t)0x10000000U)
#define GPIO_AFRH_AFSEL15_1                 ((uint32_t)0x20000000U)
#define GPIO_AFRH_AFSEL15_2                 ((uint32_t)0x40000000U)
#define GPIO_AFRH_AFSEL15_3                 ((uint32_t)0x80000000U)

/* Legacy defines */
#define GPIO_AFRH_AFRH0                      GPIO_AFRH_AFSEL8
#define GPIO_AFRH_AFRH0_0                    GPIO_AFRH_AFSEL8_0
#define GPIO_AFRH_AFRH0_1                    GPIO_AFRH_AFSEL8_1
#define GPIO_AFRH_AFRH0_2                    GPIO_AFRH_AFSEL8_2
#define GPIO_AFRH_AFRH0_3                    GPIO_AFRH_AFSEL8_3
#define GPIO_AFRH_AFRH1                      GPIO_AFRH_AFSEL9
#define GPIO_AFRH_AFRH1_0                    GPIO_AFRH_AFSEL9_0
#define GPIO_AFRH_AFRH1_1                    GPIO_AFRH_AFSEL9_1
#define GPIO_AFRH_AFRH1_2                    GPIO_AFRH_AFSEL9_2
#define GPIO_AFRH_AFRH1_3                    GPIO_AFRH_AFSEL9_3
#define GPIO_AFRH_AFRH2                      GPIO_AFRH_AFSEL10
#define GPIO_AFRH_AFRH2_0                    GPIO_AFRH_AFSEL10_0
#define GPIO_AFRH_AFRH2_1                    GPIO_AFRH_AFSEL10_1
#define GPIO_AFRH_AFRH2_2                    GPIO_AFRH_AFSEL10_2
#define GPIO_AFRH_AFRH2_3                    GPIO_AFRH_AFSEL10_3
#define GPIO_AFRH_AFRH3                      GPIO_AFRH_AFSEL11
#define GPIO_AFRH_AFRH3_0                    GPIO_AFRH_AFSEL11_0
#define GPIO_AFRH_AFRH3_1                    GPIO_AFRH_AFSEL11_1
#define GPIO_AFRH_AFRH3_2                    GPIO_AFRH_AFSEL11_2
#define GPIO_AFRH_AFRH3_3                    GPIO_AFRH_AFSEL11_3
#define GPIO_AFRH_AFRH4                      GPIO_AFRH_AFSEL12
#define GPIO_AFRH_AFRH4_0                    GPIO_AFRH_AFSEL12_0
#define GPIO_AFRH_AFRH4_1                    GPIO_AFRH_AFSEL12_1
#define GPIO_AFRH_AFRH4_2                    GPIO_AFRH_AFSEL12_2
#define GPIO_AFRH_AFRH4_3                    GPIO_AFRH_AFSEL12_3
#define GPIO_AFRH_AFRH5                      GPIO_AFRH_AFSEL13
#define GPIO_AFRH_AFRH5_0                    GPIO_AFRH_AFSEL13_0
#define GPIO_AFRH_AFRH5_1                    GPIO_AFRH_AFSEL13_1
#define GPIO_AFRH_AFRH5_2                    GPIO_AFRH_AFSEL13_2
#define GPIO_AFRH_AFRH5_3                    GPIO_AFRH_AFSEL13_3
#define GPIO_AFRH_AFRH6                      GPIO_AFRH_AFSEL14
#define GPIO_AFRH_AFRH6_0                    GPIO_AFRH_AFSEL14_0
#define GPIO_AFRH_AFRH6_1                    GPIO_AFRH_AFSEL14_1
#define GPIO_AFRH_AFRH6_2                    GPIO_AFRH_AFSEL14_2
#define GPIO_AFRH_AFRH6_3                    GPIO_AFRH_AFSEL14_3
#define GPIO_AFRH_AFRH7                      GPIO_AFRH_AFSEL15
#define GPIO_AFRH_AFRH7_0                    GPIO_AFRH_AFSEL15_0
#define GPIO_AFRH_AFRH7_1                    GPIO_AFRH_AFSEL15_1
#define GPIO_AFRH_AFRH7_2                    GPIO_AFRH_AFSEL15_2
#define GPIO_AFRH_AFRH7_3                    GPIO_AFRH_AFSEL15_3

/******************  Bits definition for GPIO_BRR register  ******************/
#define GPIO_BRR_BR0                        ((uint32_t)0x00000001U)
#define GPIO_BRR_BR1                        ((uint32_t)0x00000002U)
#define GPIO_BRR_BR2                        ((uint32_t)0x00000004U)
#define GPIO_BRR_BR3                        ((uint32_t)0x00000008U)
#define GPIO_BRR_BR4                        ((uint32_t)0x00000010U)
#define GPIO_BRR_BR5                        ((uint32_t)0x00000020U)
#define GPIO_BRR_BR6                        ((uint32_t)0x00000040U)
#define GPIO_BRR_BR7                        ((uint32_t)0x00000080U)
#define GPIO_BRR_BR8                        ((uint32_t)0x00000100U)
#define GPIO_BRR_BR9                        ((uint32_t)0x00000200U)
#define GPIO_BRR_BR10                       ((uint32_t)0x00000400U)
#define GPIO_BRR_BR11                       ((uint32_t)0x00000800U)
#define GPIO_BRR_BR12                       ((uint32_t)0x00001000U)
#define GPIO_BRR_BR13                       ((uint32_t)0x00002000U)
#define GPIO_BRR_BR14                       ((uint32_t)0x00004000U)
#define GPIO_BRR_BR15                       ((uint32_t)0x00008000U)

/******************************************************************************/
/*                                                                            */
/*                      Inter-integrated Circuit Interface                    */
/*                                                                            */
/******************************************************************************/
/*******************  Bit definition for I2C_CR1 register  ********************/
#define  I2C_CR1_PE                          0x00000001U     /*!<Peripheral Enable                             */
#define  I2C_CR1_SMBUS                       0x00000002U     /*!<SMBus Mode                                    */
#define  I2C_CR1_SMBTYPE                     0x00000008U     /*!<SMBus Type                                    */
#define  I2C_CR1_ENARP                       0x00000010U     /*!<ARP Enable                                    */
#define  I2C_CR1_ENPEC                       0x00000020U     /*!<PEC Enable                                    */
#define  I2C_CR1_ENGC                        0x00000040U     /*!<General Call Enable                           */
#define  I2C_CR1_NOSTRETCH                   0x00000080U     /*!<Clock Stretching Disable (Slave mode)  */
#define  I2C_CR1_START                       0x00000100U     /*!<Start Generation                              */
#define  I2C_CR1_STOP                        0x00000200U     /*!<Stop Generation                               */
#define  I2C_CR1_ACK                         0x00000400U     /*!<Acknowledge Enable                            */
#define  I2C_CR1_POS                         0x00000800U     /*!<Acknowledge/PEC Position (for data reception) */
#define  I2C_CR1_PEC                         0x00001000U     /*!<Packet Error Checking                         */
#define  I2C_CR1_ALERT                       0x00002000U     /*!<SMBus Alert                                   */
#define  I2C_CR1_SWRST                       0x00008000U     /*!<Software Reset                                */

/*******************  Bit definition for I2C_CR2 register  ********************/
#define  I2C_CR2_FREQ                        0x0000003FU     /*!<FREQ[5:0] bits (Peripheral Clock Frequency)   */
#define  I2C_CR2_FREQ_0                      0x00000001U     /*!<Bit 0 */
#define  I2C_CR2_FREQ_1                      0x00000002U     /*!<Bit 1 */
#define  I2C_CR2_FREQ_2                      0x00000004U     /*!<Bit 2 */
#define  I2C_CR2_FREQ_3                      0x00000008U     /*!<Bit 3 */
#define  I2C_CR2_FREQ_4                      0x00000010U     /*!<Bit 4 */
#define  I2C_CR2_FREQ_5                      0x00000020U     /*!<Bit 5 */

#define  I2C_CR2_ITERREN                     0x00000100U     /*!<Error Interrupt Enable  */
#define  I2C_CR2_ITEVTEN                     0x00000200U     /*!<Event Interrupt Enable  */
#define  I2C_CR2_ITBUFEN                     0x00000400U     /*!<Buffer Interrupt Enable */
#define  I2C_CR2_DMAEN                       0x00000800U     /*!<DMA Requests Enable     */
#define  I2C_CR2_LAST                        0x00001000U     /*!<DMA Last Transfer       */

/*******************  Bit definition for I2C_OAR1 register  *******************/
#define  I2C_OAR1_ADD1_7                     0x000000FEU     /*!<Interface Address */
#define  I2C_OAR1_ADD8_9                     0x00000300U     /*!<Interface Address */

#define  I2C_OAR1_ADD0                       0x00000001U     /*!<Bit 0 */
#define  I2C_OAR1_ADD1                       0x00000002U     /*!<Bit 1 */
#define  I2C_OAR1_ADD2                       0x00000004U     /*!<Bit 2 */
#define  I2C_OAR1_ADD3                       0x00000008U     /*!<Bit 3 */
#define  I2C_OAR1_ADD4                       0x00000010U     /*!<Bit 4 */
#define  I2C_OAR1_ADD5                       0x00000020U     /*!<Bit 5 */
#define  I2C_OAR1_ADD6                       0x00000040U     /*!<Bit 6 */
#define  I2C_OAR1_ADD7                       0x00000080U     /*!<Bit 7 */
#define  I2C_OAR1_ADD8                       0x00000100U     /*!<Bit 8 */
#define  I2C_OAR1_ADD9                       0x00000200U     /*!<Bit 9 */

#define  I2C_OAR1_ADDMODE                    0x00008000U     /*!<Addressing Mode (Slave mode) */

/*******************  Bit definition for I2C_OAR2 register  *******************/
#define  I2C_OAR2_ENDUAL                     0x00000001U        /*!<Dual addressing mode enable */
#define  I2C_OAR2_ADD2                       0x000000FEU        /*!<Interface address           */

/********************  Bit definition for I2C_DR register  ********************/
#define  I2C_DR_DR                           0x000000FFU        /*!<8-bit Data Register         */

/*******************  Bit definition for I2C_SR1 register  ********************/
#define  I2C_SR1_SB                          0x00000001U     /*!<Start Bit (Master mode)                  */
#define  I2C_SR1_ADDR                        0x00000002U     /*!<Address sent (master mode)/matched (slave mode) */
#define  I2C_SR1_BTF                         0x00000004U     /*!<Byte Transfer Finished                          */
#define  I2C_SR1_ADD10                       0x00000008U     /*!<10-bit header sent (Master mode)         */
#define  I2C_SR1_STOPF                       0x00000010U     /*!<Stop detection (Slave mode)              */
#define  I2C_SR1_RXNE                        0x00000040U     /*!<Data Register not Empty (receivers)      */
#define  I2C_SR1_TXE                         0x00000080U     /*!<Data Register Empty (transmitters)       */
#define  I2C_SR1_BERR                        0x00000100U     /*!<Bus Error                                       */
#define  I2C_SR1_ARLO                        0x00000200U     /*!<Arbitration Lost (master mode)           */
#define  I2C_SR1_AF                          0x00000400U     /*!<Acknowledge Failure                             */
#define  I2C_SR1_OVR                         0x00000800U     /*!<Overrun/Underrun                                */
#define  I2C_SR1_PECERR                      0x00001000U     /*!<PEC Error in reception                          */
#define  I2C_SR1_TIMEOUT                     0x00004000U     /*!<Timeout or Tlow Error                           */
#define  I2C_SR1_SMBALERT                    0x00008000U     /*!<SMBus Alert                                     */

/*******************  Bit definition for I2C_SR2 register  ********************/
#define  I2C_SR2_MSL                         0x00000001U     /*!<Master/Slave                              */
#define  I2C_SR2_BUSY                        0x00000002U     /*!<Bus Busy                                  */
#define  I2C_SR2_TRA                         0x00000004U     /*!<Transmitter/Receiver                      */
#define  I2C_SR2_GENCALL                     0x00000010U     /*!<General Call Address (Slave mode)  */
#define  I2C_SR2_SMBDEFAULT                  0x00000020U     /*!<SMBus Device Default Address (Slave mode) */
#define  I2C_SR2_SMBHOST                     0x00000040U     /*!<SMBus Host Header (Slave mode)     */
#define  I2C_SR2_DUALF                       0x00000080U     /*!<Dual Flag (Slave mode)             */
#define  I2C_SR2_PEC                         0x0000FF00U     /*!<Packet Error Checking Register            */

/*******************  Bit definition for I2C_CCR register  ********************/
#define  I2C_CCR_CCR                         0x00000FFFU     /*!<Clock Control Register in Fast/Standard mode (Master mode) */
#define  I2C_CCR_DUTY                        0x00004000U     /*!<Fast Mode Duty Cycle                                       */
#define  I2C_CCR_FS                          0x00008000U     /*!<I2C Master Mode Selection                                  */

/******************  Bit definition for I2C_TRISE register  *******************/
#define  I2C_TRISE_TRISE                     0x0000003FU     /*!<Maximum Rise Time in Fast/Standard mode (Master mode) */

/******************************************************************************/
/*                                                                            */
/*                           Independent WATCHDOG                             */
/*                                                                            */
/******************************************************************************/
/*******************  Bit definition for IWDG_KR register  ********************/
#define  IWDG_KR_KEY                         0x0000FFFFU        /*!<Key value (write only, read 0000h) */

/*******************  Bit definition for IWDG_PR register  ********************/
#define  IWDG_PR_PR                          0x00000007U        /*!<PR[2:0] (Prescaler divider) */
#define  IWDG_PR_PR_0                        0x00000001U        /*!<Bit 0 */
#define  IWDG_PR_PR_1                        0x00000002U        /*!<Bit 1 */
#define  IWDG_PR_PR_2                        0x00000004U        /*!<Bit 2 */

/*******************  Bit definition for IWDG_RLR register  *******************/
#define  IWDG_RLR_RL                         0x00000FFFU        /*!<Watchdog counter reload value */

/*******************  Bit definition for IWDG_SR register  ********************/
#define  IWDG_SR_PVU                         0x00000001U        /*!<Watchdog prescaler value update */
#define  IWDG_SR_RVU                         0x00000002U        /*!<Watchdog counter reload value update */

/******************************************************************************/
/*                                                                            */
/*                             Power Control                                  */
/*                                                                            */
/******************************************************************************/
/********************  Bit definition for PWR_CR register  ********************/
#define  PWR_CR_LPDS                         0x00000001U     /*!< Low-Power Deepsleep                 */
#define  PWR_CR_PDDS                         0x00000002U     /*!< Power Down Deepsleep                */
#define  PWR_CR_CWUF                         0x00000004U     /*!< Clear Wakeup Flag                   */
#define  PWR_CR_CSBF                         0x00000008U     /*!< Clear Standby Flag                  */
#define  PWR_CR_PVDE                         0x00000010U     /*!< Power Voltage Detector Enable       */

#define  PWR_CR_PLS                          0x000000E0U     /*!< PLS[2:0] bits (PVD Level Selection) */
#define  PWR_CR_PLS_0                        0x00000020U     /*!< Bit 0 */
#define  PWR_CR_PLS_1                        0x00000040U     /*!< Bit 1 */
#define  PWR_CR_PLS_2                        0x00000080U     /*!< Bit 2 */

/*!< PVD level configuration */
#define  PWR_CR_PLS_LEV0                     0x00000000U     /*!< PVD level 0 */
#define  PWR_CR_PLS_LEV1                     0x00000020U     /*!< PVD level 1 */
#define  PWR_CR_PLS_LEV2                     0x00000040U     /*!< PVD level 2 */
#define  PWR_CR_PLS_LEV3                     0x00000060U     /*!< PVD level 3 */
#define  PWR_CR_PLS_LEV4                     0x00000080U     /*!< PVD level 4 */
#define  PWR_CR_PLS_LEV5                     0x000000A0U     /*!< PVD level 5 */
#define  PWR_CR_PLS_LEV6                     0x000000C0U     /*!< PVD level 6 */
#define  PWR_CR_PLS_LEV7                     0x000000E0U     /*!< PVD level 7 */

#define  PWR_CR_DBP                          0x00000100U     /*!< Disable Backup Domain write protection                     */
#define  PWR_CR_FPDS                         0x00000200U     /*!< Flash power down in Stop mode                              */

/*******************  Bit definition for PWR_CSR register  ********************/
#define  PWR_CSR_WUF                         0x00000001U     /*!< Wakeup Flag                                      */
#define  PWR_CSR_SBF                         0x00000002U     /*!< Standby Flag                                     */
#define  PWR_CSR_PVDO                        0x00000004U     /*!< PVD Output                                       */
#define  PWR_CSR_BRR                         0x00000008U     /*!< Backup regulator ready                           */
#define  PWR_CSR_EWUP                        0x00000100U     /*!< Enable WKUP pin                                  */
#define  PWR_CSR_BRE                         0x00000200U     /*!< Backup regulator enable                          */

/******************************************************************************/
/*                                                                            */
/*                         Reset and Clock Control                            */
/*                                                                            */
/******************************************************************************/
/********************  Bit definition for RCC_CR register  ********************/
#define  RCC_CR_HSION                        0x00000001U
#define  RCC_CR_HSIRDY                       0x00000002U

#define  RCC_CR_HSITRIM                      0x000000F8U
#define  RCC_CR_HSITRIM_0                    0x00000008U/*!<Bit 0 */
#define  RCC_CR_HSITRIM_1                    0x00000010U/*!<Bit 1 */
#define  RCC_CR_HSITRIM_2                    0x00000020U/*!<Bit 2 */
#define  RCC_CR_HSITRIM_3                    0x00000040U/*!<Bit 3 */
#define  RCC_CR_HSITRIM_4                    0x00000080U/*!<Bit 4 */

#define  RCC_CR_HSICAL                       0x0000FF00U
#define  RCC_CR_HSICAL_0                     0x00000100U/*!<Bit 0 */
#define  RCC_CR_HSICAL_1                     0x00000200U/*!<Bit 1 */
#define  RCC_CR_HSICAL_2                     0x00000400U/*!<Bit 2 */
#define  RCC_CR_HSICAL_3                     0x00000800U/*!<Bit 3 */
#define  RCC_CR_HSICAL_4                     0x00001000U/*!<Bit 4 */
#define  RCC_CR_HSICAL_5                     0x00002000U/*!<Bit 5 */
#define  RCC_CR_HSICAL_6                     0x00004000U/*!<Bit 6 */
#define  RCC_CR_HSICAL_7                     0x00008000U/*!<Bit 7 */

#define  RCC_CR_HSEON                        0x00010000U
#define  RCC_CR_HSERDY                       0x00020000U
#define  RCC_CR_HSEBYP                       0x00040000U
#define  RCC_CR_CSSON                        0x00080000U
#define  RCC_CR_PLLON                        0x01000000U
#define  RCC_CR_PLLRDY                       0x02000000U
#define  RCC_CR_PLLI2SON                     0x04000000U
#define  RCC_CR_PLLI2SRDY                    0x08000000U

/********************  Bit definition for RCC_PLLCFGR register  ***************/
#define  RCC_PLLCFGR_PLLM                    0x0000003FU
#define  RCC_PLLCFGR_PLLM_0                  0x00000001U
#define  RCC_PLLCFGR_PLLM_1                  0x00000002U
#define  RCC_PLLCFGR_PLLM_2                  0x00000004U
#define  RCC_PLLCFGR_PLLM_3                  0x00000008U
#define  RCC_PLLCFGR_PLLM_4                  0x00000010U
#define  RCC_PLLCFGR_PLLM_5                  0x00000020U

#define  RCC_PLLCFGR_PLLN                     0x00007FC0U
#define  RCC_PLLCFGR_PLLN_0                   0x00000040U
#define  RCC_PLLCFGR_PLLN_1                   0x00000080U
#define  RCC_PLLCFGR_PLLN_2                   0x00000100U
#define  RCC_PLLCFGR_PLLN_3                   0x00000200U
#define  RCC_PLLCFGR_PLLN_4                   0x00000400U
#define  RCC_PLLCFGR_PLLN_5                   0x00000800U
#define  RCC_PLLCFGR_PLLN_6                   0x00001000U
#define  RCC_PLLCFGR_PLLN_7                   0x00002000U
#define  RCC_PLLCFGR_PLLN_8                   0x00004000U

#define  RCC_PLLCFGR_PLLP                    0x00030000U
#define  RCC_PLLCFGR_PLLP_0                  0x00010000U
#define  RCC_PLLCFGR_PLLP_1                  0x00020000U

#define  RCC_PLLCFGR_PLLSRC                  0x00400000U
#define  RCC_PLLCFGR_PLLSRC_HSE              0x00400000U
#define  RCC_PLLCFGR_PLLSRC_HSI              0x00000000U

#define  RCC_PLLCFGR_PLLQ                    0x0F000000U
#define  RCC_PLLCFGR_PLLQ_0                  0x01000000U
#define  RCC_PLLCFGR_PLLQ_1                  0x02000000U
#define  RCC_PLLCFGR_PLLQ_2                  0x04000000U
#define  RCC_PLLCFGR_PLLQ_3                  0x08000000U

/********************  Bit definition for RCC_CFGR register  ******************/
/*!< SW configuration */
#define  RCC_CFGR_SW                         0x00000003U        /*!< SW[1:0] bits (System clock Switch) */
#define  RCC_CFGR_SW_0                       0x00000001U        /*!< Bit 0 */
#define  RCC_CFGR_SW_1                       0x00000002U        /*!< Bit 1 */

#define  RCC_CFGR_SW_HSI                     0x00000000U        /*!< HSI selected as system clock */
#define  RCC_CFGR_SW_HSE                     0x00000001U        /*!< HSE selected as system clock */
#define  RCC_CFGR_SW_PLL                     0x00000002U        /*!< PLL selected as system clock */

/*!< SWS configuration */
#define  RCC_CFGR_SWS                        0x0000000CU        /*!< SWS[1:0] bits (System Clock Switch Status) */
#define  RCC_CFGR_SWS_0                      0x00000004U        /*!< Bit 0 */
#define  RCC_CFGR_SWS_1                      0x00000008U        /*!< Bit 1 */

#define  RCC_CFGR_SWS_HSI                    0x00000000U        /*!< HSI oscillator used as system clock */
#define  RCC_CFGR_SWS_HSE                    0x00000004U        /*!< HSE oscillator used as system clock */
#define  RCC_CFGR_SWS_PLL                    0x00000008U        /*!< PLL used as system clock */

/*!< HPRE configuration */
#define  RCC_CFGR_HPRE                       0x000000F0U        /*!< HPRE[3:0] bits (AHB prescaler) */
#define  RCC_CFGR_HPRE_0                     0x00000010U        /*!< Bit 0 */
#define  RCC_CFGR_HPRE_1                     0x00000020U        /*!< Bit 1 */
#define  RCC_CFGR_HPRE_2                     0x00000040U        /*!< Bit 2 */
#define  RCC_CFGR_HPRE_3                     0x00000080U        /*!< Bit 3 */

#define  RCC_CFGR_HPRE_DIV1                  0x00000000U        /*!< SYSCLK not divided */
#define  RCC_CFGR_HPRE_DIV2                  0x00000080U        /*!< SYSCLK divided by 2 */
#define  RCC_CFGR_HPRE_DIV4                  0x00000090U        /*!< SYSCLK divided by 4 */
#define  RCC_CFGR_HPRE_DIV8                  0x000000A0U        /*!< SYSCLK divided by 8 */
#define  RCC_CFGR_HPRE_DIV16                 0x000000B0U        /*!< SYSCLK divided by 16 */
#define  RCC_CFGR_HPRE_DIV64                 0x000000C0U        /*!< SYSCLK divided by 64 */
#define  RCC_CFGR_HPRE_DIV128                0x000000D0U        /*!< SYSCLK divided by 128 */
#define  RCC_CFGR_HPRE_DIV256                0x000000E0U        /*!< SYSCLK divided by 256 */
#define  RCC_CFGR_HPRE_DIV512                0x000000F0U        /*!< SYSCLK divided by 512 */

/*!< PPRE1 configuration */
#define  RCC_CFGR_PPRE1                      0x00001C00U        /*!< PRE1[2:0] bits (APB1 prescaler) */
#define  RCC_CFGR_PPRE1_0                    0x00000400U        /*!< Bit 0 */
#define  RCC_CFGR_PPRE1_1                    0x00000800U        /*!< Bit 1 */
#define  RCC_CFGR_PPRE1_2                    0x00001000U        /*!< Bit 2 */

#define  RCC_CFGR_PPRE1_DIV1                 0x00000000U        /*!< HCLK not divided */
#define  RCC_CFGR_PPRE1_DIV2                 0x00001000U        /*!< HCLK divided by 2 */
#define  RCC_CFGR_PPRE1_DIV4                 0x00001400U        /*!< HCLK divided by 4 */
#define  RCC_CFGR_PPRE1_DIV8                 0x00001800U        /*!< HCLK divided by 8 */
#define  RCC_CFGR_PPRE1_DIV16                0x00001C00U        /*!< HCLK divided by 16 */

/*!< PPRE2 configuration */
#define  RCC_CFGR_PPRE2                      0x0000E000U        /*!< PRE2[2:0] bits (APB2 prescaler) */
#define  RCC_CFGR_PPRE2_0                    0x00002000U        /*!< Bit 0 */
#define  RCC_CFGR_PPRE2_1                    0x00004000U        /*!< Bit 1 */
#define  RCC_CFGR_PPRE2_2                    0x00008000U        /*!< Bit 2 */

#define  RCC_CFGR_PPRE2_DIV1                 0x00000000U        /*!< HCLK not divided */
#define  RCC_CFGR_PPRE2_DIV2                 0x00008000U        /*!< HCLK divided by 2 */
#define  RCC_CFGR_PPRE2_DIV4                 0x0000A000U        /*!< HCLK divided by 4 */
#define  RCC_CFGR_PPRE2_DIV8                 0x0000C000U        /*!< HCLK divided by 8 */
#define  RCC_CFGR_PPRE2_DIV16                0x0000E000U        /*!< HCLK divided by 16 */

/*!< RTCPRE configuration */
#define  RCC_CFGR_RTCPRE                     0x001F0000U
#define  RCC_CFGR_RTCPRE_0                   0x00010000U
#define  RCC_CFGR_RTCPRE_1                   0x00020000U
#define  RCC_CFGR_RTCPRE_2                   0x00040000U
#define  RCC_CFGR_RTCPRE_3                   0x00080000U
#define  RCC_CFGR_RTCPRE_4                   0x00100000U

/*!< MCO1 configuration */
#define  RCC_CFGR_MCO1                       0x00600000U
#define  RCC_CFGR_MCO1_0                     0x00200000U
#define  RCC_CFGR_MCO1_1                     0x00400000U

#define  RCC_CFGR_I2SSRC                     0x00800000U

#define  RCC_CFGR_MCO1PRE                    0x07000000U
#define  RCC_CFGR_MCO1PRE_0                  0x01000000U
#define  RCC_CFGR_MCO1PRE_1                  0x02000000U
#define  RCC_CFGR_MCO1PRE_2                  0x04000000U

#define  RCC_CFGR_MCO2PRE                    0x38000000U
#define  RCC_CFGR_MCO2PRE_0                  0x08000000U
#define  RCC_CFGR_MCO2PRE_1                  0x10000000U
#define  RCC_CFGR_MCO2PRE_2                  0x20000000U

#define  RCC_CFGR_MCO2                       0xC0000000U
#define  RCC_CFGR_MCO2_0                     0x40000000U
#define  RCC_CFGR_MCO2_1                     0x80000000U

/********************  Bit definition for RCC_CIR register  *******************/
#define  RCC_CIR_LSIRDYF                     0x00000001U
#define  RCC_CIR_LSERDYF                     0x00000002U
#define  RCC_CIR_HSIRDYF                     0x00000004U
#define  RCC_CIR_HSERDYF                     0x00000008U
#define  RCC_CIR_PLLRDYF                     0x00000010U
#define  RCC_CIR_PLLI2SRDYF                  0x00000020U

#define  RCC_CIR_CSSF                        0x00000080U
#define  RCC_CIR_LSIRDYIE                    0x00000100U
#define  RCC_CIR_LSERDYIE                    0x00000200U
#define  RCC_CIR_HSIRDYIE                    0x00000400U
#define  RCC_CIR_HSERDYIE                    0x00000800U
#define  RCC_CIR_PLLRDYIE                    0x00001000U
#define  RCC_CIR_PLLI2SRDYIE                 0x00002000U

#define  RCC_CIR_LSIRDYC                     0x00010000U
#define  RCC_CIR_LSERDYC                     0x00020000U
#define  RCC_CIR_HSIRDYC                     0x00040000U
#define  RCC_CIR_HSERDYC                     0x00080000U
#define  RCC_CIR_PLLRDYC                     0x00100000U
#define  RCC_CIR_PLLI2SRDYC                  0x00200000U

#define  RCC_CIR_CSSC                        0x00800000U

/********************  Bit definition for RCC_AHB1RSTR register  **************/
#define  RCC_AHB1RSTR_GPIOARST               0x00000001U
#define  RCC_AHB1RSTR_GPIOBRST               0x00000002U
#define  RCC_AHB1RSTR_GPIOCRST               0x00000004U
#define  RCC_AHB1RSTR_GPIODRST               0x00000008U
#define  RCC_AHB1RSTR_GPIOERST               0x00000010U
#define  RCC_AHB1RSTR_GPIOFRST               0x00000020U
#define  RCC_AHB1RSTR_GPIOGRST               0x00000040U
#define  RCC_AHB1RSTR_GPIOHRST               0x00000080U
#define  RCC_AHB1RSTR_GPIOIRST               0x00000100U
#define  RCC_AHB1RSTR_CRCRST                 0x00001000U
#define  RCC_AHB1RSTR_DMA1RST                0x00200000U
#define  RCC_AHB1RSTR_DMA2RST                0x00400000U
#define  RCC_AHB1RSTR_OTGHRST                0x20000000U

/********************  Bit definition for RCC_AHB2RSTR register  **************/
#define  RCC_AHB2RSTR_RNGRST                 0x00000040U
#define  RCC_AHB2RSTR_OTGFSRST               0x00000080U

/********************  Bit definition for RCC_AHB3RSTR register  **************/

#define  RCC_AHB3RSTR_FSMCRST                0x00000001U

/********************  Bit definition for RCC_APB1RSTR register  **************/
#define  RCC_APB1RSTR_TIM2RST                0x00000001U
#define  RCC_APB1RSTR_TIM3RST                0x00000002U
#define  RCC_APB1RSTR_TIM4RST                0x00000004U
#define  RCC_APB1RSTR_TIM5RST                0x00000008U
#define  RCC_APB1RSTR_TIM6RST                0x00000010U
#define  RCC_APB1RSTR_TIM7RST                0x00000020U
#define  RCC_APB1RSTR_TIM12RST               0x00000040U
#define  RCC_APB1RSTR_TIM13RST               0x00000080U
#define  RCC_APB1RSTR_TIM14RST               0x00000100U
#define  RCC_APB1RSTR_WWDGRST                0x00000800U
#define  RCC_APB1RSTR_SPI2RST                0x00004000U
#define  RCC_APB1RSTR_SPI3RST                0x00008000U
#define  RCC_APB1RSTR_USART2RST              0x00020000U
#define  RCC_APB1RSTR_USART3RST              0x00040000U
#define  RCC_APB1RSTR_UART4RST               0x00080000U
#define  RCC_APB1RSTR_UART5RST               0x00100000U
#define  RCC_APB1RSTR_I2C1RST                0x00200000U
#define  RCC_APB1RSTR_I2C2RST                0x00400000U
#define  RCC_APB1RSTR_I2C3RST                0x00800000U
#define  RCC_APB1RSTR_CAN1RST                0x02000000U
#define  RCC_APB1RSTR_CAN2RST                0x04000000U
#define  RCC_APB1RSTR_PWRRST                 0x10000000U
#define  RCC_APB1RSTR_DACRST                 0x20000000U

/********************  Bit definition for RCC_APB2RSTR register  **************/
#define  RCC_APB2RSTR_TIM1RST                0x00000001U
#define  RCC_APB2RSTR_TIM8RST                0x00000002U
#define  RCC_APB2RSTR_USART1RST              0x00000010U
#define  RCC_APB2RSTR_USART6RST              0x00000020U
#define  RCC_APB2RSTR_ADCRST                 0x00000100U
#define  RCC_APB2RSTR_SDIORST                0x00000800U
#define  RCC_APB2RSTR_SPI1RST                0x00001000U
#define  RCC_APB2RSTR_SYSCFGRST              0x00004000U
#define  RCC_APB2RSTR_TIM9RST                0x00010000U
#define  RCC_APB2RSTR_TIM10RST               0x00020000U
#define  RCC_APB2RSTR_TIM11RST               0x00040000U

/* Old SPI1RST bit definition, maintained for legacy purpose */
#define  RCC_APB2RSTR_SPI1                   RCC_APB2RSTR_SPI1RST

/********************  Bit definition for RCC_AHB1ENR register  ***************/
#define  RCC_AHB1ENR_GPIOAEN                 0x00000001U
#define  RCC_AHB1ENR_GPIOBEN                 0x00000002U
#define  RCC_AHB1ENR_GPIOCEN                 0x00000004U
#define  RCC_AHB1ENR_GPIODEN                 0x00000008U
#define  RCC_AHB1ENR_GPIOEEN                 0x00000010U
#define  RCC_AHB1ENR_GPIOFEN                 0x00000020U
#define  RCC_AHB1ENR_GPIOGEN                 0x00000040U
#define  RCC_AHB1ENR_GPIOHEN                 0x00000080U
#define  RCC_AHB1ENR_GPIOIEN                 0x00000100U
#define  RCC_AHB1ENR_CRCEN                   0x00001000U
#define  RCC_AHB1ENR_BKPSRAMEN               0x00040000U
#define  RCC_AHB1ENR_DMA1EN                  0x00200000U
#define  RCC_AHB1ENR_DMA2EN                  0x00400000U

#define  RCC_AHB1ENR_OTGHSEN                 0x20000000U
#define  RCC_AHB1ENR_OTGHSULPIEN             0x40000000U

/********************  Bit definition for RCC_AHB2ENR register  ***************/
#define  RCC_AHB2ENR_RNGEN                   0x00000040U
#define  RCC_AHB2ENR_OTGFSEN                 0x00000080U

/********************  Bit definition for RCC_AHB3ENR register  ***************/

#define  RCC_AHB3ENR_FSMCEN                  0x00000001U

/********************  Bit definition for RCC_APB1ENR register  ***************/
#define  RCC_APB1ENR_TIM2EN                  0x00000001U
#define  RCC_APB1ENR_TIM3EN                  0x00000002U
#define  RCC_APB1ENR_TIM4EN                  0x00000004U
#define  RCC_APB1ENR_TIM5EN                  0x00000008U
#define  RCC_APB1ENR_TIM6EN                  0x00000010U
#define  RCC_APB1ENR_TIM7EN                  0x00000020U
#define  RCC_APB1ENR_TIM12EN                 0x00000040U
#define  RCC_APB1ENR_TIM13EN                 0x00000080U
#define  RCC_APB1ENR_TIM14EN                 0x00000100U
#define  RCC_APB1ENR_WWDGEN                  0x00000800U
#define  RCC_APB1ENR_SPI2EN                  0x00004000U
#define  RCC_APB1ENR_SPI3EN                  0x00008000U
#define  RCC_APB1ENR_USART2EN                0x00020000U
#define  RCC_APB1ENR_USART3EN                0x00040000U
#define  RCC_APB1ENR_UART4EN                 0x00080000U
#define  RCC_APB1ENR_UART5EN                 0x00100000U
#define  RCC_APB1ENR_I2C1EN                  0x00200000U
#define  RCC_APB1ENR_I2C2EN                  0x00400000U
#define  RCC_APB1ENR_I2C3EN                  0x00800000U
#define  RCC_APB1ENR_CAN1EN                  0x02000000U
#define  RCC_APB1ENR_CAN2EN                  0x04000000U
#define  RCC_APB1ENR_PWREN                   0x10000000U
#define  RCC_APB1ENR_DACEN                   0x20000000U

/********************  Bit definition for RCC_APB2ENR register  ***************/
#define  RCC_APB2ENR_TIM1EN                  0x00000001U
#define  RCC_APB2ENR_TIM8EN                  0x00000002U
#define  RCC_APB2ENR_USART1EN                0x00000010U
#define  RCC_APB2ENR_USART6EN                0x00000020U
#define  RCC_APB2ENR_ADC1EN                  0x00000100U
#define  RCC_APB2ENR_ADC2EN                  0x00000200U
#define  RCC_APB2ENR_ADC3EN                  0x00000400U
#define  RCC_APB2ENR_SDIOEN                  0x00000800U
#define  RCC_APB2ENR_SPI1EN                  0x00001000U
#define  RCC_APB2ENR_SYSCFGEN                0x00004000U
#define  RCC_APB2ENR_TIM9EN                  0x00010000U
#define  RCC_APB2ENR_TIM10EN                 0x00020000U
#define  RCC_APB2ENR_TIM11EN                 0x00040000U

/********************  Bit definition for RCC_AHB1LPENR register  *************/
#define  RCC_AHB1LPENR_GPIOALPEN             0x00000001U
#define  RCC_AHB1LPENR_GPIOBLPEN             0x00000002U
#define  RCC_AHB1LPENR_GPIOCLPEN             0x00000004U
#define  RCC_AHB1LPENR_GPIODLPEN             0x00000008U
#define  RCC_AHB1LPENR_GPIOELPEN             0x00000010U
#define  RCC_AHB1LPENR_GPIOFLPEN             0x00000020U
#define  RCC_AHB1LPENR_GPIOGLPEN             0x00000040U
#define  RCC_AHB1LPENR_GPIOHLPEN             0x00000080U
#define  RCC_AHB1LPENR_GPIOILPEN             0x00000100U
#define  RCC_AHB1LPENR_CRCLPEN               0x00001000U
#define  RCC_AHB1LPENR_FLITFLPEN             0x00008000U
#define  RCC_AHB1LPENR_SRAM1LPEN             0x00010000U
#define  RCC_AHB1LPENR_SRAM2LPEN             0x00020000U
#define  RCC_AHB1LPENR_BKPSRAMLPEN           0x00040000U
#define  RCC_AHB1LPENR_DMA1LPEN              0x00200000U
#define  RCC_AHB1LPENR_DMA2LPEN              0x00400000U
#define  RCC_AHB1LPENR_OTGHSLPEN             0x20000000U
#define  RCC_AHB1LPENR_OTGHSULPILPEN         0x40000000U

/********************  Bit definition for RCC_AHB2LPENR register  *************/
#define  RCC_AHB2LPENR_RNGLPEN               0x00000040U
#define  RCC_AHB2LPENR_OTGFSLPEN             0x00000080U

/********************  Bit definition for RCC_AHB3LPENR register  *************/

#define  RCC_AHB3LPENR_FSMCLPEN              0x00000001U

/********************  Bit definition for RCC_APB1LPENR register  *************/
#define  RCC_APB1LPENR_TIM2LPEN              0x00000001U
#define  RCC_APB1LPENR_TIM3LPEN              0x00000002U
#define  RCC_APB1LPENR_TIM4LPEN              0x00000004U
#define  RCC_APB1LPENR_TIM5LPEN              0x00000008U
#define  RCC_APB1LPENR_TIM6LPEN              0x00000010U
#define  RCC_APB1LPENR_TIM7LPEN              0x00000020U
#define  RCC_APB1LPENR_TIM12LPEN             0x00000040U
#define  RCC_APB1LPENR_TIM13LPEN             0x00000080U
#define  RCC_APB1LPENR_TIM14LPEN             0x00000100U
#define  RCC_APB1LPENR_WWDGLPEN              0x00000800U
#define  RCC_APB1LPENR_SPI2LPEN              0x00004000U
#define  RCC_APB1LPENR_SPI3LPEN              0x00008000U
#define  RCC_APB1LPENR_USART2LPEN            0x00020000U
#define  RCC_APB1LPENR_USART3LPEN            0x00040000U
#define  RCC_APB1LPENR_UART4LPEN             0x00080000U
#define  RCC_APB1LPENR_UART5LPEN             0x00100000U
#define  RCC_APB1LPENR_I2C1LPEN              0x00200000U
#define  RCC_APB1LPENR_I2C2LPEN              0x00400000U
#define  RCC_APB1LPENR_I2C3LPEN              0x00800000U
#define  RCC_APB1LPENR_CAN1LPEN              0x02000000U
#define  RCC_APB1LPENR_CAN2LPEN              0x04000000U
#define  RCC_APB1LPENR_PWRLPEN               0x10000000U
#define  RCC_APB1LPENR_DACLPEN               0x20000000U

/********************  Bit definition for RCC_APB2LPENR register  *************/
#define  RCC_APB2LPENR_TIM1LPEN              0x00000001U
#define  RCC_APB2LPENR_TIM8LPEN              0x00000002U
#define  RCC_APB2LPENR_USART1LPEN            0x00000010U
#define  RCC_APB2LPENR_USART6LPEN            0x00000020U
#define  RCC_APB2LPENR_ADC1LPEN              0x00000100U
#define  RCC_APB2LPENR_ADC2LPEN              0x00000200U
#define  RCC_APB2LPENR_ADC3LPEN              0x00000400U
#define  RCC_APB2LPENR_SDIOLPEN              0x00000800U
#define  RCC_APB2LPENR_SPI1LPEN              0x00001000U
#define  RCC_APB2LPENR_SYSCFGLPEN            0x00004000U
#define  RCC_APB2LPENR_TIM9LPEN              0x00010000U
#define  RCC_APB2LPENR_TIM10LPEN             0x00020000U
#define  RCC_APB2LPENR_TIM11LPEN             0x00040000U

/********************  Bit definition for RCC_BDCR register  ******************/
#define  RCC_BDCR_LSEON                      0x00000001U
#define  RCC_BDCR_LSERDY                     0x00000002U
#define  RCC_BDCR_LSEBYP                     0x00000004U

#define  RCC_BDCR_RTCSEL                    0x00000300U
#define  RCC_BDCR_RTCSEL_0                  0x00000100U
#define  RCC_BDCR_RTCSEL_1                  0x00000200U

#define  RCC_BDCR_RTCEN                      0x00008000U
#define  RCC_BDCR_BDRST                      0x00010000U

/********************  Bit definition for RCC_CSR register  *******************/
#define  RCC_CSR_LSION                       0x00000001U
#define  RCC_CSR_LSIRDY                      0x00000002U
#define  RCC_CSR_RMVF                        0x01000000U
#define  RCC_CSR_BORRSTF                     0x02000000U
#define  RCC_CSR_PADRSTF                     0x04000000U
#define  RCC_CSR_PORRSTF                     0x08000000U
#define  RCC_CSR_SFTRSTF                     0x10000000U
#define  RCC_CSR_WDGRSTF                     0x20000000U
#define  RCC_CSR_WWDGRSTF                    0x40000000U
#define  RCC_CSR_LPWRRSTF                    0x80000000U

/********************  Bit definition for RCC_SSCGR register  *****************/
#define  RCC_SSCGR_MODPER                    0x00001FFFU
#define  RCC_SSCGR_INCSTEP                   0x0FFFE000U
#define  RCC_SSCGR_SPREADSEL                 0x40000000U
#define  RCC_SSCGR_SSCGEN                    0x80000000U

/********************  Bit definition for RCC_PLLI2SCFGR register  ************/
#define  RCC_PLLI2SCFGR_PLLI2SN              0x00007FC0U
#define  RCC_PLLI2SCFGR_PLLI2SN_0            0x00000040U
#define  RCC_PLLI2SCFGR_PLLI2SN_1            0x00000080U
#define  RCC_PLLI2SCFGR_PLLI2SN_2            0x00000100U
#define  RCC_PLLI2SCFGR_PLLI2SN_3            0x00000200U
#define  RCC_PLLI2SCFGR_PLLI2SN_4            0x00000400U
#define  RCC_PLLI2SCFGR_PLLI2SN_5            0x00000800U
#define  RCC_PLLI2SCFGR_PLLI2SN_6            0x00001000U
#define  RCC_PLLI2SCFGR_PLLI2SN_7            0x00002000U
#define  RCC_PLLI2SCFGR_PLLI2SN_8            0x00004000U

#define  RCC_PLLI2SCFGR_PLLI2SR              0x70000000U
#define  RCC_PLLI2SCFGR_PLLI2SR_0            0x10000000U
#define  RCC_PLLI2SCFGR_PLLI2SR_1            0x20000000U
#define  RCC_PLLI2SCFGR_PLLI2SR_2            0x40000000U

/******************************************************************************/
/*                                                                            */
/*                                    RNG                                     */
/*                                                                            */
/******************************************************************************/
/********************  Bits definition for RNG_CR register  *******************/
#define RNG_CR_RNGEN                         0x00000004U
#define RNG_CR_IE                            0x00000008U

/********************  Bits definition for RNG_SR register  *******************/
#define RNG_SR_DRDY                          0x00000001U
#define RNG_SR_CECS                          0x00000002U
#define RNG_SR_SECS                          0x00000004U
#define RNG_SR_CEIS                          0x00000020U
#define RNG_SR_SEIS                          0x00000040U

/******************************************************************************/
/*                                                                            */
/*                           Real-Time Clock (RTC)                            */
/*                                                                            */
/******************************************************************************/
/********************  Bits definition for RTC_TR register  *******************/
#define RTC_TR_PM                            0x00400000U
#define RTC_TR_HT                            0x00300000U
#define RTC_TR_HT_0                          0x00100000U
#define RTC_TR_HT_1                          0x00200000U
#define RTC_TR_HU                            0x000F0000U
#define RTC_TR_HU_0                          0x00010000U
#define RTC_TR_HU_1                          0x00020000U
#define RTC_TR_HU_2                          0x00040000U
#define RTC_TR_HU_3                          0x00080000U
#define RTC_TR_MNT                           0x00007000U
#define RTC_TR_MNT_0                         0x00001000U
#define RTC_TR_MNT_1                         0x00002000U
#define RTC_TR_MNT_2                         0x00004000U
#define RTC_TR_MNU                           0x00000F00U
#define RTC_TR_MNU_0                         0x00000100U
#define RTC_TR_MNU_1                         0x00000200U
#define RTC_TR_MNU_2                         0x00000400U
#define RTC_TR_MNU_3                         0x00000800U
#define RTC_TR_ST                            0x00000070U
#define RTC_TR_ST_0                          0x00000010U
#define RTC_TR_ST_1                          0x00000020U
#define RTC_TR_ST_2                          0x00000040U
#define RTC_TR_SU                            0x0000000FU
#define RTC_TR_SU_0                          0x00000001U
#define RTC_TR_SU_1                          0x00000002U
#define RTC_TR_SU_2                          0x00000004U
#define RTC_TR_SU_3                          0x00000008U

/********************  Bits definition for RTC_DR register  *******************/
#define RTC_DR_YT                            0x00F00000U
#define RTC_DR_YT_0                          0x00100000U
#define RTC_DR_YT_1                          0x00200000U
#define RTC_DR_YT_2                          0x00400000U
#define RTC_DR_YT_3                          0x00800000U
#define RTC_DR_YU                            0x000F0000U
#define RTC_DR_YU_0                          0x00010000U
#define RTC_DR_YU_1                          0x00020000U
#define RTC_DR_YU_2                          0x00040000U
#define RTC_DR_YU_3                          0x00080000U
#define RTC_DR_WDU                           0x0000E000U
#define RTC_DR_WDU_0                         0x00002000U
#define RTC_DR_WDU_1                         0x00004000U
#define RTC_DR_WDU_2                         0x00008000U
#define RTC_DR_MT                            0x00001000U
#define RTC_DR_MU                            0x00000F00U
#define RTC_DR_MU_0                          0x00000100U
#define RTC_DR_MU_1                          0x00000200U
#define RTC_DR_MU_2                          0x00000400U
#define RTC_DR_MU_3                          0x00000800U
#define RTC_DR_DT                            0x00000030U
#define RTC_DR_DT_0                          0x00000010U
#define RTC_DR_DT_1                          0x00000020U
#define RTC_DR_DU                            0x0000000FU
#define RTC_DR_DU_0                          0x00000001U
#define RTC_DR_DU_1                          0x00000002U
#define RTC_DR_DU_2                          0x00000004U
#define RTC_DR_DU_3                          0x00000008U

/********************  Bits definition for RTC_CR register  *******************/
#define RTC_CR_COE                           0x00800000U
#define RTC_CR_OSEL                          0x00600000U
#define RTC_CR_OSEL_0                        0x00200000U
#define RTC_CR_OSEL_1                        0x00400000U
#define RTC_CR_POL                           0x00100000U
#define RTC_CR_BCK                           0x00040000U
#define RTC_CR_SUB1H                         0x00020000U
#define RTC_CR_ADD1H                         0x00010000U
#define RTC_CR_TSIE                          0x00008000U
#define RTC_CR_WUTIE                         0x00004000U
#define RTC_CR_ALRBIE                        0x00002000U
#define RTC_CR_ALRAIE                        0x00001000U
#define RTC_CR_TSE                           0x00000800U
#define RTC_CR_WUTE                          0x00000400U
#define RTC_CR_ALRBE                         0x00000200U
#define RTC_CR_ALRAE                         0x00000100U
#define RTC_CR_DCE                           0x00000080U
#define RTC_CR_FMT                           0x00000040U
#define RTC_CR_REFCKON                       0x00000010U
#define RTC_CR_TSEDGE                        0x00000008U
#define RTC_CR_WUCKSEL                       0x00000007U
#define RTC_CR_WUCKSEL_0                     0x00000001U
#define RTC_CR_WUCKSEL_1                     0x00000002U
#define RTC_CR_WUCKSEL_2                     0x00000004U

/********************  Bits definition for RTC_ISR register  ******************/
#define RTC_ISR_TAMP1F                       0x00002000U
#define RTC_ISR_TSOVF                        0x00001000U
#define RTC_ISR_TSF                          0x00000800U
#define RTC_ISR_WUTF                         0x00000400U
#define RTC_ISR_ALRBF                        0x00000200U
#define RTC_ISR_ALRAF                        0x00000100U
#define RTC_ISR_INIT                         0x00000080U
#define RTC_ISR_INITF                        0x00000040U
#define RTC_ISR_RSF                          0x00000020U
#define RTC_ISR_INITS                        0x00000010U
#define RTC_ISR_WUTWF                        0x00000004U
#define RTC_ISR_ALRBWF                       0x00000002U
#define RTC_ISR_ALRAWF                       0x00000001U

/********************  Bits definition for RTC_PRER register  *****************/
#define RTC_PRER_PREDIV_A                    0x007F0000U
#define RTC_PRER_PREDIV_S                    0x00001FFFU

/********************  Bits definition for RTC_WUTR register  *****************/
#define RTC_WUTR_WUT                         0x0000FFFFU

/********************  Bits definition for RTC_CALIBR register  ***************/
#define RTC_CALIBR_DCS                       0x00000080U
#define RTC_CALIBR_DC                        0x0000001FU

/********************  Bits definition for RTC_ALRMAR register  ***************/
#define RTC_ALRMAR_MSK4                      0x80000000U
#define RTC_ALRMAR_WDSEL                     0x40000000U
#define RTC_ALRMAR_DT                        0x30000000U
#define RTC_ALRMAR_DT_0                      0x10000000U
#define RTC_ALRMAR_DT_1                      0x20000000U
#define RTC_ALRMAR_DU                        0x0F000000U
#define RTC_ALRMAR_DU_0                      0x01000000U
#define RTC_ALRMAR_DU_1                      0x02000000U
#define RTC_ALRMAR_DU_2                      0x04000000U
#define RTC_ALRMAR_DU_3                      0x08000000U
#define RTC_ALRMAR_MSK3                      0x00800000U
#define RTC_ALRMAR_PM                        0x00400000U
#define RTC_ALRMAR_HT                        0x00300000U
#define RTC_ALRMAR_HT_0                      0x00100000U
#define RTC_ALRMAR_HT_1                      0x00200000U
#define RTC_ALRMAR_HU                        0x000F0000U
#define RTC_ALRMAR_HU_0                      0x00010000U
#define RTC_ALRMAR_HU_1                      0x00020000U
#define RTC_ALRMAR_HU_2                      0x00040000U
#define RTC_ALRMAR_HU_3                      0x00080000U
#define RTC_ALRMAR_MSK2                      0x00008000U
#define RTC_ALRMAR_MNT                       0x00007000U
#define RTC_ALRMAR_MNT_0                     0x00001000U
#define RTC_ALRMAR_MNT_1                     0x00002000U
#define RTC_ALRMAR_MNT_2                     0x00004000U
#define RTC_ALRMAR_MNU                       0x00000F00U
#define RTC_ALRMAR_MNU_0                     0x00000100U
#define RTC_ALRMAR_MNU_1                     0x00000200U
#define RTC_ALRMAR_MNU_2                     0x00000400U
#define RTC_ALRMAR_MNU_3                     0x00000800U
#define RTC_ALRMAR_MSK1                      0x00000080U
#define RTC_ALRMAR_ST                        0x00000070U
#define RTC_ALRMAR_ST_0                      0x00000010U
#define RTC_ALRMAR_ST_1                      0x00000020U
#define RTC_ALRMAR_ST_2                      0x00000040U
#define RTC_ALRMAR_SU                        0x0000000FU
#define RTC_ALRMAR_SU_0                      0x00000001U
#define RTC_ALRMAR_SU_1                      0x00000002U
#define RTC_ALRMAR_SU_2                      0x00000004U
#define RTC_ALRMAR_SU_3                      0x00000008U

/********************  Bits definition for RTC_ALRMBR register  ***************/
#define RTC_ALRMBR_MSK4                      0x80000000U
#define RTC_ALRMBR_WDSEL                     0x40000000U
#define RTC_ALRMBR_DT                        0x30000000U
#define RTC_ALRMBR_DT_0                      0x10000000U
#define RTC_ALRMBR_DT_1                      0x20000000U
#define RTC_ALRMBR_DU                        0x0F000000U
#define RTC_ALRMBR_DU_0                      0x01000000U
#define RTC_ALRMBR_DU_1                      0x02000000U
#define RTC_ALRMBR_DU_2                      0x04000000U
#define RTC_ALRMBR_DU_3                      0x08000000U
#define RTC_ALRMBR_MSK3                      0x00800000U
#define RTC_ALRMBR_PM                        0x00400000U
#define RTC_ALRMBR_HT                        0x00300000U
#define RTC_ALRMBR_HT_0                      0x00100000U
#define RTC_ALRMBR_HT_1                      0x00200000U
#define RTC_ALRMBR_HU                        0x000F0000U
#define RTC_ALRMBR_HU_0                      0x00010000U
#define RTC_ALRMBR_HU_1                      0x00020000U
#define RTC_ALRMBR_HU_2                      0x00040000U
#define RTC_ALRMBR_HU_3                      0x00080000U
#define RTC_ALRMBR_MSK2                      0x00008000U
#define RTC_ALRMBR_MNT                       0x00007000U
#define RTC_ALRMBR_MNT_0                     0x00001000U
#define RTC_ALRMBR_MNT_1                     0x00002000U
#define RTC_ALRMBR_MNT_2                     0x00004000U
#define RTC_ALRMBR_MNU                       0x00000F00U
#define RTC_ALRMBR_MNU_0                     0x00000100U
#define RTC_ALRMBR_MNU_1                     0x00000200U
#define RTC_ALRMBR_MNU_2                     0x00000400U
#define RTC_ALRMBR_MNU_3                     0x00000800U
#define RTC_ALRMBR_MSK1                      0x00000080U
#define RTC_ALRMBR_ST                        0x00000070U
#define RTC_ALRMBR_ST_0                      0x00000010U
#define RTC_ALRMBR_ST_1                      0x00000020U
#define RTC_ALRMBR_ST_2                      0x00000040U
#define RTC_ALRMBR_SU                        0x0000000FU
#define RTC_ALRMBR_SU_0                      0x00000001U
#define RTC_ALRMBR_SU_1                      0x00000002U
#define RTC_ALRMBR_SU_2                      0x00000004U
#define RTC_ALRMBR_SU_3                      0x00000008U

/********************  Bits definition for RTC_WPR register  ******************/
#define RTC_WPR_KEY                          0x000000FFU

/********************  Bits definition for RTC_TSTR register  *****************/
#define RTC_TSTR_PM                          0x00400000U
#define RTC_TSTR_HT                          0x00300000U
#define RTC_TSTR_HT_0                        0x00100000U
#define RTC_TSTR_HT_1                        0x00200000U
#define RTC_TSTR_HU                          0x000F0000U
#define RTC_TSTR_HU_0                        0x00010000U
#define RTC_TSTR_HU_1                        0x00020000U
#define RTC_TSTR_HU_2                        0x00040000U
#define RTC_TSTR_HU_3                        0x00080000U
#define RTC_TSTR_MNT                         0x00007000U
#define RTC_TSTR_MNT_0                       0x00001000U
#define RTC_TSTR_MNT_1                       0x00002000U
#define RTC_TSTR_MNT_2                       0x00004000U
#define RTC_TSTR_MNU                         0x00000F00U
#define RTC_TSTR_MNU_0                       0x00000100U
#define RTC_TSTR_MNU_1                       0x00000200U
#define RTC_TSTR_MNU_2                       0x00000400U
#define RTC_TSTR_MNU_3                       0x00000800U
#define RTC_TSTR_ST                          0x00000070U
#define RTC_TSTR_ST_0                        0x00000010U
#define RTC_TSTR_ST_1                        0x00000020U
#define RTC_TSTR_ST_2                        0x00000040U
#define RTC_TSTR_SU                          0x0000000FU
#define RTC_TSTR_SU_0                        0x00000001U
#define RTC_TSTR_SU_1                        0x00000002U
#define RTC_TSTR_SU_2                        0x00000004U
#define RTC_TSTR_SU_3                        0x00000008U

/********************  Bits definition for RTC_TSDR register  *****************/
#define RTC_TSDR_WDU                         0x0000E000U
#define RTC_TSDR_WDU_0                       0x00002000U
#define RTC_TSDR_WDU_1                       0x00004000U
#define RTC_TSDR_WDU_2                       0x00008000U
#define RTC_TSDR_MT                          0x00001000U
#define RTC_TSDR_MU                          0x00000F00U
#define RTC_TSDR_MU_0                        0x00000100U
#define RTC_TSDR_MU_1                        0x00000200U
#define RTC_TSDR_MU_2                        0x00000400U
#define RTC_TSDR_MU_3                        0x00000800U
#define RTC_TSDR_DT                          0x00000030U
#define RTC_TSDR_DT_0                        0x00000010U
#define RTC_TSDR_DT_1                        0x00000020U
#define RTC_TSDR_DU                          0x0000000FU
#define RTC_TSDR_DU_0                        0x00000001U
#define RTC_TSDR_DU_1                        0x00000002U
#define RTC_TSDR_DU_2                        0x00000004U
#define RTC_TSDR_DU_3                        0x00000008U

/********************  Bits definition for RTC_TAFCR register  ****************/
#define RTC_TAFCR_ALARMOUTTYPE               0x00040000U
#define RTC_TAFCR_TSINSEL                    0x00020000U
#define RTC_TAFCR_TAMPINSEL                  0x00010000U
#define RTC_TAFCR_TAMPIE                     0x00000004U
#define RTC_TAFCR_TAMP1TRG                   0x00000002U
#define RTC_TAFCR_TAMP1E                     0x00000001U

/********************  Bits definition for RTC_BKP0R register  ****************/
#define RTC_BKP0R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP1R register  ****************/
#define RTC_BKP1R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP2R register  ****************/
#define RTC_BKP2R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP3R register  ****************/
#define RTC_BKP3R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP4R register  ****************/
#define RTC_BKP4R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP5R register  ****************/
#define RTC_BKP5R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP6R register  ****************/
#define RTC_BKP6R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP7R register  ****************/
#define RTC_BKP7R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP8R register  ****************/
#define RTC_BKP8R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP9R register  ****************/
#define RTC_BKP9R                            0xFFFFFFFFU

/********************  Bits definition for RTC_BKP10R register  ***************/
#define RTC_BKP10R                           0xFFFFFFFFU

/********************  Bits definition for RTC_BKP11R register  ***************/
#define RTC_BKP11R                           0xFFFFFFFFU

/********************  Bits definition for RTC_BKP12R register  ***************/
#define RTC_BKP12R                           0xFFFFFFFFU

/********************  Bits definition for RTC_BKP13R register  ***************/
#define RTC_BKP13R                           0xFFFFFFFFU

/********************  Bits definition for RTC_BKP14R register  ***************/
#define RTC_BKP14R                           0xFFFFFFFFU

/********************  Bits definition for RTC_BKP15R register  ***************/
#define RTC_BKP15R                           0xFFFFFFFFU

/********************  Bits definition for RTC_BKP16R register  ***************/
#define RTC_BKP16R                           0xFFFFFFFFU

/********************  Bits definition for RTC_BKP17R register  ***************/
#define RTC_BKP17R                           0xFFFFFFFFU

/********************  Bits definition for RTC_BKP18R register  ***************/
#define RTC_BKP18R                           0xFFFFFFFFU

/********************  Bits definition for RTC_BKP19R register  ***************/
#define RTC_BKP19R                           0xFFFFFFFFU



/******************************************************************************/
/*                                                                            */
/*                          SD host Interface                                 */
/*                                                                            */
/******************************************************************************/
/******************  Bit definition for SDIO_POWER register  ******************/
#define  SDIO_POWER_PWRCTRL                  0x00000003U        /*!<PWRCTRL[1:0] bits (Power supply control bits) */
#define  SDIO_POWER_PWRCTRL_0                0x00000001U        /*!<Bit 0 */
#define  SDIO_POWER_PWRCTRL_1                0x00000002U        /*!<Bit 1 */

/******************  Bit definition for SDIO_CLKCR register  ******************/
#define  SDIO_CLKCR_CLKDIV                   0x000000FFU        /*!<Clock divide factor */
#define  SDIO_CLKCR_CLKEN                    0x00000100U        /*!<Clock enable bit */
#define  SDIO_CLKCR_PWRSAV                   0x00000200U        /*!<Power saving configuration bit */
#define  SDIO_CLKCR_BYPASS                   0x00000400U        /*!<Clock divider bypass enable bit */

#define  SDIO_CLKCR_WIDBUS                   0x00001800U        /*!<WIDBUS[1:0] bits (Wide bus mode enable bit) */
#define  SDIO_CLKCR_WIDBUS_0                 0x00000800U        /*!<Bit 0 */
#define  SDIO_CLKCR_WIDBUS_1                 0x00001000U        /*!<Bit 1 */

#define  SDIO_CLKCR_NEGEDGE                  0x00002000U        /*!<SDIO_CK dephasing selection bit */
#define  SDIO_CLKCR_HWFC_EN                  0x00004000U        /*!<HW Flow Control enable */

/*******************  Bit definition for SDIO_ARG register  *******************/
#define  SDIO_ARG_CMDARG                     0xFFFFFFFFU            /*!<Command argument */

/*******************  Bit definition for SDIO_CMD register  *******************/
#define  SDIO_CMD_CMDINDEX                   0x0000003FU        /*!<Command Index */

#define  SDIO_CMD_WAITRESP                   0x000000C0U        /*!<WAITRESP[1:0] bits (Wait for response bits) */
#define  SDIO_CMD_WAITRESP_0                 0x00000040U        /*!< Bit 0 */
#define  SDIO_CMD_WAITRESP_1                 0x00000080U        /*!< Bit 1 */

#define  SDIO_CMD_WAITINT                    0x00000100U        /*!<CPSM Waits for Interrupt Request */
#define  SDIO_CMD_WAITPEND                   0x00000200U        /*!<CPSM Waits for ends of data transfer (CmdPend internal signal) */
#define  SDIO_CMD_CPSMEN                     0x00000400U        /*!<Command path state machine (CPSM) Enable bit */
#define  SDIO_CMD_SDIOSUSPEND                0x00000800U        /*!<SD I/O suspend command */
#define  SDIO_CMD_ENCMDCOMPL                 0x00001000U        /*!<Enable CMD completion */
#define  SDIO_CMD_NIEN                       0x00002000U        /*!<Not Interrupt Enable */
#define  SDIO_CMD_CEATACMD                   0x00004000U        /*!<CE-ATA command */

/*****************  Bit definition for SDIO_RESPCMD register  *****************/
#define  SDIO_RESPCMD_RESPCMD                0x0000003FU        /*!<Response command index */

/******************  Bit definition for SDIO_RESP0 register  ******************/
#define  SDIO_RESP0_CARDSTATUS0              0xFFFFFFFFU        /*!<Card Status */

/******************  Bit definition for SDIO_RESP1 register  ******************/
#define  SDIO_RESP1_CARDSTATUS1              0xFFFFFFFFU        /*!<Card Status */

/******************  Bit definition for SDIO_RESP2 register  ******************/
#define  SDIO_RESP2_CARDSTATUS2              0xFFFFFFFFU        /*!<Card Status */

/******************  Bit definition for SDIO_RESP3 register  ******************/
#define  SDIO_RESP3_CARDSTATUS3              0xFFFFFFFFU        /*!<Card Status */

/******************  Bit definition for SDIO_RESP4 register  ******************/
#define  SDIO_RESP4_CARDSTATUS4              0xFFFFFFFFU        /*!<Card Status */

/******************  Bit definition for SDIO_DTIMER register  *****************/
#define  SDIO_DTIMER_DATATIME                0xFFFFFFFFU        /*!<Data timeout period. */

/******************  Bit definition for SDIO_DLEN register  *******************/
#define  SDIO_DLEN_DATALENGTH                0x01FFFFFFU        /*!<Data length value    */

/******************  Bit definition for SDIO_DCTRL register  ******************/
#define  SDIO_DCTRL_DTEN                     0x00000001U        /*!<Data transfer enabled bit */
#define  SDIO_DCTRL_DTDIR                    0x00000002U        /*!<Data transfer direction selection */
#define  SDIO_DCTRL_DTMODE                   0x00000004U        /*!<Data transfer mode selection */
#define  SDIO_DCTRL_DMAEN                    0x00000008U        /*!<DMA enabled bit */

#define  SDIO_DCTRL_DBLOCKSIZE               0x000000F0U        /*!<DBLOCKSIZE[3:0] bits (Data block size) */
#define  SDIO_DCTRL_DBLOCKSIZE_0             0x00000010U        /*!<Bit 0 */
#define  SDIO_DCTRL_DBLOCKSIZE_1             0x00000020U        /*!<Bit 1 */
#define  SDIO_DCTRL_DBLOCKSIZE_2             0x00000040U        /*!<Bit 2 */
#define  SDIO_DCTRL_DBLOCKSIZE_3             0x00000080U        /*!<Bit 3 */

#define  SDIO_DCTRL_RWSTART                  0x00000100U        /*!<Read wait start */
#define  SDIO_DCTRL_RWSTOP                   0x00000200U        /*!<Read wait stop */
#define  SDIO_DCTRL_RWMOD                    0x00000400U        /*!<Read wait mode */
#define  SDIO_DCTRL_SDIOEN                   0x00000800U        /*!<SD I/O enable functions */

/******************  Bit definition for SDIO_DCOUNT register  *****************/
#define  SDIO_DCOUNT_DATACOUNT               0x01FFFFFFU        /*!<Data count value */

/******************  Bit definition for SDIO_STA register  ********************/
#define  SDIO_STA_CCRCFAIL                   0x00000001U        /*!<Command response received (CRC check failed)  */
#define  SDIO_STA_DCRCFAIL                   0x00000002U        /*!<Data block sent/received (CRC check failed)   */
#define  SDIO_STA_CTIMEOUT                   0x00000004U        /*!<Command response timeout                      */
#define  SDIO_STA_DTIMEOUT                   0x00000008U        /*!<Data timeout                                  */
#define  SDIO_STA_TXUNDERR                   0x00000010U        /*!<Transmit FIFO underrun error                  */
#define  SDIO_STA_RXOVERR                    0x00000020U        /*!<Received FIFO overrun error                   */
#define  SDIO_STA_CMDREND                    0x00000040U        /*!<Command response received (CRC check passed)  */
#define  SDIO_STA_CMDSENT                    0x00000080U        /*!<Command sent (no response required)           */
#define  SDIO_STA_DATAEND                    0x00000100U        /*!<Data end (data counter, SDIDCOUNT, is zero)   */
#define  SDIO_STA_STBITERR                   0x00000200U        /*!<Start bit not detected on all data signals in wide bus mode */
#define  SDIO_STA_DBCKEND                    0x00000400U        /*!<Data block sent/received (CRC check passed)   */
#define  SDIO_STA_CMDACT                     0x00000800U        /*!<Command transfer in progress                  */
#define  SDIO_STA_TXACT                      0x00001000U        /*!<Data transmit in progress                     */
#define  SDIO_STA_RXACT                      0x00002000U        /*!<Data receive in progress                      */
#define  SDIO_STA_TXFIFOHE                   0x00004000U        /*!<Transmit FIFO Half Empty: at least 8 words can be written into the FIFO */
#define  SDIO_STA_RXFIFOHF                   0x00008000U        /*!<Receive FIFO Half Full: there are at least 8 words in the FIFO */
#define  SDIO_STA_TXFIFOF                    0x00010000U        /*!<Transmit FIFO full                            */
#define  SDIO_STA_RXFIFOF                    0x00020000U        /*!<Receive FIFO full                             */
#define  SDIO_STA_TXFIFOE                    0x00040000U        /*!<Transmit FIFO empty                           */
#define  SDIO_STA_RXFIFOE                    0x00080000U        /*!<Receive FIFO empty                            */
#define  SDIO_STA_TXDAVL                     0x00100000U        /*!<Data available in transmit FIFO               */
#define  SDIO_STA_RXDAVL                     0x00200000U        /*!<Data available in receive FIFO                */
#define  SDIO_STA_SDIOIT                     0x00400000U        /*!<SDIO interrupt received                       */
#define  SDIO_STA_CEATAEND                   0x00800000U        /*!<CE-ATA command completion signal received for CMD61 */

/*******************  Bit definition for SDIO_ICR register  *******************/
#define  SDIO_ICR_CCRCFAILC                  0x00000001U        /*!<CCRCFAIL flag clear bit */
#define  SDIO_ICR_DCRCFAILC                  0x00000002U        /*!<DCRCFAIL flag clear bit */
#define  SDIO_ICR_CTIMEOUTC                  0x00000004U        /*!<CTIMEOUT flag clear bit */
#define  SDIO_ICR_DTIMEOUTC                  0x00000008U        /*!<DTIMEOUT flag clear bit */
#define  SDIO_ICR_TXUNDERRC                  0x00000010U        /*!<TXUNDERR flag clear bit */
#define  SDIO_ICR_RXOVERRC                   0x00000020U        /*!<RXOVERR flag clear bit  */
#define  SDIO_ICR_CMDRENDC                   0x00000040U        /*!<CMDREND flag clear bit  */
#define  SDIO_ICR_CMDSENTC                   0x00000080U        /*!<CMDSENT flag clear bit  */
#define  SDIO_ICR_DATAENDC                   0x00000100U        /*!<DATAEND flag clear bit  */
#define  SDIO_ICR_STBITERRC                  0x00000200U        /*!<STBITERR flag clear bit */
#define  SDIO_ICR_DBCKENDC                   0x00000400U        /*!<DBCKEND flag clear bit  */
#define  SDIO_ICR_SDIOITC                    0x00400000U        /*!<SDIOIT flag clear bit   */
#define  SDIO_ICR_CEATAENDC                  0x00800000U        /*!<CEATAEND flag clear bit */

/******************  Bit definition for SDIO_MASK register  *******************/
#define  SDIO_MASK_CCRCFAILIE                0x00000001U        /*!<Command CRC Fail Interrupt Enable          */
#define  SDIO_MASK_DCRCFAILIE                0x00000002U        /*!<Data CRC Fail Interrupt Enable             */
#define  SDIO_MASK_CTIMEOUTIE                0x00000004U        /*!<Command TimeOut Interrupt Enable           */
#define  SDIO_MASK_DTIMEOUTIE                0x00000008U        /*!<Data TimeOut Interrupt Enable              */
#define  SDIO_MASK_TXUNDERRIE                0x00000010U        /*!<Tx FIFO UnderRun Error Interrupt Enable    */
#define  SDIO_MASK_RXOVERRIE                 0x00000020U        /*!<Rx FIFO OverRun Error Interrupt Enable     */
#define  SDIO_MASK_CMDRENDIE                 0x00000040U        /*!<Command Response Received Interrupt Enable */
#define  SDIO_MASK_CMDSENTIE                 0x00000080U        /*!<Command Sent Interrupt Enable              */
#define  SDIO_MASK_DATAENDIE                 0x00000100U        /*!<Data End Interrupt Enable                  */
#define  SDIO_MASK_STBITERRIE                0x00000200U        /*!<Start Bit Error Interrupt Enable           */
#define  SDIO_MASK_DBCKENDIE                 0x00000400U        /*!<Data Block End Interrupt Enable            */
#define  SDIO_MASK_CMDACTIE                  0x00000800U        /*!<CCommand Acting Interrupt Enable           */
#define  SDIO_MASK_TXACTIE                   0x00001000U        /*!<Data Transmit Acting Interrupt Enable      */
#define  SDIO_MASK_RXACTIE                   0x00002000U        /*!<Data receive acting interrupt enabled      */
#define  SDIO_MASK_TXFIFOHEIE                0x00004000U        /*!<Tx FIFO Half Empty interrupt Enable        */
#define  SDIO_MASK_RXFIFOHFIE                0x00008000U        /*!<Rx FIFO Half Full interrupt Enable         */
#define  SDIO_MASK_TXFIFOFIE                 0x00010000U        /*!<Tx FIFO Full interrupt Enable              */
#define  SDIO_MASK_RXFIFOFIE                 0x00020000U        /*!<Rx FIFO Full interrupt Enable              */
#define  SDIO_MASK_TXFIFOEIE                 0x00040000U        /*!<Tx FIFO Empty interrupt Enable             */
#define  SDIO_MASK_RXFIFOEIE                 0x00080000U        /*!<Rx FIFO Empty interrupt Enable             */
#define  SDIO_MASK_TXDAVLIE                  0x00100000U        /*!<Data available in Tx FIFO interrupt Enable */
#define  SDIO_MASK_RXDAVLIE                  0x00200000U        /*!<Data available in Rx FIFO interrupt Enable */
#define  SDIO_MASK_SDIOITIE                  0x00400000U        /*!<SDIO Mode Interrupt Received interrupt Enable */
#define  SDIO_MASK_CEATAENDIE                0x00800000U        /*!<CE-ATA command completion signal received Interrupt Enable */

/*****************  Bit definition for SDIO_FIFOCNT register  *****************/
#define  SDIO_FIFOCNT_FIFOCOUNT              0x00FFFFFFU        /*!<Remaining number of words to be written to or read from the FIFO */

/******************  Bit definition for SDIO_FIFO register  *******************/
#define  SDIO_FIFO_FIFODATA                  0xFFFFFFFFU        /*!<Receive and transmit FIFO data */

/******************************************************************************/
/*                                                                            */
/*                        Serial Peripheral Interface                         */
/*                                                                            */
/******************************************************************************/
/*******************  Bit definition for SPI_CR1 register  ********************/
#define  SPI_CR1_CPHA                        0x00000001U            /*!<Clock Phase      */
#define  SPI_CR1_CPOL                        0x00000002U            /*!<Clock Polarity   */
#define  SPI_CR1_MSTR                        0x00000004U            /*!<Master Selection */

#define  SPI_CR1_BR                          0x00000038U            /*!<BR[2:0] bits (Baud Rate Control) */
#define  SPI_CR1_BR_0                        0x00000008U            /*!<Bit 0 */
#define  SPI_CR1_BR_1                        0x00000010U            /*!<Bit 1 */
#define  SPI_CR1_BR_2                        0x00000020U            /*!<Bit 2 */

#define  SPI_CR1_SPE                         0x00000040U            /*!<SPI Enable                          */
#define  SPI_CR1_LSBFIRST                    0x00000080U            /*!<Frame Format                        */
#define  SPI_CR1_SSI                         0x00000100U            /*!<Internal slave select               */
#define  SPI_CR1_SSM                         0x00000200U            /*!<Software slave management           */
#define  SPI_CR1_RXONLY                      0x00000400U            /*!<Receive only                        */
#define  SPI_CR1_DFF                         0x00000800U            /*!<Data Frame Format                   */
#define  SPI_CR1_CRCNEXT                     0x00001000U            /*!<Transmit CRC next                   */
#define  SPI_CR1_CRCEN                       0x00002000U            /*!<Hardware CRC calculation enable     */
#define  SPI_CR1_BIDIOE                      0x00004000U            /*!<Output enable in bidirectional mode */
#define  SPI_CR1_BIDIMODE                    0x00008000U            /*!<Bidirectional data mode enable      */

/*******************  Bit definition for SPI_CR2 register  ********************/
#define  SPI_CR2_RXDMAEN                     0x00000001U               /*!<Rx Buffer DMA Enable                 */
#define  SPI_CR2_TXDMAEN                     0x00000002U               /*!<Tx Buffer DMA Enable                 */
#define  SPI_CR2_SSOE                        0x00000004U               /*!<SS Output Enable                     */
#define  SPI_CR2_FRF                         0x00000010U               /*!<Frame Format                         */
#define  SPI_CR2_ERRIE                       0x00000020U               /*!<Error Interrupt Enable               */
#define  SPI_CR2_RXNEIE                      0x00000040U               /*!<RX buffer Not Empty Interrupt Enable */
#define  SPI_CR2_TXEIE                       0x00000080U               /*!<Tx buffer Empty Interrupt Enable     */

/********************  Bit definition for SPI_SR register  ********************/
#define  SPI_SR_RXNE                         0x00000001U               /*!<Receive buffer Not Empty */
#define  SPI_SR_TXE                          0x00000002U               /*!<Transmit buffer Empty    */
#define  SPI_SR_CHSIDE                       0x00000004U               /*!<Channel side             */
#define  SPI_SR_UDR                          0x00000008U               /*!<Underrun flag            */
#define  SPI_SR_CRCERR                       0x00000010U               /*!<CRC Error flag           */
#define  SPI_SR_MODF                         0x00000020U               /*!<Mode fault               */
#define  SPI_SR_OVR                          0x00000040U               /*!<Overrun flag             */
#define  SPI_SR_BSY                          0x00000080U               /*!<Busy flag                */
#define  SPI_SR_FRE                          0x00000100U               /*!<Frame format error flag  */

/********************  Bit definition for SPI_DR register  ********************/
#define  SPI_DR_DR                           0x0000FFFFU            /*!<Data Register           */

/*******************  Bit definition for SPI_CRCPR register  ******************/
#define  SPI_CRCPR_CRCPOLY                   0x0000FFFFU            /*!<CRC polynomial register */

/******************  Bit definition for SPI_RXCRCR register  ******************/
#define  SPI_RXCRCR_RXCRC                    0x0000FFFFU            /*!<Rx CRC Register         */

/******************  Bit definition for SPI_TXCRCR register  ******************/
#define  SPI_TXCRCR_TXCRC                    0x0000FFFFU            /*!<Tx CRC Register         */

/******************  Bit definition for SPI_I2SCFGR register  *****************/
#define  SPI_I2SCFGR_CHLEN                   0x00000001U            /*!<Channel length (number of bits per audio channel) */

#define  SPI_I2SCFGR_DATLEN                  0x00000006U            /*!<DATLEN[1:0] bits (Data length to be transferred)  */
#define  SPI_I2SCFGR_DATLEN_0                0x00000002U            /*!<Bit 0 */
#define  SPI_I2SCFGR_DATLEN_1                0x00000004U            /*!<Bit 1 */

#define  SPI_I2SCFGR_CKPOL                   0x00000008U            /*!<steady state clock polarity               */

#define  SPI_I2SCFGR_I2SSTD                  0x00000030U            /*!<I2SSTD[1:0] bits (I2S standard selection) */
#define  SPI_I2SCFGR_I2SSTD_0                0x00000010U            /*!<Bit 0 */
#define  SPI_I2SCFGR_I2SSTD_1                0x00000020U            /*!<Bit 1 */

#define  SPI_I2SCFGR_PCMSYNC                 0x00000080U            /*!<PCM frame synchronization                 */

#define  SPI_I2SCFGR_I2SCFG                  0x00000300U            /*!<I2SCFG[1:0] bits (I2S configuration mode) */
#define  SPI_I2SCFGR_I2SCFG_0                0x00000100U            /*!<Bit 0 */
#define  SPI_I2SCFGR_I2SCFG_1                0x00000200U            /*!<Bit 1 */

#define  SPI_I2SCFGR_I2SE                    0x00000400U            /*!<I2S Enable         */
#define  SPI_I2SCFGR_I2SMOD                  0x00000800U            /*!<I2S mode selection */

/******************  Bit definition for SPI_I2SPR register  *******************/
#define  SPI_I2SPR_I2SDIV                    0x000000FFU            /*!<I2S Linear prescaler         */
#define  SPI_I2SPR_ODD                       0x00000100U            /*!<Odd factor for the prescaler */
#define  SPI_I2SPR_MCKOE                     0x00000200U            /*!<Master Clock Output Enable   */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
/******************************************************************************/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
/*                                                                            */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
/*                                 SYSCFG                                     */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
/*                                                                            */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
/******************************************************************************/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
/******************  Bit definition for SYSCFG_MEMRMP register  ***************/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#define SYSCFG_MEMRMP_MEM_MODE          0x00000003U /*!<SYSCFG_Memory Remap Config */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#define SYSCFG_MEMRMP_MEM_MODE_0        0x00000001U                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
#define SYSCFG_MEMRMP_MEM_MODE_1        0x00000002U                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
/*****************  Bit definition for SYSCFG_EXTICR1 register  ***************/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
#define SYSCFG_EXTICR1_EXTI0            0x0000000FU /*!<EXTI 0 configuration */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#define SYSCFG_EXTICR1_EXTI1            0x000000F0U /*!<EXTI 1 configuration */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#define SYSCFG_EXTICR1_EXTI2            0x00000F00U /*!<EXTI 2 configuration */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#define SYSCFG_EXTICR1_EXTI3            0x0000F000U /*!<EXTI 3 configuration */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
/**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
  * @brief   EXTI0 configuration                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
  */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#define SYSCFG_EXTICR1_EXTI0_PA         0x00000000U /*!<PA[0] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI0_PB         0x00000001U /*!<PB[0] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI0_PC         0x00000002U /*!<PC[0] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI0_PD         0x00000003U /*!<PD[0] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI0_PE         0x00000004U /*!<PE[0] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI0_PF         0x00000005U /*!<PF[0] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI0_PG         0x00000006U /*!<PG[0] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI0_PH         0x00000007U /*!<PH[0] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI0_PI         0x00000008U /*!<PI[0] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
/**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
  * @brief   EXTI1 configuration                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
  */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#define SYSCFG_EXTICR1_EXTI1_PA         0x00000000U /*!<PA[1] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI1_PB         0x00000010U /*!<PB[1] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI1_PC         0x00000020U /*!<PC[1] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI1_PD         0x00000030U /*!<PD[1] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI1_PE         0x00000040U /*!<PE[1] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI1_PF         0x00000050) /*!<PF[1] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI1_PG         0x00000060U /*!<PG[1] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI1_PH         0x00000070U /*!<PH[1] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI1_PI         0x00000080U /*!<PI[1] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
/**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
  * @brief   EXTI2 configuration                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
  */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#define SYSCFG_EXTICR1_EXTI2_PA         0x00000000U /*!<PA[2] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI2_PB         0x00000100U /*!<PB[2] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI2_PC         0x00000200U /*!<PC[2] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI2_PD         0x00000300U /*!<PD[2] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI2_PE         0x00000400U /*!<PE[2] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI2_PF         0x00000500) /*!<PF[2] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI2_PG         0x00000600) /*!<PG[2] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI2_PH         0x00000700U /*!<PH[2] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI2_PI         0x00000800U /*!<PI[2] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
/**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
  * @brief   EXTI3 configuration                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
  */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
#define SYSCFG_EXTICR1_EXTI3_PA         0x00000000U /*!<PA[3] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI3_PB         0x00001000U /*!<PB[3] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI3_PC         0x00002000U /*!<PC[3] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI3_PD         0x00003000U /*!<PD[3] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI3_PE         0x00004000U /*!<PE[3] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI3_PF         0x00005000) /*!<PF[3] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI3_PG         0x00006000U /*!<PG[3] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI3_PH         0x00007000U /*!<PH[3] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
#define SYSCFG_EXTICR1_EXTI3_PI         0x00008000U /*!<PI[3] pin */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
/*****************  Bit definition for SYSCFG_EXTICR2 register  ***************/
#define SYSCFG_EXTICR2_EXTI4            0x0000000FU /*!<EXTI 4 configuration */
#define SYSCFG_EXTICR2_EXTI5            0x000000F0U /*!<EXTI 5 configuration */
#define SYSCFG_EXTICR2_EXTI6            0x00000F00U /*!<EXTI 6 configuration */
#define SYSCFG_EXTICR2_EXTI7            0x0000F000U /*!<EXTI 7 configuration */
/** 
  * @brief   EXTI4 configuration  
  */ 
#define SYSCFG_EXTICR2_EXTI4_PA         0x00000000U /*!<PA[4] pin */
#define SYSCFG_EXTICR2_EXTI4_PB         0x00000001U /*!<PB[4] pin */
#define SYSCFG_EXTICR2_EXTI4_PC         0x00000002U /*!<PC[4] pin */
#define SYSCFG_EXTICR2_EXTI4_PD         0x00000003U /*!<PD[4] pin */
#define SYSCFG_EXTICR2_EXTI4_PE         0x00000004U /*!<PE[4] pin */
#define SYSCFG_EXTICR2_EXTI4_PF         0x00000005U /*!<PF[4] pin */
#define SYSCFG_EXTICR2_EXTI4_PG         0x00000006U /*!<PG[4] pin */
#define SYSCFG_EXTICR2_EXTI4_PH         0x00000007U /*!<PH[4] pin */
#define SYSCFG_EXTICR2_EXTI4_PI         0x00000008U /*!<PI[4] pin */
/** 
  * @brief   EXTI5 configuration  
  */ 
#define SYSCFG_EXTICR2_EXTI5_PA         0x00000000U /*!<PA[5] pin */
#define SYSCFG_EXTICR2_EXTI5_PB         0x00000010U /*!<PB[5] pin */
#define SYSCFG_EXTICR2_EXTI5_PC         0x00000020U /*!<PC[5] pin */
#define SYSCFG_EXTICR2_EXTI5_PD         0x00000030U /*!<PD[5] pin */
#define SYSCFG_EXTICR2_EXTI5_PE         0x00000040U /*!<PE[5] pin */
#define SYSCFG_EXTICR2_EXTI5_PF         0x00000050) /*!<PF[5] pin */
#define SYSCFG_EXTICR2_EXTI5_PG         0x00000060U /*!<PG[5] pin */
#define SYSCFG_EXTICR2_EXTI5_PH         0x00000070U /*!<PH[5] pin */
#define SYSCFG_EXTICR2_EXTI5_PI         0x00000080U /*!<PI[5] pin */
/** 
  * @brief   EXTI6 configuration  
  */ 
#define SYSCFG_EXTICR2_EXTI6_PA         0x00000000U /*!<PA[6] pin */
#define SYSCFG_EXTICR2_EXTI6_PB         0x00000100U /*!<PB[6] pin */
#define SYSCFG_EXTICR2_EXTI6_PC         0x00000200U /*!<PC[6] pin */
#define SYSCFG_EXTICR2_EXTI6_PD         0x00000300U /*!<PD[6] pin */
#define SYSCFG_EXTICR2_EXTI6_PE         0x00000400U /*!<PE[6] pin */
#define SYSCFG_EXTICR2_EXTI6_PF         0x00000500) /*!<PF[6] pin */
#define SYSCFG_EXTICR2_EXTI6_PG         0x00000600) /*!<PG[6] pin */
#define SYSCFG_EXTICR2_EXTI6_PH         0x00000700U /*!<PH[6] pin */
#define SYSCFG_EXTICR2_EXTI6_PI         0x00000800U /*!<PI[6] pin */
/** 
  * @brief   EXTI7 configuration  
  */ 
#define SYSCFG_EXTICR2_EXTI7_PA         0x00000000U /*!<PA[7] pin */
#define SYSCFG_EXTICR2_EXTI7_PB         0x00001000U /*!<PB[7] pin */
#define SYSCFG_EXTICR2_EXTI7_PC         0x00002000U /*!<PC[7] pin */
#define SYSCFG_EXTICR2_EXTI7_PD         0x00003000U /*!<PD[7] pin */
#define SYSCFG_EXTICR2_EXTI7_PE         0x00004000U /*!<PE[7] pin */
#define SYSCFG_EXTICR2_EXTI7_PF         0x00005000) /*!<PF[7] pin */
#define SYSCFG_EXTICR2_EXTI7_PG         0x00006000U /*!<PG[7] pin */
#define SYSCFG_EXTICR2_EXTI7_PH         0x00007000U /*!<PH[7] pin */
#define SYSCFG_EXTICR2_EXTI7_PI         0x00008000U /*!<PI[7] pin */

/*****************  Bit definition for SYSCFG_EXTICR3 register  ***************/
#define SYSCFG_EXTICR3_EXTI8            0x0000000FU /*!<EXTI 8 configuration */
#define SYSCFG_EXTICR3_EXTI9            0x000000F0U /*!<EXTI 9 configuration */
#define SYSCFG_EXTICR3_EXTI10           0x00000F00U /*!<EXTI 10 configuration */
#define SYSCFG_EXTICR3_EXTI11           0x0000F000U /*!<EXTI 11 configuration */
           
/** 
  * @brief   EXTI8 configuration  
  */ 
#define SYSCFG_EXTICR3_EXTI8_PA         0x00000000U /*!<PA[8] pin */
#define SYSCFG_EXTICR3_EXTI8_PB         0x00000001U /*!<PB[8] pin */
#define SYSCFG_EXTICR3_EXTI8_PC         0x00000002U /*!<PC[8] pin */
#define SYSCFG_EXTICR3_EXTI8_PD         0x00000003U /*!<PD[8] pin */
#define SYSCFG_EXTICR3_EXTI8_PE         0x00000004U /*!<PE[8] pin */
#define SYSCFG_EXTICR3_EXTI8_PF         0x00000005U /*!<PF[8] pin */
#define SYSCFG_EXTICR3_EXTI8_PG         0x00000006U /*!<PG[8] pin */
#define SYSCFG_EXTICR3_EXTI8_PH         0x00000007U /*!<PH[8] pin */
#define SYSCFG_EXTICR3_EXTI8_PI         0x00000008U /*!<PI[8] pin */
/** 
  * @brief   EXTI9 configuration  
  */ 
#define SYSCFG_EXTICR3_EXTI9_PA         0x00000000U /*!<PA[9] pin */
#define SYSCFG_EXTICR3_EXTI9_PB         0x00000010U /*!<PB[9] pin */
#define SYSCFG_EXTICR3_EXTI9_PC         0x00000020U /*!<PC[9] pin */
#define SYSCFG_EXTICR3_EXTI9_PD         0x00000030U /*!<PD[9] pin */
#define SYSCFG_EXTICR3_EXTI9_PE         0x00000040U /*!<PE[9] pin */
#define SYSCFG_EXTICR3_EXTI9_PF         0x00000050) /*!<PF[9] pin */
#define SYSCFG_EXTICR3_EXTI9_PG         0x00000060U /*!<PG[9] pin */
#define SYSCFG_EXTICR3_EXTI9_PH         0x00000070U /*!<PH[9] pin */
#define SYSCFG_EXTICR3_EXTI9_PI         0x00000080U /*!<PI[9] pin */
/** 
  * @brief   EXTI10 configuration  
  */ 
#define SYSCFG_EXTICR3_EXTI10_PA        0x00000000U /*!<PA[10] pin */
#define SYSCFG_EXTICR3_EXTI10_PB        0x00000100U /*!<PB[10] pin */
#define SYSCFG_EXTICR3_EXTI10_PC        0x00000200U /*!<PC[10] pin */
#define SYSCFG_EXTICR3_EXTI10_PD        0x00000300U /*!<PD[10] pin */
#define SYSCFG_EXTICR3_EXTI10_PE        0x00000400U /*!<PE[10] pin */
#define SYSCFG_EXTICR3_EXTI10_PF        0x00000500) /*!<PF[10] pin */
#define SYSCFG_EXTICR3_EXTI10_PG        0x00000600) /*!<PG[10] pin */
#define SYSCFG_EXTICR3_EXTI10_PH        0x00000700U /*!<PH[10] pin */
#define SYSCFG_EXTICR3_EXTI10_PI        0x00000800U /*!<PI[10] pin */
/** 
  * @brief   EXTI11 configuration  
  */ 
#define SYSCFG_EXTICR3_EXTI11_PA        0x00000000U /*!<PA[11] pin */
#define SYSCFG_EXTICR3_EXTI11_PB        0x00001000U /*!<PB[11] pin */
#define SYSCFG_EXTICR3_EXTI11_PC        0x00002000U /*!<PC[11] pin */
#define SYSCFG_EXTICR3_EXTI11_PD        0x00003000U /*!<PD[11] pin */
#define SYSCFG_EXTICR3_EXTI11_PE        0x00004000U /*!<PE[11] pin */
#define SYSCFG_EXTICR3_EXTI11_PF        0x00005000) /*!<PF[11] pin */
#define SYSCFG_EXTICR3_EXTI11_PG        0x00006000U /*!<PG[11] pin */
#define SYSCFG_EXTICR3_EXTI11_PH        0x00007000U /*!<PH[11] pin */
#define SYSCFG_EXTICR3_EXTI11_PI        0x00008000U /*!<PI[11] pin */

/*****************  Bit definition for SYSCFG_EXTICR4 register  ***************/
#define SYSCFG_EXTICR4_EXTI12           0x0000000FU /*!<EXTI 12 configuration */
#define SYSCFG_EXTICR4_EXTI13           0x000000F0U /*!<EXTI 13 configuration */
#define SYSCFG_EXTICR4_EXTI14           0x00000F00U /*!<EXTI 14 configuration */
#define SYSCFG_EXTICR4_EXTI15           0x0000F000U /*!<EXTI 15 configuration */
/** 
  * @brief   EXTI12 configuration  
  */ 
#define SYSCFG_EXTICR4_EXTI12_PA        0x00000000U /*!<PA[12] pin */
#define SYSCFG_EXTICR4_EXTI12_PB        0x00000001U /*!<PB[12] pin */
#define SYSCFG_EXTICR4_EXTI12_PC        0x00000002U /*!<PC[12] pin */
#define SYSCFG_EXTICR4_EXTI12_PD        0x00000003U /*!<PD[12] pin */
#define SYSCFG_EXTICR4_EXTI12_PE        0x00000004U /*!<PE[12] pin */
#define SYSCFG_EXTICR4_EXTI12_PF        0x00000005U /*!<PF[12] pin */
#define SYSCFG_EXTICR4_EXTI12_PG        0x00000006U /*!<PG[12] pin */
#define SYSCFG_EXTICR3_EXTI12_PH        0x00000007U /*!<PH[12] pin */
/** 
  * @brief   EXTI13 configuration  
  */ 
#define SYSCFG_EXTICR4_EXTI13_PA        0x00000000U /*!<PA[13] pin */
#define SYSCFG_EXTICR4_EXTI13_PB        0x00000010U /*!<PB[13] pin */
#define SYSCFG_EXTICR4_EXTI13_PC        0x00000020U /*!<PC[13] pin */
#define SYSCFG_EXTICR4_EXTI13_PD        0x00000030U /*!<PD[13] pin */
#define SYSCFG_EXTICR4_EXTI13_PE        0x00000040U /*!<PE[13] pin */
#define SYSCFG_EXTICR4_EXTI13_PF        0x00000050) /*!<PF[13] pin */
#define SYSCFG_EXTICR4_EXTI13_PG        0x00000060U /*!<PG[13] pin */
#define SYSCFG_EXTICR3_EXTI13_PH        0x00000070U /*!<PH[13] pin */
/** 
  * @brief   EXTI14 configuration  
  */ 
#define SYSCFG_EXTICR4_EXTI14_PA        0x00000000U /*!<PA[14] pin */
#define SYSCFG_EXTICR4_EXTI14_PB        0x00000100U /*!<PB[14] pin */
#define SYSCFG_EXTICR4_EXTI14_PC        0x00000200U /*!<PC[14] pin */
#define SYSCFG_EXTICR4_EXTI14_PD        0x00000300U /*!<PD[14] pin */
#define SYSCFG_EXTICR4_EXTI14_PE        0x00000400U /*!<PE[14] pin */
#define SYSCFG_EXTICR4_EXTI14_PF        0x00000500) /*!<PF[14] pin */
#define SYSCFG_EXTICR4_EXTI14_PG        0x00000600) /*!<PG[14] pin */
#define SYSCFG_EXTICR3_EXTI14_PH        0x00000700U /*!<PH[14] pin */
/** 
  * @brief   EXTI15 configuration  
  */ 
#define SYSCFG_EXTICR4_EXTI15_PA        0x00000000U /*!<PA[15] pin */
#define SYSCFG_EXTICR4_EXTI15_PB        0x00001000U /*!<PB[15] pin */
#define SYSCFG_EXTICR4_EXTI15_PC        0x00002000U /*!<PC[15] pin */
#define SYSCFG_EXTICR4_EXTI15_PD        0x00003000U /*!<PD[15] pin */
#define SYSCFG_EXTICR4_EXTI15_PE        0x00004000U /*!<PE[15] pin */
#define SYSCFG_EXTICR4_EXTI15_PF        0x00005000) /*!<PF[15] pin */
#define SYSCFG_EXTICR4_EXTI15_PG        0x00006000U /*!<PG[15] pin */
#define SYSCFG_EXTICR3_EXTI15_PH        0x00007000U /*!<PH[15] pin */

/******************  Bit definition for SYSCFG_CMPCR register  ****************/  
#define SYSCFG_CMPCR_CMP_PD             0x00000001U /*!<Compensation cell ready flag */
#define SYSCFG_CMPCR_READY              0x00000100U /*!<Compensation cell power-down */

/******************************************************************************/
/*                                                                            */
/*                                    TIM                                     */
/*                                                                            */
/******************************************************************************/
/*******************  Bit definition for TIM_CR1 register  ********************/
#define  TIM_CR1_CEN                         0x00000001U            /*!<Counter enable */
#define  TIM_CR1_UDIS                        0x00000002U            /*!<Update disable */
#define  TIM_CR1_URS                         0x00000004U            /*!<Update request source */
#define  TIM_CR1_OPM                         0x00000008U            /*!<One pulse mode */
#define  TIM_CR1_DIR                         0x00000010U            /*!<Direction */

#define  TIM_CR1_CMS                         0x00000060U            /*!<CMS[1:0] bits (Center-aligned mode selection) */
#define  TIM_CR1_CMS_0                       0x00000020U            /*!<Bit 0 */
#define  TIM_CR1_CMS_1                       0x00000040U            /*!<Bit 1 */

#define  TIM_CR1_ARPE                        0x00000080U            /*!<Auto-reload preload enable */

#define  TIM_CR1_CKD                         0x00000300U            /*!<CKD[1:0] bits (clock division) */
#define  TIM_CR1_CKD_0                       0x00000100U            /*!<Bit 0 */
#define  TIM_CR1_CKD_1                       0x00000200U            /*!<Bit 1 */

/*******************  Bit definition for TIM_CR2 register  ********************/
#define  TIM_CR2_CCPC                        0x00000001U            /*!<Capture/Compare Preloaded Control */
#define  TIM_CR2_CCUS                        0x00000004U            /*!<Capture/Compare Control Update Selection */
#define  TIM_CR2_CCDS                        0x00000008U            /*!<Capture/Compare DMA Selection */

#define  TIM_CR2_MMS                         0x00000070U            /*!<MMS[2:0] bits (Master Mode Selection) */
#define  TIM_CR2_MMS_0                       0x00000010U            /*!<Bit 0 */
#define  TIM_CR2_MMS_1                       0x00000020U            /*!<Bit 1 */
#define  TIM_CR2_MMS_2                       0x00000040U            /*!<Bit 2 */

#define  TIM_CR2_TI1S                        0x00000080U            /*!<TI1 Selection */
#define  TIM_CR2_OIS1                        0x00000100U            /*!<Output Idle state 1 (OC1 output) */
#define  TIM_CR2_OIS1N                       0x00000200U            /*!<Output Idle state 1 (OC1N output) */
#define  TIM_CR2_OIS2                        0x00000400U            /*!<Output Idle state 2 (OC2 output) */
#define  TIM_CR2_OIS2N                       0x00000800U            /*!<Output Idle state 2 (OC2N output) */
#define  TIM_CR2_OIS3                        0x00001000U            /*!<Output Idle state 3 (OC3 output) */
#define  TIM_CR2_OIS3N                       0x00002000U            /*!<Output Idle state 3 (OC3N output) */
#define  TIM_CR2_OIS4                        0x00004000U            /*!<Output Idle state 4 (OC4 output) */

/*******************  Bit definition for TIM_SMCR register  *******************/
#define  TIM_SMCR_SMS                        0x00000007U            /*!<SMS[2:0] bits (Slave mode selection) */
#define  TIM_SMCR_SMS_0                      0x00000001U            /*!<Bit 0 */
#define  TIM_SMCR_SMS_1                      0x00000002U            /*!<Bit 1 */
#define  TIM_SMCR_SMS_2                      0x00000004U            /*!<Bit 2 */

#define  TIM_SMCR_TS                         0x00000070U            /*!<TS[2:0] bits (Trigger selection) */
#define  TIM_SMCR_TS_0                       0x00000010U            /*!<Bit 0 */
#define  TIM_SMCR_TS_1                       0x00000020U            /*!<Bit 1 */
#define  TIM_SMCR_TS_2                       0x00000040U            /*!<Bit 2 */

#define  TIM_SMCR_MSM                        0x00000080U            /*!<Master/slave mode */

#define  TIM_SMCR_ETF                        0x00000F00U            /*!<ETF[3:0] bits (External trigger filter) */
#define  TIM_SMCR_ETF_0                      0x00000100U            /*!<Bit 0 */
#define  TIM_SMCR_ETF_1                      0x00000200U            /*!<Bit 1 */
#define  TIM_SMCR_ETF_2                      0x00000400U            /*!<Bit 2 */
#define  TIM_SMCR_ETF_3                      0x00000800U            /*!<Bit 3 */

#define  TIM_SMCR_ETPS                       0x00003000U            /*!<ETPS[1:0] bits (External trigger prescaler) */
#define  TIM_SMCR_ETPS_0                     0x00001000U            /*!<Bit 0 */
#define  TIM_SMCR_ETPS_1                     0x00002000U            /*!<Bit 1 */

#define  TIM_SMCR_ECE                        0x00004000U            /*!<External clock enable */
#define  TIM_SMCR_ETP                        0x00008000U            /*!<External trigger polarity */

/*******************  Bit definition for TIM_DIER register  *******************/
#define  TIM_DIER_UIE                        0x00000001U            /*!<Update interrupt enable */
#define  TIM_DIER_CC1IE                      0x00000002U            /*!<Capture/Compare 1 interrupt enable */
#define  TIM_DIER_CC2IE                      0x00000004U            /*!<Capture/Compare 2 interrupt enable */
#define  TIM_DIER_CC3IE                      0x00000008U            /*!<Capture/Compare 3 interrupt enable */
#define  TIM_DIER_CC4IE                      0x00000010U            /*!<Capture/Compare 4 interrupt enable */
#define  TIM_DIER_COMIE                      0x00000020U            /*!<COM interrupt enable */
#define  TIM_DIER_TIE                        0x00000040U            /*!<Trigger interrupt enable */
#define  TIM_DIER_BIE                        0x00000080U            /*!<Break interrupt enable */
#define  TIM_DIER_UDE                        0x00000100U            /*!<Update DMA request enable */
#define  TIM_DIER_CC1DE                      0x00000200U            /*!<Capture/Compare 1 DMA request enable */
#define  TIM_DIER_CC2DE                      0x00000400U            /*!<Capture/Compare 2 DMA request enable */
#define  TIM_DIER_CC3DE                      0x00000800U            /*!<Capture/Compare 3 DMA request enable */
#define  TIM_DIER_CC4DE                      0x00001000U            /*!<Capture/Compare 4 DMA request enable */
#define  TIM_DIER_COMDE                      0x00002000U            /*!<COM DMA request enable */
#define  TIM_DIER_TDE                        0x00004000U            /*!<Trigger DMA request enable */

/********************  Bit definition for TIM_SR register  ********************/
#define  TIM_SR_UIF                          0x00000001U            /*!<Update interrupt Flag */
#define  TIM_SR_CC1IF                        0x00000002U            /*!<Capture/Compare 1 interrupt Flag */
#define  TIM_SR_CC2IF                        0x00000004U            /*!<Capture/Compare 2 interrupt Flag */
#define  TIM_SR_CC3IF                        0x00000008U            /*!<Capture/Compare 3 interrupt Flag */
#define  TIM_SR_CC4IF                        0x00000010U            /*!<Capture/Compare 4 interrupt Flag */
#define  TIM_SR_COMIF                        0x00000020U            /*!<COM interrupt Flag */
#define  TIM_SR_TIF                          0x00000040U            /*!<Trigger interrupt Flag */
#define  TIM_SR_BIF                          0x00000080U            /*!<Break interrupt Flag */
#define  TIM_SR_CC1OF                        0x00000200U            /*!<Capture/Compare 1 Overcapture Flag */
#define  TIM_SR_CC2OF                        0x00000400U            /*!<Capture/Compare 2 Overcapture Flag */
#define  TIM_SR_CC3OF                        0x00000800U            /*!<Capture/Compare 3 Overcapture Flag */
#define  TIM_SR_CC4OF                        0x00001000U            /*!<Capture/Compare 4 Overcapture Flag */

/*******************  Bit definition for TIM_EGR register  ********************/
#define  TIM_EGR_UG                          0x00000001U               /*!<Update Generation */
#define  TIM_EGR_CC1G                        0x00000002U               /*!<Capture/Compare 1 Generation */
#define  TIM_EGR_CC2G                        0x00000004U               /*!<Capture/Compare 2 Generation */
#define  TIM_EGR_CC3G                        0x00000008U               /*!<Capture/Compare 3 Generation */
#define  TIM_EGR_CC4G                        0x00000010U               /*!<Capture/Compare 4 Generation */
#define  TIM_EGR_COMG                        0x00000020U               /*!<Capture/Compare Control Update Generation */
#define  TIM_EGR_TG                          0x00000040U               /*!<Trigger Generation */
#define  TIM_EGR_BG                          0x00000080U               /*!<Break Generation */

/******************  Bit definition for TIM_CCMR1 register  *******************/
#define  TIM_CCMR1_CC1S                      0x00000003U            /*!<CC1S[1:0] bits (Capture/Compare 1 Selection) */
#define  TIM_CCMR1_CC1S_0                    0x00000001U            /*!<Bit 0 */
#define  TIM_CCMR1_CC1S_1                    0x00000002U            /*!<Bit 1 */

#define  TIM_CCMR1_OC1FE                     0x00000004U            /*!<Output Compare 1 Fast enable */
#define  TIM_CCMR1_OC1PE                     0x00000008U            /*!<Output Compare 1 Preload enable */

#define  TIM_CCMR1_OC1M                      0x00000070U            /*!<OC1M[2:0] bits (Output Compare 1 Mode) */
#define  TIM_CCMR1_OC1M_0                    0x00000010U            /*!<Bit 0 */
#define  TIM_CCMR1_OC1M_1                    0x00000020U            /*!<Bit 1 */
#define  TIM_CCMR1_OC1M_2                    0x00000040U            /*!<Bit 2 */

#define  TIM_CCMR1_OC1CE                     0x00000080U            /*!<Output Compare 1Clear Enable */

#define  TIM_CCMR1_CC2S                      0x00000300U            /*!<CC2S[1:0] bits (Capture/Compare 2 Selection) */
#define  TIM_CCMR1_CC2S_0                    0x00000100U            /*!<Bit 0 */
#define  TIM_CCMR1_CC2S_1                    0x00000200U            /*!<Bit 1 */

#define  TIM_CCMR1_OC2FE                     0x00000400U            /*!<Output Compare 2 Fast enable */
#define  TIM_CCMR1_OC2PE                     0x00000800U            /*!<Output Compare 2 Preload enable */

#define  TIM_CCMR1_OC2M                      0x00007000U            /*!<OC2M[2:0] bits (Output Compare 2 Mode) */
#define  TIM_CCMR1_OC2M_0                    0x00001000U            /*!<Bit 0 */
#define  TIM_CCMR1_OC2M_1                    0x00002000U            /*!<Bit 1 */
#define  TIM_CCMR1_OC2M_2                    0x00004000U            /*!<Bit 2 */

#define  TIM_CCMR1_OC2CE                     0x00008000U            /*!<Output Compare 2 Clear Enable */

/*----------------------------------------------------------------------------*/

#define  TIM_CCMR1_IC1PSC                    0x0000000CU            /*!<IC1PSC[1:0] bits (Input Capture 1 Prescaler) */
#define  TIM_CCMR1_IC1PSC_0                  0x00000004U            /*!<Bit 0 */
#define  TIM_CCMR1_IC1PSC_1                  0x00000008U            /*!<Bit 1 */

#define  TIM_CCMR1_IC1F                      0x000000F0U            /*!<IC1F[3:0] bits (Input Capture 1 Filter) */
#define  TIM_CCMR1_IC1F_0                    0x00000010U            /*!<Bit 0 */
#define  TIM_CCMR1_IC1F_1                    0x00000020U            /*!<Bit 1 */
#define  TIM_CCMR1_IC1F_2                    0x00000040U            /*!<Bit 2 */
#define  TIM_CCMR1_IC1F_3                    0x00000080U            /*!<Bit 3 */

#define  TIM_CCMR1_IC2PSC                    0x00000C00U            /*!<IC2PSC[1:0] bits (Input Capture 2 Prescaler) */
#define  TIM_CCMR1_IC2PSC_0                  0x00000400U            /*!<Bit 0 */
#define  TIM_CCMR1_IC2PSC_1                  0x00000800U            /*!<Bit 1 */

#define  TIM_CCMR1_IC2F                      0x0000F000U            /*!<IC2F[3:0] bits (Input Capture 2 Filter) */
#define  TIM_CCMR1_IC2F_0                    0x00001000U            /*!<Bit 0 */
#define  TIM_CCMR1_IC2F_1                    0x00002000U            /*!<Bit 1 */
#define  TIM_CCMR1_IC2F_2                    0x00004000U            /*!<Bit 2 */
#define  TIM_CCMR1_IC2F_3                    0x00008000U            /*!<Bit 3 */

/******************  Bit definition for TIM_CCMR2 register  *******************/
#define  TIM_CCMR2_CC3S                      0x00000003U            /*!<CC3S[1:0] bits (Capture/Compare 3 Selection) */
#define  TIM_CCMR2_CC3S_0                    0x00000001U            /*!<Bit 0 */
#define  TIM_CCMR2_CC3S_1                    0x00000002U            /*!<Bit 1 */

#define  TIM_CCMR2_OC3FE                     0x00000004U            /*!<Output Compare 3 Fast enable */
#define  TIM_CCMR2_OC3PE                     0x00000008U            /*!<Output Compare 3 Preload enable */

#define  TIM_CCMR2_OC3M                      0x00000070U            /*!<OC3M[2:0] bits (Output Compare 3 Mode) */
#define  TIM_CCMR2_OC3M_0                    0x00000010U            /*!<Bit 0 */
#define  TIM_CCMR2_OC3M_1                    0x00000020U            /*!<Bit 1 */
#define  TIM_CCMR2_OC3M_2                    0x00000040U            /*!<Bit 2 */

#define  TIM_CCMR2_OC3CE                     0x00000080U            /*!<Output Compare 3 Clear Enable */

#define  TIM_CCMR2_CC4S                      0x00000300U            /*!<CC4S[1:0] bits (Capture/Compare 4 Selection) */
#define  TIM_CCMR2_CC4S_0                    0x00000100U            /*!<Bit 0 */
#define  TIM_CCMR2_CC4S_1                    0x00000200U            /*!<Bit 1 */

#define  TIM_CCMR2_OC4FE                     0x00000400U            /*!<Output Compare 4 Fast enable */
#define  TIM_CCMR2_OC4PE                     0x00000800U            /*!<Output Compare 4 Preload enable */

#define  TIM_CCMR2_OC4M                      0x00007000U            /*!<OC4M[2:0] bits (Output Compare 4 Mode) */
#define  TIM_CCMR2_OC4M_0                    0x00001000U            /*!<Bit 0 */
#define  TIM_CCMR2_OC4M_1                    0x00002000U            /*!<Bit 1 */
#define  TIM_CCMR2_OC4M_2                    0x00004000U            /*!<Bit 2 */

#define  TIM_CCMR2_OC4CE                     0x00008000U            /*!<Output Compare 4 Clear Enable */

/*----------------------------------------------------------------------------*/

#define  TIM_CCMR2_IC3PSC                    0x0000000CU            /*!<IC3PSC[1:0] bits (Input Capture 3 Prescaler) */
#define  TIM_CCMR2_IC3PSC_0                  0x00000004U            /*!<Bit 0 */
#define  TIM_CCMR2_IC3PSC_1                  0x00000008U            /*!<Bit 1 */

#define  TIM_CCMR2_IC3F                      0x000000F0U            /*!<IC3F[3:0] bits (Input Capture 3 Filter) */
#define  TIM_CCMR2_IC3F_0                    0x00000010U            /*!<Bit 0 */
#define  TIM_CCMR2_IC3F_1                    0x00000020U            /*!<Bit 1 */
#define  TIM_CCMR2_IC3F_2                    0x00000040U            /*!<Bit 2 */
#define  TIM_CCMR2_IC3F_3                    0x00000080U            /*!<Bit 3 */

#define  TIM_CCMR2_IC4PSC                    0x00000C00U            /*!<IC4PSC[1:0] bits (Input Capture 4 Prescaler) */
#define  TIM_CCMR2_IC4PSC_0                  0x00000400U            /*!<Bit 0 */
#define  TIM_CCMR2_IC4PSC_1                  0x00000800U            /*!<Bit 1 */

#define  TIM_CCMR2_IC4F                      0x0000F000U            /*!<IC4F[3:0] bits (Input Capture 4 Filter) */
#define  TIM_CCMR2_IC4F_0                    0x00001000U            /*!<Bit 0 */
#define  TIM_CCMR2_IC4F_1                    0x00002000U            /*!<Bit 1 */
#define  TIM_CCMR2_IC4F_2                    0x00004000U            /*!<Bit 2 */
#define  TIM_CCMR2_IC4F_3                    0x00008000U            /*!<Bit 3 */

/*******************  Bit definition for TIM_CCER register  *******************/
#define  TIM_CCER_CC1E                       0x00000001U            /*!<Capture/Compare 1 output enable */
#define  TIM_CCER_CC1P                       0x00000002U            /*!<Capture/Compare 1 output Polarity */
#define  TIM_CCER_CC1NE                      0x00000004U            /*!<Capture/Compare 1 Complementary output enable */
#define  TIM_CCER_CC1NP                      0x00000008U            /*!<Capture/Compare 1 Complementary output Polarity */
#define  TIM_CCER_CC2E                       0x00000010U            /*!<Capture/Compare 2 output enable */
#define  TIM_CCER_CC2P                       0x00000020U            /*!<Capture/Compare 2 output Polarity */
#define  TIM_CCER_CC2NE                      0x00000040U            /*!<Capture/Compare 2 Complementary output enable */
#define  TIM_CCER_CC2NP                      0x00000080U            /*!<Capture/Compare 2 Complementary output Polarity */
#define  TIM_CCER_CC3E                       0x00000100U            /*!<Capture/Compare 3 output enable */
#define  TIM_CCER_CC3P                       0x00000200U            /*!<Capture/Compare 3 output Polarity */
#define  TIM_CCER_CC3NE                      0x00000400U            /*!<Capture/Compare 3 Complementary output enable */
#define  TIM_CCER_CC3NP                      0x00000800U            /*!<Capture/Compare 3 Complementary output Polarity */
#define  TIM_CCER_CC4E                       0x00001000U            /*!<Capture/Compare 4 output enable */
#define  TIM_CCER_CC4P                       0x00002000U            /*!<Capture/Compare 4 output Polarity */
#define  TIM_CCER_CC4NP                      0x00008000U            /*!<Capture/Compare 4 Complementary output Polarity */

/*******************  Bit definition for TIM_CNT register  ********************/
#define  TIM_CNT_CNT                         0x0000FFFFU            /*!<Counter Value */

/*******************  Bit definition for TIM_PSC register  ********************/
#define  TIM_PSC_PSC                         0x0000FFFFU            /*!<Prescaler Value */

/*******************  Bit definition for TIM_ARR register  ********************/
#define  TIM_ARR_ARR                         0x0000FFFFU            /*!<actual auto-reload Value */

/*******************  Bit definition for TIM_RCR register  ********************/
#define  TIM_RCR_REP                         0x000000FF               /*!<Repetition Counter Value */

/*******************  Bit definition for TIM_CCR1 register  *******************/
#define  TIM_CCR1_CCR1                       0x0000FFFFU            /*!<Capture/Compare 1 Value */

/*******************  Bit definition for TIM_CCR2 register  *******************/
#define  TIM_CCR2_CCR2                       0x0000FFFFU            /*!<Capture/Compare 2 Value */

/*******************  Bit definition for TIM_CCR3 register  *******************/
#define  TIM_CCR3_CCR3                       0x0000FFFFU            /*!<Capture/Compare 3 Value */

/*******************  Bit definition for TIM_CCR4 register  *******************/
#define  TIM_CCR4_CCR4                       0x0000FFFFU            /*!<Capture/Compare 4 Value */

/*******************  Bit definition for TIM_BDTR register  *******************/
#define  TIM_BDTR_DTG                        0x000000FFU            /*!<DTG[0:7] bits (Dead-Time Generator set-up) */
#define  TIM_BDTR_DTG_0                      0x00000001U            /*!<Bit 0 */
#define  TIM_BDTR_DTG_1                      0x00000002U            /*!<Bit 1 */
#define  TIM_BDTR_DTG_2                      0x00000004U            /*!<Bit 2 */
#define  TIM_BDTR_DTG_3                      0x00000008U            /*!<Bit 3 */
#define  TIM_BDTR_DTG_4                      0x00000010U            /*!<Bit 4 */
#define  TIM_BDTR_DTG_5                      0x00000020U            /*!<Bit 5 */
#define  TIM_BDTR_DTG_6                      0x00000040U            /*!<Bit 6 */
#define  TIM_BDTR_DTG_7                      0x00000080U            /*!<Bit 7 */

#define  TIM_BDTR_LOCK                       0x00000300U            /*!<LOCK[1:0] bits (Lock Configuration) */
#define  TIM_BDTR_LOCK_0                     0x00000100U            /*!<Bit 0 */
#define  TIM_BDTR_LOCK_1                     0x00000200U            /*!<Bit 1 */

#define  TIM_BDTR_OSSI                       0x00000400U            /*!<Off-State Selection for Idle mode */
#define  TIM_BDTR_OSSR                       0x00000800U            /*!<Off-State Selection for Run mode */
#define  TIM_BDTR_BKE                        0x00001000U            /*!<Break enable */
#define  TIM_BDTR_BKP                        0x00002000U            /*!<Break Polarity */
#define  TIM_BDTR_AOE                        0x00004000U            /*!<Automatic Output enable */
#define  TIM_BDTR_MOE                        0x00008000U            /*!<Main Output enable */

/*******************  Bit definition for TIM_DCR register  ********************/
#define  TIM_DCR_DBA                         0x0000001FU            /*!<DBA[4:0] bits (DMA Base Address) */
#define  TIM_DCR_DBA_0                       0x00000001U            /*!<Bit 0 */
#define  TIM_DCR_DBA_1                       0x00000002U            /*!<Bit 1 */
#define  TIM_DCR_DBA_2                       0x00000004U            /*!<Bit 2 */
#define  TIM_DCR_DBA_3                       0x00000008U            /*!<Bit 3 */
#define  TIM_DCR_DBA_4                       0x00000010U            /*!<Bit 4 */

#define  TIM_DCR_DBL                         0x00001F00U            /*!<DBL[4:0] bits (DMA Burst Length) */
#define  TIM_DCR_DBL_0                       0x00000100U            /*!<Bit 0 */
#define  TIM_DCR_DBL_1                       0x00000200U            /*!<Bit 1 */
#define  TIM_DCR_DBL_2                       0x00000400U            /*!<Bit 2 */
#define  TIM_DCR_DBL_3                       0x00000800U            /*!<Bit 3 */
#define  TIM_DCR_DBL_4                       0x00001000U            /*!<Bit 4 */

/*******************  Bit definition for TIM_DMAR register  *******************/
#define  TIM_DMAR_DMAB                       0x0000FFFFU            /*!<DMA register for burst accesses */

/*******************  Bit definition for TIM_OR register  *********************/
#define TIM_OR_TI4_RMP                       0x000000C0U            /*!<TI4_RMP[1:0] bits (TIM5 Input 4 remap) */
#define TIM_OR_TI4_RMP_0                     0x00000040U            /*!<Bit 0 */
#define TIM_OR_TI4_RMP_1                     0x00000080U            /*!<Bit 1 */
#define TIM_OR_ITR1_RMP                      0x00000C00U            /*!<ITR1_RMP[1:0] bits (TIM2 Internal trigger 1 remap) */
#define TIM_OR_ITR1_RMP_0                    0x00000400U            /*!<Bit 0 */
#define TIM_OR_ITR1_RMP_1                    0x00000800U            /*!<Bit 1 */


/******************************************************************************/
/*                                                                            */
/*         Universal Synchronous Asynchronous Receiver Transmitter            */
/*                                                                            */
/******************************************************************************/
/*******************  Bit definition for USART_SR register  *******************/
#define  USART_SR_PE                         0x00000001U            /*!<Parity Error */
#define  USART_SR_FE                         0x00000002U            /*!<Framing Error */
#define  USART_SR_NE                         0x00000004U            /*!<Noise Error Flag */
#define  USART_SR_ORE                        0x00000008U            /*!<OverRun Error */
#define  USART_SR_IDLE                       0x00000010U            /*!<IDLE line detected */
#define  USART_SR_RXNE                       0x00000020U            /*!<Read Data Register Not Empty */
#define  USART_SR_TC                         0x00000040U            /*!<Transmission Complete */
#define  USART_SR_TXE                        0x00000080U            /*!<Transmit Data Register Empty */
#define  USART_SR_LBD                        0x00000100U            /*!<LIN Break Detection Flag */
#define  USART_SR_CTS                        0x00000200U            /*!<CTS Flag */

/*******************  Bit definition for USART_DR register  *******************/
#define  USART_DR_DR                         0x000001FFU            /*!<Data value */

/******************  Bit definition for USART_BRR register  *******************/
#define  USART_BRR_DIV_Fraction              0x0000000FU            /*!<Fraction of USARTDIV */
#define  USART_BRR_DIV_Mantissa              0x0000FFF0U            /*!<Mantissa of USARTDIV */

/******************  Bit definition for USART_CR1 register  *******************/
#define  USART_CR1_SBK                       0x00000001U            /*!<Send Break */
#define  USART_CR1_RWU                       0x00000002U            /*!<Receiver wakeup */
#define  USART_CR1_RE                        0x00000004U            /*!<Receiver Enable */
#define  USART_CR1_TE                        0x00000008U            /*!<Transmitter Enable */
#define  USART_CR1_IDLEIE                    0x00000010U            /*!<IDLE Interrupt Enable */
#define  USART_CR1_RXNEIE                    0x00000020U            /*!<RXNE Interrupt Enable */
#define  USART_CR1_TCIE                      0x00000040U            /*!<Transmission Complete Interrupt Enable */
#define  USART_CR1_TXEIE                     0x00000080U            /*!<PE Interrupt Enable */
#define  USART_CR1_PEIE                      0x00000100U            /*!<PE Interrupt Enable */
#define  USART_CR1_PS                        0x00000200U            /*!<Parity Selection */
#define  USART_CR1_PCE                       0x00000400U            /*!<Parity Control Enable */
#define  USART_CR1_WAKE                      0x00000800U            /*!<Wakeup method */
#define  USART_CR1_M                         0x00001000U            /*!<Word length */
#define  USART_CR1_UE                        0x00002000U            /*!<USART Enable */
#define  USART_CR1_OVER8                     0x00008000U            /*!<USART Oversampling by 8 enable */

/******************  Bit definition for USART_CR2 register  *******************/
#define  USART_CR2_ADD                       0x0000000FU            /*!<Address of the USART node */
#define  USART_CR2_LBDL                      0x00000020U            /*!<LIN Break Detection Length */
#define  USART_CR2_LBDIE                     0x00000040U            /*!<LIN Break Detection Interrupt Enable */
#define  USART_CR2_LBCL                      0x00000100U            /*!<Last Bit Clock pulse */
#define  USART_CR2_CPHA                      0x00000200U            /*!<Clock Phase */
#define  USART_CR2_CPOL                      0x00000400U            /*!<Clock Polarity */
#define  USART_CR2_CLKEN                     0x00000800U            /*!<Clock Enable */

#define  USART_CR2_STOP                      0x00003000U            /*!<STOP[1:0] bits (STOP bits) */
#define  USART_CR2_STOP_0                    0x00001000U            /*!<Bit 0 */
#define  USART_CR2_STOP_1                    0x00002000U            /*!<Bit 1 */

#define  USART_CR2_LINEN                     0x00004000U            /*!<LIN mode enable */

/******************  Bit definition for USART_CR3 register  *******************/
#define  USART_CR3_EIE                       0x00000001U            /*!<Error Interrupt Enable */
#define  USART_CR3_IREN                      0x00000002U            /*!<IrDA mode Enable */
#define  USART_CR3_IRLP                      0x00000004U            /*!<IrDA Low-Power */
#define  USART_CR3_HDSEL                     0x00000008U            /*!<Half-Duplex Selection */
#define  USART_CR3_NACK                      0x00000010U            /*!<Smartcard NACK enable */
#define  USART_CR3_SCEN                      0x00000020U            /*!<Smartcard mode enable */
#define  USART_CR3_DMAR                      0x00000040U            /*!<DMA Enable Receiver */
#define  USART_CR3_DMAT                      0x00000080U            /*!<DMA Enable Transmitter */
#define  USART_CR3_RTSE                      0x00000100U            /*!<RTS Enable */
#define  USART_CR3_CTSE                      0x00000200U            /*!<CTS Enable */
#define  USART_CR3_CTSIE                     0x00000400U            /*!<CTS Interrupt Enable */
#define  USART_CR3_ONEBIT                    0x00000800U            /*!<USART One bit method enable */

/******************  Bit definition for USART_GTPR register  ******************/
#define  USART_GTPR_PSC                      0x000000FFU            /*!<PSC[7:0] bits (Prescaler value) */
#define  USART_GTPR_PSC_0                    0x00000001U            /*!<Bit 0 */
#define  USART_GTPR_PSC_1                    0x00000002U            /*!<Bit 1 */
#define  USART_GTPR_PSC_2                    0x00000004U            /*!<Bit 2 */
#define  USART_GTPR_PSC_3                    0x00000008U            /*!<Bit 3 */
#define  USART_GTPR_PSC_4                    0x00000010U            /*!<Bit 4 */
#define  USART_GTPR_PSC_5                    0x00000020U            /*!<Bit 5 */
#define  USART_GTPR_PSC_6                    0x00000040U            /*!<Bit 6 */
#define  USART_GTPR_PSC_7                    0x00000080U            /*!<Bit 7 */

#define  USART_GTPR_GT                       0x0000FF00U            /*!<Guard time value */

/******************************************************************************/
/*                                                                            */
/*                            Window WATCHDOG                                 */
/*                                                                            */
/******************************************************************************/
/*******************  Bit definition for WWDG_CR register  ********************/
#define  WWDG_CR_T                           0x0000007FU            /*!<T[6:0] bits (7-Bit counter (MSB to LSB)) */
#define  WWDG_CR_T_0                         0x00000001U            /*!<Bit 0 */
#define  WWDG_CR_T_1                         0x00000002U            /*!<Bit 1 */
#define  WWDG_CR_T_2                         0x00000004U            /*!<Bit 2 */
#define  WWDG_CR_T_3                         0x00000008U            /*!<Bit 3 */
#define  WWDG_CR_T_4                         0x00000010U            /*!<Bit 4 */
#define  WWDG_CR_T_5                         0x00000020U            /*!<Bit 5 */
#define  WWDG_CR_T_6                         0x00000040U            /*!<Bit 6 */

/* Legacy defines */
#define  WWDG_CR_T0                          WWDG_CR_T_0
#define  WWDG_CR_T1                          WWDG_CR_T_1
#define  WWDG_CR_T2                          WWDG_CR_T_2
#define  WWDG_CR_T3                          WWDG_CR_T_3
#define  WWDG_CR_T4                          WWDG_CR_T_4
#define  WWDG_CR_T5                          WWDG_CR_T_5
#define  WWDG_CR_T6                          WWDG_CR_T_6
#define  WWDG_CR_WDGA                        0x00000080U            /*!<Activation bit */

/*******************  Bit definition for WWDG_CFR register  *******************/
#define  WWDG_CFR_W                          0x0000007FU            /*!<W[6:0] bits (7-bit window value) */
#define  WWDG_CFR_W_0                        0x00000001U            /*!<Bit 0 */
#define  WWDG_CFR_W_1                        0x00000002U            /*!<Bit 1 */
#define  WWDG_CFR_W_2                        0x00000004U            /*!<Bit 2 */
#define  WWDG_CFR_W_3                        0x00000008U            /*!<Bit 3 */
#define  WWDG_CFR_W_4                        0x00000010U            /*!<Bit 4 */
#define  WWDG_CFR_W_5                        0x00000020U            /*!<Bit 5 */
#define  WWDG_CFR_W_6                        0x00000040U            /*!<Bit 6 */

/* Legacy defines */
#define  WWDG_CFR_W0                         WWDG_CFR_W_0
#define  WWDG_CFR_W1                         WWDG_CFR_W_1
#define  WWDG_CFR_W2                         WWDG_CFR_W_2
#define  WWDG_CFR_W3                         WWDG_CFR_W_3
#define  WWDG_CFR_W4                         WWDG_CFR_W_4
#define  WWDG_CFR_W5                         WWDG_CFR_W_5
#define  WWDG_CFR_W6                         WWDG_CFR_W_6

#define  WWDG_CFR_WDGTB                      0x00000180U            /*!<WDGTB[1:0] bits (Timer Base) */
#define  WWDG_CFR_WDGTB_0                    0x00000080U            /*!<Bit 0 */
#define  WWDG_CFR_WDGTB_1                    0x00000100U            /*!<Bit 1 */

/* Legacy defines */
#define  WWDG_CFR_WDGTB0                     WWDG_CFR_WDGTB_0
#define  WWDG_CFR_WDGTB1                     WWDG_CFR_WDGTB_1

#define  WWDG_CFR_EWI                        0x00000200U            /*!<Early Wakeup Interrupt */

/*******************  Bit definition for WWDG_SR register  ********************/
#define  WWDG_SR_EWIF                        0x00000001U            /*!<Early Wakeup Interrupt Flag */

/******************************************************************************/
/*                                                                            */
/*                                DBG                                         */
/*                                                                            */
/******************************************************************************/
/********************  Bit definition for DBGMCU_IDCODE register  *************/
#define  DBGMCU_IDCODE_DEV_ID                0x00000FFFU
#define  DBGMCU_IDCODE_REV_ID                0xFFFF0000U

/********************  Bit definition for DBGMCU_CR register  *****************/
#define  DBGMCU_CR_DBG_SLEEP                 0x00000001U
#define  DBGMCU_CR_DBG_STOP                  0x00000002U
#define  DBGMCU_CR_DBG_STANDBY               0x00000004U
#define  DBGMCU_CR_TRACE_IOEN                0x00000020U

#define  DBGMCU_CR_TRACE_MODE                0x000000C0U
#define  DBGMCU_CR_TRACE_MODE_0              0x00000040U/*!<Bit 0 */
#define  DBGMCU_CR_TRACE_MODE_1              0x00000080U/*!<Bit 1 */

/********************  Bit definition for DBGMCU_APB1_FZ register  ************/
#define  DBGMCU_APB1_FZ_DBG_TIM2_STOP            0x00000001U
#define  DBGMCU_APB1_FZ_DBG_TIM3_STOP            0x00000002U
#define  DBGMCU_APB1_FZ_DBG_TIM4_STOP            0x00000004U
#define  DBGMCU_APB1_FZ_DBG_TIM5_STOP            0x00000008U
#define  DBGMCU_APB1_FZ_DBG_TIM6_STOP            0x00000010U
#define  DBGMCU_APB1_FZ_DBG_TIM7_STOP            0x00000020U
#define  DBGMCU_APB1_FZ_DBG_TIM12_STOP           0x00000040U
#define  DBGMCU_APB1_FZ_DBG_TIM13_STOP           0x00000080U
#define  DBGMCU_APB1_FZ_DBG_TIM14_STOP           0x00000100U
#define  DBGMCU_APB1_FZ_DBG_RTC_STOP             0x00000400U
#define  DBGMCU_APB1_FZ_DBG_WWDG_STOP            0x00000800U
#define  DBGMCU_APB1_FZ_DBG_IWDG_STOP            0x00001000U
#define  DBGMCU_APB1_FZ_DBG_I2C1_SMBUS_TIMEOUT   0x00200000U
#define  DBGMCU_APB1_FZ_DBG_I2C2_SMBUS_TIMEOUT   0x00400000U
#define  DBGMCU_APB1_FZ_DBG_I2C3_SMBUS_TIMEOUT   0x00800000U
#define  DBGMCU_APB1_FZ_DBG_CAN1_STOP            0x02000000U
#define  DBGMCU_APB1_FZ_DBG_CAN2_STOP            0x04000000U
/* Old IWDGSTOP bit definition, maintained for legacy purpose */
#define  DBGMCU_APB1_FZ_DBG_IWDEG_STOP           DBGMCU_APB1_FZ_DBG_IWDG_STOP

/********************  Bit definition for DBGMCU_APB2_FZ register  ************/
#define  DBGMCU_APB2_FZ_DBG_TIM1_STOP        0x00000001U
#define  DBGMCU_APB2_FZ_DBG_TIM8_STOP        0x00000002U
#define  DBGMCU_APB2_FZ_DBG_TIM9_STOP        0x00010000U
#define  DBGMCU_APB2_FZ_DBG_TIM10_STOP       0x00020000U
#define  DBGMCU_APB2_FZ_DBG_TIM11_STOP       0x00040000U

/******************************************************************************/
/*                                                                            */
/*                                       USB_OTG			                        */
/*                                                                            */
/******************************************************************************/
/********************  Bit definition forUSB_OTG_GOTGCTL register  ********************/
#define USB_OTG_GOTGCTL_SRQSCS                  0x00000001U            /*!< Session request success */
#define USB_OTG_GOTGCTL_SRQ                     0x00000002U            /*!< Session request */
#define USB_OTG_GOTGCTL_HNGSCS                  0x00000100U            /*!< Host negotiation success */
#define USB_OTG_GOTGCTL_HNPRQ                   0x00000200U            /*!< HNP request */
#define USB_OTG_GOTGCTL_HSHNPEN                 0x00000400U            /*!< Host set HNP enable */
#define USB_OTG_GOTGCTL_DHNPEN                  0x00000800U            /*!< Device HNP enabled */
#define USB_OTG_GOTGCTL_CIDSTS                  0x00010000U            /*!< Connector ID status */
#define USB_OTG_GOTGCTL_DBCT                    0x00020000U            /*!< Long/short debounce time */
#define USB_OTG_GOTGCTL_ASVLD                   0x00040000U            /*!< A-session valid */
#define USB_OTG_GOTGCTL_BSVLD                   0x00080000U            /*!< B-session valid */

/********************  Bit definition forUSB_OTG_HCFG register  ********************/

#define USB_OTG_HCFG_FSLSPCS                 0x00000003U            /*!< FS/LS PHY clock select */
#define USB_OTG_HCFG_FSLSPCS_0               0x00000001U            /*!<Bit 0 */
#define USB_OTG_HCFG_FSLSPCS_1               0x00000002U            /*!<Bit 1 */
#define USB_OTG_HCFG_FSLSS                   0x00000004U            /*!< FS- and LS-only support */

/********************  Bit definition forUSB_OTG_DCFG register  ********************/

#define USB_OTG_DCFG_DSPD                    0x00000003U            /*!< Device speed */
#define USB_OTG_DCFG_DSPD_0                  0x00000001U            /*!<Bit 0 */
#define USB_OTG_DCFG_DSPD_1                  0x00000002U            /*!<Bit 1 */
#define USB_OTG_DCFG_NZLSOHSK                0x00000004U            /*!< Nonzero-length status OUT handshake */

#define USB_OTG_DCFG_DAD                     0x000007F0U            /*!< Device address */
#define USB_OTG_DCFG_DAD_0                   0x00000010U            /*!<Bit 0 */
#define USB_OTG_DCFG_DAD_1                   0x00000020U            /*!<Bit 1 */
#define USB_OTG_DCFG_DAD_2                   0x00000040U            /*!<Bit 2 */
#define USB_OTG_DCFG_DAD_3                   0x00000080U            /*!<Bit 3 */
#define USB_OTG_DCFG_DAD_4                   0x00000100U            /*!<Bit 4 */
#define USB_OTG_DCFG_DAD_5                   0x00000200U            /*!<Bit 5 */
#define USB_OTG_DCFG_DAD_6                   0x00000400U            /*!<Bit 6 */

#define USB_OTG_DCFG_PFIVL                   0x00001800U            /*!< Periodic (micro)frame interval */
#define USB_OTG_DCFG_PFIVL_0                 0x00000800U            /*!<Bit 0 */
#define USB_OTG_DCFG_PFIVL_1                 0x00001000U            /*!<Bit 1 */

#define USB_OTG_DCFG_PERSCHIVL               0x03000000U            /*!< Periodic scheduling interval */
#define USB_OTG_DCFG_PERSCHIVL_0             0x01000000U            /*!<Bit 0 */
#define USB_OTG_DCFG_PERSCHIVL_1             0x02000000U            /*!<Bit 1 */

/********************  Bit definition forUSB_OTG_PCGCR register  ********************/
#define USB_OTG_PCGCR_STPPCLK                 0x00000001U            /*!< Stop PHY clock */
#define USB_OTG_PCGCR_GATEHCLK                0x00000002U            /*!< Gate HCLK */
#define USB_OTG_PCGCR_PHYSUSP                 0x00000010U            /*!< PHY suspended */

/********************  Bit definition forUSB_OTG_GOTGINT register  ********************/
#define USB_OTG_GOTGINT_SEDET                   0x00000004U            /*!< Session end detected */
#define USB_OTG_GOTGINT_SRSSCHG                 0x00000100U            /*!< Session request success status change */
#define USB_OTG_GOTGINT_HNSSCHG                 0x00000200U            /*!< Host negotiation success status change */
#define USB_OTG_GOTGINT_HNGDET                  0x00020000U            /*!< Host negotiation detected */
#define USB_OTG_GOTGINT_ADTOCHG                 0x00040000U            /*!< A-device timeout change */
#define USB_OTG_GOTGINT_DBCDNE                  0x00080000U            /*!< Debounce done */

/********************  Bit definition forUSB_OTG_DCTL register  ********************/
#define USB_OTG_DCTL_RWUSIG                  0x00000001U            /*!< Remote wakeup signaling */
#define USB_OTG_DCTL_SDIS                    0x00000002U            /*!< Soft disconnect */
#define USB_OTG_DCTL_GINSTS                  0x00000004U            /*!< Global IN NAK status */
#define USB_OTG_DCTL_GONSTS                  0x00000008U            /*!< Global OUT NAK status */

#define USB_OTG_DCTL_TCTL                    0x00000070U            /*!< Test control */
#define USB_OTG_DCTL_TCTL_0                  0x00000010U            /*!<Bit 0 */
#define USB_OTG_DCTL_TCTL_1                  0x00000020U            /*!<Bit 1 */
#define USB_OTG_DCTL_TCTL_2                  0x00000040U            /*!<Bit 2 */
#define USB_OTG_DCTL_SGINAK                  0x00000080U            /*!< Set global IN NAK */
#define USB_OTG_DCTL_CGINAK                  0x00000100U            /*!< Clear global IN NAK */
#define USB_OTG_DCTL_SGONAK                  0x00000200U            /*!< Set global OUT NAK */
#define USB_OTG_DCTL_CGONAK                  0x00000400U            /*!< Clear global OUT NAK */
#define USB_OTG_DCTL_POPRGDNE                0x00000800U            /*!< Power-on programming done */

/********************  Bit definition forUSB_OTG_HFIR register  ********************/
#define USB_OTG_HFIR_FRIVL                   0x0000FFFFU            /*!< Frame interval */

/********************  Bit definition forUSB_OTG_HFNUM register  ********************/
#define USB_OTG_HFNUM_FRNUM                   0x0000FFFFU            /*!< Frame number */
#define USB_OTG_HFNUM_FTREM                   0xFFFF0000U            /*!< Frame time remaining */

/********************  Bit definition forUSB_OTG_DSTS register  ********************/
#define USB_OTG_DSTS_SUSPSTS                 0x00000001U            /*!< Suspend status */

#define USB_OTG_DSTS_ENUMSPD                 0x00000006U            /*!< Enumerated speed */
#define USB_OTG_DSTS_ENUMSPD_0               0x00000002U            /*!<Bit 0 */
#define USB_OTG_DSTS_ENUMSPD_1               0x00000004U            /*!<Bit 1 */
#define USB_OTG_DSTS_EERR                    0x00000008U            /*!< Erratic error */
#define USB_OTG_DSTS_FNSOF                   0x003FFF00U            /*!< Frame number of the received SOF */

/********************  Bit definition forUSB_OTG_GAHBCFG register  ********************/
#define USB_OTG_GAHBCFG_GINT                    0x00000001U            /*!< Global interrupt mask */

#define USB_OTG_GAHBCFG_HBSTLEN                 0x0000001EU            /*!< Burst length/type */
#define USB_OTG_GAHBCFG_HBSTLEN_0               0x00000002U            /*!<Bit 0 */
#define USB_OTG_GAHBCFG_HBSTLEN_1               0x00000004U            /*!<Bit 1 */
#define USB_OTG_GAHBCFG_HBSTLEN_2               0x00000008U            /*!<Bit 2 */
#define USB_OTG_GAHBCFG_HBSTLEN_3               0x00000010U            /*!<Bit 3 */
#define USB_OTG_GAHBCFG_DMAEN                   0x00000020U            /*!< DMA enable */
#define USB_OTG_GAHBCFG_TXFELVL                 0x00000080U            /*!< TxFIFO empty level */
#define USB_OTG_GAHBCFG_PTXFELVL                0x00000100U            /*!< Periodic TxFIFO empty level */

/********************  Bit definition forUSB_OTG_GUSBCFG register  ********************/

#define USB_OTG_GUSBCFG_TOCAL                   0x00000007U            /*!< FS timeout calibration */
#define USB_OTG_GUSBCFG_TOCAL_0                 0x00000001U            /*!<Bit 0 */
#define USB_OTG_GUSBCFG_TOCAL_1                 0x00000002U            /*!<Bit 1 */
#define USB_OTG_GUSBCFG_TOCAL_2                 0x00000004U            /*!<Bit 2 */
#define USB_OTG_GUSBCFG_PHYSEL                  0x00000040U            /*!< USB 2.0 high-speed ULPI PHY or USB 1.1 full-speed serial transceiver select */
#define USB_OTG_GUSBCFG_SRPCAP                  0x00000100U            /*!< SRP-capable */
#define USB_OTG_GUSBCFG_HNPCAP                  0x00000200U            /*!< HNP-capable */

#define USB_OTG_GUSBCFG_TRDT                    0x00003C00U            /*!< USB turnaround time */
#define USB_OTG_GUSBCFG_TRDT_0                  0x00000400U            /*!<Bit 0 */
#define USB_OTG_GUSBCFG_TRDT_1                  0x00000800U            /*!<Bit 1 */
#define USB_OTG_GUSBCFG_TRDT_2                  0x00001000U            /*!<Bit 2 */
#define USB_OTG_GUSBCFG_TRDT_3                  0x00002000U            /*!<Bit 3 */
#define USB_OTG_GUSBCFG_PHYLPCS                 0x00008000U            /*!< PHY Low-power clock select */
#define USB_OTG_GUSBCFG_ULPIFSLS                0x00020000U            /*!< ULPI FS/LS select */
#define USB_OTG_GUSBCFG_ULPIAR                  0x00040000U            /*!< ULPI Auto-resume */
#define USB_OTG_GUSBCFG_ULPICSM                 0x00080000U            /*!< ULPI Clock SuspendM */
#define USB_OTG_GUSBCFG_ULPIEVBUSD              0x00100000U            /*!< ULPI External VBUS Drive */
#define USB_OTG_GUSBCFG_ULPIEVBUSI              0x00200000U            /*!< ULPI external VBUS indicator */
#define USB_OTG_GUSBCFG_TSDPS                   0x00400000U            /*!< TermSel DLine pulsing selection */
#define USB_OTG_GUSBCFG_PCCI                    0x00800000U            /*!< Indicator complement */
#define USB_OTG_GUSBCFG_PTCI                    0x01000000U            /*!< Indicator pass through */
#define USB_OTG_GUSBCFG_ULPIIPD                 0x02000000U            /*!< ULPI interface protect disable */
#define USB_OTG_GUSBCFG_FHMOD                   0x20000000U            /*!< Forced host mode */
#define USB_OTG_GUSBCFG_FDMOD                   0x40000000U            /*!< Forced peripheral mode */
#define USB_OTG_GUSBCFG_CTXPKT                  0x80000000U            /*!< Corrupt Tx packet */

/********************  Bit definition forUSB_OTG_GRSTCTL register  ********************/
#define USB_OTG_GRSTCTL_CSRST                   0x00000001U            /*!< Core soft reset */
#define USB_OTG_GRSTCTL_HSRST                   0x00000002U            /*!< HCLK soft reset */
#define USB_OTG_GRSTCTL_FCRST                   0x00000004U            /*!< Host frame counter reset */
#define USB_OTG_GRSTCTL_RXFFLSH                 0x00000010U            /*!< RxFIFO flush */
#define USB_OTG_GRSTCTL_TXFFLSH                 0x00000020U            /*!< TxFIFO flush */

#define USB_OTG_GRSTCTL_TXFNUM                  0x000007C0U            /*!< TxFIFO number */
#define USB_OTG_GRSTCTL_TXFNUM_0                0x00000040U            /*!<Bit 0 */
#define USB_OTG_GRSTCTL_TXFNUM_1                0x00000080U            /*!<Bit 1 */
#define USB_OTG_GRSTCTL_TXFNUM_2                0x00000100U            /*!<Bit 2 */
#define USB_OTG_GRSTCTL_TXFNUM_3                0x00000200U            /*!<Bit 3 */
#define USB_OTG_GRSTCTL_TXFNUM_4                0x00000400U            /*!<Bit 4 */
#define USB_OTG_GRSTCTL_DMAREQ                  0x40000000U            /*!< DMA request signal */
#define USB_OTG_GRSTCTL_AHBIDL                  0x80000000U            /*!< AHB master idle */

/********************  Bit definition forUSB_OTG_DIEPMSK register  ********************/
#define USB_OTG_DIEPMSK_XFRCM                   0x00000001U            /*!< Transfer completed interrupt mask */
#define USB_OTG_DIEPMSK_EPDM                    0x00000002U            /*!< Endpoint disabled interrupt mask */
#define USB_OTG_DIEPMSK_TOM                     0x00000008U            /*!< Timeout condition mask (nonisochronous endpoints) */
#define USB_OTG_DIEPMSK_ITTXFEMSK               0x00000010U            /*!< IN token received when TxFIFO empty mask */
#define USB_OTG_DIEPMSK_INEPNMM                 0x00000020U            /*!< IN token received with EP mismatch mask */
#define USB_OTG_DIEPMSK_INEPNEM                 0x00000040U            /*!< IN endpoint NAK effective mask */
#define USB_OTG_DIEPMSK_TXFURM                  0x00000100U            /*!< FIFO underrun mask */
#define USB_OTG_DIEPMSK_BIM                     0x00000200U            /*!< BNA interrupt mask */

/********************  Bit definition forUSB_OTG_HPTXSTS register  ********************/
#define USB_OTG_HPTXSTS_PTXFSAVL                0x0000FFFFU            /*!< Periodic transmit data FIFO space available */

#define USB_OTG_HPTXSTS_PTXQSAV                 0x00FF0000U            /*!< Periodic transmit request queue space available */
#define USB_OTG_HPTXSTS_PTXQSAV_0               0x00010000U            /*!<Bit 0 */
#define USB_OTG_HPTXSTS_PTXQSAV_1               0x00020000U            /*!<Bit 1 */
#define USB_OTG_HPTXSTS_PTXQSAV_2               0x00040000U            /*!<Bit 2 */
#define USB_OTG_HPTXSTS_PTXQSAV_3               0x00080000U            /*!<Bit 3 */
#define USB_OTG_HPTXSTS_PTXQSAV_4               0x00100000U            /*!<Bit 4 */
#define USB_OTG_HPTXSTS_PTXQSAV_5               0x00200000U            /*!<Bit 5 */
#define USB_OTG_HPTXSTS_PTXQSAV_6               0x00400000U            /*!<Bit 6 */
#define USB_OTG_HPTXSTS_PTXQSAV_7               0x00800000U            /*!<Bit 7 */

#define USB_OTG_HPTXSTS_PTXQTOP                 0xFF000000U            /*!< Top of the periodic transmit request queue */
#define USB_OTG_HPTXSTS_PTXQTOP_0               0x01000000U            /*!<Bit 0 */
#define USB_OTG_HPTXSTS_PTXQTOP_1               0x02000000U            /*!<Bit 1 */
#define USB_OTG_HPTXSTS_PTXQTOP_2               0x04000000U            /*!<Bit 2 */
#define USB_OTG_HPTXSTS_PTXQTOP_3               0x08000000U            /*!<Bit 3 */
#define USB_OTG_HPTXSTS_PTXQTOP_4               0x10000000U            /*!<Bit 4 */
#define USB_OTG_HPTXSTS_PTXQTOP_5               0x20000000U            /*!<Bit 5 */
#define USB_OTG_HPTXSTS_PTXQTOP_6               0x40000000U            /*!<Bit 6 */
#define USB_OTG_HPTXSTS_PTXQTOP_7               0x80000000U            /*!<Bit 7 */

/********************  Bit definition forUSB_OTG_HAINT register  ********************/
#define USB_OTG_HAINT_HAINT                   0x0000FFFFU            /*!< Channel interrupts */

/********************  Bit definition forUSB_OTG_DOEPMSK register  ********************/
#define USB_OTG_DOEPMSK_XFRCM                   0x00000001U            /*!< Transfer completed interrupt mask */
#define USB_OTG_DOEPMSK_EPDM                    0x00000002U            /*!< Endpoint disabled interrupt mask */
#define USB_OTG_DOEPMSK_STUPM                   0x00000008U            /*!< SETUP phase done mask */
#define USB_OTG_DOEPMSK_OTEPDM                  0x00000010U            /*!< OUT token received when endpoint disabled mask */
#define USB_OTG_DOEPMSK_B2BSTUP                 0x00000040U            /*!< Back-to-back SETUP packets received mask */
#define USB_OTG_DOEPMSK_OPEM                    0x00000100U            /*!< OUT packet error mask */
#define USB_OTG_DOEPMSK_BOIM                    0x00000200U            /*!< BNA interrupt mask */

/********************  Bit definition forUSB_OTG_GINTSTS register  ********************/
#define USB_OTG_GINTSTS_CMOD                    0x00000001U            /*!< Current mode of operation */
#define USB_OTG_GINTSTS_MMIS                    0x00000002U            /*!< Mode mismatch interrupt */
#define USB_OTG_GINTSTS_OTGINT                  0x00000004U            /*!< OTG interrupt */
#define USB_OTG_GINTSTS_SOF                     0x00000008U            /*!< Start of frame */
#define USB_OTG_GINTSTS_RXFLVL                  0x00000010U            /*!< RxFIFO nonempty */
#define USB_OTG_GINTSTS_NPTXFE                  0x00000020U            /*!< Nonperiodic TxFIFO empty */
#define USB_OTG_GINTSTS_GINAKEFF                0x00000040U            /*!< Global IN nonperiodic NAK effective */
#define USB_OTG_GINTSTS_BOUTNAKEFF              0x00000080U            /*!< Global OUT NAK effective */
#define USB_OTG_GINTSTS_ESUSP                   0x00000400U            /*!< Early suspend */
#define USB_OTG_GINTSTS_USBSUSP                 0x00000800U            /*!< USB suspend */
#define USB_OTG_GINTSTS_USBRST                  0x00001000U            /*!< USB reset */
#define USB_OTG_GINTSTS_ENUMDNE                 0x00002000U            /*!< Enumeration done */
#define USB_OTG_GINTSTS_ISOODRP                 0x00004000U            /*!< Isochronous OUT packet dropped interrupt */
#define USB_OTG_GINTSTS_EOPF                    0x00008000U            /*!< End of periodic frame interrupt */
#define USB_OTG_GINTSTS_IEPINT                  0x00040000U            /*!< IN endpoint interrupt */
#define USB_OTG_GINTSTS_OEPINT                  0x00080000U            /*!< OUT endpoint interrupt */
#define USB_OTG_GINTSTS_IISOIXFR                0x00100000U            /*!< Incomplete isochronous IN transfer */
#define USB_OTG_GINTSTS_PXFR_INCOMPISOOUT       0x00200000U            /*!< Incomplete periodic transfer */
#define USB_OTG_GINTSTS_DATAFSUSP               0x00400000U            /*!< Data fetch suspended */
#define USB_OTG_GINTSTS_HPRTINT                 0x01000000U            /*!< Host port interrupt */
#define USB_OTG_GINTSTS_HCINT                   0x02000000U            /*!< Host channels interrupt */
#define USB_OTG_GINTSTS_PTXFE                   0x04000000U            /*!< Periodic TxFIFO empty */
#define USB_OTG_GINTSTS_CIDSCHG                 0x10000000U            /*!< Connector ID status change */
#define USB_OTG_GINTSTS_DISCINT                 0x20000000U            /*!< Disconnect detected interrupt */
#define USB_OTG_GINTSTS_SRQINT                  0x40000000U            /*!< Session request/new session detected interrupt */
#define USB_OTG_GINTSTS_WKUINT                  0x80000000U            /*!< Resume/remote wakeup detected interrupt */

/********************  Bit definition forUSB_OTG_GINTMSK register  ********************/
#define USB_OTG_GINTMSK_MMISM                   0x00000002U            /*!< Mode mismatch interrupt mask */
#define USB_OTG_GINTMSK_OTGINT                  0x00000004U            /*!< OTG interrupt mask */
#define USB_OTG_GINTMSK_SOFM                    0x00000008U            /*!< Start of frame mask */
#define USB_OTG_GINTMSK_RXFLVLM                 0x00000010U            /*!< Receive FIFO nonempty mask */
#define USB_OTG_GINTMSK_NPTXFEM                 0x00000020U            /*!< Nonperiodic TxFIFO empty mask */
#define USB_OTG_GINTMSK_GINAKEFFM               0x00000040U            /*!< Global nonperiodic IN NAK effective mask */
#define USB_OTG_GINTMSK_GONAKEFFM               0x00000080U            /*!< Global OUT NAK effective mask */
#define USB_OTG_GINTMSK_ESUSPM                  0x00000400U            /*!< Early suspend mask */
#define USB_OTG_GINTMSK_USBSUSPM                0x00000800U            /*!< USB suspend mask */
#define USB_OTG_GINTMSK_USBRST                  0x00001000U            /*!< USB reset mask */
#define USB_OTG_GINTMSK_ENUMDNEM                0x00002000U            /*!< Enumeration done mask */
#define USB_OTG_GINTMSK_ISOODRPM                0x00004000U            /*!< Isochronous OUT packet dropped interrupt mask */
#define USB_OTG_GINTMSK_EOPFM                   0x00008000U            /*!< End of periodic frame interrupt mask */
#define USB_OTG_GINTMSK_EPMISM                  0x00020000U            /*!< Endpoint mismatch interrupt mask */
#define USB_OTG_GINTMSK_IEPINT                  0x00040000U            /*!< IN endpoints interrupt mask */
#define USB_OTG_GINTMSK_OEPINT                  0x00080000U            /*!< OUT endpoints interrupt mask */
#define USB_OTG_GINTMSK_IISOIXFRM               0x00100000U            /*!< Incomplete isochronous IN transfer mask */
#define USB_OTG_GINTMSK_PXFRM_IISOOXFRM         0x00200000U            /*!< Incomplete periodic transfer mask */
#define USB_OTG_GINTMSK_FSUSPM                  0x00400000U            /*!< Data fetch suspended mask */
#define USB_OTG_GINTMSK_PRTIM                   0x01000000U            /*!< Host port interrupt mask */
#define USB_OTG_GINTMSK_HCIM                    0x02000000U            /*!< Host channels interrupt mask */
#define USB_OTG_GINTMSK_PTXFEM                  0x04000000U            /*!< Periodic TxFIFO empty mask */
#define USB_OTG_GINTMSK_CIDSCHGM                0x10000000U            /*!< Connector ID status change mask */
#define USB_OTG_GINTMSK_DISCINT                 0x20000000U            /*!< Disconnect detected interrupt mask */
#define USB_OTG_GINTMSK_SRQIM                   0x40000000U            /*!< Session request/new session detected interrupt mask */
#define USB_OTG_GINTMSK_WUIM                    0x80000000U            /*!< Resume/remote wakeup detected interrupt mask */

/********************  Bit definition forUSB_OTG_DAINT register  ********************/
#define USB_OTG_DAINT_IEPINT                  0x0000FFFFU            /*!< IN endpoint interrupt bits */
#define USB_OTG_DAINT_OEPINT                  0xFFFF0000U            /*!< OUT endpoint interrupt bits */

/********************  Bit definition forUSB_OTG_HAINTMSK register  ********************/
#define USB_OTG_HAINTMSK_HAINTM                  0x0000FFFFU            /*!< Channel interrupt mask */

/********************  Bit definition for USB_OTG_GRXSTSP register  ********************/
#define USB_OTG_GRXSTSP_EPNUM                    0x0000000FU            /*!< IN EP interrupt mask bits */
#define USB_OTG_GRXSTSP_BCNT                     0x00007FF0U            /*!< OUT EP interrupt mask bits */
#define USB_OTG_GRXSTSP_DPID                     0x00018000U            /*!< OUT EP interrupt mask bits */
#define USB_OTG_GRXSTSP_PKTSTS                   0x001E0000U            /*!< OUT EP interrupt mask bits */

/********************  Bit definition forUSB_OTG_DAINTMSK register  ********************/
#define USB_OTG_DAINTMSK_IEPM                    0x0000FFFFU            /*!< IN EP interrupt mask bits */
#define USB_OTG_DAINTMSK_OEPM                    0xFFFF0000U            /*!< OUT EP interrupt mask bits */

/********************  Bit definition for OTG register  ********************/

#define USB_OTG_CHNUM                   0x0000000FU            /*!< Channel number */
#define USB_OTG_CHNUM_0                 0x00000001U            /*!<Bit 0 */
#define USB_OTG_CHNUM_1                 0x00000002U            /*!<Bit 1 */
#define USB_OTG_CHNUM_2                 0x00000004U            /*!<Bit 2 */
#define USB_OTG_CHNUM_3                 0x00000008U            /*!<Bit 3 */
#define USB_OTG_BCNT                    0x00007FF0U            /*!< Byte count */

#define USB_OTG_DPID                    0x00018000U            /*!< Data PID */
#define USB_OTG_DPID_0                  0x00008000U            /*!<Bit 0 */
#define USB_OTG_DPID_1                  0x00010000U            /*!<Bit 1 */

#define USB_OTG_PKTSTS                  0x001E0000U            /*!< Packet status */
#define USB_OTG_PKTSTS_0                0x00020000U            /*!<Bit 0 */
#define USB_OTG_PKTSTS_1                0x00040000U            /*!<Bit 1 */
#define USB_OTG_PKTSTS_2                0x00080000U            /*!<Bit 2 */
#define USB_OTG_PKTSTS_3                0x00100000U            /*!<Bit 3 */

#define USB_OTG_EPNUM                   0x0000000FU            /*!< Endpoint number */
#define USB_OTG_EPNUM_0                 0x00000001U            /*!<Bit 0 */
#define USB_OTG_EPNUM_1                 0x00000002U            /*!<Bit 1 */
#define USB_OTG_EPNUM_2                 0x00000004U            /*!<Bit 2 */
#define USB_OTG_EPNUM_3                 0x00000008U            /*!<Bit 3 */

#define USB_OTG_FRMNUM                  0x01E00000U            /*!< Frame number */
#define USB_OTG_FRMNUM_0                0x00200000U            /*!<Bit 0 */
#define USB_OTG_FRMNUM_1                0x00400000U            /*!<Bit 1 */
#define USB_OTG_FRMNUM_2                0x00800000U            /*!<Bit 2 */
#define USB_OTG_FRMNUM_3                0x01000000U            /*!<Bit 3 */

/********************  Bit definition for OTG register  ********************/

#define USB_OTG_CHNUM                   0x0000000FU            /*!< Channel number */
#define USB_OTG_CHNUM_0                 0x00000001U            /*!<Bit 0 */
#define USB_OTG_CHNUM_1                 0x00000002U            /*!<Bit 1 */
#define USB_OTG_CHNUM_2                 0x00000004U            /*!<Bit 2 */
#define USB_OTG_CHNUM_3                 0x00000008U            /*!<Bit 3 */
#define USB_OTG_BCNT                    0x00007FF0U            /*!< Byte count */

#define USB_OTG_DPID                    0x00018000U            /*!< Data PID */
#define USB_OTG_DPID_0                  0x00008000U            /*!<Bit 0 */
#define USB_OTG_DPID_1                  0x00010000U            /*!<Bit 1 */

#define USB_OTG_PKTSTS                  0x001E0000U            /*!< Packet status */
#define USB_OTG_PKTSTS_0                0x00020000U            /*!<Bit 0 */
#define USB_OTG_PKTSTS_1                0x00040000U            /*!<Bit 1 */
#define USB_OTG_PKTSTS_2                0x00080000U            /*!<Bit 2 */
#define USB_OTG_PKTSTS_3                0x00100000U            /*!<Bit 3 */

#define USB_OTG_EPNUM                   0x0000000FU            /*!< Endpoint number */
#define USB_OTG_EPNUM_0                 0x00000001U            /*!<Bit 0 */
#define USB_OTG_EPNUM_1                 0x00000002U            /*!<Bit 1 */
#define USB_OTG_EPNUM_2                 0x00000004U            /*!<Bit 2 */
#define USB_OTG_EPNUM_3                 0x00000008U            /*!<Bit 3 */

#define USB_OTG_FRMNUM                  0x01E00000U            /*!< Frame number */
#define USB_OTG_FRMNUM_0                0x00200000U            /*!<Bit 0 */
#define USB_OTG_FRMNUM_1                0x00400000U            /*!<Bit 1 */
#define USB_OTG_FRMNUM_2                0x00800000U            /*!<Bit 2 */
#define USB_OTG_FRMNUM_3                0x01000000U            /*!<Bit 3 */

/********************  Bit definition forUSB_OTG_GRXFSIZ register  ********************/
#define USB_OTG_GRXFSIZ_RXFD                    0x0000FFFFU            /*!< RxFIFO depth */

/********************  Bit definition forUSB_OTG_DVBUSDIS register  ********************/
#define USB_OTG_DVBUSDIS_VBUSDT                  0x0000FFFFU            /*!< Device VBUS discharge time */

/********************  Bit definition for OTG register  ********************/
#define USB_OTG_NPTXFSA                 0x0000FFFFU            /*!< Nonperiodic transmit RAM start address */
#define USB_OTG_NPTXFD                  0xFFFF0000U            /*!< Nonperiodic TxFIFO depth */
#define USB_OTG_TX0FSA                  0x0000FFFFU            /*!< Endpoint 0 transmit RAM start address */
#define USB_OTG_TX0FD                   0xFFFF0000U            /*!< Endpoint 0 TxFIFO depth */

/********************  Bit definition forUSB_OTG_DVBUSPULSE register  ********************/
#define USB_OTG_DVBUSPULSE_DVBUSP                  0x00000FFFU            /*!< Device VBUS pulsing time */

/********************  Bit definition forUSB_OTG_GNPTXSTS register  ********************/
#define USB_OTG_GNPTXSTS_NPTXFSAV                0x0000FFFFU            /*!< Nonperiodic TxFIFO space available */

#define USB_OTG_GNPTXSTS_NPTQXSAV                0x00FF0000U            /*!< Nonperiodic transmit request queue space available */
#define USB_OTG_GNPTXSTS_NPTQXSAV_0              0x00010000U            /*!<Bit 0 */
#define USB_OTG_GNPTXSTS_NPTQXSAV_1              0x00020000U            /*!<Bit 1 */
#define USB_OTG_GNPTXSTS_NPTQXSAV_2              0x00040000U            /*!<Bit 2 */
#define USB_OTG_GNPTXSTS_NPTQXSAV_3              0x00080000U            /*!<Bit 3 */
#define USB_OTG_GNPTXSTS_NPTQXSAV_4              0x00100000U            /*!<Bit 4 */
#define USB_OTG_GNPTXSTS_NPTQXSAV_5              0x00200000U            /*!<Bit 5 */
#define USB_OTG_GNPTXSTS_NPTQXSAV_6              0x00400000U            /*!<Bit 6 */
#define USB_OTG_GNPTXSTS_NPTQXSAV_7              0x00800000U            /*!<Bit 7 */

#define USB_OTG_GNPTXSTS_NPTXQTOP                0x7F000000U            /*!< Top of the nonperiodic transmit request queue */
#define USB_OTG_GNPTXSTS_NPTXQTOP_0              0x01000000U            /*!<Bit 0 */
#define USB_OTG_GNPTXSTS_NPTXQTOP_1              0x02000000U            /*!<Bit 1 */
#define USB_OTG_GNPTXSTS_NPTXQTOP_2              0x04000000U            /*!<Bit 2 */
#define USB_OTG_GNPTXSTS_NPTXQTOP_3              0x08000000U            /*!<Bit 3 */
#define USB_OTG_GNPTXSTS_NPTXQTOP_4              0x10000000U            /*!<Bit 4 */
#define USB_OTG_GNPTXSTS_NPTXQTOP_5              0x20000000U            /*!<Bit 5 */
#define USB_OTG_GNPTXSTS_NPTXQTOP_6              0x40000000U            /*!<Bit 6 */

/********************  Bit definition forUSB_OTG_DTHRCTL register  ********************/
#define USB_OTG_DTHRCTL_NONISOTHREN             0x00000001U            /*!< Nonisochronous IN endpoints threshold enable */
#define USB_OTG_DTHRCTL_ISOTHREN                0x00000002U            /*!< ISO IN endpoint threshold enable */

#define USB_OTG_DTHRCTL_TXTHRLEN                0x000007FCU            /*!< Transmit threshold length */
#define USB_OTG_DTHRCTL_TXTHRLEN_0              0x00000004U            /*!<Bit 0 */
#define USB_OTG_DTHRCTL_TXTHRLEN_1              0x00000008U            /*!<Bit 1 */
#define USB_OTG_DTHRCTL_TXTHRLEN_2              0x00000010U            /*!<Bit 2 */
#define USB_OTG_DTHRCTL_TXTHRLEN_3              0x00000020U            /*!<Bit 3 */
#define USB_OTG_DTHRCTL_TXTHRLEN_4              0x00000040U            /*!<Bit 4 */
#define USB_OTG_DTHRCTL_TXTHRLEN_5              0x00000080U            /*!<Bit 5 */
#define USB_OTG_DTHRCTL_TXTHRLEN_6              0x00000100U            /*!<Bit 6 */
#define USB_OTG_DTHRCTL_TXTHRLEN_7              0x00000200U            /*!<Bit 7 */
#define USB_OTG_DTHRCTL_TXTHRLEN_8              0x00000400U            /*!<Bit 8 */
#define USB_OTG_DTHRCTL_RXTHREN                 0x00010000U            /*!< Receive threshold enable */

#define USB_OTG_DTHRCTL_RXTHRLEN                0x03FE0000U            /*!< Receive threshold length */
#define USB_OTG_DTHRCTL_RXTHRLEN_0              0x00020000U            /*!<Bit 0 */
#define USB_OTG_DTHRCTL_RXTHRLEN_1              0x00040000U            /*!<Bit 1 */
#define USB_OTG_DTHRCTL_RXTHRLEN_2              0x00080000U            /*!<Bit 2 */
#define USB_OTG_DTHRCTL_RXTHRLEN_3              0x00100000U            /*!<Bit 3 */
#define USB_OTG_DTHRCTL_RXTHRLEN_4              0x00200000U            /*!<Bit 4 */
#define USB_OTG_DTHRCTL_RXTHRLEN_5              0x00400000U            /*!<Bit 5 */
#define USB_OTG_DTHRCTL_RXTHRLEN_6              0x00800000U            /*!<Bit 6 */
#define USB_OTG_DTHRCTL_RXTHRLEN_7              0x01000000U            /*!<Bit 7 */
#define USB_OTG_DTHRCTL_RXTHRLEN_8              0x02000000U            /*!<Bit 8 */
#define USB_OTG_DTHRCTL_ARPEN                   0x08000000U            /*!< Arbiter parking enable */

/********************  Bit definition forUSB_OTG_DIEPEMPMSK register  ********************/
#define USB_OTG_DIEPEMPMSK_INEPTXFEM               0x0000FFFFU            /*!< IN EP Tx FIFO empty interrupt mask bits */

/********************  Bit definition forUSB_OTG_DEACHINT register  ********************/
#define USB_OTG_DEACHINT_IEP1INT                 0x00000002U            /*!< IN endpoint 1interrupt bit */
#define USB_OTG_DEACHINT_OEP1INT                 0x00020000U            /*!< OUT endpoint 1 interrupt bit */

/********************  Bit definition forUSB_OTG_GCCFG register  ********************/
#define USB_OTG_GCCFG_PWRDWN                  0x00010000U            /*!< Power down */
#define USB_OTG_GCCFG_I2CPADEN                0x00020000U            /*!< Enable I2C bus connection for the external I2C PHY interface */
#define USB_OTG_GCCFG_VBUSASEN                0x00040000U            /*!< Enable the VBUS sensing device */
#define USB_OTG_GCCFG_VBUSBSEN                0x00080000U            /*!< Enable the VBUS sensing device */
#define USB_OTG_GCCFG_SOFOUTEN                0x00100000U            /*!< SOF output enable */
#define USB_OTG_GCCFG_NOVBUSSENS              0x00200000U            /*!< VBUS sensing disable option */

/********************  Bit definition forUSB_OTG_DEACHINTMSK register  ********************/
#define USB_OTG_DEACHINTMSK_IEP1INTM                0x00000002U            /*!< IN Endpoint 1 interrupt mask bit */
#define USB_OTG_DEACHINTMSK_OEP1INTM                0x00020000U            /*!< OUT Endpoint 1 interrupt mask bit */

/********************  Bit definition forUSB_OTG_CID register  ********************/
#define USB_OTG_CID_PRODUCT_ID              0xFFFFFFFFU            /*!< Product ID field */

/********************  Bit definition forUSB_OTG_DIEPEACHMSK1 register  ********************/
#define USB_OTG_DIEPEACHMSK1_XFRCM                   0x00000001U            /*!< Transfer completed interrupt mask */
#define USB_OTG_DIEPEACHMSK1_EPDM                    0x00000002U            /*!< Endpoint disabled interrupt mask */
#define USB_OTG_DIEPEACHMSK1_TOM                     0x00000008U            /*!< Timeout condition mask (nonisochronous endpoints) */
#define USB_OTG_DIEPEACHMSK1_ITTXFEMSK               0x00000010U            /*!< IN token received when TxFIFO empty mask */
#define USB_OTG_DIEPEACHMSK1_INEPNMM                 0x00000020U            /*!< IN token received with EP mismatch mask */
#define USB_OTG_DIEPEACHMSK1_INEPNEM                 0x00000040U            /*!< IN endpoint NAK effective mask */
#define USB_OTG_DIEPEACHMSK1_TXFURM                  0x00000100U            /*!< FIFO underrun mask */
#define USB_OTG_DIEPEACHMSK1_BIM                     0x00000200U            /*!< BNA interrupt mask */
#define USB_OTG_DIEPEACHMSK1_NAKM                    0x00002000U            /*!< NAK interrupt mask */

/********************  Bit definition forUSB_OTG_HPRT register  ********************/
#define USB_OTG_HPRT_PCSTS                   0x00000001U            /*!< Port connect status */
#define USB_OTG_HPRT_PCDET                   0x00000002U            /*!< Port connect detected */
#define USB_OTG_HPRT_PENA                    0x00000004U            /*!< Port enable */
#define USB_OTG_HPRT_PENCHNG                 0x00000008U            /*!< Port enable/disable change */
#define USB_OTG_HPRT_POCA                    0x00000010U            /*!< Port overcurrent active */
#define USB_OTG_HPRT_POCCHNG                 0x00000020U            /*!< Port overcurrent change */
#define USB_OTG_HPRT_PRES                    0x00000040U            /*!< Port resume */
#define USB_OTG_HPRT_PSUSP                   0x00000080U            /*!< Port suspend */
#define USB_OTG_HPRT_PRST                    0x00000100U            /*!< Port reset */

#define USB_OTG_HPRT_PLSTS                   0x00000C00U            /*!< Port line status */
#define USB_OTG_HPRT_PLSTS_0                 0x00000400U            /*!<Bit 0 */
#define USB_OTG_HPRT_PLSTS_1                 0x00000800U            /*!<Bit 1 */
#define USB_OTG_HPRT_PPWR                    0x00001000U            /*!< Port power */

#define USB_OTG_HPRT_PTCTL                   0x0001E000U            /*!< Port test control */
#define USB_OTG_HPRT_PTCTL_0                 0x00002000U            /*!<Bit 0 */
#define USB_OTG_HPRT_PTCTL_1                 0x00004000U            /*!<Bit 1 */
#define USB_OTG_HPRT_PTCTL_2                 0x00008000U            /*!<Bit 2 */
#define USB_OTG_HPRT_PTCTL_3                 0x00010000U            /*!<Bit 3 */

#define USB_OTG_HPRT_PSPD                    0x00060000U            /*!< Port speed */
#define USB_OTG_HPRT_PSPD_0                  0x00020000U            /*!<Bit 0 */
#define USB_OTG_HPRT_PSPD_1                  0x00040000U            /*!<Bit 1 */

/********************  Bit definition forUSB_OTG_DOEPEACHMSK1 register  ********************/
#define USB_OTG_DOEPEACHMSK1_XFRCM                   0x00000001U            /*!< Transfer completed interrupt mask */
#define USB_OTG_DOEPEACHMSK1_EPDM                    0x00000002U            /*!< Endpoint disabled interrupt mask */
#define USB_OTG_DOEPEACHMSK1_TOM                     0x00000008U            /*!< Timeout condition mask */
#define USB_OTG_DOEPEACHMSK1_ITTXFEMSK               0x00000010U            /*!< IN token received when TxFIFO empty mask */
#define USB_OTG_DOEPEACHMSK1_INEPNMM                 0x00000020U            /*!< IN token received with EP mismatch mask */
#define USB_OTG_DOEPEACHMSK1_INEPNEM                 0x00000040U            /*!< IN endpoint NAK effective mask */
#define USB_OTG_DOEPEACHMSK1_TXFURM                  0x00000100U            /*!< OUT packet error mask */
#define USB_OTG_DOEPEACHMSK1_BIM                     0x00000200U            /*!< BNA interrupt mask */
#define USB_OTG_DOEPEACHMSK1_BERRM                   0x00001000U            /*!< Bubble error interrupt mask */
#define USB_OTG_DOEPEACHMSK1_NAKM                    0x00002000U            /*!< NAK interrupt mask */
#define USB_OTG_DOEPEACHMSK1_NYETM                   0x00004000U            /*!< NYET interrupt mask */

/********************  Bit definition forUSB_OTG_HPTXFSIZ register  ********************/
#define USB_OTG_HPTXFSIZ_PTXSA                   0x0000FFFFU            /*!< Host periodic TxFIFO start address */
#define USB_OTG_HPTXFSIZ_PTXFD                   0xFFFF0000U            /*!< Host periodic TxFIFO depth */

/********************  Bit definition forUSB_OTG_DIEPCTL register  ********************/
#define USB_OTG_DIEPCTL_MPSIZ                   0x000007FFU            /*!< Maximum packet size */
#define USB_OTG_DIEPCTL_USBAEP                  0x00008000U            /*!< USB active endpoint */
#define USB_OTG_DIEPCTL_EONUM_DPID              0x00010000U            /*!< Even/odd frame */
#define USB_OTG_DIEPCTL_NAKSTS                  0x00020000U            /*!< NAK status */

#define USB_OTG_DIEPCTL_EPTYP                   0x000C0000U            /*!< Endpoint type */
#define USB_OTG_DIEPCTL_EPTYP_0                 0x00040000U            /*!<Bit 0 */
#define USB_OTG_DIEPCTL_EPTYP_1                 0x00080000U            /*!<Bit 1 */
#define USB_OTG_DIEPCTL_STALL                   0x00200000U            /*!< STALL handshake */

#define USB_OTG_DIEPCTL_TXFNUM                  0x03C00000U            /*!< TxFIFO number */
#define USB_OTG_DIEPCTL_TXFNUM_0                0x00400000U            /*!<Bit 0 */
#define USB_OTG_DIEPCTL_TXFNUM_1                0x00800000U            /*!<Bit 1 */
#define USB_OTG_DIEPCTL_TXFNUM_2                0x01000000U            /*!<Bit 2 */
#define USB_OTG_DIEPCTL_TXFNUM_3                0x02000000U            /*!<Bit 3 */
#define USB_OTG_DIEPCTL_CNAK                    0x04000000U            /*!< Clear NAK */
#define USB_OTG_DIEPCTL_SNAK                    0x08000000U            /*!< Set NAK */
#define USB_OTG_DIEPCTL_SD0PID_SEVNFRM          0x10000000U            /*!< Set DATA0 PID */
#define USB_OTG_DIEPCTL_SODDFRM                 0x20000000U            /*!< Set odd frame */
#define USB_OTG_DIEPCTL_EPDIS                   0x40000000U            /*!< Endpoint disable */
#define USB_OTG_DIEPCTL_EPENA                   0x80000000U            /*!< Endpoint enable */

/********************  Bit definition forUSB_OTG_HCCHAR register  ********************/
#define USB_OTG_HCCHAR_MPSIZ                   0x000007FFU            /*!< Maximum packet size */

#define USB_OTG_HCCHAR_EPNUM                   0x00007800U            /*!< Endpoint number */
#define USB_OTG_HCCHAR_EPNUM_0                 0x00000800U            /*!<Bit 0 */
#define USB_OTG_HCCHAR_EPNUM_1                 0x00001000U            /*!<Bit 1 */
#define USB_OTG_HCCHAR_EPNUM_2                 0x00002000U            /*!<Bit 2 */
#define USB_OTG_HCCHAR_EPNUM_3                 0x00004000U            /*!<Bit 3 */
#define USB_OTG_HCCHAR_EPDIR                   0x00008000U            /*!< Endpoint direction */
#define USB_OTG_HCCHAR_LSDEV                   0x00020000U            /*!< Low-speed device */

#define USB_OTG_HCCHAR_EPTYP                   0x000C0000U            /*!< Endpoint type */
#define USB_OTG_HCCHAR_EPTYP_0                 0x00040000U            /*!<Bit 0 */
#define USB_OTG_HCCHAR_EPTYP_1                 0x00080000U            /*!<Bit 1 */

#define USB_OTG_HCCHAR_MC                      0x00300000U            /*!< Multi Count (MC) / Error Count (EC) */
#define USB_OTG_HCCHAR_MC_0                    0x00100000U            /*!<Bit 0 */
#define USB_OTG_HCCHAR_MC_1                    0x00200000U            /*!<Bit 1 */

#define USB_OTG_HCCHAR_DAD                     0x1FC00000U            /*!< Device address */
#define USB_OTG_HCCHAR_DAD_0                   0x00400000U            /*!<Bit 0 */
#define USB_OTG_HCCHAR_DAD_1                   0x00800000U            /*!<Bit 1 */
#define USB_OTG_HCCHAR_DAD_2                   0x01000000U            /*!<Bit 2 */
#define USB_OTG_HCCHAR_DAD_3                   0x02000000U            /*!<Bit 3 */
#define USB_OTG_HCCHAR_DAD_4                   0x04000000U            /*!<Bit 4 */
#define USB_OTG_HCCHAR_DAD_5                   0x08000000U            /*!<Bit 5 */
#define USB_OTG_HCCHAR_DAD_6                   0x10000000U            /*!<Bit 6 */
#define USB_OTG_HCCHAR_ODDFRM                  0x20000000U            /*!< Odd frame */
#define USB_OTG_HCCHAR_CHDIS                   0x40000000U            /*!< Channel disable */
#define USB_OTG_HCCHAR_CHENA                   0x80000000U            /*!< Channel enable */

/********************  Bit definition forUSB_OTG_HCSPLT register  ********************/

#define USB_OTG_HCSPLT_PRTADDR                 0x0000007FU            /*!< Port address */
#define USB_OTG_HCSPLT_PRTADDR_0               0x00000001U            /*!<Bit 0 */
#define USB_OTG_HCSPLT_PRTADDR_1               0x00000002U            /*!<Bit 1 */
#define USB_OTG_HCSPLT_PRTADDR_2               0x00000004U            /*!<Bit 2 */
#define USB_OTG_HCSPLT_PRTADDR_3               0x00000008U            /*!<Bit 3 */
#define USB_OTG_HCSPLT_PRTADDR_4               0x00000010U            /*!<Bit 4 */
#define USB_OTG_HCSPLT_PRTADDR_5               0x00000020U            /*!<Bit 5 */
#define USB_OTG_HCSPLT_PRTADDR_6               0x00000040U            /*!<Bit 6 */

#define USB_OTG_HCSPLT_HUBADDR                 0x00003F80U            /*!< Hub address */
#define USB_OTG_HCSPLT_HUBADDR_0               0x00000080U            /*!<Bit 0 */
#define USB_OTG_HCSPLT_HUBADDR_1               0x00000100U            /*!<Bit 1 */
#define USB_OTG_HCSPLT_HUBADDR_2               0x00000200U            /*!<Bit 2 */
#define USB_OTG_HCSPLT_HUBADDR_3               0x00000400U            /*!<Bit 3 */
#define USB_OTG_HCSPLT_HUBADDR_4               0x00000800U            /*!<Bit 4 */
#define USB_OTG_HCSPLT_HUBADDR_5               0x00001000U            /*!<Bit 5 */
#define USB_OTG_HCSPLT_HUBADDR_6               0x00002000U            /*!<Bit 6 */

#define USB_OTG_HCSPLT_XACTPOS                 0x0000C000U            /*!< XACTPOS */
#define USB_OTG_HCSPLT_XACTPOS_0               0x00004000U            /*!<Bit 0 */
#define USB_OTG_HCSPLT_XACTPOS_1               0x00008000U            /*!<Bit 1 */
#define USB_OTG_HCSPLT_COMPLSPLT               0x00010000U            /*!< Do complete split */
#define USB_OTG_HCSPLT_SPLITEN                 0x80000000U            /*!< Split enable */

/********************  Bit definition forUSB_OTG_HCINT register  ********************/
#define USB_OTG_HCINT_XFRC                    0x00000001U            /*!< Transfer completed */
#define USB_OTG_HCINT_CHH                     0x00000002U            /*!< Channel halted */
#define USB_OTG_HCINT_AHBERR                  0x00000004U            /*!< AHB error */
#define USB_OTG_HCINT_STALL                   0x00000008U            /*!< STALL response received interrupt */
#define USB_OTG_HCINT_NAK                     0x00000010U            /*!< NAK response received interrupt */
#define USB_OTG_HCINT_ACK                     0x00000020U            /*!< ACK response received/transmitted interrupt */
#define USB_OTG_HCINT_NYET                    0x00000040U            /*!< Response received interrupt */
#define USB_OTG_HCINT_TXERR                   0x00000080U            /*!< Transaction error */
#define USB_OTG_HCINT_BBERR                   0x00000100U            /*!< Babble error */
#define USB_OTG_HCINT_FRMOR                   0x00000200U            /*!< Frame overrun */
#define USB_OTG_HCINT_DTERR                   0x00000400U            /*!< Data toggle error */

/********************  Bit definition forUSB_OTG_DIEPINT register  ********************/
#define USB_OTG_DIEPINT_XFRC                    0x00000001U            /*!< Transfer completed interrupt */
#define USB_OTG_DIEPINT_EPDISD                  0x00000002U            /*!< Endpoint disabled interrupt */
#define USB_OTG_DIEPINT_TOC                     0x00000008U            /*!< Timeout condition */
#define USB_OTG_DIEPINT_ITTXFE                  0x00000010U            /*!< IN token received when TxFIFO is empty */
#define USB_OTG_DIEPINT_INEPNE                  0x00000040U            /*!< IN endpoint NAK effective */
#define USB_OTG_DIEPINT_TXFE                    0x00000080U            /*!< Transmit FIFO empty */
#define USB_OTG_DIEPINT_TXFIFOUDRN              0x00000100U            /*!< Transmit Fifo Underrun */
#define USB_OTG_DIEPINT_BNA                     0x00000200U            /*!< Buffer not available interrupt */
#define USB_OTG_DIEPINT_PKTDRPSTS               0x00000800U            /*!< Packet dropped status */
#define USB_OTG_DIEPINT_BERR                    0x00001000U            /*!< Babble error interrupt */
#define USB_OTG_DIEPINT_NAK                     0x00002000U            /*!< NAK interrupt */

/********************  Bit definition forUSB_OTG_HCINTMSK register  ********************/
#define USB_OTG_HCINTMSK_XFRCM                   0x00000001U            /*!< Transfer completed mask */
#define USB_OTG_HCINTMSK_CHHM                    0x00000002U            /*!< Channel halted mask */
#define USB_OTG_HCINTMSK_AHBERR                  0x00000004U            /*!< AHB error */
#define USB_OTG_HCINTMSK_STALLM                  0x00000008U            /*!< STALL response received interrupt mask */
#define USB_OTG_HCINTMSK_NAKM                    0x00000010U            /*!< NAK response received interrupt mask */
#define USB_OTG_HCINTMSK_ACKM                    0x00000020U            /*!< ACK response received/transmitted interrupt mask */
#define USB_OTG_HCINTMSK_NYET                    0x00000040U            /*!< response received interrupt mask */
#define USB_OTG_HCINTMSK_TXERRM                  0x00000080U            /*!< Transaction error mask */
#define USB_OTG_HCINTMSK_BBERRM                  0x00000100U            /*!< Babble error mask */
#define USB_OTG_HCINTMSK_FRMORM                  0x00000200U            /*!< Frame overrun mask */
#define USB_OTG_HCINTMSK_DTERRM                  0x00000400U            /*!< Data toggle error mask */

/********************  Bit definition for USB_OTG_DIEPTSIZ register  ********************/

#define USB_OTG_DIEPTSIZ_XFRSIZ                  0x0007FFFFU            /*!< Transfer size */
#define USB_OTG_DIEPTSIZ_PKTCNT                  0x1FF80000U            /*!< Packet count */
#define USB_OTG_DIEPTSIZ_MULCNT                  0x60000000U            /*!< Packet count */
/********************  Bit definition forUSB_OTG_HCTSIZ register  ********************/
#define USB_OTG_HCTSIZ_XFRSIZ                    0x0007FFFFU            /*!< Transfer size */
#define USB_OTG_HCTSIZ_PKTCNT                    0x1FF80000U            /*!< Packet count */
#define USB_OTG_HCTSIZ_DOPING                    0x80000000U            /*!< Do PING */
#define USB_OTG_HCTSIZ_DPID                      0x60000000U            /*!< Data PID */
#define USB_OTG_HCTSIZ_DPID_0                    0x20000000U            /*!<Bit 0 */
#define USB_OTG_HCTSIZ_DPID_1                    0x40000000U            /*!<Bit 1 */

/********************  Bit definition forUSB_OTG_DIEPDMA register  ********************/
#define USB_OTG_DIEPDMA_DMAADDR                  0xFFFFFFFFU            /*!< DMA address */

/********************  Bit definition forUSB_OTG_HCDMA register  ********************/
#define USB_OTG_HCDMA_DMAADDR                    0xFFFFFFFFU            /*!< DMA address */

/********************  Bit definition forUSB_OTG_DTXFSTS register  ********************/
#define USB_OTG_DTXFSTS_INEPTFSAV                0x0000FFFFU            /*!< IN endpoint TxFIFO space avail */

/********************  Bit definition forUSB_OTG_DIEPTXF register  ********************/
#define USB_OTG_DIEPTXF_INEPTXSA                 0x0000FFFFU            /*!< IN endpoint FIFOx transmit RAM start address */
#define USB_OTG_DIEPTXF_INEPTXFD                 0xFFFF0000U            /*!< IN endpoint TxFIFO depth */

/********************  Bit definition forUSB_OTG_DOEPCTL register  ********************/

#define USB_OTG_DOEPCTL_MPSIZ                     0x000007FFU            /*!< Maximum packet size */          /*!<Bit 1 */
#define USB_OTG_DOEPCTL_USBAEP                    0x00008000U            /*!< USB active endpoint */
#define USB_OTG_DOEPCTL_NAKSTS                    0x00020000U            /*!< NAK status */
#define USB_OTG_DOEPCTL_SD0PID_SEVNFRM            0x10000000U            /*!< Set DATA0 PID */
#define USB_OTG_DOEPCTL_SODDFRM                   0x20000000U            /*!< Set odd frame */
#define USB_OTG_DOEPCTL_EPTYP                     0x000C0000U            /*!< Endpoint type */
#define USB_OTG_DOEPCTL_EPTYP_0                   0x00040000U            /*!<Bit 0 */
#define USB_OTG_DOEPCTL_EPTYP_1                   0x00080000U            /*!<Bit 1 */
#define USB_OTG_DOEPCTL_SNPM                      0x00100000U            /*!< Snoop mode */
#define USB_OTG_DOEPCTL_STALL                     0x00200000U            /*!< STALL handshake */
#define USB_OTG_DOEPCTL_CNAK                      0x04000000U            /*!< Clear NAK */
#define USB_OTG_DOEPCTL_SNAK                      0x08000000U            /*!< Set NAK */
#define USB_OTG_DOEPCTL_EPDIS                     0x40000000U            /*!< Endpoint disable */
#define USB_OTG_DOEPCTL_EPENA                     0x80000000U            /*!< Endpoint enable */

/********************  Bit definition forUSB_OTG_DOEPINT register  ********************/
#define USB_OTG_DOEPINT_XFRC                    0x00000001U            /*!< Transfer completed interrupt */
#define USB_OTG_DOEPINT_EPDISD                  0x00000002U            /*!< Endpoint disabled interrupt */
#define USB_OTG_DOEPINT_STUP                    0x00000008U            /*!< SETUP phase done */
#define USB_OTG_DOEPINT_OTEPDIS                 0x00000010U            /*!< OUT token received when endpoint disabled */
#define USB_OTG_DOEPINT_B2BSTUP                 0x00000040U            /*!< Back-to-back SETUP packets received */
#define USB_OTG_DOEPINT_NYET                    0x00004000U            /*!< NYET interrupt */

/********************  Bit definition forUSB_OTG_DOEPTSIZ register  ********************/

#define USB_OTG_DOEPTSIZ_XFRSIZ                  0x0007FFFFU            /*!< Transfer size */
#define USB_OTG_DOEPTSIZ_PKTCNT                  0x1FF80000U            /*!< Packet count */

#define USB_OTG_DOEPTSIZ_STUPCNT                 0x60000000U            /*!< SETUP packet count */
#define USB_OTG_DOEPTSIZ_STUPCNT_0               0x20000000U            /*!<Bit 0 */
#define USB_OTG_DOEPTSIZ_STUPCNT_1               0x40000000U            /*!<Bit 1 */

/********************  Bit definition for PCGCCTL register  ********************/
#define USB_OTG_PCGCCTL_STOPCLK                 0x00000001U            /*!< SETUP packet count */
#define USB_OTG_PCGCCTL_GATECLK                 0x00000002U            /*!<Bit 0 */
#define USB_OTG_PCGCCTL_PHYSUSP                 0x00000010U            /*!<Bit 1 */

/**
  * @}
  */ 

/**
  * @}
  */

/** @addtogroup Exported_macros
  * @{
  */
 
/******************************* ADC Instances ********************************/
#define IS_ADC_ALL_INSTANCE(INSTANCE) (((INSTANCE) == ADC1) || \
                                       ((INSTANCE) == ADC2) || \
                                       ((INSTANCE) == ADC3))

/******************************* CAN Instances ********************************/
#define IS_CAN_ALL_INSTANCE(INSTANCE) (((INSTANCE) == CAN1) || \
                                       ((INSTANCE) == CAN2))
 
/******************************* CRC Instances ********************************/
#define IS_CRC_ALL_INSTANCE(INSTANCE) ((INSTANCE) == CRC)

/******************************* DAC Instances ********************************/
#define IS_DAC_ALL_INSTANCE(INSTANCE) ((INSTANCE) == DAC)

/******************************** DMA Instances *******************************/
#define IS_DMA_STREAM_ALL_INSTANCE(INSTANCE) (((INSTANCE) == DMA1_Stream0) || \
                                              ((INSTANCE) == DMA1_Stream1) || \
                                              ((INSTANCE) == DMA1_Stream2) || \
                                              ((INSTANCE) == DMA1_Stream3) || \
                                              ((INSTANCE) == DMA1_Stream4) || \
                                              ((INSTANCE) == DMA1_Stream5) || \
                                              ((INSTANCE) == DMA1_Stream6) || \
                                              ((INSTANCE) == DMA1_Stream7) || \
                                              ((INSTANCE) == DMA2_Stream0) || \
                                              ((INSTANCE) == DMA2_Stream1) || \
                                              ((INSTANCE) == DMA2_Stream2) || \
                                              ((INSTANCE) == DMA2_Stream3) || \
                                              ((INSTANCE) == DMA2_Stream4) || \
                                              ((INSTANCE) == DMA2_Stream5) || \
                                              ((INSTANCE) == DMA2_Stream6) || \
                                              ((INSTANCE) == DMA2_Stream7))

/******************************* GPIO Instances *******************************/
#define IS_GPIO_ALL_INSTANCE(INSTANCE) (((INSTANCE) == GPIOA) || \
                                        ((INSTANCE) == GPIOB) || \
                                        ((INSTANCE) == GPIOC) || \
                                        ((INSTANCE) == GPIOD) || \
                                        ((INSTANCE) == GPIOE) || \
                                        ((INSTANCE) == GPIOF) || \
                                        ((INSTANCE) == GPIOG) || \
                                        ((INSTANCE) == GPIOH) || \
                                        ((INSTANCE) == GPIOI))

/******************************** I2C Instances *******************************/
#define IS_I2C_ALL_INSTANCE(INSTANCE) (((INSTANCE) == I2C1) || \
                                       ((INSTANCE) == I2C2) || \
                                       ((INSTANCE) == I2C3))

/******************************* SMBUS Instances ******************************/
#define IS_SMBUS_ALL_INSTANCE         IS_I2C_ALL_INSTANCE

/******************************** I2S Instances *******************************/
#define IS_I2S_ALL_INSTANCE(INSTANCE)  (((INSTANCE) == SPI2) || \
                                        ((INSTANCE) == SPI3))

/******************************* RNG Instances ********************************/
#define IS_RNG_ALL_INSTANCE(INSTANCE)  ((INSTANCE) == RNG)

/****************************** RTC Instances *********************************/
#define IS_RTC_ALL_INSTANCE(INSTANCE)  ((INSTANCE) == RTC)

/******************************** SPI Instances *******************************/
#define IS_SPI_ALL_INSTANCE(INSTANCE) (((INSTANCE) == SPI1) || \
                                       ((INSTANCE) == SPI2) || \
                                       ((INSTANCE) == SPI3))

/****************** TIM Instances : All supported instances *******************/
#define IS_TIM_INSTANCE(INSTANCE) (((INSTANCE) == TIM1)   || \
                                   ((INSTANCE) == TIM2)   || \
                                   ((INSTANCE) == TIM3)   || \
                                   ((INSTANCE) == TIM4)   || \
                                   ((INSTANCE) == TIM5)   || \
                                   ((INSTANCE) == TIM6)   || \
                                   ((INSTANCE) == TIM7)   || \
                                   ((INSTANCE) == TIM8)   || \
                                   ((INSTANCE) == TIM9)   || \
                                   ((INSTANCE) == TIM10)  || \
                                   ((INSTANCE) == TIM11)  || \
                                   ((INSTANCE) == TIM12)  || \
                                   ((INSTANCE) == TIM13)  || \
                                   ((INSTANCE) == TIM14))

/************* TIM Instances : at least 1 capture/compare channel *************/
#define IS_TIM_CC1_INSTANCE(INSTANCE)   (((INSTANCE) == TIM1)  || \
                                         ((INSTANCE) == TIM2)  || \
                                         ((INSTANCE) == TIM3)  || \
                                         ((INSTANCE) == TIM4)  || \
                                         ((INSTANCE) == TIM5)  || \
                                         ((INSTANCE) == TIM8)  || \
                                         ((INSTANCE) == TIM9)  || \
                                         ((INSTANCE) == TIM10) || \
                                         ((INSTANCE) == TIM11) || \
                                         ((INSTANCE) == TIM12) || \
                                         ((INSTANCE) == TIM13) || \
                                         ((INSTANCE) == TIM14))

/************ TIM Instances : at least 2 capture/compare channels *************/
#define IS_TIM_CC2_INSTANCE(INSTANCE) (((INSTANCE) == TIM1) || \
                                       ((INSTANCE) == TIM2) || \
                                       ((INSTANCE) == TIM3) || \
                                       ((INSTANCE) == TIM4) || \
                                       ((INSTANCE) == TIM5) || \
                                       ((INSTANCE) == TIM8) || \
                                       ((INSTANCE) == TIM9) || \
                                       ((INSTANCE) == TIM12))

/************ TIM Instances : at least 3 capture/compare channels *************/
#define IS_TIM_CC3_INSTANCE(INSTANCE)   (((INSTANCE) == TIM1) || \
                                         ((INSTANCE) == TIM2) || \
                                         ((INSTANCE) == TIM3) || \
                                         ((INSTANCE) == TIM4) || \
                                         ((INSTANCE) == TIM5) || \
                                         ((INSTANCE) == TIM8))

/************ TIM Instances : at least 4 capture/compare channels *************/
#define IS_TIM_CC4_INSTANCE(INSTANCE) (((INSTANCE) == TIM1) || \
                                       ((INSTANCE) == TIM2) || \
                                       ((INSTANCE) == TIM3) || \
                                       ((INSTANCE) == TIM4) || \
                                       ((INSTANCE) == TIM5) || \
                                       ((INSTANCE) == TIM8))

/******************** TIM Instances : Advanced-control timers *****************/
#define IS_TIM_ADVANCED_INSTANCE(INSTANCE) (((INSTANCE) == TIM1) || \
                                            ((INSTANCE) == TIM8))

/******************* TIM Instances : Timer input XOR function *****************/
#define IS_TIM_XOR_INSTANCE(INSTANCE)   (((INSTANCE) == TIM1) || \
                                         ((INSTANCE) == TIM2) || \
                                         ((INSTANCE) == TIM3) || \
                                         ((INSTANCE) == TIM4) || \
                                         ((INSTANCE) == TIM5) || \
                                         ((INSTANCE) == TIM8))

/****************** TIM Instances : DMA requests generation (UDE) *************/
#define IS_TIM_DMA_INSTANCE(INSTANCE) (((INSTANCE) == TIM1) || \
                                       ((INSTANCE) == TIM2) || \
                                       ((INSTANCE) == TIM3) || \
                                       ((INSTANCE) == TIM4) || \
                                       ((INSTANCE) == TIM5) || \
                                       ((INSTANCE) == TIM6) || \
                                       ((INSTANCE) == TIM7) || \
                                       ((INSTANCE) == TIM8))

/************ TIM Instances : DMA requests generation (CCxDE) *****************/
#define IS_TIM_DMA_CC_INSTANCE(INSTANCE) (((INSTANCE) == TIM1) || \
                                          ((INSTANCE) == TIM2) || \
                                          ((INSTANCE) == TIM3) || \
                                          ((INSTANCE) == TIM4) || \
                                          ((INSTANCE) == TIM5) || \
                                          ((INSTANCE) == TIM8))

/************ TIM Instances : DMA requests generation (COMDE) *****************/
#define IS_TIM_CCDMA_INSTANCE(INSTANCE)  (((INSTANCE) == TIM1) || \
                                          ((INSTANCE) == TIM2) || \
                                          ((INSTANCE) == TIM3) || \
                                          ((INSTANCE) == TIM4) || \
                                          ((INSTANCE) == TIM5) || \
                                          ((INSTANCE) == TIM8)) 

/******************** TIM Instances : DMA burst feature ***********************/
#define IS_TIM_DMABURST_INSTANCE(INSTANCE)  (((INSTANCE) == TIM1) || \
                                             ((INSTANCE) == TIM2) || \
                                             ((INSTANCE) == TIM3) || \
                                             ((INSTANCE) == TIM4) || \
                                             ((INSTANCE) == TIM5) || \
                                             ((INSTANCE) == TIM8))

/****** TIM Instances : master mode available (TIMx_CR2.MMS available )********/
#define IS_TIM_MASTER_INSTANCE(INSTANCE) (((INSTANCE) == TIM1) || \
                                          ((INSTANCE) == TIM2) || \
                                          ((INSTANCE) == TIM3) || \
                                          ((INSTANCE) == TIM4) || \
                                          ((INSTANCE) == TIM5) || \
                                          ((INSTANCE) == TIM6) || \
                                          ((INSTANCE) == TIM7) || \
                                          ((INSTANCE) == TIM8) || \
                                          ((INSTANCE) == TIM9) || \
                                          ((INSTANCE) == TIM12))

/*********** TIM Instances : Slave mode available (TIMx_SMCR available )*******/
#define IS_TIM_SLAVE_INSTANCE(INSTANCE) (((INSTANCE) == TIM1) || \
                                         ((INSTANCE) == TIM2) || \
                                         ((INSTANCE) == TIM3) || \
                                         ((INSTANCE) == TIM4) || \
                                         ((INSTANCE) == TIM5) || \
                                         ((INSTANCE) == TIM8) || \
                                         ((INSTANCE) == TIM9) || \
                                         ((INSTANCE) == TIM12))

/********************** TIM Instances : 32 bit Counter ************************/
#define IS_TIM_32B_COUNTER_INSTANCE(INSTANCE)(((INSTANCE) == TIM2) || \
                                              ((INSTANCE) == TIM5))

/***************** TIM Instances : external trigger input available ***********/
#define IS_TIM_ETR_INSTANCE(INSTANCE)  (((INSTANCE) == TIM1) || \
                                        ((INSTANCE) == TIM2) || \
                                        ((INSTANCE) == TIM3) || \
                                        ((INSTANCE) == TIM4) || \
                                        ((INSTANCE) == TIM5) || \
                                        ((INSTANCE) == TIM8))

/****************** TIM Instances : remapping capability **********************/
#define IS_TIM_REMAP_INSTANCE(INSTANCE) (((INSTANCE) == TIM2)  || \
                                         ((INSTANCE) == TIM5)  || \
                                         ((INSTANCE) == TIM11))

/******************* TIM Instances : output(s) available **********************/
#define IS_TIM_CCX_INSTANCE(INSTANCE, CHANNEL) \
    ((((INSTANCE) == TIM1) &&                  \
     (((CHANNEL) == TIM_CHANNEL_1) ||          \
      ((CHANNEL) == TIM_CHANNEL_2) ||          \
      ((CHANNEL) == TIM_CHANNEL_3) ||          \
      ((CHANNEL) == TIM_CHANNEL_4)))           \
    ||                                         \
    (((INSTANCE) == TIM2) &&                   \
     (((CHANNEL) == TIM_CHANNEL_1) ||          \
      ((CHANNEL) == TIM_CHANNEL_2) ||          \
      ((CHANNEL) == TIM_CHANNEL_3) ||          \
      ((CHANNEL) == TIM_CHANNEL_4)))           \
    ||                                         \
    (((INSTANCE) == TIM3) &&                   \
     (((CHANNEL) == TIM_CHANNEL_1) ||          \
      ((CHANNEL) == TIM_CHANNEL_2) ||          \
      ((CHANNEL) == TIM_CHANNEL_3) ||          \
      ((CHANNEL) == TIM_CHANNEL_4)))           \
    ||                                         \
    (((INSTANCE) == TIM4) &&                   \
     (((CHANNEL) == TIM_CHANNEL_1) ||          \
      ((CHANNEL) == TIM_CHANNEL_2) ||          \
      ((CHANNEL) == TIM_CHANNEL_3) ||          \
      ((CHANNEL) == TIM_CHANNEL_4)))           \
    ||                                         \
    (((INSTANCE) == TIM5) &&                   \
     (((CHANNEL) == TIM_CHANNEL_1) ||          \
      ((CHANNEL) == TIM_CHANNEL_2) ||          \
      ((CHANNEL) == TIM_CHANNEL_3) ||          \
      ((CHANNEL) == TIM_CHANNEL_4)))           \
    ||                                         \
    (((INSTANCE) == TIM8) &&                   \
     (((CHANNEL) == TIM_CHANNEL_1) ||          \
      ((CHANNEL) == TIM_CHANNEL_2) ||          \
      ((CHANNEL) == TIM_CHANNEL_3) ||          \
      ((CHANNEL) == TIM_CHANNEL_4)))           \
    ||                                         \
    (((INSTANCE) == TIM9) &&                   \
     (((CHANNEL) == TIM_CHANNEL_1) ||          \
      ((CHANNEL) == TIM_CHANNEL_2)))           \
    ||                                         \
    (((INSTANCE) == TIM10) &&                  \
     (((CHANNEL) == TIM_CHANNEL_1)))           \
    ||                                         \
    (((INSTANCE) == TIM11) &&                  \
     (((CHANNEL) == TIM_CHANNEL_1)))           \
    ||                                         \
    (((INSTANCE) == TIM12) &&                  \
     (((CHANNEL) == TIM_CHANNEL_1) ||          \
      ((CHANNEL) == TIM_CHANNEL_2)))           \
    ||                                         \
    (((INSTANCE) == TIM13) &&                  \
     (((CHANNEL) == TIM_CHANNEL_1)))           \
    ||                                         \
    (((INSTANCE) == TIM14) &&                  \
     (((CHANNEL) == TIM_CHANNEL_1))))

/************ TIM Instances : complementary output(s) available ***************/
#define IS_TIM_CCXN_INSTANCE(INSTANCE, CHANNEL) \
   ((((INSTANCE) == TIM1) &&                    \
     (((CHANNEL) == TIM_CHANNEL_1) ||           \
      ((CHANNEL) == TIM_CHANNEL_2) ||           \
      ((CHANNEL) == TIM_CHANNEL_3)))            \
    ||                                          \
    (((INSTANCE) == TIM8) &&                    \
     (((CHANNEL) == TIM_CHANNEL_1) ||           \
      ((CHANNEL) == TIM_CHANNEL_2) ||           \
      ((CHANNEL) == TIM_CHANNEL_3))))

/******************** USART Instances : Synchronous mode **********************/
#define IS_USART_INSTANCE(INSTANCE) (((INSTANCE) == USART1) || \
                                     ((INSTANCE) == USART2) || \
                                     ((INSTANCE) == USART3) || \
                                     ((INSTANCE) == USART6))

/******************** UART Instances : Asynchronous mode **********************/
#define IS_UART_INSTANCE(INSTANCE) (((INSTANCE) == USART1) || \
                                    ((INSTANCE) == USART2) || \
                                    ((INSTANCE) == USART3) || \
                                    ((INSTANCE) == UART4)  || \
                                    ((INSTANCE) == UART5)  || \
                                    ((INSTANCE) == USART6))

/****************** UART Instances : Hardware Flow control ********************/
#define IS_UART_HWFLOW_INSTANCE(INSTANCE) (((INSTANCE) == USART1) || \
                                           ((INSTANCE) == USART2) || \
                                           ((INSTANCE) == USART3) || \
                                           ((INSTANCE) == USART6))

/********************* UART Instances : Smart card mode ***********************/
#define IS_SMARTCARD_INSTANCE(INSTANCE) (((INSTANCE) == USART1) || \
                                         ((INSTANCE) == USART2) || \
                                         ((INSTANCE) == USART3) || \
                                         ((INSTANCE) == USART6))

/*********************** UART Instances : IRDA mode ***************************/
#define IS_IRDA_INSTANCE(INSTANCE) (((INSTANCE) == USART1) || \
                                    ((INSTANCE) == USART2) || \
                                    ((INSTANCE) == USART3) || \
                                    ((INSTANCE) == UART4)  || \
                                    ((INSTANCE) == UART5)  || \
                                    ((INSTANCE) == USART6))     

/*********************** PCD Instances ****************************************/
#define IS_PCD_ALL_INSTANCE(INSTANCE) (((INSTANCE) == USB_OTG_FS) || \
                                        ((INSTANCE) == USB_OTG_HS))

/*********************** HCD Instances ****************************************/
#define IS_HCD_ALL_INSTANCE(INSTANCE) (((INSTANCE) == USB_OTG_FS) || \
                                       ((INSTANCE) == USB_OTG_HS))

/****************************** IWDG Instances ********************************/
#define IS_IWDG_ALL_INSTANCE(INSTANCE)  ((INSTANCE) == IWDG)

/****************************** WWDG Instances ********************************/
#define IS_WWDG_ALL_INSTANCE(INSTANCE)  ((INSTANCE) == WWDG)

/****************************** SDIO Instances ********************************/
#define IS_SDIO_ALL_INSTANCE(INSTANCE) ((INSTANCE) == SDIO)

/****************************** USB Exported Constants ************************/
#define USB_OTG_FS_HOST_MAX_CHANNEL_NBR                8U
#define USB_OTG_FS_MAX_IN_ENDPOINTS                    4U    /* Including EP0 */
#define USB_OTG_FS_MAX_OUT_ENDPOINTS                   4U    /* Including EP0 */
#define USB_OTG_FS_TOTAL_FIFO_SIZE                     1280U /* in Bytes */

#define USB_OTG_HS_HOST_MAX_CHANNEL_NBR                12U
#define USB_OTG_HS_MAX_IN_ENDPOINTS                    6U    /* Including EP0 */
#define USB_OTG_HS_MAX_OUT_ENDPOINTS                   6U    /* Including EP0 */
#define USB_OTG_HS_TOTAL_FIFO_SIZE                     4096U /* in Bytes */

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
#endif /* __cplusplus */

#endif /* __STM32F205xx_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/

