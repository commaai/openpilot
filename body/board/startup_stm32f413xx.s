/**
  ******************************************************************************
  * @file      startup_stm32f413xx.s
  * @author    MCD Application Team
  * @brief     STM32F413xx Devices vector table for GCC based toolchains.
  *            This module performs:
  *                - Set the initial SP
  *                - Set the initial PC == Reset_Handler,
  *                - Set the vector table entries with the exceptions ISR address
  *                - Branches to main in the C library (which eventually
  *                  calls main()).
  *            After Reset the Cortex-M4 processor is in Thread mode,
  *            priority is Privileged, and the Stack is set to Main.
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

  .syntax unified
  .cpu cortex-m4
  .fpu softvfp
  .thumb

.global  g_pfnVectors
.global  Default_Handler

/* start address for the initialization values of the .data section.
defined in linker script */
.word  _sidata
/* start address for the .data section. defined in linker script */
.word  _sdata
/* end address for the .data section. defined in linker script */
.word  _edata
/* start address for the .bss section. defined in linker script */
.word  _sbss
/* end address for the .bss section. defined in linker script */
.word  _ebss
/* stack used for SystemInit_ExtMemCtl; always internal RAM used */

/**
 * @brief  This is the code that gets called when the processor first
 *          starts execution following a reset event. Only the absolutely
 *          necessary set is performed, after which the application
 *          supplied main() routine is called.
 * @param  None
 * @retval : None
*/

    .section  .text.Reset_Handler
  .weak  Reset_Handler
  .type  Reset_Handler, %function
Reset_Handler:
  ldr   sp, =_estack       /* set stack pointer */
  bl __initialize_hardware_early

/* Copy the data segment initializers from flash to SRAM */
  movs  r1, #0
  b  LoopCopyDataInit

CopyDataInit:
  ldr  r3, =_sidata
  ldr  r3, [r3, r1]
  str  r3, [r0, r1]
  adds  r1, r1, #4

LoopCopyDataInit:
  ldr  r0, =_sdata
  ldr  r3, =_edata
  adds  r2, r0, r1
  cmp  r2, r3
  bcc  CopyDataInit
  ldr  r2, =_sbss
  b  LoopFillZerobss
/* Zero fill the bss segment. */
FillZerobss:
  movs  r3, #0
  str  r3, [r2], #4

LoopFillZerobss:
  ldr  r3, = _ebss
  cmp  r2, r3
  bcc  FillZerobss

/* Call the clock system intitialization function.*/
  /* bl  SystemInit    */
/* Call static constructors */
    /* bl __libc_init_array */
/* Call the application's entry point.*/
  bl  main
  bx  lr
.size  Reset_Handler, .-Reset_Handler

/**
 * @brief  This is the code that gets called when the processor receives an
 *         unexpected interrupt.  This simply enters an infinite loop, preserving
 *         the system state for examination by a debugger.
 * @param  None
 * @retval None
*/
    .section  .text.Default_Handler,"ax",%progbits
Default_Handler:
Infinite_Loop:
  b  Infinite_Loop
  .size  Default_Handler, .-Default_Handler
/******************************************************************************
*
* The minimal vector table for a Cortex M3. Note that the proper constructs
* must be placed on this to ensure that it ends up at physical address
* 0x0000.0000.
*
*******************************************************************************/
   .section  .isr_vector,"a",%progbits
  .type  g_pfnVectors, %object
  .size  g_pfnVectors, .-g_pfnVectors

g_pfnVectors:
  .word  _estack
  .word  Reset_Handler
  .word  NMI_Handler
  .word  HardFault_Handler
  .word  MemManage_Handler
  .word  BusFault_Handler
  .word  UsageFault_Handler
  .word  0
  .word  0
  .word  0
  .word  0
  .word  SVC_Handler
  .word  DebugMon_Handler
  .word  0
  .word  PendSV_Handler
  .word  SysTick_Handler

  /* External Interrupts */
  .word     WWDG_IRQHandler                   /* Window WatchDog                             */
  .word     PVD_IRQHandler                    /* PVD through EXTI Line detection             */
  .word     TAMP_STAMP_IRQHandler             /* Tamper and TimeStamps through the EXTI line */
  .word     RTC_WKUP_IRQHandler               /* RTC Wakeup through the EXTI line            */
  .word     FLASH_IRQHandler                  /* FLASH                                       */
  .word     RCC_IRQHandler                    /* RCC                                         */
  .word     EXTI0_IRQHandler                  /* EXTI Line0                                  */
  .word     EXTI1_IRQHandler                  /* EXTI Line1                                  */
  .word     EXTI2_IRQHandler                  /* EXTI Line2                                  */
  .word     EXTI3_IRQHandler                  /* EXTI Line3                                  */
  .word     EXTI4_IRQHandler                  /* EXTI Line4                                  */
  .word     DMA1_Stream0_IRQHandler           /* DMA1 Stream 0                               */
  .word     DMA1_Stream1_IRQHandler           /* DMA1 Stream 1                               */
  .word     DMA1_Stream2_IRQHandler           /* DMA1 Stream 2                               */
  .word     DMA1_Stream3_IRQHandler           /* DMA1 Stream 3                               */
  .word     DMA1_Stream4_IRQHandler           /* DMA1 Stream 4                               */
  .word     DMA1_Stream5_IRQHandler           /* DMA1 Stream 5                               */
  .word     DMA1_Stream6_IRQHandler           /* DMA1 Stream 6                               */
  .word     ADC_IRQHandler                    /* ADC1, ADC2 and ADC3s                        */
  .word     CAN1_TX_IRQHandler                /* CAN1 TX                                     */
  .word     CAN1_RX0_IRQHandler               /* CAN1 RX0                                    */
  .word     CAN1_RX1_IRQHandler               /* CAN1 RX1                                    */
  .word     CAN1_SCE_IRQHandler               /* CAN1 SCE                                    */
  .word     EXTI9_5_IRQHandler                /* External Line[9:5]s                         */
  .word     TIM1_BRK_TIM9_IRQHandler          /* TIM1 Break and TIM9                         */
  .word     TIM1_UP_TIM10_IRQHandler          /* TIM1 Update and TIM10                       */
  .word     TIM1_TRG_COM_TIM11_IRQHandler     /* TIM1 Trigger and Commutation and TIM11      */
  .word     TIM1_CC_IRQHandler                /* TIM1 Capture Compare                        */
  .word     TIM2_IRQHandler                   /* TIM2                                        */
  .word     TIM3_IRQHandler                   /* TIM3                                        */
  .word     TIM4_IRQHandler                   /* TIM4                                        */
  .word     I2C1_EV_IRQHandler                /* I2C1 Event                                  */
  .word     I2C1_ER_IRQHandler                /* I2C1 Error                                  */
  .word     I2C2_EV_IRQHandler                /* I2C2 Event                                  */
  .word     I2C2_ER_IRQHandler                /* I2C2 Error                                  */
  .word     SPI1_IRQHandler                   /* SPI1                                        */
  .word     SPI2_IRQHandler                   /* SPI2                                        */
  .word     USART1_IRQHandler                 /* USART1                                      */
  .word     USART2_IRQHandler                 /* USART2                                      */
  .word     USART3_IRQHandler                 /* USART3                                      */
  .word     EXTI15_10_IRQHandler              /* External Line[15:10]s                       */
  .word     RTC_Alarm_IRQHandler              /* RTC Alarm (A and B) through EXTI Line       */
  .word     OTG_FS_WKUP_IRQHandler            /* USB OTG FS Wakeup through EXTI line         */
  .word     TIM8_BRK_TIM12_IRQHandler         /* TIM8 Break and TIM12                        */
  .word     TIM8_UP_TIM13_IRQHandler          /* TIM8 Update and TIM13                       */
  .word     TIM8_TRG_COM_TIM14_IRQHandler     /* TIM8 Trigger and Commutation and TIM14      */
  .word     TIM8_CC_IRQHandler                /* TIM8 Capture Compare                        */
  .word     DMA1_Stream7_IRQHandler           /* DMA1 Stream7                                */
  .word     FSMC_IRQHandler                   /* FSMC                                        */
  .word     SDIO_IRQHandler                   /* SDIO                                        */
  .word     TIM5_IRQHandler                   /* TIM5                                        */
  .word     SPI3_IRQHandler                   /* SPI3                                        */
  .word     UART4_IRQHandler                  /* UART4                                       */
  .word     UART5_IRQHandler                  /* UART5                                       */
  .word     TIM6_DAC_IRQHandler               /* TIM6, DAC1 and DAC2                         */
  .word     TIM7_IRQHandler                   /* TIM7                                        */
  .word     DMA2_Stream0_IRQHandler           /* DMA2 Stream 0                               */
  .word     DMA2_Stream1_IRQHandler           /* DMA2 Stream 1                               */
  .word     DMA2_Stream2_IRQHandler           /* DMA2 Stream 2                               */
  .word     DMA2_Stream3_IRQHandler           /* DMA2 Stream 3                               */
  .word     DMA2_Stream4_IRQHandler           /* DMA2 Stream 4                               */
  .word     DFSDM1_FLT0_IRQHandler            /* DFSDM1 Filter0                              */
  .word     DFSDM1_FLT1_IRQHandler            /* DFSDM1 Filter1                              */
  .word     CAN2_TX_IRQHandler                /* CAN2 TX                                     */
  .word     CAN2_RX0_IRQHandler               /* CAN2 RX0                                    */
  .word     CAN2_RX1_IRQHandler               /* CAN2 RX1                                    */
  .word     CAN2_SCE_IRQHandler               /* CAN2 SCE                                    */
  .word     OTG_FS_IRQHandler                 /* USB OTG FS                                  */
  .word     DMA2_Stream5_IRQHandler           /* DMA2 Stream 5                               */
  .word     DMA2_Stream6_IRQHandler           /* DMA2 Stream 6                               */
  .word     DMA2_Stream7_IRQHandler           /* DMA2 Stream 7                               */
  .word     USART6_IRQHandler                 /* USART6                                      */
  .word     I2C3_EV_IRQHandler                /* I2C3 event                                  */
  .word     I2C3_ER_IRQHandler                /* I2C3 error                                  */
  .word     CAN3_TX_IRQHandler                /* CAN3 TX                                     */
  .word     CAN3_RX0_IRQHandler               /* CAN3 RX0                                    */
  .word     CAN3_RX1_IRQHandler               /* CAN3 RX1                                    */
  .word     CAN3_SCE_IRQHandler               /* CAN3 SCE                                    */
  .word     0                                 /* Reserved                                    */
  .word     0                                 /* Reserved                                    */
  .word     RNG_IRQHandler                    /* RNG                                         */
  .word     FPU_IRQHandler                    /* FPU                                         */
  .word     UART7_IRQHandler                  /* UART7                                       */
  .word     UART8_IRQHandler                  /* UART8                                       */
  .word     SPI4_IRQHandler                   /* SPI4                                        */
  .word     SPI5_IRQHandler                   /* SPI5                                        */
  .word     0                                 /* Reserved                                    */
  .word     SAI1_IRQHandler                   /* SAI1                                        */
  .word     UART9_IRQHandler                  /* UART9                                       */
  .word     UART10_IRQHandler                 /* UART10                                      */
  .word     0                                 /* Reserved                                    */
  .word     0                                 /* Reserved                                    */
  .word     QUADSPI_IRQHandler                /* QuadSPI                                     */
  .word     0                                 /* Reserved                                    */
  .word     0                                 /* Reserved                                    */
  .word     FMPI2C1_EV_IRQHandler             /* FMPI2C1 Event                               */
  .word     FMPI2C1_ER_IRQHandler             /* FMPI2C1 Error                               */
  .word     LPTIM1_IRQHandler                 /* LPTIM1                                      */
  .word     DFSDM2_FLT0_IRQHandler            /* DFSDM2 Filter0                              */
  .word     DFSDM2_FLT1_IRQHandler            /* DFSDM2 Filter1                              */
  .word     DFSDM2_FLT2_IRQHandler            /* DFSDM2 Filter2                              */
  .word     DFSDM2_FLT3_IRQHandler            /* DFSDM2 Filter3                              */

/*******************************************************************************
*
* Provide weak aliases for each Exception handler to the Default_Handler.
* As they are weak aliases, any function with the same name will override
* this definition.
*
*******************************************************************************/
   .weak      NMI_Handler
   .thumb_set NMI_Handler,Default_Handler

   .weak      HardFault_Handler
   .thumb_set HardFault_Handler,Default_Handler

   .weak      MemManage_Handler
   .thumb_set MemManage_Handler,Default_Handler

   .weak      BusFault_Handler
   .thumb_set BusFault_Handler,Default_Handler

   .weak      UsageFault_Handler
   .thumb_set UsageFault_Handler,Default_Handler

   .weak      SVC_Handler
   .thumb_set SVC_Handler,Default_Handler

   .weak      DebugMon_Handler
   .thumb_set DebugMon_Handler,Default_Handler

   .weak      PendSV_Handler
   .thumb_set PendSV_Handler,Default_Handler

   .weak      SysTick_Handler
   .thumb_set SysTick_Handler,Default_Handler

   .weak      WWDG_IRQHandler
   .thumb_set WWDG_IRQHandler,Default_Handler

   .weak      PVD_IRQHandler
   .thumb_set PVD_IRQHandler,Default_Handler

   .weak      TAMP_STAMP_IRQHandler
   .thumb_set TAMP_STAMP_IRQHandler,Default_Handler

   .weak      RTC_WKUP_IRQHandler
   .thumb_set RTC_WKUP_IRQHandler,Default_Handler

   .weak      FLASH_IRQHandler
   .thumb_set FLASH_IRQHandler,Default_Handler

   .weak      RCC_IRQHandler
   .thumb_set RCC_IRQHandler,Default_Handler

   .weak      EXTI0_IRQHandler
   .thumb_set EXTI0_IRQHandler,Default_Handler

   .weak      EXTI1_IRQHandler
   .thumb_set EXTI1_IRQHandler,Default_Handler

   .weak      EXTI2_IRQHandler
   .thumb_set EXTI2_IRQHandler,Default_Handler

   .weak      EXTI3_IRQHandler
   .thumb_set EXTI3_IRQHandler,Default_Handler

   .weak      EXTI4_IRQHandler
   .thumb_set EXTI4_IRQHandler,Default_Handler

   .weak      DMA1_Stream0_IRQHandler
   .thumb_set DMA1_Stream0_IRQHandler,Default_Handler

   .weak      DMA1_Stream1_IRQHandler
   .thumb_set DMA1_Stream1_IRQHandler,Default_Handler

   .weak      DMA1_Stream2_IRQHandler
   .thumb_set DMA1_Stream2_IRQHandler,Default_Handler

   .weak      DMA1_Stream3_IRQHandler
   .thumb_set DMA1_Stream3_IRQHandler,Default_Handler

   .weak      DMA1_Stream4_IRQHandler
   .thumb_set DMA1_Stream4_IRQHandler,Default_Handler

   .weak      DMA1_Stream5_IRQHandler
   .thumb_set DMA1_Stream5_IRQHandler,Default_Handler

   .weak      DMA1_Stream6_IRQHandler
   .thumb_set DMA1_Stream6_IRQHandler,Default_Handler

   .weak      ADC_IRQHandler
   .thumb_set ADC_IRQHandler,Default_Handler

   .weak      CAN1_TX_IRQHandler
   .thumb_set CAN1_TX_IRQHandler,Default_Handler

   .weak      CAN1_RX0_IRQHandler
   .thumb_set CAN1_RX0_IRQHandler,Default_Handler

   .weak      CAN1_RX1_IRQHandler
   .thumb_set CAN1_RX1_IRQHandler,Default_Handler

   .weak      CAN1_SCE_IRQHandler
   .thumb_set CAN1_SCE_IRQHandler,Default_Handler

   .weak      EXTI9_5_IRQHandler
   .thumb_set EXTI9_5_IRQHandler,Default_Handler

   .weak      TIM1_BRK_TIM9_IRQHandler
   .thumb_set TIM1_BRK_TIM9_IRQHandler,Default_Handler

   .weak      TIM1_UP_TIM10_IRQHandler
   .thumb_set TIM1_UP_TIM10_IRQHandler,Default_Handler

   .weak      TIM1_TRG_COM_TIM11_IRQHandler
   .thumb_set TIM1_TRG_COM_TIM11_IRQHandler,Default_Handler

   .weak      TIM1_CC_IRQHandler
   .thumb_set TIM1_CC_IRQHandler,Default_Handler

   .weak      TIM2_IRQHandler
   .thumb_set TIM2_IRQHandler,Default_Handler

   .weak      TIM3_IRQHandler
   .thumb_set TIM3_IRQHandler,Default_Handler

   .weak      TIM4_IRQHandler
   .thumb_set TIM4_IRQHandler,Default_Handler

   .weak      I2C1_EV_IRQHandler
   .thumb_set I2C1_EV_IRQHandler,Default_Handler

   .weak      I2C1_ER_IRQHandler
   .thumb_set I2C1_ER_IRQHandler,Default_Handler

   .weak      I2C2_EV_IRQHandler
   .thumb_set I2C2_EV_IRQHandler,Default_Handler

   .weak      I2C2_ER_IRQHandler
   .thumb_set I2C2_ER_IRQHandler,Default_Handler

   .weak      SPI1_IRQHandler
   .thumb_set SPI1_IRQHandler,Default_Handler

   .weak      SPI2_IRQHandler
   .thumb_set SPI2_IRQHandler,Default_Handler

   .weak      USART1_IRQHandler
   .thumb_set USART1_IRQHandler,Default_Handler

   .weak      USART2_IRQHandler
   .thumb_set USART2_IRQHandler,Default_Handler

   .weak      USART3_IRQHandler
   .thumb_set USART3_IRQHandler,Default_Handler

   .weak      EXTI15_10_IRQHandler
   .thumb_set EXTI15_10_IRQHandler,Default_Handler

   .weak      RTC_Alarm_IRQHandler
   .thumb_set RTC_Alarm_IRQHandler,Default_Handler

   .weak      OTG_FS_WKUP_IRQHandler
   .thumb_set OTG_FS_WKUP_IRQHandler,Default_Handler

   .weak      TIM8_BRK_TIM12_IRQHandler
   .thumb_set TIM8_BRK_TIM12_IRQHandler,Default_Handler

   .weak      TIM8_UP_TIM13_IRQHandler
   .thumb_set TIM8_UP_TIM13_IRQHandler,Default_Handler

   .weak      TIM8_TRG_COM_TIM14_IRQHandler
   .thumb_set TIM8_TRG_COM_TIM14_IRQHandler,Default_Handler

   .weak      TIM8_CC_IRQHandler
   .thumb_set TIM8_CC_IRQHandler,Default_Handler

   .weak      DMA1_Stream7_IRQHandler
   .thumb_set DMA1_Stream7_IRQHandler,Default_Handler

   .weak      FSMC_IRQHandler
   .thumb_set FSMC_IRQHandler,Default_Handler

   .weak      SDIO_IRQHandler
   .thumb_set SDIO_IRQHandler,Default_Handler

   .weak      TIM5_IRQHandler
   .thumb_set TIM5_IRQHandler,Default_Handler

   .weak      SPI3_IRQHandler
   .thumb_set SPI3_IRQHandler,Default_Handler

   .weak      UART4_IRQHandler
   .thumb_set UART4_IRQHandler,Default_Handler

   .weak      UART5_IRQHandler
   .thumb_set UART5_IRQHandler,Default_Handler

   .weak      TIM6_DAC_IRQHandler
   .thumb_set TIM6_DAC_IRQHandler,Default_Handler

   .weak      TIM7_IRQHandler
   .thumb_set TIM7_IRQHandler,Default_Handler

   .weak      DMA2_Stream0_IRQHandler
   .thumb_set DMA2_Stream0_IRQHandler,Default_Handler

   .weak      DMA2_Stream1_IRQHandler
   .thumb_set DMA2_Stream1_IRQHandler,Default_Handler

   .weak      DMA2_Stream2_IRQHandler
   .thumb_set DMA2_Stream2_IRQHandler,Default_Handler

   .weak      DMA2_Stream3_IRQHandler
   .thumb_set DMA2_Stream3_IRQHandler,Default_Handler

   .weak      DMA2_Stream4_IRQHandler
   .thumb_set DMA2_Stream4_IRQHandler,Default_Handler

   .weak      DFSDM1_FLT0_IRQHandler
   .thumb_set DFSDM1_FLT0_IRQHandler,Default_Handler

   .weak      DFSDM1_FLT1_IRQHandler
   .thumb_set DFSDM1_FLT1_IRQHandler,Default_Handler

   .weak      CAN2_TX_IRQHandler
   .thumb_set CAN2_TX_IRQHandler,Default_Handler

   .weak      CAN2_RX0_IRQHandler
   .thumb_set CAN2_RX0_IRQHandler,Default_Handler

   .weak      CAN2_RX1_IRQHandler
   .thumb_set CAN2_RX1_IRQHandler,Default_Handler

   .weak      CAN2_SCE_IRQHandler
   .thumb_set CAN2_SCE_IRQHandler,Default_Handler

   .weak      OTG_FS_IRQHandler
   .thumb_set OTG_FS_IRQHandler,Default_Handler

   .weak      DMA2_Stream5_IRQHandler
   .thumb_set DMA2_Stream5_IRQHandler,Default_Handler

   .weak      DMA2_Stream6_IRQHandler
   .thumb_set DMA2_Stream6_IRQHandler,Default_Handler

   .weak      DMA2_Stream7_IRQHandler
   .thumb_set DMA2_Stream7_IRQHandler,Default_Handler

   .weak      USART6_IRQHandler
   .thumb_set USART6_IRQHandler,Default_Handler

   .weak      I2C3_EV_IRQHandler
   .thumb_set I2C3_EV_IRQHandler,Default_Handler

   .weak      I2C3_ER_IRQHandler
   .thumb_set I2C3_ER_IRQHandler,Default_Handler

   .weak      CAN3_TX_IRQHandler
   .thumb_set CAN3_TX_IRQHandler,Default_Handler

   .weak      CAN3_RX0_IRQHandler
   .thumb_set CAN3_RX0_IRQHandler,Default_Handler

   .weak      CAN3_RX1_IRQHandler
   .thumb_set CAN3_RX1_IRQHandler,Default_Handler

   .weak      CAN3_SCE_IRQHandler
   .thumb_set CAN3_SCE_IRQHandler,Default_Handler

   .weak      RNG_IRQHandler
   .thumb_set RNG_IRQHandler,Default_Handler

   .weak      FPU_IRQHandler
   .thumb_set FPU_IRQHandler,Default_Handler

   .weak      UART7_IRQHandler
   .thumb_set UART7_IRQHandler,Default_Handler

   .weak      UART8_IRQHandler
   .thumb_set UART8_IRQHandler,Default_Handler

   .weak      SPI4_IRQHandler
   .thumb_set SPI4_IRQHandler,Default_Handler

   .weak      SPI5_IRQHandler
   .thumb_set SPI5_IRQHandler,Default_Handler

   .weak      SAI1_IRQHandler
   .thumb_set SAI1_IRQHandler,Default_Handler

   .weak      UART9_IRQHandler
   .thumb_set UART9_IRQHandler,Default_Handler

   .weak      UART10_IRQHandler
   .thumb_set UART10_IRQHandler,Default_Handler

   .weak      QUADSPI_IRQHandler
   .thumb_set QUADSPI_IRQHandler,Default_Handler

    .weak     FMPI2C1_EV_IRQHandler
   .thumb_set FMPI2C1_EV_IRQHandler,Default_Handler

   .weak      FMPI2C1_ER_IRQHandler
   .thumb_set FMPI2C1_ER_IRQHandler,Default_Handler

   .weak      LPTIM1_IRQHandler
   .thumb_set LPTIM1_IRQHandler,Default_Handler

   .weak      DFSDM2_FLT0_IRQHandler
   .thumb_set DFSDM2_FLT0_IRQHandler,Default_Handler

   .weak      DFSDM2_FLT1_IRQHandler
   .thumb_set DFSDM2_FLT1_IRQHandler,Default_Handler

   .weak      DFSDM2_FLT2_IRQHandler
   .thumb_set DFSDM2_FLT2_IRQHandler,Default_Handler

   .weak      DFSDM2_FLT3_IRQHandler
   .thumb_set DFSDM2_FLT3_IRQHandler,Default_Handler
/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
