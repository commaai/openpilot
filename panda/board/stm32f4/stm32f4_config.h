#include "stm32f4xx.h"
#include "stm32f4xx_hal_gpio_ex.h"
#define MCU_IDCODE 0x463U

#define CORE_FREQ 96U // in MHz
#define APB1_FREQ (CORE_FREQ/2U)
#define APB1_TIMER_FREQ (APB1_FREQ*2U)  // APB1 is multiplied by 2 for the timer peripherals
#define APB2_FREQ (CORE_FREQ/2U)
#define APB2_TIMER_FREQ (APB2_FREQ*2U)  // APB2 is multiplied by 2 for the timer peripherals

#define BOOTLOADER_ADDRESS 0x1FFF0004U

// Around (1Mbps / 8 bits/byte / 12 bytes per message)
#define CAN_INTERRUPT_RATE 12000U

#define MAX_LED_FADE 8192U

#define NUM_INTERRUPTS 102U                // There are 102 external interrupt sources (see stm32f413.h)

#define TICK_TIMER_IRQ TIM1_BRK_TIM9_IRQn
#define TICK_TIMER TIM9

#define MICROSECOND_TIMER TIM2

#define INTERRUPT_TIMER_IRQ TIM6_DAC_IRQn
#define INTERRUPT_TIMER TIM6

#define IND_WDG IWDG

#define PROVISION_CHUNK_ADDRESS 0x1FFF79E0U
#define DEVICE_SERIAL_NUMBER_ADDRESS 0x1FFF79C0U

#include "board/can.h"
#include "board/comms_definitions.h"

#ifndef BOOTSTUB
  #include "board/main_definitions.h"
#else
  #include "board/bootstub_declarations.h"
#endif

#include "board/libc.h"
#include "board/critical.h"
#include "board/faults.h"
#include "board/utils.h"

#include "board/drivers/registers.h"
#include "board/drivers/interrupts.h"
#include "board/drivers/gpio.h"
#include "board/stm32f4/peripherals.h"
#include "board/stm32f4/interrupt_handlers.h"
#include "board/drivers/timers.h"
#include "board/stm32f4/board.h"
#include "board/stm32f4/clock.h"

#if !defined(BOOTSTUB)
  #include "board/drivers/uart.h"
  #include "board/stm32f4/lluart.h"
#endif

#ifdef BOOTSTUB
  #include "board/stm32f4/llflash.h"
#else
  #include "board/stm32f4/llbxcan.h"
#endif

#include "board/stm32f4/llusb.h"

// unused
void spi_init(void) {};
void sound_tick(void) {};
void can_tx_comms_resume_spi(void) {};

void early_gpio_float(void) {
  RCC->AHB1ENR = RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN | RCC_AHB1ENR_GPIOCEN;

  GPIOB->MODER = 0; GPIOC->MODER = 0;
  GPIOA->ODR = 0; GPIOB->ODR = 0; GPIOC->ODR = 0;
  GPIOA->PUPDR = 0; GPIOB->PUPDR = 0; GPIOC->PUPDR = 0;
}
