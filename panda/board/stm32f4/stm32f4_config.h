#include "stm32f4/inc/stm32f4xx.h"
#include "stm32f4/inc/stm32f4xx_hal_gpio_ex.h"
#define MCU_IDCODE 0x463U

// from the linker script
#define APP_START_ADDRESS 0x8004000U

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

#include "can_definitions.h"
#include "comms_definitions.h"

#ifndef BOOTSTUB
  #include "main_declarations.h"
#else
  #include "bootstub_declarations.h"
#endif

#include "libc.h"
#include "critical.h"
#include "faults.h"
#include "utils.h"

#include "drivers/registers.h"
#include "drivers/interrupts.h"
#include "drivers/gpio.h"
#include "stm32f4/peripherals.h"
#include "stm32f4/interrupt_handlers.h"
#include "drivers/timers.h"
#include "stm32f4/board.h"
#include "stm32f4/clock.h"
#include "drivers/watchdog.h"

#include "drivers/spi.h"
#include "stm32f4/llspi.h"

#if !defined(BOOTSTUB)
  #include "drivers/uart.h"
  #include "stm32f4/lluart.h"
#endif

#ifdef BOOTSTUB
  #include "stm32f4/llflash.h"
#else
  #include "stm32f4/llbxcan.h"
#endif

#include "stm32f4/llusb.h"

void early_gpio_float(void) {
  RCC->AHB1ENR = RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN | RCC_AHB1ENR_GPIOCEN;

  GPIOA->MODER = 0; GPIOB->MODER = 0; GPIOC->MODER = 0;
  GPIOA->ODR = 0; GPIOB->ODR = 0; GPIOC->ODR = 0;
  GPIOA->PUPDR = 0; GPIOB->PUPDR = 0; GPIOC->PUPDR = 0;
}
