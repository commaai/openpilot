#ifdef STM32F4
  #include "stm32fx/inc/stm32f4xx.h"
  #include "stm32fx/inc/stm32f4xx_hal_gpio_ex.h"
  #define MCU_IDCODE 0x463U
#else
  #include "stm32fx/inc/stm32f2xx.h"
  #include "stm32fx/inc/stm32f2xx_hal_gpio_ex.h"
  #define MCU_IDCODE 0x411U
#endif
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

// Threshold voltage (mV) for either of the SBUs to be below before deciding harness is connected
#define HARNESS_CONNECTED_THRESHOLD 2500U

#define NUM_INTERRUPTS 102U                // There are 102 external interrupt sources (see stm32f413.h)

#define TICK_TIMER_IRQ TIM1_BRK_TIM9_IRQn
#define TICK_TIMER TIM9

#define MICROSECOND_TIMER TIM2

#define INTERRUPT_TIMER_IRQ TIM6_DAC_IRQn
#define INTERRUPT_TIMER TIM6

#define PROVISION_CHUNK_ADDRESS 0x1FFF79E0U
#define DEVICE_SERIAL_NUMBER_ADDRESS 0x1FFF79C0U

#include "can_definitions.h"
#include "comms_definitions.h"

#ifndef BOOTSTUB
  #ifdef PANDA
    #include "main_declarations.h"
  #else
    #include "pedal/main_declarations.h"
  #endif
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
#include "stm32fx/peripherals.h"
#include "stm32fx/interrupt_handlers.h"
#include "drivers/timers.h"
#include "stm32fx/lladc.h"
#include "stm32fx/board.h"
#include "stm32fx/clock.h"

#if defined(PANDA) || defined(BOOTSTUB)
  #include "drivers/spi.h"
  #include "stm32fx/llspi.h"
#endif

#if !defined(BOOTSTUB) && (defined(PANDA) || defined(PEDAL_USB))
  #include "drivers/uart.h"
  #include "stm32fx/lluart.h"
#endif

#if !defined(PEDAL_USB) && !defined(PEDAL) && !defined(BOOTSTUB)
  #include "stm32fx/llexti.h"
#endif

#ifdef BOOTSTUB
  #include "stm32fx/llflash.h"
#else
  #include "stm32fx/llbxcan.h"
#endif

#if defined(PANDA) || defined(BOOTSTUB) || defined(PEDAL_USB)
  #include "stm32fx/llusb.h"
#endif

#ifdef PEDAL
  #include "stm32fx/lldac.h"
#endif

void early_gpio_float(void) {
  RCC->AHB1ENR = RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN | RCC_AHB1ENR_GPIOCEN;

  GPIOA->MODER = 0; GPIOB->MODER = 0; GPIOC->MODER = 0;
  GPIOA->ODR = 0; GPIOB->ODR = 0; GPIOC->ODR = 0;
  GPIOA->PUPDR = 0; GPIOB->PUPDR = 0; GPIOC->PUPDR = 0;
}
