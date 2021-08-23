#include "stm32h7/inc/stm32h7xx.h"
#include "stm32h7/inc/stm32h7xx_hal_gpio_ex.h"
#define MCU_IDCODE 0x483U

// from the linker script
#define APP_START_ADDRESS 0x8020000U

#define CORE_FREQ 240U // in Mhz
//APB1 - 120Mhz, APB2 - 120Mhz
#define APB1_FREQ CORE_FREQ/2U 
#define APB2_FREQ CORE_FREQ/2U

#define BOOTLOADER_ADDRESS 0x1FF09804U

// Around (1Mbps / 8 bits/byte / 12 bytes per message)
#define CAN_INTERRUPT_RATE 12000U // FIXME: should raise to 16000 ?

#define MAX_LED_FADE 10240U

// Threshold voltage (mV) for either of the SBUs to be below before deciding harness is connected
#define HARNESS_CONNECTED_THRESHOLD 40000U

// There are 163 external interrupt sources (see stm32f735xx.h)
#define NUM_INTERRUPTS 163U

#define TICK_TIMER_IRQ TIM8_BRK_TIM12_IRQn
#define TICK_TIMER TIM12

#define MICROSECOND_TIMER TIM2

#define INTERRUPT_TIMER_IRQ TIM6_DAC_IRQn
#define INTERRUPT_TIMER TIM6

#define PROVISION_CHUNK_ADDRESS 0x080FFFE0U
#define DEVICE_SERIAL_NUMBER_ADDRESS 0x080FFFC0U

#ifndef BOOTSTUB
  #include "main_declarations.h"
#else
  #include "bootstub_declarations.h"
#endif

#include "libc.h"
#include "critical.h"
#include "faults.h"

#include "drivers/registers.h"
#include "drivers/interrupts.h"
#include "drivers/gpio.h"
#include "stm32h7/peripherals.h"
#include "stm32h7/interrupt_handlers.h"
#include "drivers/timers.h"
#include "stm32h7/lladc.h"
#include "stm32h7/board.h"
#include "stm32h7/clock.h"

#if !defined (BOOTSTUB) && defined(PANDA)
  #include "drivers/uart.h"
  #include "stm32h7/lluart.h"
#endif

#ifdef BOOTSTUB
  #include "stm32h7/llflash.h"
#else
  #include "stm32h7/llfdcan.h"
#endif

#include "stm32h7/llusb.h"

void early_gpio_float(void) {
  RCC->AHB4ENR = RCC_AHB4ENR_GPIOAEN | RCC_AHB4ENR_GPIOBEN | RCC_AHB4ENR_GPIOCEN | RCC_AHB4ENR_GPIODEN | RCC_AHB4ENR_GPIOEEN | RCC_AHB4ENR_GPIOFEN | RCC_AHB4ENR_GPIOGEN | RCC_AHB4ENR_GPIOHEN;
  GPIOA->MODER = 0; GPIOB->MODER = 0; GPIOC->MODER = 0; GPIOD->MODER = 0; GPIOE->MODER = 0; GPIOF->MODER = 0; GPIOG->MODER = 0; GPIOH->MODER = 0;
  GPIOA->ODR = 0; GPIOB->ODR = 0; GPIOC->ODR = 0; GPIOD->ODR = 0; GPIOE->ODR = 0; GPIOF->ODR = 0; GPIOG->ODR = 0; GPIOH->ODR = 0;
  GPIOA->PUPDR = 0; GPIOB->PUPDR = 0; GPIOC->PUPDR = 0; GPIOD->PUPDR = 0; GPIOE->PUPDR = 0; GPIOF->PUPDR = 0; GPIOG->PUPDR = 0; GPIOH->PUPDR = 0;
}
