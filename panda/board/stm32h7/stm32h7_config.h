#include "stm32h7xx.h"
#include "stm32h7xx_hal_gpio_ex.h"
#define MCU_IDCODE 0x483U

#define CORE_FREQ 240U // in Mhz
//APB1 - 120Mhz, APB2 - 120Mhz
#define APB1_FREQ (CORE_FREQ/4U)
#define APB1_TIMER_FREQ (APB1_FREQ*2U) // APB1 is multiplied by 2 for the timer peripherals
#define APB2_FREQ (CORE_FREQ/4U)
#define APB2_TIMER_FREQ (APB2_FREQ*2U) // APB2 is multiplied by 2 for the timer peripherals

#define BOOTLOADER_ADDRESS 0x1FF09804U

/*
An IRQ is received on message RX/TX (or RX errors), with
separate IRQs for RX and TX.

0-byte CAN FD frame as the worst case:
- 17 slow bits = SOF + 11 ID + R1 + IDE + EDL + R0 + BRS
- 23 fast bits = ESI + 4 DLC + 0 DATA + 17 CRC + CRC delimeter
- 12 slow bits = ACK + DEL + 7 EOF + 3 IFS
- all currently supported cars are 0.5 Mbps / 2 Mbps

1 / ((29 bits / 0.5Mbps) + (23 bits / 2Mbps)) = 14388Hz
*/
#define CAN_INTERRUPT_RATE 16000U

#define MAX_LED_FADE 10240U

// There are 163 external interrupt sources (see stm32f735xx.h)
#define NUM_INTERRUPTS 163U

#define TICK_TIMER_IRQ TIM8_BRK_TIM12_IRQn
#define TICK_TIMER TIM12

#define MICROSECOND_TIMER TIM2

#define INTERRUPT_TIMER_IRQ TIM6_DAC_IRQn
#define INTERRUPT_TIMER TIM6

#define IND_WDG IWDG1

#define PROVISION_CHUNK_ADDRESS 0x080FFFE0U
#define DEVICE_SERIAL_NUMBER_ADDRESS 0x080FFFC0U

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
#include "board/stm32h7/peripherals.h"
#include "board/stm32h7/interrupt_handlers.h"
#include "board/drivers/timers.h"

#if !defined(BOOTSTUB)
  #include "board/drivers/uart.h"
  #include "board/stm32h7/lluart.h"
#endif

#ifdef PANDA_JUNGLE
#include "board/jungle/stm32h7/board.h"
#else
#include "board/stm32h7/board.h"
#endif
#include "board/stm32h7/clock.h"

#ifdef BOOTSTUB
  #include "board/stm32h7/llflash.h"
#else
  #include "board/stm32h7/llfdcan.h"
#endif

#include "board/stm32h7/llusb.h"

#include "board/drivers/spi.h"
#include "board/stm32h7/llspi.h"

void early_gpio_float(void) {
  RCC->AHB4ENR = RCC_AHB4ENR_GPIOAEN | RCC_AHB4ENR_GPIOBEN | RCC_AHB4ENR_GPIOCEN | RCC_AHB4ENR_GPIODEN | RCC_AHB4ENR_GPIOEEN | RCC_AHB4ENR_GPIOFEN | RCC_AHB4ENR_GPIOGEN | RCC_AHB4ENR_GPIOHEN;
  GPIOA->MODER = 0xAB000000U; GPIOB->MODER = 0; GPIOC->MODER = 0; GPIOD->MODER = 0; GPIOE->MODER = 0; GPIOF->MODER = 0; GPIOG->MODER = 0; GPIOH->MODER = 0;
  GPIOA->ODR = 0; GPIOB->ODR = 0; GPIOC->ODR = 0; GPIOD->ODR = 0; GPIOE->ODR = 0; GPIOF->ODR = 0; GPIOG->ODR = 0; GPIOH->ODR = 0;
  GPIOA->PUPDR = 0; GPIOB->PUPDR = 0; GPIOC->PUPDR = 0; GPIOD->PUPDR = 0; GPIOE->PUPDR = 0; GPIOF->PUPDR = 0; GPIOG->PUPDR = 0; GPIOH->PUPDR = 0;
}
