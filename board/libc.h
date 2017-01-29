#define min(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a < _b ? _a : _b; })

#define max(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a > _b ? _a : _b; })

#define __DIV(_PCLK_, _BAUD_)                        (((_PCLK_)*25)/(4*(_BAUD_)))
#define __DIVMANT(_PCLK_, _BAUD_)                    (__DIV((_PCLK_), (_BAUD_))/100)
#define __DIVFRAQ(_PCLK_, _BAUD_)                    (((__DIV((_PCLK_), (_BAUD_)) - (__DIVMANT((_PCLK_), (_BAUD_)) * 100)) * 16 + 50) / 100)
#define __USART_BRR(_PCLK_, _BAUD_)              ((__DIVMANT((_PCLK_), (_BAUD_)) << 4)|(__DIVFRAQ((_PCLK_), (_BAUD_)) & 0x0F))

#include "stm32f2xx_hal_gpio_ex.h"

// **** shitty libc ****

void clock_init() {
  #ifdef USE_INTERNAL_OSC
    // enable internal oscillator
    RCC->CR |= RCC_CR_HSION;
    while ((RCC->CR & RCC_CR_HSIRDY) == 0);
  #else
    // enable external oscillator
    RCC->CR |= RCC_CR_HSEON;
    while ((RCC->CR & RCC_CR_HSERDY) == 0);
  #endif

  // divide shit
  RCC->CFGR = RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE2_DIV2 | RCC_CFGR_PPRE1_DIV4;
  #ifdef USE_INTERNAL_OSC
    RCC->PLLCFGR = RCC_PLLCFGR_PLLQ_2 | RCC_PLLCFGR_PLLM_3 |
                   RCC_PLLCFGR_PLLN_6 | RCC_PLLCFGR_PLLN_5 | RCC_PLLCFGR_PLLSRC_HSI;
  #else
    RCC->PLLCFGR = RCC_PLLCFGR_PLLQ_2 | RCC_PLLCFGR_PLLM_3 |
                   RCC_PLLCFGR_PLLN_7 | RCC_PLLCFGR_PLLN_6 | RCC_PLLCFGR_PLLSRC_HSE;
  #endif

  // start PLL
  RCC->CR |= RCC_CR_PLLON;
  while ((RCC->CR & RCC_CR_PLLRDY) == 0);

  // Configure Flash prefetch, Instruction cache, Data cache and wait state
  // *** without this, it breaks ***
  FLASH->ACR = FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_LATENCY_5WS;

  // switch to PLL
  RCC->CFGR |= RCC_CFGR_SW_PLL;
  while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL);

  // *** running on PLL ***

  // enable GPIOB, UART2, CAN, USB clock
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN;
  RCC->AHB1ENR |= RCC_AHB1ENR_GPIOCEN;

  RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;

  RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
  RCC->APB1ENR |= RCC_APB1ENR_USART3EN;
  RCC->APB1ENR |= RCC_APB1ENR_CAN1EN;
  RCC->APB1ENR |= RCC_APB1ENR_CAN2EN;
  RCC->APB1ENR |= RCC_APB1ENR_DACEN;
  RCC->APB1ENR |= RCC_APB1ENR_TIM3EN;
  //RCC->APB1ENR |= RCC_APB1ENR_TIM4EN;
  RCC->AHB2ENR |= RCC_AHB2ENR_OTGFSEN;
  RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;
  RCC->APB2ENR |= RCC_APB2ENR_ADC1EN;
  RCC->APB2ENR |= RCC_APB2ENR_SPI1EN;

  // turn on alt USB
  RCC->AHB1ENR |= RCC_AHB1ENR_OTGHSEN;

  // fix interrupt vectors
}

// board specific
void gpio_init() {
  // analog mode
  GPIOC->MODER = GPIO_MODER_MODER3 | GPIO_MODER_MODER2 |
                 GPIO_MODER_MODER1 | GPIO_MODER_MODER0;

  // FAN on C9, aka TIM3_CH4
  #ifdef OLD_BOARD
    GPIOC->MODER |= GPIO_MODER_MODER9_1;
    GPIOC->AFR[1] = GPIO_AF2_TIM3 << ((9-8)*4);
  #else
    GPIOC->MODER |= GPIO_MODER_MODER8_1;
    GPIOC->AFR[1] = GPIO_AF2_TIM3 << ((8-8)*4);
  #endif
  // IGNITION on C13

  // set mode for LEDs and CAN
  GPIOB->MODER = GPIO_MODER_MODER10_0 | GPIO_MODER_MODER11_0 | GPIO_MODER_MODER12_0;
  // CAN 2
  GPIOB->MODER |= GPIO_MODER_MODER5_1 | GPIO_MODER_MODER6_1;
  // CAN 1
  GPIOB->MODER |= GPIO_MODER_MODER8_1 | GPIO_MODER_MODER9_1;
  // CAN enables
  GPIOB->MODER |= GPIO_MODER_MODER3_0 | GPIO_MODER_MODER4_0;

  // set mode for SERIAL and USB (DAC should be configured to in)
  GPIOA->MODER = GPIO_MODER_MODER2_1 | GPIO_MODER_MODER3_1;
  GPIOA->AFR[0] = GPIO_AF7_USART2 << (2*4) | GPIO_AF7_USART2 << (3*4);

  // GPIOC USART3
  GPIOC->MODER |= GPIO_MODER_MODER10_1 | GPIO_MODER_MODER11_1;
  GPIOC->AFR[1] |= GPIO_AF7_USART3 << ((10-8)*4) | GPIO_AF7_USART3 << ((11-8)*4);

  if (USBx == USB_OTG_FS) {
    GPIOA->MODER |= GPIO_MODER_MODER11_1 | GPIO_MODER_MODER12_1;
    GPIOA->OSPEEDR = GPIO_OSPEEDER_OSPEEDR11 | GPIO_OSPEEDER_OSPEEDR12;
    GPIOA->AFR[1] = GPIO_AF10_OTG_FS << ((11-8)*4) | GPIO_AF10_OTG_FS << ((12-8)*4);
  }

  GPIOA->PUPDR = GPIO_PUPDR_PUPDR2_0 | GPIO_PUPDR_PUPDR3_0;

  // setup SPI
  GPIOA->MODER |= GPIO_MODER_MODER4_1 | GPIO_MODER_MODER5_1 |
                  GPIO_MODER_MODER6_1 | GPIO_MODER_MODER7_1;
  GPIOA->AFR[0] |= GPIO_AF5_SPI1 << (4*4) | GPIO_AF5_SPI1 << (5*4) |
                   GPIO_AF5_SPI1 << (6*4) | GPIO_AF5_SPI1 << (7*4);

  // set mode for CAN / USB_HS pins
  GPIOB->AFR[0] = GPIO_AF9_CAN1 << (5*4) | GPIO_AF9_CAN1 << (6*4);
  GPIOB->AFR[1] = GPIO_AF9_CAN1 << ((8-8)*4) | GPIO_AF9_CAN1 << ((9-8)*4);

  if (USBx == USB_OTG_HS) {
    GPIOB->AFR[1] |= GPIO_AF12_OTG_HS_FS << ((15-8)*4) | GPIO_AF12_OTG_HS_FS << ((14-8)*4);
    GPIOB->MODER |= GPIO_MODER_MODER14_1 | GPIO_MODER_MODER15_1;
  }

  GPIOB->OSPEEDR = GPIO_OSPEEDER_OSPEEDR14 | GPIO_OSPEEDER_OSPEEDR15;

  // enable OTG out tied to ground
  GPIOA->ODR = 0;
  GPIOA->MODER |= GPIO_MODER_MODER1_0;

  // enable USB power tied to +
  GPIOA->ODR |= 1;
  GPIOA->MODER |= GPIO_MODER_MODER0_0;
}

void uart_init() {
  // enable uart and tx+rx mode
  USART->CR1 = USART_CR1_UE;
  USART->BRR = __USART_BRR(24000000, 115200);
  USART->CR1 |= USART_CR1_TE | USART_CR1_RE;
  USART->CR2 = USART_CR2_STOP_0 | USART_CR2_STOP_1;
  // ** UART is ready to work **

  // enable interrupts
  USART->CR1 |= USART_CR1_RXNEIE;
}

void delay(int a) {
  volatile int i;
  for (i=0;i<a;i++);
}

void putch(const char a) {
  while (!(USART->SR & USART_SR_TXE));
  USART->DR = a;
}

int puts(const char *a) {
  for (;*a;a++) {
    if (*a == '\n') putch('\r');
    putch(*a);
  }
  return 0;
}

void puth(unsigned int i) {
  int pos;
  char c[] = "0123456789abcdef";
  for (pos = 28; pos != -4; pos -= 4) {
    putch(c[(i >> pos) & 0xF]);
  }
}

void puth2(unsigned int i) {
  int pos;
  char c[] = "0123456789abcdef";
  for (pos = 4; pos != -4; pos -= 4) {
    putch(c[(i >> pos) & 0xF]);
  }
}

void hexdump(void *a, int l) {
  int i;
  for (i=0;i<l;i++) {
    if (i != 0 && (i&0xf) == 0) puts("\n");
    puth2(((unsigned char*)a)[i]);
    puts(" ");
  }
  puts("\n");
}

void *memset(void *str, int c, unsigned int n) {
  int i;
  for (i = 0; i < n; i++) {
    *((uint8_t*)str) = c;
    ++str;
  }
  return str;
}

void *memcpy(void *dest, const void *src, unsigned int n) {
  int i;
  // TODO: make not slow
  for (i = 0; i < n; i++) {
    ((uint8_t*)dest)[i] = *(uint8_t*)src;
    ++src;
  }
  return dest;
}

void set_led(int led_num, int state) {
  if (state) {
    // turn on
    GPIOB->ODR &= ~(1 << (10 + led_num));
  } else {
    // turn off
    GPIOB->ODR |= (1 << (10 + led_num));
  }
}

