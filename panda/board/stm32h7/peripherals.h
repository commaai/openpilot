#pragma once

#ifdef BOOTSTUB
void gpio_usb_init(void) {
#else
static void gpio_usb_init(void) {
#endif
  // A11,A12: USB
  set_gpio_alternate(GPIOA, 11, GPIO_AF10_OTG1_FS);
  set_gpio_alternate(GPIOA, 12, GPIO_AF10_OTG1_FS);
  GPIOA->OSPEEDR = GPIO_OSPEEDR_OSPEED11 | GPIO_OSPEEDR_OSPEED12;
}

void gpio_spi_init(void) {
  set_gpio_alternate(GPIOE, 11, GPIO_AF5_SPI4);
  set_gpio_alternate(GPIOE, 12, GPIO_AF5_SPI4);
  set_gpio_alternate(GPIOE, 13, GPIO_AF5_SPI4);
  set_gpio_alternate(GPIOE, 14, GPIO_AF5_SPI4);
  register_set_bits(&(GPIOE->OSPEEDR), GPIO_OSPEEDR_OSPEED11 | GPIO_OSPEEDR_OSPEED12 | GPIO_OSPEEDR_OSPEED13 | GPIO_OSPEEDR_OSPEED14);
}

// Common GPIO initialization
void common_init_gpio(void) {
  //F11: VOLT_S
  set_gpio_pullup(GPIOF, 11, PULL_NONE);
  set_gpio_mode(GPIOF, 11, MODE_ANALOG);

  gpio_usb_init();

  // B8,B9: FDCAN1
  set_gpio_pullup(GPIOB, 8, PULL_NONE);
  set_gpio_alternate(GPIOB, 8, GPIO_AF9_FDCAN1);

  set_gpio_pullup(GPIOB, 9, PULL_NONE);
  set_gpio_alternate(GPIOB, 9, GPIO_AF9_FDCAN1);

  // B5,B6 (mplex to B12,B13): FDCAN2
  set_gpio_pullup(GPIOB, 12, PULL_NONE);
  set_gpio_pullup(GPIOB, 13, PULL_NONE);

  set_gpio_pullup(GPIOB, 5, PULL_NONE);
  set_gpio_alternate(GPIOB, 5, GPIO_AF9_FDCAN2);

  set_gpio_pullup(GPIOB, 6, PULL_NONE);
  set_gpio_alternate(GPIOB, 6, GPIO_AF9_FDCAN2);

  // G9,G10: FDCAN3
  set_gpio_pullup(GPIOG, 9, PULL_NONE);
  set_gpio_alternate(GPIOG, 9, GPIO_AF2_FDCAN3);

  set_gpio_pullup(GPIOG, 10, PULL_NONE);
  set_gpio_alternate(GPIOG, 10, GPIO_AF2_FDCAN3);
}

#ifdef BOOTSTUB
void flasher_peripherals_init(void) {
  RCC->AHB1ENR |= RCC_AHB1ENR_USB1OTGHSEN;

  // SPI + DMA
  RCC->APB2ENR |= RCC_APB2ENR_SPI4EN;
  RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;

  // LED PWM
  RCC->APB1LENR |= RCC_APB1LENR_TIM3EN;
}
#endif

// FFI_SM_0, FFI_SM_1: hold everything without a clock enable in reset, RST and EN bits share positions
// all RST bits per stm32h725xx.h, except CPURST and the H73x only CRYP, HASH, OTFDEC
static void hold_unused_in_reset(void) {
  static const struct { volatile uint32_t *enr; volatile uint32_t *rstr; uint32_t rst_bits; } regs[] = {
    {&(RCC->AHB1ENR), &(RCC->AHB1RSTR), 0x02008023U},
    {&(RCC->AHB2ENR), &(RCC->AHB2RSTR), 0x00030241U},
    {&(RCC->AHB3ENR), &(RCC->AHB3RSTR), 0x00295011U},
    {&(RCC->AHB4ENR), &(RCC->AHB4RSTR), 0x032806FFU},
    {&(RCC->APB1LENR), &(RCC->APB1LRSTR), 0xEAFFC3FFU},
    {&(RCC->APB1HENR), &(RCC->APB1HRSTR), 0x03000136U},
    {&(RCC->APB2ENR), &(RCC->APB2RSTR), 0x405730F3U},
    {&(RCC->APB3ENR), &(RCC->APB3RSTR), 0x00000008U},
    {&(RCC->APB4ENR), &(RCC->APB4RSTR), 0x0420DEAAU},
  };
  for (uint8_t i = 0U; i < (sizeof(regs) / sizeof(regs[0])); i++) {
    uint32_t en = *regs[i].enr;
    register_set(regs[i].rstr, regs[i].rst_bits & ~en, 0xFFFFFFFFU);
    register_set(regs[i].enr, en, 0xFFFFFFFFU);
  }
}

// Peripheral initialization
void peripherals_init(void) {
  // enable GPIO(A,B,C,D,E,F,G,H)
  RCC->AHB4ENR |= RCC_AHB4ENR_GPIOAEN;
  RCC->AHB4ENR |= RCC_AHB4ENR_GPIOBEN;
  RCC->AHB4ENR |= RCC_AHB4ENR_GPIOCEN;
  RCC->AHB4ENR |= RCC_AHB4ENR_GPIODEN;
  RCC->AHB4ENR |= RCC_AHB4ENR_GPIOEEN;
  RCC->AHB4ENR |= RCC_AHB4ENR_GPIOFEN;
  RCC->AHB4ENR |= RCC_AHB4ENR_GPIOGEN;

  // Enable CPU access to SRAMs for DMA
  RCC->AHB2ENR |= RCC_AHB2ENR_SRAM1EN | RCC_AHB2ENR_SRAM2EN;

  // Supplemental
  RCC->AHB1ENR |= RCC_AHB1ENR_DMA1EN;  // DAC DMA
  RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;  // SPI DMA
  RCC->APB4ENR |= RCC_APB4ENR_SYSCFGEN;
  RCC->AHB4ENR |= RCC_AHB4ENR_BDMAEN; // Audio DMA
  RCC->APB4ENR |= RCC_APB4ENR_DTSEN; // Digital Temperature Sensor

  // Connectivity
  RCC->APB2ENR |= RCC_APB2ENR_SPI4EN;  // SPI
  RCC->APB1LENR |= RCC_APB1LENR_I2C5EN;  // codec I2C
  RCC->AHB1ENR |= RCC_AHB1ENR_USB1OTGHSEN; // USB
  RCC->AHB1LPENR |= RCC_AHB1LPENR_USB1OTGHSLPEN; // USB LP needed for CSleep state(__WFI())
  RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_USB1OTGHSULPILPEN); // disable USB ULPI
  RCC->APB1HENR |= RCC_APB1HENR_FDCANEN; // FDCAN core enable

  // Analog
  RCC->AHB1ENR |= RCC_AHB1ENR_ADC12EN; // Enable ADC12 clocks
  RCC->AHB4ENR |= RCC_AHB4ENR_ADC3EN; // Enable ADC3 clocks
  RCC->APB1LENR |= RCC_APB1LENR_DAC12EN; // DAC

  // Audio
  RCC->APB2ENR |= RCC_APB2ENR_DFSDM1EN; // D/S demodulator for mic
  RCC->APB4ENR |= RCC_APB4ENR_SAI4EN;  // SAI4

  // Timers
  RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;  // clock source timer
  RCC->APB1LENR |= RCC_APB1LENR_TIM2EN;  // main counter
  RCC->APB1LENR |= RCC_APB1LENR_TIM3EN;  // fan + led pwm
  RCC->APB1LENR |= RCC_APB1LENR_TIM6EN;  // interrupt timer
  RCC->APB1LENR |= RCC_APB1LENR_TIM7EN;  // DMA trigger timer
  RCC->APB2ENR |= RCC_APB2ENR_TIM8EN;  // tick timer
  RCC->APB2ENR |= RCC_APB2ENR_TIM16EN;  // clock check
  RCC->APB1LENR |= RCC_APB1LENR_TIM12EN;  // slow loop
  RCC->APB1LENR |= RCC_APB1LENR_TIM5EN; // sound trigger timer

  hold_unused_in_reset();
}

void enable_interrupt_timer(void) {
  register_set_bits(&(RCC->APB1LENR), RCC_APB1LENR_TIM6EN); // Enable interrupt timer peripheral
}

// FLASH_SM_3: nothing unlocks the option bytes, and they boot from 0x08000000 when BOOT0 is low
void check_option_bytes(void) {
  uint32_t optlock = FLASH->OPTCR & FLASH_OPTCR_OPTLOCK;
  uint32_t boot = FLASH->BOOT_CUR & FLASH_BOOT_ADD0;
  if ((optlock == 0U) || (boot != 0x0800U)) {
    fault_occurred_value(FAULT_REGISTER_DIVERGENT, (boot << 16) | optlock);
  }
}
