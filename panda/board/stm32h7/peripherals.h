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

void gpio_usart2_init(void) {
  // A2,A3: USART 2 for debugging
  set_gpio_alternate(GPIOA, 2, GPIO_AF7_USART2);
  set_gpio_alternate(GPIOA, 3, GPIO_AF7_USART2);
}

void gpio_uart7_init(void) {
  // E7,E8: UART 7 for debugging
  set_gpio_alternate(GPIOE, 7, GPIO_AF7_UART7);
  set_gpio_alternate(GPIOE, 8, GPIO_AF7_UART7);
}

// Common GPIO initialization
void common_init_gpio(void) {
  /// E2,E3,E4: RGB LED
  set_gpio_pullup(GPIOE, 2, PULL_NONE);
  set_gpio_mode(GPIOE, 2, MODE_OUTPUT);
  set_gpio_output_type(GPIOE, 2, OUTPUT_TYPE_OPEN_DRAIN);

  set_gpio_pullup(GPIOE, 3, PULL_NONE);
  set_gpio_mode(GPIOE, 3, MODE_OUTPUT);
  set_gpio_output_type(GPIOE, 3, OUTPUT_TYPE_OPEN_DRAIN);

  set_gpio_pullup(GPIOE, 4, PULL_NONE);
  set_gpio_mode(GPIOE, 4, MODE_OUTPUT);
  set_gpio_output_type(GPIOE, 4, OUTPUT_TYPE_OPEN_DRAIN);

  //C4,A1: OBD_SBU1, OBD_SBU2
  set_gpio_pullup(GPIOC, 4, PULL_NONE);
  set_gpio_mode(GPIOC, 4, MODE_ANALOG);

  set_gpio_pullup(GPIOA, 1, PULL_NONE);
  set_gpio_mode(GPIOA, 1, MODE_ANALOG);

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

void flasher_peripherals_init(void) {
  RCC->AHB1ENR |= RCC_AHB1ENR_USB1OTGHSEN;

  // SPI + DMA
  RCC->APB2ENR |= RCC_APB2ENR_SPI4EN;
  RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;
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

  // Connectivity
  RCC->APB2ENR |= RCC_APB2ENR_SPI4EN;  // SPI
  RCC->APB1LENR |= RCC_APB1LENR_I2C5EN;  // codec I2C
  RCC->AHB1ENR |= RCC_AHB1ENR_USB1OTGHSEN; // USB
  RCC->AHB1LPENR |= RCC_AHB1LPENR_USB1OTGHSLPEN; // USB LP needed for CSleep state(__WFI())
  RCC->AHB1LPENR &= ~(RCC_AHB1LPENR_USB1OTGHSULPILPEN); // disable USB ULPI
  RCC->APB1LENR |= RCC_APB1LENR_UART7EN;  // SOM uart
  RCC->APB1HENR |= RCC_APB1HENR_FDCANEN; // FDCAN core enable

  // Analog
  RCC->AHB1ENR |= RCC_AHB1ENR_ADC12EN; // Enable ADC12 clocks
  RCC->APB1LENR |= RCC_APB1LENR_DAC12EN; // DAC

  // Audio
  RCC->APB4ENR |= RCC_APB4ENR_SAI4EN;  // SAI4

  // Timers
  RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;  // clock source timer
  RCC->APB1LENR |= RCC_APB1LENR_TIM2EN;  // main counter
  RCC->APB1LENR |= RCC_APB1LENR_TIM3EN;  // fan pwm
  RCC->APB1LENR |= RCC_APB1LENR_TIM4EN;  // beeper source
  RCC->APB1LENR |= RCC_APB1LENR_TIM6EN;  // interrupt timer
  RCC->APB1LENR |= RCC_APB1LENR_TIM7EN;  // DMA trigger timer
  RCC->APB2ENR |= RCC_APB2ENR_TIM8EN;  // tick timer
  RCC->APB1LENR |= RCC_APB1LENR_TIM12EN;  // slow loop
  RCC->APB1LENR |= RCC_APB1LENR_TIM5EN; // sound trigger timer

#ifdef PANDA_JUNGLE
  RCC->AHB3ENR |= RCC_AHB3ENR_SDMMC1EN; // SDMMC
  RCC->AHB4ENR |= RCC_AHB4ENR_ADC3EN; // Enable ADC3 clocks
#endif
}

void enable_interrupt_timer(void) {
  register_set_bits(&(RCC->APB1LENR), RCC_APB1LENR_TIM6EN); // Enable interrupt timer peripheral
}
