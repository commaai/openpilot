
#define CLOCK_SOURCE_MODE_DISABLED       0U
#define CLOCK_SOURCE_MODE_FREE_RUNNING   1U

#define CLOCK_SOURCE_PERIOD_MS           50U
#define CLOCK_SOURCE_PULSE_LEN_MS        2U

uint8_t clock_source_mode = CLOCK_SOURCE_MODE_DISABLED;

void TIM1_UP_TIM10_IRQ_Handler(void) {
  if((TIM1->SR & TIM_SR_UIF) != 0) {
    if(clock_source_mode != CLOCK_SOURCE_MODE_DISABLED) {
      // Start clock pulse
      set_gpio_output(GPIOB, 14, true);
      set_gpio_output(GPIOB, 15, true);
      set_gpio_output(GPIOC, 5, true);
    }

    // Reset interrupt
    TIM1->SR &= ~(TIM_SR_UIF);
  }
}

void TIM1_CC_IRQ_Handler(void) {
  if((TIM1->SR & TIM_SR_CC1IF) != 0) {
    if(clock_source_mode != CLOCK_SOURCE_MODE_DISABLED) {
      // End clock pulse
      set_gpio_output(GPIOB, 14, false);
      set_gpio_output(GPIOB, 15, false);
      set_gpio_output(GPIOC, 5, false);
    }

    // Reset interrupt
    TIM1->SR &= ~(TIM_SR_CC1IF);
  }
}

void clock_source_init(uint8_t mode){
  // Setup timer
  REGISTER_INTERRUPT(TIM1_UP_TIM10_IRQn, TIM1_UP_TIM10_IRQ_Handler, (1200U / CLOCK_SOURCE_PERIOD_MS) , FAULT_INTERRUPT_RATE_TIM1)
  REGISTER_INTERRUPT(TIM1_CC_IRQn, TIM1_CC_IRQ_Handler, (1200U / CLOCK_SOURCE_PERIOD_MS) , FAULT_INTERRUPT_RATE_TIM1)
  register_set(&(TIM1->PSC), ((APB2_FREQ*100U)-1U), 0xFFFFU);                   // Tick on 0.1 ms
  register_set(&(TIM1->ARR), ((CLOCK_SOURCE_PERIOD_MS*10U) - 1U), 0xFFFFU);   // Period
  register_set(&(TIM1->CCMR1), 0U, 0xFFFFU);                                  // No output on compare
  register_set_bits(&(TIM1->CCER), TIM_CCER_CC1E);                            // Enable compare 1
  register_set(&(TIM1->CCR1), (CLOCK_SOURCE_PULSE_LEN_MS*10U), 0xFFFFU);      // Compare 1 value
  register_set_bits(&(TIM1->DIER), TIM_DIER_UIE | TIM_DIER_CC1IE);            // Enable interrupts
  register_set(&(TIM1->CR1), TIM_CR1_CEN, 0x3FU);                             // Enable timer

  // Set mode
  switch(mode) {
    case CLOCK_SOURCE_MODE_DISABLED:
      // No clock signal
      NVIC_DisableIRQ(TIM1_UP_TIM10_IRQn);
      NVIC_DisableIRQ(TIM1_CC_IRQn);

      // Disable pulse if we were in the middle of it
      set_gpio_output(GPIOB, 14, false);
      set_gpio_output(GPIOB, 15, false);

      clock_source_mode = CLOCK_SOURCE_MODE_DISABLED;
      break;
    case CLOCK_SOURCE_MODE_FREE_RUNNING:
      // Clock signal is based on internal timer
      NVIC_EnableIRQ(TIM1_UP_TIM10_IRQn);
      NVIC_EnableIRQ(TIM1_CC_IRQn);

      clock_source_mode = CLOCK_SOURCE_MODE_FREE_RUNNING;
      break;
    default:
      puts("Unknown clock source mode: "); puth(mode); puts("\n");
      break;
  }
}
