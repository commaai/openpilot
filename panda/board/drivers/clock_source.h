#define CLOCK_SOURCE_PERIOD_MS           50U
#define CLOCK_SOURCE_PULSE_LEN_MS        2U

void clock_source_set_period(uint8_t period) {
  register_set(&(TIM1->ARR), ((period*10U) - 1U), 0xFFFFU);
}

void clock_source_init(void) {
  // Setup timer
  register_set(&(TIM1->PSC), ((APB2_TIMER_FREQ*100U)-1U), 0xFFFFU);           // Tick on 0.1 ms
  register_set(&(TIM1->ARR), ((CLOCK_SOURCE_PERIOD_MS*10U) - 1U), 0xFFFFU);   // Period
  register_set(&(TIM1->CCMR1), 0U, 0xFFFFU);                                  // No output on compare
  register_set(&(TIM1->CCER), TIM_CCER_CC1E, 0xFFFFU);                        // Enable compare 1
  register_set(&(TIM1->CCR1), (CLOCK_SOURCE_PULSE_LEN_MS*10U), 0xFFFFU);      // Compare 1 value
  register_set(&(TIM1->CCR2), (CLOCK_SOURCE_PULSE_LEN_MS*10U), 0xFFFFU);      // Compare 1 value
  register_set(&(TIM1->CCR3), (CLOCK_SOURCE_PULSE_LEN_MS*10U), 0xFFFFU);      // Compare 1 value
  register_set_bits(&(TIM1->DIER), TIM_DIER_UIE | TIM_DIER_CC1IE);            // Enable interrupts
  register_set(&(TIM1->CR1), TIM_CR1_CEN, 0x3FU);                             // Enable timer

  // No interrupts
  NVIC_DisableIRQ(TIM1_UP_TIM10_IRQn);
  NVIC_DisableIRQ(TIM1_CC_IRQn);

  // Set GPIO as timer channels
  set_gpio_alternate(GPIOB, 14, GPIO_AF1_TIM1);
  set_gpio_alternate(GPIOB, 15, GPIO_AF1_TIM1);

  // Set PWM mode
  register_set(&(TIM1->CCMR1), (0b110 << TIM_CCMR1_OC2M_Pos), 0xFFFFU);
  register_set(&(TIM1->CCMR2), (0b110 << TIM_CCMR2_OC3M_Pos), 0xFFFFU);

  // Enable output
  register_set(&(TIM1->BDTR), TIM_BDTR_MOE, 0xFFFFU);

  // Enable complementary compares
  register_set_bits(&(TIM1->CCER), TIM_CCER_CC2NE | TIM_CCER_CC3NE);
}
