
#define BEEPER_COUNTER_OVERFLOW 25000U // 4kHz

void beeper_enable(bool enabled) {
  if (enabled) {
    register_set_bits(&(TIM4->CCER), TIM_CCER_CC3E);
  } else {
    register_clear_bits(&(TIM4->CCER), TIM_CCER_CC3E);
  }
}

void beeper_init(void) {
  // Enable timer and auto-reload
  register_set(&(TIM4->CR1), TIM_CR1_CEN | TIM_CR1_ARPE, 0x3FU);

  // Set channel as PWM mode 1 and disable output
  register_set_bits(&(TIM4->CCMR2), (TIM_CCMR2_OC3M_2 | TIM_CCMR2_OC3M_1 | TIM_CCMR2_OC3PE));
  beeper_enable(false);

  // Set max counter value and compare to get 50% duty
  register_set(&(TIM4->CCR3), BEEPER_COUNTER_OVERFLOW / 2U, 0xFFFFU);
  register_set(&(TIM4->ARR), BEEPER_COUNTER_OVERFLOW, 0xFFFFU);

  // Update registers and clear counter
  TIM4->EGR |= TIM_EGR_UG;
}
