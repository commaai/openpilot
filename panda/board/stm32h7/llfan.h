// TACH interrupt handler
void EXTI2_IRQ_Handler(void) {
  volatile unsigned int pr = EXTI->PR1 & (1U << 2);
  if ((pr & (1U << 2)) != 0U) {
    fan_state.tach_counter++;
  }
  EXTI->PR1 = (1U << 2);
}

void llfan_init(void) {
  fan_reset_cooldown();

  // 5000RPM * 4 tach edges / 60 seconds
  REGISTER_INTERRUPT(EXTI2_IRQn, EXTI2_IRQ_Handler, 700U, FAULT_INTERRUPT_RATE_TACH)

  // Init PWM speed control
  pwm_init(TIM3, 3);

  // Init TACH interrupt
  register_set(&(SYSCFG->EXTICR[0]), SYSCFG_EXTICR1_EXTI2_PD, 0xF00U);
  register_set_bits(&(EXTI->IMR1), (1U << 2));
  register_set_bits(&(EXTI->RTSR1), (1U << 2));
  register_set_bits(&(EXTI->FTSR1), (1U << 2));
  NVIC_EnableIRQ(EXTI2_IRQn);
}
