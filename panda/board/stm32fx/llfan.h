// TACH interrupt handler
void EXTI2_IRQ_Handler(void) {
  volatile unsigned int pr = EXTI->PR & (1U << 2);
  if ((pr & (1U << 2)) != 0U) {
      fan_state.tach_counter++;
  }
  EXTI->PR = (1U << 2);
}

void llfan_init(void) {
  // 5000RPM * 4 tach edges / 60 seconds
  REGISTER_INTERRUPT(EXTI2_IRQn, EXTI2_IRQ_Handler, 700U, FAULT_INTERRUPT_RATE_TACH)

  // Init PWM speed control
  pwm_init(TIM3, 3);

  // Init TACH interrupt
  register_set(&(SYSCFG->EXTICR[0]), SYSCFG_EXTICR1_EXTI2_PD, 0xF00U);
  register_set_bits(&(EXTI->IMR), (1U << 2));
  register_set_bits(&(EXTI->RTSR), (1U << 2));
  register_set_bits(&(EXTI->FTSR), (1U << 2));
  NVIC_EnableIRQ(EXTI2_IRQn);
}
