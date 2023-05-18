void EXTI_IRQ_Handler(void);

void exti_irq_init(void) {
  SYSCFG->EXTICR[2] &= ~(SYSCFG_EXTICR3_EXTI8_Msk);
  if (harness.status == HARNESS_STATUS_FLIPPED) {
    // CAN2_RX
    current_board->enable_can_transceiver(3U, false);
    SYSCFG->EXTICR[2] |=  (SYSCFG_EXTICR3_EXTI8_PA);

    // IRQ on falling edge for PC3 (SBU2, EXTI3)
    SYSCFG->EXTICR[0] &= ~(SYSCFG_EXTICR1_EXTI3_Msk);
    SYSCFG->EXTICR[0] |=  (SYSCFG_EXTICR1_EXTI3_PC);
    EXTI->IMR  |=  EXTI_IMR_MR3;
    EXTI->RTSR &=  ~EXTI_RTSR_TR3; // rising edge
    EXTI->FTSR |=  EXTI_FTSR_TR3; // falling edge
    REGISTER_INTERRUPT(EXTI3_IRQn, EXTI_IRQ_Handler, 100U, FAULT_INTERRUPT_RATE_EXTI)
    NVIC_EnableIRQ(EXTI3_IRQn);
  } else {
    // CAN0_RX
    current_board->enable_can_transceiver(1U, false);
    SYSCFG->EXTICR[2] |=  (SYSCFG_EXTICR3_EXTI8_PB);

    // IRQ on falling edge for PC0 (SBU1, EXTI0)
    SYSCFG->EXTICR[0] &= ~(SYSCFG_EXTICR1_EXTI0_Msk);
    SYSCFG->EXTICR[0] |=  (SYSCFG_EXTICR1_EXTI0_PC);
    EXTI->IMR  |=  EXTI_IMR_MR0;
    EXTI->RTSR &=  ~EXTI_RTSR_TR0; // rising edge
    EXTI->FTSR |=  EXTI_FTSR_TR0; // falling edge
    REGISTER_INTERRUPT(EXTI0_IRQn, EXTI_IRQ_Handler, 100U, FAULT_INTERRUPT_RATE_EXTI)
    NVIC_EnableIRQ(EXTI0_IRQn);
  }
  // CAN0 or CAN2 IRQ on falling edge (EXTI8)
  EXTI->IMR  |=  EXTI_IMR_MR8;
  EXTI->RTSR &=  ~EXTI_RTSR_TR8; // rising edge
  EXTI->FTSR |=  EXTI_FTSR_TR8; // falling edge
  REGISTER_INTERRUPT(EXTI9_5_IRQn, EXTI_IRQ_Handler, 100U, FAULT_INTERRUPT_RATE_EXTI)
  NVIC_EnableIRQ(EXTI9_5_IRQn);
}

bool check_exti_irq(void) {
  return ((EXTI->PR & EXTI_PR_PR8) || (EXTI->PR & EXTI_PR_PR3) || (EXTI->PR & EXTI_PR_PR0));
}

void exti_irq_clear(void) {
  // Clear pending bits
  EXTI->PR |= EXTI_PR_PR8;
  EXTI->PR |= EXTI_PR_PR0;
  EXTI->PR |= EXTI_PR_PR3;
  EXTI->PR |= EXTI_PR_PR22;

  // Disable all active EXTI IRQs
  EXTI->IMR &= ~EXTI_IMR_MR8;
  EXTI->IMR &= ~EXTI_IMR_MR0;
  EXTI->IMR &= ~EXTI_IMR_MR3;
  EXTI->IMR &= ~EXTI_IMR_MR22;
}
