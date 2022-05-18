void EXTI_IRQ_Handler(void);

void exti_irq_init(void) {
  if (car_harness_status == HARNESS_STATUS_FLIPPED) {
    // CAN2_RX IRQ on falling edge (EXTI10)
    current_board->enable_can_transceiver(3U, false);
    SYSCFG->EXTICR[2] &= ~(SYSCFG_EXTICR3_EXTI10_Msk);
    SYSCFG->EXTICR[2] |=  (SYSCFG_EXTICR3_EXTI10_PG);
    EXTI->IMR1  |=  EXTI_IMR1_IM10;
    EXTI->RTSR1 &=  ~EXTI_RTSR1_TR10; // rising edge
    EXTI->FTSR1 |=  EXTI_FTSR1_TR10; // falling edge

    // IRQ on falling edge for PA1 (SBU2, EXTI1)
    SYSCFG->EXTICR[0] &= ~(SYSCFG_EXTICR1_EXTI1_Msk);
    SYSCFG->EXTICR[0] |=  (SYSCFG_EXTICR1_EXTI1_PA);
    EXTI->IMR1  |=  EXTI_IMR1_IM1;
    EXTI->RTSR1 &=  ~EXTI_RTSR1_TR1; // rising edge
    EXTI->FTSR1 |=  EXTI_FTSR1_TR1; // falling edge
    REGISTER_INTERRUPT(EXTI1_IRQn, EXTI_IRQ_Handler, 100U, FAULT_INTERRUPT_RATE_EXTI)
    NVIC_EnableIRQ(EXTI1_IRQn);
    REGISTER_INTERRUPT(EXTI15_10_IRQn, EXTI_IRQ_Handler, 100U, FAULT_INTERRUPT_RATE_EXTI)
    NVIC_EnableIRQ(EXTI15_10_IRQn);
  } else {
    // CAN0_RX IRQ on falling edge (EXTI8)
    current_board->enable_can_transceiver(1U, false);
    SYSCFG->EXTICR[2] &= ~(SYSCFG_EXTICR3_EXTI8_Msk);
    SYSCFG->EXTICR[2] |=  (SYSCFG_EXTICR3_EXTI8_PB);
    EXTI->IMR1  |=  EXTI_IMR1_IM8;
    EXTI->RTSR1 &=  ~EXTI_RTSR1_TR8; // rising edge
    EXTI->FTSR1 |=  EXTI_FTSR1_TR8; // falling edge

    // IRQ on falling edge for PC4 (SBU1, EXTI4)
    SYSCFG->EXTICR[1] &= ~(SYSCFG_EXTICR2_EXTI4_Msk);
    SYSCFG->EXTICR[1] |=  (SYSCFG_EXTICR2_EXTI4_PC);
    EXTI->IMR1  |=  EXTI_IMR1_IM4;
    EXTI->RTSR1 &=  ~EXTI_RTSR1_TR4; // rising edge
    EXTI->FTSR1 |=  EXTI_FTSR1_TR4; // falling edge
    REGISTER_INTERRUPT(EXTI4_IRQn, EXTI_IRQ_Handler, 100U, FAULT_INTERRUPT_RATE_EXTI)
    NVIC_EnableIRQ(EXTI4_IRQn);
    REGISTER_INTERRUPT(EXTI9_5_IRQn, EXTI_IRQ_Handler, 100U, FAULT_INTERRUPT_RATE_EXTI)
    NVIC_EnableIRQ(EXTI9_5_IRQn);
  }
}

bool check_exti_irq(void) {
  return ((EXTI->PR1 & EXTI_PR1_PR8) || (EXTI->PR1 & EXTI_PR1_PR10) || (EXTI->PR1 & EXTI_PR1_PR1) || (EXTI->PR1 & EXTI_PR1_PR4));
}

void exti_irq_clear(void) {
  // Clear pending bits
  EXTI->PR1 |= EXTI_PR1_PR8;
  EXTI->PR1 |= EXTI_PR1_PR10;
  EXTI->PR1 |= EXTI_PR1_PR4;
  EXTI->PR1 |= EXTI_PR1_PR1; // works
  EXTI->PR1 |= EXTI_PR1_PR19; // works

  // Disable all active EXTI IRQs
  EXTI->IMR1 &= ~EXTI_IMR1_IM8;
  EXTI->IMR1 &= ~EXTI_IMR1_IM10;
  EXTI->IMR1 &= ~EXTI_IMR1_IM4;
  EXTI->IMR1 &= ~EXTI_IMR1_IM1;
  EXTI->IMR1 &= ~EXTI_IMR1_IM19;
}
