void timer_init(TIM_TypeDef *TIM, int psc) {
  register_set(&(TIM->PSC), (psc-1), 0xFFFFU);
  register_set(&(TIM->DIER), TIM_DIER_UIE, 0x5F5FU);
  register_set(&(TIM->CR1), TIM_CR1_CEN, 0x3FU);
  TIM->SR = 0;
}

