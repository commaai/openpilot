void timer_init(TIM_TypeDef *TIM, int psc) {
  TIM->PSC = psc-1;
  TIM->DIER = TIM_DIER_UIE;
  TIM->CR1 = TIM_CR1_CEN;
  TIM->SR = 0;
}

