#define RCC_BDCR_MASK_LSE (RCC_BDCR_RTCEN | RCC_BDCR_RTCSEL | RCC_BDCR_LSEMOD | RCC_BDCR_LSEBYP | RCC_BDCR_LSEON)
#define RCC_BDCR_MASK_LSI (RCC_BDCR_RTCEN | RCC_BDCR_RTCSEL)

void enable_bdomain_protection(void) {
  register_clear_bits(&(PWR->CR), PWR_CR_DBP);
}

void disable_bdomain_protection(void) {
  register_set_bits(&(PWR->CR), PWR_CR_DBP);
}

void rtc_wakeup_init(void) {
  EXTI->IMR  |=  EXTI_IMR_MR22;
  EXTI->RTSR |=  EXTI_RTSR_TR22; // rising edge
  EXTI->FTSR &=  ~EXTI_FTSR_TR22; // falling edge

  NVIC_DisableIRQ(RTC_WKUP_IRQn);

  // Disable write protection
  disable_bdomain_protection();
  RTC->WPR = 0xCA;
  RTC->WPR = 0x53;

  RTC->CR &= ~RTC_CR_WUTE;
  while((RTC->ISR & RTC_ISR_WUTWF) == 0){}

  RTC->CR &= ~RTC_CR_WUTIE;
  RTC->ISR &= ~RTC_ISR_WUTF;
  //PWR->CR |= PWR_CR_CWUF;

  RTC->WUTR = DEEPSLEEP_WAKEUP_DELAY;
  // Wakeup timer interrupt enable, wakeup timer enable, select 1Hz rate
  RTC->CR |= RTC_CR_WUTE | RTC_CR_WUTIE | RTC_CR_WUCKSEL_2;

  // Re-enable write protection
  RTC->WPR = 0x00;
  enable_bdomain_protection();

  NVIC_EnableIRQ(RTC_WKUP_IRQn);
}
