#define RCC_BDCR_MASK (RCC_BDCR_RTCEN | RCC_BDCR_RTCSEL | RCC_BDCR_LSEMOD | RCC_BDCR_LSEBYP | RCC_BDCR_LSEON)

void enable_bdomain_protection(void) {
  register_clear_bits(&(PWR->CR), PWR_CR_DBP);
}

void disable_bdomain_protection(void) {
  register_set_bits(&(PWR->CR), PWR_CR_DBP);
}
