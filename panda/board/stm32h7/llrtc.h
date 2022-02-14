#define RCC_BDCR_MASK_LSE (RCC_BDCR_RTCEN | RCC_BDCR_RTCSEL | RCC_BDCR_LSEDRV | RCC_BDCR_LSEBYP | RCC_BDCR_LSEON)
#define RCC_BDCR_MASK_LSI (RCC_BDCR_RTCEN | RCC_BDCR_RTCSEL)

void enable_bdomain_protection(void) {
  register_clear_bits(&(PWR->CR1), PWR_CR1_DBP);
}

void disable_bdomain_protection(void) {
  register_set_bits(&(PWR->CR1), PWR_CR1_DBP);
}
