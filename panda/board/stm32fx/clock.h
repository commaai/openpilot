void clock_init(void) {
  // enable external oscillator
  register_set_bits(&(RCC->CR), RCC_CR_HSEON);
  while ((RCC->CR & RCC_CR_HSERDY) == 0);

  // divide things
  register_set(&(RCC->CFGR), RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE2_DIV2 | RCC_CFGR_PPRE1_DIV4, 0xFF7FFCF3U);

  // 16mhz crystal
  register_set(&(RCC->PLLCFGR), RCC_PLLCFGR_PLLQ_2 | RCC_PLLCFGR_PLLM_3 | RCC_PLLCFGR_PLLN_6 | RCC_PLLCFGR_PLLN_5 | RCC_PLLCFGR_PLLSRC_HSE, 0x7F437FFFU);

  // start PLL
  register_set_bits(&(RCC->CR), RCC_CR_PLLON);
  while ((RCC->CR & RCC_CR_PLLRDY) == 0);

  // Configure Flash prefetch, Instruction cache, Data cache and wait state
  // *** without this, it breaks ***
  register_set(&(FLASH->ACR), FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_LATENCY_5WS, 0x1F0FU);

  // switch to PLL
  register_set_bits(&(RCC->CFGR), RCC_CFGR_SW_PLL);
  while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL);

  // *** running on PLL ***
}

void watchdog_init(void) {
  // setup watchdog
  IWDG->KR = 0x5555U;
  register_set(&(IWDG->PR), 0x0U, 0x7U);  // divider/4

  // 0 = 0.125 ms, let's have a 50ms watchdog
  register_set(&(IWDG->RLR), (400U-1U), 0xFFFU);
  IWDG->KR = 0xCCCCU;
}

void watchdog_feed(void) {
  IWDG->KR = 0xAAAAU;
}
