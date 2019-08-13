void clock_init(void) {
  // enable external oscillator
  RCC->CR |= RCC_CR_HSEON;
  while ((RCC->CR & RCC_CR_HSERDY) == 0);

  // divide things
  RCC->CFGR = RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE2_DIV2 | RCC_CFGR_PPRE1_DIV4;

  // 16mhz crystal
  RCC->PLLCFGR = RCC_PLLCFGR_PLLQ_2 | RCC_PLLCFGR_PLLM_3 |
                 RCC_PLLCFGR_PLLN_6 | RCC_PLLCFGR_PLLN_5 | RCC_PLLCFGR_PLLSRC_HSE;

  // start PLL
  RCC->CR |= RCC_CR_PLLON;
  while ((RCC->CR & RCC_CR_PLLRDY) == 0);

  // Configure Flash prefetch, Instruction cache, Data cache and wait state
  // *** without this, it breaks ***
  FLASH->ACR = FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_LATENCY_5WS;

  // switch to PLL
  RCC->CFGR |= RCC_CFGR_SW_PLL;
  while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL);

  // *** running on PLL ***
}

void watchdog_init(void) {
  // setup watchdog
  IWDG->KR = 0x5555;
  IWDG->PR = 0;          // divider /4
  // 0 = 0.125 ms, let's have a 50ms watchdog
  IWDG->RLR = 400 - 1;
  IWDG->KR = 0xCCCC;
}

void watchdog_feed(void) {
  IWDG->KR = 0xAAAA;
}

