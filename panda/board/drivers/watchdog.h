// TODO: why doesn't it define these?
#ifdef STM32F2
#define IWDG_PR_PR_Msk 0x7U
#define IWDG_RLR_RL_Msk 0xFFFU
#endif

typedef enum {
  WATCHDOG_50_MS = (400U - 1U),
  WATCHDOG_500_MS = 4000U,
} WatchdogTimeout;

void watchdog_feed(void) {
  IND_WDG->KR = 0xAAAAU;
}

void watchdog_init(WatchdogTimeout timeout) {
  // enable watchdog
  IND_WDG->KR = 0xCCCCU;
  IND_WDG->KR = 0x5555U;

  // 32KHz / 4 prescaler = 8000Hz
  register_set(&(IND_WDG->PR), 0x0U, IWDG_PR_PR_Msk);
  register_set(&(IND_WDG->RLR), timeout, IWDG_RLR_RL_Msk);

  // wait for watchdog to be updated
  while (IND_WDG->SR != 0U);

  // start the countdown
  watchdog_feed();
}
