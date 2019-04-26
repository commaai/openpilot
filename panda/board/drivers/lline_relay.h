#ifdef PANDA

int relay_control = 0;  // True if relay is controlled through l-line

/* Conrol a relay connected to l-line pin */

// 160us cycles, 1 high, 25 low

volatile int turn_on_relay = 0;
volatile int on_cycles = 25;

//5s timeout
#define LLINE_TIMEOUT_CYCLES 31250
volatile int timeout_cycles = LLINE_TIMEOUT_CYCLES;

void TIM5_IRQHandler(void) {
  if (TIM5->SR & TIM_SR_UIF) {
    on_cycles--;
    timeout_cycles--;
    if (timeout_cycles == 0) {
      turn_on_relay = 0;
    }
    if (on_cycles > 0) {
      if (turn_on_relay) {
        set_gpio_output(GPIOC, 10, 0);
      }
    }
    else {
      set_gpio_output(GPIOC, 10, 1);
      on_cycles = 25;
    }
  }
  TIM5->ARR = 160-1;
  TIM5->SR = 0;
}

void lline_relay_init (void) {
  set_lline_output(0);
  relay_control = 1;
  set_gpio_output(GPIOC, 10, 1);

  // setup
  TIM5->PSC = 48-1; // tick on 1 us
  TIM5->CR1 = TIM_CR1_CEN;   // enable
  TIM5->ARR = 50-1;         // 50 us
  TIM5->DIER = TIM_DIER_UIE; // update interrupt
  TIM5->CNT = 0;

  NVIC_EnableIRQ(TIM5_IRQn);

#ifdef DEBUG
  puts("INIT LLINE\n");
  puts(" SR ");
  putui(TIM5->SR);
  puts(" PSC ");
  putui(TIM5->PSC);
  puts(" CR1 ");
  putui(TIM5->CR1);
  puts(" ARR ");
  putui(TIM5->ARR);
  puts(" DIER ");
  putui(TIM5->DIER);
  puts(" SR ");
  putui(TIM5->SR);
  puts(" CNT ");
  putui(TIM5->CNT);
  puts("\n");
#endif
}

void lline_relay_release (void) {
  set_lline_output(0);
  relay_control = 0;
  puts("RELEASE LLINE\n");
  set_gpio_alternate(GPIOC, 10, GPIO_AF7_USART3);
  NVIC_DisableIRQ(TIM5_IRQn);
}

void set_lline_output(int to_set) {
  timeout_cycles = LLINE_TIMEOUT_CYCLES;
  turn_on_relay = to_set;
}

int get_lline_status() {
  return turn_on_relay;
}

#endif
