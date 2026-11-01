#pragma once

#include "board/drivers/drivers.h"

static void unused_interrupt_handler(void) {
  // Something is wrong if this handler is called!
  print("Unused interrupt handler called!\n");
  fault_occurred_value(FAULT_UNUSED_INTERRUPT_HANDLED, __get_IPSR());
}

interrupt interrupts[NUM_INTERRUPTS];

// NVIC_SM_0: every IRQ enable goes through nvic_enable/nvic_disable, except the stop mode wakeups, which exit through a reset
static uint32_t nvic_expected[8];

void nvic_enable(IRQn_Type irq) {
  ENTER_CRITICAL();
  nvic_expected[(uint32_t)irq >> 5U] |= (1UL << ((uint32_t)irq & 0x1FU));
  NVIC_EnableIRQ(irq);
  EXIT_CRITICAL();
}

void nvic_disable(IRQn_Type irq) {
  ENTER_CRITICAL();
  nvic_expected[(uint32_t)irq >> 5U] &= ~(1UL << ((uint32_t)irq & 0x1FU));
  NVIC_DisableIRQ(irq);
  EXIT_CRITICAL();
}

// logs the first divergent IRQ, 0x100 added if it's the priority
void nvic_check(void) {
  uint32_t diff = 0U;
  uint32_t first = 0U;
  for (uint8_t i = 0U; i < 8U; i++) {
    uint32_t d = NVIC->ISER[i] ^ nvic_expected[i];
    if ((diff == 0U) && (d != 0U)) {
      first = ((uint32_t)i * 32U) + __CLZ(__RBIT(d));
    }
    diff |= d;
  }
  // nothing sets a priority, all IRQs stay at 0
  for (uint8_t i = 0U; i < 240U; i++) {
    if ((diff == 0U) && (NVIC->IP[i] != 0U)) {
      first = 0x100U + (uint32_t)i;
    }
    diff |= NVIC->IP[i];
  }
  if (diff != 0U) {
    fault_occurred_value(FAULT_REGISTER_DIVERGENT, first);
  }
}

static bool check_interrupt_rate = false;

static uint32_t idle_time = 0U;
static uint32_t busy_time = 0U;
float interrupt_load = 0.0f;

void handle_interrupt(IRQn_Type irq_type){
  static uint8_t interrupt_depth = 0U;
  static uint32_t last_time = 0U;
  ENTER_CRITICAL();
  if (interrupt_depth == 0U) {
    uint32_t time = microsecond_timer_get();
    idle_time += get_ts_elapsed(time, last_time);
    last_time = time;
  }
  interrupt_depth += 1U;
  EXIT_CRITICAL();

  interrupts[irq_type].call_counter++;
  interrupts[irq_type].handler();

  // Check that the interrupts don't fire too often
  if (check_interrupt_rate && (interrupts[irq_type].call_counter > interrupts[irq_type].max_call_rate)) {
    fault_occurred(interrupts[irq_type].call_rate_fault);
  }

  ENTER_CRITICAL();
  interrupt_depth -= 1U;
  if (interrupt_depth == 0U) {
    uint32_t time = microsecond_timer_get();
    busy_time += get_ts_elapsed(time, last_time);
    last_time = time;
  }
  EXIT_CRITICAL();
}

static uint32_t interrupt_timer_windows = 0U;

// Every second
void interrupt_timer_handler(void) {
  static timer_check_t interrupt_timer_period = {0};
  if (INTERRUPT_TIMER->SR != 0U) {
    // cleared first, a flag cleared on the way out can retrigger the vector
    INTERRUPT_TIMER->SR = 0;
    interrupt_timer_windows += 1U;
    uint32_t dt = 0U;
    if (timer_check(&interrupt_timer_period, &dt)) {
      fault_occurred_value(FAULT_TIMER, dt);
    }
    for (uint16_t i = 0U; i < NUM_INTERRUPTS; i++) {
      // Log IRQ call rate faults
      if (check_interrupt_rate && (interrupts[i].call_counter > interrupts[i].max_call_rate)) {
        print("Interrupt 0x"); puth(i); print(" fired too often (0x"); puth(interrupts[i].call_counter); print("/s)!\n");
      }

      // the first window spans boot
      if (check_interrupt_rate && (interrupt_timer_windows > 1U) && (interrupts[i].call_counter < interrupts[i].min_call_rate)) {
        print("Interrupt 0x"); puth(i); print(" stopped firing (0x"); puth(interrupts[i].call_counter); print("/s)!\n");
        fault_occurred(FAULT_INTERRUPT_MISSING);
      }

      // Reset interrupt counters
      interrupts[i].call_rate = interrupts[i].call_counter;
      interrupts[i].call_counter = 0U;
    }

    // Calculate interrupt load
    // The bootstub does not have the FPU enabled, so can't do float operations.
#if !defined(BOOTSTUB)
    interrupt_load = ((busy_time + idle_time) > 0U) ? ((float) (((float) busy_time) / (busy_time + idle_time))) : 0.0f;
#endif
    idle_time = 0U;
    busy_time = 0U;
  } else {
    // TIM6 shares the vector with the DAC, which has no interrupts enabled
    fault_occurred_value(FAULT_UNUSED_INTERRUPT_HANDLED, __get_IPSR());
  }
}

// called at 1Hz from the tick, one miss is normal since the two timers drift
void interrupt_timer_check(void) {
  static uint32_t last_windows = 0U;
  static uint32_t misses = 0U;
  misses = (interrupt_timer_windows == last_windows) ? (misses + 1U) : 0U;
  last_windows = interrupt_timer_windows;
  if (check_interrupt_rate && (misses >= 2U)) {
    fault_occurred(FAULT_INTERRUPT_MISSING);
  }
}

void init_interrupts(bool check_rate_limit){
  check_interrupt_rate = check_rate_limit;

  for(uint16_t i=0U; i<NUM_INTERRUPTS; i++){
    interrupts[i].handler = unused_interrupt_handler;
    interrupts[i].min_call_rate = 0U;
  }

  // Init interrupt timer for a 1s interval
  interrupt_timer_init();
}
