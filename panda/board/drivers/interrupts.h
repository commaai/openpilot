typedef struct interrupt {
  IRQn_Type irq_type;
  void (*handler)(void);
  uint32_t call_counter;
  uint32_t max_call_rate;   // Call rate is defined as the amount of calls each second
  uint32_t call_rate_fault;
} interrupt;

void interrupt_timer_init(void);
uint32_t microsecond_timer_get(void);

void unused_interrupt_handler(void) {
  // Something is wrong if this handler is called!
  puts("Unused interrupt handler called!\n");
  fault_occurred(FAULT_UNUSED_INTERRUPT_HANDLED);
}

interrupt interrupts[NUM_INTERRUPTS];

#define REGISTER_INTERRUPT(irq_num, func_ptr, call_rate, rate_fault) \
  interrupts[irq_num].irq_type = (irq_num); \
  interrupts[irq_num].handler = (func_ptr);  \
  interrupts[irq_num].call_counter = 0U;   \
  interrupts[irq_num].max_call_rate = (call_rate); \
  interrupts[irq_num].call_rate_fault = (rate_fault);

bool check_interrupt_rate = false;

uint8_t interrupt_depth = 0U;
uint32_t last_time = 0U;
uint32_t idle_time = 0U;
uint32_t busy_time = 0U;
float interrupt_load = 0.0f;

void handle_interrupt(IRQn_Type irq_type){
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
  if(check_interrupt_rate && (interrupts[irq_type].call_counter > interrupts[irq_type].max_call_rate)){
    puts("Interrupt 0x"); puth(irq_type); puts(" fired too often (0x"); puth(interrupts[irq_type].call_counter); puts("/s)!\n");
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

// Every second
void interrupt_timer_handler(void) {
  if (INTERRUPT_TIMER->SR != 0) {
    // Reset interrupt counters
    for(uint16_t i=0U; i<NUM_INTERRUPTS; i++){
      interrupts[i].call_counter = 0U;
    }

    // Calculate interrupt load
    // The bootstub does not have the FPU enabled, so can't do float operations.
#if !defined(PEDAL) && !defined(BOOTSTUB)
    interrupt_load = ((busy_time + idle_time) > 0U) ? ((float) busy_time) / (busy_time + idle_time) : 0.0f;
#endif
    idle_time = 0U;
    busy_time = 0U;
  }
  INTERRUPT_TIMER->SR = 0;
}

void init_interrupts(bool check_rate_limit){
  check_interrupt_rate = check_rate_limit;

  for(uint16_t i=0U; i<NUM_INTERRUPTS; i++){
    interrupts[i].handler = unused_interrupt_handler;
  }

  // Init interrupt timer for a 1s interval
  interrupt_timer_init();
}
