#include "critical_declarations.h"

// ********************* Critical section helpers *********************
uint8_t global_critical_depth = 0U;

static volatile bool interrupts_enabled = false;

void enable_interrupts(void) {
  interrupts_enabled = true;
  __enable_irq();
}

void disable_interrupts(void) {
  interrupts_enabled = false;
  __disable_irq();
}
