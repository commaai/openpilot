#pragma once

typedef struct simple_watchdog_state_t {
  uint32_t fault;
  uint32_t last_ts;
  uint32_t threshold;
} simple_watchdog_state_t;

void simple_watchdog_kick(void);
void simple_watchdog_init(uint32_t fault, uint32_t threshold);
