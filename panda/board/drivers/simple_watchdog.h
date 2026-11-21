#ifndef BOARD_DRIVERS_SIMPLE_WATCHDOG_H
#define BOARD_DRIVERS_SIMPLE_WATCHDOG_H

#include "board/drivers/drivers.h"

static simple_watchdog_state_t wd_state;

void simple_watchdog_kick(void) {
  uint32_t ts = microsecond_timer_get();

  uint32_t et = get_ts_elapsed(ts, wd_state.last_ts);
  if (et > wd_state.threshold) {
    print("WD timeout 0x"); puth(et); print("\n");
    fault_occurred(wd_state.fault);
  }

  wd_state.last_ts = ts;
}

void simple_watchdog_init(uint32_t fault, uint32_t threshold) {
  wd_state.fault = fault;
  wd_state.threshold = threshold;
  wd_state.last_ts = microsecond_timer_get();
}

#endif  // BOARD_DRIVERS_SIMPLE_WATCHDOG_H
