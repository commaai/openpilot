#include "opendbc/safety/safety.h"

// this file is checked by cppcheck

uint32_t microsecond_timer_get(void) {
  return 0U;
}

// Stands in for the firmware and libsafety, the callers of the safety API
// declared in declarations.h. Referencing the entry points here keeps
// misra-c2012-8.7 from asking for them to have internal linkage.
int main(void) {
  SAFETY_UNUSED(safety.heartbeat_engaged);

  SAFETY_UNUSED(safety_rx_hook);
  SAFETY_UNUSED(safety_tx_hook);
  SAFETY_UNUSED(safety_fwd_hook);
  SAFETY_UNUSED(safety_tick);
  SAFETY_UNUSED(safety_heartbeat);
  SAFETY_UNUSED(safety_heartbeat_check);
  SAFETY_UNUSED(safety_alternative_experience);
  SAFETY_UNUSED(set_safety_hooks);
  return 0;
}
