#ifndef SAFETY_MODES_DEFAULTS_H
#define SAFETY_MODES_DEFAULTS_H

#include <stddef.h>

#include "opendbc/safety/declarations.h"

// *** no output safety mode ***
static safety_config nooutput_init(uint16_t param) {
  SAFETY_UNUSED(param);
  return (safety_config){NULL, 0, NULL, 0, true};  // NOLINT(readability/braces)
}

// no RX checks or TX msgs, so the RX and TX hooks are never called
static const safety_hooks nooutput_hooks = {
  .init = nooutput_init,
};

// *** all output safety mode ***
static safety_config alloutput_init(uint16_t param) {
  // Enables passthrough mode where relay is open and bus 0 gets forwarded to bus 2 and vice versa
  const uint16_t ALLOUTPUT_PARAM_PASSTHROUGH = 1;
  safety.controls_allowed = true;
  bool alloutput_passthrough = GET_FLAG(param, ALLOUTPUT_PARAM_PASSTHROUGH);
  return (safety_config){NULL, 0, NULL, 0, !alloutput_passthrough};  // NOLINT(readability/braces)
}

static bool alloutput_tx_hook(const CANPacket_t *msg) {
  SAFETY_UNUSED(msg);
  return true;
}

// no RX checks, so the RX hook is never called
const safety_hooks alloutput_hooks = {
  .init = alloutput_init,
  .tx = alloutput_tx_hook,
};

#endif  // SAFETY_MODES_DEFAULTS_H
