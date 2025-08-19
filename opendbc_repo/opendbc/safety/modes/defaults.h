#pragma once

#include "opendbc/safety/safety_declarations.h"

// GCOV_EXCL_START
// Unreachable by design (doesn't define any rx msgs)
void default_rx_hook(const CANPacket_t *msg) {
  UNUSED(msg);
}
// GCOV_EXCL_STOP

// *** no output safety mode ***

static safety_config nooutput_init(uint16_t param) {
  UNUSED(param);
  return (safety_config){NULL, 0, NULL, 0, true}; // NOLINT(readability/braces)
}

// GCOV_EXCL_START
// Unreachable by design (doesn't define any tx msgs)
static bool nooutput_tx_hook(const CANPacket_t *msg) {
  UNUSED(msg);
  return false;
}
// GCOV_EXCL_STOP

const safety_hooks nooutput_hooks = {
  .init = nooutput_init,
  .rx = default_rx_hook,
  .tx = nooutput_tx_hook,
};

// *** all output safety mode ***
static safety_config alloutput_init(uint16_t param) {
  // Enables passthrough mode where relay is open and bus 0 gets forwarded to bus 2 and vice versa
  const uint16_t ALLOUTPUT_PARAM_PASSTHROUGH = 1;
  controls_allowed = true;
  bool alloutput_passthrough = GET_FLAG(param, ALLOUTPUT_PARAM_PASSTHROUGH);
  return (safety_config){NULL, 0, NULL, 0, !alloutput_passthrough}; // NOLINT(readability/braces)
}

static bool alloutput_tx_hook(const CANPacket_t *msg) {
  UNUSED(msg);
  return true;
}

const safety_hooks alloutput_hooks = {
  .init = alloutput_init,
  .rx = default_rx_hook,
  .tx = alloutput_tx_hook,
};
