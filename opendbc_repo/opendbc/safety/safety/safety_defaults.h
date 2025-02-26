#pragma once

#include "safety_declarations.h"

void default_rx_hook(const CANPacket_t *to_push) {
  UNUSED(to_push);
}

// *** no output safety mode ***

static safety_config nooutput_init(uint16_t param) {
  UNUSED(param);
  return (safety_config){NULL, 0, NULL, 0};
}

static bool nooutput_tx_hook(const CANPacket_t *to_send) {
  UNUSED(to_send);
  return false;
}

static int default_fwd_hook(int bus_num, int addr) {
  UNUSED(bus_num);
  UNUSED(addr);
  return -1;
}

const safety_hooks nooutput_hooks = {
  .init = nooutput_init,
  .rx = default_rx_hook,
  .tx = nooutput_tx_hook,
  .fwd = default_fwd_hook,
};

// *** all output safety mode ***

// Enables passthrough mode where relay is open and bus 0 gets forwarded to bus 2 and vice versa
static bool alloutput_passthrough = false;

static safety_config alloutput_init(uint16_t param) {
  // Enables passthrough mode where relay is open and bus 0 gets forwarded to bus 2 and vice versa
  const uint16_t ALLOUTPUT_PARAM_PASSTHROUGH = 1;
  controls_allowed = true;
  alloutput_passthrough = GET_FLAG(param, ALLOUTPUT_PARAM_PASSTHROUGH);
  return (safety_config){NULL, 0, NULL, 0};
}

static bool alloutput_tx_hook(const CANPacket_t *to_send) {
  UNUSED(to_send);
  return true;
}

static int alloutput_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;
  UNUSED(addr);

  if (alloutput_passthrough) {
    if (bus_num == 0) {
      bus_fwd = 2;
    }
    if (bus_num == 2) {
      bus_fwd = 0;
    }
  }

  return bus_fwd;
}

const safety_hooks alloutput_hooks = {
  .init = alloutput_init,
  .rx = default_rx_hook,
  .tx = alloutput_tx_hook,
  .fwd = alloutput_fwd_hook,
};
