int default_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {
  UNUSED(to_push);
  return true;
}

// *** no output safety mode ***

static void nooutput_init(int16_t param) {
  UNUSED(param);
  controls_allowed = false;
  relay_malfunction_reset();
}

static int nooutput_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  UNUSED(to_send);
  return false;
}

static int nooutput_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  UNUSED(lin_num);
  UNUSED(data);
  UNUSED(len);
  return false;
}

static int default_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  // Volkswagen community port: Advanced Virtual Relay Technology!
  // Make Panda fully transparent from bus 0->2 and bus 2->0 if not otherwise
  // instructed by EON/OP, returning the car to stock behavior under NOOUTPUT.
  // Don't do this for BP/C2, where we have Advanced Actual Relay Technology.
  UNUSED(to_fwd);
  int bus_fwd = -1;

  if(!board_has_relay()) {
    switch (bus_num) {
      case 0:
        bus_fwd = -1;
        break;
      case 2:
        bus_fwd = -1;
        break;
      default:
        bus_fwd = -1;
        break;
    }
  }

  return bus_fwd;
}

static int no_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  // Volkswagen community port: Not actually for Volkswagen!
  // GM needs an actual no-forwarding hook to pass regression tests because
  // their non-ASCM port doesn't actually use forwarding of its own. Easier
  // to change GM to this, than change all other users of default.
  UNUSED(to_fwd);
  UNUSED(bus_num);
  return -1;
}

const safety_hooks nooutput_hooks = {
  // Volkswagen community port:
  // In NOOUTPUT mode, Panda doesn't allow any TX from EON as usual, but keeps
  // the transceivers active and goes into a transparent forwarding mode.
  .init = nooutput_init,
  .rx = default_rx_hook,
  .tx = nooutput_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = default_fwd_hook,
};

// *** all output safety mode ***

static void alloutput_init(int16_t param) {
  UNUSED(param);
  controls_allowed = true;
  relay_malfunction_reset();
}

static int alloutput_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  UNUSED(to_send);
  return true;
}

static int alloutput_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  UNUSED(lin_num);
  UNUSED(data);
  UNUSED(len);
  return true;
}

const safety_hooks alloutput_hooks = {
  .init = alloutput_init,
  .rx = default_rx_hook,
  .tx = alloutput_tx_hook,
  .tx_lin = alloutput_tx_lin_hook,
  .fwd = default_fwd_hook,
};
