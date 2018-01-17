void default_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {}

// *** no output safety mode ***

static void nooutput_init() {
  controls_allowed = 0;
}

static int nooutput_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  return false;
}

static int nooutput_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  return false;
}

const safety_hooks nooutput_hooks = {
  .init = nooutput_init,
  .rx = default_rx_hook,
  .tx = nooutput_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
};

// *** all output safety mode ***

static void alloutput_init() {
  controls_allowed = 1;
}

static int alloutput_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  return true;
}

static int alloutput_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  return true;
}

const safety_hooks alloutput_hooks = {
  .init = alloutput_init,
  .rx = default_rx_hook,
  .tx = alloutput_tx_hook,
  .tx_lin = alloutput_tx_lin_hook,
};

