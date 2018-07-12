void hyundai_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {}

int hyundai_ign_hook() {
  return -1; // use GPIO to determine ignition
}

// *** no output safety mode ***

static void hyundai_init(int16_t param) {
  controls_allowed = 1;
}

static int hyundai_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  return true;
}

static int hyundai_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  return false;
}

static int hyundai_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  return -1;
}

const safety_hooks hyundai_hooks = {
  .init = hyundai_init,
  .rx = hyundai_rx_hook,
  .tx = hyundai_tx_hook,
  .tx_lin = hyundai_tx_lin_hook,
  .ignition = hyundai_ign_hook,
  .fwd = hyundai_fwd_hook,
};

// *** all output safety mode ***

static void alloutput_init(int16_t param) {
  controls_allowed = 1;
}

static int alloutput_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  return true;
}

static int alloutput_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  return true;
}

static int alloutput_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  return -1;
}

const safety_hooks alloutput_hooks = {
  .init = alloutput_init,
  .rx = hyundai_rx_hook,
  .tx = alloutput_tx_hook,
  .tx_lin = alloutput_tx_lin_hook,
  .ignition = hyundai_ign_hook,
  .fwd = alloutput_fwd_hook,
};

