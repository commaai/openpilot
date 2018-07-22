// Kia Hyundai Safety

const int MAX_STEER = 


static int hyundai_ign_hook() {
  return true;
}
static void hyundai_init(int16_t param) {
  controls_allowed = 1;
}

static int hyundai_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {
  // Steering Value Safety Check
  
  return true;
}

static int hyundai_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  return true;
}

static int hyundai_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {
  return -1;
}

const safety_hooks hyundai_hooks = {
  .init = hyundai_init,
  .rx = default_rx_hook,
  .tx = hyundai_tx_hook,
  .tx_lin = hyundai_tx_lin_hook,
  .ignition = hyundai_ign_hook,
  .fwd = hyundai_fwd_hook,
};

