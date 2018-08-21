int hyundai_giraffe_switch_1 = 0;          // is giraffe switch 1 high?


static void hyundai_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {

  int bus = (to_push->RDTR >> 4) & 0xF;
  // 832 is lkas cmd. If it is on bus 0, then giraffe switch 1 is high and we want stock
  if ((to_push->RIR>>21) == 832 && (bus == 0)) {
    hyundai_giraffe_switch_1 = 1;
  }
}

static void hyundai_init(int16_t param) {
  controls_allowed = 0;
  hyundai_giraffe_switch_1 = 0;
}

static int hyundai_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {

  // forward camera to car and viceversa, excpet for lkas11 and mdps12
  if ((bus_num == 0 || bus_num == 2) && !hyundai_giraffe_switch_1) {
    int addr = to_fwd->RIR>>21;
    bool is_lkas_msg = (addr == 832 && bus_num == 2) || (addr == 593 && bus_num == 0);
    return is_lkas_msg? -1 : (uint8_t)(~bus_num & 0x2);
  }
  return -1;
}

const safety_hooks hyundai_hooks = {
  .init = hyundai_init,
  .rx = hyundai_rx_hook,
  .tx = nooutput_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .ignition = default_ign_hook,
  .fwd = hyundai_fwd_hook,
};
