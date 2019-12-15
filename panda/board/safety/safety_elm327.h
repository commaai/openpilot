static int elm327_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);
  int len = GET_LEN(to_send);

  //All ISO 15765-4 messages must be 8 bytes long
  if (len != 8) {
    tx = 0;
  }

  //Check valid 29 bit send addresses for ISO 15765-4
  //Check valid 11 bit send addresses for ISO 15765-4
  if ((addr != 0x18DB33F1) && ((addr & 0x1FFF00FF) != 0x18DA00F1) &&
      ((addr != 0x7DF) && ((addr & 0x7F8) != 0x7E0))) {
    tx = 0;
  }
  return tx;
}

static int elm327_tx_lin_hook(int lin_num, uint8_t *data, int len) {
  int tx = 1;
  if (lin_num != 0) {
    tx = 0;  //Only operate on LIN 0, aka serial 2
  }
  if ((len < 5) || (len > 11)) {
    tx = 0;  //Valid KWP size
  }
  if (!(((data[0] & 0xF8U) == 0xC0U) && ((data[0] & 0x07U) != 0U) &&
        (data[1] == 0x33U) && (data[2] == 0xF1U))) {
    tx = 0;  //Bad msg
  }
  return tx;
}

const safety_hooks elm327_hooks = {
  .init = nooutput_init,
  .rx = default_rx_hook,
  .tx = elm327_tx_hook,
  .tx_lin = elm327_tx_lin_hook,
  .fwd = default_fwd_hook,
};
