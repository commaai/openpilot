AddrCheckStruct body_addr_checks[] = {
  {.msg = {{0x201, 0, 8, .check_checksum = false, .max_counter = 0U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
};
#define BODY_ADDR_CHECK_LEN (sizeof(body_addr_checks) / sizeof(body_addr_checks[0]))
addr_checks body_rx_checks = {body_addr_checks, BODY_ADDR_CHECK_LEN};

static int body_rx_hook(CANPacket_t *to_push) {

  bool valid = addr_safety_check(to_push, &body_rx_checks, NULL, NULL, NULL);

  controls_allowed = valid;

  return valid;
}

static int body_tx_hook(CANPacket_t *to_send, bool longitudinal_allowed) {
  UNUSED(longitudinal_allowed);

  int tx = 0;
  int addr = GET_ADDR(to_send);

  // CAN flasher
  if (addr == 0x1) {
    tx = 1;
  }

  if ((addr == 0x250) && controls_allowed) {
    tx = 1;
  }

  return tx;
}

static const addr_checks* body_init(uint16_t param) {
  UNUSED(param);
  return &body_rx_checks;
}

const safety_hooks body_hooks = {
  .init = body_init,
  .rx = body_rx_hook,
  .tx = body_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = default_fwd_hook,
};
