const CanMsg BODY_TX_MSGS[] = {{0x250, 0, 8}, {0x250, 0, 6}, {0x251, 0, 5},  // body
                               {0x350, 0, 8}, {0x350, 0, 6}, {0x351, 0, 5}}; // knee

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

static int body_tx_hook(CANPacket_t *to_send) {

  int tx = 0;
  int addr = GET_ADDR(to_send);

  // CAN flasher
  if (addr == 0x1) {
    tx = 1;
  }

  if (msg_allowed(to_send, BODY_TX_MSGS, sizeof(BODY_TX_MSGS)/sizeof(BODY_TX_MSGS[0])) && controls_allowed) {
    tx = 1;
  }

  // Allow going into CAN flashing mode even if controls are not allowed
  if (!controls_allowed && ((uint32_t)GET_BYTES_04(to_send) == 0xdeadfaceU) && ((uint32_t)GET_BYTES_48(to_send) == 0x0ab00b1eU)) {
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
