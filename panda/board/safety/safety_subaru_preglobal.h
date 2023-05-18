const SteeringLimits SUBARU_PG_STEERING_LIMITS = {
  .max_steer = 2047,
  .max_rt_delta = 940,
  .max_rt_interval = 250000,
  .max_rate_up = 50,
  .max_rate_down = 70,
  .driver_torque_factor = 10,
  .driver_torque_allowance = 75,
  .type = TorqueDriverLimited,
};

const CanMsg SUBARU_PG_TX_MSGS[] = {
  {0x161, 0, 8},
  {0x164, 0, 8}
};
#define SUBARU_PG_TX_MSGS_LEN (sizeof(SUBARU_PG_TX_MSGS) / sizeof(SUBARU_PG_TX_MSGS[0]))

// TODO: do checksum and counter checks after adding the signals to the outback dbc file
AddrCheckStruct subaru_preglobal_addr_checks[] = {
  {.msg = {{0x140, 0, 8, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{0x371, 0, 8, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x144, 0, 8, .expected_timestep = 50000U}, { 0 }, { 0 }}},
};
#define SUBARU_PG_ADDR_CHECK_LEN (sizeof(subaru_preglobal_addr_checks) / sizeof(subaru_preglobal_addr_checks[0]))
addr_checks subaru_preglobal_rx_checks = {subaru_preglobal_addr_checks, SUBARU_PG_ADDR_CHECK_LEN};

static int subaru_preglobal_rx_hook(CANPacket_t *to_push) {

  bool valid = addr_safety_check(to_push, &subaru_preglobal_rx_checks, NULL, NULL, NULL, NULL);

  if (valid && (GET_BUS(to_push) == 0U)) {
    int addr = GET_ADDR(to_push);
    if (addr == 0x371) {
      int torque_driver_new;
      torque_driver_new = (GET_BYTE(to_push, 3) >> 5) + (GET_BYTE(to_push, 4) << 3);
      torque_driver_new = to_signed(torque_driver_new, 11);
      update_sample(&torque_driver, torque_driver_new);
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    if (addr == 0x144) {
      bool cruise_engaged = GET_BIT(to_push, 49U) != 0U;
      pcm_cruise_check(cruise_engaged);
    }

    // update vehicle moving with any non-zero wheel speed
    if (addr == 0xD4) {
      vehicle_moving = ((GET_BYTES(to_push, 0, 4) >> 12) != 0U) || (GET_BYTES(to_push, 4, 4) != 0U);
    }

    if (addr == 0xD1) {
      brake_pressed = ((GET_BYTES(to_push, 0, 4) >> 16) & 0xFFU) > 0U;
    }

    if (addr == 0x140) {
      gas_pressed = GET_BYTE(to_push, 0) != 0U;
    }

    generic_rx_checks((addr == 0x164));
  }
  return valid;
}

static int subaru_preglobal_tx_hook(CANPacket_t *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);

  if (!msg_allowed(to_send, SUBARU_PG_TX_MSGS, SUBARU_PG_TX_MSGS_LEN)) {
    tx = 0;
  }

  // steer cmd checks
  if (addr == 0x164) {
    int desired_torque = ((GET_BYTES(to_send, 0, 4) >> 8) & 0x1FFFU);
    desired_torque = -1 * to_signed(desired_torque, 13);

    if (steer_torque_cmd_checks(desired_torque, -1, SUBARU_PG_STEERING_LIMITS)) {
      tx = 0;
    }

  }
  return tx;
}

static int subaru_preglobal_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;

  if (bus_num == 0) {
    bus_fwd = 2;  // Camera CAN
  }

  if (bus_num == 2) {
    // Preglobal platform
    // 0x161 is ES_CruiseThrottle
    // 0x164 is ES_LKAS
    int block_msg = ((addr == 0x161) || (addr == 0x164));
    if (!block_msg) {
      bus_fwd = 0;  // Main CAN
    }
  }

  return bus_fwd;
}

static const addr_checks* subaru_preglobal_init(uint16_t param) {
  UNUSED(param);
  return &subaru_preglobal_rx_checks;
}

const safety_hooks subaru_preglobal_hooks = {
  .init = subaru_preglobal_init,
  .rx = subaru_preglobal_rx_hook,
  .tx = subaru_preglobal_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = subaru_preglobal_fwd_hook,
};
