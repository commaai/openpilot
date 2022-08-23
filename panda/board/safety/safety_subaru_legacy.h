const SteeringLimits SUBARU_L_STEERING_LIMITS = {
  .max_steer = 2047,
  .max_rt_delta = 940,
  .max_rt_interval = 250000,
  .max_rate_up = 50,
  .max_rate_down = 70,
  .driver_torque_factor = 10,
  .driver_torque_allowance = 75,
  .type = TorqueDriverLimited,
};

const CanMsg SUBARU_L_TX_MSGS[] = {
  {0x161, 0, 8},
  {0x164, 0, 8}
};
#define SUBARU_L_TX_MSGS_LEN (sizeof(SUBARU_L_TX_MSGS) / sizeof(SUBARU_L_TX_MSGS[0]))

// TODO: do checksum and counter checks after adding the signals to the outback dbc file
AddrCheckStruct subaru_l_addr_checks[] = {
  {.msg = {{0x140, 0, 8, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{0x371, 0, 8, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x144, 0, 8, .expected_timestep = 50000U}, { 0 }, { 0 }}},
};
#define SUBARU_L_ADDR_CHECK_LEN (sizeof(subaru_l_addr_checks) / sizeof(subaru_l_addr_checks[0]))
addr_checks subaru_l_rx_checks = {subaru_l_addr_checks, SUBARU_L_ADDR_CHECK_LEN};

static int subaru_legacy_rx_hook(CANPacket_t *to_push) {

  bool valid = addr_safety_check(to_push, &subaru_l_rx_checks, NULL, NULL, NULL);

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

    // sample wheel speed, averaging opposite corners
    if (addr == 0xD4) {
      int subaru_speed = ((GET_BYTES_04(to_push) >> 16) & 0xFFFFU) + (GET_BYTES_48(to_push) & 0xFFFFU);  // FR + RL
      subaru_speed /= 2;
      vehicle_moving = subaru_speed > SUBARU_STANDSTILL_THRSLD;
    }

    if (addr == 0xD1) {
      brake_pressed = ((GET_BYTES_04(to_push) >> 16) & 0xFFU) > 0U;
    }

    if (addr == 0x140) {
      gas_pressed = GET_BYTE(to_push, 0) != 0U;
    }

    generic_rx_checks((addr == 0x164));
  }
  return valid;
}

static int subaru_legacy_tx_hook(CANPacket_t *to_send, bool longitudinal_allowed) {
  UNUSED(longitudinal_allowed);

  int tx = 1;
  int addr = GET_ADDR(to_send);

  if (!msg_allowed(to_send, SUBARU_L_TX_MSGS, SUBARU_L_TX_MSGS_LEN)) {
    tx = 0;
  }

  // steer cmd checks
  if (addr == 0x164) {
    int desired_torque = ((GET_BYTES_04(to_send) >> 8) & 0x1FFFU);
    desired_torque = -1 * to_signed(desired_torque, 13);

    if (steer_torque_cmd_checks(desired_torque, -1, SUBARU_L_STEERING_LIMITS)) {
      tx = 0;
    }

  }
  return tx;
}

static int subaru_legacy_fwd_hook(int bus_num, CANPacket_t *to_fwd) {
  int bus_fwd = -1;

  if (bus_num == 0) {
    bus_fwd = 2;  // Camera CAN
  }

  if (bus_num == 2) {
    // Preglobal platform
    // 0x161 is ES_CruiseThrottle
    // 0x164 is ES_LKAS
    int addr = GET_ADDR(to_fwd);
    int block_msg = ((addr == 0x161) || (addr == 0x164));
    if (!block_msg) {
      bus_fwd = 0;  // Main CAN
    }
  }

  return bus_fwd;
}

static const addr_checks* subaru_legacy_init(uint16_t param) {
  UNUSED(param);
  return &subaru_l_rx_checks;
}

const safety_hooks subaru_legacy_hooks = {
  .init = subaru_legacy_init,
  .rx = subaru_legacy_rx_hook,
  .tx = subaru_legacy_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = subaru_legacy_fwd_hook,
};
