const int SUBARU_MAX_STEER = 2047;
// real time torque limit to prevent controls spamming
// the real time limit is 1500/sec
const int SUBARU_MAX_RT_DELTA = 940;          // max delta torque allowed for real time checks
const uint32_t SUBARU_RT_INTERVAL = 250000;    // 250ms between real time checks
const int SUBARU_MAX_RATE_UP = 50;
const int SUBARU_MAX_RATE_DOWN = 70;
const int SUBARU_DRIVER_TORQUE_ALLOWANCE = 60;
const int SUBARU_DRIVER_TORQUE_FACTOR = 50;
const int SUBARU_STANDSTILL_THRSLD = 20;  // about 1kph

const int SUBARU_GEN2_MAX_STEER = 1000;
const int SUBARU_GEN2_MAX_RATE_UP = 40;
const int SUBARU_GEN2_MAX_RATE_DOWN = 40;

const CanMsg SUBARU_TX_MSGS[] = {
  {0x122, 0, 8},
  {0x221, 0, 8},
  {0x321, 0, 8},
  {0x322, 0, 8}
};
#define SUBARU_TX_MSGS_LEN (sizeof(SUBARU_TX_MSGS) / sizeof(SUBARU_TX_MSGS[0]))

const CanMsg SUBARU_GEN2_TX_MSGS[] = {
  {0x122, 0, 8},
  {0x221, 1, 8},
  {0x321, 0, 8},
  {0x322, 0, 8}
};
#define SUBARU_GEN2_TX_MSGS_LEN (sizeof(SUBARU_GEN2_TX_MSGS) / sizeof(SUBARU_GEN2_TX_MSGS[0]))

AddrCheckStruct subaru_addr_checks[] = {
  {.msg = {{ 0x40, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{0x119, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x13a, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x13c, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x240, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 50000U}, { 0 }, { 0 }}},
};
#define SUBARU_ADDR_CHECK_LEN (sizeof(subaru_addr_checks) / sizeof(subaru_addr_checks[0]))
addr_checks subaru_rx_checks = {subaru_addr_checks, SUBARU_ADDR_CHECK_LEN};

AddrCheckStruct subaru_gen2_addr_checks[] = {
  {.msg = {{ 0x40, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 10000U}, { 0 }, { 0 }}},
  {.msg = {{0x119, 0, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x13a, 1, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x13c, 1, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x240, 1, 8, .check_checksum = true, .max_counter = 15U, .expected_timestep = 50000U}, { 0 }, { 0 }}},
};
#define SUBARU_GEN2_ADDR_CHECK_LEN (sizeof(subaru_gen2_addr_checks) / sizeof(subaru_gen2_addr_checks[0]))
addr_checks subaru_gen2_rx_checks = {subaru_gen2_addr_checks, SUBARU_GEN2_ADDR_CHECK_LEN};


const uint16_t SUBARU_PARAM_GEN2 = 1;
bool subaru_gen2 = false;


static uint32_t subaru_get_checksum(CANPacket_t *to_push) {
  return (uint8_t)GET_BYTE(to_push, 0);
}

static uint8_t subaru_get_counter(CANPacket_t *to_push) {
  return (uint8_t)(GET_BYTE(to_push, 1) & 0xFU);
}

static uint32_t subaru_compute_checksum(CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);
  int len = GET_LEN(to_push);
  uint8_t checksum = (uint8_t)(addr) + (uint8_t)((unsigned int)(addr) >> 8U);
  for (int i = 1; i < len; i++) {
    checksum += (uint8_t)GET_BYTE(to_push, i);
  }
  return checksum;
}

static int subaru_rx_hook(CANPacket_t *to_push) {

  bool valid = addr_safety_check(to_push, &subaru_rx_checks,
                                 subaru_get_checksum, subaru_compute_checksum, subaru_get_counter);

  if (valid) {
    const int bus = GET_BUS(to_push);
    const int alt_bus = subaru_gen2 ? 1 : 0;

    int addr = GET_ADDR(to_push);
    if ((addr == 0x119) && (bus == 0)) {
      int torque_driver_new;
      torque_driver_new = ((GET_BYTES_04(to_push) >> 16) & 0x7FFU);
      torque_driver_new = -1 * to_signed(torque_driver_new, 11);
      update_sample(&torque_driver, torque_driver_new);
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    if ((addr == 0x240) && (bus == alt_bus)) {
      int cruise_engaged = ((GET_BYTES_48(to_push) >> 9) & 1U);
      if (cruise_engaged && !cruise_engaged_prev) {
        controls_allowed = 1;
      }
      if (!cruise_engaged) {
        controls_allowed = 0;
      }
      cruise_engaged_prev = cruise_engaged;
    }

    // sample wheel speed, averaging opposite corners
    if ((addr == 0x13a) && (bus == alt_bus)) {
      int subaru_speed = ((GET_BYTES_04(to_push) >> 12) & 0x1FFFU) + ((GET_BYTES_48(to_push) >> 6) & 0x1FFFU);  // FR + RL
      subaru_speed /= 2;
      vehicle_moving = subaru_speed > SUBARU_STANDSTILL_THRSLD;
    }

    if ((addr == 0x13c) && (bus == alt_bus)) {
      brake_pressed = ((GET_BYTE(to_push, 7) >> 6) & 1U);
    }

    if ((addr == 0x40) && (bus == 0)) {
      gas_pressed = GET_BYTE(to_push, 4) != 0U;
    }

    generic_rx_checks((addr == 0x122) && (bus == 0));
  }
  return valid;
}

static int subaru_tx_hook(CANPacket_t *to_send, bool longitudinal_allowed) {
  UNUSED(longitudinal_allowed);

  int tx = 1;
  int addr = GET_ADDR(to_send);

  if (subaru_gen2) {
    tx = msg_allowed(to_send, SUBARU_GEN2_TX_MSGS, SUBARU_GEN2_TX_MSGS_LEN);
  } else {
    tx = msg_allowed(to_send, SUBARU_TX_MSGS, SUBARU_TX_MSGS_LEN);
  }

  // steer cmd checks
  if (addr == 0x122) {
    int desired_torque = ((GET_BYTES_04(to_send) >> 16) & 0x1FFFU);
    bool violation = 0;
    uint32_t ts = microsecond_timer_get();

    desired_torque = -1 * to_signed(desired_torque, 13);

    if (controls_allowed) {
      const int max_steer = subaru_gen2 ? SUBARU_GEN2_MAX_STEER : SUBARU_MAX_STEER;
      const int max_rate_up = subaru_gen2 ? SUBARU_GEN2_MAX_RATE_UP : SUBARU_MAX_RATE_UP;
      const int max_rate_down = subaru_gen2 ? SUBARU_GEN2_MAX_RATE_DOWN : SUBARU_MAX_RATE_DOWN;

      // *** global torque limit check ***
      violation |= max_limit_check(desired_torque, max_steer, -max_steer);

      // *** torque rate limit check ***
      violation |= driver_limit_check(desired_torque, desired_torque_last, &torque_driver,
        max_steer, max_rate_up, max_rate_down,
        SUBARU_DRIVER_TORQUE_ALLOWANCE, SUBARU_DRIVER_TORQUE_FACTOR);

      // used next time
      desired_torque_last = desired_torque;

      // *** torque real time rate limit check ***
      violation |= rt_rate_limit_check(desired_torque, rt_torque_last, SUBARU_MAX_RT_DELTA);

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, ts_last);
      if (ts_elapsed > SUBARU_RT_INTERVAL) {
        rt_torque_last = desired_torque;
        ts_last = ts;
      }
    }

    // no torque if controls is not allowed
    if (!controls_allowed && (desired_torque != 0)) {
      violation = 1;
    }

    // reset to 0 if either controls is not allowed or there's a violation
    if (violation || !controls_allowed) {
      desired_torque_last = 0;
      rt_torque_last = 0;
      ts_last = ts;
    }

    if (violation) {
      tx = 0;
    }

  }
  return tx;
}

static int subaru_fwd_hook(int bus_num, CANPacket_t *to_fwd) {
  int bus_fwd = -1;

  if (bus_num == 0) {
    bus_fwd = 2;  // forward to camera
  }

  if (bus_num == 2) {
    // Global platform
    // 0x122 ES_LKAS
    // 0x321 ES_DashStatus
    // 0x322 ES_LKAS_State
    int addr = GET_ADDR(to_fwd);
    bool block_lkas = (addr == 0x122) || (addr == 0x321) || (addr == 0x322);
    if (!block_lkas) {
      bus_fwd = 0;  // Main CAN
    }
  }

  return bus_fwd;
}

static const addr_checks* subaru_init(uint16_t param) {
  subaru_gen2 = GET_FLAG(param, SUBARU_PARAM_GEN2);

  if (subaru_gen2) {
    subaru_rx_checks = (addr_checks){subaru_gen2_addr_checks, SUBARU_GEN2_ADDR_CHECK_LEN};
  } else {
    subaru_rx_checks = (addr_checks){subaru_addr_checks, SUBARU_ADDR_CHECK_LEN};
  }

  return &subaru_rx_checks;
}

const safety_hooks subaru_hooks = {
  .init = subaru_init,
  .rx = subaru_rx_hook,
  .tx = subaru_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = subaru_fwd_hook,
};


