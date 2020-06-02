const int CHRYSLER_MAX_STEER = 261;
const int CHRYSLER_MAX_RT_DELTA = 112;        // max delta torque allowed for real time checks
const uint32_t CHRYSLER_RT_INTERVAL = 250000;  // 250ms between real time checks
const int CHRYSLER_MAX_RATE_UP = 3;
const int CHRYSLER_MAX_RATE_DOWN = 3;
const int CHRYSLER_MAX_TORQUE_ERROR = 80;    // max torque cmd in excess of torque motor
const int CHRYSLER_GAS_THRSLD = 30;  // 7% more than 2m/s
const int CHRYSLER_STANDSTILL_THRSLD = 10;  // about 1m/s
const AddrBus CHRYSLER_TX_MSGS[] = {{571, 0}, {658, 0}, {678, 0}};

// TODO: do checksum and counter checks
AddrCheckStruct chrysler_rx_checks[] = {
  {.addr = {544}, .bus = 0, .check_checksum = true, .max_counter = 15U, .expected_timestep = 10000U},
  {.addr = {514}, .bus = 0, .check_checksum = false, .max_counter = 0U, .expected_timestep = 10000U},
  {.addr = {500}, .bus = 0, .check_checksum = true, .max_counter = 15U, .expected_timestep = 20000U},
  {.addr = {308}, .bus = 0, .check_checksum = false, .max_counter = 15U,  .expected_timestep = 20000U},
  {.addr = {320}, .bus = 0, .check_checksum = true, .max_counter = 15U,  .expected_timestep = 20000U},
};
const int CHRYSLER_RX_CHECK_LEN = sizeof(chrysler_rx_checks) / sizeof(chrysler_rx_checks[0]);

int chrysler_speed = 0;

static uint8_t chrysler_get_checksum(CAN_FIFOMailBox_TypeDef *to_push) {
  int checksum_byte = GET_LEN(to_push) - 1;
  return (uint8_t)(GET_BYTE(to_push, checksum_byte));
}

static uint8_t chrysler_compute_checksum(CAN_FIFOMailBox_TypeDef *to_push) {
  /* This function does not want the checksum byte in the input data.
  jeep chrysler canbus checksum from http://illmatics.com/Remote%20Car%20Hacking.pdf */
  uint8_t checksum = 0xFF;
  int len = GET_LEN(to_push);
  for (int j = 0; j < (len - 1); j++) {
    uint8_t shift = 0x80;
    uint8_t curr = (uint8_t)GET_BYTE(to_push, j);
    for (int i=0; i<8; i++) {
      uint8_t bit_sum = curr & shift;
      uint8_t temp_chk = checksum & 0x80U;
      if (bit_sum != 0U) {
        bit_sum = 0x1C;
        if (temp_chk != 0U) {
          bit_sum = 1;
        }
        checksum = checksum << 1;
        temp_chk = checksum | 1U;
        bit_sum ^= temp_chk;
      } else {
        if (temp_chk != 0U) {
          bit_sum = 0x1D;
        }
        checksum = checksum << 1;
        bit_sum ^= checksum;
      }
      checksum = bit_sum;
      shift = shift >> 1;
    }
  }
  return ~checksum;
}

static uint8_t chrysler_get_counter(CAN_FIFOMailBox_TypeDef *to_push) {
  // Well defined counter only for 8 bytes messages
  return (uint8_t)(GET_BYTE(to_push, 6) >> 4);
}

static int chrysler_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {

  bool valid = addr_safety_check(to_push, chrysler_rx_checks, CHRYSLER_RX_CHECK_LEN,
                                 chrysler_get_checksum, chrysler_compute_checksum,
                                 chrysler_get_counter);

  bool unsafe_allow_gas = unsafe_mode & UNSAFE_DISABLE_DISENGAGE_ON_GAS;

  if (valid && (GET_BUS(to_push) == 0)) {
    int addr = GET_ADDR(to_push);

    // Measured eps torque
    if (addr == 544) {
      int torque_meas_new = ((GET_BYTE(to_push, 4) & 0x7U) << 8) + GET_BYTE(to_push, 5) - 1024U;

      // update array of samples
      update_sample(&torque_meas, torque_meas_new);
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    if (addr == 500) {
      int cruise_engaged = ((GET_BYTE(to_push, 2) & 0x38) >> 3) == 7;
      if (cruise_engaged && !cruise_engaged_prev) {
        controls_allowed = 1;
      }
      if (!cruise_engaged) {
        controls_allowed = 0;
      }
      cruise_engaged_prev = cruise_engaged;
    }

    // update speed
    if (addr == 514) {
      int speed_l = (GET_BYTE(to_push, 0) << 4) + (GET_BYTE(to_push, 1) >> 4);
      int speed_r = (GET_BYTE(to_push, 2) << 4) + (GET_BYTE(to_push, 3) >> 4);
      chrysler_speed = (speed_l + speed_r) / 2;
      vehicle_moving = chrysler_speed > CHRYSLER_STANDSTILL_THRSLD;
    }

    // exit controls on rising edge of gas press
    if (addr == 308) {
      bool gas_pressed = (GET_BYTE(to_push, 5) & 0x7F) != 0;
      if (!unsafe_allow_gas && gas_pressed && !gas_pressed_prev && (chrysler_speed > CHRYSLER_GAS_THRSLD)) {
        controls_allowed = 0;
      }
      gas_pressed_prev = gas_pressed;
    }

    // exit controls on rising edge of brake press
    if (addr == 320) {
      bool brake_pressed = (GET_BYTE(to_push, 0) & 0x7) == 5;
      if (brake_pressed && (!brake_pressed_prev || vehicle_moving)) {
        controls_allowed = 0;
      }
      brake_pressed_prev = brake_pressed;
    }

    // check if stock camera ECU is on bus 0
    if ((safety_mode_cnt > RELAY_TRNS_TIMEOUT) && (addr == 0x292)) {
      relay_malfunction_set();
    }
  }
  return valid;
}

static int chrysler_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  if (!msg_allowed(addr, bus, CHRYSLER_TX_MSGS, sizeof(CHRYSLER_TX_MSGS) / sizeof(CHRYSLER_TX_MSGS[0]))) {
    tx = 0;
  }

  if (relay_malfunction) {
    tx = 0;
  }

  // LKA STEER
  if (addr == 0x292) {
    int desired_torque = ((GET_BYTE(to_send, 0) & 0x7U) << 8) + GET_BYTE(to_send, 1) - 1024U;
    uint32_t ts = TIM2->CNT;
    bool violation = 0;

    if (controls_allowed) {

      // *** global torque limit check ***
      violation |= max_limit_check(desired_torque, CHRYSLER_MAX_STEER, -CHRYSLER_MAX_STEER);

      // *** torque rate limit check ***
      violation |= dist_to_meas_check(desired_torque, desired_torque_last,
        &torque_meas, CHRYSLER_MAX_RATE_UP, CHRYSLER_MAX_RATE_DOWN, CHRYSLER_MAX_TORQUE_ERROR);

      // used next time
      desired_torque_last = desired_torque;

      // *** torque real time rate limit check ***
      violation |= rt_rate_limit_check(desired_torque, rt_torque_last, CHRYSLER_MAX_RT_DELTA);

      // every RT_INTERVAL set the new limits
      uint32_t ts_elapsed = get_ts_elapsed(ts, ts_last);
      if (ts_elapsed > CHRYSLER_RT_INTERVAL) {
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

  // FORCE CANCEL: only the cancel button press is allowed
  if (addr == 571) {
    if ((GET_BYTE(to_send, 0) != 1) || ((GET_BYTE(to_send, 1) & 1) == 1)) {
      tx = 0;
    }
  }

  return tx;
}

static int chrysler_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {

  int bus_fwd = -1;
  int addr = GET_ADDR(to_fwd);

  if (!relay_malfunction) {
    // forward CAN 0 -> 2 so stock LKAS camera sees messages
    if (bus_num == 0) {
      bus_fwd = 2;
    }
    // forward all messages from camera except LKAS_COMMAND and LKAS_HUD
    if ((bus_num == 2) && (addr != 658) && (addr != 678)) {
      bus_fwd = 0;
    }
  }
  return bus_fwd;
}


const safety_hooks chrysler_hooks = {
  .init = nooutput_init,
  .rx = chrysler_rx_hook,
  .tx = chrysler_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = chrysler_fwd_hook,
  .addr_check = chrysler_rx_checks,
  .addr_check_len = sizeof(chrysler_rx_checks) / sizeof(chrysler_rx_checks[0]),
};
