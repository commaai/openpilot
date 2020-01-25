// global torque limit
const int TOYOTA_MAX_TORQUE = 1500;       // max torque cmd allowed ever

// rate based torque limit + stay within actually applied
// packet is sent at 100hz, so this limit is 1000/sec
const int TOYOTA_MAX_RATE_UP = 10;        // ramp up slow
const int TOYOTA_MAX_RATE_DOWN = 25;      // ramp down fast
const int TOYOTA_MAX_TORQUE_ERROR = 350;  // max torque cmd in excess of torque motor

// real time torque limit to prevent controls spamming
// the real time limit is 1500/sec
const int TOYOTA_MAX_RT_DELTA = 375;      // max delta torque allowed for real time checks
const uint32_t TOYOTA_RT_INTERVAL = 250000;    // 250ms between real time checks

// longitudinal limits
const int TOYOTA_MAX_ACCEL = 1500;        // 1.5 m/s2
const int TOYOTA_MIN_ACCEL = -3000;       // 3.0 m/s2

const int TOYOTA_GAS_INTERCEPTOR_THRESHOLD = 475;  // ratio between offset and gain from dbc file

const AddrBus TOYOTA_TX_MSGS[] = {{0x283, 0}, {0x2E6, 0}, {0x2E7, 0}, {0x33E, 0}, {0x344, 0}, {0x365, 0}, {0x366, 0}, {0x4CB, 0},  // DSU bus 0
                                  {0x128, 1}, {0x141, 1}, {0x160, 1}, {0x161, 1}, {0x470, 1},  // DSU bus 1
                                  {0x2E4, 0}, {0x411, 0}, {0x412, 0}, {0x343, 0}, {0x1D2, 0},  // LKAS + ACC
                                  {0x200, 0}};  // interceptor

AddrCheckStruct toyota_rx_checks[] = {
  {.addr = {0x260}, .bus = 0, .check_checksum = true, .max_counter = 0U, .expected_timestep = 20000U},
  {.addr = {0x1D2}, .bus = 0, .check_checksum = true, .max_counter = 0U, .expected_timestep = 30000U},
};
const int TOYOTA_RX_CHECKS_LEN = sizeof(toyota_rx_checks) / sizeof(toyota_rx_checks[0]);

// global actuation limit states
int toyota_dbc_eps_torque_factor = 100;   // conversion factor for STEER_TORQUE_EPS in %: see dbc file

// states
int toyota_desired_torque_last = 0;       // last desired steer torque
int toyota_rt_torque_last = 0;            // last desired torque for real time check
uint32_t toyota_ts_last = 0;
int toyota_cruise_engaged_last = 0;       // cruise state
int toyota_gas_prev = 0;
struct sample_t toyota_torque_meas;       // last 3 motor torques produced by the eps


static uint8_t toyota_compute_checksum(CAN_FIFOMailBox_TypeDef *to_push) {
  int addr = GET_ADDR(to_push);
  int len = GET_LEN(to_push);
  uint8_t checksum = (uint8_t)(addr) + (uint8_t)((unsigned int)(addr) >> 8U) + (uint8_t)(len);
  for (int i = 0; i < (len - 1); i++) {
    checksum += (uint8_t)GET_BYTE(to_push, i);
  }
  return checksum;
}

static uint8_t toyota_get_checksum(CAN_FIFOMailBox_TypeDef *to_push) {
  int checksum_byte = GET_LEN(to_push) - 1;
  return (uint8_t)(GET_BYTE(to_push, checksum_byte));
}

static int toyota_rx_hook(CAN_FIFOMailBox_TypeDef *to_push) {

  bool valid = addr_safety_check(to_push, toyota_rx_checks, TOYOTA_RX_CHECKS_LEN,
                                 toyota_get_checksum, toyota_compute_checksum, NULL);
  if (valid) {
    int bus = GET_BUS(to_push);
    int addr = GET_ADDR(to_push);

    // get eps motor torque (0.66 factor in dbc)
    if (addr == 0x260) {
      int torque_meas_new = (GET_BYTE(to_push, 5) << 8) | GET_BYTE(to_push, 6);
      torque_meas_new = to_signed(torque_meas_new, 16);

      // scale by dbc_factor
      torque_meas_new = (torque_meas_new * toyota_dbc_eps_torque_factor) / 100;

      // update array of sample
      update_sample(&toyota_torque_meas, torque_meas_new);

      // increase torque_meas by 1 to be conservative on rounding
      toyota_torque_meas.min--;
      toyota_torque_meas.max++;
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    if (addr == 0x1D2) {
      // 5th bit is CRUISE_ACTIVE
      int cruise_engaged = GET_BYTE(to_push, 0) & 0x20;
      if (!cruise_engaged) {
        controls_allowed = 0;
      }
      if (cruise_engaged && !toyota_cruise_engaged_last) {
        controls_allowed = 1;
      }
      toyota_cruise_engaged_last = cruise_engaged;
    }

    // exit controls on rising edge of interceptor gas press
    if (addr == 0x201) {
      gas_interceptor_detected = 1;
      int gas_interceptor = GET_INTERCEPTOR(to_push);
      if ((gas_interceptor > TOYOTA_GAS_INTERCEPTOR_THRESHOLD) &&
          (gas_interceptor_prev <= TOYOTA_GAS_INTERCEPTOR_THRESHOLD)) {
        controls_allowed = 0;
      }
      gas_interceptor_prev = gas_interceptor;
    }

    // exit controls on rising edge of gas press
    if (addr == 0x2C1) {
      int gas = GET_BYTE(to_push, 6) & 0xFF;
      if ((gas > 0) && (toyota_gas_prev == 0) && !gas_interceptor_detected) {
        controls_allowed = 0;
      }
      toyota_gas_prev = gas;
    }

    // 0x2E4 is lkas cmd. If it is on bus 0, then relay is unexpectedly closed
    if ((safety_mode_cnt > RELAY_TRNS_TIMEOUT) && (addr == 0x2E4) && (bus == 0)) {
      relay_malfunction = true;
    }
  }
  return valid;
}

static int toyota_tx_hook(CAN_FIFOMailBox_TypeDef *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  if (!msg_allowed(addr, bus, TOYOTA_TX_MSGS, sizeof(TOYOTA_TX_MSGS)/sizeof(TOYOTA_TX_MSGS[0]))) {
    tx = 0;
  }

  if (relay_malfunction) {
    tx = 0;
  }

  // Check if msg is sent on BUS 0
  if (bus == 0) {

    // GAS PEDAL: safety check
    if (addr == 0x200) {
      if (!controls_allowed) {
        if (GET_BYTE(to_send, 0) || GET_BYTE(to_send, 1)) {
          tx = 0;
        }
      }
    }

    // ACCEL: safety check on byte 1-2
    if (addr == 0x343) {
      int desired_accel = (GET_BYTE(to_send, 0) << 8) | GET_BYTE(to_send, 1);
      desired_accel = to_signed(desired_accel, 16);
      if (!controls_allowed) {
        if (desired_accel != 0) {
          tx = 0;
        }
      }
      bool violation = max_limit_check(desired_accel, TOYOTA_MAX_ACCEL, TOYOTA_MIN_ACCEL);
      if (violation) {
        tx = 0;
      }
    }

    // STEER: safety check on bytes 2-3
    if (addr == 0x2E4) {
      int desired_torque = (GET_BYTE(to_send, 1) << 8) | GET_BYTE(to_send, 2);
      desired_torque = to_signed(desired_torque, 16);
      bool violation = 0;

      uint32_t ts = TIM2->CNT;

      if (controls_allowed) {

        // *** global torque limit check ***
        violation |= max_limit_check(desired_torque, TOYOTA_MAX_TORQUE, -TOYOTA_MAX_TORQUE);

        // *** torque rate limit check ***
        violation |= dist_to_meas_check(desired_torque, toyota_desired_torque_last,
          &toyota_torque_meas, TOYOTA_MAX_RATE_UP, TOYOTA_MAX_RATE_DOWN, TOYOTA_MAX_TORQUE_ERROR);

        // used next time
        toyota_desired_torque_last = desired_torque;

        // *** torque real time rate limit check ***
        violation |= rt_rate_limit_check(desired_torque, toyota_rt_torque_last, TOYOTA_MAX_RT_DELTA);

        // every RT_INTERVAL set the new limits
        uint32_t ts_elapsed = get_ts_elapsed(ts, toyota_ts_last);
        if (ts_elapsed > TOYOTA_RT_INTERVAL) {
          toyota_rt_torque_last = desired_torque;
          toyota_ts_last = ts;
        }
      }

      // no torque if controls is not allowed
      if (!controls_allowed && (desired_torque != 0)) {
        violation = 1;
      }

      // reset to 0 if either controls is not allowed or there's a violation
      if (violation || !controls_allowed) {
        toyota_desired_torque_last = 0;
        toyota_rt_torque_last = 0;
        toyota_ts_last = ts;
      }

      if (violation) {
        tx = 0;
      }
    }
  }

  return tx;
}

static void toyota_init(int16_t param) {
  controls_allowed = 0;
  relay_malfunction = 0;
  toyota_dbc_eps_torque_factor = param;
}

static int toyota_fwd_hook(int bus_num, CAN_FIFOMailBox_TypeDef *to_fwd) {

  int bus_fwd = -1;
  if (!relay_malfunction) {
    if (bus_num == 0) {
      bus_fwd = 2;
    }
    if (bus_num == 2) {
      int addr = GET_ADDR(to_fwd);
      // block stock lkas messages and stock acc messages (if OP is doing ACC)
      // in TSS2, 0x191 is LTA which we need to block to avoid controls collision
      int is_lkas_msg = ((addr == 0x2E4) || (addr == 0x412) || (addr == 0x191));
      // in TSS2 the camera does ACC as well, so filter 0x343
      int is_acc_msg = (addr == 0x343);
      int block_msg = is_lkas_msg || is_acc_msg;
      if (!block_msg) {
        bus_fwd = 0;
      }
    }
  }
  return bus_fwd;
}

const safety_hooks toyota_hooks = {
  .init = toyota_init,
  .rx = toyota_rx_hook,
  .tx = toyota_tx_hook,
  .tx_lin = nooutput_tx_lin_hook,
  .fwd = toyota_fwd_hook,
  .addr_check = toyota_rx_checks,
  .addr_check_len = sizeof(toyota_rx_checks)/sizeof(toyota_rx_checks[0]),
};
