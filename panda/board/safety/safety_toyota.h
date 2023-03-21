const SteeringLimits TOYOTA_STEERING_LIMITS = {
  .max_steer = 1500,
  .max_rate_up = 15,          // ramp up slow
  .max_rate_down = 25,        // ramp down fast
  .max_torque_error = 350,    // max torque cmd in excess of motor torque
  .max_rt_delta = 450,        // the real time limit is 1800/sec, a 20% buffer
  .max_rt_interval = 250000,
  .type = TorqueMotorLimited,

  // the EPS faults when the steering angle rate is above a certain threshold for too long. to prevent this,
  // we allow setting STEER_REQUEST bit to 0 while maintaining the requested torque value for a single frame
  .min_valid_request_frames = 18,
  .max_invalid_request_frames = 1,
  .min_valid_request_rt_interval = 170000,  // 170ms; a ~10% buffer on cutting every 19 frames
  .has_steer_req_tolerance = true,
};

// longitudinal limits
const LongitudinalLimits TOYOTA_LONG_LIMITS = {
  .max_accel = 2000,   // 2.0 m/s2
  .min_accel = -3500,  // -3.5 m/s2
};

// panda interceptor threshold needs to be equivalent to openpilot threshold to avoid controls mismatches
// If thresholds are mismatched then it is possible for panda to see the gas fall and rise while openpilot is in the pre-enabled state
// Threshold calculated from DBC gains: round((((15 + 75.555) / 0.159375) + ((15 + 151.111) / 0.159375)) / 2) = 805
const int TOYOTA_GAS_INTERCEPTOR_THRSLD = 805;
#define TOYOTA_GET_INTERCEPTOR(msg) (((GET_BYTE((msg), 0) << 8) + GET_BYTE((msg), 1) + (GET_BYTE((msg), 2) << 8) + GET_BYTE((msg), 3)) / 2U) // avg between 2 tracks

const CanMsg TOYOTA_TX_MSGS[] = {{0x283, 0, 7}, {0x2E6, 0, 8}, {0x2E7, 0, 8}, {0x33E, 0, 7}, {0x344, 0, 8}, {0x365, 0, 7}, {0x366, 0, 7}, {0x4CB, 0, 8},  // DSU bus 0
                                 {0x128, 1, 6}, {0x141, 1, 4}, {0x160, 1, 8}, {0x161, 1, 7}, {0x470, 1, 4},  // DSU bus 1
                                 {0x2E4, 0, 5}, {0x191, 0, 8}, {0x411, 0, 8}, {0x412, 0, 8}, {0x343, 0, 8}, {0x1D2, 0, 8},  // LKAS + ACC
                                 {0x200, 0, 6}};  // interceptor

AddrCheckStruct toyota_addr_checks[] = {
  {.msg = {{ 0xaa, 0, 8, .check_checksum = false, .expected_timestep = 12000U}, { 0 }, { 0 }}},
  {.msg = {{0x260, 0, 8, .check_checksum = true, .expected_timestep = 20000U}, { 0 }, { 0 }}},
  {.msg = {{0x1D2, 0, 8, .check_checksum = true, .expected_timestep = 30000U}, { 0 }, { 0 }}},
  {.msg = {{0x224, 0, 8, .check_checksum = false, .expected_timestep = 25000U},
           {0x226, 0, 8, .check_checksum = false, .expected_timestep = 25000U}, { 0 }}},
};
#define TOYOTA_ADDR_CHECKS_LEN (sizeof(toyota_addr_checks) / sizeof(toyota_addr_checks[0]))
addr_checks toyota_rx_checks = {toyota_addr_checks, TOYOTA_ADDR_CHECKS_LEN};

// safety param flags
// first byte is for eps factor, second is for flags
const uint32_t TOYOTA_PARAM_OFFSET = 8U;
const uint32_t TOYOTA_EPS_FACTOR = (1U << TOYOTA_PARAM_OFFSET) - 1U;
const uint32_t TOYOTA_PARAM_ALT_BRAKE = 1U << TOYOTA_PARAM_OFFSET;
const uint32_t TOYOTA_PARAM_STOCK_LONGITUDINAL = 2U << TOYOTA_PARAM_OFFSET;
const uint32_t TOYOTA_PARAM_LTA = 4U << TOYOTA_PARAM_OFFSET;

bool toyota_alt_brake = false;
bool toyota_stock_longitudinal = false;
bool toyota_lta = false;
int toyota_dbc_eps_torque_factor = 100;   // conversion factor for STEER_TORQUE_EPS in %: see dbc file

static uint32_t toyota_compute_checksum(CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);
  int len = GET_LEN(to_push);
  uint8_t checksum = (uint8_t)(addr) + (uint8_t)((unsigned int)(addr) >> 8U) + (uint8_t)(len);
  for (int i = 0; i < (len - 1); i++) {
    checksum += (uint8_t)GET_BYTE(to_push, i);
  }
  return checksum;
}

static uint32_t toyota_get_checksum(CANPacket_t *to_push) {
  int checksum_byte = GET_LEN(to_push) - 1U;
  return (uint8_t)(GET_BYTE(to_push, checksum_byte));
}

static int toyota_rx_hook(CANPacket_t *to_push) {

  bool valid = addr_safety_check(to_push, &toyota_rx_checks,
                                 toyota_get_checksum, toyota_compute_checksum, NULL, NULL);

  if (valid && (GET_BUS(to_push) == 0U)) {
    int addr = GET_ADDR(to_push);

    // get eps motor torque (0.66 factor in dbc)
    if (addr == 0x260) {
      int torque_meas_new = (GET_BYTE(to_push, 5) << 8) | GET_BYTE(to_push, 6);
      torque_meas_new = to_signed(torque_meas_new, 16);

      // scale by dbc_factor
      torque_meas_new = (torque_meas_new * toyota_dbc_eps_torque_factor) / 100;

      // update array of sample
      update_sample(&torque_meas, torque_meas_new);

      // increase torque_meas by 1 to be conservative on rounding
      torque_meas.min--;
      torque_meas.max++;
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    // exit controls on rising edge of gas press
    if (addr == 0x1D2) {
      // 5th bit is CRUISE_ACTIVE
      bool cruise_engaged = GET_BIT(to_push, 5U) != 0U;
      pcm_cruise_check(cruise_engaged);

      // sample gas pedal
      if (!gas_interceptor_detected) {
        gas_pressed = ((GET_BYTE(to_push, 0) >> 4) & 1U) == 0U;
      }
    }

    if (addr == 0xaa) {
      // check that all wheel speeds are at zero value with offset
      bool standstill = (GET_BYTES_04(to_push) == 0x6F1A6F1AU) && (GET_BYTES_48(to_push) == 0x6F1A6F1AU);
      vehicle_moving = !standstill;
    }

    // most cars have brake_pressed on 0x226, corolla and rav4 on 0x224
    if (((addr == 0x224) && toyota_alt_brake) || ((addr == 0x226) && !toyota_alt_brake)) {
      int byte = (addr == 0x224) ? 0 : 4;
      brake_pressed = ((GET_BYTE(to_push, byte) >> 5) & 1U) != 0U;
    }

    // sample gas interceptor
    if (addr == 0x201) {
      gas_interceptor_detected = 1;
      int gas_interceptor = TOYOTA_GET_INTERCEPTOR(to_push);
      gas_pressed = gas_interceptor > TOYOTA_GAS_INTERCEPTOR_THRSLD;

      // TODO: remove this, only left in for gas_interceptor_prev test
      gas_interceptor_prev = gas_interceptor;
    }

    generic_rx_checks((addr == 0x2E4));
  }
  return valid;
}

static int toyota_tx_hook(CANPacket_t *to_send) {

  int tx = 1;
  int addr = GET_ADDR(to_send);
  int bus = GET_BUS(to_send);

  if (!msg_allowed(to_send, TOYOTA_TX_MSGS, sizeof(TOYOTA_TX_MSGS)/sizeof(TOYOTA_TX_MSGS[0]))) {
    tx = 0;
  }

  // Check if msg is sent on BUS 0
  if (bus == 0) {

    // GAS PEDAL: safety check
    if (addr == 0x200) {
      if (longitudinal_interceptor_checks(to_send)) {
        tx = 0;
      }
    }

    // ACCEL: safety check on byte 1-2
    if (addr == 0x343) {
      int desired_accel = (GET_BYTE(to_send, 0) << 8) | GET_BYTE(to_send, 1);
      desired_accel = to_signed(desired_accel, 16);

      bool violation = false;
      violation |= longitudinal_accel_checks(desired_accel, TOYOTA_LONG_LIMITS);

      // only ACC messages that cancel are allowed when openpilot is not controlling longitudinal
      if (toyota_stock_longitudinal) {
        bool cancel_req = GET_BIT(to_send, 24U) != 0U;
        if (!cancel_req) {
          violation = true;
        }
        if (desired_accel != TOYOTA_LONG_LIMITS.inactive_accel) {
          violation = true;
        }
      }

      if (violation) {
        tx = 0;
      }
    }

    // AEB: block all actuation. only used when DSU is unplugged
    if (addr == 0x283) {
      // only allow the checksum, which is the last byte
      bool block = (GET_BYTES_04(to_send) != 0U) || (GET_BYTE(to_send, 4) != 0U) || (GET_BYTE(to_send, 5) != 0U);
      if (block) {
        tx = 0;
      }
    }

    // LTA steering check
    // only sent to prevent dash errors, no actuation is accepted
    if (addr == 0x191) {
      // check the STEER_REQUEST, STEER_REQUEST_2, SETME_X64 STEER_ANGLE_CMD signals
      bool lta_request = (GET_BYTE(to_send, 0) & 1U) != 0U;
      bool lta_request2 = ((GET_BYTE(to_send, 3) >> 1) & 1U) != 0U;
      int setme_x64 = GET_BYTE(to_send, 5);
      int lta_angle = (GET_BYTE(to_send, 1) << 8) | GET_BYTE(to_send, 2);
      lta_angle = to_signed(lta_angle, 16);

      // block LTA msgs with actuation requests
      if (lta_request || lta_request2 || (lta_angle != 0) || (setme_x64 != 0)) {
        tx = 0;
      }
    }

    // STEER: safety check on bytes 2-3
    if (addr == 0x2E4) {
      int desired_torque = (GET_BYTE(to_send, 1) << 8) | GET_BYTE(to_send, 2);
      desired_torque = to_signed(desired_torque, 16);
      bool steer_req = GET_BIT(to_send, 0U) != 0U;
      if (steer_torque_cmd_checks(desired_torque, steer_req, TOYOTA_STEERING_LIMITS)) {
        tx = 0;
      }
      // When using LTA (angle control), assert no actuation on LKA message
      if (toyota_lta && ((desired_torque != 0) || steer_req)) {
        tx = 0;
      }
    }
  }

  return tx;
}

static const addr_checks* toyota_init(uint16_t param) {
  gas_interceptor_detected = 0;
  toyota_alt_brake = GET_FLAG(param, TOYOTA_PARAM_ALT_BRAKE);
  toyota_stock_longitudinal = GET_FLAG(param, TOYOTA_PARAM_STOCK_LONGITUDINAL);
  toyota_dbc_eps_torque_factor = param & TOYOTA_EPS_FACTOR;

#ifdef ALLOW_DEBUG
  toyota_lta = GET_FLAG(param, TOYOTA_PARAM_LTA);
#else
  toyota_lta = false;
#endif
  return &toyota_rx_checks;
}

static int toyota_fwd_hook(int bus_num, CANPacket_t *to_fwd) {

  int bus_fwd = -1;

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
    int block_msg = is_lkas_msg || (is_acc_msg && !toyota_stock_longitudinal);
    if (!block_msg) {
      bus_fwd = 0;
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
};
