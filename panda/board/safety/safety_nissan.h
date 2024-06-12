const SteeringLimits NISSAN_STEERING_LIMITS = {
  .angle_deg_to_can = 100,
  .angle_rate_up_lookup = {
    {0., 5., 15.},
    {5., .8, .15}
  },
  .angle_rate_down_lookup = {
    {0., 5., 15.},
    {5., 3.5, .4}
  },
};

const CanMsg NISSAN_TX_MSGS[] = {
  {0x169, 0, 8},  // LKAS
  {0x2b1, 0, 8},  // PROPILOT_HUD
  {0x4cc, 0, 8},  // PROPILOT_HUD_INFO_MSG
  {0x20b, 2, 6},  // CRUISE_THROTTLE (X-Trail)
  {0x20b, 1, 6},  // CRUISE_THROTTLE (Altima)
  {0x280, 2, 8}   // CANCEL_MSG (Leaf)
};

// Signals duplicated below due to the fact that these messages can come in on either CAN bus, depending on car model.
RxCheck nissan_rx_checks[] = {
  {.msg = {{0x2, 0, 5, .frequency = 100U},
           {0x2, 1, 5, .frequency = 100U}, { 0 }}},  // STEER_ANGLE_SENSOR
  {.msg = {{0x285, 0, 8, .frequency = 50U},
           {0x285, 1, 8, .frequency = 50U}, { 0 }}}, // WHEEL_SPEEDS_REAR
  {.msg = {{0x30f, 2, 3, .frequency = 10U},
           {0x30f, 1, 3, .frequency = 10U}, { 0 }}}, // CRUISE_STATE
  {.msg = {{0x15c, 0, 8, .frequency = 50U},
           {0x15c, 1, 8, .frequency = 50U},
           {0x239, 0, 8, .frequency = 50U}}}, // GAS_PEDAL
  {.msg = {{0x454, 0, 8, .frequency = 10U},
           {0x454, 1, 8, .frequency = 10U},
           {0x1cc, 0, 4, .frequency = 100U}}}, // DOORS_LIGHTS / BRAKE
};

// EPS Location. false = V-CAN, true = C-CAN
const int NISSAN_PARAM_ALT_EPS_BUS = 1;

bool nissan_alt_eps = false;

static void nissan_rx_hook(const CANPacket_t *to_push) {
  int bus = GET_BUS(to_push);
  int addr = GET_ADDR(to_push);

  if (bus == (nissan_alt_eps ? 1 : 0)) {
    if (addr == 0x2) {
      // Current steering angle
      // Factor -0.1, little endian
      int angle_meas_new = (GET_BYTES(to_push, 0, 4) & 0xFFFFU);
      // Multiply by -10 to match scale of LKAS angle
      angle_meas_new = to_signed(angle_meas_new, 16) * -10;

      // update array of samples
      update_sample(&angle_meas, angle_meas_new);
    }

    if (addr == 0x285) {
      // Get current speed and standstill
      uint16_t right_rear = (GET_BYTE(to_push, 0) << 8) | (GET_BYTE(to_push, 1));
      uint16_t left_rear = (GET_BYTE(to_push, 2) << 8) | (GET_BYTE(to_push, 3));
      vehicle_moving = (right_rear | left_rear) != 0U;
      UPDATE_VEHICLE_SPEED((right_rear + left_rear) / 2.0 * 0.005 / 3.6);
    }

    // X-Trail 0x15c, Leaf 0x239
    if ((addr == 0x15c) || (addr == 0x239)) {
      if (addr == 0x15c){
        gas_pressed = ((GET_BYTE(to_push, 5) << 2) | ((GET_BYTE(to_push, 6) >> 6) & 0x3U)) > 3U;
      } else {
        gas_pressed = GET_BYTE(to_push, 0) > 3U;
      }
    }

    // X-trail 0x454, Leaf 0x239
    if ((addr == 0x454) || (addr == 0x239)) {
      if (addr == 0x454){
        brake_pressed = (GET_BYTE(to_push, 2) & 0x80U) != 0U;
      } else {
        brake_pressed = ((GET_BYTE(to_push, 4) >> 5) & 1U) != 0U;
      }
    }
  }

  // Handle cruise enabled
  if ((addr == 0x30f) && (bus == (nissan_alt_eps ? 1 : 2))) {
    bool cruise_engaged = (GET_BYTE(to_push, 0) >> 3) & 1U;
    pcm_cruise_check(cruise_engaged);
  }

  generic_rx_checks((addr == 0x169) && (bus == 0));
}


static bool nissan_tx_hook(const CANPacket_t *to_send) {
  bool tx = true;
  int addr = GET_ADDR(to_send);
  bool violation = false;

  // steer cmd checks
  if (addr == 0x169) {
    int desired_angle = ((GET_BYTE(to_send, 0) << 10) | (GET_BYTE(to_send, 1) << 2) | ((GET_BYTE(to_send, 2) >> 6) & 0x3U));
    bool lka_active = (GET_BYTE(to_send, 6) >> 4) & 1U;

    // Factor is -0.01, offset is 1310. Flip to correct sign, but keep units in CAN scale
    desired_angle = -desired_angle + (1310.0f * NISSAN_STEERING_LIMITS.angle_deg_to_can);

    if (steer_angle_cmd_checks(desired_angle, lka_active, NISSAN_STEERING_LIMITS)) {
      violation = true;
    }
  }

  // acc button check, only allow cancel button to be sent
  if (addr == 0x20b) {
    // Violation of any button other than cancel is pressed
    violation |= ((GET_BYTE(to_send, 1) & 0x3dU) > 0U);
  }

  if (violation) {
    tx = false;
  }

  return tx;
}


static int nissan_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;

  if (bus_num == 0) {
    bool block_msg = (addr == 0x280); // CANCEL_MSG
    if (!block_msg) {
      bus_fwd = 2;  // ADAS
    }
  }

  if (bus_num == 2) {
    // 0x169 is LKAS, 0x2b1 LKAS_HUD, 0x4cc LKAS_HUD_INFO_MSG
    bool block_msg = ((addr == 0x169) || (addr == 0x2b1) || (addr == 0x4cc));
    if (!block_msg) {
      bus_fwd = 0;  // V-CAN
    }
  }

  return bus_fwd;
}

static safety_config nissan_init(uint16_t param) {
  nissan_alt_eps = GET_FLAG(param, NISSAN_PARAM_ALT_EPS_BUS);
  return BUILD_SAFETY_CFG(nissan_rx_checks, NISSAN_TX_MSGS);
}

const safety_hooks nissan_hooks = {
  .init = nissan_init,
  .rx = nissan_rx_hook,
  .tx = nissan_tx_hook,
  .fwd = nissan_fwd_hook,
};
