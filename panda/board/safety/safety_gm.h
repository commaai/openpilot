const SteeringLimits GM_STEERING_LIMITS = {
  .max_steer = 300,
  .max_rate_up = 10,
  .max_rate_down = 15,
  .driver_torque_allowance = 65,
  .driver_torque_factor = 4,
  .max_rt_delta = 128,
  .max_rt_interval = 250000,
  .type = TorqueDriverLimited,
};

const LongitudinalLimits GM_ASCM_LONG_LIMITS = {
  .max_gas = 3072,
  .min_gas = 1404,
  .inactive_gas = 1404,
  .max_brake = 400,
};

const LongitudinalLimits GM_CAM_LONG_LIMITS = {
  .max_gas = 3400,
  .min_gas = 1514,
  .inactive_gas = 1554,
  .max_brake = 400,
};

const LongitudinalLimits *gm_long_limits;

const int GM_STANDSTILL_THRSLD = 10;  // 0.311kph

const CanMsg GM_ASCM_TX_MSGS[] = {{0x180, 0, 4}, {0x409, 0, 7}, {0x40A, 0, 7}, {0x2CB, 0, 8}, {0x370, 0, 6},  // pt bus
                                  {0xA1, 1, 7}, {0x306, 1, 8}, {0x308, 1, 7}, {0x310, 1, 2},   // obs bus
                                  {0x315, 2, 5},  // ch bus
                                  {0x104c006c, 3, 3}, {0x10400060, 3, 5}};  // gmlan

const CanMsg GM_CAM_TX_MSGS[] = {{0x180, 0, 4},  // pt bus
                                 {0x1E1, 2, 7}, {0x184, 2, 8}};  // camera bus

const CanMsg GM_CAM_LONG_TX_MSGS[] = {{0x180, 0, 4}, {0x315, 0, 5}, {0x2CB, 0, 8}, {0x370, 0, 6},  // pt bus
                                      {0x184, 2, 8}};  // camera bus

// TODO: do checksum and counter checks. Add correct timestep, 0.1s for now.
RxCheck gm_rx_checks[] = {
  {.msg = {{0x184, 0, 8, .frequency = 10U}, { 0 }, { 0 }}},
  {.msg = {{0x34A, 0, 5, .frequency = 10U}, { 0 }, { 0 }}},
  {.msg = {{0x1E1, 0, 7, .frequency = 10U}, { 0 }, { 0 }}},
  {.msg = {{0xBE, 0, 6, .frequency = 10U},    // Volt, Silverado, Acadia Denali
           {0xBE, 0, 7, .frequency = 10U},    // Bolt EUV
           {0xBE, 0, 8, .frequency = 10U}}},  // Escalade
  {.msg = {{0x1C4, 0, 8, .frequency = 10U}, { 0 }, { 0 }}},
  {.msg = {{0xC9, 0, 8, .frequency = 10U}, { 0 }, { 0 }}},
};

const uint16_t GM_PARAM_HW_CAM = 1;
const uint16_t GM_PARAM_HW_CAM_LONG = 2;

enum {
  GM_BTN_UNPRESS = 1,
  GM_BTN_RESUME = 2,
  GM_BTN_SET = 3,
  GM_BTN_CANCEL = 6,
};

enum {GM_ASCM, GM_CAM} gm_hw = GM_ASCM;
bool gm_cam_long = false;
bool gm_pcm_cruise = false;

static void gm_rx_hook(const CANPacket_t *to_push) {
  if (GET_BUS(to_push) == 0U) {
    int addr = GET_ADDR(to_push);

    if (addr == 0x184) {
      int torque_driver_new = ((GET_BYTE(to_push, 6) & 0x7U) << 8) | GET_BYTE(to_push, 7);
      torque_driver_new = to_signed(torque_driver_new, 11);
      // update array of samples
      update_sample(&torque_driver, torque_driver_new);
    }

    // sample rear wheel speeds
    if (addr == 0x34A) {
      int left_rear_speed = (GET_BYTE(to_push, 0) << 8) | GET_BYTE(to_push, 1);
      int right_rear_speed = (GET_BYTE(to_push, 2) << 8) | GET_BYTE(to_push, 3);
      vehicle_moving = (left_rear_speed > GM_STANDSTILL_THRSLD) || (right_rear_speed > GM_STANDSTILL_THRSLD);
    }

    // ACC steering wheel buttons (GM_CAM is tied to the PCM)
    if ((addr == 0x1E1) && !gm_pcm_cruise) {
      int button = (GET_BYTE(to_push, 5) & 0x70U) >> 4;

      // enter controls on falling edge of set or rising edge of resume (avoids fault)
      bool set = (button != GM_BTN_SET) && (cruise_button_prev == GM_BTN_SET);
      bool res = (button == GM_BTN_RESUME) && (cruise_button_prev != GM_BTN_RESUME);
      if (set || res) {
        controls_allowed = true;
      }

      // exit controls on cancel press
      if (button == GM_BTN_CANCEL) {
        controls_allowed = false;
      }

      cruise_button_prev = button;
    }

    // Reference for brake pressed signals:
    // https://github.com/commaai/openpilot/blob/master/selfdrive/car/gm/carstate.py
    if ((addr == 0xBE) && (gm_hw == GM_ASCM)) {
      brake_pressed = GET_BYTE(to_push, 1) >= 8U;
    }

    if ((addr == 0xC9) && (gm_hw == GM_CAM)) {
      brake_pressed = GET_BIT(to_push, 40U);
    }

    if (addr == 0x1C4) {
      gas_pressed = GET_BYTE(to_push, 5) != 0U;

      // enter controls on rising edge of ACC, exit controls when ACC off
      if (gm_pcm_cruise) {
        bool cruise_engaged = (GET_BYTE(to_push, 1) >> 5) != 0U;
        pcm_cruise_check(cruise_engaged);
      }
    }

    if (addr == 0xBD) {
      regen_braking = (GET_BYTE(to_push, 0) >> 4) != 0U;
    }

    bool stock_ecu_detected = (addr == 0x180);  // ASCMLKASteeringCmd

    // Check ASCMGasRegenCmd only if we're blocking it
    if (!gm_pcm_cruise && (addr == 0x2CB)) {
      stock_ecu_detected = true;
    }
    generic_rx_checks(stock_ecu_detected);
  }
}

static bool gm_tx_hook(const CANPacket_t *to_send) {
  bool tx = true;
  int addr = GET_ADDR(to_send);

  // BRAKE: safety check
  if (addr == 0x315) {
    int brake = ((GET_BYTE(to_send, 0) & 0xFU) << 8) + GET_BYTE(to_send, 1);
    brake = (0x1000 - brake) & 0xFFF;
    if (longitudinal_brake_checks(brake, *gm_long_limits)) {
      tx = false;
    }
  }

  // LKA STEER: safety check
  if (addr == 0x180) {
    int desired_torque = ((GET_BYTE(to_send, 0) & 0x7U) << 8) + GET_BYTE(to_send, 1);
    desired_torque = to_signed(desired_torque, 11);

    bool steer_req = GET_BIT(to_send, 3U);

    if (steer_torque_cmd_checks(desired_torque, steer_req, GM_STEERING_LIMITS)) {
      tx = false;
    }
  }

  // GAS/REGEN: safety check
  if (addr == 0x2CB) {
    bool apply = GET_BIT(to_send, 0U);
    int gas_regen = ((GET_BYTE(to_send, 2) & 0x7FU) << 5) + ((GET_BYTE(to_send, 3) & 0xF8U) >> 3);

    bool violation = false;
    // Allow apply bit in pre-enabled and overriding states
    violation |= !controls_allowed && apply;
    violation |= longitudinal_gas_checks(gas_regen, *gm_long_limits);

    if (violation) {
      tx = false;
    }
  }

  // BUTTONS: used for resume spamming and cruise cancellation with stock longitudinal
  if ((addr == 0x1E1) && gm_pcm_cruise) {
    int button = (GET_BYTE(to_send, 5) >> 4) & 0x7U;

    bool allowed_cancel = (button == 6) && cruise_engaged_prev;
    if (!allowed_cancel) {
      tx = false;
    }
  }

  return tx;
}

static int gm_fwd_hook(int bus_num, int addr) {
  int bus_fwd = -1;

  if (gm_hw == GM_CAM) {
    if (bus_num == 0) {
      // block PSCMStatus; forwarded through openpilot to hide an alert from the camera
      bool is_pscm_msg = (addr == 0x184);
      if (!is_pscm_msg) {
        bus_fwd = 2;
      }
    }

    if (bus_num == 2) {
      // block lkas message and acc messages if gm_cam_long, forward all others
      bool is_lkas_msg = (addr == 0x180);
      bool is_acc_msg = (addr == 0x315) || (addr == 0x2CB) || (addr == 0x370);
      bool block_msg = is_lkas_msg || (is_acc_msg && gm_cam_long);
      if (!block_msg) {
        bus_fwd = 0;
      }
    }
  }

  return bus_fwd;
}

static safety_config gm_init(uint16_t param) {
  gm_hw = GET_FLAG(param, GM_PARAM_HW_CAM) ? GM_CAM : GM_ASCM;

  if (gm_hw == GM_ASCM) {
    gm_long_limits = &GM_ASCM_LONG_LIMITS;
  } else if (gm_hw == GM_CAM) {
    gm_long_limits = &GM_CAM_LONG_LIMITS;
  } else {
  }

#ifdef ALLOW_DEBUG
  gm_cam_long = GET_FLAG(param, GM_PARAM_HW_CAM_LONG);
#endif
  gm_pcm_cruise = (gm_hw == GM_CAM) && !gm_cam_long;

  safety_config ret = BUILD_SAFETY_CFG(gm_rx_checks, GM_ASCM_TX_MSGS);
  if (gm_hw == GM_CAM) {
    ret = gm_cam_long ? BUILD_SAFETY_CFG(gm_rx_checks, GM_CAM_LONG_TX_MSGS) : BUILD_SAFETY_CFG(gm_rx_checks, GM_CAM_TX_MSGS);
  }
  return ret;
}

const safety_hooks gm_hooks = {
  .init = gm_init,
  .rx = gm_rx_hook,
  .tx = gm_tx_hook,
  .fwd = gm_fwd_hook,
};
