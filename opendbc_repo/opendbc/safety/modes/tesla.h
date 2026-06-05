#pragma once

#include "opendbc/safety/declarations.h"

static bool tesla_longitudinal = false;
static bool tesla_fsd_14 = false;
static bool tesla_stock_aeb = false;

// Only rising edges while controls are not allowed are considered for these systems:
// TODO: Only LKAS (non-emergency) is currently supported since we've only seen it
static bool tesla_stock_lkas = false;
static bool tesla_stock_lkas_prev = false;

// Only Summon is currently supported due to Autopark not setting Autopark state properly
static bool tesla_autopark = false;
static bool tesla_autopark_prev = false;

static uint8_t tesla_get_counter(const CANPacket_t *msg) {

  uint8_t cnt = 0;
  if (msg->addr == 0x2b9U) {
    // Signal: DAS_controlCounter
    cnt = msg->data[6] >> 5;
  } else if (msg->addr == 0x488U) {
    // Signal: DAS_steeringControlCounter
    cnt = msg->data[2] & 0x0FU;
  } else if ((msg->addr == 0x257U) || (msg->addr == 0x118U) || (msg->addr == 0x145U) || (msg->addr == 0x286U) || (msg->addr == 0x311U)) {
    // Signal: DI_speedCounter, DI_systemStatusCounter, ESP_statusCounter, DI_locStatusCounter, UI_warningCounter
    cnt = msg->data[1] & 0x0FU;
  } else if (msg->addr == 0x155U) {
    // Signal: ESP_wheelRotationCounter
    cnt = msg->data[6] >> 4;
  } else if (msg->addr == 0x370U) {
    // Signal: EPAS3S_sysStatusCounter
    cnt = msg->data[6] & 0x0FU;
  } else {
  }
  return cnt;
}

static int _tesla_get_checksum_byte(const int addr) {
  int checksum_byte = -1;
  if ((addr == 0x370) || (addr == 0x2b9) || (addr == 0x155)) {
    // Signal: EPAS3S_sysStatusChecksum, DAS_controlChecksum, ESP_wheelRotationChecksum
    checksum_byte = 7;
  } else if (addr == 0x488) {
    // Signal: DAS_steeringControlChecksum
    checksum_byte = 3;
  } else if ((addr == 0x257) || (addr == 0x118) || (addr == 0x145) || (addr == 0x286) || (addr == 0x311)) {
    // Signal: DI_speedChecksum, DI_systemStatusChecksum, ESP_statusChecksum, DI_locStatusChecksum, UI_warningChecksum
    checksum_byte = 0;
  } else {
  }
  return checksum_byte;
}

static uint32_t tesla_get_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  int checksum_byte = _tesla_get_checksum_byte(msg->addr);
  if (checksum_byte != -1) {
    chksum = msg->data[checksum_byte];
  }
  return chksum;
}

static uint32_t tesla_compute_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  int checksum_byte = _tesla_get_checksum_byte(msg->addr);

  if (checksum_byte != -1) {
    chksum = (uint8_t)((msg->addr & 0xFFU) + ((msg->addr >> 8) & 0xFFU));
    int len = GET_LEN(msg);
    for (int i = 0; i < len; i++) {
      if (i != checksum_byte) {
        chksum += msg->data[i];
      }
    }
  }
  return chksum;
}

static bool tesla_get_quality_flag_valid(const CANPacket_t *msg) {

  bool valid = false;
  if (msg->addr == 0x155U) {
    valid = (msg->data[5] & 0x1U) == 0x1U;  // ESP_wheelSpeedsQF
  } else if (msg->addr == 0x145U) {
    int user_brake_status = (msg->data[3] >> 5) & 0x03U;
    valid = (user_brake_status != 0) && (user_brake_status != 3);  // ESP_driverBrakeApply=NotInit_orOff, Faulty_SNA
  } else {
  }
  return valid;
}

static int tesla_get_steer_ctrl_type(const int ctrl_type) {
  // Returns ANGLE_CONTROL-equivalent control type for FSD 14
  int steer_ctrl_type = ctrl_type;
  if (tesla_fsd_14) {
    if (ctrl_type == 1) {
      steer_ctrl_type = 2;
    } else if (ctrl_type == 2) {
      steer_ctrl_type = 1;
    } else {
    }
  }
  return steer_ctrl_type;
}

static void tesla_rx_hook(const CANPacket_t *msg) {

  if (msg->bus == 0U) {
    // Steering angle: (0.1 * val) - 819.2 in deg.
    if (msg->addr == 0x370U) {
      // Store it 1/10 deg to match steering request
      const int angle_meas_new = (((msg->data[4] & 0x3FU) << 8) | msg->data[5]) - 8192U;
      update_sample(&angle_meas, angle_meas_new);

      const int hands_on_level = msg->data[4] >> 6;  // EPAS3S_handsOnLevel
      const int eac_status = msg->data[6] >> 5;  // EPAS3S_eacStatus
      const int eac_error_code = msg->data[2] >> 4;  // EPAS3S_eacErrorCode

      // Disengage on normal user override, or if high angle rate fault from user overriding extremely quickly
      steering_disengage = (hands_on_level >= 3) || ((eac_status == 0) && (eac_error_code == 9));
    }

    // Vehicle speed (DI_speed)
    if (msg->addr == 0x257U) {
      // Vehicle speed: ((val * 0.08) - 40) / MS_TO_KPH
      float speed = ((((msg->data[2] << 4) | (msg->data[1] >> 4)) * 0.08) - 40.) * KPH_TO_MS;
      UPDATE_VEHICLE_SPEED(speed);
    }

    // 2nd vehicle speed (ESP_B)
    if (msg->addr == 0x155U) {
      // Disable controls if speeds from DI (Drive Inverter) and ESP ECUs are too far apart.
      float esp_speed = (((msg->data[6] & 0x0FU) << 6) | (msg->data[5] >> 2)) * 0.5 * KPH_TO_MS;
      speed_mismatch_check(esp_speed);
    }

    // Gas pressed
    if (msg->addr == 0x118U) {
      gas_pressed = (msg->data[4] != 0U);
    }

    // Brake pressed
    if (msg->addr == 0x145U) {
      brake_pressed = ((msg->data[3] >> 5) & 0x03U) == 2U;
    }

    // Cruise and Autopark/Summon state
    if (msg->addr == 0x286U) {
      // Autopark state
      int autopark_state = (msg->data[3] >> 1) & 0x0FU;  // DI_autoparkState
      bool tesla_autopark_now = (autopark_state == 3) ||  // ACTIVE
                                (autopark_state == 4) ||  // COMPLETE
                                (autopark_state == 9);    // SELFPARK_STARTED

      // Only consider rising edges while controls are not allowed
      if (tesla_autopark_now && !tesla_autopark_prev && !cruise_engaged_prev) {
        tesla_autopark = true;
      }
      if (!tesla_autopark_now) {
        tesla_autopark = false;
      }
      tesla_autopark_prev = tesla_autopark_now;

      // Cruise state
      int cruise_state = (msg->data[1] >> 4) & 0x07U;
      bool cruise_engaged = (cruise_state == 2) ||  // ENABLED
                            (cruise_state == 3) ||  // STANDSTILL
                            (cruise_state == 4) ||  // OVERRIDE
                            (cruise_state == 6) ||  // PRE_FAULT
                            (cruise_state == 7);    // PRE_CANCEL
      cruise_engaged = cruise_engaged && !tesla_autopark;

      pcm_cruise_check(cruise_engaged);
    }

    if (msg->addr == 0x155U) {
      vehicle_moving = !GET_BIT(msg, 41U);  // ESP_vehicleStandstillSts
    }
  }

  if (msg->bus == 2U) {
    // DAS_control
    if (msg->addr == 0x2b9U) {
      // "AEB_ACTIVE"
      tesla_stock_aeb = (msg->data[2] & 0x03U) == 1U;
    }

    // DAS_steeringControl
    if (msg->addr == 0x488U) {
      int steering_control_type = msg->data[2] >> 6;
      bool tesla_stock_lkas_now = steering_control_type == tesla_get_steer_ctrl_type(2);  // "LANE_KEEP_ASSIST"

      // Only consider rising edges while controls are not allowed
      if (tesla_stock_lkas_now && !tesla_stock_lkas_prev && !controls_allowed) {
        tesla_stock_lkas = true;
      }
      if (!tesla_stock_lkas_now) {
        tesla_stock_lkas = false;
      }
      tesla_stock_lkas_prev = tesla_stock_lkas_now;
    }
  }
}


static bool tesla_tx_hook(const CANPacket_t *msg) {
  const AngleSteeringLimits TESLA_STEERING_LIMITS = {
    .max_angle = 3600,  // 360 deg, EPAS faults above this
    .angle_deg_to_can = 10,
    .frequency = 50U,
  };

  // NOTE: based off TESLA_MODEL_Y to match openpilot
  const AngleSteeringParams TESLA_STEERING_PARAMS = {
    .slip_factor = -0.000580374383851451,  // calc_slip_factor(VM)
    .steer_ratio = 12.,
    .wheelbase = 2.89,
  };

  const LongitudinalLimits TESLA_LONG_LIMITS = {
    .max_accel = 425,       // 2 m/s^2
    .min_accel = 288,       // -3.48 m/s^2
    .inactive_accel = 375,  // 0. m/s^2
  };

  bool tx = true;
  bool violation = false;

  // Don't send any messages when Autopark is active
  if (tesla_autopark) {
    violation = true;
  }

  // Steering control: (0.1 * val) - 1638.35 in deg.
  if (msg->addr == 0x488U) {
    // We use 1/10 deg as a unit here
    int raw_angle_can = ((msg->data[0] & 0x7FU) << 8) | msg->data[1];
    int desired_angle = raw_angle_can - 16384;
    int steer_control_type = msg->data[2] >> 6;
    const int angle_ctrl_type = tesla_get_steer_ctrl_type(1);
    bool steer_control_enabled = steer_control_type == angle_ctrl_type;  // ANGLE_CONTROL

    if (steer_angle_cmd_checks_vm(desired_angle, steer_control_enabled, TESLA_STEERING_LIMITS, TESLA_STEERING_PARAMS)) {
      violation = true;
    }

    bool valid_steer_control_type = (steer_control_type == 0) ||  // NONE
                                    (steer_control_type == angle_ctrl_type);    // ANGLE_CONTROL
    if (!valid_steer_control_type) {
      violation = true;
    }

    if (tesla_stock_lkas) {
      // Don't allow any steering commands when stock LKAS is active
      violation = true;
    }
  }

  // DAS_control: longitudinal control message
  if (msg->addr == 0x2b9U) {
    // No AEB events may be sent by openpilot
    int aeb_event = msg->data[2] & 0x03U;
    if (aeb_event != 0) {
      violation = true;
    }

    // Don't send long/cancel messages when the stock AEB system is active
    if (tesla_stock_aeb) {
      violation = true;
    }

    int raw_accel_max = ((msg->data[6] & 0x1FU) << 4) | (msg->data[5] >> 4);
    int raw_accel_min = ((msg->data[5] & 0x0FU) << 5) | (msg->data[4] >> 3);
    int acc_state = msg->data[1] >> 4;

    if (tesla_longitudinal) {
      // Prevent both acceleration from being negative, as this could cause the car to reverse after coming to standstill
      if ((raw_accel_max < TESLA_LONG_LIMITS.inactive_accel) && (raw_accel_min < TESLA_LONG_LIMITS.inactive_accel)) {
        violation = true;
      }

      // Don't allow any acceleration limits above the safety limits
      violation |= longitudinal_accel_checks(raw_accel_max, TESLA_LONG_LIMITS);
      violation |= longitudinal_accel_checks(raw_accel_min, TESLA_LONG_LIMITS);
    } else {
      // Can only send cancel longitudinal messages when not controlling longitudinal
      if (acc_state != 13) {  // ACC_CANCEL_GENERIC_SILENT
        violation = true;
      }

      // No actuation is allowed when not controlling longitudinal
      if ((raw_accel_max != TESLA_LONG_LIMITS.inactive_accel) || (raw_accel_min != TESLA_LONG_LIMITS.inactive_accel)) {
        violation = true;
      }
    }
  }

  if (violation) {
    tx = false;
  }

  return tx;
}

static bool tesla_fwd_hook(int bus_num, int addr) {
  bool block_msg = false;

  if (bus_num == 2) {
    if (!tesla_autopark) {
      // APS_eacMonitor
      if (addr == 0x27d) {
        block_msg = true;
      }

      // DAS_steeringControl
      if ((addr == 0x488) && !tesla_stock_lkas) {
        block_msg = true;
      }

      // DAS_control
      if (tesla_longitudinal && (addr == 0x2b9) && !tesla_stock_aeb) {
        block_msg = true;
      }
    }
  }

  return block_msg;
}

static safety_config tesla_init(uint16_t param) {

  static const CanMsg TESLA_M3_Y_TX_MSGS[] = {
    {0x488, 0, 4, .check_relay = true, .disable_static_blocking = true},   // DAS_steeringControl
    {0x2b9, 0, 8, .check_relay = false},                                   // DAS_control (for cancel)
    {0x27D, 0, 3, .check_relay = true, .disable_static_blocking = true},   // APS_eacMonitor
  };

  static const CanMsg TESLA_M3_Y_LONG_TX_MSGS[] = {
    {0x488, 0, 4, .check_relay = true, .disable_static_blocking = true},  // DAS_steeringControl
    {0x2b9, 0, 8, .check_relay = true, .disable_static_blocking = true},  // DAS_control
    {0x27D, 0, 3, .check_relay = true, .disable_static_blocking = true},  // APS_eacMonitor
  };

  const uint16_t TESLA_FLAG_FSD_14 = 2;
  tesla_fsd_14 = GET_FLAG(param, TESLA_FLAG_FSD_14);

#ifdef ALLOW_DEBUG
  const uint16_t TESLA_FLAG_LONGITUDINAL_CONTROL = 1;
  tesla_longitudinal = GET_FLAG(param, TESLA_FLAG_LONGITUDINAL_CONTROL);
#endif

  tesla_stock_aeb = false;
  tesla_stock_lkas = false;
  tesla_stock_lkas_prev = false;
  // we need to assume Autopark/Summon on startup since DI_state is a low freq msg.
  // this is so that we don't fault if starting while these systems are active
  tesla_autopark = true;
  tesla_autopark_prev = false;

  static RxCheck tesla_model3_y_rx_checks[] = {
    {.msg = {{0x2b9, 2, 8, 25U, .max_counter = 7U, .ignore_quality_flag = true}, { 0 }, { 0 }}},    // DAS_control
    {.msg = {{0x488, 2, 4, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // DAS_steeringControl
    {.msg = {{0x257, 0, 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // DI_speed (speed in kph)
    {.msg = {{0x155, 0, 8, 50U, .max_counter = 15U}, { 0 }, { 0 }}},                                // ESP_B (2nd speed in kph)
    {.msg = {{0x370, 0, 8, 100U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  // EPAS3S_sysStatus (steering angle)
    {.msg = {{0x118, 0, 8, 100U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  // DI_systemStatus (gas pedal)
    {.msg = {{0x145, 0, 8, 50U, .max_counter = 15U}, { 0 }, { 0 }}},                                // ESP_status (brakes)
    {.msg = {{0x286, 0, 8, 10U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // DI_state (acc state)
    {.msg = {{0x311, 0, 7, 10U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // UI_warning (blinkers, buckle switch & doors)
  };

  safety_config ret;
  if (tesla_longitudinal) {
    ret = BUILD_SAFETY_CFG(tesla_model3_y_rx_checks, TESLA_M3_Y_LONG_TX_MSGS);
  } else {
    ret = BUILD_SAFETY_CFG(tesla_model3_y_rx_checks, TESLA_M3_Y_TX_MSGS);
  }
  return ret;
}

const safety_hooks tesla_hooks = {
  .init = tesla_init,
  .rx = tesla_rx_hook,
  .tx = tesla_tx_hook,
  .fwd = tesla_fwd_hook,
  .get_counter = tesla_get_counter,
  .get_checksum = tesla_get_checksum,
  .compute_checksum = tesla_compute_checksum,
  .get_quality_flag_valid = tesla_get_quality_flag_valid,
};
