#ifndef SAFETY_MODES_TESLA_H
#define SAFETY_MODES_TESLA_H

#include "opendbc/safety/declarations.h"

static uint8_t tesla_get_counter(const CANPacket_t *msg) {
  // default: 0x257, 0x118, 0x145, 0x286, 0x311
  // Signal: DI_speedCounter, DI_systemStatusCounter, ESP_statusCounter, DI_locStatusCounter, UI_warningCounter
  uint8_t cnt = msg->data[1] & 0x0FU;
  if (msg->addr == 0x2b9U) {
    // Signal: DAS_controlCounter
    cnt = msg->data[6] >> 5;
  }
  if (msg->addr == 0x488U) {
    // Signal: DAS_steeringControlCounter
    cnt = msg->data[2] & 0x0FU;
  }
  if (msg->addr == 0x155U) {
    // Signal: ESP_wheelRotationCounter
    cnt = msg->data[6] >> 4;
  }
  if (msg->addr == 0x370U) {
    // Signal: EPAS3S_sysStatusCounter
    cnt = msg->data[6] & 0x0FU;
  }
  return cnt;
}

static uint32_t tesla_get_checksum_byte(const uint32_t addr) {
  // default: 0x257, 0x118, 0x145, 0x286, 0x311
  // Signal: DI_speedChecksum, DI_systemStatusChecksum, ESP_statusChecksum, DI_locStatusChecksum, UI_warningChecksum
  uint32_t checksum_byte = 0U;
  if ((addr == 0x370U) || (addr == 0x2b9U) || (addr == 0x155U)) {
    // Signal: EPAS3S_sysStatusChecksum, DAS_controlChecksum, ESP_wheelRotationChecksum
    checksum_byte = 7U;
  }
  if (addr == 0x488U) {
    // Signal: DAS_steeringControlChecksum
    checksum_byte = 3U;
  }
  return checksum_byte;
}

static uint32_t tesla_get_checksum(const CANPacket_t *msg) {
  return msg->data[tesla_get_checksum_byte(msg->addr)];
}

static uint32_t tesla_compute_checksum(const CANPacket_t *msg) {
  uint32_t checksum_byte = tesla_get_checksum_byte(msg->addr);
  uint8_t chksum = (uint8_t)((msg->addr & 0xFFU) + ((msg->addr >> 8) & 0xFFU));
  uint32_t len = GET_LEN(msg);
  for (uint32_t i = 0U; i < len; i++) {
    if (i != checksum_byte) {
      chksum += msg->data[i];
    }
  }
  return chksum;
}

static bool tesla_get_quality_flag_valid(const CANPacket_t *msg) {
  bool valid = false;
  if (msg->addr == 0x155U) {
    valid = GET_BIT(msg, 40U);  // ESP_wheelSpeedsQF
  }
  if (msg->addr == 0x145U) {
    uint8_t user_brake_status = (msg->data[3] >> 5) & 0x03U;
    valid = (user_brake_status != 0U) && (user_brake_status != 3U);  // ESP_driverBrakeApply=NotInit_orOff, Faulty_SNA
  }
  return valid;
}

// latch a stock feature on its rising edge, unless it starts while engaged
static void tesla_latch_rising_edge(bool now, bool *prev, bool *latched, bool engaged) {
  if (now && !*prev && !engaged) {
    *latched = true;
  }
  if (!now) {
    *latched = false;
  }
  *prev = now;
}

static void tesla_rx_hook(const CANPacket_t *msg) {
  // Steering angle: (0.1 * val) - 819.2 in deg.
  if (msg_matches(msg, 0x370U, 0U)) {
    // Store it 1/10 deg to match steering request
    const int angle_meas_new = from_offset(GET_BYTES_BE(msg, 4, 2) & 0x3FFFU, 8192);
    update_sample(&safety.angle_meas, angle_meas_new);

    const uint8_t hands_on_level = msg->data[4] >> 6;  // EPAS3S_handsOnLevel
    const uint8_t eac_status = msg->data[6] >> 5;  // EPAS3S_eacStatus
    const uint8_t eac_error_code = msg->data[2] >> 4;  // EPAS3S_eacErrorCode

    // Disengage on normal user override, or if high angle rate fault from user overriding extremely quickly
    safety.steering_disengage = (hands_on_level >= 3U) || ((eac_status == 0U) && (eac_error_code == 9U));
  }

  // Vehicle speed (DI_speed)
  if (msg_matches(msg, 0x257U, 0U)) {
    // Vehicle speed: ((val * 0.08) - 40) / MS_TO_KPH
    uint32_t raw_speed = GET_BYTES_LE(msg, 1, 2) >> 4;
    float speed = (((float)raw_speed * 0.08f) - 40.f) * KPH_TO_MS;
    UPDATE_VEHICLE_SPEED(speed);
  }

  // 2nd vehicle speed (ESP_B)
  if (msg_matches(msg, 0x155U, 0U)) {
    // Disable controls if speeds from DI (Drive Inverter) and ESP ECUs are too far apart.
    uint32_t raw_esp_speed = (GET_BYTES_LE(msg, 5, 2) >> 2) & 0x3FFU;
    float esp_speed = (float)raw_esp_speed * 0.5f * KPH_TO_MS;
    speed_mismatch_check(esp_speed);
  }

  // Gas pressed
  if (msg_matches(msg, 0x118U, 0U)) {
    safety.gas_pressed = (msg->data[4] != 0U);
  }

  // Brake pressed
  if (msg_matches(msg, 0x145U, 0U)) {
    safety.brake_pressed = ((msg->data[3] >> 5) & 0x03U) == 2U;
  }

  // Cruise and Autopark/Summon state
  if (msg_matches(msg, 0x286U, 0U)) {
    // Autopark state
    uint8_t autopark_state = (msg->data[3] >> 1) & 0x0FU;  // DI_autoparkState
    bool tesla_autopark_now = (autopark_state == 3U) ||  // ACTIVE
                              (autopark_state == 4U) ||  // COMPLETE
                              (autopark_state == 9U);    // SELFPARK_STARTED

    tesla_latch_rising_edge(tesla_autopark_now, &safety.tesla_autopark_prev, &safety.tesla_autopark, safety.cruise_engaged_prev);

    // Cruise state
    uint8_t cruise_state = (msg->data[1] >> 4) & 0x07U;
    bool cruise_engaged = (cruise_state == 2U) ||  // ENABLED
                          (cruise_state == 3U) ||  // STANDSTILL
                          (cruise_state == 4U) ||  // OVERRIDE
                          (cruise_state == 6U) ||  // PRE_FAULT
                          (cruise_state == 7U);    // PRE_CANCEL
    cruise_engaged = cruise_engaged && !safety.tesla_autopark;

    pcm_cruise_check(cruise_engaged);
  }

  if (msg_matches(msg, 0x155U, 0U)) {
    safety.vehicle_moving = !GET_BIT(msg, 41U);  // ESP_vehicleStandstillSts
  }

  // DAS_control
  if (msg_matches(msg, 0x2b9U, 2U)) {
    // "AEB_ACTIVE"
    safety.tesla_stock_aeb = (msg->data[2] & 0x03U) == 1U;
  }

  // DAS_steeringControl
  if (msg_matches(msg, 0x488U, 2U)) {
    uint8_t steering_control_type = msg->data[2] >> 5;  // DAS_steeringControlType
    bool tesla_stock_lkas_now = steering_control_type == 2U;  // "LANE_KEEP_ASSIST"

    tesla_latch_rising_edge(tesla_stock_lkas_now, &safety.tesla_stock_lkas_prev, &safety.tesla_stock_lkas, safety.controls_allowed);
  }
}

static bool tesla_tx_hook(const CANPacket_t *msg) {
  const AngleSteeringLimits TESLA_STEERING_LIMITS = {
    .max_angle = 3600,  // 360 deg, EPAS faults above this
    .angle_deg_to_can = 10.f,
    .frequency = 50U,
  };

  // NOTE: based off TESLA_MODEL_Y to match openpilot
  const AngleSteeringParams TESLA_STEERING_PARAMS = {
    .slip_factor = -0.000580374383851451f,  // calc_slip_factor(VM)
    .steer_ratio = 12.f,
    .wheelbase = 2.89f,
  };

  const LongitudinalLimits TESLA_LONG_LIMITS = {
    .max_accel = 425,       // 2 m/s^2
    .min_accel = 288,       // -3.48 m/s^2
    .inactive_accel = 375,  // 0. m/s^2
  };

  bool tx = true;
  bool violation = false;

  // Don't send any messages when Autopark is active
  if (safety.tesla_autopark) {
    violation = true;
  }

  // Steering control: (0.1 * val) - 1638.35 in deg.
  if (msg->addr == 0x488U) {
    // We use 1/10 deg as a unit here
    int desired_angle = from_offset(GET_BYTES_BE(msg, 0, 2) & 0x7FFFU, 16384);
    uint8_t steer_control_type = msg->data[2] >> 5;  // DAS_steeringControlType
    bool steer_control_enabled = steer_control_type == 1U;  // ANGLE_CONTROL

    if (steer_angle_cmd_checks_vm(desired_angle, steer_control_enabled, TESLA_STEERING_LIMITS, TESLA_STEERING_PARAMS)) {
      violation = true;
    }

    bool valid_steer_control_type = (steer_control_type == 0U) ||  // NONE
                                    (steer_control_type == 1U);    // ANGLE_CONTROL
    if (!valid_steer_control_type) {
      violation = true;
    }

    if (safety.tesla_stock_lkas) {
      // Don't allow any steering commands when stock LKAS is active
      violation = true;
    }
  }

  // DAS_control: longitudinal control message
  if (msg->addr == 0x2b9U) {
    // No AEB events may be sent by openpilot
    uint8_t aeb_event = msg->data[2] & 0x03U;
    if (aeb_event != 0U) {
      violation = true;
    }

    // Don't send long/cancel messages when the stock AEB system is active
    if (safety.tesla_stock_aeb) {
      violation = true;
    }

    uint32_t raw_accel_max = (GET_BYTES_LE(msg, 5, 2) >> 4) & 0x1FFU;
    uint32_t raw_accel_min = (GET_BYTES_LE(msg, 4, 2) >> 3) & 0x1FFU;
    uint8_t acc_state = msg->data[1] >> 4;
    const uint32_t inactive_accel = (uint32_t)TESLA_LONG_LIMITS.inactive_accel;

    if (safety_setup.tesla_longitudinal) {
      // Prevent both acceleration from being negative, as this could cause the car to reverse after coming to standstill
      if ((raw_accel_max < inactive_accel) && (raw_accel_min < inactive_accel)) {
        violation = true;
      }

      // Don't allow any acceleration limits above the safety limits
      const bool max_accel_violation = longitudinal_accel_checks((int)raw_accel_max, TESLA_LONG_LIMITS);
      const bool min_accel_violation = longitudinal_accel_checks((int)raw_accel_min, TESLA_LONG_LIMITS);
      violation = violation || max_accel_violation || min_accel_violation;
    } else {
      // Can only send cancel longitudinal messages when not controlling longitudinal
      if (acc_state != 13U) {  // ACC_CANCEL_GENERIC_SILENT
        violation = true;
      }

      // No actuation is allowed when not controlling longitudinal
      if ((raw_accel_max != inactive_accel) || (raw_accel_min != inactive_accel)) {
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
    if (!safety.tesla_autopark) {
      // APS_eacMonitor
      if (addr == 0x27d) {
        block_msg = true;
      }

      // DAS_steeringControl
      if ((addr == 0x488) && !safety.tesla_stock_lkas) {
        block_msg = true;
      }

      // DAS_control
      if (safety_setup.tesla_longitudinal && (addr == 0x2b9) && !safety.tesla_stock_aeb) {
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

  SAFETY_UNUSED(param);
#ifdef ALLOW_DEBUG
  const uint16_t TESLA_FLAG_LONGITUDINAL_CONTROL = 1;
  safety_setup.tesla_longitudinal = GET_FLAG(param, TESLA_FLAG_LONGITUDINAL_CONTROL);
#endif

  // we need to assume Autopark/Summon on startup since DI_state is a low freq msg.
  // this is so that we don't fault if starting while these systems are active
  safety.tesla_autopark = true;

  static const RxCheck tesla_model3_y_rx_checks[] = {
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
  if (safety_setup.tesla_longitudinal) {
    ret = BUILD_SAFETY_CFG(tesla_model3_y_rx_checks, TESLA_M3_Y_LONG_TX_MSGS);
  } else {
    ret = BUILD_SAFETY_CFG(tesla_model3_y_rx_checks, TESLA_M3_Y_TX_MSGS);
  }
  return ret;
}

static const safety_hooks tesla_hooks = {
  .init = tesla_init,
  .rx = tesla_rx_hook,
  .tx = tesla_tx_hook,
  .fwd = tesla_fwd_hook,
  .get_counter = tesla_get_counter,
  .get_checksum = tesla_get_checksum,
  .compute_checksum = tesla_compute_checksum,
  .get_quality_flag_valid = tesla_get_quality_flag_valid,
};

#endif  // SAFETY_MODES_TESLA_H
