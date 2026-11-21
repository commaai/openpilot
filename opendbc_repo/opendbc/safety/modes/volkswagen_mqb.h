#ifndef SAFETY_MODES_VOLKSWAGEN_MQB_H
#define SAFETY_MODES_VOLKSWAGEN_MQB_H

#include "opendbc/safety/declarations.h"
#include "opendbc/safety/modes/volkswagen_common.h"

static safety_config volkswagen_mqb_init(uint16_t param) {
  // Transmit of GRA_ACC_01 is allowed on bus 0 and 2 to keep compatibility with gateway and camera integration
  // MSG_LH_EPS_03: openpilot needs to replace apparent driver steering input torque to pacify VW Emergency Assist
  static const CanMsg VOLKSWAGEN_MQB_STOCK_TX_MSGS[] = {
    {MSG_HCA_01, 0, 8, .check_relay = true},
    {MSG_GRA_ACC_01, 0, 8, .check_relay = false},
    {MSG_GRA_ACC_01, 2, 8, .check_relay = false},
    {MSG_LDW_02, 0, 8, .check_relay = true},
    {MSG_LH_EPS_03, 2, 8, .check_relay = true},
  };

  static const CanMsg VOLKSWAGEN_MQB_LONG_TX_MSGS[] = {
    {MSG_HCA_01, 0, 8, .check_relay = true},
    {MSG_LDW_02, 0, 8, .check_relay = true},
    {MSG_LH_EPS_03, 2, 8, .check_relay = true},
    {MSG_ACC_02, 0, 8, .check_relay = true},
    {MSG_ACC_06, 0, 8, .check_relay = true},
    {MSG_ACC_07, 0, 8, .check_relay = true},
  };

  static const RxCheck volkswagen_mqb_rx_checks[] = {
    {.msg = {{MSG_ESP_19, 0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_LH_EPS_03, 0, 8, 100U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_ESP_05, 0, 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_TSK_06, 0, 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_MOTOR_20, 0, 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_MOTOR_14, 0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_GRA_ACC_01, 0, 8, 33U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  volkswagen_common_init();

#ifdef ALLOW_DEBUG
  safety_setup.volkswagen_longitudinal = GET_FLAG(param, FLAG_VOLKSWAGEN_LONG_CONTROL);
#else
  SAFETY_UNUSED(param);
#endif

  return safety_setup.volkswagen_longitudinal ? BUILD_SAFETY_CFG(volkswagen_mqb_rx_checks, VOLKSWAGEN_MQB_LONG_TX_MSGS) : \
                                   BUILD_SAFETY_CFG(volkswagen_mqb_rx_checks, VOLKSWAGEN_MQB_STOCK_TX_MSGS);
}

static void volkswagen_mqb_rx_hook(const CANPacket_t *msg) {
  // Update in-motion state by sampling wheel speeds
  if (msg_matches(msg, MSG_ESP_19, 0U)) {
    // sum 4 wheel speeds
    uint32_t speed = 0U;
    for (int i = 0; i < 8; i += 2) {
      uint32_t wheel_speed = GET_BYTES_LE(msg, i, 2);
      speed += wheel_speed;
    }
    // Check all wheel speeds for any movement
    safety.vehicle_moving = speed > 0U;
  }

  // Update driver input torque samples
  // Signal: LH_EPS_03.EPS_Lenkmoment (absolute torque)
  // Signal: LH_EPS_03.EPS_VZ_Lenkmoment (direction)
  if (msg_matches(msg, MSG_LH_EPS_03, 0U)) {
    update_sample(&safety.torque_driver, volkswagen_mlb_mqb_driver_input_torque(msg));
  }

  if (msg_matches(msg, MSG_TSK_06, 0U)) {
    uint8_t tsk_status = msg->data[3] & 0x7U;
    volkswagen_tsk_status_check((int)tsk_status, !safety_setup.volkswagen_longitudinal);
  }

  if (msg_matches(msg, MSG_GRA_ACC_01, 0U)) {
    // If using openpilot longitudinal, enter controls on falling edge of Set or Resume with main switch on
    // Signal: GRA_ACC_01.GRA_Tip_Setzen
    // Signal: GRA_ACC_01.GRA_Tip_Wiederaufnahme
    if (safety_setup.volkswagen_longitudinal) {
      volkswagen_set_resume_check(GET_BIT(msg, 16U), GET_BIT(msg, 19U));
    }
    // Always exit controls on rising edge of Cancel
    // Signal: GRA_ACC_01.GRA_Abbrechen
    if (GET_BIT(msg, 13U)) {
      safety.controls_allowed = false;
    }
  }

  // Signal: Motor_20.MO_Fahrpedalrohwert_01
  if (msg_matches(msg, MSG_MOTOR_20, 0U)) {
    safety.gas_pressed = ((GET_BYTES_LE(msg, 0, 4) >> 12) & 0xFFU) != 0U;
  }

  // Signal: Motor_14.MO_Fahrer_bremst (ECU detected brake pedal switch F63)
  if (msg_matches(msg, MSG_MOTOR_14, 0U)) {
    safety.volkswagen_brake_pedal_switch = GET_BIT(msg, 28U);
  }

  // Signal: ESP_05.ESP_Fahrer_bremst (ESP detected driver brake pressure above platform specified threshold)
  if (msg_matches(msg, MSG_ESP_05, 0U)) {
    safety.volkswagen_brake_pressure_detected = GET_BIT(msg, 26U);
  }

  safety.brake_pressed = safety.volkswagen_brake_pedal_switch || safety.volkswagen_brake_pressure_detected;
}

static bool volkswagen_mqb_tx_hook(const CANPacket_t *msg) {
  // lateral limits
  const TorqueSteeringLimits VOLKSWAGEN_MQB_STEERING_LIMITS = {
    .max_torque = 300,             // 3.0 Nm (EPS side max of 3.0Nm with fault if violated)
    .max_rt_delta = 75,            // 4 max rate up * 50Hz send rate * 250000 RT interval / 1000000 = 50 ; 50 * 1.5 for safety pad = 75
    .max_rate_up = 4,              // 2.0 Nm/s RoC limit (EPS rack has own soft-limit of 5.0 Nm/s)
    .max_rate_down = 10,           // 5.0 Nm/s RoC limit (EPS rack has own soft-limit of 5.0 Nm/s)
    .driver_torque_allowance = 80,
    .driver_torque_multiplier = 3,
    .type = TorqueDriverLimited,
  };

  // longitudinal limits
  // acceleration in m/s2 * 1000 to avoid floating point math
  const LongitudinalLimits VOLKSWAGEN_MQB_LONG_LIMITS = {
    .max_accel = 2000,
    .min_accel = -3500,
    .inactive_accel = 3010,  // VW sends one increment above the max range when inactive
  };

  bool tx = true;

  // Safety check for HCA_01 Heading Control Assist torque
  if (msg->addr == MSG_HCA_01) {
    int desired_torque = volkswagen_mlb_mqb_steering_control_torque(msg);
    bool steer_req = GET_BIT(msg, 30U);

    if (steer_torque_cmd_checks(desired_torque, steer_req, VOLKSWAGEN_MQB_STEERING_LIMITS)) {
      tx = false;
    }
  }

  // Safety check for both ACC_06 and ACC_07 acceleration requests
  // To avoid floating point math, scale upward and compare to pre-scaled safety m/s2 boundaries
  if ((msg->addr == MSG_ACC_06) || (msg->addr == MSG_ACC_07)) {
    bool violation = false;
    int desired_accel = 0;

    if (msg->addr == MSG_ACC_06) {
      // Signal: ACC_06.ACC_Sollbeschleunigung_02 (acceleration in m/s2, scale 0.005, offset -7.22)
      desired_accel = from_offset((GET_BYTES_LE(msg, 3, 2) & 0x7FFU) * 5U, 7220);
    } else {
      // Signal: ACC_07.ACC_Folgebeschl (acceleration in m/s2, scale 0.03, offset -4.6)
      int secondary_accel = ((int)msg->data[4] * 30) - 4600;
      violation = (secondary_accel != 3020);  // enforce always inactive (one increment above max range) at this time
      // Signal: ACC_07.ACC_Sollbeschleunigung_02 (acceleration in m/s2, scale 0.005, offset -7.22)
      desired_accel = from_offset(((GET_BYTES_LE(msg, 6, 2) >> 5) & 0x7FFU) * 5U, 7220);
    }

    const bool accel_violation = longitudinal_accel_checks(desired_accel, VOLKSWAGEN_MQB_LONG_LIMITS);
    violation = violation || accel_violation;

    if (violation) {
      tx = false;
    }
  }

  // FORCE CANCEL: ensuring that only the cancel button press is sent when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  if ((msg->addr == MSG_GRA_ACC_01) && !safety.controls_allowed) {
    // disallow resume and set: bits 16 and 19
    if ((msg->data[2] & 0x9U) != 0U) {
      tx = false;
    }
  }

  return tx;
}

static const safety_hooks volkswagen_mqb_hooks = {
  .init = volkswagen_mqb_init,
  .rx = volkswagen_mqb_rx_hook,
  .tx = volkswagen_mqb_tx_hook,
  .get_counter = volkswagen_mxb_get_counter,
  .get_checksum = volkswagen_mxb_get_checksum,
  .compute_checksum = volkswagen_mxb_compute_crc,
};

#endif  // SAFETY_MODES_VOLKSWAGEN_MQB_H
