#pragma once

#include "opendbc/safety/safety_declarations.h"
#include "opendbc/safety/modes/volkswagen_common.h"

static bool volkswagen_mqb_brake_pedal_switch = false;
static bool volkswagen_mqb_brake_pressure_detected = false;


static safety_config volkswagen_mqb_init(uint16_t param) {
  // Transmit of GRA_ACC_01 is allowed on bus 0 and 2 to keep compatibility with gateway and camera integration
  // MSG_LH_EPS_03: openpilot needs to replace apparent driver steering input torque to pacify VW Emergency Assist
  static const CanMsg VOLKSWAGEN_MQB_STOCK_TX_MSGS[] = {{MSG_HCA_01, 0, 8, .check_relay = true}, {MSG_GRA_ACC_01, 0, 8, .check_relay = false}, {MSG_GRA_ACC_01, 2, 8, .check_relay = false},
                                                        {MSG_LDW_02, 0, 8, .check_relay = true}, {MSG_LH_EPS_03, 2, 8, .check_relay = true}};

  static const CanMsg VOLKSWAGEN_MQB_LONG_TX_MSGS[] = {{MSG_HCA_01, 0, 8, .check_relay = true}, {MSG_LDW_02, 0, 8, .check_relay = true}, {MSG_LH_EPS_03, 2, 8, .check_relay = true},
                                                       {MSG_ACC_02, 0, 8, .check_relay = true}, {MSG_ACC_06, 0, 8, .check_relay = true}, {MSG_ACC_07, 0, 8, .check_relay = true}};

  static RxCheck volkswagen_mqb_rx_checks[] = {
    {.msg = {{MSG_ESP_19, 0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_LH_EPS_03, 0, 8, 100U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_ESP_05, 0, 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_TSK_06, 0, 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_MOTOR_20, 0, 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_MOTOR_14, 0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_GRA_ACC_01, 0, 8, 33U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  UNUSED(param);

  volkswagen_set_button_prev = false;
  volkswagen_resume_button_prev = false;
  volkswagen_mqb_brake_pedal_switch = false;
  volkswagen_mqb_brake_pressure_detected = false;

#ifdef ALLOW_DEBUG
  volkswagen_longitudinal = GET_FLAG(param, FLAG_VOLKSWAGEN_LONG_CONTROL);
#endif
  gen_crc_lookup_table_8(0x2F, volkswagen_crc8_lut_8h2f);
  return volkswagen_longitudinal ? BUILD_SAFETY_CFG(volkswagen_mqb_rx_checks, VOLKSWAGEN_MQB_LONG_TX_MSGS) : \
                                   BUILD_SAFETY_CFG(volkswagen_mqb_rx_checks, VOLKSWAGEN_MQB_STOCK_TX_MSGS);
}

static void volkswagen_mqb_rx_hook(const CANPacket_t *msg) {
  if (msg->bus == 0U) {
    // Update in-motion state by sampling wheel speeds
    if (msg->addr == MSG_ESP_19) {
      // sum 4 wheel speeds
      int speed = 0;
      for (uint8_t i = 0U; i < 8U; i += 2U) {
        int wheel_speed = msg->data[i] | (msg->data[i + 1U] << 8);
        speed += wheel_speed;
      }
      // Check all wheel speeds for any movement
      vehicle_moving = speed > 0;
    }

    // Update driver input torque samples
    // Signal: LH_EPS_03.EPS_Lenkmoment (absolute torque)
    // Signal: LH_EPS_03.EPS_VZ_Lenkmoment (direction)
    if (msg->addr == MSG_LH_EPS_03) {
      int torque_driver_new = msg->data[5] | ((msg->data[6] & 0x1FU) << 8);
      int sign = (msg->data[6] & 0x80U) >> 7;
      if (sign == 1) {
        torque_driver_new *= -1;
      }
      update_sample(&torque_driver, torque_driver_new);
    }

    if (msg->addr == MSG_TSK_06) {
      // When using stock ACC, enter controls on rising edge of stock ACC engage, exit on disengage
      // Always exit controls on main switch off
      // Signal: TSK_06.TSK_Status
      int acc_status = (msg->data[3] & 0x7U);
      bool cruise_engaged = (acc_status == 3) || (acc_status == 4) || (acc_status == 5);
      acc_main_on = cruise_engaged || (acc_status == 2);

      if (!volkswagen_longitudinal) {
        pcm_cruise_check(cruise_engaged);
      }

      if (!acc_main_on) {
        controls_allowed = false;
      }
    }

    if (msg->addr == MSG_GRA_ACC_01) {
      // If using openpilot longitudinal, enter controls on falling edge of Set or Resume with main switch on
      // Signal: GRA_ACC_01.GRA_Tip_Setzen
      // Signal: GRA_ACC_01.GRA_Tip_Wiederaufnahme
      if (volkswagen_longitudinal) {
        bool set_button = GET_BIT(msg, 16U);
        bool resume_button = GET_BIT(msg, 19U);
        if ((volkswagen_set_button_prev && !set_button) || (volkswagen_resume_button_prev && !resume_button)) {
          controls_allowed = acc_main_on;
        }
        volkswagen_set_button_prev = set_button;
        volkswagen_resume_button_prev = resume_button;
      }
      // Always exit controls on rising edge of Cancel
      // Signal: GRA_ACC_01.GRA_Abbrechen
      if (GET_BIT(msg, 13U)) {
        controls_allowed = false;
      }
    }

    // Signal: Motor_20.MO_Fahrpedalrohwert_01
    if (msg->addr == MSG_MOTOR_20) {
      gas_pressed = ((GET_BYTES(msg, 0, 4) >> 12) & 0xFFU) != 0U;
    }

    // Signal: Motor_14.MO_Fahrer_bremst (ECU detected brake pedal switch F63)
    if (msg->addr == MSG_MOTOR_14) {
      volkswagen_mqb_brake_pedal_switch = (msg->data[3] & 0x10U) >> 4;
    }

    // Signal: ESP_05.ESP_Fahrer_bremst (ESP detected driver brake pressure above platform specified threshold)
    if (msg->addr == MSG_ESP_05) {
      volkswagen_mqb_brake_pressure_detected = (msg->data[3] & 0x4U) >> 2;
    }

    brake_pressed = volkswagen_mqb_brake_pedal_switch || volkswagen_mqb_brake_pressure_detected;
  }
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
  // Signal: HCA_01.HCA_01_LM_Offset (absolute torque)
  // Signal: HCA_01.HCA_01_LM_OffSign (direction)
  if (msg->addr == MSG_HCA_01) {
    int desired_torque = msg->data[2] | ((msg->data[3] & 0x1U) << 8);
    bool sign = GET_BIT(msg, 31U);
    if (sign) {
      desired_torque *= -1;
    }

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
      desired_accel = ((((msg->data[4] & 0x7U) << 8) | msg->data[3]) * 5U) - 7220U;
    } else {
      // Signal: ACC_07.ACC_Folgebeschl (acceleration in m/s2, scale 0.03, offset -4.6)
      int secondary_accel = (msg->data[4] * 30U) - 4600U;
      violation |= (secondary_accel != 3020);  // enforce always inactive (one increment above max range) at this time
      // Signal: ACC_07.ACC_Sollbeschleunigung_02 (acceleration in m/s2, scale 0.005, offset -7.22)
      desired_accel = (((msg->data[7] << 3) | ((msg->data[6] & 0xE0U) >> 5)) * 5U) - 7220U;
    }

    violation |= longitudinal_accel_checks(desired_accel, VOLKSWAGEN_MQB_LONG_LIMITS);

    if (violation) {
      tx = false;
    }
  }

  // FORCE CANCEL: ensuring that only the cancel button press is sent when controls are off.
  // This avoids unintended engagements while still allowing resume spam
  if ((msg->addr == MSG_GRA_ACC_01) && !controls_allowed) {
    // disallow resume and set: bits 16 and 19
    if ((msg->data[2] & 0x9U) != 0U) {
      tx = false;
    }
  }

  return tx;
}

const safety_hooks volkswagen_mqb_hooks = {
  .init = volkswagen_mqb_init,
  .rx = volkswagen_mqb_rx_hook,
  .tx = volkswagen_mqb_tx_hook,
  .get_counter = volkswagen_mqb_meb_get_counter,
  .get_checksum = volkswagen_mqb_meb_get_checksum,
  .compute_checksum = volkswagen_mqb_meb_compute_crc,
};
