#pragma once

#include "opendbc/safety/declarations.h"
#include "opendbc/safety/modes/volkswagen_common.h"

#define MSG_ESC_51           0xFCU    // RX, for wheel speeds
#define MSG_ACC_18           0x14DU   // TX by OP, ACC control instructions to the drivetrain coordinator
#define MSG_ESP_21           0xFDU    // RX, redundant vehicle speed source
#define MSG_HCA_03           0x303U
#define MSG_ACC_19           0x300U   // TX by OP, ACC HUD data to the instrument cluster
#define MSG_QFK_01           0x13DU
#define MSG_Motor_51         0x10BU   // RX for TSK state and accel pedal
#define MSG_KLR_01           0x25DU   // TX, for capacitive steering wheel
#define MSG_TA_01            0x26BU   // TX by OP, Travel Assist status

static bool volkswagen_meb_alt_crc = false;

#define VOLKSWAGEN_MEB_COMMON_RX_CHECKS \
  {.msg = {{MSG_LH_EPS_03, 0, 8, 100U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
  {.msg = {{MSG_MOTOR_14, 0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
  {.msg = {{MSG_GRA_ACC_01, 0, 8, 33U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
  {.msg = {{MSG_QFK_01, 0, 32, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, \
  {.msg = {{MSG_ESP_21, 0, 8, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},

static uint32_t volkswagen_meb_compute_crc(const CANPacket_t *msg) {
  int len = GET_LEN(msg);

  uint8_t crc = 0xFFU;
  for (int i = 1; i < len; i++) {
    crc ^= (uint8_t)msg->data[i];
    crc = volkswagen_crc8_lut_8h2f[crc];
  }

  uint8_t counter = volkswagen_mqb_meb_get_counter(msg);
  if (msg->addr == MSG_LH_EPS_03) {
    crc ^= (uint8_t[]){0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5}[counter];
  } else if (msg->addr == MSG_GRA_ACC_01) {
    crc ^= (uint8_t[]){0x6A, 0x38, 0xB4, 0x27, 0x22, 0xEF, 0xE1, 0xBB, 0xF8, 0x80, 0x84, 0x49, 0xC7, 0x9E, 0x1E, 0x2B}[counter];
  } else if (msg->addr == MSG_QFK_01) {
    crc ^= (uint8_t[]){0x20, 0xCA, 0x68, 0xD5, 0x1B, 0x31, 0xE2, 0xDA, 0x08, 0x0A, 0xD4, 0xDE, 0x9C, 0xE4, 0x35, 0x5B}[counter];
  } else if (msg->addr == MSG_ESC_51) {
    crc ^= (uint8_t[]){0x77, 0x5C, 0xA0, 0x89, 0x4B, 0x7C, 0xBB, 0xD6, 0x1F, 0x6C, 0x4F, 0xF6, 0x20, 0x2B, 0x43, 0xDD}[counter];
  } else if (msg->addr == MSG_ESP_21) {
    crc ^= (uint8_t[]){0xB4, 0xEF, 0xF8, 0x49, 0x1E, 0xE5, 0xC2, 0xC0, 0x97, 0x19, 0x3C, 0xC9, 0xF1, 0x98, 0xD6, 0x61}[counter];
  } else if (msg->addr == MSG_Motor_51) {
    crc ^= (uint8_t[]){0x77, 0x5C, 0xA0, 0x89, 0x4B, 0x7C, 0xBB, 0xD6, 0x1F, 0x6C, 0x4F, 0xF6, 0x20, 0x2B, 0x43, 0xDD}[counter];
  } else {
    // Undefined CAN message, CRC check expected to fail
  }
  crc = volkswagen_crc8_lut_8h2f[crc];

  return (uint8_t)(crc ^ 0xFFU);
}

static uint32_t volkswagen_meb_alt_crc_compute(const CANPacket_t *msg) {
  uint32_t ret = volkswagen_meb_compute_crc(msg);
  int len = 0;
  if (volkswagen_meb_alt_crc) {
    if (msg->addr == MSG_QFK_01) {
      len = 28;
    } else if (msg->addr == MSG_ESC_51) {
      len = 60;
    } else if (msg->addr == MSG_Motor_51) {
      len = 44;
    } else {
      len = 0;
    }
  }

  if (len > 0) {
    uint8_t crc = 0xFFU;
    for (int i = 1; i < len; i++) {
      crc ^= (uint8_t)msg->data[i];
      crc = volkswagen_crc8_lut_8h2f[crc];
    }

    uint8_t counter = volkswagen_mqb_meb_get_counter(msg);
    if (msg->addr == MSG_QFK_01) {
      crc ^= (uint8_t[]){0x18, 0x71, 0x10, 0x8D, 0xD7, 0xAA, 0xB0, 0x78, 0xAC, 0x12, 0xAE, 0x0C, 0xDD, 0xF1, 0x85, 0x68}[counter];
    } else if (msg->addr == MSG_ESC_51) {
      crc ^= (uint8_t[]){0x69, 0xDC, 0xF9, 0x64, 0x6A, 0xCE, 0x55, 0x2C, 0xC4, 0x38, 0x8F, 0xD1, 0xC6, 0x43, 0xB4, 0xB1}[counter];
    } else {
      crc ^= (uint8_t[]){0x2C, 0xB1, 0x1A, 0x75, 0xBB, 0x65, 0x79, 0x47, 0x81, 0x2B, 0xCC, 0x96, 0x17, 0xDB, 0xC0, 0x94}[counter];
    }

    crc = (uint8_t)(volkswagen_crc8_lut_8h2f[crc] ^ 0xFFU);
    if (crc == msg->data[0]) {
      ret = crc;
    }
  }
  return ret;
}

static safety_config volkswagen_meb_init(uint16_t param) {
  // Transmit of GRA_ACC_01 is allowed on bus 0 and 2 to keep compatibility with gateway and camera integration
  static const CanMsg VOLKSWAGEN_MEB_STOCK_TX_MSGS[] = {
    {MSG_HCA_03, 0, 24, .check_relay = true},
    {MSG_GRA_ACC_01, 0, 8, .check_relay = false},
    {MSG_GRA_ACC_01, 2, 8, .check_relay = false},
    {MSG_LDW_02, 0, 8, .check_relay = true},
    {MSG_KLR_01, 0, 8, .check_relay = false},
    {MSG_KLR_01, 2, 8, .check_relay = true},
  };

  static const CanMsg VOLKSWAGEN_MEB_LONG_TX_MSGS[] = {
    {MSG_HCA_03, 0, 24, .check_relay = true},
    {MSG_LDW_02, 0, 8, .check_relay = true},
    {MSG_KLR_01, 0, 8, .check_relay = false},
    {MSG_KLR_01, 2, 8, .check_relay = true},
    {MSG_ACC_19, 0, 48, .check_relay = true},
    {MSG_ACC_18, 0, 32, .check_relay = true},
    {MSG_TA_01, 0, 8, .check_relay = true},
  };

  static RxCheck volkswagen_meb_rx_checks[] = {
    VOLKSWAGEN_MEB_COMMON_RX_CHECKS
    {.msg = {{MSG_Motor_51, 0, 32, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_ESC_51, 0, 48, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  static RxCheck volkswagen_meb_gen2_rx_checks[] = {
    VOLKSWAGEN_MEB_COMMON_RX_CHECKS
    {.msg = {{MSG_Motor_51, 0, 48, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_ESC_51, 0, 64, 50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  volkswagen_common_init();
  const uint16_t FLAG_VOLKSWAGEN_MEB_ALT_CRC = 2;
  volkswagen_meb_alt_crc = GET_FLAG(param, FLAG_VOLKSWAGEN_MEB_ALT_CRC);

#ifdef ALLOW_DEBUG
  volkswagen_longitudinal = GET_FLAG(param, FLAG_VOLKSWAGEN_LONG_CONTROL);
#endif

  safety_config ret;
  if (volkswagen_longitudinal && volkswagen_meb_alt_crc) {
    ret = BUILD_SAFETY_CFG(volkswagen_meb_gen2_rx_checks, VOLKSWAGEN_MEB_LONG_TX_MSGS);
  } else if (volkswagen_longitudinal) {
    ret = BUILD_SAFETY_CFG(volkswagen_meb_rx_checks, VOLKSWAGEN_MEB_LONG_TX_MSGS);
  } else if (volkswagen_meb_alt_crc) {
    ret = BUILD_SAFETY_CFG(volkswagen_meb_gen2_rx_checks, VOLKSWAGEN_MEB_STOCK_TX_MSGS);
  } else {
    ret = BUILD_SAFETY_CFG(volkswagen_meb_rx_checks, VOLKSWAGEN_MEB_STOCK_TX_MSGS);
  }
  return ret;
}

static void volkswagen_meb_rx_hook(const CANPacket_t *msg) {
  if (msg->bus == 0U) {
    // Update in-motion state by sampling wheel speeds
    if (msg->addr == MSG_ESC_51) {
      uint32_t fl = msg->data[8] | (msg->data[9] << 8);
      uint32_t fr = msg->data[10] | (msg->data[11] << 8);
      uint32_t rl = msg->data[12] | (msg->data[13] << 8);
      uint32_t rr = msg->data[14] | (msg->data[15] << 8);
      vehicle_moving = (fr > 0U) || (rr > 0U) || (rl > 0U) || (fl > 0U);
      UPDATE_VEHICLE_SPEED((fr + rr + rl + fl) / 4.0 * 0.0075 * KPH_TO_MS);
    }

    // Check vehicle speed with redundant source
    if (msg->addr == MSG_ESP_21) {
      // Signal: ESP_v_Signal
      float esp_speed = ((msg->data[5] << 8) | msg->data[4]) * 0.01 * KPH_TO_MS;
      UPDATE_VEHICLE_SPEED_2(esp_speed);
    }

    if (msg->addr == MSG_QFK_01) {
      int current_curvature = ((msg->data[6] & 0x7FU) << 8) | msg->data[5];
      current_curvature *= GET_BIT(msg, 55U) ? 1 : -1;
      update_sample(&curvature_state.meas, current_curvature);
    }

    if (msg->addr == MSG_LH_EPS_03) {
      update_sample(&torque_driver, volkswagen_mlb_mqb_driver_input_torque(msg));
    }

    if (msg->addr == MSG_Motor_51) {
      int acc_status = (msg->data[11] & 0x07U);
      bool cruise_engaged = (acc_status == 3) || (acc_status == 4) || (acc_status == 5);
      acc_main_on = cruise_engaged || (acc_status == 2);

      if (!volkswagen_longitudinal) {
        pcm_cruise_check(cruise_engaged);
      }

      if (!acc_main_on) {
        controls_allowed = false;
      }

      int accel_pedal_value = ((msg->data[1] >> 4) & 0x0FU) | ((msg->data[2] & 0x1FU) << 4);
      gas_pressed = accel_pedal_value > 0;
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
      if (GET_BIT(msg, 13U)) {
        controls_allowed = false;
      }
    }

    if (msg->addr == MSG_MOTOR_14) {
      brake_pressed = GET_BIT(msg, 28U);
    }
  }
}

static bool volkswagen_meb_tx_hook(const CANPacket_t *msg) {
  // acceleration in m/s2 * 1000 to avoid floating point math
  const LongitudinalLimits VOLKSWAGEN_MEB_LONG_LIMITS = {
    .max_accel = 2000,
    .min_accel = -3500,
    .inactive_accel = 3010,  // VW sends one increment above the max range when inactive
  };

  bool tx = true;

  // Safety check for MSG_ACC_18 acceleration requests
  if (msg->addr == MSG_ACC_18) {
    // Signal: ACC_18.ACC_Sollbeschleunigung_02 (acceleration in m/s2, scale 0.005, offset -7.22)
    int desired_accel = ((((msg->data[4] & 0x7U) << 8) | msg->data[3]) * 5U) - 7220U;
    // allow ACCEL_OVERRIDE (0) while controls are allowed even when the driver is on the gas
    bool accel_override = controls_allowed && (desired_accel == 0);
    if (!accel_override && longitudinal_accel_checks(desired_accel, VOLKSWAGEN_MEB_LONG_LIMITS)) {
      tx = false;
    }
  }

  if (msg->addr == MSG_HCA_03) {
    const CurvatureSteeringLimits VOLKSWAGEN_MEB_STEERING_LIMITS = {
      .max_curvature = 29105,
      .curvature_to_can = 149253.7313f,
      .frequency = 50,                   // Hz
      .max_curvature_error = 0,          // disabled, MEB doesn't track rack
      .curvature_error_min_speed = 0.0,  // disabled
      .max_steer_power = 125,
      .inactive_curvature_is_zero = false, // MEB winds down with measured curvature
    };

    int desired_curvature_raw = GET_BYTES(msg, 3, 2) & 0x7FFFU;

    bool desired_curvature_sign = GET_BIT(msg, 39U);
    if (!desired_curvature_sign) {
      desired_curvature_raw *= -1;
    }

    bool steer_req = (((msg->data[1] >> 4) & 0x0FU) == 4U);
    int steer_power = msg->data[2];

    if (steer_curvature_cmd_checks(desired_curvature_raw, steer_power, steer_req, VOLKSWAGEN_MEB_STEERING_LIMITS)) {
      tx = false;
    }
  }

  if ((msg->addr == MSG_GRA_ACC_01) && !controls_allowed) {
    // only allow cancel button: bit 13
    if (!GET_BIT(msg, 13U)) {
      tx = false;
    }
  }

  return tx;
}

const safety_hooks volkswagen_meb_hooks = {
  .init = volkswagen_meb_init,
  .rx = volkswagen_meb_rx_hook,
  .tx = volkswagen_meb_tx_hook,
  .get_counter = volkswagen_mqb_meb_get_counter,
  .get_checksum = volkswagen_mqb_meb_get_checksum,
  .compute_checksum = volkswagen_meb_alt_crc_compute,
};
