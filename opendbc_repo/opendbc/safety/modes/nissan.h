#pragma once

#include "opendbc/safety/declarations.h"

static bool nissan_alt_eps = false;

static void nissan_rx_hook(const CANPacket_t *msg) {

  if (msg->bus == (nissan_alt_eps ? 1U : 0U)) {
    if (msg->addr == 0x2U) {
      // Current steering angle
      // Factor -0.1, little endian
      int angle_meas_new = (GET_BYTES(msg, 0, 4) & 0xFFFFU);
      // Multiply by -10 to match scale of LKAS angle
      angle_meas_new = to_signed(angle_meas_new, 16) * -10;

      // update array of samples
      update_sample(&angle_meas, angle_meas_new);
    }

    if (msg->addr == 0x285U) {
      // Get current speed and standstill
      uint16_t right_rear = (msg->data[0] << 8) | (msg->data[1]);
      uint16_t left_rear = (msg->data[2] << 8) | (msg->data[3]);
      vehicle_moving = (right_rear | left_rear) != 0U;
      UPDATE_VEHICLE_SPEED((right_rear + left_rear) / 2.0 * 0.005 * KPH_TO_MS);
    }

    // X-Trail 0x15c, Leaf 0x239
    if ((msg->addr == 0x15cU) || (msg->addr == 0x239U)) {
      if (msg->addr == 0x15cU){
        gas_pressed = ((msg->data[5] << 2) | ((msg->data[6] >> 6) & 0x3U)) > 3U;
      } else {
        gas_pressed = msg->data[0] > 3U;
      }
    }

    // X-trail 0x454, Leaf 0x239
    if ((msg->addr == 0x454U) || (msg->addr == 0x239U)) {
      if (msg->addr == 0x454U){
        brake_pressed = (msg->data[2] & 0x80U) != 0U;
      } else {
        brake_pressed = ((msg->data[4] >> 5) & 1U) != 0U;
      }
    }
  }

  // Handle cruise enabled
  if ((msg->addr == 0x30fU) && (msg->bus == (nissan_alt_eps ? 1U : 2U))) {
    bool cruise_engaged = (msg->data[0] >> 3) & 1U;
    pcm_cruise_check(cruise_engaged);
  }
}


static bool nissan_tx_hook(const CANPacket_t *msg) {
  const AngleSteeringLimits NISSAN_STEERING_LIMITS = {
    .max_angle = 60000,  // 600 deg, reasonable limit
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

  bool tx = true;
  bool violation = false;

  // steer cmd checks
  if (msg->addr == 0x169U) {
    int desired_angle = ((msg->data[0] << 10) | (msg->data[1] << 2) | ((msg->data[2] >> 6) & 0x3U));
    bool lka_active = (msg->data[6] >> 4) & 1U;

    // Factor is -0.01, offset is 1310. Flip to correct sign, but keep units in CAN scale
    desired_angle = -desired_angle + (1310.0f * NISSAN_STEERING_LIMITS.angle_deg_to_can);

    if (steer_angle_cmd_checks(desired_angle, lka_active, NISSAN_STEERING_LIMITS)) {
      violation = true;
    }
  }

  // acc button check, only allow cancel button to be sent
  if (msg->addr == 0x20bU) {
    // Violation of any button other than cancel is pressed
    violation |= ((msg->data[1] & 0x3dU) > 0U);
  }

  if (violation) {
    tx = false;
  }

  return tx;
}


static safety_config nissan_init(uint16_t param) {
  static const CanMsg NISSAN_TX_MSGS[] = {
    {0x169, 0, 8, .check_relay = true},   // LKAS
    {0x2b1, 0, 8, .check_relay = true},   // PROPILOT_HUD
    {0x4cc, 0, 8, .check_relay = true},   // PROPILOT_HUD_INFO_MSG
    {0x20b, 2, 6, .check_relay = false},  // CRUISE_THROTTLE (X-Trail)
    {0x20b, 1, 6, .check_relay = false},  // CRUISE_THROTTLE (Altima)
    {0x280, 2, 8, .check_relay = true}    // CANCEL_MSG (Leaf)
  };

  // Signals duplicated below due to the fact that these messages can come in on either CAN bus, depending on car model.
  static RxCheck nissan_rx_checks[] = {
    {.msg = {{0x2, 0, 5, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},
             {0x2, 1, 5, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }}},  // STEER_ANGLE_SENSOR
    {.msg = {{0x285, 0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},
             {0x285, 1, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }}}, // WHEEL_SPEEDS_REAR
    {.msg = {{0x30f, 2, 3, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},
             {0x30f, 1, 3, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }}}, // CRUISE_STATE
    {.msg = {{0x15c, 0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},
             {0x15c, 1, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},
             {0x239, 0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}}}, // GAS_PEDAL
    {.msg = {{0x454, 0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},
             {0x454, 1, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true},
             {0x1cc, 0, 4, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}}}, // DOORS_LIGHTS / BRAKE
  };

  // EPS Location. false = V-CAN, true = C-CAN
  const uint16_t NISSAN_PARAM_ALT_EPS_BUS = 1;

  nissan_alt_eps = GET_FLAG(param, NISSAN_PARAM_ALT_EPS_BUS);
  return BUILD_SAFETY_CFG(nissan_rx_checks, NISSAN_TX_MSGS);
}

const safety_hooks nissan_hooks = {
  .init = nissan_init,
  .rx = nissan_rx_hook,
  .tx = nissan_tx_hook,
};
