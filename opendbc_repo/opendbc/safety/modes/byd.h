#pragma once

#include "opendbc/safety/declarations.h"

static uint8_t byd_get_counter(const CANPacket_t *msg) {
  // Speed uses the high nibble; cruise status uses the low nibble.
  return (msg->addr == 0x1F0U) ? (msg->data[6] >> 4) : (msg->data[6] & 0xFU);
}

static uint32_t byd_get_checksum(const CANPacket_t *msg) {
  return msg->data[7];
}

static uint32_t byd_compute_checksum(const CANPacket_t *msg) {
  uint8_t sum = 0;
  for (int i = 0; i < 7; i++) {
    sum += msg->data[i];
  }
  return 0xFFU - sum;
}

static void byd_rx_hook(const CANPacket_t *msg) {

  if (msg->bus == 0U) {
    // Steering angle: 0.1 deg/LSB, signed
    if (msg->addr == 0x11FU) {
      int angle_meas_new = to_signed((msg->data[1] << 8) | msg->data[0], 16);  // STEER_ANGLE_2
      update_sample(&angle_meas, angle_meas_new);
    }

    // Vehicle speed: 0.1 kph/LSB
    if (msg->addr == 0x1F0U) {
      int speed = (msg->data[1] << 8) | msg->data[0];  // WHEELSPEED_CLEAN
      vehicle_moving = speed > 0;
      UPDATE_VEHICLE_SPEED(speed * 0.1 * KPH_TO_MS);
    }

    // Gas and brake pressed
    if (msg->addr == 0x242U) {
      brake_pressed = (msg->data[4] >> 5) & 0x1U;   // BRAKE_PRESSED
      gas_pressed = (msg->data[3] & 0x7FU) > 0U;    // RAW_THROTTLE
    }
  }

  if (msg->bus == 2U) {
    // Cruise state
    if (msg->addr == 0x32DU) {
      // ACC_STATE: 0=OFF, 2=ACC_ON, 3=ACC_ACTIVE, 5=FORCE_ACCEL, 7=ERROR
      uint8_t acc_state = (msg->data[2] >> 3) & 0x7U;
      bool acc_on = (acc_state == 3U) || (acc_state == 5U);
      pcm_cruise_check(acc_on);
    }
  }
}


static bool byd_tx_hook(const CANPacket_t *msg) {
  const AngleSteeringLimits BYD_STEERING_LIMITS = {
    .max_angle = 3900,  // 390 deg
    .angle_deg_to_can = 10,
    .frequency = 50U,
  };

  // NOTE: based off BYD_ATTO_3 to match openpilot
  const AngleSteeringParams BYD_STEERING_PARAMS = {
    .slip_factor = -0.0006166479109059387,  // calc_slip_factor(VM)
    .steer_ratio = 14.8,
    .wheelbase = 2.72,
  };

  bool tx = true;

  // Steering control: 0.1 deg/LSB, signed
  if (msg->addr == 0x1E2U) {
    int desired_angle = to_signed((msg->data[4] << 8) | msg->data[3], 16);  // STEER_ANGLE
    bool steer_req = ((msg->data[2] >> 5) & 0x1U) != 0U;                    // STEER_REQ

    if (steer_angle_cmd_checks_vm(desired_angle, steer_req, BYD_STEERING_LIMITS, BYD_STEERING_PARAMS)) {
      tx = false;
    }
  }

  // Only allow cancel (ACC_ON_BTN) while stock cruise is engaged, or button release.
  if (msg->addr == 0x3B0U) {
    bool other_buttons = ((msg->data[0] & 0x58U) != 0U) ||  // SET, RES, LKAS_ON
                         ((msg->data[1] & 0x80U) != 0U) ||  // DEC_DISTANCE
                         ((msg->data[2] & 0x1U) != 0U);     // INC_DISTANCE
    bool cancel = (msg->data[2] & 0x8U) != 0U;             // ACC_ON_BTN
    tx = !other_buttons && (!cancel || cruise_engaged_prev);
  }

  return tx;
}

static safety_config byd_init(uint16_t param) {
  SAFETY_UNUSED(param);

  static const CanMsg BYD_TX_MSGS[] = {
    {0x1E2, 0, 8, .check_relay = true},   // STEERING_MODULE_ADAS (lateral steering command)
    {0x316, 0, 8, .check_relay = true},   // LKAS_HUD_ADAS (dash HUD)
    {0x3B0, 0, 8, .check_relay = false},  // PCM_BUTTONS (cruise cancel button spoof)
  };

  static RxCheck byd_rx_checks[] = {
    {.msg = {{0x11F, 0, 5, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},  // STEER_MODULE_2 (steering angle)
    {.msg = {{0x1F0, 0, 8,  50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  // WHEELSPEED_CLEAN (vehicle speed)
    {.msg = {{0x242, 0, 8,  50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},  // DRIVE_STATE (gas and brake pressed)
    {.msg = {{0x32D, 2, 8,  50U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},  // ACC_HUD_ADAS (cruise state)
  };

  return BUILD_SAFETY_CFG(byd_rx_checks, BYD_TX_MSGS);
}

const safety_hooks byd_hooks = {
  .init = byd_init,
  .rx = byd_rx_hook,
  .tx = byd_tx_hook,
  .get_counter = byd_get_counter,
  .get_checksum = byd_get_checksum,
  .compute_checksum = byd_compute_checksum,
};
