#ifndef SAFETY_MODES_BYD_H
#define SAFETY_MODES_BYD_H

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
  return 0xFFU - (uint32_t)sum;
}

static void byd_rx_hook(const CANPacket_t *msg) {
  // Steering angle: 0.1 deg/LSB, signed
  if (msg_matches(msg, 0x11FU, 0U)) {
    int angle_meas_new = to_signed(GET_BYTES_LE(msg, 0, 2), 16);  // STEER_ANGLE_2
    update_sample(&safety.angle_meas, angle_meas_new);
  }

  // Vehicle speed: 0.1 kph/LSB
  if (msg_matches(msg, 0x1F0U, 0U)) {
    uint32_t speed = GET_BYTES_LE(msg, 0, 2);  // WHEELSPEED_CLEAN
    safety.vehicle_moving = speed > 0U;
    UPDATE_VEHICLE_SPEED((float)speed * 0.1f * KPH_TO_MS);
  }

  // Gas and brake pressed
  if (msg_matches(msg, 0x242U, 0U)) {
    safety.brake_pressed = GET_BIT(msg, 37U);   // BRAKE_PRESSED
    safety.gas_pressed = (msg->data[3] & 0x7FU) > 0U;    // RAW_THROTTLE
  }

  // Cruise state
  if (msg_matches(msg, 0x32DU, 2U)) {
    // ACC_STATE: 0=OFF, 2=ACC_ON, 3=ACC_ACTIVE, 5=FORCE_ACCEL, 7=ERROR
    uint8_t acc_state = (msg->data[2] >> 3) & 0x7U;
    bool acc_on = (acc_state == 3U) || (acc_state == 5U);
    pcm_cruise_check(acc_on);
  }
}

static bool byd_tx_hook(const CANPacket_t *msg) {
  const AngleSteeringLimits BYD_STEERING_LIMITS = {
    .max_angle = 3900,  // 390 deg
    .angle_deg_to_can = 10.0f,
    .frequency = 50U,
  };

  // NOTE: based off BYD_ATTO_3 to match openpilot
  const AngleSteeringParams BYD_STEERING_PARAMS = {
    .slip_factor = -0.0006166479109059387f,  // calc_slip_factor(VM)
    .steer_ratio = 14.8f,
    .wheelbase = 2.72f,
  };

  bool tx = true;

  // Steering control: 0.1 deg/LSB, signed
  if (msg->addr == 0x1E2U) {
    int desired_angle = to_signed(GET_BYTES_LE(msg, 3, 2), 16);  // STEER_ANGLE
    bool steer_req = GET_BIT(msg, 21U);  // STEER_REQ

    if (steer_angle_cmd_checks_vm(desired_angle, steer_req, BYD_STEERING_LIMITS, BYD_STEERING_PARAMS)) {
      tx = false;
    }
  }

  // Only allow cancel (ACC_ON_BTN) while stock cruise is engaged, or button release.
  if (msg->addr == 0x3B0U) {
    bool other_buttons = GET_BIT(msg, 3U) ||   // SET
                         GET_BIT(msg, 4U) ||   // RES
                         GET_BIT(msg, 6U) ||   // LKAS_ON
                         GET_BIT(msg, 15U) ||  // DEC_DISTANCE
                         GET_BIT(msg, 16U);    // INC_DISTANCE
    bool cancel = GET_BIT(msg, 19U);  // ACC_ON_BTN
    tx = !other_buttons && (!cancel || safety.cruise_engaged_prev);
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

  static const RxCheck byd_rx_checks[] = {
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

#endif  // SAFETY_MODES_BYD_H
