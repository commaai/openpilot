#ifndef SAFETY_MODES_MAZDA_H
#define SAFETY_MODES_MAZDA_H

#include "opendbc/safety/declarations.h"

// CAN msgs we care about
#define MAZDA_LKAS          0x243U
#define MAZDA_LKAS_HUD      0x440U
#define MAZDA_CRZ_CTRL      0x21cU
#define MAZDA_CRZ_BTNS      0x09dU
#define MAZDA_STEER_TORQUE  0x240U
#define MAZDA_ENGINE_DATA   0x202U
#define MAZDA_PEDALS        0x165U

// CAN bus numbers
#define MAZDA_MAIN 0

// track msgs coming from OP so that we know what CAM msgs to drop and what to forward
static void mazda_rx_hook(const CANPacket_t *msg) {
  if (msg_matches(msg, MAZDA_ENGINE_DATA, MAZDA_MAIN)) {
    // sample speed: scale by 0.01 to get kph
    uint32_t speed = GET_BYTES_BE(msg, 2, 2);
    safety.vehicle_moving = speed > 10U;  // moving when speed > 0.1 kph
  }

  if (msg_matches(msg, MAZDA_STEER_TORQUE, MAZDA_MAIN)) {
    int torque_driver_new = (int)msg->data[0] - 127;
    // update array of samples
    update_sample(&safety.torque_driver, torque_driver_new);
  }

  // enter controls on rising edge of ACC, exit controls on ACC off
  if (msg_matches(msg, MAZDA_CRZ_CTRL, MAZDA_MAIN)) {
    bool cruise_engaged = GET_BIT(msg, 3U);
    pcm_cruise_check(cruise_engaged);
  }

  if (msg_matches(msg, MAZDA_ENGINE_DATA, MAZDA_MAIN)) {
    safety.gas_pressed = (msg->data[4] | (msg->data[5] & 0xF0U)) != 0U;
  }

  if (msg_matches(msg, MAZDA_PEDALS, MAZDA_MAIN)) {
    safety.brake_pressed = GET_BIT(msg, 4U);
  }
}

static bool mazda_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits MAZDA_STEERING_LIMITS = {
    .max_torque = 800,
    .max_rate_up = 10,
    .max_rate_down = 25,
    .max_rt_delta = 300,
    .driver_torque_multiplier = 1,
    .driver_torque_allowance = 15,
    .type = TorqueDriverLimited,
  };

  bool tx = true;
  // Check if msg is sent on the main BUS
  // steer cmd checks
  if (msg_matches(msg, MAZDA_LKAS, MAZDA_MAIN)) {
    int desired_torque = from_offset(GET_BYTES_BE(msg, 0, 2) & 0xFFFU, 2048);

    if (steer_torque_cmd_checks(desired_torque, true, MAZDA_STEERING_LIMITS)) {
      tx = false;
    }
  }

  // cruise buttons check
  if (msg_matches(msg, MAZDA_CRZ_BTNS, MAZDA_MAIN)) {
    // allow resume spamming while controls allowed, but
    // only allow cancel while controls not allowed
    bool cancel_cmd = (msg->data[0] == 0x1U);
    if (!safety.controls_allowed && !cancel_cmd) {
      tx = false;
    }
  }

  return tx;
}

static safety_config mazda_init(uint16_t param) {
  static const CanMsg MAZDA_TX_MSGS[] = {
    {MAZDA_LKAS, 0, 8, .check_relay = true},
    {MAZDA_CRZ_BTNS, 0, 8, .check_relay = false},
    {MAZDA_LKAS_HUD, 0, 8, .check_relay = true},
  };

  static const RxCheck mazda_rx_checks[] = {
    {.msg = {{MAZDA_CRZ_CTRL,     0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_CRZ_BTNS,     0, 8, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_STEER_TORQUE, 0, 8, 83U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_ENGINE_DATA,  0, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MAZDA_PEDALS,       0, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  SAFETY_UNUSED(param);
  return BUILD_SAFETY_CFG(mazda_rx_checks, MAZDA_TX_MSGS);
}

static const safety_hooks mazda_hooks = {
  .init = mazda_init,
  .rx = mazda_rx_hook,
  .tx = mazda_tx_hook,
};

#endif  // SAFETY_MODES_MAZDA_H
