#pragma once

#include "opendbc/safety/safety_declarations.h"

// Preglobal platform
// 0x161 is ES_CruiseThrottle
// 0x164 is ES_LKAS

#define MSG_SUBARU_PG_CruiseControl         0x144U
#define MSG_SUBARU_PG_Throttle              0x140U
#define MSG_SUBARU_PG_Wheel_Speeds          0xD4U
#define MSG_SUBARU_PG_Brake_Pedal           0xD1U
#define MSG_SUBARU_PG_ES_LKAS               0x164U
#define MSG_SUBARU_PG_ES_Distance           0x161U
#define MSG_SUBARU_PG_Steering_Torque       0x371U

#define SUBARU_PG_MAIN_BUS 0U
#define SUBARU_PG_CAM_BUS  2U

static bool subaru_pg_reversed_driver_torque = false;

static void subaru_preglobal_rx_hook(const CANPacket_t *msg) {
  if (msg->bus == SUBARU_PG_MAIN_BUS) {
    if (msg->addr == MSG_SUBARU_PG_Steering_Torque) {
      int torque_driver_new;
      torque_driver_new = (msg->data[3] >> 5) + (msg->data[4] << 3);
      torque_driver_new = to_signed(torque_driver_new, 11);
      torque_driver_new = subaru_pg_reversed_driver_torque ? -torque_driver_new : torque_driver_new;
      update_sample(&torque_driver, torque_driver_new);
    }

    // enter controls on rising edge of ACC, exit controls on ACC off
    if (msg->addr == MSG_SUBARU_PG_CruiseControl) {
      bool cruise_engaged = (msg->data[6] >> 1) & 1U;
      pcm_cruise_check(cruise_engaged);
    }

    // update vehicle moving with any non-zero wheel speed
    if (msg->addr == MSG_SUBARU_PG_Wheel_Speeds) {
      vehicle_moving = ((GET_BYTES(msg, 0, 4) >> 12) != 0U) || (GET_BYTES(msg, 4, 4) != 0U);
    }

    if (msg->addr == MSG_SUBARU_PG_Brake_Pedal) {
      brake_pressed = ((GET_BYTES(msg, 0, 4) >> 16) & 0xFFU) > 0U;
    }

    if (msg->addr == MSG_SUBARU_PG_Throttle) {
      gas_pressed = msg->data[0] != 0U;
    }
  }
}

static bool subaru_preglobal_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits SUBARU_PG_STEERING_LIMITS = {
    .max_torque = 2047,
    .max_rt_delta = 940,
    .max_rate_up = 50,
    .max_rate_down = 70,
    .driver_torque_multiplier = 10,
    .driver_torque_allowance = 75,
    .type = TorqueDriverLimited,
  };

  bool tx = true;

  // steer cmd checks
  if (msg->addr == MSG_SUBARU_PG_ES_LKAS) {
    int desired_torque = ((GET_BYTES(msg, 0, 4) >> 8) & 0x1FFFU);
    desired_torque = -1 * to_signed(desired_torque, 13);

    bool steer_req = (msg->data[3] >> 0) & 1U;

    if (steer_torque_cmd_checks(desired_torque, steer_req, SUBARU_PG_STEERING_LIMITS)) {
      tx = false;
    }
  }
  return tx;
}

static safety_config subaru_preglobal_init(uint16_t param) {
  static const CanMsg SUBARU_PG_TX_MSGS[] = {
    {MSG_SUBARU_PG_ES_Distance, SUBARU_PG_MAIN_BUS, 8, .check_relay = true},
    {MSG_SUBARU_PG_ES_LKAS,     SUBARU_PG_MAIN_BUS, 8, .check_relay = true}
  };

  // TODO: do checksum and counter checks after adding the signals to the outback dbc file
  static RxCheck subaru_preglobal_rx_checks[] = {
    {.msg = {{MSG_SUBARU_PG_Throttle,        SUBARU_PG_MAIN_BUS, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_SUBARU_PG_Steering_Torque, SUBARU_PG_MAIN_BUS, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_SUBARU_PG_CruiseControl,   SUBARU_PG_MAIN_BUS, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_SUBARU_PG_Wheel_Speeds,    SUBARU_PG_MAIN_BUS, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{MSG_SUBARU_PG_Brake_Pedal,     SUBARU_PG_MAIN_BUS, 4, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  const int SUBARU_PG_PARAM_REVERSED_DRIVER_TORQUE = 4;

  subaru_pg_reversed_driver_torque = GET_FLAG(param, SUBARU_PG_PARAM_REVERSED_DRIVER_TORQUE);
  return BUILD_SAFETY_CFG(subaru_preglobal_rx_checks, SUBARU_PG_TX_MSGS);
}

const safety_hooks subaru_preglobal_hooks = {
  .init = subaru_preglobal_init,
  .rx = subaru_preglobal_rx_hook,
  .tx = subaru_preglobal_tx_hook,
};
