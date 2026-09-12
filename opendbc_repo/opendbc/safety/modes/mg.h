#pragma once

#include "opendbc/safety/declarations.h"

static uint8_t mg_crc_lut[256];

static uint32_t mg_get_checksum(const CANPacket_t *msg) {
  return msg->data[7];
}

static uint32_t mg_compute_checksum(const CANPacket_t *msg) {
  uint8_t checksum = 0xFFU;
  for (int i = 0; i < 7; i++) {
    checksum ^= msg->data[i];
    checksum = mg_crc_lut[checksum];
  }
  return checksum ^ 0xFFU;
}

static uint8_t mg_get_counter(const CANPacket_t *msg) {
  uint8_t counter = 0U;
  if (msg->addr == 0x23cU) {
    counter = (msg->data[4] >> 4) & 0xFU;
  } else if (msg->addr == 0x1ecU) {
    counter = (msg->data[0] >> 4) & 0xFU;
  } else if (msg->addr == 0x242U) {
    counter = (msg->data[0] >> 3) & 0xFU;
  } else if (msg->addr == 0x1b6U) {
    counter = msg->data[6] & 0xFU;
  } else {
    // No counter for this message
  }
  return counter;
}

static void mg_rx_hook(const CANPacket_t *msg) {
  if (msg->bus == 0U)  {
    // Vehicle speed
    if (msg->addr == 0x23cU) {
      float speed = (((msg->data[2] & 0x7FU) << 8) | msg->data[3]) * 0.015625;
      vehicle_moving = speed > 0.0;
      UPDATE_VEHICLE_SPEED(speed * KPH_TO_MS);
    }

    // Gas pressed
    if (msg->addr == 0xafU) {
      gas_pressed = msg->data[0] != 0U;
    }

    // Driver torque
    if (msg->addr == 0x1ecU) {
      int torque_driver_new = (((msg->data[4] & 0x7U) << 8) | msg->data[5]) - 1024U;
      update_sample(&torque_driver, torque_driver_new);
    }

    // Brake pressed
    if (msg->addr == 0x1b6U) {
      brake_pressed = GET_BIT(msg, 10U);
    }

    // Cruise state
    if (msg->addr == 0x242U) {
      int cruise_state = (msg->data[5] & 0x38U) >> 3;
      bool cruise_engaged = (cruise_state == 2) ||  // Active
                            (cruise_state == 3);    // Override
      pcm_cruise_check(cruise_engaged);
    }
  }
}

static bool mg_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits MG_STEERING_LIMITS = {
    .max_torque = 300,
    .max_rate_up = 6,
    .max_rate_down = 10,
    .max_rt_delta = 125,
    .driver_torque_multiplier = 2,
    .driver_torque_allowance = 100,
    .type = TorqueDriverLimited,
  };

  bool tx = true;
  bool violation = false;

  // Steering control
  if (msg->addr == 0x1fdU) {
    int desired_torque = (((msg->data[0] & 0x7U) << 8) | msg->data[1]) - 1024U;
    bool steer_req = GET_BIT(msg, 35U);

    violation |= steer_torque_cmd_checks(desired_torque, steer_req, MG_STEERING_LIMITS);
  }

  if (violation) {
    tx = false;
  }

  return tx;
}

static safety_config mg_init(uint16_t param) {
  SAFETY_UNUSED(param);

  gen_crc_lookup_table_8(0x1DU, mg_crc_lut);

  static const CanMsg MG_TX_MSGS[] = {{0x1fd, 0, 8, .check_relay = true}};

  static RxCheck mg_rx_checks[] = {
    {.msg = {{0x23c, 0, 8, .frequency = 50U,  .ignore_checksum = true, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // SCS_HSC2_FrP19 (speed)
    {.msg = {{0xaf,  0, 8, .frequency = 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // GW_HSC2_HCU_FrP00 (gas pedal)
    {.msg = {{0x1ec, 0, 8, .frequency = 50U,  .ignore_checksum = true, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // EPS_HSC2_FrP03 (driver torque)
    {.msg = {{0x242, 0, 8, .frequency = 50U,  .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // RADAR_HSC2_FrP00 (cruise state)
    {.msg = {{0x1b6, 0, 8, .frequency = 50U,  .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // EHBS_HSC2_FrP00 (brake pedal)
  };

  safety_config ret;
  SET_RX_CHECKS(mg_rx_checks, ret);
  SET_TX_MSGS(MG_TX_MSGS, ret);
  return ret;
}

const safety_hooks mg_hooks = {
  .init = mg_init,
  .rx = mg_rx_hook,
  .tx = mg_tx_hook,
  .get_checksum = mg_get_checksum,
  .compute_checksum = mg_compute_checksum,
  .get_counter = mg_get_counter,
};
