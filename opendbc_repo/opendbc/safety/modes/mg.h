#pragma once

#include "opendbc/safety/declarations.h"

static uint32_t mg_get_checksum(const CANPacket_t *msg) {
  uint32_t checksum = msg->data[7];
  if (msg->addr == 0x1ecU) {
    checksum = ((msg->data[2] & 0x7FU) << 8) | msg->data[3];
  }
  return checksum;
}

static uint32_t mg_compute_checksum(const CANPacket_t *msg) {
  // CRC-8 lookup table for poly 0x1D (SAE J1850)
  // generated with: python3 -c 'f=lambda c:[c:=((c<<1)^0x1D)&0xFF if c&0x80 else (c<<1)&0xFF for _ in range(8)][-1];print(",\n".join("    "+", ".join(f"0x{f(i):02X}U" for i in range(r,r+16)) for r in range(0,256,16)))'
  static const uint8_t mg_crc_lut[256] = {
    0x00U, 0x1DU, 0x3AU, 0x27U, 0x74U, 0x69U, 0x4EU, 0x53U, 0xE8U, 0xF5U, 0xD2U, 0xCFU, 0x9CU, 0x81U, 0xA6U, 0xBBU,
    0xCDU, 0xD0U, 0xF7U, 0xEAU, 0xB9U, 0xA4U, 0x83U, 0x9EU, 0x25U, 0x38U, 0x1FU, 0x02U, 0x51U, 0x4CU, 0x6BU, 0x76U,
    0x87U, 0x9AU, 0xBDU, 0xA0U, 0xF3U, 0xEEU, 0xC9U, 0xD4U, 0x6FU, 0x72U, 0x55U, 0x48U, 0x1BU, 0x06U, 0x21U, 0x3CU,
    0x4AU, 0x57U, 0x70U, 0x6DU, 0x3EU, 0x23U, 0x04U, 0x19U, 0xA2U, 0xBFU, 0x98U, 0x85U, 0xD6U, 0xCBU, 0xECU, 0xF1U,
    0x13U, 0x0EU, 0x29U, 0x34U, 0x67U, 0x7AU, 0x5DU, 0x40U, 0xFBU, 0xE6U, 0xC1U, 0xDCU, 0x8FU, 0x92U, 0xB5U, 0xA8U,
    0xDEU, 0xC3U, 0xE4U, 0xF9U, 0xAAU, 0xB7U, 0x90U, 0x8DU, 0x36U, 0x2BU, 0x0CU, 0x11U, 0x42U, 0x5FU, 0x78U, 0x65U,
    0x94U, 0x89U, 0xAEU, 0xB3U, 0xE0U, 0xFDU, 0xDAU, 0xC7U, 0x7CU, 0x61U, 0x46U, 0x5BU, 0x08U, 0x15U, 0x32U, 0x2FU,
    0x59U, 0x44U, 0x63U, 0x7EU, 0x2DU, 0x30U, 0x17U, 0x0AU, 0xB1U, 0xACU, 0x8BU, 0x96U, 0xC5U, 0xD8U, 0xFFU, 0xE2U,
    0x26U, 0x3BU, 0x1CU, 0x01U, 0x52U, 0x4FU, 0x68U, 0x75U, 0xCEU, 0xD3U, 0xF4U, 0xE9U, 0xBAU, 0xA7U, 0x80U, 0x9DU,
    0xEBU, 0xF6U, 0xD1U, 0xCCU, 0x9FU, 0x82U, 0xA5U, 0xB8U, 0x03U, 0x1EU, 0x39U, 0x24U, 0x77U, 0x6AU, 0x4DU, 0x50U,
    0xA1U, 0xBCU, 0x9BU, 0x86U, 0xD5U, 0xC8U, 0xEFU, 0xF2U, 0x49U, 0x54U, 0x73U, 0x6EU, 0x3DU, 0x20U, 0x07U, 0x1AU,
    0x6CU, 0x71U, 0x56U, 0x4BU, 0x18U, 0x05U, 0x22U, 0x3FU, 0x84U, 0x99U, 0xBEU, 0xA3U, 0xF0U, 0xEDU, 0xCAU, 0xD7U,
    0x35U, 0x28U, 0x0FU, 0x12U, 0x41U, 0x5CU, 0x7BU, 0x66U, 0xDDU, 0xC0U, 0xE7U, 0xFAU, 0xA9U, 0xB4U, 0x93U, 0x8EU,
    0xF8U, 0xE5U, 0xC2U, 0xDFU, 0x8CU, 0x91U, 0xB6U, 0xABU, 0x10U, 0x0DU, 0x2AU, 0x37U, 0x64U, 0x79U, 0x5EU, 0x43U,
    0xB2U, 0xAFU, 0x88U, 0x95U, 0xC6U, 0xDBU, 0xFCU, 0xE1U, 0x5AU, 0x47U, 0x60U, 0x7DU, 0x2EU, 0x33U, 0x14U, 0x09U,
    0x7FU, 0x62U, 0x45U, 0x58U, 0x0BU, 0x16U, 0x31U, 0x2CU, 0x97U, 0x8AU, 0xADU, 0xB0U, 0xE3U, 0xFEU, 0xD9U, 0xC4U
  };
  uint32_t checksum;
  if (msg->addr == 0x1ecU) {
    uint32_t torque = ((msg->data[0] & 0x7U) << 8) | msg->data[1];
    uint32_t valid = (msg->data[0] >> 3) & 1U;
    uint32_t status = (msg->data[6] >> 4) & 0x7U;
    uint32_t counter = msg->data[0] >> 4;
    checksum = (0U - ((torque << 1) + (status << 12) + valid + counter)) & 0x7FFFU;
  } else {
    checksum = 0xFFU;
    for (int i = 0; i < 7; i++) {
      checksum ^= msg->data[i];
      checksum = mg_crc_lut[checksum];
    }
    checksum ^= 0xFFU;
  }
  return checksum;
}

static uint8_t mg_get_counter(const CANPacket_t *msg) {
  uint8_t counter = 0U;
  if (msg->addr == 0x23cU) {
    counter = (msg->data[4] >> 4) & 0xFU;
  }
  if (msg->addr == 0x1ecU) {
    counter = (msg->data[0] >> 4) & 0xFU;
  }
  if (msg->addr == 0x242U) {
    counter = (msg->data[0] >> 3) & 0xFU;
  }
  if (msg->addr == 0x1b6U) {
    counter = msg->data[6] & 0xFU;
  }
  if (msg->addr == 0xAFU) {
    counter = msg->data[5] >> 4U;
  }
  return counter;
}

static void mg_rx_hook(const CANPacket_t *msg) {
  // Vehicle speed
  if (msg_matches(msg, 0x23cU, 0U)) {
    float speed = (((msg->data[2] & 0x7FU) << 8) | msg->data[3]) * 0.015625;
    vehicle_moving = speed > 0.0;
    UPDATE_VEHICLE_SPEED(speed * KPH_TO_MS);
  }

  // Gas pressed
  if (msg_matches(msg, 0xafU, 0U)) {
    gas_pressed = msg->data[0] != 0U;
  }

  // Driver torque
  if (msg_matches(msg, 0x1ecU, 0U)) {
    int torque_driver_new = (((msg->data[4] & 0x7U) << 8) | msg->data[5]) - 1024U;
    update_sample(&torque_driver, torque_driver_new);
  }

  // Brake pressed
  if (msg_matches(msg, 0x1b6U, 0U)) {
    brake_pressed = GET_BIT(msg, 10U);
  }

  // Cruise state
  if (msg_matches(msg, 0x242U, 0U)) {
    int cruise_state = (msg->data[5] & 0x38U) >> 3;
    bool cruise_engaged = (cruise_state == 2) ||  // Active
                          (cruise_state == 3);    // Override
    pcm_cruise_check(cruise_engaged);
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

  // The TX list dispatches only steering control (0x1fd, bus 0) to this hook.
  int desired_torque = (((msg->data[0] & 0x7U) << 8) | msg->data[1]) - 1024U;
  bool steer_req = GET_BIT(msg, 35U);
  return !steer_torque_cmd_checks(desired_torque, steer_req, MG_STEERING_LIMITS);
}

static safety_config mg_init(uint16_t param) {
  SAFETY_UNUSED(param);

  static const CanMsg MG_TX_MSGS[] = {{0x1fd, 0, 8, .check_relay = true}};

  static const RxCheck mg_rx_checks[] = {
    {.msg = {{0x23c, 0, 8, .frequency = 50U,  .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // SCS_HSC2_FrP19 (speed)
    {.msg = {{0xaf,  0, 8, .frequency = 100U, .ignore_checksum = true, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // GW_HSC2_HCU_FrP00 (gas pedal)
    {.msg = {{0x1ec, 0, 8, .frequency = 50U,  .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // EPS_HSC2_FrP03 (driver torque)
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
