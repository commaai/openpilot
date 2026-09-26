#pragma once

#include "opendbc/safety/declarations.h"

static uint8_t body_get_counter(const CANPacket_t *msg) {
  return msg->data[6] & 0xFU;
}

static uint32_t body_get_checksum(const CANPacket_t *msg) {
  return msg->data[7];
}

static uint32_t body_compute_checksum(const CANPacket_t *msg) {
  uint8_t checksum = 0xFFU;
  int len = GET_LEN(msg);
  for (int i = len - 2; i >= 0; i--) {
    checksum = crc8_update(checksum, msg->data[i], 0xD5U);
  }
  return checksum;
}

static void body_rx_hook(const CANPacket_t *msg) {
  SAFETY_UNUSED(msg);

  // controls allowed as soon as RX is valid
  controls_allowed = true;
}

static bool body_tx_hook(const CANPacket_t *msg) {
  bool tx = true;

  if (!controls_allowed && (msg->addr != 0x1U)) {
    tx = false;
  }

  // Allow going into CAN flashing mode even if controls are not allowed
  bool flash_msg = (msg->addr == 0x250U) && (GET_LEN(msg) == 8U);
  if (!controls_allowed && flash_msg && (GET_BYTES_64_LE(msg, 0, 8) == 0x0AB00B1EDEADFACEULL)) {
    tx = true;
  }

  return tx;
}

static safety_config body_init(uint16_t param) {
  static RxCheck body_rx_checks[] = {
    {.msg = {{0x201, 0, 8, 100U, .max_counter = 15U, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  static const CanMsg BODY_TX_MSGS[] = {{0x250, 0, 8, .check_relay = false}, {0x250, 0, 6, .check_relay = false}, {0x251, 0, 5, .check_relay = false},  // body
                                        {0x1, 0, 8, .check_relay = false}};  // CAN flasher

  SAFETY_UNUSED(param);
  safety_config ret = BUILD_SAFETY_CFG(body_rx_checks, BODY_TX_MSGS);
  ret.disable_forwarding = true;
  return ret;
}

const safety_hooks body_hooks = {
  .init = body_init,
  .rx = body_rx_hook,
  .tx = body_tx_hook,
  .get_counter = body_get_counter,
  .get_checksum = body_get_checksum,
  .compute_checksum = body_compute_checksum,
};
