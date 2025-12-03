#pragma once

#include "opendbc/safety/declarations.h"

static void body_rx_hook(const CANPacket_t *msg) {
  if (msg->addr == 0x201U) {
    controls_allowed = true;
  }

  #ifdef PANDA_BODY
  if ((msg->addr == 0x250U) && (GET_LEN(msg) >= 4U)) {
    int16_t left_target_deci_rpm = (int16_t)((msg->data[0] << 8U) | msg->data[1]);
    int16_t right_target_deci_rpm = (int16_t)((msg->data[2] << 8U) | msg->data[3]);
    body_can_process_target(left_target_deci_rpm, right_target_deci_rpm);
  }
  #endif
}

static bool body_tx_hook(const CANPacket_t *msg) {
  bool tx = true;

  if (!controls_allowed && (msg->addr != 0x1U)) {
    tx = false;
  }

  // Allow going into CAN flashing mode even if controls are not allowed
  bool flash_msg = (msg->addr == 0x250U) && (GET_LEN(msg) == 8U);
  if (!controls_allowed && (GET_BYTES(msg, 0, 4) == 0xdeadfaceU) && (GET_BYTES(msg, 4, 4) == 0x0ab00b1eU) && flash_msg) {
    tx = true;
  }

  return tx;
}

static safety_config body_init(uint16_t param) {
  static RxCheck body_rx_checks[] = {
    {.msg = {{0x201, 2, 8, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},

    #ifdef PANDA_BODY
    {.msg = {{0x250, 0, 6, 100U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    #endif
  };

  static const CanMsg BODY_TX_MSGS[] = {{0x250, 2, 8, .check_relay = false}, {0x250, 2, 6, .check_relay = false}, {0x251, 2, 5, .check_relay = false},  // body
                                        {0x1, 2, 8, .check_relay = false}};  // CAN flasher

  SAFETY_UNUSED(param);
  safety_config ret = BUILD_SAFETY_CFG(body_rx_checks, BODY_TX_MSGS);
  ret.disable_forwarding = true;
  return ret;
}

const safety_hooks body_hooks = {
  .init = body_init,
  .rx = body_rx_hook,
  .tx = body_tx_hook,
};
