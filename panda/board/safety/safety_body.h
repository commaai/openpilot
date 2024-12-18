#pragma once

#include "safety_declarations.h"

static void body_rx_hook(const CANPacket_t *to_push) {
  // body is never at standstill
  vehicle_moving = true;

  if (GET_ADDR(to_push) == 0x201U) {
    controls_allowed = true;
  }
}

static bool body_tx_hook(const CANPacket_t *to_send) {
  bool tx = true;
  int addr = GET_ADDR(to_send);
  int len = GET_LEN(to_send);

  if (!controls_allowed && (addr != 0x1)) {
    tx = false;
  }

  // Allow going into CAN flashing mode for base & knee even if controls are not allowed
  bool flash_msg = ((addr == 0x250) || (addr == 0x350)) && (len == 8);
  if (!controls_allowed && (GET_BYTES(to_send, 0, 4) == 0xdeadfaceU) && (GET_BYTES(to_send, 4, 4) == 0x0ab00b1eU) && flash_msg) {
    tx = true;
  }

  return tx;
}

static safety_config body_init(uint16_t param) {
  static RxCheck body_rx_checks[] = {
    {.msg = {{0x201, 0, 8, .check_checksum = false, .max_counter = 0U, .frequency = 100U}, { 0 }, { 0 }}},
  };

  static const CanMsg BODY_TX_MSGS[] = {{0x250, 0, 8}, {0x250, 0, 6}, {0x251, 0, 5},  // body
                                        {0x350, 0, 8}, {0x350, 0, 6}, {0x351, 0, 5},  // knee
                                        {0x1, 0, 8}}; // CAN flasher

  UNUSED(param);
  return BUILD_SAFETY_CFG(body_rx_checks, BODY_TX_MSGS);
}

const safety_hooks body_hooks = {
  .init = body_init,
  .rx = body_rx_hook,
  .tx = body_tx_hook,
  .fwd = default_fwd_hook,
};
