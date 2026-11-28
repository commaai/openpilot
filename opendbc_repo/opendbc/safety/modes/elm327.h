#ifndef SAFETY_MODES_ELM327_H
#define SAFETY_MODES_ELM327_H

#include "opendbc/safety/declarations.h"
#include "opendbc/safety/modes/defaults.h"

static bool elm327_tx_hook(const CANPacket_t *msg) {
  const unsigned int GM_CAMERA_DIAG_ADDR = 0x24BU;

  bool tx = true;
  uint32_t len = GET_LEN(msg);

  // All ISO 15765-4 messages must be 8 bytes long
  if (len != 8U) {
    tx = false;
  }

  // Check valid 29 bit send addresses for ISO 15765-4
  // Check valid 11 bit send addresses for ISO 15765-4
  if ((msg->addr != 0x18DB33F1U) && ((msg->addr & 0x1FFF00FFU) != 0x18DA00F1U) &&
      ((msg->addr & 0x1FFFFF00U) != 0x600U) && ((msg->addr & 0x1FFFFF00U) != 0x700U) &&
      (msg->addr != GM_CAMERA_DIAG_ADDR)) {
    tx = false;
  }

  // GM camera uses non-standard diagnostic address, this has no control message address collisions
  if ((msg->addr == GM_CAMERA_DIAG_ADDR) && (len == 8U)) {
    // Only allow known frame types for ISO 15765-2
    if ((msg->data[0] >> 4) > 3U) {
      tx = false;
    }
  }

  return tx;
}

// If safety_param == 0, bus 1 is multiplexed to the OBD-II port
static const safety_hooks elm327_hooks = {
  .init = nooutput_init,
  .tx = elm327_tx_hook,
};

#endif  // SAFETY_MODES_ELM327_H
