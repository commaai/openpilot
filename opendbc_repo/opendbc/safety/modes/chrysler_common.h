#ifndef SAFETY_MODES_CHRYSLER_COMMON_H
#define SAFETY_MODES_CHRYSLER_COMMON_H

static uint32_t chrysler_get_checksum(const CANPacket_t *msg) {
  uint32_t checksum_byte = GET_LEN(msg) - 1U;
  return (uint8_t)(msg->data[checksum_byte]);
}

static uint32_t chrysler_compute_checksum(const CANPacket_t *msg) {
  uint8_t checksum = 0xFFU;
  uint32_t len = GET_LEN(msg);
  for (uint32_t j = 0U; (j + 1U) < len; j++) {
    checksum = crc8_update(checksum, msg->data[j], 0x1DU);
  }
  return (uint8_t)(~checksum);
}

#endif  // SAFETY_MODES_CHRYSLER_COMMON_H
