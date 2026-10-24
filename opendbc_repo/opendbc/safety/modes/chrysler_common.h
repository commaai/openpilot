#pragma once

static uint32_t chrysler_get_checksum(const CANPacket_t *msg) {
  int checksum_byte = GET_LEN(msg) - 1U;
  return (uint8_t)(msg->data[checksum_byte]);
}

static uint32_t chrysler_compute_checksum(const CANPacket_t *msg) {
  uint8_t checksum = 0xFFU;
  int len = GET_LEN(msg);
  for (int j = 0; j < (len - 1); j++) {
    checksum = crc8_update(checksum, msg->data[j], 0x1DU);
  }
  return (uint8_t)(~checksum);
}
