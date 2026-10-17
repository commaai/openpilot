#pragma once

// standard CRC-32, same as zlib's crc32()
static uint32_t crc32_table[256];

void crc32_init(void) {
  for (uint32_t i = 0U; i < 256U; i++) {
    uint32_t c = i;
    for (uint8_t j = 0U; j < 8U; j++) {
      c = ((c & 1U) != 0U) ? (0xEDB88320U ^ (c >> 1U)) : (c >> 1U);
    }
    crc32_table[i] = c;
  }
}

uint32_t crc32(const uint8_t *dat, uint32_t len) {
  uint32_t crc = 0xFFFFFFFFU;
  for (uint32_t i = 0U; i < len; i++) {
    crc = crc32_table[(crc ^ dat[i]) & 0xFFU] ^ (crc >> 8U);
  }
  return crc ^ 0xFFFFFFFFU;
}
