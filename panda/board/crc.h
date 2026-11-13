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

uint32_t crc32_update(uint32_t crc, const uint8_t *dat, uint32_t len) {
  uint32_t ret = crc;
  for (uint32_t i = 0U; i < len; i++) {
    ret = crc32_table[(ret ^ dat[i]) & 0xFFU] ^ (ret >> 8U);
  }
  return ret;
}

uint32_t crc32(const uint8_t *dat, uint32_t len) {
  return crc32_update(0xFFFFFFFFU, dat, len) ^ 0xFFFFFFFFU;
}
