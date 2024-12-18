#include <array>
#include <unordered_map>

#include "opendbc/can/common.h"

unsigned int honda_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d) {
  int s = 0;
  bool extended = address > 0x7FF;
  while (address) { s += (address & 0xF); address >>= 4; }
  for (int i = 0; i < d.size(); i++) {
    uint8_t x = d[i];
    if (i == d.size()-1) x >>= 4; // remove checksum
    s += (x & 0xF) + (x >> 4);
  }
  s = 8-s;
  if (extended) s += 3;  // extended can

  return s & 0xF;
}

unsigned int toyota_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d) {
  unsigned int s = d.size();
  while (address) { s += address & 0xFF; address >>= 8; }
  for (int i = 0; i < d.size() - 1; i++) { s += d[i]; }

  return s & 0xFF;
}

unsigned int subaru_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d) {
  unsigned int s = 0;
  while (address) { s += address & 0xFF; address >>= 8; }

  // skip checksum in first byte
  for (int i = 1; i < d.size(); i++) { s += d[i]; }

  return s & 0xFF;
}

unsigned int chrysler_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d) {
  // jeep chrysler canbus checksum from http://illmatics.com/Remote%20Car%20Hacking.pdf
  uint8_t checksum = 0xFF;
  for (int j = 0; j < (d.size() - 1); j++) {
    uint8_t shift = 0x80;
    uint8_t curr = d[j];
    for (int i = 0; i < 8; i++) {
      uint8_t bit_sum = curr & shift;
      uint8_t temp_chk = checksum & 0x80U;
      if (bit_sum != 0U) {
        bit_sum = 0x1C;
        if (temp_chk != 0U) {
          bit_sum = 1;
        }
        checksum = checksum << 1;
        temp_chk = checksum | 1U;
        bit_sum ^= temp_chk;
      } else {
        if (temp_chk != 0U) {
          bit_sum = 0x1D;
        }
        checksum = checksum << 1;
        bit_sum ^= checksum;
      }
      checksum = bit_sum;
      shift = shift >> 1;
    }
  }
  return ~checksum & 0xFF;
}

// Static lookup table for fast computation of CRCs
uint8_t crc8_lut_8h2f[256]; // CRC8 poly 0x2F, aka 8H2F/AUTOSAR
uint8_t crc8_lut_j1850[256]; // CRC8 poly 0x1D, aka SAE J1850
uint16_t crc16_lut_xmodem[256]; // CRC16 poly 0x1021, aka XMODEM

void gen_crc_lookup_table_8(uint8_t poly, uint8_t crc_lut[]) {
  uint8_t crc;
  int i, j;

   for (i = 0; i < 256; i++) {
    crc = i;
    for (j = 0; j < 8; j++) {
      if ((crc & 0x80) != 0)
        crc = (uint8_t)((crc << 1) ^ poly);
      else
        crc <<= 1;
    }
    crc_lut[i] = crc;
  }
}

void gen_crc_lookup_table_16(uint16_t poly, uint16_t crc_lut[]) {
  uint16_t crc;
  int i, j;

   for (i = 0; i < 256; i++) {
    crc = i << 8;
    for (j = 0; j < 8; j++) {
      if ((crc & 0x8000) != 0) {
        crc = (uint16_t)((crc << 1) ^ poly);
      } else {
        crc <<= 1;
      }
    }
    crc_lut[i] = crc;
  }
}

// Initializes CRC lookup tables at module initialization
struct CrcInitializer {
  CrcInitializer() {
    gen_crc_lookup_table_8(0x2F, crc8_lut_8h2f);  // CRC-8 8H2F/AUTOSAR for Volkswagen
    gen_crc_lookup_table_8(0x1D, crc8_lut_j1850);  // CRC-8 SAE-J1850
    gen_crc_lookup_table_16(0x1021, crc16_lut_xmodem);  // CRC-16 XMODEM for HKG CAN FD
  }
};

static CrcInitializer crcInitializer;

static const std::unordered_map<uint32_t, std::array<uint8_t, 16>> volkswagen_mqb_crc_constants {
  {0x40,  {0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40}},  // Airbag_01
  {0x86,  {0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86, 0x86}},  // LWI_01
  {0x9F,  {0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5}},  // LH_EPS_03
  {0xAD,  {0x3F, 0x69, 0x39, 0xDC, 0x94, 0xF9, 0x14, 0x64, 0xD8, 0x6A, 0x34, 0xCE, 0xA2, 0x55, 0xB5, 0x2C}},  // Getriebe_11
  {0xFD,  {0xB4, 0xEF, 0xF8, 0x49, 0x1E, 0xE5, 0xC2, 0xC0, 0x97, 0x19, 0x3C, 0xC9, 0xF1, 0x98, 0xD6, 0x61}},  // ESP_21
  {0x101, {0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA}},  // ESP_02
  {0x106, {0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07}},  // ESP_05
  {0x116, {0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC, 0xAC}},  // ESP_10
  {0x117, {0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16, 0x16}},  // ACC_10
  {0x120, {0xC4, 0xE2, 0x4F, 0xE4, 0xF8, 0x2F, 0x56, 0x81, 0x9F, 0xE5, 0x83, 0x44, 0x05, 0x3F, 0x97, 0xDF}},  // TSK_06
  {0x121, {0xE9, 0x65, 0xAE, 0x6B, 0x7B, 0x35, 0xE5, 0x5F, 0x4E, 0xC7, 0x86, 0xA2, 0xBB, 0xDD, 0xEB, 0xB4}},  // Motor_20
  {0x122, {0x37, 0x7D, 0xF3, 0xA9, 0x18, 0x46, 0x6D, 0x4D, 0x3D, 0x71, 0x92, 0x9C, 0xE5, 0x32, 0x10, 0xB9}},  // ACC_06
  {0x126, {0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA, 0xDA}},  // HCA_01
  {0x12B, {0x6A, 0x38, 0xB4, 0x27, 0x22, 0xEF, 0xE1, 0xBB, 0xF8, 0x80, 0x84, 0x49, 0xC7, 0x9E, 0x1E, 0x2B}},  // GRA_ACC_01
  {0x12E, {0xF8, 0xE5, 0x97, 0xC9, 0xD6, 0x07, 0x47, 0x21, 0x66, 0xDD, 0xCF, 0x6F, 0xA1, 0x94, 0x74, 0x63}},  // ACC_07
  {0x187, {0x7F, 0xED, 0x17, 0xC2, 0x7C, 0xEB, 0x44, 0x21, 0x01, 0xFA, 0xDB, 0x15, 0x4A, 0x6B, 0x23, 0x05}},  // Motor_EV_01
  {0x1AB, {0x13, 0x21, 0x9B, 0x6A, 0x9A, 0x62, 0xD4, 0x65, 0x18, 0xF1, 0xAB, 0x16, 0x32, 0x89, 0xE7, 0x26}},  // ESP_33
  {0x30C, {0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F}},  // ACC_02
  {0x30F, {0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C}},  // SWA_01
  {0x324, {0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27, 0x27}},  // ACC_04
  {0x3C0, {0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3, 0xC3}},  // Klemmen_Status_01
  {0x3D5, {0xC5, 0x39, 0xC7, 0xF9, 0x92, 0xD8, 0x24, 0xCE, 0xF1, 0xB5, 0x7A, 0xC4, 0xBC, 0x60, 0xE3, 0xD1}},  // Licht_Anf_01
  {0x65D, {0xAC, 0xB3, 0xAB, 0xEB, 0x7A, 0xE1, 0x3B, 0xF7, 0x73, 0xBA, 0x7C, 0x9E, 0x06, 0x5F, 0x02, 0xD9}},  // ESP_20
};

unsigned int volkswagen_mqb_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d) {
  // This is AUTOSAR E2E Profile 2, CRC-8H2F with a "data ID" (varying by message/counter) appended to the payload

  uint8_t crc = 0xFF; // CRC-8H2F initial value

  // CRC over payload first, skipping the first byte where the CRC lives
  for (int i = 1; i < d.size(); i++) {
    crc ^= d[i];
    crc = crc8_lut_8h2f[crc];
  }

  // Continue CRC over the "data ID"
  uint8_t counter = d[1] & 0x0F;
  auto crc_const = volkswagen_mqb_crc_constants.find(address);
  if (crc_const != volkswagen_mqb_crc_constants.end()) {
    crc ^= crc_const->second[counter];
    crc = crc8_lut_8h2f[crc];
  } else {
    printf("Attempt to CRC check undefined Volkswagen message 0x%02X\n", address);
  }

  return crc ^ 0xFF; // CRC-8H2F final XOR
}

unsigned int xor_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d) {
  uint8_t checksum = 0;
  int checksum_byte = sig.start_bit / 8;

  // Simple XOR over the payload, except for the byte where the checksum lives.
  for (int i = 0; i < d.size(); i++) {
    if (i != checksum_byte) {
      checksum ^= d[i];
    }
  }

  return checksum;
}

unsigned int pedal_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d) {
  uint8_t crc = 0xFF;
  uint8_t poly = 0xD5; // standard crc8

  // skip checksum byte
  for (int i = d.size()-2; i >= 0; i--) {
    crc ^= d[i];
    for (int j = 0; j < 8; j++) {
      if ((crc & 0x80) != 0) {
        crc = (uint8_t)((crc << 1) ^ poly);
      } else {
        crc <<= 1;
      }
    }
  }
  return crc;
}

unsigned int hkg_can_fd_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d) {
  uint16_t crc = 0;

  for (int i = 2; i < d.size(); i++) {
    crc = (crc << 8) ^ crc16_lut_xmodem[(crc >> 8) ^ d[i]];
  }

  // Add address to crc
  crc = (crc << 8) ^ crc16_lut_xmodem[(crc >> 8) ^ ((address >> 0) & 0xFF)];
  crc = (crc << 8) ^ crc16_lut_xmodem[(crc >> 8) ^ ((address >> 8) & 0xFF)];

  if (d.size() == 8) {
    crc ^= 0x5f29;
  } else if (d.size() == 16) {
    crc ^= 0x041d;
  } else if (d.size() == 24) {
    crc ^= 0x819d;
  } else if (d.size() == 32) {
    crc ^= 0x9f5b;
  }

  return crc;
}

unsigned int fca_giorgio_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d) {
  // CRC is in the last byte, poly is same as SAE J1850 but uses a different init value and final XOR
  uint8_t crc = 0x00;

  for (int i = 0; i < d.size() - 1; i++) {
    crc ^= d[i];
    crc = crc8_lut_j1850[crc];
  }

  // Final XOR varies for EPS messages, all others use a common value
  if (address == 0xDE) {  // EPS_1
    return crc ^ 0x10;
  } else if (address == 0x106) {  // EPS_2
    return crc ^ 0xF6;
  } else if (address == 0x122) {  // EPS_3
    return crc ^ 0xF1;
  } else {
    return crc ^ 0xA;
  }

}
