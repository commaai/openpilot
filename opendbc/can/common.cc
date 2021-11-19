#include "common.h"

unsigned int honda_checksum(unsigned int address, uint64_t d, int l) {
  d >>= ((8-l)*8); // remove padding
  d >>= 4; // remove checksum

  int s = 0;
  bool extended = address > 0x7FF; // extended can
  while (address) { s += (address & 0xF); address >>= 4; }
  while (d) { s += (d & 0xF); d >>= 4; }
  s = 8-s;
  if (extended) s += 3;
  s &= 0xF;

  return s;
}

unsigned int toyota_checksum(unsigned int address, uint64_t d, int l) {
  d >>= ((8-l)*8); // remove padding
  d >>= 8; // remove checksum

  unsigned int s = l;
  while (address) { s += address & 0xFF; address >>= 8; }
  while (d) { s += d & 0xFF; d >>= 8; }

  return s & 0xFF;
}

unsigned int subaru_checksum(unsigned int address, uint64_t d, int l) {
  d >>= ((8-l)*8); // remove padding

  unsigned int s = 0;
  while (address) { s += address & 0xFF; address >>= 8; }
  l -= 1; // checksum is first byte
  while (l) { s += d & 0xFF; d >>= 8; l -= 1; }

  return s & 0xFF;
}

unsigned int chrysler_checksum(unsigned int address, uint64_t d, int l) {
  /* This function does not want the checksum byte in the input data.
  jeep chrysler canbus checksum from http://illmatics.com/Remote%20Car%20Hacking.pdf */
  uint8_t checksum = 0xFF;
  for (int j = 0; j < (l - 1); j++) {
    uint8_t shift = 0x80;
    uint8_t curr = (d >> 8*j) & 0xFF;
    for (int i=0; i<8; i++) {
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

// Static lookup table for fast computation of CRC8 poly 0x2F, aka 8H2F/AUTOSAR
uint8_t crc8_lut_8h2f[256];

void gen_crc_lookup_table(uint8_t poly, uint8_t crc_lut[]) {
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

void init_crc_lookup_tables() {
  // At init time, set up static lookup tables for fast CRC computation.

  gen_crc_lookup_table(0x2F, crc8_lut_8h2f);    // CRC-8 8H2F/AUTOSAR for Volkswagen
}

unsigned int volkswagen_crc(unsigned int address, uint64_t d, int l) {
  // Volkswagen uses standard CRC8 8H2F/AUTOSAR, but they compute it with
  // a magic variable padding byte tacked onto the end of the payload.
  // https://www.autosar.org/fileadmin/user_upload/standards/classic/4-3/AUTOSAR_SWS_CRCLibrary.pdf

  uint8_t crc = 0xFF; // Standard init value for CRC8 8H2F/AUTOSAR

  // CRC the payload first, skipping over the first byte where the CRC lives.
  for (int i = 1; i < l; i++) {
    crc ^= (d >> (i*8)) & 0xFF;
    crc = crc8_lut_8h2f[crc];
  }

  // Look up and apply the magic final CRC padding byte, which permutes by CAN
  // address, and additionally (for SOME addresses) by the message counter.
  uint8_t counter = ((d >> 8) & 0xFF) & 0x0F;
  switch(address) {
    case 0x86:  // LWI_01 Steering Angle
      crc ^= (uint8_t[]){0x86,0x86,0x86,0x86,0x86,0x86,0x86,0x86,0x86,0x86,0x86,0x86,0x86,0x86,0x86,0x86}[counter];
      break;
    case 0x9F:  // LH_EPS_03 Electric Power Steering
      crc ^= (uint8_t[]){0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5,0xF5}[counter];
      break;
    case 0xAD:  // Getriebe_11 Automatic Gearbox
      crc ^= (uint8_t[]){0x3F,0x69,0x39,0xDC,0x94,0xF9,0x14,0x64,0xD8,0x6A,0x34,0xCE,0xA2,0x55,0xB5,0x2C}[counter];
      break;
    case 0xFD:  // ESP_21 Electronic Stability Program
      crc ^= (uint8_t[]){0xB4,0xEF,0xF8,0x49,0x1E,0xE5,0xC2,0xC0,0x97,0x19,0x3C,0xC9,0xF1,0x98,0xD6,0x61}[counter];
      break;
    case 0x106: // ESP_05 Electronic Stability Program
      crc ^= (uint8_t[]){0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07}[counter];
      break;
    case 0x117: // ACC_10 Automatic Cruise Control
      crc ^= (uint8_t[]){0x16,0x16,0x16,0x16,0x16,0x16,0x16,0x16,0x16,0x16,0x16,0x16,0x16,0x16,0x16,0x16}[counter];
      break;
    case 0x120: // TSK_06 Drivetrain Coordinator
      crc ^= (uint8_t[]){0xC4,0xE2,0x4F,0xE4,0xF8,0x2F,0x56,0x81,0x9F,0xE5,0x83,0x44,0x05,0x3F,0x97,0xDF}[counter];
      break;
    case 0x121: // Motor_20 Driver Throttle Inputs
      crc ^= (uint8_t[]){0xE9,0x65,0xAE,0x6B,0x7B,0x35,0xE5,0x5F,0x4E,0xC7,0x86,0xA2,0xBB,0xDD,0xEB,0xB4}[counter];
      break;
    case 0x122: // ACC_06 Automatic Cruise Control
      crc ^= (uint8_t[]){0x37,0x7D,0xF3,0xA9,0x18,0x46,0x6D,0x4D,0x3D,0x71,0x92,0x9C,0xE5,0x32,0x10,0xB9}[counter];
      break;
    case 0x126: // HCA_01 Heading Control Assist
      crc ^= (uint8_t[]){0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA,0xDA}[counter];
      break;
    case 0x12B: // GRA_ACC_01 Steering wheel controls for ACC
      crc ^= (uint8_t[]){0x6A,0x38,0xB4,0x27,0x22,0xEF,0xE1,0xBB,0xF8,0x80,0x84,0x49,0xC7,0x9E,0x1E,0x2B}[counter];
      break;
    case 0x12E: // ACC_07 Automatic Cruise Control
      crc ^= (uint8_t[]){0xF8,0xE5,0x97,0xC9,0xD6,0x07,0x47,0x21,0x66,0xDD,0xCF,0x6F,0xA1,0x94,0x74,0x63}[counter];
      break;
    case 0x187: // EV_Gearshift "Gear" selection data for EVs with no gearbox
      crc ^= (uint8_t[]){0x7F,0xED,0x17,0xC2,0x7C,0xEB,0x44,0x21,0x01,0xFA,0xDB,0x15,0x4A,0x6B,0x23,0x05}[counter];
      break;
    case 0x30C: // ACC_02 Automatic Cruise Control
      crc ^= (uint8_t[]){0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F}[counter];
      break;
    case 0x30F: // SWA_01 Lane Change Assist (SpurWechselAssistent)
      crc ^= (uint8_t[]){0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C}[counter];
      break;
    case 0x324: // ACC_04 Automatic Cruise Control
      crc ^= (uint8_t[]){0x27,0x27,0x27,0x27,0x27,0x27,0x27,0x27,0x27,0x27,0x27,0x27,0x27,0x27,0x27,0x27}[counter];
      break;
    case 0x3C0: // Klemmen_Status_01 ignition and starting status
      crc ^= (uint8_t[]){0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3,0xC3}[counter];
      break;
    case 0x65D: // ESP_20 Electronic Stability Program
      crc ^= (uint8_t[]){0xAC,0xB3,0xAB,0xEB,0x7A,0xE1,0x3B,0xF7,0x73,0xBA,0x7C,0x9E,0x06,0x5F,0x02,0xD9}[counter];
      break;
    default:    // As-yet undefined CAN message, CRC check expected to fail
      printf("Attempt to CRC check undefined Volkswagen message 0x%02X\n", address);
      crc ^= (uint8_t[]){0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}[counter];
      break;
  }
  crc = crc8_lut_8h2f[crc];

  return crc ^ 0xFF; // Return after standard final XOR for CRC8 8H2F/AUTOSAR
}

unsigned int pedal_checksum(uint64_t d, int l) {
  uint8_t crc = 0xFF;
  uint8_t poly = 0xD5; // standard crc8

  d >>= ((8-l)*8); // remove padding
  d >>= 8; // remove checksum

  int i, j;
  for (i = 0; i < l - 1; i++) {
    crc ^= (d >> (i*8)) & 0xFF;
    for (j = 0; j < 8; j++) {
      if ((crc & 0x80) != 0) {
        crc = (uint8_t)((crc << 1) ^ poly);
      }
      else {
        crc <<= 1;
      }
    }
  }
  return crc;
}


uint64_t read_u64_be(const uint8_t* v) {
  return (((uint64_t)v[0] << 56)
          | ((uint64_t)v[1] << 48)
          | ((uint64_t)v[2] << 40)
          | ((uint64_t)v[3] << 32)
          | ((uint64_t)v[4] << 24)
          | ((uint64_t)v[5] << 16)
          | ((uint64_t)v[6] << 8)
          | (uint64_t)v[7]);
}

uint64_t read_u64_le(const uint8_t* v) {
  return ((uint64_t)v[0]
          | ((uint64_t)v[1] << 8)
          | ((uint64_t)v[2] << 16)
          | ((uint64_t)v[3] << 24)
          | ((uint64_t)v[4] << 32)
          | ((uint64_t)v[5] << 40)
          | ((uint64_t)v[6] << 48)
          | ((uint64_t)v[7] << 56));
}
