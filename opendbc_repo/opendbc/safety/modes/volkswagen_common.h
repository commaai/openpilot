#pragma once

extern const uint16_t FLAG_VOLKSWAGEN_LONG_CONTROL;
const uint16_t FLAG_VOLKSWAGEN_LONG_CONTROL = 1;

static uint8_t volkswagen_crc8_lut_8h2f[256]; // Static lookup table for CRC8 poly 0x2F, aka 8H2F/AUTOSAR

extern bool volkswagen_longitudinal;
bool volkswagen_longitudinal = false;

extern bool volkswagen_set_button_prev;
bool volkswagen_set_button_prev = false;

extern bool volkswagen_resume_button_prev;
bool volkswagen_resume_button_prev = false;


#define MSG_LH_EPS_03        0x09F   // RX from EPS, for driver steering torque
#define MSG_ESP_19           0x0B2   // RX from ABS, for wheel speeds
#define MSG_ESP_05           0x106   // RX from ABS, for brake switch state
#define MSG_TSK_06           0x120   // RX from ECU, for ACC status from drivetrain coordinator
#define MSG_MOTOR_20         0x121   // RX from ECU, for driver throttle input
#define MSG_ACC_06           0x122   // TX by OP, ACC control instructions to the drivetrain coordinator
#define MSG_HCA_01           0x126   // TX by OP, Heading Control Assist steering torque
#define MSG_GRA_ACC_01       0x12B   // TX by OP, ACC control buttons for cancel/resume
#define MSG_ACC_07           0x12E   // TX by OP, ACC control instructions to the drivetrain coordinator
#define MSG_ACC_02           0x30C   // TX by OP, ACC HUD data to the instrument cluster
#define MSG_LDW_02           0x397   // TX by OP, Lane line recognition and text alerts
#define MSG_MOTOR_14         0x3BE   // RX from ECU, for brake switch status


static uint32_t volkswagen_mqb_meb_get_checksum(const CANPacket_t *to_push) {
  return (uint8_t)GET_BYTE(to_push, 0);
}

static uint8_t volkswagen_mqb_meb_get_counter(const CANPacket_t *to_push) {
  // MQB/MEB message counters are consistently found at LSB 8.
  return (uint8_t)GET_BYTE(to_push, 1) & 0xFU;
}

static uint32_t volkswagen_mqb_meb_compute_crc(const CANPacket_t *to_push) {
  int addr = GET_ADDR(to_push);
  int len = GET_LEN(to_push);

  // This is CRC-8H2F/AUTOSAR with a twist. See the OpenDBC implementation
  // of this algorithm for a version with explanatory comments.

  uint8_t crc = 0xFFU;
  for (int i = 1; i < len; i++) {
    crc ^= (uint8_t)GET_BYTE(to_push, i);
    crc = volkswagen_crc8_lut_8h2f[crc];
  }

  uint8_t counter = volkswagen_mqb_meb_get_counter(to_push);
  if (addr == MSG_LH_EPS_03) {
    crc ^= (uint8_t[]){0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5}[counter];
  } else if (addr == MSG_ESP_05) {
    crc ^= (uint8_t[]){0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07}[counter];
  } else if (addr == MSG_TSK_06) {
    crc ^= (uint8_t[]){0xC4, 0xE2, 0x4F, 0xE4, 0xF8, 0x2F, 0x56, 0x81, 0x9F, 0xE5, 0x83, 0x44, 0x05, 0x3F, 0x97, 0xDF}[counter];
  } else if (addr == MSG_MOTOR_20) {
    crc ^= (uint8_t[]){0xE9, 0x65, 0xAE, 0x6B, 0x7B, 0x35, 0xE5, 0x5F, 0x4E, 0xC7, 0x86, 0xA2, 0xBB, 0xDD, 0xEB, 0xB4}[counter];
  } else if (addr == MSG_GRA_ACC_01) {
    crc ^= (uint8_t[]){0x6A, 0x38, 0xB4, 0x27, 0x22, 0xEF, 0xE1, 0xBB, 0xF8, 0x80, 0x84, 0x49, 0xC7, 0x9E, 0x1E, 0x2B}[counter];
  } else {
    // Undefined CAN message, CRC check expected to fail
  }
  crc = volkswagen_crc8_lut_8h2f[crc];

  return (uint8_t)(crc ^ 0xFFU);
}
