#pragma once

extern const uint16_t FLAG_VOLKSWAGEN_LONG_CONTROL;
const uint16_t FLAG_VOLKSWAGEN_LONG_CONTROL = 1;

extern bool volkswagen_longitudinal;
bool volkswagen_longitudinal = false;

extern bool volkswagen_set_button_prev;
bool volkswagen_set_button_prev = false;

extern bool volkswagen_resume_button_prev;
bool volkswagen_resume_button_prev = false;

extern bool volkswagen_brake_pedal_switch;
extern bool volkswagen_brake_pressure_detected;
bool volkswagen_brake_pedal_switch = false;
bool volkswagen_brake_pressure_detected = false;

#define MSG_LH_EPS_03        0x09FU   // RX from EPS, for driver steering torque
#define MSG_ESP_19           0x0B2U   // RX from ABS, for wheel speeds
#define MSG_ESP_05           0x106U   // RX from ABS, for brake switch state
#define MSG_TSK_06           0x120U   // RX from ECU, for ACC status from drivetrain coordinator
#define MSG_MOTOR_20         0x121U   // RX from ECU, for driver throttle input
#define MSG_ACC_06           0x122U   // TX by OP, ACC control instructions to the drivetrain coordinator
#define MSG_HCA_01           0x126U   // TX by OP, Heading Control Assist steering torque
#define MSG_GRA_ACC_01       0x12BU   // TX by OP, ACC control buttons for cancel/resume
#define MSG_ACC_07           0x12EU   // TX by OP, ACC control instructions to the drivetrain coordinator
#define MSG_ACC_02           0x30CU   // TX by OP, ACC HUD data to the instrument cluster
#define MSG_LDW_02           0x397U   // TX by OP, Lane line recognition and text alerts
#define MSG_MOTOR_14         0x3BEU   // RX from ECU, for brake switch status

// MLB only messages
#define MSG_ESP_03      0x103U   // RX from ABS, for wheel speeds
#define MSG_LS_01       0x10BU   // TX by OP, ACC control buttons for cancel/resume
#define MSG_MOTOR_03    0x105U   // RX from ECU, for driver throttle input and brake switch status
#define MSG_TSK_04      0x10EU   // RX from ECU, for ACC status from drivetrain coordinator

static void volkswagen_common_init(void) {
  volkswagen_longitudinal = false;
  volkswagen_set_button_prev = false;
  volkswagen_resume_button_prev = false;
  volkswagen_brake_pedal_switch = false;
  volkswagen_brake_pressure_detected = false;
  return;
}

static uint32_t volkswagen_mxb_get_checksum(const CANPacket_t *msg) {
  return (uint8_t)msg->data[0];
}

static uint8_t volkswagen_mxb_get_counter(const CANPacket_t *msg) {
  // MQB/MLB/MEB message counters are consistently found at LSB 8.
  return (uint8_t)msg->data[1] & 0xFU;
}

// This is CRC-8H2F/AUTOSAR with a counter-dependent data ID. See the
// opendbc/car/volkswagen/ implementation for explanatory comments.
static uint8_t volkswagen_compute_crc(const CANPacket_t *msg, int len, uint8_t salt) {
  // CRC-8 lookup table for poly 0x2F, aka 8H2F/AUTOSAR
  // generated with: python3 -c 'f=lambda c:[c:=((c<<1)^0x2F)&0xFF if c&0x80 else (c<<1)&0xFF for _ in range(8)][-1];print(",\n".join("    "+", ".join(f"0x{f(i):02X}U" for i in range(r,r+16)) for r in range(0,256,16)))'
  static const uint8_t volkswagen_crc8_lut_8h2f[256] = {
    0x00U, 0x2FU, 0x5EU, 0x71U, 0xBCU, 0x93U, 0xE2U, 0xCDU, 0x57U, 0x78U, 0x09U, 0x26U, 0xEBU, 0xC4U, 0xB5U, 0x9AU,
    0xAEU, 0x81U, 0xF0U, 0xDFU, 0x12U, 0x3DU, 0x4CU, 0x63U, 0xF9U, 0xD6U, 0xA7U, 0x88U, 0x45U, 0x6AU, 0x1BU, 0x34U,
    0x73U, 0x5CU, 0x2DU, 0x02U, 0xCFU, 0xE0U, 0x91U, 0xBEU, 0x24U, 0x0BU, 0x7AU, 0x55U, 0x98U, 0xB7U, 0xC6U, 0xE9U,
    0xDDU, 0xF2U, 0x83U, 0xACU, 0x61U, 0x4EU, 0x3FU, 0x10U, 0x8AU, 0xA5U, 0xD4U, 0xFBU, 0x36U, 0x19U, 0x68U, 0x47U,
    0xE6U, 0xC9U, 0xB8U, 0x97U, 0x5AU, 0x75U, 0x04U, 0x2BU, 0xB1U, 0x9EU, 0xEFU, 0xC0U, 0x0DU, 0x22U, 0x53U, 0x7CU,
    0x48U, 0x67U, 0x16U, 0x39U, 0xF4U, 0xDBU, 0xAAU, 0x85U, 0x1FU, 0x30U, 0x41U, 0x6EU, 0xA3U, 0x8CU, 0xFDU, 0xD2U,
    0x95U, 0xBAU, 0xCBU, 0xE4U, 0x29U, 0x06U, 0x77U, 0x58U, 0xC2U, 0xEDU, 0x9CU, 0xB3U, 0x7EU, 0x51U, 0x20U, 0x0FU,
    0x3BU, 0x14U, 0x65U, 0x4AU, 0x87U, 0xA8U, 0xD9U, 0xF6U, 0x6CU, 0x43U, 0x32U, 0x1DU, 0xD0U, 0xFFU, 0x8EU, 0xA1U,
    0xE3U, 0xCCU, 0xBDU, 0x92U, 0x5FU, 0x70U, 0x01U, 0x2EU, 0xB4U, 0x9BU, 0xEAU, 0xC5U, 0x08U, 0x27U, 0x56U, 0x79U,
    0x4DU, 0x62U, 0x13U, 0x3CU, 0xF1U, 0xDEU, 0xAFU, 0x80U, 0x1AU, 0x35U, 0x44U, 0x6BU, 0xA6U, 0x89U, 0xF8U, 0xD7U,
    0x90U, 0xBFU, 0xCEU, 0xE1U, 0x2CU, 0x03U, 0x72U, 0x5DU, 0xC7U, 0xE8U, 0x99U, 0xB6U, 0x7BU, 0x54U, 0x25U, 0x0AU,
    0x3EU, 0x11U, 0x60U, 0x4FU, 0x82U, 0xADU, 0xDCU, 0xF3U, 0x69U, 0x46U, 0x37U, 0x18U, 0xD5U, 0xFAU, 0x8BU, 0xA4U,
    0x05U, 0x2AU, 0x5BU, 0x74U, 0xB9U, 0x96U, 0xE7U, 0xC8U, 0x52U, 0x7DU, 0x0CU, 0x23U, 0xEEU, 0xC1U, 0xB0U, 0x9FU,
    0xABU, 0x84U, 0xF5U, 0xDAU, 0x17U, 0x38U, 0x49U, 0x66U, 0xFCU, 0xD3U, 0xA2U, 0x8DU, 0x40U, 0x6FU, 0x1EU, 0x31U,
    0x76U, 0x59U, 0x28U, 0x07U, 0xCAU, 0xE5U, 0x94U, 0xBBU, 0x21U, 0x0EU, 0x7FU, 0x50U, 0x9DU, 0xB2U, 0xC3U, 0xECU,
    0xD8U, 0xF7U, 0x86U, 0xA9U, 0x64U, 0x4BU, 0x3AU, 0x15U, 0x8FU, 0xA0U, 0xD1U, 0xFEU, 0x33U, 0x1CU, 0x6DU, 0x42U
  };
  uint8_t crc = 0xFFU;
  for (int i = 1; i < len; i++) {
    crc ^= (uint8_t)msg->data[i];
    crc = volkswagen_crc8_lut_8h2f[crc];
  }
  return volkswagen_crc8_lut_8h2f[crc ^ salt] ^ 0xFFU;
}

static uint32_t volkswagen_mxb_compute_crc(const CANPacket_t *msg) {
  uint8_t salt = 0;
  uint8_t counter = volkswagen_mxb_get_counter(msg);
  if (msg->addr == MSG_LH_EPS_03) {
    salt = 0xF5U;
  }
  if (msg->addr == MSG_ESP_05) {
    salt = 0x07U;
  }
  if (msg->addr == MSG_TSK_06) {
    salt = (uint8_t[]){0xC4, 0xE2, 0x4F, 0xE4, 0xF8, 0x2F, 0x56, 0x81, 0x9F, 0xE5, 0x83, 0x44, 0x05, 0x3F, 0x97, 0xDF}[counter];
  }
  if (msg->addr == MSG_MOTOR_20) {
    salt = (uint8_t[]){0xE9, 0x65, 0xAE, 0x6B, 0x7B, 0x35, 0xE5, 0x5F, 0x4E, 0xC7, 0x86, 0xA2, 0xBB, 0xDD, 0xEB, 0xB4}[counter];
  }
  if (msg->addr == MSG_GRA_ACC_01) {
    salt = (uint8_t[]){0x6A, 0x38, 0xB4, 0x27, 0x22, 0xEF, 0xE1, 0xBB, 0xF8, 0x80, 0x84, 0x49, 0xC7, 0x9E, 0x1E, 0x2B}[counter];
  }
  return volkswagen_compute_crc(msg, GET_LEN(msg), salt);
}

static int volkswagen_mlb_mqb_driver_input_torque(const CANPacket_t *msg) {
  // Signal: LH_EPS_03.EPS_Lenkmoment (absolute torque)
  // Signal: LH_EPS_03.EPS_VZ_Lenkmoment (direction)
  int torque_driver_new = msg->data[5] | ((msg->data[6] & 0x1FU) << 8);
  bool sign = GET_BIT(msg, 55U);
  if (sign) {
    torque_driver_new *= -1;
  }
  return torque_driver_new;
}

static int volkswagen_mlb_mqb_steering_control_torque(const CANPacket_t *msg) {
  // Signal: HCA_01.HCA_01_LM_Offset (absolute torque)
  // Signal: HCA_01.HCA_01_LM_OffSign (direction)
  int desired_torque = msg->data[2] | ((msg->data[3] & 0x1U) << 8);
  bool sign = GET_BIT(msg, 31U);
  if (sign) {
    desired_torque *= -1;
  }
  return desired_torque;
}
