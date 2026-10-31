#pragma once

#include "opendbc/safety/declarations.h"

// CRC-16 lookup table for poly 0x1021 (CRC-16/XMODEM)
// generated with: python3 -c 'f=lambda c:[c:=((c<<1)^0x1021)&0xFFFF if c&0x8000 else (c<<1)&0xFFFF for _ in range(8)][-1];print(",\n".join("  "+", ".join(f"0x{f(i<<8):04X}U" for i in range(r,min(r+12,256))) for r in range(0,256,12)))'
extern const uint16_t hyundai_canfd_crc_lut[256];
const uint16_t hyundai_canfd_crc_lut[256] = {
  0x0000U, 0x1021U, 0x2042U, 0x3063U, 0x4084U, 0x50A5U, 0x60C6U, 0x70E7U, 0x8108U, 0x9129U, 0xA14AU, 0xB16BU,
  0xC18CU, 0xD1ADU, 0xE1CEU, 0xF1EFU, 0x1231U, 0x0210U, 0x3273U, 0x2252U, 0x52B5U, 0x4294U, 0x72F7U, 0x62D6U,
  0x9339U, 0x8318U, 0xB37BU, 0xA35AU, 0xD3BDU, 0xC39CU, 0xF3FFU, 0xE3DEU, 0x2462U, 0x3443U, 0x0420U, 0x1401U,
  0x64E6U, 0x74C7U, 0x44A4U, 0x5485U, 0xA56AU, 0xB54BU, 0x8528U, 0x9509U, 0xE5EEU, 0xF5CFU, 0xC5ACU, 0xD58DU,
  0x3653U, 0x2672U, 0x1611U, 0x0630U, 0x76D7U, 0x66F6U, 0x5695U, 0x46B4U, 0xB75BU, 0xA77AU, 0x9719U, 0x8738U,
  0xF7DFU, 0xE7FEU, 0xD79DU, 0xC7BCU, 0x48C4U, 0x58E5U, 0x6886U, 0x78A7U, 0x0840U, 0x1861U, 0x2802U, 0x3823U,
  0xC9CCU, 0xD9EDU, 0xE98EU, 0xF9AFU, 0x8948U, 0x9969U, 0xA90AU, 0xB92BU, 0x5AF5U, 0x4AD4U, 0x7AB7U, 0x6A96U,
  0x1A71U, 0x0A50U, 0x3A33U, 0x2A12U, 0xDBFDU, 0xCBDCU, 0xFBBFU, 0xEB9EU, 0x9B79U, 0x8B58U, 0xBB3BU, 0xAB1AU,
  0x6CA6U, 0x7C87U, 0x4CE4U, 0x5CC5U, 0x2C22U, 0x3C03U, 0x0C60U, 0x1C41U, 0xEDAEU, 0xFD8FU, 0xCDECU, 0xDDCDU,
  0xAD2AU, 0xBD0BU, 0x8D68U, 0x9D49U, 0x7E97U, 0x6EB6U, 0x5ED5U, 0x4EF4U, 0x3E13U, 0x2E32U, 0x1E51U, 0x0E70U,
  0xFF9FU, 0xEFBEU, 0xDFDDU, 0xCFFCU, 0xBF1BU, 0xAF3AU, 0x9F59U, 0x8F78U, 0x9188U, 0x81A9U, 0xB1CAU, 0xA1EBU,
  0xD10CU, 0xC12DU, 0xF14EU, 0xE16FU, 0x1080U, 0x00A1U, 0x30C2U, 0x20E3U, 0x5004U, 0x4025U, 0x7046U, 0x6067U,
  0x83B9U, 0x9398U, 0xA3FBU, 0xB3DAU, 0xC33DU, 0xD31CU, 0xE37FU, 0xF35EU, 0x02B1U, 0x1290U, 0x22F3U, 0x32D2U,
  0x4235U, 0x5214U, 0x6277U, 0x7256U, 0xB5EAU, 0xA5CBU, 0x95A8U, 0x8589U, 0xF56EU, 0xE54FU, 0xD52CU, 0xC50DU,
  0x34E2U, 0x24C3U, 0x14A0U, 0x0481U, 0x7466U, 0x6447U, 0x5424U, 0x4405U, 0xA7DBU, 0xB7FAU, 0x8799U, 0x97B8U,
  0xE75FU, 0xF77EU, 0xC71DU, 0xD73CU, 0x26D3U, 0x36F2U, 0x0691U, 0x16B0U, 0x6657U, 0x7676U, 0x4615U, 0x5634U,
  0xD94CU, 0xC96DU, 0xF90EU, 0xE92FU, 0x99C8U, 0x89E9U, 0xB98AU, 0xA9ABU, 0x5844U, 0x4865U, 0x7806U, 0x6827U,
  0x18C0U, 0x08E1U, 0x3882U, 0x28A3U, 0xCB7DU, 0xDB5CU, 0xEB3FU, 0xFB1EU, 0x8BF9U, 0x9BD8U, 0xABBBU, 0xBB9AU,
  0x4A75U, 0x5A54U, 0x6A37U, 0x7A16U, 0x0AF1U, 0x1AD0U, 0x2AB3U, 0x3A92U, 0xFD2EU, 0xED0FU, 0xDD6CU, 0xCD4DU,
  0xBDAAU, 0xAD8BU, 0x9DE8U, 0x8DC9U, 0x7C26U, 0x6C07U, 0x5C64U, 0x4C45U, 0x3CA2U, 0x2C83U, 0x1CE0U, 0x0CC1U,
  0xEF1FU, 0xFF3EU, 0xCF5DU, 0xDF7CU, 0xAF9BU, 0xBFBAU, 0x8FD9U, 0x9FF8U, 0x6E17U, 0x7E36U, 0x4E55U, 0x5E74U,
  0x2E93U, 0x3EB2U, 0x0ED1U, 0x1EF0U
};

static const uint8_t HYUNDAI_PREV_BUTTON_SAMPLES = 8;  // roughly 160 ms

extern const uint32_t HYUNDAI_STANDSTILL_THRSLD;
const uint32_t HYUNDAI_STANDSTILL_THRSLD = 12;  // 0.375 kph

enum {
  HYUNDAI_BTN_NONE = 0,
  HYUNDAI_BTN_RESUME = 1,
  HYUNDAI_BTN_SET = 2,
  HYUNDAI_BTN_CANCEL = 4,
};

// common state
extern bool hyundai_ev_gas_signal;
bool hyundai_ev_gas_signal = false;

extern bool hyundai_hybrid_gas_signal;
bool hyundai_hybrid_gas_signal = false;

extern bool hyundai_longitudinal;
bool hyundai_longitudinal = false;

extern bool hyundai_camera_scc;
bool hyundai_camera_scc = false;

extern bool hyundai_canfd_lka_steer_msg;
bool hyundai_canfd_lka_steer_msg = false;

extern bool hyundai_alt_limits;
bool hyundai_alt_limits = false;

extern bool hyundai_fcev_gas_signal;
bool hyundai_fcev_gas_signal = false;

extern bool hyundai_alt_limits_2;
bool hyundai_alt_limits_2 = false;

static uint8_t hyundai_last_button_interaction;  // button messages since the user pressed an enable button

void hyundai_common_init(uint16_t param) {
  const uint16_t HYUNDAI_PARAM_EV_GAS = 1;
  const uint16_t HYUNDAI_PARAM_HYBRID_GAS = 2;
  const uint16_t HYUNDAI_PARAM_CAMERA_SCC = 8;
  const uint16_t HYUNDAI_PARAM_CANFD_LKA_STEER_MSG = 16;
  const uint16_t HYUNDAI_PARAM_ALT_LIMITS = 64;  // TODO: shift this down with the rest of the common flags
  const uint16_t HYUNDAI_PARAM_FCEV_GAS = 256;
  const uint16_t HYUNDAI_PARAM_ALT_LIMITS_2 = 512;

  hyundai_ev_gas_signal = GET_FLAG(param, HYUNDAI_PARAM_EV_GAS);
  hyundai_hybrid_gas_signal = !hyundai_ev_gas_signal && GET_FLAG(param, HYUNDAI_PARAM_HYBRID_GAS);
  hyundai_camera_scc = GET_FLAG(param, HYUNDAI_PARAM_CAMERA_SCC);
  hyundai_canfd_lka_steer_msg = GET_FLAG(param, HYUNDAI_PARAM_CANFD_LKA_STEER_MSG);
  hyundai_alt_limits = GET_FLAG(param, HYUNDAI_PARAM_ALT_LIMITS);
  hyundai_fcev_gas_signal = GET_FLAG(param, HYUNDAI_PARAM_FCEV_GAS);
  hyundai_alt_limits_2 = GET_FLAG(param, HYUNDAI_PARAM_ALT_LIMITS_2);

  hyundai_last_button_interaction = HYUNDAI_PREV_BUTTON_SAMPLES;

#ifdef ALLOW_DEBUG
  const uint16_t HYUNDAI_PARAM_LONGITUDINAL = 4;
  hyundai_longitudinal = GET_FLAG(param, HYUNDAI_PARAM_LONGITUDINAL);
#else
  hyundai_longitudinal = false;
#endif
}

void hyundai_common_cruise_state_check(const bool cruise_engaged) {
  // some newer HKG models can re-enable after spamming cancel button,
  // so keep track of user button presses to deny engagement if no interaction

  // enter controls on rising edge of ACC and recent user button press, exit controls when ACC off
  if (!hyundai_longitudinal) {
    if (cruise_engaged && !cruise_engaged_prev && (hyundai_last_button_interaction < HYUNDAI_PREV_BUTTON_SAMPLES)) {
      controls_allowed = true;
    }

    if (!cruise_engaged) {
      controls_allowed = false;
    }
    cruise_engaged_prev = cruise_engaged;
  }
}

void hyundai_common_cruise_buttons_check(const int cruise_button, const bool main_button) {
  if ((cruise_button == HYUNDAI_BTN_RESUME) || (cruise_button == HYUNDAI_BTN_SET) || (cruise_button == HYUNDAI_BTN_CANCEL) || main_button) {
    hyundai_last_button_interaction = 0U;
  } else {
    hyundai_last_button_interaction = SAFETY_MIN(hyundai_last_button_interaction + 1U, HYUNDAI_PREV_BUTTON_SAMPLES);
  }

  if (hyundai_longitudinal) {
    // enter controls on falling edge of resume or set
    bool set = (cruise_button != HYUNDAI_BTN_SET) && (cruise_button_prev == HYUNDAI_BTN_SET);
    bool res = (cruise_button != HYUNDAI_BTN_RESUME) && (cruise_button_prev == HYUNDAI_BTN_RESUME);
    if (set || res) {
      controls_allowed = true;
    }

    // exit controls on cancel press
    if (cruise_button == HYUNDAI_BTN_CANCEL) {
      controls_allowed = false;
    }

    cruise_button_prev = cruise_button;
  }
}

uint32_t hyundai_common_canfd_compute_checksum(const CANPacket_t *msg) {
  int len = GET_LEN(msg);
  uint32_t address = msg->addr;

  uint16_t crc = 0;

  for (int i = 2; i < len; i++) {
    crc = (crc << 8U) ^ hyundai_canfd_crc_lut[(crc >> 8U) ^ msg->data[i]];
  }

  // Add address to crc
  crc = (crc << 8U) ^ hyundai_canfd_crc_lut[(crc >> 8U) ^ ((address >> 0U) & 0xFFU)];
  crc = (crc << 8U) ^ hyundai_canfd_crc_lut[(crc >> 8U) ^ ((address >> 8U) & 0xFFU)];

  if (len == 24) {
    crc ^= 0x819dU;
  }
  if (len == 32) {
    crc ^= 0x9f5bU;
  }

  return crc;
}
