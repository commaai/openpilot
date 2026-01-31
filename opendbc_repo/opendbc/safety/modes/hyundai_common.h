#pragma once

#include "opendbc/safety/declarations.h"

extern uint16_t hyundai_canfd_crc_lut[256];
uint16_t hyundai_canfd_crc_lut[256];

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

extern bool hyundai_canfd_lka_steering;
bool hyundai_canfd_lka_steering = false;

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
  const uint16_t HYUNDAI_PARAM_CANFD_LKA_STEERING = 16;
  const uint16_t HYUNDAI_PARAM_ALT_LIMITS = 64; // TODO: shift this down with the rest of the common flags
  const uint16_t HYUNDAI_PARAM_FCEV_GAS = 256;
  const uint16_t HYUNDAI_PARAM_ALT_LIMITS_2 = 512;

  hyundai_ev_gas_signal = GET_FLAG(param, HYUNDAI_PARAM_EV_GAS);
  hyundai_hybrid_gas_signal = !hyundai_ev_gas_signal && GET_FLAG(param, HYUNDAI_PARAM_HYBRID_GAS);
  hyundai_camera_scc = GET_FLAG(param, HYUNDAI_PARAM_CAMERA_SCC);
  hyundai_canfd_lka_steering = GET_FLAG(param, HYUNDAI_PARAM_CANFD_LKA_STEERING);
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
  } else if (len == 32) {
    crc ^= 0x9f5bU;
  } else {

  }

  return crc;
}
