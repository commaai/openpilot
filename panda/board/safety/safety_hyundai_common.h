#ifndef SAFETY_HYUNDAI_COMMON_H
#define SAFETY_HYUNDAI_COMMON_H

const int HYUNDAI_PARAM_EV_GAS = 1;
const int HYUNDAI_PARAM_HYBRID_GAS = 2;
const int HYUNDAI_PARAM_LONGITUDINAL = 4;
const int HYUNDAI_PARAM_CAMERA_SCC = 8;
const int HYUNDAI_PARAM_CANFD_HDA2 = 16;
const int HYUNDAI_PARAM_ALT_LIMITS = 64; // TODO: shift this down with the rest of the common flags

const uint8_t HYUNDAI_PREV_BUTTON_SAMPLES = 8;  // roughly 160 ms
const uint32_t HYUNDAI_STANDSTILL_THRSLD = 12;  // 0.375 kph

enum {
  HYUNDAI_BTN_NONE = 0,
  HYUNDAI_BTN_RESUME = 1,
  HYUNDAI_BTN_SET = 2,
  HYUNDAI_BTN_CANCEL = 4,
};

// common state
bool hyundai_ev_gas_signal = false;
bool hyundai_hybrid_gas_signal = false;
bool hyundai_longitudinal = false;
bool hyundai_camera_scc = false;
bool hyundai_canfd_hda2 = false;
bool hyundai_alt_limits = false;
uint8_t hyundai_last_button_interaction;  // button messages since the user pressed an enable button

void hyundai_common_init(uint16_t param) {
  hyundai_ev_gas_signal = GET_FLAG(param, HYUNDAI_PARAM_EV_GAS);
  hyundai_hybrid_gas_signal = !hyundai_ev_gas_signal && GET_FLAG(param, HYUNDAI_PARAM_HYBRID_GAS);
  hyundai_camera_scc = GET_FLAG(param, HYUNDAI_PARAM_CAMERA_SCC);
  hyundai_canfd_hda2 = GET_FLAG(param, HYUNDAI_PARAM_CANFD_HDA2);
  hyundai_alt_limits = GET_FLAG(param, HYUNDAI_PARAM_ALT_LIMITS);

  hyundai_last_button_interaction = HYUNDAI_PREV_BUTTON_SAMPLES;

#ifdef ALLOW_DEBUG
  hyundai_longitudinal = GET_FLAG(param, HYUNDAI_PARAM_LONGITUDINAL);
#else
  hyundai_longitudinal = false;
#endif
}

void hyundai_common_cruise_state_check(const int cruise_engaged) {
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

void hyundai_common_cruise_buttons_check(const int cruise_button, const int main_button) {
  if ((cruise_button == HYUNDAI_BTN_RESUME) || (cruise_button == HYUNDAI_BTN_SET) || (cruise_button == HYUNDAI_BTN_CANCEL) ||
      (main_button != 0)) {
    hyundai_last_button_interaction = 0U;
  } else {
    hyundai_last_button_interaction = MIN(hyundai_last_button_interaction + 1U, HYUNDAI_PREV_BUTTON_SAMPLES);
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

#endif
