#pragma once

#include "board/drivers/drivers.h"

struct harness_t harness;

#ifdef ALLOW_DEBUG
bool relay_readback_inject = false;
uint16_t harness_adc2_offset_mV = 0U;
#endif

// The ignition relay is only used for testing purposes
void set_intercept_relay(bool intercept, bool ignition_relay) {
  bool drive_relay = intercept;
  if (harness.status == HARNESS_STATUS_NC) {
    // no harness, no relay to drive
    drive_relay = false;
  }

  if (drive_relay || ignition_relay) {
    harness.relay_driven = true;
  }

  // wait until we're not reading the analog voltages anymore
  while (harness.sbu_adc_lock) {}

  if (harness.status == HARNESS_STATUS_NORMAL) {
    set_gpio_output(current_board->harness_config->GPIO_relay_SBU1, current_board->harness_config->pin_relay_SBU1, !ignition_relay);
    set_gpio_output(current_board->harness_config->GPIO_relay_SBU2, current_board->harness_config->pin_relay_SBU2, !drive_relay);
  } else {
    set_gpio_output(current_board->harness_config->GPIO_relay_SBU1, current_board->harness_config->pin_relay_SBU1, !drive_relay);
    set_gpio_output(current_board->harness_config->GPIO_relay_SBU2, current_board->harness_config->pin_relay_SBU2, !ignition_relay);
  }

  if (!(drive_relay || ignition_relay)) {
    harness.relay_driven = false;
  }
}

// pins go analog in stop mode
static bool relay_pin_mismatch(const GPIO_TypeDef *GPIO, unsigned int pin) {
  bool is_output = ((GPIO->MODER >> (pin * 2U)) & 3U) == (uint32_t)MODE_OUTPUT;
  bool driven = (GPIO->ODR & (1UL << pin)) != 0U;
  bool read = get_gpio_input(GPIO, pin) != 0;
  return is_output && (driven != read);
}

bool harness_check_ignition(void) {
  bool ret = false;

  // wait until we're not reading the analog voltages anymore
  while (harness.sbu_adc_lock) {}

  switch(harness.status){
    case HARNESS_STATUS_NORMAL:
      ret = !get_gpio_input(current_board->harness_config->GPIO_SBU1, current_board->harness_config->pin_SBU1);
      break;
    case HARNESS_STATUS_FLIPPED:
      ret = !get_gpio_input(current_board->harness_config->GPIO_SBU2, current_board->harness_config->pin_SBU2);
      break;
    default:
      break;
  }
  return ret;
}

static uint8_t harness_detect_orientation(void) {
  uint8_t ret = harness.status;
  static uint32_t adc_mismatches = 0U;
  static uint16_t sbu1_adc2_mV = 0U;

  #ifndef BOOTSTUB
  // We can't detect orientation if the relay is being driven
  if (!harness.relay_driven) {
    harness.sbu_adc_lock = true;
    set_gpio_mode(current_board->harness_config->GPIO_SBU1, current_board->harness_config->pin_SBU1, MODE_ANALOG);
    set_gpio_mode(current_board->harness_config->GPIO_SBU2, current_board->harness_config->pin_SBU2, MODE_ANALOG);

    harness.sbu1_voltage_mV = adc_get_mV(&current_board->harness_config->adc_signal_SBU1);
    harness.sbu2_voltage_mV = adc_get_mV(&current_board->harness_config->adc_signal_SBU2);
    uint16_t detection_threshold = current_board->avdd_mV / 2U;

    // ADC_SM_4: SBU1 is on ADC1 and ADC2
    if (!adc2_busy) {
      adc_signal_t sbu1_adc2 = current_board->harness_config->adc_signal_SBU1;
      sbu1_adc2.adc = ADC2;
      sbu1_adc2_mV = adc_get_mV(&sbu1_adc2);
      #ifdef ALLOW_DEBUG
      sbu1_adc2_mV += harness_adc2_offset_mV;
      #endif
      adc_mismatches = (ABS((int32_t)harness.sbu1_voltage_mV - (int32_t)sbu1_adc2_mV) <= 100) ? 0U : (adc_mismatches + 1U);
    }

    // Detect connection and orientation
    if((harness.sbu1_voltage_mV < detection_threshold) || (harness.sbu2_voltage_mV < detection_threshold)){
      if (harness.sbu1_voltage_mV < harness.sbu2_voltage_mV) {
        // orientation flipped (PANDA_SBU1->HARNESS_SBU1(relay), PANDA_SBU2->HARNESS_SBU2(ign))
        ret = HARNESS_STATUS_FLIPPED;
      } else {
        // orientation normal (PANDA_SBU2->HARNESS_SBU1(relay), PANDA_SBU1->HARNESS_SBU2(ign))
        // (SBU1->SBU2 is the normal orientation connection per USB-C cable spec)
        ret = HARNESS_STATUS_NORMAL;
      }
    } else {
      ret = HARNESS_STATUS_NC;
    }

    // Pins are not 5V tolerant in ADC mode
    set_gpio_mode(current_board->harness_config->GPIO_SBU1, current_board->harness_config->pin_SBU1, MODE_INPUT);
    set_gpio_mode(current_board->harness_config->GPIO_SBU2, current_board->harness_config->pin_SBU2, MODE_INPUT);
    harness.sbu_adc_lock = false;
  }
  #endif

  // after the ADC lock, the safe state drives the relay
  if (adc_mismatches >= 8U) {
    fault_occurred_value(FAULT_ADC, ((uint32_t)harness.sbu1_voltage_mV << 16) | sbu1_adc2_mV);
  }

  return ret;
}

void harness_tick(void) {
  harness.status = harness_detect_orientation();

  // GPIO_SM_2: relay pins read back what is driven, one tick of slack for switching
  static uint32_t relay_mismatches = 0U;
  const harness_configuration *cfg = current_board->harness_config;
  bool mismatch = relay_pin_mismatch(cfg->GPIO_relay_SBU1, cfg->pin_relay_SBU1) ||
                  relay_pin_mismatch(cfg->GPIO_relay_SBU2, cfg->pin_relay_SBU2);
  #ifdef ALLOW_DEBUG
  mismatch = mismatch || relay_readback_inject;
  #endif
  relay_mismatches = mismatch ? (relay_mismatches + 1U) : 0U;
  if (relay_mismatches >= 2U) {
    fault_occurred(FAULT_OUTPUT_READBACK);
  }
}

void harness_init(void) {
  // init OBD_SBUx_RELAY
  set_gpio_output_type(current_board->harness_config->GPIO_relay_SBU1, current_board->harness_config->pin_relay_SBU1, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_output_type(current_board->harness_config->GPIO_relay_SBU2, current_board->harness_config->pin_relay_SBU2, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_output(current_board->harness_config->GPIO_relay_SBU1, current_board->harness_config->pin_relay_SBU1, 1);
  set_gpio_output(current_board->harness_config->GPIO_relay_SBU2, current_board->harness_config->pin_relay_SBU2, 1);

  // detect initial orientation
  harness.status = harness_detect_orientation();

  // keep buses connected by default
  set_intercept_relay(false, false);
}
