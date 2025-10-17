#include "harness_declarations.h"

struct harness_t harness;

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

  #ifndef BOOTSTUB
  // We can't detect orientation if the relay is being driven
  if (!harness.relay_driven) {
    harness.sbu_adc_lock = true;
    set_gpio_mode(current_board->harness_config->GPIO_SBU1, current_board->harness_config->pin_SBU1, MODE_ANALOG);
    set_gpio_mode(current_board->harness_config->GPIO_SBU2, current_board->harness_config->pin_SBU2, MODE_ANALOG);

    harness.sbu1_voltage_mV = adc_get_mV(&current_board->harness_config->adc_signal_SBU1);
    harness.sbu2_voltage_mV = adc_get_mV(&current_board->harness_config->adc_signal_SBU2);
    uint16_t detection_threshold = current_board->avdd_mV / 2U;

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

  return ret;
}

void harness_tick(void) {
  harness.status = harness_detect_orientation();
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
