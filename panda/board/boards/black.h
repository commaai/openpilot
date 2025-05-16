#pragma once

#include "board_declarations.h"

// /////////////////////////////// //
// Black Panda (STM32F4) + Harness //
// /////////////////////////////// //

static void black_enable_can_transceiver(uint8_t transceiver, bool enabled) {
  switch (transceiver){
    case 1U:
      set_gpio_output(GPIOC, 1, !enabled);
      break;
    case 2U:
      set_gpio_output(GPIOC, 13, !enabled);
      break;
    case 3U:
      set_gpio_output(GPIOA, 0, !enabled);
      break;
    case 4U:
      set_gpio_output(GPIOB, 10, !enabled);
      break;
    default:
      print("Invalid CAN transceiver ("); puth(transceiver); print("): enabling failed\n");
      break;
  }
}

static void black_set_usb_load_switch(bool enabled) {
  set_gpio_output(GPIOB, 1, !enabled);
}

static void black_set_can_mode(uint8_t mode) {
  black_enable_can_transceiver(2U, false);
  black_enable_can_transceiver(4U, false);
  switch (mode) {
    case CAN_MODE_NORMAL:
    case CAN_MODE_OBD_CAN2:
      if ((bool)(mode == CAN_MODE_NORMAL) != (bool)(harness.status == HARNESS_STATUS_FLIPPED)) {
        // B12,B13: disable OBD mode
        set_gpio_mode(GPIOB, 12, MODE_INPUT);
        set_gpio_mode(GPIOB, 13, MODE_INPUT);

        // B5,B6: normal CAN2 mode
        set_gpio_alternate(GPIOB, 5, GPIO_AF9_CAN2);
        set_gpio_alternate(GPIOB, 6, GPIO_AF9_CAN2);
        black_enable_can_transceiver(2U, true);

      } else {
        // B5,B6: disable normal CAN2 mode
        set_gpio_mode(GPIOB, 5, MODE_INPUT);
        set_gpio_mode(GPIOB, 6, MODE_INPUT);

        // B12,B13: OBD mode
        set_gpio_alternate(GPIOB, 12, GPIO_AF9_CAN2);
        set_gpio_alternate(GPIOB, 13, GPIO_AF9_CAN2);
        black_enable_can_transceiver(4U, true);
      }
      break;
    default:
      print("Tried to set unsupported CAN mode: "); puth(mode); print("\n");
      break;
  }
}

static bool black_check_ignition(void){
  // ignition is checked through harness
  return harness_check_ignition();
}

static void black_init(void) {
  common_init_gpio();

  // A8,A15: normal CAN3 mode
  set_gpio_alternate(GPIOA, 8, GPIO_AF11_CAN3);
  set_gpio_alternate(GPIOA, 15, GPIO_AF11_CAN3);

  // GPS OFF
  set_gpio_output(GPIOC, 5, 0);
  set_gpio_output(GPIOC, 12, 0);

  // Turn on USB load switch.
  black_set_usb_load_switch(true);
}

static void black_init_bootloader(void) {
  // GPS OFF
  set_gpio_output(GPIOC, 5, 0);
  set_gpio_output(GPIOC, 12, 0);
}

static harness_configuration black_harness_config = {
  .has_harness = true,
  .GPIO_SBU1 = GPIOC,
  .GPIO_SBU2 = GPIOC,
  .GPIO_relay_SBU1 = GPIOC,
  .GPIO_relay_SBU2 = GPIOC,
  .pin_SBU1 = 0,
  .pin_SBU2 = 3,
  .pin_relay_SBU1 = 10,
  .pin_relay_SBU2 = 11,
  .adc_channel_SBU1 = 10,
  .adc_channel_SBU2 = 13
};

board board_black = {
  .set_bootkick = unused_set_bootkick,
  .harness_config = &black_harness_config,
  .has_spi = false,
  .has_canfd = false,
  .fan_max_rpm = 0U,
  .fan_max_pwm = 100U,
  .avdd_mV = 3300U,
  .fan_stall_recovery = false,
  .fan_enable_cooldown_time = 0U,
  .init = black_init,
  .init_bootloader = black_init_bootloader,
  .enable_can_transceiver = black_enable_can_transceiver,
  .led_GPIO = {GPIOC, GPIOC, GPIOC},
  .led_pin = {9, 7, 6},
  .set_can_mode = black_set_can_mode,
  .check_ignition = black_check_ignition,
  .read_voltage_mV = white_read_voltage_mV,
  .read_current_mA = unused_read_current,
  .set_fan_enabled = unused_set_fan_enabled,
  .set_ir_power = unused_set_ir_power,
  .set_siren = unused_set_siren,
  .read_som_gpio = unused_read_som_gpio,
  .set_amp_enabled = unused_set_amp_enabled
};
