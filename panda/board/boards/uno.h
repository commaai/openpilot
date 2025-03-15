#pragma once

#include "board_declarations.h"

// /////////////////////// //
// Uno (STM32F4) + Harness //
// /////////////////////// //

static void uno_enable_can_transceiver(uint8_t transceiver, bool enabled) {
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

static void uno_enable_can_transceivers(bool enabled) {
  for(uint8_t i=1U; i<=4U; i++){
    // Leave main CAN always on for CAN-based ignition detection
    if((harness.status == HARNESS_STATUS_FLIPPED) ? (i == 3U) : (i == 1U)){
      uno_enable_can_transceiver(i, true);
    } else {
      uno_enable_can_transceiver(i, enabled);
    }
  }
}

static void uno_set_led(uint8_t color, bool enabled) {
  switch (color){
    case LED_RED:
      set_gpio_output(GPIOC, 9, !enabled);
      break;
     case LED_GREEN:
      set_gpio_output(GPIOC, 7, !enabled);
      break;
    case LED_BLUE:
      set_gpio_output(GPIOC, 6, !enabled);
      break;
    default:
      break;
  }
}

static void uno_set_bootkick(BootState state) {
  if (state == BOOT_BOOTKICK) {
    set_gpio_output(GPIOB, 14, false);
  } else {
    // We want the pin to be floating, not forced high!
    set_gpio_mode(GPIOB, 14, MODE_INPUT);
  }
}

static void uno_set_can_mode(uint8_t mode) {
  uno_enable_can_transceiver(2U, false);
  uno_enable_can_transceiver(4U, false);
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
        uno_enable_can_transceiver(2U, true);
      } else {
        // B5,B6: disable normal CAN2 mode
        set_gpio_mode(GPIOB, 5, MODE_INPUT);
        set_gpio_mode(GPIOB, 6, MODE_INPUT);

        // B12,B13: OBD mode
        set_gpio_alternate(GPIOB, 12, GPIO_AF9_CAN2);
        set_gpio_alternate(GPIOB, 13, GPIO_AF9_CAN2);
        uno_enable_can_transceiver(4U, true);
      }
      break;
    default:
      print("Tried to set unsupported CAN mode: "); puth(mode); print("\n");
      break;
  }
}

static bool uno_check_ignition(void){
  // ignition is checked through harness
  return harness_check_ignition();
}

static void uno_set_usb_switch(bool phone){
  set_gpio_output(GPIOB, 3, phone);
}

static void uno_set_ir_power(uint8_t percentage){
  pwm_set(TIM4, 2, percentage);
}

static void uno_set_fan_enabled(bool enabled){
  set_gpio_output(GPIOA, 1, enabled);
}

static void uno_init(void) {
  common_init_gpio();

  // A8,A15: normal CAN3 mode
  set_gpio_alternate(GPIOA, 8, GPIO_AF11_CAN3);
  set_gpio_alternate(GPIOA, 15, GPIO_AF11_CAN3);

  // C0: OBD_SBU1 (orientation detection)
  // C3: OBD_SBU2 (orientation detection)
  set_gpio_mode(GPIOC, 0, MODE_ANALOG);
  set_gpio_mode(GPIOC, 3, MODE_ANALOG);

  // GPS off
  set_gpio_output(GPIOB, 1, 0);
  set_gpio_output(GPIOC, 5, 0);
  set_gpio_output(GPIOC, 12, 0);

  // C10: OBD_SBU1_RELAY (harness relay driving output)
  // C11: OBD_SBU2_RELAY (harness relay driving output)
  set_gpio_mode(GPIOC, 10, MODE_OUTPUT);
  set_gpio_mode(GPIOC, 11, MODE_OUTPUT);
  set_gpio_output_type(GPIOC, 10, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_output_type(GPIOC, 11, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_output(GPIOC, 10, 1);
  set_gpio_output(GPIOC, 11, 1);

  // C8: FAN PWM aka TIM3_CH3
  set_gpio_alternate(GPIOC, 8, GPIO_AF2_TIM3);

  // Turn on phone regulator
  set_gpio_output(GPIOB, 4, true);

  // Initialize IR PWM and set to 0%
  set_gpio_alternate(GPIOB, 7, GPIO_AF2_TIM4);
  pwm_init(TIM4, 2);
  uno_set_ir_power(0U);

  // Initialize harness
  harness_init();


  // Enable CAN transceivers
  uno_enable_can_transceivers(true);

  // Disable LEDs
  uno_set_led(LED_RED, false);
  uno_set_led(LED_GREEN, false);
  uno_set_led(LED_BLUE, false);

  // Set normal CAN mode
  uno_set_can_mode(CAN_MODE_NORMAL);

  // Switch to phone usb mode if harness connection is powered by less than 7V
  if(white_read_voltage_mV() < 7000U){
    uno_set_usb_switch(true);
  } else {
    uno_set_usb_switch(false);
  }

  // Bootkick phone
  uno_set_bootkick(BOOT_BOOTKICK);
}

static void uno_init_bootloader(void) {
  // GPS off
  set_gpio_output(GPIOB, 1, 0);
  set_gpio_output(GPIOC, 5, 0);
  set_gpio_output(GPIOC, 12, 0);
}

static harness_configuration uno_harness_config = {
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

board board_uno = {
  .harness_config = &uno_harness_config,
  .has_obd = true,
  .has_spi = false,
  .has_canfd = false,
  .fan_max_rpm = 5100U,
  .fan_max_pwm = 100U,
  .avdd_mV = 3300U,
  .fan_stall_recovery = false,
  .fan_enable_cooldown_time = 0U,
  .init = uno_init,
  .init_bootloader = uno_init_bootloader,
  .enable_can_transceiver = uno_enable_can_transceiver,
  .enable_can_transceivers = uno_enable_can_transceivers,
  .set_led = uno_set_led,
  .set_can_mode = uno_set_can_mode,
  .check_ignition = uno_check_ignition,
  .read_voltage_mV = white_read_voltage_mV,
  .read_current_mA = unused_read_current,
  .set_fan_enabled = uno_set_fan_enabled,
  .set_ir_power = uno_set_ir_power,
  .set_siren = unused_set_siren,
  .set_bootkick = uno_set_bootkick,
  .read_som_gpio = unused_read_som_gpio,
  .set_amp_enabled = unused_set_amp_enabled
};
