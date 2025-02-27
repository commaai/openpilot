#pragma once

#include "board_declarations.h"

// //////////////////// //
// Grey Panda (STM32F4) //
// //////////////////// //

// Most hardware functionality is similar to white panda

board board_grey = {
  .set_bootkick = unused_set_bootkick,
  .harness_config = &white_harness_config,
  .has_obd = false,
  .has_spi = false,
  .has_canfd = false,
  .fan_max_rpm = 0U,
  .fan_max_pwm = 100U,
  .avdd_mV = 3300U,
  .fan_stall_recovery = false,
  .fan_enable_cooldown_time = 0U,
  .init = white_grey_init,
  .init_bootloader = white_grey_init_bootloader,
  .enable_can_transceiver = white_enable_can_transceiver,
  .enable_can_transceivers = white_enable_can_transceivers,
  .set_led = white_set_led,
  .set_can_mode = white_set_can_mode,
  .check_ignition = white_check_ignition,
  .read_voltage_mV = white_read_voltage_mV,
  .read_current_mA = white_read_current_mA,
  .set_fan_enabled = unused_set_fan_enabled,
  .set_ir_power = unused_set_ir_power,
  .set_siren = unused_set_siren,
  .read_som_gpio = unused_read_som_gpio,
  .set_amp_enabled = unused_set_amp_enabled
};
