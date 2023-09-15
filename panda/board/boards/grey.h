// ////////// //
// Grey Panda //
// ////////// //

// Most hardware functionality is similar to white panda

const board board_grey = {
  .board_type = "Grey",
  .board_tick = unused_board_tick,
  .harness_config = &white_harness_config,
  .has_hw_gmlan = true,
  .has_obd = false,
  .has_lin = true,
  .has_spi = false,
  .has_canfd = false,
  .has_rtc_battery = false,
  .fan_max_rpm = 0U,
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
  .read_current = white_read_current,
  .set_fan_enabled = unused_set_fan_enabled,
  .set_ir_power = unused_set_ir_power,
  .set_phone_power = unused_set_phone_power,
  .set_siren = unused_set_siren,
  .read_som_gpio = unused_read_som_gpio
};
