// ///////////////////// //
// Red Panda V2 with chiplet + Harness //
// ///////////////////// //

void red_panda_v2_init(void) {
  // common chiplet init
  red_chiplet_init();

  // Turn on USB load switch
  red_chiplet_set_fan_or_usb_load_switch(true);
}

const board board_red_v2 = {
  .board_type = "Red_v2",
  .board_tick = unused_board_tick,
  .harness_config = &red_chiplet_harness_config,
  .has_gps = false,
  .has_hw_gmlan = false,
  .has_obd = true,
  .has_lin = false,
  .has_spi = false,
  .has_canfd = true,
  .has_rtc_battery = true,
  .fan_max_rpm = 0U,
  .avdd_mV = 3300U,
  .fan_stall_recovery = false,
  .fan_enable_cooldown_time = 0U,
  .init = red_panda_v2_init,
  .enable_can_transceiver = red_chiplet_enable_can_transceiver,
  .enable_can_transceivers = red_chiplet_enable_can_transceivers,
  .set_led = red_set_led,
  .set_gps_mode = unused_set_gps_mode,
  .set_can_mode = red_set_can_mode,
  .check_ignition = red_check_ignition,
  .read_current = unused_read_current,
  .set_fan_enabled = unused_set_fan_enabled,
  .set_ir_power = unused_set_ir_power,
  .set_phone_power = unused_set_phone_power,
  .set_siren = unused_set_siren,
  .read_som_gpio = unused_read_som_gpio
};
