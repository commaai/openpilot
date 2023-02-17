// ////////// //
// Grey Panda //
// ////////// //

// Most hardware functionality is similar to white panda

void grey_init(void) {
  white_grey_common_init();

  // Set default state of GPS
  current_board->set_gps_mode(GPS_ENABLED);
}

void grey_set_gps_mode(uint8_t mode) {
  switch (mode) {
    case GPS_DISABLED:
      // GPS OFF
      set_gpio_output(GPIOC, 14, 0);
      set_gpio_output(GPIOC, 5, 0);
      break;
    case GPS_ENABLED:
      // GPS ON
      set_gpio_output(GPIOC, 14, 1);
      set_gpio_output(GPIOC, 5, 1);
      break;
    case GPS_BOOTMODE:
      set_gpio_output(GPIOC, 14, 1);
      set_gpio_output(GPIOC, 5, 0);
      break;
    default:
      print("Invalid ESP/GPS mode\n");
      break;
  }
}

const board board_grey = {
  .board_type = "Grey",
  .board_tick = unused_board_tick,
  .harness_config = &white_harness_config,
  .has_gps = true,
  .has_hw_gmlan = true,
  .has_obd = false,
  .has_lin = true,
  .has_spi = false,
  .has_canfd = false,
  .has_rtc_battery = false,
  .fan_max_rpm = 0U,
  .init = grey_init,
  .enable_can_transceiver = white_enable_can_transceiver,
  .enable_can_transceivers = white_enable_can_transceivers,
  .set_led = white_set_led,
  .set_gps_mode = grey_set_gps_mode,
  .set_can_mode = white_set_can_mode,
  .check_ignition = white_check_ignition,
  .read_current = white_read_current,
  .set_fan_enabled = unused_set_fan_enabled,
  .set_ir_power = unused_set_ir_power,
  .set_phone_power = unused_set_phone_power,
  .set_siren = unused_set_siren,
  .read_som_gpio = unused_read_som_gpio
};
