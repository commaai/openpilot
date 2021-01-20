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
      puts("Invalid ESP/GPS mode\n");
      break;
  }
}

const board board_grey = {
  .board_type = "Grey",
  .harness_config = &white_harness_config,
  .init = grey_init,
  .enable_can_transceiver = white_enable_can_transceiver,
  .enable_can_transceivers = white_enable_can_transceivers,
  .set_led = white_set_led,
  .set_usb_power_mode = white_set_usb_power_mode,
  .set_gps_mode = grey_set_gps_mode,
  .set_can_mode = white_set_can_mode,
  .usb_power_mode_tick = white_usb_power_mode_tick,
  .check_ignition = white_check_ignition,
  .read_current = white_read_current,
  .set_fan_power = white_set_fan_power,
  .set_ir_power = white_set_ir_power,
  .set_phone_power = white_set_phone_power,
  .set_clock_source_mode = white_set_clock_source_mode,
  .set_siren = white_set_siren
};
