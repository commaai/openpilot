// ///////////////////// //
// Black Panda + Harness //
// ///////////////////// //

void black_enable_can_transceiver(uint8_t transceiver, bool enabled) {
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

void black_enable_can_transceivers(bool enabled) {
  for(uint8_t i=1U; i<=4U; i++){
    // Leave main CAN always on for CAN-based ignition detection
    if((car_harness_status == HARNESS_STATUS_FLIPPED) ? (i == 3U) : (i == 1U)){
      black_enable_can_transceiver(i, true);
    } else {
      black_enable_can_transceiver(i, enabled);
    }
  }
}

void black_set_led(uint8_t color, bool enabled) {
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

void black_set_gps_load_switch(bool enabled) {
  set_gpio_output(GPIOC, 12, enabled);
}

void black_set_usb_load_switch(bool enabled) {
  set_gpio_output(GPIOB, 1, !enabled);
}

void black_set_gps_mode(uint8_t mode) {
  switch (mode) {
    case GPS_DISABLED:
      // GPS OFF
      set_gpio_output(GPIOC, 12, 0);
      set_gpio_output(GPIOC, 5, 0);
      break;
    case GPS_ENABLED:
      // GPS ON
      set_gpio_output(GPIOC, 12, 1);
      set_gpio_output(GPIOC, 5, 1);
      break;
    case GPS_BOOTMODE:
      set_gpio_output(GPIOC, 12, 1);
      set_gpio_output(GPIOC, 5, 0);
      break;
    default:
      print("Invalid GPS mode\n");
      break;
  }
}

void black_set_can_mode(uint8_t mode){
  switch (mode) {
    case CAN_MODE_NORMAL:
    case CAN_MODE_OBD_CAN2:
      if ((bool)(mode == CAN_MODE_NORMAL) != (bool)(car_harness_status == HARNESS_STATUS_FLIPPED)) {
        // B12,B13: disable OBD mode
        set_gpio_mode(GPIOB, 12, MODE_INPUT);
        set_gpio_mode(GPIOB, 13, MODE_INPUT);

        // B5,B6: normal CAN2 mode
        set_gpio_alternate(GPIOB, 5, GPIO_AF9_CAN2);
        set_gpio_alternate(GPIOB, 6, GPIO_AF9_CAN2);
      } else {
        // B5,B6: disable normal CAN2 mode
        set_gpio_mode(GPIOB, 5, MODE_INPUT);
        set_gpio_mode(GPIOB, 6, MODE_INPUT);

        // B12,B13: OBD mode
        set_gpio_alternate(GPIOB, 12, GPIO_AF9_CAN2);
        set_gpio_alternate(GPIOB, 13, GPIO_AF9_CAN2);
      }
      break;
    default:
      print("Tried to set unsupported CAN mode: "); puth(mode); print("\n");
      break;
  }
}

bool black_check_ignition(void){
  // ignition is checked through harness
  return harness_check_ignition();
}

void black_init(void) {
  common_init_gpio();

  // A8,A15: normal CAN3 mode
  set_gpio_alternate(GPIOA, 8, GPIO_AF11_CAN3);
  set_gpio_alternate(GPIOA, 15, GPIO_AF11_CAN3);

  // C0: OBD_SBU1 (orientation detection)
  // C3: OBD_SBU2 (orientation detection)
  set_gpio_mode(GPIOC, 0, MODE_ANALOG);
  set_gpio_mode(GPIOC, 3, MODE_ANALOG);

  // Set default state of GPS
  current_board->set_gps_mode(GPS_ENABLED);

  // C10: OBD_SBU1_RELAY (harness relay driving output)
  // C11: OBD_SBU2_RELAY (harness relay driving output)
  set_gpio_mode(GPIOC, 10, MODE_OUTPUT);
  set_gpio_mode(GPIOC, 11, MODE_OUTPUT);
  set_gpio_output_type(GPIOC, 10, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_output_type(GPIOC, 11, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_output(GPIOC, 10, 1);
  set_gpio_output(GPIOC, 11, 1);

  // Turn on GPS load switch.
  black_set_gps_load_switch(true);

  // Turn on USB load switch.
  black_set_usb_load_switch(true);

  // Initialize harness
  harness_init();

  // Initialize RTC
  rtc_init();

  // Enable CAN transceivers
  black_enable_can_transceivers(true);

  // Disable LEDs
  black_set_led(LED_RED, false);
  black_set_led(LED_GREEN, false);
  black_set_led(LED_BLUE, false);

  // Set normal CAN mode
  black_set_can_mode(CAN_MODE_NORMAL);

  // flip CAN0 and CAN2 if we are flipped
  if (car_harness_status == HARNESS_STATUS_FLIPPED) {
    can_flip_buses(0, 2);
  }
}

const harness_configuration black_harness_config = {
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

const board board_black = {
  .board_type = "Black",
  .board_tick = unused_board_tick,
  .harness_config = &black_harness_config,
  .has_gps = true,
  .has_hw_gmlan = false,
  .has_obd = true,
  .has_lin = false,
  .has_spi = false,
  .has_canfd = false,
  .has_rtc_battery = false,
  .fan_max_rpm = 0U,
  .init = black_init,
  .enable_can_transceiver = black_enable_can_transceiver,
  .enable_can_transceivers = black_enable_can_transceivers,
  .set_led = black_set_led,
  .set_gps_mode = black_set_gps_mode,
  .set_can_mode = black_set_can_mode,
  .check_ignition = black_check_ignition,
  .read_current = unused_read_current,
  .set_fan_enabled = unused_set_fan_enabled,
  .set_ir_power = unused_set_ir_power,
  .set_phone_power = unused_set_phone_power,
  .set_siren = unused_set_siren
};
