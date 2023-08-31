// ///////////// //
// Dos + Harness //
// ///////////// //

void dos_enable_can_transceiver(uint8_t transceiver, bool enabled) {
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

void dos_enable_can_transceivers(bool enabled) {
  for(uint8_t i=1U; i<=4U; i++){
    // Leave main CAN always on for CAN-based ignition detection
    if((harness.status == HARNESS_STATUS_FLIPPED) ? (i == 3U) : (i == 1U)){
      dos_enable_can_transceiver(i, true);
    } else {
      dos_enable_can_transceiver(i, enabled);
    }
  }
}

void dos_set_led(uint8_t color, bool enabled) {
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

void dos_set_bootkick(bool enabled){
  set_gpio_output(GPIOC, 4, !enabled);
}

bool dos_board_tick(bool ignition, bool usb_enum, bool heartbeat_seen, bool harness_inserted) {
  bool ret = false;
  if ((ignition && !usb_enum) || harness_inserted) {
    // enable bootkick if ignition seen or if plugged into a harness
    ret = true;
    dos_set_bootkick(true);
  } else if (heartbeat_seen) {
    // disable once openpilot is up
    dos_set_bootkick(false);
  } else {

  }
  return ret;
}

void dos_set_can_mode(uint8_t mode) {
  dos_enable_can_transceiver(2U, false);
  dos_enable_can_transceiver(4U, false);
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
        dos_enable_can_transceiver(2U, true);
      } else {
        // B5,B6: disable normal CAN2 mode
        set_gpio_mode(GPIOB, 5, MODE_INPUT);
        set_gpio_mode(GPIOB, 6, MODE_INPUT);

        // B12,B13: OBD mode
        set_gpio_alternate(GPIOB, 12, GPIO_AF9_CAN2);
        set_gpio_alternate(GPIOB, 13, GPIO_AF9_CAN2);
        dos_enable_can_transceiver(4U, true);
      }
      break;
    default:
      print("Tried to set unsupported CAN mode: "); puth(mode); print("\n");
      break;
  }
}

bool dos_check_ignition(void){
  // ignition is checked through harness
  return harness_check_ignition();
}

void dos_set_usb_switch(bool phone){
  set_gpio_output(GPIOB, 3, phone);
}

void dos_set_ir_power(uint8_t percentage){
  pwm_set(TIM4, 2, percentage);
}

void dos_set_fan_enabled(bool enabled){
  set_gpio_output(GPIOA, 1, enabled);
}

void dos_set_siren(bool enabled){
  set_gpio_output(GPIOC, 12, enabled);
}

bool dos_read_som_gpio (void){
  return (get_gpio_input(GPIOC, 2) != 0);
}

void dos_init(void) {
  common_init_gpio();

  // A8,A15: normal CAN3 mode
  set_gpio_alternate(GPIOA, 8, GPIO_AF11_CAN3);
  set_gpio_alternate(GPIOA, 15, GPIO_AF11_CAN3);

  // C0: OBD_SBU1 (orientation detection)
  // C3: OBD_SBU2 (orientation detection)
  set_gpio_mode(GPIOC, 0, MODE_ANALOG);
  set_gpio_mode(GPIOC, 3, MODE_ANALOG);

  // C10: OBD_SBU1_RELAY (harness relay driving output)
  // C11: OBD_SBU2_RELAY (harness relay driving output)
  set_gpio_mode(GPIOC, 10, MODE_OUTPUT);
  set_gpio_mode(GPIOC, 11, MODE_OUTPUT);
  set_gpio_output_type(GPIOC, 10, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_output_type(GPIOC, 11, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_output(GPIOC, 10, 1);
  set_gpio_output(GPIOC, 11, 1);

#ifdef ENABLE_SPI
  // SPI init
  gpio_spi_init();
#endif

  // C8: FAN PWM aka TIM3_CH3
  set_gpio_alternate(GPIOC, 8, GPIO_AF2_TIM3);

  // C2: SOM GPIO used as input (fan control at boot)
  set_gpio_mode(GPIOC, 2, MODE_INPUT);
  set_gpio_pullup(GPIOC, 2, PULL_DOWN);

  // Initialize IR PWM and set to 0%
  set_gpio_alternate(GPIOB, 7, GPIO_AF2_TIM4);
  pwm_init(TIM4, 2);
  dos_set_ir_power(0U);

  // Initialize harness
  harness_init();

  // Initialize RTC
  rtc_init();

  // Enable CAN transceivers
  dos_enable_can_transceivers(true);

  // Disable LEDs
  dos_set_led(LED_RED, false);
  dos_set_led(LED_GREEN, false);
  dos_set_led(LED_BLUE, false);

  // Bootkick
  dos_set_bootkick(true);

  // Set normal CAN mode
  dos_set_can_mode(CAN_MODE_NORMAL);

  // flip CAN0 and CAN2 if we are flipped
  if (harness.status == HARNESS_STATUS_FLIPPED) {
    can_flip_buses(0, 2);
  }

  // Init clock source (camera strobe) using PWM
  clock_source_init();
}

const harness_configuration dos_harness_config = {
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

const board board_dos = {
  .board_type = "Dos",
  .board_tick = dos_board_tick,
  .harness_config = &dos_harness_config,
  .has_hw_gmlan = false,
  .has_obd = true,
  .has_lin = false,
#ifdef ENABLE_SPI
  .has_spi = true,
#else
  .has_spi = false,
#endif
  .has_canfd = false,
  .has_rtc_battery = true,
  .fan_max_rpm = 6500U,
  .avdd_mV = 3300U,
  .fan_stall_recovery = true,
  .fan_enable_cooldown_time = 3U,
  .init = dos_init,
  .init_bootloader = unused_init_bootloader,
  .enable_can_transceiver = dos_enable_can_transceiver,
  .enable_can_transceivers = dos_enable_can_transceivers,
  .set_led = dos_set_led,
  .set_can_mode = dos_set_can_mode,
  .check_ignition = dos_check_ignition,
  .read_current = unused_read_current,
  .set_fan_enabled = dos_set_fan_enabled,
  .set_ir_power = dos_set_ir_power,
  .set_phone_power = unused_set_phone_power,
  .set_siren = dos_set_siren,
  .read_som_gpio = dos_read_som_gpio
};
