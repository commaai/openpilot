// ///////////////////// //
// Red Panda + Harness //
// ///////////////////// //

void red_enable_can_transceiver(uint8_t transceiver, bool enabled) {
  switch (transceiver) {
    case 1U:
      set_gpio_output(GPIOG, 11, !enabled);
      break;
    case 2U:
      set_gpio_output(GPIOB, 3, !enabled);
      break;
    case 3U:
      set_gpio_output(GPIOD, 7, !enabled);
      break;
    case 4U:
      set_gpio_output(GPIOB, 4, !enabled);
      break;
    default:
      break;
  }
}

void red_enable_can_transceivers(bool enabled) {
  uint8_t main_bus = (harness.status == HARNESS_STATUS_FLIPPED) ? 3U : 1U;
  for (uint8_t i=1U; i<=4U; i++) {
    // Leave main CAN always on for CAN-based ignition detection
    if (i == main_bus) {
      red_enable_can_transceiver(i, true);
    } else {
      red_enable_can_transceiver(i, enabled);
    }
  }
}

void red_set_led(uint8_t color, bool enabled) {
  switch (color) {
    case LED_RED:
      set_gpio_output(GPIOE, 4, !enabled);
      break;
     case LED_GREEN:
      set_gpio_output(GPIOE, 3, !enabled);
      break;
    case LED_BLUE:
      set_gpio_output(GPIOE, 2, !enabled);
      break;
    default:
      break;
  }
}

void red_set_can_mode(uint8_t mode) {
  red_enable_can_transceiver(2U, false);
  red_enable_can_transceiver(4U, false);
  switch (mode) {
    case CAN_MODE_NORMAL:
    case CAN_MODE_OBD_CAN2:
      if ((bool)(mode == CAN_MODE_NORMAL) != (bool)(harness.status == HARNESS_STATUS_FLIPPED)) {
        // B12,B13: disable normal mode
        set_gpio_pullup(GPIOB, 12, PULL_NONE);
        set_gpio_mode(GPIOB, 12, MODE_ANALOG);

        set_gpio_pullup(GPIOB, 13, PULL_NONE);
        set_gpio_mode(GPIOB, 13, MODE_ANALOG);

        // B5,B6: FDCAN2 mode
        set_gpio_pullup(GPIOB, 5, PULL_NONE);
        set_gpio_alternate(GPIOB, 5, GPIO_AF9_FDCAN2);

        set_gpio_pullup(GPIOB, 6, PULL_NONE);
        set_gpio_alternate(GPIOB, 6, GPIO_AF9_FDCAN2);
        red_enable_can_transceiver(2U, true);
      } else {
        // B5,B6: disable normal mode
        set_gpio_pullup(GPIOB, 5, PULL_NONE);
        set_gpio_mode(GPIOB, 5, MODE_ANALOG);

        set_gpio_pullup(GPIOB, 6, PULL_NONE);
        set_gpio_mode(GPIOB, 6, MODE_ANALOG);
        // B12,B13: FDCAN2 mode
        set_gpio_pullup(GPIOB, 12, PULL_NONE);
        set_gpio_alternate(GPIOB, 12, GPIO_AF9_FDCAN2);

        set_gpio_pullup(GPIOB, 13, PULL_NONE);
        set_gpio_alternate(GPIOB, 13, GPIO_AF9_FDCAN2);
        red_enable_can_transceiver(4U, true);
      }
      break;
    default:
      break;
  }
}

bool red_check_ignition(void) {
  // ignition is checked through harness
  return harness_check_ignition();
}

void red_init(void) {
  common_init_gpio();

  //C10,C11 : OBD_SBU1_RELAY, OBD_SBU2_RELAY
  set_gpio_output_type(GPIOC, 10, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_pullup(GPIOC, 10, PULL_NONE);
  set_gpio_mode(GPIOC, 10, MODE_OUTPUT);
  set_gpio_output(GPIOC, 10, 1);

  set_gpio_output_type(GPIOC, 11, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_pullup(GPIOC, 11, PULL_NONE);
  set_gpio_mode(GPIOC, 11, MODE_OUTPUT);
  set_gpio_output(GPIOC, 11, 1);

  // G11,B3,D7,B4: transceiver enable
  set_gpio_pullup(GPIOG, 11, PULL_NONE);
  set_gpio_mode(GPIOG, 11, MODE_OUTPUT);

  set_gpio_pullup(GPIOB, 3, PULL_NONE);
  set_gpio_mode(GPIOB, 3, MODE_OUTPUT);

  set_gpio_pullup(GPIOD, 7, PULL_NONE);
  set_gpio_mode(GPIOD, 7, MODE_OUTPUT);

  set_gpio_pullup(GPIOB, 4, PULL_NONE);
  set_gpio_mode(GPIOB, 4, MODE_OUTPUT);

  //B1: 5VOUT_S
  set_gpio_pullup(GPIOB, 1, PULL_NONE);
  set_gpio_mode(GPIOB, 1, MODE_ANALOG);

  // B14: usb load switch, enabled by pull resistor on board, obsolete for red panda
  set_gpio_output_type(GPIOB, 14, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_pullup(GPIOB, 14, PULL_UP);
  set_gpio_mode(GPIOB, 14, MODE_OUTPUT);
  set_gpio_output(GPIOB, 14, 1);

  // Initialize harness
  harness_init();

  // Initialize RTC
  rtc_init();

  // Enable CAN transceivers
  red_enable_can_transceivers(true);

  // Disable LEDs
  red_set_led(LED_RED, false);
  red_set_led(LED_GREEN, false);
  red_set_led(LED_BLUE, false);

  // Set normal CAN mode
  red_set_can_mode(CAN_MODE_NORMAL);

  // flip CAN0 and CAN2 if we are flipped
  if (harness.status == HARNESS_STATUS_FLIPPED) {
    can_flip_buses(0, 2);
  }
}

const harness_configuration red_harness_config = {
  .has_harness = true,
  .GPIO_SBU1 = GPIOC,
  .GPIO_SBU2 = GPIOA,
  .GPIO_relay_SBU1 = GPIOC,
  .GPIO_relay_SBU2 = GPIOC,
  .pin_SBU1 = 4,
  .pin_SBU2 = 1,
  .pin_relay_SBU1 = 10,
  .pin_relay_SBU2 = 11,
  .adc_channel_SBU1 = 4, //ADC12_INP4
  .adc_channel_SBU2 = 17 //ADC1_INP17
};

const board board_red = {
  .board_type = "Red",
  .board_tick = unused_board_tick,
  .harness_config = &red_harness_config,
  .has_hw_gmlan = false,
  .has_obd = true,
  .has_lin = false,
  .has_spi = false,
  .has_canfd = true,
  .has_rtc_battery = false,
  .fan_max_rpm = 0U,
  .avdd_mV = 3300U,
  .fan_stall_recovery = false,
  .fan_enable_cooldown_time = 0U,
  .init = red_init,
  .init_bootloader = unused_init_bootloader,
  .enable_can_transceiver = red_enable_can_transceiver,
  .enable_can_transceivers = red_enable_can_transceivers,
  .set_led = red_set_led,
  .set_can_mode = red_set_can_mode,
  .check_ignition = red_check_ignition,
  .read_current = unused_read_current,
  .set_fan_enabled = unused_set_fan_enabled,
  .set_ir_power = unused_set_ir_power,
  .set_phone_power = unused_set_phone_power,
  .set_siren = unused_set_siren,
  .read_som_gpio = unused_read_som_gpio
};
