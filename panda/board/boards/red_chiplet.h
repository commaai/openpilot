// ///////////////////// //
// Red Panda chiplet + Harness //
// ///////////////////// //

// Most hardware functionality is similar to red panda

void red_chiplet_enable_can_transceiver(uint8_t transceiver, bool enabled) {
  switch (transceiver) {
    case 1U:
      set_gpio_output(GPIOG, 11, !enabled);
      break;
    case 2U:
      set_gpio_output(GPIOB, 10, !enabled);
      break;
    case 3U:
      set_gpio_output(GPIOD, 7, !enabled);
      break;
    case 4U:
      set_gpio_output(GPIOB, 11, !enabled);
      break;
    default:
      break;
  }
}

void red_chiplet_enable_can_transceivers(bool enabled) {
  uint8_t main_bus = (harness.status == HARNESS_STATUS_FLIPPED) ? 3U : 1U;
  for (uint8_t i=1U; i<=4U; i++) {
    // Leave main CAN always on for CAN-based ignition detection
    if (i == main_bus) {
      red_chiplet_enable_can_transceiver(i, true);
    } else {
      red_chiplet_enable_can_transceiver(i, enabled);
    }
  }
}

void red_chiplet_set_can_mode(uint8_t mode) {
  red_chiplet_enable_can_transceiver(2U, false);
  red_chiplet_enable_can_transceiver(4U, false);
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
        red_chiplet_enable_can_transceiver(2U, true);
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
        red_chiplet_enable_can_transceiver(4U, true);
      }
      break;
    default:
      break;
  }
}

void red_chiplet_set_fan_or_usb_load_switch(bool enabled) {
  set_gpio_output(GPIOD, 3, enabled);
}

void red_chiplet_init(void) {
  common_init_gpio();

  // A8, A3: OBD_SBU1_RELAY, OBD_SBU2_RELAY
  set_gpio_output_type(GPIOA, 8, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_pullup(GPIOA, 8, PULL_NONE);
  set_gpio_output(GPIOA, 8, 1);
  set_gpio_mode(GPIOA, 8, MODE_OUTPUT);

  set_gpio_output_type(GPIOA, 3, OUTPUT_TYPE_OPEN_DRAIN);
  set_gpio_pullup(GPIOA, 3, PULL_NONE);
  set_gpio_output(GPIOA, 3, 1);
  set_gpio_mode(GPIOA, 3, MODE_OUTPUT);

  // G11,B10,D7,B11: transceiver enable
  set_gpio_pullup(GPIOG, 11, PULL_NONE);
  set_gpio_mode(GPIOG, 11, MODE_OUTPUT);

  set_gpio_pullup(GPIOB, 10, PULL_NONE);
  set_gpio_mode(GPIOB, 10, MODE_OUTPUT);

  set_gpio_pullup(GPIOD, 7, PULL_NONE);
  set_gpio_mode(GPIOD, 7, MODE_OUTPUT);

  set_gpio_pullup(GPIOB, 11, PULL_NONE);
  set_gpio_mode(GPIOB, 11, MODE_OUTPUT);

  // D3: usb load switch
  set_gpio_pullup(GPIOD, 3, PULL_NONE);
  set_gpio_mode(GPIOD, 3, MODE_OUTPUT);

  // B0: 5VOUT_S
  set_gpio_pullup(GPIOB, 0, PULL_NONE);
  set_gpio_mode(GPIOB, 0, MODE_ANALOG);

  // Initialize harness
  harness_init();

  // Initialize RTC
  rtc_init();

  // Enable CAN transceivers
  red_chiplet_enable_can_transceivers(true);

  // Disable LEDs
  red_set_led(LED_RED, false);
  red_set_led(LED_GREEN, false);
  red_set_led(LED_BLUE, false);

  // Set normal CAN mode
  red_chiplet_set_can_mode(CAN_MODE_NORMAL);

  // flip CAN0 and CAN2 if we are flipped
  if (harness.status == HARNESS_STATUS_FLIPPED) {
    can_flip_buses(0, 2);
  }
}

const harness_configuration red_chiplet_harness_config = {
  .has_harness = true,
  .GPIO_SBU1 = GPIOC,
  .GPIO_SBU2 = GPIOA,
  .GPIO_relay_SBU1 = GPIOA,
  .GPIO_relay_SBU2 = GPIOA,
  .pin_SBU1 = 4,
  .pin_SBU2 = 1,
  .pin_relay_SBU1 = 8,
  .pin_relay_SBU2 = 3,
  .adc_channel_SBU1 = 4, // ADC12_INP4
  .adc_channel_SBU2 = 17 // ADC1_INP17
};
