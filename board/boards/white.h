// /////////// //
// White Panda //
// /////////// //

void white_enable_can_transciever(uint8_t transciever, bool enabled) {
  switch (transciever){
    case 1U:
      set_gpio_output(GPIOC, 1, !enabled);
      break;
    case 2U:
      set_gpio_output(GPIOC, 13, !enabled);
      break;
    case 3U:
      set_gpio_output(GPIOA, 0, !enabled);
      break;
    default:
      puts("Invalid CAN transciever ("); puth(transciever); puts("): enabling failed\n");
      break;
  }
}

void white_enable_can_transcievers(bool enabled) {
  uint8_t t1 = enabled ? 1U : 2U;  // leave transciever 1 enabled to detect CAN ignition
  for(uint8_t i=t1; i<=3U; i++) {
    white_enable_can_transciever(i, enabled);
  }
}

void white_set_led(uint8_t color, bool enabled) {
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

void white_set_usb_power_mode(uint8_t mode){
  bool valid_mode = true;
  switch (mode) {
    case USB_POWER_CLIENT:
      // B2,A13: set client mode
      set_gpio_output(GPIOB, 2, 0);
      set_gpio_output(GPIOA, 13, 1);
      break;
    case USB_POWER_CDP:
      // B2,A13: set CDP mode
      set_gpio_output(GPIOB, 2, 1);
      set_gpio_output(GPIOA, 13, 1);
      break;
    case USB_POWER_DCP:
      // B2,A13: set DCP mode on the charger (breaks USB!)
      set_gpio_output(GPIOB, 2, 0);
      set_gpio_output(GPIOA, 13, 0);
      break;
    default:
      valid_mode = false;
      puts("Invalid usb power mode\n");
      break;
  }

  if (valid_mode) {
    usb_power_mode = mode;
  }
}

void white_set_esp_gps_mode(uint8_t mode) {
  switch (mode) {
    case ESP_GPS_DISABLED:
      // ESP OFF
      set_gpio_output(GPIOC, 14, 0);
      set_gpio_output(GPIOC, 5, 0);
      break;
#ifndef EON
    case ESP_GPS_ENABLED:
      // ESP ON
      set_gpio_output(GPIOC, 14, 1);
      set_gpio_output(GPIOC, 5, 1);
      break;
#endif
    case ESP_GPS_BOOTMODE:
      set_gpio_output(GPIOC, 14, 1);
      set_gpio_output(GPIOC, 5, 0);
      break;
    default:
      puts("Invalid ESP/GPS mode\n");
      break;
  }
}

void white_set_can_mode(uint8_t mode){
  switch (mode) {
    case CAN_MODE_NORMAL:
      // B12,B13: disable GMLAN mode
      set_gpio_mode(GPIOB, 12, MODE_INPUT);
      set_gpio_mode(GPIOB, 13, MODE_INPUT);

      // B3,B4: disable GMLAN mode
      set_gpio_mode(GPIOB, 3, MODE_INPUT);
      set_gpio_mode(GPIOB, 4, MODE_INPUT);

      // B5,B6: normal CAN2 mode
      set_gpio_alternate(GPIOB, 5, GPIO_AF9_CAN2);
      set_gpio_alternate(GPIOB, 6, GPIO_AF9_CAN2);

      // A8,A15: normal CAN3 mode
      set_gpio_alternate(GPIOA, 8, GPIO_AF11_CAN3);
      set_gpio_alternate(GPIOA, 15, GPIO_AF11_CAN3);
      break;
    case CAN_MODE_GMLAN_CAN2:
      // B5,B6: disable CAN2 mode
      set_gpio_mode(GPIOB, 5, MODE_INPUT);
      set_gpio_mode(GPIOB, 6, MODE_INPUT);

      // B3,B4: disable GMLAN mode
      set_gpio_mode(GPIOB, 3, MODE_INPUT);
      set_gpio_mode(GPIOB, 4, MODE_INPUT);

      // B12,B13: GMLAN mode
      set_gpio_alternate(GPIOB, 12, GPIO_AF9_CAN2);
      set_gpio_alternate(GPIOB, 13, GPIO_AF9_CAN2);

      // A8,A15: normal CAN3 mode
      set_gpio_alternate(GPIOA, 8, GPIO_AF11_CAN3);
      set_gpio_alternate(GPIOA, 15, GPIO_AF11_CAN3);
      break;
    case CAN_MODE_GMLAN_CAN3:
      // A8,A15: disable CAN3 mode
      set_gpio_mode(GPIOA, 8, MODE_INPUT);
      set_gpio_mode(GPIOA, 15, MODE_INPUT);

      // B12,B13: disable GMLAN mode
      set_gpio_mode(GPIOB, 12, MODE_INPUT);
      set_gpio_mode(GPIOB, 13, MODE_INPUT);

      // B3,B4: GMLAN mode
      set_gpio_alternate(GPIOB, 3, GPIO_AF11_CAN3);
      set_gpio_alternate(GPIOB, 4, GPIO_AF11_CAN3);

      // B5,B6: normal CAN2 mode
      set_gpio_alternate(GPIOB, 5, GPIO_AF9_CAN2);
      set_gpio_alternate(GPIOB, 6, GPIO_AF9_CAN2);
      break;
    default:
      puts("Tried to set unsupported CAN mode: "); puth(mode); puts("\n");
      break;
  }
}

uint32_t white_read_current(void){
  return adc_get(ADCCHAN_CURRENT);
}

uint32_t marker = 0;
void white_usb_power_mode_tick(uint32_t uptime){

  // on EON or BOOTSTUB, no state machine
#if !defined(BOOTSTUB) && !defined(EON)
  #define CURRENT_THRESHOLD 0xF00U
  #define CLICKS 5U // 5 seconds to switch modes

  uint32_t current = white_read_current();

  // ~0x9a = 500 ma
  // puth(current); puts("\n");

  switch (usb_power_mode) {
    case USB_POWER_CLIENT:
      if ((uptime - marker) >= CLICKS) {
        if (!is_enumerated) {
          puts("USBP: didn't enumerate, switching to CDP mode\n");
          // switch to CDP
          white_set_usb_power_mode(USB_POWER_CDP);
          marker = uptime;
        }
      }
      // keep resetting the timer if it's enumerated
      if (is_enumerated) {
        marker = uptime;
      }
      break;
    case USB_POWER_CDP:
      // been CLICKS clicks since we switched to CDP
      if ((uptime - marker) >= CLICKS) {
        // measure current draw, if positive and no enumeration, switch to DCP
        if (!is_enumerated && (current < CURRENT_THRESHOLD)) {
          puts("USBP: no enumeration with current draw, switching to DCP mode\n");
          white_set_usb_power_mode(USB_POWER_DCP);
          marker = uptime;
        }
      }
      // keep resetting the timer if there's no current draw in CDP
      if (current >= CURRENT_THRESHOLD) {
        marker = uptime;
      }
      break;
    case USB_POWER_DCP:
      // been at least CLICKS clicks since we switched to DCP
      if ((uptime - marker) >= CLICKS) {
        // if no current draw, switch back to CDP
        if (current >= CURRENT_THRESHOLD) {
          puts("USBP: no current draw, switching back to CDP mode\n");
          white_set_usb_power_mode(USB_POWER_CDP);
          marker = uptime;
        }
      }
      // keep resetting the timer if there's current draw in DCP
      if (current < CURRENT_THRESHOLD) {
        marker = uptime;
      }
      break;
    default:
      puts("USB power mode invalid\n");  // set_usb_power_mode prevents assigning invalid values
      break;
  }
#else
  UNUSED(uptime);
#endif
}

void white_set_ir_power(uint8_t percentage){
  UNUSED(percentage);
}

void white_set_fan_power(uint8_t percentage){
  UNUSED(percentage);
}

bool white_check_ignition(void){
  // ignition is on PA1
  return !get_gpio_input(GPIOA, 1);
}

void white_set_phone_power(bool enabled){
  UNUSED(enabled);
}

void white_grey_common_init(void) {
  common_init_gpio();

  // C3: current sense
  set_gpio_mode(GPIOC, 3, MODE_ANALOG);

  // A1: started_alt
  set_gpio_pullup(GPIOA, 1, PULL_UP);

  // A2, A3: USART 2 for debugging
  set_gpio_alternate(GPIOA, 2, GPIO_AF7_USART2);
  set_gpio_alternate(GPIOA, 3, GPIO_AF7_USART2);

  // A4, A5, A6, A7: SPI
  set_gpio_alternate(GPIOA, 4, GPIO_AF5_SPI1);
  set_gpio_alternate(GPIOA, 5, GPIO_AF5_SPI1);
  set_gpio_alternate(GPIOA, 6, GPIO_AF5_SPI1);
  set_gpio_alternate(GPIOA, 7, GPIO_AF5_SPI1);

  // B12: GMLAN, ignition sense, pull up
  set_gpio_pullup(GPIOB, 12, PULL_UP);

  /* GMLAN mode pins:
      M0(B15)  M1(B14)  mode
      =======================
      0        0        sleep
      1        0        100kbit
      0        1        high voltage wakeup
      1        1        33kbit (normal)
  */
  set_gpio_output(GPIOB, 14, 1);
  set_gpio_output(GPIOB, 15, 1);

  // B7: K-line enable
  set_gpio_output(GPIOB, 7, 1);

  // C12, D2: Setup K-line (UART5)
  set_gpio_alternate(GPIOC, 12, GPIO_AF8_UART5);
  set_gpio_alternate(GPIOD, 2, GPIO_AF8_UART5);
  set_gpio_pullup(GPIOD, 2, PULL_UP);

  // L-line enable
  set_gpio_output(GPIOA, 14, 1);

  // C10, C11: L-Line setup (USART3)
  set_gpio_alternate(GPIOC, 10, GPIO_AF7_USART3);
  set_gpio_alternate(GPIOC, 11, GPIO_AF7_USART3);
  set_gpio_pullup(GPIOC, 11, PULL_UP);

  // Enable CAN transcievers
  white_enable_can_transcievers(true);

  // Disable LEDs
  white_set_led(LED_RED, false);
  white_set_led(LED_GREEN, false);
  white_set_led(LED_BLUE, false);

  // Set normal CAN mode
  white_set_can_mode(CAN_MODE_NORMAL);

  // Init usb power mode
  uint32_t voltage = adc_get_voltage();
  // Init in CDP mode only if panda is powered by 12V.
  // Otherwise a PC would not be able to flash a standalone panda with EON build
  if (voltage > 8000U) {  // 8V threshold
    white_set_usb_power_mode(USB_POWER_CDP);
  } else {
    white_set_usb_power_mode(USB_POWER_CLIENT);
  }
}

void white_init(void) {
  white_grey_common_init();

  // Set default state of ESP
  #ifdef EON
    current_board->set_esp_gps_mode(ESP_GPS_DISABLED);
  #else
    current_board->set_esp_gps_mode(ESP_GPS_ENABLED);
  #endif
}

const harness_configuration white_harness_config = {
  .has_harness = false
};

const board board_white = {
  .board_type = "White",
  .harness_config = &white_harness_config,
  .init = white_init,
  .enable_can_transciever = white_enable_can_transciever,
  .enable_can_transcievers = white_enable_can_transcievers,
  .set_led = white_set_led,
  .set_usb_power_mode = white_set_usb_power_mode,
  .set_esp_gps_mode = white_set_esp_gps_mode,
  .set_can_mode = white_set_can_mode,
  .usb_power_mode_tick = white_usb_power_mode_tick,
  .check_ignition = white_check_ignition,
  .read_current = white_read_current,
  .set_fan_power = white_set_fan_power,
  .set_ir_power = white_set_ir_power,
  .set_phone_power = white_set_phone_power
};
