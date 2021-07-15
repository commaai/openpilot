// ///////////// //
// Uno + Harness //
// ///////////// //
#define BOOTKICK_TIME 3U
uint8_t bootkick_timer = 0U;

void uno_enable_can_transceiver(uint8_t transceiver, bool enabled) {
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
      puts("Invalid CAN transceiver ("); puth(transceiver); puts("): enabling failed\n");
      break;
  }
}

void uno_enable_can_transceivers(bool enabled) {
  for(uint8_t i=1U; i<=4U; i++){
    // Leave main CAN always on for CAN-based ignition detection
    if((car_harness_status == HARNESS_STATUS_FLIPPED) ? (i == 3U) : (i == 1U)){
      uno_enable_can_transceiver(i, true);
    } else {
      uno_enable_can_transceiver(i, enabled);
    }
  }
}

void uno_set_led(uint8_t color, bool enabled) {
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

void uno_set_gps_load_switch(bool enabled) {
  set_gpio_output(GPIOC, 12, enabled);
}

void uno_set_bootkick(bool enabled){
  if(enabled){
    set_gpio_output(GPIOB, 14, false);
  } else {
    // We want the pin to be floating, not forced high!
    set_gpio_mode(GPIOB, 14, MODE_INPUT);
  }
}

void uno_bootkick(void) {
  bootkick_timer = BOOTKICK_TIME;
  uno_set_bootkick(true);
}

void uno_set_phone_power(bool enabled){
  set_gpio_output(GPIOB, 4, enabled);
}

void uno_set_usb_power_mode(uint8_t mode) {
  bool valid = false;
  switch (mode) {
    case USB_POWER_CLIENT:
      valid = true;
      break;
    case USB_POWER_CDP:
      uno_bootkick();
      valid = true;
      break;
    default:
      puts("Invalid USB power mode\n");
      break;
  }
  if (valid) {
    usb_power_mode = mode;
  }
}

void uno_set_gps_mode(uint8_t mode) {
  switch (mode) {
    case GPS_DISABLED:
      // GPS OFF
      set_gpio_output(GPIOB, 1, 0);
      set_gpio_output(GPIOC, 5, 0);
      uno_set_gps_load_switch(false);
      break;
    case GPS_ENABLED:
      // GPS ON
      set_gpio_output(GPIOB, 1, 1);
      set_gpio_output(GPIOC, 5, 1);
      uno_set_gps_load_switch(true);
      break;
    case GPS_BOOTMODE:
      set_gpio_output(GPIOB, 1, 1);
      set_gpio_output(GPIOC, 5, 0);
      uno_set_gps_load_switch(true);
      break;
    default:
      puts("Invalid ESP/GPS mode\n");
      break;
  }
}

void uno_set_can_mode(uint8_t mode){
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
      puts("Tried to set unsupported CAN mode: "); puth(mode); puts("\n");
      break;
  }
}

void uno_usb_power_mode_tick(uint32_t uptime){
  UNUSED(uptime);
  if(bootkick_timer != 0U){
    bootkick_timer--;
  } else {
    uno_set_bootkick(false);
  }
}

bool uno_check_ignition(void){
  // ignition is checked through harness
  return harness_check_ignition();
}

void uno_set_usb_switch(bool phone){
  set_gpio_output(GPIOB, 3, phone);
}

void uno_set_ir_power(uint8_t percentage){
  pwm_set(TIM4, 2, percentage);
}

void uno_set_fan_power(uint8_t percentage){
  // Enable fan power only if percentage is non-zero.
  set_gpio_output(GPIOA, 1, (percentage != 0U));
  fan_set_power(percentage);
}

uint32_t uno_read_current(void){
  // No current sense on Uno
  return 0U;
}

void uno_set_clock_source_mode(uint8_t mode){
  UNUSED(mode);
}

void uno_set_siren(bool enabled){
  UNUSED(enabled);
}

void uno_init(void) {
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

  // C8: FAN PWM aka TIM3_CH3
  set_gpio_alternate(GPIOC, 8, GPIO_AF2_TIM3);

  // Turn on GPS load switch.
  uno_set_gps_load_switch(true);

  // Turn on phone regulator
  uno_set_phone_power(true);

  // Initialize IR PWM and set to 0%
  set_gpio_alternate(GPIOB, 7, GPIO_AF2_TIM4);
  pwm_init(TIM4, 2);
  uno_set_ir_power(0U);

  // Initialize fan and set to 0%
  fan_init();
  uno_set_fan_power(0U);

  // Initialize harness
  harness_init();

  // Initialize RTC
  rtc_init();

  // Enable CAN transceivers
  uno_enable_can_transceivers(true);

  // Disable LEDs
  uno_set_led(LED_RED, false);
  uno_set_led(LED_GREEN, false);
  uno_set_led(LED_BLUE, false);

  // Set normal CAN mode
  uno_set_can_mode(CAN_MODE_NORMAL);

  // flip CAN0 and CAN2 if we are flipped
  if (car_harness_status == HARNESS_STATUS_FLIPPED) {
    can_flip_buses(0, 2);
  }

  // Switch to phone usb mode if harness connection is powered by less than 7V
  if(adc_get_voltage() < 7000U){
    uno_set_usb_switch(true);
  } else {
    uno_set_usb_switch(false);
  }

  // Bootkick phone
  uno_bootkick();
}

const harness_configuration uno_harness_config = {
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

const board board_uno = {
  .board_type = "Uno",
  .harness_config = &uno_harness_config,
  .has_gps = true,
  .has_hw_gmlan = false,
  .has_obd = true,
  .has_lin = false,
  .has_rtc = true,
  .init = uno_init,
  .enable_can_transceiver = uno_enable_can_transceiver,
  .enable_can_transceivers = uno_enable_can_transceivers,
  .set_led = uno_set_led,
  .set_usb_power_mode = uno_set_usb_power_mode,
  .set_gps_mode = uno_set_gps_mode,
  .set_can_mode = uno_set_can_mode,
  .usb_power_mode_tick = uno_usb_power_mode_tick,
  .check_ignition = uno_check_ignition,
  .read_current = uno_read_current,
  .set_fan_power = uno_set_fan_power,
  .set_ir_power = uno_set_ir_power,
  .set_phone_power = uno_set_phone_power,
  .set_clock_source_mode = uno_set_clock_source_mode,
  .set_siren = uno_set_siren
};
