// ///////////// //
// Dos + Harness //
// ///////////// //

void dos_enable_can_transciever(uint8_t transciever, bool enabled) {
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
    case 4U:
      set_gpio_output(GPIOB, 10, !enabled);
      break;
    default:
      puts("Invalid CAN transciever ("); puth(transciever); puts("): enabling failed\n");
      break;
  }
}

void dos_enable_can_transcievers(bool enabled) {
  for(uint8_t i=1U; i<=4U; i++){
    // Leave main CAN always on for CAN-based ignition detection
    if((car_harness_status == HARNESS_STATUS_FLIPPED) ? (i == 3U) : (i == 1U)){
      uno_enable_can_transciever(i, true);
    } else {
      uno_enable_can_transciever(i, enabled);
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

void dos_set_gps_load_switch(bool enabled) {
  UNUSED(enabled);
}

void dos_set_bootkick(bool enabled){
  UNUSED(enabled);
}

void dos_bootkick(void) {}

void dos_set_phone_power(bool enabled){
  UNUSED(enabled);
}

void dos_set_usb_power_mode(uint8_t mode) {
  UNUSED(mode);
}

void dos_set_esp_gps_mode(uint8_t mode) {
  UNUSED(mode);
}

void dos_set_can_mode(uint8_t mode){
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

void dos_usb_power_mode_tick(uint32_t uptime){
  UNUSED(uptime);
  if(bootkick_timer != 0U){
    bootkick_timer--;
  } else {
    dos_set_bootkick(false);
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

void dos_set_fan_power(uint8_t percentage){
  // Enable fan power only if percentage is non-zero.
  set_gpio_output(GPIOA, 1, (percentage != 0U));
  fan_set_power(percentage);
}

uint32_t dos_read_current(void){
  // No current sense on Dos
  return 0U;
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

  // C8: FAN PWM aka TIM3_CH3
  set_gpio_alternate(GPIOC, 8, GPIO_AF2_TIM3);

  // Initialize IR PWM and set to 0%
  set_gpio_alternate(GPIOB, 7, GPIO_AF2_TIM4);
  pwm_init(TIM4, 2);
  dos_set_ir_power(0U);

  // Initialize fan and set to 0%
  fan_init();
  dos_set_fan_power(0U);

  // Initialize harness
  harness_init();

  // Initialize RTC
  rtc_init();

  // Enable CAN transcievers
  dos_enable_can_transcievers(true);

  // Disable LEDs
  dos_set_led(LED_RED, false);
  dos_set_led(LED_GREEN, false);
  dos_set_led(LED_BLUE, false);

  // Set normal CAN mode
  dos_set_can_mode(CAN_MODE_NORMAL);

  // flip CAN0 and CAN2 if we are flipped
  if (car_harness_status == HARNESS_STATUS_FLIPPED) {
    can_flip_buses(0, 2);
  }

  // init multiplexer
  can_set_obd(car_harness_status, false);
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
  .harness_config = &dos_harness_config,
  .init = dos_init,
  .enable_can_transciever = dos_enable_can_transciever,
  .enable_can_transcievers = dos_enable_can_transcievers,
  .set_led = dos_set_led,
  .set_usb_power_mode = dos_set_usb_power_mode,
  .set_esp_gps_mode = dos_set_esp_gps_mode,
  .set_can_mode = dos_set_can_mode,
  .usb_power_mode_tick = dos_usb_power_mode_tick,
  .check_ignition = dos_check_ignition,
  .read_current = dos_read_current,
  .set_fan_power = dos_set_fan_power,
  .set_ir_power = dos_set_ir_power,
  .set_phone_power = dos_set_phone_power
};
