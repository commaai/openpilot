// ///// //
// Pedal //
// ///// //

void pedal_enable_can_transciever(uint8_t transciever, bool enabled) {
  switch (transciever){
    case 1:
      set_gpio_output(GPIOB, 3, !enabled);
      break;
    default:
      puts("Invalid CAN transciever ("); puth(transciever); puts("): enabling failed\n");
      break;
  }
}

void pedal_enable_can_transcievers(bool enabled) {
  pedal_enable_can_transciever(1U, enabled);
}

void pedal_set_led(uint8_t color, bool enabled) {
  switch (color){
    case LED_RED:
      set_gpio_output(GPIOB, 10, !enabled);
      break;
     case LED_GREEN:
      set_gpio_output(GPIOB, 11, !enabled);
      break;
    default:
      break;
  }
}

void pedal_set_usb_power_mode(uint8_t mode){
  usb_power_mode = mode;
  puts("Trying to set USB power mode on pedal. This is not supported.\n");
}

void pedal_set_esp_gps_mode(uint8_t mode) {
  UNUSED(mode);
  puts("Trying to set ESP/GPS mode on pedal. This is not supported.\n");
}

void pedal_set_can_mode(uint8_t mode){
  switch (mode) {
    case CAN_MODE_NORMAL:
      break;
    default:
      puts("Tried to set unsupported CAN mode: "); puth(mode); puts("\n");
      break;
  }
}

void pedal_usb_power_mode_tick(uint64_t tcnt){
  UNUSED(tcnt);
  // Not applicable
}

bool pedal_check_ignition(void){
  // not supported on pedal
  return false;
}

void pedal_init(void) {
  common_init_gpio();

  // C0, C1: Throttle inputs
  set_gpio_mode(GPIOC, 0, MODE_ANALOG);
  set_gpio_mode(GPIOC, 1, MODE_ANALOG);
  // DAC outputs on A4 and A5
  //   apparently they don't need GPIO setup

  // Enable transciever
  pedal_enable_can_transcievers(true);

  // Disable LEDs
  pedal_set_led(LED_RED, false);
  pedal_set_led(LED_GREEN, false);
}

const harness_configuration pedal_harness_config = {
  .has_harness = false
};

const board board_pedal = {
  .board_type = "Pedal",
  .harness_config = &pedal_harness_config,
  .init = pedal_init,
  .enable_can_transciever = pedal_enable_can_transciever,
  .enable_can_transcievers = pedal_enable_can_transcievers,
  .set_led = pedal_set_led,
  .set_usb_power_mode = pedal_set_usb_power_mode,
  .set_esp_gps_mode = pedal_set_esp_gps_mode,
  .set_can_mode = pedal_set_can_mode,
  .usb_power_mode_tick = pedal_usb_power_mode_tick,
  .check_ignition = pedal_check_ignition,
};