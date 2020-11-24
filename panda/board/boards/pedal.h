// ///// //
// Pedal //
// ///// //

void pedal_enable_can_transceiver(uint8_t transceiver, bool enabled) {
  switch (transceiver){
    case 1:
      set_gpio_output(GPIOB, 3, !enabled);
      break;
    default:
      puts("Invalid CAN transceiver ("); puth(transceiver); puts("): enabling failed\n");
      break;
  }
}

void pedal_enable_can_transceivers(bool enabled) {
  pedal_enable_can_transceiver(1U, enabled);
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

void pedal_set_gps_mode(uint8_t mode) {
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

void pedal_usb_power_mode_tick(uint32_t uptime){
  UNUSED(uptime);
  // Not applicable
}

bool pedal_check_ignition(void){
  // not supported on pedal
  return false;
}

uint32_t pedal_read_current(void){
  // No current sense on pedal
  return 0U;
}

void pedal_set_ir_power(uint8_t percentage){
  UNUSED(percentage);
}

void pedal_set_fan_power(uint8_t percentage){
  UNUSED(percentage);
}

void pedal_set_phone_power(bool enabled){
  UNUSED(enabled);
}

void pedal_set_clock_source_mode(uint8_t mode){
  UNUSED(mode);
}

void pedal_set_siren(bool enabled){
  UNUSED(enabled);
}

void pedal_init(void) {
  common_init_gpio();

  // C0, C1: Throttle inputs
  set_gpio_mode(GPIOC, 0, MODE_ANALOG);
  set_gpio_mode(GPIOC, 1, MODE_ANALOG);
  // DAC outputs on A4 and A5
  //   apparently they don't need GPIO setup

  // Enable transceiver
  pedal_enable_can_transceivers(true);

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
  .enable_can_transceiver = pedal_enable_can_transceiver,
  .enable_can_transceivers = pedal_enable_can_transceivers,
  .set_led = pedal_set_led,
  .set_usb_power_mode = pedal_set_usb_power_mode,
  .set_gps_mode = pedal_set_gps_mode,
  .set_can_mode = pedal_set_can_mode,
  .usb_power_mode_tick = pedal_usb_power_mode_tick,
  .check_ignition = pedal_check_ignition,
  .read_current = pedal_read_current,
  .set_fan_power = pedal_set_fan_power,
  .set_ir_power = pedal_set_ir_power,
  .set_phone_power = pedal_set_phone_power,
  .set_clock_source_mode = pedal_set_clock_source_mode,
  .set_siren = pedal_set_siren
};
