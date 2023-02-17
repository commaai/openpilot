// ///// //
// Pedal //
// ///// //

void pedal_enable_can_transceiver(uint8_t transceiver, bool enabled) {
  switch (transceiver){
    case 1:
      set_gpio_output(GPIOB, 3, !enabled);
      break;
    default:
      print("Invalid CAN transceiver ("); puth(transceiver); print("): enabling failed\n");
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

void pedal_set_gps_mode(uint8_t mode) {
  UNUSED(mode);
  print("Trying to set ESP/GPS mode on pedal. This is not supported.\n");
}

void pedal_set_can_mode(uint8_t mode){
  switch (mode) {
    case CAN_MODE_NORMAL:
      break;
    default:
      print("Tried to set unsupported CAN mode: "); puth(mode); print("\n");
      break;
  }
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
  .board_tick = unused_board_tick,
  .harness_config = &pedal_harness_config,
  .has_gps = false,
  .has_hw_gmlan = false,
  .has_obd = false,
  .has_lin = false,
  .has_spi = false,
  .has_canfd = false,
  .has_rtc_battery = false,
  .fan_max_rpm = 0U,
  .init = pedal_init,
  .enable_can_transceiver = pedal_enable_can_transceiver,
  .enable_can_transceivers = pedal_enable_can_transceivers,
  .set_led = pedal_set_led,
  .set_gps_mode = pedal_set_gps_mode,
  .set_can_mode = pedal_set_can_mode,
  .check_ignition = pedal_check_ignition,
  .read_current = unused_read_current,
  .set_fan_enabled = unused_set_fan_enabled,
  .set_ir_power = unused_set_ir_power,
  .set_phone_power = unused_set_phone_power,
  .set_siren = unused_set_siren,
  .read_som_gpio = unused_read_som_gpio
};
