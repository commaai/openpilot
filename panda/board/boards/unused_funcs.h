void unused_init_bootloader(void) {
}

void unused_set_ir_power(uint8_t percentage) {
  UNUSED(percentage);
}

void unused_set_fan_enabled(bool enabled) {
  UNUSED(enabled);
}

void unused_set_phone_power(bool enabled) {
  UNUSED(enabled);
}

void unused_set_siren(bool enabled) {
  UNUSED(enabled);
}

uint32_t unused_read_current(void) {
  return 0U;
}

bool unused_board_tick(bool ignition, bool usb_enum, bool heartbeat_seen, bool harness_inserted) {
  UNUSED(ignition);
  UNUSED(usb_enum);
  UNUSED(heartbeat_seen);
  UNUSED(harness_inserted);
  return false;
}

bool unused_read_som_gpio(void) {
  return false;
}