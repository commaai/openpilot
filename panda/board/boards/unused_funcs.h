void unused_init_bootloader(void) {
}

void unused_set_ir_power(uint8_t percentage) {
  UNUSED(percentage);
}

void unused_set_fan_enabled(bool enabled) {
  UNUSED(enabled);
}

void unused_set_siren(bool enabled) {
  UNUSED(enabled);
}

uint32_t unused_read_current(void) {
  return 0U;
}

void unused_set_bootkick(BootState state) {
  UNUSED(state);
}

bool unused_read_som_gpio(void) {
  return false;
}