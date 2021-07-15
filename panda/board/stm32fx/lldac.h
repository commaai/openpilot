void dac_init(void) {
  // No buffers required since we have an opamp
  register_set(&(DAC->DHR12R1), 0U, 0xFFFU);
  register_set(&(DAC->DHR12R2), 0U, 0xFFFU);
  register_set(&(DAC->CR), DAC_CR_EN1 | DAC_CR_EN2, 0x3FFF3FFFU);
}

void dac_set(int channel, uint32_t value) {
  if (channel == 0) {
    register_set(&(DAC->DHR12R1), value, 0xFFFU);
  } else if (channel == 1) {
    register_set(&(DAC->DHR12R2), value, 0xFFFU);
  } else {
    puts("Failed to set DAC: invalid channel value: 0x"); puth(value); puts("\n");
  }
}
