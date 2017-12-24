void dac_init() {
  // no buffers required since we have an opamp
  //DAC->CR = DAC_CR_EN1 | DAC_CR_BOFF1 | DAC_CR_EN2 | DAC_CR_BOFF2;
  DAC->DHR12R1 = 0;
  DAC->DHR12R2 = 0;
  DAC->CR = DAC_CR_EN1 | DAC_CR_EN2;
}

void dac_set(int channel, uint32_t value) {
  if (channel == 0) {
    DAC->DHR12R1 = value;
  } else if (channel == 1) {
    DAC->DHR12R2 = value;
  }
}

