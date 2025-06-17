
void register_set(volatile uint32_t *addr, uint32_t val, uint32_t mask);

void adc_init(void) {
  register_set(&(ADC->CCR), ADC_CCR_TSVREFE | ADC_CCR_VBATE, 0xC30000U);
  register_set(&(ADC1->CR2), ADC_CR2_ADON, 0xFF7F0F03U);
  register_set(&(ADC1->SMPR1), ADC_SMPR1_SMP12 | ADC_SMPR1_SMP13, 0x7FFFFFFU);
}

uint16_t adc_get_raw(uint8_t channel) {
  // Select channel
  register_set(&(ADC1->JSQR), ((uint32_t) channel << 15U), 0x3FFFFFU);

  // Start conversion
  ADC1->SR &= ~(ADC_SR_JEOC);
  ADC1->CR2 |= ADC_CR2_JSWSTART;
  while (!(ADC1->SR & ADC_SR_JEOC));

  return ADC1->JDR1;
}

uint16_t adc_get_mV(uint8_t channel) {
  return (adc_get_raw(channel) * current_board->avdd_mV) / 4095U;
}
