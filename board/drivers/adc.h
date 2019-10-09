// ACCEL1 = ADC10
// ACCEL2 = ADC11
// VOLT_S = ADC12
// CURR_S = ADC13

#define ADCCHAN_ACCEL0 10
#define ADCCHAN_ACCEL1 11
#define ADCCHAN_VOLTAGE 12
#define ADCCHAN_CURRENT 13

void adc_init(void) {
  // global setup
  ADC->CCR = ADC_CCR_TSVREFE | ADC_CCR_VBATE;
  //ADC1->CR2 = ADC_CR2_ADON | ADC_CR2_EOCS | ADC_CR2_DDS;
  ADC1->CR2 = ADC_CR2_ADON;

  // long
  //ADC1->SMPR1 = ADC_SMPR1_SMP10 | ADC_SMPR1_SMP11 | ADC_SMPR1_SMP12 | ADC_SMPR1_SMP13;
  ADC1->SMPR1 = ADC_SMPR1_SMP12 | ADC_SMPR1_SMP13;
}

uint32_t adc_get(unsigned int channel) {
  // includes length
  //ADC1->SQR1 = 0;

  // select channel
  ADC1->JSQR = channel << 15;

  //ADC1->CR1 = ADC_CR1_DISCNUM_0;
  //ADC1->CR1 = ADC_CR1_EOCIE;

  ADC1->SR &= ~(ADC_SR_JEOC);
  ADC1->CR2 |= ADC_CR2_JSWSTART;
  while (!(ADC1->SR & ADC_SR_JEOC));

  return ADC1->JDR1;
}

uint32_t adc_get_voltage(void) {
  // REVC has a 10, 1 (1/11) voltage divider
  // Here is the calculation for the scale (s)
  // ADCV = VIN_S * (1/11) * (4095/3.3)
  // RETVAL = ADCV * s = VIN_S*1000
  // s = 1000/((4095/3.3)*(1/11)) = 8.8623046875

  // Avoid needing floating point math, so output in mV
  return (adc_get(ADCCHAN_VOLTAGE) * 8862U) / 1000U;
}
