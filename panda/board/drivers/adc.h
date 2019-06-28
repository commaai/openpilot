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

uint32_t adc_get(int channel) {
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

