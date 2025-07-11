
typedef struct {
  ADC_TypeDef *adc;
  uint8_t channel;
} adc_channel_t;

void adc_init(ADC_TypeDef *adc) {
  adc->CR &= ~(ADC_CR_DEEPPWD); // Reset deep-power-down mode
  adc->CR |= ADC_CR_ADVREGEN; // Enable ADC regulator
  while(!(adc->ISR & ADC_ISR_LDORDY) && (adc != ADC3));

  if (adc != ADC3) {
    adc->CR &= ~(ADC_CR_ADCALDIF); // Choose single-ended calibration
    adc->CR |= ADC_CR_ADCALLIN; // Lineriality calibration
  }
  adc->CR |= ADC_CR_ADCAL; // Start calibrtation
  while((adc->CR & ADC_CR_ADCAL) != 0);

  adc->ISR |= ADC_ISR_ADRDY;
  adc->CR |= ADC_CR_ADEN;
  while(!(adc->ISR & ADC_ISR_ADRDY));
}

uint16_t adc_get_raw(ADC_TypeDef *adc, uint8_t channel) {
  adc->SQR1 &= ~(ADC_SQR1_L);
  adc->SQR1 = ((uint32_t) channel << 6U);

  if (channel < 10U) {
    adc->SMPR1 = (0x7U << (channel * 3U));
  } else {
    adc->SMPR2 = (0x7U << ((channel - 10U) * 3U));
  }
  adc->PCSEL_RES0 = (0x1U << channel);

  adc->CR |= ADC_CR_ADSTART;
  while (!(adc->ISR & ADC_ISR_EOC));

  uint16_t res = adc->DR;

  while (!(adc->ISR & ADC_ISR_EOS));
  adc->ISR |= ADC_ISR_EOS;

  return res;
}

uint16_t adc_get_mV(ADC_TypeDef *adc, uint8_t channel) {
  uint16_t ret = 0;
  if ((adc == ADC1) || (adc == ADC2)) {
    ret = (adc_get_raw(adc, channel) * current_board->avdd_mV) / 65535U;
  } else if (adc == ADC3) {
    ret = (adc_get_raw(adc, channel) * current_board->avdd_mV) / 4095U;
  } else {}
  return ret;
}
