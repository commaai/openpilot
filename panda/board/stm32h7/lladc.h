#include "lladc_declarations.h"

static uint32_t adc_avdd_mV = 0U;

void adc_init(ADC_TypeDef *adc) {
  adc->CR &= ~(ADC_CR_ADEN); // Disable ADC
  adc->CR &= ~(ADC_CR_DEEPPWD); // Reset deep-power-down mode
  adc->CR |= ADC_CR_ADVREGEN; // Enable ADC regulator
  while(!(adc->ISR & ADC_ISR_LDORDY) && (adc != ADC3));

  if (adc != ADC3) {
    adc->CR &= ~(ADC_CR_ADCALDIF); // Choose single-ended calibration
    adc->CR |= ADC_CR_ADCALLIN; // Lineriality calibration
  }
  adc->CR |= ADC_CR_ADCAL; // Start calibration
  while((adc->CR & ADC_CR_ADCAL) != 0U);

  adc->ISR |= ADC_ISR_ADRDY;
  adc->CR |= ADC_CR_ADEN;
  while(!(adc->ISR & ADC_ISR_ADRDY));
}

uint16_t adc_get_raw(const adc_signal_t *signal) {
  signal->adc->SQR1 &= ~(ADC_SQR1_L);
  signal->adc->SQR1 = ((uint32_t) signal->channel << 6U);

  // sample time
  if (signal->channel < 10U) {
    signal->adc->SMPR1 = ((uint32_t) signal->sample_time << (signal->channel * 3U));
  } else {
    signal->adc->SMPR2 = ((uint32_t) signal->sample_time << ((signal->channel - 10U) * 3U));
  }

  // select channel
  signal->adc->PCSEL_RES0 = (0x1UL << signal->channel);

  // oversampling
  signal->adc->CFGR2 = (((1U << (uint32_t) signal->oversampling) - 1U) << ADC_CFGR2_OVSR_Pos) | ((uint32_t) signal->oversampling << ADC_CFGR2_OVSS_Pos);
  signal->adc->CFGR2 |= (signal->oversampling != OVERSAMPLING_1) ? ADC_CFGR2_ROVSE : 0U;

  // start conversion
  signal->adc->CR |= ADC_CR_ADSTART;
  while (!(signal->adc->ISR & ADC_ISR_EOC));

  uint16_t res = signal->adc->DR;

  while (!(signal->adc->ISR & ADC_ISR_EOS));
  signal->adc->ISR |= ADC_ISR_EOS;

  return res;
}

static void adc_calibrate_vdda(void) {
  // ADC2 used for calibration
  adc_init(ADC2);

  // enable VREFINT channel
  ADC3_COMMON->CCR |= ADC_CCR_VREFEN;
  SYSCFG->ADC2ALT |= SYSCFG_ADC2ALT_ADC2_ROUT1;

  // measure VREFINT and derive AVDD
  uint16_t raw_vrefint = adc_get_raw(&(adc_signal_t){.adc = ADC2, .channel = 17U, .sample_time = SAMPLETIME_810_CYCLES, .oversampling = OVERSAMPLING_256});
  adc_avdd_mV = (uint32_t) *VREFINT_CAL_ADDR * 16U * 3300U / raw_vrefint;
  print("  AVDD: 0x"); puth(adc_avdd_mV); print(" mV\n");
}

uint16_t adc_get_mV(const adc_signal_t *signal) {
  uint16_t ret = 0;

  if (adc_avdd_mV == 0U) {
    adc_calibrate_vdda();
  }

  if ((signal->adc == ADC1) || (signal->adc == ADC2)) {
    ret = (adc_get_raw(signal) * adc_avdd_mV) / 65535U;
  } else if (signal->adc == ADC3) {
    ret = (adc_get_raw(signal) * adc_avdd_mV) / 4095U;
  } else {}
  return ret;
}
