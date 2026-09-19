#pragma once

// Digital temperature sensor (DTS)

#define DTS_SMP_TIME 15U // sensor periods per measurement (1 nibble)
#define DTS_PCLK_FREQ 60000000U // APB4 frequency, see clock.h
#define DTS_HSREF_DIV 64U // calibration counter must run below 1MHz

void dts_init(void) {
  // set sampling time, pclk reference, software trigger, calibrated measurement, calibration clock prescaler
  register_set(&(DTS->CFGR1), (((uint32_t) DTS_SMP_TIME << DTS_CFGR1_TS1_SMP_TIME_Pos) | ((uint32_t) DTS_HSREF_DIV << DTS_CFGR1_HSREF_CLK_DIV_Pos)),
    (DTS_CFGR1_TS1_SMP_TIME_Msk | DTS_CFGR1_REFCLK_SEL_Msk | DTS_CFGR1_Q_MEAS_OPT_Msk | DTS_CFGR1_HSREF_CLK_DIV_Msk | DTS_CFGR1_TS1_INTRIG_SEL_Msk));
  register_set_bits(&(DTS->CFGR1), DTS_CFGR1_TS1_EN);
  while ((DTS->SR & DTS_SR_TS1_RDY) == 0U);
  // continuous measurements w/ sw trigger
  register_set_bits(&(DTS->CFGR1), DTS_CFGR1_TS1_START);
}

float dts_get_temperature(void) {
  // formula with pclk used
  float ret = 0.0f;
  uint32_t measurement_cycles = DTS->DR & DTS_DR_TS1_MFREQ_Msk;
  uint32_t reference_frequency = (DTS->T0VALR1 & DTS_T0VALR1_TS1_FMT0_Msk) * 100U; // T0 value as Hz
  uint32_t ramp_coefficient = DTS->RAMPVALR & DTS_RAMPVALR_TS1_RAMP_COEFF_Msk; // Hz per deg C
  float reference_temperature = (((DTS->T0VALR1 & DTS_T0VALR1_TS1_T0_Msk) >> DTS_T0VALR1_TS1_T0_Pos) == 0U) ? 30.0f : 130.0f; // t0 reference temp

  if ((measurement_cycles != 0U) && (ramp_coefficient != 0U)) {
    float measurement_frequency = ((float) DTS_PCLK_FREQ * (float) DTS_SMP_TIME) / (float) measurement_cycles;
    ret = reference_temperature + ((measurement_frequency - (float) reference_frequency) / (float) ramp_coefficient);
  }
  return ret;
}
