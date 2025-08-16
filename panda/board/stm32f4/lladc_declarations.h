#pragma once

typedef enum {
  SAMPLETIME_3_CYCLES = 0,
  SAMPLETIME_15_CYCLES = 1,
  SAMPLETIME_28_CYCLES = 2,
  SAMPLETIME_56_CYCLES = 3,
  SAMPLETIME_84_CYCLES = 4,
  SAMPLETIME_112_CYCLES = 5,
  SAMPLETIME_144_CYCLES = 6,
  SAMPLETIME_480_CYCLES = 7
} adc_sample_time_t;

typedef struct {
  ADC_TypeDef *adc;
  uint8_t channel;
  adc_sample_time_t sample_time;
} adc_signal_t;

#define ADC_CHANNEL_DEFAULT(a, c) {.adc = (a), .channel = (c), .sample_time = SAMPLETIME_480_CYCLES}
