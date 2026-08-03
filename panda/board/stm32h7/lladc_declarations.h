#pragma once

typedef enum {
  SAMPLETIME_1_CYCLE = 0,
  SAMPLETIME_2_CYCLES = 1,
  SAMPLETIME_8_CYCLES = 2,
  SAMPLETIME_16_CYCLES = 3,
  SAMPLETIME_32_CYCLES = 4,
  SAMPLETIME_64_CYCLES = 5,
  SAMPLETIME_387_CYCLES = 6,
  SAMPLETIME_810_CYCLES = 7
} adc_sample_time_t;

typedef enum {
  OVERSAMPLING_1 = 0,
  OVERSAMPLING_2 = 1,
  OVERSAMPLING_4 = 2,
  OVERSAMPLING_8 = 3,
  OVERSAMPLING_16 = 4,
  OVERSAMPLING_32 = 5,
  OVERSAMPLING_64 = 6,
  OVERSAMPLING_128 = 7,
  OVERSAMPLING_256 = 8,
  OVERSAMPLING_512 = 9,
  OVERSAMPLING_1024 = 10
} adc_oversampling_t;

typedef struct {
  ADC_TypeDef *adc;
  uint8_t channel;
  adc_sample_time_t sample_time;
  adc_oversampling_t oversampling;
} adc_signal_t;

#define ADC_CHANNEL_DEFAULT(a, c) {.adc = (a), .channel = (c), .sample_time = SAMPLETIME_32_CYCLES, .oversampling = OVERSAMPLING_64}

#define VREFINT_CAL_ADDR ((uint16_t *)0x1FF1E860UL)
