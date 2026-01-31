#pragma once

#include <stdint.h>

#include "board/body/motor_common.h"

// Encoder pin map:
// Left motor:  PB6 -> TIM4_CH1, PB7 -> TIM4_CH2
// Right motor: PA6 -> TIM3_CH1, PA7 -> TIM3_CH2

typedef struct {
  TIM_TypeDef *timer;
  GPIO_TypeDef *pin_a_port;
  uint8_t pin_a;
  uint8_t pin_a_af;
  GPIO_TypeDef *pin_b_port;
  uint8_t pin_b;
  uint8_t pin_b_af;
  int8_t direction;
  uint32_t counts_per_output_rev;
  uint32_t min_dt_us;
  float speed_alpha;
  uint32_t filter;
} motor_encoder_config_t;

typedef struct {
  const motor_encoder_config_t *config;
  uint16_t last_timer_count;
  int32_t accumulated_count;
  int32_t last_speed_count;
  uint32_t last_speed_timestamp_us;
  float cached_speed_rps;
} motor_encoder_state_t;

static const motor_encoder_config_t motor_encoder_config[BODY_MOTOR_COUNT] = {
  [BODY_MOTOR_LEFT - 1U] = {
    .timer = TIM4,
    .pin_a_port = GPIOB, .pin_a = 6U, .pin_a_af = GPIO_AF2_TIM4,
    .pin_b_port = GPIOB, .pin_b = 7U, .pin_b_af = GPIO_AF2_TIM4,
    .direction = -1,
    .counts_per_output_rev = 44U * 90U,
    .min_dt_us = 250U,
    .speed_alpha = 0.2f,
    .filter = 3U,
  },
  [BODY_MOTOR_RIGHT - 1U] = {
    .timer = TIM3,
    .pin_a_port = GPIOA, .pin_a = 6U, .pin_a_af = GPIO_AF2_TIM3,
    .pin_b_port = GPIOA, .pin_b = 7U, .pin_b_af = GPIO_AF2_TIM3,
    .direction = 1,
    .counts_per_output_rev = 44U * 90U,
    .min_dt_us = 250U,
    .speed_alpha = 0.2f,
    .filter = 3U,
  },
};

static motor_encoder_state_t motor_encoders[BODY_MOTOR_COUNT] = {
  { .config = &motor_encoder_config[0] },
  { .config = &motor_encoder_config[1] },
};

static void motor_encoder_configure_gpio(const motor_encoder_config_t *cfg) {
  set_gpio_pullup(cfg->pin_a_port, cfg->pin_a, PULL_UP);
  set_gpio_output_type(cfg->pin_a_port, cfg->pin_a, OUTPUT_TYPE_PUSH_PULL);
  set_gpio_alternate(cfg->pin_a_port, cfg->pin_a, cfg->pin_a_af);

  set_gpio_pullup(cfg->pin_b_port, cfg->pin_b, PULL_UP);
  set_gpio_output_type(cfg->pin_b_port, cfg->pin_b, OUTPUT_TYPE_PUSH_PULL);
  set_gpio_alternate(cfg->pin_b_port, cfg->pin_b, cfg->pin_b_af);
}

static void motor_encoder_configure_timer(motor_encoder_state_t *state) {
  const motor_encoder_config_t *cfg = state->config;
  TIM_TypeDef *timer = cfg->timer;
  timer->CR1 = 0U;
  timer->CR2 = 0U;
  timer->SMCR = 0U;
  timer->DIER = 0U;
  timer->SR = 0U;
  timer->CCMR1 = (TIM_CCMR1_CC1S_0) | (TIM_CCMR1_CC2S_0) | (cfg->filter << TIM_CCMR1_IC1F_Pos) | (cfg->filter << TIM_CCMR1_IC2F_Pos);
  timer->CCER = TIM_CCER_CC1E | TIM_CCER_CC2E;
  timer->PSC = 0U;
  timer->ARR = 0xFFFFU;
  timer->CNT = 0U;
  state->last_timer_count = 0U;
  state->accumulated_count = 0;
  state->last_speed_count = 0;
  state->cached_speed_rps = 0.0f;
  timer->SMCR = (TIM_SMCR_SMS_0 | TIM_SMCR_SMS_1);
  timer->CR1 = TIM_CR1_CEN;
}

static void motor_encoder_init(void) {
  register_set_bits(&(RCC->APB1LENR), RCC_APB1LENR_TIM4EN | RCC_APB1LENR_TIM3EN);
  register_set_bits(&(RCC->APB1LRSTR), RCC_APB1LRSTR_TIM4RST | RCC_APB1LRSTR_TIM3RST);
  register_clear_bits(&(RCC->APB1LRSTR), RCC_APB1LRSTR_TIM4RST | RCC_APB1LRSTR_TIM3RST);

  for (uint8_t i = 0U; i < BODY_MOTOR_COUNT; i++) {
    motor_encoder_state_t *state = &motor_encoders[i];
    const motor_encoder_config_t *cfg = state->config;
    motor_encoder_configure_gpio(cfg);
    motor_encoder_configure_timer(state);
    state->last_speed_timestamp_us = 0U;
  }
}

static inline int32_t motor_encoder_refresh(motor_encoder_state_t *state) {
  const motor_encoder_config_t *cfg = state->config;
  TIM_TypeDef *timer = cfg->timer;
  uint16_t raw = (uint16_t)timer->CNT;
  int32_t delta = (int16_t)(raw - state->last_timer_count);
  delta *= cfg->direction;
  state->last_timer_count = raw;
  state->accumulated_count += delta;
  return state->accumulated_count;
}

static inline int32_t motor_encoder_get_position(uint8_t motor) {
  if (!body_motor_is_valid(motor)) {
    return 0;
  }
  motor_encoder_state_t *state = &motor_encoders[motor - 1U];
  return motor_encoder_refresh(state);
}

static void motor_encoder_reset(uint8_t motor) {
  if (!body_motor_is_valid(motor)) {
    return;
  }
  motor_encoder_state_t *state = &motor_encoders[motor - 1U];
  state->config->timer->CNT = 0U;
  state->last_timer_count = 0U;
  state->accumulated_count = 0;
  state->last_speed_count = 0;
  state->last_speed_timestamp_us = 0U;
  state->cached_speed_rps = 0.0f;
}

static float motor_encoder_get_speed_rpm(uint8_t motor) {
  if (!body_motor_is_valid(motor)) {
    return 0.0f;
  }
  motor_encoder_state_t *state = &motor_encoders[motor - 1U];

  const motor_encoder_config_t *cfg = state->config;
  motor_encoder_refresh(state);

  uint32_t now = microsecond_timer_get();
  if (state->last_speed_timestamp_us == 0U) {
    state->last_speed_timestamp_us = now;
    state->last_speed_count = state->accumulated_count;
    state->cached_speed_rps = 0.0f;
    return 0.0f;
  }

  uint32_t dt = now - state->last_speed_timestamp_us;
  int32_t delta = state->accumulated_count - state->last_speed_count;
  if ((dt < cfg->min_dt_us) || (delta == 0)) {
    return state->cached_speed_rps * 60.0f;
  }

  state->last_speed_count = state->accumulated_count;
  state->last_speed_timestamp_us = now;

  float counts_per_second = ((float)delta * 1000000.0f) / (float)dt;
  float new_speed_rps = (cfg->counts_per_output_rev != 0U) ? (counts_per_second / (float)cfg->counts_per_output_rev) : 0.0f;
  state->cached_speed_rps += cfg->speed_alpha * (new_speed_rps - state->cached_speed_rps);
  return state->cached_speed_rps * 60.0f;
}
