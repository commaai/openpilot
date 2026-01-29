#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "board/body/motor_common.h"
#include "board/body/motor_encoder.h"

// Motor pin map:
// M1 drive: PB8 -> TIM16_CH1, PB9 -> TIM17_CH1, PE2/PE3 enables
// M2 drive: PA2 -> TIM15_CH1, PA3 -> TIM15_CH2, PE8/PE9 enables

#define PWM_TIMER_CLOCK_HZ 120000000U
#define PWM_FREQUENCY_HZ 5000U
#define PWM_PERCENT_MAX 100
#define PWM_RELOAD_TICKS ((PWM_TIMER_CLOCK_HZ + (PWM_FREQUENCY_HZ / 2U)) / PWM_FREQUENCY_HZ)

#define KP         0.25f
#define KI         0.5f
#define KD         0.008f
#define KFF        0.9f
#define MAX_RPM    100.0f
#define OUTPUT_MAX 100.0f
#define DEADBAND_RPM 1.0f
#define UPDATE_PERIOD_US 1000U

typedef struct {
  TIM_TypeDef *forward_timer;
  uint8_t forward_channel;
  TIM_TypeDef *reverse_timer;
  uint8_t reverse_channel;
  GPIO_TypeDef *pwm_port[2];
  uint8_t pwm_pin[2];
  uint8_t pwm_af[2];
  GPIO_TypeDef *enable_port[2];
  uint8_t enable_pin[2];
} motor_pwm_config_t;

static const motor_pwm_config_t motor_pwm_config[BODY_MOTOR_COUNT] = {
  [BODY_MOTOR_LEFT - 1U] = {
    TIM16, 1U, TIM17, 1U,
    {GPIOB, GPIOB}, {8U, 9U}, {GPIO_AF1_TIM16, GPIO_AF1_TIM17},
    {GPIOE, GPIOE}, {2U, 3U},
  },
  [BODY_MOTOR_RIGHT - 1U] = {
    TIM15, 2U, TIM15, 1U,
    {GPIOA, GPIOA}, {2U, 3U}, {GPIO_AF4_TIM15, GPIO_AF4_TIM15},
    {GPIOE, GPIOE}, {8U, 9U},
  },
};

typedef struct {
  bool active;
  float target_rpm;
  float integral;
  float previous_error;
  float last_output;
  uint32_t last_update_us;
} motor_speed_state_t;

static inline float motor_absf(float value) {
  return (value < 0.0f) ? -value : value;
}

static inline float motor_clampf(float value, float min_value, float max_value) {
  if (value < min_value) {
    return min_value;
  }
  if (value > max_value) {
    return max_value;
  }
  return value;
}

static motor_speed_state_t motor_speed_states[BODY_MOTOR_COUNT];

static inline void motor_pwm_write(TIM_TypeDef *timer, uint8_t channel, uint8_t percentage) {
  uint32_t period = (timer->ARR != 0U) ? timer->ARR : PWM_RELOAD_TICKS;
  uint16_t comp = (uint16_t)((period * (uint32_t)percentage) / 100U);
  if (channel == 1U) {
    register_set(&(timer->CCR1), comp, 0xFFFFU);
  } else if (channel == 2U) {
    register_set(&(timer->CCR2), comp, 0xFFFFU);
  }
}

static inline motor_speed_state_t *motor_speed_state_get(uint8_t motor) {
  return body_motor_is_valid(motor) ? &motor_speed_states[motor - 1U] : NULL;
}

static inline void motor_speed_state_reset(motor_speed_state_t *state) {
  state->active = false;
  state->target_rpm = 0.0f;
  state->integral = 0.0f;
  state->previous_error = 0.0f;
  state->last_output = 0.0f;
  state->last_update_us = 0U;
}

static void motor_speed_controller_init(void) {
  for (uint8_t i = 0U; i < BODY_MOTOR_COUNT; i++) {
    motor_speed_state_reset(&motor_speed_states[i]);
  }
}

static void motor_speed_controller_disable(uint8_t motor) {
  motor_speed_state_t *state = motor_speed_state_get(motor);
  if (state == NULL) {
    return;
  }
  motor_speed_state_reset(state);
}

static inline void motor_write_enable(uint8_t motor_index, bool enable) {
  const motor_pwm_config_t *cfg = &motor_pwm_config[motor_index];
  uint8_t level = enable ? 1U : 0U;
  set_gpio_output(cfg->enable_port[0], cfg->enable_pin[0], level);
  set_gpio_output(cfg->enable_port[1], cfg->enable_pin[1], level);
}

static void motor_init(void) {
  register_set_bits(&(RCC->AHB4ENR), RCC_AHB4ENR_GPIOAEN | RCC_AHB4ENR_GPIOBEN | RCC_AHB4ENR_GPIOEEN);
  register_set_bits(&(RCC->APB2ENR), RCC_APB2ENR_TIM16EN | RCC_APB2ENR_TIM17EN | RCC_APB2ENR_TIM15EN);

  for (uint8_t i = 0U; i < BODY_MOTOR_COUNT; i++) {
    const motor_pwm_config_t *cfg = &motor_pwm_config[i];
    motor_write_enable(i, false);
    set_gpio_alternate(cfg->pwm_port[0], cfg->pwm_pin[0], cfg->pwm_af[0]);
    set_gpio_alternate(cfg->pwm_port[1], cfg->pwm_pin[1], cfg->pwm_af[1]);

    pwm_init(cfg->forward_timer, cfg->forward_channel);
    pwm_init(cfg->reverse_timer, cfg->reverse_channel);

    TIM_TypeDef *forward_timer = cfg->forward_timer;
    register_set(&(forward_timer->PSC), 0U, 0xFFFFU);
    register_set(&(forward_timer->ARR), PWM_RELOAD_TICKS, 0xFFFFU);
    forward_timer->EGR |= TIM_EGR_UG;
    register_set(&(forward_timer->BDTR), TIM_BDTR_MOE, 0xFFFFU);

    if (cfg->reverse_timer != cfg->forward_timer) {
      TIM_TypeDef *reverse_timer = cfg->reverse_timer;
      register_set(&(reverse_timer->PSC), 0U, 0xFFFFU);
      register_set(&(reverse_timer->ARR), PWM_RELOAD_TICKS, 0xFFFFU);
      reverse_timer->EGR |= TIM_EGR_UG;
      register_set(&(reverse_timer->BDTR), TIM_BDTR_MOE, 0xFFFFU);
    }
  }
}

static void motor_apply_pwm(uint8_t motor, int32_t speed_command) {
  if (!body_motor_is_valid(motor)) {
    return;
  }

  int8_t speed = (int8_t)motor_clampf((float)speed_command, -(float)PWM_PERCENT_MAX, (float)PWM_PERCENT_MAX);
  uint8_t pwm_value = (uint8_t)((speed < 0) ? -speed : speed);
  uint8_t motor_index = motor - 1U;
  motor_write_enable(motor_index, speed != 0);
  const motor_pwm_config_t *cfg = &motor_pwm_config[motor_index];

  if (speed > 0) {
    motor_pwm_write(cfg->forward_timer, cfg->forward_channel, pwm_value);
    motor_pwm_write(cfg->reverse_timer, cfg->reverse_channel, 0U);
  } else if (speed < 0) {
    motor_pwm_write(cfg->forward_timer, cfg->forward_channel, 0U);
    motor_pwm_write(cfg->reverse_timer, cfg->reverse_channel, pwm_value);
  } else {
    motor_pwm_write(cfg->forward_timer, cfg->forward_channel, 0U);
    motor_pwm_write(cfg->reverse_timer, cfg->reverse_channel, 0U);
  }
}

static inline void motor_set_speed(uint8_t motor, int8_t speed) {
  motor_speed_controller_disable(motor);
  motor_apply_pwm(motor, (int32_t)speed);
}

static inline void motor_speed_controller_set_target_rpm(uint8_t motor, float target_rpm) {
  motor_speed_state_t *state = motor_speed_state_get(motor);
  if (state == NULL) {
    return;
  }

  target_rpm = motor_clampf(target_rpm, -MAX_RPM, MAX_RPM);
  if (motor_absf(target_rpm) <= DEADBAND_RPM) {
    motor_speed_controller_disable(motor);
    motor_apply_pwm(motor, 0);
    return;
  }

  state->active = true;
  state->target_rpm = target_rpm;
  state->integral = 0.0f;
  state->previous_error = target_rpm - motor_encoder_get_speed_rpm(motor);
  state->last_output = 0.0f;
  state->last_update_us = 0U;
}

static inline void motor_speed_controller_update(uint32_t now_us) {
  for (uint8_t motor = BODY_MOTOR_LEFT; motor <= BODY_MOTOR_RIGHT; motor++) {
    motor_speed_state_t *state = motor_speed_state_get(motor);
    if (!state->active) {
      continue;
    }

    bool first_update = (state->last_update_us == 0U);
    uint32_t dt_us = first_update ? UPDATE_PERIOD_US : (now_us - state->last_update_us);
    if (!first_update && (dt_us < UPDATE_PERIOD_US)) {
      continue;
    }

    float measured_rpm = motor_encoder_get_speed_rpm(motor);
    float error = state->target_rpm - measured_rpm;

    if ((motor_absf(state->target_rpm) <= DEADBAND_RPM) && (motor_absf(error) <= DEADBAND_RPM) && (motor_absf(measured_rpm) <= DEADBAND_RPM)) {
      motor_speed_controller_disable(motor);
      motor_apply_pwm(motor, 0);
      continue;
    }

    float dt_s = (float)dt_us * 1.0e-6f;
    float control = KFF * state->target_rpm;

    if (dt_s > 0.0f) {
      state->integral += error * dt_s;
      float derivative = first_update ? 0.0f : (error - state->previous_error) / dt_s;

      control += (KP * error) + (KI * state->integral) + (KD * derivative);
    } else {
      state->integral = 0.0f;
      control += KP * error;
    }

    if ((state->target_rpm > 0.0f) && (control < 0.0f)) {
      control = 0.0f;
      state->integral = 0.0f;
    } else if ((state->target_rpm < 0.0f) && (control > 0.0f)) {
      control = 0.0f;
      state->integral = 0.0f;
    }

    control = motor_clampf(control, -OUTPUT_MAX, OUTPUT_MAX);

    int32_t command = (control >= 0.0f) ? (int32_t)(control + 0.5f) : (int32_t)(control - 0.5f);
    motor_apply_pwm(motor, command);

    state->previous_error = error;
    state->last_output = control;
    state->last_update_us = now_us;
  }
}
