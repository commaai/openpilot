#pragma once

#include <stdbool.h>
#include <stdint.h>

#define BODY_MOTOR_COUNT 2U

typedef enum {
  BODY_MOTOR_LEFT = 1U,
  BODY_MOTOR_RIGHT = 2U,
} body_motor_id_e;

static inline bool body_motor_is_valid(uint8_t motor) {
  return (motor > 0U) && (motor <= BODY_MOTOR_COUNT);
}
