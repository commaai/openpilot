#pragma once

#include <stdint.h>
#include <stdbool.h>

// ******************** Prototypes ********************
typedef void (*board_init)(void);
typedef void (*board_init_bootloader)(void);
typedef void (*board_enable_can_transceiver)(uint8_t transceiver, bool enabled);

struct board {
  GPIO_TypeDef * const led_GPIO[3];
  const uint8_t led_pin[3];
  const uint8_t led_pwm_channels[3]; // leave at 0 to disable PWM
  board_init init;
  board_init_bootloader init_bootloader;
  const bool has_spi;
};

// ******************* Definitions ********************
#define HW_TYPE_BODY 0xB1U
