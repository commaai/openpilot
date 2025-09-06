#pragma once

#include <stdint.h>
#include <stdbool.h>

// ******************** Prototypes ********************
typedef enum {
  BOOT_STANDBY,
  BOOT_BOOTKICK,
  BOOT_RESET,
} BootState;

typedef void (*board_init)(void);
typedef void (*board_init_bootloader)(void);
typedef void (*board_enable_can_transceiver)(uint8_t transceiver, bool enabled);
typedef void (*board_set_can_mode)(uint8_t mode);
typedef uint32_t (*board_read_voltage_mV)(void);
typedef uint32_t (*board_read_current_mA)(void);
typedef void (*board_set_ir_power)(uint8_t percentage);
typedef void (*board_set_fan_enabled)(bool enabled);
typedef void (*board_set_siren)(bool enabled);
typedef void (*board_set_bootkick)(BootState state);
typedef bool (*board_read_som_gpio)(void);
typedef void (*board_set_amp_enabled)(bool enabled);

struct board {
  harness_configuration *harness_config;
  GPIO_TypeDef * const led_GPIO[3];
  const uint8_t led_pin[3];
  const uint8_t led_pwm_channels[3]; // leave at 0 to disable PWM
  const bool has_spi;
  const uint16_t fan_max_rpm;
  const uint16_t avdd_mV;
  const uint8_t fan_enable_cooldown_time;
  const uint8_t fan_max_pwm;
  board_init init;
  board_init_bootloader init_bootloader;
  board_enable_can_transceiver enable_can_transceiver;
  board_set_can_mode set_can_mode;
  board_read_voltage_mV read_voltage_mV;
  board_read_current_mA read_current_mA;
  board_set_ir_power set_ir_power;
  board_set_fan_enabled set_fan_enabled;
  board_set_siren set_siren;
  board_set_bootkick set_bootkick;
  board_read_som_gpio read_som_gpio;
  board_set_amp_enabled set_amp_enabled;
};

// ******************* Definitions ********************
// These should match the enums in cereal/log.capnp and __init__.py
#define HW_TYPE_UNKNOWN 0U
#define HW_TYPE_RED_PANDA 7U
#define HW_TYPE_TRES 9U
#define HW_TYPE_CUATRO 10U

// CAN modes
#define CAN_MODE_NORMAL 0U
#define CAN_MODE_OBD_CAN2 1U

extern struct board board_tres;
extern struct board board_cuatro;
extern struct board board_red;
