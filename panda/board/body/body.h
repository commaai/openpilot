#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "board/can.h"
#include "board/body/bldc/bldc_defs.h"

// Motor control
extern volatile uint16_t batt_voltage_raw;
extern volatile uint16_t batt_percentage;
extern volatile int rpm_left;
extern volatile int rpm_right;
extern volatile bool enable_motors;

void motor_set_enable(bool enable);
float motor_encoder_get_speed_rpm(uint8_t motor);
uint8_t motor_get_error_code(uint8_t motor);
void bldc_init(void);
void bldc_step(void);

// CAN communication
void body_can_rx(CANPacket_t *msg);
void body_can_init(void);
void body_can_periodic(uint32_t now, bool ignition, bool plug_charging);

// Status LEDs
typedef struct {
  uint8_t r;
  uint8_t g;
  uint8_t b;
} dotstar_rgb_t;

void dotstar_init(void);
void dotstar_show(void);
void dotstar_run_rainbow(uint32_t now_us);
void dotstar_apply_breathe(dotstar_rgb_t color, uint32_t now_us, uint32_t cycle_us);

#ifndef BOOTSTUB
extern int _app_start[0xc000];
#endif
