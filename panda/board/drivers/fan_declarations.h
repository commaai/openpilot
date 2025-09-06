#pragma once

struct fan_state_t {
  uint16_t tach_counter;
  uint16_t rpm;
  uint16_t target_rpm;
  uint8_t power;
  float error_integral;
  uint8_t cooldown_counter;
};
extern struct fan_state_t fan_state;

void fan_set_power(uint8_t percentage);
void llfan_init(void);
void fan_init(void);
// Call this at FAN_TICK_FREQ
void fan_tick(void);
