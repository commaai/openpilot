struct fan_state_t {
  uint16_t tach_counter;
  uint16_t rpm;
  uint16_t target_rpm;
  uint8_t power;
  float error_integral;
  uint8_t stall_counter;
} fan_state_t;
struct fan_state_t fan_state;

#define FAN_I 0.001f
#define FAN_STALL_THRESHOLD 25U

void fan_set_power(uint8_t percentage){
  fan_state.target_rpm = ((current_board->fan_max_rpm * MIN(100U, MAX(0U, percentage))) / 100U);
}

// Call this at 8Hz
void fan_tick(void){
  if (current_board->fan_max_rpm > 0U) {
    // 4 interrupts per rotation
    uint16_t fan_rpm_fast = fan_state.tach_counter * (60U * 8U / 4U);
    fan_state.tach_counter = 0U;
    fan_state.rpm = (fan_rpm_fast + (3U * fan_state.rpm)) / 4U;

    // Enable fan if we want it to spin
    current_board->set_fan_enabled(fan_state.target_rpm > 0U);

    // Stall detection
    if(fan_state.power > 0U) {
      if (fan_rpm_fast == 0U) {
        fan_state.stall_counter = MIN(fan_state.stall_counter + 1U, 255U);
      } else {
        fan_state.stall_counter = 0U;
      }

      if (fan_state.stall_counter > FAN_STALL_THRESHOLD) {
        // Stall detected, power cycling fan controller
        current_board->set_fan_enabled(false);
        fan_state.error_integral = 0U;
      }
    } else {
      fan_state.stall_counter = 0U;
    }

    // Update controller
    float feedforward = (fan_state.target_rpm * 100.0f) / current_board->fan_max_rpm;
    float error = fan_state.target_rpm - fan_rpm_fast;

    fan_state.error_integral += FAN_I * error;
    fan_state.error_integral = MIN(70.0f, MAX(-70.0f, fan_state.error_integral));

    fan_state.power = MIN(100U, MAX(0U, feedforward + fan_state.error_integral));
    pwm_set(TIM3, 3, fan_state.power);
  }
}
