struct fan_state_t {
  uint16_t tach_counter;
  uint16_t rpm;
  uint16_t target_rpm;
  uint8_t power;
  float error_integral;
  uint8_t stall_counter;
  uint8_t stall_threshold;
  uint8_t total_stall_count;
  uint8_t cooldown_counter;
} fan_state_t;
struct fan_state_t fan_state;

const float FAN_I = 0.001f;

const uint8_t FAN_TICK_FREQ = 8U;
const uint8_t FAN_STALL_THRESHOLD_MIN = 3U;
const uint8_t FAN_STALL_THRESHOLD_MAX = 8U;


void fan_set_power(uint8_t percentage) {
  fan_state.target_rpm = ((current_board->fan_max_rpm * CLAMP(percentage, 0U, 100U)) / 100U);
}

void llfan_init(void);
void fan_init(void) {
  fan_state.stall_threshold = FAN_STALL_THRESHOLD_MIN;
  fan_state.cooldown_counter = current_board->fan_enable_cooldown_time * FAN_TICK_FREQ;
  llfan_init();
}

// Call this at FAN_TICK_FREQ
void fan_tick(void) {
  if (current_board->fan_max_rpm > 0U) {
    // Measure fan RPM
    uint16_t fan_rpm_fast = fan_state.tach_counter * (60U * FAN_TICK_FREQ / 4U);   // 4 interrupts per rotation
    fan_state.tach_counter = 0U;
    fan_state.rpm = (fan_rpm_fast + (3U * fan_state.rpm)) / 4U;

    // Stall detection
    bool fan_stalled = false;
    if (current_board->fan_stall_recovery) {
      if (fan_state.target_rpm > 0U) {
        if (fan_rpm_fast == 0U) {
          fan_state.stall_counter = MIN(fan_state.stall_counter + 1U, 255U);
        } else {
          fan_state.stall_counter = 0U;
        }

        if (fan_state.stall_counter > (fan_state.stall_threshold*FAN_TICK_FREQ)) {
          fan_stalled = true;
          fan_state.stall_counter = 0U;
          fan_state.stall_threshold = CLAMP(fan_state.stall_threshold + 2U, FAN_STALL_THRESHOLD_MIN, FAN_STALL_THRESHOLD_MAX);
          fan_state.total_stall_count += 1U;

          // datasheet gives this range as the minimum startup duty
          fan_state.error_integral = CLAMP(fan_state.error_integral, 20.0f, 45.0f);
        }
      } else {
        fan_state.stall_counter = 0U;
        fan_state.stall_threshold = FAN_STALL_THRESHOLD_MIN;
      }
    }

    #ifdef DEBUG_FAN
      puth(fan_state.target_rpm);
      print(" "); puth(fan_rpm_fast);
      print(" "); puth(fan_state.power);
      print(" "); puth(fan_state.stall_counter);
      print("\n");
    #endif

    // Cooldown counter
    if (fan_state.target_rpm > 0U) {
      fan_state.cooldown_counter = current_board->fan_enable_cooldown_time * FAN_TICK_FREQ;
    } else {
      if (fan_state.cooldown_counter > 0U) {
        fan_state.cooldown_counter--;
      }
    }

    // Update controller
    if (fan_state.target_rpm == 0U) {
      fan_state.error_integral = 0.0f;
    } else {
      float error = fan_state.target_rpm - fan_rpm_fast;
      fan_state.error_integral += FAN_I * error;
    }
    fan_state.power = CLAMP(fan_state.error_integral, 0U, 100U);

    // Set PWM and enable line
    pwm_set(TIM3, 3, fan_state.power);
    current_board->set_fan_enabled(!fan_stalled && ((fan_state.target_rpm > 0U) || (fan_state.cooldown_counter > 0U)));
  }
}
