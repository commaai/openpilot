uint16_t fan_tach_counter = 0U;
uint16_t fan_rpm = 0U;

void fan_set_power(uint8_t percentage){
  pwm_set(TIM3, 3, percentage);
}

// Can be way more acurate than this, but this is probably good enough for our purposes.
// Call this every second
void fan_tick(void){
    // 4 interrupts per rotation
    fan_rpm = fan_tach_counter * 15U;
    fan_tach_counter = 0U;
}
