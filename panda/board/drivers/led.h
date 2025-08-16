
#define LED_RED 0U
#define LED_GREEN 1U
#define LED_BLUE 2U

#define LED_PWM_POWER 5U

void led_set(uint8_t color, bool enabled) {
  if (color < 3U) {
    if (current_board->led_pwm_channels[color] != 0U) {
      pwm_set(TIM3, current_board->led_pwm_channels[color], 100U - (enabled ? LED_PWM_POWER : 0U));
    } else {
      set_gpio_output(current_board->led_GPIO[color], current_board->led_pin[color], !enabled);
    }
  }
}

void led_init(void) {
  for (uint8_t i = 0U; i<3U; i++){
    set_gpio_pullup(current_board->led_GPIO[i], current_board->led_pin[i], PULL_NONE);
    set_gpio_output_type(current_board->led_GPIO[i], current_board->led_pin[i], OUTPUT_TYPE_OPEN_DRAIN);

    if (current_board->led_pwm_channels[i] != 0U) {
      set_gpio_alternate(current_board->led_GPIO[i], current_board->led_pin[i], GPIO_AF2_TIM3);
      pwm_init(TIM3, current_board->led_pwm_channels[i]);
    } else {
      set_gpio_mode(current_board->led_GPIO[i], current_board->led_pin[i], MODE_OUTPUT);
    }

    led_set(i, false);
  }
}
