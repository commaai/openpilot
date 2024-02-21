// ///////////////////////// //
// Jungle board v1 (STM32F4) //
// ///////////////////////// //

void board_v1_set_led(uint8_t color, bool enabled) {
  switch (color) {
    case LED_RED:
      set_gpio_output(GPIOC, 9, !enabled);
      break;
     case LED_GREEN:
      set_gpio_output(GPIOC, 7, !enabled);
      break;
    case LED_BLUE:
      set_gpio_output(GPIOC, 6, !enabled);
      break;
    default:
      break;
  }
}

void board_v1_enable_can_transciever(uint8_t transciever, bool enabled) {
  switch (transciever) {
    case 1U:
      set_gpio_output(GPIOC, 1, !enabled);
      break;
    case 2U:
      set_gpio_output(GPIOC, 13, !enabled);
      break;
    case 3U:
      set_gpio_output(GPIOA, 0, !enabled);
      break;
    case 4U:
      set_gpio_output(GPIOB, 10, !enabled);
      break;
    default:
      print("Invalid CAN transciever ("); puth(transciever); print("): enabling failed\n");
      break;
  }
}

void board_v1_set_can_mode(uint8_t mode) {
  board_v1_enable_can_transciever(2U, false);
  board_v1_enable_can_transciever(4U, false);
  switch (mode) {
    case CAN_MODE_NORMAL:
      print("Setting normal CAN mode\n");
      // B12,B13: disable OBD mode
      set_gpio_mode(GPIOB, 12, MODE_INPUT);
      set_gpio_mode(GPIOB, 13, MODE_INPUT);

      // B5,B6: normal CAN2 mode
      set_gpio_alternate(GPIOB, 5, GPIO_AF9_CAN2);
      set_gpio_alternate(GPIOB, 6, GPIO_AF9_CAN2);
      can_mode = CAN_MODE_NORMAL;
      board_v1_enable_can_transciever(2U, true);
      break;
    case CAN_MODE_OBD_CAN2:
      print("Setting OBD CAN mode\n");
      // B5,B6: disable normal CAN2 mode
      set_gpio_mode(GPIOB, 5, MODE_INPUT);
      set_gpio_mode(GPIOB, 6, MODE_INPUT);

      // B12,B13: OBD mode
      set_gpio_alternate(GPIOB, 12, GPIO_AF9_CAN2);
      set_gpio_alternate(GPIOB, 13, GPIO_AF9_CAN2);
      can_mode = CAN_MODE_OBD_CAN2;
      board_v1_enable_can_transciever(4U, true);
      break;
    default:
      print("Tried to set unsupported CAN mode: "); puth(mode); print("\n");
      break;
  }
}

void board_v1_set_harness_orientation(uint8_t orientation) {
  switch (orientation) {
    case HARNESS_ORIENTATION_NONE:
      set_gpio_output(GPIOA, 2, false);
      set_gpio_output(GPIOA, 3, false);
      set_gpio_output(GPIOA, 4, false);
      set_gpio_output(GPIOA, 5, false);
      harness_orientation = orientation;
      break;
    case HARNESS_ORIENTATION_1:
      set_gpio_output(GPIOA, 2, false);
      set_gpio_output(GPIOA, 3, (ignition != 0U));
      set_gpio_output(GPIOA, 4, true);
      set_gpio_output(GPIOA, 5, false);
      harness_orientation = orientation;
      break;
    case HARNESS_ORIENTATION_2:
      set_gpio_output(GPIOA, 2, (ignition != 0U));
      set_gpio_output(GPIOA, 3, false);
      set_gpio_output(GPIOA, 4, false);
      set_gpio_output(GPIOA, 5, true);
      harness_orientation = orientation;
      break;
    default:
      print("Tried to set an unsupported harness orientation: "); puth(orientation); print("\n");
      break;
  }
}

bool panda_power = false;
void board_v1_set_panda_power(bool enable) {
  panda_power = enable;
  set_gpio_output(GPIOB, 14, enable);
}

bool board_v1_get_button(void) {
  return get_gpio_input(GPIOC, 8);
}

void board_v1_set_ignition(bool enabled) {
  ignition = enabled ? 0xFFU : 0U;
  board_v1_set_harness_orientation(harness_orientation);
}

float board_v1_get_channel_power(uint8_t channel) {
  UNUSED(channel);
  return 0.0f;
}

uint16_t board_v1_get_sbu_mV(uint8_t channel, uint8_t sbu) {
  UNUSED(channel); UNUSED(sbu);
  return 0U;
}

void board_v1_init(void) {
  common_init_gpio();

  // A8,A15: normal CAN3 mode
  set_gpio_alternate(GPIOA, 8, GPIO_AF11_CAN3);
  set_gpio_alternate(GPIOA, 15, GPIO_AF11_CAN3);

  board_v1_set_can_mode(CAN_MODE_NORMAL);

  // Enable CAN transcievers
  for(uint8_t i = 1; i <= 4; i++) {
    board_v1_enable_can_transciever(i, true);
  }

  // Disable LEDs
  board_v1_set_led(LED_RED, false);
  board_v1_set_led(LED_GREEN, false);
  board_v1_set_led(LED_BLUE, false);

  // Set normal CAN mode
  board_v1_set_can_mode(CAN_MODE_NORMAL);

  // Set to no harness orientation
  board_v1_set_harness_orientation(HARNESS_ORIENTATION_NONE);

  // Enable panda power by default
  board_v1_set_panda_power(true);
}

void board_v1_tick(void) {}

const board board_v1 = {
  .has_canfd = false,
  .has_sbu_sense = false,
  .avdd_mV = 3300U,
  .init = &board_v1_init,
  .set_led = &board_v1_set_led,
  .board_tick = &board_v1_tick,
  .get_button = &board_v1_get_button,
  .set_panda_power = &board_v1_set_panda_power,
  .set_panda_individual_power = &unused_set_panda_individual_power,
  .set_ignition = &board_v1_set_ignition,
  .set_individual_ignition = &unused_set_individual_ignition,
  .set_harness_orientation = &board_v1_set_harness_orientation,
  .set_can_mode = &board_v1_set_can_mode,
  .enable_can_transciever = &board_v1_enable_can_transciever,
  .enable_header_pin = &unused_board_enable_header_pin,
  .get_channel_power = &board_v1_get_channel_power,
  .get_sbu_mV = &board_v1_get_sbu_mV,
};
