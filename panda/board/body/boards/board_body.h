#include "board/body/boards/board_declarations.h"

void board_body_init(void) {
  // Initialize CAN pins
  set_gpio_pullup(CAN_RX_PORT, CAN_RX_PIN, PULL_NONE);
  set_gpio_alternate(CAN_RX_PORT, CAN_RX_PIN, GPIO_AF9_FDCAN1);
  set_gpio_pullup(CAN_TX_PORT, CAN_TX_PIN, PULL_NONE);
  set_gpio_alternate(CAN_TX_PORT, CAN_TX_PIN, GPIO_AF9_FDCAN1);

  // Initialize button input (PC15)
  set_gpio_mode(IGNITION_SW_PORT, IGNITION_SW_PIN, MODE_INPUT);
  SYSCFG->EXTICR[3] &= ~(SYSCFG_EXTICR4_EXTI15);
  SYSCFG->EXTICR[3] |= SYSCFG_EXTICR4_EXTI15_PC;
  EXTI->PR1 = (1U << 15);
  EXTI->RTSR1 |= (1U << 15);
  EXTI->FTSR1 |= (1U << 15);
  EXTI->IMR1 |= (1U << 15);

  // Initialize barrel jack detection input (PC13)
  set_gpio_pullup(CHARGING_DETECT_PORT, CHARGING_DETECT_PIN, PULL_UP);
  set_gpio_mode(CHARGING_DETECT_PORT, CHARGING_DETECT_PIN, MODE_INPUT);
  SYSCFG->EXTICR[3] &= ~(SYSCFG_EXTICR4_EXTI13);
  SYSCFG->EXTICR[3] |= SYSCFG_EXTICR4_EXTI13_PC;
  EXTI->PR1 = (1U << 13);
  EXTI->RTSR1 |= (1U << 13);
  EXTI->FTSR1 |= (1U << 13);
  EXTI->IMR1 |= (1U << 13);

  // Initialize and turn on mici power
  set_gpio_mode(OBDC_POWER_ON_PORT, OBDC_POWER_ON_PIN, MODE_OUTPUT);
  set_gpio_output(OBDC_POWER_ON_PORT, OBDC_POWER_ON_PIN, 1);

  // Initialize and turn off gpu power
  set_gpio_mode(GPU_POWER_ON_PORT, GPU_POWER_ON_PIN, MODE_OUTPUT);
  set_gpio_output(GPU_POWER_ON_PORT, GPU_POWER_ON_PIN, 0);

  // Initialize and turn off ignition output
  set_gpio_mode(OBDC_IGNITION_ON_PORT, OBDC_IGNITION_ON_PIN, MODE_OUTPUT);
  set_gpio_output(OBDC_IGNITION_ON_PORT, OBDC_IGNITION_ON_PIN, 0);
}

board board_body = {
  .led_GPIO = {GPIOA, GPIOA, GPIOA},
  .led_pin = {10, 10, 10},
  .init = board_body_init,
};