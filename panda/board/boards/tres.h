#pragma once

#include "board_declarations.h"

// ///////////////////////////
// Tres (STM32H7) + Harness //
// ///////////////////////////

static bool tres_ir_enabled;
static bool tres_fan_enabled;
static void tres_update_fan_ir_power(void) {
  red_chiplet_set_fan_or_usb_load_switch(tres_ir_enabled || tres_fan_enabled);
}

static void tres_set_ir_power(uint8_t percentage){
  tres_ir_enabled = (percentage > 0U);
  tres_update_fan_ir_power();
  pwm_set(TIM3, 4, percentage);
}

static void tres_set_bootkick(BootState state) {
  set_gpio_output(GPIOA, 0, state != BOOT_BOOTKICK);
  set_gpio_output(GPIOC, 12, state != BOOT_RESET);
}

static void tres_set_fan_enabled(bool enabled) {
  // NOTE: fan controller reset doesn't work on a tres if IR is enabled
  tres_fan_enabled = enabled;
  tres_update_fan_ir_power();
}

static bool tres_read_som_gpio (void) {
  return (get_gpio_input(GPIOC, 2) != 0);
}

static void tres_init(void) {
  // Enable USB 3.3V LDO for USB block
  register_set_bits(&(PWR->CR3), PWR_CR3_USBREGEN);
  register_set_bits(&(PWR->CR3), PWR_CR3_USB33DEN);
  while ((PWR->CR3 & PWR_CR3_USB33RDY) == 0U);

  red_chiplet_init();

  // C2: SOM GPIO used as input (fan control at boot)
  set_gpio_mode(GPIOC, 2, MODE_INPUT);
  set_gpio_pullup(GPIOC, 2, PULL_DOWN);

  // SOM bootkick + reset lines
  // WARNING: make sure output state is set before configuring as output
  tres_set_bootkick(BOOT_BOOTKICK);
  set_gpio_mode(GPIOC, 12, MODE_OUTPUT);

  // SOM debugging UART
  gpio_uart7_init();
  uart_init(&uart_ring_som_debug, 115200);

  // SPI init
  gpio_spi_init();

  // fan setup
  set_gpio_alternate(GPIOC, 8, GPIO_AF2_TIM3);

  // Initialize IR PWM and set to 0%
  set_gpio_alternate(GPIOC, 9, GPIO_AF2_TIM3);
  pwm_init(TIM3, 4);
  tres_set_ir_power(0U);

  // Fake siren
  set_gpio_alternate(GPIOC, 10, GPIO_AF4_I2C5);
  set_gpio_alternate(GPIOC, 11, GPIO_AF4_I2C5);
  register_set_bits(&(GPIOC->OTYPER), GPIO_OTYPER_OT10 | GPIO_OTYPER_OT11); // open drain

  // Clock source
  clock_source_init();
}

board board_tres = {
  .harness_config = &red_chiplet_harness_config,
  .has_obd = true,
  .has_spi = true,
  .has_canfd = true,
  .fan_max_rpm = 6600U,
  .fan_max_pwm = 100U,
  .avdd_mV = 1800U,
  .fan_stall_recovery = false,
  .fan_enable_cooldown_time = 3U,
  .init = tres_init,
  .init_bootloader = unused_init_bootloader,
  .enable_can_transceiver = red_chiplet_enable_can_transceiver,
  .enable_can_transceivers = red_chiplet_enable_can_transceivers,
  .set_led = red_set_led,
  .set_can_mode = red_chiplet_set_can_mode,
  .check_ignition = red_check_ignition,
  .read_voltage_mV = red_read_voltage_mV,
  .read_current_mA = unused_read_current,
  .set_fan_enabled = tres_set_fan_enabled,
  .set_ir_power = tres_set_ir_power,
  .set_siren = fake_siren_set,
  .set_bootkick = tres_set_bootkick,
  .read_som_gpio = tres_read_som_gpio,
  .set_amp_enabled = unused_set_amp_enabled
};
