#pragma once

#include "board_declarations.h"

// ////////////////////////// //
// Cuatro (STM32H7) + Harness //
// ////////////////////////// //

static void cuatro_enable_can_transceiver(uint8_t transceiver, bool enabled) {
  switch (transceiver) {
    case 1U:
      set_gpio_output(GPIOB, 7, !enabled);
      break;
    case 2U:
      set_gpio_output(GPIOB, 10, !enabled);
      break;
    case 3U:
      set_gpio_output(GPIOD, 8, !enabled);
      break;
    case 4U:
      set_gpio_output(GPIOB, 11, !enabled);
      break;
    default:
      break;
  }
}

static uint32_t cuatro_read_voltage_mV(void) {
  return adc_get_mV(8) * 11U;
}

static uint32_t cuatro_read_current_mA(void) {
  return adc_get_mV(3) * 2U;
}

static void cuatro_set_fan_enabled(bool enabled) {
  set_gpio_output(GPIOD, 3, !enabled);
}

static void cuatro_set_bootkick(BootState state) {
  set_gpio_output(GPIOA, 0, state != BOOT_BOOTKICK);
  // TODO: confirm we need this
  //set_gpio_output(GPIOC, 12, state != BOOT_RESET);
}

static void cuatro_set_amp_enabled(bool enabled){
  set_gpio_output(GPIOA, 5, enabled);
}

static void cuatro_init(void) {
  common_init_gpio();

  // open drain
  set_gpio_output_type(GPIOD, 3, OUTPUT_TYPE_OPEN_DRAIN); // FAN_EN
  set_gpio_output_type(GPIOC, 12, OUTPUT_TYPE_OPEN_DRAIN); // VBAT_EN

  // Power readout
  set_gpio_mode(GPIOC, 5, MODE_ANALOG);
  set_gpio_mode(GPIOA, 6, MODE_ANALOG);

  // CAN transceiver enables
  set_gpio_pullup(GPIOB, 7, PULL_NONE);
  set_gpio_mode(GPIOB, 7, MODE_OUTPUT);
  set_gpio_pullup(GPIOD, 8, PULL_NONE);
  set_gpio_mode(GPIOD, 8, MODE_OUTPUT);

  // FDCAN3, different pins on this package than the rest of the reds
  set_gpio_pullup(GPIOD, 12, PULL_NONE);
  set_gpio_alternate(GPIOD, 12, GPIO_AF5_FDCAN3);
  set_gpio_pullup(GPIOD, 13, PULL_NONE);
  set_gpio_alternate(GPIOD, 13, GPIO_AF5_FDCAN3);

  // C2: SOM GPIO used as input (fan control at boot)
  set_gpio_mode(GPIOC, 2, MODE_INPUT);
  set_gpio_pullup(GPIOC, 2, PULL_DOWN);

  // SOM bootkick + reset lines
  cuatro_set_bootkick(BOOT_BOOTKICK);

  // SOM debugging UART
  gpio_uart7_init();
  uart_init(&uart_ring_som_debug, 115200);

  // fan setup
  set_gpio_alternate(GPIOC, 8, GPIO_AF2_TIM3);
  register_set_bits(&(GPIOC->OTYPER), GPIO_OTYPER_OT8); // open drain

  // Initialize IR PWM and set to 0%
  set_gpio_alternate(GPIOC, 9, GPIO_AF2_TIM3);
  pwm_init(TIM3, 4);
  tres_set_ir_power(0U);

  // Clock source
  clock_source_init(true);

  // Sound codec
  cuatro_set_amp_enabled(false);
  set_gpio_alternate(GPIOA, 2, GPIO_AF8_SAI4);    // SAI4_SCK_B
  set_gpio_alternate(GPIOC, 0, GPIO_AF8_SAI4);    // SAI4_FS_B
  set_gpio_alternate(GPIOD, 11, GPIO_AF10_SAI4);  // SAI4_SD_A
  set_gpio_alternate(GPIOE, 3, GPIO_AF8_SAI4);    // SAI4_SD_B
  set_gpio_alternate(GPIOE, 4, GPIO_AF3_DFSDM1);  // DFSDM1_DATIN3
  set_gpio_alternate(GPIOE, 9, GPIO_AF3_DFSDM1);  // DFSDM1_CKOUT
  set_gpio_alternate(GPIOE, 6, GPIO_AF10_SAI4);   // SAI4_MCLK_B
  sound_init();
}

static harness_configuration cuatro_harness_config = {
  .has_harness = true,
  .GPIO_SBU1 = GPIOC,
  .GPIO_SBU2 = GPIOA,
  .GPIO_relay_SBU1 = GPIOA,
  .GPIO_relay_SBU2 = GPIOA,
  .pin_SBU1 = 4,
  .pin_SBU2 = 1,
  .pin_relay_SBU1 = 9,
  .pin_relay_SBU2 = 3,
  .adc_channel_SBU1 = 4, // ADC12_INP4
  .adc_channel_SBU2 = 17 // ADC1_INP17
};

board board_cuatro = {
  .harness_config = &cuatro_harness_config,
  .has_spi = true,
  .has_canfd = true,
  .fan_max_rpm = 12500U,
  .fan_max_pwm = 99U, // it can go up to 14k RPM, but 99% -> 100% is very non-linear
  .avdd_mV = 1800U,
  .fan_stall_recovery = false,
  .fan_enable_cooldown_time = 3U,
  .init = cuatro_init,
  .init_bootloader = unused_init_bootloader,
  .enable_can_transceiver = cuatro_enable_can_transceiver,
  .led_GPIO = {GPIOC, GPIOC, GPIOC},
  .led_pin = {6, 7, 9},
  .set_can_mode = tres_set_can_mode,
  .check_ignition = red_check_ignition,
  .read_voltage_mV = cuatro_read_voltage_mV,
  .read_current_mA = cuatro_read_current_mA,
  .set_fan_enabled = cuatro_set_fan_enabled,
  .set_ir_power = unused_set_ir_power,
  .set_siren = unused_set_siren,
  .set_bootkick = cuatro_set_bootkick,
  .read_som_gpio = tres_read_som_gpio,
  .set_amp_enabled = cuatro_set_amp_enabled
};
