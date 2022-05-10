#pragma once

// Pin definitions
#ifdef QCOM2
  #define GPIO_HUB_RST_N        30
  #define GPIO_UBLOX_RST_N      32
  #define GPIO_UBLOX_SAFEBOOT_N 33
  #define GPIO_UBLOX_PWR_EN     34
  #define GPIO_STM_RST_N        124
  #define GPIO_STM_BOOT0        134
#else
  #define GPIO_HUB_RST_N        0
  #define GPIO_UBLOX_RST_N      0
  #define GPIO_UBLOX_SAFEBOOT_N 0
  #define GPIO_UBLOX_PWR_EN     0
  #define GPIO_STM_RST_N        0
  #define GPIO_STM_BOOT0        0
#endif

enum Edgetypes { Rising, Falling, Both, None };

int gpio_init(int pin_nr, bool output);
int gpio_set(int pin_nr, bool high);
int gpio_set_edge(int pin_nr, Edgetypes etype);
int gpio_get_ro_value_fd(int pin_nr);
