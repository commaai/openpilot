#pragma once

// Pin definitions
#ifdef QCOM2
  #define GPIO_HUB_RST_N        30
  #define GPIO_UBLOX_RST_N      32
  #define GPIO_UBLOX_SAFEBOOT_N 33
  #define GPIO_UBLOX_PWR_EN     34
  #define GPIO_STM_RST_N        124
  #define GPIO_STM_BOOT0        134
  #define GPIO_BMX_ACCEL_INT    21
  #define GPIO_BMX_GYRO_INT     23
  #define GPIO_BMX_MAGN_INT     87
  #define GPIO_LSM_INT          84
  #define GPIOCHIP_INT          0
#else
  #define GPIO_HUB_RST_N        0
  #define GPIO_UBLOX_RST_N      0
  #define GPIO_UBLOX_SAFEBOOT_N 0
  #define GPIO_UBLOX_PWR_EN     0
  #define GPIO_STM_RST_N        0
  #define GPIO_STM_BOOT0        0
  #define GPIO_BMX_ACCEL_INT    0
  #define GPIO_BMX_GYRO_INT     0
  #define GPIO_BMX_MAGN_INT     0
  #define GPIO_LSM_INT          0
  #define GPIOCHIP_INT          0
#endif

int gpio_init(int pin_nr, bool output);
int gpio_set(int pin_nr, bool high);

int gpiochip_get_ro_value_fd(const char* consumer_label, int gpiochiop_id, int pin_nr);
