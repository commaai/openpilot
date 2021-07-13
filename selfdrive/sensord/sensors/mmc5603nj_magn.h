#pragma once

#include "selfdrive/sensord/sensors/i2c_sensor.h"

// Address of the chip on the bus
#define MMC5603NJ_I2C_ADDR       0x30

// Registers of the chip
#define MMC5603NJ_I2C_REG_XOUT0       0x00
#define MMC5603NJ_I2C_REG_ODR         0x1A
#define MMC5603NJ_I2C_REG_INTERNAL_0  0x1B
#define MMC5603NJ_I2C_REG_INTERNAL_1  0x1C
#define MMC5603NJ_I2C_REG_INTERNAL_2  0x1D
#define MMC5603NJ_I2C_REG_ID          0x39

// Constants
#define MMC5603NJ_CHIP_ID        0x10
#define MMC5603NJ_CMM_FREQ_EN    (1 << 7)
#define MMC5603NJ_AUTO_SR_EN     (1 << 5)
#define MMC5603NJ_CMM_EN         (1 << 4)
#define MMC5603NJ_EN_PRD_SET     (1 << 3)

class MMC5603NJ_Magn : public I2CSensor {
  uint8_t get_device_address() {return MMC5603NJ_I2C_ADDR;}
public:
  MMC5603NJ_Magn(I2CBus *bus);
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
