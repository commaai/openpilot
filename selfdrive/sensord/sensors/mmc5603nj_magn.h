#pragma once

#include "selfdrive/sensord/sensors/i2c_sensor.h"

// Address of the chip on the bus
#define MMC5603NJ_I2C_ADDR       0x30

// Registers of the chip
#define MMC5603NJ_I2C_REG_ID           0x39

// Constants
#define MMC5603NJ_CHIP_ID        0x10

class MMC5603NJ_Magn : public I2CSensor {
  uint8_t get_device_address() {return MMC5603NJ_I2C_ADDR;}
public:
  MMC5603NJ_Magn(I2CBus *bus);
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
