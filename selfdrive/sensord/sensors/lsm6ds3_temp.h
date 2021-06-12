#pragma once

#include "selfdrive/sensord/sensors/i2c_sensor.h"

// Address of the chip on the bus
#define LSM6DS3_TEMP_I2C_ADDR       0x6A

// Registers of the chip
#define LSM6DS3_TEMP_I2C_REG_ID           0x0F
#define LSM6DS3_TEMP_I2C_REG_OUT_TEMP_L   0x20

// Constants
#define LSM6DS3_TEMP_CHIP_ID        0x69


class LSM6DS3_Temp : public I2CSensor {
  uint8_t get_device_address() {return LSM6DS3_TEMP_I2C_ADDR;}
public:
  LSM6DS3_Temp(I2CBus *bus);
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
