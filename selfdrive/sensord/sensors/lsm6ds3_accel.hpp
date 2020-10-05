#pragma once

#include "sensors/i2c_sensor.hpp"

// Address of the chip on the bus
#define LSM6DS3_ACCEL_I2C_ADDR       (0xD4 >> 1)

// Registers of the chip
#define LSM6DS3_ACCEL_I2C_REG_ID     0x0F

// Constants
#define LSM6DS3_ACCEL_CHIP_ID        0x69


class LSM6DS3_Accel : public I2CSensor {
  uint8_t get_device_address() {return LSM6DS3_ACCEL_I2C_ADDR;}
public:
  LSM6DS3_Accel(I2CBus *bus);
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
