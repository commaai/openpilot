#pragma once

#include "sensors/i2c_sensor.hpp"

// Address of the chip on the bus
#define BMX055_ACCEL_I2C_ADDR       0x18

// Registers of the chip
#define BMX055_ACCEL_I2C_REG_ID     0x00
#define BMX055_ACCEL_I2C_REG_FIFO       0x3F

// Constants
#define BMX055_ACCEL_CHIP_ID        0xFA

class BMX055_Accel : public I2CSensor {
  uint8_t get_device_address() {return BMX055_ACCEL_I2C_ADDR;}
public:
  BMX055_Accel(I2CBus *bus);
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
