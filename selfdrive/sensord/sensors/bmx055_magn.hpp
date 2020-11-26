#pragma once

#include "sensors/i2c_sensor.hpp"

// Address of the chip on the bus
#define BMX055_MAGN_I2C_ADDR        0x10

// Registers of the chip
#define BMX055_MAGN_I2C_REG_ID        0x40
#define BMX055_MAGN_I2C_REG_PWR_0     0x4B
#define BMX055_MAGN_I2C_REG_MAG       0x4C
#define BMX055_MAGN_I2C_REG_DATAX_LSB 0x42
#define BMX055_MAGN_I2C_REG_RHALL_LSB 0x48
#define BMX055_MAGN_I2C_REG_REPXY     0x51
#define BMX055_MAGN_I2C_REG_REPZ      0x52

// Constants
#define BMX055_MAGN_CHIP_ID     0x32
#define BMX055_MAGN_FORCED      (0b01 << 1)

class BMX055_Magn : public I2CSensor{
  uint8_t get_device_address() {return BMX055_MAGN_I2C_ADDR;}
public:
  BMX055_Magn(I2CBus *bus);
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
