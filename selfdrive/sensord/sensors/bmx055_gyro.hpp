#pragma once

#include "sensors/i2c_sensor.hpp"

// Address of the chip on the bus
#define BMX055_GYRO_I2C_ADDR        0x68

// Registers of the chip
#define BMX055_GYRO_I2C_REG_ID      0x00

// Constants
#define BMX055_GYRO_CHIP_ID         0x0F

class BMX055_Gyro : public I2CSensor {
  uint8_t get_device_address() {return BMX055_GYRO_I2C_ADDR;}
  public:
    BMX055_Gyro(I2CBus *bus);
    int init();
};
