#pragma once

#include "common/i2c.h"

// Address of the chip on the bus
#define BMX055_GYRO_I2C_ADDR        0x68

// Registers of the chip
#define BMX055_GYRO_I2C_REG_ID      0x00

// Constants
#define BMX055_GYRO_CHIP_ID         0x0F

class BMX055_Gyro {
  private:
    I2CBus *bus;

  public:
    BMX055_Gyro(I2CBus *i2c_bus);
    ~BMX055_Gyro();

    int init();
};
