#pragma once

#include "common/i2c.h"

// Address of the chip on the bus
#define BMX055_ACCEL_I2C_ADDR       0x18

// Registers of the chip
#define BMX055_ACCEL_I2C_REG_ID     0x00

// Constants
#define BMX055_ACCEL_CHIP_ID        0xFA

class BMX055_Accel {
  private:
    I2CBus *bus;

  public:
    BMX055_Accel(I2CBus *i2c_bus);
    ~BMX055_Accel();

    int init();
};
