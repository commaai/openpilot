#pragma once

// Address of the chip on the bus
#define BMX055_MAGN_I2C_ADDR        0x10

// Registers of the chip
#define BMX055_MAGN_I2C_REG_ID      0x40
#define BMX055_MAGN_I2C_REG_PWR_0   0x4B

// Constants
#define BMX055_MAGN_CHIP_ID         0x32

class BMX055_Magn {
  private:
    I2CBus *bus;

  public:
    BMX055_Magn(I2CBus *i2c_bus);
    ~BMX055_Magn();

    int init();
};
