#pragma once

#include "selfdrive/sensord/sensors/i2c_sensor.h"

// Address of the chip on the bus
#define BMX055_GYRO_I2C_ADDR        0x68

// Registers of the chip
#define BMX055_GYRO_I2C_REG_ID         0x00
#define BMX055_GYRO_I2C_REG_RATE_X_LSB 0x02
#define BMX055_GYRO_I2C_REG_RANGE      0x0F
#define BMX055_GYRO_I2C_REG_BW         0x10
#define BMX055_GYRO_I2C_REG_LPM1       0x11
#define BMX055_GYRO_I2C_REG_HBW        0x13
#define BMX055_GYRO_I2C_REG_FIFO       0x3F

// Constants
#define BMX055_GYRO_CHIP_ID         0x0F

#define BMX055_GYRO_HBW_ENABLE       0b10000000
#define BMX055_GYRO_HBW_DISABLE      0b00000000
#define BMX055_GYRO_DEEP_SUSPEND     0b00100000
#define BMX055_GYRO_NORMAL_MODE      0b00000000

#define BMX055_GYRO_RANGE_2000      0b000
#define BMX055_GYRO_RANGE_1000      0b001
#define BMX055_GYRO_RANGE_500       0b010
#define BMX055_GYRO_RANGE_250       0b011
#define BMX055_GYRO_RANGE_125       0b100

#define BMX055_GYRO_BW_116HZ 0b0010


class BMX055_Gyro : public I2CSensor {
  uint8_t get_device_address() {return BMX055_GYRO_I2C_ADDR;}
public:
  BMX055_Gyro(I2CBus *bus);
  int init();
  bool get_event(MessageBuilder &msg, uint64_t ts = 0);
  int shutdown();
};
