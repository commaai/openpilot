#pragma once
#include <tuple>

#include "selfdrive/sensord/sensors/i2c_sensor.h"

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

#define BMX055_MAGN_I2C_REG_DIG_X1       0x5D
#define BMX055_MAGN_I2C_REG_DIG_Y1       0x5E
#define BMX055_MAGN_I2C_REG_DIG_Z4_LSB   0x62
#define BMX055_MAGN_I2C_REG_DIG_Z4_MSB   0x63
#define BMX055_MAGN_I2C_REG_DIG_X2       0x64
#define BMX055_MAGN_I2C_REG_DIG_Y2       0x65
#define BMX055_MAGN_I2C_REG_DIG_Z2_LSB   0x68
#define BMX055_MAGN_I2C_REG_DIG_Z2_MSB   0x69
#define BMX055_MAGN_I2C_REG_DIG_Z1_LSB   0x6A
#define BMX055_MAGN_I2C_REG_DIG_Z1_MSB   0x6B
#define BMX055_MAGN_I2C_REG_DIG_XYZ1_LSB 0x6C
#define BMX055_MAGN_I2C_REG_DIG_XYZ1_MSB 0x6D
#define BMX055_MAGN_I2C_REG_DIG_Z3_LSB   0x6E
#define BMX055_MAGN_I2C_REG_DIG_Z3_MSB   0x6F
#define BMX055_MAGN_I2C_REG_DIG_XY2      0x70
#define BMX055_MAGN_I2C_REG_DIG_XY1      0x71

// Constants
#define BMX055_MAGN_CHIP_ID     0x32
#define BMX055_MAGN_FORCED      (0b01 << 1)

struct trim_data_t {
    int8_t dig_x1;
    int8_t dig_y1;
    int8_t dig_x2;
    int8_t dig_y2;
    uint16_t dig_z1;
    int16_t dig_z2;
    int16_t dig_z3;
    int16_t dig_z4;
    uint8_t dig_xy1;
    int8_t dig_xy2;
    uint16_t dig_xyz1;
};


class BMX055_Magn : public I2CSensor{
  uint8_t get_device_address() {return BMX055_MAGN_I2C_ADDR;}
  trim_data_t trim_data = {0};
  bool perform_self_test();
  bool parse_xyz(uint8_t buffer[8], int16_t *x, int16_t *y, int16_t *z);
public:
  BMX055_Magn(I2CBus *bus);
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
