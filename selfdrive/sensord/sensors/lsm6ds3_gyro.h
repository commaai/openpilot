#pragma once

#include "selfdrive/sensord/sensors/i2c_sensor.h"

// Address of the chip on the bus
#define LSM6DS3_GYRO_I2C_ADDR       0x6A

// Registers of the chip
#define LSM6DS3_GYRO_I2C_REG_ID        0x0F
#define LSM6DS3_GYRO_I2C_REG_CTRL2_G   0x11
#define LSM6DS3_GYRO_I2C_REG_OUTX_L_G  0x22

// Constants
#define LSM6DS3_GYRO_CHIP_ID        0x69
#define LSM6DS3TRC_GYRO_CHIP_ID     0x6A
#define LSM6DS3_GYRO_ODR_104HZ      (0b0100 << 4)


class LSM6DS3_Gyro : public I2CSensor {
  uint8_t get_device_address() {return LSM6DS3_GYRO_I2C_ADDR;}
  cereal::SensorEventData::SensorSource source = cereal::SensorEventData::SensorSource::LSM6DS3;
public:
  LSM6DS3_Gyro(I2CBus *bus);
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
