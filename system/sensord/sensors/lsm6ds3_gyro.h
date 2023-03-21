#pragma once

#include "system/sensord/sensors/i2c_sensor.h"

// Address of the chip on the bus
#define LSM6DS3_GYRO_I2C_ADDR       0x6A

// Registers of the chip
#define LSM6DS3_GYRO_I2C_REG_DRDY_CFG  0x0B
#define LSM6DS3_GYRO_I2C_REG_ID        0x0F
#define LSM6DS3_GYRO_I2C_REG_INT1_CTRL 0x0D
#define LSM6DS3_GYRO_I2C_REG_CTRL2_G   0x11
#define LSM6DS3_GYRO_I2C_REG_CTRL5_C   0x14
#define LSM6DS3_GYRO_I2C_REG_STAT_REG  0x1E
#define LSM6DS3_GYRO_I2C_REG_OUTX_L_G  0x22
#define LSM6DS3_GYRO_POSITIVE_TEST   (0b01 << 2)
#define LSM6DS3_GYRO_NEGATIVE_TEST   (0b11 << 2)

// Constants
#define LSM6DS3_GYRO_CHIP_ID         0x69
#define LSM6DS3TRC_GYRO_CHIP_ID      0x6A
#define LSM6DS3_GYRO_FS_2000dps      (0b11 << 2)
#define LSM6DS3_GYRO_ODR_104HZ       (0b0100 << 4)
#define LSM6DS3_GYRO_ODR_208HZ       (0b0101 << 4)
#define LSM6DS3_GYRO_INT1_DRDY_G     0b10
#define LSM6DS3_GYRO_DRDY_GDA        0b10
#define LSM6DS3_GYRO_DRDY_PULSE_MODE (1 << 7)
#define LSM6DS3_GYRO_MIN_ST_LIMIT_mdps 150000.0f
#define LSM6DS3_GYRO_MAX_ST_LIMIT_mdps 700000.0f


class LSM6DS3_Gyro : public I2CSensor {
  uint8_t get_device_address() {return LSM6DS3_GYRO_I2C_ADDR;}
  cereal::SensorEventData::SensorSource source = cereal::SensorEventData::SensorSource::LSM6DS3;

  // self test functions
  int self_test(int test_type);
  void wait_for_data_ready();
  void read_and_avg_data(float* val_st_off);
public:
  LSM6DS3_Gyro(I2CBus *bus, int gpio_nr = 0, bool shared_gpio = false);
  int init();
  bool get_event(MessageBuilder &msg, uint64_t ts = 0);
  int shutdown();
};
