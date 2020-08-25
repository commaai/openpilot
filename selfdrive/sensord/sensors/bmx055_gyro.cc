#include "common/swaglog.h"

#include "bmx055_gyro.hpp"


BMX055_Gyro::BMX055_Gyro(I2CBus *bus) : I2CSensor(bus) {}

int BMX055_Gyro::init(){
  int ret = 0;
  uint8_t buffer[1];

  ret =read_register(BMX055_GYRO_I2C_REG_ID, buffer, 1);
  if(ret < 0){
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != BMX055_GYRO_CHIP_ID){
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], BMX055_GYRO_CHIP_ID);
    ret = -1;
    goto fail;
  }

fail:
  return ret;
}
