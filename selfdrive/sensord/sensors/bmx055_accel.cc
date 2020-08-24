#include "bmx055_accel.h"

#include "common/swaglog.h"

BMX055_Accel::BMX055_Accel(I2CBus *i2c_bus){
  bus = i2c_bus;
}

BMX055_Accel::~BMX055_Accel(){
  // TODO: Cleanup
}

int BMX055_Accel::init(){
  int ret = 0;
  uint8_t buffer[1];

  ret = bus->read_register(BMX055_ACCEL_I2C_ADDR, BMX055_ACCEL_I2C_REG_ID, buffer, 1);
  if(ret < 0){
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != BMX055_ACCEL_CHIP_ID){
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], BMX055_ACCEL_CHIP_ID);
    ret = -1;
    goto fail;
  }

fail:
  return ret;
}
