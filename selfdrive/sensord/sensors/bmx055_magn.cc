#include "bmx055_magn.h"

#include "common/swaglog.h"

BMX055_Magn::BMX055_Magn(I2CBus *i2c_bus){
  bus = i2c_bus;
}

BMX055_Magn::~BMX055_Magn(){
  // TODO: Cleanup
}

int BMX055_Magn::init(){
  int ret;
  uint8_t buffer[1];

  // suspend -> sleep
  ret = i2c_set_register(i2c_fd, BMX055_MAGN_I2C_ADDR, BMX055_MAGN_I2C_REG_PWR_0, 0x01);
  if(ret < 0){
    LOGE("Enabling power failed: %d", ret);
    return ret;
  }
  usleep(5 * 1000); // wait until the chip is powered on

  // read chip ID
  ret = i2c_read_register(i2c_fd, BMX055_MAGN_I2C_ADDR, BMX055_MAGN_I2C_REG_ID, buffer, 1);
  if(ret < 0){
    LOGE("Reading chip ID failed: %d", ret);
    return ret;
  }

  if(buffer[0] != BMX055_MAGN_CHIP_ID){
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], BMX055_MAGN_CHIP_ID);
    return -1;
  }

  // perform self-test


  // sleep -> active (normal, high-precision)



  return 0;
}
