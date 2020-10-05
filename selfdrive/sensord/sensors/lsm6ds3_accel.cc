#include <cassert>
#include "common/swaglog.h"
#include "common/timing.h"

#include "lsm6ds3_accel.hpp"


LSM6DS3_Accel::LSM6DS3_Accel(I2CBus *bus) : I2CSensor(bus) {}

int LSM6DS3_Accel::init(){
  int ret = 0;
  uint8_t buffer[1];

  ret = read_register(LSM6DS3_ACCEL_I2C_REG_ID, buffer, 1);
  if(ret < 0){
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != LSM6DS3_ACCEL_CHIP_ID){
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], LSM6DS3_ACCEL_CHIP_ID);
    ret = -1;
    goto fail;
  }


fail:
  return ret;
}

void LSM6DS3_Accel::get_event(cereal::SensorEventData::Builder &event){

}
