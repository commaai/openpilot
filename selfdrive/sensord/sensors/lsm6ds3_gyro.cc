#include <cassert>
#include <cmath>
#include "common/swaglog.h"
#include "common/timing.h"

#include "lsm6ds3_gyro.hpp"

#define DEG2RAD(x) ((x) * M_PI / 180.0)


LSM6DS3_Gyro::LSM6DS3_Gyro(I2CBus *bus) : I2CSensor(bus) {}

int LSM6DS3_Gyro::init(){
  int ret = 0;
  uint8_t buffer[1];

  ret = read_register(LSM6DS3_GYRO_I2C_REG_ID, buffer, 1);
  if(ret < 0){
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != LSM6DS3_GYRO_CHIP_ID){
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], LSM6DS3_GYRO_CHIP_ID);
    ret = -1;
    goto fail;
  }

  // TODO: set scale. Default is +- 250 deg/s
  ret = set_register(LSM6DS3_GYRO_I2C_REG_CTRL2_G, LSM6DS3_GYRO_ODR_104HZ);
  if (ret < 0){
    goto fail;
  }


fail:
  return ret;
}

void LSM6DS3_Gyro::get_event(cereal::SensorEventData::Builder &event){

  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[6];
  int len = read_register(LSM6DS3_GYRO_I2C_REG_OUTX_L_G, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  float scale = 250.0f / (1 << 15);
  float x = DEG2RAD(read_16_bit(buffer[0], buffer[1]) * scale);
  float y = DEG2RAD(read_16_bit(buffer[2], buffer[3]) * scale);
  float z = DEG2RAD(read_16_bit(buffer[4], buffer[5]) * scale);

  event.setSource(cereal::SensorEventData::SensorSource::LSM6DS3);
  event.setVersion(1);
  event.setSensor(SENSOR_GYRO_UNCALIBRATED);
  event.setType(SENSOR_TYPE_GYROSCOPE_UNCALIBRATED);
  event.setTimestamp(start_time);

  float xyz[] = {y, -x, z};
  auto svec = event.initGyroUncalibrated();
  svec.setV(xyz);
  svec.setStatus(true);

}
