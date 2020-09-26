#include <cassert>
#include <cmath>
#include "common/swaglog.h"

#include "bmx055_gyro.hpp"

#define DEG2RAD(x) ((x) * M_PI / 180.0)


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

  // High bandwidth
  // ret = set_register(BMX055_GYRO_I2C_REG_HBW, BMX055_GYRO_HBW_ENABLE);
  // if (ret < 0){
  //   goto fail;
  // }

  // Low bandwidth
  ret = set_register(BMX055_GYRO_I2C_REG_HBW, BMX055_GYRO_HBW_DISABLE);
  if (ret < 0){
    goto fail;
  }

  // 116 Hz filter
  ret = set_register(BMX055_GYRO_I2C_REG_BW, BMX055_GYRO_BW_116HZ);
  if (ret < 0){
    goto fail;
  }

  // +- 125 deg/s range
  ret = set_register(BMX055_GYRO_I2C_REG_RANGE, BMX055_GYRO_RANGE_125);
  if (ret < 0){
    goto fail;
  }

fail:
  return ret;
}

void BMX055_Gyro::get_event(cereal::SensorEventData::Builder &event){
  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[6];
  int len = read_register(BMX055_GYRO_I2C_REG_FIFO, buffer, sizeof(buffer));
  assert(len == 6);

  // 16 bit = +- 125 deg/s
  float scale = 125.0f / (1 << 15);
  float x = -DEG2RAD(read_16_bit(buffer[0], buffer[1]) * scale);
  float y = -DEG2RAD(read_16_bit(buffer[2], buffer[3]) * scale);
  float z = DEG2RAD(read_16_bit(buffer[4], buffer[5]) * scale);

  event.setSource(cereal::SensorEventData::SensorSource::BMX055);
  event.setVersion(1);
  event.setSensor(SENSOR_GYRO_UNCALIBRATED);
  event.setType(SENSOR_TYPE_GYROSCOPE_UNCALIBRATED);
  event.setTimestamp(start_time);

  float xyz[] = {x, y, z};
  kj::ArrayPtr<const float> vs(&xyz[0], 3);

  auto svec = event.initGyroUncalibrated();
  svec.setV(vs);
  svec.setStatus(true);

}
