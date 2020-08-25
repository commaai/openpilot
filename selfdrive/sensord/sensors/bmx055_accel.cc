#include <cassert>
#include "common/swaglog.h"

#include "bmx055_accel.hpp"


BMX055_Accel::BMX055_Accel(I2CBus *bus) : I2CSensor(bus) {}

int BMX055_Accel::init(){
  int ret = 0;
  uint8_t buffer[1];

  ret = read_register(BMX055_ACCEL_I2C_REG_ID, buffer, 1);
  if(ret < 0){
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != BMX055_ACCEL_CHIP_ID){
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], BMX055_ACCEL_CHIP_ID);
    ret = -1;
    goto fail;
  }

  // High bandwidth
  ret = set_register(BMX055_ACCEL_I2C_REG_HBW, 0b10000000);
  if (ret < 0){
    goto fail;
  }

  // Low bandwidth
  // ret = set_register(BMX055_ACCEL_I2C_REG_HBW, 0b00000000);
  // if (ret < 0){
  //   goto fail;
  // }

  // // 10 Hz
  // ret = set_register(BMX055_ACCEL_I2C_REG_BW, BMX055_ACCEL_BW_7_81HZ);
  // if (ret < 0){
  //   goto fail;
  // }

fail:
  return ret;
}

void BMX055_Accel::get_event(cereal::SensorEventData::Builder &event){
  uint8_t buffer[6];
  int len = read_register(BMX055_ACCEL_I2C_REG_FIFO, buffer, sizeof(buffer));
  assert(len == 6);

  // 12 bit = +-2g
  float scale = 9.81 * 2.0f / (1 << 11);
  float x = read_12_bit(buffer[0], buffer[1]) * scale;
  float y = read_12_bit(buffer[2], buffer[3]) * scale;
  float z = read_12_bit(buffer[4], buffer[5]) * scale;

  event.setSource(cereal::SensorEventData::SensorSource::ANDROID);
  event.setVersion(1);
  event.setSensor(SENSOR_ACCELEROMETER);
  event.setType(SENSOR_TYPE_ACCELEROMETER);

  float xyz[] = {x, y, z};
  kj::ArrayPtr<const float> vs(&xyz[0], 3);

  auto svec = event.initAcceleration();
  svec.setV(vs);
  svec.setStatus(true);

}
