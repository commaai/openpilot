#include <cassert>
#include "common/swaglog.h"
#include "common/timing.h"

#include "bmx055_temp.hpp"
#include "bmx055_accel.hpp"


BMX055_Temp::BMX055_Temp(I2CBus *bus) : I2CSensor(bus) {}

int BMX055_Temp::init(){
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

fail:
  return ret;
}

void BMX055_Temp::get_event(cereal::SensorEventData::Builder &event){
  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[1];
  int len = read_register(BMX055_ACCEL_I2C_REG_TEMP, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  float temp = 23.0f + int8_t(buffer[0]) / 2.0f;

  event.setSource(cereal::SensorEventData::SensorSource::BMX055);
  event.setVersion(1);
  event.setType(SENSOR_TYPE_AMBIENT_TEMPERATURE);
  event.setTimestamp(start_time);
  event.setTemperature(temp);
}
