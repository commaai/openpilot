#include "lsm6ds3_temp.h"

#include <cassert>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"

LSM6DS3_Temp::LSM6DS3_Temp(I2CBus *bus) : I2CSensor(bus) {}

int LSM6DS3_Temp::init() {
  int ret = 0;
  uint8_t buffer[1];

  ret = read_register(LSM6DS3_TEMP_I2C_REG_ID, buffer, 1);
  if(ret < 0) {
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != LSM6DS3_TEMP_CHIP_ID) {
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], LSM6DS3_TEMP_CHIP_ID);
    ret = -1;
    goto fail;
  }

fail:
  return ret;
}

void LSM6DS3_Temp::get_event(cereal::SensorEventData::Builder &event) {

  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[2];
  int len = read_register(LSM6DS3_TEMP_I2C_REG_OUT_TEMP_L, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  float temp = 25.0f + read_16_bit(buffer[0], buffer[1]) / 16.0f;

  event.setSource(cereal::SensorEventData::SensorSource::LSM6DS3);
  event.setVersion(1);
  event.setType(SENSOR_TYPE_AMBIENT_TEMPERATURE);
  event.setTimestamp(start_time);
  event.setTemperature(temp);

}
