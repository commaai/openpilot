#include "system/sensord/sensors/lsm6ds3_temp.h"

#include <cassert>

#include "common/swaglog.h"
#include "common/timing.h"

LSM6DS3_Temp::LSM6DS3_Temp(I2CBus *bus) : I2CSensor(bus) {}

int LSM6DS3_Temp::init() {
  int ret = verify_chip_id(LSM6DS3_TEMP_I2C_REG_ID, {LSM6DS3_TEMP_CHIP_ID, LSM6DS3TRC_TEMP_CHIP_ID});
  if (ret == -1) return -1;

  if (ret == LSM6DS3TRC_TEMP_CHIP_ID) {
    source = cereal::SensorEventData::SensorSource::LSM6DS3TRC;
  }
  return 0;
}

bool LSM6DS3_Temp::get_event(MessageBuilder &msg, uint64_t ts) {
  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[2];
  int len = read_register(LSM6DS3_TEMP_I2C_REG_OUT_TEMP_L, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  float scale = (source == cereal::SensorEventData::SensorSource::LSM6DS3TRC) ? 256.0f : 16.0f;
  float temp = 25.0f + read_16_bit(buffer[0], buffer[1]) / scale;

  auto event = msg.initEvent().initTemperatureSensor();
  event.setSource(source);
  event.setVersion(1);
  event.setType(SENSOR_TYPE_AMBIENT_TEMPERATURE);
  event.setTimestamp(start_time);
  event.setTemperature(temp);

  return true;
}
