#include "bmx055_temp.h"

#include <cassert>

#include "system/sensord/sensors/bmx055_accel.h"
#include "common/swaglog.h"
#include "common/timing.h"

BMX055_Temp::BMX055_Temp(I2CBus *bus) : I2CSensor(bus) {}

int BMX055_Temp::init() {
  return verify_chip_id(BMX055_ACCEL_I2C_REG_ID, {BMX055_ACCEL_CHIP_ID}) == -1 ? -1 : 0;
}

bool BMX055_Temp::get_event(MessageBuilder &msg, uint64_t ts) {
  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[1];
  int len = read_register(BMX055_ACCEL_I2C_REG_TEMP, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  float temp = 23.0f + int8_t(buffer[0]) / 2.0f;

  auto event = msg.initEvent().initTemperatureSensor();
  event.setSource(cereal::SensorEventData::SensorSource::BMX055);
  event.setVersion(1);
  event.setType(SENSOR_TYPE_AMBIENT_TEMPERATURE);
  event.setTimestamp(start_time);
  event.setTemperature(temp);

  return true;
}
