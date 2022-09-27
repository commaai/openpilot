#include "light_sensor.h"

#include <string>

#include "common/timing.h"
#include "selfdrive/sensord/sensors/constants.h"

LightSensor::LightSensor(std::string filename) : FileSensor(filename) {}

bool LightSensor::get_event(MessageBuilder &msg, uint64_t ts) {
  uint64_t start_time = nanos_since_boot();
  file.clear();
  file.seekg(0);

  int value;
  file >> value;

  auto event = msg.initEvent().initLightSensor();
  event.setSource(cereal::SensorEventData::SensorSource::RPR0521);
  event.setVersion(1);
  event.setSensor(SENSOR_LIGHT);
  event.setType(SENSOR_TYPE_LIGHT);
  event.setTimestamp(start_time);
  event.setLight(value);

  return true;
}
