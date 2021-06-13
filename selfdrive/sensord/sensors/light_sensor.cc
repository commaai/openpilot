#include "light_sensor.h"

#include <string>

#include "selfdrive/common/timing.h"
#include "selfdrive/sensord/sensors/constants.h"

void LightSensor::get_event(cereal::SensorEventData::Builder &event) {
  uint64_t start_time = nanos_since_boot();
  file.clear();
  file.seekg(0);

  int value;
  file >> value;

  event.setSource(cereal::SensorEventData::SensorSource::RPR0521);
  event.setVersion(1);
  event.setSensor(SENSOR_LIGHT);
  event.setType(SENSOR_TYPE_LIGHT);
  event.setTimestamp(start_time);
  event.setLight(value);
}
