#pragma once
#include "file_sensor.h"

class LightSensor : public FileSensor {
public:
  LightSensor(std::string filename) : FileSensor(filename){};
  bool get_event(cereal::SensorEventData::Builder &event);
};
