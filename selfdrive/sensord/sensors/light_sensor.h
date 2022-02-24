#pragma once
#include "file_sensor.h"

class LightSensor : public FileSensor {
public:
  LightSensor(std::string filename) : FileSensor(filename){};
  void get_event(cereal::SensorEventData::Builder &event);
};
