#pragma once

#include "cereal/gen/cpp/log.capnp.h"

class Sensor {
public:
  virtual int init() = 0;
  virtual void get_event(cereal::SensorEventData::Builder &event) = 0;
};
