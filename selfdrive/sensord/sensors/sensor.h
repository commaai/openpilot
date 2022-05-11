#pragma once

#include "cereal/gen/cpp/log.capnp.h"

class Sensor {
public:
  int gpio_fd = -1;
  virtual ~Sensor() {};
  virtual int init() = 0;
  virtual bool get_event(cereal::SensorEventData::Builder &event) = 0;
  virtual bool has_interrupt_enabled() = 0;
};
