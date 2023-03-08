#pragma once

#include <fstream>
#include <string>

#include "cereal/gen/cpp/log.capnp.h"
#include "system/sensord/sensors/sensor.h"

class FileSensor : public Sensor {
protected:
  std::ifstream file;

public:
  FileSensor(std::string filename);
  ~FileSensor();
  int init();
  bool has_interrupt_enabled();
  virtual bool get_event(MessageBuilder &msg, uint64_t ts = 0) = 0;
};
