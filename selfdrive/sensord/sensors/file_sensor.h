#pragma once

#include <fstream>
#include <string>

#include "cereal/gen/cpp/log.capnp.h"
#include "selfdrive/sensord/sensors/sensor.h"

class FileSensor : public Sensor {
protected:
  std::ifstream file;

public:
  FileSensor(std::string filename, uint64_t init_delay = 0);
  ~FileSensor();
  int init();
  bool has_interrupt_enabled();
  virtual bool get_event(MessageBuilder &msg, std::string &service, uint64_t ts = 0) = 0;
};
