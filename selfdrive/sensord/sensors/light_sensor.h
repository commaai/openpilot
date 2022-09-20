#pragma once
#include "file_sensor.h"

class LightSensor : public FileSensor {
public:
  LightSensor(std::string filename, uint64_t init_delay = 0);
  bool get_event(MessageBuilder &msg, std::string &service, uint64_t ts = 0);
  int shutdown() { return 0; }
};
