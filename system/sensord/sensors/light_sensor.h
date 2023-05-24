#pragma once
#include "file_sensor.h"

class LightSensor : public FileSensor {
public:
  LightSensor(std::string filename);
  bool get_event(MessageBuilder &msg, uint64_t ts = 0);
  int shutdown() { return 0; }
};
