#pragma once
#include "file_sensor.h"

class LightSensor : public FileSensor {
public:
  LightSensor(std::string filename) : FileSensor(filename){};
  bool get_event(MessageBuilder &msg, std::string &service, uint64_t ts = 0);
  int shutdown() { return 0; }
};
