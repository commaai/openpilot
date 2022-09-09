#pragma once

#include "cereal/messaging/messaging.h"

class Sensor {
public:
  int gpio_fd = -1;
  virtual ~Sensor() {};
  virtual int init() = 0;
  virtual bool get_event(MessageBuilder &msg, std::string &service, uint64_t ts = 0) = 0;
  virtual bool has_interrupt_enabled() = 0;
  virtual int shutdown() = 0;
};
