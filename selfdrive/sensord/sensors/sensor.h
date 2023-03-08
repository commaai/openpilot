#pragma once

#include "cereal/messaging/messaging.h"

class Sensor {
public:
  int gpio_fd = -1;
  uint64_t init_delay = 500e6; // default dealy 500ms
  virtual ~Sensor() {};
  virtual int init() = 0;
  virtual bool get_event(MessageBuilder &msg, uint64_t ts = 0) = 0;
  virtual bool has_interrupt_enabled() = 0;
  virtual int shutdown() = 0;

  virtual bool is_data_valid(uint64_t st, uint64_t ct) {
    return (ct - st) > init_delay;
  }
};
