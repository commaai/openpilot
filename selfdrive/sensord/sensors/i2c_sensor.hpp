#pragma once

#include "common/i2c.h"

class I2CSensor {
private:
  I2CBus *bus;
  virtual uint8_t get_device_address() = 0;

public:
  I2CSensor(I2CBus *bus);
  int read_register(uint register_address, uint8_t *buffer, uint8_t len);
  int set_register(uint register_address, uint8_t data);
  virtual int init() = 0;
};
