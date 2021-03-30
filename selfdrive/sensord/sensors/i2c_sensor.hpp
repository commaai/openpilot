#pragma once

#include <cstdint>
#include "cereal/gen/cpp/log.capnp.h"
#include "common/i2c.h"
#include "sensors/sensor.hpp"
#include "sensors/constants.hpp"

int16_t read_12_bit(uint8_t lsb, uint8_t msb);
int16_t read_16_bit(uint8_t lsb, uint8_t msb);


class I2CSensor : public Sensor {
private:
  I2CBus *bus;
  virtual uint8_t get_device_address() = 0;

public:
  I2CSensor(I2CBus *bus);
  int read_register(uint register_address, uint8_t *buffer, uint8_t len);
  int set_register(uint register_address, uint8_t data);
  virtual int init() = 0;
  virtual void get_event(cereal::SensorEventData::Builder &event) = 0;
};
