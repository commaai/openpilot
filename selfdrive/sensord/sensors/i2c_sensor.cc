#include "i2c_sensor.h"

int16_t read_12_bit(uint8_t lsb, uint8_t msb) {
  uint16_t combined = (uint16_t(msb) << 8) | uint16_t(lsb & 0xF0);
  return int16_t(combined) / (1 << 4);
}

int16_t read_16_bit(uint8_t lsb, uint8_t msb) {
  uint16_t combined = (uint16_t(msb) << 8) | uint16_t(lsb);
  return int16_t(combined);
}

int32_t read_20_bit(uint8_t b2, uint8_t b1, uint8_t b0) {
  uint32_t combined = (uint32_t(b0) << 16) | (uint32_t(b1) << 8) | uint32_t(b2);
  return int32_t(combined) / (1 << 4);
}


I2CSensor::I2CSensor(I2CBus *bus) : bus(bus) {
}

int I2CSensor::read_register(uint register_address, uint8_t *buffer, uint8_t len) {
  return bus->read_register(get_device_address(), register_address, buffer, len);
}

int I2CSensor::set_register(uint register_address, uint8_t data) {
  return bus->set_register(get_device_address(), register_address, data);
}
