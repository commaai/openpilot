#include <iostream>
#include "i2c_sensor.hpp"

I2CSensor::I2CSensor(I2CBus *bus) : bus(bus){
  std::cout << "sensor init" << std::endl;
}

int I2CSensor::read_register(uint register_address, uint8_t *buffer, uint8_t len){
  return bus->read_register(get_device_address(), register_address, buffer, len);
}

int I2CSensor::set_register(uint register_address, uint8_t data){
  return bus->set_register(get_device_address(), register_address, data);
}
