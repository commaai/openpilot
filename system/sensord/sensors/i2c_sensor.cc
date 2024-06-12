#include "system/sensord/sensors/i2c_sensor.h"

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

I2CSensor::I2CSensor(I2CBus *bus, int gpio_nr, bool shared_gpio) :
  bus(bus), gpio_nr(gpio_nr), shared_gpio(shared_gpio) {}

I2CSensor::~I2CSensor() {
  if (gpio_fd != -1) {
    close(gpio_fd);
  }
}

int I2CSensor::read_register(uint register_address, uint8_t *buffer, uint8_t len) {
  return bus->read_register(get_device_address(), register_address, buffer, len);
}

int I2CSensor::set_register(uint register_address, uint8_t data) {
  return bus->set_register(get_device_address(), register_address, data);
}

int I2CSensor::init_gpio() {
  if (shared_gpio || gpio_nr == 0) {
    return 0;
  }

  gpio_fd = gpiochip_get_ro_value_fd("sensord", GPIOCHIP_INT, gpio_nr);
  if (gpio_fd < 0) {
    return -1;
  }

  return 0;
}

bool I2CSensor::has_interrupt_enabled() {
  return gpio_nr != 0;
}
