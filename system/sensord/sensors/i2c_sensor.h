#pragma once

#include <cstdint>
#include <unistd.h>
#include <vector>
#include "cereal/gen/cpp/log.capnp.h"

#include "common/i2c.h"
#include "common/gpio.h"

#include "common/swaglog.h"
#include "system/sensord/sensors/constants.h"
#include "system/sensord/sensors/sensor.h"

int16_t read_12_bit(uint8_t lsb, uint8_t msb);
int16_t read_16_bit(uint8_t lsb, uint8_t msb);
int32_t read_20_bit(uint8_t b2, uint8_t b1, uint8_t b0);


class I2CSensor : public Sensor {
private:
  I2CBus *bus;
  int gpio_nr;
  bool shared_gpio;
  virtual uint8_t get_device_address() = 0;

public:
  I2CSensor(I2CBus *bus, int gpio_nr = 0, bool shared_gpio = false);
  ~I2CSensor();
  int read_register(uint register_address, uint8_t *buffer, uint8_t len);
  int set_register(uint register_address, uint8_t data);
  int init_gpio();
  bool has_interrupt_enabled();
  virtual int init() = 0;
  virtual bool get_event(MessageBuilder &msg, uint64_t ts = 0) = 0;
  virtual int shutdown() = 0;

  int verify_chip_id(uint8_t address, const std::vector<uint8_t> &expected_ids) {
    uint8_t chip_id = 0;
    int ret = read_register(address, &chip_id, 1);
    if (ret < 0) {
      LOGW("Reading chip ID failed: %d", ret);
      return -1;
    }
    for (int i = 0; i < expected_ids.size(); ++i) {
      if (chip_id == expected_ids[i]) return chip_id;
    }
    LOGE("Chip ID wrong. Got: %d, Expected %d", chip_id, expected_ids[0]);
    return -1;
  }
};
