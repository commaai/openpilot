#pragma once

#include <cstdint>
#include <mutex>

#include <sys/types.h>

class I2CBus {
  private:
    int i2c_fd;
    std::mutex m;

  public:
    I2CBus(uint8_t bus_id);
    ~I2CBus();

    int read_register(uint8_t device_address, uint register_address, uint8_t *buffer, uint8_t len);
    int set_register(uint8_t device_address, uint register_address, uint8_t data);
};
