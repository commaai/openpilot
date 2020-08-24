#include "i2c.h"

#include <unistd.h>
#include <stdexcept>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

#include "common/swaglog.h"

I2CBus::I2CBus(uint8_t bus_id){
  char bus_name[20];
  snprintf(bus_name, 20, "/dev/i2c-%d", bus_id);

  i2c_fd = open(bus_name, O_RDWR);
  if(i2c_fd < 0){
    throw std::runtime_error("Failed to open I2C bus");
  }
}

I2CBus::~I2CBus(){
  if(i2c_fd >= 0){ close(i2c_fd); }
}

int I2CBus::read_register(uint8_t device_address, uint register_address, uint8_t *buffer, uint8_t len){
  int ret = 0;

  ret = ioctl(i2c_fd, I2C_SLAVE, device_address);
  if(ret < 0) { goto fail; }

  ret = i2c_smbus_read_i2c_block_data(i2c_fd, register_address, len, buffer);
  if((ret < 0) || (ret != len)) { goto fail; }

fail:
  return ret;
}

int I2CBus::set_register(uint8_t device_address, uint register_address, uint8_t data){
  int ret = 0;

  ret = ioctl(i2c_fd, I2C_SLAVE, device_address);
  if(ret < 0) { goto fail; }

  ret = i2c_smbus_write_byte_data(i2c_fd, register_address, data);
  if(ret < 0) { goto fail; }

fail:
  return ret;
}