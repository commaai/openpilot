#pragma once

#include <vector>

#include "system/sensord/sensors/i2c_sensor.h"

// Address of the chip on the bus
#define MMC5603NJ_I2C_ADDR       0x30

// Registers of the chip
#define MMC5603NJ_I2C_REG_XOUT0       0x00
#define MMC5603NJ_I2C_REG_ODR         0x1A
#define MMC5603NJ_I2C_REG_INTERNAL_0  0x1B
#define MMC5603NJ_I2C_REG_INTERNAL_1  0x1C
#define MMC5603NJ_I2C_REG_INTERNAL_2  0x1D
#define MMC5603NJ_I2C_REG_ID          0x39

// Constants
#define MMC5603NJ_CHIP_ID        0x10
#define MMC5603NJ_CMM_FREQ_EN    (1 << 7)
#define MMC5603NJ_AUTO_SR_EN     (1 << 5)
#define MMC5603NJ_CMM_EN         (1 << 4)
#define MMC5603NJ_EN_PRD_SET     (1 << 3)
#define MMC5603NJ_SET            (1 << 3)
#define MMC5603NJ_RESET          (1 << 4)

class MMC5603NJ_Magn : public I2CSensor {
private:
  uint8_t get_device_address() {return MMC5603NJ_I2C_ADDR;}
  void start_measurement();
  std::vector<float> read_measurement();
public:
  MMC5603NJ_Magn(I2CBus *bus);
  int init();
  bool get_event(MessageBuilder &msg, uint64_t ts = 0);
  int shutdown();
};
