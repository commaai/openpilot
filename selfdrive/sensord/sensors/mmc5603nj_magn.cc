#include "mmc5603nj_magn.h"

#include <cassert>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"

MMC5603NJ_Magn::MMC5603NJ_Magn(I2CBus *bus) : I2CSensor(bus) {}

int MMC5603NJ_Magn::init() {
  int ret = 0;
  uint8_t buffer[1];

  ret = read_register(MMC5603NJ_I2C_REG_ID, buffer, 1);
  if(ret < 0) {
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != MMC5603NJ_CHIP_ID) {
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], MMC5603NJ_CHIP_ID);
    ret = -1;
    goto fail;
  }

  // Set 100 Hz
  ret = set_register(MMC5603NJ_I2C_REG_ODR, 100);
  if (ret < 0) {
    goto fail;
  }

  // Set BW to 0b01 for 1-150 Hz operation
  ret = set_register(MMC5603NJ_I2C_REG_INTERNAL_1, 0b01);
  if (ret < 0) {
    goto fail;
  }

  // Set compute measurement rate
  ret = set_register(MMC5603NJ_I2C_REG_INTERNAL_0, MMC5603NJ_CMM_FREQ_EN | MMC5603NJ_AUTO_SR_EN);
  if (ret < 0) {
    goto fail;
  }

  // Enable continuous mode, set every 100 measurements
  ret = set_register(MMC5603NJ_I2C_REG_INTERNAL_2, MMC5603NJ_CMM_EN | MMC5603NJ_EN_PRD_SET | 0b11);
  if (ret < 0) {
    goto fail;
  }

fail:
  return ret;
}

void MMC5603NJ_Magn::get_event(cereal::SensorEventData::Builder &event) {

  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[9];
  int len = read_register(MMC5603NJ_I2C_REG_XOUT0, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  float scale = 1.0 / 16384.0;
  float x = read_20_bit(buffer[6], buffer[1], buffer[0]) * scale;
  float y = read_20_bit(buffer[7], buffer[3], buffer[2]) * scale;
  float z = read_20_bit(buffer[8], buffer[5], buffer[4]) * scale;

  event.setSource(cereal::SensorEventData::SensorSource::MMC5603NJ);
  event.setVersion(1);
  event.setSensor(SENSOR_MAGNETOMETER_UNCALIBRATED);
  event.setType(SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED);
  event.setTimestamp(start_time);

  float xyz[] = {x, y, z};
  auto svec = event.initMagneticUncalibrated();
  svec.setV(xyz);
  svec.setStatus(true);

}
