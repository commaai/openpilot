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

fail:
  return ret;
}

void MMC5603NJ_Magn::get_event(cereal::SensorEventData::Builder &event) {

  uint64_t start_time = nanos_since_boot();
  // uint8_t buffer[6];
  // int len = read_register(LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL, buffer, sizeof(buffer));
  // assert(len == sizeof(buffer));

  // float scale = 9.81 * 2.0f / (1 << 15);
  // float x = read_16_bit(buffer[0], buffer[1]) * scale;
  // float y = read_16_bit(buffer[2], buffer[3]) * scale;
  // float z = read_16_bit(buffer[4], buffer[5]) * scale;

  event.setSource(cereal::SensorEventData::SensorSource::LSM6DS3);
  event.setVersion(1);
  event.setSensor(SENSOR_MAGNETOMETER_UNCALIBRATED);
  event.setType(SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED);
  event.setTimestamp(start_time);

  float xyz[] = {0, 0, 0};
  auto svec = event.initMagneticUncalibrated();
  svec.setV(xyz);
  svec.setStatus(true);

}
