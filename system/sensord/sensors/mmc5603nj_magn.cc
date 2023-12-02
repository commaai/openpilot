#include "system/sensord/sensors/mmc5603nj_magn.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"

MMC5603NJ_Magn::MMC5603NJ_Magn(I2CBus *bus) : I2CSensor(bus) {}

int MMC5603NJ_Magn::init() {
  int ret = verify_chip_id(MMC5603NJ_I2C_REG_ID, {MMC5603NJ_CHIP_ID});
  if (ret == -1) return -1;

  // Set ODR to 0
  ret = set_register(MMC5603NJ_I2C_REG_ODR, 0);
  if (ret < 0) {
    goto fail;
  }

  // Set BW to 0b01 for 1-150 Hz operation
  ret = set_register(MMC5603NJ_I2C_REG_INTERNAL_1, 0b01);
  if (ret < 0) {
    goto fail;
  }

fail:
  return ret;
}

int MMC5603NJ_Magn::shutdown() {
  int ret = 0;

  // disable auto reset of measurements
  uint8_t value = 0;
  ret = read_register(MMC5603NJ_I2C_REG_INTERNAL_0, &value, 1);
  if (ret < 0) {
    goto fail;
  }

  value &= ~(MMC5603NJ_CMM_FREQ_EN | MMC5603NJ_AUTO_SR_EN);
  ret = set_register(MMC5603NJ_I2C_REG_INTERNAL_0, value);
  if (ret < 0) {
    goto fail;
  }

  // set ODR to 0 to leave continuous mode
  ret = set_register(MMC5603NJ_I2C_REG_ODR, 0);
  if (ret < 0) {
    goto fail;
  }
  return ret;

fail:
  LOGE("Could not disable mmc5603nj auto set reset");
  return ret;
}

void MMC5603NJ_Magn::start_measurement() {
  set_register(MMC5603NJ_I2C_REG_INTERNAL_0, 0b01);
  util::sleep_for(5);
}

std::vector<float> MMC5603NJ_Magn::read_measurement() {
  int len;
  uint8_t buffer[9];
  len = read_register(MMC5603NJ_I2C_REG_XOUT0, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));
  float scale = 1.0 / 16384.0;
  float x = (read_20_bit(buffer[6], buffer[1], buffer[0]) * scale) - 32.0;
  float y = (read_20_bit(buffer[7], buffer[3], buffer[2]) * scale) - 32.0;
  float z = (read_20_bit(buffer[8], buffer[5], buffer[4]) * scale) - 32.0;
  std::vector<float> xyz = {x, y, z};
  return xyz;
}

bool MMC5603NJ_Magn::get_event(MessageBuilder &msg, uint64_t ts) {
  uint64_t start_time = nanos_since_boot();
  // SET - RESET cycle
  set_register(MMC5603NJ_I2C_REG_INTERNAL_0, MMC5603NJ_SET);
  util::sleep_for(5);
  MMC5603NJ_Magn::start_measurement();
  std::vector<float> xyz = MMC5603NJ_Magn::read_measurement();

  set_register(MMC5603NJ_I2C_REG_INTERNAL_0, MMC5603NJ_RESET);
  util::sleep_for(5);
  MMC5603NJ_Magn::start_measurement();
  std::vector<float> reset_xyz = MMC5603NJ_Magn::read_measurement();

  auto event = msg.initEvent().initMagnetometer();
  event.setSource(cereal::SensorEventData::SensorSource::MMC5603NJ);
  event.setVersion(1);
  event.setSensor(SENSOR_MAGNETOMETER_UNCALIBRATED);
  event.setType(SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED);
  event.setTimestamp(start_time);

  float vals[] = {xyz[0], xyz[1], xyz[2], reset_xyz[0], reset_xyz[1], reset_xyz[2]};
  bool valid = true;
  if (std::any_of(std::begin(vals), std::end(vals), [](float val) { return val == -32.0; })) {
    valid = false;
  }
  auto svec = event.initMagneticUncalibrated();
  svec.setV(vals);
  svec.setStatus(valid);
  return true;
}
