#include "lsm6ds3_accel.h"

#include <cassert>

#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"

LSM6DS3_Accel::LSM6DS3_Accel(I2CBus *bus) : I2CSensor(bus) {}

int LSM6DS3_Accel::init() {
  int ret = 0;
  uint8_t buffer[1];

  ret = read_register(LSM6DS3_ACCEL_I2C_REG_ID, buffer, 1);
  if(ret < 0) {
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != LSM6DS3_ACCEL_CHIP_ID && buffer[0] != LSM6DS3TRC_ACCEL_CHIP_ID) {
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], LSM6DS3_ACCEL_CHIP_ID);
    ret = -1;
    goto fail;
  }

  if (buffer[0] == LSM6DS3TRC_ACCEL_CHIP_ID) {
    source = cereal::SensorEventData::SensorSource::LSM6DS3TRC;
  }

  // TODO: set scale and bandwith. Default is +- 2G, 50 Hz
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, LSM6DS3_ACCEL_ODR_104HZ);
  if (ret < 0) {
    goto fail;
  }


fail:
  return ret;
}

void LSM6DS3_Accel::get_event(cereal::SensorEventData::Builder &event) {

  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[6];
  int len = read_register(LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  float scale = 9.81 * 2.0f / (1 << 15);
  float x = read_16_bit(buffer[0], buffer[1]) * scale;
  float y = read_16_bit(buffer[2], buffer[3]) * scale;
  float z = read_16_bit(buffer[4], buffer[5]) * scale;

  event.setSource(source);
  event.setVersion(1);
  event.setSensor(SENSOR_ACCELEROMETER);
  event.setType(SENSOR_TYPE_ACCELEROMETER);
  event.setTimestamp(start_time);

  float xyz[] = {y, -x, z};
  auto svec = event.initAcceleration();
  svec.setV(xyz);
  svec.setStatus(true);

}
