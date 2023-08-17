#include "bmx055_accel.h"

#include <cassert>

#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"

BMX055_Accel::BMX055_Accel(I2CBus *bus) : I2CSensor(bus) {}

int BMX055_Accel::init() {
  int ret = verify_chip_id(BMX055_ACCEL_I2C_REG_ID, {BMX055_ACCEL_CHIP_ID});
  if (ret == -1) return -1;

  ret = set_register(BMX055_ACCEL_I2C_REG_PMU, BMX055_ACCEL_NORMAL_MODE);
  if (ret < 0) {
    goto fail;
  }
  // bmx055 accel has a 1.3ms wakeup time from deep suspend mode
  util::sleep_for(10);

  // High bandwidth
  // ret = set_register(BMX055_ACCEL_I2C_REG_HBW, BMX055_ACCEL_HBW_ENABLE);
  // if (ret < 0) {
  //   goto fail;
  // }

  // Low bandwidth
  ret = set_register(BMX055_ACCEL_I2C_REG_HBW, BMX055_ACCEL_HBW_DISABLE);
  if (ret < 0) {
    goto fail;
  }

  ret = set_register(BMX055_ACCEL_I2C_REG_BW, BMX055_ACCEL_BW_125HZ);
  if (ret < 0) {
    goto fail;
  }

fail:
  return ret;
}

int BMX055_Accel::shutdown()  {
  // enter deep suspend mode (lowest power mode)
  int ret = set_register(BMX055_ACCEL_I2C_REG_PMU, BMX055_ACCEL_DEEP_SUSPEND);
  if (ret < 0) {
    LOGE("Could not move BMX055 ACCEL in deep suspend mode!");
  }

  return ret;
}

bool BMX055_Accel::get_event(MessageBuilder &msg, uint64_t ts) {
  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[6];
  int len = read_register(BMX055_ACCEL_I2C_REG_X_LSB, buffer, sizeof(buffer));
  assert(len == 6);

  // 12 bit = +-2g
  float scale = 9.81 * 2.0f / (1 << 11);
  float x = -read_12_bit(buffer[0], buffer[1]) * scale;
  float y = -read_12_bit(buffer[2], buffer[3]) * scale;
  float z = read_12_bit(buffer[4], buffer[5]) * scale;

  auto event = msg.initEvent().initAccelerometer2();
  event.setSource(cereal::SensorEventData::SensorSource::BMX055);
  event.setVersion(1);
  event.setSensor(SENSOR_ACCELEROMETER);
  event.setType(SENSOR_TYPE_ACCELEROMETER);
  event.setTimestamp(start_time);

  float xyz[] = {x, y, z};
  auto svec = event.initAcceleration();
  svec.setV(xyz);
  svec.setStatus(true);

  return true;
}
