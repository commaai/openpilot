#include "system/sensord/sensors/bmx055_gyro.h"

#include <cassert>
#include <cmath>

#include "common/swaglog.h"
#include "common/util.h"

#define DEG2RAD(x) ((x) * M_PI / 180.0)


BMX055_Gyro::BMX055_Gyro(I2CBus *bus) : I2CSensor(bus) {}

int BMX055_Gyro::init() {
  int ret = verify_chip_id(BMX055_GYRO_I2C_REG_ID, {BMX055_GYRO_CHIP_ID});
  if (ret == -1) return -1;

  ret = set_register(BMX055_GYRO_I2C_REG_LPM1, BMX055_GYRO_NORMAL_MODE);
  if (ret < 0) {
    goto fail;
  }
  // bmx055 gyro has a 30ms wakeup time from deep suspend mode
  util::sleep_for(50);

  // High bandwidth
  // ret = set_register(BMX055_GYRO_I2C_REG_HBW, BMX055_GYRO_HBW_ENABLE);
  // if (ret < 0) {
  //   goto fail;
  // }

  // Low bandwidth
  ret = set_register(BMX055_GYRO_I2C_REG_HBW, BMX055_GYRO_HBW_DISABLE);
  if (ret < 0) {
    goto fail;
  }

  // 116 Hz filter
  ret = set_register(BMX055_GYRO_I2C_REG_BW, BMX055_GYRO_BW_116HZ);
  if (ret < 0) {
    goto fail;
  }

  // +- 125 deg/s range
  ret = set_register(BMX055_GYRO_I2C_REG_RANGE, BMX055_GYRO_RANGE_125);
  if (ret < 0) {
    goto fail;
  }

  enabled = true;

fail:
  return ret;
}

int BMX055_Gyro::shutdown()  {
  if (!enabled) return 0;

  // enter deep suspend mode (lowest power mode)
  int ret = set_register(BMX055_GYRO_I2C_REG_LPM1, BMX055_GYRO_DEEP_SUSPEND);
  if (ret < 0) {
    LOGE("Could not move BMX055 GYRO in deep suspend mode!");
  }

  return ret;
}

bool BMX055_Gyro::get_event(MessageBuilder &msg, uint64_t ts) {
  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[6];
  int len = read_register(BMX055_GYRO_I2C_REG_RATE_X_LSB, buffer, sizeof(buffer));
  assert(len == 6);

  // 16 bit = +- 125 deg/s
  float scale = 125.0f / (1 << 15);
  float x = -DEG2RAD(read_16_bit(buffer[0], buffer[1]) * scale);
  float y = -DEG2RAD(read_16_bit(buffer[2], buffer[3]) * scale);
  float z = DEG2RAD(read_16_bit(buffer[4], buffer[5]) * scale);

  auto event = msg.initEvent().initGyroscope2();
  event.setSource(cereal::SensorEventData::SensorSource::BMX055);
  event.setVersion(1);
  event.setSensor(SENSOR_GYRO_UNCALIBRATED);
  event.setType(SENSOR_TYPE_GYROSCOPE_UNCALIBRATED);
  event.setTimestamp(start_time);

  float xyz[] = {x, y, z};
  auto svec = event.initGyroUncalibrated();
  svec.setV(xyz);
  svec.setStatus(true);

  return true;
}
