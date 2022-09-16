#include "lsm6ds3_accel.h"

#include <cassert>

#include "common/swaglog.h"
#include "common/timing.h"

LSM6DS3_Accel::LSM6DS3_Accel(I2CBus *bus, int gpio_nr, bool shared_gpio) : I2CSensor(bus, gpio_nr, shared_gpio) {}

int LSM6DS3_Accel::init() {
  int ret = 0;
  uint8_t buffer[1];
  uint8_t value = 0;

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

  ret = init_gpio();
  if (ret < 0) {
    goto fail;
  }

  // TODO: set scale and bandwidth. Default is +- 2G, 50 Hz
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, LSM6DS3_ACCEL_ODR_104HZ);
  if (ret < 0) {
    goto fail;
  }

  ret = set_register(LSM6DS3_ACCEL_I2C_REG_DRDY_CFG, LSM6DS3_ACCEL_DRDY_PULSE_MODE);
  if (ret < 0) {
    goto fail;
  }

  // enable data ready interrupt for accel on INT1
  // (without resetting existing interrupts)
  ret = read_register(LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, &value, 1);
  if (ret < 0) {
    goto fail;
  }

  value |= LSM6DS3_ACCEL_INT1_DRDY_XL;
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, value);

fail:
  return ret;
}

int LSM6DS3_Accel::shutdown() {
  int ret = 0;

  // disable data ready interrupt for accel on INT1
  uint8_t value = 0;
  ret = read_register(LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, &value, 1);
  if (ret < 0) {
    goto fail;
  }

  value &= ~(LSM6DS3_ACCEL_INT1_DRDY_XL);
  ret = set_register(LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, value);
  if (ret < 0) {
    goto fail;
  }
  return ret;

fail:
  LOGE("Could not disable lsm6ds3 acceleration interrupt!")
  return ret;
}

bool LSM6DS3_Accel::get_event(cereal::SensorEventData::Builder &event) {

  if (has_interrupt_enabled()) {
    // INT1 shared with gyro, check STATUS_REG who triggered
    uint8_t status_reg = 0;
    read_register(LSM6DS3_ACCEL_I2C_REG_STAT_REG, &status_reg, sizeof(status_reg));
    if ((status_reg & LSM6DS3_ACCEL_DRDY_XLDA) == 0) {
      return false;
    }
  }

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

  float xyz[] = {y, -x, z};
  auto svec = event.initAcceleration();
  svec.setV(xyz);
  svec.setStatus(true);

  return true;
}
