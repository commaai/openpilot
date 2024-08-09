#include "system/sensord/sensors/bmx055_magn.h"

#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdio>

#include "common/swaglog.h"
#include "common/util.h"

static int16_t compensate_x(trim_data_t trim_data, int16_t mag_data_x, uint16_t data_rhall) {
  uint16_t process_comp_x0 = data_rhall;
  int32_t process_comp_x1 = ((int32_t)trim_data.dig_xyz1) * 16384;
  uint16_t process_comp_x2 = ((uint16_t)(process_comp_x1 / process_comp_x0)) - ((uint16_t)0x4000);
  int16_t retval = ((int16_t)process_comp_x2);
  int32_t process_comp_x3 = (((int32_t)retval) * ((int32_t)retval));
  int32_t process_comp_x4 = (((int32_t)trim_data.dig_xy2) * (process_comp_x3 / 128));
  int32_t process_comp_x5 = (int32_t)(((int16_t)trim_data.dig_xy1) * 128);
  int32_t process_comp_x6 = ((int32_t)retval) * process_comp_x5;
  int32_t process_comp_x7 = (((process_comp_x4 + process_comp_x6) / 512) + ((int32_t)0x100000));
  int32_t process_comp_x8 = ((int32_t)(((int16_t)trim_data.dig_x2) + ((int16_t)0xA0)));
  int32_t process_comp_x9 = ((process_comp_x7 * process_comp_x8) / 4096);
  int32_t process_comp_x10 = ((int32_t)mag_data_x) * process_comp_x9;
  retval = ((int16_t)(process_comp_x10 / 8192));
  retval = (retval + (((int16_t)trim_data.dig_x1) * 8)) / 16;

  return retval;
}

static int16_t compensate_y(trim_data_t trim_data, int16_t mag_data_y, uint16_t data_rhall) {
  uint16_t process_comp_y0 = trim_data.dig_xyz1;
  int32_t process_comp_y1 = (((int32_t)trim_data.dig_xyz1) * 16384) / process_comp_y0;
  uint16_t process_comp_y2 = ((uint16_t)process_comp_y1) - ((uint16_t)0x4000);
  int16_t retval = ((int16_t)process_comp_y2);
  int32_t process_comp_y3 = ((int32_t) retval) * ((int32_t)retval);
  int32_t process_comp_y4 = ((int32_t)trim_data.dig_xy2) * (process_comp_y3 / 128);
  int32_t process_comp_y5 = ((int32_t)(((int16_t)trim_data.dig_xy1) * 128));
  int32_t process_comp_y6 = ((process_comp_y4 + (((int32_t)retval) * process_comp_y5)) / 512);
  int32_t process_comp_y7 = ((int32_t)(((int16_t)trim_data.dig_y2) + ((int16_t)0xA0)));
  int32_t process_comp_y8 = (((process_comp_y6 + ((int32_t)0x100000)) * process_comp_y7) / 4096);
  int32_t process_comp_y9 = (((int32_t)mag_data_y) * process_comp_y8);
  retval = (int16_t)(process_comp_y9 / 8192);
  retval = (retval + (((int16_t)trim_data.dig_y1) * 8)) / 16;

  return retval;
}

static int16_t compensate_z(trim_data_t trim_data, int16_t mag_data_z, uint16_t data_rhall) {
  int16_t process_comp_z0 = ((int16_t)data_rhall) - ((int16_t) trim_data.dig_xyz1);
  int32_t process_comp_z1 = (((int32_t)trim_data.dig_z3) * ((int32_t)(process_comp_z0))) / 4;
  int32_t process_comp_z2 = (((int32_t)(mag_data_z - trim_data.dig_z4)) * 32768);
  int32_t process_comp_z3 = ((int32_t)trim_data.dig_z1) * (((int16_t)data_rhall) * 2);
  int16_t process_comp_z4 = (int16_t)((process_comp_z3 + (32768)) / 65536);
  int32_t retval = ((process_comp_z2 - process_comp_z1) / (trim_data.dig_z2 + process_comp_z4));

  /* saturate result to +/- 2 micro-tesla */
  retval = std::clamp(retval, -32767, 32767);

  /* Conversion of LSB to micro-tesla*/
  retval = retval / 16;

  return (int16_t)retval;
}

BMX055_Magn::BMX055_Magn(I2CBus *bus) : I2CSensor(bus) {}

int BMX055_Magn::init() {
  uint8_t trim_x1y1[2] = {0};
  uint8_t trim_x2y2[2] = {0};
  uint8_t trim_xy1xy2[2] = {0};
  uint8_t trim_z1[2] = {0};
  uint8_t trim_z2[2] = {0};
  uint8_t trim_z3[2] = {0};
  uint8_t trim_z4[2] = {0};
  uint8_t trim_xyz1[2] = {0};

  // suspend -> sleep
  int ret = set_register(BMX055_MAGN_I2C_REG_PWR_0, 0x01);
  if (ret < 0) {
    LOGW("Enabling power failed: %d", ret);
    goto fail;
  }
  util::sleep_for(5); // wait until the chip is powered on

  ret = verify_chip_id(BMX055_MAGN_I2C_REG_ID, {BMX055_MAGN_CHIP_ID});
  if (ret == -1) {
    goto fail;
  }

  // Load magnetometer trim
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_X1, trim_x1y1, 2);
  if (ret < 0) goto fail;
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_X2, trim_x2y2, 2);
  if (ret < 0) goto fail;
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_XY2, trim_xy1xy2, 2);
  if (ret < 0) goto fail;
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_Z1_LSB, trim_z1, 2);
  if (ret < 0) goto fail;
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_Z2_LSB, trim_z2, 2);
  if (ret < 0) goto fail;
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_Z3_LSB, trim_z3, 2);
  if (ret < 0) goto fail;
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_Z4_LSB, trim_z4, 2);
  if (ret < 0) goto fail;
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_XYZ1_LSB, trim_xyz1, 2);
  if (ret < 0) goto fail;

  // Read trim data
  trim_data.dig_x1 = trim_x1y1[0];
  trim_data.dig_y1 = trim_x1y1[1];

  trim_data.dig_x2 = trim_x2y2[0];
  trim_data.dig_y2 = trim_x2y2[1];

  trim_data.dig_xy1 = trim_xy1xy2[1]; // NB: MSB/LSB swapped
  trim_data.dig_xy2 = trim_xy1xy2[0];

  trim_data.dig_z1 = read_16_bit(trim_z1[0], trim_z1[1]);
  trim_data.dig_z2 = read_16_bit(trim_z2[0], trim_z2[1]);
  trim_data.dig_z3 = read_16_bit(trim_z3[0], trim_z3[1]);
  trim_data.dig_z4 = read_16_bit(trim_z4[0], trim_z4[1]);

  trim_data.dig_xyz1 = read_16_bit(trim_xyz1[0], trim_xyz1[1] & 0x7f);
  assert(trim_data.dig_xyz1 != 0);

  perform_self_test();

  // f_max = 1 / (145us * nXY + 500us * NZ + 980us)
  // Chose NXY = 7, NZ = 12, which gives 125 Hz,
  // and has the same ratio as the high accuracy preset
  ret = set_register(BMX055_MAGN_I2C_REG_REPXY, (7 - 1) / 2);
  if (ret < 0) {
    goto fail;
  }

  ret = set_register(BMX055_MAGN_I2C_REG_REPZ, 12 - 1);
  if (ret < 0) {
    goto fail;
  }


  return 0;

 fail:
  return ret;
}

int BMX055_Magn::shutdown() {
  // move to suspend mode
  int ret = set_register(BMX055_MAGN_I2C_REG_PWR_0, 0);
  if (ret < 0) {
    LOGE("Could not move BMX055 MAGN in suspend mode!");
  }

  return ret;
}

bool BMX055_Magn::perform_self_test() {
  uint8_t buffer[8];
  int16_t x, y;
  int16_t neg_z, pos_z;

  // Increase z reps for less false positives (~30 Hz ODR)
  set_register(BMX055_MAGN_I2C_REG_REPXY, 1);
  set_register(BMX055_MAGN_I2C_REG_REPZ, 64 - 1);

  // Clean existing measurement
  read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));

  uint8_t forced = BMX055_MAGN_FORCED;

  // Negative current
  set_register(BMX055_MAGN_I2C_REG_MAG, forced | (uint8_t(0b10) << 6));
  util::sleep_for(100);

  read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));
  parse_xyz(buffer, &x, &y, &neg_z);

  // Positive current
  set_register(BMX055_MAGN_I2C_REG_MAG, forced | (uint8_t(0b11) << 6));
  util::sleep_for(100);

  read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));
  parse_xyz(buffer, &x, &y, &pos_z);

  // Put back in normal mode
  set_register(BMX055_MAGN_I2C_REG_MAG, 0);

  int16_t diff = pos_z - neg_z;
  bool passed = (diff > 180) && (diff < 240);

  if (!passed) {
    LOGE("self test failed: neg %d pos %d diff %d", neg_z, pos_z, diff);
  }

  return passed;
}

bool BMX055_Magn::parse_xyz(uint8_t buffer[8], int16_t *x, int16_t *y, int16_t *z) {
  bool ready = buffer[6] & 0x1;
  if (ready) {
    int16_t mdata_x = (int16_t) (((int16_t)buffer[1] << 8) | buffer[0]) >> 3;
    int16_t mdata_y = (int16_t) (((int16_t)buffer[3] << 8) | buffer[2]) >> 3;
    int16_t mdata_z = (int16_t) (((int16_t)buffer[5] << 8) | buffer[4]) >> 1;
    uint16_t data_r = (uint16_t) (((uint16_t)buffer[7] << 8) | buffer[6]) >> 2;
    assert(data_r != 0);

    *x = compensate_x(trim_data, mdata_x, data_r);
    *y = compensate_y(trim_data, mdata_y, data_r);
    *z = compensate_z(trim_data, mdata_z, data_r);
  }
  return ready;
}


bool BMX055_Magn::get_event(MessageBuilder &msg, uint64_t ts) {
  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[8];
  int16_t _x, _y, x, y, z;

  int len = read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  bool parsed = parse_xyz(buffer, &_x, &_y, &z);
  if (parsed) {

    auto event = msg.initEvent().initMagnetometer();
    event.setSource(cereal::SensorEventData::SensorSource::BMX055);
    event.setVersion(2);
    event.setSensor(SENSOR_MAGNETOMETER_UNCALIBRATED);
    event.setType(SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED);
    event.setTimestamp(start_time);

    // Move magnetometer into same reference frame as accel/gryo
    x = -_y;
    y = _x;

    // Axis convention
    x = -x;
    y = -y;

    float xyz[] = {(float)x, (float)y, (float)z};
    auto svec = event.initMagneticUncalibrated();
    svec.setV(xyz);
    svec.setStatus(true);
  }

  // The BMX055 Magnetometer has no FIFO mode. Self running mode only goes
  // up to 30 Hz. Therefore we put in forced mode, and request measurements
  // at a 100 Hz. When reading the registers we have to check the ready bit
  // To verify the measurement was completed this cycle.
  set_register(BMX055_MAGN_I2C_REG_MAG, BMX055_MAGN_FORCED);

  return parsed;
}
