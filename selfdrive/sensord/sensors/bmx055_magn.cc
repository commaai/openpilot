#include <cassert>
#include <cstdio>
#include <algorithm>
#include <unistd.h>

#include "common/swaglog.h"
#include "common/util.h"

#include "bmx055_magn.hpp"

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

static int16_t parse_xy(uint8_t lsb, uint8_t msb){
  // 13 bit
  uint16_t combined = (uint16_t(msb) << 5) | uint16_t(lsb >> 3);
  return int16_t(combined << 3) / (1 << 3);
}

static int16_t parse_z(uint8_t lsb, uint8_t msb){
  // 15 bit
  uint16_t combined = (uint16_t(msb) << 7) | uint16_t(lsb >> 1);
  return int16_t(combined << 1) / (1 << 1);
}

static uint16_t parse_rhall(uint8_t lsb, uint8_t msb){
  // 14 bit
  return (uint16_t(msb) << 6) | uint16_t(lsb >> 2);
}

BMX055_Magn::BMX055_Magn(I2CBus *bus) : I2CSensor(bus) {}

int BMX055_Magn::init(){
  int ret;
  uint8_t buffer[1];
  uint8_t trim_x1y1[2] = {0};
  uint8_t trim_xyz_data[4] = {0};
  uint8_t trim_xy1xy2[10] = {0};

  // suspend -> sleep
  ret = set_register(BMX055_MAGN_I2C_REG_PWR_0, 0x01);
  if(ret < 0){
    LOGE("Enabling power failed: %d", ret);
    goto fail;
  }
  util::sleep_for(5); // wait until the chip is powered on

  // read chip ID
  ret = read_register(BMX055_MAGN_I2C_REG_ID, buffer, 1);
  if(ret < 0){
    LOGE("Reading chip ID failed: %d", ret);
    goto fail;
  }

  if(buffer[0] != BMX055_MAGN_CHIP_ID){
    LOGE("Chip ID wrong. Got: %d, Expected %d", buffer[0], BMX055_MAGN_CHIP_ID);
    return -1;
  }

  // Load magnetometer trim
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_X1, trim_x1y1, 2);
  if(ret < 0){
    goto fail;
  }
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_Z4, trim_xyz_data, 4);
  if(ret < 0){
    goto fail;
  }
  ret = read_register(BMX055_MAGN_I2C_REG_DIG_Z2, trim_xy1xy2, 10);
  if(ret < 0){
    goto fail;
  }

  // Read trim data
  trim_data.dig_x1 = (int8_t)trim_x1y1[0];
  trim_data.dig_y1 = (int8_t)trim_x1y1[1];

  trim_data.dig_x2 = (int8_t)trim_xyz_data[2];
  trim_data.dig_y2 = (int8_t)trim_xyz_data[3];

  trim_data.dig_z1 = read_16_bit(trim_xy1xy2[2], trim_xy1xy2[3]);
  trim_data.dig_z2 = read_16_bit(trim_xy1xy2[0], trim_xy1xy2[1]);
  trim_data.dig_z3 = read_16_bit(trim_xy1xy2[6], trim_xy1xy2[7]);
  trim_data.dig_z4 = read_16_bit(trim_xyz_data[0], trim_xyz_data[1]);

  trim_data.dig_xy1 = trim_xy1xy2[9];
  trim_data.dig_xy2 = (int8_t)trim_xy1xy2[8];

  trim_data.dig_xyz1 = read_16_bit(trim_xy1xy2[4], trim_xy1xy2[5] & 0x7f);

  // TODO: perform self-test

  // 9 REPXY and 15 REPZ for 100 Hz
  // 3 REPXY and 3 REPZ for > 300 Hz
  ret = set_register(BMX055_MAGN_I2C_REG_REPXY, (3 - 1) / 2);
  if (ret < 0){
    goto fail;
  }

  ret = set_register(BMX055_MAGN_I2C_REG_REPZ, 3 - 1);
  if (ret < 0){
    goto fail;
  }

  return 0;

 fail:
  return ret;
}


void BMX055_Magn::get_event(cereal::SensorEventData::Builder &event){
  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[8];

  int len = read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  bool ready = buffer[6] & 0x1;
  if (ready){
    int16_t x = parse_xy(buffer[0], buffer[1]);
    int16_t y = parse_xy(buffer[2], buffer[3]);
    int16_t z = parse_z(buffer[4], buffer[5]);
    int16_t rhall = parse_rhall(buffer[5], buffer[6]);

    x = compensate_x(trim_data, x, rhall);
    y = compensate_y(trim_data, y, rhall);
    z = compensate_z(trim_data, z, rhall);

    // TODO: convert to micro tesla:
    // https://github.com/BoschSensortec/BMM150-Sensor-API/blob/master/bmm150.c#L1614

    event.setSource(cereal::SensorEventData::SensorSource::BMX055);
    event.setVersion(1);
    event.setSensor(SENSOR_MAGNETOMETER_UNCALIBRATED);
    event.setType(SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED);
    event.setTimestamp(start_time);

    float xyz[] = {(float)x, (float)y, (float)z};
    auto svec = event.initMagneticUncalibrated();
    svec.setV(xyz);
    svec.setStatus(true);
  }

  // The BMX055 Magnetometer has no FIFO mode. Self running mode only goes
  // up to 30 Hz. Therefore we put in forced mode, and request measurements
  // at a 100 Hz. When reading the registers we have to check the ready bit
  // To verify the measurement was comleted this cycle.
  set_register(BMX055_MAGN_I2C_REG_MAG, BMX055_MAGN_FORCED);
}
