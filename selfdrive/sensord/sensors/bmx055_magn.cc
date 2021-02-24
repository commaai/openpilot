#include <cassert>
#include <cstdio>
#include <algorithm>
#include <unistd.h>

#include "common/swaglog.h"
#include "common/util.h"

#include "bmx055_magn.hpp"


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

  perform_self_test();

  return 0;

 fail:
  return ret;
}

bool BMX055_Magn::perform_self_test(){
  uint8_t buffer[8];
  int16_t x, y, z;

  // Clean existing measurement
  read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));

  // Normal
  uint8_t reg = BMX055_MAGN_FORCED;
  set_register(BMX055_MAGN_I2C_REG_MAG, reg);
  util::sleep_for(25);

  read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));
  parse_xyz(buffer, &x, &y, &z);
  LOGE("No current %d", z);

  // Negative current
  set_register(BMX055_MAGN_I2C_REG_MAG, reg | (uint8_t(0b01) << 6));
  util::sleep_for(25);

  read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));
  parse_xyz(buffer, &x, &y, &z);
  LOGE("Negative current %d", z);

  // Positive current
  set_register(BMX055_MAGN_I2C_REG_MAG, reg | (uint8_t(0b11) << 6));
  util::sleep_for(25);

  read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));
  parse_xyz(buffer, &x, &y, &z);
  LOGE("Negative current %d", z);

  // Put back in normal mode
  set_register(BMX055_MAGN_I2C_REG_MAG, reg);

  return true;
}

bool BMX055_Magn::parse_xyz(uint8_t buffer[8], int16_t *x, int16_t *y, int16_t *z){
  bool ready = buffer[6] & 0x1;
  if (ready){
    int16_t mdata_x = (int16_t) (((int16_t)buffer[1] << 8) | buffer[0]) >> 3;
    int16_t mdata_y = (int16_t) (((int16_t)buffer[3] << 8) | buffer[2]) >> 3;
    int16_t mdata_z = (int16_t) (((int16_t)buffer[5] << 8) | buffer[4]) >> 1;
    uint16_t data_r = (uint16_t) (((uint16_t)buffer[7] << 8) | buffer[6]) >> 2;

    // calculate temperature compensated 16-bit magnetic fields
    int16_t temp = ((int16_t)(((uint16_t)((((int32_t)trim_data.dig_xyz1) << 14)/(data_r != 0 ? data_r : trim_data.dig_xyz1))) - ((uint16_t)0x4000)));
    *x = ((int16_t)((((int32_t)mdata_x) *
          ((((((((int32_t)trim_data.dig_xy2) * ((((int32_t)temp) * ((int32_t)temp)) >> 7)) +
            (((int32_t)temp) * ((int32_t)(((int16_t)trim_data.dig_xy1) << 7)))) >> 9) +
          ((int32_t)0x100000)) * ((int32_t)(((int16_t)trim_data.dig_x2) + ((int16_t)0xA0)))) >> 12)) >> 13)) +
        (((int16_t)trim_data.dig_x1) << 3);

    temp = ((int16_t)(((uint16_t)((((int32_t)trim_data.dig_xyz1) << 14)/(data_r != 0 ? data_r : trim_data.dig_xyz1))) - ((uint16_t)0x4000)));
    *y = ((int16_t)((((int32_t)mdata_y) *
          ((((((((int32_t)trim_data.dig_xy2) * ((((int32_t)temp) * ((int32_t)temp)) >> 7)) +
            (((int32_t)temp) * ((int32_t)(((int16_t)trim_data.dig_xy1) << 7)))) >> 9) +
                ((int32_t)0x100000)) * ((int32_t)(((int16_t)trim_data.dig_y2) + ((int16_t)0xA0)))) >> 12)) >> 13)) +
        (((int16_t)trim_data.dig_y1) << 3);
    *z = (((((int32_t)(mdata_z - trim_data.dig_z4)) << 15) - ((((int32_t)trim_data.dig_z3) * ((int32_t)(((int16_t)data_r) -
    ((int16_t)trim_data.dig_xyz1))))>>2))/(trim_data.dig_z2 + ((int16_t)(((((int32_t)trim_data.dig_z1) * ((((int16_t)data_r) << 1)))+(1<<15))>>16))));
  }
  return ready;
}


void BMX055_Magn::get_event(cereal::SensorEventData::Builder &event){
  uint64_t start_time = nanos_since_boot();
  uint8_t buffer[8];
  int16_t x, y, z;

  int len = read_register(BMX055_MAGN_I2C_REG_DATAX_LSB, buffer, sizeof(buffer));
  assert(len == sizeof(buffer));

  if (parse_xyz(buffer, &x, &y, &z)){
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
