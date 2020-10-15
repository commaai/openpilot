#include <cassert>
#include <cstdio>
#include <unistd.h>

#include "common/swaglog.h"

#include "bmx055_magn.hpp"

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

/*
static uint16_t parse_rhall(uint8_t lsb, uint8_t msb){
  // 14 bit
  return (uint16_t(msb) << 6) | uint16_t(lsb >> 2);
}
*/

BMX055_Magn::BMX055_Magn(I2CBus *bus) : I2CSensor(bus) {}

int BMX055_Magn::init(){
  int ret;
  uint8_t buffer[1];

  // suspend -> sleep
  ret = set_register(BMX055_MAGN_I2C_REG_PWR_0, 0x01);
  if(ret < 0){
    LOGE("Enabling power failed: %d", ret);
    goto fail;
  }
  usleep(5 * 1000); // wait until the chip is powered on

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
    float x = parse_xy(buffer[0], buffer[1]);
    float y = parse_xy(buffer[2], buffer[3]);
    float z = parse_z(buffer[4], buffer[5]);
    //uint16_t rhall = parse_rhall(buffer[5], buffer[6]);

    // TODO: convert to micro tesla:
    // https://github.com/BoschSensortec/BMM150-Sensor-API/blob/master/bmm150.c#L1614

    event.setSource(cereal::SensorEventData::SensorSource::BMX055);
    event.setVersion(1);
    event.setSensor(SENSOR_MAGNETOMETER_UNCALIBRATED);
    event.setType(SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED);
    event.setTimestamp(start_time);

    float xyz[] = {x, y, z};
    kj::ArrayPtr<const float> vs(&xyz[0], 3);

    auto svec = event.initMagneticUncalibrated();
    svec.setV(vs);
    svec.setStatus(true);
  }

  // The BMX055 Magnetometer has no FIFO mode. Self running mode only goes
  // up to 30 Hz. Therefore we put in forced mode, and request measurements
  // at a 100 Hz. When reading the registers we have to check the ready bit
  // To verify the measurement was comleted this cycle.
  set_register(BMX055_MAGN_I2C_REG_MAG, BMX055_MAGN_FORCED);
}
