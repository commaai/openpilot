#pragma once

#include "system/sensord/sensors/bmx055_accel.h"
#include "system/sensord/sensors/i2c_sensor.h"

class BMX055_Temp : public I2CSensor {
  uint8_t get_device_address() {return BMX055_ACCEL_I2C_ADDR;}
public:
  BMX055_Temp(I2CBus *bus);
  int init();
  void get_event(cereal::SensorEventData::Builder &event);
};
