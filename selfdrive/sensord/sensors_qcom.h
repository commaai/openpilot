#include <stdint.h>
#include <hardware/sensors.h>

#define SENSOR_ACCELEROMETER 1
#define SENSOR_MAGNETOMETER 2
#define SENSOR_GYRO 4

// ACCELEROMETER_UNCALIBRATED is only in Android O
// https://developer.android.com/reference/android/hardware/Sensor.html#STRING_TYPE_ACCELEROMETER_UNCALIBRATED
#define SENSOR_MAGNETOMETER_UNCALIBRATED 3
#define SENSOR_GYRO_UNCALIBRATED 5

#define SENSOR_PROXIMITY 6
#define SENSOR_LIGHT 7



int init_sensor(struct sensors_poll_device_t* device, int sensor, int64_t delay) {
  int err;
  err = device->activate(device, sensor, 0);
  if (err != 0) goto fail;
  err = device->activate(device, sensor, 1);
  if (err != 0) goto fail;
  err = device->setDelay(device, sensor, delay);
  if (err != 0) goto fail;
  
  return 0;

fail:
  return -1;
}
