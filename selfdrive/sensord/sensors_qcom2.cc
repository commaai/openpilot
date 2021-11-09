#include <sys/resource.h>

#include <chrono>
#include <thread>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/i2c.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/sensord/sensors/bmx055_accel.h"
#include "selfdrive/sensord/sensors/bmx055_gyro.h"
#include "selfdrive/sensord/sensors/bmx055_magn.h"
#include "selfdrive/sensord/sensors/bmx055_temp.h"
#include "selfdrive/sensord/sensors/constants.h"
#include "selfdrive/sensord/sensors/light_sensor.h"
#include "selfdrive/sensord/sensors/lsm6ds3_accel.h"
#include "selfdrive/sensord/sensors/lsm6ds3_gyro.h"
#include "selfdrive/sensord/sensors/lsm6ds3_temp.h"
#include "selfdrive/sensord/sensors/mmc5603nj_magn.h"
#include "selfdrive/sensord/sensors/sensor.h"

#define I2C_BUS_IMU 1

ExitHandler do_exit;

int sensor_loop() {
  I2CBus *i2c_bus_imu;

  try {
    i2c_bus_imu = new I2CBus(I2C_BUS_IMU);
  } catch (std::exception &e) {
    LOGE("I2CBus init failed");
    return -1;
  }

  BMX055_Accel bmx055_accel(i2c_bus_imu);
  BMX055_Gyro bmx055_gyro(i2c_bus_imu);
  BMX055_Magn bmx055_magn(i2c_bus_imu);
  BMX055_Temp bmx055_temp(i2c_bus_imu);

  LSM6DS3_Accel lsm6ds3_accel(i2c_bus_imu);
  LSM6DS3_Gyro lsm6ds3_gyro(i2c_bus_imu);
  LSM6DS3_Temp lsm6ds3_temp(i2c_bus_imu);

  MMC5603NJ_Magn mmc5603nj_magn(i2c_bus_imu);

  LightSensor light("/sys/class/i2c-adapter/i2c-2/2-0038/iio:device1/in_intensity_both_raw");

  // Sensor init
  std::vector<std::pair<Sensor *, bool>> sensors_init; // Sensor, required
  sensors_init.push_back({&bmx055_accel, false});
  sensors_init.push_back({&bmx055_gyro, false});
  sensors_init.push_back({&bmx055_magn, false});
  sensors_init.push_back({&bmx055_temp, false});

  sensors_init.push_back({&lsm6ds3_accel, true});
  sensors_init.push_back({&lsm6ds3_gyro, true});
  sensors_init.push_back({&lsm6ds3_temp, true});

  sensors_init.push_back({&mmc5603nj_magn, false});

  sensors_init.push_back({&light, true});

  bool has_magnetometer = false;

  // Initialize sensors
  std::vector<Sensor *> sensors;
  for (auto &sensor : sensors_init) {
    int err = sensor.first->init();
    if (err < 0) {
      // Fail on required sensors
      if (sensor.second) {
        LOGE("Error initializing sensors");
        return -1;
      }
    } else {
      if (sensor.first == &bmx055_magn || sensor.first == &mmc5603nj_magn) {
        has_magnetometer = true;
      }
      sensors.push_back(sensor.first);
    }
  }

  if (!has_magnetometer) {
    LOGE("No magnetometer present");
    return -1;
  }

  PubMaster pm({"sensorEvents"});

  while (!do_exit) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const int num_events = sensors.size();
    MessageBuilder msg;
    auto sensor_events = msg.initEvent().initSensorEvents(num_events);

    for (int i = 0; i < num_events; i++) {
      auto event = sensor_events[i];
      sensors[i]->get_event(event);
    }

    pm.send("sensorEvents", msg);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10) - (end - begin));
  }
  return 0;
}

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -18);
  return sensor_loop();
}
