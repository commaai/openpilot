#include <vector>
#include <csignal>
#include <chrono>
#include <thread>
#include <sys/resource.h>

#include "messaging.hpp"
#include "common/i2c.h"
#include "common/timing.h"
#include "common/swaglog.h"

#include "sensors/sensor.hpp"
#include "sensors/constants.hpp"

#include "sensors/bmx055_accel.hpp"
#include "sensors/bmx055_gyro.hpp"
#include "sensors/bmx055_magn.hpp"
#include "sensors/bmx055_temp.hpp"

#include "sensors/lsm6ds3_accel.hpp"
#include "sensors/lsm6ds3_gyro.hpp"
#include "sensors/lsm6ds3_temp.hpp"

#include "sensors/light_sensor.hpp"

volatile sig_atomic_t do_exit = 0;

#define I2C_BUS_IMU 1


void set_do_exit(int sig) {
  do_exit = 1;
}

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

  LightSensor light("/sys/class/i2c-adapter/i2c-2/2-0038/iio:device1/in_intensity_both_raw");

  // Sensor init
  std::vector<Sensor *> sensors;
  sensors.push_back(&bmx055_accel);
  sensors.push_back(&bmx055_gyro);
  sensors.push_back(&bmx055_magn);
  sensors.push_back(&bmx055_temp);

  sensors.push_back(&lsm6ds3_accel);
  sensors.push_back(&lsm6ds3_gyro);
  sensors.push_back(&lsm6ds3_temp);

  sensors.push_back(&light);


  for (Sensor * sensor : sensors){
    int err = sensor->init();
    if (err < 0){
      LOGE("Error initializing sensors");
      return -1;
    }
  }

  PubMaster pm({"sensorEvents"});

  while (!do_exit){
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const int num_events = sensors.size();
    MessageBuilder msg;
    auto sensor_events = msg.initEvent().initSensorEvents(num_events);

    for (int i = 0; i < num_events; i++){
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
  setpriority(PRIO_PROCESS, 0, -13);
  signal(SIGINT, set_do_exit);
  signal(SIGTERM, set_do_exit);

  return sensor_loop();
}
