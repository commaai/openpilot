#include <sys/resource.h>

#include <chrono>
#include <thread>
#include <vector>

// TODO: check if really needed
#include <poll.h>
#include <fcntl.h>

#include "cereal/messaging/messaging.h"
#include "common/i2c.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
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
  //sensors_init.push_back({&bmx055_gyro, false});
  //sensors_init.push_back({&bmx055_magn, false});
  //sensors_init.push_back({&bmx055_temp, false});

  //sensors_init.push_back({&lsm6ds3_accel, true});
  //sensors_init.push_back({&lsm6ds3_gyro, true});
  //sensors_init.push_back({&lsm6ds3_temp, true});

  //sensors_init.push_back({&mmc5603nj_magn, false});

  //sensors_init.push_back({&light, true});

  bool has_magnetometer = false;

  // Initialize sensors
  std::vector<Sensor *> sensors;
  for (auto &sensor : sensors_init) {
    int err = sensor.first->init();
    if (err < 0) {
      // Fail on required sensors
      if (sensor.second) {
        LOGE("Error initializing sensors");
        //return -1;
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
    //return -1;
  }

  PubMaster pm({"sensorEvents"});



  const int num_events = sensors.size();

  struct pollfd fdlist[1];
  int fd;

  // assumed to be configured as exported, defined as input and trigger on raising edge
  // TODO: cleanUp
  fd = open("/sys/class/gpio/gpio21/value", O_RDONLY);
  if (fd < 0) {
    return 0;
  }

  fdlist[0].fd = fd;
  fdlist[0].events = POLLPRI;

  while (1) {
    int err;
    //char buf[3];

    err = poll(fdlist, 1, -1);
    if (-1 == err) {
      LOGE("poll error");
      //perror("poll");
      return 0;
    }
    // interrupt reset after 50us

    MessageBuilder msg;
    auto sensor_events = msg.initEvent().initSensorEvents(num_events);

    for (int i = 0; i < num_events; i++) {
      auto event = sensor_events[i];
      sensors[i]->get_event(event);
    }

    pm.send("sensorEvents", msg);
  }
  return 0;
}

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -18);
  return sensor_loop();
}
