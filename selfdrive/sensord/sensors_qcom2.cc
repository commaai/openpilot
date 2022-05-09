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
  //sensors_init.push_back({&bmx055_accel, false});
  //sensors_init.push_back({&bmx055_gyro, false});
  sensors_init.push_back({&bmx055_magn, false});
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

  struct pollfd fdlist[3];

  // assumed to be configured as exported, defined as input and trigger on raising edge
  // TODO: cleanUp but works for now
  int fd_accel = open("/sys/class/gpio/gpio21/value", O_RDONLY);
  if (fd_accel < 0) { LOGE("FD ACCEL failed"); return 0; }
  int fd_gyro  = open("/sys/class/gpio/gpio23/value", O_RDONLY);
  if (fd_gyro < 0)  { LOGE("FD GYRO failed"); return 0; }
  int fd_magn  = open("/sys/class/gpio/gpio87/value", O_RDONLY);
  if (fd_magn < 0)  { LOGE("FD MAGN failed"); return 0; }

  fdlist[0].fd = fd_accel;
  fdlist[0].events = POLLPRI;

  fdlist[1].fd = fd_gyro;
  fdlist[1].events = POLLPRI;

  fdlist[2].fd = fd_magn;
  fdlist[2].events = POLLPRI;

  uint64_t start_time = nanos_since_boot();
  uint64_t f_avg = 0;
  uint64_t a_cnt = 0;

  while (1) {
    int err;

    /* events are received at 2kHz frequency, the bandwidth devider returns in the case of 125Hz
       7 times the same data, this needs to be filtered or handled smart */
    err = poll(fdlist, 3, -1);
    if (-1 == err) {
      return -1;
    }

    // TODO: check which interrupt triggered using revents

    // timing measurement
    uint64_t int_time = nanos_since_boot();
    f_avg += int_time - start_time;
    a_cnt++;
    LOGE("t: %lu, %lu", int_time - start_time, f_avg/a_cnt);
    start_time = int_time;

    // we dont read from the fd as its reset automatically after 50us
    // so we directly create the message and publish it
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
