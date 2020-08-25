#include <vector>
#include <csignal>
#include <unistd.h>
#include <sys/resource.h>

#include "messaging.hpp"
#include "common/i2c.h"
#include "common/timing.h"
#include "common/swaglog.h"

#include "sensors/constants.hpp"
#include "sensors/bmx055_accel.hpp"
#include "sensors/bmx055_gyro.hpp"
#include "sensors/bmx055_magn.hpp"

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

  // Sensor init
  std::vector<I2CSensor *> sensors;
  sensors.push_back(new BMX055_Accel(i2c_bus_imu));
  sensors.push_back(new BMX055_Gyro(i2c_bus_imu));
  sensors.push_back(new BMX055_Magn(i2c_bus_imu));


  for (I2CSensor * sensor : sensors){
    int err = sensor->init();
    if (err < 0){
      LOGE("Error initializing sensors");
      return -1;
    }
  }

  PubMaster pm({"sensorEvents"});

  while (!do_exit){
    uint64_t log_time = nanos_since_boot();

    capnp::MallocMessageBuilder msg;
    cereal::Event::Builder event = msg.initRoot<cereal::Event>();
    event.setLogMonoTime(log_time);

    int num_events = sensors.size();
    auto sensor_events = event.initSensorEvents(num_events);

    for (size_t i = 0; i < num_events; i++){
      auto event = sensor_events[i];
      sensors[i]->get_event(event);
    }

    pm.send("sensorEvents", msg);
    usleep(10 * 1000); // ~100 Hz
  }
  return 0;
}

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -13);
  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  return sensor_loop();
}
