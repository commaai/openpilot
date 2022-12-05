#include <sys/resource.h>

#include <chrono>
#include <set>
#include <thread>
#include <vector>
#include <poll.h>
#include <linux/gpio.h>

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
uint64_t init_ts = 0;

struct SensorState {
  Sensor *sensor = nullptr;
  const char *name = nullptr;
  bool required = false;
};

std::vector<const char*> get_pubsock_names(std::vector<SensorState>& sensors) {
  struct cstrless {
    bool operator()(const char *a, const char *b) const { return strcmp(a, b) < 0;}
  };
  std::set<const char*, cstrless> names;
  for (auto &s : sensors) names.insert(s.name);
  return std::vector(names.begin(), names.end());
}

void interrupt_loop(std::vector<SensorState>& sensors) {
  PubMaster pm_int(get_pubsock_names(sensors));

  int fd = sensors[0].sensor->gpio_fd;
  struct pollfd fd_list[1] = {0};
  fd_list[0].fd = fd;
  fd_list[0].events = POLLIN | POLLPRI;

  while (!do_exit) {
    int err = poll(fd_list, 1, 100);
    if (err == -1) {
      if (errno == EINTR) {
        continue;
      }
      return;
    } else if (err == 0) {
      LOGE("poll timed out");
      continue;
    }

    if ((fd_list[0].revents & (POLLIN | POLLPRI)) == 0) {
      LOGE("no poll events set");
      continue;
    }

    // Read all events
    struct gpioevent_data evdata[16];
    err = read(fd, evdata, sizeof(evdata));
    if (err < 0 || err % sizeof(*evdata) != 0) {
      LOGE("error reading event data %d", err);
      continue;
    }

    int num_events = err / sizeof(*evdata);
    uint64_t offset = nanos_since_epoch() - nanos_since_boot();
    uint64_t ts = evdata[num_events - 1].timestamp - offset;

    for (auto &s : sensors) {
      MessageBuilder msg;
      if (s.sensor->get_event(msg, ts) && s.sensor->is_data_valid(init_ts, ts)) {
        pm_int.send(s.name, msg);
      }
    }
  }

  // poweroff sensors, disable interrupts
  for (auto &s : sensors) {
    s.sensor->shutdown();
  }
}

int sensor_loop(I2CBus *i2c_bus_imu) {
  std::vector<SensorState> all_sensors = {
    {new BMX055_Accel(i2c_bus_imu), "accelerometer2"},
    {new BMX055_Gyro(i2c_bus_imu), "gyroscope2"},
    {new BMX055_Magn(i2c_bus_imu), "magnetometer"},
    {new BMX055_Temp(i2c_bus_imu), "temperatureSensor"},
    {new LSM6DS3_Accel(i2c_bus_imu, GPIO_LSM_INT), "accelerometer", true},
    {new LSM6DS3_Gyro(i2c_bus_imu, GPIO_LSM_INT, true), "gyroscope", true},
    {new LSM6DS3_Temp(i2c_bus_imu), "temperatureSensor", true},
    {new MMC5603NJ_Magn(i2c_bus_imu), "magnetometer"},
    {new LightSensor("/sys/class/i2c-adapter/i2c-2/2-0038/iio:device1/in_intensity_both_raw"), "lightSensor", true},
  };

  bool has_magnetometer = false;
  // Initialize sensors
  std::vector<SensorState> sensors, lsm_interrupt_sensors;
  for (auto &s : all_sensors) {
    int err = s.sensor->init();
    if (err >= 0) {
      has_magnetometer |= (strcmp(s.name, "magnetometer") == 0);
      (s.sensor->has_interrupt_enabled() ? lsm_interrupt_sensors : sensors).push_back(s);
    } else if (s.required) {
      LOGE("Error initializing sensors");
      return -1;
    }
  }

  if (!has_magnetometer) {
    LOGE("No magnetometer present");
    return -1;
  }

  // increase interrupt quality by pinning interrupt and process to core 1
  setpriority(PRIO_PROCESS, 0, -18);
  util::set_core_affinity({1});
  std::system("sudo su -c 'echo 1 > /proc/irq/336/smp_affinity_list'");

  init_ts = nanos_since_boot();

  // thread for reading events via interrupts
  std::thread lsm_interrupt_thread(&interrupt_loop, std::ref(lsm_interrupt_sensors));

  // polling loop for non interrupt handled sensors
  PubMaster pm_non_int(get_pubsock_names(sensors));
  while (!do_exit) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (auto &s : sensors) {
      MessageBuilder msg;
      if (s.sensor->get_event(msg) && s.sensor->is_data_valid(init_ts, nanos_since_boot())) {
        pm_non_int.send(s.name, msg);
      }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10) - (end - begin));
  }

  for (auto &s : sensors) {
    s.sensor->shutdown();
  }

  lsm_interrupt_thread.join();
  for (auto &s : all_sensors) {
    delete s.sensor;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  try {
    auto i2c_bus_imu = std::make_unique<I2CBus>(I2C_BUS_IMU);
    return sensor_loop(i2c_bus_imu.get());
  } catch (std::exception &e) {
    LOGE("I2CBus init failed");
    return -1;
  }
}
