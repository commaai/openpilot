#include <sys/resource.h>

#include <chrono>
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
std::mutex pm_mutex;

void publish_events(PubMaster &pm, const std::vector<Sensor *> &sensors, std::optional<int64_t> ts = {}) {
  MessageBuilder msg;
  auto sensor_events = msg.initEvent().initSensorEvents(sensors.size());
  for (int i = 0; i < sensors.size(); i++) {
    auto event = sensor_events[i];
    sensors[i]->get_event(event);
    if (ts) {
      event.setTimestamp(*ts);
    }
  }
  std::lock_guard<std::mutex> lock(pm_mutex);
  pm.send("sensorEvents", msg);
}

void interrupt_loop(int fd, std::vector<Sensor *>& sensors, PubMaster& pm) {
  struct pollfd fd_list[1] = {0};
  fd_list[0].fd = fd;
  fd_list[0].events = POLLIN | POLLPRI;

  const uint64_t offset = nanos_since_epoch() - nanos_since_boot();

  while (!do_exit) {
    int err = poll(fd_list, 1, 100);
    if (err == -1) {
      if (errno == EINTR) continue;
      LOGE("poll failed (%d - %d)", err, errno);
      break;
    }

    if ((fd_list[0].revents & (POLLIN | POLLPRI)) == 0) {
      LOGE("poll timed out");
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
    uint64_t ts = evdata[num_events - 1].timestamp - offset;

    std::vector<Sensor *> s;
    for (auto sensor : sensors) {
      if (sensor->triggered()) s.push_back(sensor);
    }
    if (s.empty()) continue;

    publish_events(pm, s, ts);
  }

  // poweroff sensors, disable interrupts
  for (Sensor *sensor : sensors) {
    sensor->shutdown();
  }
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

  LSM6DS3_Accel lsm6ds3_accel(i2c_bus_imu, GPIO_LSM_INT);
  LSM6DS3_Gyro lsm6ds3_gyro(i2c_bus_imu, GPIO_LSM_INT, true); // GPIO shared with accel
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
        delete i2c_bus_imu;
        return -1;
      }
    } else {
      if (sensor.first == &bmx055_magn || sensor.first == &mmc5603nj_magn) {
        has_magnetometer = true;
      }

      if (!sensor.first->has_interrupt_enabled()) {
        sensors.push_back(sensor.first);
      }
    }
  }

  if (!has_magnetometer) {
    LOGE("No magnetometer present");
    delete i2c_bus_imu;
    return -1;
  }

  PubMaster pm({"sensorEvents"});

  // thread for reading events via interrupts
  std::vector<Sensor *> lsm_interrupt_sensors = {&lsm6ds3_accel, &lsm6ds3_gyro};
  std::thread lsm_interrupt_thread(&interrupt_loop, lsm6ds3_accel.gpio_fd, std::ref(lsm_interrupt_sensors), std::ref(pm));

  // polling loop for non interrupt handled sensors
  while (!do_exit) {
    auto begin = std::chrono::steady_clock::now();
    publish_events(pm, sensors);
    auto end = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10) - (end - begin));
  }

  lsm_interrupt_thread.join();
  delete i2c_bus_imu;
  return 0;
}

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -18);
  return sensor_loop();
}
