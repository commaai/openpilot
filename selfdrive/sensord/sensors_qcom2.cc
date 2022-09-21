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

// filter first values (0.5sec) as those may contain inaccuracies
uint64_t init_ts = 0;
constexpr uint64_t init_delay = 500*1e6;

void interrupt_loop(const std::vector<Sensor *> &sensors, PubMaster &pm) {
  int fd = sensors[0]->gpio_fd;
  struct pollfd fd_list[1] = {0};
  fd_list[0].fd = fd;
  fd_list[0].events = POLLIN | POLLPRI;

  uint64_t offset = nanos_since_epoch() - nanos_since_boot();

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
    uint64_t ts = evdata[num_events - 1].timestamp - offset;

    MessageBuilder msg;
    auto orphanage = msg.getOrphanage();
    std::vector<capnp::Orphan<cereal::SensorEventData>> collected_events;
    collected_events.reserve(sensors.size());

    for (Sensor *sensor : sensors) {
      auto orphan = orphanage.newOrphan<cereal::SensorEventData>();
      auto event = orphan.get();
      if (!sensor->get_event(event)) {
        continue;
      }

      event.setTimestamp(ts);
      collected_events.push_back(kj::mv(orphan));
    }

    if (collected_events.size() == 0) {
      continue;
    }

    auto events = msg.initEvent().initSensorEvents(collected_events.size());
    for (int i = 0; i < collected_events.size(); ++i) {
      events.adoptWithCaveats(i, kj::mv(collected_events[i]));
    }

    if (ts - init_ts < init_delay) {
      continue;
    }

    std::lock_guard<std::mutex> lock(pm_mutex);
    pm.send("sensorEvents", msg);
  }
}

int sensor_loop(I2CBus *i2c_bus_imu) {
  std::pair<Sensor *, bool> all_sensors[] = {
    {new BMX055_Accel(i2c_bus_imu), false},
    {new BMX055_Gyro(i2c_bus_imu), false},
    {new BMX055_Magn(i2c_bus_imu), false},
    {new BMX055_Temp(i2c_bus_imu), false},
    {new LSM6DS3_Accel(i2c_bus_imu, GPIO_LSM_INT), true},
    {new LSM6DS3_Gyro(i2c_bus_imu, GPIO_LSM_INT, true), true},
    {new LSM6DS3_Temp(i2c_bus_imu), true},
    {new MMC5603NJ_Magn(i2c_bus_imu), false},
    {new LightSensor("/sys/class/i2c-adapter/i2c-2/2-0038/iio:device1/in_intensity_both_raw"), true},
  };

  std::vector<Sensor*> intr_sensors, non_intr_sensors;
  bool has_magnetometer = false;
  for (auto [sensor, required] : all_sensors) {
    int err = sensor->init();
    if (err < 0) {
      if (required) {
        LOGE("Error initializing sensors");
        return -1;
      }
    } else {
      auto &v = sensor->has_interrupt_enabled() ? intr_sensors : non_intr_sensors;
      v.push_back(sensor);
      has_magnetometer = has_magnetometer || dynamic_cast<BMX055_Magn *>(sensor) || dynamic_cast<MMC5603NJ_Magn *>(sensor);
    }
  }

  if (!has_magnetometer) {
    LOGE("No magnetometer present");
    return -1;
  }

  PubMaster pm({"sensorEvents"});
  init_ts = nanos_since_boot();

  // thread for reading events via interrupts
  std::thread interrupt_thread(&interrupt_loop, std::ref(intr_sensors), std::ref(pm));

  // polling loop for non interrupt handled sensors
  while (!do_exit) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const int num_events = non_intr_sensors.size();
    MessageBuilder msg;
    auto sensor_events = msg.initEvent().initSensorEvents(num_events);

    for (int i = 0; i < num_events; i++) {
      auto event = sensor_events[i];
      non_intr_sensors[i]->get_event(event);
    }

    if (nanos_since_boot() - init_ts < init_delay) {
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(pm_mutex);
      pm.send("sensorEvents", msg);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10) - (end - begin));
  }

  interrupt_thread.join();

  for (auto [sensor, _] : all_sensors) {
    sensor->shutdown();
    delete sensor;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -18);

  try {
    auto i2c_bus_imu = std::make_unique<I2CBus>(I2C_BUS_IMU);
    return sensor_loop(i2c_bus_imu.get());
  } catch (std::exception &e) {
    LOGE("I2CBus init failed");
    return -1;
  }
}
