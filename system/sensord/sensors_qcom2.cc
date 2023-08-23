#include <sys/resource.h>

#include <chrono>
#include <thread>
#include <vector>
#include <map>
#include <poll.h>
#include <linux/gpio.h>

#include "cereal/messaging/messaging.h"
#include "common/i2c.h"
#include "common/ratekeeper.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "system/sensord/sensors/bmx055_accel.h"
#include "system/sensord/sensors/bmx055_gyro.h"
#include "system/sensord/sensors/bmx055_magn.h"
#include "system/sensord/sensors/bmx055_temp.h"
#include "system/sensord/sensors/constants.h"
#include "system/sensord/sensors/lsm6ds3_accel.h"
#include "system/sensord/sensors/lsm6ds3_gyro.h"
#include "system/sensord/sensors/lsm6ds3_temp.h"
#include "system/sensord/sensors/mmc5603nj_magn.h"

#define I2C_BUS_IMU 1

ExitHandler do_exit;

void interrupt_loop(std::vector<std::tuple<Sensor *, std::string, bool, int>> sensors) {
  PubMaster pm({"gyroscope", "accelerometer"});

  int fd = -1;
  for (auto &[sensor, msg_name, required, polling_freq] : sensors) {
    if (sensor->has_interrupt_enabled()) {
      fd = sensor->gpio_fd;
      break;
    }
  }

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

    for (auto &[sensor, msg_name, required, polling_freq] : sensors) {
      if (!sensor->has_interrupt_enabled()) {
        continue;
      }

      MessageBuilder msg;
      if (!sensor->get_event(msg, ts)) {
        continue;
      }

      if (!sensor->is_data_valid(ts)) {
        continue;
      }

      pm.send(msg_name.c_str(), msg);
    }
  }

  // poweroff sensors, disable interrupts
  for (auto &[sensor, msg_name, required, polling_freq] : sensors) {
    if (sensor->has_interrupt_enabled()) {
      sensor->shutdown();
    }
  }
}

void polling_loop(Sensor *sensor, std::string msg_name, int frequency) {
  PubMaster pm({msg_name.c_str()});
  RateKeeper rk(msg_name, frequency);
  while (!do_exit) {
    MessageBuilder msg;
    if (sensor->get_event(msg) && sensor->is_data_valid(nanos_since_boot())) {
      pm.send(msg_name.c_str(), msg);
    }
    rk.keepTime();
  }

  sensor->shutdown();
}

int sensor_loop(I2CBus *i2c_bus_imu) {
  BMX055_Accel bmx055_accel(i2c_bus_imu);
  BMX055_Gyro bmx055_gyro(i2c_bus_imu);
  BMX055_Magn bmx055_magn(i2c_bus_imu);
  BMX055_Temp bmx055_temp(i2c_bus_imu);

  LSM6DS3_Accel lsm6ds3_accel(i2c_bus_imu, GPIO_LSM_INT);
  LSM6DS3_Gyro lsm6ds3_gyro(i2c_bus_imu, GPIO_LSM_INT, true); // GPIO shared with accel
  LSM6DS3_Temp lsm6ds3_temp(i2c_bus_imu);

  MMC5603NJ_Magn mmc5603nj_magn(i2c_bus_imu);

  // Sensor init
  std::vector<std::tuple<Sensor *, std::string, bool, int>> sensors_init; // Sensor, required
  sensors_init.push_back({&bmx055_accel, "accelerometer2", false, 100});
  sensors_init.push_back({&bmx055_gyro, "gyroscope2", false, 100});
  sensors_init.push_back({&bmx055_magn, "magnetometer", false, 25});
  sensors_init.push_back({&bmx055_temp, "temperatureSensor2", false, 2});

  sensors_init.push_back({&lsm6ds3_accel, "accelerometer", true, 100});
  sensors_init.push_back({&lsm6ds3_gyro, "gyroscope", true, 100});
  sensors_init.push_back({&lsm6ds3_temp, "temperatureSensor", true, 2});

  sensors_init.push_back({&mmc5603nj_magn, "magnetometer", false, 25});

  bool has_magnetometer = false;

  // Initialize sensors
  std::vector<std::thread> sensor_threads;
  for (auto &[sensor, msg_name, required, polling_freq] : sensors_init) {
    int err = sensor->init();
    if (err < 0) {
      if (required) {
        LOGE("Error initializing sensors");
        return -1;
      }
    } else {
      if (sensor == &bmx055_magn || sensor == &mmc5603nj_magn) {
        has_magnetometer = true;
      }
      if (!sensor->has_interrupt_enabled()) {
        sensor_threads.emplace_back(polling_loop, sensor, msg_name, polling_freq);
      }
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

  // thread for reading events via interrupts
  std::thread lsm_interrupt_thread(&interrupt_loop, std::ref(sensors_init));

  lsm_interrupt_thread.join();
  for (auto &thread : sensor_threads) {
    thread.join();
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
