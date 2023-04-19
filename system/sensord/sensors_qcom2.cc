#include <sys/resource.h>

#include <chrono>
#include <thread>
#include <vector>
#include <map>
#include <poll.h>
#include <linux/gpio.h>

#include "cereal/messaging/messaging.h"
#include "common/i2c.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/util.h"
#include "system/sensord/sensors/bmx055_accel.h"
#include "system/sensord/sensors/bmx055_gyro.h"
#include "system/sensord/sensors/bmx055_magn.h"
#include "system/sensord/sensors/bmx055_temp.h"
#include "system/sensord/sensors/constants.h"
#include "system/sensord/sensors/light_sensor.h"
#include "system/sensord/sensors/lsm6ds3_accel.h"
#include "system/sensord/sensors/lsm6ds3_gyro.h"
#include "system/sensord/sensors/lsm6ds3_temp.h"
#include "system/sensord/sensors/mmc5603nj_magn.h"
#include "system/sensord/sensors/sensor.h"

#define I2C_BUS_IMU 1

ExitHandler do_exit;
uint64_t init_ts = 0;

void interrupt_loop(std::vector<Sensor *>& sensors,
                    std::map<Sensor*, std::string>& sensor_service)
{
  PubMaster pm_int({"gyroscope", "accelerometer"});

  int fd = sensors[0]->gpio_fd;
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

    for (Sensor *sensor : sensors) {
      MessageBuilder msg;
      if (!sensor->get_event(msg, ts)) {
        continue;
      }

      if (!sensor->is_data_valid(init_ts, ts)) {
        continue;
      }

      pm_int.send(sensor_service[sensor].c_str(), msg);
    }
  }

  // poweroff sensors, disable interrupts
  for (Sensor *sensor : sensors) {
    sensor->shutdown();
  }
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

  LightSensor light("/sys/class/i2c-adapter/i2c-2/2-0038/iio:device1/in_intensity_both_raw");

  std::map<Sensor*, std::string> sensor_service = {
    {&bmx055_accel, "accelerometer2"},
    {&bmx055_gyro, "gyroscope2"},
    {&bmx055_magn, "magnetometer"},
    {&bmx055_temp, "temperatureSensor"},

    {&lsm6ds3_accel, "accelerometer"},
    {&lsm6ds3_gyro, "gyroscope"},
    {&lsm6ds3_temp, "temperatureSensor"},

    {&mmc5603nj_magn, "magnetometer"},
    {&light, "lightSensor"}
  };

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
  for (auto &[sensor, required] : sensors_init) {
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
        sensors.push_back(sensor);
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

  PubMaster pm_non_int({"gyroscope2", "accelerometer2", "temperatureSensor",
                        "lightSensor", "magnetometer"});
  init_ts = nanos_since_boot();

  // thread for reading events via interrupts
  std::vector<Sensor *> lsm_interrupt_sensors = {&lsm6ds3_accel, &lsm6ds3_gyro};
  std::thread lsm_interrupt_thread(&interrupt_loop, std::ref(lsm_interrupt_sensors),
                                   std::ref(sensor_service));

  // polling loop for non interrupt handled sensors
  while (!do_exit) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (Sensor *sensor : sensors) {
      MessageBuilder msg;
      if (!sensor->get_event(msg)) {
        continue;
      }

      if (!sensor->is_data_valid(init_ts, nanos_since_boot())) {
        continue;
      }

      pm_non_int.send(sensor_service[sensor].c_str(), msg);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10) - (end - begin));
  }

  for (Sensor *sensor : sensors) {
    sensor->shutdown();
  }

  lsm_interrupt_thread.join();
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
