#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/resource.h>

#include <map>
#include <set>

#include <cutils/log.h>
#include <hardware/sensors.h>
#include <utils/Timers.h>

#include "messaging.hpp"
#include "common/timing.h"
#include "common/swaglog.h"

// ACCELEROMETER_UNCALIBRATED is only in Android O
// https://developer.android.com/reference/android/hardware/Sensor.html#STRING_TYPE_ACCELEROMETER_UNCALIBRATED

#define SENSOR_ACCELEROMETER 1
#define SENSOR_MAGNETOMETER 2
#define SENSOR_GYRO 4
#define SENSOR_MAGNETOMETER_UNCALIBRATED 3
#define SENSOR_GYRO_UNCALIBRATED 5
#define SENSOR_PROXIMITY 6
#define SENSOR_LIGHT 7

volatile sig_atomic_t do_exit = 0;
volatile sig_atomic_t re_init_sensors = 0;

namespace {

void set_do_exit(int sig) {
  do_exit = 1;
}

void sigpipe_handler(int sig) {
  LOGE("SIGPIPE received");
  re_init_sensors = true;
}

void sensor_loop() {
  LOG("*** sensor loop");

  uint64_t frame = 0;
  bool low_power_mode = false;

  while (!do_exit) {
    SubMaster sm({"thermal"});
    PubMaster pm({"sensorEvents"});

    struct sensors_poll_device_t* device;
    struct sensors_module_t* module;

    hw_get_module(SENSORS_HARDWARE_MODULE_ID, (hw_module_t const**)&module);
    sensors_open(&module->common, &device);

    // required
    struct sensor_t const* list;
    int count = module->get_sensors_list(module, &list);
    LOG("%d sensors found", count);

    if (getenv("SENSOR_TEST")) {
      exit(count);
    }

    for (int i = 0; i < count; i++) {
      LOGD("sensor %4d: %4d %60s  %d-%ld us", i, list[i].handle, list[i].name, list[i].minDelay, list[i].maxDelay);
    }

    std::set<int> sensor_types = {
      SENSOR_TYPE_ACCELEROMETER,
      SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED,
      SENSOR_TYPE_MAGNETIC_FIELD,
      SENSOR_TYPE_GYROSCOPE_UNCALIBRATED,
      SENSOR_TYPE_GYROSCOPE,
      SENSOR_TYPE_PROXIMITY,
      SENSOR_TYPE_LIGHT,
    };

    std::map<int, int64_t> sensors = {
      {SENSOR_GYRO_UNCALIBRATED, ms2ns(10)},
      {SENSOR_MAGNETOMETER_UNCALIBRATED, ms2ns(100)},
      {SENSOR_ACCELEROMETER, ms2ns(10)},
      {SENSOR_GYRO, ms2ns(10)},
      {SENSOR_MAGNETOMETER, ms2ns(100)},
      {SENSOR_PROXIMITY, ms2ns(100)},
      {SENSOR_LIGHT, ms2ns(100)}
    };

    // sensors needed while offroad
    std::set<int> offroad_sensors = {
      SENSOR_LIGHT,
      SENSOR_ACCELEROMETER,
      SENSOR_GYRO_UNCALIBRATED,
    };

    // init all the sensors
    for (auto &s : sensors) {
      device->activate(device, s.first, 0);
      device->activate(device, s.first, 1);
      device->setDelay(device, s.first, s.second);
    }

    // TODO: why is this 16?
    static const size_t numEvents = 16;
    sensors_event_t buffer[numEvents];

    while (!do_exit) {
      int n = device->poll(device, buffer, numEvents);
      if (n == 0) continue;
      if (n < 0) {
        LOG("sensor_loop poll failed: %d", n);
        continue;
      }

      int log_events = 0;
      for (int i=0; i < n; i++) {
        if (sensor_types.find(buffer[i].type) != sensor_types.end()) {
          log_events++;
        }
      }

      MessageBuilder msg;
      auto sensor_events = msg.initEvent().initSensorEvents(log_events);

      int log_i = 0;
      for (int i = 0; i < n; i++) {

        const sensors_event_t& data = buffer[i];

        if (sensor_types.find(data.type) == sensor_types.end()) {
          continue;
        }

        auto log_event = sensor_events[log_i];
        log_event.setSource(cereal::SensorEventData::SensorSource::ANDROID);
        log_event.setVersion(data.version);
        log_event.setSensor(data.sensor);
        log_event.setType(data.type);
        log_event.setTimestamp(data.timestamp);

        switch (data.type) {
        case SENSOR_TYPE_ACCELEROMETER: {
          auto svec = log_event.initAcceleration();
          kj::ArrayPtr<const float> vs(&data.acceleration.v[0], 3);
          svec.setV(vs);
          svec.setStatus(data.acceleration.status);
          break;
        }
        case SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED: {
          auto svec = log_event.initMagneticUncalibrated();
          // assuming the uncalib and bias floats are contiguous in memory
          kj::ArrayPtr<const float> vs(&data.uncalibrated_magnetic.uncalib[0], 6);
          svec.setV(vs);
          break;
        }
        case SENSOR_TYPE_MAGNETIC_FIELD: {
          auto svec = log_event.initMagnetic();
          kj::ArrayPtr<const float> vs(&data.magnetic.v[0], 3);
          svec.setV(vs);
          svec.setStatus(data.magnetic.status);
          break;
        }
        case SENSOR_TYPE_GYROSCOPE_UNCALIBRATED: {
          auto svec = log_event.initGyroUncalibrated();
          // assuming the uncalib and bias floats are contiguous in memory
          kj::ArrayPtr<const float> vs(&data.uncalibrated_gyro.uncalib[0], 6);
          svec.setV(vs);
          break;
        }
        case SENSOR_TYPE_GYROSCOPE: {
          auto svec = log_event.initGyro();
          kj::ArrayPtr<const float> vs(&data.gyro.v[0], 3);
          svec.setV(vs);
          svec.setStatus(data.gyro.status);
          break;
        }
        case SENSOR_TYPE_PROXIMITY: {
          log_event.setProximity(data.distance);
          break;
        }
        case SENSOR_TYPE_LIGHT:
          log_event.setLight(data.light);
          break;
        }

        log_i++;
      }

      pm.send("sensorEvents", msg);

      if (re_init_sensors){
        LOGE("Resetting sensors");
        re_init_sensors = false;
        break;
      }

      // Check whether to go into low power mode at 5Hz
      if (frame % 20 == 0 && sm.update(0) > 0) {
        bool offroad = !sm["thermal"].getThermal().getStarted();
        if (low_power_mode != offroad) {
          for (auto &s : sensors) {
            device->activate(device, s.first, 0);
            if (!offroad || offroad_sensors.find(s.first) != offroad_sensors.end()) {
              device->activate(device, s.first, 1);
            }
          }
          low_power_mode = offroad;
        }
      }

      frame++;
    }
    sensors_close(device);
  }
}

}// Namespace end

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -13);
  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);
  signal(SIGPIPE, (sighandler_t)sigpipe_handler);

  sensor_loop();

  return 0;
}
