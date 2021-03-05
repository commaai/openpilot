#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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
#include "common/util.h"
#include "common/swaglog.h"

// ACCELEROMETER_UNCALIBRATED is only in Android O
// https://developer.android.com/reference/android/hardware/Sensor.html#STRING_TYPE_ACCELEROMETER_UNCALIBRATED

ExitHandler do_exit;
volatile sig_atomic_t re_init_sensors = 0;

typedef struct SensorState {
  int handle;
  bool offroad;
  nsecs_t delay;
} SensorState;

static std::map<int, SensorState> sensors = {
    {SENSOR_TYPE_GYROSCOPE, {.delay = ms2ns(10)}},
    {SENSOR_TYPE_MAGNETIC_FIELD, {.delay = ms2ns(100)}},
    {SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED, {.delay = ms2ns(100)}},
    {SENSOR_TYPE_PROXIMITY, {.delay = ms2ns(100)}},
    {SENSOR_TYPE_GYROSCOPE_UNCALIBRATED, {.delay = ms2ns(10), .offroad = true}},
    {SENSOR_TYPE_ACCELEROMETER, {.delay = ms2ns(10), .offroad = true}},
    {SENSOR_TYPE_LIGHT, {.delay = ms2ns(100), .offroad = true}}};

namespace {

void sigpipe_handler(int sig) {
  LOGE("SIGPIPE received");
  re_init_sensors = true;
}

void sensor_loop() {
  LOG("*** sensor loop");

  SubMaster sm({"deviceState"});
  PubMaster pm({"sensorEvents"});

  uint64_t frame = 0;
  bool low_power_mode = false;

  while (!do_exit) {
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

    // init sensors
    for (int i = 0; i < count; i++) {
      auto &s = list[i];
      LOGW("sensor %4d: %4d %60s  %d-%ld us", s.type, s.handle, s.name, s.minDelay, s.maxDelay);
      // LOG("sensor %4d: %4d %60s  %d-%ld us", i, list[i].handle, list[i].name, list[i].minDelay, list[i].maxDelay);
      device->activate(device, s.handle, 0);
      auto it = sensors.find(s.type);
      if(it != sensors.end() && it->second.handle == 0) {
        it->second.handle = s.handle;
        device->activate(device, it->second.handle, 1);
        device->setDelay(device, it->second.handle, it->second.delay);
      }
    }

    sensors_event_t buffer[sensors.size()];

    while (!do_exit) {
      int n = device->poll(device, buffer, sensors.size());
      if (n <= 0) {
        LOG("sensor_loop poll failed: %d", n);
        continue;
      }

      MessageBuilder msg;
      auto sensor_events = msg.initEvent().initSensorEvents(n);
      for (int i = 0; i < n; i++) {
        const sensors_event_t& data = buffer[i];

        auto log_event = sensor_events[i];
        log_event.setSource(cereal::SensorEventData::SensorSource::ANDROID);
        log_event.setVersion(data.version);
        log_event.setSensor(data.sensor);
        log_event.setType(data.type);
        log_event.setTimestamp(data.timestamp);

        switch (data.type) {
        case SENSOR_TYPE_ACCELEROMETER: {
          auto svec = log_event.initAcceleration();
          svec.setV(data.acceleration.v);
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
          svec.setV(data.magnetic.v);
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
          svec.setV(data.gyro.v);
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
        default:
          assert(0);
        }
      }

      pm.send("sensorEvents", msg);

      if (re_init_sensors){
        LOGE("Resetting sensors");
        re_init_sensors = false;
        break;
      }

      // Check whether to go into low power mode at 5Hz
      if (frame % 20 == 0 && sm.update(0) > 0) {
        bool offroad = !sm["deviceState"].getDeviceState().getStarted();
        if (low_power_mode != offroad) {
          for (auto &[type, s] : sensors) {
            device->activate(device, s.handle, 0);
            if (!offroad || s.offroad) {
              device->activate(device, s.handle, 1);
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
  signal(SIGPIPE, (sighandler_t)sigpipe_handler);

  sensor_loop();

  return 0;
}
