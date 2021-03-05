#include <assert.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <map>
#include <hardware/sensors.h>
#include <utils/Timers.h>

#include "messaging.hpp"
#include "common/util.h"
#include "common/swaglog.h"

// ACCELEROMETER_UNCALIBRATED is only in Android O
// https://developer.android.com/reference/android/hardware/Sensor.html#STRING_TYPE_ACCELEROMETER_UNCALIBRATED

namespace {

ExitHandler do_exit;
volatile sig_atomic_t re_init_sensors = 0;
typedef cereal::SensorEventData::Builder SensorBuilder;

void build_accel(SensorBuilder& log_event, const sensors_event_t& data) {
  auto svec = log_event.initAcceleration();
  svec.setV(data.acceleration.v);
  svec.setStatus(data.acceleration.status);
}

void build_magnetic_uncalib(SensorBuilder& log_event, const sensors_event_t& data) {
  auto svec = log_event.initMagneticUncalibrated();
  // assuming the uncalib and bias floats are contiguous in memory
  kj::ArrayPtr<const float> vs(&data.uncalibrated_magnetic.uncalib[0], 6);
  svec.setV(vs);
}

void build_magnetic(SensorBuilder& log_event, const sensors_event_t& data) {
  auto svec = log_event.initMagnetic();
  svec.setV(data.magnetic.v);
  svec.setStatus(data.magnetic.status);
}

void build_gyro_uncalib(SensorBuilder& log_event, const sensors_event_t& data) {
  auto svec = log_event.initGyroUncalibrated();
  // assuming the uncalib and bias floats are contiguous in memory
  kj::ArrayPtr<const float> vs(&data.uncalibrated_gyro.uncalib[0], 6);
  svec.setV(vs);
}

void build_gyro(SensorBuilder& log_event, const sensors_event_t& data) {
  auto svec = log_event.initGyro();
  svec.setV(data.gyro.v);
  svec.setStatus(data.gyro.status);
}

void build_proximity(SensorBuilder& log_event, const sensors_event_t& data) {
  log_event.setProximity(data.distance);
}

void build_light(SensorBuilder& log_event, const sensors_event_t& data) {
  log_event.setLight(data.light);
}

typedef struct SensorState {
  int handle;
  bool offroad;
  nsecs_t delay;
  void (*build)(SensorBuilder& log_event, const sensors_event_t& data);
} SensorState;

static std::map<int, SensorState> sensors = {
  {SENSOR_TYPE_GYROSCOPE, {.delay = ms2ns(10), .build = build_gyro}},
  {SENSOR_TYPE_MAGNETIC_FIELD, {.delay = ms2ns(100), .build = build_magnetic}},
  {SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED, {.delay = ms2ns(100), .build = build_magnetic_uncalib}},
  {SENSOR_TYPE_PROXIMITY, {.delay = ms2ns(100), .build = build_proximity}},
  {SENSOR_TYPE_GYROSCOPE_UNCALIBRATED, {.delay = ms2ns(10), .build = build_gyro_uncalib, .offroad = true}},
  {SENSOR_TYPE_ACCELEROMETER, {.delay = ms2ns(10), .build = build_accel, .offroad = true}},
  {SENSOR_TYPE_LIGHT, {.delay = ms2ns(100), .build = build_light, .offroad = true}}};

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

    int err = hw_get_module(SENSORS_HARDWARE_MODULE_ID, (hw_module_t const**)&module);
    assert(err == 0);
    err = sensors_open(&module->common, &device);
    assert(err == 0);

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
      LOG("sensor %4d: %4d %60s  %d-%ld us", i, s.handle, s.name, s.minDelay, s.maxDelay);
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

        sensors.at(data.type).build(log_event, data);
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
