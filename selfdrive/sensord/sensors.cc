#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <pthread.h>

#include <cutils/log.h>

#include <hardware/gps.h>
#include <hardware/sensors.h>
#include <utils/Timers.h>

#include <zmq.h>

#include <capnp/serialize.h>

#include "common/timing.h"

#include "cereal/gen/cpp/log.capnp.h"

// zmq output
static void *gps_publisher;

#define SENSOR_ACCELEROMETER 1
#define SENSOR_MAGNETOMETER 2
#define SENSOR_GYRO 4

void sensor_loop() {
  printf("*** sensor loop\n");
  struct sensors_poll_device_t* device;
  struct sensors_module_t* module;

  hw_get_module(SENSORS_HARDWARE_MODULE_ID, (hw_module_t const**)&module);
  sensors_open(&module->common, &device);

  // required
  struct sensor_t const* list;
  int count = module->get_sensors_list(module, &list);
  printf("%d sensors found\n", count);

  device->activate(device, SENSOR_ACCELEROMETER, 0);
  device->activate(device, SENSOR_MAGNETOMETER, 0);
  device->activate(device, SENSOR_GYRO, 0);

  device->activate(device, SENSOR_ACCELEROMETER, 1);
  device->activate(device, SENSOR_MAGNETOMETER, 1);
  device->activate(device, SENSOR_GYRO, 1);

  device->setDelay(device, SENSOR_ACCELEROMETER, ms2ns(10));
  device->setDelay(device, SENSOR_GYRO, ms2ns(10));
  device->setDelay(device, SENSOR_MAGNETOMETER, ms2ns(100));

  static const size_t numEvents = 16;
  sensors_event_t buffer[numEvents];

  // zmq output
  void *context = zmq_ctx_new();
  void *publisher = zmq_socket(context, ZMQ_PUB);
  zmq_bind(publisher, "tcp://*:8003");

  while (1) {
    int n = device->poll(device, buffer, numEvents);
    if (n == 0) continue;
    if (n < 0) {
      printf("sensor_loop poll failed: %d\n", n);
      continue;
    }

    uint64_t log_time = nanos_since_boot();

    capnp::MallocMessageBuilder msg;
    cereal::Event::Builder event = msg.initRoot<cereal::Event>();
    event.setLogMonoTime(log_time);

    auto sensorEvents = event.initSensorEvents(n);

    for (int i = 0; i < n; i++) {

      const sensors_event_t& data = buffer[i];

      sensorEvents[i].setVersion(data.version);
      sensorEvents[i].setSensor(data.sensor);
      sensorEvents[i].setType(data.type);
      sensorEvents[i].setTimestamp(data.timestamp);

      switch (data.type) {
      case SENSOR_TYPE_ACCELEROMETER: {
        auto svec = sensorEvents[i].initAcceleration();
        kj::ArrayPtr<const float> vs(&data.acceleration.v[0], 3);
        svec.setV(vs);
        svec.setStatus(data.acceleration.status);
        break;
      }
      case SENSOR_TYPE_MAGNETIC_FIELD: {
        auto svec = sensorEvents[i].initMagnetic();
        kj::ArrayPtr<const float> vs(&data.magnetic.v[0], 3);
        svec.setV(vs);
        svec.setStatus(data.magnetic.status);
        break;
      }
      case SENSOR_TYPE_GYROSCOPE: {
        auto svec = sensorEvents[i].initGyro();
        kj::ArrayPtr<const float> vs(&data.gyro.v[0], 3);
        svec.setV(vs);
        svec.setStatus(data.gyro.status);
        break;
      }
      default:
        continue;
      }
    }

    auto words = capnp::messageToFlatArray(msg);
    auto bytes = words.asBytes();
    // printf("send %d\n", bytes.size());
    zmq_send(publisher, bytes.begin(), bytes.size(), 0);

  }
}

static const GpsInterface* gGpsInterface = NULL;
static const AGpsInterface* gAGpsInterface = NULL;
static const GpsMeasurementInterface* gGpsMeasurementInterface = NULL;

static void nmea_callback(GpsUtcTime timestamp, const char* nmea, int length) {

  uint64_t log_time = nanos_since_boot();
  uint64_t log_time_wall = nanos_since_epoch();

  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(log_time);

  auto nmeaData = event.initGpsNMEA();
  nmeaData.setTimestamp(timestamp);
  nmeaData.setLocalWallTime(log_time_wall);
  nmeaData.setNmea(nmea);

  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  // printf("gps send %d\n", bytes.size());
  zmq_send(gps_publisher, bytes.begin(), bytes.size(), 0);
}

static pthread_t create_thread_callback(const char* name, void (*start)(void *), void* arg) {
  printf("creating thread: %s\n", name);
  pthread_t thread;
  pthread_attr_t attr;
  int err;

  err = pthread_attr_init(&attr);
  err = pthread_create(&thread, &attr, (void*(*)(void*))start, arg);

  return thread;
}

static GpsCallbacks gps_callbacks = {
  sizeof(GpsCallbacks),
  NULL,
  NULL,
  NULL,
  nmea_callback,
  NULL,
  NULL,
  NULL,
  create_thread_callback,
};

static void agps_status_cb(AGpsStatus *status) {
  switch (status->status) {
    case GPS_REQUEST_AGPS_DATA_CONN:
      fprintf(stdout, "*** data_conn_open\n");
      gAGpsInterface->data_conn_open("internet");
      break;
    case GPS_RELEASE_AGPS_DATA_CONN:
      fprintf(stdout, "*** data_conn_closed\n");
      gAGpsInterface->data_conn_closed();
      break;
  }
}

static AGpsCallbacks agps_callbacks = {
  agps_status_cb,
  create_thread_callback,
};



static void gps_init() {
  printf("*** init GPS\n");
  hw_module_t* module;
  hw_get_module(GPS_HARDWARE_MODULE_ID, (hw_module_t const**)&module);

  hw_device_t* device;
  module->methods->open(module, GPS_HARDWARE_MODULE_ID, &device);

  // ** get gps interface **
  gps_device_t* gps_device = (gps_device_t *)device;
  gGpsInterface = gps_device->get_gps_interface(gps_device);
  gAGpsInterface = (const AGpsInterface*)gGpsInterface->get_extension(AGPS_INTERFACE);



  gGpsInterface->init(&gps_callbacks);
  gAGpsInterface->init(&agps_callbacks);
  gAGpsInterface->set_server(AGPS_TYPE_SUPL, "supl.google.com", 7276);

  gGpsInterface->delete_aiding_data(GPS_DELETE_ALL);
  gGpsInterface->start();
  gGpsInterface->set_position_mode(GPS_POSITION_MODE_MS_BASED,
                                   GPS_POSITION_RECURRENCE_PERIODIC,
                                   1000, 0, 0);
  void *gps_context = zmq_ctx_new();
  gps_publisher = zmq_socket(gps_context, ZMQ_PUB);
  zmq_bind(gps_publisher, "tcp://*:8004");
}

int main(int argc, char *argv[]) {
  gps_init();
  sensor_loop();
}

