#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/timerfd.h>
#include <sys/resource.h>

#include <pthread.h>

#include <cutils/log.h>

#include <hardware/gps.h>
#include <utils/Timers.h>

#include <zmq.h>

#include <capnp/serialize.h>

#include "common/timing.h"
#include "common/swaglog.h"

#include "cereal/gen/cpp/log.capnp.h"

#include "rawgps.h"

volatile sig_atomic_t do_exit = 0;

namespace {

pthread_t clock_thread_handle;

// zmq output
void *gps_context;
void *gps_publisher;
void *gps_location_publisher;

const GpsInterface* gGpsInterface = NULL;
const AGpsInterface* gAGpsInterface = NULL;

void set_do_exit(int sig) {
  do_exit = 1;
}

void nmea_callback(GpsUtcTime timestamp, const char* nmea, int length) {

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

void location_callback(GpsLocation* location) {
  //printf("got location callback\n");
  uint64_t log_time = nanos_since_boot();

  capnp::MallocMessageBuilder msg;
  cereal::Event::Builder event = msg.initRoot<cereal::Event>();
  event.setLogMonoTime(log_time);

  auto locationData = event.initGpsLocation();
  locationData.setFlags(location->flags);
  locationData.setLatitude(location->latitude);
  locationData.setLongitude(location->longitude);
  locationData.setAltitude(location->altitude);
  locationData.setSpeed(location->speed);
  locationData.setBearing(location->bearing);
  locationData.setAccuracy(location->accuracy);
  locationData.setTimestamp(location->timestamp);
  locationData.setSource(cereal::GpsLocationData::SensorSource::ANDROID);

  auto words = capnp::messageToFlatArray(msg);
  auto bytes = words.asBytes();
  zmq_send(gps_location_publisher, bytes.begin(), bytes.size(), 0);
}

pthread_t create_thread_callback(const char* name, void (*start)(void *), void* arg) {
  LOG("creating thread: %s", name);
  pthread_t thread;
  pthread_attr_t attr;
  int err;

  err = pthread_attr_init(&attr);
  err = pthread_create(&thread, &attr, (void*(*)(void*))start, arg);

  return thread;
}

GpsCallbacks gps_callbacks = {
  sizeof(GpsCallbacks),
  location_callback,
  NULL,
  NULL,
  nmea_callback,
  NULL,
  NULL,
  NULL,
  create_thread_callback,
};

void agps_status_cb(AGpsStatus *status) {
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

AGpsCallbacks agps_callbacks = {
  agps_status_cb,
  create_thread_callback,
};



void gps_init() {
  LOG("*** init GPS");
  hw_module_t* module = NULL;
  hw_get_module(GPS_HARDWARE_MODULE_ID, (hw_module_t const**)&module);
  assert(module);

  static hw_device_t* device = NULL;
  module->methods->open(module, GPS_HARDWARE_MODULE_ID, &device);
  assert(device);

  // ** get gps interface **
  gps_device_t* gps_device = (gps_device_t *)device;
  gGpsInterface = gps_device->get_gps_interface(gps_device);
  assert(gGpsInterface);

  gAGpsInterface = (const AGpsInterface*)gGpsInterface->get_extension(AGPS_INTERFACE);
  assert(gAGpsInterface);


  gGpsInterface->init(&gps_callbacks);
  gAGpsInterface->init(&agps_callbacks);
  gAGpsInterface->set_server(AGPS_TYPE_SUPL, "supl.google.com", 7276);

  // gGpsInterface->delete_aiding_data(GPS_DELETE_ALL);
  gGpsInterface->start();
  gGpsInterface->set_position_mode(GPS_POSITION_MODE_MS_BASED,
                                   GPS_POSITION_RECURRENCE_PERIODIC,
                                   100, 0, 0);

  gps_context = zmq_ctx_new();
  gps_publisher = zmq_socket(gps_context, ZMQ_PUB);
  zmq_bind(gps_publisher, "tcp://*:8004");

  gps_location_publisher = zmq_socket(gps_context, ZMQ_PUB);
  zmq_bind(gps_location_publisher, "tcp://*:8026");
}

void gps_destroy() {
  gGpsInterface->stop();
  gGpsInterface->cleanup();
}


int64_t arm_cntpct() {
  int64_t v;
  asm volatile("mrs %0, cntpct_el0" : "=r"(v));
  return v;
}

// TODO: move this out of here
void* clock_thread(void* args) {
  int err = 0;

  void* clock_publisher = zmq_socket(gps_context, ZMQ_PUB);
  zmq_bind(clock_publisher, "tcp://*:8034");

  int timerfd = timerfd_create(CLOCK_BOOTTIME, 0);
  assert(timerfd >= 0);

  struct itimerspec spec = {0};
  spec.it_interval.tv_sec = 1;
  spec.it_interval.tv_nsec = 0;
  spec.it_value.tv_sec = 1;
  spec.it_value.tv_nsec = 0;

  err = timerfd_settime(timerfd, 0, &spec, 0);
  assert(err == 0);

  uint64_t expirations = 0;
  while ((err = read(timerfd, &expirations, sizeof(expirations)))) {
    if (err < 0) break;

    if (do_exit) break;

    uint64_t boottime = nanos_since_boot();
    uint64_t monotonic = nanos_monotonic();
    uint64_t monotonic_raw = nanos_monotonic_raw();
    uint64_t wall_time = nanos_since_epoch();

    uint64_t modem_uptime_v = arm_cntpct() / 19200ULL; // 19.2 mhz clock

    capnp::MallocMessageBuilder msg;
    cereal::Event::Builder event = msg.initRoot<cereal::Event>();
    event.setLogMonoTime(boottime);
    auto clocks = event.initClocks();

    clocks.setBootTimeNanos(boottime);
    clocks.setMonotonicNanos(monotonic);
    clocks.setMonotonicRawNanos(monotonic_raw);
    clocks.setWallTimeNanos(wall_time);
    clocks.setModemUptimeMillis(modem_uptime_v);

    auto words = capnp::messageToFlatArray(msg);
    auto bytes = words.asBytes();
    zmq_send(clock_publisher, bytes.begin(), bytes.size(), 0);
  }

  close(timerfd);
  zmq_close(clock_publisher);

  return NULL;
}


}

int main() {
  int err = 0;
  setpriority(PRIO_PROCESS, 0, -13);

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  gps_init();

  rawgps_init();

  err = pthread_create(&clock_thread_handle, NULL,
                       clock_thread, NULL);
  assert(err == 0);

  while(!do_exit) pause();

  err = pthread_join(clock_thread_handle, NULL);
  assert(err == 0);

  rawgps_destroy();

  gps_destroy();

  return 0;
}
