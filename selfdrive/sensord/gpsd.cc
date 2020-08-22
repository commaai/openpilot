#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/resource.h>

#include <pthread.h>

#include <cutils/log.h>

#include <hardware/gps.h>
#include <utils/Timers.h>

#include "messaging.hpp"
#include "common/timing.h"
#include "common/swaglog.h"

volatile sig_atomic_t do_exit = 0;

namespace {

PubMaster *pm;

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

  pm->send("gpsNMEA", msg);
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

  pm->send("gpsLocation", msg);
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
  pm = new PubMaster({"gpsNMEA", "gpsLocation"});
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

}

void gps_destroy() {
  delete pm;
  gGpsInterface->stop();
  gGpsInterface->cleanup();
}

}

int main() {
  setpriority(PRIO_PROCESS, 0, -13);

  signal(SIGINT, (sighandler_t)set_do_exit);
  signal(SIGTERM, (sighandler_t)set_do_exit);

  gps_init();

  while(!do_exit) pause();

  gps_destroy();

  return 0;
}
