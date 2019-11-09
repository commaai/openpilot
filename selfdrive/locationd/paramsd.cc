#include <future>
#include <iostream>
#include <cassert>
#include <csignal>
#include <unistd.h>


#include <capnp/message.h>
#include <capnp/serialize-packed.h>

#include "json11.hpp"
#include "cereal/gen/cpp/log.capnp.h"

#include "common/swaglog.h"
#include "common/messaging.h"
#include "common/params.h"
#include "common/timing.h"

#include "messaging.hpp"
#include "locationd_yawrate.h"
#include "params_learner.h"


void sigpipe_handler(int sig) {
  LOGE("SIGPIPE received");
}


int main(int argc, char *argv[]) {
  signal(SIGPIPE, (sighandler_t)sigpipe_handler);

  Context * c = Context::create();
  SubSocket * controls_state_sock = SubSocket::create(c, "controlsState");
  SubSocket * sensor_events_sock = SubSocket::create(c, "sensorEvents");
  SubSocket * camera_odometry_sock = SubSocket::create(c, "cameraOdometry");
  PubSocket * live_parameters_sock = PubSocket::create(c, "liveParameters");
  Poller * poller = Poller::create({controls_state_sock, sensor_events_sock, camera_odometry_sock});

  Localizer localizer;

  // Read car params
  char *value;
  size_t value_sz = 0;

  LOGW("waiting for params to set vehicle model");
  while (true) {
    read_db_value(NULL, "CarParams", &value, &value_sz);
    if (value_sz > 0) break;
    usleep(100*1000);
  }
  LOGW("got %d bytes CarParams", value_sz);

  // make copy due to alignment issues
  auto amsg = kj::heapArray<capnp::word>((value_sz / sizeof(capnp::word)) + 1);
  memcpy(amsg.begin(), value, value_sz);
  free(value);

  capnp::FlatArrayMessageReader cmsg(amsg);
  cereal::CarParams::Reader car_params = cmsg.getRoot<cereal::CarParams>();

  // Read params from previous run
  const int result = read_db_value(NULL, "LiveParameters", &value, &value_sz);

  std::string fingerprint = car_params.getCarFingerprint();
  std::string vin = car_params.getCarVin();
  double sR = car_params.getSteerRatio();
  double x = 1.0;
  double ao = 0.0;
  double posenet_invalid_count = 0;

  if (result == 0){
    auto str = std::string(value, value_sz);
    free(value);

    std::string err;
    auto json = json11::Json::parse(str, err);
    if (json.is_null() || !err.empty()) {
      std::string log = "Error parsing json: " + err;
      LOGW(log.c_str());
    } else {
      std::string new_fingerprint = json["carFingerprint"].string_value();
      std::string new_vin = json["carVin"].string_value();

      if (fingerprint == new_fingerprint && vin == new_vin) {
        std::string log = "Parameter starting with: " + str;
        LOGW(log.c_str());

        sR = json["steerRatio"].number_value();
        x = json["stiffnessFactor"].number_value();
        ao = json["angleOffsetAverage"].number_value();
      }
    }
  }

  ParamsLearner learner(car_params, ao, x, sR, 1.0);

  // Main loop
  int save_counter = 0;
  while (true){
    for (auto s : poller->poll(-1)){
      Message * msg = s->receive();

      auto amsg = kj::heapArray<capnp::word>((msg->getSize() / sizeof(capnp::word)) + 1);
      memcpy(amsg.begin(), msg->getData(), msg->getSize());

      capnp::FlatArrayMessageReader capnp_msg(amsg);
      cereal::Event::Reader event = capnp_msg.getRoot<cereal::Event>();

      localizer.handle_log(event);

      auto which = event.which();
      // Throw vision failure if posenet and odometric speed too different
      if (which == cereal::Event::CAMERA_ODOMETRY){
        if (std::abs(localizer.posenet_speed - localizer.car_speed) > std::max(0.4 * localizer.car_speed, 5.0)) {
          posenet_invalid_count++;
        } else {
          posenet_invalid_count = 0;
        }
      } else if (which == cereal::Event::CONTROLS_STATE){
        save_counter++;

        double yaw_rate = -localizer.x[0];
        bool valid = learner.update(yaw_rate, localizer.car_speed, localizer.steering_angle);

        // TODO: Fix in replay
        double sensor_data_age = localizer.controls_state_time - localizer.sensor_data_time;
        double camera_odometry_age = localizer.controls_state_time - localizer.camera_odometry_time;

        double angle_offset_degrees = RADIANS_TO_DEGREES * learner.ao;
        double angle_offset_average_degrees = RADIANS_TO_DEGREES * learner.slow_ao;

        capnp::MallocMessageBuilder msg;
        cereal::Event::Builder event = msg.initRoot<cereal::Event>();
        event.setLogMonoTime(nanos_since_boot());
        auto live_params = event.initLiveParameters();
        live_params.setValid(valid);
        live_params.setYawRate(localizer.x[0]);
        live_params.setGyroBias(localizer.x[2]);
        live_params.setSensorValid(sensor_data_age < 5.0);
        live_params.setAngleOffset(angle_offset_degrees);
        live_params.setAngleOffsetAverage(angle_offset_average_degrees);
        live_params.setStiffnessFactor(learner.x);
        live_params.setSteerRatio(learner.sR);
        live_params.setPosenetSpeed(localizer.posenet_speed);
        live_params.setPosenetValid((posenet_invalid_count < 4) && (camera_odometry_age < 5.0));

        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        live_parameters_sock->send((char*)bytes.begin(), bytes.size());

        // Save parameters every minute
        if (save_counter % 6000 == 0) {
          json11::Json json = json11::Json::object {
                                                    {"carVin", vin},
                                                    {"carFingerprint", fingerprint},
                                                    {"steerRatio", learner.sR},
                                                    {"stiffnessFactor", learner.x},
                                                    {"angleOffsetAverage", angle_offset_average_degrees},
          };

          std::string out = json.dump();
          std::async(std::launch::async,
                     [out]{
                       write_db_value(NULL, "LiveParameters", out.c_str(), out.length());
                     });
        }
      }
      delete msg;
    }
  }

  delete live_parameters_sock;
  delete controls_state_sock;
  delete camera_odometry_sock;
  delete sensor_events_sock;
  delete poller;
  delete c;

  return 0;
}
