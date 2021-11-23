#pragma once

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <memory>
#include <string>

#include "cereal/messaging/messaging.h"
#include "common/transformations/coordinates.hpp"
#include "common/transformations/orientation.hpp"
#include "selfdrive/common/params.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

#include "selfdrive/sensord/sensors/constants.h"
#define VISION_DECIMATION 2
#define SENSOR_DECIMATION 10
#include "selfdrive/locationd/models/live_kf.h"

#define POSENET_STD_HIST_HALF 20

class Localizer {
public:
  Localizer();

  int locationd_thread();

  void reset_kalman(double current_time = NAN);
  void reset_kalman(double current_time, Eigen::VectorXd init_orient, Eigen::VectorXd init_pos, Eigen::VectorXd init_vel, MatrixXdr init_pos_R, MatrixXdr init_vel_R);
  void reset_kalman(double current_time, Eigen::VectorXd init_x, MatrixXdr init_P);
  void finite_check(double current_time = NAN);
  void time_check(double current_time = NAN);
  void update_reset_tracker();
  bool isGpsOK();

  kj::ArrayPtr<capnp::byte> get_message_bytes(MessageBuilder& msg_builder, uint64_t logMonoTime,
    bool inputsOK, bool sensorsOK, bool gpsOK);
  void build_live_location(cereal::LiveLocationKalman::Builder& fix);

  Eigen::VectorXd get_position_geodetic();
  Eigen::VectorXd get_state();
  Eigen::VectorXd get_stdev();

  void handle_msg_bytes(const char *data, const size_t size);
  void handle_msg(const cereal::Event::Reader& log);
  void handle_sensors(double current_time, const capnp::List<cereal::SensorEventData, capnp::Kind::STRUCT>::Reader& log);
  void handle_gps(double current_time, const cereal::GpsLocationData::Reader& log);
  void handle_car_state(double current_time, const cereal::CarState::Reader& log);
  void handle_cam_odo(double current_time, const cereal::CameraOdometry::Reader& log);
  void handle_live_calib(double current_time, const cereal::LiveCalibrationData::Reader& log);

  void input_fake_gps_observations(double current_time);

private:
  std::unique_ptr<LiveKalman> kf;

  Eigen::VectorXd calib;
  MatrixXdr device_from_calib;
  MatrixXdr calib_from_device;
  bool calibrated = false;

  double car_speed = 0.0;
  double last_reset_time = NAN;
  std::deque<double> posenet_stds;

  std::unique_ptr<LocalCoord> converter;

  int64_t unix_timestamp_millis = 0;
  double last_gps_fix = 0;
  double reset_tracker = 0.0;
  bool device_fell = false;
};
