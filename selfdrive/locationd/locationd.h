#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <memory>

#include <eigen3/Eigen/Dense>

#include "messaging.hpp"
#include "common/params.h"
#include "common/util.h"
#include "common/swaglog.h"
#include "common/timing.h"
#include "common/transformations/coordinates.hpp"
#include "common/transformations/orientation.hpp"

#include "models/live_kf.h"

#define DEG2RAD(x) ((x) * M_PI / 180.0)
#define RAD2DEG(x) ((x) * 180.0 / M_PI)

#define VISION_DECIMATION 2
#define SENSOR_DECIMATION 10
#define POSENET_STD_HIST 40

Eigen::VectorXd floatlist_to_vector(const capnp::List<float, capnp::Kind::PRIMITIVE>::Reader& floatlist);
Eigen::VectorXd quat2vector(const Eigen::Quaterniond& quat);
Eigen::Quaterniond vector2quat(const Eigen::VectorXd& vec);

class Localizer {
public:
  Localizer();

  int locationd_thread();

  void update_kalman(double t, int kind, std::vector<Eigen::VectorXd> meas, std::vector<MatrixXdr> R = {});
  void reset_kalman(double current_time = NAN);
  void reset_kalman(double current_time, Eigen::VectorXd init_orient, Eigen::VectorXd init_pos);

  cereal::LiveLocationKalman liveLocationMsg();
  static cereal::LiveLocationKalman msg_from_state(/*converter, */MatrixXdr calib_from_device, //H,
      Eigen::VectorXd predicted_state, MatrixXdr predicted_cov, bool calibrated);

  void handle_sensors(double current_time, const capnp::List<cereal::SensorEventData, capnp::Kind::STRUCT>::Reader& log);
  void handle_gps(double current_time, const cereal::GpsLocationData::Reader& log);
  void handle_car_state(double current_time, const cereal::CarState::Reader& log);
  void handle_cam_odo(double current_time, const cereal::CameraOdometry::Reader& log);
  void handle_live_calib(double current_time, const cereal::LiveCalibrationData::Reader& log);

private:
  std::shared_ptr<LiveKalman> kf;

  double max_age = 0.1;

  Eigen::VectorXd calib;
  MatrixXdr device_from_calib;
  MatrixXdr calib_from_device;
  bool calibrated = false;

//     self.H = get_H()

  int posenet_invalid_count = 0;
  int car_speed = 0;
  Eigen::VectorXd posenet_stds;
  int posenet_stds_i = 0;

  std::shared_ptr<LocalCoord> converter;

  int unix_timestamp_millis = 0;
  double last_gps_fix = 0;
  bool device_fell = false;

  int gyro_counter = 0;
  int acc_counter = 0;
  int speed_counter = 0;
  int cam_counter = 0;
};
