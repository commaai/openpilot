#pragma once

#include <eigen3/Eigen/Dense>
#include "cereal/gen/cpp/log.capnp.h"

#define DEGREES_TO_RADIANS 0.017453292519943295

class Localizer
{
  Eigen::Matrix2d A;
  Eigen::Matrix2d I;
  Eigen::Matrix2d Q;
  Eigen::Matrix2d P;
  Eigen::Matrix<double, 1, 2> C_posenet;
  Eigen::Matrix<double, 1, 2> C_gyro;

  double R_gyro;

  void update_state(const Eigen::Matrix<double, 1, 2> &C, const double R, double current_time, double meas);
  void handle_sensor_events(capnp::List<cereal::SensorEventData>::Reader sensor_events, double current_time);
  void handle_camera_odometry(cereal::CameraOdometry::Reader camera_odometry, double current_time);
  void handle_controls_state(cereal::ControlsState::Reader controls_state, double current_time);

public:
  Eigen::Vector2d x;
  double steering_angle = 0;
  double car_speed = 0;
  double posenet_speed = 0;
  double prev_update_time = -1;
  double controls_state_time = -1;
  double sensor_data_time = -1;

  Localizer();
  void handle_log(cereal::Event::Reader event);

};
