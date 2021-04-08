#pragma once

#include <string>
#include <cmath>
#include <memory>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "rednose/helpers/ekf_sym.h"

#define KIND_UNKNOWN 0
#define KIND_NO_OBSERVATION 1
#define KIND_GPS_NED 2
#define KIND_ODOMETRIC_SPEED 3
#define KIND_PHONE_GYRO 4
#define KIND_GPS_VEL 5
#define KIND_PSEUDORANGE_GPS 6
#define KIND_PSEUDORANGE_RATE_GPS 7
#define KIND_SPEED 8
#define KIND_NO_ROT 9
#define KIND_PHONE_ACCEL 10
#define KIND_ORB_POINT 11
#define KIND_ECEF_POS 12
#define KIND_CAMERA_ODO_TRANSLATION 13
#define KIND_CAMERA_ODO_ROTATION 14
#define KIND_ORB_FEATURES 15
#define KIND_MSCKF_TEST 16
#define KIND_FEATURE_TRACK_TEST 17
#define KIND_LANE_PT 18
#define KIND_IMU_FRAME 19
#define KIND_PSEUDORANGE_GLONASS 20
#define KIND_PSEUDORANGE_RATE_GLONASS 21
#define KIND_PSEUDORANGE 22
#define KIND_PSEUDORANGE_RATE 23
#define KIND_ECEF_VEL 31
#define KIND_ECEF_ORIENTATION_FROM_GPS 32
#define KIND_ROAD_FRAME_XY_SPEED 24  // (x, y) [m/s]
#define KIND_ROAD_FRAME_YAW_RATE 25  // [rad/s]
#define KIND_STEER_ANGLE 26  // [rad]
#define KIND_ANGLE_OFFSET_FAST 27  // [rad]
#define KIND_STIFFNESS 28  // [-]
#define KIND_STEER_RATIO 29  // [-]
#define KIND_ROAD_FRAME_X_SPEED 30  // (x) [m/s]

#define STATE_ECEF_POS_START 0  // x, y and z in ECEF in meters
#define STATE_ECEF_POS_END 3
#define STATE_ECEF_ORIENTATION_START 3  // quat for pose of phone in ecef
#define STATE_ECEF_ORIENTATION_END 7
#define STATE_ECEF_VELOCITY_START 7  // ecef velocity in m/s
#define STATE_ECEF_VELOCITY_END 10
#define STATE_ANGULAR_VELOCITY_START 10  // roll, pitch and yaw rates in device frame in radians/s
#define STATE_ANGULAR_VELOCITY_END 13
#define STATE_GYRO_BIAS_START 13  // roll, pitch and yaw biases
#define STATE_GYRO_BIAS_END 16
#define STATE_ODO_SCALE_START 16  // odometer scale
#define STATE_ODO_SCALE_END 17
#define STATE_ACCELERATION_START 17  // Acceleration in device frame in m/s**2
#define STATE_ACCELERATION_END 20
#define STATE_IMU_OFFSET_START 20  // imu offset angles in radians
#define STATE_IMU_OFFSET_END 23

// Error-state has different slices because it is an ESKF
#define STATE_ECEF_POS_ERR_START 0
#define STATE_ECEF_POS_ERR_END 3
#define STATE_ECEF_ORIENTATION_ERR_START 3  // euler angles for orientation error
#define STATE_ECEF_ORIENTATION_ERR_END 6
#define STATE_ECEF_VELOCITY_ERR_START 6
#define STATE_ECEF_VELOCITY_ERR_END 9
#define STATE_ANGULAR_VELOCITY_ERR_START 9
#define STATE_ANGULAR_VELOCITY_ERR_END 12
#define STATE_GYRO_BIAS_ERR_START 12
#define STATE_GYRO_BIAS_ERR_END 15
#define STATE_ODO_SCALE_ERR_START 15
#define STATE_ODO_SCALE_ERR_END 16
#define STATE_ACCELERATION_ERR_START 16
#define STATE_ACCELERATION_ERR_END 19
#define STATE_IMU_OFFSET_ERR_START 19
#define STATE_IMU_OFFSET_ERR_END 22

#define EARTH_GM 3.986005e14  // m^3/s^2 (gravitational constant * mass of earth)

using namespace EKFS;

Eigen::Map<Eigen::VectorXd> get_mapvec(Eigen::VectorXd& vec) {
  return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.rows(), vec.cols());
}
Eigen::Map<MatrixXdr> get_mapmat(MatrixXdr& mat) {
  return Eigen::Map<MatrixXdr>(mat.data(), mat.rows(), mat.cols());
}
std::vector<Eigen::Map<Eigen::VectorXd>> get_vec_mapvec(std::vector<Eigen::VectorXd>& vec_vec) {
  std::vector<Eigen::Map<Eigen::VectorXd>> res;
  for (Eigen::VectorXd& vec : vec_vec) {
    res.push_back(get_mapvec(vec));
  }
  return res;
}
std::vector<Eigen::Map<MatrixXdr>> get_vec_mapmat(std::vector<MatrixXdr>& mat_vec) {
  std::vector<Eigen::Map<MatrixXdr>> res;
  for (MatrixXdr& mat : mat_vec) {
    res.push_back(get_mapmat(mat));
  }
  return res;
}

class LiveKalman {
public:
  LiveKalman();

  void init_state(Eigen::VectorXd& state, Eigen::VectorXd& covs_diag, double filter_time);
  void init_state(Eigen::VectorXd& state, MatrixXdr& covs, double filter_time);
  void init_state(Eigen::VectorXd& state, double filter_time);

  std::vector<MatrixXdr> get_R(int kind, int n);

  std::optional<Estimate> predict_and_observe(double t, int kind, std::vector<Eigen::VectorXd> meas, std::vector<MatrixXdr> R = {});
  std::optional<Estimate> predict_and_update_odo_speed(std::vector<Eigen::VectorXd> speed, double t, int kind);
  std::optional<Estimate> predict_and_update_odo_trans(std::vector<Eigen::VectorXd> trans, double t, int kind);
  std::optional<Estimate> predict_and_update_odo_rot(std::vector<Eigen::VectorXd> rot, double t, int kind);

private:
  std::string name = "live";

  std::shared_ptr<EKFSym> filter;

  int dim_state;
  int dim_state_err;

  Eigen::VectorXd initial_x;
  Eigen::VectorXd initial_P_diag;
  MatrixXdr Q;  // process noise
  std::unordered_map<int, MatrixXdr> obs_noise;
};
