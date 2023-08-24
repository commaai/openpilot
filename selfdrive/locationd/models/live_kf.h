#pragma once

#include <string>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "generated/live_kf_constants.h"
#include "rednose/helpers/ekf_sym.h"

#define EARTH_GM 3.986005e14  // m^3/s^2 (gravitational constant * mass of earth)

using namespace EKFS;

Eigen::Map<Eigen::VectorXd> get_mapvec(const Eigen::VectorXd &vec);
Eigen::Map<MatrixXdr> get_mapmat(const MatrixXdr &mat);
std::vector<Eigen::Map<Eigen::VectorXd>> get_vec_mapvec(const std::vector<Eigen::VectorXd> &vec_vec);
std::vector<Eigen::Map<MatrixXdr>> get_vec_mapmat(const std::vector<MatrixXdr> &mat_vec);

class LiveKalman {
public:
  LiveKalman();

  void init_state(const Eigen::VectorXd &state, const Eigen::VectorXd &covs_diag, double filter_time);
  void init_state(const Eigen::VectorXd &state, const MatrixXdr &covs, double filter_time);
  void init_state(const Eigen::VectorXd &state, double filter_time);

  Eigen::VectorXd get_x();
  MatrixXdr get_P();
  double get_filter_time();
  std::vector<MatrixXdr> get_R(int kind, int n);

  std::optional<Estimate> predict_and_observe(double t, int kind, const std::vector<Eigen::VectorXd> &meas, std::vector<MatrixXdr> R = {});
  std::optional<Estimate> predict_and_update_odo_speed(std::vector<Eigen::VectorXd> speed, double t, int kind);
  std::optional<Estimate> predict_and_update_odo_trans(std::vector<Eigen::VectorXd> trans, double t, int kind);
  std::optional<Estimate> predict_and_update_odo_rot(std::vector<Eigen::VectorXd> rot, double t, int kind);
  void predict(double t);

  const Eigen::VectorXd &get_initial_x();
  const MatrixXdr &get_initial_P();
  const MatrixXdr &get_fake_gps_pos_cov();
  const MatrixXdr &get_fake_gps_vel_cov();
  const MatrixXdr &get_reset_orientation_P();

  MatrixXdr H(const Eigen::VectorXd &in);

private:
  std::string name = "live";

  std::shared_ptr<EKFSym> filter;

  int dim_state;
  int dim_state_err;

  Eigen::VectorXd initial_x;
  MatrixXdr initial_P;
  MatrixXdr fake_gps_pos_cov;
  MatrixXdr fake_gps_vel_cov;
  MatrixXdr reset_orientation_P;
  MatrixXdr Q;  // process noise
  std::unordered_map<int, MatrixXdr> obs_noise;
};
