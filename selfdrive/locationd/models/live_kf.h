#pragma once

#include <string>
#include <cmath>
#include <memory>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "generated/live_kf_constants.h"
#include "rednose/helpers/ekf_sym.h"

#define EARTH_GM 3.986005e14  // m^3/s^2 (gravitational constant * mass of earth)

using namespace EKFS;

Eigen::Map<Eigen::VectorXd> get_mapvec(Eigen::VectorXd& vec);
Eigen::Map<MatrixXdr> get_mapmat(MatrixXdr& mat);
std::vector<Eigen::Map<Eigen::VectorXd>> get_vec_mapvec(std::vector<Eigen::VectorXd>& vec_vec);
std::vector<Eigen::Map<MatrixXdr>> get_vec_mapmat(std::vector<MatrixXdr>& mat_vec);

class LiveKalman {
public:
  LiveKalman();

  void init_state(Eigen::VectorXd& state, Eigen::VectorXd& covs_diag, double filter_time);
  void init_state(Eigen::VectorXd& state, MatrixXdr& covs, double filter_time);
  void init_state(Eigen::VectorXd& state, double filter_time);

  Eigen::VectorXd get_x();
  MatrixXdr get_P();
  double get_filter_time();
  std::vector<MatrixXdr> get_R(int kind, int n);

  std::optional<Estimate> predict_and_observe(double t, int kind, std::vector<Eigen::VectorXd> meas, std::vector<MatrixXdr> R = {});
  std::optional<Estimate> predict_and_update_odo_speed(std::vector<Eigen::VectorXd> speed, double t, int kind);
  std::optional<Estimate> predict_and_update_odo_trans(std::vector<Eigen::VectorXd> trans, double t, int kind);
  std::optional<Estimate> predict_and_update_odo_rot(std::vector<Eigen::VectorXd> rot, double t, int kind);

  Eigen::VectorXd get_initial_x();
  MatrixXdr get_initial_P();

  MatrixXdr H(Eigen::VectorXd in);

private:
  std::string name = "live";

  std::shared_ptr<EKFSym> filter;

  int dim_state;
  int dim_state_err;

  Eigen::VectorXd initial_x;
  MatrixXdr initial_P;
  MatrixXdr Q;  // process noise
  std::unordered_map<int, MatrixXdr> obs_noise;
};
