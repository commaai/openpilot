#include "live_kf.h"

using namespace EKFS;
using namespace Eigen;

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

LiveKalman::LiveKalman() {
  this->dim_state = 26;
  this->dim_state_err = 25;

  this->initial_x = live_initial_x;
  this->initial_P = live_initial_P_diag.asDiagonal();
  this->reset_orientation_P = live_reset_orientation_diag.asDiagonal();
  this->Q = live_Q_diag.asDiagonal();
  for (auto& pair : live_obs_noise_diag) {
    this->obs_noise[pair.first] = pair.second.asDiagonal();
  }

  // init filter
  this->filter = std::make_shared<EKFSym>(this->name, get_mapmat(this->Q), get_mapvec(this->initial_x),
    get_mapmat(initial_P),  this->dim_state, this->dim_state_err, 0, 0, 0, std::vector<int>(),
    std::vector<int>{3}, std::vector<std::string>(), 0.2);
}

void LiveKalman::init_state(VectorXd& state, VectorXd& covs_diag, double filter_time) {
  MatrixXdr covs = covs_diag.asDiagonal();
  this->filter->init_state(get_mapvec(state), get_mapmat(covs), filter_time);
}

void LiveKalman::init_state(VectorXd& state, MatrixXdr& covs, double filter_time) {
  this->filter->init_state(get_mapvec(state), get_mapmat(covs), filter_time);
}

void LiveKalman::init_state(VectorXd& state, double filter_time) {
  MatrixXdr covs = this->filter->covs();
  this->filter->init_state(get_mapvec(state), get_mapmat(covs), filter_time);
}

VectorXd LiveKalman::get_x() {
  return this->filter->state();
}

MatrixXdr LiveKalman::get_P() {
  return this->filter->covs();
}

double LiveKalman::get_filter_time() {
  return this->filter->get_filter_time();
}

std::vector<MatrixXdr> LiveKalman::get_R(int kind, int n) {
  std::vector<MatrixXdr> R;
  for (int i = 0; i < n; i++) {
    R.push_back(this->obs_noise[kind]);
  }
  return R;
}

std::optional<Estimate> LiveKalman::predict_and_observe(double t, int kind, std::vector<VectorXd> meas, std::vector<MatrixXdr> R) {
  std::optional<Estimate> r;
  if (R.size() == 0) {
    R = this->get_R(kind, meas.size());
  }
  r = this->filter->predict_and_update_batch(t, kind, get_vec_mapvec(meas), get_vec_mapmat(R));
  return r;
}

Eigen::VectorXd LiveKalman::get_initial_x() {
  return this->initial_x;
}

MatrixXdr LiveKalman::get_initial_P() {
  return this->initial_P;
}

MatrixXdr LiveKalman::get_reset_orientation_P() {
  return this->reset_orientation_P;
}

MatrixXdr LiveKalman::H(VectorXd in) {
  assert(in.size() == 6);
  Matrix<double, 3, 6, Eigen::RowMajor> res;
  this->filter->get_extra_routine("H")(in.data(), res.data());
  return res;
}
