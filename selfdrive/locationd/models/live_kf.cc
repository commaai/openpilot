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
  this->dim_state = 23;
  this->dim_state_err = 22;

  this->initial_x = VectorXd(this->dim_state);
  this->initial_x << -2.7e6, 4.2e6, 3.8e6,
                      1.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      1.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0;

  VectorXd initial_P_diag(this->dim_state_err);
  initial_P_diag << 1e16, 1e16, 1e16,
                    1e6, 1e6, 1e6,
                    1e4, 1e4, 1e4,
                    std::pow(1.0, 2), std::pow(1.0, 2), std::pow(1.0, 2),
                    std::pow(0.05, 2), std::pow(0.05, 2), std::pow(0.05, 2),
                    std::pow(0.02, 2),
                    std::pow(1.0, 2), std::pow(1.0, 2), std::pow(1.0, 2),
                    std::pow(0.01, 2), std::pow(0.01, 2), std::pow(0.01, 2);
  this->initial_P = initial_P_diag.asDiagonal();

  VectorXd Q_diag(this->dim_state_err);
  Q_diag << std::pow(0.03, 2), std::pow(0.03, 2), std::pow(0.03, 2),
            std::pow(0.001, 2), std::pow(0.001, 2), std::pow(0.001, 2),
            std::pow(0.01, 2), std::pow(0.01, 2), std::pow(0.01, 2),
            std::pow(0.1, 2), std::pow(0.1, 2), std::pow(0.1, 2),
            std::pow(0.005 / 100, 2), std::pow(0.005 / 100, 2), std::pow(0.005 / 100, 2),
            std::pow(0.02 / 100, 2),
            std::pow(3.0, 2), std::pow(3.0, 2), std::pow(3.0, 2),
            std::pow(0.05 / 60, 2), std::pow(0.05 / 60, 2), std::pow(0.05 / 60, 2);
  this->Q = Q_diag.asDiagonal();

  this->obs_noise = {  // TODO? create two large diagional matrices
    { KIND_ODOMETRIC_SPEED, (VectorXd(1) << std::pow(0.2, 2)).finished().asDiagonal() },
    { KIND_PHONE_GYRO, (VectorXd(3) << std::pow(0.025, 2), std::pow(0.025, 2), std::pow(0.025, 2)).finished().asDiagonal() },
    { KIND_PHONE_ACCEL, (VectorXd(3) << std::pow(0.5, 2), std::pow(0.5, 2), std::pow(0.5, 2)).finished().asDiagonal() },
    { KIND_CAMERA_ODO_ROTATION, (VectorXd(3) << std::pow(0.05, 2), std::pow(0.05, 2), std::pow(0.05, 2)).finished().asDiagonal() },
    { KIND_IMU_FRAME, (VectorXd(3) << std::pow(0.05, 2), std::pow(0.05, 2), std::pow(0.05, 2)).finished().asDiagonal() },
    { KIND_NO_ROT, (VectorXd(3) << std::pow(0.005, 2), std::pow(0.005, 2), std::pow(0.005, 2)).finished().asDiagonal() },
    { KIND_ECEF_POS, (VectorXd(3) << std::pow(5.0, 2), std::pow(5.0, 2), std::pow(5.0, 2)).finished().asDiagonal() },
    { KIND_ECEF_VEL, (VectorXd(3) << std::pow(0.5, 2), std::pow(0.5, 2), std::pow(0.5, 2)).finished().asDiagonal() },
    { KIND_ECEF_ORIENTATION_FROM_GPS, (VectorXd(4) << std::pow(0.2, 2), std::pow(0.2, 2), std::pow(0.2, 2), std::pow(0.2, 2)).finished().asDiagonal() },
  };

  // init filter
  this->filter = std::make_shared<EKFSym>(this->name, get_mapmat(this->Q), get_mapvec(this->initial_x), get_mapmat(initial_P),
      this->dim_state, this->dim_state_err, 0, 0, 0, std::vector<int>(), std::vector<std::string>(), 0.2);
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

std::vector<MatrixXdr> LiveKalman::get_R(int kind, int n) {
  std::vector<MatrixXdr> R;
  for (int i = 0; i < n; i++) {
    R.push_back(this->obs_noise[kind]);
  }
  return R;
}

std::optional<Estimate> LiveKalman::predict_and_observe(double t, int kind, std::vector<VectorXd> meas, std::vector<MatrixXdr> R) {
  std::optional<Estimate> r;
  switch (kind) {
  case KIND_CAMERA_ODO_TRANSLATION:
    r = this->predict_and_update_odo_trans(meas, t, kind);
    break;
  case KIND_CAMERA_ODO_ROTATION:
    r = this->predict_and_update_odo_rot(meas, t, kind);
    break;
  case KIND_ODOMETRIC_SPEED:
    r = this->predict_and_update_odo_speed(meas, t, kind);
    break;
  default:
    if (R.size() == 0) {
      R = this->get_R(kind, meas.size());
    }
    r = this->filter->predict_and_update_batch(t, kind, get_vec_mapvec(meas), get_vec_mapmat(R));
    break;
  }
  this->filter->normalize_state(STATE_ECEF_ORIENTATION_START, STATE_ECEF_ORIENTATION_END);
  return r;
}

std::optional<Estimate> LiveKalman::predict_and_update_odo_speed(std::vector<VectorXd> speed, double t, int kind) {
  std::vector<MatrixXdr> R;
  R.assign(speed.size(), (MatrixXdr(1, 1) << std::pow(0.2, 2)).finished().asDiagonal());
  return this->filter->predict_and_update_batch(t, kind, get_vec_mapvec(speed), get_vec_mapmat(R));
}

std::optional<Estimate> LiveKalman::predict_and_update_odo_trans(std::vector<VectorXd> trans, double t, int kind) {
  std::vector<VectorXd> z;
  std::vector<MatrixXdr> R;
  for (VectorXd& trns : trans) {
    assert(trns.size() == 6); // TODO remove
    z.push_back(trns.head(3));
    R.push_back(trns.segment<3>(3).array().square().matrix().asDiagonal());
  }
  return this->filter->predict_and_update_batch(t, kind, get_vec_mapvec(z), get_vec_mapmat(R));
}

std::optional<Estimate> LiveKalman::predict_and_update_odo_rot(std::vector<VectorXd> rot, double t, int kind) {
  std::vector<VectorXd> z;
  std::vector<MatrixXdr> R;
  for (VectorXd& rt : rot) {
    assert(rt.size() == 6); // TODO remove
    z.push_back(rt.head(3));
    R.push_back(rt.segment<3>(3).array().square().matrix().asDiagonal());
  }
  return this->filter->predict_and_update_batch(t, kind, get_vec_mapvec(z), get_vec_mapmat(R));
}

Eigen::VectorXd LiveKalman::get_initial_x() {
  return this->initial_x;
}

MatrixXdr LiveKalman::get_initial_P() {
  return this->initial_P;
}

MatrixXdr LiveKalman::H(VectorXd in) {
  assert(in.size() == 6);
  Matrix<double, 3, 6, Eigen::RowMajor> res;
  this->filter->get_extra_routine("H")(in.data(), res.data());
  return res;
}
