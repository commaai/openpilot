#pragma once

#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <map>
#include <cmath>
#include <optional>

#include <eigen3/Eigen/Dense>

#include "common_ekf.h"

#define REWIND_TO_KEEP 512

namespace EKFS {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdr;

typedef struct Observation {
  double t;
  int kind;
  std::vector<Eigen::VectorXd> z;
  std::vector<MatrixXdr> R;
  std::vector<std::vector<double>> extra_args;
} Observation;

typedef struct Estimate {
  Eigen::VectorXd xk1;
  Eigen::VectorXd xk;
  MatrixXdr Pk1;
  MatrixXdr Pk;
  double t;
  int kind;
  std::vector<Eigen::VectorXd> y;
  std::vector<Eigen::VectorXd> z;
  std::vector<std::vector<double>> extra_args;
} Estimate;

class EKFSym {
public:
  EKFSym(std::string name, Eigen::Map<MatrixXdr> Q, Eigen::Map<Eigen::VectorXd> x_initial,
      Eigen::Map<MatrixXdr> P_initial, int dim_main, int dim_main_err, int N = 0, int dim_augment = 0,
      int dim_augment_err = 0, std::vector<int> maha_test_kinds = std::vector<int>(),
      std::vector<int> quaternion_idxs = std::vector<int>(),
      std::vector<std::string> global_vars = std::vector<std::string>(), double max_rewind_age = 1.0);
  void init_state(Eigen::Map<Eigen::VectorXd> state, Eigen::Map<MatrixXdr> covs, double filter_time);

  Eigen::VectorXd state();
  MatrixXdr covs();
  void set_filter_time(double t);
  double get_filter_time();
  void normalize_quaternions();
  void normalize_slice(int slice_start, int slice_end_ex);
  void set_global(std::string global_var, double val);
  void reset_rewind();

  void predict(double t);
  std::optional<Estimate> predict_and_update_batch(double t, int kind, std::vector<Eigen::Map<Eigen::VectorXd>> z,
      std::vector<Eigen::Map<MatrixXdr>> R, std::vector<std::vector<double>> extra_args = {{}}, bool augment = false);

  extra_routine_t get_extra_routine(const std::string& routine);

private:
  std::deque<Observation> rewind(double t);
  void checkpoint(Observation& obs);

  Estimate predict_and_update_batch(Observation& obs, bool augment);
  Eigen::VectorXd update(int kind, Eigen::VectorXd z, MatrixXdr R, std::vector<double> extra_args);

  // stuct with linked sympy generated functions
  const EKF *ekf = NULL;

  Eigen::VectorXd x;  // state
  MatrixXdr P;  // covs

  bool msckf;
  int N;
  int dim_augment;
  int dim_augment_err;
  int dim_main;
  int dim_main_err;

  // state
  int dim_x;
  int dim_err;

  double filter_time;

  std::vector<int> maha_test_kinds;
  std::vector<int> quaternion_idxs;

  std::vector<std::string> global_vars;

  // process noise
  MatrixXdr Q;

  // rewind stuff
  double max_rewind_age;
  std::deque<double> rewind_t;
  std::deque<std::pair<Eigen::VectorXd, MatrixXdr>> rewind_states;
  std::deque<Observation> rewind_obscache;

  Eigen::VectorXd augment_times;

  std::vector<int> feature_track_kinds;
};

}
