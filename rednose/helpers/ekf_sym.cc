#include "ekf_sym.h"
#include "logger/logger.h"

using namespace EKFS;
using namespace Eigen;

EKFSym::EKFSym(std::string name, Map<MatrixXdr> Q, Map<VectorXd> x_initial, Map<MatrixXdr> P_initial, int dim_main,
    int dim_main_err, int N, int dim_augment, int dim_augment_err, std::vector<int> maha_test_kinds,
    std::vector<int> quaternion_idxs, std::vector<std::string> global_vars, double max_rewind_age)
{
  // TODO: add logger

  this->ekf = ekf_lookup(name);
  assert(this->ekf);

  this->msckf = N > 0;
  this->N = N;
  this->dim_augment = dim_augment;
  this->dim_augment_err = dim_augment_err;
  this->dim_main = dim_main;
  this->dim_main_err = dim_main_err;

  this->dim_x = x_initial.rows();
  this->dim_err = P_initial.rows();

  assert(dim_main + dim_augment * N == dim_x);
  assert(dim_main_err + dim_augment_err * N == this->dim_err);
  assert(Q.rows() == P_initial.rows() && Q.cols() == P_initial.cols());

  // kinds that should get mahalanobis distance
  // tested for outlier rejection
  this->maha_test_kinds = maha_test_kinds;

  // quaternions need normalization
  this->quaternion_idxs = quaternion_idxs;

  this->global_vars = global_vars;

  // Process noise
  this->Q = Q;

  this->max_rewind_age = max_rewind_age;
  this->init_state(x_initial, P_initial, NAN);
}

void EKFSym::init_state(Map<VectorXd> state, Map<MatrixXdr> covs, double init_filter_time) {
  this->x = state;
  this->P = covs;
  this->filter_time = init_filter_time;
  this->augment_times = VectorXd::Zero(this->N);
  this->reset_rewind();
}

VectorXd EKFSym::state() {
  return this->x;
}

MatrixXdr EKFSym::covs() {
  return this->P;
}

void EKFSym::set_filter_time(double t) {
  this->filter_time = t;
}

double EKFSym::get_filter_time() {
  return this->filter_time;
}

void EKFSym::normalize_quaternions() {
  for(std::size_t i = 0; i < this->quaternion_idxs.size(); ++i) {
    this->normalize_slice(this->quaternion_idxs[i], this->quaternion_idxs[i] + 4);
  }
}

void EKFSym::normalize_slice(int slice_start, int slice_end_ex) {
  this->x.block(slice_start, 0, slice_end_ex - slice_start, this->x.cols()).normalize();
}

void EKFSym::set_global(std::string global_var, double val) {
  this->ekf->sets.at(global_var)(val);
}

std::optional<Estimate> EKFSym::predict_and_update_batch(double t, int kind, std::vector<Map<VectorXd>> z_map,
    std::vector<Map<MatrixXdr>> R_map, std::vector<std::vector<double>> extra_args, bool augment)
{
  // TODO handle rewinding at this level

  std::deque<Observation> rewound;
  if (!std::isnan(this->filter_time) && t < this->filter_time) {
    if (this->rewind_t.empty() || t < this->rewind_t.front() || t < this->rewind_t.back() - this->max_rewind_age) {
      LOGD("observation too old at %d with filter at %d, ignoring!", t, this->filter_time);
      return std::nullopt;
    }
    rewound = this->rewind(t);
  }

  Observation obs;
  obs.t = t;
  obs.kind = kind;
  obs.extra_args = extra_args;
  for (Map<VectorXd> zi : z_map) {
    obs.z.push_back(zi);
  }
  for (Map<MatrixXdr> Ri : R_map) {
    obs.R.push_back(Ri);
  }

  std::optional<Estimate> res = std::make_optional(this->predict_and_update_batch(obs, augment));

  // optional fast forward
  while (!rewound.empty()) {
    this->predict_and_update_batch(rewound.front(), false);
    rewound.pop_front();
  }

  return res;
}

void EKFSym::reset_rewind() {
  this->rewind_obscache.clear();
  this->rewind_t.clear();
  this->rewind_states.clear();
}

std::deque<Observation> EKFSym::rewind(double t) {
  std::deque<Observation> rewound;

  // rewind observations until t is after previous observation
  while (this->rewind_t.back() > t) {
    rewound.push_front(this->rewind_obscache.back());
    this->rewind_t.pop_back();
    this->rewind_states.pop_back();
    this->rewind_obscache.pop_back();
  }

  // set the state to the time right before that
  this->filter_time = this->rewind_t.back();
  this->x = this->rewind_states.back().first;
  this->P = this->rewind_states.back().second;

  return rewound;
}

void EKFSym::checkpoint(Observation& obs) {
  // push to rewinder
  this->rewind_t.push_back(this->filter_time);
  this->rewind_states.push_back(std::make_pair(this->x, this->P));
  this->rewind_obscache.push_back(obs);

  // only keep a certain number around
  if (this->rewind_t.size() > REWIND_TO_KEEP) {
    this->rewind_t.pop_front();
    this->rewind_states.pop_front();
    this->rewind_obscache.pop_front();
  }
}

Estimate EKFSym::predict_and_update_batch(Observation& obs, bool augment) {
  assert(obs.z.size() == obs.R.size());
  assert(obs.z.size() == obs.extra_args.size());

  this->predict(obs.t);

  Estimate res;
  res.t = obs.t;
  res.kind = obs.kind;
  res.z = obs.z;
  res.extra_args = obs.extra_args;
  res.xk1 = this->x;
  res.Pk1 = this->P;

  // update batch
  std::vector<VectorXd> y;
  for (int i = 0; i < obs.z.size(); i++) {
    assert(obs.z[i].rows() == obs.R[i].rows());
    assert(obs.z[i].rows() == obs.R[i].cols());

    // update state
    y.push_back(this->update(obs.kind, obs.z[i], obs.R[i], obs.extra_args[i]));
  }

  res.xk = this->x;
  res.Pk = this->P;
  res.y = y;

  assert(!augment); // TODO
  // if (augment) {
  //   this->augment();
  // }

  this->checkpoint(obs);

  return res;
}

void EKFSym::predict(double t) {
  // initialize time
  if (std::isnan(this->filter_time)) {
    this->filter_time = t;
  }

  // predict
  double dt = t - this->filter_time;
  assert(dt >= 0.0);

  this->ekf->predict(this->x.data(), this->P.data(), this->Q.data(), dt);
  this->normalize_quaternions();
  this->filter_time = t;
}

VectorXd EKFSym::update(int kind, VectorXd z, MatrixXdr R, std::vector<double> extra_args) {
  this->ekf->updates.at(kind)(this->x.data(), this->P.data(), z.data(), R.data(), extra_args.data());
  this->normalize_quaternions();

  if (this->msckf && std::find(this->feature_track_kinds.begin(), this->feature_track_kinds.end(), kind) != this->feature_track_kinds.end()) {
    return z.head(z.rows() - extra_args.size());
  }
  return z;
}

extra_routine_t EKFSym::get_extra_routine(const std::string& routine) {
  return this->ekf->extra_routines.at(routine);
}
