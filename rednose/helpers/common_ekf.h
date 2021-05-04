#pragma once

#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <map>
#include <cmath>

#include <eigen3/Eigen/Dense>

typedef void (*extra_routine_t)(double *, double *);

struct EKF {
  std::string name;
  std::vector<int> kinds;
  std::vector<int> feature_kinds;

  void (*f_fun)(double *, double, double *);
  void (*F_fun)(double *, double, double *);
  void (*err_fun)(double *, double *, double *);
  void (*inv_err_fun)(double *, double *, double *);
  void (*H_mod_fun)(double *, double *);
  void (*predict)(double *, double *, double *, double);
  std::unordered_map<int, void (*)(double *, double *, double *)> hs = {};
  std::unordered_map<int, void (*)(double *, double *, double *)> Hs = {};
  std::unordered_map<int, void (*)(double *, double *, double *, double *, double *)> updates = {};
  std::unordered_map<int, void (*)(double *, double *, double *)> Hes = {};
  std::unordered_map<std::string, void (*)(double)> sets = {};
  std::unordered_map<std::string, extra_routine_t> extra_routines = {};
};

std::vector<const EKF*>& get_ekfs();
const EKF* ekf_lookup(const std::string& ekf_name);

void ekf_register(const EKF* ekf);

#define ekf_init(ekf) \
static void __attribute__((constructor)) do_ekf_init_ ## ekf(void) { \
  ekf_register(&ekf); \
}
