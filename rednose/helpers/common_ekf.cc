#include "common_ekf.h"

std::vector<const EKF*>& get_ekfs() {
  static std::vector<const EKF*> vec;
  return vec;
}

void ekf_register(const EKF* ekf) {
  get_ekfs().push_back(ekf);
}

const EKF* ekf_lookup(const std::string& ekf_name) {
  for (const auto& ekfi : get_ekfs()) {
    if (ekf_name == ekfi->name) {
      return ekfi;
    }
  }
  return NULL;
}
