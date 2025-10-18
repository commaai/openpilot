#include <vector>
#include <string>

#include "ekf.h"

std::vector<const EKF*>& ekf_get_all();
const EKF* ekf_lookup(const std::string& ekf_name);
void ekf_register(const EKF* ekf);
void ekf_load_and_register(const std::string& ekf_directory, const std::string& ekf_name);
