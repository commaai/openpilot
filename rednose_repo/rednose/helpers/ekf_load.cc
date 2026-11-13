#include "ekf_load.h"
#include <dlfcn.h>

std::vector<const EKF*>& ekf_get_all() {
  static std::vector<const EKF*> vec;
  return vec;
}

void ekf_register(const EKF* ekf) {
  ekf_get_all().push_back(ekf);
}

const EKF* ekf_lookup(const std::string& ekf_name) {
  for (const auto& ekfi : ekf_get_all()) {
    if (ekf_name == ekfi->name) {
      return ekfi;
    }
  }
  return NULL;
}

void ekf_load_and_register(const std::string& ekf_directory, const std::string& ekf_name) {
  if (ekf_lookup(ekf_name)) {
    return;
  }

#ifdef __APPLE__
  std::string dylib_ext = ".dylib";
#else
  std::string dylib_ext = ".so";
#endif
  std::string ekf_path = ekf_directory + "/lib" + ekf_name + dylib_ext;
  void* handle = dlopen(ekf_path.c_str(), RTLD_NOW);
  assert(handle);
  void* (*ekf_get)() = (void*(*)())dlsym(handle, "ekf_get");
  assert(ekf_get != NULL);
  const EKF* ekf = (const EKF*)ekf_get();
  ekf_register(ekf);
}
