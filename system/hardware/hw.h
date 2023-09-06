#pragma once

#include <string>

#include "system/hardware/base.h"
#include "common/util.h"

#if QCOM2
#include "system/hardware/tici/hardware.h"
#define Hardware HardwareTici
#else
#include "system/hardware/pc/hardware.h"
#define Hardware HardwarePC
#endif

namespace Path {
inline std::string comma_home() {
  return util::getenv("HOME") + "/.comma" + util::getenv("OPENPILOT_PREFIX", "");
}
inline std::string log_root() {
  if (const char *env = getenv("LOG_ROOT")) {
    return env;
  }
  return Hardware::PC() ? Path::comma_home() + "/media/0/realdata" : "/data/media/0/realdata";
}
inline std::string params() {
  return Hardware::PC() ? util::getenv("PARAMS_ROOT", Path::comma_home() + "/params") : "/data/params";
}
inline std::string rsa_file() {
  return Hardware::PC() ? Path::comma_home() + "/persist/comma/id_rsa" : "/persist/comma/id_rsa";
}
}  // namespace Path
