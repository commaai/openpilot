#pragma once

#include "selfdrive/hardware/base.h"
#include "selfdrive/common/util.h"

#ifdef QCOM
#include "selfdrive/hardware/eon/hardware.h"
#define Hardware HardwareEon
#elif QCOM2
#include "selfdrive/hardware/tici/hardware.h"
#define Hardware HardwareTici
#else
class HardwarePC : public HardwareNone {
public:
  static std::string get_os_version() { return "openpilot for PC"; }
  static constexpr bool PC() { return true; }
  static bool TICI() { return util::getenv("TICI", 0) == 1; }
};
#define Hardware HardwarePC
#endif

namespace Path {
inline static std::string HOME = util::getenv("HOME");
inline std::string log_root() {
  if (const char *env = getenv("LOG_ROOT")) {
    return env;
  }
  return Hardware::PC() ? HOME + "/.comma/media/0/realdata" : "/data/media/0/realdata";
}
inline std::string params() {
  return Hardware::PC() ? HOME + "/.comma/params" : "/data/params";
}
inline std::string rsa_file() {
  return Hardware::PC() ? HOME + "/.comma/persist/comma/id_rsa" : "/persist/comma/id_rsa";
}
}  // namespace Path
