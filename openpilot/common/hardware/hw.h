#pragma once

#include <string>

#include "common/hardware/base.h"
#include "common/util.h"

#if __COMMA_HARDWARE__
#include "common/hardware/comma/hardware.h"
#define Hardware HardwareComma
#else
#include "common/hardware/pc/hardware.h"
#define Hardware HardwarePC
#endif

namespace Path {
  inline std::string openpilot_prefix() {
    return util::getenv("OPENPILOT_PREFIX", "");
  }

  inline std::string comma_home() {
    return util::getenv("HOME") + "/.comma" + Path::openpilot_prefix();
  }

  inline std::string log_root() {
    if (const char *env = getenv("LOG_ROOT")) {
      return env;
    }
    return Hardware::PC() ? Path::comma_home() + "/media/0/realdata" : "/data/media/0/realdata";
  }

  // where params are read and written, which is RAM on the device
  inline std::string params() {
    return util::getenv("PARAMS_ROOT", Hardware::PC() ? (Path::comma_home() + "/params") : "/dev/shm/params");
  }

  // where the params in RAM are saved. empty when params() is the disk, and for tests
  inline std::string params_disk() {
    bool ram = !Hardware::PC() && getenv("PARAMS_ROOT") == nullptr && Path::openpilot_prefix().empty();
    return util::getenv("PARAMS_DISK_ROOT", ram ? "/data/params" : "");
  }

  inline std::string swaglog_ipc() {
    return "/tmp/logmessage" + Path::openpilot_prefix();
  }

  inline std::string download_cache_root() {
    if (const char *env = getenv("COMMA_CACHE")) {
      return env;
    }
    return "/tmp/comma_download_cache" + Path::openpilot_prefix() + "/";
  }
}  // namespace Path
