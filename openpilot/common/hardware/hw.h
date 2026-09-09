#pragma once

#include <cstdint>
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

  inline std::string home() {
    return util::getenv("USERPROFILE", util::getenv("HOME"));  // Python's Path.home() ignores HOME on Windows
  }

  inline std::string comma_home() {
    return home() + "/.comma" + Path::openpilot_prefix();
  }

  inline std::string log_root() {
    if (const char *env = getenv("LOG_ROOT")) {
      return env;
    }
    return Hardware::PC() ? Path::comma_home() + "/media/0/realdata" : "/data/media/0/realdata";
  }

  inline std::string params() {
    return util::getenv("PARAMS_ROOT", Hardware::PC() ? (Path::comma_home() + "/params") : "/data/params");
  }

  inline std::string rsa_file() {
    return Hardware::PC() ? Path::comma_home() + "/persist/comma/id_rsa" : "/persist/comma/id_rsa";
  }

  inline std::string swaglog_ipc() {
#ifdef _WIN32
    // libzmq has no ipc:// transport on MinGW: derive a loopback port from the prefix (FNV-1a, mirrored in hw.py)
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : Path::openpilot_prefix()) {
      h ^= c;
      h *= 1099511628211ULL;
    }
    return "tcp://127.0.0.1:" + std::to_string(26000 + h % 1000);
#else
    return "ipc:///tmp/logmessage" + Path::openpilot_prefix();
#endif
  }

  inline std::string tmp_dir() {
    return util::getenv("TEMP", "/tmp");  // hw.TMP_DIR
  }

  inline std::string download_cache_root() {
    if (const char *env = getenv("COMMA_CACHE")) {
      return env;
    }
    return tmp_dir() + "/comma_download_cache" + Path::openpilot_prefix() + "/";
  }

 inline std::string shm_path() {
    #ifdef __APPLE__
     return"/tmp";
    #elif defined(_WIN32)
     return tmp_dir();
    #else
     return "/dev/shm";
    #endif
 }
}  // namespace Path
