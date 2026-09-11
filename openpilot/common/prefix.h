#pragma once

#include <cassert>
#include <filesystem>
#include <string>

#include "common/util.h"
#include "common/hardware/hw.h"

class OpenpilotPrefix {
public:
  OpenpilotPrefix(std::string prefix = {}) {
    if (prefix.empty()) {
      prefix = util::random_string(15);
    }
    msgq_path = Path::shm_path() + "/msgq_" + prefix;
    bool ret = util::create_directories(msgq_path, 0777);
    assert(ret);
    setenv("OPENPILOT_PREFIX", prefix.c_str(), 1);
  }

  ~OpenpilotPrefix() {
    std::error_code ec;
    // Params::getParamPath() without params.h: its BOOL/INT/FLOAT enumerators clash with cabana's Win32 typedefs
    auto param_path = Path::params() + "/" + util::getenv("OPENPILOT_PREFIX");
    std::filesystem::remove_all(util::readlink(param_path), ec);  // the temp folder behind the symlink, see params.cc
    std::filesystem::remove_all(param_path, ec);
    if (getenv("COMMA_CACHE") == nullptr) {
      std::filesystem::remove_all(Path::download_cache_root(), ec);
    }
    std::filesystem::remove_all(Path::comma_home(), ec);
    std::filesystem::remove_all(msgq_path, ec);
    unsetenv("OPENPILOT_PREFIX");
  }

private:
  std::string msgq_path;
};
