#pragma once

#include <cassert>
#include <string>

#include "common/params.h"
#include "common/util.h"
#include "system/hardware/hw.h"

class OpenpilotPrefix {
public:
  OpenpilotPrefix(std::string prefix = {}) {
    if (prefix.empty()) {
      prefix = util::random_string(15);
    }
    msgq_path = Path::shm_path() + "/" + prefix;
    bool ret = util::create_directories(msgq_path, 0777);
    assert(ret);
    setenv("OPENPILOT_PREFIX", prefix.c_str(), 1);
  }

  ~OpenpilotPrefix() {
    auto param_path = Params().getParamPath();
    if (util::file_exists(param_path)) {
      std::string real_path = util::readlink(param_path);
      system(util::string_format("rm %s -rf", real_path.c_str()).c_str());
      unlink(param_path.c_str());
    }
    if (getenv("COMMA_CACHE") == nullptr) {
      system(util::string_format("rm %s -rf", Path::download_cache_root().c_str()).c_str());
    }
    system(util::string_format("rm %s -rf", Path::comma_home().c_str()).c_str());
    system(util::string_format("rm %s -rf", msgq_path.c_str()).c_str());
    unsetenv("OPENPILOT_PREFIX");
  }

private:
  std::string msgq_path;
};
