#pragma once

#include <cassert>
#include <random>
#include <string>

#include "common/params.h"
#include "common/util.h"

class OpenpilotPrefix {
public:
  OpenpilotPrefix(std::string prefix = {}) {
    if (prefix.empty()) {
      prefix = random_string(15);
    }
    msgq_path = "/dev/shm/" + prefix;
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
    system(util::string_format("rm %s -rf", msgq_path.c_str()).c_str());
    unsetenv("OPENPILOT_PREFIX");
  }

  inline static std::string random_string(std::string::size_type length) {
    const char *chrs = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::mt19937 rg{std::random_device{}()};
    std::uniform_int_distribution<std::string::size_type> pick(0, sizeof(chrs) - 2);
    std::string s;
    s.reserve(length);
    while (length--)
      s += chrs[pick(rg)];
    return s;
  }

private:
  std::string msgq_path;
};
