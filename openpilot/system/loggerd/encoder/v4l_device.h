#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <filesystem>
#include <string>
#include <system_error>

#include "common/util.h"

// upstream Venus registers its encoder and decoder video nodes in probe order, so they're found by name.
// returns -1 if there's none. the downstream kernel's msm_vidc nodes have fixed paths instead.
inline int open_video_device(const std::string &name) {
  std::error_code ec;
  for (std::filesystem::directory_iterator it("/sys/class/video4linux", ec), end; !ec && it != end; it.increment(ec)) {
    if (util::strip(util::read_file(it->path() / "name")) == name) {
      return HANDLE_EINTR(open(("/dev/" + it->path().filename().string()).c_str(), O_RDWR | O_NONBLOCK));
    }
  }
  return -1;
}
