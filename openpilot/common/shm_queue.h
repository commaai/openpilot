#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>

#include "common/util.h"

// Protocol shared with shm_queue.py. A slot symlink claims one message;
// rename publishes its complete payload without shared locks or persistent fds.
class ShmQueue {
public:
  static constexpr size_t MAX_MESSAGE_SIZE = 128 * 1024;
  static constexpr int SLOT_COUNT = 1024;

  explicit ShmQueue(std::string path) : path(std::move(path)) {}

  bool send(const std::string &data) const {
    if (data.size() > MAX_MESSAGE_SIZE) return false;

    const std::string pending_dir = path + "/pending";
    const std::string ready_dir = path + "/ready";
    const std::string slots_dir = path + "/slots";
    for (const auto &directory : {path, pending_dir, ready_dir, slots_dir}) {
      if (mkdir(directory.c_str(), 0700) != 0 && errno != EEXIST) return false;
    }

    struct statvfs filesystem = {};
    const size_t required_space = data.size() + 128 * 1024 * 1024;
    if (statvfs(path.c_str(), &filesystem) != 0 || filesystem.f_bavail * filesystem.f_frsize < required_space) return false;

    struct timespec now = {};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return false;
    const auto timestamp = static_cast<unsigned long long>(now.tv_sec) * 1000000000ULL + now.tv_nsec;
    std::string filename;
    std::string slot_path;
    bool claimed = false;
    for (int attempt = 0; attempt < 8; ++attempt) {
      const int slot = util::random_int(0, SLOT_COUNT - 1);
      char name[128];
      snprintf(name, sizeof(name), "%020llu-%ld-%zu-%d", timestamp, static_cast<long>(getpid()), data.size(), slot);
      filename = name;
      slot_path = slots_dir + "/" + std::to_string(slot);
      if (symlink(filename.c_str(), slot_path.c_str()) == 0) {
        claimed = true;
        break;
      }
      if (errno != EEXIST) return false;
    }
    if (!claimed) return false;
    const std::string pending_path = pending_dir + "/" + filename;
    const int fd = open(pending_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
      unlink(slot_path.c_str());
      return false;
    }
    bool success = write_all(fd, data);
    // Always close, including when the write failed.
    if (close(fd) != 0) {
      success = false;
    }
    if (success) {
      const std::string ready_path = ready_dir + "/" + filename;
      success = rename(pending_path.c_str(), ready_path.c_str()) == 0;
    }
    if (!success) {
      unlink(pending_path.c_str());
      unlink(slot_path.c_str());
    }
    return success;
  }

private:
  static bool write_all(int fd, const std::string &data) {
    size_t offset = 0;
    while (offset < data.size()) {
      const ssize_t written = write(fd, data.data() + offset, data.size() - offset);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) return false;
      offset += written;
    }
    return true;
  }

  const std::string path;
};
