#pragma once

#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// Protocol shared with shm_queue.py. Each pending filename reserves its full
// payload size; rename publishes a complete record without shared locks or fds.
class ShmQueue {
public:
  explicit ShmQueue(std::string path, size_t capacity = 64 * 1024 * 1024)
      : path(std::move(path)), capacity(capacity) {}

  bool send(const std::string &data) const {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || data.size() > capacity || charge(data.size(), page_size) > capacity) return false;
    const std::string pending = path + "/pending", ready = path + "/ready";
    for (const auto &dir : {path, pending, ready}) {
      if (mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) return false;
    }
    struct timespec now = {};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return false;
    const auto timestamp = static_cast<unsigned long long>(now.tv_sec) * 1000000000ULL + now.tv_nsec;
    char name[128];
    snprintf(name, sizeof(name), "%020llu-%ld-%zu-XXXXXX", timestamp, static_cast<long>(getpid()), data.size());
    std::string temporary = pending + "/" + name;
    const int fd = mkstemp(temporary.data());
    if (fd < 0) return false;
    bool ok = fcntl(fd, F_SETFD, FD_CLOEXEC) == 0;
    size_t used = 0;
    // A file moving from pending to ready can be counted twice, never missed.
    if (ok) ok = count_bytes(pending, page_size, used) && count_bytes(ready, page_size, used);
    size_t offset = 0;
    while (ok && offset < data.size()) {
      const ssize_t written = write(fd, data.data() + offset, data.size() - offset);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) ok = false;
      else offset += written;
    }
    if (close(fd) != 0) ok = false;
    if (ok) ok = rename(temporary.c_str(), (ready + temporary.substr(pending.size())).c_str()) == 0;
    if (!ok) unlink(temporary.c_str());
    return ok;
  }

private:
  static size_t charge(size_t size, size_t page_size) {
    return (1 + (size + page_size - 1) / page_size) * page_size;
  }

  bool count_bytes(const std::string &directory, size_t page_size, size_t &used) const {
    DIR *dir = opendir(directory.c_str());
    if (dir == nullptr) return false;
    bool ok = true;
    while (ok) {
      errno = 0;
      const auto *entry = readdir(dir);
      if (entry == nullptr) {
        ok = errno == 0;
        break;
      }
      std::string_view name(entry->d_name);
      uint64_t fields[3] = {};
      bool valid = true;
      for (int i = 0; i < 3; ++i) {
        const auto end = name.find('-');
        if (end == std::string_view::npos || end == 0 || (i == 0 && end != 20)) {
          valid = false;
          break;
        }
        const auto result = std::from_chars(name.data(), name.data() + end, fields[i]);
        if (result.ec != std::errc() || result.ptr != name.data() + end) {
          valid = false;
          break;
        }
        name.remove_prefix(end + 1);
      }
      if (!valid || fields[1] == 0 || name.empty() || name.find('-') != std::string_view::npos) continue;
      if (fields[2] > capacity || charge(fields[2], page_size) > capacity - used) ok = false;
      else used += charge(fields[2], page_size);
    }
    closedir(dir);
    return ok;
  }

  const std::string path;
  const size_t capacity;
};
