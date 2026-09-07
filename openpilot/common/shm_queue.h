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
    if (page_size <= 0 || data.size() > capacity || storage_cost(data.size(), page_size) > capacity) return false;

    const std::string pending_dir = path + "/pending";
    const std::string ready_dir = path + "/ready";
    for (const auto &directory : {path, pending_dir, ready_dir}) {
      if (mkdir(directory.c_str(), 0700) != 0 && errno != EEXIST) return false;
    }

    struct timespec now = {};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return false;
    const auto timestamp = static_cast<unsigned long long>(now.tv_sec) * 1000000000ULL + now.tv_nsec;
    char name[128];
    snprintf(name, sizeof(name), "%020llu-%ld-%zu-XXXXXX", timestamp, static_cast<long>(getpid()), data.size());
    std::string pending_path = pending_dir + "/" + name;
    const int fd = mkstemp(pending_path.data());
    if (fd < 0) return false;

    bool success = fcntl(fd, F_SETFD, FD_CLOEXEC) == 0;
    if (success) {
      size_t used = 0;
      // Reserve before writing. Scan pending first so publication can only overcount.
      success = check_capacity(pending_dir, page_size, used) && check_capacity(ready_dir, page_size, used);
    }
    if (success) {
      success = write_all(fd, data);
    }
    // Always close, including when the capacity check or write failed.
    if (close(fd) != 0) {
      success = false;
    }
    if (success) {
      const std::string filename = pending_path.substr(pending_dir.size() + 1);
      const std::string ready_path = ready_dir + "/" + filename;
      success = rename(pending_path.c_str(), ready_path.c_str()) == 0;
    }
    if (!success) {
      unlink(pending_path.c_str());
    }
    return success;
  }

private:
  static size_t storage_cost(size_t payload_size, size_t page_size) {
    const size_t payload_pages = (payload_size + page_size - 1) / page_size;
    return (payload_pages + 1) * page_size;  // One extra page for file metadata.
  }

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

  // Consume a decimal number and its trailing '-' from the filename.
  static bool read_field(std::string_view &name, uint64_t &value) {
    const size_t separator = name.find('-');
    if (separator == std::string_view::npos || separator == 0) return false;
    const auto result = std::from_chars(name.data(), name.data() + separator, value);
    if (result.ec != std::errc() || result.ptr != name.data() + separator) return false;
    name.remove_prefix(separator + 1);
    return true;
  }

  static bool parse_payload_size(std::string_view name, uint64_t &payload_size) {
    // Filename: <20-digit monotonic timestamp>-<pid>-<payload bytes>-<random suffix>.
    if (name.find('-') != 20) return false;
    uint64_t timestamp = 0;
    uint64_t pid = 0;
    if (!read_field(name, timestamp) || !read_field(name, pid) || !read_field(name, payload_size)) return false;
    return pid != 0 && !name.empty() && name.find('-') == std::string_view::npos;
  }

  bool check_capacity(const std::string &directory, size_t page_size, size_t &used) const {
    DIR *dir = opendir(directory.c_str());
    if (dir == nullptr) return false;

    bool success = true;
    while (true) {
      errno = 0;
      const auto *entry = readdir(dir);
      if (entry == nullptr) {
        success = errno == 0;
        break;
      }
      uint64_t payload_size = 0;
      if (!parse_payload_size(entry->d_name, payload_size)) continue;
      if (payload_size > capacity || storage_cost(payload_size, page_size) > capacity - used) {
        success = false;
        break;
      }
      used += storage_cost(payload_size, page_size);
    }
    closedir(dir);
    return success;
  }

  const std::string path;
  const size_t capacity;
};
