#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
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
      // The pending filename must be visible before opening a budget generation.
      success = reserve_capacity(data.size(), page_size);
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

  bool reserve_capacity(size_t payload_size, size_t page_size) const {
    const size_t pages = storage_cost(payload_size, page_size) / page_size;
    const size_t limit = capacity / page_size;
    const int fd = open((path + "/budget").c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) return false;

    struct stat budget = {};
    bool success = fstat(fd, &budget) == 0 && budget.st_size >= 0 &&
                   static_cast<size_t>(budget.st_size) <= limit && pages <= limit - budget.st_size;
    if (success) {
      // One byte reserves one page. A single append gives this descriptor its
      // own reservation end, even when other producers append concurrently.
      const std::string credits(pages, '\0');
      success = write(fd, credits.data(), credits.size()) == static_cast<ssize_t>(credits.size());
    }
    if (success) {
      const off_t end = lseek(fd, 0, SEEK_CUR);
      success = end >= 0 && static_cast<size_t>(end) <= limit;
    }
    // Failed sends leave conservative credits until the consumer replaces the
    // budget. Never truncate or decrement a generation another producer uses.
    if (close(fd) != 0) {
      success = false;
    }
    return success;
  }

  const std::string path;
  const size_t capacity;
};
