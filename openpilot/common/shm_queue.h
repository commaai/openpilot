#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Wire layout and locking must match shm_queue.py. Callers serialize threads.
class ShmQueue {
public:
  explicit ShmQueue(std::string path, size_t capacity = 64 * 1024 * 1024)
      : path(std::move(path)), capacity(capacity) {}
  ~ShmQueue() { close_queue(); }
  ShmQueue(const ShmQueue &) = delete;
  ShmQueue &operator=(const ShmQueue &) = delete;

  bool send(const std::string &data) {
    if (data.size() + sizeof(uint32_t) >= capacity || !open_queue()) return false;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) return false;
    auto *positions = reinterpret_cast<uint64_t *>(mem);
    size_t read = positions[0], write = positions[1];
    bool fits = data.size() + sizeof(uint32_t) <= (read + capacity - write - 1) % capacity;
    if (fits) {
      uint32_t size = data.size();
      copy(write, &size, sizeof(size));
      copy((write + sizeof(size)) % capacity, data.data(), size);
      // Publish only after the complete record has been copied.
      __atomic_store_n(&positions[1], (write + sizeof(size) + size) % capacity, __ATOMIC_RELEASE);
    }
    flock(fd, LOCK_UN);
    return fits;
  }

private:
  void close_queue() {
    if (mem != nullptr) munmap(mem, 16 + capacity);
    if (fd >= 0) close(fd);
    mem = nullptr;
    fd = -1;
  }

  bool open_queue() {
    if (pid != getpid()) close_queue();  // flock descriptors must not be shared after fork.
    if (mem != nullptr) return true;
    pid = getpid();
    fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    struct stat st = {};
    bool ok = flock(fd, LOCK_EX | LOCK_NB) == 0 && fstat(fd, &st) == 0;
    if (ok && st.st_size == 0) ok = ftruncate(fd, 16 + capacity) == 0;
    else if (ok) ok = st.st_size == static_cast<off_t>(16 + capacity);
    if (ok) {
      void *mapping = mmap(nullptr, 16 + capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (mapping != MAP_FAILED) mem = static_cast<char *>(mapping);
    }
    flock(fd, LOCK_UN);
    if (mem == nullptr) close_queue();
    return mem != nullptr;
  }

  void copy(size_t pos, const void *data, size_t size) {
    size_t first = std::min(size, capacity - pos);
    memcpy(mem + 16 + pos, data, first);
    memcpy(mem + 16, static_cast<const char *>(data) + first, size - first);
  }

  std::string path;
  size_t capacity;
  pid_t pid = 0;
  int fd = -1;
  char *mem = nullptr;
};
