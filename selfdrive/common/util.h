#pragma once

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <csignal>
#include <memory>
#include <atomic>
#include <string>
#include <thread>
#include <chrono>

#ifndef sighandler_t
typedef void (*sighandler_t)(int sig);
#endif

#undef ALIGN
#define ALIGN(x, align) (((x) + (align)-1) & ~((align)-1))

// Reads a file into a newly allocated buffer.
//
// Returns NULL on failure, otherwise the NULL-terminated file contents.
// The result must be freed by the caller.
void* read_file(const char* path, size_t* out_len);
int write_file(const char* path, const void* data, size_t size);

void set_thread_name(const char* name);

int set_realtime_priority(int level);
int set_core_affinity(int core);

struct unique_fd {
  unique_fd(int fd = -1) : fd_(fd) {}
  unique_fd& operator=(unique_fd&& uf) {
    fd_ = uf.fd_;
    uf.fd_ = -1;
    return *this;
  }
  ~unique_fd() {
    if (fd_ != -1) close(fd_);
  }
  operator int() const { return fd_; }
  int fd_;
};

class ExitHandler {
public:
  ExitHandler() {
    std::signal(SIGINT, (sighandler_t)set_do_exit);
    std::signal(SIGTERM, (sighandler_t)set_do_exit);
  };
  inline operator bool() { return do_exit; }
  inline ExitHandler& operator=(bool v) {
    do_exit = v;
    return *this;
  }
private:
  static void set_do_exit(int sig) { do_exit = true; }
  inline static std::atomic<bool> do_exit = false;
};

namespace util {

inline bool starts_with(std::string s, std::string prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
}

template <typename... Args>
inline std::string string_format(const std::string& format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

std::string read_file(std::string fn);
std::string tohex(const uint8_t* buf, size_t buf_size);
std::string base_name(std::string const& path);
std::string dir_name(std::string const& path);
std::string readlink(std::string path);
inline std::string getenv_default(const char* env_var, const char * suffix, const char* default_val) {
  const char* env_val = getenv(env_var);
  if (env_val != NULL){
    return std::string(env_val) + std::string(suffix);
  } else {
    return std::string(default_val);
  }
}

inline void sleep_for(const int milliseconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}
}  // namespace util
