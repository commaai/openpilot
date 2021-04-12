#pragma once

#include <cstdio>
#include <csignal>
#include <cassert>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>

#include <string>
#include <memory>
#include <atomic>
#include <sstream>
#include <fstream>
#include <thread>
#include <chrono>
#include <algorithm>

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
int write_file(const char* path, const void* data, size_t size, int flags=O_WRONLY, mode_t mode=0777);

void set_thread_name(const char* name);

int set_realtime_priority(int level);
int set_core_affinity(int core);

namespace util {

// ***** math helpers *****

// map x from [a1, a2] to [b1, b2]
template<typename T>
T map_val(T x, T a1, T a2, T b1, T b2) {
  x = std::clamp(x, a1, a2);
  T ra = a2 - a1;
  T rb = b2 - b1;
  return (x - a1)*rb / ra + b1;
}

// ***** string helpers *****

inline bool starts_with(const std::string &s, const std::string &prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
}

template <typename... Args>
inline std::string string_format(const std::string& format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

inline std::string read_file(const std::string &fn) {
  std::ifstream t(fn);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

inline std::string tohex(const uint8_t* buf, size_t buf_size) {
  std::unique_ptr<char[]> hexbuf(new char[buf_size*2+1]);
  for (size_t i=0; i < buf_size; i++) {
    sprintf(&hexbuf[i*2], "%02x", buf[i]);
  }
  hexbuf[buf_size*2] = 0;
  return std::string(hexbuf.get(), hexbuf.get() + buf_size*2);
}

inline std::string base_name(std::string const & path) {
  size_t pos = path.find_last_of("/");
  if (pos == std::string::npos) return path;
  return path.substr(pos + 1);
}

inline std::string dir_name(std::string const & path) {
  size_t pos = path.find_last_of("/");
  if (pos == std::string::npos) return "";
  return path.substr(0, pos);
}

inline std::string readlink(const std::string &path) {
  char buff[4096];
  ssize_t len = ::readlink(path.c_str(), buff, sizeof(buff)-1);
  if (len != -1) {
    buff[len] = '\0';
    return std::string(buff);
  }
  return "";
}

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

inline bool file_exists(const std::string& fn) {
  std::ifstream f(fn);
  return f.good();
}

}

class ExitHandler {
public:
  ExitHandler() {
    std::signal(SIGINT, (sighandler_t)set_do_exit);
    std::signal(SIGTERM, (sighandler_t)set_do_exit);

#ifndef __APPLE__
    std::signal(SIGPWR, (sighandler_t)set_do_exit);
#endif
  };
  inline static std::atomic<bool> power_failure = false;
  inline operator bool() { return do_exit; }
  inline ExitHandler& operator=(bool v) {
    do_exit = v;
    return *this;
  }
private:
  static void set_do_exit(int sig) {
#ifndef __APPLE__
    power_failure = (sig == SIGPWR);
#endif
    do_exit = true;
  }
  inline static std::atomic<bool> do_exit = false;
};

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

class FirstOrderFilter {
public:
  FirstOrderFilter(float x0, float ts, float dt) {
    k_ = (dt / ts) / (1.0 + dt / ts);
    x_ = x0;
  }
  inline float update(float x) {
    x_ = (1. - k_) * x_ + k_ * x;
    return x_;
  }
  inline void reset(float x) { x_ = x; }

private:
  float x_, k_;
};
