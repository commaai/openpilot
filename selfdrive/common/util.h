#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <map>
#include <memory>
#include <string>
#include <thread>

#ifndef sighandler_t
typedef void (*sighandler_t)(int sig);
#endif

void set_thread_name(const char* name);

int set_realtime_priority(int level);
int set_core_affinity(int core);

namespace util {

// ***** Time helpers *****
struct tm get_time();
bool time_valid(struct tm sys_time);

// ***** math helpers *****

// map x from [a1, a2] to [b1, b2]
template <typename T>
T map_val(T x, T a1, T a2, T b1, T b2) {
  x = std::clamp(x, a1, a2);
  T ra = a2 - a1;
  T rb = b2 - b1;
  return (x - a1) * rb / ra + b1;
}

// ***** string helpers *****

inline bool starts_with(const std::string& s, const std::string& prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
}

template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

std::string getenv_default(const char* env_var, const char* suffix, const char* default_val);
std::string tohex(const uint8_t* buf, size_t buf_size);
std::string hexdump(const std::string& in);
std::string base_name(std::string const& path);
std::string dir_name(std::string const& path);
bool is_valid_dongle_id(std::string const& dongle_id);

// **** file fhelpers *****
std::string read_file(const std::string& fn);
int read_files_in_dir(const std::string& path, std::map<std::string, std::string>* contents);
int write_file(const char* path, const void* data, size_t size, int flags = O_WRONLY, mode_t mode = 0777);
std::string readlink(const std::string& path);
bool file_exists(const std::string& fn);

inline void sleep_for(const int milliseconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

}  // namespace util

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
  inline static std::atomic<int> signal = 0;
  inline operator bool() { return do_exit; }
  inline ExitHandler& operator=(bool v) {
    signal = 0;
    do_exit = v;
    return *this;
  }
private:
  static void set_do_exit(int sig) {
#ifndef __APPLE__
    power_failure = (sig == SIGPWR);
#endif
    signal = sig;
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
