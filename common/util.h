#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// keep trying if x gets interrupted by a signal
#define HANDLE_EINTR(x)                                        \
  ({                                                           \
    decltype(x) ret_;                                          \
    int try_cnt = 0;                                           \
    do {                                                       \
      ret_ = (x);                                              \
    } while (ret_ == -1 && errno == EINTR && try_cnt++ < 100); \
    ret_;                                                       \
  })

#ifndef sighandler_t
typedef void (*sighandler_t)(int sig);
#endif

const double MILE_TO_KM = 1.609344;
const double KM_TO_MILE = 1. / MILE_TO_KM;
const double MS_TO_KPH = 3.6;
const double MS_TO_MPH = MS_TO_KPH * KM_TO_MILE;
const double METER_TO_MILE = KM_TO_MILE / 1000.0;
const double METER_TO_FOOT = 3.28084;

#define ALIGNED_SIZE(x, align) (((x) + (align)-1) & ~((align)-1))

namespace util {

void set_thread_name(const char* name);
int set_realtime_priority(int level);
int set_core_affinity(std::vector<int> cores);
int set_file_descriptor_limit(uint64_t limit);

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

template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

std::string getenv(const char* key, std::string default_val = "");
int getenv(const char* key, int default_val);
float getenv(const char* key, float default_val);

std::string hexdump(const uint8_t* in, const size_t size);
bool starts_with(const std::string &s1, const std::string &s2);
bool ends_with(const std::string &s, const std::string &suffix);

// ***** random helpers *****
int random_int(int min, int max);
std::string random_string(std::string::size_type length);

// **** file helpers *****
std::string read_file(const std::string& fn);
std::map<std::string, std::string> read_files_in_dir(const std::string& path);
int write_file(const char* path, const void* data, size_t size, int flags = O_WRONLY, mode_t mode = 0664);

FILE* safe_fopen(const char* filename, const char* mode);
size_t safe_fwrite(const void * ptr, size_t size, size_t count, FILE * stream);
int safe_fflush(FILE *stream);
int safe_ioctl(int fd, unsigned long request, void *argp);

std::string readlink(const std::string& path);
bool file_exists(const std::string& fn);
bool create_directories(const std::string &dir, mode_t mode);

std::string check_output(const std::string& command);

inline void sleep_for(const int milliseconds) {
  if (milliseconds > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
  }
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
  }
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
  FirstOrderFilter(float x0, float ts, float dt, bool initialized = true) {
    k_ = (dt / ts) / (1.0 + dt / ts);
    x_ = x0;
    initialized_ = initialized;
  }
  inline float update(float x) {
    if (initialized_) {
      x_ = (1. - k_) * x_ + k_ * x;
    } else {
      initialized_ = true;
      x_ = x;
    }
    return x_;
  }
  inline void reset(float x) { x_ = x; }
  inline float x(){ return x_; }

private:
  float x_, k_;
  bool initialized_;
};

template<typename T>
void update_max_atomic(std::atomic<T>& max, T const& value) {
  T prev = max;
  while (prev < value && !max.compare_exchange_weak(prev, value)) {}
}

typedef struct Rect {
  int x;
  int y;
  int w;
  int h;
} Rect;
