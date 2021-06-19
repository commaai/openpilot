#include "selfdrive/common/util.h"

#include <cassert>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef __linux__
#include <sys/prctl.h>
#include <sys/syscall.h>
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sched.h>
#endif  // __linux__

void set_thread_name(const char* name) {
#ifdef __linux__
  // pthread_setname_np is dumb (fails instead of truncates)
  prctl(PR_SET_NAME, (unsigned long)name, 0, 0, 0);
#endif
}

int set_realtime_priority(int level) {
#ifdef __linux__
  long tid = syscall(SYS_gettid);

  // should match python using chrt
  struct sched_param sa;
  memset(&sa, 0, sizeof(sa));
  sa.sched_priority = level;
  return sched_setscheduler(tid, SCHED_FIFO, &sa);
#else
  return -1;
#endif
}

int set_core_affinity(int core) {
#ifdef __linux__
  long tid = syscall(SYS_gettid);
  cpu_set_t rt_cpu;

  CPU_ZERO(&rt_cpu);
  CPU_SET(core, &rt_cpu);
  return sched_setaffinity(tid, sizeof(rt_cpu), &rt_cpu);
#else
  return -1;
#endif
}

namespace util {

std::string read_file(const std::string& fn) {
  std::ifstream f(fn, std::ios::binary | std::ios::in);
  if (f.is_open()) {
    f.seekg(0, std::ios::end);
    int size = f.tellg();
    if (f.good() && size > 0) {
      std::string result(size, '\0');
      f.seekg(0, std::ios::beg);
      f.read(result.data(), size);
      // return either good() or has reached end-of-file (e.g. /sys/power/wakeup_count)
      if (f.good() || f.eof()) {
        result.resize(f.gcount());
        return result;
      }
    }
    // fallback for files created on read, e.g. procfs
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
  }
  return std::string();
}

int read_files_in_dir(const std::string &path, std::map<std::string, std::string> *contents) {
  DIR *d = opendir(path.c_str());
  if (!d) return -1;

  struct dirent *de = NULL;
  while ((de = readdir(d))) {
    if (isalnum(de->d_name[0])) {
      (*contents)[de->d_name] = util::read_file(path + "/" + de->d_name);
    }
  }

  closedir(d);
  return 0;
}

int write_file(const char* path, const void* data, size_t size, int flags, mode_t mode) {
  int fd = open(path, flags, mode);
  if (fd == -1) {
    return -1;
  }
  ssize_t n = write(fd, data, size);
  close(fd);
  return (n >= 0 && (size_t)n == size) ? 0 : -1;
}

std::string readlink(const std::string &path) {
  char buff[4096];
  ssize_t len = ::readlink(path.c_str(), buff, sizeof(buff)-1);
  if (len != -1) {
    buff[len] = '\0';
    return std::string(buff);
  }
  return "";
}

bool file_exists(const std::string& fn) {
  std::ifstream f(fn);
  return f.good();
}

std::string getenv_default(const char* env_var, const char * suffix, const char* default_val) {
  const char* env_val = getenv(env_var);
  if (env_val != NULL) {
    return std::string(env_val) + std::string(suffix);
  } else {
    return std::string(default_val);
  }
}

std::string tohex(const uint8_t *buf, size_t buf_size) {
  std::unique_ptr<char[]> hexbuf(new char[buf_size * 2 + 1]);
  for (size_t i = 0; i < buf_size; i++) {
    sprintf(&hexbuf[i * 2], "%02x", buf[i]);
  }
  hexbuf[buf_size * 2] = 0;
  return std::string(hexbuf.get(), hexbuf.get() + buf_size * 2);
}

std::string hexdump(const std::string& in) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < in.size(); i++) {
        ss << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(in[i]));
    }
    return ss.str();
}

std::string base_name(std::string const &path) {
  size_t pos = path.find_last_of("/");
  if (pos == std::string::npos) return path;
  return path.substr(pos + 1);
}

std::string dir_name(std::string const &path) {
  size_t pos = path.find_last_of("/");
  if (pos == std::string::npos) return "";
  return path.substr(0, pos);
}

bool is_valid_dongle_id(std::string const& dongle_id) {
  return !dongle_id.empty() && dongle_id != "UnregisteredDevice";
}

struct tm get_time() {
  time_t rawtime;
  time(&rawtime);

  struct tm sys_time;
  gmtime_r(&rawtime, &sys_time);

  return sys_time;
}

bool time_valid(struct tm sys_time) {
  int year = 1900 + sys_time.tm_year;
  int month = 1 + sys_time.tm_mon;
  return (year > 2020) || (year == 2020 && month >= 10);
}

}  // namespace util
