#include "common/util.h"

#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/resource.h>

#include <cassert>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

#ifdef __linux__
#include <sys/prctl.h>
#include <sys/syscall.h>
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sched.h>
#endif  // __linux__

namespace util {

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

int set_core_affinity(std::vector<int> cores) {
#ifdef __linux__
  long tid = syscall(SYS_gettid);
  cpu_set_t cpu;

  CPU_ZERO(&cpu);
  for (const int n : cores) {
    CPU_SET(n, &cpu);
  }
  return sched_setaffinity(tid, sizeof(cpu), &cpu);
#else
  return -1;
#endif
}

int set_file_descriptor_limit(uint64_t limit_val) {
  struct rlimit limit;
  int status;

  if ((status = getrlimit(RLIMIT_NOFILE, &limit)) < 0)
    return status;

  limit.rlim_cur = limit_val;
  if ((status = setrlimit(RLIMIT_NOFILE, &limit)) < 0)
    return status;

  return 0;
}

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

std::map<std::string, std::string> read_files_in_dir(const std::string &path) {
  std::map<std::string, std::string> ret;
  DIR *d = opendir(path.c_str());
  if (!d) return ret;

  struct dirent *de = NULL;
  while ((de = readdir(d))) {
    if (de->d_type != DT_DIR) {
      ret[de->d_name] = util::read_file(path + "/" + de->d_name);
    }
  }

  closedir(d);
  return ret;
}

int write_file(const char* path, const void* data, size_t size, int flags, mode_t mode) {
  int fd = HANDLE_EINTR(open(path, flags, mode));
  if (fd == -1) {
    return -1;
  }
  ssize_t n = HANDLE_EINTR(write(fd, data, size));
  close(fd);
  return (n >= 0 && (size_t)n == size) ? 0 : -1;
}

FILE* safe_fopen(const char* filename, const char* mode) {
  FILE* fp = NULL;
  do {
    fp = fopen(filename, mode);
  } while ((nullptr == fp) && (errno == EINTR));
  return fp;
}

size_t safe_fwrite(const void* ptr, size_t size, size_t count, FILE* stream) {
  size_t written = 0;
  do {
    size_t ret = ::fwrite((void*)((char *)ptr + written * size), size, count - written, stream);
    if (ret == 0 && errno != EINTR) break;
    written += ret;
  } while (written != count);
  return written;
}

int safe_fflush(FILE *stream) {
  int ret = EOF;
  do {
    ret = fflush(stream);
  } while ((EOF == ret) && (errno == EINTR));
  return ret;
}

int safe_ioctl(int fd, unsigned long request, void *argp) {
  int ret;
  do {
    ret = ioctl(fd, request, argp);
  } while ((ret == -1) && (errno == EINTR));
  return ret;
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
  struct stat st = {};
  return stat(fn.c_str(), &st) != -1;
}

static bool createDirectory(std::string dir, mode_t mode) {
  auto verify_dir = [](const std::string& dir) -> bool {
    struct stat st = {};
    return (stat(dir.c_str(), &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR);
  };
  // remove trailing /'s
  while (dir.size() > 1 && dir.back() == '/') {
    dir.pop_back();
  }
  // try to mkdir this directory
  if (mkdir(dir.c_str(), mode) == 0) return true;
  if (errno == EEXIST) return verify_dir(dir);
  if (errno != ENOENT) return false;

  // mkdir failed because the parent dir doesn't exist, so try to create it
  size_t slash = dir.rfind('/');
  if ((slash == std::string::npos || slash < 1) ||
      !createDirectory(dir.substr(0, slash), mode)) {
    return false;
  }

  // try again
  if (mkdir(dir.c_str(), mode) == 0) return true;
  return errno == EEXIST && verify_dir(dir);
}

bool create_directories(const std::string& dir, mode_t mode) {
  if (dir.empty()) return false;
  return createDirectory(dir, mode);
}

std::string getenv(const char* key, std::string default_val) {
  const char* val = ::getenv(key);
  return val ? val : default_val;
}

int getenv(const char* key, int default_val) {
  const char* val = ::getenv(key);
  return val ? atoi(val) : default_val;
}

float getenv(const char* key, float default_val) {
  const char* val = ::getenv(key);
  return val ? atof(val) : default_val;
}

std::string hexdump(const uint8_t* in, const size_t size) {
  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  for (size_t i = 0; i < size; i++) {
    ss << std::setw(2) << static_cast<unsigned int>(in[i]);
  }
  return ss.str();
}

int random_int(int min, int max) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
  return dist(rng);
}

std::string random_string(std::string::size_type length) {
  const std::string chrs = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::mt19937 rg{std::random_device{}()};
  std::uniform_int_distribution<std::string::size_type> pick(0, chrs.length() - 1);
  std::string s;
  s.reserve(length);
  while (length--) {
    s += chrs[pick(rg)];
  }
  return s;
}

bool starts_with(const std::string &s1, const std::string &s2) {
  return strncmp(s1.c_str(), s2.c_str(), s2.size()) == 0;
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         strcmp(s.c_str() + (s.size() - suffix.size()), suffix.c_str()) == 0;
}

std::string strip(const std::string &str) {
  auto should_trim = [](unsigned char ch) {
    // trim whitespace or a null character
    return std::isspace(ch) || ch == '\0';
  };

  size_t start = 0;
  while (start < str.size() && should_trim(static_cast<unsigned char>(str[start]))) {
    start++;
  }

  if (start == str.size()) {
    return "";
  }

  size_t end = str.size() - 1;
  while (end > 0 && should_trim(static_cast<unsigned char>(str[end]))) {
    end--;
  }

  return str.substr(start, end - start + 1);
}

std::string check_output(const std::string& command) {
  char buffer[128];
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);

  if (!pipe) {
    return "";
  }

  while (fgets(buffer, std::size(buffer), pipe.get()) != nullptr) {
    result += std::string(buffer);
  }

  return result;
}

}  // namespace util
