#include <errno.h>
#include <sstream>

#include "common/util.h"

#ifdef __linux__
#include <sys/prctl.h>
#include <sys/syscall.h>
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sched.h>
#endif // __linux__

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
  std::ifstream ifs(fn, std::ios::binary | std::ios::ate);
  if (ifs) {
    std::ifstream::pos_type pos = ifs.tellg();
    if (pos != std::ios::beg) {
      std::string result;
      result.resize(pos);
      ifs.seekg(0, std::ios::beg);
      ifs.read(result.data(), pos);
      if (ifs) {
        return result;
      }
    }
  }
  ifs.close();

  // fallback for files created on read, e.g. procfs
  std::ifstream f(fn);
  std::stringstream buffer;
  buffer << f.rdbuf();
  return buffer.str();
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

}  // namespace util
