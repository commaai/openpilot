#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sstream>
#include <fstream>
#include <memory>
#ifdef __linux__
#include <sys/prctl.h>
#include <sys/syscall.h>
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <sched.h>
#endif // __linux__

void* read_file(const char* path, size_t* out_len) {
  FILE* f = fopen(path, "r");
  if (!f) {
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long f_len = ftell(f);
  rewind(f);

  // malloc one extra byte so the file will always be NULL terminated
  // cl_cached_program_from_file relies on this
  char* buf = (char*)malloc(f_len+1);
  assert(buf);

  size_t num_read = fread(buf, f_len, 1, f);
  fclose(f);

  if (num_read != 1) {
    free(buf);
    return NULL;
  }

  buf[f_len] = '\0';
  if (out_len) {
    *out_len = f_len;
  }

  return buf;
}

int write_file(const char* path, const void* data, size_t size) {
  int fd = open(path, O_WRONLY);
  if (fd == -1) {
    return -1;
  }
  ssize_t n = write(fd, data, size);
  close(fd);
  return (n >= 0 && (size_t)n == size) ? 0 : -1;
}

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

std::string util::read_file(std::string fn) {
  std::ifstream t(fn);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

std::string util::tohex(const uint8_t* buf, size_t buf_size) {
  std::unique_ptr<char[]> hexbuf(new char[buf_size * 2 + 1]);
  for (size_t i = 0; i < buf_size; i++) {
    sprintf(&hexbuf[i * 2], "%02x", buf[i]);
  }
  hexbuf[buf_size * 2] = 0;
  return std::string(hexbuf.get(), hexbuf.get() + buf_size * 2);
}

std::string util::base_name(std::string const& path) {
  size_t pos = path.find_last_of("/");
  if (pos == std::string::npos) return path;
  return path.substr(pos + 1);
}

std::string util::dir_name(std::string const& path) {
  size_t pos = path.find_last_of("/");
  if (pos == std::string::npos) return "";
  return path.substr(0, pos);
}
std::string util::readlink(std::string path) {
  char buff[4096];
  ssize_t len = ::readlink(path.c_str(), buff, sizeof(buff) - 1);
  if (len != -1) {
    buff[len] = '\0';
    return std::string(buff);
  }
  return "";
}
