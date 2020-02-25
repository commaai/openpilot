#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sched.h>
#endif

void* read_file(const char* path, size_t* out_len) {
  FILE* f = fopen(path, "r");
  if (!f) {
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long f_len = ftell(f);
  rewind(f);

  char* buf = (char*)calloc(f_len, 1);
  assert(buf);

  size_t num_read = fread(buf, f_len, 1, f);
  fclose(f);

  if (num_read != 1) {
    free(buf);
    return NULL;
  }

  if (out_len) {
    *out_len = f_len;
  }

  return buf;
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

