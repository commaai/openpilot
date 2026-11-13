// LD_PRELOAD shim for test_onroad: makes every file operation on disk (ext4) slow,
// so code that blocks on the disk in a realtime loop fails the timing tests every run,
// instead of only when the page cache misses or writeback stalls the filesystem.
//
// SLOWDISK_FLAG: delays are on while this file exists
// SLOWDISK_LOG: one line per delayed operation
// SLOWDISK_EXEMPT: comma-separated process names whose job is disk IO

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#include <time.h>
#include <unistd.h>

#define DELAY_MS 100
#define EXT4_SUPER_MAGIC 0xEF53
#define MAX_FDS 4096

#define REAL(fn) static __typeof__(fn) *real_##fn; if (!real_##fn) real_##fn = dlsym(RTLD_NEXT, #fn)

static char *fd_paths[MAX_FDS];  // path of each open fd on disk
static __thread int in_hook;

static ssize_t read_proc(const char *path, char *buf, size_t n) {
  int fd = syscall(SYS_openat, AT_FDCWD, path, O_RDONLY | O_CLOEXEC);
  ssize_t r = fd < 0 ? -1 : syscall(SYS_read, fd, buf, n - 1);
  if (fd >= 0) syscall(SYS_close, fd);
  buf[r > 0 ? r : 0] = '\0';
  return r;
}

// re-checked every second and after fork, since processes are renamed with setproctitle after fork.
// no lock: a fork while another thread holds it would deadlock the child, and racing here is harmless
static int enabled(void) {
  static double last_check = -10.;
  static pid_t last_pid;
  static int on, exempt;

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double now = ts.tv_sec + ts.tv_nsec * 1e-9;
  if (now - last_check > 1. || getpid() != last_pid) {
    last_check = now;
    last_pid = getpid();
    const char *flag = getenv("SLOWDISK_FLAG");
    on = flag && syscall(SYS_faccessat, AT_FDCWD, flag, F_OK) == 0;

    char name[256], list[1024], needle[260];
    read_proc("/proc/self/cmdline", name, sizeof(name));
    snprintf(list, sizeof(list), ",%s,", getenv("SLOWDISK_EXEMPT") ?: "");
    snprintf(needle, sizeof(needle), ",%s,", name);
    exempt = strstr(list, needle) != NULL;
  }
  return on && !exempt;
}

// true if path (or its parent, when path doesn't exist) is on ext4
static int on_disk(int dirfd, const char *path) {
  if (!path) return 0;
  if (dirfd != AT_FDCWD && path[0] != '/') {
    return dirfd >= 0 && dirfd < MAX_FDS && fd_paths[dirfd];
  }
  struct statfs s;
  if (syscall(SYS_statfs, path, &s) != 0) {
    char dir[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash == NULL) snprintf(dir, sizeof(dir), ".");
    else if (slash == dir) dir[1] = '\0';
    else *slash = '\0';
    if (syscall(SYS_statfs, dir, &s) != 0) return 0;
  }
  return s.f_type == EXT4_SUPER_MAGIC;
}

static void slow(const char *op, const char *path) {
  if (in_hook || !enabled()) return;
  in_hook = 1;

  const char *log = getenv("SLOWDISK_LOG");
  if (log) {
    char name[256], thread[32] = "main", line[1024];
    read_proc("/proc/self/cmdline", name, sizeof(name));
    if (syscall(SYS_gettid) != getpid()) {
      read_proc("/proc/thread-self/comm", thread, sizeof(thread));
      thread[strcspn(thread, "\n")] = '\0';
    }
    int n = snprintf(line, sizeof(line), "%s\t%s\t%s\t%s\n", name, thread, op, path);
    int fd = syscall(SYS_openat, AT_FDCWD, log, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
    if (fd >= 0) {
      syscall(SYS_write, fd, line, n < (int)sizeof(line) ? n : (int)sizeof(line) - 1);
      syscall(SYS_close, fd);
    }
  }

  // msgq wakes readers with signals, so sleep through EINTR
  struct timespec ts = {0, DELAY_MS * 1000000L};
  while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {}
  in_hook = 0;
}

static void opened(int fd, int dirfd, const char *path) {
  if (fd >= 0 && fd < MAX_FDS) {
    free(fd_paths[fd]);
    fd_paths[fd] = NULL;
  }
  if (on_disk(dirfd, path)) {
    if (fd >= 0 && fd < MAX_FDS) fd_paths[fd] = strdup(path);
    slow("open", path);
  }
}

static int fd_on_disk(int fd) {
  return fd >= 0 && fd < MAX_FDS && fd_paths[fd];
}

#define GET_MODE(flags) \
  mode_t mode = 0; \
  if (flags & (O_CREAT | O_TMPFILE)) { va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap); }

int open(const char *path, int flags, ...) {
  REAL(open); GET_MODE(flags);
  int fd = real_open(path, flags, mode);
  opened(fd, AT_FDCWD, path);
  return fd;
}

int open64(const char *path, int flags, ...) {
  REAL(open64); GET_MODE(flags);
  int fd = real_open64(path, flags, mode);
  opened(fd, AT_FDCWD, path);
  return fd;
}

int openat(int dirfd, const char *path, int flags, ...) {
  REAL(openat); GET_MODE(flags);
  int fd = real_openat(dirfd, path, flags, mode);
  opened(fd, dirfd, path);
  return fd;
}

int openat64(int dirfd, const char *path, int flags, ...) {
  REAL(openat64); GET_MODE(flags);
  int fd = real_openat64(dirfd, path, flags, mode);
  opened(fd, dirfd, path);
  return fd;
}

FILE *fopen(const char *path, const char *m) {
  REAL(fopen);
  FILE *f = real_fopen(path, m);
  opened(f ? fileno(f) : -1, AT_FDCWD, path);
  return f;
}

FILE *fopen64(const char *path, const char *m) {
  REAL(fopen64);
  FILE *f = real_fopen64(path, m);
  opened(f ? fileno(f) : -1, AT_FDCWD, path);
  return f;
}

int close(int fd) {
  REAL(close);
  if (fd >= 0 && fd < MAX_FDS) {
    free(fd_paths[fd]);
    fd_paths[fd] = NULL;
  }
  return real_close(fd);
}

ssize_t read(int fd, void *buf, size_t n) {
  REAL(read);
  if (fd_on_disk(fd)) slow("read", fd_paths[fd]);
  return real_read(fd, buf, n);
}

ssize_t pread64(int fd, void *buf, size_t n, off64_t off) {
  REAL(pread64);
  if (fd_on_disk(fd)) slow("read", fd_paths[fd]);
  return real_pread64(fd, buf, n, off);
}

ssize_t write(int fd, const void *buf, size_t n) {
  REAL(write);
  if (fd_on_disk(fd)) slow("write", fd_paths[fd]);
  return real_write(fd, buf, n);
}

ssize_t pwrite64(int fd, const void *buf, size_t n, off64_t off) {
  REAL(pwrite64);
  if (fd_on_disk(fd)) slow("write", fd_paths[fd]);
  return real_pwrite64(fd, buf, n, off);
}

int fsync(int fd) {
  REAL(fsync);
  if (fd_on_disk(fd)) slow("fsync", fd_paths[fd]);
  return real_fsync(fd);
}

int fdatasync(int fd) {
  REAL(fdatasync);
  if (fd_on_disk(fd)) slow("fsync", fd_paths[fd]);
  return real_fdatasync(fd);
}

#define STAT_OP(fn, st_t) \
  int fn(const char *path, struct st_t *st) { \
    REAL(fn); \
    if (on_disk(AT_FDCWD, path)) slow("stat", path); \
    return real_##fn(path, st); \
  }
STAT_OP(stat, stat)
STAT_OP(lstat, stat)
STAT_OP(stat64, stat64)
STAT_OP(lstat64, stat64)

#define FSTATAT_OP(fn, st_t) \
  int fn(int dirfd, const char *path, struct st_t *st, int flags) { \
    REAL(fn); \
    if (path[0] != '\0' && on_disk(dirfd, path)) slow("stat", path); \
    return real_##fn(dirfd, path, st, flags); \
  }
FSTATAT_OP(fstatat, stat)
FSTATAT_OP(fstatat64, stat64)

int access(const char *path, int mode) {
  REAL(access);
  if (on_disk(AT_FDCWD, path)) slow("stat", path);
  return real_access(path, mode);
}

int unlink(const char *path) {
  REAL(unlink);
  if (on_disk(AT_FDCWD, path)) slow("unlink", path);
  return real_unlink(path);
}

int mkdir(const char *path, mode_t mode) {
  REAL(mkdir);
  if (on_disk(AT_FDCWD, path)) slow("mkdir", path);
  return real_mkdir(path, mode);
}

int rename(const char *from, const char *to) {
  REAL(rename);
  if (on_disk(AT_FDCWD, from)) slow("rename", from);
  return real_rename(from, to);
}
