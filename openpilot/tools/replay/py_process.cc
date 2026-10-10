#include "tools/replay/py_process.h"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <spawn.h>
#ifdef __APPLE__
#include <crt_externs.h>
#endif
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "tools/replay/util.h"

namespace PyProcess {

std::string runModule(const std::string &module, const std::vector<std::string> &args,
                      std::atomic<bool> *abort, bool trim, const StderrLineCallback &stderr_line_cb) {
  // Build argv for the Python module
  std::vector<const char *> argv;
  argv.push_back("python3");
  argv.push_back("-m");
  argv.push_back(module.c_str());
  for (const auto &a : args) {
    argv.push_back(a.c_str());
  }
  argv.push_back(nullptr);

  auto open_pipe = [](int (&fds)[2]) {
#ifdef __linux__
    return pipe2(fds, O_CLOEXEC);
#else
    if (pipe(fds) != 0) return -1;
    if (fcntl(fds[0], F_SETFD, FD_CLOEXEC) == 0 && fcntl(fds[1], F_SETFD, FD_CLOEXEC) == 0) return 0;
    close(fds[0]); close(fds[1]);
    return -1;
#endif
  };
  int stdout_pipe[2], stderr_pipe[2];
  if (open_pipe(stdout_pipe) != 0) {
    rWarning("py_process: pipe() failed");
    return {};
  }
  if (open_pipe(stderr_pipe) != 0) {
    rWarning("py_process: pipe() failed");
    close(stdout_pipe[0]); close(stdout_pipe[1]);
    return {};
  }

  // Avoid copying the large replay address space and running atfork handlers on
  // every spawn: both can stall rendering even from a worker thread.
  std::vector<std::string> environment;
#ifdef __APPLE__
  char **parent_environment = *_NSGetEnviron();
#else
  char **parent_environment = environ;
#endif
  for (char **entry = parent_environment; *entry; ++entry) {
    if (strncmp(*entry, "OPENPILOT_PREFIX=", 17) != 0) environment.emplace_back(*entry);
  }
  std::vector<char *> envp;
  for (auto &entry : environment) envp.push_back(entry.data());
  envp.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  int error = posix_spawn_file_actions_init(&actions);
  const bool actions_initialized = error == 0;
  if (!error) error = posix_spawnattr_init(&attributes);
  const bool attributes_initialized = error == 0;
  if (!error) error = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  if (!error) error = posix_spawn_file_actions_adddup2(&actions, stdout_pipe[1], STDOUT_FILENO);
  if (!error) error = posix_spawn_file_actions_adddup2(&actions, stderr_pipe[1], STDERR_FILENO);
  for (int fd : {stdout_pipe[0], stdout_pipe[1], stderr_pipe[0], stderr_pipe[1]}) {
    if (!error) error = posix_spawn_file_actions_addclose(&actions, fd);
  }
#ifdef POSIX_SPAWN_SETSID
  if (!error) error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSID);
#else
  if (!error) error = posix_spawnattr_setpgroup(&attributes, 0);
  if (!error) error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
#endif
  pid_t pid = -1;
  if (!error) error = posix_spawnp(&pid, "python3", &actions, &attributes, const_cast<char *const *>(argv.data()), envp.data());
  if (attributes_initialized) posix_spawnattr_destroy(&attributes);
  if (actions_initialized) posix_spawn_file_actions_destroy(&actions);
  if (error) {
    rWarning("py_process: posix_spawnp() failed: %s", strerror(error));
    close(stdout_pipe[0]); close(stdout_pipe[1]);
    close(stderr_pipe[0]); close(stderr_pipe[1]);
    return {};
  }

  // Parent process
  close(stdout_pipe[1]);
  close(stderr_pipe[1]);

  // stderr is read in a thread so it can be inspected while the loop below waits on stdout
  std::thread stderr_thread([fd = stderr_pipe[0], cb = stderr_line_cb]() {
    FILE *f = fdopen(fd, "r");
    if (!f) {
      close(fd);
      return;
    }
    char *line = nullptr;
    size_t cap = 0;
    while (getline(&line, &cap, f) > 0) {
      if (cb) {
        cb(line);
      } else {
        fputs(line, stderr);
      }
    }
    free(line);
    fclose(f);
  });

  std::string stdout_data;
  char buf[4096];

  // Use select() so abort can interrupt while waiting for Python output.
  fd_set rfds;
  bool stdout_open = true;

  while (stdout_open) {
    if (abort && *abort) {
      kill(pid, SIGTERM);
      break;
    }

    FD_ZERO(&rfds);
    FD_SET(stdout_pipe[0], &rfds);

    struct timeval tv = {0, 100000};  // 100ms timeout
    int ret = select(stdout_pipe[0] + 1, &rfds, nullptr, nullptr, &tv);
    if (ret < 0) break;

    if (FD_ISSET(stdout_pipe[0], &rfds)) {
      ssize_t n = read(stdout_pipe[0], buf, sizeof(buf));
      if (n <= 0) {
        stdout_open = false;
      } else {
        stdout_data.append(buf, n);
      }
    }
  }

  // Drain remaining pipe data to prevent child from blocking on write
  while (true) {
    ssize_t n = read(stdout_pipe[0], buf, sizeof(buf));
    if (n <= 0) break;
    stdout_data.append(buf, n);
  }
  close(stdout_pipe[0]);
  stderr_thread.join();

  int status;
  waitpid(pid, &status, 0);

  const bool aborted = abort && *abort;
  const bool expected_sigterm = aborted && WIFSIGNALED(status) && WTERMSIG(status) == SIGTERM;
  bool failed = aborted ||
                (WIFEXITED(status) && WEXITSTATUS(status) != 0) ||
                WIFSIGNALED(status);
  if (failed) {
    if (expected_sigterm) {
      // Caller signaled abort; expected shutdown path.
    } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
      rWarning("py_process: %s exited with code %d", module.c_str(), WEXITSTATUS(status));
    } else if (WIFSIGNALED(status)) {
      rWarning("py_process: %s killed by signal %d", module.c_str(), WTERMSIG(status));
    }
    return {};
  }

  // Trim trailing newline
  if (trim) {
    while (!stdout_data.empty() && (stdout_data.back() == '\n' || stdout_data.back() == '\r')) {
      stdout_data.pop_back();
    }
  }

  return stdout_data;
}

}  // namespace PyProcess
