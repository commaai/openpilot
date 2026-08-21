#include "tools/replay/py_process.h"

#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "tools/replay/util.h"

namespace PyProcess {

std::string runModule(const std::string &module, const std::vector<std::string> &args,
                      std::atomic<bool> *abort, bool trim) {
  // Build argv for execvp
  std::vector<const char *> argv;
  argv.push_back("python3");
  argv.push_back("-m");
  argv.push_back(module.c_str());
  for (const auto &a : args) {
    argv.push_back(a.c_str());
  }
  argv.push_back(nullptr);

  int stdout_pipe[2];
  if (pipe(stdout_pipe) != 0) {
    rWarning("py_process: pipe() failed");
    return {};
  }

  pid_t pid = fork();
  if (pid < 0) {
    rWarning("py_process: fork() failed");
    close(stdout_pipe[0]); close(stdout_pipe[1]);
    return {};
  }

  if (pid == 0) {
    // Child process — detach from controlling terminal so Python
    // cannot corrupt terminal settings needed by ncurses in the parent.
    setsid();
    int devnull = open("/dev/null", O_RDONLY);
    if (devnull >= 0) {
      dup2(devnull, STDIN_FILENO);
      if (devnull > STDERR_FILENO) close(devnull);
    }

    // Clear OPENPILOT_PREFIX so the Python process uses default paths
    // (e.g. ~/.comma/auth.json). The prefix is only for IPC in the parent.
    unsetenv("OPENPILOT_PREFIX");

    close(stdout_pipe[0]);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    close(stdout_pipe[1]);

    execvp("python3", const_cast<char *const *>(argv.data()));
    _exit(127);
  }

  // Parent process
  close(stdout_pipe[1]);

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
