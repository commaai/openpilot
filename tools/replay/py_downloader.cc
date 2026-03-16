#include "tools/replay/py_downloader.h"

#include <csignal>
#include <fcntl.h>
#include <mutex>
#include <sys/wait.h>
#include <unistd.h>

#include "tools/replay/util.h"

namespace {

static std::mutex handler_mutex;
static DownloadProgressHandler progress_handler = nullptr;

// Run a Python command and capture stdout. Optionally parse stderr for PROGRESS lines.
// Returns stdout content. If abort is signaled, kills the child process.
std::string runPython(const std::vector<std::string> &args, std::atomic<bool> *abort = nullptr, bool parse_progress = false) {
  // Build argv for execvp
  std::vector<const char *> argv;
  argv.push_back("python3");
  argv.push_back("-m");
  argv.push_back("openpilot.tools.lib.file_downloader");
  for (const auto &a : args) {
    argv.push_back(a.c_str());
  }
  argv.push_back(nullptr);

  int stdout_pipe[2];
  int stderr_pipe[2];
  if (pipe(stdout_pipe) != 0) {
    rWarning("py_downloader: pipe() failed");
    return {};
  }
  if (pipe(stderr_pipe) != 0) {
    rWarning("py_downloader: pipe() failed");
    close(stdout_pipe[0]); close(stdout_pipe[1]);
    return {};
  }

  pid_t pid = fork();
  if (pid < 0) {
    rWarning("py_downloader: fork() failed");
    close(stdout_pipe[0]); close(stdout_pipe[1]);
    close(stderr_pipe[0]); close(stderr_pipe[1]);
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
    close(stderr_pipe[0]);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    execvp("python3", const_cast<char *const *>(argv.data()));
    _exit(127);
  }

  // Parent process
  close(stdout_pipe[1]);
  close(stderr_pipe[1]);

  std::string stdout_data;
  std::string stderr_buf;
  char buf[4096];

  // Use select() to read from both pipes
  fd_set rfds;
  int max_fd = std::max(stdout_pipe[0], stderr_pipe[0]);
  bool stdout_open = true, stderr_open = true;

  while (stdout_open || stderr_open) {
    if (abort && *abort) {
      kill(pid, SIGTERM);
      break;
    }

    FD_ZERO(&rfds);
    if (stdout_open) FD_SET(stdout_pipe[0], &rfds);
    if (stderr_open) FD_SET(stderr_pipe[0], &rfds);

    struct timeval tv = {0, 100000};  // 100ms timeout
    int ret = select(max_fd + 1, &rfds, nullptr, nullptr, &tv);
    if (ret < 0) break;

    if (stdout_open && FD_ISSET(stdout_pipe[0], &rfds)) {
      ssize_t n = read(stdout_pipe[0], buf, sizeof(buf));
      if (n <= 0) {
        stdout_open = false;
      } else {
        stdout_data.append(buf, n);
      }
    }

    if (stderr_open && FD_ISSET(stderr_pipe[0], &rfds)) {
      ssize_t n = read(stderr_pipe[0], buf, sizeof(buf));
      if (n <= 0) {
        stderr_open = false;
      } else {
        stderr_buf.append(buf, n);
        // Parse complete lines from stderr
        size_t pos;
        while ((pos = stderr_buf.find('\n')) != std::string::npos) {
          std::string line = stderr_buf.substr(0, pos);
          stderr_buf.erase(0, pos + 1);

          if (parse_progress && line.rfind("PROGRESS:", 0) == 0) {
            // Parse "PROGRESS:<cur>:<total>"
            auto colon1 = line.find(':', 9);
            if (colon1 != std::string::npos) {
              try {
                uint64_t cur = std::stoull(line.c_str() + 9);
                uint64_t total = std::stoull(line.c_str() + colon1 + 1);
                std::lock_guard<std::mutex> lk(handler_mutex);
                if (progress_handler) {
                  progress_handler(cur, total, true);
                }
              } catch (...) {}
            }
          } else if (line.rfind("ERROR:", 0) == 0) {
            rWarning("py_downloader: %s", line.c_str() + 6);
          }
        }
      }
    }
  }

  // Drain remaining pipe data to prevent child from blocking on write
  for (int fd : {stdout_pipe[0], stderr_pipe[0]}) {
    while (read(fd, buf, sizeof(buf)) > 0) {}
    close(fd);
  }

  int status;
  waitpid(pid, &status, 0);

  bool failed = (abort && *abort) ||
                (WIFEXITED(status) && WEXITSTATUS(status) != 0) ||
                WIFSIGNALED(status);
  if (failed) {
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
      rWarning("py_downloader: process exited with code %d", WEXITSTATUS(status));
    } else if (WIFSIGNALED(status)) {
      rWarning("py_downloader: process killed by signal %d", WTERMSIG(status));
    }
    std::lock_guard<std::mutex> lk(handler_mutex);
    if (progress_handler) {
      progress_handler(0, 0, false);
    }
    return {};
  }

  // Trim trailing newline
  while (!stdout_data.empty() && (stdout_data.back() == '\n' || stdout_data.back() == '\r')) {
    stdout_data.pop_back();
  }

  return stdout_data;
}

}  // namespace

void installDownloadProgressHandler(DownloadProgressHandler handler) {
  std::lock_guard<std::mutex> lk(handler_mutex);
  progress_handler = handler;
}

namespace PyDownloader {

std::string download(const std::string &url, bool use_cache, std::atomic<bool> *abort) {
  std::vector<std::string> args = {"download", url};
  if (!use_cache) {
    args.push_back("--no-cache");
  }
  return runPython(args, abort, true);
}

std::string getRouteFiles(const std::string &route) {
  return runPython({"route-files", route});
}

std::string getDevices() {
  return runPython({"devices"});
}

std::string getDeviceRoutes(const std::string &dongle_id, int64_t start_ms, int64_t end_ms, bool preserved) {
  std::vector<std::string> args = {"device-routes", dongle_id};
  if (preserved) {
    args.push_back("--preserved");
  } else {
    if (start_ms > 0) {
      args.push_back("--start");
      args.push_back(std::to_string(start_ms));
    }
    if (end_ms > 0) {
      args.push_back("--end");
      args.push_back(std::to_string(end_ms));
    }
  }
  return runPython(args);
}

}  // namespace PyDownloader
