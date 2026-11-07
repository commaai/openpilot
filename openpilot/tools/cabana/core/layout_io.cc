#include "tools/cabana/core/analysis.h"

#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <csignal>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

#ifndef CABANA_REPO_ROOT
#define CABANA_REPO_ROOT "."
#endif
extern char **environ;

namespace cabana {
std::string convertLegacyLayout(const std::string &path) {
  const char *venv = std::getenv("VIRTUAL_ENV");
  std::string python = venv ? std::string(venv) + "/bin/python" : "python3";
  std::string helper = std::string(CABANA_REPO_ROOT) + "/openpilot/tools/cabana/core/layout_convert.py";
  std::vector<char *> args{python.data(), helper.data(), const_cast<char *>(path.c_str()), nullptr};
  int output[2];
  if (pipe(output)) throw std::runtime_error("Cannot create layout converter pipe");
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, output[1], STDERR_FILENO);
  posix_spawn_file_actions_addclose(&actions, output[0]);
  posix_spawn_file_actions_addclose(&actions, output[1]);
  pid_t child = -1;
  const int spawn_error = posix_spawnp(&child, python.c_str(), &actions, nullptr, args.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(output[1]);
  if (spawn_error) { close(output[0]); throw std::runtime_error("Cannot start XML layout converter"); }
  std::string text;
  int status = 0;
  try {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
      if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("XML layout conversion timed out");
      pollfd descriptor{output[0], POLLIN | POLLHUP, 0};
      const int ready = poll(&descriptor, 1, 50);
      if (ready < 0) { if (errno == EINTR) continue; throw std::runtime_error("Cannot read XML conversion"); }
      if (!ready) continue;
      char buffer[8192];
      const auto bytes = read(output[0], buffer, sizeof(buffer));
      if (bytes == 0) break;
      if (bytes < 0) { if (errno == EINTR) continue; throw std::runtime_error("Cannot read XML conversion"); }
      if (text.size() + bytes > 32ULL * 1024 * 1024) throw std::runtime_error("Converted XML layout exceeds 32 MiB");
      text.append(buffer, bytes);
    }
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    close(output[0]);
  } catch (...) {
    kill(child, SIGKILL);
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    close(output[0]);
    throw;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status)) throw std::runtime_error("XML conversion failed: " + text.substr(0, 4096));
  return text;
}
}  // namespace cabana
