#include "tools/jotpluggler/stream_bridge.h"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
namespace {
std::filesystem::path msgq_root() {
#ifdef __APPLE__
  return "/tmp";
#else
  return "/dev/shm";
#endif
}

pid_t wait_nointr(pid_t pid, int *status, int options) {
  pid_t result;
  do { result = waitpid(pid, status, options); } while (result < 0 && errno == EINTR);
  return result;
}

std::string exit_error(const std::string &executable, int status) {
  if (WIFEXITED(status)) return executable + " exited with status " + std::to_string(WEXITSTATUS(status));
  if (WIFSIGNALED(status)) return executable + " terminated by signal " + std::to_string(WTERMSIG(status));
  return executable + " exited unexpectedly";
}
}  // namespace

std::string stream_bridge_whitelist(const std::vector<std::string> &service_names) {
  std::string whitelist;
  for (const std::string &name : service_names) whitelist += "/\"" + name + "\"/";
  return whitelist;
}
ScopedMsgqPrefix::ScopedMsgqPrefix() {
  std::string path_template = (msgq_root() / ("msgq_jotp_" + std::to_string(getpid()) + "_XXXXXX")).string();
  char *created = mkdtemp(path_template.data());
  if (created == nullptr) throw std::runtime_error("Failed to create a private MSGQ namespace: " + std::string(std::strerror(errno)));
  path_ = created;
  prefix_ = path_.filename().string().substr(std::strlen("msgq_"));
}
ScopedMsgqPrefix::~ScopedMsgqPrefix() {
  restore();
  std::error_code error;
  std::filesystem::remove_all(path_, error);
}
void ScopedMsgqPrefix::activate() {
  if (active_) throw std::runtime_error("MSGQ namespace is already active");
  previous_prefix_.reset();
  if (const char *prefix = std::getenv("OPENPILOT_PREFIX")) previous_prefix_ = prefix;
  if (setenv("OPENPILOT_PREFIX", prefix_.c_str(), 1) != 0) {
    throw std::runtime_error("Failed to activate MSGQ namespace: " + std::string(std::strerror(errno)));
  }
  active_ = true;
}
void ScopedMsgqPrefix::restore() noexcept {
  if (!active_) return;
  if (previous_prefix_) setenv("OPENPILOT_PREFIX", previous_prefix_->c_str(), 1);
  else unsetenv("OPENPILOT_PREFIX");
  active_ = false;
}
StreamBridgeProcess::StreamBridgeProcess(std::chrono::milliseconds terminate_grace) : terminate_grace_(terminate_grace) {}
StreamBridgeProcess::~StreamBridgeProcess() { stop(); }
void StreamBridgeProcess::start(const std::filesystem::path &executable, const std::vector<std::string> &arguments) {
  if (owned()) throw std::runtime_error("A stream bridge process is already running");
  if (executable.empty()) throw std::runtime_error("Stream bridge executable path is empty");
  executable_ = executable.string();
  std::vector<std::string> storage{executable_};
  storage.insert(storage.end(), arguments.begin(), arguments.end());
  std::vector<char *> argv;
  for (std::string &argument : storage) argv.push_back(argument.data());
  argv.push_back(nullptr);
  const int result = posix_spawn(&pid_, executable_.c_str(), nullptr, nullptr, argv.data(), environ);
  if (result != 0) {
    pid_ = -1;
    throw std::runtime_error("Failed to start stream bridge " + executable_ + ": " + std::strerror(result));
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  do {
    check_running();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  } while (std::chrono::steady_clock::now() < deadline);
  check_running();
}
void StreamBridgeProcess::check_running() {
  if (!owned()) throw std::runtime_error("Stream bridge process is not running");
  int status = 0;
  const pid_t result = wait_nointr(pid_, &status, WNOHANG);
  if (result == 0) return;
  if (result == pid_) {
    pid_ = -1;
    throw std::runtime_error(exit_error(executable_, status));
  }
  if (result < 0 && errno == ECHILD) {
    pid_ = -1;
    throw std::runtime_error(executable_ + " is no longer waitable");
  }
  if (result < 0) throw std::runtime_error("Failed to inspect stream bridge " + executable_ + ": " + std::strerror(errno));
}
void StreamBridgeProcess::stop() noexcept {
  if (!owned()) return;
  const pid_t child = pid_;
  int status = 0;
  const pid_t initial = wait_nointr(child, &status, WNOHANG);
  if (initial == child || (initial < 0 && errno == ECHILD)) {
    pid_ = -1; return;
  }
  kill(child, SIGTERM);
  const auto deadline = std::chrono::steady_clock::now() + terminate_grace_;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t result = wait_nointr(child, &status, WNOHANG);
    if (result == child || (result < 0 && errno == ECHILD)) {
      pid_ = -1; return;
    }
    if (result < 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  kill(child, SIGKILL);
  wait_nointr(child, &status, 0);
  pid_ = -1;
}
