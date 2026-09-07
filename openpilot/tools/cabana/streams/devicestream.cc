#include "tools/cabana/streams/devicestream.h"

#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <spawn.h>
#include <string>
#include <thread>
#include <utility>
#include <unistd.h>
#include <sys/wait.h>

#include "openpilot/cereal/services.h"
#include "tools/cabana/utils/util.h"

extern char **environ;

// DeviceStream

DeviceStream::DeviceStream(std::string address) : zmq_address(std::move(address)) {
}

DeviceStream::~DeviceStream() {
  stop();
  stopBridge();
}

void DeviceStream::stopBridge() {
  if (bridge_pid <= 0) return;

  ::kill(bridge_pid, SIGTERM);
  for (int i = 0; i < 30; ++i) {
    int status = 0;
    pid_t r = ::waitpid(bridge_pid, &status, WNOHANG);
    if (r == bridge_pid || (r < 0 && errno == ECHILD)) {
      bridge_pid = -1;
      return;
    }
    usleep(100000);  // 100ms, up to ~3s
  }
  ::kill(bridge_pid, SIGKILL);
  ::waitpid(bridge_pid, nullptr, 0);
  bridge_pid = -1;
}

void DeviceStream::start() {
  if (!zmq_address.empty()) {
    stopBridge();
    const std::string path = (executableDir() / "../../cereal/messaging/bridge").lexically_normal().string();
    const char *can_filter = "/\"can/\"";

    // Self-pipe: write end is CLOEXEC so it closes on successful exec. If exec
    // fails, the child writes errno and the parent aborts stream start.
    int err_pipe[2] = {-1, -1};
    if (::pipe(err_pipe) != 0) {
      error(std::string("Failed to start bridge: ") + strerror(errno));
      return;
    }

    pid_t pid = ::fork();
    if (pid == 0) {
      ::close(err_pipe[0]);
      ::fcntl(err_pipe[1], F_SETFD, FD_CLOEXEC);
      execl(path.c_str(), path.c_str(), zmq_address.c_str(), can_filter, static_cast<char *>(nullptr));
      const int err = errno;
      (void)!::write(err_pipe[1], &err, sizeof(err));
      _exit(127);
    }

    ::close(err_pipe[1]);
    if (pid < 0) {
      ::close(err_pipe[0]);
      error(std::string("Failed to start bridge: ") + strerror(errno));
      return;
    }

    int exec_errno = 0;
    const ssize_t n = ::read(err_pipe[0], &exec_errno, sizeof(exec_errno));
    ::close(err_pipe[0]);
    if (n == static_cast<ssize_t>(sizeof(exec_errno))) {
      // Child failed to exec; reap and surface the error.
      int status = 0;
      ::waitpid(pid, &status, 0);
      error(std::string("Failed to start bridge: ") + strerror(exec_errno));
      return;
    }

    bridge_pid = pid;
  }

  LiveStream::start();
}

bool DeviceStream::prepare(std::string &failure) {
  failure.clear();
  if (remote_ready_) return true;
  if (zmq_address.empty() || zmq_address == "127.0.0.1" || zmq_address == "localhost" || zmq_address == "::1") return true;

  // Keep user input out of the remote shell command. The tmux window belongs
  // to the device session and survives both SSH and Cabana disconnects.
  const char *command = R"SH(
cd /data/openpilot || exit 1
pgrep -x bridge >/dev/null && exit 0
if ! tmux has-session -t '=comma' 2>/dev/null; then
  echo 'The openpilot tmux session (comma) is not running.' >&2
  exit 1
fi
test -x ./openpilot/cereal/messaging/bridge || { echo 'Cereal bridge executable is missing.' >&2; exit 1; }
tmux new-window -d -t comma: -n cabana-bridge 'cd /data/openpilot && unset ZMQ && exec ./openpilot/cereal/messaging/bridge' || exit 1
sleep 1
pgrep -x bridge >/dev/null || { echo 'Device bridge exited; check the cabana-bridge tmux window.' >&2; exit 1; }
)SH";
  const std::string destination = "comma@" + zmq_address;
  const char *args[] = {"ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        "-o", "StrictHostKeyChecking=accept-new", destination.c_str(), command, nullptr};
  FILE *output = tmpfile();
  if (!output) {
    failure = strerror(errno);
  } else {
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, fileno(output), STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, fileno(output), STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, fileno(output));
    pid_t pid = -1;
    int result = posix_spawnp(&pid, "ssh", &actions, nullptr, const_cast<char **>(args), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (result != 0) {
      failure = strerror(result);
    } else {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
      int status = 0;
      while (true) {
        pid_t finished = waitpid(pid, &status, WNOHANG);
        if (finished == pid) {
          if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) failure = "SSH command failed.";
          break;
        }
        if (finished < 0 && errno != EINTR) {
          failure = strerror(errno);
          break;
        }
        if (exit_ || std::chrono::steady_clock::now() >= deadline) {
          kill(pid, SIGKILL);
          while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
          failure = "SSH connection timed out.";
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
    if (!failure.empty()) {
      rewind(output);
      char buffer[4096];
      size_t count = fread(buffer, 1, sizeof(buffer), output);
      if (count) {
        std::string details(buffer, count);
        while (!details.empty() && (details.back() == '\n' || details.back() == '\r')) details.pop_back();
        failure = failure == "SSH command failed." ? details : failure + "\n" + details;
      }
    }
    fclose(output);
  }
  if (exit_) return false;
  if (!failure.empty()) {
    failure = "openpilot must be running. Check the device IP address.\n"
              "Enable SSH and add your SSH keys in openpilot.\n\n"
              "SSH error:\n" + failure;
    return false;
  }
  remote_ready_ = true;
  return true;
}

void DeviceStream::streamThread() {
  std::string failure;
  if (!prepare(failure)) {
    if (!exit_) postToMainThread([this, failure]() { error(failure); });
    exit_ = true;
    stopBridge();
    return;
  }
  zmq_address.empty() ? unsetenv("ZMQ") : setenv("ZMQ", "1", 1);

  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> sock(SubSocket::create(context.get(), "can", "127.0.0.1", false, true, services.at("can").queue_size));
  assert(sock != NULL);
  // run as fast as messages come in
  while (!exit_) {
    std::unique_ptr<Message> msg(sock->receive(true));
    if (!msg) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    handleEvent(kj::ArrayPtr<capnp::word>((capnp::word*)msg->getData(), msg->getSize() / sizeof(capnp::word)));
  }
}
