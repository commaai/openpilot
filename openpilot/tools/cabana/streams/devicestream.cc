#include "tools/cabana/streams/devicestream.h"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

#include "openpilot/cereal/services.h"

// DeviceStream

DeviceStream::DeviceStream(std::string address) : zmq_address(address) {
}

DeviceStream::~DeviceStream() {
  stop();
  if (bridge_pid > 0) {
    kill(bridge_pid, SIGKILL);
    int status;
    waitpid(bridge_pid, &status, 0);
    bridge_pid = -1;
  }
}

void DeviceStream::start() {
  if (!zmq_address.empty()) {
    std::string bridge_path = std::string(CABANA_REPO_ROOT) + "/openpilot/cereal/messaging/bridge";
    bridge_pid = fork();
    if (bridge_pid == 0) {
      execl(bridge_path.c_str(), bridge_path.c_str(), zmq_address.c_str(), "/\"can/\"", (char *)nullptr);
      _exit(127);
    } else if (bridge_pid < 0) {
      fprintf(stderr, "Failed to start bridge\n");
      bridge_pid = -1;
      return;
    }
  }

  LiveStream::start();
}

void DeviceStream::streamThread() {
  zmq_address.empty() ? unsetenv("ZMQ") : setenv("ZMQ", "1", 1);

  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<SubSocket> sock(SubSocket::create(context.get(), "can", "127.0.0.1", false, true, services.at("can").queue_size));
  assert(sock != NULL);
  // run as fast as messages come in
  while (!stop_requested_) {
    std::unique_ptr<Message> msg(sock->receive(true));
    if (!msg) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    handleEvent(kj::ArrayPtr<capnp::word>((capnp::word*)msg->getData(), msg->getSize() / sizeof(capnp::word)));
  }
}
