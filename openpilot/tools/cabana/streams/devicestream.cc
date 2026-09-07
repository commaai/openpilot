#include "tools/cabana/streams/devicestream.h"

#include <memory>
#include "cereal/services.h"

DeviceStream::DeviceStream(std::string address) : zmq_address(std::move(address)) {}
DeviceStream::~DeviceStream() { stop(); }
void DeviceStream::start() { LiveStream::start(); }

void DeviceStream::streamThread() {
  zmq_address.empty() ? unsetenv("ZMQ") : setenv("ZMQ", "1", 1);
  std::unique_ptr<Context> context(Context::create());
  std::unique_ptr<Poller> poller(Poller::create());
  std::vector<std::unique_ptr<SubSocket>> sockets;
  for (const auto &[name, service] : services) {
    auto socket = std::unique_ptr<SubSocket>(SubSocket::create(context.get(), name,
      zmq_address.empty() ? "127.0.0.1" : zmq_address, false, true, service.queue_size));
    if (!socket) continue;
    poller->registerSocket(socket.get());
    sockets.push_back(std::move(socket));
  }
  while (!exit_) {
    for (auto *socket : poller->poll(50)) {
      std::unique_ptr<Message> msg(socket->receive(true));
      if (msg) handleEvent(kj::ArrayPtr<capnp::word>((capnp::word*)msg->getData(), msg->getSize() / sizeof(capnp::word)));
    }
  }
}
