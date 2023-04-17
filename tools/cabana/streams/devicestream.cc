#include "tools/cabana/streams/devicestream.h"

DeviceStream::DeviceStream(QObject *parent, QString address) : zmq_address(address), LiveStream(parent) {
}

void DeviceStream::streamThread() {
  if (!zmq_address.isEmpty()) {
    setenv("ZMQ", "1", 1);
  }

  std::unique_ptr<Context> context(Context::create());
  std::string address = zmq_address.isEmpty() ? "127.0.0.1" : zmq_address.toStdString();
  std::unique_ptr<SubSocket> sock(SubSocket::create(context.get(), "can", address));
  assert(sock != NULL);
  sock->setTimeout(50);
  // run as fast as messages come in
  while (!QThread::currentThread()->isInterruptionRequested()) {
    Message *msg = sock->receive(true);
    if (!msg) {
      QThread::msleep(50);
      continue;
    }

    std::lock_guard lk(lock);
    handleEvent(messages.emplace_back(msg).event);
  }
}
