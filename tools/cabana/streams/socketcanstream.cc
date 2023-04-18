#include "tools/cabana/streams/socketcanstream.h"

SocketCanStream::SocketCanStream(QObject *parent, SocketCanStreamConfig config_) : config(config_), LiveStream(parent) {
  qDebug() << "Connecting to SocketCAN device" << config.device;
  if (!connect()) {
    throw std::runtime_error("Failed to connect to SocketCAN device");
  }
}

bool SocketCanStream::connect() {
  return true;
}

void SocketCanStream::streamThread() {

  while (!QThread::currentThread()->isInterruptionRequested()) {
    QThread::msleep(1);

  }
}
