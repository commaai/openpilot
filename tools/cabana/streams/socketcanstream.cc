#include "tools/cabana/streams/socketcanstream.h"

SocketCanStream::SocketCanStream(QObject *parent, SocketCanStreamConfig config_) : config(config_), LiveStream(parent) {
  if (!QCanBus::instance()->plugins().contains("socketcan")) {
    throw std::runtime_error("SocketCAN plugin not available");
  }

  qDebug() << "Connecting to SocketCAN device" << config.device;
  if (!connect()) {
    throw std::runtime_error("Failed to connect to SocketCAN device");
  }
}

bool SocketCanStream::connect() {
  // Connecting might generate some warnings about missing socketcan/libsocketcan libraries
  // These are expected and can be ignored, we don't need the advanced features of libsocketcan
  QString errorString;
  device.reset(QCanBus::instance()->createDevice("socketcan", config.device, &errorString));

  if (!device) {
    qDebug() << "Failed to create SocketCAN device" << errorString;
    return false;
  }

  if (!device->connectDevice()) {
    qDebug() << "Failed to connect to device";
    return false;
  }

  return true;
}

void SocketCanStream::streamThread() {

  while (!QThread::currentThread()->isInterruptionRequested()) {
    QThread::msleep(1);

    auto frames = device->readAllFrames();
    if (frames.size() == 0) continue;

    MessageBuilder msg;
    auto evt = msg.initEvent();
    auto canData = evt.initCan(frames.size());


    for (uint i = 0; i < frames.size(); i++) {
      if (!frames[i].isValid()) continue;

      canData[i].setAddress(frames[i].frameId());
      canData[i].setSrc(0);

      auto payload = frames[i].payload();
      canData[i].setDat(kj::arrayPtr((uint8_t*)payload.data(), payload.size()));
    }

    auto bytes = msg.toBytes();
    handleEvent((const char*)bytes.begin(), bytes.size());
  }
}
