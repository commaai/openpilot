#pragma once

#include "tools/cabana/streams/livestream.h"

struct SocketCanStreamConfig {
  QString device = "";
  std::vector<BusConfig> bus_config;
};

class SocketCanStream : public LiveStream {
  Q_OBJECT
public:
  SocketCanStream(QObject *parent, SocketCanStreamConfig config_ = {});

  inline QString routeName() const override {
    return QString("Live Streaming From Socket CAN %1").arg(config.device);
  }

protected:
  void streamThread() override;
  bool connect();

  SocketCanStreamConfig config = {};
};
