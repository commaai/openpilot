#pragma once

#include <QComboBox>

#include "tools/cabana/streams/livestream.h"

struct SocketCanStreamConfig {
  std::string device = ""; // TODO: support multiple devices/buses at once
};

class SocketCanStream : public LiveStream {
  Q_OBJECT
public:
  SocketCanStream(QObject *parent, SocketCanStreamConfig config_ = {});
  ~SocketCanStream();
  static bool available();

  inline std::string routeName() const override {
    return "Live Streaming From Socket CAN " + config.device;
  }

protected:
  void streamThread() override;
  bool connect();

  SocketCanStreamConfig config = {};
  int sock_fd = -1;
};

class OpenSocketCanWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenSocketCanWidget(QWidget *parent = nullptr);
  AbstractStream *open() override;

private:
  void refreshDevices();

  QComboBox *device_edit;
  SocketCanStreamConfig config = {};
};
