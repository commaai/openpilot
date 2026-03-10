#pragma once

#include "tools/cabana/streams/livestream.h"

#include <QProcess>

class DeviceStream : public LiveStream {
  Q_OBJECT
public:
  DeviceStream(QObject *parent, QString address = {});
  ~DeviceStream();
  inline std::string routeName() const override {
    return "Live Streaming From " + (zmq_address.isEmpty() ? std::string("127.0.0.1") : zmq_address.toStdString());
  }

protected:
  void start() override;
  void streamThread() override;
  QProcess *bridge_process = nullptr;
  const QString zmq_address;
};

class OpenDeviceWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenDeviceWidget(QWidget *parent = nullptr);
  AbstractStream *open() override;

private:
  QLineEdit *ip_address;
  QButtonGroup *group;
};
