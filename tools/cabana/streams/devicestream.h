#pragma once

#include "tools/cabana/streams/livestream.h"

class DeviceStream : public LiveStream {
  Q_OBJECT
public:
  DeviceStream(QObject *parent, QString address = {});
  static AbstractOpenStreamWidget *widget(AbstractStream **stream);
  inline QString routeName() const override {
    return QString("Live Streaming From %1").arg(zmq_address.isEmpty() ? "127.0.0.1" : zmq_address);
  }

protected:
  void streamThread() override;
  const QString zmq_address;
};

class OpenDeviceWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenDeviceWidget(AbstractStream **stream);
  bool open() override;
  QString title() override { return tr("&Device"); }

private:
  QLineEdit *ip_address;
  QButtonGroup *group;
};
