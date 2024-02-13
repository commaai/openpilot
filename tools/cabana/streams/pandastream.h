#pragma once

#include <memory>
#include <vector>

#include <QComboBox>
#include <QFormLayout>

#include "tools/cabana/streams/livestream.h"
#include "selfdrive/boardd/panda.h"

const uint32_t speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U};
const uint32_t data_speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U, 2000U, 5000U};

struct PandaStreamConfig {
  QString serial = "";
  std::vector<BusConfig> bus_config;
};

class PandaStream : public LiveStream {
  Q_OBJECT
public:
  PandaStream(QObject *parent, PandaStreamConfig config_ = {});
  static AbstractOpenStreamWidget *widget(AbstractStream **stream);
  inline QString routeName() const override {
    return QString("Live Streaming From Panda %1").arg(config.serial);
  }

protected:
  void streamThread() override;
  bool connect();

  std::unique_ptr<Panda> panda;
  PandaStreamConfig config = {};
};

class OpenPandaWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenPandaWidget(AbstractStream **stream);
  bool open() override;
  QString title() override { return tr("&Panda"); }

private:
  void refreshSerials();
  void buildConfigForm();

  QComboBox *serial_edit;
  QFormLayout *config_layout;
  PandaStreamConfig config = {};
};
