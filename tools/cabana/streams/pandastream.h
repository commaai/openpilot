#pragma once

#include <memory>
#include <vector>

#include <QComboBox>
#include <QFormLayout>

#include "tools/cabana/streams/livestream.h"
#include "tools/cabana/panda.h"

const uint32_t speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U};
const uint32_t data_speeds[] = {10U, 20U, 50U, 100U, 125U, 250U, 500U, 1000U, 2000U, 5000U};

struct BusConfig {
  int can_speed_kbps = 500;
  int data_speed_kbps = 2000;
  bool can_fd = false;
};

struct PandaStreamConfig {
  QString serial = "";
  std::vector<BusConfig> bus_config;
};

class PandaStream : public LiveStream {
  Q_OBJECT
public:
  PandaStream(QObject *parent, PandaStreamConfig config_ = {});
  ~PandaStream() { stop(); }
  inline QString routeName() const override {
    return QString("Panda: %1").arg(config.serial);
  }

protected:
  bool connect();
  void streamThread() override;

  std::unique_ptr<Panda> panda;
  PandaStreamConfig config = {};
};

class OpenPandaWidget : public AbstractOpenStreamWidget {
  Q_OBJECT

public:
  OpenPandaWidget(QWidget *parent = nullptr);
  AbstractStream *open() override;

private:
  void refreshSerials();
  void buildConfigForm();

  QComboBox *serial_edit;
  QFormLayout *form_layout;
  PandaStreamConfig config = {};
};
