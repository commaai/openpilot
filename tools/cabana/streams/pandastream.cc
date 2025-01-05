#include "tools/cabana/streams/pandastream.h"

#include <QDebug>
#include <QCheckBox>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QThread>
#include <QTimer>

PandaStream::PandaStream(QObject *parent, PandaStreamConfig config_) : config(config_), LiveStream(parent) {
  if (!connect()) {
    throw std::runtime_error("Failed to connect to panda");
  }
}

bool PandaStream::connect() {
  try {
    qDebug() << "Connecting to panda " << config.serial;
    panda.reset(new Panda(config.serial.toStdString()));
    config.bus_config.resize(3);
    qDebug() << "Connected";
  } catch (const std::exception& e) {
    return false;
  }

  panda->set_safety_model(cereal::CarParams::SafetyModel::SILENT);
  for (int bus = 0; bus < config.bus_config.size(); bus++) {
    panda->set_can_speed_kbps(bus, config.bus_config[bus].can_speed_kbps);

    // CAN-FD
    if (panda->hw_type == cereal::PandaState::PandaType::RED_PANDA || panda->hw_type == cereal::PandaState::PandaType::RED_PANDA_V2) {
      if (config.bus_config[bus].can_fd) {
        panda->set_data_speed_kbps(bus, config.bus_config[bus].data_speed_kbps);
      } else {
        // Hack to disable can-fd by setting data speed to a low value
        panda->set_data_speed_kbps(bus, 10);
      }
    }
  }
  return true;
}

void PandaStream::streamThread() {
  std::vector<can_frame> raw_can_data;

  while (!QThread::currentThread()->isInterruptionRequested()) {
    QThread::msleep(1);

    if (!panda->connected()) {
      qDebug() << "Connection to panda lost. Attempting reconnect.";
      if (!connect()){
        QThread::msleep(1000);
        continue;
      }
    }

    raw_can_data.clear();
    if (!panda->can_receive(raw_can_data)) {
      qDebug() << "failed to receive";
      continue;
    }

    MessageBuilder msg;
    auto evt = msg.initEvent();
    auto canData = evt.initCan(raw_can_data.size());
    for (uint i = 0; i<raw_can_data.size(); i++) {
      canData[i].setAddress(raw_can_data[i].address);
      canData[i].setDat(kj::arrayPtr((uint8_t*)raw_can_data[i].dat.data(), raw_can_data[i].dat.size()));
      canData[i].setSrc(raw_can_data[i].src);
    }

    handleEvent(capnp::messageToFlatArray(msg));

    panda->send_heartbeat(false, false);
  }
}

// OpenPandaWidget

OpenPandaWidget::OpenPandaWidget(QWidget *parent) : AbstractOpenStreamWidget(parent) {
  form_layout = new QFormLayout(this);
  if (can && dynamic_cast<PandaStream *>(can) != nullptr) {
    form_layout->addWidget(new QLabel(tr("Already connected to %1.").arg(can->routeName())));
    form_layout->addWidget(new QLabel("Close the current connection via [File menu -> Close Stream] before connecting to another Panda."));
    QTimer::singleShot(0, [this]() { emit enableOpenButton(false); });
    return;
  }

  QHBoxLayout *serial_layout = new QHBoxLayout();
  serial_layout->addWidget(serial_edit = new QComboBox());

  QPushButton *refresh = new QPushButton(tr("Refresh"));
  refresh->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
  serial_layout->addWidget(refresh);
  form_layout->addRow(tr("Serial"), serial_layout);

  QObject::connect(refresh, &QPushButton::clicked, this, &OpenPandaWidget::refreshSerials);
  QObject::connect(serial_edit, &QComboBox::currentTextChanged, this, &OpenPandaWidget::buildConfigForm);

  // Populate serials
  refreshSerials();
  buildConfigForm();
}

void OpenPandaWidget::refreshSerials() {
  serial_edit->clear();
  for (auto serial : Panda::list()) {
    serial_edit->addItem(QString::fromStdString(serial));
  }
}

void OpenPandaWidget::buildConfigForm() {
  for (int i = form_layout->rowCount() - 1; i > 0; --i) {
    form_layout->removeRow(i);
  }

  QString serial = serial_edit->currentText();
  bool has_fd = false;
  bool has_panda = !serial.isEmpty();
  if (has_panda) {
    try {
      Panda panda(serial.toStdString());
      has_fd = (panda.hw_type == cereal::PandaState::PandaType::RED_PANDA) || (panda.hw_type == cereal::PandaState::PandaType::RED_PANDA_V2);
    } catch (const std::exception& e) {
      qDebug() << "failed to open panda" << serial;
      has_panda = false;
    }
  }

  if (has_panda) {
    config.serial = serial;
    config.bus_config.resize(3);
    for (int i = 0; i < config.bus_config.size(); i++) {
      QHBoxLayout *bus_layout = new QHBoxLayout;

      // CAN Speed
      bus_layout->addWidget(new QLabel(tr("CAN Speed (kbps):")));
      QComboBox *can_speed = new QComboBox;
      for (int j = 0; j < std::size(speeds); j++) {
        can_speed->addItem(QString::number(speeds[j]));

        if (data_speeds[j] == config.bus_config[i].can_speed_kbps) {
          can_speed->setCurrentIndex(j);
        }
      }
      QObject::connect(can_speed, qOverload<int>(&QComboBox::currentIndexChanged), [=](int index) {config.bus_config[i].can_speed_kbps = speeds[index];});
      bus_layout->addWidget(can_speed);

      // CAN-FD Speed
      if (has_fd) {
        QCheckBox *enable_fd = new QCheckBox("CAN-FD");
        bus_layout->addWidget(enable_fd);
        bus_layout->addWidget(new QLabel(tr("Data Speed (kbps):")));
        QComboBox *data_speed = new QComboBox;
        for (int j = 0; j < std::size(data_speeds); j++) {
          data_speed->addItem(QString::number(data_speeds[j]));

          if (data_speeds[j] == config.bus_config[i].data_speed_kbps) {
            data_speed->setCurrentIndex(j);
          }
        }

        data_speed->setEnabled(false);
        bus_layout->addWidget(data_speed);

        QObject::connect(data_speed, qOverload<int>(&QComboBox::currentIndexChanged), [=](int index) {config.bus_config[i].data_speed_kbps = data_speeds[index];});
        QObject::connect(enable_fd, &QCheckBox::stateChanged, data_speed, &QComboBox::setEnabled);
        QObject::connect(enable_fd, &QCheckBox::stateChanged, [=](int state) {config.bus_config[i].can_fd = (bool)state;});
      }

      form_layout->addRow(tr("Bus %1:").arg(i), bus_layout);
    }
  } else {
    config.serial = "";
    form_layout->addWidget(new QLabel(tr("No panda found")));
  }
}

AbstractStream *OpenPandaWidget::open() {
  try {
    return new PandaStream(qApp, config);
  } catch (std::exception &e) {
    QMessageBox::warning(nullptr, tr("Warning"), tr("Failed to connect to panda: '%1'").arg(e.what()));
    return nullptr;
  }
}
