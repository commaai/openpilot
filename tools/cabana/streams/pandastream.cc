#include "tools/cabana/streams/pandastream.h"

PandaStream::PandaStream(QObject *parent, PandaStreamConfig config_) : config(config_), LiveStream(parent) {
  if (config.serial.isEmpty()) {
    auto serials = Panda::list();
    if (serials.size() == 0) {
      throw std::runtime_error("No panda found");
    }
    config.serial = QString::fromStdString(serials[0]);
  }

  qDebug() << "Connecting to panda with serial" << config.serial;
  if (!connect()) {
    throw std::runtime_error("Failed to connect to panda");
  }
  startStreamThread();
}

bool PandaStream::connect() {
  try {
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
      canData[i].setBusTime(raw_can_data[i].busTime);
      canData[i].setDat(kj::arrayPtr((uint8_t*)raw_can_data[i].dat.data(), raw_can_data[i].dat.size()));
      canData[i].setSrc(raw_can_data[i].src);
    }

    {
      std::lock_guard lk(lock);
      auto bytes = msg.toBytes();
      handleEvent(messages.emplace_back((const char*)bytes.begin(), bytes.size()).event);
    }

    panda->send_heartbeat(false);
  }
}
