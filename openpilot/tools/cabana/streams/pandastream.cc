#include "tools/cabana/streams/pandastream.h"

#include <chrono>
#include <cstdio>
#include <thread>

PandaStream::PandaStream(PandaStreamConfig config_) : config(config_) {
  if (!connect()) {
    throw std::runtime_error("Failed to connect to panda");
  }
}

bool PandaStream::connect() {
  try {
    fprintf(stderr, "Connecting to panda %s\n", config.serial.c_str());
    panda.reset(new Panda(config.serial));
    config.bus_config.resize(3);
    fprintf(stderr, "Connected\n");
  } catch (const std::exception& e) {
    return false;
  }

  panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
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

  while (!stop_requested_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    if (!panda->connected()) {
      fprintf(stderr, "Connection to panda lost. Attempting reconnect.\n");
      if (!connect()){
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        continue;
      }
    }

    raw_can_data.clear();
    if (!panda->can_receive(raw_can_data)) {
      fprintf(stderr, "failed to receive\n");
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

    panda->send_heartbeat(false);
  }
}
