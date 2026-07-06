#include "tools/loggy/backend/panda_live.h"

#include "tools/cabana/panda.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace loggy {
namespace {

constexpr uint16_t kDefaultPandaDataSpeedKbpsDisabled = 10;

bool panda_supports_can_fd(cereal::PandaState::PandaType type) {
  return type == cereal::PandaState::PandaType::RED_PANDA ||
         type == cereal::PandaState::PandaType::RED_PANDA_V2;
}

uint8_t panda_source_id(long source) {
  if (source <= 0) return 0;
  return static_cast<uint8_t>(std::min<long>(source, std::numeric_limits<uint8_t>::max()));
}

uint32_t panda_address_id(long address) {
  if (address <= 0) return 0;
  return static_cast<uint32_t>(std::min<unsigned long>(
    static_cast<unsigned long>(address), std::numeric_limits<uint32_t>::max()));
}

void configure_panda(Panda *panda, const std::array<PandaBusConfig, kPandaBusCount> &bus_config) {
  if (panda == nullptr) return;
  panda->set_safety_model(cereal::CarParams::SafetyModel::NO_OUTPUT);
  const bool can_fd_capable = panda_supports_can_fd(panda->hw_type);
  for (size_t bus = 0; bus < bus_config.size(); ++bus) {
    const PandaBusConfig config = normalize_live_panda_bus_config(bus_config[bus]);
    panda->set_can_speed_kbps(static_cast<uint16_t>(bus), config.can_speed_kbps);
    if (can_fd_capable) {
      panda->set_data_speed_kbps(static_cast<uint16_t>(bus),
                                 config.can_fd ? config.data_speed_kbps : kDefaultPandaDataSpeedKbpsDisabled);
    }
  }
}

}  // namespace

std::vector<std::string> panda_live_serials() {
  return Panda::list();
}

class PandaLiveReader::Impl {
public:
  explicit Impl(std::string serial, std::array<PandaBusConfig, kPandaBusCount> bus_config)
    : serial_(std::move(serial)), bus_config_(bus_config) {
    const std::vector<std::string> serials = Panda::list();
    if (serial_.empty()) {
      if (serials.empty()) throw std::runtime_error("No Panda USB detected");
      serial_ = serials.front();
    } else if (std::find(serials.begin(), serials.end(), serial_) == serials.end()) {
      throw std::runtime_error("Panda USB serial not detected");
    }
    panda_ = std::make_unique<Panda>(serial_);
    configure_panda(panda_.get(), bus_config_);
  }

  bool connected() {
    return panda_ != nullptr && panda_->connected();
  }

  bool receive(std::vector<PandaLiveFrame> *frames) {
    if (frames == nullptr) return false;
    frames->clear();
    if (panda_ == nullptr) return false;
    std::vector<can_frame> raw_frames;
    if (!panda_->can_receive(raw_frames)) return false;
    frames->reserve(raw_frames.size());
    for (const can_frame &raw : raw_frames) {
      frames->push_back({
        .address = panda_address_id(raw.address),
        .source = panda_source_id(raw.src),
        .data = std::vector<uint8_t>(raw.dat.begin(), raw.dat.end()),
      });
    }
    return true;
  }

  void send_heartbeat(bool engaged) {
    if (panda_ != nullptr) panda_->send_heartbeat(engaged);
  }

private:
  std::string serial_;
  std::array<PandaBusConfig, kPandaBusCount> bus_config_{};
  std::unique_ptr<Panda> panda_;
};

PandaLiveReader::PandaLiveReader(std::string serial, std::array<PandaBusConfig, kPandaBusCount> bus_config)
  : impl_(std::make_unique<Impl>(std::move(serial), bus_config)) {}

PandaLiveReader::~PandaLiveReader() = default;

bool PandaLiveReader::connected() {
  return impl_ != nullptr && impl_->connected();
}

bool PandaLiveReader::receive(std::vector<PandaLiveFrame> *frames) {
  return impl_ != nullptr && impl_->receive(frames);
}

void PandaLiveReader::send_heartbeat(bool engaged) {
  if (impl_ != nullptr) impl_->send_heartbeat(engaged);
}

}  // namespace loggy
