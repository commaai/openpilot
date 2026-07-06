#pragma once

#include "tools/loggy/backend/live.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loggy {

struct PandaLiveFrame {
  uint32_t address = 0;
  uint8_t source = 0;
  std::vector<uint8_t> data;
};

std::vector<std::string> panda_live_serials();

class PandaLiveReader {
public:
  explicit PandaLiveReader(std::string serial = {},
                           std::array<PandaBusConfig, kPandaBusCount> bus_config = {});
  ~PandaLiveReader();

  PandaLiveReader(const PandaLiveReader &) = delete;
  PandaLiveReader &operator=(const PandaLiveReader &) = delete;

  bool connected();
  bool receive(std::vector<PandaLiveFrame> *frames);
  void send_heartbeat(bool engaged);

private:
  // pimpl: keeps libusb/Panda USB headers out of every pane include; allowed only here and video (REVIEW §2.4)
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace loggy
