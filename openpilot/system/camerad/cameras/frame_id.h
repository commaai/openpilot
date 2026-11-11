#pragma once

#include <cstdint>
#include <optional>

class CameraFrameCounter {
public:
  std::optional<uint32_t> update(uint64_t timestamp, uint64_t sync_timestamp) {
    // IFE IDs count received SOF events, not physical sensor periods. Compare
    // successive SOFs so a lost event cannot permanently shift one camera's IDs,
    // and clock drift cannot accumulate against a fixed epoch.
    constexpr uint64_t period = 50'000'000;
    uint64_t previous = last_sof ? last_sof : sync_timestamp;
    if (timestamp <= previous || timestamp - previous < period / 2) return std::nullopt;
    uint64_t periods = (timestamp - previous + period / 2) / period;
    frame_id = last_sof ? frame_id + periods : periods - 1;
    last_sof = timestamp;
    return frame_id;
  }

private:
  uint64_t last_sof = 0;
  uint32_t frame_id = 0;
};
