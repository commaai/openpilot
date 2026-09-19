#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "tools/cabana/core/color.h"
#include "tools/cabana/core/message_id.h"

struct CanData {
  void compute(const MessageId &msg_id, const uint8_t *data, int size, double current_sec,
               double playback_speed, const std::vector<uint8_t> &mask, double frequency = 0);

  double ts = 0.;
  uint32_t count = 0;
  double freq = 0;
  std::vector<uint8_t> dat;
  std::vector<CabanaColor> colors;

  struct ByteLastChange {
    double ts = 0;
    int delta = 0;
    int same_delta_counter = 0;
    bool suppressed = false;
  };
  std::vector<ByteLastChange> last_changes;
  std::vector<std::array<uint32_t, 8>> bit_flip_counts;
  double last_freq_update_ts = 0;
};

struct CanEvent {
  uint8_t src;
  uint32_t address;
  uint64_t mono_time;
  uint8_t size;
  uint8_t dat[];
};

struct CompareCanEvent {
  constexpr bool operator()(const CanEvent *const event, uint64_t ts) const { return event->mono_time < ts; }
  constexpr bool operator()(uint64_t ts, const CanEvent *const event) const { return ts < event->mono_time; }
};

using MessageEventsMap = std::unordered_map<MessageId, std::vector<const CanEvent *>>;
using CanEventIter = std::vector<const CanEvent *>::const_iterator;
