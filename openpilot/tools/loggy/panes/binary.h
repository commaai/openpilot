#pragma once

#include "tools/loggy/panes/messages.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace loggy {

struct BinaryBitCell {
  bool valid = false;
  uint8_t value = 0;
  uint32_t flip_count = 0;
};

struct BinaryGrid {
  MessageId id;
  double first_time = 0.0;
  double last_time = 0.0;
  size_t event_count = 0;
  std::vector<uint8_t> latest_data;
  std::vector<std::array<BinaryBitCell, 8>> rows;
  uint32_t max_flip_count = 0;
};

inline std::optional<BinaryGrid> build_binary_grid(const Store &store, const MessageId &id, TimeRange range) {
  const CanEventView view = store.canEvents(id, range);
  if (view.events.empty()) return std::nullopt;

  BinaryGrid grid;
  grid.id = id;
  grid.event_count = view.events.size();
  grid.first_time = view.events.front().mono_time;
  grid.last_time = view.events.back().mono_time;
  grid.latest_data = view.events.back().data;
  grid.rows.resize(grid.latest_data.size());

  for (size_t byte_index = 0; byte_index < grid.latest_data.size(); ++byte_index) {
    const uint8_t byte = grid.latest_data[byte_index];
    for (int bit = 0; bit < 8; ++bit) {
      BinaryBitCell &cell = grid.rows[byte_index][static_cast<size_t>(bit)];
      cell.valid = true;
      cell.value = static_cast<uint8_t>((byte >> (7 - bit)) & 1U);
    }
  }

  if (view.events.size() > 1) {
    std::vector<uint8_t> previous = view.events.front().data;
    for (size_t event_index = 1; event_index < view.events.size(); ++event_index) {
      const std::vector<uint8_t> &current = view.events[event_index].data;
      const size_t byte_count = std::min(previous.size(), current.size());
      for (size_t byte_index = 0; byte_index < byte_count && byte_index < grid.rows.size(); ++byte_index) {
        const uint8_t diff = static_cast<uint8_t>(previous[byte_index] ^ current[byte_index]);
        for (int bit = 0; bit < 8; ++bit) {
          if ((diff & (1U << (7 - bit))) == 0) continue;
          BinaryBitCell &cell = grid.rows[byte_index][static_cast<size_t>(bit)];
          ++cell.flip_count;
          grid.max_flip_count = std::max(grid.max_flip_count, cell.flip_count);
        }
      }
      previous = current;
    }
  }
  return grid;
}

void draw_binary_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
