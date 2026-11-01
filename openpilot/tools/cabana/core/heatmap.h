#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

// Count adjacent samples, never treating a newly appearing byte as a change.
struct HeatmapCounts {
  HeatmapCounts() = default;
  explicit HeatmapCounts(size_t size) : bits(size), bytes(size) {}

  void add(const uint8_t *data, size_t size) {
    const size_t common = std::min({size, previous.size(), bits.size()});
    for (size_t i = 0; i < common; ++i) {
      const uint8_t diff = data[i] ^ previous[i];
      if (diff) ++bytes[i];
      for (int bit = 0; bit < 8; ++bit) {
        if (diff & (1u << bit)) ++bits[i][7 - bit];
      }
    }
    previous.assign(data, data + size);
  }

  std::vector<std::array<uint32_t, 8>> bits;
  std::vector<uint32_t> bytes;
  std::vector<uint8_t> previous;
};
