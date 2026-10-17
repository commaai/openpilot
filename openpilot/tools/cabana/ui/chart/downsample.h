#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <vector>

namespace chart {

// Keep endpoints and both extrema in time order in each pixel column. Only rendering
// uses this envelope; calculations, cursor values and exports retain every sample.
template <class Iterator>
auto pixelEnvelope(Iterator begin, Iterator end, double min, double max, int pixels) {
  using Point = typename std::iterator_traits<Iterator>::value_type;
  std::vector<Point> result;
  if (begin == end || pixels <= 0 || max <= min) return result;
  result.reserve(std::min<size_t>(end - begin, (size_t)pixels * 4));
  auto column = [&](const auto &p) {
    return (int)std::clamp((p.x - min) / (max - min) * pixels, 0.0, double(pixels - 1));
  };
  while (begin != end) {
    auto last = begin, low = begin, high = begin, next = begin + 1;
    const int bucket = column(*begin);
    while (next != end && column(*next) == bucket) {
      if (next->y < low->y) low = next;
      if (next->y > high->y) high = next;
      last = next++;
    }
    std::array<Iterator, 4> selected{begin, low, high, last};
    std::sort(selected.begin(), selected.end());
    auto unique_end = std::unique(selected.begin(), selected.end());
    for (auto it = selected.begin(); it != unique_end; ++it) result.push_back(**it);
    begin = next;
  }
  return result;
}

}  // namespace chart
