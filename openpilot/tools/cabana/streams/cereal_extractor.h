#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "openpilot/cereal/gen/cpp/log.capnp.h"

namespace cabana {

struct CerealSample {
  // Keep the original integer clock until plotting, avoiding precision loss and
  // dependence on when replay discovers the route's start time.
  uint64_t mono_time;
  double value;
};

struct CerealSeries {
  std::vector<CerealSample> samples;
  std::map<uint16_t, std::string> enum_names;
};

using CerealSeriesMap = std::map<std::string, CerealSeries>;

// Call on the decoding worker. Paths match JotPluggler layouts, including
// numeric list indices, e.g. /carControl/orientationNED/0. CAN is decoded by
// Cabana's existing DBC pipeline; text and opaque data are not numeric series.
void extractCerealEvent(cereal::Event::Reader event, CerealSeriesMap &series);
void sortCerealSeries(CerealSeriesMap &series);

// Publish complete, immutable segments. A renderer can retain a snapshot while
// the replay worker replaces or evicts segments, without holding a lock while
// plotting and without copying every sample on each frame.
class CerealSeriesStore {
public:
  using Segment = std::shared_ptr<const CerealSeriesMap>;
  using Segments = std::map<int, Segment>;
  struct Snapshot {
    uint64_t revision;
    Segments segments;
  };

  void replaceSegment(int number, CerealSeriesMap series);
  void retainSegments(const std::vector<int> &numbers);
  Snapshot snapshot() const;
  void clear();

private:
  mutable std::mutex mutex_;
  Segments segments_;
  uint64_t revision_ = 0;
};

}  // namespace cabana
