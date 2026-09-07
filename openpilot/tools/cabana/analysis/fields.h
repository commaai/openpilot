#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <capnp/dynamic.h>
#include "cereal/gen/cpp/log.capnp.h"

namespace cabana {
struct Sample {
  double x = 0, y = 0;
  Sample() = default;
  Sample(double x, double y) : x(x), y(y) {}
};
using Samples = std::vector<Sample>;
using FieldsSnapshot = std::map<std::string, std::shared_ptr<const Samples>>;
using Fields = std::map<std::string, Samples>;

// Reuses paths, schema fields and sample destinations across a batch of events.
// The destination map must outlive the extractor and must not erase entries while it is in use.
class FieldExtractor {
public:
  explicit FieldExtractor(Fields &destination);
  ~FieldExtractor();
  void extract(cereal::Event::Reader event);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Prepare only changed series without modifying the published data. Callers can swap
// these replacements into the destination and release its old buffers off the UI thread.
void prepareFieldsMerge(const FieldsSnapshot &destination, Fields &batch);
}  // namespace cabana
