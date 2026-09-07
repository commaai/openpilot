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
using TelemetrySnapshot = std::map<std::string, std::shared_ptr<const Samples>>;
using Telemetry = std::map<std::string, std::vector<Sample>>;

// PlotJuggler's cereal path convention, including indexed lists, active unions and root metadata.
void extractTelemetry(cereal::Event::Reader event, Telemetry &out);
void mergeTelemetry(Telemetry &destination, Telemetry source);
// Prepare only changed series without modifying the published data. Callers can swap
// these replacements into the destination and release its old buffers off the UI thread.
void prepareTelemetryMerge(const TelemetrySnapshot &destination, Telemetry &batch);
}  // namespace cabana
