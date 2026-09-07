#pragma once

#include <map>
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
using Telemetry = std::map<std::string, std::vector<Sample>>;

// PlotJuggler's cereal path convention, including indexed lists, active unions and root metadata.
void extractTelemetry(cereal::Event::Reader event, Telemetry &out);
void mergeTelemetry(Telemetry &destination, Telemetry source);
}  // namespace cabana
