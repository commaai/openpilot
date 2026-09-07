#pragma once

#include "tools/cabana/analysis/telemetry.h"

namespace cabana {
struct Equation {
  std::string name, source, globals, function;
  std::vector<std::string> additional;
};
// Matches PlotJuggler's nearest-sample alignment (ties select the later sample).
double nearestValue(const std::vector<Sample> &samples, double time);
// Each evaluation has an isolated Lua state. No file, process or network libraries are exposed.
std::vector<Sample> evaluateEquation(const Equation &equation, const Telemetry &data);
}  // namespace cabana
