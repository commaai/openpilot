#pragma once

#include "tools/cabana/analysis/telemetry.h"

namespace cabana {
struct Equation {
  std::string name, source, globals, function;
  std::vector<std::string> additional;
};
// Matches PlotJuggler's nearest-sample alignment (ties select the later sample).
double nearestValue(const std::vector<Sample> &samples, double time);
// Migrate only the known, ported equations from old saved layouts.
void portLegacyEquation(Equation &equation);
// Each evaluation has fresh Python globals, shared by its samples.
std::vector<Sample> evaluateEquation(const Equation &equation, const Telemetry &data);
}  // namespace cabana
